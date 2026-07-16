"""Qdrant-only multi-channel candidate retrieval.

The online retriever obtains repository vectors and ranking metadata directly
from Qdrant points; it does not perform a second metadata lookup.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

from .config import (
    DISCOVERY_CHANNELS,
    DISCOVERY_LIMIT,
    EMBEDDING_DIM,
    FALLBACK_REPOS,
    OVERFETCH_MULTIPLIER,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_NAME,
    SEMANTIC_LIMIT,
    TOTAL_CANDIDATE_POOL,
)

try:
    from embedding.qdrant_store import QdrantRepositoryStore

    HAS_QDRANT = True
except ImportError:  # pragma: no cover - exercised only without optional dependency
    QdrantRepositoryStore = Any  # type: ignore[misc,assignment]
    HAS_QDRANT = False

logger = logging.getLogger("pipeline.retrieval")


class CandidateRetriever:
    """Retrieve a bounded, ranker-ready candidate pool from Qdrant.

    Candidate contract
    ------------------
    Every non-fallback candidate contains:

    ``repo_id``
        Stable application repository identity from the Qdrant payload.
    ``full_name``
        Stable GitHub ``owner/name`` identity.
    ``repo_embedding``
        The named repository vector as a flat list of floats.
    ``payload``
        The complete Qdrant payload used by retrieval and ranking.
    ``retrieval_source``
        ``semantic``, ``trending``, ``active``, ``popular``, or ``fresh``.
    ``retrieval_score``
        Similarity for semantic results; ordered-channel value otherwise.

    Payload fields are also copied to the candidate's top level for backward
    compatibility with the current ranker. The nested ``payload`` remains the
    canonical metadata object.
    """

    def __init__(
        self,
        db_connector: Any = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        *,
        qdrant_store: Any = None,
    ) -> None:
        # Kept as an ignored compatibility argument until retrieval_engine.py
        # is updated by its owner. No database object is retained or used.
        del db_connector

        self._qdrant_store = qdrant_store
        self._qdrant_available = qdrant_store is not None

        if self._qdrant_store is not None:
            return
        if not HAS_QDRANT:
            logger.error("qdrant-client is unavailable; all retrieval channels are disabled")
            return

        try:
            self._qdrant_store = QdrantRepositoryStore(
                url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY"),
                collection_name=QDRANT_COLLECTION_NAME,
                vector_name=QDRANT_VECTOR_NAME,
                vector_size=EMBEDDING_DIM,
            )
            self._qdrant_store.validate_collection()
            self._qdrant_available = True
        except Exception as exc:
            logger.error("Qdrant repository collection is unavailable: %s", exc)
            self._qdrant_store = None

    def retrieve_candidates(
        self,
        user_embedding: Sequence[float] | None = None,
        user_interests: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``TOTAL_CANDIDATE_POOL`` Qdrant candidates.

        ``user_interests`` is accepted for API compatibility. All information
        needed by online retrieval is read from Qdrant, and the ranker computes
        cross-features from the returned payload.
        """
        del user_interests

        embedding = self._valid_embedding(user_embedding)
        if not self._qdrant_available or self._qdrant_store is None:
            return self._build_fallback_candidates()

        channel_failures = 0
        semantic: list[dict[str, Any]] = []
        if embedding is not None:
            semantic, failed = self._retrieve_semantic_with_status(
                embedding, SEMANTIC_LIMIT
            )
            channel_failures += int(failed)

        # With no profile, or when semantic retrieval fails/returns no usable
        # candidates, discovery receives the whole pool. Otherwise it fills at
        # least its reserved quota and any semantic shortfall.
        if embedding is None or not semantic:
            discovery_target = TOTAL_CANDIDATE_POOL
        else:
            discovery_target = max(
                DISCOVERY_LIMIT,
                TOTAL_CANDIDATE_POOL - len(self._deduplicate(semantic)),
            )

        discovery_lists: list[list[dict[str, Any]]] = []
        per_channel_quota = max(
            1, math.ceil(discovery_target / max(1, len(DISCOVERY_CHANNELS)))
        )
        for source, channel, order_field in DISCOVERY_CHANNELS:
            candidates, failed = self._retrieve_discovery_with_status(
                source=source,
                channel=channel,
                order_field=order_field,
                quota=per_channel_quota,
            )
            discovery_lists.append(candidates)
            channel_failures += int(failed)

        # Keep the overfetch buffer through cross-channel deduplication. This
        # lets later unique points replace repositories repeated by multiple
        # ordered channels or already present in semantic results.
        discovery = self._round_robin(
            discovery_lists,
            min(TOTAL_CANDIDATE_POOL, self._overfetch_limit(discovery_target)),
        )
        merged = self._merge_and_deduplicate(
            semantic,
            discovery,
            pool_limit=TOTAL_CANDIDATE_POOL,
        )

        attempted_channels = len(DISCOVERY_CHANNELS) + int(embedding is not None)
        if not merged and channel_failures == attempted_channels:
            logger.error("Every attempted Qdrant retrieval channel failed; using static fallback")
            return self._build_fallback_candidates()

        logger.info(
            "Qdrant retrieval returned %d candidates: semantic=%d discovery=%d failures=%d",
            len(merged),
            len(semantic),
            len(discovery),
            channel_failures,
        )
        return merged

    def _valid_embedding(
        self, user_embedding: Sequence[float] | None
    ) -> list[float] | None:
        if user_embedding is None:
            return None
        if isinstance(user_embedding, (str, bytes)) or len(user_embedding) != EMBEDDING_DIM:
            logger.warning(
                "Ignoring invalid user embedding; expected %d values", EMBEDDING_DIM
            )
            return None
        try:
            vector = [float(value) for value in user_embedding]
        except (TypeError, ValueError):
            logger.warning("Ignoring user embedding containing non-numeric values")
            return None
        if not all(math.isfinite(value) for value in vector):
            logger.warning("Ignoring user embedding containing non-finite values")
            return None
        return vector

    def _retrieve_semantic(
        self, user_embedding: Sequence[float] | None, quota: int
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper returning approximate semantic candidates."""
        embedding = self._valid_embedding(user_embedding)
        if embedding is None:
            return []
        candidates, _ = self._retrieve_semantic_with_status(embedding, quota)
        return candidates

    def _retrieve_semantic_with_status(
        self, user_embedding: list[float], quota: int
    ) -> tuple[list[dict[str, Any]], bool]:
        if quota <= 0 or self._qdrant_store is None:
            return [], self._qdrant_store is None

        fetch_limit = self._overfetch_limit(quota)
        try:
            points = self._qdrant_store.semantic_search(
                user_embedding,
                limit=fetch_limit,
                with_vectors=True,
            )
            candidates = [
                candidate
                for point in points
                if (
                    candidate := self._candidate_from_point(
                        point,
                        source="semantic",
                        score=self._get_value(point, "score", 0.0),
                    )
                )
                is not None
            ]
            return self._deduplicate(candidates)[:quota], False
        except Exception as exc:
            logger.error("Semantic Qdrant retrieval failed: %s", exc)
            return [], True

    def _retrieve_discovery(
        self, *, source: str, channel: str, order_field: str, quota: int
    ) -> list[dict[str, Any]]:
        """Return one Qdrant payload-ordered discovery channel."""
        candidates, _ = self._retrieve_discovery_with_status(
            source=source,
            channel=channel,
            order_field=order_field,
            quota=quota,
        )
        return candidates

    def _retrieve_discovery_with_status(
        self, *, source: str, channel: str, order_field: str, quota: int
    ) -> tuple[list[dict[str, Any]], bool]:
        if quota <= 0 or self._qdrant_store is None:
            return [], self._qdrant_store is None

        fetch_limit = self._overfetch_limit(quota)
        try:
            points = self._qdrant_store.discover(
                channel,
                limit=fetch_limit,
                with_vectors=True,
            )
            candidates = [
                candidate
                for point in points
                if (
                    candidate := self._candidate_from_point(
                        point,
                        source=source,
                        score=self._ordered_score(
                            self._payload_from_point(point).get(order_field, 0.0)
                        ),
                    )
                )
                is not None
            ]
            return self._deduplicate(candidates), False
        except Exception as exc:
            logger.error("Qdrant %s channel failed: %s", source, exc)
            return [], True

    def _candidate_from_point(
        self, point: Any, *, source: str, score: Any
    ) -> dict[str, Any] | None:
        payload = self._payload_from_point(point)
        repo_id = payload.get("repo_id")
        full_name = payload.get("full_name")
        vector = self._vector_from_point(point)

        if not repo_id or not full_name:
            logger.warning("Skipping Qdrant point without repo_id/full_name")
            return None
        if vector is None or len(vector) != EMBEDDING_DIM:
            logger.warning("Skipping Qdrant point %s with missing/invalid vector", repo_id)
            return None

        try:
            retrieval_score = float(score or 0.0)
        except (TypeError, ValueError):
            retrieval_score = 0.0

        candidate = dict(payload)
        candidate.update(
            {
                "repo_id": str(repo_id),
                "full_name": str(full_name),
                "repo_embedding": vector,
                "payload": dict(payload),
                "retrieval_source": source,
                "retrieval_score": retrieval_score,
            }
        )
        return candidate

    @staticmethod
    def _ordered_score(value: Any) -> float:
        """Convert numeric or ISO date ordering values to a comparable score."""
        if isinstance(value, datetime):
            current = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return current.timestamp()
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        if isinstance(value, str):
            try:
                current = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if current.tzinfo is None:
                    current = current.replace(tzinfo=timezone.utc)
                return current.timestamp()
            except ValueError:
                pass
        try:
            result = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return result if math.isfinite(result) else 0.0

    def _payload_from_point(self, point: Any) -> dict[str, Any]:
        payload = self._get_value(point, "payload", {}) or {}
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _vector_from_point(self, point: Any) -> list[float] | None:
        raw_vector = self._get_value(point, "vector")
        if isinstance(raw_vector, Mapping):
            raw_vector = raw_vector.get(QDRANT_VECTOR_NAME)
        if raw_vector is None or isinstance(raw_vector, (str, bytes)):
            return None
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError):
            return None
        return vector if all(math.isfinite(value) for value in vector) else None

    @staticmethod
    def _get_value(point: Any, key: str, default: Any = None) -> Any:
        if isinstance(point, Mapping):
            return point.get(key, default)
        return getattr(point, key, default)

    @staticmethod
    def _identity(candidate: Mapping[str, Any]) -> str | None:
        repo_id = candidate.get("repo_id")
        if repo_id:
            return f"id:{str(repo_id).strip().lower()}"
        full_name = candidate.get("full_name")
        if full_name:
            return f"name:{str(full_name).strip().lower()}"
        return None

    def _deduplicate(
        self, candidates: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen_repo_ids: set[str] = set()
        seen_full_names: set[str] = set()
        for candidate in candidates:
            repo_id = str(candidate.get("repo_id") or "").strip().lower()
            full_name = str(candidate.get("full_name") or "").strip().lower()
            if not repo_id and not full_name:
                continue
            if (repo_id and repo_id in seen_repo_ids) or (
                full_name and full_name in seen_full_names
            ):
                continue
            if repo_id:
                seen_repo_ids.add(repo_id)
            if full_name:
                seen_full_names.add(full_name)
            unique.append(candidate)
        return unique

    def _merge_and_deduplicate(
        self,
        semantic: Iterable[dict[str, Any]],
        discovery: Iterable[dict[str, Any]],
        semantic_limit: int | None = None,
        pool_limit: int = TOTAL_CANDIDATE_POOL,
    ) -> list[dict[str, Any]]:
        """Merge semantic priority with round-robin discovery candidates.

        ``semantic_limit`` remains accepted for compatibility with older unit
        tests/callers and is otherwise unnecessary.
        """
        del semantic_limit
        return self._deduplicate([*semantic, *discovery])[:pool_limit]

    def _round_robin(
        self,
        channels: Sequence[Sequence[dict[str, Any]]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        positions = [0] * len(channels)
        merged: list[dict[str, Any]] = []
        seen_repo_ids: set[str] = set()
        seen_full_names: set[str] = set()
        while len(merged) < limit:
            progressed = False
            for index, channel in enumerate(channels):
                if positions[index] >= len(channel):
                    continue
                candidate = channel[positions[index]]
                positions[index] += 1
                progressed = True

                repo_id = str(candidate.get("repo_id") or "").strip().lower()
                full_name = str(candidate.get("full_name") or "").strip().lower()
                if not repo_id and not full_name:
                    continue
                if (repo_id and repo_id in seen_repo_ids) or (
                    full_name and full_name in seen_full_names
                ):
                    continue
                if repo_id:
                    seen_repo_ids.add(repo_id)
                if full_name:
                    seen_full_names.add(full_name)
                merged.append(candidate)
                if len(merged) >= limit:
                    break
            if not progressed:
                break
        return merged

    @staticmethod
    def _overfetch_limit(quota: int) -> int:
        return max(quota, int(math.ceil(quota * OVERFETCH_MULTIPLIER)))

    def _build_fallback_candidates(self) -> list[dict[str, Any]]:
        """Return a bounded static list only when Qdrant completely fails."""
        candidates = []
        for full_name in FALLBACK_REPOS[:TOTAL_CANDIDATE_POOL]:
            payload = {
                "repo_id": full_name,
                "full_name": full_name,
                "description": "",
                "languages": [],
                "topics": [],
                "tags": [],
            }
            candidates.append(
                {
                    **payload,
                    "repo_embedding": [0.0] * EMBEDDING_DIM,
                    "payload": payload,
                    "retrieval_source": "fallback",
                    "retrieval_score": 0.0,
                }
            )
        return candidates
