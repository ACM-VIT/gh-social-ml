"""Qdrant-native multi-channel candidate retrieval for online inference."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

from .config import (
    EMBEDDING_DIM,
    FALLBACK_REPOS,
    OVERFETCH_MULTIPLIER,
    QDRANT_COLLECTION_NAME,
    QDRANT_TIMEOUT_SECONDS,
    QDRANT_VECTOR_NAME,
    SEMANTIC_LIMIT,
    TOTAL_CANDIDATE_POOL,
    TRENDING_LIMIT,
)

try:
    from embedding.qdrant_store import QdrantRepositoryStore

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

logger = logging.getLogger("pipeline.retrieval")

# Each discovery signal has a payload index in Qdrant. Round-robin merging keeps
# the fallback pool from collapsing into a single popularity-only list.
DISCOVERY_SIGNALS = (
    "trend_velocity",
    "activity_score",
    "star_count",
    "updated_at",
)


class CandidateRetriever:
    """Retrieve and hydrate candidates exclusively from the ML-owned Qdrant corpus.

    The online service performs approximate semantic retrieval for personalized
    candidates and ordered Qdrant scrolls for discovery/freshness candidates.
    Repository payloads and vectors are returned in the same calls, avoiding
    Postgres reads, N+1 hydration, and on-request embedding work.
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self._qdrant_store: QdrantRepositoryStore | None = None
        if not HAS_QDRANT:
            logger.error("qdrant-client is unavailable; candidate retrieval is disabled.")
            return

        url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        try:
            self._qdrant_store = QdrantRepositoryStore(
                url=url,
                api_key=api_key,
                collection_name=QDRANT_COLLECTION_NAME,
                vector_name=QDRANT_VECTOR_NAME,
                vector_size=EMBEDDING_DIM,
                timeout=QDRANT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.error("Qdrant client initialization failed: %s", exc)

    def retrieve_candidates(
        self,
        user_embedding: list[float] | None = None,
        user_interests: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a merged semantic and discovery pool ready for ranking."""
        if not self._valid_embedding(user_embedding):
            if user_embedding:
                logger.warning(
                    "Ignoring user embedding with invalid dimension; expected %d.",
                    EMBEDDING_DIM,
                )
            user_embedding = None

        semantic_quota = SEMANTIC_LIMIT if user_embedding is not None else 0
        semantic = self._retrieve_semantic(user_embedding, semantic_quota)

        # A failed or unavailable semantic channel transfers its quota to
        # discovery so cold starts and transient profile misses still get a feed.
        discovery_quota = (
            TRENDING_LIMIT if semantic else TOTAL_CANDIDATE_POOL
        )
        discovery = self._retrieve_discovery(discovery_quota)
        candidates = self._merge_and_deduplicate(
            semantic,
            discovery,
            semantic_limit=semantic_quota,
            pool_limit=TOTAL_CANDIDATE_POOL,
        )

        if not candidates:
            logger.error("Qdrant retrieval returned no candidates; using static fallback.")
            return self._build_fallback_candidates()

        logger.info(
            "Qdrant retrieval returned %d candidates (%d semantic, %d discovery).",
            len(candidates),
            len(semantic),
            len(discovery),
        )
        return candidates

    def _retrieve_semantic(
        self,
        user_embedding: list[float] | None,
        quota: int,
    ) -> list[dict[str, Any]]:
        if user_embedding is None or quota <= 0 or self._qdrant_store is None:
            return []

        fetch_limit = min(
            int(math.ceil(quota * OVERFETCH_MULTIPLIER)),
            TOTAL_CANDIDATE_POOL + 100,
        )
        try:
            matches = self._qdrant_store.search(
                vector=user_embedding,
                limit=fetch_limit,
                with_vectors=True,
                exact=False,
            )
            return [
                candidate
                for match in matches
                if (candidate := self._point_to_candidate(match, source="semantic"))
            ]
        except Exception as exc:
            logger.error("Qdrant semantic retrieval failed: %s", exc)
            return []

    def _retrieve_discovery(self, quota: int) -> list[dict[str, Any]]:
        if quota <= 0 or self._qdrant_store is None:
            return []

        fetch_limit = min(
            int(math.ceil(quota * OVERFETCH_MULTIPLIER)),
            TOTAL_CANDIDATE_POOL + 50,
        )
        per_signal = max(1, math.ceil(fetch_limit / len(DISCOVERY_SIGNALS)))
        channels: list[list[dict[str, Any]]] = []

        for signal in DISCOVERY_SIGNALS:
            try:
                points = self._qdrant_store.list_points_ordered(
                    order_by=signal,
                    limit=per_signal,
                    descending=True,
                    with_vectors=True,
                )
                channel = [
                    candidate
                    for point in points
                    if (
                        candidate := self._point_to_candidate(
                            point,
                            source=f"discovery_{signal}",
                        )
                    )
                ]
                if channel:
                    channels.append(channel)
            except Exception as exc:
                logger.warning("Qdrant discovery signal %s failed: %s", signal, exc)

        if not channels:
            # This fallback supports collections created before ordered payload
            # indexes were added. It remains bounded and reads Qdrant only.
            try:
                points = self._qdrant_store.list_points(
                    limit=fetch_limit,
                    with_vectors=True,
                )
                return [
                    candidate
                    for point in points
                    if (candidate := self._point_to_candidate(point, source="discovery"))
                ][:quota]
            except Exception as exc:
                logger.error("Qdrant discovery fallback failed: %s", exc)
                return []

        # Interleave channels before deduplication to preserve diversity.
        interleaved: list[dict[str, Any]] = []
        for index in range(max(len(channel) for channel in channels)):
            for channel in channels:
                if index < len(channel):
                    interleaved.append(channel[index])
        return self._deduplicate(interleaved, quota)

    def _point_to_candidate(
        self,
        point: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any] | None:
        payload = dict(point.get("payload") or {})
        repo_id = payload.get("repo_id") or point.get("repo_id")
        full_name = payload.get("full_name") or point.get("full_name") or repo_id
        vector = point.get("vector")
        if isinstance(vector, dict):
            vector = vector.get(QDRANT_VECTOR_NAME)

        if not repo_id or not full_name or not self._valid_embedding(vector):
            return None

        candidate = payload
        candidate.update(
            {
                "repo_id": str(repo_id),
                "full_name": str(full_name),
                "repo_embedding": list(vector),
                "embedding_source": "qdrant",
                "retrieval_source": source,
                "retrieval_score": point.get("score"),
            }
        )
        candidate.setdefault("forks_count", candidate.get("fork_count", 0))
        return candidate

    def _merge_and_deduplicate(
        self,
        semantic_candidates: list[dict[str, Any]],
        discovery_candidates: list[dict[str, Any]],
        semantic_limit: int,
        pool_limit: int,
    ) -> list[dict[str, Any]]:
        semantic = self._deduplicate(semantic_candidates, semantic_limit)
        return self._deduplicate([*semantic, *discovery_candidates], pool_limit)

    @staticmethod
    def _deduplicate(
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for candidate in candidates:
            identity = str(candidate.get("repo_id") or candidate.get("full_name") or "")
            if not identity or identity in seen:
                continue
            seen.add(identity)
            result.append(candidate)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _valid_embedding(vector: Any) -> bool:
        return isinstance(vector, (list, tuple)) and len(vector) == EMBEDDING_DIM

    @staticmethod
    def _build_fallback_candidates() -> list[dict[str, Any]]:
        return [
            {
                "repo_id": repo_name,
                "full_name": repo_name,
                "retrieval_source": "fallback",
                "retrieval_score": 0.0,
                "repo_embedding": [0.0] * EMBEDDING_DIM,
                "embedding_source": "zero_fallback",
            }
            for repo_name in FALLBACK_REPOS
        ]
