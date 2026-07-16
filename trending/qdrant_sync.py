"""Synchronize fresh GitHub Trending signals onto existing Qdrant points."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any

from ingestion.features import activity_score


@dataclass(slots=True)
class TrendingSyncResult:
    """Exact outcome of a trending payload synchronization pass."""

    updated: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class TrendingQdrantSynchronizer:
    """Patch ranking signals without recomputing or replacing vectors."""

    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            from embedding.qdrant_store import QdrantRepositoryStore

            store = QdrantRepositoryStore()
        self.store = store

    def synchronize(
        self,
        repositories: list[dict[str, Any]],
        *,
        refreshed_at: datetime | None = None,
    ) -> TrendingSyncResult:
        """Apply mutable trend payload fields to already indexed repositories."""
        result = TrendingSyncResult()
        if not repositories:
            return result

        refresh_time = refreshed_at or datetime.now(timezone.utc)
        if refresh_time.tzinfo is None:
            refresh_time = refresh_time.replace(tzinfo=timezone.utc)
        refresh_time = refresh_time.astimezone(timezone.utc)

        self.store.validate_collection()
        candidates: list[tuple[str, dict[str, Any], str, int]] = []
        seen: set[str] = set()
        for rank, repository in enumerate(repositories, start=1):
            name = _normalize_repository_name(repository.get("full_name"))
            identity = name.casefold()
            if not name:
                result.failed[str(repository.get("full_name") or "unknown")] = (
                    "invalid repository identity"
                )
                continue
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append((name, repository, self.store._point_id(name), rank))

        if not candidates:
            return result

        points = self.store.client.retrieve(
            collection_name=self.store.collection_name,
            ids=[point_id for _, _, point_id, _ in candidates],
            with_payload=True,
            with_vectors=False,
        )
        existing_payloads = {
            str(point.id): dict(getattr(point, "payload", None) or {})
            for point in points
        }

        for name, repository, point_id, rank in candidates:
            if point_id not in existing_payloads:
                result.missing.append(name)
                continue
            try:
                existing_payload = existing_payloads[point_id]
                daily_stars = _non_negative_int(repository.get("daily_stars"))
                pushed_at = repository.get("pushed_at") or existing_payload.get(
                    "pushed_at"
                )
                pushed_days_ago = _pushed_days_ago(
                    pushed_at,
                    now=refresh_time,
                    fallback=existing_payload.get("pushed_days_ago", 999),
                )
                payload = {
                    "star_count": _non_negative_int(repository.get("star_count")),
                    "fork_count": _non_negative_int(repository.get("fork_count")),
                    "daily_stars": daily_stars,
                    "trend_velocity": round(
                        min(math.log1p(daily_stars) / math.log1p(500.0), 1.0),
                        4,
                    ),
                    "trending_rank": rank,
                    "trending_refreshed_at": refresh_time.isoformat(),
                    "pushed_days_ago": pushed_days_ago,
                    "activity_score": activity_score(
                        {
                            "pushed_days_ago": pushed_days_ago,
                            "mentionable_users_count": existing_payload.get(
                                "mentionable_users_count", 0
                            ),
                        }
                    ),
                }
                if pushed_at:
                    payload["pushed_at"] = pushed_at
                self.store.client.set_payload(
                    collection_name=self.store.collection_name,
                    payload=payload,
                    points=[point_id],
                )
                result.updated.append(name)
            except Exception as exc:
                result.failed[name] = str(exc)[:500]
        return result


def _normalize_repository_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().strip("/")
    if cleaned.lower().startswith("https://github.com/"):
        cleaned = cleaned[len("https://github.com/") :].strip("/")
    owner, separator, name = cleaned.partition("/")
    if not separator or not owner.strip() or not name.strip() or "/" in name:
        return ""
    return f"{owner.strip()}/{name.strip()}"


def _non_negative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _pushed_days_ago(value: Any, *, now: datetime, fallback: Any) -> int:
    """Calculate a non-negative push age while tolerating legacy payloads."""
    try:
        if isinstance(value, datetime):
            pushed_at = value
        elif isinstance(value, str) and value.strip():
            pushed_at = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        else:
            raise ValueError("missing pushed_at")
        if pushed_at.tzinfo is None:
            pushed_at = pushed_at.replace(tzinfo=timezone.utc)
        return max(int((now - pushed_at.astimezone(timezone.utc)).days), 0)
    except (TypeError, ValueError, OverflowError):
        return _non_negative_int(fallback)
