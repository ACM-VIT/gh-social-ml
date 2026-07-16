"""Backend v2 publisher implementing the scheduler's storage boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from acquisition.backend_client import BackendIngestionClient, repository_upsert_record
from acquisition.github_graphql_client import GitHubGraphQLClient
from acquisition.repository_enricher import RepositoryEnricher


class BackendTrendingStorage:
    """Enrich and atomically publish one complete trending snapshot."""

    enabled = True

    def __init__(
        self,
        *,
        backend: BackendIngestionClient,
        github_token: str,
    ) -> None:
        self.backend = backend
        self.enricher = RepositoryEnricher(GitHubGraphQLClient(token=github_token))
        self._last_refresh: datetime | None = None

    def init_schema(self) -> None:
        """Compatibility no-op: backend migrations own durable schemas."""

    def get_last_refresh_time(self) -> datetime | None:
        return self._last_refresh

    def upsert_repositories(
        self,
        repositories: list[dict[str, Any]],
        refresh_timestamp: datetime | None = None,
    ) -> int:
        computed_at = refresh_timestamp or datetime.now(timezone.utc)
        enriched = self.enricher.get_repositories_batch(repositories)
        if len(enriched) != len(repositories):
            raise RuntimeError(
                f"refusing incomplete trending snapshot: enriched {len(enriched)}/"
                f"{len(repositories)} repositories"
            )
        trending_by_name = {
            str(repository.get("full_name")): repository
            for repository in repositories
        }
        records: list[dict[str, Any]] = []
        for rank, source in enumerate(enriched, start=1):
            record = repository_upsert_record(source)
            # A trending snapshot is one atomic request under the backend's
            # 256 KB limit; repository embedding jobs retain the full content.
            record["readme"] = record["readme"][:4000]
            trend = trending_by_name.get(record["full_name"], {})
            record["rank"] = rank
            if trend.get("daily_stars") is not None:
                record["score"] = float(trend.get("daily_stars") or 0)
            records.append(record)
        self.backend.publish_trending_snapshot(
            {
                "snapshot_id": str(uuid.uuid4()),
                "period": "daily",
                "computed_at": computed_at.astimezone(timezone.utc).isoformat(),
                "source": "github-trending",
                "repositories": records,
            }
        )
        self._last_refresh = computed_at
        return len(records)
