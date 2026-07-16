"""Contract tests for the offline trending-to-Qdrant handoff."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import pytest

from trending_service import parse_args
from trending.qdrant_sync import TrendingQdrantSynchronizer


class _FailureIsolatingClient:
    def __init__(self, point_ids: set[str], failing_point_id: str) -> None:
        self.point_ids = point_ids
        self.failing_point_id = failing_point_id
        self.updates: list[dict] = []

    def retrieve(self, *, ids, **_kwargs):
        return [
            SimpleNamespace(id=point_id, payload={"mentionable_users_count": 4})
            for point_id in ids
            if point_id in self.point_ids
        ]

    def set_payload(self, **kwargs) -> None:
        point_id = kwargs["points"][0]
        if point_id == self.failing_point_id:
            raise RuntimeError("simulated payload update failure")
        self.updates.append(kwargs)


class _Store:
    collection_name = "repositories"

    def __init__(self) -> None:
        point_ids = {self._point_id("owner/one"), self._point_id("owner/two")}
        self.client = _FailureIsolatingClient(
            point_ids,
            failing_point_id=self._point_id("owner/one"),
        )

    def validate_collection(self) -> None:
        return None

    @staticmethod
    def _point_id(repo_id: str) -> str:
        return f"point:{repo_id}"


def test_qdrant_sync_flag_is_strictly_validated(monkeypatch) -> None:
    import trending.config as config

    monkeypatch.setattr(config, "TRENDING_QDRANT_SYNC_STR", "sometimes")
    errors = config.validate_config()
    assert any("TRENDING_QDRANT_SYNC_ENABLED" in error for error in errors)

    monkeypatch.setattr(config, "TRENDING_QDRANT_SYNC_STR", "true")
    errors = config.validate_config()
    assert not any("TRENDING_QDRANT_SYNC_ENABLED" in error for error in errors)
    assert config.TRENDING_QDRANT_SYNC_ENABLED is True


def test_production_trending_service_rejects_direct_qdrant_flags() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--once", "--sync-qdrant"])
    with pytest.raises(SystemExit):
        parse_args(["--once", "--no-sync-qdrant"])


def test_sync_isolates_one_payload_failure_and_continues() -> None:
    store = _Store()

    result = TrendingQdrantSynchronizer(store=store).synchronize(
        [
            {"full_name": "owner/one", "daily_stars": 9},
            {"full_name": "owner/two", "daily_stars": 4},
        ]
    )

    assert result.updated == ["owner/two"]
    assert result.missing == []
    assert "simulated payload update failure" in result.failed["owner/one"]
    assert store.client.updates[0]["points"] == ["point:owner/two"]
    assert "vector" not in store.client.updates[0]


def test_sync_refreshes_activity_and_push_age_without_touching_vectors() -> None:
    store = _Store()
    refreshed_at = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)

    result = TrendingQdrantSynchronizer(store=store).synchronize(
        [
            {
                "full_name": "owner/two",
                "daily_stars": 4,
                "pushed_at": "2026-07-12T12:00:00Z",
            }
        ],
        refreshed_at=refreshed_at,
    )

    assert result.updated == ["owner/two"]
    update = store.client.updates[0]
    assert update["payload"]["pushed_days_ago"] == 2
    assert 0.0 < update["payload"]["activity_score"] <= 1.0
    assert update["payload"]["pushed_at"] == "2026-07-12T12:00:00Z"
    assert "vector" not in update


def test_sync_reports_repository_without_an_existing_qdrant_point() -> None:
    store = _Store()
    store.client.point_ids.remove(store._point_id("owner/two"))

    result = TrendingQdrantSynchronizer(store=store).synchronize(
        [{"full_name": "owner/two", "daily_stars": 4}]
    )

    assert result.updated == []
    assert result.missing == ["owner/two"]
    assert result.failed == {}
    assert store.client.updates == []
