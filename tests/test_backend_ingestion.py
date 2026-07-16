"""Backend v2 ingestion boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
import uuid

from acquisition.backend_client import BackendIngestionClient, repository_upsert_record
from trending.backend_storage import BackendTrendingStorage


class _Response:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(
            {
                "mappings": [
                    {
                        "github_id": "123",
                        "repo_id": str(uuid.uuid4()),
                        "content_version": 1,
                        "changed": True,
                    }
                ]
            }
        )


def _source():
    return SimpleNamespace(
        payload={
            "id": "owner/repo",
            "full_name": "owner/repo",
            "github_id": "123",
            "github_node_id": "R_kg_test",
            "html_url": "https://github.com/owner/repo",
            "description": "demo",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["ml"],
            "star_count": 10,
            "fork_count": 2,
            "open_issues_count": 1,
            "pushed_at": "2026-07-15T12:00:00Z",
            "observed_at": "2026-07-15T12:05:00Z",
        },
        raw_repository={},
        readme=SimpleNamespace(clean_text="README"),
    )


def test_repository_record_keeps_source_and_backend_identity_fields_separate():
    record = repository_upsert_record(_source())

    assert record["github_id"] == "123"
    assert record["github_node_id"] == "R_kg_test"
    assert record["full_name"] == "owner/repo"
    assert "repo_id" not in record
    assert record["url"] == "https://github.com/owner/repo"


def test_backend_mapping_is_validated_and_retained_for_the_run():
    session = _Session()
    client = BackendIngestionClient(
        base_url="http://backend.test",
        internal_secret="secret",
        session=session,
    )

    result = client.upsert_repositories_detailed([_source()])

    assert result.succeeded == ["owner/repo"]
    assert result.mappings["123"]["content_version"] == 1
    assert session.calls[0][0].endswith(
        "/api/internal/v2/ingestion/repositories/upsert"
    )
    assert session.calls[0][1]["headers"]["x-internal-secret"] == "secret"


def test_trending_snapshot_is_enriched_and_published_atomically():
    backend = SimpleNamespace(publish_trending_snapshot=lambda payload: captured.append(payload))
    storage = BackendTrendingStorage.__new__(BackendTrendingStorage)
    storage.backend = backend
    storage.enricher = SimpleNamespace(get_repositories_batch=lambda _repos: [_source()])
    storage._last_refresh = None
    captured = []

    count = storage.upsert_repositories(
        [{"full_name": "owner/repo", "daily_stars": 7}]
    )

    assert count == 1
    assert captured[0]["period"] == "daily"
    assert captured[0]["repositories"][0]["rank"] == 1
    assert captured[0]["repositories"][0]["score"] == 7.0
    assert "repo_id" not in captured[0]["repositories"][0]
