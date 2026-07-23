import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.contracts import RepositoryJob
from api.v2 import (
    FeedbackBatch,
    RecommendationRequest,
    _repository_embedding_payload,
    _repository_job_lock,
    _repository_job_status,
    router,
)
from embedding.qdrant_store import QdrantRepositoryStore
from retrieval.v2_retriever import RecommendationBatch, RankedRepository


def test_canonical_application_exposes_only_v2_api_paths():
    from app import app as canonical_app

    paths = set(canonical_app.openapi()["paths"])
    assert paths
    assert all(path.startswith("/api/v2/") for path in paths)


def test_recommendation_contract_rejects_duplicate_exclusions():
    item = uuid.uuid4()
    with pytest.raises(ValidationError):
        RecommendationRequest(
            schema_version=2,
            generation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            feed_version=1,
            limit=45,
            exclude_repo_ids=[item, item],
            context={"cold_start": False},
        )


def test_feedback_contract_enforces_dwell_and_unique_events():
    base = {
        "event_id": uuid.uuid4(), "user_id": uuid.uuid4(), "repo_id": uuid.uuid4(),
        "feedback_version": 1, "event_type": "dwell", "occurred_at": "2026-07-14T00:00:00Z",
    }
    with pytest.raises(ValidationError):
        FeedbackBatch(schema_version=2, events=[{**base, "dwell_ms": 2_999}])
    valid = {**base, "dwell_ms": 3_000}
    with pytest.raises(ValidationError):
        FeedbackBatch(schema_version=2, events=[valid, valid])


def test_repository_point_id_is_the_canonical_backend_uuid():
    repo_id = str(uuid.uuid4())
    assert QdrantRepositoryStore._point_id(repo_id) == repo_id
    with pytest.raises(ValueError):
        QdrantRepositoryStore._point_id("owner/repository")


def test_repository_job_accepts_canonical_node_v2_outbox_payload():
    repo_id = uuid.uuid4()
    job = RepositoryJob.model_validate(
        {
            "schema_version": 2,
            "job_id": str(uuid.uuid4()),
            "repo_id": str(repo_id),
            "content_version": 1,
            "repository": {
                "repo_id": str(repo_id),
                "github_id": "711550638",
                "github_node_id": "R_kgDOKmlmrg",
                "full_name": "datawhalechina/llm-universe",
                "owner": "datawhalechina",
                "name": "llm-universe",
                "url": "https://github.com/datawhalechina/llm-universe",
                "description": None,
                "readme": "Repository documentation",
                "primary_language": None,
                "languages": ["Jupyter Notebook", "Python"],
                "topics": ["langchain", "rag"],
                "star_count": 13_612,
                "fork_count": 1_383,
                "open_issues_count": 10,
                "pushed_at": "2026-02-24T14:33:21Z",
                "observed_at": "2026-07-22T14:18:33Z",
                "content_hash": "4fea9174cc2f3aca308a150360f01641",
            },
        }
    )

    assert job.repository.repo_id == repo_id
    assert job.repository.html_url == (
        "https://github.com/datawhalechina/llm-universe"
    )
    payload = _repository_embedding_payload(job)
    assert payload["repo_id"] == str(repo_id)
    assert payload["description"] == ""
    assert payload["primary_language"] == "Unknown"
    assert payload["readme_length"] == len("Repository documentation")
    assert payload["extracted_paragraphs"] == ["Repository documentation"]


def test_repository_job_rejects_mismatched_nested_repo_id():
    with pytest.raises(ValidationError, match="repository.repo_id must match repo_id"):
        RepositoryJob.model_validate(
            {
                "schema_version": 2,
                "job_id": str(uuid.uuid4()),
                "repo_id": str(uuid.uuid4()),
                "content_version": 1,
                "repository": {
                    "repo_id": str(uuid.uuid4()),
                    "full_name": "owner/repository",
                },
            }
        )


def test_v2_health_requires_internal_auth(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    response = client.get("/api/v2/health")
    assert response.status_code == 401

    monkeypatch.delenv("INTERNAL_API_SECRET")
    response = client.get("/api/v2/health")
    assert response.status_code == 503


def test_v2_auth_uses_configured_internal_header(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    monkeypatch.setenv("INTERNAL_API_HEADER", "x-ml-service-secret")
    healthy = SimpleNamespace(health=lambda: {"qdrant": "healthy"})
    producer = SimpleNamespace(health=lambda: {
        "redis": "healthy",
        "feedback_consumer_active": True,
    })

    with patch("api.v2.retriever", return_value=healthy), patch(
        "api.v2.producer", return_value=producer
    ):
        assert client.get(
            "/api/v2/health",
            headers={"x-internal-secret": "test-internal-secret"},
        ).status_code == 401
        assert client.get(
            "/api/v2/health",
            headers={"x-ml-service-secret": "test-internal-secret"},
        ).status_code == 200


def test_recommendation_response_reports_the_model_that_served_the_request(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    user_id = uuid.uuid4()
    repo_id = uuid.uuid4()
    batch = RecommendationBatch(
        items=[RankedRepository(str(repo_id), 0.75, "semantic")],
        model_version="heavy-ranker-v2",
        embedding_version="repo-embedding-v2",
        ranker_applied=True,
    )
    fake_retriever = SimpleNamespace(
        recommend_batch=lambda *_args: batch,
        model_version="wrong-static-version",
        embedding_version="wrong-static-version",
    )

    with patch("api.v2.retriever", return_value=fake_retriever):
        response = client.post(
            "/api/v2/recommendations/generate",
            headers={"x-internal-secret": "test-internal-secret"},
            json={
                "schema_version": 2,
                "generation_id": str(uuid.uuid4()),
                "user_id": str(user_id),
                "feed_version": 1,
                "limit": 10,
                "exclude_repo_ids": [],
                "context": {"cold_start": False},
            },
        )

    assert response.status_code == 200
    assert response.json()["model_version"] == "heavy-ranker-v2"
    assert response.json()["embedding_version"] == "repo-embedding-v2"


def test_repository_jobs_are_idempotent_and_monotonic():
    job_id = str(uuid.uuid4())
    points = [
        SimpleNamespace(
            payload={"content_version": 7, "content_job_id": job_id}
        )
    ]
    assert _repository_job_status(
        points,
        version_field="content_version",
        job_field="content_job_id",
        requested_version=7,
        job_id=job_id,
    ) == ("duplicate", 7)
    assert _repository_job_status(
        points,
        version_field="content_version",
        job_field="content_job_id",
        requested_version=7,
        job_id=str(uuid.uuid4()),
    ) == ("current", 7)

    with pytest.raises(HTTPException) as exc_info:
        _repository_job_status(
            points,
            version_field="content_version",
            job_field="content_job_id",
            requested_version=6,
            job_id=str(uuid.uuid4()),
        )
    assert exc_info.value.status_code == 409


def test_repository_job_lock_uses_token_checked_release():
    redis = MagicMock()
    redis.set.return_value = True
    with patch("api.v2.producer", return_value=SimpleNamespace(redis=redis)):
        with _repository_job_lock(str(uuid.uuid4())):
            pass

    redis.set.assert_called_once()
    assert redis.set.call_args.kwargs == {"nx": True, "px": 600_000}
    redis.eval.assert_called_once()
