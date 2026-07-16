from __future__ import annotations

import hmac
import math
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.vector_contract import repository_point_ids
from feedback.v2 import DurableFeedbackProducer
from retrieval.v2_retriever import QdrantV2Retriever
from scripts.user_onboarding import UserOnboardingPipeline

router = APIRouter(prefix="/api/v2", tags=["v2"])
EventType = Literal[
    "impression", "dwell", "readme_open", "github_open", "like", "unlike",
    "dislike", "undislike", "save", "unsave", "share",
]


async def require_internal_secret(
    x_internal_secret: str | None = Header(default=None, alias="X-Internal-Secret"),
) -> None:
    expected = os.getenv("INTERNAL_API_SECRET")
    if not expected:
        raise HTTPException(status_code=503, detail="Internal API secret is not configured.")
    if not x_internal_secret or not hmac.compare_digest(x_internal_secret, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing internal secret.")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecommendationContext(StrictModel):
    cold_start: bool = False
    locale: str | None = Field(default=None, max_length=32)


class RecommendationRequest(StrictModel):
    schema_version: Literal[2]
    generation_id: uuid.UUID
    user_id: uuid.UUID
    feed_version: int = Field(ge=1)
    limit: int = Field(ge=1, le=100)
    exclude_repo_ids: list[uuid.UUID] = Field(default_factory=list, max_length=500)
    context: RecommendationContext

    @field_validator("exclude_repo_ids")
    @classmethod
    def unique_exclusions(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(set(value)) != len(value):
            raise ValueError("exclude_repo_ids must be unique")
        return value


class FeedbackEvent(StrictModel):
    event_id: uuid.UUID
    user_id: uuid.UUID
    repo_id: uuid.UUID
    feedback_version: int = Field(ge=1)
    event_type: EventType
    dwell_ms: int | None = None
    occurred_at: str

    @model_validator(mode="after")
    def validate_dwell(self):
        if self.event_type == "impression":
            raise ValueError("impressions are offline-only and must not be sent to ML")
        if self.event_type == "dwell":
            if self.dwell_ms is None or not 3_000 <= self.dwell_ms <= 300_000:
                raise ValueError("dwell_ms must be between 3000 and 300000")
        elif self.dwell_ms is not None:
            raise ValueError("only dwell events may carry dwell_ms")
        return self


class FeedbackBatch(StrictModel):
    schema_version: Literal[2]
    events: list[FeedbackEvent] = Field(min_length=1, max_length=100)

    @field_validator("events")
    @classmethod
    def unique_events(cls, value: list[FeedbackEvent]) -> list[FeedbackEvent]:
        if len({event.event_id for event in value}) != len(value):
            raise ValueError("event_id values must be unique within a batch")
        return value


class RepositoryJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    repo_id: uuid.UUID
    content_version: int = Field(ge=1)
    repository: dict[str, Any]


class RepositoryRefreshJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    repo_id: uuid.UUID
    feature_version: int = Field(ge=1)
    features: dict[str, Any]


class OnboardingJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    user_id: uuid.UUID
    profile_version: int = Field(ge=1)
    profile: dict[str, Any]


@lru_cache(maxsize=1)
def retriever() -> QdrantV2Retriever:
    return QdrantV2Retriever()


@lru_cache(maxsize=1)
def producer() -> DurableFeedbackProducer:
    return DurableFeedbackProducer()


def _repository_points(repo_id: str) -> list[Any]:
    canonical, legacy = repository_point_ids(repo_id)
    return retriever().client.retrieve(
        collection_name=retriever().repository_collection,
        ids=[canonical, legacy],
        with_payload=True,
        with_vectors=False,
    )


def _repository_job_status(
    points: list[Any],
    *,
    version_field: str,
    job_field: str,
    requested_version: int,
    job_id: str,
) -> tuple[str, int]:
    payloads = [dict(point.payload or {}) for point in points]
    if any(str(payload.get(job_field) or "") == job_id for payload in payloads):
        current = max(
            (int(payload.get(version_field) or 0) for payload in payloads), default=0
        )
        return "duplicate", current
    current = max(
        (int(payload.get(version_field) or 0) for payload in payloads), default=0
    )
    if requested_version < current:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{version_field} {requested_version} is older than stored version {current}."
            ),
        )
    if requested_version == current and current > 0:
        return "current", current
    return "apply", current


@contextmanager
def _repository_job_lock(repo_id: str):
    redis = producer().redis
    key = f"ml:repository-job-lock:{repo_id}"
    token = str(uuid.uuid4())
    ttl_ms = int(os.getenv("REPOSITORY_JOB_LOCK_TTL_MS", "600000"))
    wait_seconds = float(os.getenv("REPOSITORY_JOB_LOCK_WAIT_SECONDS", "30"))
    deadline = time.monotonic() + wait_seconds
    while not redis.set(key, token, nx=True, px=ttl_ms):
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Timed out waiting for repository job lock for {repo_id}.",
            )
        time.sleep(0.05)
    try:
        yield
    finally:
        redis.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end",
            1,
            key,
            token,
        )


def _embed_repository_job(request: RepositoryJob) -> dict[str, Any]:
    repo_id = str(request.repo_id)
    job_id = str(request.job_id)
    with _repository_job_lock(repo_id):
        points = _repository_points(repo_id)
        job_status, current = _repository_job_status(
            points,
            version_field="content_version",
            job_field="content_job_id",
            requested_version=request.content_version,
            job_id=job_id,
        )
        if job_status != "apply":
            return {
                "accepted": True,
                "status": job_status,
                "repo_id": repo_id,
                "content_version": current,
            }

        payload = dict(request.repository)
        payload.update({
            "id": repo_id,
            "repo_id": repo_id,
            "content_version": request.content_version,
        })
        pipeline = RepositoryEmbeddingPipeline()
        result = pipeline.embed_repository(payload)
        result.payload["content_job_id"] = job_id

        store = QdrantRepositoryStore(
            vector_size=pipeline.config.embedding_dim,
            client=retriever().client,
        )
        store.ensure_collection()
        store.upsert([result])
        _, legacy_repo_id = repository_point_ids(repo_id)
        if any(str(point.id) == legacy_repo_id for point in points):
            # The canonical upsert succeeded, so removing the old identity can
            # no longer make the repository unavailable during cutover.
            retriever().client.delete(
                collection_name=retriever().repository_collection,
                points_selector=[legacy_repo_id],
                wait=True,
            )
        return {
            "accepted": True,
            "status": "applied",
            "repo_id": repo_id,
            "content_version": request.content_version,
            "embedding_version": result.embedding_version,
        }


def _refresh_repository_job(request: RepositoryRefreshJob) -> dict[str, Any]:
    repo_id = str(request.repo_id)
    job_id = str(request.job_id)
    with _repository_job_lock(repo_id):
        points = _repository_points(repo_id)
        if not points:
            raise HTTPException(
                status_code=404, detail=f"Repository {repo_id} is not indexed."
            )
        job_status, current = _repository_job_status(
            points,
            version_field="feature_version",
            job_field="feature_job_id",
            requested_version=request.feature_version,
            job_id=job_id,
        )
        if job_status != "apply":
            return {
                "accepted": True,
                "status": job_status,
                "repo_id": repo_id,
                "feature_version": current,
            }

        retriever().client.set_payload(
            collection_name=retriever().repository_collection,
            payload={
                **request.features,
                "feature_version": request.feature_version,
                "feature_job_id": job_id,
            },
            points=[point.id for point in points],
            wait=True,
        )
        return {
            "accepted": True,
            "status": "applied",
            "repo_id": repo_id,
            "feature_version": request.feature_version,
        }


@router.post(
    "/recommendations/generate", dependencies=[Depends(require_internal_secret)]
)
async def generate_recommendations(request: RecommendationRequest):
    items = await run_in_threadpool(
        retriever().recommend,
        str(request.user_id),
        request.limit,
        [str(item) for item in request.exclude_repo_ids],
    )
    invalid_scores = any(not math.isfinite(item.score) for item in items)
    if len({item.repo_id for item in items}) != len(items) or invalid_scores:
        raise HTTPException(
            status_code=500, detail="Retriever produced invalid recommendations."
        )
    return {
        "schema_version": 2,
        "generation_id": str(request.generation_id),
        "user_id": str(request.user_id),
        "feed_version": request.feed_version,
        "model_version": retriever().model_version,
        "embedding_version": retriever().embedding_version,
        "items": [asdict(item) for item in items],
    }


@router.post(
    "/feedback/batch",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_internal_secret)],
)
async def submit_feedback(request: FeedbackBatch):
    events = [event.model_dump(mode="json") for event in request.events]
    accepted, duplicates = await run_in_threadpool(producer().enqueue, events)
    return {"accepted": accepted, "duplicates": duplicates, "durable": True}


@router.post("/repositories/embed", dependencies=[Depends(require_internal_secret)])
async def embed_repository(request: RepositoryJob):
    return await run_in_threadpool(_embed_repository_job, request)


@router.post("/repositories/refresh", dependencies=[Depends(require_internal_secret)])
async def refresh_repository(request: RepositoryRefreshJob):
    return await run_in_threadpool(_refresh_repository_job, request)


@router.post("/users/onboard", dependencies=[Depends(require_internal_secret)])
async def onboard_user(request: OnboardingJob):
    pipeline = UserOnboardingPipeline()
    vector = await run_in_threadpool(pipeline.generate_interest_vector, request.profile)
    payload = {
        **request.profile,
        "job_id": str(request.job_id),
        "profile_version": request.profile_version,
    }
    await run_in_threadpool(
        pipeline.save_to_qdrant, str(request.user_id), vector, payload
    )
    return {
        "accepted": True,
        "user_id": str(request.user_id),
        "profile_version": request.profile_version,
    }


@router.get("/health", dependencies=[Depends(require_internal_secret)])
async def health():
    try:
        qdrant = await run_in_threadpool(retriever().health)
        redis = await run_in_threadpool(producer().health)
        return {"status": "healthy", **qdrant, **redis, "database_required": False}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dependency health check failed: {exc}") from exc
