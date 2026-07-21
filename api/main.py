"""Production API for feedback ingestion and ML operations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import importlib
import logging
import math
import os
import re
import time
from typing import Any, Literal
import uuid
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from config import constant_time_secret_matches, internal_api_header_name
from api.metrics import record_api_request
from feedback.consumer import FeedbackConsumer
from feedback.event_handlers import FeedbackHandler
from feedback.interactions import INTERACTIONS, get_interaction, normalize_interaction
from feedback.producer import FeedbackProducer, create_redis_client
from feedback.settings import FeedbackSettings

load_dotenv()

logger = logging.getLogger("pipeline.api")
MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024

producer: FeedbackProducer | None = None
consumer: FeedbackConsumer | None = None
feedback_handler: FeedbackHandler | None = None
retrieval_engine: Any | None = None
onboarding_pipeline: Any | None = None
repo_embedding_pipeline: Any | None = None


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "") or uuid.uuid4())


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    headers: dict[str, str] | None = None,
    details: Any | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    body: dict[str, Any] = {
        # Keep FastAPI's historical detail field for backend compatibility.
        "detail": message,
        "code": code,
        "message": message,
        "retryable": retryable,
        "request_id": request_id,
    }
    if details is not None:
        body["details"] = details
    response_headers = dict(headers or {})
    response_headers["x-request-id"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=response_headers,
    )


def legacy_api_enabled() -> bool:
    default = "false" if os.getenv("APP_ENV", "development").lower() == "production" else "true"
    return os.getenv("LEGACY_ML_API_ENABLED", default).strip().lower() in {
        "1", "true", "yes", "on",
    }


class FeedbackRequest(BaseModel):
    event_id: str = Field(..., min_length=1, max_length=128)
    user_id: UUID
    repo_id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    occurred_at: datetime
    schema_version: Literal[1] = 1
    dwell_seconds: float | None = Field(default=None, ge=0)


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    is_cold_start: bool = False


class OnboardingRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    github_username: str | None = None
    username: str | None = None
    full_name: str | None = None
    bio: str | None = None
    interests: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    avatar_url: str | None = None


class EmbedRepoRequest(BaseModel):
    repo_id: str = Field(..., min_length=1)
    github_repo: str = Field(..., min_length=1)
    github_repo_url: str | None = None
    description: str | None = None
    primary_language: str | None = None
    languages: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    readme_summary: str | None = None
    star_count: int = Field(default=0, ge=0)
    fork_count: int = Field(default=0, ge=0)
    open_issues_count: int = Field(default=0, ge=0)
    created_at: str | None = None
    updated_at: str | None = None


def _build_feedback_runtime(
    settings: FeedbackSettings,
) -> tuple[FeedbackProducer, FeedbackConsumer, FeedbackHandler]:
    redis_client = create_redis_client(settings)
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30.0,
    )
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    stream_producer = FeedbackProducer(redis_client=redis_client, settings=settings)
    stream_consumer = FeedbackConsumer(
        handler=handler, redis_client=redis_client, settings=settings
    )
    return stream_producer, stream_consumer, handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start validated feedback dependencies and exactly one consumer task."""
    global producer, consumer, feedback_handler
    if not legacy_api_enabled():
        from api.v2 import shutdown_v2_runtime, validate_v2_runtime_configuration
        from embedding.runtime import embedding_warmup_enabled, warm_embedding_runtime
        from scripts.validate_production_config import validate_production_config

        if os.getenv("APP_ENV", "development").strip().casefold() == "production":
            production_issues = validate_production_config()
            if production_issues:
                issue_names = ", ".join(
                    sorted({issue.name for issue in production_issues})
                )
                raise RuntimeError(
                    f"Production configuration is invalid: {issue_names}"
                )
        settings = validate_v2_runtime_configuration()
        secret = os.getenv("INTERNAL_API_SECRET", "")
        if settings.production and not re.fullmatch(r"[0-9a-f]{64}", secret):
            raise RuntimeError(
                "INTERNAL_API_SECRET must be 64 lowercase hexadecimal characters in production"
            )
        warmup_enabled = embedding_warmup_enabled()
        if settings.production and not warmup_enabled:
            raise RuntimeError(
                "EMBEDDING_WARMUP_ON_STARTUP must be enabled in production"
            )
        app.state.feedback_settings = settings
        if warmup_enabled:
            app.state.embedding_runtime = await asyncio.to_thread(warm_embedding_runtime)
        try:
            yield
        finally:
            await asyncio.to_thread(shutdown_v2_runtime)
        return
    settings = FeedbackSettings.from_env()
    if settings.production and not os.getenv("INTERNAL_API_SECRET"):
        raise RuntimeError("INTERNAL_API_SECRET is required in production")
    producer, consumer, feedback_handler = _build_feedback_runtime(settings)
    await producer.start()
    await asyncio.to_thread(feedback_handler.healthy)
    await consumer.start()
    app.state.feedback_settings = settings
    try:
        yield
    finally:
        if consumer:
            await consumer.stop()
        redis_client = producer.redis_client if producer else None
        close = getattr(redis_client, "close", None)
        if close:
            await asyncio.to_thread(close)
        qdrant_client = feedback_handler.qdrant if feedback_handler else None
        close_qdrant = getattr(qdrant_client, "close", None)
        if close_qdrant:
            await asyncio.to_thread(close_qdrant)
        producer = None
        consumer = None
        feedback_handler = None


app = FastAPI(
    title="Git Social ML API",
    description="Authenticated ML operations and durable feedback ingestion.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    details = [
        {
            "path": ".".join(str(item) for item in error.get("loc", ())),
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "validation_error"),
        }
        # FastAPI's validation wrapper supports a narrower ``errors`` signature
        # than Pydantic on some supported versions.  We copy only safe fields
        # below, so raw inputs are never returned or logged.
        for error in exc.errors()
    ]
    return _error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="REQUEST_VALIDATION_FAILED",
        message="Request validation failed.",
        retryable=False,
        details=details,
    )


@app.exception_handler(StarletteHTTPException)
async def service_http_error(request: Request, exc: StarletteHTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    code_by_status = {
        500: "INTERNAL_ERROR",
        401: "UNAUTHORIZED",
        404: "NOT_FOUND",
        409: "VERSION_CONFLICT",
        422: "REQUEST_VALIDATION_FAILED",
        429: "RATE_LIMITED",
        503: "DEPENDENCY_UNAVAILABLE",
    }
    explicit_code = getattr(exc, "error_code", None)
    explicit_retryable = getattr(exc, "retryable", None)
    explicit_details = getattr(exc, "safe_details", None)
    return _error_response(
        request,
        status_code=exc.status_code,
        code=(
            explicit_code
            if isinstance(explicit_code, str) and explicit_code
            else code_by_status.get(exc.status_code, f"HTTP_{exc.status_code}")
        ),
        message=message,
        retryable=(
            explicit_retryable
            if isinstance(explicit_retryable, bool)
            else exc.status_code in {408, 425, 429, 502, 503, 504}
        ),
        headers=dict(exc.headers or {}),
        details=explicit_details if isinstance(explicit_details, dict) else None,
    )


@app.exception_handler(Exception)
async def unhandled_service_error(request: Request, exc: Exception):
    logger.error(
        "Unhandled ML API error request_id=%s path=%s error_type=%s",
        _request_id(request),
        request.url.path,
        type(exc).__name__,
        extra={
            "request_context": {
                "request_id": _request_id(request),
                "path": request.url.path,
                "error_type": type(exc).__name__,
            }
        },
    )
    return _error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        message="The ML service could not complete the request.",
        retryable=False,
    )


@app.middleware("http")
async def authenticate_non_health_routes(request: Request, call_next):
    """Fail closed for every route except the single health endpoint."""
    supplied_request_id = request.headers.get("x-request-id", "").strip()
    request.state.request_id = (
        supplied_request_id
        if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_request_id)
        else str(uuid.uuid4())
    )
    if request.method in {"POST", "PUT", "PATCH"}:
        raw_content_length = request.headers.get("content-length")
        if raw_content_length is None:
            return _error_response(
                request,
                status_code=status.HTTP_411_LENGTH_REQUIRED,
                code="CONTENT_LENGTH_REQUIRED",
                message="Content-Length is required for request bodies.",
                retryable=False,
            )
        try:
            content_length = int(raw_content_length)
        except ValueError:
            content_length = -1
        if content_length < 0:
            return _error_response(
                request,
                status_code=status.HTTP_400_BAD_REQUEST,
                code="INVALID_CONTENT_LENGTH",
                message="Content-Length is invalid.",
                retryable=False,
            )
        if content_length > MAX_REQUEST_BODY_BYTES:
            return _error_response(
                request,
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                code="REQUEST_TOO_LARGE",
                message="Request body exceeds the service limit.",
                retryable=False,
            )
    if request.url.path.startswith("/api/v1/") and not legacy_api_enabled():
        return _error_response(
            request,
            status_code=404,
            code="LEGACY_API_DISABLED",
            message="Legacy ML API is disabled.",
            retryable=False,
        )
    if request.url.path == "/api/v1/health":
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response
    secret = os.getenv("INTERNAL_API_SECRET")
    if not secret:
        return _error_response(
            request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="AUTH_NOT_CONFIGURED",
            message="Internal API authentication is not configured.",
            retryable=False,
        )
    supplied = request.headers.get(internal_api_header_name())
    if not constant_time_secret_matches(supplied, secret):
        return _error_response(
            request,
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="UNAUTHORIZED",
            message="Unauthorized.",
            retryable=False,
        )
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response


@app.middleware("http")
async def observe_api_requests(request: Request, call_next):
    """Record bounded, fixed-cardinality request metrics for the V2 API."""

    started = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        if request.url.path.startswith("/api/v2/"):
            record_api_request(
                path=request.url.path,
                method=request.method,
                status_code=response.status_code if response is not None else 500,
                duration_seconds=time.perf_counter() - started,
            )


@app.post("/api/v1/feedback", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(request: FeedbackRequest):
    action = normalize_interaction(request.action)
    try:
        definition = get_interaction(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.dwell_seconds is not None and not math.isfinite(request.dwell_seconds):
        raise HTTPException(status_code=422, detail="dwell_seconds must be finite")
    if request.occurred_at.tzinfo is None:
        raise HTTPException(status_code=422, detail="occurred_at must include a timezone")
    if action == "dwell" and request.dwell_seconds is None:
        raise HTTPException(status_code=422, detail="dwell_seconds is required for dwell")
    if action != "dwell" and request.dwell_seconds is not None:
        raise HTTPException(status_code=422, detail="dwell_seconds is only valid for dwell")

    queued = False
    if definition.realtime:
        if producer is None:
            raise HTTPException(status_code=503, detail="Feedback stream is unavailable.")
        try:
            await producer.submit_feedback(
                user_id=str(request.user_id),
                repo_id=request.repo_id,
                action=action,
                event_id=str(request.event_id),
                occurred_at=request.occurred_at.isoformat(),
                schema_version=request.schema_version,
                dwell_seconds=request.dwell_seconds,
            )
            queued = True
        except Exception as exc:
            logger.error(
                "Unable to publish legacy feedback error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(status_code=503, detail="Feedback stream is unavailable.")

    return {
        "status": "accepted",
        "data": {
            "event_id": str(request.event_id),
            "user_id": str(request.user_id),
            "repo_id": request.repo_id,
            "action": action,
            "reference_score": definition.reference_score,
            "queued_for_realtime_ml": queued,
        },
    }


def _load_service(module_name: str, class_name: str) -> Any:
    """Load other owned workstreams only when their endpoint is called."""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


@app.post("/api/v1/recommendations/generate")
async def generate_recommendations(request: RecommendationRequest):
    global retrieval_engine
    try:
        retrieval_engine = retrieval_engine or _load_service("retrieval_engine", "RetrievalEngine")
        batches = await run_in_threadpool(
            retrieval_engine.fetch_onboarding_batches,
            request.user_id,
            is_cold_start=request.is_cold_start,
        )
        return {"status": "success", "user_id": request.user_id, "data": batches}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Legacy recommendation generation failed error_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail="Recommendation generation failed.")


@app.post("/api/v1/onboard")
async def onboard_user(request: OnboardingRequest):
    global onboarding_pipeline
    onboarding_pipeline = onboarding_pipeline or _load_service(
        "scripts.user_onboarding", "UserOnboardingPipeline"
    )
    data = request.model_dump(exclude_none=True)
    user_id = data.pop("user_id")
    try:
        success = await run_in_threadpool(onboarding_pipeline.onboard_user, user_id, data)
        if not success:
            raise HTTPException(status_code=500, detail="User onboarding failed.")
        return {"status": "success", "user_id": user_id}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Legacy user onboarding failed error_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail="User onboarding failed.")


@app.post("/api/v1/embed-repo")
async def embed_repo(request: EmbedRepoRequest):
    global repo_embedding_pipeline
    repo_embedding_pipeline = repo_embedding_pipeline or _load_service(
        "embedding.embedding_pipeline", "RepositoryEmbeddingPipeline"
    )
    payload = {
        "id": request.repo_id,
        "repo_id": request.repo_id,
        "full_name": request.github_repo,
        "html_url": request.github_repo_url,
        "description": request.description or "",
        "primary_language": request.primary_language or "Unknown",
        "languages": request.languages,
        "topics": request.topics,
        "extracted_paragraphs": [request.readme_summary] if request.readme_summary else [],
        "readme_length": len(request.readme_summary or ""),
        "star_count": request.star_count,
        "fork_count": request.fork_count,
        "open_issues_count": request.open_issues_count,
        "created_at": request.created_at,
        "updated_at": request.updated_at,
    }
    try:
        results = await run_in_threadpool(repo_embedding_pipeline.index_batch, [payload])
        result = results[0] if results else None
        return {
            "status": "success",
            "repo_id": request.repo_id,
            "github_repo": request.github_repo,
            "embedding_version": getattr(result, "embedding_version", None),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Legacy repository embedding failed error_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail="Repository embedding failed.")


@app.get("/api/v1/health")
async def health_check():
    checks = {"lifecycle": producer is not None and feedback_handler is not None}
    checks["consumer"] = bool(consumer and consumer.healthy)
    try:
        checks["redis"] = bool(
            producer and producer.redis_client
            and await asyncio.to_thread(producer.redis_client.ping)
        )
    except Exception:
        checks["redis"] = False
    try:
        checks["qdrant"] = bool(
            feedback_handler and await asyncio.to_thread(feedback_handler.healthy)
        )
    except Exception:
        checks["qdrant"] = False
    healthy = all(checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "unhealthy", "checks": checks},
    )


from api.v2 import router as v2_router

app.include_router(v2_router)
