import logging
import os

# Load .env FIRST — before any imports that call os.getenv() at module level
# (e.g. scripts/user_onboarding.py reads QDRANT_URL on import)
from dotenv import load_dotenv
load_dotenv()

# Force PyTorch and underlying math libraries to use a single thread to save RAM during deployment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from feedback.producer import FeedbackProducer
from feedback.consumer import FeedbackConsumer
from feedback.event_handlers import normalize_feedback_action
from retrieval_engine import RetrievalEngine
from scripts.user_onboarding import UserOnboardingPipeline
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline.api")

import hmac

# ── Shared-secret guard ──────────────────────────────────────────────────────

async def require_internal_secret(
    x_internal_secret: str | None = Header(default=None, alias="x-internal-secret"),
) -> None:
    """FastAPI dependency that validates the X-Internal-Secret header.

    Fails closed (503) if INTERNAL_API_SECRET is not configured on this server
    so a misconfigured deploy is never accidentally open.
    """
    internal_secret = os.getenv("INTERNAL_API_SECRET")
    if not internal_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal API secret is not configured on this server.",
        )
    if not x_internal_secret or not hmac.compare_digest(x_internal_secret.encode("utf-8"), internal_secret.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: invalid or missing X-Internal-Secret header.",
        )


# Global instances
producer: FeedbackProducer | None = None
consumer: FeedbackConsumer | None = None
retrieval_engine: RetrievalEngine | None = None
onboarding_pipeline: UserOnboardingPipeline | None = None
repo_embedding_pipeline: RepositoryEmbeddingPipeline | None = None


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="Application user UUID")
    repo_id: str = Field(..., min_length=1, max_length=255, description="Repository UUID")
    action: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Canonical backend feedback action",
    )
    dwell_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Observed dwell time in seconds. Required when action is 'dwell'; "
            "ignored for all other actions."
        ),
    )


class FeedbackBatchRequest(BaseModel):
    events: list[FeedbackRequest] = Field(..., min_length=1, max_length=100)


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="Application user UUID")
    is_cold_start: bool = Field(default=False, description="True if user has zero interactions")


class OnboardingRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="Application user UUID")
    github_username: str | None = Field(default=None, max_length=255, description="Linked GitHub username")
    username: str | None = Field(default=None, max_length=255)
    full_name: str | None = Field(default=None, max_length=255)
    bio: str | None = Field(default=None, max_length=2000)
    interests: list[str] = Field(default_factory=list, max_length=100)
    skills: list[str] = Field(default_factory=list, max_length=100)
    tech_stack: list[str] = Field(default_factory=list, max_length=100)
    avatar_url: str | None = Field(default=None, max_length=2048)


class EmbedRepoRequest(BaseModel):
    repo_id: str = Field(..., min_length=1, max_length=255, description="Backend repository UUID")
    github_repo: str = Field(..., min_length=1, max_length=255, description="GitHub owner/name")
    github_repo_url: str | None = Field(default=None, max_length=2048)
    description: str | None = Field(default=None, max_length=5000)
    primary_language: str | None = Field(default=None, max_length=255)
    languages: list[str] = Field(default_factory=list, max_length=100)
    topics: list[str] = Field(default_factory=list, max_length=100)
    readme_summary: str | None = Field(default=None, max_length=50000)
    star_count: int = Field(default=0, ge=0)
    fork_count: int = Field(default=0, ge=0)
    open_issues_count: int = Field(default=0, ge=0)
    created_at: str | None = None
    updated_at: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize components and background worker tasks."""
    global producer, consumer, retrieval_engine, onboarding_pipeline, repo_embedding_pipeline
    logger.info("Initializing API components...")
    producer = FeedbackProducer()
    consumer = FeedbackConsumer()
    retrieval_engine = RetrievalEngine()
    onboarding_pipeline = UserOnboardingPipeline()
    repo_embedding_pipeline = RepositoryEmbeddingPipeline()
    
    # Start background event consume worker loop
    await consumer.start()
    logger.info("Feedback Ingestion API and Background Consumer started successfully.")
    
    yield
    
    # Shutdown components
    logger.info("Shutting down API components...")
    if consumer:
        consumer.stop()
    logger.info("API components shut down.")


app = FastAPI(
    title="Git Social ML - Feedback Ingestion API",
    description="Real-time ingestion endpoint for user feedback events to update recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post(
    "/api/v1/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_internal_secret)],
)
async def submit_feedback(request: FeedbackRequest):
    """Submit a user interaction event.

    Pushes the event to the processing queue and returns 202 Accepted.
    Canonical backend actions and the legacy ``click``/``skip`` aliases are accepted.
    When ``action`` is ``dwell``, ``dwell_seconds`` must be provided and > 0.
    """
    action = normalize_feedback_action(request.action)
    if action is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action: '{request.action}'.",
        )

    # Validate dwell-specific contract
    if action == "dwell" and request.dwell_seconds is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'dwell_seconds' is required when action is 'dwell'.",
        )

    try:
        # Enqueue the event (dwell_seconds is forwarded as keyword-only arg)
        success = await producer.submit_feedback(
            user_id=request.user_id,
            repo_id=request.repo_id,
            action=action,
            dwell_seconds=request.dwell_seconds,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enqueue feedback event.",
            )

        response_data: dict = {
            "user_id": request.user_id,
            "repo_id": request.repo_id,
            "action": action,
        }
        if request.dwell_seconds is not None:
            response_data["dwell_seconds"] = request.dwell_seconds

        return {
            "status": "accepted",
            "message": "Feedback event received and queued successfully.",
            "data": response_data,
        }

    except HTTPException:
        # Re-raise HTTPExceptions explicitly to prevent wrapping them in 500
        raise
    except Exception as exc:
        logger.error("Failed to process feedback submission: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal API error while queuing feedback.",
        )


@app.post(
    "/api/v1/feedback/batch",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_internal_secret)],
)
async def submit_feedback_batch(request: FeedbackBatchRequest):
    """Enqueue one backend activity batch with a single Redis round trip."""
    events: list[dict] = []
    for event in request.events:
        action = normalize_feedback_action(event.action)
        if action is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: '{event.action}'.",
            )
        if action == "dwell" and event.dwell_seconds is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="'dwell_seconds' is required when action is 'dwell'.",
            )
        payload = {
            "user_id": event.user_id,
            "repo_id": event.repo_id,
            "action": action,
        }
        if event.dwell_seconds is not None:
            payload["dwell_seconds"] = event.dwell_seconds
        events.append(payload)

    try:
        success = await producer.submit_feedback_batch(events)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enqueue feedback events.",
            )
        return {"status": "accepted", "accepted_count": len(events)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to process feedback batch: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal API error while queuing feedback.",
        ) from exc


@app.post("/api/v1/recommendations/generate", dependencies=[Depends(require_internal_secret)])
async def generate_recommendations(request: RecommendationRequest):
    """Generate ranked recommendation batches for a user."""
    if retrieval_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine is not initialized.",
        )

    try:
        batches = await run_in_threadpool(
            retrieval_engine.fetch_onboarding_batches,
            request.user_id,
            is_cold_start=request.is_cold_start,
        )
        return {
            "status": "success",
            "user_id": request.user_id,
            "data": batches,
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Recommendation generation failed for user '%s'", request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations.",
        )


@app.post("/api/v1/onboard", dependencies=[Depends(require_internal_secret)])
async def onboard_user(request: OnboardingRequest):
    """Create or update a user's Qdrant profile embedding."""
    if onboarding_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Onboarding pipeline is not initialized.",
        )

    user_data = request.model_dump(exclude_none=True)
    user_id = user_data.pop("user_id")
    github_username = user_data.pop("github_username", None)
    if github_username:
        user_data["github_username"] = github_username

    try:
        success = await run_in_threadpool(
            onboarding_pipeline.onboard_user,
            user_id,
            user_data,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User onboarding pipeline failed.",
            )
        return {"status": "success", "user_id": user_id}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("User onboarding failed for '%s'", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to onboard user.",
        )


@app.post("/api/v1/embed-repo", dependencies=[Depends(require_internal_secret)])
async def embed_repo(request: EmbedRepoRequest):
    """Embed a repository and upsert its vector into Qdrant."""
    if repo_embedding_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository embedding pipeline is not initialized.",
        )

    payload = {
        "id": request.repo_id,
        "full_name": request.github_repo,
        "repo_id": request.repo_id,
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
            "embedding_version": result.embedding_version if result else None,
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Repository embedding failed for '%s'", request.github_repo)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to embed repository.",
        )


@app.get("/api/v1/health")
async def health_check():
    """Readiness check for durable online feedback processing."""
    feedback_ready = bool(
        producer
        and producer.redis_client
        and consumer
        and consumer.redis_client
        and consumer.running
    )
    if not feedback_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Durable feedback processing is unavailable.",
        )
    return {
        "status": "healthy",
        "consumer_running": True,
    }
