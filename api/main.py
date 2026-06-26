import logging
import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from feedback.producer import FeedbackProducer
from feedback.consumer import FeedbackConsumer
from retrieval_engine import RetrievalEngine
from scripts.user_onboarding import UserOnboardingPipeline
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline.api")

# ── Shared-secret guard ──────────────────────────────────────────────────────
_INTERNAL_SECRET = os.getenv("INTERNAL_API_SECRET")


async def require_internal_secret(
    x_internal_secret: str | None = Header(default=None, alias="x-internal-secret"),
) -> None:
    """FastAPI dependency that validates the X-Internal-Secret header.

    Fails closed (503) if INTERNAL_API_SECRET is not configured on this server
    so a misconfigured deploy is never accidentally open.
    """
    if not _INTERNAL_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal API secret is not configured on this server.",
        )
    if x_internal_secret != _INTERNAL_SECRET:
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
    user_id: str = Field(..., description="Unique ID of the user performing the action")
    repo_id: str = Field(..., description="Full name or UUID of the repository")
    action: str = Field(
        ...,
        description="Interaction action type: click, like, save, skip, or dwell",
    )
    dwell_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Observed dwell time in seconds. Required when action is 'dwell'; "
            "ignored for all other actions."
        ),
    )


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Application user UUID")


class OnboardingRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Application user UUID")
    github_username: str | None = Field(default=None, description="Linked GitHub username")
    username: str | None = None
    full_name: str | None = None
    bio: str | None = None
    interests: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    avatar_url: str | None = None


class EmbedRepoRequest(BaseModel):
    repo_id: str = Field(..., min_length=1, description="Backend repository UUID")
    github_repo: str = Field(..., min_length=1, description="GitHub owner/name")
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


@app.post("/api/v1/feedback", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(request: FeedbackRequest):
    """Submit a user interaction event.

    Pushes the event to the processing queue and returns 202 Accepted.
    Supported actions: ``click``, ``like``, ``save``, ``skip``, ``dwell``.
    When ``action`` is ``dwell``, ``dwell_seconds`` must be provided and > 0.
    """
    action = request.action.lower()
    valid_actions = {"click", "like", "save", "skip", "dwell"}
    if action not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action: '{request.action}'. Supported actions are: {sorted(valid_actions)}",
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
            detail=f"Internal API error: {str(exc)}",
        )


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
            detail=f"Failed to generate recommendations: {str(exc)}",
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
            detail=f"Failed to onboard user: {str(exc)}",
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
            detail=f"Failed to embed repository: {str(exc)}",
        )


@app.get("/api/v1/health")
async def health_check():
    """Basic service health check."""
    return {
        "status": "healthy",
        "consumer_running": consumer.running if consumer else False,
    }
