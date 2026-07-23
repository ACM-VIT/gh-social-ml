"""Validated runtime settings for the online feedback path.

This module intentionally does not import the root configuration module.  The
online worker must be importable and startable without any database settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class FeedbackSettings:
    environment: str
    redis_url: str | None
    allow_memory_fallback: bool
    stream_name: str
    stream_maxlen: int
    consumer_group: str
    consumer_name_prefix: str
    read_batch_size: int
    read_block_ms: int
    reclaim_idle_ms: int
    reclaim_interval_seconds: float
    idempotency_ttl_seconds: int
    user_lock_ttl_seconds: int
    user_lock_wait_seconds: float
    max_delivery_attempts: int
    dead_letter_stream: str
    qdrant_url: str
    qdrant_api_key: str | None
    repository_collection: str
    repository_vector_name: str
    user_collection: str
    user_vector_name: str | None
    vector_dimension: int
    dwell_min_seconds: float
    dwell_full_credit_seconds: float
    dwell_max_alpha: float
    processed_event_history: int

    @property
    def production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    @classmethod
    def from_env(cls) -> "FeedbackSettings":
        environment = os.getenv("APP_ENV", "development").strip().lower()
        allow_memory = _bool("FEEDBACK_ALLOW_MEMORY_FALLBACK", False)
        if environment in {"production", "prod"} and allow_memory:
            raise ValueError("FEEDBACK_ALLOW_MEMORY_FALLBACK cannot be enabled in production")

        dwell_min = _float("FEEDBACK_DWELL_MIN_SECONDS", 3.0)
        dwell_full = _float("FEEDBACK_DWELL_FULL_CREDIT_SECONDS", 300.0)
        if dwell_full <= dwell_min:
            raise ValueError(
                "FEEDBACK_DWELL_FULL_CREDIT_SECONDS must exceed FEEDBACK_DWELL_MIN_SECONDS"
            )
        dwell_max_alpha = _float("FEEDBACK_DWELL_MAX_ALPHA", 0.15)
        if not 0.0 < dwell_max_alpha <= 0.15:
            raise ValueError("FEEDBACK_DWELL_MAX_ALPHA must be in (0, 0.15]")

        vector_name = os.getenv("USER_PROFILE_VECTOR_NAME") or os.getenv("TARGET_VECTOR_NAME")
        redis_url = os.getenv("REDIS_URL") or (
            "redis://localhost:6379/0"
            if environment not in {"production", "prod"}
            else None
        )
        return cls(
            environment=environment,
            redis_url=redis_url,
            allow_memory_fallback=allow_memory,
            stream_name=os.getenv("FEEDBACK_STREAM_NAME", "feedback_stream"),
            stream_maxlen=_int("FEEDBACK_STREAM_MAXLEN", 100_000),
            consumer_group=os.getenv("FEEDBACK_CONSUMER_GROUP", "feedback_group"),
            consumer_name_prefix=os.getenv("FEEDBACK_CONSUMER_PREFIX", "feedback-worker"),
            read_batch_size=_int("FEEDBACK_READ_BATCH_SIZE", 20),
            read_block_ms=_int("FEEDBACK_READ_BLOCK_MS", 1_000),
            reclaim_idle_ms=_int("FEEDBACK_RECLAIM_IDLE_MS", 60_000),
            reclaim_interval_seconds=_float("FEEDBACK_RECLAIM_INTERVAL_SECONDS", 30.0),
            idempotency_ttl_seconds=_int("FEEDBACK_IDEMPOTENCY_TTL_SECONDS", 604_800),
            user_lock_ttl_seconds=_int("FEEDBACK_USER_LOCK_TTL_SECONDS", 60),
            user_lock_wait_seconds=_float("FEEDBACK_USER_LOCK_WAIT_SECONDS", 10.0),
            max_delivery_attempts=_int("FEEDBACK_MAX_DELIVERY_ATTEMPTS", 5),
            dead_letter_stream=os.getenv("FEEDBACK_DEAD_LETTER_STREAM", "feedback_dead_letter"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            repository_collection=os.getenv(
                "QDRANT_COLLECTION_NAME",
                "osiris_research_corpus_v2_20260722_r1",
            ),
            repository_vector_name=os.getenv("QDRANT_VECTOR_NAME", "repo_embedding"),
            user_collection=os.getenv("USER_PROFILES_COLLECTION", "user_profiles"),
            user_vector_name=vector_name,
            vector_dimension=_int("VECTOR_DIMENSION", 384),
            dwell_min_seconds=dwell_min,
            dwell_full_credit_seconds=dwell_full,
            dwell_max_alpha=dwell_max_alpha,
            processed_event_history=_int("FEEDBACK_PROCESSED_EVENT_HISTORY", 512),
        )
