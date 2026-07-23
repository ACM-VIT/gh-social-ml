"""Bounded Redis Stream publisher for feedback events."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .events import FeedbackEvent
from .settings import FeedbackSettings

logger = logging.getLogger("pipeline.feedback.producer")
_in_memory_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()


def get_in_memory_queue() -> asyncio.Queue[dict[str, str]]:
    return _in_memory_queue


def create_redis_client(settings: FeedbackSettings) -> Any:
    if not settings.redis_url:
        return None
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("redis is required by the production feedback API") from exc
    return redis.from_url(settings.redis_url, decode_responses=True)


class FeedbackProducer:
    def __init__(
        self,
        redis_client: Any | None = None,
        settings: FeedbackSettings | None = None,
        *,
        queue: asyncio.Queue[dict[str, str]] | None = None,
    ) -> None:
        self.settings = settings or FeedbackSettings.from_env()
        self.redis_client = redis_client if redis_client is not None else create_redis_client(self.settings)
        self.queue = queue or _in_memory_queue

    async def start(self) -> None:
        if self.redis_client is not None:
            await asyncio.to_thread(self.redis_client.ping)
            return
        if self.settings.production or not self.settings.allow_memory_fallback:
            raise RuntimeError("Redis is required; no REDIS_URL was configured")
        logger.warning("Explicit development-only in-memory feedback queue is enabled")

    async def submit(self, event: FeedbackEvent) -> bool:
        if not event or event.action == "impression":
            return True
        fields = event.as_redis_fields()
        if self.redis_client is not None:
            await asyncio.to_thread(
                self.redis_client.xadd,
                self.settings.stream_name,
                fields,
                maxlen=self.settings.stream_maxlen,
                approximate=True,
            )
            return True
        if self.settings.production or not self.settings.allow_memory_fallback:
            raise RuntimeError("Redis is unavailable and memory fallback is disabled")
        await self.queue.put(fields)
        return True

    async def submit_feedback(
        self,
        user_id: str,
        repo_id: str,
        action: str,
        *,
        event_id: str,
        occurred_at: str,
        schema_version: int = 2,
        dwell_seconds: float | None = None,
    ) -> bool:
        return await self.submit(
            FeedbackEvent(
                event_id=event_id,
                user_id=user_id,
                repo_id=repo_id,
                action=action,
                occurred_at=occurred_at,
                schema_version=schema_version,
                dwell_seconds=dwell_seconds,
            )
        )
