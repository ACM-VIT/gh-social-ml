import os
import logging
import asyncio
from collections.abc import Iterable
from typing import Any, Dict, Optional

logger = logging.getLogger("pipeline.feedback.producer")

# Global in-memory queue fallback for non-Redis environments
_in_memory_queue: asyncio.Queue = asyncio.Queue()


def get_in_memory_queue() -> asyncio.Queue:
    return _in_memory_queue


class FeedbackProducer:
    def __init__(
        self,
        redis_url: str | None = None,
        *,
        allow_in_memory: bool | None = None,
    ) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.allow_in_memory = (
            allow_in_memory
            if allow_in_memory is not None
            else os.getenv("ALLOW_IN_MEMORY_FEEDBACK", "false").lower() == "true"
        )
        self.stream_maxlen = max(10_000, int(os.getenv("FEEDBACK_STREAM_MAXLEN", "1000000")))
        self.redis_client = None

        if self.redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis at %s for feedback streaming", self.redis_url)
            except Exception as exc:
                logger.warning("Redis connection failed: %s", exc)
                self.redis_client = None

    async def submit_feedback(
        self,
        user_id: str,
        repo_id: str,
        action: str,
        *,
        dwell_seconds: Optional[float] = None,
    ) -> bool:
        """Submit a feedback event to the processing queue.

        Pushes to Redis Stream if available. The in-memory queue is used only
        when explicitly enabled for development. ``dwell_seconds`` is included
        when non-None so the consumer can resolve the embedding alpha.
        """
        event: Dict[str, Any] = {
            "user_id": user_id,
            "repo_id": repo_id,
            "action": action,
        }
        # The below conditional is for keeping the event compact — only
        # dwell events carry this field; all other actions leave it absent.
        if dwell_seconds is not None:
            event["dwell_seconds"] = dwell_seconds

        return await self.submit_feedback_batch([event])

    async def submit_feedback_batch(self, events: Iterable[Dict[str, Any]]) -> bool:
        """Publish a bounded event batch without blocking the API event loop."""
        batch = list(events)
        if not batch:
            return True

        if self.redis_client:
            try:
                def publish() -> None:
                    pipeline = self.redis_client.pipeline(transaction=False)
                    for event in batch:
                        pipeline.xadd(
                            "feedback_stream",
                            {key: str(value) for key, value in event.items()},
                            maxlen=self.stream_maxlen,
                            approximate=True,
                        )
                    pipeline.execute()

                await asyncio.to_thread(publish)
                logger.info("Published %d feedback event(s) to Redis Stream", len(batch))
                return True
            except Exception as exc:
                logger.error("Failed to publish feedback batch to Redis Stream: %s", exc)

        if not self.allow_in_memory:
            logger.error("Feedback rejected because durable Redis streaming is unavailable.")
            return False

        for event in batch:
            await _in_memory_queue.put(event)
        logger.info("Enqueued %d feedback event(s) in memory", len(batch))
        return True
