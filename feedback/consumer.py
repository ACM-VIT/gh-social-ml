import os
import logging
import asyncio
import socket
import time
from typing import Dict, Any

from .producer import get_in_memory_queue
from .event_handlers import FeedbackHandler

logger = logging.getLogger("pipeline.feedback.consumer")


class FeedbackConsumer:
    def __init__(
        self,
        handler: FeedbackHandler | None = None,
        redis_url: str | None = None,
        *,
        allow_in_memory: bool | None = None,
    ) -> None:
        self.handler = handler or FeedbackHandler()
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.allow_in_memory = (
            allow_in_memory
            if allow_in_memory is not None
            else os.getenv("ALLOW_IN_MEMORY_FEEDBACK", "false").lower() == "true"
        )
        self.redis_client = None
        self.running = False
        self.task: asyncio.Task | None = None

        if self.redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for consuming feedback stream")
            except Exception:
                self.redis_client = None

    async def start(self) -> None:
        """Start the consumer loop in the background."""
        self.running = True
        logger.info("Starting Feedback Consumer worker...")

        if self.redis_client:
            # Run Redis consumer loop
            self.task = asyncio.create_task(self._redis_consume_loop())
        elif self.allow_in_memory:
            # Run in-memory consumer loop
            self.task = asyncio.create_task(self._in_memory_consume_loop())
        else:
            self.running = False
            raise RuntimeError("REDIS_URL is required for durable feedback processing")

    def stop(self) -> None:
        """Stop the consumer loop."""
        self.running = False
        if self.task:
            self.task.cancel()

    async def _in_memory_consume_loop(self) -> None:
        queue = get_in_memory_queue()
        logger.info("Feedback Consumer running in In-Memory Queue mode.")

        while self.running:
            try:
                # Wait for an event
                event = await queue.get()
                user_id = event.get("user_id")
                repo_id = event.get("repo_id")
                action = event.get("action")
                # The below extraction is for passing observed dwell time to
                # the handler so it can compute the embedding alpha correctly.
                dwell_seconds_raw = event.get("dwell_seconds")
                dwell_seconds = float(dwell_seconds_raw) if dwell_seconds_raw is not None else None

                if not user_id or not repo_id or not action:
                    queue.task_done()
                    continue

                # Process feedback in a background thread to prevent blocking the asyncio event loop
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self.handler.handle_feedback(
                            user_id, repo_id, action, dwell_seconds=dwell_seconds
                        )
                    )
                except Exception as exc:
                    logger.error("Exception occurred while handling feedback: %s", exc)
                finally:
                    queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in in-memory consume loop: %s", exc)
                await asyncio.sleep(1)

    async def _redis_consume_loop(self) -> None:
        logger.info("Feedback Consumer running in Redis Streams mode.")
        stream_name = "feedback_stream"
        group_name = "feedback_group"
        claim_idle_ms = max(10_000, int(os.getenv("FEEDBACK_CLAIM_IDLE_MS", "60000")))
        last_claim_at = 0.0
        # Dynamic consumer name to allow safe horizontal scaling
        consumer_name = f"worker_{socket.gethostname()}_{os.getpid()}"

        # Setup consumer group
        try:
            self.redis_client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except Exception as exc:
            # Group might already exist (BUSYGROUP)
            if "BUSYGROUP" not in str(exc):
                logger.error("Failed to create Redis Stream group: %s", exc)
                if self.allow_in_memory:
                    self.task = asyncio.create_task(self._in_memory_consume_loop())
                else:
                    self.running = False
                return

        loop = asyncio.get_running_loop()
        while self.running:
            try:
                claimed_messages = []
                now = time.monotonic()
                if now - last_claim_at >= 60.0:
                    last_claim_at = now
                    try:
                        claim_response = await loop.run_in_executor(
                            None,
                            lambda: self.redis_client.xautoclaim(
                                stream_name,
                                group_name,
                                consumer_name,
                                claim_idle_ms,
                                start_id="0-0",
                                count=32,
                            ),
                        )
                        if (
                            isinstance(claim_response, (list, tuple))
                            and len(claim_response) >= 2
                            and isinstance(claim_response[1], list)
                        ):
                            claimed_messages = claim_response[1]
                    except Exception as claim_exc:
                        logger.warning("Failed to reclaim stale feedback messages: %s", claim_exc)

                # Read from group
                # Since redis-py blocking commands block the event loop, we run in executor
                response = await loop.run_in_executor(
                    None,
                    lambda: self.redis_client.xreadgroup(
                        group_name,
                        consumer_name,
                        {stream_name: ">"},
                        count=32,
                        block=1000,
                    ),
                )

                if claimed_messages:
                    response = [(stream_name, claimed_messages), *response]

                if not response:
                    await asyncio.sleep(0.1)
                    continue

                for stream, messages in response:
                    for message_id, payload in messages:
                        user_id = payload.get("user_id")
                        repo_id = payload.get("repo_id")
                        action = payload.get("action")
                        # Redis Stream values are strings — parse dwell_seconds back to float.
                        dwell_seconds_raw = payload.get("dwell_seconds")
                        try:
                            dwell_seconds = float(dwell_seconds_raw) if dwell_seconds_raw is not None else None
                        except (TypeError, ValueError):
                            logger.warning(
                                "Could not parse dwell_seconds '%s' for message %s. Treating as None.",
                                dwell_seconds_raw, message_id,
                            )
                            dwell_seconds = None

                        if user_id and repo_id and action:
                            processed_key = f"feedback:processed:{message_id}"
                            
                            # Check if already processed (using Redis exists check) to prevent dual processing
                            already_processed = False
                            try:
                                already_processed = await loop.run_in_executor(
                                    None,
                                    self.redis_client.exists,
                                    processed_key
                                )
                            except Exception as check_exc:
                                logger.warning("Failed to check processed status in Redis: %s", check_exc)
                            
                            if already_processed:
                                logger.info(
                                    "Message %s was already processed successfully. Skipping processing and retrying ack.",
                                    message_id
                                )
                                processed_success = True
                            else:
                                processed_success = False
                                try:
                                    # Execute the ML-owned user-vector update.
                                    handler_succeeded = await loop.run_in_executor(
                                        None,
                                        lambda: self.handler.handle_feedback(
                                            user_id, repo_id, action,
                                            dwell_seconds=dwell_seconds,
                                        ),
                                    )
                                    if not handler_succeeded:
                                        raise RuntimeError("Feedback handler rejected the event")
                                    processed_success = True
                                    
                                    # Mark as processed in Redis (expires in 24 hours)
                                    try:
                                        await loop.run_in_executor(
                                            None,
                                            lambda: self.redis_client.set(processed_key, "1", ex=86400)
                                        )
                                    except Exception as set_exc:
                                        logger.warning("Failed to mark message %s as processed in Redis: %s", message_id, set_exc)
                                except Exception as exc:
                                    logger.error("Exception handling Redis Stream feedback: %s", exc)

                            if processed_success:
                                # Acknowledge message with retries on failure to handle transient connection issues
                                ack_success = False
                                for attempt in range(3):
                                    try:
                                        await loop.run_in_executor(
                                            None,
                                            self.redis_client.xack,
                                            stream_name,
                                            group_name,
                                            message_id,
                                        )
                                        ack_success = True
                                        break
                                    except Exception as ack_exc:
                                        logger.warning(
                                            "Attempt %d failed to acknowledge message %s: %s. Retrying...",
                                            attempt + 1,
                                            message_id,
                                            ack_exc,
                                        )
                                        await asyncio.sleep(2 ** attempt)

                                if not ack_success:
                                    logger.error(
                                        "Failed to acknowledge successfully processed message %s after 3 attempts. "
                                        "Message remains in PEL and may be re-processed.",
                                        message_id,
                                    )
                        else:
                            logger.error(
                                "Skipping malformed message (missing user_id, repo_id, or action): ID %s, payload %s",
                                message_id,
                                payload,
                            )
                            try:
                                # Acknowledge to remove it from PEL and prevent infinite redelivery
                                await loop.run_in_executor(
                                    None,
                                    self.redis_client.xack,
                                    stream_name,
                                    group_name,
                                    message_id,
                                )
                            except Exception as exc:
                                logger.error("Failed to acknowledge malformed message: %s", exc)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in Redis stream consume loop: %s", exc)
                await asyncio.sleep(1)
