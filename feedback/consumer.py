import os
import logging
import asyncio
import socket
from typing import Dict, Any

from .producer import get_in_memory_queue
from .event_handlers import FeedbackHandler

logger = logging.getLogger("pipeline.feedback.consumer")


class FeedbackConsumer:
    def __init__(
        self,
        handler: FeedbackHandler | None = None,
        redis_url: str | None = None,
    ) -> None:
        self.handler = handler or FeedbackHandler()
        self.redis_url = redis_url or os.getenv("REDIS_URL")
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
        else:
            # Run in-memory consumer loop
            self.task = asyncio.create_task(self._in_memory_consume_loop())

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

                if not user_id or not repo_id or not action:
                    queue.task_done()
                    continue

                # Process feedback
                try:
                    # Execute vector updates and metric increments
                    self.handler.handle_feedback(user_id, repo_id, action)
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
        # Dynamic consumer name to allow safe horizontal scaling
        consumer_name = f"worker_{socket.gethostname()}_{os.getpid()}"

        # Setup consumer group
        try:
            self.redis_client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except Exception as exc:
            # Group might already exist (BUSYGROUP)
            if "BUSYGROUP" not in str(exc):
                logger.error("Failed to create Redis Stream group: %s. Falling back to In-Memory.", exc)
                self.task = asyncio.create_task(self._in_memory_consume_loop())
                return

        while self.running:
            try:
                # Read from group
                # Since redis-py blocking commands block the event loop, we run in executor
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.redis_client.xreadgroup(
                        group_name,
                        consumer_name,
                        {stream_name: ">"},
                        count=1,
                        block=1000,
                    ),
                )

                if not response:
                    await asyncio.sleep(0.1)
                    continue

                for stream, messages in response:
                    for message_id, payload in messages:
                        user_id = payload.get("user_id")
                        repo_id = payload.get("repo_id")
                        action = payload.get("action")

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
                                    # Execute vector updates and metric increments
                                    await loop.run_in_executor(
                                        None,
                                        self.handler.handle_feedback,
                                        user_id,
                                        repo_id,
                                        action
                                    )
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
