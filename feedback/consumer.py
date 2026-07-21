"""Reliable Redis Streams consumer for online feedback."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from typing import Any, Iterable

from .event_handlers import FeedbackHandler
from .events import FeedbackEvent
from .producer import create_redis_client, get_in_memory_queue
from .settings import FeedbackSettings

logger = logging.getLogger("pipeline.feedback.consumer")


class FeedbackConsumer:
    def __init__(
        self,
        handler: FeedbackHandler | None = None,
        redis_client: Any | None = None,
        settings: FeedbackSettings | None = None,
    ) -> None:
        self.settings = settings or FeedbackSettings.from_env()
        self.handler = handler or FeedbackHandler(settings=self.settings)
        self.redis_client = (
            redis_client if redis_client is not None else create_redis_client(self.settings)
        )
        self.consumer_name = (
            f"{self.settings.consumer_name_prefix}-{socket.gethostname()}-{os.getpid()}"
        )
        self.running = False
        self.task: asyncio.Task[None] | None = None
        self._memory_locks: dict[str, asyncio.Lock] = {}

    @property
    def healthy(self) -> bool:
        return bool(self.running and self.task and not self.task.done())

    async def start(self) -> None:
        if self.task and not self.task.done():
            return
        if self.redis_client is None:
            if self.settings.production or not self.settings.allow_memory_fallback:
                raise RuntimeError("Redis is required for the feedback consumer")
            self.running = True
            self.task = asyncio.create_task(self._memory_loop(), name="feedback-memory-consumer")
            return
        await asyncio.to_thread(self.redis_client.ping)
        await self._ensure_group()
        self.running = True
        self.task = asyncio.create_task(self._redis_loop(), name="feedback-redis-consumer")

    async def stop(self) -> None:
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _ensure_group(self) -> None:
        try:
            await asyncio.to_thread(
                self.redis_client.xgroup_create,
                self.settings.stream_name,
                self.settings.consumer_group,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise RuntimeError("failed to create feedback consumer group") from exc

    async def _memory_loop(self) -> None:
        queue = get_in_memory_queue()
        while self.running:
            try:
                payload = await queue.get()
                try:
                    event = FeedbackEvent.from_mapping(payload)
                    lock = self._memory_locks.setdefault(event.user_id, asyncio.Lock())
                    async with lock:
                        success = await asyncio.to_thread(self._call_handler, event)
                    if not success:
                        await queue.put(payload)
                        await asyncio.sleep(0.1)
                finally:
                    queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Development feedback consumer failed error_type=%s",
                    type(exc).__name__,
                )
                await asyncio.sleep(0.25)

    async def _redis_loop(self) -> None:
        last_reclaim = 0.0
        while self.running:
            try:
                now = time.monotonic()
                if now - last_reclaim >= self.settings.reclaim_interval_seconds:
                    for message_id, payload in await self._reclaim_stale():
                        await self._process_message(message_id, payload)
                    last_reclaim = now

                response = await asyncio.to_thread(
                    self.redis_client.xreadgroup,
                    self.settings.consumer_group,
                    self.consumer_name,
                    {self.settings.stream_name: ">"},
                    count=self.settings.read_batch_size,
                    block=self.settings.read_block_ms,
                )
                for _stream, messages in response or []:
                    for message_id, payload in messages:
                        await self._process_message(str(message_id), payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Redis feedback consumer loop failed error_type=%s",
                    type(exc).__name__,
                )
                await asyncio.sleep(1.0)

    async def _reclaim_stale(self) -> list[tuple[str, dict[str, str]]]:
        try:
            response = await asyncio.to_thread(
                self.redis_client.xautoclaim,
                self.settings.stream_name,
                self.settings.consumer_group,
                self.consumer_name,
                self.settings.reclaim_idle_ms,
                "0-0",
                count=self.settings.read_batch_size,
            )
        except (AttributeError, TypeError):
            logger.warning("Redis client does not support XAUTOCLAIM; pending reclaim is disabled")
            return []
        if not response:
            return []
        messages: Iterable[Any] = response[1] if len(response) > 1 else []
        return [(str(message_id), payload) for message_id, payload in messages]

    async def _process_message(self, message_id: str, payload: dict[str, str]) -> None:
        try:
            event = FeedbackEvent.from_mapping(payload)
        except (TypeError, ValueError, KeyError) as exc:
            await self._dead_letter(message_id, payload, f"invalid event: {exc}")
            await self._ack(message_id)
            return

        processed_key = f"feedback:processed:{event.event_id}"
        if await asyncio.to_thread(self.redis_client.exists, processed_key):
            await self._ack(message_id)
            return

        lock = self.redis_client.lock(
            f"feedback:user-lock:{event.user_id}",
            timeout=self.settings.user_lock_ttl_seconds,
            blocking_timeout=self.settings.user_lock_wait_seconds,
        )
        acquired = await asyncio.to_thread(lock.acquire)
        if not acquired:
            return
        try:
            if await asyncio.to_thread(self.redis_client.exists, processed_key):
                await self._ack(message_id)
                return
            try:
                success = await asyncio.to_thread(self._call_handler, event)
            except ValueError as exc:
                await self._dead_letter(message_id, payload, f"non-retryable event: {exc}")
                await self._ack(message_id)
                return
            if not success:
                await self._record_failure(message_id, payload, event.event_id)
                return
            await asyncio.to_thread(
                self.redis_client.set,
                processed_key,
                "1",
                ex=self.settings.idempotency_ttl_seconds,
            )
            await self._ack(message_id)
        finally:
            try:
                await asyncio.to_thread(lock.release)
            except Exception:
                logger.warning("Feedback user lock expired before release for %s", event.user_id)

    def _call_handler(self, event: FeedbackEvent) -> bool:
        return self.handler.handle_feedback(
            event.user_id,
            event.repo_id,
            event.action,
            event_id=event.event_id,
            dwell_seconds=event.dwell_seconds,
        )

    async def _record_failure(
        self, message_id: str, payload: dict[str, str], event_id: str
    ) -> None:
        attempts_key = f"feedback:attempts:{event_id}"
        attempts = await asyncio.to_thread(self.redis_client.incr, attempts_key)
        await asyncio.to_thread(
            self.redis_client.expire, attempts_key, self.settings.idempotency_ttl_seconds
        )
        if int(attempts) >= self.settings.max_delivery_attempts:
            await self._dead_letter(message_id, payload, "retry limit exceeded")
            await self._ack(message_id)

    async def _dead_letter(
        self, message_id: str, payload: dict[str, str], reason: str
    ) -> None:
        dead = dict(payload)
        dead.update({"source_message_id": message_id, "failure_reason": reason[:500]})
        await asyncio.to_thread(
            self.redis_client.xadd,
            self.settings.dead_letter_stream,
            dead,
            maxlen=self.settings.stream_maxlen,
            approximate=True,
        )

    async def _ack(self, message_id: str) -> None:
        await asyncio.to_thread(
            self.redis_client.xack,
            self.settings.stream_name,
            self.settings.consumer_group,
            message_id,
        )
