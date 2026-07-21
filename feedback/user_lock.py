"""Token-checked renewable Redis locks for vector mutations.

All user-vector read/modify/write operations must use :func:`user_vector_lock`.
The generic context manager is also suitable for long-running repository jobs.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from threading import Event, Thread
from typing import Any, Iterator

from .v2_settings import V2FeedbackSettings


logger = logging.getLogger(__name__)

RENEW_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""
RELEASE_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


class LockAcquisitionError(RuntimeError):
    """Raised when a Redis lock cannot be acquired within its bounded wait."""


class LockLostError(RuntimeError):
    """Raised when ownership is lost while the protected operation is running."""


class RenewableRedisLock:
    """A Redis NX/PX lock whose token is checked on renew and release."""

    def __init__(
        self,
        redis_client: Any,
        key: str,
        *,
        ttl_ms: int,
        wait_seconds: float,
        renew_interval_seconds: float | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if not isinstance(key, str) or not key or len(key) > 512:
            raise ValueError("lock key must contain 1-512 characters")
        if ttl_ms < 100:
            raise ValueError("lock ttl_ms must be >= 100")
        if wait_seconds < 0:
            raise ValueError("lock wait_seconds must be >= 0")
        default_renew = ttl_ms / 3_000.0
        renew_interval = default_renew if renew_interval_seconds is None else renew_interval_seconds
        if renew_interval <= 0 or renew_interval >= ttl_ms / 2_000.0:
            raise ValueError("lock renewal interval must be positive and less than half the TTL")
        if poll_interval_seconds <= 0:
            raise ValueError("lock poll interval must be positive")

        self.redis = redis_client
        self.key = key
        self.ttl_ms = int(ttl_ms)
        self.wait_seconds = float(wait_seconds)
        self.renew_interval_seconds = float(renew_interval)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.token = str(uuid.uuid4())
        self._stop = Event()
        self._lost = Event()
        self._acquired = False
        self._renew_thread: Thread | None = None

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def acquire(self) -> "RenewableRedisLock":
        deadline = time.monotonic() + self.wait_seconds
        while True:
            if self.redis.set(self.key, self.token, nx=True, px=self.ttl_ms):
                self._acquired = True
                self._renew_thread = Thread(
                    target=self._renew_loop,
                    name="redis-lock-renewal",
                    daemon=True,
                )
                self._renew_thread.start()
                return self
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockAcquisitionError("timed out waiting for a protected resource lock")
            self._stop.wait(min(self.poll_interval_seconds, remaining))

    def _renew_loop(self) -> None:
        while not self._stop.wait(self.renew_interval_seconds):
            try:
                renewed = self.redis.eval(
                    RENEW_LOCK_LUA,
                    1,
                    self.key,
                    self.token,
                    str(self.ttl_ms),
                )
                if int(renewed or 0) != 1:
                    self._lost.set()
                    logger.error(
                        "redis lock ownership was lost",
                        extra={"lock_context": {"key": self.key, "status": "lost"}},
                    )
                    return
            except Exception:
                # A failed renewal cannot be assumed to have reached Redis.
                # Fail closed; redelivery/version checks make mutations idempotent.
                self._lost.set()
                logger.error(
                    "redis lock renewal failed",
                    extra={"lock_context": {"key": self.key, "status": "renew_failed"}},
                )
                return

    def assert_owned(self) -> None:
        if not self._acquired or self._lost.is_set():
            raise LockLostError("protected resource lock ownership was lost")

    def release(self) -> bool:
        if not self._acquired:
            return False
        self._stop.set()
        if self._renew_thread is not None:
            self._renew_thread.join(timeout=max(0.1, self.renew_interval_seconds + 0.1))
        try:
            released = self.redis.eval(RELEASE_LOCK_LUA, 1, self.key, self.token)
            owned = int(released or 0) == 1
        except Exception:
            self._lost.set()
            logger.error(
                "redis lock release failed",
                extra={"lock_context": {"key": self.key, "status": "release_failed"}},
            )
            owned = False
        self._acquired = False
        if not owned:
            self._lost.set()
        return owned and not self._lost.is_set()

    def __enter__(self) -> "RenewableRedisLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        owned = self.release()
        if exc_type is None and not owned:
            raise LockLostError("protected resource lock expired before release")
        return False


@contextmanager
def renewable_redis_lock(
    redis_client: Any,
    key: str,
    *,
    ttl_ms: int,
    wait_seconds: float,
    renew_interval_seconds: float | None = None,
) -> Iterator[RenewableRedisLock]:
    """Acquire a renewable token-checked Redis lock as a context manager."""

    with RenewableRedisLock(
        redis_client,
        key,
        ttl_ms=ttl_ms,
        wait_seconds=wait_seconds,
        renew_interval_seconds=renew_interval_seconds,
    ) as lock:
        yield lock


@contextmanager
def user_vector_lock(
    redis_client: Any,
    user_id: str,
    *,
    settings: V2FeedbackSettings | None = None,
    ttl_ms: int | None = None,
    wait_seconds: float | None = None,
    renew_interval_seconds: float | None = None,
) -> Iterator[RenewableRedisLock]:
    """Use the one shared lock namespace for every user-vector mutation."""

    runtime = settings or V2FeedbackSettings.from_env()
    selected_ttl = runtime.user_lock_ttl_ms if ttl_ms is None else ttl_ms
    selected_wait = runtime.user_lock_wait_seconds if wait_seconds is None else wait_seconds
    selected_renew = (
        runtime.user_lock_renew_interval_seconds
        if renew_interval_seconds is None
        else renew_interval_seconds
    )
    key = f"{runtime.user_lock_prefix}:{user_id}"
    with renewable_redis_lock(
        redis_client,
        key,
        ttl_ms=selected_ttl,
        wait_seconds=selected_wait,
        renew_interval_seconds=selected_renew,
    ) as lock:
        yield lock
