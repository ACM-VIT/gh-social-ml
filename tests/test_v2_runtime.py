from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from inference import runtime


def test_recommendation_runtime_rejects_an_undersized_capacity(monkeypatch) -> None:
    monkeypatch.setenv("V2_RECOMMENDATION_EXECUTOR_WORKERS", "4")
    monkeypatch.setenv("V2_RECOMMENDATION_MAX_OUTSTANDING", "3")

    with pytest.raises(ValueError, match="must be at least"):
        runtime.validate_recommendation_runtime()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_cancelled_recommendation_keeps_admission_until_worker_exits(
    monkeypatch,
    anyio_backend,
) -> None:
    class RecordingAdmission:
        def __init__(self) -> None:
            self.held = False
            self.releases = 0

        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            if self.held:
                return False
            self.held = True
            return True

        def release(self) -> None:
            assert self.held
            self.held = False
            self.releases += 1

    started = threading.Event()
    allow_finish = threading.Event()
    finished = threading.Event()
    admission = RecordingAdmission()
    executor = ThreadPoolExecutor(max_workers=1)

    def blocking_job() -> str:
        started.set()
        assert allow_finish.wait(timeout=2)
        finished.set()
        return "finished"

    monkeypatch.setattr(runtime, "recommendation_admission", lambda: admission)
    monkeypatch.setattr(runtime, "recommendation_executor", lambda: executor)
    monkeypatch.setattr(runtime, "recommendation_timeout_seconds", lambda: 120.0)
    task = asyncio.create_task(runtime.run_recommendation_job(blocking_job))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert admission.releases == 0
        with pytest.raises(runtime.RecommendationCapacityError):
            await runtime.run_recommendation_job(lambda: "too-early")

        allow_finish.set()
        assert await asyncio.to_thread(finished.wait, 2)
        for _ in range(100):
            if admission.releases:
                break
            await asyncio.sleep(0)
        assert admission.releases == 1
        assert await runtime.run_recommendation_job(lambda: "accepted") == "accepted"
        assert admission.releases == 2
    finally:
        allow_finish.set()
        executor.shutdown(wait=True, cancel_futures=True)
