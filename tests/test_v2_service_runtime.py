from __future__ import annotations

import asyncio
import threading

import pytest

from api import runtime


@pytest.fixture(autouse=True)
def _clean_service_runtime():
    runtime.shutdown_service_runtime()
    yield
    runtime.shutdown_service_runtime()


def test_service_runtime_rejects_undersized_capacity(monkeypatch) -> None:
    monkeypatch.setenv("V2_FEEDBACK_EXECUTOR_WORKERS", "3")
    monkeypatch.setenv("V2_FEEDBACK_MAX_OUTSTANDING", "2")

    with pytest.raises(ValueError, match="must be at least"):
        runtime.validate_service_runtime()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_cancelled_service_job_retains_capacity_until_worker_exits(
    monkeypatch,
    anyio_backend,
) -> None:
    monkeypatch.setenv("V2_REFRESH_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_REFRESH_MAX_OUTSTANDING", "1")
    monkeypatch.setenv("V2_REFRESH_TIMEOUT_SECONDS", "30")
    started = threading.Event()
    allow_finish = threading.Event()

    def blocking_job() -> str:
        started.set()
        assert allow_finish.wait(timeout=2)
        return "finished"

    task = asyncio.create_task(runtime.run_service_job("refresh", blocking_job))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(runtime.ServiceCapacityError):
        await runtime.run_service_job("refresh", lambda: "too-early")

    allow_finish.set()
    for _ in range(100):
        if runtime.service_runtime_status()["refresh"]["outstanding"] == 0:
            break
        await asyncio.sleep(0)
    assert runtime.service_runtime_status()["refresh"]["outstanding"] == 0
    assert await runtime.run_service_job("refresh", lambda: "accepted") == "accepted"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_timed_out_service_job_retains_capacity_until_worker_exits(
    monkeypatch,
    anyio_backend,
) -> None:
    monkeypatch.setenv("V2_HEALTH_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_HEALTH_MAX_OUTSTANDING", "1")
    monkeypatch.setenv("V2_HEALTH_TIMEOUT_SECONDS", "0.1")
    started = threading.Event()
    allow_finish = threading.Event()

    def blocking_job() -> str:
        started.set()
        assert allow_finish.wait(timeout=2)
        return "finished"

    with pytest.raises(runtime.ServiceDeadlineExceeded):
        await runtime.run_service_job("health", blocking_job)
    assert started.is_set()
    status = runtime.service_runtime_status()["health"]
    assert status["outstanding"] == 1
    assert status["timed_out_total"] == 1

    with pytest.raises(runtime.ServiceCapacityError):
        await runtime.run_service_job("health", lambda: "too-early")

    allow_finish.set()
    for _ in range(100):
        if runtime.service_runtime_status()["health"]["outstanding"] == 0:
            break
        await asyncio.sleep(0)
    assert runtime.service_runtime_status()["health"]["outstanding"] == 0


def test_runtime_status_uses_immutable_pool_settings(monkeypatch) -> None:
    monkeypatch.setenv("V2_FEEDBACK_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_FEEDBACK_MAX_OUTSTANDING", "2")
    runtime.service_runtime_status()

    monkeypatch.setenv("V2_FEEDBACK_EXECUTOR_WORKERS", "8")

    assert runtime.service_runtime_status()["feedback"]["workers"] == 1
