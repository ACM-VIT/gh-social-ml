from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import threading
import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from feedback.consumer import FeedbackConsumer
from feedback.event_handlers import (
    ADJUSTMENTS_KEY,
    APPLIED_SIGNALS_KEY,
    LATENT_KEY,
    PROCESSED_KEY,
    FeedbackHandler,
    dwell_alpha,
    normalize_vector,
    shift_vector,
    vector_delta,
)
from feedback.interactions import INTERACTIONS, get_interaction
from feedback.producer import FeedbackProducer
from feedback.settings import FeedbackSettings

USER_ID = "123e4567-e89b-12d3-a456-426614174000"
REPO_ID = "123e4567-e89b-12d3-a456-426614174001"


@pytest.fixture
def settings(monkeypatch) -> FeedbackSettings:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("VECTOR_DIMENSION", "3")
    monkeypatch.setenv("FEEDBACK_ALLOW_MEMORY_FALLBACK", "true")
    monkeypatch.setenv("FEEDBACK_DWELL_MIN_SECONDS", "3")
    monkeypatch.setenv("FEEDBACK_DWELL_FULL_CREDIT_SECONDS", "30")
    return FeedbackSettings.from_env()


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeQdrant:
    def __init__(self) -> None:
        self.user = SimpleNamespace(
            id=USER_ID,
            vector=[1.0, 0.0, 0.0],
            payload={"user_id": USER_ID},
        )
        self.repo = SimpleNamespace(
            id=REPO_ID,
            vector={"repo_embedding": [0.0, 1.0, 0.0]},
            payload={"repo_id": REPO_ID},
        )
        self.upserts = 0
        self.repo_reads = 0

    def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
        if collection_name == "user_profiles":
            return [self.user] if self.user and self.user.id in ids else []
        self.repo_reads += 1
        return [self.repo] if self.repo and self.repo.id in ids else []

    def upsert(self, *, collection_name, points, wait):
        point = points[0]
        self.user = SimpleNamespace(
            id=str(point.id), vector=copy.deepcopy(point.vector), payload=copy.deepcopy(point.payload)
        )
        self.upserts += 1

    def get_collections(self):
        return SimpleNamespace(collections=[])


def test_action_registry_is_complete_and_immutable():
    assert set(INTERACTIONS) == {
        "impression", "dwell", "readme_open", "github_open", "share", "like",
        "unlike", "dislike", "undislike", "save", "unsave",
    }
    assert get_interaction(" LIKE ").embedding_alpha == 0.15
    assert get_interaction("dislike").embedding_alpha == -0.15
    assert get_interaction("save").embedding_alpha == 0.20
    assert get_interaction("impression").realtime is False
    assert get_interaction("readme_open").apply_once is True
    assert get_interaction("dwell").apply_once is False
    assert {
        action: definition.reference_score
        for action, definition in INTERACTIONS.items()
    } == {
        "impression": 0.0,
        "dwell": 0.0,
        "readme_open": 0.2,
        "github_open": 0.3,
        "share": 0.6,
        "like": 1.0,
        "unlike": 0.0,
        "dislike": -1.0,
        "undislike": 0.0,
        "save": 0.8,
        "unsave": 0.0,
    }
    assert get_interaction("like").feedback_score == 1.0
    with pytest.raises(TypeError):
        INTERACTIONS["new"] = get_interaction("like")
    with pytest.raises(ValueError):
        get_interaction("star")


def test_pdf_vector_formula_and_normalization():
    delta = vector_delta([1.0, 0.0], [0.0, 1.0], 0.5)
    assert delta == pytest.approx([-0.5, 0.5])
    shifted = shift_vector([1.0, 0.0], [0.0, 1.0], 0.5)
    assert shifted == pytest.approx([2 ** -0.5, 2 ** -0.5])
    assert np.linalg.norm(shifted) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [[0.0, 0.0], [1.0, float("nan")], [1.0, float("inf")]])
def test_invalid_vectors_are_rejected(bad):
    with pytest.raises(ValueError):
        normalize_vector(bad)


def test_dwell_policy_boundaries():
    assert dwell_alpha(2.99, minimum_seconds=3, full_credit_seconds=30) is None
    assert dwell_alpha(3, minimum_seconds=3, full_credit_seconds=30) == 0
    assert 0 < dwell_alpha(15, minimum_seconds=3, full_credit_seconds=30) < 0.15
    assert dwell_alpha(1000, minimum_seconds=3, full_credit_seconds=30) == 0.15
    with pytest.raises(ValueError):
        dwell_alpha(float("nan"))


def test_like_and_unlike_remove_the_exact_stored_delta(settings):
    qdrant = FakeQdrant()
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    original = list(qdrant.user.vector)
    assert handler.handle_feedback(USER_ID, REPO_ID, "like", event_id="like-1")
    stored = qdrant.user.payload[ADJUSTMENTS_KEY][REPO_ID]["reaction"]
    assert stored["action"] == "like"
    assert stored["delta"] == pytest.approx([-0.15, 0.15, 0.0])
    assert handler.handle_feedback(USER_ID, REPO_ID, "unlike", event_id="unlike-1")
    assert qdrant.user.payload[LATENT_KEY] == pytest.approx(original)
    assert REPO_ID not in qdrant.user.payload[ADJUSTMENTS_KEY]
    assert np.linalg.norm(qdrant.user.vector) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("forward", "reverse", "family"),
    [("dislike", "undislike", "reaction"), ("save", "unsave", "save")],
)
def test_other_reversible_actions_restore_latent(settings, forward, reverse, family):
    qdrant = FakeQdrant()
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    original = list(qdrant.user.vector)
    assert handler.handle_feedback(USER_ID, REPO_ID, forward, event_id=f"{forward}-1")
    assert qdrant.user.payload[ADJUSTMENTS_KEY][REPO_ID][family]["action"] == forward
    assert handler.handle_feedback(USER_ID, REPO_ID, reverse, event_id=f"{reverse}-1")
    assert qdrant.user.payload[LATENT_KEY] == pytest.approx(original)


def test_reaction_switch_and_save_are_independent(settings):
    qdrant = FakeQdrant()
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    handler.handle_feedback(USER_ID, REPO_ID, "like", event_id="1")
    handler.handle_feedback(USER_ID, REPO_ID, "save", event_id="2")
    handler.handle_feedback(USER_ID, REPO_ID, "dislike", event_id="3")
    state = qdrant.user.payload[ADJUSTMENTS_KEY][REPO_ID]
    assert state["reaction"]["action"] == "dislike"
    assert state["save"]["action"] == "save"


def test_duplicate_event_and_duplicate_state_do_not_shift_twice(settings):
    qdrant = FakeQdrant()
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    handler.handle_feedback(USER_ID, REPO_ID, "like", event_id="same")
    latent = list(qdrant.user.payload[LATENT_KEY])
    upserts = qdrant.upserts
    assert handler.handle_feedback(USER_ID, REPO_ID, "like", event_id="same")
    assert qdrant.upserts == upserts
    assert handler.handle_feedback(USER_ID, REPO_ID, "like", event_id="different")
    assert qdrant.user.payload[LATENT_KEY] == pytest.approx(latent)


@pytest.mark.parametrize("action", ["readme_open", "github_open", "share"])
def test_passive_action_updates_once_per_user_repository(settings, action):
    qdrant = FakeQdrant()
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    assert handler.handle_feedback(USER_ID, REPO_ID, action, event_id=f"{action}-1")
    latent = list(qdrant.user.payload[LATENT_KEY])
    repo_reads = qdrant.repo_reads

    assert qdrant.user.payload[APPLIED_SIGNALS_KEY][REPO_ID] == [action]
    assert handler.handle_feedback(USER_ID, REPO_ID, action, event_id=f"{action}-2")
    assert qdrant.user.payload[LATENT_KEY] == pytest.approx(latent)
    assert qdrant.repo_reads == repo_reads


def test_distinct_dwell_events_continue_learning(settings):
    qdrant = FakeQdrant()
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    assert handler.handle_feedback(
        USER_ID, REPO_ID, "dwell", event_id="dwell-1", dwell_seconds=30
    )
    first_latent = list(qdrant.user.payload[LATENT_KEY])
    assert handler.handle_feedback(
        USER_ID, REPO_ID, "dwell", event_id="dwell-2", dwell_seconds=30
    )
    assert qdrant.user.payload[LATENT_KEY] != pytest.approx(first_latent)
    assert qdrant.repo_reads == 2


def test_missing_user_or_repository_is_retryable(settings):
    qdrant = FakeQdrant()
    qdrant.repo = None
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    assert not handler.handle_feedback(USER_ID, REPO_ID, "like", event_id="1")
    qdrant = FakeQdrant()
    qdrant.user = None
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    assert not handler.handle_feedback(USER_ID, REPO_ID, "like", event_id="1")


@pytest.mark.parametrize(
    ("payload", "error_fragment"),
    [
        (["legacy-payload"], "user profile payload"),
        ({PROCESSED_KEY: "event-1"}, PROCESSED_KEY),
        ({ADJUSTMENTS_KEY: []}, ADJUSTMENTS_KEY),
        ({ADJUSTMENTS_KEY: {REPO_ID: []}}, ADJUSTMENTS_KEY),
        (
            {
                ADJUSTMENTS_KEY: {
                    REPO_ID: {
                        "reaction": {"action": "like", "delta": [0.1]}
                    }
                }
            },
            "delta",
        ),
        ({APPLIED_SIGNALS_KEY: []}, APPLIED_SIGNALS_KEY),
        ({APPLIED_SIGNALS_KEY: {REPO_ID: "readme_open"}}, APPLIED_SIGNALS_KEY),
    ],
)
def test_malformed_qdrant_feedback_payload_is_rejected_safely(
    settings, payload, error_fragment
):
    qdrant = FakeQdrant()
    qdrant.user.payload = payload
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)

    with pytest.raises(ValueError) as exc_info:
        handler.handle_feedback(USER_ID, REPO_ID, "like", event_id="event-1")

    assert error_fragment in str(exc_info.value)
    assert qdrant.upserts == 0
    assert qdrant.repo_reads == 0


@pytest.mark.anyio
async def test_producer_uses_bounded_xadd(settings):
    redis = MagicMock()
    redis.ping.return_value = True
    producer = FeedbackProducer(redis_client=redis, settings=settings)
    await producer.start()
    await producer.submit_feedback(
        user_id=USER_ID, repo_id=REPO_ID, action="like", event_id="evt",
        occurred_at=datetime.now(timezone.utc).isoformat(),
    )
    _, kwargs = redis.xadd.call_args
    assert kwargs == {"maxlen": settings.stream_maxlen, "approximate": True}


@pytest.mark.anyio
async def test_redis_is_required_without_explicit_development_fallback(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("FEEDBACK_ALLOW_MEMORY_FALLBACK", "false")
    producer = FeedbackProducer(settings=FeedbackSettings.from_env())
    with pytest.raises(RuntimeError, match="Redis is required"):
        await producer.start()


class FakeLock:
    def acquire(self):
        return True

    def release(self):
        return None


def _consumer_redis() -> MagicMock:
    redis = MagicMock()
    redis.exists.return_value = 0
    redis.lock.return_value = FakeLock()
    redis.set.return_value = True
    redis.incr.return_value = 1
    return redis


@pytest.mark.anyio
async def test_consumer_acks_only_after_success(settings):
    redis = _consumer_redis()
    handler = MagicMock()
    handler.handle_feedback.return_value = True
    consumer = FeedbackConsumer(handler=handler, redis_client=redis, settings=settings)
    payload = {
        "event_id": "evt", "user_id": USER_ID, "repo_id": REPO_ID,
        "action": "like", "occurred_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1",
    }
    await consumer._process_message("1-0", payload)
    handler.handle_feedback.assert_called_once()
    assert redis.set.call_args_list[0].args[0] == "feedback:processed:evt"
    redis.xack.assert_called_once_with(settings.stream_name, settings.consumer_group, "1-0")


@pytest.mark.anyio
async def test_retryable_failure_is_not_acked(settings):
    redis = _consumer_redis()
    handler = MagicMock()
    handler.handle_feedback.return_value = False
    consumer = FeedbackConsumer(handler=handler, redis_client=redis, settings=settings)
    payload = {
        "event_id": "evt", "user_id": USER_ID, "repo_id": REPO_ID,
        "action": "like", "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    await consumer._process_message("1-0", payload)
    redis.xack.assert_not_called()
    redis.incr.assert_called_once_with("feedback:attempts:evt")


@pytest.mark.anyio
async def test_pending_messages_are_reclaimed(settings):
    redis = _consumer_redis()
    redis.xautoclaim.return_value = ["0-0", [("1-0", {"event_id": "evt"})], []]
    consumer = FeedbackConsumer(handler=MagicMock(), redis_client=redis, settings=settings)
    assert await consumer._reclaim_stale() == [("1-0", {"event_id": "evt"})]


@pytest.mark.anyio
async def test_same_user_messages_are_serialized(settings):
    redis = _consumer_redis()
    underlying = threading.Lock()

    class SharedLock:
        def acquire(self):
            return underlying.acquire(timeout=1)

        def release(self):
            underlying.release()

    redis.lock.side_effect = lambda *args, **kwargs: SharedLock()
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    def handle(*args, **kwargs):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        return True

    handler = MagicMock()
    handler.handle_feedback.side_effect = handle
    consumer = FeedbackConsumer(handler=handler, redis_client=redis, settings=settings)
    base = {
        "user_id": USER_ID, "repo_id": REPO_ID, "action": "like",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    await asyncio.gather(
        consumer._process_message("1-0", {**base, "event_id": "one"}),
        consumer._process_message("2-0", {**base, "event_id": "two"}),
    )
    assert maximum_active == 1


def _event_payload(action="like", **updates):
    payload = {
        "event_id": str(uuid.uuid4()),
        "user_id": USER_ID,
        "repo_id": REPO_ID,
        "action": action,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
    }
    payload.update(updates)
    return payload


def test_api_auth_and_feedback_contract(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-secret")
    fake_producer = MagicMock()
    fake_producer.submit_feedback = AsyncMock(return_value=True)
    with patch.object(api_main, "producer", fake_producer):
        client = TestClient(api_main.app)
        assert client.post("/api/v1/feedback", json=_event_payload()).status_code == 401
        response = client.post(
            "/api/v1/feedback", json=_event_payload(),
            headers={"x-internal-secret": "test-secret"},
        )
        assert response.status_code == 202
        assert response.json()["data"]["queued_for_realtime_ml"] is True


def test_every_non_health_route_is_guarded(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-secret")
    client = TestClient(api_main.app)
    for route in api_main.app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set())
        if not path or path == "/api/v1/health":
            continue
        method = "POST" if "POST" in methods else "GET"
        response = client.request(method, path)
        assert response.status_code == 401, path


def test_impression_is_not_sent_to_realtime_stream(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-secret")
    fake_producer = MagicMock()
    fake_producer.submit_feedback = AsyncMock(return_value=True)
    with patch.object(api_main, "producer", fake_producer):
        response = TestClient(api_main.app).post(
            "/api/v1/feedback", json=_event_payload("impression"),
            headers={"x-internal-secret": "test-secret"},
        )
    assert response.status_code == 202
    assert response.json()["data"]["queued_for_realtime_ml"] is False
    fake_producer.submit_feedback.assert_not_called()


def test_health_is_public_and_checks_real_dependencies(monkeypatch):
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    fake_producer = SimpleNamespace(redis_client=MagicMock())
    fake_producer.redis_client.ping.return_value = True
    fake_consumer = SimpleNamespace(healthy=True)
    fake_handler = MagicMock()
    fake_handler.healthy.return_value = True
    with patch.object(api_main, "producer", fake_producer), patch.object(
        api_main, "consumer", fake_consumer
    ), patch.object(api_main, "feedback_handler", fake_handler):
        response = TestClient(api_main.app).get("/api/v1/health")
    assert response.status_code == 200
    assert all(response.json()["checks"].values())


@pytest.mark.anyio
async def test_lifespan_starts_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("FEEDBACK_ALLOW_MEMORY_FALLBACK", "true")
    fake_producer = SimpleNamespace(start=AsyncMock(), redis_client=None)
    fake_consumer = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    fake_handler = SimpleNamespace(healthy=MagicMock(return_value=True), qdrant=SimpleNamespace())
    with patch.object(
        api_main, "_build_feedback_runtime",
        return_value=(fake_producer, fake_consumer, fake_handler),
    ):
        async with api_main.lifespan(api_main.app):
            assert "DATABASE_URL" not in __import__("os").environ
            fake_consumer.start.assert_awaited_once()
    fake_consumer.stop.assert_awaited_once()
