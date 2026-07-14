import pytest
import math
import numpy as np
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from embedding.qdrant_store import repository_point_id
from feedback.event_handlers import (
    ACTION_WEIGHTS,
    FeedbackHandler,
    _dwell_alpha,
    normalize_feedback_action,
    shift_vector,
)
from feedback.producer import FeedbackProducer
from feedback.consumer import FeedbackConsumer

INTERNAL_HEADERS = {"x-internal-secret": "test-internal-secret"}


def test_shift_vector_math():
    """Verify the vector shifting formula: User' + alpha * Repo, normalized to 1."""
    user_vec = [1.0, 0.0]
    repo_vec = [0.0, 1.0]
    alpha = 0.5  # shift coefficient

    updated = shift_vector(user_vec, repo_vec, alpha)

    # Manual calculation:
    # updated_unnorm = [1.0, 0.5]
    # norm = sqrt(1.0^2 + 0.5^2) = sqrt(1.25) = 1.11803
    # normalized = [1.0 / 1.11803, 0.5 / 1.11803] = [0.894427, 0.447213]
    
    assert len(updated) == 2
    assert pytest.approx(updated[0], rel=1e-5) == 0.894427
    assert pytest.approx(updated[1], rel=1e-5) == 0.447213

    # Normalized check (L2 norm should be exactly 1)
    norm = np.linalg.norm(updated)
    assert pytest.approx(norm, rel=1e-5) == 1.0


def test_shift_vector_negative():
    """Verify shifting away works for negative alpha (e.g. skip/ignore)."""
    user_vec = [1.0, 0.0]
    repo_vec = [0.0, 1.0]
    alpha = -0.5

    updated = shift_vector(user_vec, repo_vec, alpha)
    # updated_unnorm = [1.0, -0.5]
    # normalized = [1.0 / 1.11803, -0.5 / 1.11803]
    
    assert pytest.approx(updated[0], rel=1e-5) == 0.894427
    assert pytest.approx(updated[1], rel=1e-5) == -0.447213
    assert pytest.approx(np.linalg.norm(updated), rel=1e-5) == 1.0


@patch("feedback.event_handlers.QdrantClient")
def test_handler_like_event_updates_only_qdrant(mock_qdrant_cls):
    """Feedback changes the ML profile vector without touching app Postgres."""
    mock_qdrant = MagicMock()
    mock_qdrant_cls.return_value = mock_qdrant

    # Mock user retrieval from Qdrant
    mock_user_point = MagicMock()
    # Unnamed 384-dimensional user vector
    user_vec = [0.1] * 384
    mock_user_point.vector = user_vec
    mock_user_point.payload = {"user_id": "test_user", "skills": ["Python"]}
    mock_qdrant.retrieve.side_effect = [
        [mock_user_point],  # first call: user profile
        [MagicMock(vector=[0.2] * 384)]  # second call: repository
    ]

    handler = FeedbackHandler(qdrant_url="http://localhost:6333")
    
    # Process like event
    success = handler.handle_feedback("test_user", "test-owner/test-repo", "like")
    
    assert success is True

    # Assert Qdrant upsert was called with updated vector
    assert mock_qdrant.upsert.call_count == 1
    _, kwargs = mock_qdrant.upsert.call_args
    points = kwargs["points"]
    assert len(points) == 1
    point = points[0]
    
    # Assert vector shifted: [0.1]*384 + 0.15 * [0.2]*384 = [0.13]*384, normalized
    expected_shifted = np.array([0.13] * 384)
    expected_shifted = (expected_shifted / np.linalg.norm(expected_shifted)).tolist()
    
    for val, exp in zip(point.vector, expected_shifted):
        assert pytest.approx(val, rel=1e-5) == exp


@patch("feedback.event_handlers.QdrantClient")
def test_handler_uses_backend_repo_uuid_as_qdrant_point_id(mock_qdrant_cls):
    """Backend UUID feedback resolves through ML-owned deterministic Qdrant IDs."""
    mock_qdrant = MagicMock()
    mock_qdrant_cls.return_value = mock_qdrant

    user_point = MagicMock(vector=[0.1] * 384, payload={"user_id": "user_1"})
    repo_point = MagicMock(vector={"repo_embedding": [0.2] * 384})
    mock_qdrant.retrieve.side_effect = [[user_point], [repo_point]]

    repo_id = "f62ccf78-9f47-48c4-b448-b1082f21b5ea"
    handler = FeedbackHandler(qdrant_url="http://localhost:6333")

    assert handler.handle_feedback("user_1", repo_id, "save") is True

    expected_repo_point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"github:{repo_id}"))
    assert repository_point_id(repo_id) == expected_repo_point_id
    assert mock_qdrant.retrieve.call_args_list[1].kwargs["ids"] == [expected_repo_point_id]


@patch("feedback.event_handlers.QdrantClient")
def test_handler_does_not_require_database_url(mock_qdrant_cls, monkeypatch):
    """The online feedback handler initializes with no app database config."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    handler = FeedbackHandler(qdrant_url="http://localhost:6333")

    assert handler.qdrant is mock_qdrant_cls.return_value
    assert not hasattr(handler, "db")


def test_api_feedback_submission():
    """Verify FastAPI handles request validation and returns HTTP 202."""
    client = TestClient(app)

    # Mock the producer to avoid hitting Redis or async Queue in testing
    with patch("api.main.producer") as mock_producer:
        mock_producer.submit_feedback = AsyncMock(return_value=True)

        # Test valid request (non-dwell action)
        response = client.post(
            "/api/v1/feedback",
            headers=INTERNAL_HEADERS,
            json={
                "user_id": "user_123",
                "repo_id": "facebook/react",
                "action": "like",
            },
        )
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"
        # New signature includes dwell_seconds=None for non-dwell actions
        mock_producer.submit_feedback.assert_called_once_with(
            user_id="user_123",
            repo_id="facebook/react",
            action="like",
            dwell_seconds=None,
        )


def test_api_dwell_feedback_submission():
    """Verify that a dwell event with dwell_seconds is accepted and threaded correctly."""
    client = TestClient(app)

    with patch("api.main.producer") as mock_producer:
        mock_producer.submit_feedback = AsyncMock(return_value=True)

        response = client.post(
            "/api/v1/feedback",
            headers=INTERNAL_HEADERS,
            json={
                "user_id": "user_123",
                "repo_id": "facebook/react",
                "action": "dwell",
                "dwell_seconds": 45.0,
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "accepted"
        assert data["data"]["dwell_seconds"] == 45.0
        mock_producer.submit_feedback.assert_called_once_with(
            user_id="user_123",
            repo_id="facebook/react",
            action="dwell",
            dwell_seconds=45.0,
        )


def test_api_dwell_missing_dwell_seconds():
    """Verify dwell action without dwell_seconds returns HTTP 422."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        headers=INTERNAL_HEADERS,
        json={
            "user_id": "user_123",
            "repo_id": "facebook/react",
            "action": "dwell",
        },
    )
    assert response.status_code == 422
    assert "dwell_seconds" in response.json()["detail"]


def test_dwell_alpha_boundary_cases():
    """Verify _dwell_alpha boundary and monotonicity conditions."""
    from config import MIN_DWELL_SECONDS, MAX_DWELL_SECONDS, DWELL_BASE_ALPHA

    # Below threshold -> None (ignored, not an error)
    assert _dwell_alpha(0.0) is None
    assert _dwell_alpha(MIN_DWELL_SECONDS - 0.1) is None

    # At threshold -> positive alpha
    at_min = _dwell_alpha(MIN_DWELL_SECONDS)
    assert at_min is not None
    assert at_min > 0

    # At max -> exactly DWELL_BASE_ALPHA (saturated)
    assert pytest.approx(_dwell_alpha(MAX_DWELL_SECONDS), rel=1e-9) == DWELL_BASE_ALPHA

    # Beyond max -> capped at DWELL_BASE_ALPHA
    assert pytest.approx(_dwell_alpha(MAX_DWELL_SECONDS * 10), rel=1e-9) == DWELL_BASE_ALPHA

    # All non-None values are in range (0, DWELL_BASE_ALPHA]
    for secs in [MIN_DWELL_SECONDS, 10.0, 30.0, 120.0, MAX_DWELL_SECONDS]:
        a = _dwell_alpha(secs)
        assert a is not None
        assert 0 < a <= DWELL_BASE_ALPHA + 1e-9

    # Monotonicity: longer dwell -> larger alpha
    alphas = [_dwell_alpha(s) for s in [MIN_DWELL_SECONDS, 10, 60, MAX_DWELL_SECONDS]]
    for a1, a2 in zip(alphas, alphas[1:]):
        assert a1 <= a2


def test_handler_dwell_below_threshold_is_noop():
    """Verify that a dwell shorter than MIN_DWELL_SECONDS does not touch Qdrant."""
    from config import MIN_DWELL_SECONDS

    with patch("feedback.event_handlers.QdrantClient") as mock_qdrant_cls:
        mock_qdrant_cls.return_value = MagicMock()

        handler = FeedbackHandler(qdrant_url="http://localhost:6333")

        # A dwell of 1 second (below MIN_DWELL_SECONDS=3) should be a clean no-op
        result = handler.handle_feedback(
            "user_x", "owner/repo", "dwell", dwell_seconds=1.0
        )

        assert result is True  # not an error — just silently ignored
        # No Qdrant operations should have been triggered
        handler.qdrant.retrieve.assert_not_called()
        handler.qdrant.upsert.assert_not_called()


@pytest.mark.parametrize(
    ("action", "expected_alpha"),
    [
        ("readme_open", 0.06),
        ("github_open", 0.10),
        ("like", 0.15),
        ("save", 0.20),
        ("share", 0.12),
        ("dislike", -0.15),
        ("unlike", -0.15),
        ("unsave", -0.20),
        ("undislike", 0.15),
    ],
)
def test_handler_supports_canonical_backend_actions(action, expected_alpha):
    with patch("feedback.event_handlers.QdrantClient"):
        handler = FeedbackHandler(qdrant_url="http://localhost:6333")
        with patch.object(handler, "update_user_embedding", return_value=True) as update:
            assert handler.handle_feedback("u1", "r1", action) is True

    update.assert_called_once_with("u1", "r1", expected_alpha)
    assert ACTION_WEIGHTS[action] == expected_alpha


def test_impression_is_an_explicit_neutral_noop():
    with patch("feedback.event_handlers.QdrantClient"):
        handler = FeedbackHandler(qdrant_url="http://localhost:6333")
        with patch.object(handler, "update_user_embedding") as update:
            assert handler.handle_feedback("u1", "r1", "impression") is True

    update.assert_not_called()


def test_legacy_feedback_aliases_are_normalized():
    assert normalize_feedback_action("click") == "readme_open"
    assert normalize_feedback_action("skip") == "impression"


def test_api_invalid_action():
    """Verify FastAPI rejects invalid actions with HTTP 400."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        headers=INTERNAL_HEADERS,
        json={
            "user_id": "user_123",
            "repo_id": "facebook/react",
            "action": "invalid_action",
        },
    )
    assert response.status_code == 400
    assert "Invalid action" in response.json()["detail"]


def test_api_feedback_rejects_missing_internal_secret():
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        json={
            "user_id": "user_123",
            "repo_id": "repo_123",
            "action": "like",
        },
    )

    assert response.status_code == 401


def test_api_feedback_batch_uses_one_producer_call():
    client = TestClient(app)

    with patch("api.main.producer") as mock_producer:
        mock_producer.submit_feedback_batch = AsyncMock(return_value=True)
        response = client.post(
            "/api/v1/feedback/batch",
            headers=INTERNAL_HEADERS,
            json={
                "events": [
                    {"user_id": "u1", "repo_id": "r1", "action": "like"},
                    {
                        "user_id": "u1",
                        "repo_id": "r2",
                        "action": "dwell",
                        "dwell_seconds": 12,
                    },
                ]
            },
        )

    assert response.status_code == 202
    assert response.json()["accepted_count"] == 2
    mock_producer.submit_feedback_batch.assert_awaited_once_with(
        [
            {"user_id": "u1", "repo_id": "r1", "action": "like"},
            {"user_id": "u1", "repo_id": "r2", "action": "dwell", "dwell_seconds": 12.0},
        ]
    )


def test_health_fails_when_durable_feedback_is_unavailable():
    client = TestClient(app)
    with patch("api.main.producer", None), patch("api.main.consumer", None):
        response = client.get("/api/v1/health")

    assert response.status_code == 503


def test_health_reports_ready_with_redis_backed_producer_and_consumer():
    client = TestClient(app)
    mock_producer = MagicMock(redis_client=MagicMock())
    mock_consumer = MagicMock(redis_client=MagicMock(), running=True)
    with patch("api.main.producer", mock_producer), patch("api.main.consumer", mock_consumer):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "consumer_running": True}


@pytest.mark.anyio
async def test_producer_batches_redis_stream_writes_in_one_pipeline():
    producer = FeedbackProducer()
    producer.redis_client = MagicMock()
    pipeline = producer.redis_client.pipeline.return_value

    success = await producer.submit_feedback_batch(
        [
            {"user_id": "u1", "repo_id": "r1", "action": "like"},
            {"user_id": "u1", "repo_id": "r2", "action": "skip"},
        ]
    )

    assert success is True
    producer.redis_client.pipeline.assert_called_once_with(transaction=False)
    assert pipeline.xadd.call_count == 2
    for call in pipeline.xadd.call_args_list:
        assert call.kwargs["maxlen"] == 1_000_000
        assert call.kwargs["approximate"] is True
    pipeline.execute.assert_called_once_with()


@pytest.mark.anyio
async def test_producer_rejects_when_durable_queue_is_unavailable():
    producer = FeedbackProducer(redis_url=None, allow_in_memory=False)
    producer.redis_client = None

    success = await producer.submit_feedback_batch(
        [{"user_id": "u1", "repo_id": "r1", "action": "like"}]
    )

    assert success is False


@pytest.mark.anyio
async def test_consumer_requires_redis_unless_dev_fallback_is_explicit():
    consumer = FeedbackConsumer(
        handler=MagicMock(),
        redis_url=None,
        allow_in_memory=False,
    )
    consumer.redis_client = None

    with pytest.raises(RuntimeError, match="REDIS_URL"):
        await consumer.start()


@pytest.mark.anyio
async def test_consumer_redis_loop_success():
    """Test that a message is successfully processed and acknowledged in Redis stream loop."""
    mock_handler = MagicMock()
    mock_redis = MagicMock()
    
    consumer = FeedbackConsumer(handler=mock_handler)
    consumer.redis_client = mock_redis
    consumer.running = True
    
    payload = {"user_id": "u1", "repo_id": "r1", "action": "like"}
    mock_redis.xreadgroup.return_value = [("feedback_stream", [("msg_1", payload)])]
    mock_redis.exists.return_value = False  # Not processed yet
    
    def mock_xack(*args, **kwargs):
        consumer.running = False
        return 1
    mock_redis.xack.side_effect = mock_xack
    
    await consumer._redis_consume_loop()
    
    # Verify handle_feedback was called with new dwell_seconds kwarg (None when not in payload)
    mock_handler.handle_feedback.assert_called_once_with(
        "u1", "r1", "like", dwell_seconds=None
    )
    # Verify key was set in redis
    mock_redis.set.assert_called_once_with("feedback:processed:msg_1", "1", ex=86400)
    # Verify xack was called
    mock_redis.xack.assert_called_once_with("feedback_stream", "feedback_group", "msg_1")


@pytest.mark.anyio
async def test_consumer_redis_loop_already_processed():
    """Test that if a message was already processed, it skips processing and just acknowledges."""
    mock_handler = MagicMock()
    mock_redis = MagicMock()
    
    consumer = FeedbackConsumer(handler=mock_handler)
    consumer.redis_client = mock_redis
    consumer.running = True
    
    payload = {"user_id": "u1", "repo_id": "r1", "action": "like"}
    mock_redis.xreadgroup.return_value = [("feedback_stream", [("msg_1", payload)])]
    mock_redis.exists.return_value = True  # Already processed!
    
    def mock_xack(*args, **kwargs):
        consumer.running = False
        return 1
    mock_redis.xack.side_effect = mock_xack
    
    await consumer._redis_consume_loop()
    
    # Verify handle_feedback was NOT called since it was already processed
    mock_handler.handle_feedback.assert_not_called()
    # Verify set was NOT called
    mock_redis.set.assert_not_called()
    # Verify xack was still called to clean up
    mock_redis.xack.assert_called_once_with("feedback_stream", "feedback_group", "msg_1")


@pytest.mark.anyio
async def test_consumer_redis_loop_retry_ack():
    """Test that if acknowledgement fails with a transient error, it retries and succeeds."""
    mock_handler = MagicMock()
    mock_redis = MagicMock()
    
    consumer = FeedbackConsumer(handler=mock_handler)
    consumer.redis_client = mock_redis
    consumer.running = True
    
    payload = {"user_id": "u1", "repo_id": "r1", "action": "like"}
    mock_redis.xreadgroup.return_value = [("feedback_stream", [("msg_1", payload)])]
    mock_redis.exists.return_value = False
    
    call_count = 0
    def mock_xack_with_failures(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Transient Redis connection error")
        consumer.running = False
        return 1
    mock_redis.xack.side_effect = mock_xack_with_failures
    
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        await consumer._redis_consume_loop()
        assert mock_sleep.call_count == 2
        
    mock_handler.handle_feedback.assert_called_once_with(
        "u1", "r1", "like", dwell_seconds=None
    )
    # Verify set was called
    mock_redis.set.assert_called_once_with("feedback:processed:msg_1", "1", ex=86400)
    # xack called 3 times total
    assert mock_redis.xack.call_count == 3


@pytest.mark.anyio
async def test_consumer_reclaims_stale_message_after_worker_crash():
    mock_handler = MagicMock(return_value=True)
    mock_redis = MagicMock()
    payload = {"user_id": "u1", "repo_id": "r1", "action": "like"}
    mock_redis.xautoclaim.return_value = ["0-0", [("stale_1", payload)], []]
    mock_redis.xreadgroup.return_value = []
    mock_redis.exists.return_value = False

    consumer = FeedbackConsumer(handler=mock_handler, allow_in_memory=False)
    consumer.redis_client = mock_redis
    consumer.running = True

    def stop_after_ack(*_args, **_kwargs):
        consumer.running = False
        return 1

    mock_redis.xack.side_effect = stop_after_ack
    await consumer._redis_consume_loop()

    mock_handler.handle_feedback.assert_called_once_with(
        "u1", "r1", "like", dwell_seconds=None
    )
    mock_redis.xack.assert_called_once_with(
        "feedback_stream", "feedback_group", "stale_1"
    )
