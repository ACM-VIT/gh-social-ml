import pytest
import math
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock, ANY

from fastapi.testclient import TestClient

from api.main import app
from feedback.event_handlers import shift_vector, FeedbackHandler, _dwell_alpha
from feedback.producer import FeedbackProducer
from feedback.consumer import FeedbackConsumer


USER_UUID = "123e4567-e89b-12d3-a456-426614174000"


class _MemoryFeedbackStore:
    def __init__(self) -> None:
        self.active: set[tuple[str, str, str]] = set()

    def record(self, user_id: str, repo_id: str, interaction_type: str, feedback_score: float, conn=None):
        key = (user_id, repo_id, interaction_type)
        if key in self.active:
            return None
        self.active.add(key)
        return object()

    def delete(self, user_id: str, repo_id: str, *, interaction_type: str | None = None, conn=None) -> bool:
        key = (user_id, repo_id, interaction_type or "")
        if key not in self.active:
            return False
        self.active.remove(key)
        return True


def _transition_handler() -> FeedbackHandler:
    mock_db = MagicMock()
    mock_db.enabled = False
    handler = FeedbackHandler(db_connector=mock_db)
    handler.store = _MemoryFeedbackStore()
    handler.update_postgres_metrics = MagicMock(return_value=True)
    handler.invalidate_user_feed_cache = MagicMock(return_value=True)
    handler.update_user_embedding = MagicMock(return_value=True)
    return handler


def _embedding_alphas(handler: FeedbackHandler) -> list[float]:
    return [call.args[2] for call in handler.update_user_embedding.call_args_list]


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


@patch("feedback.event_handlers.PostgreSQLConnector")
@patch("feedback.event_handlers.QdrantClient")
def test_handler_like_event(mock_qdrant_cls, mock_db_cls):
    """Test that handle_feedback runs updates in Postgres and shifts in Qdrant."""
    mock_db = MagicMock()
    mock_db.enabled = True
    mock_db_cls.return_value = mock_db
    
    mock_conn = MagicMock()
    mock_db.connect.return_value = mock_conn
    mock_db._get_connection.return_value = mock_conn

    mock_qdrant = MagicMock()
    mock_qdrant_cls.return_value = mock_qdrant

    # Mock user retrieval from Qdrant
    mock_user_point = MagicMock()
    # Unnamed 384-dimensional user vector
    user_vec = [0.1] * 384
    mock_user_point.vector = user_vec
    mock_user_point.payload = {"user_id": USER_UUID, "skills": ["Python"]}
    mock_qdrant.retrieve.side_effect = [
        [mock_user_point],  # first call: user profile
        [MagicMock(vector=[0.2] * 384)]  # second call: repository
    ]

    handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    
    # Process like event
    success = handler.handle_feedback(USER_UUID, "test-owner/test-repo", "like")
    
    assert success is True

    # Assert Postgres metric increment was executed
    assert mock_db.connect.call_count == 1  # 1 for transaction
    
    # Retrieve connection and cursor mock instances to check execution history
    mock_conn = mock_db.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    execute_calls = mock_cursor.execute.call_args_list
    
    assert len(execute_calls) >= 3
    
    # Verify increment SQL was called (index 1 because 0 is store.record)
    sql = execute_calls[1][0][0]
    assert "UPDATE Repo" in sql
    assert "likes_count" in sql

    # Verify cache invalidation SQL was called (index 2)
    sql_cache = execute_calls[2][0][0]
    assert "DELETE FROM user_recommendation_batches" in sql_cache

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


def test_api_feedback_submission():
    """Verify FastAPI handles request validation and returns HTTP 202."""
    client = TestClient(app)

    # Mock the producer to avoid hitting Redis or async Queue in testing
    with patch("api.main.producer") as mock_producer:
        mock_producer.submit_feedback = AsyncMock(return_value=True)

        # Test valid request (non-dwell action)
        response = client.post(
            "/api/v1/feedback",
            json={
                "user_id": USER_UUID,
                "repo_id": "facebook/react",
                "action": "like",
            },
        )
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"
        # New signature includes dwell_seconds=None for non-dwell actions
        mock_producer.submit_feedback.assert_called_once_with(
            user_id=USER_UUID,
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
            json={
                "user_id": USER_UUID,
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
            user_id=USER_UUID,
            repo_id="facebook/react",
            action="dwell",
            dwell_seconds=45.0,
        )


def test_api_dwell_missing_dwell_seconds():
    """Verify dwell action without dwell_seconds returns HTTP 422."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        json={
            "user_id": USER_UUID,
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
    """Verify that a dwell shorter than MIN_DWELL_SECONDS does not touch Qdrant or Postgres."""
    from config import MIN_DWELL_SECONDS

    with patch("feedback.event_handlers.PostgreSQLConnector") as mock_db_cls, \
         patch("feedback.event_handlers.QdrantClient") as mock_qdrant_cls:

        mock_db = MagicMock()
        mock_db.enabled = True
        mock_db_cls.return_value = mock_db
        mock_qdrant_cls.return_value = MagicMock()

        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")

        # A dwell of 1 second (below MIN_DWELL_SECONDS=3) should be a clean no-op
        result = handler.handle_feedback(
            "user_x", "owner/repo", "dwell", dwell_seconds=1.0
        )

        assert result is True  # not an error — just silently ignored
        # No Postgres or Qdrant operations should have been triggered
        mock_db.connect.assert_not_called()
        handler.qdrant.upsert.assert_not_called()


def test_api_invalid_action():
    """Verify FastAPI rejects invalid actions with HTTP 400."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        json={
            "user_id": USER_UUID,
            "repo_id": "facebook/react",
            "action": "invalid_action",
        },
    )
    assert response.status_code == 400
    assert "Invalid action" in response.json()["detail"]


def test_api_rejects_malformed_user_id():
    """Verify feedback rejects user IDs that cannot persist to UUID-backed tables."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        json={
            "user_id": "user_123",
            "repo_id": "facebook/react",
            "action": "like",
        },
    )

    assert response.status_code == 422


def test_clear_actions_only_delete_the_matching_positive_signal():
    """unlike/unsave must not remove an unrelated effective feedback row."""
    mock_db = MagicMock()
    mock_db.enabled = False
    with patch("feedback.event_handlers.QdrantClient"):
        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    handler.store = MagicMock()
    handler.store.delete.return_value = True
    handler.update_postgres_metrics = MagicMock(return_value=True)
    handler.invalidate_user_feed_cache = MagicMock(return_value=True)
    handler.update_user_embedding = MagicMock(return_value=True)

    assert handler.handle_feedback(USER_UUID, "facebook/react", "unlike") is True
    handler.store.delete.assert_called_once_with(
        USER_UUID,
        "facebook/react",
        interaction_type="like",
        conn=ANY
    )
    handler.store.record.assert_not_called()


def test_undislike_only_clears_dislike_signal():
    """Toggling thumb-down off must not be reported as unlike."""
    mock_db = MagicMock()
    mock_db.enabled = False
    with patch("feedback.event_handlers.QdrantClient"):
        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    handler.store = MagicMock()
    handler.store.delete.return_value = True
    handler.update_postgres_metrics = MagicMock(return_value=True)
    handler.invalidate_user_feed_cache = MagicMock(return_value=True)
    handler.update_user_embedding = MagicMock(return_value=True)

    assert handler.handle_feedback(USER_UUID, "facebook/react", "undislike") is True
    handler.store.delete.assert_called_once_with(
        USER_UUID,
        "facebook/react",
        interaction_type="dislike",
        conn=ANY
    )
    handler.store.record.assert_not_called()


def test_impression_does_not_create_effective_feedback():
    """A passive impression is neutral and must not overwrite explicit feedback."""
    mock_db = MagicMock()
    mock_db.enabled = False
    with patch("feedback.event_handlers.QdrantClient"):
        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    handler.store = MagicMock()
    handler.update_postgres_metrics = MagicMock(return_value=True)
    handler.invalidate_user_feed_cache = MagicMock(return_value=True)

    assert handler.handle_feedback(USER_UUID, "facebook/react", "impression") is True
    handler.store.record.assert_not_called()
    handler.store.delete.assert_not_called()


def test_like_and_save_are_independent_states_in_any_order():
    for first, second in (("like", "save"), ("save", "like")):
        handler = _transition_handler()

        assert handler.handle_feedback(USER_UUID, "facebook/react", first) is True
        assert handler.handle_feedback(USER_UUID, "facebook/react", second) is True

        assert handler.store.active == {
            (USER_UUID, "facebook/react", "like"),
            (USER_UUID, "facebook/react", "save"),
        }
        assert sorted(_embedding_alphas(handler)) == [0.15, 0.2]


def test_unsave_preserves_like_and_unlike_preserves_save():
    handler = _transition_handler()
    for action in ("like", "save", "unsave"):
        assert handler.handle_feedback(USER_UUID, "facebook/react", action) is True

    assert handler.store.active == {(USER_UUID, "facebook/react", "like")}
    assert _embedding_alphas(handler) == [0.15, 0.2, -0.2]

    handler = _transition_handler()
    for action in ("save", "like", "unlike"):
        assert handler.handle_feedback(USER_UUID, "facebook/react", action) is True

    assert handler.store.active == {(USER_UUID, "facebook/react", "save")}
    assert _embedding_alphas(handler) == [0.2, 0.15, -0.15]


def test_replaying_identical_state_events_does_not_shift_twice():
    handler = _transition_handler()

    for action in ("like", "like", "save", "save"):
        assert handler.handle_feedback(USER_UUID, "facebook/react", action) is True

    assert handler.store.active == {
        (USER_UUID, "facebook/react", "like"),
        (USER_UUID, "facebook/react", "save"),
    }
    assert _embedding_alphas(handler) == [0.15, 0.2]


def test_undo_actions_apply_inverse_delta_once():
    cases = [
        ("like", "unlike", [0.15, -0.15]),
        ("save", "unsave", [0.2, -0.2]),
        ("dislike", "undislike", [-0.15, 0.15]),
    ]

    for action, undo_action, expected_alphas in cases:
        handler = _transition_handler()

        assert handler.handle_feedback(USER_UUID, "facebook/react", action) is True
        assert handler.handle_feedback(USER_UUID, "facebook/react", undo_action) is True
        assert handler.handle_feedback(USER_UUID, "facebook/react", undo_action) is True

        assert handler.store.active == set()
        assert _embedding_alphas(handler) == expected_alphas


def test_like_save_dislike_state_transitions_do_not_clear_each_other():
    actions = ("like", "save", "dislike")
    for first in actions:
        for second in actions:
            if first == second:
                continue
            handler = _transition_handler()

            assert handler.handle_feedback(USER_UUID, "facebook/react", first) is True
            assert handler.handle_feedback(USER_UUID, "facebook/react", second) is True

            assert handler.store.active == {
                (USER_UUID, "facebook/react", first),
                (USER_UUID, "facebook/react", second),
            }


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


def test_handler_impression_does_not_invalidate_cache():
    """Verify that a neutral impression does not invalidate the user feed cache."""
    mock_db = MagicMock()
    mock_db.enabled = False
    with patch("feedback.event_handlers.QdrantClient"):
        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    handler.store = MagicMock()
    handler.update_postgres_metrics = MagicMock(return_value=True)
    handler.invalidate_user_feed_cache = MagicMock(return_value=True)
    handler.update_user_embedding = MagicMock(return_value=True)

    assert handler.handle_feedback(USER_UUID, "facebook/react", "impression") is True
    handler.invalidate_user_feed_cache.assert_not_called()


def test_readme_open_changes_state_and_invalidates_cache():
    """Verify that readme_open triggers a cache invalidation and state change."""
    mock_db = MagicMock()
    mock_db.enabled = False
    with patch("feedback.event_handlers.QdrantClient"):
        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    handler.store = MagicMock()
    handler.update_postgres_metrics = MagicMock(return_value=True)
    handler.invalidate_user_feed_cache = MagicMock(return_value=True)
    handler.update_user_embedding = MagicMock(return_value=True)

    assert handler.handle_feedback(USER_UUID, "facebook/react", "readme_open") is True
    handler.invalidate_user_feed_cache.assert_called_once_with(USER_UUID)
    handler.update_user_embedding.assert_called_once_with(USER_UUID, "facebook/react", 0.05)


def test_reversal_events_emit_correct_alpha():
    """Verify that unlike and undislike trigger negative alpha shifts."""
    mock_db = MagicMock()
    mock_db.enabled = False
    with patch("feedback.event_handlers.QdrantClient"):
        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    handler.store = MagicMock()
    handler.store.delete.return_value = True
    handler.update_postgres_metrics = MagicMock(return_value=True)
    handler.invalidate_user_feed_cache = MagicMock(return_value=True)
    handler.update_user_embedding = MagicMock(return_value=True)

    assert handler.handle_feedback(USER_UUID, "facebook/react", "unlike") is True
    handler.update_user_embedding.assert_called_with(USER_UUID, "facebook/react", -0.15)
    
    assert handler.handle_feedback(USER_UUID, "facebook/react", "undislike") is True
    handler.update_user_embedding.assert_called_with(USER_UUID, "facebook/react", 0.15)

    assert handler.handle_feedback(USER_UUID, "facebook/react", "unsave") is True


def test_unnormalized_preference_accumulator_undo_math():
    """Verify that applying an alpha and its inverse perfectly restores the vector using the accumulator."""
    mock_db = MagicMock()
    with patch("feedback.event_handlers.QdrantClient") as MockQdrant:
        qdrant = MockQdrant.return_value
        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
    
    # Original state
    user_vector = [0.1] * 384
    repo_vector = [0.05] * 384
    
    # Payload state
    user_payload = {"preference_accumulator": [0.1] * 384}
    
    mock_user_point = MagicMock()
    mock_user_point.vector = user_vector
    mock_user_point.payload = user_payload
    
    mock_repo_point = MagicMock()
    mock_repo_point.vector = repo_vector
    
    qdrant.retrieve.side_effect = [
        [mock_user_point],  # 1st call: user
        [mock_repo_point],  # 2nd call: repo
        [mock_user_point],  # 3rd call: user (for undo)
        [mock_repo_point],  # 4th call: repo (for undo)
    ]
    
    # Apply alpha = 0.15
    res = handler.update_user_embedding(USER_UUID, "facebook/react", 0.15)
    assert res is True
    
    # Get the resulting vector and new accumulator
    upsert_call_1 = qdrant.upsert.call_args_list[0]
    points_1 = upsert_call_1.kwargs["points"][0]
    new_accum_1 = points_1.payload["preference_accumulator"]
    
    # Update the mock for the second call
    mock_user_point.payload = {"preference_accumulator": new_accum_1}
    
    # Apply alpha = -0.15 (Undo)
    res = handler.update_user_embedding(USER_UUID, "facebook/react", -0.15)
    assert res is True
    
    upsert_call_2 = qdrant.upsert.call_args_list[1]
    points_2 = upsert_call_2.kwargs["points"][0]
    new_accum_2 = points_2.payload["preference_accumulator"]
    
    # Accumulator should be precisely equal to original
    for orig, restored in zip(user_vector, new_accum_2):
        assert math.isclose(orig, restored, abs_tol=1e-5)


def test_qdrant_failure_does_not_rollback_postgres_transaction():
    """Verify that an exception in Qdrant does not roll back Postgres, as it's already committed."""
    mock_db = MagicMock()
    mock_db.enabled = True
    mock_conn = MagicMock()
    mock_db._get_connection.return_value = mock_conn
    mock_db.connect.return_value = mock_conn
    
    with patch("feedback.event_handlers.QdrantClient") as MockQdrant:
        qdrant = MockQdrant.return_value
        handler = FeedbackHandler(db_connector=mock_db, qdrant_url="http://localhost:6333")
        
    handler.store = MagicMock()
    handler.store.record.return_value = object() # state changed
    handler.update_postgres_metrics = MagicMock(return_value=True)
    handler.invalidate_user_feed_cache = MagicMock(return_value=True)
    
    # Force Qdrant update to fail
    handler.update_user_embedding = MagicMock(return_value=False)
    
    # Should catch exception and return False to trigger retry
    res = handler.handle_feedback(USER_UUID, "facebook/react", "like")
    assert res is False
    
    # Transaction should NOT be rolled back, it's committed BEFORE Qdrant
    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_not_called()


def test_dislike_to_like_switch_emits_correct_sequence():
    """Simulate undislike + like sequence (switching from dislike→like).

    The backend emits two events: undislike first (to undo the negative signal),
    then like (to apply the positive signal). Each should produce the correct alpha.
    """
    handler = _transition_handler()

    # Step 1: Dislike first to set up state
    handler.store.active.add((USER_UUID, "facebook/react", "dislike"))
    assert handler.handle_feedback(USER_UUID, "facebook/react", "undislike") is True
    # undislike clears the dislike record and emits negative of dislike's alpha
    # dislike has embedding_alpha = -0.15, so undo = +0.15
    handler.update_user_embedding.assert_called_with(USER_UUID, "facebook/react", 0.15)

    # Step 2: Now like
    assert handler.handle_feedback(USER_UUID, "facebook/react", "like") is True
    # like has embedding_alpha = 0.15
    handler.update_user_embedding.assert_called_with(USER_UUID, "facebook/react", 0.15)

    alphas = _embedding_alphas(handler)
    assert alphas == [0.15, 0.15]  # undislike reversal + like forward


def test_like_to_dislike_switch_emits_correct_sequence():
    """Simulate unlike + dislike sequence (switching from like→dislike).

    The backend emits two events: unlike first (to undo the positive signal),
    then dislike (to apply the negative signal). Each should produce the correct alpha.
    """
    handler = _transition_handler()

    # Step 1: Like first to set up state
    assert handler.handle_feedback(USER_UUID, "facebook/react", "like") is True
    handler.update_user_embedding.assert_called_with(USER_UUID, "facebook/react", 0.15)

    # Step 2: Unlike (reversal of like)
    assert handler.handle_feedback(USER_UUID, "facebook/react", "unlike") is True
    handler.update_user_embedding.assert_called_with(USER_UUID, "facebook/react", -0.15)

    # Step 3: Dislike (forward negative signal)
    assert handler.handle_feedback(USER_UUID, "facebook/react", "dislike") is True
    # dislike has embedding_alpha = -0.15
    handler.update_user_embedding.assert_called_with(USER_UUID, "facebook/react", -0.15)

    alphas = _embedding_alphas(handler)
    assert alphas == [0.15, -0.15, -0.15]  # like + unlike reversal + dislike forward


def test_github_open_shifts_embedding_without_persisting():
    """github_open should apply alpha=0.07 without calling store.record (non-persisting)."""
    handler = _transition_handler()
    handler.store = MagicMock()

    assert handler.handle_feedback(USER_UUID, "facebook/react", "github_open") is True

    # Should NOT persist to database
    handler.store.record.assert_not_called()
    handler.store.delete.assert_not_called()

    # Should still shift embedding
    handler.update_user_embedding.assert_called_once_with(USER_UUID, "facebook/react", 0.07)

    # Should invalidate cache since alpha != 0
    handler.invalidate_user_feed_cache.assert_called_once_with(USER_UUID)


def test_share_shifts_embedding_without_persisting():
    """share should apply alpha=0.10 without calling store.record (non-persisting)."""
    handler = _transition_handler()
    handler.store = MagicMock()

    assert handler.handle_feedback(USER_UUID, "facebook/react", "share") is True

    # Should NOT persist to database
    handler.store.record.assert_not_called()
    handler.store.delete.assert_not_called()

    # Should still shift embedding
    handler.update_user_embedding.assert_called_once_with(USER_UUID, "facebook/react", 0.10)

    # Should invalidate cache since alpha != 0
    handler.invalidate_user_feed_cache.assert_called_once_with(USER_UUID)

def test_transient_persistence_error_raises_for_retry():
    """ConnectionError from store.record should propagate so the consumer
    does NOT acknowledge the message and can retry later."""
    handler = _transition_handler()
    handler.store = MagicMock()
    handler.store.record.side_effect = ConnectionError("database connection lost")

    with pytest.raises(ConnectionError, match="database connection lost"):
        handler.handle_feedback(USER_UUID, "facebook/react", "like")



