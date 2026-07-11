import pytest
from unittest.mock import MagicMock

from feedback.interactions import get_interaction
from feedback.storage import FeedbackStore, apply_feedback_scores


USER_UUID = "123e4567-e89b-12d3-a456-426614174000"


def test_initial_feedback_weights_are_centralized_and_bounded():
    assert get_interaction("like").feedback_score == 1.0
    assert get_interaction("save").feedback_score == 0.8
    assert get_interaction("dislike").feedback_score == -1.0
    assert all(
        -1.0 <= get_interaction(action).feedback_score <= 1.0
        for action in ("like", "save", "dislike")
    )


def test_feedback_adjustment_is_bounded_and_resorts_candidates():
    candidates = [
        {"full_name": "org/a", "final_score": 10.0},
        {"full_name": "org/b", "final_score": 9.0},
    ]

    ranked = apply_feedback_scores(candidates, {"org/b": 0.8})

    assert ranked[0]["full_name"] == "org/b"
    assert ranked[0]["feedback_adjustment"] == pytest.approx(2.0)
    assert ranked[0]["final_score"] == pytest.approx(11.0)
    assert ranked[1]["final_score"] == pytest.approx(10.0)


def test_explicit_dislike_filters_exact_repository():
    candidates = [
        {"full_name": "org/liked", "final_score": 8.0},
        {"full_name": "org/disliked", "final_score": 12.0},
    ]

    ranked = apply_feedback_scores(candidates, {"org/disliked": -1.0})

    assert [item["full_name"] for item in ranked] == ["org/liked"]


def test_feedback_delete_can_target_only_one_interaction_type():
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 1
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.enabled = True
    mock_db._get_connection.return_value = mock_conn

    deleted = FeedbackStore(mock_db).delete(
        USER_UUID,
        "org/repo",
        interaction_type="like",
    )

    assert deleted is True
    sql, params = mock_cursor.execute.call_args_list[-1][0]
    assert "interaction_type = %s" in sql
    assert params == (USER_UUID, "org/repo", "org/repo", "like", "like")
