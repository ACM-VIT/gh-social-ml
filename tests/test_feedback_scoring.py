import pytest
from unittest.mock import MagicMock

from feedback.interactions import get_interaction
from feedback.storage import FeedbackStore, apply_feedback_scores


USER_UUID = "123e4567-e89b-12d3-a456-426614174000"


def test_initial_feedback_weights_are_centralized_and_bounded():
    assert get_interaction("impression").feedback_score == 0.0
    assert get_interaction("impression").persists_feedback is False
    assert get_interaction("readme_open").feedback_score == 0.2
    assert get_interaction("github_open").feedback_score == 0.3
    assert get_interaction("like").feedback_score == 1.0
    assert get_interaction("save").feedback_score == 0.8
    assert get_interaction("share").feedback_score == 0.6
    assert get_interaction("dislike").feedback_score == -1.0
    assert get_interaction("undislike").clears_interaction_type == "dislike"
    assert all(
        -1.0 <= get_interaction(action).feedback_score <= 1.0
        for action in (
            "impression",
            "readme_open",
            "github_open",
            "like",
            "save",
            "share",
            "dislike",
            "undislike",
            "unlike",
            "unsave",
        )
    )


def test_positive_feedback_exact_match_is_treated_as_consumed():
    candidates = [
        {"full_name": "org/a", "final_score": 10.0},
        {"full_name": "org/b", "final_score": 9.0},
    ]

    ranked = apply_feedback_scores(candidates, {"org/b": 0.8})

    assert [item["full_name"] for item in ranked] == ["org/a"]


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


def test_scores_for_user_aggregates_independent_states_with_cap():
    store = FeedbackStore(MagicMock(enabled=False))
    store.list_for_user = MagicMock(return_value=[
        MagicMock(repo_id="org/repo", feedback_score=1.0),
        MagicMock(repo_id="org/repo", feedback_score=0.8),
        MagicMock(repo_id="org/negative", feedback_score=-1.0),
    ])

    assert store.scores_for_user(USER_UUID) == {
        "org/repo": 1.0,
        "org/negative": -1.0,
    }
