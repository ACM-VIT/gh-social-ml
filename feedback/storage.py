import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from database.connector import PostgreSQLConnector

logger = logging.getLogger("pipeline.feedback.storage")


def _normalize_user_uuid(user_id: str) -> str:
    try:
        return str(uuid.UUID(str(user_id)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid user_id UUID: {user_id}") from exc


@dataclass(frozen=True)
class FeedbackRecord:
    user_id: str
    repo_id: str
    interaction_type: str
    feedback_score: float
    updated_at: datetime | str | None = None


class FeedbackStore:
    """Persistent effective feedback, one row per user/repository pair."""

    def __init__(self, db_connector: PostgreSQLConnector | None = None) -> None:
        self.db = db_connector or PostgreSQLConnector()

    def init_schema(self) -> None:
        if not self.db.enabled:
            return
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                repo_id UUID NOT NULL REFERENCES repo(repo_id) ON DELETE CASCADE,
                interaction_type VARCHAR(50) NOT NULL,
                feedback_score DOUBLE PRECISION NOT NULL
                    CHECK (feedback_score >= -1.0 AND feedback_score <= 1.0),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, repo_id)
            );
            CREATE INDEX IF NOT EXISTS user_feedback_user_updated_idx
                ON user_feedback (user_id, updated_at DESC);
            """
        )
        conn.commit()

    def record(
        self,
        user_id: str,
        repo_id: str,
        interaction_type: str,
        feedback_score: float,
    ) -> FeedbackRecord | None:
        if not -1.0 <= feedback_score <= 1.0:
            raise ValueError("feedback_score must be in the range [-1.0, 1.0]")
        user_id = _normalize_user_uuid(user_id)
        if not self.db.enabled:
            return None

        self.init_schema()
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_feedback (
                user_id, repo_id, interaction_type, feedback_score, updated_at
            )
            SELECT %s::uuid, repo_id, %s, %s, CURRENT_TIMESTAMP
            FROM repo
            WHERE repo_id::text = %s OR full_name = %s
            ON CONFLICT (user_id, repo_id) DO UPDATE SET
                interaction_type = EXCLUDED.interaction_type,
                feedback_score = EXCLUDED.feedback_score,
                updated_at = CURRENT_TIMESTAMP
            RETURNING user_id::text, repo_id::text, interaction_type,
                      feedback_score, updated_at;
            """,
            (user_id, interaction_type, feedback_score, repo_id, repo_id),
        )
        row = cursor.fetchone()
        conn.commit()
        if not row:
            raise ValueError(f"Repository not found: {repo_id}")
        if not isinstance(row, (tuple, list)):
            # Some unit-test database doubles do not model RETURNING rows.
            return None
        return FeedbackRecord(*row)

    def delete(
        self,
        user_id: str,
        repo_id: str,
        *,
        interaction_type: str | None = None,
    ) -> bool:
        user_id = _normalize_user_uuid(user_id)
        if not self.db.enabled:
            return False
        self.init_schema()
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM user_feedback
            WHERE user_id = %s::uuid
              AND repo_id IN (
                  SELECT repo_id FROM repo
                  WHERE repo_id::text = %s OR full_name = %s
              )
              AND (%s IS NULL OR interaction_type = %s);
            """,
            (user_id, repo_id, repo_id, interaction_type, interaction_type),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted

    def list_for_user(self, user_id: str) -> list[FeedbackRecord]:
        user_id = _normalize_user_uuid(user_id)
        if not self.db.enabled:
            return []
        self.init_schema()
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT f.user_id::text, r.full_name, f.interaction_type,
                   f.feedback_score, f.updated_at
            FROM user_feedback f
            JOIN repo r ON r.repo_id = f.repo_id
            WHERE f.user_id = %s::uuid
            ORDER BY f.updated_at DESC;
            """,
            (user_id,),
        )
        return [FeedbackRecord(*row) for row in cursor.fetchall()]

    def scores_for_user(self, user_id: str) -> dict[str, float]:
        return {record.repo_id: record.feedback_score for record in self.list_for_user(user_id)}


def apply_feedback_scores(
    candidates: list[dict[str, Any]],
    feedback_by_repo: dict[str, float],
    *,
    max_adjustment: float = 2.5,
    dislike_filter_threshold: float = -0.9,
) -> list[dict[str, Any]]:
    """Apply a bounded explicit-feedback signal without replacing base ranking."""
    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        identity = str(candidate.get("full_name") or candidate.get("repo_id") or "")
        score = feedback_by_repo.get(identity, 0.0)
        if score <= dislike_filter_threshold:
            continue
        item = dict(candidate)
        adjustment = max(-max_adjustment, min(max_adjustment, score * max_adjustment))
        item["feedback_score"] = score
        item["feedback_adjustment"] = adjustment
        item["final_score"] = float(item.get("final_score") or 0.0) + adjustment
        adjusted.append(item)
    adjusted.sort(key=lambda item: item["final_score"], reverse=True)
    return adjusted
