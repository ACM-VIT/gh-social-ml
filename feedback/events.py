"""Versioned event exchanged between the API and feedback worker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping

from .interactions import get_interaction, normalize_interaction


@dataclass(frozen=True)
class FeedbackEvent:
    event_id: str
    user_id: str
    repo_id: str
    action: str
    occurred_at: str
    schema_version: int = 2
    dwell_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.user_id or not self.repo_id:
            raise ValueError("event_id, user_id, and repo_id are required")
        try:
            occurred = datetime.fromisoformat(self.occurred_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("occurred_at must be an RFC3339 timestamp") from exc
        if occurred.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        action = normalize_interaction(self.action)
        get_interaction(action)
        object.__setattr__(self, "action", action)
        if self.schema_version != 2:
            raise ValueError("unsupported feedback schema_version")
        if action == "dwell":
            if self.dwell_seconds is None:
                raise ValueError("dwell_seconds is required for dwell")
            value = float(self.dwell_seconds)
            if value < 0 or not math.isfinite(value):
                raise ValueError("dwell_seconds must be finite and non-negative")
            object.__setattr__(self, "dwell_seconds", value)
        elif self.dwell_seconds is not None:
            raise ValueError("dwell_seconds is only valid for dwell")

    def as_redis_fields(self) -> dict[str, str]:
        fields = {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "repo_id": self.repo_id,
            "action": self.action,
            "occurred_at": self.occurred_at,
            "schema_version": str(self.schema_version),
        }
        if self.dwell_seconds is not None:
            fields["dwell_seconds"] = str(self.dwell_seconds)
        return fields

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FeedbackEvent":
        return cls(
            event_id=str(payload.get("event_id", "")),
            user_id=str(payload.get("user_id", "")),
            repo_id=str(payload.get("repo_id", "")),
            action=str(payload.get("action", "")),
            occurred_at=str(payload.get("occurred_at", "")),
            schema_version=int(payload.get("schema_version", 2)),
            dwell_seconds=(
                float(payload["dwell_seconds"])
                if payload.get("dwell_seconds") not in (None, "") else None
            ),
        )

    @classmethod
    def now(
        cls, *, event_id: str, user_id: str, repo_id: str, action: str,
        dwell_seconds: float | None = None,
    ) -> "FeedbackEvent":
        return cls(
            event_id=event_id,
            user_id=user_id,
            repo_id=repo_id,
            action=action,
            dwell_seconds=dwell_seconds,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
