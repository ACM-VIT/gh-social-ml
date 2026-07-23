"""Canonical, versioned feedback vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class InteractionDefinition:
    reference_score: float
    embedding_alpha: float
    realtime: bool = True
    reversal_of: str | None = None
    state_family: str | None = None
    apply_once: bool = False

    @property
    def feedback_score(self) -> float:
        """Compatibility name used by API clients for the reference score."""
        return self.reference_score

    @property
    def persists_feedback(self) -> bool:
        """Whether the backend treats this as active product state.

        The online ML worker never persists product feedback; this property is
        retained only for callers transitioning to the V2 worker-owned state.
        """
        return self.state_family is not None and self.reversal_of is None

    @property
    def clears_interaction_type(self) -> str | None:
        return self.reversal_of

    @property
    def metric_column(self) -> None:
        """Online ML never owns product counters."""
        return None


_INTERACTIONS = {
    "impression": InteractionDefinition(0.0, 0.0, realtime=False),
    "dwell": InteractionDefinition(0.0, 0.0),
    "readme_open": InteractionDefinition(0.2, 0.05, apply_once=True),
    "github_open": InteractionDefinition(0.3, 0.07, apply_once=True),
    "share": InteractionDefinition(0.6, 0.10, apply_once=True),
    "like": InteractionDefinition(1.0, 0.15, state_family="reaction"),
    "unlike": InteractionDefinition(0.0, 0.0, reversal_of="like", state_family="reaction"),
    "dislike": InteractionDefinition(-1.0, -0.15, state_family="reaction"),
    "undislike": InteractionDefinition(
        0.0, 0.0, reversal_of="dislike", state_family="reaction"
    ),
    "save": InteractionDefinition(0.8, 0.20, state_family="save"),
    "unsave": InteractionDefinition(0.0, 0.0, reversal_of="save", state_family="save"),
}

INTERACTIONS: Mapping[str, InteractionDefinition] = MappingProxyType(_INTERACTIONS)


def normalize_interaction(interaction_type: str) -> str:
    if not isinstance(interaction_type, str):
        raise ValueError("interaction type must be a string")
    return interaction_type.strip().lower()


def get_interaction(interaction_type: str) -> InteractionDefinition:
    normalized = normalize_interaction(interaction_type)
    try:
        return INTERACTIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported interaction type: {interaction_type}") from exc
