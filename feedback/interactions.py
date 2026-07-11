from dataclasses import dataclass


@dataclass(frozen=True)
class InteractionDefinition:
    feedback_score: float
    embedding_alpha: float
    metric_column: str | None = None
    clears_interaction_type: str | None = None


# Canonical interaction configuration. New interaction types should be added here.
INTERACTIONS: dict[str, InteractionDefinition] = {
    "like": InteractionDefinition(1.0, 0.15, "likes_count"),
    "save": InteractionDefinition(0.8, 0.20, "saves_count"),
    "dislike": InteractionDefinition(-1.0, -0.15),
    "unlike": InteractionDefinition(0.0, 0.0, clears_interaction_type="like"),
    "unsave": InteractionDefinition(0.0, 0.0, clears_interaction_type="save"),
    # Existing implicit signals remain supported with deliberately lower weight.
    "click": InteractionDefinition(0.2, 0.05, "views_count"),
    "skip": InteractionDefinition(-0.2, -0.05),
}


def normalize_interaction(interaction_type: str) -> str:
    return interaction_type.strip().lower()


def get_interaction(interaction_type: str) -> InteractionDefinition:
    normalized = normalize_interaction(interaction_type)
    try:
        return INTERACTIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported interaction type: {interaction_type}") from exc

