"""Canonical feature dimensions and ordering for the heavy ranker."""

FEATURE_SPEC_VERSION = "v2"
RANKER_MODEL_VERSION = "heavy-ranker-v2"

FEATURE_ORDER = [
    "doc_quality",
    "code_health",
    "readme_length",
    "star_count",
    "fork_count",
    "open_issues_count",
    "pushed_days_ago",
    "activity_score",
    "trend_velocity",
    "skill_match_score",
]

FEATURE_COUNT = len(FEATURE_ORDER)
EMBEDDING_DIM = 384
INPUT_DIM = (EMBEDDING_DIM * 2) + FEATURE_COUNT  # 778
