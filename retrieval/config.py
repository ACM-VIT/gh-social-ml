"""Configuration for Qdrant-only online candidate retrieval."""

from embedding.vector_contract import (
    REPOSITORY_COLLECTION_CONTRACT,
    REPOSITORY_DISCOVERY_CHANNELS,
)

SEMANTIC_LIMIT = 130
DISCOVERY_LIMIT = 20
# Backward-compatible alias for callers that still display the old label.
TRENDING_LIMIT = DISCOVERY_LIMIT
TOTAL_CANDIDATE_POOL = 150
OVERFETCH_MULTIPLIER = 1.5

QDRANT_COLLECTION_NAME = REPOSITORY_COLLECTION_CONTRACT.collection_name
QDRANT_VECTOR_NAME = REPOSITORY_COLLECTION_CONTRACT.vector_name
EMBEDDING_DIM = REPOSITORY_COLLECTION_CONTRACT.vector_size

# Candidate source label, Person 2 public channel, and its frozen score field.
# The payload field is looked up from the published vector contract rather than
# duplicated here. Quality remains a ranker feature, not a Person 3 retrieval
# channel under the agreed four-channel scope.
_DISCOVERY_SOURCES = (
    ("trending", "trend"),
    ("active", "activity"),
    ("popular", "popularity"),
    ("fresh", "freshness"),
)
DISCOVERY_CHANNELS = tuple(
    (source, channel, REPOSITORY_DISCOVERY_CHANNELS[channel])
    for source, channel in _DISCOVERY_SOURCES
)

# Used only when the Qdrant service/collection fails completely. It is not a
# fallback for an empty but successful query.
FALLBACK_REPOS = (
    "facebook/react",
    "vuejs/vue",
    "tensorflow/tensorflow",
    "torvalds/linux",
    "microsoft/vscode",
    "docker/compose",
    "kubernetes/kubernetes",
    "golang/go",
    "rust-lang/rust",
    "flutter/flutter",
    "django/django",
    "pallets/flask",
    "fastapi/fastapi",
    "pytorch/pytorch",
    "nodejs/node",
    "angular/angular",
    "sveltejs/svelte",
    "vercel/next.js",
    "denoland/deno",
    "supabase/supabase",
)

# Temporary compatibility exports for the current retrieval_engine.py, which
# is owned by the ranking/feed workstream. CandidateRetriever does not import
# or use these values; they can be removed when that owner replaces the old
# cold-start path with Qdrant discovery.
COLD_START_SKILL_MATCH_LIMIT = 60
COLD_START_TRENDING_LIMIT = 30
COLD_START_MIN_STARS = 0
COLD_START_SKILL_WEIGHT = 0.6
COLD_START_STARS_WEIGHT = 0.4
