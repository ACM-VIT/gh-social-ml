"""Retrieval configuration for the L1 candidate retrieval pipeline.

All limits and constants used by the CandidateRetriever are defined here
so they can be tuned without modifying retrieval logic.
"""

# ── Channel Retrieval Limits ─────────────────────────────────────────────────
# These define how many candidates each channel fetches before merge.

SEMANTIC_LIMIT      = 130   # Max repos from Qdrant exact cosine search
TRENDING_LIMIT      = 20    # Max repos from trending velocity ranking
TOTAL_CANDIDATE_POOL = 150  # Final pool size sent to the ranking model

# ── Over-Fetch Buffer ────────────────────────────────────────────────────────
# We query more than the limit to absorb deduplication losses without looping.
# e.g. for SEMANTIC_LIMIT=130, we query 130 * 1.5 ≈ 195 items from Qdrant,
# then slice after dedup.

OVERFETCH_MULTIPLIER = 1.5

# ── Qdrant Configuration ────────────────────────────────────────────────────
# Must match the collection created by the embedding pipeline.

QDRANT_COLLECTION_NAME = "osiris_research_corpus"
QDRANT_VECTOR_NAME     = "repo_embedding"
EMBEDDING_DIM          = 384

# ── Timeout & Safety ────────────────────────────────────────────────────────

QDRANT_TIMEOUT_SECONDS = 10     # Max wait for a single Qdrant query
DB_QUERY_TIMEOUT_SECONDS = 10   # Max wait for a single PostgreSQL query

# ── Fallback Repositories ───────────────────────────────────────────────────
# Hardcoded stable repo full_names returned when both Qdrant and PostgreSQL
# are unreachable (catastrophic failure). These must exist in the corpus.

FALLBACK_REPOS = [
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
]

# ── Cold Start Configuration ────────────────────────────────────────────────
COLD_START_SKILL_MATCH_LIMIT = 60   # Max repos from skill-matched Postgres query
COLD_START_TRENDING_LIMIT    = 30   # Max supplemental trending repos for diversity
COLD_START_MIN_STARS         = 0    # Quality floor for cold-start repos (lowered for local dev)
COLD_START_SKILL_WEIGHT      = 0.6  # Weight for skill/language match in scoring
COLD_START_STARS_WEIGHT      = 0.4  # Weight for normalised star count in scoring
