NOVELTY_THRESHOLD               = 0.35
TOP_K_COMPARISONS               = 5
EMBEDDING_MODEL                 = "all-MiniLM-L6-v2"
EMBEDDING_DIM                   = 384
COLLECTION_NAME                 = "osiris_research_corpus"
QDRANT_VECTOR_NAME              = "repo_embedding"
MAX_DOC_SCORE                   = 100
GATE_APPROVAL_THRESHOLD         = 0.60
MIN_STARS_PREFILTER             = 50
MIN_README_PREFILTER            = 200


DUPLICATE_SIMILARITY_THRESHOLD  = 0.94
WRAPPER_SIMILARITY_THRESHOLD    = 0.85

HYBRID_WEIGHTS = {
    "readme":       0.40,
    "description":  0.25,
    "topics":       0.20,
    "category":     0.10,
    "language":     0.05,
}

NOVELTY_WEIGHTS = {
    "semantic":   0.60,
    "tech_stack": 0.20,
    "category":   0.10,
    "activity":   0.10,
}
