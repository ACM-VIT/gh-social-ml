"""Qdrant-only candidate retrieval package for gh-social-ml."""

from .candidate_retriever import CandidateRetriever
from .config import (
    SEMANTIC_LIMIT,
    DISCOVERY_LIMIT,
    TRENDING_LIMIT,
    TOTAL_CANDIDATE_POOL,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_NAME,
    EMBEDDING_DIM,
)

__all__ = [
    "CandidateRetriever",
    "SEMANTIC_LIMIT",
    "DISCOVERY_LIMIT",
    "TRENDING_LIMIT",
    "TOTAL_CANDIDATE_POOL",
    "QDRANT_COLLECTION_NAME",
    "QDRANT_VECTOR_NAME",
    "EMBEDDING_DIM",
]
