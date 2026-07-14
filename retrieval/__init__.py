"""Candidate retrieval package for gh-social-ml.

Provides multi-channel L1 candidate retrieval using Qdrant semantic search
and Qdrant-native discovery queries.
"""

from .candidate_retriever import CandidateRetriever
from .config import (
    SEMANTIC_LIMIT,
    TRENDING_LIMIT,
    TOTAL_CANDIDATE_POOL,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_NAME,
    EMBEDDING_DIM,
)

__all__ = [
    "CandidateRetriever",
    "SEMANTIC_LIMIT",
    "TRENDING_LIMIT",
    "TOTAL_CANDIDATE_POOL",
    "QDRANT_COLLECTION_NAME",
    "QDRANT_VECTOR_NAME",
    "EMBEDDING_DIM",
]
