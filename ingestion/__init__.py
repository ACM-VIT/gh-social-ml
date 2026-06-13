from .pipeline import ingest_repository, ingest_batch, print_batch_summary
from .features import extract_tags, score_documentation, activity_score, trend_velocity, build_structured_summary
from .classification import classify_category
from .embeddings import generate_hybrid_embedding
from .novelty import compute_multi_dimensional_novelty, calculate_quadrant
from .corpus import CorpusStore, dynamic_cluster_discovery
from .result import IngestionResult
from .vector_store import (
    _resolve_qdrant_client,
    _verify_qdrant_collection,
    _qdrant_count_points,
    _query_qdrant_neighbors,
    _query_internal_neighbors,
    _index_qdrant_point,
    _qdrant_uses_named_vectors,
    _qdrant_query_vector,
    _QDRANT_OK,
    NamedVector
)

__all__ = [
    "ingest_repository",
    "ingest_batch",
    "print_batch_summary",
    "extract_tags",
    "score_documentation",
    "activity_score",
    "trend_velocity",
    "build_structured_summary",
    "classify_category",
    "generate_hybrid_embedding",
    "compute_multi_dimensional_novelty",
    "calculate_quadrant",
    "CorpusStore",
    "dynamic_cluster_discovery",
    "IngestionResult",
    "_resolve_qdrant_client",
    "_verify_qdrant_collection",
    "_qdrant_count_points",
    "_query_qdrant_neighbors",
    "_query_internal_neighbors",
    "_index_qdrant_point",
    "_qdrant_uses_named_vectors",
    "_qdrant_query_vector",
    "_QDRANT_OK",
    "NamedVector"
]
