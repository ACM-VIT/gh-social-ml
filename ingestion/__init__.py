from .pipeline import ingest_repository, ingest_batch, print_batch_summary
from .features import extract_tags, score_documentation, activity_score, trend_velocity, build_structured_summary
from .classification import classify_category
from .corpus import CorpusStore, dynamic_cluster_discovery
from .result import IngestionResult

# The below exports are for the public embedding/Qdrant API used by main.py and
# validation scripts.
from .embedding_pipeline import RepositoryEmbeddingPipeline, embed_repositories, index_repositories
from .repository_embedding import RepositoryEmbeddingConfig, RepositoryEmbeddingResult
from .qdrant_store import QdrantRepositoryStore

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
    "CorpusStore",
    "dynamic_cluster_discovery",
    "IngestionResult",
    "RepositoryEmbeddingPipeline",
    "RepositoryEmbeddingConfig",
    "RepositoryEmbeddingResult",
    "QdrantRepositoryStore",
    "embed_repositories",
    "index_repositories",
]

