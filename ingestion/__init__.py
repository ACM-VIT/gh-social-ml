from .pipeline import ingest_repository, ingest_batch, print_batch_summary
from .features import extract_tags, score_documentation, activity_score, trend_velocity, build_structured_summary
from .classification import classify_category
from .corpus import CorpusStore, dynamic_cluster_discovery
from .result import IngestionResult, NoveltyMatrix

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
    "NoveltyMatrix",
    
    # Legacy re-exports for backwards compatibility
    "RepositoryEmbeddingPipeline",
    "RepositoryEmbeddingConfig",
    "RepositoryEmbeddingResult",
    "QdrantRepositoryStore",
    "embed_repositories",
    "index_repositories",
]

def __getattr__(name: str):
    if name in {
        "RepositoryEmbeddingPipeline",
        "RepositoryEmbeddingConfig",
        "RepositoryEmbeddingResult",
        "QdrantRepositoryStore",
        "embed_repositories",
        "index_repositories",
    }:
        import embedding
        return getattr(embedding, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(__all__)
