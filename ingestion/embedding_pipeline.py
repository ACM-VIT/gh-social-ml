"""End-to-end repository embedding pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from .embeddings import SentenceTransformerEmbedder, aggregate_vectors, chunk_text
from .qdrant_store import QdrantRepositoryStore
from .repository_embedding import (
    RepositoryEmbeddingConfig,
    RepositoryEmbeddingResult,
    build_metadata_text,
    build_readme_text,
    build_topic_text,
    build_vector_payload,
    coerce_payload,
    combine_repo_tower,
    source_fingerprint,
)

logger = logging.getLogger(__name__)


class RepositoryEmbeddingPipeline:
    """Build README, metadata, topic, and final repository embeddings."""

    def __init__(
        self,
        *,
        config: RepositoryEmbeddingConfig | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
        store: QdrantRepositoryStore | None = None,
    ) -> None:
        self.config = config or RepositoryEmbeddingConfig()
        self.embedder = embedder or SentenceTransformerEmbedder(self.config.model_name)
        self.store = store

    def embed_repository(self, source: Any) -> RepositoryEmbeddingResult:
        """Embed one approved repository payload or EnrichmentResult."""
        repo = coerce_payload(source)
        repo_id = str(repo.get("id") or "unknown/repository")
        # The below text builders are for splitting source material into the
        # three approved towers before final weighted composition.
        readme_text = build_readme_text(source)
        metadata_text = build_metadata_text(repo)
        topic_text = build_topic_text(repo)

        readme_chunks = chunk_text(
            readme_text,
            max_chars=self.config.readme_chunk_chars,
            overlap_chars=self.config.readme_chunk_overlap_chars,
        )
        readme_vectors = self.embedder.embed_texts([chunk.text for chunk in readme_chunks])
        # The below aggregation is for making each repository contribute one
        # README vector regardless of README length.
        readme_embedding = aggregate_vectors(readme_vectors)
        metadata_embedding = self.embedder.embed_text(metadata_text)
        topic_embedding = self.embedder.embed_text(topic_text)
        final_embedding = combine_repo_tower(
            readme_embedding=readme_embedding,
            metadata_embedding=metadata_embedding,
            topic_embedding=topic_embedding,
            weights=self.config.tower_weights,
        )
        self._validate_embedding_dim(repo_id, final_embedding)

        source_hash = source_fingerprint(
            self.config.model_name,
            self.config.version,
            readme_text,
            metadata_text,
            topic_text,
        )
        payload = build_vector_payload(
            repo,
            final_embedding=final_embedding,
            readme_chunks=len(readme_chunks),
            source_hash=source_hash,
            config=self.config,
        )
        logger.info("Embedded repository %s with %d README chunks", repo_id, len(readme_chunks))
        return RepositoryEmbeddingResult(
            repo_id=repo_id,
            final_embedding=final_embedding,
            readme_embedding=readme_embedding,
            metadata_embedding=metadata_embedding,
            topic_embedding=topic_embedding,
            payload=payload,
            readme_chunks=len(readme_chunks),
            source_hash=source_hash,
            embedding_model=self.config.model_name,
            embedding_version=self.config.version,
        )

    def embed_batch(self, sources: Iterable[Any]) -> list[RepositoryEmbeddingResult]:
        """Embed multiple approved repositories."""
        return [self.embed_repository(source) for source in sources]

    def index_batch(self, sources: Iterable[Any]) -> list[RepositoryEmbeddingResult]:
        """Embed repositories and upsert them to Qdrant."""
        # The below lazy store initialization is for allowing embedding-only
        # callers to run without a Qdrant instance.
        if self.store is None:
            self.store = QdrantRepositoryStore(vector_size=self.config.embedding_dim)
        self.store.ensure_collection()
        results = self.embed_batch(sources)
        self.store.upsert(results)
        return results

    def search(self, query: str, *, limit: int = 5, exact: bool = True) -> list[dict]:
        """Embed a text query and search the configured Qdrant collection."""
        # The below query embedding is for searching with the same model/config
        # used during repository indexing.
        if self.store is None:
            self.store = QdrantRepositoryStore(vector_size=self.config.embedding_dim)
        self.store.validate_collection()
        query_vector = self.embedder.embed_text(query)
        self._validate_embedding_dim("query", query_vector)
        return self.store.search(query_vector, limit=limit, exact=exact)

    def _validate_embedding_dim(self, label: str, vector: list[float]) -> None:
        if len(vector) != self.config.embedding_dim:
            raise ValueError(
                f"Embedding for {label} has dimension {len(vector)}, "
                f"expected {self.config.embedding_dim}."
            )


def embed_repositories(
    sources: Iterable[Any],
    *,
    pipeline: RepositoryEmbeddingPipeline | None = None,
) -> list[RepositoryEmbeddingResult]:
    """Convenience function for batch repository embedding."""
    active_pipeline = pipeline or RepositoryEmbeddingPipeline()
    return active_pipeline.embed_batch(sources)


def index_repositories(
    sources: Iterable[Any],
    *,
    pipeline: RepositoryEmbeddingPipeline | None = None,
) -> list[RepositoryEmbeddingResult]:
    """Convenience function for embedding and storing repositories in Qdrant."""
    active_pipeline = pipeline or RepositoryEmbeddingPipeline()
    return active_pipeline.index_batch(sources)
