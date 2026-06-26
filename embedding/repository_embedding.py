"""Repository embedding data models and repo tower composition."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from config import (
    README_CHUNK_CHARS,
    README_CHUNK_OVERLAP_CHARS,
    REPO_TOWER_WEIGHTS,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
)
from .embeddings import Vector, aggregate_vectors
from ingestion.features import (
    activity_score,
    extract_tags,
    score_code_health,
    score_documentation,
    trend_velocity,
)
from ingestion.classification import classify_category


SUPPORTED_REPOSITORY_EMBEDDING_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


@dataclass(slots=True)
class RepositoryEmbeddingConfig:
    """Configuration for repository embedding generation."""

    model_name: str = REPOSITORY_EMBEDDING_MODEL
    embedding_dim: int = REPOSITORY_EMBEDDING_DIM
    version: str = REPOSITORY_EMBEDDING_VERSION
    readme_chunk_chars: int = README_CHUNK_CHARS
    readme_chunk_overlap_chars: int = README_CHUNK_OVERLAP_CHARS
    tower_weights: dict[str, float] = field(default_factory=lambda: dict(REPO_TOWER_WEIGHTS))

    def __post_init__(self) -> None:
        expected_dim = SUPPORTED_REPOSITORY_EMBEDDING_DIMS.get(self.model_name)
        if expected_dim is None:
            supported = ", ".join(sorted(SUPPORTED_REPOSITORY_EMBEDDING_DIMS))
            raise ValueError(
                f"Unsupported repository embedding model {self.model_name!r}. "
                "The current Qdrant schema requires a known repository embedding dimension. "
                f"Supported models: {supported}."
            )
        if int(self.embedding_dim) != expected_dim:
            raise ValueError(
                f"Repository embedding model {self.model_name!r} produces {expected_dim} dimensions, "
                f"but embedding_dim is configured as {self.embedding_dim}."
            )


@dataclass(slots=True)
class RepositoryEmbeddingResult:
    """Embeddings and Qdrant payload generated for one repository."""

    repo_id: str
    final_embedding: Vector
    readme_embedding: Vector
    metadata_embedding: Vector
    topic_embedding: Vector
    payload: dict[str, Any]
    readme_chunks: int
    source_hash: str
    embedding_model: str
    embedding_version: str


def build_readme_text(source: Any) -> str:
    """Return cleaned README text from an EnrichmentResult or payload dict."""
    # The below preference is for using the cleaned README from acquisition when
    # available; staged JSON payloads fall back to extracted paragraphs.
    readme = getattr(source, "readme", None)
    clean_text = getattr(readme, "clean_text", None)
    if clean_text:
        return str(clean_text)

    payload = coerce_payload(source)
    paragraphs = payload.get("extracted_paragraphs") or []
    if isinstance(paragraphs, list):
        return "\n\n".join(str(item) for item in paragraphs if item)
    return str(paragraphs or "")


def coerce_payload(source: Any) -> dict[str, Any]:
    """Extract the repository payload from supported pipeline inputs."""
    if isinstance(source, Mapping):
        return dict(source)
    payload = getattr(source, "payload", None)
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError("repository source must be a payload dict or an object with a payload dict")


def build_metadata_text(repo: Mapping[str, Any]) -> str:
    """Project structured repository metadata into stable embedding text."""
    lines = [
        f"Repository: {repo.get('full_name') or repo.get('id', '')}",
        f"Description: {repo.get('description') or ''}",
        f"Primary language: {repo.get('primary_language') or 'Unknown'}",
        f"Stars: {int(repo.get('star_count') or 0)}",
        f"Forks: {int(repo.get('fork_count') or 0)}",
        f"Open issues: {int(repo.get('open_issues_count') or 0)}",
        f"Pushed days ago: {int(repo.get('pushed_days_ago') or 999)}",
        f"Star deltas: 3d={int(repo.get('delta_3d') or 0)}, 7d={int(repo.get('delta_7d') or 0)}, 30d={int(repo.get('delta_30d') or 0)}",
        f"Contributor signal: mentionable_users_count={int(repo.get('mentionable_users_count') or 0)}",
        f"Discovery category: {repo.get('discovery_category') or ''}",
        f"Discovery band: {repo.get('discovery_band') or ''}",
    ]
    return "\n".join(lines)


def build_topic_text(repo: Mapping[str, Any]) -> str:
    """Project topics and languages into embedding text."""
    topics = ", ".join(str(item) for item in repo.get("topics", []) if item)
    languages = ", ".join(str(item) for item in repo.get("languages", []) if item)
    return f"GitHub topics: {topics}\nLanguages: {languages}".strip()


def combine_repo_tower(
    *,
    readme_embedding: Sequence[float],
    metadata_embedding: Sequence[float],
    topic_embedding: Sequence[float],
    weights: Mapping[str, float],
) -> Vector:
    """Combine component embeddings with a simple weighted normalized tower."""
    # The below weighted average is for the first repo tower implementation; it
    # keeps the interface stable while leaving room for a learned tower later.
    vectors: list[Sequence[float]] = []
    vector_weights: list[float] = []
    for key, vector in (
        ("readme", readme_embedding),
        ("metadata", metadata_embedding),
        ("topics", topic_embedding),
    ):
        if vector:
            vectors.append(vector)
            vector_weights.append(float(weights.get(key, 0.0)))
    return aggregate_vectors(vectors, weights=vector_weights)


def build_vector_payload(
    repo: Mapping[str, Any],
    *,
    final_embedding: Sequence[float],
    readme_chunks: int,
    source_hash: str,
    config: RepositoryEmbeddingConfig,
) -> dict[str, Any]:
    """Build the Qdrant payload schema for one repository vector."""
    # The below payload fields are for Qdrant filtering and inspection without
    # fetching the original repository object again.
    repo_id = str(repo.get("id") or "unknown/repository")
    tags = extract_tags(repo_id, repo.get("extracted_paragraphs", []))
    category = classify_category(dict(repo), tags)
    documentation = score_documentation(dict(repo))

    return {
        "repo_id": repo_id,
        "html_url": repo.get("html_url"),
        "description": repo.get("description") or "",
        "primary_language": repo.get("primary_language") or "Unknown",
        "languages": list(repo.get("languages") or []),
        "topics": list(repo.get("topics") or []),
        "star_count": int(repo.get("star_count") or 0),
        "fork_count": int(repo.get("fork_count") or 0),
        "open_issues_count": int(repo.get("open_issues_count") or 0),
        "readme_length": int(repo.get("readme_length") or 0),
        "readme_chunks": readme_chunks,
        "pushed_days_ago": int(repo.get("pushed_days_ago") or 999),
        "delta_3d": int(repo.get("delta_3d") or 0),
        "delta_7d": int(repo.get("delta_7d") or 0),
        "delta_30d": int(repo.get("delta_30d") or 0),
        "mentionable_users_count": int(repo.get("mentionable_users_count") or 0),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "discovery_category": repo.get("discovery_category"),
        "discovery_band": repo.get("discovery_band"),
        "category": category,
        "tags": tags,
        "doc_quality": documentation.score,
        "code_health": score_code_health(dict(repo)),
        "activity_score": activity_score(dict(repo)),
        "trend_velocity": trend_velocity(dict(repo)),
        "embedding_dim": len(final_embedding),
        "embedding_model": config.model_name,
        "embedding_version": config.version,
        "source_hash": source_hash,
    }


def source_fingerprint(*parts: str) -> str:
    """Hash source text used for future re-embedding decisions."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()
