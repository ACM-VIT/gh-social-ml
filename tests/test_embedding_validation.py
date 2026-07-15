"""Phase 2 tests for repository input and embedding validation."""

import math
from datetime import datetime, timezone

import pytest

from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.embeddings import aggregate_vectors
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_vector_payload
from embedding.vector_contract import resolve_repository_identity, validate_embedding_vector


EMBEDDING_DIM = 384
REPO_ID = "00000000-0000-4000-8000-000000000001"


class FakeEmbedder:
    """Small deterministic embedder that avoids loading the real ML model."""

    def __init__(self, *, dimension: int = EMBEDDING_DIM, value: float = 1.0) -> None:
        self.dimension = dimension
        self.value = value

    def embed_texts(self, texts, *, normalize=True):
        return [self._vector() for _ in texts]

    def embed_text(self, text, *, normalize=True):
        return self._vector()

    def _vector(self):
        return [self.value] + [0.0] * (self.dimension - 1)


class UnusedStore:
    def validate_collection(self):
        raise AssertionError("invalid queries must fail before contacting Qdrant")


def _repository(**overrides):
    repo = {
        "repo_id": REPO_ID,
        "github_id": "123456789",
        "content_version": 7,
        "full_name": "owner/repository",
        "description": "A documented machine-learning repository.",
        "primary_language": "Python",
        "languages": ["Python"],
        "topics": ["machine-learning"],
        "extracted_paragraphs": ["Installation and usage documentation."],
        "readme_length": 40,
        "star_count": 10,
        "fork_count": 2,
        "open_issues_count": 1,
        "pushed_days_ago": 3,
    }
    repo.update(overrides)
    return repo


def test_pipeline_emits_finite_normalized_vector_and_canonical_identity():
    result = RepositoryEmbeddingPipeline(embedder=FakeEmbedder()).embed_repository(
        _repository()
    )

    assert result.repo_id == REPO_ID
    assert result.payload["repo_id"] == REPO_ID
    assert result.payload["full_name"] == "owner/repository"
    assert len(result.final_embedding) == EMBEDDING_DIM
    assert all(math.isfinite(value) for value in result.final_embedding)
    assert math.sqrt(sum(value * value for value in result.final_embedding)) == pytest.approx(1.0)


def test_payload_tags_use_repository_name_instead_of_backend_uuid():
    result = RepositoryEmbeddingPipeline(embedder=FakeEmbedder()).embed_repository(
        _repository(full_name="owner/vector-search-engine")
    )

    assert {"vector", "search", "engine"}.issubset(result.payload["tags"])


def test_full_name_cannot_be_used_as_a_canonical_repository_id():
    repo = _repository()
    repo.pop("repo_id")
    repo.pop("full_name")
    repo["id"] = "owner/repository"

    with pytest.raises(ValueError, match="backend-issued UUID"):
        RepositoryEmbeddingPipeline(embedder=FakeEmbedder()).embed_repository(repo)


@pytest.mark.parametrize(
    "repo",
    [
        {},
        {"repo_id": REPO_ID},
        {"repo_id": REPO_ID, "full_name": "not-a-full-name"},
        {"repo_id": REPO_ID, "full_name": "owner/repository/extra"},
    ],
)
def test_invalid_or_incomplete_repository_identity_is_rejected(repo):
    with pytest.raises((TypeError, ValueError)):
        resolve_repository_identity(repo)


def test_pipeline_rejects_wrong_embedding_dimension():
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder(dimension=3))

    with pytest.raises(ValueError, match="dimension 3, expected 384"):
        pipeline.embed_repository(_repository())


def test_pipeline_rejects_non_finite_embedding_values():
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder(value=float("nan")))

    with pytest.raises(ValueError, match="must be finite"):
        pipeline.embed_repository(_repository())


def test_search_rejects_empty_query_before_contacting_qdrant():
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder(), store=UnusedStore())

    with pytest.raises(ValueError, match="non-empty"):
        pipeline.search("   ")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"version": ""}, "version"),
        ({"readme_chunk_chars": 0}, "readme_chunk_chars"),
        (
            {"readme_chunk_chars": 100, "readme_chunk_overlap_chars": 100},
            "smaller",
        ),
        ({"tower_weights": {"readme": 1.0}}, "exactly"),
        (
            {"tower_weights": {"readme": 0.0, "metadata": 0.0, "topics": 0.0}},
            "positive total",
        ),
        (
            {
                "tower_weights": {
                    "readme": 1.0,
                    "metadata": -0.1,
                    "topics": 0.1,
                }
            },
            "non-negative",
        ),
    ],
)
def test_repository_embedding_config_rejects_unsafe_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RepositoryEmbeddingConfig(**kwargs)


def test_aggregate_vectors_requires_compatible_finite_vectors():
    with pytest.raises(ValueError, match="dimension"):
        aggregate_vectors([[1.0, 0.0], [1.0]])
    with pytest.raises(ValueError, match="finite"):
        aggregate_vectors([[float("inf"), 0.0]])
    with pytest.raises(ValueError, match="zero-length"):
        aggregate_vectors([[0.0, 0.0]])


def test_aggregate_vectors_rejects_invalid_weights():
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(ValueError, match="positive total"):
        aggregate_vectors(vectors, weights=[0.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_vectors(vectors, weights=[1.0, -1.0])


def test_payload_normalizes_supported_values_and_timestamps():
    payload = build_vector_payload(
        _repository(
            star_count="10",
            topics='["machine-learning", "vectors"]',
            created_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        ),
        repo_id=REPO_ID,
        final_embedding=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
        readme_chunks=1,
        source_hash="source-hash",
        config=RepositoryEmbeddingConfig(),
    )

    assert payload["star_count"] == 10
    assert payload["topics"] == ["machine-learning", "vectors"]
    assert payload["created_at"] == "2026-07-15T00:00:00+00:00"
    assert payload["github_id"] == "123456789"
    assert payload["content_version"] == 7
    assert payload["content_hash"] == "source-hash"
    assert payload["model_version"] == payload["embedding_model"]
    assert payload["indexed_at"].endswith("+00:00")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"star_count": -1}, "star_count"),
        ({"fork_count": 2.5}, "fork_count"),
        ({"delta_3d": True}, "delta_3d"),
        ({"created_at": "not-a-timestamp"}, "created_at"),
        ({"created_at": "2026-07-15T00:00:00+05:30"}, "UTC"),
        ({"github_id": "github-123"}, "github_id"),
        ({"content_version": -1}, "content_version"),
    ],
)
def test_payload_rejects_invalid_numeric_and_timestamp_values(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_vector_payload(
            _repository(**overrides),
            repo_id=REPO_ID,
            final_embedding=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
            readme_chunks=1,
            source_hash="source-hash",
            config=RepositoryEmbeddingConfig(),
        )


def test_vector_validation_rejects_boolean_values():
    with pytest.raises(TypeError, match=r"embedding\[0\]"):
        validate_embedding_vector([True], expected_size=1)
