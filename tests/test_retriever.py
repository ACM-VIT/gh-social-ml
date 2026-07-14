from unittest.mock import MagicMock

import pytest

from retrieval.candidate_retriever import CandidateRetriever
from retrieval.config import EMBEDDING_DIM, FALLBACK_REPOS, TOTAL_CANDIDATE_POOL


def qdrant_point(
    repo_id: str,
    *,
    score: float | None = None,
    stars: int = 100,
    vector_value: float = 0.2,
) -> dict:
    return {
        "id": f"point-{repo_id}",
        "repo_id": repo_id,
        "full_name": repo_id,
        "score": score,
        "payload": {
            "repo_id": repo_id,
            "full_name": repo_id,
            "description": f"Description for {repo_id}",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["machine-learning"],
            "star_count": stars,
            "fork_count": 10,
            "doc_quality": 0.8,
            "code_health": 0.7,
            "activity_score": 0.6,
            "trend_velocity": 0.5,
        },
        "vector": [vector_value] * EMBEDDING_DIM,
    }


@pytest.fixture
def retriever() -> CandidateRetriever:
    instance = CandidateRetriever.__new__(CandidateRetriever)
    instance._qdrant_store = MagicMock()
    return instance


def test_semantic_retrieval_uses_approximate_qdrant_search(retriever):
    retriever._qdrant_store.search.return_value = [
        qdrant_point("org/repo-1", score=0.95),
        qdrant_point("org/repo-2", score=0.88),
    ]

    results = retriever._retrieve_semantic([0.1] * EMBEDDING_DIM, quota=10)

    assert [item["repo_id"] for item in results] == ["org/repo-1", "org/repo-2"]
    assert results[0]["retrieval_source"] == "semantic"
    assert results[0]["retrieval_score"] == 0.95
    assert results[0]["repo_embedding"] == [0.2] * EMBEDDING_DIM
    retriever._qdrant_store.search.assert_called_once_with(
        vector=[0.1] * EMBEDDING_DIM,
        limit=15,
        with_vectors=True,
        exact=False,
    )


def test_discovery_interleaves_qdrant_signals(retriever):
    retriever._qdrant_store.list_points_ordered.side_effect = [
        [qdrant_point("org/trending")],
        [qdrant_point("org/active")],
        [qdrant_point("org/popular")],
        [qdrant_point("org/fresh")],
    ]

    results = retriever._retrieve_discovery(quota=4)

    assert [item["repo_id"] for item in results] == [
        "org/trending",
        "org/active",
        "org/popular",
        "org/fresh",
    ]
    assert retriever._qdrant_store.list_points_ordered.call_count == 4


def test_merge_prefers_semantic_and_deduplicates(retriever):
    semantic = [
        {"repo_id": "r1", "retrieval_score": 0.9},
        {"repo_id": "r2", "retrieval_score": 0.8},
    ]
    discovery = [
        {"repo_id": "r2", "star_count": 1000},
        {"repo_id": "r3", "star_count": 500},
    ]

    merged = retriever._merge_and_deduplicate(
        semantic,
        discovery,
        semantic_limit=2,
        pool_limit=3,
    )

    assert [item["repo_id"] for item in merged] == ["r1", "r2", "r3"]
    assert "star_count" not in merged[1]


def test_qdrant_unavailable_returns_bounded_fallback(retriever):
    retriever._qdrant_store = None

    results = retriever.retrieve_candidates(user_embedding=[0.1] * EMBEDDING_DIM)

    assert len(results) == len(FALLBACK_REPOS)
    assert results[0]["repo_id"] == FALLBACK_REPOS[0]
    assert results[0]["retrieval_source"] == "fallback"
    assert results[0]["repo_embedding"] == [0.0] * EMBEDDING_DIM


def test_end_to_end_retrieval_never_needs_postgres_hydration(retriever):
    retriever._qdrant_store.search.return_value = [
        qdrant_point("org/semantic", score=0.9, vector_value=0.5)
    ]
    retriever._qdrant_store.list_points_ordered.side_effect = [
        [qdrant_point("org/discovery", stars=500)],
        [],
        [],
        [],
    ]

    candidates = retriever.retrieve_candidates(user_embedding=[0.1] * EMBEDDING_DIM)

    assert [item["repo_id"] for item in candidates] == [
        "org/semantic",
        "org/discovery",
    ]
    assert candidates[0]["repo_embedding"] == [0.5] * EMBEDDING_DIM
    assert candidates[1]["star_count"] == 500
    assert candidates[1]["forks_count"] == 10


def test_semantic_failure_reallocates_full_pool_to_discovery(retriever):
    retriever._qdrant_store.search.return_value = []
    discovery = [qdrant_point(f"org/repo-{index}") for index in range(10)]
    retriever._qdrant_store.list_points_ordered.side_effect = [discovery, [], [], []]

    candidates = retriever.retrieve_candidates(user_embedding=[0.1] * EMBEDDING_DIM)

    first_call = retriever._qdrant_store.list_points_ordered.call_args_list[0]
    # Full-pool discovery over-fetch is split across four indexed signals.
    assert first_call.kwargs["limit"] == 50
    assert len(candidates) == 10
    assert len(candidates) <= TOTAL_CANDIDATE_POOL


def test_candidates_without_indexed_vectors_are_rejected(retriever):
    point = qdrant_point("org/missing-vector")
    point["vector"] = None
    retriever._qdrant_store.search.return_value = [point]

    assert retriever._retrieve_semantic([0.1] * EMBEDDING_DIM, quota=10) == []
