from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from retrieval.candidate_retriever import CandidateRetriever
from retrieval.config import (
    DISCOVERY_CHANNELS,
    EMBEDDING_DIM,
    FALLBACK_REPOS,
    QDRANT_VECTOR_NAME,
    SEMANTIC_LIMIT,
    TOTAL_CANDIDATE_POOL,
)


def point(
    repo_id: str,
    *,
    full_name: str | None = None,
    vector_value: float = 0.1,
    **payload_fields,
):
    payload = {
        "repo_id": repo_id,
        "full_name": full_name or f"org/{repo_id}",
        "description": f"Description for {repo_id}",
        "languages": ["Python"],
        "topics": ["machine-learning"],
        "tags": ["recommendation"],
        "star_count": 100,
        "activity_score": 0.7,
        "trend_velocity": 0.4,
        "updated_at": "2026-07-14T00:00:00Z",
        **payload_fields,
    }
    return SimpleNamespace(
        id=f"point-{repo_id}",
        payload=payload,
        vector={QDRANT_VECTOR_NAME: [vector_value] * EMBEDDING_DIM},
        score=0.91,
    )


def formatted_point(*args, score: float | None = None, **kwargs):
    raw = point(*args, **kwargs)
    return {
        "id": raw.id,
        "score": raw.score if score is None else score,
        "repo_id": raw.payload["repo_id"],
        "full_name": raw.payload["full_name"],
        "payload": raw.payload,
        "vector": raw.vector[QDRANT_VECTOR_NAME],
    }


@pytest.fixture
def store():
    mock_store = MagicMock()
    mock_store.semantic_search.return_value = []
    mock_store.discover.return_value = []
    return mock_store


@pytest.fixture
def retriever(store):
    # A legacy connector can still be passed by the untouched retrieval engine,
    # but the Qdrant-only retriever must neither retain nor call it.
    forbidden_connector = MagicMock()
    result = CandidateRetriever(
        db_connector=forbidden_connector,
        qdrant_store=store,
    )
    assert all(value is not forbidden_connector for value in result.__dict__.values())
    return result


def test_semantic_retrieval_is_approximate_and_returns_complete_candidate(retriever, store):
    semantic_point = formatted_point("repo-1", full_name="acme/repo-1")
    store.semantic_search.return_value = [semantic_point]

    results = retriever._retrieve_semantic([0.1] * EMBEDDING_DIM, quota=10)

    assert len(results) == 1
    candidate = results[0]
    assert candidate["repo_id"] == "repo-1"
    assert candidate["full_name"] == "acme/repo-1"
    assert candidate["repo_embedding"] == [0.1] * EMBEDDING_DIM
    assert candidate["payload"] == semantic_point["payload"]
    assert candidate["retrieval_source"] == "semantic"
    assert candidate["retrieval_score"] == pytest.approx(0.91)
    # Flattened payload fields keep the current ranker compatible.
    assert candidate["languages"] == ["Python"]
    store.semantic_search.assert_called_once_with(
        [0.1] * EMBEDDING_DIM,
        limit=15,
        with_vectors=True,
    )


@pytest.mark.parametrize("source,channel,order_field", DISCOVERY_CHANNELS)
def test_each_discovery_channel_orders_in_qdrant(
    retriever, store, source, channel, order_field
):
    store.discover.return_value = [formatted_point(source, **{order_field: 42})]

    results = retriever._retrieve_discovery(
        source=source,
        channel=channel,
        order_field=order_field,
        quota=4,
    )

    assert len(results) == 1
    assert results[0]["retrieval_source"] == source
    assert results[0]["retrieval_score"] == pytest.approx(42.0)
    store.discover.assert_called_once_with(
        channel,
        limit=6,
        with_vectors=True,
    )


def test_discovery_channels_are_round_robin(retriever):
    channels = [
        [{"repo_id": "t1"}, {"repo_id": "t2"}],
        [{"repo_id": "a1"}, {"repo_id": "a2"}],
        [{"repo_id": "p1"}],
        [{"repo_id": "f1"}],
    ]

    merged = retriever._round_robin(channels, limit=6)

    assert [item["repo_id"] for item in merged] == [
        "t1",
        "a1",
        "p1",
        "f1",
        "t2",
        "a2",
    ]


def test_fresh_channel_exposes_iso_timestamp_as_numeric_retrieval_score(
    retriever, store
):
    store.discover.return_value = [
        formatted_point("fresh-repo", pushed_at="2026-07-14T00:00:00Z")
    ]

    results = retriever._retrieve_discovery(
        source="fresh",
        channel="freshness",
        order_field="pushed_at",
        quota=1,
    )

    assert results[0]["retrieval_score"] > 0


def test_round_robin_duplicates_do_not_consume_unique_quota(retriever):
    channels = [
        [
            {"repo_id": "shared", "full_name": "org/shared"},
            {"repo_id": "t2", "full_name": "org/t2"},
        ],
        [
            {"repo_id": "different-id", "full_name": "ORG/SHARED"},
            {"repo_id": "a2", "full_name": "org/a2"},
        ],
    ]

    merged = retriever._round_robin(channels, limit=3)

    assert [item["repo_id"] for item in merged] == ["shared", "t2", "a2"]


def test_deduplicates_by_repo_id_and_full_name(retriever):
    semantic = [
        {"repo_id": "repo-1", "full_name": "Acme/One", "retrieval_source": "semantic"},
        {"repo_id": "repo-2", "full_name": "Acme/Two", "retrieval_source": "semantic"},
    ]
    discovery = [
        {"repo_id": "repo-1", "full_name": "other/name", "retrieval_source": "trending"},
        {"repo_id": "different-id", "full_name": "acme/two", "retrieval_source": "active"},
        {"repo_id": "repo-3", "full_name": "Acme/Three", "retrieval_source": "fresh"},
    ]

    merged = retriever._merge_and_deduplicate(
        semantic, discovery, pool_limit=TOTAL_CANDIDATE_POOL
    )

    assert [item["repo_id"] for item in merged] == ["repo-1", "repo-2", "repo-3"]
    assert merged[0]["retrieval_source"] == "semantic"


def test_missing_user_vector_expands_discovery_to_full_pool(retriever, monkeypatch):
    calls = []

    def discovery(*, source, channel, order_field, quota):
        calls.append((source, channel, order_field, quota))
        return (
            [
                {
                    "repo_id": f"{source}-{index}",
                    "full_name": f"org/{source}-{index}",
                    "repo_embedding": [0.1] * EMBEDDING_DIM,
                    "payload": {},
                    "retrieval_source": source,
                    "retrieval_score": float(index),
                }
                for index in range(quota)
            ],
            False,
        )

    monkeypatch.setattr(retriever, "_retrieve_discovery_with_status", discovery)

    candidates = retriever.retrieve_candidates(user_embedding=None)

    assert len(candidates) == TOTAL_CANDIDATE_POOL
    assert len(calls) == len(DISCOVERY_CHANNELS)
    assert {quota for _, _, _, quota in calls} == {38}
    assert {candidate["retrieval_source"] for candidate in candidates} == {
        "trending",
        "active",
        "popular",
        "fresh",
    }


def test_semantic_shortfall_is_filled_by_discovery(retriever, monkeypatch):
    semantic = [
        {
            "repo_id": f"semantic-{index}",
            "full_name": f"org/semantic-{index}",
            "repo_embedding": [0.1] * EMBEDDING_DIM,
            "payload": {},
            "retrieval_source": "semantic",
            "retrieval_score": 1.0,
        }
        for index in range(100)
    ]
    monkeypatch.setattr(
        retriever,
        "_retrieve_semantic_with_status",
        lambda embedding, quota: (semantic, False),
    )
    quotas = []

    def discovery(*, source, channel, order_field, quota):
        quotas.append(quota)
        return (
            [
                {
                    "repo_id": f"{source}-{index}",
                    "full_name": f"org/{source}-{index}",
                    "repo_embedding": [0.2] * EMBEDDING_DIM,
                    "payload": {},
                    "retrieval_source": source,
                    "retrieval_score": 1.0,
                }
                for index in range(quota)
            ],
            False,
        )

    monkeypatch.setattr(retriever, "_retrieve_discovery_with_status", discovery)

    candidates = retriever.retrieve_candidates([0.1] * EMBEDDING_DIM)

    assert len(candidates) == TOTAL_CANDIDATE_POOL
    assert quotas == [13, 13, 13, 13]
    assert sum(item["retrieval_source"] == "semantic" for item in candidates) == 100


def test_one_channel_failure_does_not_discard_other_channels(retriever, store):
    def discover(channel, **kwargs):
        if channel == "trend":
            raise RuntimeError("trending index unavailable")
        source = {
            published_channel: source
            for source, published_channel, _ in DISCOVERY_CHANNELS
        }[channel]
        return [formatted_point(source)]

    store.discover.side_effect = discover

    candidates = retriever.retrieve_candidates(user_embedding=None)

    assert candidates
    assert all(candidate["retrieval_source"] != "fallback" for candidate in candidates)
    assert {candidate["retrieval_source"] for candidate in candidates} == {
        "active",
        "popular",
        "fresh",
    }


def test_static_fallback_is_used_only_when_every_qdrant_channel_fails(
    retriever, store
):
    store.semantic_search.side_effect = RuntimeError("semantic unavailable")
    store.discover.side_effect = RuntimeError("discovery unavailable")

    candidates = retriever.retrieve_candidates([0.1] * EMBEDDING_DIM)

    assert len(candidates) == len(FALLBACK_REPOS)
    assert all(candidate["retrieval_source"] == "fallback" for candidate in candidates)
    assert all(len(candidate["repo_embedding"]) == EMBEDDING_DIM for candidate in candidates)


def test_empty_successful_qdrant_queries_do_not_use_static_fallback(retriever, store):
    store.semantic_search.return_value = []
    store.discover.return_value = []

    assert retriever.retrieve_candidates([0.1] * EMBEDDING_DIM) == []


def test_invalid_user_vector_uses_discovery_instead_of_semantic(
    retriever, store, monkeypatch
):
    monkeypatch.setattr(
        retriever,
        "_retrieve_discovery_with_status",
        lambda **kwargs: ([], False),
    )

    retriever.retrieve_candidates([0.1] * (EMBEDDING_DIM - 1))

    store.semantic_search.assert_not_called()


def test_pool_is_bounded(retriever, monkeypatch):
    monkeypatch.setattr(
        retriever,
        "_retrieve_discovery_with_status",
        lambda source, channel, order_field, quota: (
            [
                {
                    "repo_id": f"{source}-{index}",
                    "full_name": f"org/{source}-{index}",
                    "repo_embedding": [0.1] * EMBEDDING_DIM,
                    "payload": {},
                    "retrieval_source": source,
                    "retrieval_score": 1.0,
                }
                for index in range(quota + 100)
            ],
            False,
        ),
    )

    candidates = retriever.retrieve_candidates(None)

    assert len(candidates) == TOTAL_CANDIDATE_POOL
    assert TOTAL_CANDIDATE_POOL == 150


def test_person_three_retriever_contains_no_online_database_operations():
    import inspect
    import retrieval.candidate_retriever as module

    source = inspect.getsource(module).lower()
    forbidden = ("postgres", "database.connector", "cursor.execute", "select ", "hydrate")
    assert not any(token in source for token in forbidden)
    assert "db_connector" in source  # ignored compatibility shim only
