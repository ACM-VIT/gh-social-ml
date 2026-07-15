"""Phase 3 tests for the public repository Qdrant interface."""

from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from config import QDRANT_PAYLOAD_INDEX_FIELDS
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.vector_contract import (
    REPOSITORY_DISCOVERY_CHANNELS,
    repository_point_id,
)


EMBEDDING_DIM = 384
VECTOR_NAME = "repo_embedding"
REPO_ID = "00000000-0000-4000-8000-000000000001"
OTHER_REPO_ID = "00000000-0000-4000-8000-000000000002"


def _vector(value=1.0):
    return [value] + [0.0] * (EMBEDDING_DIM - 1)


def _collection_info(
    *,
    vector_name=VECTOR_NAME,
    vector_size=EMBEDDING_DIM,
    distance=models.Distance.COSINE,
):
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=distance,
                    )
                }
            )
        )
    )


def _point(repo_id, *, score=None):
    return SimpleNamespace(
        id=repository_point_id(repo_id),
        score=score,
        payload={"repo_id": repo_id, "full_name": f"owner/{repo_id}"},
        vector={VECTOR_NAME: _vector()},
    )


class FakeQdrantClient:
    def __init__(self, *, exists=True, collection_info=None):
        self.exists = exists
        self.collection_info = collection_info or _collection_info()
        self.created_collection = None
        self.created_indexes = []
        self.query_kwargs = None
        self.scroll_kwargs = None
        self.retrieve_kwargs = None
        self.query_response = []
        self.scroll_response = []
        self.retrieve_response = []
        self.fail_index = None

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created_collection = kwargs
        self.exists = True
        vectors = kwargs["vectors_config"]
        self.collection_info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
        )

    def get_collection(self, collection_name):
        return self.collection_info

    def create_payload_index(self, **kwargs):
        if kwargs["field_name"] == self.fail_index:
            raise RuntimeError("index creation failed")
        self.created_indexes.append(kwargs)

    def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return SimpleNamespace(points=self.query_response)

    def scroll(self, **kwargs):
        self.scroll_kwargs = kwargs
        return self.scroll_response, None

    def retrieve(self, **kwargs):
        self.retrieve_kwargs = kwargs
        return self.retrieve_response


def _store(client, **overrides):
    return QdrantRepositoryStore(client=client, **overrides)


def test_ensure_collection_creates_and_validates_schema_and_all_indexes():
    client = FakeQdrantClient(exists=False)
    store = _store(client)

    store.ensure_collection()

    assert client.created_collection["collection_name"] == store.collection_name
    params = client.created_collection["vectors_config"][VECTOR_NAME]
    assert params.size == EMBEDDING_DIM
    assert params.distance == models.Distance.COSINE

    schemas = {
        call["field_name"]: call["field_schema"] for call in client.created_indexes
    }
    assert set(schemas) == set(QDRANT_PAYLOAD_INDEX_FIELDS)
    assert schemas["star_count"] == models.PayloadSchemaType.INTEGER
    assert schemas["pushed_days_ago"] == models.PayloadSchemaType.INTEGER
    assert schemas["trend_velocity"] == models.PayloadSchemaType.FLOAT
    assert schemas["activity_score"] == models.PayloadSchemaType.FLOAT
    assert schemas["doc_quality"] == models.PayloadSchemaType.FLOAT
    assert schemas["code_health"] == models.PayloadSchemaType.FLOAT
    assert schemas["pushed_at"] == models.PayloadSchemaType.DATETIME


@pytest.mark.parametrize(
    ("collection_info", "message"),
    [
        (_collection_info(vector_name="wrong_name"), "does not define vector"),
        (
            SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=models.VectorParams(
                            size=EMBEDDING_DIM,
                            distance=models.Distance.COSINE,
                        )
                    )
                )
            ),
            "does not define vector",
        ),
        (_collection_info(vector_size=3), "has size 3, expected 384"),
        (_collection_info(distance=models.Distance.DOT), "uses distance"),
    ],
)
def test_validate_collection_rejects_contract_mismatches(collection_info, message):
    store = _store(FakeQdrantClient(collection_info=collection_info))

    with pytest.raises(ValueError, match=message):
        store.validate_collection()


def test_payload_index_creation_failure_is_not_silently_ignored():
    client = FakeQdrantClient()
    client.fail_index = "trend_velocity"

    with pytest.raises(RuntimeError, match="trend_velocity"):
        _store(client).ensure_collection()


def test_semantic_search_uses_named_approximate_vector_and_normalizes_output():
    client = FakeQdrantClient()
    client.query_response = [_point(REPO_ID, score=0.91)]
    store = _store(client)

    results = store.semantic_search(_vector(), limit=7)

    assert client.query_kwargs["using"] == VECTOR_NAME
    assert client.query_kwargs["limit"] == 7
    assert client.query_kwargs["search_params"].exact is False
    assert client.query_kwargs["with_vectors"] is True
    assert results[0]["repo_id"] == REPO_ID
    assert results[0]["score"] == pytest.approx(0.91)
    assert results[0]["vector"] == _vector()


def test_search_rejects_bad_vectors_before_calling_qdrant():
    client = FakeQdrantClient()

    with pytest.raises(ValueError, match="dimension"):
        _store(client).search([1.0, 0.0])
    assert client.query_kwargs is None


@pytest.mark.parametrize(
    ("channel", "field_name"),
    list(REPOSITORY_DISCOVERY_CHANNELS.items()),
)
def test_discover_orders_each_frozen_channel_descending(channel, field_name):
    client = FakeQdrantClient()
    client.scroll_response = [_point(REPO_ID)]
    store = _store(client)

    results = store.discover(channel, limit=4)

    order_by = client.scroll_kwargs["order_by"]
    assert order_by.key == field_name
    assert order_by.direction == models.Direction.DESC
    assert client.scroll_kwargs["with_vectors"] == [VECTOR_NAME]
    assert results[0]["repo_id"] == REPO_ID


def test_discover_rejects_unknown_channel():
    with pytest.raises(ValueError, match="Unsupported discovery channel"):
        _store(FakeQdrantClient()).discover("random")


def test_retrieve_batch_deduplicates_request_and_restores_caller_order():
    client = FakeQdrantClient()
    client.retrieve_response = [_point(OTHER_REPO_ID), _point(REPO_ID)]
    store = _store(client)

    results = store.retrieve_batch([REPO_ID, OTHER_REPO_ID, REPO_ID])

    assert client.retrieve_kwargs["ids"] == [
        repository_point_id(REPO_ID),
        repository_point_id(OTHER_REPO_ID),
    ]
    assert client.retrieve_kwargs["with_vectors"] == [VECTOR_NAME]
    assert [result["repo_id"] for result in results] == [REPO_ID, OTHER_REPO_ID]


def test_retrieve_batch_canonicalizes_uuid_case_before_deduplication():
    repo_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    client = FakeQdrantClient()
    client.retrieve_response = [_point(repo_id)]
    store = _store(client)

    results = store.retrieve_batch([repo_id.upper(), repo_id])

    assert client.retrieve_kwargs["ids"] == [repository_point_id(repo_id)]
    assert [result["repo_id"] for result in results] == [repo_id]


@pytest.mark.parametrize("repo_ids", ["repo-1", [""], [123]])
def test_retrieve_batch_rejects_invalid_repo_id_inputs(repo_ids):
    with pytest.raises((TypeError, ValueError)):
        _store(FakeQdrantClient()).retrieve_batch(repo_ids)


@pytest.mark.parametrize(
    "overrides",
    [
        {"collection_name": ""},
        {"vector_name": ""},
        {"vector_size": 0},
        {"distance": "not-a-distance"},
    ],
)
def test_store_rejects_invalid_local_contract_configuration(overrides):
    with pytest.raises(ValueError):
        _store(FakeQdrantClient(), **overrides)
