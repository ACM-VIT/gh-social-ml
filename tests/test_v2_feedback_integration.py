"""Opt-in end-to-end test for the actual synchronous v2 feedback boundary."""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import replace
from unittest.mock import patch

import pytest
import redis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from api.v2 import router
from config import (
    EMBEDDING_MODEL_REVISION,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    REPOSITORY_FEATURE_SPEC_VERSION,
)
from embedding.vector_contract import (
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
    repository_point_id,
    user_point_id,
)
from feedback.v2 import (
    DurableFeedbackProducer,
    OrderedFeedbackApplier,
    OrderedFeedbackConsumer,
)
from feedback.v2_settings import V2FeedbackSettings


pytestmark = pytest.mark.integration


def _enabled() -> bool:
    return os.getenv("RUN_V2_FEEDBACK_INTEGRATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_v2_api_stream_consumer_restart_qdrant_round_trip(monkeypatch):
    if not _enabled():
        pytest.skip(
            "set RUN_V2_FEEDBACK_INTEGRATION=true to exercise real Redis and Qdrant"
        )

    base = V2FeedbackSettings.from_env()
    if not base.redis_url:
        pytest.fail("REDIS_URL is required when RUN_V2_FEEDBACK_INTEGRATION=true")

    suffix = uuid.uuid4().hex
    settings = replace(
        base,
        stream_name=f"ml:feedback:v2:integration:{suffix}",
        stream_maxlen=1_000,
        consumer_group=f"ml-feedback-v2-integration-{suffix}",
        consumer_name_prefix=f"v2-integration-{suffix}",
        heartbeat_key=f"ml:feedback:v2:integration:{suffix}:heartbeat",
        read_block_ms=100,
        reclaim_idle_ms=100,
        user_lock_prefix=f"ml:user-vector-lock:integration:{suffix}",
        max_delivery_attempts=2,
        dead_letter_stream=f"ml:feedback:v2:integration:{suffix}:dead",
        dead_letter_maxlen=100,
        repository_collection=f"v2_feedback_repositories_{suffix}",
        user_collection=f"v2_feedback_users_{suffix}",
        user_vector_name=None,
        vector_dimension=2,
    )
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )
    user_id = str(uuid.uuid4())
    repo_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())
    created: list[str] = []

    try:
        # Once explicitly enabled, unavailable services are a test failure; CI
        # must not silently skip the production architecture gate.
        redis_client.ping()
        qdrant.get_collections()
        qdrant.create_collection(
            collection_name=settings.user_collection,
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        created.append(settings.user_collection)
        qdrant.create_collection(
            collection_name=settings.repository_collection,
            vectors_config={
                settings.repository_vector_name: models.VectorParams(
                    size=2, distance=models.Distance.COSINE
                )
            },
        )
        created.append(settings.repository_collection)
        qdrant.upsert(
            collection_name=settings.user_collection,
            points=[
                models.PointStruct(
                    id=user_point_id(user_id),
                    vector=[1.0, 0.0],
                    payload={"user_id": user_id, "last_feedback_version": 0},
                )
            ],
            wait=True,
        )
        qdrant.upsert(
            collection_name=settings.repository_collection,
            points=[
                models.PointStruct(
                    id=repository_point_id(repo_id),
                    vector={settings.repository_vector_name: [0.0, 1.0]},
                    payload={
                        "repo_id": repo_id,
                        REPOSITORY_SERVING_ELIGIBILITY_FIELD: REPOSITORY_SERVING_ELIGIBILITY_VERSION,
                        "content_version": 1,
                        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
                        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                        "embedding_version": REPOSITORY_EMBEDDING_VERSION,
                        "embedding_dim": 2,
                        "feature_spec_version": REPOSITORY_FEATURE_SPEC_VERSION,
                    },
                )
            ],
            wait=True,
        )

        producer = DurableFeedbackProducer(redis_client, settings)
        first_consumer = OrderedFeedbackConsumer(
            redis_client,
            OrderedFeedbackApplier(qdrant, settings),
            settings,
        )
        app = FastAPI()
        app.include_router(router)
        monkeypatch.setenv("INTERNAL_API_SECRET", "v2-feedback-integration-secret")
        with patch("api.v2.producer", return_value=producer):
            response = TestClient(app).post(
                "/api/v2/feedback/batch",
                headers={"x-internal-secret": "v2-feedback-integration-secret"},
                json={
                    "schema_version": 2,
                    "events": [
                        {
                            "event_id": event_id,
                            "user_id": user_id,
                            "repo_id": repo_id,
                            "feedback_version": 1,
                            "event_type": "like",
                            "dwell_ms": None,
                            "occurred_at": "2026-07-21T00:00:00Z",
                        }
                    ],
                },
            )
        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] == 1
        assert body["duplicates"] == 0
        assert body["durable"] is True

        # Simulate a worker stopping after Redis delivery but before apply/ACK.
        delivered = redis_client.xreadgroup(
            settings.consumer_group,
            first_consumer.consumer,
            {settings.stream_name: ">"},
            count=1,
            block=1_000,
        )
        assert delivered and delivered[0][1]

        # Wait on the observable pending idle time rather than an arbitrary
        # fixed sleep, then start a new consumer identity and reclaim it.
        deadline = time.monotonic() + 2
        pending_ready = threading.Event()
        while time.monotonic() < deadline:
            entries = redis_client.xpending_range(
                settings.stream_name,
                settings.consumer_group,
                min="-",
                max="+",
                count=1,
            )
            if entries and int(entries[0]["time_since_delivered"]) >= settings.reclaim_idle_ms:
                pending_ready.set()
                break
            pending_ready.wait(0.01)
        assert pending_ready.is_set(), "pending entry did not become reclaimable"

        restarted_consumer = OrderedFeedbackConsumer(
            redis_client,
            OrderedFeedbackApplier(qdrant, settings),
            settings,
        )
        assert restarted_consumer.run_once() == 1

        updated = qdrant.retrieve(
            collection_name=settings.user_collection,
            ids=[user_point_id(user_id)],
            with_payload=True,
            with_vectors=True,
        )[0]
        assert updated.payload["last_feedback_version"] == 1
        assert updated.payload["last_feedback_event_id"] == event_id
        assert updated.vector != pytest.approx([1.0, 0.0])
        pending = redis_client.xpending(settings.stream_name, settings.consumer_group)
        assert int(pending["pending"]) == 0
        assert producer.health()["feedback_consumer_active"] is True
    finally:
        try:
            redis_client.delete(
                settings.stream_name,
                settings.dead_letter_stream,
                settings.heartbeat_key,
                f"{settings.stream_name}:accepted:{event_id}",
                f"{settings.stream_name}:attempts:{event_id}",
                f"{settings.user_lock_prefix}:{user_id}",
            )
        except Exception:
            pass
        for collection_name in reversed(created):
            try:
                qdrant.delete_collection(collection_name=collection_name)
            except Exception:
                pass
        try:
            redis_client.close()
        finally:
            qdrant.close()
