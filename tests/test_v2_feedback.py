import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from embedding.vector_contract import legacy_repository_point_id, legacy_user_point_id
from feedback.event_handlers import ADJUSTMENTS_KEY, APPLIED_SIGNALS_KEY, LATENT_KEY
from feedback.v2 import (
    CONSUMER_HEARTBEAT,
    DurableFeedbackProducer,
    OrderedFeedbackApplier,
    OrderedFeedbackConsumer,
)


class FakeQdrant:
    def __init__(self, user_id, repo_id, last=0):
        self.user_id = user_id
        self.repo_id = repo_id
        self.user = SimpleNamespace(
            id=user_id,
            vector=[1.0, 0.0],
            payload={"last_feedback_version": last},
        )
        self.upserts = []

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        if ids[0] == self.user_id:
            return [self.user]
        return [SimpleNamespace(id=self.repo_id, vector={"repo_embedding": [0.0, 1.0]}, payload={})]

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        self.user = kwargs["points"][0]


def event(user_id, repo_id, version, event_type="like"):
    return {"event_id": str(uuid.uuid4()), "user_id": user_id, "repo_id": repo_id,
            "feedback_version": str(version), "event_type": event_type, "dwell_ms": ""}


def test_v2_feedback_health_reports_dedicated_consumer_heartbeat():
    redis = MagicMock()
    redis.xinfo_groups.return_value = [
        {"name": "ml-feedback-v2", "pending": 2, "lag": 3}
    ]
    redis.exists.return_value = 1

    health = DurableFeedbackProducer(redis).health()

    assert health["feedback_pending"] == 2
    assert health["feedback_lag"] == 3
    assert health["feedback_consumer_active"] is True


def test_consumer_refreshes_heartbeat_during_slow_processing(monkeypatch):
    redis = MagicMock()
    heartbeat_renewed = threading.Event()
    heartbeat_writes = 0

    def record_set(key, *args, **kwargs):
        nonlocal heartbeat_writes
        if key == CONSUMER_HEARTBEAT:
            heartbeat_writes += 1
            if heartbeat_writes >= 2:
                heartbeat_renewed.set()
        return True

    redis.set.side_effect = record_set
    applier = MagicMock()
    applier.apply.side_effect = lambda payload: (
        SimpleNamespace(status="applied")
        if heartbeat_renewed.wait(timeout=1)
        else pytest.fail("heartbeat was not refreshed while processing")
    )
    consumer = OrderedFeedbackConsumer(redis_client=redis, applier=applier)
    consumer._messages = lambda: iter(
        [("1-0", {"user_id": str(uuid.uuid4())})]
    )
    monkeypatch.setattr("feedback.v2.CONSUMER_HEARTBEAT_TTL_SECONDS", 0.3)

    assert consumer.run_once() == 1
    assert heartbeat_writes >= 3


def test_feedback_applies_version_with_vector_in_one_upsert():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    result = OrderedFeedbackApplier(client).apply(event(user_id, repo_id, 1))
    assert result.status == "applied"
    point = client.upserts[0]["points"][0]
    assert point.payload["last_feedback_version"] == 1
    assert np.allclose(point.payload[LATENT_KEY], [0.85, 0.15])


def test_feedback_reads_and_updates_pre_v2_uuid5_points():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    client.user.id = legacy_user_point_id(user_id)
    legacy_repo_id = legacy_repository_point_id(repo_id)

    def retrieve(collection_name, ids, with_payload, with_vectors):
        if client.user.id in ids:
            return [client.user]
        assert legacy_repo_id in ids
        return [
            SimpleNamespace(
                id=legacy_repo_id,
                vector={"repo_embedding": [0.0, 1.0]},
                payload={"repo_id": repo_id},
            )
        ]

    client.retrieve = retrieve
    result = OrderedFeedbackApplier(client).apply(event(user_id, repo_id, 1))

    assert result.status == "applied"
    assert client.upserts[0]["points"][0].id == legacy_user_point_id(user_id)
    assert client.user.payload["last_feedback_version"] == 1


def test_feedback_skips_duplicate_and_holds_version_gap():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    duplicate = OrderedFeedbackApplier(
        FakeQdrant(user_id, repo_id, last=2)
    ).apply(event(user_id, repo_id, 2))
    gap_client = FakeQdrant(user_id, repo_id, last=2)
    gap = OrderedFeedbackApplier(gap_client).apply(event(user_id, repo_id, 4))
    assert duplicate.status == "duplicate"
    assert gap.status == "gap"
    assert gap_client.upserts == []


@pytest.mark.parametrize(
    ("forward", "reverse"),
    [("like", "unlike"), ("dislike", "undislike"), ("save", "unsave")],
)
def test_reversal_clears_state_without_rewriting_learned_vector(forward, reverse):
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(client)

    applier.apply(event(user_id, repo_id, 1, forward))
    assert not np.allclose(client.user.vector, [1.0, 0.0])
    learned_vector = np.asarray(client.user.vector)
    learned_latent = np.asarray(client.user.payload[LATENT_KEY])

    result = applier.apply(event(user_id, repo_id, 2, reverse))

    assert result.status == "applied"
    assert np.allclose(client.user.vector, learned_vector)
    assert np.allclose(client.user.payload[LATENT_KEY], learned_latent)
    assert repo_id not in client.user.payload[ADJUSTMENTS_KEY]
    assert client.user.payload["last_feedback_version"] == 2


def test_reversal_preserves_later_unrelated_feedback():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(client)

    applier.apply(event(user_id, repo_id, 1, "like"))
    applier.apply(event(user_id, repo_id, 2, "readme_open"))
    learned_vector = np.asarray(client.user.vector)
    learned_latent = np.asarray(client.user.payload[LATENT_KEY])
    applier.apply(event(user_id, repo_id, 3, "unlike"))

    assert np.allclose(client.user.vector, learned_vector)
    assert np.allclose(client.user.payload[LATENT_KEY], learned_latent)
    assert repo_id not in client.user.payload[ADJUSTMENTS_KEY]


def test_forward_action_can_apply_again_after_zero_alpha_reversal():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(client)

    applier.apply(event(user_id, repo_id, 1, "like"))
    applier.apply(event(user_id, repo_id, 2, "unlike"))
    latent_after_reversal = np.asarray(client.user.payload[LATENT_KEY])
    applier.apply(event(user_id, repo_id, 3, "like"))

    assert not np.allclose(client.user.payload[LATENT_KEY], latent_after_reversal)
    assert client.user.payload[ADJUSTMENTS_KEY][repo_id]["reaction"]["action"] == "like"
    assert client.user.payload["last_feedback_version"] == 3


@pytest.mark.parametrize("passive_action", ["readme_open", "github_open", "share"])
def test_passive_signal_is_applied_only_once(passive_action):
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(client)

    applier.apply(event(user_id, repo_id, 1, passive_action))
    first_vector = np.asarray(client.user.vector)
    result = applier.apply(event(user_id, repo_id, 2, passive_action))

    assert result.status == "applied"
    assert np.allclose(client.user.vector, first_vector)
    assert client.user.payload[APPLIED_SIGNALS_KEY][repo_id] == [passive_action]
    assert client.user.payload["last_feedback_version"] == 2
