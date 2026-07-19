from __future__ import annotations

import json
import logging
import math
import os
import socket
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_VECTOR_NAME
from embedding.vector_contract import repository_point_ids, user_point_ids
from feedback.event_handlers import (
    ADJUSTMENTS_KEY,
    APPLIED_SIGNALS_KEY,
    LATENT_KEY,
    vector_delta,
)
from feedback.interactions import get_interaction
from scripts.user_onboarding import TARGET_VECTOR_NAME, USER_PROFILES_COLLECTION

logger = logging.getLogger(__name__)

STREAM = "ml:feedback:v2"
GROUP = "ml-feedback-v2"
CONSUMER_HEARTBEAT = "ml:feedback:v2:consumer-heartbeat"
CONSUMER_HEARTBEAT_TTL_SECONDS = 15
ACCEPT_LUA = """
if redis.call('set', KEYS[1], '1', 'NX', 'EX', ARGV[1]) then
  return redis.call('xadd', KEYS[2], '*', unpack(ARGV, 2))
end
return 'duplicate'
"""
RELEASE_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end
return 0
"""

ALPHAS = {
    "readme_open": 0.05,
    "github_open": 0.07,
    "share": 0.10,
    "like": 0.15,
    "dislike": -0.15,
    "save": 0.20,
    "unlike": 0.0,
    "undislike": 0.0,
    "unsave": 0.0,
}


def _redis_client(redis_url: str | None = None):
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("redis>=5 is required for the production v2 feedback boundary") from exc
    url = redis_url or os.getenv("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL is required for durable v2 feedback")
    client = redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
        health_check_interval=30,
    )
    client.ping()
    return client


class DurableFeedbackProducer:
    def __init__(self, redis_client=None) -> None:
        self.redis = redis_client or _redis_client()

    def enqueue(self, events: Iterable[dict[str, Any]]) -> tuple[int, int]:
        accepted = 0
        duplicates = 0
        for event in events:
            fields: list[str] = []
            for key, value in event.items():
                fields.extend([key, json.dumps(value) if isinstance(value, (dict, list)) else str(value or "")])
            result = self.redis.eval(
                ACCEPT_LUA,
                2,
                f"ml:feedback:v2:accepted:{event['event_id']}",
                STREAM,
                str(30 * 24 * 60 * 60),
                *fields,
            )
            if result == "duplicate":
                duplicates += 1
            else:
                accepted += 1
        return accepted, duplicates

    def health(self) -> dict[str, Any]:
        self.redis.ping()
        try:
            groups = self.redis.xinfo_groups(STREAM)
            group = next((item for item in groups if item.get("name") == GROUP), {})
        except Exception:
            group = {}
        return {
            "redis": "healthy",
            "feedback_pending": int(group.get("pending", 0)),
            "feedback_lag": int(group.get("lag", 0) or 0),
            "feedback_consumer_active": bool(self.redis.exists(CONSUMER_HEARTBEAT)),
        }


@dataclass(frozen=True, slots=True)
class ApplyResult:
    status: str
    last_feedback_version: int


class OrderedFeedbackApplier:
    def __init__(self, qdrant: QdrantClient | None = None) -> None:
        self.qdrant = qdrant or QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10.0)

    @staticmethod
    def _vector(value: Any, preferred: str | None = None) -> tuple[list[float], str | None]:
        if isinstance(value, dict):
            if preferred and preferred in value:
                return list(value[preferred]), preferred
            if len(value) == 1:
                name, vector = next(iter(value.items()))
                return list(vector), name
            raise ValueError("ambiguous named vector")
        if value is None:
            raise ValueError("missing vector")
        return list(value), None

    @staticmethod
    def _alpha(event: dict[str, Any]) -> float:
        if event["event_type"] != "dwell":
            return ALPHAS[event["event_type"]]
        dwell = min(300_000, max(3_000, int(event.get("dwell_ms") or 3_000)))
        return 0.15 * math.log1p(dwell) / math.log1p(300_000)

    @staticmethod
    def _finite_vector(value: Any, dimension: int, *, label: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.ndim != 1 or len(vector) != dimension:
            raise ValueError(f"{label} must contain exactly {dimension} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} contains a non-finite value")
        return vector

    @staticmethod
    def _adjustments(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        value = payload.get(ADJUSTMENTS_KEY, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"{ADJUSTMENTS_KEY} must be an object")
        adjustments = deepcopy(dict(value))
        for repo_id, repo_state in adjustments.items():
            if not isinstance(repo_id, str) or not isinstance(repo_state, Mapping):
                raise ValueError(f"{ADJUSTMENTS_KEY} contains invalid repository state")
            validated_state: dict[str, Any] = {}
            for family, stored in repo_state.items():
                if not isinstance(family, str) or not isinstance(stored, Mapping):
                    raise ValueError(f"{ADJUSTMENTS_KEY} contains invalid family state")
                action = stored.get("action")
                if not isinstance(action, str) or not action:
                    raise ValueError(f"{ADJUSTMENTS_KEY} contains an invalid action")
                validated_state[family] = dict(stored)
            adjustments[repo_id] = validated_state
        return adjustments

    @staticmethod
    def _applied_signals(payload: Mapping[str, Any]) -> dict[str, list[str]]:
        value = payload.get(APPLIED_SIGNALS_KEY, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"{APPLIED_SIGNALS_KEY} must be an object")
        signals: dict[str, list[str]] = {}
        for repo_id, actions in value.items():
            if not isinstance(repo_id, str) or not isinstance(actions, list):
                raise ValueError(f"{APPLIED_SIGNALS_KEY} contains invalid repository state")
            if any(not isinstance(action, str) or not action for action in actions):
                raise ValueError(f"{APPLIED_SIGNALS_KEY} contains an invalid action")
            signals[repo_id] = list(actions)
        return signals

    def apply(self, event: dict[str, Any]) -> ApplyResult:
        user_id = str(uuid.UUID(event["user_id"]))
        repo_id = str(uuid.UUID(event["repo_id"]))
        version = int(event["feedback_version"])
        canonical_user_id, legacy_user_id = user_point_ids(user_id)
        users = self.qdrant.retrieve(
            collection_name=USER_PROFILES_COLLECTION,
            ids=[canonical_user_id, legacy_user_id],
            with_payload=True,
            with_vectors=True,
        )
        if not users:
            raise LookupError(f"user vector {user_id} is not indexed")
        users_by_id = {str(point.id): point for point in users}
        user = users_by_id.get(canonical_user_id) or users_by_id.get(legacy_user_id)
        if user is None:
            raise LookupError(f"user vector {user_id} is not indexed")
        payload = dict(user.payload or {})
        last = int(payload.get("last_feedback_version") or 0)
        if version <= last:
            return ApplyResult("duplicate", last)
        if version != last + 1:
            return ApplyResult("gap", last)

        user_vector, user_vector_name = self._vector(user.vector, TARGET_VECTOR_NAME)
        dimension = len(user_vector)
        current = self._finite_vector(user_vector, dimension, label="user vector")
        accumulator = self._finite_vector(
            payload.get(LATENT_KEY, current), dimension, label="feedback latent vector"
        ).copy()
        adjustments = self._adjustments(payload)
        applied_signals = self._applied_signals(payload)
        repo_state = adjustments.setdefault(repo_id, {})
        definition = get_interaction(event["event_type"])

        def repository_vector() -> np.ndarray:
            canonical_repo_id, legacy_repo_id = repository_point_ids(repo_id)
            repos = self.qdrant.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=[canonical_repo_id, legacy_repo_id],
                with_payload=False,
                with_vectors=True,
            )
            if not repos:
                raise LookupError(f"repository vector {repo_id} is not indexed")
            repos_by_id = {str(point.id): point for point in repos}
            repo = repos_by_id.get(canonical_repo_id) or repos_by_id.get(legacy_repo_id)
            if repo is None:
                raise LookupError(f"repository vector {repo_id} is not indexed")
            value, _ = self._vector(repo.vector, QDRANT_VECTOR_NAME)
            return self._finite_vector(value, dimension, label="repository vector")

        def transition_delta() -> np.ndarray:
            return self._finite_vector(
                vector_delta(accumulator, repository_vector(), self._alpha(event)),
                dimension,
                label="feedback delta",
            )

        if definition.reversal_of:
            family = definition.state_family or ""
            stored = repo_state.get(family)
            if stored and stored.get("action") == definition.reversal_of:
                # Backend product state is authoritative. The initial v2 ML
                # policy records reversals as zero-alpha audit transitions:
                # clear active ML state, but do not claim that subtracting a
                # historical delta reconstructs the counterfactual vector.
                repo_state.pop(family, None)
        elif definition.state_family:
            family = definition.state_family
            stored = repo_state.get(family)
            if not stored or stored.get("action") != event["event_type"]:
                if stored:
                    accumulator -= self._finite_vector(
                        stored.get("delta"), dimension, label="stored feedback delta"
                    )
                delta = transition_delta()
                accumulator += delta
                repo_state[family] = {
                    "action": event["event_type"],
                    "delta": delta.tolist(),
                    "event_id": event["event_id"],
                }
        elif definition.apply_once:
            repo_signals = applied_signals.setdefault(repo_id, [])
            if event["event_type"] not in repo_signals:
                accumulator += transition_delta()
                repo_signals.append(event["event_type"])
        else:
            accumulator += transition_delta()

        if not repo_state:
            adjustments.pop(repo_id, None)
        norm = float(np.linalg.norm(accumulator))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("feedback produced an invalid vector")
        vector = (accumulator / norm).tolist()
        payload[LATENT_KEY] = accumulator.tolist()
        payload[ADJUSTMENTS_KEY] = adjustments
        payload[APPLIED_SIGNALS_KEY] = applied_signals
        payload["last_feedback_version"] = version
        payload["last_feedback_event_id"] = event["event_id"]
        stored_vector: Any = vector if user_vector_name is None else {user_vector_name: vector}
        self.qdrant.upsert(
            collection_name=USER_PROFILES_COLLECTION,
            points=[PointStruct(id=user.id, vector=stored_vector, payload=payload)],
            wait=True,
        )
        return ApplyResult("applied", version)


class OrderedFeedbackConsumer:
    def __init__(self, redis_client=None, applier: OrderedFeedbackApplier | None = None) -> None:
        self.redis = redis_client or _redis_client()
        self.applier = applier or OrderedFeedbackApplier()
        self.consumer = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4()}"
        try:
            self.redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _messages(self):
        try:
            claimed = self.redis.xautoclaim(STREAM, GROUP, self.consumer, 30_000, "0-0", count=20)
            for message_id, payload in claimed[1] if len(claimed) > 1 else []:
                yield message_id, payload
        except Exception as exc:
            logger.warning("pending feedback reclaim failed: %s", exc)
        for _, messages in self.redis.xreadgroup(GROUP, self.consumer, {STREAM: ">"}, count=20, block=1_000):
            yield from messages

    def _refresh_heartbeat(self) -> None:
        self.redis.set(
            CONSUMER_HEARTBEAT,
            self.consumer,
            ex=CONSUMER_HEARTBEAT_TTL_SECONDS,
        )

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        interval = max(0.1, CONSUMER_HEARTBEAT_TTL_SECONDS / 3)
        while not stop.wait(interval):
            try:
                self._refresh_heartbeat()
            except Exception as exc:
                logger.warning("feedback consumer heartbeat refresh failed: %s", exc)

    def run_once(self) -> int:
        self._refresh_heartbeat()
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_stop,),
            name="feedback-v2-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            processed = 0
            for message_id, payload in self._messages():
                user_id = payload.get("user_id")
                if not user_id:
                    self.redis.xack(STREAM, GROUP, message_id)
                    continue
                token = str(uuid.uuid4())
                lock = f"ml:feedback:v2:user-lock:{user_id}"
                if not self.redis.set(lock, token, nx=True, px=30_000):
                    continue
                try:
                    result = self.applier.apply(payload)
                    if result.status != "gap":
                        self.redis.xack(STREAM, GROUP, message_id)
                        processed += 1
                except (ValueError, KeyError, LookupError) as exc:
                    logger.error("feedback %s rejected: %s", message_id, exc)
                    self.redis.xadd(f"{STREAM}:dead", {**payload, "error": str(exc)})
                    self.redis.xack(STREAM, GROUP, message_id)
                finally:
                    self.redis.eval(RELEASE_LOCK_LUA, 1, lock, token)
            return processed
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
            self._refresh_heartbeat()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    consumer = OrderedFeedbackConsumer()
    while True:
        consumer.run_once()
