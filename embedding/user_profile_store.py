"""Qdrant storage adapter for Day-1 user profile embeddings."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from config import QDRANT_API_KEY, QDRANT_URL
from .embeddings import aggregate_vectors
from .vector_contract import (
    USER_PROFILE_COLLECTION_CONTRACT,
    canonical_backend_uuid,
    user_point_id,
    validate_embedding_vector,
)


class QdrantUserProfileStore:
    """Create, validate, and write the frozen unnamed user-vector collection."""

    def __init__(
        self,
        *,
        url: str = QDRANT_URL,
        api_key: str | None = QDRANT_API_KEY,
        timeout: int = 30,
        client: Any | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for user-vector storage. "
                "Install dependencies from requirements.txt."
            ) from exc

        self.models = models
        self.contract = USER_PROFILE_COLLECTION_CONTRACT
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout must be a positive integer number of seconds")
        self.timeout = timeout
        self.client = (
            client
            if client is not None
            else QdrantClient(url=url, api_key=api_key, timeout=timeout)
        )

    def ensure_collection(self) -> None:
        """Create the user collection when absent, then validate its contract."""
        if not self._collection_exists():
            try:
                self.client.create_collection(
                    collection_name=self.contract.collection_name,
                    vectors_config=self.models.VectorParams(
                        size=self.contract.vector_size,
                        distance=self._distance(),
                    ),
                )
            except Exception as exc:
                # Two simultaneous first-time onboarding requests may both see
                # an absent collection.  The loser still validates the winner's
                # schema before writing anything.
                if "already exists" not in str(exc).lower():
                    raise
        self.validate_collection()

    def validate_collection(self) -> None:
        """Reject named, wrong-sized, or wrong-distance user collections."""
        if not self._collection_exists():
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} does not exist."
            )
        info = self.client.get_collection(self.contract.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, Mapping):
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} must use one "
                "unnamed user vector."
            )
        if vectors is None:
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} has no vector configuration."
            )
        if int(vectors.size) != self.contract.vector_size:
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} has vector size "
                f"{vectors.size}, expected {self.contract.vector_size}."
            )
        if vectors.distance != self._distance():
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} uses distance "
                f"{vectors.distance}, expected {self._distance()}."
            )

    def upsert_user(
        self,
        user_id: str,
        vector: Sequence[float],
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Normalize and upsert one user using its deterministic point ID."""
        canonical_user_id = canonical_backend_uuid(user_id, field_name="user_id")
        point_id = user_point_id(canonical_user_id)
        validated = validate_embedding_vector(
            vector,
            expected_size=self.contract.vector_size,
            field_name=f"user embedding for {canonical_user_id}",
        )
        normalized = aggregate_vectors([validated])
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("user payload must be a mapping or None")
        user_payload = copy.deepcopy(dict(payload or {}))
        user_payload["user_id"] = canonical_user_id

        self.client.upsert(
            collection_name=self.contract.collection_name,
            points=[
                self.models.PointStruct(
                    id=point_id,
                    vector=normalized,
                    payload=user_payload,
                )
            ],
        )

    def _collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.contract.collection_name))
        try:
            self.client.get_collection(self.contract.collection_name)
            return True
        except Exception:
            return False

    def _distance(self):
        try:
            return getattr(self.models.Distance, self.contract.distance.upper())
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported user collection distance {self.contract.distance!r}"
            ) from exc
