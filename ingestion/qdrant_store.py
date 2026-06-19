"""Qdrant storage adapter for repository embeddings."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable

from .config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_PAYLOAD_INDEX_FIELDS,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
)
from .repository_embedding import RepositoryEmbeddingResult

logger = logging.getLogger(__name__)


class QdrantRepositoryStore:
    """Creates and writes repository vectors to Qdrant."""

    def __init__(
        self,
        *,
        url: str = QDRANT_URL,
        api_key: str | None = QDRANT_API_KEY,
        collection_name: str = QDRANT_COLLECTION_NAME,
        vector_name: str = QDRANT_VECTOR_NAME,
        vector_size: int = REPOSITORY_EMBEDDING_DIM,
        distance: str = QDRANT_DISTANCE,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for vector storage. "
                "Install dependencies from requirements.txt."
            ) from exc

        self.client = QdrantClient(url=url, api_key=api_key)
        self.models = models
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.vector_size = vector_size
        self.distance = distance

    def ensure_collection(self) -> None:
        """Create or validate the repository collection and payload indexes."""
        # The below check is for safe startup: existing collections are
        # validated instead of recreated, so stored vectors are not dropped.
        if not self._collection_exists():
            distance = self._distance()
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.vector_name: self.models.VectorParams(
                        size=self.vector_size,
                        distance=distance,
                    )
                },
            )
            logger.info("Created Qdrant collection: %s", self.collection_name)
        else:
            self._validate_collection()

        for field_name in QDRANT_PAYLOAD_INDEX_FIELDS:
            self._create_payload_index(field_name)

    def validate_collection(self) -> None:
        """Validate the configured collection without creating indexes."""
        if not self._collection_exists():
            raise ValueError(f"Qdrant collection {self.collection_name!r} does not exist.")
        self._validate_collection()

    def upsert(self, results: Iterable[RepositoryEmbeddingResult]) -> None:
        """Upsert embedding results into Qdrant."""
        points = []
        for result in results:
            # The below deterministic ID is for safe re-runs; the same repo is
            # updated instead of inserted as a duplicate vector.
            points.append(
                self.models.PointStruct(
                    id=self._point_id(result.repo_id),
                    vector={self.vector_name: result.final_embedding},
                    payload=result.payload,
                )
            )
        if not points:
            return
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Upserted %d repository vectors into %s", len(points), self.collection_name)

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 5,
        with_vectors: bool = False,
        exact: bool = True,
    ) -> list[dict]:
        """Search Qdrant by final repository embedding vector."""
        # The below query uses the named vector configured for repository
        # embeddings, so search targets the final repo embedding field. Exact
        # search is the default for evaluation-grade nearest-neighbor results.
        search_params = self.models.SearchParams(exact=exact)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            using=self.vector_name,
            search_params=search_params,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "repo_id": payload.get("repo_id"),
                    "payload": payload,
                    "vector": point.vector if with_vectors else None,
                }
            )
        return results

    def list_points(self, *, limit: int = 100, with_vectors: bool = True) -> list[dict]:
        """Load repository points from Qdrant for offline evaluation."""
        points: list[dict] = []
        offset = None
        while len(points) < limit:
            # The below scroll call is for evaluation/reporting workflows that
            # need existing payloads and vectors from Qdrant without a query.
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=min(100, limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=[self.vector_name] if with_vectors else False,
            )
            if not records:
                break
            for record in records:
                payload = record.payload or {}
                vector = None
                if with_vectors:
                    vector = self._extract_vector(record.vector)
                points.append(
                    {
                        "id": str(record.id),
                        "repo_id": payload.get("repo_id"),
                        "payload": payload,
                        "vector": vector,
                    }
                )
            if offset is None:
                break
        return points

    def _extract_vector(self, vector_data) -> list[float] | None:
        if vector_data is None:
            return None
        if isinstance(vector_data, dict):
            vector = vector_data.get(self.vector_name)
            return list(vector) if vector is not None else None
        return list(vector_data)

    def _collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.collection_name))
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _validate_collection(self) -> None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        vector_config = vectors.get(self.vector_name) if isinstance(vectors, dict) else vectors
        if vector_config is None:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} does not define vector "
                f"{self.vector_name!r}."
            )
        if int(vector_config.size) != int(self.vector_size):
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} vector {self.vector_name!r} "
                f"has size {vector_config.size}, expected {self.vector_size}."
            )
        expected_distance = self._distance()
        if vector_config.distance != expected_distance:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} vector {self.vector_name!r} "
                f"uses distance {vector_config.distance}, expected {expected_distance}."
            )

    def _create_payload_index(self, field_name: str) -> None:
        # The below schema selection is for keeping payload indexes aligned with
        # the payload fields emitted by build_vector_payload.
        schema = self.models.PayloadSchemaType.KEYWORD
        if field_name in {"star_count"}:
            schema = self.models.PayloadSchemaType.INTEGER
        if field_name in {"updated_at", "pushed_at"}:
            schema = self.models.PayloadSchemaType.DATETIME
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception as exc:
            logger.debug("Payload index skipped for %s: %s", field_name, exc)

    @staticmethod
    def _point_id(repo_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"github:{repo_id}"))

    def _distance(self):
        try:
            return getattr(self.models.Distance, self.distance.upper())
        except AttributeError as exc:
            allowed = ", ".join(item.name for item in self.models.Distance)
            raise ValueError(f"Unsupported Qdrant distance {self.distance!r}. Use one of: {allowed}") from exc
