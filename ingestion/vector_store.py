import os
import uuid
from typing import Optional
import numpy as np

from .config import COLLECTION_NAME, QDRANT_VECTOR_NAME

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    try:
        from qdrant_client.models import NamedVector
    except ImportError:
        NamedVector = None
    _QDRANT_OK = True
except ImportError:
    NamedVector = None
    _QDRANT_OK = False

_qdrant_instance: Optional[object] = None

def _resolve_qdrant_client(url=None, api_key=None):
    global _qdrant_instance
    if _qdrant_instance is None and _QDRANT_OK:
        url = url or os.getenv("QDRANT_URL")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        if url:
            _qdrant_instance = QdrantClient(url=url, api_key=api_key)
        else:
            _qdrant_instance = QdrantClient(":memory:")
    return _qdrant_instance

def _verify_qdrant_collection(client, target_dim: int):
    try:
        collections = {c.name for c in client.get_collections().collections}
        if COLLECTION_NAME not in collections:
            client.create_collection(
                COLLECTION_NAME,
                vectors_config={QDRANT_VECTOR_NAME: VectorParams(size=target_dim, distance=Distance.COSINE)},
            )
            print(f"[QDRANT] Created collection={COLLECTION_NAME} vector_name={QDRANT_VECTOR_NAME} vector_dim={target_dim} distance=COSINE")
            return

        diagnostics = _qdrant_collection_diagnostics(client)
        configured_dim = diagnostics.get("vector_size")
        if configured_dim and configured_dim != target_dim:
            print(
                f"[QDRANT WARNING] Collection vector dimension mismatch: "
                f"collection_dim={configured_dim} query_dim={target_dim}. "
                f"Neighbor retrieval may return zero results until the collection is rebuilt."
            )
    except Exception as e:
        print(f"[QDRANT WARNING] Collection verification failed: {e}")

def _index_qdrant_point(client, repo_id: str, vector: np.ndarray, metadata: dict):
    try:
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_id))
        vector_payload = _qdrant_point_vector(client, vector)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(id=point_uuid, vector=vector_payload, payload={"repo_id": repo_id, **metadata})
            ],
            wait=True,
        )
        collection_size = _qdrant_count_points(client)
        print(f"[QDRANT] Inserted point_id={point_uuid} repo_id={repo_id} collection_size={collection_size}")
    except Exception as e:
        print(f"\n[CRITICAL DB ERROR] Qdrant Insert Failed for {repo_id}: {e}")

def _qdrant_collection_diagnostics(client) -> dict:
    """Returns best-effort collection details across qdrant-client versions."""
    try:
        info = client.get_collection(COLLECTION_NAME)
        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        vector_size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)
        vector_names = []
        if isinstance(vectors, dict):
            vector_names = list(vectors.keys())
            selected = vectors.get(QDRANT_VECTOR_NAME) or next(iter(vectors.values()), None)
            vector_size = getattr(selected, "size", None)
            distance = getattr(selected, "distance", None)
        return {
            "vector_size": vector_size,
            "vector_names": vector_names,
            "vector_mode": "named" if vector_names else "unnamed",
            "distance": str(distance) if distance is not None else None,
            "points_count": getattr(info, "points_count", None),
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        }
    except Exception as e:
        return {"error": str(e)}

def _qdrant_count_points(client) -> int:
    try:
        return int(client.count(COLLECTION_NAME, exact=True).count)
    except TypeError:
        return int(client.count(COLLECTION_NAME).count)
    except Exception:
        return 0

def _qdrant_uses_named_vectors(client) -> bool:
    diagnostics = _qdrant_collection_diagnostics(client)
    return diagnostics.get("vector_mode") == "named"

def _qdrant_point_vector(client, vector: np.ndarray):
    if _qdrant_uses_named_vectors(client):
        return {QDRANT_VECTOR_NAME: vector.tolist()}
    return vector.tolist()

def _qdrant_query_vector(client, vector: np.ndarray):
    if _qdrant_uses_named_vectors(client):
        if NamedVector is not None:
            return NamedVector(name=QDRANT_VECTOR_NAME, vector=vector.tolist())
        return {"name": QDRANT_VECTOR_NAME, "vector": vector.tolist()}
    return vector.tolist()

def _query_qdrant_neighbors(client, query_vector: np.ndarray, *, limit: int = 20) -> list[dict]:
    """Queries Qdrant with compatibility fallbacks and payload diagnostics."""
    hits = []
    try:
        # Use client.search consistently
        q_vector = _qdrant_query_vector(client, query_vector)
        # qdrant_client>=1.9.0 expects tuple (name, vector) for named vectors in search
        if isinstance(q_vector, dict) and "name" in q_vector:
            q_vector = (q_vector["name"], q_vector["vector"])
        elif hasattr(q_vector, "name") and hasattr(q_vector, "vector"):
            q_vector = (q_vector.name, q_vector.vector)
            
        points = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=q_vector,
            limit=limit,
            with_payload=True,
        )
    except Exception as search_error:
        print(f"[QDRANT WARNING] Search API failed: {search_error}")
        return []

    for hit in points or []:
        payload = getattr(hit, "payload", None) or {}
        hits.append({
            "repo_id": payload.get("repo_id", ""),
            "sim": float(getattr(hit, "score", 0.0)),
            "category": payload.get("category", "General / Other"),
            "activity": payload.get("activity", 0.5),
            "tags": payload.get("tags", []),
        })
    return hits

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _query_internal_neighbors(query_vector: np.ndarray, corpus_history: list[dict], *, limit: int = 20) -> list[dict]:
    """Fallback ANN scan over in-process corpus history."""
    internal_hits = []
    for node in corpus_history:
        internal_hits.append({
            "repo_id": node["repo_id"],
            "sim": _cosine(query_vector, node["vector"]),
            "category": node.get("category", "General / Other"),
            "activity": node.get("activity", 0.5),
            "tags": node.get("tags", []),
        })
    internal_hits.sort(key=lambda hit: hit["sim"], reverse=True)
    return internal_hits[:limit]
