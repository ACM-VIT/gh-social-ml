"""Stable identifiers and data contracts for the vector platform.

This module is the shared boundary published by Person 2. Downstream code
must pass backend-issued UUIDs and use the helpers here whenever it needs the
corresponding Qdrant point ID.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real
from types import MappingProxyType
from typing import Any

from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    USER_PROFILES_COLLECTION_NAME,
)


@dataclass(frozen=True, slots=True)
class VectorCollectionContract:
    """Immutable description of one Qdrant collection."""

    collection_name: str
    vector_name: str | None
    vector_size: int
    distance: str
    model_name: str


REPOSITORY_COLLECTION_CONTRACT = VectorCollectionContract(
    collection_name=QDRANT_COLLECTION_NAME,
    vector_name=QDRANT_VECTOR_NAME,
    vector_size=REPOSITORY_EMBEDDING_DIM,
    distance=QDRANT_DISTANCE,
    model_name=REPOSITORY_EMBEDDING_MODEL,
)

# The existing user_profiles collection stores one unnamed vector per user.
# Keeping that choice explicit prevents consumers from guessing a vector name.
USER_PROFILE_COLLECTION_CONTRACT = VectorCollectionContract(
    collection_name=USER_PROFILES_COLLECTION_NAME,
    vector_name=None,
    vector_size=REPOSITORY_EMBEDDING_DIM,
    distance=QDRANT_DISTANCE,
    model_name=REPOSITORY_EMBEDDING_MODEL,
)

# Stable public names for Qdrant-only discovery channels.  Person 3 can select
# a channel without duplicating knowledge of the underlying payload field.
REPOSITORY_DISCOVERY_CHANNELS: Mapping[str, str] = MappingProxyType(
    {
        "trend": "trend_velocity",
        "activity": "activity_score",
        "popularity": "star_count",
        "freshness": "pushed_at",
        "quality": "doc_quality",
    }
)


def _canonical_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    canonical = value.strip()
    if not canonical:
        raise ValueError(f"{field_name} must be a non-empty string")
    return canonical


def canonical_backend_uuid(value: str, *, field_name: str) -> str:
    """Validate and canonicalize a backend-issued UUID identifier."""
    canonical = _canonical_identifier(value, field_name=field_name)
    try:
        parsed = uuid.UUID(canonical)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid backend-issued UUID") from exc
    return str(parsed)


def repository_point_id(repo_id: str) -> str:
    """Return the canonical backend repository UUID used as the Qdrant point ID."""
    return canonical_backend_uuid(repo_id, field_name="repo_id")


def legacy_repository_point_id(repo_id: str) -> str:
    """Return the point ID emitted by the pre-v2 repository helper."""
    canonical = canonical_backend_uuid(repo_id, field_name="repo_id")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"github:{canonical}"))


def repository_point_ids(repo_id: str) -> tuple[str, str]:
    """Return canonical then legacy IDs for upgrade-safe reads."""
    canonical = repository_point_id(repo_id)
    return canonical, legacy_repository_point_id(canonical)


def user_point_id(user_id: str) -> str:
    """Return the canonical backend user UUID used as the Qdrant point ID."""
    return canonical_backend_uuid(user_id, field_name="user_id")


def legacy_user_point_id(user_id: str) -> str:
    """Return the point ID emitted by the pre-v2 user helper."""
    canonical = canonical_backend_uuid(user_id, field_name="user_id")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"user:{canonical}"))


def user_point_ids(user_id: str) -> tuple[str, str]:
    """Return canonical then legacy IDs for upgrade-safe reads."""
    canonical = user_point_id(user_id)
    return canonical, legacy_user_point_id(canonical)


def resolve_repository_identity(repo: Mapping[str, Any]) -> tuple[str, str]:
    """Return the canonical ``(repo_id, full_name)`` for a repository input.

    ``repo_id`` must be the UUID issued by the backend ingestion API. Names,
    URLs, GitHub handles, and ``full_name`` are attributes only.
    """
    if not isinstance(repo, Mapping):
        raise TypeError("repository payload must be a mapping")

    raw_repo_id = repo.get("repo_id") or repo.get("id")
    repo_id = canonical_backend_uuid(raw_repo_id, field_name="repo_id")

    raw_full_name = repo.get("full_name")
    full_name = _canonical_identifier(raw_full_name, field_name="full_name")

    owner, separator, name = full_name.partition("/")
    if separator != "/" or not owner or not name or "/" in name:
        raise ValueError("full_name must use the GitHub 'owner/repository' format")
    return repo_id, full_name


def validate_embedding_vector(
    vector: Sequence[Real],
    *,
    expected_size: int,
    field_name: str = "embedding",
) -> list[float]:
    """Validate and return a JSON-safe finite embedding vector."""
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise TypeError(f"{field_name} must be a sequence of numbers")
    if len(vector) != expected_size:
        raise ValueError(
            f"{field_name} has dimension {len(vector)}, expected {expected_size}."
        )

    validated: list[float] = []
    for index, item in enumerate(vector):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{field_name}[{index}] must be a real number")
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"{field_name}[{index}] must be finite")
        validated.append(value)
    return validated


# Every repository point publishes these keys.  Nullable values are still
# present so retrieval and ranking consumers receive one predictable shape.
REPOSITORY_PAYLOAD_FIELD_TYPES: Mapping[str, type | tuple[type, ...]] = MappingProxyType(
    {
        "repo_id": str,
        "github_id": (str, type(None)),
        "full_name": str,
        "html_url": (str, type(None)),
        "description": str,
        "primary_language": str,
        "languages": list,
        "topics": list,
        "star_count": int,
        "fork_count": int,
        "open_issues_count": int,
        "readme_length": int,
        "readme_chunks": int,
        "pushed_days_ago": int,
        "delta_3d": int,
        "delta_7d": int,
        "delta_30d": int,
        "mentionable_users_count": int,
        "created_at": (str, type(None)),
        "updated_at": (str, type(None)),
        "pushed_at": (str, type(None)),
        "discovery_category": (str, type(None)),
        "discovery_band": (str, type(None)),
        "category": str,
        "tags": list,
        "doc_quality": (int, float),
        "code_health": (int, float),
        "activity_score": (int, float),
        "trend_velocity": (int, float),
        "embedding_dim": int,
        "embedding_model": str,
        "embedding_version": str,
        "content_version": int,
        "content_hash": str,
        "model_version": str,
        "indexed_at": str,
        "source_hash": str,
    }
)

REPOSITORY_PAYLOAD_REQUIRED_FIELDS = tuple(REPOSITORY_PAYLOAD_FIELD_TYPES)

_INTEGER_FIELDS = {
    "star_count",
    "fork_count",
    "open_issues_count",
    "readme_length",
    "readme_chunks",
    "pushed_days_ago",
    "mentionable_users_count",
    "embedding_dim",
    "delta_3d",
    "delta_7d",
    "delta_30d",
    "content_version",
}
_NON_NEGATIVE_INTEGER_FIELDS = _INTEGER_FIELDS - {"delta_3d", "delta_7d", "delta_30d"}
_FINITE_NUMBER_FIELDS = {
    "doc_quality",
    "code_health",
    "activity_score",
    "trend_velocity",
}
_TIMESTAMP_FIELDS = {"created_at", "updated_at", "pushed_at", "indexed_at"}
_STRING_LIST_FIELDS = {"languages", "topics", "tags"}


def validate_repository_payload(payload: Mapping[str, Any]) -> None:
    """Validate one emitted repository payload against the frozen contract."""
    if not isinstance(payload, Mapping):
        raise TypeError("repository payload must be a mapping")

    missing = [field for field in REPOSITORY_PAYLOAD_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"repository payload is missing required fields: {', '.join(missing)}")

    for field_name, expected_type in REPOSITORY_PAYLOAD_FIELD_TYPES.items():
        value = payload[field_name]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"repository payload field {field_name!r} has type "
                f"{type(value).__name__}, expected {_type_label(expected_type)}"
            )

    resolve_repository_identity(payload)

    github_id = payload["github_id"]
    if github_id is not None and (not github_id or not github_id.isdecimal()):
        raise ValueError("repository payload field 'github_id' must be a decimal string")

    for field_name in _INTEGER_FIELDS:
        value = payload[field_name]
        if isinstance(value, bool):
            raise TypeError(f"repository payload field {field_name!r} must be an integer")

    for field_name in _NON_NEGATIVE_INTEGER_FIELDS:
        if payload[field_name] < 0:
            raise ValueError(f"repository payload field {field_name!r} must be non-negative")

    for field_name in _FINITE_NUMBER_FIELDS:
        value = payload[field_name]
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"repository payload field {field_name!r} must be finite")

    for field_name in _TIMESTAMP_FIELDS:
        value = payload[field_name]
        if value is not None:
            _parse_iso_timestamp(value, field_name=field_name)

    for field_name in _STRING_LIST_FIELDS:
        if any(not isinstance(item, str) for item in payload[field_name]):
            raise TypeError(f"repository payload field {field_name!r} must contain strings")

    if payload["embedding_dim"] != REPOSITORY_COLLECTION_CONTRACT.vector_size:
        raise ValueError(
            "repository payload embedding_dim does not match the collection contract"
        )
    for field_name in (
        "embedding_model",
        "embedding_version",
        "content_hash",
        "model_version",
        "source_hash",
    ):
        if not payload[field_name].strip():
            raise ValueError(f"repository payload field {field_name!r} must be non-empty")


def _parse_iso_timestamp(value: str, *, field_name: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp")


def _type_label(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(item.__name__ for item in expected_type)
    return expected_type.__name__


def repository_payload_defaults() -> dict[str, object]:
    """Return fresh defaults for optional repository metadata fields."""
    return {
        "html_url": None,
        "github_id": None,
        "description": "",
        "primary_language": "Unknown",
        "languages": [],
        "topics": [],
        "star_count": 0,
        "fork_count": 0,
        "open_issues_count": 0,
        "readme_length": 0,
        "readme_chunks": 0,
        "pushed_days_ago": 999,
        "delta_3d": 0,
        "delta_7d": 0,
        "delta_30d": 0,
        "mentionable_users_count": 0,
        "created_at": None,
        "updated_at": None,
        "pushed_at": None,
        "discovery_category": None,
        "discovery_band": None,
        "category": "Unknown",
        "tags": [],
        "doc_quality": 0.0,
        "code_health": 0.0,
        "activity_score": 0.0,
        "trend_velocity": 0.0,
        "embedding_dim": REPOSITORY_EMBEDDING_DIM,
        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_version": REPOSITORY_EMBEDDING_VERSION,
        "content_version": 0,
        "model_version": REPOSITORY_EMBEDDING_MODEL,
    }
