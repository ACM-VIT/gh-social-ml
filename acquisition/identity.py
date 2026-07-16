"""Canonical repository identity helpers used across offline ingestion stages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def normalize_repository_name(value: object) -> str:
    """Return a validated ``owner/repository`` name while preserving casing."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().strip("/")
    if cleaned.lower().startswith("https://github.com/"):
        cleaned = cleaned[len("https://github.com/") :].strip("/")
    owner, separator, name = cleaned.partition("/")
    if not separator or not owner.strip() or not name.strip() or "/" in name:
        return ""
    return f"{owner.strip()}/{name.strip()}"


def repository_identity_key(value: object) -> str:
    """Return the case-insensitive identity key GitHub repository names use."""
    return normalize_repository_name(value).casefold()


def repository_name_from_candidate(candidate: Any) -> str:
    """Extract a repository name from a discovery candidate."""
    if isinstance(candidate, str):
        return normalize_repository_name(candidate)
    if isinstance(candidate, dict):
        return normalize_repository_name(candidate.get("full_name"))
    return ""


def deduplicate_candidates(candidates: Iterable[Any]) -> tuple[list[Any], int]:
    """Keep the first valid candidate for each case-insensitive repository name."""
    unique: list[Any] = []
    seen: set[str] = set()
    removed = 0
    for candidate in candidates:
        key = repository_identity_key(repository_name_from_candidate(candidate))
        if not key or key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(candidate)
    return unique, removed
