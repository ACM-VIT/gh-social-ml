"""Atomic, non-sensitive checkpoint state for resumable corpus ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .identity import repository_identity_key, normalize_repository_name


class CorpusCheckpoint:
    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "pending_persistence": [],
            "pending_index": [],
            "failures": {},
            "rejected": {},
            "last_run": {},
            "updated_at": None,
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid acquisition checkpoint {self.path}: {exc}"
            ) from exc
        if not isinstance(loaded, dict) or loaded.get("version") != self.VERSION:
            raise ValueError(
                f"Unsupported acquisition checkpoint format in {self.path}"
            )
        defaults = self._empty()
        defaults.update(loaded)
        return defaults

    @property
    def pending_persistence(self) -> list[str]:
        return list(self.data["pending_persistence"])

    @property
    def pending_index(self) -> list[str]:
        return list(self.data["pending_index"])

    def set_pending_persistence(self, names: list[str]) -> None:
        self.data["pending_persistence"] = self._normalized_unique(names)

    def add_pending_persistence(self, names: list[str]) -> None:
        self.set_pending_persistence([*self.pending_persistence, *names])

    def clear_pending_persistence(self, names: list[str]) -> None:
        self.data["pending_persistence"] = self._without(
            self.pending_persistence, names
        )

    def set_pending_index(self, names: list[str]) -> None:
        self.data["pending_index"] = self._normalized_unique(names)

    def add_pending_index(self, names: list[str]) -> None:
        self.set_pending_index([*self.pending_index, *names])

    def clear_pending_index(self, names: list[str]) -> None:
        self.data["pending_index"] = self._without(self.pending_index, names)

    def record_failure(self, repository: str, stage: str, error: object) -> None:
        name = normalize_repository_name(repository) or str(repository)
        self.data["failures"][name] = {
            "stage": stage,
            "error": str(error)[:500],
            "timestamp": self._now(),
        }

    def clear_failure(self, repository: str) -> None:
        key = repository_identity_key(repository)
        for existing in list(self.data["failures"]):
            if repository_identity_key(existing) == key:
                self.data["failures"].pop(existing, None)

    def record_rejection(self, repository: str, reasons: list[str]) -> None:
        name = normalize_repository_name(repository) or str(repository)
        self.data["rejected"][name] = {
            "reasons": [str(reason)[:300] for reason in reasons],
            "timestamp": self._now(),
        }

    def record_run(self, report: dict[str, Any]) -> None:
        self.data["last_run"] = {**report, "completed_at": self._now()}

    def save(self) -> None:
        self.data["updated_at"] = self._now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalized_unique(names: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in names:
            normalized = normalize_repository_name(value)
            key = repository_identity_key(normalized)
            if normalized and key not in seen:
                seen.add(key)
                unique.append(normalized)
        return unique

    @staticmethod
    def _without(existing: list[str], removed: list[str]) -> list[str]:
        removed_keys = {repository_identity_key(value) for value in removed}
        return [
            value
            for value in existing
            if repository_identity_key(value) not in removed_keys
        ]
