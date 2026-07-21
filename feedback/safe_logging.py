"""Bounded JSON logging for the standalone feedback worker.

The formatter deliberately ignores ``exc_info`` and redacts credential-shaped
content.  Feedback code logs stable error codes and identifiers, never raw
dependency exceptions, URLs, vectors, or request bodies.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any, Mapping


_MAX_MESSAGE_LENGTH = 512
_MAX_CONTEXT_FIELDS = 32
_MAX_CONTEXT_VALUE_LENGTH = 256
_SENSITIVE_KEY_PARTS = (
    "authorization",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "redis_url",
    "qdrant_url",
    "vector",
)
_CREDENTIAL_URL = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)([^/@\s]+)@")
_NAMED_SECRET = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*[^\s,;]+"
)


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "")
    text = " ".join(text.replace("\x00", "").splitlines())
    text = _CREDENTIAL_URL.sub(r"\1[redacted]@", text)
    text = _NAMED_SECRET.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    return text[:limit]


def _sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _context_value(key: str, value: Any) -> str | int | float | bool | None:
    if _sensitive_key(key):
        return "[redacted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(value, _MAX_CONTEXT_VALUE_LENGTH)


def _safe_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in list(value.items())[:_MAX_CONTEXT_FIELDS]:
        key = _bounded_text(raw_key, 64)
        if not key:
            continue
        result[key] = _context_value(key, raw_value)
    return result


class BoundedJsonFormatter(logging.Formatter):
    """Serialize one safe, bounded JSON object without exception tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": _bounded_text(record.name, 128),
            "message": _bounded_text(record.getMessage(), _MAX_MESSAGE_LENGTH),
        }
        context = getattr(record, "feedback_context", None)
        if context is None:
            context = getattr(record, "lock_context", None)
        safe_context = _safe_context(context)
        if safe_context:
            payload["context"] = safe_context
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def configure_feedback_worker_logging(level: str | int = "INFO") -> logging.Logger:
    """Route feedback package logs to worker stdout with the safe formatter."""

    selected_level: str | int = level
    if isinstance(level, str) and level.upper() not in {
        "CRITICAL",
        "ERROR",
        "WARNING",
        "INFO",
        "DEBUG",
    }:
        selected_level = "INFO"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(BoundedJsonFormatter())
    package_logger = logging.getLogger("feedback")
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(selected_level)
    package_logger.propagate = False
    return package_logger
