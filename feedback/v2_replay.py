"""Safe, idempotent replay for v2 feedback dead-letter entries.

Dry-run is the default.  Terminal-invalid events require an explicit override;
normal operator replay is intended for retry-exhausted dependency/gap events
after their prerequisite data has been repaired.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from .v2 import FeedbackStreamFullError, _redis_client
from .v2_settings import V2FeedbackSettings


_STREAM_ID = re.compile(r"^\d+-\d+$")
_EVENT_FIELDS = {
    "event_id",
    "user_id",
    "repo_id",
    "feedback_version",
    "event_type",
    "dwell_ms",
    "occurred_at",
}

REPLAY_LUA = """
if redis.call('exists', KEYS[1]) == 1 then
  return 'duplicate:' .. redis.call('get', KEYS[1])
end
if #redis.call('xrange', KEYS[4], ARGV[3], ARGV[3], 'COUNT', 1) == 0 then
  return 'source_missing'
end
if redis.call('xlen', KEYS[3]) >= tonumber(ARGV[2]) then
  return 'overloaded'
end
local result = redis.pcall('xadd', KEYS[3], '*', unpack(ARGV, 5))
if type(result) == 'table' and result.err then
  return redis.error_reply(result.err)
end
redis.call('set', KEYS[1], ARGV[4] .. '|' .. result, 'EX', ARGV[1])
redis.call('del', KEYS[2])
redis.call('xdel', KEYS[4], ARGV[3])
return result
"""


@dataclass(frozen=True, slots=True)
class ReplayResult:
    status: str
    source_id: str
    event_id: str
    replay_message_id: str | None = None


class DeadLetterReplayer:
    def __init__(
        self,
        redis_client: Any | None = None,
        settings: V2FeedbackSettings | None = None,
    ) -> None:
        self.settings = settings or V2FeedbackSettings.from_env()
        self._owns_redis = redis_client is None
        self.redis = redis_client or _redis_client(self.settings)

    @staticmethod
    def _validate_source_id(source_id: str) -> None:
        if not _STREAM_ID.fullmatch(source_id):
            raise ValueError("source_id must be a Redis stream ID such as 123456789-0")

    def _replay_key(self, source_id: str) -> str:
        return f"{self.settings.stream_name}:replayed:{source_id}"

    def _completed_replay(self, source_id: str) -> ReplayResult | None:
        marker = self.redis.get(self._replay_key(source_id))
        if not marker:
            return None
        marker_text = marker.decode("utf-8") if isinstance(marker, bytes) else str(marker)
        event_id, separator, message_id = marker_text.partition("|")
        return ReplayResult(
            "duplicate",
            source_id,
            event_id if separator else "unknown",
            message_id if separator else None,
        )

    def _load(self, source_id: str) -> Mapping[str, Any]:
        self._validate_source_id(source_id)
        rows = self.redis.xrange(
            self.settings.dead_letter_stream,
            min=source_id,
            max=source_id,
            count=1,
        )
        if not rows:
            raise LookupError("dead-letter entry was not found")
        raw_id = rows[0][0]
        stored_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
        if stored_id != source_id:
            raise LookupError("dead-letter entry was not found")
        payload = rows[0][1]
        if not isinstance(payload, Mapping):
            raise ValueError("dead-letter entry has an invalid payload")
        return payload

    @staticmethod
    def _event(payload: Mapping[str, Any]) -> dict[str, str]:
        event = {
            field: str(payload[field])
            for field in _EVENT_FIELDS
            if field in payload and payload[field] not in {None, ""}
        }
        for required in ("event_id", "user_id", "repo_id", "feedback_version", "event_type"):
            if required not in event:
                raise ValueError(f"dead-letter event is missing {required}")
        for field in ("event_id", "user_id", "repo_id"):
            try:
                event[field] = str(uuid.UUID(event[field]))
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"dead-letter event has an invalid {field}") from exc
        try:
            version = int(event["feedback_version"])
        except ValueError as exc:
            raise ValueError("dead-letter event has an invalid feedback_version") from exc
        if version < 1:
            raise ValueError("dead-letter event has an invalid feedback_version")
        event["feedback_version"] = str(version)
        return event

    def replay(
        self,
        source_id: str,
        *,
        execute: bool = False,
        allow_terminal: bool = False,
    ) -> ReplayResult:
        self._validate_source_id(source_id)
        completed = self._completed_replay(source_id)
        if completed is not None:
            return completed
        payload = self._load(source_id)
        retryable = str(payload.get("retryable", "0")).lower() in {"1", "true", "yes"}
        cursor_advanced = str(payload.get("cursor_advanced", "0")).lower() in {
            "1",
            "true",
            "yes",
        }
        if not retryable and cursor_advanced:
            raise ValueError(
                "cursor-advanced terminal feedback cannot be replayed; issue a "
                "corrected compensating event at the next backend feedback version"
            )
        if not retryable and not allow_terminal:
            raise ValueError(
                "terminal-invalid feedback is not replayable without allow_terminal=True"
            )
        event = self._event(payload)
        if not execute:
            return ReplayResult("dry_run", source_id, event["event_id"])

        fields: list[str] = []
        for key, value in event.items():
            fields.extend([key, value])
        replay_key = self._replay_key(source_id)
        attempts_key = f"{self.settings.stream_name}:attempts:{event['event_id']}"
        result = self.redis.eval(
            REPLAY_LUA,
            4,
            replay_key,
            attempts_key,
            self.settings.stream_name,
            self.settings.dead_letter_stream,
            str(self.settings.idempotency_ttl_seconds),
            str(self.settings.stream_maxlen),
            source_id,
            event["event_id"],
            *fields,
        )
        result_text = result.decode("utf-8") if isinstance(result, bytes) else str(result)
        if result_text.startswith("duplicate:"):
            marker = result_text.removeprefix("duplicate:")
            marker_event_id, separator, message_id = marker.partition("|")
            return ReplayResult(
                "duplicate",
                source_id,
                marker_event_id or event["event_id"],
                message_id if separator else None,
            )
        if result_text == "overloaded":
            raise FeedbackStreamFullError(
                accepted=0,
                duplicates=0,
                failed_event_id=event["event_id"],
                capacity=self.settings.stream_maxlen,
            )
        if result_text == "source_missing":
            raise LookupError("dead-letter entry disappeared before replay")
        return ReplayResult("replayed", source_id, event["event_id"], result_text)

    def close(self) -> None:
        if self._owns_redis and hasattr(self.redis, "close"):
            self.redis.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay one v2 feedback DLQ entry")
    parser.add_argument("--source-id", required=True, help="Redis DLQ stream entry ID")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually enqueue the event; without this flag the command is a dry run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="explicitly request the default non-mutating preview",
    )
    parser.add_argument(
        "--allow-terminal",
        action="store_true",
        help="allow replay of a terminal-invalid entry after an operator repair",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry-run are mutually exclusive")
    replayer = DeadLetterReplayer()
    try:
        result = replayer.replay(
            args.source_id,
            execute=args.execute,
            allow_terminal=args.allow_terminal,
        )
        print(json.dumps(result.__dict__ if hasattr(result, "__dict__") else {
            "status": result.status,
            "source_id": result.source_id,
            "event_id": result.event_id,
            "replay_message_id": result.replay_message_id,
        }, sort_keys=True))
        return 0
    finally:
        replayer.close()


if __name__ == "__main__":
    raise SystemExit(main())
