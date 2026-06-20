
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from qdrant_client import QdrantClient

from ingestion.config import (
    QDRANT_API_KEY,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
    QDRANT_COLLECTION_NAME,
)
from ingestion.qdrant_store import QdrantRepositoryStore
from user_onboarding import USER_PROFILES_COLLECTION

logger = logging.getLogger("pipeline.retrieval")

BATCH_SIZE = 15
NUM_BATCHES = 3
TOTAL_LIMIT = BATCH_SIZE * NUM_BATCHES  # 45

# ── Postgres table for caching recommendation batches ─────────────────────────

_RECOMMENDATIONS_TABLE = "user_recommendation_batches"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_RECOMMENDATIONS_TABLE} (
    user_id      VARCHAR(255) PRIMARY KEY,
    batch_data   JSONB NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_UPSERT_SQL = f"""
INSERT INTO {_RECOMMENDATIONS_TABLE} (user_id, batch_data)
VALUES (%s, CAST(%s AS jsonb))
ON CONFLICT (user_id) DO UPDATE SET
    batch_data = EXCLUDED.batch_data,
    updated_at = CURRENT_TIMESTAMP;
"""

_SELECT_SQL = f"""
SELECT batch_data FROM {_RECOMMENDATIONS_TABLE} WHERE user_id = %s;
"""


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalEngine:
    """Produces post-onboarding repo recommendations for a single user.

    Usage::

        engine = RetrievalEngine()
        result = engine.fetch_onboarding_batches("user_123")
        # result == {"batch_1": [...15 items...], "batch_2": [...], "batch_3": [...]}
    """

    def __init__(
        self,
        *,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self._url = qdrant_url or QDRANT_URL
        self._api_key = qdrant_api_key or QDRANT_API_KEY

        # Direct client for user_profiles (unnamed-vector collection)
        self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=30.0)

        # Repo store handles named-vector search against osiris_research_corpus
        self._repo_store = QdrantRepositoryStore(
            url=self._url,
            api_key=self._api_key,
        )

        # Postgres connector — lazy, may be None if DATABASE_URL is not set
        self._db = None

    # ── Lazy Postgres connector ───────────────────────────────────────────────

    @property
    def db(self):
        """Lazy-load the database connector to avoid import-time failures."""
        if self._db is None:
            try:
                from database import PostgreSQLConnector
                self._db = PostgreSQLConnector()
            except Exception as exc:
                logger.warning("Could not initialize PostgreSQLConnector: %s", exc)
                self._db = False  # sentinel: tried and failed
        return self._db if self._db is not False else None

    # ── Core retrieval ────────────────────────────────────────────────────────

    def fetch_onboarding_batches(self, user_id: str) -> dict[str, list[dict[str, Any]]]:
        # ── 1. Check cache ────────────────────────────────────────────────────
        cached = self._load_cached_batches(user_id)
        if cached is not None:
            logger.info("Returning cached recommendation batches for '%s'.", user_id)
            return cached

        # ── 2. Get user embedding from user_profiles ──────────────────────────
        user_vector = self._get_user_vector(user_id)

        # ── 3. Single Qdrant query — see module docstring for why one query ───
        all_matches = self._repo_store.search(
            user_vector,
            limit=TOTAL_LIMIT,
            exact=True,
        )

        # ── 4. Project each match to a slim downstream-friendly dict ──────────
        items = []
        for match in all_matches:
            payload = match.get("payload") or {}
            items.append({
                "repo_id":            match.get("repo_id") or payload.get("repo_id"),
                "cosine_score":       float(match.get("score", 0.0)),
                "category":           payload.get("category") or payload.get("discovery_category") or "Unknown",
                "primary_language":   payload.get("primary_language") or "Unknown",
                "description":        payload.get("description") or "",
                "star_count":         int(payload.get("star_count") or 0),
                "topics":             payload.get("topics") or [],
                "html_url":           payload.get("html_url") or "",
                "discovery_category": payload.get("discovery_category") or "",
                "discovery_band":     payload.get("discovery_band") or "",
            })

        # ── 5. In-memory slice into three ranked batches ──────────────────────
        # Qdrant returns results pre-sorted by descending cosine score.
        # batch_1 = highest similarity, batch_3 = lowest.  Do NOT paginate
        # with separate queries — see module docstring.
        batches = {
            "batch_1": items[0:BATCH_SIZE],
            "batch_2": items[BATCH_SIZE : BATCH_SIZE * 2],
            "batch_3": items[BATCH_SIZE * 2 : BATCH_SIZE * 3],
        }

        # ── 6. Persist to Postgres ────────────────────────────────────────────
        self._persist_batches(user_id, batches)

        logger.info(
            "Generated onboarding batches for '%s': %d / %d / %d items.",
            user_id,
            len(batches["batch_1"]),
            len(batches["batch_2"]),
            len(batches["batch_3"]),
        )
        return batches

    # ── Qdrant helpers ────────────────────────────────────────────────────────

    def _get_user_vector(self, user_id: str) -> list[float]:
        """Retrieve the user's interest embedding from the user_profiles collection.

        The point ID is a deterministic UUID5 matching the scheme in
        ``user_onboarding.py:save_to_qdrant``.
        """
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"user:{user_id}"))

        response = self._client.retrieve(
            collection_name=USER_PROFILES_COLLECTION,
            ids=[point_uuid],
            with_vectors=True,
        )

        if not response:
            raise ValueError(
                f"User '{user_id}' (point {point_uuid}) not found in "
                f"Qdrant collection '{USER_PROFILES_COLLECTION}'."
            )

        point = response[0]

        # user_profiles stores unnamed vectors (list), but handle the
        # named-vector case defensively in case the schema evolves.
        if isinstance(point.vector, dict):
            # Named vectors: take the first (and only) vector value
            vectors = list(point.vector.values())
            if not vectors:
                raise ValueError(f"User '{user_id}' has an empty named-vector dict.")
            return list(vectors[0])

        return list(point.vector)

    # ── Postgres persistence ──────────────────────────────────────────────────

    def _ensure_recommendations_table(self, conn) -> None:
        """Create the recommendation batches table if it doesn't exist."""
        cursor = conn.cursor()
        try:
            cursor.execute(_CREATE_TABLE_SQL)
            conn.commit()
        except Exception as exc:
            logger.warning("Could not create %s table: %s", _RECOMMENDATIONS_TABLE, exc)
            conn.rollback()

    def _persist_batches(
        self,
        user_id: str,
        batches: dict[str, list[dict[str, Any]]],
    ) -> bool:
        """Upsert the recommendation batches into Postgres.

        Follows the same connection / SAVEPOINT pattern used by
        ``database.connector.PostgreSQLConnector.upsert_repositories``.
        """
        db = self.db
        if db is None or not db.enabled:
            logger.info("DATABASE_URL not set; skipping batch persistence.")
            return False

        conn = None
        try:
            conn = db.connect()
            self._ensure_recommendations_table(conn)

            cursor = conn.cursor()
            batch_json = json.dumps(batches, default=str)

            cursor.execute("SAVEPOINT batch_upsert;")
            cursor.execute(_UPSERT_SQL, (user_id, batch_json))
            cursor.execute("RELEASE SAVEPOINT batch_upsert;")

            conn.commit()
            logger.info("Persisted recommendation batches for '%s' to Postgres.", user_id)
            return True

        except Exception as exc:
            logger.error("Failed to persist batches for '%s': %s", user_id, exc)
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _load_cached_batches(
        self,
        user_id: str,
    ) -> dict[str, list[dict[str, Any]]] | None:
        """Load previously persisted batches from Postgres, or None if missing."""
        db = self.db
        if db is None or not db.enabled:
            return None

        conn = None
        try:
            conn = db.connect()
            self._ensure_recommendations_table(conn)

            cursor = conn.cursor()
            cursor.execute(_SELECT_SQL, (user_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            data = row[0]
            # pg8000 may return the JSONB column as a string or as a dict
            if isinstance(data, str):
                data = json.loads(data)

            # Validate shape
            required_batches = {"batch_1", "batch_2", "batch_3"}
            if (
                isinstance(data, dict)
                and required_batches.issubset(data)
                and all(isinstance(data[key], list) for key in required_batches)
            ):
                logger.info("Loaded cached batches for '%s' from Postgres.", user_id)
                return data

            return None

        except Exception as exc:
            logger.debug("Cache lookup failed for '%s': %s", user_id, exc)
            return None

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    # ── Utility: list onboarded users ─────────────────────────────────────────

    def list_onboarded_users(self, batch_size: int = 100) -> list[dict[str, Any]]:
        """Scroll the user_profiles collection and return all user metadata.

        Returns an empty list if the collection doesn't exist yet
        (no users have been onboarded). Uses Qdrant's scroll API to gracefully
        handle any number of users.
        """
        users = []
        next_offset = None

        while True:
            try:
                records, next_offset = self._client.scroll(
                    collection_name=USER_PROFILES_COLLECTION,
                    limit=batch_size,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                # Collection doesn't exist — no users onboarded yet
                return users

            for record in records:
                payload = record.payload or {}
                users.append({
                    "point_id": str(record.id),
                    "user_id": payload.get("user_id", "unknown"),
                    "skills": payload.get("skills", []),
                    "interests": payload.get("interests", []),
                })

            if next_offset is None:
                break

        return users



# ══════════════════════════════════════════════════════════════════════════════
#  MANUAL TEST
# ══════════════════════════════════════════════════════════════════════════════

def _print_batch(name: str, batch: list[dict[str, Any]]) -> None:
    """Pretty-print one batch for eyeball inspection."""
    if not batch:
        print(f"  {name}: (empty)")
        return
    print(f"  {name}  ({len(batch)} repos)")
    print(f"  {'#':<3} {'Score':>7}  {'Repo':<42} {'Category'}")
    print(f"  {'-'*3} {'-'*7}  {'-'*42} {'-'*28}")
    for i, item in enumerate(batch, 1):
        print(
            f"  {i:<3} {item['cosine_score']:>7.4f}  "
            f"{(item['repo_id'] or '?'):<42} "
            f"{item['category']}"
        )
    print()


def main() -> None:
    """Run retrieval for onboarded users and print batches.

    If no users are onboarded, automatically onboards 5 sample users
    with diverse interests so the test is self-contained.

    This is a quick manual sanity check: batch_1 scores should be
    consistently higher than batch_3 scores.  If they're not, the
    Qdrant query itself is broken (not the batching logic).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    engine = RetrievalEngine()

    # ── Discover onboarded users ──────────────────────────────────────────────
    users = engine.list_onboarded_users()

    if not users:
        print("\nNo onboarded users found. Please ensure users are onboarded first.")
        return

    print(f"\nFound {len(users)} onboarded user(s).  Running retrieval...\n")
    print("=" * 80)

    for user_info in users:
        user_id = user_info["user_id"]
        interests = ", ".join(user_info.get("interests", [])) or "(none)"
        print(f"\n{'=' * 80}")
        print(f"  User: {user_id}")
        print(f"  Interests: {interests}")
        print(f"{'=' * 80}\n")

        try:
            batches = engine.fetch_onboarding_batches(user_id)
            _print_batch("batch_1 (highest similarity)", batches["batch_1"])
            _print_batch("batch_2 (mid similarity)",     batches["batch_2"])
            _print_batch("batch_3 (lowest similarity)",  batches["batch_3"])

            # ── Quick monotonicity check ──────────────────────────────────────
            scores_1 = [r["cosine_score"] for r in batches["batch_1"]] or [0]
            scores_3 = [r["cosine_score"] for r in batches["batch_3"]] or [0]
            if scores_1 and scores_3 and min(scores_1) >= max(scores_3):
                print("  [PASS]  Monotonicity check passed: batch_1 min >= batch_3 max")
            elif not scores_3:
                print("  [WARN]  batch_3 is empty (corpus may have < 45 repos)")
            else:
                print(
                    f"  [FAIL]  Monotonicity FAILED: batch_1 min={min(scores_1):.4f} "
                    f"< batch_3 max={max(scores_3):.4f}  -- investigate Qdrant query"
                )

        except Exception as exc:
            print(f"  [FAIL]  Retrieval failed for '{user_id}': {exc}")

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == "__main__":
    main()
