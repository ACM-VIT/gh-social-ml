"""Multi-channel L1 candidate retrieval engine.

Retrieves repository candidates from Qdrant (semantic search) and PostgreSQL
(trending), merges and deduplicates them, then hydrates the final pool with
full metadata and embedding vectors for downstream ranking models.

Usage::

    from retrieval import CandidateRetriever
    from database import PostgreSQLConnector

    db = PostgreSQLConnector()
    retriever = CandidateRetriever(db_connector=db)

    candidates = retriever.retrieve_candidates(
        user_embedding=[0.12, -0.45, ...],  # 384-d vector
        user_interests=["AI/ML", "Backend"],
    )
"""

from __future__ import annotations

import logging
import os
import math
import uuid
from typing import Any, Optional

from .config import (
    SEMANTIC_LIMIT,
    TRENDING_LIMIT,
    TOTAL_CANDIDATE_POOL,
    OVERFETCH_MULTIPLIER,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_NAME,
    QDRANT_TIMEOUT_SECONDS,
    EMBEDDING_DIM,
    FALLBACK_REPOS,
)

try:
    from embedding.qdrant_store import QdrantRepositoryStore
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

logger = logging.getLogger("pipeline.retrieval")


class CandidateRetriever:
    """Orchestrates L1 multi-channel candidate retrieval.

    Channels
    --------
    1. **Semantic** — Qdrant exact cosine similarity search using the user
       embedding vector.  Returns up to ``SEMANTIC_LIMIT`` candidates.
    2. **Trending** — PostgreSQL query ordered by ``star_count`` descending
       (proxy for trending velocity until the dedicated column is added).
       Returns up to ``TRENDING_LIMIT`` candidates.

    The two lists are merged, deduplicated, and sliced to
    ``TOTAL_CANDIDATE_POOL`` items.  Each candidate is then hydrated with
    full metadata from PostgreSQL and its 384-d embedding from Qdrant.
    """

    def __init__(
        self,
        db_connector: Any,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self.db = db_connector

        # ── Qdrant client setup ──────────────────────────────────────────
        self._qdrant_store: QdrantRepositoryStore | None = None
        if HAS_QDRANT:
            url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
            api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
            try:
                self._qdrant_store = QdrantRepositoryStore(
                    url=url,
                    api_key=api_key,
                    collection_name=QDRANT_COLLECTION_NAME,
                    vector_name=QDRANT_VECTOR_NAME,
                    vector_size=EMBEDDING_DIM,
                )
                # Quick health-check: verify the collection exists
                info = self._qdrant_store.client.get_collection(QDRANT_COLLECTION_NAME)
                logger.info(
                    "Qdrant connected — collection '%s' has %d vectors",
                    QDRANT_COLLECTION_NAME,
                    info.points_count,
                )
            except Exception as exc:
                logger.warning("Qdrant connection failed: %s. Semantic channel disabled.", exc)
                self._qdrant_store = None
        else:
            logger.warning(
                "qdrant-client is not installed. Semantic retrieval will be disabled. "
                "Run 'pip install qdrant-client' to enable."
            )

        # Lazy embedder for on-the-fly embedding of trending repos
        self._embedder = None

    # ══════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def retrieve_candidates(
        self,
        user_embedding: list[float] | None = None,
        user_interests: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run all retrieval channels and return a hydrated candidate pool.

        Parameters
        ----------
        user_embedding : list[float] | None
            384-dimensional user persona vector.  If ``None``, the semantic
            channel is skipped and its quota is transferred to trending.
        user_interests : list[str] | None
            Onboarding category names (e.g. ``["AI/ML", "Backend"]``).
            Currently used for logging; category-weighted retrieval will be
            added once the Repo table has a ``category`` column.

        Returns
        -------
        list[dict]
            Up to ``TOTAL_CANDIDATE_POOL`` candidate dicts, each containing
            PostgreSQL metadata fields and a ``repo_embedding`` key with the
            384-d float list.
        """
        # ── Validate inputs ──────────────────────────────────────────────
        if user_embedding is not None:
            if not isinstance(user_embedding, (list, tuple)) or len(user_embedding) != EMBEDDING_DIM:
                logger.error(
                    "user_embedding must be a list of %d floats, got length %s. "
                    "Skipping semantic channel.",
                    EMBEDDING_DIM,
                    len(user_embedding) if isinstance(user_embedding, (list, tuple)) else type(user_embedding).__name__,
                )
                user_embedding = None

        if user_interests is not None:
            if not isinstance(user_interests, (list, tuple)) or len(user_interests) == 0:
                logger.warning(
                    "user_interests is empty or invalid (%s). "
                    "Category channel quota transferred to trending.",
                    type(user_interests).__name__,
                )
                user_interests = None

        # ── Determine initial semantic quota ─────────────────────────────
        semantic_quota = SEMANTIC_LIMIT if user_embedding is not None else 0

        # ── Channel 1: Semantic retrieval (Qdrant) ───────────────────────
        semantic_ids = self._retrieve_semantic(user_embedding, semantic_quota)

        # ── Determine trending quota ─────────────────────────────────────
        # If Qdrant is disabled or user_embedding is None (cold start), we want to
        # reallocate/transfer the full semantic quota to the trending channel to backfill
        # the pool up to TOTAL_CANDIDATE_POOL (150).
        # Otherwise, if Qdrant is enabled, we query semantic search. If it fails or returns 
        # 0 matches, we fall back to querying TOTAL_CANDIDATE_POOL from trending.
        # If semantic search succeeds (returns > 0 matches), we cap trending at TRENDING_LIMIT.
        if user_embedding is None or self._qdrant_store is None:
            trending_quota = TOTAL_CANDIDATE_POOL
        else:
            unique_semantic = len({c.get("full_name") or c.get("repo_id") for c in semantic_ids if c.get("full_name") or c.get("repo_id")})
            if unique_semantic == 0:
                trending_quota = TOTAL_CANDIDATE_POOL
            else:
                trending_quota = TRENDING_LIMIT

        logger.info(
            "Retrieval quotas — Semantic (actual fetched): %d, Trending: %d (total target: %d)",
            len(semantic_ids), trending_quota, TOTAL_CANDIDATE_POOL,
        )

        # ── Channel 2: Trending retrieval (PostgreSQL) ───────────────────
        trending_ids = self._retrieve_trending(trending_quota)

        # ── Merge & deduplicate ──────────────────────────────────────────
        merged_ids = self._merge_and_deduplicate(
            semantic_ids, trending_ids, semantic_quota, TOTAL_CANDIDATE_POOL,
        )

        if not merged_ids:
            logger.warning("All retrieval channels returned empty. Returning fallback repos.")
            return self._build_fallback_candidates()

        # ── Hydrate with metadata + embeddings ───────────────────────────
        hydrated = self._hydrate_candidates(merged_ids)

        logger.info(
            "Candidate retrieval complete — %d candidates hydrated and ready for ranking.",
            len(hydrated),
        )
        return hydrated

    # ══════════════════════════════════════════════════════════════════════
    #  CHANNEL 1 — SEMANTIC (Qdrant)
    # ══════════════════════════════════════════════════════════════════════

    def _retrieve_semantic(
        self,
        user_embedding: list[float] | None,
        quota: int,
    ) -> list[dict[str, Any]]:
        """Query Qdrant for the top-K semantically similar repositories.

        Returns a list of dicts with keys ``id`` (Qdrant point ID) and
        ``score`` (cosine similarity).
        """
        if user_embedding is None or quota <= 0:
            return []

        if self._qdrant_store is None:
            logger.warning("Qdrant store unavailable. Semantic channel returns empty.")
            return []

        # Over-fetch to absorb deduplication losses (single query, no loop)
        fetch_limit = min(int(math.ceil(quota * OVERFETCH_MULTIPLIER)), quota + 100)

        try:
            matches = self._qdrant_store.search(
                vector=user_embedding,
                limit=fetch_limit,
                exact=True,
            )
            results = []
            for match in matches:
                results.append({
                    "point_id": match["id"],
                    "repo_id": match["repo_id"],
                    "full_name": match["repo_id"],
                    "score": match["score"],
                    "source": "semantic",
                })
            logger.info("Semantic channel retrieved %d candidates from Qdrant.", len(results))
            return results

        except Exception as exc:
            logger.error("Qdrant semantic search failed: %s", exc)
            return []

    # ══════════════════════════════════════════════════════════════════════
    #  CHANNEL 2 — TRENDING (PostgreSQL)
    # ══════════════════════════════════════════════════════════════════════

    def _retrieve_trending(self, quota: int) -> list[dict[str, Any]]:
        """Query PostgreSQL for trending repositories.

        First queries the ``trending_repositories`` table populated by the
        dedicated trending service.  Falls back to a star-count blended
        engagement query on the ``Repo`` table if the trending table is empty
        or does not exist.
        """
        if quota <= 0:
            return []

        if self.db is None or not self.db.enabled:
            logger.warning("PostgreSQL connector disabled. Trending channel returns empty.")
            return []

        # Over-fetch to absorb deduplication losses
        fetch_limit = min(int(math.ceil(quota * OVERFETCH_MULTIPLIER)), quota + 50)

        conn = None
        try:
            conn = self.db.connect()
            cursor = conn.cursor()

            # ── Primary: trending_repositories table ──────────────────────────
            results = self._query_trending_table(conn, cursor, fetch_limit)

            if results:
                logger.info(
                    "Trending channel retrieved %d candidates from trending_repositories.",
                    len(results),
                )
                return results

            # ── Fallback: Repo table with engagement blended score ────────────
            logger.info(
                "trending_repositories table empty or missing. "
                "Falling back to Repo table engagement score."
            )
            results = self._query_repo_table_trending(cursor, fetch_limit)
            logger.info(
                "Trending channel retrieved %d candidates from Repo (fallback).",
                len(results),
            )
            return results

        except Exception as exc:
            logger.error("PostgreSQL trending query failed: %s", exc)
            return []
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _query_trending_table(self, conn, cursor, fetch_limit: int) -> list[dict[str, Any]]:
        """Query the trending_repositories table ordered by trending_rank.

        Returns an empty list if the table does not exist or has no rows.
        """
        try:
            cursor.execute(
                """
                SELECT full_name, star_count, daily_stars, primary_language,
                       topics, description, trending_rank
                FROM trending_repositories
                ORDER BY trending_rank ASC
                LIMIT %s;
                """,
                (fetch_limit,),
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                full_name = row[0]
                results.append({
                    "repo_id": full_name,  # Fall back to full_name as the identifier
                    "full_name": full_name,
                    "star_count": row[1] or 0,
                    "daily_stars": row[2] or 0,
                    "primary_language": row[3] or "Unknown",
                    "topics": row[4] or [],
                    "description": row[5] or "",
                    "trending_rank": row[6],
                    "source": "trending",
                })
            return results
        except Exception as exc:
            # Table may not exist yet — treat as empty
            err_str = str(exc).lower()
            if "does not exist" in err_str or "undefined" in err_str or "relation" in err_str:
                logger.debug("trending_repositories table not found: %s", exc)
                try:
                    conn.rollback()
                except Exception:
                    pass
                return []
            raise

    def _query_repo_table_trending(self, cursor, fetch_limit: int) -> list[dict[str, Any]]:
        """Query the Repo table ordered by a blended engagement score (fallback)."""
        cursor.execute(
            """
            SELECT repo_id, full_name, star_count
            FROM Repo
            ORDER BY (star_count + COALESCE(forks_count, 0) * 5 + COALESCE(pr_count, 0) * 10) DESC
            LIMIT %s;
            """,
            (fetch_limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "repo_id": str(row[0]),
                "full_name": row[1],
                "star_count": row[2],
                "source": "trending",
            }
            for row in rows
        ]

    # ══════════════════════════════════════════════════════════════════════
    #  MERGE & DEDUPLICATION
    # ══════════════════════════════════════════════════════════════════════

    def _merge_and_deduplicate(
        self,
        semantic_candidates: list[dict],
        trending_candidates: list[dict],
        semantic_limit: int,
        pool_limit: int,
    ) -> list[dict[str, Any]]:
        """Merge channel results, deduplicate by full_name, and cap at pool_limit.

        Semantic candidates are placed first (higher relevance), followed by
        trending candidates that were not already captured by semantic search.
        This is a single-pass merge — no loops or recursive re-fetching.
        """
        seen_names: set[str] = set()
        merged: list[dict[str, Any]] = []

        # Pass 1: Semantic candidates (priority, up to semantic_limit)
        for candidate in semantic_candidates:
            name = candidate.get("full_name") or candidate.get("repo_id")
            if name and name not in seen_names:
                seen_names.add(name)
                merged.append(candidate)
                if len(merged) >= semantic_limit:
                    break

        # Pass 2: Trending candidates (fill remaining slots up to pool_limit)
        for candidate in trending_candidates:
            name = candidate.get("full_name") or candidate.get("repo_id")
            if name and name not in seen_names:
                seen_names.add(name)
                merged.append(candidate)
                if len(merged) >= pool_limit:
                    break

        # Slice to the target pool size (no loop — just a list slice)
        final = merged[:pool_limit]

        logger.info(
            "Merge complete — %d semantic + %d trending → %d unique → %d after cap.",
            len(semantic_candidates),
            len(trending_candidates),
            len(merged),
            len(final),
        )
        return final

    # ══════════════════════════════════════════════════════════════════════
    #  HYDRATION
    # ══════════════════════════════════════════════════════════════════════

    def _hydrate_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Enrich each candidate with full PostgreSQL metadata and Qdrant embeddings.

        This runs two batch queries (one to each database) to avoid N+1 problems.
        Trending candidates that have no Qdrant point are embedded on-the-fly
        using the same SentenceTransformerEmbedder used by the main pipeline,
        so they always enter the ranker with a real semantic vector.
        """
        # Ensure every candidate has full_name and point_id populated
        for c in candidates:
            name = c.get("full_name") or c.get("repo_id")
            if name:
                c["full_name"] = name
                if not c.get("point_id"):
                    c["point_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"github:{name}"))

        # ── Step 1: Batch-fetch metadata from PostgreSQL ─────────────────
        full_names = [c.get("full_name") for c in candidates if c.get("full_name")]
        metadata_map = self._batch_fetch_metadata(full_names)

        # ── Step 2: Batch-fetch embeddings from Qdrant ───────────────────
        point_ids = [c.get("point_id") for c in candidates if c.get("point_id")]
        embedding_map = self._batch_fetch_embeddings(point_ids)

        # ── Step 3: Join into final payload ──────────────────────────────
        # Collect candidates missing a Qdrant vector for on-the-fly embedding
        needs_embedding: list[tuple[int, dict[str, Any]]] = []
        hydrated: list[dict[str, Any]] = [{}] * len(candidates)

        for i, candidate in enumerate(candidates):
            name = candidate.get("full_name")
            pid = candidate.get("point_id")

            entry: dict[str, Any] = {
                "retrieval_source": candidate.get("source", "unknown"),
                "retrieval_score": candidate.get("score"),
            }

            # Merge PostgreSQL metadata
            if name and name in metadata_map:
                entry.update(metadata_map[name])
            else:
                entry["full_name"] = name

            # Attach embedding vector — or mark for on-the-fly generation
            if pid and pid in embedding_map:
                entry["repo_embedding"] = embedding_map[pid]
                entry["embedding_source"] = "qdrant"
            else:
                # No Qdrant point (typical for trending-only repos).
                # Real embedding will be generated in Step 4.
                entry["repo_embedding"] = None
                needs_embedding.append((i, entry))

            hydrated[i] = entry

        # ── Step 4: Batch embed all candidates missing a Qdrant vector ─────
        if needs_embedding:
            texts = [self._make_embedding_text(e) for _, e in needs_embedding]
            try:
                vectors = self._get_embedder().embed_texts(texts, normalize=True)
                for (_, entry), vector in zip(needs_embedding, vectors):
                    entry["repo_embedding"] = vector
                    entry["embedding_source"] = "on_the_fly"
                logger.info(
                    "On-the-fly embedded %d trending repo(s) with no Qdrant vector.",
                    len(needs_embedding),
                )
            except Exception as exc:
                logger.warning(
                    "On-the-fly embedding failed (%s). "
                    "Falling back to zero-vector for %d repos.",
                    exc, len(needs_embedding),
                )
                for _, entry in needs_embedding:
                    if entry["repo_embedding"] is None:
                        entry["repo_embedding"] = [0.0] * EMBEDDING_DIM
                        entry["embedding_source"] = "zero_fallback"

        return hydrated

    def _get_embedder(self):
        """Lazy-load the SentenceTransformerEmbedder (same model as main pipeline)."""
        if self._embedder is None:
            from embedding.embeddings import SentenceTransformerEmbedder
            self._embedder = SentenceTransformerEmbedder()
            logger.info("Lazy-loaded SentenceTransformerEmbedder for on-the-fly trending embeddings.")
        return self._embedder

    @staticmethod
    def _make_embedding_text(entry: dict[str, Any]) -> str:
        """Build short descriptive text for embedding a trending repo.

        Uses metadata + topic fields (no README) — the same signals as the
        main pipeline's metadata and topic towers.
        """
        parts: list[str] = []

        name = entry.get("full_name") or ""
        if name:
            parts.append(f"Repository: {name}")

        desc = entry.get("description") or ""
        if desc:
            parts.append(f"Description: {desc}")

        lang = entry.get("primary_language") or ""
        if lang and lang != "Unknown":
            parts.append(f"Primary language: {lang}")

        topics = entry.get("topics") or []
        if isinstance(topics, str):
            import json as _json
            try:
                topics = _json.loads(topics)
            except Exception:
                topics = []
        if topics:
            parts.append("Topics: " + ", ".join(str(t) for t in topics))

        stars = int(entry.get("star_count") or 0)
        if stars:
            parts.append(f"Stars: {stars}")

        return "\n".join(parts) or name or "unknown repository"

    def _batch_fetch_metadata(
        self,
        full_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch full metadata rows from PostgreSQL for a batch of repository full_names.

        First queries the ``Repo`` table.  For any repositories not found there
        (e.g. newly trending repos not yet ingested by the main pipeline), it
        performs a supplementary lookup in ``trending_repositories``.
        """
        if not full_names or self.db is None or not self.db.enabled:
            return {}

        conn = None
        try:
            conn = self.db.connect()
            cursor = conn.cursor()

            # ── Primary: Repo table ───────────────────────────────────────────
            cursor.execute(
                """
                SELECT repo_id, github_repo_url, owner_id, repo_name, full_name,
                       description, primary_language, language_used, topics,
                       readme_summary, star_count, forks_count, pr_count,
                       likes_count, comments_count, saves_count, views_count,
                       created_at, updated_at
                FROM Repo
                WHERE full_name = ANY(%s);
                """,
                (full_names,),
            )

            columns = [
                "repo_id", "github_repo_url", "owner_id", "repo_name", "full_name",
                "description", "primary_language", "language_used", "topics",
                "readme_summary", "star_count", "forks_count", "pr_count",
                "likes_count", "comments_count", "saves_count", "views_count",
                "created_at", "updated_at",
            ]

            result_map: dict[str, dict[str, Any]] = {}
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                result_map[row_dict["full_name"]] = row_dict

            logger.info(
                "Metadata hydration: %d/%d full names found in Repo table.",
                len(result_map), len(full_names),
            )

            # ── Supplement: trending_repositories table for missing repos ─────
            missing = [n for n in full_names if n not in result_map]
            if missing:
                trending_meta = self._fetch_metadata_from_trending(cursor, missing)
                result_map.update(trending_meta)
                if trending_meta:
                    logger.info(
                        "Metadata hydration: %d supplemental entries from trending_repositories.",
                        len(trending_meta),
                    )

            return result_map

        except Exception as exc:
            logger.error("PostgreSQL metadata hydration failed: %s", exc)
            return {}
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _fetch_metadata_from_trending(
        self,
        cursor,
        full_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch repository metadata from trending_repositories for names not in Repo."""
        try:
            cursor.execute(
                """
                SELECT full_name, description, primary_language, topics,
                       star_count, daily_stars, fork_count, url, trending_rank
                FROM trending_repositories
                WHERE full_name = ANY(%s);
                """,
                (full_names,),
            )
            result_map: dict[str, dict[str, Any]] = {}
            for row in cursor.fetchall():
                full_name = row[0]
                # Normalise to the same shape expected by the ranker/assembler
                topics = row[3]
                if isinstance(topics, str):
                    import json
                    try:
                        topics = json.loads(topics)
                    except Exception:
                        topics = []
                result_map[full_name] = {
                    "repo_id": full_name,
                    "github_repo_url": row[7] or f"https://github.com/{full_name}",
                    "full_name": full_name,
                    "description": row[1] or "",
                    "primary_language": row[2] or "Unknown",
                    "topics": topics or [],
                    "star_count": row[4] or 0,
                    "daily_stars": row[5] or 0,
                    "forks_count": row[6] or 0,
                    "trending_rank": row[8],
                }
            return result_map
        except Exception as exc:
            err_str = str(exc).lower()
            if "does not exist" in err_str or "relation" in err_str:
                logger.debug("trending_repositories table unavailable for supplement: %s", exc)
                return {}
            logger.warning("Supplemental trending metadata fetch failed: %s", exc)
            return {}

    def _batch_fetch_embeddings(
        self,
        point_ids: list[str],
    ) -> dict[str, list[float]]:
        """Fetch embedding vectors from Qdrant for a batch of point IDs."""
        if not point_ids or self._qdrant_store is None:
            return {}

        try:
            # Qdrant retrieve() fetches points by their IDs in a single call
            points = self._qdrant_store.client.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=point_ids,
                with_vectors=True,
                with_payload=False,
            )

            embedding_map: dict[str, list[float]] = {}
            for point in points:
                vec = point.vector
                # Handle named vectors (collection may store vectors under a name)
                if isinstance(vec, dict):
                    vec = vec.get(QDRANT_VECTOR_NAME, [0.0] * EMBEDDING_DIM)
                if vec is None:
                    vec = [0.0] * EMBEDDING_DIM
                embedding_map[str(point.id)] = list(vec)

            logger.info(
                "Embedding hydration: %d/%d point IDs retrieved from Qdrant.",
                len(embedding_map), len(point_ids),
            )
            return embedding_map

        except Exception as exc:
            logger.error("Qdrant embedding hydration failed: %s", exc)
            return {}

    # ══════════════════════════════════════════════════════════════════════
    #  FALLBACK
    # ══════════════════════════════════════════════════════════════════════

    def _build_fallback_candidates(self) -> list[dict[str, Any]]:
        """Return a static list of well-known repositories as a last resort.

        Used only when both Qdrant and PostgreSQL are completely unreachable.
        Each entry gets a zero-vector embedding so downstream models don't crash.
        """
        logger.warning(
            "Building fallback candidate list (%d repos). "
            "This indicates ALL retrieval channels failed.",
            len(FALLBACK_REPOS),
        )
        return [
            {
                "repo_id": None,
                "full_name": repo_name,
                "retrieval_source": "fallback",
                "retrieval_score": 0.0,
                "repo_embedding": [0.0] * EMBEDDING_DIM,
            }
            for repo_name in FALLBACK_REPOS
        ]
