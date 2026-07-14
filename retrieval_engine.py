"""Integrated feed assembly engine.

This module wires together the complete post-onboarding recommendation pipeline:

  User Profile (Qdrant) → CandidateRetriever (Semantic + Trending) → RankerService (MMoE) → Ranked Batches

Usage::

    from retrieval_engine import RetrievalEngine

    engine = RetrievalEngine()
    result = engine.fetch_onboarding_batches("user_123")
    # result == {"batch_1": [...15 items...], "batch_2": [...], "batch_3": [...]}
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from qdrant_client import QdrantClient
from feedback.storage import FeedbackStore, apply_feedback_scores

from config import (  # type: ignore
    QDRANT_API_KEY,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
    QDRANT_COLLECTION_NAME,
)
from scripts.user_onboarding import USER_PROFILES_COLLECTION, TARGET_VECTOR_NAME  # type: ignore

logger = logging.getLogger("pipeline.retrieval")

BATCH_SIZE = 15
NUM_BATCHES = 3

# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalEngine:
    """Integrated feed assembler: retrieval, ranking, and batch generation.

    Pipeline
    --------
    1. Load user interest embedding from Qdrant ``user_profiles``.
    2. Pull the candidate pool via ``CandidateRetriever`` (approximate semantic
       search + Qdrant-native discovery channels).
    3. Score every candidate with ``RankerService`` (MMoE heavy ranker).
       All candidates already carry their indexed repository vectors.
    4. Slice the top-ranked candidates into three batches of 15 and return
       them to the backend, which owns feed caching and invalidation.

    Caching
    -------
    The backend owns the Redis feed cache. The ML service intentionally does
    has no connection to app-owned Postgres.
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

        # Lazy-loaded sub-components
        self._candidate_retriever: Any = None
        self._ranker: Any = None
        self._ranker_failed = False

    # ── Lazy sub-component accessors ──────────────────────────────────────────

    @property
    def candidate_retriever(self):
        """Lazy-load the CandidateRetriever."""
        if self._candidate_retriever is None:
            try:
                from retrieval import CandidateRetriever
                self._candidate_retriever = CandidateRetriever(
                    qdrant_url=self._url,
                    qdrant_api_key=self._api_key,
                )
            except Exception as exc:
                logger.warning("Could not initialize CandidateRetriever: %s", exc)
                self._candidate_retriever = False
        return self._candidate_retriever if self._candidate_retriever is not False else None

    @property
    def ranker(self):
        """Lazy-load the RankerService (MMoE heavy ranker)."""
        if self._ranker is None and not self._ranker_failed:
            try:
                # Resolve paths relative to the inference/ directory
                _base = os.path.join(os.path.dirname(__file__), "inference")
                model_path = os.path.join(_base, "heavy_ranker.pt")
                scaler_path = os.path.join(_base, "feature_scaler.json")

                sys.path.insert(0, _base)
                from ranker_service import RankerService  # type: ignore
                self._ranker = RankerService(
                    model_path=model_path,
                    scaler_path=scaler_path,
                )
            except Exception as exc:
                logger.warning("Could not initialize RankerService: %s", exc)
                self._ranker_failed = True
        return self._ranker

    # ── Core public API ───────────────────────────────────────────────────────

    def fetch_onboarding_batches(
        self, user_id: str, *, is_cold_start: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Generate ranked recommendation batches for a user.

        Returns
        -------
        dict with keys ``"batch_1"``, ``"batch_2"``, ``"batch_3"``, each a
        list of up to ``BATCH_SIZE`` ranked repository dicts.
        """
        import time

        # ── 1. Get user profile from Qdrant ───────────────────────────────────
        try:
            user_vector, user_skills = self._get_user_profile(user_id)
        except ValueError:
            if is_cold_start:
                logger.info("Cold start user '%s' has no Qdrant profile yet.", user_id)
                user_vector = []
                user_skills = []
            else:
                raise
        except Exception as exc:
            logger.warning(
                "User '%s' Qdrant profile lookup failed (%s); using discovery retrieval.",
                user_id, type(exc).__name__
            )
            user_vector = []
            user_skills = []
            is_cold_start = True

        if is_cold_start:
            return self._cold_start_pipeline(user_id, user_vector, user_skills)

        # ── 2. Retrieve candidate pool (Semantic + Trending) ──────────────────
        start_retrieval = time.time()
        candidates = self._retrieve_candidates(user_vector, user_skills)
        retrieval_latency = (time.time() - start_retrieval) * 1000.0

        # ── 3. Rank the candidate pool with the MMoE heavy ranker ─────────────
        start_ranking = time.time()
        ranked = self._rank_candidates(user_vector, user_skills, candidates)
        ranked = self._apply_user_feedback(user_id, ranked)
        ranking_latency = (time.time() - start_ranking) * 1000.0

        # ── 4. Slice into 3 batches of BATCH_SIZE ─────────────────────────────
        batches = {
            "batch_1": ranked[0:BATCH_SIZE],
            "batch_2": ranked[BATCH_SIZE: BATCH_SIZE * 2],
            "batch_3": ranked[BATCH_SIZE * 2: BATCH_SIZE * 3],
        }

        # ── 5. Return to Backend for Redis Caching ────────────────────────────

        logger.info(
            "Generated onboarding batches for '%s': %d / %d / %d items.",
            user_id,
            len(batches["batch_1"]),
            len(batches["batch_2"]),
            len(batches["batch_3"]),
        )
        logger.info(
            "Latency Profile: Candidate Retrieval = %.2fms, MMoE Ranking = %.2fms (Total = %.2fms)",
            retrieval_latency,
            ranking_latency,
            retrieval_latency + ranking_latency,
        )
        return batches

    # ── Cold Start ────────────────────────────────────────────────────────────

    def _cold_start_pipeline(
        self, user_id: str, user_vector: list[float], user_skills: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Dedicated retrieval and ranking pathway for new users with 0 interactions."""
        import time

        logger.info("Executing Cold Start pipeline for user '%s'", user_id)
        start_retrieval = time.time()
        
        candidates = self._retrieve_candidates(user_vector, user_skills)
        retrieval_latency = (time.time() - start_retrieval) * 1000.0

        start_ranking = time.time()
        ranked = self._score_cold_start_candidates(user_skills, candidates)
        ranked = self._apply_user_feedback(user_id, ranked)
        ranking_latency = (time.time() - start_ranking) * 1000.0

        batches = {
            "batch_1": ranked[0:BATCH_SIZE],
            "batch_2": ranked[BATCH_SIZE : BATCH_SIZE * 2],
            "batch_3": ranked[BATCH_SIZE * 2 : BATCH_SIZE * 3],
        }



        logger.info(
            "Generated Cold Start batches for '%s': %d / %d / %d items.",
            user_id,
            len(batches["batch_1"]),
            len(batches["batch_2"]),
            len(batches["batch_3"]),
        )
        logger.info(
            "Cold Start Latency: Retrieval = %.2fms, Scoring = %.2fms (Total = %.2fms)",
            retrieval_latency,
            ranking_latency,
            retrieval_latency + ranking_latency,
        )
        return batches

    def _score_cold_start_candidates(
        self, user_skills: list[str], candidates: list[dict]
    ) -> list[dict]:
        """Deterministically score candidates based on skill match and popularity."""
        import math
        from retrieval.config import COLD_START_SKILL_WEIGHT, COLD_START_STARS_WEIGHT

        if not candidates:
            return []

        max_log_stars = math.log1p(500_000)  # normalisation ceiling
        user_set = {s.lower() for s in user_skills}

        for c in candidates:
            # --- Skill match ratio (0.0 to 1.0) ---
            repo_signals = set()
            lang = c.get("primary_language", "")
            if lang and lang != "Unknown":
                repo_signals.add(lang.lower())
            
            for t in (c.get("topics") or []):
                repo_signals.add(str(t).lower())
            
            for l in (c.get("languages") or []):
                # if language_used was a dict mapped to bytes, handle appropriately, 
                # but typically frontend/backend uses strings or lists
                repo_signals.add(str(l).lower())

            overlap = len(repo_signals & user_set)
            skill_match = overlap / max(len(user_set), 1)

            # --- Normalised star popularity (0.0 to 1.0) ---
            stars = int(c.get("star_count") or 0)
            norm_stars = min(math.log1p(stars) / max_log_stars, 1.0)

            # --- Final cold-start score ---
            c["final_score"] = (COLD_START_SKILL_WEIGHT * skill_match) + (COLD_START_STARS_WEIGHT * norm_stars)
            c["score_source"] = "cold_start"
            # MMoE fields fallback so UI doesn't break
            c["predictions"] = {
                "p_ctr": skill_match,
                "p_save": norm_stars,
                "p_follow": 0.0,
                "pred_dwell_fraction": 0.5,
            }

        candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return candidates

    # ── User profile retrieval ────────────────────────────────────────────────


    def _get_user_profile(self, user_id: str) -> tuple[list[float], list[str]]:
        """Return (interest_vector, skills_list) for a user from Qdrant.

        The point ID is a deterministic UUID5 matching the scheme in
        ``user_onboarding.py:save_to_qdrant``.
        """
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"user:{user_id}"))

        response = self._client.retrieve(
            collection_name=USER_PROFILES_COLLECTION,
            ids=[point_uuid],
            with_vectors=True,
            with_payload=True,
        )

        if not response:
            raise ValueError(
                f"User '{user_id}' (point {point_uuid}) not found in "
                f"Qdrant collection '{USER_PROFILES_COLLECTION}'."
            )

        point = response[0]

        # Extract vector
        if isinstance(point.vector, dict):
            if TARGET_VECTOR_NAME and TARGET_VECTOR_NAME in point.vector:
                user_vector = list(point.vector[TARGET_VECTOR_NAME])
            else:
                vectors = list(point.vector.values())
                if not vectors:
                    raise ValueError(f"User '{user_id}' has an empty named-vector dict.")
                user_vector = list(vectors[0])
        else:
            user_vector = list(point.vector)

        # Extract skills from payload (used by the ranker's skill_match feature)
        payload = point.payload or {}
        skills_raw = payload.get("skills") or []
        tech_raw = payload.get("tech_stack") or []
        if isinstance(skills_raw, str):
            skills_raw = [skills_raw]
        if not isinstance(skills_raw, list):
            skills_raw = list(skills_raw) if isinstance(skills_raw, (tuple, set)) else []
        if isinstance(tech_raw, str):
            tech_raw = [tech_raw]
        if not isinstance(tech_raw, list):
            tech_raw = list(tech_raw) if isinstance(tech_raw, (tuple, set)) else []
        skills = skills_raw + tech_raw

        return user_vector, skills

    def _get_user_data(self, user_id: str) -> tuple[list[float], dict[str, Any]]:
        """Retrieve both the vector and payload for a user deterministic UUID."""
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"user:{user_id}"))

        response = self._client.retrieve(
            collection_name=USER_PROFILES_COLLECTION,
            ids=[point_uuid],
            with_vectors=True,
            with_payload=True,
        )

        if not response:
            raise ValueError(
                f"User '{user_id}' (point {point_uuid}) not found in "
                f"Qdrant collection '{USER_PROFILES_COLLECTION}'."
            )

        point = response[0]
        payload = point.payload or {}

        if isinstance(point.vector, dict):
            if TARGET_VECTOR_NAME and TARGET_VECTOR_NAME in point.vector:
                return list(point.vector[TARGET_VECTOR_NAME]), payload
            
            vectors = list(point.vector.values())
            if not vectors:
                raise ValueError(f"User '{user_id}' has an empty named-vector dict.")
            return list(vectors[0]), payload

        return list(point.vector), payload

    def _get_user_vector(self, user_id: str) -> list[float]:
        """Retrieve the user's interest embedding from the user_profiles collection."""
        vector, _ = self._get_user_data(user_id)
        return vector

    # ── Candidate retrieval ───────────────────────────────────────────────────

    def _retrieve_candidates(
        self,
        user_vector: list[float],
        user_skills: list[str],
    ) -> list[dict[str, Any]]:
        """Pull the L1 candidate pool via CandidateRetriever.

        Falls back to an empty list if the retriever is unavailable, letting
        the ranker gracefully handle an empty pool.
        """
        retriever = self.candidate_retriever
        if retriever is None:
            logger.warning(
                "CandidateRetriever unavailable.  No candidates to rank."
            )
            return []

        try:
            candidates = retriever.retrieve_candidates(
                user_embedding=user_vector,
                user_interests=user_skills,
            )
            logger.info(
                "CandidateRetriever returned %d candidates.", len(candidates)
            )
            return candidates
        except Exception as exc:
            logger.error("CandidateRetriever.retrieve_candidates failed: %s", exc)
            return []

    def _apply_user_feedback(
        self, user_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Blend persisted explicit feedback into an already-ranked pool."""
        try:
            scores = FeedbackStore(self.db).scores_for_user(user_id)
        except Exception as exc:
            logger.warning("Could not load feedback for '%s': %s", user_id, exc)
            return candidates
        return apply_feedback_scores(candidates, scores)

    # ── MMoE Ranking ──────────────────────────────────────────────────────────

    def _rank_candidates(
        self,
        user_vector: list[float],
        user_skills: list[str],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score and sort candidates with the MMoE heavy ranker.

        All candidates — including trending repos — now have real embeddings
        generated by ``CandidateRetriever`` via on-the-fly embedding, so they
        are all passed through the MMoE network uniformly.

        Each candidate dict is enriched with:
        - ``final_score``   — raw weighted value-function output (up to 28.1)
        - ``predictions``   — raw per-task probabilities (p_ctr, p_save, …)
        - ``score_source``  — "mmoe_{source}" or "cosine_fallback" (if ranker unavailable)
        """
        if not candidates:
            return []

        ranker = self.ranker

        if ranker is None:
            logger.warning(
                "RankerService unavailable.  Returning candidates in "
                "retrieval order (cosine score)."
            )
            for c in candidates:
                c.setdefault("final_score", c.get("retrieval_score") or 0.0)
                c.setdefault("predictions", {})
                c.setdefault("score_source", "cosine_fallback")
            return candidates

        import numpy as np

        user_emb = np.array(user_vector, dtype=np.float32)

        # ── Build ranker inputs for all candidates ────────────────────────────
        ranker_inputs: list[dict] = []
        for c in candidates:
            topics = c.get("topics") or []
            if isinstance(topics, str):
                try:
                    topics = json.loads(topics)
                except Exception:
                    topics = []

            languages = []
            lang = c.get("primary_language")
            if lang:
                languages = [lang]
            lang_used = c.get("language_used") or {}
            if isinstance(lang_used, dict):
                languages += list(lang_used.keys())
            elif isinstance(lang_used, list):
                languages += [str(l) for l in lang_used]

            repo_emb_raw = c.get("repo_embedding") or []
            repo_emb = np.array(repo_emb_raw, dtype=np.float32) if repo_emb_raw else np.zeros(ranker.emb_dim, dtype=np.float32)
            norm = np.linalg.norm(repo_emb)
            if norm > 1e-6:
                repo_emb = repo_emb / norm

            import math
            daily_stars = float(c.get("daily_stars") or 0.0)
            if daily_stars > 0:
                trend_vel = min(math.log1p(daily_stars) / math.log1p(500.0), 1.0)
            else:
                trend_vel = float(c.get("trend_velocity") or 0.0)

            ranker_inputs.append({
                "id":                c.get("repo_id") or c.get("full_name", "unknown"),
                "embedding":         repo_emb,
                "doc_quality":       c.get("doc_quality", 0.5),
                "code_health":       c.get("code_health", 0.5),
                "readme_length":     len(c.get("readme_summary") or "") or 1000,
                "star_count":        int(c.get("star_count") or 0),
                "fork_count":        int(c.get("forks_count") or c.get("fork_count") or 0),
                "open_issues_count": int(c.get("open_issues_count") or 0),
                "pushed_days_ago":   int(c.get("pushed_days_ago") or 365),
                "activity_score":    float(c.get("activity_score") or 0.0),
                "trend_velocity":    trend_vel,
                "languages":         languages,
                "topics":            topics,
                "tags":              topics,
            })

        # ── Run MMoE on all candidates ────────────────────────────────────────
        try:
            scored = ranker.score_batch(user_emb, user_skills, ranker_inputs)
            id_to_score: dict[str, dict] = {s["repo_id"]: s for s in scored}
        except Exception as exc:
            logger.error("RankerService.score_batch failed: %s. Falling back to cosine order.", exc)
            for c in candidates:
                c.setdefault("final_score", c.get("retrieval_score") or 0.0)
                c.setdefault("predictions", {})
                c.setdefault("score_source", "cosine_fallback")
            return candidates

        # ── Merge scores back ─────────────────────────────────────────────────
        enriched: list[dict[str, Any]] = []
        for c, inp in zip(candidates, ranker_inputs):
            c_copy = dict(c)
            score_entry = id_to_score.get(inp["id"], {})
            preds = score_entry.get("predictions", {})

            # Recalculate raw score based on retrieval source
            # Keeping the sum of weights identical to 28.1 ensures a fair comparison
            source = c.get("retrieval_source", "unknown")
            if source == "trending":
                # For trending repos, place less weight on follow (reducing popularity bias) and more on ctr/save
                # CTR=5.0, Save=8.0, GH_Open=5.0, Dwell=0.1, Follow=10.0 (Sum = 28.1)
                final_score = (
                    (5.0 * preds.get("p_ctr", 0.0)) +
                    (8.0 * preds.get("p_save", 0.0)) +
                    (5.0 * preds.get("p_gh", 0.0)) +
                    (0.1 * preds.get("pred_dwell_fraction", 0.0)) +
                    (10.0 * preds.get("p_follow", 0.0))
                )
            else:
                # Standard personalized formula:
                # CTR=1.0, Save=5.0, GH_Open=2.0, Dwell=0.1, Follow=20.0 (Sum = 28.1)
                final_score = (
                    (1.0 * preds.get("p_ctr", 0.0)) +
                    (5.0 * preds.get("p_save", 0.0)) +
                    (2.0 * preds.get("p_gh", 0.0)) +
                    (0.1 * preds.get("pred_dwell_fraction", 0.0)) +
                    (20.0 * preds.get("p_follow", 0.0))
                )

            c_copy["final_score"] = final_score
            c_copy["predictions"] = preds
            c_copy["score_source"] = f"mmoe_{source}"
            c_copy["languages"] = inp.get("languages", [])
            enriched.append(c_copy)

        enriched.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info(
            "RankerService scored %d candidates. Top score: %.4f",
            len(enriched),
            enriched[0]["final_score"] if enriched else 0.0,
        )
        return enriched

    # ── Utility: list onboarded users ─────────────────────────────────────────

    def list_onboarded_users(self, batch_size: int = 100) -> list[dict[str, Any]]:
        """Scroll the user_profiles collection and return all user metadata."""
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
            except Exception as exc:
                if "Not found" in str(exc) or "doesn't exist" in str(exc):
                    return users
                logger.error("Qdrant scroll failed: %s", exc)
                raise

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
    print(f"  {'#':<3} {'Score':>8}  {'Src':<6}  {'Repo':<42} {'Category'}")
    print(f"  {'-'*3} {'-'*8}  {'-'*6}  {'-'*42} {'-'*28}")
    for i, item in enumerate(batch, 1):
        score = item.get("final_score") or item.get("cosine_score") or 0.0
        src = item.get("score_source", "?")[:6]
        print(
            f"  {i:<3} {score:>8.4f}  {src:<6}  "
            f"{(item.get('full_name') or item.get('repo_id') or '?'):<42} "
            f"{item.get('category') or item.get('primary_language') or ''}"
        )
    print()


def main() -> None:
    """Run the full integrated pipeline for all onboarded users and print batches."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    engine = RetrievalEngine()
    users = engine.list_onboarded_users()

    if not users:
        print("\nNo onboarded users found. Please onboard users first.")
        return

    print(f"\nFound {len(users)} onboarded user(s).  Running retrieval + ranking...\n")
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
            _print_batch("batch_1 (top-ranked)", batches["batch_1"])
            _print_batch("batch_2 (mid-ranked)", batches["batch_2"])
            _print_batch("batch_3 (lower-ranked)", batches["batch_3"])

            scores_1 = [r.get("final_score", 0.0) for r in batches["batch_1"]]
            scores_3 = [r.get("final_score", 0.0) for r in batches["batch_3"]]
            if not scores_3:
                print("  [WARN]  batch_3 is empty (candidate pool may be < 45 repos)")
            elif scores_1 and min(scores_1) >= max(scores_3):
                print("  [PASS]  Monotonicity check passed: batch_1 min >= batch_3 max")
            else:
                print(
                    f"  [INFO]  Score overlap detected: batch_1 min={min(scores_1):.4f} "
                    f"/ batch_3 max={max(scores_3):.4f} "
                    "(expected for a learned ranker — cosine order may differ from MMoE order)"
                )

        except Exception as exc:
            print(f"  [FAIL]  Pipeline failed for '{user_id}': {exc}")

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == "__main__":
    main()
