#!/usr/bin/env python3
"""Integrated Retrieval + Ranking Demo.

Demonstrates the full end-to-end recommendation pipeline:

  User Onboarding (Qdrant) → CandidateRetriever (Semantic + Trending) → MMoE Ranker → Ranked Batches

Usage::

    # Run for all onboarded users
    python3 scripts/run_integrated_demo.py

    # Run for a specific user
    python3 scripts/run_integrated_demo.py --user-id alice_ml_expert

    # Onboard mock users first, then run
    python3 scripts/run_integrated_demo.py --onboard-first

    # Force re-generate batches (bypass 24-hour cache)
    python3 scripts/run_integrated_demo.py --no-cache
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Resolve repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo.integrated")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _onboard_mock_users() -> None:
    """Onboard the predefined mock users into Qdrant user_profiles."""
    from scripts.mock_users import MOCK_USERS
    from scripts.user_onboarding import onboard_user

    logger.info("Onboarding %d mock users...", len(MOCK_USERS))
    for user in MOCK_USERS:
        uid = user["user_id"]
        success = onboard_user(user_id=uid, user_data=user)
        if success:
            logger.info("  ✅  Onboarded: %s", uid)
        else:
            logger.warning("  ⚠️   Failed to onboard: %s", uid)


def _invalidate_cache(user_id: str, engine) -> None:
    """No-op: backend Redis now owns feed cache invalidation."""
    logger.info("Skipping local cache invalidation for '%s'; backend Redis owns feed cache.", user_id)


def _print_batch(
    batch_name: str,
    batch: list[dict],
    show_scores: bool = True,
) -> None:
    """Pretty-print a single recommendation batch."""
    if not batch:
        print(f"  {batch_name}: (empty)")
        return
    width = 90
    print(f"\n  ── {batch_name} ({len(batch)} repos) ────────────────────────────────")
    if show_scores:
        print(f"  {'#':<3} {'MMoE Score':>10}  {'Source':<10} {'Repo':<42} Language")
        print(f"  {'-'*3} {'-'*10}  {'-'*10} {'-'*42} {'-'*12}")
    else:
        print(f"  {'#':<3} {'Source':<10} {'Repo':<42} Language")
        print(f"  {'-'*3} {'-'*10} {'-'*42} {'-'*12}")

    for i, item in enumerate(batch, 1):
        score = item.get("final_score") or item.get("cosine_score") or 0.0
        repo_name = (item.get("full_name") or item.get("repo_id") or "?")[:42]
        source = item.get("retrieval_source", "?")[:10]
        lang = (item.get("primary_language") or "")[:12]

        if show_scores:
            print(f"  {i:<3} {score:>10.4f}  {source:<10} {repo_name:<42} {lang}")
        else:
            print(f"  {i:<3} {source:<10} {repo_name:<42} {lang}")


def _run_for_user(engine, user_id: str, no_cache: bool) -> None:
    """Run the integrated pipeline for a single user and print the results."""
    print(f"\n{'=' * 80}")
    print(f"  USER: {user_id}")
    print(f"{'=' * 80}")

    if no_cache:
        _invalidate_cache(user_id, engine)

    try:
        batches = engine.fetch_onboarding_batches(user_id)
    except ValueError as exc:
        print(f"  [ERROR] {exc}")
        print("  Hint: Run with --onboard-first to onboard mock users.")
        return
    except Exception as exc:
        print(f"  [ERROR] Pipeline failed: {exc}")
        logger.exception("Pipeline error for user '%s'", user_id)
        return

    _print_batch("Batch 1 — Top Picks", batches["batch_1"])
    _print_batch("Batch 2 — Mid Tier",  batches["batch_2"])
    _print_batch("Batch 3 — Discovery", batches["batch_3"])

    # Quick monotonicity check
    scores_1 = [r.get("final_score", 0.0) for r in batches["batch_1"]]
    scores_3 = [r.get("final_score", 0.0) for r in batches["batch_3"]]
    print()
    if not scores_3:
        print("  [WARN]  batch_3 is empty — candidate pool may be < 45 repos.")
    elif scores_1 and min(scores_1) >= max(scores_3):
        print("  [PASS]  Monotonicity: batch_1 min-score >= batch_3 max-score  ✅")
    else:
        print(
            f"  [INFO]  Score spread: batch_1 min={min(scores_1):.4f}, "
            f"batch_3 max={max(scores_3):.4f} "
            f"(MMoE reorders by multi-task utility, cosine order may differ)"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="gh-social-ml: Integrated Retrieval + Ranking Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Run for a specific user_id only.",
    )
    p.add_argument(
        "--onboard-first",
        action="store_true",
        help="Onboard mock users into Qdrant before running the pipeline.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the 24-hour recommendation cache and regenerate batches.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Adjust log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.onboard_first:
        _onboard_mock_users()

    from retrieval_engine import RetrievalEngine
    engine = RetrievalEngine()

    if args.user_id:
        _run_for_user(engine, args.user_id, no_cache=args.no_cache)
    else:
        users = engine.list_onboarded_users()
        if not users:
            print("\nNo onboarded users found.")
            print("Run with --onboard-first to onboard the mock users first.")
            sys.exit(0)

        logger.info("Found %d onboarded user(s). Running integrated pipeline...", len(users))
        for user_info in users:
            _run_for_user(engine, user_info["user_id"], no_cache=args.no_cache)

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == "__main__":
    main()
