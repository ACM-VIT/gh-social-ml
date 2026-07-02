"""
gh-social-ml  ·  Acquisition Pipeline
======================================

Stage 1 of the full architecture:
  Discovery  (GraphQL search across categories + maturity bands)
      ↓
  Enrichment  (metadata, languages, topics, README, star deltas)
      ↓
  Quality Filter  (drop no-README shells, content-free repos)
      ↓
  EnrichmentResult list  (ready for Stage 2 — Feature Extraction)

Usage:
    python3 main.py [--limit N] [--batch-size N] [--min-readme-chars N] [--log-level LEVEL]

Environment:
    GITHUB_TOKEN  — required, set in .env"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("pipeline.acquisition")


# ══════════════════════════════════════════════════════════════════════════════
#  ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def run_acquisition(
    token: str,
    *,
    limit: int = 150,
    batch_size: int = 15,
    workers: int = 4,
    existing_repos: set[str] | None = None,
) -> list:
    """
    Discover and enrich GitHub repositories via GraphQL only.

    Delegates to acquisition.pipeline.run_acquisition which implements
    concurrent enrichment via ThreadPoolExecutor.
    """
    from acquisition.pipeline import run_acquisition as _run_acquisition
    return _run_acquisition(
        token,
        limit=limit,
        batch_size=batch_size,
        workers=workers,
        existing_repos=existing_repos,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY FILTER
# ══════════════════════════════════════════════════════════════════════════════

# Signals used to classify a repo as a content-free shell:
#   • readme_length < min_readme_chars  → no README or too thin to embed
#   • no description AND no languages AND no topics
#     → the repo has nothing meaningful (bookmark list, config dump, etc.)

def filter_enriched(
    enriched: list,
    *,
    min_readme_chars: int = 200,
) -> tuple[list, list]:
    """
    Split enriched repos into (kept, dropped).

    dropped is a list of (EnrichmentResult, list[str]) tuples where the
    second element is the list of reasons the repo was dropped.

    Args:
        enriched:         Raw output of run_acquisition().
        min_readme_chars: Repos whose README is shorter than this are dropped.
                          Default 200 — enough for a meaningful description but
                          short enough not to penalise compact technical READMEs.

    Returns:
        kept    — clean list ready for Stage 2 (Feature Extraction)
        dropped — audit list so the team can inspect what was filtered
    """
    kept:    list = []
    dropped: list = []   # list of (EnrichmentResult, reasons: list[str])

    for r in enriched:
        p       = r.payload
        reasons = []

        # ── Check 1: README quality ───────────────────────────────────────────
        readme_len = p.get("readme_length", 0)
        if readme_len == 0:
            reasons.append("no README")
        elif readme_len < min_readme_chars:
            reasons.append(f"README too thin ({readme_len} chars < {min_readme_chars})")

        # ── Check 2: Content-free shell ───────────────────────────────────────
        has_description = bool((p.get("description") or "").strip())
        has_languages   = bool(p.get("languages"))
        has_topics      = bool(p.get("topics"))

        if not has_description and not has_languages and not has_topics:
            reasons.append("shell repo: no description, languages, or topics")

        if reasons:
            dropped.append((r, reasons))
        else:
            kept.append(r)

    return kept, dropped


def index_approved_repositories(
    approved: list,
    *,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_collection: str | None = None,
    embedding_model: str | None = None,
) -> list:
    """Embed approved repositories and persist their vectors to Qdrant."""
    if not approved:
        logger.warning("Skipping Qdrant indexing because no repositories passed the filter.")
        return []

    from config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from embedding.qdrant_store import QdrantRepositoryStore
    from embedding.repository_embedding import RepositoryEmbeddingConfig

    embedding_config = RepositoryEmbeddingConfig(
        model_name=embedding_model or os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2",
    )
    store = QdrantRepositoryStore(
        url=qdrant_url or QDRANT_URL,
        api_key=qdrant_api_key or QDRANT_API_KEY,
        collection_name=qdrant_collection or QDRANT_COLLECTION_NAME,
        vector_size=embedding_config.embedding_dim,
    )
    pipeline = RepositoryEmbeddingPipeline(config=embedding_config, store=store)
    return pipeline.index_batch(approved)


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(kept: list, dropped: list) -> None:
    width = 95

    # ── Kept repos ────────────────────────────────────────────────────────────
    if not kept:
        logger.warning("No repos passed the quality filter.")
    else:
        sorted_repos = sorted(kept, key=lambda r: r.payload.get("star_count", 0), reverse=True)
        print(f"\n{'═' * width}")
        print(f"  ✅  {len(kept)} repos passed quality filter")
        print(f"{'═' * width}")
        print(f"{'#':<4} {'Repository':<42} {'⭐ Stars':>8} {'Language':<14} {'README':>8}  Topics")
        print("─" * width)
        for i, r in enumerate(sorted_repos, 1):
            p = r.payload
            topics_str = ", ".join(p.get("topics", [])[:3]) or "—"
            print(
                f"{i:<4} {p['id']:<42} {p.get('star_count', 0):>8,}  "
                f"{p.get('primary_language', 'Unknown'):<14} "
                f"{p.get('readme_length', 0):>7,}c  {topics_str}"
            )
        print(f"{'═' * width}\n")

    # ── Dropped repos ─────────────────────────────────────────────────────────
    if dropped:
        print(f"{'─' * width}")
        print(f"  ⚠️   {len(dropped)} repos dropped by quality filter")
        print(f"{'─' * width}")
        print(f"{'Repository':<45}  {'⭐ Stars':>8}  Reason")
        print("─" * width)
        for r, reasons in sorted(dropped, key=lambda x: x[0].payload.get("star_count", 0), reverse=True):
            stars      = r.payload.get("star_count", 0)
            reason_str = " | ".join(reasons)
            print(f"  {r.repo_id:<43}  {stars:>8,}  {reason_str}")
        print(f"{'─' * width}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _positive_int(value: str) -> int:
    """argparse type that rejects zero and negative integers."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (>= 1), got {n}")
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="gh-social-ml acquisition pipeline: Discovery → Enrichment → Quality Filter",
    )
    p.add_argument("--limit",            type=_positive_int, default=150,    help="Maximum number of repositories to fetch in this run (default: 150)")
    p.add_argument("--batch-size",       type=_positive_int, default=15,     help="Enrichment batch size (default: 15)")
    p.add_argument("--workers",          type=_positive_int, default=4,      help="Concurrent enrichment workers, must be >= 1 (default: 4)")
    p.add_argument("--min-readme-chars", type=int, default=200,    help="Minimum README length to keep a repo (default: 200)")
    p.add_argument("--index-qdrant",     action="store_true",      help="Deprecated: Qdrant indexing now runs by default")
    p.add_argument("--no-index-qdrant",  action="store_true",      help="Skip automatic Qdrant indexing after filtering")
    p.add_argument("--qdrant-url",       type=str, default=None,    help="Qdrant URL (default: QDRANT_URL or http://localhost:6333)")
    p.add_argument("--qdrant-api-key",   type=str, default=None,    help="Qdrant API key (default: QDRANT_API_KEY)")
    p.add_argument("--qdrant-collection", type=str, default=None,   help="Qdrant collection name override")
    p.add_argument("--embedding-model",  type=str, default=None,    help="SentenceTransformer model override")
    p.add_argument("--log-level",        type=str, default="INFO", help="Logging level (default: INFO)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "your_github_token_here":
        print("❌  ERROR: Set GITHUB_TOKEN in your .env file first.")
        sys.exit(1)

    _setup_logging(args.log_level)

    logger.info("╔══════════════════════════════════╗")
    logger.info("║  gh-social-ml  ·  Acquisition    ║")
    logger.info("╚══════════════════════════════════╝")

    # ── Step 1: Database Check ────────────────────────────────────────────────
    from database import PostgreSQLConnector
    db = PostgreSQLConnector()
    
    current_count = 0
    db_verified = False
    if db.enabled:
        if db.verify_connection():
            try:
                db.init_db()
                current_count = db.get_repo_count()
                logger.info(f"Current repositories in PostgreSQL database: {current_count}")
                db_verified = True
            except Exception as db_exc:
                logger.error(f"Failed to query database repository count: {db_exc}")
        else:
            logger.warning("Database connection failed. Ingestion/hydration will be disabled. Check DATABASE_URL in .env.")
    else:
        logger.info("Database connector is not enabled. Ingestion/hydration will be disabled.")

    target_count = 3500
    kept = []

    # ── Step 2: Fetch & Index if under target ─────────────────────────────────
    while True:
        # If database is enabled and connected, we target reaching target_count.
        # Otherwise, we just fetch args.limit repositories directly.
        if db_verified:
            should_fetch = current_count < target_count
            fetch_limit = min(target_count - current_count, args.limit)
        else:
            should_fetch = True
            fetch_limit = args.limit

        if not should_fetch:
            logger.info(f"Approved corpus has {current_count} repositories (>= {target_count}). Skipping new repository acquisition.")
            break
        else:
            if db_verified:
                logger.info(f"Approved corpus has {current_count} repositories. Fetching up to {fetch_limit} repositories to reach the {target_count} target...")
            else:
                logger.info(f"Database ingestion is disabled. Fetching exactly {fetch_limit} repositories for local execution...")

            # Load existing repos to filter out duplicates in run_acquisition
            existing_repos = set()
            if db.enabled and db_verified:
                conn = None
                try:
                    conn = db.connect()
                    cursor = conn.cursor()
                    cursor.execute("SELECT full_name FROM Repo;")
                    existing_repos = {row[0] for row in cursor.fetchall()}
                    logger.info(f"Loaded {len(existing_repos)} existing repository names from database.")
                except Exception as e:
                    logger.warning(f"Could not fetch existing repository names: {e}")
                finally:
                    if conn:
                        try:
                            conn.close()
                        except Exception:
                            pass

            enriched = run_acquisition(token, limit=fetch_limit, batch_size=args.batch_size, workers=args.workers, existing_repos=existing_repos)
            if not enriched:
                logger.warning("No new repositories acquired in this iteration. Stopping to avoid infinite loop.")
                break

            kept, dropped = filter_enriched(enriched, min_readme_chars=args.min_readme_chars)

            logger.info(
                "Quality filter: %d kept, %d dropped  (min_readme_chars=%d)",
                len(kept), len(dropped), args.min_readme_chars,
            )

            _print_summary(kept, dropped)

            if not kept:
                logger.warning("No repositories passed the quality filter. Stopping to avoid infinite reacquisition loop.")
                break

            # PostgreSQL Ingestion (First to ensure consistency and prevent orphaned points in Qdrant)
            db_ingestion_success = False
            if kept and db_verified:
                try:
                    saved_count = db.upsert_repositories(kept)
                    current_count = db.get_repo_count()
                    logger.info(
                        f"Database ingestion complete: {saved_count} upserted this run, "
                        f"{current_count} total repos in database."
                    )
                    db_ingestion_success = True
                except Exception as db_exc:
                    logger.error(f"Failed to ingest repositories into database: {db_exc}")
                    break
            elif kept and not db_verified:
                # If DB is not enabled/verified, treat as success to proceed with Qdrant indexing
                db_ingestion_success = True

            # Qdrant Indexing
            if kept and db_ingestion_success and not args.no_index_qdrant:
                try:
                    indexed = index_approved_repositories(
                        kept,
                        qdrant_url=args.qdrant_url,
                        qdrant_api_key=args.qdrant_api_key,
                        qdrant_collection=args.qdrant_collection,
                        embedding_model=args.embedding_model,
                    )
                    logger.info("Qdrant indexing complete: %d repository vectors stored", len(indexed))
                except Exception as exc:
                    logger.error("Qdrant indexing failed: %s", exc)
                    break
                    
            if not db_verified:
                break

    # ── Step 3: Integrated Candidate Retrieval + Ranking Demo ──────────────────
    if current_count >= target_count:
        logger.info("Corpus target of 3500 reached. Executing Integrated Retrieval + Ranking Demo...")
        try:
            from scripts.mock_users import MOCK_USERS
            from scripts.user_onboarding import onboard_user
            from retrieval_engine import RetrievalEngine

            # 1. Onboard the mock users to Qdrant so the retrieval engine can fetch their profiles
            logger.info("Onboarding mock users to Qdrant...")
            for user in MOCK_USERS:
                onboard_user(user_id=user["user_id"], user_data=user)

            engine = RetrievalEngine(
                qdrant_url=args.qdrant_url,
                qdrant_api_key=args.qdrant_api_key,
            )

            print("\n" + "═" * 80)
            print("         INTEGRATED L1 RETRIEVAL & L2 MMoE RANKING DEMO")
            print("═" * 80)

            for user in MOCK_USERS:
                uid = user["user_id"]
                print(f"\n👤 USER: {user['full_name']} (@{uid})")
                print(f"   Bio: {user['bio']}")
                print(f"   Interests: {user['interests']}")
                print("   Fetching and ranking recommendation batches (bypassing cache)...")

                # Force invalidate cache for demo freshness
                if engine.db and engine.db.enabled:
                    conn = None
                    try:
                        conn = engine.db.connect()
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM user_recommendation_batches WHERE user_id = %s;", (uid,))
                        conn.commit()
                    except Exception:
                        if conn:
                            conn.rollback()
                    finally:
                        if conn:
                            conn.close()

                batches = engine.fetch_onboarding_batches(uid)

                for batch_name, title in [
                    ("batch_1", "Batch 1 — Top Picks"),
                    ("batch_2", "Batch 2 — Mid Tier"),
                    ("batch_3", "Batch 3 — Discovery"),
                ]:
                    batch = batches.get(batch_name, [])
                    print(f"\n  ── {title} ({len(batch)} repos) ──")
                    print(f"  {'#':<3} {'MMoE Score':>10}  {'Source':<12} {'Repo':<42} Language")
                    print(f"  {'-'*3} {'-'*10}  {'-'*12} {'-'*42} {'-'*12}")
                    for i, item in enumerate(batch, 1):
                        score = item.get("final_score") or item.get("cosine_score") or 0.0
                        repo_name = (item.get("full_name") or item.get("repo_id") or "?")[:42]
                        source = item.get("retrieval_source", "?")[:12]
                        lang = (item.get("primary_language") or "")[:12]
                        print(f"  {i:<3} {score:>10.4f}  {source:<12} {repo_name:<42} {lang}")
                print("═" * 80)
        except Exception as exc:
            logger.error(f"Failed to run Integrated Retrieval + Ranking demo: {exc}", exc_info=True)
    else:
        logger.warning(
            f"Approved corpus size is currently {current_count} repositories. "
            f"Integrated Retrieval + Ranking Demo will run once the corpus reaches the target of {target_count} repositories."
        )
