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
    limit: int = 100,
    batch_size: int = 10,
    existing_repos: set[str] | None = None,
) -> list:
    """
    Discover and enrich GitHub repositories via GraphQL only.

    Returns a list of EnrichmentResult objects. Each carries:
      .repo_id          — "owner/repo"
      .payload          — Osiris-compatible dict (star_count, language, topics, …)
      .raw_repository   — raw GraphQL response fields
      .readme           — ReadmeDocument (clean_text, extracted_paragraphs, …)
      .topics           — list[str]
      .languages        — dict[str, int]  (language → bytes)
    """
    from acquisition.github_graphql_client import GitHubGraphQLClient
    from acquisition.github_discovery import GitHubDiscoveryEngine, DiscoveryConfig
    from acquisition.repository_enricher import RepositoryEnricher

    client   = GitHubGraphQLClient(token=token)
    # Fetch a larger buffer of candidate repositories to account for filtering duplicates
    discovery_limit = limit + 50 if existing_repos else limit + 20
    config   = DiscoveryConfig(total_limit=discovery_limit)
    discovery = GitHubDiscoveryEngine(client, config=config)
    enricher  = RepositoryEnricher(graphql_client=client)

    # ── Step 1: Discovery ─────────────────────────────────────────────────────
    logger.info("Discovering repositories …")
    discovered = discovery.discover(limit=discovery_limit)
    logger.info("Discovered %d candidate repos", len(discovered))

    if existing_repos:
        new_discovered = []
        for r in discovered:
            full_name = r if isinstance(r, str) else r.get("full_name", "")
            if full_name not in existing_repos:
                new_discovered.append(r)
        logger.info(
            "Filtered out %d already existing repos from candidates. %d new candidates remain.",
            len(discovered) - len(new_discovered),
            len(new_discovered),
        )
        discovered = new_discovered

    # ── Step 2: Enrichment in batches ─────────────────────────────────────────
    logger.info("Enriching in batches of %d …", batch_size)
    enriched: list = []
    targets       = discovered[:limit]
    total_batches = (len(targets) + batch_size - 1) // batch_size

    for i in range(total_batches):
        batch = targets[i * batch_size : (i + 1) * batch_size]
        try:
            results = enricher.get_repositories_batch(batch)
            enriched.extend(results)
            logger.info(
                "  Batch %d/%d → +%d enriched  (total: %d)",
                i + 1, total_batches, len(results), len(enriched),
            )
        except Exception as exc:
            logger.warning("  Batch %d failed (%s). Falling back to one-by-one …", i + 1, exc)
            for repo in batch:
                full_name = repo if isinstance(repo, str) else repo.get("full_name", "")
                try:
                    r = enricher.enrich(full_name)
                    if r:
                        enriched.append(r)
                        logger.info("    ✓  %s", full_name)
                except Exception as exc2:
                    logger.warning("    ✗  %s: %s", full_name, exc2)

    logger.info("Acquisition complete — %d / %d repos enriched", len(enriched), limit)
    return enriched


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

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="gh-social-ml acquisition pipeline: Discovery → Enrichment → Quality Filter",
    )
    p.add_argument("--limit",            type=int, default=100,    help="Maximum number of repositories to fetch in this run (default: 100)")
    p.add_argument("--batch-size",       type=int, default=10,     help="Enrichment batch size (default: 10)")
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

    target_count = 1000
    kept = []

    # ── Step 2: Fetch & Index if under target ─────────────────────────────────
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

        enriched = run_acquisition(token, limit=fetch_limit, batch_size=args.batch_size, existing_repos=existing_repos)
        kept, dropped = filter_enriched(enriched, min_readme_chars=args.min_readme_chars)

        logger.info(
            "Quality filter: %d kept, %d dropped  (min_readme_chars=%d)",
            len(kept), len(dropped), args.min_readme_chars,
        )

        _print_summary(kept, dropped)

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

    # ── Step 3: Candidate Retrieval for Hardcoded Users ───────────────────────
    if current_count >= target_count:
        logger.info("Corpus target of 1000 reached. Executing L1 Candidate Retrieval Demo...")
        try:
            from scripts.mock_users import MOCK_USERS
            from retrieval import CandidateRetriever
            from scripts.user_onboarding import generate_interest_vector

            retriever = CandidateRetriever(
                db_connector=db,
                qdrant_url=args.qdrant_url,
                qdrant_api_key=args.qdrant_api_key
            )

            print("\n" + "═" * 80)
            print("                 L1 CANDIDATE RETRIEVAL PIPELINE DEMO")
            print("═" * 80)

            for user in MOCK_USERS:
                print(f"\n👤 USER: {user['full_name']} (@{user['user_id']})")
                print(f"   Bio: {user['bio']}")
                print(f"   Interests: {user['interests']}")
                print("   Generating user interest vector...")

                user_vector = generate_interest_vector(user)

                print("   Running multi-channel retrieval (Target: 120 Semantic, 30 Trending)...")
                candidates = retriever.retrieve_candidates(
                    user_embedding=user_vector,
                    user_interests=user["interests"]
                )

                print(f"   Successfully retrieved {len(candidates)} candidates.")
                print("-" * 80)
                print(f"{'#':<4} {'Repository':<42} {'Source':<10} {'Score/Stars':<12} {'Embedding Hydrated?'}")
                print("-" * 80)
                for i, c in enumerate(candidates, 1):
                    source = c.get("retrieval_source", "unknown")
                    score_str = "—"
                    if source == "semantic":
                        score_str = f"{c.get('retrieval_score', 0.0):.4f}"
                    elif source == "trending":
                        score_str = f"{c.get('star_count', 0):,} stars"
                    
                    has_embedding = "Yes (384-d)" if c.get("repo_embedding") is not None and len(c.get("repo_embedding")) == 384 else "No"
                    print(f"{i:<4} {c.get('full_name') or 'Unknown':<42} {source:<10} {score_str:<12} {has_embedding}")
                print("═" * 80)
        except Exception as exc:
            logger.error(f"Failed to run Candidate Retrieval demo: {exc}", exc_info=True)
    else:
        logger.warning(
            f"Approved corpus size is currently {current_count} repositories. "
            f"L1 Candidate Retrieval Demo will run once the corpus reaches the target of {target_count} repositories."
        )
