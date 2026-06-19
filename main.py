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
    config   = DiscoveryConfig(total_limit=limit + 20)   # small buffer to hit the target
    discovery = GitHubDiscoveryEngine(client, config=config)
    enricher  = RepositoryEnricher(graphql_client=client)

    # ── Step 1: Discovery ─────────────────────────────────────────────────────
    logger.info("Discovering repositories …")
    discovered = discovery.discover(limit=limit + 20)
    logger.info("Discovered %d candidate repos", len(discovered))

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

    from ingestion.config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL
    from ingestion.embedding_pipeline import RepositoryEmbeddingPipeline
    from ingestion.qdrant_store import QdrantRepositoryStore
    from ingestion.repository_embedding import RepositoryEmbeddingConfig

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
    p.add_argument("--limit",            type=int, default=100,    help="Target number of repos (default: 100)")
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

    enriched          = run_acquisition(token, limit=args.limit, batch_size=args.batch_size)
    kept, dropped     = filter_enriched(enriched, min_readme_chars=args.min_readme_chars)

    logger.info(
        "Quality filter: %d kept, %d dropped  (min_readme_chars=%d)",
        len(kept), len(dropped), args.min_readme_chars,
    )

    _print_summary(kept, dropped)

    if not args.no_index_qdrant:
        # The below block is for automatic Qdrant indexing after filtering; use
        # --no-index-qdrant only for acquisition-only local runs.
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
            logger.error("Qdrant indexing failed; continuing to database ingestion: %s", exc)

    # ── Step 3: Database Ingestion ────────────────────────────────────────────
    from database import PostgreSQLConnector
    db = PostgreSQLConnector()
    if db.enabled:
        if db.verify_connection():
            try:
                db.init_db()
                saved_count = db.upsert_repositories(kept)
                total = db.get_repo_count()
                logger.info(
                    f"Database ingestion complete: {saved_count} upserted this run, "
                    f"{total} total repos in database."
                )
            except Exception as db_exc:
                logger.error(f"Failed to ingest repositories into database: {db_exc}")
        else:
            logger.error(
                "Database connection test failed. Check DATABASE_URL in .env. "
                "Skipping database upload."
            )
    else:
        logger.info("DATABASE_URL not set; skipping database upload.")
