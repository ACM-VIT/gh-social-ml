#!/usr/bin/env python3
"""
scripts/e2e_mini_pipeline_test.py
==================================
End-to-end mini pipeline test for 5 repositories.

Flow:
  1. Live-fetch 5 repos from GitHub
  2. Quality filter (README, description, language checks)
  3. Gemma README Markdown enrichment (generates readme_markdown)
  4. Upsert into Supabase (PostgreSQL)
  5. Embed + index into Qdrant
  6. Run retrieval engine for one user (medhansh_generalist)
  7. Pull FULL metadata (including readme_markdown) from DB for ranked repos
  8. Display rich output

Usage:
    python3 scripts/e2e_mini_pipeline_test.py
"""

# Ensure the project root is on the path regardless of where we run from
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()


# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e.mini")

# Silence noisy sub-loggers during test
for noisy in ("httpx", "urllib3", "sentence_transformers", "pipeline.retrieval"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NUM_REPOS   = 5
TARGET_USER = "medhansh_generalist"

DIVIDER     = "═" * 90
SUBDIV      = "─" * 90


# ──────────────────────────────────────────────────────────────────────────── #
# STEP 1 — Acquire 5 live repos from GitHub
# ──────────────────────────────────────────────────────────────────────────── #
def step_acquire(token: str) -> list:
    print(f"\n{DIVIDER}")
    print("  STEP 1 — Live GitHub Acquisition (fetching 5 repositories)")
    print(DIVIDER)

    from acquisition.pipeline import run_acquisition

    enriched = run_acquisition(
        token,
        limit=NUM_REPOS,
        batch_size=NUM_REPOS,
        workers=2,
        existing_repos=set(),   # no deduplication — fresh fetch
    )
    logger.info("Acquired %d repositories from GitHub.", len(enriched))
    return enriched


# ──────────────────────────────────────────────────────────────────────────── #
# STEP 2 — Quality filter
# ──────────────────────────────────────────────────────────────────────────── #
def step_quality_filter(enriched: list) -> list:
    print(f"\n{DIVIDER}")
    print("  STEP 2 — Quality Filter")
    print(DIVIDER)

    from main import filter_enriched
    kept, dropped = filter_enriched(enriched, min_readme_chars=200)

    print(f"  ✅  Kept   : {len(kept)}")
    print(f"  ❌  Dropped: {len(dropped)}")
    for r, reasons in dropped:
        print(f"     └─ {r.repo_id}: {' | '.join(reasons)}")

    if not kept:
        print("\n  ⚠️  No repositories passed quality filter. Exiting early.")
        sys.exit(0)
    return kept


# ──────────────────────────────────────────────────────────────────────────── #
# STEP 3 — Gemma README Markdown enrichment
# ──────────────────────────────────────────────────────────────────────────── #
def step_gemma_enrichment(kept: list) -> list:
    print(f"\n{DIVIDER}")
    print("  STEP 3 — Gemma README Markdown Enrichment")
    print(DIVIDER)

    from utils.groq_client import generate_readme_markdown
    from utils.readme_processor import process_markdown

    for r in kept:
        p = r.payload
        raw_readme = p.get("readme", "") or p.get("readme_summary", "") or p.get("description", "")
        if not raw_readme:
            logger.warning("No README content for %s — skipping Groq enrichment.", r.repo_id)
            continue

        # Clean raw text first
        readme_doc = process_markdown(raw_readme)
        clean_text = readme_doc.clean_text
        if not clean_text:
            logger.warning("ReadmeProcessor returned empty clean_text for %s.", r.repo_id)
            continue

        # Generate structured Markdown via Groq API
        md = generate_readme_markdown(clean_text[:3000])   # cap to avoid huge token usage
        if md:
            p["readme_markdown"] = md
            readme_doc.readme_markdown = md
            r.readme = readme_doc
            logger.info("✅ Groq generated Markdown for %s (%d chars).", r.repo_id, len(md))
        else:
            logger.warning("Groq returned empty result for %s.", r.repo_id)

    return kept


# ──────────────────────────────────────────────────────────────────────────── #
# STEP 4 — Upsert into Supabase
# ──────────────────────────────────────────────────────────────────────────── #
def step_upsert_db(kept: list, db) -> None:
    print(f"\n{DIVIDER}")
    print("  STEP 4 — Supabase Upsert")
    print(DIVIDER)

    saved = db.upsert_repositories(kept)
    logger.info("Upserted %d repositories into Supabase.", saved)


# ──────────────────────────────────────────────────────────────────────────── #
# STEP 5 — Embed + index into Qdrant
# ──────────────────────────────────────────────────────────────────────────── #
def step_embed_qdrant(kept: list) -> bool:
    """Returns True if Qdrant indexing succeeded, False if Qdrant is down."""
    print(f"\n{DIVIDER}")
    print("  STEP 5 — Embedding + Qdrant Indexing")
    print(DIVIDER)

    # Quick reachability check first
    import urllib.request
    import urllib.error
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        urllib.request.urlopen(f"{qdrant_url}/healthz", timeout=3)
    except Exception:
        print(f"  ⚠️  Qdrant is not reachable at {qdrant_url}.")
        print("  ℹ️  Skipping Qdrant embedding. Retrieval will use DB-direct fallback.")
        print("  ➡️  To start Qdrant locally: docker run -p 6333:6333 qdrant/qdrant")
        return False

    from main import index_approved_repositories
    indexed = index_approved_repositories(kept)
    logger.info("Indexed %d repository vectors into Qdrant.", len(indexed))
    return True


# ──────────────────────────────────────────────────────────────────────────── #
# STEP 6 — Retrieval Engine (onboard user + fetch ranked batches)
# ──────────────────────────────────────────────────────────────────────────── #
def step_retrieve_fallback_db(kept: list, db) -> dict:
    """DB-direct fallback when Qdrant is unavailable.
    Returns a batches dict with the ingested repos as batch_1.
    """
    print(f"  ℹ️  Using DB-direct fallback retrieval (Qdrant offline).")
    items = []
    for r in kept:
        p = r.payload
        items.append({
            "repo_id":          r.repo_id,
            "full_name":        p.get("id") or r.repo_id,
            "final_score":      p.get("star_count", 0) / 1000.0,  # use star count as proxy score
            "retrieval_source": "db_direct_fallback",
            "primary_language": p.get("primary_language", ""),
        })
    # Sort by star count descending
    items.sort(key=lambda x: x["final_score"], reverse=True)
    return {"batch_1": items, "batch_2": [], "batch_3": []}


def step_retrieve(kept: list, db, qdrant_ok: bool = True):
    print(f"\n{DIVIDER}")
    print(f"  STEP 6 — Retrieval Engine for user: {TARGET_USER}")
    print(DIVIDER)

    if not qdrant_ok:
        return step_retrieve_fallback_db(kept, db)

    from scripts.mock_users import MOCK_USERS
    from scripts.user_onboarding import onboard_user
    from retrieval_engine import RetrievalEngine

    target_user_data = next(u for u in MOCK_USERS if u["user_id"] == TARGET_USER)

    # Onboard the user so Qdrant has their profile
    onboard_user(user_id=TARGET_USER, user_data=target_user_data)
    logger.info("User '%s' onboarded into Qdrant.", TARGET_USER)

    engine = RetrievalEngine()

    # Invalidate any cached batch so we get a fresh result
    if db and db.enabled:
        conn = None
        try:
            conn = db.connect()
            cur  = conn.cursor()
            cur.execute("DELETE FROM user_recommendation_batches WHERE user_id = %s;", (TARGET_USER,))
            conn.commit()
        except Exception:
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    batches = engine.fetch_onboarding_batches(TARGET_USER)
    return batches


# ──────────────────────────────────────────────────────────────────────────── #
# STEP 7 — Fetch full metadata from DB for ranked repos
# ──────────────────────────────────────────────────────────────────────────── #
def fetch_full_metadata(repo_ids: list[str], db) -> dict[str, dict]:
    """Given a list of full_name or UUID repo_ids, fetch complete metadata from DB."""
    if not db or not db.enabled:
        return {}

    conn = None
    metadata: dict[str, dict] = {}
    try:
        conn = db.connect()
        cur  = conn.cursor()
        # Fetch by UUID or full_name
        for rid in repo_ids:
            try:
                cur.execute(
                    """
                    SELECT
                        repo_id, full_name, owner_id, repo_name,
                        description, primary_language, language_used,
                        topics, star_count, forks_count, pr_count,
                        likes_count, comments_count, saves_count, views_count,
                        readme_summary, readme_markdown,
                        github_repo_url, created_at, updated_at
                    FROM Repo
                    WHERE full_name = %s OR CAST(repo_id AS TEXT) = %s
                    LIMIT 1;
                    """,
                    (rid, rid),
                )
                row = cur.fetchone()
                if row:
                    metadata[rid] = {
                        "repo_id":          str(row[0]),
                        "full_name":        row[1],
                        "owner_id":         row[2],
                        "repo_name":        row[3],
                        "description":      row[4] or "—",
                        "primary_language": row[5] or "Unknown",
                        "language_used":    row[6] or [],
                        "topics":           row[7] or [],
                        "star_count":       row[8] or 0,
                        "forks_count":      row[9] or 0,
                        "pr_count":         row[10] or 0,
                        "likes_count":      row[11] or 0,
                        "comments_count":   row[12] or 0,
                        "saves_count":      row[13] or 0,
                        "views_count":      row[14] or 0,
                        "readme_summary":   row[15] or "",
                        "readme_markdown":  row[16] or "",
                        "github_repo_url":  row[17] or "",
                        "created_at":       str(row[18]),
                        "updated_at":       str(row[19]),
                    }
            except Exception as e:
                logger.warning("Could not fetch metadata for '%s': %s", rid, e)
    except Exception as e:
        logger.error("DB metadata fetch error: %s", e)
    finally:
        if conn:
            conn.close()
    return metadata


# ──────────────────────────────────────────────────────────────────────────── #
# STEP 8 — Rich display
# ──────────────────────────────────────────────────────────────────────────── #
def display_results(batches: dict, db) -> None:
    print(f"\n{DIVIDER}")
    print(f"  STEP 8 — Rich Output for User: {TARGET_USER}")
    print(DIVIDER)

    # Collect all ranked repo IDs from all batches
    all_ranked_ids: list[str] = []
    for batch_key in ["batch_1", "batch_2", "batch_3"]:
        for item in batches.get(batch_key, []):
            rid = item.get("full_name") or item.get("repo_id") or ""
            if rid and rid not in all_ranked_ids:
                all_ranked_ids.append(rid)

    # Fetch full metadata from DB
    metadata = fetch_full_metadata(all_ranked_ids, db)

    for batch_key, batch_title in [
        ("batch_1", "🏆 Batch 1 — Top Picks"),
        ("batch_2", "🔥 Batch 2 — Mid Tier"),
        ("batch_3", "🎲 Batch 3 — Discovery"),
    ]:
        items = batches.get(batch_key, [])
        print(f"\n  {batch_title} ({len(items)} repos)")
        print(f"  {SUBDIV}")

        for rank_idx, item in enumerate(items, 1):
            rid  = item.get("full_name") or item.get("repo_id") or "?"
            meta = metadata.get(rid, {})

            # Scored from ranking engine
            score = item.get("final_score") or item.get("cosine_score") or 0.0
            source = item.get("retrieval_source", "?")

            print(f"\n  ── #{rank_idx}  {rid}  (score={score:.4f}, source={source})")
            print(f"  {'─' * 60}")

            if meta:
                desc     = (meta["description"] or "—")[:120]
                lang     = meta["primary_language"]
                stars    = f"{meta['star_count']:,}"
                forks    = f"{meta['forks_count']:,}"
                prs      = f"{meta['pr_count']:,}"
                likes    = f"{meta['likes_count']:,}"
                views    = f"{meta['views_count']:,}"
                saves    = f"{meta['saves_count']:,}"
                comments = f"{meta['comments_count']:,}"

                langs = meta["language_used"]
                if isinstance(langs, str):
                    try:
                        langs = json.loads(langs)
                    except Exception:
                        langs = []
                langs_str = ", ".join(langs[:5]) if isinstance(langs, list) else str(langs)

                topics = meta["topics"]
                if isinstance(topics, str):
                    try:
                        topics = json.loads(topics)
                    except Exception:
                        topics = []
                topics_str = ", ".join(topics[:5]) if isinstance(topics, list) else str(topics)

                print(f"  📄  Description    : {desc}")
                print(f"  🔗  GitHub URL     : {meta['github_repo_url']}")
                print(f"  💻  Language       : {lang}")
                print(f"  🌐  All Languages  : {langs_str or '—'}")
                print(f"  🏷️   Topics         : {topics_str or '—'}")
                print(f"  ⭐  Stars          : {stars}")
                print(f"  🍴  Forks          : {forks}")
                print(f"  🔀  Pull Requests  : {prs}")
                print(f"  ❤️   Likes          : {likes}")
                print(f"  💬  Comments       : {comments}")
                print(f"  🔖  Saves          : {saves}")
                print(f"  👁️   Views          : {views}")
                print(f"  📅  Created At     : {meta['created_at']}")
                print(f"  🔄  Updated At     : {meta['updated_at']}")

                # README Markdown section
                readme_md = meta.get("readme_markdown", "").strip()
                if readme_md:
                    print(f"\n  📝  README (Groq Formatted Markdown):")
                    print(f"  {'·' * 60}")
                    # Print the first 25 lines of the Markdown for readability
                    md_lines = readme_md.splitlines()
                    for line in md_lines[:25]:
                        print(f"      {line}")
                    if len(md_lines) > 25:
                        print(f"      ... [{len(md_lines) - 25} more lines]")
                    print(f"  {'·' * 60}")
                else:
                    print(f"\n  📝  README (Groq Formatted Markdown): ⚠️  Not generated yet")
                    summary = meta.get("readme_summary", "").strip()
                    if summary:
                        print(f"  Raw README Summary: {summary[:300]}")
            else:
                print(f"  ⚠️  No DB metadata found for this repo (may be from Qdrant cache only)")
                print(f"      Engine metadata: {json.dumps({k: v for k, v in item.items() if k != 'embedding'}, default=str)[:300]}")

    print(f"\n{DIVIDER}")
    print("  ✅  End-to-end mini pipeline test COMPLETE.")
    print(DIVIDER)


# ──────────────────────────────────────────────────────────────────────────── #
# MAIN
# ──────────────────────────────────────────────────────────────────────────── #
def main():
    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "your_github_token_here":
        print("❌ ERROR: Set GITHUB_TOKEN in your .env file first.")
        sys.exit(1)

    # Init DB
    from database import PostgreSQLConnector
    db = PostgreSQLConnector()
    if db.enabled and db.verify_connection():
        db.init_db()
    else:
        logger.warning("DB not connected — metadata display will be limited.")

    print("\n" + DIVIDER)
    print("   gh-social-ml · END-TO-END MINI PIPELINE TEST (5 repos)")
    print(DIVIDER)

    # Run all steps
    enriched   = step_acquire(token)
    kept       = step_quality_filter(enriched)
    kept       = step_gemma_enrichment(kept)
    step_upsert_db(kept, db)
    qdrant_ok  = step_embed_qdrant(kept)
    batches    = step_retrieve(kept, db, qdrant_ok=qdrant_ok)
    display_results(batches, db)


if __name__ == "__main__":
    main()
