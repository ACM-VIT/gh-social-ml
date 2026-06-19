"""Run qualitative semantic retrieval queries against repository embeddings."""

from __future__ import annotations

import argparse
import logging
import os
from time import perf_counter

from dotenv import load_dotenv

from ingestion.config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL
from ingestion.embedding_pipeline import RepositoryEmbeddingPipeline
from ingestion.qdrant_store import QdrantRepositoryStore
from ingestion.repository_embedding import RepositoryEmbeddingConfig


DEFAULT_QUERIES = [
    "large language model framework",
    "frontend ui library",
    "database ORM",
    "computer vision",
    "devops kubernetes",
]


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic text-query retrieval tests.")
    parser.add_argument("queries", nargs="*", help="Optional custom queries")
    parser.add_argument("--top-k", type=int, default=10, help="Results per query")
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--qdrant-api-key", default=QDRANT_API_KEY, help="Qdrant API key")
    parser.add_argument("--collection", default=QDRANT_COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="SentenceTransformer model")
    parser.add_argument("--compare-current", action="store_true", help="Also run the previous non-exact Qdrant search")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    _setup_logging(args.log_level)

    config = RepositoryEmbeddingConfig(model_name=args.model)
    store = QdrantRepositoryStore(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        collection_name=args.collection,
        vector_size=config.embedding_dim,
    )
    pipeline = RepositoryEmbeddingPipeline(config=config, store=store)
    queries = args.queries or DEFAULT_QUERIES

    # The below loop is for quick qualitative inspection of whether natural
    # language intents retrieve repositories in the expected semantic area.
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 78)
        started = perf_counter()
        query_vector = pipeline.embedder.embed_text(query)
        embedding_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        matches = store.search(query_vector, limit=args.top_k, exact=True)
        exact_ms = (perf_counter() - started) * 1000
        print(f"Search mode: ENN exact=True  embedding_ms={embedding_ms:.2f}  qdrant_latency_ms={exact_ms:.2f}")
        for index, match in enumerate(matches, 1):
            payload = match.get("payload", {})
            category = payload.get("discovery_category") or payload.get("category") or "Unknown"
            print(f"{index:>2}. score={match['score']:.4f}  repo={match.get('repo_id')}  category={category}")
        if args.compare_current:
            started = perf_counter()
            current_matches = store.search(query_vector, limit=args.top_k, exact=False)
            current_ms = (perf_counter() - started) * 1000
            overlap = _ranked_overlap(matches, current_matches)
            print(
                f"Current search comparison: exact=False qdrant_latency_ms={current_ms:.2f} "
                f"top_k_overlap={overlap}/{args.top_k}"
            )


def _ranked_overlap(left: list[dict], right: list[dict]) -> int:
    left_ids = {item.get("repo_id") for item in left}
    right_ids = {item.get("repo_id") for item in right}
    return len(left_ids & right_ids)


if __name__ == "__main__":
    main()
