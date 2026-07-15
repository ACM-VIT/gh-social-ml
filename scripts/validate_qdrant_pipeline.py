"""Validate repository embedding generation and Qdrant indexing."""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.repository_embedding import RepositoryEmbeddingConfig


SAMPLE_REPOSITORY = {
    # This is the shape acquisition must emit after receiving the canonical
    # backend repository mapping.
    "id": "00000000-0000-4000-8000-000000000001",
    "repo_id": "00000000-0000-4000-8000-000000000001",
    "github_id": "123456789",
    "full_name": "sample/repository-embedding-demo",
    "content_version": 1,
    "description": "Sample repository used to validate README, metadata, topic, and Qdrant vector indexing.",
    "primary_language": "Python",
    "star_count": 128,
    "fork_count": 12,
    "open_issues_count": 3,
    "pushed_days_ago": 2,
    "mentionable_users_count": 4,
    "delta_3d": 2,
    "delta_7d": 7,
    "delta_30d": 21,
    "topics": ["embeddings", "qdrant", "semantic-search"],
    "languages": ["Python"],
    "readme_length": 920,
    "readme_to_codebase_ratio": 0.03,
    "extracted_paragraphs": [
        "This sample repository demonstrates a production pipeline that embeds README documentation, "
        "repository metadata, and GitHub topics before storing the final repository vector in Qdrant.",
        "The validation flow creates or verifies the vector collection, upserts a deterministic point, "
        "and performs a semantic similarity search against the stored repository embedding.",
    ],
    "discovery_category": "Developer Tools",
    "discovery_band": "validation",
}


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate repository embeddings and Qdrant storage with a sample repository."
    )
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--qdrant-api-key", default=QDRANT_API_KEY, help="Qdrant API key")
    parser.add_argument("--collection", default=QDRANT_COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="SentenceTransformer model")
    parser.add_argument("--limit", type=int, default=5, help="Search result limit")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
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

    # The below validation path is for exercising the same bootstrap, upsert,
    # and search methods used by the production indexing flow.
    print(f"Verifying Qdrant collection: {args.collection} ({args.qdrant_url})")
    store.ensure_collection()

    print("Embedding and upserting sample repository...")
    result = pipeline.index_batch([SAMPLE_REPOSITORY])[0]
    print(f"Stored repo_id={result.repo_id} dim={len(result.final_embedding)} chunks={result.readme_chunks}")

    print("Running similarity search...")
    matches = store.search(result.final_embedding, limit=args.limit)
    for index, match in enumerate(matches, 1):
        print(f"{index}. score={match['score']:.4f} repo_id={match['repo_id']}")


if __name__ == "__main__":
    main()
