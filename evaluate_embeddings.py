"""Evaluate semantic quality of repository embeddings stored in Qdrant."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from itertools import combinations
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from ingestion.config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL
from ingestion.embedding_pipeline import RepositoryEmbeddingPipeline
from ingestion.qdrant_store import QdrantRepositoryStore
from ingestion.repository_embedding import RepositoryEmbeddingConfig


QUALITATIVE_QUERIES = [
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
    parser = argparse.ArgumentParser(description="Evaluate repository embedding retrieval quality.")
    parser.add_argument("--corpus-json", default=None, help="Approved repository payload JSON file")
    parser.add_argument("--sample-size", type=int, default=50, help="Maximum repositories to evaluate")
    parser.add_argument("--query-count", type=int, default=10, help="Repository examples to query")
    parser.add_argument("--top-k", type=int, default=10, help="Nearest neighbors per query")
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--qdrant-api-key", default=QDRANT_API_KEY, help="Qdrant API key")
    parser.add_argument("--collection", default=QDRANT_COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="SentenceTransformer model")
    parser.add_argument("--compare-current", action="store_true", help="Compare ENN exact search with non-exact search")
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
    store.ensure_collection()

    if args.corpus_json:
        corpus = _load_corpus(args.corpus_json, args.sample_size)
        print(f"Indexing/evaluating {len(corpus)} approved repositories from {args.corpus_json}")
        embedded = pipeline.index_batch(corpus)
        points = [
            {"repo_id": item.repo_id, "payload": item.payload, "vector": item.final_embedding}
            for item in embedded
        ]
    else:
        print(f"Loading up to {args.sample_size} repository vectors from Qdrant collection {args.collection}")
        points = store.list_points(limit=args.sample_size, with_vectors=True)

    points = [point for point in points if point.get("vector")]
    if len(points) < 2:
        raise SystemExit("Need at least two repository vectors to evaluate semantic retrieval quality.")

    query_points = _select_query_points(points, args.query_count)
    query_reports = _evaluate_repository_queries(store, query_points, top_k=args.top_k, exact=True)
    metrics = _compute_metrics(points, query_reports)
    text_reports = _evaluate_text_queries(pipeline, top_k=args.top_k, exact=True)
    comparison = None
    if args.compare_current:
        current_reports = _evaluate_repository_queries(store, query_points, top_k=args.top_k, exact=False)
        comparison = _compare_reports(query_reports, current_reports)

    _print_repository_reports(query_reports)
    _print_metrics(metrics)
    _print_text_reports(text_reports)
    if comparison:
        _print_comparison(comparison)
    _print_recommendations(metrics)


def _load_corpus(path: str, sample_size: int) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = data.get("repositories") or data.get("items") or data.get("data") or []
    repos = []
    for item in data:
        payload = item.get("payload") if isinstance(item, dict) else None
        repos.append(payload if isinstance(payload, dict) else item)
    return repos[:sample_size]


def _select_query_points(points: list[dict], query_count: int) -> list[dict]:
    # The below category-balanced selection is for making the evaluation cover
    # more than one discovery/category cluster when the sample allows it.
    selected: list[dict] = []
    seen_categories = set()
    for point in points:
        category = _category(point)
        if category not in seen_categories:
            selected.append(point)
            seen_categories.add(category)
        if len(selected) >= query_count:
            return selected
    for point in points:
        if point not in selected:
            selected.append(point)
        if len(selected) >= query_count:
            break
    return selected


def _evaluate_repository_queries(
    store: QdrantRepositoryStore,
    query_points: list[dict],
    *,
    top_k: int,
    exact: bool,
) -> list[dict]:
    reports = []
    for point in query_points:
        started = perf_counter()
        neighbors = store.search(point["vector"], limit=top_k + 1, exact=exact)
        latency_ms = (perf_counter() - started) * 1000
        filtered = [item for item in neighbors if item.get("repo_id") != point.get("repo_id")][:top_k]
        reports.append(
            {
                "repo_id": point.get("repo_id"),
                "category": _category(point),
                "neighbors": filtered,
                "exact": exact,
                "latency_ms": latency_ms,
            }
        )
    return reports


def _evaluate_text_queries(pipeline: RepositoryEmbeddingPipeline, *, top_k: int, exact: bool) -> list[dict]:
    reports = []
    if pipeline.store is None:
        raise RuntimeError("pipeline.store is required for text-query evaluation")
    for query in QUALITATIVE_QUERIES:
        started = perf_counter()
        query_vector = pipeline.embedder.embed_text(query)
        embedding_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        neighbors = pipeline.store.search(query_vector, limit=top_k, exact=exact)
        latency_ms = (perf_counter() - started) * 1000
        reports.append(
            {
                "query": query,
                "neighbors": neighbors,
                "exact": exact,
                "embedding_ms": embedding_ms,
                "latency_ms": latency_ms,
            }
        )
    return reports


def _compare_reports(exact_reports: list[dict], current_reports: list[dict]) -> dict[str, float]:
    overlaps = []
    latency_deltas = []
    for exact_report, current_report in zip(exact_reports, current_reports):
        exact_ids = [item.get("repo_id") for item in exact_report["neighbors"]]
        current_ids = [item.get("repo_id") for item in current_report["neighbors"]]
        overlaps.append(len(set(exact_ids) & set(current_ids)) / max(len(exact_ids), 1))
        latency_deltas.append(float(exact_report["latency_ms"]) - float(current_report["latency_ms"]))
    return {
        "average_top_k_overlap": _safe_mean(overlaps),
        "average_exact_latency_ms": _safe_mean([float(item["latency_ms"]) for item in exact_reports]),
        "average_current_latency_ms": _safe_mean([float(item["latency_ms"]) for item in current_reports]),
        "average_latency_delta_ms": _safe_mean(latency_deltas),
    }


def _compute_metrics(points: list[dict], query_reports: list[dict]) -> dict[str, float | dict]:
    same_scores: list[float] = []
    cross_scores: list[float] = []
    same_category_ratios: list[float] = []
    has_same_category_hit = 0

    for report in query_reports:
        category = report["category"]
        neighbors = report["neighbors"]
        same_count = 0
        for neighbor in neighbors:
            score = float(neighbor["score"])
            if _payload_category(neighbor.get("payload", {})) == category:
                same_scores.append(score)
                same_count += 1
            else:
                cross_scores.append(score)
        if neighbors:
            same_category_ratios.append(same_count / len(neighbors))
        if same_count:
            has_same_category_hit += 1

    norms = [_norm(point["vector"]) for point in points]
    pairwise = [_cosine(a["vector"], b["vector"]) for a, b in combinations(points[:100], 2)]
    categories: dict[str, int] = {}
    for point in points:
        categories[_category(point)] = categories.get(_category(point), 0) + 1

    return {
        "average_same_category_similarity": _safe_mean(same_scores),
        "average_cross_category_similarity": _safe_mean(cross_scores),
        "category_clustering_quality": _safe_mean(same_category_ratios),
        "retrieval_consistency": has_same_category_hit / max(len(query_reports), 1),
        "vector_norm_mean": _safe_mean(norms),
        "vector_norm_stdev": pstdev(norms) if len(norms) > 1 else 0.0,
        "pairwise_similarity_mean": _safe_mean(pairwise),
        "pairwise_similarity_stdev": pstdev(pairwise) if len(pairwise) > 1 else 0.0,
        "pairwise_similarity_min": min(pairwise) if pairwise else 0.0,
        "pairwise_similarity_max": max(pairwise) if pairwise else 0.0,
        "category_counts": categories,
    }


def _print_repository_reports(reports: list[dict]) -> None:
    print("\nRepository nearest-neighbor checks")
    print("=" * 78)
    for report in reports:
        print(f"\nQuery repo: {report['repo_id']}  category={report['category']}")
        for index, neighbor in enumerate(report["neighbors"], 1):
            payload = neighbor.get("payload", {})
            print(
                f"  {index:>2}. score={neighbor['score']:.4f}  "
                f"repo={neighbor.get('repo_id')}  category={_payload_category(payload)}"
            )


def _print_metrics(metrics: dict) -> None:
    print("\nMetrics summary")
    print("=" * 78)
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")


def _print_comparison(comparison: dict[str, float]) -> None:
    print("\nENN vs current search comparison")
    print("=" * 78)
    print(f"average_top_k_overlap: {comparison['average_top_k_overlap']:.4f}")
    print(f"average_exact_latency_ms: {comparison['average_exact_latency_ms']:.2f}")
    print(f"average_current_latency_ms: {comparison['average_current_latency_ms']:.2f}")
    print(f"average_latency_delta_ms: {comparison['average_latency_delta_ms']:.2f}")


def _print_text_reports(reports: list[dict]) -> None:
    print("\nQualitative text-query checks")
    print("=" * 78)
    for report in reports:
        print(f"\nQuery: {report['query']}")
        print(
            f"  exact={report['exact']} embedding_ms={report['embedding_ms']:.2f} "
            f"qdrant_latency_ms={report['latency_ms']:.2f}"
        )
        for index, neighbor in enumerate(report["neighbors"], 1):
            payload = neighbor.get("payload", {})
            print(
                f"  {index:>2}. score={neighbor['score']:.4f}  "
                f"repo={neighbor.get('repo_id')}  category={_payload_category(payload)}"
            )


def _print_recommendations(metrics: dict) -> None:
    same = float(metrics["average_same_category_similarity"])
    cross = float(metrics["average_cross_category_similarity"])
    quality = float(metrics["category_clustering_quality"])
    spread = float(metrics["pairwise_similarity_stdev"])

    print("\nQualitative report")
    print("=" * 78)
    print(f"Strong retrieval signal: same-category avg={same:.4f}, cross-category avg={cross:.4f}")
    print(f"Weak retrieval signal: clustering quality={quality:.4f}, pairwise spread={spread:.4f}")
    if same <= cross:
        print("Potential issue: same-category neighbors are not scoring above cross-category neighbors.")
    if spread < 0.03:
        print("Potential issue: embeddings may be too compressed; many repositories look similarly close.")
    if quality < 0.35:
        print("Recommendation: increase topic/metadata tower weights or improve discovery/category labels.")
    elif same - cross < 0.05:
        print("Recommendation: slightly increase README/topic weights and re-run evaluation.")
    else:
        print("Recommendation: current tower weights show useful category separation; validate on a larger sample.")


def _category(point: dict) -> str:
    return _payload_category(point.get("payload", {}))


def _payload_category(payload: dict) -> str:
    return payload.get("discovery_category") or payload.get("category") or "Unknown"


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    denom = _norm(left) * _norm(right)
    if not denom:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denom


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


if __name__ == "__main__":
    main()
