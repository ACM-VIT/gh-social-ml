"""Run exact nearest-neighbor evaluation over repository vectors in Qdrant."""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from ingestion.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
)
from ingestion.qdrant_store import QdrantRepositoryStore
from ingestion.repository_embedding import RepositoryEmbeddingConfig


BATCH_SIZE = 10
DEFAULT_TOP_K = 5
INSPECT_CATEGORY_GROUPS = {
    "Databases": ("database", "storage", "sql", "postgres", "mysql", "redis", "mongo"),
    "Backend": ("backend", "api", "server", "graphql", "rest"),
    "DevOps": ("devops", "infrastructure", "security", "kubernetes", "docker", "cloud"),
    "Frontend": ("frontend", "web", "ui", "react", "vue", "angular"),
    "LLM": ("llm", "large language", "language model", "generative ai"),
    "RAG": ("rag", "retrieval augmented", "retrieval-augmented", "vector database"),
    "Computer Vision": ("computer vision", "vision", "image", "opencv"),
    "Robotics": ("robotics", "robot", "ros"),
}


@dataclass(slots=True)
class NeighborRecord:
    query_repo: str
    query_category: str
    neighbor_repo: str
    neighbor_category: str
    score: float


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact nearest-neighbor retrieval for repository vectors stored in Qdrant."
    )
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--qdrant-api-key", default=QDRANT_API_KEY, help="Qdrant API key")
    parser.add_argument("--collection", default=QDRANT_COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--vector-name", default=QDRANT_VECTOR_NAME, help="Named vector field")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Neighbors to keep after self-match removal")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Qdrant scroll batch size")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum repositories to evaluate")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip exact=True versus exact=False comparison")
    parser.add_argument("--report", default="nearest_neighbor_diagnostic_report.md", help="Markdown report output path")
    parser.add_argument("--no-report", action="store_true", help="Print only; do not write a Markdown report")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    _setup_logging(args.log_level)

    config = RepositoryEmbeddingConfig()
    store = QdrantRepositoryStore(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        collection_name=args.collection,
        vector_name=args.vector_name,
        vector_size=config.embedding_dim,
    )

    collection_report = inspect_collection(store, config)
    points = load_points_in_batches(store, batch_size=args.batch_size, limit=args.limit or None)
    points = [point for point in points if point.get("vector")]
    if len(points) < 2:
        raise SystemExit("Need at least two stored repository vectors to run nearest-neighbor evaluation.")

    query_reports, records = evaluate_neighbors(store, points, top_k=args.top_k, exact=True)
    metrics = compute_metrics(points, records)
    comparison = None
    if not args.skip_comparison:
        current_reports, _ = evaluate_neighbors(store, points, top_k=args.top_k, exact=False)
        comparison = compare_reports(query_reports, current_reports)
    diagnostics = compute_corpus_diagnostics(points)
    root_cause = analyze_database_retrieval(diagnostics, metrics)

    report = format_report(
        collection_report=collection_report,
        query_reports=query_reports,
        metrics=metrics,
        comparison=comparison,
        diagnostics=diagnostics,
        root_cause=root_cause,
        top_k=args.top_k,
    )
    print(report)

    if not args.no_report:
        with open(args.report, "w", encoding="utf-8") as file:
            file.write(report)
            file.write("\n")
        print(f"\nWrote diagnostic report: {args.report}")


def inspect_collection(store: QdrantRepositoryStore, config: RepositoryEmbeddingConfig) -> dict[str, Any]:
    """Read live Qdrant collection settings without mutating the collection."""
    info = store.client.get_collection(store.collection_name)
    vectors = info.config.params.vectors
    vector_config = vectors.get(store.vector_name) if isinstance(vectors, dict) else vectors
    if vector_config is None:
        raise ValueError(f"Collection does not contain vector field {store.vector_name!r}.")

    return {
        "collection": store.collection_name,
        "vector_name": store.vector_name,
        "configured_dim": config.embedding_dim,
        "qdrant_dim": int(vector_config.size),
        "configured_distance": QDRANT_DISTANCE,
        "qdrant_distance": str(vector_config.distance.value),
        "points_count": int(getattr(info, "points_count", 0) or 0),
        "indexed_vectors_count": int(getattr(info, "indexed_vectors_count", 0) or 0),
        "hnsw_config": str(getattr(info.config, "hnsw_config", "")),
        "previous_wrapper_search": "Qdrant query_points without exact=True; normal Qdrant indexed search behavior.",
        "active_wrapper_search": "Qdrant query_points with SearchParams(exact=True).",
    }


def load_points_in_batches(
    store: QdrantRepositoryStore,
    *,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Scroll repository points from Qdrant in fixed-size batches."""
    points: list[dict[str, Any]] = []
    offset = None
    while True:
        remaining = None if limit is None else limit - len(points)
        if remaining is not None and remaining <= 0:
            break
        current_limit = batch_size if remaining is None else min(batch_size, remaining)
        records, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=current_limit,
            offset=offset,
            with_payload=True,
            with_vectors=[store.vector_name],
        )
        if not records:
            break
        for record in records:
            payload = record.payload or {}
            points.append(
                {
                    "id": str(record.id),
                    "repo_id": payload.get("repo_id"),
                    "payload": payload,
                    "vector": store._extract_vector(record.vector),
                }
            )
        if offset is None:
            break
    return points


def evaluate_neighbors(
    store: QdrantRepositoryStore,
    points: list[dict[str, Any]],
    *,
    top_k: int,
    exact: bool,
) -> tuple[list[dict[str, Any]], list[NeighborRecord]]:
    reports: list[dict[str, Any]] = []
    records: list[NeighborRecord] = []

    for point in points:
        started = perf_counter()
        matches = store.search(point["vector"], limit=top_k + 1, exact=exact)
        latency_ms = (perf_counter() - started) * 1000
        query_repo = str(point.get("repo_id") or point.get("id"))
        query_category = category(point)
        neighbors = []
        for match in matches:
            payload = match.get("payload") or {}
            neighbor_repo = str(match.get("repo_id") or match.get("id"))
            if neighbor_repo == query_repo or str(match.get("id")) == str(point.get("id")):
                continue
            neighbor_category = payload_category(payload)
            neighbor = {
                "repo_id": neighbor_repo,
                "category": neighbor_category,
                "score": float(match["score"]),
            }
            neighbors.append(neighbor)
            records.append(
                NeighborRecord(
                    query_repo=query_repo,
                    query_category=query_category,
                    neighbor_repo=neighbor_repo,
                    neighbor_category=neighbor_category,
                    score=float(match["score"]),
                )
            )
            if len(neighbors) >= top_k:
                break
        reports.append(
            {
                "repo_id": query_repo,
                "category": query_category,
                "neighbors": neighbors,
                "exact": exact,
                "latency_ms": latency_ms,
            }
        )
    return reports, records


def compare_reports(exact_reports: list[dict[str, Any]], current_reports: list[dict[str, Any]]) -> dict[str, float]:
    overlaps = []
    same_rank = []
    latency_deltas = []
    for exact_report, current_report in zip(exact_reports, current_reports):
        exact_ids = [item["repo_id"] for item in exact_report["neighbors"]]
        current_ids = [item["repo_id"] for item in current_report["neighbors"]]
        overlaps.append(percent(len(set(exact_ids) & set(current_ids)), max(len(exact_ids), 1)))
        rank_matches = sum(1 for left, right in zip(exact_ids, current_ids) if left == right)
        same_rank.append(percent(rank_matches, max(len(exact_ids), 1)))
        latency_deltas.append(float(exact_report["latency_ms"]) - float(current_report["latency_ms"]))
    return {
        "average_top_k_overlap_pct": safe_mean(overlaps),
        "average_same_rank_pct": safe_mean(same_rank),
        "average_exact_latency_ms": safe_mean([float(item["latency_ms"]) for item in exact_reports]),
        "median_exact_latency_ms": safe_median([float(item["latency_ms"]) for item in exact_reports]),
        "average_current_latency_ms": safe_mean([float(item["latency_ms"]) for item in current_reports]),
        "median_current_latency_ms": safe_median([float(item["latency_ms"]) for item in current_reports]),
        "average_latency_delta_ms": safe_mean(latency_deltas),
    }


def compute_metrics(points: list[dict[str, Any]], records: list[NeighborRecord]) -> dict[str, Any]:
    scores = [record.score for record in records]
    same = [record for record in records if record.query_category == record.neighbor_category]
    cross = [record for record in records if record.query_category != record.neighbor_category]
    most_similar = sorted(records, key=lambda item: item.score, reverse=True)[:10]
    least_similar = sorted(records, key=lambda item: item.score)[:10]

    category_stats: dict[str, dict[str, Any]] = {}
    by_category: dict[str, list[NeighborRecord]] = defaultdict(list)
    for record in records:
        by_category[record.query_category].append(record)
    for cat, cat_records in sorted(by_category.items()):
        same_count = sum(1 for item in cat_records if item.query_category == item.neighbor_category)
        cat_scores = [item.score for item in cat_records]
        category_stats[cat] = {
            "neighbors": len(cat_records),
            "same_category_pct": percent(same_count, len(cat_records)),
            "avg_similarity": safe_mean(cat_scores),
            "median_similarity": safe_median(cat_scores),
        }

    vector_dims = Counter(len(point["vector"]) for point in points if point.get("vector"))
    return {
        "evaluated_repositories": len(points),
        "neighbor_edges": len(records),
        "average_similarity": safe_mean(scores),
        "median_similarity": safe_median(scores),
        "same_category_retrieval_pct": percent(len(same), len(records)),
        "cross_category_retrieval_pct": percent(len(cross), len(records)),
        "same_category_average_similarity": safe_mean([item.score for item in same]),
        "cross_category_average_similarity": safe_mean([item.score for item in cross]),
        "vector_dimensions_seen": dict(sorted(vector_dims.items())),
        "most_similar_pairs": most_similar,
        "least_similar_pairs": least_similar,
        "category_clustering_statistics": category_stats,
    }


def compute_corpus_diagnostics(points: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(category(point) for point in points)
    group_counts = {
        group: sum(1 for point in points if matches_category_group(point, terms))
        for group, terms in INSPECT_CATEGORY_GROUPS.items()
    }
    readme_lengths = [int((point.get("payload") or {}).get("readme_length") or 0) for point in points]
    doc_scores = [float((point.get("payload") or {}).get("doc_quality") or 0.0) for point in points]
    topic_counts = [len((point.get("payload") or {}).get("topics") or []) for point in points]

    return {
        "total_repositories_indexed": len(points),
        "repository_count_per_category": dict(sorted(category_counts.items())),
        "top_20_categories_by_count": category_counts.most_common(20),
        "specified_category_group_counts": group_counts,
        "avg_readme_length": safe_mean(readme_lengths),
        "median_readme_length": safe_median(readme_lengths),
        "avg_doc_quality": safe_mean(doc_scores),
        "median_doc_quality": safe_median(doc_scores),
        "avg_topic_count": safe_mean(topic_counts),
        "median_topic_count": safe_median(topic_counts),
    }


def analyze_database_retrieval(diagnostics: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    causes: list[str] = []
    total = int(diagnostics["total_repositories_indexed"])
    database_count = int(diagnostics["specified_category_group_counts"].get("Databases", 0))
    database_share = database_count / total if total else 0.0
    same_pct = float(metrics["same_category_retrieval_pct"])
    topic_count = float(diagnostics["avg_topic_count"])
    readme_length = float(diagnostics["median_readme_length"])

    if database_share < 0.08:
        causes.append("A. Corpus imbalance: database-related repositories are underrepresented.")
    if same_pct < 35.0:
        causes.append("B/E. Category separation is weak, so metadata/category distribution is likely contributing.")
    else:
        causes.append("B. Metadata weighting is not proven as the primary issue from ENN alone.")
    if topic_count < 3.0:
        causes.append("C. Topic extraction quality may be weak because indexed repositories carry few topics.")
    else:
        causes.append("C. Topic coverage appears usable on average; inspect database-specific topics next.")
    if readme_length < 500:
        causes.append("D. README quality may be weak because median README length is low.")
    else:
        causes.append("D. README length is not obviously low; content relevance still needs qualitative review.")
    if database_count and database_share >= 0.08:
        causes.append("E. Database category distribution exists, so failures may be semantic overlap with backend/devops.")
    return causes


def format_report(
    *,
    collection_report: dict[str, Any],
    query_reports: list[dict[str, Any]],
    metrics: dict[str, Any],
    comparison: dict[str, float] | None,
    diagnostics: dict[str, Any],
    root_cause: list[str],
    top_k: int,
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# Exact Nearest Neighbor Repository Evaluation",
            "",
            "## Architecture analysis",
            "- Approved repositories are embedded through README, metadata, and topic towers.",
            "- Repo Tower weights remain README=0.60, metadata=0.25, topics=0.15.",
            "- Final repository vectors are stored as the named Qdrant vector field.",
            "- This script reads already-indexed vectors only; it does not re-embed or mutate repositories.",
            "",
            "## Qdrant analysis",
            f"- Collection: {collection_report['collection']}",
            f"- Vector field: {collection_report['vector_name']}",
            f"- Configured vector dimension: {collection_report['configured_dim']}",
            f"- Qdrant vector dimension: {collection_report['qdrant_dim']}",
            f"- Configured distance: {collection_report['configured_distance']}",
            f"- Qdrant distance: {collection_report['qdrant_distance']}",
            f"- Points count: {collection_report['points_count']}",
            f"- Indexed vectors count: {collection_report['indexed_vectors_count']}",
            f"- HNSW config: {collection_report['hnsw_config']}",
            f"- Previous retrieval implementation: {collection_report['previous_wrapper_search']}",
            f"- Active retrieval implementation: {collection_report['active_wrapper_search']}",
            "",
            "## ENN evaluation results",
            f"- Evaluated repositories: {metrics['evaluated_repositories']}",
            f"- Neighbor edges: {metrics['neighbor_edges']}",
            f"- Top K per repository: {top_k}",
            f"- Average similarity score: {metrics['average_similarity']:.4f}",
            f"- Median similarity score: {metrics['median_similarity']:.4f}",
            f"- Same-category retrieval percentage: {metrics['same_category_retrieval_pct']:.2f}%",
            f"- Cross-category retrieval percentage: {metrics['cross_category_retrieval_pct']:.2f}%",
            f"- Same-category average similarity: {metrics['same_category_average_similarity']:.4f}",
            f"- Cross-category average similarity: {metrics['cross_category_average_similarity']:.4f}",
            f"- Vector dimensions seen: {metrics['vector_dimensions_seen']}",
            "",
            "### ENN vs previous retrieval comparison",
        ]
    )
    if comparison:
        lines.extend(
            [
                f"- Average top-k overlap: {comparison['average_top_k_overlap_pct']:.2f}%",
                f"- Average same-rank match: {comparison['average_same_rank_pct']:.2f}%",
                f"- Average ENN latency: {comparison['average_exact_latency_ms']:.2f} ms",
                f"- Median ENN latency: {comparison['median_exact_latency_ms']:.2f} ms",
                f"- Average previous-search latency: {comparison['average_current_latency_ms']:.2f} ms",
                f"- Median previous-search latency: {comparison['median_current_latency_ms']:.2f} ms",
                f"- Average ENN latency delta: {comparison['average_latency_delta_ms']:.2f} ms",
            ]
        )
    else:
        lines.append("- Comparison skipped.")
    lines.extend(
        [
            "",
            "### Query repository nearest neighbors",
        ]
    )
    for report in query_reports:
        lines.append(f"- Query repository: {report['repo_id']} | category: {report['category']}")
        for index, neighbor in enumerate(report["neighbors"], 1):
            lines.append(
                f"  {index}. {neighbor['repo_id']} | score={neighbor['score']:.4f} "
                f"| category={neighbor['category']}"
            )

    lines.extend(["", "### Most similar repository pairs"])
    lines.extend(format_pairs(metrics["most_similar_pairs"]))
    lines.extend(["", "### Least similar repository pairs"])
    lines.extend(format_pairs(metrics["least_similar_pairs"]))

    lines.extend(
        [
            "",
            "### Category clustering statistics",
            "| Category | Neighbor edges | Same-category % | Avg similarity | Median similarity |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for cat, stats in metrics["category_clustering_statistics"].items():
        lines.append(
            f"| {cat} | {stats['neighbors']} | {stats['same_category_pct']:.2f}% | "
            f"{stats['avg_similarity']:.4f} | {stats['median_similarity']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Corpus diagnostics",
            f"- Total repositories indexed: {diagnostics['total_repositories_indexed']}",
            f"- Average README length: {diagnostics['avg_readme_length']:.1f}",
            f"- Median README length: {diagnostics['median_readme_length']:.1f}",
            f"- Average documentation quality: {diagnostics['avg_doc_quality']:.2f}",
            f"- Median documentation quality: {diagnostics['median_doc_quality']:.2f}",
            f"- Average topic count: {diagnostics['avg_topic_count']:.2f}",
            f"- Median topic count: {diagnostics['median_topic_count']:.2f}",
            "",
            "### Repository count per category",
        ]
    )
    for cat, count in diagnostics["repository_count_per_category"].items():
        lines.append(f"- {cat}: {count}")

    lines.extend(["", "### Top 20 categories by count"])
    for cat, count in diagnostics["top_20_categories_by_count"]:
        lines.append(f"- {cat}: {count}")

    lines.extend(["", "### Specified category inspection"])
    for cat, count in diagnostics["specified_category_group_counts"].items():
        lines.append(f"- {cat}: {count}")

    lines.extend(["", "## Root-cause analysis"])
    lines.extend(f"- {item}" for item in root_cause)

    lines.extend(
        [
            "",
            "## Recommended next steps",
            "- Review database-query neighbor examples qualitatively against README text and topics.",
            "- Compare exact ENN results with the existing wrapper search to quantify ANN/index effects.",
            "- Do not change embeddings, tower weights, or schema until the corpus/category diagnostics are reviewed.",
            "",
            "## Run instructions",
            "```powershell",
            "python nearest_neighbor_eval.py",
            "python nearest_neighbor_eval.py --limit 50 --top-k 5",
            "python nearest_neighbor_eval.py --skip-comparison",
            "python nearest_neighbor_eval.py --no-report",
            "```",
        ]
    )
    return "\n".join(lines)


def format_pairs(records: list[NeighborRecord]) -> list[str]:
    if not records:
        return ["- No pairs available."]
    return [
        f"- {record.query_repo} -> {record.neighbor_repo} | score={record.score:.4f} "
        f"| {record.query_category} -> {record.neighbor_category}"
        for record in records
    ]


def matches_category_group(point: dict[str, Any], terms: tuple[str, ...]) -> bool:
    payload = point.get("payload") or {}
    fields = [
        category(point),
        str(payload.get("description") or ""),
        str(payload.get("primary_language") or ""),
        " ".join(str(item) for item in payload.get("topics") or []),
        " ".join(str(item) for item in payload.get("languages") or []),
        " ".join(str(item) for item in payload.get("tags") or []),
    ]
    haystack = " ".join(fields).lower()
    return any(term in haystack for term in terms)


def category(point: dict[str, Any]) -> str:
    return payload_category(point.get("payload") or {})


def payload_category(payload: dict[str, Any]) -> str:
    return str(payload.get("discovery_category") or payload.get("category") or "Unknown")


def safe_mean(values: list[float] | list[int]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: list[float] | list[int]) -> float:
    return float(median(values)) if values else 0.0


def percent(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


if __name__ == "__main__":
    main()
