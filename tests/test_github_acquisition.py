from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Any

import ingestion_engine
from acquisition.github_discovery import DiscoveryConfig, GitHubDiscoveryEngine
from utils.readme_processor import process_markdown, process_readme_payload
from acquisition.repository_enricher import RepositoryEnricher


class FakeGitHubClient:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str]] = []

    def search_repositories(self, query: str, *, sort: str, order: str, per_page: int, max_pages: int) -> list[dict[str, Any]]:
        self.search_calls.append((query, sort))
        band = "recent" if "pushed:" in query else "emerging" if "created:" in query else "mid" if "50..500" in query else "high"
        return [
            {"full_name": f"owner/{band}-{len(self.search_calls)}-{idx}", "stargazers_count": 100 + idx}
            for idx in range(2)
        ]


def test_readme_processor_decodes_and_extracts_meaningful_paragraphs() -> None:
    markdown = """
# Project
![badge](https://img.shields.io/badge/build-passing.svg)

This repository provides a robust automation framework for running developer workflows across cloud systems and local machines.

```python
print("not semantic docs")
```

![screenshot](docs/screenshot.png)

## Install
pip install example

The architecture includes a scheduler, plugin runtime, event router, and integration adapters for production operations.
"""
    payload = {"encoding": "base64", "content": base64.b64encode(markdown.encode()).decode()}

    document = process_readme_payload(payload)

    assert document.readme_length == len(markdown)
    assert len(document.extracted_paragraphs) == 2
    assert "badge" not in document.clean_text.lower()
    assert "screenshot" not in document.clean_text.lower()




def test_ingestion_engine_uses_internal_neighbors_when_qdrant_unavailable() -> None:
    original_qdrant_ok = ingestion_engine._QDRANT_OK
    ingestion_engine._QDRANT_OK = False
    store = ingestion_engine.CorpusStore()
    try:
        # Passes pre-filters and blended score (0.6206 >= 0.60)
        first = ingestion_engine.ingest_repository(_osiris_repo("owner/seed"), corpus_store=store, auto_index=True)
        # Low star repo - rejected by pre-filter
        low_stars = _osiris_repo("owner/low-stars")
        low_stars["star_count"] = 10
        second = ingestion_engine.ingest_repository(low_stars, corpus_store=store, auto_index=True)
        # Low readme repo - rejected by pre-filter
        low_readme = _osiris_repo("owner/low-readme")
        low_readme["readme_length"] = 50
        third = ingestion_engine.ingest_repository(low_readme, corpus_store=store, auto_index=True)
    finally:
        ingestion_engine._QDRANT_OK = original_qdrant_ok

    assert first.decision == "APPROVED"
    assert first.novelty.final == 1.0
    assert second.decision == "REJECTED"
    assert "star count (10) is below minimum threshold" in second.rejection_reason
    assert third.decision == "REJECTED"
    assert "README length (50 chars) is below minimum threshold" in third.rejection_reason



class EnricherFakeClient:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def get_repository(self, full_name: str) -> dict[str, Any]:
        return {
            "full_name": full_name,
            "stargazers_count": 144,
            "language": "Python",
            "size": 100,
            "forks_count": 12,
            "open_issues_count": 4,
            "description": "A useful AI repository",
            "html_url": "https://github.com/owner/repo",
            "created_at": (self.now - timedelta(days=60)).isoformat(),
            "updated_at": (self.now - timedelta(days=1)).isoformat(),
            "pushed_at": (self.now - timedelta(days=2)).isoformat(),
            "topics": ["ai", "rag"],
            "owner": {"login": "owner"},
        }

    def get_readme(self, full_name: str) -> dict[str, Any]:
        markdown = "This repository implements retrieval augmented generation workflows with ingestion, indexing, and evaluation components for production systems."
        return {"encoding": "base64", "content": base64.b64encode(markdown.encode()).decode()}

    def get_topics(self, full_name: str) -> list[str]:
        return ["ai", "rag"]

    def get_languages(self, full_name: str) -> dict[str, int]:
        return {"Python": 1000, "Shell": 100}

    def get_contributors(self, full_name: str, *, max_pages: int = 1) -> list[dict[str, Any]]:
        return [{"login": "a"}, {"login": "b"}, {"login": "c"}]

    def get_events(self, full_name: str) -> list[dict[str, Any]]:
        return []

    def get_stargazers(self, full_name: str, *, max_pages: int = 4) -> list[dict[str, Any]]:
        return [
            {"starred_at": (self.now - timedelta(days=1)).isoformat()},
            {"starred_at": (self.now - timedelta(days=5)).isoformat()},
            {"starred_at": (self.now - timedelta(days=20)).isoformat()},
        ]


def _osiris_repo(repo_id: str) -> dict[str, Any]:
    return {
        "id": repo_id,
        "star_count": 100,
        "pushed_days_ago": 1,
        "mentionable_users_count": 5,
        "primary_language": "Python",
        "readme_length": 2000,
        "readme_to_codebase_ratio": 0.05,
        "extracted_paragraphs": [
            "This Python AI agent framework provides retrieval augmented generation workflows, tool calling, and production automation for developer systems."
        ],
        "delta_3d": 3,
        "delta_7d": 7,
        "delta_30d": 20,
    }
