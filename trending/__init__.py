"""GitHub Trending Repository Ingestion Engine.

This module provides a high-performance ingestion service that fetches the top 30
repositories from GitHub's Trending page, bypasses standard quality filters, and
refreshes this list every 24 hours.

Architecture:
- fetcher: Fetches trending repositories from GitHub Trending page via HTML parsing
- scheduler: Manages 24-hour refresh scheduling
- backend_storage: Publishes atomic snapshots to the backend; the backend outbox
  and ML refresh pipeline own Qdrant propagation
- storage: Local PostgreSQL test adapter; production uses backend_storage
- config: Trending-specific configuration
- logger: Centralized logging setup
"""

from .fetcher import TrendingFetcher
from .backend_storage import BackendTrendingStorage
from .scheduler import TrendingScheduler
from . import config

__all__ = [
    "TrendingFetcher",
    "BackendTrendingStorage",
    "TrendingScheduler",
    "config",
]
