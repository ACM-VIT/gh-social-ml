"""Trending-specific configuration for the GitHub Trending ingestion engine.

This module contains all configuration parameters for the trending repository
ingestion service, including database settings, API limits, and scheduling parameters.
"""

import os
import re


# ── Core Configuration ─────────────────────────────────────────────────────────────

# Number of trending repositories to fetch per refresh cycle
TRENDING_REPO_LIMIT_STR: str = os.getenv("TRENDING_REPO_LIMIT", "30")
TRENDING_REPO_LIMIT: int = 30  # Default, will be validated and cast in validate_config()

# Refresh interval in hours (default: 24 hours)
TRENDING_REFRESH_HOURS_STR: str = os.getenv("TRENDING_REFRESH_HOURS", "24")
TRENDING_REFRESH_HOURS: int = 24  # Default, will be validated and cast in validate_config()

# PostgreSQL table names for trending repositories
TRENDING_TABLE_NAME: str = os.getenv("TRENDING_TABLE_NAME", "trending_repositories")
TRENDING_METADATA_TABLE_NAME: str = os.getenv("TRENDING_METADATA_TABLE_NAME", "trending_metadata")

# Payload synchronization is opt-in so the Postgres trending refresh can still
# run in deployments that do not provide Qdrant to this offline service.
TRENDING_QDRANT_SYNC_STR: str = os.getenv("TRENDING_QDRANT_SYNC_ENABLED", "false")
TRENDING_QDRANT_SYNC_ENABLED: bool = False


# ── GitHub API Configuration ───────────────────────────────────────────────────────

# Request timeout in seconds (for HTTP requests to GitHub Trending page)
GITHUB_TIMEOUT_SECONDS_STR: str = os.getenv("GITHUB_TIMEOUT_SECONDS", "30.0")
GITHUB_TIMEOUT_SECONDS: float = 30.0  # Default, will be validated and cast in validate_config()

# Maximum retry attempts for failed requests
GITHUB_MAX_RETRIES_STR: str = os.getenv("GITHUB_MAX_RETRIES", "4")
GITHUB_MAX_RETRIES: int = 4  # Default, will be validated and cast in validate_config()


# ── Data Enrichment Configuration ───────────────────────────────────────────────────

# Maximum README length to store (in characters)
README_MAX_LENGTH_STR: str = os.getenv("README_MAX_LENGTH", "10000")
README_MAX_LENGTH: int = 10000  # Default, will be validated and cast in validate_config()


# ── Database Configuration ─────────────────────────────────────────────────────────

# PostgreSQL database URL (shared with main application)
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
BACKEND_URL: str | None = os.getenv("BACKEND_URL")
INTERNAL_API_SECRET: str | None = os.getenv("INTERNAL_API_SECRET")

# Connection pool size for database operations
DB_POOL_SIZE_STR: str = os.getenv("DB_POOL_SIZE", "5")
DB_POOL_SIZE: int = 5  # Default, will be validated and cast in validate_config()


# ── Logging Configuration ───────────────────────────────────────────────────────────

# Log level for trending service
LOG_LEVEL: str = os.getenv("TRENDING_LOG_LEVEL", "INFO")

# Log file path (if not set, logs to stdout)
LOG_FILE: str | None = os.getenv("TRENDING_LOG_FILE")

# Log format
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ── Error Handling Configuration ───────────────────────────────────────────────────

# Whether to continue processing on individual repository failures
CONTINUE_ON_ERROR: bool = os.getenv("CONTINUE_ON_ERROR", "true").lower() == "true"

# Maximum number of consecutive failures before stopping the pipeline
MAX_CONSECUTIVE_FAILURES_STR: str = os.getenv("MAX_CONSECUTIVE_FAILURES", "10")
MAX_CONSECUTIVE_FAILURES: int = 10  # Default, will be validated and cast in validate_config()


# ── Validation ─────────────────────────────────────────────────────────────────────

def validate_config() -> list[str]:
    """Validate trending configuration and return list of errors.

    This function also performs type casting from string env vars to their proper types.

    Returns:
        List of error messages. Empty list if configuration is valid.
    """
    global TRENDING_REPO_LIMIT, TRENDING_REFRESH_HOURS, GITHUB_TIMEOUT_SECONDS
    global GITHUB_MAX_RETRIES, README_MAX_LENGTH, DB_POOL_SIZE, MAX_CONSECUTIVE_FAILURES
    global TRENDING_QDRANT_SYNC_ENABLED
    
    errors: list[str] = []

    normalized_sync = str(TRENDING_QDRANT_SYNC_STR).strip().lower()
    if normalized_sync not in {"true", "false"}:
        errors.append(
            "TRENDING_QDRANT_SYNC_ENABLED must be 'true' or 'false', "
            f"got {TRENDING_QDRANT_SYNC_STR!r}"
        )
    else:
        TRENDING_QDRANT_SYNC_ENABLED = normalized_sync == "true"

    # Cast and validate TRENDING_REPO_LIMIT
    try:
        TRENDING_REPO_LIMIT = int(TRENDING_REPO_LIMIT_STR)
        if TRENDING_REPO_LIMIT <= 0:
            errors.append(f"TRENDING_REPO_LIMIT must be positive, got {TRENDING_REPO_LIMIT}")
    except (ValueError, TypeError):
        errors.append(f"TRENDING_REPO_LIMIT must be a valid integer, got '{TRENDING_REPO_LIMIT_STR}'")

    # Cast and validate TRENDING_REFRESH_HOURS
    try:
        TRENDING_REFRESH_HOURS = int(TRENDING_REFRESH_HOURS_STR)
        if TRENDING_REFRESH_HOURS <= 0:
            errors.append(f"TRENDING_REFRESH_HOURS must be positive, got {TRENDING_REFRESH_HOURS}")
    except (ValueError, TypeError):
        errors.append(f"TRENDING_REFRESH_HOURS must be a valid integer, got '{TRENDING_REFRESH_HOURS_STR}'")

    # Cast and validate GITHUB_TIMEOUT_SECONDS
    try:
        GITHUB_TIMEOUT_SECONDS = float(GITHUB_TIMEOUT_SECONDS_STR)
        if GITHUB_TIMEOUT_SECONDS <= 0:
            errors.append(f"GITHUB_TIMEOUT_SECONDS must be positive, got {GITHUB_TIMEOUT_SECONDS}")
    except (ValueError, TypeError):
        errors.append(f"GITHUB_TIMEOUT_SECONDS must be a valid float, got '{GITHUB_TIMEOUT_SECONDS_STR}'")

    # Cast and validate GITHUB_MAX_RETRIES
    try:
        GITHUB_MAX_RETRIES = int(GITHUB_MAX_RETRIES_STR)
        if GITHUB_MAX_RETRIES < 1:
            errors.append(f"GITHUB_MAX_RETRIES must be at least 1, got {GITHUB_MAX_RETRIES}")
    except (ValueError, TypeError):
        errors.append(f"GITHUB_MAX_RETRIES must be a valid integer, got '{GITHUB_MAX_RETRIES_STR}'")

    # Cast and validate README_MAX_LENGTH
    try:
        README_MAX_LENGTH = int(README_MAX_LENGTH_STR)
        if README_MAX_LENGTH <= 0:
            errors.append(f"README_MAX_LENGTH must be positive, got {README_MAX_LENGTH}")
    except (ValueError, TypeError):
        errors.append(f"README_MAX_LENGTH must be a valid integer, got '{README_MAX_LENGTH_STR}'")

    # Cast and validate DB_POOL_SIZE
    try:
        DB_POOL_SIZE = int(DB_POOL_SIZE_STR)
        if DB_POOL_SIZE <= 0:
            errors.append(f"DB_POOL_SIZE must be positive, got {DB_POOL_SIZE}")
    except (ValueError, TypeError):
        errors.append(f"DB_POOL_SIZE must be a valid integer, got '{DB_POOL_SIZE_STR}'")

    # Cast and validate MAX_CONSECUTIVE_FAILURES
    try:
        MAX_CONSECUTIVE_FAILURES = int(MAX_CONSECUTIVE_FAILURES_STR)
        if MAX_CONSECUTIVE_FAILURES <= 0:
            errors.append(f"MAX_CONSECUTIVE_FAILURES must be positive, got {MAX_CONSECUTIVE_FAILURES}")
    except (ValueError, TypeError):
        errors.append(f"MAX_CONSECUTIVE_FAILURES must be a valid integer, got '{MAX_CONSECUTIVE_FAILURES_STR}'")

    # Validate table names to prevent SQL injection
    table_name_pattern = r'^[a-z_][a-z0-9_]*$'
    if not re.match(table_name_pattern, TRENDING_TABLE_NAME):
        errors.append(f"TRENDING_TABLE_NAME must match pattern {table_name_pattern}, got '{TRENDING_TABLE_NAME}'")
    if not re.match(table_name_pattern, TRENDING_METADATA_TABLE_NAME):
        errors.append(f"TRENDING_METADATA_TABLE_NAME must match pattern {table_name_pattern}, got '{TRENDING_METADATA_TABLE_NAME}'")

    return errors
