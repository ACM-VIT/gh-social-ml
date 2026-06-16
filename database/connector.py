"""PostgreSQL database connector for repository ingestion and updates."""

from __future__ import annotations

import json
import logging
import os
from urllib.parse import urlparse
from typing import Any

import pg8000.dbapi

logger = logging.getLogger("pipeline.database")


class PostgreSQLConnector:
    """Connector for standard PostgreSQL / Supabase databases."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.enabled = bool(self.database_url)

        if not self.enabled:
            logger.warning("DATABASE_URL is not set. Database integration will be disabled.")
            return

        try:
            self.conn_params = self._parse_url(self.database_url)
        except Exception as exc:
            logger.error(f"Failed to parse DATABASE_URL: {exc}. Database integration disabled.")
            self.enabled = False

    def _parse_url(self, url: str) -> dict[str, Any]:
        """Parse PostgreSQL connection URL into pg8000 parameters."""
        # Handles postgresql:// and postgres:// formats
        result = urlparse(url)
        username = result.username
        password = result.password
        database = result.path[1:] if result.path else ""
        hostname = result.hostname
        port = result.port or 5432

        return {
            "user": username,
            "password": password,
            "host": hostname,
            "port": int(port),
            "database": database,
        }

    def connect(self) -> pg8000.dbapi.Connection:
        """Establish a connection to the PostgreSQL database."""
        if not self.enabled:
            raise RuntimeError("Database connector is not enabled (missing or invalid DATABASE_URL).")
        return pg8000.dbapi.connect(**self.conn_params)

    def init_db(self) -> None:
        """Initialize pgcrypto extension and the Repo table if they do not exist."""
        if not self.enabled:
            return

        logger.info("Initializing PostgreSQL database schemas...")
        conn = None
        try:
            conn = self.connect()
            cursor = conn.cursor()

            # Enable pgcrypto for UUID gen_random_uuid() if it isn't already enabled
            try:
                cursor.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
            except Exception as exc:
                # Sometimes user might not have superuser privileges on Supabase,
                # but pgcrypto is usually enabled by default on new projects.
                logger.warning(f"Could not run CREATE EXTENSION pgcrypto: {exc}. Proceeding anyway...")

            # Create table if missing (corresponds to backend team's schema)
            create_table_query = """
            CREATE TABLE IF NOT EXISTS Repo (
                repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                github_repo_url VARCHAR(200) NOT NULL UNIQUE,
                owner_id VARCHAR(100) NOT NULL,
                repo_name VARCHAR(200) NOT NULL,
                full_name VARCHAR(255) NOT NULL,
                description TEXT,
                language_used JSONB DEFAULT '[]'::jsonb,
                topics JSONB DEFAULT '[]'::jsonb,
                readme_summary TEXT,
                likes_count INT DEFAULT 0,
                comments_count INT DEFAULT 0,
                saves_count INT DEFAULT 0,
                views_count INT DEFAULT 0,
                forks_count INT DEFAULT 0,
                pr_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor.execute(create_table_query)
            conn.commit()
            logger.info("Database schemas verified successfully.")
        except Exception as exc:
            logger.error(f"Database initialization failed: {exc}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def upsert_repositories(self, results: list[Any]) -> int:
        """
        Upsert a list of EnrichmentResult objects into the Repo table.
        
        Returns the number of successfully upserted repositories.
        """
        if not self.enabled:
            logger.warning("Database integration disabled; skipping upsert.")
            return 0

        if not results:
            logger.info("No repositories to save.")
            return 0

        logger.info(f"Upserting {len(results)} repositories into PostgreSQL...")
        conn = None
        upserted_count = 0
        try:
            conn = self.connect()
            cursor = conn.cursor()

            upsert_query = """
            INSERT INTO Repo (
                github_repo_url, owner_id, repo_name, full_name, description,
                language_used, topics, readme_summary, forks_count, pr_count
            ) VALUES (
                %s, %s, %s, %s, %s, CAST(%s AS jsonb), CAST(%s AS jsonb), %s, %s, %s
            )
            ON CONFLICT (github_repo_url) DO UPDATE SET
                owner_id = EXCLUDED.owner_id,
                repo_name = EXCLUDED.repo_name,
                full_name = EXCLUDED.full_name,
                description = EXCLUDED.description,
                language_used = EXCLUDED.language_used,
                topics = EXCLUDED.topics,
                readme_summary = EXCLUDED.readme_summary,
                forks_count = EXCLUDED.forks_count,
                pr_count = EXCLUDED.pr_count,
                updated_at = CURRENT_TIMESTAMP;
            """

            for r in results:
                p = r.payload
                raw = r.raw_repository

                # Mapping fields
                github_repo_url = p.get("html_url") or f"https://github.com/{r.repo_id}"
                owner_id = (raw.get("owner") or {}).get("login") or r.repo_id.partition("/")[0]
                repo_name = raw.get("name") or r.repo_id.partition("/")[2]
                full_name = r.repo_id
                description = p.get("description") or ""

                # Convert languages dict and topics list to JSON string
                languages_json = json.dumps(r.languages or {})
                topics_json = json.dumps(r.topics or [])

                # Limit readme clean text to first 5000 characters for readme_summary
                readme_text = getattr(r.readme, "clean_text", "") or ""
                readme_summary = readme_text[:5000]

                forks_count = int(p.get("fork_count") or 0)
                pr_count = int(raw.get("pull_requests_count") or 0)

                params = (
                    github_repo_url,
                    owner_id,
                    repo_name,
                    full_name,
                    description,
                    languages_json,
                    topics_json,
                    readme_summary,
                    forks_count,
                    pr_count,
                )

                try:
                    cursor.execute(upsert_query, params)
                    upserted_count += 1
                except Exception as row_exc:
                    logger.error(f"Failed to upsert repo {full_name}: {row_exc}")
                    # Continue attempting remaining rows in batch

            conn.commit()
            logger.info(f"Database upsert complete. {upserted_count}/{len(results)} rows successfully upserted.")
        except Exception as exc:
            logger.error(f"Database transaction failed: {exc}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

        return upserted_count
