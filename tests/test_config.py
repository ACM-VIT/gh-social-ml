"""Unit tests for trending/config.py module."""

import os
import pytest

from trending.config import (
    TRENDING_REPO_LIMIT,
    TRENDING_REFRESH_HOURS,
    TRENDING_TABLE_NAME,
    TRENDING_METADATA_TABLE_NAME,
    GITHUB_TIMEOUT_SECONDS,
    GITHUB_MAX_RETRIES,
    README_MAX_LENGTH,
    DATABASE_URL,
    DB_POOL_SIZE,
    LOG_LEVEL,
    LOG_FILE,
    LOG_FORMAT,
    CONTINUE_ON_ERROR,
    MAX_CONSECUTIVE_FAILURES,
    validate_config,
)


@pytest.mark.unit
class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_repo_limit(self):
        """Test default TRENDING_REPO_LIMIT is 30."""
        assert TRENDING_REPO_LIMIT == 30

    def test_default_refresh_hours(self):
        """Test default TRENDING_REFRESH_HOURS is 24."""
        assert TRENDING_REFRESH_HOURS == 24

    def test_default_table_names(self):
        """Test default table names."""
        assert TRENDING_TABLE_NAME == "trending_repositories"
        assert TRENDING_METADATA_TABLE_NAME == "trending_metadata"

    def test_default_github_config(self):
        """Test default GitHub configuration."""
        assert GITHUB_TIMEOUT_SECONDS == 30.0
        assert GITHUB_MAX_RETRIES == 4

    def test_default_data_enrichment_config(self):
        """Test default data enrichment configuration."""
        assert README_MAX_LENGTH == 10000

    def test_default_database_config(self):
        """Test default database configuration."""
        assert DB_POOL_SIZE == 5

    def test_default_logging_config(self):
        """Test default logging configuration."""
        assert LOG_LEVEL == "INFO"
        assert LOG_FILE is None
        assert LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def test_default_error_handling_config(self):
        """Test default error handling configuration."""
        assert CONTINUE_ON_ERROR is True
        assert MAX_CONSECUTIVE_FAILURES == 10


@pytest.mark.unit
class TestConfigEnvironmentVariables:
    """Test configuration loading from environment variables."""

    def test_repo_limit_from_env(self, monkeypatch):
        """Test TRENDING_REPO_LIMIT from environment variable."""
        monkeypatch.setenv("TRENDING_REPO_LIMIT", "50")
        # Re-import to get new value
        import importlib
        import trending.config
        importlib.reload(trending.config)
        trending.config.validate_config()
        assert trending.config.TRENDING_REPO_LIMIT == 50

    def test_refresh_hours_from_env(self, monkeypatch):
        """Test TRENDING_REFRESH_HOURS from environment variable."""
        monkeypatch.setenv("TRENDING_REFRESH_HOURS", "12")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        trending.config.validate_config()
        assert trending.config.TRENDING_REFRESH_HOURS == 12

    def test_database_url_from_env(self, monkeypatch):
        """Test DATABASE_URL from environment variable."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        assert trending.config.DATABASE_URL == "postgresql://test:test@localhost:5432/test"

    def test_boolean_config_from_env(self, monkeypatch):
        """Test boolean configuration from environment variables."""
        monkeypatch.setenv("CONTINUE_ON_ERROR", "false")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        trending.config.validate_config()
        assert trending.config.CONTINUE_ON_ERROR is False


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation function."""

    def test_validate_config_with_valid_config(self, monkeypatch):
        """Test validation with valid configuration."""
        monkeypatch.setenv("BACKEND_URL", "http://backend.test")
        monkeypatch.setenv("INTERNAL_API_SECRET", "test-secret")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        errors = trending.config.validate_config()
        assert errors == []

    def test_validate_config_invalid_repo_limit(self, monkeypatch):
        """Test validation fails when TRENDING_REPO_LIMIT is invalid."""
        monkeypatch.setenv("TRENDING_REPO_LIMIT", "0")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        errors = trending.config.validate_config()
        assert any("TRENDING_REPO_LIMIT" in error for error in errors)

    def test_validate_config_invalid_refresh_hours(self, monkeypatch):
        """Test validation fails when TRENDING_REFRESH_HOURS is invalid."""
        monkeypatch.setenv("TRENDING_REFRESH_HOURS", "-1")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        errors = trending.config.validate_config()
        assert any("TRENDING_REFRESH_HOURS" in error for error in errors)

    def test_validate_config_multiple_errors(self, monkeypatch):
        """Test validation returns multiple errors."""
        monkeypatch.setenv("TRENDING_REPO_LIMIT", "-5")
        monkeypatch.setenv("TRENDING_REFRESH_HOURS", "-1")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        errors = trending.config.validate_config()
        assert len(errors) >= 2


@pytest.mark.unit
class TestConfigTypeConversion:
    """Test type conversion in configuration loading."""

    def test_repo_limit_type_conversion(self, monkeypatch):
        """Test TRENDING_REPO_LIMIT converts to int."""
        monkeypatch.setenv("TRENDING_REPO_LIMIT", "100")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        trending.config.validate_config()
        assert isinstance(trending.config.TRENDING_REPO_LIMIT, int)
        assert trending.config.TRENDING_REPO_LIMIT == 100

    def test_timeout_seconds_type_conversion(self, monkeypatch):
        """Test GITHUB_TIMEOUT_SECONDS converts to float."""
        monkeypatch.setenv("GITHUB_TIMEOUT_SECONDS", "60.5")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        trending.config.validate_config()
        assert isinstance(trending.config.GITHUB_TIMEOUT_SECONDS, float)
        assert trending.config.GITHUB_TIMEOUT_SECONDS == 60.5

    def test_max_retries_type_conversion(self, monkeypatch):
        """Test GITHUB_MAX_RETRIES converts to int."""
        monkeypatch.setenv("GITHUB_MAX_RETRIES", "10")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        trending.config.validate_config()
        assert isinstance(trending.config.GITHUB_MAX_RETRIES, int)
        assert trending.config.GITHUB_MAX_RETRIES == 10

    def test_readme_max_length_type_conversion(self, monkeypatch):
        """Test README_MAX_LENGTH converts to int."""
        monkeypatch.setenv("README_MAX_LENGTH", "50000")
        import importlib
        import trending.config
        importlib.reload(trending.config)
        trending.config.validate_config()
        assert isinstance(trending.config.README_MAX_LENGTH, int)
        assert trending.config.README_MAX_LENGTH == 50000
