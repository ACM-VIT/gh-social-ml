"""Unit tests for trending/scheduler.py module."""

from unittest.mock import MagicMock, patch
import pytest

from trending.scheduler import TrendingScheduler, run_scheduler, run_once


@pytest.mark.unit
class TestTrendingSchedulerInit:
    """Test TrendingScheduler initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default fetcher and storage."""
        scheduler = TrendingScheduler()
        assert scheduler.fetcher is not None
        assert scheduler.storage is not None
        assert scheduler.running is False

    def test_init_with_custom_fetcher(self):
        """Test initialization with custom fetcher."""
        from trending.fetcher import TrendingFetcher
        custom_fetcher = TrendingFetcher()
        scheduler = TrendingScheduler(fetcher=custom_fetcher)
        assert scheduler.fetcher is custom_fetcher

    def test_init_with_custom_storage(self, mock_database_connector):
        """Test initialization with custom storage."""
        from trending.storage import TrendingStorage
        custom_storage = TrendingStorage()
        custom_storage.connector = mock_database_connector
        scheduler = TrendingScheduler(storage=custom_storage)
        assert scheduler.storage is custom_storage

    def test_init_without_schedule_library(self, monkeypatch):
        """Test initialization fails when schedule library is not installed."""
        monkeypatch.setattr("trending.scheduler.HAS_SCHEDULE", False)
        
        with pytest.raises(ImportError, match="schedule library is not installed"):
            TrendingScheduler()


@pytest.mark.unit
class TestRefreshTrendingRepositories:
    """Test refresh_trending_repositories method."""

    def test_refresh_with_config_errors(self, monkeypatch):
        """Test refresh fails with configuration errors."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        import importlib
        import trending.config
        import trending.scheduler
        importlib.reload(trending.config)
        importlib.reload(trending.scheduler)
        
        scheduler = TrendingScheduler()
        result = scheduler.refresh_trending_repositories()
        
        assert result is False

    def test_refresh_with_storage_disabled(self, monkeypatch):
        """Test refresh when storage is disabled."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        import importlib
        import trending.config
        import trending.scheduler
        importlib.reload(trending.config)
        importlib.reload(trending.scheduler)
        
        scheduler = TrendingScheduler()
        result = scheduler.refresh_trending_repositories()
        
        assert result is False

    def test_refresh_success(self, mock_database_connector):
        """Test successful refresh cycle."""
        import os
        import importlib
        os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
        
        # Reload config to pick up the environment variable
        import trending.config
        importlib.reload(trending.config)
        
        mock_database_connector.enabled = True
        
        from trending.fetcher import TrendingFetcher
        from trending.storage import TrendingStorage
        
        custom_fetcher = TrendingFetcher()
        custom_storage = TrendingStorage()
        custom_storage.connector = mock_database_connector
        custom_storage.enabled = True
        
        scheduler = TrendingScheduler(fetcher=custom_fetcher, storage=custom_storage)
        
        with patch.object(custom_storage, 'init_schema'):
            with patch.object(custom_storage, 'upsert_repositories', return_value=3):
                with patch.object(custom_fetcher, 'fetch_trending_repositories', return_value=[
                    {"full_name": "test/repo", "name": "repo", "owner": "test",
                     "url": "https://github.com/test/repo", "description": "",
                     "star_count": 100, "fork_count": 0, "created_at": "",
                     "pushed_at": "", "primary_language": "Python",
                     "topics": [], "readme": "", "default_branch": "main"}
                ]):
                    result = scheduler.refresh_trending_repositories()
        
        assert result is True

    def test_refresh_fetch_failure(self, mock_database_connector):
        """Test refresh when fetch fails."""
        from requests.exceptions import RequestException
        mock_database_connector.enabled = True
        
        from trending.fetcher import TrendingFetcher
        from trending.storage import TrendingStorage
        
        custom_fetcher = TrendingFetcher()
        custom_storage = TrendingStorage()
        custom_storage.connector = mock_database_connector
        custom_storage.enabled = True
        
        scheduler = TrendingScheduler(fetcher=custom_fetcher, storage=custom_storage)
        
        with patch.object(custom_storage, 'init_schema'):
            with patch.object(custom_fetcher, 'fetch_trending_repositories', side_effect=RequestException("Network error")):
                result = scheduler.refresh_trending_repositories()
        
        assert result is False

    def test_refresh_empty_repositories(self, mock_database_connector):
        """Test refresh when no repositories are fetched."""
        mock_database_connector.enabled = True
        
        from trending.fetcher import TrendingFetcher
        from trending.storage import TrendingStorage
        
        custom_fetcher = TrendingFetcher()
        custom_storage = TrendingStorage()
        custom_storage.connector = mock_database_connector
        custom_storage.enabled = True
        
        scheduler = TrendingScheduler(fetcher=custom_fetcher, storage=custom_storage)
        
        with patch.object(custom_storage, 'init_schema'):
            with patch.object(custom_fetcher, 'fetch_trending_repositories', return_value=[]):
                result = scheduler.refresh_trending_repositories()
        
        assert result is False

    def test_refresh_upsert_failure(self, mock_database_connector):
        """Test refresh when upsert fails."""
        mock_database_connector.enabled = True
        
        from trending.fetcher import TrendingFetcher
        from trending.storage import TrendingStorage
        
        custom_fetcher = TrendingFetcher()
        custom_storage = TrendingStorage()
        custom_storage.connector = mock_database_connector
        custom_storage.enabled = True
        
        scheduler = TrendingScheduler(fetcher=custom_fetcher, storage=custom_storage)
        
        with patch.object(custom_storage, 'init_schema'):
            with patch.object(custom_storage, 'upsert_repositories', return_value=0):
                with patch.object(custom_fetcher, 'fetch_trending_repositories', return_value=[
                    {"full_name": "test/repo", "name": "repo", "owner": "test",
                     "url": "https://github.com/test/repo", "description": "",
                     "star_count": 100, "fork_count": 0, "created_at": "",
                     "pushed_at": "", "primary_language": "Python",
                     "topics": [], "readme": "", "default_branch": "main"}
                ]):
                    result = scheduler.refresh_trending_repositories()
        
        assert result is False


@pytest.mark.unit
class TestStartOnce:
    """Test start_once method."""

    def test_start_once_success(self, mock_database_connector):
        """Test start_once returns True on success."""
        mock_database_connector.enabled = True
        
        from trending.fetcher import TrendingFetcher
        from trending.storage import TrendingStorage
        
        custom_fetcher = TrendingFetcher()
        custom_storage = TrendingStorage()
        custom_storage.connector = mock_database_connector
        custom_storage.enabled = True
        
        scheduler = TrendingScheduler(fetcher=custom_fetcher, storage=custom_storage)
        
        with patch.object(scheduler, 'refresh_trending_repositories', return_value=True):
            result = scheduler.start_once()
        
        assert result is True

    def test_start_once_failure(self, mock_database_connector):
        """Test start_once returns False on failure."""
        mock_database_connector.enabled = True
        
        from trending.fetcher import TrendingFetcher
        from trending.storage import TrendingStorage
        
        custom_fetcher = TrendingFetcher()
        custom_storage = TrendingStorage()
        custom_storage.connector = mock_database_connector
        custom_storage.enabled = True
        
        scheduler = TrendingScheduler(fetcher=custom_fetcher, storage=custom_storage)
        
        with patch.object(scheduler, 'refresh_trending_repositories', return_value=False):
            result = scheduler.start_once()
        
        assert result is False


@pytest.mark.unit
class TestStartScheduled:
    """Test start_scheduled method."""

    def test_start_scheduled_already_running(self):
        """Test start_scheduled when already running."""
        scheduler = TrendingScheduler()
        scheduler.running = True
        
        # Should not raise error, just log warning
        scheduler.start_scheduled()

    def test_start_scheduled_sets_running_flag(self):
        """Test start_scheduled sets running flag."""
        scheduler = TrendingScheduler()
        
        with patch.object(scheduler.scheduler, 'every') as mock_schedule:
            mock_job = MagicMock()
            mock_schedule.return_value.hours.return_value.do.return_value = mock_job
            with patch.object(scheduler.scheduler, 'run_pending'):
                with patch('time.sleep'):
                    # Make it exit immediately
                    def sleep_side_effect(seconds):
                        scheduler.running = False
                    import time
                    time.sleep = sleep_side_effect
                    
                    scheduler.start_scheduled()
        
        assert scheduler.running is False

    def test_stop_sets_running_flag(self):
        """Test stop method sets running flag."""
        scheduler = TrendingScheduler()
        scheduler.running = True
        
        scheduler.stop()
        
        assert scheduler.running is False


@pytest.mark.unit
class TestSignalHandlers:
    """Test signal handler setup."""

    def test_signal_handler_sets_running_flag(self):
        """Test that signal handler sets running flag to False."""
        scheduler = TrendingScheduler()
        scheduler.running = True
        
        # Simulate signal handler
        scheduler._signal_handler(2, None)  # SIGINT
        
        assert scheduler.running is False


@pytest.mark.unit
class TestModuleFunctions:
    """Test module-level functions."""

    def test_run_scheduler(self):
        """Test run_scheduler function."""
        with patch('trending.scheduler.TrendingScheduler') as MockScheduler:
            mock_instance = MagicMock()
            MockScheduler.return_value = mock_instance
            
            run_scheduler()
            
            MockScheduler.assert_called_once()
            mock_instance.start_scheduled.assert_called_once()

    def test_run_once(self):
        """Test run_once function."""
        with patch('trending.scheduler.TrendingScheduler') as MockScheduler:
            mock_instance = MagicMock()
            mock_instance.start_once.return_value = True
            MockScheduler.return_value = mock_instance
            
            result = run_once()
            
            MockScheduler.assert_called_once()
            mock_instance.start_once.assert_called_once()
            assert result is True

    def test_run_once_with_exception(self):
        """Test run_once function handles exceptions."""
        with patch('trending.scheduler.TrendingScheduler') as MockScheduler:
            MockScheduler.side_effect = Exception("Scheduler error")
            
            result = run_once()
            
            assert result is False
