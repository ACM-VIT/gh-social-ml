"""Unit tests for trending/fetcher.py module."""

from unittest.mock import MagicMock, patch
import pytest

from trending.fetcher import TrendingFetcher, GITHUB_TRENDING_URL
from requests.exceptions import RequestException


@pytest.mark.unit
class TestTrendingFetcherInit:
    """Test TrendingFetcher initialization."""

    def test_init_default(self):
        """Test initialization with default values."""
        fetcher = TrendingFetcher()
        assert fetcher.consecutive_failures == 0
        assert fetcher.session is not None
        assert 'User-Agent' in fetcher.session.headers


@pytest.mark.unit
class TestFetchTrendingPage:
    """Test _fetch_trending_page method."""

    @patch('trending.fetcher.requests.Session.get')
    def test_fetch_success(self, mock_get):
        """Test successful fetch of trending page."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>test</html>"
        mock_get.return_value = mock_response
        
        fetcher = TrendingFetcher()
        html = fetcher._fetch_trending_page()
        
        assert html == "<html>test</html>"
        mock_get.assert_called_once_with(GITHUB_TRENDING_URL, timeout=30.0)

    @patch('trending.fetcher.requests.Session.get')
    def test_fetch_retry_on_failure(self, mock_get):
        """Test retry behavior on failed requests."""
        mock_get.side_effect = [RequestException("Network error")] * 4
        
        fetcher = TrendingFetcher()
        
        with pytest.raises(RequestException):
            fetcher._fetch_trending_page()
        
        assert mock_get.call_count == 4  # Initial + 3 retries

    @patch('trending.fetcher.requests.Session.get')
    def test_fetch_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = RequestException("404 Not Found")
        mock_get.return_value = mock_response
        
        fetcher = TrendingFetcher()
        
        with pytest.raises(RequestException):
            fetcher._fetch_trending_page()


@pytest.mark.unit
class TestParseTrendingHtml:
    """Test _parse_trending_html method."""

    def test_parse_empty_html(self):
        """Test parsing empty HTML."""
        fetcher = TrendingFetcher()
        repos = fetcher._parse_trending_html("")
        
        assert repos == []

    def test_parse_html_with_repos(self):
        """Test parsing HTML with repository entries."""
        html = """
        <article class="Box-row">
            <h2>
                <a href="/test-owner/test-repo">test-repo</a>
            </h2>
            <p class="col-9">A test repository</p>
            <a href="/test-owner/test-repo/stargazers">
                <svg>star</svg>
                1,234
            </a>
            <a href="/test-owner/test-repo/forks">
                <svg>fork</svg>
                56
            </a>
            <span itemprop="programmingLanguage">Python</span>
        </article>
        """
        
        fetcher = TrendingFetcher()
        repos = fetcher._parse_trending_html(html)
        
        assert len(repos) == 1
        assert repos[0]["full_name"] == "test-owner/test-repo"
        assert repos[0]["owner"] == "test-owner"
        assert repos[0]["name"] == "test-repo"
        assert repos[0]["description"] == "A test repository"
        assert repos[0]["star_count"] == 1234
        assert repos[0]["fork_count"] == 56
        assert repos[0]["primary_language"] == "Python"

    def test_parse_html_with_multiple_repos(self):
        """Test parsing HTML with multiple repository entries."""
        html = """
        <article class="Box-row">
            <a href="/owner1/repo1">repo1</a>
            <span aria-label="star">100</span>
            <span itemprop="programmingLanguage">JavaScript</span>
        </article>
        <article class="Box-row">
            <a href="/owner2/repo2">repo2</a>
            <span aria-label="star">200</span>
            <span itemprop="programmingLanguage">Python</span>
        </article>
        """
        
        fetcher = TrendingFetcher()
        repos = fetcher._parse_trending_html(html)
        
        assert len(repos) == 2
        assert repos[0]["full_name"] == "owner1/repo1"
        assert repos[1]["full_name"] == "owner2/repo2"

    def test_parse_html_malformed_entry(self):
        """Test graceful handling of malformed entries."""
        html = """
        <article class="Box-row">
            <p>Invalid entry without proper structure</p>
        </article>
        <article class="Box-row">
            <a href="/valid-owner/valid-repo">valid-repo</a>
            <span aria-label="star">100</span>
            <span itemprop="programmingLanguage">Python</span>
        </article>
        """
        
        fetcher = TrendingFetcher()
        repos = fetcher._parse_trending_html(html)
        
        # Should skip malformed entry and parse valid one
        assert len(repos) == 1
        assert repos[0]["full_name"] == "valid-owner/valid-repo"


@pytest.mark.unit
class TestNormalizeRepository:
    """Test _normalize_repository method."""

    def test_normalize_complete_repository(self):
        """Test normalization of a complete repository."""
        fetcher = TrendingFetcher()
        repo = {
            "full_name": "test-owner/test-repo",
            "name": "test-repo",
            "owner": "test-owner",
            "url": "https://github.com/test-owner/test-repo",
            "description": "A test repository",
            "star_count": 1000,
            "fork_count": 50,
            "created_at": "2024-01-01T00:00:00Z",
            "pushed_at": "2024-06-01T00:00:00Z",
            "primary_language": "Python",
            "topics": [],
            "readme": "",
            "default_branch": "main",
        }
        
        result = fetcher._normalize_repository(repo)
        
        assert result["full_name"] == "test-owner/test-repo"
        assert result["name"] == "test-repo"
        assert result["owner"] == "test-owner"
        assert result["star_count"] == 1000
        assert result["fork_count"] == 50
        assert result["primary_language"] == "Python"
        assert result["default_branch"] == "main"

    def test_normalize_repository_with_missing_fields(self):
        """Test normalization of repository with missing fields."""
        fetcher = TrendingFetcher()
        repo = {
            "full_name": "test-owner/test-repo",
            "name": "test-repo",
            "owner": "test-owner",
            "url": "https://github.com/test-owner/test-repo",
            "description": None,
            "star_count": 100,
            "fork_count": 0,
            "created_at": None,
            "pushed_at": None,
            "primary_language": None,
            "topics": [],
            "readme": "",
            "default_branch": "main",
        }
        
        result = fetcher._normalize_repository(repo)
        
        assert result["description"] == ""
        assert result["primary_language"] == "Unknown"

    def test_normalize_resets_consecutive_failures(self):
        """Test that successful normalization resets consecutive failures."""
        fetcher = TrendingFetcher()
        fetcher.consecutive_failures = 5
        
        repo = {
            "full_name": "test-owner/test-repo",
            "name": "test-repo",
            "owner": "test-owner",
            "url": "https://github.com/test-owner/test-repo",
            "description": "Test",
            "star_count": 100,
            "fork_count": 0,
            "created_at": "",
            "pushed_at": "",
            "primary_language": "Python",
            "topics": [],
            "readme": "",
            "default_branch": "main",
        }
        
        fetcher._normalize_repository(repo)
        assert fetcher.consecutive_failures == 0


@pytest.mark.unit
class TestFetchTrendingRepositories:
    """Test fetch_trending_repositories method."""

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_fetch_success(self, mock_fetch, mock_parse):
        """Test successful fetch of trending repositories."""
        mock_fetch.return_value = "<html>test</html>"
        mock_parse.return_value = [
            {
                "full_name": "owner1/repo1",
                "name": "repo1",
                "owner": "owner1",
                "url": "https://github.com/owner1/repo1",
                "description": "Test repo 1",
                "star_count": 100,
                "fork_count": 10,
                "created_at": "",
                "pushed_at": "",
                "primary_language": "Python",
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
            {
                "full_name": "owner2/repo2",
                "name": "repo2",
                "owner": "owner2",
                "url": "https://github.com/owner2/repo2",
                "description": "Test repo 2",
                "star_count": 200,
                "fork_count": 20,
                "created_at": "",
                "pushed_at": "",
                "primary_language": "JavaScript",
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
        ]
        
        fetcher = TrendingFetcher()
        repos = fetcher.fetch_trending_repositories(limit=10)
        
        assert len(repos) == 2
        assert all("full_name" in repo for repo in repos)
        assert all("star_count" in repo for repo in repos)

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_fetch_empty_response(self, mock_fetch, mock_parse):
        """Test handling of empty response."""
        mock_fetch.return_value = "<html>test</html>"
        mock_parse.return_value = []
        
        fetcher = TrendingFetcher()
        repos = fetcher.fetch_trending_repositories()
        
        assert repos == []

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_fetch_respects_limit(self, mock_fetch, mock_parse):
        """Test that fetch respects the limit parameter."""
        mock_fetch.return_value = "<html>test</html>"
        mock_parse.return_value = [
            {
                "full_name": f"owner{i}/repo{i}",
                "name": f"repo{i}",
                "owner": f"owner{i}",
                "url": f"https://github.com/owner{i}/repo{i}",
                "description": f"Test repo {i}",
                "star_count": 100 + i,
                "fork_count": 10 + i,
                "created_at": "",
                "pushed_at": "",
                "primary_language": "Python",
                "topics": [],
                "readme": "",
                "default_branch": "main",
            }
            for i in range(50)
        ]
        
        fetcher = TrendingFetcher()
        repos = fetcher.fetch_trending_repositories(limit=10)
        
        assert len(repos) == 10

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_fetch_network_error(self, mock_fetch, mock_parse):
        """Test handling of network errors."""
        mock_fetch.side_effect = RequestException("Network error")
        
        fetcher = TrendingFetcher()
        
        with pytest.raises(RequestException):
            fetcher.fetch_trending_repositories()

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_fetch_with_continue_on_error(self, mock_fetch, mock_parse, monkeypatch):
        """Test continue-on-error behavior during normalization."""
        mock_fetch.return_value = "<html>test</html>"
        
        # Make the first repo fail, second succeed
        call_count = [0]
        
        def mock_normalize(repo):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Normalization error")
            return {
                "full_name": "test/repo",
                "name": "repo",
                "owner": "test",
                "url": "https://github.com/test/repo",
                "description": "",
                "star_count": 100,
                "fork_count": 0,
                "created_at": "",
                "pushed_at": "",
                "primary_language": "Python",
                "topics": [],
                "readme": "",
                "default_branch": "main"
            }
        
        mock_parse.return_value = [
            {
                "full_name": "owner1/repo1",
                "name": "repo1",
                "owner": "owner1",
                "url": "https://github.com/owner1/repo1",
                "description": "Test repo 1",
                "star_count": 100,
                "fork_count": 10,
                "created_at": "",
                "pushed_at": "",
                "primary_language": "Python",
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
            {
                "full_name": "owner2/repo2",
                "name": "repo2",
                "owner": "owner2",
                "url": "https://github.com/owner2/repo2",
                "description": "Test repo 2",
                "star_count": 200,
                "fork_count": 20,
                "created_at": "",
                "pushed_at": "",
                "primary_language": "JavaScript",
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
        ]
        
        import trending.config
        monkeypatch.setattr(trending.config, "CONTINUE_ON_ERROR", True)
        
        fetcher = TrendingFetcher()
        with patch.object(fetcher, '_normalize_repository', side_effect=mock_normalize):
            repos = fetcher.fetch_trending_repositories()
        
        # Should continue despite first failure
        assert len(repos) >= 1

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_fetch_max_consecutive_failures(self, mock_fetch, mock_parse, monkeypatch):
        """Test stopping after max consecutive failures."""
        mock_fetch.return_value = "<html>test</html>"
        
        def mock_normalize(repo):
            raise ValueError("Always fails")
        
        mock_parse.return_value = [
            {
                "full_name": f"owner{i}/repo{i}",
                "name": f"repo{i}",
                "owner": f"owner{i}",
                "url": f"https://github.com/owner{i}/repo{i}",
                "description": f"Test repo {i}",
                "star_count": 100 + i,
                "fork_count": 10 + i,
                "created_at": "",
                "pushed_at": "",
                "primary_language": "Python",
                "topics": [],
                "readme": "",
                "default_branch": "main",
            }
            for i in range(10)
        ]
        
        import trending.config
        monkeypatch.setattr(trending.config, "CONTINUE_ON_ERROR", True)
        monkeypatch.setattr(trending.config, "MAX_CONSECUTIVE_FAILURES", 3)
        
        fetcher = TrendingFetcher()
        with patch.object(fetcher, '_normalize_repository', side_effect=mock_normalize):
            repos = fetcher.fetch_trending_repositories()
        
        # Should stop after max failures
        assert len(repos) == 0
        assert fetcher.consecutive_failures >= 3
