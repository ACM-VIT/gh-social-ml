import os
from unittest.mock import MagicMock, patch
import pytest
import requests

from utils.readme_processor import ReadmeDocument, process_markdown
from utils.groq_client import generate_readme_markdown
from acquisition.repository_enricher import EnrichmentResult, RepositoryEnricher
from database.connector import PostgreSQLConnector


@pytest.mark.unit
class TestGroqReadmeMarkdown:
    """Unit tests for Groq README Markdown generation."""

    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key", "GROQ_MODEL_ID": "llama-3.3-70b-versatile"})
    @patch("requests.post")
    def test_generate_readme_markdown_success(self, mock_post):
        """Test successful markdown generation using the Groq client."""
        # Setup mock response from Groq API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Cool Project\nThis is a cool project."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        clean_text = "Cool Project. This is a cool project."
        result = generate_readme_markdown(clean_text)

        # Assertions
        assert result == "# Cool Project\nThis is a cool project."
        mock_post.assert_called_once()
        called_url = mock_post.call_args[0][0]
        assert "api.groq.com/openai/v1/chat/completions" in called_url
        
        called_headers = mock_post.call_args[1]["headers"]
        assert called_headers["Authorization"] == "Bearer test_key"
        
        called_payload = mock_post.call_args[1]["json"]
        assert called_payload["model"] == "llama-3.3-70b-versatile"
        assert clean_text in called_payload["messages"][0]["content"]

    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @patch("requests.post")
    def test_generate_readme_markdown_strips_code_fences(self, mock_post):
        """Test that the Groq client cleans up markdown block wrappers from the model output."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "```markdown\n# Project Title\n\n- Bullet list\n```"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = generate_readme_markdown("Clean text")
        assert result == "# Project Title\n\n- Bullet list"

    @patch.dict(os.environ, {}, clear=True)
    def test_generate_readme_markdown_missing_key(self):
        """Test that generation fails gracefully (returns empty string) if no API key is configured."""
        # Ensure no keys exist in environ
        with patch.dict(os.environ, {}, clear=True):
            result = generate_readme_markdown("Clean text")
            assert result == ""

    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @patch("requests.post")
    def test_generate_readme_markdown_timeout_graceful(self, mock_post):
        """Test that HTTP timeouts are handled gracefully without raising exceptions."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        
        result = generate_readme_markdown("Clean text")
        assert result == ""

    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @patch("requests.post")
    def test_generate_readme_markdown_http_error_graceful(self, mock_post):
        """Test that HTTP errors (e.g. 500) are handled gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        result = generate_readme_markdown("Clean text")
        assert result == ""

    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @patch("utils.groq_client.rate_limiter")
    @patch("time.sleep")
    @patch("requests.post")
    def test_generate_readme_markdown_retry_on_429_success(self, mock_post, mock_sleep, mock_limiter):
        """Test that the Groq client retries on HTTP 429 and eventually succeeds."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            "choices": [{"message": {"content": "# Success"}}]
        }
        
        mock_post.side_effect = [mock_response_429, mock_response_200]
        
        result = generate_readme_markdown("Clean text")
        
        assert result == "# Success"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_with(2.0)

    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @patch("utils.groq_client.rate_limiter")
    @patch("time.sleep")
    @patch("requests.post")
    def test_generate_readme_markdown_retry_on_429_exhausted(self, mock_post, mock_sleep, mock_limiter):
        """Test that the Groq client exhausts retries on HTTP 429 and returns empty string."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_post.return_value = mock_response_429
        
        result = generate_readme_markdown("Clean text")
        
        assert result == ""
        assert mock_post.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("time.sleep")
    @patch("time.time")
    def test_rate_limiter_spacing(self, mock_time, mock_sleep):
        """Test that the GroqRateLimiter spaces out consecutive requests properly."""
        from utils.groq_client import GroqRateLimiter
        limiter = GroqRateLimiter(rpm_limit=15.0)  # 4s spacing
        
        mock_time.return_value = 100.0
        limiter.wait_if_needed()
        mock_sleep.assert_not_called()
        
        mock_time.return_value = 101.0
        limiter.wait_if_needed()
        mock_sleep.assert_called_with(3.0)


@pytest.mark.unit
class TestReadmeDocumentExtension:
    """Test that ReadmeDocument supports the new readme_markdown field."""

    def test_readme_document_fields(self):
        doc = ReadmeDocument(
            raw_markdown="# Raw",
            clean_text="Clean",
            extracted_paragraphs=["Clean"],
            readme_length=5,
            readme_markdown="# Formatted"
        )
        assert doc.readme_markdown == "# Formatted"
        
    def test_process_markdown_defaults_empty(self):
        doc = process_markdown("# Raw markdown title\n\nSome paragraph content that is long enough.")
        assert doc.readme_markdown == ""


@pytest.mark.unit
class TestEnricherIntegration:
    """Test that RepositoryEnricher populates the readme_markdown field."""

    @patch("acquisition.repository_enricher.generate_readme_markdown")
    def test_enricher_batch_populates_markdown(self, mock_generate):
        mock_generate.return_value = "# Processed Markdown"

        graphql_client = MagicMock()
        graphql_client.get_repositories_batch.return_value = {
            "test/repo": {
                "nameWithOwner": "test/repo",
                "name": "repo",
                "description": "desc",
                "url": "https://github.com/test/repo",
                "stargazerCount": 100,
                "size": 1024,
                "languages": {"edges": [{"size": 100, "node": {"name": "Python"}}]}
            }
        }
        # GraphQL readme mock
        graphql_client.get_readme.return_value = "# Test Repository\n\nThis is a longer test paragraph that passes length checks."

        enricher = RepositoryEnricher(graphql_client=graphql_client)
        results = enricher.get_repositories_batch([{"full_name": "test/repo"}])

        assert len(results) == 1
        res = results[0]
        assert res.readme.readme_markdown == "# Processed Markdown"
        assert res.payload["readme_markdown"] == "# Processed Markdown"
        mock_generate.assert_called_once()


@pytest.mark.unit
class TestDatabaseConnectorSchema:
    """Test database schema updates for readme_markdown."""

    def test_migration_columns_contains_readme_markdown(self):
        db = PostgreSQLConnector()
        col_names = [col[0] for col in getattr(db, "_migration_columns", [])]
        assert "readme_markdown" in col_names

    @patch("database.connector.PostgreSQLConnector.connect")
    def test_upsert_includes_readme_markdown(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        db = PostgreSQLConnector()
        db.enabled = True

        # Mock result item
        mock_readme = ReadmeDocument(
            raw_markdown="raw",
            clean_text="clean",
            extracted_paragraphs=[],
            readme_length=5,
            readme_markdown="# Markdown Content"
        )
        mock_result = MagicMock()
        mock_result.repo_id = "test/repo"
        mock_result.payload = {"html_url": "url", "star_count": 100, "fork_count": 5}
        mock_result.raw_repository = {}
        mock_result.languages = {}
        mock_result.topics = []
        mock_result.readme = mock_readme

        db.upsert_repositories([mock_result])

        # Verify SQL execution parameters
        mock_cursor.execute.assert_any_call("SAVEPOINT row_upsert;")
        
        # Check that one of the calls was the INSERT query
        insert_calls = [
            call for call in mock_cursor.execute.call_args_list 
            if "INSERT INTO Repo" in str(call[0][0])
        ]
        assert len(insert_calls) == 1
        
        # Verify that readme_markdown is in the query columns
        query_str = insert_calls[0][0][0]
        assert "readme_markdown" in query_str
        
        # Verify that # Markdown Content is in the parameters passed
        params = insert_calls[0][0][1]
        assert "# Markdown Content" in params
