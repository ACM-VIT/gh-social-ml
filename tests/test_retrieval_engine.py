import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from retrieval_engine import RetrievalEngine


@pytest.fixture
def mock_retrieval_dependencies():
    """Mock Qdrant and ranking dependencies for RetrievalEngine."""
    with patch("retrieval_engine.QdrantClient") as mock_qdrant_cls, \
         patch("retrieval.CandidateRetriever") as mock_retriever_cls:
        
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        
        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever
        
        yield mock_qdrant, mock_retriever


def test_retrieval_engine_lazy_loading(mock_retrieval_dependencies):
    """Verify that online retrieval dependencies are loaded lazily."""
    _, _ = mock_retrieval_dependencies
    
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    
    # Internal references should start as None/unloaded
    assert engine._candidate_retriever is None
    assert engine._ranker is None
    
    # Trigger lazy loading properties
    retriever = engine.candidate_retriever
    assert retriever is not None
    assert engine._candidate_retriever is not None
    
    # Mocking RankerService instantiation to avoid loading actual torch weights in unit tests
    with patch("inference.ranker_service.RankerService") as mock_ranker_cls:
        ranker = engine.ranker
        assert ranker is not None
        assert engine._ranker is not None


def test_retrieval_engine_fetch_and_rank(mock_retrieval_dependencies):
    """Verify RetrievalEngine fetches user interest profile and ranks them using RankerService."""
    mock_qdrant, mock_retriever = mock_retrieval_dependencies
    
    # 1. Setup mock user profile with interests
    mock_user_point = MagicMock()
    mock_user_point.vector = [0.1] * 384
    mock_user_point.payload = {
        "user_id": "user_456",
        "skills": ["Python", "AI/ML"],
    }
    mock_qdrant.retrieve.return_value = [mock_user_point]
    
    # 2. Setup mock retriever returning 3 candidate points
    mock_retriever.retrieve_candidates.return_value = [
        {
            "repo_id": "owner/repo1",
            "full_name": "owner/repo1",
            "retrieval_source": "semantic",
            "retrieval_score": 0.85,
            "repo_embedding": [0.2] * 384,
            "star_count": 100,
            "primary_language": "Python",
        },
        {
            "repo_id": "owner/repo2",
            "full_name": "owner/repo2",
            "retrieval_source": "semantic",
            "retrieval_score": 0.80,
            "repo_embedding": [0.3] * 384,
            "star_count": 200,
            "primary_language": "Python",
        },
        {
            "repo_id": "owner/repo3",
            "full_name": "owner/repo3",
            "retrieval_source": "trending",
            "retrieval_score": 0.75,
            "repo_embedding": [0.4] * 384,
            "star_count": 300,
            "primary_language": "JavaScript",
        }
    ]
    
    # 3. Setup mock RankerService that ranks repo2 as #1, repo3 as #2, and repo1 as #3
    mock_ranker = MagicMock()
    mock_ranker.emb_dim = 384
    mock_ranker.score_batch.return_value = [
        {"repo_id": "owner/repo2", "predictions": {"p_follow": 0.525}},
        {"repo_id": "owner/repo3", "predictions": {"p_follow": 0.520}},
        {"repo_id": "owner/repo1", "predictions": {"p_follow": 0.055}},
    ]
    
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker = mock_ranker
    
    # Call fetch onboarding batches
    batches = engine.fetch_onboarding_batches("user_456")
    
    # Assert retriever is queried
    mock_retriever.retrieve_candidates.assert_called_once_with(
        user_embedding=[0.1] * 384,
        user_interests=["Python", "AI/ML"]
    )
    
    # Assert RankerService.score_batch was called with the candidates
    assert mock_ranker.score_batch.call_count == 1
    call_args = mock_ranker.score_batch.call_args[0]
    np.testing.assert_array_almost_equal(call_args[0], np.array([0.1] * 384, dtype=np.float32))
    assert call_args[1] == ["Python", "AI/ML"]  # user skills
    
    # Assert batches are constructed in the ranker-sorted order
    batch_1 = batches["batch_1"]
    assert len(batch_1) == 3
    assert batch_1[0]["repo_id"] == "owner/repo2"
    assert batch_1[1]["repo_id"] == "owner/repo3"
    assert batch_1[2]["repo_id"] == "owner/repo1"
    
    # Assert scores are updated to the ranker score
    # repo2 final_score: p_follow (0.525) * 20.0 = 10.5
    # repo3 final_score: p_follow (0.520) * 10.0 = 5.2
    # repo1 final_score: p_follow (0.055) * 20.0 = 1.1
    assert batch_1[0]["final_score"] == pytest.approx(10.5)
    assert batch_1[1]["final_score"] == pytest.approx(5.2)
    assert batch_1[2]["final_score"] == pytest.approx(1.1)


def test_cold_start_uses_qdrant_profile_without_database_url(
    mock_retrieval_dependencies,
    monkeypatch,
):
    mock_qdrant, mock_retriever = mock_retrieval_dependencies
    monkeypatch.delenv("DATABASE_URL", raising=False)

    user_point = MagicMock()
    user_point.vector = [0.1] * 384
    user_point.payload = {"skills": ["Python"], "tech_stack": ["FastAPI"]}
    mock_qdrant.retrieve.return_value = [user_point]
    mock_retriever.retrieve_candidates.return_value = []

    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    batches = engine.fetch_onboarding_batches("cold-user", is_cold_start=True)

    assert batches == {"batch_1": [], "batch_2": [], "batch_3": []}
    mock_retriever.retrieve_candidates.assert_called_once_with(
        user_embedding=[0.1] * 384,
        user_interests=["Python", "FastAPI"],
    )
    assert not hasattr(engine, "_db")
