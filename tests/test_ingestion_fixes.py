import pytest
import numpy as np
from ingestion_engine import (
    extract_tags, classify_category, compute_multi_dimensional_novelty,
    ingest_repository, CorpusStore, IngestionResult, _qdrant_uses_named_vectors, _qdrant_query_vector
)


# 1. extract_tags repository-name parsing
def test_extract_tags_repository_name_parsing():
    # duplicate words
    tags = extract_tags("my-react-react-app", [])
    assert tags.count("react") == 1
    assert "my" not in tags
    assert "app" not in tags

    # mixed case words
    tags = extract_tags("Vue-VuE-vUe", [])
    assert tags.count("vue") == 1

    # stop words
    tags = extract_tags("the-demo-project", [])
    assert len(tags) == 0

    # hyphenated repository names
    tags = extract_tags("react-admin-dashboard", [])
    assert set(tags) == {"react", "admin", "dashboard"}

    # underscore-separated names
    tags = extract_tags("python_machine_learning", [])
    assert set(tags) == {"python", "machine", "learning"}

# M1: Phrase deduplication
def test_extract_tags_phrase_deduplication():
    # test compound semantic concepts
    paragraphs = ["This is an AI Assistant that acts as a local AI."]
    tags = extract_tags("test-repo", paragraphs)
    assert tags.count("Ai Assistant") == 1
    assert tags.count("Local Ai") == 1
    
    # Should not add duplicates if lowercase exists
    paragraphs = ["The AI Assistant uses LLM."]
    # The title will have 'Ai Assistant', the tokens won't capture it as a single token but 'ai' and 'assistant'
    # Wait, 'Ai Assistant' should just be appended once.
    tags2 = extract_tags("ai-assistant", paragraphs)
    assert tags2.count("Ai Assistant") == 1

# 2. Qdrant named-vector retrieval (Mocked)
def test_qdrant_named_vector_query():
    class MockClient:
        def get_collection(self, name):
            class Config:
                class Params:
                    vectors = {"repo_embedding": type("V", (), {"size": 384, "distance": "COSINE"})}
                params = Params()
            class Info:
                config = Config()
            return Info()
            
    client = MockClient()
    vector = np.array([0.1, 0.2, 0.3])
    # check that named vector query formats correctly
    q_vector = _qdrant_query_vector(client, vector)
    assert getattr(q_vector, "name", None) == "repo_embedding" or isinstance(q_vector, dict) and q_vector.get("name") == "repo_embedding"

# 3. ingestion -> retrieval embedding flow
def test_ingestion_embedding_flow():
    repo = {"id": "test/repo", "star_count": 100, "readme_length": 500}
    store = CorpusStore()
    result = ingest_repository(repo, corpus_store=store)
    assert isinstance(result.embedding, list)
    assert len(result.embedding) == 384

# 4 & 5. topology disabled / enabled
def test_topology_paths():
    repo = {"id": "test/topo", "category": "Frontend", "primary_language": "JS", "star_count": 100, "readme_length": 500}
    store = CorpusStore()
    store.add_node({"repo_id": "test/1", "vector": np.ones(384), "category": "Frontend", "tags": [], "activity": 1.0}, 0.5)
    store.add_node({"repo_id": "test/2", "vector": np.ones(384), "category": "Backend", "tags": [], "activity": 1.0}, 0.5)
    store.add_node({"repo_id": "test/3", "vector": np.ones(384), "category": "Frontend", "tags": [], "activity": 1.0}, 0.5)
    store.add_node({"repo_id": "test/4", "vector": np.ones(384), "category": "Frontend", "tags": [], "activity": 1.0}, 0.5)

    # disabled path
    res_disabled = ingest_repository(repo, corpus_store=store, compute_topology=False)
    assert res_disabled.topological_metrics == {}

    # enabled path
    res_enabled = ingest_repository(repo, corpus_store=store, compute_topology=True)
    assert "current_cluster" in res_enabled.topological_metrics
    assert res_enabled.topological_metrics["current_cluster"] == "Frontend"

# 6. multi-neighbor novelty calculation
def test_multi_neighbor_novelty():
    qdrant_hits = [
        {"repo_id": "r1", "sim": 0.9, "category": "C1", "activity": 0.5, "tags": ["a", "b"]},
        {"repo_id": "r2", "sim": 0.8, "category": "C1", "activity": 0.5, "tags": ["a", "c"]},
    ]
    # target has tag "a"
    result = compute_multi_dimensional_novelty("new_repo", "C1", 0.5, ["a"], qdrant_hits, 100)
    # Jaccard for r1: a vs a,b => 1/2
    # Jaccard for r2: a vs a,c => 1/2
    # Mean overlap = 0.5
    # Tech stack novelty = 1.0 - 0.5 = 0.5
    assert result.tech_stack_penalty == 0.2

# 7. React vs ReAct classification
def test_react_classification():
    # Frontend react
    repo_ui = {
        "id": "facebook/react",
        "primary_language": "JavaScript",
        "extracted_paragraphs": ["React is a library for building user interfaces."]
    }
    cat_ui = classify_category(repo_ui, ["react", "ui", "javascript"])
    assert cat_ui == "Web/Frontend Frameworks"

    # AI Agent ReAct
    repo_agent = {
        "id": "reasoning/react-agent",
        "primary_language": "Python",
        "extracted_paragraphs": ["An autonomous agent using ReAct reasoning framework with LLM thought and action."]
    }
    cat_agent = classify_category(repo_agent, ["react", "agent", "llm"])
    assert cat_agent == "Data Engineering & AI/ML Pipelines"

# 8. concurrent ingestion behavior
import threading
def test_concurrent_ingestion():
    store = CorpusStore()
    
    def worker(i):
        store.add_node({"repo_id": str(i)}, 1.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.size() == 10
    assert len(store.get_timeline()) == 10
