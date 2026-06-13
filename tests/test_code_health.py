import math
from datetime import datetime, timedelta, timezone
import pytest
from ingestion.features import score_code_health, activity_score
from ingestion.pipeline import ingest_repository
from ingestion.corpus import CorpusStore

def test_score_code_health_boundaries():
    # 1. Base case: 0 issues, pushed 0 days ago, 1 language
    repo = {
        "open_issues_count": 0,
        "fork_count": 0,
        "pushed_days_ago": 0,
        "languages": ["Python"]
    }
    # Density score = 1.0 / (1.0 + 0) = 1.0
    # Recency score = exp(0) = 1.0
    # Complexity score = 1.0
    # Health score = 0.4*1 + 0.4*1 + 0.2*1 = 1.0
    assert score_code_health(repo) == 1.0

    # 2. High issue density
    repo_high_density = {
        "open_issues_count": 10,
        "fork_count": 2,  # density = 5.0
        "pushed_days_ago": 0,
        "languages": ["Python"]
    }
    # Density score = 1.0 / (1.0 + 5.0) = 1.0 / 6.0 = 0.1667
    # Recency score = 1.0
    # Complexity score = 1.0
    # Health score = 0.4 * 0.1667 + 0.4 * 1.0 + 0.2 * 1.0 = 0.0667 + 0.4 + 0.2 = 0.6667
    expected = 0.4 * (1.0 / 6.0) + 0.4 * 1.0 + 0.2 * 1.0
    assert math.isclose(score_code_health(repo_high_density), round(expected, 4))

    # 3. Old push date
    repo_old_push = {
        "open_issues_count": 0,
        "fork_count": 0,
        "pushed_days_ago": 90, # 2 half-lives of 45 days
        "languages": ["Python"]
    }
    # Density score = 1.0
    # Recency score = exp(-2) = 0.1353
    # Complexity score = 1.0
    # Health score = 0.4 * 1.0 + 0.4 * 0.1353 + 0.2 * 1.0 = 0.4 + 0.0541 + 0.2 = 0.6541
    expected_old = 0.4 * 1.0 + 0.4 * math.exp(-90 / 45.0) + 0.2 * 1.0
    assert math.isclose(score_code_health(repo_old_push), round(expected_old, 4))

    # 4. Multi-language complexity penalty
    repo_multi_lang = {
        "open_issues_count": 0,
        "fork_count": 0,
        "pushed_days_ago": 0,
        "languages": ["Python", "JavaScript", "Go", "C++"]  # 4 languages
    }
    # Complexity score = 1.0 - (4 - 2) * 0.15 = 0.70
    # Health score = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.70 = 0.4 + 0.4 + 0.14 = 0.94
    assert score_code_health(repo_multi_lang) == 0.94

    # 5. Capped complexity penalty
    repo_many_langs = {
        "open_issues_count": 0,
        "fork_count": 0,
        "pushed_days_ago": 0,
        "languages": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"] # 7 languages -> 1.0 - 5 * 0.15 = 0.25 -> capped at 0.40
    }
    # Health score = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.40 = 0.88
    assert score_code_health(repo_many_langs) == 0.88

def test_blended_gate_decision():
    store = CorpusStore()
    
    # 1. High documentation and high code health -> APPROVED
    repo_approved = {
        "id": "owner/good-repo",
        "star_count": 100,
        "pushed_days_ago": 0,
        "readme_length": 1000,
        "readme_to_codebase_ratio": 0.05,
        "extracted_paragraphs": [
            "Installation: pip install this",
            "Usage: import this",
            "API: reference rules",
            "Contributing: guidelines",
            "License: MIT",
            "FAQ: answers"
        ],
        "languages": ["Python"]
    }
    res = ingest_repository(repo_approved, corpus_store=store, auto_index=False)
    assert res.decision == "APPROVED"
    assert res.quadrant in ["🔥 Viral Rockets", "💎 Hidden Gems"]

    # 2. Low blended score -> REJECTED
    repo_rejected = {
        "id": "owner/poor-repo",
        "star_count": 100,
        "pushed_days_ago": 180,  # low code health
        "readme_length": 250,
        "extracted_paragraphs": ["Short readme without sections."],
        "languages": ["Python", "JS", "C++", "Java", "Go", "Rust", "HTML"] # high complexity penalty
    }
    res_rej = ingest_repository(repo_rejected, corpus_store=store, auto_index=False)
    assert res_rej.decision == "REJECTED"
    assert res_rej.quadrant in ["⚠️ Copycats / Clones", "💤 Dormant Ecosystem Nodes"]

def test_activity_score_geometric_mean():
    # Test fallback path
    repo_fallback = {
        "recent_commits": [],
        "pushed_days_ago": 0,
        "mentionable_users_count": 10
    }
    # recency = exp(0) = 1.0
    # contrib = min(10 / 10, 1.0) = 1.0
    # score = sqrt(1.0 * 1.0) = 1.0
    assert activity_score(repo_fallback) == 1.0

    # Test commit history path with 3 commits
    now = datetime.now(timezone.utc)
    commits = [
        (now - timedelta(days=1)).isoformat(),
        (now - timedelta(days=2)).isoformat(),
        (now - timedelta(days=3)).isoformat()
    ]
    repo_commits = {
        "recent_commits": commits
    }
    # commit_days: [1, 2, 3]
    # most_recent: 1 -> recency = exp(-1/14) = 0.9311
    # n_commits = 3, span = 2 days -> rate = 3 / 2 = 1.5 commits/day
    # frequency = min(log(2.5)/log(4.0), 1.0) = 0.6610
    # score = sqrt(0.9311 * 0.6610) = 0.7845
    score = activity_score(repo_commits)
    assert 0.0 < score < 1.0

def test_stargazer_extrapolation_formulas():
    from acquisition.repository_enricher import RepositoryEnricher
    
    class SimpleFakeClient:
        pass
        
    enricher = RepositoryEnricher(SimpleFakeClient())
    now = datetime.now(timezone.utc)
    
    # 1. Total stars <= 100: No extrapolation
    stargazers = [
        {"starred_at": (now - timedelta(days=i)).isoformat()}
        for i in range(1, 11)
    ]
    repo = {"stargazers_count": 10}
    deltas = enricher._estimate_star_deltas(repo, stargazers=stargazers, events=[])
    # Should just be exact count
    assert deltas[3] == 3
    assert deltas[7] == 7
    assert deltas[30] == 10

    # 2. Total stars > 100, but stargazers returned is >= 100 -> Extrapolation happens
    stargazers_large = [
        {"starred_at": (now - timedelta(days=i * 0.1)).isoformat()}  # 100 stargazers spanning 10 days
        for i in range(1, 101)
    ]
    repo_large = {"stargazers_count": 500}
    deltas_extrap = enricher._estimate_star_deltas(repo_large, stargazers=stargazers_large, events=[])
    # span_days = 9.9 days, rate = 100 / 9.9 = 10.101 stars/day
    # oldest is 10 days ago.
    # W=3: <= 10 days ago, so exact count. sum(starred_at within 3 days) = 30
    assert deltas_extrap[3] in [29, 30]
    # W=7: <= 10 days ago, so exact count. sum(starred_at within 7 days) = 70
    assert deltas_extrap[7] in [69, 70]
    # W=30: > 10 days ago. Extrapolated = observed + rate * (1 - exp(-0.05 * 20)) / 0.05
    # = 100 + 10.1 * (1 - exp(-1.0)) / 0.05 = 100 + 127.7 = 228
    assert deltas_extrap[30] > 100
    assert deltas_extrap[30] < 500
