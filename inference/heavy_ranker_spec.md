# Heavy Ranker Architecture Specification

## 1. Overview
The Heavy Ranker is a PyTorch-based Multi-gate Mixture-of-Experts (MMoE) Neural Network. It acts as the second stage in a Two-Stage Recommendation Pipeline, taking a candidate pool of 150 repos from the Vector Database (Light Ranker) and scoring them in real-time.

## 2. Input Tensor (Dimension: 778)
The network requires a single concatenated tensor of size 778, consisting of 3 core blocks:

### A. Semantic Embeddings (768 dims)
- `user_embedding`: 384-dimensional dense float array (Tracks semantic intent)
- `repo_embedding`: 384-dimensional dense float array (Tracks semantic content)

### B. Dense Metadata Features (10 dims)
*Note: All 10 features must be processed using a `StandardScaler` (to normalize them to a standard distribution) before entering the network.*

**Quality & Health Metrics**
1. `doc_quality` (0.0 to 1.0 scale)
2. `code_health` (0.0 to 1.0 scale)
3. `readme_length` (Integer word/character count)

**Popularity & Engagement Metrics**
4. `star_count` (Integer)
5. `fork_count` (Integer)
6. `open_issues_count` (Integer)

**Freshness & Momentum Metrics**
7. `pushed_days_ago` (Integer days)
8. `activity_score` (Float)
9. `trend_velocity` (Float)

**Dynamic Cross-Features**
10. `skill_match_score` (Float percentage 0.0-1.0)
    - This is entirely stateless and calculated inside `RankerService`. It performs a fast set intersection between the user's profile terms and the repository `languages/topics/tags` Qdrant payload fields.

## 3. Network Outputs (5 Tasks)
The MMoE architecture routes the 778-dim input through 4 Shared Experts, and outputs 5 distinct predictions via 5 Task Heads:

1. `Click Probability (CTR)` (Binary Classification -> Sigmoid, 0.0 to 1.0)
2. `Save Probability` (Binary Classification -> Sigmoid, 0.0 to 1.0)
3. `GitHub Open Probability` (Binary Classification -> Sigmoid, 0.0 to 1.0)
4. `Follow Probability` (Binary Classification -> Sigmoid, 0.0 to 1.0)
5. `Dwell Time` (Regression -> ReLU, normalized fraction 0.0 to 1.0, where 1.0 = 600s)

## 4. The Value Function (Final Ranking)
The 5 outputs are fed into the canonical ML value function in `inference/value_function.py`. The candidate pool is sorted descending by this internal score before feed shaping.

`Score = (1.0 * p_ctr) + (5.0 * p_save) + (2.0 * p_gh) + (0.1 * p_dwell) + (20.0 * p_follow)`

## 5. Backend v2 Serving Contract
The Heavy Ranker evaluates the entire candidate pool before shaping or truncation. Qdrant records must use the backend-issued UUID `repo_id`; names and URLs are attributes only.

**The Production Flow:**
1. **Retrieval:** Qdrant supplies semantic and discovery candidates with complete payloads and 384-dimensional vectors.
2. **True Ranking:** ML passes the complete candidate pool through the network and sorts it by the canonical value score.
3. **Feed Shaping:** ML removes seen UUIDs and applies freshness, language diversity, and exploration.
4. **Recommendation Response:** ML returns only unique `{repo_id, score, source}` items plus `schema_version`, `generation_id`, `user_id`, `feed_version`, `model_version`, and `embedding_version`.
5. **Backend Serve:** The backend persists the feed serve and hydrates repository cards from PostgreSQL. ML never serves repository metadata directly to the frontend.

Candidate ranking is performed directly inside the canonical V2 retrieval pipeline. No earlier feed-batch adapter participates in serving.
