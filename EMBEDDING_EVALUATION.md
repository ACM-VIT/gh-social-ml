# Embedding Evaluation

The below scripts are for measuring repository embedding quality after the
acquisition and Qdrant indexing pipeline has run.

## Embedding Architecture

The repository embedding pipeline uses `all-MiniLM-L6-v2` from
`sentence-transformers`. The model outputs 384-dimensional vectors and the
pipeline stores normalized vectors for cosine similarity search.

Each approved repository is projected through three embedding layers:

- README layer: cleaned README text is split into overlapping chunks, each
  chunk is embedded, and the chunk vectors are averaged into one README vector.
- Metadata layer: repository description, language, stars, forks, issues,
  recency, trend deltas, contributor signal, discovery category, and discovery
  band are converted into text and embedded.
- Topic layer: GitHub topics and repository languages are converted into text
  and embedded.

The Repo Tower combines these layers with fixed weights:

```text
README:   0.60
Metadata: 0.25
Topics:   0.15
```

The aggregation is a weighted average followed by L2 normalization. This is a
manual weighted tower, not a trained neural ranking model.

## Qdrant Configuration

Repository vectors are stored in Qdrant with the following expected collection
configuration:

```text
Collection:  osiris_research_corpus
Vector name: repo_embedding
Vector size: 384
Distance:    Cosine
```

The Qdrant payload stores repository inspection fields such as `repo_id`,
`description`, `primary_language`, `languages`, `topics`, `star_count`,
`readme_length`, `discovery_category`, `category`, `tags`, `doc_quality`,
`code_health`, `activity_score`, `trend_velocity`, `embedding_model`,
`embedding_version`, and `source_hash`.

Qdrant may use vector indexing such as HNSW for approximate nearest-neighbor
retrieval on larger collections. For evaluation, repository search explicitly
uses exact nearest-neighbor retrieval with `SearchParams(exact=True)`, so Qdrant
compares the query vector against stored repository vectors exactly.

The current retrieval path is:

```text
Query text or stored repository vector
        ↓
384-dimensional vector
        ↓
Qdrant query_points
        ↓
SearchParams(exact=True)
        ↓
Cosine similarity
        ↓
Top-k nearest repositories
```

Exact nearest-neighbor retrieval is not a separate ML model. It is a Qdrant
search mode used to remove approximate-index effects during retrieval quality
evaluation.

## Current Observations

On the existing indexed corpus used during ENN validation:

```text
Total vectors:                  314
Average similarity:             0.6334
Median similarity:              0.6327
Same-category retrieval:        59.68%
Cross-category retrieval:       40.32%
ENN/current top-k overlap:      100.00%
ENN/current same-rank match:    100.00%
Average ENN latency:            9.63 ms
Average previous-search latency: 10.94 ms
```

Because the corpus is small, ENN and the previous non-exact search path returned
the same top-k results during validation. ENN remains the preferred mode for
evaluation because it gives deterministic nearest-neighbor results.

The current system does not implement feed ranking, CTR prediction, novelty
scoring, contributor reputation scoring, collaborative filtering, or a learned
reranker. It only evaluates semantic repository vectors and Qdrant nearest
neighbor retrieval.

## Start Qdrant

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Optional environment variables:

```powershell
$env:QDRANT_URL="http://localhost:6333"
$env:QDRANT_COLLECTION_NAME="osiris_research_corpus"
$env:EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

## Index Repositories

The below command is for running discovery, enrichment, filtering, embedding,
and Qdrant indexing in one flow.

```powershell
python main.py --limit 50 --batch-size 10
```

Use `--no-index-qdrant` only when you want acquisition and filtering without
vector persistence.

## Evaluate Repository Embeddings

The below command is for evaluating vectors that already exist in Qdrant.
Repository search uses exact nearest-neighbor retrieval with
`SearchParams(exact=True)`.

```powershell
python evaluate_embeddings.py --sample-size 50 --query-count 10 --top-k 10
```

Compare exact nearest-neighbor retrieval against the previous non-exact Qdrant
search path:

```powershell
python evaluate_embeddings.py --sample-size 50 --query-count 10 --top-k 10 --compare-current
```

The below command is for indexing a local approved repository payload JSON file
before running the evaluation.

```powershell
python evaluate_embeddings.py --corpus-json staged_repositories.json --sample-size 50
```

The evaluation prints repository nearest neighbors, same-category and
cross-category similarity, clustering quality, vector distribution statistics,
retrieval consistency, qualitative examples, ENN/current overlap and latency
when requested, and tower-weight recommendations.

## Run Exact Nearest-Neighbor Evaluation

The below command evaluates every repository vector already indexed in Qdrant
by scrolling stored vectors in batches of 10 and querying with
`SearchParams(exact=True)`.

```powershell
python nearest_neighbor_eval.py
```

The generated `nearest_neighbor_diagnostic_report.md` includes ENN quality
metrics, corpus diagnostics, and a latency/quality comparison against
`exact=False`. Use `--skip-comparison` when you only need the ENN report.

## Run Qualitative Retrieval Tests

The below command is for checking natural-language retrieval behavior. Searches
use ENN by default.

```powershell
python retrieval_test.py
```

Compare ENN against the previous non-exact search behavior:

```powershell
python retrieval_test.py --compare-current "database ORM" "large language model framework"
```

Custom queries can be passed directly:

```powershell
python retrieval_test.py "python web framework" "vector database client"
```
