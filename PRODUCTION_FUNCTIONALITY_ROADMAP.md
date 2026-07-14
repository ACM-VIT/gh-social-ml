# WEave Production Functionality Roadmap

This roadmap captures the current review findings and turns them into a concrete work plan for making WEave a functional, production-level GitHub repository discovery app. The near-term target is a 50K repository corpus with reliable recommendations, clean app flows, and a backend/ML boundary that can handle heavy traffic.

## Current State Summary

The app already has a meaningful foundation: repository ingestion, README enrichment, Qdrant embeddings, retrieval, an MMoE-style ranker, feed caching, onboarding, and feedback signals. The main gaps are not lack of ambition; they are integration boundaries, production hardening, and unfinished pieces that are already visible in the code.

The biggest architectural issue is that backend, database, ML, and feed cache ownership are mixed. The ML repo currently touches Postgres, updates repo counters, invalidates feed cache rows, updates Qdrant, ranks candidates, and accepts feedback. For production, backend should own transactional database state, while ML should own model state, embeddings, ranking, and derived features.

## Validation Snapshot

- Priority 0 items 1 and 2 are complete for the deployed online path; their detailed completion status and scope are recorded below.
- ML online architecture, retrieval, and feedback tests: 47 passed. The two remaining warnings are third-party deprecations.
- ML suite: 178 product tests passed. `tests/test_issue_30.py` is an untracked intentional debug test with a forced failure and is not a product regression.
- Backend TypeScript check and feedback contract tests: passed.
- Docker build validation remains pending because the local Docker daemon is unavailable.
- Frontend TypeScript check previously passed; frontend lint still needs its recorded 18 errors and 18 warnings addressed.
- Embedding model swap remains pending because only MiniLM dimensions/models are supported in code today.
- Current ingestion target is still around 3.5K repositories, while the product target is 50K.

## Product Goal

WEave should feel like a real personalized GitHub discovery feed:

- Users should see fresh, relevant, diverse repositories immediately after onboarding.
- Likes, saves, dislikes, dwell time, README opens, GitHub opens, and skips should all improve future recommendations.
- The feed should avoid repeated, stale, dead, low-quality, duplicate, or wrapper repositories.
- Users should be able to save, organize, revisit, and share repositories without the recommendation loop becoming inconsistent.
- The system should support at least 50K repositories and heavy concurrent feed traffic.

## Priority 0: Make The System Correct

These are blockers before any serious scaling or model work.

Completed items are marked with `✅ Completed` and the verification date once their success criteria have been met.

### 1. Normalize Feedback Events — ✅ Completed (2026-07-14)

Backend and ML previously disagreed on feedback actions. They now use the canonical vocabulary `impression`, `dwell`, `readme_open`, `github_open`, `like`, `save`, `share`, `dislike`, `unlike`, `unsave`, and `undislike`, with legacy `click` and `skip` normalized at ingestion.

Implementation status (2026-07-14): complete. The frontend and backend consume the shared feedback contract and schema, canonical events are stored in an append-only event log for deterministic feature replay, impressions are explicitly log-only, long dwell and open/save actions carry stronger intent, reversal events clear prior state, and contract tests cover every backend-to-ML disposition. The focused ML feedback suite also accepts the same canonical vocabulary.

Completed work:

- Define one canonical event contract shared by frontend, backend, and ML.
- Add an event schema file or OpenAPI contract.
- Map frontend events to canonical event names before sending to ML.
- Support negative/reversal events: `dislike`, `unlike`, `unsave`, `undislike`.
- Treat `impression` as a logged event, not a strong model update.
- Treat `readme_open`, `github_open`, `save`, and long dwell as stronger intent.
- Add tests proving every backend feedback event is accepted or intentionally ignored by ML.

Verified success criteria:

- No feedback action is silently dropped.
- Backend and ML tests use the same event vocabulary.
- Feedback can be replayed into model features.

### 2. Remove ML Direct Writes To App Database — ✅ Completed (2026-07-14)

The ML service previously mutated app-owned Postgres tables by incrementing repo counters and deleting recommendation cache rows. Those writes have been removed from the deployed online path.

Implementation status (2026-07-14): complete for the deployed online path. Backend interaction handlers own app-state writes and invalidate the Redis feed queue after committed changes. Backend-to-ML calls use a fail-closed shared secret and one bounded batch request. Feedback updates only Qdrant through a durable, bounded Redis Stream. Recommendation generation now uses approximate semantic search and indexed discovery channels in Qdrant, with no Postgres reads, writes, cache tables, or reverse backend dependency. Offline ingestion and legacy trending utilities still have optional Postgres integration; moving those jobs is a separate corpus-pipeline migration.

Completed work:

- Backend owns all writes to `activity`, `user_feedback`, repo/user counters, and aggregate state.
- ML feedback endpoint should enqueue or process ML-side updates only.
- ML can update Qdrant user vectors and model feature stores.
- Backend should invalidate Redis feed cache when user interaction state changes.
- Remove `update_postgres_metrics` from the ML feedback hot path.

Verified success criteria:

- ML can run without `DATABASE_URL` for online inference.
- Backend remains the source of truth for user/repo interaction state.
- ML consumes events through API, Redis stream, or an event log.

### 3. Normalize Likes, Saves, And Counters

Likes and saves are currently represented in multiple places: user counters, repo counters, activity state, and feedback rows. This creates drift.

Recommended schema direction:

- `interaction_events`: append-only events for audit/training.
- `user_repo_state`: current per-user repo state: liked, disliked, saved, last_seen, dwell_total, opens.
- `repo_stats`: materialized aggregate counters.
- `user_stats`: materialized aggregate counters.
- `user_feedback_features`: ML-ready feature table or event-derived feature view.

Work needed:

- Keep `activity` or replace it with `user_repo_state`.
- Move mutable aggregate counters out of core `repo` and `users`, or treat them as eventually consistent aggregates.
- Use database functions or backend services for toggles.
- Add unique indexes and migration scripts.
- Add backfill scripts from current activity data.

Success criteria:

- A like/save toggle updates one source of truth.
- Aggregate counters can be recomputed.
- ML training can use event history, not synthetic-only labels.

### 4. Fix Recommendation Cache Flow

The backend Redis delivery queue is the selected recommendation-cache owner. The obsolete ML Postgres batch table and its create/read/write paths were removed on 2026-07-14.

Work needed:

- Keep recommendation caching and delivery in backend Redis.
- Invalidate the backend Redis delivery queue after meaningful feedback.
- Add cache hit/miss metrics.

Success criteria:

- Feed generation is not repeated unnecessarily.
- Cache invalidation is deterministic.
- Multiple concurrent requests do not stampede ML.

### 5. Make CI Green

Current tests and checks reveal real hygiene problems.

Work needed:

- Fix the ML feedback unit test that expects `connect.call_count`, while the implementation uses `_get_connection`.
- Remove or quarantine `tests/test_issue_30.py`, which performs a live GitHub request and intentionally fails.
- Mark live-network tests as `slow` or `integration` and skip by default.
- Fix backend TypeScript `rowCount` usage.
- Fix frontend lint errors around refs/effects/auth ordering.
- Add a CI job for backend typecheck and frontend lint/typecheck.

Success criteria:

- ML tests pass locally without external network.
- Backend `npx tsc --noEmit` passes.
- Frontend `npx tsc --noEmit` and `npm run lint` pass.
- CI blocks regressions.

## Priority 1: Make Recommendations Actually Useful

### 6. Upgrade Candidate Retrieval

Current retrieval is too narrow for 50K repos. The candidate pool is around 150, while the desired design expects closer to 1000 before filtering/ranking.

Work needed:

- Increase candidate pool size in stages: 150 -> 300 -> 600 -> 1000.
- Add channel budgets: semantic, category, trending, freshness, exploration, collaborative, saved-similar.
- Use approximate Qdrant search for production requests.
- Add payload filters for language, category, activity, freshness, and quality.
- Add category-aware retrieval once repo category is consistently stored.
- Avoid `ORDER BY RANDOM()` for fallback retrieval at scale.

Success criteria:

- Ranking receives enough diverse candidates.
- Retrieval latency stays within target under load.
- Cold-start and warm-start users both get relevant results.

### 7. Implement Real Quality, Novelty, And Dedup Gates

The code has thresholds and design concepts for novelty, duplicate detection, wrapper detection, documentation quality, code health, and contributor reputation, but several are mocked or bypassed.

Work needed:

- Implement semantic duplicate detection using nearest-neighbor similarity.
- Add wrapper/template detection for low-originality repos.
- Use README quality, code health, activity, freshness, and contributor signals in ingestion gates.
- Store rejection reasons for observability.
- Add manual review tooling for borderline repos.
- Add corpus quality dashboards.

Success criteria:

- Low-quality repos are filtered before ranking.
- Duplicate/wrapper repos do not dominate the feed.
- Ingestion decisions are explainable.

### 8. Wire Feed Assembly Into The Real Path

Freshness and exploration injection exist separately, but they are not fully integrated into the main backend recommendation path.

Work needed:

- Apply feed assembly after heavy ranking in `/api/v1/recommendations/generate`.
- Add seen-repo removal.
- Add dead-repo removal.
- Add duplicate and creator diversity filtering.
- Add freshness slots.
- Add exploration slots.
- Add session-level feed shaping so the first 5, next 5, and next 5 items feel coherent.

Success criteria:

- Feed is not just sorted by rank score.
- Users see variety without losing relevance.
- Repeated feed requests do not return already-consumed repos.

### 9. Improve Personalization

The diagram references short-term persona, long-term persona, persona fusion, historical sessions, and onboarded interests. Some pieces exist, but the fusion layer is not mature.

Work needed:

- Store long-term user profile separately from session profile.
- Maintain short-term session intent from current impressions/dwell/clicks.
- Add a persona fusion layer with tunable weights.
- Separate positive signals, negative signals, and neutral exposure.
- Track interests by category, language, topic, and repo cluster.
- Add user controls to reset, tune, or explain interests.

Success criteria:

- The feed reacts to recent behavior without forgetting onboarding interests.
- Dislikes and skips reduce similar content.
- Saves and long dwell increase similar but not duplicate content.

## Priority 2: Scale To 50K Repositories

### 10. Build A Real Ingestion Pipeline

The current ingestion flow is demo-like in places and has a lower target count than the product goal.

Work needed:

- Change target corpus size to 50K through config, not hardcoding.
- Make ingestion resumable and idempotent.
- Add job queues for discovery, enrichment, embedding, and indexing.
- Respect GitHub API limits.
- Cache README/enrichment outputs.
- Add retry policies and dead-letter queues.
- Track ingestion state per repo.

Success criteria:

- The system can crawl and enrich 50K repos over time.
- Failed repos do not block the pipeline.
- Re-running ingestion does not duplicate data.

### 11. Upgrade Embedding Model Safely

MiniLM is the current default. The code blocks other models unless their dimensions are explicitly supported.

Recommended first migration:

- Move from `all-MiniLM-L6-v2` to `BAAI/bge-small-en-v1.5` first because it is still 384-dimensional.
- Later evaluate code-specialized or larger models such as Jina code embeddings or E5 variants.

Work needed:

- Add an embedding model registry with model name, dimension, version, distance metric, and normalization behavior.
- Create a new Qdrant collection per embedding version.
- Support dual-read or shadow-read during migration.
- Backfill repo embeddings.
- Backfill user profile embeddings.
- Evaluate retrieval quality before switching traffic.
- Update ranker input assumptions if dimensions change.

Success criteria:

- Embedding changes do not break Qdrant, ranker, user profiles, or tests.
- Old and new embeddings can be compared.
- Rollback is possible.

### 12. Make Qdrant Production-Ready

Work needed:

- Use approximate search for online retrieval.
- Add proper HNSW/index config.
- Add payload indexes for category, language, stars, freshness, quality, and status.
- Separate collections by vector version.
- Add collection validation at startup.
- Add vector count and search latency metrics.

Success criteria:

- Search latency remains predictable with 50K repos.
- Payload-filtered retrieval works.
- Collection mismatches fail fast.

### 13. Add Database Indexes And Query Hygiene

Work needed:

- Add unique index on `repo.full_name`.
- Add indexes on `repo.star_count`, `repo.updated_at`, `repo.created_at`, and owner/language/category fields.
- Add indexes on `activity.user_id`, `activity.repo_id`, and `activity.updated_at` or equivalent state table fields.
- Add indexes for saved repos and liked/disliked repos.
- Add GIN indexes for topics and language JSONB if those remain JSONB.
- Remove expensive random ordering from production paths.

Success criteria:

- Feed enrichment and activity lookup do not degrade with corpus size.
- Query plans are known and monitored.

## Priority 3: Model And Ranking Maturity

### 14. Replace Synthetic-Only Training

The current ranker training path is not production-grade. DVC references a missing training file, and the available generator is synthetic.

Work needed:

- Restore or rewrite `train_ranker.py`.
- Train from real events once enough data exists.
- Use synthetic data only for bootstrapping and tests.
- Define labels for CTR, save, GitHub open, README open, dwell, dislike, and follow.
- Add negative sampling.
- Add time-based train/validation splits.
- Add model artifact versioning.

Success criteria:

- Training is reproducible.
- Model metrics are reported.
- Deployed model can be traced to data and config.

### 15. Add Offline Evaluation

Work needed:

- Add Recall@K, NDCG@K, MAP@K.
- Track diversity, freshness, novelty, coverage, and duplicate rate.
- Add cold-start and warm-start evaluation sets.
- Add embedding quality evaluation.
- Add ranker ablation tests.

Success criteria:

- Every model or embedding change has a measurable result.
- The team can choose models based on evidence.

### 16. Add Online Experimentation

Work needed:

- Add experiment assignment.
- Add A/B or interleaving support.
- Track impression -> action funnels.
- Track feed latency and drop-off.
- Track per-source candidate contribution.

Success criteria:

- New ranking/feed strategies can be tested safely.
- Product decisions are based on user behavior.

## Priority 4: App Functionality

### 17. Improve Feed UX

Work needed:

- Make feed loading, empty, error, and retry states polished.
- Make saved/liked/disliked state consistent across refreshes.
- Avoid duplicate cards in one session.
- Add explainability hints such as "Because you like TypeScript tooling."
- Add controls for "show less like this" and "not interested."
- Make README open, GitHub open, save, like, and dwell events reliable.

Success criteria:

- The feed feels alive and personalized.
- User actions visibly affect future recommendations.

### 18. Improve Repository Detail And Save Flows

Work needed:

- Add a strong repository detail view.
- Show README summary, topics, languages, stars, freshness, and why recommended.
- Let users save to boards/collections.
- Make board saves update user state and ML feedback consistently.
- Add search and filters over saved repos.

Success criteria:

- Users can do something useful with recommendations beyond scrolling.

### 19. Improve Onboarding

Work needed:

- Use GitHub profile data, selected interests, skills, followed repos, starred repos, and tech stack.
- Generate initial persona and explain what it captured.
- Add skip/edit paths.
- Re-onboard or refresh persona when GitHub profile changes.

Success criteria:

- First feed is useful even for new users.
- Onboarding data maps cleanly into ML profile state.

## Priority 5: Reliability, Security, And Operations

### 20. Production API Boundaries

Work needed:

- Protect all internal ML endpoints with `INTERNAL_API_SECRET`.
- Add request IDs across backend and ML.
- Add structured logs.
- Add timeout, retry, and circuit breaker behavior for ML calls.
- Make Redis required in production, no in-memory feedback fallback.
- Add health checks for API, Redis, Qdrant, and DB.

Success criteria:

- Internal APIs are not publicly writable.
- Failures are visible and bounded.

### 21. Observability

Work needed:

- Track feed generation latency.
- Track candidate counts by channel.
- Track ranker latency.
- Track Qdrant latency.
- Track feedback queue lag.
- Track ingestion throughput.
- Track model version and embedding version per recommendation.

Success criteria:

- The team can debug bad feeds and slow feeds.

### 22. Deployment And Configuration

Work needed:

- Move ML Python dependencies to `uv`.
- Add `pyproject.toml` and `uv.lock`.
- Update Dockerfile to use `uv sync --frozen`.
- Add production env validation.
- Remove hardcoded local credentials from production compose.
- Add separate worker processes for ingestion, feedback consumption, and API.
- Add autoscaling strategy for API and workers.

Success criteria:

- Deployments are reproducible.
- API workers do not run heavy background jobs inline.

## Suggested Implementation Order

### Sprint 1: Stabilize Contracts

- Fix feedback event schema.
- Remove ML Postgres counter writes.
- Fix backend TypeScript.
- Fix frontend lint blockers.
- Fix ML tests and quarantine live-network tests.
- Protect ML feedback endpoint.

### Sprint 2: Feed Correctness

- Wire feed assembly into main generation path.
- Add seen/dead/duplicate filtering.
- Fix recommendation cache ownership.
- Add Redis feed invalidation from backend.
- Add basic feed metrics.

### Sprint 3: DB And Event Model

- Add normalized interaction events.
- Add user repo state table.
- Add aggregate stats tables or materialized views.
- Backfill from current activity.
- Update backend services to use the new source of truth.

### Sprint 4: 50K Corpus Pipeline

- Make ingestion resumable.
- Add job queues.
- Add enrichment caching.
- Add source-hash embedding skip logic.
- Scale Qdrant retrieval settings.
- Increase candidate pool gradually.

### Sprint 5: Embedding And Ranking Upgrade

- Add embedding model registry.
- Add BGE small support.
- Build new Qdrant collection.
- Backfill and evaluate embeddings.
- Restore reproducible ranker training.
- Add offline ranking metrics.

### Sprint 6: Product Polish

- Improve feed UI states.
- Improve repo detail page.
- Improve boards/saves flow.
- Add recommendation explanations.
- Add user controls for feedback and personalization.

## Open Decisions

- Should backend Redis be the only feed cache, or should ML also persist recommendation batches?
- Should ML consume feedback through synchronous API calls, Redis stream, or a durable event table?
- Should the first embedding upgrade preserve 384 dimensions, or should the team accept a full ranker/Qdrant migration to a larger vector size?
- Should repo counters remain denormalized for fast reads, or move fully to materialized aggregate tables?
- What is the target feed latency for first page and next page?
- What is the minimum acceptable quality threshold for adding a repo to the 50K corpus?

## Definition Of Production-Ready

The app should be considered production-ready for the 50K repo version when:

- All local and CI checks pass.
- Feedback contracts are shared and tested.
- Backend owns transactional DB state.
- ML owns embeddings, ranking, model state, and feature computation.
- Feed generation is cached, invalidated, and observable.
- Retrieval scales to 50K with predictable latency.
- Ingestion is resumable and idempotent.
- Embedding/ranker versions are traceable.
- Users see fresh, relevant, diverse, non-duplicate recommendations.
- The app provides useful save, board, detail, and feedback workflows.
