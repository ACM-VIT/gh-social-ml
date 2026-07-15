# Person 5 Workstream Handoff

## Status

The Person 5 online feedback/API workstream has been implemented and committed locally,
but has not been pushed or opened as a pull request.

The focused Person 5 suite and the existing feedback-scoring/retrieval compatibility
tests pass: **37 passed**. The full repository suite previously completed with **165 passed and 8
unrelated failures** caused by blocked external GitHub access and Windows logger-file
cleanup. See **Verification** below before merging.

## Scope implemented

Person 5 owns the online feedback worker, Redis transport, ML API security/lifecycle,
health reporting, API dependencies, and online-architecture tests. The revised online
path now uses Redis and Qdrant without importing or initializing Postgres.

### Canonical feedback contract

The action vocabulary is centralized in `feedback/interactions.py` and exposed as an
immutable mapping:

| Action | Reference score | Vector alpha | Real-time ML |
|---|---:|---:|---:|
| `impression` | `0.0` | `0.0` | No |
| `dwell` | `0.0` | Dynamic, maximum `0.15` | After threshold |
| `readme_open` | `0.2` | `0.05` | Yes |
| `github_open` | `0.3` | `0.07` | Yes |
| `share` | `0.6` | `0.10` | Yes |
| `like` | `1.0` | `0.15` | Yes |
| `dislike` | `-1.0` | `-0.15` | Yes |
| `save` | `0.8` | `0.20` | Yes |
| `unlike` | clears like | exact stored reversal | Yes |
| `undislike` | clears dislike | exact stored reversal | Yes |
| `unsave` | clears save | exact stored reversal | Yes |

Compatibility properties such as `feedback_score`, `persists_feedback`, and
`clears_interaction_type` remain read-only so existing ranking tests can transition
without restoring online database behavior.

### Versioned event schema

`feedback/events.py` introduces `FeedbackEvent`. The API/stream contract contains:

```json
{
  "event_id": "stable idempotency key",
  "user_id": "application user UUID",
  "repo_id": "stable repository identity",
  "action": "canonical action",
  "occurred_at": "RFC3339 timestamp with timezone",
  "schema_version": 1,
  "dwell_seconds": null
}
```

Validation now rejects unknown actions, unsupported schema versions, invalid/missing
identities, non-finite or negative dwell values, dwell without `dwell_seconds`, and
`dwell_seconds` on non-dwell actions. Alpha is always calculated inside ML; it is not
accepted from callers.

`impression` is accepted as canonical telemetry but is not placed on the real-time ML
stream.

### Vector update and exact reversals

The forward update implements the formula selected from the architecture PDF:

```text
delta = alpha * (repository_vector - latent_user_vector)
updated_latent = latent_user_vector + delta
search_vector = L2_normalize(updated_latent)
```

The existing `user_profiles` Qdrant point is used as the ML-owned durable state boundary.
Its payload stores:

- `feedback_latent_vector`: the unnormalized working vector;
- `feedback_adjustments`: exact active reversible deltas, separated into reaction and
  save families;
- `feedback_applied_signals`: compact per-repository markers for one-time passive signals;
- `feedback_processed_events`: a bounded durable replay guard.

The normalized vector remains the Qdrant search vector. The normalized vector, latent
vector, adjustment state, and processed-event history are written in one Qdrant point
upsert.

Exact action-state behavior:

- duplicate `like`, `dislike`, or `save` state is a no-op;
- a reversal without its active forward state is a no-op;
- `unlike`, `undislike`, and `unsave` subtract the exact delta stored by the forward
  action;
- like and dislike share the mutually exclusive `reaction` family;
- switching like to dislike, or dislike to like, removes the previous delta before
  calculating and adding the new delta;
- save state is independent and can coexist with either reaction;
- `readme_open`, `github_open`, and `share` update each user/repository profile only once;
- later occurrences of those passive actions remain valid telemetry but do not shift the
  vector again;
- distinct qualifying dwell events continue to update the profile;
- duplicate `event_id` does not update the vector again;
- missing user/repository points are retryable failures, not zero-vector fallbacks;
- wrong-dimensional, zero, NaN, or infinite vectors are rejected.

User and repository Qdrant IDs preserve the existing deterministic conventions:

```text
user point: uuid5(NAMESPACE_URL, "user:{user_id}")
repo point: uuid5(NAMESPACE_URL, "github:{repo_id}")
```

Default collections/vectors remain:

- repository collection: `osiris_research_corpus`;
- repository named vector: `repo_embedding`;
- user collection: `user_profiles`;
- dimension: `384`.

No new shared Qdrant collection was introduced.

### Dwell policy

Dwell is implemented as a pure, configurable linear function:

- below `FEEDBACK_DWELL_MIN_SECONDS`: no vector update;
- between minimum and full-credit duration: linear alpha growth;
- at or above full-credit duration: alpha capped at `0.15`.

Current defaults:

```text
FEEDBACK_DWELL_MIN_SECONDS=3
FEEDBACK_DWELL_FULL_CREDIT_SECONDS=300
FEEDBACK_DWELL_MAX_ALPHA=0.15
```

These are visible configuration values rather than hidden constants.

### Redis producer

`feedback/producer.py` now:

- requires Redis in production;
- only permits memory fallback through explicit
  `FEEDBACK_ALLOW_MEMORY_FALLBACK=true` in a non-production environment;
- rejects enabling memory fallback in production;
- publishes with bounded `XADD MAXLEN ~ FEEDBACK_STREAM_MAXLEN`;
- publishes the stable logical `event_id` and versioned event fields;
- performs connectivity validation during API startup;
- does not silently fall back to memory after a Redis failure.

### Redis consumer

`feedback/consumer.py` now implements:

- idempotent consumer-group creation;
- unique per-process consumer names;
- `XREADGROUP` for new messages;
- `XAUTOCLAIM` for abandoned pending messages;
- configurable batch, blocking, reclaim, and idle settings;
- logical idempotency keys based on `event_id`;
- a Redis distributed per-user lock to prevent concurrent read-modify-write loss;
- ACK only after the handler succeeds and the Redis completion marker is written;
- no ACK for retryable failures before the configured retry limit;
- bounded attempt tracking;
- a bounded dead-letter stream for malformed, non-retryable, or exhausted messages;
- graceful task cancellation and shutdown;
- explicit development-only memory processing.

Qdrant also stores the processed event ID in the same point upsert as the vector. This
closes the failure window where Qdrant succeeds but the Redis completion marker/ACK fails:
the reclaimed event becomes a Qdrant-level idempotent no-op.

### API security

Authentication is centralized in FastAPI middleware. Every route except exactly
`/api/v1/health` requires the internal secret, including feedback, recommendation,
onboarding, embedding, documentation, OpenAPI, and unknown routes.

Defaults:

```text
INTERNAL_API_HEADER=x-internal-secret
INTERNAL_API_SECRET=<required secret>
```

Secret comparison uses `hmac.compare_digest`. Missing server configuration fails closed
with `503`; missing/incorrect caller credentials return `401`. Secrets and raw internal
exceptions are not returned or logged.

The feedback endpoint returns `202` only after Redis accepts a real-time event. Redis
publication failures return `503`.

The transitional `GET /api/v1/feedback/{user_id}` SQL-backed endpoint was removed.

### API lifecycle and health

The FastAPI lifespan now:

1. loads and validates online feedback settings;
2. creates Redis and Qdrant clients without any database connector;
3. pings Redis;
4. checks Qdrant connectivity;
5. creates/validates the consumer group;
6. starts exactly one managed consumer task;
7. gracefully cancels the task and closes clients on shutdown.

No network clients or background tasks start at module import time. The application and
its complete feedback lifecycle can be imported/started with `DATABASE_URL` absent.

`GET /api/v1/health` is public and reports real state for:

- lifecycle initialization;
- Redis connectivity;
- Qdrant connectivity;
- consumer task liveness.

It returns `200` only when all checks pass, otherwise `503`.

### Other ML services

Recommendation, onboarding, and repository-embedding implementations are loaded lazily
when their endpoints are called. This prevents the API module and feedback lifecycle from
transitively importing the current Postgres-dependent `retrieval_engine.py`.

This is an integration boundary, not the final fix for ranking: Person 4 still needs to
remove `FeedbackStore`, SQL batch caching, and Postgres from `retrieval_engine.py` before
the recommendation endpoint is Qdrant-only at runtime.

## Files changed

### Added

- `feedback/events.py` — versioned feedback event and validation.
- `feedback/settings.py` — isolated, validated online configuration.
- `requirements-api.txt` — API-only dependencies without PostgreSQL drivers.
- `tests/test_online_architecture.py` — transitive online import/SQL/database guardrails.
- `handoff.md` — this handoff.

### Updated

- `feedback/interactions.py` — immutable canonical action registry.
- `feedback/event_handlers.py` — Qdrant-only vector learning and exact reversal state.
- `feedback/producer.py` — bounded production Redis Stream publishing.
- `feedback/consumer.py` — groups, reclaim, locking, idempotency, ACK, retry, and DLQ.
- `feedback/__init__.py` — revised public feedback API.
- `api/main.py` — authentication, schema, lifecycle, health, lazy service boundaries.
- `.env.example` — documented Redis/Qdrant/API/dwell/worker settings.
- `tests/test_feedback.py` — focused action, vector, Redis, API, lifecycle, and concurrency tests.

### Deliberately not changed

- `app.py` was already modified before this work began. Its change is unrelated and must
  not be included in the Person 5 commit unless its owner explicitly requests it.
- `feedback/storage.py` remains temporarily because the current Person 4
  `retrieval_engine.py` and existing compatibility tests still import it. It is no longer
  reachable from `api.main` or the feedback worker. Person 4 must remove the final callers,
  after which the legacy file and `tests/test_feedback_scoring.py` SQL-store tests can be
  deleted or replaced.
- Root `config.py`, `requirements.txt`, `tests/conftest.py`, retrieval/ranking, embedding,
  acquisition, ingestion, database, and backend-owned files were not edited.

## Configuration added

The following variables are documented in `.env.example`:

```text
APP_ENV
INTERNAL_API_SECRET
INTERNAL_API_HEADER
REDIS_URL
FEEDBACK_ALLOW_MEMORY_FALLBACK
FEEDBACK_STREAM_NAME
FEEDBACK_STREAM_MAXLEN
FEEDBACK_CONSUMER_GROUP
FEEDBACK_CONSUMER_PREFIX
FEEDBACK_READ_BATCH_SIZE
FEEDBACK_READ_BLOCK_MS
FEEDBACK_RECLAIM_IDLE_MS
FEEDBACK_RECLAIM_INTERVAL_SECONDS
FEEDBACK_IDEMPOTENCY_TTL_SECONDS
FEEDBACK_USER_LOCK_TTL_SECONDS
FEEDBACK_USER_LOCK_WAIT_SECONDS
FEEDBACK_MAX_DELIVERY_ATTEMPTS
FEEDBACK_DEAD_LETTER_STREAM
FEEDBACK_PROCESSED_EVENT_HISTORY
FEEDBACK_DWELL_MIN_SECONDS
FEEDBACK_DWELL_FULL_CREDIT_SECONDS
FEEDBACK_DWELL_MAX_ALPHA
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION_NAME
QDRANT_VECTOR_NAME
USER_PROFILES_COLLECTION
USER_PROFILE_VECTOR_NAME
VECTOR_DIMENSION
```

Notable defaults:

- stream length: `100000`;
- Redis idempotency TTL: `604800` seconds (7 days);
- Qdrant processed-event history: `512` IDs per user;
- per-user Redis lock TTL: `60` seconds;
- maximum delivery attempts: `5`;
- reclaim idle time: `60000` ms.

## Verification

Commands completed successfully:

```powershell
python -m py_compile feedback\settings.py feedback\interactions.py `
  feedback\events.py feedback\event_handlers.py feedback\producer.py `
  feedback\consumer.py api\main.py

python -m pytest tests\test_feedback.py tests\test_online_architecture.py `
  tests\test_feedback_scoring.py tests\test_retrieval_engine.py -q
```

Result:

```text
37 passed, 2 warnings
```

The warnings were unrelated dependency deprecations:

- Starlette/FastAPI TestClient warning about the installed `httpx` transition;
- PyTorch warning that `pynvml` is deprecated.

Coverage includes:

- canonical actions and alpha values;
- PDF vector formula and normalization;
- invalid vector handling;
- dwell thresholds/cap;
- one-time readme-open, GitHub-open, and share learning per user/repository;
- continued learning from distinct qualifying dwell events;
- exact like/unlike, dislike/undislike, and save/unsave reversal;
- reaction switching and save independence;
- duplicate event/state handling;
- missing Qdrant data retry behavior;
- bounded stream publishing;
- production Redis requirement;
- success ACK ordering and failed-message non-ACK behavior;
- pending-message reclaim;
- same-user serialization;
- API authentication and event validation;
- impression not entering the real-time stream;
- health dependency checks;
- lifecycle startup with `DATABASE_URL` absent;
- transitive online import-graph database checks;
- SQL and `FeedbackStore` detection in reachable online modules;
- compatibility with existing feedback-scoring/retrieval tests.

The full suite was also run:

```powershell
python -m pytest -q
```

Result:

```text
165 passed, 8 failed, 7 warnings
```

None of the failures were in Person 5 files or tests:

- `tests/test_fetcher.py`: 1 failure because sandboxed HTTPS access to
  `github.com/trending` was blocked;
- `tests/test_integration.py`: 5 failures caused by the same unavailable GitHub network
  dependency;
- `tests/test_logger.py`: 2 Windows-only cleanup failures because active logging handlers
  still held temporary files open.

All Person 5 tests passed within the full run. Re-run the external/network tests in an
environment that permits GitHub access, and run the logger tests after their Windows
handler-cleanup issue is resolved.

## Known limitations and decisions requiring confirmation

1. **Qdrant payload growth:** every active reversible repo state stores a 384-float delta
   in the user point payload. This avoids inventing a new collection and provides exact
   reversal, but users with very large numbers of active saved/reaction states may create
   large payloads and increasingly expensive point rewrites. Confirm expected state volume.

2. **Bounded idempotency:** Qdrant retains the latest 512 event IDs and Redis markers live
   for seven days by default. A replay older than both bounds may be treated as new. Confirm
   the required replay horizon and size these values accordingly.

3. **Redis/Qdrant are not one transaction:** Qdrant point-level idempotency closes the main
   Qdrant-success/Redis-failure replay window, but Redis stream state and Qdrant cannot be
   committed atomically. The implementation uses retry-safe state and documented ordering.

4. **Distributed lock lease:** the per-user lock defaults to 60 seconds. If Qdrant updates
   can exceed this, increase the lease or implement lease renewal before horizontal scaling.

5. **Dwell policy tuning:** dwell is approved as a repeatable learning signal. The current
   function is linear from 3 to 300 seconds and caps at 0.15; tune these production values
   if observed behavior warrants it.

6. **Passive-signal repetition:** the product decision is implemented. `readme_open`,
   `github_open`, and `share` apply once per user/repository regardless of new event IDs.
   Qualifying dwell events remain repeatable learning signals.

7. **Telemetry ownership:** `impression` is accepted but intentionally not published to the
   real-time stream. Confirm that the backend/offline pipeline persists it for training.

8. **Repository identity:** the worker assumes the incoming `repo_id` is exactly the identity
   used by repository indexing in `uuid5("github:{repo_id}")`. Person 2 and Person 6 must
   freeze and verify that contract.

9. **Multi-vector user profiles:** if `user_profiles` has multiple named vectors,
   `USER_PROFILE_VECTOR_NAME` must be configured. A single unnamed or unambiguous named
   vector works automatically.

10. **Person 4 dependency:** `retrieval_engine.py` still contains SQL, `FeedbackStore`, and
    Postgres batch caching. It is no longer part of API import/startup, but must be fixed by
    Person 4 before recommendations satisfy the final online architecture.

11. **Real infrastructure validation:** current tests mock Redis and Qdrant. Run at least one
    integration test against supported production versions to verify `XAUTOCLAIM`, lock
    behavior, Qdrant named-vector preservation, and payload size/performance.

## Suggested next steps

1. Wait for/finalize the Person 2 Qdrant identity, vector-name, and collection contract.
2. Agree with Person 6 on the event schema, secret header, state-transition behavior, and
   impression/offline-telemetry ownership.
3. Tune dwell thresholds/function if production observations require it; passive-event
   repetition is already resolved and implemented.
4. Re-run the eight unrelated full-suite failures in an environment with GitHub network
   access and corrected Windows logger-handler cleanup.
5. Run Redis/Qdrant integration tests using production-compatible versions.
6. After Person 4 removes the final callers, delete `feedback/storage.py` and its legacy SQL
   tests.
7. Merge Person 5 after vector, acquisition, retrieval, and ranking workstreams, per the
   documented merge order.
8. Make the final shared dependency/CI commit after all ML feature branches merge.
9. Exclude the pre-existing unrelated `app.py` modification from the Person 5 commit.

## No remote actions taken

Local commits were created for the Person 5 work. No push, pull request, deployment,
database change, Redis mutation, or Qdrant mutation was performed.
