# Offline Pipeline Operations Runbook

This runbook covers the two Person 1 worker processes:

1. repository corpus acquisition, enrichment, and backend v2 delivery;
2. enriched GitHub Trending snapshot delivery to backend v2.

Both are offline jobs. Do not invoke either command from an API request handler.

## Safety contract

- The backend is the production source of truth and creates canonical repository UUIDs.
- Workers write neither PostgreSQL nor Qdrant directly.
- Backend repository mappings are validated before a source is counted as delivered.
- Trending snapshots are complete and atomically activated by the backend.
- Repository and user embeddings must use the model and dimension published by the embedding owner.
- Checkpoints contain repository identities and bounded error messages, not tokens or README content.

## Configuration

Copy `.env.example` to `.env` and replace every placeholder used by the worker.

Required for production corpus acquisition:

- `GITHUB_TOKEN`
- `BACKEND_URL`
- `INTERNAL_API_SECRET`
- `CORPUS_TARGET_COUNT` (default `50000`)
- `ACQUISITION_MAX_CYCLES` (default `1`)
- `ACQUISITION_CHECKPOINT_PATH`

`OPENROUTER_API_KEY` is optional. Without it, README Markdown restructuring is skipped while normal README processing continues.

Trending uses the same backend credentials and enriches scraped repositories before publishing a snapshot.

## Network-free validation

These commands validate parsing and required configuration without contacting GitHub, Postgres, Qdrant, or OpenRouter:

```bash
python3 main.py --validate-config
python3 trending_service.py --validate-config
```

If Qdrant is intentionally unavailable during a Postgres-only maintenance window:

```bash
python3 main.py --validate-config --no-index-qdrant
```

Validation success does not prove that credentials are accepted or that a remote service is healthy. It proves the local worker configuration is internally valid.

## Corpus acquisition

Run one bounded production cycle:

```bash
python3 main.py \
  --limit 150 \
  --batch-size 15 \
  --workers 4 \
  --min-readme-chars 200 \
  --max-cycles 1
```

The target defaults to `CORPUS_TARGET_COUNT`. `--limit` bounds one discovery cycle; it does not request the full corpus target at once.

Embedding and Qdrant indexing are scheduled by the backend outbox after successful repository delivery.

### Stop and resume

- Stop the foreground process with `Ctrl+C` or the platform's normal `SIGTERM` shutdown.
- Do not delete the checkpoint after an interrupted or degraded run.
- Restart with the same configuration and checkpoint path.
- Pending Qdrant indexing is attempted before new discovery.
- Pending persistence is re-enriched from its repository identity before retry.

The default checkpoint is `.cache/acquisition_checkpoint.json`. Writes use atomic replacement. Do not edit the file while a worker is running.

### Corpus exit codes

- `0`: the bounded invocation completed, including a no-work target-reached run.
- `1`: configuration, startup, Postgres verification, or pipeline execution failed.
- `2`: failures were recorded and the run made no persistence or indexing progress.

Inspect the final `Corpus run report` log and the checkpoint's `last_run`, `failures`, `pending_persistence`, and `pending_index` fields before retrying an exit code `2`.

## Trending refresh

Run one forced refresh and synchronize existing Qdrant points:

```bash
python3 trending_service.py --once
```

Run the long-lived scheduler:

```bash
python3 trending_service.py --scheduled
```

The scheduler handles `SIGINT` and `SIGTERM`. A normal stop clears its in-process schedule and exits the loop.

A partial Postgres trending write prevents Qdrant synchronization. Missing Qdrant points are logged and skipped. Per-point payload failures are isolated, reported, and cause the refresh command to return failure so operations can investigate.

## Diagnosis

| Symptom | Meaning | Safe action |
|---|---|---|
| `GITHUB_TOKEN` configuration error | Token is absent or still a placeholder | Set a real token and rerun network-free validation |
| Postgres verification failure | Production persistence cannot be trusted | Correct `DATABASE_URL`; do not use the development bypass |
| No acquisition progress | Discovery returned nothing new, filtering rejected everything, or the database count did not advance | Inspect rejection and failure records before increasing limits |
| `pending_persistence` is non-empty | Repositories still require enrichment/Postgres persistence | Rerun the same bounded corpus command |
| `pending_index` is non-empty | Postgres succeeded but Qdrant indexing did not | Restore Qdrant, then rerun without `--no-index-qdrant` |
| Trending repository is missing in Qdrant | The repo has no existing corpus point | Let normal corpus acquisition approve and index it; do not create a vector from trending HTML |
| Unsupported embedding model | The embedding owner has not published a compatible model/dimension contract | Keep the current model until the shared embedding interface is updated |

## Verification before handoff

Run the deterministic suite:

```bash
pytest -q
```

Tests mock external services. A default test run must not contact GitHub, Postgres, Qdrant, or OpenRouter.
