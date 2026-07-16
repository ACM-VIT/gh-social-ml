"""Failure-path tests for production corpus acquisition orchestration."""

from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from acquisition.checkpoint import CorpusCheckpoint
from acquisition.config import CorpusPipelineSettings
from acquisition.corpus_pipeline import CorpusPipeline
from acquisition.identity import deduplicate_candidates, normalize_repository_name
from acquisition.models import AcquisitionFailure, AcquisitionRunResult
from acquisition.pipeline import enrich_repositories
from acquisition.repository_enricher import RepositoryEnricher
from database.connector import RepositoryUpsertResult
from main import _parse_args


def _source(name: str):
    return SimpleNamespace(repo_id=name, payload={"id": name})


class _Database:
    enabled = True

    def __init__(self, outcome: RepositoryUpsertResult | None = None) -> None:
        self.count = 0
        self.outcome = outcome or RepositoryUpsertResult()

    def verify_connection(self) -> bool:
        return True

    def init_db(self) -> None:
        pass

    def get_repo_count(self) -> int:
        return self.count

    def get_existing_repository_names(self) -> set[str]:
        return set()

    def upsert_repositories_detailed(self, sources):
        self.count += len(self.outcome.succeeded)
        return self.outcome

    def get_repositories_by_full_names(self, names):
        return [{"id": name, "full_name": name} for name in names]


def _settings(
    path: Path, *, target: int = 50_000, max_cycles: int = 1
) -> CorpusPipelineSettings:
    return CorpusPipelineSettings(
        target_count=target,
        max_cycles=max_cycles,
        checkpoint_path=path,
    )


class _SuccessfulDatabase(_Database):
    def upsert_repositories_detailed(self, sources):
        names = [source.repo_id for source in sources]
        self.count += len(names)
        return RepositoryUpsertResult(succeeded=names)


def test_identity_normalization_and_case_insensitive_deduplication():
    assert (
        normalize_repository_name(" https://github.com/OpenAI/Codex/ ")
        == "OpenAI/Codex"
    )
    unique, removed = deduplicate_candidates(
        ["OpenAI/Codex", {"full_name": "openai/codex"}, "invalid"]
    )
    assert unique == ["OpenAI/Codex"]
    assert removed == 2


def test_transient_readme_failure_is_reported_for_retry(monkeypatch):
    def return_warning(_self, batch):
        name = batch[0]
        return [SimpleNamespace(repo_id=name, warnings=["README fetch failed"])]

    monkeypatch.setattr(RepositoryEnricher, "get_repositories_batch", return_warning)
    result = enrich_repositories("test-token", ["owner/repo"], batch_size=1, workers=1)

    assert result.repositories == []
    assert len(result.failures) == 1
    assert result.failures[0].stage == "readme_enrichment"


def test_concurrent_enrichment_preserves_discovery_order(monkeypatch):
    second_batch_completed = threading.Event()

    def finish_second_batch_first(_self, batch):
        name = batch[0]
        if name == "owner/first":
            assert second_batch_completed.wait(timeout=1)
        else:
            second_batch_completed.set()
        return [SimpleNamespace(repo_id=name, warnings=[])]

    monkeypatch.setattr(
        RepositoryEnricher,
        "get_repositories_batch",
        finish_second_batch_first,
    )
    result = enrich_repositories(
        "test-token",
        ["owner/first", "owner/second"],
        batch_size=1,
        workers=2,
    )

    assert [repository.repo_id for repository in result.repositories] == [
        "owner/first",
        "owner/second",
    ]


def test_pipeline_indexes_only_successfully_persisted_subset(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    database = _Database(
        RepositoryUpsertResult(
            succeeded=["owner/good"],
            failed={"owner/bad": "constraint failure"},
        )
    )
    indexed = []
    pipeline = CorpusPipeline(
        database=database,
        acquire=lambda **_: AcquisitionRunResult(
            repositories=[_source("owner/good"), _source("owner/bad")],
            discovered_count=2,
        ),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: indexed.extend(values) or values,
        settings=_settings(path),
        checkpoint=checkpoint,
    )

    report = pipeline.run(limit=2, batch_size=1, workers=1, min_readme_chars=1)

    assert [item.repo_id for item in indexed] == ["owner/good"]
    assert report.persisted == 1
    assert report.indexed == 1
    assert checkpoint.pending_persistence == ["owner/bad"]
    assert checkpoint.pending_index == []


def test_enrichment_failure_is_checkpointed_for_identity_only_retry(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    pipeline = CorpusPipeline(
        database=_Database(),
        acquire=lambda **_: AcquisitionRunResult(
            failures=[
                AcquisitionFailure(
                    "owner/retry", "readme_enrichment", "temporary timeout"
                )
            ],
            discovered_count=1,
        ),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(path),
        checkpoint=checkpoint,
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert checkpoint.pending_persistence == ["owner/retry"]
    assert "owner/retry" in report.failures
    assert "temporary timeout" in path.read_text(encoding="utf-8")


def test_rejected_repository_never_reaches_postgres_or_qdrant(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _Database()
    database.upsert_repositories_detailed = lambda _: pytest.fail(
        "rejected repository reached Postgres"
    )
    indexed = []
    rejected_source = _source("owner/rejected")
    pipeline = CorpusPipeline(
        database=database,
        acquire=lambda **_: AcquisitionRunResult(repositories=[rejected_source]),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (
            [],
            [(values[0], ["no README", "shell repository"])],
        ),
        indexer=lambda values: indexed.extend(values) or values,
        settings=_settings(path),
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert report.rejected == 1
    assert report.persisted == 0
    assert indexed == []
    checkpoint = CorpusCheckpoint(path)
    assert checkpoint.data["rejected"]["owner/rejected"]["reasons"] == [
        "no README",
        "shell repository",
    ]


def test_zero_database_progress_stops_additional_cycles(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _Database(RepositoryUpsertResult(succeeded=["owner/repo"]))
    database.upsert_repositories_detailed = lambda _: database.outcome
    acquisition_calls = 0

    def acquire(**_):
        nonlocal acquisition_calls
        acquisition_calls += 1
        return AcquisitionRunResult(repositories=[_source("owner/repo")])

    pipeline = CorpusPipeline(
        database=database,
        acquire=acquire,
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(path, target=10, max_cycles=5),
    )

    pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert acquisition_calls == 1


def test_max_cycles_bounds_successful_acquisition(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _SuccessfulDatabase()
    acquisition_calls = 0

    def acquire(**_):
        nonlocal acquisition_calls
        acquisition_calls += 1
        return AcquisitionRunResult(
            repositories=[_source(f"owner/repo-{acquisition_calls}")]
        )

    pipeline = CorpusPipeline(
        database=database,
        acquire=acquire,
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(path, target=10, max_cycles=2),
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert acquisition_calls == 2
    assert report.persisted == 2
    assert report.indexed == 2


def test_batch_index_failure_isolates_repo_and_resume_clears_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _SuccessfulDatabase()
    sources = [_source("owner/good"), _source("owner/retry")]
    indexing_calls = []

    def partially_failing_indexer(values):
        names = [value.repo_id for value in values]
        indexing_calls.append(names)
        if len(values) > 1:
            raise RuntimeError("batch unavailable")
        if names == ["owner/retry"]:
            raise RuntimeError("point unavailable")
        return values

    first_run = CorpusPipeline(
        database=database,
        acquire=lambda **_: AcquisitionRunResult(repositories=sources),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=partially_failing_indexer,
        settings=_settings(path, target=2),
    )
    first_report = first_run.run(limit=2, batch_size=2, workers=1, min_readme_chars=1)

    assert indexing_calls == [
        ["owner/good", "owner/retry"],
        ["owner/good"],
        ["owner/retry"],
    ]
    assert first_report.indexed == 1
    assert CorpusCheckpoint(path).pending_index == ["owner/retry"]

    resumed = []
    second_run = CorpusPipeline(
        database=database,
        acquire=lambda **_: pytest.fail("resume should happen before discovery"),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: resumed.extend(values) or values,
        settings=_settings(path, target=2),
    )
    second_report = second_run.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert resumed == [{"id": "owner/retry", "full_name": "owner/retry"}]
    assert second_report.resumed_indexing == 1
    assert CorpusCheckpoint(path).pending_index == []


def test_pending_index_is_resumed_before_new_discovery(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.add_pending_index(["owner/repo"])
    checkpoint.save()
    database = _Database()
    database.count = 1
    indexed = []
    pipeline = CorpusPipeline(
        database=database,
        acquire=lambda **_: pytest.fail("discovery should not run after target"),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: indexed.extend(values) or values,
        settings=_settings(path, target=1),
        checkpoint=CorpusCheckpoint(path),
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert indexed == [{"id": "owner/repo", "full_name": "owner/repo"}]
    assert report.resumed_indexing == 1
    assert pipeline.checkpoint.pending_index == []


def test_pipeline_refuses_implicit_qdrant_only_production_run(tmp_path):
    pipeline = CorpusPipeline(
        database=SimpleNamespace(enabled=False),
        acquire=lambda **_: [],
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(tmp_path / "checkpoint.json"),
    )

    with pytest.raises(RuntimeError, match="requires a verified Postgres"):
        pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)


def test_checkpoint_round_trip_is_atomic_and_contains_only_retry_state(tmp_path):
    path = tmp_path / "nested" / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.add_pending_persistence(["Owner/Repo", "owner/repo"])
    checkpoint.record_failure("Owner/Repo", "persistence", "failed")
    checkpoint.save()

    loaded = CorpusCheckpoint(path)
    assert loaded.pending_persistence == ["Owner/Repo"]
    assert "readme" not in path.read_text(encoding="utf-8").lower()


def test_invalid_environment_and_cli_values_fail_before_pipeline(monkeypatch):
    monkeypatch.setenv("CORPUS_TARGET_COUNT", "0")
    with pytest.raises(ValueError, match="CORPUS_TARGET_COUNT"):
        CorpusPipelineSettings.from_environment()

    monkeypatch.setenv("CORPUS_TARGET_COUNT", "50000")
    with pytest.raises(SystemExit):
        _parse_args(["--workers", "0"])
    with pytest.raises(SystemExit):
        _parse_args(["--min-readme-chars", "-1"])


def test_settings_validation_returns_runtime_safe_integer_values(tmp_path):
    settings = CorpusPipelineSettings(
        target_count="5",  # type: ignore[arg-type]
        max_cycles="2",  # type: ignore[arg-type]
        checkpoint_path=tmp_path / "checkpoint.json",
    ).validated()

    assert settings.target_count == 5
    assert settings.max_cycles == 2
