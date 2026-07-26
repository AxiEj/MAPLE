from __future__ import annotations

import json

import pytest

from maple.function.dispatcher.solvfe.artifacts import (
    ArtifactStore,
    ConcurrentArtifactWrite,
    RestartHashMismatch,
    canonical_sha256,
)


def test_artifact_store_writes_atomically_and_resumes_exact_hash(tmp_path):
    store = ArtifactStore(tmp_path / "run.solvfe", run_hash="a" * 64)
    manifest = {"schema_version": 1, "run_hash": "a" * 64}
    store.initialize(manifest)

    output = {"stage": "PRECHECK", "status": "completed"}
    output_hash = store.complete_stage(
        "PRECHECK",
        input_hash="b" * 64,
        output=output,
    )

    assert output_hash == canonical_sha256(output)
    assert json.loads((store.root / "manifest.json").read_text()) == manifest
    assert store.resume_stage("PRECHECK", input_hash="b" * 64) == output
    ledger = json.loads((store.root / "stage-ledger.json").read_text())
    assert ledger["schema_version"] == 2
    assert ledger["generation"] == 1
    assert ledger["stages"]["PRECHECK"]["generation"] == 1
    assert len(ledger["stages"]["PRECHECK"]["attempt_id"]) == 32

    with pytest.raises(RestartHashMismatch, match="RESTART_HASH_MISMATCH"):
        store.resume_stage("PRECHECK", input_hash="c" * 64)


def test_artifact_store_rejects_existing_run_with_different_manifest(tmp_path):
    root = tmp_path / "run.solvfe"
    ArtifactStore(root, run_hash="a" * 64).initialize(
        {"schema_version": 1, "run_hash": "a" * 64}
    )
    with pytest.raises(RestartHashMismatch, match="RESTART_HASH_MISMATCH"):
        ArtifactStore(root, run_hash="b" * 64).initialize(
            {"schema_version": 1, "run_hash": "b" * 64}
        )


def test_artifact_store_rejects_a_second_writer_for_same_run(tmp_path):
    root = tmp_path / "run.solvfe"
    first = ArtifactStore(root, run_hash="a" * 64)
    second = ArtifactStore(root, run_hash="a" * 64)

    with first.exclusive_session():
        first.initialize({"schema_version": 1, "run_hash": "a" * 64})
        with pytest.raises(
            ConcurrentArtifactWrite,
            match="ARTIFACT_WRITER_ACTIVE",
        ):
            with second.exclusive_session():
                pass


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "failed"),
        ("generation", 999999),
        ("attempt_id", "not-a-valid-attempt"),
        ("output_path", "../outside.json"),
    ],
)
def test_artifact_store_rejects_tampered_stage_audit_record(
    tmp_path,
    field,
    value,
):
    store = ArtifactStore(tmp_path / "run.solvfe", run_hash="a" * 64)
    store.initialize({"schema_version": 1, "run_hash": "a" * 64})
    store.complete_stage(
        "PRECHECK",
        input_hash="b" * 64,
        output={"status": "ok"},
    )
    ledger_path = store.root / "stage-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["stages"]["PRECHECK"][field] = value
    ledger_path.write_text(json.dumps(ledger))

    with pytest.raises(RestartHashMismatch, match="RESTART_HASH_MISMATCH"):
        store.resume_stage("PRECHECK", input_hash="b" * 64)


def test_artifact_store_rejects_inconsistent_top_level_generation(tmp_path):
    store = ArtifactStore(tmp_path / "run.solvfe", run_hash="a" * 64)
    store.initialize({"schema_version": 1, "run_hash": "a" * 64})
    store.complete_stage(
        "PRECHECK",
        input_hash="b" * 64,
        output={"status": "ok"},
    )
    ledger_path = store.root / "stage-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["generation"] = 2
    ledger_path.write_text(json.dumps(ledger))

    with pytest.raises(RestartHashMismatch, match="RESTART_HASH_MISMATCH"):
        store.resume_stage("PRECHECK", input_hash="b" * 64)


@pytest.mark.parametrize("stage", ["../escape", "stage/name", ""])
def test_artifact_store_rejects_unsafe_stage_names(tmp_path, stage):
    store = ArtifactStore(tmp_path / "run.solvfe", run_hash="a" * 64)
    store.initialize({"schema_version": 1, "run_hash": "a" * 64})

    with pytest.raises(ValueError, match="stage names"):
        store.complete_stage(
            stage,
            input_hash="b" * 64,
            output={"status": "ok"},
        )
