from __future__ import annotations

import json

import pytest

from maple.function.dispatcher.solvfe.artifacts import (
    ArtifactStore,
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

