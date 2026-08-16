from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.route2_release.verify_hybrid_smd_development_v2 import (
    HybridDevelopmentVerificationError,
    _record_identity,
    audit_hybrid_development,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path, *, count: int = 3) -> dict[str, Path | int]:
    runner = tmp_path / "runner.py"
    aggregator = tmp_path / "aggregator.py"
    mdp = tmp_path / "mdp.model"
    polar = tmp_path / "polar.model"
    source_root = tmp_path / "source"
    source = source_root / "maple/example.py"
    for path, content in (
        (runner, "runner-v2\n"),
        (aggregator, "aggregator-v2\n"),
        (mdp, "mdp-checkpoint\n"),
        (polar, "polar-checkpoint\n"),
        (source, "source-v2\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    preregistration = tmp_path / "preregistration.json"
    _write(
        preregistration,
        {
            "artifact_id": "route2-hybrid-smd-development-prereg-v2",
            "status": "locked-before-first-v2-hybrid-evaluation",
            "partition": "development",
            "confirmation_partition_opened": False,
            "fitting_or_calibration_permitted": False,
            "record_count": count,
            "hard_accuracy_target": {
                "comparison": "<=",
                "metric": "mean_absolute_error_kcal_mol",
                "threshold_kcal_mol": 1.5,
            },
            "runner_sha256": _sha256(runner),
            "aggregator_sha256": _sha256(aggregator),
            "mdp_checkpoint_sha256": _sha256(mdp),
            "polar_checkpoint_sha256": _sha256(polar),
            "source_files_sha256": {"maple/example.py": _sha256(source)},
        },
    )
    records = tmp_path / "records"
    records.mkdir()
    preregistration_sha256 = _sha256(preregistration)
    runner_sha256 = _sha256(runner)
    mdp_sha256 = _sha256(mdp)
    polar_sha256 = _sha256(polar)
    for index in range(count):
        experimental = -1.5
        predicted = -1.4 - 0.1 * index
        signed = predicted - experimental
        _write(
            records / f"index-{index:03d}.json",
            {
                "artifact": "route2-hybrid-smd-development-record-v2",
                "do_not_commit": True,
                "partition": "development",
                "confirmation_partition_opened": False,
                "selection_index": index,
                "opaque_record_id": hashlib.sha256(
                    f"opaque-{index}".encode()
                ).hexdigest(),
                "geometry_sha256": hashlib.sha256(
                    f"geometry-{index}".encode()
                ).hexdigest(),
                "record_identity_sha256": _record_identity(
                    preregistration_sha256=preregistration_sha256,
                    runner_sha256=runner_sha256,
                    mdp_checkpoint_sha256=mdp_sha256,
                    polar_checkpoint_sha256=polar_sha256,
                    selection_index=index,
                ),
                "preregistration_sha256": preregistration_sha256,
                "runner_sha256": runner_sha256,
                "mdp_checkpoint_sha256": mdp_sha256,
                "polar_checkpoint_sha256": polar_sha256,
                "hybrid_configuration_sha256": "1" * 64,
                "pes_configuration_sha256": hashlib.sha256(
                    f"pes-{index}".encode()
                ).hexdigest(),
                "state_sha256": hashlib.sha256(f"state-{index}".encode()).hexdigest(),
                "electrostatic_root_sha256": hashlib.sha256(
                    f"root-{index}".encode()
                ).hexdigest(),
                "continuum_state_sha256": hashlib.sha256(
                    f"continuum-{index}".encode()
                ).hexdigest(),
                "canonical_solvent": "water" if index < 2 else "toluene",
                "status": "pass",
                "predicted_delta_g_kcal_mol": predicted,
                "experimental_delta_g_kcal_mol": experimental,
                "signed_error_kcal_mol": signed,
                "absolute_error_kcal_mol": abs(signed),
                "root_residual_eV": 1.0e-12,
                "vacuum_energy_eV": -10.0,
                "continuum_polarization_kcal_mol": -2.0,
                "smd_cds_kcal_mol": 0.5,
                "solve_wall_seconds": 1.0,
            },
        )
    return {
        "input_dir": records,
        "preregistration_path": preregistration,
        "runner_path": runner,
        "aggregator_path": aggregator,
        "source_root": source_root,
        "mdp_checkpoint_path": mdp,
        "polar_checkpoint_path": polar,
        "expected_count": count,
    }


def test_integrity_audit_recomputes_complete_metrics_and_bindings(tmp_path: Path):
    paths = _fixture(tmp_path)
    result = audit_hybrid_development(**paths)
    assert result["status"] == "pass"
    assert result["record_count"] == 3
    assert result["failure_count"] == 0
    assert result["hard_accuracy_target"]["passed"] is True
    assert result["aggregate_metrics"]["mean_absolute_error_kcal_mol"] == (
        pytest.approx(0.06666666666666672)
    )
    assert set(result["per_solvent_metrics"]) == {"toluene", "water"}
    assert len(result["verification_sha256"]) == 64


def test_integrity_audit_fails_closed_on_incomplete_or_tampered_records(
    tmp_path: Path,
):
    paths = _fixture(tmp_path)
    missing = paths["input_dir"] / "index-002.json"
    missing.unlink()
    with pytest.raises(HybridDevelopmentVerificationError, match="incomplete"):
        audit_hybrid_development(**paths)
    partial = audit_hybrid_development(**paths, allow_incomplete=True)
    assert partial["status"] == "incomplete"
    assert partial["hard_accuracy_target"]["passed"] is None

    paths = _fixture(tmp_path / "tampered")
    record = paths["input_dir"] / "index-001.json"
    payload = json.loads(record.read_text())
    payload["absolute_error_kcal_mol"] += 0.25
    _write(record, payload)
    with pytest.raises(HybridDevelopmentVerificationError, match="ledger drifted"):
        audit_hybrid_development(**paths)


def test_integrity_audit_rejects_frozen_source_drift(tmp_path: Path):
    paths = _fixture(tmp_path)
    (paths["source_root"] / "maple/example.py").write_text("changed\n")
    with pytest.raises(HybridDevelopmentVerificationError, match="source file drifted"):
        audit_hybrid_development(**paths)
