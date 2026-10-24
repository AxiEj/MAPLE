from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs" / "implicit-solvation" / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import validate_independent_obc2_default as validation


def test_validation_output_uses_the_current_run_date_without_rewriting_stage0():
    assert validation.RECORDED_DATE == "2026-10-24"
    assert validation.OUTPUT_REL.endswith(
        "candidate-validation.json"
    )
    assert validation.STAGE0_MAIN_REL.endswith(
        "route1-obc2-am1bcc-526-reference-2026-09-27.json"
    )


def test_sealed_stage0_inputs_are_exact_and_complete():
    main, portable = validation.load_stage0()

    assert main["content_sha256"] == validation.STAGE0_MAIN_CONTENT_SHA256
    assert portable["content_sha256"] == validation.STAGE0_PORTABLE_CONTENT_SHA256
    assert main["record_count"] == 1052
    assert portable["comparison_count"] == 48
    assert main["semantic_decoder_guard"]["runtime_violation_count"] == 0


def test_full_h_selection_was_frozen_before_provider_calculations():
    _main, portable = validation.load_stage0()
    selection = validation.select_full_h_cases(portable)

    assert selection["selected_cases"] == [
        "water",
        "methanol",
        "methyl-hexanoate",
        "nitralin",
    ]
    assert selection["selected_largest_case"] == "nitralin"
    assert selection["maximum_atom_count"] == 42
    assert selection["rule"] == validation.FULL_H_SELECTION_RULE


def test_hvp_directions_are_seeded_normalized_and_replay_exactly():
    first_seed, first = validation.normalized_direction(
        case_index=4,
        geometry_index=1,
        direction_index=2,
        size=69,
    )
    second_seed, second = validation.normalized_direction(
        case_index=4,
        geometry_index=1,
        direction_index=2,
        size=69,
    )

    assert first_seed == second_seed
    np.testing.assert_array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0, rtol=0.0, atol=1.0e-15)


def test_fd_ladder_uses_approved_d64_steps_and_both_finest_pair_gate():
    analytic = np.zeros((3, 3), dtype=np.float64)
    matrices = {
        0.001: np.full((3, 3), 4.0e-6),
        0.0005: np.full((3, 3), 2.0e-6),
        0.00025: np.full((3, 3), 1.0e-6),
    }
    result = validation.assess_fd_ladder(analytic, matrices)

    assert result["step_order"] == [0.001, 0.0005, 0.00025]
    assert result["all_three_pass_observed"] is True
    assert result["both_finest_pass"] is True
    assert result["finest_pair_pass"] is True
    assert result["passed"] is True


def test_fd_ladder_logs_coarse_without_turning_it_into_an_extra_gate():
    analytic = np.zeros((3, 3), dtype=np.float64)
    matrices = {
        0.001: np.full((3, 3), 3.0e-5),
        0.0005: np.full((3, 3), 2.0e-6),
        0.00025: np.full((3, 3), 1.0e-6),
    }
    result = validation.assess_fd_ladder(analytic, matrices)

    assert result["both_finest_pass"] is True
    assert result["coarse_pass_observed"] is False
    assert result["all_three_pass_observed"] is False
    assert result["passed"] is True


def test_fd_ladder_rejects_when_either_predeclared_finest_step_fails():
    analytic = np.zeros((3, 3), dtype=np.float64)
    matrices = {
        0.001: np.zeros((3, 3)),
        0.0005: np.full((3, 3), 3.0e-5),
        0.00025: np.full((3, 3), 1.0e-6),
    }
    result = validation.assess_fd_ladder(analytic, matrices)

    assert result["both_finest_pass"] is False
    assert result["passed"] is False


def _seal(value):
    payload = dict(value)
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _candidate_record(nonpolar):
    return {
        "nonpolar": nonpolar,
        "same_scalar_energy_abs_hartree": 0.0,
        "same_scalar_force_max_hartree_per_angstrom": 0.0,
        "hessian_asymmetry_max_hartree_per_angstrom2": 0.0,
        "hessian_composition_max_hartree_per_angstrom2": 0.0,
        "hessian_force_fd_ladder": [
            {
                "step_angstrom": step,
                "rms_hartree_per_angstrom2": 0.0,
                "max_hartree_per_angstrom2": 0.0,
                "passed": True,
            }
            for step in (1.0e-6, 2.0e-6, 5.0e-6)
        ],
        "direct_hvp_dense_max_hartree_per_angstrom2": 0.0,
        "direct_hvp_dense_hessian_forbidden": True,
        "gate_identity": {
            "model": "ani2x",
            "device": "cpu",
            "d4": False,
            "checkpoint_sha256": (
                "c8c41e89d5ccbe7b5c9f46ce4a262b913ad0c428a3a3742cc6e83ac10d4d43aa"
            ),
            "dtype": "float64",
            "provider": {
                "runtime_provider": "torch-obc2",
                "model": "obc2",
                "profile": "obc2-mbondi2",
                "nonpolar": nonpolar,
            },
        },
        "passed": True,
    }


def _candidate_evidence(**overrides):
    source_files = {"README.md": "b" * 64}
    evidence = {
        "artifact_type": "independent-obc2-candidate-exact-cell-v1",
        "qualification_stage": "candidate qualification, not installed qualification",
        "nonpolar": ["ace", "none"],
        "expected_count": 2,
        "passed_count": 2,
        "passed": True,
        "checkpoint_sha256": (
            "c8c41e89d5ccbe7b5c9f46ce4a262b913ad0c428a3a3742cc6e83ac10d4d43aa"
        ),
        "model": "ani2x",
        "device": "cpu",
        "dtype": "float64",
        "d4": False,
        "candidate_git_sha": "a" * 40,
        "harness_sha256": validation.core.sha256_file(
            validation.ROOT / "tests/solvation/run_independent_obc2_candidate_cell.py"
        ),
        "candidate_source_snapshot": {
            "git_head": "a" * 40,
            "files_sha256": source_files,
            "files_aggregate_sha256": validation.core.sha256_bytes(
                validation.core.canonical_json_bytes(source_files)
            ),
            "worktree_status": [],
        },
        "records": [_candidate_record("ace"), _candidate_record("none")],
    }
    evidence.update(overrides)
    return _seal(evidence)


def test_candidate_cell_evidence_requires_both_frozen_nonpolar_cells():
    evidence = _candidate_evidence()

    validation.validate_candidate_cell_evidence(evidence)


def test_candidate_cell_evidence_rejects_self_resealed_empty_records():
    evidence = _candidate_evidence(records=[])

    with pytest.raises(ValueError, match="records"):
        validation.validate_candidate_cell_evidence(evidence)


def test_candidate_cell_evidence_rejects_self_resealed_stale_harness():
    evidence = _candidate_evidence(harness_sha256="0" * 64)

    with pytest.raises(ValueError, match="harness SHA"):
        validation.validate_candidate_cell_evidence(evidence)


def test_candidate_cell_evidence_rejects_self_resealed_failed_record_gate():
    records = [_candidate_record("ace"), _candidate_record("none")]
    records[0]["hessian_composition_max_hartree_per_angstrom2"] = 2.0e-12
    evidence = _candidate_evidence(records=records)

    with pytest.raises(ValueError, match="hessian_composition"):
        validation.validate_candidate_cell_evidence(evidence)


def test_candidate_cell_evidence_rejects_mismatched_qualification_source():
    evidence = _candidate_evidence()
    expected = dict(evidence["candidate_source_snapshot"])
    expected["files_sha256"] = {"different.py": "d" * 64}
    expected["files_aggregate_sha256"] = validation.core.sha256_bytes(
        validation.core.canonical_json_bytes(expected["files_sha256"])
    )

    with pytest.raises(ValueError, match="source inventory"):
        validation.validate_candidate_cell_evidence(
            evidence,
            expected_source_snapshot=expected,
        )


def test_qualification_source_inventory_covers_runtime_data_and_contract_tests():
    files = validation.source_snapshot(require_clean=False)["files_sha256"]

    required = {
        "maple/function/calculator/extra_correction/implicit/data/obc2_profile_manifest.json",
        "maple/function/calculator/extra_correction/implicit/data/OPENMM_CUSTOMGBFORCES_LICENSE.txt",
        "tests/solvation/test_command_contract.py",
        "tests/solvation/test_direct_solvent_contract.py",
        "tests/solvation/test_implicit_ts.py",
        "tests/test_prfo_saddle_convergence.py",
        "docs/implicit-solvation/INDEPENDENT_OBC2.md",
    }
    assert required <= set(files)
    assert validation.CANDIDATE_CELL_HARNESS_REL in files


def test_qualification_source_inventory_excludes_only_final_data_artifacts():
    snapshot = validation.source_snapshot(require_clean=False)

    assert set(snapshot["excluded_data_only_paths"]) == set(
        validation.QUALIFICATION_SOURCE_EXCLUDES
    )
    assert not set(snapshot["files_sha256"]) & set(
        validation.QUALIFICATION_SOURCE_EXCLUDES
    )


def test_qualification_source_snapshot_rejects_dirty_tree(monkeypatch):
    monkeypatch.setattr(
        validation,
        "_git",
        lambda *arguments: " M maple/changed.py"
        if arguments[:2] == ("status", "--porcelain=v1")
        else "a" * 40,
    )

    with pytest.raises(RuntimeError, match="clean committed tree"):
        validation.source_snapshot(require_clean=True)


def test_candidate_cell_evidence_rejects_installed_qualification_claim():
    evidence = _candidate_evidence(
        qualification_stage="installed qualification",
    )

    with pytest.raises(ValueError, match="overstates installed qualification"):
        validation.validate_candidate_cell_evidence(evidence)


def test_count_gate_keeps_failures_in_the_denominator():
    gate = validation._count_gate(
        {"passed": False, "records": [{"passed": True}, {"passed": False}]},
        2,
    )

    assert gate == {"expected_count": 2, "passed_count": 1, "passed": False}


def test_fresh_validation_artifact_is_self_consistent_when_sealed():
    path = validation.ROOT / validation.OUTPUT_REL
    if not path.is_file():
        pytest.skip(
            "final validation waits for a stable independent provider source tree"
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))

    validation.validate_artifact(artifact)
    assert artifact["passed"] is True
    assert artifact["replay_526"]["mae_drift_upper_bound_hartree"] == pytest.approx(
        artifact["replay_526"]["mean_absolute_prediction_drift_hartree"],
        rel=0.0,
        abs=0.0,
    )
    assert artifact["direct_hvp_matrix"]["record_count"] == 144
    assert all(
        row["dense_hessian_forbidden_during_call"]
        for row in artifact["direct_hvp_matrix"]["records"]
    )


@pytest.fixture
def qualification_repo(tmp_path, monkeypatch):
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Qualification Test")
    (tmp_path / "runtime.py").write_text("frozen source\n")
    git("add", ".")
    git("commit", "-qm", "candidate")
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    return tmp_path, git, validation.source_snapshot()


def test_current_binding_allows_data_only_descendant_without_circular_head(qualification_repo):
    root, git, frozen = qualification_repo
    for relative in validation.QUALIFICATION_SOURCE_EXCLUDES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    git("add", ".")
    git("commit", "-qm", "seal data only")
    assert git("rev-parse", "HEAD") != frozen["git_head"]
    validation.validate_current_source_binding(frozen)


def test_current_binding_rejects_changed_source_inventory(qualification_repo):
    root, git, frozen = qualification_repo
    (root / "runtime.py").write_text("changed source\n")
    git("add", ".")
    git("commit", "-qm", "source changed")
    with pytest.raises(ValueError, match="source inventory"):
        validation.validate_current_source_binding(frozen)


def test_current_binding_rejects_nonancestor_even_with_identical_files(qualification_repo):
    root, git, frozen = qualification_repo
    git("checkout", "--orphan", "unrelated")
    git("commit", "-qm", "unrelated identical files")
    with pytest.raises(ValueError, match="ancestor"):
        validation.validate_current_source_binding(frozen)


def test_current_binding_rejects_source_changed_then_reverted(qualification_repo):
    root, git, frozen = qualification_repo
    (root / "runtime.py").write_text("temporary change\n")
    git("add", ".")
    git("commit", "-qm", "temporary source change")
    git("revert", "--no-edit", "HEAD")
    with pytest.raises(ValueError, match="data-only"):
        validation.validate_current_source_binding(frozen)


def test_current_binding_rejects_dirty_data_pointer(qualification_repo):
    root, _git, frozen = qualification_repo
    path = root / next(iter(validation.QUALIFICATION_SOURCE_EXCLUDES))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    with pytest.raises(RuntimeError, match="clean committed tree"):
        validation.validate_current_source_binding(frozen)
