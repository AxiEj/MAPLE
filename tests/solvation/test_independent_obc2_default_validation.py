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



def test_current_binding_accepts_unchanged_candidate_head(qualification_repo):
    _root, _git, frozen = qualification_repo
    validation.validate_current_source_binding(frozen)


def _summary_only_artifact():
    candidate = _candidate_evidence()
    counts = {
        "sealed_oracle_1052_energy": 1052,
        "portable_oracle_48_energy_force": 48,
        "analytic_hessian_16_fd": 16,
        "direct_hvp_144": 144,
        "no_openmm_candidate_runtime": 1,
        "candidate_exact_cell_public_composition": 2,
        "source_unchanged": 1,
    }
    gates = {key: {"expected_count": count, "passed_count": count, "passed": True}
             for key, count in counts.items()}
    gates["analytic_hessian_16_fd"]["steps_angstrom"] = list(validation.SOLVENT_FD_STEPS_ANGSTROM)
    gates["direct_hvp_144"]["dense_hessian_forbidden"] = True
    gates["candidate_exact_cell_public_composition"].update(
        nonpolar=["ace", "none"], evidence_content_sha256=candidate["content_sha256"])
    return {
        "schema_version": 1,
        "artifact_type": "route1-independent-obc2-validation-v1",
        "recorded_date": validation.RECORDED_DATE,
        "source_unchanged": True,
        "source_snapshot": candidate["candidate_source_snapshot"],
        "candidate_exact_cell": candidate,
        "selection": {"selected_cases": list(validation.FULL_H_CASES)},
        "guarded_no_openmm": {"passed": True},
        "replay_526": {"record_count": 1052, "passed": True},
        "portable_48_ef_radius": {"comparison_count": 48, "passed": True},
        "full_hessian_matrix": {"record_count": 16, "passed": True},
        "direct_hvp_matrix": {"record_count": 144, "passed": True},
        "qualification_gates": gates,
        "passed": True,
    }


def test_summary_only_full_matrix_is_rejected_before_sealing(monkeypatch):
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    artifact = _seal(_summary_only_artifact())
    with pytest.raises(ValueError, match="records"):
        validation.validate_artifact(artifact)


def _complete_matrix_fixture():
    """Synthetic zero-error evidence for validator tests, never qualification."""
    from maple.function.calculator.extra_correction.implicit.obc2_topology import OBC2Topology

    main, portable = validation.load_stage0()
    artifact = _summary_only_artifact()
    artifact.update(stage0_binding={
                        "generator": validation.STAGE0_GENERATOR_REL,
                        "generator_sha256": validation.STAGE0_GENERATOR_SHA256,
                        "oracle": validation.STAGE0_MAIN_REL,
                        "oracle_file_sha256": validation.STAGE0_MAIN_FILE_SHA256,
                        "oracle_content_sha256": validation.STAGE0_MAIN_CONTENT_SHA256,
                        "portable_fixture": validation.STAGE0_PORTABLE_REL,
                        "portable_file_sha256": validation.STAGE0_PORTABLE_FILE_SHA256,
                        "portable_content_sha256": validation.STAGE0_PORTABLE_CONTENT_SHA256,
                        "source_freeze_sha256": validation.STAGE0_SOURCE_FREEZE_SHA256,
                    },
                    gates=dict(validation.GATES),
                    fd_steps_angstrom=list(validation.SOLVENT_FD_STEPS_ANGSTROM),
                    selection=validation.select_full_h_cases(portable))
    for section_name, golden_rows, is_portable in (
        ("replay_526", main["records"], False),
        ("portable_48_ef_radius", portable["records"], True),
    ):
        rows = []
        for golden in golden_rows:
            row = {
                key: golden[key] for key in (
                    ("case", "nonpolar", "geometry") if is_portable
                    else ("compound_id", "nonpolar", "input_identity_sha256")
                )
            }
            row.update(
                golden_record_content_sha256=golden["record_content_sha256"],
                topology_sha256=golden["topology"]["topology_sha256"],
                topology_match=True, charge_identity_match=True,
                energy={"total_abs_hartree": 0.0, "polar_abs_hartree": 0.0,
                        "nonpolar_abs_hartree": 0.0,
                        "total_limit_hartree": validation.GATES["energy_abs_hartree"]
                        + validation.GATES["energy_relative"] * abs(golden["total_hartree"]),
                        "passed": True},
                parameter_max_abs=0.0,
                parameter_deltas={name: 0.0 for name in (
                    "provider_parameters_radius_nm_screen", "intrinsic_radii_angstrom",
                    "screen_factors", "offset_radii_nm", "scaled_offset_radii_nm",
                    "finalized_particle_charge_offset_scaled")},
                provider_provenance={
                    "runtime_provider": "torch-obc2", "runtime_openmm_required": False,
                    "model": "obc2", "profile": "obc2-mbondi2", "nonpolar": golden["nonpolar"],
                    "ordered_charge_sha256": golden["charge_vector_sha256"],
                    "topology_sha256": golden["topology"]["topology_sha256"]},
                passed=True,
            )
            if is_portable:
                row.update(force_max_abs_hartree_per_angstrom=0.0,
                           force_rms_hartree_per_angstrom=0.0)
            rows.append(row)
        section = artifact[section_name]
        section.update(records=rows, maximum_parameter_abs=0.0)
        if is_portable:
            section.update(case_count=12, maximum_total_energy_abs_hartree=0.0,
                           maximum_force_abs_hartree_per_angstrom=0.0,
                           maximum_force_rms_hartree_per_angstrom=0.0)
        else:
            section.update(case_count=526, profile_count=2,
                           maximum_absolute_prediction_drift_hartree=0.0,
                           mean_absolute_prediction_drift_hartree=0.0,
                           mae_drift_upper_bound_hartree=0.0,
                           mae_drift_upper_bound_kcal_mol=0.0)
    h_rows = []
    for case, profile, geometry, atoms, _charges, source in validation._full_h_inputs(portable):
        matrix = np.zeros((3 * len(atoms),) * 2)
        h_rows.append({
            "case": case, "nonpolar": profile, "geometry": geometry,
            "atom_count": len(atoms), "source_identity_sha256": source,
            "coordinate_sha256": validation.core.sha256_bytes(
                validation.core.canonical_json_bytes(atoms.positions.tolist())),
            "topology_sha256": OBC2Topology.from_atoms(atoms).identity_sha256,
            "raw_hessian_shape": list(matrix.shape),
            "raw_hessian_sha256": validation._canonical_array_sha256(matrix),
            "raw_hessian_max_abs": 0.0, "raw_hessian_frobenius": 0.0,
            "raw_hessian_asymmetry_max": 0.0, "translation_norms": [0.0] * 3,
            "translation_limit": validation.GATES["translation_absolute_hartree_per_angstrom2"],
            "fd_ladder": validation.assess_fd_ladder(
                matrix, {step: matrix.copy() for step in validation.SOLVENT_FD_STEPS_ANGSTROM}),
            "passed": True,
        })
    artifact["full_hessian_matrix"].update(
        records=h_rows, selection=artifact["selection"],
        live_force_fd_oracle={"version": validation.OPENMM_VERSION,
                             "module": "openmm.app.internal.customgbforces",
                             "source_sha256": validation.OPENMM_SOURCE_SHA256,
                             "platform": "Reference"},
        water_displacement_seed=validation.WATER_DISPLACEMENT_SEED,
        displacement_scale_angstrom=validation.DISPLACEMENT_SCALE_ANGSTROM,
    )
    h_keys = {(row["case"], row["nonpolar"], row["geometry"]) for row in h_rows}
    index = validation._portable_index(portable)
    hvp_rows = []
    for case_index, case in enumerate(validation.PORTABLE_CASES):
        for geometry_index, geometry in enumerate(validation.GEOMETRY_ORDER):
            for profile in validation.PROFILE_ORDER:
                size = 3 * index[(case, profile, geometry)]["atom_count"]
                for direction_index in range(validation.HVP_DIRECTION_COUNT):
                    seed, direction = validation.normalized_direction(
                        case_index=case_index, geometry_index=geometry_index,
                        direction_index=direction_index, size=size)
                    vector = np.zeros(size)
                    compared = (case, profile, geometry) in h_keys
                    hvp_rows.append({
                        "case": case, "nonpolar": profile, "geometry": geometry,
                        "direction_index": direction_index, "seed": seed,
                        "direction": direction.tolist(),
                        "direction_sha256": validation._canonical_array_sha256(direction),
                        "direction_l2_norm": float(np.linalg.norm(direction)),
                        "hvp": vector.tolist(),
                        "hvp_sha256": validation._canonical_array_sha256(vector),
                        "same_scalar_energy_abs_hartree": 0.0,
                        "same_scalar_force_max_hartree_per_angstrom": 0.0,
                        "raw_hessian_product": vector.tolist() if compared else None,
                        "raw_hessian_product_sha256": validation._canonical_array_sha256(vector) if compared else None,
                        "hvp_vs_raw_hessian_max": 0.0 if compared else None,
                        "hvp_vs_raw_hessian_scaled_max": 0.0 if compared else None,
                        "dense_hessian_forbidden_during_call": True,
                        "passed": True,
                    })
    artifact["direct_hvp_matrix"].update(
        records=hvp_rows, base_seed=validation.HVP_BASE_SEED,
        direction_count_per_case_geometry_profile=validation.HVP_DIRECTION_COUNT)
    artifact["guarded_no_openmm"] = {
        "script_sha256": validation._sha256_text(validation._no_openmm_script()),
        "network_disabled": True, "import_blocker": True, "metadata_blocker": True,
        "passed": True,
        "result": {"provider": "torch-obc2", "energy_hartree": -1.0,
                   "force_sha256": "a" * 64, "hessian_sha256": "b" * 64,
                   "hvp_sha256": "c" * 64, "openmm_loaded": False, "ok": True},
    }
    return artifact


@pytest.fixture(scope="module")
def complete_matrix():
    return _complete_matrix_fixture()


def test_complete_synthetic_full_matrix_replays_all_frozen_verdicts(complete_matrix, monkeypatch):
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    validation.validate_artifact(_seal(complete_matrix))


@pytest.mark.parametrize("section,count", [
    ("replay_526", 1052), ("portable_48_ef_radius", 48),
    ("full_hessian_matrix", 16), ("direct_hvp_matrix", 144),
])
@pytest.mark.parametrize("damage", ["missing", "empty", "duplicate"])
def test_full_matrix_rejects_incomplete_or_duplicate_records(
    complete_matrix, monkeypatch, section, count, damage
):
    import copy
    artifact = copy.deepcopy(complete_matrix)
    rows = artifact[section]["records"]
    assert len(rows) == count
    if damage == "missing":
        artifact[section].pop("records")
    elif damage == "empty":
        artifact[section]["records"] = []
    else:
        rows[-1] = copy.deepcopy(rows[0])
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    with pytest.raises(ValueError, match="records|identity"):
        validation.validate_artifact(_seal(artifact))


@pytest.mark.parametrize("section,field,value", [
    ("replay_526", "parameter_max_abs", 1.0),
    ("portable_48_ef_radius", "force_max_abs_hartree_per_angstrom", 1.0),
    ("portable_48_ef_radius", "force_rms_hartree_per_angstrom", 1.0),
    ("full_hessian_matrix", "raw_hessian_asymmetry_max", 1.0),
    ("full_hessian_matrix", "raw_hessian_frobenius", float("nan")),
    ("direct_hvp_matrix", "same_scalar_energy_abs_hartree", 1.0),
    ("direct_hvp_matrix", "same_scalar_force_max_hartree_per_angstrom", float("inf")),
])
def test_full_matrix_rejects_hidden_bad_metric_despite_self_resealed_pass(
    complete_matrix, monkeypatch, section, field, value
):
    import copy
    artifact = copy.deepcopy(complete_matrix)
    artifact[section]["records"][0][field] = value
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    with pytest.raises(ValueError):
        validation.validate_artifact(_seal(artifact))


@pytest.mark.parametrize("gate", [
    "sealed_oracle_1052_energy", "portable_oracle_48_energy_force",
    "analytic_hessian_16_fd", "direct_hvp_144", "no_openmm_candidate_runtime",
    "candidate_exact_cell_public_composition", "source_unchanged",
])
def test_full_matrix_rejects_changed_gate_counts(complete_matrix, monkeypatch, gate):
    import copy
    artifact = copy.deepcopy(complete_matrix)
    artifact["qualification_gates"][gate].update(expected_count=0, passed_count=0)
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    with pytest.raises(ValueError, match="denominator"):
        validation.validate_artifact(_seal(artifact))


def test_full_matrix_preserves_logged_coarse_fd_failure(complete_matrix, monkeypatch):
    import copy
    artifact = copy.deepcopy(complete_matrix)
    ladder = artifact["full_hessian_matrix"]["records"][0]["fd_ladder"]
    ladder["records"][0].update(rms_hartree_per_angstrom2=1.0, max_hartree_per_angstrom2=1.0, passed=False)
    ladder.update(coarse_pass_observed=False, all_three_pass_observed=False)
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    validation.validate_artifact(_seal(artifact))


@pytest.mark.parametrize("damage", ["step", "fine_fail", "pair_fail", "nan", "flag"])
def test_full_matrix_replays_frozen_fd_ladder(complete_matrix, monkeypatch, damage):
    import copy
    artifact = copy.deepcopy(complete_matrix)
    ladder = artifact["full_hessian_matrix"]["records"][0]["fd_ladder"]
    if damage == "step":
        ladder["records"][1]["step_angstrom"] = 1e-7
    elif damage == "fine_fail":
        ladder["records"][1]["rms_hartree_per_angstrom2"] = 1.0
    elif damage == "pair_fail":
        ladder["finest_pair_difference_max_hartree_per_angstrom2"] = 1.0
    elif damage == "nan":
        ladder["records"][0]["max_hartree_per_angstrom2"] = float("nan")
    else:
        ladder["all_three_pass_observed"] = False
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    with pytest.raises(ValueError):
        validation.validate_artifact(_seal(artifact))


def test_full_matrix_replays_hvp_allclose_not_claimed_scalars(complete_matrix, monkeypatch):
    import copy
    artifact = copy.deepcopy(complete_matrix)
    row = next(row for row in artifact["direct_hvp_matrix"]["records"]
               if row["raw_hessian_product"] is not None)
    row["hvp"][0] = 1.0
    row["hvp_sha256"] = validation._canonical_array_sha256(np.asarray(row["hvp"]))
    row["hvp_vs_raw_hessian_max"] = 1.0
    row["hvp_vs_raw_hessian_scaled_max"] = float(1.0 / validation.GATES["hvp_atol_hartree_per_angstrom2"])
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    with pytest.raises(ValueError, match="verdict|original thresholds"):
        validation.validate_artifact(_seal(artifact))


def test_full_matrix_replays_original_hvp_relative_tolerance(complete_matrix, monkeypatch):
    import copy
    artifact = copy.deepcopy(complete_matrix)
    row = next(row for row in artifact["direct_hvp_matrix"]["records"]
               if row["raw_hessian_product"] is not None)
    # Above absolute tolerance but admitted by the unchanged relative term.
    row["raw_hessian_product"][0] = 10.0
    row["hvp"][0] = 10.0 + 5e-9
    product, hvp = map(np.asarray, (row["raw_hessian_product"], row["hvp"]))
    diff = np.abs(hvp - product)
    row.update(
        raw_hessian_product_sha256=validation._canonical_array_sha256(product),
        hvp_sha256=validation._canonical_array_sha256(hvp),
        hvp_vs_raw_hessian_max=float(np.max(diff)),
        hvp_vs_raw_hessian_scaled_max=float(np.max(
            diff / np.maximum(np.abs(product), validation.GATES["hvp_atol_hartree_per_angstrom2"]))),
    )
    monkeypatch.setattr(validation, "validate_current_source_binding", lambda source: None)
    validation.validate_artifact(_seal(artifact))
