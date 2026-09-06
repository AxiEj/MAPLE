from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.route2_release import (
    create_mace_mdp_polar_zero_adt_stage_b_preregistration as creator,
)
from tools.route2_release import run_mace_mdp_polar_zero_adt_stage_b as runner

REPO_ROOT = Path(__file__).resolve().parents[2]


def _selection(index: int) -> dict[str, object]:
    return {
        "screen_index": index,
        "source_record_index": index,
        "source_record_sha256": f"{index + 1:064x}",
        "selection_identity": {
            "atom_count": 2,
            "atomic_numbers_sha256": "a" * 64,
            "configuration": 0,
            "molecule": f"molecule-{index}",
            "positions_raw_sha256": "b" * 64,
        },
        "selection_category": "synthetic",
        "selection_metric": "synthetic",
        "selection_metric_value": float(index),
        "selection_rank_before_duplicate_filter": index + 1,
    }


def _preregistration() -> dict[str, object]:
    selection = [_selection(index) for index in range(creator.EXPECTED_RECORD_COUNT)]
    payload: dict[str, object] = {
        "artifact": creator.ARTIFACT_ID,
        "schema_version": 1,
        "status": "locked-after-stage-a-pass-before-any-stage-b-evaluation",
        "created_utc": "2026-08-20T00:00:00+00:00",
        "source_git_head": "c" * 40,
        "source_worktree_dirty_at_lock": True,
        "stage_a_preregistration": {},
        "stage_a_aggregate": {},
        "inherited_stage_a_source_files_sha256": {},
        "stage_b_source_files_sha256": {},
        "inputs_sha256": {},
        "selection": selection,
        "selection_sha256": creator._canonical_sha256(selection),
        "method": {
            "profile_id": runner.EXPECTED_PROFILE_ID,
            "solve_stage": "five-start-cold-replay-rigid-finite-field-stage-b",
            "rotation_matrix": [list(row) for row in creator.ROTATION_MATRIX],
            "translation_angstrom": list(creator.TRANSLATION_ANGSTROM),
            "finite_field_steps": list(creator.FINITE_FIELD_STEPS),
        },
        "stage_b_gates": creator.STAGE_B_GATES,
        "claim_boundary": creator.CLAIM_BOUNDARY,
        "next_if_stage_b_passes": "accuracy",
        "next_if_stage_b_fails": "reject",
    }
    payload["preregistration_sha256"] = creator._canonical_sha256(payload)
    return payload


def _stage_a_aggregate(stage_a_prereg_sha: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact": creator.EXPECTED_STAGE_A_AGGREGATE_ARTIFACT,
        "schema_version": 1,
        "status": "pass-stage-a-zero-start-coupled-screen",
        "preregistration_file_sha256": "d" * 64,
        "preregistration_artifact_sha256": stage_a_prereg_sha,
        "record_count": creator.EXPECTED_RECORD_COUNT,
        "pass_count": creator.EXPECTED_RECORD_COUNT,
        "provider_or_gate_failure_count": 0,
        "runtime_identity_sha256": "e" * 64,
        "summary": {},
        "stage_a_passed": True,
        "next_step": "stage-b",
        "accuracy_claim_made": False,
        "capabilities_admitted": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "claim_boundary": {},
        "record_sha256s": [
            f"{index + 10:064x}" for index in range(creator.EXPECTED_RECORD_COUNT)
        ],
    }
    payload["aggregate_sha256"] = creator._canonical_sha256(payload)
    return payload


def _pass_record(
    prereg: dict[str, object], prereg_file_sha: str, index: int, runtime: dict
) -> dict[str, object]:
    gates = {name: True for name in runner.GATE_KEYS}
    payload: dict[str, object] = {
        "artifact": f"{runner.ARTIFACT_ID}-record",
        "schema_version": 1,
        "status": "pass",
        "screen_index": index,
        "selection": prereg["selection"][index],
        "preregistration_file_sha256": prereg_file_sha,
        "preregistration_artifact_sha256": prereg["preregistration_sha256"],
        "profile_id": runner.EXPECTED_PROFILE_ID,
        "evaluator_configuration_sha256": "a" * 64,
        "response_configuration_sha256": "b" * 64,
        "geometry_sha256": "c" * 64,
        "root": {
            "base": {
                "maximum_start_residual_eV": 1.0e-12,
                "start_energy_span_eV": 2.0e-13,
                "field_replay_contract": {
                    "maximum_native_field_component_difference_eV": 3.0e-9,
                },
            }
        },
        "linearization": {
            "jvp_vjp_relative_defect": 1.0e-13,
            "finite_field_maximum_errors": [4.0e-8, 1.0e-8, 2.5e-9],
        },
        "projected_local_root": {
            "maximum_sampled_state_map_gain": 0.2,
            "minimum_projected_residual_singular_value": 0.8,
            "minimum_projected_residual_symmetric_eigenvalue": 0.8,
        },
        "rigid_replay": {
            "total_energy_absolute_error_eV": 1.0e-6,
            "native_field_relative": 2.0e-6,
        },
        "gates": gates,
        "stage_b_passed": True,
        "runtime_identity": runtime,
        "runtime_identity_sha256": creator._canonical_sha256(runtime),
        "claim_boundary": creator.CLAIM_BOUNDARY,
    }
    payload["record_sha256"] = creator._canonical_sha256(payload)
    return payload


def test_stage_a_aggregate_is_a_hard_precondition() -> None:
    stage_a_prereg = {"preregistration_sha256": "a" * 64}
    aggregate = _stage_a_aggregate(stage_a_prereg["preregistration_sha256"])
    assert (
        creator.validate_stage_a_aggregate(aggregate, stage_a_prereg=stage_a_prereg)
        == aggregate["aggregate_sha256"]
    )

    failed = json.loads(json.dumps(aggregate))
    failed["status"] = "fail"
    failed["stage_a_passed"] = False
    failed["pass_count"] = 11
    failed["provider_or_gate_failure_count"] = 1
    failed.pop("aggregate_sha256")
    failed["aggregate_sha256"] = creator._canonical_sha256(failed)
    with pytest.raises(ValueError, match="did not pass all twelve"):
        creator.validate_stage_a_aggregate(failed, stage_a_prereg=stage_a_prereg)


def test_creator_is_source_root_bound(tmp_path: Path) -> None:
    args = SimpleNamespace(
        source_root=tmp_path,
        output=tmp_path / "output.json",
        dataset=tmp_path / "dataset",
        mdp_checkpoint=tmp_path / "mdp",
        polar_checkpoint=tmp_path / "polar",
        stage_a_preregistration=tmp_path / "stage-a-prereg.json",
        stage_a_aggregate=tmp_path / "stage-a-aggregate.json",
    )
    with pytest.raises(ValueError, match="own source root"):
        creator.create(args)


def test_preregistration_is_exact_and_target_free(tmp_path: Path) -> None:
    payload = _preregistration()
    path = tmp_path / "stage-b.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    assert (
        runner._load_preregistration(path)["claim_boundary"] == creator.CLAIM_BOUNDARY
    )

    tampered = json.loads(json.dumps(payload))
    tampered["experimental_solvation_targets"] = [-5.0]
    tampered.pop("preregistration_sha256")
    tampered["preregistration_sha256"] = creator._canonical_sha256(tampered)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="schema changed"):
        runner._load_preregistration(bad)


def test_preregistration_binds_the_evaluator_field_replay_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _preregistration()
    path = tmp_path / "stage-b.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    assert runner._load_preregistration(path)["stage_b_gates"][
        "maximum_five_start_field_span_eV"
    ] == pytest.approx(3.0e-9)

    monkeypatch.setattr(runner, "_evaluator_root_replay_field_atol_ev", lambda: 4.0e-9)
    with pytest.raises(ValueError, match="does not match the evaluator"):
        runner._load_preregistration(path)


def test_root_summary_states_the_field_replay_evidence_boundary() -> None:
    starts = (
        SimpleNamespace(
            label="zero",
            iterations=3,
            residual_norm_ev=1.0e-12,
            polarization_energy_ev=-0.1,
            record_sha256="a" * 64,
        ),
        SimpleNamespace(
            label="permanent",
            iterations=4,
            residual_norm_ev=2.0e-12,
            polarization_energy_ev=-0.1 + 1.0e-13,
            record_sha256="b" * 64,
        ),
    )
    state = SimpleNamespace(
        root_starts=starts,
        root_sha256="c" * 64,
        primal_residual_ev=2.0e-12,
        polarization_energy_ev=-0.1,
        total_energy_ev=-1.0,
    )
    summary = runner._root_summary(state, evaluator_field_replay_atol_ev=3.0e-9)
    contract = summary["field_replay_contract"]
    assert contract["maximum_native_field_component_difference_eV"] == pytest.approx(
        3.0e-9
    )
    assert contract["enforced_inside_evaluator_solve"] is True
    assert contract["achieved_span_serialized"] is False
    assert (
        "does not claim an independently measured span" in contract["evidence_boundary"]
    )


def test_projected_root_diagnostic_recovers_known_linear_map() -> None:
    field_shape = (2, 8)
    uniform = np.zeros((*field_shape, 3), dtype=float)
    uniform[0, 0, 0] = 1.0
    uniform[0, 1, 1] = 1.0
    uniform[0, 2, 2] = 1.0
    result = runner._projected_root_diagnostic(
        state_map_jvp=lambda direction: 0.2 * direction,
        field_shape=field_shape,
        uniform_native_basis=uniform,
        random_seed=7,
        random_probe_count=5,
    )
    assert result["full_dimension"] == 16
    assert result["projected_dimension"] == 8
    assert result["maximum_sampled_state_map_gain"] == pytest.approx(0.2)
    assert result["minimum_projected_residual_singular_value"] == pytest.approx(0.8)
    assert result["minimum_projected_residual_symmetric_eigenvalue"] == pytest.approx(
        0.8
    )
    assert result["projected_state_map_spectral_radius"] == pytest.approx(0.2)
    assert "not a full-space" in result["claim_boundary"]


def test_rotation_conventions_are_inverse_consistent() -> None:
    rotation = runner._validate_rotation(creator.ROTATION_MATRIX)
    source = np.arange(12.0).reshape(3, 4)
    field = np.arange(24.0).reshape(3, 8)
    assert np.allclose(
        runner._rotate_raw_l1(runner._rotate_raw_l1(source, rotation), rotation.T),
        source,
        atol=2.0e-14,
        rtol=0.0,
    )
    assert np.allclose(
        runner._rotate_native_field(
            runner._rotate_native_field(field, rotation), rotation.T
        ),
        field,
        atol=4.0e-14,
        rtol=0.0,
    )


def test_synthetic_aggregate_closes_records_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg = _preregistration()
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    prereg_file_sha = runner._sha256_file(prereg_path)
    runtime = {"python": "synthetic", "device": "cpu"}
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    for index in range(creator.EXPECTED_RECORD_COUNT):
        record = _pass_record(prereg, prereg_file_sha, index, runtime)
        (records_dir / f"record-{index:03d}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )

    monkeypatch.setattr(runner, "_load_preregistration", lambda path: prereg)
    monkeypatch.setattr(runner, "_validate_lineage_and_inputs", lambda *a, **k: None)
    dummy = tmp_path / "input"
    dummy.write_text("input")
    args = SimpleNamespace(
        preregistration=prereg_path,
        dataset=dummy,
        mdp_checkpoint=dummy,
        polar_checkpoint=dummy,
        output_dir=records_dir,
        aggregate_output=tmp_path / "aggregate.json",
    )
    aggregate = runner.aggregate(args)
    assert aggregate["status"] == "pass-stage-b"
    assert aggregate["pass_count"] == creator.EXPECTED_RECORD_COUNT
    assert aggregate["summary"]["maximum_finest_finite_field_error"] == pytest.approx(
        2.5e-9
    )

    changed = json.loads((records_dir / "record-011.json").read_text())
    changed["runtime_identity"]["device"] = "different"
    changed["runtime_identity_sha256"] = creator._canonical_sha256(
        changed["runtime_identity"]
    )
    changed.pop("record_sha256")
    changed["record_sha256"] = creator._canonical_sha256(changed)
    (records_dir / "record-011.json").write_text(
        json.dumps(changed, indent=2, sort_keys=True) + "\n"
    )
    args.aggregate_output = tmp_path / "aggregate-mixed.json"
    with pytest.raises(ValueError, match="mixed numerical runtimes"):
        runner.aggregate(args)


def test_help_surface_does_not_import_optional_stack() -> None:
    source = (
        REPO_ROOT / "tools/route2_release/run_mace_mdp_polar_zero_adt_stage_b.py"
    ).read_text()
    top_level = source.split("def _run_one", maxsplit=1)[0]
    assert "import torch" not in top_level
    assert "import pyddx" not in top_level
    assert "import mace" not in top_level
