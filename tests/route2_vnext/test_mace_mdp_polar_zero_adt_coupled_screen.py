from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.route2_release import (
    create_mace_mdp_polar_zero_adt_coupled_screen_preregistration as creator,
)
from tools.route2_release import (
    run_mace_mdp_polar_zero_adt_coupled_screen as runner,
)
from tools.route2_release import run_mdp_mbis_pcm_source_gate as source_parent

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_EVIDENCE = REPO_ROOT / creator.SOURCE_EVIDENCE_RELATIVE


def _selection_item(index: int) -> dict[str, object]:
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
        "selection_category": "synthetic-test",
        "selection_metric": "synthetic",
        "selection_metric_value": float(index),
        "selection_rank_before_duplicate_filter": index + 1,
    }


def _preregistration() -> dict[str, object]:
    selection = [
        _selection_item(index) for index in range(creator.EXPECTED_SCREEN_RECORD_COUNT)
    ]
    payload: dict[str, object] = {
        "artifact": creator.ARTIFACT_ID,
        "schema_version": 1,
        "status": (
            "locked-after-array-identity-remediation-before-any-selected-v3-"
            "coupled-state-evaluation"
        ),
        "created_utc": "2026-08-19T00:00:00+00:00",
        "source_git_head": "c" * 40,
        "source_worktree_dirty_at_lock": True,
        "source_files_sha256": {},
        "superseded_preregistration": {},
        "provider_remediation": {},
        "source_evidence": {},
        "inputs_sha256": {},
        "selection_policy": {},
        "selection": selection,
        "selection_sha256": creator._canonical_sha256(selection),
        "method": {
            "profile_id": runner.EXPECTED_PROFILE_ID,
            "lmax": 12,
            "n_lebedev": 1202,
            "solve_stage": "one-zero-start-rejection-screen",
        },
        "stage_a_gates": {},
        "claim_boundary": creator.CLAIM_BOUNDARY,
        "next_if_stage_a_passes": "full screen",
        "next_if_stage_a_fails": "reject",
    }
    payload["preregistration_sha256"] = creator._canonical_sha256(payload)
    return payload


def _write_preregistration(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _pass_record(
    prereg: dict[str, object],
    prereg_file_sha256: str,
    index: int,
    *,
    runtime_sha256: str,
) -> dict[str, object]:
    runtime = {
        "python": "synthetic",
        "implementation": "CPython",
        "platform": "synthetic",
        "machine": "synthetic",
        "cpu_model": "synthetic",
        "packages": {},
        "numpy": None,
        "torch": None,
        "environment": {},
        "selected_mace_polar_device": "cpu",
    }
    assert runtime_sha256 == creator._canonical_sha256(runtime)
    gates = {name: True for name in runner.GATE_KEYS}
    payload: dict[str, object] = {
        "artifact": f"{runner.ARTIFACT_ID}-record",
        "schema_version": 1,
        "status": "pass",
        "screen_index": index,
        "selection": prereg["selection"][index],
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_artifact_sha256": prereg["preregistration_sha256"],
        "profile_id": runner.EXPECTED_PROFILE_ID,
        "evaluator_configuration_sha256": "d" * 64,
        "response_configuration_sha256": "e" * 64,
        "geometry_sha256": "f" * 64,
        "screen_state_sha256": "1" * 64,
        "continuum_state_sha256": "2" * 64,
        "root": {
            "iterations": 7,
            "primal_residual_eV": 1.0e-12,
            "polarization_energy_eV_diagnostic_only": -0.1,
            "native_field_l2": 0.2,
            "radial_residual_source_l2": 0.01,
            "adt_atomic_dipoles_l2_eangstrom": 0.02,
        },
        "charge_closure": {
            "permanent_charge_e": -2.0e-12,
            "radial_response_charge_e": 3.0e-12,
        },
        "uniform_chart": {
            "left_inverse_maximum_absolute_error": 1.0e-15,
            "zero_field_uniform_radial_cancellation_maximum_absolute": 2.0e-12,
            "zero_field_adt_molecular_closure_l2_eangstrom": 3.0e-13,
            "root_adt_molecular_closure_l2_eangstrom": 2.0e-13,
            "molecular_alpha_eigenvalues": [0.1, 0.2, 0.3],
        },
        "linearization": {
            "jvp_vjp_dot_left": 0.5,
            "jvp_vjp_dot_right": 0.5,
            "jvp_vjp_absolute_defect": 0.0,
            "jvp_vjp_relative_defect": 4.0e-13,
        },
        "gates": gates,
        "stage_a_passed": True,
        "runtime_identity": runtime,
        "runtime_identity_sha256": runtime_sha256,
        "claim_boundary": creator.CLAIM_BOUNDARY,
    }
    payload["record_sha256"] = creator._canonical_sha256(payload)
    return payload


def test_adversarial_selection_is_exact_and_target_independent() -> None:
    records = creator.load_source_records(SOURCE_EVIDENCE)
    selected = creator.select_screen_records(records)
    assert [item["source_record_index"] for item in selected] == [
        44,
        42,
        6,
        23,
        7,
        37,
        17,
        3,
        21,
        29,
        1,
        30,
    ]
    assert len({item["source_record_index"] for item in selected}) == 12
    serialized = json.dumps(selected, sort_keys=True).lower()
    assert "mnsol" not in serialized
    assert "freesolv" not in serialized
    assert "experimental_delta_g" not in serialized


def test_geometry_loader_never_reads_qm_source_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numbers = np.asarray([6, 1], dtype=np.int64)
    positions_raw = np.asarray([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    accessed: list[str] = []

    class Group:
        def __getitem__(self, key: str) -> np.ndarray:
            accessed.append(key)
            if key == "atomic_numbers":
                return numbers
            if key == "positions":
                return positions_raw
            raise AssertionError(f"forbidden dataset read: {key}")

    class Handle:
        def __enter__(self) -> "Handle":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __contains__(self, key: str) -> bool:
            return key == "molecule"

        def __getitem__(self, key: str) -> Group:
            assert key == "molecule"
            return Group()

    fake_h5py = SimpleNamespace(File=lambda *args, **kwargs: Handle())
    monkeypatch.setitem(__import__("sys").modules, "h5py", fake_h5py)
    selected = {
        "selection_identity": {
            "molecule": "molecule",
            "configuration": 0,
            "atom_count": 2,
            "atomic_numbers_sha256": source_parent._array_sha256(numbers),
            "positions_raw_sha256": source_parent._array_sha256(positions_raw[0]),
        }
    }
    arrays = runner._load_geometry_arrays(Path("unused.h5"), selected)
    assert accessed == ["atomic_numbers", "positions"]
    assert np.array_equal(arrays["numbers"], numbers)
    assert np.array_equal(arrays["positions_angstrom"], positions_raw[0] * 10.0)
    assert runner._array_sha256(numbers) == source_parent._array_sha256(numbers)


def test_preregistration_rejects_rehashed_extra_or_wrong_profile(
    tmp_path: Path,
) -> None:
    valid = _preregistration()
    path = tmp_path / "valid.json"
    _write_preregistration(path, valid)
    assert (
        runner._load_preregistration(path)["selection_sha256"]
        == valid["selection_sha256"]
    )

    extra = dict(valid)
    extra["experimental_solvation_targets"] = [1.0]
    extra.pop("preregistration_sha256")
    extra["preregistration_sha256"] = creator._canonical_sha256(extra)
    extra_path = tmp_path / "extra.json"
    _write_preregistration(extra_path, extra)
    with pytest.raises(ValueError, match="unregistered fields"):
        runner._load_preregistration(extra_path)

    wrong = json.loads(json.dumps(valid))
    wrong["method"]["profile_id"] = "wrong-profile"
    wrong.pop("preregistration_sha256")
    wrong["preregistration_sha256"] = creator._canonical_sha256(wrong)
    wrong_path = tmp_path / "wrong.json"
    _write_preregistration(wrong_path, wrong)
    with pytest.raises(ValueError, match="profile identity"):
        runner._load_preregistration(wrong_path)


def test_synthetic_aggregate_closes_exact_records_and_rejects_runtime_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg = _preregistration()
    prereg_path = tmp_path / "prereg.json"
    _write_preregistration(prereg_path, prereg)
    prereg_file_sha256 = runner._sha256_file(prereg_path)
    runtime_sha256 = creator._canonical_sha256(
        {
            "python": "synthetic",
            "implementation": "CPython",
            "platform": "synthetic",
            "machine": "synthetic",
            "cpu_model": "synthetic",
            "packages": {},
            "numpy": None,
            "torch": None,
            "environment": {},
            "selected_mace_polar_device": "cpu",
        }
    )

    records_dir = tmp_path / "records"
    records_dir.mkdir()
    for index in range(creator.EXPECTED_SCREEN_RECORD_COUNT):
        payload = _pass_record(
            prereg,
            prereg_file_sha256,
            index,
            runtime_sha256=runtime_sha256,
        )
        (records_dir / f"record-{index:03d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    monkeypatch.setattr(runner, "_load_preregistration", lambda path: prereg)
    monkeypatch.setattr(
        runner,
        "_validate_source_and_inputs",
        lambda *args, **kwargs: None,
    )
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
    assert aggregate["status"] == "pass-stage-a-zero-start-coupled-screen"
    assert aggregate["pass_count"] == 12
    assert aggregate["summary"]["maximum_absolute_permanent_charge_e"] == 2.0e-12

    tampered = json.loads((records_dir / "record-011.json").read_text())
    tampered["runtime_identity"]["platform"] = "different"
    tampered["runtime_identity_sha256"] = creator._canonical_sha256(
        tampered["runtime_identity"]
    )
    tampered.pop("record_sha256")
    tampered["record_sha256"] = creator._canonical_sha256(tampered)
    (records_dir / "record-011.json").write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n"
    )
    args.aggregate_output = tmp_path / "aggregate-mixed.json"
    with pytest.raises(ValueError, match="mixed numerical runtimes"):
        runner.aggregate(args)


def test_help_does_not_import_optional_scientific_stack() -> None:
    source = (
        REPO_ROOT / "tools/route2_release/run_mace_mdp_polar_zero_adt_coupled_screen.py"
    ).read_text()
    top_level = source.split("def _configure_torch", maxsplit=1)[0]
    assert "import torch" not in top_level
    assert "import pyddx" not in top_level
    assert "import mace" not in top_level
