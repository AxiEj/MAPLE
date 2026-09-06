from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.route2_release import (
    aggregate_mace_mdp_polar_zero_adt_ddx_development as aggregator,
)
from tools.route2_release import (
    create_mace_mdp_polar_zero_adt_ddx_development_preregistration as creator,
)
from tools.route2_release import (
    run_mace_mdp_polar_zero_adt_ddx_development as runner,
)
from tools.route2_release import run_mace_mdp_polar_zero_adt_stage_b as stage_b_runner


def _write_inputs(root: Path) -> None:
    root.mkdir()
    for index, relative in enumerate(aggregator.INPUT_FILES.values()):
        (root / relative).write_text(f"input-{index}\n")


def _aggregate_preregistration(
    *, path: Path, input_root: Path, runner_path: Path
) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact_id": aggregator.EXPECTED_PREREGISTRATION,
        "status": "locked-after-stage-b-pass-before-zero-adt-505",
        "record_count": aggregator.EXPECTED_RECORD_COUNT,
        "partition": "development",
        "confirmation_partition_opened": False,
        "prediction_runner_uses_experimental_targets": False,
        "prediction_runner_emits_experimental_targets": False,
        "runner_sha256": aggregator._sha256(runner_path),
        "aggregator_sha256": aggregator._sha256(Path(aggregator.__file__)),
    }
    payload.update(
        {
            key: aggregator._sha256(input_root / relative)
            for key, relative in aggregator.INPUT_FILES.items()
        }
    )
    payload["preregistration_sha256"] = aggregator._canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _prediction_record(
    *, index: int, prereg_file_sha: str, runner_sha: str
) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact": aggregator.EXPECTED_RECORD_ARTIFACT,
        "do_not_commit": True,
        "partition": "development",
        "confirmation_partition_opened": False,
        "selection_index": index,
        "opaque_record_id": f"opaque-{index}",
        "geometry_sha256": f"{index + 1:064x}",
        "dataset_row_sha256": f"{index + 2:064x}",
        "selection_score_sha256": f"{index + 3:064x}",
        "canonical_solvent": "water",
        "atom_count": 2,
        "record_identity_sha256": f"{index + 4:064x}",
        "preregistration_sha256": prereg_file_sha,
        "runner_sha256": runner_sha,
        "mdp_checkpoint_sha256": "a" * 64,
        "polar_checkpoint_sha256": "b" * 64,
        "response_configuration_sha256": "c" * 64,
        "profile_id": aggregator.EXPECTED_PROFILE_ID,
        "experimental_targets_parsed_by_selection_loader": True,
        "experimental_targets_used_by_prediction": False,
        "experimental_targets_emitted": False,
        "claim_boundary": "synthetic target-unused prediction",
        "status": "pass",
        "m0_electrostatic_only": {"predicted_delta_g_kcal_mol": -5.0},
        "m1_electrostatic_plus_stock_smd_cds": {"predicted_delta_g_kcal_mol": -6.0},
        "solve_wall_seconds": 1.0,
    }
    payload["record_sha256"] = aggregator._canonical_sha256(payload)
    return payload


def _reference(index: int) -> SimpleNamespace:
    geometry = SimpleNamespace(sha256=f"{index + 1:064x}")
    record = SimpleNamespace(raw_row_sha256=f"{index + 2:064x}", delta_g_kcal_mol=-6.0)
    return SimpleNamespace(
        opaque_record_id=f"opaque-{index}",
        selection_score_sha256=f"{index + 3:064x}",
        canonical_solvent="water",
        eligible_record=SimpleNamespace(geometry=geometry, record=record),
    )


def test_prediction_writer_never_overwrites_existing_evidence(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    original = {"record": "first"}
    runner._write_json_exclusive(output, original)
    original_bytes = output.read_bytes()

    with pytest.raises(FileExistsError):
        runner._write_json_exclusive(output, {"record": "replacement"})

    assert output.read_bytes() == original_bytes


def test_aggregate_opens_targets_only_after_exact_prediction_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    _write_inputs(input_root)
    prereg_path = tmp_path / "prereg.json"
    runner_path = Path(runner.__file__)
    _aggregate_preregistration(
        path=prereg_path, input_root=input_root, runner_path=runner_path
    )
    prereg_file_sha = aggregator._sha256(prereg_path)
    runner_sha = aggregator._sha256(runner_path)
    records = tmp_path / "records"
    records.mkdir()
    for index in range(aggregator.EXPECTED_RECORD_COUNT):
        payload = _prediction_record(
            index=index,
            prereg_file_sha=prereg_file_sha,
            runner_sha=runner_sha,
        )
        (records / f"index-{index:03d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    monkeypatch.setattr(
        aggregator,
        "_load_reference_rows",
        lambda root: [_reference(index) for index in range(505)],
    )
    output = tmp_path / "aggregate.json"
    result = aggregator.aggregate(
        SimpleNamespace(
            input_dir=records,
            input_root=input_root,
            preregistration=prereg_path,
            runner=runner_path,
            output=output,
        )
    )
    assert result["status"] == "pass"
    assert result["target_opening_stage"] == (
        "after-exact-505-prediction-record-closure"
    )
    assert result["m1_electrostatic_plus_stock_smd_cds_metrics"][
        "mean_absolute_error_kcal_mol"
    ] == pytest.approx(0.0)

    tampered_path = records / "index-000.json"
    tampered = json.loads(tampered_path.read_text())
    tampered["experimental_delta_g_kcal_mol"] = -6.0
    tampered.pop("record_sha256")
    tampered["record_sha256"] = aggregator._canonical_sha256(tampered)
    tampered_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    prereg = aggregator._load_preregistration(prereg_path, runner_path, input_root)
    with pytest.raises(ValueError, match="contains target/error data"):
        aggregator._load_predictions(
            input_dir=records,
            preregistration_path=prereg_path,
            prereg=prereg,
            runner=runner_path,
        )


def _runner_method() -> dict[str, object]:
    return {
        "profile_id": runner.POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
        "scalar": (
            "G_ddX,pol(MACE-POLAR zero-field permanent point l<=1 + "
            "MACE-POLAR nonuniform Gaussian residual l<=1 with its uniform "
            "tangent removed + canonical MDP-alpha ADT dipoles) + PySCF "
            "stock SMD CDS"
        ),
        "permanent_source": "official frozen MACE-POLAR zero-field point q/p",
        "nonuniform_response": (
            "official frozen MACE-POLAR-1-M zero-anchored residual response "
            "minus its exact zero-field uniform tangent"
        ),
        "uniform_response": (
            "canonical ADT lift of the official frozen MACE-MDP molecular "
            "polarizability"
        ),
        "mace_polar_long_range_evaluator": (
            runner.MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
        "continuum_model": "pcm",
        "lmax": 12,
        "n_lebedev": 1202,
        "solver_tolerance": 1.0e-12,
        "eta": 0.1,
        "n_proc": 1,
        "mdp_device": "cpu",
        "polar_device": "cuda",
        "standard_state": "MNSol protocol gas-to-solution 1M convention",
        "cds": "unmodified PySCF 2.13.1 stock SMD CDS",
        "fitting_or_calibration": False,
    }


def test_runner_requires_exact_stage_b_pass_and_target_unused_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    _write_inputs(input_root)
    mdp = tmp_path / "mdp"
    polar = tmp_path / "polar"
    mdp.write_text("mdp")
    polar.write_text("polar")
    stage_b_prereg_path = tmp_path / "stage-b-prereg.json"
    stage_b_prereg: dict[str, object] = {"artifact": "stage-b"}
    stage_b_prereg["preregistration_sha256"] = runner._canonical_sha256(stage_b_prereg)
    stage_b_prereg_path.write_text(json.dumps(stage_b_prereg) + "\n")
    stage_b_aggregate_path = tmp_path / "stage-b-aggregate.json"
    stage_b_aggregate: dict[str, object] = {
        "artifact": "stage-b-aggregate",
        "status": "pass-stage-b",
        "stage_b_passed": True,
        "record_count": 12,
        "pass_count": 12,
        "failure_count": 0,
        "preregistration_artifact_sha256": stage_b_prereg["preregistration_sha256"],
    }
    stage_b_aggregate["aggregate_sha256"] = runner._canonical_sha256(stage_b_aggregate)
    stage_b_aggregate_path.write_text(json.dumps(stage_b_aggregate) + "\n")
    prereg_path = tmp_path / "prereg.json"
    prereg: dict[str, object] = {
        "artifact_id": runner.EXPECTED_PREREGISTRATION_ID,
        "status": "locked-after-stage-b-pass-before-zero-adt-505",
        "partition": "development",
        "confirmation_partition_opened": False,
        "record_count": 505,
        "hard_accuracy_target": {
            "comparison": "<=",
            "metric": "mean_absolute_error_kcal_mol",
            "threshold_kcal_mol": 1.5,
        },
        "fitting_or_calibration_permitted": False,
        "development_targets_previously_opened": True,
        "profile_selected_from_experimental_targets": False,
        "prediction_runner_uses_experimental_targets": False,
        "prediction_runner_emits_experimental_targets": False,
        "runner_sha256": runner._sha256(Path(runner.__file__)),
        "mdp_checkpoint_sha256": runner._sha256(mdp),
        "polar_checkpoint_sha256": runner._sha256(polar),
        "source_files_sha256": {},
        "preregistration_path": str(prereg_path),
        "method": _runner_method(),
        "stage_b_preregistration_path": str(stage_b_prereg_path),
        "stage_b_preregistration_file_sha256": runner._sha256(stage_b_prereg_path),
        "stage_b_preregistration_artifact_sha256": stage_b_prereg[
            "preregistration_sha256"
        ],
        "stage_b_aggregate_path": str(stage_b_aggregate_path),
        "stage_b_aggregate_file_sha256": runner._sha256(stage_b_aggregate_path),
        "stage_b_aggregate_artifact_sha256": stage_b_aggregate["aggregate_sha256"],
    }
    prereg.update(
        {
            key: runner._sha256(input_root / relative)
            for key, relative in runner.INPUT_FILES.items()
        }
    )
    prereg["preregistration_sha256"] = runner._canonical_sha256(prereg)
    monkeypatch.setattr(runner, "SOURCE_FILES", ())
    runner._validate_preregistration(
        prereg,
        input_root=input_root,
        preregistration_path=prereg_path,
        mdp_checkpoint=mdp,
        polar_checkpoint=polar,
        polar_device="cuda",
    )

    tampered = json.loads(json.dumps(prereg))
    tampered["prediction_runner_uses_experimental_targets"] = True
    tampered.pop("preregistration_sha256")
    tampered["preregistration_sha256"] = runner._canonical_sha256(tampered)
    with pytest.raises(ValueError, match="target-use boundary"):
        runner._validate_preregistration(
            tampered,
            input_root=input_root,
            preregistration_path=prereg_path,
            mdp_checkpoint=mdp,
            polar_checkpoint=polar,
            polar_device="cuda",
        )


def test_creator_rejects_a_nonpassing_stage_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg_path = tmp_path / "stage-b-prereg.json"
    prereg_path.write_text("{}\n")
    aggregate_path = tmp_path / "stage-b-aggregate.json"
    prereg = {"preregistration_sha256": "a" * 64}
    aggregate: dict[str, object] = {
        "artifact": f"{stage_b_runner.ARTIFACT_ID}-aggregate",
        "status": "fail",
        "record_count": 12,
        "pass_count": 11,
        "failure_count": 1,
        "stage_b_passed": False,
        "preregistration_file_sha256": creator._sha256(prereg_path),
        "preregistration_artifact_sha256": prereg["preregistration_sha256"],
        "accuracy_claim_made": False,
        "capabilities_admitted": {tier: False for tier in ("E", "F", "H", "V", "M")},
    }
    aggregate["aggregate_sha256"] = stage_b_runner._canonical_sha256(aggregate)
    aggregate_path.write_text(json.dumps(aggregate) + "\n")
    monkeypatch.setattr(stage_b_runner, "_load_preregistration", lambda path: prereg)
    monkeypatch.setattr(
        stage_b_runner, "_validate_lineage_and_inputs", lambda *args, **kwargs: None
    )
    with pytest.raises(ValueError, match="did not pass"):
        creator._validate_stage_b(
            source_root=Path(creator.__file__).resolve().parents[2],
            stage_b_preregistration=prereg_path,
            stage_b_aggregate=aggregate_path,
            stage_b_records_dir=tmp_path,
            stage_a_dataset=tmp_path,
            mdp_checkpoint=tmp_path,
            polar_checkpoint=tmp_path,
        )
