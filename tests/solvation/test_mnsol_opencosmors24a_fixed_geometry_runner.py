from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_mnsol_opencosmors24a_fixed_geometry as runner  # noqa: E402


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def _selection():
    geometry = SimpleNamespace(
        sha256="a" * 64,
        atomic_numbers=(6, 1),
        coordinates_angstrom=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        charge=0,
        multiplicity=1,
    )
    record = SimpleNamespace(
        entry_number=7,
        geometry_handle="geom",
        solute_name="private-solute",
        formula="CH",
        subset="[g]",
        process_type="abs",
        charge=0,
        delta_g_kcal_mol=-2.5,
    )
    item = SimpleNamespace(
        record=record,
        geometry=geometry,
        partition="confirmation",
    )
    return SimpleNamespace(
        eligible_record=item,
        canonical_solvent="water",
        opaque_record_id="b" * 64,
    )


def _private_record(*, selection_index: int = 0):
    selected = _selection()
    record = selected.eligible_record.record
    geometry = selected.eligible_record.geometry
    baseline_comparisons = {
        method_id: {
            "baseline_total_solvation_kcal_mol": -2.0,
            "baseline_absolute_error_kcal_mol": 0.5,
            "opencosmors_minus_baseline_prediction_kcal_mol": -0.2,
            "opencosmors_minus_baseline_absolute_error_kcal_mol": -0.3,
        }
        for method_id in runner.BASELINE_METHOD_IDS
    }
    return {
        "selection_index": selection_index,
        "canonical_solvent": selected.canonical_solvent,
        "partition": selected.eligible_record.partition,
        "opaque_record_id": selected.opaque_record_id,
        "entry_number": record.entry_number,
        "geometry_handle": record.geometry_handle,
        "geometry_sha256": geometry.sha256,
        "solute_name": record.solute_name,
        "formula": record.formula,
        "atom_count": 2,
        "subset": record.subset,
        "process_type": record.process_type,
        "charge": record.charge,
        "functional_group_class": "halogenated-hydrocarbon",
        "experimental_delta_g_kcal_mol": record.delta_g_kcal_mol,
        "opencosmors_delta_g_kcal_mol": -2.2,
        "signed_error_kcal_mol": 0.3,
        "absolute_error_kcal_mol": 0.3,
        "wall_seconds": 4.0,
        "baseline_comparisons": baseline_comparisons,
    }


def test_preregistration_locks_overlap_serial_runtime_smokes_and_budget():
    preregistration = runner._validate_preregistration(runner.PREREGISTRATION_PATH)

    assert runner.PREREGISTRATION_SHA256 == (
        "67924ba47e286d976cf0dc9f93d3dd2f2fa73768b71fd7e9535b1a35fb976234"
    )
    assert preregistration["training_overlap"]["status"] == (
        "known-overlap-training-domain-reproduction"
    )
    assert preregistration["runtime"]["nprocs"] == 1
    assert preregistration["smoke_plan"]["selection_indices"] == [1, 6, 9]
    assert preregistration["exact_evaluation_budget"]["full_dft_single_points"] == 30
    assert preregistration["exact_evaluation_budget"]["smoke_dft_single_points"] == 9
    assert preregistration["method"]["strict_published_24a_geometry_workflow"] is False
    assert (
        tuple(preregistration["comparison_baseline"]["method_ids"])
        == runner.BASELINE_METHOD_IDS
    )


def test_smoke_and_single_record_modes_remain_private():
    full_selection = tuple(_selection() for _ in range(10))

    indexed, complete = runner._indexed_selection(
        full_selection,
        smoke=True,
        record_index=None,
    )
    assert complete is False
    assert [index for index, _ in indexed] == [1, 6, 9]

    private = ROOT / ".omx/test-opencosmors/private.json"
    public = ROOT / ".omx/test-opencosmors/public.json"
    work = ROOT / ".omx/test-opencosmors/work"
    assert runner._validated_output_paths(
        private_output=private,
        public_output=public,
        work_root=work,
        complete_panel=False,
    ) == (private.resolve(), public.resolve(), work.resolve())

    with pytest.raises(ValueError, match="Subset"):
        runner._validated_output_paths(
            private_output=private,
            public_output=ROOT / "docs/not-private.json",
            work_root=work,
            complete_panel=False,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner._indexed_selection(full_selection, smoke=True, record_index=1)


def test_baseline_record_binding_is_exact():
    selected = _selection()
    record = _private_record()

    runner._validate_record_binding(record, selected, selection_index=0)

    record["geometry_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="geometry_sha256"):
        runner._validate_record_binding(record, selected, selection_index=0)


def test_orca_version_and_child_completion_fail_closed(tmp_path):
    executable = tmp_path / "orca"
    executable.write_bytes(
        b"binary-prefix\x00Program Version 6.1.0-f.0 - RELEASE\x00binary-suffix"
    )
    assert runner._inspect_orca_version(executable) == "6.1.0-f.0"

    runner._validate_child_output(
        "child result\n****ORCA TERMINATED NORMALLY****\n",
        label="child",
    )
    with pytest.raises(ValueError, match="failure marker"):
        runner._validate_child_output(
            "error termination\n****ORCA TERMINATED NORMALLY****\n",
            label="child",
        )


def test_aggregate_and_paired_metrics_use_only_requested_records():
    records = [_private_record(), _private_record(selection_index=1)]

    aggregate = runner._aggregate_metrics(records)
    paired = runner._paired_comparisons(records)

    assert aggregate["record_count"] == 2
    assert aggregate["mean_absolute_error_kcal_mol"] == pytest.approx(0.3)
    assert aggregate["mean_wall_seconds"] == pytest.approx(4.0)
    for comparison in paired.values():
        assert comparison["record_count"] == 2
        assert comparison[
            "mean_opencosmors_minus_baseline_absolute_error_kcal_mol"
        ] == pytest.approx(-0.3)
        assert comparison["opencosmors_lower_absolute_error_count"] == 2


def test_public_summary_omits_mnsol_rows_and_orca_asset_paths(monkeypatch):
    preregistration = json.loads(
        runner.PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"runner": "d" * 64})
    protocol = SimpleNamespace(
        temperature_k=298.0,
        standard_state="1M-ideal-gas-to-1M-ideal-solution",
    )
    dataset = SimpleNamespace(
        source_artifact_sha256="e" * 64,
        table_sha256="f" * 64,
        normalized_bundle_sha256="1" * 64,
    )
    public = runner._public_artifact(
        execution_git_head="2" * 40,
        preregistration=preregistration,
        protocol=protocol,
        selection_manifest={
            "selection_fingerprint": preregistration["selection"]["fingerprint"]
        },
        dataset=dataset,
        experimental_checks={"all_values_finite": True},
        records=[_private_record()],
        complete_panel=True,
        aggregate_metrics={"record_count": 10},
        paired_comparisons={"method": {"record_count": 10}},
        actual_total_wall_seconds=1.0,
        runtime={
            "orca_path": "/private/orca",
            "orca_version": "6.1.0-f.0",
            "orca_sha256": "3" * 64,
            "opencosmors_path": "/private/openCOSMORS",
            "opencosmors_sha256": "4" * 64,
            "nprocs": 1,
        },
    )

    forbidden = {
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "experimental_delta_g_kcal_mol",
        "opencosmors_delta_g_kcal_mol",
        "input_bundle",
        "orca_path",
        "opencosmors_path",
        "records",
    }
    assert forbidden.isdisjoint(_all_keys(public))
    assert public["visibility"] == "public-aggregate-only"
    assert public["scientific_identity"]["continuum_equation_switch"] is False
    assert public["scientific_identity"]["strict_published_24a_geometry_workflow"] is (
        False
    )
    assert "orca_path" not in public["runtime"]
    assert "opencosmors_path" not in public["runtime"]
