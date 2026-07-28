from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_mnsol_fixed_source_pyscf_pcm_family as runner  # noqa: E402


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


def _source_record():
    selected = _selection()
    record = selected.eligible_record.record
    geometry = selected.eligible_record.geometry
    return {
        "selection_index": 0,
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
        "aimnet2_charges_e": [0.2, -0.2],
        "mace_gas_density_coefficients": [
            [0.1, 0.2, -0.3, 0.4],
            [-0.1, -0.2, 0.3, -0.4],
        ],
    }


def test_preregistration_locks_sources_equations_budget_and_cosmors_boundary():
    preregistration = runner._validate_preregistration(runner.PREREGISTRATION_PATH)

    assert runner.PREREGISTRATION_SHA256 == (
        "90d84d89d96d1db14ceb2a2e2d5d4569f9544177ac72ac64b500570ca315da54"
    )
    assert [item["method_id"] for item in preregistration["solute_sources"]] == [
        "aimnet2_fixed_l0",
        "mace_fixed_l1",
    ]
    assert [
        item["continuum_model"] for item in preregistration["continuum_equations"]
    ] == ["iefpcm", "cpcm", "cosmo"]
    assert (
        preregistration["exact_evaluation_budget"]["continuum_scalar_evaluations"] == 60
    )
    assert preregistration["excluded_axes"]["cosmo_rs"].startswith(
        "Excluded because COSMO-RS"
    )
    assert len(runner.METHOD_IDS) == 6
    assert len(runner.PAIRED_METHODS) == 9
    assert len(preregistration["selection"]["functional_group_coverage"]) == 10
    source_hashes = runner._source_hashes()
    assert source_hashes
    assert all(len(digest) == 64 for digest in source_hashes.values())


def test_fixed_source_binding_checks_verified_mnsol_identity():
    selected = _selection()
    source = _source_record()

    runner._validate_source_record_binding(
        source,
        selected,
        selection_index=0,
    )

    drifted = dict(source)
    drifted["geometry_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="geometry_sha256"):
        runner._validate_source_record_binding(
            drifted,
            selected,
            selection_index=0,
        )


def test_fixed_sources_preserve_l0_and_l1_and_fail_closed_on_charge():
    source = _source_record()

    sources = runner._fixed_sources(source, atom_count=2)

    np.testing.assert_allclose(
        sources["aimnet2_fixed_l0"],
        np.asarray([[0.2, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(
        sources["mace_fixed_l1"],
        np.asarray(source["mace_gas_density_coefficients"]),
    )

    nonneutral = copy.deepcopy(source)
    nonneutral["aimnet2_charges_e"][0] += 0.01
    with pytest.raises(ValueError, match="not neutral"):
        runner._fixed_sources(nonneutral, atom_count=2)


def test_same_surface_gate_requires_exact_shared_points_and_areas():
    points = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    areas = np.asarray([0.5, 0.7])
    surfaces = {
        equation.method_id: (points.copy(), areas.copy())
        for equation in runner.CONTINUUM_EQUATIONS
    }

    parity = runner._validate_same_surface(surfaces)

    assert parity["surface_size"] == 2
    assert all(
        item["maximum_point_difference_bohr"] == 0.0
        for item in parity["comparisons"].values()
    )

    drifted = dict(surfaces)
    changed = points.copy()
    changed[0, 0] = np.nextafter(0.0, 1.0)
    drifted["pyscf_swig_cosmo"] = (changed, areas.copy())
    with pytest.raises(RuntimeError, match="exact PySCF SWIG surface"):
        runner._validate_same_surface(drifted)


def test_single_record_smoke_keeps_both_outputs_private():
    full_selection = tuple(_selection() for _ in range(10))
    indexed, complete = runner._indexed_selection(full_selection, 4)
    assert complete is False
    assert indexed == [(4, full_selection[4])]

    private = ROOT / ".omx/test-pcm-family/private.json"
    public = ROOT / ".omx/test-pcm-family/public.json"
    assert runner._validated_output_paths(
        private_output=private,
        public_output=public,
        complete_panel=False,
    ) == (private.resolve(), public.resolve())

    with pytest.raises(ValueError, match="Single-record"):
        runner._validated_output_paths(
            private_output=private,
            public_output=ROOT / "docs/not-private.json",
            complete_panel=False,
        )
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        runner._indexed_selection(full_selection, 10)


def test_public_summary_omits_row_level_mnsol_and_frozen_coefficients(
    monkeypatch,
):
    preregistration = json.loads(
        runner.PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    record = _source_record()
    record["methods"] = {}
    monkeypatch.setattr(
        runner,
        "import_module",
        lambda name: SimpleNamespace(__version__="2.13.1"),
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
        records=[record],
        complete_panel=True,
        aggregate_metrics={"method": {"record_count": 10}},
        paired_comparisons={"pair": {"record_count": 10}},
        surface_summary={"maximum_point_difference_bohr": 0.0},
        total_wall_seconds=1.0,
    )

    forbidden = {
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "experimental_delta_g_kcal_mol",
        "aimnet2_charges_e",
        "mace_gas_density_coefficients",
        "records",
    }
    assert forbidden.isdisjoint(_all_keys(public))
    assert public["visibility"] == "public-aggregate-only"
    assert public["scientific_identity"]["cosmo_rs_included"] is False
