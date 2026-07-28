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

import aggregate_mnsol_opencosmors24a_fixed_geometry as aggregation  # noqa: E402
from benchmark_core import sha256_file  # noqa: E402
import run_mnsol_opencosmors24a_fixed_geometry as runner  # noqa: E402
from maple.function.cosmo_rs import (  # noqa: E402
    OpenCOSMORS24aInputBundle,
    render_orca_opencosmors24a_input,
)


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
    return SimpleNamespace(
        eligible_record=SimpleNamespace(
            record=record,
            geometry=geometry,
            partition="confirmation",
        ),
        canonical_solvent="water",
        opaque_record_id="b" * 64,
    )


def _baseline_record():
    methods = {}
    for method_id in runner.BASELINE_METHOD_IDS:
        methods[method_id] = {
            "total_solvation_kcal_mol": -2.0,
            "signed_error_kcal_mol": 0.5,
            "absolute_error_kcal_mol": 0.5,
        }
    return {"methods": methods}


def _record_with_assets(tmp_path: Path):
    selected = _selection()
    workdir = tmp_path / "work"
    workdir.mkdir()
    stem = "record-00"
    symbols = ("C", "H")
    input_path = workdir / f"{stem}.inp"
    main_path = workdir / f"{stem}.out"
    stderr_path = workdir / f"{stem}.stderr"
    input_path.write_text(
        render_orca_opencosmors24a_input(
            symbols,
            selected.eligible_record.geometry.coordinates_angstrom,
            solvent_alias="Water",
        ),
        encoding="utf-8",
    )
    main_path.write_text(
        """
        OPENCOSMO-RS CALCULATION
        Reference temperature : 298.15 K
        Free energy of solvation (dGsolv) : -0.006626234385 Eh -4.158026 kcal/mol
        ****ORCA TERMINATED NORMALLY****
        """,
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")
    child_hashes = {}
    for suffix in (
        "solute_vac.lastout",
        "solute_cpcm.lastout",
        "solvent_cpcm.lastout",
    ):
        child = workdir / f"{stem}.{suffix}"
        child.write_text(
            "child\n****ORCA TERMINATED NORMALLY****\n",
            encoding="utf-8",
        )
        child_hashes[child.name] = sha256_file(child)
    (workdir / f"{stem}.solute.orcacosmo").write_text("solute", encoding="utf-8")
    (workdir / f"{stem}.solvent.orcacosmo").write_text("solvent", encoding="utf-8")
    bundle = OpenCOSMORS24aInputBundle.from_orca_run(workdir, stem)
    prediction = -4.158026
    experiment = -2.5
    signed_error = prediction - experiment
    comparisons = {
        method_id: {
            "baseline_total_solvation_kcal_mol": -2.0,
            "baseline_absolute_error_kcal_mol": 0.5,
            "opencosmors_minus_baseline_prediction_kcal_mol": prediction + 2.0,
            "opencosmors_minus_baseline_absolute_error_kcal_mol": (
                abs(signed_error) - 0.5
            ),
        }
        for method_id in runner.BASELINE_METHOD_IDS
    }
    record = {
        "selection_index": 0,
        "canonical_solvent": "water",
        "opencosmors_solvent_alias": "Water",
        "partition": "confirmation",
        "opaque_record_id": "b" * 64,
        "entry_number": 7,
        "geometry_handle": "geom",
        "geometry_sha256": "a" * 64,
        "solute_name": "private-solute",
        "formula": "CH",
        "atom_count": 2,
        "subset": "[g]",
        "functional_group_class": "halogenated-hydrocarbon",
        "process_type": "abs",
        "charge": 0,
        "experimental_delta_g_kcal_mol": experiment,
        "opencosmors_delta_g_kcal_mol": prediction,
        "signed_error_kcal_mol": signed_error,
        "absolute_error_kcal_mol": abs(signed_error),
        "wall_seconds": 2.0,
        "input_sha256": sha256_file(input_path),
        "main_output_sha256": sha256_file(main_path),
        "stderr_sha256": sha256_file(stderr_path),
        "child_output_sha256": child_hashes,
        "input_bundle": bundle.as_manifest(),
        "baseline_comparisons": comparisons,
    }
    return record, selected, workdir


def test_fragment_header_locks_private_runtime_and_scientific_inputs():
    preregistration = json.loads(
        runner.PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    dataset = {
        "source_artifact_sha256": "1" * 64,
        "table_sha256": "2" * 64,
        "normalized_bundle_sha256": "3" * 64,
    }
    fragment = {
        "artifact": runner.ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "status": "complete",
        "complete_panel": False,
        "execution_git_head": "4" * 40,
        "preregistration_sha256": runner.PREREGISTRATION_SHA256,
        "protocol_fingerprint": runner.PROTOCOL_FINGERPRINT,
        "selection_fingerprint": "5" * 64,
        "dataset": dataset,
        "comparison_baseline_sha256": runner.BASELINE_SHA256,
        "runtime": {
            "orca_sha256": runner.ORCA_SHA256,
            "orca_version": runner.ORCA_VERSION,
            "opencosmors_sha256": runner.OPEN_COSMORS_SHA256,
            "nprocs": 1,
            "maxcore_mb": 2000,
            "timeout_seconds_per_record": 900,
        },
    }

    aggregation._validate_fragment_header(
        fragment,
        preregistration=preregistration,
        protocol_fingerprint=runner.PROTOCOL_FINGERPRINT,
        selection_fingerprint="5" * 64,
        dataset_hashes=dataset,
        baseline_sha256=runner.BASELINE_SHA256,
    )

    fragment["runtime"]["nprocs"] = 2
    with pytest.raises(ValueError, match="nprocs"):
        aggregation._validate_fragment_header(
            fragment,
            preregistration=preregistration,
            protocol_fingerprint=runner.PROTOCOL_FINGERPRINT,
            selection_fingerprint="5" * 64,
            dataset_hashes=dataset,
            baseline_sha256=runner.BASELINE_SHA256,
        )


def test_record_aggregation_replays_input_output_assets_and_baseline(
    tmp_path,
    monkeypatch,
):
    record, selected, workdir = _record_with_assets(tmp_path)
    preregistration = json.loads(
        runner.PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        aggregation.runner,
        "_require_private_path",
        lambda path, label: Path(path).resolve(),
    )

    aggregation._validate_record_assets(
        record,
        selected,
        _baseline_record(),
        selection_index=0,
        preregistration=preregistration,
    )

    (workdir / "record-00.inp").write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic rendering"):
        aggregation._validate_record_assets(
            record,
            selected,
            _baseline_record(),
            selection_index=0,
            preregistration=preregistration,
        )
