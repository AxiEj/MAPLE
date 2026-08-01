#!/usr/bin/env python3
"""Run the preregistered fixed-geometry ORCA/openCOSMO-RS MNSol panel.

The ORCA ``COSMORS`` keyword performs three BP86/def2-TZVPD single-point
calculations and one openCOSMO-RS 24a statistical-thermodynamics calculation
per record.  MNSol row-level data and all ORCA work files remain below
``.omx``.  Only a complete ten-record run may emit a public aggregate result.

The three-record ``--smoke`` mode is intentionally bounded and cannot support
an accuracy, generalization, or production-default claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)

from ase.data import chemical_symbols
from benchmark_core import sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_pilot import (
    PILOT_POST_SELECTION_FUNCTIONAL_GROUP_CLASSES,
    MNSolPilotSelection,
    validate_frozen_mnsol_pilot_selection,
)

from maple.function.cosmo_rs import (
    OpenCOSMORS24aInputBundle,
    parse_orca_opencosmors_solvation_output,
    render_orca_opencosmors24a_input,
    validate_orca_opencosmors_completion,
)

ARTIFACT_NAME = "route2-mnsol-opencosmors24a-fixed-geometry-v1"
PREREGISTRATION_PATH = (
    BENCHMARK_DIR / "route2-mnsol-opencosmors24a-fixed-geometry-prereg-v1.json"
)
PREREGISTRATION_SHA256 = (
    "67924ba47e286d976cf0dc9f93d3dd2f2fa73768b71fd7e9535b1a35fb976234"
)
FULL_PANEL_RECORD_COUNT = 10
SMOKE_SELECTION_INDICES = (1, 6, 9)
PROTOCOL_ARTIFACT_SHA256 = (
    "70c34de4b9812f557c123cd124f1b49d4b58e33467e6ba893f4c267b905b9a21"
)
PROTOCOL_FINGERPRINT = (
    "3fbc6b6c7fedd97afdd05cc654bf529b75d28fa55765ffca2311a2ff83f58fa0"
)
SELECTION_ARTIFACT_SHA256 = (
    "689dee172e7d8da3f2e5a501dd1ca905f18cf4dae9f830cabea1cb05e933a069"
)
ORCA_VERSION = "6.1.0-f.0"
ORCA_SHA256 = "3de3506205ffff90e9eaa435f4e56cd43ae27ba724cfe07a3280525aed400d8e"
OPEN_COSMORS_SHA256 = "0e4067d0de52c896cd95b7b699a506382b521036102c1f4a2fd4b59a21b569d2"
BASELINE_SHA256 = "7ea80dbf4f9b7e75106b41f3f7f1d42f7b9d38bbded12da1e581c6cf5a696e5f"
FUNCTIONAL_GROUP_COVERAGE = PILOT_POST_SELECTION_FUNCTIONAL_GROUP_CLASSES
BASELINE_METHOD_IDS = (
    "aimnet2_fixed_l0__pyscf_swig_iefpcm",
    "aimnet2_fixed_l0__pyscf_swig_cpcm",
    "aimnet2_fixed_l0__pyscf_swig_cosmo",
    "mace_fixed_l1__pyscf_swig_iefpcm",
    "mace_fixed_l1__pyscf_swig_cpcm",
    "mace_fixed_l1__pyscf_swig_cosmo",
)
_ORCA_VERSION_RE = re.compile(r"Program Version\s+(\S+)\s+-\s+RELEASE")
_NORMAL_TERMINATION = "****ORCA TERMINATED NORMALLY****"
_FAILURE_MARKERS = ("error termination", "unable to open file")
_TIE_TOLERANCE_KCAL_MOL = 1.0e-12


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return payload


def _execution_git_head() -> str:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "The openCOSMO-RS benchmark requires a clean Git checkout so "
            "every result is tied to one execution commit."
        )
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("Could not resolve a full execution Git commit.")
    return head


def _validate_preregistration(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTRATION_SHA256:
        raise ValueError("openCOSMO-RS preregistration hash drifted.")
    preregistration = _load_json_mapping(path, label="Preregistration")
    required = {
        "artifact": "route2-mnsol-opencosmors24a-fixed-geometry-prereg-v1",
        "status": "pre-registered-before-mnsol-opencosmors24a-run",
        "protocol_id": "route2-mnsol-opencosmors24a-fixed-geometry-v1",
    }
    for field, expected in required.items():
        if preregistration.get(field) != expected:
            raise ValueError(f"Preregistered field {field!r} drifted.")
    if preregistration["smoke_plan"].get("selection_indices") != list(
        SMOKE_SELECTION_INDICES
    ):
        raise ValueError("Preregistered smoke indices drifted.")
    if preregistration["runtime"].get("nprocs") != 1:
        raise ValueError("Preregistered openCOSMO-RS runtime must remain serial.")
    if preregistration["method"].get("strict_published_24a_geometry_workflow"):
        raise ValueError(
            "Fixed-geometry benchmark was relabelled as the full workflow."
        )
    if (
        preregistration["training_overlap"].get("status")
        != "known-overlap-training-domain-reproduction"
    ):
        raise ValueError("openCOSMO-RS training-overlap disclosure drifted.")
    if preregistration["selection"].get("functional_group_coverage") != list(
        FUNCTIONAL_GROUP_COVERAGE
    ):
        raise ValueError("Preregistered functional-group coverage drifted.")
    if (
        preregistration.get("protocol_artifact_sha256") != PROTOCOL_ARTIFACT_SHA256
        or preregistration.get("protocol_fingerprint") != PROTOCOL_FINGERPRINT
        or preregistration["selection"].get("artifact_sha256")
        != SELECTION_ARTIFACT_SHA256
    ):
        raise ValueError("Preregistered protocol or selection binding drifted.")
    if (
        preregistration["comparison_baseline"].get("private_artifact_sha256")
        != BASELINE_SHA256
    ):
        raise ValueError("Preregistered comparison baseline drifted.")
    if (
        tuple(preregistration["comparison_baseline"].get("method_ids", ()))
        != BASELINE_METHOD_IDS
    ):
        raise ValueError("Preregistered baseline method order drifted.")
    runtime = preregistration["runtime"]
    if runtime["orca_executable"].get("sha256") != ORCA_SHA256:
        raise ValueError("Preregistered ORCA executable hash drifted.")
    if runtime["opencosmors_executable"].get("sha256") != OPEN_COSMORS_SHA256:
        raise ValueError("Preregistered openCOSMO-RS executable hash drifted.")
    return preregistration


def _require_private_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to((REPO_ROOT / ".omx").resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain below '.omx'.") from exc
    return resolved


def _validated_output_paths(
    *,
    private_output: Path,
    public_output: Path,
    work_root: Path,
    complete_panel: bool,
) -> tuple[Path, Path, Path]:
    private = _require_private_path(
        private_output,
        label="Row-level openCOSMO-RS output",
    )
    work = _require_private_path(
        work_root,
        label="ORCA/openCOSMO-RS work root",
    )
    public = public_output.expanduser().resolve()
    if not complete_panel:
        public = _require_private_path(
            public,
            label="Subset openCOSMO-RS output",
        )
    return private, public, work


def _indexed_selection(
    full_selection: Sequence[MNSolPilotSelection],
    *,
    smoke: bool,
    record_index: int | None,
) -> tuple[list[tuple[int, MNSolPilotSelection]], bool]:
    if len(full_selection) != FULL_PANEL_RECORD_COUNT:
        raise RuntimeError(
            "The frozen MNSol openCOSMO-RS panel must contain exactly "
            f"{FULL_PANEL_RECORD_COUNT} records."
        )
    if smoke and record_index is not None:
        raise ValueError("--smoke and --record-index are mutually exclusive.")
    if smoke:
        return [
            (index, full_selection[index]) for index in SMOKE_SELECTION_INDICES
        ], False
    if record_index is not None:
        if not 0 <= record_index < len(full_selection):
            raise ValueError(
                f"--record-index must lie in [0, {len(full_selection) - 1}]."
            )
        return [(record_index, full_selection[record_index])], False
    return list(enumerate(full_selection)), True


def _dataset_hashes(dataset: Any) -> dict[str, str | None]:
    return {
        "source_artifact_sha256": dataset.source_artifact_sha256,
        "table_sha256": dataset.table_sha256,
        "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
    }


def _validate_experimental_selection(
    selection: Sequence[MNSolPilotSelection],
    *,
    temperature_k: float,
    standard_state: str,
) -> dict[str, object]:
    if temperature_k != 298.0:
        raise RuntimeError("The frozen MNSol comparison requires 298 K.")
    if standard_state != "1M-ideal-gas-to-1M-ideal-solution":
        raise RuntimeError("The MNSol standard-state contract drifted.")
    for selected in selection:
        record = selected.eligible_record.record
        geometry = selected.eligible_record.geometry
        if (
            record.process_type != "abs"
            or record.charge != 0
            or geometry.charge != 0
            or geometry.multiplicity != 1
        ):
            raise RuntimeError(
                "Only neutral closed-shell absolute MNSol records are valid."
            )
        if not math.isfinite(record.delta_g_kcal_mol):
            raise RuntimeError("MNSol experimental values must be finite.")
    return {
        "all_records_absolute_gas_to_solvent": True,
        "all_records_neutral_closed_shell": True,
        "all_values_finite": True,
        "temperature_k": temperature_k,
        "standard_state": standard_state,
    }


def _validate_record_binding(
    record: Mapping[str, Any],
    selected: MNSolPilotSelection,
    *,
    selection_index: int,
) -> None:
    item = selected.eligible_record
    expected = {
        "selection_index": selection_index,
        "canonical_solvent": selected.canonical_solvent,
        "partition": item.partition,
        "opaque_record_id": selected.opaque_record_id,
        "entry_number": item.record.entry_number,
        "geometry_handle": item.record.geometry_handle,
        "geometry_sha256": item.geometry.sha256,
        "solute_name": item.record.solute_name,
        "formula": item.record.formula,
        "atom_count": len(item.geometry.atomic_numbers),
        "subset": item.record.subset,
        "process_type": item.record.process_type,
        "charge": item.record.charge,
        "functional_group_class": FUNCTIONAL_GROUP_COVERAGE[selection_index],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                "Frozen baseline record does not match the verified MNSol "
                f"selection at index {selection_index}: {field}."
            )
    experiment = float(record.get("experimental_delta_g_kcal_mol", math.nan))
    if not math.isclose(
        experiment,
        item.record.delta_g_kcal_mol,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "Frozen baseline experiment does not match the verified MNSol row."
        )


def _load_baseline_records(
    path: Path,
    *,
    preregistration: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    selection: Sequence[MNSolPilotSelection],
    protocol_fingerprint: str,
    dataset_hashes: Mapping[str, str | None],
) -> tuple[dict[str, Any], ...]:
    private_path = _require_private_path(
        path,
        label="Frozen row-level PCM-family baseline",
    )
    if sha256_file(private_path) != BASELINE_SHA256:
        raise ValueError("Frozen PCM-family baseline hash drifted.")
    artifact = _load_json_mapping(private_path, label="PCM-family baseline")
    expected = {
        "artifact": preregistration["comparison_baseline"]["artifact"],
        "visibility": "private-user-supplied-mnsol-row-level",
        "status": "complete",
        "complete_panel": True,
        "completed_record_count": FULL_PANEL_RECORD_COUNT,
        "execution_git_head": preregistration["comparison_baseline"][
            "execution_git_head"
        ],
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "protocol_fingerprint": protocol_fingerprint,
        "dataset": dict(dataset_hashes),
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise ValueError(f"Frozen PCM-family baseline field {field!r} drifted.")
    raw_records = artifact.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(selection):
        raise ValueError("Frozen PCM-family baseline record count drifted.")
    indexed: dict[int, dict[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, dict):
            raise ValueError("Frozen PCM-family baseline records must be objects.")
        index = record.get("selection_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in indexed:
            raise ValueError("Frozen PCM-family baseline indices are invalid.")
        indexed[index] = record
    if set(indexed) != set(range(len(selection))):
        raise ValueError("Frozen PCM-family baseline indices are incomplete.")
    ordered = tuple(indexed[index] for index in range(len(selection)))
    for index, (record, selected) in enumerate(zip(ordered, selection, strict=True)):
        _validate_record_binding(record, selected, selection_index=index)
        methods = record.get("methods")
        if not isinstance(methods, Mapping) or set(methods) != set(BASELINE_METHOD_IDS):
            raise ValueError("Frozen PCM-family baseline method set drifted.")
        experiment = float(record["experimental_delta_g_kcal_mol"])
        for method_id in BASELINE_METHOD_IDS:
            method = methods[method_id]
            prediction = float(method["total_solvation_kcal_mol"])
            signed_error = float(method["signed_error_kcal_mol"])
            absolute_error = float(method["absolute_error_kcal_mol"])
            if not all(
                math.isfinite(value)
                for value in (prediction, signed_error, absolute_error)
            ):
                raise ValueError("Frozen PCM-family baseline contains nonfinite data.")
            if not math.isclose(
                signed_error,
                prediction - experiment,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                absolute_error,
                abs(signed_error),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("Frozen PCM-family baseline error ledger drifted.")
    return ordered


def _validated_executable(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} must be one executable regular file.")
    if resolved.stat().st_size != expected_size_bytes:
        raise ValueError(f"{label} size drifted.")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA256 drifted.")
    return resolved


def _inspect_orca_version(orca_executable: Path) -> str:
    binary_text = orca_executable.read_bytes().decode("latin-1")
    versions = _ORCA_VERSION_RE.findall(binary_text)
    if versions != [ORCA_VERSION]:
        raise ValueError(
            f"Expected ORCA {ORCA_VERSION}, observed version markers {versions!r}."
        )
    return versions[0]


def _validate_child_output(output: str, *, label: str) -> None:
    if output.upper().count(_NORMAL_TERMINATION) != 1:
        raise ValueError(f"{label} did not terminate normally exactly once.")
    lowered = output.lower()
    for marker in _FAILURE_MARKERS:
        if marker in lowered:
            raise ValueError(f"{label} contains failure marker {marker!r}.")


def _run_orca_record(
    selected: MNSolPilotSelection,
    baseline_record: Mapping[str, Any],
    *,
    selection_index: int,
    solvent_alias: str,
    orca_executable: Path,
    opencosmors_executable: Path,
    work_root: Path,
    maxcore_mb: int,
    timeout_seconds: int,
) -> dict[str, object]:
    workdir = work_root / f"record-{selection_index:02d}"
    if workdir.exists():
        raise FileExistsError(
            f"ORCA work directory already exists; use a new --work-root: {workdir}"
        )
    workdir.mkdir(parents=True)
    stem = f"record-{selection_index:02d}"
    geometry = selected.eligible_record.geometry
    symbols = tuple(chemical_symbols[number] for number in geometry.atomic_numbers)
    rendered = render_orca_opencosmors24a_input(
        symbols,
        geometry.coordinates_angstrom,
        solvent_alias=solvent_alias,
        maxcore_mb=maxcore_mb,
        nprocs=1,
    )
    input_path = workdir / f"{stem}.inp"
    output_path = workdir / f"{stem}.out"
    stderr_path = workdir / f"{stem}.stderr"
    input_path.write_text(rendered, encoding="utf-8")

    environment = os.environ.copy()
    executable_dirs = [
        str(orca_executable.parent),
        str(opencosmors_executable.parent),
    ]
    environment["PATH"] = os.pathsep.join(
        [*dict.fromkeys(executable_dirs), environment.get("PATH", "")]
    )
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [str(orca_executable), input_path.name],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=environment,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    output_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"ORCA returned nonzero status {completed.returncode}.")
    validate_orca_opencosmors_completion(completed.stdout)

    expected_children = (
        f"{stem}.solute_vac.lastout",
        f"{stem}.solute_cpcm.lastout",
        f"{stem}.solvent_cpcm.lastout",
    )
    child_hashes: dict[str, str] = {}
    for child_name in expected_children:
        child = workdir / child_name
        if not child.is_file():
            raise ValueError(f"ORCA omitted required child output {child_name!r}.")
        _validate_child_output(
            child.read_text(encoding="utf-8", errors="replace"),
            label=child_name,
        )
        child_hashes[child_name] = sha256_file(child)
    observed_children = {path.name for path in workdir.glob("*.lastout")}
    if observed_children != set(expected_children):
        raise ValueError("ORCA child-output set drifted from the audited workflow.")

    inputs = OpenCOSMORS24aInputBundle.from_orca_run(workdir, stem)
    result = parse_orca_opencosmors_solvation_output(
        completed.stdout,
        inputs=inputs,
    )
    prediction = float(result.delta_g_solvation_kcal_mol)
    experiment = float(selected.eligible_record.record.delta_g_kcal_mol)
    signed_error = prediction - experiment
    values = (prediction, experiment, signed_error, wall_seconds)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("openCOSMO-RS record contains nonfinite values.")

    baseline_comparisons: dict[str, object] = {}
    for method_id in BASELINE_METHOD_IDS:
        baseline = baseline_record["methods"][method_id]
        baseline_prediction = float(baseline["total_solvation_kcal_mol"])
        baseline_absolute_error = float(baseline["absolute_error_kcal_mol"])
        baseline_comparisons[method_id] = {
            "baseline_total_solvation_kcal_mol": baseline_prediction,
            "baseline_absolute_error_kcal_mol": baseline_absolute_error,
            "opencosmors_minus_baseline_prediction_kcal_mol": (
                prediction - baseline_prediction
            ),
            "opencosmors_minus_baseline_absolute_error_kcal_mol": (
                abs(signed_error) - baseline_absolute_error
            ),
        }

    item = selected.eligible_record
    record = item.record
    return {
        "selection_index": selection_index,
        "canonical_solvent": selected.canonical_solvent,
        "opencosmors_solvent_alias": solvent_alias,
        "partition": item.partition,
        "opaque_record_id": selected.opaque_record_id,
        "entry_number": record.entry_number,
        "geometry_handle": record.geometry_handle,
        "geometry_sha256": geometry.sha256,
        "solute_name": record.solute_name,
        "formula": record.formula,
        "atom_count": len(symbols),
        "subset": record.subset,
        "functional_group_class": FUNCTIONAL_GROUP_COVERAGE[selection_index],
        "process_type": record.process_type,
        "charge": record.charge,
        "experimental_delta_g_kcal_mol": experiment,
        "opencosmors_delta_g_kcal_mol": prediction,
        "signed_error_kcal_mol": signed_error,
        "absolute_error_kcal_mol": abs(signed_error),
        "wall_seconds": wall_seconds,
        "input_sha256": sha256_file(input_path),
        "main_output_sha256": sha256_file(output_path),
        "stderr_sha256": sha256_file(stderr_path),
        "child_output_sha256": child_hashes,
        "input_bundle": inputs.as_manifest(),
        "baseline_comparisons": baseline_comparisons,
    }


def _aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("Cannot aggregate zero openCOSMO-RS records.")
    signed_errors = [float(record["signed_error_kcal_mol"]) for record in records]
    predictions = [float(record["opencosmors_delta_g_kcal_mol"]) for record in records]
    wall = [float(record["wall_seconds"]) for record in records]
    return {
        "record_count": len(records),
        "mean_signed_error_kcal_mol": sum(signed_errors) / len(records),
        "mean_absolute_error_kcal_mol": (
            sum(abs(value) for value in signed_errors) / len(records)
        ),
        "root_mean_square_error_kcal_mol": math.sqrt(
            sum(value * value for value in signed_errors) / len(records)
        ),
        "maximum_absolute_error_kcal_mol": max(abs(value) for value in signed_errors),
        "mean_predicted_delta_g_kcal_mol": sum(predictions) / len(records),
        "mean_wall_seconds": sum(wall) / len(records),
        "total_wall_seconds": sum(wall),
    }


def _paired_comparisons(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    comparisons: dict[str, dict[str, float | int]] = {}
    for method_id in BASELINE_METHOD_IDS:
        absolute_differences = []
        prediction_shifts = []
        wins = ties = losses = 0
        for record in records:
            item = record["baseline_comparisons"][method_id]
            absolute_difference = float(
                item["opencosmors_minus_baseline_absolute_error_kcal_mol"]
            )
            prediction_shifts.append(
                float(item["opencosmors_minus_baseline_prediction_kcal_mol"])
            )
            absolute_differences.append(absolute_difference)
            if absolute_difference < -_TIE_TOLERANCE_KCAL_MOL:
                wins += 1
            elif absolute_difference > _TIE_TOLERANCE_KCAL_MOL:
                losses += 1
            else:
                ties += 1
        comparisons[method_id] = {
            "record_count": len(records),
            "mean_opencosmors_minus_baseline_absolute_error_kcal_mol": (
                sum(absolute_differences) / len(records)
            ),
            "mean_opencosmors_minus_baseline_prediction_kcal_mol": (
                sum(prediction_shifts) / len(records)
            ),
            "opencosmors_lower_absolute_error_count": wins,
            "absolute_error_tie_count": ties,
            "opencosmors_higher_absolute_error_count": losses,
        }
    return comparisons


def _source_hashes() -> dict[str, str]:
    relative_paths = (
        "maple/function/cosmo_rs.py",
        "docs/implicit-solvation/benchmarks/benchmark_core.py",
        "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
        "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
        (
            "docs/implicit-solvation/benchmarks/"
            "route2-mnsol-opencosmors24a-fixed-geometry-prereg-v1.json"
        ),
        (
            "docs/implicit-solvation/benchmarks/"
            "run_mnsol_opencosmors24a_fixed_geometry.py"
        ),
    )
    return {relative: sha256_file(REPO_ROOT / relative) for relative in relative_paths}


def _private_artifact(
    *,
    execution_git_head: str,
    preregistration: Mapping[str, Any],
    protocol_fingerprint: str,
    selection_manifest: Mapping[str, Any],
    dataset_hashes: Mapping[str, str | None],
    records: Sequence[Mapping[str, Any]],
    status: str,
    complete_panel: bool,
    baseline_path: Path,
    runtime: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "status": status,
        "complete_panel": complete_panel,
        "execution_git_head": execution_git_head,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "protocol_fingerprint": protocol_fingerprint,
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "dataset": dict(dataset_hashes),
        "comparison_baseline_sha256": sha256_file(baseline_path),
        "runtime": dict(runtime),
        "completed_record_count": len(records),
        "records": list(records),
    }


def _public_artifact(
    *,
    execution_git_head: str,
    preregistration: Mapping[str, Any],
    protocol: Any,
    selection_manifest: Mapping[str, Any],
    dataset: Any,
    experimental_checks: Mapping[str, object],
    records: Sequence[Mapping[str, Any]],
    complete_panel: bool,
    aggregate_metrics: Mapping[str, object],
    paired_comparisons: Mapping[str, object],
    actual_total_wall_seconds: float,
    runtime: Mapping[str, Any],
) -> dict[str, object]:
    solvents = sorted({str(record["canonical_solvent"]) for record in records})
    public_runtime = {
        key: value
        for key, value in runtime.items()
        if key not in {"orca_path", "opencosmors_path"}
    }
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": (
            "public-aggregate-only"
            if complete_panel
            else "private-subset-smoke-do-not-commit"
        ),
        "do_not_commit": not complete_panel,
        "complete_panel": complete_panel,
        "execution_git_head": execution_git_head,
        "preregistration": {
            "artifact": preregistration["artifact"],
            "sha256": PREREGISTRATION_SHA256,
            "status": preregistration["status"],
        },
        "scientific_identity": {
            "method": "ORCA 6.1.0/openCOSMO-RS 24a",
            "qc_level": "BP86/def2-TZVPD",
            "geometry_policy": preregistration["method"]["geometry_policy"],
            "strict_published_24a_geometry_workflow": False,
            "training_overlap_status": preregistration["training_overlap"]["status"],
            "continuum_equation_switch": False,
        },
        "claim_boundary": preregistration["claim_boundary"],
        "experimental_reference": {
            "dataset": "Minnesota Solvation Database",
            "version": "2012",
            "doi": "10.13020/3eks-j059",
            "temperature_k": protocol.temperature_k,
            "standard_state": protocol.standard_state,
            "source_artifact_sha256": dataset.source_artifact_sha256,
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "selected_record_checks": dict(experimental_checks),
            "row_level_data_emitted": False,
        },
        "selection": {
            "artifact_sha256": preregistration["selection"]["artifact_sha256"],
            "fingerprint": selection_manifest["selection_fingerprint"],
            "record_count": len(records),
            "full_preregistered_record_count": FULL_PANEL_RECORD_COUNT,
            "solvent_count": len(solvents),
            "solvents": solvents,
            "functional_group_coverage": preregistration["selection"][
                "functional_group_coverage"
            ],
            "used_experimental_values": False,
            "used_model_outputs": False,
        },
        "aggregate_metrics": dict(aggregate_metrics),
        "paired_method_comparisons": dict(paired_comparisons),
        "runtime": public_runtime,
        "timing_seconds": {
            "actual_total_wall": actual_total_wall_seconds,
            "status": "metadata-only-serial-fixed-order-not-a-speed-ranking",
        },
        "source_files_sha256": _source_hashes(),
        "references": preregistration["references"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--orca-executable", type=Path, required=True)
    parser.add_argument("--opencosmors-executable", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=PREREGISTRATION_PATH,
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run exactly the three preregistered representative records.",
    )
    mode.add_argument(
        "--record-index",
        type=int,
        help="Run one zero-based selected record as a private engineering smoke.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    preregistration = _validate_preregistration(args.preregistration)
    execution_git_head = _execution_git_head()
    if sha256_file(args.protocol) != PROTOCOL_ARTIFACT_SHA256:
        raise ValueError("MNSol protocol artifact hash drifted.")
    if sha256_file(args.selection) != SELECTION_ARTIFACT_SHA256:
        raise ValueError("MNSol selection artifact hash drifted.")
    protocol = load_mnsol_protocol(args.protocol)
    if protocol.fingerprint != PROTOCOL_FINGERPRINT:
        raise ValueError("MNSol protocol fingerprint drifted.")
    dataset = load_mnsol_v2012(args.source, protocol)
    selection_manifest = _load_json_mapping(
        args.selection,
        label="MNSol pilot selection",
    )
    full_selection = validate_frozen_mnsol_pilot_selection(
        selection_manifest,
        dataset,
        protocol,
    )
    indexed_selection, complete_panel = _indexed_selection(
        full_selection,
        smoke=args.smoke,
        record_index=args.record_index,
    )
    private_output, public_output, work_root = _validated_output_paths(
        private_output=args.private_output,
        public_output=args.public_output,
        work_root=args.work_root,
        complete_panel=complete_panel,
    )
    if work_root.exists():
        raise FileExistsError("--work-root must not already exist.")

    runtime_spec = preregistration["runtime"]
    orca = _validated_executable(
        args.orca_executable,
        label="ORCA executable",
        expected_sha256=ORCA_SHA256,
        expected_size_bytes=runtime_spec["orca_executable"]["size_bytes"],
    )
    opencosmors = _validated_executable(
        args.opencosmors_executable,
        label="openCOSMO-RS executable",
        expected_sha256=OPEN_COSMORS_SHA256,
        expected_size_bytes=runtime_spec["opencosmors_executable"]["size_bytes"],
    )
    if opencosmors.parent != orca.parent:
        raise ValueError(
            "The audited ORCA and bundled openCOSMO-RS executables must share "
            "one installation directory."
        )
    runtime = {
        "orca_path": str(orca),
        "orca_sha256": sha256_file(orca),
        "orca_version": _inspect_orca_version(orca),
        "opencosmors_path": str(opencosmors),
        "opencosmors_sha256": sha256_file(opencosmors),
        "nprocs": 1,
        "maxcore_mb": runtime_spec["maxcore_mb"],
        "timeout_seconds_per_record": runtime_spec["timeout_seconds_per_record"],
        "python": platform.python_version(),
    }
    hashes = _dataset_hashes(dataset)
    baseline_records = _load_baseline_records(
        args.baseline_artifact,
        preregistration=preregistration,
        selection_manifest=selection_manifest,
        selection=full_selection,
        protocol_fingerprint=protocol.fingerprint,
        dataset_hashes=hashes,
    )
    experimental_checks = _validate_experimental_selection(
        [selected for _, selected in indexed_selection],
        temperature_k=protocol.temperature_k,
        standard_state=protocol.standard_state,
    )
    aliases = preregistration["method"]["solvent_aliases"]
    work_root.mkdir(parents=True)

    wall_started = time.perf_counter()
    records: list[dict[str, object]] = []
    for ordinal, (selection_index, selected) in enumerate(
        indexed_selection,
        start=1,
    ):
        print(
            f"[{ordinal}/{len(indexed_selection)}] index={selection_index} "
            f"solvent={selected.canonical_solvent}",
            flush=True,
        )
        try:
            record = _run_orca_record(
                selected,
                baseline_records[selection_index],
                selection_index=selection_index,
                solvent_alias=aliases[selected.canonical_solvent],
                orca_executable=orca,
                opencosmors_executable=opencosmors,
                work_root=work_root,
                maxcore_mb=int(runtime_spec["maxcore_mb"]),
                timeout_seconds=int(runtime_spec["timeout_seconds_per_record"]),
            )
        except Exception as exc:
            failure = _private_artifact(
                execution_git_head=execution_git_head,
                preregistration=preregistration,
                protocol_fingerprint=protocol.fingerprint,
                selection_manifest=selection_manifest,
                dataset_hashes=hashes,
                records=records,
                status="failed",
                complete_panel=complete_panel,
                baseline_path=args.baseline_artifact,
                runtime=runtime,
            )
            failure["failure"] = {
                "selection_index": selection_index,
                "stage": "orca610-opencosmors24a-record",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            write_json_atomic(private_output, failure)
            raise
        records.append(record)
        write_json_atomic(
            private_output,
            _private_artifact(
                execution_git_head=execution_git_head,
                preregistration=preregistration,
                protocol_fingerprint=protocol.fingerprint,
                selection_manifest=selection_manifest,
                dataset_hashes=hashes,
                records=records,
                status="running",
                complete_panel=complete_panel,
                baseline_path=args.baseline_artifact,
                runtime=runtime,
            ),
        )

    aggregate_metrics = _aggregate_metrics(records)
    paired_comparisons = _paired_comparisons(records)
    actual_total_wall_seconds = time.perf_counter() - wall_started
    private = _private_artifact(
        execution_git_head=execution_git_head,
        preregistration=preregistration,
        protocol_fingerprint=protocol.fingerprint,
        selection_manifest=selection_manifest,
        dataset_hashes=hashes,
        records=records,
        status="complete",
        complete_panel=complete_panel,
        baseline_path=args.baseline_artifact,
        runtime=runtime,
    )
    private.update(
        {
            "aggregate_metrics": aggregate_metrics,
            "paired_method_comparisons": paired_comparisons,
            "actual_total_wall_seconds": actual_total_wall_seconds,
        }
    )
    write_json_atomic(private_output, private)
    public = _public_artifact(
        execution_git_head=execution_git_head,
        preregistration=preregistration,
        protocol=protocol,
        selection_manifest=selection_manifest,
        dataset=dataset,
        experimental_checks=experimental_checks,
        records=records,
        complete_panel=complete_panel,
        aggregate_metrics=aggregate_metrics,
        paired_comparisons=paired_comparisons,
        actual_total_wall_seconds=actual_total_wall_seconds,
        runtime=runtime,
    )
    write_json_atomic(public_output, public)
    print(
        f"Wrote {len(records)} record(s) to {private_output} and {public_output}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
