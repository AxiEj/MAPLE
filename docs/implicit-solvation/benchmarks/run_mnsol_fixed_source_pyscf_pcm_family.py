#!/usr/bin/env python3
"""Run the preregistered fixed-source PySCF PCM-family MNSol panel.

The runner reuses frozen AIMNet2 monopoles and MACE-POLAR ``l<=1``
coefficients.  It changes only the PySCF SWIG boundary equation
(``IEFPCM``, ``C-PCM``, or ``COSMO``).  Row-level MNSol data remain below
``.omx``; a complete ten-record run may emit an aggregate-only public result.

``--record-index`` is an engineering smoke.  Both of its outputs must remain
below ``.omx`` and it cannot support an accuracy or equation-ranking claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from importlib import import_module
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)

import ase
from ase import Atoms
import numpy as np

from benchmark_core import sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_pilot import (
    MNSolPilotSelection,
    validate_frozen_mnsol_pilot_selection,
)
from mnsol_response_ablation import (
    aggregate_method_metrics,
    compose_method_ledger,
    paired_method_comparison,
    solve_fixed_multipole_continuum,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.function.calculator.extra_correction.implicit.pyscf_swig_response import (
    PySCFSWIGCPCMResponse,
    PySCFSWIGCOSMOResponse,
    PySCFSWIGIEFPCMResponse,
    PySCFSWIGPCMResponse,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import (
    DDPCM_MULTISOLVENT_SMD_PROFILE,
)
from maple.function.route2_solvents import route2_solvent_spec

ARTIFACT_NAME = "route2-mnsol-fixed-source-pyscf-pcm-family-v1"
PREREGISTRATION_PATH = (
    BENCHMARK_DIR / "route2-mnsol-fixed-source-pyscf-pcm-family-prereg-v1.json"
)
PREREGISTRATION_SHA256 = (
    "90d84d89d96d1db14ceb2a2e2d5d4569f9544177ac72ac64b500570ca315da54"
)
FULL_PANEL_RECORD_COUNT = 10
PYSCF_LEBEDEV_ORDER = 17
SOURCE_CHARGE_TOLERANCE_E = 1.0e-8
ENERGY_IDENTITY_TOLERANCE_EV = 2.0e-10
FROZEN_CDS_TOLERANCE_KCAL_MOL = 1.0e-10
FUNCTIONAL_GROUP_COVERAGE = (
    "halogenated-hydrocarbon",
    "ketone",
    "aromatic-hydrocarbon",
    "nitro",
    "amide",
    "cyclic-diether",
    "phenol",
    "thiophenol",
    "alcohol",
    "carboxylic-acid",
)


@dataclass(frozen=True)
class ContinuumEquation:
    """One PySCF SWIG equation with all other scientific axes fixed."""

    method_id: str
    continuum_model: str
    response_type: Callable[..., PySCFSWIGPCMResponse]


CONTINUUM_EQUATIONS = (
    ContinuumEquation(
        method_id="pyscf_swig_iefpcm",
        continuum_model="iefpcm",
        response_type=PySCFSWIGIEFPCMResponse,
    ),
    ContinuumEquation(
        method_id="pyscf_swig_cpcm",
        continuum_model="cpcm",
        response_type=PySCFSWIGCPCMResponse,
    ),
    ContinuumEquation(
        method_id="pyscf_swig_cosmo",
        continuum_model="cosmo",
        response_type=PySCFSWIGCOSMOResponse,
    ),
)
SOURCE_FIELDS = {
    "aimnet2_fixed_l0": "aimnet2_charges_e",
    "mace_fixed_l1": "mace_gas_density_coefficients",
}
METHOD_IDS = tuple(
    f"{source_id}__{equation.method_id}"
    for equation in CONTINUUM_EQUATIONS
    for source_id in SOURCE_FIELDS
)
PAIRED_METHODS = (
    *(
        (
            f"{source_id}__pyscf_swig_iefpcm",
            f"{source_id}__pyscf_swig_cpcm",
        )
        for source_id in SOURCE_FIELDS
    ),
    *(
        (
            f"{source_id}__pyscf_swig_iefpcm",
            f"{source_id}__pyscf_swig_cosmo",
        )
        for source_id in SOURCE_FIELDS
    ),
    *(
        (
            f"{source_id}__pyscf_swig_cpcm",
            f"{source_id}__pyscf_swig_cosmo",
        )
        for source_id in SOURCE_FIELDS
    ),
    *(
        (
            f"aimnet2_fixed_l0__{equation.method_id}",
            f"mace_fixed_l1__{equation.method_id}",
        )
        for equation in CONTINUUM_EQUATIONS
    ),
)


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
            "The fixed-source PCM-family benchmark requires a clean Git "
            "checkout so every result remains tied to one execution commit."
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
        raise ValueError("PCM-family preregistration hash drifted.")
    preregistration = _load_json_mapping(path, label="Preregistration")
    if preregistration.get("artifact") != (
        "route2-mnsol-fixed-source-pyscf-pcm-family-prereg-v1"
    ):
        raise ValueError("Unexpected PCM-family preregistration artifact.")
    if preregistration.get("status") != ("pre-registered-before-pyscf-pcm-family-run"):
        raise ValueError("PCM-family preregistration is not in its locked state.")
    expected_sources = list(SOURCE_FIELDS)
    observed_sources = [
        item.get("method_id")
        for item in preregistration.get("solute_sources", ())
        if isinstance(item, Mapping)
    ]
    expected_equations = [item.method_id for item in CONTINUUM_EQUATIONS]
    observed_equations = [
        item.get("method_id")
        for item in preregistration.get("continuum_equations", ())
        if isinstance(item, Mapping)
    ]
    if observed_sources != expected_sources:
        raise ValueError("Preregistered fixed-source order drifted.")
    if observed_equations != expected_equations:
        raise ValueError("Preregistered continuum-equation order drifted.")
    if preregistration["selection"].get("functional_group_coverage") != list(
        FUNCTIONAL_GROUP_COVERAGE
    ):
        raise ValueError("Preregistered functional-group coverage drifted.")
    budget = preregistration.get("exact_evaluation_budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("continuum_scalar_evaluations") != 60
    ):
        raise ValueError("Preregistered continuum evaluation budget drifted.")
    return preregistration


def _require_private_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((REPO_ROOT / ".omx").resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain below '.omx'.") from exc
    return resolved


def _validated_output_paths(
    *,
    private_output: Path,
    public_output: Path,
    complete_panel: bool,
) -> tuple[Path, Path]:
    private = _require_private_path(
        private_output,
        label="Row-level fixed-source PCM-family output",
    )
    public = public_output.resolve()
    if not complete_panel:
        public = _require_private_path(
            public,
            label="Single-record fixed-source PCM-family output",
        )
    return private, public


def _indexed_selection(
    full_selection: Sequence[MNSolPilotSelection],
    record_index: int | None,
) -> tuple[list[tuple[int, MNSolPilotSelection]], bool]:
    if len(full_selection) != FULL_PANEL_RECORD_COUNT:
        raise RuntimeError(
            "The frozen MNSol PCM-family panel must contain exactly "
            f"{FULL_PANEL_RECORD_COUNT} records."
        )
    if record_index is None:
        return list(enumerate(full_selection)), True
    if not 0 <= record_index < len(full_selection):
        raise ValueError(f"--record-index must lie in [0, {len(full_selection) - 1}].")
    return [(record_index, full_selection[record_index])], False


def _validate_experimental_selection(
    selection: Sequence[MNSolPilotSelection],
    *,
    temperature_k: float,
    standard_state: str,
) -> dict[str, object]:
    if temperature_k != 298.0:
        raise RuntimeError("The frozen comparison requires 298 K.")
    if standard_state != "1M-ideal-gas-to-1M-ideal-solution":
        raise RuntimeError("The MNSol standard-state contract drifted.")
    subsets: Counter[str] = Counter()
    for selected in selection:
        record = selected.eligible_record.record
        if record.process_type != "abs" or record.charge != 0:
            raise RuntimeError(
                "Only neutral absolute gas-to-solvent MNSol records are valid."
            )
        if not math.isfinite(record.delta_g_kcal_mol):
            raise RuntimeError("MNSol experimental values must be finite.")
        subsets[record.subset] += 1
    return {
        "all_records_absolute_gas_to_solvent": True,
        "all_records_neutral": True,
        "all_values_finite": True,
        "subset_counts": dict(sorted(subsets.items())),
    }


def _validate_source_record_binding(
    source_record: Mapping[str, Any],
    selected: MNSolPilotSelection,
    *,
    selection_index: int,
) -> None:
    item = selected.eligible_record
    record = item.record
    exact_fields = {
        "selection_index": selection_index,
        "canonical_solvent": selected.canonical_solvent,
        "partition": item.partition,
        "opaque_record_id": selected.opaque_record_id,
        "entry_number": record.entry_number,
        "geometry_handle": record.geometry_handle,
        "geometry_sha256": item.geometry.sha256,
        "solute_name": record.solute_name,
        "formula": record.formula,
        "atom_count": len(item.geometry.atomic_numbers),
        "subset": record.subset,
        "process_type": record.process_type,
        "charge": record.charge,
        "functional_group_class": FUNCTIONAL_GROUP_COVERAGE[selection_index],
    }
    for field, expected in exact_fields.items():
        if source_record.get(field) != expected:
            raise ValueError(
                "Frozen source record does not match the verified MNSol "
                f"selection at index {selection_index}: {field}."
            )
    observed_experiment = float(
        source_record.get("experimental_delta_g_kcal_mol", float("nan"))
    )
    if not math.isclose(
        observed_experiment,
        record.delta_g_kcal_mol,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "Frozen source record experimental value does not match the "
            f"verified MNSol row at index {selection_index}."
        )


def _load_frozen_source_records(
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
        label="Frozen row-level source artifact",
    )
    source_spec = preregistration["frozen_source_artifact"]
    if sha256_file(private_path) != source_spec["sha256"]:
        raise ValueError("Frozen row-level source artifact hash drifted.")
    artifact = _load_json_mapping(private_path, label="Frozen source artifact")
    required = {
        "artifact": source_spec["artifact"],
        "visibility": source_spec["visibility"],
        "execution_git_head": source_spec["execution_git_head"],
        "complete_panel": source_spec["complete_panel"],
        "completed_record_count": source_spec["completed_record_count"],
        "status": "complete",
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "protocol_fingerprint": protocol_fingerprint,
    }
    for field, expected in required.items():
        if artifact.get(field) != expected:
            raise ValueError(f"Frozen source artifact field {field!r} drifted.")
    if artifact.get("dataset") != dict(dataset_hashes):
        raise ValueError("Frozen source artifact dataset hashes drifted.")
    checkpoints = artifact.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("Frozen source artifact omitted checkpoint metadata.")
    if (
        checkpoints.get("aimnet2", {}).get("sha256")
        != source_spec["checkpoints"]["aimnet2_sha256"]
    ):
        raise ValueError("Frozen AIMNet2 checkpoint binding drifted.")
    mace = checkpoints.get("mace_polar", {})
    if (
        mace.get("identifier") != source_spec["checkpoints"]["mace_polar_identifier"]
        or mace.get("sha256") != source_spec["checkpoints"]["mace_polar_sha256"]
    ):
        raise ValueError("Frozen MACE-POLAR checkpoint binding drifted.")
    raw_records = artifact.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(selection):
        raise ValueError("Frozen source artifact record count drifted.")
    indexed: dict[int, dict[str, Any]] = {}
    for source_record in raw_records:
        if not isinstance(source_record, dict):
            raise ValueError("Frozen source records must be JSON objects.")
        index = source_record.get("selection_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Frozen source selection indices must be integers.")
        if index in indexed:
            raise ValueError("Frozen source selection indices must be unique.")
        indexed[index] = source_record
    if set(indexed) != set(range(len(selection))):
        raise ValueError("Frozen source selection indices are incomplete.")
    ordered = tuple(indexed[index] for index in range(len(selection)))
    for index, (source_record, selected) in enumerate(
        zip(ordered, selection, strict=True)
    ):
        _validate_source_record_binding(
            source_record,
            selected,
            selection_index=index,
        )
    return ordered


def _fixed_sources(
    source_record: Mapping[str, Any],
    *,
    atom_count: int,
) -> dict[str, np.ndarray]:
    aimnet_charges = np.asarray(
        source_record[SOURCE_FIELDS["aimnet2_fixed_l0"]],
        dtype=float,
    )
    if aimnet_charges.shape != (atom_count,) or not np.all(np.isfinite(aimnet_charges)):
        raise ValueError("Frozen AIMNet2 charges violate their shape contract.")
    aimnet = np.zeros((atom_count, 4), dtype=float)
    aimnet[:, 0] = aimnet_charges

    mace = np.asarray(
        source_record[SOURCE_FIELDS["mace_fixed_l1"]],
        dtype=float,
    )
    if mace.shape != (atom_count, 4) or not np.all(np.isfinite(mace)):
        raise ValueError("Frozen MACE-POLAR coefficients violate their shape contract.")
    sources = {
        "aimnet2_fixed_l0": aimnet,
        "mace_fixed_l1": mace,
    }
    for source_id, coefficients in sources.items():
        charge_error = abs(float(np.sum(coefficients[:, 0])))
        if charge_error > SOURCE_CHARGE_TOLERANCE_E:
            raise ValueError(
                f"Frozen source {source_id!r} is not neutral "
                f"(absolute residual={charge_error:.3e} e)."
            )
    return sources


def _validated_frozen_cds(
    source_record: Mapping[str, Any],
    *,
    current_cds_kcal_mol: float,
) -> float:
    methods = source_record.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("Frozen source record omitted method ledgers.")
    frozen_values = []
    for source_id in SOURCE_FIELDS:
        method = methods.get(source_id)
        if not isinstance(method, Mapping):
            raise ValueError(f"Frozen source record omitted {source_id!r}.")
        frozen_values.append(float(method["smd_cds_kcal_mol"]))
    if not all(math.isfinite(value) for value in frozen_values):
        raise ValueError("Frozen SMD-CDS values must be finite.")
    if not math.isclose(
        frozen_values[0],
        frozen_values[1],
        rel_tol=0.0,
        abs_tol=FROZEN_CDS_TOLERANCE_KCAL_MOL,
    ):
        raise ValueError("Frozen source methods did not share one SMD-CDS value.")
    if not math.isclose(
        current_cds_kcal_mol,
        frozen_values[0],
        rel_tol=0.0,
        abs_tol=FROZEN_CDS_TOLERANCE_KCAL_MOL,
    ):
        raise ValueError("Current PySCF SMD-CDS value drifted from frozen source.")
    return frozen_values[0]


def _validate_same_surface(
    surfaces: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    reference_id = CONTINUUM_EQUATIONS[0].method_id
    if set(surfaces) != {item.method_id for item in CONTINUUM_EQUATIONS}:
        raise ValueError("Same-surface comparison requires every continuum equation.")
    reference_points, reference_areas = surfaces[reference_id]
    comparisons: dict[str, object] = {}
    for equation in CONTINUUM_EQUATIONS[1:]:
        points, areas = surfaces[equation.method_id]
        same_shape = (
            points.shape == reference_points.shape
            and areas.shape == reference_areas.shape
        )
        if not same_shape:
            raise RuntimeError(
                f"{equation.method_id} did not share the reference surface shape."
            )
        point_difference = float(np.max(np.abs(points - reference_points)))
        area_difference = float(np.max(np.abs(areas - reference_areas)))
        if point_difference != 0.0 or area_difference != 0.0:
            raise RuntimeError(
                f"{equation.method_id} did not share the exact PySCF SWIG surface."
            )
        comparisons[equation.method_id] = {
            "same_shape": True,
            "maximum_point_difference_bohr": point_difference,
            "maximum_area_difference_bohr2": area_difference,
        }
    return {
        "reference_method": reference_id,
        "surface_size": int(reference_points.shape[0]),
        "comparisons": comparisons,
    }


def _atoms(selected: MNSolPilotSelection) -> Atoms:
    geometry = selected.eligible_record.geometry
    atoms = Atoms(
        numbers=geometry.atomic_numbers,
        positions=geometry.coordinates_angstrom,
    )
    atoms.info.update(charge=geometry.charge, mult=geometry.multiplicity)
    return atoms


def _solve_fixed_source(
    reaction_field: Any,
    coefficients: np.ndarray,
    *,
    source_id: str,
    equation: ContinuumEquation,
    experimental_kcal_mol: float,
    cds_kcal_mol: float,
    cds_wall_seconds: float,
    build_wall_seconds: float,
    runtime_provenance: Mapping[str, object],
    surface_size: int,
) -> dict[str, object]:
    started = time.perf_counter()
    state = solve_fixed_multipole_continuum(
        reaction_field,
        coefficients,
        charge_tolerance_e=SOURCE_CHARGE_TOLERANCE_E,
        energy_identity_tolerance_ev=ENERGY_IDENTITY_TOLERANCE_EV,
    )
    solve_wall_seconds = time.perf_counter() - started
    continuum_kcal_mol = (
        float(state["polarization_energy_hartree"]) * HARTREE_TO_KCAL_MOL
    )
    return compose_method_ledger(
        experimental_kcal_mol=experimental_kcal_mol,
        solute_polarization_kcal_mol=0.0,
        continuum_polarization_kcal_mol=continuum_kcal_mol,
        smd_cds_kcal_mol=cds_kcal_mol,
        wall_seconds=(cds_wall_seconds + build_wall_seconds + solve_wall_seconds),
        extra={
            "source_id": source_id,
            "continuum_equation": equation.continuum_model,
            "continuum_method_id": equation.method_id,
            "surface_size": surface_size,
            "continuum_build_wall_seconds": build_wall_seconds,
            "continuum_solve_wall_seconds": solve_wall_seconds,
            "shared_cds_wall_seconds": cds_wall_seconds,
            "half_coupling_identity_error_ev": state["energy_identity_error_ev"],
            "reaction_field_inf_ev": float(
                np.max(np.abs(state["reaction_field_values_ev"]))
            ),
            "runtime_provenance": dict(runtime_provenance),
        },
    )


def _run_record(
    selected: MNSolPilotSelection,
    source_record: Mapping[str, Any],
    *,
    selection_index: int,
) -> dict[str, object]:
    atoms = _atoms(selected)
    symbols = tuple(atoms.get_chemical_symbols())
    positions = np.asarray(atoms.get_positions(), dtype=float)
    atom_count = len(atoms)
    solvent = selected.canonical_solvent
    experiment = float(selected.eligible_record.record.delta_g_kcal_mol)
    sources = _fixed_sources(source_record, atom_count=atom_count)

    current_radii = route2_coulomb_radii(
        symbols,
        solvent=solvent,
        profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
    )
    frozen_radii = np.asarray(
        source_record["cavity_radii_angstrom"],
        dtype=float,
    )
    if (
        frozen_radii.shape != (atom_count,)
        or not np.all(np.isfinite(frozen_radii))
        or np.any(frozen_radii <= 0.0)
        or not np.array_equal(current_radii, frozen_radii)
    ):
        raise ValueError("Current SMD Coulomb radii drifted from frozen source.")

    cds_started = time.perf_counter()
    cds = pyscf_smd_cds(symbols, positions, solvent=solvent)
    cds_wall_seconds = time.perf_counter() - cds_started
    frozen_cds = _validated_frozen_cds(
        source_record,
        current_cds_kcal_mol=float(cds.energy_kcal_mol),
    )

    dielectric = float(route2_solvent_spec(solvent).descriptors.dielectric)
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    methods: dict[str, object] = {}
    for equation in CONTINUUM_EQUATIONS:
        build_started = time.perf_counter()
        response = equation.response_type(
            symbols,
            positions,
            current_radii,
            dielectric=dielectric,
            lebedev_order=PYSCF_LEBEDEV_ORDER,
        )
        reaction_field = response.reaction_field_linear_map()
        build_wall_seconds = time.perf_counter() - build_started
        surfaces[equation.method_id] = (
            response.surface_points_bohr,
            response.surface_areas_bohr2,
        )
        for source_id, coefficients in sources.items():
            method_id = f"{source_id}__{equation.method_id}"
            methods[method_id] = _solve_fixed_source(
                reaction_field,
                coefficients,
                source_id=source_id,
                equation=equation,
                experimental_kcal_mol=experiment,
                cds_kcal_mol=frozen_cds,
                cds_wall_seconds=cds_wall_seconds,
                build_wall_seconds=build_wall_seconds,
                runtime_provenance=response.runtime_provenance,
                surface_size=response.surface_size,
            )
    if tuple(methods) != METHOD_IDS:
        raise RuntimeError("Fixed-source PCM-family method order drifted.")
    surface_parity = _validate_same_surface(surfaces)
    item = selected.eligible_record
    record = item.record
    return {
        "selection_index": selection_index,
        "canonical_solvent": solvent,
        "partition": item.partition,
        "opaque_record_id": selected.opaque_record_id,
        "entry_number": record.entry_number,
        "geometry_handle": record.geometry_handle,
        "geometry_sha256": item.geometry.sha256,
        "solute_name": record.solute_name,
        "formula": record.formula,
        "atom_count": atom_count,
        "subset": record.subset,
        "functional_group_class": FUNCTIONAL_GROUP_COVERAGE[selection_index],
        "process_type": record.process_type,
        "charge": record.charge,
        "experimental_delta_g_kcal_mol": experiment,
        "dielectric": dielectric,
        "smd_cds_runtime_provenance": dict(cds.runtime_provenance),
        "pyscf_same_surface_parity": surface_parity,
        "methods": methods,
    }


def _aggregate_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    return {
        method_id: aggregate_method_metrics(records, method_id)
        for method_id in METHOD_IDS
    }


def _paired_comparisons(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | str]]:
    return {
        f"{left}__to__{right}": paired_method_comparison(
            records,
            left=left,
            right=right,
        )
        for left, right in PAIRED_METHODS
    }


def _surface_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    sizes = []
    point_differences = []
    area_differences = []
    for record in records:
        parity = record["pyscf_same_surface_parity"]
        sizes.append(int(parity["surface_size"]))
        for comparison in parity["comparisons"].values():
            point_differences.append(float(comparison["maximum_point_difference_bohr"]))
            area_differences.append(float(comparison["maximum_area_difference_bohr2"]))
    return {
        "minimum_surface_size": min(sizes),
        "maximum_surface_size": max(sizes),
        "maximum_point_difference_bohr": max(point_differences, default=0.0),
        "maximum_area_difference_bohr2": max(area_differences, default=0.0),
    }


def _source_hashes() -> dict[str, str]:
    implicit = "maple/function/calculator/extra_correction/implicit"
    relative_paths = (
        f"{implicit}/electrostatic_pairing.py",
        f"{implicit}/gto_density.py",
        f"{implicit}/pyscf_runtime.py",
        f"{implicit}/pyscf_smd_cds.py",
        f"{implicit}/pyscf_swig_response.py",
        f"{implicit}/route2_pcm_response.py",
        f"{implicit}/smd_cds.py",
        "maple/function/route2_smd_profiles.py",
        "maple/function/route2_solvents.py",
        "docs/implicit-solvation/benchmarks/benchmark_core.py",
        "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
        "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
        "docs/implicit-solvation/benchmarks/mnsol_response_ablation.py",
        (
            "docs/implicit-solvation/benchmarks/"
            "route2-mnsol-fixed-source-pyscf-pcm-family-prereg-v1.json"
        ),
        (
            "docs/implicit-solvation/benchmarks/"
            "run_mnsol_fixed_source_pyscf_pcm_family.py"
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
    source_artifact_path: Path,
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
        "frozen_source_artifact": {
            "artifact": preregistration["frozen_source_artifact"]["artifact"],
            "sha256": sha256_file(source_artifact_path),
            "execution_git_head": preregistration["frozen_source_artifact"][
                "execution_git_head"
            ],
        },
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
    surface_summary: Mapping[str, object],
    total_wall_seconds: float,
) -> dict[str, object]:
    pyscf = import_module("pyscf")
    solvents = sorted({str(record["canonical_solvent"]) for record in records})
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": (
            "public-aggregate-only"
            if complete_panel
            else "private-single-record-smoke-do-not-commit"
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
            "solute_sources": {
                "aimnet2_fixed_l0": (
                    "frozen AIMNet2 atom-centred monopoles; no source response"
                ),
                "mace_fixed_l1": (
                    "frozen gas MACE-POLAR atom-centred l<=1 coefficients; "
                    "no source response"
                ),
            },
            "continuum_provider": f"PySCF {pyscf.__version__} SWIG",
            "continuum_equations": [
                item.continuum_model for item in CONTINUUM_EQUATIONS
            ],
            "shared_cavity": ("per-solvent PySCF-2.13.1 SMD Coulomb radii"),
            "shared_nonpolar_model": "PySCF 2.13.1 SMD-CDS",
            "strict_original_smd_equivalence": False,
            "cosmo_rs_included": False,
        },
        "claim_boundary": preregistration["claim_boundary"],
        "experimental_reference": {
            "dataset": "Minnesota Solvation Database",
            "version": "2012",
            "doi": "10.13020/3eks-j059",
            "official_homepage": "https://comp.chem.umn.edu/mnsol/",
            "temperature_k": protocol.temperature_k,
            "standard_state": protocol.standard_state,
            "source_artifact_sha256": dataset.source_artifact_sha256,
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "selected_record_checks": dict(experimental_checks),
            "row_level_data_emitted": not complete_panel,
            "row_level_disclosure_reason": (
                "Only aggregate statistics over the complete panel are emitted."
                if complete_panel
                else (
                    "A one-record aggregate reproduces a row-level experiment "
                    "and therefore remains private below .omx."
                )
            ),
        },
        "selection": {
            "artifact_sha256": preregistration["selection"]["artifact_sha256"],
            "fingerprint": selection_manifest["selection_fingerprint"],
            "record_count": len(records),
            "full_preregistered_record_count": FULL_PANEL_RECORD_COUNT,
            "selection_indices": [int(record["selection_index"]) for record in records],
            "solvent_count": len(solvents),
            "solvents": solvents,
            "functional_group_assignment": preregistration["selection"][
                "functional_group_assignment"
            ],
            "functional_group_coverage": preregistration["selection"][
                "functional_group_coverage"
            ],
            "used_experimental_values": False,
            "used_model_outputs": False,
        },
        "frozen_source_artifact": {
            "artifact": preregistration["frozen_source_artifact"]["artifact"],
            "sha256": preregistration["frozen_source_artifact"]["sha256"],
            "execution_git_head": preregistration["frozen_source_artifact"][
                "execution_git_head"
            ],
            "checkpoints": preregistration["frozen_source_artifact"]["checkpoints"],
        },
        "aggregate_metrics": dict(aggregate_metrics),
        "paired_method_comparisons": dict(paired_comparisons),
        "same_surface_summary": dict(surface_summary),
        "numerics": {
            "pyscf_version": str(pyscf.__version__),
            "surface_discretization_method": "SWIG",
            "lebedev_order": PYSCF_LEBEDEV_ORDER,
            "source_charge_tolerance_e": SOURCE_CHARGE_TOLERANCE_E,
            "energy_identity_tolerance_ev": ENERGY_IDENTITY_TOLERANCE_EV,
            "frozen_cds_tolerance_kcal_mol": (FROZEN_CDS_TOLERANCE_KCAL_MOL),
        },
        "timing_seconds": {
            "actual_total_wall": total_wall_seconds,
            "per_method_definition": (
                "shared CDS + continuum build + fixed-source solve; frozen "
                "AIMNet2/MACE inference time is excluded"
            ),
            "status": "metadata-only-not-a-randomized-speed-ranking",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "pyscf": str(pyscf.__version__),
        },
        "source_files_sha256": _source_hashes(),
        "references": preregistration["references"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--frozen-source-artifact", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=PREREGISTRATION_PATH,
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument(
        "--record-index",
        type=int,
        help="Run one zero-based selected record as a private smoke test.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    preregistration = _validate_preregistration(args.preregistration)
    execution_git_head = _execution_git_head()
    protocol = load_mnsol_protocol(args.protocol)
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
        args.record_index,
    )
    private_output, public_output = _validated_output_paths(
        private_output=args.private_output,
        public_output=args.public_output,
        complete_panel=complete_panel,
    )
    dataset_hashes = {
        "source_artifact_sha256": dataset.source_artifact_sha256,
        "table_sha256": dataset.table_sha256,
        "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
    }
    frozen_records = _load_frozen_source_records(
        args.frozen_source_artifact,
        preregistration=preregistration,
        selection_manifest=selection_manifest,
        selection=full_selection,
        protocol_fingerprint=protocol.fingerprint,
        dataset_hashes=dataset_hashes,
    )
    experimental_checks = _validate_experimental_selection(
        [selected for _, selected in indexed_selection],
        temperature_k=protocol.temperature_k,
        standard_state=protocol.standard_state,
    )

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
            result = _run_record(
                selected,
                frozen_records[selection_index],
                selection_index=selection_index,
            )
        except Exception as exc:
            failure = _private_artifact(
                execution_git_head=execution_git_head,
                preregistration=preregistration,
                protocol_fingerprint=protocol.fingerprint,
                selection_manifest=selection_manifest,
                dataset_hashes=dataset_hashes,
                records=records,
                status="failed",
                complete_panel=complete_panel,
                source_artifact_path=args.frozen_source_artifact,
            )
            failure["failure"] = {
                "selection_index": selection_index,
                "stage": "fixed-source-pyscf-pcm-family-record",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            write_json_atomic(private_output, failure)
            raise
        records.append(result)
        write_json_atomic(
            private_output,
            _private_artifact(
                execution_git_head=execution_git_head,
                preregistration=preregistration,
                protocol_fingerprint=protocol.fingerprint,
                selection_manifest=selection_manifest,
                dataset_hashes=dataset_hashes,
                records=records,
                status="running",
                complete_panel=complete_panel,
                source_artifact_path=args.frozen_source_artifact,
            ),
        )

    aggregate_metrics = _aggregate_metrics(records)
    paired_comparisons = _paired_comparisons(records)
    surface_summary = _surface_summary(records)
    total_wall_seconds = time.perf_counter() - wall_started
    private = _private_artifact(
        execution_git_head=execution_git_head,
        preregistration=preregistration,
        protocol_fingerprint=protocol.fingerprint,
        selection_manifest=selection_manifest,
        dataset_hashes=dataset_hashes,
        records=records,
        status="complete",
        complete_panel=complete_panel,
        source_artifact_path=args.frozen_source_artifact,
    )
    private.update(
        {
            "aggregate_metrics": aggregate_metrics,
            "paired_method_comparisons": paired_comparisons,
            "same_surface_summary": surface_summary,
            "actual_total_wall_seconds": total_wall_seconds,
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
        surface_summary=surface_summary,
        total_wall_seconds=total_wall_seconds,
    )
    write_json_atomic(public_output, public)
    print(
        f"Wrote {len(records)} record(s) to {private_output} and " f"{public_output}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
