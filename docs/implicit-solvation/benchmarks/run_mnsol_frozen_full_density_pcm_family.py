#!/usr/bin/env python3
"""Run the preregistered frozen full-density PCM-family diagnostic.

The new arms use one fixed gas-phase DF-RKS density in two representations:
the AO density evaluated directly on the cavity and the same density projected
into a SALTED-compatible RI Gaussian basis.  The latter is a representation
oracle, not a learned SALTED prediction.  Existing MACE fixed-``l<=1`` ledgers
are rebound from an immutable matched-panel artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)

import ase
from ase import Atoms
from ase.units import Hartree
import numpy as np
import pyscf
from pyscf import df, dft, gto, lib

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
)
from run_mnsol_fixed_source_pyscf_pcm_family import (
    CONTINUUM_EQUATIONS,
    FROZEN_CDS_TOLERANCE_KCAL_MOL,
    FULL_PANEL_RECORD_COUNT,
    FUNCTIONAL_GROUP_COVERAGE,
    PYSCF_LEBEDEV_ORDER,
    _validate_experimental_selection,
    _validate_same_surface,
    _validate_source_record_binding,
    _validated_frozen_cds,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.function.calculator.extra_correction.implicit.salted_ri_density import (
    pyscf_auxiliary_l1_slices,
    pyscf_coefficients_to_salted_order,
    salted_ri_surface_mep,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    route2_coulomb_radii,
)
from maple.function.calculator.extra_correction.implicit.source import (
    FrozenDensityMEPSource,
    solve_frozen_density_mep_continuum,
)
from maple.function.route2_smd_profiles import (
    DDPCM_MULTISOLVENT_SMD_PROFILE,
)
from maple.function.route2_solvents import route2_solvent_spec

ARTIFACT_NAME = "route2-mnsol-frozen-full-density-pcm-family-v1"
PREREGISTRATION_PATH = (
    BENCHMARK_DIR
    / "route2-mnsol-frozen-full-density-pcm-family-prereg-v1.json"
)
PREREGISTRATION_SHA256 = (
    "faf10a95c58c8d5c0b7b1fd84b7683a1030c38e9119df2638261687fc92c14e3"
)
QM_FUNCTIONAL = "wb97m-v"
QM_BASIS = "def2-svpd"
AUXILIARY_BASIS = "def2-svp-jkfit"
QM_GRID_LEVEL = 3
QM_NLC_ATOM_GRID = (50, 194)
QM_MAX_CYCLES = 100
QM_ENERGY_TOLERANCE_HARTREE = 1.0e-10
QM_GRADIENT_TOLERANCE = 1.0e-7
QM_THREADS = 8
QM_MAX_MEMORY_MB = 8000
AO_CHARGE_TOLERANCE_E = 1.0e-7
RI_RAW_CHARGE_TOLERANCE_E = 0.1
RI_PROJECTED_CHARGE_TOLERANCE_E = 1.0e-8
MEP_POINT_CHUNK_SIZE = 200
SOURCE_IDS = (
    "mace_fixed_l1",
    "qm_ao_frozen",
    "qm_ri_salted_space_frozen",
)
NEW_SOURCE_IDS = SOURCE_IDS[1:]
METHOD_IDS = tuple(
    f"{source_id}__{equation.method_id}"
    for equation in CONTINUUM_EQUATIONS
    for source_id in SOURCE_IDS
)
PAIRED_METHODS = tuple(
    (
        f"{left}__{equation.method_id}",
        f"{right}__{equation.method_id}",
    )
    for equation in CONTINUUM_EQUATIONS
    for left, right in (
        ("mace_fixed_l1", "qm_ao_frozen"),
        ("mace_fixed_l1", "qm_ri_salted_space_frozen"),
        ("qm_ao_frozen", "qm_ri_salted_space_frozen"),
    )
)


@dataclass(frozen=True)
class QMDensityState:
    molecule: Any
    density_matrix: np.ndarray
    energy_hartree: float
    electron_count_e: float
    target_electron_count_e: float
    charge_residual_e: float
    scf_cycles: int
    scf_wall_seconds: float
    semilocal_grid_point_count: int
    nonlocal_grid_point_count: int


@dataclass(frozen=True)
class RIDensityFit:
    salted_coefficients: np.ndarray
    coefficient_count: int
    fit_wall_seconds: float


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
            "The frozen full-density benchmark requires a clean Git checkout."
        )
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_preregistration(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTRATION_SHA256:
        raise ValueError("Frozen full-density preregistration hash drifted.")
    preregistration = _load_json_mapping(path, label="Preregistration")
    expected = {
        "artifact": (
            "route2-mnsol-frozen-full-density-pcm-family-prereg-v1"
        ),
        "status": "pre-registered-before-full-density-run",
        "protocol_id": ARTIFACT_NAME,
    }
    for field, value in expected.items():
        if preregistration.get(field) != value:
            raise ValueError(f"Preregistration field {field!r} drifted.")
    numerics = preregistration.get("numerics")
    if not isinstance(numerics, Mapping):
        raise ValueError("Preregistration omitted locked numerics.")
    locked_numerics = {
        "electronic_structure_method": "omegaB97M-V",
        "electronic_structure_basis": QM_BASIS,
        "auxiliary_basis": AUXILIARY_BASIS,
        "semilocal_grid_level": QM_GRID_LEVEL,
        "maximum_scf_cycles": QM_MAX_CYCLES,
        "scf_energy_tolerance_hartree": QM_ENERGY_TOLERANCE_HARTREE,
        "scf_gradient_tolerance": QM_GRADIENT_TOLERANCE,
        "ri_charge_constraint": (
            "Coulomb-metric minimum correction to the exact "
            "closed-shell electron count"
        ),
    }
    for field, value in locked_numerics.items():
        if numerics.get(field) != value:
            raise ValueError(f"Preregistered numeric {field!r} drifted.")
    if preregistration["selection"]["functional_group_coverage"] != list(
        FUNCTIONAL_GROUP_COVERAGE
    ):
        raise ValueError("Preregistered functional-group coverage drifted.")
    observed_sources = [
        item.get("method_id")
        for item in preregistration.get("solute_sources", ())
        if isinstance(item, Mapping)
    ]
    if observed_sources != list(SOURCE_IDS):
        raise ValueError("Preregistered full-density source order drifted.")
    return preregistration


def _require_private_output(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((REPO_ROOT / ".omx").resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain below this worktree's .omx.") from exc
    return resolved


def _validated_output_paths(
    *,
    private_output: Path,
    public_output: Path,
    complete_panel: bool,
) -> tuple[Path, Path]:
    private = _require_private_output(
        private_output,
        label="Row-level full-density output",
    )
    public = public_output.resolve()
    if not complete_panel:
        public = _require_private_output(
            public,
            label="Single-record full-density output",
        )
    for output in (private, public):
        if output.exists():
            raise FileExistsError(output)
    return private, public


def _indexed_selection(
    selection: Sequence[MNSolPilotSelection],
    record_index: int | None,
) -> tuple[list[tuple[int, MNSolPilotSelection]], bool]:
    if len(selection) != FULL_PANEL_RECORD_COUNT:
        raise RuntimeError("The frozen full-density panel must have ten records.")
    if record_index is None:
        return list(enumerate(selection)), True
    if not 0 <= record_index < len(selection):
        raise ValueError(f"--record-index must lie in [0, {len(selection) - 1}].")
    return [(record_index, selection[record_index])], False


def _ordered_records(
    artifact: Mapping[str, Any],
    selection: Sequence[MNSolPilotSelection],
    *,
    expected_artifact: str,
    label: str,
) -> tuple[dict[str, Any], ...]:
    if (
        artifact.get("artifact") != expected_artifact
        or artifact.get("status") != "complete"
        or artifact.get("complete_panel") is not True
        or artifact.get("completed_record_count") != len(selection)
    ):
        raise ValueError(f"{label} completion metadata drifted.")
    raw_records = artifact.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(selection):
        raise ValueError(f"{label} record count drifted.")
    by_index: dict[int, dict[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, dict):
            raise ValueError(f"{label} records must be JSON objects.")
        index = record.get("selection_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{label} selection indices must be integers.")
        if index in by_index:
            raise ValueError(f"{label} selection indices must be unique.")
        by_index[index] = record
    if set(by_index) != set(range(len(selection))):
        raise ValueError(f"{label} selection indices are incomplete.")
    ordered = tuple(by_index[index] for index in range(len(selection)))
    for index, (record, selected) in enumerate(
        zip(ordered, selection, strict=True)
    ):
        _validate_source_record_binding(
            record,
            selected,
            selection_index=index,
        )
    return ordered


def _load_bound_artifacts(
    *,
    preregistration: Mapping[str, Any],
    baseline_path: Path,
    source_path: Path,
    selection: Sequence[MNSolPilotSelection],
    protocol_fingerprint: str,
    selection_fingerprint: str,
    dataset_hashes: Mapping[str, str | None],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    baseline_spec = preregistration["frozen_baseline_artifact"]
    if sha256_file(baseline_path) != baseline_spec["sha256"]:
        raise ValueError("Frozen PCM-family baseline artifact hash drifted.")
    baseline = _load_json_mapping(baseline_path, label="Baseline artifact")
    source_spec = preregistration["frozen_source_artifact"]
    if sha256_file(source_path) != source_spec["sha256"]:
        raise ValueError("Frozen source artifact hash drifted.")
    source = _load_json_mapping(source_path, label="Source artifact")
    for label, artifact in (("Baseline", baseline), ("Source", source)):
        if artifact.get("protocol_fingerprint") != protocol_fingerprint:
            raise ValueError(f"{label} protocol fingerprint drifted.")
        if artifact.get("selection_fingerprint") != selection_fingerprint:
            raise ValueError(f"{label} selection fingerprint drifted.")
        if artifact.get("dataset") != dict(dataset_hashes):
            raise ValueError(f"{label} dataset hashes drifted.")
    baseline_records = _ordered_records(
        baseline,
        selection,
        expected_artifact=baseline_spec["artifact"],
        label="Baseline artifact",
    )
    source_records = _ordered_records(
        source,
        selection,
        expected_artifact=source_spec["artifact"],
        label="Source artifact",
    )
    for record in baseline_records:
        methods = record.get("methods")
        if not isinstance(methods, Mapping):
            raise ValueError("Baseline record omitted method ledgers.")
        for equation in CONTINUUM_EQUATIONS:
            method_id = f"mace_fixed_l1__{equation.method_id}"
            if not isinstance(methods.get(method_id), Mapping):
                raise ValueError(f"Baseline record omitted {method_id!r}.")
    return baseline_records, source_records


def _atoms(selected: MNSolPilotSelection) -> Atoms:
    geometry = selected.eligible_record.geometry
    atoms = Atoms(
        numbers=geometry.atomic_numbers,
        positions=geometry.coordinates_angstrom,
    )
    atoms.info.update(charge=geometry.charge, mult=geometry.multiplicity)
    return atoms


def _run_qm_density(atoms: Atoms) -> QMDensityState:
    symbols = tuple(atoms.get_chemical_symbols())
    positions = np.asarray(atoms.get_positions(), dtype=float)
    molecule = gto.M(
        atom=list(zip(symbols, positions, strict=True)),
        basis=QM_BASIS,
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
        max_memory=QM_MAX_MEMORY_MB,
    )
    mean_field = dft.RKS(molecule, xc=QM_FUNCTIONAL).density_fit(
        auxbasis=AUXILIARY_BASIS
    )
    mean_field.grids.level = QM_GRID_LEVEL
    mean_field.nlcgrids.atom_grid = {
        symbol: QM_NLC_ATOM_GRID for symbol in sorted(set(symbols))
    }
    mean_field.nlcgrids.prune = dft.gen_grid.sg1_prune
    mean_field.conv_tol = QM_ENERGY_TOLERANCE_HARTREE
    mean_field.conv_tol_grad = QM_GRADIENT_TOLERANCE
    mean_field.max_cycle = QM_MAX_CYCLES
    mean_field.max_memory = QM_MAX_MEMORY_MB
    cycles = 0

    def count_cycle(_environment):
        nonlocal cycles
        cycles += 1

    mean_field.callback = count_cycle
    started = time.perf_counter()
    energy = float(mean_field.kernel())
    elapsed = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError("The frozen gas-phase QM density did not converge.")
    density = np.asarray(mean_field.make_rdm1(), dtype=float)
    if (
        density.shape != (molecule.nao_nr(), molecule.nao_nr())
        or not np.all(np.isfinite(density))
    ):
        raise RuntimeError("The frozen gas-phase AO density is invalid.")
    density = 0.5 * (density + density.T)
    electron_count = float(
        np.einsum(
            "ij,ji->",
            density,
            molecule.intor_symmetric("int1e_ovlp"),
        )
    )
    target = float(np.sum(molecule.atom_charges()))
    residual = abs(electron_count - target)
    if residual > AO_CHARGE_TOLERANCE_E:
        raise RuntimeError(
            "The frozen AO density violates its electron-count gate "
            f"({residual:.3e} e)."
        )
    return QMDensityState(
        molecule=molecule,
        density_matrix=density,
        energy_hartree=energy,
        electron_count_e=electron_count,
        target_electron_count_e=target,
        charge_residual_e=residual,
        scf_cycles=cycles,
        scf_wall_seconds=elapsed,
        semilocal_grid_point_count=int(mean_field.grids.coords.shape[0]),
        nonlocal_grid_point_count=int(mean_field.nlcgrids.coords.shape[0]),
    )


def _fit_salted_ri_density(state: QMDensityState) -> RIDensityFit:
    started = time.perf_counter()
    auxiliary = df.addons.make_auxmol(state.molecule, AUXILIARY_BASIS)
    three_center = df.incore.aux_e2(state.molecule, auxiliary)
    right_hand_side = np.einsum(
        "ijp,ij->p",
        three_center,
        state.density_matrix,
    )
    coefficients_pyscf = np.linalg.solve(
        auxiliary.intor("int2c2e_sph"),
        right_hand_side,
    )
    coefficients_salted = pyscf_coefficients_to_salted_order(
        coefficients_pyscf,
        l1_slices=pyscf_auxiliary_l1_slices(auxiliary),
    )
    if not np.all(np.isfinite(coefficients_salted)):
        raise RuntimeError("The SALTED-compatible RI density fit is invalid.")
    elapsed = time.perf_counter() - started
    return RIDensityFit(
        salted_coefficients=coefficients_salted,
        coefficient_count=int(coefficients_salted.size),
        fit_wall_seconds=elapsed,
    )


def _nuclear_surface_mep(molecule: Any, points_bohr: np.ndarray) -> np.ndarray:
    result = np.zeros(points_bohr.shape[0], dtype=float)
    for charge, center in zip(
        molecule.atom_charges(),
        molecule.atom_coords(unit="Bohr"),
        strict=True,
    ):
        distances = np.linalg.norm(points_bohr - center, axis=1)
        if np.any(distances <= 1.0e-12):
            raise ValueError("A PCM surface point coincides with a QM nucleus.")
        result += float(charge) / distances
    return result


def _ao_surface_mep(
    state: QMDensityState,
    points_bohr: np.ndarray,
) -> tuple[FrozenDensityMEPSource, float]:
    started = time.perf_counter()
    electronic = np.empty(points_bohr.shape[0], dtype=float)
    for start in range(0, points_bohr.shape[0], MEP_POINT_CHUNK_SIZE):
        stop = min(start + MEP_POINT_CHUNK_SIZE, points_bohr.shape[0])
        fake_points = gto.fakemol_for_charges(points_bohr[start:stop])
        kernel = np.asarray(
            df.incore.aux_e2(state.molecule, fake_points),
            dtype=float,
        )
        expected_shape = (
            state.molecule.nao_nr(),
            state.molecule.nao_nr(),
            stop - start,
        )
        if kernel.shape != expected_shape:
            raise RuntimeError("PySCF returned an unexpected AO MEP kernel.")
        electronic[start:stop] = np.einsum(
            "ijp,ij->p",
            kernel,
            state.density_matrix,
        )
    potential = _nuclear_surface_mep(state.molecule, points_bohr) - electronic
    source = FrozenDensityMEPSource(
        surface_points_bohr=points_bohr,
        surface_potential_hartree_per_e=potential,
        declared_total_charge_e=0.0,
        observed_total_charge_e=(
            state.target_electron_count_e - state.electron_count_e
        ),
        source_model=(
            f"pyscf-{pyscf.__version__}:{QM_FUNCTIONAL}/{QM_BASIS}:DF-RKS"
        ),
        density_representation=f"ao-density:{QM_BASIS}",
        charge_tolerance_e=AO_CHARGE_TOLERANCE_E,
    )
    return source, time.perf_counter() - started


def _solve_full_density(
    response: Any,
    source: FrozenDensityMEPSource,
    *,
    source_id: str,
    equation: Any,
    experiment_kcal_mol: float,
    cds_kcal_mol: float,
    shared_wall_seconds: float,
    build_wall_seconds: float,
    source_wall_seconds: float,
    source_provenance: Mapping[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    coupled = solve_frozen_density_mep_continuum(response, source)
    solve_wall_seconds = time.perf_counter() - started
    state = coupled.response_state
    identity_error_ev = abs(
        float(state.polarization_energy_hartree)
        - 0.5
        * float(
            np.dot(
                state.surface_potential_hartree_per_e,
                state.energy_conjugate_surface_charge_e,
            )
        )
    ) * Hartree
    if identity_error_ev > 2.0e-10:
        raise RuntimeError("Frozen full-density half-coupling identity failed.")
    continuum_kcal_mol = (
        float(state.polarization_energy_hartree) * HARTREE_TO_KCAL_MOL
    )
    return compose_method_ledger(
        experimental_kcal_mol=experiment_kcal_mol,
        solute_polarization_kcal_mol=0.0,
        continuum_polarization_kcal_mol=continuum_kcal_mol,
        smd_cds_kcal_mol=cds_kcal_mol,
        wall_seconds=(
            shared_wall_seconds
            + source_wall_seconds
            + build_wall_seconds
            + solve_wall_seconds
        ),
        extra={
            "source_id": source_id,
            "source_provenance": dict(source_provenance),
            "continuum_equation": equation.continuum_model,
            "continuum_method_id": equation.method_id,
            "surface_size": response.surface_size,
            "continuum_build_wall_seconds": build_wall_seconds,
            "continuum_solve_wall_seconds": solve_wall_seconds,
            "shared_qm_scf_and_cds_wall_seconds": shared_wall_seconds,
            "source_construction_wall_seconds": source_wall_seconds,
            "half_coupling_identity_error_ev": identity_error_ev,
            "runtime_provenance": dict(response.runtime_provenance),
        },
    )


def _run_record(
    selected: MNSolPilotSelection,
    baseline_record: Mapping[str, Any],
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
    radii = route2_coulomb_radii(
        symbols,
        solvent=solvent,
        profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
    )
    frozen_radii = np.asarray(
        source_record["cavity_radii_angstrom"],
        dtype=float,
    )
    if not np.array_equal(radii, frozen_radii):
        raise ValueError("Current cavity radii drifted from the frozen source.")

    cds_started = time.perf_counter()
    cds = pyscf_smd_cds(symbols, positions, solvent=solvent)
    cds_wall_seconds = time.perf_counter() - cds_started
    frozen_cds = _validated_frozen_cds(
        source_record,
        current_cds_kcal_mol=float(cds.energy_kcal_mol),
    )

    dielectric = float(route2_solvent_spec(solvent).descriptors.dielectric)
    responses: dict[str, Any] = {}
    build_times: dict[str, float] = {}
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for equation in CONTINUUM_EQUATIONS:
        started = time.perf_counter()
        response = equation.response_type(
            symbols,
            positions,
            radii,
            dielectric=dielectric,
            lebedev_order=PYSCF_LEBEDEV_ORDER,
        )
        build_times[equation.method_id] = time.perf_counter() - started
        responses[equation.method_id] = response
        surfaces[equation.method_id] = (
            response.surface_points_bohr,
            response.surface_areas_bohr2,
        )
    surface_parity = _validate_same_surface(surfaces)
    reference_points = responses[
        CONTINUUM_EQUATIONS[0].method_id
    ].surface_points_bohr

    qm_state = _run_qm_density(atoms)
    ri_fit = _fit_salted_ri_density(qm_state)
    ao_source, ao_mep_wall_seconds = _ao_surface_mep(
        qm_state,
        reference_points,
    )
    ri_started = time.perf_counter()
    ri_result = salted_ri_surface_mep(
        symbols=symbols,
        atom_positions_angstrom=positions,
        surface_points_bohr=reference_points,
        salted_coefficients=ri_fit.salted_coefficients,
        auxiliary_basis=AUXILIARY_BASIS,
        source_model=(
            f"qm-ri-representation-oracle:{QM_FUNCTIONAL}/{QM_BASIS}"
        ),
        declared_total_charge_e=0.0,
        charge_tolerance_e=RI_PROJECTED_CHARGE_TOLERANCE_E,
        charge_constraint="coulomb-metric",
        point_chunk_size=MEP_POINT_CHUNK_SIZE,
    )
    ri_mep_wall_seconds = time.perf_counter() - ri_started
    raw_ri_charge_residual = abs(
        qm_state.target_electron_count_e - ri_result.raw_electron_count_e
    )
    projected_ri_charge_residual = abs(ri_result.observed_total_charge_e)
    if raw_ri_charge_residual > RI_RAW_CHARGE_TOLERANCE_E:
        raise RuntimeError(
            "The unprojected RI density violates the preregistered charge gate "
            f"({raw_ri_charge_residual:.3e} e)."
        )
    if projected_ri_charge_residual > RI_PROJECTED_CHARGE_TOLERANCE_E:
        raise RuntimeError("The projected RI density violates its charge gate.")

    mep_difference = (
        ri_result.surface_potential_hartree_per_e
        - ao_source.surface_potential_hartree_per_e
    )
    representation = {
        "ao_to_ri_surface_mep_rmse_hartree_per_e": float(
            np.sqrt(np.mean(np.square(mep_difference)))
        ),
        "ao_to_ri_surface_mep_maximum_absolute_difference_hartree_per_e": (
            float(np.max(np.abs(mep_difference)))
        ),
        "ri_raw_charge_residual_e": raw_ri_charge_residual,
        "ri_projected_charge_residual_e": projected_ri_charge_residual,
        "ri_charge_correction_coulomb_norm": (
            ri_result.charge_correction_coulomb_norm
        ),
        "ri_maximum_absolute_coefficient_correction": (
            ri_result.max_abs_coefficient_correction
        ),
    }

    methods: dict[str, object] = {}
    baseline_methods = baseline_record["methods"]
    shared_qm_and_cds = qm_state.scf_wall_seconds + cds_wall_seconds
    for equation in CONTINUUM_EQUATIONS:
        baseline_id = f"mace_fixed_l1__{equation.method_id}"
        baseline = dict(baseline_methods[baseline_id])
        if baseline["surface_size"] != responses[equation.method_id].surface_size:
            raise RuntimeError("Current PCM surface size drifted from baseline.")
        if not math.isclose(
            float(baseline["smd_cds_kcal_mol"]),
            frozen_cds,
            rel_tol=0.0,
            abs_tol=FROZEN_CDS_TOLERANCE_KCAL_MOL,
        ):
            raise RuntimeError("Baseline SMD-CDS drifted.")
        baseline["result_origin"] = "immutable-baseline-artifact"
        methods[baseline_id] = baseline

        response = responses[equation.method_id]
        methods[f"qm_ao_frozen__{equation.method_id}"] = (
            _solve_full_density(
                response,
                ao_source,
                source_id="qm_ao_frozen",
                equation=equation,
                experiment_kcal_mol=experiment,
                cds_kcal_mol=frozen_cds,
                shared_wall_seconds=shared_qm_and_cds,
                build_wall_seconds=build_times[equation.method_id],
                source_wall_seconds=ao_mep_wall_seconds,
                source_provenance=ao_source.provenance,
            )
        )
        methods[f"qm_ri_salted_space_frozen__{equation.method_id}"] = (
            _solve_full_density(
                response,
                ri_result.source,
                source_id="qm_ri_salted_space_frozen",
                equation=equation,
                experiment_kcal_mol=experiment,
                cds_kcal_mol=frozen_cds,
                shared_wall_seconds=shared_qm_and_cds,
                build_wall_seconds=build_times[equation.method_id],
                source_wall_seconds=(
                    ri_fit.fit_wall_seconds + ri_mep_wall_seconds
                ),
                source_provenance={
                    **ri_result.source.provenance,
                    "scientific_status": (
                        "SALTED-compatible RI representation upper bound; "
                        "not learned SALTED inference"
                    ),
                },
            )
        )
    if tuple(methods) != METHOD_IDS:
        raise RuntimeError("Full-density method order drifted.")

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
        "qm_density": {
            "method": "omegaB97M-V",
            "basis": QM_BASIS,
            "density_fit": True,
            "auxiliary_basis": AUXILIARY_BASIS,
            "energy_hartree": qm_state.energy_hartree,
            "electron_count_e": qm_state.electron_count_e,
            "charge_residual_e": qm_state.charge_residual_e,
            "scf_cycles": qm_state.scf_cycles,
            "scf_wall_seconds": qm_state.scf_wall_seconds,
            "semilocal_grid_point_count": (
                qm_state.semilocal_grid_point_count
            ),
            "nonlocal_grid_point_count": (
                qm_state.nonlocal_grid_point_count
            ),
        },
        "representation_diagnostics": representation,
        "timing_seconds": {
            "shared_smd_cds": cds_wall_seconds,
            "qm_scf": qm_state.scf_wall_seconds,
            "ri_fit": ri_fit.fit_wall_seconds,
            "ao_surface_mep": ao_mep_wall_seconds,
            "ri_surface_mep": ri_mep_wall_seconds,
            "continuum_build_by_equation": build_times,
        },
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


def _representation_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    diagnostics = [
        record["representation_diagnostics"] for record in records
    ]
    rmse = [
        float(item["ao_to_ri_surface_mep_rmse_hartree_per_e"])
        for item in diagnostics
    ]
    maximum = [
        float(
            item[
                "ao_to_ri_surface_mep_maximum_absolute_difference_hartree_per_e"
            ]
        )
        for item in diagnostics
    ]
    raw_charge = [
        float(item["ri_raw_charge_residual_e"]) for item in diagnostics
    ]
    projected_charge = [
        float(item["ri_projected_charge_residual_e"])
        for item in diagnostics
    ]
    correction_norm = [
        float(item["ri_charge_correction_coulomb_norm"])
        for item in diagnostics
    ]
    coefficient_correction = [
        float(item["ri_maximum_absolute_coefficient_correction"])
        for item in diagnostics
    ]
    return {
        "record_count": len(records),
        "mean_AO_to_RI_surface_MEP_RMSE_hartree_per_e": float(
            np.mean(rmse)
        ),
        "maximum_AO_to_RI_surface_MEP_RMSE_hartree_per_e": max(rmse),
        "maximum_AO_to_RI_surface_MEP_absolute_difference_hartree_per_e": (
            max(maximum)
        ),
        "mean_RI_raw_charge_residual_e": float(np.mean(raw_charge)),
        "maximum_RI_raw_charge_residual_e": max(raw_charge),
        "maximum_RI_projected_charge_residual_e": max(projected_charge),
        "mean_RI_charge_correction_Coulomb_norm": float(
            np.mean(correction_norm)
        ),
        "maximum_RI_charge_correction_Coulomb_norm": max(correction_norm),
        "maximum_RI_absolute_coefficient_correction": max(
            coefficient_correction
        ),
    }


def _source_hashes() -> dict[str, str]:
    relative_paths = (
        "maple/function/calculator/extra_correction/implicit/salted_ri_density.py",
        (
            "maple/function/calculator/extra_correction/implicit/source/"
            "frozen_density_mep.py"
        ),
        (
            "maple/function/calculator/extra_correction/implicit/"
            "pyscf_swig_response.py"
        ),
        (
            "maple/function/calculator/extra_correction/implicit/"
            "pyscf_smd_cds.py"
        ),
        "maple/function/calculator/extra_correction/implicit/smd_cds.py",
        "maple/function/route2_smd_profiles.py",
        "maple/function/route2_solvents.py",
        "docs/implicit-solvation/benchmarks/benchmark_core.py",
        "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
        "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
        "docs/implicit-solvation/benchmarks/mnsol_response_ablation.py",
        (
            "docs/implicit-solvation/benchmarks/"
            "run_mnsol_fixed_source_pyscf_pcm_family.py"
        ),
        (
            "docs/implicit-solvation/benchmarks/"
            "route2-mnsol-frozen-full-density-pcm-family-prereg-v1.json"
        ),
        (
            "docs/implicit-solvation/benchmarks/"
            "run_mnsol_frozen_full_density_pcm_family.py"
        ),
    )
    return {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in relative_paths
    }


def _private_artifact(
    *,
    execution_git_head: str,
    preregistration: Mapping[str, Any],
    protocol_fingerprint: str,
    selection_fingerprint: str,
    dataset_hashes: Mapping[str, str | None],
    records: Sequence[Mapping[str, Any]],
    status: str,
    complete_panel: bool,
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
        "selection_fingerprint": selection_fingerprint,
        "dataset": dict(dataset_hashes),
        "frozen_baseline_artifact": dict(
            preregistration["frozen_baseline_artifact"]
        ),
        "frozen_source_artifact": dict(
            preregistration["frozen_source_artifact"]
        ),
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
    representation_summary: Mapping[str, object],
    total_wall_seconds: float,
) -> dict[str, object]:
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
                "mace_fixed_l1": (
                    "immutable gas MACE-POLAR l<=1 baseline ledger"
                ),
                "qm_ao_frozen": (
                    "fixed gas omegaB97M-V/def2-SVPD DF-RKS AO density"
                ),
                "qm_ri_salted_space_frozen": (
                    "the same QM density in a charge-constrained "
                    "SALTED-compatible def2-SVP-JKFIT RI basis; "
                    "representation upper bound, not learned SALTED"
                ),
            },
            "continuum_provider": f"PySCF {pyscf.__version__} SWIG",
            "continuum_equations": [
                equation.continuum_model
                for equation in CONTINUUM_EQUATIONS
            ],
            "shared_cavity": (
                "per-solvent PySCF-2.13.1 SMD Coulomb radii"
            ),
            "shared_nonpolar_model": "PySCF 2.13.1 SMD-CDS",
            "strict_original_smd_equivalence": False,
            "learned_salted_prediction_included": False,
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
        },
        "selection": {
            "artifact_sha256": preregistration["selection"][
                "artifact_sha256"
            ],
            "fingerprint": selection_manifest["selection_fingerprint"],
            "record_count": len(records),
            "full_preregistered_record_count": FULL_PANEL_RECORD_COUNT,
            "selection_indices": [
                int(record["selection_index"]) for record in records
            ],
            "solvent_count": len(solvents),
            "solvents": solvents,
            "functional_group_coverage": preregistration["selection"][
                "functional_group_coverage"
            ],
            "used_existing_experimental_values": True,
            "used_new_AO_or_RI_results_at_registration": False,
        },
        "aggregate_metrics": dict(aggregate_metrics),
        "paired_method_comparisons": dict(paired_comparisons),
        "representation_summary": dict(representation_summary),
        "numerics": {
            "pyscf_version": str(pyscf.__version__),
            "electronic_structure_method": "omegaB97M-V",
            "electronic_structure_basis": QM_BASIS,
            "auxiliary_basis": AUXILIARY_BASIS,
            "surface_discretization_method": "SWIG",
            "lebedev_order": PYSCF_LEBEDEV_ORDER,
            "ri_charge_constraint": "coulomb-metric",
        },
        "timing_seconds": {
            "actual_total_wall": total_wall_seconds,
            "status": "metadata-only-not-a-randomized-speed-ranking",
            "warning": (
                "QM AO/RI timings include gas-phase QM density generation. "
                "They are representation-oracle costs and are not learned "
                "SALTED inference timings. Baseline MACE ledgers exclude "
                "their frozen checkpoint inference."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "pyscf": str(pyscf.__version__),
            "salted_runtime_imported": False,
            "threads": lib.num_threads(),
        },
        "salted_upstream": preregistration["salted_upstream"],
        "source_files_sha256": _source_hashes(),
        "references": preregistration["references"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--frozen-source-artifact", type=Path, required=True)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
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
        help="Run one selected record as a private engineering smoke.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    preregistration = _validate_preregistration(arguments.preregistration)
    execution_git_head = _execution_git_head()
    lib.num_threads(QM_THREADS)
    protocol = load_mnsol_protocol(arguments.protocol)
    dataset = load_mnsol_v2012(arguments.source, protocol)
    selection_manifest = _load_json_mapping(
        arguments.selection,
        label="MNSol pilot selection",
    )
    full_selection = validate_frozen_mnsol_pilot_selection(
        selection_manifest,
        dataset,
        protocol,
    )
    indexed_selection, complete_panel = _indexed_selection(
        full_selection,
        arguments.record_index,
    )
    private_output, public_output = _validated_output_paths(
        private_output=arguments.private_output,
        public_output=arguments.public_output,
        complete_panel=complete_panel,
    )
    dataset_hashes = {
        "source_artifact_sha256": dataset.source_artifact_sha256,
        "table_sha256": dataset.table_sha256,
        "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
    }
    baseline_records, source_records = _load_bound_artifacts(
        preregistration=preregistration,
        baseline_path=arguments.baseline_artifact,
        source_path=arguments.frozen_source_artifact,
        selection=full_selection,
        protocol_fingerprint=protocol.fingerprint,
        selection_fingerprint=selection_manifest["selection_fingerprint"],
        dataset_hashes=dataset_hashes,
    )
    experimental_checks = _validate_experimental_selection(
        [selected for _, selected in indexed_selection],
        temperature_k=protocol.temperature_k,
        standard_state=protocol.standard_state,
    )

    wall_started = time.perf_counter()
    records: list[dict[str, object]] = []
    for ordinal, (index, selected) in enumerate(indexed_selection, start=1):
        print(
            f"[{ordinal}/{len(indexed_selection)}] index={index} "
            f"solute={selected.eligible_record.record.solute_name} "
            f"solvent={selected.canonical_solvent}",
            flush=True,
        )
        try:
            result = _run_record(
                selected,
                baseline_records[index],
                source_records[index],
                selection_index=index,
            )
        except Exception as exc:
            failure = _private_artifact(
                execution_git_head=execution_git_head,
                preregistration=preregistration,
                protocol_fingerprint=protocol.fingerprint,
                selection_fingerprint=selection_manifest[
                    "selection_fingerprint"
                ],
                dataset_hashes=dataset_hashes,
                records=records,
                status="failed",
                complete_panel=complete_panel,
            )
            failure["failure"] = {
                "selection_index": index,
                "stage": "frozen-full-density-record",
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
                selection_fingerprint=selection_manifest[
                    "selection_fingerprint"
                ],
                dataset_hashes=dataset_hashes,
                records=records,
                status="running",
                complete_panel=complete_panel,
            ),
        )

    aggregate_metrics = _aggregate_metrics(records)
    paired_comparisons = _paired_comparisons(records)
    representation_summary = _representation_summary(records)
    total_wall_seconds = time.perf_counter() - wall_started
    private = _private_artifact(
        execution_git_head=execution_git_head,
        preregistration=preregistration,
        protocol_fingerprint=protocol.fingerprint,
        selection_fingerprint=selection_manifest["selection_fingerprint"],
        dataset_hashes=dataset_hashes,
        records=records,
        status="complete",
        complete_panel=complete_panel,
    )
    private.update(
        {
            "aggregate_metrics": aggregate_metrics,
            "paired_method_comparisons": paired_comparisons,
            "representation_summary": representation_summary,
            "actual_total_wall_seconds": total_wall_seconds,
        }
    )
    write_json_atomic(private_output, private)
    write_json_atomic(
        public_output,
        _public_artifact(
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
            representation_summary=representation_summary,
            total_wall_seconds=total_wall_seconds,
        ),
    )
    print(
        f"Wrote {len(records)} record(s) to {private_output} and "
        f"{public_output}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
