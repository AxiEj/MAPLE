#!/usr/bin/env python3
"""Evaluate the preregistered V0-ARSP permanent source at frozen QM MEP points.

This helper is deliberately an array-level source falsifier.  It builds the
all-electron stationary atomic-response source, then compares its permanent
vacuum potential with one hash-pinned QM checkpoint.  It does not construct a
PCM cavity, solve a continuum, read an experimental solvation label, or make a
runtime/accuracy claim.  The parent runner owns source hashes, preregistration,
and the pass/reject decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.route2_v0_atomic_independent_particle_response import (  # noqa: E402
    load_route2_v0_atomic_independent_particle_response_table,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_atomic_response_stationary_source import (  # noqa: E402
    V0_ATOMIC_RESPONSE_DENSITY_GRID_BUFFER_BOHR,
    V0_ATOMIC_RESPONSE_DENSITY_GRID_SPACING_BOHR,
    build_route2_v0_atomic_response_pyscf_reference,
    solve_route2_v0_atomic_response_stationary_state,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_response_kernel import (  # noqa: E402
    complete_route2_v0_response_kernel,
)

EXPECTED_PYSCF_VERSION = "2.13.1"
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Cl": 17}
EXPECTED_SYMBOLS = ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic-table", type=Path, required=True)
    parser.add_argument("--atomic-manifest", type=Path, required=True)
    parser.add_argument("--mace-atomic-map", type=Path, required=True)
    parser.add_argument("--mace-response", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--qm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _immutable_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array.") from error
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must have {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be exactly one JSON object.")
    return payload


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1.0e-30)
    )


def _relative_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(np.abs(left - right), initial=0.0)
        / max(float(np.max(np.abs(right), initial=0.0)), 1.0e-30)
    )


def _load_frozen_mace_inputs(
    *,
    atomic_map_path: Path,
    response_path: Path,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    atomic = _load_json(atomic_map_path, label="frozen MACE-MDP atom partition")
    response = _load_json(response_path, label="frozen MACE-MDP response")
    if (
        atomic.get("artifact") != "route2-v0-mace-mdp-atomic-map-acetone-v1"
        or atomic.get("status") != "pass"
        or atomic.get("decision", {}).get("verdict")
        != "admit-atomic-moment-partition-for-source-map-gates-only"
        or response.get("artifact") != "route2-v0-mace-mdp-acetone-response-v1"
        or response.get("status") != "pass"
        or response.get("scientific_falsification", {}).get("verdict")
        != "admit-frozen-mace-mdp-response-coefficients-only"
    ):
        raise RuntimeError("Frozen MACE-MDP moment inputs are not admissible.")
    atomic_system = atomic.get("system")
    response_system = response.get("system")
    decomposition = atomic.get("mace_mdp_atomic_decomposition")
    response_values = response.get("mace_mdp_response")
    if not all(
        isinstance(value, dict)
        for value in (atomic_system, response_system, decomposition, response_values)
    ):
        raise RuntimeError("Frozen MACE-MDP moment inputs are incomplete.")
    symbols = tuple(atomic_system.get("atom_symbols", ()))
    positions = _immutable_array(
        atomic_system.get("positions_angstrom"),
        name="frozen MACE-MDP positions",
    )
    response_positions = _immutable_array(
        response_system.get("positions_angstrom"),
        name="frozen MACE-MDP response positions",
    )
    weights = _immutable_array(
        decomposition.get("atomic_dipole_weights"),
        name="frozen MACE-MDP atomic dipole partition",
    )
    polarizability = _immutable_array(
        response_values.get("canonical_polarizability_bohr3"),
        name="frozen canonical MACE-MDP polarizability",
    )
    raw_polarizability = _immutable_array(
        response_values.get("polarizability_bohr3"),
        name="frozen raw MACE-MDP polarizability",
    )
    if (
        symbols != EXPECTED_SYMBOLS
        or positions.shape != (10, 3)
        or response_positions.shape != positions.shape
        or not np.array_equal(response_positions, positions)
        or weights.shape != (10, 3, 3)
        or polarizability.shape != (3, 3)
        or raw_polarizability.shape != (3, 3)
        or _relative_frobenius(polarizability, raw_polarizability) > 1.0e-14
        or any(symbol not in ATOMIC_NUMBERS for symbol in symbols)
    ):
        raise RuntimeError("Frozen MACE-MDP acetone moment inputs are invalid.")
    return symbols, positions, weights, polarizability


def _load_points(path: Path) -> np.ndarray:
    try:
        points = _immutable_array(
            np.load(path, allow_pickle=False),
            name="frozen exterior QM-MEP points",
        )
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot load frozen exterior QM-MEP points.") from error
    if points.shape != (516, 3):
        raise RuntimeError("The stationary-source gate requires all 516 frozen points.")
    return points


def _checkpoint_density(checkpoint: Path):
    try:
        import pyscf
        from pyscf import lib
    except ImportError as error:
        raise RuntimeError(
            "PySCF is required only for this static-MEP helper."
        ) from error
    if pyscf.__version__ != EXPECTED_PYSCF_VERSION:
        raise RuntimeError(
            "The static-MEP helper requires the preregistered PySCF version."
        )
    molecule = lib.chkfile.load_mol(str(checkpoint))
    try:
        energy = float(lib.chkfile.load(str(checkpoint), "scf/e_tot"))
        coefficients = _immutable_array(
            lib.chkfile.load(str(checkpoint), "scf/mo_coeff"),
            name="QM checkpoint MO coefficients",
        )
        occupations = _immutable_array(
            lib.chkfile.load(str(checkpoint), "scf/mo_occ"),
            name="QM checkpoint MO occupations",
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("QM checkpoint is missing its SCF state.") from error
    nao = molecule.nao_nr()
    if (
        not np.isfinite(energy)
        or coefficients.shape != (nao, nao)
        or occupations.shape != (nao,)
        or np.any(occupations < 0.0)
        or np.any(occupations > 2.0)
        or np.any(np.minimum(np.abs(occupations), np.abs(occupations - 2.0)) > 1.0e-8)
    ):
        raise RuntimeError("QM checkpoint is not a closed-shell AO state.")
    overlap = molecule.intor_symmetric("int1e_ovlp")
    metric_error = float(
        np.linalg.norm(coefficients.T @ overlap @ coefficients - np.eye(nao), ord=2)
    )
    density = (coefficients * occupations) @ coefficients.T
    density = 0.5 * (density + density.T)
    electron_count = float(np.einsum("ij,ji->", density, overlap, optimize=True))
    total_charge = float(np.sum(molecule.atom_charges()) - electron_count)
    return (
        molecule,
        density,
        {
            "ao_count": int(nao),
            "energy_hartree": energy,
            "mo_metric_error": metric_error,
            "electron_count_e": electron_count,
            "total_charge_e": total_charge,
        },
    )


def _validate_common_geometry(
    *,
    molecule,
    atomic_numbers: np.ndarray,
    positions_angstrom: np.ndarray,
) -> float:
    checkpoint_numbers = np.asarray(molecule.atom_charges(), dtype=int)
    checkpoint_positions_bohr = np.asarray(molecule.atom_coords(), dtype=float)
    source_positions_bohr = (
        np.asarray(positions_angstrom, dtype=float) / 0.5291772105638411
    )
    if not np.array_equal(checkpoint_numbers, atomic_numbers):
        raise RuntimeError("Stationary source and QM checkpoint atomic numbers differ.")
    if checkpoint_positions_bohr.shape != source_positions_bohr.shape:
        raise RuntimeError("Stationary source and QM checkpoint atom counts differ.")
    coordinate_error = float(
        np.max(np.abs(checkpoint_positions_bohr - source_positions_bohr), initial=0.0)
    )
    if coordinate_error > 1.0e-8:
        raise RuntimeError("Stationary source and QM checkpoint geometries differ.")
    return coordinate_error


def _total_potential_hartree_per_e(
    *,
    molecule,
    density: np.ndarray,
    points_bohr: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(molecule.atom_coords(), dtype=float)
    charges = np.asarray(molecule.atom_charges(), dtype=float)
    potential = np.empty(len(points_bohr), dtype=float)
    for index, point in enumerate(points_bohr):
        distances = np.linalg.norm(positions - point, axis=1)
        if np.any(distances <= 1.0e-12):
            raise ValueError("A frozen MEP point coincides with a source nucleus.")
        nuclear = float(np.dot(charges, 1.0 / distances))
        with molecule.with_rinv_origin(point):
            inverse_distance = molecule.intor("int1e_rinv")
        potential[index] = nuclear - float(
            np.einsum("ij,ij->", density, inverse_distance, optimize=True)
        )
    return _immutable_array(
        potential,
        name="QM total permanent potential",
        shape=(len(points_bohr),),
    )


def _total_dipole_e_bohr(*, molecule, density: np.ndarray) -> np.ndarray:
    positions = np.asarray(molecule.atom_coords(), dtype=float)
    charges = np.asarray(molecule.atom_charges(), dtype=float)
    nuclear = np.einsum("a,ax->x", charges, positions, optimize=True)
    position_integrals = molecule.intor_symmetric("int1e_r", comp=3)
    electronic = np.einsum("xij,ij->x", position_integrals, density, optimize=True)
    return _immutable_array(
        nuclear - electronic,
        name="QM total permanent dipole",
        shape=(3,),
    )


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    table_path = arguments.atomic_table.resolve()
    manifest_path = arguments.atomic_manifest.resolve()
    atomic_map_path = arguments.mace_atomic_map.resolve()
    response_path = arguments.mace_response.resolve()
    points_path = arguments.points.resolve()
    checkpoint_path = arguments.qm_checkpoint.resolve()
    source_table = load_route2_v0_atomic_independent_particle_response_table(
        table_path=table_path,
        manifest_path=manifest_path,
    )
    symbols, positions, atomic_weights, polarizability = _load_frozen_mace_inputs(
        atomic_map_path=atomic_map_path,
        response_path=response_path,
    )
    atomic_numbers = np.asarray([ATOMIC_NUMBERS[symbol] for symbol in symbols])
    points = _load_points(points_path)
    reference = build_route2_v0_atomic_response_pyscf_reference(
        response_table=source_table,
        atomic_numbers=atomic_numbers,
        atom_positions_angstrom=positions,
    )
    completion = complete_route2_v0_response_kernel(
        baseline_response_covariance_coefficient_dual=(
            reference.baseline.baseline_response_covariance_coefficient_dual
        ),
        atom_dipole_map_coefficient_to_ebohr=(
            reference.baseline.atom_dipole_map_coefficient_to_ebohr
        ),
        atomic_dipole_partition_molecular_to_ebohr=atomic_weights.reshape(30, 3),
        molecular_polarizability_bohr3=polarizability,
        charge_constraint_vector=None,
    )
    reference_dual = reference.reference_coefficient_dual_hartree()
    state = solve_route2_v0_atomic_response_stationary_state(
        completion=completion,
        reference_coefficient_dual_hartree=reference_dual,
    )
    density_audit = reference.permanent_density_audit(state)
    density_grid = reference.density_positivity_grid_bohr()
    density_values = reference.electron_number_density_e_per_bohr3(
        density_audit,
        density_grid,
    )
    candidate_potential = reference.total_permanent_potential_hartree_per_e(
        density_audit,
        points,
    )
    candidate_dipole = reference.total_permanent_dipole_e_bohr(density_audit)
    qm_molecule, qm_density, qm_audit = _checkpoint_density(checkpoint_path)
    geometry_error = _validate_common_geometry(
        molecule=qm_molecule,
        atomic_numbers=atomic_numbers,
        positions_angstrom=positions,
    )
    qm_potential = _total_potential_hartree_per_e(
        molecule=qm_molecule,
        density=qm_density,
        points_bohr=points,
    )
    qm_dipole = _total_dipole_e_bohr(molecule=qm_molecule, density=qm_density)
    artifact = {
        "artifact": "route2-v0-atomic-response-stationary-source-acetone-static-mep-helper-v1",
        "schema_version": 1,
        "candidate": {
            "construction": "route2-v0-atomic-response-stationary-permanent-source-v1",
            "coefficient_dual_pairing": (
                "b_m=integral tau_m[-V_nuc,other+J[n_other]]; "
                "x0=-C b; n0=n_ref+sum_m x0_m tau_m"
            ),
            "permanent_source": "all-electron nuclei minus stationary AO electron density",
            "response_completion": "frozen V0-RK MACE-MDP moment covariance only",
            "response_or_continuum": "not invoked",
            "table_sha256": source_table.table_sha256,
            "table_manifest_sha256": source_table.manifest_sha256,
        },
        "input_sha256": {
            "atomic_table": _sha256(table_path),
            "atomic_manifest": _sha256(manifest_path),
            "mace_atomic_map": _sha256(atomic_map_path),
            "mace_response": _sha256(response_path),
            "points": _sha256(points_path),
            "qm_checkpoint": _sha256(checkpoint_path),
        },
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
            "source_point_count": int(len(points)),
            "source_points_sha256": _sha256_array(points),
        },
        "stationary_reference": {
            "ao_count": int(reference.molecule.nao_nr()),
            "coefficient_count": int(reference.coefficient_count),
            "atomic_mo_metric_errors": reference.atomic_mo_metric_errors.tolist(),
            "reference_electron_count_e": reference.reference_electron_count_e,
            "reference_electron_count_error_e": abs(
                float(
                    np.einsum(
                        "ij,ji->",
                        reference.ground_density_matrix,
                        reference.overlap_matrix,
                        optimize=True,
                    )
                )
                - reference.reference_electron_count_e
            ),
            "reference_coefficient_dual_sha256": _sha256_array(reference_dual),
            "permanent_response_coefficients_sha256": _sha256_array(
                state.permanent_response_coefficients
            ),
            "electronic_correction_energy_hartree": state.electronic_correction_energy_hartree,
            "stationary_energy_identity_error_hartree": state.stationary_energy_identity_error_hartree,
            "stationarity_residual_inf_hartree": state.stationarity_residual_inf_hartree,
            "support_constraint_residual_inf": state.support_constraint_residual_inf,
            "support_minimum_curvature_hartree": state.support_minimum_curvature_hartree,
            "stationarity_tolerance_hartree": state.stationarity_tolerance_hartree,
        },
        "permanent_density": {
            "electron_count_e": density_audit.electron_count_e,
            "reference_electron_count_e": density_audit.reference_electron_count_e,
            "electron_count_error_e": density_audit.electron_count_error_e,
            "minimum_ao_metric_density_eigenvalue": (
                density_audit.minimum_ao_metric_density_eigenvalue
            ),
            "maximum_ao_metric_density_eigenvalue": (
                density_audit.maximum_ao_metric_density_eigenvalue
            ),
            "density_symmetry_error": density_audit.density_symmetry_error,
            "density_matrix_sha256": _sha256_array(density_audit.density_matrix),
            "density_positivity_grid": {
                "buffer_bohr": V0_ATOMIC_RESPONSE_DENSITY_GRID_BUFFER_BOHR,
                "spacing_bohr": V0_ATOMIC_RESPONSE_DENSITY_GRID_SPACING_BOHR,
                "point_count": int(len(density_grid)),
                "points_sha256": _sha256_array(density_grid),
                "minimum_electron_number_density_e_per_bohr3": float(
                    np.min(density_values)
                ),
                "maximum_electron_number_density_e_per_bohr3": float(
                    np.max(density_values)
                ),
            },
            "candidate_dipole_e_bohr": candidate_dipole.tolist(),
            "candidate_potential_sha256": _sha256_array(candidate_potential),
        },
        "qm_checkpoint": {
            **qm_audit,
            "geometry_max_abs_error_bohr": geometry_error,
            "permanent_dipole_e_bohr": qm_dipole.tolist(),
            "potential_sha256": _sha256_array(qm_potential),
        },
        "static_comparison": {
            "mep_relative_frobenius": _relative_frobenius(
                candidate_potential, qm_potential
            ),
            "mep_relative_max_abs": _relative_max_abs(
                candidate_potential, qm_potential
            ),
            "dipole_relative_frobenius": _relative_frobenius(
                candidate_dipole, qm_dipole
            ),
        },
        "runtime": {
            "python": sys.version,
            "python_resolved_sha256": _sha256(Path(sys.executable).resolve()),
            "pyscf_version": __import__("pyscf").__version__,
            "boundary": (
                "Frozen gas-phase permanent-source falsifier only; no PCM, "
                "solvation energy, force, experimental label, or Route-2/QM speed claim."
            ),
        },
    }
    _write_exclusive_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
