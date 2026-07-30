#!/usr/bin/env python3
"""Falsify a frozen variational V0 source against acetone finite-field QM MEP.

The source space is a direct sum of neutral transition densities from frozen,
spherical atomic Hartree--Fock calculations.  Its positive independent-particle
covariance is completed only by the already frozen MACE-MDP *dipole*
polarizability and its audited atom-dipole partition.  This is the exact
no-training V0-RK construction; it does not identify a molecular density from
an error target.

At each of 516 immutable exterior points the script evaluates the induced
*electronic* potential with the same transition-density AO integrals used to
define the source.  It compares that array with independent fixed-geometry QM
finite-field data.  A pass is a narrow gas-phase source-admission result only:
it establishes neither a continuum, a force/PES, nonpolar free energy, nor any
experimental solvation accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ID = "route2-v0-atomic-independent-particle-source-acetone-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-independent-particle-source-acetone-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_atomic_independent_particle_source_acetone.py"
)
ATOMIC_SOURCE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_independent_particle_response.py"
)
RESPONSE_KERNEL_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_response_kernel.py"
)
TABLE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1.npz"
)
TABLE_MANIFEST_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1.json"
)
TABLE_PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-independent-particle-hf-def2-tzvpd-prereg-v1.json"
)
ATOMIC_MAP_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-mace-mdp-atomic-map-acetone-v1.json"
)
RESPONSE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-mace-mdp-acetone-response-v1.json"
)
RAW_QM_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-mace-mdp-induced-source-acetone-v1/qm-induced-mep.json"
)
POINTS_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-atomic-displacement-source-acetone-v1/"
    "frozen-exterior-qm-mep-points-bohr.npy"
)
POINTS_PROVENANCE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-atomic-displacement-source-acetone-v1/"
    "frozen-exterior-qm-mep-points-provenance.json"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    ATOMIC_SOURCE_MODULE_RELATIVE_PATH,
    RESPONSE_KERNEL_MODULE_RELATIVE_PATH,
)
INPUT_RELATIVE_PATHS = (
    TABLE_RELATIVE_PATH,
    TABLE_MANIFEST_RELATIVE_PATH,
    TABLE_PREREG_RELATIVE_PATH,
    ATOMIC_MAP_RELATIVE_PATH,
    RESPONSE_RELATIVE_PATH,
    RAW_QM_RELATIVE_PATH,
    POINTS_RELATIVE_PATH,
    POINTS_PROVENANCE_RELATIVE_PATH,
)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH
DIRECTIONS = ("x", "y", "z")
FIELD_STEPS = (3.0e-4, 1.0e-3)
SELECTED_FIELD_STEP = 3.0e-4
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Cl": 17}
EXPECTED_QM_METHOD = {
    "electronic_structure": "omegaB97M-V",
    "pyscf_xc_token": "wb97m-v",
    "basis": "def2-tzvpd",
    "reference": "RKS",
    "density_fitting": True,
    "charge": 0,
    "spin": 0,
    "semilocal_grid_level": 3,
    "nonlocal_grid_profile": "50x194-SG1",
    "scf_energy_tolerance_hartree": 1.0e-10,
    "scf_gradient_tolerance": 1.0e-7,
    "maximum_scf_cycles": 100,
}
NUMERICAL_GATE_NAMES = (
    "atomic_partition_moment_identity",
    "maximum_transition_charge",
    "completed_atom_covariance_relative_error",
    "completed_molecular_polarizability_relative_error",
    "completed_moore_penrose_relative_error",
    "support_constraint_response_relative_error",
    "uniform_field_molecular_dipole_relative_error",
    "qm_mep_step_consistency_relative_frobenius",
    "qm_dipole_step_consistency_relative_frobenius",
)
SCIENTIFIC_GATE_KEYS = (
    "mep_response_relative_frobenius_max",
    "mep_response_relative_direction_max",
    "induced_dipole_response_relative_frobenius_max",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain exactly one JSON object.")
    return payload


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _upper_check(value: float, maximum: float) -> dict[str, float | bool]:
    return {"value": value, "maximum": maximum, "passes": bool(value <= maximum)}


def _require_clean_tracked_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The atomic independent-particle QM-MEP falsifier requires a clean "
            f"tracked checkout; git reported:\n{status}"
        )
    for relative in (
        PREREG_RELATIVE_PATH,
        *SOURCE_RELATIVE_PATHS,
        *INPUT_RELATIVE_PATHS,
    ):
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _validate_preregistration() -> tuple[dict[str, Any], dict[str, str], str]:
    preregistration = _load_json(
        DEFAULT_PREREGISTRATION,
        label="atomic independent-particle acetone QM-MEP preregistration",
    )
    if (
        preregistration.get("protocol_id")
        != "route2-v0-atomic-independent-particle-source-acetone-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The atomic independent-particle source protocol is not frozen.")
    source_hashes = {
        relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
    }
    input_hashes = {
        relative: _sha256(REPO_ROOT / relative) for relative in INPUT_RELATIVE_PATHS
    }
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict) or (
        contract.get("source_sha256") != source_hashes
        or contract.get("input_sha256") != input_hashes
    ):
        raise RuntimeError("The frozen atomic independent-particle inputs changed.")
    if preregistration.get("qm_method") != EXPECTED_QM_METHOD:
        raise RuntimeError("The frozen QM-MEP method changed after preregistration.")
    if preregistration.get("finite_field_protocol") != {
        "directions": list(DIRECTIONS),
        "field_steps_au": list(FIELD_STEPS),
        "selected_reporting_step_au": SELECTED_FIELD_STEP,
        "signs": [-1, 1],
    }:
        raise RuntimeError("The frozen QM-MEP finite-field protocol changed.")
    numerical = preregistration.get("numerical_gates")
    scientific = preregistration.get("scientific_falsification_gates")
    if not isinstance(numerical, dict) or not isinstance(scientific, dict):
        raise RuntimeError("The frozen source protocol omits required gates.")
    if set(numerical) != {f"{name}_max" for name in NUMERICAL_GATE_NAMES}:
        raise RuntimeError("The frozen numerical-gate schema changed.")
    if not set(SCIENTIFIC_GATE_KEYS).issubset(scientific):
        raise RuntimeError("The frozen scientific-gate schema changed.")
    return preregistration, input_hashes, _sha256(DEFAULT_PREREGISTRATION)


def _load_module(relative_path: str, *, module_name: str) -> Any:
    path = REPO_ROOT / relative_path
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load frozen source module: {relative_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _load_atomic_table(source_module: Any) -> Any:
    table_path = REPO_ROOT / TABLE_RELATIVE_PATH
    manifest_path = REPO_ROOT / TABLE_MANIFEST_RELATIVE_PATH
    prereg_path = REPO_ROOT / TABLE_PREREG_RELATIVE_PATH
    manifest = _load_json(manifest_path, label="atomic independent-particle table")
    prereg = _load_json(prereg_path, label="atomic independent-particle table prereg")
    if (
        manifest.get("artifact")
        != "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1"
        or manifest.get("status") != "pass"
        or manifest.get("table", {}).get("sha256") != _sha256(table_path)
        or manifest.get("preregistration", {}).get("sha256") != _sha256(prereg_path)
        or prereg.get("protocol_id")
        != "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1-prereg"
    ):
        raise RuntimeError("The atomic independent-particle table provenance is invalid.")
    return source_module.load_route2_v0_atomic_independent_particle_response_table(
        table_path=table_path,
        manifest_path=manifest_path,
    )


def _load_mace_mdp_inputs() -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    atomic = _load_json(
        REPO_ROOT / ATOMIC_MAP_RELATIVE_PATH,
        label="frozen MACE-MDP atomic partition",
    )
    response = _load_json(
        REPO_ROOT / RESPONSE_RELATIVE_PATH,
        label="frozen MACE-MDP response",
    )
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
        raise RuntimeError("The frozen MACE-MDP response inputs are not admissible.")
    atomic_system = atomic.get("system")
    response_system = response.get("system")
    decomposition = atomic.get("mace_mdp_atomic_decomposition")
    response_values = response.get("mace_mdp_response")
    if not all(
        isinstance(value, dict)
        for value in (atomic_system, response_system, decomposition, response_values)
    ):
        raise TypeError("The frozen MACE-MDP response payload is incomplete.")
    symbols = tuple(atomic_system.get("atom_symbols", ()))
    positions = np.asarray(atomic_system.get("positions_angstrom"), dtype=float)
    response_positions = np.asarray(
        response_system.get("positions_angstrom"), dtype=float
    )
    weights = np.asarray(decomposition.get("atomic_dipole_weights"), dtype=float)
    polarizability = np.asarray(
        response_values.get("canonical_polarizability_bohr3"), dtype=float
    )
    raw_polarizability = np.asarray(
        response_values.get("polarizability_bohr3"), dtype=float
    )
    if (
        len(symbols) != 10
        or positions.shape != (10, 3)
        or not np.array_equal(response_positions, positions)
        or weights.shape != (10, 3, 3)
        or polarizability.shape != (3, 3)
        # The source artifact records the canonical scalar-compatible tensor
        # after its raw antisymmetry gate.  JSON round trips retain its
        # sub-ulp difference from the raw float64 record, but a material
        # change must never be silently accepted here.
        or _relative_frobenius(polarizability, raw_polarizability) > 1.0e-14
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(polarizability))
        or any(symbol not in ATOMIC_NUMBERS for symbol in symbols)
    ):
        raise RuntimeError("The frozen MACE-MDP acetone inputs are invalid.")
    return symbols, positions, weights, polarizability


def _load_frozen_qm_mep_points() -> np.ndarray:
    provenance_path = REPO_ROOT / POINTS_PROVENANCE_RELATIVE_PATH
    points_path = REPO_ROOT / POINTS_RELATIVE_PATH
    raw_path = REPO_ROOT / RAW_QM_RELATIVE_PATH
    provenance = _load_json(provenance_path, label="frozen exterior QM-MEP points")
    if (
        provenance.get("artifact")
        != "route2-v0-atomic-displacement-source-acetone-v1-frozen-exterior-qm-mep-points"
        or provenance.get("status") != "pass"
        or provenance.get("origin", {}).get("raw_qm_induced_mep_sha256")
        != _sha256(raw_path)
        or provenance.get("points", {}).get("file_sha256") != _sha256(points_path)
    ):
        raise RuntimeError("The frozen exterior QM-MEP point provenance is invalid.")
    try:
        points = np.asarray(np.load(points_path, allow_pickle=False), dtype=float)
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot load frozen exterior QM-MEP points.") from error
    if (
        points.shape != (516, 3)
        or not np.all(np.isfinite(points))
        or provenance.get("points", {}).get("array_sha256") != _sha256_array(points)
    ):
        raise RuntimeError("The frozen exterior QM-MEP point set is invalid.")
    return points


def _load_qm_responses(
    *,
    points: np.ndarray,
    positions: np.ndarray,
) -> dict[float, dict[str, dict[str, np.ndarray]]]:
    raw_path = REPO_ROOT / RAW_QM_RELATIVE_PATH
    payload = _load_json(raw_path, label="frozen independent QM induced-MEP response")
    if (
        payload.get("status") != "pass"
        or payload.get("method") != EXPECTED_QM_METHOD
        or payload.get("finite_field_protocol")
        != {
            "directions": list(DIRECTIONS),
            "field_steps_au": list(FIELD_STEPS),
            "signs": [-1, 1],
        }
    ):
        raise RuntimeError("The frozen QM induced-MEP response is invalid.")
    raw_input = payload.get("input")
    raw_responses = payload.get("central_difference_responses")
    if not isinstance(raw_input, dict) or not isinstance(raw_responses, dict):
        raise TypeError("The frozen QM induced-MEP response is incomplete.")
    if (
        raw_input.get("source_point_count") != len(points)
        or raw_input.get("source_points_bohr_sha256") != _sha256_array(points)
        or not np.array_equal(
            np.asarray(raw_input.get("positions_angstrom"), dtype=float), positions
        )
    ):
        raise RuntimeError("The frozen QM induced-MEP geometry or point set changed.")
    parsed: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    for step in FIELD_STEPS:
        records = raw_responses.get(f"{step:.1e}")
        if not isinstance(records, dict) or set(records) != set(DIRECTIONS):
            raise RuntimeError("The frozen QM induced-MEP directions are incomplete.")
        parsed[step] = {}
        for direction in DIRECTIONS:
            record = records[direction]
            if not isinstance(record, dict):
                raise TypeError("A frozen QM induced-MEP direction record is invalid.")
            potential = np.asarray(
                record.get("electronic_potential_response_hartree_per_e_per_field_au"),
                dtype=float,
            )
            dipole = np.asarray(
                record.get("molecular_dipole_response_bohr3"), dtype=float
            )
            if (
                potential.shape != (len(points),)
                or dipole.shape != (3,)
                or not np.all(np.isfinite(potential))
                or not np.all(np.isfinite(dipole))
            ):
                raise RuntimeError("A frozen QM induced-MEP response array is invalid.")
            parsed[step][direction] = {"potential": potential, "dipole": dipole}
    return parsed


def _response_tensor(
    responses: dict[float, dict[str, dict[str, np.ndarray]]],
    *,
    step: float,
    observable: str,
) -> np.ndarray:
    if observable not in {"potential", "dipole"}:
        raise ValueError(f"Unsupported frozen QM observable: {observable}.")
    try:
        tensor = np.column_stack(
            [responses[step][direction][observable] for direction in DIRECTIONS]
        )
    except KeyError as error:
        raise RuntimeError("The frozen QM response tensor is incomplete.") from error
    if tensor.ndim != 2 or not np.all(np.isfinite(tensor)):
        raise RuntimeError("The frozen QM response tensor is invalid.")
    return tensor


def _source_potential_mode_matrix(
    *,
    source_table: Any,
    atomic_numbers: np.ndarray,
    positions_angstrom: np.ndarray,
    points_bohr: np.ndarray,
    coefficient_slices: tuple[slice, ...],
) -> np.ndarray:
    """Evaluate ``int tau_m(r) / |r-s| dr`` for every frozen source mode."""

    try:
        from pyscf import gto
    except ImportError as error:
        raise RuntimeError(
            "PySCF is required only to evaluate the frozen AO transition-density "
            "potential at the immutable QM-MEP points."
        ) from error
    bohr_per_angstrom = 1.0 / 0.5291772105638411
    positions_bohr = positions_angstrom * bohr_per_angstrom
    molecules: dict[int, Any] = {}
    transition_densities: dict[int, np.ndarray] = {}
    for number in sorted(set(int(value) for value in atomic_numbers)):
        response = source_table.response_for_atomic_number(number)
        molecules[number] = gto.M(
            atom=f"{response.symbol} 0 0 0",
            basis=response.basis,
            charge=0,
            spin=response.spin_2s,
            unit="Bohr",
            verbose=0,
        )
        transition_densities[number] = response.transition_density_matrices()
    result = np.empty((len(points_bohr), coefficient_slices[-1].stop), dtype=float)
    for atom_index, (number, coefficient_slice) in enumerate(
        zip(atomic_numbers, coefficient_slices, strict=True)
    ):
        molecule = molecules[int(number)]
        transitions = transition_densities[int(number)]
        for point_index, point in enumerate(points_bohr):
            with molecule.with_rinv_origin(point - positions_bohr[atom_index]):
                rinv = molecule.intor("int1e_rinv")
            result[point_index, coefficient_slice] = np.einsum(
                "mij,ij->m", transitions, rinv, optimize=True
            )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Frozen atomic transition potentials are non-finite.")
    return result


def _build_completed_response(
    *,
    source_table: Any,
    response_kernel_module: Any,
    atomic_numbers: np.ndarray,
    atomic_dipole_weights: np.ndarray,
    molecular_polarizability: np.ndarray,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    baseline = source_table.assemble(atomic_numbers)
    completion = response_kernel_module.complete_route2_v0_response_kernel(
        baseline_response_covariance_coefficient_dual=(
            baseline.baseline_response_covariance_coefficient_dual
        ),
        atom_dipole_map_coefficient_to_ebohr=(
            baseline.atom_dipole_map_coefficient_to_ebohr
        ),
        atomic_dipole_partition_molecular_to_ebohr=atomic_dipole_weights.reshape(
            3 * len(atomic_numbers), 3
        ),
        molecular_polarizability_bohr3=molecular_polarizability,
        charge_constraint_vector=None,
    )
    field_dual = baseline.transition_density_dipoles_ebohr
    coefficients_per_field = -(
        completion.response_covariance_coefficient_dual @ field_dual
    )
    atom_sum = np.hstack([np.eye(3) for _ in atomic_numbers])
    molecular_dipole_map = atom_sum @ baseline.atom_dipole_map_coefficient_to_ebohr
    molecular_dipole_per_field = molecular_dipole_map @ coefficients_per_field
    return baseline, completion, coefficients_per_field, molecular_dipole_per_field


def _numerical_checks(
    *,
    baseline: Any,
    completion: Any,
    atomic_dipole_weights: np.ndarray,
    molecular_polarizability: np.ndarray,
    coefficients_per_field: np.ndarray,
    molecular_dipole_per_field: np.ndarray,
    qm_mep: dict[float, np.ndarray],
    qm_dipole: dict[float, np.ndarray],
    numerical_gates: dict[str, Any],
) -> dict[str, dict[str, float | bool]]:
    atom_count = len(baseline.atomic_numbers)
    atom_sum = np.hstack([np.eye(3) for _ in range(atom_count)])
    partition = atomic_dipole_weights.reshape(3 * atom_count, 3)
    support_residual = completion.response_support_constraints @ coefficients_per_field
    return {
        "atomic_partition_moment_identity": _upper_check(
            float(np.linalg.norm(atom_sum @ partition - np.eye(3), ord="fro")),
            float(numerical_gates["atomic_partition_moment_identity_max"]),
        ),
        "maximum_transition_charge": _upper_check(
            float(np.max(np.abs(baseline.transition_charges_e))),
            float(numerical_gates["maximum_transition_charge_max"]),
        ),
        "completed_atom_covariance_relative_error": _upper_check(
            completion.atom_covariance_error
            / max(
                float(np.linalg.norm(completion.target_atom_dipole_covariance_bohr3)),
                1.0e-30,
            ),
            float(numerical_gates["completed_atom_covariance_relative_error_max"]),
        ),
        "completed_molecular_polarizability_relative_error": _upper_check(
            completion.molecular_polarizability_error
            / max(float(np.linalg.norm(molecular_polarizability)), 1.0e-30),
            float(
                numerical_gates[
                    "completed_molecular_polarizability_relative_error_max"
                ]
            ),
        ),
        "completed_moore_penrose_relative_error": _upper_check(
            completion.moore_penrose_error
            / max(float(np.linalg.norm(completion.response_support_projector)), 1.0e-30),
            float(numerical_gates["completed_moore_penrose_relative_error_max"]),
        ),
        "support_constraint_response_relative_error": _upper_check(
            float(np.linalg.norm(support_residual))
            / max(float(np.linalg.norm(coefficients_per_field)), 1.0e-30),
            float(
                numerical_gates["support_constraint_response_relative_error_max"]
            ),
        ),
        "uniform_field_molecular_dipole_relative_error": _upper_check(
            _relative_frobenius(molecular_dipole_per_field, molecular_polarizability),
            float(
                numerical_gates[
                    "uniform_field_molecular_dipole_relative_error_max"
                ]
            ),
        ),
        "qm_mep_step_consistency_relative_frobenius": _upper_check(
            _relative_frobenius(qm_mep[FIELD_STEPS[0]], qm_mep[FIELD_STEPS[1]]),
            float(
                numerical_gates[
                    "qm_mep_step_consistency_relative_frobenius_max"
                ]
            ),
        ),
        "qm_dipole_step_consistency_relative_frobenius": _upper_check(
            _relative_frobenius(
                qm_dipole[FIELD_STEPS[0]], qm_dipole[FIELD_STEPS[1]]
            ),
            float(
                numerical_gates[
                    "qm_dipole_step_consistency_relative_frobenius_max"
                ]
            ),
        ),
    }


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    git_head = _require_clean_tracked_checkout()
    preregistration, input_hashes, preregistration_sha = _validate_preregistration()
    source_module = _load_module(
        ATOMIC_SOURCE_MODULE_RELATIVE_PATH,
        module_name="route2_v0_atomic_independent_particle_response_frozen_source",
    )
    response_kernel_module = _load_module(
        RESPONSE_KERNEL_MODULE_RELATIVE_PATH,
        module_name="route2_v0_response_kernel_frozen_source",
    )
    source_table = _load_atomic_table(source_module)
    symbols, positions, atomic_weights, polarizability = _load_mace_mdp_inputs()
    points = _load_frozen_qm_mep_points()
    qm_responses = _load_qm_responses(points=points, positions=positions)
    atomic_numbers = np.asarray([ATOMIC_NUMBERS[symbol] for symbol in symbols])
    baseline, completion, coefficients, candidate_dipole = _build_completed_response(
        source_table=source_table,
        response_kernel_module=response_kernel_module,
        atomic_numbers=atomic_numbers,
        atomic_dipole_weights=atomic_weights,
        molecular_polarizability=polarizability,
    )
    source_potentials = _source_potential_mode_matrix(
        source_table=source_table,
        atomic_numbers=atomic_numbers,
        positions_angstrom=positions,
        points_bohr=points,
        coefficient_slices=baseline.coefficient_slices,
    )
    candidate_mep = -(source_potentials @ coefficients)
    qm_mep = {
        step: _response_tensor(qm_responses, step=step, observable="potential")
        for step in FIELD_STEPS
    }
    qm_dipole = {
        step: _response_tensor(qm_responses, step=step, observable="dipole")
        for step in FIELD_STEPS
    }
    numerical_gates = preregistration["numerical_gates"]
    scientific_gates = preregistration["scientific_falsification_gates"]
    numerical_checks = _numerical_checks(
        baseline=baseline,
        completion=completion,
        atomic_dipole_weights=atomic_weights,
        molecular_polarizability=polarizability,
        coefficients_per_field=coefficients,
        molecular_dipole_per_field=candidate_dipole,
        qm_mep=qm_mep,
        qm_dipole=qm_dipole,
        numerical_gates=numerical_gates,
    )
    comparisons_by_step: dict[str, dict[str, object]] = {}
    for step in FIELD_STEPS:
        comparisons_by_step[f"{step:.1e}"] = {
            direction: {
                "candidate_induced_dipole_bohr3": candidate_dipole[
                    :, direction_index
                ].tolist(),
                "candidate_electronic_potential_response_sha256": _sha256_array(
                    candidate_mep[:, direction_index]
                ),
                "qm_induced_dipole_bohr3": qm_dipole[step][
                    :, direction_index
                ].tolist(),
                "qm_electronic_potential_response_sha256": _sha256_array(
                    qm_mep[step][:, direction_index]
                ),
                "mep_response_relative_error": _relative_frobenius(
                    candidate_mep[:, direction_index], qm_mep[step][:, direction_index]
                ),
                "dipole_response_relative_error": _relative_frobenius(
                    candidate_dipole[:, direction_index],
                    qm_dipole[step][:, direction_index],
                ),
            }
            for direction_index, direction in enumerate(DIRECTIONS)
        }
    selected_comparison = comparisons_by_step[f"{SELECTED_FIELD_STEP:.1e}"]
    selected_direction_errors = [
        float(selected_comparison[direction]["mep_response_relative_error"])
        for direction in DIRECTIONS
    ]
    scientific_checks = {
        "mep_response_relative_frobenius": _upper_check(
            _relative_frobenius(candidate_mep, qm_mep[SELECTED_FIELD_STEP]),
            float(scientific_gates["mep_response_relative_frobenius_max"]),
        ),
        "mep_response_relative_direction_max": _upper_check(
            max(selected_direction_errors),
            float(scientific_gates["mep_response_relative_direction_max"]),
        ),
        "induced_dipole_response_relative_frobenius": _upper_check(
            _relative_frobenius(candidate_dipole, qm_dipole[SELECTED_FIELD_STEP]),
            float(
                scientific_gates["induced_dipole_response_relative_frobenius_max"]
            ),
        ),
    }
    numerical_pass = all(bool(check["passes"]) for check in numerical_checks.values())
    scientific_pass = all(bool(check["passes"]) for check in scientific_checks.values())
    passes_all = numerical_pass and scientific_pass
    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass" if passes_all else "reject",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": git_head,
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": preregistration_sha,
            "protocol_id": preregistration["protocol_id"],
        },
        "claim_boundary": preregistration["claim_boundary"],
        "hard_constraints": preregistration["hard_constraints"],
        "source_files_sha256": {
            relative: _sha256(REPO_ROOT / relative)
            for relative in SOURCE_RELATIVE_PATHS
        },
        "input_files_sha256": input_hashes,
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
            "source_point_count": len(points),
            "source_points_bohr_sha256": _sha256_array(points),
        },
        "source_representation": {
            "construction": "frozen-atomic-independent-particle-direct-sum-v0-rk",
            "atomic_response_asset": (
                "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1"
            ),
            "coefficient_dual_pairing": (
                "f_m=integral tau_m(r)V(r)dr; x=-C f; "
                "delta n=sum_m x_m tau_m"
            ),
            "response_completion": (
                "C=C0-LS0L^T+L(W alpha W^T)L^T; "
                "L=C0 A^T S0^-1; no fitted coefficient"
            ),
            "external_uniform_field_dual": (
                "f_m,k=integral tau_m(r) r_k dr; all response modes are "
                "intrinsically charge neutral"
            ),
            "electronic_potential": (
                "delta V_e(s)=-sum_m x_m integral tau_m(r)/|r-s|dr"
            ),
            "atomic_dipole_partition": (
                "W_a from frozen MACE-MDP atomic map without reweighting"
            ),
            "molecular_polarizability": (
                "frozen canonical MACE-MDP tensor after its prior raw "
                "antisymmetry gate; no rescaling, clipping, or fitted edit"
            ),
            "source_potential_mode_matrix_sha256": _sha256_array(source_potentials),
            "response_coefficient_per_field_sha256": _sha256_array(coefficients),
            "table_sha256": source_table.table_sha256,
            "table_manifest_sha256": source_table.manifest_sha256,
        },
        "response_kernel": {
            "coefficient_count": baseline.coefficient_count,
            "support_constraint_count": int(
                completion.response_support_constraints.shape[0]
            ),
            "baseline_minimum_eigenvalue": completion.baseline_minimum_eigenvalue,
            "completed_minimum_eigenvalue": completion.completed_minimum_eigenvalue,
            "baseline_atom_covariance_minimum_eigenvalue": (
                completion.baseline_atom_covariance_minimum_eigenvalue
            ),
            "baseline_symmetry_error": completion.baseline_symmetry_error,
            "completed_symmetry_error": completion.completed_symmetry_error,
            "completed_charge_response_error": completion.completed_charge_response_error,
            "charge_constraint_vector": None,
        },
        "comparison_by_qm_field_step": comparisons_by_step,
        "numerical_checks": numerical_checks,
        "scientific_falsification": {
            "checks": scientific_checks,
            "passes_all_registered_checks": passes_all,
            "verdict": (
                "admit-frozen-atomic-independent-particle-source-to-same-basis-"
                "common-scalar-kkt-gates-only"
                if passes_all
                else "reject-frozen-atomic-independent-particle-source"
            ),
            "admission_boundary": (
                "A pass admits only this exact frozen gas-phase source to a "
                "separately registered same-basis common-scalar/KKT, continuum-"
                "duality, and force gate. It does not establish a molecular "
                "electronic functional, PCM result, nonpolar term, force/PES "
                "result, or experimental solvation accuracy."
            ),
        },
        "runtime": {
            "python": sys.version,
            "python_resolved_sha256": _sha256(Path(sys.executable).resolve()),
            "boundary": (
                "This is an array-level gas-phase QM-MEP source falsifier using "
                "frozen raw QM data; it is not an end-to-end Route-2 or QM runtime "
                "comparison."
            ),
        },
    }
    _write_exclusive_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
