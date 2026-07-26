#!/usr/bin/env python3
"""One-shot fixed-density analytic frame-VJP canary for JGP94 ddPCM.

This benchmark-local runner differentiates the exact molecule-frame scalar
accepted by ``run_route2_jgp94_ddpcm_canary.py``.  It deliberately excludes
MACE calls, the ML--SCF root, CDS, public provider wiring, and total forces.

``preflight`` performs only pure NumPy finite differences plus pyddx cavity
construction; it does not solve a continuum equation.  ``freeze`` records a
clean committed protocol.  ``run`` consumes that lock once and performs
exactly two base derivative solves plus eight displaced scalar solves.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXPCMReactionFieldLinearMap,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader
from route2_jgp94_frame import (
    JGP94Frame,
    body_dipoles,
    build_jgp94_frame,
    jgp94_frame_vjp,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "route2-ddpcm-ri-jgp94-acetone-fixed-density-frame-vjp-v1"
DEFAULT_WORK_DIR = (
    REPO_ROOT
    / ".omx"
    / "benchmarks"
    / "route2-ddpcm-ri-jgp94-frame-vjp-acetone-20260726"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_jgp94_frame_vjp_canary.py"
)
FRAME_MODULE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2_jgp94_frame.py"
)
DOCUMENT_RELATIVE_PATH = (
    "docs/implicit-solvation/ROUTE2_FRAME_VJP_CANARY.md"
)
MOL2_RELATIVE_PATH = (
    ".omx/benchmarks/route2-macepolar-smd-smoke/"
    "dataset/mol2files_gaff/mobley_3867265.mol2"
)
STATE_RELATIVE_PATH = (
    ".omx/benchmarks/route2-pyddx-mace-coupled-methanol-20260725/"
    "public_profile_kspace40_acetone_force.out.implicit/"
    "route2-ddpcm-state.npz"
)
COMPONENT_SOURCE_RELATIVE_PATH = (
    ".omx/benchmarks/route2-pyddx-mace-coupled-methanol-20260725/"
    "full_mlscf_kspace40_force_fd_acetone_arrays_l15_n1202.npz"
)
SCALAR_RESULT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-ddpcm-ri-jgp94-acetone-v1.json"
)

EXPECTED_MOL2_SHA256 = (
    "713d87b5cb8d1a2a14cb217a061102d8535643f86fe97068b270ea398c5970c1"
)
EXPECTED_STATE_SHA256 = (
    "640d30827f75dbc0246febf8bea75de868e114c441f1ec4300a01432caffb69a"
)
EXPECTED_COMPONENT_SOURCE_SHA256 = (
    "997fcc45ee7cb78ccf2b5fb2465461ea77388570b7ca0cf91a4fb7e161be8497"
)
EXPECTED_SCALAR_RESULT_SHA256 = (
    "f1b7b01eb80b42489c1306b2006f919e5cc79bc295f73c84824626abccbeeeb7"
)
EXPECTED_PYDDX_EXTENSION_SHA256 = (
    "697bafe818a749bb70963ef13a36496bcba52fc427046e8eefe6e769a0d68845"
)
EXPECTED_BASE_BODY_ENERGY_EV = -0.39974434990574453
EXPECTED_BASE_ACTIVE_MAPPING_SHA256 = (
    "f4de8cfcb1e3cc99131227ad92851bd9d43006a4cd5a88e045342bf0e9d49300"
)
EXPECTED_BASE_ACTIVE_PAIR_COUNT = 4993

PROVIDER = {
    "backend": "pyddx",
    "pyddx_version": "0.8.0",
    "model": "pcm",
    "dielectric": 78.39,
    "eta": 0.1,
    "shift": 0.0,
    "lmax": 15,
    "n_lebedev": 1202,
    "n_proc": 1,
    "solver_tolerance": 1.0e-12,
    "enable_fmm": False,
    "enable_force": True,
}

SELECTED_COMPONENTS = (
    {
        "label": "carbonyl-carbon-y",
        "atom_index_zero_based": 1,
        "axis_index": 1,
        "axis": "y",
        "selection_source_value_ev_per_angstrom": -0.806069893702285,
    },
    {
        "label": "carbonyl-oxygen-y",
        "atom_index_zero_based": 2,
        "axis_index": 1,
        "axis": "y",
        "selection_source_value_ev_per_angstrom": 0.5830010743437304,
    },
)
FINITE_DIFFERENCE_STEPS_ANGSTROM = (1.0e-3, 5.0e-4)
SYNTHETIC_STEPS = (1.0e-4, 5.0e-5, 2.5e-5)
SYNTHETIC_SEED = 20260726
ROTATIONS = (
    {
        "label": "identity",
        "axis": [1.0, 0.0, 0.0],
        "angle_degrees": 0.0,
    },
    {
        "label": "axis-1-2-3-angle-37",
        "axis": [1.0, 2.0, 3.0],
        "angle_degrees": 37.0,
    },
    {
        "label": "axis-minus2-1-0p5-angle-113",
        "axis": [-2.0, 1.0, 0.5],
        "angle_degrees": 113.0,
    },
)

GATES = {
    "minimum_relative_eigenvalue_gap": 0.05,
    "minimum_reference_axis_overlap": 0.999999,
    "maximum_frame_orthogonality_error": 1.0e-12,
    "maximum_frame_determinant_error": 1.0e-12,
    "minimum_synthetic_orientation_correction_norm": 1.0e-3,
    "maximum_synthetic_translation_error": 1.0e-12,
    "maximum_synthetic_position_error_by_step": {
        "0.0001": 2.0e-7,
        "5e-05": 6.0e-8,
        "2.5e-05": 2.0e-8,
    },
    "maximum_synthetic_position_error_ratio": 0.4,
    "maximum_synthetic_dipole_error": 5.0e-10,
    "maximum_base_energy_reproduction_error_ev": 1.0e-10,
    "maximum_base_derivative_solve_energy_difference_ev": 1.0e-10,
    "maximum_half_coupling_identity_error_ev": 1.0e-10,
    "maximum_translation_error_ev_per_angstrom": 1.0e-10,
    "maximum_physical_torque_error_ev": 1.0e-8,
    "maximum_rotation_covariance_error_ev_per_angstrom": 1.0e-10,
    "maximum_rotation_dipole_covariance_error_ev": 1.0e-10,
    "maximum_finite_difference_absolute_error_ev_per_angstrom": 2.0e-5,
    "maximum_finite_difference_relative_error": 1.0e-4,
    "maximum_fine_to_coarse_error_ratio": 0.5,
}

EVALUATION_BUDGET = {
    "base_coordinate_derivative_state_solves": 1,
    "base_density_derivative_state_solves": 1,
    "finite_difference_scalar_state_solves": 8,
    "total_provider_state_solves": 10,
    "selected_components": 2,
    "finite_difference_steps": 2,
    "signs_per_step": 2,
    "mace_calls": 0,
    "ml_scf_roots": 0,
    "cds_calls": 0,
    "retries": 0,
}

CLAIM_BOUNDARY = (
    "A pass establishes only the complete local analytic frame VJP of the "
    "previously accepted fixed-density molecule-frame pyddx ddPCM scalar for "
    "two acetone Cartesian components inside one nondegenerate, fixed-active-"
    "set chart. It does not establish an ML-SCF force, CDS derivative, total "
    "solution-phase PES, chemical accuracy, public provider, or broad domain."
)

STOP_CONDITION = (
    "Failing preflight rejects this derivative experiment before a continuum "
    "solve. Failing any one-shot gate rejects the candidate without tuning or "
    "retry. Passing authorizes only a separately pre-registered local "
    "fixed-density smoothness/rotation derivative stage or integration design."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(value))


def _write_atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(value))
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _require_clean_committed_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The one-shot lock/run requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative in (
        RUNNER_RELATIVE_PATH,
        FRAME_MODULE_RELATIVE_PATH,
        DOCUMENT_RELATIVE_PATH,
        SCALAR_RESULT_RELATIVE_PATH,
    ):
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _axis_angle_rotation(axis: list[float], angle_degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Rotation axis must be finite and nonzero.")
    unit = vector / norm
    angle = math.radians(float(angle_degrees))
    cross = np.asarray(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ],
        dtype=float,
    )
    return (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(unit, unit)
        + math.sin(angle) * cross
    )


def _load_inputs() -> dict[str, Any]:
    paths = {
        "mol2": REPO_ROOT / MOL2_RELATIVE_PATH,
        "state": REPO_ROOT / STATE_RELATIVE_PATH,
        "component_source": REPO_ROOT / COMPONENT_SOURCE_RELATIVE_PATH,
        "scalar_result": REPO_ROOT / SCALAR_RESULT_RELATIVE_PATH,
    }
    expected_hashes = {
        "mol2": EXPECTED_MOL2_SHA256,
        "state": EXPECTED_STATE_SHA256,
        "component_source": EXPECTED_COMPONENT_SOURCE_SHA256,
        "scalar_result": EXPECTED_SCALAR_RESULT_SHA256,
    }
    for label, path in paths.items():
        if _sha256_file(path) != expected_hashes[label]:
            raise RuntimeError(f"The frozen {label} input hash changed.")

    atoms = MOL2Reader(str(paths["mol2"]), charge=0, mult=1)
    with np.load(paths["state"]) as state:
        positions = np.asarray(state["positions_angstrom"], dtype=float)
        radii = np.asarray(state["cavity_radii_angstrom"], dtype=float)
        density_raw = np.asarray(state["density_coefficients"], dtype=float)
    if positions.shape != (len(atoms), 3):
        raise RuntimeError("The frozen acetone positions have the wrong shape.")
    if radii.shape != (len(atoms),) or density_raw.shape != (len(atoms), 4):
        raise RuntimeError("The frozen acetone radii or density have the wrong shape.")
    position_error = float(
        np.max(np.abs(positions - np.asarray(atoms.get_positions(), dtype=float)))
    )
    if position_error > 1.0e-12:
        raise RuntimeError("The frozen state no longer matches the acetone MOL2.")

    with np.load(paths["component_source"]) as source:
        source_positions = np.asarray(source["positions_angstrom"], dtype=float)
        source_gradient = np.asarray(
            source["continuum_gradient_ev_per_angstrom"],
            dtype=float,
        )
    source_translation = np.mean(source_positions - positions, axis=0)
    source_alignment_error = float(
        np.max(
            np.abs(
                source_positions - positions - source_translation[None, :]
            )
        )
    )
    if source_alignment_error > 1.0e-12:
        raise RuntimeError(
            "The component-selection source is not the same translated geometry."
        )
    ranked = sorted(
        (
            (
                abs(float(source_gradient[atom_index, axis_index])),
                atom_index,
                axis_index,
            )
            for atom_index in range(source_gradient.shape[0])
            for axis_index in range(3)
        ),
        reverse=True,
    )
    selected_pairs = [
        (
            int(record["atom_index_zero_based"]),
            int(record["axis_index"]),
        )
        for record in SELECTED_COMPONENTS
    ]
    if selected_pairs != [
        (ranked[0][1], ranked[0][2]),
        (ranked[1][1], ranked[1][2]),
    ]:
        raise RuntimeError("The predeclared components are not the two source maxima.")
    for record in SELECTED_COMPONENTS:
        observed = float(
            source_gradient[
                int(record["atom_index_zero_based"]),
                int(record["axis_index"]),
            ]
        )
        if observed != float(record["selection_source_value_ev_per_angstrom"]):
            raise RuntimeError("A frozen component-selection value changed.")

    scalar_result = json.loads(paths["scalar_result"].read_text())
    if scalar_result.get("status") != "pass":
        raise RuntimeError("The prerequisite scalar canary is not passing.")
    identity_record = scalar_result["records"][0]["candidate"]
    if float(identity_record["energy_ev"]) != EXPECTED_BASE_BODY_ENERGY_EV:
        raise RuntimeError("The prerequisite body-frame energy changed.")
    mapping = identity_record["aligned_active_mapping"]
    if (
        mapping["sha256"] != EXPECTED_BASE_ACTIVE_MAPPING_SHA256
        or int(mapping["pair_count"]) != EXPECTED_BASE_ACTIVE_PAIR_COUNT
        or int(mapping["unique_pair_count"]) != EXPECTED_BASE_ACTIVE_PAIR_COUNT
    ):
        raise RuntimeError("The prerequisite body-frame active mapping changed.")

    density_cartesian = density_to_external_field_order(density_raw)
    return {
        "positions_angstrom": positions,
        "radii_angstrom": radii,
        "density_raw": density_raw,
        "charges": density_cartesian[:, 0].copy(),
        "dipoles_angstrom": density_cartesian[:, 1:].copy(),
        "nuclear_charges": np.asarray(atoms.get_atomic_numbers(), dtype=float),
        "symbols": list(atoms.get_chemical_symbols()),
        "mol2_position_error_angstrom": position_error,
        "component_source_translation_angstrom": source_translation,
        "component_source_alignment_error_angstrom": source_alignment_error,
    }


def _body_density_raw(
    charges: np.ndarray,
    dipoles_angstrom: np.ndarray,
    frame: JGP94Frame,
) -> np.ndarray:
    body_cartesian = np.column_stack(
        (charges, body_dipoles(dipoles_angstrom, frame))
    )
    return external_field_to_density_order(body_cartesian)


def _synthetic_loss(
    frame: JGP94Frame,
    dipoles_angstrom: np.ndarray,
    body_position_cotangent: np.ndarray,
    body_dipole_cotangent: np.ndarray,
) -> float:
    return float(
        np.vdot(body_position_cotangent, frame.body_positions)
        + np.vdot(
            body_dipole_cotangent,
            body_dipoles(dipoles_angstrom, frame),
        )
    )


def _synthetic_frame_preflight(inputs: dict[str, Any]) -> dict[str, Any]:
    positions = inputs["positions_angstrom"]
    dipoles = inputs["dipoles_angstrom"]
    nuclear_charges = inputs["nuclear_charges"]
    base_frame = build_jgp94_frame(
        positions,
        nuclear_charges,
        minimum_relative_eigengap=GATES["minimum_relative_eigenvalue_gap"],
    )
    random = np.random.default_rng(SYNTHETIC_SEED)
    body_position_cotangent = random.normal(size=positions.shape)
    body_dipole_cotangent = random.normal(size=dipoles.shape)
    analytic = jgp94_frame_vjp(
        base_frame,
        nuclear_charges,
        dipoles,
        body_position_cotangent,
        body_dipole_cotangent,
    )

    records: list[dict[str, Any]] = []
    for step in SYNTHETIC_STEPS:
        position_fd = np.empty_like(positions)
        dipole_fd = np.empty_like(dipoles)
        minimum_axis_overlap = 1.0
        for atom_index in range(positions.shape[0]):
            for axis_index in range(3):
                position_values = []
                for sign in (-1.0, 1.0):
                    displaced_positions = positions.copy()
                    displaced_positions[atom_index, axis_index] += sign * step
                    displaced_frame = build_jgp94_frame(
                        displaced_positions,
                        nuclear_charges,
                        minimum_relative_eigengap=GATES[
                            "minimum_relative_eigenvalue_gap"
                        ],
                        reference_orientation=base_frame.orientation,
                        minimum_reference_axis_overlap=GATES[
                            "minimum_reference_axis_overlap"
                        ],
                    )
                    if displaced_frame.reference_axis_overlaps is None:
                        raise RuntimeError("The local frame overlap was not retained.")
                    minimum_axis_overlap = min(
                        minimum_axis_overlap,
                        float(np.min(displaced_frame.reference_axis_overlaps)),
                    )
                    position_values.append(
                        _synthetic_loss(
                            displaced_frame,
                            dipoles,
                            body_position_cotangent,
                            body_dipole_cotangent,
                        )
                    )
                position_fd[atom_index, axis_index] = (
                    position_values[1] - position_values[0]
                ) / (2.0 * step)

                dipole_values = []
                for sign in (-1.0, 1.0):
                    displaced_dipoles = dipoles.copy()
                    displaced_dipoles[atom_index, axis_index] += sign * step
                    dipole_values.append(
                        _synthetic_loss(
                            base_frame,
                            displaced_dipoles,
                            body_position_cotangent,
                            body_dipole_cotangent,
                        )
                    )
                dipole_fd[atom_index, axis_index] = (
                    dipole_values[1] - dipole_values[0]
                ) / (2.0 * step)
        records.append(
            {
                "step": step,
                "maximum_position_absolute_error": float(
                    np.max(np.abs(position_fd - analytic.position_cotangent))
                ),
                "maximum_dipole_absolute_error": float(
                    np.max(np.abs(dipole_fd - analytic.dipole_cotangent))
                ),
                "minimum_reference_axis_overlap": minimum_axis_overlap,
            }
        )

    position_errors = [
        float(record["maximum_position_absolute_error"])
        for record in records
    ]
    position_error_ratios = [
        position_errors[index + 1] / position_errors[index]
        for index in range(len(position_errors) - 1)
    ]
    translation_error = float(
        np.max(np.abs(np.sum(analytic.position_cotangent, axis=0)))
    )
    orientation_correction_norm = float(
        np.linalg.norm(analytic.orientation_position_cotangent)
    )
    return {
        "seed": SYNTHETIC_SEED,
        "all_position_component_count": int(positions.size),
        "all_dipole_component_count": int(dipoles.size),
        "base_relative_minimum_eigengap": (
            base_frame.relative_minimum_eigengap
        ),
        "base_orthogonality_error": base_frame.orthogonality_error,
        "base_determinant_error": base_frame.determinant_error,
        "orientation_correction_norm": orientation_correction_norm,
        "translation_error": translation_error,
        "position_error_ratios": position_error_ratios,
        "records": records,
    }


def _runtime_provenance() -> dict[str, Any]:
    module = importlib.import_module("pyddx")
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("The pyddx extension has no filesystem path.")
    module_path = Path(module_file).resolve()
    module_hash = _sha256_file(module_path)
    if str(getattr(module, "__version__", "unknown")) != PROVIDER["pyddx_version"]:
        raise RuntimeError("The pyddx version differs from the frozen protocol.")
    if module_hash != EXPECTED_PYDDX_EXTENSION_SHA256:
        raise RuntimeError("The pyddx extension hash differs from the protocol.")
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": np.__version__,
        "pyddx_version": str(module.__version__),
        "pyddx_extension_path": str(module_path),
        "pyddx_extension_sha256": module_hash,
    }


def _new_response(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
) -> PyDDXPCMReactionFieldLinearMap:
    return PyDDXPCMReactionFieldLinearMap(
        positions_angstrom,
        radii_angstrom,
        dielectric=PROVIDER["dielectric"],
        lmax=PROVIDER["lmax"],
        n_lebedev=PROVIDER["n_lebedev"],
        n_proc=PROVIDER["n_proc"],
        solver_tolerance=PROVIDER["solver_tolerance"],
        eta=PROVIDER["eta"],
    )


def _lebedev_directions(runtime_module: Any) -> np.ndarray:
    model = runtime_module.Model(
        "pcm",
        np.zeros((3, 1), dtype=float),
        np.ones(1, dtype=float),
        PROVIDER["dielectric"],
        eta=PROVIDER["eta"],
        shift=PROVIDER["shift"],
        lmax=PROVIDER["lmax"],
        n_lebedev=PROVIDER["n_lebedev"],
        enable_fmm=PROVIDER["enable_fmm"],
        n_proc=PROVIDER["n_proc"],
        enable_force=PROVIDER["enable_force"],
    )
    directions = np.asarray(model.cavity, dtype=float).T
    if directions.shape != (PROVIDER["n_lebedev"], 3):
        raise RuntimeError("The auxiliary Lebedev grid has the wrong shape.")
    return directions


def _active_mapping_signature(
    response: PyDDXPCMReactionFieldLinearMap,
    lebedev_directions: np.ndarray,
) -> dict[str, Any]:
    model = response._model
    cavity = np.asarray(model.cavity, dtype=float).T
    centres = np.asarray(model.sphere_centres, dtype=float).T
    radii = np.asarray(model.sphere_radii, dtype=float)
    if cavity.ndim != 2 or cavity.shape[0] == 0 or cavity.shape[1] != 3:
        raise RuntimeError("The active cavity has the wrong shape.")
    residuals = np.abs(
        np.linalg.norm(cavity[:, None, :] - centres[None, :, :], axis=2)
        - radii[None, :]
    )
    owners = np.argmin(residuals, axis=1)
    directions = (cavity - centres[owners]) / radii[owners, None]
    lebedev_indices = np.empty(cavity.shape[0], dtype=np.int64)
    maximum_direction_error = 0.0
    for start in range(0, cavity.shape[0], 512):
        stop = min(start + 512, cavity.shape[0])
        dots = directions[start:stop] @ lebedev_directions.T
        indices = np.argmax(dots, axis=1)
        lebedev_indices[start:stop] = indices
        maximum_direction_error = max(
            maximum_direction_error,
            float(
                np.max(
                    np.linalg.norm(
                        directions[start:stop] - lebedev_directions[indices],
                        axis=1,
                    )
                )
            ),
        )
    pairs = np.column_stack((owners, lebedev_indices)).astype(np.int64)
    sorted_pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return {
        "sha256": hashlib.sha256(sorted_pairs.tobytes()).hexdigest(),
        "pair_count": int(sorted_pairs.shape[0]),
        "unique_pair_count": int(np.unique(sorted_pairs, axis=0).shape[0]),
        "maximum_owner_residual_bohr": float(
            np.max(residuals[np.arange(cavity.shape[0]), owners])
        ),
        "maximum_lebedev_direction_error": maximum_direction_error,
    }


def _cavity_preflight(inputs: dict[str, Any]) -> dict[str, Any]:
    positions = inputs["positions_angstrom"]
    nuclear_charges = inputs["nuclear_charges"]
    radii = inputs["radii_angstrom"]
    base_frame = build_jgp94_frame(
        positions,
        nuclear_charges,
        minimum_relative_eigengap=GATES["minimum_relative_eigenvalue_gap"],
    )
    pyddx = importlib.import_module("pyddx")
    lebedev_directions = _lebedev_directions(pyddx)
    records = []
    for component in SELECTED_COMPONENTS:
        for step in FINITE_DIFFERENCE_STEPS_ANGSTROM:
            for sign in (-1.0, 1.0):
                displaced_positions = positions.copy()
                displaced_positions[
                    int(component["atom_index_zero_based"]),
                    int(component["axis_index"]),
                ] += sign * step
                frame = build_jgp94_frame(
                    displaced_positions,
                    nuclear_charges,
                    minimum_relative_eigengap=GATES[
                        "minimum_relative_eigenvalue_gap"
                    ],
                    reference_orientation=base_frame.orientation,
                    minimum_reference_axis_overlap=GATES[
                        "minimum_reference_axis_overlap"
                    ],
                )
                reference_axis_overlaps = frame.reference_axis_overlaps
                if reference_axis_overlaps is None:
                    raise RuntimeError(
                        "Displaced cavity frame did not record reference-axis "
                        "overlaps."
                    )
                response = _new_response(frame.body_positions, radii)
                mapping = _active_mapping_signature(
                    response,
                    lebedev_directions,
                )
                records.append(
                    {
                        "component": component["label"],
                        "step_angstrom": step,
                        "sign": sign,
                        "minimum_reference_axis_overlap": float(
                            np.min(reference_axis_overlaps)
                        ),
                        "active_mapping": mapping,
                    }
                )
    return {
        "continuum_state_solve_count": 0,
        "cavity_model_count": len(records),
        "records": records,
    }


def _preflight_payload() -> dict[str, Any]:
    inputs = _load_inputs()
    synthetic = _synthetic_frame_preflight(inputs)
    cavity = _cavity_preflight(inputs)
    synthetic_position_limits = GATES[
        "maximum_synthetic_position_error_by_step"
    ]
    synthetic_records = synthetic["records"]
    cavity_records = cavity["records"]
    gates = {
        "frame_eigengap": (
            synthetic["base_relative_minimum_eigengap"]
            >= GATES["minimum_relative_eigenvalue_gap"]
        ),
        "frame_orthogonality": (
            synthetic["base_orthogonality_error"]
            <= GATES["maximum_frame_orthogonality_error"]
        ),
        "frame_determinant": (
            synthetic["base_determinant_error"]
            <= GATES["maximum_frame_determinant_error"]
        ),
        "synthetic_nontrivial_orientation": (
            synthetic["orientation_correction_norm"]
            >= GATES["minimum_synthetic_orientation_correction_norm"]
        ),
        "synthetic_translation": (
            synthetic["translation_error"]
            <= GATES["maximum_synthetic_translation_error"]
        ),
        "synthetic_position_absolute": all(
            float(record["maximum_position_absolute_error"])
            <= float(synthetic_position_limits[str(record["step"])])
            for record in synthetic_records
        ),
        "synthetic_position_second_order": all(
            float(ratio)
            <= GATES["maximum_synthetic_position_error_ratio"]
            for ratio in synthetic["position_error_ratios"]
        ),
        "synthetic_dipole_absolute": all(
            float(record["maximum_dipole_absolute_error"])
            <= GATES["maximum_synthetic_dipole_error"]
            for record in synthetic_records
        ),
        "synthetic_local_axis_gauge": all(
            float(record["minimum_reference_axis_overlap"])
            >= GATES["minimum_reference_axis_overlap"]
            for record in synthetic_records
        ),
        "cavity_local_axis_gauge": all(
            float(record["minimum_reference_axis_overlap"])
            >= GATES["minimum_reference_axis_overlap"]
            for record in cavity_records
        ),
        "cavity_active_mapping": all(
            record["active_mapping"]["sha256"]
            == EXPECTED_BASE_ACTIVE_MAPPING_SHA256
            for record in cavity_records
        ),
        "cavity_pair_count": all(
            int(record["active_mapping"]["pair_count"])
            == EXPECTED_BASE_ACTIVE_PAIR_COUNT
            for record in cavity_records
        ),
        "cavity_pairs_unique": all(
            int(record["active_mapping"]["unique_pair_count"])
            == int(record["active_mapping"]["pair_count"])
            for record in cavity_records
        ),
        "cavity_mapping_reconstruction": all(
            float(record["active_mapping"]["maximum_owner_residual_bohr"])
            <= 1.0e-10
            and float(
                record["active_mapping"]["maximum_lebedev_direction_error"]
            )
            <= 1.0e-10
            for record in cavity_records
        ),
        "zero_continuum_state_solves": (
            int(cavity["continuum_state_solve_count"]) == 0
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "phase": "preflight",
        "provider_state_solve_count": 0,
        "mace_loaded": False,
        "ml_scf_root_evaluated": False,
        "cds_evaluated": False,
        "synthetic": synthetic,
        "cavity": cavity,
        "gates": gates,
        "failed_gates": sorted(
            key for key, passed in gates.items() if not passed
        ),
        "status": "pass" if all(gates.values()) else "fail",
    }


def _frozen_lock() -> dict[str, Any]:
    head = _require_clean_committed_checkout()
    preflight = _preflight_payload()
    if preflight["status"] != "pass":
        raise RuntimeError(
            "Preflight gates failed; refusing to freeze the one-shot canary."
        )
    tracked_files = {
        "runner": RUNNER_RELATIVE_PATH,
        "frame_module": FRAME_MODULE_RELATIVE_PATH,
        "document": DOCUMENT_RELATIVE_PATH,
        "scalar_result": SCALAR_RESULT_RELATIVE_PATH,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": _utc_now(),
        "one_shot": True,
        "no_retry_after_attempt_marker": True,
        "required_git_head": head,
        "required_clean_tracked_checkout": True,
        "tracked_files": {
            key: {
                "path": relative,
                "sha256": _sha256_file(REPO_ROOT / relative),
            }
            for key, relative in tracked_files.items()
        },
        "untracked_inputs": {
            "mol2": {
                "path": MOL2_RELATIVE_PATH,
                "sha256": EXPECTED_MOL2_SHA256,
            },
            "state": {
                "path": STATE_RELATIVE_PATH,
                "sha256": EXPECTED_STATE_SHA256,
            },
            "component_source": {
                "path": COMPONENT_SOURCE_RELATIVE_PATH,
                "sha256": EXPECTED_COMPONENT_SOURCE_SHA256,
            },
        },
        "runtime": _runtime_provenance(),
        "provider": PROVIDER,
        "selected_components": list(SELECTED_COMPONENTS),
        "finite_difference_steps_angstrom": list(
            FINITE_DIFFERENCE_STEPS_ANGSTROM
        ),
        "rotations": list(ROTATIONS),
        "gates": GATES,
        "evaluation_budget": EVALUATION_BUDGET,
        "preflight": preflight,
        "claim_boundary": CLAIM_BOUNDARY,
        "stop_condition": STOP_CONDITION,
    }


def _verify_lock(lock: dict[str, Any], lock_path: Path) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "frozen_at_utc",
        "one_shot",
        "no_retry_after_attempt_marker",
        "required_git_head",
        "required_clean_tracked_checkout",
        "tracked_files",
        "untracked_inputs",
        "runtime",
        "provider",
        "selected_components",
        "finite_difference_steps_angstrom",
        "rotations",
        "gates",
        "evaluation_budget",
        "preflight",
        "claim_boundary",
        "stop_condition",
    }
    if set(lock) != expected_keys:
        raise RuntimeError("The lock fields differ from the frozen schema.")
    constants = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "one_shot": True,
        "no_retry_after_attempt_marker": True,
        "required_clean_tracked_checkout": True,
        "provider": PROVIDER,
        "selected_components": list(SELECTED_COMPONENTS),
        "finite_difference_steps_angstrom": list(
            FINITE_DIFFERENCE_STEPS_ANGSTROM
        ),
        "rotations": list(ROTATIONS),
        "gates": GATES,
        "evaluation_budget": EVALUATION_BUDGET,
        "claim_boundary": CLAIM_BOUNDARY,
        "stop_condition": STOP_CONDITION,
    }
    for key, expected in constants.items():
        if lock.get(key) != expected:
            raise RuntimeError(f"The frozen {key} differs from the runner.")
    if _preflight_payload() != lock.get("preflight"):
        raise RuntimeError("The recomputed preflight differs from the lock.")
    head = _require_clean_committed_checkout()
    if head != lock.get("required_git_head"):
        raise RuntimeError("Git head differs from the frozen one-shot lock.")
    expected_tracked = {
        "runner": RUNNER_RELATIVE_PATH,
        "frame_module": FRAME_MODULE_RELATIVE_PATH,
        "document": DOCUMENT_RELATIVE_PATH,
        "scalar_result": SCALAR_RESULT_RELATIVE_PATH,
    }
    for key, relative in expected_tracked.items():
        record = lock["tracked_files"].get(key)
        if record is None or record.get("path") != relative:
            raise RuntimeError(f"The frozen {key} path changed.")
        if _sha256_file(REPO_ROOT / relative) != record.get("sha256"):
            raise RuntimeError(f"The frozen {key} hash changed.")
    expected_untracked = {
        "mol2": (MOL2_RELATIVE_PATH, EXPECTED_MOL2_SHA256),
        "state": (STATE_RELATIVE_PATH, EXPECTED_STATE_SHA256),
        "component_source": (
            COMPONENT_SOURCE_RELATIVE_PATH,
            EXPECTED_COMPONENT_SOURCE_SHA256,
        ),
    }
    for key, (relative, digest) in expected_untracked.items():
        record = lock["untracked_inputs"].get(key)
        if record != {"path": relative, "sha256": digest}:
            raise RuntimeError(f"The frozen {key} declaration changed.")
        if _sha256_file(REPO_ROOT / relative) != digest:
            raise RuntimeError(f"The frozen {key} hash changed.")
    if _runtime_provenance() != lock.get("runtime"):
        raise RuntimeError("Runtime provenance differs from the lock.")
    if not lock_path.is_file():
        raise RuntimeError("The lock disappeared during verification.")


def _state_energy_ev(response: PyDDXPCMReactionFieldLinearMap) -> float:
    state = response._scf_state
    if state is None:
        raise RuntimeError("The pyddx diagnostic state was not retained.")
    energy_ev = float(state.energy()) * Hartree
    if not math.isfinite(energy_ev):
        raise RuntimeError("The retained pyddx energy is non-finite.")
    return energy_ev


def _coordinate_gradient_with_retained_state(
    response: PyDDXPCMReactionFieldLinearMap,
    density_coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate the tested private pyddx derivative while retaining its state."""

    density = response._validated_density(
        density_coefficients,
        name="density_coefficients",
    )
    multipoles, electrostatics, state = response._solve_state(
        density,
        coordinate_derivative=True,
        solve_adjoint=True,
        warm_start=True,
    )
    coordinate_gradient = np.asarray(
        state.solvation_force_terms(electrostatics),
        dtype=float,
    )
    coordinate_gradient += np.asarray(
        state.multipole_force_terms(multipoles),
        dtype=float,
    )
    expected_shape = (3, response.atom_count)
    if coordinate_gradient.shape != expected_shape or not np.all(
        np.isfinite(coordinate_gradient)
    ):
        raise RuntimeError(
            "pyddx ddPCM coordinate derivative must be finite with shape "
            f"{expected_shape}; received {coordinate_gradient.shape}."
        )
    return coordinate_gradient.T


def _physical_torque(
    centered_positions: np.ndarray,
    position_cotangent: np.ndarray,
    dipoles: np.ndarray,
    dipole_cotangent: np.ndarray,
) -> np.ndarray:
    return np.sum(
        np.cross(centered_positions, position_cotangent)
        + np.cross(dipoles, dipole_cotangent),
        axis=0,
    )


def _execute(lock: dict[str, Any], lock_path: Path) -> dict[str, Any]:
    inputs = _load_inputs()
    positions = inputs["positions_angstrom"]
    radii = inputs["radii_angstrom"]
    charges = inputs["charges"]
    dipoles = inputs["dipoles_angstrom"]
    nuclear_charges = inputs["nuclear_charges"]
    base_frame = build_jgp94_frame(
        positions,
        nuclear_charges,
        minimum_relative_eigengap=GATES["minimum_relative_eigenvalue_gap"],
    )
    body_density_raw = _body_density_raw(charges, dipoles, base_frame)
    response = _new_response(base_frame.body_positions, radii)
    provider_state_solves = 0

    started = time.perf_counter()
    body_position_cotangent = (
        _coordinate_gradient_with_retained_state(
            response,
            body_density_raw,
        )
        * Hartree
        / Bohr
    )
    provider_state_solves += 1
    coordinate_solve_energy_ev = _state_energy_ev(response)
    raw_density_cotangent_hartree = response._raw_reaction_gradient_hartree(
        body_density_raw,
        warm_start=True,
    )
    provider_state_solves += 1
    density_solve_energy_ev = _state_energy_ev(response)
    body_density_cotangent_ev = (
        density_to_external_field_order(raw_density_cotangent_hartree)
        * Hartree
    )
    frame_vjp = jgp94_frame_vjp(
        base_frame,
        nuclear_charges,
        dipoles,
        body_position_cotangent,
        body_density_cotangent_ev[:, 1:],
    )
    analytic_seconds = time.perf_counter() - started

    paired_energy_ev = 0.5 * float(
        np.vdot(body_density_raw, raw_density_cotangent_hartree)
    ) * Hartree
    half_coupling_error_ev = abs(
        paired_energy_ev - density_solve_energy_ev
    )
    base_energy_reproduction_error_ev = abs(
        density_solve_energy_ev - EXPECTED_BASE_BODY_ENERGY_EV
    )
    derivative_solve_energy_difference_ev = abs(
        coordinate_solve_energy_ev - density_solve_energy_ev
    )
    translation_error = float(
        np.max(np.abs(np.sum(frame_vjp.position_cotangent, axis=0)))
    )
    torque = _physical_torque(
        base_frame.centered_positions,
        frame_vjp.position_cotangent,
        dipoles,
        frame_vjp.dipole_cotangent,
    )
    torque_norm = float(np.linalg.norm(torque))

    rotation_records = []
    maximum_rotation_covariance_error = 0.0
    maximum_rotation_dipole_covariance_error = 0.0
    for specification in ROTATIONS:
        rotation = _axis_angle_rotation(
            specification["axis"],
            specification["angle_degrees"],
        )
        rotated_positions = (
            base_frame.center
            + base_frame.centered_positions @ rotation.T
        )
        rotated_dipoles = dipoles @ rotation.T
        rotated_frame = build_jgp94_frame(
            rotated_positions,
            nuclear_charges,
            minimum_relative_eigengap=GATES[
                "minimum_relative_eigenvalue_gap"
            ],
            reference_orientation=rotation @ base_frame.orientation,
            minimum_reference_axis_overlap=GATES[
                "minimum_reference_axis_overlap"
            ],
        )
        rotated_vjp = jgp94_frame_vjp(
            rotated_frame,
            nuclear_charges,
            rotated_dipoles,
            body_position_cotangent,
            body_density_cotangent_ev[:, 1:],
        )
        position_covariance_error = float(
            np.max(
                np.abs(
                    rotated_vjp.position_cotangent
                    - frame_vjp.position_cotangent @ rotation.T
                )
            )
        )
        dipole_covariance_error = float(
            np.max(
                np.abs(
                    rotated_vjp.dipole_cotangent
                    - frame_vjp.dipole_cotangent @ rotation.T
                )
            )
        )
        maximum_rotation_covariance_error = max(
            maximum_rotation_covariance_error,
            position_covariance_error,
        )
        maximum_rotation_dipole_covariance_error = max(
            maximum_rotation_dipole_covariance_error,
            dipole_covariance_error,
        )
        reference_axis_overlaps = rotated_frame.reference_axis_overlaps
        if reference_axis_overlaps is None:
            raise RuntimeError(
                "Rotated frame did not record reference-axis overlaps."
            )
        rotation_records.append(
            {
                **specification,
                "minimum_reference_axis_overlap": float(
                    np.min(reference_axis_overlaps)
                ),
                "body_position_error_angstrom": float(
                    np.max(
                        np.abs(
                            rotated_frame.body_positions
                            - base_frame.body_positions
                        )
                    )
                ),
                "body_dipole_error": float(
                    np.max(
                        np.abs(
                            body_dipoles(rotated_dipoles, rotated_frame)
                            - body_dipoles(dipoles, base_frame)
                        )
                    )
                ),
                "position_covariance_error_ev_per_angstrom": (
                    position_covariance_error
                ),
                "dipole_covariance_error_ev": dipole_covariance_error,
            }
        )

    pyddx = importlib.import_module("pyddx")
    lebedev_directions = _lebedev_directions(pyddx)
    base_mapping = _active_mapping_signature(response, lebedev_directions)
    finite_difference_records = []
    finite_difference_started = time.perf_counter()
    for component in SELECTED_COMPONENTS:
        analytic_component = float(
            frame_vjp.position_cotangent[
                int(component["atom_index_zero_based"]),
                int(component["axis_index"]),
            ]
        )
        for step in FINITE_DIFFERENCE_STEPS_ANGSTROM:
            displaced_records = []
            energies = []
            for sign in (-1.0, 1.0):
                displaced_positions = positions.copy()
                displaced_positions[
                    int(component["atom_index_zero_based"]),
                    int(component["axis_index"]),
                ] += sign * step
                displaced_frame = build_jgp94_frame(
                    displaced_positions,
                    nuclear_charges,
                    minimum_relative_eigengap=GATES[
                        "minimum_relative_eigenvalue_gap"
                    ],
                    reference_orientation=base_frame.orientation,
                    minimum_reference_axis_overlap=GATES[
                        "minimum_reference_axis_overlap"
                    ],
                )
                displaced_density_raw = _body_density_raw(
                    charges,
                    dipoles,
                    displaced_frame,
                )
                displaced_response = _new_response(
                    displaced_frame.body_positions,
                    radii,
                )
                case_started = time.perf_counter()
                energy_ev = (
                    displaced_response.polarization_energy_hartree(
                        displaced_density_raw
                    )
                    * Hartree
                )
                provider_state_solves += 1
                mapping = _active_mapping_signature(
                    displaced_response,
                    lebedev_directions,
                )
                reference_axis_overlaps = (
                    displaced_frame.reference_axis_overlaps
                )
                if reference_axis_overlaps is None:
                    raise RuntimeError(
                        "Displaced derivative frame did not record "
                        "reference-axis overlaps."
                    )
                energies.append(float(energy_ev))
                displaced_records.append(
                    {
                        "sign": sign,
                        "energy_ev": float(energy_ev),
                        "elapsed_seconds": time.perf_counter() - case_started,
                        "minimum_reference_axis_overlap": float(
                            np.min(reference_axis_overlaps)
                        ),
                        "active_mapping": mapping,
                    }
                )
            finite_difference = (energies[1] - energies[0]) / (2.0 * step)
            absolute_error = abs(finite_difference - analytic_component)
            relative_error = absolute_error / max(
                abs(finite_difference),
                abs(analytic_component),
                1.0e-12,
            )
            finite_difference_records.append(
                {
                    "component": component,
                    "step_angstrom": step,
                    "analytic_ev_per_angstrom": analytic_component,
                    "finite_difference_ev_per_angstrom": finite_difference,
                    "absolute_error_ev_per_angstrom": absolute_error,
                    "relative_error": relative_error,
                    "displaced": displaced_records,
                }
            )
    finite_difference_seconds = time.perf_counter() - finite_difference_started

    fine_to_coarse_error_ratios = {}
    for component in SELECTED_COMPONENTS:
        component_records = [
            record
            for record in finite_difference_records
            if record["component"]["label"] == component["label"]
        ]
        component_records.sort(
            key=lambda record: float(record["step_angstrom"]),
            reverse=True,
        )
        coarse_error = float(
            component_records[0]["absolute_error_ev_per_angstrom"]
        )
        fine_error = float(
            component_records[1]["absolute_error_ev_per_angstrom"]
        )
        fine_to_coarse_error_ratios[str(component["label"])] = (
            fine_error / coarse_error if coarse_error > 0.0 else math.inf
        )

    displaced_mappings = [
        displaced["active_mapping"]
        for record in finite_difference_records
        for displaced in record["displaced"]
    ]
    gates = {
        "preflight": lock["preflight"]["status"] == "pass",
        "exact_evaluation_budget": (
            provider_state_solves
            == EVALUATION_BUDGET["total_provider_state_solves"]
        ),
        "base_energy_reproduction": (
            base_energy_reproduction_error_ev
            <= GATES["maximum_base_energy_reproduction_error_ev"]
        ),
        "base_derivative_solve_energy": (
            derivative_solve_energy_difference_ev
            <= GATES[
                "maximum_base_derivative_solve_energy_difference_ev"
            ]
        ),
        "half_coupling_identity": (
            half_coupling_error_ev
            <= GATES["maximum_half_coupling_identity_error_ev"]
        ),
        "translation": (
            translation_error
            <= GATES["maximum_translation_error_ev_per_angstrom"]
        ),
        "physical_torque": (
            torque_norm <= GATES["maximum_physical_torque_error_ev"]
        ),
        "rotation_position_covariance": (
            maximum_rotation_covariance_error
            <= GATES[
                "maximum_rotation_covariance_error_ev_per_angstrom"
            ]
        ),
        "rotation_dipole_covariance": (
            maximum_rotation_dipole_covariance_error
            <= GATES["maximum_rotation_dipole_covariance_error_ev"]
        ),
        "finite_difference_absolute": all(
            float(record["absolute_error_ev_per_angstrom"])
            <= GATES[
                "maximum_finite_difference_absolute_error_ev_per_angstrom"
            ]
            for record in finite_difference_records
        ),
        "finite_difference_relative": all(
            float(record["relative_error"])
            <= GATES["maximum_finite_difference_relative_error"]
            for record in finite_difference_records
        ),
        "finite_difference_refinement": all(
            float(ratio) <= GATES["maximum_fine_to_coarse_error_ratio"]
            for ratio in fine_to_coarse_error_ratios.values()
        ),
        "base_active_mapping": (
            base_mapping["sha256"] == EXPECTED_BASE_ACTIVE_MAPPING_SHA256
            and base_mapping["pair_count"] == EXPECTED_BASE_ACTIVE_PAIR_COUNT
            and base_mapping["unique_pair_count"]
            == EXPECTED_BASE_ACTIVE_PAIR_COUNT
        ),
        "displaced_active_mapping": all(
            mapping["sha256"] == EXPECTED_BASE_ACTIVE_MAPPING_SHA256
            and mapping["pair_count"] == EXPECTED_BASE_ACTIVE_PAIR_COUNT
            and mapping["unique_pair_count"]
            == EXPECTED_BASE_ACTIVE_PAIR_COUNT
            for mapping in displaced_mappings
        ),
    }
    failed_gates = sorted(key for key, passed in gates.items() if not passed)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "completed_at_utc": _utc_now(),
        "git_head": _git("rev-parse", "HEAD"),
        "lock_path": str(lock_path.relative_to(REPO_ROOT)),
        "lock_sha256": _sha256_file(lock_path),
        "provider": PROVIDER,
        "base": {
            "coordinate_solve_energy_ev": coordinate_solve_energy_ev,
            "density_solve_energy_ev": density_solve_energy_ev,
            "paired_energy_ev": paired_energy_ev,
            "base_energy_reproduction_error_ev": (
                base_energy_reproduction_error_ev
            ),
            "derivative_solve_energy_difference_ev": (
                derivative_solve_energy_difference_ev
            ),
            "half_coupling_identity_error_ev": half_coupling_error_ev,
            "relative_minimum_eigengap": (
                base_frame.relative_minimum_eigengap
            ),
            "translation_error_ev_per_angstrom": translation_error,
            "physical_torque_ev": torque.tolist(),
            "physical_torque_norm_ev": torque_norm,
            "active_mapping": base_mapping,
        },
        "rotation_records": rotation_records,
        "maximum_rotation_covariance_error_ev_per_angstrom": (
            maximum_rotation_covariance_error
        ),
        "maximum_rotation_dipole_covariance_error_ev": (
            maximum_rotation_dipole_covariance_error
        ),
        "finite_difference_records": finite_difference_records,
        "fine_to_coarse_error_ratios": fine_to_coarse_error_ratios,
        "evaluation_counts": {
            "provider_state_solves": provider_state_solves,
            "base_coordinate_derivative_state_solves": 1,
            "base_density_derivative_state_solves": 1,
            "finite_difference_scalar_state_solves": (
                provider_state_solves - 2
            ),
            "mace_calls": 0,
            "ml_scf_roots": 0,
            "cds_calls": 0,
        },
        "timing_seconds": {
            "base_analytic_derivative": analytic_seconds,
            "finite_difference_scalar_cases": finite_difference_seconds,
        },
        "gates": gates,
        "failed_gates": failed_gates,
        "status": "pass" if not failed_gates else "fail",
        "claim_boundary": CLAIM_BOUNDARY,
        "stop_condition": STOP_CONDITION,
    }


def _command_preflight() -> int:
    payload = _preflight_payload()
    _write_atomic_json(DEFAULT_WORK_DIR / "preflight.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


def _command_freeze() -> int:
    lock_path = DEFAULT_WORK_DIR / "lock.json"
    if (DEFAULT_WORK_DIR / "attempt.json").exists():
        raise RuntimeError("An attempt marker already exists; retry is forbidden.")
    lock = _frozen_lock()
    _write_exclusive_json(lock_path, lock)
    print(
        json.dumps(
            {
                "status": "frozen",
                "lock_path": str(lock_path.relative_to(REPO_ROOT)),
                "lock_sha256": _sha256_file(lock_path),
                "required_git_head": lock["required_git_head"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_run() -> int:
    lock_path = DEFAULT_WORK_DIR / "lock.json"
    attempt_path = DEFAULT_WORK_DIR / "attempt.json"
    result_path = DEFAULT_WORK_DIR / "results.json"
    failure_path = DEFAULT_WORK_DIR / "failure.json"
    if not lock_path.is_file():
        raise RuntimeError("Freeze the canary before running it.")
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("This one-shot canary was already attempted.")
    lock = json.loads(lock_path.read_text())
    _verify_lock(lock, lock_path)
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempted_at_utc": _utc_now(),
        "required_git_head": lock["required_git_head"],
        "lock_sha256": _sha256_file(lock_path),
        "one_shot": True,
    }
    _write_exclusive_json(attempt_path, attempt)
    try:
        result = _execute(lock, lock_path)
        _write_exclusive_json(result_path, result)
    except Exception as exc:
        failure = {
            **attempt,
            "failed_at_utc": _utc_now(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "status": "execution-failure",
            "claim_boundary": CLAIM_BOUNDARY,
            "stop_condition": STOP_CONDITION,
        }
        _write_exclusive_json(failure_path, failure)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot JGP94 fixed-density frame-VJP canary."
    )
    parser.add_argument(
        "phase",
        choices=("preflight", "freeze", "run"),
    )
    arguments = parser.parse_args()
    if arguments.phase == "preflight":
        return _command_preflight()
    if arguments.phase == "freeze":
        return _command_freeze()
    return _command_run()


if __name__ == "__main__":
    raise SystemExit(main())
