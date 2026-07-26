#!/usr/bin/env python3
"""One-shot fixed-density canary for a molecule-following ddPCM frame.

This runner is deliberately outside MAPLE's production calculator path.  It
tests one question only: does evaluating the existing pyddx 0.8.0 ddPCM scalar
in the Johnson--Gill--Pople nuclear-charge principal-axis frame remove the
finite-grid rigid-rotation dependence for one frozen acetone density?

The runner has three explicit phases:

``preflight``
    Check immutable inputs and frame conditioning without constructing a
    continuum model or solving a continuum equation.
``freeze``
    On a clean committed checkout, write an exclusive one-shot lock containing
    the exact Git head, runner/document/runtime/input hashes, rotations,
    provider settings, and pass/fail gates.
``run``
    Consume that lock once, write an attempt marker before any solve, perform
    exactly three laboratory-frame controls and three molecule-frame
    candidates, and freeze either a result or a failure artifact.

It does not load MACE, iterate the ML--SCF root, evaluate CDS, evaluate forces,
or change a public Route-2 profile.
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
from ase.units import Hartree, kcal, mol

from maple.function.calculator.extra_correction.implicit.gto_density import (
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXPCMReactionFieldLinearMap,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader


SCHEMA_VERSION = 1
PROTOCOL_ID = "route2-ddpcm-ri-jgp94-acetone-fixed-density-v1"
DEFAULT_WORK_DIR = (
    REPO_ROOT
    / ".omx"
    / "benchmarks"
    / "route2-ddpcm-ri-jgp94-acetone-20260726"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_jgp94_ddpcm_canary.py"
)
DOCUMENT_RELATIVE_PATH = "docs/implicit-solvation/ROUTE2_PROVIDER_CANARY.md"
MOL2_RELATIVE_PATH = (
    ".omx/benchmarks/route2-macepolar-smd-smoke/"
    "dataset/mol2files_gaff/mobley_3867265.mol2"
)
STATE_RELATIVE_PATH = (
    ".omx/benchmarks/route2-pyddx-mace-coupled-methanol-20260725/"
    "public_profile_kspace40_acetone_force.out.implicit/"
    "route2-ddpcm-state.npz"
)

EXPECTED_MOL2_SHA256 = (
    "713d87b5cb8d1a2a14cb217a061102d8535643f86fe97068b270ea398c5970c1"
)
EXPECTED_STATE_SHA256 = (
    "640d30827f75dbc0246febf8bea75de868e114c441f1ec4300a01432caffb69a"
)
EXPECTED_PYDDX_EXTENSION_SHA256 = (
    "697bafe818a749bb70963ef13a36496bcba52fc427046e8eefe6e769a0d68845"
)

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
    # Engineering guard against the eigenvector singularities identified in
    # Johnson, Gill, and Pople.  It is fixed before the canary and is not an
    # accuracy-tuned physical parameter.
    "minimum_relative_eigenvalue_gap": 0.05,
    "maximum_frame_orthogonality_error": 1.0e-12,
    "maximum_frame_determinant_error": 1.0e-12,
    "maximum_aligned_geometry_error_angstrom": 2.0e-12,
    "maximum_aligned_density_error": 2.0e-12,
    "maximum_surface_owner_residual_bohr": 1.0e-10,
    "minimum_surface_owner_second_gap_bohr": 1.0e-8,
    "maximum_lebedev_direction_error": 1.0e-10,
    "minimum_lebedev_second_neighbor_distance": 1.0e-3,
    "maximum_half_coupling_identity_error_ev": 1.0e-10,
    "maximum_candidate_rotation_span_ev": 1.0e-10,
    "maximum_identity_profile_shift_kcal_mol": 0.01,
}

EVALUATION_BUDGET = {
    "laboratory_frame_controls": 3,
    "molecule_frame_candidates": 3,
    "total_provider_energy_cases": 6,
    "auxiliary_cavity_only_models": 1,
    "mace_calls": 0,
    "cds_calls": 0,
    "force_calls": 0,
    "retries": 0,
}

CLAIM_BOUNDARY = (
    "A pass establishes only fixed-density scalar rigid-rotation invariance "
    "and negligible identity-orientation profile shift for one acetone "
    "geometry. It does not establish a frame VJP, smooth torsion, ML-SCF "
    "force, CDS derivative, QM fidelity, PES, or production profile."
)

STOP_CONDITION = (
    "Failing any locked gate rejects this candidate without tuning. Passing "
    "all gates authorizes only a separately pre-registered analytic "
    "frame-VJP canary."
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
    for relative in (RUNNER_RELATIVE_PATH, DOCUMENT_RELATIVE_PATH):
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _axis_angle_rotation(axis: list[float], angle_degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Rotation axis must be finite and nonzero.")
    unit = vector / norm
    angle = math.radians(float(angle_degrees))
    cross = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ],
        dtype=float,
    )
    rotation = (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(unit, unit)
        + math.sin(angle) * cross
    )
    return rotation


def _raw_density_to_cartesian(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.all(np.isfinite(raw)):
        raise ValueError("Raw density must be finite with shape (n_atoms, 4).")
    return raw[:, [0, 3, 1, 2]].copy()


def _cartesian_density_to_raw(values: np.ndarray) -> np.ndarray:
    cartesian = np.asarray(values, dtype=float)
    if (
        cartesian.ndim != 2
        or cartesian.shape[1] != 4
        or not np.all(np.isfinite(cartesian))
    ):
        raise ValueError(
            "Cartesian density must be finite with shape (n_atoms, 4)."
        )
    return cartesian[:, [0, 2, 3, 1]].copy()


def _nuclear_charge_frame(
    positions_angstrom: np.ndarray,
    nuclear_charges: np.ndarray,
) -> dict[str, Any]:
    positions = np.asarray(positions_angstrom, dtype=float)
    charges = np.asarray(nuclear_charges, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions must have shape (n_atoms, 3).")
    if charges.shape != (positions.shape[0],) or np.any(charges <= 0.0):
        raise ValueError("Nuclear charges must be positive with shape (n_atoms,).")
    center = np.sum(charges[:, None] * positions, axis=0) / float(
        np.sum(charges)
    )
    centered = positions - center
    moment = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for charge, displacement in zip(charges, centered, strict=True):
        moment += charge * (
            float(np.dot(displacement, displacement)) * identity
            - np.outer(displacement, displacement)
        )
    eigenvalues, orientation = np.linalg.eigh(moment)
    if float(np.linalg.det(orientation)) < 0.0:
        orientation[:, -1] *= -1.0
    gaps = np.diff(eigenvalues)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    relative_gap = float(np.min(gaps) / scale)
    orthogonality_error = float(
        np.max(np.abs(orientation.T @ orientation - identity))
    )
    determinant_error = abs(float(np.linalg.det(orientation)) - 1.0)
    return {
        "center_angstrom": center,
        "centered_positions_angstrom": centered,
        "moment_tensor": moment,
        "eigenvalues": eigenvalues,
        "eigenvalue_gaps": gaps,
        "relative_minimum_eigenvalue_gap": relative_gap,
        "orientation": orientation,
        "orthogonality_error": orthogonality_error,
        "determinant_error": determinant_error,
        "body_positions_angstrom": centered @ orientation,
    }


def _proper_signed_axis_align(
    moving: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, float]:
    candidates: list[tuple[float, np.ndarray]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            sz = sx * sy
            transform = np.diag([sx, sy, sz])
            error = float(np.max(np.abs(moving @ transform - reference)))
            candidates.append((error, transform))
    error, transform = min(candidates, key=lambda item: item[0])
    return transform, error


def _load_inputs() -> dict[str, Any]:
    mol2_path = REPO_ROOT / MOL2_RELATIVE_PATH
    state_path = REPO_ROOT / STATE_RELATIVE_PATH
    if _sha256_file(mol2_path) != EXPECTED_MOL2_SHA256:
        raise RuntimeError("The frozen acetone MOL2 hash changed.")
    if _sha256_file(state_path) != EXPECTED_STATE_SHA256:
        raise RuntimeError("The frozen acetone ddPCM state hash changed.")
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    with np.load(state_path) as state:
        positions = np.asarray(state["positions_angstrom"], dtype=float)
        radii = np.asarray(state["cavity_radii_angstrom"], dtype=float)
        density = np.asarray(state["density_coefficients"], dtype=float)
    mol2_positions = np.asarray(atoms.get_positions(), dtype=float)
    if positions.shape != mol2_positions.shape:
        raise RuntimeError("MOL2 and frozen-state atom counts differ.")
    position_error = float(np.max(np.abs(positions - mol2_positions)))
    if position_error > 1.0e-12:
        raise RuntimeError(
            "Frozen-state positions do not match the immutable MOL2 "
            f"({position_error:.3e} angstrom)."
        )
    if radii.shape != (positions.shape[0],):
        raise RuntimeError("Frozen cavity radii have the wrong shape.")
    if density.shape != (positions.shape[0], 4):
        raise RuntimeError("Frozen density has the wrong shape.")
    return {
        "mol2_path": mol2_path,
        "state_path": state_path,
        "positions_angstrom": positions,
        "cavity_radii_angstrom": radii,
        "density_coefficients": density,
        "nuclear_charges": np.asarray(atoms.get_atomic_numbers(), dtype=float),
        "symbols": list(atoms.get_chemical_symbols()),
        "mol2_position_error_angstrom": position_error,
    }


def _preflight_payload() -> dict[str, Any]:
    inputs = _load_inputs()
    positions = inputs["positions_angstrom"]
    density_cartesian = _raw_density_to_cartesian(
        inputs["density_coefficients"]
    )
    charges = inputs["nuclear_charges"]
    base_frame = _nuclear_charge_frame(positions, charges)
    center = base_frame["center_angstrom"]
    base_body_positions = base_frame["body_positions_angstrom"]
    base_body_density = np.column_stack(
        (
            density_cartesian[:, 0],
            density_cartesian[:, 1:] @ base_frame["orientation"],
        )
    )

    records = []
    for specification in ROTATIONS:
        rotation = _axis_angle_rotation(
            specification["axis"],
            specification["angle_degrees"],
        )
        rotated_positions = center + (positions - center) @ rotation.T
        rotated_density = np.column_stack(
            (
                density_cartesian[:, 0],
                density_cartesian[:, 1:] @ rotation.T,
            )
        )
        frame = _nuclear_charge_frame(rotated_positions, charges)
        body_density = np.column_stack(
            (
                rotated_density[:, 0],
                rotated_density[:, 1:] @ frame["orientation"],
            )
        )
        signed_alignment, geometry_error = _proper_signed_axis_align(
            frame["body_positions_angstrom"],
            base_body_positions,
        )
        aligned_density = np.column_stack(
            (
                body_density[:, 0],
                body_density[:, 1:] @ signed_alignment,
            )
        )
        density_error = float(
            np.max(np.abs(aligned_density - base_body_density))
        )
        records.append(
            {
                **specification,
                "rotation_orthogonality_error": float(
                    np.max(np.abs(rotation.T @ rotation - np.eye(3)))
                ),
                "rotation_determinant_error": abs(
                    float(np.linalg.det(rotation)) - 1.0
                ),
                "frame_eigenvalues": frame["eigenvalues"].tolist(),
                "frame_eigenvalue_gaps": frame["eigenvalue_gaps"].tolist(),
                "frame_relative_minimum_eigenvalue_gap": frame[
                    "relative_minimum_eigenvalue_gap"
                ],
                "frame_orthogonality_error": frame["orthogonality_error"],
                "frame_determinant_error": frame["determinant_error"],
                "proper_signed_axis_alignment": signed_alignment.tolist(),
                "aligned_geometry_error_angstrom": geometry_error,
                "aligned_density_error": density_error,
            }
        )

    metrics = {
        "minimum_relative_eigenvalue_gap": min(
            record["frame_relative_minimum_eigenvalue_gap"]
            for record in records
        ),
        "maximum_frame_orthogonality_error": max(
            record["frame_orthogonality_error"] for record in records
        ),
        "maximum_frame_determinant_error": max(
            record["frame_determinant_error"] for record in records
        ),
        "maximum_aligned_geometry_error_angstrom": max(
            record["aligned_geometry_error_angstrom"] for record in records
        ),
        "maximum_aligned_density_error": max(
            record["aligned_density_error"] for record in records
        ),
    }
    gates = {
        "eigenvalue_gap": (
            metrics["minimum_relative_eigenvalue_gap"]
            >= GATES["minimum_relative_eigenvalue_gap"]
        ),
        "frame_orthogonality": (
            metrics["maximum_frame_orthogonality_error"]
            <= GATES["maximum_frame_orthogonality_error"]
        ),
        "frame_determinant": (
            metrics["maximum_frame_determinant_error"]
            <= GATES["maximum_frame_determinant_error"]
        ),
        "canonical_geometry": (
            metrics["maximum_aligned_geometry_error_angstrom"]
            <= GATES["maximum_aligned_geometry_error_angstrom"]
        ),
        "canonical_density": (
            metrics["maximum_aligned_density_error"]
            <= GATES["maximum_aligned_density_error"]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "phase": "preflight",
        "provider_solve_count": 0,
        "mace_loaded": False,
        "cds_evaluated": False,
        "force_evaluated": False,
        "input_hashes": {
            MOL2_RELATIVE_PATH: EXPECTED_MOL2_SHA256,
            STATE_RELATIVE_PATH: EXPECTED_STATE_SHA256,
        },
        "atom_count": int(positions.shape[0]),
        "symbols": inputs["symbols"],
        "frozen_density_net_charge_e": float(
            np.sum(inputs["density_coefficients"][:, 0])
        ),
        "mol2_position_error_angstrom": inputs[
            "mol2_position_error_angstrom"
        ],
        "records": records,
        "metrics": metrics,
        "gates": gates,
        "status": "pass" if all(gates.values()) else "fail",
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
        raise RuntimeError("The pyddx extension hash differs from the frozen protocol.")
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": np.__version__,
        "pyddx_version": str(module.__version__),
        "pyddx_extension_path": str(module_path),
        "pyddx_extension_sha256": module_hash,
    }


def _frozen_lock() -> dict[str, Any]:
    head = _require_clean_committed_checkout()
    preflight = _preflight_payload()
    if preflight["status"] != "pass":
        raise RuntimeError("Preflight gates failed; refusing to freeze the canary.")
    runner_path = REPO_ROOT / RUNNER_RELATIVE_PATH
    document_path = REPO_ROOT / DOCUMENT_RELATIVE_PATH
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": _utc_now(),
        "one_shot": True,
        "no_retry_after_attempt_marker": True,
        "required_git_head": head,
        "required_clean_tracked_checkout": True,
        "runner": {
            "path": RUNNER_RELATIVE_PATH,
            "sha256": _sha256_file(runner_path),
        },
        "document": {
            "path": DOCUMENT_RELATIVE_PATH,
            "sha256": _sha256_file(document_path),
        },
        "inputs": {
            "mol2_path": MOL2_RELATIVE_PATH,
            "mol2_sha256": EXPECTED_MOL2_SHA256,
            "state_path": STATE_RELATIVE_PATH,
            "state_sha256": EXPECTED_STATE_SHA256,
        },
        "runtime": _runtime_provenance(),
        "provider": PROVIDER,
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
        "runner",
        "document",
        "inputs",
        "runtime",
        "provider",
        "rotations",
        "gates",
        "evaluation_budget",
        "preflight",
        "claim_boundary",
        "stop_condition",
    }
    if set(lock) != expected_keys:
        raise RuntimeError("Canary lock fields differ from the frozen schema.")
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported canary lock schema.")
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Canary lock protocol mismatch.")
    if lock.get("one_shot") is not True:
        raise RuntimeError("Canary lock no longer requires one-shot execution.")
    if lock.get("no_retry_after_attempt_marker") is not True:
        raise RuntimeError("Canary lock no longer forbids retries.")
    if lock.get("required_clean_tracked_checkout") is not True:
        raise RuntimeError("Canary lock no longer requires a clean checkout.")
    if lock.get("provider") != PROVIDER:
        raise RuntimeError("Canary provider settings differ from the runner.")
    if lock.get("rotations") != list(ROTATIONS):
        raise RuntimeError("Canary rotations differ from the runner.")
    if lock.get("gates") != GATES:
        raise RuntimeError("Canary gates differ from the runner.")
    if lock.get("evaluation_budget") != EVALUATION_BUDGET:
        raise RuntimeError("Canary evaluation budget differs from the runner.")
    if lock.get("claim_boundary") != CLAIM_BOUNDARY:
        raise RuntimeError("Canary claim boundary differs from the runner.")
    if lock.get("stop_condition") != STOP_CONDITION:
        raise RuntimeError("Canary stop condition differs from the runner.")
    expected_inputs = {
        "mol2_path": MOL2_RELATIVE_PATH,
        "mol2_sha256": EXPECTED_MOL2_SHA256,
        "state_path": STATE_RELATIVE_PATH,
        "state_sha256": EXPECTED_STATE_SHA256,
    }
    if lock.get("inputs") != expected_inputs:
        raise RuntimeError("Canary input declaration differs from the runner.")
    recomputed_preflight = _preflight_payload()
    if recomputed_preflight != lock.get("preflight"):
        raise RuntimeError(
            "Canary preflight differs from the frozen/recomputed protocol."
        )
    head = _require_clean_committed_checkout()
    if head != lock.get("required_git_head"):
        raise RuntimeError("Git head differs from the frozen one-shot lock.")
    for key, relative in (
        ("runner", RUNNER_RELATIVE_PATH),
        ("document", DOCUMENT_RELATIVE_PATH),
    ):
        record = lock[key]
        if record["path"] != relative:
            raise RuntimeError(f"Frozen {key} path changed.")
        if _sha256_file(REPO_ROOT / relative) != record["sha256"]:
            raise RuntimeError(f"Frozen {key} hash changed.")
    if _sha256_file(REPO_ROOT / MOL2_RELATIVE_PATH) != EXPECTED_MOL2_SHA256:
        raise RuntimeError("Frozen MOL2 hash changed.")
    if _sha256_file(REPO_ROOT / STATE_RELATIVE_PATH) != EXPECTED_STATE_SHA256:
        raise RuntimeError("Frozen state hash changed.")
    if _runtime_provenance() != lock["runtime"]:
        raise RuntimeError("Runtime provenance differs from the frozen lock.")
    if not lock_path.is_file():
        raise RuntimeError("Canary lock disappeared during verification.")


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


def _evaluate_scalar_case(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    density_coefficients: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = _new_response(
        positions_angstrom,
        radii_angstrom,
    )
    reaction_field_ev = response.apply_scf(density_coefficients)
    raw_reaction_field_ev = external_field_to_density_order(reaction_field_ev)
    paired_energy_ev = 0.5 * float(
        np.vdot(density_coefficients, raw_reaction_field_ev)
    )
    # This is a version-locked diagnostic read of the exact state just solved
    # by apply_scf().  It avoids a second continuum solve solely to repeat the
    # half-coupling identity already enforced inside the MAPLE adapter.
    state = response._scf_state
    if state is None:
        raise RuntimeError("The pyddx diagnostic state was not retained.")
    state_energy_ev = float(state.energy()) * Hartree
    identity_error_ev = abs(paired_energy_ev - state_energy_ev)
    model = response._model
    cavity_bohr = np.asarray(model.cavity, dtype=float).T.copy()
    sphere_centres_bohr = np.asarray(
        model.sphere_centres,
        dtype=float,
    ).T.copy()
    sphere_radii_bohr = np.asarray(model.sphere_radii, dtype=float).copy()
    elapsed = time.perf_counter() - started
    return {
        "energy_ev": state_energy_ev,
        "energy_kcal_mol": state_energy_ev / (kcal / mol),
        "paired_energy_ev": paired_energy_ev,
        "half_coupling_identity_error_ev": identity_error_ev,
        "n_cav": int(model.n_cav),
        "cavity_bohr": cavity_bohr,
        "sphere_centres_bohr": sphere_centres_bohr,
        "sphere_radii_bohr": sphere_radii_bohr,
        "elapsed_seconds": elapsed,
        "runtime_provenance": response.runtime_provenance,
    }


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
        raise RuntimeError("Auxiliary one-sphere Lebedev model has wrong shape.")
    norms = np.linalg.norm(directions, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 1.0e-12:
        raise RuntimeError("Auxiliary one-sphere cavity is not a unit grid.")
    return directions


def _active_mapping_signature(
    cavity_bohr: np.ndarray,
    sphere_centres_bohr: np.ndarray,
    sphere_radii_bohr: np.ndarray,
    lebedev_directions: np.ndarray,
) -> dict[str, Any]:
    if cavity_bohr.ndim != 2 or cavity_bohr.shape[1] != 3:
        raise RuntimeError("The active cavity must have shape (n_surface, 3).")
    if cavity_bohr.shape[0] == 0:
        raise RuntimeError("The active cavity must contain at least one point.")
    if sphere_centres_bohr.ndim != 2 or sphere_centres_bohr.shape[1] != 3:
        raise RuntimeError("Sphere centres must have shape (n_sphere, 3).")
    if sphere_centres_bohr.shape[0] < 2:
        raise RuntimeError("At least two spheres are required for owner-gap checks.")
    if sphere_radii_bohr.shape != (sphere_centres_bohr.shape[0],):
        raise RuntimeError("Sphere radii do not match the sphere centres.")
    if lebedev_directions.ndim != 2 or lebedev_directions.shape[1] != 3:
        raise RuntimeError("Lebedev directions must have shape (n_grid, 3).")
    if lebedev_directions.shape[0] < 2:
        raise RuntimeError(
            "At least two Lebedev directions are required for uniqueness checks."
        )
    arrays = (
        cavity_bohr,
        sphere_centres_bohr,
        sphere_radii_bohr,
        lebedev_directions,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("The active-mapping inputs must be finite.")
    if np.any(sphere_radii_bohr <= 0.0):
        raise RuntimeError("Sphere radii must be positive.")

    radial_residuals = np.abs(
        np.linalg.norm(
            cavity_bohr[:, None, :] - sphere_centres_bohr[None, :, :],
            axis=2,
        )
        - sphere_radii_bohr[None, :]
    )
    owners = np.argmin(radial_residuals, axis=1)
    owner_residual = radial_residuals[
        np.arange(cavity_bohr.shape[0]),
        owners,
    ]
    two_owner_residuals = np.partition(
        radial_residuals,
        kth=1,
        axis=1,
    )[:, :2]
    owner_second_gap = (
        two_owner_residuals[:, 1] - two_owner_residuals[:, 0]
    )
    directions = (
        cavity_bohr - sphere_centres_bohr[owners]
    ) / sphere_radii_bohr[owners, None]
    lebedev_indices = np.empty(cavity_bohr.shape[0], dtype=int)
    direction_errors = np.empty(cavity_bohr.shape[0], dtype=float)
    minimum_second_neighbor_distance = math.inf
    for start in range(0, cavity_bohr.shape[0], 512):
        stop = min(start + 512, cavity_bohr.shape[0])
        dots = directions[start:stop] @ lebedev_directions.T
        local_indices = np.argmax(dots, axis=1)
        two_largest_dots = np.partition(
            dots,
            kth=dots.shape[1] - 2,
            axis=1,
        )[:, -2:]
        second_best_dots = np.min(two_largest_dots, axis=1)
        lebedev_indices[start:stop] = local_indices
        direction_errors[start:stop] = np.linalg.norm(
            directions[start:stop] - lebedev_directions[local_indices],
            axis=1,
        )
        second_neighbor_distances = np.sqrt(
            np.maximum(0.0, 2.0 - 2.0 * second_best_dots)
        )
        minimum_second_neighbor_distance = min(
            minimum_second_neighbor_distance,
            float(np.min(second_neighbor_distances)),
        )
    pairs = np.column_stack((owners, lebedev_indices)).astype(np.int64)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    sorted_pairs = pairs[order]
    return {
        "sha256": hashlib.sha256(sorted_pairs.tobytes()).hexdigest(),
        "pair_count": int(sorted_pairs.shape[0]),
        "unique_pair_count": int(np.unique(sorted_pairs, axis=0).shape[0]),
        "maximum_surface_owner_residual_bohr": float(
            np.max(owner_residual)
        ),
        "minimum_surface_owner_second_gap_bohr": float(
            np.min(owner_second_gap)
        ),
        "maximum_lebedev_direction_error": float(
            np.max(direction_errors)
        ),
        "minimum_lebedev_second_neighbor_distance": (
            minimum_second_neighbor_distance
        ),
    }


def _public_case_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        not in {
            "cavity_bohr",
            "sphere_centres_bohr",
            "sphere_radii_bohr",
        }
    }


def _execute(lock: dict[str, Any], lock_path: Path) -> dict[str, Any]:
    inputs = _load_inputs()
    positions = inputs["positions_angstrom"]
    radii = inputs["cavity_radii_angstrom"]
    density_raw = inputs["density_coefficients"]
    density_cartesian = _raw_density_to_cartesian(density_raw)
    charges = inputs["nuclear_charges"]
    base_frame = _nuclear_charge_frame(positions, charges)
    center = base_frame["center_angstrom"]
    base_body_positions = base_frame["body_positions_angstrom"]
    base_body_density = np.column_stack(
        (
            density_cartesian[:, 0],
            density_cartesian[:, 1:] @ base_frame["orientation"],
        )
    )

    pyddx = importlib.import_module("pyddx")
    lebedev_directions = _lebedev_directions(pyddx)
    records = []
    provider_energy_cases = 0
    reference_candidate_mapping_sha256: str | None = None
    candidate_mapping_hashes: list[str] = []
    mapping_owner_residuals: list[float] = []
    mapping_owner_second_gaps: list[float] = []
    mapping_direction_errors: list[float] = []
    mapping_second_neighbor_distances: list[float] = []
    mapping_pair_counts: list[int] = []
    mapping_unique_pair_counts: list[int] = []
    candidate_geometry_errors: list[float] = []
    candidate_density_errors: list[float] = []

    for specification in ROTATIONS:
        rotation = _axis_angle_rotation(
            specification["axis"],
            specification["angle_degrees"],
        )
        rotated_positions = center + (positions - center) @ rotation.T
        rotated_density_cartesian = np.column_stack(
            (
                density_cartesian[:, 0],
                density_cartesian[:, 1:] @ rotation.T,
            )
        )
        rotated_density_raw = _cartesian_density_to_raw(
            rotated_density_cartesian
        )

        control = _evaluate_scalar_case(
            rotated_positions,
            radii,
            rotated_density_raw,
        )
        provider_energy_cases += 1

        frame = _nuclear_charge_frame(rotated_positions, charges)
        body_density_cartesian = np.column_stack(
            (
                rotated_density_cartesian[:, 0],
                rotated_density_cartesian[:, 1:] @ frame["orientation"],
            )
        )
        body_density_raw = _cartesian_density_to_raw(
            body_density_cartesian
        )
        candidate = _evaluate_scalar_case(
            frame["body_positions_angstrom"],
            radii,
            body_density_raw,
        )
        provider_energy_cases += 1

        signed_alignment, geometry_error = _proper_signed_axis_align(
            frame["body_positions_angstrom"],
            base_body_positions,
        )
        aligned_body_density = np.column_stack(
            (
                body_density_cartesian[:, 0],
                body_density_cartesian[:, 1:] @ signed_alignment,
            )
        )
        density_error = float(
            np.max(np.abs(aligned_body_density - base_body_density))
        )
        aligned_cavity = candidate["cavity_bohr"] @ signed_alignment
        aligned_centres = (
            candidate["sphere_centres_bohr"] @ signed_alignment
        )
        mapping = _active_mapping_signature(
            aligned_cavity,
            aligned_centres,
            candidate["sphere_radii_bohr"],
            lebedev_directions,
        )
        if reference_candidate_mapping_sha256 is None:
            reference_candidate_mapping_sha256 = mapping["sha256"]
        candidate_mapping_hashes.append(mapping["sha256"])
        mapping_owner_residuals.append(
            mapping["maximum_surface_owner_residual_bohr"]
        )
        mapping_owner_second_gaps.append(
            mapping["minimum_surface_owner_second_gap_bohr"]
        )
        mapping_direction_errors.append(
            mapping["maximum_lebedev_direction_error"]
        )
        mapping_second_neighbor_distances.append(
            mapping["minimum_lebedev_second_neighbor_distance"]
        )
        mapping_pair_counts.append(mapping["pair_count"])
        mapping_unique_pair_counts.append(mapping["unique_pair_count"])
        candidate_geometry_errors.append(geometry_error)
        candidate_density_errors.append(density_error)

        records.append(
            {
                "rotation": specification,
                "frame": {
                    "eigenvalues": frame["eigenvalues"].tolist(),
                    "relative_minimum_eigenvalue_gap": frame[
                        "relative_minimum_eigenvalue_gap"
                    ],
                    "orthogonality_error": frame[
                        "orthogonality_error"
                    ],
                    "determinant_error": frame["determinant_error"],
                    "proper_signed_axis_alignment": (
                        signed_alignment.tolist()
                    ),
                    "aligned_geometry_error_angstrom": geometry_error,
                    "aligned_density_error": density_error,
                },
                "control": _public_case_record(control),
                "candidate": {
                    **_public_case_record(candidate),
                    "aligned_active_mapping": mapping,
                },
            }
        )

    if provider_energy_cases != lock["evaluation_budget"][
        "total_provider_energy_cases"
    ]:
        raise RuntimeError("Provider evaluation count violated the lock.")

    control_energies = [
        record["control"]["energy_ev"] for record in records
    ]
    candidate_energies = [
        record["candidate"]["energy_ev"] for record in records
    ]
    control_span_ev = max(control_energies) - min(control_energies)
    candidate_span_ev = max(candidate_energies) - min(candidate_energies)
    identity_profile_shift_kcal_mol = abs(
        records[0]["candidate"]["energy_kcal_mol"]
        - records[0]["control"]["energy_kcal_mol"]
    )
    identity_errors = [
        case[branch]["half_coupling_identity_error_ev"]
        for case in records
        for branch in ("control", "candidate")
    ]
    candidate_n_cav = [
        record["candidate"]["n_cav"] for record in records
    ]

    metrics = {
        "provider_energy_case_count": provider_energy_cases,
        "control_rotation_span_ev": control_span_ev,
        "control_rotation_span_kcal_mol": control_span_ev / (kcal / mol),
        "candidate_rotation_span_ev": candidate_span_ev,
        "candidate_rotation_span_kcal_mol": (
            candidate_span_ev / (kcal / mol)
        ),
        "identity_profile_shift_kcal_mol": (
            identity_profile_shift_kcal_mol
        ),
        "maximum_half_coupling_identity_error_ev": max(identity_errors),
        "candidate_n_cav": candidate_n_cav,
        "candidate_active_mapping_hashes": candidate_mapping_hashes,
        "maximum_surface_owner_residual_bohr": max(
            mapping_owner_residuals
        ),
        "minimum_surface_owner_second_gap_bohr": min(
            mapping_owner_second_gaps
        ),
        "maximum_lebedev_direction_error": max(
            mapping_direction_errors
        ),
        "minimum_lebedev_second_neighbor_distance": min(
            mapping_second_neighbor_distances
        ),
        "candidate_active_mapping_pair_counts": mapping_pair_counts,
        "candidate_active_mapping_unique_pair_counts": (
            mapping_unique_pair_counts
        ),
        "maximum_aligned_geometry_error_angstrom": max(
            candidate_geometry_errors
        ),
        "maximum_aligned_density_error": max(
            candidate_density_errors
        ),
        "total_case_elapsed_seconds": sum(
            record[branch]["elapsed_seconds"]
            for record in records
            for branch in ("control", "candidate")
        ),
    }
    gates = {
        "exact_evaluation_budget": (
            metrics["provider_energy_case_count"]
            == lock["evaluation_budget"]["total_provider_energy_cases"]
        ),
        "frame_preflight": lock["preflight"]["status"] == "pass",
        "half_coupling_identity": (
            metrics["maximum_half_coupling_identity_error_ev"]
            <= GATES["maximum_half_coupling_identity_error_ev"]
        ),
        "candidate_rotation_span": (
            metrics["candidate_rotation_span_ev"]
            <= GATES["maximum_candidate_rotation_span_ev"]
        ),
        "identity_profile_shift": (
            metrics["identity_profile_shift_kcal_mol"]
            <= GATES["maximum_identity_profile_shift_kcal_mol"]
        ),
        "candidate_cavity_size": len(set(candidate_n_cav)) == 1,
        "candidate_active_mapping": (
            len(set(candidate_mapping_hashes)) == 1
            and reference_candidate_mapping_sha256
            == candidate_mapping_hashes[0]
        ),
        "surface_owner_mapping": (
            metrics["maximum_surface_owner_residual_bohr"]
            <= GATES["maximum_surface_owner_residual_bohr"]
            and metrics["minimum_surface_owner_second_gap_bohr"]
            >= GATES["minimum_surface_owner_second_gap_bohr"]
        ),
        "lebedev_direction_mapping": (
            metrics["maximum_lebedev_direction_error"]
            <= GATES["maximum_lebedev_direction_error"]
            and metrics["minimum_lebedev_second_neighbor_distance"]
            >= GATES["minimum_lebedev_second_neighbor_distance"]
        ),
        "active_mapping_pairs_unique": all(
            unique == count
            for unique, count in zip(
                metrics["candidate_active_mapping_unique_pair_counts"],
                metrics["candidate_active_mapping_pair_counts"],
                strict=True,
            )
        ),
        "canonical_geometry": (
            metrics["maximum_aligned_geometry_error_angstrom"]
            <= GATES["maximum_aligned_geometry_error_angstrom"]
        ),
        "canonical_density": (
            metrics["maximum_aligned_density_error"]
            <= GATES["maximum_aligned_density_error"]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "completed_at_utc": _utc_now(),
        "git_head": lock["required_git_head"],
        "lock_path": str(lock_path.relative_to(REPO_ROOT)),
        "lock_sha256": _sha256_file(lock_path),
        "provider": PROVIDER,
        "records": records,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": sorted(
            name for name, passed in gates.items() if not passed
        ),
        "status": "pass" if all(gates.values()) else "fail",
        "claim_boundary": lock["claim_boundary"],
        "stop_condition": lock["stop_condition"],
    }


def _command_preflight(args: argparse.Namespace) -> None:
    payload = _preflight_payload()
    output = Path(args.output).resolve() if args.output else None
    if output is not None:
        _write_atomic_json(output, payload)
    print(_canonical_json_bytes(payload).decode("utf-8"), end="")
    if payload["status"] != "pass":
        raise SystemExit(1)


def _command_freeze(args: argparse.Namespace) -> None:
    del args
    work_dir = DEFAULT_WORK_DIR
    lock_path = work_dir / "lock.json"
    if lock_path.exists():
        raise FileExistsError(f"One-shot lock already exists: {lock_path}")
    lock = _frozen_lock()
    _write_exclusive_json(lock_path, lock)
    print(
        json.dumps(
            {
                "lock_path": str(lock_path),
                "lock_sha256": _sha256_file(lock_path),
                "required_git_head": lock["required_git_head"],
                "status": "frozen",
            },
            sort_keys=True,
        )
    )


def _command_run(args: argparse.Namespace) -> None:
    del args
    work_dir = DEFAULT_WORK_DIR
    lock_path = work_dir / "lock.json"
    attempt_path = work_dir / "attempt.json"
    result_path = work_dir / "results.json"
    failure_path = work_dir / "failure.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Freeze the one-shot lock before running.")
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError(
            "This one-shot canary already has an attempt/result artifact; "
            "retries are forbidden."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _verify_lock(lock, lock_path)
    _write_exclusive_json(
        attempt_path,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "started_at_utc": _utc_now(),
            "git_head": lock["required_git_head"],
            "lock_sha256": _sha256_file(lock_path),
        },
    )
    try:
        result = _execute(lock, lock_path)
        _write_exclusive_json(result_path, result)
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "failed_at_utc": _utc_now(),
            "git_head": lock["required_git_head"],
            "lock_sha256": _sha256_file(lock_path),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "status": "execution-failure-no-retry",
        }
        _write_exclusive_json(failure_path, failure)
        raise
    print(_canonical_json_bytes(result).decode("utf-8"), end="")
    if result["status"] != "pass":
        raise SystemExit(1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="check hashes and body-frame conditioning without a provider solve",
    )
    preflight.add_argument("--output")
    preflight.set_defaults(function=_command_preflight)

    freeze = subparsers.add_parser(
        "freeze",
        help="write the immutable one-shot lock on a clean committed head",
    )
    freeze.set_defaults(function=_command_freeze)

    run = subparsers.add_parser(
        "run",
        help="consume the frozen lock exactly once and evaluate six scalar cases",
    )
    run.set_defaults(function=_command_run)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
