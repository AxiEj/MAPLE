#!/usr/bin/env python3
"""Fixed-topology torsion-force gate on the historical flexible failure case.

This runner deliberately revisits the 2-acetoxyethyl-acetate central C3--C4
torsion that exposed the pyddx active-set discontinuity.  It does *not* reuse
the old ddPCM values: every point is evaluated through the new fixed-cardinality
FC-aSWIG C-PCM direct ledger and the D2-canonical MACE boundary.

For a rigid downstream rotation by angle theta, the total declared scalar is

  E(theta) = E_MACE,gas + 1/2<c_MACE,f_reac> + G_fixed-topology-SMD-CDS,

and the analytic generalized force is F_theta = sum_i F_i . dr_i/dtheta.
Seven predeclared angles supply three central finite differences at theta=0
and a five-point local torsional path.  All points must have a nominal SCF
root, agreement across the three required starts, and the same fixed surface
dimension.  This is evidence for a restricted research force domain; it does
not assert that the legacy learned fixed point is a common stationary
MACE--PCM electronic free energy or immediately open a public force API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from ase.units import Hartree

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_route2_fc_aswig_d2_force_kinematics as common


SCHEMA_VERSION = 1
PROTOCOL_ID = "route2-fc-aswig-d2-mace-flexible-torsion-force-v1"
COMPOUND_ID = "mobley_352111"

# The historical failure coordinate, frozen before evaluating FC-aSWIG:
# downstream rotation about the central C3--C4 single bond of
# 2-acetoxyethyl acetate.  Indices are zero based.
AXIS_ORIGIN_INDEX = 4
AXIS_DIRECTION_INDEX = 5
ROTATED_FRAGMENT_INDICES = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)
ANGLES_DEGREES = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
GATES = {
    "maximum_centered_fd_error_ev_per_radian": 2.0e-4,
    "energy_force_fd_noise_floor_ev_per_radian": 2.0e-6,
    "maximum_fine_to_coarse_error_ratio": 0.40,
    "require_fixed_surface_cardinality": True,
    "require_nominal_roots": True,
    "require_local_multistart_agreement": True,
    "maximum_net_force_hartree_per_angstrom": 1.0e-10,
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".omx"
    / "benchmarks"
    / "route2-fc-aswig-d2-mace-flexible-torsion-force"
    / "result.json"
)


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rotate_fragment(
    positions: np.ndarray,
    *,
    angle_radians: float,
    axis_origin_index: int = AXIS_ORIGIN_INDEX,
    axis_direction_index: int = AXIS_DIRECTION_INDEX,
    rotated_fragment_indices: tuple[int, ...] = ROTATED_FRAGMENT_INDICES,
) -> np.ndarray:
    """Return one absolute-angle rigid downstream torsion geometry."""

    source = np.asarray(positions, dtype=float)
    rotated = np.array(source, copy=True)
    origin = source[axis_origin_index]
    direction = source[axis_direction_index] - origin
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= 1.0e-12:
        raise ValueError("Central torsion axis is degenerate.")
    unit_axis = direction / length
    fragment = np.asarray(rotated_fragment_indices, dtype=int)
    relative = source[fragment] - origin
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    rotated[fragment] = (
        relative * cosine
        + np.cross(unit_axis, relative) * sine
        + np.outer(relative @ unit_axis, unit_axis) * (1.0 - cosine)
        + origin
    )
    return rotated


def torsional_tangent(
    positions: np.ndarray,
    *,
    axis_origin_index: int = AXIS_ORIGIN_INDEX,
    axis_direction_index: int = AXIS_DIRECTION_INDEX,
    rotated_fragment_indices: tuple[int, ...] = ROTATED_FRAGMENT_INDICES,
) -> np.ndarray:
    """Return dr/dtheta in Angstrom/radian for the absolute-angle torsion."""

    coordinates = np.asarray(positions, dtype=float)
    origin = coordinates[axis_origin_index]
    direction = coordinates[axis_direction_index] - origin
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= 1.0e-12:
        raise ValueError("Central torsion axis is degenerate.")
    tangent = np.zeros_like(coordinates)
    fragment = np.asarray(rotated_fragment_indices, dtype=int)
    tangent[fragment] = np.cross(
        direction / length,
        coordinates[fragment] - origin,
    )
    return tangent


def _total_sample(atoms, calculator, settings: dict[str, Any], angle_degrees: float) -> dict[str, Any]:
    """Evaluate the one declared total scalar and its analytic total force."""

    result = common._evaluate(atoms, calculator, settings["solv"], with_force=True)
    root = common._nominal_root(result)
    correction_force = np.asarray(result.forces_hartree_per_angstrom, dtype=float)
    if correction_force.shape != (len(atoms), 3) or not np.all(np.isfinite(correction_force)):
        raise RuntimeError("Flexible torsion received an invalid correction force.")

    gas_state, _ = calculator.polar_state(atoms, compute_forces=True)
    gas_force_ev = gas_state.fixed_field_forces_ev_per_angstrom
    if gas_force_ev is None:
        raise RuntimeError("MACE-POLAR did not return the gas fixed-field force.")
    gas_force = np.asarray(gas_force_ev, dtype=float) / Hartree
    if gas_force.shape != correction_force.shape or not np.all(np.isfinite(gas_force)):
        raise RuntimeError("Flexible torsion received an invalid gas force.")

    total_force = gas_force + correction_force
    tangent = torsional_tangent(atoms.get_positions())
    surface_size = result.provenance["continuum_provider"].get("surface_size")
    if not isinstance(surface_size, int) or surface_size <= 0:
        raise RuntimeError("Fixed-topology continuum omitted a positive surface size.")
    return {
        "angle_degrees": float(angle_degrees),
        "angle_radians": float(np.deg2rad(angle_degrees)),
        "total_energy_hartree": float(gas_state.energy_ev / Hartree + result.energy_hartree),
        "gas_energy_hartree": float(gas_state.energy_ev / Hartree),
        "correction_energy_hartree": float(result.energy_hartree),
        "analytic_generalized_force_hartree_per_radian": float(np.sum(total_force * tangent)),
        "total_force_hartree_per_angstrom": total_force.tolist(),
        "surface_size": surface_size,
        "scf_reason": str(root["scf_convergence"]["reason"]),
        "local_multistart_agreed": bool(root["multi_start_root_agreement"]["agreed"]),
        "force_sum_hartree_per_angstrom": np.sum(total_force, axis=0).tolist(),
    }


def _centered_differences(samples: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare three fixed central energy slopes to the center torque."""

    center = samples[0.0]
    analytic = float(center["analytic_generalized_force_hartree_per_radian"])
    rows: list[dict[str, Any]] = []
    for step_degrees in (1.0, 0.5, 0.25):
        plus = samples[step_degrees]
        minus = samples[-step_degrees]
        step_radians = float(np.deg2rad(step_degrees))
        numeric = -(
            float(plus["total_energy_hartree"])
            - float(minus["total_energy_hartree"])
        ) / (2.0 * step_radians)
        error = abs(numeric - analytic)
        rows.append(
            {
                "step_degrees": step_degrees,
                "step_radians": step_radians,
                "analytic_generalized_force_hartree_per_radian": analytic,
                "centered_fd_generalized_force_hartree_per_radian": float(numeric),
                "absolute_error_hartree_per_radian": float(error),
                "absolute_error_ev_per_radian": float(error * Hartree),
            }
        )
    return rows


def _path_derivatives(samples: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the half-degree path mesh to check three interior analytic torques."""

    records: list[dict[str, Any]] = []
    values = (-1.0, -0.5, 0.0, 0.5, 1.0)
    for previous_angle, current_angle, following_angle in zip(
        values[:-2], values[1:-1], values[2:], strict=True
    ):
        previous = samples[previous_angle]
        current = samples[current_angle]
        following = samples[following_angle]
        step_radians = float(np.deg2rad(following_angle - current_angle))
        numeric = -(
            float(following["total_energy_hartree"])
            - float(previous["total_energy_hartree"])
        ) / (2.0 * step_radians)
        error = abs(numeric - float(current["analytic_generalized_force_hartree_per_radian"]))
        records.append(
            {
                "angle_degrees": current_angle,
                "analytic_generalized_force_hartree_per_radian": current[
                    "analytic_generalized_force_hartree_per_radian"
                ],
                "centered_fd_generalized_force_hartree_per_radian": float(numeric),
                "absolute_error_ev_per_radian": float(error * Hartree),
            }
        )
    return records


def _gate(samples: dict[float, dict[str, Any]], central: list[dict[str, Any]], path: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(row["absolute_error_ev_per_radian"]) for row in central]
    coarse_to_fine = [
        errors[index + 1]
        / max(errors[index], GATES["energy_force_fd_noise_floor_ev_per_radian"])
        for index in range(len(errors) - 1)
    ]
    trend = [
        errors[index] <= GATES["energy_force_fd_noise_floor_ev_per_radian"]
        and errors[index + 1] <= GATES["energy_force_fd_noise_floor_ev_per_radian"]
        or errors[index + 1]
        <= GATES["maximum_fine_to_coarse_error_ratio"] * errors[index]
        + GATES["energy_force_fd_noise_floor_ev_per_radian"]
        for index in range(len(errors) - 1)
    ]
    cardinalities = {int(row["surface_size"]) for row in samples.values()}
    maximum_net_force = max(
        float(np.max(np.abs(np.asarray(row["force_sum_hartree_per_angstrom"], dtype=float))))
        for row in samples.values()
    )
    maximum_path_error = max(float(row["absolute_error_ev_per_radian"]) for row in path)
    checks = {
        "fixed_surface_cardinality": len(cardinalities) == 1,
        "nominal_roots": all(
            row["scf_reason"] == "nominal-density-and-energy-v1"
            for row in samples.values()
        ),
        "local_multistart_agreement": all(
            bool(row["local_multistart_agreed"]) for row in samples.values()
        ),
        "net_force": maximum_net_force <= GATES["maximum_net_force_hartree_per_angstrom"],
        "centered_energy_force_agreement": max(errors)
        <= GATES["maximum_centered_fd_error_ev_per_radian"],
        "second_order_finite_difference_trend": all(trend),
        "path_energy_force_agreement": maximum_path_error
        <= GATES["maximum_centered_fd_error_ev_per_radian"],
    }
    return {
        "gates": dict(GATES),
        "checks": {name: bool(value) for name, value in checks.items()},
        "central_fd_errors_ev_per_radian": errors,
        "fine_over_coarse_error_ratios": coarse_to_fine,
        "second_order_trend_checks": [bool(value) for value in trend],
        "maximum_path_error_ev_per_radian": maximum_path_error,
        "surface_cardinalities": sorted(cardinalities),
        "maximum_net_force_hartree_per_angstrom": maximum_net_force,
        "passed": bool(all(checks.values())),
        "remaining_force_admission_scope": (
            "A second torsional closed-loop check and final clean-tree replay "
            "remain required before any narrow public force admission."
        ),
    }


def run(*, prepared_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    base, candidate, mol2_path, mol2_digest = common._load_atoms(
        prepared_path,
        compound_id=COMPOUND_ID,
    )
    settings = common._settings()
    calculator = common._calculator(base, settings, device)
    samples: dict[float, dict[str, Any]] = {}
    source_positions = np.asarray(base.get_positions(), dtype=float)
    for angle_degrees in ANGLES_DEGREES:
        atoms = base.copy()
        atoms.set_positions(
            rotate_fragment(
                source_positions,
                angle_radians=float(np.deg2rad(angle_degrees)),
            )
        )
        samples[angle_degrees] = _total_sample(
            atoms,
            calculator,
            settings,
            angle_degrees,
        )
    central = _centered_differences(samples)
    path = _path_derivatives(samples)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc(),
        "git_head": common._git_head(),
        "git_worktree_status": common._git_worktree_status(),
        "runner": str(Path(__file__).relative_to(REPO_ROOT)),
        "runner_sha256": _sha256(Path(__file__)),
        "shared_kinematics_runner": str(Path(common.__file__).resolve().relative_to(REPO_ROOT)),
        "shared_kinematics_runner_sha256": _sha256(Path(common.__file__).resolve()),
        "claim_boundary": (
            "This is a frozen flexible torsion test of the declared direct-PCM "
            "total operational scalar. It neither establishes a common stationary "
            "MACE--PCM electronic free energy nor opens a universal public force API."
        ),
        "record": {
            "compound_id": COMPOUND_ID,
            "name": candidate["name"],
            "atom_count": len(base),
            "mol2_path": str(mol2_path),
            "mol2_sha256": mol2_digest,
        },
        "profile": common.FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
        "total_declared_scalar": (
            "E_MACE-POLAR(gas)+0.5*<c_MACE-POLAR,f_reac_CPCM>+"
            "fixed-topology-aqueous-SMD-CDS"
        ),
        "torsion": {
            "axis_origin_index_zero_based": AXIS_ORIGIN_INDEX,
            "axis_direction_index_zero_based": AXIS_DIRECTION_INDEX,
            "rotated_fragment_indices_zero_based": list(ROTATED_FRAGMENT_INDICES),
            "angle_convention": "positive-right-hand-rule-about-C3-C4",
            "angles_degrees": list(ANGLES_DEGREES),
        },
        "samples": [samples[angle] for angle in ANGLES_DEGREES],
        "central_energy_force": central,
        "path_energy_force": path,
    }
    payload["flexible_torsion_gate"] = _gate(samples, central, path)
    _write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=common.DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    arguments = parser.parse_args()
    if arguments.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = arguments.device
    payload = run(
        prepared_path=arguments.prepared.resolve(),
        output_path=arguments.output.resolve(),
        device=device,
    )
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
