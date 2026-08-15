"""Shared execution and trajectory contract for molecular IRC paths.

The four legacy integrators differ in their propagation formula, but a path
has one common scientific ledger: two explicitly terminated downhill branches,
the exact validated transition-state record, endpoint force convergence, and
an endpoint-to-endpoint trajectory with the transition state inserted once.
"""

from __future__ import annotations

import math
import os
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

from ...utility.xyz_io import write_xyz
from .parameters import (
    DEFAULT_IRC_MAX_FORCE_HARTREE_PER_ANGSTROM,
    DEFAULT_IRC_MAX_STEPS,
    DEFAULT_IRC_RMS_FORCE_HARTREE_PER_ANGSTROM,
    IRCPathParams,
    PATH_ENERGY_ABSOLUTE_TOLERANCE_HARTREE,
    coerce_nonnegative_float,
    coerce_positive_float,
    coerce_strict_bool,
    validate_irc_path_params,
)

KCAL_PER_HARTREE = 627.509474
_DIRECTIONS = ("forward", "backward")
_TERMINATION_REASONS = (
    "force_converged",
    "maximum_steps",
    "predictor_stalled",
    "step_too_small",
    "zero_gradient",
)


def irc_force_criteria_satisfied(
    *,
    maximum_force_hartree_per_A: object,
    rms_force_hartree_per_A: object,
    maximum_force_threshold_hartree_per_A: object,
    rms_force_threshold_hartree_per_A: object,
) -> bool:
    """Evaluate the shared dual Cartesian-force convergence criterion."""

    maximum_force = coerce_nonnegative_float(
        maximum_force_hartree_per_A,
        "maximum_force_hartree_per_A",
    )
    rms_force = coerce_nonnegative_float(
        rms_force_hartree_per_A,
        "rms_force_hartree_per_A",
    )
    maximum_threshold = coerce_positive_float(
        maximum_force_threshold_hartree_per_A,
        "maximum_force_threshold_hartree_per_A",
    )
    rms_threshold = coerce_positive_float(
        rms_force_threshold_hartree_per_A,
        "rms_force_threshold_hartree_per_A",
    )
    return bool(maximum_force <= maximum_threshold and rms_force <= rms_threshold)


def make_irc_record(
    *,
    energy_hartree: object,
    forces_hartree_per_A: object,
    positions_angstrom: object,
    point_kind: str = "path",
) -> dict:
    """Create one finite, unit-explicit legacy IRC record."""

    try:
        energy = float(energy_hartree)
    except (TypeError, ValueError) as exc:
        raise ValueError("IRC record energy must be finite.") from exc
    if not math.isfinite(energy):
        raise ValueError("IRC record energy must be finite.")

    positions = np.asarray(positions_angstrom, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("IRC record positions must be finite with shape (N, 3).")
    forces = np.asarray(forces_hartree_per_A, dtype=float)
    if forces.shape == (positions.size,):
        forces = forces.reshape(positions.shape)
    if forces.shape != positions.shape or not np.all(np.isfinite(forces)):
        raise ValueError("IRC record forces must be finite with shape (N, 3).")
    if point_kind not in {"path", "transition_state"}:
        raise ValueError("IRC record point_kind must be 'path' or 'transition_state'.")

    positions = np.array(positions, copy=True)
    positions.setflags(write=False)
    forces = np.array(forces, copy=True)
    forces.setflags(write=False)
    return {
        "E": energy,
        "maxG": float(np.max(np.abs(forces), initial=0.0)),
        "rmsG": float(np.sqrt(np.mean(forces**2))) if forces.size else 0.0,
        "x": positions,
        "forces_hartree_per_A": forces,
        "point_kind": point_kind,
    }


def _normalized_existing_record(record: Mapping[str, object]) -> dict:
    try:
        energy = float(record["E"])
        maximum_force = float(record["maxG"])
        rms_force = float(record["rmsG"])
        positions = np.asarray(record["x"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("IRC path record has an invalid legacy schema.") from exc
    if (
        not math.isfinite(energy)
        or not math.isfinite(maximum_force)
        or not math.isfinite(rms_force)
        or maximum_force < 0.0
        or rms_force < 0.0
        or positions.ndim != 2
        or positions.shape[1:] != (3,)
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "IRC path record must contain finite energy, forces, and (N, 3) positions."
        )
    normalized_positions = np.array(positions, copy=True)
    normalized_positions.setflags(write=False)
    normalized = {
        "E": energy,
        "maxG": maximum_force,
        "rmsG": rms_force,
        "x": normalized_positions,
        "point_kind": "path",
    }
    if "forces_hartree_per_A" in record:
        forces = np.asarray(record["forces_hartree_per_A"], dtype=float)
        if forces.shape != normalized_positions.shape or not np.all(
            np.isfinite(forces)
        ):
            raise ValueError(
                "IRC record forces_hartree_per_A must be finite with shape (N, 3)."
            )
        normalized_forces = np.array(forces, copy=True)
        normalized_forces.setflags(write=False)
        normalized["forces_hartree_per_A"] = normalized_forces
    return normalized


def finalize_irc_branch(
    *,
    title: str,
    direction: str,
    records: Sequence[Mapping[str, object]],
    transition_state_energy_hartree: object,
    termination_reason: str,
    iterations_attempted: object,
    maximum_force_threshold_hartree_per_A: object,
    rms_force_threshold_hartree_per_A: object,
) -> dict:
    """Attach an explicit, criteria-derived termination status to one branch."""

    if direction not in _DIRECTIONS:
        raise ValueError("IRC branch direction must be 'forward' or 'backward'.")
    if termination_reason not in _TERMINATION_REASONS:
        raise ValueError("IRC branch has an unsupported termination reason.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("IRC branch title must be non-empty.")
    normalized_records = [_normalized_existing_record(record) for record in records]
    if not normalized_records:
        raise ValueError("IRC branch must contain at least one displaced path point.")
    shape = normalized_records[0]["x"].shape
    if any(record["x"].shape != shape for record in normalized_records[1:]):
        raise ValueError("IRC branch records must have one consistent molecular shape.")

    try:
        transition_state_energy = float(transition_state_energy_hartree)
    except (TypeError, ValueError) as exc:
        raise ValueError("IRC transition-state energy must be finite.") from exc
    if not math.isfinite(transition_state_energy):
        raise ValueError("IRC transition-state energy must be finite.")
    if isinstance(iterations_attempted, bool):
        raise ValueError("iterations_attempted must be a non-negative integer.")
    try:
        iterations = int(iterations_attempted)
        numeric_iterations = float(iterations_attempted)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "iterations_attempted must be a non-negative integer."
        ) from exc
    if (
        not math.isfinite(numeric_iterations)
        or numeric_iterations != iterations
        or iterations < 0
    ):
        raise ValueError("iterations_attempted must be a non-negative integer.")

    maximum_threshold = coerce_positive_float(
        maximum_force_threshold_hartree_per_A,
        "maximum_force_threshold_hartree_per_A",
    )
    rms_threshold = coerce_positive_float(
        rms_force_threshold_hartree_per_A,
        "rms_force_threshold_hartree_per_A",
    )
    final = normalized_records[-1]
    force_criteria_satisfied = irc_force_criteria_satisfied(
        maximum_force_hartree_per_A=final["maxG"],
        rms_force_hartree_per_A=final["rmsG"],
        maximum_force_threshold_hartree_per_A=maximum_threshold,
        rms_force_threshold_hartree_per_A=rms_threshold,
    )
    if termination_reason == "force_converged" and not force_criteria_satisfied:
        raise RuntimeError(
            "IRC branch declared force convergence without satisfying both force thresholds."
        )
    converged = bool(
        termination_reason == "force_converged" and force_criteria_satisfied
    )

    return {
        "title": title,
        "direction": direction,
        "records": normalized_records,
        "E_ts": transition_state_energy,
        "status": {
            "converged": converged,
            "force_criteria_satisfied": force_criteria_satisfied,
            "termination_reason": termination_reason,
            "iterations_attempted": iterations,
            "accepted_macro_steps": max(len(normalized_records) - 1, 0),
            "final_maximum_force_hartree_per_A": final["maxG"],
            "final_rms_force_hartree_per_A": final["rmsG"],
            "maximum_force_threshold_hartree_per_A": maximum_threshold,
            "rms_force_threshold_hartree_per_A": rms_threshold,
        },
    }


def _validated_branch(branch: Mapping[str, object], direction: str) -> None:
    if branch.get("direction") != direction:
        raise ValueError(f"IRC {direction} branch has the wrong direction label.")
    records = branch.get("records")
    status = branch.get("status")
    if not isinstance(records, list) or not records:
        raise ValueError(f"IRC {direction} branch has no path records.")
    if not isinstance(status, Mapping) or type(status.get("converged")) is not bool:
        raise ValueError(f"IRC {direction} branch has no validated termination status.")


def _validated_transition_state_record(record: Mapping[str, object]) -> dict:
    if record.get("point_kind") != "transition_state":
        raise ValueError("IRC path requires an explicit transition-state record.")
    normalized = _normalized_existing_record(record)
    if "forces_hartree_per_A" not in normalized:
        raise ValueError(
            "IRC transition-state record must preserve its Cartesian forces."
        )
    normalized["point_kind"] = "transition_state"
    return normalized


def assemble_irc_path(
    *,
    method_label: str,
    forward: Mapping[str, object],
    backward: Mapping[str, object],
    transition_state_record: Mapping[str, object],
    path_energy_tolerance_hartree: object = PATH_ENERGY_ABSOLUTE_TOLERANCE_HARTREE,
) -> dict:
    """Assemble endpoint-to-TS-to-endpoint order with the exact TS once."""

    _validated_branch(forward, "forward")
    _validated_branch(backward, "backward")
    if not isinstance(method_label, str) or not method_label.strip():
        raise ValueError("IRC method label must be non-empty.")
    transition_state = _validated_transition_state_record(transition_state_record)
    path_energy_tolerance = coerce_positive_float(
        path_energy_tolerance_hartree,
        "path_energy_tolerance_hartree",
    )
    ts_positions = transition_state["x"]
    expected_shape = ts_positions.shape
    all_path_records = list(forward["records"]) + list(backward["records"])
    if any(
        np.asarray(record["x"]).shape != expected_shape for record in all_path_records
    ):
        raise ValueError("IRC path records do not match the transition-state molecule.")
    coordinate_scale = max(1.0, float(np.max(np.abs(ts_positions), initial=0.0)))
    displacement_tolerance = 8.0 * np.finfo(float).eps * coordinate_scale
    for direction, branch in (("forward", forward), ("backward", backward)):
        for point_index, record in enumerate(branch["records"]):
            displacement_from_ts = float(
                np.max(
                    np.abs(np.asarray(record["x"]) - ts_positions),
                    initial=0.0,
                )
            )
            if displacement_from_ts <= displacement_tolerance:
                raise ValueError(
                    f"IRC {direction} branch path point {point_index} is not "
                    "numerically distinct from the transition state."
                )

    ts_energy = float(transition_state["E"])
    for direction, branch in (("forward", forward), ("backward", backward)):
        try:
            branch_ts_energy = float(branch["E_ts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"IRC {direction} branch has no finite transition-state energy."
            ) from exc
    all_path_energies = [float(record["E"]) for record in all_path_records]
    for direction, branch in (("forward", forward), ("backward", backward)):
        branch_ts_energy = float(branch["E_ts"])
        if not math.isfinite(branch_ts_energy) or not math.isclose(
            branch_ts_energy,
            ts_energy,
            rel_tol=0.0,
            abs_tol=path_energy_tolerance,
        ):
            raise ValueError(
                f"IRC {direction} branch does not share the validated "
                "transition-state energy."
            )
    highest_path_energy = max(all_path_energies)
    if highest_path_energy > ts_energy + path_energy_tolerance:
        raise ValueError(
            "IRC path contains a point above the validated transition state; "
            "the selected displacement/integration is not demonstrably downhill."
        )

    forward_endpoint = float(forward["records"][-1]["E"])
    backward_endpoint = float(backward["records"][-1]["E"])
    # Keep trajectory orientation deterministic.  Energy-based branch swapping
    # would reverse the entire path when nearly degenerate endpoint energies
    # exchange order between otherwise comparable runs.
    first_direction = "forward"
    ordered_records = list(reversed(forward["records"]))
    ordered_records.append(transition_state)
    ordered_records.extend(backward["records"])
    ts_index = len(forward["records"]) + 1
    reference_energy = min(forward_endpoint, backward_endpoint)

    rows = []
    for index, record in enumerate(ordered_records, start=1):
        delta_kcal = (float(record["E"]) - reference_energy) * KCAL_PER_HARTREE
        line = (
            f"{index:4d}  {float(record['E']):14.6f}  {delta_kcal:12.6f}    "
            f"{float(record['maxG']):8.6f}  {float(record['rmsG']):8.6f}"
        )
        if index == ts_index:
            line += " <= TS"
        rows.append(line + "\n")

    converged = bool(forward["status"]["converged"] and backward["status"]["converged"])
    return {
        "method": str(method_label),
        "E_ref": reference_energy,
        "ts_index": ts_index,
        "first_direction": first_direction,
        "records": ordered_records,
        "transition_state_record": transition_state,
        "merged_rows": rows,
        "forward_status": forward["status"],
        "backward_status": backward["status"],
        "converged": converged,
        "endpoint_force_convergence_admitted": converged,
        "transition_state_is_strict_path_maximum": highest_path_energy < ts_energy,
        "transition_state_is_path_maximum_within_tolerance": True,
        "maximum_path_energy_excess_hartree": highest_path_energy - ts_energy,
        "path_energy_tolerance_hartree": path_energy_tolerance,
        "initial_displacement_tolerance_angstrom": displacement_tolerance,
        "endpoint_minima_verified": False,
        "chemical_connectivity_verified": False,
        "step_size_convergence_verified": False,
    }


def render_irc_path_summary(summary: Mapping[str, object]) -> list[str]:
    """Render the common path and termination ledger."""

    method = str(summary["method"])
    rendered = [
        "\n---------------------------------------------------------------\n",
        f"{(method + '-IRC PATH SUMMARY').center(63)}\n",
        "---------------------------------------------------------------\n",
        "All energies are in Eh and forces are in Eh/Angstrom.\n\n",
        "Step        E(Eh)      dE(kcal/mol)  max(|F|)   RMS(F) \n",
    ]
    rendered.extend(summary["merged_rows"])
    rendered.append(
        "\nPath-energy admission: "
        f"tolerance={float(summary['path_energy_tolerance_hartree']):.6g} Eh; "
        "maximum displaced-image excess="
        f"{float(summary['maximum_path_energy_excess_hartree']):.6g} Eh; "
        "strict TS maximum="
        f"{bool(summary['transition_state_is_strict_path_maximum'])}; "
        "TS maximum within tolerance="
        f"{bool(summary['transition_state_is_path_maximum_within_tolerance'])}.\n"
    )
    for direction in _DIRECTIONS:
        status = summary[f"{direction}_status"]
        label = "CONVERGED" if status["converged"] else "NOT CONVERGED"
        rendered.append(
            f"\n{direction.capitalize()} endpoint: {label}; "
            f"reason={status['termination_reason']}; "
            f"max|F|={status['final_maximum_force_hartree_per_A']:.6g}; "
            f"RMS(F)={status['final_rms_force_hartree_per_A']:.6g}.\n"
        )
    rendered.append(
        "Endpoint minimum identity, chemical connectivity, and step-size "
        "convergence are not certified by this run.\n"
    )
    return rendered


def enforce_irc_path_admission(
    summary: Mapping[str, object],
    *,
    require_converged_endpoints: object,
) -> None:
    """Fail closed when either requested endpoint misses the force criteria."""

    required = coerce_strict_bool(
        require_converged_endpoints,
        "require_converged_endpoints",
    )
    if not required or summary.get("converged") is True:
        return
    failures = []
    for direction in _DIRECTIONS:
        status = summary[f"{direction}_status"]
        if not status["converged"]:
            failures.append(
                f"{direction} endpoint did not converge "
                f"({status['termination_reason']}; "
                f"max|F|={status['final_maximum_force_hartree_per_A']:.6g}, "
                f"RMS(F)={status['final_rms_force_hartree_per_A']:.6g} Eh/Angstrom)"
            )
    raise RuntimeError("IRC path admission failed: " + "; ".join(failures) + ".")


def _records_to_atoms(
    template_atoms: Atoms,
    records: Sequence[Mapping[str, object]],
) -> tuple[list[Atoms], list[float]]:
    frames = []
    energies = []
    for record in records:
        frame = template_atoms.copy()
        frame.set_positions(np.asarray(record["x"], dtype=float))
        frames.append(frame)
        energies.append(float(record["E"]))
    return frames, energies


def write_irc_trajectories(
    *,
    template_atoms: Atoms,
    output: str,
    forward: Mapping[str, object],
    backward: Mapping[str, object],
    summary: Mapping[str, object],
) -> dict[str, str]:
    """Write unit-labelled branch and full trajectories in scientific order."""

    base, _ = os.path.splitext(output)
    paths = {
        "forward": base + "_forward.xyz",
        "backward": base + "_backward.xyz",
        "full": base + "_full.xyz",
    }
    transition_state = summary["transition_state_record"]
    record_sets = {
        "forward": [transition_state, *forward["records"]],
        "backward": [transition_state, *backward["records"]],
        "full": summary["records"],
    }
    for name, records in record_sets.items():
        frames, energies = _records_to_atoms(template_atoms, records)
        write_xyz(
            paths[name],
            frames,
            energies=energies,
            energy_key="energy_hartree",
            start_index=0,
        )
    return paths


__all__ = [
    "DEFAULT_IRC_MAX_FORCE_HARTREE_PER_ANGSTROM",
    "DEFAULT_IRC_MAX_STEPS",
    "DEFAULT_IRC_RMS_FORCE_HARTREE_PER_ANGSTROM",
    "IRCPathParams",
    "PATH_ENERGY_ABSOLUTE_TOLERANCE_HARTREE",
    "assemble_irc_path",
    "enforce_irc_path_admission",
    "finalize_irc_branch",
    "irc_force_criteria_satisfied",
    "make_irc_record",
    "render_irc_path_summary",
    "validate_irc_path_params",
    "write_irc_trajectories",
]
