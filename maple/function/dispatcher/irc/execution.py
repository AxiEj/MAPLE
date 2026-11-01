from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import wraps
import os

import numpy as np

from ..curvature import (
    build_mass_weighted_mode_basis,
    resolve_hessian_precision_profile,
    validate_cartesian_hessian,
)
from ..numerical_state import (
    PublishedCalculatorCacheSnapshot,
    ScientificAtomsSnapshot,
    evaluate_energy,
    evaluate_forces,
    evaluate_hessian,
    to_numpy_f64,
)


MATERIAL_NEGATIVE_MODE_CM1 = -10.0
COMMON_PARAMETER_ALIASES = {
    "sd_len_bohr": "step_length_bohr",
    "steplength_bohr": "step_length_bohr",
    "max_points": "max_steps",
    "tol_maxf": "f_max_th",
    "tol_rmsf": "f_rms_th",
}
IRC_PARAMETER_RULES = {
    "target_mode": "positive_integer",
    "step_length_bohr": "positive_finite",
    "max_steps": "nonnegative_integer",
    "max_micro_cycles": "nonnegative_integer",
    "micro_step_thresh": "positive_finite",
    "euler_n": "positive_integer",
    "max_pred_steps": "positive_integer",
    "loose_cycles": "nonnegative_integer",
    "hessian_recalc": "optional_nonnegative_integer",
    "hessian_update": "hessian_update",
    "dwi_n": "positive_even_integer",
    "mbs_max_k": "at_least_two_integer",
    "mbs_points": "at_least_two_integer",
    "mbs_tol": "positive_finite",
    "f_max_th": "positive_finite",
    "f_rms_th": "positive_finite",
    "print_each": "boolean",
    "write_traj": "boolean",
    "project_rigid_modes": "boolean",
}


@dataclass(frozen=True)
class IRCStart:
    positions: np.ndarray
    energy: float
    forces: np.ndarray
    hessian: np.ndarray
    inverse_sqrt_mass: np.ndarray
    negative_mode: np.ndarray
    eigenvalue: float
    record: dict


def irc_parameter_spec(method: str, params_type, aliases: dict | None = None) -> dict:
    canonical = tuple(field.name for field in fields(params_type))
    combined_aliases = dict(COMMON_PARAMETER_ALIASES)
    if aliases:
        combined_aliases.update(aliases)
    return {
        "method": method.lower(),
        "parameters": canonical,
        "aliases": {
            name: target
            for name, target in combined_aliases.items()
            if target in canonical
        },
        "rules": {
            name: IRC_PARAMETER_RULES[name]
            for name in canonical
            if name in IRC_PARAMETER_RULES
        },
    }


def coerce_irc_parameters(params, paras, *, method: str, aliases: dict | None = None):
    if paras is None:
        _validate_parameters(params, method)
        return params
    if not isinstance(paras, dict):
        raise TypeError(f"{method.upper()}-IRC parameters must be a dictionary.")
    lowered = _lower_unique_parameters(paras, method)
    nested = [key for key in (method.lower(), "irc") if key in lowered]
    if len(nested) > 1:
        raise ValueError(f"Conflicting {method.upper()}-IRC parameter sections.")
    selected = lowered[nested[0]] if nested else lowered
    if not isinstance(selected, dict):
        raise TypeError(f"{method.upper()}-IRC parameter section must be a dictionary.")
    selected = _lower_unique_parameters(selected, method)
    spec = irc_parameter_spec(method, type(params), aliases)
    allowed = set(spec["parameters"])
    mapped = {}
    for name, value in selected.items():
        canonical = spec["aliases"].get(name, name)
        if canonical not in allowed:
            raise ValueError(f"Unknown {method.upper()}-IRC parameter: {name}")
        if canonical in mapped:
            raise ValueError(f"Conflicting {method.upper()}-IRC parameter: {canonical}")
        mapped[canonical] = value
    for name, value in mapped.items():
        setattr(params, name, value)
    _validate_parameters(params, method)
    return params


def _lower_unique_parameters(values: dict, method: str) -> dict:
    lowered = {}
    for raw_name, value in values.items():
        name = str(raw_name).lower()
        if name in lowered:
            raise ValueError(f"Conflicting {method.upper()}-IRC parameter: {name}")
        lowered[name] = value
    return lowered


def _validate_parameters(params, method: str) -> None:
    for field in fields(type(params)):
        name = field.name
        rule = IRC_PARAMETER_RULES.get(name)
        if rule is None:
            continue
        value = getattr(params, name)
        if rule == "nonnegative_integer":
            valid = type(value) is int and value >= 0
        elif rule == "positive_integer":
            valid = type(value) is int and value >= 1
        elif rule == "at_least_two_integer":
            valid = type(value) is int and value >= 2
        elif rule == "positive_even_integer":
            valid = type(value) is int and value >= 1 and value % 2 == 0
        elif rule == "optional_nonnegative_integer":
            valid = value is None or (type(value) is int and value >= 0)
        elif rule == "positive_finite":
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                valid = False
            else:
                valid = (
                    not isinstance(value, (bool, np.bool_))
                    and np.isscalar(value)
                    and np.isfinite(numeric)
                    and numeric > 0.0
                )
            if valid:
                setattr(params, name, numeric)
        elif rule == "boolean":
            valid = type(value) is bool
        elif rule == "hessian_update":
            normalized = str(value).lower()
            valid = normalized in {"bfgs", "bofill"}
            if valid:
                setattr(params, name, normalized)
        else:
            raise RuntimeError(f"Unknown IRC parameter rule: {rule}")
        if not valid:
            raise ValueError(
                f"{method.upper()}-IRC {name} violates the {rule} rule."
            )


@contextmanager
def restore_scientific_state_on_error(atoms):
    snapshot = ScientificAtomsSnapshot.capture(atoms)
    calculator_snapshot = (
        PublishedCalculatorCacheSnapshot.capture(atoms.calc)
        if atoms.calc is not None
        else None
    )
    try:
        yield
    except BaseException as exc:
        restore_errors = []
        try:
            snapshot.restore(atoms)
        except Exception as restore_error:
            restore_errors.append(f"scientific Atoms: {restore_error!r}")
        if calculator_snapshot is not None:
            try:
                calculator_snapshot.restore()
            except Exception as restore_error:
                restore_errors.append(f"calculator cache: {restore_error!r}")
        if restore_errors:
            exc.add_note("Restoration also failed for " + "; ".join(restore_errors))
        raise


def restores_scientific_state_on_error(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with restore_scientific_state_on_error(self.atoms):
            return function(self, *args, **kwargs)

    return wrapped


def prepare_irc_start(
    job,
    *,
    method: str,
    target_mode: int,
    project_rigid_modes: bool = True,
) -> IRCStart:
    atoms = job.atoms
    positions = to_numpy_f64(atoms.get_positions())
    expected_positions = (len(atoms), 3)
    if positions.shape != expected_positions:
        raise ValueError(f"{method.upper()}-IRC requires finite Cartesian coordinates.")
    energy = evaluate_energy(atoms)
    forces = to_numpy_f64(evaluate_forces(atoms)).reshape(-1)
    if forces.shape != (3 * len(atoms),):
        raise ValueError(f"{method.upper()}-IRC requires one finite 3N force vector.")
    raw_hessian = evaluate_hessian(atoms)
    job.log_numerical_hessian_diagnostics(atoms.calc)
    profile = resolve_hessian_precision_profile(atoms, purpose=f"{method}-irc")
    validated = validate_cartesian_hessian(atoms, raw_hessian, profile)
    modes = build_mass_weighted_mode_basis(
        atoms,
        validated,
        project_rigid_modes=project_rigid_modes,
    )
    material = np.flatnonzero(modes.frequencies_cm1 < MATERIAL_NEGATIVE_MODE_CM1)
    if material.size == 0:
        raise RuntimeError(f"{method.upper()}-IRC: no negative internal mode at TS.")
    ordered = material[np.argsort(modes.eigenvalues[material])]
    if target_mode > ordered.size:
        raise RuntimeError(
            f"{method.upper()}-IRC: requested negative mode does not exist."
        )
    index = int(ordered[target_mode - 1])
    record = {
        "E": float(energy),
        "maxG": float(np.max(np.abs(forces))),
        "rmsG": float(np.sqrt(np.mean(forces**2))),
        "x": positions.copy(),
    }
    return IRCStart(
        positions=positions.copy(),
        energy=float(energy),
        forces=forces.copy(),
        hessian=validated.physical_hessian.copy(),
        inverse_sqrt_mass=modes.inverse_sqrt_mass.copy(),
        negative_mode=modes.eigenvectors[:, index].copy(),
        eigenvalue=float(modes.eigenvalues[index]),
        record=record,
    )


def evaluate_irc_state(atoms, positions: np.ndarray) -> tuple[float, np.ndarray]:
    positions = to_numpy_f64(positions)
    if positions.shape != (len(atoms), 3):
        raise ValueError("IRC evaluation requires finite Cartesian positions with shape (N, 3).")
    atoms.set_positions(positions)
    energy = evaluate_energy(atoms)
    forces = to_numpy_f64(evaluate_forces(atoms)).reshape(-1)
    if forces.shape != (3 * len(atoms),):
        raise ValueError("IRC evaluation requires one finite 3N force vector.")
    return float(energy), forces


def evaluate_irc_hessian(job, *, method: str) -> np.ndarray:
    raw_hessian = evaluate_hessian(job.atoms)
    job.log_numerical_hessian_diagnostics(job.atoms.calc)
    profile = resolve_hessian_precision_profile(
        job.atoms,
        purpose=f"{method}-irc",
    )
    return validate_cartesian_hessian(
        job.atoms,
        raw_hessian,
        profile,
    ).physical_hessian


def empty_side(method: str, forward: bool, energy: float) -> dict:
    direction = "FORWARD" if forward else "BACKWARD"
    return {
        "title": f"{direction} {method.upper()}-IRC",
        "records": [],
        "E_ts": energy,
        "iterations": 0,
        "termination_reason": "maximum_steps",
        "converged": False,
    }


def assemble_irc_result(forward: dict, backward: dict, input_ts: dict) -> dict:
    ordered_path = list(reversed(backward["records"])) + [input_ts] + forward["records"]
    converged = bool(forward["converged"] and backward["converged"])
    endpoints = [
        side["records"][-1]["E"]
        for side in (forward, backward)
        if side["records"]
    ]
    reference = float(min(endpoints)) if endpoints else float(input_ts["E"])
    ts_index = len(backward["records"])
    energies = [float(record["E"]) for record in ordered_path]
    relative_energies = [
        (energy - reference) * 627.509474 for energy in energies
    ]
    rows = []
    for index, (record, energy, relative) in enumerate(
        zip(ordered_path, energies, relative_energies),
        start=1,
    ):
        line = (
            f"{index:4d}  {energy:14.6f}  {relative:12.6f}    "
            f"{record['maxG']:8.6f}  {record['rmsG']:8.6f}"
        )
        if index - 1 == ts_index:
            line += " <= TS"
        rows.append(line + "\n")
    summary = {
        "E_ref": reference,
        "ts_index": ts_index,
        "merged_rows": rows,
        "images": [record["x"].copy() for record in ordered_path],
        "energies": energies,
        "relative_energies_kcal_mol": relative_energies,
        "status": "converged" if converged else "bounded_nonconverged",
    }
    return {
        "status": summary["status"],
        "converged": converged,
        "termination_reason": (
            "force_converged"
            if converged
            else f"forward:{forward['termination_reason']};backward:{backward['termination_reason']}"
        ),
        "iterations": forward["iterations"] + backward["iterations"],
        "forward": forward,
        "backward": backward,
        "input_ts": input_ts,
        "ts_index": ts_index,
        "ordered_path": ordered_path,
        "summary": summary,
    }


def write_irc_trajectories(atoms, output, forward, backward, input_ts, writer) -> None:
    base, _ = os.path.splitext(output)

    def materialize(records):
        images = []
        energies = []
        for record in records:
            image = atoms.copy()
            image.set_positions(record["x"])
            images.append(image)
            energies.append(record["E"])
        return images, energies

    forward_images, forward_energies = materialize(forward["records"])
    backward_images, backward_energies = materialize(backward["records"])
    full_records = list(reversed(backward["records"])) + [input_ts] + forward["records"]
    full_images, full_energies = materialize(full_records)
    writer(base + "_forward.xyz", forward_images, forward_energies)
    writer(base + "_backward.xyz", backward_images, backward_energies)
    writer(base + "_full.xyz", full_images, full_energies)
