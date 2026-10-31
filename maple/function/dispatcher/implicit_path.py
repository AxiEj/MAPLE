"""Exact admission boundary for AIMNet2025 plus Torch OBC-II paths."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, fields
from numbers import Real
from types import MappingProxyType

import numpy as np
from ase import Atoms


_ALIASES = {
    "gs": {"sd_len_bohr": "step_length_bohr", "steplength_bohr": "step_length_bohr",
           "max_points": "max_steps", "tol_maxf": "f_max_th", "tol_rmsf": "f_rms_th"},
    "neb": {},
    "string": {},
}


@dataclass(frozen=True)
class ParameterRule:
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    choices: frozenset | None = None
    inactive_reason: str | None = None


def _rule(kind: str, **kwargs) -> ParameterRule:
    return ParameterRule(kind=kind, **kwargs)


def _inactive(reason: str) -> ParameterRule:
    return ParameterRule(kind="inactive", inactive_reason=reason)


PATH_PARAMETER_SCHEMAS = MappingProxyType({
    "gs": MappingProxyType({
        "target_mode": _rule("int", minimum=1),
        "step_length_bohr": _rule("real", minimum=0.0, minimum_inclusive=False),
        "max_steps": _rule("int", minimum=0),
        "max_micro_cycles": _rule("int", minimum=0),
        "micro_step_thresh": _rule("real", minimum=0.0, minimum_inclusive=False),
        "hessian_recalc": _rule("optional_int", minimum=0),
        "hessian_update": _rule("enum", choices=frozenset({"bfgs", "bofill"})),
        "f_max_th": _rule("real", minimum=0.0, minimum_inclusive=False),
        "f_rms_th": _rule("real", minimum=0.0, minimum_inclusive=False),
        "print_each": _rule("bool"),
        "write_traj": _rule("bool"),
        "project_rigid_modes": _rule("bool"),
    }),
    "neb": MappingProxyType({
        "n_images": _rule("int", minimum=1),
        "k_min": _rule("real", minimum=0.0, minimum_inclusive=False),
        "k_max": _rule("real", minimum=0.0, minimum_inclusive=False),
        "use_dynamic_k": _rule("bool"),
        "k_decay": _rule("real", minimum=0.0, minimum_inclusive=False),
        "max_iter": _rule("int", minimum=0),
        "lbfgs_m": _rule("int", minimum=1),
        "step0": _rule("real", minimum=0.0, minimum_inclusive=False),
        "ifidpp": _rule("int", choices=frozenset({0, 1})),
        "neb_f_max_th": _rule("real", minimum=0.0, minimum_inclusive=False),
        "neb_f_rms_th": _rule("real", minimum=0.0, minimum_inclusive=False),
        "initial_opt": _rule("bool"),
        "refine": _rule("none"),
        "cineb_f_max_th": _inactive("CINEB refinement is closed for this path-only opening"),
        "cineb_f_rms_th": _inactive("CINEB refinement is closed for this path-only opening"),
        "cilbfgs_m": _inactive("CINEB refinement is closed for this path-only opening"),
        "cistep0": _inactive("CINEB refinement is closed for this path-only opening"),
    }),
    "string": MappingProxyType({
        "grow_step": _rule("real", minimum=0.0, minimum_inclusive=False),
        "grow_step_min": _rule("real", minimum=0.0, minimum_inclusive=False),
        "grow_backtrack": _rule(
            "real", minimum=0.0, maximum=1.0,
            minimum_inclusive=False, maximum_inclusive=False,
        ),
        "grow_relax_iters": _rule("int", minimum=0),
        "grow_relax_tol": _rule("real", minimum=0.0, minimum_inclusive=False),
        "grow_energy_tol": _rule("real", minimum=0.0, minimum_inclusive=False),
        "pause_imbalance": _rule("int", minimum=0),
        "refine_end_every": _rule("int", minimum=0),
        "merge_threshold": _rule("real", minimum=0.0, minimum_inclusive=False),
        "n_images": _rule("int", minimum=2),
        "max_iter_relax": _inactive("full String relaxation is not run when refine=None"),
        "string_f_max_th": _inactive("full String relaxation is not run when refine=None"),
        "string_f_rms_th": _inactive("full String relaxation is not run when refine=None"),
        "lbfgs_memory": _inactive("full String relaxation is not run when refine=None"),
        "lbfgs_curvature": _inactive("full String relaxation is not run when refine=None"),
        "lbfgs_max_step": _inactive("full String relaxation is not run when refine=None"),
        "reparam_every": _inactive("full String relaxation is not run when refine=None"),
        "refine": _rule("none"),
        "cistring_f_max_th": _inactive("CI-String refinement is closed for this path-only opening"),
        "cistring_f_rms_th": _inactive("CI-String refinement is closed for this path-only opening"),
        "max_iter_grow": _rule("int", minimum=0),
        "log_every": _rule("int", minimum=1),
        "guide_conn_min": _rule("real", minimum=0.0, maximum=1.0),
        "guide_d_switch": _rule("real", minimum=0.0, minimum_inclusive=False),
        "guide_inc_trigger": _rule("int", minimum=0),
    }),
})


def path_images(value) -> list[Atoms]:
    """Normalize Atoms, Molecules, or a sequence without copying calculators."""
    if isinstance(value, Atoms):
        return [value]
    structures = getattr(value, "multiatoms", value)
    if not isinstance(structures, (list, tuple)) or not structures:
        raise TypeError("A path requires Atoms, Molecules, or a nonempty image sequence.")
    if not all(isinstance(image, Atoms) for image in structures):
        raise TypeError("Every path image must be an ASE Atoms object.")
    return list(structures)


def validate_path_parameters(method: str, params: dict, params_class) -> dict:
    """Validate one method's public parameter mapping before model work."""
    method = method.lower()
    if method not in PATH_PARAMETER_SCHEMAS:
        raise ValueError(f"Unknown implicit path method: {method}")
    source = params if isinstance(params, dict) else {}
    lowered = {str(key).lower(): value for key, value in source.items()}
    nested = lowered.get(method)
    selected = nested if isinstance(nested, dict) else lowered
    declared = {field.name for field in fields(params_class)}
    schema = PATH_PARAMETER_SCHEMAS[method]
    if set(schema) != declared:
        raise RuntimeError(f"{method} parameter schema does not match {params_class.__name__}")
    aliases = _ALIASES[method]
    normalized = {}
    for raw_key, value in selected.items():
        key = aliases.get(str(raw_key).lower(), str(raw_key).lower())
        if key not in schema:
            raise ValueError(f"Unknown {method} parameter: {raw_key}")
        normalized[key] = value

    defaults = params_class()
    for key, value in normalized.items():
        rule = schema[key]
        if rule.inactive_reason is not None:
            raise ValueError(
                f"{method}.{key} is inactive for this path: {rule.inactive_reason}"
            )
        if rule.kind == "bool":
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{method}.{key} must be boolean")
        elif rule.kind in {"int", "optional_int"}:
            if rule.kind == "optional_int" and value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{method}.{key} must be an integer")
        elif rule.kind == "real":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{method}.{key} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{method}.{key} must be finite")
        elif rule.kind == "enum":
            if not isinstance(value, str):
                raise TypeError(f"{method}.{key} must be a string")
        elif rule.kind == "none":
            if value is not None:
                raise ValueError(f"Implicit {method} is path-only and requires {key}=None")
            continue
        else:
            raise RuntimeError(f"Unknown parameter rule kind: {rule.kind}")

        if rule.choices is not None and value not in rule.choices:
            choices = ", ".join(str(choice) for choice in sorted(rule.choices, key=str))
            if rule.choices == frozenset({0, 1}):
                raise ValueError(f"{method}.{key} must be 0 or 1")
            raise ValueError(f"{method}.{key} must be one of: {choices}")
        if rule.minimum is not None:
            below = value < rule.minimum if rule.minimum_inclusive else value <= rule.minimum
            if below:
                if rule.minimum == 0 and rule.minimum_inclusive:
                    raise ValueError(f"{method}.{key} must be nonnegative")
                if (
                    (rule.minimum == 0 and not rule.minimum_inclusive)
                    or (rule.minimum == 1 and rule.minimum_inclusive)
                ):
                    raise ValueError(f"{method}.{key} must be positive")
                relation = "at least" if rule.minimum_inclusive else "greater than"
                raise ValueError(f"{method}.{key} must be {relation} {rule.minimum}")
        if rule.maximum is not None:
            above = value > rule.maximum if rule.maximum_inclusive else value >= rule.maximum
            if above:
                relation = "at most" if rule.maximum_inclusive else "less than"
                raise ValueError(f"{method}.{key} must be {relation} {rule.maximum}")

    if (
        method == "neb"
        and normalized.get("k_min", defaults.k_min)
        > normalized.get("k_max", defaults.k_max)
    ):
        raise ValueError("neb.k_min must not exceed neb.k_max")
    if (
        method == "string"
        and normalized.get("grow_step_min", defaults.grow_step_min)
        > normalized.get("grow_step", defaults.grow_step)
    ):
        raise ValueError("string.grow_step_min must not exceed string.grow_step")
    return normalized


def uses_admitted_implicit_path(images) -> bool:
    """Return whether images carry one path-bound implicit calculator."""
    images = path_images(images)
    calculator = images[0].calc
    if calculator is None or any(image.calc is not calculator for image in images):
        return False
    correction = getattr(calculator, "solvent_correction", None)
    return (
        correction is not None
        and getattr(correction, "_path_frozen_masses", None) is not None
    )


def evaluate_path_images(images, *, label: str):
    """Evaluate implicit paths in one batch and gas paths per ASE image."""
    images = path_images(images)
    if uses_admitted_implicit_path(images):
        result = images[0].calc.calculate_many(images, properties=("energy", "forces"))
        if result.energies is None or result.forces is None:
            raise RuntimeError(f"{label} batch evaluation did not return energy and forces.")
        return list(result.energies), list(result.forces)
    energies = [float(image.get_potential_energy(force_consistent=True)) for image in images]
    forces = [np.asarray(image.get_forces(), dtype=np.float64) for image in images]
    return energies, forces


_MISSING = object()


@contextmanager
def preserve_path_calculator_caches(images, *, enabled: bool):
    """Restore published calculator cache fields for admitted implicit paths."""
    if not enabled:
        yield
        return
    calculators = []
    for image in path_images(images):
        if image.calc is not None and all(image.calc is not item for item in calculators):
            calculators.append(image.calc)
    names = ("atoms", "results", "solvation_result", "_evaluated_correction")
    saved = {
        calculator: {name: getattr(calculator, name, _MISSING) for name in names}
        for calculator in calculators
    }
    try:
        yield
    finally:
        for calculator, state in saved.items():
            for name, value in state.items():
                if value is _MISSING:
                    if hasattr(calculator, name):
                        delattr(calculator, name)
                else:
                    setattr(calculator, name, value)


def bind_admitted_implicit_path(images, method: str, *, require_hessian: bool = False) -> bool:
    """Bind one exact AIMNet2025/Torch-OBC-II composition to all images."""
    images = path_images(images)
    calculators = [image.calc for image in images]
    if any(calc is None for calc in calculators):
        if any(getattr(calc, "solvent_correction", None) is not None for calc in calculators if calc):
            raise ValueError("Every implicit path image must carry the composed calculator.")
        return False
    if not any(getattr(calc, "solvent_correction", None) is not None for calc in calculators):
        return False
    if any(calc is not calculators[0] for calc in calculators[1:]):
        raise ValueError("All implicit path images must share one exact composed calculator instance.")
    calculator = calculators[0]
    correction = getattr(calculator, "solvent_correction", None)
    if correction is None:
        raise ValueError("Implicit path calculator is missing its correction.")
    from maple.function.calculator.aimnet._aimnet2025_calculator import (
        AIMNet2025Calculator,
    )
    if type(calculator) is not AIMNet2025Calculator:
        raise ValueError("Implicit paths admit only AIMNet2025 plus independent Torch OBC-II.")
    options = getattr(correction, "solvation_options", {})
    if (
        options.get("provider") != "torch"
        or options.get("model") != "obc2"
        or options.get("profile") != "obc2-mbondi2"
        or options.get("nonpolar") not in {"ace", "none"}
        or getattr(correction, "mode", None) != "fixed"
        or getattr(correction, "inner_mode", None) is not None
    ):
        raise ValueError("Implicit paths require fixed, single-solute Torch OBC-II ACE/NONE.")
    if (
        getattr(calculator, "analytic_implicit_derivatives_admitted", False) is not True
        or getattr(correction, "analytic_task_derivatives_admitted", False) is not True
    ):
        raise ValueError("Implicit paths require the qualified AIMNet2025/Torch-OBC-II cell.")
    if str(getattr(calculator, "hessian", "")).lower() != "analytic":
        raise ValueError(f"Implicit {method} requires an analytic enrolled model configuration.")
    correction.bind_compatible_geometries(images)
    return True
