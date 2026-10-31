"""Exact admission boundary for AIMNet2025 plus Torch OBC-II paths."""

from __future__ import annotations

import math
from dataclasses import fields
from numbers import Real

import numpy as np
from ase import Atoms


_ALIASES = {
    "gs": {"sd_len_bohr": "step_length_bohr", "steplength_bohr": "step_length_bohr",
           "max_points": "max_steps", "tol_maxf": "f_max_th", "tol_rmsf": "f_rms_th"},
    "neb": {},
    "string": {},
}


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
    source = params if isinstance(params, dict) else {}
    lowered = {str(key).lower(): value for key, value in source.items()}
    nested = lowered.get(method)
    selected = nested if isinstance(nested, dict) else lowered
    allowed = {field.name: field for field in fields(params_class)}
    aliases = _ALIASES[method]
    normalized = {}
    for raw_key, value in selected.items():
        key = aliases.get(str(raw_key).lower(), str(raw_key).lower())
        if key not in allowed:
            raise ValueError(f"Unknown {method} parameter: {raw_key}")
        normalized[key] = value

    defaults = params_class()
    for key, value in normalized.items():
        default = getattr(defaults, key)
        if isinstance(default, bool):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{method}.{key} must be boolean")
        elif isinstance(default, int):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{method}.{key} must be an integer")
        elif isinstance(default, float):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{method}.{key} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{method}.{key} must be finite")
        elif default is None and value is not None and not isinstance(value, (str, int)):
            raise TypeError(f"{method}.{key} has an invalid type")

    nonnegative = {
        "max_steps", "max_micro_cycles", "max_iter", "max_iter_grow",
        "grow_relax_iters", "hessian_recalc",
    }
    positive = {
        "target_mode", "step_length_bohr", "micro_step_thresh", "f_max_th",
        "f_rms_th", "n_images", "k_min", "k_max", "step0", "grow_step",
        "grow_step_min", "grow_relax_tol", "merge_threshold",
    }
    for key in nonnegative.intersection(normalized):
        if normalized[key] is not None and normalized[key] < 0:
            raise ValueError(f"{method}.{key} must be nonnegative")
    for key in positive.intersection(normalized):
        if normalized[key] <= 0:
            raise ValueError(f"{method}.{key} must be positive")
    if method == "gs" and normalized.get("hessian_update", "bofill") not in {"bofill", "bfgs"}:
        raise ValueError("gs.hessian_update must be bofill or bfgs")
    if method in {"neb", "string"} and normalized.get("refine") is not None:
        raise ValueError(f"Implicit {method} is path-only and requires refine=None")
    return normalized


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
    if require_hessian and str(getattr(calculator, "hessian", "")).lower() != "analytic":
        raise ValueError("Implicit GS-IRC requires an analytic composed Hessian.")
    correction.bind_compatible_geometries(images)
    return True
