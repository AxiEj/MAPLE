"""Exact admission boundary for AIMNet2025 plus Torch OBC-II paths."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from ase import Atoms

from .method_registry import (
    normalize_public_method,
    resolve_method_capability,
    required_capabilities,
    validate_method_parameters,
)


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
    from .numerical_state import PublishedCalculatorCacheSnapshot

    saved = [PublishedCalculatorCacheSnapshot.capture(item) for item in calculators]
    try:
        yield
    finally:
        for snapshot in saved:
            snapshot.restore()


def bind_admitted_implicit_path(
    images,
    method: str,
    *,
    require_hessian: bool = False,
    normalized_params=None,
) -> bool:
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
    task = "irc" if method in {"gs", "hpc", "lqa", "eulerpc"} else "ts"
    public_key = normalize_public_method(task, method)
    if normalized_params is None:
        normalized_params = validate_method_parameters(public_key, {})
    requested_capabilities = required_capabilities(public_key, normalized_params)
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
    for capability in requested_capabilities:
        resolve_method_capability(capability, calculator)
    correction.bind_compatible_geometries(images)
    return True
