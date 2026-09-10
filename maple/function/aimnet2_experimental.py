"""Public input contract for the opt-in AIMNet2 smooth-ddPCM workflows.

This module intentionally contains no calculator or model imports.  It is the
single lightweight validation seam shared by command parsing, coordinate
reading, and calculator construction; formal solvation capability registries
remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import math
from typing import Any

from maple.solvation.api import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_PROFILE_V1,
)

AIMNET2_EXPERIMENTAL_PROVIDER = "aimnet2-smooth-ddpcm"
AIMNET2_EXPERIMENTAL_PROFILE = CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_PROFILE_V1
AIMNET2_EXPERIMENTAL_WORKFLOW_ID = "aimnet2-smooth-ddpcm-experimental-workflows-v2"

_ALLOWED_MODEL_OPTIONS = frozenset({"model_path", "hessian"})
_ALLOWED_SOLVATION_OPTIONS = frozenset(
    {"method", "implicit", "provider", "profile", "experimental"}
)
_TASK_METHODS = {
    "sp": None,
    "opt": "lbfgs",
    "freq": "mw",
    "ts": "prfo",
}
_SUPPORTED_ELEMENTS = frozenset({"H", "C", "N", "O"})


def _normalized(value: object) -> str:
    return str(value).strip().lower()


def is_aimnet2_experimental_request(solv: object) -> bool:
    """Recognize either half of the exact selector so mismatches fail closed."""

    if not isinstance(solv, Mapping):
        return False
    return (
        _normalized(solv.get("provider", "")) == AIMNET2_EXPERIMENTAL_PROVIDER
        or _normalized(solv.get("profile", "")) == AIMNET2_EXPERIMENTAL_PROFILE
    )


def validate_aimnet2_experimental_settings(
    *,
    model: object,
    model_options: Mapping[str, object] | None,
    solvation_options: Mapping[str, object],
    task: str | None = None,
    task_params: Mapping[str, object] | None = None,
    device: object = None,
    d4: bool = False,
    charge_options: Mapping[str, object] | None = None,
    pbc: object = None,
) -> None:
    """Validate and canonicalize the exact public experimental selector."""

    if not is_aimnet2_experimental_request(solvation_options):
        raise ValueError("Not an AIMNet2 smooth-ddPCM experimental request.")

    provider = _normalized(solvation_options.get("provider", ""))
    profile = _normalized(solvation_options.get("profile", ""))
    if (
        provider != AIMNET2_EXPERIMENTAL_PROVIDER
        or profile != AIMNET2_EXPERIMENTAL_PROFILE
    ):
        raise ValueError(
            "AIMNet2 experimental selector mismatch: provider and profile must "
            "both use the exact smooth-ddPCM experimental values."
        )
    if _normalized(solvation_options.get("method", "")) != "smd":
        raise ValueError(
            "AIMNet2 smooth-ddPCM experimental workflows require method=smd."
        )
    implicit = _normalized(solvation_options.get("implicit", ""))
    if implicit in {"", "none", "null", "false"}:
        raise ValueError(
            "AIMNet2 smooth-ddPCM experimental workflows require implicit=<solvent>."
        )
    if solvation_options.get("experimental") is not True:
        raise ValueError(
            "AIMNet2 smooth-ddPCM is uncertified; set experimental=true explicitly."
        )
    unknown_solvation = sorted(
        set(solvation_options).difference(_ALLOWED_SOLVATION_OPTIONS)
    )
    if unknown_solvation:
        raise ValueError(
            "Unsupported AIMNet2 experimental solvation option(s): "
            + ", ".join(unknown_solvation)
            + "."
        )

    if _normalized(model) != "aimnet2":
        raise ValueError(
            "AIMNet2 smooth-ddPCM experimental workflows require model=aimnet2."
        )
    options = model_options if model_options is not None else {}
    unknown_model = sorted(set(options).difference(_ALLOWED_MODEL_OPTIONS))
    if unknown_model:
        raise ValueError(
            "Unsupported AIMNet2 experimental model option(s): "
            + ", ".join(unknown_model)
            + "."
        )
    model_path = options.get("model_path")
    if model_path is None or not str(model_path).strip():
        raise ValueError(
            "AIMNet2 smooth-ddPCM requires an explicit model_path to the original "
            "hash-bound checkpoint."
        )
    hessian = options.get("hessian")
    if hessian is not None and _normalized(hessian) != "numerical":
        raise ValueError("AIMNet2 smooth-ddPCM accepts only hessian=numerical.")
    if isinstance(options, MutableMapping):
        options.setdefault("hessian", "numerical")

    effective_device = "cpu" if device is None else _normalized(device)
    if effective_device != "cpu":
        raise ValueError("AIMNet2 smooth-ddPCM experimental workflows are CPU-only.")
    if d4 is not False:
        raise ValueError(
            "AIMNet2 smooth-ddPCM experimental workflows do not compose D4."
        )
    if charge_options:
        raise ValueError(
            "AIMNet2 smooth-ddPCM owns its charge source; remove #charge(...)."
        )
    if pbc is not None and pbc is not False:
        raise ValueError(
            "AIMNet2 smooth-ddPCM experimental workflows are non-periodic; remove #pbc."
        )

    if task is None:
        return
    normalized_task = _normalized(task)
    if normalized_task not in _TASK_METHODS:
        raise ValueError(
            "AIMNet2 smooth-ddPCM experimental workflows support only SP, OPT/LBFGS, "
            "FREQ/MW, and TS/PRFO."
        )
    actual_method = None
    if task_params is not None and task_params.get("method") is not None:
        actual_method = _normalized(task_params["method"])
    required_method = _TASK_METHODS[normalized_task]
    if actual_method != required_method:
        if required_method is None:
            raise ValueError(
                "AIMNet2 smooth-ddPCM SP does not accept an optimization method."
            )
        raise ValueError(
            f"AIMNet2 smooth-ddPCM {normalized_task.upper()} requires "
            f"method={required_method}."
        )


def validate_aimnet2_experimental_atoms(atoms: Any) -> None:
    """Validate the small, neutral-singlet, unconstrained HCNO domain."""

    if atoms is None:
        raise ValueError("AIMNet2 smooth-ddPCM requires one molecule.")
    try:
        atom_count = len(atoms)
    except TypeError as exc:
        raise TypeError("AIMNet2 smooth-ddPCM requires an ASE Atoms molecule.") from exc
    if atom_count < 1:
        raise ValueError("AIMNet2 smooth-ddPCM requires one non-empty molecule.")

    info = getattr(atoms, "info", None)
    if not isinstance(info, Mapping) or "charge" not in info or "mult" not in info:
        raise ValueError(
            "AIMNet2 smooth-ddPCM requires an explicit '0 1' charge/multiplicity line."
        )
    if info["charge"] != 0 or info["mult"] != 1:
        raise ValueError(
            "AIMNet2 smooth-ddPCM experimental workflows require a neutral singlet "
            "with charge/multiplicity '0 1'."
        )

    symbols = tuple(str(symbol) for symbol in atoms.get_chemical_symbols())
    unsupported = sorted(set(symbols).difference(_SUPPORTED_ELEMENTS))
    if unsupported:
        raise ValueError(
            "AIMNet2 smooth-ddPCM currently supports only H, C, N, and O; "
            "unsupported elements: " + ", ".join(unsupported) + "."
        )
    positions = atoms.get_positions()
    if len(positions) != atom_count or any(
        len(row) != 3 or any(not math.isfinite(float(value)) for value in row)
        for row in positions
    ):
        raise ValueError("AIMNet2 smooth-ddPCM coordinates must be finite Nx3 values.")
    if any(bool(value) for value in atoms.get_pbc()):
        raise ValueError("AIMNet2 smooth-ddPCM accepts only non-periodic molecules.")
    if getattr(atoms, "constraints", None):
        raise ValueError(
            "AIMNet2 smooth-ddPCM experimental workflows do not accept atom constraints."
        )


__all__ = [
    "AIMNET2_EXPERIMENTAL_PROFILE",
    "AIMNET2_EXPERIMENTAL_PROVIDER",
    "AIMNET2_EXPERIMENTAL_WORKFLOW_ID",
    "is_aimnet2_experimental_request",
    "validate_aimnet2_experimental_atoms",
    "validate_aimnet2_experimental_settings",
]
