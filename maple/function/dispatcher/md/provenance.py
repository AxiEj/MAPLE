"""MD provenance manifest (WS3).

Every production MD run emits a ``*_md_manifest.json`` recording exactly what was
run and in what software/hardware/parameter context, so a result can be audited
and reproduced.  The manifest is assembled from three sinks:

* the environment (MAPLE git state, Python/NumPy/Torch/ASE versions, CUDA/GPU),
* the calculator (model, backend package + version, declared capabilities,
  model options) — captured by ``engine._mlp_initiator`` and attached to the
  calculator as ``maple_provenance``,
* the MD run (ensemble parameters, seed/RNG hash, DOF policy, cell/PBC/min-image,
  constraints status, partial-PBC policy, thermo basis, RST path, final-state
  hash) — captured by ``MDLogger.start_simulation`` / ``end_simulation``.

Everything here is best-effort: a failure to collect provenance must never abort
or perturb the simulation itself.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from maple.function.calculator._ase_unit_contract import (
    ASE_STRESS_UNIT,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
)
from ase import Atoms

from .rst_io import RST_HEADER

MANIFEST_SCHEMA_VERSION = "1.0.0"
# Keep the RST schema version visible alongside the manifest so constrained-MD /
# partial-PBC / anisotropic-NPT / new velocity representations can evolve the
# formats with an explicit, backward-compatible version bump.
RST_SCHEMA_VERSION = RST_HEADER


def _safe(callable_, default=None):
    try:
        return callable_()
    except Exception:
        return default


@lru_cache(maxsize=1)
def collect_environment_provenance() -> Dict[str, Any]:
    """Software/hardware environment provenance (cached for the process)."""
    repo = Path(__file__).resolve().parents[4]

    def _git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()

    env: Dict[str, Any] = {
        "maple_git_commit": _safe(lambda: _git("rev-parse", "HEAD")),
        "maple_git_branch": _safe(lambda: _git("rev-parse", "--abbrev-ref", "HEAD")),
        "maple_git_dirty": _safe(lambda: bool(_git("status", "--porcelain"))),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _safe(lambda: __import__("numpy").__version__),
        "ase": _safe(lambda: __import__("ase").__version__),
    }

    def _torch_info():
        import torch

        info = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
        return info

    env.update(_safe(_torch_info, {"torch": None, "cuda_available": None}))
    return env


def _package_version(module_name: str) -> Optional[str]:
    import importlib.metadata as md

    return _safe(lambda: md.version(module_name))


def collect_calculator_provenance(
    calc,
    model: Optional[str] = None,
    device: Optional[Any] = None,
    model_options: Optional[dict] = None,
) -> Dict[str, Any]:
    """Calculator/model/backend provenance from a built calculator."""
    backend_class = type(calc).__name__ if calc is not None else None
    backend_module = type(calc).__module__.split(".")[0] if calc is not None else None

    # Map the MAPLE calculator family to its backing PyPI/git package.
    pkg_candidates = {"mace", "aimnet", "fairchem", "graph_longrange", "torchani"}
    backend_packages = {
        name: ver for name in pkg_candidates if (ver := _package_version(name)) is not None
    }

    options = model_options
    if options is None:
        options = getattr(calc, "maple_model_options", None)

    long_range_method = _long_range_method(options)

    return {
        "model": model if model is not None else getattr(calc, "maple_model_name", None),
        "backend_class": backend_class,
        "backend_module": backend_module,
        "backend_packages": backend_packages,
        "device": str(device) if device is not None else None,
        "model_options": dict(options) if isinstance(options, dict) else None,
        "capabilities": {
            "pbc_md_supported": bool(getattr(calc, "maple_pbc_md_supported", False)),
            "stress_supported": bool(getattr(calc, "maple_stress_supported", False)),
            # Calculator's self-declared MD unit contract (the values the admission
            # gate checked against); the units the MD layer enforced are recorded in
            # the run context under "unit_contract".
            "energy_unit": getattr(calc, "maple_energy_unit", None),
            "force_unit": getattr(calc, "maple_force_unit", None),
            "stress_unit": getattr(calc, "maple_stress_unit", None),
            "neighbor_cutoff_A": _safe(lambda: _calc_cutoff(calc)),
            "long_range_method": long_range_method,
        },
    }


def _calc_cutoff(calc) -> Optional[float]:
    for attr in ("neighbor_cutoff_A", "cutoff_A", "maple_neighbor_cutoff"):
        value = getattr(calc, attr, None)
        if value is not None:
            return float(value)
    return None


def _long_range_method(model_options) -> str:
    """Long-range/electrostatics method from MAPLE model options.

    MAPLE encodes it under ``model_options["coulomb"]``; ``"none"`` when absent
    (short-range only).  Single definition shared by the manifest
    (``collect_calculator_provenance``) and the MD start banner so the two stay
    in sync.
    """
    if isinstance(model_options, dict):
        return model_options.get("coulomb") or "none"
    return "none"


def system_provenance(atoms: Atoms) -> Dict[str, Any]:
    """Geometry/PBC provenance, including the minimum-image radius when periodic."""
    pbc = [bool(flag) for flag in atoms.pbc]
    info: Dict[str, Any] = {
        "n_atoms": len(atoms),
        "formula": _safe(lambda: atoms.get_chemical_formula()),
        "pbc": pbc,
        "cell": _safe(lambda: np.asarray(atoms.cell.array, dtype=float).tolist()),
        "volume_A3": _safe(lambda: float(atoms.get_volume())) if any(pbc) else None,
    }
    if any(pbc):
        from maple.function.calculator.set_calculator import _minimum_image_radius_A

        radius = _safe(lambda: _minimum_image_radius_A(atoms))
        if radius is not None:
            info["min_image_radius_A"] = float(radius[0])
            info["shortest_lattice_vector_A"] = float(radius[1])
    return info


def final_state_hash(atoms: Atoms, velocities: Optional[np.ndarray]) -> str:
    """Stable SHA-256 over the final positions, cell, species and velocities."""
    hasher = hashlib.sha256()
    for array in (
        np.asarray(atoms.get_positions(), dtype=float),
        np.asarray(atoms.cell.array, dtype=float),
        np.asarray(atoms.get_atomic_numbers()),
        np.asarray(velocities, dtype=float) if velocities is not None else np.zeros(0),
    ):
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _params_snapshot(params) -> Dict[str, Any]:
    if is_dataclass(params):
        return {k: _jsonable(v) for k, v in asdict(params).items()}
    if isinstance(params, dict):
        return {k: _jsonable(v) for k, v in params.items()}
    return {}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_run_context(
    *,
    params,
    dof_policy,
    ensemble: str,
    rng_state_hex: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the run-specific manifest context an ensemble passes to the logger."""
    pbc = None  # filled in by the logger from atoms; kept here for completeness
    thermo_basis = (
        {"pressure": "post-rescale", "diagnostic_columns": "pre-rescale"}
        if str(ensemble).lower() == "npt"
        else None
    )
    return {
        "ensemble": str(ensemble).lower(),
        "params": _params_snapshot(params),
        "seed": _jsonable(getattr(params, "random_seed", None)),
        "rng_state_hex": rng_state_hex,
        # Surfaced to the manifest top level by build_md_manifest (popped there
        # so it is not duplicated inside the "run" block).
        "validation_artifact_id": getattr(params, "validation_artifact_id", "") or None,
        "dof_policy": {
            "init_n_dof": dof_policy.init_n_dof,
            "runtime_n_dof": dof_policy.runtime_n_dof,
            "init_description": dof_policy.init_description,
            "runtime_description": dof_policy.runtime_description,
        },
        # MD rejects ASE constraints up front (WS0-B); a run that got this far has none.
        "constraints_status": "none (rejected before MD; not supported)",
        "partial_pbc": {
            "allowed": bool(getattr(params, "allow_partial_pbc", False)),
        },
        "cutoff_policy": {
            # Whether this run opted out of the minimum-image neighbor-cutoff gate.
            "allow_unknown_cutoff": bool(getattr(params, "allow_unknown_cutoff", False)),
        },
        "unit_contract": {
            # Units the MD layer enforced at admission; the calculator's own declared
            # units are under calculator.capabilities.{energy,force,stress}_unit.
            "energy": MAPLE_ENERGY_UNIT,
            "force": MAPLE_FORCE_UNIT,
            "stress": ASE_STRESS_UNIT,
        },
        "thermo_basis": thermo_basis,
    }


def build_md_manifest(
    *,
    atoms: Atoms,
    run_context: Dict[str, Any],
    calc_provenance: Optional[Dict[str, Any]],
    rst_path: Optional[str] = None,
    final_velocities: Optional[np.ndarray] = None,
    validation_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the full MD provenance manifest dict."""
    run_context = dict(run_context or {})
    # The id is surfaced once at the manifest top level; prefer an explicit
    # argument, else take the one captured into the run context at run start.
    ctx_artifact_id = run_context.pop("validation_artifact_id", None)
    if validation_artifact_id is None:
        validation_artifact_id = ctx_artifact_id
    partial = run_context.get("partial_pbc") or {}
    pbc = [bool(flag) for flag in atoms.pbc]
    partial["is_partial"] = any(pbc) and not all(pbc)
    run_context["partial_pbc"] = partial

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "rst_schema_version": RST_SCHEMA_VERSION,
        "environment": collect_environment_provenance(),
        "calculator": calc_provenance,
        "system": system_provenance(atoms),
        "run": run_context,
        "rst_path": rst_path,
        "final_state_hash": _safe(lambda: final_state_hash(atoms, final_velocities)),
        "validation_artifact_id": validation_artifact_id,
    }


def write_md_manifest(path, manifest: Dict[str, Any]) -> None:
    """Write the manifest as pretty JSON."""
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
