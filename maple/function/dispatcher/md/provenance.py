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

    long_range_method = _calc_long_range_method(calc, options)

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
            "local_descriptor_cutoff_A": _safe(
                lambda: _calc_optional_float(calc, "local_descriptor_cutoff_A")
            ),
            "short_range_realspace_cutoff_A": _safe(
                lambda: _calc_optional_float(calc, "short_range_realspace_cutoff_A")
            ),
            "long_range_coulomb_cutoff_A": _safe(
                lambda: _calc_optional_float(
                    calc, "long_range_coulomb_cutoff_A", "lrcoulomb_cutoff_A"
                )
            ),
            "long_range_method": long_range_method,
        },
    }


def _calc_cutoff(calc) -> Optional[float]:
    for attr in ("neighbor_cutoff_A", "cutoff_A", "maple_neighbor_cutoff"):
        value = getattr(calc, attr, None)
        if value is not None:
            return float(value)
    return None


def _calc_optional_float(calc, *attrs: str) -> Optional[float]:
    for attr in attrs:
        value = getattr(calc, attr, None)
        if value is not None:
            return float(value)
    return None


def _calc_long_range_method(calc, model_options=None) -> str:
    method = _long_range_method(model_options)
    if method == "none" and calc is not None:
        method = _normalize_long_range_method(getattr(calc, "lrcoulomb_method", None))
    return method


def _long_range_method(model_options) -> str:
    """Long-range/electrostatics method from MAPLE model options.

    MAPLE encodes it under ``model_options["coulomb"]``; ``"none"`` when absent
    (short-range only).  Single definition shared by the manifest
    (``collect_calculator_provenance``) and the MD start banner so the two stay
    in sync.
    """
    if isinstance(model_options, dict):
        return _normalize_long_range_method(model_options.get("coulomb"))
    return "none"


def _normalize_long_range_method(value) -> str:
    """Normalize absent/short-range long-range-method labels for comparisons."""
    if value is None:
        return "none"
    text = str(value).strip().lower()
    if text in {"", "none", "n/a", "na", "null"}:
        return "none"
    return text


def companion_manifest_for_rst(used_path) -> Path:
    """Return the provenance manifest path that lives next to an RST source.

    ``MDLogger`` writes ``<base>_md.rst``, ``<base>_md_prev.rst`` and
    ``<base>_md_manifest.json``.  An explicit ``rst_file`` may point to a
    different directory, so derive the companion from the actual RST path rather
    than from the current output prefix.
    """
    path = Path(used_path)
    stem = path.stem
    if stem.endswith("_md_prev"):
        manifest_stem = stem[:-5]  # strip "_prev", keep the canonical "_md"
    elif stem.endswith("_md"):
        manifest_stem = stem
    else:
        manifest_stem = stem
    return path.with_name(f"{manifest_stem}_manifest.json")


def _manifest_marked_legacy_or_incomplete(manifest: Dict[str, Any]) -> bool:
    markers = [
        manifest.get("manifest_status"),
        manifest.get("status"),
        manifest.get("schema_status"),
        (manifest.get("run") or {}).get("manifest_status"),
    ]
    if any(str(marker).strip().lower() in {"legacy", "incomplete"} for marker in markers if marker is not None):
        return True
    return bool(manifest.get("legacy") or manifest.get("incomplete"))


def _path_get(mapping: Optional[Dict[str, Any]], dotted: str, default=None):
    current: Any = mapping
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def restart_manifest_consistency_issues(
    manifest: Dict[str, Any],
    *,
    atoms: Atoms,
    state: Dict[str, Any],
) -> tuple[list[str], list[str], bool]:
    """Return (missing_fields, mismatches, legacy_or_incomplete)."""
    if _manifest_marked_legacy_or_incomplete(manifest):
        return [], [], True

    pbc = [bool(flag) for flag in atoms.pbc]
    calc = getattr(atoms, "calc", None)
    is_pbc = any(pbc)
    required = [
        "system.n_atoms",
        "system.pbc",
        "system.cell",
        "run.ensemble",
        "run.params.timestep",
        "calculator.model",
        "calculator.capabilities.energy_unit",
        "calculator.capabilities.force_unit",
    ]
    if is_pbc:
        required += [
            "calculator.capabilities.neighbor_cutoff_A",
            "calculator.capabilities.long_range_method",
        ]
    if bool(getattr(calc, "maple_stress_supported", False)):
        required.append("calculator.capabilities.stress_unit")

    missing = [field for field in required if _path_get(manifest, field) is None]
    mismatches: list[str] = []
    if missing:
        return missing, mismatches, False

    def add_mismatch(field: str, got, expected) -> None:
        mismatches.append(f"{field}: manifest={got!r}, current={expected!r}")

    if int(_path_get(manifest, "system.n_atoms")) != int(state["natoms"]):
        add_mismatch("system.n_atoms", _path_get(manifest, "system.n_atoms"), state["natoms"])

    manifest_pbc = [bool(flag) for flag in _path_get(manifest, "system.pbc")]
    if manifest_pbc != pbc:
        add_mismatch("system.pbc", manifest_pbc, pbc)

    manifest_cell = np.asarray(_path_get(manifest, "system.cell"), dtype=float)
    current_cell = np.asarray(atoms.cell.array, dtype=float)
    if manifest_cell.shape != (3, 3) or not np.allclose(
        manifest_cell, current_cell, rtol=1e-9, atol=1e-9
    ):
        add_mismatch("system.cell", manifest_cell.tolist(), current_cell.tolist())

    manifest_ensemble = str(_path_get(manifest, "run.ensemble")).lower()
    if manifest_ensemble != str(state["ensemble"]).lower():
        add_mismatch("run.ensemble", manifest_ensemble, state["ensemble"])

    manifest_timestep = float(_path_get(manifest, "run.params.timestep"))
    if not np.isclose(manifest_timestep, float(state["timestep"]), rtol=0.0, atol=1e-12):
        add_mismatch("run.params.timestep", manifest_timestep, state["timestep"])

    current_model = getattr(calc, "maple_model_name", None)
    manifest_model = _path_get(manifest, "calculator.model")
    if current_model is not None and str(manifest_model).lower() != str(current_model).lower():
        add_mismatch("calculator.model", manifest_model, current_model)

    for cap_name, attr in (
        ("energy_unit", "maple_energy_unit"),
        ("force_unit", "maple_force_unit"),
        ("stress_unit", "maple_stress_unit"),
    ):
        current_value = getattr(calc, attr, None)
        manifest_value = _path_get(manifest, f"calculator.capabilities.{cap_name}")
        if current_value is not None and manifest_value != current_value:
            add_mismatch(f"calculator.capabilities.{cap_name}", manifest_value, current_value)

    if is_pbc:
        manifest_cutoff = float(_path_get(manifest, "calculator.capabilities.neighbor_cutoff_A"))
        current_cutoff = _calc_cutoff(calc)
        if current_cutoff is not None and not np.isclose(
            manifest_cutoff, float(current_cutoff), rtol=1e-7, atol=1e-9
        ):
            add_mismatch("calculator.capabilities.neighbor_cutoff_A", manifest_cutoff, current_cutoff)

        manifest_lr = _normalize_long_range_method(
            _path_get(manifest, "calculator.capabilities.long_range_method")
        )
        current_lr = _calc_long_range_method(calc, getattr(calc, "maple_model_options", None))
        if manifest_lr != current_lr:
            add_mismatch("calculator.capabilities.long_range_method", manifest_lr, current_lr)

    return missing, mismatches, False


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
    is_npt = str(ensemble).lower() == "npt"
    thermo_basis = (
        {"pressure": "post-rescale", "diagnostic_columns": "pre-rescale"}
        if is_npt
        else None
    )
    # NPT barostats here scale the cell by a single scalar: machine-visible isotropic-only
    # flag so a consumer never mistakes this for anisotropic (Parrinello-Rahman / MTTK)
    # cell relaxation. Clamp counts are appended by the NPT driver after the run.
    barostat = (
        {
            "type": getattr(params, "barostat", None),
            "mode": "isotropic",
            "integrator": (
                "bernetti-bussi-reversible-euler"
                if getattr(params, "barostat", None) == "c-rescale"
                else "sequential-equilibration"
            ),
            "stride_NP": int(getattr(params, "barostat_stride", 1) or 1),
            "note": "isotropic hydrostatic scaling only; no shear / cell-shape / surface tension",
        }
        if is_npt
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
        "barostat": barostat,
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
