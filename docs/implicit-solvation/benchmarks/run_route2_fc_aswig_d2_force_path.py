#!/usr/bin/env python3
"""Local coordinate-path gate for the D2-canonical Route-2 force profile.

The same direct-PCM scalar and the same analytic solvent-correction force are
evaluated along a five-point Cartesian displacement path.  Each point must
reach a nominal fixed point and three-start local root agreement; no path
point is allowed to silently fall back to a finite-resolution root.

The gate asks two non-overlapping questions:

1. Does the fixed-topology continuum retain exactly the same number of surface
   variables throughout the path?
2. Does the centered energy slope agree with the analytic correction force at
   every interior point?

It is intentionally not a public-force admission result: closed-loop work and
short-NVE behavior remain separate gates.
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


_COMMON_PATH = Path(common.__file__).resolve()

PROTOCOL_ID = "route2-fc-aswig-d2-mace-acetone-path-force-v1"
SCHEMA_VERSION = 1
ATOM_INDEX = 0
CARTESIAN_INDEX = 0
PATH_STEP_ANGSTROM = 5.0e-3
OFFSETS_ANGSTROM = (-1.0e-2, -5.0e-3, 0.0, 5.0e-3, 1.0e-2)
GATES = {
    "maximum_centered_fd_error_ev_per_angstrom": 2.0e-5,
    "require_fixed_surface_cardinality": True,
    "require_nominal_roots": True,
    "require_local_multistart_agreement": True,
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".omx"
    / "benchmarks"
    / "route2-fc-aswig-d2-mace-acetone-path-force"
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


def _surface_cardinality(result) -> int:
    continuum = result.provenance["continuum_provider"]
    value = continuum.get("surface_size")
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError("Fixed-topology continuum provenance omitted surface_size.")
    return value


def _sample(base, calculator, settings: dict[str, Any], offset: float) -> dict[str, Any]:
    atoms = base.copy()
    atoms.positions[ATOM_INDEX, CARTESIAN_INDEX] += offset
    result = common._evaluate(atoms, calculator, settings["solv"], with_force=True)
    root = common._nominal_root(result)
    force = np.asarray(result.forces_hartree_per_angstrom, dtype=float)
    if force.shape != (len(atoms), 3) or not np.all(np.isfinite(force)):
        raise RuntimeError("Path sample returned a non-finite correction force.")
    return {
        "offset_angstrom": float(offset),
        "energy_hartree": float(result.energy_hartree),
        "force_component_hartree_per_angstrom": float(
            force[ATOM_INDEX, CARTESIAN_INDEX]
        ),
        "force_sum_hartree_per_angstrom": np.sum(force, axis=0).tolist(),
        "surface_size": _surface_cardinality(result),
        "continuum": dict(result.provenance["continuum_provider"]),
        "cds": dict(result.provenance["cds_provider"]),
        "root": root,
    }


def _interior_derivative_records(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for previous, current, following in zip(
        samples[:-2], samples[1:-1], samples[2:], strict=True
    ):
        step = float(following["offset_angstrom"] - current["offset_angstrom"])
        if not np.isclose(
            current["offset_angstrom"] - previous["offset_angstrom"], step
        ):
            raise RuntimeError("Path offsets are not symmetric around an interior sample.")
        numeric_force = -(
            following["energy_hartree"] - previous["energy_hartree"]
        ) / (2.0 * step)
        error = abs(numeric_force - current["force_component_hartree_per_angstrom"])
        records.append(
            {
                "offset_angstrom": current["offset_angstrom"],
                "analytic_force_hartree_per_angstrom": current[
                    "force_component_hartree_per_angstrom"
                ],
                "centered_fd_force_hartree_per_angstrom": float(numeric_force),
                "absolute_error_hartree_per_angstrom": float(error),
                "absolute_error_ev_per_angstrom": float(error * Hartree),
            }
        )
    return records


def _gate(samples: list[dict[str, Any]], derivative: list[dict[str, Any]]) -> dict[str, Any]:
    cardinalities = {int(sample["surface_size"]) for sample in samples}
    nominal = all(
        sample["root"]["scf_convergence"]["reason"]
        == "nominal-density-and-energy-v1"
        for sample in samples
    )
    multi_start = all(
        bool(sample["root"]["multi_start_root_agreement"]["agreed"])
        for sample in samples
    )
    maximum_error = max(
        float(record["absolute_error_ev_per_angstrom"]) for record in derivative
    )
    checks = {
        "fixed_surface_cardinality": len(cardinalities) == 1,
        "nominal_roots": nominal,
        "local_multistart_agreement": multi_start,
        "centered_energy_force_agreement": maximum_error
        <= GATES["maximum_centered_fd_error_ev_per_angstrom"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "gates": dict(GATES),
        "checks": checks,
        "surface_cardinalities": sorted(cardinalities),
        "maximum_centered_fd_error_ev_per_angstrom": maximum_error,
        "passed": bool(all(checks.values())),
        "open_force_admission_gates": ("closed_loop_work", "short_nve"),
    }


def run(*, prepared_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    base, candidate, mol2_path, mol2_digest = common._load_atoms(prepared_path)
    settings = common._settings()
    calculator = common._calculator(base, settings, device)
    samples = [_sample(base, calculator, settings, offset) for offset in OFFSETS_ANGSTROM]
    derivative = _interior_derivative_records(samples)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc(),
        "git_head": common._git_head(),
        "git_worktree_status": common._git_worktree_status(),
        "runner": str(Path(__file__).relative_to(REPO_ROOT)),
        "runner_sha256": _sha256(Path(__file__)),
        "shared_kinematics_runner": str(_COMMON_PATH.relative_to(REPO_ROOT)),
        "shared_kinematics_runner_sha256": _sha256(_COMMON_PATH),
        "claim_boundary": (
            "A pass establishes local coordinate-path smoothness of the declared "
            "Route-2 direct-PCM correction scalar for one acetone Cartesian path. "
            "It does not establish closed-loop work, NVE behavior, public ASE "
            "forces, a common stationary electronic free-energy, or experimental accuracy."
        ),
        "record": {
            "compound_id": common.COMPOUND_ID,
            "name": candidate["name"],
            "atom_count": len(base),
            "mol2_path": str(mol2_path),
            "mol2_sha256": mol2_digest,
        },
        "profile": common.FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
        "declared_scalar": "0.5*<c_MACE-POLAR,f_reac_CPCM>+fixed-topology-aqueous-SMD-CDS",
        "path": {
            "atom_index": ATOM_INDEX,
            "cartesian_index": CARTESIAN_INDEX,
            "step_angstrom": PATH_STEP_ANGSTROM,
            "offsets_angstrom": list(OFFSETS_ANGSTROM),
        },
        "samples": samples,
        "interior_centered_energy_force": derivative,
    }
    payload["path_gate"] = _gate(samples, derivative)
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
