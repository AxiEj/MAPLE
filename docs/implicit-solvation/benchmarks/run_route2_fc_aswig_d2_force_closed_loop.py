#!/usr/bin/env python3
"""Two-coordinate closed-work gate for the D2-canonical Route-2 force.

The loop is a fixed small rectangle in two Cartesian atom coordinates around
one frozen acetone geometry.  At all corners and edge midpoints this runner
uses the same direct-PCM scalar and the same fixed-point-adjoint
solvent-correction force.  Composite Simpson quadrature estimates

    integral F_correction(R) . dR.

A conservative implementation must return zero within the predeclared
quadrature/SCF tolerance.  This is a local integrability gate, not a claim of
a common stationary MACE--PCM electronic free-energy functional.
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


PROTOCOL_ID = "route2-fc-aswig-d2-mace-acetone-closed-work-v1"
SCHEMA_VERSION = 1
FIRST_COORDINATE = (0, 0)
SECOND_COORDINATE = (1, 1)
HALF_WIDTHS_ANGSTROM = (3.0e-3, 3.0e-3)
GATES = {
    "absolute_closed_work_ev": 1.0e-6,
    "relative_closed_work_path_scale": 1.0e-5,
    "require_fixed_surface_cardinality": True,
    "require_nominal_roots": True,
    "require_local_multistart_agreement": True,
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".omx"
    / "benchmarks"
    / "route2-fc-aswig-d2-mace-acetone-closed-loop"
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


def _offset_atoms(base, offset: tuple[float, float]):
    atoms = base.copy()
    (first_atom, first_axis), (second_atom, second_axis) = (
        FIRST_COORDINATE,
        SECOND_COORDINATE,
    )
    atoms.positions[first_atom, first_axis] += offset[0]
    atoms.positions[second_atom, second_axis] += offset[1]
    return atoms


def _sample(base, calculator, settings: dict[str, Any], offset: tuple[float, float]):
    atoms = _offset_atoms(base, offset)
    result = common._evaluate(atoms, calculator, settings["solv"], with_force=True)
    root = common._nominal_root(result)
    force = np.asarray(result.forces_hartree_per_angstrom, dtype=float)
    if force.shape != (len(atoms), 3) or not np.all(np.isfinite(force)):
        raise RuntimeError("Closed-loop sample returned a non-finite correction force.")
    surface_size = result.provenance["continuum_provider"].get("surface_size")
    if not isinstance(surface_size, int) or surface_size <= 0:
        raise RuntimeError("Closed-loop sample lacks fixed-topology surface size.")
    return {
        "offset_angstrom": [float(offset[0]), float(offset[1])],
        "energy_hartree": float(result.energy_hartree),
        "force_hartree_per_angstrom": force.tolist(),
        "force_sum_hartree_per_angstrom": np.sum(force, axis=0).tolist(),
        "surface_size": surface_size,
        "root": root,
    }


def _simpson_edge_work(
    start: dict[str, Any],
    midpoint: dict[str, Any],
    end: dict[str, Any],
) -> float:
    start_offset = np.asarray(start["offset_angstrom"], dtype=float)
    end_offset = np.asarray(end["offset_angstrom"], dtype=float)
    displacement = end_offset - start_offset
    if np.count_nonzero(np.abs(displacement) > 0.0) != 1:
        raise RuntimeError("Closed-loop edge must vary exactly one coordinate.")
    direction = np.zeros((2, 3), dtype=float)
    direction[0, FIRST_COORDINATE[1]] = displacement[0]
    direction[1, SECOND_COORDINATE[1]] = displacement[1]
    forces = (
        np.asarray(start["force_hartree_per_angstrom"], dtype=float),
        np.asarray(midpoint["force_hartree_per_angstrom"], dtype=float),
        np.asarray(end["force_hartree_per_angstrom"], dtype=float),
    )
    scalar_forces = np.asarray(
        (
            forces[0][FIRST_COORDINATE[0]] @ direction[0]
            + forces[0][SECOND_COORDINATE[0]] @ direction[1],
            forces[1][FIRST_COORDINATE[0]] @ direction[0]
            + forces[1][SECOND_COORDINATE[0]] @ direction[1],
            forces[2][FIRST_COORDINATE[0]] @ direction[0]
            + forces[2][SECOND_COORDINATE[0]] @ direction[1],
        ),
        dtype=float,
    )
    return float((scalar_forces[0] + 4.0 * scalar_forces[1] + scalar_forces[2]) / 6.0)


def _gate(samples: dict[str, dict[str, Any]], edge_work_hartree: dict[str, float]) -> dict[str, Any]:
    cardinalities = {int(sample["surface_size"]) for sample in samples.values()}
    nominal = all(
        sample["root"]["scf_convergence"]["reason"]
        == "nominal-density-and-energy-v1"
        for sample in samples.values()
    )
    agreed = all(
        bool(sample["root"]["multi_start_root_agreement"]["agreed"])
        for sample in samples.values()
    )
    closed_work_hartree = float(sum(edge_work_hartree.values()))
    energies = np.asarray([sample["energy_hartree"] for sample in samples.values()])
    energy_span_ev = float((np.max(energies) - np.min(energies)) * Hartree)
    allowable_ev = max(
        GATES["absolute_closed_work_ev"],
        GATES["relative_closed_work_path_scale"] * energy_span_ev,
    )
    checks = {
        "fixed_surface_cardinality": len(cardinalities) == 1,
        "nominal_roots": nominal,
        "local_multistart_agreement": agreed,
        "closed_work": abs(closed_work_hartree * Hartree) <= allowable_ev,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "gates": dict(GATES),
        "checks": checks,
        "surface_cardinalities": sorted(cardinalities),
        "edge_work_hartree": dict(edge_work_hartree),
        "closed_work_hartree": closed_work_hartree,
        "closed_work_ev": float(closed_work_hartree * Hartree),
        "energy_span_ev": energy_span_ev,
        "allowed_absolute_closed_work_ev": allowable_ev,
        "passed": bool(all(checks.values())),
        "open_force_admission_gates": ("short_nve",),
    }


def run(*, prepared_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    base, candidate, mol2_path, mol2_digest = common._load_atoms(prepared_path)
    settings = common._settings()
    calculator = common._calculator(base, settings, device)
    half_x, half_y = HALF_WIDTHS_ANGSTROM
    offsets = {
        "A": (-half_x, -half_y),
        "AB_mid": (0.0, -half_y),
        "B": (half_x, -half_y),
        "BC_mid": (half_x, 0.0),
        "C": (half_x, half_y),
        "CD_mid": (0.0, half_y),
        "D": (-half_x, half_y),
        "DA_mid": (-half_x, 0.0),
    }
    samples = {
        label: _sample(base, calculator, settings, offset)
        for label, offset in offsets.items()
    }
    edge_work = {
        "A_to_B": _simpson_edge_work(samples["A"], samples["AB_mid"], samples["B"]),
        "B_to_C": _simpson_edge_work(samples["B"], samples["BC_mid"], samples["C"]),
        "C_to_D": _simpson_edge_work(samples["C"], samples["CD_mid"], samples["D"]),
        "D_to_A": _simpson_edge_work(samples["D"], samples["DA_mid"], samples["A"]),
    }
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
            "A pass establishes only local two-coordinate work conservation of "
            "the declared Route-2 direct-PCM correction scalar for one acetone "
            "conformer. It does not establish short-NVE behavior, public ASE "
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
        "coordinates": {
            "first": {"atom_index": FIRST_COORDINATE[0], "cartesian_index": FIRST_COORDINATE[1]},
            "second": {"atom_index": SECOND_COORDINATE[0], "cartesian_index": SECOND_COORDINATE[1]},
            "half_widths_angstrom": list(HALF_WIDTHS_ANGSTROM),
            "quadrature": "composite-simpson-four-edges",
        },
        "samples": samples,
    }
    payload["closed_loop_gate"] = _gate(samples, edge_work)
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
