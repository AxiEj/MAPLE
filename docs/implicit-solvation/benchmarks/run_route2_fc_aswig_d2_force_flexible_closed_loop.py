#!/usr/bin/env python3
"""Two-torsion closed-work gate on the flexible historical cavity case.

This is the second independent force test for 2-acetoxyethyl acetate.  The
same fixed-topology direct-PCM total scalar used by the flexible torsion
finite-difference gate is sampled on a small rectangle in two *physical*
torsional coordinates: central C3--C4 and terminal C4--O3.  Composite Simpson
quadrature evaluates the total work around the closed loop.

Unlike the old pyddx calculation, every geometry keeps every FC-aSWIG surface
unknown.  The test therefore discriminates a genuine same-scalar conservative
force from an apparent single-point finite-difference agreement that crosses a
changing cavity active set.
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

import run_route2_fc_aswig_d2_force_flexible_torsion as torsion
import run_route2_fc_aswig_d2_force_kinematics as common


SCHEMA_VERSION = 1
PROTOCOL_ID = "route2-fc-aswig-d2-mace-flexible-two-torsion-closed-work-v1"
COMPOUND_ID = torsion.COMPOUND_ID

# First coordinate is the historical C3--C4 rotatable bond.  The second
# coordinate rotates only the terminal acetate around C4--O3.  These are the
# frozen, topologically distinct coordinates from the prior active-set audit.
FIRST = {
    "axis_origin_index": torsion.AXIS_ORIGIN_INDEX,
    "axis_direction_index": torsion.AXIS_DIRECTION_INDEX,
    "fragment_indices": torsion.ROTATED_FRAGMENT_INDICES,
    "label": "central-C3-C4",
}
SECOND = {
    "axis_origin_index": 5,
    "axis_direction_index": 6,
    "fragment_indices": (6, 7, 8, 9, 17, 18, 19),
    "label": "terminal-C4-O3",
}
HALF_WIDTH_DEGREES = (0.30, 0.30)
GATES = {
    "absolute_closed_work_ev": 1.0e-6,
    "relative_closed_work_path_scale": 1.0e-5,
    "require_fixed_surface_cardinality": True,
    "require_nominal_roots": True,
    "require_local_multistart_agreement": True,
    "maximum_net_force_hartree_per_angstrom": 1.0e-10,
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".omx"
    / "benchmarks"
    / "route2-fc-aswig-d2-mace-flexible-closed-loop"
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


def _coordinate_tuple(specification: dict[str, object]) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(specification["axis_origin_index"]),
        int(specification["axis_direction_index"]),
        tuple(int(index) for index in specification["fragment_indices"]),
    )


def two_torsion_positions(
    source_positions: np.ndarray,
    offset_degrees: tuple[float, float],
) -> np.ndarray:
    """Apply absolute central then terminal rotations without path history."""

    first_origin, first_direction, first_fragment = _coordinate_tuple(FIRST)
    second_origin, second_direction, second_fragment = _coordinate_tuple(SECOND)
    after_first = torsion.rotate_fragment(
        source_positions,
        angle_radians=float(np.deg2rad(offset_degrees[0])),
        axis_origin_index=first_origin,
        axis_direction_index=first_direction,
        rotated_fragment_indices=first_fragment,
    )
    return torsion.rotate_fragment(
        after_first,
        angle_radians=float(np.deg2rad(offset_degrees[1])),
        axis_origin_index=second_origin,
        axis_direction_index=second_direction,
        rotated_fragment_indices=second_fragment,
    )


def _generalized_forces(positions: np.ndarray, force: np.ndarray) -> np.ndarray:
    first_origin, first_direction, first_fragment = _coordinate_tuple(FIRST)
    second_origin, second_direction, second_fragment = _coordinate_tuple(SECOND)
    first_tangent = torsion.torsional_tangent(
        positions,
        axis_origin_index=first_origin,
        axis_direction_index=first_direction,
        rotated_fragment_indices=first_fragment,
    )
    second_tangent = torsion.torsional_tangent(
        positions,
        axis_origin_index=second_origin,
        axis_direction_index=second_direction,
        rotated_fragment_indices=second_fragment,
    )
    return np.asarray(
        (np.sum(force * first_tangent), np.sum(force * second_tangent)),
        dtype=float,
    )


def _sample(base, calculator, settings: dict[str, Any], offset: tuple[float, float]) -> dict[str, Any]:
    atoms = base.copy()
    atoms.set_positions(two_torsion_positions(base.get_positions(), offset))
    # The flexible runner already evaluates the complete total direct scalar,
    # verifies the nominal root, and returns total (gas + correction) force.
    total = torsion._total_sample(atoms, calculator, settings, float(offset[0]))
    force = np.asarray(total["total_force_hartree_per_angstrom"], dtype=float)
    if force.shape != (len(atoms), 3) or not np.all(np.isfinite(force)):
        raise RuntimeError("Flexible closed-loop sample returned an invalid total force.")
    generalized = _generalized_forces(atoms.get_positions(), force)
    return {
        "offset_degrees": [float(offset[0]), float(offset[1])],
        "offset_radians": [float(np.deg2rad(offset[0])), float(np.deg2rad(offset[1]))],
        "total_energy_hartree": total["total_energy_hartree"],
        "generalized_forces_hartree_per_radian": generalized.tolist(),
        "surface_size": total["surface_size"],
        "scf_reason": total["scf_reason"],
        "local_multistart_agreed": total["local_multistart_agreed"],
        "force_sum_hartree_per_angstrom": total["force_sum_hartree_per_angstrom"],
    }


def _simpson_edge_work(
    start: dict[str, Any],
    midpoint: dict[str, Any],
    end: dict[str, Any],
) -> float:
    start_offset = np.asarray(start["offset_radians"], dtype=float)
    end_offset = np.asarray(end["offset_radians"], dtype=float)
    displacement = end_offset - start_offset
    if np.count_nonzero(np.abs(displacement) > 0.0) != 1:
        raise RuntimeError("Each closed-loop edge must vary exactly one torsion.")
    force_rows = (
        np.asarray(start["generalized_forces_hartree_per_radian"], dtype=float),
        np.asarray(midpoint["generalized_forces_hartree_per_radian"], dtype=float),
        np.asarray(end["generalized_forces_hartree_per_radian"], dtype=float),
    )
    scalar = np.asarray(
        tuple(float(force @ displacement) for force in force_rows),
        dtype=float,
    )
    return float((scalar[0] + 4.0 * scalar[1] + scalar[2]) / 6.0)


def _gate(samples: dict[str, dict[str, Any]], edge_work: dict[str, float]) -> dict[str, Any]:
    cardinalities = {int(row["surface_size"]) for row in samples.values()}
    closed_work_hartree = float(sum(edge_work.values()))
    energy_values = np.asarray(
        [float(row["total_energy_hartree"]) for row in samples.values()], dtype=float
    )
    span_ev = float((np.max(energy_values) - np.min(energy_values)) * Hartree)
    allowed_ev = max(
        GATES["absolute_closed_work_ev"],
        GATES["relative_closed_work_path_scale"] * span_ev,
    )
    max_net_force = max(
        float(np.max(np.abs(np.asarray(row["force_sum_hartree_per_angstrom"], dtype=float))))
        for row in samples.values()
    )
    checks = {
        "fixed_surface_cardinality": len(cardinalities) == 1,
        "nominal_roots": all(
            row["scf_reason"] == "nominal-density-and-energy-v1"
            for row in samples.values()
        ),
        "local_multistart_agreement": all(
            bool(row["local_multistart_agreed"]) for row in samples.values()
        ),
        "net_force": max_net_force <= GATES["maximum_net_force_hartree_per_angstrom"],
        "closed_work": abs(closed_work_hartree * Hartree) <= allowed_ev,
    }
    return {
        "gates": dict(GATES),
        "checks": {name: bool(value) for name, value in checks.items()},
        "closed_work_hartree": closed_work_hartree,
        "closed_work_ev": float(closed_work_hartree * Hartree),
        "energy_span_ev": span_ev,
        "allowed_absolute_closed_work_ev": allowed_ev,
        "surface_cardinalities": sorted(cardinalities),
        "maximum_net_force_hartree_per_angstrom": max_net_force,
        "passed": bool(all(checks.values())),
        "remaining_force_admission_scope": (
            "A clean-tree replay and explicit narrow-domain public API review "
            "remain required; this does not certify all molecular geometries."
        ),
    }


def run(*, prepared_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    base, candidate, mol2_path, mol2_digest = common._load_atoms(
        prepared_path,
        compound_id=COMPOUND_ID,
    )
    settings = common._settings()
    calculator = common._calculator(base, settings, device)
    half_first, half_second = HALF_WIDTH_DEGREES
    offsets = {
        "A": (-half_first, -half_second),
        "AB_mid": (0.0, -half_second),
        "B": (half_first, -half_second),
        "BC_mid": (half_first, 0.0),
        "C": (half_first, half_second),
        "CD_mid": (0.0, half_second),
        "D": (-half_first, half_second),
        "DA_mid": (-half_first, 0.0),
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
        "torsion_runner": str(Path(torsion.__file__).resolve().relative_to(REPO_ROOT)),
        "torsion_runner_sha256": _sha256(Path(torsion.__file__).resolve()),
        "shared_kinematics_runner": str(Path(common.__file__).resolve().relative_to(REPO_ROOT)),
        "shared_kinematics_runner_sha256": _sha256(Path(common.__file__).resolve()),
        "claim_boundary": (
            "This is a local flexible two-torsion work-conservation test of the "
            "declared total direct-PCM operational scalar. It is not evidence of "
            "a universal physical free-energy functional or broad public force support."
        ),
        "record": {
            "compound_id": COMPOUND_ID,
            "name": candidate["name"],
            "atom_count": len(base),
            "mol2_path": str(mol2_path),
            "mol2_sha256": mol2_digest,
        },
        "profile": common.FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
        "total_declared_scalar": (
            "E_MACE-POLAR(gas)+0.5*<c_MACE-POLAR,f_reac_CPCM>+"
            "fixed-topology-aqueous-SMD-CDS"
        ),
        "coordinates": {"first": FIRST, "second": SECOND, "half_width_degrees": list(HALF_WIDTH_DEGREES)},
        "samples": samples,
        "edge_work_hartree": edge_work,
    }
    payload["flexible_closed_loop_gate"] = _gate(samples, edge_work)
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
