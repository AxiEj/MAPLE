#!/usr/bin/env python3
"""Run the pure MACE-POLAR frozen-source ddX E/F/V/HVP water canary.

This is a derivative-consistency artifact, not a solvation-accuracy benchmark.
The default 302-point smooth CDS grid is intentionally inexpensive and has a
different identity from the production grid; the output records that choice.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

from ase import Atoms
import numpy as np

from maple.solvation.experimental.mace_polar_frozen_ddx import (
    build_water_mace_polar_frozen_ddx_pes,
)
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter

ARTIFACT_ID = "route2-pure-mace-polar-frozen-ddx-water-derivative-canary-v1"
DEFAULT_CDS_GRID_POINTS = 302
FORCE_STEPS_ANGSTROM = (2.0e-4, 1.0e-4, 5.0e-5)
STRAIN_STEPS = (4.0e-4, 2.0e-4, 1.0e-4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]]
        ),
        info={"charge": 0, "mult": 1},
    )


def _direction(seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(3, 3))
    values -= np.mean(values, axis=0, keepdims=True)
    values /= np.linalg.norm(values)
    return values


def run(args: argparse.Namespace) -> dict[str, object]:
    atoms = _water()
    model = build_official_mace_polar_1_m_radial_gto_adapter(
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    pes = build_water_mace_polar_frozen_ddx_pes(
        model,
        atoms.get_chemical_symbols(),
        cds_grid_points=args.cds_grid_points,
    )
    state = pes.solve(atoms)
    force = pes.evaluate_forces(atoms, central_state=state)

    force_direction = _direction(20260816)
    analytic_gradient = -float(np.vdot(force.total_forces_eV_per_A, force_direction))
    force_rows = []
    for step in FORCE_STEPS_ANGSTROM:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * force_direction
        minus.positions -= step * force_direction
        finite_difference = (
            pes.get_potential_energy(plus) - pes.get_potential_energy(minus)
        ) / (2.0 * step)
        force_rows.append(
            {
                "step_angstrom": step,
                "finite_difference_gradient_eV_per_A": finite_difference,
                "absolute_error_eV_per_A": abs(finite_difference - analytic_gradient),
            }
        )

    virial = pes.molecular_virial(atoms, force_evaluation=force)
    strain = np.asarray([[0.12, 0.02, -0.01], [0.02, -0.08, 0.03], [-0.01, 0.03, 0.04]])
    analytic_strain_gradient = float(np.vdot(virial.strain_gradient_eV, strain))
    strain_rows = []
    for step in STRAIN_STEPS:
        energies = []
        for sign in (1.0, -1.0):
            displaced = atoms.copy()
            transform = np.eye(3) + sign * step * strain
            displaced.positions = (
                virial.origin_angstrom
                + (atoms.positions - virial.origin_angstrom) @ transform.T
            )
            energies.append(pes.get_potential_energy(displaced))
        finite_difference = (energies[0] - energies[1]) / (2.0 * step)
        strain_rows.append(
            {
                "step": step,
                "finite_difference_strain_gradient_eV": finite_difference,
                "absolute_error_eV": abs(finite_difference - analytic_strain_gradient),
            }
        )

    hvp_direction = _direction(20260817)
    hvp = pes.hessian_vector_product(atoms, hvp_direction, central_force=force)
    gates = {
        "source_charge": abs(state.source_total_charge_e) <= 2.0e-12,
        "net_force": float(np.max(np.abs(np.sum(force.total_forces_eV_per_A, axis=0))))
        <= 3.0e-10,
        "directional_force": min(
            float(row["absolute_error_eV_per_A"]) for row in force_rows
        )
        <= 2.0e-6,
        "molecular_virial": min(float(row["absolute_error_eV"]) for row in strain_rows)
        <= 3.0e-6,
        "hvp_error_estimate": hvp.maximum_error_estimate_eV_per_A2 <= 5.0e-3,
    }
    script = Path(__file__).resolve()
    return {
        "artifact_id": ARTIFACT_ID,
        "status": "pass" if all(gates.values()) else "fail",
        "claim_boundary": (
            "one-water derivative consistency only; not chemical accuracy, "
            "broad covariance, frequency, optimization, or MD admission"
        ),
        "configuration": {
            "device": args.device,
            "checkpoint_path_is_external": args.checkpoint is not None,
            "cds_grid_points": args.cds_grid_points,
            "cds_grid_status": (
                "low-cost derivative canary; not production accuracy identity"
            ),
            "model_configuration_sha256": model.configuration_sha256(),
            "model_provenance_sha256": model.provenance_sha256,
            "pes_configuration_sha256": pes.configuration_sha256(),
            "runner_sha256": _sha256(script),
        },
        "energy_state": {
            "state_sha256": state.state_sha256,
            "topology_id": state.topology_id,
            "source_total_charge_e": state.source_total_charge_e,
            "vacuum_energy_eV": state.vacuum_energy_eV,
            "polarization_energy_eV": state.polarization_energy_eV,
            "cds_energy_eV": state.cds_energy_eV,
            "solvation_energy_eV": state.solvation_energy_eV,
            "total_energy_eV": state.total_energy_eV,
        },
        "force": {
            "evaluation_sha256": force.evaluation_sha256,
            "norm_eV_per_A": float(np.linalg.norm(force.total_forces_eV_per_A)),
            "net_force_eV_per_A": np.sum(force.total_forces_eV_per_A, axis=0).tolist(),
            "direction": force_direction.tolist(),
            "analytic_gradient_eV_per_A": analytic_gradient,
            "finite_difference": force_rows,
        },
        "molecular_virial": {
            "evaluation_sha256": virial.evaluation_sha256,
            "origin_angstrom": virial.origin_angstrom.tolist(),
            "raw_virial_eV": virial.raw_virial_eV.tolist(),
            "symmetric_virial_eV": virial.symmetric_virial_eV.tolist(),
            "maximum_antisymmetry_eV": virial.maximum_antisymmetry_eV,
            "strain_direction": strain.tolist(),
            "analytic_strain_gradient_eV": analytic_strain_gradient,
            "finite_difference": strain_rows,
        },
        "hvp": {
            "evaluation_sha256": hvp.evaluation_sha256,
            "direction": hvp_direction.tolist(),
            "norm_eV_per_A2": float(np.linalg.norm(hvp.hvp_eV_per_A2)),
            "maximum_error_estimate_eV_per_A2": (hvp.maximum_error_estimate_eV_per_A2),
        },
        "gates": gates,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "ase": _version("ase"),
                "mace-torch": _version("mace-torch"),
                "numpy": _version("numpy"),
                "pyddx": _version("pyddx"),
                "torch": _version("torch"),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path(os.environ["ROUTE2_MACE_CHECKPOINT"])
            if "ROUTE2_MACE_CHECKPOINT" in os.environ
            else None
        ),
    )
    parser.add_argument("--cds-grid-points", type=int, default=DEFAULT_CDS_GRID_POINTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(args)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
