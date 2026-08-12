#!/usr/bin/env python3
"""Run the disabled local-jet/fixed-cavity vNext diagnostic.

This command exists only to reproduce migration-parity measurements.  The
profile it uses has no E/F/H/V/M admission and cannot publish a Route-2 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from ase import Atoms
import numpy as np

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import (
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
)
from maple.solvation.api.profiles import (
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
)
from maple.solvation.continuum import FixedTopologyCPCMBackend
from maple.solvation.coupling.energy import OperationalElectrostaticScalar
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.spaces import AffineChargeCoordinates
from maple.solvation.coupling.state_equation import (
    ReducedStateEquation,
    geometry_sha256,
)
from maple.solvation.models.equation_adapter import (
    ElectronicResponseEquationAdapter,
    VacuumScalarEquationAdapter,
)
from maple.solvation.models.mace_polar import (
    build_official_mace_polar_1_m_adapter,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", required=True, type=Path)
    parser.add_argument("--atomic-numbers", required=True, nargs="+", type=int)
    parser.add_argument("--compound-id", default="unspecified")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dielectric", default=78.39, type=float)
    parser.add_argument("--lebedev-order", default=15, type=int)
    parser.add_argument("--total-charge", default=0.0, type=float)
    parser.add_argument("--tolerance", default=1.0e-11, type=float)
    parser.add_argument("--max-iterations", default=80, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = args.source_npz.expanduser().resolve(strict=True)
    with np.load(source_path) as archive:
        positions = np.asarray(archive["positions_angstrom"], dtype=float)
    numbers = np.asarray(args.atomic_numbers, dtype=int)
    if positions.shape != (len(numbers), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("source NPZ positions and --atomic-numbers do not match.")

    profile = get_solvation_profile(DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1)
    if profile.enabled or profile.capabilities.enabled_tiers:
        raise RuntimeError("The local-jet diagnostic profile must remain disabled.")

    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.info.update(charge=int(args.total_charge), mult=1)
    started = time.perf_counter()
    model = build_official_mace_polar_1_m_adapter(device=args.device)
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent="water",
        profile=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
    )
    continuum = FixedTopologyCPCMBackend(
        atoms.get_chemical_symbols(),
        radii,
        dielectric=args.dielectric,
        lebedev_order=args.lebedev_order,
    )
    electronic = ElectronicResponseEquationAdapter(
        model, coupling_id=continuum.coupling_id
    )
    vacuum = VacuumScalarEquationAdapter(model)
    coordinates = AffineChargeCoordinates(
        len(atoms),
        total_charge=args.total_charge,
        monopole_scale=1.0,
        dipole_scale=1.0,
    )
    equation = ReducedStateEquation(coordinates, electronic, continuum)
    scalar = OperationalElectrostaticScalar(
        equation,
        vacuum,
        scalar_id=DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
        profile_id=DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1,
    )
    state = solve_fixed_point(
        equation,
        atoms,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id=(
            f"localjet-fixed-cavity/{args.compound_id}/{geometry_sha256(atoms)}"
        ),
        options=FixedPointOptions(
            method="anderson",
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            damping=1.0,
            history=6,
        ),
    )
    source = state.source_array()
    continuum_state = continuum.build_state(atoms, source)
    vacuum_state = model.evaluate_vacuum(atoms, need_forces=False)

    replay = FixedTopologyCPCMBackend(
        atoms.get_chemical_symbols(),
        radii,
        dielectric=args.dielectric,
        lebedev_order=args.lebedev_order,
    )
    replay_field = replay.field(atoms, source)
    replay_source = model.evaluate_source(
        atoms, replay_field, need_fixed_field_forces=False
    ).source
    status = _git("status", "--porcelain=v1")
    payload = {
        "schema_version": 1,
        "status": "diagnostic-success",
        "claim_boundary": (
            "disabled local-jet diagnostic; not an admitted scalar, force, or "
            "complete solvation free energy"
        ),
        "execution_git_head": _git("rev-parse", "HEAD"),
        "working_tree_clean": status == "",
        "source_npz_sha256": _sha256(source_path),
        "compound_id": args.compound_id,
        "profile_id": scalar.profile_id,
        "scalar_id": scalar.scalar_id,
        "coupling_id": continuum.coupling_id,
        "checkpoint_sha256": model.provenance.checkpoint_sha256,
        "geometry_sha256": geometry_sha256(atoms),
        "dielectric": args.dielectric,
        "lebedev_order": args.lebedev_order,
        "iterations": len(state.iterations) - 1,
        "residual_norm": state.actual_unmixed_residual_norm,
        "root_hash": state.root_hash,
        "source_total_charge_e": float(np.sum(source[:, 0])),
        "vacuum_energy_eV": vacuum_state.energy_eV,
        "continuum_energy_eV": continuum_state.polarization_energy_ev,
        "continuum_energy_hartree": continuum_state.polarization_energy_hartree,
        "continuum_energy_kcal_mol": (
            continuum_state.polarization_energy_hartree * HARTREE_TO_KCAL_MOL
        ),
        "operational_total_energy_eV": (
            vacuum_state.energy_eV + continuum_state.polarization_energy_ev
        ),
        "cold_replay_field_max_abs_eV": float(
            np.max(np.abs(replay_field - state.field_array()))
        ),
        "cold_replay_response_max_abs": float(np.max(np.abs(replay_source - source))),
        "half_coupling_error_eV": abs(
            continuum_state.polarization_energy_ev
            - 0.5 * continuum.pairing.pair(source, continuum_state.reaction_field)
        ),
        "surface_size": continuum_state.surface.candidate_count,
        "topology_hash": continuum_state.surface.topology_hash,
        "runtime_seconds": time.perf_counter() - started,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print("ROUTE2_VNEXT_LOCALJET_DIAGNOSTIC=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
