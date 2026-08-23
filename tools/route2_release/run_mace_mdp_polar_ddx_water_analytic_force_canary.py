#!/usr/bin/env python3
"""Run the real-checkpoint hybrid/ddX analytic-force water canary.

The artifact compares the complete block-implicit-adjoint force against
directional finite differences of the *same* operational scalar.  It also
checks the newly exposed MACE-MDP permanent-source coordinate VJP directly.
This is a derivative-consistency canary, not chemical-accuracy or public-force
admission evidence.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from ase import Atoms
import numpy as np

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.derivatives import RichardsonScalarForce
from maple.solvation.experimental import MACE_MDPPolarHybridDDXPES
from maple.solvation.models import (
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)

ARTIFACT_ID = "route2-mace-mdp-polar-ddx-water-analytic-force-canary-v1"
FORCE_STEPS_ANGSTROM = (2.0e-4, 1.0e-4, 5.0e-5)
MDP_VJP_STEPS_ANGSTROM = (2.0e-4, 1.0e-4, 5.0e-5)
SOURCE_FILES = (
    "maple/solvation/continuum/radial_gto_ddx.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/experimental/mace_mdp_polar_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar_separated.py",
)


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
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]]
        ),
        info={"charge": 0, "multiplicity": 1},
    )


def _direction(seed: int) -> np.ndarray:
    direction = np.random.default_rng(seed).normal(size=(3, 3))
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    return direction


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _configure_torch() -> object:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = __import__("torch")
    torch.manual_seed(20260816)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260816)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch


def run(args: argparse.Namespace) -> dict[str, object]:
    _configure_torch()
    repo = Path(__file__).resolve().parents[2]
    atoms = _water()
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=args.mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=args.polar_checkpoint,
        device=args.polar_device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
    )
    pes = MACE_MDPPolarHybridDDXPES(
        hybrid=hybrid,
        symbols=tuple(atoms.get_chemical_symbols()),
        cavity_radii_angstrom=smd_water_coulomb_radii(atoms.get_chemical_symbols()),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
        eta=0.1,
        n_proc=1,
        force_backend=RichardsonScalarForce(
            coarse_step_angstrom=5.0e-4,
            maximum_error_eV_per_A=2.0e-4,
        ),
    )
    state = pes.solve(atoms)
    force = pes.evaluate_forces(atoms, central_state=state)

    force_direction = _direction(20260816)
    analytic_gradient = -float(
        np.vdot(force.total_forces_ev_per_angstrom, force_direction)
    )
    force_rows = []
    for step in FORCE_STEPS_ANGSTROM:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * force_direction
        minus.positions -= step * force_direction
        finite_difference = (
            pes.sample(plus).energy_eV - pes.sample(minus).energy_eV
        ) / (2.0 * step)
        force_rows.append(
            {
                "step_angstrom": step,
                "finite_difference_gradient_eV_per_A": finite_difference,
                "absolute_error_eV_per_A": abs(finite_difference - analytic_gradient),
            }
        )

    source_cotangent = np.random.default_rng(20260817).normal(size=(3, 4))
    source_direction = _direction(20260818)
    source_gradient = mdp.source_position_vjp(atoms, source_cotangent)
    analytic_source_directional = float(np.vdot(source_gradient, source_direction))
    source_rows = []
    for step in MDP_VJP_STEPS_ANGSTROM:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * source_direction
        minus.positions -= step * source_direction
        plus_value = float(np.vdot(mdp.evaluate(plus).source4_raw_l1, source_cotangent))
        minus_value = float(
            np.vdot(mdp.evaluate(minus).source4_raw_l1, source_cotangent)
        )
        finite_difference = (plus_value - minus_value) / (2.0 * step)
        source_rows.append(
            {
                "step_angstrom": step,
                "finite_difference": finite_difference,
                "absolute_error": abs(finite_difference - analytic_source_directional),
            }
        )

    force_errors = [float(row["absolute_error_eV_per_A"]) for row in force_rows]
    source_errors = [float(row["absolute_error"]) for row in source_rows]
    net_force = np.sum(force.total_forces_ev_per_angstrom, axis=0)
    gates = {
        "root_residual": state.primal_residual_ev < 1.0e-10,
        "adjoint_residual": force.adjoint_residual_ev < 1.0e-9,
        "directional_force": min(force_errors) < 2.0e-6,
        "directional_force_converges": force_errors[-1] < force_errors[0],
        "mdp_source_position_vjp": min(source_errors) < 2.0e-7,
        "mdp_source_position_vjp_converges": source_errors[-1] < source_errors[0],
        "translation_covariance": float(np.max(np.abs(net_force))) < 2.0e-8,
        "fixed_total_charge": abs(float(np.sum(state.total_source4[:, 0]))) < 2.0e-10,
    }
    script = Path(__file__).resolve()
    head = _git(repo, "rev-parse", "HEAD")
    dirty = bool(_git(repo, "status", "--porcelain=v1"))
    return {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "claim_boundary": (
            "one-water real-checkpoint derivative consistency only; not a "
            "chemical-accuracy panel, global topology/SO(3) guarantee, public "
            "force admission, Hessian/FREQ/MD admission, or Tier V"
        ),
        "execution": {
            "git_head": head,
            "worktree_dirty": dirty,
            "runner_sha256": _sha256(script),
            "source_files_sha256": {
                name: _sha256(repo / name) for name in SOURCE_FILES
            },
        },
        "configuration": {
            "pes_configuration_sha256": pes.configuration_sha256(),
            "hybrid_configuration_sha256": hybrid.configuration_sha256(),
            "hybrid_provenance_sha256": hybrid.provenance_sha256,
            "mdp_checkpoint_sha256": mdp.checkpoint_sha256,
            "polar_checkpoint_sha256": radial.provenance.checkpoint_sha256,
            "polar_device": args.polar_device,
            "continuum": "ddPCM",
            "dielectric": 78.39,
            "lmax": 8,
            "n_lebedev": 194,
            "energy_ledger": "vacuum MACE-POLAR plus ddX polarization",
            "permanent_source": "MACE-MDP point q/p",
            "induced_source": "MACE-POLAR 1.5-A Gaussian q/p increment",
            "model_drive": "external-MEP 1.5/3.0-A phi-side adjoint",
        },
        "state": {
            "root_sha256": state.root_sha256,
            "continuum_state_sha256": state.continuum_state_sha256,
            "energy_eV": state.total_energy_ev,
            "polarization_energy_eV": state.polarization_energy_ev,
            "root_residual_eV": state.primal_residual_ev,
            "cold_iterations": state.cold_iterations,
            "wide_iterations": state.wide_iterations,
            "replay_field_max_abs_difference_eV": (
                state.replay_field_max_abs_difference_ev
            ),
            "replay_energy_abs_difference_eV": (state.replay_energy_abs_difference_ev),
        },
        "analytic_force": {
            "evaluation_sha256": force.evaluation_sha256,
            "adjoint_residual_eV": force.adjoint_residual_ev,
            "adjoint_iterations": force.adjoint_iterations,
            "norm_eV_per_A": float(np.linalg.norm(force.total_forces_ev_per_angstrom)),
            "net_force_eV_per_A": net_force.tolist(),
            "direction": force_direction.tolist(),
            "analytic_gradient_eV_per_A": analytic_gradient,
            "finite_difference": force_rows,
        },
        "mace_mdp_source_position_vjp": {
            "direction": source_direction.tolist(),
            "analytic_directional": analytic_source_directional,
            "translation_sum": np.sum(source_gradient, axis=0).tolist(),
            "finite_difference": source_rows,
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
                "scipy": _version("scipy"),
                "torch": _version("torch"),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mdp-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-MDP.model",
    )
    parser.add_argument(
        "--polar-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACEPOLAR1Mmodel",
    )
    parser.add_argument(
        "--polar-device",
        default=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cuda"),
    )
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
