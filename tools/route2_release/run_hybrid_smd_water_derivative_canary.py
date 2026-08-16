#!/usr/bin/env python3
"""Audit the frozen hybrid SMD scalar and its first derivative on water.

This deliberately reuses the exact scientific profile of
``route2-hybrid-smd-development-prereg-v2``:

* MACE-MDP permanent point ``q/p``;
* zero-anchored MACE-POLAR induced Gaussian ``q/p``;
* ddX PCM with ``lmax=15`` and 1202 Lebedev points;
* the official PySCF SMD-CDS term;
* CPU execution for both frozen checkpoints.

The artifact checks cold replay, a multi-step directional derivative, rigid
translation, and rigid rotation.  It is intentionally a one-water diagnostic;
even a passing artifact is not a distorted-PES panel or public force admission.
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

from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.experimental import build_smd_mace_mdp_polar_hybrid_ddx_pes
from maple.solvation.models import (
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)


ARTIFACT_ID = "route2-hybrid-smd-water-derivative-canary-v1"
FORCE_STEPS_ANGSTROM = (4.0e-4, 2.0e-4, 1.0e-4)
SOURCE_FILES = (
    "maple/function/calculator/extra_correction/implicit/pyscf_smd_cds.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/experimental/mace_mdp_polar_ddx.py",
    "maple/solvation/experimental/mace_mdp_polar_solvated_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/solvent_terms.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _configure_torch() -> object:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = __import__("torch")
    torch.manual_seed(20260816)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return torch


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]],
            dtype=float,
        ),
        info={"charge": 0, "multiplicity": 1},
    )


def _direction(seed: int, atom_count: int) -> np.ndarray:
    direction = np.random.default_rng(seed).normal(size=(atom_count, 3))
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    return direction


def _rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    q, _ = np.linalg.qr(matrix)
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / (1.0 + np.linalg.norm(right)))


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
        device="cpu",
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
    )
    pes = build_smd_mace_mdp_polar_hybrid_ddx_pes(
        hybrid,
        atoms.get_chemical_symbols(),
        solvent="water",
        continuum_model="pcm",
        lmax=15,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
        eta=0.1,
        n_proc=1,
    )

    first = pes.solve(atoms)
    second = pes.solve(atoms)
    force = pes.evaluate_forces(atoms, central_state=first)
    central_forces = force.total_forces_eV_per_A

    direction = _direction(20260816, len(atoms))
    analytic_gradient = -float(np.vdot(central_forces, direction))
    finite_difference_rows: list[dict[str, float]] = []
    for step in FORCE_STEPS_ANGSTROM:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        finite_difference = (
            pes.get_potential_energy(plus) - pes.get_potential_energy(minus)
        ) / (2.0 * step)
        finite_difference_rows.append(
            {
                "step_angstrom": step,
                "finite_difference_gradient_eV_per_A": finite_difference,
                "absolute_error_eV_per_A": abs(finite_difference - analytic_gradient),
            }
        )

    translation = np.asarray([0.37, -0.29, 0.41], dtype=float)
    translated = atoms.copy()
    translated.positions += translation
    translated_state = pes.solve(translated)
    translated_force = pes.evaluate_forces(
        translated, central_state=translated_state
    ).total_forces_eV_per_A

    rotation = _rotation(20260817)
    center = np.mean(atoms.positions, axis=0)
    rotated = atoms.copy()
    rotated.positions = center + (atoms.positions - center) @ rotation.T
    rotated_state = pes.solve(rotated)
    rotated_force = pes.evaluate_forces(
        rotated, central_state=rotated_state
    ).total_forces_eV_per_A
    expected_rotated_force = central_forces @ rotation.T

    force_errors = tuple(
        float(row["absolute_error_eV_per_A"])
        for row in finite_difference_rows
    )
    translation_energy_error = abs(
        translated_state.total_energy_eV - first.total_energy_eV
    )
    translation_force_error = float(
        np.max(np.abs(translated_force - central_forces))
    )
    rotation_energy_error = abs(rotated_state.total_energy_eV - first.total_energy_eV)
    rotation_force_error = _relative_error(rotated_force, expected_rotated_force)
    net_force = np.sum(central_forces, axis=0)
    gates = {
        "cold_replay": first.root_sha256 == second.root_sha256,
        "root_residual": first.electrostatic_state.primal_residual_ev < 1.0e-10,
        "adjoint_residual": (
            force.electrostatic_evaluation.adjoint_residual_ev < 1.0e-9
        ),
        "directional_force": min(force_errors) <= 2.0e-4,
        "directional_force_converges_or_is_at_roundoff": (
            force_errors[-1] <= max(force_errors[0], 1.0e-10)
        ),
        "translation_energy": translation_energy_error <= 1.0e-8,
        "translation_force": translation_force_error <= 1.0e-6,
        "net_force": float(np.max(np.abs(net_force))) <= 1.0e-6,
        "rotation_energy": rotation_energy_error <= 2.0e-5,
        "rotation_force": rotation_force_error <= 2.0e-4,
    }

    script = Path(__file__).resolve()
    return {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "claim_boundary": (
            "one-water frozen-profile E/F derivative and rigid-motion diagnostic; "
            "not the 505 accuracy decision, distorted-PES or closed-loop evidence, "
            "public force/Hessian/FREQ/OPT/MD admission, or strict Tier V"
        ),
        "execution": {
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "worktree_dirty": bool(_git(repo, "status", "--porcelain=v1")),
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
            "device": "cpu",
            "solvent": "water",
            "continuum_model": "pcm",
            "lmax": 15,
            "n_lebedev": 1202,
            "solver_tolerance": 1.0e-12,
            "eta": 0.1,
            "energy_ledger": "vacuum + ddX polarization + PySCF SMD-CDS",
        },
        "state": {
            "root_sha256": first.root_sha256,
            "cold_replay_root_sha256": second.root_sha256,
            "energy_eV": first.total_energy_eV,
            "solvation_energy_eV": first.solvation_energy_eV,
            "polarization_energy_eV": first.polarization_energy_eV,
            "cds_energy_eV": first.cds_energy_eV,
            "root_residual_eV": first.electrostatic_state.primal_residual_ev,
        },
        "force": {
            "evaluation_sha256": force.evaluation_sha256,
            "adjoint_residual_eV": (
                force.electrostatic_evaluation.adjoint_residual_ev
            ),
            "norm_eV_per_A": float(np.linalg.norm(central_forces)),
            "net_force_eV_per_A": net_force.tolist(),
            "direction": direction.tolist(),
            "analytic_gradient_eV_per_A": analytic_gradient,
            "finite_difference": finite_difference_rows,
        },
        "rigid_translation": {
            "translation_angstrom": translation.tolist(),
            "energy_absolute_error_eV": translation_energy_error,
            "force_max_absolute_error_eV_per_A": translation_force_error,
        },
        "rigid_rotation": {
            "rotation_matrix": rotation.tolist(),
            "energy_absolute_error_eV": rotation_energy_error,
            "force_relative_error": rotation_force_error,
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
                "pyscf": _version("pyscf"),
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
