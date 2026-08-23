#!/usr/bin/env python3
"""Run the real-checkpoint MDP+POLAR harmonic-ddPCM force canary.

This runner binds the heterogeneous source semantics used by the hybrid:

* MACE-MDP supplies permanent exterior point monopoles/dipoles;
* MACE-POLAR supplies the field-induced 1.5-A Gaussian source increment;
* the ddPCM adjoint supplies the checkpoint-native 1.5/3.0-A field;
* Phi0 is the frozen MACE-POLAR vacuum energy plus the stationary continuum
  polarization energy.

Every displaced geometry rebuilds both ML source anchors and the continuum,
then solves a cold root.  The output is a one-water derivative canary, not a
public capability admission or a Tier-V claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from ase import Atoms
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.solvation.api.profiles import (  # noqa: E402
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.continuum.harmonic_ddpcm_functional import (  # noqa: E402
    SmoothPartitionHarmonicDDPCMFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_ddpcm_hybrid import (  # noqa: E402
    HARMONIC_DDPCM_HYBRID_CONTRACT_ID,
    build_harmonic_ddpcm_hybrid_snapshot,
)
from maple.solvation.coupling.fixed_point import FixedPointOptions  # noqa: E402
from maple.solvation.coupling.permanent_induced_ledgers import (  # noqa: E402
    HybridHarmonicDDPCMPhi0Ledger,
)
from maple.solvation.coupling.permanent_induced_state import (  # noqa: E402
    PermanentInducedOperationalStateEquation,
)
from maple.solvation.coupling.separated_fixed_point import (  # noqa: E402
    separated_roots_numerically_equivalent,
    solve_separated_fixed_point,
)
from maple.solvation.coupling.spaces import (  # noqa: E402
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.models import (  # noqa: E402
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release.root_well_posedness import (  # noqa: E402
    certify_root_well_posedness,
    dense_state_map_jacobian,
)


SCHEMA_VERSION = "route2-mace-mdp-polar-harmonic-ddpcm-general-source-water-v2"
DEFAULT_MDP_CHECKPOINT = Path.home() / ".cache/mace/MACE-MDP.model"
DEFAULT_POLAR_CHECKPOINT = Path.home() / ".cache/mace/MACEPOLAR1Mmodel"
DEFAULT_OUTPUT = (
    Path(".omx/route2")
    / "mace-mdp-polar-harmonic-ddpcm-general-source-water-v2.json"
)
WATER_RADII_ANGSTROM = (1.52, 1.2, 1.2)
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
SOURCE_FILES = (
    "maple/solvation/continuum/harmonic_ddpcm_functional.py",
    "maple/solvation/continuum/harmonic_ddpcm_hybrid.py",
    "maple/solvation/continuum/harmonic_ddpcm_primitives.py",
    "maple/solvation/continuum/harmonic_schwarz_primitives.py",
    "maple/solvation/coupling/permanent_induced_ledgers.py",
    "maple/solvation/coupling/permanent_induced_state.py",
    "maple/solvation/coupling/separated_fixed_point.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar_separated.py",
    "tools/route2_release/run_mace_mdp_polar_harmonic_ddpcm_water_canary.py",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp-checkpoint", type=Path, default=DEFAULT_MDP_CHECKPOINT)
    parser.add_argument(
        "--polar-checkpoint", type=Path, default=DEFAULT_POLAR_CHECKPOINT
    )
    parser.add_argument("--polar-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surface-lmax", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1.0e-11)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--wide-start-scale", type=float, default=0.1)
    parser.add_argument("--force-fd-step", type=float, default=2.0e-4)
    parser.add_argument("--force-fd-levels", type=int, default=3)
    parser.add_argument("--force-fd-tolerance", type=float, default=2.0e-5)
    return parser.parse_args()


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]]
        ),
        info={"charge": 0, "multiplicity": 1},
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _state_record(state) -> dict[str, object]:
    return {
        "root_hash": state.root_hash,
        "initialization": state.initialization,
        "converged": state.converged,
        "iterations": len(state.iterations) - 1,
        "actual_unmixed_residual_norm": state.actual_unmixed_residual_norm,
        "induced_total_charge_e": float(np.sum(state.source_array()[:, 0])),
        "induced_source_norm": float(np.linalg.norm(state.source_array())),
        "field_norm": float(np.linalg.norm(state.field_array())),
        "boundary_state_norm": float(np.linalg.norm(state.boundary_state_array())),
    }


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[2]
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser()
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    torch.manual_seed(20260816)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260816)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.polar_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for MACE-POLAR but is unavailable.")

    if int(args.force_fd_levels) < 2:
        raise ValueError("force-fd-levels must be at least two.")
    for name, value in (
        ("force-fd-step", args.force_fd_step),
        ("force-fd-tolerance", args.force_fd_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    started = time.perf_counter()
    atoms = _water()
    mdp_started = time.perf_counter()
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    mdp_load_seconds = time.perf_counter() - mdp_started
    polar_started = time.perf_counter()
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.polar_device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    response = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=response,
    )
    polar_load_seconds = time.perf_counter() - polar_started

    maximum = int(args.surface_lmax)
    functional = SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8, 1, 1),
        radii_angstrom=WATER_RADII_ANGSTROM,
        transition_width_angstrom2=0.08,
        surface_lmax=maximum,
        partition_lmax=2 * maximum,
        partition_radial_quadrature_order=64,
        source_radial_quadrature_order=80,
        double_layer_radial_quadrature_order=80,
        dielectric=80.0,
        dtype=torch.float64,
        device="cpu",
    )
    coordinates = AffineChargeCoordinates(
        atom_count=len(atoms),
        total_charge=0.0,
        source_space=ATOMIC_L1_SOURCE_SPACE,
    )
    options = FixedPointOptions(
        method="anderson",
        tolerance=float(args.tolerance),
        max_iterations=int(args.max_iterations),
        damping=float(args.damping),
        history=int(args.history),
    )
    context = "water-macedp-polar-harmonic-ddpcm-general-source-v2"

    anchor_seconds = 0.0
    continuum_assembly_seconds = 0.0
    equation_assembly_seconds = 0.0
    root_seconds = 0.0

    def build_equation(geometry: Atoms) -> PermanentInducedOperationalStateEquation:
        nonlocal anchor_seconds, continuum_assembly_seconds, equation_assembly_seconds
        tick = time.perf_counter()
        anchor = hybrid.prepare(geometry)
        anchor_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        continuum = build_harmonic_ddpcm_hybrid_snapshot(
            functional,
            geometry,
            receiver_radial_quadrature_order=80,
        )
        continuum_assembly_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        result = PermanentInducedOperationalStateEquation(
            coordinates,
            hybrid,
            anchor,
            continuum,
        )
        equation_assembly_seconds += time.perf_counter() - tick
        return result

    equation = build_equation(atoms)

    def resolved_energy(displaced: Atoms, *, label: str) -> tuple[float, object]:
        nonlocal root_seconds
        displaced_equation = build_equation(displaced)
        tick = time.perf_counter()
        displaced_state = solve_separated_fixed_point(
            displaced_equation,
            displaced,
            root_context_id=f"{context}-force-fd-{label}",
            options=options,
        )
        root_seconds += time.perf_counter() - tick
        displaced_ledger = HybridHarmonicDDPCMPhi0Ledger(displaced_equation)
        displaced_evaluation = displaced_ledger.evaluate_root(
            displaced,
            displaced_state.y_array(),
            root_tolerance=options.tolerance,
        )
        return displaced_evaluation.total_energy_eV, displaced_state

    root_started = time.perf_counter()
    cold = solve_separated_fixed_point(
        equation,
        atoms,
        root_context_id=context,
        options=options,
    )
    generator = np.random.default_rng(20260816)
    wide_initial = cold.y_array() + float(args.wide_start_scale) * generator.normal(
        size=cold.y_array().shape
    )
    wide = solve_separated_fixed_point(
        equation,
        atoms,
        root_context_id=context,
        initial_y=wide_initial,
        options=options,
    )
    central_root_seconds = time.perf_counter() - root_started
    root_seconds += central_root_seconds
    roots_equivalent = separated_roots_numerically_equivalent(cold, wide)
    root_delta = float(np.linalg.norm(cold.y_array() - wide.y_array()))

    y = cold.y_array()
    reduced_direction = generator.normal(size=y.shape)
    reduced_cotangent = generator.normal(size=y.shape)
    residual_jvp = equation.residual_jvp(atoms, y, reduced_direction)
    residual_vjp = equation.residual_vjp(atoms, y, reduced_cotangent)
    adjoint_defect = abs(
        float(np.vdot(reduced_cotangent, residual_jvp))
        - float(np.vdot(residual_vjp, reduced_direction))
    )
    identity = np.eye(equation.reduced_dimension)
    state_map_jacobian = dense_state_map_jacobian(
        lambda vector: vector - equation.residual_jvp(atoms, y, vector),
        dimension=equation.reduced_dimension,
    )
    residual_jacobian = identity - state_map_jacobian
    direct_residual_jacobian = np.column_stack(
        [
            equation.residual_jvp(atoms, y, identity[:, index])
            for index in range(equation.reduced_dimension)
        ]
    )
    residual_jacobian_replay_error = float(
        np.max(np.abs(residual_jacobian - direct_residual_jacobian))
    )
    certificate = certify_root_well_posedness(state_map_jacobian)

    ledger = HybridHarmonicDDPCMPhi0Ledger(equation)
    ledger_evaluation = ledger.evaluate_root(
        atoms,
        y,
        root_tolerance=options.tolerance,
    )
    gradient_started = time.perf_counter()
    operational_gradient = ledger.implicit_gradient(atoms, cold)
    gradient_seconds = time.perf_counter() - gradient_started

    coordinate_direction = generator.normal(size=(len(atoms), 3))
    coordinate_direction -= np.mean(coordinate_direction, axis=0, keepdims=True)
    coordinate_direction /= np.linalg.norm(coordinate_direction)
    analytic_directional_derivative = float(
        np.vdot(operational_gradient.gradient_array(), coordinate_direction)
    )
    force_fd_started = time.perf_counter()
    force_fd_records: list[dict[str, object]] = []
    for level in range(int(args.force_fd_levels)):
        step = float(args.force_fd_step) / (2.0**level)
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * coordinate_direction
        minus.positions -= step * coordinate_direction
        plus_energy, plus_state = resolved_energy(plus, label=f"plus-{level}")
        minus_energy, minus_state = resolved_energy(minus, label=f"minus-{level}")
        finite_difference = (plus_energy - minus_energy) / (2.0 * step)
        force_fd_records.append(
            {
                "level": level,
                "step_angstrom": step,
                "plus_energy_eV": plus_energy,
                "minus_energy_eV": minus_energy,
                "central_difference_eV_per_angstrom": finite_difference,
                "analytic_eV_per_angstrom": analytic_directional_derivative,
                "absolute_error_eV_per_angstrom": abs(
                    finite_difference - analytic_directional_derivative
                ),
                "plus_root_residual": plus_state.actual_unmixed_residual_norm,
                "minus_root_residual": minus_state.actual_unmixed_residual_norm,
                "plus_iterations": len(plus_state.iterations) - 1,
                "minus_iterations": len(minus_state.iterations) - 1,
            }
        )
    force_fd_seconds = time.perf_counter() - force_fd_started
    finest_force_error = float(
        force_fd_records[-1]["absolute_error_eV_per_angstrom"]
    )

    continuum = equation.continuum
    source_hashes = {
        relative: _sha256_file(root / relative) for relative in SOURCE_FILES
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass"
        if (
            roots_equivalent
            and cold.converged
            and wide.converged
            and certificate.local_implicit_branch_certified
            and adjoint_defect < 2.0e-8
            and residual_jacobian_replay_error < 2.0e-12
            and operational_gradient.adjoint.converged
            and finest_force_error < float(args.force_fd_tolerance)
        )
        else "fail",
        "git": {
            "head": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "dirty": bool(_git(root, "status", "--porcelain=v1")),
        },
        "checkpoints": {
            "mace_mdp": {
                "path": str(mdp_checkpoint),
                "sha256": mdp.checkpoint_sha256,
                "device": "cpu",
                "dtype": "float64",
                "role": "geometry-only permanent point q/p",
            },
            "mace_polar": {
                "path": str(polar_checkpoint),
                "sha256": response.checkpoint_sha256,
                "device": radial.provenance.device,
                "role": "vacuum PES plus field-induced Gaussian q/p",
            },
        },
        "hybrid": {
            "provider_id": hybrid.provider_id,
            "model_profile_id": hybrid.model_profile_id,
            "configuration_sha256": hybrid.configuration_sha256(),
            "provenance_sha256": hybrid.provenance_sha256,
            "permanent_source": hybrid.permanent_source_kernel,
            "induced_source": hybrid.induced_source_kernel,
            "permanent_total_charge_e": float(
                np.sum(equation.permanent_source[:, 0])
            ),
        },
        "continuum": {
            "contract_id": HARMONIC_DDPCM_HYBRID_CONTRACT_ID,
            "configuration_sha256": continuum.configuration_sha256(),
            "provenance_sha256": continuum.provenance_sha256,
            "surface_lmax": maximum,
            "stored_boundary_dimension": continuum.boundary_dimension,
            "forward_boundary_dimension": continuum.forward_boundary_dimension,
            "raw_general_source_asymmetry_relative_defect": (
                continuum.general_source_raw_asymmetry_relative_defect
            ),
            "radii_angstrom": list(WATER_RADII_ANGSTROM),
            "model": "finite-dielectric smooth-partition harmonic ddPCM",
        },
        "state_equation": {
            "id": equation.state_equation_id,
            "fingerprint_sha256": equation.fingerprint_sha256(),
            "permanent_source_shape": [len(atoms), 4],
            "induced_source_shape": [len(atoms), 4],
            "native_field_shape": [len(atoms), 8],
            "residual_jvp_vjp_absolute_defect": adjoint_defect,
            "residual_jacobian_replay_max_abs_error": (
                residual_jacobian_replay_error
            ),
        },
        "root_options": {
            "method": options.method,
            "tolerance": options.tolerance,
            "max_iterations": options.max_iterations,
            "damping": options.damping,
            "history": options.history,
            "wide_start_scale": float(args.wide_start_scale),
        },
        "cold_root": _state_record(cold),
        "wide_root": _state_record(wide),
        "cold_wide": {
            "numerically_equivalent": roots_equivalent,
            "reduced_root_l2_difference": root_delta,
        },
        "local_root_certificate": certificate.as_dict(),
        "energy_ledger_diagnostic": {
            **ledger_evaluation.as_dict(),
            "admission": "disabled-research-diagnostic",
        },
        "operational_implicit_gradient": {
            "direct_coordinate_gradient_eV_per_angstrom": (
                operational_gradient.direct_coordinate_gradient_eV_per_angstrom
            ),
            "residual_coordinate_pullback_eV_per_angstrom": (
                operational_gradient.residual_coordinate_pullback_eV_per_angstrom
            ),
            "total_coordinate_gradient_eV_per_angstrom": (
                operational_gradient.total_coordinate_gradient_eV_per_angstrom
            ),
            "forces_eV_per_angstrom": operational_gradient.forces_array().tolist(),
            "translation_gradient_sum_eV_per_angstrom": np.sum(
                operational_gradient.gradient_array(), axis=0
            ).tolist(),
            "adjoint_iterations": operational_gradient.adjoint.iterations,
            "adjoint_gmres_info": operational_gradient.adjoint.gmres_info,
            "adjoint_true_residual_norm": (
                operational_gradient.adjoint.true_residual_norm
            ),
            "adjoint_acceptance_tolerance": (
                operational_gradient.adjoint.acceptance_tolerance
            ),
            "adjoint_converged": operational_gradient.adjoint.converged,
            "claim_boundary": (
                "Complete first coordinate derivative of the registered Phi0 "
                "ledger for this bound root, including MACE-POLAR vacuum, "
                "moving harmonic-ddPCM, MACE-MDP permanent-source, native-field, "
                "MACE-POLAR induced-source, and implicit-state terms."
            ),
        },
        "resolved_force_directional_finite_difference": {
            "direction": coordinate_direction.tolist(),
            "analytic_eV_per_angstrom": analytic_directional_derivative,
            "levels": force_fd_records,
            "finest_absolute_error_eV_per_angstrom": finest_force_error,
            "acceptance_tolerance_eV_per_angstrom": float(
                args.force_fd_tolerance
            ),
            "each_displacement_rebuilds_mdp_and_polar_anchor": True,
            "each_displacement_rebuilds_continuum": True,
            "each_displacement_uses_cold_root": True,
        },
        "capabilities": NO_CAPABILITIES,
        "timing_seconds": {
            "mace_mdp_load_cpu_float64": mdp_load_seconds,
            "mace_polar_load": polar_load_seconds,
            "all_mdp_plus_polar_anchor_evaluations": anchor_seconds,
            "all_continuum_assembly": continuum_assembly_seconds,
            "all_state_equation_assembly": equation_assembly_seconds,
            "central_two_root_solves": central_root_seconds,
            "all_root_solves_including_fd": root_seconds,
            "complete_implicit_gradient": gradient_seconds,
            "resolved_force_finite_difference": force_fd_seconds,
            "total": time.perf_counter() - started,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
        },
        "source_files_sha256": source_hashes,
        "claim_boundary": (
            "This proves one real-checkpoint MACE-MDP permanent plus MACE-POLAR "
            "induced general-source root, cold/wide replay, local nonsingularity, "
            "implemented JVP/VJP transpose consistency, and one resolved "
            "non-rigid directional finite-difference check of the complete Phi0 "
            "gradient. It does not prove chemical accuracy, a global unique root, "
            "distorted-domain force admission, public E/F/H/V/M, or Tier V."
        ),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
