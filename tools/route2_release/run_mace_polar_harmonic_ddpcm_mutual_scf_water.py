#!/usr/bin/env python3
"""Run the first real-checkpoint 4-source/8-field harmonic-ddPCM SCF canary.

This is a research evidence runner, not a capability admission job.  It binds
one official MACE-POLAR checkpoint, the analytic Gaussian-multipole evaluator,
the point-l<=1 harmonic ddPCM source, and the checkpoint-native two-width
Gaussian reaction-field receiver.  Cold and deliberately wide starts must
reach the same locally nonsingular operational root.
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
from maple.solvation.continuum.harmonic_ddpcm_separated import (  # noqa: E402
    build_mace_polar_point_harmonic_ddpcm_separated_snapshot,
)
from maple.solvation.coupling.fixed_point import FixedPointOptions  # noqa: E402
from maple.solvation.coupling.separated_ledgers import (  # noqa: E402
    HarmonicDDPCMFrozenVacuumLedger,
)
from maple.solvation.coupling.separated_fixed_point import (  # noqa: E402
    separated_roots_numerically_equivalent,
    solve_separated_fixed_point,
)
from maple.solvation.coupling.separated_state import (  # noqa: E402
    SeparatedOperationalStateEquation,
)
from maple.solvation.coupling.spaces import (  # noqa: E402
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.models import (  # noqa: E402
    MACEPolarOriginalSourceNativeFieldAdapter,
    NativeSemanticsCanary,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release.root_well_posedness import (  # noqa: E402
    certify_root_well_posedness,
    dense_state_map_jacobian,
)
from maple.solvation.release.field_semantics import (  # noqa: E402
    build_unverified_mace_polar_field_semantics_manifest,
)


SCHEMA_VERSION = "route2-macepolar-harmonic-ddpcm-mutual-scf-water-v1"
DEFAULT_CHECKPOINT = Path.home() / ".cache/mace/MACEPOLAR1Mmodel"
DEFAULT_OUTPUT = (
    Path(".omx/route2") / "mace-polar-harmonic-ddpcm-mutual-scf-water-v1.json"
)
WATER_RADII_ANGSTROM = (2.294, 1.2, 1.2)
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
SOURCE_FILES = (
    "maple/solvation/continuum/harmonic_ddpcm_functional.py",
    "maple/solvation/continuum/harmonic_ddpcm_separated.py",
    "maple/solvation/continuum/harmonic_gaussian_source.py",
    "maple/solvation/continuum/harmonic_single_layer.py",
    "maple/solvation/coupling/fixed_point.py",
    "maple/solvation/coupling/separated_fixed_point.py",
    "maple/solvation/coupling/separated_ledgers.py",
    "maple/solvation/coupling/linearization.py",
    "maple/solvation/coupling/adjoint.py",
    "maple/solvation/coupling/separated_operators.py",
    "maple/solvation/coupling/separated_state.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "tools/route2_release/run_mace_polar_harmonic_ddpcm_mutual_scf_water.py",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surface-lmax", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1.0e-11)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--wide-start-scale", type=float, default=0.25)
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
        info={"charge": 0, "mult": 1},
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
        "source_total_charge_e": float(np.sum(state.source_array()[:, 0])),
        "source_norm": float(np.linalg.norm(state.source_array())),
        "field_norm": float(np.linalg.norm(state.field_array())),
        "boundary_state_norm": float(np.linalg.norm(state.boundary_state_array())),
    }


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser()
    output = output if output.is_absolute() else (root / output)
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    torch.manual_seed(20260816)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260816)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    started = time.perf_counter()
    atoms = _water()
    model_started = time.perf_counter()
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        device=args.device,
        checkpoint_path=checkpoint,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    electronic = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    native_semantics = NativeSemanticsCanary.from_adapter(radial)
    field_semantics = build_unverified_mace_polar_field_semantics_manifest(
        native_canary=native_semantics,
        adapter=electronic,
    )
    model_seconds = time.perf_counter() - model_started

    continuum_started = time.perf_counter()
    maximum = int(args.surface_lmax)
    functional = SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8, 1, 1),
        radii_angstrom=WATER_RADII_ANGSTROM,
        transition_width_angstrom2=0.08,
        surface_lmax=maximum,
        partition_lmax=2 * maximum,
        partition_radial_quadrature_order=96,
        source_radial_quadrature_order=128,
        double_layer_radial_quadrature_order=128,
        dielectric=80.0,
        dtype=torch.float64,
        device="cpu",
    )
    continuum = build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
        functional, atoms
    )
    continuum_seconds = time.perf_counter() - continuum_started

    coordinates = AffineChargeCoordinates(
        atom_count=len(atoms),
        total_charge=0.0,
        source_space=ATOMIC_L1_SOURCE_SPACE,
    )
    equation = SeparatedOperationalStateEquation(
        coordinates, electronic, continuum
    )
    options = FixedPointOptions(
        method="anderson",
        tolerance=float(args.tolerance),
        max_iterations=int(args.max_iterations),
        damping=float(args.damping),
        history=int(args.history),
    )

    def resolved_energy(displaced: Atoms, *, label: str) -> tuple[float, object]:
        displaced_continuum = (
            build_mace_polar_point_harmonic_ddpcm_separated_snapshot(
                functional, displaced
            )
        )
        displaced_equation = SeparatedOperationalStateEquation(
            coordinates, electronic, displaced_continuum
        )
        displaced_state = solve_separated_fixed_point(
            displaced_equation,
            displaced,
            root_context_id=f"{context}-force-fd-{label}",
            options=options,
        )
        displaced_ledger = HarmonicDDPCMFrozenVacuumLedger(
            equation=displaced_equation,
            vacuum=electronic,
            field_semantics_manifest=field_semantics,
        )
        displaced_evaluation = displaced_ledger.evaluate_root(
            displaced,
            displaced_state.y_array(),
            root_tolerance=options.tolerance,
        )
        return displaced_evaluation.total_energy_eV, displaced_state

    solve_started = time.perf_counter()
    context = "water-official-macepolar-harmonic-ddpcm-mutual-scf-v1"
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
    solve_seconds = time.perf_counter() - solve_started
    roots_equivalent = separated_roots_numerically_equivalent(cold, wide)
    root_delta = float(np.linalg.norm(cold.y_array() - wide.y_array()))

    y = cold.y_array()
    direction = generator.normal(size=y.shape)
    cotangent = generator.normal(size=y.shape)
    residual_jvp = equation.residual_jvp(atoms, y, direction)
    residual_vjp = equation.residual_vjp(atoms, y, cotangent)
    adjoint_defect = abs(
        float(np.vdot(cotangent, residual_jvp))
        - float(np.vdot(residual_vjp, direction))
    )
    jacobian = dense_state_map_jacobian(
        lambda vector: equation.state_map_jvp(atoms, y, vector),
        dimension=equation.reduced_dimension,
    )
    certificate = certify_root_well_posedness(jacobian)
    ledger = HarmonicDDPCMFrozenVacuumLedger(
        equation=equation,
        vacuum=electronic,
        field_semantics_manifest=field_semantics,
    )
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
    if int(args.force_fd_levels) < 2:
        raise ValueError("force-fd-levels must be at least two.")
    if not np.isfinite(args.force_fd_step) or args.force_fd_step <= 0.0:
        raise ValueError("force-fd-step must be finite and positive.")
    if not np.isfinite(args.force_fd_tolerance) or args.force_fd_tolerance <= 0.0:
        raise ValueError("force-fd-tolerance must be finite and positive.")
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
            and operational_gradient.adjoint.converged
            and finest_force_error < float(args.force_fd_tolerance)
        )
        else "fail",
        "git": {
            "head": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "dirty": bool(_git(root, "status", "--porcelain=v1")),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
            "device": radial.provenance.device,
        },
        "native_semantics": native_semantics.as_dict(),
        "continuum": {
            "configuration_sha256": continuum.configuration_sha256(),
            "provenance_sha256": continuum.provenance_sha256,
            "surface_lmax": maximum,
            "boundary_dimension": continuum.boundary_dimension,
            "point_receiver_reconstruction_max_abs_error": (
                continuum.point_receiver_reconstruction_max_abs_error
            ),
            "model": "finite-dielectric-smooth-harmonic-ddPCM",
        },
        "state_equation": {
            "id": equation.state_equation_id,
            "fingerprint_sha256": equation.fingerprint_sha256(),
            "source_shape": [len(atoms), 4],
            "field_shape": [len(atoms), 8],
            "residual_jvp_vjp_absolute_defect": adjoint_defect,
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
                "This is the complete first coordinate derivative of the "
                "registered operational Phi0 scalar for this bound root, "
                "including vacuum, moving ddPCM, native-field, model source, "
                "and source-state implicit response terms."
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
            "each_displacement_rebuilds_continuum": True,
            "each_displacement_uses_cold_root": True,
        },
        "capabilities": NO_CAPABILITIES,
        "timing_seconds": {
            "model_load": model_seconds,
            "continuum_build": continuum_seconds,
            "two_root_solves": solve_seconds,
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
            "This canary proves one real-checkpoint rectangular mutual-SCF root, "
            "cold/wide replay, local nonsingularity, exact implemented JVP/VJP "
            "transpose consistency, and one resolved non-rigid directional "
            "finite-difference check of the complete operational Phi0 force. "
            "It does not prove a global unique root, quantitative physical "
            "validity of Phi0, distorted-domain force admission, or Tier V."
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
