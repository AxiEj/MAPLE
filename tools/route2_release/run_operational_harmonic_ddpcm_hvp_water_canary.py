#!/usr/bin/env python3
"""Real-checkpoint HVP canary for both Route-2 harmonic-ddPCM profiles.

This bounded canary evaluates exactly two non-rigid Hessian-vector directions
for each of:

* pure MACE-POLAR original 1.5-A Gaussian source + native 1.5/3.0-A field;
* MACE-MDP point permanent source + MACE-POLAR Gaussian induced response.

Every displaced force rebuilds the continuum and solves a cold operational
root.  HVPs are fourth-order Richardson derivatives of the exact implicit
same-scalar force.  The artifact is research evidence; it does not enable a
public capability or assert Tier V.
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
from maple.solvation.coupling.fixed_point import FixedPointOptions  # noqa: E402
from maple.solvation.coupling.operational_pes import (  # noqa: E402
    OperationalImplicitPES,
)
from maple.solvation.derivatives import RichardsonScalarHessian  # noqa: E402
from maple.solvation.experimental.harmonic_ddpcm_operational import (  # noqa: E402
    MACEPolarGaussianHarmonicDDPCMBuilder,
    MACE_MDPPolarGeneralSourceHarmonicDDPCMBuilder,
)
from maple.solvation.models import (  # noqa: E402
    MACEPolarOriginalSourceNativeFieldAdapter,
    NativeSemanticsCanary,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release.field_semantics import (  # noqa: E402
    build_unverified_mace_polar_field_semantics_manifest,
)


SCHEMA_VERSION = "route2-operational-harmonic-ddpcm-water-hvp-canary-v1"
DEFAULT_MDP_CHECKPOINT = Path.home() / ".cache/mace/MACE-MDP.model"
DEFAULT_POLAR_CHECKPOINT = Path.home() / ".cache/mace/MACEPOLAR1Mmodel"
DEFAULT_OUTPUT = (
    Path(".omx/route2") / "operational-harmonic-ddpcm-water-hvp-canary-v1.json"
)
WATER_RADII_ANGSTROM = (1.52, 1.2, 1.2)
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
SOURCE_FILES = (
    "maple/solvation/continuum/harmonic_ddpcm_functional.py",
    "maple/solvation/continuum/harmonic_ddpcm_gaussian.py",
    "maple/solvation/continuum/harmonic_ddpcm_hybrid.py",
    "maple/solvation/coupling/operational_pes.py",
    "maple/solvation/coupling/permanent_induced_ledgers.py",
    "maple/solvation/coupling/permanent_induced_state.py",
    "maple/solvation/coupling/separated_ledgers.py",
    "maple/solvation/coupling/separated_state.py",
    "maple/solvation/coupling/state_equation.py",
    "maple/solvation/derivatives/molecular_virial.py",
    "maple/solvation/derivatives/scalar_finite_difference.py",
    "maple/solvation/experimental/harmonic_ddpcm_operational.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar_separated.py",
    "tools/route2_release/run_operational_harmonic_ddpcm_hvp_water_canary.py",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp-checkpoint", type=Path, default=DEFAULT_MDP_CHECKPOINT)
    parser.add_argument(
        "--polar-checkpoint", type=Path, default=DEFAULT_POLAR_CHECKPOINT
    )
    parser.add_argument("--polar-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surface-lmax", type=int, default=1)
    parser.add_argument("--quadrature-order", type=int, default=32)
    parser.add_argument("--root-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--hvp-coarse-step", type=float, default=1.25e-4)
    parser.add_argument("--hvp-max-error", type=float, default=5.0e-3)
    parser.add_argument("--bilinear-tolerance", type=float, default=2.0e-4)
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


def _directions(atom_count: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(20260903)
    values: list[np.ndarray] = []
    for _ in range(2):
        direction = generator.normal(size=(atom_count, 3))
        direction -= np.mean(direction, axis=0, keepdims=True)
        for previous in values:
            direction -= float(np.vdot(previous, direction)) * previous
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1.0e-12:
            raise RuntimeError("failed to construct independent HVP directions.")
        values.append(direction / norm)
    return values[0], values[1]


def _audit_profile(
    name: str,
    pes: OperationalImplicitPES,
    geometry: Atoms,
    directions: tuple[np.ndarray, np.ndarray],
) -> dict[str, object]:
    started = time.perf_counter()
    central = pes.evaluate(geometry, need_forces=True)
    central_force = central.force_sample
    virial = pes.molecular_virial(geometry, central_force=central_force)
    first = pes.hessian_vector_product(
        geometry, directions[0], central_force=central_force
    )
    second = pes.hessian_vector_product(
        geometry, directions[1], central_force=central_force
    )
    h11 = float(np.vdot(directions[0], first.hvp_eV_per_A2))
    h12 = float(np.vdot(directions[0], second.hvp_eV_per_A2))
    h21 = float(np.vdot(first.hvp_eV_per_A2, directions[1]))
    h22 = float(np.vdot(directions[1], second.hvp_eV_per_A2))
    defect = abs(h12 - h21)
    maximum_error = max(
        first.maximum_error_estimate_eV_per_A2,
        second.maximum_error_estimate_eV_per_A2,
    )
    return {
        "name": name,
        "provider_id": pes.provider_id,
        "scalar_id": pes.scalar_id,
        "configuration_sha256": pes.configuration_sha256(),
        "energy_eV": central.energy_eV,
        "state_sha256": central.state_sha256,
        "force_evaluation_sha256": central.evaluation_sha256,
        "root_residual_norm": central.root_residual_norm,
        "adjoint_true_residual_norm": central.adjoint_true_residual_norm,
        "forces_eV_per_A": central.force_sample.forces_eV_per_A.tolist(),
        "net_force_eV_per_A": np.sum(
            central.force_sample.forces_eV_per_A, axis=0
        ).tolist(),
        "virial": {
            "evaluation_sha256": virial.evaluation_sha256,
            "raw_eV": virial.raw_virial_eV.tolist(),
            "symmetric_eV": virial.symmetric_virial_eV.tolist(),
            "strain_gradient_eV": virial.strain_gradient_eV.tolist(),
            "maximum_antisymmetry_eV": virial.maximum_antisymmetry_eV,
        },
        "directions": [direction.tolist() for direction in directions],
        "hvp_eV_per_A2": [
            first.hvp_eV_per_A2.tolist(),
            second.hvp_eV_per_A2.tolist(),
        ],
        "hvp_error_estimates_eV_per_A2": [
            first.error_estimates_eV_per_A2.tolist(),
            second.error_estimates_eV_per_A2.tolist(),
        ],
        "hvp_maximum_error_estimate_eV_per_A2": maximum_error,
        "projected_hessian_eV_per_A2": [[h11, h12], [h21, h22]],
        "projected_hessian_bilinear_asymmetry_eV_per_A2": defect,
        "topology_id": central.topology_id,
        "timing_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser()
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)

    import torch

    torch.manual_seed(20260903)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260903)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.polar_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    started = time.perf_counter()
    geometry = _water()
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.polar_device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    response = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    native_semantics = NativeSemanticsCanary.from_adapter(radial)
    field_semantics = build_unverified_mace_polar_field_semantics_manifest(
        native_canary=native_semantics,
        adapter=response,
    )
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=response,
    )

    maximum = int(args.surface_lmax)
    quadrature = int(args.quadrature_order)
    continuum = SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8, 1, 1),
        radii_angstrom=WATER_RADII_ANGSTROM,
        transition_width_angstrom2=0.08,
        surface_lmax=maximum,
        partition_lmax=2 * maximum,
        partition_radial_quadrature_order=quadrature,
        source_radial_quadrature_order=quadrature,
        double_layer_radial_quadrature_order=quadrature,
        dielectric=80.0,
        dtype=torch.float64,
        device="cpu",
    )
    root_options = FixedPointOptions(
        method="anderson",
        tolerance=float(args.root_tolerance),
        max_iterations=100,
        damping=0.5,
        history=6,
    )
    hessian_backend = RichardsonScalarHessian(
        coarse_step_angstrom=float(args.hvp_coarse_step),
        maximum_error_eV_per_A2=float(args.hvp_max_error),
        maximum_antisymmetry_eV_per_A2=float(args.bilinear_tolerance),
    )
    pure = OperationalImplicitPES(
        MACEPolarGaussianHarmonicDDPCMBuilder(
            electronic=response,
            continuum=continuum,
            field_semantics_manifest=field_semantics,
            receiver_radial_quadrature_order=quadrature,
        ),
        root_options=root_options,
        hessian_backend=hessian_backend,
    )
    combined = OperationalImplicitPES(
        MACE_MDPPolarGeneralSourceHarmonicDDPCMBuilder(
            hybrid=hybrid,
            continuum=continuum,
            receiver_radial_quadrature_order=quadrature,
        ),
        root_options=root_options,
        hessian_backend=hessian_backend,
    )
    directions = _directions(len(geometry))
    records = (
        _audit_profile("pure-macepolar-gto1p5", pure, geometry, directions),
        _audit_profile("mace-mdp-point-plus-macepolar-induced", combined, geometry, directions),
    )
    pass_status = all(
        record["root_residual_norm"] <= root_options.tolerance
        and record["adjoint_true_residual_norm"] <= 1.0e-8
        and record["hvp_maximum_error_estimate_eV_per_A2"]
        <= float(args.hvp_max_error)
        and record["projected_hessian_bilinear_asymmetry_eV_per_A2"]
        <= float(args.bilinear_tolerance)
        for record in records
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if pass_status else "fail",
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
            },
            "mace_polar": {
                "path": str(polar_checkpoint),
                "sha256": response.checkpoint_sha256,
                "device": radial.provenance.device,
            },
        },
        "continuum": {
            "configuration_sha256": continuum.configuration_sha256(),
            "provenance_sha256": continuum.provenance_sha256,
            "surface_lmax": maximum,
            "quadrature_order": quadrature,
            "dielectric": 80.0,
            "radii_angstrom": WATER_RADII_ANGSTROM,
        },
        "root_options": {
            "method": root_options.method,
            "tolerance": root_options.tolerance,
            "max_iterations": root_options.max_iterations,
            "damping": root_options.damping,
            "history": root_options.history,
        },
        "hvp_options": {
            "coarse_step_angstrom": hessian_backend.coarse_step_angstrom,
            "fine_step_angstrom": hessian_backend.fine_step_angstrom,
            "maximum_error_eV_per_A2": (
                hessian_backend.maximum_error_eV_per_A2
            ),
            "bilinear_tolerance_eV_per_A2": float(args.bilinear_tolerance),
            "derivative": (
                "fourth-order Richardson differentiation of the exact implicit "
                "same-scalar force; every stencil point uses a cold root"
            ),
        },
        "profiles": records,
        "capabilities": NO_CAPABILITIES,
        "timing_seconds": {"total": time.perf_counter() - started},
        "runtime": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
        },
        "source_files_sha256": {
            relative: _sha256_file(root / relative) for relative in SOURCE_FILES
        },
        "claim_boundary": (
            "This bounded canary proves callable same-scalar E/F/molecular-virial "
            "and two-direction HVP consistency for both exact source profiles on "
            "one water geometry. The generic full Hessian closure is separately "
            "unit-tested. It does not prove a global single root, distorted-PES "
            "admission, H/FREQ accuracy, total solvation free energy, or Tier V."
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
