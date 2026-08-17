"""Shared real-stack setup for disabled AIMNet2 geometry-mediated diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
from maple.function.route2_solvents import route2_solvent_spec
from maple.solvation.api import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1,
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_PROFILE_V1,
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
)
from maple.solvation.continuum.atomic_l1_pyddx import AtomicL1PyDDXPCMBackend
from maple.solvation.continuum.harmonic_point_torch_functional import (
    SmoothPointChargeHarmonicGalerkinFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_point_ddpcm_torch_functional import (
    SmoothPointChargeHarmonicDDPCMFunctionalCandidate,
    build_water_aimnet2_frozen_charge_harmonic_ddpcm_candidate,
)
from maple.solvation.coupling.geometry_mediated import (
    GeometryMediatedElectrostaticScalar,
)
from maple.solvation.models.aimnet2 import (
    AIMNET2_WB97M_D3_CHECKPOINT_SHA256,
    AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES,
    AIMNET2_WB97M_D3_FROZEN_CHARGE_WATER_FLOAT64_CONTRACT,
    AIMNET2_WB97M_D3_RECONSTRUCTED_FLOAT64_CONTRACT,
    AIMNet2GeometryMediatedModelAdapter,
)
from maple.solvation.release import sha256_file
from maple.solvation.release.geometry_mediated import (
    GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV,
    GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A,
    GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE,
)

NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
COMMON_REQUIRED_SOURCE_PATHS = (
    "maple/function/calculator/aimnet/_aimnet2_calculator.py",
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/api/state_registry.py",
    "maple/solvation/coupling/geometry_mediated.py",
    "maple/solvation/coupling/metrics.py",
    "maple/solvation/models/aimnet2.py",
    "maple/solvation/release/evidence.py",
    "maple/solvation/release/geometry_mediated.py",
    "tools/route2_release/aimnet2_geometry_mediated_common.py",
)
CONTINUUM_REQUIRED_SOURCE_PATHS = {
    "ddpcm": (
        "maple/function/calculator/extra_correction/implicit/pyddx_pcm_response.py",
        "maple/solvation/continuum/atomic_l1_pyddx.py",
    ),
    "harmonic-point": (
        "maple/solvation/continuum/functional.py",
        "maple/solvation/continuum/harmonic_coefficients.py",
        "maple/solvation/continuum/harmonic_exposure.py",
        "maple/solvation/continuum/harmonic_point_source.py",
        "maple/solvation/continuum/harmonic_point_torch_functional.py",
        "maple/solvation/continuum/harmonic_single_layer.py",
        "maple/solvation/continuum/harmonic_torch_functional.py",
        "maple/solvation/continuum/harmonic_torch_primitives.py",
    ),
    "harmonic-ddpcm": (
        "maple/solvation/continuum/functional.py",
        "maple/solvation/continuum/harmonic_coefficients.py",
        "maple/solvation/continuum/harmonic_exposure.py",
        "maple/solvation/continuum/harmonic_point_ddpcm_torch_functional.py",
        "maple/solvation/continuum/harmonic_point_source.py",
        "maple/solvation/continuum/harmonic_point_torch_functional.py",
        "maple/solvation/continuum/harmonic_single_layer.py",
        "maple/solvation/continuum/harmonic_torch_functional.py",
        "maple/solvation/continuum/harmonic_torch_primitives.py",
    ),
    "harmonic-ddpcm-water": (
        "maple/function/calculator/extra_correction/implicit/smd_cds.py",
        "maple/function/route2_smd_profiles.py",
        "maple/function/route2_solvents.py",
        "maple/solvation/continuum/functional.py",
        "maple/solvation/continuum/harmonic_coefficients.py",
        "maple/solvation/continuum/harmonic_exposure.py",
        "maple/solvation/continuum/harmonic_point_ddpcm_torch_functional.py",
        "maple/solvation/continuum/harmonic_point_source.py",
        "maple/solvation/continuum/harmonic_point_torch_functional.py",
        "maple/solvation/continuum/harmonic_single_layer.py",
        "maple/solvation/continuum/harmonic_torch_functional.py",
        "maple/solvation/continuum/harmonic_torch_primitives.py",
    ),
}
MODEL_RUNTIME_REQUIRED_SOURCE_PATHS = {
    "legacy-jit-float32": (),
    "reconstructed-python-float64": (
        "maple/function/calculator/aimnet/_aimnet2_float64_source.py",
    ),
}


def verify_route2_checkpoint(checkpoint: Path) -> None:
    """Reject non-contract checkpoint bytes before deserialization."""

    if (
        checkpoint.stat().st_size != AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES
        or sha256_file(checkpoint) != AIMNET2_WB97M_D3_CHECKPOINT_SHA256
    ):
        raise ValueError(
            "The AIMNet2 checkpoint bytes do not match the Route-2 contract."
        )


def build_geometry_mediated_stack(
    atoms: Atoms,
    checkpoint: Path,
    device: str,
    continuum_kind: str,
    aimnet_runtime: str = "legacy-jit-float32",
):
    """Build one geometry-sized disabled model/continuum/scalar stack."""

    import torch

    torch.manual_seed(20260815)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260815)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if continuum_kind == "harmonic-ddpcm-water" and aimnet_runtime != (
        "reconstructed-python-float64"
    ):
        raise ValueError(
            "The solvent-bound frozen-charge candidate requires the source-bound "
            "reconstructed AIMNet2 float64 runtime."
        )
    if aimnet_runtime == "legacy-jit-float32":
        calculator = AIMNet2Calculator(
            device=torch.device(device),
            model="aimnet2",
            model_path=str(checkpoint),
            coulomb_method="simple",
        )
        model = AIMNet2GeometryMediatedModelAdapter(calculator)
        model_runtime = {
            "runtime_kind": aimnet_runtime,
            "coordinate_dtype": "torch.float32",
            "checkpoint_weights_changed": False,
            "forward_semantics": (
                "legacy TorchScript _prepare_dtype hard-casts coordinates to "
                "torch.float32"
            ),
            "negative_control": True,
            "ase_calculator_implementation": True,
            "route2_public_ase_admitted": False,
        }
    elif aimnet_runtime == "reconstructed-python-float64":
        if device != "cpu":
            raise ValueError(
                "The source-bound reconstructed AIMNet2 runtime is CPU-only."
            )
        from maple.function.calculator.aimnet._aimnet2_float64_source import (
            AIMNet2ReconstructedFloat64SourceCalculator,
        )

        calculator = AIMNet2ReconstructedFloat64SourceCalculator(
            model_path=checkpoint,
            device=device,
        )
        model = AIMNet2GeometryMediatedModelAdapter(
            calculator,
            contract=(
                AIMNET2_WB97M_D3_FROZEN_CHARGE_WATER_FLOAT64_CONTRACT
                if continuum_kind == "harmonic-ddpcm-water"
                else AIMNET2_WB97M_D3_RECONSTRUCTED_FLOAT64_CONTRACT
            ),
        )
        model_runtime = calculator.runtime_provenance()
        model_runtime.update(
            {
                "checkpoint_weights_changed": False,
                "negative_control": False,
            }
        )
    else:
        raise ValueError(f"unsupported AIMNet2 runtime: {aimnet_runtime}")
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent="water",
        profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
    )
    if continuum_kind == "ddpcm":
        dielectric = float(route2_solvent_spec("water").descriptors.dielectric)
        continuum = AtomicL1PyDDXPCMBackend(
            atoms,
            radii,
            dielectric=dielectric,
            lmax=7,
            n_lebedev=302,
            n_proc=1,
            solver_tolerance=1.0e-12,
            eta=0.1,
        )
        scalar = GeometryMediatedElectrostaticScalar(model, continuum)
        protocol = {
            "model": "ddPCM",
            "dielectric": dielectric,
            "radii_A": np.asarray(radii, dtype=float).tolist(),
            "lmax": 7,
            "n_lebedev": 302,
            "solver_tolerance": 1.0e-12,
            "eta": 0.1,
            "post_solve_residual_available": False,
        }
    elif continuum_kind == "harmonic-point":
        continuum = SmoothPointChargeHarmonicGalerkinFunctionalCandidate(
            atomic_numbers=tuple(int(value) for value in atoms.numbers),
            radii_angstrom=tuple(float(value) for value in radii),
            transition_width_angstrom2=0.18,
            surface_lmax=1,
            exposure_lmax=2,
            exposure_radial_quadrature_order=32,
            green_radial_quadrature_order=32,
            dtype=torch.float64,
            device=device,
        )
        scalar = GeometryMediatedElectrostaticScalar(
            model,
            continuum,
            scalar_id=(
                DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1
            ),
            profile_id=(
                DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_PROFILE_V1
            ),
        )
        protocol = {
            "model": "smooth-weighted-harmonic-conductor-reference",
            "finite_dielectric_parameterization": False,
            "radii_A": np.asarray(radii, dtype=float).tolist(),
            "transition_width_A2": 0.18,
            "surface_lmax": 1,
            "exposure_lmax": 2,
            "exposure_radial_quadrature_order": 32,
            "green_radial_quadrature_order": 32,
            "point_source_map": "analytic-Laplace-addition-theorem",
            "post_solve_residual_available": True,
        }
    elif continuum_kind == "harmonic-ddpcm":
        dielectric = float(route2_solvent_spec("water").descriptors.dielectric)
        continuum = SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
            atomic_numbers=tuple(int(value) for value in atoms.numbers),
            radii_angstrom=tuple(float(value) for value in radii),
            dielectric=dielectric,
            transition_width_angstrom2=0.18,
            surface_lmax=1,
            exposure_lmax=2,
            exposure_radial_quadrature_order=32,
            green_radial_quadrature_order=32,
            dtype=torch.float64,
            device=device,
        )
        scalar = GeometryMediatedElectrostaticScalar(
            model,
            continuum,
            scalar_id=(
                DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
            ),
            profile_id=(
                DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
            ),
        )
        protocol = {
            "model": "smooth-weighted-harmonic-finite-dielectric-ddpcm",
            "finite_dielectric_parameterization": True,
            "dielectric": dielectric,
            "uniform_cosmo_dielectric_energy_scaling": False,
            "admission_identity": (
                "parameterized-diagnostic-only; solvent-bound profile required"
            ),
            "radii_A": np.asarray(radii, dtype=float).tolist(),
            "transition_width_A2": 0.18,
            "surface_lmax": 1,
            "exposure_lmax": 2,
            "exposure_radial_quadrature_order": 32,
            "green_radial_quadrature_order": 32,
            "double_layer_quadrature": (
                "pair-axis split Gauss-Legendre with exact retained-band "
                "azimuthal contraction"
            ),
            "point_source_map": "analytic-Laplace-addition-theorem",
            "finite_dielectric_equation": (
                "R_epsilon Phi_epsilon=R_infinity Phi; " "S sigma=-Phi_epsilon"
            ),
            "provider_field_semantics": (
                "energy-conjugate source derivative from the transpose/KKT chain; "
                "not the generally nonsymmetric primal apparent-charge response"
            ),
            "post_solve_residual_available": True,
        }
    elif continuum_kind == "harmonic-ddpcm-water":
        continuum = build_water_aimnet2_frozen_charge_harmonic_ddpcm_candidate(
            tuple(atoms.get_chemical_symbols()),
            dtype=torch.float64,
            device=device,
        )
        scalar = GeometryMediatedElectrostaticScalar(
            model,
            continuum,
            scalar_id=(
                CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
            ),
            profile_id=(
                CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
            ),
        )
        protocol = {
            "model": "water-bound-smooth-weighted-harmonic-finite-dielectric-ddpcm",
            "finite_dielectric_parameterization": True,
            "solvent": "water",
            "dielectric": continuum.dielectric,
            "uniform_cosmo_dielectric_energy_scaling": False,
            "admission_identity": (
                "water-bound-frozen-charge-candidate; capabilities-none"
            ),
            "aimnet2_source_evaluation": "one-shot-per-geometry",
            "continuum_field_supplied_to_aimnet2": False,
            "electronic_scf_iteration": False,
            "radii_A": list(continuum.radii_angstrom),
            "transition_width_A2": 0.18,
            "surface_lmax": 1,
            "exposure_lmax": 2,
            "exposure_radial_quadrature_order": 32,
            "green_radial_quadrature_order": 32,
            "double_layer_quadrature": (
                "pair-axis split Gauss-Legendre with exact retained-band "
                "azimuthal contraction"
            ),
            "point_source_map": "analytic-Laplace-addition-theorem",
            "finite_dielectric_equation": (
                "R_epsilon Phi_epsilon=R_infinity Phi; S sigma=-Phi_epsilon"
            ),
            "provider_field_semantics": (
                "energy-conjugate source derivative from the transpose/KKT chain; "
                "never supplied to AIMNet2"
            ),
            "post_solve_residual_available": True,
        }
    else:
        raise ValueError(f"unsupported continuum kind: {continuum_kind}")
    return model, continuum, scalar, protocol, model_runtime


def geometry_mediated_topologies(model, continuum, atoms: Atoms):
    """Capture the hard model and continuum topology records."""

    return model.neighbor_topology(atoms).as_dict(), continuum.topology_state(atoms)


def deterministic_replay(first, second) -> dict[str, object]:
    """Compare two independent evaluations of the same explicit scalar."""

    energy_error = abs(first.energy.total_energy_eV - second.energy.total_energy_eV)
    source_error = float(np.linalg.norm(first.source - second.source))
    gradient_error = float(
        np.linalg.norm(first.total_gradient_eV_per_A - second.total_gradient_eV_per_A)
    )
    return {
        "energy_absolute_error_eV": energy_error,
        "source_difference_norm": source_error,
        "gradient_difference_norm_eV_per_A": gradient_error,
        "gate_passed": (
            energy_error <= GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV
            and source_error <= GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE
            and gradient_error <= GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A
        ),
    }


__all__ = [
    "COMMON_REQUIRED_SOURCE_PATHS",
    "CONTINUUM_REQUIRED_SOURCE_PATHS",
    "MODEL_RUNTIME_REQUIRED_SOURCE_PATHS",
    "NO_CAPABILITIES",
    "build_geometry_mediated_stack",
    "deterministic_replay",
    "geometry_mediated_topologies",
    "verify_route2_checkpoint",
]
