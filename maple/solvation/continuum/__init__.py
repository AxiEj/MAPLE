"""Lazy public exports for continuum protocols and candidate adapters.

Importing a dependency-light numeric submodule must not execute every legacy or
optional continuum adapter.  Public compatibility names are therefore resolved
only when requested, while direct submodule imports retain a clean import graph.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ContinuumBackend": ("base", "ContinuumBackend"),
    "ConjugateRadialCPCMState": (
        "conjugate_fixed_topology_cpcm",
        "ConjugateRadialCPCMState",
    ),
    "ConjugateRadialGTOFixedTopologyCPCMBackend": (
        "conjugate_fixed_topology_cpcm",
        "ConjugateRadialGTOFixedTopologyCPCMBackend",
    ),
    "build_water_radial_gto_cpcm_1202_candidate": (
        "conjugate_fixed_topology_cpcm",
        "build_water_radial_gto_cpcm_1202_candidate",
    ),
    "build_water_radial_gto_cpcm_590_candidate": (
        "conjugate_fixed_topology_cpcm",
        "build_water_radial_gto_cpcm_590_candidate",
    ),
    "build_water_radial_gto_cpcm_backend": (
        "conjugate_fixed_topology_cpcm",
        "build_water_radial_gto_cpcm_backend",
    ),
    "FixedTopologyCPCMBackend": ("fixed_topology_cpcm", "FixedTopologyCPCMBackend"),
    "FixedTopologyCPCMState": ("fixed_topology_cpcm", "FixedTopologyCPCMState"),
    "FIXED_RECIPROCAL_CPCM_FUNCTIONAL_PROVIDER_ID": (
        "fixed_reciprocal_functional",
        "FIXED_RECIPROCAL_CPCM_FUNCTIONAL_PROVIDER_ID",
    ),
    "FixedReciprocalCPCMFunctional": (
        "fixed_reciprocal_functional",
        "FixedReciprocalCPCMFunctional",
    ),
    "ContinuumEnergyFunctional": ("functional", "ContinuumEnergyFunctional"),
    "FixedHarmonicGalerkinCPCMCandidate": (
        "harmonic_galerkin",
        "FixedHarmonicGalerkinCPCMCandidate",
    ),
    "FixedHarmonicGalerkinSnapshot": (
        "harmonic_galerkin",
        "FixedHarmonicGalerkinSnapshot",
    ),
    "HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID": (
        "harmonic_galerkin",
        "HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID",
    ),
    "HARMONIC_GALERKIN_CPCM_FUNCTIONAL_PROVIDER_ID": (
        "harmonic_galerkin",
        "HARMONIC_GALERKIN_CPCM_FUNCTIONAL_PROVIDER_ID",
    ),
    "HARMONIC_GALERKIN_STATIONARY_SCALAR_CONTRACT_ID": (
        "harmonic_galerkin",
        "HARMONIC_GALERKIN_STATIONARY_SCALAR_CONTRACT_ID",
    ),
    "PerAtomHarmonicSpace": ("harmonic_galerkin", "PerAtomHarmonicSpace"),
    "radial_gto_source_rotation_matrix": (
        "harmonic_galerkin",
        "radial_gto_source_rotation_matrix",
    ),
    "real_wigner_generators": ("harmonic_galerkin", "real_wigner_generators"),
    "real_wigner_matrix": ("harmonic_galerkin", "real_wigner_matrix"),
    "HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE": (
        "harmonic_exposure",
        "HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE",
    ),
    "SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID": (
        "harmonic_exposure",
        "SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID",
    ),
    "SMOOTH_HARMONIC_EXPOSURE_PROVIDER_ID": (
        "harmonic_exposure",
        "SMOOTH_HARMONIC_EXPOSURE_PROVIDER_ID",
    ),
    "SmoothHarmonicExposureSnapshot": (
        "harmonic_exposure",
        "SmoothHarmonicExposureSnapshot",
    ),
    "build_smooth_harmonic_exposure": (
        "harmonic_exposure",
        "build_smooth_harmonic_exposure",
    ),
    "harmonic_multiplication_matrix": (
        "harmonic_exposure",
        "harmonic_multiplication_matrix",
    ),
    "harmonic_weighted_basis_operator": (
        "harmonic_exposure",
        "harmonic_weighted_basis_operator",
    ),
    "project_harmonic_product": (
        "harmonic_exposure",
        "project_harmonic_product",
    ),
    "smooth_flat_step": ("harmonic_exposure", "smooth_flat_step"),
    "smooth_pair_exposure_coefficients": (
        "harmonic_exposure",
        "smooth_pair_exposure_coefficients",
    ),
    "HARMONIC_GAUSSIAN_SOURCE_CONTRACT_ID": (
        "harmonic_gaussian_source",
        "HARMONIC_GAUSSIAN_SOURCE_CONTRACT_ID",
    ),
    "HARMONIC_GAUSSIAN_SOURCE_PROVIDER_ID": (
        "harmonic_gaussian_source",
        "HARMONIC_GAUSSIAN_SOURCE_PROVIDER_ID",
    ),
    "HarmonicGaussianSourceSnapshot": (
        "harmonic_gaussian_source",
        "HarmonicGaussianSourceSnapshot",
    ),
    "build_harmonic_gaussian_source": (
        "harmonic_gaussian_source",
        "build_harmonic_gaussian_source",
    ),
    "gaussian_harmonic_source_operator": (
        "harmonic_gaussian_source",
        "gaussian_harmonic_source_operator",
    ),
    "HARMONIC_POINT_SOURCE_CONTRACT_ID": (
        "harmonic_point_source",
        "HARMONIC_POINT_SOURCE_CONTRACT_ID",
    ),
    "HARMONIC_POINT_SOURCE_PROVIDER_ID": (
        "harmonic_point_source",
        "HARMONIC_POINT_SOURCE_PROVIDER_ID",
    ),
    "harmonic_point_source_implementation_sha256": (
        "harmonic_point_source",
        "harmonic_point_source_implementation_sha256",
    ),
    "point_harmonic_source_operator": (
        "harmonic_point_source",
        "point_harmonic_source_operator",
    ),
    "COULOMB_EV_ANGSTROM_PER_E2": (
        "harmonic_single_layer",
        "COULOMB_EV_ANGSTROM_PER_E2",
    ),
    "HARMONIC_SINGLE_LAYER_CONTRACT_ID": (
        "harmonic_single_layer",
        "HARMONIC_SINGLE_LAYER_CONTRACT_ID",
    ),
    "HARMONIC_SINGLE_LAYER_PROVIDER_ID": (
        "harmonic_single_layer",
        "HARMONIC_SINGLE_LAYER_PROVIDER_ID",
    ),
    "canonical_harmonic_cross_block": (
        "harmonic_single_layer",
        "canonical_harmonic_cross_block",
    ),
    "harmonic_single_layer_operator": (
        "harmonic_single_layer",
        "harmonic_single_layer_operator",
    ),
    "SMOOTH_WEIGHTED_HARMONIC_GALERKIN_CONTRACT_ID": (
        "harmonic_weighted_galerkin",
        "SMOOTH_WEIGHTED_HARMONIC_GALERKIN_CONTRACT_ID",
    ),
    "SMOOTH_WEIGHTED_HARMONIC_GALERKIN_PROVIDER_ID": (
        "harmonic_weighted_galerkin",
        "SMOOTH_WEIGHTED_HARMONIC_GALERKIN_PROVIDER_ID",
    ),
    "SmoothWeightedHarmonicGalerkinAssembly": (
        "harmonic_weighted_galerkin",
        "SmoothWeightedHarmonicGalerkinAssembly",
    ),
    "build_smooth_weighted_harmonic_galerkin": (
        "harmonic_weighted_galerkin",
        "build_smooth_weighted_harmonic_galerkin",
    ),
    "SMOOTH_HARMONIC_GALERKIN_TORCH_FUNCTIONAL_CONTRACT_ID": (
        "harmonic_torch_functional",
        "SMOOTH_HARMONIC_GALERKIN_TORCH_FUNCTIONAL_CONTRACT_ID",
    ),
    "SMOOTH_HARMONIC_GALERKIN_TORCH_FUNCTIONAL_PROVIDER_ID": (
        "harmonic_torch_functional",
        "SMOOTH_HARMONIC_GALERKIN_TORCH_FUNCTIONAL_PROVIDER_ID",
    ),
    "SmoothWeightedHarmonicGalerkinFunctionalCandidate": (
        "harmonic_torch_functional",
        "SmoothWeightedHarmonicGalerkinFunctionalCandidate",
    ),
    "OrderedPairFrameEnsembleRadialGTOCPCMBackend": (
        "pair_frame_ensemble_cpcm",
        "OrderedPairFrameEnsembleRadialGTOCPCMBackend",
    ),
    "PairFrameCPCMState": ("pair_frame_ensemble_cpcm", "PairFrameCPCMState"),
    "build_pair_frame_water_cpcm_110_candidate": (
        "pair_frame_ensemble_cpcm",
        "build_pair_frame_water_cpcm_110_candidate",
    ),
    "DDX_CAVITY_PROFILE_ID": ("radial_gto_ddx", "DDX_CAVITY_PROFILE_ID"),
    "DDX_COSMO_PROFILE_ID": ("radial_gto_ddx", "DDX_COSMO_PROFILE_ID"),
    "DDX_PCM_PROFILE_ID": ("radial_gto_ddx", "DDX_PCM_PROFILE_ID"),
    "DDX_RADIAL_PROVIDER_ID": ("radial_gto_ddx", "DDX_RADIAL_PROVIDER_ID"),
    "DDX_SOURCE_COMMIT": ("radial_gto_ddx", "DDX_SOURCE_COMMIT"),
    "DDX_WATER_194_CONFIGURATION_CONTRACT_ID": (
        "radial_gto_ddx",
        "DDX_WATER_194_CONFIGURATION_CONTRACT_ID",
    ),
    "RadialGTODDXBackend": ("radial_gto_ddx", "RadialGTODDXBackend"),
    "RadialGTODDXState": ("radial_gto_ddx", "RadialGTODDXState"),
    "build_water_radial_gto_ddpcm_194_candidate": (
        "radial_gto_ddx",
        "build_water_radial_gto_ddpcm_194_candidate",
    ),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
