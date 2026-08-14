"""Continuum response protocols and production adapters."""

from .base import ContinuumBackend
from .conjugate_fixed_topology_cpcm import (
    ConjugateRadialCPCMState,
    ConjugateRadialGTOFixedTopologyCPCMBackend,
    build_water_radial_gto_cpcm_1202_candidate,
    build_water_radial_gto_cpcm_590_candidate,
    build_water_radial_gto_cpcm_backend,
)
from .fixed_topology_cpcm import FixedTopologyCPCMBackend, FixedTopologyCPCMState
from .fixed_reciprocal_functional import (
    FIXED_RECIPROCAL_CPCM_FUNCTIONAL_PROVIDER_ID,
    FixedReciprocalCPCMFunctional,
)
from .functional import ContinuumEnergyFunctional
from .harmonic_galerkin import (
    FixedHarmonicGalerkinCPCMCandidate,
    FixedHarmonicGalerkinSnapshot,
    HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID,
    HARMONIC_GALERKIN_CPCM_FUNCTIONAL_PROVIDER_ID,
    HARMONIC_GALERKIN_STATIONARY_SCALAR_CONTRACT_ID,
    PerAtomHarmonicSpace,
    radial_gto_source_rotation_matrix,
    real_wigner_matrix,
)
from .harmonic_exposure import (
    HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE,
    SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID,
    SMOOTH_HARMONIC_EXPOSURE_PROVIDER_ID,
    SmoothHarmonicExposureSnapshot,
    build_smooth_harmonic_exposure,
    harmonic_multiplication_matrix,
    project_harmonic_product,
    smooth_flat_step,
    smooth_pair_exposure_coefficients,
)
from .pair_frame_ensemble_cpcm import (
    OrderedPairFrameEnsembleRadialGTOCPCMBackend,
    PairFrameCPCMState,
    build_pair_frame_water_cpcm_110_candidate,
)
from .radial_gto_ddx import (
    DDX_CAVITY_PROFILE_ID,
    DDX_COSMO_PROFILE_ID,
    DDX_PCM_PROFILE_ID,
    DDX_RADIAL_PROVIDER_ID,
    DDX_SOURCE_COMMIT,
    DDX_WATER_194_CONFIGURATION_CONTRACT_ID,
    RadialGTODDXBackend,
    RadialGTODDXState,
    build_water_radial_gto_ddpcm_194_candidate,
)

__all__ = [
    "ConjugateRadialCPCMState",
    "ConjugateRadialGTOFixedTopologyCPCMBackend",
    "build_water_radial_gto_cpcm_1202_candidate",
    "build_water_radial_gto_cpcm_590_candidate",
    "build_water_radial_gto_cpcm_backend",
    "ContinuumBackend",
    "ContinuumEnergyFunctional",
    "FIXED_RECIPROCAL_CPCM_FUNCTIONAL_PROVIDER_ID",
    "FixedReciprocalCPCMFunctional",
    "FixedTopologyCPCMBackend",
    "FixedTopologyCPCMState",
    "FixedHarmonicGalerkinCPCMCandidate",
    "FixedHarmonicGalerkinSnapshot",
    "HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID",
    "HARMONIC_GALERKIN_CPCM_FUNCTIONAL_PROVIDER_ID",
    "HARMONIC_GALERKIN_STATIONARY_SCALAR_CONTRACT_ID",
    "HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE",
    "OrderedPairFrameEnsembleRadialGTOCPCMBackend",
    "PairFrameCPCMState",
    "SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID",
    "SMOOTH_HARMONIC_EXPOSURE_PROVIDER_ID",
    "SmoothHarmonicExposureSnapshot",
    "build_pair_frame_water_cpcm_110_candidate",
    "build_smooth_harmonic_exposure",
    "DDX_CAVITY_PROFILE_ID",
    "DDX_COSMO_PROFILE_ID",
    "DDX_PCM_PROFILE_ID",
    "DDX_RADIAL_PROVIDER_ID",
    "DDX_SOURCE_COMMIT",
    "DDX_WATER_194_CONFIGURATION_CONTRACT_ID",
    "RadialGTODDXBackend",
    "RadialGTODDXState",
    "PerAtomHarmonicSpace",
    "radial_gto_source_rotation_matrix",
    "real_wigner_matrix",
    "harmonic_multiplication_matrix",
    "project_harmonic_product",
    "smooth_flat_step",
    "smooth_pair_exposure_coefficients",
    "build_water_radial_gto_ddpcm_194_candidate",
]
