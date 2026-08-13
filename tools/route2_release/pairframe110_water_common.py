"""Shared setup for the disabled equivariant pair-frame C-PCM candidate."""

from __future__ import annotations

from pathlib import Path

from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
)
from maple.solvation.api.profiles import (
    DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.continuum import build_pair_frame_water_cpcm_110_candidate
from maple.solvation.coupling.energy import OperationalElectrostaticScalar
from maple.solvation.coupling.exact_gto import MACE_POLAR_RADIAL_GTO_COUPLING_ID
from maple.solvation.coupling.spaces import (
    LinearChargeCoordinates,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import ReducedStateEquation
from maple.solvation.models import (
    ElectronicResponseEquationAdapter,
    VacuumScalarEquationAdapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)

DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"


def build_system(atoms, checkpoint: Path, device: str):
    model = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=device,
        long_range_evaluator_profile=(MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE),
    )
    return build_system_with_model(atoms, model)


def build_system_with_model(atoms, model):
    profile = get_solvation_profile(DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1)
    if profile.enabled or profile.capabilities.enabled_tiers:
        raise RuntimeError("pair-frame diagnostic profile must remain disabled.")
    model.domain.validate_atoms(atoms)
    continuum = build_pair_frame_water_cpcm_110_candidate(atoms.get_chemical_symbols())
    coordinates = LinearChargeCoordinates(
        len(atoms),
        total_charge=0.0,
        source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
        component_scales=(1.0,) * 8,
    )
    equation = ReducedStateEquation(
        coordinates,
        ElectronicResponseEquationAdapter(model, MACE_POLAR_RADIAL_GTO_COUPLING_ID),
        continuum,
    )
    scalar = OperationalElectrostaticScalar(
        equation,
        VacuumScalarEquationAdapter(model),
        profile_id=profile.profile_id,
    )
    return profile, model, continuum, equation, scalar


__all__ = [
    "DEFAULT_CHECKPOINT",
    "build_system",
    "build_system_with_model",
]
