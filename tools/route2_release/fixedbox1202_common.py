"""Shared setup for the disabled fixed-box48/CPCM1202 diagnostic."""

from __future__ import annotations

from maple.solvation.api.profiles import (
    DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.continuum import build_water_radial_gto_cpcm_1202_candidate
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
)

from fixedbox590_water_common import COMMON_REQUIRED_SOURCE_PATHS

FIXEDBOX1202_BOX_LENGTH_A = 48
FIXEDBOX1202_REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/symmetry_panel.py",
    "tools/route2_release/run_fixedbox590_symmetry_panel.py",
    "tools/route2_release/aggregate_fixedbox590_symmetry_panel.py",
    "tools/route2_release/fixedbox1202_common.py",
    "tools/route2_release/run_fixedbox1202_symmetry_panel.py",
    "tools/route2_release/aggregate_fixedbox1202_symmetry_panel.py",
)


def build_fixedbox1202_system_with_model(atoms, model):
    """Build one geometry-sized 1202-node system around a loaded box48 model."""

    profile = get_solvation_profile(
        DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1
    )
    if profile.enabled or profile.capabilities.enabled_tiers:
        raise RuntimeError("The fixed-box48/CPCM1202 profile must remain disabled.")
    model.domain.validate_atoms(atoms)
    if model.model_profile_id != profile.model_profile:
        raise ValueError("The loaded model does not implement the box48 profile.")
    continuum = build_water_radial_gto_cpcm_1202_candidate(
        atoms.get_chemical_symbols()
    )
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
    "FIXEDBOX1202_BOX_LENGTH_A",
    "FIXEDBOX1202_REQUIRED_SOURCE_PATHS",
    "build_fixedbox1202_system_with_model",
]
