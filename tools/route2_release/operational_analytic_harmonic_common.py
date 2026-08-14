"""Shared identity/build helpers for the disabled operational harmonic scalar."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api import (
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    get_solvation_profile,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.continuum import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)
from maple.solvation.coupling.operational_state import (
    build_disabled_operational_electrostatic_scalar,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter

DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
SCALAR_ID = (
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
)
PROFILE_ID = OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
SURFACE_LMAX = 1
EXPOSURE_LMAX = 2
TRANSITION_WIDTH_ANGSTROM2 = 0.18
EXPOSURE_RADIAL_QUADRATURE_ORDER = 32
SOURCE_RADIAL_QUADRATURE_ORDER = 32
GREEN_RADIAL_QUADRATURE_ORDER = 32
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}

COMMON_REQUIRED_SOURCE_PATHS = (
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/api/state_registry.py",
    "maple/solvation/continuum/functional.py",
    "maple/solvation/continuum/harmonic_torch_functional.py",
    "maple/solvation/continuum/harmonic_torch_primitives.py",
    "maple/solvation/coupling/adjoint.py",
    "maple/solvation/coupling/energy.py",
    "maple/solvation/coupling/fixed_point.py",
    "maple/solvation/coupling/operational_state.py",
    "maple/solvation/coupling/state_equation.py",
    "maple/solvation/coupling/variational_adapters.py",
    "maple/solvation/models/equation_adapter.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/runtime/analytic_gaussian_multipole.py",
    "maple/solvation/release/evidence.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/calculator/mace/_macepol_long_range.py",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "tools/route2_release/operational_analytic_harmonic_common.py",
)


def build_model(checkpoint: Path, device: str):
    """Load the exact analytic-evaluator checkpoint adapter."""

    return build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=Path(checkpoint),
        device=str(device),
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )


def build_system_with_model(atoms: Atoms, model):
    """Build the registered geometry-sized continuum/equation/scalar tuple."""

    profile = get_solvation_profile(PROFILE_ID)
    if profile.enabled or profile.capabilities.enabled_tiers:
        raise RuntimeError("The operational harmonic profile must remain disabled.")
    model.domain.validate_atoms(atoms)
    radii = tuple(
        float(value) for value in smd_water_coulomb_radii(atoms.get_chemical_symbols())
    )
    continuum = SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=radii,
        transition_width_angstrom2=TRANSITION_WIDTH_ANGSTROM2,
        surface_lmax=SURFACE_LMAX,
        exposure_lmax=EXPOSURE_LMAX,
        exposure_radial_quadrature_order=EXPOSURE_RADIAL_QUADRATURE_ORDER,
        source_radial_quadrature_order=SOURCE_RADIAL_QUADRATURE_ORDER,
        green_radial_quadrature_order=GREEN_RADIAL_QUADRATURE_ORDER,
        dtype=model.dtype,
        device=model.device,
        scalar_id=SCALAR_ID,
    )
    scalar = build_disabled_operational_electrostatic_scalar(
        model,
        continuum,
        atoms,
        scalar_id=SCALAR_ID,
        profile_id=PROFILE_ID,
    )
    return profile, model, continuum, scalar.equation, scalar


def build_system(atoms: Atoms, checkpoint: Path, device: str):
    """Load the model and build the registered operational system."""

    return build_system_with_model(atoms, build_model(checkpoint, device))


def root_context(
    geometry: Atoms,
    label: str,
    *,
    box_length: int = 0,
    system_id: str = "water",
) -> str:
    """Return a geometry-bound context; ``box_length`` is compatibility-only."""

    del box_length
    normalized_system = str(system_id).strip()
    normalized_label = str(label).strip()
    if not normalized_system or "/" in normalized_system:
        raise ValueError("system_id must be a non-empty path-segment identifier.")
    if not normalized_label:
        raise ValueError("label must be non-empty.")
    return (
        "operational-analytic-original-source-harmonic-v1/"
        f"{normalized_system}/{normalized_label}/{geometry_sha256(geometry)}"
    )


def identity_record(model, continuum, equation, scalar) -> dict[str, object]:
    """Return the complete model/continuum/equation/scalar binding."""

    runtime = continuum.runtime_provenance()
    return {
        "profile_id": scalar.profile_id,
        "scalar_id": scalar.scalar_id,
        "state_equation_id": equation.state_equation_id,
        "model_provider_id": model.provider_id,
        "model_profile_id": model.model_profile_id,
        "model_provenance_sha256": model.provenance_sha256,
        "model_provenance": model.provenance.metadata(),
        "model_configuration_sha256": model.configuration_sha256(),
        "model_release_contract": model.release_contract.metadata(),
        "continuum_provider_id": continuum.provider_id,
        "continuum_profile_id": continuum.continuum_profile_id,
        "cavity_profile_id": continuum.cavity_profile_id,
        "coupling_id": continuum.coupling_id,
        "continuum_configuration_contract_id": (continuum.configuration_contract_id),
        "continuum_configuration_sha256": continuum.configuration_sha256(),
        "continuum_topology_sha256": continuum.topology_sha256(),
        "continuum_provenance_sha256": continuum.provenance_sha256,
        "continuum_runtime_provenance": dict(runtime),
        "equation_sha256": equation.fingerprint_sha256(),
        "scalar_sha256": scalar.fingerprint_sha256(),
    }


__all__ = [
    "COMMON_REQUIRED_SOURCE_PATHS",
    "DEFAULT_CHECKPOINT",
    "EXPOSURE_LMAX",
    "EXPOSURE_RADIAL_QUADRATURE_ORDER",
    "GREEN_RADIAL_QUADRATURE_ORDER",
    "NO_CAPABILITIES",
    "PROFILE_ID",
    "SCALAR_ID",
    "SOURCE_RADIAL_QUADRATURE_ORDER",
    "SURFACE_LMAX",
    "TRANSITION_WIDTH_ANGSTROM2",
    "build_model",
    "build_system",
    "build_system_with_model",
    "identity_record",
    "root_context",
]
