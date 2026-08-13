"""Shared real-stack setup for disabled fixed-box590 water diagnostics."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
import numpy as np

from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES,
)
from maple.solvation.api.profiles import (
    DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS,
    get_solvation_profile,
)
from maple.solvation.continuum import build_water_radial_gto_cpcm_590_candidate
from maple.solvation.coupling.energy import OperationalElectrostaticScalar
from maple.solvation.coupling.exact_gto import MACE_POLAR_RADIAL_GTO_COUPLING_ID
from maple.solvation.coupling.fixed_point import roots_numerically_equivalent
from maple.solvation.coupling.spaces import (
    LinearChargeCoordinates,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import (
    ReducedStateEquation,
    geometry_sha256,
)
from maple.solvation.models import (
    ElectronicResponseEquationAdapter,
    VacuumScalarEquationAdapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)

DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
COMMON_REQUIRED_SOURCE_PATHS = (
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/api/state_registry.py",
    "maple/solvation/continuum/conjugate_fixed_topology_cpcm.py",
    "maple/solvation/coupling/adjoint.py",
    "maple/solvation/coupling/energy.py",
    "maple/solvation/coupling/fixed_point.py",
    "maple/solvation/coupling/linearization.py",
    "maple/solvation/coupling/state_equation.py",
    "maple/solvation/models/equation_adapter.py",
    "maple/solvation/models/fixed_box_evaluator.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_feature_vjp.py",
    "maple/solvation/release/evidence.py",
    "maple/solvation/release/pes_validation.py",
    "maple/function/calculator/mace/_macepol_long_range.py",
    "maple/function/route2_smd_profiles.py",
    "tools/route2_release/fixedbox590_water_common.py",
)


def water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]],
            dtype=float,
        ),
        info={"charge": 0, "mult": 1},
    )


def root_context(geometry: Atoms, label: str, *, box_length: int = 40) -> str:
    if box_length not in DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS:
        raise ValueError(f"Unregistered fixed-box length: {box_length} Angstrom.")
    return (
        f"fixedbox{box_length}-cpcm590-water-pes-v1/"
        f"{label}/{geometry_sha256(geometry)}"
    )


def build_system(atoms: Atoms, checkpoint: Path, device: str, *, box_length: int = 40):
    try:
        profile_id = DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS[box_length]
        evaluator_profile = MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[box_length]
    except KeyError as exc:
        raise ValueError(
            f"Unregistered fixed-box length: {box_length} Angstrom."
        ) from exc
    profile = get_solvation_profile(profile_id)
    if profile.enabled or profile.capabilities.enabled_tiers:
        raise RuntimeError("The diagnostic profile must remain completely disabled.")
    model = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=device,
        long_range_evaluator_profile=evaluator_profile,
    )
    continuum = build_water_radial_gto_cpcm_590_candidate(atoms.get_chemical_symbols())
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
        profile_id=profile_id,
    )
    return profile, model, continuum, equation, scalar


def state_record(state, scalar, geometry: Atoms) -> dict[str, object]:
    evaluated = scalar.evaluate(geometry, state.y)
    return {
        "initialization": state.initialization,
        "iterations": len(state.iterations) - 1,
        "root_context_id": state.root_context_id,
        "root_hash": state.root_hash,
        "primal_tolerance": state.primal_tolerance,
        "actual_unmixed_residual_norm": state.actual_unmixed_residual_norm,
        "source": state.source_array().tolist(),
        "field": state.field_array().tolist(),
        "vacuum_energy_eV": evaluated.vacuum_energy,
        "continuum_energy_eV": evaluated.continuum_energy,
        "total_energy_eV": evaluated.total_energy,
    }


def cold_warm_record(cold, warm, scalar, atoms: Atoms) -> dict[str, object]:
    cold_scalar = scalar.evaluate(atoms, cold.y)
    warm_scalar = scalar.evaluate(atoms, warm.y)
    cold_source = cold.source_array()
    warm_source = warm.source_array()
    source_difference = float(np.linalg.norm(cold_source - warm_source))
    source_relative = source_difference / max(
        float(np.linalg.norm(cold_source)),
        float(np.linalg.norm(warm_source)),
        1.0e-15,
    )
    field_difference = float(np.linalg.norm(cold.field_array() - warm.field_array()))
    energy_difference = abs(cold_scalar.total_energy - warm_scalar.total_energy)
    return {
        "contract": "route2-root-equivalence-v1",
        "numerically_equivalent": roots_numerically_equivalent(cold, warm),
        "source_l2_difference": source_difference,
        "source_relative_difference": source_relative,
        "field_l2_difference": field_difference,
        "energy_abs_difference_eV": energy_difference,
        "gates": {
            "source_relative_le_1e-8": source_relative <= 1.0e-8,
            "energy_le_1e-8_eV": energy_difference <= 1.0e-8,
        },
        "cold": state_record(cold, scalar, atoms),
        "warm": state_record(warm, scalar, atoms),
    }


def identity_record(model, continuum, equation, scalar) -> dict[str, object]:
    return {
        "profile_id": scalar.profile_id,
        "scalar_id": scalar.scalar_id,
        "state_equation_id": equation.state_equation_id,
        "model_provider_id": model.provider_id,
        "model_profile_id": model.model_profile_id,
        "model_provenance_sha256": model.provenance_sha256,
        "model_provenance": model.provenance.metadata(),
        "model_configuration_sha256": model.configuration_sha256(),
        "long_range_evaluator_provenance": (
            model._calculator.long_range_evaluator_provenance
        ),
        "continuum_provider_id": continuum.provider_id,
        "continuum_profile_id": continuum.continuum_profile_id,
        "cavity_profile_id": continuum.cavity_profile_id,
        "coupling_id": continuum.coupling_id,
        "continuum_configuration_contract_id": (continuum.configuration_contract_id),
        "continuum_configuration_sha256": continuum.configuration_sha256(),
        "continuum_provenance_sha256": continuum.provenance_sha256,
        "continuum_runtime_provenance": dict(continuum.runtime_provenance),
        "equation_sha256": equation.fingerprint_sha256(),
        "scalar_sha256": scalar.fingerprint_sha256(),
    }


__all__ = [
    "COMMON_REQUIRED_SOURCE_PATHS",
    "DEFAULT_CHECKPOINT",
    "build_system",
    "cold_warm_record",
    "identity_record",
    "root_context",
    "state_record",
    "water",
]
