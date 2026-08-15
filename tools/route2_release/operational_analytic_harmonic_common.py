"""Shared identity/build helpers for the disabled operational harmonic scalar."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
import numpy as np

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
from maple.solvation.coupling.fixed_point import roots_numerically_equivalent
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
MINIMUM_RELATIVE_BASIS_SINGULAR_VALUE = 1.0e-10
MAXIMUM_SURFACE_CONDITION_NUMBER = 1.0e12
MINIMUM_SURFACE_EIGENVALUE = 1.0e-12
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


def state_record(
    state, scalar, geometry: Atoms, *, evaluation=None
) -> dict[str, object]:
    """Serialize one root and its exact operational scalar decomposition."""

    evaluated = (
        scalar.evaluate_energy_components(geometry, state.y)
        if evaluation is None
        else evaluation
    )
    if evaluated.scalar_id != scalar.scalar_id:
        raise ValueError("State-record scalar evaluation has the wrong identity.")
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


def cold_warm_record(
    cold,
    warm,
    scalar,
    atoms: Atoms,
    *,
    cold_evaluation=None,
    warm_evaluation=None,
) -> dict[str, object]:
    """Record independent-root equivalence without promoting a capability."""

    cold_scalar = (
        scalar.evaluate_energy_components(atoms, cold.y)
        if cold_evaluation is None
        else cold_evaluation
    )
    warm_scalar = (
        scalar.evaluate_energy_components(atoms, warm.y)
        if warm_evaluation is None
        else warm_evaluation
    )
    if (
        cold_scalar.scalar_id != scalar.scalar_id
        or warm_scalar.scalar_id != scalar.scalar_id
    ):
        raise ValueError("Cold/warm scalar evaluation has the wrong identity.")
    cold_source = cold.source_array()
    warm_source = warm.source_array()
    source_difference = float(np.linalg.norm(cold_source - warm_source))
    source_relative = source_difference / max(
        float(np.linalg.norm(cold_source)),
        float(np.linalg.norm(warm_source)),
        1.0e-15,
    )
    field_difference = float(np.linalg.norm(cold.field_array() - warm.field_array()))
    field_relative = field_difference / max(
        float(np.linalg.norm(cold.field_array())),
        float(np.linalg.norm(warm.field_array())),
        1.0e-15,
    )
    energy_difference = abs(cold_scalar.total_energy - warm_scalar.total_energy)
    return {
        "contract": "route2-root-equivalence-v1",
        "numerically_equivalent": roots_numerically_equivalent(cold, warm),
        "source_l2_difference": source_difference,
        "source_relative_difference": source_relative,
        "field_l2_difference": field_difference,
        "field_relative_difference": field_relative,
        "energy_abs_difference_eV": energy_difference,
        "gates": {
            "source_relative_le_1e-8": source_relative <= 1.0e-8,
            "field_relative_le_1e-8": field_relative <= 1.0e-8,
            "energy_le_1e-8_eV": energy_difference <= 1.0e-8,
        },
        "cold": state_record(cold, scalar, atoms, evaluation=cold_scalar),
        "warm": state_record(warm, scalar, atoms, evaluation=warm_scalar),
    }


def scalar_identity_record(continuum, scalar, atoms: Atoms, state) -> dict[str, object]:
    """Check the registered harmonic scalar and original-source embedding."""

    source = state.source_array()
    field = state.field_array()
    functional = continuum.energy_eV(atoms, source)
    half_coupling = 0.5 * float(scalar.metric.pair(source, field))
    error = abs(functional - half_coupling)
    missing_maximum = float(np.max(np.abs(source[:, (1, 5, 6, 7)])))
    return {
        "functional_energy_eV": functional,
        "half_coupling_energy_eV": half_coupling,
        "absolute_error_eV": error,
        "missing_radial_block_max_abs": missing_maximum,
        "gate_passed": bool(error <= 1.0e-10 and missing_maximum <= 1.0e-8),
    }


def harmonic_domain_record(continuum, atoms: Atoms) -> dict[str, object]:
    """Return raw rank/conditioning margins for one geometry.

    The production scalar already fails closed on these conditions.  The
    explicit record lets an independent evidence verifier distinguish a clean
    margin from a merely successful solve without changing the scalar graph.
    """

    matrices = continuum.debug_geometry_matrices(atoms)
    weighted_basis = np.asarray(matrices["weighted_basis"], dtype=float)
    surface = np.asarray(matrices["surface_operator"], dtype=float)
    if (
        weighted_basis.ndim != 2
        or surface.ndim != 2
        or surface.shape[0] != surface.shape[1]
        or not np.all(np.isfinite(weighted_basis))
        or not np.all(np.isfinite(surface))
    ):
        raise RuntimeError("Harmonic geometry diagnostics are non-finite or invalid.")
    singular_values = np.linalg.svd(weighted_basis, compute_uv=False)
    eigenvalues = np.linalg.eigvalsh(0.5 * (surface + surface.T))
    relative_basis = float(singular_values[-1] / singular_values[0])
    minimum_eigenvalue = float(eigenvalues[0])
    condition = float(eigenvalues[-1] / minimum_eigenvalue)
    positions = np.asarray(atoms.positions, dtype=float)
    if len(positions) < 2:
        raise RuntimeError("The frozen Tier-F panel requires at least two centres.")
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    minimum_center_distance = float(
        np.min(distances[np.triu_indices(len(positions), k=1)])
    )
    gates = {
        "distinct_centres": minimum_center_distance > 0.0,
        "relative_basis_singular_value_gt_1e-10": (
            relative_basis > MINIMUM_RELATIVE_BASIS_SINGULAR_VALUE
        ),
        "surface_minimum_eigenvalue_gt_1e-12": (
            minimum_eigenvalue > MINIMUM_SURFACE_EIGENVALUE
        ),
        "surface_condition_number_le_1e12": (
            np.isfinite(condition) and condition <= MAXIMUM_SURFACE_CONDITION_NUMBER
        ),
    }
    return {
        "minimum_center_distance_A": minimum_center_distance,
        "weighted_basis_shape": list(weighted_basis.shape),
        "weighted_basis_rank": int(np.linalg.matrix_rank(weighted_basis)),
        "minimum_basis_singular_value": float(singular_values[-1]),
        "minimum_relative_basis_singular_value": relative_basis,
        "surface_dimension": int(surface.shape[0]),
        "surface_minimum_eigenvalue": minimum_eigenvalue,
        "surface_maximum_eigenvalue": float(eigenvalues[-1]),
        "surface_condition_number": condition,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


__all__ = [
    "COMMON_REQUIRED_SOURCE_PATHS",
    "DEFAULT_CHECKPOINT",
    "EXPOSURE_LMAX",
    "EXPOSURE_RADIAL_QUADRATURE_ORDER",
    "GREEN_RADIAL_QUADRATURE_ORDER",
    "MAXIMUM_SURFACE_CONDITION_NUMBER",
    "MINIMUM_RELATIVE_BASIS_SINGULAR_VALUE",
    "MINIMUM_SURFACE_EIGENVALUE",
    "NO_CAPABILITIES",
    "PROFILE_ID",
    "SCALAR_ID",
    "SOURCE_RADIAL_QUADRATURE_ORDER",
    "SURFACE_LMAX",
    "TRANSITION_WIDTH_ANGSTROM2",
    "build_model",
    "build_system",
    "build_system_with_model",
    "cold_warm_record",
    "harmonic_domain_record",
    "identity_record",
    "root_context",
    "scalar_identity_record",
    "state_record",
]
