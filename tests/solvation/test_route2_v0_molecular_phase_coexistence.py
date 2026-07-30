from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0FrozenMaceGaussianSource,
    Route2V0MolecularConfigurations,
    Route2V0MolecularExternalPotential,
    Route2V0MolecularSolventReference,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_ideal_gas import (
    RIGID_MOLECULAR_ORIENTATION_MEASURE,
    Route2V0MolecularConfigurationQuadrature,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_phase_coexistence import (
    V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION,
    analyze_route2_v0_molecular_homogeneous_phase_coexistence,
    evaluate_route2_v0_molecular_homogeneous_phase,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_thermodynamics import (
    molecular_hnc_bulk_functional_pressure_hartree_per_bohr3,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_weighted_density_bridge import (
    Route2V0MolecularCenterProjection,
    Route2V0MolecularWeightedDensityBridgeAsset,
    Route2V0MolecularWeightedDensityBridgeFunctional,
    Route2V0PeriodicWeightedDensityKernel,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_promolecular_density import (
    Route2V0PromolecularDensityTable,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_site_hnc import (
    Route2V0SiteHNCAsset,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)

_BULK_DENSITY_BOHR3 = 0.02
_KBT_HARTREE = 0.1
_TARGET_PRESSURE_FRACTION = 0.05
# This synthetic-only coefficient makes the analytic homogeneous cubic-plus-
# quartic control have a low-density stationary phase degenerate with x = 1.
# It is deliberately not an input to the Route-2 implementation or a physical
# liquid parameter; it only gives the phase gate a known positive control.
_COEXISTENT_DIMENSIONLESS_QUARTIC = 7.119206180987243


def _translation_invariant_pure_liquid_functional(
    *,
    quartic_scale: float = 1.0,
    external_energy_hartree: float = 0.0,
) -> Route2V0MolecularWeightedDensityBridgeFunctional:
    """Build a source-free periodic molecular scalar for phase-gate controls."""

    point_count = 16
    grid = RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.ones(3),
        shape=(point_count, 1, 1),
    )
    table = Route2V0PromolecularDensityTable(
        radial_grid_bohr=np.array([0.0, 1.0, 2.0]),
        densities_by_atomic_number={1: np.zeros(3)},
        table_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )
    translations = np.zeros((point_count, 3))
    translations[:, 0] = np.arange(point_count, dtype=float)
    configurations = Route2V0MolecularConfigurations(
        translations_bohr=translations,
        rotations=np.repeat(np.eye(3)[None], point_count, axis=0),
    )
    solvent = Route2V0MolecularSolventReference(
        atomic_numbers=np.array([1]),
        site_charges_e=np.array([0.0]),
        reference_positions_bohr=np.zeros((1, 3)),
        provenance_label="synthetic translation-invariant pure-liquid control",
    )
    external_values = np.full(point_count, external_energy_hartree)
    external = Route2V0MolecularExternalPotential(
        integration_grid=grid,
        promolecular_table=table,
        solute_atomic_numbers=np.array([1]),
        solute_atom_positions_angstrom=np.zeros((1, 3)),
        mace_source=Route2V0FrozenMaceGaussianSource(
            density_coefficients=np.zeros((1, 4)),
            atom_positions_angstrom=np.zeros((1, 3)),
        ),
        solvent=solvent,
        configurations=configurations,
        solute_reference_density_e_per_bohr3=np.zeros(grid.shape),
        pauli_repulsion_hartree=np.zeros(point_count),
        electrostatic_energy_hartree=external_values,
        external_potential_hartree=external_values,
    )
    hnc_asset = Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("X",),
        bulk_number_density_bohr3=np.array([_BULK_DENSITY_BOHR3]),
        direct_correlation_dimensionless=np.zeros((1, 1, *grid.shape)),
        kbt_hartree=_KBT_HARTREE,
    )
    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=configurations,
        phase_space_weights_bohr3=np.full(
            point_count,
            RIGID_MOLECULAR_ORIENTATION_MEASURE,
        ),
    )
    site_occupancy = np.zeros((1, *grid.shape, point_count))
    centre_occupancy = np.zeros((*grid.shape, point_count))
    for configuration_index in range(point_count):
        site_occupancy[0, configuration_index, 0, 0, configuration_index] = 1.0
        centre_occupancy[configuration_index, 0, 0, configuration_index] = 1.0
    projection = Route2V0MolecularSiteProjection(
        quadrature=quadrature,
        external_potential=external,
        site_hnc_asset=hnc_asset,
        solvent_site_type_indices=np.array([0]),
        site_occupancy_weights=site_occupancy,
    )
    hnc = Route2V0MolecularSiteHNCFunctional(projection)
    centre = Route2V0MolecularCenterProjection(
        projection=projection,
        center_occupancy_weights=centre_occupancy,
    )
    kernel_values = np.zeros(grid.shape)
    kernel_values[0, 0, 0] = 1.0 / grid.volume_element_bohr3
    kernel = Route2V0PeriodicWeightedDensityKernel(
        grid=grid,
        kernel_bohr_minus3=kernel_values,
    )
    hnc_pressure = molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(hnc)
    quartic = (
        quartic_scale
        * _COEXISTENT_DIMENSIONLESS_QUARTIC
        * _KBT_HARTREE
        / _BULK_DENSITY_BOHR3**5
    )
    bridge = Route2V0MolecularWeightedDensityBridgeAsset.from_pure_solvent_anchors(
        center_projection=centre,
        kernel=kernel,
        hnc_bulk_pressure_hartree_per_bohr3=hnc_pressure,
        target_bulk_pressure_hartree_per_bohr3=(
            _TARGET_PRESSURE_FRACTION * _BULK_DENSITY_BOHR3 * _KBT_HARTREE
        ),
        quartic_coefficient_hartree_bohr15=quartic,
        target_surface_tension_hartree_per_bohr2=0.001,
        pure_solvent_certificate_sha256="2" * 64,
    )
    return Route2V0MolecularWeightedDensityBridgeFunctional(
        hnc_functional=hnc,
        bridge_asset=bridge,
    )


def test_homogeneous_phase_derivatives_come_from_the_exact_molecular_scalar():
    functional = _translation_invariant_pure_liquid_functional()
    scale = 0.26
    step = 1.0e-5
    phase = evaluate_route2_v0_molecular_homogeneous_phase(
        functional,
        density_scale=scale,
        curvature_tolerance_hartree_per_bohr3=1.0e-14,
    )
    upper = evaluate_route2_v0_molecular_homogeneous_phase(
        functional,
        density_scale=scale + step,
        curvature_tolerance_hartree_per_bohr3=1.0e-14,
    )
    lower = evaluate_route2_v0_molecular_homogeneous_phase(
        functional,
        density_scale=scale - step,
        curvature_tolerance_hartree_per_bohr3=1.0e-14,
    )

    assert phase.construction == V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION
    assert phase.gradient_uniformity_residual == pytest.approx(0.0, abs=1.0e-13)
    assert phase.grand_potential_density_hartree_per_bohr3 == pytest.approx(
        phase.grand_potential_hartree / 16.0,
        rel=0.0,
        abs=1.0e-15,
    )
    assert (
        upper.grand_potential_density_hartree_per_bohr3
        - lower.grand_potential_density_hartree_per_bohr3
    ) / (2.0 * step) == pytest.approx(
        phase.directional_derivative_density_hartree_per_bohr3,
        rel=3.0e-9,
        abs=2.0e-12,
    )
    assert (
        upper.directional_derivative_density_hartree_per_bohr3
        - lower.directional_derivative_density_hartree_per_bohr3
    ) / (2.0 * step) == pytest.approx(
        phase.curvature_density_hartree_per_bohr3,
        rel=3.0e-8,
        abs=2.0e-10,
    )


def test_phase_gate_requires_full_stable_coexistence_not_just_a_pressure_identity():
    coexistent = analyze_route2_v0_molecular_homogeneous_phase_coexistence(
        _translation_invariant_pure_liquid_functional(),
        minimum_density_scale=1.0e-5,
        root_sample_count=513,
        directional_derivative_tolerance_hartree_per_bohr3=1.0e-13,
        curvature_tolerance_hartree_per_bohr3=1.0e-13,
        coexistence_tolerance_hartree_per_bohr3=1.0e-12,
    )
    assert coexistent.passes is True
    assert coexistent.gas_phase is not None
    assert coexistent.gas_phase.density_scale == pytest.approx(
        0.0457932428,
        rel=2.0e-8,
    )
    assert coexistent.gas_phase.classification == "stable"
    assert len(coexistent.gas_candidates) == 2
    assert (
        coexistent.grand_potential_density_difference_hartree_per_bohr3
        == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
    )

    noncoexistent = analyze_route2_v0_molecular_homogeneous_phase_coexistence(
        _translation_invariant_pure_liquid_functional(quartic_scale=0.5),
        minimum_density_scale=1.0e-5,
        root_sample_count=513,
        directional_derivative_tolerance_hartree_per_bohr3=1.0e-13,
        curvature_tolerance_hartree_per_bohr3=1.0e-13,
        coexistence_tolerance_hartree_per_bohr3=1.0e-12,
    )
    assert noncoexistent.passes is False
    assert noncoexistent.gas_phase is not None
    assert noncoexistent.gas_phase.classification == "stable"
    assert (
        noncoexistent.grand_potential_density_difference_hartree_per_bohr3 is not None
    )
    assert abs(noncoexistent.grand_potential_density_difference_hartree_per_bohr3) > (
        noncoexistent.coexistence_tolerance_hartree_per_bohr3
    )


def test_phase_gate_refuses_to_relabel_a_solute_external_potential_as_pure_liquid():
    functional = _translation_invariant_pure_liquid_functional(
        external_energy_hartree=1.0e-7,
    )
    with pytest.raises(ValueError, match="zero-external-potential pure-liquid scalar"):
        analyze_route2_v0_molecular_homogeneous_phase_coexistence(functional)


def test_phase_gate_argument_ranges_fail_closed():
    functional = _translation_invariant_pure_liquid_functional()
    with pytest.raises(ValueError, match="0 < minimum < upper < 1"):
        analyze_route2_v0_molecular_homogeneous_phase_coexistence(
            functional,
            minimum_density_scale=0.8,
            gas_search_upper_density_scale=0.7,
        )
    with pytest.raises(ValueError, match="at least three"):
        analyze_route2_v0_molecular_homogeneous_phase_coexistence(
            functional,
            root_sample_count=2,
        )
    with pytest.raises(ValueError, match="positive"):
        evaluate_route2_v0_molecular_homogeneous_phase(
            functional,
            density_scale=0.0,
        )
