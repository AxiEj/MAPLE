from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0FrozenMaceGaussianSource,
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
    evaluate_route2_v0_molecular_external_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_ideal_gas import (
    RIGID_MOLECULAR_ORIENTATION_MEASURE,
    Route2V0MolecularConfigurationQuadrature,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_stability import (
    certify_route2_v0_molecular_hessian_stability,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_thermodynamics import (
    molecular_hnc_bulk_functional_pressure_hartree_per_bohr3,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_weighted_density_bridge import (
    V0_MOLECULAR_CENTER_PROJECTION_CONSTRUCTION,
    V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION,
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


def _base_functional(*, direct_correlation: np.ndarray | None = None):
    grid = RegularCartesianGrid(
        origin_bohr=np.array([-0.5, 0.0, 0.0]),
        spacing_bohr=np.ones(3),
        shape=(2, 1, 1),
    )
    table = Route2V0PromolecularDensityTable(
        radial_grid_bohr=np.array([0.0, 0.5, 1.0, 1.5]),
        densities_by_atomic_number={1: np.array([1.0, 0.5, 0.1, 0.0])},
        table_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )
    configurations = Route2V0MolecularConfigurations(
        translations_bohr=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        rotations=np.repeat(np.eye(3)[None], 2, axis=0),
    )
    solvent = Route2V0MolecularSolventReference(
        atomic_numbers=np.array([1]),
        site_charges_e=np.array([0.0]),
        reference_positions_bohr=np.zeros((1, 3)),
        provenance_label="synthetic molecular weighted-density bridge control",
    )
    external = evaluate_route2_v0_molecular_external_potential(
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
    )
    if direct_correlation is None:
        direct_correlation = np.zeros((1, 1, *grid.shape))
    asset = Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("X",),
        bulk_number_density_bohr3=np.array([0.02]),
        direct_correlation_dimensionless=direct_correlation,
        kbt_hartree=0.1,
    )
    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=configurations,
        phase_space_weights_bohr3=np.full(2, RIGID_MOLECULAR_ORIENTATION_MEASURE),
    )
    occupancy = np.zeros((1, *grid.shape, 2))
    occupancy[0, 0, 0, 0, 0] = 1.0
    occupancy[0, 1, 0, 0, 1] = 1.0
    projection = Route2V0MolecularSiteProjection(
        quadrature=quadrature,
        external_potential=external,
        site_hnc_asset=asset,
        solvent_site_type_indices=np.array([0]),
        site_occupancy_weights=occupancy,
    )
    return Route2V0MolecularSiteHNCFunctional(projection)


def _center_projection(base: Route2V0MolecularSiteHNCFunctional):
    grid = base.projection.site_hnc_asset.grid
    occupancy = np.zeros((*grid.shape, base.projection.quadrature.configuration_count))
    occupancy[0, 0, 0, 0] = 1.0
    occupancy[1, 0, 0, 1] = 1.0
    return Route2V0MolecularCenterProjection(
        projection=base.projection,
        center_occupancy_weights=occupancy,
    )


def _bridge_functional(*, direct_correlation: np.ndarray | None = None):
    base = _base_functional(direct_correlation=direct_correlation)
    center = _center_projection(base)
    kernel = Route2V0PeriodicWeightedDensityKernel(
        grid=center.grid,
        kernel_bohr_minus3=np.array([[[0.5]], [[0.5]]]),
    )
    hnc_pressure = molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(base)
    asset = Route2V0MolecularWeightedDensityBridgeAsset.from_pure_solvent_anchors(
        center_projection=center,
        kernel=kernel,
        hnc_bulk_pressure_hartree_per_bohr3=hnc_pressure,
        target_bulk_pressure_hartree_per_bohr3=0.0002,
        quartic_coefficient_hartree_bohr15=2.0e7,
        target_surface_tension_hartree_per_bohr2=0.001,
        pure_solvent_certificate_sha256="2" * 64,
    )
    return Route2V0MolecularWeightedDensityBridgeFunctional(
        hnc_functional=base,
        bridge_asset=asset,
    )


def test_center_projection_and_periodic_kernel_preserve_the_exact_pairings():
    functional = _bridge_functional()
    center = functional.center_projection
    bulk = functional.projection.uniform_configuration_density_bohr3
    density = bulk * np.array([1.2, 0.8])
    direction = bulk * np.array([0.2, -0.1])
    field = np.array([[[0.3]], [[-0.4]]])

    assert center.construction == V0_MOLECULAR_CENTER_PROJECTION_CONSTRUCTION
    np.testing.assert_allclose(
        center.project_configuration_density(np.full(2, bulk)),
        center.molecular_bulk_number_density_bohr3,
        rtol=0.0,
        atol=2.0e-16,
    )
    lhs = center.grid.volume_element_bohr3 * np.sum(
        field * center.project_configuration_direction(direction)
    )
    rhs = np.sum(
        functional.projection.quadrature.phase_space_weights_bohr3
        * center.center_field_adjoint_dimensionless(field)
        * direction
    )
    assert lhs == pytest.approx(rhs, rel=2.0e-13, abs=2.0e-15)

    kernel = functional.bridge_asset.kernel
    other = np.array([[[0.7]], [[-0.2]]])
    left = center.grid.volume_element_bohr3 * np.sum(field * kernel.convolve(other))
    right = center.grid.volume_element_bohr3 * np.sum(other * kernel.convolve(field))
    assert left == pytest.approx(right, rel=2.0e-13, abs=2.0e-15)
    np.testing.assert_allclose(
        functional.weighted_molecular_density(density),
        np.full(center.grid.shape, center.molecular_bulk_number_density_bohr3),
        rtol=0.0,
        atol=2.0e-16,
    )


def test_bridge_cubic_coefficient_and_pressure_are_same_functional_identities():
    functional = _bridge_functional()
    asset = functional.bridge_asset
    density = asset.molecular_bulk_number_density_bohr3
    expected_cubic = (
        asset.hnc_bulk_pressure_hartree_per_bohr3
        - asset.target_bulk_pressure_hartree_per_bohr3
    ) / density**3

    assert functional.construction == V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION
    assert asset.cubic_coefficient_hartree_bohr6 == pytest.approx(expected_cubic)
    assert asset.coexistence_pressure_hartree_per_bohr3 == pytest.approx(
        asset.target_bulk_pressure_hartree_per_bohr3,
        rel=2.0e-14,
        abs=2.0e-16,
    )
    assert functional.bulk_functional_pressure_hartree_per_bohr3 == pytest.approx(
        asset.target_bulk_pressure_hartree_per_bohr3,
        rel=2.0e-14,
        abs=2.0e-16,
    )


def test_bridge_gradient_and_hessian_are_the_derivatives_of_one_scalar():
    direct = np.zeros((1, 1, 2, 1, 1))
    direct[0, 0, :, 0, 0] = np.array([0.02, 0.02])
    functional = _bridge_functional(direct_correlation=direct)
    bulk = functional.projection.uniform_configuration_density_bohr3
    density = bulk * np.array([1.12, 0.91])
    direction = bulk * np.array([0.18, -0.12])
    other_direction = bulk * np.array([-0.11, 0.14])
    weights = functional.projection.quadrature.phase_space_weights_bohr3

    step = 1.0e-5
    finite_difference = (
        functional.grand_potential_hartree(density + step * direction)
        - functional.grand_potential_hartree(density - step * direction)
    ) / (2.0 * step)
    analytic = functional.kbt_hartree * np.sum(
        weights * functional.dimensionless_gradient(density) * direction
    )
    assert finite_difference == pytest.approx(analytic, rel=4.0e-9, abs=2.0e-14)

    left_action = functional.dimensionless_hessian_matvec(density, direction)
    right_action = functional.dimensionless_hessian_matvec(density, other_direction)
    left_pairing = np.sum(weights * direction * right_action)
    right_pairing = np.sum(weights * other_direction * left_action)
    assert left_pairing == pytest.approx(right_pairing, rel=2.0e-12, abs=2.0e-15)

    gradient_step = 1.0e-5
    gradient_derivative = (
        functional.dimensionless_gradient(density + gradient_step * direction)
        - functional.dimensionless_gradient(density - gradient_step * direction)
    ) / (2.0 * gradient_step)
    np.testing.assert_allclose(
        gradient_derivative,
        left_action,
        rtol=5.0e-8,
        atol=5.0e-12,
    )

    scalar_second_derivative = (
        functional.grand_potential_hartree(density + 2.0e-3 * direction)
        - 2.0 * functional.grand_potential_hartree(density)
        + functional.grand_potential_hartree(density - 2.0e-3 * direction)
    ) / (2.0e-3) ** 2
    assert scalar_second_derivative == pytest.approx(
        functional.hessian_quadratic_hartree(density, direction),
        rel=5.0e-5,
        abs=3.0e-13,
    )


def test_bridge_hessian_stability_certificate_uses_the_same_scalar_hessian_action():
    functional = _bridge_functional()
    state = functional.solve_picard(
        residual_tolerance=1.0e-11,
        picard_mixing=0.05,
        max_iterations=3000,
    )
    certificate = certify_route2_v0_molecular_hessian_stability(
        functional,
        state.configuration_density_bohr3,
        maximum_dimension=2,
    )
    density = state.configuration_density_bohr3
    direction = density * np.array([0.17, -0.09])
    weights = functional.projection.quadrature.phase_space_weights_bohr3
    transformed_direction = np.sqrt(weights) * direction

    assert (
        certificate.reciprocity_relative_frobenius_residual
        <= certificate.reciprocity_relative_tolerance
    )
    assert certificate.classification in {
        "positive-definite",
        "negative-mode",
        "numerically-singular",
    }
    assert transformed_direction @ certificate.weighted_hessian_matrix @ (
        transformed_direction
    ) == pytest.approx(
        functional.hessian_quadratic_hartree(density, direction)
        / functional.kbt_hartree,
        rel=2.0e-12,
        abs=2.0e-15,
    )

def test_bridge_state_and_picard_solver_keep_all_four_scalar_components():
    functional = _bridge_functional()
    state = functional.solve_picard(
        residual_tolerance=1.0e-11,
        picard_mixing=0.05,
        max_iterations=3000,
    )

    assert state.residual_inf < 1.0e-11
    assert state.grand_potential_hartree == pytest.approx(
        state.ideal_contribution_hartree
        + state.hnc_excess_contribution_hartree
        + state.bridge_contribution_hartree
        + state.external_contribution_hartree,
        rel=2.0e-13,
        abs=2.0e-15,
    )
    assert state.bridge_contribution_hartree != pytest.approx(0.0)


def test_bridge_fails_closed_for_nonphysical_or_mismatched_source_contracts():
    functional = _bridge_functional()
    asset = functional.bridge_asset
    center = functional.center_projection

    with pytest.raises(ValueError, match="cubic coefficient"):
        replace(
            asset,
            cubic_coefficient_hartree_bohr6=1.01
            * asset.cubic_coefficient_hartree_bohr6,
        )
    with pytest.raises(ValueError, match="Target pure-solvent pressure"):
        replace(
            asset,
            target_bulk_pressure_hartree_per_bohr3=(
                asset.hnc_bulk_pressure_hartree_per_bohr3
            ),
        )
    with pytest.raises(ValueError, match="SHA256"):
        replace(asset, pure_solvent_certificate_sha256="not-a-digest")
    with pytest.raises(ValueError, match="integral one"):
        Route2V0PeriodicWeightedDensityKernel(
            grid=center.grid,
            kernel_bohr_minus3=np.ones(center.grid.shape),
        )
    nonreciprocal_grid = RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.ones(3),
        shape=(3, 1, 1),
    )
    with pytest.raises(ValueError, match="reciprocal"):
        Route2V0PeriodicWeightedDensityKernel(
            grid=nonreciprocal_grid,
            kernel_bohr_minus3=np.array([[[0.6]], [[0.3]], [[0.1]]]),
        )

    other_base = _base_functional()
    with pytest.raises(ValueError, match="exact projection"):
        Route2V0MolecularWeightedDensityBridgeFunctional(
            hnc_functional=other_base,
            bridge_asset=asset,
        )
    with pytest.raises(ValueError, match="source pressure"):
        Route2V0MolecularWeightedDensityBridgeFunctional(
            hnc_functional=functional.hnc_functional,
            bridge_asset=replace(
                asset,
                hnc_bulk_pressure_hartree_per_bohr3=(
                    asset.hnc_bulk_pressure_hartree_per_bohr3 * 1.0001
                ),
                cubic_coefficient_hartree_bohr6=(
                    (
                        asset.hnc_bulk_pressure_hartree_per_bohr3 * 1.0001
                        - asset.target_bulk_pressure_hartree_per_bohr3
                    )
                    / asset.molecular_bulk_number_density_bohr3**3
                ),
            ),
        )
