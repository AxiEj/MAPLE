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
    V0_MOLECULAR_SITE_HNC_CONSTRUCTION,
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_stability import (
    certify_route2_v0_molecular_hessian_stability,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_thermodynamics import (
    V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS,
    Route2V0MolecularHNCFixedSoluteThermodynamics,
    evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics,
    molecular_hnc_bulk_functional_pressure_hartree_per_bohr3,
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


def _external_potential(
    *,
    solvent: Route2V0MolecularSolventReference | None = None,
):
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
    if solvent is None:
        solvent = Route2V0MolecularSolventReference(
            atomic_numbers=np.array([1]),
            site_charges_e=np.array([0.0]),
            reference_positions_bohr=np.zeros((1, 3)),
            provenance_label="synthetic one-site molecular HNC bridge control",
        )
    return evaluate_route2_v0_molecular_external_potential(
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


def _projection(*, direct_correlation: np.ndarray | None = None):
    external = _external_potential()
    grid = external.integration_grid
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
        configurations=external.configurations,
        phase_space_weights_bohr3=np.full(2, RIGID_MOLECULAR_ORIENTATION_MEASURE),
    )
    occupancy = np.zeros((1, *grid.shape, 2))
    occupancy[0, 0, 0, 0, 0] = 1.0
    occupancy[0, 1, 0, 0, 1] = 1.0
    return Route2V0MolecularSiteProjection(
        quadrature=quadrature,
        external_potential=external,
        site_hnc_asset=asset,
        solvent_site_type_indices=np.array([0]),
        site_occupancy_weights=occupancy,
    )


def _functional(*, direct_correlation: np.ndarray | None = None):
    projection = _projection(direct_correlation=direct_correlation)
    return Route2V0MolecularSiteHNCFunctional(projection)


def _two_site_one_type_functional():
    external = _external_potential(
        solvent=Route2V0MolecularSolventReference(
            atomic_numbers=np.array([1, 1]),
            site_charges_e=np.array([0.0, 0.0]),
            reference_positions_bohr=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            provenance_label="synthetic two-site molecular pressure control",
        )
    )
    grid = external.integration_grid
    asset = Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("X",),
        bulk_number_density_bohr3=np.array([0.04]),
        direct_correlation_dimensionless=np.zeros((1, 1, *grid.shape)),
        kbt_hartree=0.1,
    )
    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=external.configurations,
        phase_space_weights_bohr3=np.full(2, RIGID_MOLECULAR_ORIENTATION_MEASURE),
    )
    occupancy = np.zeros((1, *grid.shape, 2))
    occupancy[0, 0, 0, 0, 0] = 2.0
    occupancy[0, 1, 0, 0, 1] = 2.0
    return Route2V0MolecularSiteHNCFunctional(
        Route2V0MolecularSiteProjection(
            quadrature=quadrature,
            external_potential=external,
            site_hnc_asset=asset,
            solvent_site_type_indices=np.array([0, 0]),
            site_occupancy_weights=occupancy,
        )
    )


def test_molecular_site_projection_preserves_bulk_and_its_discrete_adjoint_pairing():
    projection = _projection()
    uniform = np.full(
        projection.quadrature.configuration_count,
        projection.uniform_configuration_density_bohr3,
    )
    trial = uniform * np.array([1.2, 0.8])
    field = np.array([[[[0.3]], [[-0.4]]]])

    np.testing.assert_allclose(
        projection.project_configuration_density(uniform),
        0.02,
        rtol=0.0,
        atol=2.0e-16,
    )
    projected_delta = projection.project_configuration_density(
        trial
    ) - projection.project_configuration_density(uniform)
    lhs = projection.site_hnc_asset.grid.volume_element_bohr3 * np.sum(
        field * projected_delta
    )
    rhs = np.sum(
        projection.quadrature.phase_space_weights_bohr3
        * projection.site_field_adjoint_dimensionless(field)
        * (trial - uniform)
    )
    assert lhs == pytest.approx(rhs, rel=2.0e-13, abs=2.0e-15)
    np.testing.assert_array_equal(projection.site_multiplicity, np.array([1]))


def test_molecular_site_hnc_zero_correlation_has_the_analytic_molecular_ideal_state():
    functional = _functional()
    state = functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.5,
        max_iterations=300,
    )
    external = functional.projection.external_potential.external_potential_hartree
    expected = functional.projection.uniform_configuration_density_bohr3 * np.exp(
        -external / functional.projection.site_hnc_asset.kbt_hartree
    )

    assert functional.construction == V0_MOLECULAR_SITE_HNC_CONSTRUCTION
    np.testing.assert_allclose(
        state.configuration_density_bohr3,
        expected,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert state.excess_contribution_hartree == pytest.approx(0.0, abs=2.0e-16)
    assert state.residual_inf < 1.0e-12


def test_molecular_site_hnc_gradient_is_the_derivative_of_the_projected_scalar():
    direct = np.zeros((1, 1, 2, 1, 1))
    direct[0, 0, :, 0, 0] = np.array([0.03, 0.03])
    functional = _functional(direct_correlation=direct)
    bulk = functional.projection.uniform_configuration_density_bohr3
    density = bulk * np.array([1.1, 0.9])
    direction = bulk * np.array([0.2, -0.1])
    step = 1.0e-5

    finite_difference = (
        functional.grand_potential_hartree(density + step * direction)
        - functional.grand_potential_hartree(density - step * direction)
    ) / (2.0 * step)
    analytic = functional.projection.site_hnc_asset.kbt_hartree * np.sum(
        functional.projection.quadrature.phase_space_weights_bohr3
        * functional.dimensionless_gradient(density)
        * direction
    )
    assert finite_difference == pytest.approx(analytic, rel=2.0e-9, abs=2.0e-15)
    state = functional.solve_picard(
        residual_tolerance=1.0e-11,
        picard_mixing=0.4,
        max_iterations=500,
    )
    assert state.residual_inf < 1.0e-11


def test_molecular_site_hnc_hessian_is_self_adjoint_and_the_scalar_second_derivative():
    direct = np.zeros((1, 1, 2, 1, 1))
    direct[0, 0, :, 0, 0] = np.array([0.03, 0.03])
    functional = _functional(direct_correlation=direct)
    bulk = functional.projection.uniform_configuration_density_bohr3
    density = bulk * np.array([1.2, 0.8])
    left_direction = bulk * np.array([0.25, -0.1])
    right_direction = bulk * np.array([-0.15, 0.2])
    hessian_left = functional.dimensionless_hessian_matvec(
        density,
        left_direction,
    )
    hessian_right = functional.dimensionless_hessian_matvec(
        density,
        right_direction,
    )
    weights = functional.projection.quadrature.phase_space_weights_bohr3

    left_pairing = np.sum(weights * left_direction * hessian_right)
    right_pairing = np.sum(weights * right_direction * hessian_left)
    assert left_pairing == pytest.approx(right_pairing, rel=2.0e-13, abs=2.0e-15)

    step = 5.0e-3
    scalar_second_derivative = (
        functional.grand_potential_hartree(density + step * left_direction)
        - 2.0 * functional.grand_potential_hartree(density)
        + functional.grand_potential_hartree(density - step * left_direction)
    ) / step**2
    hessian_quadratic = functional.hessian_quadratic_hartree(
        density,
        left_direction,
    )
    assert scalar_second_derivative == pytest.approx(
        hessian_quadratic,
        rel=2.0e-6,
        abs=3.0e-14,
    )

    gradient_step = 1.0e-5
    gradient_derivative = (
        functional.dimensionless_gradient(density + gradient_step * left_direction)
        - functional.dimensionless_gradient(density - gradient_step * left_direction)
    ) / (2.0 * gradient_step)
    np.testing.assert_allclose(
        gradient_derivative,
        hessian_left,
        rtol=3.0e-9,
        atol=3.0e-13,
    )


def test_molecular_site_hnc_hessian_stability_certificate_is_the_exact_weighted_action():
    functional = _functional()
    state = functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.5,
        max_iterations=300,
    )
    certificate = certify_route2_v0_molecular_hessian_stability(
        functional,
        state.configuration_density_bohr3,
        maximum_dimension=2,
    )
    density = state.configuration_density_bohr3
    direction = density * np.array([0.2, -0.1])
    weights = functional.projection.quadrature.phase_space_weights_bohr3
    transformed_direction = np.sqrt(weights) * direction

    assert certificate.classification == "positive-definite"
    assert certificate.is_positive_definite
    assert (
        certificate.reciprocity_relative_frobenius_residual
        <= certificate.reciprocity_relative_tolerance
    )
    np.testing.assert_allclose(
        certificate.weighted_hessian_matrix,
        np.diag(1.0 / density),
        rtol=2.0e-15,
        atol=3.0e-12,
    )
    assert transformed_direction @ certificate.weighted_hessian_matrix @ (
        transformed_direction
    ) == pytest.approx(
        functional.hessian_quadratic_hartree(density, direction)
        / functional.projection.site_hnc_asset.kbt_hartree,
        rel=2.0e-13,
        abs=2.0e-15,
    )

def test_molecular_site_hnc_bulk_pressure_is_the_same_functional_vacuum_limit():
    direct = np.zeros((1, 1, 2, 1, 1))
    direct[0, 0, :, 0, 0] = np.array([0.03, 0.03])
    functional = _functional(direct_correlation=direct)
    density = np.full(
        functional.projection.quadrature.configuration_count,
        1.0e-12 * functional.projection.uniform_configuration_density_bohr3,
    )
    ideal, excess, _ = functional.energy_components(density)
    cell_volume = (
        functional.projection.site_hnc_asset.grid.point_count
        * functional.projection.site_hnc_asset.grid.volume_element_bohr3
    )
    expected_pressure = 0.1 * (0.02 - 0.5 * 0.02**2 * (0.03 + 0.03))

    pressure = molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(functional)
    assert pressure == pytest.approx(
        expected_pressure,
        rel=2.0e-14,
        abs=2.0e-16,
    )
    assert (ideal + excess) / cell_volume == pytest.approx(
        expected_pressure,
        rel=3.0e-11,
        abs=3.0e-14,
    )


def test_molecular_hnc_bulk_pressure_has_one_ideal_term_per_molecule_not_site():
    functional = _two_site_one_type_functional()

    assert functional.projection.site_multiplicity.tolist() == [2]
    assert functional.projection.molecular_bulk_number_density_bohr3 == pytest.approx(
        0.02
    )
    assert molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(
        functional
    ) == pytest.approx(0.1 * 0.02)


def test_molecular_site_hnc_fixed_solute_thermodynamics_uses_one_stationary_scalar():
    functional = _functional()
    state = functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.5,
        max_iterations=300,
    )
    result = evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics(
        functional,
        state,
        residual_tolerance=1.0e-12,
    )
    weights = functional.projection.quadrature.phase_space_weights_bohr3
    observed_molecules = float(np.sum(weights * state.configuration_density_bohr3))
    cell_volume = (
        functional.projection.site_hnc_asset.grid.point_count
        * functional.projection.site_hnc_asset.grid.volume_element_bohr3
    )
    bulk_molecules = (
        functional.projection.molecular_bulk_number_density_bohr3 * cell_volume
    )
    expected_volume = (
        bulk_molecules - observed_molecules
    ) / functional.projection.molecular_bulk_number_density_bohr3
    expected_pressure = (
        functional.projection.molecular_bulk_number_density_bohr3
        * functional.projection.site_hnc_asset.kbt_hartree
    )
    expected_work = expected_pressure * expected_volume

    assert isinstance(result, Route2V0MolecularHNCFixedSoluteThermodynamics)
    assert result.construction == V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS
    assert result.functional is functional
    assert result.state is state
    assert result.residual_tolerance == pytest.approx(1.0e-12)
    assert result.bulk_reference_molecule_count == pytest.approx(bulk_molecules)
    assert result.stationary_molecule_count == pytest.approx(observed_molecules)
    assert result.partial_molar_volume_bohr3 == pytest.approx(expected_volume)
    assert result.bulk_functional_pressure_hartree_per_bohr3 == pytest.approx(
        expected_pressure
    )
    assert result.bulk_functional_pressure_work_hartree == pytest.approx(expected_work)
    assert result.fixed_solute_pressure_corrected_free_energy_hartree == pytest.approx(
        state.grand_potential_hartree - expected_work
    )


def test_molecular_site_hnc_fixed_solute_thermodynamics_fails_closed():
    functional = _functional()
    nonstationary = functional.stationary_state(
        np.full(
            functional.projection.quadrature.configuration_count,
            functional.projection.uniform_configuration_density_bohr3,
        ),
        iterations=0,
    )
    with pytest.raises(ValueError, match="stationary molecular HNC state"):
        evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics(
            functional,
            nonstationary,
            residual_tolerance=1.0e-12,
        )

    other = _functional()
    state = functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.5,
        max_iterations=300,
    )
    ledger = evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics(
        functional,
        state,
        residual_tolerance=1.0e-12,
    )
    with pytest.raises(ValueError, match="exact molecular HNC projection"):
        replace(ledger, functional=other)
    with pytest.raises(ValueError, match="exact molecular HNC projection"):
        evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics(
            other,
            state,
            residual_tolerance=1.0e-12,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics(
            functional,
            state,
            residual_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="independently verifies"):
        evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics(
            functional,
            replace(
                nonstationary,
                residual_inf=0.0,
            ),
            residual_tolerance=1.0e-12,
        )
    with pytest.raises(ValueError, match="energy components"):
        forged_ideal = state.ideal_contribution_hartree + 1.0
        evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics(
            functional,
            replace(
                state,
                ideal_contribution_hartree=forged_ideal,
                grand_potential_hartree=(
                    forged_ideal
                    + state.excess_contribution_hartree
                    + state.external_contribution_hartree
                ),
            ),
            residual_tolerance=1.0e-12,
        )


def test_molecular_site_hnc_rejects_inconsistent_measure_or_occupancy_and_bad_iterations():
    external = _external_potential()
    grid = external.integration_grid
    asset = Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("X",),
        bulk_number_density_bohr3=np.array([0.02]),
        direct_correlation_dimensionless=np.zeros((1, 1, *grid.shape)),
        kbt_hartree=0.1,
    )
    bad_measure = Route2V0MolecularConfigurationQuadrature(
        configurations=external.configurations,
        phase_space_weights_bohr3=np.ones(2),
    )
    occupancy = np.zeros((1, *grid.shape, 2))
    occupancy[0, 0, 0, 0, 0] = 1.0
    occupancy[0, 1, 0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="cover exactly"):
        Route2V0MolecularSiteProjection(
            quadrature=bad_measure,
            external_potential=external,
            site_hnc_asset=asset,
            solvent_site_type_indices=np.array([0]),
            site_occupancy_weights=occupancy,
        )

    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=external.configurations,
        phase_space_weights_bohr3=np.full(2, RIGID_MOLECULAR_ORIENTATION_MEASURE),
    )
    with pytest.raises(ValueError, match="sum to each declared"):
        Route2V0MolecularSiteProjection(
            quadrature=quadrature,
            external_potential=external,
            site_hnc_asset=asset,
            solvent_site_type_indices=np.array([0]),
            site_occupancy_weights=np.zeros_like(occupancy),
        )

    nonuniform_bulk_occupancy = np.zeros_like(occupancy)
    nonuniform_bulk_occupancy[0, 0, 0, 0, :] = 1.0
    with pytest.raises(ValueError, match="project the uniform"):
        Route2V0MolecularSiteProjection(
            quadrature=quadrature,
            external_potential=external,
            site_hnc_asset=asset,
            solvent_site_type_indices=np.array([0]),
            site_occupancy_weights=nonuniform_bulk_occupancy,
        )

    mismatched_grid = RegularCartesianGrid(
        origin_bohr=grid.origin_bohr,
        spacing_bohr=grid.spacing_bohr,
        shape=(1, 2, 1),
    )
    mismatched_asset = Route2V0SiteHNCAsset(
        grid=mismatched_grid,
        site_names=("X",),
        bulk_number_density_bohr3=np.array([0.02]),
        direct_correlation_dimensionless=np.zeros((1, 1, *mismatched_grid.shape)),
        kbt_hartree=0.1,
    )
    with pytest.raises(ValueError, match="shared external-potential"):
        Route2V0MolecularSiteProjection(
            quadrature=quadrature,
            external_potential=external,
            site_hnc_asset=mismatched_asset,
            solvent_site_type_indices=np.array([0]),
            site_occupancy_weights=occupancy,
        )

    functional = _functional()
    with pytest.raises(ValueError, match="nonnegative integer"):
        functional.stationary_state(
            np.full(2, functional.projection.uniform_configuration_density_bohr3),
            iterations=True,
        )
