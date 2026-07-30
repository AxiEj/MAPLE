from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_coexistence_continuation import (
    solve_route2_v0_molecular_quartic_coexistence_continuation,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0FrozenMaceGaussianSource,
    Route2V0MolecularExternalPotential,
    Route2V0MolecularSolventReference,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_planar_interface import (
    V0_MOLECULAR_EQUIMOLAR_DIVIDING_SURFACE,
    Route2V0MolecularPlanarSymmetry,
    evaluate_route2_v0_molecular_constrained_planar_interface,
    solve_route2_v0_molecular_constrained_planar_interface,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_so3_quadrature import (
    Route2V0CartesianEulerProductQuadrature,
    build_route2_v0_cartesian_euler_product_quadrature,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_surface_tension_continuation import (
    Route2V0MolecularGaussianSurfaceTensionBracket,
    Route2V0MolecularGaussianSurfaceTensionPoint,
    evaluate_route2_v0_molecular_gaussian_surface_tension_point,
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
_COEXISTENT_DIMENSIONLESS_QUARTIC = 7.119206180987243


def _full_so3_planar_control(
    *,
    external_energy_hartree: float = 0.0,
) -> tuple[
    Route2V0MolecularWeightedDensityBridgeFunctional,
    Route2V0CartesianEulerProductQuadrature,
]:
    """Build a synthetic exact-product scalar with a zero-cost sharp interface."""

    grid = RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.ones(3),
        shape=(2, 2, 16),
    )
    product = build_route2_v0_cartesian_euler_product_quadrature(
        grid=grid,
        polar_order=1,
    )
    configuration_count = product.quadrature.configuration_count
    orientation_count = product.orientation_quadrature.orientation_count
    table = Route2V0PromolecularDensityTable(
        radial_grid_bohr=np.array([0.0, 1.0, 2.0]),
        densities_by_atomic_number={1: np.zeros(3)},
        table_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )
    solvent = Route2V0MolecularSolventReference(
        atomic_numbers=np.array([1]),
        site_charges_e=np.array([0.0]),
        reference_positions_bohr=np.zeros((1, 3)),
        provenance_label="synthetic full-SO(3) planar control",
    )
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
        configurations=product.configurations,
        solute_reference_density_e_per_bohr3=np.zeros(grid.shape),
        pauli_repulsion_hartree=np.full(configuration_count, external_energy_hartree),
        electrostatic_energy_hartree=np.zeros(configuration_count),
        external_potential_hartree=np.full(
            configuration_count,
            external_energy_hartree,
        ),
    )
    hnc_asset = Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("X",),
        bulk_number_density_bohr3=np.array([_BULK_DENSITY_BOHR3]),
        direct_correlation_dimensionless=np.zeros((1, 1, *grid.shape)),
        kbt_hartree=_KBT_HARTREE,
    )
    site_occupancy = np.zeros((1, *grid.shape, configuration_count))
    center_occupancy = np.zeros((*grid.shape, configuration_count))
    flat_site = site_occupancy.reshape((1, grid.point_count, configuration_count))
    flat_center = center_occupancy.reshape((grid.point_count, configuration_count))
    for point in range(grid.point_count):
        indices = slice(point * orientation_count, (point + 1) * orientation_count)
        flat_site[0, point, indices] = 1.0
        flat_center[point, indices] = 1.0
    projection = Route2V0MolecularSiteProjection(
        quadrature=product.quadrature,
        external_potential=external,
        site_hnc_asset=hnc_asset,
        solvent_site_type_indices=np.array([0]),
        site_occupancy_weights=site_occupancy,
    )
    hnc = Route2V0MolecularSiteHNCFunctional(projection)
    center_projection = Route2V0MolecularCenterProjection(
        projection=projection,
        center_occupancy_weights=center_occupancy,
    )
    kernel_values = np.zeros(grid.shape)
    kernel_values[0, 0, 0] = 1.0 / grid.volume_element_bohr3
    kernel = Route2V0PeriodicWeightedDensityKernel(
        grid=grid,
        kernel_bohr_minus3=kernel_values,
    )
    quartic = _COEXISTENT_DIMENSIONLESS_QUARTIC * _KBT_HARTREE / _BULK_DENSITY_BOHR3**5
    bridge = Route2V0MolecularWeightedDensityBridgeAsset.from_pure_solvent_anchors(
        center_projection=center_projection,
        kernel=kernel,
        hnc_bulk_pressure_hartree_per_bohr3=(
            molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(hnc)
        ),
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
    ), product


@pytest.fixture(scope="module")
def planar_control():
    """Return the frozen root scalar, product quadrature, and coexistence evidence."""

    functional, product = _full_so3_planar_control()
    coefficient = functional.bridge_asset.quartic_coefficient_hartree_bohr15
    continuation = solve_route2_v0_molecular_quartic_coexistence_continuation(
        functional,
        lower_quartic_coefficient_hartree_bohr15=0.5 * coefficient,
        upper_quartic_coefficient_hartree_bohr15=1.5 * coefficient,
        coexistence_tolerance_hartree_per_bohr3=1.0e-10,
        coefficient_relative_tolerance=1.0e-8,
        maximum_iterations=64,
        minimum_density_scale=1.0e-5,
        root_sample_count=129,
        directional_derivative_tolerance_hartree_per_bohr3=1.0e-13,
        curvature_tolerance_hartree_per_bohr3=1.0e-13,
    )
    frozen = Route2V0MolecularWeightedDensityBridgeFunctional(
        hnc_functional=functional.hnc_functional,
        bridge_asset=replace(
            functional.bridge_asset,
            quartic_coefficient_hartree_bohr15=(
                continuation.root.quartic_coefficient_hartree_bohr15
            ),
        ),
    )
    return frozen, product, continuation


def test_planar_embedding_preserves_full_so3_scalar_pairing(planar_control):
    functional, product, _ = planar_control
    symmetry = Route2V0MolecularPlanarSymmetry(functional, product, normal_axis=2)
    normal = np.arange(symmetry.normal_point_count, dtype=float)
    orientation = np.arange(symmetry.orientation_count, dtype=float)
    reduced_density = functional.projection.uniform_configuration_density_bohr3 * (
        0.70
        + 0.05 * np.cos(2.0 * np.pi * normal[:, None] / symmetry.normal_point_count)
        + 0.01 * orientation[None, :]
    )
    direction = (
        0.02
        * functional.projection.uniform_configuration_density_bohr3
        * (
            np.sin(2.0 * np.pi * normal[:, None] / symmetry.normal_point_count)
            + 0.25 * orientation[None, :]
        )
    )
    full_density = symmetry.expand_configuration_density(reduced_density)
    full_direction = symmetry.expand_reduced_field(
        direction,
        name="Synthetic planar signed direction",
    )
    reduced_gradient = symmetry.reduce_dimensionless_gradient(
        functional.dimensionless_gradient(full_density)
    )
    directional_derivative = functional.kbt_hartree * np.sum(
        symmetry.reduced_phase_space_weights_bohr3 * reduced_gradient * direction
    )
    step = 1.0e-5
    upper = functional.grand_potential_hartree(full_density + step * full_direction)
    lower = functional.grand_potential_hartree(full_density - step * full_direction)

    assert (
        symmetry.orientation_count == product.orientation_quadrature.orientation_count
    )
    assert symmetry.reduced_shape == (16, 4)
    assert symmetry.transverse_area_bohr2 == pytest.approx(4.0)
    assert symmetry.molecule_count(reduced_density) == pytest.approx(
        np.sum(
            functional.projection.quadrature.phase_space_weights_bohr3 * full_density
        )
    )
    assert (upper - lower) / (2.0 * step) == pytest.approx(
        directional_derivative,
        rel=2.0e-8,
        abs=2.0e-13,
    )
    assert symmetry.transverse_gradient_nonuniformity(
        functional.dimensionless_gradient(full_density),
        reduced_gradient=reduced_gradient,
    ) == pytest.approx(0.0, abs=1.0e-13)


def test_constrained_planar_control_requires_zero_multiplier_and_has_no_fake_excess(
    planar_control,
):
    functional, product, continuation = planar_control
    symmetry = Route2V0MolecularPlanarSymmetry(functional, product, normal_axis=2)
    target_scale = 0.5 * (
        continuation.root.gas_phase.density_scale
        + continuation.root.liquid_phase.density_scale
    )
    state = solve_route2_v0_molecular_constrained_planar_interface(
        functional,
        planar_symmetry=symmetry,
        coexistence_continuation=continuation,
        target_mean_density_scale=target_scale,
        stationarity_tolerance=1.0e-8,
        transverse_uniformity_tolerance=1.0e-8,
        constraint_multiplier_tolerance=1.0e-8,
        maximum_iterations=20,
    )

    assert state.passes is True
    assert state.iterations == 0
    assert state.dividing_surface == V0_MOLECULAR_EQUIMOLAR_DIVIDING_SURFACE
    assert state.interface_count == 2
    assert state.constraint_multiplier_dimensionless == pytest.approx(0.0, abs=2.0e-10)
    assert state.unconstrained_residual_inf <= state.stationarity_tolerance
    assert state.surface_tension_hartree_per_bohr2 == pytest.approx(0.0, abs=2.0e-15)
    assert (
        "target_surface_tension"
        not in inspect.signature(
            solve_route2_v0_molecular_constrained_planar_interface
        ).parameters
    )


def test_nonzero_constraint_multiplier_is_not_an_unconstrained_interface(
    planar_control,
):
    functional, product, continuation = planar_control
    symmetry = Route2V0MolecularPlanarSymmetry(functional, product, normal_axis=2)
    target_scale = 0.5 * (
        continuation.root.gas_phase.density_scale
        + continuation.root.liquid_phase.density_scale
    )
    reduced_density = (
        functional.projection.uniform_configuration_density_bohr3
        * np.where(
            np.arange(symmetry.normal_point_count)[:, None] < 8,
            continuation.root.liquid_phase.density_scale,
            continuation.root.gas_phase.density_scale,
        )
        * np.ones((1, symmetry.orientation_count))
    )
    state = evaluate_route2_v0_molecular_constrained_planar_interface(
        functional,
        planar_symmetry=symmetry,
        coexistence_continuation=continuation,
        reduced_configuration_density_bohr3=reduced_density,
        target_mean_density_scale=target_scale,
        constraint_multiplier_dimensionless=1.0e-4,
        stationarity_tolerance=1.0e-8,
        transverse_uniformity_tolerance=1.0e-8,
        constraint_multiplier_tolerance=1.0e-8,
    )

    assert state.constrained_residual_inf > state.stationarity_tolerance
    assert state.is_unconstrained_stationary is False
    assert state.passes is False


@pytest.mark.parametrize(
    ("normal_axis", "expected_shape", "expected_area"),
    [(0, (2, 4), 32.0), (1, (2, 4), 32.0)],
)
def test_planar_solver_retains_the_exact_selected_normal_axis(
    planar_control,
    normal_axis,
    expected_shape,
    expected_area,
):
    functional, product, continuation = planar_control
    symmetry = Route2V0MolecularPlanarSymmetry(
        functional,
        product,
        normal_axis=normal_axis,
    )
    target_scale = 0.5 * (
        continuation.root.gas_phase.density_scale
        + continuation.root.liquid_phase.density_scale
    )
    state = solve_route2_v0_molecular_constrained_planar_interface(
        functional,
        planar_symmetry=symmetry,
        coexistence_continuation=continuation,
        target_mean_density_scale=target_scale,
        stationarity_tolerance=1.0e-8,
        transverse_uniformity_tolerance=1.0e-8,
        constraint_multiplier_tolerance=1.0e-8,
        maximum_iterations=20,
    )

    assert symmetry.reduced_shape == expected_shape
    assert symmetry.transverse_area_bohr2 == pytest.approx(expected_area)
    assert state.passes is True
    assert state.interface_count == 2
    assert state.surface_tension_hartree_per_bohr2 == pytest.approx(0.0, abs=2.0e-15)


def test_planar_symmetry_requires_the_exact_full_product_and_frozen_root(
    planar_control,
):
    frozen, product, continuation = planar_control
    original, _ = _full_so3_planar_control()
    wrong_product = build_route2_v0_cartesian_euler_product_quadrature(
        grid=product.grid,
        polar_order=2,
    )

    with pytest.raises(ValueError, match="exact Cartesian-Euler product"):
        Route2V0MolecularPlanarSymmetry(frozen, wrong_product)

    symmetry = Route2V0MolecularPlanarSymmetry(original, product, normal_axis=2)
    target_scale = 0.5 * (
        continuation.root.gas_phase.density_scale
        + continuation.root.liquid_phase.density_scale
    )
    with pytest.raises(ValueError, match="coefficient must equal"):
        solve_route2_v0_molecular_constrained_planar_interface(
            original,
            planar_symmetry=symmetry,
            coexistence_continuation=continuation,
            target_mean_density_scale=target_scale,
            maximum_iterations=1,
        )


def test_planar_interface_rejects_a_nonzero_external_scalar_after_root_freeze(
    planar_control,
):
    _, _, continuation = planar_control
    field_conditioned, product = _full_so3_planar_control(
        external_energy_hartree=1.0e-7,
    )
    functional = Route2V0MolecularWeightedDensityBridgeFunctional(
        hnc_functional=field_conditioned.hnc_functional,
        bridge_asset=replace(
            field_conditioned.bridge_asset,
            quartic_coefficient_hartree_bohr15=(
                continuation.root.quartic_coefficient_hartree_bohr15
            ),
        ),
    )
    symmetry = Route2V0MolecularPlanarSymmetry(functional, product, normal_axis=2)
    target_scale = 0.5 * (
        continuation.root.gas_phase.density_scale
        + continuation.root.liquid_phase.density_scale
    )

    with pytest.raises(ValueError, match="zero-external-potential"):
        solve_route2_v0_molecular_constrained_planar_interface(
            functional,
            planar_symmetry=symmetry,
            coexistence_continuation=continuation,
            target_mean_density_scale=target_scale,
            maximum_iterations=1,
        )


def test_gaussian_point_runs_the_inner_coexistence_solve_before_planar_control(
    planar_control,
):
    """The one-width Gaussian control has no access to a gamma target."""

    template, product, _ = planar_control
    coefficient = template.bridge_asset.quartic_coefficient_hartree_bohr15
    point = evaluate_route2_v0_molecular_gaussian_surface_tension_point(
        template,
        cartesian_euler_quadrature=product,
        gaussian_width_bohr=0.01,
        lower_quartic_coefficient_hartree_bohr15=0.5 * coefficient,
        upper_quartic_coefficient_hartree_bohr15=1.5 * coefficient,
        coexistence_tolerance_hartree_per_bohr3=1.0e-10,
        coefficient_relative_tolerance=1.0e-8,
        maximum_coexistence_iterations=64,
        minimum_density_scale=1.0e-5,
        root_sample_count=129,
        directional_derivative_tolerance_hartree_per_bohr3=1.0e-13,
        curvature_tolerance_hartree_per_bohr3=1.0e-13,
        planar_stationarity_tolerance=1.0e-8,
        planar_transverse_uniformity_tolerance=1.0e-8,
        planar_constraint_multiplier_tolerance=1.0e-8,
        maximum_planar_iterations=20,
    )

    assert isinstance(point, Route2V0MolecularGaussianSurfaceTensionPoint)
    assert point.coexistence_continuation.passes is True
    assert point.planar_state.passes is True
    assert point.surface_tension_hartree_per_bohr2 == pytest.approx(0.0, abs=2.0e-15)
    assert point.physical_liquid_admitted is False
    assert (
        "target_surface_tension"
        not in inspect.signature(
            evaluate_route2_v0_molecular_gaussian_surface_tension_point
        ).parameters
    )

    with pytest.raises(ValueError, match="kernel does not match"):
        replace(point, gaussian_width_bohr=0.5)


def test_gaussian_outer_bracket_refuses_unresolved_length_aliases(planar_control):
    """A coarse grid cannot pretend that three aliased widths determine sigma."""

    functional, product, continuation = planar_control
    symmetry = Route2V0MolecularPlanarSymmetry(functional, product, normal_axis=2)
    target_scale = 0.5 * (
        continuation.root.gas_phase.density_scale
        + continuation.root.liquid_phase.density_scale
    )
    state = solve_route2_v0_molecular_constrained_planar_interface(
        functional,
        planar_symmetry=symmetry,
        coexistence_continuation=continuation,
        target_mean_density_scale=target_scale,
        stationarity_tolerance=1.0e-8,
        transverse_uniformity_tolerance=1.0e-8,
        constraint_multiplier_tolerance=1.0e-8,
        maximum_iterations=20,
    )
    lower = Route2V0MolecularGaussianSurfaceTensionPoint(
        gaussian_width_bohr=0.01,
        functional=functional,
        coexistence_continuation=continuation,
        planar_symmetry=symmetry,
        planar_state=state,
    )
    root = replace(lower, gaussian_width_bohr=0.015)
    upper = replace(lower, gaussian_width_bohr=0.02)

    with pytest.raises(ValueError, match="do not resolve distinct discrete kernels"):
        Route2V0MolecularGaussianSurfaceTensionBracket(
            lower=lower,
            root=root,
            upper=upper,
            surface_tension_tolerance_hartree_per_bohr2=1.0e-10,
        )
