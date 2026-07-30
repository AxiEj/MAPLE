from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential_contract import (
    Route2V0MolecularExternalPotentialContract,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_site_bspline import (
    Route2V0MolecularSiteCubicBSplineDeposition,
    build_route2_v0_cartesian_euler_cubic_bspline_site_deposition,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
    build_route2_v0_cartesian_euler_cubic_bspline_site_occupancy,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_so3_quadrature import (
    Route2V0CartesianEulerProductQuadrature,
    build_route2_v0_cartesian_euler_product_quadrature,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_site_hnc import (
    Route2V0SiteHNCAsset,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


@dataclass(frozen=True)
class _ExternalPotential(Route2V0MolecularExternalPotentialContract):
    """Minimal declared source used only to compare projection representations."""

    integration_grid: RegularCartesianGrid
    solvent: Route2V0MolecularSolventReference
    configurations: Route2V0MolecularConfigurations
    external_potential_hartree: np.ndarray


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-1.0, 0.5, 2.0]),
        spacing_bohr=np.array([0.75, 1.25, 0.5]),
        shape=(2, 3, 2),
    )


def _solvent() -> Route2V0MolecularSolventReference:
    return Route2V0MolecularSolventReference(
        atomic_numbers=np.array([8, 1, 1]),
        site_charges_e=np.array([-0.8, 0.4, 0.4]),
        reference_positions_bohr=np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [-0.4, 0.95, -0.2]]
        ),
        provenance_label="matrix-free cubic B-spline molecular-site control",
    )


def _product() -> Route2V0CartesianEulerProductQuadrature:
    return build_route2_v0_cartesian_euler_product_quadrature(
        grid=_grid(),
        polar_order=2,
    )


def _site_asset(grid: RegularCartesianGrid) -> Route2V0SiteHNCAsset:
    direct = np.zeros((2, 2, *grid.shape))
    direct[0, 0, 0, 0, 0] = 0.03
    direct[1, 1, 0, 0, 0] = 0.02
    direct[0, 1, 0, 0, 0] = 0.01
    direct[1, 0, 0, 0, 0] = 0.01
    return Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("O", "H1"),
        bulk_number_density_bohr3=np.array([0.012, 0.024]),
        direct_correlation_dimensionless=direct,
        kbt_hartree=0.001,
    )


def _external(
    product: Route2V0CartesianEulerProductQuadrature,
    solvent: Route2V0MolecularSolventReference,
) -> _ExternalPotential:
    values = np.linspace(
        -0.0008,
        0.0012,
        product.configurations.configuration_count,
    )
    return _ExternalPotential(
        integration_grid=product.grid,
        solvent=solvent,
        configurations=product.configurations,
        external_potential_hartree=values,
    )


def test_compact_molecular_site_bspline_deposition_is_dense_reference_equivalent():
    product = _product()
    solvent = _solvent()
    type_indices = np.array([0, 1, 1])
    compact = build_route2_v0_cartesian_euler_cubic_bspline_site_deposition(
        cartesian_euler_quadrature=product,
        solvent=solvent,
        solvent_site_type_indices=type_indices,
        site_count=2,
    )
    dense = build_route2_v0_cartesian_euler_cubic_bspline_site_occupancy(
        cartesian_euler_quadrature=product,
        solvent=solvent,
        solvent_site_type_indices=type_indices,
        site_count=2,
    )
    rng = np.random.default_rng(20260730)
    values = rng.normal(size=product.configurations.configuration_count)
    field = rng.normal(size=(2, *product.grid.shape))

    expected_projected = (
        np.einsum(
            "agi,i,i->ag",
            dense.reshape(
                (
                    2,
                    product.grid.point_count,
                    product.configurations.configuration_count,
                )
            ),
            product.quadrature.phase_space_weights_bohr3,
            values,
            optimize=True,
        )
        / product.grid.volume_element_bohr3
    ).reshape((2, *product.grid.shape))
    expected_adjoint = np.einsum(
        "agi,ag->i",
        dense.reshape(
            (2, product.grid.point_count, product.configurations.configuration_count)
        ),
        field.reshape((2, product.grid.point_count)),
        optimize=True,
    )

    projected = compact.project_configuration_values(
        values,
        phase_space_weights_bohr3=product.quadrature.phase_space_weights_bohr3,
        volume_element_bohr3=product.grid.volume_element_bohr3,
    )
    adjoint = compact.site_field_adjoint_dimensionless(field)
    # The implementations add the same finite terms in different orders.  This
    # tolerance covers only float64 reduction roundoff; the exact discrete
    # pairing is checked separately below.
    np.testing.assert_allclose(
        projected,
        expected_projected,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        adjoint,
        expected_adjoint,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert (
        product.grid.volume_element_bohr3 * np.vdot(field, projected)
    ) == pytest.approx(
        np.dot(
            product.quadrature.phase_space_weights_bohr3 * adjoint,
            values,
        ),
        rel=2.0e-14,
        abs=2.0e-14,
    )
    assert compact.site_multiplicity.tolist() == [1, 2]
    assert compact.storage_entry_count == (
        solvent.site_count * product.configurations.configuration_count * 64
    )
    assert compact.dense_reference_byte_count == dense.nbytes
    assert compact.storage_byte_count == (
        solvent.site_count
        * product.configurations.configuration_count
        * (
            3 * np.dtype(float).itemsize
            + 64 * (np.dtype(np.int64).itemsize + np.dtype(float).itemsize)
        )
    )


def test_compact_deposition_becomes_smaller_than_dense_on_a_larger_grid():
    """The compact map is linear, while the dense product tensor is quadratic."""

    grid = RegularCartesianGrid(
        origin_bohr=np.array([-3.0, -3.0, -1.5]),
        spacing_bohr=np.array([0.5, 0.5, 0.5]),
        shape=(7, 7, 5),
    )
    configurations = Route2V0MolecularConfigurations(
        translations_bohr=np.array([[0.0, 0.0, 0.0]]),
        rotations=np.eye(3)[None, :, :],
    )
    compact = Route2V0MolecularSiteCubicBSplineDeposition(
        grid=grid,
        configurations=configurations,
        solvent=_solvent(),
        solvent_site_type_indices=np.array([0, 1, 1]),
        site_count=2,
    )

    assert compact.grid.point_count == 245
    assert compact.storage_byte_count < compact.dense_reference_byte_count


def test_matrix_free_projection_preserves_the_dense_scalar_gradient_and_hessian():
    product = _product()
    solvent = _solvent()
    external = _external(product, solvent)
    asset = _site_asset(product.grid)
    type_indices = np.array([0, 1, 1])
    dense = build_route2_v0_cartesian_euler_cubic_bspline_site_occupancy(
        cartesian_euler_quadrature=product,
        solvent=solvent,
        solvent_site_type_indices=type_indices,
        site_count=asset.site_count,
    )
    compact = Route2V0MolecularSiteCubicBSplineDeposition(
        grid=product.grid,
        configurations=product.configurations,
        solvent=solvent,
        solvent_site_type_indices=type_indices,
        site_count=asset.site_count,
    )
    with pytest.raises(ValueError, match="exactly one dense occupancy"):
        Route2V0MolecularSiteProjection(
            quadrature=product.quadrature,
            external_potential=external,
            site_hnc_asset=asset,
            solvent_site_type_indices=type_indices,
        )
    with pytest.raises(ValueError, match="exactly one dense occupancy"):
        Route2V0MolecularSiteProjection(
            quadrature=product.quadrature,
            external_potential=external,
            site_hnc_asset=asset,
            solvent_site_type_indices=type_indices,
            site_occupancy_weights=dense,
            compact_site_deposition=compact,
        )
    dense_projection = Route2V0MolecularSiteProjection(
        quadrature=product.quadrature,
        external_potential=external,
        site_hnc_asset=asset,
        solvent_site_type_indices=type_indices,
        site_occupancy_weights=dense,
    )
    matrix_free_projection = Route2V0MolecularSiteProjection(
        quadrature=product.quadrature,
        external_potential=external,
        site_hnc_asset=asset,
        solvent_site_type_indices=type_indices,
        compact_site_deposition=compact,
    )
    dense_functional = Route2V0MolecularSiteHNCFunctional(dense_projection)
    matrix_free_functional = Route2V0MolecularSiteHNCFunctional(matrix_free_projection)
    bulk = dense_projection.uniform_configuration_density_bohr3
    count = product.configurations.configuration_count
    density = bulk * (1.0 + 0.05 * np.sin(np.arange(count)))
    direction = bulk * 0.1 * np.cos(np.arange(count))

    assert matrix_free_projection.is_matrix_free is True
    assert matrix_free_projection.site_occupancy_weights is None
    np.testing.assert_allclose(
        matrix_free_projection.project_configuration_density(density),
        dense_projection.project_configuration_density(density),
        rtol=0.0,
        atol=4.0e-15,
    )
    np.testing.assert_allclose(
        matrix_free_projection.project_configuration_direction(direction),
        dense_projection.project_configuration_direction(direction),
        rtol=0.0,
        atol=4.0e-15,
    )
    np.testing.assert_allclose(
        matrix_free_functional.dimensionless_gradient(density),
        dense_functional.dimensionless_gradient(density),
        rtol=0.0,
        atol=6.0e-15,
    )
    np.testing.assert_allclose(
        matrix_free_functional.dimensionless_hessian_matvec(density, direction),
        dense_functional.dimensionless_hessian_matvec(density, direction),
        rtol=0.0,
        atol=8.0e-15,
    )
    assert matrix_free_functional.grand_potential_hartree(density) == pytest.approx(
        dense_functional.grand_potential_hartree(density),
        rel=0.0,
        abs=8.0e-17,
    )


def test_matrix_free_projection_rejects_a_compact_map_from_another_configuration_grid():
    product = _product()
    solvent = _solvent()
    external = _external(product, solvent)
    asset = _site_asset(product.grid)
    other = build_route2_v0_cartesian_euler_product_quadrature(
        grid=product.grid,
        polar_order=1,
    )
    compact = build_route2_v0_cartesian_euler_cubic_bspline_site_deposition(
        cartesian_euler_quadrature=other,
        solvent=solvent,
        solvent_site_type_indices=np.array([0, 1, 1]),
        site_count=asset.site_count,
    )

    with pytest.raises(ValueError, match="configuration grid"):
        Route2V0MolecularSiteProjection(
            quadrature=product.quadrature,
            external_potential=external,
            site_hnc_asset=asset,
            solvent_site_type_indices=np.array([0, 1, 1]),
            compact_site_deposition=compact,
        )
