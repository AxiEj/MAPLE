from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0MolecularSolventReference,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_site_hnc import (
    build_route2_v0_cartesian_euler_cubic_bspline_site_occupancy,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_so3_quadrature import (
    Route2V0CartesianEulerProductQuadrature,
    Route2V0EulerSO3Quadrature,
    build_route2_v0_cartesian_euler_product_quadrature,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-1.0, 2.0, 0.5]),
        spacing_bohr=np.array([0.5, 0.75, 1.25]),
        shape=(2, 3, 1),
    )


def test_euler_so3_quadrature_has_the_declared_full_haar_measure_and_moments():
    quadrature = Route2V0EulerSO3Quadrature(polar_order=2)

    assert quadrature.orientation_count == 32
    assert quadrature.periodic_order == 4
    assert quadrature.wigner_rank_bandlimit == 3
    assert quadrature.orientation_measure == pytest.approx(8.0 * math.pi**2)
    assert np.all(quadrature.orientation_weights > 0.0)
    np.testing.assert_allclose(
        np.einsum("nij,nkj->nik", quadrature.rotations, quadrature.rotations),
        np.broadcast_to(np.eye(3), (quadrature.orientation_count, 3, 3)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.linalg.det(quadrature.rotations),
        np.ones(quadrature.orientation_count),
        rtol=0.0,
        atol=1.0e-12,
    )

    first_moment = np.einsum(
        "n,nij->ij",
        quadrature.orientation_weights,
        quadrature.rotations,
    )
    second_moment = np.einsum(
        "n,nij,nkl->ijkl",
        quadrature.orientation_weights,
        quadrature.rotations,
        quadrature.rotations,
    )
    expected_second_moment = (
        quadrature.orientation_measure
        / 3.0
        * np.einsum("ik,jl->ijkl", np.eye(3), np.eye(3))
    )
    np.testing.assert_allclose(first_moment, np.zeros((3, 3)), rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        second_moment,
        expected_second_moment,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_cartesian_euler_product_matches_the_existing_molecular_configuration_measure():
    grid = _grid()
    product = build_route2_v0_cartesian_euler_product_quadrature(
        grid=grid,
        polar_order=2,
    )

    orientation = product.orientation_quadrature
    assert product.configurations.configuration_count == (
        grid.point_count * orientation.orientation_count
    )
    np.testing.assert_allclose(
        product.configurations.translations_bohr,
        np.repeat(grid.points_bohr(), orientation.orientation_count, axis=0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        product.configurations.rotations,
        np.tile(orientation.rotations, (grid.point_count, 1, 1)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        product.quadrature.phase_space_weights_bohr3,
        np.tile(
            grid.volume_element_bohr3 * orientation.orientation_weights,
            grid.point_count,
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert product.quadrature.total_phase_space_measure_bohr3 == pytest.approx(
        grid.point_count * grid.volume_element_bohr3 * 8.0 * math.pi**2,
        rel=0.0,
        abs=1.0e-12,
    )


def test_euler_so3_quadrature_is_deterministic_and_fails_closed_for_invalid_orders():
    left = Route2V0EulerSO3Quadrature(polar_order=np.int64(3))
    right = Route2V0EulerSO3Quadrature(polar_order=3)
    np.testing.assert_array_equal(left.rotations, right.rotations)
    np.testing.assert_array_equal(left.orientation_weights, right.orientation_weights)

    for invalid in (True, "3", None):
        with pytest.raises(TypeError, match="positive integer"):
            Route2V0EulerSO3Quadrature(polar_order=invalid)
    for invalid in (0, -1, 1.5, float("inf")):
        with pytest.raises(ValueError, match="positive integer"):
            Route2V0EulerSO3Quadrature(polar_order=invalid)
    with pytest.raises(ValueError, match=r"Unsupported Route-2 V0 SO\(3\)"):
        replace(left, construction="another-rule")

    with pytest.raises(TypeError, match="regular Cartesian grid"):
        Route2V0CartesianEulerProductQuadrature(
            grid=None,
            orientation_quadrature=left,
        )
    with pytest.raises(TypeError, match=r"SO\(3\) quadrature"):
        Route2V0CartesianEulerProductQuadrature(
            grid=_grid(),
            orientation_quadrature=None,
        )


def test_cartesian_euler_cubic_bspline_site_map_preserves_bulk_multiplicity():
    grid = _grid()
    product = build_route2_v0_cartesian_euler_product_quadrature(
        grid=grid,
        polar_order=2,
    )
    solvent = Route2V0MolecularSolventReference(
        atomic_numbers=np.array([8, 1, 1]),
        site_charges_e=np.array([-0.8, 0.4, 0.4]),
        reference_positions_bohr=np.array(
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [-0.3, 1.1, 0.2]]
        ),
        provenance_label="synthetic rigid water-like B-spline projection control",
    )
    occupancy = build_route2_v0_cartesian_euler_cubic_bspline_site_occupancy(
        cartesian_euler_quadrature=product,
        solvent=solvent,
        solvent_site_type_indices=np.array([0, 1, 1]),
        site_count=2,
    )

    assert occupancy.shape == (
        2,
        *grid.shape,
        product.configurations.configuration_count,
    )
    assert np.all(occupancy >= 0.0)
    flat = occupancy.reshape(
        (2, grid.point_count, product.configurations.configuration_count)
    )
    np.testing.assert_allclose(
        np.sum(flat, axis=1),
        np.broadcast_to(
            np.array([[1.0], [2.0]]), (2, product.configurations.configuration_count)
        ),
        rtol=0.0,
        atol=2.0e-14,
    )

    molecular_density = 0.123
    uniform_configuration_density = molecular_density / (8.0 * math.pi**2)
    projected = (
        np.einsum(
            "agi,i,i->ag",
            flat,
            product.quadrature.phase_space_weights_bohr3,
            np.full(
                product.configurations.configuration_count,
                uniform_configuration_density,
            ),
            optimize=True,
        )
        / grid.volume_element_bohr3
    ).reshape((2, *grid.shape))
    np.testing.assert_allclose(
        projected,
        np.stack(
            (
                np.full(grid.shape, molecular_density),
                np.full(grid.shape, 2.0 * molecular_density),
            )
        ),
        rtol=0.0,
        atol=5.0e-14,
    )

    with pytest.raises(ValueError, match="Every HNC site type"):
        build_route2_v0_cartesian_euler_cubic_bspline_site_occupancy(
            cartesian_euler_quadrature=product,
            solvent=solvent,
            solvent_site_type_indices=np.array([0, 0, 0]),
            site_count=2,
        )
