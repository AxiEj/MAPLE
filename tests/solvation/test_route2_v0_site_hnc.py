from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_site_hnc import (
    Route2V0SiteHNCAsset,
    Route2V0SiteHNCFunctional,
    V0_SITE_HNC_CONSTRUCTION,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
    evaluate_route2_v0_structured_solvent_electrostatic_source,
)


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-1.0, -1.5, -2.0]),
        spacing_bohr=np.array([0.5, 0.75, 1.0]),
        shape=(3, 4, 5),
    )


def _asset(
    *,
    direct_correlation: np.ndarray | None = None,
    site_count: int = 1,
) -> Route2V0SiteHNCAsset:
    grid = _grid()
    if direct_correlation is None:
        direct_correlation = np.zeros((site_count, site_count, *grid.shape))
    return Route2V0SiteHNCAsset(
        grid=grid,
        site_names=tuple(f"site-{index}" for index in range(site_count)),
        bulk_number_density_bohr3=np.linspace(0.002, 0.003, site_count),
        direct_correlation_dimensionless=direct_correlation,
        kbt_hartree=0.001,
    )


def _reciprocal_direct(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    for left in range(values.shape[0]):
        for right in range(values.shape[1]):
            result[left, right] = 0.5 * (
                values[left, right] + _reverse(values[right, left])
            )
    return result


def test_hnc_zero_correlation_solves_the_exact_ideal_gas_stationarity():
    asset = _asset()
    axes = np.indices(asset.grid.shape)
    external = 0.0002 * np.cos(axes[0])[None, :, :, :]
    functional = Route2V0SiteHNCFunctional(asset, external)

    state = functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.4,
        max_iterations=300,
    )
    expected = asset.bulk_number_density_bohr3.reshape((1, 1, 1, 1)) * np.exp(
        -external / asset.kbt_hartree
    )
    np.testing.assert_allclose(state.density_bohr3, expected, rtol=0.0, atol=2.0e-14)
    assert state.residual_inf < 1.0e-12
    assert state.construction == V0_SITE_HNC_CONSTRUCTION


def test_hnc_energy_gradient_is_the_stationarity_residual_of_the_same_scalar():
    rng = np.random.default_rng(20260729)
    grid = _grid()
    direct = _reciprocal_direct(rng.normal(scale=0.01, size=(1, 1, *grid.shape)))
    asset = _asset(direct_correlation=direct)
    external = rng.normal(scale=0.0001, size=(1, *grid.shape))
    functional = Route2V0SiteHNCFunctional(asset, external)
    bulk = asset.bulk_number_density_bohr3.reshape((1, 1, 1, 1))
    density = bulk * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=(1, *grid.shape)))
    direction = bulk * rng.uniform(-1.0, 1.0, size=(1, *grid.shape))
    epsilon = 1.0e-6

    finite_difference = (
        functional.grand_potential_hartree(density + epsilon * direction)
        - functional.grand_potential_hartree(density - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = (
        asset.kbt_hartree
        * asset.grid.volume_element_bohr3
        * np.sum(functional.dimensionless_gradient(density) * direction)
    )
    assert finite_difference == pytest.approx(analytic, rel=2.0e-8, abs=2.0e-12)


def _reverse(values: np.ndarray) -> np.ndarray:
    indices = np.ix_(*((-np.arange(extent)) % extent for extent in values.shape))
    return values[indices]


def test_hnc_reciprocal_direct_correlation_has_a_symmetric_convolution_pairing():
    rng = np.random.default_rng(29)
    grid = _grid()
    direct = _reciprocal_direct(rng.normal(scale=0.02, size=(2, 2, *grid.shape)))
    asset = _asset(direct_correlation=direct, site_count=2)
    functional = Route2V0SiteHNCFunctional(
        asset,
        np.zeros((2, *grid.shape)),
    )
    left = rng.normal(size=(2, *grid.shape))
    right = rng.normal(size=(2, *grid.shape))

    lhs = asset.grid.volume_element_bohr3 * np.sum(
        left * functional.convolve_direct_correlation(right)
    )
    rhs = asset.grid.volume_element_bohr3 * np.sum(
        right * functional.convolve_direct_correlation(left)
    )
    assert lhs == pytest.approx(rhs, rel=2.0e-13, abs=2.0e-13)


def test_hnc_rejects_a_nonreciprocal_correlation_asset():
    direct = np.zeros((1, 1, *_grid().shape))
    direct[0, 0, 1, 0, 0] = 0.1
    with pytest.raises(ValueError, match="c_ab"):
        _asset(direct_correlation=direct)


def test_hnc_rejects_complex_assets_and_nonintegral_iteration_limits():
    direct = np.zeros((1, 1, *_grid().shape), dtype=complex)
    with pytest.raises(ValueError, match="real-valued"):
        _asset(direct_correlation=direct)

    asset = _asset()
    functional = Route2V0SiteHNCFunctional(
        asset,
        np.zeros((1, *asset.grid.shape)),
    )
    with pytest.raises(ValueError, match="positive integer"):
        functional.solve_picard(max_iterations=1.5)


def test_hnc_accepts_the_mace_gaussian_grid_as_an_electrostatic_external_term():
    grid = RegularCartesianGrid(
        origin_bohr=np.array([-1.0, -1.0, -1.0]),
        spacing_bohr=np.ones(3),
        shape=(3, 3, 3),
    )
    source = evaluate_route2_v0_structured_solvent_electrostatic_source(
        density_coefficients=np.array([[0.4, 0.0, 0.0, 0.0], [-0.4, 0.0, 0.0, 0.0]]),
        atom_positions_angstrom=np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        grid=grid,
    )
    asset = Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("probe",),
        bulk_number_density_bohr3=np.array([0.002]),
        direct_correlation_dimensionless=np.zeros((1, 1, *grid.shape)),
        kbt_hartree=0.001,
    )
    functional = Route2V0SiteHNCFunctional(
        asset,
        source.site_electrostatic_energy_hartree(0.5)[None, :, :, :],
    )

    state = functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.5,
        max_iterations=300,
    )
    assert state.residual_inf < 1.0e-12
    assert np.all(state.density_bohr3 > 0.0)
