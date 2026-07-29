from __future__ import annotations

import math

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_v0_rism_bulk import (
    Route2V0RismShortRangeDirectCorrelation,
    Route2V0RismXvvMetadata,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_reciprocal import (
    Route2V0RismShortRangeReciprocalControl,
    V0_RISM_SHORT_RANGE_RECIPROCAL_CONSTRUCTION,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_site_hnc import (
    Route2V0SiteHNCAsset,
    Route2V0SiteHNCFunctional,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)

ALPHA_BOHR_MINUS2 = 0.4
PAIR_SCALE = np.array([[1.0, 0.25], [0.25, 0.5]])


def _grid(*, spacing_bohr: float = 1.0) -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.full(3, spacing_bohr),
        shape=(4, 4, 4),
    )


def _radial_short_range(
    *,
    constant_tail: bool = False,
    omit_origin: bool = False,
) -> Route2V0RismShortRangeDirectCorrelation:
    radii_angstrom = np.linspace(0.0, 6.0, 601)
    if omit_origin:
        radii_angstrom = radii_angstrom[1:]
    radii_bohr = radii_angstrom / Bohr
    if constant_tail:
        profile = np.full(radii_bohr.size, 0.01)
    else:
        profile = np.exp(-ALPHA_BOHR_MINUS2 * radii_bohr**2)
    values = PAIR_SCALE[..., None] * profile
    metadata = Route2V0RismXvvMetadata(
        site_names=("A", "B"),
        site_multiplicity=np.array([1, 1]),
        bulk_number_density_angstrom3=np.array([0.01, 0.01]),
        site_charges_sqrt_kT_angstrom=np.array([-1.0, 1.0]),
        temperature_kelvin=298.0,
        dielectric_constant=78.5,
        coulomb_smear_angstrom=1.0,
        radial_spacing_angstrom=0.01,
        radial_point_count=radii_angstrom.size,
        component_count=1,
    )
    return Route2V0RismShortRangeDirectCorrelation(
        metadata=metadata,
        radii_angstrom=radii_angstrom,
        values_dimensionless=values,
    )


def _wavevector_squared(grid: RegularCartesianGrid) -> np.ndarray:
    axes = tuple(
        2.0 * math.pi * np.fft.fftfreq(grid.shape[axis], d=grid.spacing_bohr[axis])
        for axis in range(3)
    )
    return sum(axis**2 for axis in np.meshgrid(*axes, indexing="ij"))


def test_reciprocal_control_matches_a_gaussian_radial_fourier_transform():
    grid = _grid()
    control = Route2V0RismShortRangeReciprocalControl.from_radial(
        radial=_radial_short_range(),
        grid=grid,
        tail_start_angstrom=4.0,
        tail_tolerance_dimensionless=1.0e-8,
    )
    transformed = grid.volume_element_bohr3 * np.fft.fftn(
        control.direct_correlation_dimensionless,
        axes=(2, 3, 4),
    )
    expected_scalar = (math.pi / ALPHA_BOHR_MINUS2) ** 1.5 * np.exp(
        -_wavevector_squared(grid) / (4.0 * ALPHA_BOHR_MINUS2)
    )
    expected = PAIR_SCALE[..., None, None, None] * expected_scalar

    np.testing.assert_allclose(
        np.real(transformed),
        expected,
        rtol=3.0e-9,
        atol=3.0e-11,
    )
    assert np.max(np.abs(np.imag(transformed))) < 3.0e-14
    assert control.maximum_tail_abs < control.tail_tolerance_dimensionless
    assert control.construction == V0_RISM_SHORT_RANGE_RECIPROCAL_CONSTRUCTION


def test_reciprocal_control_preserves_the_hnc_reciprocal_pairing_and_scalar_gradient():
    rng = np.random.default_rng(20260729)
    grid = _grid()
    control = Route2V0RismShortRangeReciprocalControl.from_radial(
        radial=_radial_short_range(),
        grid=grid,
        tail_start_angstrom=4.0,
        tail_tolerance_dimensionless=1.0e-8,
    )
    reverse = np.ix_(*((-np.arange(extent)) % extent for extent in grid.shape))
    direct = control.direct_correlation_dimensionless
    np.testing.assert_allclose(direct[0, 1], direct[1, 0][reverse])

    asset = Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("A", "B"),
        bulk_number_density_bohr3=np.array([0.01, 0.01]) * Bohr**3,
        direct_correlation_dimensionless=direct,
        kbt_hartree=0.001,
    )
    functional = Route2V0SiteHNCFunctional(
        asset,
        np.zeros((asset.site_count, *grid.shape)),
    )
    bulk = asset.bulk_number_density_bohr3.reshape((asset.site_count, 1, 1, 1))
    density = bulk * (1.0 + 0.02 * rng.uniform(-1.0, 1.0, size=(2, *grid.shape)))
    direction = bulk * rng.uniform(-1.0, 1.0, size=(2, *grid.shape))
    epsilon = 1.0e-6

    finite_difference = (
        functional.grand_potential_hartree(density + epsilon * direction)
        - functional.grand_potential_hartree(density - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = (
        asset.kbt_hartree
        * grid.volume_element_bohr3
        * np.sum(functional.dimensionless_gradient(density) * direction)
    )
    assert finite_difference == pytest.approx(analytic, rel=3.0e-8, abs=3.0e-12)


def test_reciprocal_control_rejects_an_unregistered_tail_or_radial_aliasing():
    with pytest.raises(ValueError, match="tail exceeds"):
        Route2V0RismShortRangeReciprocalControl.from_radial(
            radial=_radial_short_range(constant_tail=True),
            grid=_grid(),
            tail_start_angstrom=4.0,
            tail_tolerance_dimensionless=1.0e-4,
        )

    with pytest.raises(ValueError, match="Nyquist"):
        Route2V0RismShortRangeReciprocalControl.from_radial(
            radial=_radial_short_range(),
            grid=_grid(spacing_bohr=0.001),
            tail_start_angstrom=4.0,
            tail_tolerance_dimensionless=1.0e-8,
        )

    with pytest.raises(ValueError, match="r=0"):
        Route2V0RismShortRangeReciprocalControl.from_radial(
            radial=_radial_short_range(omit_origin=True),
            grid=_grid(),
            tail_start_angstrom=4.0,
            tail_tolerance_dimensionless=1.0e-8,
        )
