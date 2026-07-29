from __future__ import annotations

import math

import numpy as np
import pytest
from ase.units import Bohr, Hartree, kB

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
from maple.function.calculator.extra_correction.implicit.route2_v0_periodic_coulomb import (
    Route2V0PeriodicCoulombOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_promolecular_density import (
    Route2V0PromolecularDensityTable,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_bulk import (
    Route2V0RismShortRangeDirectCorrelation,
    Route2V0RismXvvMetadata,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_energy_conjugate import (
    Route2V0RismEnergyConjugateKernel,
    V0_RISM_ENERGY_CONJUGATE_CONSTRUCTION,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_reciprocal import (
    Route2V0RismShortRangeReciprocalControl,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid(shape: tuple[int, int, int] = (4, 4, 4)) -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.ones(3),
        shape=shape,
    )


def _short_range(grid: RegularCartesianGrid) -> Route2V0RismShortRangeReciprocalControl:
    radii_angstrom = np.linspace(0.0, 6.0, 601)
    radii_bohr = radii_angstrom / Bohr
    profile = np.exp(-0.4 * radii_bohr**2)
    values = np.array([[1.0, 0.25], [0.25, 0.5]])[..., None] * profile
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
    radial = Route2V0RismShortRangeDirectCorrelation(
        metadata=metadata,
        radii_angstrom=radii_angstrom,
        values_dimensionless=values,
    )
    return Route2V0RismShortRangeReciprocalControl.from_radial(
        radial=radial,
        grid=grid,
        tail_start_angstrom=4.0,
        tail_tolerance_dimensionless=1.0e-8,
    )


def _kernel() -> Route2V0RismEnergyConjugateKernel:
    grid = _grid()
    return Route2V0RismEnergyConjugateKernel(
        short_range=_short_range(grid),
        periodic_coulomb=Route2V0PeriodicCoulombOperator(
            grid,
            smear_bohr=1.0 / Bohr,
        ),
    )


def _neutral_site_density_difference(
    kernel: Route2V0RismEnergyConjugateKernel,
    rng: np.random.Generator,
) -> np.ndarray:
    result = rng.normal(
        scale=2.0e-4,
        size=(kernel.site_charge_scales_sqrt_bohr.size, *kernel.grid.shape),
    )
    weighted = np.einsum(
        "a,axyz->xyz",
        kernel.site_charge_scales_sqrt_bohr,
        result,
        optimize=True,
    )
    result[0] -= np.mean(weighted) / kernel.site_charge_scales_sqrt_bohr[0]
    return result


def test_energy_conjugate_rism_full_convolution_equals_the_short_range_plus_poisson_split():
    rng = np.random.default_rng(20260729)
    kernel = _kernel()
    difference = _neutral_site_density_difference(kernel, rng)

    expected = kernel.short_range_convolution(difference) - (
        kernel.periodic_coulomb.site_kernel_from_density_difference(
            difference,
            kernel.site_charge_scales_sqrt_bohr,
        )
    )
    np.testing.assert_allclose(
        kernel.full_convolution(difference),
        expected,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert kernel.construction == V0_RISM_ENERGY_CONJUGATE_CONSTRUCTION
    assert kernel.site_hnc_asset.site_names == ("A", "B")
    np.testing.assert_allclose(
        kernel.bulk_number_density_bohr3,
        np.array([0.01, 0.01]) * Bohr**3,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(kernel.site_multiplicity, np.array([1, 1]))
    assert kernel.kbt_hartree == pytest.approx(kB * 298.0 / Hartree)


def test_energy_conjugate_rism_split_scalar_and_gradient_match_the_full_hnc_kernel():
    rng = np.random.default_rng(29)
    kernel = _kernel()
    difference = _neutral_site_density_difference(kernel, rng)
    direction = _neutral_site_density_difference(kernel, rng)
    step = 1.0e-5

    finite_difference = (
        kernel.excess_energy_hartree(difference + step * direction)
        - kernel.excess_energy_hartree(difference - step * direction)
    ) / (2.0 * step)
    analytic = (
        kernel.kbt_hartree
        * kernel.grid.volume_element_bohr3
        * np.sum(kernel.dimensionless_excess_gradient(difference) * direction)
    )
    full_scalar = (
        -0.5
        * kernel.kbt_hartree
        * kernel.grid.volume_element_bohr3
        * np.sum(difference * kernel.full_convolution(difference))
    )
    short_range, long_range = kernel.excess_energy_components_hartree(difference)

    assert finite_difference == pytest.approx(analytic, rel=2.0e-9, abs=2.0e-14)
    assert kernel.excess_energy_hartree(difference) == pytest.approx(
        full_scalar,
        rel=2.0e-13,
        abs=2.0e-14,
    )
    assert short_range + long_range == pytest.approx(full_scalar, rel=2.0e-13)
    assert long_range >= 0.0


def test_energy_conjugate_rism_rejects_a_mismatched_grid_or_source_smear():
    short_range = _short_range(_grid())
    with pytest.raises(ValueError, match="SMEAR"):
        Route2V0RismEnergyConjugateKernel(
            short_range=short_range,
            periodic_coulomb=Route2V0PeriodicCoulombOperator(
                short_range.grid,
                smear_bohr=0.0,
            ),
        )
    with pytest.raises(ValueError, match="grids differ"):
        Route2V0RismEnergyConjugateKernel(
            short_range=short_range,
            periodic_coulomb=Route2V0PeriodicCoulombOperator(
                _grid(shape=(5, 4, 4)),
                smear_bohr=1.0 / Bohr,
            ),
        )


def test_energy_conjugate_rism_kernel_can_supply_the_molecular_hnc_scalar_without_a_raw_cvv():
    kernel = _kernel()
    grid = kernel.grid
    table = Route2V0PromolecularDensityTable(
        radial_grid_bohr=np.array([0.0, 0.5, 1.0]),
        densities_by_atomic_number={1: np.array([1.0, 0.1, 0.0])},
        table_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )
    configurations = Route2V0MolecularConfigurations(
        translations_bohr=np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]),
        rotations=np.repeat(np.eye(3)[None], 2, axis=0),
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
        solvent=Route2V0MolecularSolventReference(
            atomic_numbers=np.array([1, 1]),
            site_charges_e=np.array([0.0, 0.0]),
            reference_positions_bohr=np.array([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
            provenance_label="synthetic two-site RISM bridge control",
        ),
        configurations=configurations,
    )
    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=configurations,
        phase_space_weights_bohr3=np.full(
            2,
            grid.point_count
            * grid.volume_element_bohr3
            * RIGID_MOLECULAR_ORIENTATION_MEASURE
            / 2.0,
        ),
    )
    occupancy = np.full(
        (kernel.site_charge_scales_sqrt_bohr.size, *grid.shape, 2),
        1.0 / grid.point_count,
    )
    projection = Route2V0MolecularSiteProjection(
        quadrature=quadrature,
        external_potential=external,
        site_hnc_asset=kernel.site_hnc_asset,
        solvent_site_type_indices=np.array([0, 1]),
        site_occupancy_weights=occupancy,
    )
    functional = Route2V0MolecularSiteHNCFunctional(projection)
    density = projection.uniform_configuration_density_bohr3 * np.array([1.1, 0.9])

    state = functional.stationary_state(density, iterations=0)

    assert np.isfinite(state.grand_potential_hartree)
    assert np.isfinite(state.residual_inf)
    np.testing.assert_allclose(
        functional.projected_site_density(
            np.full(2, projection.uniform_configuration_density_bohr3)
        ),
        np.broadcast_to(
            kernel.bulk_number_density_bohr3.reshape((-1, 1, 1, 1)),
            (kernel.bulk_number_density_bohr3.size, *grid.shape),
        ),
        rtol=0.0,
        atol=3.0e-16,
    )
