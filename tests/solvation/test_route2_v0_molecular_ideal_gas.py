from __future__ import annotations

from dataclasses import replace
import math

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
    Route2V0MolecularIdealGasFunctional,
    V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_promolecular_density import (
    Route2V0PromolecularDensityTable,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _external_potential(
    *,
    translations_bohr: np.ndarray | None = None,
):
    grid = RegularCartesianGrid(
        origin_bohr=np.array([-2.0, -2.0, -2.0]),
        spacing_bohr=np.ones(3),
        shape=(5, 5, 5),
    )
    table = Route2V0PromolecularDensityTable(
        radial_grid_bohr=np.array([0.0, 0.5, 1.0]),
        densities_by_atomic_number={
            1: np.array([1.0, 0.2, 0.0]),
            8: np.array([3.0, 0.4, 0.0]),
        },
        table_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )
    configurations = Route2V0MolecularConfigurations(
        translations_bohr=(
            np.array([[6.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
            if translations_bohr is None
            else translations_bohr
        ),
        rotations=np.repeat(np.eye(3)[None], 2, axis=0),
    )
    return evaluate_route2_v0_molecular_external_potential(
        integration_grid=grid,
        promolecular_table=table,
        solute_atomic_numbers=np.array([1]),
        solute_atom_positions_angstrom=np.zeros((1, 3)),
        mace_source=Route2V0FrozenMaceGaussianSource(
            density_coefficients=np.array([[0.5, 0.0, 0.0, 0.0]]),
            atom_positions_angstrom=np.zeros((1, 3)),
            target_total_charge_e=0.5,
        ),
        solvent=Route2V0MolecularSolventReference(
            atomic_numbers=np.array([1, 8]),
            site_charges_e=np.array([0.3, -0.3]),
            reference_positions_bohr=np.array([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
            provenance_label="synthetic neutral rigid molecular control",
        ),
        configurations=configurations,
    )


def _functional() -> Route2V0MolecularIdealGasFunctional:
    external = _external_potential()
    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=external.configurations,
        phase_space_weights_bohr3=np.array([0.4, 0.7]),
    )
    return Route2V0MolecularIdealGasFunctional(
        quadrature=quadrature,
        external_potential=external,
        bulk_molecular_number_density_bohr3=0.02,
        kbt_hartree=0.01,
    )


def test_molecular_ideal_gas_has_the_exact_configuration_space_stationary_state():
    functional = _functional()
    state = functional.solve_analytic()
    potential = functional.external_potential.external_potential_hartree
    bulk = functional.bulk_molecular_number_density_bohr3 / (8.0 * math.pi**2)
    expected_density = bulk * np.exp(-potential / functional.kbt_hartree)
    expected_grand_potential = (
        -functional.kbt_hartree
        * bulk
        * np.sum(
            functional.quadrature.phase_space_weights_bohr3
            * (np.exp(-potential / functional.kbt_hartree) - 1.0)
        )
    )

    assert functional.construction == V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION
    assert functional.quadrature.orientation_measure == pytest.approx(
        RIGID_MOLECULAR_ORIENTATION_MEASURE
    )
    np.testing.assert_allclose(
        state.configuration_density_bohr3,
        expected_density,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert state.residual_inf < 1.0e-14
    assert state.grand_potential_hartree == pytest.approx(
        expected_grand_potential,
        rel=0.0,
        abs=2.0e-16,
    )
    assert state.grand_potential_hartree == pytest.approx(
        state.ideal_contribution_hartree + state.external_contribution_hartree,
        rel=0.0,
        abs=2.0e-16,
    )


def test_molecular_ideal_gas_gradient_is_the_derivative_of_its_single_scalar():
    functional = _functional()
    bulk = functional.uniform_configuration_density_bohr3
    density = bulk * np.array([1.2, 0.8])
    direction = bulk * np.array([0.15, -0.1])
    step = 1.0e-5

    finite_difference = (
        functional.grand_potential_hartree(density + step * direction)
        - functional.grand_potential_hartree(density - step * direction)
    ) / (2.0 * step)
    analytic = functional.kbt_hartree * np.sum(
        functional.quadrature.phase_space_weights_bohr3
        * functional.dimensionless_gradient(density)
        * direction
    )
    assert finite_difference == pytest.approx(analytic, rel=2.0e-10, abs=2.0e-15)


def test_molecular_ideal_gas_recovers_the_declared_bulk_reference_at_zero_external_energy():
    functional = _functional()
    zero_external = replace(
        functional.external_potential,
        pauli_repulsion_hartree=np.zeros(2),
        electrostatic_energy_hartree=np.zeros(2),
        external_potential_hartree=np.zeros(2),
    )
    zero_functional = replace(functional, external_potential=zero_external)

    state = zero_functional.solve_analytic()

    np.testing.assert_allclose(
        state.configuration_density_bohr3,
        zero_functional.uniform_configuration_density_bohr3,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert state.residual_inf == pytest.approx(0.0, abs=1.0e-15)
    assert state.grand_potential_hartree == pytest.approx(0.0, abs=2.0e-16)


def test_molecular_ideal_gas_rejects_mixed_configuration_grids_and_conventions():
    external = _external_potential()
    different_configurations = Route2V0MolecularConfigurations(
        translations_bohr=np.array([[6.1, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        rotations=np.repeat(np.eye(3)[None], 2, axis=0),
    )
    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=different_configurations,
        phase_space_weights_bohr3=np.array([0.4, 0.7]),
    )
    with pytest.raises(ValueError, match="configurations differ"):
        Route2V0MolecularIdealGasFunctional(
            quadrature=quadrature,
            external_potential=external,
            bulk_molecular_number_density_bohr3=0.02,
            kbt_hartree=0.01,
        )
    with pytest.raises(ValueError, match=r"8\*pi\*\*2"):
        Route2V0MolecularConfigurationQuadrature(
            configurations=external.configurations,
            phase_space_weights_bohr3=np.array([0.4, 0.7]),
            orientation_measure=1.0,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        Route2V0MolecularConfigurationQuadrature(
            configurations=external.configurations,
            phase_space_weights_bohr3=np.array([0.4, 0.0]),
        )


def test_molecular_ideal_gas_rejects_nonpositive_trial_configuration_density():
    functional = _functional()
    with pytest.raises(ValueError, match="strictly positive"):
        functional.stationary_state(
            np.array([functional.uniform_configuration_density_bohr3, 0.0])
        )
