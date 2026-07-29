from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from maple.function.calculator.calculator_base import ROUTE2_SMD_CALCULATOR_PROFILE
from maple.function.calculator.extra_correction.implicit.route2_v0_mace_cluster_external_potential import (
    Route2V0MaceClusterMolecularExternalPotential,
    Route2V0MaceZeroFieldClusterInteraction,
    V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_CONSTRUCTION,
    V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_SCOPE,
    evaluate_route2_v0_mace_cluster_molecular_external_potential,
    evaluate_route2_v0_mace_zero_field_cluster_interaction,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential_contract import (
    Route2V0MolecularExternalPotentialContract,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_ideal_gas import (
    RIGID_MOLECULAR_ORIENTATION_MEASURE,
    Route2V0MolecularConfigurationQuadrature,
    Route2V0MolecularIdealGasFunctional,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_site_hnc import (
    Route2V0SiteHNCAsset,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)
from maple.function.route2_smd_profiles import MACEPOL_MOLECULAR_REALSPACE_PROFILE


class _FakeZeroFieldMACE:
    """Translation-invariant scalar MACE stand-in with analytic forces."""

    mace_polar_checkpoint_provenance = {
        "identifier": "polar-1-m",
        "release_url": (
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
            "mace_polar_1/MACE-POLAR-1-M.model"
        ),
        "sha256": "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a",
        "size_bytes": 68_133_235,
    }
    mace_torch_version = "0.3.16"
    route2_smd_profile = ROUTE2_SMD_CALCULATOR_PROFILE
    long_range_evaluator_profile = MACEPOL_MOLECULAR_REALSPACE_PROFILE

    def __init__(self, *, omit_forces: bool = False) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.omit_forces = omit_forces

    def polar_state(self, atoms: Atoms, *, compute_forces: bool):
        self.calls.append((len(atoms), compute_forces))
        positions = atoms.get_positions()
        numbers = np.asarray(atoms.numbers, dtype=float)
        energy = float(np.sum(0.11 * numbers))
        forces = np.zeros_like(positions)
        for left in range(len(atoms)):
            for right in range(left + 1, len(atoms)):
                displacement = positions[left] - positions[right]
                coefficient = 0.01 * numbers[left] * numbers[right]
                pair_energy = coefficient * np.exp(-np.dot(displacement, displacement))
                energy += pair_energy
                pair_force = 2.0 * pair_energy * displacement
                forces[left] += pair_force
                forces[right] -= pair_force
        return (
            SimpleNamespace(
                energy_ev=energy,
                fixed_field_forces_ev_per_angstrom=(
                    None if self.omit_forces else forces
                ),
            ),
            {},
        )


def _solute(*, shift: np.ndarray | None = None) -> Atoms:
    positions = np.array([[0.0, 0.0, 0.0]])
    if shift is not None:
        positions += shift
    return Atoms("H", positions=positions, pbc=False, info={"charge": 0.0, "mult": 1})


def _solvent(*, shift: np.ndarray | None = None) -> Atoms:
    positions = np.array([[1.3, 0.2, -0.1]])
    if shift is not None:
        positions += shift
    return Atoms("O", positions=positions, pbc=False, info={"charge": 0.0, "mult": 1})


def _evaluate(
    *,
    calculator: _FakeZeroFieldMACE | None = None,
    solute: Atoms | None = None,
    solvent: Atoms | None = None,
    compute_forces: bool = False,
) -> Route2V0MaceZeroFieldClusterInteraction:
    return evaluate_route2_v0_mace_zero_field_cluster_interaction(
        calculator=_FakeZeroFieldMACE() if calculator is None else calculator,
        solute=_solute() if solute is None else solute,
        solvent=_solvent() if solvent is None else solvent,
        compute_forces=compute_forces,
    )


def test_zero_field_cluster_interaction_is_the_exact_three_energy_difference():
    calculator = _FakeZeroFieldMACE()
    state = _evaluate(calculator=calculator)

    assert state.construction == V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_CONSTRUCTION
    assert state.interaction_scope == V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_SCOPE
    assert state.interaction_energy_ev == pytest.approx(
        state.cluster_energy_ev - state.solute_energy_ev - state.solvent_energy_ev
    )
    assert state.interaction_energy_hartree == pytest.approx(
        state.interaction_energy_ev / Hartree
    )
    assert state.forces_available is False
    assert state.solute_interaction_forces_ev_per_angstrom is None
    assert state.solvent_interaction_forces_ev_per_angstrom is None
    assert calculator.calls == [(1, False), (1, False), (2, False)]


def test_zero_field_cluster_interaction_force_is_the_energy_derivative_and_is_covariant():
    state = _evaluate(compute_forces=True)
    assert state.forces_available is True
    assert state.solute_interaction_forces_ev_per_angstrom is not None
    assert state.solvent_interaction_forces_ev_per_angstrom is not None
    np.testing.assert_allclose(
        state.solute_interaction_forces_ev_per_angstrom
        + state.solvent_interaction_forces_ev_per_angstrom,
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    step = 1.0e-5
    plus = _solute()
    minus = _solute()
    plus.positions[0, 0] += step
    minus.positions[0, 0] -= step
    numerical_force = -(
        _evaluate(solute=plus).interaction_energy_ev
        - _evaluate(solute=minus).interaction_energy_ev
    ) / (2.0 * step)
    assert state.solute_interaction_forces_ev_per_angstrom[0, 0] == pytest.approx(
        numerical_force,
        rel=1.0e-9,
        abs=1.0e-11,
    )

    shift = np.array([0.7, -0.5, 0.3])
    shifted = _evaluate(
        solute=_solute(shift=shift),
        solvent=_solvent(shift=shift),
        compute_forces=True,
    )
    assert shifted.interaction_energy_ev == pytest.approx(
        state.interaction_energy_ev,
        abs=1.0e-15,
    )
    np.testing.assert_allclose(
        shifted.solute_interaction_forces_ev_per_angstrom,
        state.solute_interaction_forces_ev_per_angstrom,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        shifted.solvent_interaction_forces_ev_per_angstrom,
        state.solvent_interaction_forces_ev_per_angstrom,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_zero_field_cluster_interaction_rejects_response_or_fragment_shortcuts():
    calculator = _FakeZeroFieldMACE()
    calculator.long_range_evaluator_profile = "forced-box"
    with pytest.raises(ValueError, match="real-space long-range evaluator"):
        _evaluate(calculator=calculator)

    charged = _solute()
    charged.info["charge"] = 1.0
    with pytest.raises(ValueError, match="must be neutral"):
        _evaluate(solute=charged)

    open_shell = _solvent()
    open_shell.info["mult"] = 2
    with pytest.raises(ValueError, match="must be a singlet"):
        _evaluate(solvent=open_shell)

    periodic = _solvent()
    periodic.pbc = True
    with pytest.raises(ValueError, match="nonperiodic"):
        _evaluate(solvent=periodic)

    with pytest.raises(ValueError, match="zero-field forces"):
        _evaluate(calculator=_FakeZeroFieldMACE(omit_forces=True), compute_forces=True)


def test_zero_field_cluster_state_rejects_a_broken_energy_or_force_ledger():
    state = _evaluate(compute_forces=True)
    with pytest.raises(ValueError, match="cluster minus both"):
        replace(state, interaction_energy_ev=state.interaction_energy_ev + 0.1)
    with pytest.raises(ValueError, match="supplied together"):
        replace(state, solvent_interaction_forces_ev_per_angstrom=None)


def test_cluster_source_is_a_whole_molecular_external_potential_for_the_stationary_hnc_scalar():
    grid = RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.ones(3),
        shape=(2, 1, 1),
    )
    solvent = Route2V0MolecularSolventReference(
        atomic_numbers=np.array([8]),
        site_charges_e=np.array([0.0]),
        reference_positions_bohr=np.zeros((1, 3)),
        provenance_label="synthetic neutral oxygen site for MACE scalar tests",
    )
    configurations = Route2V0MolecularConfigurations(
        translations_bohr=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        rotations=np.repeat(np.eye(3)[None], 2, axis=0),
    )
    source = evaluate_route2_v0_mace_cluster_molecular_external_potential(
        calculator=_FakeZeroFieldMACE(),
        integration_grid=grid,
        solute_atomic_numbers=np.array([1]),
        solute_atom_positions_angstrom=np.array([[-1.0, 0.0, 0.0]]),
        solvent=solvent,
        configurations=configurations,
        compute_forces=True,
    )

    assert isinstance(source, Route2V0MaceClusterMolecularExternalPotential)
    assert isinstance(source, Route2V0MolecularExternalPotentialContract)
    assert source.forces_available is True
    np.testing.assert_allclose(
        source.external_potential_hartree,
        [
            interaction.interaction_energy_ev / Hartree
            for interaction in source.interactions
        ],
        rtol=0.0,
        atol=1.0e-15,
    )
    expected_positions = configurations.site_positions_bohr(solvent) * Bohr
    for index, interaction in enumerate(source.interactions):
        np.testing.assert_allclose(
            interaction.solvent_positions_angstrom,
            expected_positions[index],
            rtol=0.0,
            atol=0.0,
        )

    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=configurations,
        phase_space_weights_bohr3=np.full(
            configurations.configuration_count,
            grid.volume_element_bohr3 * RIGID_MOLECULAR_ORIENTATION_MEASURE,
        ),
    )
    ideal = Route2V0MolecularIdealGasFunctional(
        quadrature=quadrature,
        external_potential=source,
        bulk_molecular_number_density_bohr3=0.1,
        kbt_hartree=1.0,
    )
    ideal_state = ideal.solve_analytic()
    assert ideal_state.residual_inf < 1.0e-14

    asset = Route2V0SiteHNCAsset(
        grid=grid,
        site_names=("O",),
        bulk_number_density_bohr3=np.array([0.1]),
        direct_correlation_dimensionless=np.zeros((1, 1, *grid.shape)),
        kbt_hartree=1.0,
    )
    occupancy = np.zeros((1, *grid.shape, configurations.configuration_count))
    occupancy[0, 0, 0, 0, 0] = 1.0
    occupancy[0, 1, 0, 0, 1] = 1.0
    projection = Route2V0MolecularSiteProjection(
        quadrature=quadrature,
        external_potential=source,
        site_hnc_asset=asset,
        solvent_site_type_indices=np.array([0]),
        site_occupancy_weights=occupancy,
    )
    molecular_hnc = Route2V0MolecularSiteHNCFunctional(projection=projection)
    hnc_state = molecular_hnc.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=1.0,
        max_iterations=10,
    )
    assert hnc_state.residual_inf < 1.0e-12
    np.testing.assert_allclose(
        hnc_state.configuration_density_bohr3,
        ideal_state.configuration_density_bohr3,
        rtol=0.0,
        atol=1.0e-14,
    )
