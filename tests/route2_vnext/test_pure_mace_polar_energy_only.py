from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.route2._mace_polar_frozen_ddx_calculator import (
    PureMACEPolarDDXCalculator,
)
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
    route2_smd_profile_spec,
)


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=((0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)),
        info={"charge": 0, "mult": 1},
    )


class _CountingPES:
    provider_id = "test.counting-pure-frozen-total-pes"

    def __init__(self) -> None:
        profile = route2_smd_profile_spec(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE
        )
        self.scalar_contract_id = profile.scalar_contract_id
        self.solve_calls = 0
        self.force_calls = 0
        self.force_central_states: list[object | None] = []

    def solve(self, atoms: Atoms) -> SimpleNamespace:
        self.solve_calls += 1
        positions = np.asarray(atoms.positions, dtype=float)
        return SimpleNamespace(
            total_energy_eV=float(2.0 + 0.5 * np.vdot(positions, positions)),
            state_id=self.solve_calls,
        )

    def evaluate_forces(
        self,
        atoms: Atoms,
        *,
        central_state: object | None = None,
    ) -> SimpleNamespace:
        self.force_calls += 1
        self.force_central_states.append(central_state)
        state = central_state if central_state is not None else self.solve(atoms)
        return SimpleNamespace(
            central_state=state,
            total_forces_eV_per_A=-np.asarray(atoms.positions, dtype=float),
        )


def _calculator(pes: _CountingPES) -> PureMACEPolarDDXCalculator:
    return PureMACEPolarDDXCalculator._from_test_pes(
        atoms=_water(), solvent="water", pes=pes
    )


@pytest.mark.parametrize("property_name", ("energy", "free_energy"))
def test_energy_properties_solve_without_evaluating_forces(property_name: str) -> None:
    atoms = _water()
    pes = _CountingPES()
    calculator = _calculator(pes)
    atoms.calc = calculator

    value = calculator.get_property(property_name, atoms)

    assert np.isfinite(value)
    assert pes.solve_calls == 1
    assert pes.force_calls == 0
    assert set(calculator.results) == {"energy", "free_energy"}


def test_energy_then_forces_reuses_the_energy_state() -> None:
    atoms = _water()
    pes = _CountingPES()
    calculator = _calculator(pes)
    atoms.calc = calculator

    energy = atoms.get_potential_energy()
    energy_state = calculator.last_energy_state
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    np.testing.assert_allclose(forces, -atoms.positions)
    assert pes.solve_calls == 1
    assert pes.force_calls == 1
    assert pes.force_central_states == [energy_state]
    assert calculator.last_energy_state is energy_state


def test_forces_then_energy_uses_the_force_central_state() -> None:
    atoms = _water()
    pes = _CountingPES()
    calculator = _calculator(pes)
    atoms.calc = calculator

    forces = atoms.get_forces()
    force_state = calculator.last_energy_state
    energy = atoms.get_potential_energy()

    np.testing.assert_allclose(forces, -atoms.positions)
    assert np.isfinite(energy)
    assert pes.solve_calls == 1
    assert pes.force_calls == 1
    assert pes.force_central_states == [None]
    assert calculator.last_energy_state is force_state


def test_geometry_change_invalidates_energy_and_force_caches() -> None:
    atoms = _water()
    original_positions = atoms.positions.copy()
    pes = _CountingPES()
    calculator = _calculator(pes)
    atoms.calc = calculator

    first_energy = atoms.get_potential_energy()
    atoms.get_forces()
    assert (pes.solve_calls, pes.force_calls) == (1, 1)

    atoms.positions[1, 0] += 0.05
    second_energy = atoms.get_potential_energy()
    assert second_energy != first_energy
    assert (pes.solve_calls, pes.force_calls) == (2, 1)

    atoms.get_forces()
    assert (pes.solve_calls, pes.force_calls) == (2, 2)
    assert pes.force_central_states[-1] is calculator.last_energy_state

    atoms.positions[:] = original_positions
    atoms.get_forces()
    assert (pes.solve_calls, pes.force_calls) == (3, 3)
    assert pes.force_central_states[-1] is None


def test_combined_energy_and_forces_request_uses_force_path() -> None:
    atoms = _water()
    pes = _CountingPES()
    calculator = _calculator(pes)

    calculator.calculate(atoms, properties=["energy", "forces"])

    assert pes.solve_calls == 1
    assert pes.force_calls == 1
    assert set(calculator.results) == {"energy", "free_energy", "forces"}
