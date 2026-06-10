"""WS3 — single combined backend evaluation with a typed, unit-named result."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT
from maple.function.dispatcher.md.evaluator import MDProperties, evaluate_md_properties
from maple.function.dispatcher.md.utils import (
    HA_PER_ANG_TO_AU,
    compute_instantaneous_pressure,
)


class _CountingCalc(Calculator):
    """Property-selective calculator that counts backend evaluations."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, forces, stress):
        super().__init__()
        self.maple_model_name = "counting"
        self.maple_pbc_md_supported = True
        self.maple_stress_supported = True
        self.maple_stress_unit = ASE_STRESS_UNIT
        self._forces = np.asarray(forces, dtype=float)
        self._stress = np.asarray(stress, dtype=float)
        self.n_calculate = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.n_calculate += 1
        self.results["energy"] = 0.0
        self.results["forces"] = self._forces.copy()
        # Only populate stress when asked, mimicking a property-selective backend.
        if "stress" in properties:
            self.results["stress"] = self._stress.copy()


def _periodic(calc):
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.calc = calc
    return atoms


def test_returns_typed_properties_without_stress():
    forces = np.array([[0.5, -0.25, 0.125]])
    atoms = _periodic(_CountingCalc(forces, np.zeros(6)))
    props = evaluate_md_properties(atoms, need_stress=False)
    assert isinstance(props, MDProperties)
    np.testing.assert_allclose(props.forces_ha_per_ang, forces)
    np.testing.assert_allclose(props.forces_au, forces * HA_PER_ANG_TO_AU)
    assert props.stress_ev_per_ang3 is None
    assert props.pressure_bar is None


def test_returns_stress_and_pressure_when_requested():
    stress = np.array([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0])
    atoms = _periodic(_CountingCalc(np.zeros((1, 3)), stress))
    velocities = np.zeros((1, 3))
    props = evaluate_md_properties(atoms, need_stress=True, velocities_au=velocities)
    np.testing.assert_allclose(props.stress_ev_per_ang3, stress)
    assert props.pressure_bar == pytest.approx(
        compute_instantaneous_pressure(atoms, velocities)
    )


def test_pressure_request_can_exclude_com_kinetic_term():
    atoms = Atoms(
        "Ar2",
        positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.calc = _CountingCalc(np.zeros((2, 3)), np.zeros(6))
    velocities = np.tile([0.01, 0.0, 0.0], (2, 1))

    props = evaluate_md_properties(
        atoms,
        need_stress=True,
        velocities_au=velocities,
        exclude_com_kinetic=True,
    )

    assert props.pressure_bar == pytest.approx(0.0, abs=1e-12)


def test_pressure_with_prevalidated_stress_still_checks_unit_contract():
    atoms = _periodic(_CountingCalc(np.zeros((1, 3)), np.zeros(6)))
    atoms.calc.maple_stress_unit = "GPa"

    with pytest.raises(ValueError, match="stress unit contract mismatch"):
        compute_instantaneous_pressure(
            atoms,
            np.zeros((1, 3)),
            stress_ev_per_ang3=np.zeros(6),
        )


def test_single_backend_evaluation_for_energy_forces_stress():
    # Perf regression: requesting energy + forces + stress (+ pressure) must cost
    # exactly one backend evaluation, even for a property-selective calculator.
    calc = _CountingCalc(np.array([[1.0, 0.0, 0.0]]), np.zeros(6))
    atoms = _periodic(calc)
    evaluate_md_properties(atoms, need_stress=True, velocities_au=np.zeros((1, 3)))
    assert calc.n_calculate == 1


def test_reuses_cache_when_geometry_unchanged():
    # The integrator typically just evaluated forces; routing the logged energy
    # read through the evaluator at the same geometry must not recompute.
    calc = _CountingCalc(np.array([[1.0, 0.0, 0.0]]), np.zeros(6))
    atoms = _periodic(calc)
    atoms.get_forces()  # one backend pass (energy + forces)
    assert calc.n_calculate == 1
    props = evaluate_md_properties(atoms, need_stress=False)
    assert calc.n_calculate == 1  # no extra backend pass
    np.testing.assert_allclose(props.forces_ha_per_ang, [[1.0, 0.0, 0.0]])


def test_recomputes_once_when_only_stress_missing():
    # energy+forces cached but stress not yet computed (property-selective calc):
    # need_stress must trigger exactly one more pass, and a second need_stress
    # call at the same geometry must reuse the now-cached stress.
    calc = _CountingCalc(np.zeros((1, 3)), np.array([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]))
    atoms = _periodic(calc)
    atoms.get_forces()
    assert calc.n_calculate == 1
    props = evaluate_md_properties(atoms, need_stress=True, velocities_au=np.zeros((1, 3)))
    assert calc.n_calculate == 2
    assert props.stress_ev_per_ang3 is not None
    evaluate_md_properties(atoms, need_stress=True, velocities_au=np.zeros((1, 3)))
    assert calc.n_calculate == 2  # stress now cached -> no further pass


def test_md_properties_is_frozen():
    atoms = _periodic(_CountingCalc(np.zeros((1, 3)), np.zeros(6)))
    props = evaluate_md_properties(atoms)
    with pytest.raises(Exception):
        props.energy_ha = 1.0


def test_requires_attached_calculator():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
    with pytest.raises(ValueError, match="calculator"):
        evaluate_md_properties(atoms)
