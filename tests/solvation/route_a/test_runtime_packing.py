from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.solvfe.packing import (
    CavityBiasedCalculator,
    GhostObservationCavity,
)


class _ZeroWaterCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
        }


def _cavity() -> GhostObservationCavity:
    return GhostObservationCavity(
        centers_angstrom=np.array([[0.0, 0.0, 0.0]]),
        radii_angstrom=np.array([1.5]),
        lambda_s_angstrom=1.0,
        force_constant_ev_per_angstrom2=2.0,
        measure_id="fixed-ghost-acetone-frame-v1",
    )


def test_ghost_cavity_uses_minimum_image_and_pushes_oxygen_outward():
    atoms = Atoms(
        "OH2",
        positions=[
            [9.8, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [9.8, 0.8, 0.0],
        ],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    evaluation = _cavity().evaluate(atoms, oxygen_indices=(0,))

    assert evaluation.empty is False
    assert evaluation.signed_distances_angstrom[0] == pytest.approx(-1.3)
    assert evaluation.penetration_angstrom[0] == pytest.approx(2.3)
    assert evaluation.energy_ev == pytest.approx(0.5 * 2.0 * 2.3**2)
    assert evaluation.forces_ev_per_angstrom[0, 0] < 0.0
    np.testing.assert_allclose(
        evaluation.forces_ev_per_angstrom[1:],
        0.0,
    )


def test_ghost_cavity_force_matches_finite_difference():
    atoms = Atoms(
        "O",
        positions=[[2.0, 0.0, 0.0]],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    cavity = _cavity()
    step = 1.0e-6
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[0, 0] += step
    minus.positions[0, 0] -= step
    finite_difference_force = -(
        cavity.evaluate(plus, oxygen_indices=(0,)).energy_ev
        - cavity.evaluate(minus, oxygen_indices=(0,)).energy_ev
    ) / (2.0 * step)
    analytic = cavity.evaluate(
        atoms,
        oxygen_indices=(0,),
    ).forces_ev_per_angstrom[0, 0]

    assert analytic == pytest.approx(finite_difference_force, abs=1.0e-8)


def test_cavity_biased_calculator_adds_energy_and_force_once():
    atoms = Atoms(
        "O",
        positions=[[2.0, 0.0, 0.0]],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    atoms.calc = CavityBiasedCalculator(
        base_calculator=_ZeroWaterCalculator(),
        cavity=_cavity(),
        oxygen_indices=(0,),
        scale=0.25,
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert energy == pytest.approx(0.25 * 0.25)
    assert forces[0, 0] == pytest.approx(0.25)
    assert atoms.calc.results["components"]["bias_scale"] == 0.25
    assert atoms.calc.results["components"]["empty"] is False


def test_ghost_cavity_marks_volume_empty_only_beyond_shell_boundary():
    atoms = Atoms(
        "O",
        positions=[[3.0, 0.0, 0.0]],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )

    result = _cavity().evaluate(atoms, oxygen_indices=(0,))

    assert result.empty is True
    assert result.energy_ev == 0.0
    np.testing.assert_allclose(result.forces_ev_per_angstrom, 0.0)
