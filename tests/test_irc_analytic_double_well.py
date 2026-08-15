"""Model-independent propagation canary for all molecular IRC integrators."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.irc.algorithm.eulerpc import EulerPC, EulerPCParams
from maple.function.dispatcher.irc.algorithm.gs import GS, GSParams
from maple.function.dispatcher.irc.algorithm.hpc import HPC, HPCParams
from maple.function.dispatcher.irc.algorithm.lqa import LQA, LQAParams


class _LegacyPairDoubleWellCalculator(Calculator):
    """Translation/rotation-invariant analytic potential in IRC legacy units."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, *, saddle_distance_A=1.5, quartic=0.2, quadratic=0.1):
        super().__init__()
        self.saddle_distance_A = float(saddle_distance_A)
        self.quartic = float(quartic)
        self.quadratic = float(quadratic)

    def _values(self, atoms):
        displacement = atoms.positions[1] - atoms.positions[0]
        distance = float(np.linalg.norm(displacement))
        unit = displacement / distance
        coordinate = distance - self.saddle_distance_A

        energy = self.quartic * coordinate**4 - self.quadratic * coordinate**2
        radial_gradient = (
            4.0 * self.quartic * coordinate**3
            - 2.0 * self.quadratic * coordinate
        )
        radial_curvature = (
            12.0 * self.quartic * coordinate**2 - 2.0 * self.quadratic
        )

        forces = np.zeros((2, 3))
        forces[0] = radial_gradient * unit
        forces[1] = -radial_gradient * unit

        longitudinal = np.outer(unit, unit)
        pair_hessian = radial_curvature * longitudinal + (
            radial_gradient / distance
        ) * (np.eye(3) - longitudinal)
        hessian = np.block(
            [[pair_hessian, -pair_hessian], [-pair_hessian, pair_hessian]]
        )
        return float(energy), forces, hessian

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, forces, _ = self._values(atoms)
        # The four IRC implementations receive this object directly as their
        # private Hartree job view, so these intentionally are not public ASE
        # eV-family values.
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }

    def get_hessian(self, atoms=None):
        selected = self.atoms if atoms is None else atoms
        return self._values(selected)[2]


@pytest.mark.parametrize(
    ("integrator_class", "params_class"),
    [
        (GS, GSParams),
        (LQA, LQAParams),
        (HPC, HPCParams),
        (EulerPC, EulerPCParams),
    ],
)
def test_integrator_connects_both_analytic_wells_without_model_fixture(
    tmp_path,
    integrator_class,
    params_class,
):
    atoms = Atoms(
        "H2",
        positions=[[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]],
    )
    atoms.calc = _LegacyPairDoubleWellCalculator()
    params = params_class(
        max_steps=50,
        # This fixture checks propagation and dual-threshold termination, not
        # the stricter production default. Both values are explicit legacy
        # Hartree/Angstrom thresholds for this analytic canary.
        f_max_th=6.1e-3,
        f_rms_th=3.5e-3,
        print_each=False,
        write_traj=False,
    )

    result = integrator_class(
        atoms,
        output=str(tmp_path / f"{integrator_class.__name__.lower()}.out"),
        params=params,
    ).run()

    summary = result["summary"]
    assert summary["converged"] is True
    assert sum(
        record["point_kind"] == "transition_state"
        for record in summary["records"]
    ) == 1
    transition_state = summary["records"][summary["ts_index"] - 1]
    assert transition_state["E"] == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(
        transition_state["x"],
        [[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]],
        atol=1.0e-14,
        rtol=0.0,
    )

    endpoint_distances = []
    for direction in ("forward", "backward"):
        branch = result[direction]
        assert branch["status"]["termination_reason"] == "force_converged"
        assert branch["status"]["force_criteria_satisfied"] is True
        assert branch["status"]["accepted_macro_steps"] <= 8
        endpoint = branch["records"][-1]
        endpoint_distances.append(float(np.linalg.norm(endpoint["x"][1] - endpoint["x"][0])))
        assert endpoint["E"] < -1.2e-2

    assert sorted(endpoint_distances) == pytest.approx([1.0, 2.0], abs=2.0e-2)
