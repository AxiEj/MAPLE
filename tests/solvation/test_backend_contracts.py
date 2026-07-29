from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.backends import (
    AlchemicalBackend,
    AlchemicalResult,
    CompositeCalculator,
    PotentialBackend,
    PotentialResult,
    SolvationBackend,
)
from maple.function.calculator.extra_correction.implicit.result import SolvationResult
from maple.function.calculator.model_capabilities import (
    ModelCapabilities,
    SolvationCapabilities,
)


class HarmonicPotential(PotentialBackend):
    capabilities = ModelCapabilities(
        energy=True,
        forces=True,
        conservative_forces=True,
        hessian="finite_difference",
        supports_md=True,
        energy_reference="absolute",
    )

    def evaluate(self, atoms, *, need_forces=False, need_hessian=False):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        return PotentialResult(
            energy_hartree=float(0.5 * np.sum(positions**2)),
            forces_hartree_per_angstrom=-positions if need_forces else None,
            provenance={"provider": "harmonic"},
        )


class HarmonicSolvent(SolvationBackend):
    capabilities = SolvationCapabilities(
        energy=True,
        forces=True,
        conservative_forces=True,
        energy_reference="relative",
    )

    def evaluate(self, atoms, *, need_forces=False, calculator=None):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        return SolvationResult(
            energy_hartree=float(0.25 * np.sum(positions**2)),
            forces_hartree_per_angstrom=-0.5 * positions if need_forces else None,
            components_hartree={"pmf": float(0.25 * np.sum(positions**2))},
            provenance={"provider": "harmonic-solvent"},
        )


class LambdaPotential(AlchemicalBackend):
    capabilities = ModelCapabilities(
        energy=True,
        solvation_mode="alchemical",
        supports_alchemical_lambda=True,
    )

    def evaluate_alchemical(self, atoms, lambda_state):
        value = float(lambda_state["lambda_global"])
        return AlchemicalResult(
            energy_hartree=value,
            dU_dlambda_hartree={"lambda_global": 1.0},
        )


def test_composite_calculator_combines_energy_force_and_numerical_hessian():
    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    calc = CompositeCalculator(HarmonicPotential(), HarmonicSolvent())

    assert calc.get_potential_energy(atoms) == pytest.approx(0.75)
    np.testing.assert_allclose(calc.get_forces(atoms), [[-1.5, 0.0, 0.0]])
    calc.calculate(atoms, properties=("hessian",))
    assert calc.results["hessian"] == pytest.approx(np.eye(3) * 1.5, abs=1e-8)
    assert (
        calc.results["solvation"]["frequency_type"]
        == "effective_solution_pmf"
    )


def test_native_potential_cannot_be_double_solvated():
    class Native(HarmonicPotential):
        capabilities = ModelCapabilities(
            energy=True,
            forces=True,
            conservative_forces=True,
            hessian="finite_difference",
            solvation_mode="native",
        )

    calc = CompositeCalculator(Native(), HarmonicSolvent())
    with pytest.raises(ValueError, match="Native solution-phase"):
        calc.get_potential_energy(Atoms("H", positions=[[0.0, 0.0, 0.0]]))


def test_alchemical_contract_exposes_lambda_derivative():
    result = LambdaPotential().evaluate_alchemical(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        {"lambda_global": 0.5},
    )
    assert result.energy_hartree == 0.5
    assert result.dU_dlambda_hartree["lambda_global"] == 1.0
