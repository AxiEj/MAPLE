from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes

from maple.function.calculator.calculator_base import (
    CalcABC,
    EV2HARTREE,
    HARTREE2EV,
)
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView


class _PublicEVCalculator(CalcABC):
    implemented_properties = ("energy", "free_energy", "forces")
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTS_PBC = False

    def __init__(self, *, solvent_correction=None):
        super().__init__()
        self.solvent_correction = solvent_correction
        self.calculate_calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        atoms = super().calculate(atoms, properties, system_changes)
        self.calculate_calls += 1
        requested = set(properties or ("energy",))
        forces = np.array([[1.0, -2.0, 3.0]]) if "forces" in requested else None
        self._finalize_results(atoms, energy=-2.5, forces=forces)

    @staticmethod
    def get_hessian(_atoms):
        return np.diag([2.0, 3.0, 4.0])

    @staticmethod
    def get_hvp(_atoms, direction):
        direction = np.asarray(direction, dtype=float)
        return 2.0 * direction, np.array([1.0, -2.0, 3.0]), -2.5


class _StructuredCorrection:
    supported_properties = {"energy", "forces"}

    @staticmethod
    def evaluate(_atoms, *, need_forces=False, calculator=None):
        del calculator
        return SimpleNamespace(
            energy_hartree=-0.25,
            forces_hartree_per_angstrom=(
                np.array([[0.1, 0.2, 0.3]]) if need_forces else None
            ),
            components_hartree={"delta_g_solv": -0.25},
            leaf_components_hartree={"pcm": -0.25},
            derived_totals_hartree={"delta_g_solv": -0.25},
            provenance={"provider": "fake", "standard_state": "test"},
        )


class _LegacyGetEnergyCorrection:
    supported_properties = {"energy"}

    @staticmethod
    def get_energy(_atoms):
        return np.asarray(0.125), None


def test_calcabc_public_results_are_ase_units_and_disabled_route2_is_identity():
    atoms = Atoms("H")
    calc = _PublicEVCalculator()
    atoms.calc = calc

    np.testing.assert_array_equal(atoms.get_forces(), [[1.0, -2.0, 3.0]])
    assert atoms.get_potential_energy() == -2.5
    assert calc.results["energy"] == -2.5
    assert calc.results["free_energy"] == -2.5


def test_hartree_native_backend_and_solvation_are_converted_before_public_merge():
    atoms = Atoms("H")
    calc = CalcABC()
    calc.solvent_correction = _StructuredCorrection()
    calc._finalize_results(
        atoms,
        energy=-1.0,
        forces=np.array([[0.5, -0.5, 0.0]]),
        unit="hartree",
    )

    assert calc.results["energy"] == pytest.approx(-1.25 * HARTREE2EV)
    np.testing.assert_allclose(
        calc.results["forces"],
        np.array([[0.6, -0.3, 0.3]]) * HARTREE2EV,
    )
    solvation = calc.results["solvation"]
    assert solvation["energy_hartree"] == -0.25
    assert solvation["energy_role"] == "solvation_correction"
    assert solvation["solvation_correction_hartree"] == -0.25
    assert solvation["delta_g_solv_hartree"] == -0.25
    assert solvation["gas_energy_hartree"] == pytest.approx(-1.0)
    assert solvation["combined_energy_hartree"] == pytest.approx(-1.25)
    assert solvation["leaf_components_hartree"] == {"pcm": -0.25}


def test_ev_native_backend_adds_hartree_solvation_in_ev_but_keeps_hartree_ledger():
    atoms = Atoms("H")
    calc = _PublicEVCalculator(solvent_correction=_StructuredCorrection())
    atoms.calc = calc

    assert atoms.get_potential_energy() == pytest.approx(
        -2.5 - 0.25 * HARTREE2EV
    )
    solvation = calc.results["solvation"]
    assert solvation["gas_energy_hartree"] == pytest.approx(-2.5 * EV2HARTREE)
    assert solvation["combined_energy_hartree"] == pytest.approx(
        -2.5 * EV2HARTREE - 0.25
    )
    assert solvation["delta_g_solv_hartree"] == -0.25


def test_legacy_get_energy_correction_converts_hartree_and_rejects_forces():
    atoms = Atoms("H")
    calc = _PublicEVCalculator(solvent_correction=_LegacyGetEnergyCorrection())
    calc.chargecalc = lambda _atoms, *, total_charge: np.asarray([total_charge])
    atoms.calc = calc

    assert atoms.get_potential_energy() == pytest.approx(
        -2.5 + 0.125 * HARTREE2EV
    )
    with pytest.raises(NotImplementedError):
        atoms.get_forces()


def test_legacy_view_roundtrips_results_hessian_and_hvp_without_mutation():
    atoms = Atoms("H")
    raw = _PublicEVCalculator()
    atoms.calc = raw
    raw.calculate(atoms, properties=["energy", "forces"])
    original = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in raw.results.items()
    }
    view = LegacyHartreeJobView(raw)

    assert view.results["energy"] == pytest.approx(-2.5 * EV2HARTREE)
    np.testing.assert_allclose(view.results["forces"], original["forces"] * EV2HARTREE)
    np.testing.assert_allclose(
        view.get_hessian(atoms), np.diag([2.0, 3.0, 4.0]) * EV2HARTREE
    )
    hvp, forces, energy = view.get_hvp(atoms, np.ones(3))
    np.testing.assert_allclose(hvp, np.full(3, 2.0 * EV2HARTREE))
    np.testing.assert_allclose(forces, np.array([1.0, -2.0, 3.0]) * EV2HARTREE)
    assert energy == pytest.approx(-2.5 * EV2HARTREE)
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(raw.results[key], value)
        else:
            assert raw.results[key] == value


def test_base_analytic_hessian_is_converted_once_to_public_units_and_results():
    class _HartreeHessianCalculator(_PublicEVCalculator):
        hessian = "analytic"

        @staticmethod
        def _analytic_hessian(_atoms):
            return np.diag([1.0, 2.0, 3.0])

        get_hessian = CalcABC.get_hessian

    atoms = Atoms("H")
    calc = _HartreeHessianCalculator()
    public = calc.get_hessian(atoms)
    np.testing.assert_allclose(
        public, np.diag([1.0, 2.0, 3.0]) * HARTREE2EV
    )
    calc._finalize_results(atoms, energy=0.0, hessian=public)
    np.testing.assert_allclose(calc.results["hessian"], public)


def test_dispatcher_installs_one_legacy_view_and_restores_public_calculator(tmp_path):
    atoms = Atoms("H")
    raw = _PublicEVCalculator()
    atoms.calc = raw
    output = tmp_path / "sp.out"
    command = SimpleNamespace(params={"level": "medium", "sp": {"verbose": 1}})

    Dispatcher()(command, "sp", atoms, str(output))

    assert atoms.calc is raw
    assert raw.calculate_calls == 1
    assert raw.results["energy"] == -2.5
    np.testing.assert_array_equal(raw.results["forces"], [[1.0, -2.0, 3.0]])
    text = output.read_text(encoding="utf-8")
    assert f"Energy: {-2.5 * EV2HARTREE:.10f} Hartree" in text
    assert f"{-1.0 * EV2HARTREE:.8f}" in text


def test_direct_single_point_uses_same_boundary_without_double_conversion(tmp_path):
    from maple.function.dispatcher.sp.sp import SinglePoint

    atoms = Atoms("H")
    raw = _PublicEVCalculator()
    atoms.calc = raw
    output = tmp_path / "direct-sp.out"

    SinglePoint(str(output), atoms, {"sp": {"verbose": 1}}).run()

    assert atoms.calc is raw
    assert raw.calculate_calls == 1
    assert raw.results["energy"] == -2.5
    text = output.read_text(encoding="utf-8")
    assert f"Energy: {-2.5 * EV2HARTREE:.10f} Hartree" in text


def test_legacy_view_scales_stress_by_energy_dimension_and_rejects_double_wrap():
    raw = SimpleNamespace(results={"stress": np.arange(6.0), "virial": 4.0})
    view = LegacyHartreeJobView(raw)
    np.testing.assert_allclose(view.results["stress"], np.arange(6.0) * EV2HARTREE)
    assert view.results["virial"] == pytest.approx(4.0 * EV2HARTREE)
    with pytest.raises(ValueError, match="double-wrapped"):
        LegacyHartreeJobView(view)
