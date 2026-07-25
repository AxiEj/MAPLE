import builtins

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.calculator.cluster_continuum import (
    ClusterContinuumCalculator,
    GBPolarOuterProvider,
    TBLiteALPBDeltaProvider,
)
from maple.function.dispatcher.sp.sp import SinglePoint


class HarmonicInner(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, coefficient=1.0):
        super().__init__()
        self.coefficient = coefficient
        self.seen_atom_counts = []

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.seen_atom_counts.append(len(atoms))
        energy = self.coefficient * np.sum(positions**2)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -2.0 * self.coefficient * positions,
        }


class HarmonicOuter:
    implemented_properties = ("energy", "forces")

    def __init__(self, coefficient=0.25):
        self.coefficient = coefficient
        self.seen_atom_counts = []

    def calculate(self, atoms, properties):
        positions = atoms.get_positions()
        self.seen_atom_counts.append(len(atoms))
        result = {"energy": self.coefficient * np.sum(positions**2)}
        if "forces" in properties:
            result["forces"] = -2.0 * self.coefficient * positions
        return result


class EnergyOnlyOuter:
    implemented_properties = ("energy",)

    def calculate(self, atoms, properties):
        return {"energy": 0.5}


def test_composite_adds_inner_and_outer_results_for_entire_cluster():
    atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    inner = HarmonicInner(coefficient=1.0)
    outer = HarmonicOuter(coefficient=0.25)
    calc = ClusterContinuumCalculator(inner, outer)

    calc.calculate(atoms, properties=["energy", "forces"])

    assert calc.results["energy"] == pytest.approx(6.25)
    assert calc.results["inner_energy"] == pytest.approx(5.0)
    assert calc.results["outer_correction"] == pytest.approx(1.25)
    np.testing.assert_allclose(calc.results["forces"], -2.5 * atoms.positions)
    assert inner.seen_atom_counts == [3]
    assert outer.seen_atom_counts == [3]


def test_energy_only_outer_rejects_derivatives():
    calc = ClusterContinuumCalculator(HarmonicInner(), EnergyOnlyOuter())

    with pytest.raises(NotImplementedError, match="energy-only"):
        calc.calculate(Atoms("H", positions=[[0, 0, 0]]), properties=["forces"])


def test_builtin_gb_provider_evaluates_complete_cluster():
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239, 0.927, 0]],
        info={"charge": 0, "mult": 1},
    )
    provider = GBPolarOuterProvider("water", device="cpu")

    result = provider.calculate(atoms, ["energy"])

    assert np.isfinite(result["energy"])
    assert result["energy"] < 0.0


def test_single_point_reports_composed_energy_terms(tmp_path):
    atoms = Atoms("H", positions=[[1, 0, 0]])
    atoms.calc = ClusterContinuumCalculator(
        HarmonicInner(coefficient=1.0),
        HarmonicOuter(coefficient=0.25),
    )
    energy = atoms.get_potential_energy()
    job = SinglePoint(str(tmp_path / "out"), atoms)

    lines = "".join(job._single_energy_lines(energy))

    assert "Inner cluster energy: 1.0000000000 Hartree" in lines
    assert "Outer continuum correction: 0.2500000000 Hartree" in lines
    assert "Energy: 1.2500000000 Hartree" in lines


def test_force_capable_composite_provides_numerical_hessian_and_hvp():
    atoms = Atoms("H", positions=[[0.3, -0.2, 0.1]])
    calc = ClusterContinuumCalculator(
        HarmonicInner(coefficient=1.0),
        HarmonicOuter(coefficient=0.25),
    )

    hessian = calc.get_hessian(atoms, delta=1.0e-4)
    np.testing.assert_allclose(hessian, 2.5 * np.eye(3), atol=1.0e-8)

    direction = np.array([1.0, 0.0, 0.0])
    hvp, forces, energy = calc.get_hvp(atoms, direction, delta=1.0e-4)
    np.testing.assert_allclose(hvp.numpy(), [2.5, 0.0, 0.0], atol=1.0e-8)
    np.testing.assert_allclose(forces.numpy(), (-2.5 * atoms.positions).reshape(-1))
    assert energy.item() == pytest.approx(1.25 * np.sum(atoms.positions**2))


class FakeTBLite(Calculator):
    implemented_properties = ["energy", "forces"]
    init_kwargs = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.init_kwargs.append(kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        solvated = self.kwargs.get("solvation") is not None
        coefficient = 3.0 if solvated else 1.0
        positions = atoms.get_positions()
        self.results = {
            "energy": coefficient * np.sum(positions**2),
            "forces": -2.0 * coefficient * positions,
        }


def test_tblite_provider_returns_alpb_minus_vacuum_in_maple_units():
    FakeTBLite.init_kwargs.clear()
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], info={"charge": 0, "mult": 1})
    provider = TBLiteALPBDeltaProvider("water", calculator_cls=FakeTBLite)

    result = provider.calculate(atoms, ["energy", "forces"])

    assert result["energy"] == pytest.approx(2.0 * EV2HARTREE)
    np.testing.assert_allclose(
        result["forces"],
        [[0.0, 0.0, 0.0], [-4.0 * EV2HARTREE, 0.0, 0.0]],
    )
    assert {entry.get("solvation") for entry in FakeTBLite.init_kwargs} == {
        None,
        ("alpb", "water", "gsolv"),
    }


def test_tblite_provider_has_clear_optional_dependency_error(monkeypatch):
    real_import = builtins.__import__

    def fail_tblite(name, *args, **kwargs):
        if name == "tblite.ase":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_tblite)

    with pytest.raises(ImportError, match=r"conda install.*tblite"):
        TBLiteALPBDeltaProvider("water")
