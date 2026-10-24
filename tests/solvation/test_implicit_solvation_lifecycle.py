"""Frozen GB/PB identity: conformers may move, Hamiltonians may not drift."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.extra_correction.implicit import (
    correction as correction_module,
)
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB
from maple.function.calculator.extra_correction.implicit.result import SolvationResult
from maple.function.read.filereader.mol2_reader import MOL2Reader


class HarmonicBase(CalcABC):
    implemented_properties = ("energy", "forces", "free_energy")
    MODEL_ENERGY_UNIT = "hartree"

    def __init__(self, correction):
        super().__init__()
        self.solvent_correction = correction
        self.calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=None):
        atoms = super().calculate(atoms, properties)
        self.calls += 1
        self._finalize_results(
            atoms,
            energy=0.5 * np.sum(atoms.positions**2),
            forces=-atoms.positions,
        )


class StubPB:
    """No external APBS work; identity is owned by the real correction layer."""

    supported_properties = frozenset({"energy"})

    def __init__(self, atoms, charges, **kwargs):
        self.charges = charges.copy()

    def evaluate(self, atoms, **kwargs):
        return SolvationResult(energy_hartree=-0.01)


@pytest.fixture(params=["gb", "pb"])
def prepared(request, water_mol2, tmp_path, monkeypatch):
    if request.param == "pb":
        monkeypatch.setattr(
            "maple.function.calculator.extra_correction.implicit.apbs_pb.APBSLPB",
            StubPB,
        )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed"},
        {"method": request.param, "experimental": True},
        output=tmp_path / "job.out",
    )
    return atoms, correction


def _drift(atoms, kind):
    metadata = atoms.info["mol2"]
    if kind == "atom-label":
        metadata["atom_names"][0] = "different-oxygen"
    elif kind == "atom-type":
        metadata["atom_types"][0] = "O.2"
    elif kind == "bond-order":
        metadata["bonds"][0] = [0, 1, 2.0]
    elif kind == "bond-graph":
        metadata["bonds"] = [[0, 1, 1.0], [1, 2, 1.0]]
    elif kind == "element-order":
        atoms.set_atomic_numbers([1, 8, 1])
    elif kind == "charge":
        atoms.info["charge"] = 1
    elif kind == "multiplicity":
        atoms.info["mult"] = 3
    elif kind == "missing-state":
        del atoms.info["mult"]
    elif kind == "partial-charges":
        atoms.set_initial_charges([-0.4, 0.2, 0.2])
    elif kind == "charge-declaration":
        metadata["charges_present"] = False
    elif kind == "pbc":
        atoms.pbc = True
    else:
        raise AssertionError(kind)


DRIFTS = (
    "atom-label",
    "atom-type",
    "bond-order",
    "bond-graph",
    "element-order",
    "charge",
    "multiplicity",
    "missing-state",
    "partial-charges",
    "charge-declaration",
    "pbc",
)


@pytest.mark.parametrize("kind", DRIFTS)
def test_direct_correction_rejects_frozen_identity_drift(prepared, kind):
    atoms, correction = prepared
    assert np.isfinite(correction.evaluate(atoms).energy_hartree)
    # The correction must freeze values, not reference this mutable nested info.
    _drift(atoms, kind)
    with pytest.raises(ValueError):
        correction.evaluate(atoms)


@pytest.mark.parametrize("kind", DRIFTS)
def test_same_geometry_in_place_drift_cannot_return_ase_cached_energy(prepared, kind):
    atoms, correction = prepared
    calc = HarmonicBase(correction)
    atoms.calc = calc
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_potential_energy())
    assert calc.calls == 1
    _drift(atoms, kind)
    with pytest.raises(ValueError):
        atoms.get_potential_energy()
    assert calc.results == {}
    assert calc.calls == 1, (
        "Invalid identity must fail before gas/provider recomputation"
    )


@pytest.mark.parametrize("kind", DRIFTS)
def test_stored_atoms_drift_cannot_return_no_atoms_ase_cache(prepared, kind):
    atoms, correction = prepared
    calc = HarmonicBase(correction)
    atoms.calc = calc
    expected = atoms.get_potential_energy()
    assert calc.get_potential_energy() == expected
    _drift(calc.atoms, kind)
    with pytest.raises(ValueError):
        calc.get_potential_energy()
    assert calc.results == {}
    assert calc.calls == 1


@pytest.mark.parametrize("stored", [False, True], ids=["provided", "stored"])
def test_periodic_capable_base_cannot_bypass_nonperiodic_solvent(prepared, stored):
    class PeriodicBase(HarmonicBase):
        SUPPORTS_PBC = True

    atoms, correction = prepared
    calc = PeriodicBase(correction)
    atoms.calc = calc
    atoms.get_potential_energy()
    target = calc.atoms if stored else atoms
    target.pbc = True
    with pytest.raises(ValueError, match="non-periodic"):
        calc.get_potential_energy() if stored else atoms.get_potential_energy()
    assert calc.results == {}
    assert calc.calls == 1


def test_both_cache_entrypoints_validate_once_without_provider_or_charge_work(
    prepared,
    monkeypatch,
):
    atoms, correction = prepared
    calc = HarmonicBase(correction)
    atoms.calc = calc
    expected = atoms.get_potential_energy()
    original_validate = correction.validate_atoms
    validations = []

    def validate(target):
        validations.append(target)
        original_validate(target)

    monkeypatch.setattr(correction, "validate_atoms", validate)
    monkeypatch.setattr(
        correction.provider,
        "evaluate",
        lambda *a, **kw: pytest.fail("Provider rerun on cache hit"),
    )
    monkeypatch.setattr(
        correction_module,
        "prepare_charges",
        lambda *a: pytest.fail("Charges regenerated"),
    )
    assert atoms.get_potential_energy() == expected
    assert calc.get_potential_energy() == expected
    assert calc.get_property("energy", allow_calculation=False) == expected
    assert len(validations) == 3
    assert calc.calls == 1


def test_charge_distinct_valid_mol2_cannot_reuse_reference_hamiltonian(
    water_mol2, tmp_path
):
    a = MOL2Reader(str(water_mol2), charge=0, mult=1)
    b_path = tmp_path / "different-charges.mol2"
    b_path.write_text(
        water_mol2.read_text()
        .replace("-0.834000", "-0.400000")
        .replace("0.417000", "0.200000")
    )
    b = MOL2Reader(str(b_path), charge=0, mult=1)
    opts = {"method": "gb", "experimental": True}
    ca = ImplicitSolvationCorrection(
        a, {"source": "mol2"}, opts, output=tmp_path / "a.out"
    )
    cb = ImplicitSolvationCorrection(
        b, {"source": "mol2"}, opts, output=tmp_path / "b.out"
    )
    assert ca.evaluate(a).energy_hartree != cb.evaluate(b).energy_hartree
    with pytest.raises(ValueError):
        ca.evaluate(b)


def test_coordinate_conformers_and_identical_input_at_other_path_are_valid(
    prepared,
    water_mol2,
    tmp_path,
    monkeypatch,
):
    atoms, correction = prepared
    original = correction.evaluate(atoms).energy_hartree
    monkeypatch.setattr(
        correction_module,
        "prepare_charges",
        lambda *a: pytest.fail("Charges regenerated"),
    )
    other_path = tmp_path / "same-identity.mol2"
    other_path.write_bytes(water_mol2.read_bytes())
    conformer = MOL2Reader(str(other_path), charge=0, mult=1)
    assert correction.evaluate(conformer).energy_hartree == original
    conformer.positions[1, 0] += 0.1
    assert np.isfinite(correction.evaluate(conformer).energy_hartree)
    # Bond list order/orientation is serialization, not a different graph.
    conformer.info["mol2"]["bonds"] = [
        [j, i, order] for i, j, order in reversed(conformer.info["mol2"]["bonds"])
    ]
    assert np.isfinite(correction.evaluate(conformer).energy_hartree)
    atoms.calc = HarmonicBase(correction)
    atoms.get_potential_energy()
    conformer.calc = atoms.calc
    conformer.get_potential_energy()
    assert atoms.calc.calls == 2


def test_maple_fixed_generated_charges_remain_frozen_and_ignore_raw_input_charges(
    water_mol2,
    tmp_path,
    monkeypatch,
):
    raw = MOL2Reader(str(water_mol2), charge=0, mult=1)
    original_raw_charges = raw.get_initial_charges().copy()
    atoms = raw.copy()
    atoms.info = deepcopy(raw.info)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "maple", "method": "qeq-gto", "mode": "fixed"},
        {"method": "gb", "experimental": True},
        output=tmp_path / "generated.out",
    )
    generated = correction.charge_result.charges.copy()
    assert not np.array_equal(original_raw_charges, generated)
    monkeypatch.setattr(
        correction_module,
        "prepare_charges",
        lambda *a: pytest.fail("Charges regenerated"),
    )
    raw.positions[1, 0] += 0.1
    raw.set_initial_charges([-0.4, 0.2, 0.2])
    raw.info["mol2"]["path"] = "/same-identity/different-input.mol2"
    result = correction.evaluate(raw, need_forces=True)
    direct = OpenMMGB(raw, generated).evaluate(raw, need_forces=True)
    assert result.energy_hartree == direct.energy_hartree
    np.testing.assert_array_equal(
        result.forces_hartree_per_angstrom, direct.forces_hartree_per_angstrom
    )
    np.testing.assert_array_equal(correction.charge_result.charges, generated)
    np.testing.assert_array_equal(raw.get_initial_charges(), [-0.4, 0.2, 0.2])
    raw.calc = HarmonicBase(correction)
    raw.get_potential_energy()
    raw.get_potential_energy()
    assert raw.calc.calls == 1
    _drift(raw, "atom-label")
    with pytest.raises(ValueError):
        raw.get_potential_energy()
    assert raw.calc.results == {}


def test_smd_retains_its_separate_validation_contract(
    water_mol2, tmp_path, monkeypatch
):
    class StubSMD:
        supported_properties = frozenset({"energy"})

        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, atoms, **kwargs):
            return "separate-smd-provider-contract"

    monkeypatch.setattr(correction_module, "SMDImplicitSolvation", StubSMD)
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    correction = ImplicitSolvationCorrection(
        atoms,
        {},
        {"method": "smd", "experimental": True},
        output=tmp_path / "smd.out",
    )
    del atoms.info["mol2"]
    assert correction.evaluate(atoms) == "separate-smd-provider-contract"
