from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from maple.function.read.filereader.mol2_reader import MOL2Reader


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openmm") is None,
    reason="OpenMM optional dependency is not installed",
)


@pytest.mark.parametrize("model", ["hct", "obc1", "obc2", "gbn", "gbn2"])
def test_all_five_amber_gb_models_return_finite_energy_and_force(water_mol2, model):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(atoms, atoms.get_initial_charges(), model=model, nonpolar="ace")
    result = provider.evaluate(atoms, need_forces=True)

    assert np.isfinite(result.energy_hartree)
    assert result.forces_hartree_per_angstrom.shape == (3, 3)
    assert np.isfinite(result.forces_hartree_per_angstrom).all()
    assert np.isclose(
        result.components_hartree["polar"] + result.components_hartree["nonpolar"],
        result.energy_hartree,
        atol=1e-10,
    )


def test_openmm_gb_force_matches_finite_difference(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(atoms, atoms.get_initial_charges(), model="obc2", nonpolar="ace")
    result = provider.evaluate(atoms, need_forces=True)
    h = 1.0e-4
    plus = atoms.copy(); plus.positions[1, 0] += h
    minus = atoms.copy(); minus.positions[1, 0] -= h
    fd_force = -(provider.evaluate(plus).energy_hartree - provider.evaluate(minus).energy_hartree) / (2*h)
    assert np.isclose(result.forces_hartree_per_angstrom[1, 0], fd_force, atol=2e-6)


def test_mol2_name_cannot_trigger_biomolecular_residue_typing(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import (
        build_openmm_topology,
    )

    text = water_mol2.read_text().replace("WATER\n", "DA\n", 1)
    water_mol2.write_text(text)
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = build_openmm_topology(atoms)
    assert [residue.name for residue in topology.residues()] == ["MOL"]


def test_openmm_platform_selection_is_case_insensitive(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar="ace",
        platform="reference",
    )
    assert np.isfinite(provider.evaluate(atoms).energy_hartree)


def test_lcpo_has_an_explicit_openmm_version_boundary(methanol_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(methanol_mol2), charge=0, mult=1)
    major_minor = tuple(
        int(part) for part in importlib.metadata.version("openmm").split(".")[:2]
    )
    if major_minor < (8, 5):
        with pytest.raises(ImportError, match="OpenMM>=8.5"):
            OpenMMGB(atoms, atoms.get_initial_charges(), nonpolar="lcpo")
    else:
        result = OpenMMGB(
            atoms, atoms.get_initial_charges(), nonpolar="lcpo"
        ).evaluate(atoms, need_forces=True)
        assert np.isfinite(result.energy_hartree)
        assert np.isfinite(result.forces_hartree_per_angstrom).all()
        assert result.components_hartree["nonpolar"] > 0.0


def test_lcpo_complete_force_matches_finite_difference(methanol_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    major_minor = tuple(
        int(part) for part in importlib.metadata.version("openmm").split(".")[:2]
    )
    if major_minor < (8, 5):
        pytest.skip("LCPO requires OpenMM>=8.5")
    atoms = MOL2Reader(str(methanol_mol2), charge=0, mult=1)
    provider = OpenMMGB(atoms, atoms.get_initial_charges(), model="obc2", nonpolar="lcpo")
    result = provider.evaluate(atoms, need_forces=True)
    h = 1.0e-4
    plus = atoms.copy(); plus.positions[1, 0] += h
    minus = atoms.copy(); minus.positions[1, 0] -= h
    fd_force = -(
        provider.evaluate(plus).energy_hartree - provider.evaluate(minus).energy_hartree
    ) / (2 * h)
    assert np.isclose(result.forces_hartree_per_angstrom[1, 0], fd_force, atol=2e-6)


def test_openmm_gb_rejects_invalid_dynamic_charge_array(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(atoms, atoms.get_initial_charges())

    with pytest.raises(ValueError, match="one finite partial charge per atom"):
        provider.evaluate(atoms, charges=np.array([0.0, 0.0]))


def test_gbn2_phosphorus_fails_closed_instead_of_using_openmm_default_parameters():
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    path = Path(__file__).parent / "data/gbn2_phosphorus_negative.mol2"
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    charges = atoms.get_initial_charges()
    with pytest.raises(NotImplementedError, match="phosphorus-specific"):
        OpenMMGB(atoms, charges, model="gbn2", nonpolar="ace")
