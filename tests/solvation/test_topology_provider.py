from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.topology import TopologyProvider
from maple.function.read.filereader.mol2_reader import MOL2Reader


@pytest.mark.parametrize("require_single_fragment", [True, False])
def test_mol2_metadata_provider_validates_counts_and_fragments(water_mol2, require_single_fragment):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms, require_single_fragment=require_single_fragment)

    assert topology.source == "MOL2"
    assert topology.symbols == tuple(atoms.get_chemical_symbols())
    assert topology.atom_names == tuple(atoms.info["mol2"]["atom_names"])
    assert topology.atom_types == tuple(atoms.info["mol2"]["atom_types"])
    assert topology.bonds == ((0, 1, 1.0), (0, 2, 1.0))
    assert topology.total_formal_charge == 0
    assert topology.nfragments == 1
    if not require_single_fragment:
        assert topology.fragments == (0, 0, 0)


def test_mol2_disconnected_topology_is_rejected(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    atoms.info["mol2"]["bonds"] = []

    with pytest.raises(ValueError, match="disconnected fragments"):
        TopologyProvider.from_mol2_atoms(atoms)


def test_openff_like_object_accepts_programmatic_atoms_and_bonds_and_mappings():
    class FakeAtom:
        def __init__(self, atomic_number, name, mapping=None):
            self.atomic_number = atomic_number
            self.atom_name = name
            self.molecule_atom_index = mapping

    class FakeBond:
        def __init__(self, i, j, order):
            self.atom1_index = i
            self.atom2_index = j
            self.bond_order = order

    class FakeMolecule:
        def __init__(self):
            self.atoms = [
                FakeAtom(8, "O", mapping=101),
                FakeAtom(1, "H1", mapping=102),
                FakeAtom(1, "H2", mapping=103),
            ]
            self.bonds = [
                FakeBond(0, 1, 1),
                FakeBond(0, 2, 1),
            ]

    topology = TopologyProvider.from_openff_molecule(FakeMolecule())
    assert topology.symbols == ("O", "H", "H")
    assert topology.mappings == (101, 102, 103)
    assert topology.bonds == ((0, 1, 1.0), (0, 2, 1.0))
    assert topology.nfragments == 1
    assert topology.source == "OpenFF"


def test_openff_like_mapping_validation_is_fail_closed(tmp_path):
    class FakeAtom:
        def __init__(self, atomic_number, name, mapping):
            self.atomic_number = atomic_number
            self.atom_name = name
            self.molecule_atom_index = mapping

    class FakeBond:
        def __init__(self, i, j, order):
            self.atom1_index = i
            self.atom2_index = j
            self.bond_order = order

    class FakeMolecule:
        def __init__(self):
            self.atoms = [FakeAtom(8, "O", mapping=1), FakeAtom(1, "H", mapping=1)]
            self.bonds = [FakeBond(0, 1, 1)]

    with pytest.raises(ValueError, match="Duplicate mapping"):
        TopologyProvider.from_openff_molecule(FakeMolecule())


def test_openmm_topology_provider_and_openmm_topology_export(tmp_path):
    if importlib.util.find_spec("openmm") is None:
        pytest.skip("OpenMM optional dependency is not installed")

    from openmm import app

    top = app.Topology()
    chain = top.addChain("A")
    residue = top.addResidue("MOL", chain)
    o = top.addAtom("O", app.Element.getBySymbol("O"), residue)
    h1 = top.addAtom("H1", app.Element.getBySymbol("H"), residue)
    h2 = top.addAtom("H2", app.Element.getBySymbol("H"), residue)
    top.addBond(o, h1, 1)
    top.addBond(o, h2, 1)

    topo = TopologyProvider.from_openmm_topology(top)
    assert topo.symbols == ("O", "H", "H")
    assert topo.atom_names == ("O", "H1", "H2")
    assert topo.bonds == ((0, 1, 1.0), (0, 2, 1.0))
    assert topo.nfragments == 1
    assert np.array_equal(np.array(topo.fragments), [0, 0, 0])


@pytest.mark.skipif(
    importlib.util.find_spec("rdkit") is None,
    reason="RDKit optional dependency is not installed",
)
def test_sdf_provider_parses_first_mol_and_keeps_atom_order(tmp_path):
    out = tmp_path / "water.sdf"
    from rdkit import Chem

    water = Chem.AddHs(Chem.MolFromSmiles("C"))
    writer = Chem.SDWriter(str(out))
    writer.write(water)
    writer.close()

    topo = TopologyProvider.from_sdf_file(out)
    assert topo.symbols == ("C", "H", "H", "H", "H")
    assert topo.source == "SDF"
    assert topo.nfragments == 1
    assert topo.mappings == (None, None, None, None, None)
    assert topo.bonds == (
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (0, 4, 1.0),
    )


def test_sdf_provider_preserves_formal_charge_metadata(tmp_path):
    out = tmp_path / "ammonium.sdf"
    from rdkit import Chem

    ammonium = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
    writer = Chem.SDWriter(str(out))
    writer.write(ammonium)
    writer.close()

    topo = TopologyProvider.from_sdf_file(out)
    assert topo.formal_charges == (1, 0, 0, 0, 0)
    assert topo.formal_charge_sum == 1
    assert topo.total_formal_charge == 1


def test_topology_rejects_fractional_formal_charge_and_total_charge_drift(
    water_mol2,
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(ValueError, match="finite integer"):
        TopologyProvider._validate_open_atoms(
            ("H",),
            ("H",),
            ("H",),
            (),
            (None,),
            (0.5,),
            source="test",
            require_single_fragment=True,
        )

    topology = TopologyProvider.from_mol2_atoms(atoms)
    changed_charge = atoms.copy()
    changed_charge.info["charge"] = 1
    with pytest.raises(ValueError, match="total formal charge"):
        topology.validate_atoms(changed_charge)
