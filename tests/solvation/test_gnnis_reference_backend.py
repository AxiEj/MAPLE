from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit import TopologyProvider
from maple.function.calculator.extra_correction.implicit import gnnis as gnnis_module
from maple.function.calculator.extra_correction.implicit.gnnis import (
    GNNISReferenceBackend,
    GNNISReferenceResult,
    _canonical_topology_to_rdkit,
    SUPPORTED_GNNIS_SOLVENTS,
    build_gnnis_reference_adapter,
    resolve_gnnis_solvent,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def evaluate(self, positions, *, need_forces):
        self.calls.append(np.asarray(positions))
        return GNNISReferenceResult(
            energy_hartree=-0.420,
            forces_hartree_per_angstrom=(
                np.ones((3, 3), dtype=float) * 0.010 if need_forces else None
            ),
        )


def _checkpoint(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "GNN.pt"
    path.write_bytes(b"dummy checkpoint")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_runtime_factory():
    runtime = FakeRuntime()

    def factory(**kwargs):
        assert kwargs["model_path"].endswith("GNN.pt")
        assert kwargs["solvent"].upstream_key == "tip3p"
        assert kwargs["forcefield"] == "openff-2.0.0"
        return runtime

    return factory, runtime


def test_gnnis_reference_adapter_requires_explicit_model_and_topology(
    water_mol2, tmp_path, fake_runtime_factory, monkeypatch
):
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.gnnis._sha256",
        lambda _path: "304a6cb2e1f804d30dcc4e1b135be1aa768074b4cb268ba26ee34acaa513e5f6",
    )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms)
    fake_factory, _ = fake_runtime_factory

    with pytest.raises(ValueError, match="model_path"):
        GNNISReferenceBackend(
            atoms,
            atoms.get_initial_charges(),
            topology,
            model_path="",
            solvent="water",
        )

    model_path, _checksum = _checkpoint(tmp_path)
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    with pytest.raises(ValueError, match="explicit solvent"):
        GNNISReferenceBackend(
            atoms,
            atoms.get_initial_charges(),
            topology,
            model_path=str(model_path),
            runtime_factory=fake_factory,
        )

    with pytest.raises(ValueError, match="topology"):
        GNNISReferenceBackend(
            atoms,
            atoms.get_initial_charges(),
            None,
            model_path=str(model_path),
            solvent="water",
            checkpoint_sha256=gnnis_module.GNNIS_CHECKPOINT_SHA256,
            runtime_factory=fake_factory,
        )

    missing_metadata = atoms.copy()
    del missing_metadata.info["mult"]
    with pytest.raises(ValueError, match=r"explicit atoms\.info"):
        GNNISReferenceBackend(
            missing_metadata,
            missing_metadata.get_initial_charges(),
            topology,
            model_path=str(model_path),
            solvent="water",
            runtime_factory=fake_factory,
        )

    with pytest.raises(ValueError, match="one partial charge per atom"):
        GNNISReferenceBackend(
            atoms,
            np.array([0.0, 0.0]),
            topology,
            model_path=str(model_path),
            solvent="water",
            checkpoint_sha256=gnnis_module.GNNIS_CHECKPOINT_SHA256,
            runtime_factory=fake_factory,
        )


def test_gnnis_checkpoint_is_sha256_pinned(
    water_mol2, tmp_path, fake_runtime_factory, monkeypatch
):
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.gnnis._sha256",
        lambda _path: "304a6cb2e1f804d30dcc4e1b135be1aa768074b4cb268ba26ee34acaa513e5f6",
    )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms)
    model_path, _ = _checkpoint(tmp_path)
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    fake_factory, _ = fake_runtime_factory

    with pytest.raises(ValueError, match="pinned to the official checkpoint SHA256"):
        GNNISReferenceBackend(
            atoms,
            atoms.get_initial_charges(),
            topology,
            model_path=str(model_path),
            solvent="water",
            checkpoint_sha256="0" * 64,
            runtime_factory=fake_factory,
        )


def test_gnnis_reference_adapter_evaluates_full_reference_hamiltonian(
    water_mol2, tmp_path, fake_runtime_factory, monkeypatch
):
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.gnnis._sha256",
        lambda _path: "304a6cb2e1f804d30dcc4e1b135be1aa768074b4cb268ba26ee34acaa513e5f6",
    )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms)
    model_path, _checksum = _checkpoint(tmp_path)
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    fake_factory, runtime = fake_runtime_factory

    backend = GNNISReferenceBackend(
        atoms,
        atoms.get_initial_charges(),
        topology,
        model_path=str(model_path),
        solvent="water",
        checkpoint_sha256=gnnis_module.GNNIS_CHECKPOINT_SHA256,
        runtime_factory=fake_factory,
    )
    result = backend.evaluate(atoms, need_forces=True)

    assert result.energy_hartree == pytest.approx(-0.420)
    assert result.forces_hartree_per_angstrom.shape == (3, 3)
    assert result.provenance["reference_hamiltonian"] == "openff-2.0.0+GNNIS"
    assert result.provenance["solvent"] == "tip3p"
    assert runtime.calls[0].shape == (3, 3)


def test_gnnis_reference_adapter_rejects_coordinate_topology_drift_and_pbc(
    water_mol2, tmp_path, fake_runtime_factory, monkeypatch
):
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.gnnis._sha256",
        lambda _path: "304a6cb2e1f804d30dcc4e1b135be1aa768074b4cb268ba26ee34acaa513e5f6",
    )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms)
    model_path, _checksum = _checkpoint(tmp_path)
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    fake_factory, _ = fake_runtime_factory
    backend = GNNISReferenceBackend(
        atoms,
        atoms.get_initial_charges(),
        topology,
        model_path=str(model_path),
        solvent="water",
        checkpoint_sha256=gnnis_module.GNNIS_CHECKPOINT_SHA256,
        runtime_factory=fake_factory,
    )

    reordered = Atoms(
        symbols=list(reversed(atoms.get_chemical_symbols())),
        positions=np.asarray(atoms.get_positions())[::-1],
    )
    with pytest.raises(ValueError, match="element order mismatch"):
        backend.evaluate(reordered)

    periodic = atoms.copy()
    periodic.set_cell([20.0, 20.0, 20.0])
    periodic.set_pbc(True)
    with pytest.raises(NotImplementedError, match="non-periodic"):
        backend.evaluate(periodic)


def test_gnnis_solvent_registry_is_exactly_the_upstream_39():
    assert len(SUPPORTED_GNNIS_SOLVENTS) == 39
    assert resolve_gnnis_solvent("water").upstream_key == "tip3p"
    assert resolve_gnnis_solvent("1,4-dioxane").solvent_id == 15
    assert resolve_gnnis_solvent("toluene").upstream_key == "Toluol"
    with pytest.raises(ValueError, match="Unsupported GNNIS solvent"):
        resolve_gnnis_solvent("not-a-solvent")


def test_gnnis_reference_model_card_declares_key_boundaries():
    card = (
        Path(__file__).parents[2]
        / "maple"
        / "function"
        / "calculator"
        / "model_cards"
        / "gnnis-reference.yaml"
    )
    content = json.loads(card.read_text(encoding="utf-8"))
    assert content["checkpoint_sha256"].startswith("304a6cb2")
    assert content["capabilities"]["supports_absolute_solvation"] is False
    assert content["capabilities"]["supports_charge"] is False
    assert "arbitrary_mlip_gnnis_correction" in content["forbidden_tasks"]


def test_gnnis_reference_rejects_declared_charge_mismatch(
    tmp_path, water_mol2, fake_runtime_factory, monkeypatch
):
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.gnnis._sha256",
        lambda _path: "304a6cb2e1f804d30dcc4e1b135be1aa768074b4cb268ba26ee34acaa513e5f6",
    )
    atoms = MOL2Reader(str(water_mol2), charge=1, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms)
    model_path, _checksum = _checkpoint(tmp_path)
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    fake_factory, _ = fake_runtime_factory
    atoms.set_initial_charges([0.5, 0.0, 0.0])

    with pytest.raises(ValueError, match="charge mismatch"):
        GNNISReferenceBackend(
            atoms,
            atoms.get_initial_charges(),
            topology,
            model_path=str(model_path),
            solvent="water",
            checkpoint_sha256=gnnis_module.GNNIS_CHECKPOINT_SHA256,
            runtime_factory=fake_factory,
        )


def test_gnnis_reference_rejects_nonzero_total_charge_with_matching_partial_sum(
    tmp_path, water_mol2, fake_runtime_factory, monkeypatch
):
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.gnnis._sha256",
        lambda _path: "304a6cb2e1f804d30dcc4e1b135be1aa768074b4cb268ba26ee34acaa513e5f6",
    )
    atoms = MOL2Reader(str(water_mol2), charge=1, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms)
    model_path, _checksum = _checkpoint(tmp_path)
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    fake_factory, _ = fake_runtime_factory
    atoms.set_initial_charges([0.5, 0.2, 0.3])

    with pytest.raises(ValueError, match="neutral-only"):
        GNNISReferenceBackend(
            atoms,
            atoms.get_initial_charges(),
            topology,
            model_path=str(model_path),
            solvent="water",
            checkpoint_sha256=gnnis_module.GNNIS_CHECKPOINT_SHA256,
            runtime_factory=fake_factory,
        )


def test_gnnis_canonical_topology_to_rdkit_preserves_formal_charge_metadata(tmp_path):
    from rdkit import Chem

    ammonium = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
    out = tmp_path / "ammonium.sdf"
    writer = Chem.SDWriter(str(out))
    writer.write(ammonium)
    writer.close()

    topology = TopologyProvider.from_sdf_file(out)
    atoms = Atoms(
        symbols=[atom.GetSymbol() for atom in ammonium.GetAtoms()],
        positions=np.zeros((len(topology.symbols), 3), dtype=float),
    )
    rdmol = _canonical_topology_to_rdkit(atoms, topology)

    formal = [
        rdmol.GetAtomWithIdx(i).GetFormalCharge() for i in range(rdmol.GetNumAtoms())
    ]
    assert formal == [1, 0, 0, 0, 0]


def test_gnnis_canonical_topology_to_rdkit_preserves_aromaticity():
    topology = TopologyProvider._validate_open_atoms(
        ("C",) * 6,
        tuple(f"C{index + 1}" for index in range(6)),
        ("C.ar",) * 6,
        tuple((index, (index + 1) % 6, 1.5) for index in range(6)),
        (None,) * 6,
        (0,) * 6,
        total_formal_charge=0,
        source="test-benzene",
        require_single_fragment=True,
    )
    atoms = Atoms("C6", positions=np.zeros((6, 3), dtype=float))

    molecule = _canonical_topology_to_rdkit(atoms, topology)

    assert all(atom.GetIsAromatic() for atom in molecule.GetAtoms())
    assert all(bond.GetIsAromatic() for bond in molecule.GetBonds())


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (GNNISReferenceResult(energy_hartree=np.nan), "energy must be finite"),
        (
            GNNISReferenceResult(
                energy_hartree=-0.42,
                forces_hartree_per_angstrom=np.full((3, 3), np.inf),
            ),
            "forces must be finite",
        ),
    ],
)
def test_gnnis_reference_rejects_nonfinite_runtime_results(
    result,
    message,
    tmp_path,
    water_mol2,
    monkeypatch,
):
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms)
    model_path, _ = _checkpoint(tmp_path)

    class NonfiniteRuntime:
        def evaluate(self, _positions, *, need_forces):
            del need_forces
            return result

    backend = GNNISReferenceBackend(
        atoms,
        atoms.get_initial_charges(),
        topology,
        model_path=str(model_path),
        solvent="water",
        runtime_factory=lambda **_kwargs: NonfiniteRuntime(),
    )

    with pytest.raises(ValueError, match=message):
        backend.evaluate(
            atoms, need_forces=result.forces_hartree_per_angstrom is not None
        )


def test_gnnis_reference_revalidates_charge_and_multiplicity_at_runtime(
    tmp_path, water_mol2, fake_runtime_factory, monkeypatch
):
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = TopologyProvider.from_mol2_atoms(atoms)
    model_path, _ = _checkpoint(tmp_path)
    fake_factory, _ = fake_runtime_factory
    backend = GNNISReferenceBackend(
        atoms,
        atoms.get_initial_charges(),
        topology,
        model_path=str(model_path),
        solvent="water",
        runtime_factory=fake_factory,
    )

    changed_multiplicity = atoms.copy()
    changed_multiplicity.info["mult"] = 3
    with pytest.raises(ValueError, match="runtime multiplicity"):
        backend.evaluate(changed_multiplicity)

    changed_charges = atoms.copy()
    changed_charges.set_initial_charges([0.1, 0.0, 0.0])
    with pytest.raises(ValueError, match="runtime partial charges"):
        backend.evaluate(changed_charges)

    missing_metadata = atoms.copy()
    del missing_metadata.info["charge"]
    with pytest.raises(ValueError, match="runtime requires explicit"):
        backend.evaluate(missing_metadata)


def test_gnnis_reference_revalidates_charge_when_topology_charge_is_unknown(
    tmp_path, water_mol2, fake_runtime_factory, monkeypatch
):
    monkeypatch.setattr(
        gnnis_module, "_sha256", lambda _path: gnnis_module.GNNIS_CHECKPOINT_SHA256
    )
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = replace(
        TopologyProvider.from_mol2_atoms(atoms),
        total_formal_charge=None,
    )
    model_path, _ = _checkpoint(tmp_path)
    fake_factory, _ = fake_runtime_factory
    backend = GNNISReferenceBackend(
        atoms,
        atoms.get_initial_charges(),
        topology,
        model_path=str(model_path),
        solvent="water",
        runtime_factory=fake_factory,
    )

    changed_charge = atoms.copy()
    changed_charge.info["charge"] = 1
    with pytest.raises(ValueError, match="runtime total charge"):
        backend.evaluate(changed_charge)


def test_gnnis_reference_adapter_route_is_reference_only():
    with pytest.raises(NotImplementedError, match="Arbitrary MLIP"):
        build_gnnis_reference_adapter(mode="correction")
