"""Portable topology contract for the OpenMM-independent OBC-II provider."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ase import Atoms

from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    MOL2_IDENTITY_SHA256_KEY,
    MOL2Reader,
    mol2_identity_sha256,
)


def _atoms_with_mol2(
    symbols: list[str],
    bonds: list[list[int | str]],
    *,
    component_ids: list[int] | None = None,
    atom_names: list[str] | None = None,
) -> Atoms:
    """Build the smallest identity-valid MOL2-like ASE object for unit tests."""
    count = len(symbols)
    components = list(component_ids if component_ids is not None else [0] * count)
    metadata = {
        "path": "synthetic.mol2",
        "source_sha256": "synthetic",
        "name": "SYNTHETIC",
        "molecule_type": "SMALL",
        "charge_type": "USER_CHARGES",
        "atom_ids": list(range(1, count + 1)),
        "atom_names": list(atom_names or symbols),
        "atom_types": list(symbols),
        "subst_ids": [component + 1 for component in components],
        "subst_names": [f"MOL{component + 1}" for component in components],
        "bonds": bonds,
        "charges_present": True,
        "component_ids": components,
        "component_count": len(set(components)),
        "component_charge_sums_e": [0.0] * len(set(components)),
    }
    metadata[MOL2_IDENTITY_SHA256_KEY] = mol2_identity_sha256(metadata)
    atoms = Atoms(symbols, positions=np.zeros((count, 3), dtype=np.float64))
    atoms.new_array(MOL2_ATOM_ID_ARRAY, np.arange(1, count + 1, dtype=np.int64))
    atoms.info["mol2"] = metadata
    return atoms


def test_snapshot_is_frozen_canonical_and_position_independent(water_mol2):
    from maple.function.calculator.extra_correction.implicit.obc2_topology import (
        OBC2Topology,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    snapshot = OBC2Topology.from_atoms(atoms)

    assert snapshot.atomic_numbers == (8, 1, 1)
    assert snapshot.atom_names == ("O1", "H1", "H2")
    assert snapshot.component_ids == (0, 0, 0)
    assert snapshot.bond_indices == ((0, 1), (0, 2))
    assert snapshot.bonded_indices == ((1, 2), (0,), (0,))
    assert snapshot.component_count == 1
    assert len(snapshot.identity_sha256) == 64

    displaced = atoms.copy()
    displaced.positions += np.array([2.0, -3.0, 1.0])
    assert OBC2Topology.from_atoms(displaced).identity_sha256 == snapshot.identity_sha256

    with pytest.raises((FrozenInstanceError, AttributeError)):
        snapshot.component_count = 2
    with pytest.raises(TypeError):
        snapshot.bonds[0] = (1, 2)


def test_snapshot_hash_binds_atom_order_names_elements_components_and_bonds():
    from maple.function.calculator.extra_correction.implicit.obc2_topology import (
        OBC2Topology,
    )

    baseline = _atoms_with_mol2(["C", "H"], [[0, 1, "1"]], atom_names=["C1", "H1"])
    baseline_hash = OBC2Topology.from_atoms(baseline).identity_sha256

    variants = [
        _atoms_with_mol2(["N", "H"], [[0, 1, "1"]], atom_names=["C1", "H1"]),
        _atoms_with_mol2(["C", "H"], [[0, 1, "1"]], atom_names=["CX", "H1"]),
        _atoms_with_mol2(
            ["C", "H"], [[0, 1, "1"]], component_ids=[0, 0], atom_names=["H1", "C1"]
        ),
    ]
    assert all(
        OBC2Topology.from_atoms(candidate).identity_sha256 != baseline_hash
        for candidate in variants
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda metadata: metadata.pop("atom_names"), "atom name"),
        (lambda metadata: metadata.__setitem__("component_ids", [0]), "component"),
        (lambda metadata: metadata.__setitem__("component_ids", [0, 2]), "dense"),
        (lambda metadata: metadata.__setitem__("bonds", [[0, 2, "1"]]), "range"),
        (lambda metadata: metadata.__setitem__("bonds", [[0, 0, "1"]]), "self"),
        (
            lambda metadata: metadata.__setitem__(
                "bonds", [[0, 1, "1"], [1, 0, "1"]]
            ),
            "duplicate",
        ),
    ],
)
def test_snapshot_rejects_malformed_mol2_metadata(mutate, message):
    from maple.function.calculator.extra_correction.implicit.obc2_topology import (
        OBC2Topology,
    )

    atoms = _atoms_with_mol2(["C", "H"], [[0, 1, "1"]])
    mutate(atoms.info["mol2"])
    atoms.info["mol2"][MOL2_IDENTITY_SHA256_KEY] = mol2_identity_sha256(
        atoms.info["mol2"]
    )

    with pytest.raises(ValueError, match=message):
        OBC2Topology.from_atoms(atoms)


def test_snapshot_rejects_missing_or_stale_mol2_identity():
    from maple.function.calculator.extra_correction.implicit.obc2_topology import (
        OBC2Topology,
    )

    missing = Atoms("CH", positions=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="MOL2"):
        OBC2Topology.from_atoms(missing)

    stale = _atoms_with_mol2(["C", "H"], [[0, 1, "1"]])
    stale.info["mol2"]["atom_names"][0] = "changed"
    with pytest.raises(ValueError, match="identity|changed"):
        OBC2Topology.from_atoms(stale)


@pytest.mark.parametrize(
    ("symbols", "bonds", "components", "message"),
    [
        (["H", "C"], [], [0, 1], "hydrogen|bond"),
        (["C", "H", "N"], [[0, 1, "1"], [1, 2, "1"]], [0, 0, 0], "hydrogen|ambiguous"),
        (["C", "H"], [[0, 1, "1"]], [0, 1], "component"),
    ],
)
def test_snapshot_rejects_nonportable_hydrogen_connectivity(
    symbols, bonds, components, message
):
    from maple.function.calculator.extra_correction.implicit.obc2_topology import (
        OBC2Topology,
    )

    atoms = _atoms_with_mol2(symbols, bonds, component_ids=components)
    with pytest.raises(ValueError, match=message):
        OBC2Topology.from_atoms(atoms)
