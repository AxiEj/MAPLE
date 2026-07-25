"""Shared utilities for fixed-charge implicit-solvation providers."""

from __future__ import annotations

KJ_PER_MOL_PER_HARTREE = 2625.4996394799
EV_PER_HARTREE = 27.211386245988


def build_openmm_topology(atoms):
    """Build the neutral small-molecule topology used by OpenMM-backed providers."""
    try:
        from openmm import app
    except ImportError as exc:
        raise ImportError(
            "OpenMM topology construction requires the optional dependency. Install with "
            "`pip install 'maple[implicit-gb]'`."
        ) from exc

    metadata = atoms.info.get("mol2")
    if not metadata:
        raise ValueError(
            "OpenMM-backed implicit solvation requires MOL2 topology metadata."
        )
    topology = app.Topology()
    chain = topology.addChain("A")
    # MOL2 molecule names are identifiers, not biomolecular residue types.
    # A synthetic name prevents an arbitrary input name such as "DA" from
    # selecting GBn2's nucleic-acid-specific parameter branch.
    component_ids = list(metadata.get("component_ids") or [0] * len(atoms))
    if len(component_ids) != len(atoms):
        raise ValueError("MOL2 component metadata does not match the atom count.")
    unique_components = sorted(set(component_ids))
    if unique_components != list(range(len(unique_components))):
        raise ValueError("MOL2 component ids must be dense and zero-based.")
    residues = [
        topology.addResidue(
            "MOL" if len(unique_components) == 1 else f"MOL{index + 1}",
            chain,
        )
        for index in unique_components
    ]
    omm_atoms = []
    names = metadata.get("atom_names") or atoms.get_chemical_symbols()
    for name, symbol, component_id in zip(
        names, atoms.get_chemical_symbols(), component_ids
    ):
        omm_atoms.append(
            topology.addAtom(
                str(name),
                app.Element.getBySymbol(symbol),
                residues[int(component_id)],
            )
        )
    for i, j, _bond_type in metadata.get("bonds", []):
        topology.addBond(omm_atoms[int(i)], omm_atoms[int(j)])
    return topology
