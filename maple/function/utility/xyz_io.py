# -*- coding: utf-8 -*-
"""Shared extxyz trajectory writer.

Plain XYZ stores only atom symbols and positions, so it drops the periodic
cell and pbc flags on write. Extended XYZ preserves ``Lattice`` and ``pbc``
headers, and also round-trips the frame index, energies, and other ``info``
dictionary keys through ASE's extxyz reader. ASE's reserved ``energy`` key is
restored through a SinglePointCalculator; unit-labelled custom energy keys
remain explicit frame metadata.
"""

from typing import List, Optional

from ase import Atoms
from ase.io import write as ase_write


def write_xyz(
    filename: str,
    atoms_list: List[Atoms],
    energies: Optional[List[float]] = None,
    mode: str = "w",
    start_index: Optional[int] = None,
    *,
    energy_key: str = "energy",
) -> None:
    """Write structures to *filename* in extxyz format."""
    if mode not in {"w", "a"}:
        raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")
    if not isinstance(energy_key, str) or not energy_key:
        raise ValueError("energy_key must be a non-empty string")
    if energies is not None and len(energies) != len(atoms_list):
        raise ValueError("energies must match atoms_list length")

    if not atoms_list:
        with open(filename, mode):
            pass
        return

    frames = []
    for i, at in enumerate(atoms_list):
        frame = at.copy()
        if start_index is None:
            frame.info.setdefault("image", i)
        else:
            frame.info["image"] = start_index + i
        if energies is not None:
            frame.info[energy_key] = float(energies[i])
        frames.append(frame)

    ase_write(filename, frames, format="extxyz", append=(mode == "a"))
