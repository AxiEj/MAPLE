# -*- coding: utf-8 -*-
"""Shared extxyz trajectory writer.

Plain XYZ stores only atom symbols and positions, so it drops the periodic
cell and pbc flags on write. Extended XYZ preserves ``Lattice`` and ``pbc``
headers, and also round-trips the frame index, energies, and any other
``info`` dictionary keys through ASE's extxyz reader. Energies are restored
via ASE's SinglePointCalculator on re-read.
"""
from typing import List, Optional

from ase import Atoms
from ase.io import write as ase_write


def write_xyz(filename: str, atoms_list: List[Atoms],
              energies: Optional[List[float]] = None,
              mode: str = "w",
              start_index: Optional[int] = None) -> None:
    """Write structures to *filename* in extxyz format."""
    if mode not in {"w", "a"}:
        raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")

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
            frame.info["energy"] = float(energies[i])
        frames.append(frame)

    ase_write(filename, frames, format="extxyz", append=(mode == "a"))
