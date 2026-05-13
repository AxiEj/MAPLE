# -*- coding: utf-8 -*-
"""Shared extxyz writer used by OPT / SCAN / TS / IRC algorithms.

Plain XYZ stores only `N + comment + Elem x y z`, dropping the periodic
cell and pbc flags. Re-reading such a file then yields atoms.pbc == False,
which breaks every PBC-aware downstream stage (NPT being the obvious one).
Writing extxyz keeps the box, frame index, and energies round-trippable.

Round-trip locations after ``ase.io.read``:
  * cell / pbc        — atoms.cell, atoms.pbc
  * frame index       — atoms.info['image']
  * potential energy  — atoms.calc.results['energy']  (SinglePointCalculator)
                        equivalently atoms.get_potential_energy()

The maple XYZReader (maple/function/read/filereader/xyz_reader.py) already
parses the extxyz ``Lattice="..."`` / ``pbc="..."`` headers, so callers
see no behavioural change beyond no-longer-losing the cell.
"""
from typing import List, Optional

from ase import Atoms
from ase.io import write as ase_write


def write_xyz(filename: str, atoms_list: List[Atoms],
              energies: Optional[List[float]] = None) -> None:
    """Write structures to *filename* in extxyz format.

    Parameters
    ----------
    filename : str
        Destination path; overwritten if it exists.
    atoms_list : list[Atoms]
        Frames to dump. Cell and pbc are serialised automatically.
    energies : list[float] | None
        Optional per-frame potential energies. Written as
        ``energy=...`` on the extxyz comment line; on re-read they are
        attached to a SinglePointCalculator (so retrieve them via
        ``atoms.get_potential_energy()``).
    """
    frames = []
    for i, at in enumerate(atoms_list):
        frame = at.copy()
        frame.info.setdefault('image', i)
        if energies is not None:
            frame.info['energy'] = float(energies[i])
        frames.append(frame)
    ase_write(filename, frames, format='extxyz')
