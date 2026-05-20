"""Extended-XYZ trajectory frame writers for MD.

Leaf module (numpy/ASE only); re-exported by ``...md.utils``.
"""

from typing import Optional

import numpy as np
from ase import Atoms


def write_xyz_frame(
    file_handle,
    atoms: Atoms,
    energy: float,
    frame_number: int,
    velocity: Optional[np.ndarray] = None,
    include_velocities: bool = False,
    rng_state: Optional[str] = None,
    velocity_representation: Optional[str] = None,
    coordinate_mode: str = "wrapped",
):
    """
    Write a single frame to XYZ file.

    Format:
        N_atoms
        Frame=<number> maple_energy_hartree=<energy> CoordinateMode=<mode>
        Lattice="..." Properties=species:S:1:pos:R:3[...]
        Symbol  x  y  z  [vx  vy  vz]

    Parameters
    ----------
    file_handle : file object
        Opened file handle
    atoms : ase.Atoms
        Atomic system
    energy : float
        Total energy in Hartree
    frame_number : int
        Frame index
    velocity : np.ndarray, optional
        Velocities available for optional debug output (in atomic units)
        Shape: (N_atoms, 3)
    include_velocities : bool, default=False
        Whether to include velocity columns in the XYZ atom lines.
    rng_state : str, optional
        Reserved for API compatibility. Strict restart state is written only to
        RST checkpoints, never to XYZ comment lines.
    velocity_representation : str, optional
        Reserved for API compatibility. Velocity representation metadata is
        stored only in RST checkpoints, never in XYZ comment lines.
    coordinate_mode : str, default="wrapped"
        Human-readable coordinate semantics for the XYZ comment line.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Header lines
    file_handle.write(f"{len(symbols)}\n")
    comment_fields = [
        f"Frame={frame_number}",
        f"maple_energy_hartree={energy:.10f}",
        f"CoordinateMode={coordinate_mode}",
    ]
    properties = "species:S:1:pos:R:3"
    if include_velocities and velocity is not None:
        properties += ":vel:R:3"
    if any(atoms.pbc):
        lattice = " ".join(f"{value:.10f}" for value in np.asarray(atoms.cell.array).reshape(-1))
        pbc_tokens = ["T" if periodic else "F" for periodic in atoms.pbc]
        comment_fields.append(f'Lattice="{lattice}"')
        comment_fields.append(f'pbc="{" ".join(pbc_tokens)}"')
    comment_fields.append(f"Properties={properties}")
    # frame_number stores the MD step number (not sequential frame index) so that
    # resume_simulation() can recover the exact step offset without knowing traj_every.
    file_handle.write(" ".join(comment_fields) + "\n")
    # NOTE: frame_number is the MD *step* number (passed as `step` from the ensemble loop).
    # The regex _TRAJ_COMMENT_RE parses this as frame_num; resume_simulation uses it
    # directly as step_offset (no multiplication by traj_every needed).

    # Atomic coordinates (and optionally velocities)
    if include_velocities and velocity is not None:
        for symbol, (x, y, z), (vx, vy, vz) in zip(symbols, positions, velocity):
            file_handle.write(
                f"{symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}  "
                f"{vx:12.6f} {vy:12.6f} {vz:12.6f}\n"
            )
    else:
        for symbol, (x, y, z) in zip(symbols, positions):
            file_handle.write(f"{symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def write_xyz_trajectory(
    filename: str,
    atoms_list: list,
    energies: list,
    append: bool = False
):
    """
    Write multiple frames to XYZ file.

    Parameters
    ----------
    filename : str
        Output file path
    atoms_list : list of ase.Atoms
        List of atomic configurations
    energies : list of float
        Corresponding energies in Hartree
    append : bool, default=False
        Append to existing file or overwrite
    """
    mode = 'a' if append else 'w'

    with open(filename, mode) as f:
        for i, (atoms, energy) in enumerate(zip(atoms_list, energies)):
            write_xyz_frame(f, atoms, energy, frame_number=i)
