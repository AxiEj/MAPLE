"""
Restart (RST) checkpoint file I/O for MD simulations.

RST files store the complete simulation state for restart/resume:
    - Atomic positions and velocities
    - Cell parameters (for periodic systems)
    - Simulation metadata (step, time, ensemble, timestep, energy)
    - RNG state (for NVT/NPT deterministic continuation)

File format
-----------
::

    MAPLE_RST_V1
    natoms = N
    step = S
    time = T  (fs)
    ensemble = nve|nvt|npt
    timestep = dt  (fs)
    energy = E  (Hartree)
    [rng_state = hex_string]  (NVT/NPT only)
    [cell = a b c alpha beta gamma]  (PBC only; back-compat)
    [cell_matrix = ax ay az bx by bz cx cy cz]  (PBC only; exact orientation)
    [pbc = T/F T/F T/F]  (PBC only)
    Symbol  x  y  z  vx  vy  vz  [ix iy iz]
    ...
    END_RST

Velocities are stored in atomic units (Bohr/a.u. time).

Compatible with GROMACS checkpoint concept; enables exact continuation
of MD trajectories with identical thermodynamic evolution.
"""

import json
from pathlib import Path

import numpy as np
from ase.cell import Cell

from .utils import ensure_image_flags


RST_HEADER = "MAPLE_RST_V1"



def get_rng_state_hex(rng: np.random.Generator) -> str:
    """
    Serialize RNG state to hex string for checkpoint storage.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    str
        Hex-encoded JSON representation of the RNG state.
        Can be restored via ``restore_rng_from_hex()``.
    """
    state_json = json.dumps(rng.bit_generator.state, sort_keys=True)
    return state_json.encode().hex()


def restore_rng_from_hex(rng: np.random.Generator, hex_str: str) -> None:
    """
    Restore RNG state from hex string.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator instance to restore into.
    hex_str : str
        Hex-encoded RNG state from ``get_rng_state_hex()``.
    """
    state = json.loads(bytes.fromhex(hex_str).decode())
    rng.bit_generator.state = state


def write_rst(
    path,
    atoms,
    velocities,
    step,
    timestep,
    ensemble,
    energy,
    rng_state=None,
    velocity_representation=None,
):
    """
    Write MD restart checkpoint file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    atoms : ase.Atoms
        Atomic system.
    velocities : np.ndarray
        Velocities in atomic units (Bohr/a.u. time), shape (N, 3).
    step : int
        Current MD step number.
    timestep : float
        Timestep in fs.
    ensemble : str
        Ensemble type ('nve', 'nvt', 'npt').
    energy : float
        Total energy in Hartree.
    rng_state : str, optional
        Hex-encoded RNG state (for NVT/NPT deterministic continuation).
    velocity_representation : str, optional
        Label describing the semantics of the stored velocities.

    Raises
    ------
    ValueError
        If velocities shape mismatch with atoms count.
    """
    path = Path(path)
    expected_shape = (len(atoms), 3)
    if np.shape(velocities) != expected_shape:
        raise ValueError(
            "Velocities must have shape "
            f"{expected_shape}, got {np.shape(velocities)}"
        )
    time_fs = step * timestep
    cell_line = ""
    cell_matrix_line = ""
    pbc_line = ""
    if any(atoms.pbc):
        cell = atoms.cell.cellpar()
        cell_line = (
            f"cell = {cell[0]:.10f} {cell[1]:.10f} {cell[2]:.10f} "
            f"{cell[3]:.10f} {cell[4]:.10f} {cell[5]:.10f}\n"
        )
        # Full 3x3 cell matrix at :.17g (exact IEEE-754 double round-trip).
        # cellpar() alone loses the cell *orientation*, which an oriented
        # triclinic restart needs to reproduce scaled coords / image flags /
        # unwrapped positions.  The cellpar line is kept for back-compat.
        matrix = np.asarray(atoms.cell.array, dtype=float).reshape(-1)
        cell_matrix_line = "cell_matrix = " + " ".join(f"{x:.17g}" for x in matrix) + "\n"
        pbc_flags = ["T" if flag else "F" for flag in atoms.pbc]
        pbc_line = f"pbc = {' '.join(pbc_flags)}\n"

    lines = [
        f"{RST_HEADER}\n",
        f"natoms = {len(atoms)}\n",
        f"step = {step}\n",
        # :.17g for the scalar float fields too — timestep especially: restart
        # validation rejects a >1e-12 timestep mismatch, which .10f would trip
        # for a non-round dt. Keeps the whole checkpoint an exact double round-trip.
        f"time = {time_fs:.17g}\n",
        f"ensemble = {ensemble}\n",
        f"timestep = {timestep:.17g}\n",
        f"energy = {energy:.17g}\n",
    ]
    if velocity_representation is not None:
        lines.append(f"velocity_representation = {velocity_representation}\n")
    if rng_state is not None:
        lines.append(f"rng_state = {rng_state}\n")
    if cell_line:
        lines.append(cell_line)
    if cell_matrix_line:
        lines.append(cell_matrix_line)
    if pbc_line:
        lines.append(pbc_line)

    image_flags = ensure_image_flags(atoms) if any(atoms.pbc) else None
    for idx, (symbol, pos, vel) in enumerate(
        zip(atoms.get_chemical_symbols(), atoms.get_positions(), velocities)
    ):
        image_suffix = ""
        if image_flags is not None:
            ix, iy, iz = image_flags[idx]
            image_suffix = f" {int(ix):d} {int(iy):d} {int(iz):d}"
        # :.17g — exact IEEE-754 double round-trip for positions/velocities so
        # the whole checkpoint reproduces the run state on restart (not just the
        # cell).  Image flags are integers and already exact.
        lines.append(
            f"{symbol:<2s} {pos[0]:.17g} {pos[1]:.17g} {pos[2]:.17g}"
            f" {vel[0]:.17g} {vel[1]:.17g} {vel[2]:.17g}{image_suffix}\n"
        )
    lines.append("END_RST\n")
    path.write_text("".join(lines))



def read_rst(path):
    """
    Read MD restart checkpoint file.

    Parameters
    ----------
    path : str or Path
        RST file path.

    Returns
    -------
    dict
        Dictionary containing:
        - ``natoms`` : int — Number of atoms
        - ``step`` : int — MD step number
        - ``time`` : float — Simulation time in fs
        - ``ensemble`` : str — Ensemble type
        - ``timestep`` : float — Timestep in fs
        - ``energy`` : float — Total energy in Hartree
        - ``rng_state`` : str or None — Hex-encoded RNG state
        - ``velocity_representation`` : str — Stored velocity semantics label
        - ``symbols`` : list[str] — Element symbols
        - ``positions`` : np.ndarray — Positions in Angstrom, shape (N, 3)
        - ``velocities`` : np.ndarray — Velocities in a.u., shape (N, 3)
        - ``cell`` : list or None — Cell parameters [a,b,c,alpha,beta,gamma]
        - ``cell_matrix`` : np.ndarray or None — full 3x3 cell matrix in Å
          (exact triclinic orientation; ``None`` for pre-cell_matrix files)
        - ``pbc`` : list or None — Periodic boundary flags

    Raises
    ------
    ValueError
        If file is not a valid RST file or has missing/invalid fields.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != RST_HEADER:
        raise ValueError(f"Not a valid MAPLE RST file: {path}")
    if not lines or lines[-1].strip() != "END_RST":
        raise ValueError(f"Missing END_RST in {path}")

    header = {}
    atom_lines = []
    for line in lines[1:-1]:
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
        elif line.strip():
            atom_lines.append(line)

    required_fields = ["natoms", "step", "time", "ensemble", "timestep", "energy"]
    missing_fields = [field for field in required_fields if field not in header]
    if missing_fields:
        missing_list = ", ".join(missing_fields)
        raise ValueError(f"Missing required RST header fields: {missing_list}")

    natoms = int(header["natoms"])
    if len(atom_lines) != natoms:
        raise ValueError(
            f"Atom count mismatch inside RST file: expected {natoms}, found {len(atom_lines)}"
        )

    symbols = []
    positions = []
    velocities = []
    image_flags = []
    for idx, line in enumerate(atom_lines, 1):
        parts = line.split()
        if len(parts) not in (7, 10):
            raise ValueError(f"Invalid atom line {idx} in {path}: {line!r}")
        symbol = parts[0]
        xyz = [float(x) for x in parts[1:4]]
        vel = [float(x) for x in parts[4:7]]
        symbols.append(symbol)
        positions.append(xyz)
        velocities.append(vel)
        if len(parts) == 10:
            image_flags.append([int(x) for x in parts[7:10]])

    cell = None
    if "cell" in header:
        cell = [float(x) for x in header["cell"].split()]
        if len(cell) != 6:
            raise ValueError(f"Invalid cell line in {path}")

    cell_matrix = None
    if "cell_matrix" in header:
        values = [float(x) for x in header["cell_matrix"].split()]
        if len(values) != 9:
            raise ValueError(f"Invalid cell_matrix line in {path}")
        cell_matrix = np.array(values, dtype=float).reshape(3, 3)

    pbc = None
    if "pbc" in header:
        pbc = [flag == "T" for flag in header["pbc"].split()]
        if len(pbc) != 3:
            raise ValueError(f"Invalid pbc line in {path}")

    parsed_image_flags = None
    if image_flags:
        if len(image_flags) != natoms:
            raise ValueError(f"Incomplete image flags in {path}")
        parsed_image_flags = np.array(image_flags, dtype=np.int64)

    return {
        "natoms": natoms,
        "step": int(header["step"]),
        "time": float(header["time"]),
        "ensemble": header["ensemble"],
        "timestep": float(header["timestep"]),
        "energy": float(header["energy"]),
        "rng_state": header.get("rng_state"),
        "velocity_representation": header.get("velocity_representation", "standard"),
        "symbols": symbols,
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "cell": cell,
        "cell_matrix": cell_matrix,
        "pbc": pbc,
        "image_flags": parsed_image_flags,
    }


def rotate_rst_checkpoint(
    rst_path,
    rst_prev_path,
    atoms,
    velocities,
    step,
    timestep,
    ensemble,
    energy,
    rng_state=None,
    velocity_representation=None,
):
    """
    Rotate runtime checkpoint files and write a new checkpoint.

    Fresh-start backup of pre-existing ``*_md.rst`` and ``*_md_prev.rst``
    files is handled earlier by the MD logger using GROMACS-style numbered
    backups. During an active MD run, the checkpoint writer still preserves
    the most recent previous checkpoint by moving ``rst_path`` to
    ``rst_prev_path`` before writing the new ``rst_path``.

    Parameters
    ----------
    rst_path : str or Path
        Current checkpoint file path.
    rst_prev_path : str or Path
        Previous runtime checkpoint file path.
    atoms : ase.Atoms
        Atomic system.
    velocities : np.ndarray
        Velocities in atomic units, shape (N, 3).
    step : int
        Current MD step number.
    timestep : float
        Timestep in fs.
    ensemble : str
        Ensemble type ('nve', 'nvt', 'npt').
    energy : float
        Total energy in Hartree.
    rng_state : str, optional
        Hex-encoded RNG state (for NVT/NPT).
    velocity_representation : str, optional
        Label describing the semantics of the stored velocities.
    """
    rst_path = Path(rst_path)
    rst_prev_path = Path(rst_prev_path)

    if rst_path.exists() and rst_path.stat().st_size > 0:
        if rst_prev_path.exists():
            rst_prev_path.unlink()
        rst_path.replace(rst_prev_path)

    write_rst(
        rst_path,
        atoms=atoms,
        velocities=velocities,
        step=step,
        timestep=timestep,
        ensemble=ensemble,
        energy=energy,
        rng_state=rng_state,
        velocity_representation=velocity_representation,
    )
