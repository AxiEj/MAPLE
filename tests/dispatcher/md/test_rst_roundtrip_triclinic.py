"""WS1 — oriented-triclinic RST round-trip (exact double round-trip).

cellpar() throws away the cell *orientation*: an oriented triclinic cell and
its ``Cell.fromcellpar(cellpar)`` reconstruction share lengths/angles but differ
as 3x3 matrices, so scaled coordinates, image flags and unwrapped positions all
shift on restart.  The RST now persists the full 3x3 ``cell_matrix`` at ``:.17g``
(and positions/velocities at ``:.17g``), so write -> read -> logger-restore
reproduces the exact state.  Pre-cell_matrix files still read via the cellpar
fallback.
"""

import numpy as np
from ase import Atoms
from ase.cell import Cell

from maple.function.dispatcher.md.rst_io import write_rst, read_rst
from maple.function.dispatcher.md.utils import IMAGE_FLAGS_ARRAY
from maple.function.dispatcher.md.logger import MDLogger


# Oriented (a-vector not along x) skew triclinic cell.
ORIENTED_TRICLINIC = np.array([
    [5.2, 0.7, 0.3],
    [0.6, 5.5, 0.4],
    [0.5, 0.8, 5.8],
])


def _atoms():
    atoms = Atoms(
        "Ar3",
        positions=[[0.1, 0.2, 0.3], [2.0, 1.0, 0.5], [3.1, 4.2, 1.7]],
        cell=ORIENTED_TRICLINIC,
        pbc=True,
    )
    atoms.new_array(
        IMAGE_FLAGS_ARRAY, np.array([[1, -2, 0], [0, 3, -1], [-4, 0, 2]], dtype=np.int64)
    )
    return atoms


def _velocities():
    return np.array([
        [1.234567890123e-3, -9.87e-4, 5.5e-5],
        [-2.2e-3, 3.3e-4, -1.1e-6],
        [7.7e-5, -8.8e-5, 9.9e-7],
    ])


def test_write_read_roundtrip_is_exact(tmp_path):
    atoms = _atoms()
    velocities = _velocities()
    rst = tmp_path / "tri_md.rst"
    write_rst(rst, atoms, velocities, step=10, timestep=0.5, ensemble="nve", energy=-1.5)

    state = read_rst(rst)
    assert state["cell_matrix"] is not None
    assert np.array_equal(state["cell_matrix"], ORIENTED_TRICLINIC)
    assert np.array_equal(state["positions"], atoms.get_positions())
    assert np.array_equal(state["velocities"], velocities)
    assert np.array_equal(state["image_flags"], atoms.arrays[IMAGE_FLAGS_ARRAY])


def test_cellpar_reconstruction_loses_orientation(tmp_path):
    atoms = _atoms()
    rst = tmp_path / "tri_md.rst"
    write_rst(rst, atoms, np.zeros((3, 3)), step=0, timestep=0.5, ensemble="nve", energy=0.0)
    state = read_rst(rst)

    # The cellpar path (the old behaviour) reconstructs a *differently oriented*
    # matrix — this is the bug the cell_matrix line fixes.
    cellpar_cell = np.asarray(Cell.fromcellpar(state["cell"]))
    assert not np.allclose(cellpar_cell, ORIENTED_TRICLINIC, atol=1e-6)
    # The exact matrix round-trips bit-for-bit.
    assert np.array_equal(state["cell_matrix"], ORIENTED_TRICLINIC)


def test_old_format_without_cell_matrix_is_readable(tmp_path):
    atoms = _atoms()
    rst = tmp_path / "old_md.rst"
    write_rst(rst, atoms, np.zeros((3, 3)), step=0, timestep=0.5, ensemble="nve", energy=0.0)

    # Emulate a pre-cell_matrix checkpoint by removing the line.
    stripped = "\n".join(
        line for line in rst.read_text().splitlines() if not line.startswith("cell_matrix")
    ) + "\n"
    rst.write_text(stripped)

    state = read_rst(rst)
    assert state["cell_matrix"] is None
    assert state["cell"] is not None  # cellpar line still present for fallback


def test_logger_restart_restores_exact_orientation(tmp_path):
    atoms = _atoms()
    velocities = _velocities()
    rst = tmp_path / "first_md.rst"
    write_rst(rst, atoms, velocities, step=5, timestep=0.5, ensemble="nve", energy=-1.0)

    # Fresh atoms carrying a *different* axis-aligned cell, to prove the restore
    # overrides it with the exact triclinic matrix from the checkpoint.
    fresh = Atoms("Ar3", positions=np.zeros((3, 3)), cell=np.eye(3) * 7.0, pbc=True)
    logger = MDLogger(
        output_path=str(tmp_path / "restart.out"),
        log_every=1, traj_every=1, traj_format="xyz", verbose=0, debug=False,
    )
    out_atoms, out_v, step_offset = logger.restart_simulation(
        ensemble="nve", timestep=0.5, n_steps=1000, temperature=300.0,
        atoms=fresh, rst_file=str(rst), load_state=True,
    )

    assert np.array_equal(out_atoms.cell.array, ORIENTED_TRICLINIC)
    assert np.array_equal(out_atoms.get_positions(), atoms.get_positions())
    assert np.array_equal(out_v, velocities)


def test_logger_restart_falls_back_for_old_format(tmp_path):
    atoms = _atoms()
    rst = tmp_path / "first_md.rst"
    write_rst(rst, atoms, _velocities(), step=5, timestep=0.5, ensemble="nve", energy=-1.0)
    stripped = "\n".join(
        line for line in rst.read_text().splitlines() if not line.startswith("cell_matrix")
    ) + "\n"
    rst.write_text(stripped)

    fresh = Atoms("Ar3", positions=np.zeros((3, 3)), cell=np.eye(3) * 7.0, pbc=True)
    logger = MDLogger(
        output_path=str(tmp_path / "restart.out"),
        log_every=1, traj_every=1, traj_format="xyz", verbose=0, debug=False,
    )
    out_atoms, _out_v, _offset = logger.restart_simulation(
        ensemble="nve", timestep=0.5, n_steps=1000, temperature=300.0,
        atoms=fresh, rst_file=str(rst), load_state=True,
    )
    # Fallback path: cell is reconstructed from cellpar (orientation not exact,
    # but the run restarts without error and the cell stays a valid rank-3 cell).
    assert out_atoms.cell.rank == 3
    np.testing.assert_allclose(out_atoms.cell.cellpar(), atoms.cell.cellpar(), rtol=1e-9, atol=1e-9)
