"""WS5 — wrapping boundary robustness.

The image-flag count and the wrapped position must come from the same integer
crossing count, so continuous (unwrapped) reconstruction is exact even at cell
boundaries and stable against floating-point noise.  Previously the increment was
np.floor(scaled) while the wrap used ase.wrap()'s own epsilon, which disagreed at
the boundary and produced an off-by-one (sawtooth) error.
"""

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.md.utils import (
    IMAGE_FLAGS_ARRAY,
    ensure_image_flags,
    get_unwrapped_positions,
    wrap_positions_with_image_flags,
)


def _ortho(scaled, cell=(2.0, 2.0, 2.0)) -> Atoms:
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=list(cell), pbc=True)
    atoms.set_scaled_positions(np.array([scaled], dtype=float))
    return atoms


@pytest.mark.parametrize("frac", [0.0, 1.0, 1.0 - 1e-12, 1.0 + 1e-12, -1e-12, 2.0, -1.0])
def test_wrap_boundary_reconstruction_is_exact(frac):
    atoms = _ortho([frac, 0.3, 0.4])
    original = atoms.get_positions().copy()
    wrap_positions_with_image_flags(atoms)

    # Wrapped fractional x lands in [0, 1) up to boundary epsilon.
    sx = atoms.cell.scaled_positions(atoms.get_positions())[0, 0]
    assert -1e-7 <= sx < 1.0 + 1e-7
    # Continuous reconstruction is exact (no off-by-one).
    np.testing.assert_allclose(get_unwrapped_positions(atoms), original, atol=1e-9)
    assert atoms.arrays[IMAGE_FLAGS_ARRAY].dtype.kind == "i"


@pytest.mark.parametrize("noise", [1e-13, -1e-13, 1e-14, -1e-14, 5e-10, -5e-10])
def test_boundary_increment_is_deterministic_under_fp_noise(noise):
    # An atom at a true boundary (scaled == 1.0) read back with FP noise must
    # always count exactly one crossing — never flip between 0 and 1.
    atoms = _ortho([1.0 + noise, 0.5, 0.5])
    ensure_image_flags(atoms)
    wrap_positions_with_image_flags(atoms)
    assert atoms.arrays[IMAGE_FLAGS_ARRAY][0, 0] == 1
    np.testing.assert_allclose(
        get_unwrapped_positions(atoms)[0, 0], (1.0 + noise) * 2.0, atol=1e-9
    )


def test_unwrapped_tracks_continuous_trajectory_no_sawtooth():
    # Drive the wrapped position by Cartesian displacements across a boundary and
    # back; the unwrapped position must equal the cumulative continuous path.
    atoms = _ortho([0.9, 0.5, 0.5])  # x = 1.8 Å in a 2 Å cell
    ensure_image_flags(atoms)
    wrap_positions_with_image_flags(atoms)
    expected = get_unwrapped_positions(atoms)[0, 0]
    for disp in (0.3, 0.3, -0.3, -0.3, -0.3):  # cross +x boundary then return
        pos = atoms.get_positions()
        pos[0, 0] += disp
        atoms.set_positions(pos)
        wrap_positions_with_image_flags(atoms)
        expected += disp
        np.testing.assert_allclose(get_unwrapped_positions(atoms)[0, 0], expected, atol=1e-9)


def test_high_speed_multi_cell_jump():
    atoms = _ortho([0.1, 0.5, 0.5])  # x = 0.2 Å
    ensure_image_flags(atoms)
    pos = atoms.get_positions()
    pos[0, 0] += 4.5  # 0.2 -> 4.7 Å  (scaled 2.35 -> +2 cells)
    atoms.set_positions(pos)
    wrap_positions_with_image_flags(atoms)
    assert atoms.arrays[IMAGE_FLAGS_ARRAY][0, 0] == 2
    np.testing.assert_allclose(get_unwrapped_positions(atoms)[0, 0], 4.7, atol=1e-9)


def test_triclinic_scaled_space_displacement_reconstructs():
    cell = np.array([[2.0, 0.0, 0.0], [0.4, 2.1, 0.0], [0.2, 0.3, 2.2]])
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    atoms.set_scaled_positions([[0.9, 0.95, 0.98]])
    ensure_image_flags(atoms)
    new_scaled = np.array([0.9, 0.95, 0.98]) + np.array([0.4, 0.3, 0.25])  # crosses each axis
    atoms.set_scaled_positions([new_scaled])
    wrap_positions_with_image_flags(atoms)
    np.testing.assert_array_equal(
        atoms.arrays[IMAGE_FLAGS_ARRAY][0], np.floor(new_scaled).astype(np.int64)
    )
    np.testing.assert_allclose(get_unwrapped_positions(atoms)[0], new_scaled @ cell, atol=1e-9)


def test_partial_pbc_axis_is_not_wrapped():
    # Non-periodic axis keeps its (possibly out-of-cell) coordinate and never
    # accrues an image flag.
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[2.0, 2.0, 30.0], pbc=[True, True, False])
    atoms.set_scaled_positions([[1.3, 0.5, 1.4]])
    ensure_image_flags(atoms)
    original = atoms.get_positions().copy()
    wrap_positions_with_image_flags(atoms)
    assert atoms.arrays[IMAGE_FLAGS_ARRAY][0, 2] == 0  # z is non-periodic
    assert atoms.arrays[IMAGE_FLAGS_ARRAY][0, 0] == 1
    np.testing.assert_allclose(get_unwrapped_positions(atoms)[0], original[0], atol=1e-9)
