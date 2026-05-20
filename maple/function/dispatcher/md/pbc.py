"""Periodic image flags and wrap/unwrap reconstruction.

Owns the per-atom integer image counters and the coordinate wrapping that keeps
the wrapped position and the image-flag crossing count derived from the *same*
integer, so continuous (unwrapped) reconstruction is exact at cell boundaries.
Leaf module (numpy/ASE only); re-exported by ``...md.utils``.
"""

import numpy as np
from ase import Atoms


# Per-atom integer periodic image counters.  Wrapped coordinates remain in
# ``atoms.positions`` for calculator compatibility; this array records how many
# lattice vectors each atom has crossed so analysis trajectories can be
# reconstructed as continuous/unwrapped coordinates.
IMAGE_FLAGS_ARRAY = "maple_image_flags"


def ensure_image_flags(atoms: Atoms) -> np.ndarray:
    """Return MAPLE per-atom image counters, creating zero counters if absent."""
    expected_shape = (len(atoms), 3)
    flags = atoms.arrays.get(IMAGE_FLAGS_ARRAY)
    if flags is None:
        flags = np.zeros(expected_shape, dtype=np.int64)
        atoms.new_array(IMAGE_FLAGS_ARRAY, flags)
        return atoms.arrays[IMAGE_FLAGS_ARRAY]

    flags = np.asarray(flags)
    if flags.shape != expected_shape:
        raise ValueError(
            f"{IMAGE_FLAGS_ARRAY!r} must have shape {expected_shape}, got {flags.shape}."
        )
    if not np.issubdtype(flags.dtype, np.integer):
        atoms.arrays[IMAGE_FLAGS_ARRAY] = flags.astype(np.int64)
    return atoms.arrays[IMAGE_FLAGS_ARRAY]


# Fractional tolerance for snapping a coordinate that sits on a cell boundary to
# the exact integer before flooring.  A coordinate within this distance of an
# integer is treated as being *on* that boundary, so floating-point noise around
# a true boundary cannot flip the integer crossing count between n-1 and n.
WRAP_BOUNDARY_EPS = 1e-9


def wrap_positions_with_image_flags(atoms: Atoms, eps: float = WRAP_BOUNDARY_EPS) -> None:
    """Wrap periodic coordinates and increment image counters consistently.

    Call this after setting positions to the drifted coordinates.  The current
    positions may be outside the primary unit cell; after the call,
    ``atoms.positions`` are wrapped and ``IMAGE_FLAGS_ARRAY`` stores the lattice
    crossings needed to reconstruct continuous coordinates.

    The image-flag increment and the wrapped position are both derived from the
    *same* integer crossing count, so continuous reconstruction is exact at cell
    boundaries.  A symmetric near-integer snap is applied before flooring so a
    coordinate at a true boundary (read back as ``1 ± fp_noise``) always counts
    the same crossing instead of flipping with the sign of the noise.  This
    avoids the off-by-one (sawtooth) error that arose when the increment used
    ``floor()`` while the wrap used ``ase.wrap()``'s independent epsilon.

    Boundary invariant: after the call the wrapped fractional coordinates lie in
    ``[0, 1)`` up to round-off.  ``WRAP_BOUNDARY_EPS`` (1e-9) is the near-integer
    snap tolerance that makes the *image-flag* crossing count deterministic under
    floating-point noise; downstream consumers reading back
    ``set_scaled_positions`` round-off should allow a looser +/-1e-7 slack on the
    ``[0, 1)`` bound (asserted in ``tests/dispatcher/md/test_wrap_boundary.py``).
    """
    if not any(atoms.pbc):
        return

    pbc = np.asarray(atoms.pbc, dtype=bool)
    scaled = atoms.cell.scaled_positions(atoms.get_positions())

    # Symmetric integer-boundary snap (handles values just below AND just above
    # an integer); do NOT use the asymmetric floor(scaled + eps) form.
    nearest = np.round(scaled)
    snapped = np.where(np.isclose(scaled, nearest, atol=eps, rtol=0.0), nearest, scaled)
    image_increment = np.floor(snapped).astype(np.int64)
    image_increment[:, ~pbc] = 0

    flags = ensure_image_flags(atoms)
    flags[:] = flags + image_increment

    # Wrap with the exact integer count we just recorded so the wrapped position
    # and the image flags stay consistent (independent of ase.wrap()'s epsilon).
    # Use the original (un-snapped) fractional value so reconstruction is exact.
    wrapped_scaled = scaled.copy()
    wrapped_scaled[:, pbc] = scaled[:, pbc] - image_increment[:, pbc]
    atoms.set_scaled_positions(wrapped_scaled)


def get_unwrapped_positions(atoms: Atoms) -> np.ndarray:
    """Return continuous Cartesian coordinates reconstructed from image flags."""
    positions = np.asarray(atoms.get_positions(), dtype=float)
    if not any(atoms.pbc):
        return positions.copy()

    flags = ensure_image_flags(atoms).astype(float)
    scaled = atoms.cell.scaled_positions(positions)
    return np.dot(scaled + flags, np.asarray(atoms.cell.array, dtype=float))


def copy_with_unwrapped_positions(atoms: Atoms) -> Atoms:
    """Return an Atoms copy whose positions are continuous/unwrapped."""
    out = atoms.copy()
    out.set_positions(get_unwrapped_positions(atoms))
    out.info["coordinate_mode"] = "unwrapped"
    return out
