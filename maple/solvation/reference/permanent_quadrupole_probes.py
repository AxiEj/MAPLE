"""Denser zero-field probes for the P13 permanent source representation.

The P13/R8 model has ``13N-1`` charge-constrained permanent degrees of freedom
but only ``8N-1`` dynamic response degrees.  The response batch therefore keeps
its frozen 50-point partitions, while this zero-field-only builder uses the
existing PySCF 86-point Lebedev rule in fit and rotated audit frames.  No SCF or
field perturbation is required to evaluate a converged checkpoint density on
these points.
"""

from __future__ import annotations

import numpy as np
from ase.units import Bohr

from .exterior_probe_partition import (
    DENSE_EXTERIOR_AUDIT_ROTATION,
    DenseExteriorProbePartition,
)


PERMANENT_P13_LEBEDEV_ORDER = 15
PERMANENT_P13_LEBEDEV_POINTS = 86


def build_permanent_p13_probe_partition(
    atom_positions_angstrom: object,
    atomic_radii_angstrom: object,
    *,
    clearance_angstrom: float,
) -> DenseExteriorProbePartition:
    """Return fit/audit zero-field probes for a 13N permanent source."""

    import pyscf
    from pyscf.dft import gen_grid

    if pyscf.__version__ != "2.13.1":
        raise RuntimeError("P13 permanent probes require frozen PySCF 2.13.1.")
    positions = np.asarray(atom_positions_angstrom, dtype=np.float64)
    radii = np.asarray(atomic_radii_angstrom, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("atom_positions_angstrom must have finite shape (N,3).")
    if (
        radii.shape != (len(positions),)
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError("atomic_radii_angstrom must be finite and positive.")
    clearance = float(clearance_angstrom)
    if not np.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("clearance_angstrom must be finite and positive.")
    if int(gen_grid.LEBEDEV_ORDER[PERMANENT_P13_LEBEDEV_ORDER]) != (
        PERMANENT_P13_LEBEDEV_POINTS
    ):
        raise RuntimeError("PySCF 86-point Lebedev identity changed.")
    angular = np.asarray(
        gen_grid.MakeAngularGrid(PERMANENT_P13_LEBEDEV_POINTS),
        dtype=np.float64,
    )
    if angular.shape != (PERMANENT_P13_LEBEDEV_POINTS, 4):
        raise RuntimeError("PySCF returned an invalid 86-point angular grid.")
    directions = angular[:, :3]
    weights = 4.0 * np.pi * angular[:, 3]
    expanded = radii + clearance
    parents = np.repeat(
        np.arange(len(positions), dtype=np.int64),
        PERMANENT_P13_LEBEDEV_POINTS,
    )
    tiled_weights = np.tile(weights, len(positions))
    point_parts = []
    weight_parts = []
    parent_parts = []
    partition_parts = []
    for partition, frame in enumerate(
        (directions, directions @ DENSE_EXTERIOR_AUDIT_ROTATION.T)
    ):
        tiled_directions = np.tile(frame, (len(positions), 1))
        candidates = (
            positions[parents] + expanded[parents, None] * tiled_directions
        )
        distances = np.linalg.norm(
            candidates[:, None, :] - positions[None, :, :],
            axis=2,
        )
        retained = np.all(distances + 2.0e-12 >= expanded[None, :], axis=1)
        if int(np.count_nonzero(retained)) <= 13 * len(positions) - 1:
            raise RuntimeError(
                "P13 fit/audit partition does not exceed 13N-1 after burial."
            )
        point_parts.append(candidates[retained] / Bohr)
        weight_parts.append(tiled_weights[retained])
        parent_parts.append(parents[retained])
        partition_parts.append(
            np.full(np.count_nonzero(retained), partition, dtype=np.int64)
        )
    return DenseExteriorProbePartition(
        surface_points_bohr=np.concatenate(point_parts),
        quadrature_weights=np.concatenate(weight_parts),
        parent_atom_indices=np.concatenate(parent_parts),
        partition_indices=np.concatenate(partition_parts),
        candidate_count_per_partition=(
            len(positions) * PERMANENT_P13_LEBEDEV_POINTS
        ),
        clearance_angstrom=clearance,
    )


__all__ = [
    "PERMANENT_P13_LEBEDEV_ORDER",
    "PERMANENT_P13_LEBEDEV_POINTS",
    "build_permanent_p13_probe_partition",
]
