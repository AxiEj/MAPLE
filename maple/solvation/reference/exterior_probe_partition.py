"""Dense target-free exterior MEP fit/audit probes.

The original 26-direction shell is intentionally sparse and remains useful for
cheap numerical canaries.  Observable-supervised source learning needs more
probe points than the 8N radial-GTO source dimension.  This module therefore
uses PySCF's existing 50-point Lebedev rule twice: once in its tabulated frame
and once under one frozen generic rotation.  The two partitions are generated
from geometry and ASE radii only; they are not a PCM cavity or a continuum
surface.

The laboratory-frame sampling is not itself an SO(3)-equivariant operator and
must never be used as one.  Equivariance belongs to the learned scalar/source
architecture; these points are only finite observable loss and audit probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.units import Bohr


DENSE_EXTERIOR_LEBEDEV_ORDER = 11
DENSE_EXTERIOR_LEBEDEV_POINTS = 50
DENSE_EXTERIOR_PARTITION_NAMES = ("fit", "audit")
_AUDIT_QUATERNION = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float64)
_AUDIT_QUATERNION /= np.linalg.norm(_AUDIT_QUATERNION)


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            (
                w * w + x * x - y * y - z * z,
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ),
            (
                2.0 * (x * y + w * z),
                w * w - x * x + y * y - z * z,
                2.0 * (y * z - w * x),
            ),
            (
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                w * w - x * x - y * y + z * z,
            ),
        ],
        dtype=np.float64,
    )


DENSE_EXTERIOR_AUDIT_ROTATION = _quaternion_rotation(_AUDIT_QUATERNION)


@dataclass(frozen=True, slots=True)
class DenseExteriorProbePartition:
    surface_points_bohr: np.ndarray
    quadrature_weights: np.ndarray
    parent_atom_indices: np.ndarray
    partition_indices: np.ndarray
    candidate_count_per_partition: int
    clearance_angstrom: float

    def __post_init__(self) -> None:
        points = np.asarray(self.surface_points_bohr, dtype=np.float64)
        weights = np.asarray(self.quadrature_weights, dtype=np.float64)
        parents = np.asarray(self.parent_atom_indices)
        partitions = np.asarray(self.partition_indices)
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("surface_points_bohr must have finite shape (n,3).")
        if (
            weights.shape != (len(points),)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("quadrature_weights must be finite and positive.")
        for name, values in (("parent", parents), ("partition", partitions)):
            if (
                values.shape != (len(points),)
                or not np.issubdtype(values.dtype, np.integer)
            ):
                raise ValueError(f"{name} indices must be integer vectors.")
        parents = parents.astype(np.int64, copy=True)
        partitions = partitions.astype(np.int64, copy=True)
        if np.any(parents < 0):
            raise ValueError("parent indices must be nonnegative.")
        if not np.array_equal(np.unique(partitions), np.asarray((0, 1))):
            raise ValueError("both fit and audit partitions must be nonempty.")
        candidate_count = int(self.candidate_count_per_partition)
        if candidate_count < len(points) // 2 or candidate_count < 1:
            raise ValueError("candidate_count_per_partition is invalid.")
        clearance = float(self.clearance_angstrom)
        if not np.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("clearance_angstrom must be finite and positive.")
        for name, values in (
            ("surface_points_bohr", points),
            ("quadrature_weights", weights),
            ("parent_atom_indices", parents),
            ("partition_indices", partitions),
        ):
            result = np.array(values, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        object.__setattr__(self, "candidate_count_per_partition", candidate_count)
        object.__setattr__(self, "clearance_angstrom", clearance)

    def mask(self, name: str) -> np.ndarray:
        try:
            index = DENSE_EXTERIOR_PARTITION_NAMES.index(str(name))
        except ValueError as exc:
            raise ValueError(f"unknown exterior probe partition {name!r}.") from exc
        result = self.partition_indices == index
        result.setflags(write=False)
        return result


def build_dense_exterior_probe_partition(
    atom_positions_angstrom: object,
    atomic_radii_angstrom: object,
    *,
    clearance_angstrom: float,
) -> DenseExteriorProbePartition:
    """Return deterministic dense fit/audit exterior probes.

    Each atom contributes the same PySCF 50-point Lebedev candidates at its
    expanded radius.  Candidates inside another expanded atomic sphere are
    discarded independently in each frozen angular frame.
    """

    import pyscf
    from pyscf.dft import gen_grid

    if pyscf.__version__ != "2.13.1":
        raise RuntimeError("Dense exterior probes require the frozen PySCF 2.13.1.")
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
    if int(gen_grid.LEBEDEV_ORDER[DENSE_EXTERIOR_LEBEDEV_ORDER]) != (
        DENSE_EXTERIOR_LEBEDEV_POINTS
    ):
        raise RuntimeError("PySCF Lebedev order-to-size identity changed.")
    angular = np.asarray(
        gen_grid.MakeAngularGrid(DENSE_EXTERIOR_LEBEDEV_POINTS),
        dtype=np.float64,
    )
    if angular.shape != (DENSE_EXTERIOR_LEBEDEV_POINTS, 4):
        raise RuntimeError("PySCF returned an invalid 50-point angular grid.")
    directions = angular[:, :3]
    solid_angle_weights = 4.0 * np.pi * angular[:, 3]
    if (
        not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=2.0e-14)
        or not np.isclose(np.sum(solid_angle_weights), 4.0 * np.pi, atol=2.0e-14)
    ):
        raise RuntimeError("PySCF angular directions or weights are inconsistent.")

    expanded_radii = radii + clearance
    parent_template = np.repeat(
        np.arange(len(positions), dtype=np.int64),
        DENSE_EXTERIOR_LEBEDEV_POINTS,
    )
    weights_template = np.tile(solid_angle_weights, len(positions))
    points_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []
    parents_parts: list[np.ndarray] = []
    partition_parts: list[np.ndarray] = []
    for partition, partition_directions in enumerate(
        (directions, directions @ DENSE_EXTERIOR_AUDIT_ROTATION.T)
    ):
        tiled_directions = np.tile(partition_directions, (len(positions), 1))
        candidate_points = (
            positions[parent_template]
            + expanded_radii[parent_template, None] * tiled_directions
        )
        distances = np.linalg.norm(
            candidate_points[:, None, :] - positions[None, :, :],
            axis=2,
        )
        retained = np.all(
            distances + 2.0e-12 >= expanded_radii[None, :],
            axis=1,
        )
        if int(np.count_nonzero(retained)) < len(positions):
            raise RuntimeError("Dense exterior partition retained too few points.")
        points_parts.append(candidate_points[retained] / Bohr)
        weights_parts.append(weights_template[retained])
        parents_parts.append(parent_template[retained])
        partition_parts.append(
            np.full(np.count_nonzero(retained), partition, dtype=np.int64)
        )
    return DenseExteriorProbePartition(
        surface_points_bohr=np.concatenate(points_parts, axis=0),
        quadrature_weights=np.concatenate(weights_parts),
        parent_atom_indices=np.concatenate(parents_parts),
        partition_indices=np.concatenate(partition_parts),
        candidate_count_per_partition=(
            len(positions) * DENSE_EXTERIOR_LEBEDEV_POINTS
        ),
        clearance_angstrom=clearance,
    )


__all__ = [
    "DENSE_EXTERIOR_AUDIT_ROTATION",
    "DENSE_EXTERIOR_LEBEDEV_ORDER",
    "DENSE_EXTERIOR_LEBEDEV_POINTS",
    "DENSE_EXTERIOR_PARTITION_NAMES",
    "DenseExteriorProbePartition",
    "build_dense_exterior_probe_partition",
]
