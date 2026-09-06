"""SO(3)-intertwining one-centre Coulomb whitening for auxiliary functions.

This reference builder derives a fixed element-local coordinate transform from
an isolated atom.  It never uses a molecular geometry, cavity, model output, or
solvation target.  Radial functions are mixed only within one angular momentum
``l`` and the identical radial transform is applied to every ``m`` component;
the resulting transform therefore commutes with rotations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class AtomicAuxiliaryWhiteningBlock:
    angular_momentum: int
    radial_dimension: int
    radial_transform: np.ndarray
    normalized_condition_number: float
    whitened_identity_max_absolute_error: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.angular_momentum, bool)
            or not isinstance(self.angular_momentum, int)
            or self.angular_momentum < 0
        ):
            raise ValueError("angular_momentum must be a nonnegative integer.")
        if (
            isinstance(self.radial_dimension, bool)
            or not isinstance(self.radial_dimension, int)
            or self.radial_dimension < 1
        ):
            raise ValueError("radial_dimension must be positive.")
        transform = _readonly(
            self.radial_transform,
            shape=(self.radial_dimension, self.radial_dimension),
            name="radial_transform",
        )
        condition = float(self.normalized_condition_number)
        error = float(self.whitened_identity_max_absolute_error)
        if (
            not math.isfinite(condition)
            or condition < 1.0
            or not math.isfinite(error)
            or error < 0.0
        ):
            raise ValueError("atomic whitening diagnostics are invalid.")
        object.__setattr__(self, "radial_transform", transform)
        object.__setattr__(self, "normalized_condition_number", condition)
        object.__setattr__(self, "whitened_identity_max_absolute_error", error)


@dataclass(frozen=True, slots=True)
class AtomicAuxiliaryWhitening:
    dimension: int
    transform: np.ndarray
    blocks: tuple[AtomicAuxiliaryWhiteningBlock, ...]
    full_identity_max_absolute_error: float
    forbidden_angular_coupling_max_absolute_value: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 1
        ):
            raise ValueError("dimension must be positive.")
        transform = _readonly(
            self.transform,
            shape=(self.dimension, self.dimension),
            name="transform",
        )
        if not self.blocks or sum(
            (2 * block.angular_momentum + 1) * block.radial_dimension
            for block in self.blocks
        ) != self.dimension:
            raise ValueError("whitening blocks do not cover the auxiliary space.")
        full_error = float(self.full_identity_max_absolute_error)
        angular_error = float(self.forbidden_angular_coupling_max_absolute_value)
        if (
            not math.isfinite(full_error)
            or full_error < 0.0
            or not math.isfinite(angular_error)
            or angular_error < 0.0
        ):
            raise ValueError("full whitening diagnostics are invalid.")
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "full_identity_max_absolute_error", full_error)
        object.__setattr__(
            self,
            "forbidden_angular_coupling_max_absolute_value",
            angular_error,
        )


def build_atomic_coulomb_whitening(
    auxiliary: Any,
    *,
    angular_block_absolute_tolerance: float = 2.0e-10,
) -> AtomicAuxiliaryWhitening:
    """Return a fixed one-centre transform ``W`` with ``W.T J W = I``.

    ``auxiliary`` must contain exactly one atom and use real spherical AOs.
    No eigenvalue is discarded.  A nonpositive radial metric or a violation of
    the expected one-centre angular selection rules fails closed.
    """

    if int(auxiliary.natm) != 1 or bool(auxiliary.cart):
        raise ValueError("atomic whitening requires one atom and spherical AOs.")
    tolerance = float(angular_block_absolute_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("angular-block tolerance must be positive and finite.")
    metric = np.asarray(auxiliary.intor("int2c2e"), dtype=np.float64)
    dimension = int(auxiliary.nao_nr())
    if metric.shape != (dimension, dimension) or not np.all(np.isfinite(metric)):
        raise RuntimeError("atomic auxiliary Coulomb metric is invalid.")
    metric = 0.5 * (metric + metric.T)
    ao_locations = np.asarray(auxiliary.ao_loc_nr(), dtype=np.int64)
    channels: dict[int, list[tuple[int, int]]] = {}
    ao_groups: dict[int, list[int]] = {}
    for shell in range(int(auxiliary.nbas)):
        angular = int(auxiliary.bas_angular(shell))
        contractions = int(auxiliary.bas_nctr(shell))
        width = 2 * angular + 1
        start = int(ao_locations[shell])
        channels.setdefault(angular, [])
        ao_groups.setdefault(angular, [])
        for contraction in range(contractions):
            channels[angular].append((shell, contraction))
            ao_groups[angular].extend(
                range(
                    start + contraction * width,
                    start + (contraction + 1) * width,
                )
            )

    forbidden = 0.0
    angular_values = sorted(channels)
    for left_index, left in enumerate(angular_values):
        for right in angular_values[left_index + 1 :]:
            block = metric[np.ix_(ao_groups[left], ao_groups[right])]
            forbidden = max(forbidden, float(np.max(np.abs(block), initial=0.0)))
    scale = max(1.0, float(np.max(np.abs(metric))))
    if forbidden > tolerance * scale:
        raise RuntimeError("one-centre metric couples different angular momenta.")

    transform = np.zeros_like(metric)
    block_records = []
    for angular in angular_values:
        width = 2 * angular + 1
        radial_dimension = len(channels[angular])
        radial_blocks = []
        indices_by_m = []
        for m_offset in range(width):
            indices = []
            for shell, contraction in channels[angular]:
                start = int(ao_locations[shell])
                indices.append(start + contraction * width + m_offset)
            indices_by_m.append(indices)
            radial_blocks.append(metric[np.ix_(indices, indices)])
        radial_metric = np.mean(radial_blocks, axis=0)
        radial_metric = 0.5 * (radial_metric + radial_metric.T)
        block_mismatch = max(
            float(np.max(np.abs(block - radial_metric))) for block in radial_blocks
        )
        if block_mismatch > tolerance * scale:
            raise RuntimeError("one-centre radial metric differs across m channels.")
        radial_diagonal = np.diag(radial_metric)
        if np.any(radial_diagonal <= 0.0):
            raise RuntimeError("one-centre radial metric has nonpositive diagonal.")
        inverse_sqrt_diagonal = 1.0 / np.sqrt(radial_diagonal)
        normalized_radial_metric = (
            inverse_sqrt_diagonal[:, None]
            * radial_metric
            * inverse_sqrt_diagonal[None, :]
        )
        normalized_radial_metric = 0.5 * (
            normalized_radial_metric + normalized_radial_metric.T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(normalized_radial_metric)
        if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
            raise RuntimeError("one-centre radial Coulomb metric is not positive.")
        normalized_transform = eigenvectors @ (
            eigenvectors.T / np.sqrt(eigenvalues)[:, None]
        )
        radial_transform = inverse_sqrt_diagonal[:, None] * normalized_transform
        whitened = radial_transform.T @ radial_metric @ radial_transform
        identity_error = float(
            np.max(np.abs(whitened - np.eye(radial_dimension)))
        )
        for indices in indices_by_m:
            transform[np.ix_(indices, indices)] = radial_transform
        block_records.append(
            AtomicAuxiliaryWhiteningBlock(
                angular_momentum=angular,
                radial_dimension=radial_dimension,
                radial_transform=radial_transform,
                normalized_condition_number=float(eigenvalues[-1] / eigenvalues[0]),
                whitened_identity_max_absolute_error=identity_error,
            )
        )
    full_whitened = transform.T @ metric @ transform
    full_error = float(np.max(np.abs(full_whitened - np.eye(dimension))))
    return AtomicAuxiliaryWhitening(
        dimension=dimension,
        transform=transform,
        blocks=tuple(block_records),
        full_identity_max_absolute_error=full_error,
        forbidden_angular_coupling_max_absolute_value=forbidden,
    )


__all__ = [
    "AtomicAuxiliaryWhitening",
    "AtomicAuxiliaryWhiteningBlock",
    "build_atomic_coulomb_whitening",
]
