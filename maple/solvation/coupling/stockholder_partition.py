"""Smooth promolecular stockholder partition for local scalar-field targets.

The partition reuses the independently frozen positive atomic Gaussian-mixture
asset from the rho-DROP work.  It does not reinterpret that promolecular
density as a molecular electronic state.  Only the normalized stockholder
weights are used to assign a reference QM field to unique smooth atomic owners.

All public coordinates are in bohr.  The implementation evaluates log densities
and nested softmaxes, so weights remain finite far outside the molecule without
introducing a distance cutoff.  Spatial gradients and nuclear VJPs are analytic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


STOCKHOLDER_PARTITION_PROVIDER_ID = (
    "maple.route2.coupling.promolecular-stockholder-partition.v1"
)
STOCKHOLDER_PARTITION_ALGORITHM = (
    "positive-gaussian-mixture-logsumexp-hirshfeld-v1"
)


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _points(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != 3
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be finite with shape (M,3).")
    return np.ascontiguousarray(array)


def _numbers(values: object, *, atom_count: int) -> np.ndarray:
    raw = np.asarray(values)
    numeric = np.asarray(raw, dtype=np.float64)
    rounded = np.rint(numeric)
    if (
        numeric.shape != (atom_count,)
        or not np.all(np.isfinite(numeric))
        or np.any(rounded != numeric)
        or np.any(rounded < 1.0)
    ):
        raise ValueError("atomic_numbers must contain one positive integer per atom.")
    return rounded.astype(np.int64)


@dataclass(frozen=True, slots=True)
class StockholderPartitionEvaluation:
    """One immutable point/geometry-local partition evaluation."""

    configuration_sha256: str
    weights: np.ndarray
    atomic_log_density_spatial_gradients_bohr_inv: np.ndarray
    weight_spatial_gradients_bohr_inv: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.configuration_sha256, str)
            or len(self.configuration_sha256) != 64
        ):
            raise ValueError("configuration_sha256 must be a SHA256 digest.")
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 2 or not weights.shape[0] or not weights.shape[1]:
            raise ValueError("weights must have nonempty shape (M,N).")
        point_count, atom_count = weights.shape
        weights = _readonly(
            weights,
            shape=(point_count, atom_count),
            name="weights",
        )
        log_gradients = _readonly(
            self.atomic_log_density_spatial_gradients_bohr_inv,
            shape=(point_count, atom_count, 3),
            name="atomic log-density spatial gradients",
        )
        gradients = _readonly(
            self.weight_spatial_gradients_bohr_inv,
            shape=(point_count, atom_count, 3),
            name="weight spatial gradients",
        )
        if np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("stockholder weights must lie in [0,1].")
        if not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            rtol=0.0,
            atol=4.0e-15,
        ):
            raise ValueError("stockholder weights must sum to one per point.")
        if not np.allclose(
            np.sum(gradients, axis=1),
            0.0,
            rtol=0.0,
            atol=2.0e-13,
        ):
            raise ValueError("partition spatial gradients must sum to zero.")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "atomic_log_density_spatial_gradients_bohr_inv",
            log_gradients,
        )
        object.__setattr__(
            self,
            "weight_spatial_gradients_bohr_inv",
            gradients,
        )

    @property
    def point_count(self) -> int:
        return int(self.weights.shape[0])

    @property
    def atom_count(self) -> int:
        return int(self.weights.shape[1])

    def point_vjp(self, weight_cotangent: object) -> np.ndarray:
        """Return ``d sum(w_bar*w)/d points`` in inverse-bohr pairing."""

        cotangent = np.asarray(weight_cotangent, dtype=np.float64)
        if cotangent.shape != self.weights.shape or not np.all(np.isfinite(cotangent)):
            raise ValueError("weight_cotangent must match the partition weights.")
        mean = np.sum(self.weights * cotangent, axis=1, keepdims=True)
        log_cotangent = self.weights * (cotangent - mean)
        return np.einsum(
            "ma,max->mx",
            log_cotangent,
            self.atomic_log_density_spatial_gradients_bohr_inv,
            optimize=True,
        )

    def nuclear_vjp(self, weight_cotangent: object) -> np.ndarray:
        """Return the fixed-point-coordinate VJP with respect to atom centres."""

        cotangent = np.asarray(weight_cotangent, dtype=np.float64)
        if cotangent.shape != self.weights.shape or not np.all(np.isfinite(cotangent)):
            raise ValueError("weight_cotangent must match the partition weights.")
        mean = np.sum(self.weights * cotangent, axis=1, keepdims=True)
        log_cotangent = self.weights * (cotangent - mean)
        return -np.einsum(
            "ma,max->ax",
            log_cotangent,
            self.atomic_log_density_spatial_gradients_bohr_inv,
            optimize=True,
        )


class PromolecularStockholderPartition:
    """Content-addressed partition provider over a verified atomic asset."""

    __slots__ = ("_asset", "_asset_content_sha256", "_configuration_sha256")

    provider_id = STOCKHOLDER_PARTITION_PROVIDER_ID
    algorithm = STOCKHOLDER_PARTITION_ALGORITHM

    def __init__(self, asset: Any) -> None:
        for name in ("mixture", "require_content_integrity"):
            if not callable(getattr(asset, name, None)):
                raise TypeError(f"atomic reference-density asset must implement {name}().")
        content_sha256 = getattr(asset, "content_sha256", None)
        if not isinstance(content_sha256, str) or len(content_sha256) != 64:
            raise ValueError("atomic reference-density asset lacks content identity.")
        asset.require_content_integrity()
        self._asset = asset
        self._asset_content_sha256 = content_sha256
        self._configuration_sha256 = self._current_configuration_sha256()

    def _current_configuration_sha256(self) -> str:
        return _hash(
            {
                "provider_id": self.provider_id,
                "algorithm": self.algorithm,
                "atomic_reference_density_content_sha256": (
                    self._asset_content_sha256
                ),
                "distance_or_density_cutoff": None,
                "coordinate_unit": "bohr",
                "implementation_sha256": _implementation_sha256(),
            }
        )

    def configuration_sha256(self) -> str:
        self._asset.require_content_integrity()
        if self._current_configuration_sha256() != self._configuration_sha256:
            raise RuntimeError("stockholder partition configuration drifted.")
        return self._configuration_sha256

    def evaluate(
        self,
        *,
        points_bohr: object,
        atomic_numbers: object,
        positions_bohr: object,
    ) -> StockholderPartitionEvaluation:
        points = _points(points_bohr, name="points_bohr")
        positions = _points(positions_bohr, name="positions_bohr")
        numbers = _numbers(atomic_numbers, atom_count=len(positions))
        point_count = len(points)
        atom_count = len(positions)
        log_density = np.empty((point_count, atom_count), dtype=np.float64)
        log_gradient = np.empty((point_count, atom_count, 3), dtype=np.float64)
        for atom_index, (number, centre) in enumerate(
            zip(numbers, positions, strict=True)
        ):
            mixture = self._asset.mixture(int(number))
            counts = np.asarray(mixture.electron_counts, dtype=np.float64)
            exponents = np.asarray(
                mixture.gaussian_exponents_bohr2,
                dtype=np.float64,
            )
            positive = counts > 0.0
            counts = counts[positive]
            exponents = exponents[positive]
            displacement = points - centre
            radius_squared = np.sum(displacement * displacement, axis=1)
            log_components = (
                np.log(counts)[None, :]
                + 1.5 * (np.log(exponents)[None, :] - math.log(math.pi))
                - radius_squared[:, None] * exponents[None, :]
            )
            component_maximum = np.max(log_components, axis=1, keepdims=True)
            component_weights = np.exp(log_components - component_maximum)
            component_sum = np.sum(component_weights, axis=1, keepdims=True)
            probabilities = component_weights / component_sum
            log_density[:, atom_index] = (
                component_maximum[:, 0] + np.log(component_sum[:, 0])
            )
            mean_exponent = probabilities @ exponents
            log_gradient[:, atom_index] = (
                -2.0 * displacement * mean_exponent[:, None]
            )
        atomic_maximum = np.max(log_density, axis=1, keepdims=True)
        unnormalized = np.exp(log_density - atomic_maximum)
        weights = unnormalized / np.sum(unnormalized, axis=1, keepdims=True)
        mean_log_gradient = np.einsum(
            "ma,max->mx",
            weights,
            log_gradient,
            optimize=True,
        )
        weight_gradient = weights[:, :, None] * (
            log_gradient - mean_log_gradient[:, None, :]
        )
        return StockholderPartitionEvaluation(
            configuration_sha256=self.configuration_sha256(),
            weights=weights,
            atomic_log_density_spatial_gradients_bohr_inv=log_gradient,
            weight_spatial_gradients_bohr_inv=weight_gradient,
        )


__all__ = [
    "PromolecularStockholderPartition",
    "STOCKHOLDER_PARTITION_ALGORITHM",
    "STOCKHOLDER_PARTITION_PROVIDER_ID",
    "StockholderPartitionEvaluation",
]
