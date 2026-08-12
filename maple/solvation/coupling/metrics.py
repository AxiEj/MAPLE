"""Authoritative source--field energy pairing for Route 2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _string_tuple(values: object, *, name: str) -> tuple[str, ...]:
    try:
        result = tuple(_nonempty(value, name=name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings.") from exc
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


@dataclass(frozen=True, slots=True)
class PairingMetric:
    """One immutable ``Q`` block and all metadata needed to interpret it."""

    scalar_id: str
    source_components: tuple[str, ...]
    field_components: tuple[str, ...]
    source_units: tuple[str, ...]
    field_units: tuple[str, ...]
    field_to_source_indices: tuple[int, ...]
    gauge: str
    field_convention: str
    energy_unit: str = "eV"

    def __post_init__(self) -> None:
        for name in ("scalar_id", "gauge", "field_convention", "energy_unit"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        for name in (
            "source_components",
            "field_components",
            "source_units",
            "field_units",
        ):
            object.__setattr__(
                self, name, _string_tuple(getattr(self, name), name=name)
            )
        try:
            permutation = tuple(self.field_to_source_indices)
        except TypeError as exc:
            raise TypeError(
                "field_to_source_indices must be an iterable of integers."
            ) from exc
        object.__setattr__(self, "field_to_source_indices", permutation)

        size = len(self.source_components)
        if len(self.field_components) != size:
            raise ValueError("source and field component counts must be equal.")
        if (
            len(set(self.source_components)) != size
            or len(set(self.field_components)) != size
        ):
            raise ValueError("source and field component names must be unique.")
        if len(self.source_units) != size or len(self.field_units) != size:
            raise ValueError("component unit metadata must match the component count.")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in permutation
        ):
            raise TypeError("field_to_source_indices must contain only integers.")
        if tuple(sorted(permutation)) != tuple(range(size)):
            raise ValueError("field_to_source_indices must be a complete permutation.")

    @property
    def component_count(self) -> int:
        return len(self.source_components)

    @property
    def block(self) -> np.ndarray:
        result = np.zeros((self.component_count, self.component_count), dtype=float)
        result[np.arange(self.component_count), self.field_to_source_indices] = 1.0
        result.setflags(write=False)
        return result

    @property
    def Q(self) -> np.ndarray:
        return self.block

    def _blocks(self, values: object, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if (
            array.ndim < 1
            or array.shape[-1] != self.component_count
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(
                f"{name} must be finite with trailing dimension {self.component_count}; "
                f"received {array.shape}."
            )
        return array

    def field_to_source_dual(self, field: object) -> np.ndarray:
        """Apply ``Q`` to blocks shaped ``(..., component_count)``."""

        values = self._blocks(field, name="field")
        return np.einsum("ij,...j->...i", self.block, values)

    def source_to_field_dual(self, source: object) -> np.ndarray:
        """Apply ``Q.T`` to blocks shaped ``(..., component_count)``."""

        values = self._blocks(source, name="source")
        return np.einsum("ji,...j->...i", self.block, values)

    def pair(self, source: object, field: object) -> float:
        source_array = self._blocks(source, name="source")
        field_array = self._blocks(field, name="field")
        if source_array.shape != field_array.shape:
            raise ValueError("source and field must have identical shapes.")
        return float(np.vdot(source_array, self.field_to_source_dual(field_array)))

    def metadata(self) -> dict[str, object]:
        return {
            "scalar_id": self.scalar_id,
            "source_components": list(self.source_components),
            "field_components": list(self.field_components),
            "source_units": list(self.source_units),
            "field_units": list(self.field_units),
            "field_to_source_indices": list(self.field_to_source_indices),
            "gauge": self.gauge,
            "field_convention": self.field_convention,
            "energy_unit": self.energy_unit,
        }

    def metadata_hash(self) -> str:
        encoded = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


ATOMIC_L1_PAIRING = PairingMetric(
    scalar_id="maple.route2.atomic-l1-pairing.v1",
    source_components=("net_monopole", "real_l1_m0", "real_l1_m1", "real_l1_m_minus1"),
    field_components=(
        "potential",
        "potential_gradient_x",
        "potential_gradient_y",
        "potential_gradient_z",
    ),
    source_units=("e", "e*angstrom", "e*angstrom", "e*angstrom"),
    field_units=("eV/e", "eV/(e*angstrom)", "eV/(e*angstrom)", "eV/(e*angstrom)"),
    field_to_source_indices=(0, 2, 3, 1),
    gauge="continuum-zero-at-infinity",
    field_convention="positive-energy-dual: pair(c,u)=c^T Q u",
)

# The official MACE-POLAR-1 receiver contains two radial ``l<=1`` GTO
# channels.  This layout keeps the physical potential/gradient values (before
# the checkpoint's diagonal normalization) as the public energy-dual field.
# Each source component is therefore an ordinary unit-multipole coefficient
# for the matching radial Gaussian, and the pairing is the identity in this
# deliberately raw-dual ordering.
MACE_POLAR_RADIAL_GTO_PAIRING = PairingMetric(
    scalar_id="maple.route2.mace-polar-radial-gto-pairing.v1",
    source_components=(
        "net_monopole_sigma_1p5",
        "net_monopole_sigma_3p0",
        "real_l1_m0_sigma_1p5",
        "real_l1_m1_sigma_1p5",
        "real_l1_mminus1_sigma_1p5",
        "real_l1_m0_sigma_3p0",
        "real_l1_m1_sigma_3p0",
        "real_l1_mminus1_sigma_3p0",
    ),
    field_components=(
        "potential_sigma_1p5",
        "potential_sigma_3p0",
        "potential_gradient_y_sigma_1p5",
        "potential_gradient_z_sigma_1p5",
        "potential_gradient_x_sigma_1p5",
        "potential_gradient_y_sigma_3p0",
        "potential_gradient_z_sigma_3p0",
        "potential_gradient_x_sigma_3p0",
    ),
    source_units=(
        "e",
        "e",
        "e*angstrom",
        "e*angstrom",
        "e*angstrom",
        "e*angstrom",
        "e*angstrom",
        "e*angstrom",
    ),
    field_units=(
        "eV/e",
        "eV/e",
        "eV/(e*angstrom)",
        "eV/(e*angstrom)",
        "eV/(e*angstrom)",
        "eV/(e*angstrom)",
        "eV/(e*angstrom)",
        "eV/(e*angstrom)",
    ),
    field_to_source_indices=tuple(range(8)),
    gauge="continuum-zero-at-infinity",
    field_convention=(
        "positive-energy-dual radial GTO field; checkpoint feature "
        "normalization is a model-adapter representation transform"
    ),
)

AUTHORITATIVE_Q = ATOMIC_L1_PAIRING
QPairing = PairingMetric
PAIRING_REGISTRY = MappingProxyType(
    {
        metric.scalar_id: metric
        for metric in (ATOMIC_L1_PAIRING, MACE_POLAR_RADIAL_GTO_PAIRING)
    }
)


def get_pairing_metric(metric_id: str) -> PairingMetric:
    try:
        return PAIRING_REGISTRY[metric_id]
    except KeyError as exc:
        raise KeyError(f"Unregistered Route-2 pairing metric: {metric_id!r}.") from exc


__all__ = [
    "ATOMIC_L1_PAIRING",
    "AUTHORITATIVE_Q",
    "MACE_POLAR_RADIAL_GTO_PAIRING",
    "PAIRING_REGISTRY",
    "PairingMetric",
    "QPairing",
    "get_pairing_metric",
]
