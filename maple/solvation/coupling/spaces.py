"""Canonical source/field spaces and matrix-free affine charge coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Protocol

import numpy as np

from maple.solvation.api.profiles import (
    LEGACY_UNBOUND_COORDINATE_CONTRACT_ID,
    MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
)

from .metrics import (
    ATOMIC_L1_PAIRING,
    MACE_POLAR_RADIAL_GTO_PAIRING,
    PairingMetric,
)


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


def _atom_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("atom_count must be a positive integer.")
    return value


def _validated(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {shape}; received {array.shape}."
        )
    return np.array(array, copy=True)


@dataclass(frozen=True, slots=True)
class SourceSpace:
    scalar_id: str
    representation: str
    components: tuple[str, ...]
    units: tuple[str, ...]
    charge_component: int = 0
    charge_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scalar_id", _nonempty(self.scalar_id, name="scalar_id")
        )
        object.__setattr__(
            self,
            "representation",
            _nonempty(self.representation, name="representation"),
        )
        object.__setattr__(
            self, "components", _string_tuple(self.components, name="components")
        )
        object.__setattr__(self, "units", _string_tuple(self.units, name="units"))
        if len(self.components) != len(self.units):
            raise ValueError("source components and units must have equal lengths.")
        if len(set(self.components)) != len(self.components):
            raise ValueError("source component names must be unique.")
        if isinstance(self.charge_component, bool) or not isinstance(
            self.charge_component, int
        ):
            raise TypeError("charge_component must be an integer.")
        if not 0 <= self.charge_component < len(self.components):
            raise ValueError("charge_component is outside the component range.")
        if self.charge_weights is not None:
            try:
                weights = tuple(float(value) for value in self.charge_weights)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "charge_weights must be an iterable of finite numbers or None."
                ) from exc
            if len(weights) != len(self.components):
                raise ValueError(
                    "charge_weights must match the source component count."
                )
            if not np.all(np.isfinite(weights)) or not np.any(np.asarray(weights)):
                raise ValueError(
                    "charge_weights must be finite and contain a nonzero value."
                )
            object.__setattr__(self, "charge_weights", weights)

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def effective_charge_weights(self) -> tuple[float, ...]:
        """Return the authoritative per-component total-charge functional.

        ``None`` retains the original one-hot ``charge_component`` contract and
        therefore preserves all legacy metadata hashes.  Explicit weights are
        used only by source spaces whose physical charge is distributed across
        more than one representation channel.
        """

        if self.charge_weights is not None:
            return self.charge_weights
        return tuple(
            1.0 if index == self.charge_component else 0.0
            for index in range(self.component_count)
        )

    def shape(self, atom_count: int) -> tuple[int, int]:
        return (_atom_count(atom_count), self.component_count)

    expected_shape = shape

    def validate(
        self, values: object, *, atom_count: int, name: str = "source"
    ) -> np.ndarray:
        return _validated(values, shape=self.shape(atom_count), name=name)

    validate_source = validate

    def total_charge(self, values: object, *, atom_count: int) -> float:
        source = self.validate(values, atom_count=atom_count)
        return float(np.sum(source * np.asarray(self.effective_charge_weights)))

    def metadata(self) -> dict[str, object]:
        metadata = {
            "scalar_id": self.scalar_id,
            "representation": self.representation,
            "components": list(self.components),
            "units": list(self.units),
            "charge_component": self.charge_component,
        }
        if self.charge_weights is not None:
            metadata["charge_weights"] = list(self.charge_weights)
        return metadata

    def metadata_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.metadata(), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class FieldDualSpace:
    scalar_id: str
    representation: str
    components: tuple[str, ...]
    units: tuple[str, ...]
    source_space: SourceSpace
    pairing_metric: PairingMetric

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scalar_id", _nonempty(self.scalar_id, name="scalar_id")
        )
        object.__setattr__(
            self,
            "representation",
            _nonempty(self.representation, name="representation"),
        )
        object.__setattr__(
            self, "components", _string_tuple(self.components, name="components")
        )
        object.__setattr__(self, "units", _string_tuple(self.units, name="units"))
        if not isinstance(self.source_space, SourceSpace):
            raise TypeError("source_space must be a SourceSpace.")
        if not isinstance(self.pairing_metric, PairingMetric):
            raise TypeError("pairing_metric must be a PairingMetric.")
        if len(self.components) != len(self.units):
            raise ValueError("field components and units must have equal lengths.")
        if len(set(self.components)) != len(self.components):
            raise ValueError("field component names must be unique.")
        metric = self.pairing_metric
        if (
            self.source_space.components != metric.source_components
            or self.source_space.units != metric.source_units
        ):
            raise ValueError(
                "source_space is incompatible with pairing_metric source metadata."
            )
        if (
            self.components != metric.field_components
            or self.units != metric.field_units
        ):
            raise ValueError("field metadata is incompatible with pairing_metric.")

    @property
    def gauge(self) -> str:
        return self.pairing_metric.gauge

    @property
    def field_convention(self) -> str:
        return self.pairing_metric.field_convention

    def shape(self, atom_count: int) -> tuple[int, int]:
        return (_atom_count(atom_count), len(self.components))

    expected_shape = shape

    def validate(
        self, values: object, *, atom_count: int, name: str = "field"
    ) -> np.ndarray:
        return _validated(values, shape=self.shape(atom_count), name=name)

    validate_field = validate

    def pair(self, source: object, field: object, *, atom_count: int) -> float:
        source_values = self.source_space.validate(source, atom_count=atom_count)
        field_values = self.validate(field, atom_count=atom_count)
        return self.pairing_metric.pair(source_values, field_values)

    def metadata(self) -> dict[str, object]:
        return {
            "scalar_id": self.scalar_id,
            "representation": self.representation,
            "components": list(self.components),
            "units": list(self.units),
            "source_space": self.source_space.metadata(),
            "gauge": self.gauge,
            "field_convention": self.field_convention,
            "pairing_metric": self.pairing_metric.metadata(),
        }

    def metadata_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.metadata(), sort_keys=True).encode()
        ).hexdigest()


class ReducedCoordinates(Protocol):
    """Static, matrix-free coordinates for one affine source constraint."""

    atom_count: int
    total_charge: float
    source_space: SourceSpace

    @property
    def source_dimension(self) -> int: ...

    @property
    def reduced_dimension(self) -> int: ...

    @property
    def c_ref(self) -> np.ndarray: ...

    def expand(self, y: object) -> np.ndarray: ...

    def expand_direction(self, dy: object) -> np.ndarray: ...

    def reduce_tangent(self, dc: object) -> np.ndarray: ...

    def project_tangent(self, dc: object) -> np.ndarray: ...

    def project_affine(self, source: object) -> np.ndarray: ...

    def reduce(self, source: object) -> np.ndarray: ...

    def lift_reduced_cotangent(self, ybar: object) -> np.ndarray: ...

    def reduce_source_cotangent(self, cbar: object) -> np.ndarray: ...

    def metadata(self) -> dict[str, object]: ...

    def metadata_hash(self) -> str: ...


@dataclass(frozen=True, slots=True)
class AffineChargeCoordinates:
    """Matrix-free coordinates for ``c = c_ref + T y`` with dimensionless ``y``."""

    atom_count: int
    total_charge: float
    monopole_scale: float = 1.0
    dipole_scale: float = 1.0
    source_space: SourceSpace = field(default_factory=lambda: ATOMIC_L1_SOURCE_SPACE)
    component_scales: tuple[float, ...] | None = None
    dense_debug_limit: int = 64

    def __post_init__(self) -> None:
        _atom_count(self.atom_count)
        if not np.isfinite(self.total_charge):
            raise ValueError("total_charge must be finite.")
        if not isinstance(self.source_space, SourceSpace):
            raise TypeError("source_space must be a SourceSpace.")
        for name in ("monopole_scale", "dipole_scale"):
            scale = getattr(self, name)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            isinstance(self.dense_debug_limit, bool)
            or not isinstance(self.dense_debug_limit, int)
            or self.dense_debug_limit < 1
        ):
            raise ValueError("dense_debug_limit must be a positive integer.")
        if self.component_scales is None:
            scales = tuple(
                (
                    self.monopole_scale
                    if index == self.source_space.charge_component
                    else self.dipole_scale
                )
                for index in range(self.source_space.component_count)
            )
        else:
            scales = tuple(self.component_scales)
        if len(scales) != self.source_space.component_count:
            raise ValueError(
                "component_scales must match source-space component count."
            )
        if any(not np.isfinite(scale) or scale <= 0.0 for scale in scales):
            raise ValueError("all component scales must be finite and positive.")
        object.__setattr__(self, "component_scales", scales)

    @property
    def source_dimension(self) -> int:
        return self.source_space.component_count * self.atom_count

    @property
    def reduced_dimension(self) -> int:
        return self.source_dimension - 1

    @property
    def c_ref(self) -> np.ndarray:
        result = np.zeros(self.source_space.shape(self.atom_count))
        result[:, self.source_space.charge_component] = (
            self.total_charge / self.atom_count
        )
        result.setflags(write=False)
        return result

    def _reduced(self, values: object, *, name: str) -> np.ndarray:
        return _validated(values, shape=(self.reduced_dimension,), name=name)

    def _source(self, values: object, *, name: str) -> np.ndarray:
        return self.source_space.validate(values, atom_count=self.atom_count, name=name)

    def _helmert_apply(self, values: np.ndarray) -> np.ndarray:
        """Apply the N by N-1 orthonormal Helmert basis in O(N)."""

        if self.atom_count == 1:
            return np.zeros(1)
        indices = np.arange(1, self.atom_count, dtype=float)
        scaled = values / np.sqrt(indices * (indices + 1.0))
        suffix = np.cumsum(scaled[::-1])[::-1]
        result = np.empty(self.atom_count)
        result[0] = suffix[0]
        result[1:] = np.r_[suffix[1:], 0.0] - indices * scaled
        return result

    def _helmert_transpose(self, values: np.ndarray) -> np.ndarray:
        """Apply the transpose of the Helmert basis in O(N)."""

        if self.atom_count == 1:
            return np.empty(0)
        indices = np.arange(1, self.atom_count, dtype=float)
        prefix = np.cumsum(values)[:-1]
        return (prefix - indices * values[1:]) / np.sqrt(indices * (indices + 1.0))

    def _split_reduced(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        charge_size = self.atom_count - 1
        return values[:charge_size], values[charge_size:]

    def _apply_T(self, values: np.ndarray) -> np.ndarray:
        charge_values, other_values = self._split_reduced(values)
        result = np.zeros(self.source_space.shape(self.atom_count))
        charge = self.source_space.charge_component
        result[:, charge] = self.component_scales[charge] * self._helmert_apply(
            charge_values
        )
        cursor = 0
        for component in range(self.source_space.component_count):
            if component == charge:
                continue
            result[:, component] = (
                self.component_scales[component]
                * other_values[cursor : cursor + self.atom_count]
            )
            cursor += self.atom_count
        return result

    def _apply_T_plus(self, values: np.ndarray) -> np.ndarray:
        charge = self.source_space.charge_component
        parts = [
            self._helmert_transpose(values[:, charge]) / self.component_scales[charge]
        ]
        parts.extend(
            values[:, component] / self.component_scales[component]
            for component in range(self.source_space.component_count)
            if component != charge
        )
        return np.concatenate(parts)

    def expand(self, y: object) -> np.ndarray:
        return self.c_ref + self._apply_T(self._reduced(y, name="reduced coordinates"))

    def expand_direction(self, dy: object) -> np.ndarray:
        return self._apply_T(self._reduced(dy, name="reduced direction"))

    def reduce_tangent(self, dc: object) -> np.ndarray:
        return self._apply_T_plus(self._source(dc, name="source tangent"))

    def project_tangent(self, dc: object) -> np.ndarray:
        values = self._source(dc, name="source tangent")
        result = values.copy()
        charge = self.source_space.charge_component
        result[:, charge] -= np.mean(result[:, charge])
        return result

    def project_affine(self, source: object) -> np.ndarray:
        values = self._source(source, name="affine source candidate")
        return self.c_ref + self.project_tangent(values - self.c_ref)

    def reduce(self, source: object) -> np.ndarray:
        values = self._source(source, name="affine source")
        return self.reduce_tangent(values - self.c_ref)

    def lift_reduced_cotangent(self, ybar: object) -> np.ndarray:
        charge_values, other_values = self._split_reduced(
            self._reduced(ybar, name="reduced cotangent")
        )
        result = np.zeros(self.source_space.shape(self.atom_count))
        charge = self.source_space.charge_component
        result[:, charge] = (
            self._helmert_apply(charge_values) / self.component_scales[charge]
        )
        cursor = 0
        for component in range(self.source_space.component_count):
            if component == charge:
                continue
            result[:, component] = (
                other_values[cursor : cursor + self.atom_count]
                / self.component_scales[component]
            )
            cursor += self.atom_count
        return result

    def reduce_source_cotangent(self, cbar: object) -> np.ndarray:
        values = self._source(cbar, name="source cotangent")
        charge = self.source_space.charge_component
        parts = [
            self.component_scales[charge] * self._helmert_transpose(values[:, charge])
        ]
        parts.extend(
            self.component_scales[component] * values[:, component]
            for component in range(self.source_space.component_count)
            if component != charge
        )
        return np.concatenate(parts)

    def metadata(self) -> dict[str, object]:
        """Return the complete, JSON-serializable coordinate identity."""

        return {
            "coordinate_contract": "affine-charge-coordinates-matrix-free-v1",
            "atom_count": self.atom_count,
            "total_charge": self.total_charge,
            "source_space": self.source_space.metadata(),
            "component_scales": list(self.component_scales),
        }

    def metadata_hash(self) -> str:
        encoded = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _guard_dense_debug(self) -> None:
        if self.source_dimension > self.dense_debug_limit:
            raise ValueError(
                "dense coordinate matrices are debug-only for source_dimension "
                f"<= {self.dense_debug_limit}."
            )

    @property
    def T(self) -> np.ndarray:
        self._guard_dense_debug()
        result = np.column_stack(
            [
                self.expand_direction(np.eye(self.reduced_dimension)[column]).reshape(
                    -1
                )
                for column in range(self.reduced_dimension)
            ]
        )
        result.setflags(write=False)
        return result

    @property
    def T_plus(self) -> np.ndarray:
        self._guard_dense_debug()
        shape = self.source_space.shape(self.atom_count)
        result = np.column_stack(
            [
                self.reduce_tangent(
                    np.eye(self.source_dimension)[column].reshape(shape)
                )
                for column in range(self.source_dimension)
            ]
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class LinearChargeCoordinates:
    """Matrix-free coordinates for an arbitrary linear charge functional.

    Let ``a`` be the atom-major repetition of
    :attr:`SourceSpace.effective_charge_weights`, and let ``S`` be the
    diagonal component scaling.  A symmetric Householder reflector ``H`` is
    chosen so that ``H e0 = normalize(S a)``.  The maps are then

    ``T = S H[:, 1:]`` and ``T_plus = H[1:, :] S^-1``.

    Consequently ``a.T @ T = 0`` and ``T_plus @ T = I`` without materializing
    either matrix.  This is the coordinate system needed when physical total
    charge is carried by more than one radial source channel.
    """

    atom_count: int
    total_charge: float
    source_space: SourceSpace
    component_scales: tuple[float, ...] | None = None
    dense_debug_limit: int = 64

    def __post_init__(self) -> None:
        _atom_count(self.atom_count)
        if not np.isfinite(self.total_charge):
            raise ValueError("total_charge must be finite.")
        if not isinstance(self.source_space, SourceSpace):
            raise TypeError("source_space must be a SourceSpace.")
        if (
            isinstance(self.dense_debug_limit, bool)
            or not isinstance(self.dense_debug_limit, int)
            or self.dense_debug_limit < 1
        ):
            raise ValueError("dense_debug_limit must be a positive integer.")
        scales = (
            (1.0,) * self.source_space.component_count
            if self.component_scales is None
            else tuple(float(value) for value in self.component_scales)
        )
        if len(scales) != self.source_space.component_count:
            raise ValueError(
                "component_scales must match source-space component count."
            )
        if any(not np.isfinite(scale) or scale <= 0.0 for scale in scales):
            raise ValueError("all component scales must be finite and positive.")
        object.__setattr__(self, "component_scales", scales)

    @property
    def source_dimension(self) -> int:
        return self.atom_count * self.source_space.component_count

    @property
    def reduced_dimension(self) -> int:
        return self.source_dimension - 1

    @property
    def _flat_scales(self) -> np.ndarray:
        return np.tile(np.asarray(self.component_scales, dtype=float), self.atom_count)

    @property
    def _flat_charge_weights(self) -> np.ndarray:
        return np.tile(
            np.asarray(self.source_space.effective_charge_weights, dtype=float),
            self.atom_count,
        )

    @property
    def _scaled_constraint(self) -> np.ndarray:
        return self._flat_scales * self._flat_charge_weights

    def _householder_apply(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.source_dimension,):
            raise ValueError(
                "Householder input must match the flattened source dimension."
            )
        scaled_constraint = self._scaled_constraint
        direction = scaled_constraint / np.linalg.norm(scaled_constraint)
        axis = np.zeros(self.source_dimension)
        axis[0] = 1.0
        reflector = axis - direction
        reflector_norm = np.linalg.norm(reflector)
        if reflector_norm <= 16.0 * np.finfo(float).eps:
            return np.array(values, copy=True)
        reflector /= reflector_norm
        return values - 2.0 * reflector * np.dot(reflector, values)

    @property
    def c_ref(self) -> np.ndarray:
        scaled_constraint = self._scaled_constraint
        flattened = self._flat_scales * (
            self.total_charge
            * scaled_constraint
            / np.dot(scaled_constraint, scaled_constraint)
        )
        result = flattened.reshape(self.source_space.shape(self.atom_count))
        result.setflags(write=False)
        return result

    def _reduced(self, values: object, *, name: str) -> np.ndarray:
        return _validated(values, shape=(self.reduced_dimension,), name=name)

    def _source(self, values: object, *, name: str) -> np.ndarray:
        return self.source_space.validate(values, atom_count=self.atom_count, name=name)

    def _apply_T(self, values: np.ndarray) -> np.ndarray:
        embedded = np.empty(self.source_dimension)
        embedded[0] = 0.0
        embedded[1:] = values
        flattened = self._flat_scales * self._householder_apply(embedded)
        return flattened.reshape(self.source_space.shape(self.atom_count))

    def _apply_T_plus(self, values: np.ndarray) -> np.ndarray:
        flattened = values.reshape(-1) / self._flat_scales
        return self._householder_apply(flattened)[1:]

    def expand(self, y: object) -> np.ndarray:
        return self.c_ref + self._apply_T(self._reduced(y, name="reduced coordinates"))

    def expand_direction(self, dy: object) -> np.ndarray:
        return self._apply_T(self._reduced(dy, name="reduced direction"))

    def reduce_tangent(self, dc: object) -> np.ndarray:
        return self._apply_T_plus(self._source(dc, name="source tangent"))

    def project_tangent(self, dc: object) -> np.ndarray:
        values = self._source(dc, name="source tangent")
        return self._apply_T(self._apply_T_plus(values))

    def project_affine(self, source: object) -> np.ndarray:
        values = self._source(source, name="affine source candidate")
        return self.c_ref + self.project_tangent(values - self.c_ref)

    def reduce(self, source: object) -> np.ndarray:
        values = self._source(source, name="affine source")
        return self.reduce_tangent(values - self.c_ref)

    def lift_reduced_cotangent(self, ybar: object) -> np.ndarray:
        embedded = np.empty(self.source_dimension)
        embedded[0] = 0.0
        embedded[1:] = self._reduced(ybar, name="reduced cotangent")
        flattened = self._householder_apply(embedded) / self._flat_scales
        return flattened.reshape(self.source_space.shape(self.atom_count))

    def reduce_source_cotangent(self, cbar: object) -> np.ndarray:
        values = self._source(cbar, name="source cotangent")
        flattened = self._flat_scales * values.reshape(-1)
        return self._householder_apply(flattened)[1:]

    def metadata(self) -> dict[str, object]:
        return {
            "coordinate_contract": "linear-charge-coordinates-householder-v1",
            "atom_count": self.atom_count,
            "total_charge": self.total_charge,
            "source_space": self.source_space.metadata(),
            "component_scales": list(self.component_scales),
        }

    def metadata_hash(self) -> str:
        encoded = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _guard_dense_debug(self) -> None:
        if self.source_dimension > self.dense_debug_limit:
            raise ValueError(
                "dense coordinate matrices are debug-only for source_dimension "
                f"<= {self.dense_debug_limit}."
            )

    @property
    def T(self) -> np.ndarray:
        self._guard_dense_debug()
        eye = np.eye(self.reduced_dimension)
        result = np.column_stack(
            [
                self.expand_direction(eye[column]).reshape(-1)
                for column in range(self.reduced_dimension)
            ]
        )
        result.setflags(write=False)
        return result

    @property
    def T_plus(self) -> np.ndarray:
        self._guard_dense_debug()
        shape = self.source_space.shape(self.atom_count)
        eye = np.eye(self.source_dimension)
        result = np.column_stack(
            [
                self.reduce_tangent(eye[column].reshape(shape))
                for column in range(self.source_dimension)
            ]
        )
        result.setflags(write=False)
        return result


ATOMIC_L1_SOURCE_SPACE = SourceSpace(
    scalar_id="maple.route2.atomic-l1-source-space.v1",
    representation="atom-centred net monopole plus checkpoint-native real-spherical l=1",
    components=ATOMIC_L1_PAIRING.source_components,
    units=ATOMIC_L1_PAIRING.source_units,
)
ATOMIC_L1_FIELD_DUAL_SPACE = FieldDualSpace(
    scalar_id="maple.route2.atomic-l1-field-dual-space.v1",
    representation="atom-centred energy-dual potential and Cartesian potential gradient",
    components=ATOMIC_L1_PAIRING.field_components,
    units=ATOMIC_L1_PAIRING.field_units,
    source_space=ATOMIC_L1_SOURCE_SPACE,
    pairing_metric=ATOMIC_L1_PAIRING,
)
MACE_POLAR_RADIAL_GTO_SOURCE_SPACE = SourceSpace(
    scalar_id="maple.route2.mace-polar-radial-gto-source-space.v1",
    representation=(
        "atom-centred physical unit multipoles in the official MACE-POLAR "
        "sigma=1.5 and 3.0 angstrom radial GTO channels"
    ),
    components=MACE_POLAR_RADIAL_GTO_PAIRING.source_components,
    units=MACE_POLAR_RADIAL_GTO_PAIRING.source_units,
    charge_component=0,
    charge_weights=(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE = FieldDualSpace(
    scalar_id="maple.route2.mace-polar-radial-gto-field-dual-space.v1",
    representation=(
        "physical Gaussian-smoothed potentials and Cartesian gradients for "
        "the official MACE-POLAR sigma=1.5 and 3.0 angstrom receiver channels"
    ),
    components=MACE_POLAR_RADIAL_GTO_PAIRING.field_components,
    units=MACE_POLAR_RADIAL_GTO_PAIRING.field_units,
    source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
    pairing_metric=MACE_POLAR_RADIAL_GTO_PAIRING,
)


@dataclass(frozen=True, slots=True)
class ReducedCoordinateContract:
    """Profile-bound dimensionless coordinate scaling and implementation."""

    contract_id: str
    source_space: SourceSpace
    implementation_contract: str | None
    component_scales: tuple[float, ...] | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contract_id", _nonempty(self.contract_id, name="contract_id")
        )
        if not isinstance(self.source_space, SourceSpace):
            raise TypeError("source_space must be a SourceSpace.")
        if self.implementation_contract is not None:
            object.__setattr__(
                self,
                "implementation_contract",
                _nonempty(self.implementation_contract, name="implementation_contract"),
            )
        if self.component_scales is not None:
            scales = tuple(float(value) for value in self.component_scales)
            if len(scales) != self.source_space.component_count:
                raise ValueError(
                    "coordinate component scales must match the source component count."
                )
            if any(not np.isfinite(value) or value <= 0.0 for value in scales):
                raise ValueError(
                    "coordinate component scales must be finite and positive."
                )
            object.__setattr__(self, "component_scales", scales)

    def validate(self, coordinates: ReducedCoordinates) -> None:
        if not isinstance(
            coordinates, (AffineChargeCoordinates, LinearChargeCoordinates)
        ):
            raise TypeError(
                "coordinates must be a vetted affine or linear charge implementation."
            )
        if (
            coordinates.source_space.metadata_hash()
            != self.source_space.metadata_hash()
        ):
            raise ValueError(
                "Coordinate source identity does not match the exact authoritative "
                "coordinate contract."
            )
        metadata = coordinates.metadata()
        if (
            self.implementation_contract is not None
            and metadata.get("coordinate_contract") != self.implementation_contract
        ):
            raise ValueError(
                "Coordinate implementation does not match the profile contract."
            )
        if (
            self.component_scales is not None
            and tuple(coordinates.component_scales) != self.component_scales
        ):
            raise ValueError(
                "Dimensionless coordinate scales do not match the profile contract."
            )

    def build(self, atom_count: int, total_charge: float) -> ReducedCoordinates:
        if self.implementation_contract == "linear-charge-coordinates-householder-v1":
            if self.component_scales is None:  # guarded by registry construction
                raise RuntimeError(
                    "Linear coordinate contract omitted component scales."
                )
            return LinearChargeCoordinates(
                atom_count=atom_count,
                total_charge=total_charge,
                source_space=self.source_space,
                component_scales=self.component_scales,
            )
        raise ValueError(
            f"Coordinate contract {self.contract_id!r} is validation-only and cannot build."
        )

    def metadata(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "source_space_sha256": self.source_space.metadata_hash(),
            "implementation_contract": self.implementation_contract,
            "component_scales": (
                None if self.component_scales is None else list(self.component_scales)
            ),
        }

    def metadata_hash(self) -> str:
        encoded = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


LEGACY_UNBOUND_COORDINATE_CONTRACT = ReducedCoordinateContract(
    contract_id=LEGACY_UNBOUND_COORDINATE_CONTRACT_ID,
    source_space=ATOMIC_L1_SOURCE_SPACE,
    implementation_contract=None,
    component_scales=None,
)
MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT = ReducedCoordinateContract(
    contract_id=MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT_ID,
    source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
    implementation_contract="linear-charge-coordinates-householder-v1",
    # One unit in each component's declared native unit: e for monopoles and
    # e*angstrom for dipoles.  This fixes dimensionless residual/Krylov norms.
    component_scales=(1.0,) * 8,
)
COORDINATE_CONTRACT_REGISTRY = MappingProxyType(
    {
        contract.contract_id: contract
        for contract in (
            LEGACY_UNBOUND_COORDINATE_CONTRACT,
            MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT,
        )
    }
)

SOURCE_SPACE_REGISTRY = MappingProxyType(
    {
        space.scalar_id: space
        for space in (
            ATOMIC_L1_SOURCE_SPACE,
            MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
        )
    }
)
FIELD_DUAL_SPACE_REGISTRY = MappingProxyType(
    {
        space.scalar_id: space
        for space in (
            ATOMIC_L1_FIELD_DUAL_SPACE,
            MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
        )
    }
)


def get_source_space(space_id: str) -> SourceSpace:
    try:
        return SOURCE_SPACE_REGISTRY[space_id]
    except KeyError as exc:
        raise KeyError(f"Unregistered Route-2 source space: {space_id!r}.") from exc


def get_field_dual_space(space_id: str) -> FieldDualSpace:
    try:
        return FIELD_DUAL_SPACE_REGISTRY[space_id]
    except KeyError as exc:
        raise KeyError(f"Unregistered Route-2 field-dual space: {space_id!r}.") from exc


def get_coordinate_contract(contract_id: str) -> ReducedCoordinateContract:
    try:
        return COORDINATE_CONTRACT_REGISTRY[contract_id]
    except KeyError as exc:
        raise KeyError(
            f"Unregistered Route-2 coordinate contract: {contract_id!r}."
        ) from exc


AtomicL1SourceSpace = SourceSpace
AtomicL1FieldDualSpace = FieldDualSpace
ChargeConstrainedCoordinates = AffineChargeCoordinates

__all__ = [
    "ATOMIC_L1_FIELD_DUAL_SPACE",
    "ATOMIC_L1_SOURCE_SPACE",
    "AffineChargeCoordinates",
    "AtomicL1FieldDualSpace",
    "AtomicL1SourceSpace",
    "ChargeConstrainedCoordinates",
    "COORDINATE_CONTRACT_REGISTRY",
    "FIELD_DUAL_SPACE_REGISTRY",
    "FieldDualSpace",
    "LinearChargeCoordinates",
    "LEGACY_UNBOUND_COORDINATE_CONTRACT",
    "MACE_POLAR_RADIAL_GTO_COORDINATE_CONTRACT",
    "MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE",
    "MACE_POLAR_RADIAL_GTO_SOURCE_SPACE",
    "ReducedCoordinates",
    "ReducedCoordinateContract",
    "SOURCE_SPACE_REGISTRY",
    "SourceSpace",
    "get_field_dual_space",
    "get_coordinate_contract",
    "get_source_space",
]
