"""Provider-independent reduced Route-2 state equation.

The public residual is exactly

``T_plus [c - Pi_q M(R, P_R(c))]`` with ``c = c_ref + T y``.

The reduced coordinates ``y`` are dimensionless by contract.  Providers own
their native implementations; this module contains no model-family branches.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import marshal
import re
import types
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.state_registry import OPERATIONAL_STATE_EQUATION_ID

from .spaces import AffineChargeCoordinates, FieldDualSpace, SourceSpace

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _stable_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty stable identifier.")
    return value.strip()


def _provenance_sha256(value: object, name: str) -> str:
    digest = _stable_id(value, name).lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return digest


def geometry_sha256(geometry: Any) -> str:
    """Hash a finite numeric geometry or an ASE-compatible geometry identity."""

    if all(
        callable(getattr(geometry, name, None))
        for name in ("get_atomic_numbers", "get_positions", "get_cell", "get_pbc")
    ):
        numbers = np.asarray(geometry.get_atomic_numbers())
        positions = np.asarray(geometry.get_positions(), dtype=np.float64)
        cell = np.asarray(geometry.get_cell(), dtype=np.float64)
        pbc = np.asarray(geometry.get_pbc(), dtype=bool)
        if (
            numbers.ndim != 1
            or numbers.size < 1
            or not np.issubdtype(numbers.dtype, np.integer)
        ):
            raise ValueError(
                "ASE geometry atomic numbers must be a non-empty integer vector."
            )
        numbers = numbers.astype(np.int64, copy=False)
        if np.any(numbers <= 0):
            raise ValueError("ASE geometry atomic numbers must be positive.")
        if positions.shape != (numbers.size, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                "ASE geometry positions must be finite with shape (atom_count, 3)."
            )
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError("ASE geometry cell must be finite with shape (3, 3).")
        if pbc.shape != (3,):
            raise ValueError("ASE geometry pbc must have shape (3,).")
        payload = {
            "kind": "ase-geometry-v1",
            "atomic_numbers": numbers.tolist(),
            "positions_A": positions.tolist(),
            "cell_A": cell.tolist(),
            "pbc": pbc.tolist(),
        }
    else:
        values = np.asarray(geometry, dtype=np.float64)
        if values.ndim == 0 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Numeric geometry must be a non-empty finite array.")
        payload = {
            "kind": "numeric-geometry-v1",
            "shape": list(values.shape),
            "values": values.tolist(),
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_vector(values: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must be finite with shape ({size},); received {result.shape}."
        )
    return np.array(result, copy=True)


def provider_behavior_sha256(
    provider: object,
    method_names: tuple[str, ...],
    *,
    label: str,
) -> str:
    """Hash class-defined provider callables and reject instance shadows.

    A configuration digest binds numerical parameters, but cannot detect a
    derivative method replaced while retaining the old digest.  Operational
    providers therefore expose behavior through class descriptors only.  This
    runtime implementation seal is recomputed at every equation/scalar entry.
    """

    instance_attributes = getattr(provider, "__dict__", {})

    def dependency_digest(value: object, stack: frozenset[int] = frozenset()) -> str:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            header = json.dumps(
                {"dtype": array.dtype.str, "shape": list(array.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()
        if isinstance(value, np.generic):
            return dependency_digest(value.item(), stack)
        if isinstance(value, (str, int, float, bool, type(None))):
            return hashlib.sha256(repr(value).encode()).hexdigest()
        identity = id(value)
        if identity in stack:
            cycle = f"cycle:{type(value).__module__}.{type(value).__qualname__}"
            return hashlib.sha256(cycle.encode()).hexdigest()
        next_stack = stack | {identity}
        if isinstance(value, dict):
            parts = tuple(
                (str(key), dependency_digest(item, next_stack))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
            return hashlib.sha256(repr(parts).encode()).hexdigest()
        if isinstance(value, (tuple, list)):
            parts = tuple(dependency_digest(item, next_stack) for item in value)
            return hashlib.sha256(repr(parts).encode()).hexdigest()
        if isinstance(value, (frozenset, set)):
            parts = tuple(sorted(dependency_digest(item, next_stack) for item in value))
            return hashlib.sha256(repr(parts).encode()).hexdigest()
        if isinstance(value, types.FunctionType):
            closure = tuple(
                dependency_digest(cell.cell_contents, next_stack)
                for cell in (value.__closure__ or ())
            )
            defaults = dependency_digest(value.__defaults__ or (), next_stack)
            keyword_defaults = dependency_digest(
                tuple(sorted((value.__kwdefaults__ or {}).items())), next_stack
            )
            referenced_globals: tuple[tuple[str, str], ...] = ()
            global_namespace = value.__globals__
            dependencies: list[tuple[str, str]] = []
            for name in sorted(set(value.__code__.co_names)):
                if name not in global_namespace:
                    continue
                dependency = global_namespace[name]
                if isinstance(dependency, (types.ModuleType, type, np.ufunc)):
                    continue
                dependencies.append((name, dependency_digest(dependency, next_stack)))
            referenced_globals = tuple(dependencies)
            payload = (
                value.__module__,
                value.__qualname__,
                hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
                defaults,
                keyword_defaults,
                closure,
                referenced_globals,
            )
            return hashlib.sha256(repr(payload).encode()).hexdigest()
        return hashlib.sha256(
            f"{type(value).__module__}.{type(value).__qualname__}".encode()
        ).hexdigest()

    payload: list[dict[str, object]] = []
    for name in method_names:
        if name in instance_attributes:
            raise TypeError(
                f"{label}.{name} must be a class-defined method; "
                "instance-level callable rebinding is forbidden."
            )
        owner = next(
            (cls for cls in type(provider).__mro__ if name in vars(cls)),
            None,
        )
        if owner is None:
            raise TypeError(f"{label}.{name} must be a class-defined callable.")
        descriptor = inspect.getattr_static(owner, name)
        implementation = (
            descriptor.__func__
            if isinstance(descriptor, (staticmethod, classmethod))
            else descriptor
        )
        if not callable(getattr(provider, name, None)) or not callable(implementation):
            raise TypeError(f"{label}.{name} must be callable.")
        code = getattr(implementation, "__code__", None)
        code_digest = hashlib.sha256(
            marshal.dumps(code)
            if code is not None
            else repr(type(implementation)).encode("utf-8")
        ).hexdigest()
        global_dependencies: dict[str, str] = {}
        if code is not None:
            global_namespace = getattr(implementation, "__globals__", {})
            for dependency_name in sorted(set(code.co_names)):
                if dependency_name not in global_namespace:
                    continue
                dependency = global_namespace[dependency_name]
                if isinstance(
                    dependency,
                    (types.ModuleType, type, np.ufunc),
                ):
                    continue
                global_dependencies[dependency_name] = dependency_digest(dependency)
        payload.append(
            {
                "method": name,
                "owner_module": owner.__module__,
                "owner_qualname": owner.__qualname__,
                "implementation_module": getattr(implementation, "__module__", ""),
                "implementation_qualname": getattr(
                    implementation, "__qualname__", type(implementation).__qualname__
                ),
                "code_sha256": code_digest,
                "defaults_sha256": dependency_digest(
                    getattr(implementation, "__defaults__", ()) or ()
                ),
                "keyword_defaults_sha256": dependency_digest(
                    tuple(
                        sorted(
                            (
                                getattr(implementation, "__kwdefaults__", {}) or {}
                            ).items()
                        )
                    )
                ),
                "closure_sha256": dependency_digest(
                    tuple(
                        cell.cell_contents
                        for cell in (getattr(implementation, "__closure__", ()) or ())
                    )
                ),
                "referenced_globals": global_dependencies,
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class ElectronicResponseProvider(Protocol):
    """Electronic source map and its exact local derivatives."""

    provider_id: str
    model_profile_id: str
    coupling_id: str
    provenance_sha256: str
    source_space: SourceSpace
    field_space: FieldDualSpace

    def configuration_sha256(self) -> str: ...

    def evaluate_source(self, geometry: Any, field: np.ndarray) -> np.ndarray: ...
    def field_jvp(
        self, geometry: Any, field: np.ndarray, field_direction: np.ndarray
    ) -> np.ndarray: ...
    def field_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray: ...
    def coordinate_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray: ...


@runtime_checkable
class ContinuumResponseProvider(Protocol):
    """Reaction-field map and its exact local derivatives."""

    provider_id: str
    continuum_profile_id: str
    cavity_profile_id: str
    coupling_id: str
    provenance_sha256: str
    source_space: SourceSpace
    field_space: FieldDualSpace

    def configuration_sha256(self) -> str: ...

    def evaluate_field(self, geometry: Any, source: np.ndarray) -> np.ndarray: ...
    def source_jvp(
        self, geometry: Any, source: np.ndarray, source_direction: np.ndarray
    ) -> np.ndarray: ...
    def source_vjp(
        self, geometry: Any, source: np.ndarray, field_cotangent: np.ndarray
    ) -> np.ndarray: ...
    def coordinate_vjp(
        self, geometry: Any, source: np.ndarray, field_cotangent: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class StateEvaluation:
    """Immutable snapshot of one residual evaluation."""

    y: tuple[float, ...]
    source: tuple[tuple[float, ...], ...]
    field: tuple[tuple[float, ...], ...]
    response_source: tuple[tuple[float, ...], ...]
    projected_source: tuple[tuple[float, ...], ...]
    residual: tuple[float, ...]

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residual))


@dataclass(frozen=True)
class ReducedStateEquation:
    """Charge-constrained mutual-polarization residual in dimensionless ``y``.

    Phase 3 uses geometry-independent affine coordinates and a static pairing
    metric.  Coordinate partials here therefore include provider maps only.  A
    future geometry-dependent ``c_ref``, ``T``, or ``Q`` must extend the
    coordinate/pairing protocols with their own coordinate pullbacks before it
    can use this implementation.
    """

    coordinates: AffineChargeCoordinates
    electronic: ElectronicResponseProvider
    continuum: ContinuumResponseProvider
    state_equation_id: str = OPERATIONAL_STATE_EQUATION_ID
    _construction_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, AffineChargeCoordinates):
            raise TypeError("coordinates must be AffineChargeCoordinates.")
        if (
            not isinstance(self.state_equation_id, str)
            or not self.state_equation_id.strip()
        ):
            raise ValueError("state_equation_id must be non-empty.")
        coordinate_source = self.coordinates.source_space
        for label, provider in (
            ("electronic", self.electronic),
            ("continuum", self.continuum),
        ):
            _stable_id(getattr(provider, "provider_id", None), f"{label}.provider_id")
            compatibility_name = (
                "model_profile_id" if label == "electronic" else "continuum_profile_id"
            )
            _stable_id(
                getattr(provider, compatibility_name, None),
                f"{label}.{compatibility_name}",
            )
            _provenance_sha256(
                getattr(provider, "provenance_sha256", None),
                f"{label}.provenance_sha256",
            )
            _stable_id(getattr(provider, "coupling_id", None), f"{label}.coupling_id")
            configuration = getattr(provider, "configuration_sha256", None)
            if not callable(configuration):
                raise TypeError(f"{label}.configuration_sha256 must be callable.")
            _provenance_sha256(configuration(), f"{label} configuration fingerprint")
            if label == "continuum":
                _stable_id(
                    getattr(provider, "cavity_profile_id", None),
                    "continuum.cavity_profile_id",
                )
            source_space = getattr(provider, "source_space", None)
            field_space = getattr(provider, "field_space", None)
            if not isinstance(source_space, SourceSpace):
                raise TypeError(f"{label}.source_space must be SourceSpace.")
            if not isinstance(field_space, FieldDualSpace):
                raise TypeError(f"{label}.field_space must be FieldDualSpace.")
            if source_space.metadata_hash() != coordinate_source.metadata_hash():
                raise ValueError(
                    f"{label}.source_space does not exactly match coordinate source identity."
                )
            if (
                field_space.source_space.metadata_hash()
                != coordinate_source.metadata_hash()
            ):
                raise ValueError(
                    f"{label}.field_space is not paired with the coordinate source identity."
                )
        if (
            self.electronic.field_space.metadata_hash()
            != self.continuum.field_space.metadata_hash()
        ):
            raise ValueError(
                "Electronic and continuum field-space identities must match exactly."
            )
        if self.electronic.coupling_id != self.continuum.coupling_id:
            raise ValueError(
                "Electronic and continuum coupling identities must match exactly."
            )
        object.__setattr__(
            self, "_construction_fingerprint", self._current_fingerprint_sha256()
        )

    @property
    def source_space(self) -> SourceSpace:
        return self.coordinates.source_space

    @property
    def field_space(self) -> FieldDualSpace:
        return self.electronic.field_space

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "state_equation_id": self.state_equation_id,
            "coordinates_sha256": self.coordinates.metadata_hash(),
            "source_space_sha256": self.source_space.metadata_hash(),
            "field_space_sha256": self.field_space.metadata_hash(),
            "pairing_sha256": self.field_space.pairing_metric.metadata_hash(),
            "electronic": {
                "provider_id": self.electronic.provider_id,
                "model_profile_id": self.electronic.model_profile_id,
                "coupling_id": self.electronic.coupling_id,
                "provenance_sha256": self.electronic.provenance_sha256,
                "configuration_sha256": self.electronic.configuration_sha256(),
                "behavior_sha256": provider_behavior_sha256(
                    self.electronic,
                    (
                        "configuration_sha256",
                        "evaluate_source",
                        "field_jvp",
                        "field_vjp",
                        "coordinate_vjp",
                    ),
                    label="electronic",
                ),
            },
            "continuum": {
                "provider_id": self.continuum.provider_id,
                "continuum_profile_id": self.continuum.continuum_profile_id,
                "cavity_profile_id": self.continuum.cavity_profile_id,
                "coupling_id": self.continuum.coupling_id,
                "provenance_sha256": self.continuum.provenance_sha256,
                "configuration_sha256": self.continuum.configuration_sha256(),
                "behavior_sha256": provider_behavior_sha256(
                    self.continuum,
                    (
                        "configuration_sha256",
                        "evaluate_field",
                        "source_jvp",
                        "source_vjp",
                        "coordinate_vjp",
                    ),
                    label="continuum",
                ),
            },
        }

    def _current_fingerprint_sha256(self) -> str:
        encoded = json.dumps(
            self.fingerprint_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fingerprint_sha256(self) -> str:
        current = self._current_fingerprint_sha256()
        if self._construction_fingerprint and current != self._construction_fingerprint:
            raise ValueError(
                "State-equation provider configuration drifted after construction."
            )
        return current

    @property
    def reduced_dimension(self) -> int:
        return self.coordinates.reduced_dimension

    def _y(
        self, y: object, name: str = "dimensionless reduced coordinates"
    ) -> np.ndarray:
        return _finite_vector(y, self.reduced_dimension, name)

    def _source(self, values: object, name: str) -> np.ndarray:
        source = np.asarray(values, dtype=float)
        expected = self.coordinates.source_space.shape(self.coordinates.atom_count)
        if source.shape != expected or not np.all(np.isfinite(source)):
            raise ValueError(
                f"{name} must be finite with shape {expected}; received {source.shape}."
            )
        return np.array(source, copy=True)

    def evaluate(self, geometry: Any, y: object) -> StateEvaluation:
        self.fingerprint_sha256()
        reduced = self._y(y)
        source = self.coordinates.expand(reduced)
        field = self._source(
            self.continuum.evaluate_field(geometry, source), "continuum field"
        )
        response = self._source(
            self.electronic.evaluate_source(geometry, field),
            "electronic response source",
        )
        projected = self.coordinates.project_affine(response)
        # This spelling deliberately follows the authoritative equation rather
        # than simplifying to y - reduce(projected).
        residual = self.coordinates.reduce_tangent(source - projected)
        return StateEvaluation(
            y=tuple(float(value) for value in reduced),
            source=tuple(tuple(float(value) for value in row) for row in source),
            field=tuple(tuple(float(value) for value in row) for row in field),
            response_source=tuple(
                tuple(float(value) for value in row) for row in response
            ),
            projected_source=tuple(
                tuple(float(value) for value in row) for row in projected
            ),
            residual=tuple(float(value) for value in residual),
        )

    def residual(self, geometry: Any, y: object) -> np.ndarray:
        result = np.asarray(self.evaluate(geometry, y).residual, dtype=float)
        result.setflags(write=False)
        return result

    def fixed_point_map(self, geometry: Any, y: object) -> np.ndarray:
        evaluation = self.evaluate(geometry, y)
        result = self.coordinates.reduce(np.asarray(evaluation.projected_source))
        result.setflags(write=False)
        return result

    def jvp(self, geometry: Any, y: object, dy: object) -> np.ndarray:
        self.fingerprint_sha256()
        evaluation = self.evaluate(geometry, y)
        direction = self._y(dy, "reduced direction")
        source = np.asarray(evaluation.source)
        field = np.asarray(evaluation.field)
        dc = self.coordinates.expand_direction(direction)
        du = self._source(
            self.continuum.source_jvp(geometry, source, dc), "continuum field JVP"
        )
        dm = self._source(
            self.electronic.field_jvp(geometry, field, du), "electronic source JVP"
        )
        projected_dm = self.coordinates.project_tangent(dm)
        result = self.coordinates.reduce_tangent(dc - projected_dm)
        result.setflags(write=False)
        return result

    def vjp(self, geometry: Any, y: object, residual_cotangent: object) -> np.ndarray:
        """Apply ``r_y.T`` using ``T_plus.T`` then the final ``T.T`` pullback."""

        self.fingerprint_sha256()
        evaluation = self.evaluate(geometry, y)
        cotangent = self._y(residual_cotangent, "residual cotangent")
        source = np.asarray(evaluation.source)
        field = np.asarray(evaluation.field)

        # residual = T_plus (source - Pi_q response).  The cotangent of the
        # bracket is therefore T_plus.T @ residual_cotangent.
        bracket_bar = self.coordinates.lift_reduced_cotangent(cotangent)
        direct_source_bar = bracket_bar
        # Pi tangent = T T_plus, so Pi.T = T_plus.T T.T.
        response_bar = -self.coordinates.lift_reduced_cotangent(
            self.coordinates.reduce_source_cotangent(bracket_bar)
        )
        field_bar = self._source(
            self.electronic.field_vjp(geometry, field, response_bar),
            "electronic field VJP",
        )
        continuum_source_bar = self._source(
            self.continuum.source_vjp(geometry, source, field_bar),
            "continuum source VJP",
        )
        # source = c_ref + T y, hence the final cotangent map is T.T.
        result = self.coordinates.reduce_source_cotangent(
            direct_source_bar + continuum_source_bar
        )
        result.setflags(write=False)
        return result

    def coordinate_vjp(
        self, geometry: Any, y: object, residual_cotangent: object
    ) -> np.ndarray:
        """Apply the partial coordinate pullback ``r_R.T`` at fixed ``y``."""

        self.fingerprint_sha256()
        evaluation = self.evaluate(geometry, y)
        cotangent = self._y(residual_cotangent, "residual cotangent")
        source = np.asarray(evaluation.source)
        field = np.asarray(evaluation.field)
        bracket_bar = self.coordinates.lift_reduced_cotangent(cotangent)
        response_bar = -self.coordinates.lift_reduced_cotangent(
            self.coordinates.reduce_source_cotangent(bracket_bar)
        )
        field_bar = self._source(
            self.electronic.field_vjp(geometry, field, response_bar),
            "electronic field VJP",
        )
        electronic_geometry_bar = np.asarray(
            self.electronic.coordinate_vjp(geometry, field, response_bar), dtype=float
        )
        continuum_geometry_bar = np.asarray(
            self.continuum.coordinate_vjp(geometry, source, field_bar), dtype=float
        )
        if (
            electronic_geometry_bar.shape != continuum_geometry_bar.shape
            or not np.all(np.isfinite(electronic_geometry_bar))
            or not np.all(np.isfinite(continuum_geometry_bar))
        ):
            raise ValueError(
                "Provider coordinate VJPs must be finite and equally shaped."
            )
        result = electronic_geometry_bar + continuum_geometry_bar
        result.setflags(write=False)
        return result


__all__ = [
    "ContinuumResponseProvider",
    "ElectronicResponseProvider",
    "ReducedStateEquation",
    "StateEvaluation",
    "geometry_sha256",
    "provider_behavior_sha256",
]
