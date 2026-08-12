"""Canonical source/surface coupling contracts and conjugate matrix map."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
    FieldDualSpace,
    SourceSpace,
)


@runtime_checkable
class FixedSurfaceGeometryLike(Protocol):
    """Minimal geometry view consumed by fixed-surface coupling kernels."""

    @property
    def atom_positions_angstrom(self) -> np.ndarray: ...

    @property
    def surface_points_bohr(self) -> np.ndarray: ...


def validate_fixed_surface_geometry(
    geometry: FixedSurfaceGeometryLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated atom and surface coordinates without changing units."""

    try:
        positions = np.asarray(geometry.atom_positions_angstrom, dtype=float)
        points = np.asarray(geometry.surface_points_bohr, dtype=float)
    except AttributeError as exc:
        raise TypeError(
            "geometry must expose atom_positions_angstrom and surface_points_bohr"
        ) from exc
    if (
        positions.ndim != 2
        or positions.shape[0] < 1
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("atom_positions_angstrom must be finite with shape (n, 3).")
    if (
        points.ndim != 2
        or points.shape[0] < 1
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("surface_points_bohr must be finite with shape (m, 3).")
    return positions, points


class CoordinateDerivativeUnavailable(RuntimeError):
    """Raised when a coupling has no admitted total coordinate derivative."""


def canonical_metadata_sha256(value: object) -> str:
    """Hash JSON-compatible scientific metadata with one canonical encoding."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def configuration_items(
    values: dict[str, object],
) -> tuple[tuple[str, str], ...]:
    """Normalize a small provider configuration for stable hashing."""

    return tuple(sorted((str(key), str(value)) for key, value in values.items()))


def source_files_sha256(
    files: dict[str, str | Path],
) -> tuple[tuple[str, str], ...]:
    """Hash executable source dependencies without storing host paths.

    The mapping key is a stable logical module identifier.  Absolute paths are
    used only to read bytes and never enter provenance.  This deliberately
    differs from an algorithm-contract digest: the former changes when the
    loaded implementation files change, while the latter versions the intended
    mathematics.
    """

    if not files:
        raise ValueError("At least one executable source file must be bound.")
    result: list[tuple[str, str]] = []
    for logical_name, raw_path in files.items():
        if not isinstance(logical_name, str) or not logical_name.strip():
            raise ValueError("Source-file logical names must be non-empty strings.")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Executable source dependency {logical_name!r} is unavailable."
            )
        result.append(
            (logical_name.strip(), hashlib.sha256(path.read_bytes()).hexdigest())
        )
    if len({name for name, _ in result}) != len(result):
        raise ValueError("Executable source-file logical names must be unique.")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class CouplingProvenance:
    """Content-addressed scientific identity for one coupling configuration."""

    coupling_id: str
    scalar_id: str
    provider_id: str
    implementation_version: str
    algorithm_contract_sha256: str
    source_files_sha256: tuple[tuple[str, str], ...]
    tested_git_commit: str | None
    release_binding_status: str
    representation: str
    source_kernel: str
    basis_definition: str
    receiver_definition: str
    coordinate_derivative: str
    checkpoint_attachment: str
    charged_source_gauge_status: str
    production_status: str
    configuration: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        text_fields = (
            "coupling_id",
            "scalar_id",
            "provider_id",
            "implementation_version",
            "release_binding_status",
            "representation",
            "source_kernel",
            "basis_definition",
            "receiver_definition",
            "coordinate_derivative",
            "checkpoint_attachment",
            "charged_source_gauge_status",
            "production_status",
        )
        for name in text_fields:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        digest = self.algorithm_contract_sha256.lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                "algorithm_contract_sha256 must contain 64 hexadecimal digits."
            )
        object.__setattr__(self, "algorithm_contract_sha256", digest)
        source_files = tuple(self.source_files_sha256)
        if not source_files:
            raise ValueError("source_files_sha256 must bind executable source files.")
        if len({name for name, _ in source_files}) != len(source_files):
            raise ValueError("source_files_sha256 logical names must be unique.")
        normalized_sources: list[tuple[str, str]] = []
        for logical_name, source_digest in source_files:
            if not isinstance(logical_name, str) or not logical_name.strip():
                raise ValueError(
                    "source_files_sha256 logical names must be non-empty strings."
                )
            digest_value = str(source_digest).lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest_value) is None:
                raise ValueError(
                    "source_files_sha256 values must contain 64 hexadecimal digits."
                )
            normalized_sources.append((logical_name.strip(), digest_value))
        object.__setattr__(
            self, "source_files_sha256", tuple(sorted(normalized_sources))
        )
        if self.tested_git_commit is not None:
            git_commit = str(self.tested_git_commit).lower()
            if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git_commit) is None:
                raise ValueError(
                    "tested_git_commit must be a full 40- or 64-digit Git object ID."
                )
            object.__setattr__(self, "tested_git_commit", git_commit)
            if "unbound" in self.release_binding_status.lower():
                raise ValueError(
                    "A tested Git commit cannot use an unbound release status."
                )
        elif "unbound" not in self.release_binding_status.lower():
            raise ValueError(
                "Missing tested_git_commit must be declared unbound in release_binding_status."
            )
        configuration = tuple(self.configuration)
        if not configuration or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in configuration
        ):
            raise ValueError(
                "configuration must contain non-empty string key/value pairs."
            )
        if len({key for key, _ in configuration}) != len(configuration):
            raise ValueError("configuration keys must be unique.")
        object.__setattr__(self, "configuration", tuple(sorted(configuration)))

    def metadata(self) -> dict[str, object]:
        return {
            "coupling_id": self.coupling_id,
            "scalar_id": self.scalar_id,
            "provider_id": self.provider_id,
            "implementation_version": self.implementation_version,
            "algorithm_contract_sha256": self.algorithm_contract_sha256,
            "source_files_sha256": dict(self.source_files_sha256),
            "tested_git_commit": self.tested_git_commit,
            "release_binding_status": self.release_binding_status,
            "representation": self.representation,
            "source_kernel": self.source_kernel,
            "basis_definition": self.basis_definition,
            "receiver_definition": self.receiver_definition,
            "coordinate_derivative": self.coordinate_derivative,
            "checkpoint_attachment": self.checkpoint_attachment,
            "charged_source_gauge_status": self.charged_source_gauge_status,
            "production_status": self.production_status,
            "configuration": dict(self.configuration),
        }

    def configuration_hash(self) -> str:
        return canonical_metadata_sha256(dict(self.configuration))

    def metadata_hash(self) -> str:
        return canonical_metadata_sha256(self.metadata())


@runtime_checkable
class CouplingOperator(Protocol):
    """A coupling bound to the canonical Route-2 source and dual spaces."""

    @property
    def coupling_id(self) -> str: ...

    @property
    def scalar_id(self) -> str: ...

    @property
    def provider_id(self) -> str: ...

    @property
    def provenance_sha256(self) -> str: ...

    @property
    def source_space(self) -> SourceSpace: ...

    @property
    def field_space(self) -> FieldDualSpace: ...

    def apply_source(
        self, geometry: FixedSurfaceGeometryLike, source: np.ndarray
    ) -> np.ndarray: ...

    def apply_adjoint(
        self, geometry: FixedSurfaceGeometryLike, surface_cotangent: np.ndarray
    ) -> np.ndarray: ...

    def source_jvp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        source_direction: np.ndarray,
    ) -> np.ndarray: ...

    def source_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        surface_cotangent: np.ndarray,
    ) -> np.ndarray: ...

    def coordinate_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        surface_cotangent: np.ndarray,
    ) -> np.ndarray: ...


MatrixBuilder = Callable[[FixedSurfaceGeometryLike], np.ndarray]
PartialCoordinateVJP = Callable[
    [FixedSurfaceGeometryLike, np.ndarray, np.ndarray], np.ndarray
]


def _immutable_matrix(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or min(array.shape) < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty matrix.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


class ConjugateSurfaceMap:
    """One canonical-Q matrix map used for both ``B`` and ``B*``.

    A callback named ``partial_coordinate_vjp`` is deliberately *not* wired to
    :meth:`coordinate_vjp`: a fixed-surface kernel term is not the total
    derivative of a moving continuum surface.
    """

    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE

    def __init__(
        self,
        *,
        matrix_builder: MatrixBuilder,
        partial_coordinate_vjp: PartialCoordinateVJP | None = None,
    ) -> None:
        if not callable(matrix_builder):
            raise TypeError("matrix_builder must be callable.")
        if partial_coordinate_vjp is not None and not callable(partial_coordinate_vjp):
            raise TypeError("partial_coordinate_vjp must be callable or None.")
        self._matrix_builder = matrix_builder
        self._partial_coordinate_vjp = partial_coordinate_vjp

    def matrix(self, geometry: FixedSurfaceGeometryLike) -> np.ndarray:
        positions, _ = validate_fixed_surface_geometry(geometry)
        matrix = _immutable_matrix(self._matrix_builder(geometry), name="coupling B")
        expected_columns = positions.shape[0] * self.source_space.component_count
        if matrix.shape[1] != expected_columns:
            raise RuntimeError(
                f"coupling B must have {expected_columns} columns; got {matrix.shape[1]}."
            )
        return matrix

    def apply_source(
        self, geometry: FixedSurfaceGeometryLike, source: object
    ) -> np.ndarray:
        positions, _ = validate_fixed_surface_geometry(geometry)
        values = self.source_space.validate(source, atom_count=len(positions))
        return self.matrix(geometry) @ values.reshape(-1)

    def apply_adjoint(
        self, geometry: FixedSurfaceGeometryLike, surface_cotangent: object
    ) -> np.ndarray:
        positions, _ = validate_fixed_surface_geometry(geometry)
        matrix = self.matrix(geometry)
        cotangent = np.asarray(surface_cotangent, dtype=float)
        if cotangent.shape != (matrix.shape[0],) or not np.all(np.isfinite(cotangent)):
            raise ValueError(
                "surface_cotangent must be finite with one value per surface point."
            )
        raw_dual = (matrix.T @ cotangent).reshape(
            len(positions), self.source_space.component_count
        )
        field = self.field_space.pairing_metric.source_to_field_dual(raw_dual)
        return self.field_space.validate(field, atom_count=len(positions))

    def source_jvp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: object,
        source_direction: object,
    ) -> np.ndarray:
        positions, _ = validate_fixed_surface_geometry(geometry)
        self.source_space.validate(source, atom_count=len(positions))
        direction = self.source_space.validate(
            source_direction, atom_count=len(positions), name="source_direction"
        )
        return self.matrix(geometry) @ direction.reshape(-1)

    def source_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: object,
        surface_cotangent: object,
    ) -> np.ndarray:
        positions, _ = validate_fixed_surface_geometry(geometry)
        self.source_space.validate(source, atom_count=len(positions))
        return self.apply_adjoint(geometry, surface_cotangent)

    def coordinate_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: object,
        surface_cotangent: object,
    ) -> np.ndarray:
        del geometry, source, surface_cotangent
        raise CoordinateDerivativeUnavailable(
            "This coupling has no admitted total analytic coordinate VJP; "
            "fixed-surface kernel terms are partial derivatives only."
        )

    def partial_fixed_surface_kernel_coordinate_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: object,
        surface_cotangent: object,
    ) -> np.ndarray:
        """Return the explicitly named fixed-node kernel term, when available."""

        if self._partial_coordinate_vjp is None:
            raise CoordinateDerivativeUnavailable(
                "This coupling has no fixed-surface kernel coordinate VJP."
            )
        positions, _ = validate_fixed_surface_geometry(geometry)
        source_values = self.source_space.validate(source, atom_count=len(positions))
        matrix = self.matrix(geometry)
        cotangent = np.asarray(surface_cotangent, dtype=float)
        if cotangent.shape != (matrix.shape[0],) or not np.all(np.isfinite(cotangent)):
            raise ValueError(
                "surface_cotangent must be finite with one value per surface point."
            )
        result = np.asarray(
            self._partial_coordinate_vjp(geometry, source_values, cotangent),
            dtype=float,
        )
        if result.shape != positions.shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "partial coordinate VJP must be finite with shape (n_atoms, 3)."
            )
        return result


@dataclass(frozen=True, slots=True)
class AdjointValidation:
    source_to_surface: float
    source_from_surface: float
    absolute_error: float
    tolerance: float
    passed: bool


def validate_adjoint_dot_product(
    operator: CouplingOperator,
    geometry: FixedSurfaceGeometryLike,
    source: object,
    surface_cotangent: object,
    *,
    atom_count: int,
    surface_pairing=None,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
) -> AdjointValidation:
    """Validate ``<B c,sigma> = <c,B* sigma>_Q`` and fail closed."""

    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("adjoint tolerances must be non-negative.")
    if operator.source_space is not ATOMIC_L1_SOURCE_SPACE:
        raise ValueError("operator must use the canonical atomic-l1 source space.")
    if operator.field_space is not ATOMIC_L1_FIELD_DUAL_SPACE:
        raise ValueError("operator must use the canonical atomic-l1 field dual space.")
    source_values = operator.source_space.validate(source, atom_count=atom_count)
    cotangent = np.asarray(surface_cotangent, dtype=float)
    if cotangent.size == 0 or not np.all(np.isfinite(cotangent)):
        raise ValueError("surface_cotangent must be finite and non-empty.")
    mapped = np.asarray(operator.apply_source(geometry, source_values), dtype=float)
    if mapped.shape != cotangent.shape or not np.all(np.isfinite(mapped)):
        raise ValueError(
            "apply_source output must be finite and match surface_cotangent shape."
        )
    field = operator.field_space.validate(
        operator.apply_adjoint(geometry, cotangent),
        atom_count=atom_count,
        name="coupling adjoint field",
    )
    pairing = np.vdot if surface_pairing is None else surface_pairing
    lhs = float(pairing(mapped, cotangent))
    rhs = operator.field_space.pair(source_values, field, atom_count=atom_count)
    if not np.isfinite(lhs) or not np.isfinite(rhs):
        raise ValueError("adjoint pairings must be finite.")
    tolerance = absolute_tolerance + relative_tolerance * max(abs(lhs), abs(rhs))
    error = abs(lhs - rhs)
    return AdjointValidation(lhs, rhs, error, tolerance, error <= tolerance)


validate_coupling_adjoint = validate_adjoint_dot_product

__all__ = [
    "AdjointValidation",
    "ConjugateSurfaceMap",
    "CoordinateDerivativeUnavailable",
    "CouplingProvenance",
    "CouplingOperator",
    "FixedSurfaceGeometryLike",
    "canonical_metadata_sha256",
    "configuration_items",
    "source_files_sha256",
    "validate_adjoint_dot_product",
    "validate_coupling_adjoint",
    "validate_fixed_surface_geometry",
]
