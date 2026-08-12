"""Immutable adapter for the audited fixed-cardinality amplitude-SWIG asset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Sequence

import numpy as np
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_fixed_topology_surface import (
    build_fixed_topology_amplitude_swig_surface,
    load_pyscf_amplitude_swig_angular_grid,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SURFACE_PROVIDER_ID = "maple.route2.surface.fixed-topology-amplitude-swig.impl.v1"
CAVITY_PROFILE_ID = "fixed-topology-amplitude-swig-v1"
_TOPOLOGY_LABEL = "fixed-topology-parent-major-grid-minor-v1"
_STATE_LABEL = "fixed-topology-amplitude-swig-state-v2"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, name: str) -> str:
    result = _text(value, name).lower()
    if not _SHA256.fullmatch(result):
        raise ValueError(
            f"{name} must contain exactly 64 lowercase hexadecimal digits."
        )
    return result


def _array(
    values: object, shape: tuple[int, ...], name: str, dtype=float
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.shape != shape or (
        np.issubdtype(result.dtype, np.floating) and not np.all(np.isfinite(result))
    ):
        raise ValueError(
            f"{name} must be finite with shape {shape}; received {result.shape}."
        )
    return np.array(result, copy=True)


def _floats(values: object, shape: tuple[int, ...], name: str):
    return tuple(float(v) for v in _array(values, shape, name).reshape(-1))


def _ints(values: object, shape: tuple[int, ...], name: str):
    return tuple(int(v) for v in _array(values, shape, name, np.int64).reshape(-1))


def _matrix(values: tuple[float, ...], rows: int, columns: int) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(rows, columns).copy()
    result.setflags(write=False)
    return result


def _vector(values: tuple[float, ...] | tuple[int, ...], dtype=float) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _hash(
    label: str, arrays: tuple[np.ndarray, ...], metadata: dict[str, object]
) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for values in arrays:
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.integer):
            canonical = np.ascontiguousarray(array, dtype="<i8")
        elif np.issubdtype(array.dtype, np.floating):
            canonical = np.ascontiguousarray(array, dtype="<f8")
        else:
            raise TypeError(
                "Only integer and floating arrays can enter surface hashes."
            )
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(json.dumps(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _positions_angstrom(geometry: Any, atom_count: int) -> np.ndarray:
    getter = getattr(geometry, "get_positions", None)
    values = getter() if callable(getter) else geometry
    return _array(values, (atom_count, 3), "geometry positions")


@dataclass(frozen=True, slots=True)
class FixedTopologySurfaceSnapshot:
    """Content-addressed, immutable all-candidate surface snapshot."""

    provider_id: str
    configuration_sha256: str
    provenance_sha256: str
    topology_hash: str
    state_hash: str
    atom_count: int
    grid_points_per_atom: int
    reference_positions_bohr_values: tuple[float, ...]
    radii_bohr_values: tuple[float, ...]
    parent_atom_index_values: tuple[int, ...]
    surface_point_values: tuple[float, ...]
    unit_direction_values: tuple[float, ...]
    quadrature_weight_values: tuple[float, ...]
    exposure_amplitude_values: tuple[float, ...]
    base_area_bohr2_values: tuple[float, ...]
    effective_area_bohr2_values: tuple[float, ...]
    cavity_profile_id: str = CAVITY_PROFILE_ID
    fixed_topology: bool = True

    def __post_init__(self) -> None:
        provider_id = _text(self.provider_id, "provider_id")
        if provider_id != SURFACE_PROVIDER_ID:
            raise ValueError(
                "Surface snapshot provider_id does not name this implementation."
            )
        if self.cavity_profile_id != CAVITY_PROFILE_ID:
            raise ValueError("Surface snapshot cavity_profile_id is invalid.")
        configuration = _digest(self.configuration_sha256, "configuration_sha256")
        provenance = _digest(self.provenance_sha256, "provenance_sha256")
        expected_provenance = hashlib.sha256(
            json.dumps(
                {"provider_id": provider_id, "configuration_sha256": configuration},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if provenance != expected_provenance:
            raise ValueError(
                "Surface provenance does not bind provider and configuration."
            )
        if isinstance(self.atom_count, bool) or self.atom_count < 1:
            raise ValueError("atom_count must be a positive integer.")
        if isinstance(self.grid_points_per_atom, bool) or self.grid_points_per_atom < 1:
            raise ValueError("grid_points_per_atom must be a positive integer.")
        if self.fixed_topology is not True:
            raise ValueError(
                "FixedTopologySurfaceSnapshot requires fixed_topology=True."
            )
        count = self.atom_count * self.grid_points_per_atom
        positions = _matrix(self.reference_positions_bohr_values, self.atom_count, 3)
        radii = _vector(self.radii_bohr_values)
        parents = _vector(self.parent_atom_index_values, np.int64)
        points = _matrix(self.surface_point_values, count, 3)
        directions = _matrix(self.unit_direction_values, count, 3)
        weights = _vector(self.quadrature_weight_values)
        amplitudes = _vector(self.exposure_amplitude_values)
        base = _vector(self.base_area_bohr2_values)
        effective = _vector(self.effective_area_bohr2_values)
        for values, shape, name in (
            (positions, (self.atom_count, 3), "reference_positions_bohr"),
            (radii, (self.atom_count,), "radii_bohr"),
            (parents, (count,), "parent_atom_indices"),
            (points, (count, 3), "surface_points_bohr"),
            (directions, (count, 3), "unit_directions"),
            (weights, (count,), "quadrature_weights"),
            (amplitudes, (count,), "exposure_amplitudes"),
            (base, (count,), "base_areas_bohr2"),
            (effective, (count,), "effective_areas_bohr2"),
        ):
            _array(values, shape, name, values.dtype)
        if (
            np.any(radii <= 0.0)
            or np.any(parents < 0)
            or np.any(parents >= self.atom_count)
        ):
            raise ValueError("Surface radii/parent ownership are invalid.")
        expected_parents = np.repeat(
            np.arange(self.atom_count), self.grid_points_per_atom
        )
        if not np.array_equal(parents, expected_parents):
            raise ValueError(
                "Candidate order must be parent-major/grid-minor without deletion."
            )
        if not np.allclose(
            np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=2e-12
        ):
            raise ValueError("unit_directions must have unit norm.")
        if np.any(weights <= 0.0) or np.any(base <= 0.0):
            raise ValueError("quadrature weights and base areas must be positive.")
        expected_points = positions[parents] + radii[parents, None] * directions
        if not np.allclose(points, expected_points, rtol=0.0, atol=2e-12):
            raise ValueError(
                "Surface points do not match parent/radius/direction geometry."
            )
        expected_base = weights * radii[parents] ** 2
        if not np.allclose(base, expected_base, rtol=2e-13, atol=1e-14):
            raise ValueError("base areas do not match weights and radii.")
        if np.any(amplitudes < 0.0) or np.any(amplitudes > 1.0):
            raise ValueError("exposure amplitudes must lie in [0, 1].")
        if not np.allclose(effective, base * amplitudes**2, rtol=1e-13, atol=1e-15):
            raise ValueError("effective areas must equal base_area * amplitude**2.")
        expected_topology = _hash(
            _TOPOLOGY_LABEL,
            (parents, directions, weights),
            {
                "atom_count": self.atom_count,
                "grid_points_per_atom": self.grid_points_per_atom,
            },
        )
        if _digest(self.topology_hash, "topology_hash") != expected_topology:
            raise ValueError("topology_hash does not match snapshot topology.")
        expected_state = _hash(
            _STATE_LABEL,
            (positions, radii, points, amplitudes, base, effective),
            {
                "provider_id": provider_id,
                "cavity_profile_id": self.cavity_profile_id,
                "provenance_sha256": provenance,
                "topology_hash": expected_topology,
            },
        )
        if _digest(self.state_hash, "state_hash") != expected_state:
            raise ValueError("state_hash does not match snapshot contents.")

    def validate_integrity(self) -> None:
        """Recompute every identity and geometry invariant after construction."""

        self.__post_init__()

    @property
    def candidate_count(self) -> int:
        return self.atom_count * self.grid_points_per_atom

    @property
    def reference_positions_bohr(self):
        return _matrix(self.reference_positions_bohr_values, self.atom_count, 3)

    @property
    def radii_bohr(self):
        return _vector(self.radii_bohr_values)

    @property
    def parent_atom_indices(self):
        return _vector(self.parent_atom_index_values, np.int64)

    @property
    def surface_points_bohr(self):
        return _matrix(self.surface_point_values, self.candidate_count, 3)

    @property
    def unit_directions(self):
        return _matrix(self.unit_direction_values, self.candidate_count, 3)

    @property
    def quadrature_weights(self):
        return _vector(self.quadrature_weight_values)

    @property
    def exposure_amplitudes(self):
        return _vector(self.exposure_amplitude_values)

    @property
    def base_areas_bohr2(self):
        return _vector(self.base_area_bohr2_values)

    @property
    def effective_areas_bohr2(self):
        return _vector(self.effective_area_bohr2_values)


class FixedTopologyAmplitudeSWIGSurfaceProvider:
    """Immutable configuration that calls the existing audited constructor."""

    __slots__ = (
        "_configuration",
        "_configuration_sha256",
        "_provenance_sha256",
        "_sealed",
    )
    provider_id = SURFACE_PROVIDER_ID
    cavity_profile_id = CAVITY_PROFILE_ID
    fixed_topology = True

    def __init__(
        self,
        symbols: Sequence[str],
        cavity_radii_angstrom: object,
        *,
        lebedev_order: int | None = None,
        unit_sphere: object | None = None,
        switching_constant: float | None = None,
        runtime_version: str | None = None,
    ) -> None:
        symbols_tuple = tuple(str(symbol) for symbol in symbols)
        if not symbols_tuple or any(not symbol for symbol in symbols_tuple):
            raise ValueError("symbols must be a non-empty sequence.")
        radii = _array(
            cavity_radii_angstrom, (len(symbols_tuple),), "cavity_radii_angstrom"
        )
        if np.any(radii <= 0.0):
            raise ValueError("cavity radii must be strictly positive.")
        injected = unit_sphere is not None or switching_constant is not None
        if injected and (unit_sphere is None or switching_constant is None):
            raise ValueError(
                "unit_sphere and switching_constant must be supplied together."
            )
        if injected:
            angular = np.asarray(unit_sphere, dtype=float)
            if (
                angular.ndim != 2
                or angular.shape[0] < 1
                or angular.shape[1] != 4
                or not np.all(np.isfinite(angular))
            ):
                raise ValueError("unit_sphere must be finite with shape (n_grid, 4).")
            constant, version, order = (
                float(switching_constant),
                str(runtime_version or "injected-test-grid"),
                None,
            )
        else:
            if lebedev_order is None:
                raise ValueError("lebedev_order is required without an injected grid.")
            angular, constant, version = load_pyscf_amplitude_swig_angular_grid(
                int(lebedev_order)
            )
            order = int(lebedev_order)
        if not math.isfinite(constant) or constant <= 0.0:
            raise ValueError("switching_constant must be finite and positive.")
        configuration = (
            self.cavity_profile_id,
            symbols_tuple,
            tuple(float(v) for v in radii),
            tuple(tuple(float(v) for v in row) for row in angular),
            constant,
            version,
            order,
        )
        configuration_sha = hashlib.sha256(
            json.dumps(configuration, separators=(",", ":")).encode()
        ).hexdigest()
        provenance = hashlib.sha256(
            json.dumps(
                {
                    "provider_id": self.provider_id,
                    "configuration_sha256": configuration_sha,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        object.__setattr__(self, "_configuration", configuration)
        object.__setattr__(self, "_configuration_sha256", configuration_sha)
        object.__setattr__(self, "_provenance_sha256", provenance)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "FixedTopologyAmplitudeSWIGSurfaceProvider is immutable."
            )
        object.__setattr__(self, name, value)

    @property
    def symbols(self):
        return self._configuration[1]

    @property
    def cavity_radii_angstrom(self):
        return np.asarray(self._configuration[2], dtype=float)

    @property
    def unit_sphere(self):
        return np.asarray(self._configuration[3], dtype=float)

    @property
    def switching_constant(self):
        return self._configuration[4]

    @property
    def runtime_version(self):
        return self._configuration[5]

    @property
    def lebedev_order(self):
        return self._configuration[6]

    @property
    def configuration_sha256(self):
        return self._configuration_sha256

    @property
    def provenance_sha256(self):
        return self._provenance_sha256

    def _validate_configuration(self) -> None:
        if not all(
            hasattr(self, name)
            for name in (
                "_configuration",
                "_configuration_sha256",
                "_provenance_sha256",
            )
        ):
            raise RuntimeError("Surface provider configuration is missing.")
        expected = hashlib.sha256(
            json.dumps(self._configuration, separators=(",", ":")).encode()
        ).hexdigest()
        if expected != self._configuration_sha256:
            raise RuntimeError("Surface provider configuration fingerprint changed.")
        expected_provenance = hashlib.sha256(
            json.dumps(
                {"provider_id": self.provider_id, "configuration_sha256": expected},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if expected_provenance != self._provenance_sha256:
            raise RuntimeError("Surface provider provenance fingerprint changed.")

    def build_state(self, geometry: Any) -> FixedTopologySurfaceSnapshot:
        self._validate_configuration()
        positions_angstrom = _positions_angstrom(geometry, len(self.symbols))
        radii_bohr = self.cavity_radii_angstrom / Bohr
        surface = build_fixed_topology_amplitude_swig_surface(
            positions_angstrom / Bohr,
            radii_bohr,
            self.unit_sphere,
            switching_constant=self.switching_constant,
        )
        topology = _hash(
            _TOPOLOGY_LABEL,
            (
                surface.parent_atom_indices,
                surface.unit_directions,
                surface.quadrature_weights,
            ),
            {
                "atom_count": len(self.symbols),
                "grid_points_per_atom": surface.grid_points_per_atom,
            },
        )
        state = _hash(
            _STATE_LABEL,
            (
                surface.reference_positions_bohr,
                radii_bohr,
                surface.surface_points_bohr,
                surface.exposure_amplitudes,
                surface.base_areas_bohr2,
                surface.effective_areas_bohr2,
            ),
            {
                "provider_id": self.provider_id,
                "cavity_profile_id": self.cavity_profile_id,
                "provenance_sha256": self.provenance_sha256,
                "topology_hash": topology,
            },
        )
        return FixedTopologySurfaceSnapshot(
            self.provider_id,
            self.configuration_sha256,
            self.provenance_sha256,
            topology,
            state,
            len(self.symbols),
            surface.grid_points_per_atom,
            _floats(
                surface.reference_positions_bohr, (len(self.symbols), 3), "positions"
            ),
            _floats(radii_bohr, (len(self.symbols),), "radii"),
            _ints(surface.parent_atom_indices, (surface.surface_size,), "parents"),
            _floats(surface.surface_points_bohr, (surface.surface_size, 3), "points"),
            _floats(surface.unit_directions, (surface.surface_size, 3), "directions"),
            _floats(surface.quadrature_weights, (surface.surface_size,), "weights"),
            _floats(surface.exposure_amplitudes, (surface.surface_size,), "amplitudes"),
            _floats(surface.base_areas_bohr2, (surface.surface_size,), "base areas"),
            _floats(
                surface.effective_areas_bohr2,
                (surface.surface_size,),
                "effective areas",
            ),
        )


__all__ = [
    "CAVITY_PROFILE_ID",
    "FixedTopologyAmplitudeSWIGSurfaceProvider",
    "FixedTopologySurfaceSnapshot",
]
