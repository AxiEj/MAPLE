"""Thin vNext wrapper for the audited pyddx atomic ``l<=1`` reaction map.

This module does not reimplement ddPCM.  It adapts the existing
``PyDDXPCMReactionFieldLinearMap`` to the provider-independent vNext continuum
protocol and binds atom identity, radii, dielectric, grid, and solver settings
in one immutable configuration.  The finite union-of-spheres/Lebedev backend
is linear and reciprocal at fixed geometry, but is not declared fixed-topology
or structurally ``SO(3)``; every public capability therefore remains false.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID,
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1,
)
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
)

ATOMIC_L1_DDX_PROVIDER_ID = "maple.route2.continuum.ddx-atomic-l1.impl.v1"
ATOMIC_L1_DDX_PCM_PROFILE_ID = "ddx-ddpcm-atomic-l1-v1"
ATOMIC_L1_DDX_CAVITY_PROFILE_ID = "ddx-union-of-spheres-exposed-lebedev-v0p8p0"
TESTED_PYDDX_VERSION = "0.8.0"


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _callable_source_identity(factory: object) -> dict[str, str]:
    target = (
        factory
        if inspect.isfunction(factory) or inspect.isclass(factory)
        else type(factory)
    )
    try:
        raw = inspect.getsourcefile(target)
    except (TypeError, OSError):
        raw = None
    if raw is None:
        raise RuntimeError("pyddx map-factory source is unavailable for provenance.")
    path = Path(raw).resolve()
    if not path.is_file():
        raise RuntimeError("pyddx map-factory source path is unavailable.")
    return {
        "module": str(getattr(target, "__module__", "")),
        "qualname": str(getattr(target, "__qualname__", type(target).__qualname__)),
        "source_sha256": _sha256_file(path),
    }


def _legacy_map_source_sha256() -> str:
    path = (
        Path(__file__).resolve().parents[2]
        / "function"
        / "calculator"
        / "extra_correction"
        / "implicit"
        / "pyddx_pcm_response.py"
    )
    if not path.is_file():
        raise RuntimeError("MAPLE's audited pyddx reaction-map source is unavailable.")
    return _sha256_file(path)


def _default_map_factory(positions: np.ndarray, radii: np.ndarray, **kwargs: object):
    from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
        PyDDXPCMReactionFieldLinearMap,
    )

    return PyDDXPCMReactionFieldLinearMap(positions, radii, **kwargs)


@dataclass(frozen=True, slots=True)
class AtomicL1PyDDXState:
    configuration_sha256: str
    geometry_sha256: str
    cavity_topology_sha256: str
    cavity_active_node_count: int
    minimum_cavity_active_set_clearance_angstrom: float
    source: np.ndarray
    field: np.ndarray
    polarization_energy_eV: float
    runtime_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "configuration_sha256",
            "geometry_sha256",
            "cavity_topology_sha256",
        ):
            digest = str(getattr(self, name)).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a SHA256 digest.")
            object.__setattr__(self, name, digest)
        if (
            isinstance(self.cavity_active_node_count, bool)
            or not isinstance(self.cavity_active_node_count, int)
            or self.cavity_active_node_count < 1
        ):
            raise ValueError("cavity_active_node_count must be a positive integer.")
        clearance = float(self.minimum_cavity_active_set_clearance_angstrom)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                "minimum_cavity_active_set_clearance_angstrom must be finite "
                "and non-negative."
            )
        object.__setattr__(
            self, "minimum_cavity_active_set_clearance_angstrom", clearance
        )
        source = np.asarray(self.source, dtype=float)
        field = np.asarray(self.field, dtype=float)
        if (
            source.ndim != 2
            or source.shape[1] != 4
            or field.shape != source.shape
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(field))
        ):
            raise ValueError("source and field must be finite matching (N,4) arrays.")
        if not math.isfinite(float(self.polarization_energy_eV)):
            raise ValueError("polarization_energy_eV must be finite.")
        source = np.array(source, copy=True)
        field = np.array(field, copy=True)
        source.setflags(write=False)
        field.setflags(write=False)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "field", field)
        object.__setattr__(
            self, "runtime_provenance", MappingProxyType(dict(self.runtime_provenance))
        )


class AtomicL1PyDDXPCMBackend:
    """Geometry-rebuilt ddPCM map in the authoritative atomic ``l<=1`` spaces."""

    __slots__ = (
        "_atomic_numbers",
        "_configuration_sha256",
        "_dielectric",
        "_eta",
        "_lmax",
        "_map_factory",
        "_map_factory_identity",
        "_n_lebedev",
        "_n_proc",
        "_radii_angstrom",
        "_sealed",
        "_solver_tolerance",
        "configuration_contract_id",
        "provenance_sha256",
    )

    provider_id = ATOMIC_L1_DDX_PROVIDER_ID
    continuum_profile_id = ATOMIC_L1_DDX_PCM_PROFILE_ID
    cavity_profile_id = ATOMIC_L1_DDX_CAVITY_PROFILE_ID
    coupling_id = AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID
    scalar_id = DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1
    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE
    pairing = ATOMIC_L1_PAIRING
    capabilities = CapabilityStatus()
    fixed_topology = False
    linear_response = True
    reciprocal = True
    source_dependent_geometry = False
    electrostatics_only = True
    include_nonpolar = False

    def __init__(
        self,
        reference_atoms: object,
        radii_angstrom: object,
        *,
        dielectric: float,
        lmax: int,
        n_lebedev: int,
        n_proc: int = 1,
        solver_tolerance: float = 1.0e-10,
        eta: float = 0.1,
        configuration_contract_id: str = (UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID),
        _map_factory: Callable[..., object] | None = None,
    ) -> None:
        numbers = self._atomic_numbers_from(reference_atoms)
        self._positions_from(reference_atoms, expected_count=len(numbers))
        self._reject_pbc(reference_atoms)
        radii = np.asarray(radii_angstrom, dtype=float)
        if (
            radii.shape != (len(numbers),)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "radii_angstrom must be finite and positive with shape (N,)."
            )
        dielectric_value = float(dielectric)
        if not math.isfinite(dielectric_value) or dielectric_value <= 1.0:
            raise ValueError("dielectric must be finite and greater than one.")
        tolerance = float(solver_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("solver_tolerance must be finite and positive.")
        eta_value = float(eta)
        if not math.isfinite(eta_value) or not 0.0 <= eta_value <= 1.0:
            raise ValueError("eta must be finite and lie in [0,1].")
        factory = _default_map_factory if _map_factory is None else _map_factory
        if not callable(factory):
            raise TypeError("_map_factory must be callable.")
        frozen_radii = np.array(radii, copy=True)
        frozen_radii.setflags(write=False)
        object.__setattr__(self, "_atomic_numbers", numbers)
        object.__setattr__(self, "_radii_angstrom", frozen_radii)
        object.__setattr__(self, "_dielectric", dielectric_value)
        object.__setattr__(self, "_lmax", _positive_integer(lmax, name="lmax"))
        object.__setattr__(
            self, "_n_lebedev", _positive_integer(n_lebedev, name="n_lebedev")
        )
        object.__setattr__(self, "_n_proc", _positive_integer(n_proc, name="n_proc"))
        object.__setattr__(self, "_solver_tolerance", tolerance)
        object.__setattr__(self, "_eta", eta_value)
        object.__setattr__(self, "_map_factory", factory)
        object.__setattr__(
            self,
            "_map_factory_identity",
            tuple(sorted(_callable_source_identity(factory).items())),
        )
        object.__setattr__(
            self,
            "configuration_contract_id",
            _text(configuration_contract_id, name="configuration_contract_id"),
        )
        configuration = self._current_configuration_sha256()
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(
            self,
            "provenance_sha256",
            _hash(
                {
                    "provider_id": self.provider_id,
                    "configuration_sha256": configuration,
                    "pyddx_version": TESTED_PYDDX_VERSION,
                    "legacy_map_source_sha256": _legacy_map_source_sha256(),
                }
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("AtomicL1PyDDXPCMBackend is immutable.")
        object.__setattr__(self, name, value)

    @staticmethod
    def _atomic_numbers_from(atoms: object) -> tuple[int, ...]:
        getter = getattr(atoms, "get_atomic_numbers", None)
        if not callable(getter):
            raise TypeError("geometry must expose callable get_atomic_numbers().")
        values = np.asarray(getter())
        if (
            values.ndim != 1
            or values.size < 1
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values <= 0)
        ):
            raise ValueError("atomic numbers must be a non-empty positive vector.")
        return tuple(int(value) for value in values)

    @staticmethod
    def _positions_from(atoms: object, *, expected_count: int) -> np.ndarray:
        getter = getattr(atoms, "get_positions", None)
        if not callable(getter):
            raise TypeError("geometry must expose callable get_positions().")
        values = np.asarray(getter(), dtype=float)
        if values.shape != (expected_count, 3) or not np.all(np.isfinite(values)):
            raise ValueError("geometry positions must be finite with shape (N,3).")
        return np.array(values, copy=True)

    @staticmethod
    def _reject_pbc(atoms: object) -> None:
        getter = getattr(atoms, "get_pbc", None)
        if callable(getter) and np.any(np.asarray(getter(), dtype=bool)):
            raise ValueError("Atomic-l1 pyddx geometry-mediated PCM is nonperiodic.")

    def _geometry(self, geometry: object) -> np.ndarray:
        numbers = self._atomic_numbers_from(geometry)
        if numbers != self._atomic_numbers:
            raise ValueError(
                "continuum atom identity/order changed after construction."
            )
        self._reject_pbc(geometry)
        return self._positions_from(geometry, expected_count=len(numbers))

    def _current_configuration_sha256(self) -> str:
        return _hash(
            {
                "schema": "route2-atomic-l1-pyddx-pcm-backend-v1",
                "provider_id": self.provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "configuration_contract_id": self.configuration_contract_id,
                "coupling_id": self.coupling_id,
                "scalar_id": self.scalar_id,
                "atomic_numbers": list(self._atomic_numbers),
                "radii_angstrom": self._radii_angstrom.tolist(),
                "dielectric": self._dielectric,
                "lmax": self._lmax,
                "n_lebedev": self._n_lebedev,
                "n_proc": self._n_proc,
                "solver_tolerance": self._solver_tolerance,
                "eta": self._eta,
                "pyddx_version": TESTED_PYDDX_VERSION,
                "map_factory": dict(self._map_factory_identity),
                "legacy_map_source_sha256": _legacy_map_source_sha256(),
                "source_space_sha256": self.source_space.metadata_hash(),
                "field_space_sha256": self.field_space.metadata_hash(),
                "pairing_sha256": self.pairing.metadata_hash(),
                "fixed_topology": self.fixed_topology,
                "linear_response": self.linear_response,
                "reciprocal": self.reciprocal,
                "capabilities": [],
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise ValueError("Atomic-l1 pyddx backend configuration drifted.")
        return current

    def _map(self, geometry: object):
        self.configuration_sha256()
        positions = self._geometry(geometry)
        return self._map_factory(
            positions,
            self._radii_angstrom.copy(),
            dielectric=self._dielectric,
            lmax=self._lmax,
            n_lebedev=self._n_lebedev,
            n_proc=self._n_proc,
            solver_tolerance=self._solver_tolerance,
            eta=self._eta,
        )

    @staticmethod
    def _topology_from_map(reaction_map: object) -> tuple[str, int, float]:
        digest = getattr(reaction_map, "cavity_topology_sha256", None)
        if not isinstance(digest, str):
            raise RuntimeError(
                "pyddx reaction map does not expose cavity_topology_sha256."
            )
        digest = digest.lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError("pyddx cavity topology identity is not a SHA256 digest.")
        pairs = getattr(reaction_map, "cavity_active_node_pairs", None)
        if not isinstance(pairs, tuple) or not pairs:
            raise RuntimeError("pyddx reaction map has no active cavity-node identity.")
        for pair in pairs:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in pair
                )
                or pair[0] < 0
                or pair[1] < 0
            ):
                raise RuntimeError("pyddx active cavity-node identity is malformed.")
        raw_clearance = getattr(
            reaction_map, "minimum_cavity_active_set_clearance_angstrom", None
        )
        if isinstance(raw_clearance, bool):
            raise RuntimeError(
                "pyddx reaction map cavity active-set clearance is invalid."
            )
        try:
            clearance = float(raw_clearance)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "pyddx reaction map does not expose a numeric cavity active-set "
                "clearance."
            ) from exc
        if not math.isfinite(clearance) or clearance < 0.0:
            raise RuntimeError(
                "pyddx reaction map cavity active-set clearance must be finite "
                "and non-negative."
            )
        return digest, len(pairs), clearance

    def topology_state(self, geometry: object) -> dict[str, object]:
        """Return the exact exposed sphere/Lebedev active-set identity."""

        reaction_map = self._map(geometry)
        digest, count, clearance = self._topology_from_map(reaction_map)
        return {
            "configuration_sha256": self.configuration_sha256(),
            "cavity_topology_sha256": digest,
            "cavity_active_node_count": count,
            "minimum_cavity_active_set_clearance_angstrom": clearance,
        }

    def _source(self, values: object, *, name: str) -> np.ndarray:
        return self.source_space.validate(
            values, atom_count=len(self._atomic_numbers), name=name
        )

    def _field(self, values: object, *, name: str) -> np.ndarray:
        return self.field_space.validate(
            values, atom_count=len(self._atomic_numbers), name=name
        )

    def evaluate_field(self, geometry: object, source: object) -> np.ndarray:
        values = self._source(source, name="source")
        result = self._map(geometry).apply(values)
        return self._field(result, name="ddPCM field")

    field = evaluate_field

    def source_jvp(
        self, geometry: object, source: object, source_direction: object
    ) -> np.ndarray:
        self._source(source, name="source")
        direction = self._source(source_direction, name="source_direction")
        result = self._map(geometry).apply(direction)
        return self._field(result, name="ddPCM source JVP")

    def source_vjp(
        self, geometry: object, source: object, field_cotangent: object
    ) -> np.ndarray:
        self._source(source, name="source")
        cotangent = self._field(field_cotangent, name="field_cotangent")
        result = self._map(geometry).adjoint(cotangent)
        return self._source(result, name="ddPCM source VJP")

    def coordinate_vjp(
        self, geometry: object, source: object, field_cotangent: object
    ) -> np.ndarray:
        values = self._source(source, name="source")
        cotangent = self._field(field_cotangent, name="field_cotangent")
        result = np.asarray(
            self._map(geometry).full_position_vjp(values, cotangent), dtype=float
        )
        expected = (len(self._atomic_numbers), 3)
        if result.shape != expected or not np.all(np.isfinite(result)):
            raise ValueError("ddPCM coordinate VJP must be finite with shape (N,3).")
        return result.reshape(-1).copy()

    def energy(self, geometry: object, source: object) -> float:
        values = self._source(source, name="source")
        field = self.evaluate_field(geometry, values)
        return 0.5 * self.pairing.pair(values, field)

    def build_state(self, geometry: object, source: object) -> AtomicL1PyDDXState:
        values = self._source(source, name="source")
        reaction_map = self._map(geometry)
        field = self._field(reaction_map.apply(values), name="ddPCM field")
        positions = self._geometry(geometry)
        runtime = getattr(reaction_map, "runtime_provenance", {})
        if not isinstance(runtime, Mapping):
            raise TypeError("pyddx reaction map runtime_provenance must be a mapping.")
        geometry_digest = _hash(
            {
                "atomic_numbers": list(self._atomic_numbers),
                "positions_angstrom": positions.tolist(),
            }
        )
        topology_digest, active_node_count, clearance = self._topology_from_map(
            reaction_map
        )
        return AtomicL1PyDDXState(
            configuration_sha256=self.configuration_sha256(),
            geometry_sha256=geometry_digest,
            cavity_topology_sha256=topology_digest,
            cavity_active_node_count=active_node_count,
            minimum_cavity_active_set_clearance_angstrom=clearance,
            source=values,
            field=field,
            polarization_energy_eV=0.5 * self.pairing.pair(values, field),
            runtime_provenance=runtime,
        )


__all__ = [
    "ATOMIC_L1_DDX_CAVITY_PROFILE_ID",
    "ATOMIC_L1_DDX_PCM_PROFILE_ID",
    "ATOMIC_L1_DDX_PROVIDER_ID",
    "AtomicL1PyDDXPCMBackend",
    "AtomicL1PyDDXState",
    "TESTED_PYDDX_VERSION",
]
