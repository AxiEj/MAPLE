"""Production adapter for the audited fixed-topology C-PCM implementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Sequence

import numpy as np
from ase.data import atomic_numbers
from ase.units import Hartree as ASE_HARTREE_TO_EV

from maple.function.calculator.extra_correction.implicit.route2_fc_aswig_cpcm import (
    FixedTopologyAmplitudeSWIGCPCMResponse,
)
from maple.function.calculator.extra_correction.implicit.route2_pcm_response import (
    FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    LOCAL_JET_DIAGNOSTIC_COUPLING_ID,
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING, PairingMetric
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
    FieldDualSpace,
    SourceSpace,
)
from maple.solvation.surfaces.fixed_topology import (
    CAVITY_PROFILE_ID,
    FixedTopologyAmplitudeSWIGSurfaceProvider,
    FixedTopologySurfaceSnapshot,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTINUUM_PROVIDER_ID = (
    "maple.route2.continuum.fixed-topology-amplitude-swig-cpcm.impl.v1"
)
CONTINUUM_PROFILE_ID = "fixed-topology-linear-reciprocal-cpcm-v1"
_FIELD_SCALE = HARTREE_TO_EV / float(ASE_HARTREE_TO_EV)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(
            f"{name} must contain exactly 64 lowercase hexadecimal digits."
        )
    return value


def _block(values: object, atom_count: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.size != atom_count * 4 or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must be finite with shape ({atom_count}, 4); received {result.shape}."
        )
    frozen = np.array(result, copy=True).reshape(atom_count, 4)
    frozen.setflags(write=False)
    return frozen


def _block_values(values: object, atom_count: int, name: str) -> tuple[float, ...]:
    return tuple(float(v) for v in _block(values, atom_count, name).reshape(-1))


def _positions(geometry: Any, symbols: tuple[str, ...]) -> np.ndarray:
    getter = getattr(geometry, "get_positions", None)
    values = getter() if callable(getter) else geometry
    result = np.asarray(values, dtype=float)
    if result.shape != (len(symbols), 3) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"geometry positions must be finite with shape ({len(symbols)}, 3)."
        )
    number_getter = getattr(geometry, "get_atomic_numbers", None)
    if callable(number_getter):
        expected = np.asarray([atomic_numbers[symbol] for symbol in symbols], dtype=int)
        actual = np.asarray(number_getter(), dtype=int)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                "ASE geometry atomic numbers do not match backend symbols."
            )
    return np.array(result, copy=True)


def _state_hash(
    *,
    provider_id: str,
    configuration_sha256: str,
    provenance_sha256: str,
    scalar_id: str,
    continuum_profile_id: str,
    cavity_profile_id: str,
    coupling_id: str,
    surface_configuration_sha256: str,
    surface_provenance_sha256: str,
    surface_state_hash: str,
    source: np.ndarray,
    field: np.ndarray,
    energy_ev: float,
    energy_hartree: float,
    coordinate_complete: bool,
) -> str:
    return _sha(
        {
            "state_contract": "fixed-topology-cpcm-state-v2",
            "provider_id": provider_id,
            "configuration_sha256": configuration_sha256,
            "provenance_sha256": provenance_sha256,
            "scalar_id": scalar_id,
            "continuum_profile_id": continuum_profile_id,
            "cavity_profile_id": cavity_profile_id,
            "coupling_id": coupling_id,
            "surface_configuration_sha256": surface_configuration_sha256,
            "surface_provenance_sha256": surface_provenance_sha256,
            "surface_state_hash": surface_state_hash,
            "source_units": list(ATOMIC_L1_PAIRING.source_units),
            "field_units": list(ATOMIC_L1_PAIRING.field_units),
            "energy_unit": "eV",
            "source": source.tolist(),
            "field": field.tolist(),
            "energy_ev": energy_ev,
            "energy_hartree": energy_hartree,
            "linear_response": True,
            "reciprocal": True,
            "fixed_topology": True,
            "coordinate_derivative_complete": coordinate_complete,
        }
    )


@dataclass(frozen=True, slots=True)
class FixedTopologyCPCMState:
    """Self-validating immutable snapshot of one same-scalar continuum solve."""

    provider_id: str
    continuum_profile_id: str
    scalar_id: str
    configuration_sha256: str
    provenance_sha256: str
    state_hash: str
    backend_configuration: tuple[object, ...]
    surface: FixedTopologySurfaceSnapshot
    source_values: tuple[float, ...]
    reaction_field_values: tuple[float, ...]
    polarization_energy_ev: float
    polarization_energy_hartree: float
    linear_response: bool
    reciprocal: bool
    fixed_topology: bool
    coordinate_derivative_complete: bool
    cavity_profile_id: str = CAVITY_PROFILE_ID
    coupling_id: str = LOCAL_JET_DIAGNOSTIC_COUPLING_ID
    capabilities: CapabilityStatus = CapabilityStatus()

    def __post_init__(self) -> None:
        if self.provider_id != CONTINUUM_PROVIDER_ID:
            raise ValueError(
                "C-PCM state provider_id does not name this implementation."
            )
        if self.continuum_profile_id != CONTINUUM_PROFILE_ID:
            raise ValueError("C-PCM state continuum_profile_id is invalid.")
        if self.cavity_profile_id != CAVITY_PROFILE_ID:
            raise ValueError("C-PCM state cavity_profile_id is invalid.")
        if self.coupling_id != LOCAL_JET_DIAGNOSTIC_COUPLING_ID:
            raise ValueError("C-PCM state coupling_id is invalid.")
        if self.scalar_id != DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1:
            raise ValueError("C-PCM state scalar_id is invalid.")
        configuration = _digest(self.configuration_sha256, "configuration_sha256")
        backend_configuration = tuple(self.backend_configuration)
        if len(backend_configuration) != 10:
            raise ValueError("C-PCM backend_configuration has an invalid schema.")
        if _sha(backend_configuration) != configuration:
            raise ValueError(
                "C-PCM backend_configuration does not match configuration_sha256."
            )
        provenance = _digest(self.provenance_sha256, "provenance_sha256")
        expected_provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration,
                "legacy_asset": "FixedTopologyAmplitudeSWIGCPCMResponse",
            }
        )
        if provenance != expected_provenance:
            raise ValueError(
                "C-PCM provenance does not bind provider and configuration."
            )
        if not isinstance(self.surface, FixedTopologySurfaceSnapshot):
            raise TypeError("surface must be a FixedTopologySurfaceSnapshot.")
        expected_configuration_members = (
            self.surface.configuration_sha256,
            backend_configuration[1],
            self.scalar_id,
            self.continuum_profile_id,
            self.cavity_profile_id,
            self.coupling_id,
            ATOMIC_L1_PAIRING.metadata_hash(),
            ATOMIC_L1_SOURCE_SPACE.metadata_hash(),
            ATOMIC_L1_FIELD_DUAL_SPACE.metadata_hash(),
            HARTREE_TO_EV,
        )
        if backend_configuration != expected_configuration_members:
            raise ValueError(
                "C-PCM backend_configuration is not bound to this surface and contracts."
            )
        dielectric = backend_configuration[1]
        if (
            isinstance(dielectric, bool)
            or not isinstance(dielectric, float)
            or not math.isfinite(dielectric)
            or dielectric <= 1.0
        ):
            raise ValueError("C-PCM backend_configuration dielectric is invalid.")
        source = _block(self.source_values, self.surface.atom_count, "source")
        field = _block(
            self.reaction_field_values, self.surface.atom_count, "reaction field"
        )
        energy_ev, energy_hartree = float(self.polarization_energy_ev), float(
            self.polarization_energy_hartree
        )
        if not math.isfinite(energy_ev) or not math.isfinite(energy_hartree):
            raise ValueError("polarization energies must be finite.")
        if not all((self.linear_response, self.reciprocal, self.fixed_topology)):
            raise ValueError(
                "C-PCM state must declare linear, reciprocal, fixed topology."
            )
        if self.capabilities.enabled_tiers:
            raise ValueError("Phase-4 fixed-topology C-PCM capabilities remain closed.")
        expected_energy = 0.5 * ATOMIC_L1_PAIRING.pair(source, field)
        if abs(energy_ev - expected_energy) > max(1e-11, 1e-11 * abs(expected_energy)):
            raise ValueError(
                "C-PCM state energy is not the authoritative half coupling."
            )
        if abs(energy_ev - energy_hartree * HARTREE_TO_EV) > max(
            1e-11, 1e-11 * abs(energy_ev)
        ):
            raise ValueError(
                "C-PCM energy does not use the authoritative Hartree-to-eV constant."
            )
        expected_hash = _state_hash(
            provider_id=self.provider_id,
            configuration_sha256=configuration,
            provenance_sha256=provenance,
            scalar_id=self.scalar_id,
            continuum_profile_id=self.continuum_profile_id,
            cavity_profile_id=self.cavity_profile_id,
            coupling_id=self.coupling_id,
            surface_configuration_sha256=self.surface.configuration_sha256,
            surface_provenance_sha256=self.surface.provenance_sha256,
            surface_state_hash=self.surface.state_hash,
            source=source,
            field=field,
            energy_ev=energy_ev,
            energy_hartree=energy_hartree,
            coordinate_complete=bool(self.coordinate_derivative_complete),
        )
        if _digest(self.state_hash, "state_hash") != expected_hash:
            raise ValueError("state_hash does not match C-PCM state contents.")

    @property
    def source(self) -> np.ndarray:
        return _block(self.source_values, self.surface.atom_count, "source")

    @property
    def reaction_field(self) -> np.ndarray:
        return _block(
            self.reaction_field_values, self.surface.atom_count, "reaction field"
        )


class FixedTopologyCPCMBackend:
    """Immutable vNext backend delegating all numerical work to legacy assets."""

    __slots__ = (
        "_surface_provider",
        "_configuration",
        "_configuration_sha256",
        "_provenance_sha256",
        "_sealed",
    )
    provider_id = CONTINUUM_PROVIDER_ID
    continuum_profile_id = CONTINUUM_PROFILE_ID
    cavity_profile_id = CAVITY_PROFILE_ID
    configuration_contract_id = UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID
    coupling_id = LOCAL_JET_DIAGNOSTIC_COUPLING_ID
    scalar_id = DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1
    capabilities = CapabilityStatus()
    source_space: SourceSpace = ATOMIC_L1_SOURCE_SPACE
    field_space: FieldDualSpace = ATOMIC_L1_FIELD_DUAL_SPACE
    pairing: PairingMetric = ATOMIC_L1_PAIRING
    fixed_topology = True
    linear_response = True
    reciprocal = True
    source_dependent_geometry = False

    def __init__(
        self,
        symbols: Sequence[str],
        cavity_radii_angstrom: object,
        *,
        dielectric: float,
        lebedev_order: int | None = None,
        unit_sphere: object | None = None,
        switching_constant: float | None = None,
        runtime_version: str | None = None,
    ) -> None:
        dielectric_value = float(dielectric)
        if not math.isfinite(dielectric_value) or dielectric_value <= 1.0:
            raise ValueError("C-PCM dielectric must be finite and greater than one.")
        surface = FixedTopologyAmplitudeSWIGSurfaceProvider(
            symbols,
            cavity_radii_angstrom,
            lebedev_order=lebedev_order,
            unit_sphere=unit_sphere,
            switching_constant=switching_constant,
            runtime_version=runtime_version,
        )
        configuration = (
            surface.configuration_sha256,
            dielectric_value,
            self.scalar_id,
            self.continuum_profile_id,
            self.cavity_profile_id,
            self.coupling_id,
            self.pairing.metadata_hash(),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            HARTREE_TO_EV,
        )
        configuration_sha = _sha(configuration)
        provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration_sha,
                "legacy_asset": "FixedTopologyAmplitudeSWIGCPCMResponse",
            }
        )
        object.__setattr__(self, "_surface_provider", surface)
        object.__setattr__(self, "_configuration", configuration)
        object.__setattr__(self, "_configuration_sha256", configuration_sha)
        object.__setattr__(self, "_provenance_sha256", provenance)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("FixedTopologyCPCMBackend is immutable.")
        object.__setattr__(self, name, value)

    @property
    def surface_provider(self):
        return self._surface_provider

    @property
    def symbols(self):
        return self._surface_provider.symbols

    @property
    def cavity_radii_angstrom(self):
        return self._surface_provider.cavity_radii_angstrom

    @property
    def dielectric(self):
        return self._configuration[1]

    def configuration_sha256(self):
        self._validate_configuration()
        return self._configuration_sha256

    @property
    def provenance_sha256(self):
        return self._provenance_sha256

    def _validate_configuration(self) -> None:
        if not all(
            hasattr(self, name)
            for name in (
                "_surface_provider",
                "_configuration",
                "_configuration_sha256",
                "_provenance_sha256",
            )
        ):
            raise RuntimeError("C-PCM backend configuration is missing.")
        self._surface_provider._validate_configuration()
        current = (
            self._surface_provider.configuration_sha256,
            self.dielectric,
            self.scalar_id,
            self.continuum_profile_id,
            self.cavity_profile_id,
            self.coupling_id,
            self.pairing.metadata_hash(),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            HARTREE_TO_EV,
        )
        if (
            current != self._configuration
            or _sha(current) != self._configuration_sha256
        ):
            raise RuntimeError("C-PCM backend configuration fingerprint changed.")
        expected_provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": self._configuration_sha256,
                "legacy_asset": "FixedTopologyAmplitudeSWIGCPCMResponse",
            }
        )
        if expected_provenance != self._provenance_sha256:
            raise RuntimeError("C-PCM backend provenance fingerprint changed.")

    @property
    def runtime_provenance(self) -> tuple[tuple[str, str], ...]:
        self._validate_configuration()
        return tuple(
            sorted(
                {
                    "provider_id": self.provider_id,
                    "continuum_profile_id": self.continuum_profile_id,
                    "cavity_profile_id": self.cavity_profile_id,
                    "coupling_id": self.coupling_id,
                    "scalar_id": self.scalar_id,
                    "configuration_sha256": self.configuration_sha256(),
                    "provenance_sha256": self.provenance_sha256,
                    "surface_provider_id": self.surface_provider.provider_id,
                    "surface_provenance_sha256": self.surface_provider.provenance_sha256,
                    "continuum_model": "C-PCM",
                    "surface_cardinality": "all-candidates-retained",
                    "response": "linear-reciprocal",
                    "coordinate_derivative": "legacy-full-position-vjp-required",
                    "continuum_oracle": "torch-parity-available-not-release-admitted",
                    "surface_primitive_oracle": "missing-not-admitted",
                    "capabilities": "none",
                }.items()
            )
        )

    def _legacy_response(self, positions: np.ndarray):
        injected = self.surface_provider.lebedev_order is None
        kwargs = (
            {
                "_unit_sphere": self.surface_provider.unit_sphere,
                "_switching_constant": self.surface_provider.switching_constant,
                "_runtime_version": self.surface_provider.runtime_version,
            }
            if injected
            else {"lebedev_order": self.surface_provider.lebedev_order}
        )
        return FixedTopologyAmplitudeSWIGCPCMResponse(
            self.symbols,
            positions,
            self.cavity_radii_angstrom,
            dielectric=self.dielectric,
            **kwargs,
        )

    def _map_and_surface(self, geometry: Any):
        self._validate_configuration()
        positions = _positions(geometry, self.symbols)
        surface = self.surface_provider.build_state(positions)
        response = self._legacy_response(positions)
        if response.surface_size != surface.candidate_count:
            raise RuntimeError("Legacy response changed fixed surface cardinality.")
        if not np.array_equal(
            response.surface_parent_atom_indices, surface.parent_atom_indices
        ):
            raise RuntimeError("Legacy response changed candidate ownership/order.")
        if not np.allclose(
            response.surface_points_bohr,
            surface.surface_points_bohr,
            rtol=0,
            atol=2e-12,
        ):
            raise RuntimeError(
                "Legacy response and snapshot disagree on candidate positions."
            )
        if not np.allclose(
            response.exposure_amplitudes,
            surface.exposure_amplitudes,
            rtol=0,
            atol=2e-14,
        ):
            raise RuntimeError(
                "Legacy response and snapshot disagree on exposure amplitudes."
            )
        reaction_map = response.reaction_field_linear_map(positions)
        complete = getattr(
            reaction_map, "full_position_derivative_contract_version", None
        ) == FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION and callable(
            getattr(reaction_map, "full_position_vjp", None)
        )
        return surface, reaction_map, bool(complete)

    def build_state(self, geometry: Any, source: np.ndarray) -> FixedTopologyCPCMState:
        surface, reaction_map, complete = self._map_and_surface(geometry)
        values = _block(source, len(self.symbols), "source")
        field = (
            _block(reaction_map.apply_scf(values), len(self.symbols), "reaction field")
            * _FIELD_SCALE
        )
        energy_hartree = float(reaction_map.scf_polarization_energy_hartree(values))
        energy_ev = energy_hartree * HARTREE_TO_EV
        state_hash = _state_hash(
            provider_id=self.provider_id,
            configuration_sha256=self.configuration_sha256(),
            provenance_sha256=self.provenance_sha256,
            scalar_id=self.scalar_id,
            continuum_profile_id=self.continuum_profile_id,
            cavity_profile_id=self.cavity_profile_id,
            coupling_id=self.coupling_id,
            surface_configuration_sha256=surface.configuration_sha256,
            surface_provenance_sha256=surface.provenance_sha256,
            surface_state_hash=surface.state_hash,
            source=values,
            field=field,
            energy_ev=energy_ev,
            energy_hartree=energy_hartree,
            coordinate_complete=complete,
        )
        return FixedTopologyCPCMState(
            self.provider_id,
            self.continuum_profile_id,
            self.scalar_id,
            self.configuration_sha256(),
            self.provenance_sha256,
            state_hash,
            self._configuration,
            surface,
            _block_values(values, len(self.symbols), "source"),
            _block_values(field, len(self.symbols), "reaction field"),
            energy_ev,
            energy_hartree,
            True,
            True,
            True,
            complete,
        )

    def energy(self, geometry: Any, source: np.ndarray) -> float:
        return self.build_state(geometry, source).polarization_energy_ev

    def field(self, geometry: Any, source: np.ndarray) -> np.ndarray:
        return self.build_state(geometry, source).reaction_field

    evaluate_field = field

    def source_jvp(
        self, geometry: Any, source: np.ndarray, source_direction: np.ndarray
    ) -> np.ndarray:
        _block(source, len(self.symbols), "source")
        _, reaction_map, _ = self._map_and_surface(geometry)
        direction = _block(source_direction, len(self.symbols), "source direction")
        return (
            _block(reaction_map.apply(direction), len(self.symbols), "source JVP")
            * _FIELD_SCALE
        )

    def source_vjp(
        self, geometry: Any, source: np.ndarray, field_cotangent: np.ndarray
    ) -> np.ndarray:
        _block(source, len(self.symbols), "source")
        _, reaction_map, _ = self._map_and_surface(geometry)
        cotangent = _block(field_cotangent, len(self.symbols), "field cotangent")
        return (
            _block(reaction_map.adjoint(cotangent), len(self.symbols), "source VJP")
            * _FIELD_SCALE
        )

    def coordinate_vjp(
        self, geometry: Any, source: np.ndarray, field_cotangent: np.ndarray
    ) -> np.ndarray:
        values = _block(source, len(self.symbols), "source")
        _, reaction_map, complete = self._map_and_surface(geometry)
        if not complete:
            raise NotImplementedError(
                "Fixed-topology C-PCM lacks the complete coordinate derivative contract."
            )
        cotangent = _block(field_cotangent, len(self.symbols), "field cotangent")
        result = (
            np.asarray(reaction_map.full_position_vjp(values, cotangent), dtype=float)
            * _FIELD_SCALE
        )
        if result.shape != (len(self.symbols), 3) or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Legacy full_position_vjp returned an invalid coordinate derivative."
            )
        return result.reshape(-1).copy()


__all__ = ["FixedTopologyCPCMBackend", "FixedTopologyCPCMState"]
