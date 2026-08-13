"""Conjugate radial-GTO fixed-topology C-PCM provider.

This module composes exactly three audited pieces without introducing a second
electrostatic model:

* the immutable fixed-cardinality amplitude-SWIG surface,
* the stationary symmetric C-PCM external-MEP response, and
* the physical two-width radial-GTO ``B``/``B*`` coupling.

The returned reaction field is therefore
``P_R c = B_R* Q_R (B_R c / HARTREE_TO_EV)`` in eV energy-dual units.  All
source and coordinate derivatives are contractions of that same operator.
Public E/F/H/V/M admission remains closed pending the complete real-stack gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Sequence

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_fc_aswig_cpcm import (
    FixedTopologyAmplitudeSWIGCPCMResponse,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
    WATER_CPCM_194_CONFIGURATION_CONTRACT_ID,
    WATER_CPCM_590_CONFIGURATION_CONTRACT_ID,
    WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.exact_gto import (
    MACE_POLAR_RADIAL_GTO_COUPLING_ID,
    MACEPolarRadialGTOCoupling,
    OwnedFixedSurfaceGeometry,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.surfaces.fixed_topology import (
    CAVITY_PROFILE_ID,
    FixedTopologyAmplitudeSWIGSurfaceProvider,
    FixedTopologySurfaceSnapshot,
)

from .fixed_topology_cpcm import CONTINUUM_PROFILE_ID, _positions

CONJUGATE_RADIAL_CPCM_PROVIDER_ID = (
    "maple.route2.continuum.fixed-topology-radial-gto-cpcm.impl.v1"
)
_STATE_CONTRACT = "fixed-topology-radial-gto-cpcm-state-v1"
WATER_CPCM_DIELECTRIC = 78.39
WATER_CPCM_LEBEDEV_ORDER = 23
WATER_CPCM_GRID_POINTS_PER_ATOM = 194
WATER_CPCM_590_LEBEDEV_ORDER = 41
WATER_CPCM_590_GRID_POINTS_PER_ATOM = 590
WATER_CPCM_1202_LEBEDEV_ORDER = 59
WATER_CPCM_1202_GRID_POINTS_PER_ATOM = 1202


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tuple_array(
    values: object, shape: tuple[int, ...], name: str
) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {shape}; got {array.shape}."
        )
    return tuple(float(value) for value in array.reshape(-1))


def _readonly(values: tuple[float, ...], shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(shape).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ConjugateRadialCPCMState:
    """Immutable evidence snapshot for one radial-GTO C-PCM solve."""

    provider_id: str
    configuration_sha256: str
    provenance_sha256: str
    state_hash: str
    surface: FixedTopologySurfaceSnapshot
    source_values: tuple[float, ...]
    field_values: tuple[float, ...]
    surface_potential_ev_values: tuple[float, ...]
    surface_charge_values: tuple[float, ...]
    polarization_energy_ev: float
    polarization_energy_hartree: float
    continuum_configuration_contract_id: str
    continuum_profile_id: str = CONTINUUM_PROFILE_ID
    cavity_profile_id: str = CAVITY_PROFILE_ID
    coupling_id: str = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    scalar_id: str = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    capabilities: CapabilityStatus = CapabilityStatus()

    def __post_init__(self) -> None:
        if self.provider_id != CONJUGATE_RADIAL_CPCM_PROVIDER_ID:
            raise ValueError("Radial C-PCM state provider_id is invalid.")
        if self.continuum_profile_id != CONTINUUM_PROFILE_ID:
            raise ValueError("Radial C-PCM continuum profile is invalid.")
        if self.cavity_profile_id != CAVITY_PROFILE_ID:
            raise ValueError("Radial C-PCM cavity profile is invalid.")
        if self.coupling_id != MACE_POLAR_RADIAL_GTO_COUPLING_ID:
            raise ValueError("Radial C-PCM coupling identity is invalid.")
        if self.scalar_id != OPERATIONAL_CPCM_ELECTROSTATIC_V1:
            raise ValueError("Radial C-PCM scalar identity is invalid.")
        if self.continuum_configuration_contract_id not in (
            UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
            WATER_CPCM_194_CONFIGURATION_CONTRACT_ID,
            WATER_CPCM_590_CONFIGURATION_CONTRACT_ID,
            WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID,
        ):
            raise ValueError(
                "Radial C-PCM continuum configuration contract is invalid."
            )
        if not isinstance(self.surface, FixedTopologySurfaceSnapshot):
            raise TypeError("surface must be FixedTopologySurfaceSnapshot.")
        if self.capabilities.enabled_tiers:
            raise ValueError("Radial C-PCM E/F/H/V/M capabilities remain closed.")
        for value, name in (
            (self.configuration_sha256, "configuration_sha256"),
            (self.provenance_sha256, "provenance_sha256"),
            (self.state_hash, "state_hash"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest.")
        source = self.source
        field = self.reaction_field
        potential = self.surface_potential_ev
        charge = self.surface_charge
        energy_ev = float(self.polarization_energy_ev)
        energy_hartree = float(self.polarization_energy_hartree)
        if not math.isfinite(energy_ev) or not math.isfinite(energy_hartree):
            raise ValueError("Radial C-PCM energies must be finite.")
        paired = 0.5 * MACE_POLAR_RADIAL_GTO_PAIRING.pair(source, field)
        surface_energy = 0.5 * float(np.dot(potential, charge))
        tolerance = max(2e-11, 2e-11 * abs(energy_ev))
        if abs(energy_ev - paired) > tolerance:
            raise ValueError("Radial C-PCM energy violates source/field half coupling.")
        if abs(energy_ev - surface_energy) > tolerance:
            raise ValueError(
                "Radial C-PCM energy violates surface-charge half coupling."
            )
        if abs(energy_ev - energy_hartree * HARTREE_TO_EV) > tolerance:
            raise ValueError("Radial C-PCM Hartree/eV energies are inconsistent.")
        expected_hash = _sha(
            {
                "contract": _STATE_CONTRACT,
                "provider_id": self.provider_id,
                "configuration_sha256": self.configuration_sha256,
                "provenance_sha256": self.provenance_sha256,
                "surface_state_hash": self.surface.state_hash,
                "continuum_configuration_contract_id": (
                    self.continuum_configuration_contract_id
                ),
                "source": source.tolist(),
                "field": field.tolist(),
                "surface_potential_ev": potential.tolist(),
                "surface_charge": charge.tolist(),
                "polarization_energy_ev": energy_ev,
                "polarization_energy_hartree": energy_hartree,
            }
        )
        if self.state_hash != expected_hash:
            raise ValueError("Radial C-PCM state_hash does not match its contents.")

    @property
    def source(self) -> np.ndarray:
        return _readonly(self.source_values, (self.surface.atom_count, 8))

    @property
    def reaction_field(self) -> np.ndarray:
        return _readonly(self.field_values, (self.surface.atom_count, 8))

    @property
    def surface_potential_ev(self) -> np.ndarray:
        return _readonly(
            self.surface_potential_ev_values, (self.surface.candidate_count,)
        )

    @property
    def surface_charge(self) -> np.ndarray:
        return _readonly(self.surface_charge_values, (self.surface.candidate_count,))


class ConjugateRadialGTOFixedTopologyCPCMBackend:
    """Same-scalar radial-GTO C-PCM reaction-field provider."""

    __slots__ = (
        "_configuration",
        "_configuration_sha256",
        "_configuration_contract_id",
        "_coupling",
        "_geometry_cache_key",
        "_geometry_cache_lock",
        "_geometry_cache_value",
        "_provenance_sha256",
        "_sealed",
        "_surface_provider",
    )
    provider_id = CONJUGATE_RADIAL_CPCM_PROVIDER_ID
    continuum_profile_id = CONTINUUM_PROFILE_ID
    cavity_profile_id = CAVITY_PROFILE_ID
    coupling_id = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    scalar_id = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE
    pairing = MACE_POLAR_RADIAL_GTO_PAIRING
    capabilities = CapabilityStatus()
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
        configuration_contract_id: str = (UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID),
    ) -> None:
        dielectric_value = float(dielectric)
        if not math.isfinite(dielectric_value) or dielectric_value <= 1.0:
            raise ValueError("C-PCM dielectric must be finite and greater than one.")
        if configuration_contract_id not in (
            UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
            WATER_CPCM_194_CONFIGURATION_CONTRACT_ID,
            WATER_CPCM_590_CONFIGURATION_CONTRACT_ID,
            WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID,
        ):
            raise ValueError("Unknown radial C-PCM continuum configuration contract.")
        radii = np.asarray(cavity_radii_angstrom, dtype=float)
        if configuration_contract_id == WATER_CPCM_194_CONFIGURATION_CONTRACT_ID:
            if dielectric_value != WATER_CPCM_DIELECTRIC:
                raise ValueError(
                    "The water C-PCM profile requires dielectric=78.39 exactly."
                )
            if lebedev_order != WATER_CPCM_LEBEDEV_ORDER:
                raise ValueError(
                    "The water C-PCM profile requires Lebedev order 23 "
                    "(194 points per atom)."
                )
            if any(
                value is not None
                for value in (unit_sphere, switching_constant, runtime_version)
            ):
                raise ValueError(
                    "The water C-PCM profile forbids injected angular grids or "
                    "runtime labels."
                )
            expected_radii = smd_water_coulomb_radii(tuple(symbols))
            if radii.shape != expected_radii.shape or not np.array_equal(
                radii, expected_radii
            ):
                raise ValueError(
                    "The water C-PCM profile requires the exact versioned SMD-water "
                    "Coulomb radii policy."
                )
        elif configuration_contract_id == WATER_CPCM_590_CONFIGURATION_CONTRACT_ID:
            if dielectric_value != WATER_CPCM_DIELECTRIC:
                raise ValueError(
                    "The high-order water C-PCM profile requires dielectric=78.39 "
                    "exactly."
                )
            if lebedev_order != WATER_CPCM_590_LEBEDEV_ORDER:
                raise ValueError(
                    "The high-order water C-PCM profile requires Lebedev order 41 "
                    "(590 points per atom)."
                )
            if any(
                value is not None
                for value in (unit_sphere, switching_constant, runtime_version)
            ):
                raise ValueError(
                    "The high-order water C-PCM profile forbids injected angular "
                    "grids or runtime labels."
                )
            expected_radii = smd_water_coulomb_radii(tuple(symbols))
            if radii.shape != expected_radii.shape or not np.array_equal(
                radii, expected_radii
            ):
                raise ValueError(
                    "The high-order water C-PCM profile requires the exact versioned "
                    "SMD-water Coulomb radii policy."
                )
        elif configuration_contract_id == WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID:
            if dielectric_value != WATER_CPCM_DIELECTRIC:
                raise ValueError(
                    "The 1202-node water C-PCM profile requires dielectric=78.39 "
                    "exactly."
                )
            if lebedev_order != WATER_CPCM_1202_LEBEDEV_ORDER:
                raise ValueError(
                    "The 1202-node water C-PCM profile requires Lebedev order 59 "
                    "(1202 points per atom)."
                )
            if any(
                value is not None
                for value in (unit_sphere, switching_constant, runtime_version)
            ):
                raise ValueError(
                    "The 1202-node water C-PCM profile forbids injected angular "
                    "grids or runtime labels."
                )
            expected_radii = smd_water_coulomb_radii(tuple(symbols))
            if radii.shape != expected_radii.shape or not np.array_equal(
                radii, expected_radii
            ):
                raise ValueError(
                    "The 1202-node water C-PCM profile requires the exact versioned "
                    "SMD-water Coulomb radii policy."
                )
        surface_provider = FixedTopologyAmplitudeSWIGSurfaceProvider(
            symbols,
            radii,
            lebedev_order=lebedev_order,
            unit_sphere=unit_sphere,
            switching_constant=switching_constant,
            runtime_version=runtime_version,
        )
        if (
            configuration_contract_id == WATER_CPCM_194_CONFIGURATION_CONTRACT_ID
            and surface_provider.unit_sphere.shape[0] != WATER_CPCM_GRID_POINTS_PER_ATOM
        ):
            raise RuntimeError(
                "The pinned water C-PCM grid did not contain 194 points per atom."
            )
        if (
            configuration_contract_id == WATER_CPCM_590_CONFIGURATION_CONTRACT_ID
            and surface_provider.unit_sphere.shape[0]
            != WATER_CPCM_590_GRID_POINTS_PER_ATOM
        ):
            raise RuntimeError(
                "The pinned high-order water C-PCM grid did not contain 590 points "
                "per atom."
            )
        if (
            configuration_contract_id == WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID
            and surface_provider.unit_sphere.shape[0]
            != WATER_CPCM_1202_GRID_POINTS_PER_ATOM
        ):
            raise RuntimeError(
                "The pinned 1202-node water C-PCM grid did not contain 1202 points "
                "per atom."
            )
        coupling = MACEPolarRadialGTOCoupling()
        configuration = (
            surface_provider.configuration_sha256,
            dielectric_value,
            self.scalar_id,
            self.continuum_profile_id,
            self.cavity_profile_id,
            self.coupling_id,
            coupling.configuration_sha256(),
            coupling.provenance_sha256,
            self.pairing.metadata_hash(),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            configuration_contract_id,
            HARTREE_TO_EV,
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        configuration_sha256 = _sha(configuration)
        provenance_sha256 = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration_sha256,
                "legacy_asset": "FixedTopologyAmplitudeSWIGCPCMResponse",
                "coupling_provenance_sha256": coupling.provenance_sha256,
                "continuum_configuration_contract_id": configuration_contract_id,
            }
        )
        object.__setattr__(self, "_surface_provider", surface_provider)
        object.__setattr__(self, "_coupling", coupling)
        object.__setattr__(self, "_geometry_cache_key", None)
        object.__setattr__(self, "_geometry_cache_lock", threading.RLock())
        object.__setattr__(self, "_geometry_cache_value", None)
        object.__setattr__(self, "_configuration", configuration)
        object.__setattr__(
            self, "_configuration_contract_id", configuration_contract_id
        )
        object.__setattr__(self, "_configuration_sha256", configuration_sha256)
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "ConjugateRadialGTOFixedTopologyCPCMBackend is immutable."
            )
        object.__setattr__(self, name, value)

    @property
    def surface_provider(self) -> FixedTopologyAmplitudeSWIGSurfaceProvider:
        return self._surface_provider

    @property
    def coupling(self) -> MACEPolarRadialGTOCoupling:
        return self._coupling

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.surface_provider.symbols

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        return self.surface_provider.cavity_radii_angstrom

    @property
    def dielectric(self) -> float:
        return self._configuration[1]

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256

    @property
    def configuration_contract_id(self) -> str:
        return self._configuration_contract_id

    def _current_configuration(self) -> tuple[object, ...]:
        self.surface_provider._validate_configuration()
        return (
            self.surface_provider.configuration_sha256,
            self.dielectric,
            self.scalar_id,
            self.continuum_profile_id,
            self.cavity_profile_id,
            self.coupling_id,
            self.coupling.configuration_sha256(),
            self.coupling.provenance_sha256,
            self.pairing.metadata_hash(),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            self.configuration_contract_id,
            HARTREE_TO_EV,
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if (
            current != self._configuration
            or _sha(current) != self._configuration_sha256
        ):
            raise RuntimeError("Radial C-PCM configuration fingerprint changed.")
        expected_provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": self._configuration_sha256,
                "legacy_asset": "FixedTopologyAmplitudeSWIGCPCMResponse",
                "coupling_provenance_sha256": self.coupling.provenance_sha256,
                "continuum_configuration_contract_id": (self.configuration_contract_id),
            }
        )
        if expected_provenance != self._provenance_sha256:
            raise RuntimeError("Radial C-PCM provenance fingerprint changed.")
        return self._configuration_sha256

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

    @staticmethod
    def _make_geometry_cache_key(
        positions: np.ndarray,
    ) -> tuple[tuple[int, ...], bytes]:
        canonical = np.ascontiguousarray(positions, dtype="<f8")
        return canonical.shape, canonical.tobytes(order="C")

    def _response_geometry(self, geometry: Any):
        """Return one exact geometry-local surface/operator snapshot.

        A fixed geometry is queried repeatedly by the nonlinear solve, the
        scalar ledger, and the adjoint.  The surface, dense C-PCM curvature,
        and its Cholesky factor depend only on that exact Cartesian geometry,
        never on the source.  Cache only the most recent immutable-by-contract
        triple, keyed by raw canonical float64 bytes; no tolerance, nearest-
        neighbour, or geometry aliasing is permitted.  Single-entry eviction
        bounds the dense surface-matrix memory across PES paths.
        """

        self.configuration_sha256()
        positions = _positions(geometry, self.symbols)
        key = self._make_geometry_cache_key(positions)
        with self._geometry_cache_lock:
            if key == self._geometry_cache_key:
                cached = self._geometry_cache_value
                if cached is None:  # pragma: no cover - invariant guard
                    raise RuntimeError("C-PCM geometry cache value is missing.")
                return cached

        surface = self.surface_provider.build_state(positions)
        response = self._legacy_response(positions)
        if response.surface_size != surface.candidate_count:
            raise RuntimeError("C-PCM response changed fixed surface cardinality.")
        if not np.array_equal(
            response.surface_parent_atom_indices, surface.parent_atom_indices
        ):
            raise RuntimeError("C-PCM response changed surface owner ordering.")
        if not np.allclose(
            response.surface_points_bohr,
            surface.surface_points_bohr,
            rtol=0.0,
            atol=2e-12,
        ):
            raise RuntimeError("C-PCM response and surface points disagree.")
        if not np.allclose(
            response.exposure_amplitudes,
            surface.exposure_amplitudes,
            rtol=0.0,
            atol=2e-14,
        ):
            raise RuntimeError("C-PCM response and surface amplitudes disagree.")
        owned = OwnedFixedSurfaceGeometry(
            positions,
            surface.surface_points_bohr,
            surface.parent_atom_indices,
        )
        source_operator = self.coupling.surface_operator(owned)
        source_operator.setflags(write=False)
        result = (surface, response, owned, source_operator)
        with self._geometry_cache_lock:
            object.__setattr__(self, "_geometry_cache_key", key)
            object.__setattr__(self, "_geometry_cache_value", result)
        return result

    def _solve_field(self, geometry: Any, source: object):
        surface, response, owned, source_operator = self._response_geometry(geometry)
        values = self.source_space.validate(source, atom_count=surface.atom_count)
        potential_ev = source_operator @ values.reshape(-1)
        amplitude_state = response.solve_amplitude_state(potential_ev / HARTREE_TO_EV)
        charge = amplitude_state.physical_surface_charge_e
        field = (source_operator.T @ charge).reshape(surface.atom_count, 8)
        field = self.field_space.validate(
            field, atom_count=surface.atom_count, name="reaction field"
        )
        return (
            surface,
            response,
            owned,
            values,
            potential_ev,
            charge,
            field,
            amplitude_state,
        )

    def build_state(self, geometry: Any, source: object) -> ConjugateRadialCPCMState:
        (
            surface,
            _response,
            _owned,
            values,
            potential_ev,
            charge,
            field,
            amplitude_state,
        ) = self._solve_field(geometry, source)
        energy_hartree = float(amplitude_state.polarization_energy_hartree)
        energy_ev = energy_hartree * HARTREE_TO_EV
        payload = {
            "contract": _STATE_CONTRACT,
            "provider_id": self.provider_id,
            "configuration_sha256": self.configuration_sha256(),
            "provenance_sha256": self.provenance_sha256,
            "surface_state_hash": surface.state_hash,
            "continuum_configuration_contract_id": self.configuration_contract_id,
            "source": values.tolist(),
            "field": field.tolist(),
            "surface_potential_ev": potential_ev.tolist(),
            "surface_charge": charge.tolist(),
            "polarization_energy_ev": energy_ev,
            "polarization_energy_hartree": energy_hartree,
        }
        return ConjugateRadialCPCMState(
            self.provider_id,
            self.configuration_sha256(),
            self.provenance_sha256,
            _sha(payload),
            surface,
            _tuple_array(values, (surface.atom_count, 8), "source"),
            _tuple_array(field, (surface.atom_count, 8), "reaction field"),
            _tuple_array(potential_ev, (surface.candidate_count,), "surface potential"),
            _tuple_array(charge, (surface.candidate_count,), "surface charge"),
            energy_ev,
            energy_hartree,
            self.configuration_contract_id,
        )

    def energy(self, geometry: Any, source: object) -> float:
        return float(
            self._solve_field(geometry, source)[-1].polarization_energy_hartree
            * HARTREE_TO_EV
        )

    def evaluate_field(self, geometry: Any, source: object) -> np.ndarray:
        return np.array(self._solve_field(geometry, source)[-2], copy=True)

    field = evaluate_field

    def _linear_field(self, geometry: Any, direction: object) -> np.ndarray:
        return np.array(self._solve_field(geometry, direction)[-2], copy=True)

    def reaction_field_matrix(self, geometry: Any) -> np.ndarray:
        """Return the exact dense geometry-local linear reaction map.

        This is an execution view of the same source operator, stationary
        C-PCM solve, and transpose receiver used by ``evaluate_field``.  It is
        useful when one geometry is queried for many source directions (for
        example an equivariant frame ensemble); it does not define another
        electrostatic kernel.
        """

        surface, response, _owned, source_operator = self._response_geometry(geometry)
        response_map = np.column_stack(
            [
                response.apply_energy_conjugate(
                    source_operator[:, column] / HARTREE_TO_EV
                )
                for column in range(source_operator.shape[1])
            ]
        )
        matrix = source_operator.T @ response_map
        expected = surface.atom_count * 8
        if matrix.shape != (expected, expected) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("Radial C-PCM reaction-field matrix is invalid.")
        reciprocity_error = float(np.max(np.abs(matrix - matrix.T)))
        reciprocity_scale = max(1.0, float(np.max(np.abs(matrix))))
        if reciprocity_error > 2.0e-12 * reciprocity_scale:
            raise RuntimeError(
                "Radial C-PCM reaction-field matrix violates reciprocity."
            )
        matrix = np.array(matrix, copy=True)
        matrix.setflags(write=False)
        return matrix

    def source_jvp(
        self, geometry: Any, source: object, source_direction: object
    ) -> np.ndarray:
        count = len(self.symbols)
        self.source_space.validate(source, atom_count=count)
        return self.field_space.validate(
            self._linear_field(geometry, source_direction),
            atom_count=count,
            name="source JVP",
        )

    def source_vjp(
        self, geometry: Any, source: object, field_cotangent: object
    ) -> np.ndarray:
        count = len(self.symbols)
        self.source_space.validate(source, atom_count=count)
        cotangent = self.field_space.validate(
            field_cotangent, atom_count=count, name="field cotangent"
        )
        # The radial physical pairing is identity and the C-PCM response is
        # symmetric, so the exact transpose is the same linear map.
        return self.source_space.validate(
            self._linear_field(geometry, cotangent),
            atom_count=count,
            name="source VJP",
        )

    def coordinate_vjp(
        self, geometry: Any, source: object, field_cotangent: object
    ) -> np.ndarray:
        surface, response, owned, source_operator = self._response_geometry(geometry)
        values = self.source_space.validate(source, atom_count=surface.atom_count)
        cotangent = self.field_space.validate(
            field_cotangent,
            atom_count=surface.atom_count,
            name="field cotangent",
        )
        source_potential_ev = source_operator @ values.reshape(-1)
        left_potential_ev = source_operator @ cotangent.reshape(-1)
        source_charge = response.apply_energy_conjugate(
            source_potential_ev / HARTREE_TO_EV
        )
        left_charge = response.apply_energy_conjugate(left_potential_ev / HARTREE_TO_EV)
        result = self.coupling.coordinate_vjp(owned, cotangent, source_charge)
        result += self.coupling.coordinate_vjp(owned, values, left_charge)
        result += HARTREE_TO_EV * response.operator_position_vjp(
            left_potential_ev / HARTREE_TO_EV,
            source_potential_ev / HARTREE_TO_EV,
        )
        if result.shape != (surface.atom_count, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError("Radial C-PCM coordinate VJP is invalid.")
        return result.reshape(-1).copy()

    @property
    def runtime_provenance(self) -> tuple[tuple[str, str], ...]:
        self.configuration_sha256()
        return tuple(
            sorted(
                {
                    "provider_id": self.provider_id,
                    "continuum_profile_id": self.continuum_profile_id,
                    "cavity_profile_id": self.cavity_profile_id,
                    "coupling_id": self.coupling_id,
                    "scalar_id": self.scalar_id,
                    "configuration_sha256": self.configuration_sha256(),
                    "configuration_contract_id": self.configuration_contract_id,
                    "provenance_sha256": self.provenance_sha256,
                    "response": "linear-reciprocal-stationary-cpcm",
                    "coordinate_derivative": "complete-fixed-topology-bilinear-vjp",
                    "capabilities": "none-pending-real-stack-gates",
                }.items()
            )
        )


def build_water_radial_gto_cpcm_backend(
    symbols: Sequence[str],
) -> ConjugateRadialGTOFixedTopologyCPCMBackend:
    """Build the exact disabled water-profile backend without free parameters."""

    normalized = tuple(str(symbol) for symbol in symbols)
    return ConjugateRadialGTOFixedTopologyCPCMBackend(
        normalized,
        smd_water_coulomb_radii(normalized),
        dielectric=WATER_CPCM_DIELECTRIC,
        lebedev_order=WATER_CPCM_LEBEDEV_ORDER,
        configuration_contract_id=WATER_CPCM_194_CONFIGURATION_CONTRACT_ID,
    )


def build_water_radial_gto_cpcm_590_candidate(
    symbols: Sequence[str],
) -> ConjugateRadialGTOFixedTopologyCPCMBackend:
    """Build the disabled 590-node water continuum candidate without knobs."""

    normalized = tuple(str(symbol) for symbol in symbols)
    return ConjugateRadialGTOFixedTopologyCPCMBackend(
        normalized,
        smd_water_coulomb_radii(normalized),
        dielectric=WATER_CPCM_DIELECTRIC,
        lebedev_order=WATER_CPCM_590_LEBEDEV_ORDER,
        configuration_contract_id=WATER_CPCM_590_CONFIGURATION_CONTRACT_ID,
    )


def build_water_radial_gto_cpcm_1202_candidate(
    symbols: Sequence[str],
) -> ConjugateRadialGTOFixedTopologyCPCMBackend:
    """Build the disabled 1202-node water continuum candidate without knobs."""

    normalized = tuple(str(symbol) for symbol in symbols)
    return ConjugateRadialGTOFixedTopologyCPCMBackend(
        normalized,
        smd_water_coulomb_radii(normalized),
        dielectric=WATER_CPCM_DIELECTRIC,
        lebedev_order=WATER_CPCM_1202_LEBEDEV_ORDER,
        configuration_contract_id=WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID,
    )


__all__ = [
    "CONJUGATE_RADIAL_CPCM_PROVIDER_ID",
    "ConjugateRadialCPCMState",
    "ConjugateRadialGTOFixedTopologyCPCMBackend",
    "WATER_CPCM_DIELECTRIC",
    "WATER_CPCM_GRID_POINTS_PER_ATOM",
    "WATER_CPCM_590_GRID_POINTS_PER_ATOM",
    "WATER_CPCM_590_LEBEDEV_ORDER",
    "WATER_CPCM_1202_GRID_POINTS_PER_ATOM",
    "WATER_CPCM_1202_LEBEDEV_ORDER",
    "WATER_CPCM_LEBEDEV_ORDER",
    "build_water_radial_gto_cpcm_backend",
    "build_water_radial_gto_cpcm_590_candidate",
    "build_water_radial_gto_cpcm_1202_candidate",
]
