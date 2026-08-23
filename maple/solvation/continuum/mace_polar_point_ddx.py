"""Point-multipole SourceEmbedding for the frozen MACE-POLAR ddX route.

MACE-POLAR predicts one learned atom-centred ``(q,l=1)`` block.  The radial
adapter embeds that block into the first of two physical Gaussian channels.
For quantitative PCM this module provides a distinct, content-addressed
operational profile: it preserves the learned monopoles and dipoles exactly,
but evaluates their cavity potential as point multipoles.  No coefficient is
fitted and the checkpoint is unchanged.

The choice is intentionally not called the checkpoint electron density.  It
is a ``SourceEmbedding`` from the learned coarse multipoles to the continuum
boundary.  The underlying ddX scalar, its source adjoint and its coordinate
gradient all come from the same point-multipole operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.coupling.exact_gto import MACE_POLAR_RADIAL_GTO_COUPLING_ID
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import geometry_sha256

from .radial_gto_ddx import DDX_CAVITY_PROFILE_ID
from .separated_source_ddx import (
    SeparatedSourceDDXBackend,
    embed_atomic_l1_in_first_radial_channel,
    extract_atomic_l1_first_radial_cotangent,
)


MACE_POLAR_POINT_DDX_PROVIDER_ID = (
    "maple.route2.continuum.ddx-macepolar-point-source-embedding.impl.v1"
)
MACE_POLAR_POINT_DDX_PROFILE_ID = "ddx-ddpcm-macepolar-point-l1-embedding-v1"
MACE_POLAR_POINT_DDX_SCALAR_ID = (
    "route2-operational-macepolar-frozen-point-l1-ddpcm-electrostatic-v1"
)
_STATE_CONTRACT = "macepolar-frozen-point-l1-ddx-state-v1"
_UNUSED_RADIAL_INDICES = (1, 5, 6, 7)
_IMPLEMENTATION_FILES = (
    "continuum/mace_polar_point_ddx.py",
    "continuum/radial_gto_ddx.py",
    "continuum/separated_source_ddx.py",
)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_sha(values: object) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode()
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()


def _implementation_sha256() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[1]
    return tuple(
        (name, hashlib.sha256((root / name).read_bytes()).hexdigest())
        for name in _IMPLEMENTATION_FILES
    )


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array, dtype=float)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MACEPolarPointDDXState:
    """Immutable point-embedded ddX scalar and radial-space derivative."""

    provider_id: str
    continuum_profile_id: str
    cavity_profile_id: str
    scalar_id: str
    configuration_sha256: str
    provenance_sha256: str
    geometry_sha256: str
    cavity_topology_sha256: str
    atom_count: int
    cavity_point_count: int
    source_values: np.ndarray
    field_values: np.ndarray
    polarization_energy_ev: float
    state_hash: str = ""
    capabilities: CapabilityStatus = CapabilityStatus()

    def __post_init__(self) -> None:
        identities = {
            "provider_id": MACE_POLAR_POINT_DDX_PROVIDER_ID,
            "continuum_profile_id": MACE_POLAR_POINT_DDX_PROFILE_ID,
            "cavity_profile_id": DDX_CAVITY_PROFILE_ID,
            "scalar_id": MACE_POLAR_POINT_DDX_SCALAR_ID,
        }
        for name, expected in identities.items():
            if getattr(self, name) != expected:
                raise ValueError(f"point-embedded ddX {name} is invalid.")
        for name in (
            "configuration_sha256",
            "provenance_sha256",
            "geometry_sha256",
            "cavity_topology_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA256 digest.")
        if type(self.atom_count) is not int or self.atom_count < 1:
            raise ValueError("atom_count must be positive.")
        if type(self.cavity_point_count) is not int or self.cavity_point_count < 1:
            raise ValueError("cavity_point_count must be positive.")
        source = _readonly(
            self.source_values,
            shape=(self.atom_count, 8),
            name="source_values",
        )
        field = _readonly(
            self.field_values,
            shape=(self.atom_count, 8),
            name="field_values",
        )
        if np.any(source[:, _UNUSED_RADIAL_INDICES] != 0.0):
            raise ValueError("point-embedded source must have an exact zero second block.")
        if np.any(field[:, _UNUSED_RADIAL_INDICES] != 0.0):
            raise ValueError("point-embedded field must remain in the learned dual block.")
        energy = float(self.polarization_energy_ev)
        if not math.isfinite(energy):
            raise ValueError("polarization_energy_ev must be finite.")
        paired = 0.5 * MACE_POLAR_RADIAL_GTO_PAIRING.pair(source, field)
        if abs(energy - paired) > max(2.0e-10, 2.0e-10 * abs(energy)):
            raise ValueError("point-embedded ddX state violates half coupling.")
        if self.capabilities.enabled_tiers:
            raise ValueError("point-embedded continuum release tiers remain closed.")
        payload = {
            "contract": _STATE_CONTRACT,
            **identities,
            "configuration_sha256": self.configuration_sha256,
            "provenance_sha256": self.provenance_sha256,
            "geometry_sha256": self.geometry_sha256,
            "cavity_topology_sha256": self.cavity_topology_sha256,
            "atom_count": self.atom_count,
            "cavity_point_count": self.cavity_point_count,
            "source_sha256": _array_sha(source),
            "field_sha256": _array_sha(field),
            "polarization_energy_ev": energy,
        }
        expected_hash = _sha(payload)
        if self.state_hash and self.state_hash != expected_hash:
            raise ValueError("point-embedded ddX state hash is inconsistent.")
        object.__setattr__(self, "source_values", source)
        object.__setattr__(self, "field_values", field)
        object.__setattr__(self, "polarization_energy_ev", energy)
        object.__setattr__(self, "state_hash", expected_hash)

    @property
    def source(self) -> np.ndarray:
        return self.source_values

    @property
    def reaction_field(self) -> np.ndarray:
        return self.field_values


class MACEPolarPointEmbeddedDDXBackend:
    """Reciprocal ddPCM map on the learned MACE-POLAR point-l<=1 subspace."""

    __slots__ = (
        "_configuration_sha256",
        "_provenance_sha256",
        "_sealed",
        "_separated",
    )

    provider_id = MACE_POLAR_POINT_DDX_PROVIDER_ID
    continuum_profile_id = MACE_POLAR_POINT_DDX_PROFILE_ID
    cavity_profile_id = DDX_CAVITY_PROFILE_ID
    scalar_id = MACE_POLAR_POINT_DDX_SCALAR_ID
    coupling_id = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE
    pairing = MACE_POLAR_RADIAL_GTO_PAIRING
    capabilities = CapabilityStatus()
    fixed_topology = False
    linear_response = True
    reciprocal = True
    source_dependent_geometry = False
    structurally_rotation_equivariant = False

    def __init__(
        self,
        symbols: Sequence[str],
        cavity_radii_angstrom: object,
        *,
        continuum_model: Literal["pcm"] = "pcm",
        dielectric: float,
        lmax: int,
        n_lebedev: int,
        solver_tolerance: float = 1.0e-12,
        eta: float = 0.1,
        n_proc: int = 1,
    ) -> None:
        if continuum_model != "pcm":
            raise ValueError("the frozen point-embedded profile currently admits ddPCM only.")
        separated = SeparatedSourceDDXBackend(
            symbols,
            cavity_radii_angstrom,
            continuum_model="pcm",
            dielectric=dielectric,
            lmax=lmax,
            n_lebedev=n_lebedev,
            solver_tolerance=solver_tolerance,
            eta=eta,
            n_proc=n_proc,
        )
        configuration = _sha(
            {
                "provider_id": self.provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "scalar_id": self.scalar_id,
                "coupling_id": self.coupling_id,
                "separated_configuration_sha256": separated.configuration_sha256(),
                "source_space_sha256": self.source_space.metadata_hash(),
                "field_space_sha256": self.field_space.metadata_hash(),
                "pairing_sha256": self.pairing.metadata_hash(),
                "source_embedding": (
                    "learned-[q,y,z,x]-to-ddx-point-multipoles; second radial "
                    "source block must be exactly zero"
                ),
                "implementation_files_sha256": _implementation_sha256(),
            }
        )
        provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration,
                "separated_provenance_sha256": separated.provenance_sha256,
                "scientific_status": (
                    "operational-source-embedding-candidate; no fitted parameters; "
                    "finite-ddx-grid-not-structurally-SO3-admitted"
                ),
            }
        )
        object.__setattr__(self, "_separated", separated)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_provenance_sha256", provenance)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarPointEmbeddedDDXBackend is immutable.")
        object.__setattr__(self, name, value)

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._separated.symbols

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        return self._separated.radial_backend.cavity_radii_angstrom

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._provenance_sha256

    def configuration_sha256(self) -> str:
        current = _sha(
            {
                "provider_id": self.provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "scalar_id": self.scalar_id,
                "coupling_id": self.coupling_id,
                "separated_configuration_sha256": (
                    self._separated.configuration_sha256()
                ),
                "source_space_sha256": self.source_space.metadata_hash(),
                "field_space_sha256": self.field_space.metadata_hash(),
                "pairing_sha256": self.pairing.metadata_hash(),
                "source_embedding": (
                    "learned-[q,y,z,x]-to-ddx-point-multipoles; second radial "
                    "source block must be exactly zero"
                ),
                "implementation_files_sha256": _implementation_sha256(),
            }
        )
        if current != self._configuration_sha256:
            raise RuntimeError("point-embedded ddX configuration drifted.")
        return current

    def _source(self, source: object) -> np.ndarray:
        values = self.source_space.validate(source, atom_count=len(self.symbols))
        if np.any(values[:, _UNUSED_RADIAL_INDICES] != 0.0):
            raise ValueError(
                "point-embedded MACE-POLAR ddX requires an exact zero second "
                "radial source block."
            )
        return values

    @staticmethod
    def _learned(values: np.ndarray) -> np.ndarray:
        return extract_atomic_l1_first_radial_cotangent(values)

    @staticmethod
    def _field(permanent_gradient: object) -> np.ndarray:
        return embed_atomic_l1_in_first_radial_channel(permanent_gradient)

    def _build(
        self, geometry: Any, source: object, *, need_coordinate_gradient: bool
    ) -> tuple[MACEPolarPointDDXState, np.ndarray | None]:
        self.configuration_sha256()
        values = self._source(source)
        prepared = self._separated.prepare(geometry, self._learned(values))
        radial_zero = np.zeros_like(values)
        if need_coordinate_gradient:
            separated_state, gradient = (
                prepared.solve_with_fixed_source_energy_derivatives(radial_zero)
            )
        else:
            separated_state = prepared.solve(radial_zero)
            gradient = None
        field = self._field(separated_state.permanent_energy_gradient)
        state = MACEPolarPointDDXState(
            provider_id=self.provider_id,
            continuum_profile_id=self.continuum_profile_id,
            cavity_profile_id=self.cavity_profile_id,
            scalar_id=self.scalar_id,
            configuration_sha256=self.configuration_sha256(),
            provenance_sha256=self.provenance_sha256,
            geometry_sha256=geometry_sha256(geometry),
            cavity_topology_sha256=prepared.cavity_topology_sha256,
            atom_count=len(self.symbols),
            cavity_point_count=separated_state.cavity_point_count,
            source_values=values,
            field_values=field,
            polarization_energy_ev=separated_state.polarization_energy_ev,
        )
        return state, None if gradient is None else np.array(gradient, copy=True)

    def build_state(self, geometry: Any, source: object) -> MACEPolarPointDDXState:
        return self._build(geometry, source, need_coordinate_gradient=False)[0]

    def build_state_with_fixed_source_coordinate_gradient(
        self, geometry: Any, source: object
    ) -> tuple[MACEPolarPointDDXState, np.ndarray]:
        state, gradient = self._build(
            geometry, source, need_coordinate_gradient=True
        )
        if gradient is None:  # pragma: no cover - construction invariant
            raise AssertionError("point-embedded coordinate gradient is unavailable.")
        return state, gradient

    def fixed_source_coordinate_gradient(
        self, geometry: Any, source: object
    ) -> np.ndarray:
        return self.build_state_with_fixed_source_coordinate_gradient(
            geometry, source
        )[1]

    def energy(self, geometry: Any, source: object) -> float:
        return self.build_state(geometry, source).polarization_energy_ev

    def evaluate_field(self, geometry: Any, source: object) -> np.ndarray:
        return np.array(self.build_state(geometry, source).reaction_field, copy=True)

    field = evaluate_field

    def source_jvp(
        self, geometry: Any, source: object, source_direction: object
    ) -> np.ndarray:
        self._source(source)
        return self.evaluate_field(geometry, source_direction)

    def source_vjp(
        self, geometry: Any, source: object, field_cotangent: object
    ) -> np.ndarray:
        self._source(source)
        cotangent = self.field_space.validate(
            field_cotangent,
            atom_count=len(self.symbols),
            name="point-embedded field cotangent",
        )
        projected = embed_atomic_l1_in_first_radial_channel(
            extract_atomic_l1_first_radial_cotangent(cotangent)
        )
        return self.evaluate_field(geometry, projected)

    def coordinate_vjp(
        self,
        geometry: Any,
        source: object,
        field_cotangent: object,
    ) -> np.ndarray:
        values = self._source(source)
        cotangent = self.field_space.validate(
            field_cotangent,
            atom_count=len(self.symbols),
            name="point-embedded coordinate cotangent",
        )
        projected = embed_atomic_l1_in_first_radial_channel(
            extract_atomic_l1_first_radial_cotangent(cotangent)
        )
        combined = values + projected
        return (
            self.fixed_source_coordinate_gradient(geometry, combined)
            - self.fixed_source_coordinate_gradient(geometry, values)
            - self.fixed_source_coordinate_gradient(geometry, projected)
        )


__all__ = [
    "MACE_POLAR_POINT_DDX_PROFILE_ID",
    "MACE_POLAR_POINT_DDX_PROVIDER_ID",
    "MACE_POLAR_POINT_DDX_SCALAR_ID",
    "MACEPolarPointDDXState",
    "MACEPolarPointEmbeddedDDXBackend",
]
