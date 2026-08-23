"""Pure MACE-POLAR Gaussian-source view of the general-source ddPCM kernel.

MACE-POLAR emits one four-channel ``l<=1`` source whose physical source
kernel is the checkpoint 1.5-A Gaussian density.  Its model-driving receiver
is a distinct eight-channel, two-width Gaussian potential feature space.

The heterogeneous MDP+POLAR continuum already implements the exact
finite-dielectric general-source primal/adjoint.  This module exposes the
mathematically exact pure-source restriction ``permanent=0, induced=c`` as a
standard :class:`SeparatedContinuumProvider`.  It deliberately does not
duplicate ddPCM assembly or silently substitute point multipoles.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2,
)
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID,
    SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE

from .harmonic_ddpcm_hybrid import (
    HarmonicDDPCMHybridSnapshot,
    build_harmonic_ddpcm_hybrid_snapshot,
)


PURE_GAUSSIAN_DDPCM_PROVIDER_ID = (
    "maple.route2.continuum.smooth-harmonic-ddpcm-pure-gto1p5.impl.v2"
)
PURE_GAUSSIAN_DDPCM_PROFILE_ID = (
    "smooth-partition-harmonic-ddpcm-gaussian-general-source-v2"
)
PURE_GAUSSIAN_DDPCM_CAVITY_PROFILE_ID = (
    "smooth-partition-harmonic-ddpcm-cavity-v1"
)
PURE_GAUSSIAN_DDPCM_CONTRACT_ID = (
    "route2-harmonic-ddpcm-pure-gto1p5-general-source-restriction-v2"
)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class HarmonicDDPCMPureGaussianSnapshot:
    """Immutable ``p=0,d=c`` restriction of a general-source ddPCM state."""

    __slots__ = (
        "_configuration_sha256",
        "_general",
        "_provenance_sha256",
        "_sealed",
        "atom_count",
        "boundary_space",
        "cavity_profile_id",
        "continuum_configuration_sha256",
        "continuum_profile_id",
        "continuum_provider_id",
        "coupling_id",
        "geometry_sha256",
        "receiver_space",
        "scalar_id",
        "source_space",
        "topology_sha256",
    )

    contract_id = PURE_GAUSSIAN_DDPCM_CONTRACT_ID
    capabilities = ()
    linear_response = True
    source_kernel = "checkpoint-1.5-A-Gaussian-l<=1"
    receiver_kernel = "checkpoint-native-1.5/3.0-A-Gaussian-potential-features"
    source_receiver_duality_assumed = False

    def __init__(self, general: HarmonicDDPCMHybridSnapshot) -> None:
        if not isinstance(general, HarmonicDDPCMHybridSnapshot):
            raise TypeError("general must be HarmonicDDPCMHybridSnapshot.")
        general.configuration_sha256()
        object.__setattr__(self, "_general", general)
        object.__setattr__(self, "atom_count", general.atom_count)
        object.__setattr__(self, "source_space", ATOMIC_L1_SOURCE_SPACE)
        object.__setattr__(
            self, "receiver_space", MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
        )
        object.__setattr__(
            self, "boundary_space", SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE
        )
        object.__setattr__(
            self, "coupling_id", SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID
        )
        object.__setattr__(
            self,
            "scalar_id",
            OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2,
        )
        object.__setattr__(self, "geometry_sha256", general.geometry_sha256)
        object.__setattr__(self, "topology_sha256", general.topology_sha256)
        object.__setattr__(
            self, "cavity_profile_id", PURE_GAUSSIAN_DDPCM_CAVITY_PROFILE_ID
        )
        object.__setattr__(
            self, "continuum_provider_id", PURE_GAUSSIAN_DDPCM_PROVIDER_ID
        )
        object.__setattr__(
            self, "continuum_profile_id", PURE_GAUSSIAN_DDPCM_PROFILE_ID
        )
        object.__setattr__(
            self,
            "continuum_configuration_sha256",
            general.continuum_configuration_sha256,
        )
        configuration = self._current_configuration_sha256()
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(
            self,
            "_provenance_sha256",
            _sha(
                {
                    "contract": self.contract_id,
                    "configuration_sha256": configuration,
                    "general_source_provenance_sha256": general.provenance_sha256,
                    "implementation_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "capabilities": "none",
                }
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("pure Gaussian ddPCM snapshot is immutable.")
        object.__setattr__(self, name, value)

    @property
    def boundary_dimension(self) -> int:
        return self._general.boundary_dimension

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._provenance_sha256

    def _current_configuration_sha256(self) -> str:
        return _sha(
            {
                "contract": self.contract_id,
                "restriction": "permanent=0; induced=original-MACE-POLAR-source",
                "general_source_configuration_sha256": (
                    self._general.configuration_sha256()
                ),
                "coupling_id": self.coupling_id,
                "scalar_id": self.scalar_id,
                "geometry_sha256": self.geometry_sha256,
                "topology_sha256": self.topology_sha256,
                "continuum_provider_id": self.continuum_provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": self.receiver_space.metadata_hash(),
                "boundary_space_sha256": self.boundary_space.metadata_hash(),
                "source_kernel": self.source_kernel,
                "receiver_kernel": self.receiver_kernel,
                "source_receiver_duality_assumed": False,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("pure Gaussian ddPCM configuration drifted.")
        return current

    def _source(self, source: object, *, name: str) -> np.ndarray:
        return self.source_space.validate(
            source, atom_count=self.atom_count, name=name
        )

    def _zero(self) -> np.ndarray:
        return np.zeros(self.source_space.shape(self.atom_count), dtype=float)

    def solve_boundary(self, source: object) -> np.ndarray:
        return self._general.solve_boundary(
            self._zero(), self._source(source, name="Gaussian source")
        )

    def native_field_from_boundary(self, boundary_state: object) -> np.ndarray:
        return self._general.native_field_from_boundary(boundary_state)

    def native_field(self, source: object) -> np.ndarray:
        return self.native_field_from_boundary(self.solve_boundary(source))

    def source_field_jvp(self, source_direction: object) -> np.ndarray:
        return self._general.induced_field_jvp(
            self._source(source_direction, name="Gaussian source direction")
        )

    def source_field_vjp(self, field_cotangent: object) -> np.ndarray:
        _permanent, induced = self._general.model_field_vjps(field_cotangent)
        return self._source(induced, name="Gaussian source-field VJP")

    def continuum_energy_eV(self, source: object) -> float:
        return self._general.continuum_energy_eV(
            self._zero(), self._source(source, name="Gaussian source")
        )

    def continuum_source_gradient(self, source: object) -> np.ndarray:
        _permanent, induced = self._general.continuum_source_gradients(
            self._zero(), self._source(source, name="Gaussian source")
        )
        return self._source(induced, name="Gaussian continuum source gradient")

    def source_field_position_vjp(
        self, source: object, field_cotangent: object
    ) -> np.ndarray:
        return self._general.model_field_position_vjp(
            self._zero(),
            self._source(source, name="Gaussian source"),
            field_cotangent,
        )

    def continuum_energy_position_gradient(self, source: object) -> np.ndarray:
        return self._general.continuum_energy_position_gradient(
            self._zero(), self._source(source, name="Gaussian source")
        )


def build_mace_polar_gaussian_harmonic_ddpcm_snapshot(
    continuum: object,
    geometry: object,
    *,
    receiver_radial_quadrature_order: int = 128,
) -> HarmonicDDPCMPureGaussianSnapshot:
    """Build the exact pure-Gaussian restriction without duplicating assembly."""

    general = build_harmonic_ddpcm_hybrid_snapshot(
        continuum,
        geometry,
        receiver_radial_quadrature_order=receiver_radial_quadrature_order,
    )
    return HarmonicDDPCMPureGaussianSnapshot(general)


__all__ = [
    "HarmonicDDPCMPureGaussianSnapshot",
    "PURE_GAUSSIAN_DDPCM_CAVITY_PROFILE_ID",
    "PURE_GAUSSIAN_DDPCM_CONTRACT_ID",
    "PURE_GAUSSIAN_DDPCM_PROFILE_ID",
    "PURE_GAUSSIAN_DDPCM_PROVIDER_ID",
    "build_mace_polar_gaussian_harmonic_ddpcm_snapshot",
]
