"""Unregistered exterior local-jet diagnostic coupling.

The available nuclear derivative differentiates only the point-multipole
kernel with fixed surface nodes.  It is intentionally exposed under a partial
derivative name; the public total-coordinate VJP remains fail closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
    point_multipole_potential_position_vjp,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import LOCAL_JET_DIAGNOSTIC_COUPLING_ID
from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1
from maple.solvation.api.units import HARTREE_TO_EV

from .metrics import ATOMIC_L1_PAIRING
from .operator import (
    canonical_metadata_sha256,
    ConjugateSurfaceMap,
    CouplingProvenance,
    FixedSurfaceGeometryLike,
    configuration_items,
    source_files_sha256,
    validate_fixed_surface_geometry,
)
from .spaces import ATOMIC_L1_FIELD_DUAL_SPACE, ATOMIC_L1_SOURCE_SPACE

LOCAL_JET_COUPLING_ID = LOCAL_JET_DIAGNOSTIC_COUPLING_ID
LOCAL_JET_PROVIDER_ID = "maple.route2.legacy-point-multipole-diagnostic.v2"
LOCAL_JET_DIAGNOSTIC_PROFILE_ID = "route2-diagnostic-local-jet-nonpes-v1"
_IMPLEMENTATION_VERSION = "exterior-local-l1-jet-diagnostic/2"
_ALGORITHM_CONTRACT = {
    "algorithm": "legacy exterior point_multipole_potential times MAPLE Hartree-to-eV",
    "adjoint": "exact discrete transpose transformed only by canonical Q",
    "coordinate_vjp": "public total VJP blocked; explicit fixed-surface kernel partial",
    "implementation_version": _IMPLEMENTATION_VERSION,
}
_ALGORITHM_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(_ALGORITHM_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _executable_source_files() -> tuple[tuple[str, str], ...]:
    here = Path(__file__).resolve()
    legacy_root = here.parents[2] / "function/calculator/extra_correction/implicit"
    return source_files_sha256(
        {
            "maple.solvation.coupling.local_jet": here,
            "maple.solvation.coupling.operator": here.with_name("operator.py"),
            "maple.solvation.coupling.metrics": here.with_name("metrics.py"),
            "maple.solvation.coupling.spaces": here.with_name("spaces.py"),
            "maple.solvation.api.units": here.parents[1] / "api/units.py",
            "maple.legacy.gto_density": legacy_root / "gto_density.py",
            "maple.legacy.electrostatic_pairing": (
                legacy_root / "electrostatic_pairing.py"
            ),
        }
    )


# Compatibility name for provisional tests/imports.  It is not a registered
# PES profile and must never be used as a scalar ID.
LOCAL_JET_COUPLING_PROFILE_ID = LOCAL_JET_DIAGNOSTIC_PROFILE_ID


class LocalJetDiagnosticCouplingAdapter:
    """Disabled non-PES diagnostic for the exterior point ``l<=1`` jet."""

    coupling_id = LOCAL_JET_COUPLING_ID
    scalar_id = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    provider_id = LOCAL_JET_PROVIDER_ID
    diagnostic_profile_id = LOCAL_JET_DIAGNOSTIC_PROFILE_ID
    diagnostic_profile_registered_as_pes = False
    diagnostic_profile_enabled = False
    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE
    pairing_metric = ATOMIC_L1_PAIRING
    capabilities = CapabilityStatus()
    coordinate_derivative_available = False
    partial_fixed_surface_kernel_coordinate_vjp_available = True

    def __init__(self) -> None:
        source_files = _executable_source_files()
        source_bundle_sha256 = canonical_metadata_sha256(dict(source_files))
        configuration = configuration_items(
            {
                "algorithm_contract_sha256": _ALGORITHM_CONTRACT_SHA256,
                "basis_definition": "exterior unsmeared point monopole/dipole local jet",
                "hartree_to_ev": format(HARTREE_TO_EV, ".17g"),
                "implementation_version": _IMPLEMENTATION_VERSION,
                "pairing_q_sha256": ATOMIC_L1_PAIRING.metadata_hash(),
                "release_binding_status": "unbound-disabled-diagnostic",
                "source_bundle_sha256": source_bundle_sha256,
                "source_space_sha256": ATOMIC_L1_SOURCE_SPACE.metadata_hash(),
                "field_space_sha256": ATOMIC_L1_FIELD_DUAL_SPACE.metadata_hash(),
            }
        )
        self.provenance = CouplingProvenance(
            coupling_id=self.coupling_id,
            scalar_id=self.scalar_id,
            provider_id=self.provider_id,
            implementation_version=_IMPLEMENTATION_VERSION,
            algorithm_contract_sha256=_ALGORITHM_CONTRACT_SHA256,
            source_files_sha256=source_files,
            tested_git_commit=None,
            release_binding_status="unbound-disabled-diagnostic",
            representation="exterior point monopole/dipole local jet diagnostic",
            source_kernel="legacy.point_multipole_potential",
            basis_definition="unsmeared exterior point l<=1 kernel; diagnostic only",
            receiver_definition="exact-discrete-transpose-of-source-B under canonical Q",
            coordinate_derivative=(
                "total-coordinate-vjp-unavailable; "
                "fixed-surface-kernel-partial-available-by-explicit-name"
            ),
            checkpoint_attachment="none:non-checkpoint-native-diagnostic",
            charged_source_gauge_status=(
                "mean-potential gauge not closed for charged sources; "
                "zero-at-infinity convention required"
            ),
            production_status=(
                "unregistered disabled non-PES diagnostic; E/F/H/V/M all blocked"
            ),
            configuration=configuration,
        )
        self.configuration_sha256 = self.provenance.configuration_hash()
        self.provenance_sha256 = self.provenance.metadata_hash()
        self._coupling = ConjugateSurfaceMap(
            matrix_builder=self._matrix,
            partial_coordinate_vjp=self._partial_fixed_surface_kernel_coordinate_vjp,
        )

    @staticmethod
    def _matrix(geometry: FixedSurfaceGeometryLike) -> np.ndarray:
        positions, points = validate_fixed_surface_geometry(geometry)
        atom_count = len(positions)
        operator = np.empty((len(points), atom_count * 4), dtype=float)
        for atom_index in range(atom_count):
            for component in range(4):
                unit = np.zeros((1, 4), dtype=float)
                unit[0, component] = 1.0
                operator[:, 4 * atom_index + component] = point_multipole_potential(
                    points, positions[atom_index : atom_index + 1], unit
                )
        return operator * HARTREE_TO_EV

    @staticmethod
    def _partial_fixed_surface_kernel_coordinate_vjp(
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        surface_cotangent: np.ndarray,
    ) -> np.ndarray:
        positions, points = validate_fixed_surface_geometry(geometry)
        return (
            point_multipole_potential_position_vjp(
                points, positions, source, surface_cotangent
            )
            * HARTREE_TO_EV
        )

    def metadata(self) -> dict[str, object]:
        return {
            "coupling_id": self.coupling_id,
            "scalar_id": self.scalar_id,
            "provider_id": self.provider_id,
            "diagnostic_profile_id": self.diagnostic_profile_id,
            "diagnostic_profile_registered_as_pes": (
                self.diagnostic_profile_registered_as_pes
            ),
            "diagnostic_profile_enabled": self.diagnostic_profile_enabled,
            "configuration_sha256": self.configuration_sha256,
            "provenance_sha256": self.provenance_sha256,
            "source_space": self.source_space.metadata(),
            "field_space": self.field_space.metadata(),
            "provenance": self.provenance.metadata(),
            "capabilities": {
                "E": self.capabilities.energy,
                "F": self.capabilities.conservative_force,
                "H": self.capabilities.hessian,
                "V": self.capabilities.variational_functional,
                "M": self.capabilities.molecular_dynamics,
            },
        }

    def apply_source(
        self, geometry: FixedSurfaceGeometryLike, source: np.ndarray
    ) -> np.ndarray:
        return self._coupling.apply_source(geometry, source)

    def apply_adjoint(
        self, geometry: FixedSurfaceGeometryLike, surface_cotangent: np.ndarray
    ) -> np.ndarray:
        return self._coupling.apply_adjoint(geometry, surface_cotangent)

    def source_jvp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        source_direction: np.ndarray,
    ) -> np.ndarray:
        return self._coupling.source_jvp(geometry, source, source_direction)

    def source_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        surface_cotangent: np.ndarray,
    ) -> np.ndarray:
        return self._coupling.source_vjp(geometry, source, surface_cotangent)

    def coordinate_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        surface_cotangent: np.ndarray,
    ) -> np.ndarray:
        return self._coupling.coordinate_vjp(geometry, source, surface_cotangent)

    def partial_fixed_surface_kernel_coordinate_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        surface_cotangent: np.ndarray,
    ) -> np.ndarray:
        return self._coupling.partial_fixed_surface_kernel_coordinate_vjp(
            geometry, source, surface_cotangent
        )


__all__ = [
    "LOCAL_JET_COUPLING_ID",
    "LOCAL_JET_COUPLING_PROFILE_ID",
    "LOCAL_JET_DIAGNOSTIC_PROFILE_ID",
    "LOCAL_JET_PROVIDER_ID",
    "LocalJetDiagnosticCouplingAdapter",
]
