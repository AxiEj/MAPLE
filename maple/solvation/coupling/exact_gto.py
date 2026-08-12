"""Single-width same-basis GTO coupling candidate for Route 2.

This is a conjugate ``B``/``B*`` kernel candidate, not a checkpoint-native
precision adapter.  It reuses the legacy normalized one-width GTO primitive,
but no evidence currently binds that basis to the radial basis and receiver
normalization of a real MACE-POLAR checkpoint.  All PES capabilities therefore
remain disabled and the missing total coordinate derivative fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    GTO_GALERKIN_LAYOUT,
    AtomCenteredL1GTOBasis,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import EXACT_GTO_COUPLING_CANDIDATE_ID
from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1
from maple.solvation.api.units import HARTREE_TO_EV

from .metrics import ATOMIC_L1_PAIRING
from .operator import (
    ConjugateSurfaceMap,
    CoordinateDerivativeUnavailable,
    CouplingProvenance,
    FixedSurfaceGeometryLike,
    configuration_items,
    canonical_metadata_sha256,
    source_files_sha256,
    validate_fixed_surface_geometry,
)
from .spaces import ATOMIC_L1_FIELD_DUAL_SPACE, ATOMIC_L1_SOURCE_SPACE

SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID = EXACT_GTO_COUPLING_CANDIDATE_ID
SINGLE_WIDTH_SAME_BASIS_GTO_PROVIDER_ID = (
    "maple.route2.legacy-gto-galerkin-kernel-candidate.v2"
)
_IMPLEMENTATION_VERSION = "single-width-same-basis-gto-candidate/2"

# This content address pins the mathematical adapter contract, not a checkpoint
# or a claim of physical precision.  Changing the declared implementation
# contract must update its version or descriptor and therefore this digest.
_ALGORITHM_CONTRACT = {
    "algorithm": "legacy AtomCenteredL1GTOBasis.surface_operator times MAPLE Hartree-to-eV",
    "adjoint": "exact discrete transpose transformed only by canonical Q",
    "coordinate_vjp": "unavailable-fail-closed",
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
            "maple.solvation.coupling.exact_gto": here,
            "maple.solvation.coupling.operator": here.with_name("operator.py"),
            "maple.solvation.coupling.metrics": here.with_name("metrics.py"),
            "maple.solvation.coupling.spaces": here.with_name("spaces.py"),
            "maple.solvation.api.units": here.parents[1] / "api/units.py",
            "maple.legacy.gto_galerkin": legacy_root / "gto_galerkin.py",
            "maple.legacy.gto_density": legacy_root / "gto_density.py",
            "maple.legacy.electrostatic_pairing": (
                legacy_root / "electrostatic_pairing.py"
            ),
        }
    )


# Backwards-compatible names are identifiers/aliases only.  They deliberately
# do not preserve the old, overclaiming "precision profile" semantics.
EXACT_GTO_COUPLING_ID = SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID
EXACT_GTO_PROVIDER_ID = SINGLE_WIDTH_SAME_BASIS_GTO_PROVIDER_ID
EXACT_GTO_COUPLING_PROFILE_ID = SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID


@dataclass(frozen=True, slots=True)
class FixedSurfaceGeometry:
    """Immutable concrete implementation of the fixed-surface geometry view."""

    atom_positions_angstrom: np.ndarray
    surface_points_bohr: np.ndarray

    def __post_init__(self) -> None:
        positions, points = validate_fixed_surface_geometry(self)
        positions = np.array(positions, copy=True)
        points = np.array(points, copy=True)
        positions.setflags(write=False)
        points.setflags(write=False)
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(self, "surface_points_bohr", points)


class SingleWidthSameBasisGTOCouplingCandidate:
    """Unadmitted one-width normalized-GTO source/receiver kernel candidate."""

    coupling_id = SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID
    scalar_id = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    provider_id = SINGLE_WIDTH_SAME_BASIS_GTO_PROVIDER_ID
    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE
    pairing_metric = ATOMIC_L1_PAIRING
    capabilities = CapabilityStatus()
    coordinate_derivative_available = False
    checkpoint_native = False
    precision_capability_blocked = True

    def __init__(self, *, sigma_angstrom: float = 1.5) -> None:
        self.basis = AtomCenteredL1GTOBasis((float(sigma_angstrom),))
        source_files = _executable_source_files()
        source_bundle_sha256 = canonical_metadata_sha256(dict(source_files))
        configuration = configuration_items(
            {
                "algorithm_contract_sha256": _ALGORITHM_CONTRACT_SHA256,
                "basis_layout": GTO_GALERKIN_LAYOUT,
                "basis_radial_count": self.basis.radial_count,
                "hartree_to_ev": format(HARTREE_TO_EV, ".17g"),
                "implementation_version": _IMPLEMENTATION_VERSION,
                "pairing_q_sha256": ATOMIC_L1_PAIRING.metadata_hash(),
                "release_binding_status": "unbound-disabled-candidate",
                "sigma_angstrom": format(self.sigma_angstrom, ".17g"),
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
            release_binding_status="unbound-disabled-candidate",
            representation="single-width normalized atom-centered l<=1 GTO candidate",
            source_kernel="legacy.AtomCenteredL1GTOBasis.surface_operator",
            basis_definition=(
                "one analytic normalized Gaussian width shared by all atoms; "
                "not verified against checkpoint-native radial basis"
            ),
            receiver_definition="exact-discrete-transpose-of-source-B under canonical Q",
            coordinate_derivative="total-coordinate-vjp-unavailable-fail-closed",
            checkpoint_attachment="blocked:no-real-checkpoint-basis-and-receiver-canary",
            charged_source_gauge_status=(
                "mean-potential gauge not closed for charged sources; "
                "zero-at-infinity convention required"
            ),
            production_status="kernel-candidate; precision/E/F/H/V/M all blocked",
            configuration=configuration,
        )
        self.configuration_sha256 = self.provenance.configuration_hash()
        self.provenance_sha256 = self.provenance.metadata_hash()
        self._coupling = ConjugateSurfaceMap(matrix_builder=self._matrix)

    @property
    def sigma_angstrom(self) -> float:
        return self.basis.sigmas_angstrom[0]

    def _matrix(self, geometry: FixedSurfaceGeometryLike) -> np.ndarray:
        positions, points = validate_fixed_surface_geometry(geometry)
        # Legacy kernel returns Hartree/e.  The authoritative Route-2 factor is
        # versioned independently of ASE's runtime CODATA table.
        return self.basis.surface_operator(points, positions) * HARTREE_TO_EV

    def metadata(self) -> dict[str, object]:
        return {
            "coupling_id": self.coupling_id,
            "scalar_id": self.scalar_id,
            "provider_id": self.provider_id,
            "configuration_sha256": self.configuration_sha256,
            "provenance_sha256": self.provenance_sha256,
            "sigma_angstrom": self.sigma_angstrom,
            "checkpoint_native": self.checkpoint_native,
            "precision_capability_blocked": self.precision_capability_blocked,
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


# Compatibility alias for provisional code written before the review corrected
# the scientific name.  Metadata and capability flags expose the true status.
ExactGTOCouplingAdapter = SingleWidthSameBasisGTOCouplingCandidate


__all__ = [
    "CoordinateDerivativeUnavailable",
    "CouplingProvenance",
    "EXACT_GTO_COUPLING_ID",
    "EXACT_GTO_COUPLING_PROFILE_ID",
    "EXACT_GTO_PROVIDER_ID",
    "ExactGTOCouplingAdapter",
    "FixedSurfaceGeometry",
    "SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID",
    "SINGLE_WIDTH_SAME_BASIS_GTO_PROVIDER_ID",
    "SingleWidthSameBasisGTOCouplingCandidate",
]
