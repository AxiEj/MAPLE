"""Analytic same-basis GTO coupling implementations for Route 2.

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
from ase.units import Bohr

from maple.solvation.coupling.gaussian_multipole_derivatives import (
    gaussian_multipole_potential_position_vjp,
    gaussian_multipole_potential_surface_position_vjp,
)
from maple.function.calculator.extra_correction.implicit.gto_field_projection import (
    ExactGTOFieldProjector,
    MACEPolarGTOFieldProjectionSpec,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    GTO_GALERKIN_LAYOUT,
    AtomCenteredL1GTOBasis,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    EXACT_GTO_COUPLING_CANDIDATE_ID,
    MACE_POLAR_RADIAL_GTO_COUPLING_ID,
)
from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1
from maple.solvation.api.units import HARTREE_TO_EV

from .metrics import ATOMIC_L1_PAIRING, MACE_POLAR_RADIAL_GTO_PAIRING
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
from .spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SIGMAS_ANGSTROM,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)

SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID = EXACT_GTO_COUPLING_CANDIDATE_ID
SINGLE_WIDTH_SAME_BASIS_GTO_PROVIDER_ID = (
    "maple.route2.legacy-gto-galerkin-kernel-candidate.v2"
)
_IMPLEMENTATION_VERSION = "single-width-same-basis-gto-candidate/2"
MACE_POLAR_RADIAL_GTO_PROVIDER_ID = (
    "maple.route2.coupling.mace-polar-native-radial-gto.impl.v1"
)
MACE_POLAR_RADIAL_FIELD_TRANSFORM_ID = (
    "maple.route2.model-field.mace-polar-radial-gto-to-native-features.v1"
)
MACE_POLAR_RADIAL_SIGMAS_ANGSTROM = MACE_POLAR_RADIAL_GTO_SIGMAS_ANGSTROM
_RADIAL_IMPLEMENTATION_VERSION = "mace-polar-native-radial-gto/1"
_LEARNED_SOURCE_INDICES = (0, 2, 3, 4)

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

_RADIAL_ALGORITHM_CONTRACT = {
    "algorithm": (
        "two-width AtomCenteredL1GTOBasis.surface_operator in physical unit-"
        "multipole coordinates times MAPLE Hartree-to-eV"
    ),
    "basis_sigmas_angstrom": list(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM),
    "source_layout": "atom-major radial physical q/p channels",
    "adjoint": "exact discrete transpose under identity radial energy pairing",
    "coordinate_vjp": (
        "analytic Gaussian atom-centre derivative plus rigid parent-owned "
        "surface-node derivative"
    ),
    "implementation_version": _RADIAL_IMPLEMENTATION_VERSION,
}
_RADIAL_ALGORITHM_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _RADIAL_ALGORITHM_CONTRACT, sort_keys=True, separators=(",", ":")
    ).encode()
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
            "maple.solvation.coupling.gaussian_multipole_derivatives": (
                here.with_name("gaussian_multipole_derivatives.py")
            ),
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


@dataclass(frozen=True, slots=True)
class OwnedFixedSurfaceGeometry:
    """Fixed-cardinality surface whose nodes translate with parent atoms."""

    atom_positions_angstrom: np.ndarray
    surface_points_bohr: np.ndarray
    surface_parent_atom_indices: np.ndarray

    def __post_init__(self) -> None:
        positions, points = validate_fixed_surface_geometry(self)
        raw_parents = np.asarray(self.surface_parent_atom_indices)
        if raw_parents.shape != (len(points),) or not np.issubdtype(
            raw_parents.dtype, np.integer
        ):
            raise ValueError(
                "surface_parent_atom_indices must be an integer vector with "
                "one owner per surface point."
            )
        parents = raw_parents.astype(np.int64, copy=True)
        if np.any(parents < 0) or np.any(parents >= len(positions)):
            raise ValueError("surface parent indices are outside the atom range.")
        positions = np.array(positions, copy=True)
        points = np.array(points, copy=True)
        positions.setflags(write=False)
        points.setflags(write=False)
        parents.setflags(write=False)
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(self, "surface_points_bohr", points)
        object.__setattr__(self, "surface_parent_atom_indices", parents)


def embed_mace_polar_learned_source(density_coefficients: object) -> np.ndarray:
    """Embed the learned one-width ``(q,l=1)`` source into radial GTO space."""

    values = np.asarray(density_coefficients, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not np.all(np.isfinite(values)):
        raise ValueError(
            "MACE-POLAR learned density must be finite with shape (n_atoms, 4)."
        )
    result = np.zeros((values.shape[0], 8), dtype=float)
    result[:, _LEARNED_SOURCE_INDICES] = values
    return result


def mace_polar_learned_source_embedding_matrix() -> np.ndarray:
    """Return the authoritative one-atom ``4 -> 8`` source embedding ``S``."""

    result = np.zeros((8, 4), dtype=float)
    result[np.asarray(_LEARNED_SOURCE_INDICES), np.arange(4)] = 1.0
    result.setflags(write=False)
    return result


def extract_mace_polar_learned_source_cotangent(
    radial_source_cotangent: object,
) -> np.ndarray:
    """Transpose the fixed learned-source embedding."""

    values = np.asarray(radial_source_cotangent, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
        raise ValueError(
            "Radial source cotangent must be finite with shape (n_atoms, 8)."
        )
    return np.array(values[:, _LEARNED_SOURCE_INDICES], copy=True)


def _radial_source_to_basis_coefficients(source: np.ndarray) -> np.ndarray:
    """Convert the public atom-major radial layout to ``(atom,radial,raw-lm)``."""

    values = np.asarray(source, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
        raise ValueError("Radial source must be finite with shape (n_atoms, 8).")
    coefficients = np.empty((values.shape[0], 2, 4), dtype=float)
    coefficients[:, 0, :] = values[:, (0, 2, 3, 4)]
    coefficients[:, 1, :] = values[:, (1, 5, 6, 7)]
    return coefficients


def _basis_operator_to_radial_layout(
    operator: np.ndarray, atom_count: int
) -> np.ndarray:
    native = np.asarray(operator, dtype=float).reshape(-1, atom_count, 2, 4)
    radial = np.empty((native.shape[0], atom_count, 8), dtype=float)
    radial[:, :, 0] = native[:, :, 0, 0]
    radial[:, :, 1] = native[:, :, 1, 0]
    radial[:, :, 2:5] = native[:, :, 0, 1:4]
    radial[:, :, 5:8] = native[:, :, 1, 1:4]
    return radial.reshape(native.shape[0], atom_count * 8)


def _radial_field_to_smoothed_arrays(
    radial_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(radial_field, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
        raise ValueError("Radial field must be finite with shape (n_atoms, 8).")
    potentials = np.stack((values[:, 0], values[:, 1]), axis=0)
    gradients = np.stack(
        (
            values[:, (4, 2, 3)],
            values[:, (7, 5, 6)],
        ),
        axis=0,
    )
    return potentials, gradients


class MACEPolarRadialFieldTransform:
    """Immutable physical radial-field to checkpoint-feature representation map."""

    __slots__ = (
        "_configuration_sha256",
        "_matrix_values",
        "_provenance",
        "_sealed",
    )
    transform_id = MACE_POLAR_RADIAL_FIELD_TRANSFORM_ID
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE

    def __init__(self, spec: MACEPolarGTOFieldProjectionSpec) -> None:
        if not isinstance(spec, MACEPolarGTOFieldProjectionSpec):
            raise TypeError("spec must be MACEPolarGTOFieldProjectionSpec.")
        if tuple(spec.receiver_sigmas_angstrom) != MACE_POLAR_RADIAL_SIGMAS_ANGSTROM:
            raise ValueError(
                "MACE-POLAR radial transform requires receiver sigmas (1.5, 3.0) A."
            )
        if spec.upstream_matrix.shape != (8, 4):
            raise ValueError(
                "MACE-POLAR radial transform requires eight native features."
            )
        projector = ExactGTOFieldProjector(spec)
        matrix = np.empty((8, 8), dtype=float)
        for column in range(8):
            basis = np.zeros((1, 8), dtype=float)
            basis[0, column] = 1.0
            potentials, gradients = _radial_field_to_smoothed_arrays(basis)
            matrix[:, column] = projector.project_smoothed_fields(
                potentials,
                gradients,
                scalar_potential_gauge_reference_ev=0.0,
            )[0]
        if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) != 8:
            raise ValueError(
                "Checkpoint receiver projection is not invertible on radial field space."
            )
        provenance = {
            "transform_id": self.transform_id,
            "contract": "physical-radial-field-to-native-feature-linear-map-v1",
            "projection_spec": spec.provenance,
            "source_space_sha256": self.source_space.metadata_hash(),
            "field_space_sha256": self.field_space.metadata_hash(),
            "matrix_sha256": canonical_metadata_sha256(matrix.tolist()),
            "gauge": "continuum-zero-at-infinity; no atom-mean subtraction",
        }
        object.__setattr__(
            self, "_matrix_values", tuple(float(value) for value in matrix.reshape(-1))
        )
        object.__setattr__(self, "_provenance", provenance)
        object.__setattr__(
            self, "_configuration_sha256", canonical_metadata_sha256(provenance)
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarRadialFieldTransform is immutable.")
        object.__setattr__(self, name, value)

    @property
    def matrix(self) -> np.ndarray:
        result = np.asarray(self._matrix_values, dtype=float).reshape(8, 8).copy()
        result.setflags(write=False)
        return result

    @property
    def provenance(self) -> dict[str, object]:
        return json.loads(json.dumps(self._provenance))

    def configuration_sha256(self) -> str:
        current = canonical_metadata_sha256(self._provenance)
        if current != self._configuration_sha256:
            raise RuntimeError("Radial field-transform configuration drifted.")
        return current

    def to_model_features(self, radial_field: object) -> np.ndarray:
        values = np.asarray(radial_field, dtype=float)
        if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
            raise ValueError("Radial field must be finite with shape (n_atoms, 8).")
        self.configuration_sha256()
        return values @ self.matrix.T

    jvp = to_model_features

    def vjp(self, feature_cotangent: object) -> np.ndarray:
        values = np.asarray(feature_cotangent, dtype=float)
        if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
            raise ValueError(
                "Model-feature cotangent must be finite with shape (n_atoms, 8)."
            )
        self.configuration_sha256()
        return values @ self.matrix


class MACEPolarRadialGTOCoupling:
    """Exact conjugate two-width GTO coupling in physical radial coordinates."""

    __slots__ = (
        "_basis",
        "_configuration_sha256",
        "_coupling",
        "_provenance",
        "_provenance_sha256",
        "_sealed",
    )
    coupling_id = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    scalar_id = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    provider_id = MACE_POLAR_RADIAL_GTO_PROVIDER_ID
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE
    pairing_metric = MACE_POLAR_RADIAL_GTO_PAIRING
    capabilities = CapabilityStatus()
    coordinate_derivative_available = True
    checkpoint_native_receiver_basis = True

    def __init__(self) -> None:
        basis = AtomCenteredL1GTOBasis(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM)
        source_files = _executable_source_files()
        configuration = configuration_items(
            {
                "algorithm_contract_sha256": _RADIAL_ALGORITHM_CONTRACT_SHA256,
                "basis_layout": GTO_GALERKIN_LAYOUT,
                "basis_sigmas_angstrom": repr(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM),
                "hartree_to_ev": format(HARTREE_TO_EV, ".17g"),
                "implementation_version": _RADIAL_IMPLEMENTATION_VERSION,
                "pairing_q_sha256": self.pairing_metric.metadata_hash(),
                "release_binding_status": "unbound-force-gates-open",
                "source_bundle_sha256": canonical_metadata_sha256(dict(source_files)),
                "source_space_sha256": self.source_space.metadata_hash(),
                "field_space_sha256": self.field_space.metadata_hash(),
            }
        )
        provenance = CouplingProvenance(
            coupling_id=self.coupling_id,
            scalar_id=self.scalar_id,
            provider_id=self.provider_id,
            implementation_version=_RADIAL_IMPLEMENTATION_VERSION,
            algorithm_contract_sha256=_RADIAL_ALGORITHM_CONTRACT_SHA256,
            source_files_sha256=source_files,
            tested_git_commit=None,
            release_binding_status="unbound-force-gates-open",
            representation=(
                "physical two-width atom-centred l<=1 unit-multipole GTO space"
            ),
            source_kernel="legacy.AtomCenteredL1GTOBasis.surface_operator",
            basis_definition=(
                "analytic normalized sigma=(1.5,3.0) angstrom Gaussian "
                "monopole/dipole basis"
            ),
            receiver_definition=(
                "exact discrete transpose in the identical physical radial basis; "
                "checkpoint normalization is a separate invertible transform"
            ),
            coordinate_derivative=(
                "complete for rigid parent-owned fixed-cardinality surface nodes"
            ),
            checkpoint_attachment=(
                "receiver widths/layout represented; release canary still required"
            ),
            charged_source_gauge_status="zero-at-infinity fixed; no gauge subtraction",
            production_status="mathematical coupling complete; E/F/H/V/M remain blocked",
            configuration=configuration,
        )
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(
            self, "_configuration_sha256", provenance.configuration_hash()
        )
        object.__setattr__(self, "_provenance", provenance)
        object.__setattr__(self, "_provenance_sha256", provenance.metadata_hash())
        object.__setattr__(
            self,
            "_coupling",
            ConjugateSurfaceMap(
                matrix_builder=self._matrix,
                source_space=self.source_space,
                field_space=self.field_space,
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarRadialGTOCoupling is immutable.")
        object.__setattr__(self, name, value)

    @property
    def basis(self) -> AtomCenteredL1GTOBasis:
        return self._basis

    @property
    def provenance(self) -> CouplingProvenance:
        return self._provenance

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256

    def configuration_sha256(self) -> str:
        if self._provenance.configuration_hash() != self._configuration_sha256:
            raise RuntimeError("Radial GTO coupling configuration drifted.")
        return self._configuration_sha256

    def _matrix(self, geometry: FixedSurfaceGeometryLike) -> np.ndarray:
        positions, points = validate_fixed_surface_geometry(geometry)
        native = self.basis.surface_operator(points, positions)
        return _basis_operator_to_radial_layout(native, len(positions)) * HARTREE_TO_EV

    def apply_source(
        self, geometry: FixedSurfaceGeometryLike, source: np.ndarray
    ) -> np.ndarray:
        self.configuration_sha256()
        return self._coupling.apply_source(geometry, source)

    def surface_operator(self, geometry: FixedSurfaceGeometryLike) -> np.ndarray:
        """Return the one matrix used by both the source map and its adjoint.

        Fixed-geometry continuum solvers may retain this immutable matrix while
        applying many source directions.  This is an execution optimization of
        the public ``apply_source``/``apply_adjoint`` pair, not a second
        receiver implementation.
        """

        self.configuration_sha256()
        return self._coupling.matrix(geometry)

    def apply_adjoint(
        self, geometry: FixedSurfaceGeometryLike, surface_cotangent: np.ndarray
    ) -> np.ndarray:
        self.configuration_sha256()
        return self._coupling.apply_adjoint(geometry, surface_cotangent)

    def source_jvp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        source_direction: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        return self._coupling.source_jvp(geometry, source, source_direction)

    def source_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        surface_cotangent: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        return self._coupling.source_vjp(geometry, source, surface_cotangent)

    def coordinate_vjp(
        self,
        geometry: FixedSurfaceGeometryLike,
        source: np.ndarray,
        surface_cotangent: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        positions, points = validate_fixed_surface_geometry(geometry)
        parents = np.asarray(getattr(geometry, "surface_parent_atom_indices", None))
        if (
            parents.shape != (len(points),)
            or not np.issubdtype(parents.dtype, np.integer)
            or np.any(parents < 0)
            or np.any(parents >= len(positions))
        ):
            raise CoordinateDerivativeUnavailable(
                "Radial GTO total coordinate VJP requires one valid parent atom "
                "for every rigid surface node."
            )
        values = self.source_space.validate(source, atom_count=len(positions))
        cotangent = np.asarray(surface_cotangent, dtype=float)
        if cotangent.shape != (len(points),) or not np.all(np.isfinite(cotangent)):
            raise ValueError(
                "surface_cotangent must be finite with one value per surface point."
            )
        coefficients = _radial_source_to_basis_coefficients(values)
        atom_vjp = np.zeros_like(positions)
        surface_vjp_per_bohr = np.zeros_like(points)
        for radial_index, sigma in enumerate(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM):
            radial_coefficients = coefficients[:, radial_index, :]
            atom_vjp += gaussian_multipole_potential_position_vjp(
                points,
                positions,
                radial_coefficients,
                cotangent,
                sigma_angstrom=sigma,
            )
            surface_vjp_per_bohr += gaussian_multipole_potential_surface_position_vjp(
                points,
                positions,
                radial_coefficients,
                cotangent,
                sigma_angstrom=sigma,
            )
        np.add.at(atom_vjp, parents.astype(np.int64), surface_vjp_per_bohr / Bohr)
        result = atom_vjp * HARTREE_TO_EV
        if result.shape != positions.shape or not np.all(np.isfinite(result)):
            raise RuntimeError("Radial GTO coordinate VJP is invalid.")
        return result


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
    "MACE_POLAR_RADIAL_FIELD_TRANSFORM_ID",
    "MACE_POLAR_RADIAL_GTO_COUPLING_ID",
    "MACE_POLAR_RADIAL_GTO_PROVIDER_ID",
    "MACE_POLAR_RADIAL_SIGMAS_ANGSTROM",
    "MACEPolarRadialFieldTransform",
    "MACEPolarRadialGTOCoupling",
    "OwnedFixedSurfaceGeometry",
    "SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID",
    "SINGLE_WIDTH_SAME_BASIS_GTO_PROVIDER_ID",
    "SingleWidthSameBasisGTOCouplingCandidate",
    "embed_mace_polar_learned_source",
    "extract_mace_polar_learned_source_cotangent",
    "mace_polar_learned_source_embedding_matrix",
]
