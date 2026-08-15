"""Separated source, boundary, and native-field contracts for Route 2.

The operational MACE-POLAR path does not have one square source/field dual
space.  The checkpoint emits one-width ``l<=1`` source multipoles, while its
receiver consumes two radial ``l<=1`` potential-feature blocks.  This module
represents that fact directly:

``c --B--> boundary rhs``, ``A sigma = B c``, ``u = L sigma``.

No identity between ``L`` and ``B.T`` is assumed.  The smooth harmonic
continuum happens to construct both maps from one eight-channel Gaussian
operator, but the four-channel source embedding remains explicit and
content-addressed.  These contracts admit no scientific capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .exact_gto import mace_polar_learned_source_embedding_matrix
from .metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from .spaces import ATOMIC_L1_SOURCE_SPACE, SourceSpace
from .state_equation import geometry_sha256

SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID = (
    "route2-coupling-mace-polar-source4-harmonic-boundary-nativefield8-v1"
)
MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE_ID = (
    "maple.route2.mace-polar-native-radial-field-space.v1"
)
SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE_ID = (
    "maple.route2.smooth-harmonic-boundary-coefficient-space.v1"
)
SEPARATED_HARMONIC_SNAPSHOT_CONTRACT = "route2-separated-harmonic-continuum-snapshot-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _nonempty(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _string_tuple(
    values: object, *, name: str, require_unique: bool = True
) -> tuple[str, ...]:
    try:
        result = tuple(_nonempty(value, name=name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings.") from exc
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    if require_unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_sha256(values: object, *, name: str, ndim: int = 2) -> str:
    array = np.asarray(values)
    if array.ndim != ndim or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite {ndim}-D array.")
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()


def _finite_matrix(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or min(array.shape) < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite matrix.")
    return np.array(array, copy=True)


def _finite_vector(values: object, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({size},).")
    return np.array(array, copy=True)


@dataclass(frozen=True, slots=True)
class NativeFieldSpace:
    """A model receiver space that is not asserted dual to the source space."""

    space_id: str
    representation: str
    components: tuple[str, ...]
    units: tuple[str, ...]
    gauge: str
    field_convention: str

    def __post_init__(self) -> None:
        for name in ("space_id", "representation", "gauge", "field_convention"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        object.__setattr__(
            self, "components", _string_tuple(self.components, name="components")
        )
        object.__setattr__(
            self,
            "units",
            _string_tuple(self.units, name="units", require_unique=False),
        )
        if len(self.components) != len(self.units):
            raise ValueError(
                "native field components and units must have equal length."
            )

    @property
    def component_count(self) -> int:
        return len(self.components)

    def shape(self, atom_count: int) -> tuple[int, int]:
        if (
            isinstance(atom_count, bool)
            or not isinstance(atom_count, int)
            or atom_count < 1
        ):
            raise ValueError("atom_count must be a positive integer.")
        return (atom_count, self.component_count)

    def validate(
        self, values: object, *, atom_count: int, name: str = "native field"
    ) -> np.ndarray:
        expected = self.shape(atom_count)
        array = np.asarray(values, dtype=float)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {expected}.")
        return np.array(array, copy=True)

    def metadata(self) -> dict[str, object]:
        return {
            "space_id": self.space_id,
            "representation": self.representation,
            "components": list(self.components),
            "units": list(self.units),
            "gauge": self.gauge,
            "field_convention": self.field_convention,
        }

    def metadata_hash(self) -> str:
        return _canonical_sha256(self.metadata())


@dataclass(frozen=True, slots=True)
class BoundaryCoefficientSpace:
    """Geometry-sized continuum coefficient space with stable ordering metadata."""

    space_id: str
    representation: str
    coefficient_order: str
    units: str
    metric_convention: str

    def __post_init__(self) -> None:
        for name in (
            "space_id",
            "representation",
            "coefficient_order",
            "units",
            "metric_convention",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))

    def validate(
        self, values: object, *, dimension: int, name: str = "boundary coefficients"
    ) -> np.ndarray:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("boundary dimension must be a positive integer.")
        return _finite_vector(values, size=dimension, name=name)

    def metadata(self) -> dict[str, object]:
        return {
            "space_id": self.space_id,
            "representation": self.representation,
            "coefficient_order": self.coefficient_order,
            "units": self.units,
            "metric_convention": self.metric_convention,
        }

    def metadata_hash(self) -> str:
        return _canonical_sha256(self.metadata())


MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE = NativeFieldSpace(
    space_id=MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE_ID,
    representation=(
        "physical Gaussian-smoothed potential and Cartesian-gradient receiver "
        "coordinates before checkpoint feature normalization"
    ),
    components=MACE_POLAR_RADIAL_GTO_PAIRING.field_components,
    units=MACE_POLAR_RADIAL_GTO_PAIRING.field_units,
    gauge="continuum-zero-at-infinity; no atom-mean subtraction",
    field_convention=(
        "two radial receiver blocks; no source-duality assertion is made by this space"
    ),
)

SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE = BoundaryCoefficientSpace(
    space_id=SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE_ID,
    representation="atom-major real spherical-harmonic surface coefficients",
    coefficient_order="atom-major/l-ascending/m-ascending/real-harmonic",
    units="continuum stationary coefficient chart",
    metric_convention="Euclidean coefficient covectors; A supplies the energy metric",
)


class SeparatedContinuumSnapshot:
    """Immutable geometry-bound ``A/B/L`` operational continuum snapshot."""

    __slots__ = (
        "_configuration_sha256",
        "_external_provenance_sha256",
        "_provenance_sha256",
        "_sealed",
        "_surface_values",
        "_source_values",
        "_receiver_values",
        "_embedding_values",
        "atom_count",
        "boundary_space",
        "cavity_profile_id",
        "continuum_configuration_sha256",
        "continuum_profile_id",
        "continuum_provider_id",
        "scalar_id",
        "coupling_id",
        "geometry_sha256",
        "receiver_space",
        "source_space",
        "topology_sha256",
    )

    contract_id = SEPARATED_HARMONIC_SNAPSHOT_CONTRACT
    capabilities = ()

    def __init__(
        self,
        *,
        atom_count: int,
        geometry_digest: str,
        continuum_provider_id: str,
        continuum_profile_id: str,
        cavity_profile_id: str,
        scalar_id: str,
        continuum_configuration_sha256: str,
        continuum_provenance_sha256: str,
        topology_sha256: str,
        surface_operator: object,
        source_to_boundary: object,
        boundary_to_native_field: object,
        source_embedding: object,
        source_space: SourceSpace = ATOMIC_L1_SOURCE_SPACE,
        receiver_space: NativeFieldSpace = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
        boundary_space: BoundaryCoefficientSpace = (
            SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE
        ),
        coupling_id: str = SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID,
    ) -> None:
        if (
            isinstance(atom_count, bool)
            or not isinstance(atom_count, int)
            or atom_count < 1
        ):
            raise ValueError("atom_count must be a positive integer.")
        if not isinstance(source_space, SourceSpace):
            raise TypeError("source_space must be SourceSpace.")
        if not isinstance(receiver_space, NativeFieldSpace):
            raise TypeError("receiver_space must be NativeFieldSpace.")
        if not isinstance(boundary_space, BoundaryCoefficientSpace):
            raise TypeError("boundary_space must be BoundaryCoefficientSpace.")

        surface = _finite_matrix(surface_operator, name="surface operator A")
        source = _finite_matrix(source_to_boundary, name="source operator B")
        receiver = _finite_matrix(boundary_to_native_field, name="receiver operator L")
        embedding = _finite_matrix(source_embedding, name="source embedding")
        boundary_dimension = surface.shape[0]
        source_dimension = atom_count * source_space.component_count
        receiver_dimension = atom_count * receiver_space.component_count
        if surface.shape != (boundary_dimension, boundary_dimension):
            raise ValueError("surface operator A must be square.")
        if source.shape != (boundary_dimension, source_dimension):
            raise ValueError(
                "source operator B shape is incompatible with source space."
            )
        if receiver.shape != (receiver_dimension, boundary_dimension):
            raise ValueError(
                "receiver operator L shape is incompatible with field space."
            )
        if embedding.shape != (receiver_dimension, source_dimension):
            raise ValueError("source embedding shape is incompatible with C and U.")
        scale = max(1.0, float(np.linalg.norm(surface, ord="fro")))
        if not np.allclose(surface, surface.T, rtol=0.0, atol=2.0e-12 * scale):
            raise ValueError("surface operator A must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(0.5 * (surface + surface.T))
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 1.0e-12:
            raise ValueError("surface operator A must be strictly positive definite.")
        if np.linalg.matrix_rank(embedding) != source_dimension:
            raise ValueError("source embedding must have full source-column rank.")

        object.__setattr__(self, "atom_count", atom_count)
        object.__setattr__(
            self, "geometry_sha256", _digest(geometry_digest, name="geometry_digest")
        )
        object.__setattr__(
            self,
            "continuum_provider_id",
            _nonempty(continuum_provider_id, name="continuum_provider_id"),
        )
        object.__setattr__(
            self,
            "continuum_profile_id",
            _nonempty(continuum_profile_id, name="continuum_profile_id"),
        )
        object.__setattr__(
            self,
            "cavity_profile_id",
            _nonempty(cavity_profile_id, name="cavity_profile_id"),
        )
        object.__setattr__(self, "scalar_id", _nonempty(scalar_id, name="scalar_id"))
        object.__setattr__(
            self,
            "continuum_configuration_sha256",
            _digest(
                continuum_configuration_sha256,
                name="continuum_configuration_sha256",
            ),
        )
        object.__setattr__(
            self,
            "_external_provenance_sha256",
            _digest(
                continuum_provenance_sha256,
                name="continuum_provenance_sha256",
            ),
        )
        object.__setattr__(
            self, "topology_sha256", _digest(topology_sha256, name="topology_sha256")
        )
        object.__setattr__(
            self, "coupling_id", _nonempty(coupling_id, name="coupling_id")
        )
        object.__setattr__(self, "source_space", source_space)
        object.__setattr__(self, "receiver_space", receiver_space)
        object.__setattr__(self, "boundary_space", boundary_space)
        object.__setattr__(self, "_surface_values", tuple(surface.reshape(-1)))
        object.__setattr__(self, "_source_values", tuple(source.reshape(-1)))
        object.__setattr__(self, "_receiver_values", tuple(receiver.reshape(-1)))
        object.__setattr__(self, "_embedding_values", tuple(embedding.reshape(-1)))
        configuration = self._current_configuration_sha256()
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(
            self,
            "_provenance_sha256",
            _canonical_sha256(
                {
                    "contract_id": self.contract_id,
                    "configuration_sha256": configuration,
                    "continuum_provenance_sha256": self._external_provenance_sha256,
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
            raise AttributeError("SeparatedContinuumSnapshot is immutable.")
        object.__setattr__(self, name, value)

    @property
    def boundary_dimension(self) -> int:
        return len(self._source_values) // (
            self.atom_count * self.source_space.component_count
        )

    @property
    def source_dimension(self) -> int:
        return self.atom_count * self.source_space.component_count

    @property
    def receiver_dimension(self) -> int:
        return self.atom_count * self.receiver_space.component_count

    @property
    def surface_operator(self) -> np.ndarray:
        result = np.asarray(self._surface_values, dtype=float).reshape(
            self.boundary_dimension, self.boundary_dimension
        )
        result = result.copy()
        result.setflags(write=False)
        return result

    @property
    def source_to_boundary(self) -> np.ndarray:
        result = np.asarray(self._source_values, dtype=float).reshape(
            self.boundary_dimension, self.source_dimension
        )
        result = result.copy()
        result.setflags(write=False)
        return result

    @property
    def boundary_to_native_field(self) -> np.ndarray:
        result = np.asarray(self._receiver_values, dtype=float).reshape(
            self.receiver_dimension, self.boundary_dimension
        )
        result = result.copy()
        result.setflags(write=False)
        return result

    @property
    def source_embedding(self) -> np.ndarray:
        result = np.asarray(self._embedding_values, dtype=float).reshape(
            self.receiver_dimension, self.source_dimension
        )
        result = result.copy()
        result.setflags(write=False)
        return result

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._provenance_sha256

    def _current_configuration_sha256(self) -> str:
        return _canonical_sha256(
            {
                "contract_id": self.contract_id,
                "coupling_id": self.coupling_id,
                "atom_count": self.atom_count,
                "geometry_sha256": self.geometry_sha256,
                "continuum_provider_id": self.continuum_provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "scalar_id": self.scalar_id,
                "continuum_configuration_sha256": (self.continuum_configuration_sha256),
                "continuum_provenance_sha256": self._external_provenance_sha256,
                "topology_sha256": self.topology_sha256,
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": self.receiver_space.metadata_hash(),
                "boundary_space_sha256": self.boundary_space.metadata_hash(),
                "surface_operator_sha256": _array_sha256(
                    self.surface_operator, name="surface operator A"
                ),
                "source_to_boundary_sha256": _array_sha256(
                    self.source_to_boundary, name="source operator B"
                ),
                "boundary_to_native_field_sha256": _array_sha256(
                    self.boundary_to_native_field, name="receiver operator L"
                ),
                "source_embedding_sha256": _array_sha256(
                    self.source_embedding, name="source embedding"
                ),
                "surface_equation": "A sigma = B c",
                "native_field_equation": "u = L sigma",
                "continuum_energy": "G0 = -1/2 (B c)^T A^-1 (B c)",
                "source_receiver_duality_assumed": False,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("Separated continuum snapshot configuration drifted.")
        return current

    def source_rhs(self, source: object) -> np.ndarray:
        values = self.source_space.validate(source, atom_count=self.atom_count)
        self.configuration_sha256()
        return self.source_to_boundary @ values.reshape(-1)

    def solve_boundary(self, source: object) -> np.ndarray:
        return np.linalg.solve(self.surface_operator, self.source_rhs(source))

    def native_field_from_boundary(self, boundary_state: object) -> np.ndarray:
        sigma = self.boundary_space.validate(
            boundary_state, dimension=self.boundary_dimension
        )
        field = self.boundary_to_native_field @ sigma
        return self.receiver_space.validate(
            field.reshape(self.receiver_space.shape(self.atom_count)),
            atom_count=self.atom_count,
        )

    def native_field(self, source: object) -> np.ndarray:
        return self.native_field_from_boundary(self.solve_boundary(source))

    def continuum_energy_eV(self, source: object) -> float:
        rhs = self.source_rhs(source)
        sigma = np.linalg.solve(self.surface_operator, rhs)
        result = -0.5 * float(np.vdot(rhs, sigma))
        if not np.isfinite(result):
            raise RuntimeError("Separated continuum energy is non-finite.")
        return result

    def continuum_source_gradient(self, source: object) -> np.ndarray:
        sigma = self.solve_boundary(source)
        result = -(self.source_to_boundary.T @ sigma).reshape(
            self.source_space.shape(self.atom_count)
        )
        return self.source_space.validate(
            result, atom_count=self.atom_count, name="continuum source gradient"
        )


def build_mace_polar_harmonic_separated_snapshot(
    continuum: object,
    geometry: object,
) -> SeparatedContinuumSnapshot:
    """Build the exact ``4 -> boundary -> 8`` view of a harmonic functional."""

    debug = getattr(continuum, "debug_geometry_matrices", None)
    configuration = getattr(continuum, "configuration_sha256", None)
    topology = getattr(continuum, "topology_sha256", None)
    if not callable(debug) or not callable(configuration) or not callable(topology):
        raise TypeError(
            "continuum must expose debug_geometry_matrices(), "
            "configuration_sha256(), and topology_sha256()."
        )
    matrices = debug(geometry)
    if not isinstance(matrices, dict):
        raise TypeError("continuum debug matrices must be returned as a dictionary.")
    surface = _finite_matrix(
        matrices.get("surface_operator"), name="harmonic surface operator"
    )
    source8 = _finite_matrix(
        matrices.get("source_operator"), name="harmonic eight-channel source operator"
    )
    if source8.shape[1] % MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE.component_count:
        raise ValueError("harmonic source operator is not atom-major eight-channel.")
    atom_count = (
        source8.shape[1] // MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE.component_count
    )
    one_atom_embedding = mace_polar_learned_source_embedding_matrix()
    embedding = np.kron(np.eye(atom_count), one_atom_embedding)
    source4 = source8 @ embedding
    receiver8 = -source8.T
    # This is a construction identity, not an assumption imposed on arbitrary
    # operational plugins.  It proves that the current harmonic implementation
    # has one source kernel while keeping C4 and U8 categorically distinct.
    if not np.allclose(source4, source8 @ embedding, rtol=0.0, atol=0.0):
        raise RuntimeError("four-channel harmonic source embedding is inconsistent.")
    provider_id = _nonempty(
        getattr(continuum, "provider_id", None), name="continuum.provider_id"
    )
    continuum_profile_id = _nonempty(
        getattr(continuum, "continuum_profile_id", None),
        name="continuum.continuum_profile_id",
    )
    cavity_profile_id = _nonempty(
        getattr(continuum, "cavity_profile_id", None),
        name="continuum.cavity_profile_id",
    )
    scalar_id = _nonempty(
        getattr(continuum, "scalar_id", None), name="continuum.scalar_id"
    )
    provenance = _digest(
        getattr(continuum, "provenance_sha256", None),
        name="continuum.provenance_sha256",
    )
    return SeparatedContinuumSnapshot(
        atom_count=atom_count,
        geometry_digest=geometry_sha256(geometry),
        continuum_provider_id=provider_id,
        continuum_profile_id=continuum_profile_id,
        cavity_profile_id=cavity_profile_id,
        scalar_id=scalar_id,
        continuum_configuration_sha256=_digest(
            configuration(), name="continuum configuration"
        ),
        continuum_provenance_sha256=provenance,
        topology_sha256=_digest(topology(), name="continuum topology"),
        surface_operator=surface,
        source_to_boundary=source4,
        boundary_to_native_field=receiver8,
        source_embedding=embedding,
    )


__all__ = [
    "BoundaryCoefficientSpace",
    "MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE",
    "MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE_ID",
    "NativeFieldSpace",
    "SEPARATED_HARMONIC_SNAPSHOT_CONTRACT",
    "SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID",
    "SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE",
    "SMOOTH_HARMONIC_BOUNDARY_COEFFICIENT_SPACE_ID",
    "SeparatedContinuumSnapshot",
    "build_mace_polar_harmonic_separated_snapshot",
]
