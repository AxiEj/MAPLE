"""Provider-independent operational state equation with C, Sigma, and U split.

Unlike :mod:`state_equation`, this kernel does not require the electronic
source and model-driving field to have the same dimension.  It is the generic
extension point for MACE-POLAR and future MLIPs that accept nonuniform field
features but emit a distinct electrostatic source representation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.state_registry import (
    SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
)

from .separated_operators import (
    NativeFieldSpace,
    SeparatedContinuumSnapshot,
)
from .spaces import ReducedCoordinates, SourceSpace
from .state_equation import geometry_sha256, provider_behavior_sha256


@runtime_checkable
class SeparatedElectronicResponseProvider(Protocol):
    """Original source response to a categorically separate native field."""

    provider_id: str
    model_profile_id: str
    coupling_id: str
    provenance_sha256: str
    source_space: SourceSpace
    receiver_space: NativeFieldSpace

    def configuration_sha256(self) -> str: ...

    def evaluate_source(self, geometry: Any, field: np.ndarray) -> np.ndarray: ...

    def field_jvp(
        self, geometry: Any, field: np.ndarray, field_direction: np.ndarray
    ) -> np.ndarray: ...

    def field_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray: ...

    def coordinate_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray: ...


def _vector(values: object, *, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({size},).")
    return np.array(result, copy=True)


class SeparatedOperationalStateEquation:
    """Reduced root ``y = T+ M(R, L A^-1 B(c_ref+Ty))``."""

    __slots__ = (
        "_fingerprint_sha256",
        "_sealed",
        "continuum",
        "coordinates",
        "electronic",
        "receiver_space",
        "source_space",
        "state_equation_id",
    )

    def __init__(
        self,
        coordinates: ReducedCoordinates,
        electronic: SeparatedElectronicResponseProvider,
        continuum: SeparatedContinuumSnapshot,
        *,
        state_equation_id: str = SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
    ) -> None:
        if not isinstance(continuum, SeparatedContinuumSnapshot):
            raise TypeError("continuum must be SeparatedContinuumSnapshot.")
        source_space = getattr(coordinates, "source_space", None)
        if not isinstance(source_space, SourceSpace):
            raise TypeError("coordinates must expose one SourceSpace.")
        if not isinstance(getattr(electronic, "source_space", None), SourceSpace):
            raise TypeError("electronic.source_space must be SourceSpace.")
        if not isinstance(
            getattr(electronic, "receiver_space", None), NativeFieldSpace
        ):
            raise TypeError("electronic.receiver_space must be NativeFieldSpace.")
        if source_space.metadata_hash() != electronic.source_space.metadata_hash():
            raise ValueError("coordinate and electronic source spaces do not match.")
        if source_space.metadata_hash() != continuum.source_space.metadata_hash():
            raise ValueError("coordinate and continuum source spaces do not match.")
        if (
            electronic.receiver_space.metadata_hash()
            != continuum.receiver_space.metadata_hash()
        ):
            raise ValueError(
                "electronic and continuum native-field spaces do not match."
            )
        if electronic.coupling_id != continuum.coupling_id:
            raise ValueError("electronic and continuum coupling IDs do not match.")
        if coordinates.atom_count != continuum.atom_count:
            raise ValueError("coordinate and continuum atom counts do not match.")
        if state_equation_id != SEPARATED_OPERATIONAL_STATE_EQUATION_ID:
            raise ValueError("unsupported separated operational state equation ID.")
        configuration = getattr(electronic, "configuration_sha256", None)
        if not callable(configuration):
            raise TypeError("electronic.configuration_sha256 must be callable.")
        configuration()
        continuum.configuration_sha256()

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "electronic", electronic)
        object.__setattr__(self, "continuum", continuum)
        object.__setattr__(self, "source_space", source_space)
        object.__setattr__(self, "receiver_space", continuum.receiver_space)
        object.__setattr__(self, "state_equation_id", state_equation_id)
        object.__setattr__(self, "_fingerprint_sha256", self._current_fingerprint())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("SeparatedOperationalStateEquation is immutable.")
        object.__setattr__(self, name, value)

    @property
    def reduced_dimension(self) -> int:
        return self.coordinates.reduced_dimension

    def _current_fingerprint(self) -> str:
        payload = {
            "state_equation_id": self.state_equation_id,
            "formula": "y=T+M(R,L A^-1 B(c_ref+T y))",
            "coordinate_sha256": self.coordinates.metadata_hash(),
            "source_space_sha256": self.source_space.metadata_hash(),
            "receiver_space_sha256": self.receiver_space.metadata_hash(),
            "electronic": {
                "provider_id": self.electronic.provider_id,
                "model_profile_id": self.electronic.model_profile_id,
                "coupling_id": self.electronic.coupling_id,
                "provenance_sha256": self.electronic.provenance_sha256,
                "configuration_sha256": self.electronic.configuration_sha256(),
                "behavior_sha256": provider_behavior_sha256(
                    self.electronic,
                    (
                        "configuration_sha256",
                        "evaluate_source",
                        "field_jvp",
                        "field_vjp",
                        "coordinate_vjp",
                    ),
                    label="separated_electronic",
                ),
            },
            "continuum": {
                "coupling_id": self.continuum.coupling_id,
                "configuration_sha256": self.continuum.configuration_sha256(),
                "provenance_sha256": self.continuum.provenance_sha256,
                "geometry_sha256": self.continuum.geometry_sha256,
                "topology_sha256": self.continuum.topology_sha256,
            },
            "capabilities": "none",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def fingerprint_sha256(self) -> str:
        current = self._current_fingerprint()
        if current != self._fingerprint_sha256:
            raise RuntimeError("separated operational equation configuration drifted.")
        return current

    def _validate_geometry(self, geometry: object) -> None:
        if geometry_sha256(geometry) != self.continuum.geometry_sha256:
            raise ValueError("geometry does not match the continuum snapshot.")

    def source(self, y: object) -> np.ndarray:
        return self.coordinates.expand(
            _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        )

    def boundary_state(self, y: object) -> np.ndarray:
        return self.continuum.solve_boundary(self.source(y))

    def field(self, y: object) -> np.ndarray:
        return self.continuum.native_field_from_boundary(self.boundary_state(y))

    def mapping(self, geometry: object, y: object) -> np.ndarray:
        self.fingerprint_sha256()
        self._validate_geometry(geometry)
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        candidate = self.electronic.evaluate_source(geometry, self.field(values))
        source = self.source_space.validate(
            candidate,
            atom_count=self.coordinates.atom_count,
            name="electronic source candidate",
        )
        # project_affine is the fail-closed total-charge policy; reduce(project)
        # equals reduce(source) but also exposes a canonical full source state.
        return self.coordinates.reduce(self.coordinates.project_affine(source))

    def residual(self, geometry: object, y: object) -> np.ndarray:
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        return values - self.mapping(geometry, values)

    def state_map_jvp(
        self, geometry: object, y: object, direction: object
    ) -> np.ndarray:
        self.fingerprint_sha256()
        self._validate_geometry(geometry)
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        reduced_direction = _vector(
            direction, size=self.reduced_dimension, name="reduced direction"
        )
        source_direction = self.coordinates.expand_direction(reduced_direction)
        rhs_direction = self.continuum.source_to_boundary @ source_direction.reshape(-1)
        boundary_direction = np.linalg.solve(
            self.continuum.surface_operator, rhs_direction
        )
        field_direction = (
            self.continuum.boundary_to_native_field @ boundary_direction
        ).reshape(self.receiver_space.shape(self.coordinates.atom_count))
        source_direction_out = self.electronic.field_jvp(
            geometry, self.field(values), field_direction
        )
        projected = self.coordinates.project_tangent(
            self.source_space.validate(
                source_direction_out,
                atom_count=self.coordinates.atom_count,
                name="electronic source JVP",
            )
        )
        return self.coordinates.reduce_tangent(projected)

    def residual_jvp(
        self, geometry: object, y: object, direction: object
    ) -> np.ndarray:
        reduced_direction = _vector(
            direction, size=self.reduced_dimension, name="reduced direction"
        )
        return reduced_direction - self.state_map_jvp(geometry, y, reduced_direction)

    def residual_vjp(
        self, geometry: object, y: object, cotangent: object
    ) -> np.ndarray:
        self.fingerprint_sha256()
        self._validate_geometry(geometry)
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        reduced_cotangent = _vector(
            cotangent, size=self.reduced_dimension, name="reduced cotangent"
        )
        source_cotangent = self.coordinates.lift_reduced_cotangent(reduced_cotangent)
        field_cotangent = self.electronic.field_vjp(
            geometry, self.field(values), source_cotangent
        )
        field_cotangent = self.receiver_space.validate(
            field_cotangent,
            atom_count=self.coordinates.atom_count,
            name="native field cotangent",
        ).reshape(-1)
        boundary_cotangent = self.continuum.boundary_to_native_field.T @ field_cotangent
        adjoint_boundary = np.linalg.solve(
            self.continuum.surface_operator.T, boundary_cotangent
        )
        continuum_source_cotangent = (
            self.continuum.source_to_boundary.T @ adjoint_boundary
        ).reshape(self.source_space.shape(self.coordinates.atom_count))
        mapped_cotangent = self.coordinates.reduce_source_cotangent(
            continuum_source_cotangent
        )
        return reduced_cotangent - mapped_cotangent


__all__ = [
    "SeparatedElectronicResponseProvider",
    "SeparatedOperationalStateEquation",
]
