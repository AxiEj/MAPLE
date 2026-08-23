"""Reduced operational root for permanent and induced source branches.

The state variable is the zero-total-charge induced ``l<=1`` source.  A
geometry-only permanent source enters the continuum through a different
kernel.  This keeps model response, continuum polarization, and native field
as separate categories while retaining the matrix-free fixed-point/adjoint
infrastructure used by pure MACE-POLAR.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.state_registry import (
    PERMANENT_INDUCED_SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
)

from .separated_operators import NativeFieldSpace
from .spaces import ReducedCoordinates, SourceSpace
from .state_equation import geometry_sha256, provider_behavior_sha256


@runtime_checkable
class PermanentInducedContinuumProvider(Protocol):
    atom_count: int
    coupling_id: str
    geometry_sha256: str
    topology_sha256: str
    source_space: SourceSpace
    receiver_space: NativeFieldSpace

    @property
    def provenance_sha256(self) -> str: ...

    def configuration_sha256(self) -> str: ...

    def solve_boundary(
        self, permanent_source: object, induced_source: object
    ) -> np.ndarray: ...

    def native_field_from_boundary(self, boundary_state: object) -> np.ndarray: ...

    def native_field(
        self, permanent_source: object, induced_source: object
    ) -> np.ndarray: ...

    def induced_field_jvp(self, induced_direction: object) -> np.ndarray: ...

    def model_field_vjps(
        self, field_cotangent: object
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def continuum_energy_eV(
        self, permanent_source: object, induced_source: object
    ) -> float: ...

    def continuum_source_gradients(
        self, permanent_source: object, induced_source: object
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def continuum_energy_position_gradient(
        self, permanent_source: object, induced_source: object
    ) -> np.ndarray: ...

    def model_field_position_vjp(
        self,
        permanent_source: object,
        induced_source: object,
        field_cotangent: object,
    ) -> np.ndarray: ...


@runtime_checkable
class PermanentInducedResponseModel(Protocol):
    provider_id: str
    model_profile_id: str
    provenance_sha256: str
    source_space: SourceSpace
    receiver_space: NativeFieldSpace

    def configuration_sha256(self) -> str: ...

    def induced_source(
        self, geometry: object, anchor: object, field: object
    ) -> np.ndarray: ...

    def field_jvp(
        self,
        geometry: object,
        anchor: object,
        field: object,
        field_direction: object,
    ) -> np.ndarray: ...

    def field_vjp(
        self,
        geometry: object,
        anchor: object,
        field: object,
        source_cotangent: object,
    ) -> np.ndarray: ...

    def permanent_source_position_vjp(
        self, geometry: object, anchor: object, source_cotangent: object
    ) -> np.ndarray: ...

    def induced_source_position_vjp(
        self,
        geometry: object,
        anchor: object,
        field: object,
        source_cotangent: object,
    ) -> np.ndarray: ...


def _vector(values: object, *, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({size},).")
    return np.array(result, copy=True)


class PermanentInducedOperationalStateEquation:
    """Root ``d=T y=Pi_0[M_resp(R,u(p,d))-M_resp(R,0)]``."""

    __slots__ = (
        "_fingerprint_sha256",
        "_sealed",
        "anchor",
        "continuum",
        "coordinates",
        "hybrid",
        "receiver_space",
        "source_space",
        "state_equation_id",
    )

    def __init__(
        self,
        coordinates: ReducedCoordinates,
        hybrid: PermanentInducedResponseModel,
        anchor: object,
        continuum: PermanentInducedContinuumProvider,
        *,
        state_equation_id: str = (
            PERMANENT_INDUCED_SEPARATED_OPERATIONAL_STATE_EQUATION_ID
        ),
    ) -> None:
        if not isinstance(hybrid, PermanentInducedResponseModel):
            raise TypeError("hybrid must satisfy PermanentInducedResponseModel.")
        if not isinstance(continuum, PermanentInducedContinuumProvider):
            raise TypeError(
                "continuum must satisfy PermanentInducedContinuumProvider."
            )
        source_space = getattr(coordinates, "source_space", None)
        if not isinstance(source_space, SourceSpace):
            raise TypeError("coordinates must expose one SourceSpace.")
        if abs(float(getattr(coordinates, "total_charge", np.nan))) > 1.0e-15:
            raise ValueError("induced-source coordinates must have total_charge=0.")
        if source_space.metadata_hash() != hybrid.source_space.metadata_hash():
            raise ValueError("coordinate and hybrid source spaces do not match.")
        if source_space.metadata_hash() != continuum.source_space.metadata_hash():
            raise ValueError("coordinate and continuum source spaces do not match.")
        if (
            hybrid.receiver_space.metadata_hash()
            != continuum.receiver_space.metadata_hash()
        ):
            raise ValueError("hybrid and continuum native-field spaces do not match.")
        if coordinates.atom_count != continuum.atom_count:
            raise ValueError("coordinate and continuum atom counts do not match.")
        if state_equation_id != (
            PERMANENT_INDUCED_SEPARATED_OPERATIONAL_STATE_EQUATION_ID
        ):
            raise ValueError("unsupported permanent/induced state equation ID.")
        for name in (
            "geometry_sha256",
            "hybrid_configuration_sha256",
            "state_sha256",
            "permanent_source4",
            "response_zero_source4",
        ):
            if not hasattr(anchor, name):
                raise TypeError(f"anchor must expose {name}.")
        if anchor.geometry_sha256 != continuum.geometry_sha256:
            raise ValueError("anchor and continuum geometries do not match.")
        if anchor.hybrid_configuration_sha256 != hybrid.configuration_sha256():
            raise ValueError("anchor and hybrid configurations do not match.")
        permanent = source_space.validate(
            anchor.permanent_source4,
            atom_count=coordinates.atom_count,
            name="permanent source anchor",
        )
        if not np.array_equal(permanent, np.asarray(anchor.permanent_source4)):
            raise ValueError("anchor permanent source did not validate exactly.")
        hybrid.configuration_sha256()
        continuum.configuration_sha256()

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "hybrid", hybrid)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "continuum", continuum)
        object.__setattr__(self, "source_space", source_space)
        object.__setattr__(self, "receiver_space", continuum.receiver_space)
        object.__setattr__(self, "state_equation_id", state_equation_id)
        object.__setattr__(self, "_fingerprint_sha256", self._current_fingerprint())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PermanentInducedOperationalStateEquation is immutable.")
        object.__setattr__(self, name, value)

    @property
    def reduced_dimension(self) -> int:
        return self.coordinates.reduced_dimension

    @property
    def permanent_source(self) -> np.ndarray:
        return self.source_space.validate(
            self.anchor.permanent_source4,
            atom_count=self.coordinates.atom_count,
            name="permanent source",
        )

    def _current_fingerprint(self) -> str:
        payload = {
            "state_equation_id": self.state_equation_id,
            "formula": (
                "d=T y; u=K_R(p,d); r=y-T_plus Pi_0["
                "M_resp(R,u)-M_resp(R,0)]"
            ),
            "coordinate_sha256": self.coordinates.metadata_hash(),
            "anchor_state_sha256": self.anchor.state_sha256,
            "source_space_sha256": self.source_space.metadata_hash(),
            "receiver_space_sha256": self.receiver_space.metadata_hash(),
            "hybrid": {
                "provider_id": self.hybrid.provider_id,
                "model_profile_id": self.hybrid.model_profile_id,
                "provenance_sha256": self.hybrid.provenance_sha256,
                "configuration_sha256": self.hybrid.configuration_sha256(),
                "behavior_sha256": provider_behavior_sha256(
                    self.hybrid,
                    (
                        "configuration_sha256",
                        "induced_source",
                        "field_jvp",
                        "field_vjp",
                        "permanent_source_position_vjp",
                        "induced_source_position_vjp",
                    ),
                    label="permanent_induced_response_model",
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
            raise RuntimeError("permanent/induced state equation drifted.")
        return current

    def _validate_geometry(self, geometry: object) -> None:
        if geometry_sha256(geometry) != self.continuum.geometry_sha256:
            raise ValueError("geometry does not match the continuum snapshot.")
        if geometry_sha256(geometry) != self.anchor.geometry_sha256:
            raise ValueError("geometry does not match the source anchor.")

    def source(self, y: object) -> np.ndarray:
        return self.coordinates.expand(
            _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        )

    def boundary_state(self, y: object) -> np.ndarray:
        return self.continuum.solve_boundary(self.permanent_source, self.source(y))

    def field(self, y: object) -> np.ndarray:
        return self.continuum.native_field_from_boundary(self.boundary_state(y))

    def mapping(self, geometry: object, y: object) -> np.ndarray:
        self.fingerprint_sha256()
        self._validate_geometry(geometry)
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        candidate = self.hybrid.induced_source(
            geometry, self.anchor, self.field(values)
        )
        projected = self.coordinates.project_tangent(
            self.source_space.validate(
                candidate,
                atom_count=self.coordinates.atom_count,
                name="hybrid induced-source candidate",
            )
        )
        return self.coordinates.reduce_tangent(projected)

    def residual(self, geometry: object, y: object) -> np.ndarray:
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        return values - self.mapping(geometry, values)

    def residual_jvp(
        self, geometry: object, y: object, direction: object
    ) -> np.ndarray:
        self.fingerprint_sha256()
        self._validate_geometry(geometry)
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        reduced_direction = _vector(
            direction, size=self.reduced_dimension, name="reduced direction"
        )
        induced_direction = self.coordinates.expand_direction(reduced_direction)
        field_direction = self.continuum.induced_field_jvp(induced_direction)
        response_direction = self.hybrid.field_jvp(
            geometry,
            self.anchor,
            self.field(values),
            field_direction,
        )
        mapped = self.coordinates.reduce_tangent(
            self.coordinates.project_tangent(
                self.source_space.validate(
                    response_direction,
                    atom_count=self.coordinates.atom_count,
                    name="hybrid induced-source JVP",
                )
            )
        )
        return reduced_direction - mapped

    def residual_vjp(
        self, geometry: object, y: object, cotangent: object
    ) -> np.ndarray:
        self.fingerprint_sha256()
        self._validate_geometry(geometry)
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        reduced_cotangent = _vector(
            cotangent, size=self.reduced_dimension, name="reduced cotangent"
        )
        source_cotangent = self.coordinates.lift_reduced_cotangent(
            reduced_cotangent
        )
        field_cotangent = self.hybrid.field_vjp(
            geometry,
            self.anchor,
            self.field(values),
            source_cotangent,
        )
        _permanent_cotangent, induced_cotangent = self.continuum.model_field_vjps(
            field_cotangent
        )
        mapped = self.coordinates.reduce_source_cotangent(induced_cotangent)
        return reduced_cotangent - mapped

    def coordinate_vjp(
        self, geometry: object, y: object, residual_cotangent: object
    ) -> np.ndarray:
        """Apply ``r_R.T`` at fixed induced reduced coordinates."""

        self.fingerprint_sha256()
        self._validate_geometry(geometry)
        values = _vector(y, size=self.reduced_dimension, name="reduced coordinates")
        cotangent = _vector(
            residual_cotangent,
            size=self.reduced_dimension,
            name="residual cotangent",
        )
        induced = self.source(values)
        field = self.field(values)
        response_cotangent = -self.coordinates.lift_reduced_cotangent(cotangent)
        field_cotangent = self.hybrid.field_vjp(
            geometry, self.anchor, field, response_cotangent
        )
        permanent_cotangent, _induced_cotangent = self.continuum.model_field_vjps(
            field_cotangent
        )
        terms = (
            np.asarray(
                self.hybrid.induced_source_position_vjp(
                    geometry,
                    self.anchor,
                    field,
                    response_cotangent,
                ),
                dtype=float,
            ),
            np.asarray(
                self.hybrid.permanent_source_position_vjp(
                    geometry, self.anchor, permanent_cotangent
                ),
                dtype=float,
            ),
            np.asarray(
                self.continuum.model_field_position_vjp(
                    self.permanent_source,
                    induced,
                    field_cotangent,
                ),
                dtype=float,
            ),
        )
        expected = (self.coordinates.atom_count, 3)
        if any(term.shape != expected for term in terms) or any(
            not np.all(np.isfinite(term)) for term in terms
        ):
            raise ValueError(
                "permanent/induced coordinate VJPs must be finite with shape "
                f"{expected}."
            )
        result = sum(terms, start=np.zeros(expected, dtype=float))
        result.setflags(write=False)
        return result


__all__ = [
    "PermanentInducedContinuumProvider",
    "PermanentInducedOperationalStateEquation",
    "PermanentInducedResponseModel",
]
