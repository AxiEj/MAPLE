"""MACE-MDP permanent moments plus MACE-POLAR induced source response.

This research-only adapter keeps the two scientific roles explicit:

* MACE-MDP supplies the zero-field permanent atom-centred ``(q, p)`` source.
* MACE-POLAR supplies only the field-induced increment
  ``M(R, u) - M(R, 0)``.

The permanent and induced sources may use different source-to-boundary
kernels.  Consequently this object deliberately does *not* implement the
ordinary :class:`SeparatedElectronicResponseProvider` protocol, whose single
``B`` matrix would silently erase that distinction.  The current admitted
research construction uses exterior point multipoles for the permanent term
and the checkpoint's 1.5-A Gaussian density for the induced term.

No common variational functional or coordinate derivative is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.coupling.operator import (
    CoordinateDerivativeUnavailable,
    canonical_metadata_sha256,
)
from maple.solvation.coupling.separated_operators import NativeFieldSpace
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE, SourceSpace
from maple.solvation.coupling.state_equation import geometry_sha256

from .base import atom_count
from .mace_mdp import MACE_MDPMomentAdapter
from .mace_polar_separated import MACEPolarOriginalSourceNativeFieldAdapter

MACE_MDP_PERMANENT_SOURCE_PROVIDER_ID = (
    "maple.route2.model.mace-mdp-permanent-source4.impl.v1"
)
MACE_MDP_POLAR_HYBRID_PROVIDER_ID = (
    "maple.route2.model.mace-mdp-permanent-mace-polar-induced.impl.v1"
)
MACE_MDP_POLAR_HYBRID_PROFILE_ID = (
    "route2-research-mace-mdp-point-permanent-macepolar-gto-induced-v1"
)
MACE_MDP_POLAR_HYBRID_CONTRACT = "mace-mdp-permanent-plus-mace-polar-field-increment-v1"


@runtime_checkable
class PermanentAtomicL1SourceProvider(Protocol):
    """Geometry-only permanent source used as an affine response anchor."""

    provider_id: str
    model_profile_id: str
    source_space: SourceSpace

    def configuration_sha256(self) -> str: ...

    def evaluate_source(self, geometry: Any) -> np.ndarray: ...


@runtime_checkable
class FieldResponsiveAtomicL1SourceProvider(Protocol):
    """Field-responsive source whose zero-field value is subtracted exactly."""

    provider_id: str
    model_profile_id: str
    provenance_sha256: str
    source_space: SourceSpace
    receiver_space: NativeFieldSpace
    long_range_evaluator_profile: str

    def configuration_sha256(self) -> str: ...

    def vacuum_energy_ev(self, geometry: Any) -> float: ...

    def evaluate_source(self, geometry: Any, field: np.ndarray) -> np.ndarray: ...

    def field_jvp(
        self, geometry: Any, field: np.ndarray, field_direction: np.ndarray
    ) -> np.ndarray: ...

    def field_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray: ...


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True, slots=True)
class PermanentInducedSourceAnchor:
    """Immutable geometry-bound affine source anchor."""

    geometry_sha256: str
    hybrid_configuration_sha256: str
    permanent_source4: np.ndarray
    response_zero_source4: np.ndarray
    state_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("geometry_sha256", "hybrid_configuration_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest.")
        permanent = np.asarray(self.permanent_source4, dtype=float)
        zero = np.asarray(self.response_zero_source4, dtype=float)
        if permanent.ndim != 2 or permanent.shape[1] != 4:
            raise ValueError("permanent_source4 must have shape (N,4).")
        permanent = _readonly(
            permanent, shape=permanent.shape, name="permanent_source4"
        )
        zero = _readonly(zero, shape=permanent.shape, name="response_zero_source4")
        payload = {
            "contract": MACE_MDP_POLAR_HYBRID_CONTRACT,
            "geometry_sha256": self.geometry_sha256,
            "hybrid_configuration_sha256": self.hybrid_configuration_sha256,
            "permanent_source4": permanent.tolist(),
            "response_zero_source4": zero.tolist(),
        }
        expected = canonical_metadata_sha256(payload)
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("anchor state_sha256 does not match its content.")
        object.__setattr__(self, "permanent_source4", permanent)
        object.__setattr__(self, "response_zero_source4", zero)
        object.__setattr__(self, "state_sha256", expected)

    @property
    def atom_count(self) -> int:
        return int(self.permanent_source4.shape[0])

    @property
    def total_charge_e(self) -> float:
        return float(np.sum(self.permanent_source4[:, 0]))


class MACE_MDPPermanentSourceAdapter:
    """Expose the frozen MACE-MDP moment state as a geometry-only source."""

    __slots__ = ("_base", "_configuration_sha256", "_sealed")

    provider_id = MACE_MDP_PERMANENT_SOURCE_PROVIDER_ID
    model_profile_id = MACE_MDP_POLAR_HYBRID_PROFILE_ID
    source_space = ATOMIC_L1_SOURCE_SPACE
    capabilities = ()

    def __init__(self, base: MACE_MDPMomentAdapter) -> None:
        if not isinstance(base, MACE_MDPMomentAdapter):
            raise TypeError("base must be MACE_MDPMomentAdapter.")
        base.configuration_sha256()
        object.__setattr__(self, "_base", base)
        object.__setattr__(
            self,
            "_configuration_sha256",
            canonical_metadata_sha256(
                {
                    "contract": "mace-mdp-permanent-source4-adapter-v1",
                    "provider_id": self.provider_id,
                    "model_profile_id": self.model_profile_id,
                    "base_provider_id": base.provider_id,
                    "base_model_profile_id": base.model_profile_id,
                    "base_configuration_sha256": base.configuration_sha256(),
                    "checkpoint_sha256": base.checkpoint_sha256,
                    "source_space_sha256": self.source_space.metadata_hash(),
                    "capabilities": "none",
                }
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACE_MDPPermanentSourceAdapter is immutable.")
        object.__setattr__(self, name, value)

    def configuration_sha256(self) -> str:
        self._base.configuration_sha256()
        return self._configuration_sha256

    def evaluate_source(self, geometry: object) -> np.ndarray:
        self.configuration_sha256()
        return self.source_space.validate(
            self._base.evaluate(geometry).source4_raw_l1,
            atom_count=atom_count(geometry),
            name="MACE-MDP permanent source",
        )


class PermanentAnchoredInducedSourceModel:
    """Compose a permanent source with a field-induced source increment."""

    __slots__ = (
        "_configuration_sha256",
        "_permanent",
        "_response",
        "_sealed",
        "model_profile_id",
        "provenance_sha256",
        "provider_id",
    )

    source_space = ATOMIC_L1_SOURCE_SPACE
    capabilities = ()
    variational_functional_admitted = False
    coordinate_derivative_available = False
    permanent_source_kernel = "exterior point monopoles and dipoles"
    induced_source_kernel = "checkpoint 1.5-A normalized Gaussian multipoles"

    def __init__(
        self,
        permanent: PermanentAtomicL1SourceProvider,
        response: FieldResponsiveAtomicL1SourceProvider,
        *,
        provider_id: str = MACE_MDP_POLAR_HYBRID_PROVIDER_ID,
        model_profile_id: str = MACE_MDP_POLAR_HYBRID_PROFILE_ID,
    ) -> None:
        for owner, names in (
            (permanent, ("configuration_sha256", "evaluate_source")),
            (
                response,
                (
                    "configuration_sha256",
                    "vacuum_energy_ev",
                    "evaluate_source",
                    "field_jvp",
                    "field_vjp",
                ),
            ),
        ):
            for name in names:
                if not callable(getattr(owner, name, None)):
                    raise TypeError(f"hybrid provider requires callable {name}().")
        permanent_space = getattr(permanent, "source_space", None)
        response_space = getattr(response, "source_space", None)
        receiver_space = getattr(response, "receiver_space", None)
        if not isinstance(permanent_space, SourceSpace) or not isinstance(
            response_space, SourceSpace
        ):
            raise TypeError("hybrid source providers must expose SourceSpace.")
        if permanent_space.metadata_hash() != self.source_space.metadata_hash():
            raise ValueError("permanent source space is not canonical atom-l<=1.")
        if response_space.metadata_hash() != self.source_space.metadata_hash():
            raise ValueError("responsive source space is not canonical atom-l<=1.")
        if not isinstance(receiver_space, NativeFieldSpace):
            raise TypeError("responsive provider must expose NativeFieldSpace.")
        permanent.configuration_sha256()
        response.configuration_sha256()
        object.__setattr__(self, "_permanent", permanent)
        object.__setattr__(self, "_response", response)
        object.__setattr__(self, "provider_id", str(provider_id))
        object.__setattr__(self, "model_profile_id", str(model_profile_id))
        provenance = canonical_metadata_sha256(
            {
                "contract": MACE_MDP_POLAR_HYBRID_CONTRACT,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "permanent_provider_id": getattr(permanent, "provider_id", None),
                "permanent_model_profile_id": getattr(
                    permanent, "model_profile_id", None
                ),
                "permanent_configuration_sha256": permanent.configuration_sha256(),
                "response_provider_id": getattr(response, "provider_id", None),
                "response_model_profile_id": getattr(
                    response, "model_profile_id", None
                ),
                "response_provenance_sha256": getattr(
                    response, "provenance_sha256", None
                ),
                "response_configuration_sha256": response.configuration_sha256(),
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": receiver_space.metadata_hash(),
                "permanent_source_kernel": self.permanent_source_kernel,
                "induced_source_kernel": self.induced_source_kernel,
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "capabilities": "none",
            }
        )
        object.__setattr__(self, "provenance_sha256", provenance)
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PermanentAnchoredInducedSourceModel is immutable.")
        object.__setattr__(self, name, value)

    @property
    def receiver_space(self) -> NativeFieldSpace:
        return self._response.receiver_space

    @property
    def long_range_evaluator_profile(self) -> str:
        return str(self._response.long_range_evaluator_profile)

    def _current_configuration(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": MACE_MDP_POLAR_HYBRID_CONTRACT,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "provenance_sha256": self.provenance_sha256,
                "permanent_configuration_sha256": (
                    self._permanent.configuration_sha256()
                ),
                "response_configuration_sha256": (
                    self._response.configuration_sha256()
                ),
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": self.receiver_space.metadata_hash(),
                "long_range_evaluator_profile": self.long_range_evaluator_profile,
                "permanent_source_kernel": self.permanent_source_kernel,
                "induced_source_kernel": self.induced_source_kernel,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("permanent/induced hybrid configuration drifted.")
        return current

    def prepare(self, geometry: object) -> PermanentInducedSourceAnchor:
        self.configuration_sha256()
        count = atom_count(geometry)
        permanent = self.source_space.validate(
            self._permanent.evaluate_source(geometry),
            atom_count=count,
            name="permanent source",
        )
        zero_field = np.zeros(self.receiver_space.shape(count), dtype=float)
        response_zero = self.source_space.validate(
            self._response.evaluate_source(geometry, zero_field),
            atom_count=count,
            name="zero-field responsive source",
        )
        return PermanentInducedSourceAnchor(
            geometry_sha256=geometry_sha256(geometry),
            hybrid_configuration_sha256=self.configuration_sha256(),
            permanent_source4=permanent,
            response_zero_source4=response_zero,
        )

    def vacuum_energy_ev(self, geometry: object) -> float:
        """Return the MACE-POLAR zero-field energy for the operational ledger."""

        self.configuration_sha256()
        value = float(self._response.vacuum_energy_ev(geometry))
        if not np.isfinite(value):
            raise RuntimeError("hybrid vacuum energy is non-finite.")
        return value

    def _validate_anchor(
        self, geometry: object, anchor: PermanentInducedSourceAnchor
    ) -> int:
        self.configuration_sha256()
        if not isinstance(anchor, PermanentInducedSourceAnchor):
            raise TypeError("anchor must be PermanentInducedSourceAnchor.")
        if anchor.hybrid_configuration_sha256 != self.configuration_sha256():
            raise ValueError("anchor belongs to a different hybrid configuration.")
        if anchor.geometry_sha256 != geometry_sha256(geometry):
            raise ValueError("anchor geometry does not match the requested geometry.")
        count = atom_count(geometry)
        if anchor.atom_count != count:
            raise ValueError("anchor atom count does not match the geometry.")
        return count

    def induced_source(
        self,
        geometry: object,
        anchor: PermanentInducedSourceAnchor,
        field: object,
    ) -> np.ndarray:
        count = self._validate_anchor(geometry, anchor)
        field_values = self.receiver_space.validate(
            field, atom_count=count, name="native receiver field"
        )
        response = self.source_space.validate(
            self._response.evaluate_source(geometry, field_values),
            atom_count=count,
            name="field-responsive source",
        )
        induced = response - anchor.response_zero_source4
        charge_drift = abs(float(np.sum(induced[:, 0])))
        if charge_drift > 1.0e-8:
            raise RuntimeError(
                "field-induced source changed the checkpoint total charge."
            )
        return induced

    def total_source(
        self,
        geometry: object,
        anchor: PermanentInducedSourceAnchor,
        field: object,
    ) -> np.ndarray:
        return self.source_space.validate(
            anchor.permanent_source4 + self.induced_source(geometry, anchor, field),
            atom_count=anchor.atom_count,
            name="hybrid total source",
        )

    def field_jvp(
        self,
        geometry: object,
        anchor: PermanentInducedSourceAnchor,
        field: object,
        field_direction: object,
    ) -> np.ndarray:
        count = self._validate_anchor(geometry, anchor)
        field_values = self.receiver_space.validate(field, atom_count=count)
        direction = self.receiver_space.validate(
            field_direction, atom_count=count, name="native field direction"
        )
        return self.source_space.validate(
            self._response.field_jvp(geometry, field_values, direction),
            atom_count=count,
            name="hybrid induced-source JVP",
        )

    def field_vjp(
        self,
        geometry: object,
        anchor: PermanentInducedSourceAnchor,
        field: object,
        source_cotangent: object,
    ) -> np.ndarray:
        count = self._validate_anchor(geometry, anchor)
        field_values = self.receiver_space.validate(field, atom_count=count)
        cotangent = self.source_space.validate(
            source_cotangent, atom_count=count, name="source cotangent"
        )
        return self.receiver_space.validate(
            self._response.field_vjp(geometry, field_values, cotangent),
            atom_count=count,
            name="hybrid native-field VJP",
        )

    def coordinate_vjp(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise CoordinateDerivativeUnavailable(
            "The hybrid total coordinate VJP requires MACE-MDP atomic q/p "
            "coordinate derivatives and distinct point/GTO kernel derivatives."
        )

    def conditioned_raw_energy_ev(self, geometry: object, field: object) -> float:
        method = getattr(self._response, "conditioned_raw_energy_ev", None)
        if not callable(method):
            raise NotImplementedError(
                "responsive provider exposes no conditioned raw-energy scalar."
            )
        count = atom_count(geometry)
        return float(
            method(geometry, self.receiver_space.validate(field, atom_count=count))
        )


def build_mace_mdp_anchored_mace_polar_hybrid(
    *,
    permanent: MACE_MDPMomentAdapter,
    response: MACEPolarOriginalSourceNativeFieldAdapter,
) -> PermanentAnchoredInducedSourceModel:
    """Build the frozen research hybrid without loading either checkpoint."""

    return PermanentAnchoredInducedSourceModel(
        MACE_MDPPermanentSourceAdapter(permanent), response
    )


__all__ = [
    "FieldResponsiveAtomicL1SourceProvider",
    "MACE_MDP_PERMANENT_SOURCE_PROVIDER_ID",
    "MACE_MDP_POLAR_HYBRID_CONTRACT",
    "MACE_MDP_POLAR_HYBRID_PROFILE_ID",
    "MACE_MDP_POLAR_HYBRID_PROVIDER_ID",
    "MACE_MDPPermanentSourceAdapter",
    "PermanentAnchoredInducedSourceModel",
    "PermanentAtomicL1SourceProvider",
    "PermanentInducedSourceAnchor",
    "build_mace_mdp_anchored_mace_polar_hybrid",
]
