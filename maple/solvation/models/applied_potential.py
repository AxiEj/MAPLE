"""Explicit applied-potential work completion for separated response models.

MACE-POLAR's native field-response path and its applied-potential work are
distinct terms.  This module assembles the latter as an auditable operational
scalar without asserting a common MACE--continuum variational functional.  The
implementation depends only on a small response protocol and an immutable
linear source/receiver bridge so another MLIP can replace MACE-POLAR.

All release capabilities remain closed.  A differentiable scalar candidate is
necessary for energies and forces, but it is not accuracy or PES admission.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.coupling.metrics import (
    MACE_POLAR_RADIAL_GTO_PAIRING,
    PairingMetric,
)
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    NativeFieldSpace,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE, SourceSpace
from maple.solvation.coupling.state_equation import provider_behavior_sha256

from .base import atom_count

APPLIED_POTENTIAL_WORK_COMPLETION_ID = (
    "maple.route2.scalar.applied-potential-work-completion.v1"
)
MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE_ID = (
    "maple.route2.bridge.mace-polar-original-source4-native-field8-work.v1"
)
MACE_POLAR_APPLIED_POTENTIAL_COMPLETION_PROVIDER_ID = (
    "maple.route2.model.mace-polar-applied-potential-completion.impl.v1"
)


@runtime_checkable
class AppliedPotentialResponseProvider(Protocol):
    """Minimal replaceable-MLIP protocol consumed by the completion."""

    provider_id: str
    model_profile_id: str
    coupling_id: str
    provenance_sha256: str
    source_space: SourceSpace
    receiver_space: NativeFieldSpace

    def configuration_sha256(self) -> str: ...

    def evaluate_source(self, geometry: object, field: object) -> np.ndarray: ...

    def field_vjp(
        self, geometry: object, field: object, source_cotangent: object
    ) -> np.ndarray: ...

    def coordinate_vjp(
        self, geometry: object, field: object, source_cotangent: object
    ) -> np.ndarray: ...

    def conditioned_raw_energy_ev(self, geometry: object, field: object) -> float: ...

    def conditioned_raw_energy_field_gradient(
        self, geometry: object, field: object
    ) -> np.ndarray: ...

    def conditioned_raw_energy_fixed_field_coordinate_gradient(
        self, geometry: object, field: object
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SourceReceiverWorkBridge:
    """Linear source embedding and authoritative energy pairing.

    If ``S`` embeds a source ``c`` into the receiver-dual source chart and
    ``Q`` is the registered pairing, the applied work is

    ``W(c,u) = work_sign * (S c)^T Q u``.

    The transpose operations below are the exact derivatives of this one
    scalar; they are not model-specific response approximations.
    """

    bridge_id: str
    source_space: SourceSpace
    receiver_space: NativeFieldSpace
    receiver_pairing: PairingMetric
    one_atom_source_embedding: np.ndarray
    work_sign: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.bridge_id, str) or not self.bridge_id.strip():
            raise ValueError("bridge_id must be a non-empty string.")
        if not isinstance(self.source_space, SourceSpace):
            raise TypeError("source_space must be SourceSpace.")
        if not isinstance(self.receiver_space, NativeFieldSpace):
            raise TypeError("receiver_space must be NativeFieldSpace.")
        if not isinstance(self.receiver_pairing, PairingMetric):
            raise TypeError("receiver_pairing must be PairingMetric.")
        if self.receiver_pairing.component_count != self.receiver_space.component_count:
            raise ValueError("receiver pairing and field space dimensions differ.")
        if isinstance(self.work_sign, bool) or self.work_sign not in (-1, 1):
            raise ValueError("work_sign must be exactly +1 or -1.")
        matrix = np.asarray(self.one_atom_source_embedding, dtype=float)
        expected = (
            self.receiver_space.component_count,
            self.source_space.component_count,
        )
        if matrix.shape != expected or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"one_atom_source_embedding must be finite with shape {expected}."
            )
        if np.linalg.matrix_rank(matrix) != self.source_space.component_count:
            raise ValueError("source embedding must have full column rank.")
        immutable = np.array(matrix, copy=True)
        immutable.setflags(write=False)
        object.__setattr__(self, "one_atom_source_embedding", immutable)

    def metadata(self) -> dict[str, object]:
        return {
            "bridge_id": self.bridge_id,
            "source_space_sha256": self.source_space.metadata_hash(),
            "receiver_space_sha256": self.receiver_space.metadata_hash(),
            "receiver_pairing_sha256": self.receiver_pairing.metadata_hash(),
            "one_atom_source_embedding": self.one_atom_source_embedding.tolist(),
            "work_sign": self.work_sign,
            "formula": "W=work_sign*(S*c)^T*Q*u",
        }

    def configuration_sha256(self) -> str:
        return canonical_metadata_sha256(self.metadata())

    def _source(self, source: object) -> np.ndarray:
        values = np.asarray(source, dtype=float)
        if values.ndim != 2:
            raise ValueError("source must have shape (N, source_components).")
        return self.source_space.validate(values, atom_count=values.shape[0])

    def _field(self, field: object) -> np.ndarray:
        values = np.asarray(field, dtype=float)
        if values.ndim != 2:
            raise ValueError("field must have shape (N, receiver_components).")
        return self.receiver_space.validate(values, atom_count=values.shape[0])

    def embed_source(self, source: object) -> np.ndarray:
        values = self._source(source)
        return np.einsum("rs,ns->nr", self.one_atom_source_embedding, values)

    def source_cotangent(self, field: object) -> np.ndarray:
        """Return ``dW/dc = sign*S.T*Q*u``."""

        values = self._field(field)
        receiver_dual = self.receiver_pairing.field_to_source_dual(values)
        return self.work_sign * np.einsum(
            "rs,nr->ns", self.one_atom_source_embedding, receiver_dual
        )

    def direct_field_gradient(self, source: object) -> np.ndarray:
        """Return the direct term ``dW/du = sign*Q.T*S*c``."""

        embedded = self.embed_source(source)
        return self.work_sign * self.receiver_pairing.source_to_field_dual(embedded)

    def work_ev(self, source: object, field: object) -> float:
        embedded = self.embed_source(source)
        values = self._field(field)
        if embedded.shape != values.shape:
            raise ValueError("source and field atom counts differ.")
        return self.work_sign * self.receiver_pairing.pair(embedded, values)


MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE = SourceReceiverWorkBridge(
    bridge_id=MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE_ID,
    source_space=ATOMIC_L1_SOURCE_SPACE,
    receiver_space=MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    receiver_pairing=MACE_POLAR_RADIAL_GTO_PAIRING,
    one_atom_source_embedding=mace_polar_learned_source_embedding_matrix(),
    work_sign=1,
)


class AppliedPotentialEnergyCompletion:
    """Complete one raw field-conditioned MLIP scalar with explicit work."""

    __slots__ = (
        "_bridge",
        "_configuration_sha256",
        "_response",
        "_sealed",
        "capabilities",
        "completion_id",
        "coupling_id",
        "model_profile_id",
        "provenance_sha256",
        "provider_id",
        "receiver_space",
        "source_space",
        "variational_functional_admitted",
    )

    def __init__(
        self,
        response: AppliedPotentialResponseProvider,
        bridge: SourceReceiverWorkBridge,
        *,
        provider_id: str = MACE_POLAR_APPLIED_POTENTIAL_COMPLETION_PROVIDER_ID,
    ) -> None:
        if not isinstance(bridge, SourceReceiverWorkBridge):
            raise TypeError("bridge must be SourceReceiverWorkBridge.")
        required = (
            "configuration_sha256",
            "evaluate_source",
            "field_vjp",
            "coordinate_vjp",
            "conditioned_raw_energy_ev",
            "conditioned_raw_energy_field_gradient",
            "conditioned_raw_energy_fixed_field_coordinate_gradient",
        )
        for name in required:
            if not callable(getattr(response, name, None)):
                raise TypeError(f"response must expose callable {name}().")
        response.configuration_sha256()
        if getattr(response, "source_space", None) != bridge.source_space:
            raise ValueError("response and bridge source spaces differ.")
        if getattr(response, "receiver_space", None) != bridge.receiver_space:
            raise ValueError("response and bridge receiver spaces differ.")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string.")
        response_provenance = getattr(response, "provenance_sha256", None)
        if not isinstance(response_provenance, str) or len(response_provenance) != 64:
            raise ValueError("response provenance_sha256 must be one SHA256 digest.")
        object.__setattr__(self, "_response", response)
        object.__setattr__(self, "_bridge", bridge)
        object.__setattr__(self, "provider_id", provider_id.strip())
        object.__setattr__(self, "model_profile_id", response.model_profile_id)
        object.__setattr__(self, "coupling_id", response.coupling_id)
        object.__setattr__(self, "completion_id", APPLIED_POTENTIAL_WORK_COMPLETION_ID)
        object.__setattr__(self, "source_space", bridge.source_space)
        object.__setattr__(self, "receiver_space", bridge.receiver_space)
        object.__setattr__(self, "capabilities", ())
        object.__setattr__(self, "variational_functional_admitted", False)
        object.__setattr__(
            self,
            "provenance_sha256",
            self._provenance_sha256(response_provenance),
        )
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("AppliedPotentialEnergyCompletion is immutable.")
        object.__setattr__(self, name, value)

    @property
    def response(self) -> AppliedPotentialResponseProvider:
        self.configuration_sha256()
        return self._response

    @property
    def bridge(self) -> SourceReceiverWorkBridge:
        self.configuration_sha256()
        return self._bridge

    def _provenance_sha256(self, response_provenance: str) -> str:
        payload = {
            "provider_id": self.provider_id,
            "completion_id": self.completion_id,
            "response_provenance_sha256": response_provenance,
            "bridge_sha256": self._bridge.configuration_sha256(),
            "implementation_sha256": hashlib.sha256(
                __file__.encode() + Path(__file__).read_bytes()
            ).hexdigest(),
            "capabilities": "none",
        }
        return canonical_metadata_sha256(payload)

    def _current_configuration(self) -> str:
        behavior = provider_behavior_sha256(
            self._response,
            (
                "configuration_sha256",
                "evaluate_source",
                "field_vjp",
                "coordinate_vjp",
                "conditioned_raw_energy_ev",
                "conditioned_raw_energy_field_gradient",
                "conditioned_raw_energy_fixed_field_coordinate_gradient",
            ),
            label="applied_potential_response",
        )
        payload = {
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "coupling_id": self.coupling_id,
            "completion_id": self.completion_id,
            "provenance_sha256": self.provenance_sha256,
            "response_configuration_sha256": self._response.configuration_sha256(),
            "response_behavior_sha256": behavior,
            "bridge_sha256": self._bridge.configuration_sha256(),
            "source_space_sha256": self.source_space.metadata_hash(),
            "receiver_space_sha256": self.receiver_space.metadata_hash(),
            "formula": "E_complete=E_raw+work_sign*(S*c(u))^T*Q*u",
            "capabilities": "none",
            "variational_functional_admitted": False,
        }
        return canonical_metadata_sha256(payload)

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("applied-potential completion configuration drifted.")
        return current

    def _field(self, geometry: object, field: object) -> np.ndarray:
        return self.receiver_space.validate(
            field, atom_count=atom_count(geometry), name="applied native field"
        )

    def components_ev(self, geometry: object, field: object) -> dict[str, float]:
        self.configuration_sha256()
        values = self._field(geometry, field)
        raw = float(self._response.conditioned_raw_energy_ev(geometry, values))
        source = self.source_space.validate(
            self._response.evaluate_source(geometry, values),
            atom_count=atom_count(geometry),
            name="applied-work source",
        )
        work = self._bridge.work_ev(source, values)
        if not np.isfinite(raw) or not np.isfinite(work):
            raise RuntimeError("applied-potential energy components are non-finite.")
        return {
            "conditioned_raw_energy_ev": raw,
            "explicit_applied_work_ev": work,
            "complete_energy_ev": raw + work,
        }

    def evaluate_energy_ev(self, geometry: object, field: object) -> float:
        return self.components_ev(geometry, field)["complete_energy_ev"]

    def field_gradient(self, geometry: object, field: object) -> np.ndarray:
        """Return the exact field derivative of the completed scalar."""

        self.configuration_sha256()
        values = self._field(geometry, field)
        count = atom_count(geometry)
        source = self.source_space.validate(
            self._response.evaluate_source(geometry, values), atom_count=count
        )
        source_cotangent = self._bridge.source_cotangent(values)
        raw = self.receiver_space.validate(
            self._response.conditioned_raw_energy_field_gradient(geometry, values),
            atom_count=count,
            name="raw-energy field gradient",
        )
        response = self.receiver_space.validate(
            self._response.field_vjp(geometry, values, source_cotangent),
            atom_count=count,
            name="applied-work response field gradient",
        )
        direct = self._bridge.direct_field_gradient(source)
        return self.receiver_space.validate(
            raw + direct + response,
            atom_count=count,
            name="complete-energy field gradient",
        )

    def fixed_field_coordinate_gradient(
        self, geometry: object, field: object
    ) -> np.ndarray:
        """Return ``partial E_complete / partial R`` with ``u`` fixed."""

        self.configuration_sha256()
        values = self._field(geometry, field)
        count = atom_count(geometry)
        source_cotangent = self._bridge.source_cotangent(values)
        raw = np.asarray(
            self._response.conditioned_raw_energy_fixed_field_coordinate_gradient(
                geometry, values
            ),
            dtype=float,
        )
        response = np.asarray(
            self._response.coordinate_vjp(geometry, values, source_cotangent),
            dtype=float,
        )
        if raw.shape != (count, 3) or response.shape != (count, 3):
            raise RuntimeError("coordinate gradients must have shape (N,3).")
        result = raw + response
        if not np.all(np.isfinite(result)):
            raise RuntimeError("complete-energy coordinate gradient is non-finite.")
        return result

    def metadata(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "coupling_id": self.coupling_id,
            "completion_id": self.completion_id,
            "configuration_sha256": self.configuration_sha256(),
            "provenance_sha256": self.provenance_sha256,
            "response_provider_id": self._response.provider_id,
            "response_configuration_sha256": self._response.configuration_sha256(),
            "bridge": self._bridge.metadata(),
            "scalar_formula": "E_complete=E_raw+work_sign*(S*c(u))^T*Q*u",
            "claim_boundary": (
                "operational applied-potential scalar; not a common MACE-continuum "
                "variational functional and not release admission"
            ),
            "capabilities": {tier: False for tier in "EFHVM"},
        }


def build_mace_polar_applied_potential_completion(
    response: AppliedPotentialResponseProvider,
) -> AppliedPotentialEnergyCompletion:
    """Bind the official positive-work source4/field8 MACE-POLAR bridge."""

    return AppliedPotentialEnergyCompletion(
        response, MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE
    )


__all__ = [
    "APPLIED_POTENTIAL_WORK_COMPLETION_ID",
    "MACE_POLAR_APPLIED_POTENTIAL_COMPLETION_PROVIDER_ID",
    "MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE",
    "MACE_POLAR_ORIGINAL_SOURCE_APPLIED_WORK_BRIDGE_ID",
    "AppliedPotentialEnergyCompletion",
    "AppliedPotentialResponseProvider",
    "SourceReceiverWorkBridge",
    "build_mace_polar_applied_potential_completion",
]
