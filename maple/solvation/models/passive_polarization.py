"""Structurally passive scalar-first polarization head contract.

This module defines a model-independent wrapper for a *trained* head whose
only learned outputs are a vacuum scalar, a permanent reduced-source
covector, and a response factor.  The public field energy is fixed to

``E(R, xi) = E0(R) + m0(R).xi - 1/2 ||C(R) xi||^2``.

Consequently ``-d2E/dxi2 = C.T C`` is positive semidefinite by construction.
All source, field-response, HVP, and mixed coordinate derivatives remain
sealed in :class:`~maple.solvation.models.field_energy.FieldEnergyFunctional`
and are generated from this one scalar graph.

The wrapper is not a trained model and admits no scientific capability.  A
concrete head must provide a content-addressed checkpoint, a live parameter
digest, and optimizer-parameter-group audit evidence.  Source physics,
finite-field accuracy, coupled-root stability, and coordinate-domain gates
remain separate release requirements.

Torch is imported lazily so dependency-light contract tooling can import this
module without a model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.coupling.state_equation import provider_behavior_sha256

from .base import ModelProvenance, model_input_sha256
from .field_energy import FieldEnergyFunctional, GaugeReducedDualityMap, TorchGeometry
from .passive_training import (
    PASSIVE_QUADRATIC_CONSTRUCTION_ID,
    PassiveTrainingRunBinding,
)

PASSIVE_QUADRATIC_FIELD_ENERGY_PROVIDER_ID = (
    "maple.route2.models.passive-quadratic-field-energy.v1"
)


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    digest = _text(value, name=name).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return digest


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _torch():
    return __import__("torch")


@dataclass(frozen=True, slots=True)
class PassiveHeadTensors:
    """Graph-local tensors emitted by one trained scalar-first head.

    ``response_factor`` has shape ``(rank, reduced_dimension)``.  It is never
    interpreted as a source by itself; the fixed scalar construction is the
    sole owner of that meaning.
    """

    vacuum_energy_eV: Any
    permanent_reduced_covector: Any
    response_factor: Any


@runtime_checkable
class TrainedPassivePolarizationHead(Protocol):
    """Typing aid for a trained head; runtime admission uses explicit checks."""

    provider_id: str
    model_profile_id: str
    coupling_id: str
    provenance: ModelProvenance
    provenance_sha256: str
    duality_map_sha256: str
    training_preregistration_sha256: str
    training_run_sha256: str

    def configuration_sha256(self) -> str: ...

    def parameter_state_sha256(self) -> str: ...

    def coefficients_torch(
        self, geometry: TorchGeometry, *, reduced_dimension: int
    ) -> PassiveHeadTensors: ...


@dataclass(frozen=True, slots=True)
class PassiveQuadraticPointCertificate:
    """Pointwise diagnostic implied by the factorized scalar construction."""

    provider_id: str
    configuration_sha256: str
    model_input_sha256: str
    reduced_dimension: int
    response_factor_rank: int
    response_factor_singular_values: tuple[float, ...]
    susceptibility_min_eigenvalue_lower_bound: float
    susceptibility_max_eigenvalue: float
    structural_passivity: bool = True
    strict_passivity_at_this_point: bool = False
    global_domain_passivity_admitted: bool = False
    coupled_root_stability_admitted: bool = False
    public_capability_admitted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, name="provider_id")
        )
        object.__setattr__(
            self,
            "configuration_sha256",
            _digest(self.configuration_sha256, name="configuration_sha256"),
        )
        object.__setattr__(
            self,
            "model_input_sha256",
            _digest(self.model_input_sha256, name="model_input_sha256"),
        )
        if (
            isinstance(self.reduced_dimension, bool)
            or not isinstance(self.reduced_dimension, int)
            or self.reduced_dimension < 1
        ):
            raise ValueError("reduced_dimension must be a positive integer.")
        if (
            isinstance(self.response_factor_rank, bool)
            or not isinstance(self.response_factor_rank, int)
            or not 0 <= self.response_factor_rank <= self.reduced_dimension
        ):
            raise ValueError("response_factor_rank is outside the reduced space.")
        singular_values = tuple(
            float(value) for value in self.response_factor_singular_values
        )
        if not singular_values or any(
            not np.isfinite(value) or value < 0.0 for value in singular_values
        ):
            raise ValueError(
                "response factor singular values must be finite and nonnegative."
            )
        object.__setattr__(self, "response_factor_singular_values", singular_values)
        for name in (
            "susceptibility_min_eigenvalue_lower_bound",
            "susceptibility_max_eigenvalue",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        for name in (
            "structural_passivity",
            "strict_passivity_at_this_point",
            "global_domain_passivity_admitted",
            "coupled_root_stability_admitted",
            "public_capability_admitted",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        if not self.structural_passivity:
            raise ValueError(
                "the quadratic certificate must remain structurally passive."
            )
        if (
            self.global_domain_passivity_admitted
            or self.coupled_root_stability_admitted
            or self.public_capability_admitted
        ):
            raise ValueError(
                "pointwise construction diagnostics cannot admit capabilities."
            )


class PassiveQuadraticFieldEnergy(FieldEnergyFunctional):
    """Final scalar wrapper around one trained passive-polarization head."""

    __slots__ = (
        "_configuration_sha256",
        "_head",
        "_head_behavior_sha256",
        "_head_configuration_sha256",
        "_head_parameter_state_sha256",
        "_training_run",
        "capabilities",
        "coordinate_frame_policy",
        "coupling_id",
        "device",
        "domain",
        "dtype",
        "field_convention",
        "field_space",
        "model_profile_id",
        "provenance",
        "provenance_sha256",
        "provider_id",
        "source_space",
        "training_preregistration_sha256",
        "training_run_sha256",
        "variational_functional_admitted",
    )

    structural_passivity = True
    scalar_first = True

    def __init_subclass__(cls, **kwargs) -> None:
        del kwargs
        raise TypeError(
            "PassiveQuadraticFieldEnergy is final; model-specific code belongs "
            "only in a trained head's coefficients_torch()."
        )

    def __init__(
        self,
        head: TrainedPassivePolarizationHead,
        *,
        training_run: PassiveTrainingRunBinding,
        duality_map: GaugeReducedDualityMap,
        dtype: object,
        device: object,
    ) -> None:
        if not isinstance(duality_map, GaugeReducedDualityMap):
            raise TypeError("duality_map must be GaugeReducedDualityMap.")
        if not isinstance(training_run, PassiveTrainingRunBinding):
            raise TypeError("training_run must be PassiveTrainingRunBinding.")
        if duality_map.conjugacy_sign != 1:
            raise ValueError(
                "the public scalar-first contract requires positive energy-dual "
                "conjugacy_sign=+1."
            )
        for name in (
            "configuration_sha256",
            "parameter_state_sha256",
            "coefficients_torch",
        ):
            if not callable(getattr(head, name, None)):
                raise TypeError(f"trained passive head must implement {name}().")
        provenance = getattr(head, "provenance", None)
        if not isinstance(provenance, ModelProvenance):
            raise TypeError("trained passive head provenance must be ModelProvenance.")
        head_provider = _text(
            getattr(head, "provider_id", None), name="head.provider_id"
        )
        head_profile = _text(
            getattr(head, "model_profile_id", None), name="head.model_profile_id"
        )
        coupling_id = _text(getattr(head, "coupling_id", None), name="head.coupling_id")
        if provenance.provider_id != head_provider:
            raise ValueError("head provider_id and provenance identity differ.")
        if provenance.model_profile_id != head_profile:
            raise ValueError("head model_profile_id and provenance identity differ.")
        if getattr(head, "provenance_sha256", None) != provenance.sha256:
            raise ValueError("head provenance_sha256 is stale or mismatched.")
        if not provenance.optimizer_parameter_groups_audited:
            raise ValueError(
                "a trained passive head requires optimizer parameter-group audit evidence."
            )
        training_run.validate_head(head)
        if provenance.field_convention != duality_map.field_space.field_convention:
            raise ValueError("head and duality-map field conventions differ.")
        head_duality = _digest(
            getattr(head, "duality_map_sha256", None),
            name="head.duality_map_sha256",
        )
        if head_duality != duality_map.configuration_sha256():
            raise ValueError("head was trained for a different duality map.")

        normalized_dtype = str(dtype).replace("torch.", "")
        if normalized_dtype != provenance.dtype:
            raise ValueError("runtime dtype differs from trained-head provenance.")
        if str(device) != provenance.device:
            raise ValueError("runtime device differs from trained-head provenance.")

        head_configuration = _digest(
            head.configuration_sha256(), name="head configuration_sha256"
        )
        parameter_state = _digest(
            head.parameter_state_sha256(), name="head parameter_state_sha256"
        )
        head_behavior = provider_behavior_sha256(
            head,
            (
                "configuration_sha256",
                "parameter_state_sha256",
                "coefficients_torch",
            ),
            label="trained_passive_polarization_head",
        )
        wrapper_inference_sha256 = _hash(
            {
                "construction_id": PASSIVE_QUADRATIC_CONSTRUCTION_ID,
                "implementation_sha256": _implementation_sha256(),
                "head_inference_code_sha256": provenance.inference_code_sha256,
                "head_configuration_sha256": head_configuration,
                "head_parameter_state_sha256": parameter_state,
                "head_behavior_sha256": head_behavior,
                "duality_map_sha256": duality_map.configuration_sha256(),
                "coupling_id": coupling_id,
                "training_preregistration_sha256": (
                    training_run.preregistration_sha256
                ),
                "training_run_sha256": training_run.content_sha256,
            }
        )
        wrapper_provenance = ModelProvenance(
            provider_id=PASSIVE_QUADRATIC_FIELD_ENERGY_PROVIDER_ID,
            model_profile_id=head_profile,
            model_family=f"{provenance.model_family}:passive-quadratic-field-energy",
            checkpoint_sha256=provenance.checkpoint_sha256,
            upstream_version=provenance.upstream_version,
            upstream_commit=provenance.upstream_commit,
            inference_code_sha256=wrapper_inference_sha256,
            dtype=provenance.dtype,
            device=provenance.device,
            domain=provenance.domain,
            field_convention=provenance.field_convention,
            coordinate_frame_policy=provenance.coordinate_frame_policy,
            optimizer_parameter_groups_audited=True,
            optimizer_audit_evidence_sha256=(
                provenance.optimizer_audit_evidence_sha256
            ),
        )

        object.__setattr__(self, "_head", head)
        object.__setattr__(self, "_training_run", training_run)
        object.__setattr__(self, "_head_configuration_sha256", head_configuration)
        object.__setattr__(self, "_head_parameter_state_sha256", parameter_state)
        object.__setattr__(self, "_head_behavior_sha256", head_behavior)
        object.__setattr__(self, "provider_id", wrapper_provenance.provider_id)
        object.__setattr__(self, "model_profile_id", head_profile)
        object.__setattr__(self, "coupling_id", coupling_id)
        object.__setattr__(
            self,
            "training_preregistration_sha256",
            training_run.preregistration_sha256,
        )
        object.__setattr__(self, "training_run_sha256", training_run.content_sha256)
        object.__setattr__(self, "provenance", wrapper_provenance)
        object.__setattr__(self, "provenance_sha256", wrapper_provenance.sha256)
        object.__setattr__(self, "dtype", wrapper_provenance.dtype)
        object.__setattr__(self, "device", wrapper_provenance.device)
        object.__setattr__(self, "domain", wrapper_provenance.domain)
        object.__setattr__(
            self, "field_convention", wrapper_provenance.field_convention
        )
        object.__setattr__(
            self,
            "coordinate_frame_policy",
            wrapper_provenance.coordinate_frame_policy,
        )
        object.__setattr__(self, "source_space", duality_map.source_space)
        object.__setattr__(self, "field_space", duality_map.field_space)
        object.__setattr__(self, "capabilities", CapabilityStatus())
        object.__setattr__(self, "variational_functional_admitted", False)
        super().__init__(duality_map=duality_map, dtype=dtype, device=device)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )

    @property
    def trained_head(self) -> TrainedPassivePolarizationHead:
        self.configuration_sha256()
        return self._head

    @property
    def training_run(self) -> PassiveTrainingRunBinding:
        self.configuration_sha256()
        return self._training_run

    def _live_head_binding(self) -> dict[str, str]:
        provenance = getattr(self._head, "provenance", None)
        if not isinstance(provenance, ModelProvenance):
            raise RuntimeError("trained passive head provenance drifted.")
        if (
            getattr(self._head, "provider_id", None) != provenance.provider_id
            or getattr(self._head, "model_profile_id", None)
            != provenance.model_profile_id
            or getattr(self._head, "provenance_sha256", None) != provenance.sha256
        ):
            raise RuntimeError("trained passive head identity drifted.")
        try:
            self._training_run.validate_head(self._head)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "trained passive head training binding drifted."
            ) from exc
        return {
            "head_provenance_sha256": provenance.sha256,
            "head_configuration_sha256": _digest(
                self._head.configuration_sha256(),
                name="head configuration_sha256",
            ),
            "head_parameter_state_sha256": _digest(
                self._head.parameter_state_sha256(),
                name="head parameter_state_sha256",
            ),
            "head_behavior_sha256": provider_behavior_sha256(
                self._head,
                (
                    "configuration_sha256",
                    "parameter_state_sha256",
                    "coefficients_torch",
                ),
                label="trained_passive_polarization_head",
            ),
            "training_preregistration_sha256": (
                self._training_run.preregistration_sha256
            ),
            "training_run_sha256": self._training_run.content_sha256,
        }

    def _configuration_payload(self, live: dict[str, str]) -> dict[str, object]:
        return {
            "construction_id": PASSIVE_QUADRATIC_CONSTRUCTION_ID,
            "implementation_sha256": _implementation_sha256(),
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "coupling_id": self.coupling_id,
            "training_preregistration_sha256": (self.training_preregistration_sha256),
            "training_run_sha256": self.training_run_sha256,
            "provenance_sha256": self.provenance_sha256,
            "duality_map_sha256": self.duality_map.configuration_sha256(),
            "source_space_sha256": self.source_space.metadata_hash(),
            "field_space_sha256": self.field_space.metadata_hash(),
            "dtype": self.dtype,
            "device": self.device,
            "field_convention": self.field_convention,
            "coordinate_frame_policy": self.coordinate_frame_policy,
            "capabilities": self.capabilities.enabled_tiers,
            "variational_functional_admitted": False,
            "structural_identity": "-H_xi=C.T@C positive-semidefinite",
            **live,
        }

    def _current_configuration_sha256(self) -> str:
        return _hash(self._configuration_payload(self._live_head_binding()))

    def configuration_sha256(self) -> str:
        if self.capabilities.enabled_tiers or self.variational_functional_admitted:
            raise RuntimeError("passive-head capabilities must remain fail-closed.")
        live = self._live_head_binding()
        current = _hash(self._configuration_payload(live))
        if current != self._configuration_sha256:
            raise RuntimeError("passive quadratic field-energy configuration drifted.")
        if self._head_configuration_sha256 != live["head_configuration_sha256"]:
            raise RuntimeError("trained passive head configuration drifted.")
        if self._head_parameter_state_sha256 != live["head_parameter_state_sha256"]:
            raise RuntimeError("trained passive head parameters drifted.")
        if self._head_behavior_sha256 != live["head_behavior_sha256"]:
            raise RuntimeError("trained passive head behavior drifted.")
        return current

    def _coefficient_tensors(
        self, geometry: TorchGeometry, reduced_field: Any
    ) -> PassiveHeadTensors:
        torch = _torch()
        self.domain.validate_atoms(geometry.atoms)
        tensors = self._head.coefficients_torch(
            geometry, reduced_dimension=int(reduced_field.shape[0])
        )
        if not isinstance(tensors, PassiveHeadTensors):
            raise TypeError("coefficients_torch must return PassiveHeadTensors.")
        energy = tensors.vacuum_energy_eV
        permanent = tensors.permanent_reduced_covector
        factor = tensors.response_factor
        for name, value in (
            ("vacuum_energy_eV", energy),
            ("permanent_reduced_covector", permanent),
            ("response_factor", factor),
        ):
            if not torch.is_tensor(value) or not torch.is_floating_point(value):
                raise TypeError(f"{name} must be a floating Torch tensor.")
            if (
                value.dtype != reduced_field.dtype
                or value.device != reduced_field.device
            ):
                raise ValueError(f"{name} must share reduced-field dtype and device.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} is non-finite.")
        if energy.numel() != 1:
            raise ValueError("vacuum_energy_eV must be scalar.")
        if permanent.shape != reduced_field.shape:
            raise ValueError("permanent reduced covector has the wrong shape.")
        if (
            factor.ndim != 2
            or factor.shape[0] < 1
            or factor.shape[1] != reduced_field.shape[0]
        ):
            raise ValueError(
                "response_factor must have shape (positive_rank, reduced_dimension)."
            )
        return PassiveHeadTensors(energy.reshape(()), permanent, factor)

    def _energy_torch(self, geometry: TorchGeometry, reduced_field: Any):
        self.configuration_sha256()
        tensors = self._coefficient_tensors(geometry, reduced_field)
        induced_coordinates = tensors.response_factor @ reduced_field
        return (
            tensors.vacuum_energy_eV
            + _torch().dot(tensors.permanent_reduced_covector, reduced_field)
            - 0.5 * _torch().dot(induced_coordinates, induced_coordinates)
        )

    def pointwise_passivity_certificate(
        self, atoms: object, *, total_charge: float
    ) -> PassiveQuadraticPointCertificate:
        self.configuration_sha256()
        total_charge = self._validated_total_charge(atoms, total_charge)
        geometry = self._geometry(atoms, requires_grad=False)
        coordinates = self.duality_map.coordinates(
            atom_count=len(atoms), total_charge=total_charge  # type: ignore[arg-type]
        )
        reduced = self._reduced_tensor(
            np.zeros(coordinates.reduced_dimension),
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
            requires_grad=False,
        )
        factor = self._coefficient_tensors(geometry, reduced).response_factor
        singular_values = np.asarray(
            _torch().linalg.svdvals(factor).detach().cpu(), dtype=float
        )
        tolerance = (
            max(factor.shape)
            * np.finfo(float).eps
            * max(1.0, float(np.max(singular_values)))
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        strict = bool(rank == coordinates.reduced_dimension)
        minimum = float(np.min(singular_values) ** 2) if strict else 0.0
        maximum = float(np.max(singular_values) ** 2)
        return PassiveQuadraticPointCertificate(
            provider_id=self.provider_id,
            configuration_sha256=self.configuration_sha256(),
            model_input_sha256=model_input_sha256(atoms),
            reduced_dimension=coordinates.reduced_dimension,
            response_factor_rank=rank,
            response_factor_singular_values=tuple(
                float(value) for value in singular_values
            ),
            susceptibility_min_eigenvalue_lower_bound=minimum,
            susceptibility_max_eigenvalue=maximum,
            strict_passivity_at_this_point=strict,
        )


__all__ = [
    "PASSIVE_QUADRATIC_CONSTRUCTION_ID",
    "PASSIVE_QUADRATIC_FIELD_ENERGY_PROVIDER_ID",
    "PassiveHeadTensors",
    "PassiveQuadraticFieldEnergy",
    "PassiveQuadraticPointCertificate",
    "TrainedPassivePolarizationHead",
]
