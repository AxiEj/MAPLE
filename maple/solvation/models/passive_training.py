"""Content-addressed training contract for scalar-first polarization heads.

This module does not train a model and cannot admit a scientific capability.
It freezes the data partitions, allowed QM targets, chemical/numerical domain,
validation protocols, and random seeds before a passive head is trained.  A
completed training run is then bound to that preregistration and to the exact
checkpoint, live parameter state, optimizer audit, code, runtime, and logs.

The contract deliberately rejects experimental/total-solvation targets and
conformer-family leakage across train, validation, and blind partitions.  It
is dependency-light and imports no model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import TYPE_CHECKING

from .base import ModelDomain, ModelProvenance

if TYPE_CHECKING:
    from .passive_polarization import TrainedPassivePolarizationHead


PASSIVE_QUADRATIC_CONSTRUCTION_ID = (
    "route2-scalar-first-passive-quadratic-reduced-field-v1"
)
PASSIVE_TRAINING_PREREGISTRATION_VERSION = "route2-passive-training-preregistration-v1"
PASSIVE_TRAINING_RUN_BINDING_VERSION = "route2-passive-training-run-binding-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_ROLES = ("train", "validation", "blind")
_TARGET_KINDS = frozenset(
    {
        "vacuum_energy",
        "vacuum_force",
        "permanent_source",
        "external_field_energy",
        "external_field_force",
        "dipole",
        "polarizability",
        "far_field_mep",
        "cavity_surface_mep",
        "fixed_source_pcm_energy",
    }
)
_ADMISSION_ONLY_TARGETS = frozenset({"fixed_source_pcm_energy"})


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return result


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be exactly bool.")
    return value


def _sorted_digests(values: object, *, name: str) -> tuple[str, ...]:
    try:
        result = tuple(_digest(value, name=name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of SHA256 digests.") from exc
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be unique and sorted.")
    return result


@dataclass(frozen=True, slots=True)
class PassiveTrainingDataSplit:
    """One immutable dataset partition with molecule-family leakage guards."""

    split_id: str
    role: str
    index_artifact_sha256: str
    record_sha256s: tuple[str, ...]
    molecule_group_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_id", _text(self.split_id, name="split_id"))
        role = _text(self.role, name="role").lower()
        if role not in _SPLIT_ROLES:
            raise ValueError(f"role must be one of {_SPLIT_ROLES}.")
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "index_artifact_sha256",
            _digest(self.index_artifact_sha256, name="index_artifact_sha256"),
        )
        object.__setattr__(
            self,
            "record_sha256s",
            _sorted_digests(self.record_sha256s, name="record_sha256s"),
        )
        object.__setattr__(
            self,
            "molecule_group_sha256s",
            _sorted_digests(self.molecule_group_sha256s, name="molecule_group_sha256s"),
        )
        if len(self.molecule_group_sha256s) > len(self.record_sha256s):
            raise ValueError(
                "a split cannot contain more molecule groups than records."
            )

    @property
    def content_sha256(self) -> str:
        return _hash(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "split_id": self.split_id,
            "role": self.role,
            "index_artifact_sha256": self.index_artifact_sha256,
            "record_sha256s": list(self.record_sha256s),
            "molecule_group_sha256s": list(self.molecule_group_sha256s),
        }


@dataclass(frozen=True, slots=True)
class PassiveTrainingTarget:
    """One preregistered QM/component target and its allowed use."""

    target_id: str
    kind: str
    unit: str
    reference_method_id: str
    reference_protocol_sha256: str
    objective_weight: float
    use_for_training: bool
    use_for_model_selection: bool
    use_for_admission: bool

    def __post_init__(self) -> None:
        for name in ("target_id", "unit", "reference_method_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        kind = _text(self.kind, name="kind").lower()
        if kind not in _TARGET_KINDS:
            raise ValueError(
                "target kind is not an allowed scalar-first QM/component target."
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "reference_protocol_sha256",
            _digest(
                self.reference_protocol_sha256,
                name="reference_protocol_sha256",
            ),
        )
        if isinstance(self.objective_weight, bool):
            raise TypeError("objective_weight must be numeric, not bool.")
        weight = float(self.objective_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("objective_weight must be finite and non-negative.")
        object.__setattr__(self, "objective_weight", weight)
        for name in (
            "use_for_training",
            "use_for_model_selection",
            "use_for_admission",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name=name))
        if not any(
            (
                self.use_for_training,
                self.use_for_model_selection,
                self.use_for_admission,
            )
        ):
            raise ValueError("a target must have at least one declared use.")
        if self.use_for_training is not (weight > 0.0):
            raise ValueError(
                "objective_weight must be positive exactly for training targets."
            )
        if kind in _ADMISSION_ONLY_TARGETS and (
            self.use_for_training or self.use_for_model_selection
        ):
            raise ValueError(
                "fixed-source PCM energy is an admission component, not a fit or "
                "model-selection target."
            )

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class PassiveTrainingDomain:
    """Finite chemical, geometry, field, and representation training domain."""

    model_domain: ModelDomain
    maximum_atom_count: int
    minimum_interatomic_distance_angstrom: float
    maximum_reduced_field_norm: float
    coordinate_frame_policy: str
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_domain, ModelDomain):
            raise TypeError("model_domain must be ModelDomain.")
        if (
            isinstance(self.maximum_atom_count, bool)
            or not isinstance(self.maximum_atom_count, int)
            or self.maximum_atom_count < 1
        ):
            raise ValueError("maximum_atom_count must be a positive integer.")
        for name in (
            "minimum_interatomic_distance_angstrom",
            "maximum_reduced_field_norm",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise TypeError(f"{name} must be numeric, not bool.")
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        for name in ("coordinate_frame_policy", "dtype"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))

    def as_dict(self) -> dict[str, object]:
        return {
            "model_domain": self.model_domain.metadata(),
            "maximum_atom_count": self.maximum_atom_count,
            "minimum_interatomic_distance_angstrom": (
                self.minimum_interatomic_distance_angstrom
            ),
            "maximum_reduced_field_norm": self.maximum_reduced_field_norm,
            "coordinate_frame_policy": self.coordinate_frame_policy,
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class PassiveTrainingPreregistration:
    """Frozen plan that must exist before training or blind evaluation."""

    preregistration_id: str
    head_provider_id: str
    model_profile_id: str
    coupling_id: str
    duality_map_sha256: str
    domain: PassiveTrainingDomain
    data_splits: tuple[PassiveTrainingDataSplit, ...]
    targets: tuple[PassiveTrainingTarget, ...]
    training_code_sha256: str
    inference_code_sha256: str
    optimizer_protocol_sha256: str
    source_gate_protocol_sha256: str
    response_gate_protocol_sha256: str
    root_gate_protocol_sha256: str
    random_seeds: tuple[int, ...]
    experimental_solvation_labels_used: bool = False
    blind_split_used_for_model_selection: bool = False
    scalar_construction_id: str = PASSIVE_QUADRATIC_CONSTRUCTION_ID
    contract_version: str = PASSIVE_TRAINING_PREREGISTRATION_VERSION

    def __post_init__(self) -> None:
        for name in (
            "preregistration_id",
            "head_provider_id",
            "model_profile_id",
            "coupling_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "duality_map_sha256",
            "training_code_sha256",
            "inference_code_sha256",
            "optimizer_protocol_sha256",
            "source_gate_protocol_sha256",
            "response_gate_protocol_sha256",
            "root_gate_protocol_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if not isinstance(self.domain, PassiveTrainingDomain):
            raise TypeError("domain must be PassiveTrainingDomain.")
        if self.scalar_construction_id != PASSIVE_QUADRATIC_CONSTRUCTION_ID:
            raise ValueError("scalar construction differs from the passive contract.")
        if self.contract_version != PASSIVE_TRAINING_PREREGISTRATION_VERSION:
            raise ValueError(
                "training preregistration contract version is unsupported."
            )
        splits = tuple(self.data_splits)
        if not all(isinstance(split, PassiveTrainingDataSplit) for split in splits):
            raise TypeError("data_splits must contain typed split records.")
        if tuple(sorted(split.role for split in splits)) != tuple(sorted(_SPLIT_ROLES)):
            raise ValueError(
                "exactly one train, validation, and blind split is required."
            )
        if len({split.split_id for split in splits}) != len(splits):
            raise ValueError("data split identities must be unique.")
        splits_by_role = {split.role: split for split in splits}
        splits = tuple(splits_by_role[role] for role in _SPLIT_ROLES)
        self._validate_partition_disjointness(splits)
        object.__setattr__(self, "data_splits", splits)
        targets = tuple(self.targets)
        if not targets or not all(
            isinstance(target, PassiveTrainingTarget) for target in targets
        ):
            raise TypeError("targets must contain typed target records.")
        if len({target.target_id for target in targets}) != len(targets):
            raise ValueError("target identities must be unique.")
        targets = tuple(sorted(targets, key=lambda target: target.target_id))
        if not any(target.use_for_training for target in targets):
            raise ValueError("at least one target must train the scalar head.")
        if not any(target.use_for_model_selection for target in targets):
            raise ValueError("at least one validation target must be preregistered.")
        if not any(target.use_for_admission for target in targets):
            raise ValueError("at least one independent admission target is required.")
        object.__setattr__(self, "targets", targets)
        seeds = tuple(self.random_seeds)
        if not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        ):
            raise ValueError("random_seeds must contain non-negative integers.")
        if tuple(sorted(set(seeds))) != seeds:
            raise ValueError("random_seeds must be unique and sorted.")
        object.__setattr__(self, "random_seeds", seeds)
        if _bool(
            self.experimental_solvation_labels_used,
            name="experimental_solvation_labels_used",
        ):
            raise ValueError(
                "experimental solvation labels are forbidden for training."
            )
        if _bool(
            self.blind_split_used_for_model_selection,
            name="blind_split_used_for_model_selection",
        ):
            raise ValueError("the blind split cannot be used for model selection.")

    @staticmethod
    def _validate_partition_disjointness(
        splits: tuple[PassiveTrainingDataSplit, ...],
    ) -> None:
        for field in ("record_sha256s", "molecule_group_sha256s"):
            owner: dict[str, str] = {}
            for split in splits:
                for digest in getattr(split, field):
                    previous = owner.setdefault(digest, split.role)
                    if previous != split.role:
                        raise ValueError(
                            f"{field} overlap between {previous} and {split.role}."
                        )

    @property
    def content_sha256(self) -> str:
        return _hash(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "preregistration_id": self.preregistration_id,
            "head_provider_id": self.head_provider_id,
            "model_profile_id": self.model_profile_id,
            "coupling_id": self.coupling_id,
            "scalar_construction_id": self.scalar_construction_id,
            "duality_map_sha256": self.duality_map_sha256,
            "domain": self.domain.as_dict(),
            "data_splits": [split.as_dict() for split in self.data_splits],
            "targets": [target.as_dict() for target in self.targets],
            "training_code_sha256": self.training_code_sha256,
            "inference_code_sha256": self.inference_code_sha256,
            "optimizer_protocol_sha256": self.optimizer_protocol_sha256,
            "source_gate_protocol_sha256": self.source_gate_protocol_sha256,
            "response_gate_protocol_sha256": self.response_gate_protocol_sha256,
            "root_gate_protocol_sha256": self.root_gate_protocol_sha256,
            "random_seeds": list(self.random_seeds),
            "experimental_solvation_labels_used": False,
            "blind_split_used_for_model_selection": False,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
            "claim_boundary": (
                "This freezes a training and validation plan only. It is not a "
                "trained checkpoint, source-physics result, or E/F/H/V/M evidence."
            ),
        }


@dataclass(frozen=True, slots=True)
class PassiveTrainingRunBinding:
    """Bind one completed training execution to its immutable preregistration."""

    training_run_id: str
    preregistration_sha256: str
    head_provider_id: str
    model_profile_id: str
    coupling_id: str
    scalar_construction_id: str
    duality_map_sha256: str
    checkpoint_sha256: str
    parameter_state_sha256: str
    training_code_sha256: str
    inference_code_sha256: str
    optimizer_protocol_sha256: str
    optimizer_state_sha256: str
    optimizer_parameter_group_audit_sha256: str
    dataset_access_audit_sha256: str
    runtime_manifest_sha256: str
    training_log_sha256: str
    random_seed: int
    completed_successfully: bool
    contract_version: str = PASSIVE_TRAINING_RUN_BINDING_VERSION

    def __post_init__(self) -> None:
        for name in (
            "training_run_id",
            "head_provider_id",
            "model_profile_id",
            "coupling_id",
            "scalar_construction_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "preregistration_sha256",
            "duality_map_sha256",
            "checkpoint_sha256",
            "parameter_state_sha256",
            "training_code_sha256",
            "inference_code_sha256",
            "optimizer_protocol_sha256",
            "optimizer_state_sha256",
            "optimizer_parameter_group_audit_sha256",
            "dataset_access_audit_sha256",
            "runtime_manifest_sha256",
            "training_log_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.scalar_construction_id != PASSIVE_QUADRATIC_CONSTRUCTION_ID:
            raise ValueError("training run used a different scalar construction.")
        if self.contract_version != PASSIVE_TRAINING_RUN_BINDING_VERSION:
            raise ValueError("training run binding contract version is unsupported.")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer.")
        if not _bool(self.completed_successfully, name="completed_successfully"):
            raise ValueError("an incomplete or failed training run cannot be bound.")

    @property
    def content_sha256(self) -> str:
        return _hash(self._payload())

    def _payload(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
            "claim_boundary": (
                "A completed training binding proves identity consistency only; "
                "source, response, root, force, and public gates remain separate."
            ),
        }

    def validate_preregistration(
        self, preregistration: PassiveTrainingPreregistration
    ) -> None:
        if not isinstance(preregistration, PassiveTrainingPreregistration):
            raise TypeError("preregistration must be PassiveTrainingPreregistration.")
        expected = {
            "head_provider_id": preregistration.head_provider_id,
            "model_profile_id": preregistration.model_profile_id,
            "coupling_id": preregistration.coupling_id,
            "scalar_construction_id": preregistration.scalar_construction_id,
            "duality_map_sha256": preregistration.duality_map_sha256,
            "training_code_sha256": preregistration.training_code_sha256,
            "inference_code_sha256": preregistration.inference_code_sha256,
            "optimizer_protocol_sha256": preregistration.optimizer_protocol_sha256,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"training run does not match preregistered {name}.")
        if self.random_seed not in preregistration.random_seeds:
            raise ValueError("training seed was not preregistered.")
        if self.preregistration_sha256 != preregistration.content_sha256:
            raise ValueError(
                "training run does not match preregistered content SHA256."
            )

    def validate_head(self, head: "TrainedPassivePolarizationHead") -> None:
        provenance = getattr(head, "provenance", None)
        if not isinstance(provenance, ModelProvenance):
            raise TypeError("trained head provenance must be ModelProvenance.")
        expected = {
            "head.provider_id": (
                getattr(head, "provider_id", None),
                self.head_provider_id,
            ),
            "head.model_profile_id": (
                getattr(head, "model_profile_id", None),
                self.model_profile_id,
            ),
            "head.coupling_id": (getattr(head, "coupling_id", None), self.coupling_id),
            "head.duality_map_sha256": (
                getattr(head, "duality_map_sha256", None),
                self.duality_map_sha256,
            ),
            "head.training_preregistration_sha256": (
                getattr(head, "training_preregistration_sha256", None),
                self.preregistration_sha256,
            ),
            "head.training_run_sha256": (
                getattr(head, "training_run_sha256", None),
                self.content_sha256,
            ),
            "provenance.checkpoint_sha256": (
                provenance.checkpoint_sha256,
                self.checkpoint_sha256,
            ),
            "provenance.inference_code_sha256": (
                provenance.inference_code_sha256,
                self.inference_code_sha256,
            ),
            "provenance.optimizer_audit_evidence_sha256": (
                provenance.optimizer_audit_evidence_sha256,
                self.optimizer_parameter_group_audit_sha256,
            ),
        }
        for name, (actual, wanted) in expected.items():
            if actual != wanted:
                raise ValueError(f"{name} does not match the training run binding.")
        parameter_state = getattr(head, "parameter_state_sha256", None)
        if not callable(parameter_state):
            raise TypeError("trained head must implement parameter_state_sha256().")
        if parameter_state() != self.parameter_state_sha256:
            raise ValueError("live head parameters do not match the training run.")


def bind_passive_training_run(
    preregistration: PassiveTrainingPreregistration,
    *,
    training_run_id: str,
    checkpoint_sha256: str,
    parameter_state_sha256: str,
    optimizer_state_sha256: str,
    optimizer_parameter_group_audit_sha256: str,
    dataset_access_audit_sha256: str,
    runtime_manifest_sha256: str,
    training_log_sha256: str,
    random_seed: int,
) -> PassiveTrainingRunBinding:
    """Construct and cross-check one completed run against a frozen plan."""

    if not isinstance(preregistration, PassiveTrainingPreregistration):
        raise TypeError("preregistration must be PassiveTrainingPreregistration.")
    result = PassiveTrainingRunBinding(
        training_run_id=training_run_id,
        preregistration_sha256=preregistration.content_sha256,
        head_provider_id=preregistration.head_provider_id,
        model_profile_id=preregistration.model_profile_id,
        coupling_id=preregistration.coupling_id,
        scalar_construction_id=preregistration.scalar_construction_id,
        duality_map_sha256=preregistration.duality_map_sha256,
        checkpoint_sha256=checkpoint_sha256,
        parameter_state_sha256=parameter_state_sha256,
        training_code_sha256=preregistration.training_code_sha256,
        inference_code_sha256=preregistration.inference_code_sha256,
        optimizer_protocol_sha256=preregistration.optimizer_protocol_sha256,
        optimizer_state_sha256=optimizer_state_sha256,
        optimizer_parameter_group_audit_sha256=(optimizer_parameter_group_audit_sha256),
        dataset_access_audit_sha256=dataset_access_audit_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        training_log_sha256=training_log_sha256,
        random_seed=random_seed,
        completed_successfully=True,
    )
    result.validate_preregistration(preregistration)
    return result


__all__ = [
    "PASSIVE_QUADRATIC_CONSTRUCTION_ID",
    "PASSIVE_TRAINING_PREREGISTRATION_VERSION",
    "PASSIVE_TRAINING_RUN_BINDING_VERSION",
    "PassiveTrainingDataSplit",
    "PassiveTrainingDomain",
    "PassiveTrainingPreregistration",
    "PassiveTrainingRunBinding",
    "PassiveTrainingTarget",
    "bind_passive_training_run",
]
