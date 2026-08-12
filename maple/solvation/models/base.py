"""Dependency-light capability contracts for Route-2 solute models.

The continuum stack consumes these structural protocols and never branches on a
model family or checkpoint name.  Array values are copied into read-only NumPy
buffers at state boundaries so that a converged state remains auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from maple.solvation.coupling.spaces import FieldDualSpace, SourceSpace
from maple.solvation.coupling.state_equation import geometry_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    digest = _text(value, name=name).lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(
            f"{name} must contain exactly 64 lowercase hexadecimal digits."
        )
    return digest


def _readonly_array(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {shape}; received {array.shape}."
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def array_sha256(value: object, *, name: str) -> str:
    """Hash a finite numeric array including shape and canonical float64 values."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite numeric array.")
    payload = {
        "kind": "route2-model-array-v1",
        "shape": list(array.shape),
        "values": array.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _python_float(value: object, *, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a Python float.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def atom_count(atoms: object) -> int:
    """Return the atom count without importing ASE or another optional runtime."""

    try:
        count = len(atoms)  # type: ignore[arg-type]
    except (TypeError, AttributeError) as exc:
        raise TypeError("atoms must be a sized geometry object.") from exc
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("atoms must contain at least one atom.")
    return count


def _integer_alias(
    info: Mapping[object, object], keys: tuple[str, ...], *, default: int, name: str
) -> int:
    values: list[int] = []
    for key in keys:
        if key not in info:
            continue
        raw = info[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise TypeError(f"atoms.info[{key!r}] for {name} must be an integer.")
        values.append(int(raw))
    if not values:
        return default
    if len(set(values)) != 1:
        raise ValueError(f"conflicting atoms.info aliases for {name}: {keys}.")
    return values[0]


def _model_charge_and_multiplicity(atoms: object) -> tuple[int, int]:
    raw_info = getattr(atoms, "info", {})
    if raw_info is None:
        raw_info = {}
    if not isinstance(raw_info, Mapping):
        raise TypeError("atoms.info must be a mapping when present.")
    charge = _integer_alias(
        raw_info, ("charge", "total_charge"), default=0, name="total charge"
    )
    multiplicity = _integer_alias(
        raw_info,
        ("mult", "spin", "multiplicity"),
        default=1,
        name="spin multiplicity",
    )
    if multiplicity < 1:
        raise ValueError("spin multiplicity must be positive.")
    return charge, multiplicity


def _atomic_numbers(atoms: object) -> tuple[int, ...]:
    getter = getattr(atoms, "get_atomic_numbers", None)
    if not callable(getter):
        raise TypeError("model atoms must expose callable get_atomic_numbers().")
    raw = np.asarray(getter())
    if (
        raw.ndim != 1
        or raw.size < 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.any(raw <= 0)
    ):
        raise ValueError(
            "model atomic numbers must be a non-empty positive integer vector."
        )
    return tuple(int(value) for value in raw)


def model_input_sha256(atoms: object) -> str:
    """Hash geometry plus the canonical charge/spin model-input identity.

    ``charge`` and ``total_charge`` are equal-priority aliases; ``mult``,
    ``spin`` and ``multiplicity`` are equal-priority multiplicity aliases.
    Present aliases must agree.  Missing metadata has explicit neutral-singlet
    defaults ``total_charge=0`` and ``multiplicity=1``.
    """

    charge, multiplicity = _model_charge_and_multiplicity(atoms)
    payload = {
        "kind": "route2-model-input-v1",
        "geometry_sha256": geometry_sha256(atoms),
        "total_charge": charge,
        "spin_multiplicity": multiplicity,
        "alias_policy": (
            "charge=total_charge;mult=spin=multiplicity;missing=neutral-singlet"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelDomain:
    """Explicit chemical and numerical domain of one provider artifact."""

    atomic_numbers: tuple[int, ...]
    total_charge_range: tuple[int, int]
    spin_multiplicities: tuple[int, ...]

    def __post_init__(self) -> None:
        numbers = tuple(self.atomic_numbers)
        spins = tuple(self.spin_multiplicities)
        charges = tuple(self.total_charge_range)
        if not numbers or any(
            isinstance(z, bool) or not isinstance(z, int) or z < 1 for z in numbers
        ):
            raise ValueError("atomic_numbers must contain positive integers.")
        if tuple(sorted(set(numbers))) != numbers:
            raise ValueError("atomic_numbers must be unique and sorted.")
        if len(charges) != 2 or any(
            isinstance(q, bool) or not isinstance(q, int) for q in charges
        ):
            raise ValueError("total_charge_range must contain two integers.")
        if charges[0] > charges[1]:
            raise ValueError(
                "total_charge_range lower bound must not exceed its upper bound."
            )
        if not spins or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in spins
        ):
            raise ValueError("spin_multiplicities must contain positive integers.")
        if tuple(sorted(set(spins))) != spins:
            raise ValueError("spin_multiplicities must be unique and sorted.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "total_charge_range", (charges[0], charges[1]))
        object.__setattr__(self, "spin_multiplicities", spins)

    def metadata(self) -> dict[str, object]:
        return {
            "atomic_numbers": list(self.atomic_numbers),
            "total_charge_range": list(self.total_charge_range),
            "spin_multiplicities": list(self.spin_multiplicities),
        }

    def validate_atoms(self, atoms: object) -> tuple[int, int]:
        """Fail closed when one model input lies outside this artifact domain."""

        numbers = _atomic_numbers(atoms)
        unsupported = sorted(set(numbers) - set(self.atomic_numbers))
        if unsupported:
            raise ValueError(f"atomic numbers outside model domain: {unsupported}.")
        charge, multiplicity = _model_charge_and_multiplicity(atoms)
        lower, upper = self.total_charge_range
        if not lower <= charge <= upper:
            raise ValueError(
                f"total charge {charge} is outside model domain [{lower}, {upper}]."
            )
        if multiplicity not in self.spin_multiplicities:
            raise ValueError(
                f"spin multiplicity {multiplicity} is outside model domain "
                f"{self.spin_multiplicities}."
            )
        return charge, multiplicity


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Mandatory release identity for a vacuum or field-response provider."""

    provider_id: str
    model_profile_id: str
    model_family: str
    checkpoint_sha256: str
    upstream_version: str
    upstream_commit: str
    inference_code_sha256: str
    dtype: str
    device: str
    domain: ModelDomain
    field_convention: str
    coordinate_frame_policy: str
    optimizer_parameter_groups_audited: bool = False
    optimizer_audit_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "model_profile_id",
            "model_family",
            "upstream_version",
            "upstream_commit",
            "dtype",
            "device",
            "field_convention",
            "coordinate_frame_policy",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256(self.checkpoint_sha256, name="checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "inference_code_sha256",
            _sha256(self.inference_code_sha256, name="inference_code_sha256"),
        )
        if not isinstance(self.domain, ModelDomain):
            raise TypeError("domain must be a ModelDomain.")
        if type(self.optimizer_parameter_groups_audited) is not bool:
            raise TypeError("optimizer_parameter_groups_audited must be a bool.")
        if self.optimizer_audit_evidence_sha256 is not None:
            object.__setattr__(
                self,
                "optimizer_audit_evidence_sha256",
                _sha256(
                    self.optimizer_audit_evidence_sha256,
                    name="optimizer_audit_evidence_sha256",
                ),
            )
        if self.optimizer_parameter_groups_audited != (
            self.optimizer_audit_evidence_sha256 is not None
        ):
            raise ValueError(
                "optimizer audit status and evidence SHA256 must be declared together."
            )

    def metadata(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "model_family": self.model_family,
            "checkpoint_sha256": self.checkpoint_sha256,
            "upstream_version": self.upstream_version,
            "upstream_commit": self.upstream_commit,
            "inference_code_sha256": self.inference_code_sha256,
            "dtype": self.dtype,
            "device": self.device,
            "domain": self.domain.metadata(),
            "field_convention": self.field_convention,
            "coordinate_frame_policy": self.coordinate_frame_policy,
            "optimizer_parameter_groups_audited": self.optimizer_parameter_groups_audited,
            "optimizer_audit_evidence_sha256": self.optimizer_audit_evidence_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class VacuumState:
    provider_id: str
    provenance_sha256: str
    model_input_sha256: str
    atom_count: int
    energy_eV: float
    need_forces: bool
    forces_eV_per_A: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, name="provider_id")
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            _sha256(self.provenance_sha256, name="provenance_sha256"),
        )
        object.__setattr__(
            self,
            "model_input_sha256",
            _sha256(self.model_input_sha256, name="model_input_sha256"),
        )
        if (
            isinstance(self.atom_count, bool)
            or not isinstance(self.atom_count, int)
            or self.atom_count < 1
        ):
            raise ValueError("atom_count must be a positive integer.")
        _python_float(self.energy_eV, name="energy_eV")
        if type(self.need_forces) is not bool:
            raise TypeError("need_forces must be a bool.")
        if self.forces_eV_per_A is not None:
            object.__setattr__(
                self,
                "forces_eV_per_A",
                _readonly_array(
                    self.forces_eV_per_A,
                    shape=(self.atom_count, 3),
                    name="forces_eV_per_A",
                ),
            )


@dataclass(frozen=True, slots=True)
class ElectronicSourceState:
    provider_id: str
    provenance_sha256: str
    model_input_sha256: str
    input_field_sha256: str
    source_space_sha256: str
    field_space_sha256: str
    atom_count: int
    source: np.ndarray
    need_fixed_field_forces: bool
    fixed_field_forces_eV_per_A: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, name="provider_id")
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            _sha256(self.provenance_sha256, name="provenance_sha256"),
        )
        for name in (
            "model_input_sha256",
            "input_field_sha256",
            "source_space_sha256",
            "field_space_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if (
            isinstance(self.atom_count, bool)
            or not isinstance(self.atom_count, int)
            or self.atom_count < 1
        ):
            raise ValueError("atom_count must be a positive integer.")
        array = np.asarray(self.source, dtype=float)
        if (
            array.ndim != 2
            or array.shape[0] != self.atom_count
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(
                "source must be a finite (atom_count, component_count) array."
            )
        source = np.array(array, copy=True)
        source.setflags(write=False)
        object.__setattr__(self, "source", source)
        if type(self.need_fixed_field_forces) is not bool:
            raise TypeError("need_fixed_field_forces must be a bool.")
        if self.fixed_field_forces_eV_per_A is not None:
            object.__setattr__(
                self,
                "fixed_field_forces_eV_per_A",
                _readonly_array(
                    self.fixed_field_forces_eV_per_A,
                    shape=(self.atom_count, 3),
                    name="fixed_field_forces_eV_per_A",
                ),
            )


@dataclass(frozen=True, slots=True)
class FieldEnergyState:
    provider_id: str
    provenance_sha256: str
    model_input_sha256: str
    input_field_sha256: str
    source_space_sha256: str
    field_space_sha256: str
    atom_count: int
    energy_eV: float
    need_fixed_field_forces: bool
    fixed_field_forces_eV_per_A: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, name="provider_id")
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            _sha256(self.provenance_sha256, name="provenance_sha256"),
        )
        for name in (
            "model_input_sha256",
            "input_field_sha256",
            "source_space_sha256",
            "field_space_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if (
            isinstance(self.atom_count, bool)
            or not isinstance(self.atom_count, int)
            or self.atom_count < 1
        ):
            raise ValueError("atom_count must be a positive integer.")
        _python_float(self.energy_eV, name="energy_eV")
        if type(self.need_fixed_field_forces) is not bool:
            raise TypeError("need_fixed_field_forces must be a bool.")
        if self.fixed_field_forces_eV_per_A is not None:
            object.__setattr__(
                self,
                "fixed_field_forces_eV_per_A",
                _readonly_array(
                    self.fixed_field_forces_eV_per_A,
                    shape=(self.atom_count, 3),
                    name="fixed_field_forces_eV_per_A",
                ),
            )


@dataclass(frozen=True, slots=True)
class VariationalIdentityDeclaration:
    """Content-addressed provider declaration; never release admission."""

    provider_id: str
    provenance_sha256: str
    evidence_sha256: str
    field_energy_source_finite_difference: bool = False
    source_jvp_vjp_transpose: bool = False
    q_self_adjoint_susceptibility: bool = False
    passive_stable_spectrum: bool = False
    locally_invertible_unique_root: bool = False
    field_sign_gauge_origin: bool = False
    finite_field_domain: bool = False
    same_checkpoint_and_inference: bool = False
    no_untrained_response_submodule: bool = False
    full_coordinate_derivative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, name="provider_id")
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            _sha256(self.provenance_sha256, name="provenance_sha256"),
        )
        object.__setattr__(
            self,
            "evidence_sha256",
            _sha256(self.evidence_sha256, name="evidence_sha256"),
        )
        for name in self._gate_names:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")

    @property
    def _gate_names(self) -> tuple[str, ...]:
        return (
            "field_energy_source_finite_difference",
            "source_jvp_vjp_transpose",
            "q_self_adjoint_susceptibility",
            "passive_stable_spectrum",
            "locally_invertible_unique_root",
            "field_sign_gauge_origin",
            "finite_field_domain",
            "same_checkpoint_and_inference",
            "no_untrained_response_submodule",
            "full_coordinate_derivative",
        )

    @property
    def complete(self) -> bool:
        return all(getattr(self, name) for name in self._gate_names)


@runtime_checkable
class VacuumEnergyModel(Protocol):
    provider_id: str
    model_profile_id: str
    provenance_sha256: str
    dtype: str
    device: str
    domain: ModelDomain
    coordinate_frame_policy: str
    provenance: ModelProvenance

    def evaluate_vacuum(self, atoms: object, *, need_forces: bool) -> VacuumState: ...


@runtime_checkable
class ElectronicSourceModel(Protocol):
    source_space: SourceSpace
    field_space: FieldDualSpace
    provider_id: str
    model_profile_id: str
    provenance_sha256: str
    dtype: str
    device: str
    domain: ModelDomain
    field_convention: str
    coordinate_frame_policy: str
    provenance: ModelProvenance

    def evaluate_source(
        self, atoms: object, field: object, *, need_fixed_field_forces: bool
    ) -> ElectronicSourceState: ...


@runtime_checkable
class FieldResponsiveModel(ElectronicSourceModel, Protocol):
    def source_jvp(
        self, atoms: object, field: object, field_direction: object
    ) -> np.ndarray: ...
    def source_vjp(
        self, atoms: object, field: object, source_cotangent: object
    ) -> np.ndarray: ...
    def source_position_vjp(
        self, atoms: object, field: object, source_cotangent: object
    ) -> np.ndarray: ...


@runtime_checkable
class VariationalFieldModel(FieldResponsiveModel, Protocol):
    def field_energy(
        self, atoms: object, field: object, *, need_fixed_field_forces: bool
    ) -> FieldEnergyState: ...
    def field_energy_source_identity_evidence(
        self,
    ) -> VariationalIdentityDeclaration: ...


def require_model_methods(model: object, *method_names: str) -> None:
    """Enforce runtime callability; protocols remain typing aids only."""

    for name in method_names:
        if not callable(getattr(model, name, None)):
            raise TypeError(f"model method {name} must be callable.")


def validate_model_identity(model: object) -> ModelProvenance:
    """Validate duplicated fast-path metadata against immutable provenance."""

    provenance = getattr(model, "provenance", None)
    if not isinstance(provenance, ModelProvenance):
        raise TypeError("model provenance must be ModelProvenance.")
    bindings = {
        "provider_id": provenance.provider_id,
        "model_profile_id": provenance.model_profile_id,
        "provenance_sha256": provenance.sha256,
        "dtype": provenance.dtype,
        "device": provenance.device,
        "domain": provenance.domain,
        "coordinate_frame_policy": provenance.coordinate_frame_policy,
    }
    for name, expected in bindings.items():
        if not hasattr(model, name):
            raise TypeError(f"model must declare {name}.")
        if getattr(model, name) != expected:
            raise ValueError(f"model {name} does not match immutable provenance.")
    return provenance


def validate_vacuum_evaluation(
    model: VacuumEnergyModel, atoms: object, *, need_forces: bool
) -> VacuumState:
    provenance = validate_model_identity(model)
    provenance.domain.validate_atoms(atoms)
    require_model_methods(model, "evaluate_vacuum")
    state = model.evaluate_vacuum(atoms, need_forces=need_forces)
    if not isinstance(state, VacuumState):
        raise TypeError("evaluate_vacuum must return VacuumState.")
    _validate_state_binding(model, state, atoms)
    if state.need_forces is not need_forces:
        raise ValueError(
            "vacuum state request flag does not match the evaluation request."
        )
    if need_forces and state.forces_eV_per_A is None:
        raise ValueError("requested vacuum forces were not returned.")
    return state


def validate_source_model_identity(model: ElectronicSourceModel) -> ModelProvenance:
    """Validate source/field spaces in addition to common provider metadata."""

    provenance = validate_model_identity(model)
    require_model_methods(model, "evaluate_source")
    if not isinstance(model.source_space, SourceSpace):
        raise TypeError("model source_space must be SourceSpace.")
    if not isinstance(model.field_space, FieldDualSpace):
        raise TypeError("model field_space must be FieldDualSpace.")
    if not hasattr(model, "field_convention"):
        raise TypeError("source model must declare field_convention.")
    if model.field_convention != provenance.field_convention:
        raise ValueError("model field_convention does not match immutable provenance.")
    if model.source_space != model.field_space.source_space:
        raise ValueError(
            "model source_space must equal the FieldDualSpace source dual."
        )
    if model.field_convention != model.field_space.field_convention:
        raise ValueError(
            "model field convention must match the authoritative FieldDualSpace."
        )
    return provenance


def validate_source_evaluation(
    model: ElectronicSourceModel,
    atoms: object,
    field: object,
    *,
    need_fixed_field_forces: bool,
) -> ElectronicSourceState:
    provenance = validate_source_model_identity(model)
    provenance.domain.validate_atoms(atoms)
    count = atom_count(atoms)
    field_value = model.field_space.validate(field, atom_count=count)
    state = model.evaluate_source(
        atoms, field_value, need_fixed_field_forces=need_fixed_field_forces
    )
    if not isinstance(state, ElectronicSourceState):
        raise TypeError("evaluate_source must return ElectronicSourceState.")
    _validate_state_binding(model, state, atoms)
    _validate_field_state_binding(
        model,
        state,
        field_value,
        need_fixed_field_forces=need_fixed_field_forces,
    )
    model.source_space.validate(state.source, atom_count=count)
    if need_fixed_field_forces and state.fixed_field_forces_eV_per_A is None:
        raise ValueError("requested fixed-field forces were not returned.")
    return state


def validate_response_linearization(
    model: FieldResponsiveModel,
    atoms: object,
    field: object,
    *,
    field_direction: object,
    source_cotangent: object,
    transpose_atol: float = 1.0e-10,
    transpose_rtol: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate and validate source JVP, source VJP, and coordinate VJP shapes."""

    provenance = validate_source_model_identity(model)
    provenance.domain.validate_atoms(atoms)
    require_model_methods(model, "source_jvp", "source_vjp", "source_position_vjp")
    count = atom_count(atoms)
    field_value = model.field_space.validate(field, atom_count=count)
    direction = model.field_space.validate(
        field_direction, atom_count=count, name="field_direction"
    )
    cotangent = model.source_space.validate(
        source_cotangent, atom_count=count, name="source_cotangent"
    )
    jvp = model.source_space.validate(
        model.source_jvp(atoms, field_value, direction),
        atom_count=count,
        name="source_jvp",
    )
    vjp = model.field_space.validate(
        model.source_vjp(atoms, field_value, cotangent),
        atom_count=count,
        name="source_vjp",
    )
    position_vjp = _readonly_array(
        model.source_position_vjp(atoms, field_value, cotangent),
        shape=(count, 3),
        name="source_position_vjp",
    )
    if not np.isfinite(transpose_atol) or transpose_atol < 0.0:
        raise ValueError("transpose_atol must be finite and non-negative.")
    if not np.isfinite(transpose_rtol) or transpose_rtol < 0.0:
        raise ValueError("transpose_rtol must be finite and non-negative.")
    lhs = float(np.vdot(jvp, cotangent))
    rhs = float(np.vdot(direction, vjp))
    if not np.isclose(lhs, rhs, atol=transpose_atol, rtol=transpose_rtol):
        raise ValueError(
            f"source JVP/VJP transpose contract failed: lhs={lhs:.16e}, rhs={rhs:.16e}."
        )
    return jvp, vjp, position_vjp


def validate_field_energy_evaluation(
    model: VariationalFieldModel,
    atoms: object,
    field: object,
    *,
    need_fixed_field_forces: bool,
) -> FieldEnergyState:
    """Validate the energy branch used by the strict field/source identity."""

    provenance = validate_source_model_identity(model)
    provenance.domain.validate_atoms(atoms)
    require_model_methods(
        model, "source_jvp", "source_vjp", "source_position_vjp", "field_energy"
    )
    count = atom_count(atoms)
    field_value = model.field_space.validate(field, atom_count=count)
    state = model.field_energy(
        atoms, field_value, need_fixed_field_forces=need_fixed_field_forces
    )
    if not isinstance(state, FieldEnergyState):
        raise TypeError("field_energy must return FieldEnergyState.")
    _validate_state_binding(model, state, atoms)
    _validate_field_state_binding(
        model,
        state,
        field_value,
        need_fixed_field_forces=need_fixed_field_forces,
    )
    if need_fixed_field_forces and state.fixed_field_forces_eV_per_A is None:
        raise ValueError("requested fixed-field energy forces were not returned.")
    return state


def validate_variational_declaration(
    model: VariationalFieldModel,
) -> VariationalIdentityDeclaration:
    """Validate provider declarations without granting release Tier V."""

    provenance = validate_source_model_identity(model)
    require_model_methods(
        model,
        "source_jvp",
        "source_vjp",
        "source_position_vjp",
        "field_energy",
        "field_energy_source_identity_evidence",
    )
    evidence = model.field_energy_source_identity_evidence()
    if not isinstance(evidence, VariationalIdentityDeclaration):
        raise TypeError(
            "field_energy_source_identity_evidence must return VariationalIdentityDeclaration."
        )
    if (
        evidence.provider_id != provenance.provider_id
        or evidence.provenance_sha256 != provenance.sha256
    ):
        raise ValueError(
            "variational declaration is not bound to this provider provenance."
        )
    if not provenance.optimizer_parameter_groups_audited:
        raise ValueError("Tier V requires optimizer parameter-group audit evidence.")
    if not evidence.complete:
        missing = [name for name in evidence._gate_names if not getattr(evidence, name)]
        raise ValueError(
            f"variational declaration is incomplete: {', '.join(missing)}."
        )
    return evidence


def _validate_state_binding(model: object, state: object, atoms: object) -> None:
    if getattr(state, "provider_id") != getattr(model, "provider_id"):
        raise ValueError("state provider_id does not match the evaluating model.")
    if getattr(state, "provenance_sha256") != getattr(model, "provenance_sha256"):
        raise ValueError("state provenance SHA256 does not match the evaluating model.")
    if getattr(state, "atom_count") != atom_count(atoms):
        raise ValueError("state atom_count does not match the evaluated geometry.")
    if getattr(state, "model_input_sha256") != model_input_sha256(atoms):
        raise ValueError(
            "state model-input SHA256 does not match geometry, charge, and spin."
        )


def _validate_field_state_binding(
    model: ElectronicSourceModel,
    state: ElectronicSourceState | FieldEnergyState,
    field: np.ndarray,
    *,
    need_fixed_field_forces: bool,
) -> None:
    expected = {
        "input_field_sha256": array_sha256(field, name="field"),
        "source_space_sha256": model.source_space.metadata_hash(),
        "field_space_sha256": model.field_space.metadata_hash(),
        "need_fixed_field_forces": need_fixed_field_forces,
    }
    for name, value in expected.items():
        if getattr(state, name) != value:
            raise ValueError(f"state {name} does not match the evaluation input.")


__all__ = [
    "ElectronicSourceModel",
    "ElectronicSourceState",
    "FieldEnergyState",
    "FieldResponsiveModel",
    "ModelDomain",
    "ModelProvenance",
    "VacuumEnergyModel",
    "VacuumState",
    "VariationalFieldModel",
    "VariationalIdentityDeclaration",
    "array_sha256",
    "atom_count",
    "model_input_sha256",
    "require_model_methods",
    "validate_field_energy_evaluation",
    "validate_model_identity",
    "validate_response_linearization",
    "validate_source_evaluation",
    "validate_source_model_identity",
    "validate_vacuum_evaluation",
    "validate_variational_declaration",
]
