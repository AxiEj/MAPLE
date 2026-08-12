"""Generic model-to-state-equation and vacuum-scalar adapters.

The public model protocols return immutable, provenance-bound state objects.
The reduced equation deliberately consumes bare numerical maps.  These thin
adapters are the only bridge between those contracts; the generic solver never
inspects a model family or calculator name.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    FieldResponsiveModel,
    VacuumEnergyModel,
    atom_count,
    require_model_methods,
    validate_model_identity,
    validate_source_evaluation,
    validate_source_model_identity,
    validate_vacuum_evaluation,
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty stable identifier.")
    return value.strip()


def _model_configuration_sha256(model: object) -> str:
    method = getattr(model, "configuration_sha256", None)
    if not callable(method):
        raise TypeError("model.configuration_sha256 must be callable.")
    value = method()
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError("model.configuration_sha256 must return a SHA256 digest.")
    return value.lower()


def _adapter_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ElectronicResponseEquationAdapter:
    """Expose one :class:`FieldResponsiveModel` as reduced-equation maps."""

    model: FieldResponsiveModel
    coupling_id: str
    provider_id: str = ""
    provenance_sha256: str = ""
    _construction_configuration_sha256: str = ""

    def __post_init__(self) -> None:
        provenance = validate_source_model_identity(self.model)
        require_model_methods(
            self.model,
            "source_jvp",
            "source_vjp",
            "source_position_vjp",
            "configuration_sha256",
        )
        coupling_id = _text(self.coupling_id, "coupling_id")
        provider_id = f"{provenance.provider_id}.reduced-equation-adapter.v1"
        object.__setattr__(self, "coupling_id", coupling_id)
        object.__setattr__(self, "provider_id", provider_id)
        configuration = self._current_configuration_sha256()
        object.__setattr__(self, "_construction_configuration_sha256", configuration)
        object.__setattr__(
            self,
            "provenance_sha256",
            _hash(
                {
                    "provider_id": provider_id,
                    "model_provenance_sha256": provenance.sha256,
                    "configuration_sha256": configuration,
                }
            ),
        )

    @property
    def model_profile_id(self) -> str:
        return self.model.provenance.model_profile_id

    @property
    def source_space(self):
        return self.model.source_space

    @property
    def field_space(self):
        return self.model.field_space

    def _current_configuration_sha256(self) -> str:
        provenance = validate_source_model_identity(self.model)
        return _hash(
            {
                "schema": "route2-electronic-response-equation-adapter-v1",
                "adapter_source_sha256": _adapter_source_sha256(),
                "coupling_id": self.coupling_id,
                "model_provider_id": provenance.provider_id,
                "model_profile_id": provenance.model_profile_id,
                "model_provenance_sha256": provenance.sha256,
                "model_configuration_sha256": _model_configuration_sha256(self.model),
                "source_space_sha256": self.model.source_space.metadata_hash(),
                "field_space_sha256": self.model.field_space.metadata_hash(),
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if (
            self._construction_configuration_sha256
            and current != self._construction_configuration_sha256
        ):
            raise ValueError(
                "Electronic model/equation adapter configuration drifted after construction."
            )
        return current

    def evaluate_source(self, geometry: Any, field: np.ndarray) -> np.ndarray:
        state = validate_source_evaluation(
            self.model,
            geometry,
            field,
            need_fixed_field_forces=False,
        )
        return np.array(state.source, copy=True)

    def field_jvp(
        self, geometry: Any, field: np.ndarray, field_direction: np.ndarray
    ) -> np.ndarray:
        validate_source_model_identity(self.model).domain.validate_atoms(geometry)
        count = atom_count(geometry)
        field_value = self.field_space.validate(field, atom_count=count)
        direction = self.field_space.validate(
            field_direction, atom_count=count, name="field_direction"
        )
        return self.source_space.validate(
            self.model.source_jvp(geometry, field_value, direction),
            atom_count=count,
            name="source_jvp",
        )

    def field_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray:
        validate_source_model_identity(self.model).domain.validate_atoms(geometry)
        count = atom_count(geometry)
        field_value = self.field_space.validate(field, atom_count=count)
        cotangent = self.source_space.validate(
            source_cotangent, atom_count=count, name="source_cotangent"
        )
        return self.field_space.validate(
            self.model.source_vjp(geometry, field_value, cotangent),
            atom_count=count,
            name="source_vjp",
        )

    def coordinate_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray:
        validate_source_model_identity(self.model).domain.validate_atoms(geometry)
        count = atom_count(geometry)
        field_value = self.field_space.validate(field, atom_count=count)
        cotangent = self.source_space.validate(
            source_cotangent, atom_count=count, name="source_cotangent"
        )
        result = np.asarray(
            self.model.source_position_vjp(geometry, field_value, cotangent),
            dtype=float,
        )
        if result.shape != (count, 3) or not np.all(np.isfinite(result)):
            raise ValueError(
                "source_position_vjp must be finite with shape (atom_count, 3)."
            )
        return result.reshape(-1).copy()


@dataclass(frozen=True, slots=True)
class VacuumScalarEquationAdapter:
    """Expose a :class:`VacuumEnergyModel` as the operational scalar leaf."""

    model: VacuumEnergyModel
    provider_id: str = ""
    provenance_sha256: str = ""
    _construction_configuration_sha256: str = ""

    def __post_init__(self) -> None:
        provenance = validate_model_identity(self.model)
        require_model_methods(self.model, "evaluate_vacuum", "configuration_sha256")
        provider_id = f"{provenance.provider_id}.vacuum-scalar-adapter.v1"
        object.__setattr__(self, "provider_id", provider_id)
        configuration = self._current_configuration_sha256()
        object.__setattr__(self, "_construction_configuration_sha256", configuration)
        object.__setattr__(
            self,
            "provenance_sha256",
            _hash(
                {
                    "provider_id": provider_id,
                    "model_provenance_sha256": provenance.sha256,
                    "configuration_sha256": configuration,
                }
            ),
        )

    @property
    def model_profile_id(self) -> str:
        return self.model.provenance.model_profile_id

    def _current_configuration_sha256(self) -> str:
        provenance = validate_model_identity(self.model)
        return _hash(
            {
                "schema": "route2-vacuum-scalar-equation-adapter-v1",
                "adapter_source_sha256": _adapter_source_sha256(),
                "model_provider_id": provenance.provider_id,
                "model_profile_id": provenance.model_profile_id,
                "model_provenance_sha256": provenance.sha256,
                "model_configuration_sha256": _model_configuration_sha256(self.model),
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if (
            self._construction_configuration_sha256
            and current != self._construction_configuration_sha256
        ):
            raise ValueError(
                "Vacuum model/scalar adapter configuration drifted after construction."
            )
        return current

    def evaluate_energy(self, geometry: Any) -> float:
        return validate_vacuum_evaluation(
            self.model, geometry, need_forces=False
        ).energy_eV

    def coordinate_gradient(self, geometry: Any) -> np.ndarray:
        state = validate_vacuum_evaluation(self.model, geometry, need_forces=True)
        if state.forces_eV_per_A is None:  # guarded by the validator
            raise RuntimeError("Vacuum model omitted requested forces.")
        return (-np.asarray(state.forces_eV_per_A, dtype=float)).reshape(-1).copy()


__all__ = ["ElectronicResponseEquationAdapter", "VacuumScalarEquationAdapter"]
