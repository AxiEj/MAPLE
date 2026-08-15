"""Same-scalar geometry-mediated source/continuum composition.

For a field-independent geometry-adaptive source ``c(R)``, this module owns

``E(R) = E_vac(R) + 1/2 <c(R), P_R(c(R))>_Q``.

Its derivative is assembled exactly once from model and continuum protocol
VJPs.  The construction deliberately has no electronic fixed-point variable:
it represents geometry-mediated charge response and must not be described as
fixed-geometry electronic mutual polarization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from maple.solvation.api.profiles import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1,
    get_scalar_definition,
)
from maple.solvation.models.base import (
    FieldResponsiveModel,
    atom_count,
    model_charge_and_multiplicity,
    require_model_methods,
    validate_source_evaluation,
    validate_source_model_identity,
    validate_vacuum_evaluation,
)

from .energy import nonlinear_half_coupling
from .metrics import PairingMetric, get_pairing_metric
from .spaces import get_coordinate_contract, get_field_dual_space, get_source_space
from .state_equation import ContinuumResponseProvider, provider_behavior_sha256

GEOMETRY_MEDIATED_RECIPROCITY_SEED = 20260815
GEOMETRY_MEDIATED_RECIPROCITY_PROBES = 4
GEOMETRY_MEDIATED_CHARGE_FD_STEPS_E = (1.0e-3, 3.0e-4, 1.0e-4)
GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV = 1.0e-8
GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE = 1.0e-8
GEOMETRY_MEDIATED_CHARGE_FD_ABSOLUTE_TOLERANCE_EV_PER_E = 5.0e-6
GEOMETRY_MEDIATED_CHARGE_FD_RELATIVE_TOLERANCE = 5.0e-5
GEOMETRY_MEDIATED_GAUGE_VJP_TOLERANCE_EV_PER_A = 1.0e-7


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string.")
    result = value.lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _relative_error(first: float, second: float, *, floor: float = 1.0e-15) -> float:
    return abs(first - second) / max(abs(first), abs(second), floor)


def _charge_tangent_direction(rng: np.random.Generator, atom_count: int) -> np.ndarray:
    values = rng.normal(size=atom_count)
    values -= float(np.mean(values))
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("deterministic fixed-charge tangent direction is singular.")
    result = np.zeros((atom_count, 4), dtype=float)
    result[:, 0] = values / norm
    return result


@dataclass(frozen=True, slots=True)
class BilinearReciprocityRecord:
    probe_index: int
    left_P_right_eV: float
    right_P_left_eV: float
    apply_adjoint_eV: float
    reciprocity_absolute_error_eV: float
    reciprocity_relative_error: float
    apply_adjoint_absolute_error_eV: float
    apply_adjoint_relative_error: float


@dataclass(frozen=True, slots=True)
class ChargeDirectionalFDRecord:
    probe_index: int
    step_e: float
    analytic_eV_per_e: float
    finite_difference_eV_per_e: float
    absolute_error_eV_per_e: float
    relative_error: float


@dataclass(frozen=True, slots=True)
class GeometryMediatedReciprocityAudit:
    """Deterministic metric, cotangent, and charge-gauge audit.

    The random probes live in the fixed-total-charge monopole tangent space.
    They are a fail-closed numerical audit of the discretized operator, not a
    proof of global reciprocity across geometries or cavity strata.
    """

    seed: int
    requested_probe_count: int
    bilinear_records: tuple[BilinearReciprocityRecord, ...]
    charge_directional_fd_records: tuple[ChargeDirectionalFDRecord, ...]
    source_gradient_half_error_eV_per_source_unit: float
    charge_gauge_vjp_norm_eV_per_A: float
    maximum_reciprocity_absolute_error_eV: float
    maximum_reciprocity_relative_error: float
    maximum_apply_adjoint_absolute_error_eV: float
    maximum_apply_adjoint_relative_error: float
    maximum_charge_fd_absolute_error_eV_per_e: float
    maximum_charge_fd_relative_error: float
    gate_passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "requested_probe_count": self.requested_probe_count,
            "effective_probe_count": len(self.bilinear_records),
            "bilinear_records": [
                {
                    name: getattr(record, name)
                    for name in BilinearReciprocityRecord.__dataclass_fields__
                }
                for record in self.bilinear_records
            ],
            "charge_directional_fd_records": [
                {
                    name: getattr(record, name)
                    for name in ChargeDirectionalFDRecord.__dataclass_fields__
                }
                for record in self.charge_directional_fd_records
            ],
            "source_gradient_half_error_eV_per_source_unit": (
                self.source_gradient_half_error_eV_per_source_unit
            ),
            "charge_gauge_vjp_norm_eV_per_A": self.charge_gauge_vjp_norm_eV_per_A,
            "maximum_reciprocity_absolute_error_eV": (
                self.maximum_reciprocity_absolute_error_eV
            ),
            "maximum_reciprocity_relative_error": (
                self.maximum_reciprocity_relative_error
            ),
            "maximum_apply_adjoint_absolute_error_eV": (
                self.maximum_apply_adjoint_absolute_error_eV
            ),
            "maximum_apply_adjoint_relative_error": (
                self.maximum_apply_adjoint_relative_error
            ),
            "maximum_charge_fd_absolute_error_eV_per_e": (
                self.maximum_charge_fd_absolute_error_eV_per_e
            ),
            "maximum_charge_fd_relative_error": self.maximum_charge_fd_relative_error,
            "thresholds": {
                "reciprocity_absolute_eV": (
                    GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV
                ),
                "reciprocity_relative": (
                    GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE
                ),
                "charge_fd_absolute_eV_per_e": (
                    GEOMETRY_MEDIATED_CHARGE_FD_ABSOLUTE_TOLERANCE_EV_PER_E
                ),
                "charge_fd_relative": (GEOMETRY_MEDIATED_CHARGE_FD_RELATIVE_TOLERANCE),
                "charge_gauge_vjp_norm_eV_per_A": (
                    GEOMETRY_MEDIATED_GAUGE_VJP_TOLERANCE_EV_PER_A
                ),
            },
            "gate_passed": self.gate_passed,
        }


@dataclass(frozen=True, slots=True)
class GeometryMediatedEnergyEvaluation:
    scalar_id: str
    profile_id: str
    vacuum_energy_eV: float
    continuum_energy_eV: float
    total_energy_eV: float

    def __post_init__(self) -> None:
        for name in ("scalar_id", "profile_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty.")
        values = (
            float(self.vacuum_energy_eV),
            float(self.continuum_energy_eV),
            float(self.total_energy_eV),
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("geometry-mediated energy leaves must be finite.")
        if abs(values[0] + values[1] - values[2]) > 1.0e-10:
            raise ValueError("geometry-mediated energy ledger does not close.")


@dataclass(frozen=True, slots=True)
class GeometryMediatedEvaluation:
    energy: GeometryMediatedEnergyEvaluation
    source: np.ndarray
    reaction_field: np.ndarray
    intrinsic_gradient_eV_per_A: np.ndarray
    continuum_fixed_source_gradient_eV_per_A: np.ndarray
    source_response_gradient_eV_per_A: np.ndarray
    total_gradient_eV_per_A: np.ndarray
    reciprocity_gradient_error_eV_per_source_unit: float
    reciprocity_audit: GeometryMediatedReciprocityAudit

    def __post_init__(self) -> None:
        if not isinstance(self.energy, GeometryMediatedEnergyEvaluation):
            raise TypeError("energy must be GeometryMediatedEnergyEvaluation.")
        source = np.asarray(self.source, dtype=float)
        field = np.asarray(self.reaction_field, dtype=float)
        if (
            source.ndim != 2
            or source.shape[1] != 4
            or field.shape != source.shape
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(field))
        ):
            raise ValueError("source and reaction field must be finite matching (N,4).")
        count = source.shape[0]
        source = _readonly(source, shape=(count, 4), name="source")
        field = _readonly(field, shape=(count, 4), name="reaction_field")
        names = (
            "intrinsic_gradient_eV_per_A",
            "continuum_fixed_source_gradient_eV_per_A",
            "source_response_gradient_eV_per_A",
            "total_gradient_eV_per_A",
        )
        gradients = {
            name: _readonly(getattr(self, name), shape=(count, 3), name=name)
            for name in names
        }
        expected = (
            gradients["intrinsic_gradient_eV_per_A"]
            + gradients["continuum_fixed_source_gradient_eV_per_A"]
            + gradients["source_response_gradient_eV_per_A"]
        )
        if not np.allclose(
            expected,
            gradients["total_gradient_eV_per_A"],
            rtol=0.0,
            atol=2.0e-10,
        ):
            raise ValueError(
                "geometry-mediated gradient component ledger does not close."
            )
        error = float(self.reciprocity_gradient_error_eV_per_source_unit)
        if not np.isfinite(error) or error < 0.0:
            raise ValueError(
                "reciprocity gradient error must be finite and non-negative."
            )
        if not isinstance(self.reciprocity_audit, GeometryMediatedReciprocityAudit):
            raise TypeError(
                "reciprocity_audit must be GeometryMediatedReciprocityAudit."
            )
        if not self.reciprocity_audit.gate_passed:
            raise ValueError("geometry-mediated reciprocity audit did not pass.")
        if not np.isclose(
            error,
            self.reciprocity_audit.source_gradient_half_error_eV_per_source_unit,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("reciprocity audit and scalar gradient ledger disagree.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "reaction_field", field)
        for name, values in gradients.items():
            object.__setattr__(self, name, values)

    @property
    def forces_eV_per_A(self) -> np.ndarray:
        result = -np.asarray(self.total_gradient_eV_per_A)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class GeometryMediatedElectrostaticScalar:
    """Registered diagnostic AIMNet2 geometry-mediated continuum scalar."""

    model: FieldResponsiveModel
    continuum: ContinuumResponseProvider
    scalar_id: str = DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1
    profile_id: str = DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_PROFILE_V1
    metric: PairingMetric | None = None
    reciprocity_tolerance: float = 1.0e-9
    reciprocity_seed: int = GEOMETRY_MEDIATED_RECIPROCITY_SEED
    reciprocity_probe_count: int = GEOMETRY_MEDIATED_RECIPROCITY_PROBES
    charge_fd_steps_e: tuple[float, ...] = GEOMETRY_MEDIATED_CHARGE_FD_STEPS_E
    _construction_fingerprint: str = ""

    def __post_init__(self) -> None:
        definition = get_scalar_definition(self.scalar_id)
        profile = get_solvation_profile(self.profile_id)
        provenance = validate_source_model_identity(self.model)
        require_model_methods(
            self.model,
            "configuration_sha256",
            "evaluate_vacuum",
            "source_position_vjp",
        )
        if getattr(self.model, "field_independent", None) is not True:
            raise ValueError(
                "geometry-mediated scalar requires field_independent=True."
            )
        if getattr(self.model, "electronic_mutual_polarization", None) is not False:
            raise ValueError(
                "geometry-mediated scalar requires electronic_mutual_polarization=False."
            )
        if definition.implementation_entry_point != (
            "maple.solvation.coupling.geometry_mediated:"
            "GeometryMediatedElectrostaticScalar"
        ):
            raise ValueError("registered scalar implementation entry point drifted.")
        if (
            profile.scalar_id != self.scalar_id
            or profile.state_equation_id != definition.state_equation_id
            or profile.model_profile != provenance.model_profile_id
        ):
            raise ValueError(
                "geometry-mediated scalar/profile/model binding is inconsistent."
            )
        for name in (
            "provider_id",
            "continuum_profile_id",
            "cavity_profile_id",
            "configuration_contract_id",
            "coupling_id",
            "scalar_id",
            "provenance_sha256",
        ):
            value = getattr(self.continuum, name, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"continuum.{name} must be a non-empty identity.")
        _sha(self.continuum.provenance_sha256, name="continuum.provenance_sha256")
        if not callable(getattr(self.continuum, "configuration_sha256", None)):
            raise TypeError("continuum.configuration_sha256 must be callable.")
        expected = {
            "continuum_profile_id": profile.continuum_profile,
            "cavity_profile_id": profile.cavity_profile,
            "configuration_contract_id": profile.continuum_configuration_contract_id,
            "coupling_id": profile.coupling_id,
            "scalar_id": profile.scalar_id,
        }
        for name, value in expected.items():
            if getattr(self.continuum, name) != value:
                raise ValueError(f"continuum.{name} does not match the profile.")
        if getattr(self.model, "coupling_id", None) != profile.coupling_id:
            raise ValueError("model coupling ID does not match the profile.")
        if getattr(self.continuum, "linear_response", None) is not True:
            raise ValueError(
                "geometry-mediated continuum requires linear_response=True."
            )
        if getattr(self.continuum, "reciprocal", None) is not True:
            raise ValueError("geometry-mediated continuum requires reciprocal=True.")
        registered_source = get_source_space(profile.source_space_id)
        registered_field = get_field_dual_space(profile.field_space_id)
        registered_metric = get_pairing_metric(profile.pairing_id)
        coordinate_contract = get_coordinate_contract(profile.coordinate_contract_id)
        if (
            coordinate_contract.source_space.metadata_hash()
            != registered_source.metadata_hash()
        ):
            raise ValueError(
                "direct-source coordinate contract uses a different source space."
            )
        for provider in (self.model, self.continuum):
            if (
                provider.source_space.metadata_hash()
                != registered_source.metadata_hash()
                or provider.field_space.metadata_hash()
                != registered_field.metadata_hash()
            ):
                raise ValueError(
                    "provider source/field identity does not match the profile."
                )
        metric = registered_metric if self.metric is None else self.metric
        if not isinstance(metric, PairingMetric):
            raise TypeError("metric must be PairingMetric or None.")
        if metric.metadata_hash() != registered_metric.metadata_hash():
            raise ValueError("geometry-mediated pairing does not match the profile.")
        tolerance = float(self.reciprocity_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("reciprocity_tolerance must be finite and positive.")
        if isinstance(self.reciprocity_seed, bool) or not isinstance(
            self.reciprocity_seed, int
        ):
            raise TypeError("reciprocity_seed must be an integer.")
        if (
            isinstance(self.reciprocity_probe_count, bool)
            or not isinstance(self.reciprocity_probe_count, int)
            or self.reciprocity_probe_count < 1
        ):
            raise ValueError("reciprocity_probe_count must be a positive integer.")
        steps = tuple(float(value) for value in self.charge_fd_steps_e)
        if (
            len(steps) < 3
            or any(not math.isfinite(value) or value <= 0.0 for value in steps)
            or any(first <= second for first, second in zip(steps, steps[1:]))
        ):
            raise ValueError(
                "charge_fd_steps_e must contain at least three strictly decreasing "
                "positive finite steps."
            )
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "reciprocity_tolerance", tolerance)
        object.__setattr__(self, "charge_fd_steps_e", steps)
        object.__setattr__(
            self, "_construction_fingerprint", self._current_fingerprint_sha256()
        )

    def _current_fingerprint_sha256(self) -> str:
        return _hash(
            {
                "schema": "route2-geometry-mediated-electrostatic-scalar-v1",
                "scalar_id": self.scalar_id,
                "profile_id": self.profile_id,
                "exact_formula": get_scalar_definition(self.scalar_id).exact_formula,
                "metric_sha256": self.metric.metadata_hash(),
                "reciprocity_tolerance": self.reciprocity_tolerance,
                "reciprocity_seed": self.reciprocity_seed,
                "reciprocity_probe_count": self.reciprocity_probe_count,
                "charge_fd_steps_e": list(self.charge_fd_steps_e),
                "model": {
                    "provider_id": self.model.provider_id,
                    "provenance_sha256": self.model.provenance_sha256,
                    "configuration_sha256": self.model.configuration_sha256(),
                    "behavior_sha256": provider_behavior_sha256(
                        self.model,
                        (
                            "configuration_sha256",
                            "evaluate_vacuum",
                            "evaluate_source",
                            "source_position_vjp",
                        ),
                        label="geometry_mediated_model",
                    ),
                },
                "continuum": {
                    "provider_id": self.continuum.provider_id,
                    "provenance_sha256": self.continuum.provenance_sha256,
                    "configuration_sha256": self.continuum.configuration_sha256(),
                    "behavior_sha256": provider_behavior_sha256(
                        self.continuum,
                        (
                            "configuration_sha256",
                            "evaluate_field",
                            "source_vjp",
                            "coordinate_vjp",
                        ),
                        label="geometry_mediated_continuum",
                    ),
                },
            }
        )

    def fingerprint_sha256(self) -> str:
        current = self._current_fingerprint_sha256()
        if self._construction_fingerprint and current != self._construction_fingerprint:
            raise ValueError("geometry-mediated scalar/provider configuration drifted.")
        return current

    def _source(self, geometry: Any) -> tuple[np.ndarray, np.ndarray]:
        count = atom_count(geometry)
        zero_field = np.zeros(self.model.field_space.shape(count), dtype=float)
        state = validate_source_evaluation(
            self.model,
            geometry,
            zero_field,
            need_fixed_field_forces=False,
        )
        source = np.asarray(state.source, dtype=float)
        charge, _ = model_charge_and_multiplicity(geometry)
        if (
            abs(self.model.source_space.total_charge(source, atom_count=count) - charge)
            > 1.0e-10
        ):
            raise ValueError("geometry-mediated source violates total charge.")
        if not np.array_equal(source[:, 1:], np.zeros_like(source[:, 1:])):
            raise ValueError(
                "AIMNet2 point-l0 profile requires exactly zero l=1 source."
            )
        return np.array(source, copy=True), zero_field

    def evaluate_energy_components(
        self, geometry: Any
    ) -> GeometryMediatedEnergyEvaluation:
        self.fingerprint_sha256()
        source, _ = self._source(geometry)
        count = source.shape[0]
        field = self.model.field_space.validate(
            self.continuum.evaluate_field(geometry, source),
            atom_count=count,
            name="continuum field",
        )
        continuum_energy = 0.5 * self.metric.pair(source, field)
        vacuum = validate_vacuum_evaluation(self.model, geometry, need_forces=False)
        return GeometryMediatedEnergyEvaluation(
            scalar_id=self.scalar_id,
            profile_id=self.profile_id,
            vacuum_energy_eV=vacuum.energy_eV,
            continuum_energy_eV=continuum_energy,
            total_energy_eV=vacuum.energy_eV + continuum_energy,
        )

    def evaluate_energy(self, geometry: Any) -> float:
        return self.evaluate_energy_components(geometry).total_energy_eV

    def _reciprocity_audit(
        self,
        geometry: Any,
        source: np.ndarray,
        zero_field: np.ndarray,
        reaction_field: np.ndarray,
        *,
        source_gradient_half_error: float,
    ) -> GeometryMediatedReciprocityAudit:
        count = source.shape[0]
        reaction_field = self.model.field_space.validate(
            reaction_field, atom_count=count, name="continuum field"
        )
        gauge_cotangent = np.zeros_like(source)
        gauge_cotangent[:, 0] = 1.0
        gauge_vjp = np.asarray(
            self.model.source_position_vjp(
                geometry,
                zero_field,
                gauge_cotangent,
            ),
            dtype=float,
        )
        if gauge_vjp.shape != (count, 3) or not np.all(np.isfinite(gauge_vjp)):
            raise ValueError("model charge-gauge VJP must be finite with shape (N,3).")
        gauge_norm = float(np.linalg.norm(gauge_vjp))

        bilinear_records: list[BilinearReciprocityRecord] = []
        charge_records: list[ChargeDirectionalFDRecord] = []
        if count > 1:
            rng = np.random.default_rng(self.reciprocity_seed)
            for probe_index in range(self.reciprocity_probe_count):
                left = _charge_tangent_direction(rng, count)
                right = _charge_tangent_direction(rng, count)
                left_field = self.model.field_space.validate(
                    self.continuum.evaluate_field(geometry, left),
                    atom_count=count,
                    name="left-probe continuum field",
                )
                right_field = self.model.field_space.validate(
                    self.continuum.evaluate_field(geometry, right),
                    atom_count=count,
                    name="right-probe continuum field",
                )
                left_P_right = self.metric.pair(left, right_field)
                right_P_left = self.metric.pair(right, left_field)
                field_cotangent = self.metric.source_to_field_dual(left)
                adjoint_left = self.model.source_space.validate(
                    self.continuum.source_vjp(
                        geometry,
                        source,
                        field_cotangent,
                    ),
                    atom_count=count,
                    name="continuum probe source VJP",
                )
                apply_adjoint = float(np.vdot(adjoint_left, right))
                bilinear_records.append(
                    BilinearReciprocityRecord(
                        probe_index=probe_index,
                        left_P_right_eV=left_P_right,
                        right_P_left_eV=right_P_left,
                        apply_adjoint_eV=apply_adjoint,
                        reciprocity_absolute_error_eV=abs(left_P_right - right_P_left),
                        reciprocity_relative_error=_relative_error(
                            left_P_right, right_P_left
                        ),
                        apply_adjoint_absolute_error_eV=abs(
                            left_P_right - apply_adjoint
                        ),
                        apply_adjoint_relative_error=_relative_error(
                            left_P_right, apply_adjoint
                        ),
                    )
                )

                analytic = self.metric.pair(left, reaction_field)
                for step in self.charge_fd_steps_e:
                    plus = source + step * left
                    minus = source - step * left
                    plus_energy = 0.5 * self.metric.pair(
                        plus,
                        self.continuum.evaluate_field(geometry, plus),
                    )
                    minus_energy = 0.5 * self.metric.pair(
                        minus,
                        self.continuum.evaluate_field(geometry, minus),
                    )
                    finite_difference = (plus_energy - minus_energy) / (2.0 * step)
                    charge_records.append(
                        ChargeDirectionalFDRecord(
                            probe_index=probe_index,
                            step_e=step,
                            analytic_eV_per_e=analytic,
                            finite_difference_eV_per_e=finite_difference,
                            absolute_error_eV_per_e=abs(analytic - finite_difference),
                            relative_error=_relative_error(
                                analytic,
                                finite_difference,
                                floor=1.0e-12,
                            ),
                        )
                    )

        max_reciprocity_absolute = max(
            (record.reciprocity_absolute_error_eV for record in bilinear_records),
            default=0.0,
        )
        max_reciprocity_relative = max(
            (record.reciprocity_relative_error for record in bilinear_records),
            default=0.0,
        )
        max_adjoint_absolute = max(
            (record.apply_adjoint_absolute_error_eV for record in bilinear_records),
            default=0.0,
        )
        max_adjoint_relative = max(
            (record.apply_adjoint_relative_error for record in bilinear_records),
            default=0.0,
        )
        max_fd_absolute = max(
            (record.absolute_error_eV_per_e for record in charge_records),
            default=0.0,
        )
        max_fd_relative = max(
            (record.relative_error for record in charge_records),
            default=0.0,
        )
        gate = (
            source_gradient_half_error
            <= self.reciprocity_tolerance
            * max(1.0, float(np.linalg.norm(reaction_field)))
            and max_reciprocity_absolute
            <= GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV
            and max_reciprocity_relative
            <= GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE
            and max_adjoint_absolute
            <= GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV
            and max_adjoint_relative <= GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE
            and max_fd_absolute
            <= GEOMETRY_MEDIATED_CHARGE_FD_ABSOLUTE_TOLERANCE_EV_PER_E
            and max_fd_relative <= GEOMETRY_MEDIATED_CHARGE_FD_RELATIVE_TOLERANCE
            and gauge_norm <= GEOMETRY_MEDIATED_GAUGE_VJP_TOLERANCE_EV_PER_A
        )
        return GeometryMediatedReciprocityAudit(
            seed=self.reciprocity_seed,
            requested_probe_count=self.reciprocity_probe_count,
            bilinear_records=tuple(bilinear_records),
            charge_directional_fd_records=tuple(charge_records),
            source_gradient_half_error_eV_per_source_unit=(source_gradient_half_error),
            charge_gauge_vjp_norm_eV_per_A=gauge_norm,
            maximum_reciprocity_absolute_error_eV=max_reciprocity_absolute,
            maximum_reciprocity_relative_error=max_reciprocity_relative,
            maximum_apply_adjoint_absolute_error_eV=max_adjoint_absolute,
            maximum_apply_adjoint_relative_error=max_adjoint_relative,
            maximum_charge_fd_absolute_error_eV_per_e=max_fd_absolute,
            maximum_charge_fd_relative_error=max_fd_relative,
            gate_passed=gate,
        )

    def evaluate(self, geometry: Any) -> GeometryMediatedEvaluation:
        self.fingerprint_sha256()
        source, zero_field = self._source(geometry)
        count = source.shape[0]
        continuum = nonlinear_half_coupling(
            self.continuum, geometry, source, metric=self.metric
        )
        direct = np.asarray(continuum.direct_source_gradient)
        response = np.asarray(continuum.response_source_gradient)
        reciprocity_error = float(np.linalg.norm(direct - response))
        scale = max(1.0, float(np.linalg.norm(direct)), float(np.linalg.norm(response)))
        if reciprocity_error > self.reciprocity_tolerance * scale:
            raise ValueError(
                "continuum source derivative violates fixed-geometry reciprocity: "
                f"error={reciprocity_error:.6e}."
            )
        reaction_field = self.model.field_space.validate(
            self.continuum.evaluate_field(geometry, source),
            atom_count=count,
            name="continuum field",
        )
        audit = self._reciprocity_audit(
            geometry,
            source,
            zero_field,
            reaction_field,
            source_gradient_half_error=reciprocity_error,
        )
        if not audit.gate_passed:
            raise ValueError(
                "geometry-mediated metric/reciprocity/charge-gauge audit failed."
            )
        vacuum = validate_vacuum_evaluation(self.model, geometry, need_forces=True)
        if vacuum.forces_eV_per_A is None:  # guarded by validator
            raise RuntimeError("model omitted requested vacuum forces.")
        intrinsic = -np.asarray(vacuum.forces_eV_per_A, dtype=float)
        source_gradient = continuum.total_source_gradient_array()
        source_response = np.asarray(
            self.model.source_position_vjp(geometry, zero_field, source_gradient),
            dtype=float,
        )
        if source_response.shape != (count, 3) or not np.all(
            np.isfinite(source_response)
        ):
            raise ValueError(
                "model source-position VJP must be finite with shape (N,3)."
            )
        fixed_source = np.asarray(
            continuum.coordinate_gradient_array(), dtype=float
        ).reshape(count, 3)
        total = intrinsic + fixed_source + source_response
        energy = GeometryMediatedEnergyEvaluation(
            scalar_id=self.scalar_id,
            profile_id=self.profile_id,
            vacuum_energy_eV=vacuum.energy_eV,
            continuum_energy_eV=continuum.energy,
            total_energy_eV=vacuum.energy_eV + continuum.energy,
        )
        return GeometryMediatedEvaluation(
            energy=energy,
            source=source,
            reaction_field=reaction_field,
            intrinsic_gradient_eV_per_A=intrinsic,
            continuum_fixed_source_gradient_eV_per_A=fixed_source,
            source_response_gradient_eV_per_A=source_response,
            total_gradient_eV_per_A=total,
            reciprocity_gradient_error_eV_per_source_unit=reciprocity_error,
            reciprocity_audit=audit,
        )


__all__ = [
    "BilinearReciprocityRecord",
    "ChargeDirectionalFDRecord",
    "GeometryMediatedElectrostaticScalar",
    "GeometryMediatedEnergyEvaluation",
    "GeometryMediatedEvaluation",
    "GeometryMediatedReciprocityAudit",
]
