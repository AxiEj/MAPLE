"""Fail-closed audits for checkpoint-native external-field energy semantics.

The operational fixed point determines a state, not an energy.  Before a
field-conditioned checkpoint scalar can be used by a registered ledger, this
module binds the exact field chart and separately audits zero-field parity,
upstream replay, directional derivatives, work sign, and a charging identity.
Passing these numerical checks does not establish source-energy conjugacy or
admit any public capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable

import numpy as np

FIELD_SEMANTICS_CONTRACT_VERSION = "route2-field-semantics-manifest-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ZERO_ENERGY_ATOL_EV = 2.0e-10
_ZERO_FORCE_ATOL_EV_PER_A = 2.0e-9
_REPLAY_SOURCE_ATOL = 2.0e-10
_REPLAY_ENERGY_ATOL_EV = 2.0e-10
_REPLAY_FORCE_ATOL_EV_PER_A = 2.0e-9
_REPLAY_DIPOLE_ATOL_E_A = 2.0e-10
_REPLAY_DERIVATIVE_ATOL_E_A = 2.0e-9
_DERIVATIVE_ATOL_EV = 2.0e-9
_DERIVATIVE_RTOL = 2.0e-7
_CHARGING_ATOL_EV = 2.0e-9
_CHARGING_RTOL = 2.0e-7
_WORK_SIGN_ATOL = 2.0e-9
_WORK_SIGN_RTOL = 2.0e-7


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return result


def _string_tuple(values: object, *, name: str, unique: bool = True) -> tuple[str, ...]:
    try:
        result = tuple(_text(value, name=name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings.") from exc
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique.")
    return result


def _finite_scalar(value: object, *, name: str, nonnegative: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _finite_array(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.size < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite array.")
    return np.array(result, copy=True)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_unverified_mace_polar_field_semantics_manifest(
    *, native_canary: object, adapter: object
) -> "FieldSemanticsManifest":
    """Bind verified runtime layout while leaving energy semantics unresolved.

    This constructor intentionally cannot set signs, explicit-work flags, the
    spin-channel factor, or evidence IDs.  Those facts must come from the five
    executable canaries and a separately retained artifact.
    """

    from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING

    configuration = getattr(adapter, "configuration_sha256", None)
    source_space = getattr(adapter, "source_space", None)
    receiver_space = getattr(adapter, "receiver_space", None)
    if (
        not callable(configuration)
        or not callable(getattr(source_space, "metadata_hash", None))
        or not callable(getattr(receiver_space, "metadata_hash", None))
    ):
        raise TypeError(
            "adapter must expose configuration_sha256 and source/receiver spaces."
        )
    checkpoint = _digest(
        getattr(native_canary, "checkpoint_sha256", None),
        name="native_canary.checkpoint_sha256",
    )
    adapter_checkpoint = _digest(
        getattr(adapter, "checkpoint_sha256", None),
        name="adapter.checkpoint_sha256",
    )
    if checkpoint != adapter_checkpoint:
        raise ValueError("native canary and adapter checkpoints do not match.")
    if getattr(native_canary, "model_profile_id", None) != getattr(
        adapter, "model_profile_id", None
    ):
        raise ValueError("native canary and adapter model profiles do not match.")
    pairing_sha256 = _digest(
        getattr(adapter, "field_energy_pairing_sha256", None),
        name="adapter.field_energy_pairing_sha256",
    )
    if pairing_sha256 != MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash():
        raise ValueError(
            "adapter field-energy pairing is not the canonical radial metric."
        )
    receiver_widths = tuple(
        float(value) for value in getattr(native_canary, "receiver_sigmas_angstrom", ())
    )
    return FieldSemanticsManifest(
        checkpoint_sha256=checkpoint,
        adapter_configuration_sha256=_digest(
            configuration(), name="adapter.configuration_sha256"
        ),
        adapter_provenance_sha256=_digest(
            getattr(adapter, "provenance_sha256", None),
            name="adapter.provenance_sha256",
        ),
        model_provider_id=_text(
            getattr(adapter, "provider_id", None), name="adapter.provider_id"
        ),
        model_profile_id=_text(
            getattr(adapter, "model_profile_id", None),
            name="adapter.model_profile_id",
        ),
        source_space_sha256=_digest(
            source_space.metadata_hash(), name="source_space.metadata_hash"
        ),
        native_field_space_sha256=_digest(
            receiver_space.metadata_hash(), name="receiver_space.metadata_hash"
        ),
        pairing_metric_sha256=pairing_sha256,
        field_channel_order=tuple(receiver_space.components),
        field_radial_widths_angstrom=receiver_widths,
        real_ylm_convention=(
            "raw real-l1 columns [m0,m1,m-1] map to Cartesian [y,z,x]"
        ),
        cartesian_spherical_l1_transform=(
            "[q,l1_m0,l1_m1,l1_mminus1] -> [q,x,y,z]=" "[q,l1_mminus1,l1_m0,l1_m1]"
        ),
        field_units=tuple(receiver_space.units),
        energy_unit=MACE_POLAR_RADIAL_GTO_PAIRING.energy_unit,
        external_potential_sign=None,
        uniform_field_sign=None,
        spin_channel_factor=None,
        native_injection_explicit_work_included=None,
        upstream_uniform_explicit_work_included=None,
        origin_convention="unverified",
        evidence_measurement_sha256s=(),
    )


@dataclass(frozen=True, slots=True)
class FieldSemanticsManifest:
    """Content-addressed interpretation of one checkpoint field-energy path."""

    checkpoint_sha256: str
    adapter_configuration_sha256: str
    adapter_provenance_sha256: str
    model_provider_id: str
    model_profile_id: str
    source_space_sha256: str
    native_field_space_sha256: str
    pairing_metric_sha256: str
    field_channel_order: tuple[str, ...]
    field_radial_widths_angstrom: tuple[float, ...]
    real_ylm_convention: str
    cartesian_spherical_l1_transform: str
    field_units: tuple[str, ...]
    energy_unit: str
    external_potential_sign: int | None
    uniform_field_sign: int | None
    spin_channel_factor: float | None
    native_injection_explicit_work_included: bool | None
    upstream_uniform_explicit_work_included: bool | None
    origin_convention: str
    evidence_measurement_sha256s: tuple[str, ...]
    contract_version: str = FIELD_SEMANTICS_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "adapter_configuration_sha256",
            "adapter_provenance_sha256",
            "source_space_sha256",
            "native_field_space_sha256",
            "pairing_metric_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        for name in (
            "model_provider_id",
            "model_profile_id",
            "real_ylm_convention",
            "cartesian_spherical_l1_transform",
            "energy_unit",
            "origin_convention",
            "contract_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "field_channel_order",
            _string_tuple(self.field_channel_order, name="field_channel_order"),
        )
        object.__setattr__(
            self,
            "field_units",
            _string_tuple(self.field_units, name="field_units", unique=False),
        )
        if len(self.field_channel_order) != len(self.field_units):
            raise ValueError("field channel order and units must have equal length.")
        try:
            widths = tuple(float(value) for value in self.field_radial_widths_angstrom)
        except (TypeError, ValueError) as exc:
            raise TypeError("field_radial_widths_angstrom must be numeric.") from exc
        if not widths or any(
            not np.isfinite(value) or value <= 0.0 for value in widths
        ):
            raise ValueError("field radial widths must be positive and finite.")
        object.__setattr__(self, "field_radial_widths_angstrom", widths)
        for name in ("external_potential_sign", "uniform_field_sign"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value not in (-1, 1)):
                raise ValueError(f"{name} must be sign -1, sign +1, or None.")
        factor = self.spin_channel_factor
        if factor is not None:
            if (
                isinstance(factor, bool)
                or not np.isfinite(float(factor))
                or factor <= 0
            ):
                raise ValueError(
                    "spin_channel_factor must be positive, finite, or None."
                )
            object.__setattr__(self, "spin_channel_factor", float(factor))
        for name in (
            "native_injection_explicit_work_included",
            "upstream_uniform_explicit_work_included",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be exactly bool or None.")
        evidence = tuple(
            _digest(value, name="evidence_measurement_sha256s")
            for value in self.evidence_measurement_sha256s
        )
        if len(set(evidence)) != len(evidence):
            raise ValueError("evidence_measurement_sha256s must be unique.")
        object.__setattr__(self, "evidence_measurement_sha256s", evidence)

    @property
    def phi1_semantics_complete(self) -> bool:
        """Whether complete external-enthalpy semantics have been evidenced.

        This property deliberately requires explicit work to be present in the
        exact native-injection branch consumed by Phi1.  A verified upstream
        branch with a different energy graph is not sufficient.
        """

        return (
            self.external_potential_sign in (-1, 1)
            and self.uniform_field_sign in (-1, 1)
            and self.spin_channel_factor is not None
            and self.native_injection_explicit_work_included is True
            and self.upstream_uniform_explicit_work_included is True
            and self.origin_convention.lower() != "unverified"
            and bool(self.evidence_measurement_sha256s)
        )

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "adapter_configuration_sha256": self.adapter_configuration_sha256,
            "adapter_provenance_sha256": self.adapter_provenance_sha256,
            "model_provider_id": self.model_provider_id,
            "model_profile_id": self.model_profile_id,
            "source_space_sha256": self.source_space_sha256,
            "native_field_space_sha256": self.native_field_space_sha256,
            "pairing_metric_sha256": self.pairing_metric_sha256,
            "field_channel_order": list(self.field_channel_order),
            "field_radial_widths_angstrom": list(self.field_radial_widths_angstrom),
            "real_ylm_convention": self.real_ylm_convention,
            "cartesian_spherical_l1_transform": self.cartesian_spherical_l1_transform,
            "field_units": list(self.field_units),
            "energy_unit": self.energy_unit,
            "external_potential_sign": self.external_potential_sign,
            "uniform_field_sign": self.uniform_field_sign,
            "spin_channel_factor": self.spin_channel_factor,
            "native_injection_explicit_work_included": (
                self.native_injection_explicit_work_included
            ),
            "upstream_uniform_explicit_work_included": (
                self.upstream_uniform_explicit_work_included
            ),
            "origin_convention": self.origin_convention,
            "evidence_measurement_sha256s": list(self.evidence_measurement_sha256s),
        }

    def configuration_sha256(self) -> str:
        return _sha(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "configuration_sha256": self.configuration_sha256(),
            "phi1_semantics_complete": self.phi1_semantics_complete,
            "capability_admitted": False,
            "claim_boundary": (
                "A complete manifest interprets one energy graph only. It does not "
                "prove original-source conjugacy, physical PCM source accuracy, or "
                "any public Route-2 capability."
            ),
        }


@dataclass(frozen=True, slots=True)
class ZeroFieldBaselineCanary:
    energy_absolute_error_eV: float
    force_max_absolute_error_eV_per_A: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "energy_absolute_error_eV": self.energy_absolute_error_eV,
            "force_max_absolute_error_eV_per_A": (
                self.force_max_absolute_error_eV_per_A
            ),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class UniformFieldReplayCanary:
    source_max_abs_error: float
    energy_absolute_error_eV: float
    force_max_abs_error_eV_per_A: float
    dipole_max_abs_error_e_angstrom: float
    energy_derivative_absolute_error_e_angstrom: float
    source_path_matched: bool
    passed: bool
    branch_semantics: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_max_abs_error": self.source_max_abs_error,
            "energy_absolute_error_eV": self.energy_absolute_error_eV,
            "force_max_abs_error_eV_per_A": self.force_max_abs_error_eV_per_A,
            "dipole_max_abs_error_e_angstrom": self.dipole_max_abs_error_e_angstrom,
            "energy_derivative_absolute_error_e_angstrom": (
                self.energy_derivative_absolute_error_e_angstrom
            ),
            "source_path_matched": self.source_path_matched,
            "passed": self.passed,
            "branch_semantics": self.branch_semantics,
        }


@dataclass(frozen=True, slots=True)
class DirectionalDerivativeCanary:
    analytic_directional_derivative_eV: float
    finite_difference_values_eV: tuple[float, ...]
    steps: tuple[float, ...]
    minimum_absolute_error_eV: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "analytic_directional_derivative_eV": (
                self.analytic_directional_derivative_eV
            ),
            "finite_difference_values_eV": list(self.finite_difference_values_eV),
            "steps": list(self.steps),
            "minimum_absolute_error_eV": self.minimum_absolute_error_eV,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class FieldWorkSignCanary:
    sign: int
    residual_l2: float
    relative_residual: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "sign": self.sign,
            "residual_l2": self.residual_l2,
            "relative_residual": self.relative_residual,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ChargingPathCanary:
    endpoint_energy_difference_eV: float
    integrated_work_eV: float
    refined_integrated_work_eV: float
    endpoint_identity_absolute_error_eV: float
    quadrature_refinement_absolute_error_eV: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint_energy_difference_eV": self.endpoint_energy_difference_eV,
            "integrated_work_eV": self.integrated_work_eV,
            "refined_integrated_work_eV": self.refined_integrated_work_eV,
            "endpoint_identity_absolute_error_eV": (
                self.endpoint_identity_absolute_error_eV
            ),
            "quadrature_refinement_absolute_error_eV": (
                self.quadrature_refinement_absolute_error_eV
            ),
            "passed": self.passed,
        }


def audit_zero_field_baseline(
    *,
    conditioned_zero_energy_eV: float,
    vacuum_energy_eV: float,
    conditioned_zero_forces_eV_per_A: object,
    vacuum_forces_eV_per_A: object,
) -> ZeroFieldBaselineCanary:
    conditioned_energy = _finite_scalar(
        conditioned_zero_energy_eV, name="conditioned_zero_energy_eV"
    )
    vacuum_energy = _finite_scalar(vacuum_energy_eV, name="vacuum_energy_eV")
    conditioned_forces = _finite_array(
        conditioned_zero_forces_eV_per_A,
        name="conditioned_zero_forces_eV_per_A",
    )
    vacuum_forces = _finite_array(vacuum_forces_eV_per_A, name="vacuum_forces_eV_per_A")
    if conditioned_forces.shape != vacuum_forces.shape:
        raise ValueError("zero-field and vacuum forces must have identical shapes.")
    energy_error = abs(conditioned_energy - vacuum_energy)
    force_error = float(np.max(np.abs(conditioned_forces - vacuum_forces)))
    return ZeroFieldBaselineCanary(
        energy_absolute_error_eV=energy_error,
        force_max_absolute_error_eV_per_A=force_error,
        passed=(
            energy_error <= _ZERO_ENERGY_ATOL_EV
            and force_error <= _ZERO_FORCE_ATOL_EV_PER_A
        ),
    )


def audit_uniform_field_replay(
    *,
    source_max_abs_error: float,
    energy_absolute_error_eV: float,
    force_max_abs_error_eV_per_A: float,
    dipole_max_abs_error_e_angstrom: float,
    energy_derivative_absolute_error_e_angstrom: float,
) -> UniformFieldReplayCanary:
    values = {
        name: _finite_scalar(value, name=name, nonnegative=True)
        for name, value in {
            "source_max_abs_error": source_max_abs_error,
            "energy_absolute_error_eV": energy_absolute_error_eV,
            "force_max_abs_error_eV_per_A": force_max_abs_error_eV_per_A,
            "dipole_max_abs_error_e_angstrom": dipole_max_abs_error_e_angstrom,
            "energy_derivative_absolute_error_e_angstrom": (
                energy_derivative_absolute_error_e_angstrom
            ),
        }.items()
    }
    source_passed = values["source_max_abs_error"] <= _REPLAY_SOURCE_ATOL
    passed = (
        source_passed
        and values["energy_absolute_error_eV"] <= _REPLAY_ENERGY_ATOL_EV
        and values["force_max_abs_error_eV_per_A"] <= _REPLAY_FORCE_ATOL_EV_PER_A
        and values["dipole_max_abs_error_e_angstrom"] <= _REPLAY_DIPOLE_ATOL_E_A
        and values["energy_derivative_absolute_error_e_angstrom"]
        <= _REPLAY_DERIVATIVE_ATOL_E_A
    )
    if passed:
        semantics = "same-upstream-and-native-energy-path"
    elif source_passed:
        semantics = "hybrid-branch-dependent"
    else:
        semantics = "field-adapter-mismatch"
    return UniformFieldReplayCanary(
        source_path_matched=source_passed,
        passed=passed,
        branch_semantics=semantics,
        **values,
    )


def audit_directional_derivative(
    *,
    energy: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], object],
    field: object,
    direction: object,
    steps: tuple[float, ...] = (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4),
) -> DirectionalDerivativeCanary:
    values = _finite_array(field, name="field")
    tangent = _finite_array(direction, name="direction")
    if values.shape != tangent.shape or float(np.linalg.norm(tangent)) == 0.0:
        raise ValueError("field and nonzero direction must have identical shapes.")
    step_values = tuple(float(step) for step in steps)
    if not step_values or any(
        not np.isfinite(step) or step <= 0 for step in step_values
    ):
        raise ValueError("steps must be positive and finite.")
    gradient_values = _finite_array(gradient(values), name="gradient(field)")
    if gradient_values.shape != values.shape:
        raise ValueError("gradient(field) must have the field shape.")
    analytic = float(np.vdot(gradient_values, tangent))
    finite_differences = []
    for step in step_values:
        plus = _finite_scalar(energy(values + step * tangent), name="energy(plus)")
        minus = _finite_scalar(energy(values - step * tangent), name="energy(minus)")
        finite_differences.append((plus - minus) / (2.0 * step))
    errors = np.abs(np.asarray(finite_differences) - analytic)
    minimum = float(np.min(errors))
    threshold = _DERIVATIVE_ATOL_EV + _DERIVATIVE_RTOL * abs(analytic)
    return DirectionalDerivativeCanary(
        analytic_directional_derivative_eV=analytic,
        finite_difference_values_eV=tuple(float(value) for value in finite_differences),
        steps=step_values,
        minimum_absolute_error_eV=minimum,
        passed=minimum <= threshold,
    )


def audit_field_work_sign(
    *,
    energy_gradient: object,
    reference_source: object,
    sign: int,
    pairing_matrix: object | None = None,
) -> FieldWorkSignCanary:
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1.")
    gradient_values = _finite_array(energy_gradient, name="energy_gradient")
    source_values = _finite_array(reference_source, name="reference_source")
    if gradient_values.shape != source_values.shape:
        raise ValueError("energy gradient and reference source must have equal shape.")
    flattened_source = source_values.reshape(-1)
    if pairing_matrix is None:
        dual_source = flattened_source
    else:
        metric = np.asarray(pairing_matrix, dtype=float)
        expected = flattened_source.size
        if metric.shape != (expected, expected) or not np.all(np.isfinite(metric)):
            raise ValueError("pairing_matrix must be finite and square on the source.")
        dual_source = metric.T @ flattened_source
    residual = gradient_values.reshape(-1) - sign * dual_source
    residual_norm = float(np.linalg.norm(residual))
    scale = max(
        float(np.linalg.norm(gradient_values)),
        float(np.linalg.norm(dual_source)),
        np.finfo(float).tiny,
    )
    relative = residual_norm / scale
    return FieldWorkSignCanary(
        sign=sign,
        residual_l2=residual_norm,
        relative_residual=relative,
        passed=(residual_norm <= _WORK_SIGN_ATOL + _WORK_SIGN_RTOL * scale),
    )


def _integrated_charging_work(
    gradient: Callable[[np.ndarray], object], endpoint: np.ndarray, *, order: int
) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    parameters = 0.5 * (nodes + 1.0)
    total = 0.0
    for parameter, weight in zip(parameters, weights, strict=True):
        value = _finite_array(gradient(parameter * endpoint), name="charging gradient")
        if value.shape != endpoint.shape:
            raise ValueError("charging gradient must have the field shape.")
        total += float(weight) * float(np.vdot(value, endpoint))
    return 0.5 * total


def audit_charging_path(
    *,
    energy: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], object],
    endpoint_field: object,
) -> ChargingPathCanary:
    endpoint = _finite_array(endpoint_field, name="endpoint_field")
    if float(np.linalg.norm(endpoint)) == 0.0:
        raise ValueError("endpoint_field must be nonzero.")
    zero = np.zeros_like(endpoint)
    energy_difference = _finite_scalar(
        energy(endpoint), name="endpoint energy"
    ) - _finite_scalar(energy(zero), name="zero-field energy")
    integrated = _integrated_charging_work(gradient, endpoint, order=16)
    refined = _integrated_charging_work(gradient, endpoint, order=32)
    endpoint_error = abs(energy_difference - refined)
    refinement_error = abs(integrated - refined)
    scale = max(
        abs(energy_difference),
        abs(refined),
        np.finfo(float).tiny,
    )
    threshold = _CHARGING_ATOL_EV + _CHARGING_RTOL * scale
    return ChargingPathCanary(
        endpoint_energy_difference_eV=energy_difference,
        integrated_work_eV=integrated,
        refined_integrated_work_eV=refined,
        endpoint_identity_absolute_error_eV=endpoint_error,
        quadrature_refinement_absolute_error_eV=refinement_error,
        passed=endpoint_error <= threshold and refinement_error <= threshold,
    )


__all__ = [
    "ChargingPathCanary",
    "DirectionalDerivativeCanary",
    "FIELD_SEMANTICS_CONTRACT_VERSION",
    "FieldSemanticsManifest",
    "FieldWorkSignCanary",
    "UniformFieldReplayCanary",
    "ZeroFieldBaselineCanary",
    "audit_charging_path",
    "audit_directional_derivative",
    "audit_field_work_sign",
    "audit_uniform_field_replay",
    "audit_zero_field_baseline",
    "build_unverified_mace_polar_field_semantics_manifest",
]
