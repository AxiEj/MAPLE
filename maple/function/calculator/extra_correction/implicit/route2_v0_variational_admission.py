"""Fail-closed physical admission for no-training Route-2 V0 response.

The V0-Q KKT solver proves that a supplied curvature can produce a common
energy with a reciprocal PCM operator.  This module separately checks whether
that curvature is physical enough to enter the implicit-continuum ledger.  It
never authorizes a total-solvation or experimental-accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .route2_v0_variational_quadratic import Route2V0VariationalQuadraticState

V0_VARIATIONAL_ELECTRONIC_ADMISSION_CONSTRUCTION = (
    "route2-v0-variational-electronic-admission-v1"
)

# Frozen from route2-v0-qeq-acetone-qm-field-prereg-v1.json.  A different
# scientific screen needs a new construction rather than runtime relaxation.
GAS_RESPONSE_RELATIVE_FROBENIUS_MAX = 0.20
GAS_RESPONSE_TRACE_RATIO_MIN = 0.80
GAS_RESPONSE_TRACE_RATIO_MAX = 1.20
GAS_RESPONSE_PRINCIPAL_VALUE_RELATIVE_MAX = 0.30

_SOURCE_KINDS = frozenset(
    {
        "frozen-external-model",
        "independent-physical-theory",
        "synthetic-structural-control",
    }
)
_REFERENCE_STATIONARITY_RELATIVE_MAX = 1.0e-10
_STRUCTURAL_RESIDUAL_RELATIVE_MAX = 1.0e-9
_RESPONSE_RELATIVE_MAX = 1.0e-10
_CONSTRAINT_ABSOLUTE_MAX = 1.0e-10

FunctionalSourceKind = Literal[
    "frozen-external-model",
    "independent-physical-theory",
    "synthetic-structural-control",
]


@dataclass(frozen=True)
class FrozenNoTrainingFunctionalProvenance:
    """Provenance that must be frozen before a gas-response screen is read."""

    candidate_id: str
    source_record: str
    coefficient_dual_pairing: str
    source_kind: FunctionalSourceKind
    scalar_energy_functional_declared: bool
    post_training: bool = False
    fine_tuning: bool = False
    experimental_solvation_fit: bool = False
    error_selected_parameters: bool = False

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.candidate_id,
                self.source_record,
                self.coefficient_dual_pairing,
            )
        ):
            raise ValueError("Functional provenance requires nonempty identifiers.")
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError("Unsupported frozen functional source kind.")
        if not self.scalar_energy_functional_declared:
            raise ValueError(
                "V0 admission requires a declared scalar energy functional."
            )
        if any(
            (
                self.post_training,
                self.fine_tuning,
                self.experimental_solvation_fit,
                self.error_selected_parameters,
            )
        ):
            raise ValueError(
                "V0 electronic admission forbids training, solvation fitting, and "
                "error-selected parameters."
            )

    @property
    def is_synthetic_control(self) -> bool:
        return self.source_kind == "synthetic-structural-control"


@dataclass(frozen=True)
class GasPhasePolarizabilityScreen:
    """Matched-geometry, preregistered gas QM finite-field comparison."""

    candidate_id: str
    candidate_polarizability_bohr3: np.ndarray
    qm_reference_polarizability_bohr3: np.ndarray
    candidate_geometry_id: str
    qm_reference_geometry_id: str
    candidate_charge_e: int
    qm_reference_charge_e: int
    candidate_spin_multiplicity: int
    qm_reference_spin_multiplicity: int
    qm_protocol_id: str
    preregistered_before_execution: bool
    qm_numerical_gates_passed: bool


@dataclass(frozen=True)
class GasPhasePolarizabilityDecision:
    relative_frobenius_mismatch: float
    trace_ratio: float
    principal_value_relative_max: float
    passes: bool


@dataclass(frozen=True)
class Route2V0ElectronicAdmission:
    """Component decision; total-solvation scoring is deliberately always false."""

    provenance: FrozenNoTrainingFunctionalProvenance
    reference_neutral_stationarity_residual: float
    structural_checks_passed: bool
    gas_response_decision: GasPhasePolarizabilityDecision | None
    status: Literal[
        "rejected-structural",
        "structural-only",
        "rejected-gas-response",
        "electronic-component-admitted",
    ]
    eligible_for_implicit_electrostatic_composition: bool
    eligible_for_total_solvation_scoring: bool
    reasons: tuple[str, ...]
    construction: str = V0_VARIATIONAL_ELECTRONIC_ADMISSION_CONSTRUCTION


def _finite_array(
    values: np.ndarray, *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return result


def _nullspace(constraints: np.ndarray) -> np.ndarray:
    matrix = np.asarray(constraints, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Induced constraints must be a nonempty matrix.")
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    threshold = 1.0e-12 * max(1.0, float(np.linalg.norm(matrix, ord=2)))
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank != matrix.shape[0] or rank >= matrix.shape[1]:
        raise ValueError(
            "Induced constraints must leave an independent response space."
        )
    return right_vectors[rank:, :].T


def _symmetric_positive_semidefinite(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    matrix = _finite_array(values, name=name, shape=(3, 3))
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    if float(np.linalg.norm(matrix - matrix.T, ord=2)) > _RESPONSE_RELATIVE_MAX * scale:
        raise ValueError(f"{name} must be symmetric.")
    result = 0.5 * (matrix + matrix.T)
    if float(np.min(np.linalg.eigvalsh(result))) < -_RESPONSE_RELATIVE_MAX * scale:
        raise ValueError(f"{name} must be positive semidefinite.")
    return result


def evaluate_gas_phase_polarizability_screen(
    screen: GasPhasePolarizabilityScreen,
) -> GasPhasePolarizabilityDecision:
    """Apply the immutable V0-Q finite-field thresholds to one candidate."""

    text_fields = (
        screen.candidate_id,
        screen.candidate_geometry_id,
        screen.qm_reference_geometry_id,
        screen.qm_protocol_id,
    )
    if not all(isinstance(value, str) and value.strip() for value in text_fields):
        raise ValueError("Gas-response screen requires nonempty identifiers.")
    if (
        screen.candidate_geometry_id != screen.qm_reference_geometry_id
        or int(screen.candidate_charge_e) != int(screen.qm_reference_charge_e)
        or int(screen.candidate_spin_multiplicity)
        != int(screen.qm_reference_spin_multiplicity)
        or int(screen.candidate_spin_multiplicity) < 1
    ):
        raise ValueError("Gas-response screen must match geometry, charge, and spin.")
    candidate = _symmetric_positive_semidefinite(
        screen.candidate_polarizability_bohr3,
        name="Candidate gas polarizability",
    )
    reference = _symmetric_positive_semidefinite(
        screen.qm_reference_polarizability_bohr3,
        name="QM reference gas polarizability",
    )
    reference_norm = float(np.linalg.norm(reference, ord="fro"))
    reference_trace = float(np.trace(reference))
    if reference_norm <= 0.0 or reference_trace <= 0.0:
        raise ValueError(
            "QM reference gas polarizability must be nonzero and positive."
        )
    mismatch = float(np.linalg.norm(candidate - reference, ord="fro") / reference_norm)
    trace_ratio = float(np.trace(candidate) / reference_trace)
    candidate_eigenvalues = np.linalg.eigvalsh(candidate)
    reference_eigenvalues = np.linalg.eigvalsh(reference)
    principal_error = float(
        np.max(
            np.abs(candidate_eigenvalues - reference_eigenvalues)
            / np.maximum(np.abs(reference_eigenvalues), 1.0e-12)
        )
    )
    passed = bool(
        screen.preregistered_before_execution
        and screen.qm_numerical_gates_passed
        and mismatch <= GAS_RESPONSE_RELATIVE_FROBENIUS_MAX
        and GAS_RESPONSE_TRACE_RATIO_MIN <= trace_ratio <= GAS_RESPONSE_TRACE_RATIO_MAX
        and principal_error <= GAS_RESPONSE_PRINCIPAL_VALUE_RELATIVE_MAX
    )
    return GasPhasePolarizabilityDecision(
        relative_frobenius_mismatch=mismatch,
        trace_ratio=trace_ratio,
        principal_value_relative_max=principal_error,
        passes=passed,
    )


def _structural_reasons(
    state: Route2V0VariationalQuadraticState,
    gradient: np.ndarray,
) -> tuple[float, tuple[str, ...]]:
    electronic = np.asarray(state.electronic_curvature_coefficient_dual, dtype=float)
    joint = np.asarray(state.joint_hessian_coefficient_dual, dtype=float)
    constraints = np.asarray(state.induced_constraint_matrix, dtype=float)
    response = np.asarray(state.joint_external_dual_response, dtype=float)
    coefficient_count = int(electronic.shape[0])
    reference_gradient = _finite_array(
        gradient,
        name="Gas reference gradient",
        shape=(coefficient_count,),
    )
    allowed = _nullspace(constraints)
    residual = float(np.linalg.norm(allowed.T @ reference_gradient, ord=np.inf))
    gradient_scale = max(1.0, float(np.linalg.norm(reference_gradient, ord=np.inf)))
    reasons: list[str] = []
    if residual > _REFERENCE_STATIONARITY_RELATIVE_MAX * gradient_scale:
        reasons.append("gas reference state is not stationary in the allowed subspace")
    for name, matrix in (("electronic", electronic), ("joint", joint)):
        scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
        if (
            float(np.linalg.norm(matrix - matrix.T, ord=2))
            > _RESPONSE_RELATIVE_MAX * scale
        ):
            reasons.append(f"{name} curvature is not symmetric in the declared pairing")
        if float(np.min(np.linalg.eigvalsh(allowed.T @ matrix @ allowed))) <= (
            state.curvature_stability_threshold
        ):
            reasons.append(f"{name} curvature is not positive in the allowed subspace")
    if state.stationarity_residual_inf > _STRUCTURAL_RESIDUAL_RELATIVE_MAX * max(
        1.0, float(np.linalg.norm(joint, ord=2))
    ):
        reasons.append("joint KKT stationarity residual exceeds admission tolerance")
    if state.charge_constraint_residual_e > _CONSTRAINT_ABSOLUTE_MAX:
        reasons.append("joint KKT state violates total-charge conservation")
    if state.additional_constraint_residual_inf > _CONSTRAINT_ABSOLUTE_MAX:
        reasons.append("joint KKT state violates an induced response constraint")
    response_scale = max(1.0, float(np.linalg.norm(response, ord=2)))
    if (
        float(np.linalg.norm(response - response.T, ord=2))
        > _RESPONSE_RELATIVE_MAX * response_scale
    ):
        reasons.append("joint external response is not reciprocal")
    if (
        float(np.max(np.linalg.eigvalsh(response)))
        > _RESPONSE_RELATIVE_MAX * response_scale
    ):
        reasons.append("joint external response is not passive")
    if (
        float(np.linalg.norm(constraints @ response, ord=np.inf))
        > _CONSTRAINT_ABSOLUTE_MAX
    ):
        reasons.append("joint response violates its exact constraints")
    return residual, tuple(reasons)


def admit_route2_v0_electronic_component(
    *,
    state: Route2V0VariationalQuadraticState,
    provenance: FrozenNoTrainingFunctionalProvenance,
    reference_gradient_coefficient_dual_hartree: np.ndarray,
    gas_response_screen: GasPhasePolarizabilityScreen | None = None,
) -> Route2V0ElectronicAdmission:
    """Admit only a screened physical component; never a solvation score."""

    if (
        gas_response_screen is not None
        and gas_response_screen.candidate_id != provenance.candidate_id
    ):
        raise ValueError("Gas response screen and electronic provenance disagree.")
    residual, structural_reasons = _structural_reasons(
        state, reference_gradient_coefficient_dual_hartree
    )
    if structural_reasons:
        return Route2V0ElectronicAdmission(
            provenance,
            residual,
            False,
            None,
            "rejected-structural",
            False,
            False,
            structural_reasons,
        )
    if provenance.is_synthetic_control:
        return Route2V0ElectronicAdmission(
            provenance,
            residual,
            True,
            None,
            "structural-only",
            False,
            False,
            ("synthetic structural controls are permanently excluded from chemistry",),
        )
    if gas_response_screen is None:
        return Route2V0ElectronicAdmission(
            provenance,
            residual,
            True,
            None,
            "structural-only",
            False,
            False,
            ("a preregistered, numerically valid QM gas-response screen is required",),
        )
    decision = evaluate_gas_phase_polarizability_screen(gas_response_screen)
    if not decision.passes:
        reasons: list[str] = []
        if not gas_response_screen.preregistered_before_execution:
            reasons.append("gas-response comparison was not preregistered")
        if not gas_response_screen.qm_numerical_gates_passed:
            reasons.append("QM gas-response numerical gates did not pass")
        if decision.relative_frobenius_mismatch > GAS_RESPONSE_RELATIVE_FROBENIUS_MAX:
            reasons.append(
                "gas polarizability Frobenius mismatch exceeds the fixed limit"
            )
        if (
            not GAS_RESPONSE_TRACE_RATIO_MIN
            <= decision.trace_ratio
            <= GAS_RESPONSE_TRACE_RATIO_MAX
        ):
            reasons.append(
                "gas polarizability trace ratio lies outside the fixed interval"
            )
        if (
            decision.principal_value_relative_max
            > GAS_RESPONSE_PRINCIPAL_VALUE_RELATIVE_MAX
        ):
            reasons.append(
                "gas polarizability principal-value error exceeds the fixed limit"
            )
        return Route2V0ElectronicAdmission(
            provenance,
            residual,
            True,
            decision,
            "rejected-gas-response",
            False,
            False,
            tuple(reasons),
        )
    return Route2V0ElectronicAdmission(
        provenance,
        residual,
        True,
        decision,
        "electronic-component-admitted",
        True,
        False,
        (
            (
                "electronic component passed fixed gas-response gates; "
                "total-solvation and accuracy gates remain separately required"
            ),
        ),
    )


def require_route2_v0_implicit_electronic_admission(
    admission: Route2V0ElectronicAdmission,
) -> None:
    """Raise unless a component may enter the implicit-electrostatic ledger."""

    if not admission.eligible_for_implicit_electrostatic_composition:
        raise RuntimeError(
            "Electronic component is not admitted for implicit-continuum composition: "
            + "; ".join(admission.reasons)
        )


__all__ = [
    "GAS_RESPONSE_PRINCIPAL_VALUE_RELATIVE_MAX",
    "GAS_RESPONSE_RELATIVE_FROBENIUS_MAX",
    "GAS_RESPONSE_TRACE_RATIO_MAX",
    "GAS_RESPONSE_TRACE_RATIO_MIN",
    "V0_VARIATIONAL_ELECTRONIC_ADMISSION_CONSTRUCTION",
    "FrozenNoTrainingFunctionalProvenance",
    "FunctionalSourceKind",
    "GasPhasePolarizabilityDecision",
    "GasPhasePolarizabilityScreen",
    "Route2V0ElectronicAdmission",
    "admit_route2_v0_electronic_component",
    "evaluate_gas_phase_polarizability_screen",
    "require_route2_v0_implicit_electronic_admission",
]
