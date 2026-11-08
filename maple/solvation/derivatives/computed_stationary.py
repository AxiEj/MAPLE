"""Numerical stationary-point assessment of one stored Hessian operator.

This module verifies the eigendecomposition of the evaluated, symmetrized,
mass-weighted internal Hessian.  Its resolution radius is an FP64 eigensolver
property.  It is not an error bound for the analytic Hessian or the physical
potential-energy surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256


def _immutable_array(
    value: object, *, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array shape {array.shape} does not match required {shape}.")
    if not np.isfinite(array).all():
        raise ValueError("computed stationary-point inputs must be finite.")
    copy = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64)
    copy = copy.reshape(array.shape)
    copy.flags.writeable = False
    return copy


def _scaled_frobenius_norm(value: np.ndarray) -> float:
    """Return the mathematical Frobenius norm without avoidable overflow."""

    array = np.abs(np.asarray(value, dtype=np.float64)).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("Frobenius-norm input must be finite.")
    if array.size == 0:
        return 0.0
    scale = float(np.max(array))
    if scale == 0.0:
        return 0.0
    result = scale * float(np.sqrt(np.sum(np.square(array / scale))))
    if not np.isfinite(result):
        raise ValueError("Frobenius norm is not representable in FP64.")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    payload = repr(array.shape).encode() + b"\0" + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _sha256_identity(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 identity.")
    return text


def _nonempty_identity(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True, slots=True)
class ComputedStationaryPointAssessment:
    """Immutable QA and inertia of one evaluated internal Hessian operator."""

    internal_operator_eV_per_A2_amu: np.ndarray
    computed_eigenvalues_eV_per_A2_amu: np.ndarray
    computed_negative_count: int
    computed_positive_count: int
    computed_unresolved_count: int
    normalized_eigenpair_residual: float
    eigenvector_orthogonality_defect: float
    rigid_subspace_leakage: float
    computed_resolution_radius: float
    quality_gate: float
    computed_operator_quality_passed: bool
    computed_index_one: bool
    computed_minimum: bool
    smallest_absolute_eigenvalue: float
    geometry_sha256: str
    mass_sha256: str
    hessian_sha256: str
    model_input_sha256: str
    charge: int
    multiplicity: int
    evaluation_sha256: str
    configuration_sha256: str
    scalar_contract_id: str
    provider_id: str | None
    profile_id: str | None
    verification_evidence_id: str | None
    computation_dtype: str
    device: str
    independent_hessian_uncertainty: None = None
    interval_certified: None = None


@dataclass(frozen=True, slots=True)
class ComputedHessianAnalysis:
    """One identified evaluation, one eigensolve, and its derived claims."""

    context: object
    mode_analysis: object
    stationary_assessment: ComputedStationaryPointAssessment


@dataclass(frozen=True, slots=True)
class _OperatorAssessment:
    internal_operator: np.ndarray
    computed_eigenvalues: np.ndarray
    computed_negative_count: int
    computed_positive_count: int
    computed_unresolved_count: int
    normalized_eigenpair_residual: float
    eigenvector_orthogonality_defect: float
    rigid_subspace_leakage: float
    computed_resolution_radius: float
    quality_gate: float
    computed_operator_quality_passed: bool
    independent_hessian_uncertainty: None = None
    interval_certified: None = None


def _assess_internal_operator(
    operator: object,
    *,
    rigid_internal_overlap: object,
    eigensystem: tuple[object, object] | None = None,
    cartesian_dimension: int | None = None,
) -> _OperatorAssessment:
    """Assess one already-formed internal operator.

    ``eigensystem`` is an explicit verification hook. Production callers omit
    it and use NumPy's FP64 symmetric eigensolver.
    """

    matrix = _immutable_array(operator)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError("internal operator must be a non-empty square matrix.")
    dimension = matrix.shape[0]
    matrix = _immutable_array(0.5 * (matrix + matrix.T), shape=matrix.shape)
    overlap = _immutable_array(rigid_internal_overlap)
    if overlap.ndim != 2 or overlap.shape[1] != dimension:
        raise ValueError("rigid/internal overlap must have one column per mode.")
    if eigensystem is None:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    else:
        eigenvalues, eigenvectors = eigensystem
    eigenvalues = _immutable_array(eigenvalues, shape=(dimension,))
    eigenvectors = _immutable_array(eigenvectors, shape=(dimension, dimension))

    residual = matrix @ eigenvectors - eigenvectors * eigenvalues
    scale = _scaled_frobenius_norm(matrix)
    residual_norm = _scaled_frobenius_norm(residual)
    tiny64 = np.finfo(np.float64).tiny
    normalized_residual = residual_norm / max(scale, tiny64)
    orthogonality = _scaled_frobenius_norm(
        eigenvectors.T @ eigenvectors - np.eye(dimension)
    )
    rigid_leakage = _scaled_frobenius_norm(overlap @ eigenvectors)
    if cartesian_dimension is None:
        cartesian_dimension = dimension
    if type(cartesian_dimension) is not int or cartesian_dimension < dimension:
        raise ValueError("cartesian_dimension must contain the internal operator.")
    gate = 128.0 * max(cartesian_dimension, 1) * np.finfo(np.float64).eps
    radius = gate * scale + residual_norm
    if scale == 0.0:
        negative = positive = 0
        unresolved = dimension
    else:
        negative = int(np.count_nonzero(eigenvalues < -radius))
        positive = int(np.count_nonzero(eigenvalues > radius))
        unresolved = dimension - negative - positive
    quality = bool(
        normalized_residual <= gate and orthogonality <= gate and rigid_leakage <= gate
    )
    return _OperatorAssessment(
        internal_operator=matrix,
        computed_eigenvalues=eigenvalues,
        computed_negative_count=negative,
        computed_positive_count=positive,
        computed_unresolved_count=unresolved,
        normalized_eigenpair_residual=float(normalized_residual),
        eigenvector_orthogonality_defect=float(orthogonality),
        rigid_subspace_leakage=float(rigid_leakage),
        computed_resolution_radius=float(radius),
        quality_gate=float(gate),
        computed_operator_quality_passed=quality,
    )


def _assess_from_mode_analysis(
    atoms, context, analysis
) -> ComputedStationaryPointAssessment:

    from maple.function.calculator.route2 import is_identified_hessian_evaluation
    from maple.solvation.models.base import (
        model_charge_and_multiplicity,
        model_input_sha256,
    )

    if not is_identified_hessian_evaluation(context):
        raise TypeError(
            "computed assessment requires an identified analytic Hessian context."
        )
    evaluation = context.evaluation

    positions = _immutable_array(atoms.get_positions())
    masses = _immutable_array(atoms.get_masses(), shape=(len(atoms),))
    if positions.shape != (len(atoms), 3):
        raise ValueError("positions must have shape (N,3).")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive.")
    geometry_identity = geometry_sha256(atoms)
    if str(getattr(evaluation, "geometry_sha256", "")) != geometry_identity:
        raise ValueError("evaluation geometry identity does not match atoms.")
    current_model_input = model_input_sha256(atoms)
    current_charge, current_multiplicity = model_charge_and_multiplicity(atoms)
    if current_model_input != context.model_input_sha256:
        raise ValueError("Hessian model-input identity does not match atoms.")
    hessian = _immutable_array(
        evaluation.hessian_eV_per_A2, shape=(3 * len(atoms), 3 * len(atoms))
    )
    evaluation_identity = _sha256_identity(
        evaluation.evaluation_sha256, name="evaluation_sha256"
    )
    configuration_identity = _sha256_identity(
        evaluation.configuration_sha256, name="configuration_sha256"
    )
    scalar_identity = _nonempty_identity(
        evaluation.scalar_contract_id, name="scalar_contract_id"
    )
    computation_dtype = _nonempty_identity(
        evaluation.computation_dtype, name="computation_dtype"
    )
    device = _nonempty_identity(evaluation.device, name="device")
    if (
        context.configuration_sha256 != configuration_identity
        or context.profile_id != evaluation.profile_id
        or context.scalar_contract_id != scalar_identity
        or context.device != device
        or context.charge != current_charge
        or context.multiplicity != current_multiplicity
    ):
        raise ValueError("identified Hessian context does not match its evaluation.")
    internal_basis = analysis.internal_basis_mass_weighted
    rigid_basis = analysis.rigid_basis_mass_weighted
    if not internal_basis.shape[1]:
        raise ValueError(
            "computed stationary-point assessment requires internal modes."
        )
    inv_root_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    symmetric_hessian = 0.5 * (hessian + hessian.T)
    mass_weighted = inv_root_mass[:, None] * symmetric_hessian * inv_root_mass[None, :]
    operator = internal_basis.T @ mass_weighted @ internal_basis
    overlap = rigid_basis.T @ internal_basis
    modes_mass_weighted = (
        np.sqrt(np.repeat(masses, 3))[:, None] * analysis.modes_cartesian.T
    )
    eigenvectors_internal = internal_basis.T @ modes_mass_weighted
    qa = _assess_internal_operator(
        operator,
        rigid_internal_overlap=overlap,
        eigensystem=(
            analysis.eigenvalues_eV_per_A2_amu,
            eigenvectors_internal,
        ),
        cartesian_dimension=3 * len(atoms),
    )

    return ComputedStationaryPointAssessment(
        internal_operator_eV_per_A2_amu=qa.internal_operator,
        computed_eigenvalues_eV_per_A2_amu=qa.computed_eigenvalues,
        computed_negative_count=qa.computed_negative_count,
        computed_positive_count=qa.computed_positive_count,
        computed_unresolved_count=qa.computed_unresolved_count,
        normalized_eigenpair_residual=qa.normalized_eigenpair_residual,
        eigenvector_orthogonality_defect=qa.eigenvector_orthogonality_defect,
        rigid_subspace_leakage=qa.rigid_subspace_leakage,
        computed_resolution_radius=qa.computed_resolution_radius,
        quality_gate=qa.quality_gate,
        computed_operator_quality_passed=qa.computed_operator_quality_passed,
        computed_index_one=bool(
            qa.computed_operator_quality_passed
            and qa.computed_unresolved_count == 0
            and qa.computed_negative_count == 1
        ),
        computed_minimum=bool(
            qa.computed_operator_quality_passed
            and qa.computed_unresolved_count == 0
            and qa.computed_negative_count == 0
        ),
        smallest_absolute_eigenvalue=float(np.min(np.abs(qa.computed_eigenvalues))),
        geometry_sha256=geometry_identity,
        mass_sha256=_array_sha256(masses),
        hessian_sha256=_array_sha256(hessian),
        model_input_sha256=current_model_input,
        charge=context.charge,
        multiplicity=context.multiplicity,
        evaluation_sha256=evaluation_identity,
        configuration_sha256=configuration_identity,
        scalar_contract_id=scalar_identity,
        provider_id=getattr(evaluation, "provider_id", None),
        profile_id=getattr(evaluation, "profile_id", None),
        verification_evidence_id=getattr(evaluation, "verification_evidence_id", None),
        computation_dtype=computation_dtype,
        device=device,
    )


def assess_computed_stationary_point(
    atoms, context
) -> ComputedStationaryPointAssessment:
    """Derive the mode basis internally; callers cannot inject stale analysis."""

    from maple.function.calculator.route2 import is_identified_hessian_evaluation
    from .molecular_modes import analyze_hessian_evaluation

    if not is_identified_hessian_evaluation(context):
        raise TypeError(
            "computed assessment requires an identified analytic Hessian context."
        )
    return _assess_from_mode_analysis(
        atoms,
        context,
        analyze_hessian_evaluation(atoms, context.evaluation),
    )


def analyze_identified_hessian(atoms, context) -> ComputedHessianAnalysis:
    """Reuse one molecular-mode eigensolve for interval and computed claims."""

    from maple.function.calculator.route2 import is_identified_hessian_evaluation
    from .molecular_modes import analyze_hessian_evaluation

    if not is_identified_hessian_evaluation(context):
        raise TypeError(
            "computed analysis requires an identified analytic Hessian context."
        )
    analysis = analyze_hessian_evaluation(atoms, context.evaluation)
    assessment = _assess_from_mode_analysis(atoms, context, analysis)
    return ComputedHessianAnalysis(
        context=context,
        mode_analysis=analysis,
        stationary_assessment=assessment,
    )


def computed_assessment_record(
    assessment: ComputedStationaryPointAssessment,
) -> dict[str, object]:
    """Serialize claims while naming unavailable certification explicitly."""

    return {
        "computed_eigenvalues_eV_per_A2_amu": assessment.computed_eigenvalues_eV_per_A2_amu.tolist(),
        "computed_negative_count": assessment.computed_negative_count,
        "computed_positive_count": assessment.computed_positive_count,
        "computed_unresolved_count": assessment.computed_unresolved_count,
        "normalized_eigenpair_residual": assessment.normalized_eigenpair_residual,
        "eigenvector_orthogonality_defect": assessment.eigenvector_orthogonality_defect,
        "rigid_subspace_leakage": assessment.rigid_subspace_leakage,
        "computed_resolution_radius_eV_per_A2_amu": assessment.computed_resolution_radius,
        "computed_operator_quality_gate": assessment.quality_gate,
        "computed_operator_quality_passed": assessment.computed_operator_quality_passed,
        "computed_index_one": assessment.computed_index_one,
        "computed_minimum": assessment.computed_minimum,
        "smallest_absolute_eigenvalue_eV_per_A2_amu": assessment.smallest_absolute_eigenvalue,
        "geometry_sha256": assessment.geometry_sha256,
        "mass_sha256": assessment.mass_sha256,
        "hessian_sha256": assessment.hessian_sha256,
        "model_input_sha256": assessment.model_input_sha256,
        "charge": assessment.charge,
        "multiplicity": assessment.multiplicity,
        "evaluation_sha256": assessment.evaluation_sha256,
        "configuration_sha256": assessment.configuration_sha256,
        "scalar_contract_id": assessment.scalar_contract_id,
        "provider_id": assessment.provider_id,
        "profile_id": assessment.profile_id,
        "verification_evidence_id": assessment.verification_evidence_id,
        "computation_dtype": assessment.computation_dtype,
        "device": assessment.device,
        "independent_hessian_uncertainty": None,
        "interval_certified": None,
        "claim_boundary": "stored_computed_operator_only",
    }


__all__ = [
    "ComputedHessianAnalysis",
    "ComputedStationaryPointAssessment",
    "analyze_identified_hessian",
    "assess_computed_stationary_point",
    "computed_assessment_record",
]
