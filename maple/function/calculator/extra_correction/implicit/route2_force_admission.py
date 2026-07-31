"""Fail-closed admission evidence for a Route-2 force profile.

This module intentionally does *not* expose an ASE force.  It turns the
separate numerical prerequisites for a conservative Route-2 fixed-point force
into one machine-readable record: a local residual condition screen, primal
and adjoint residuals, continuum smoothness evidence, root-repeat evidence,
and the unresolved common-energy question.

In particular, an iterative estimate of ``||J_M J_P||`` is not a proof of a
lower singular-value bound.  Small systems use an explicit neutral-space SVD;
larger systems are recorded as an insufficient matrix-free screen and remain
fail-closed until a validated bound is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .route2_response import FixedChargeCoordinates
from .route2_thermodynamic_diagnostics import (
    FixedPointFeedbackSpectrumDiagnostic,
    fixed_point_feedback_spectrum_diagnostic,
)

FORCE_ADMISSION_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ContinuumSmoothnessContract:
    """Evidence supplied by one continuum implementation/profile.

    ``fixed_node_topology`` means the discrete degree of freedom count and its
    ownership map remain fixed under the admitted coordinate neighbourhood.
    A smooth switching weight alone is insufficient when upstream code drops
    grid points or changes their parent atom.
    """

    profile_kind: str
    same_energy_coordinate_derivative: bool
    fixed_node_topology: bool
    geometry_path_smoothness_verified: bool
    evidence: str

    def __post_init__(self) -> None:
        if not self.profile_kind.strip():
            raise ValueError("A continuum smoothness profile kind is required.")
        if not self.evidence.strip():
            raise ValueError("Continuum smoothness evidence is required.")

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_kind": self.profile_kind,
            "same_energy_coordinate_derivative": (
                self.same_energy_coordinate_derivative
            ),
            "fixed_node_topology": self.fixed_node_topology,
            "geometry_path_smoothness_verified": (
                self.geometry_path_smoothness_verified
            ),
            "evidence": self.evidence,
        }


PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT = ContinuumSmoothnessContract(
    profile_kind="pyddx-ddpcm-active-set-v1",
    same_energy_coordinate_derivative=True,
    fixed_node_topology=False,
    geometry_path_smoothness_verified=False,
    evidence=(
        "The same-energy coordinate VJP exists at one fixed active set, but "
        "Lebedev/sphere ownership changes under admitted geometry scans."
    ),
)

PYSCF_SWIG_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT = ContinuumSmoothnessContract(
    profile_kind="pyscf-swig-variable-surface-v1",
    same_energy_coordinate_derivative=True,
    fixed_node_topology=False,
    geometry_path_smoothness_verified=False,
    evidence=(
        "PySCF SWIG supplies a same-energy operator gradient, but current "
        "surface construction changes retained grid-point counts/parents "
        "across geometry/orientation canaries."
    ),
)

UNSPECIFIED_CONTINUUM_SMOOTHNESS_CONTRACT = ContinuumSmoothnessContract(
    profile_kind="unspecified-route2-continuum-v1",
    same_energy_coordinate_derivative=False,
    fixed_node_topology=False,
    geometry_path_smoothness_verified=False,
    evidence=(
        "No profile-specific same-energy, fixed-topology, or geometry-path "
        "smoothness evidence was supplied."
    ),
)


@dataclass(frozen=True)
class ForceAdmissionPolicy:
    """Predeclared numerical requirements for a small-system force gate."""

    maximum_primal_monopole_residual_e: float = 1.0e-9
    maximum_primal_dipole_residual_e_angstrom: float = 1.0e-9
    maximum_adjoint_relative_residual: float = 1.0e-8
    maximum_continuum_identity_error_ev: float = 1.0e-10
    minimum_residual_singular_value: float = 1.0e-5
    maximum_residual_condition_number_2: float = 1.0e5
    maximum_feedback_singular_value: float = 0.95
    maximum_dense_dimension: int = 128

    def __post_init__(self) -> None:
        positive = (
            self.maximum_primal_monopole_residual_e,
            self.maximum_primal_dipole_residual_e_angstrom,
            self.maximum_adjoint_relative_residual,
            self.maximum_continuum_identity_error_ev,
            self.minimum_residual_singular_value,
            self.maximum_residual_condition_number_2,
            self.maximum_feedback_singular_value,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Route-2 force-admission thresholds must be positive.")
        if self.maximum_feedback_singular_value >= 1.0:
            raise ValueError(
                "The feedback singular-value gate must be strictly below one."
            )
        if self.maximum_dense_dimension <= 0:
            raise ValueError("The dense condition-screen dimension must be positive.")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "maximum_primal_monopole_residual_e": (
                self.maximum_primal_monopole_residual_e
            ),
            "maximum_primal_dipole_residual_e_angstrom": (
                self.maximum_primal_dipole_residual_e_angstrom
            ),
            "maximum_adjoint_relative_residual": (
                self.maximum_adjoint_relative_residual
            ),
            "maximum_continuum_identity_error_ev": (
                self.maximum_continuum_identity_error_ev
            ),
            "minimum_residual_singular_value": (self.minimum_residual_singular_value),
            "maximum_residual_condition_number_2": (
                self.maximum_residual_condition_number_2
            ),
            "maximum_feedback_singular_value": (self.maximum_feedback_singular_value),
            "maximum_dense_dimension": self.maximum_dense_dimension,
        }


@dataclass(frozen=True)
class FixedPointConditionScreen:
    """Local conditioning evidence for ``A = I - J_M J_P``.

    ``exact_small_system`` is deliberately narrow: it means all reduced
    columns were evaluated and a dense floating-point SVD was performed.  A
    matrix-free Ritz value is useful for triage, but cannot certify a force
    admission lower bound by itself.
    """

    dimension: int
    method: str
    exact_small_system: bool
    feedback_largest_singular_value: float | None
    residual_smallest_singular_value: float | None
    residual_condition_number_2: float | None
    residual_is_numerically_singular: bool | None
    gate_passed: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Fixed-point condition-screen dimension must be positive.")
        for value in (
            self.feedback_largest_singular_value,
            self.residual_smallest_singular_value,
            self.residual_condition_number_2,
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError("Fixed-point condition metrics must be finite.")
        if self.gate_passed and not self.exact_small_system:
            raise ValueError(
                "A matrix-free condition screen cannot certify force admission."
            )
        if self.gate_passed and self.reason is not None:
            raise ValueError(
                "A passing condition screen must not carry a failure reason."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "method": self.method,
            "exact_small_system": self.exact_small_system,
            "feedback_largest_singular_value": self.feedback_largest_singular_value,
            "residual_smallest_singular_value": self.residual_smallest_singular_value,
            "residual_condition_number_2": self.residual_condition_number_2,
            "residual_is_numerically_singular": (self.residual_is_numerically_singular),
            "gate_passed": self.gate_passed,
            "reason": self.reason,
        }


def _condition_screen_from_dense_diagnostic(
    diagnostic: FixedPointFeedbackSpectrumDiagnostic,
    policy: ForceAdmissionPolicy,
) -> FixedPointConditionScreen:
    condition = diagnostic.residual_operator_i_minus_feedback
    feedback = diagnostic.largest_singular_value
    smallest = condition.smallest_singular_value
    condition_number = condition.condition_number_2
    failures = []
    if condition.numerically_singular:
        failures.append("residual-operator-numerically-singular")
    if feedback > policy.maximum_feedback_singular_value:
        failures.append("feedback-gain-above-predeclared-gate")
    if smallest < policy.minimum_residual_singular_value:
        failures.append("residual-smallest-singular-value-below-gate")
    if (
        condition_number is None
        or condition_number > policy.maximum_residual_condition_number_2
    ):
        failures.append("residual-condition-number-above-gate")
    return FixedPointConditionScreen(
        dimension=diagnostic.dimension,
        method="dense-neutral-space-svd-v1",
        exact_small_system=True,
        feedback_largest_singular_value=feedback,
        residual_smallest_singular_value=smallest,
        residual_condition_number_2=condition_number,
        residual_is_numerically_singular=condition.numerically_singular,
        gate_passed=not failures,
        reason=None if not failures else ",".join(failures),
    )


def fixed_point_condition_screen(
    residual_linearization,
    *,
    policy: ForceAdmissionPolicy = ForceAdmissionPolicy(),
) -> FixedPointConditionScreen:
    """Return exact bounded-dimensional conditioning or an explicit non-pass."""

    atom_count = int(getattr(residual_linearization, "atom_count", 0))
    if atom_count <= 0:
        raise ValueError("Fixed-point condition screen requires a positive atom count.")
    dimension = FixedChargeCoordinates(atom_count).dimension
    if dimension <= policy.maximum_dense_dimension:
        diagnostic = fixed_point_feedback_spectrum_diagnostic(
            residual_linearization,
            maximum_dimension=policy.maximum_dense_dimension,
        )
        return _condition_screen_from_dense_diagnostic(diagnostic, policy)

    # The standalone matrix-free gain diagnostic is intentionally not used by
    # a public/admission calculation path.  Its Ritz value is a useful
    # diagnostic estimate, but it is not a certified upper gain bound or a
    # lower bound on sigma_min(I - J_M J_P).  Do not turn a non-pass screen
    # into a potentially misleading expensive calculation.
    return FixedPointConditionScreen(
        dimension=dimension,
        method="dense-fixed-charge-condition-screen-required-v1",
        exact_small_system=False,
        feedback_largest_singular_value=None,
        residual_smallest_singular_value=None,
        residual_condition_number_2=None,
        residual_is_numerically_singular=None,
        gate_passed=False,
        reason=(
            "full-fixed-charge-condition-screen-exceeds-declared-dense-"
            "dimension-bound"
        ),
    )


@dataclass(frozen=True)
class ForceAdmissionCertificate:
    """One fail-closed Route-2 force-release decision record."""

    contract_version: int
    policy: ForceAdmissionPolicy
    nominal_root: bool
    primal_monopole_residual_e: float
    primal_dipole_residual_e_angstrom: float
    adjoint_relative_residual: float
    continuum_identity_error_ev: float
    condition: FixedPointConditionScreen
    continuum: ContinuumSmoothnessContract
    multi_start_root_agreement: bool | None
    operational_energy_semantics_closed: bool
    release_admitted: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_version != FORCE_ADMISSION_CONTRACT_VERSION:
            raise ValueError("Unsupported Route-2 force-admission contract version.")
        for value in (
            self.primal_monopole_residual_e,
            self.primal_dipole_residual_e_angstrom,
            self.adjoint_relative_residual,
            self.continuum_identity_error_ev,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Force-admission residuals must be finite and nonnegative."
                )
        if self.release_admitted != (not self.failure_reasons):
            raise ValueError("Force-admission decision must match its failure reasons.")

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "policy": self.policy.as_dict(),
            "nominal_root": self.nominal_root,
            "primal_monopole_residual_e": self.primal_monopole_residual_e,
            "primal_dipole_residual_e_angstrom": (
                self.primal_dipole_residual_e_angstrom
            ),
            "adjoint_relative_residual": self.adjoint_relative_residual,
            "continuum_identity_error_ev": self.continuum_identity_error_ev,
            "condition": self.condition.as_dict(),
            "continuum": self.continuum.as_dict(),
            "multi_start_root_agreement": self.multi_start_root_agreement,
            "operational_energy_semantics_closed": (
                self.operational_energy_semantics_closed
            ),
            "release_admitted": self.release_admitted,
            "failure_reasons": list(self.failure_reasons),
        }


def evaluate_force_admission(
    residual_linearization,
    *,
    nominal_root: bool,
    primal_monopole_residual_e: float,
    primal_dipole_residual_e_angstrom: float,
    adjoint_relative_residual: float,
    continuum_identity_error_ev: float,
    continuum: ContinuumSmoothnessContract,
    multi_start_root_agreement: bool | None,
    operational_energy_semantics_closed: bool,
    policy: ForceAdmissionPolicy = ForceAdmissionPolicy(),
) -> ForceAdmissionCertificate:
    """Assess every prerequisite without changing the public force API."""

    values = {
        "primal_monopole_residual_e": primal_monopole_residual_e,
        "primal_dipole_residual_e_angstrom": (primal_dipole_residual_e_angstrom),
        "adjoint_relative_residual": adjoint_relative_residual,
        "continuum_identity_error_ev": continuum_identity_error_ev,
    }
    normalized = {}
    for name, raw_value in values.items():
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        normalized[name] = value

    condition = fixed_point_condition_screen(
        residual_linearization,
        policy=policy,
    )
    failures = []
    if not nominal_root:
        failures.append("fixed-point-root-is-not-nominal")
    if (
        normalized["primal_monopole_residual_e"]
        > policy.maximum_primal_monopole_residual_e
    ):
        failures.append("primal-monopole-residual-above-gate")
    if (
        normalized["primal_dipole_residual_e_angstrom"]
        > policy.maximum_primal_dipole_residual_e_angstrom
    ):
        failures.append("primal-dipole-residual-above-gate")
    if (
        normalized["adjoint_relative_residual"]
        > policy.maximum_adjoint_relative_residual
    ):
        failures.append("adjoint-residual-above-gate")
    if (
        normalized["continuum_identity_error_ev"]
        > policy.maximum_continuum_identity_error_ev
    ):
        failures.append("continuum-energy-identity-above-gate")
    if not condition.gate_passed:
        failures.append("fixed-point-condition-screen-not-admitted")
    if not continuum.same_energy_coordinate_derivative:
        failures.append("continuum-coordinate-derivative-does-not-match-energy")
    if not continuum.fixed_node_topology:
        failures.append("continuum-node-topology-is-not-fixed")
    if not continuum.geometry_path_smoothness_verified:
        failures.append("continuum-geometry-path-smoothness-unverified")
    if multi_start_root_agreement is not True:
        failures.append("multi-start-root-uniqueness-unverified")
    if not operational_energy_semantics_closed:
        failures.append("common-energy-semantics-unresolved")

    return ForceAdmissionCertificate(
        contract_version=FORCE_ADMISSION_CONTRACT_VERSION,
        policy=policy,
        nominal_root=bool(nominal_root),
        primal_monopole_residual_e=(normalized["primal_monopole_residual_e"]),
        primal_dipole_residual_e_angstrom=(
            normalized["primal_dipole_residual_e_angstrom"]
        ),
        adjoint_relative_residual=normalized["adjoint_relative_residual"],
        continuum_identity_error_ev=normalized["continuum_identity_error_ev"],
        condition=condition,
        continuum=continuum,
        multi_start_root_agreement=multi_start_root_agreement,
        operational_energy_semantics_closed=bool(operational_energy_semantics_closed),
        release_admitted=not failures,
        failure_reasons=tuple(failures),
    )


__all__ = [
    "FORCE_ADMISSION_CONTRACT_VERSION",
    "ContinuumSmoothnessContract",
    "FixedPointConditionScreen",
    "ForceAdmissionCertificate",
    "ForceAdmissionPolicy",
    "PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT",
    "PYSCF_SWIG_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT",
    "UNSPECIFIED_CONTINUUM_SMOOTHNESS_CONTRACT",
    "evaluate_force_admission",
    "fixed_point_condition_screen",
]
