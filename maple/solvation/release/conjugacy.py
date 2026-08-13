"""Preregistered algebra for the MACE-POLAR Tier-V no-go canary.

This module is deliberately NumPy-only.  It does not load a checkpoint and it
does not know how MACE computes either energy or source.  A real-stack runner
supplies raw values from the independently exposed checkpoint interfaces and
this module evaluates the four frozen questions:

* direct energy--source conjugacy for both possible signs;
* intrinsic-energy stationarity for both possible signs;
* the source-embedding missing-subspace witness; and
* reciprocity on the fixed-charge, gauge-reduced field space.

One material missing-subspace component is a counterexample to a common
scalar that preserves both the original checkpoint energy and the original
four-channel source.  It is negative evidence only; passing at finitely many
states would not prove global variationality.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

CONJUGACY_NOGO_CONTRACT_VERSION = "route2-mace-conjugacy-nogo-v1"

# Frozen before the real-checkpoint result is evaluated.  Components are
# measured in the declared unit-component-scaled coordinates (one e for a
# monopole channel and one e*angstrom for a dipole channel), so directional
# derivatives have eV units.
VECTOR_ZERO_ABSOLUTE_TOLERANCE = 1.0e-10
VECTOR_ZERO_RELATIVE_TOLERANCE = 1.0e-9
GAUGE_REDUCED_RECIPROCITY_RELATIVE_TOLERANCE = 1.0e-9
ENERGY_DIRECTIONAL_FD_ABSOLUTE_TOLERANCE_EV = 2.0e-7
ENERGY_DIRECTIONAL_FD_RELATIVE_TOLERANCE = 2.0e-6
JVP_VJP_DOT_ABSOLUTE_TOLERANCE = 1.0e-10
JVP_VJP_DOT_RELATIVE_TOLERANCE = 1.0e-9

_SVD_FACTOR = 64.0


def _finite_array(values: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (ndim is not None and array.ndim != ndim) or not np.all(np.isfinite(array)):
        expected = "" if ndim is None else f" with ndim={ndim}"
        raise ValueError(f"{name} must be a finite array{expected}; got {array.shape}.")
    return np.array(array, copy=True)


def _orthonormal_nullspace(matrix: np.ndarray) -> np.ndarray:
    values = _finite_array(matrix, name="nullspace constraint", ndim=2)
    _, singular_values, vh = np.linalg.svd(values, full_matrices=True)
    scale = 1.0 if singular_values.size == 0 else max(1.0, float(singular_values[0]))
    tolerance = _SVD_FACTOR * np.finfo(float).eps * max(values.shape, default=1) * scale
    rank = int(np.count_nonzero(singular_values > tolerance))
    basis = vh[rank:].T.copy()
    if basis.size:
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), rtol=0.0, atol=2.0e-13):
            raise RuntimeError("SVD failed to construct an orthonormal nullspace.")
    return basis


def _relative_norm(residual_norm: float, reference_norm: float) -> float:
    return float(residual_norm / max(reference_norm, np.finfo(float).tiny))


def _zero_threshold(reference_norm: float) -> float:
    return float(
        VECTOR_ZERO_ABSOLUTE_TOLERANCE + VECTOR_ZERO_RELATIVE_TOLERANCE * reference_norm
    )


@dataclass(frozen=True, slots=True)
class VectorDefect:
    """Normed zero test in the frozen dimensionless component coordinates."""

    absolute_l2: float
    relative_l2: float
    reference_l2: float
    threshold: float
    numerically_zero: bool

    @classmethod
    def from_vector(cls, vector: np.ndarray, *, reference_l2: float) -> "VectorDefect":
        absolute = float(np.linalg.norm(vector))
        reference = float(reference_l2)
        if not all(
            math.isfinite(value) and value >= 0.0 for value in (absolute, reference)
        ):
            raise ValueError("Vector-defect norms must be finite and nonnegative.")
        threshold = _zero_threshold(reference)
        return cls(
            absolute_l2=absolute,
            relative_l2=_relative_norm(absolute, reference),
            reference_l2=reference,
            threshold=threshold,
            numerically_zero=absolute <= threshold,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "absolute_l2": self.absolute_l2,
            "relative_l2": self.relative_l2,
            "reference_l2": self.reference_l2,
            "threshold": self.threshold,
            "numerically_zero": self.numerically_zero,
        }


@dataclass(frozen=True, slots=True)
class SignedConjugacyResult:
    """One sign candidate for a field-covector identity."""

    sign: int
    full: VectorDefect
    gauge_reduced: VectorDefect
    gauge_direction_absolute: float
    gauge_direction_threshold: float
    gauge_direction_zero: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "sign": self.sign,
            "full": self.full.as_dict(),
            "gauge_reduced": self.gauge_reduced.as_dict(),
            "gauge_direction_absolute": self.gauge_direction_absolute,
            "gauge_direction_threshold": self.gauge_direction_threshold,
            "gauge_direction_zero": self.gauge_direction_zero,
            "all_allowed_directions_zero": (
                self.gauge_reduced.numerically_zero and self.gauge_direction_zero
            ),
        }


@dataclass(frozen=True, slots=True)
class MissingSubspaceResult:
    """Projection of the energy gradient onto directions unseen by the source."""

    full_dimension: int
    gauge_reduced_dimension: int
    full: VectorDefect
    gauge_reduced: VectorDefect
    per_atom_full_l2: tuple[float, ...]
    full_projected_gradient: tuple[float, ...]
    gauge_reduced_projected_gradient: tuple[float, ...]

    @property
    def no_go_witness_detected(self) -> bool:
        # The reduced witness cannot be dismissed as a constant-potential
        # gauge direction and is therefore the decisive counterexample.
        return not self.gauge_reduced.numerically_zero

    def as_dict(self) -> dict[str, object]:
        return {
            "full_dimension": self.full_dimension,
            "gauge_reduced_dimension": self.gauge_reduced_dimension,
            "full": self.full.as_dict(),
            "gauge_reduced": self.gauge_reduced.as_dict(),
            "per_atom_full_l2": list(self.per_atom_full_l2),
            "full_projected_gradient": list(self.full_projected_gradient),
            "gauge_reduced_projected_gradient": list(
                self.gauge_reduced_projected_gradient
            ),
            "no_go_witness_detected": self.no_go_witness_detected,
        }


@dataclass(frozen=True, slots=True)
class GaugeReducedReciprocityResult:
    """Antisymmetry of ``W.T Q.T J_M W`` on the fixed-charge field chart."""

    reduced_dimension: int
    absolute_frobenius: float
    relative_frobenius: float
    tolerance: float
    reciprocal: bool
    susceptibility_frobenius: float

    def as_dict(self) -> dict[str, object]:
        return {
            "reduced_dimension": self.reduced_dimension,
            "absolute_frobenius": self.absolute_frobenius,
            "relative_frobenius": self.relative_frobenius,
            "tolerance": self.tolerance,
            "reciprocal": self.reciprocal,
            "susceptibility_frobenius": self.susceptibility_frobenius,
        }


@dataclass(frozen=True, slots=True)
class MACEPolarConjugacyNoGoResult:
    """Complete result for one geometry and one physical radial field."""

    atom_count: int
    component_count: int
    learned_component_count: int
    energy_gradient_l2: float
    source_l2: float
    total_charge: float
    gauge_energy_directional_derivative: float
    gauge_source_directional_derivative: float
    gauge_source_response_l2: float
    direct: tuple[SignedConjugacyResult, ...]
    intrinsic_stationarity: tuple[SignedConjugacyResult, ...]
    missing_subspace: MissingSubspaceResult
    gauge_reduced_reciprocity: GaugeReducedReciprocityResult
    contract_version: str = CONJUGACY_NOGO_CONTRACT_VERSION

    @property
    def no_go_witness_detected(self) -> bool:
        return self.missing_subspace.no_go_witness_detected

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "atom_count": self.atom_count,
            "component_count": self.component_count,
            "learned_component_count": self.learned_component_count,
            "energy_gradient_l2": self.energy_gradient_l2,
            "source_l2": self.source_l2,
            "total_charge": self.total_charge,
            "gauge_energy_directional_derivative": (
                self.gauge_energy_directional_derivative
            ),
            "gauge_source_directional_derivative": (
                self.gauge_source_directional_derivative
            ),
            "gauge_source_response_l2": self.gauge_source_response_l2,
            "direct": {str(item.sign): item.as_dict() for item in self.direct},
            "intrinsic_stationarity": {
                str(item.sign): item.as_dict() for item in self.intrinsic_stationarity
            },
            "missing_subspace": self.missing_subspace.as_dict(),
            "gauge_reduced_reciprocity": self.gauge_reduced_reciprocity.as_dict(),
            "no_go_witness_detected": self.no_go_witness_detected,
            "claim_boundary": (
                "A nonzero witness is a decisive counterexample at this state; "
                "a zero witness at finitely many states is not a global proof."
            ),
        }


def _signed_result(
    residual: np.ndarray,
    *,
    field_chart: np.ndarray,
    gauge_direction_unit: np.ndarray,
    reference_l2: float,
) -> SignedConjugacyResult:
    reduced = field_chart.T @ residual
    gauge_value = abs(float(np.vdot(gauge_direction_unit, residual)))
    threshold = _zero_threshold(reference_l2)
    return SignedConjugacyResult(
        sign=0,  # replaced by caller
        full=VectorDefect.from_vector(residual, reference_l2=reference_l2),
        gauge_reduced=VectorDefect.from_vector(reduced, reference_l2=reference_l2),
        gauge_direction_absolute=gauge_value,
        gauge_direction_threshold=threshold,
        gauge_direction_zero=gauge_value <= threshold,
    )


def analyze_mace_polar_conjugacy(
    *,
    intrinsic_energy_field_gradient: object,
    original_embedded_source: object,
    physical_field: object,
    source_jacobian: object,
    source_embedding: object,
    pairing_block: object,
    charge_weights: object,
) -> MACEPolarConjugacyNoGoResult:
    """Evaluate the frozen no-go identities from raw checkpoint derivatives.

    ``source_jacobian`` uses the flattened atom-major convention
    ``J[output_source, input_field]``.  ``source_embedding`` maps one atom's
    learned source column into the public radial source column.  No assumption
    that the pairing block is symmetric is made.
    """

    gradient = _finite_array(
        intrinsic_energy_field_gradient,
        name="intrinsic_energy_field_gradient",
        ndim=2,
    )
    source = _finite_array(
        original_embedded_source, name="original_embedded_source", ndim=2
    )
    field = _finite_array(physical_field, name="physical_field", ndim=2)
    if gradient.shape != source.shape or field.shape != source.shape:
        raise ValueError("Energy gradient, source, and field must share shape (N,C).")
    atom_count, component_count = source.shape
    if atom_count < 1 or component_count < 2:
        raise ValueError(
            "The conjugacy canary requires at least one atom and two components."
        )

    jacobian = _finite_array(source_jacobian, name="source_jacobian", ndim=2)
    dimension = atom_count * component_count
    if jacobian.shape != (dimension, dimension):
        raise ValueError(
            f"source_jacobian must have shape {(dimension, dimension)}; "
            f"received {jacobian.shape}."
        )
    embedding = _finite_array(source_embedding, name="source_embedding", ndim=2)
    if (
        embedding.shape[0] != component_count
        or not 0 < embedding.shape[1] < component_count
    ):
        raise ValueError("source_embedding must be a strict C by L embedding.")
    if np.linalg.matrix_rank(embedding) != embedding.shape[1]:
        raise ValueError("source_embedding must have full column rank.")
    pairing = _finite_array(pairing_block, name="pairing_block", ndim=2)
    if pairing.shape != (component_count, component_count):
        raise ValueError("pairing_block must be square on the component space.")
    if np.linalg.matrix_rank(pairing) != component_count:
        raise ValueError("pairing_block must be invertible.")
    weights = _finite_array(charge_weights, name="charge_weights", ndim=1)
    if weights.shape != (component_count,) or not np.any(weights):
        raise ValueError("charge_weights must be one nonzero component block.")

    q_full = np.kron(np.eye(atom_count), pairing)
    s_full = np.kron(np.eye(atom_count), embedding)
    a = np.tile(weights, atom_count)
    gradient_flat = gradient.reshape(-1)
    source_flat = source.reshape(-1)
    field_flat = field.reshape(-1)

    # The no-go theorem being tested assumes the advertised original response
    # really is M_orig=S m.  Reject an inconsistent caller instead of
    # attributing an adapter bug to checkpoint physics.
    source_projector = s_full @ np.linalg.pinv(s_full)
    range_scale = max(1.0, float(np.linalg.norm(source_flat)))
    if float(np.linalg.norm((np.eye(dimension) - source_projector) @ source_flat)) > (
        2.0e-13 * range_scale
    ):
        raise ValueError("original_embedded_source is outside range(source_embedding).")
    jacobian_scale = max(1.0, float(np.linalg.norm(jacobian, ord="fro")))
    if (
        float(
            np.linalg.norm((np.eye(dimension) - source_projector) @ jacobian, ord="fro")
        )
        > 2.0e-13 * jacobian_scale
    ):
        raise ValueError("source_jacobian outputs are outside range(source_embedding).")

    # T is an orthonormal source-tangent basis.  W is its Q-dual field chart:
    # T.T Q W = I.  This is the exact reduced space from the audit theorem.
    source_tangent = _orthonormal_nullspace(a.reshape(1, -1))
    field_chart = np.linalg.solve(q_full, source_tangent)
    duality = source_tangent.T @ q_full @ field_chart
    if not np.allclose(duality, np.eye(dimension - 1), rtol=0.0, atol=5.0e-13):
        raise RuntimeError("Unable to construct the Q-dual reduced field chart.")

    gauge_direction = np.linalg.solve(q_full, a)
    gauge_norm = float(np.linalg.norm(gauge_direction))
    if not math.isfinite(gauge_norm) or gauge_norm <= np.finfo(float).tiny:
        raise RuntimeError("Constant-potential gauge direction is singular.")
    gauge_direction_unit = gauge_direction / gauge_norm

    source_dual = q_full.T @ source_flat
    q_field = q_full @ field_flat
    source_response_dual = jacobian.T @ q_field
    gradient_norm = float(np.linalg.norm(gradient_flat))
    source_dual_norm = float(np.linalg.norm(source_dual))

    direct_results: list[SignedConjugacyResult] = []
    intrinsic_results: list[SignedConjugacyResult] = []
    for sign in (-1, 1):
        direct_residual = gradient_flat - sign * source_dual
        direct = _signed_result(
            direct_residual,
            field_chart=field_chart,
            gauge_direction_unit=gauge_direction_unit,
            reference_l2=max(gradient_norm, source_dual_norm),
        )
        direct_results.append(
            SignedConjugacyResult(
                sign,
                direct.full,
                direct.gauge_reduced,
                direct.gauge_direction_absolute,
                direct.gauge_direction_threshold,
                direct.gauge_direction_zero,
            )
        )

        intrinsic_residual = gradient_flat + sign * source_response_dual
        intrinsic = _signed_result(
            intrinsic_residual,
            field_chart=field_chart,
            gauge_direction_unit=gauge_direction_unit,
            reference_l2=max(
                gradient_norm, float(np.linalg.norm(source_response_dual))
            ),
        )
        intrinsic_results.append(
            SignedConjugacyResult(
                sign,
                intrinsic.full,
                intrinsic.gauge_reduced,
                intrinsic.gauge_direction_absolute,
                intrinsic.gauge_direction_threshold,
                intrinsic.gauge_direction_zero,
            )
        )

    missing_block = _orthonormal_nullspace(embedding.T @ pairing)
    full_missing = np.kron(np.eye(atom_count), missing_block)
    full_projection = full_missing.T @ gradient_flat
    # Remove the one constant-potential gauge coordinate without assuming Q=I.
    reduced_missing_constraints = np.vstack(
        (s_full.T @ q_full, a.reshape(1, -1) @ q_full)
    )
    reduced_missing = _orthonormal_nullspace(reduced_missing_constraints)
    reduced_projection = reduced_missing.T @ gradient_flat
    per_atom = tuple(
        float(np.linalg.norm(gradient[index] @ missing_block))
        for index in range(atom_count)
    )
    missing_result = MissingSubspaceResult(
        full_dimension=full_missing.shape[1],
        gauge_reduced_dimension=reduced_missing.shape[1],
        full=VectorDefect.from_vector(full_projection, reference_l2=gradient_norm),
        gauge_reduced=VectorDefect.from_vector(
            reduced_projection, reference_l2=gradient_norm
        ),
        per_atom_full_l2=per_atom,
        full_projected_gradient=tuple(float(value) for value in full_projection),
        gauge_reduced_projected_gradient=tuple(
            float(value) for value in reduced_projection
        ),
    )

    reduced_susceptibility = field_chart.T @ q_full.T @ jacobian @ field_chart
    antisymmetric = reduced_susceptibility - reduced_susceptibility.T
    susceptibility_norm = float(np.linalg.norm(reduced_susceptibility, ord="fro"))
    antisymmetric_norm = float(np.linalg.norm(antisymmetric, ord="fro"))
    reciprocity_relative = antisymmetric_norm / max(
        susceptibility_norm, np.finfo(float).tiny
    )
    reciprocity = GaugeReducedReciprocityResult(
        reduced_dimension=dimension - 1,
        absolute_frobenius=antisymmetric_norm,
        relative_frobenius=reciprocity_relative,
        tolerance=GAUGE_REDUCED_RECIPROCITY_RELATIVE_TOLERANCE,
        reciprocal=(
            reciprocity_relative <= GAUGE_REDUCED_RECIPROCITY_RELATIVE_TOLERANCE
        ),
        susceptibility_frobenius=susceptibility_norm,
    )

    return MACEPolarConjugacyNoGoResult(
        atom_count=atom_count,
        component_count=component_count,
        learned_component_count=embedding.shape[1],
        energy_gradient_l2=gradient_norm,
        source_l2=float(np.linalg.norm(source_flat)),
        total_charge=float(np.vdot(a, source_flat)),
        gauge_energy_directional_derivative=float(
            np.vdot(gradient_flat, gauge_direction)
        ),
        gauge_source_directional_derivative=float(
            np.vdot(source_flat, q_full @ gauge_direction)
        ),
        gauge_source_response_l2=float(np.linalg.norm(jacobian @ gauge_direction)),
        direct=tuple(direct_results),
        intrinsic_stationarity=tuple(intrinsic_results),
        missing_subspace=missing_result,
        gauge_reduced_reciprocity=reciprocity,
    )


def tolerance_contract() -> dict[str, object]:
    """Return the immutable thresholds embedded in this source version."""

    return {
        "contract_version": CONJUGACY_NOGO_CONTRACT_VERSION,
        "vector_zero_absolute": VECTOR_ZERO_ABSOLUTE_TOLERANCE,
        "vector_zero_relative": VECTOR_ZERO_RELATIVE_TOLERANCE,
        "gauge_reduced_reciprocity_relative": (
            GAUGE_REDUCED_RECIPROCITY_RELATIVE_TOLERANCE
        ),
        "energy_directional_fd_absolute_eV": (
            ENERGY_DIRECTIONAL_FD_ABSOLUTE_TOLERANCE_EV
        ),
        "energy_directional_fd_relative": (ENERGY_DIRECTIONAL_FD_RELATIVE_TOLERANCE),
        "jvp_vjp_dot_absolute": JVP_VJP_DOT_ABSOLUTE_TOLERANCE,
        "jvp_vjp_dot_relative": JVP_VJP_DOT_RELATIVE_TOLERANCE,
        "decision": (
            "no-go iff the gauge-reduced projection of the intrinsic-energy "
            "field gradient onto ker(S.T Q) is not numerically zero"
        ),
    }


__all__ = [
    "CONJUGACY_NOGO_CONTRACT_VERSION",
    "ENERGY_DIRECTIONAL_FD_ABSOLUTE_TOLERANCE_EV",
    "ENERGY_DIRECTIONAL_FD_RELATIVE_TOLERANCE",
    "GAUGE_REDUCED_RECIPROCITY_RELATIVE_TOLERANCE",
    "JVP_VJP_DOT_ABSOLUTE_TOLERANCE",
    "JVP_VJP_DOT_RELATIVE_TOLERANCE",
    "MACEPolarConjugacyNoGoResult",
    "VECTOR_ZERO_ABSOLUTE_TOLERANCE",
    "VECTOR_ZERO_RELATIVE_TOLERANCE",
    "analyze_mace_polar_conjugacy",
    "tolerance_contract",
]
