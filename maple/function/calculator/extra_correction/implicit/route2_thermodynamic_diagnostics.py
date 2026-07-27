"""Thermodynamic and fixed-point diagnostics for Route-2 response.

These diagnostics do not change the fixed-point equation or the scalar energy.
They test whether the learned density response and the reported field-dependent
energy are compatible with either of two candidate physical identities:

``intrinsic``
    The reported energy is an internal functional evaluated at the
    field-relaxed density.

``coupled``
    The reported energy already includes the full external-field coupling.

The local-field diagnostics retain their physical ``[V, grad V]`` pairing.
The native-feature conjugacy diagnostic instead composes the exact learned
feature-to-density VJP with that same density-dual continuum field; it does not
invent a density-to-feature envelope identity.  The fixed-point spectrum acts
only on the common neutral density residual and therefore supports either
local-jet or exact-GTO drives without changing the production solver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
    ElectrostaticPairing,
)
from .gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from .route2_feature_response import FeatureDensityResponseLinearization
from .route2_response import (
    DensityResponseLinearization,
    NeutralDensityCoordinates,
)

_RELATIVE_FLOOR = 1.0e-30


class UnmixedResidualLinearization(Protocol):
    """Common neutral residual interface used by the feedback spectrum."""

    atom_count: int

    def jvp(self, density_direction: np.ndarray) -> np.ndarray:
        """Apply ``Pi[c - M(P(c))]`` to a neutral density direction."""
        ...


def _validated_block(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    valid_shape = (
        array.ndim == 2
        and array.shape[0] > 0
        and array.shape[1] == 4
        and (expected_shape is None or array.shape == expected_shape)
    )
    if not valid_shape or not np.all(np.isfinite(array)):
        expected = "(n_atoms, 4)" if expected_shape is None else expected_shape
        raise ValueError(
            f"{name} must be finite with shape {expected}; "
            f"received {array.shape}."
        )
    return array


def _relative_error(
    residual_norm: float,
    left_norm: float,
    right_norm: float,
) -> float:
    denominator = left_norm + right_norm
    if denominator <= _RELATIVE_FLOOR:
        return 0.0 if residual_norm <= _RELATIVE_FLOOR else float("inf")
    return float(residual_norm / denominator)


def _validated_feature_block(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    valid_shape = (
        array.ndim == 2
        and array.shape[0] > 0
        and array.shape[1] > 0
        and (expected_shape is None or array.shape == expected_shape)
    )
    if not valid_shape or not np.all(np.isfinite(array)):
        expected = (
            "(n_atoms, n_features)"
            if expected_shape is None
            else expected_shape
        )
        raise ValueError(
            f"{name} must be finite with shape {expected}; "
            f"received {array.shape}."
        )
    return array


@dataclass(frozen=True)
class RelativeVectorDefect:
    """Unit-preserving L2 defect between two model-feature cotangents."""

    absolute_l2: float
    relative_l2: float
    left_l2: float
    right_l2: float
    cosine_similarity: float

    def __post_init__(self) -> None:
        nonnegative = (
            self.absolute_l2,
            self.relative_l2,
            self.left_l2,
            self.right_l2,
        )
        if not all(
            np.isfinite(value) and value >= 0.0 for value in nonnegative
        ):
            raise ValueError(
                "Feature-conjugacy defect norms must be finite and "
                "nonnegative."
            )
        if (
            not np.isfinite(self.cosine_similarity)
            or not -1.0 - 1.0e-12
            <= self.cosine_similarity
            <= 1.0 + 1.0e-12
        ):
            raise ValueError(
                "Feature-conjugacy cosine similarity must lie in [-1, 1]."
            )


def _relative_vector_defect(
    left: np.ndarray,
    right: np.ndarray,
) -> RelativeVectorDefect:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError(
            "Feature-conjugacy vectors must have one shared nonempty shape."
        )
    residual_norm = float(np.linalg.norm(lhs - rhs))
    left_norm = float(np.linalg.norm(lhs))
    right_norm = float(np.linalg.norm(rhs))
    product = left_norm * right_norm
    if product <= _RELATIVE_FLOOR:
        cosine = 1.0 if residual_norm <= _RELATIVE_FLOOR else 0.0
    else:
        cosine = float(np.vdot(lhs, rhs) / product)
        cosine = float(np.clip(cosine, -1.0, 1.0))
    return RelativeVectorDefect(
        absolute_l2=residual_norm,
        relative_l2=_relative_error(
            residual_norm,
            left_norm,
            right_norm,
        ),
        left_l2=left_norm,
        right_l2=right_norm,
        cosine_similarity=cosine,
    )


@dataclass(frozen=True)
class IntrinsicFeatureConjugacyDiagnostic:
    """Test ``g_z + J_M(z)^T Q f = 0`` in native feature space."""

    feature_count: int
    l0_feature_count: int
    all_features: RelativeVectorDefect
    l0_radial_features: RelativeVectorDefect
    l1_radial_cartesian_features: RelativeVectorDefect
    identity: str = "g_z + J_M(z)^T Q f = 0"
    field_interface: str = "model-native-gto-v1"
    coupled_candidate: None = None
    coupled_candidate_reason: str = (
        "No density-to-native-feature conjugacy map is defined; "
        "Q^T c is not inferred in feature space."
    )

    def __post_init__(self) -> None:
        if self.feature_count <= 1:
            raise ValueError(
                "Native-feature conjugacy requires at least two features."
            )
        if not 0 < self.l0_feature_count < self.feature_count:
            raise ValueError(
                "The l=0 feature count must split the native feature axis."
            )
        if self.field_interface != "model-native-gto-v1":
            raise ValueError(
                "Unsupported native-feature thermodynamic interface."
            )
        if self.coupled_candidate is not None:
            raise ValueError(
                "The native-feature diagnostic cannot invent a coupled "
                "envelope candidate."
            )


def intrinsic_feature_conjugacy_diagnostic(
    *,
    reaction_field_values_ev: np.ndarray,
    intrinsic_energy_feature_gradient: np.ndarray,
    density_response: FeatureDensityResponseLinearization,
    l0_feature_count: int,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
) -> IntrinsicFeatureConjugacyDiagnostic:
    """Evaluate intrinsic energy--density conjugacy through native features.

    The model is driven by a native feature tensor ``z`` while the continuum
    energy retains the explicit density-dual field ``f``.  If the reported
    intrinsic energy is generated by one stationary density functional, then

    ``g_z + J_M(z).T @ Q f = 0``.

    A coupled-energy envelope candidate is deliberately omitted because no
    physical map from the returned density to the rectangular feature
    cotangent has been established.
    """

    field = _validated_block(
        reaction_field_values_ev,
        name="reaction_field_values_ev",
    )
    gradient = _validated_feature_block(
        intrinsic_energy_feature_gradient,
        name="intrinsic_energy_feature_gradient",
    )
    if gradient.shape[0] != field.shape[0]:
        raise ValueError(
            "The reaction field and native feature gradient must have the "
            "same atom count."
        )
    feature_count = int(gradient.shape[1])
    if not 0 < l0_feature_count < feature_count:
        raise ValueError(
            "l0_feature_count must split the native feature dimension."
        )
    expected = -_validated_feature_block(
        density_response.vjp(pairing.field_to_density_order(field)),
        name="density-response native-feature VJP",
        expected_shape=gradient.shape,
    )
    return IntrinsicFeatureConjugacyDiagnostic(
        feature_count=feature_count,
        l0_feature_count=int(l0_feature_count),
        all_features=_relative_vector_defect(gradient, expected),
        l0_radial_features=_relative_vector_defect(
            gradient[:, :l0_feature_count],
            expected[:, :l0_feature_count],
        ),
        l1_radial_cartesian_features=_relative_vector_defect(
            gradient[:, l0_feature_count:],
            expected[:, l0_feature_count:],
        ),
    )


@dataclass(frozen=True)
class BlockConjugacyDefect:
    """Unit-preserving defect for scalar and dipolar response channels."""

    monopole_absolute_l2_e: float
    monopole_relative_l2: float
    dipole_absolute_l2_e_angstrom: float
    dipole_relative_l2: float
    maximum_relative_block_error: float

    def __post_init__(self) -> None:
        values = (
            self.monopole_absolute_l2_e,
            self.monopole_relative_l2,
            self.dipole_absolute_l2_e_angstrom,
            self.dipole_relative_l2,
            self.maximum_relative_block_error,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(
                "Conjugacy-defect metrics must be finite and nonnegative."
            )


def _block_conjugacy_defect(
    residual: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> BlockConjugacyDefect:
    shape = residual.shape
    lhs = _validated_block(left, name="left conjugacy term", expected_shape=shape)
    rhs = _validated_block(
        right,
        name="right conjugacy term",
        expected_shape=shape,
    )
    defect = _validated_block(
        residual,
        name="conjugacy residual",
        expected_shape=shape,
    )

    monopole_absolute = float(np.linalg.norm(defect[:, 0]))
    monopole_relative = _relative_error(
        monopole_absolute,
        float(np.linalg.norm(lhs[:, 0])),
        float(np.linalg.norm(rhs[:, 0])),
    )
    dipole_absolute = float(np.linalg.norm(defect[:, 1:]))
    dipole_relative = _relative_error(
        dipole_absolute,
        float(np.linalg.norm(lhs[:, 1:])),
        float(np.linalg.norm(rhs[:, 1:])),
    )
    return BlockConjugacyDefect(
        monopole_absolute_l2_e=monopole_absolute,
        monopole_relative_l2=monopole_relative,
        dipole_absolute_l2_e_angstrom=dipole_absolute,
        dipole_relative_l2=dipole_relative,
        maximum_relative_block_error=max(
            monopole_relative,
            dipole_relative,
        ),
    )


@dataclass(frozen=True)
class EnergyDensityConjugacyDiagnostic:
    """Compare two identities without assigning a physical pass threshold."""

    intrinsic_candidate: BlockConjugacyDefect
    coupled_candidate: BlockConjugacyDefect
    lower_defect_candidate: str
    field_interface: str = "local-potential-gradient-v1"

    def __post_init__(self) -> None:
        if self.lower_defect_candidate not in {
            "intrinsic",
            "coupled",
            "indistinguishable",
        }:
            raise ValueError("Invalid lower-defect candidate identity.")
        if self.field_interface != "local-potential-gradient-v1":
            raise ValueError("Unsupported thermodynamic diagnostic interface.")


def energy_density_conjugacy_diagnostic(
    *,
    reaction_field_values_ev: np.ndarray,
    response_density_coefficients: np.ndarray,
    intrinsic_energy_field_gradient: np.ndarray,
    density_response: DensityResponseLinearization,
) -> EnergyDensityConjugacyDiagnostic:
    """Evaluate the two candidate energy--density conjugacy identities.

    For the intrinsic-energy interpretation the expected identity is

    ``g_f + J_M.T @ Q f = 0``.

    If the reported energy already contains the full field coupling, the
    envelope-theorem identity is instead

    ``g_f - Q.T c = 0``.

    Monopole and dipole blocks are normalized separately because their natural
    units are elementary charge and elementary-charge angstrom.  The density
    must be the model response returned at this exact field.  At a converged
    Route-2 fixed point it agrees with the root density up to the explicitly
    recorded unmixed residual.
    """

    field = _validated_block(
        reaction_field_values_ev,
        name="reaction_field_values_ev",
    )
    density = _validated_block(
        response_density_coefficients,
        name="response_density_coefficients",
        expected_shape=field.shape,
    )
    gradient = _validated_block(
        intrinsic_energy_field_gradient,
        name="intrinsic_energy_field_gradient",
        expected_shape=field.shape,
    )

    polarization_stationarity_gradient = _validated_block(
        density_response.vjp(external_field_to_density_order(field)),
        name="density-response field VJP",
        expected_shape=field.shape,
    )
    intrinsic_right = -polarization_stationarity_gradient
    intrinsic = _block_conjugacy_defect(
        gradient - intrinsic_right,
        gradient,
        intrinsic_right,
    )

    coupled_right = density_to_external_field_order(density)
    coupled = _block_conjugacy_defect(
        gradient - coupled_right,
        gradient,
        coupled_right,
    )

    intrinsic_score = intrinsic.maximum_relative_block_error
    coupled_score = coupled.maximum_relative_block_error
    if np.isclose(intrinsic_score, coupled_score, rtol=1.0e-12, atol=1.0e-15):
        lower = "indistinguishable"
    elif intrinsic_score < coupled_score:
        lower = "intrinsic"
    else:
        lower = "coupled"
    return EnergyDensityConjugacyDiagnostic(
        intrinsic_candidate=intrinsic,
        coupled_candidate=coupled,
        lower_defect_candidate=lower,
    )


@dataclass(frozen=True)
class ResponseReciprocityDiagnostic:
    """Directional symmetry check of the learned field response."""

    forward_pairing_ev: float
    reverse_pairing_ev: float
    absolute_error_ev: float
    relative_error: float

    def __post_init__(self) -> None:
        signed = (self.forward_pairing_ev, self.reverse_pairing_ev)
        nonnegative = (self.absolute_error_ev, self.relative_error)
        if not all(np.isfinite(value) for value in signed) or not all(
            np.isfinite(value) and value >= 0.0 for value in nonnegative
        ):
            raise ValueError("Response-reciprocity metrics must be finite.")


def response_reciprocity_diagnostic(
    density_response: DensityResponseLinearization,
    first_field_direction: np.ndarray,
    second_field_direction: np.ndarray,
    *,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
) -> ResponseReciprocityDiagnostic:
    """Return ``<J v,u> - <J u,v>`` in the electrostatic pairing."""

    first = _validated_block(
        first_field_direction,
        name="first_field_direction",
    )
    second = _validated_block(
        second_field_direction,
        name="second_field_direction",
        expected_shape=first.shape,
    )
    response_second = _validated_block(
        density_response.jvp(second),
        name="density response to second direction",
        expected_shape=first.shape,
    )
    response_first = _validated_block(
        density_response.jvp(first),
        name="density response to first direction",
        expected_shape=first.shape,
    )
    forward = pairing.pair(response_second, first)
    reverse = pairing.pair(response_first, second)
    absolute = abs(forward - reverse)
    scale = abs(forward) + abs(reverse)
    relative = 0.0 if scale <= _RELATIVE_FLOOR else absolute / scale
    return ResponseReciprocityDiagnostic(
        forward_pairing_ev=forward,
        reverse_pairing_ev=reverse,
        absolute_error_ev=absolute,
        relative_error=relative,
    )


@dataclass(frozen=True)
class ResponseStabilityDiagnostic:
    """Symmetrized susceptibility spectrum in explicit scaled coordinates."""

    dimension: int
    potential_direction_scale_ev: float
    gradient_direction_scale_ev_per_angstrom: float
    minimum_eigenvalue_ev: float
    maximum_eigenvalue_ev: float
    positive_eigenvalue_count: int
    negative_eigenvalue_count: int
    near_zero_eigenvalue_count: int
    passivity_violation_ev: float
    antisymmetric_frobenius_norm_ev: float
    stable_sign_convention: str = "nonpositive"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Response stability requires positive dimension.")
        if (
            self.potential_direction_scale_ev <= 0.0
            or self.gradient_direction_scale_ev_per_angstrom <= 0.0
        ):
            raise ValueError("Response direction scales must be positive.")
        finite = (
            self.minimum_eigenvalue_ev,
            self.maximum_eigenvalue_ev,
            self.passivity_violation_ev,
            self.antisymmetric_frobenius_norm_ev,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("Response-stability metrics must be finite.")
        if (
            self.positive_eigenvalue_count < 0
            or self.negative_eigenvalue_count < 0
            or self.near_zero_eigenvalue_count < 0
            or (
                self.positive_eigenvalue_count
                + self.negative_eigenvalue_count
                + self.near_zero_eigenvalue_count
            )
            != self.dimension
        ):
            raise ValueError("Invalid response-stability eigenvalue counts.")
        if (
            self.passivity_violation_ev < 0.0
            or self.antisymmetric_frobenius_norm_ev < 0.0
        ):
            raise ValueError(
                "Response-stability violations must be nonnegative."
            )
        if self.stable_sign_convention != "nonpositive":
            raise ValueError("Unsupported response-stability sign convention.")


def response_stability_diagnostic(
    density_response: DensityResponseLinearization,
    *,
    atom_count: int,
    potential_direction_scale_ev: float = 1.0,
    gradient_direction_scale_ev_per_angstrom: float = 1.0,
    eigenvalue_tolerance_ev: float = 1.0e-10,
    maximum_dimension: int = 256,
) -> ResponseStabilityDiagnostic:
    """Build a small JVP-only susceptibility discriminator.

    The scaled matrix is the work-conjugate response

    ``K = S Q.T J_M S``

    where ``S`` defines explicit potential and gradient direction scales.
    Stable minimization of ``F(c)+<c,f>`` implies a nonpositive symmetric
    susceptibility.  The inertia of a truly symmetric response is unchanged
    by positive diagonal rescaling; the reported eigenvalue magnitudes remain
    scale-dependent and are therefore accompanied by the two scales.

    This dense discriminator is intentionally limited to small canaries.  It
    forms columns only through the public JVP and never materializes an
    autograd Jacobian.
    """

    if atom_count <= 0:
        raise ValueError("Response stability requires at least one atom.")
    if potential_direction_scale_ev <= 0.0:
        raise ValueError("Potential direction scale must be positive.")
    if gradient_direction_scale_ev_per_angstrom <= 0.0:
        raise ValueError("Gradient direction scale must be positive.")
    if eigenvalue_tolerance_ev <= 0.0:
        raise ValueError("Eigenvalue tolerance must be positive.")
    dimension = 4 * int(atom_count)
    if maximum_dimension <= 0 or dimension > maximum_dimension:
        raise ValueError(
            "Response-stability dense discriminator exceeds its declared "
            f"small-system limit ({dimension} > {maximum_dimension})."
        )

    scales = np.tile(
        np.asarray(
            [
                potential_direction_scale_ev,
                gradient_direction_scale_ev_per_angstrom,
                gradient_direction_scale_ev_per_angstrom,
                gradient_direction_scale_ev_per_angstrom,
            ],
            dtype=float,
        ),
        atom_count,
    )
    matrix = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        direction = np.zeros((atom_count, 4), dtype=float)
        direction.reshape(-1)[column] = scales[column]
        density_direction = _validated_block(
            density_response.jvp(direction),
            name="density-response stability JVP",
            expected_shape=(atom_count, 4),
        )
        work_dual = density_to_external_field_order(
            density_direction
        ).reshape(-1)
        matrix[:, column] = scales * work_dual

    symmetric = 0.5 * (matrix + matrix.T)
    antisymmetric = 0.5 * (matrix - matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    positive = int(np.count_nonzero(eigenvalues > eigenvalue_tolerance_ev))
    negative = int(np.count_nonzero(eigenvalues < -eigenvalue_tolerance_ev))
    near_zero = int(eigenvalues.size - positive - negative)
    maximum = float(eigenvalues[-1])
    return ResponseStabilityDiagnostic(
        dimension=dimension,
        potential_direction_scale_ev=potential_direction_scale_ev,
        gradient_direction_scale_ev_per_angstrom=(
            gradient_direction_scale_ev_per_angstrom
        ),
        minimum_eigenvalue_ev=float(eigenvalues[0]),
        maximum_eigenvalue_ev=maximum,
        positive_eigenvalue_count=positive,
        negative_eigenvalue_count=negative,
        near_zero_eigenvalue_count=near_zero,
        passivity_violation_ev=max(0.0, maximum),
        antisymmetric_frobenius_norm_ev=float(
            np.linalg.norm(antisymmetric, ord="fro")
        ),
    )


@dataclass(frozen=True)
class MixingIterationDiagnostic:
    """Local convergence metrics for one damped Picard iteration matrix."""

    mixing: float
    spectral_radius: float
    largest_singular_value: float
    locally_contracting_by_eigenvalues: bool
    contractive_in_euclidean_2_norm: bool

    def __post_init__(self) -> None:
        if not 0.0 < self.mixing <= 1.0:
            raise ValueError("Feedback-spectrum mixing must lie in (0, 1].")
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in (
                self.spectral_radius,
                self.largest_singular_value,
            )
        ):
            raise ValueError(
                "Mixing iteration metrics must be finite and nonnegative."
            )


@dataclass(frozen=True)
class ResidualOperatorConditionDiagnostic:
    """Condition proxy for the physical residual Jacobian ``I-J_F``."""

    largest_singular_value: float
    smallest_singular_value: float
    condition_number_2: float | None
    inverse_norm_2: float | None
    numerically_singular: bool

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in (
                self.largest_singular_value,
                self.smallest_singular_value,
            )
        ):
            raise ValueError(
                "Residual singular values must be finite and nonnegative."
            )
        optional = (self.condition_number_2, self.inverse_norm_2)
        if self.numerically_singular:
            if any(value is not None for value in optional):
                raise ValueError(
                    "A numerically singular residual must not report finite "
                    "condition or inverse norms."
                )
        elif not all(
            value is not None
            and np.isfinite(value)
            and value >= 0.0
            for value in optional
        ):
            raise ValueError(
                "A nonsingular residual requires finite condition and "
                "inverse norms."
            )


@dataclass(frozen=True)
class FixedPointFeedbackSpectrumDiagnostic:
    """Complete small-system spectrum of ``J_F=J_M J_P``."""

    dimension: int
    spectral_radius: float
    maximum_real_eigenvalue: float
    minimum_real_eigenvalue: float
    maximum_absolute_imaginary_eigenvalue: float
    largest_singular_value: float
    smallest_singular_value: float
    symmetric_frobenius_norm: float
    antisymmetric_frobenius_norm: float
    antisymmetric_to_symmetric_frobenius_ratio: float | None
    residual_operator_i_minus_feedback: (
        ResidualOperatorConditionDiagnostic
    )
    mixing_iteration_matrices: tuple[MixingIterationDiagnostic, ...]
    eigenvalues_real: tuple[float, ...]
    eigenvalues_imaginary: tuple[float, ...]
    operator: str = (
        "Pi J_M J_P restricted to neutral density coordinates"
    )
    interpretation: str = (
        "Spectral radii diagnose local linear convergence. Singular values "
        "and the antisymmetric norm expose non-normal transient "
        "amplification. These are solver/feedback diagnostics, not a "
        "thermodynamic passivity proof."
    )

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(
                "Fixed-point feedback spectrum requires positive dimension."
            )
        nonnegative = (
            self.spectral_radius,
            self.maximum_absolute_imaginary_eigenvalue,
            self.largest_singular_value,
            self.smallest_singular_value,
            self.symmetric_frobenius_norm,
            self.antisymmetric_frobenius_norm,
        )
        signed = (
            self.maximum_real_eigenvalue,
            self.minimum_real_eigenvalue,
        )
        ratio = self.antisymmetric_to_symmetric_frobenius_ratio
        if not all(
            np.isfinite(value) and value >= 0.0 for value in nonnegative
        ) or not all(np.isfinite(value) for value in signed):
            raise ValueError(
                "Feedback-spectrum metrics must be finite with nonnegative "
                "norms."
            )
        if ratio is not None and (
            not np.isfinite(ratio) or ratio < 0.0
        ):
            raise ValueError(
                "The feedback antisymmetric/symmetric ratio must be "
                "nonnegative when defined."
            )
        if (
            len(self.eigenvalues_real) != self.dimension
            or len(self.eigenvalues_imaginary) != self.dimension
            or not all(
                np.isfinite(value)
                for value in (
                    *self.eigenvalues_real,
                    *self.eigenvalues_imaginary,
                )
            )
        ):
            raise ValueError(
                "Feedback-spectrum eigenvalue records must match dimension."
            )
        if not self.mixing_iteration_matrices:
            raise ValueError(
                "Feedback-spectrum diagnostics require mixing records."
            )


def fixed_point_feedback_spectrum_diagnostic(
    residual_linearization: UnmixedResidualLinearization,
    *,
    mixing_values: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    maximum_dimension: int = 256,
    singularity_relative_tolerance: float = 1.0e-14,
) -> FixedPointFeedbackSpectrumDiagnostic:
    """Materialize a bounded neutral-space fixed-point feedback operator.

    The production residual is ``R(c)=Pi[c-F(c)]``.  For a neutral direction,
    one JVP therefore gives ``J_F d = d - J_R d``.  This diagnostic forms the
    complete matrix only for explicitly bounded small canaries; the production
    SCF and adjoint remain matrix-free.
    """

    atom_count = int(getattr(residual_linearization, "atom_count", 0))
    if atom_count <= 0:
        raise ValueError(
            "Feedback-spectrum residual requires a positive atom count."
        )
    if maximum_dimension <= 0:
        raise ValueError(
            "Feedback-spectrum maximum dimension must be positive."
        )
    if singularity_relative_tolerance <= 0.0:
        raise ValueError(
            "Residual singularity tolerance must be positive."
        )
    mixing = tuple(float(value) for value in mixing_values)
    if (
        not mixing
        or len(set(mixing)) != len(mixing)
        or any(
            not np.isfinite(value) or not 0.0 < value <= 1.0
            for value in mixing
        )
    ):
        raise ValueError(
            "Feedback-spectrum mixing values must be unique and lie in "
            "(0, 1]."
        )

    coordinates = NeutralDensityCoordinates(atom_count)
    dimension = coordinates.dimension
    if dimension > maximum_dimension:
        raise ValueError(
            "Feedback-spectrum dense discriminator exceeds its declared "
            f"small-system limit ({dimension} > {maximum_dimension})."
        )
    matrix = np.empty((dimension, dimension), dtype=float)
    identity = np.eye(dimension)
    for column in range(dimension):
        reduced_direction = identity[:, column]
        density_direction = coordinates.expand(reduced_direction)
        residual_direction = _validated_block(
            residual_linearization.jvp(density_direction),
            name="unmixed residual JVP",
            expected_shape=(atom_count, 4),
        )
        feedback_direction = density_direction - residual_direction
        matrix[:, column] = coordinates.reduce(feedback_direction)

    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalue_order = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_eigenvalues = eigenvalues[eigenvalue_order]
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    symmetric = 0.5 * (matrix + matrix.T)
    antisymmetric = 0.5 * (matrix - matrix.T)
    symmetric_norm = float(np.linalg.norm(symmetric, ord="fro"))
    antisymmetric_norm = float(
        np.linalg.norm(antisymmetric, ord="fro")
    )
    ratio_zero_tolerance = (
        16.0
        * np.finfo(float).eps
        * dimension
        * max(1.0, symmetric_norm, antisymmetric_norm)
    )

    residual_matrix = identity - matrix
    residual_singular_values = np.linalg.svd(
        residual_matrix,
        compute_uv=False,
    )
    residual_maximum = float(residual_singular_values[0])
    residual_minimum = float(residual_singular_values[-1])
    singular_threshold = (
        singularity_relative_tolerance
        * max(1.0, residual_maximum)
    )
    residual_is_singular = residual_minimum <= singular_threshold
    residual_condition = ResidualOperatorConditionDiagnostic(
        largest_singular_value=residual_maximum,
        smallest_singular_value=residual_minimum,
        condition_number_2=(
            None
            if residual_is_singular
            else residual_maximum / residual_minimum
        ),
        inverse_norm_2=(
            None if residual_is_singular else 1.0 / residual_minimum
        ),
        numerically_singular=residual_is_singular,
    )

    mixing_records = []
    for value in mixing:
        iteration_matrix = (
            (1.0 - value) * identity + value * matrix
        )
        iteration_eigenvalues = np.linalg.eigvals(iteration_matrix)
        iteration_singular_values = np.linalg.svd(
            iteration_matrix,
            compute_uv=False,
        )
        spectral_radius = float(
            np.max(np.abs(iteration_eigenvalues))
        )
        largest_singular = float(iteration_singular_values[0])
        mixing_records.append(
            MixingIterationDiagnostic(
                mixing=value,
                spectral_radius=spectral_radius,
                largest_singular_value=largest_singular,
                locally_contracting_by_eigenvalues=(
                    spectral_radius < 1.0
                ),
                contractive_in_euclidean_2_norm=(
                    largest_singular < 1.0
                ),
            )
        )

    return FixedPointFeedbackSpectrumDiagnostic(
        dimension=dimension,
        spectral_radius=float(np.max(np.abs(eigenvalues))),
        maximum_real_eigenvalue=float(np.max(eigenvalues.real)),
        minimum_real_eigenvalue=float(np.min(eigenvalues.real)),
        maximum_absolute_imaginary_eigenvalue=float(
            np.max(np.abs(eigenvalues.imag))
        ),
        largest_singular_value=float(singular_values[0]),
        smallest_singular_value=float(singular_values[-1]),
        symmetric_frobenius_norm=symmetric_norm,
        antisymmetric_frobenius_norm=antisymmetric_norm,
        antisymmetric_to_symmetric_frobenius_ratio=(
            None
            if (
                symmetric_norm <= ratio_zero_tolerance
                and antisymmetric_norm > ratio_zero_tolerance
            )
            else (
                0.0
                if symmetric_norm <= ratio_zero_tolerance
                else antisymmetric_norm / symmetric_norm
            )
        ),
        residual_operator_i_minus_feedback=residual_condition,
        mixing_iteration_matrices=tuple(mixing_records),
        eigenvalues_real=tuple(
            float(value.real) for value in sorted_eigenvalues
        ),
        eigenvalues_imaginary=tuple(
            float(value.imag) for value in sorted_eigenvalues
        ),
    )


@dataclass(frozen=True)
class FieldLoopWorkDiagnostic:
    """Trapezoidal work around one four-edge field-space rectangle."""

    edge_work_ev: tuple[float, float, float, float]
    closed_loop_work_ev: float
    absolute_edge_work_ev: float
    relative_closed_loop_work: float
    quadrature: str = "endpoint-trapezoid-v1"

    def __post_init__(self) -> None:
        if len(self.edge_work_ev) != 4 or not all(
            np.isfinite(value) for value in self.edge_work_ev
        ):
            raise ValueError("Field-loop diagnostic requires four finite edges.")
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in (
                self.absolute_edge_work_ev,
                self.relative_closed_loop_work,
            )
        ) or not np.isfinite(self.closed_loop_work_ev):
            raise ValueError("Field-loop work metrics must be finite.")
        if self.quadrature != "endpoint-trapezoid-v1":
            raise ValueError("Unsupported field-loop quadrature.")


def field_loop_work_diagnostic(
    density_evaluator: Callable[[np.ndarray], np.ndarray],
    *,
    base_field: np.ndarray,
    first_field_step: np.ndarray,
    second_field_step: np.ndarray,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
) -> FieldLoopWorkDiagnostic:
    """Integrate ``<c(f),df>`` around a small rectangular field loop."""

    origin = _validated_block(base_field, name="base_field")
    first = _validated_block(
        first_field_step,
        name="first_field_step",
        expected_shape=origin.shape,
    )
    second = _validated_block(
        second_field_step,
        name="second_field_step",
        expected_shape=origin.shape,
    )
    corners = (
        origin,
        origin + first,
        origin + first + second,
        origin + second,
    )
    densities = tuple(
        _validated_block(
            density_evaluator(field),
            name=f"loop density {index}",
            expected_shape=origin.shape,
        )
        for index, field in enumerate(corners)
    )
    field_edges = (first, second, -first, -second)
    density_edges = (
        (densities[0], densities[1]),
        (densities[1], densities[2]),
        (densities[2], densities[3]),
        (densities[3], densities[0]),
    )
    edge_work = tuple(
        0.5
        * (
            pairing.pair(start_density, field_step)
            + pairing.pair(end_density, field_step)
        )
        for (start_density, end_density), field_step in zip(
            density_edges,
            field_edges,
            strict=True,
        )
    )
    closed = float(sum(edge_work))
    absolute = float(sum(abs(value) for value in edge_work))
    relative = 0.0 if absolute <= _RELATIVE_FLOOR else abs(closed) / absolute
    return FieldLoopWorkDiagnostic(
        edge_work_ev=edge_work,
        closed_loop_work_ev=closed,
        absolute_edge_work_ev=absolute,
        relative_closed_loop_work=relative,
    )


__all__ = [
    "BlockConjugacyDefect",
    "EnergyDensityConjugacyDiagnostic",
    "FieldLoopWorkDiagnostic",
    "FixedPointFeedbackSpectrumDiagnostic",
    "IntrinsicFeatureConjugacyDiagnostic",
    "MixingIterationDiagnostic",
    "RelativeVectorDefect",
    "ResidualOperatorConditionDiagnostic",
    "ResponseReciprocityDiagnostic",
    "ResponseStabilityDiagnostic",
    "energy_density_conjugacy_diagnostic",
    "field_loop_work_diagnostic",
    "fixed_point_feedback_spectrum_diagnostic",
    "intrinsic_feature_conjugacy_diagnostic",
    "response_reciprocity_diagnostic",
    "response_stability_diagnostic",
]
