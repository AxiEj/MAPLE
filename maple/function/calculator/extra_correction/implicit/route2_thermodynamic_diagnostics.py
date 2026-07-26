"""Thermodynamic diagnostics for the learned Route-2 field response.

These diagnostics do not change the fixed-point equation or the scalar energy.
They test whether the learned density response and the reported field-dependent
energy are compatible with either of two candidate physical identities:

``intrinsic``
    The reported energy is an internal functional evaluated at the
    field-relaxed density.

``coupled``
    The reported energy already includes the full external-field coupling.

The module operates only on the differentiable local-field interface currently
exposed by MACE-POLAR.  It must not be used to claim that the exact-GTO
energy-only profile has a validated response adjoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
    ElectrostaticPairing,
)
from .gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from .route2_response import DensityResponseLinearization


_RELATIVE_FLOOR = 1.0e-30


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
    "ResponseReciprocityDiagnostic",
    "ResponseStabilityDiagnostic",
    "energy_density_conjugacy_diagnostic",
    "field_loop_work_diagnostic",
    "response_reciprocity_diagnostic",
    "response_stability_diagnostic",
]
