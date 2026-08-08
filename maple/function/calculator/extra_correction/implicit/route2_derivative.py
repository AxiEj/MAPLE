"""Energy-gradient ingredients for the Route-2 fixed-point adjoint.

This module owns the neutral density-space right-hand side, the diagnostic
fixed-surface coordinate-gradient slice, the contract for differentiating one
complete continuum reaction-field map, and an internal bookkeeping boundary
for adding a separately validated CDS gradient.  Its complete derivative is
used only by the explicit, non-default experimental single-point force
candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, cast

import numpy as np
from ase.units import Hartree

from .electrostatic_pairing import ElectrostaticPairing, MACE_POLAR_L1_PAIRING
from .route2_feature_response import ModelFeatureLinearMap
from .route2_response import (
    DensityResponseLinearization,
    ReactionFieldLinearMap,
    project_neutral_density_tangent,
)


FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION = 1


class FixedSurfaceReactionField(ReactionFieldLinearMap, Protocol):
    """Fixed-surface reaction map with a nuclear-coordinate pairing VJP."""

    atom_count: int
    reciprocal_energy_pairing: bool

    def position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate ``<field_cotangent, P_R density>`` at fixed surface."""
        ...


class FullReactionFieldPositionDerivative(
    ReactionFieldLinearMap,
    Protocol,
):
    """Reaction map with the exact coordinate VJP of its energy-path map."""

    atom_count: int
    reciprocal_energy_pairing: bool
    full_position_derivative_contract_version: int

    def full_position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate ``<field_cotangent, P_R density>`` completely.

        The same reaction-field object must own the forward/adjoint maps and
        this derivative.  The result includes solute-MEP projection,
        moving-surface kernels, the continuum response operator, and
        reaction-field back-projection for the exact discrete energy path.
        """
        ...


def _validated_block(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None,
    name: str,
) -> np.ndarray:
    block = np.asarray(values, dtype=float)
    valid_shape = (
        block.ndim == 2
        and block.shape[1] == 4
        and block.shape[0] > 0
        and (expected_shape is None or block.shape == expected_shape)
    )
    if not valid_shape or not np.all(np.isfinite(block)):
        shape = "(n_atoms, 4)" if expected_shape is None else str(expected_shape)
        raise ValueError(
            f"{name} must be finite with shape {shape}; received {block.shape}."
        )
    return block


def _validated_coordinate_block(
    values: np.ndarray,
    *,
    atom_count: int | None,
    name: str,
) -> np.ndarray:
    block = np.asarray(values, dtype=float)
    expected_shape = None if atom_count is None else (atom_count, 3)
    valid_shape = (
        block.ndim == 2
        and block.shape[0] > 0
        and block.shape[1] == 3
        and (expected_shape is None or block.shape == expected_shape)
    )
    if not valid_shape or not np.all(np.isfinite(block)):
        shape = "(n_atoms, 3)" if expected_shape is None else str(expected_shape)
        raise ValueError(
            f"{name} must be finite with shape {shape}; "
            f"received {block.shape}."
        )
    return block


@dataclass(frozen=True)
class TotalSolvationCoordinateGradient:
    """Component-resolved total solvation gradient and correction force.

    Both stored component gradients use hartree/angstrom.  The continuum
    component is already the derivative of the solvation correction
    ``E_intrinsic(solvent)-E_gas+E_PCM``; therefore the correction force below
    must be added exactly once to the independently returned gas-phase force.
    """

    continuum_position_gradient_hartree_per_angstrom: np.ndarray
    cds_position_gradient_hartree_per_angstrom: np.ndarray
    total_position_gradient_hartree_per_angstrom: np.ndarray = field(
        init=False
    )
    solvent_correction_forces_hartree_per_angstrom: np.ndarray = field(
        init=False
    )

    def __post_init__(self) -> None:
        continuum = _validated_coordinate_block(
            self.continuum_position_gradient_hartree_per_angstrom,
            atom_count=None,
            name="continuum_position_gradient_hartree_per_angstrom",
        ).copy()
        cds = _validated_coordinate_block(
            self.cds_position_gradient_hartree_per_angstrom,
            atom_count=continuum.shape[0],
            name="cds_position_gradient_hartree_per_angstrom",
        ).copy()
        total = continuum + cds
        solvent_forces = -total
        for values in (continuum, cds, total, solvent_forces):
            values.setflags(write=False)
        object.__setattr__(
            self,
            "continuum_position_gradient_hartree_per_angstrom",
            continuum,
        )
        object.__setattr__(
            self,
            "cds_position_gradient_hartree_per_angstrom",
            cds,
        )
        object.__setattr__(
            self,
            "total_position_gradient_hartree_per_angstrom",
            total,
        )
        object.__setattr__(
            self,
            "solvent_correction_forces_hartree_per_angstrom",
            solvent_forces,
        )


def assemble_total_solvation_coordinate_gradient(
    continuum_position_gradient_ev_per_angstrom: np.ndarray,
    cds_position_gradient_hartree_per_angstrom: np.ndarray,
) -> TotalSolvationCoordinateGradient:
    """Add continuum and CDS position gradients with explicit unit conversion.

    This provider-neutral research boundary accepts only solvent-correction
    components.  It intentionally has no gas-force argument: CalcABC's
    finalizer owns the single addition of the gas-phase force.
    """

    continuum_ev = _validated_coordinate_block(
        continuum_position_gradient_ev_per_angstrom,
        atom_count=None,
        name="continuum_position_gradient_ev_per_angstrom",
    )
    cds_hartree = _validated_coordinate_block(
        cds_position_gradient_hartree_per_angstrom,
        atom_count=continuum_ev.shape[0],
        name="cds_position_gradient_hartree_per_angstrom",
    )
    return TotalSolvationCoordinateGradient(
        continuum_position_gradient_hartree_per_angstrom=(
            continuum_ev / Hartree
        ),
        cds_position_gradient_hartree_per_angstrom=cds_hartree,
    )


def fixed_cavity_energy_density_gradient(
    reaction_field: ReactionFieldLinearMap,
    *,
    reaction_field_values: np.ndarray,
    intrinsic_energy_field_gradient: np.ndarray,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
) -> np.ndarray:
    """Return the physical fixed-cavity energy gradient in neutral density space.

    At fixed geometry/cavity let ``f=P c`` and

    ``E(c) = E_model,conditioned(f) + 0.5 c.T Q f``.

    For the reciprocal ``MATRIXSYMM=TRUE`` response used by Route 2,
    differentiating the half-coupling gives the full ``Q f`` term, so

    ``dE/dc = P.T g_f + Q f``

    with ``g_f=dE_model,conditioned/df``.  The returned right-hand side is its
    orthogonal projection into the zero-total-monopole tangent space.  CDS and
    all coordinate/cavity derivatives are outside this fixed-cavity quantity.
    """

    if getattr(reaction_field, "reciprocal_energy_pairing", False) is not True:
        raise ValueError(
            "The fixed-cavity energy gradient requires a reciprocal reaction "
            "field with MATRIXSYMM=TRUE."
        )
    field = _validated_block(
        reaction_field_values,
        expected_shape=None,
        name="reaction_field_values",
    )
    field_gradient = _validated_block(
        intrinsic_energy_field_gradient,
        expected_shape=field.shape,
        name="intrinsic_energy_field_gradient",
    )
    model_energy_chain = _validated_block(
        reaction_field.adjoint(field_gradient),
        expected_shape=field.shape,
        name="reaction-field energy VJP",
    )
    polarization_gradient = pairing.field_to_density_order(field)
    return project_neutral_density_tangent(
        model_energy_chain + polarization_gradient
    )


def pcm_half_coupling_energy_density_gradient(
    reaction_field: ReactionFieldLinearMap,
    *,
    reaction_field_values: np.ndarray,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
) -> np.ndarray:
    """Return the direct-PCM ledger derivative in the neutral density space.

    For the explicitly selected direct ledger

    ``E_0(c, R) = 0.5 * <c, P_R c>``,

    reciprocity gives ``dE_0/dc = P_R c``.  The adapter-provided fixed point
    still supplies ``c`` and is differentiated through the outer adjoint
    later, but the field-conditioned model energy and its field derivative do
    *not* enter this right-hand side.  This separation prevents silently
    reusing the legacy operational-energy adjoint for a different scalar
    ledger.
    """

    if getattr(reaction_field, "reciprocal_energy_pairing", False) is not True:
        raise ValueError(
            "The PCM half-coupling density gradient requires a reciprocal "
            "reaction field energy pairing."
        )
    field = _validated_block(
        reaction_field_values,
        expected_shape=None,
        name="reaction_field_values",
    )
    if reaction_field.atom_count != field.shape[0]:
        raise ValueError(
            "Reaction-field atom count does not match the energy-dual field."
        )
    return project_neutral_density_tangent(pairing.field_to_density_order(field))


def pcm_half_coupling_source_gradient(
    reaction_field: ReactionFieldLinearMap,
    *,
    source: np.ndarray,
    field: np.ndarray,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
) -> np.ndarray:
    """Differentiate ``0.5*<c, f(c)>`` for static or source-dependent maps.

    A fixed-cavity reciprocal map retains the existing fast path
    ``grad_c E = pairing(f)``.  If the provider declares
    ``source_dependent_geometry = True``, the reaction map is nonlinear and
    the correct local derivative is instead

    ``0.5*pairing(f) + 0.5*J_f(c).T*pairing(c)``.

    The nonlinear provider must prove that its cached JVP/VJP state belongs to
    the exact ``source`` and ``field`` supplied here.  This prevents a stale
    cavity from silently entering the direct PCM ledger.
    """

    source_values = _validated_block(
        source,
        expected_shape=None,
        name="PCM half-coupling source",
    )
    field_values = _validated_block(
        field,
        expected_shape=source_values.shape,
        name="PCM half-coupling field",
    )
    if reaction_field.atom_count != source_values.shape[0]:
        raise ValueError(
            "Reaction-field atom count does not match the half-coupling source."
        )
    if getattr(reaction_field, "reciprocal_energy_pairing", False) is not True:
        raise ValueError(
            "The PCM half-coupling source gradient requires a reciprocal "
            "reaction-field energy pairing."
        )
    if getattr(reaction_field, "source_dependent_geometry", False) is not True:
        return pcm_half_coupling_energy_density_gradient(
            reaction_field,
            reaction_field_values=field_values,
            pairing=pairing,
        )

    validate_state = getattr(reaction_field, "validate_linearization_state", None)
    if not callable(validate_state):
        raise TypeError(
            "A source-dependent reaction field must validate its exact cached "
            "source/field linearization state."
        )
    validate_state(source_values, field_values)
    direct = 0.5 * pairing.field_to_density_order(field_values)
    field_cotangent = pairing.density_to_field_order(source_values)
    response = 0.5 * _validated_block(
        reaction_field.adjoint(field_cotangent),
        expected_shape=source_values.shape,
        name="nonlinear PCM half-coupling source response",
    )
    return project_neutral_density_tangent(direct + response)


def fixed_cavity_model_feature_energy_density_gradient(
    reaction_field: ModelFeatureLinearMap,
    *,
    reaction_field_values: np.ndarray,
    intrinsic_energy_feature_gradient: np.ndarray,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
) -> np.ndarray:
    """Return ``J_z.T g_z + Q f`` in the neutral density tangent space.

    The energy-dual field ``f`` remains the point-l1 continuum reaction field
    used by the half-coupling ledger.  The field-conditioned model energy is
    driven by a separate native feature tensor ``z(c)``.  This fixed-geometry
    right-hand side composes only their exact discrete derivatives; CDS and
    every coordinate/cavity derivative remain outside it.
    """

    if getattr(reaction_field, "reciprocal_energy_pairing", False) is not True:
        raise ValueError(
            "The fixed-cavity energy gradient requires a reciprocal reaction "
            "field energy pairing."
        )
    field = _validated_block(
        reaction_field_values,
        expected_shape=None,
        name="reaction_field_values",
    )
    if reaction_field.atom_count != field.shape[0]:
        raise ValueError(
            "Reaction-field atom count does not match the energy-dual field."
        )
    feature_count = reaction_field.model_feature_count
    if not isinstance(feature_count, (int, np.integer)) or feature_count <= 0:
        raise ValueError("Reaction-field model-feature count must be positive.")
    feature_count = int(feature_count)
    feature_gradient = np.asarray(
        intrinsic_energy_feature_gradient,
        dtype=float,
    )
    expected_feature_shape = (field.shape[0], feature_count)
    if (
        feature_gradient.shape != expected_feature_shape
        or not np.all(np.isfinite(feature_gradient))
    ):
        raise ValueError(
            "intrinsic_energy_feature_gradient must be finite with shape "
            f"{expected_feature_shape}; received {feature_gradient.shape}."
        )
    model_energy_chain = _validated_block(
        reaction_field.model_feature_vjp(feature_gradient),
        expected_shape=field.shape,
        name="model-feature energy VJP",
    )
    polarization_gradient = pairing.field_to_density_order(field)
    return project_neutral_density_tangent(
        model_energy_chain + polarization_gradient
    )


def _coupled_solvation_coordinate_gradient(
    reaction_field: ReactionFieldLinearMap,
    density_response: DensityResponseLinearization,
    *,
    reaction_position_vjp: Callable[
        [np.ndarray, np.ndarray],
        np.ndarray,
    ],
    reaction_position_vjp_name: str,
    reciprocity_error: str,
    result_name: str,
    density_coefficients: np.ndarray,
    intrinsic_energy_field_gradient: np.ndarray,
    adjoint_solution: np.ndarray,
    adjoint_density_position_vjp: np.ndarray,
    solvent_fixed_field_forces_ev_per_angstrom: np.ndarray,
    gas_forces_ev_per_angstrom: np.ndarray,
    pairing: ElectrostaticPairing,
    neutral_tolerance: float = 1.0e-10,
) -> np.ndarray:
    if neutral_tolerance <= 0.0:
        raise ValueError("Neutral tangent tolerance must be positive.")
    if getattr(reaction_field, "reciprocal_energy_pairing", False) is not True:
        raise ValueError(reciprocity_error)

    density = _validated_block(
        density_coefficients,
        expected_shape=None,
        name="density_coefficients",
    )
    atom_count = density.shape[0]
    if reaction_field.atom_count != atom_count:
        raise ValueError(
            "Reaction-field atom count does not match density coefficients "
            f"({reaction_field.atom_count} != {atom_count})."
        )
    field_gradient = _validated_block(
        intrinsic_energy_field_gradient,
        expected_shape=density.shape,
        name="intrinsic_energy_field_gradient",
    )
    adjoint = _validated_block(
        adjoint_solution,
        expected_shape=density.shape,
        name="adjoint_solution",
    )
    charge_sum = abs(float(np.sum(adjoint[:, 0])))
    charge_scale = max(1.0, float(np.linalg.norm(adjoint[:, 0], ord=1)))
    if charge_sum > neutral_tolerance * charge_scale:
        raise ValueError(
            "adjoint_solution must lie in the neutral density tangent space "
            f"(monopole sum={float(np.sum(adjoint[:, 0])):.6e})."
        )

    response_field_cotangent = _validated_block(
        density_response.vjp(adjoint),
        expected_shape=density.shape,
        name="Electronic-source response field VJP",
    )
    half_coupling_field_cotangent = (
        0.5 * pairing.density_to_field_order(density)
    )
    combined_field_cotangent = (
        field_gradient
        + half_coupling_field_cotangent
        + response_field_cotangent
    )
    reaction_position_gradient = _validated_coordinate_block(
        reaction_position_vjp(
            density,
            combined_field_cotangent,
        ),
        atom_count=atom_count,
        name=reaction_position_vjp_name,
    )
    density_position_gradient = _validated_coordinate_block(
        adjoint_density_position_vjp,
        atom_count=atom_count,
        name="adjoint_density_position_vjp",
    )
    solvent_forces = _validated_coordinate_block(
        solvent_fixed_field_forces_ev_per_angstrom,
        atom_count=atom_count,
        name="solvent_fixed_field_forces_ev_per_angstrom",
    )
    gas_forces = _validated_coordinate_block(
        gas_forces_ev_per_angstrom,
        atom_count=atom_count,
        name="gas_forces_ev_per_angstrom",
    )

    # Forces are negative coordinate gradients.  Therefore the direct
    # derivative of E_intrinsic(solvent)-E_gas is F_gas-F_solvent.
    result = (
        gas_forces
        - solvent_forces
        + reaction_position_gradient
        + density_position_gradient
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{result_name} is non-finite.")
    return result


def fixed_surface_solvation_coordinate_gradient(
    reaction_field: FixedSurfaceReactionField,
    density_response: DensityResponseLinearization,
    *,
    density_coefficients: np.ndarray,
    intrinsic_energy_field_gradient: np.ndarray,
    adjoint_solution: np.ndarray,
    adjoint_density_position_vjp: np.ndarray,
    solvent_fixed_field_forces_ev_per_angstrom: np.ndarray,
    gas_forces_ev_per_angstrom: np.ndarray,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
    neutral_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Compose the fixed-surface coupled solvation-energy coordinate gradient.

    With ``f=P_R c``, residual
    ``r=Pi0[c-M(R,f)]``, and adjoint
    ``J_c r.T lambda = Pi0[dE/dc]``, this returns

    ``d(E_intrinsic(R,f)-E_gas(R)+0.5*c.T*Q*f)/dR``

    for the fixed-surface/operator derivative slice.  The three field
    cotangents entering the single ``P_R`` position VJP are:

    - ``dE_intrinsic/df``;
    - ``0.5*Q.T*c`` from the PCM half-coupling;
    - ``J_M.T*lambda`` from the implicit density response.

    ``adjoint_density_position_vjp`` must be
    ``(dM/dR|f).T*lambda`` with the supplied atom-indexed field samples held
    fixed.  The returned value is an energy gradient in eV/Angstrom, not a
    force.  Cavity/operator motion and SMD CDS derivatives are absent.
    """

    return _coupled_solvation_coordinate_gradient(
        reaction_field,
        density_response,
        reaction_position_vjp=reaction_field.position_vjp,
        reaction_position_vjp_name=(
            "fixed-surface reaction-field position VJP"
        ),
        reciprocity_error=(
            "The fixed-surface coordinate gradient requires a reciprocal "
            "reaction field with MATRIXSYMM=TRUE."
        ),
        result_name="Fixed-surface solvation coordinate gradient",
        density_coefficients=density_coefficients,
        intrinsic_energy_field_gradient=intrinsic_energy_field_gradient,
        adjoint_solution=adjoint_solution,
        adjoint_density_position_vjp=adjoint_density_position_vjp,
        solvent_fixed_field_forces_ev_per_angstrom=(
            solvent_fixed_field_forces_ev_per_angstrom
        ),
        gas_forces_ev_per_angstrom=gas_forces_ev_per_angstrom,
        pairing=pairing,
        neutral_tolerance=neutral_tolerance,
    )


def continuum_coupled_solvation_coordinate_gradient(
    reaction_field: FullReactionFieldPositionDerivative,
    density_response: DensityResponseLinearization,
    *,
    density_coefficients: np.ndarray,
    intrinsic_energy_field_gradient: np.ndarray,
    adjoint_solution: np.ndarray,
    adjoint_density_position_vjp: np.ndarray,
    solvent_fixed_field_forces_ev_per_angstrom: np.ndarray,
    gas_forces_ev_per_angstrom: np.ndarray,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
    neutral_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Compose the coupled gradient with the full continuum-map derivative.

    This uses the same combined adjoint cotangent as the fixed-surface
    diagnostic but contracts it through ``reaction_field.full_position_vjp``.
    That method must differentiate the exact same forward/adjoint reaction map,
    including moving surface and continuum-operator response.  Passing
    separately computed derivative arrays is intentionally unsupported.

    The result is the continuum-electrostatic coupled energy gradient in
    eV/Angstrom.  SMD CDS is absent, so this is not a total solvent force and
    does not make Route 2 PES-capable.
    """

    derivative_version = getattr(
        reaction_field,
        "full_position_derivative_contract_version",
        None,
    )
    if derivative_version is None:
        raise NotImplementedError(
            "The reaction-field backend does not provide a full reaction-field "
            "coordinate derivative."
        )
    if derivative_version != (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    ):
        raise ValueError(
            "Unsupported full reaction-field coordinate-derivative contract "
            "version."
        )
    implementation = getattr(reaction_field, "full_position_vjp", None)
    if not callable(implementation):
        raise NotImplementedError(
            "The reaction-field backend does not provide a full reaction-field "
            "coordinate derivative."
        )
    typed_implementation = cast(
        Callable[[np.ndarray, np.ndarray], np.ndarray],
        implementation,
    )

    return _coupled_solvation_coordinate_gradient(
        reaction_field,
        density_response,
        reaction_position_vjp=typed_implementation,
        reaction_position_vjp_name="full reaction-field position VJP",
        reciprocity_error=(
            "The continuum-coupled coordinate gradient requires a reciprocal "
            "reaction-field energy pairing."
        ),
        result_name="Continuum-coupled solvation coordinate gradient",
        density_coefficients=density_coefficients,
        intrinsic_energy_field_gradient=intrinsic_energy_field_gradient,
        adjoint_solution=adjoint_solution,
        adjoint_density_position_vjp=adjoint_density_position_vjp,
        solvent_fixed_field_forces_ev_per_angstrom=(
            solvent_fixed_field_forces_ev_per_angstrom
        ),
        gas_forces_ev_per_angstrom=gas_forces_ev_per_angstrom,
        pairing=pairing,
        neutral_tolerance=neutral_tolerance,
    )


def pcm_half_coupling_continuum_coordinate_gradient(
    reaction_field: FullReactionFieldPositionDerivative,
    density_response: DensityResponseLinearization,
    *,
    density_coefficients: np.ndarray,
    adjoint_solution: np.ndarray,
    adjoint_density_position_vjp: np.ndarray,
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING,
    neutral_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Differentiate the direct-PCM leaf ledger through the model fixed point.

    This is the ledger-specific counterpart of
    :func:`continuum_coupled_solvation_coordinate_gradient`.  It evaluates

    ``d[0.5<c, P_R c>]/dR``

    at a converged electronic-model fixed point using an outer adjoint.  Its
    combined field cotangent is exactly

    ``0.5 * D.T c + J_M.T lambda``.

    The legacy field-conditioned model energy, its field gradient, and its
    fixed-field coordinate partial are intentionally absent.  ``CDS`` also
    remains a separately supplied same-profile scalar/gradient component.
    """

    density = _validated_block(
        density_coefficients,
        expected_shape=None,
        name="density_coefficients",
    )
    zero_field_gradient = np.zeros_like(density)
    zero_coordinates = np.zeros((density.shape[0], 3), dtype=float)
    return continuum_coupled_solvation_coordinate_gradient(
        reaction_field,
        density_response,
        density_coefficients=density,
        intrinsic_energy_field_gradient=zero_field_gradient,
        adjoint_solution=adjoint_solution,
        adjoint_density_position_vjp=adjoint_density_position_vjp,
        solvent_fixed_field_forces_ev_per_angstrom=zero_coordinates,
        gas_forces_ev_per_angstrom=zero_coordinates,
        pairing=pairing,
        neutral_tolerance=neutral_tolerance,
    )


__all__ = [
    "FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION",
    "FixedSurfaceReactionField",
    "FullReactionFieldPositionDerivative",
    "TotalSolvationCoordinateGradient",
    "assemble_total_solvation_coordinate_gradient",
    "continuum_coupled_solvation_coordinate_gradient",
    "fixed_cavity_energy_density_gradient",
    "fixed_cavity_model_feature_energy_density_gradient",
    "fixed_surface_solvation_coordinate_gradient",
    "pcm_half_coupling_continuum_coordinate_gradient",
    "pcm_half_coupling_energy_density_gradient",
    "pcm_half_coupling_source_gradient",
]
