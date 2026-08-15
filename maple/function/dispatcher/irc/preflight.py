"""Shared first-order-saddle preflight for legacy-unit IRC integrators."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from ase import Atoms

from ...calculator.calculator_base import HARTREE2EV
from ..frequency.domain import validate_molecular_vibrational_domain
from ..frequency.normal_modes import (
    analyze_cartesian_hessian,
    rigid_body_hessian_residual_cm1,
)
from ..frequency.stationary_points import (
    StationaryPointAssessment,
    assess_stationary_point,
)


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return converted


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return converted


@dataclass
class IRCPreflightParams:
    """Common scientific admission settings for every IRC integrator."""

    target_mode: int = 1
    stationarity_tolerance_ev_per_a: float = 1.0e-3
    hessian_symmetry_relative_tolerance: float = 1.0e-6
    rigid_mode_tolerance_cm1: float = 5.0
    transition_state_imaginary_threshold_cm1: float = 50.0


@dataclass(frozen=True, slots=True)
class IRCTransitionStatePreflight:
    """Validated IRC starting Hessian and its unique downhill mode."""

    assessment: StationaryPointAssessment
    negative_mode_mass_weighted: np.ndarray
    negative_eigenvalue_hartree_per_A2_amu: float
    maximum_force_eV_per_A: float
    hessian_symmetry_max_abs_eV_per_A2: float
    rigid_residual_cm1: float


def _validated_settings(
    params: IRCPreflightParams,
) -> tuple[float, float, float, float]:
    if type(params.target_mode) is not int or params.target_mode != 1:
        raise ValueError(
            "IRC first-order-saddle admission requires target_mode=1; higher "
            "negative modes describe a different or higher-order saddle contract."
        )
    return (
        _positive_float(
            params.stationarity_tolerance_ev_per_a,
            "stationarity_tolerance_ev_per_a",
        ),
        _nonnegative_float(
            params.hessian_symmetry_relative_tolerance,
            "hessian_symmetry_relative_tolerance",
        ),
        _nonnegative_float(
            params.rigid_mode_tolerance_cm1,
            "rigid_mode_tolerance_cm1",
        ),
        _positive_float(
            params.transition_state_imaginary_threshold_cm1,
            "transition_state_imaginary_threshold_cm1",
        ),
    )


def validate_irc_preflight_params(params: IRCPreflightParams) -> None:
    """Fail early when shared IRC admission parameters are invalid."""

    _validated_settings(params)


def validate_irc_transition_state(
    atoms: Atoms,
    hessian_hartree_per_A2: object,
    forces_hartree_per_A: object,
    params: IRCPreflightParams,
) -> IRCTransitionStatePreflight:
    """Validate and project an IRC start using the public FREQ mathematics.

    IRC algorithms still operate behind MAPLE's private Hartree job view. This
    boundary converts their initial Hessian and forces back to public ASE units
    exactly once before applying the shared mass-metric stationary-point gates.
    """

    (
        stationarity_tolerance,
        symmetry_relative_tolerance,
        rigid_mode_tolerance,
        imaginary_threshold,
    ) = _validated_settings(params)
    validate_molecular_vibrational_domain(
        atoms,
        operation="IRC first-order-saddle preflight",
    )

    coordinate_count = 3 * len(atoms)
    hessian_hartree = np.asarray(hessian_hartree_per_A2, dtype=float)
    if hessian_hartree.ndim == 3 and hessian_hartree.shape[0] == 1:
        hessian_hartree = hessian_hartree[0]
    expected_hessian_shape = (coordinate_count, coordinate_count)
    if hessian_hartree.shape != expected_hessian_shape or not np.all(
        np.isfinite(hessian_hartree)
    ):
        raise ValueError(
            "IRC Cartesian Hessian must be finite with shape "
            f"{expected_hessian_shape}."
        )

    forces_hartree = np.asarray(forces_hartree_per_A, dtype=float)
    expected_force_shape = (len(atoms), 3)
    if forces_hartree.shape != expected_force_shape or not np.all(
        np.isfinite(forces_hartree)
    ):
        raise ValueError(
            f"IRC forces must be finite with shape {expected_force_shape}."
        )

    forces_eV_per_A = forces_hartree * HARTREE2EV
    maximum_force = float(np.max(np.abs(forces_eV_per_A), initial=0.0))
    if maximum_force > stationarity_tolerance:
        raise ValueError(
            "IRC requires a stationary geometry before path integration: "
            f"maximum |force|={maximum_force:.6g} eV/Angstrom exceeds "
            f"{stationarity_tolerance:.6g} eV/Angstrom."
        )

    hessian_eV_per_A2 = hessian_hartree * HARTREE2EV
    hessian_scale = float(np.max(np.abs(hessian_eV_per_A2), initial=0.0))
    symmetry_tolerance = 1.0e-8 + symmetry_relative_tolerance * hessian_scale
    analysis = analyze_cartesian_hessian(
        hessian_eV_per_A2,
        atoms.get_masses(),
        atoms.get_positions(),
        symmetry_tolerance_eV_per_A2=symmetry_tolerance,
    )
    rigid_residual = rigid_body_hessian_residual_cm1(analysis)
    if rigid_residual > rigid_mode_tolerance:
        raise ValueError(
            "IRC Cartesian Hessian violates stationary rigid-body invariance: "
            f"operator residual is equivalent to {rigid_residual:.6g} cm^-1, "
            f"above {rigid_mode_tolerance:.6g} cm^-1."
        )

    assessment = assess_stationary_point(
        analysis.frequencies_cm1,
        target="transition_state",
        imaginary_threshold_cm1=imaginary_threshold,
    )
    mode_index = assessment.robust_imaginary_mode_indices[0]
    negative_mode = np.array(
        analysis.modes_mass_weighted[:, mode_index],
        copy=True,
    )
    negative_mode.setflags(write=False)
    negative_eigenvalue = float(
        analysis.eigenvalues_eV_per_A2_amu[mode_index] / HARTREE2EV
    )

    return IRCTransitionStatePreflight(
        assessment=assessment,
        negative_mode_mass_weighted=negative_mode,
        negative_eigenvalue_hartree_per_A2_amu=negative_eigenvalue,
        maximum_force_eV_per_A=maximum_force,
        hessian_symmetry_max_abs_eV_per_A2=(
            analysis.hessian_symmetry_max_abs_eV_per_A2
        ),
        rigid_residual_cm1=rigid_residual,
    )


__all__ = [
    "IRCPreflightParams",
    "IRCTransitionStatePreflight",
    "validate_irc_preflight_params",
    "validate_irc_transition_state",
]
