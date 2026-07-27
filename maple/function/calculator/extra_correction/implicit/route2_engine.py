"""Internal orchestration for self-consistent polarizable continuum coupling.

This module contains no continuum-provider construction or public dispatch.
One provider wrapper supplies a same-energy reaction-field factory and a
matching CDS evaluator; the engine owns only the common ML--SCF root,
implicit-function adjoint, and component ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
import math
from typing import Any, TypedDict, cast

import numpy as np
from ase.units import Hartree

from .electrostatic_pairing import MACE_POLAR_L1_PAIRING
from .route2_derivative import (
    assemble_total_solvation_coordinate_gradient,
    continuum_coupled_solvation_coordinate_gradient,
    fixed_cavity_energy_density_gradient,
)
from .route2_field_state import ReactionFieldDrive
from .route2_fixed_point import (
    DAMPED_PICARD_SOLVER,
    SAFEGUARDED_ANDERSON_SOLVER,
    SUPPORTED_FIXED_POINT_SOLVERS,
    FixedPointSample,
    next_fixed_point_density,
    project_density_total_charge,
)
from .route2_response import (
    DensityResponseLinearization,
    UnmixedDensityResidualLinearization,
    solve_adjoint,
    solve_newton_correction,
)

NEAR_ROOT_NEWTON_CORRECTOR = "near-root-newton-gmres-v1"


class Route2SCFHistoryRecord(TypedDict):
    """One complete JSON-serializable fixed-point iteration record."""

    iteration: int
    density_residual_e: float
    energy_residual_ev: float | None
    intrinsic_energy_ev: float
    root_total_charge_e: float
    raw_response_total_charge_e: float
    response_charge_projection_max_e: float
    arrived_by: str | None
    anderson_history_reset: bool
    next_density_update: str | None
    fixed_point_history_size: int | None
    anderson_predicted_residual_l2: float | None
    anderson_coefficient_l1: float | None
    anderson_step_ratio_to_picard: float | None
    anderson_fallback_reason: str | None
    newton_attempted: bool
    newton_accepted: bool
    newton_linear_residual_l2: float | None
    newton_relative_linear_residual: float | None
    newton_operator_applications: int | None
    newton_step_ratio_to_picard: float | None
    newton_accepted_alpha: float | None
    newton_trial_residual_e: float | None
    newton_fallback_reason: str | None


@dataclass(frozen=True)
class Route2SCFIterationState:
    """Immutable same-root arrays for one fixed-point iteration."""

    iteration: int
    density_residual_e: float
    intrinsic_energy_ev: float
    density_coefficients: np.ndarray
    response_density_coefficients: np.ndarray
    reaction_field_values_ev: np.ndarray
    model_local_field_values_ev: np.ndarray | None
    model_field_features: np.ndarray | None

    def __post_init__(self) -> None:
        for name in (
            "density_coefficients",
            "response_density_coefficients",
            "reaction_field_values_ev",
        ):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for name in (
            "model_local_field_values_ev",
            "model_field_features",
        ):
            current = getattr(self, name)
            if current is None:
                continue
            values = np.array(current, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class _NearRootNewtonTrial:
    """One fail-closed Newton attempt and its fresh-map acceptance evidence."""

    accepted_density: np.ndarray | None = None
    linear_residual_l2: float | None = None
    relative_linear_residual: float | None = None
    operator_applications: int | None = None
    step_ratio_to_picard: float | None = None
    accepted_alpha: float | None = None
    trial_residual_e: float | None = None
    fallback_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.accepted_density is not None


class Route2SCFConvergenceError(RuntimeError):
    """Fail-closed SCF error carrying the complete numerical history."""

    def __init__(
        self,
        message: str,
        *,
        history: list[Route2SCFHistoryRecord],
        best_state: Route2SCFIterationState | None = None,
    ) -> None:
        super().__init__(message)
        self.history: tuple[Route2SCFHistoryRecord, ...] = tuple(
            cast(Route2SCFHistoryRecord, dict(record)) for record in history
        )
        self.best_state = best_state


@dataclass(frozen=True)
class Route2EngineSettings:
    """Numerical policy for one versioned Route-2 provider profile."""

    continuum_label: str
    scf_mixing: float
    scf_density_tolerance: float
    scf_energy_tolerance_ev: float
    scf_max_iterations: int
    adjoint_relative_tolerance: float
    adjoint_absolute_tolerance: float
    adjoint_max_iterations: int
    energy_identity_tolerance_ev: float
    force_state_energy_tolerance_ev: float
    neutral_density_tolerance: float
    scf_require_two_energy_samples: bool = False
    scf_solver: str = DAMPED_PICARD_SOLVER
    scf_anderson_depth: int = 6
    scf_anderson_regularization: float = 1.0e-12
    scf_anderson_coefficient_l1_limit: float = 100.0
    scf_anderson_step_ratio_limit: float = 100.0
    scf_anderson_residual_growth_limit: float = 2.0
    scf_total_charge_e: float = 0.0
    scf_newton_trigger_factor: float | None = None
    scf_newton_relative_tolerance: float = 1.0e-8
    scf_newton_absolute_tolerance: float = 1.0e-15
    scf_newton_max_iterations: int = 100
    scf_newton_max_attempts: int = 2
    scf_newton_line_search_steps: int = 3
    scf_newton_step_ratio_limit: float = 10.0

    def __post_init__(self) -> None:
        if not self.continuum_label:
            raise ValueError("A continuum label is required.")
        if self.scf_solver not in SUPPORTED_FIXED_POINT_SOLVERS:
            raise ValueError(f"Unsupported Route-2 SCF solver: {self.scf_solver}.")
        if not math.isfinite(self.scf_total_charge_e):
            raise ValueError("The Route-2 total-charge constraint must be finite.")
        if self.scf_newton_trigger_factor is not None and (
            not math.isfinite(self.scf_newton_trigger_factor)
            or self.scf_newton_trigger_factor <= 1.0
        ):
            raise ValueError(
                "The near-root Newton trigger factor must be greater than 1."
            )
        if not 0.0 < self.scf_mixing <= 1.0:
            raise ValueError("SCF mixing must lie in (0, 1].")
        positive = {
            "scf_density_tolerance": self.scf_density_tolerance,
            "scf_energy_tolerance_ev": self.scf_energy_tolerance_ev,
            "adjoint_relative_tolerance": self.adjoint_relative_tolerance,
            "adjoint_absolute_tolerance": self.adjoint_absolute_tolerance,
            "energy_identity_tolerance_ev": (self.energy_identity_tolerance_ev),
            "force_state_energy_tolerance_ev": (self.force_state_energy_tolerance_ev),
            "neutral_density_tolerance": self.neutral_density_tolerance,
            "scf_anderson_regularization": (self.scf_anderson_regularization),
            "scf_anderson_coefficient_l1_limit": (
                self.scf_anderson_coefficient_l1_limit
            ),
            "scf_anderson_step_ratio_limit": (self.scf_anderson_step_ratio_limit),
            "scf_anderson_residual_growth_limit": (
                self.scf_anderson_residual_growth_limit
            ),
            "scf_newton_relative_tolerance": (
                self.scf_newton_relative_tolerance
            ),
            "scf_newton_absolute_tolerance": (
                self.scf_newton_absolute_tolerance
            ),
            "scf_newton_step_ratio_limit": (
                self.scf_newton_step_ratio_limit
            ),
        }
        invalid = [
            name
            for name, value in positive.items()
            if not math.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(
                "Route-2 engine tolerances must be positive: "
                + ", ".join(invalid)
                + "."
            )
        if (
            self.scf_max_iterations <= 0
            or self.adjoint_max_iterations <= 0
            or self.scf_newton_max_iterations <= 0
            or self.scf_newton_max_attempts <= 0
            or self.scf_newton_line_search_steps <= 0
        ):
            raise ValueError(
                "Route-2 SCF, adjoint, and Newton limits must be positive."
            )
        if self.scf_anderson_depth <= 0:
            raise ValueError("Route-2 Anderson history depth must be positive.")


@dataclass(frozen=True)
class Route2CoupledState:
    """One same-root geometry/provider state reusable by a public wrapper."""

    calculator_identity: int
    atomic_numbers: np.ndarray
    provider_cache_signature: Hashable
    positions_angstrom: np.ndarray
    reaction_field: Any
    density_coefficients: np.ndarray
    response_density_coefficients: np.ndarray
    reaction_field_values_ev: np.ndarray
    model_local_field_values_ev: np.ndarray | None
    model_field_features: np.ndarray | None
    reaction_field_projector: str
    model_field_gauge: str
    model_field_gauge_reference_ev: float
    solvent_state: Any
    polarization_energy_hartree: float
    energy_identity_error_ev: float
    cds_result: Any
    history: tuple[Route2SCFHistoryRecord, ...]

    def __post_init__(self) -> None:
        for name in (
            "atomic_numbers",
            "positions_angstrom",
            "density_coefficients",
            "response_density_coefficients",
            "reaction_field_values_ev",
        ):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.model_field_features is not None:
            features = np.array(self.model_field_features, copy=True)
            features.setflags(write=False)
            object.__setattr__(self, "model_field_features", features)
        if self.model_local_field_values_ev is not None:
            model_field = np.array(
                self.model_local_field_values_ev,
                copy=True,
            )
            model_field.setflags(write=False)
            object.__setattr__(
                self,
                "model_local_field_values_ev",
                model_field,
            )

    @property
    def root_density_coefficients(self) -> np.ndarray:
        """Density used to generate both the stored field and PCM energy."""

        return self.density_coefficients

    @property
    def density_residual_coefficients(self) -> np.ndarray:
        """Unmixed physical residual ``M(P(c)) - c``."""

        residual = self.response_density_coefficients - self.density_coefficients
        residual.setflags(write=False)
        return residual

    @property
    def density_residual_inf(self) -> float:
        return float(np.max(np.abs(self.density_residual_coefficients)))

    def matches(
        self,
        calculator,
        atoms,
        *,
        provider_cache_signature: Hashable,
    ) -> bool:
        return (
            self.calculator_identity == id(calculator)
            and np.array_equal(
                self.atomic_numbers,
                np.asarray(atoms.numbers, dtype=int),
            )
            and self.provider_cache_signature == provider_cache_signature
            and np.array_equal(
                self.positions_angstrom,
                np.asarray(atoms.get_positions(), dtype=float),
            )
        )


@dataclass(frozen=True)
class Route2ContinuumEngine:
    """Provider-independent ML--SCF, adjoint, and energy-ledger engine."""

    reaction_field_factory: Callable[[Any], Any]
    cds_evaluator: Callable[[Any], Any]
    settings: Route2EngineSettings

    def validate_density(
        self,
        values: np.ndarray,
        atom_count: int,
        *,
        name: str,
    ) -> np.ndarray:
        density = np.asarray(values, dtype=float)
        expected_shape = (atom_count, 4)
        if density.shape != expected_shape or not np.all(np.isfinite(density)):
            raise RuntimeError(
                f"{name} must be finite with shape {expected_shape}; "
                f"received {density.shape}."
            )
        monopole_sum = float(np.sum(density[:, 0]))
        if abs(monopole_sum) > self.settings.neutral_density_tolerance:
            raise RuntimeError(
                f"{name} violates the neutral charge constraint "
                f"(sum={monopole_sum:.6e} e)."
            )
        return density.copy()

    def _validate_field(
        self,
        values: np.ndarray,
        atom_count: int,
    ) -> np.ndarray:
        field = np.asarray(values, dtype=float)
        expected_shape = (atom_count, 4)
        if field.shape != expected_shape or not np.all(np.isfinite(field)):
            raise RuntimeError(
                f"The {self.settings.continuum_label} reaction field must be "
                f"finite with shape {expected_shape}; received {field.shape}."
            )
        return field.copy()

    def _reaction_field_drive(
        self,
        reaction_field,
        density: np.ndarray,
        atom_count: int,
    ) -> ReactionFieldDrive:
        apply_drive = getattr(reaction_field, "apply_scf_drive", None)
        if callable(apply_drive):
            drive = apply_drive(density)
            if not isinstance(drive, ReactionFieldDrive):
                raise RuntimeError(
                    f"The {self.settings.continuum_label} reaction-field "
                    "provider returned an invalid model-drive state."
                )
        else:
            drive = ReactionFieldDrive.local_jet(reaction_field.apply_scf(density))
        field = self._validate_field(
            drive.density_dual_field_ev,
            atom_count,
        )
        if drive.model_field_features is None:
            if drive.model_local_field_ev is None:
                return ReactionFieldDrive.local_jet(field)
            return ReactionFieldDrive(
                density_dual_field_ev=field,
                model_local_field_ev=drive.model_local_field_ev,
                model_field_features=None,
                projector=drive.projector,
                model_field_gauge=drive.model_field_gauge,
                model_field_gauge_reference_ev=(drive.model_field_gauge_reference_ev),
            )
        if drive.model_field_features.shape[0] != atom_count:
            raise RuntimeError(
                f"The {self.settings.continuum_label} model features do not "
                "match the atom count."
            )
        return ReactionFieldDrive(
            density_dual_field_ev=field,
            model_local_field_ev=None,
            model_field_features=drive.model_field_features,
            projector=drive.projector,
            model_field_gauge=drive.model_field_gauge,
            model_field_gauge_reference_ev=(drive.model_field_gauge_reference_ev),
        )

    @staticmethod
    def _polarize(calculator, atoms, drive: ReactionFieldDrive, **kwargs):
        if drive.model_field_features is not None:
            return calculator.polar_state(
                atoms,
                model_field_features=drive.model_field_features,
                **kwargs,
            )
        field = (
            drive.density_dual_field_ev
            if drive.model_local_field_ev is None
            else drive.model_local_field_ev
        )
        return calculator.polar_state(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
            **kwargs,
        )

    @staticmethod
    def gas_state(calculator, atoms, *, need_forces: bool) -> Any:
        cached = getattr(calculator, "cached_polar_state", None)
        if callable(cached):
            state = cached(atoms, require_forces=need_forces)
        else:
            state = getattr(calculator, "_last_polar_state", None)
        if state is None or (
            need_forces
            and getattr(
                state,
                "fixed_field_forces_ev_per_angstrom",
                None,
            )
            is None
        ):
            state, _ = calculator.polar_state(
                atoms,
                compute_forces=need_forces,
            )
        return state

    def _try_near_root_newton(
        self,
        *,
        atoms,
        calculator,
        reaction_field,
        density: np.ndarray,
        response_density: np.ndarray,
        drive: ReactionFieldDrive,
        density_residual: float,
    ) -> _NearRootNewtonTrial:
        settings = self.settings
        if (
            drive.model_field_features is not None
            or drive.model_local_field_ev is not None
        ):
            return _NearRootNewtonTrial(
                fallback_reason=(
                    "unsupported-nonlocal-or-gauge-transformed-model-field"
                )
            )
        linearize = getattr(calculator, "linearize_density_response", None)
        if not callable(linearize):
            return _NearRootNewtonTrial(
                fallback_reason="calculator-omits-local-field-linearization"
            )
        try:
            density_response = cast(
                DensityResponseLinearization,
                linearize(
                    atoms,
                    node_potential_ev=drive.density_dual_field_ev[:, 0],
                    node_gradient_ev_per_angstrom=(
                        drive.density_dual_field_ev[:, 1:]
                    ),
                ),
            )
            linearization = UnmixedDensityResidualLinearization(
                atom_count=len(atoms),
                reaction_field=reaction_field,
                density_response=density_response,
                neutral_tolerance=settings.neutral_density_tolerance,
            )
            update_residual = response_density - density
            correction = solve_newton_correction(
                linearization,
                update_residual,
                relative_tolerance=settings.scf_newton_relative_tolerance,
                absolute_tolerance=settings.scf_newton_absolute_tolerance,
                max_iterations=settings.scf_newton_max_iterations,
            )
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            return _NearRootNewtonTrial(
                fallback_reason=(
                    f"linear-solve-failed:{type(exc).__name__}:{exc}"
                )
            )

        picard_norm = float(np.linalg.norm(update_residual))
        correction_norm = float(np.linalg.norm(correction.solution))
        if picard_norm == 0.0:
            return _NearRootNewtonTrial(
                linear_residual_l2=correction.residual_norm,
                relative_linear_residual=correction.relative_residual,
                operator_applications=correction.operator_applications,
                fallback_reason="zero-picard-residual",
            )
        step_ratio = correction_norm / picard_norm
        common = {
            "linear_residual_l2": correction.residual_norm,
            "relative_linear_residual": correction.relative_residual,
            "operator_applications": correction.operator_applications,
            "step_ratio_to_picard": step_ratio,
        }
        if (
            not math.isfinite(step_ratio)
            or step_ratio > settings.scf_newton_step_ratio_limit
        ):
            return _NearRootNewtonTrial(
                **common,
                fallback_reason="newton-step-ratio-limit",
            )

        minimum_trial_residual: float | None = None
        trial_failures: list[str] = []
        strict_decrease_guard = (
            8.0
            * np.finfo(float).eps
            * max(1.0, abs(density_residual))
        )
        for line_search_index in range(settings.scf_newton_line_search_steps):
            alpha = 0.5**line_search_index
            candidate = project_density_total_charge(
                density + alpha * correction.solution,
                total_charge_e=settings.scf_total_charge_e,
            )
            try:
                candidate_drive = self._reaction_field_drive(
                    reaction_field,
                    candidate,
                    len(atoms),
                )
                candidate_state, _ = self._polarize(
                    calculator,
                    atoms,
                    candidate_drive,
                )
                raw_candidate_response = self.validate_density(
                    candidate_state.density_coefficients,
                    len(atoms),
                    name="Newton-trial MACE-POLAR density",
                )
                candidate_response = project_density_total_charge(
                    raw_candidate_response,
                    total_charge_e=settings.scf_total_charge_e,
                )
                candidate_residual = float(
                    np.max(np.abs(candidate_response - candidate))
                )
                candidate_energy = float(candidate_state.energy_ev)
                if not math.isfinite(candidate_energy):
                    raise RuntimeError(
                        "Newton-trial MACE-POLAR energy is non-finite."
                    )
            except (NotImplementedError, RuntimeError, ValueError) as exc:
                trial_failures.append(
                    f"alpha={alpha:g}:{type(exc).__name__}:{exc}"
                )
                continue
            minimum_trial_residual = (
                candidate_residual
                if minimum_trial_residual is None
                else min(minimum_trial_residual, candidate_residual)
            )
            if candidate_residual < (
                density_residual - strict_decrease_guard
            ):
                return _NearRootNewtonTrial(
                    accepted_density=candidate,
                    **common,
                    accepted_alpha=alpha,
                    trial_residual_e=candidate_residual,
                )

        if minimum_trial_residual is None and trial_failures:
            fallback_reason = (
                "fresh-map-line-search-failed:" + ";".join(trial_failures)
            )
        elif trial_failures:
            fallback_reason = (
                "fresh-map-line-search-rejected-with-errors:"
                + ";".join(trial_failures)
            )
        else:
            fallback_reason = "fresh-map-line-search-rejected"
        return _NearRootNewtonTrial(
            **common,
            trial_residual_e=minimum_trial_residual,
            fallback_reason=fallback_reason,
        )

    def solve_coupled_state(
        self,
        atoms,
        calculator,
        gas_state,
        *,
        provider_cache_signature: Hashable,
    ) -> Route2CoupledState:
        settings = self.settings
        reaction_field = self.reaction_field_factory(atoms)
        density = project_density_total_charge(
            self.validate_density(
                gas_state.density_coefficients,
                len(atoms),
                name="Gas MACE-POLAR density",
            ),
            total_charge_e=settings.scf_total_charge_e,
        )
        previous_energy_ev: float | None = None
        previous_density_residual: float | None = None
        previous_update_method: str | None = None
        history: list[Route2SCFHistoryRecord] = []
        fixed_point_samples: list[FixedPointSample] = []
        best_iteration_state: Route2SCFIterationState | None = None
        newton_attempt_count = 0

        for iteration in range(1, settings.scf_max_iterations + 1):
            drive = self._reaction_field_drive(
                reaction_field,
                density,
                len(atoms),
            )
            field = drive.density_dual_field_ev
            solvent_state, _ = self._polarize(
                calculator,
                atoms,
                drive,
            )
            raw_response_density = self.validate_density(
                solvent_state.density_coefficients,
                len(atoms),
                name="Field-polarized MACE-POLAR density",
            )
            response_density = project_density_total_charge(
                raw_response_density,
                total_charge_e=settings.scf_total_charge_e,
            )
            density_residual = float(np.max(np.abs(response_density - density)))
            current_energy_ev = float(solvent_state.energy_ev)
            if not math.isfinite(current_energy_ev):
                raise RuntimeError("Field-polarized MACE-POLAR energy is non-finite.")
            energy_residual = (
                None
                if previous_energy_ev is None
                else abs(current_energy_ev - previous_energy_ev)
            )
            reset_anderson_history = (
                settings.scf_solver == SAFEGUARDED_ANDERSON_SOLVER
                and previous_update_method == SAFEGUARDED_ANDERSON_SOLVER
                and previous_density_residual is not None
                and density_residual
                > (
                    settings.scf_anderson_residual_growth_limit
                    * previous_density_residual
                )
            )
            if reset_anderson_history:
                fixed_point_samples.clear()
            record: Route2SCFHistoryRecord = {
                "iteration": iteration,
                "density_residual_e": density_residual,
                "energy_residual_ev": energy_residual,
                "intrinsic_energy_ev": current_energy_ev,
                "root_total_charge_e": float(np.sum(density[:, 0])),
                "raw_response_total_charge_e": float(
                    np.sum(raw_response_density[:, 0])
                ),
                "response_charge_projection_max_e": float(
                    np.max(np.abs(response_density - raw_response_density))
                ),
                "arrived_by": previous_update_method,
                "anderson_history_reset": reset_anderson_history,
                "next_density_update": None,
                "fixed_point_history_size": None,
                "anderson_predicted_residual_l2": None,
                "anderson_coefficient_l1": None,
                "anderson_step_ratio_to_picard": None,
                "anderson_fallback_reason": None,
                "newton_attempted": False,
                "newton_accepted": False,
                "newton_linear_residual_l2": None,
                "newton_relative_linear_residual": None,
                "newton_operator_applications": None,
                "newton_step_ratio_to_picard": None,
                "newton_accepted_alpha": None,
                "newton_trial_residual_e": None,
                "newton_fallback_reason": None,
            }
            history.append(record)
            if (
                best_iteration_state is None
                or density_residual < best_iteration_state.density_residual_e
            ):
                best_iteration_state = Route2SCFIterationState(
                    iteration=iteration,
                    density_residual_e=density_residual,
                    intrinsic_energy_ev=current_energy_ev,
                    density_coefficients=density,
                    response_density_coefficients=response_density,
                    reaction_field_values_ev=field,
                    model_local_field_values_ev=drive.model_local_field_ev,
                    model_field_features=drive.model_field_features,
                )
            if energy_residual is None:
                energy_converged = not settings.scf_require_two_energy_samples
            else:
                energy_converged = energy_residual <= settings.scf_energy_tolerance_ev
            if density_residual <= settings.scf_density_tolerance and energy_converged:
                record["next_density_update"] = "converged"
                break
            should_try_newton = (
                settings.scf_newton_trigger_factor is not None
                and density_residual > settings.scf_density_tolerance
                and density_residual
                <= (
                    settings.scf_newton_trigger_factor
                    * settings.scf_density_tolerance
                )
                and newton_attempt_count < settings.scf_newton_max_attempts
                and iteration < settings.scf_max_iterations
            )
            if should_try_newton:
                newton_attempt_count += 1
                trial = self._try_near_root_newton(
                    atoms=atoms,
                    calculator=calculator,
                    reaction_field=reaction_field,
                    density=density,
                    response_density=response_density,
                    drive=drive,
                    density_residual=density_residual,
                )
                record.update(
                    {
                        "newton_attempted": True,
                        "newton_accepted": trial.accepted,
                        "newton_linear_residual_l2": (
                            trial.linear_residual_l2
                        ),
                        "newton_relative_linear_residual": (
                            trial.relative_linear_residual
                        ),
                        "newton_operator_applications": (
                            trial.operator_applications
                        ),
                        "newton_step_ratio_to_picard": (
                            trial.step_ratio_to_picard
                        ),
                        "newton_accepted_alpha": trial.accepted_alpha,
                        "newton_trial_residual_e": (
                            trial.trial_residual_e
                        ),
                        "newton_fallback_reason": trial.fallback_reason,
                    }
                )
                if trial.accepted_density is not None:
                    record["next_density_update"] = (
                        NEAR_ROOT_NEWTON_CORRECTOR
                    )
                    fixed_point_samples.clear()
                    density = trial.accepted_density
                    previous_energy_ev = current_energy_ev
                    previous_density_residual = density_residual
                    previous_update_method = NEAR_ROOT_NEWTON_CORRECTOR
                    continue
            fixed_point_samples.append(
                FixedPointSample(
                    density=density,
                    residual=response_density - density,
                )
            )
            fixed_point_samples = fixed_point_samples[
                -(settings.scf_anderson_depth + 1) :
            ]
            step = next_fixed_point_density(
                fixed_point_samples,
                solver=settings.scf_solver,
                mixing=settings.scf_mixing,
                anderson_depth=settings.scf_anderson_depth,
                anderson_regularization=(settings.scf_anderson_regularization),
                anderson_coefficient_l1_limit=(
                    settings.scf_anderson_coefficient_l1_limit
                ),
                anderson_step_ratio_limit=(settings.scf_anderson_step_ratio_limit),
            )
            record.update(
                {
                    "next_density_update": step.method,
                    "fixed_point_history_size": step.history_size,
                    "anderson_predicted_residual_l2": (step.predicted_residual_l2),
                    "anderson_coefficient_l1": step.coefficient_l1,
                    "anderson_step_ratio_to_picard": (step.step_ratio_to_picard),
                    "anderson_fallback_reason": step.fallback_reason,
                }
            )
            density = step.density
            previous_energy_ev = current_energy_ev
            previous_density_residual = density_residual
            previous_update_method = step.method
        else:
            last = history[-1]
            minimum_density_residual = min(
                record["density_residual_e"] for record in history
            )
            raise Route2SCFConvergenceError(
                "MACE-POLAR/"
                f"{settings.continuum_label} reaction-field SCF did not "
                f"converge in {settings.scf_max_iterations} iterations "
                f"(density residual={last['density_residual_e']:.3e} e, "
                f"energy residual={last['energy_residual_ev']!r} eV, "
                f"minimum density residual={minimum_density_residual:.3e} e).",
                history=history,
                best_state=best_iteration_state,
            )

        polarization_energy_hartree = float(
            reaction_field.scf_polarization_energy_hartree(density)
        )
        if not math.isfinite(polarization_energy_hartree):
            raise RuntimeError(
                f"{settings.continuum_label} polarization energy is " "non-finite."
            )
        paired_energy_ev = 0.5 * float(MACE_POLAR_L1_PAIRING.pair(density, field))
        provider_energy_ev = polarization_energy_hartree * Hartree
        identity_error_ev = abs(paired_energy_ev - provider_energy_ev)
        if identity_error_ev > settings.energy_identity_tolerance_ev:
            raise RuntimeError(
                f"{settings.continuum_label} reaction field failed the "
                "polarization-energy identity "
                f"(absolute error={identity_error_ev:.3e} eV)."
            )

        cds_result = self.cds_evaluator(atoms)
        return Route2CoupledState(
            calculator_identity=id(calculator),
            atomic_numbers=np.asarray(atoms.numbers, dtype=int).copy(),
            provider_cache_signature=provider_cache_signature,
            positions_angstrom=np.asarray(
                atoms.get_positions(),
                dtype=float,
            ).copy(),
            reaction_field=reaction_field,
            density_coefficients=density,
            response_density_coefficients=response_density,
            reaction_field_values_ev=field,
            model_local_field_values_ev=drive.model_local_field_ev,
            model_field_features=drive.model_field_features,
            reaction_field_projector=drive.projector,
            model_field_gauge=drive.model_field_gauge,
            model_field_gauge_reference_ev=(drive.model_field_gauge_reference_ev),
            solvent_state=solvent_state,
            polarization_energy_hartree=polarization_energy_hartree,
            energy_identity_error_ev=identity_error_ev,
            cds_result=cds_result,
            history=tuple(history),
        )

    def solvent_correction_force(
        self,
        atoms,
        calculator,
        gas_state,
        coupled: Route2CoupledState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.settings
        if (
            coupled.model_field_features is not None
            or coupled.model_local_field_values_ev is not None
        ):
            raise NotImplementedError(
                "A non-default model-field profile is energy-only until its "
                "gauge/projector response adjoint and position VJP are derived."
            )
        field = coupled.reaction_field_values_ev
        density = coupled.density_coefficients
        solvent_state, _ = calculator.polar_state(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
            compute_forces=True,
        )
        response_density = project_density_total_charge(
            self.validate_density(
                solvent_state.density_coefficients,
                len(atoms),
                name="Force-evaluation MACE-POLAR density",
            ),
            total_charge_e=settings.scf_total_charge_e,
        )
        force_state_residual = float(np.max(np.abs(response_density - density)))
        if force_state_residual > max(
            10.0 * settings.scf_density_tolerance,
            1.0e-10,
        ):
            raise RuntimeError(
                "The MACE-POLAR force state does not match the converged "
                f"density root (residual={force_state_residual:.3e} e)."
            )
        force_state_energy_error_ev = abs(
            float(solvent_state.energy_ev) - float(coupled.solvent_state.energy_ev)
        )
        if (
            not math.isfinite(force_state_energy_error_ev)
            or force_state_energy_error_ev > settings.force_state_energy_tolerance_ev
        ):
            raise RuntimeError(
                "The MACE-POLAR force evaluation does not reproduce the "
                "converged intrinsic energy "
                f"(absolute error={force_state_energy_error_ev:.3e} eV)."
            )

        gas_forces = getattr(
            gas_state,
            "fixed_field_forces_ev_per_angstrom",
            None,
        )
        solvent_forces = getattr(
            solvent_state,
            "fixed_field_forces_ev_per_angstrom",
            None,
        )
        if gas_forces is None or solvent_forces is None:
            raise RuntimeError(
                "MACE-POLAR omitted the gas or fixed-field force partial."
            )

        intrinsic_gradient = calculator.intrinsic_energy_field_gradient(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
        )
        density_response = calculator.linearize_density_response(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
        )
        physical_rhs = fixed_cavity_energy_density_gradient(
            coupled.reaction_field,
            reaction_field_values=field,
            intrinsic_energy_field_gradient=intrinsic_gradient,
        )
        residual = UnmixedDensityResidualLinearization(
            atom_count=len(atoms),
            reaction_field=coupled.reaction_field,
            density_response=density_response,
        )
        adjoint = solve_adjoint(
            residual,
            physical_rhs,
            relative_tolerance=settings.adjoint_relative_tolerance,
            absolute_tolerance=settings.adjoint_absolute_tolerance,
            max_iterations=settings.adjoint_max_iterations,
        )
        density_position_vjp = calculator.density_position_vjp(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
            density_cotangent=adjoint.solution,
        )
        continuum_gradient = continuum_coupled_solvation_coordinate_gradient(
            coupled.reaction_field,
            density_response,
            density_coefficients=density,
            intrinsic_energy_field_gradient=intrinsic_gradient,
            adjoint_solution=adjoint.solution,
            adjoint_density_position_vjp=density_position_vjp,
            solvent_fixed_field_forces_ev_per_angstrom=solvent_forces,
            gas_forces_ev_per_angstrom=gas_forces,
        )
        total = assemble_total_solvation_coordinate_gradient(
            continuum_gradient,
            coupled.cds_result.position_gradient_hartree_per_angstrom,
        )
        derivative = {
            "continuum_position_gradient_ev_per_angstrom": (continuum_gradient),
            "cds_position_gradient_hartree_per_angstrom": (
                coupled.cds_result.position_gradient_hartree_per_angstrom
            ),
            "total_position_gradient_hartree_per_angstrom": (
                total.total_position_gradient_hartree_per_angstrom
            ),
            "solvent_correction_forces_hartree_per_angstrom": (
                total.solvent_correction_forces_hartree_per_angstrom
            ),
            "force_state_density_residual_e": force_state_residual,
            "force_state_energy_error_ev": force_state_energy_error_ev,
            "adjoint": {
                "method": adjoint.method,
                "relative_tolerance": settings.adjoint_relative_tolerance,
                "absolute_tolerance": settings.adjoint_absolute_tolerance,
                "restart_size": adjoint.restart_size,
                "maximum_inner_iterations": (adjoint.maximum_inner_iterations),
                "operator_applications": adjoint.operator_applications,
                "residual_callback_count": (adjoint.residual_callback_count),
                "residual_norm": adjoint.residual_norm,
                "relative_residual": adjoint.relative_residual,
            },
        }
        return (
            total.solvent_correction_forces_hartree_per_angstrom,
            derivative,
        )

    @staticmethod
    def compose_energy_components(
        *,
        gas_energy_ev: float,
        solvent_energy_ev: float,
        polarization_energy_hartree: float,
        cds_energy_hartree: float,
    ) -> dict[str, float]:
        """Compose the single Route-2 scalar-energy ledger."""

        delta_e_solute = (float(solvent_energy_ev) - float(gas_energy_ev)) / Hartree
        pcm_polarization = float(polarization_energy_hartree)
        electrostatic = delta_e_solute + pcm_polarization
        cds_energy = float(cds_energy_hartree)
        total_energy = electrostatic + cds_energy
        return {
            "solute_polarization": delta_e_solute,
            "pcm_polarization": pcm_polarization,
            "electrostatic": electrostatic,
            "cds": cds_energy,
            "standard_state": 0.0,
            "delta_g_solv": total_energy,
        }

    @classmethod
    def energy_components(
        cls,
        gas_state,
        coupled: Route2CoupledState,
    ) -> dict[str, float]:
        return cls.compose_energy_components(
            gas_energy_ev=float(gas_state.energy_ev),
            solvent_energy_ev=float(coupled.solvent_state.energy_ev),
            polarization_energy_hartree=(coupled.polarization_energy_hartree),
            cds_energy_hartree=float(coupled.cds_result.energy_hartree),
        )


__all__ = [
    "Route2ContinuumEngine",
    "Route2CoupledState",
    "Route2EngineSettings",
]
