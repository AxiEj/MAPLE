"""Internal orchestration for self-consistent polarizable continuum coupling.

This module contains no continuum-provider construction or public dispatch.
One provider wrapper supplies a same-energy reaction-field factory and a
matching CDS evaluator; the engine owns only the common ML--SCF root,
implicit-function adjoint, and component ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
import hashlib
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
    UnmixedDensityResidualLinearization,
    solve_adjoint,
)


class Route2SCFHistoryRecord(TypedDict):
    """One complete JSON-serializable fixed-point iteration record."""

    iteration: int
    density_residual_e: float
    monopole_residual_e: float
    dipole_residual_e_angstrom: float
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


@dataclass(frozen=True)
class Route2FiniteResolutionPolicy:
    """Finite-resolution policy for an energy-only approximate fixed-point gate."""

    version: str
    history_length: int
    map_replay_count: int
    monopole_residual_ceiling_e: float
    dipole_residual_ceiling_e_angstrom: float
    potential_span_tolerance_ev: float
    gradient_span_tolerance_ev_per_angstrom: float
    ledger_span_tolerance_ev: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "version": self.version,
            "history_length": self.history_length,
            "map_replay_count": self.map_replay_count,
            "monopole_residual_ceiling_e": self.monopole_residual_ceiling_e,
            "dipole_residual_ceiling_e_angstrom": self.dipole_residual_ceiling_e_angstrom,
            "potential_span_tolerance_ev": self.potential_span_tolerance_ev,
            "gradient_span_tolerance_ev_per_angstrom": (
                self.gradient_span_tolerance_ev_per_angstrom
            ),
            "ledger_span_tolerance_ev": self.ledger_span_tolerance_ev,
        }

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("Route-2 finite-resolution policy must define a version.")
        for name in ("history_length", "map_replay_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"Route-2 finite-resolution {name} must be a positive integer."
                )
        positive = {
            "monopole_residual_ceiling_e": (self.monopole_residual_ceiling_e),
            "dipole_residual_ceiling_e_angstrom": (
                self.dipole_residual_ceiling_e_angstrom
            ),
            "potential_span_tolerance_ev": (self.potential_span_tolerance_ev),
            "gradient_span_tolerance_ev_per_angstrom": (
                self.gradient_span_tolerance_ev_per_angstrom
            ),
            "ledger_span_tolerance_ev": (self.ledger_span_tolerance_ev),
        }
        invalid = [
            name
            for name, value in positive.items()
            if isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0.0
        ]
        if invalid:
            raise ValueError(
                "Route-2 finite-resolution tolerances must be positive: "
                + ", ".join(invalid)
                + "."
            )


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
    scf_dipole_tolerance_e_angstrom: float | None = None
    scf_finite_resolution_policy: Route2FiniteResolutionPolicy | None = None
    scf_require_two_energy_samples: bool = False
    scf_solver: str = DAMPED_PICARD_SOLVER
    scf_anderson_depth: int = 6
    scf_anderson_regularization: float = 1.0e-12
    scf_anderson_coefficient_l1_limit: float = 100.0
    scf_anderson_step_ratio_limit: float = 100.0
    scf_anderson_residual_growth_limit: float = 2.0
    scf_total_charge_e: float = 0.0

    def __post_init__(self) -> None:
        if not self.continuum_label:
            raise ValueError("A continuum label is required.")
        if self.scf_solver not in SUPPORTED_FIXED_POINT_SOLVERS:
            raise ValueError(f"Unsupported Route-2 SCF solver: {self.scf_solver}.")
        if not math.isfinite(self.scf_total_charge_e):
            raise ValueError("The Route-2 total-charge constraint must be finite.")
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
        if self.scf_max_iterations <= 0 or self.adjoint_max_iterations <= 0:
            raise ValueError(
                "Route-2 SCF and adjoint iteration limits must be positive."
            )
        if self.scf_anderson_depth <= 0:
            raise ValueError("Route-2 Anderson history depth must be positive.")
        scf_dipole_tolerance = (
            self.scf_density_tolerance
            if self.scf_dipole_tolerance_e_angstrom is None
            else self.scf_dipole_tolerance_e_angstrom
        )
        object.__setattr__(
            self,
            "scf_dipole_tolerance_e_angstrom",
            scf_dipole_tolerance,
        )
        if (
            not math.isfinite(scf_dipole_tolerance)
            or scf_dipole_tolerance <= 0.0
        ):
            raise ValueError("scf_dipole_tolerance_e_angstrom must be positive.")
        if (
            self.scf_finite_resolution_policy is not None
            and not isinstance(
                self.scf_finite_resolution_policy,
                Route2FiniteResolutionPolicy,
            )
        ):
            raise ValueError("Route-2 finite-resolution policy is malformed.")


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
    scf_convergence: dict[str, Any] = field(default_factory=dict)

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
                reaction_field_drive = ReactionFieldDrive.local_jet(field)
            else:
                reaction_field_drive = ReactionFieldDrive(
                    density_dual_field_ev=field,
                    model_local_field_ev=drive.model_local_field_ev,
                    model_field_features=None,
                    projector=drive.projector,
                    model_field_gauge=drive.model_field_gauge,
                    model_field_gauge_reference_ev=(
                        drive.model_field_gauge_reference_ev
                    ),
                )
        elif drive.model_field_features.shape[0] != atom_count:
            raise RuntimeError(
                f"The {self.settings.continuum_label} model features do not "
                "match the atom count."
            )
        else:
            reaction_field_drive = ReactionFieldDrive(
                density_dual_field_ev=field,
                model_local_field_ev=None,
                model_field_features=drive.model_field_features,
                projector=drive.projector,
                model_field_gauge=drive.model_field_gauge,
                model_field_gauge_reference_ev=(
                    drive.model_field_gauge_reference_ev
                ),
            )
        return reaction_field_drive

    @staticmethod
    def _maximum_component_span(values: np.ndarray) -> float:
        """Return the largest per-component peak-to-peak span over iterations."""

        array = np.asarray(values, dtype=float)
        if array.ndim < 1 or array.shape[0] == 0:
            raise ValueError("A non-empty iteration axis is required.")
        return float(np.max(np.ptp(array, axis=0)))

    @staticmethod
    def _array_sha256(values: np.ndarray) -> str:
        canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

    @staticmethod
    def _maximum_ulp_distance(left: np.ndarray, right: np.ndarray) -> int:
        left_values = np.asarray(left, dtype=np.float64).copy()
        right_values = np.asarray(right, dtype=np.float64).copy()
        if left_values.shape != right_values.shape:
            raise ValueError("ULP inputs must be shape-compatible.")
        left_values[left_values == 0.0] = 0.0
        right_values[right_values == 0.0] = 0.0

        SIGN = 1 << 63
        MAG_MASK = SIGN - 1

        def ordered(raw: int) -> int:
            magnitude = raw & MAG_MASK
            return SIGN - magnitude if (raw & SIGN) else SIGN + magnitude

        return max(
            abs(
                ordered(int(raw))
                - ordered(int(raw_rhs))
            )
            for raw, raw_rhs in zip(
                left_values.view(np.uint64).ravel(order="C"),
                right_values.view(np.uint64).ravel(order="C"),
            )
        )

    @staticmethod
    def _reaction_field_signature(
        drive: ReactionFieldDrive,
    ) -> tuple[Any, ...]:
        return (
            drive.projector,
            drive.model_field_gauge,
            float(drive.model_field_gauge_reference_ev),
            None
            if drive.model_field_features is None
            else tuple(drive.model_field_features.shape),
            None
            if drive.model_local_field_ev is None
            else tuple(drive.model_local_field_ev.shape),
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

    def _finite_resolution_candidate_window(
        self,
        *,
        policy: Route2FiniteResolutionPolicy,
        iteration_history: list[Route2SCFHistoryRecord],
        density_history: list[np.ndarray],
        residual_history: list[np.ndarray],
        field_history: list[np.ndarray],
        intrinsic_energy_history: list[float],
    ) -> dict[str, float | int] | None:
        window_size = policy.history_length
        if len(iteration_history) < window_size:
            return None
        window_records = iteration_history[-window_size:]
        if any(
            record["arrived_by"] != SAFEGUARDED_ANDERSON_SOLVER
            or record["anderson_history_reset"]
            for record in window_records
        ):
            return None
        energy_deltas = [record["energy_residual_ev"] for record in window_records]
        if any(
            delta is None or delta > self.settings.scf_energy_tolerance_ev
            for delta in energy_deltas
        ):
            return None
        density_window = np.stack(density_history[-window_size:], axis=0)
        residual_window = np.stack(residual_history[-window_size:], axis=0)
        field_window = np.stack(field_history[-window_size:], axis=0)
        intrinsic_energy_window = intrinsic_energy_history[-window_size:]

        monopole_residual_max = float(np.max(np.abs(residual_window[:, :, 0])))
        dipole_residual_max = float(np.max(np.abs(residual_window[:, :, 1:])))
        if (
            monopole_residual_max > policy.monopole_residual_ceiling_e
            or dipole_residual_max > policy.dipole_residual_ceiling_e_angstrom
        ):
            return None
        residual_reference = residual_window[-1]
        for residual in residual_window[:-1]:
            for current_channel, reference_channel in (
                (residual[:, 0], residual_reference[:, 0]),
                (residual[:, 1:].reshape(-1), residual_reference[:, 1:].reshape(-1)),
            ):
                reference_norm = float(np.linalg.norm(reference_channel))
                current_norm = float(np.linalg.norm(current_channel))
                if reference_norm <= 0.0:
                    if current_norm > 0.0:
                        return None
                    continue
                if float(np.dot(current_channel, reference_channel)) < 0.0:
                    return None

        root_monopole_span_e = self._maximum_component_span(
            density_window[:, :, 0]
        )
        root_dipole_span_e_angstrom = self._maximum_component_span(
            density_window[:, :, 1:]
        )
        residual_monopole_span_e = self._maximum_component_span(
            residual_window[:, :, 0]
        )
        residual_dipole_span_e_angstrom = self._maximum_component_span(
            residual_window[:, :, 1:]
        )
        dipole_tolerance_e_angstrom = cast(
            float,
            self.settings.scf_dipole_tolerance_e_angstrom,
        )
        if (
            root_monopole_span_e > self.settings.scf_density_tolerance
            or root_dipole_span_e_angstrom
            > dipole_tolerance_e_angstrom
            or residual_monopole_span_e > self.settings.scf_density_tolerance
            or residual_dipole_span_e_angstrom
            > dipole_tolerance_e_angstrom
        ):
            return None

        potential_span_ev = self._maximum_component_span(
            field_window[:, :, 0]
        )
        gradient_span_ev_per_angstrom = self._maximum_component_span(
            field_window[:, :, 1:]
        )
        if (
            potential_span_ev > policy.potential_span_tolerance_ev
            or gradient_span_ev_per_angstrom
            > policy.gradient_span_tolerance_ev_per_angstrom
        ):
            return None

        intrinsic_span_ev = float(
            max(intrinsic_energy_window) - min(intrinsic_energy_window)
        )
        if intrinsic_span_ev > self.settings.scf_energy_tolerance_ev:
            return None

        return {
            "start_iteration": window_records[0]["iteration"],
            "end_iteration": window_records[-1]["iteration"],
            "maximum_monopole_residual_e": monopole_residual_max,
            "maximum_dipole_residual_e_angstrom": dipole_residual_max,
            "root_monopole_span_e": root_monopole_span_e,
            "root_dipole_span_e_angstrom": root_dipole_span_e_angstrom,
            "residual_monopole_span_e": residual_monopole_span_e,
            "residual_dipole_span_e_angstrom": residual_dipole_span_e_angstrom,
            "potential_span_ev": potential_span_ev,
            "gradient_span_ev_per_angstrom": gradient_span_ev_per_angstrom,
            "maximum_energy_delta_ev": max(
                float(delta) for delta in energy_deltas if delta is not None
            ),
            "intrinsic_energy_span_ev": intrinsic_span_ev,
        }

    def _run_finite_resolution_map_replays(
        self,
        *,
        atoms,
        calculator,
        gas_state,
        policy: Route2FiniteResolutionPolicy,
        candidate_reaction_field,
        candidate_density: np.ndarray,
        candidate_field: np.ndarray,
        candidate_drive_signature: tuple[Any, ...],
        candidate_response: np.ndarray,
        candidate_residual: np.ndarray,
        candidate_intrinsic_energy_ev: float,
    ) -> dict[str, Any]:
        atom_count = len(atoms)
        field_reference = np.array(candidate_field, copy=True)
        response_reference = np.array(candidate_response, copy=True)
        candidate_residual = np.array(candidate_residual, copy=True)

        replay_fields: list[np.ndarray] = []
        replay_responses: list[np.ndarray] = []
        replay_residuals: list[np.ndarray] = []
        candidate_pcm_polarization_hartree = float(
            candidate_reaction_field.scf_polarization_energy_hartree(
                candidate_density,
            )
        )
        candidate_paired_energy_ev = 0.5 * float(
            MACE_POLAR_L1_PAIRING.pair(
                candidate_density,
                candidate_field,
            )
        )
        candidate_provider_energy_ev = candidate_pcm_polarization_hartree * Hartree
        candidate_polarization_identity_error_ev = abs(
            candidate_paired_energy_ev - candidate_provider_energy_ev
        )
        candidate_electrostatic_ledger_ev = (
            candidate_intrinsic_energy_ev - float(gas_state.energy_ev)
        ) + candidate_provider_energy_ev
        candidate_scalars = {
            "online intrinsic energy": candidate_intrinsic_energy_ev,
            "online PCM polarization energy": candidate_pcm_polarization_hartree,
            "online electrostatic ledger": candidate_electrostatic_ledger_ev,
            "online polarization identity error": (
                candidate_polarization_identity_error_ev
            ),
        }
        if any(not math.isfinite(value) for value in candidate_scalars.values()):
            raise Route2SCFConvergenceError(
                "Finite-resolution online map evidence contains a non-finite "
                "energy scalar.",
                history=[],
            )

        observed_intrinsic_energies_ev = [candidate_intrinsic_energy_ev]
        observed_pcm_polarization_hartree = [
            candidate_pcm_polarization_hartree
        ]
        observed_electrostatic_ledger_ev = [
            candidate_electrostatic_ledger_ev
        ]
        observed_polarization_identity_errors_ev = [
            candidate_polarization_identity_error_ev
        ]

        for _ in range(policy.map_replay_count):
            replay_reaction_field = self.reaction_field_factory(atoms)
            replay_drive = self._reaction_field_drive(
                replay_reaction_field,
                candidate_density,
                atom_count,
            )
            if self._reaction_field_signature(replay_drive) != (
                candidate_drive_signature
            ):
                raise Route2SCFConvergenceError(
                    "Finite-resolution replay identity did not match the online "
                    "candidate reaction-field signature.",
                    history=[],
                )
            replay_field = replay_drive.density_dual_field_ev
            replay_solvent_state, _ = self._polarize(
                calculator,
                atoms,
                replay_drive,
            )
            replay_response_raw = self.validate_density(
                replay_solvent_state.density_coefficients,
                atom_count,
                name="Map-replay MACE-POLAR density",
            )
            replay_response = project_density_total_charge(
                replay_response_raw,
                total_charge_e=self.settings.scf_total_charge_e,
            )
            replay_residual = replay_response - candidate_density
            replay_fields.append(replay_field)
            replay_responses.append(replay_response)
            replay_residuals.append(replay_residual)
            replay_intrinsic = float(replay_solvent_state.energy_ev)
            replay_pcm_polarization = float(
                replay_reaction_field.scf_polarization_energy_hartree(
                    candidate_density,
                )
            )
            paired_energy_ev = 0.5 * float(
                MACE_POLAR_L1_PAIRING.pair(
                    candidate_density,
                    replay_field,
                )
            )
            provider_energy_ev = replay_pcm_polarization * Hartree
            replay_polarization_identity_error_ev = abs(
                paired_energy_ev - provider_energy_ev
            )
            replay_electrostatic_ledger_ev = (
                (replay_intrinsic - float(gas_state.energy_ev))
                + provider_energy_ev
            )
            replay_scalars = {
                "replayed intrinsic energy": replay_intrinsic,
                "replayed PCM polarization energy": replay_pcm_polarization,
                "replayed electrostatic ledger": replay_electrostatic_ledger_ev,
                "replayed polarization identity error": (
                    replay_polarization_identity_error_ev
                ),
            }
            if any(not math.isfinite(value) for value in replay_scalars.values()):
                raise Route2SCFConvergenceError(
                    "Finite-resolution map replay contains a non-finite energy "
                    "scalar.",
                    history=[],
                )
            observed_intrinsic_energies_ev.append(replay_intrinsic)
            observed_pcm_polarization_hartree.append(
                replay_pcm_polarization
            )
            observed_electrostatic_ledger_ev.append(
                replay_electrostatic_ledger_ev
            )
            observed_polarization_identity_errors_ev.append(
                replay_polarization_identity_error_ev
            )

        field_sha256_by_evaluation = [
            self._array_sha256(field_reference),
            *[self._array_sha256(field) for field in replay_fields],
        ]
        response_sha256_by_evaluation = [
            self._array_sha256(response_reference),
            *[self._array_sha256(response) for response in replay_responses],
        ]
        cold_replay_field_arrays_identical = (
            len(set(field_sha256_by_evaluation[1:])) == 1
        )
        cold_replay_response_arrays_identical = (
            len(set(response_sha256_by_evaluation[1:])) == 1
        )
        response_monopole_tolerance = self.settings.scf_density_tolerance
        response_dipole_tolerance = cast(
            float,
            self.settings.scf_dipole_tolerance_e_angstrom,
        )
        field_potential_deltas = []
        field_gradient_deltas = []
        response_monopole_deltas = []
        response_dipole_deltas = []
        field_potential_ulp = []
        field_gradient_ulp = []
        response_monopole_ulp = []
        response_dipole_ulp = []
        for replay_field, replay_response in zip(
            replay_fields,
            replay_responses,
        ):
            field_delta = np.abs(field_reference - replay_field)
            response_delta = np.abs(response_reference - replay_response)
            field_potential_delta = float(np.max(field_delta[:, 0]))
            field_gradient_delta = float(np.max(field_delta[:, 1:]))
            response_monopole_delta = float(np.max(response_delta[:, 0]))
            response_dipole_delta = float(np.max(response_delta[:, 1:]))
            if field_potential_delta > policy.potential_span_tolerance_ev:
                raise Route2SCFConvergenceError(
                    "Finite-resolution map-replay field potential delta exceeds "
                    "policy tolerance.",
                    history=[],
                )
            if field_gradient_delta > policy.gradient_span_tolerance_ev_per_angstrom:
                raise Route2SCFConvergenceError(
                    "Finite-resolution map-replay field gradient delta exceeds "
                    "policy tolerance.",
                    history=[],
                )
            if response_monopole_delta > response_monopole_tolerance:
                raise Route2SCFConvergenceError(
                    "Finite-resolution map-replay response monopole delta exceeds "
                    "nominal density tolerance.",
                    history=[],
                )
            if response_dipole_delta > response_dipole_tolerance:
                raise Route2SCFConvergenceError(
                    "Finite-resolution map-replay response dipole delta exceeds "
                    "nominal dipole tolerance.",
                    history=[],
                )
            field_potential_deltas.append(field_potential_delta)
            field_gradient_deltas.append(field_gradient_delta)
            response_monopole_deltas.append(response_monopole_delta)
            response_dipole_deltas.append(response_dipole_delta)
            field_potential_ulp.append(
                self._maximum_ulp_distance(
                    field_reference[:, 0],
                    replay_field[:, 0],
                )
            )
            field_gradient_ulp.append(
                self._maximum_ulp_distance(
                    field_reference[:, 1:],
                    replay_field[:, 1:],
                )
            )
            response_monopole_ulp.append(
                self._maximum_ulp_distance(
                    response_reference[:, 0],
                    replay_response[:, 0],
                )
            )
            response_dipole_ulp.append(
                self._maximum_ulp_distance(
                    response_reference[:, 1:],
                    replay_response[:, 1:],
                )
            )

        maximum_monopole_residual_e = float(
            np.max(np.abs(candidate_residual[:, 0]))
        )
        maximum_dipole_residual_e_angstrom = float(
            np.max(np.abs(candidate_residual[:, 1:]))
        )
        for replay_residual in replay_residuals:
            maximum_monopole_residual_e = max(
                maximum_monopole_residual_e,
                float(np.max(np.abs(replay_residual[:, 0]))),
            )
            maximum_dipole_residual_e_angstrom = max(
                maximum_dipole_residual_e_angstrom,
                float(np.max(np.abs(replay_residual[:, 1:]))),
            )

        if np.any(
            np.abs(candidate_residual[:, 0]) > policy.monopole_residual_ceiling_e
        ):
            raise Route2SCFConvergenceError(
                "Finite-resolution map-replay monopole residual exceeds policy "
                "ceiling.",
                history=[],
            )
        if np.any(
            np.abs(candidate_residual[:, 1:]) > policy.dipole_residual_ceiling_e_angstrom
        ):
            raise Route2SCFConvergenceError(
                "Finite-resolution map-replay dipole residual exceeds policy "
                "ceiling.",
                history=[],
            )
        for replay_residual in replay_residuals:
            if np.any(
                np.abs(replay_residual[:, 0]) > policy.monopole_residual_ceiling_e
            ):
                raise Route2SCFConvergenceError(
                    "Finite-resolution map-replay monopole residual exceeds policy "
                    "ceiling.",
                    history=[],
                )
            if np.any(
                np.abs(replay_residual[:, 1:])
                > policy.dipole_residual_ceiling_e_angstrom
            ):
                raise Route2SCFConvergenceError(
                    "Finite-resolution map-replay dipole residual exceeds policy "
                    "ceiling.",
                    history=[],
                )

        intrinsic_span_ev = float(
            max(observed_intrinsic_energies_ev)
            - min(observed_intrinsic_energies_ev)
        )
        observed_pcm_energies_ev = [
            energy * Hartree for energy in observed_pcm_polarization_hartree
        ]
        pcm_span_ev = float(
            max(observed_pcm_energies_ev) - min(observed_pcm_energies_ev)
        )
        electrostatic_span_ev = float(
            max(observed_electrostatic_ledger_ev)
            - min(observed_electrostatic_ledger_ev)
        )
        maximum_polarization_identity_error_ev = max(
            observed_polarization_identity_errors_ev
        )
        if (
            intrinsic_span_ev > policy.ledger_span_tolerance_ev
            or pcm_span_ev > policy.ledger_span_tolerance_ev
            or electrostatic_span_ev > policy.ledger_span_tolerance_ev
        ):
            raise Route2SCFConvergenceError(
                "Finite-resolution online/map-replay ledger span exceeded "
                "policy tolerance.",
                history=[],
            )
        if (
            maximum_polarization_identity_error_ev
            > self.settings.energy_identity_tolerance_ev
        ):
            raise Route2SCFConvergenceError(
                "Finite-resolution online/map-replay evidence failed the "
                "polarization-energy identity.",
                history=[],
            )
        if not cold_replay_field_arrays_identical:
            raise Route2SCFConvergenceError(
                "Finite-resolution map-replay field arrays are not byte-identical.",
                history=[],
            )
        if not cold_replay_response_arrays_identical:
            raise Route2SCFConvergenceError(
                "Finite-resolution map-replay response arrays are not byte-identical.",
                history=[],
            )

        return {
            "replay_count": policy.map_replay_count,
            "evaluation_count": policy.map_replay_count + 1,
            "includes_online_candidate": True,
            "cold_replay_field_arrays_identical": (
                cold_replay_field_arrays_identical
            ),
            "cold_replay_response_arrays_identical": (
                cold_replay_response_arrays_identical
            ),
            "field_sha256_by_evaluation": field_sha256_by_evaluation,
            "response_sha256_by_evaluation": response_sha256_by_evaluation,
            "maximum_online_to_replay_potential_delta_ev_per_e": max(
                field_potential_deltas
            ),
            "maximum_online_to_replay_gradient_delta_ev_per_e_angstrom": max(
                field_gradient_deltas
            ),
            "maximum_online_to_replay_monopole_response_delta_e": max(
                response_monopole_deltas
            ),
            "maximum_online_to_replay_dipole_response_delta_e_angstrom": max(
                response_dipole_deltas
            ),
            "maximum_online_to_replay_potential_ulp": max(field_potential_ulp),
            "maximum_online_to_replay_gradient_ulp": max(field_gradient_ulp),
            "maximum_online_to_replay_monopole_ulp": max(response_monopole_ulp),
            "maximum_online_to_replay_dipole_ulp": max(response_dipole_ulp),
            "maximum_monopole_residual_e": maximum_monopole_residual_e,
            "maximum_dipole_residual_e_angstrom": maximum_dipole_residual_e_angstrom,
            "intrinsic_ledger_span_ev": intrinsic_span_ev,
            "pcm_ledger_span_ev": pcm_span_ev,
            "electrostatic_ledger_span_ev": electrostatic_span_ev,
            "maximum_polarization_identity_error_ev": (
                maximum_polarization_identity_error_ev
            ),
        }

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

    def solve_coupled_state(
        self,
        atoms,
        calculator,
        gas_state,
        *,
        provider_cache_signature: Hashable,
        finite_resolution_runtime_identity: dict[str, object] | None = None,
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
        scf_convergence: dict[str, Any] = {}
        previous_energy_ev: float | None = None
        previous_density_residual: float | None = None
        previous_update_method: str | None = None
        history: list[Route2SCFHistoryRecord] = []
        density_history: list[np.ndarray] = []
        residual_history: list[np.ndarray] = []
        field_history: list[np.ndarray] = []
        intrinsic_energy_history: list[float] = []
        fixed_point_samples: list[FixedPointSample] = []
        best_iteration_state: Route2SCFIterationState | None = None

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
            residual = response_density - density
            density_residual = float(np.max(np.abs(residual)))
            monopole_residual_e = float(np.max(np.abs(residual[:, 0])))
            dipole_residual_e_angstrom = float(
                np.max(np.abs(residual[:, 1:]))
            )
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
                "monopole_residual_e": monopole_residual_e,
                "dipole_residual_e_angstrom": dipole_residual_e_angstrom,
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
            }
            history.append(record)
            density_history.append(np.array(density, copy=True))
            residual_history.append(np.array(residual, copy=True))
            field_history.append(np.array(field, copy=True))
            intrinsic_energy_history.append(current_energy_ev)
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
            if (
                monopole_residual_e <= settings.scf_density_tolerance
                and dipole_residual_e_angstrom
                <= cast(float, settings.scf_dipole_tolerance_e_angstrom)
                and energy_converged
            ):
                record["next_density_update"] = "converged"
                scf_convergence = {
                    "reason": "nominal-density-and-energy-v1",
                    "online_candidate_iteration": iteration,
                    "final_monopole_residual_e": monopole_residual_e,
                    "final_dipole_residual_e_angstrom": (
                        dipole_residual_e_angstrom
                    ),
                    "runtime_identity": None,
                    "history_window": None,
                    "fresh_map_replay": None,
                }
                break

            finite_resolution_policy = settings.scf_finite_resolution_policy
            if (
                finite_resolution_policy is not None
                and finite_resolution_runtime_identity is not None
                and not scf_convergence
            ):
                window = self._finite_resolution_candidate_window(
                    policy=finite_resolution_policy,
                    iteration_history=history,
                    density_history=density_history,
                    residual_history=residual_history,
                    field_history=field_history,
                    intrinsic_energy_history=intrinsic_energy_history,
                )
                if window is not None:
                    try:
                        replay = self._run_finite_resolution_map_replays(
                            atoms=atoms,
                            calculator=calculator,
                            gas_state=gas_state,
                            policy=finite_resolution_policy,
                            candidate_reaction_field=reaction_field,
                            candidate_density=np.array(density, copy=True),
                            candidate_field=np.array(field, copy=True),
                            candidate_drive_signature=(
                                self._reaction_field_signature(drive)
                            ),
                            candidate_response=np.array(
                                response_density,
                                copy=True,
                            ),
                            candidate_residual=np.array(residual, copy=True),
                            candidate_intrinsic_energy_ev=current_energy_ev,
                        )
                    except Route2SCFConvergenceError as exc:
                        raise Route2SCFConvergenceError(
                            str(exc),
                            history=history,
                            best_state=best_iteration_state,
                        ) from exc
                    record["next_density_update"] = (
                        "converged-finite-resolution"
                    )
                    scf_convergence = {
                        "reason": finite_resolution_policy.version,
                        "online_candidate_iteration": iteration,
                        "final_monopole_residual_e": (
                            monopole_residual_e
                        ),
                        "final_dipole_residual_e_angstrom": (
                            dipole_residual_e_angstrom
                        ),
                        "runtime_identity": dict(
                            finite_resolution_runtime_identity,
                        ),
                        "history_window": {
                            "start_iteration": int(window["start_iteration"]),
                            "end_iteration": int(window["end_iteration"]),
                            "root_monopole_span_e": (
                                window["root_monopole_span_e"]
                            ),
                            "root_dipole_span_e_angstrom": (
                                window["root_dipole_span_e_angstrom"]
                            ),
                            "residual_monopole_span_e": (
                                window["residual_monopole_span_e"]
                            ),
                            "residual_dipole_span_e_angstrom": (
                                window["residual_dipole_span_e_angstrom"]
                            ),
                            "potential_span_ev": window["potential_span_ev"],
                            "gradient_span_ev_per_angstrom": (
                                window["gradient_span_ev_per_angstrom"]
                            ),
                            "maximum_monopole_residual_e": (
                                window["maximum_monopole_residual_e"]
                            ),
                            "maximum_dipole_residual_e_angstrom": (
                                window[
                                    "maximum_dipole_residual_e_angstrom"
                                ]
                            ),
                            "maximum_energy_delta_ev": (
                                window["maximum_energy_delta_ev"]
                            ),
                            "intrinsic_energy_span_ev": (
                                window["intrinsic_energy_span_ev"]
                            ),
                        },
                        "fresh_map_replay": replay,
                    }
                    break
            fixed_point_samples.append(
                FixedPointSample(
                    density=density,
                    residual=residual,
                )
            )
            fixed_point_samples = fixed_point_samples[
                -(settings.scf_anderson_depth + 1) :
            ]
            if iteration < settings.scf_max_iterations:
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
                f"(monopole residual={last['monopole_residual_e']:.3e} e, "
                "dipole residual="
                f"{last['dipole_residual_e_angstrom']:.3e} e angstrom, "
                f"energy residual={last['energy_residual_ev']!r} eV, "
                f"minimum density residual={minimum_density_residual:.3e} "
                "(legacy raw-component scalar)).",
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
            scf_convergence=scf_convergence,
        )

    def solvent_correction_force(
        self,
        atoms,
        calculator,
        gas_state,
        coupled: Route2CoupledState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.settings
        finite_resolution_policy = settings.scf_finite_resolution_policy
        if (
            finite_resolution_policy is not None
            and coupled.scf_convergence.get("reason")
            == finite_resolution_policy.version
        ):
            raise RuntimeError(
                "Finite-resolution approximate fixed-point acceptance is "
                "energy-only; Route-2 forces require nominal SCF convergence."
            )
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
