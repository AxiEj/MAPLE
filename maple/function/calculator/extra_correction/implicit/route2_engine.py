"""Internal orchestration for self-consistent polarizable continuum coupling.

This module contains no continuum-provider construction or public dispatch.
One provider wrapper supplies a same-energy reaction-field factory and a
matching CDS evaluator; the engine owns only the common ML--SCF root,
implicit-function adjoint, and component ledger.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast

import numpy as np
from ase.units import Hartree

from ....route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    PCM_HALF_COUPLING_ONLY_V1,
    validate_route2_electrostatic_energy_ledger,
)
from .route2_electronic_model import (
    ATOMIC_L1_SOURCE_SPACE,
    FIELD_CONDITIONED_OPERATIONAL_ENERGY,
    AtomicL1SourceSpace,
    Route2ElectronicModel,
    resolve_route2_electronic_model,
)
from .route2_derivative import (
    assemble_total_solvation_coordinate_gradient,
    continuum_coupled_solvation_coordinate_gradient,
    fixed_cavity_energy_density_gradient,
    pcm_half_coupling_continuum_coordinate_gradient,
    pcm_half_coupling_energy_density_gradient,
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
from .route2_force_admission import (
    ContinuumSmoothnessContract,
    ForceAdmissionPolicy,
    ForcePESValidationContract,
    UNSPECIFIED_FORCE_PES_VALIDATION_CONTRACT,
    UNSPECIFIED_CONTINUUM_SMOOTHNESS_CONTRACT,
    evaluate_force_admission,
    force_energy_semantics_contract,
)
from .route2_nonuniform_response import (
    molecular_dipole_response_from_density_coefficients,
)
from .route2_response import (
    UnmixedDensityResidualLinearization,
    solve_adjoint,
)

SCF_ACTUAL_RESIDUAL_OBJECTIVE_FORMULA = (
    "max(monopole/tau_monopole,dipole/tau_dipole)"
)
SCF_ACCEPTED_RESIDUAL_SOURCE = "evaluated-actual-unmixed-physical-residual"
SCF_REJECTED_GROWTH_ACTION = (
    "reject-trial-rollback-prior-accepted-anchor-one-picard-restart"
)
SCF_FINITE_RESOLUTION_HISTORY_SOURCE = "accepted-solver-states-only"
SCF_ENERGY_RESIDUAL_SOURCE_MACE_FIELD = "field-conditioned-mace-energy-v1"
# Compatibility value retained in audit schemas; internal code uses the
# model-neutral name.
SCF_ENERGY_RESIDUAL_SOURCE_FIELD_CONDITIONED_MODEL = (
    SCF_ENERGY_RESIDUAL_SOURCE_MACE_FIELD
)
SCF_ENERGY_RESIDUAL_SOURCE_PCM_HALF_COUPLING = "pcm-half-coupling-v1"
FROZEN_SOURCE_CONVERGENCE_REASON = "frozen-source-no-fixed-point-v1"
Route2ResponseMode = Literal["frozen", "scf"]


class Route2SCFHistoryRecord(TypedDict):
    """One complete JSON-serializable fixed-point iteration record."""

    iteration: int
    density_residual_e: float
    monopole_residual_e: float
    monopole_residual_rms_e: float
    dipole_residual_e_angstrom: float
    dipole_residual_rms_e_angstrom: float
    source_residual_rms_normalized: float
    raw_response_charge_delta_e: float
    projected_total_charge_residual_e: float
    molecular_dipole_residual_vector_e_angstrom: list[float]
    molecular_dipole_residual_l2_e_angstrom: float
    reaction_potential_change_ev: float | None
    reaction_gradient_change_ev_per_angstrom: float | None
    energy_residual_ev: float | None
    intrinsic_energy_ev: float
    ledger_energy_ev: float
    energy_residual_source: str
    root_total_charge_e: float
    raw_response_total_charge_e: float
    response_charge_projection_max_e: float
    arrived_by: str | None
    anderson_history_reset: bool
    attempt_status: Literal[
        "accepted",
        "rejected-anderson-actual-residual-growth",
    ]
    accepted: bool
    rejected: bool
    actual_residual_objective: float
    actual_residual_growth_baseline_objective: float | None
    actual_residual_growth_ratio: float | None
    accepted_parent_attempt: int | None
    accepted_state_index: int | None
    solver_epoch: int
    rollback_anchor_attempt: int | None
    next_density_update: str | None
    fixed_point_history_size: int | None
    anderson_predicted_residual_l2: float | None
    anderson_coefficient_l1: float | None
    anderson_step_ratio_to_picard: float | None
    anderson_fallback_reason: str | None
    density_sha256: str
    response_sha256: str
    field_sha256: str


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
    scf_raw_response_charge_tolerance_e: float | None = None
    scf_total_charge_residual_tolerance_e: float | None = None
    scf_molecular_dipole_tolerance_e_angstrom: float | None = None

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
        optional_positive = {
            "scf_raw_response_charge_tolerance_e": (
                self.scf_raw_response_charge_tolerance_e
            ),
            "scf_total_charge_residual_tolerance_e": (
                self.scf_total_charge_residual_tolerance_e
            ),
            "scf_molecular_dipole_tolerance_e_angstrom": (
                self.scf_molecular_dipole_tolerance_e_angstrom
            ),
        }
        invalid_optional = [
            name
            for name, value in optional_positive.items()
            if value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            )
        ]
        if invalid_optional:
            raise ValueError(
                "Optional Route-2 aggregate SCF tolerances must be positive: "
                + ", ".join(invalid_optional)
                + "."
            )
        if (
            self.scf_finite_resolution_policy is not None
            and not isinstance(
                self.scf_finite_resolution_policy,
                Route2FiniteResolutionPolicy,
            )
        ):
            raise ValueError("Route-2 finite-resolution policy is malformed.")

    @property
    def effective_raw_response_charge_tolerance_e(self) -> float | None:
        """Configured pre-projection charge gate, if this profile admits one."""

        return (
            None
            if self.scf_raw_response_charge_tolerance_e is None
            else float(self.scf_raw_response_charge_tolerance_e)
        )

    @property
    def effective_total_charge_residual_tolerance_e(self) -> float | None:
        """Configured post-projection total-charge gate, if present."""

        return (
            None
            if self.scf_total_charge_residual_tolerance_e is None
            else float(self.scf_total_charge_residual_tolerance_e)
        )

    @property
    def effective_molecular_dipole_tolerance_e_angstrom(self) -> float | None:
        """Configured aggregate molecular-dipole gate, if present."""

        return (
            None
            if self.scf_molecular_dipole_tolerance_e_angstrom is None
            else float(self.scf_molecular_dipole_tolerance_e_angstrom)
        )


@dataclass(frozen=True)
class Route2CoupledState:
    """One same-root geometry/provider state reusable by a public wrapper."""

    electronic_model_identity: int
    electronic_model_family: str
    electronic_energy_semantics: str
    response_mode: Route2ResponseMode
    atomic_numbers: np.ndarray
    provider_cache_signature: Hashable
    positions_angstrom: np.ndarray
    initial_density_label: str
    initial_density_sha256: str
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
    source_field_pairing_ev: float
    energy_identity_error_ev: float
    cds_result: Any
    history: tuple[Route2SCFHistoryRecord, ...]
    scf_convergence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.response_mode not in {"frozen", "scf"}:
            raise ValueError(
                f"Unsupported Route-2 response mode: {self.response_mode!r}."
            )
        if not self.initial_density_label.strip():
            raise ValueError("Route-2 initial density label must be non-empty.")
        if not self.electronic_model_family.strip():
            raise ValueError("Route-2 electronic model family must be non-empty.")
        if not self.electronic_energy_semantics.strip():
            raise ValueError("Route-2 electronic energy semantics must be non-empty.")
        if len(self.initial_density_sha256) != 64:
            raise ValueError("Route-2 initial density SHA256 must be canonical.")
        for name in (
            "polarization_energy_hartree",
            "source_field_pairing_ev",
            "energy_identity_error_ev",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"Route-2 coupled-state {name} must be finite.")
            object.__setattr__(self, name, value)
        if self.energy_identity_error_ev < 0.0:
            raise ValueError("Route-2 energy identity error must be non-negative.")
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
    def fixed_point_applicable(self) -> bool:
        """Whether this state is the root of the learned response map."""

        return self.response_mode == "scf"

    @property
    def calculator_identity(self) -> int:
        """Compatibility alias for coupled states archived before adapter v1."""

        return self.electronic_model_identity

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
        electronic_model,
        atoms,
        *,
        provider_cache_signature: Hashable,
    ) -> bool:
        return (
            self.electronic_model_identity == electronic_model.cache_identity
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
    source_space: AtomicL1SourceSpace = ATOMIC_L1_SOURCE_SPACE

    def _resolve_electronic_model(self, candidate) -> Route2ElectronicModel:
        model = resolve_route2_electronic_model(candidate)
        if model.descriptor.source_space != self.source_space:
            raise TypeError(
                "Route-2 engine/model source-space mismatch: engine expects "
                f"{self.source_space.name!r}, received "
                f"{model.descriptor.source_space.name!r}."
            )
        if (
            model.descriptor.energy_semantics
            != FIELD_CONDITIONED_OPERATIONAL_ENERGY
        ):
            raise TypeError(
                "The fixed-point Route-2 engine accepts only explicitly "
                "operational field-conditioned electronic models; a common "
                "variational electronic functional requires the separate KKT "
                "engine."
            )
        return model

    def validate_source(
        self,
        values: np.ndarray,
        atom_count: int,
        *,
        name: str,
    ) -> np.ndarray:
        try:
            density = self.source_space.validate_source(
                values,
                atom_count=atom_count,
                name=name,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        total_charge = self.source_space.total_charge(density)
        charge_error = total_charge - self.settings.scf_total_charge_e
        if abs(charge_error) > self.settings.neutral_density_tolerance:
            raise RuntimeError(
                f"{name} violates the fixed-total-charge constraint "
                f"(actual={total_charge:.6e} e, expected="
                f"{self.settings.scf_total_charge_e:.6e} e)."
            )
        return density

    def validate_density(
        self,
        values: np.ndarray,
        atom_count: int,
        *,
        name: str,
    ) -> np.ndarray:
        """Compatibility alias for the original density-only engine API."""

        return self.validate_source(values, atom_count, name=name)

    def _validate_field(
        self,
        values: np.ndarray,
        atom_count: int,
    ) -> np.ndarray:
        try:
            return self.source_space.validate_field(
                values,
                atom_count=atom_count,
                name=(
                    f"The {self.settings.continuum_label} reaction field"
                ),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

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
                raise TypeError(
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

    def _initial_source(
        self,
        gas_state,
        atom_count: int,
        *,
        initial_density_coefficients: np.ndarray | None,
        initial_density_label: str | None,
    ) -> tuple[np.ndarray, str, str]:
        """Validate, constrain, label, and hash one Route-2 source."""

        if initial_density_coefficients is None:
            initial_density = self.validate_source(
                gas_state.density_coefficients,
                atom_count,
                name="Gas electronic source",
            )
            label = "gas-electronic-source"
            if initial_density_label is not None:
                raise ValueError(
                    "initial_density_label requires an explicit "
                    "initial_density_coefficients array."
                )
        else:
            initial_density = self.validate_source(
                initial_density_coefficients,
                atom_count,
                name="Route-2 supplied initial density",
            )
            label = str(initial_density_label or "caller-supplied-density")
            if not label.strip():
                raise ValueError("initial_density_label must be non-empty.")
        density = project_density_total_charge(
            initial_density,
            total_charge_e=self.settings.scf_total_charge_e,
        )
        return density, label, self._array_sha256(density)

    def _checked_polarization_energy(
        self,
        reaction_field,
        density: np.ndarray,
        field: np.ndarray,
    ) -> tuple[float, float, float]:
        """Return PCM energy, full pairing, and half-coupling identity error."""

        polarization_energy_hartree = float(
            reaction_field.scf_polarization_energy_hartree(density)
        )
        if not math.isfinite(polarization_energy_hartree):
            raise RuntimeError(
                f"{self.settings.continuum_label} polarization energy is "
                "non-finite."
            )
        source_field_pairing_ev = float(self.source_space.pair(density, field))
        paired_energy_ev = 0.5 * source_field_pairing_ev
        provider_energy_ev = polarization_energy_hartree * Hartree
        identity_error_ev = abs(paired_energy_ev - provider_energy_ev)
        if identity_error_ev > self.settings.energy_identity_tolerance_ev:
            raise RuntimeError(
                f"{self.settings.continuum_label} reaction field failed the "
                "polarization-energy identity "
                f"(absolute error={identity_error_ev:.3e} eV)."
            )
        return (
            polarization_energy_hartree,
            source_field_pairing_ev,
            identity_error_ev,
        )

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
    def _normalized_actual_residual_objective(
        *,
        monopole_residual_e: float,
        dipole_residual_e_angstrom: float,
        density_tolerance_e: float,
        dipole_tolerance_e_angstrom: float,
    ) -> float:
        return float(
            max(
                monopole_residual_e / density_tolerance_e,
                dipole_residual_e_angstrom / dipole_tolerance_e_angstrom,
            )
        )

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
    def _polarize(
        electronic_model: Route2ElectronicModel,
        atoms,
        drive: ReactionFieldDrive,
        **kwargs,
    ):
        return electronic_model.evaluate_state(atoms, drive, **kwargs)

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
        solver_epoch = window_records[0]["solver_epoch"]
        if any(
            record["arrived_by"] != SAFEGUARDED_ANDERSON_SOLVER
            or record["anderson_history_reset"]
            or record["solver_epoch"] != solver_epoch
            for record in window_records
        ):
            return None
        if (
            window_records[0]["attempt_status"] != "accepted"
            or not window_records[0]["accepted"]
            or window_records[0]["rejected"]
        ):
            return None
        for index in range(1, len(window_records)):
            prior = window_records[index - 1]
            current = window_records[index]
            if (
                not current["accepted"]
                or current["rejected"]
                or current["attempt_status"] != "accepted"
                or current["accepted_parent_attempt"] != prior["iteration"]
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
        electronic_model: Route2ElectronicModel,
        gas_state,
        policy: Route2FiniteResolutionPolicy,
        candidate_reaction_field,
        candidate_density: np.ndarray,
        candidate_field: np.ndarray,
        candidate_drive_signature: tuple[Any, ...],
        candidate_response: np.ndarray,
        candidate_residual: np.ndarray,
        candidate_intrinsic_energy_ev: float,
        electrostatic_energy_ledger: str = (
            LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
        ),
    ) -> dict[str, Any]:
        selected_energy_ledger = validate_route2_electrostatic_energy_ledger(
            electrostatic_energy_ledger
        )
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
            electronic_model.descriptor.source_space.pair(
                candidate_density, candidate_field
            )
        )
        candidate_provider_energy_ev = candidate_pcm_polarization_hartree * Hartree
        candidate_polarization_identity_error_ev = abs(
            candidate_paired_energy_ev - candidate_provider_energy_ev
        )
        candidate_electrostatic_ledger_ev = candidate_provider_energy_ev
        if selected_energy_ledger == LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1:
            candidate_electrostatic_ledger_ev += (
                candidate_intrinsic_energy_ev - float(gas_state.energy_ev)
            )
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
                electronic_model,
                atoms,
                replay_drive,
            )
            replay_response_raw = self.validate_source(
                replay_solvent_state.density_coefficients,
                atom_count,
                name="Map-replay electronic source",
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
                electronic_model.descriptor.source_space.pair(
                    candidate_density, replay_field
                )
            )
            provider_energy_ev = replay_pcm_polarization * Hartree
            replay_polarization_identity_error_ev = abs(
                paired_energy_ev - provider_energy_ev
            )
            replay_electrostatic_ledger_ev = provider_energy_ev
            if selected_energy_ledger == LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1:
                replay_electrostatic_ledger_ev += (
                    replay_intrinsic - float(gas_state.energy_ev)
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
    def gas_state(electronic_model, atoms, *, need_forces: bool) -> Any:
        model = resolve_route2_electronic_model(electronic_model)
        state = model.cached_state(atoms, require_forces=need_forces)
        if state is None or (
            need_forces
            and getattr(
                state,
                "fixed_field_forces_ev_per_angstrom",
                None,
            )
            is None
        ):
            state, _ = model.evaluate_state(
                atoms, None, compute_forces=need_forces
            )
        return state

    def solve_coupled_state(
        self,
        atoms,
        electronic_model,
        gas_state,
        *,
        provider_cache_signature: Hashable,
        finite_resolution_runtime_identity: dict[str, object] | None = None,
        initial_density_coefficients: np.ndarray | None = None,
        initial_density_label: str | None = None,
        electrostatic_energy_ledger: str = (
            LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
        ),
    ) -> Route2CoupledState:
        model = self._resolve_electronic_model(electronic_model)
        settings = self.settings
        selected_energy_ledger = validate_route2_electrostatic_energy_ledger(
            electrostatic_energy_ledger
        )
        energy_residual_source = (
            SCF_ENERGY_RESIDUAL_SOURCE_PCM_HALF_COUPLING
            if selected_energy_ledger == PCM_HALF_COUPLING_ONLY_V1
            else SCF_ENERGY_RESIDUAL_SOURCE_FIELD_CONDITIONED_MODEL
        )
        reaction_field = self.reaction_field_factory(atoms)
        density, initial_label, initial_density_sha256 = self._initial_source(
            gas_state,
            len(atoms),
            initial_density_coefficients=initial_density_coefficients,
            initial_density_label=initial_density_label,
        )
        positions_angstrom = np.asarray(atoms.get_positions(), dtype=float)
        raw_charge_tolerance_e = (
            settings.effective_raw_response_charge_tolerance_e
        )
        total_charge_residual_tolerance_e = (
            settings.effective_total_charge_residual_tolerance_e
        )
        molecular_dipole_tolerance_e_angstrom = (
            settings.effective_molecular_dipole_tolerance_e_angstrom
        )
        dipole_component_tolerance_e_angstrom = cast(
            float,
            settings.scf_dipole_tolerance_e_angstrom,
        )
        scf_convergence: dict[str, Any] = {}
        previous_energy_ev: float | None = None
        previous_update_method: str | None = None
        history: list[Route2SCFHistoryRecord] = []
        accepted_attempts: list[Route2SCFHistoryRecord] = []
        accepted_density_history: list[np.ndarray] = []
        accepted_residual_history: list[np.ndarray] = []
        accepted_field_history: list[np.ndarray] = []
        accepted_intrinsic_energy_history: list[float] = []
        fixed_point_samples: list[FixedPointSample] = []
        best_iteration_state: Route2SCFIterationState | None = None
        last_accepted_actual_residual: float | None = None
        last_accepted_sample: FixedPointSample | None = None
        last_accepted_field: np.ndarray | None = None
        last_accepted_density: np.ndarray | None = None
        last_accepted_residual: np.ndarray | None = None
        last_accepted_intrinsic_energy: float | None = None
        last_accepted_attempt: int | None = None
        last_accepted_history: Route2SCFHistoryRecord | None = None
        accepted_state_count = 0
        solver_epoch = 0

        for iteration in range(1, settings.scf_max_iterations + 1):
            drive = self._reaction_field_drive(
                reaction_field,
                density,
                len(atoms),
            )
            field = drive.density_dual_field_ev
            solvent_state, _ = self._polarize(
                model,
                atoms,
                drive,
            )
            raw_response_density = self.validate_source(
                solvent_state.density_coefficients,
                len(atoms),
                name="Field-conditioned electronic source",
            )
            response_density = project_density_total_charge(
                raw_response_density,
                total_charge_e=settings.scf_total_charge_e,
            )
            residual = response_density - density
            density_residual = float(np.max(np.abs(residual)))
            monopole_residual_e = float(np.max(np.abs(residual[:, 0])))
            dipole_residual_e_angstrom = float(np.max(np.abs(residual[:, 1:])))
            monopole_residual_rms_e = float(
                np.sqrt(np.mean(np.square(residual[:, 0])))
            )
            dipole_residual_rms_e_angstrom = float(
                np.sqrt(np.mean(np.square(residual[:, 1:])))
            )
            scaled_source_residual = np.concatenate(
                (
                    residual[:, 0].reshape(-1) / settings.scf_density_tolerance,
                    residual[:, 1:].reshape(-1)
                    / dipole_component_tolerance_e_angstrom,
                )
            )
            source_residual_rms_normalized = float(
                np.sqrt(np.mean(np.square(scaled_source_residual)))
            )
            root_total_charge_e = float(np.sum(density[:, 0]))
            raw_response_total_charge_e = float(
                np.sum(raw_response_density[:, 0])
            )
            raw_response_charge_delta_e = (
                raw_response_total_charge_e - root_total_charge_e
            )
            projected_total_charge_residual_e = abs(
                float(np.sum(response_density[:, 0])) - root_total_charge_e
            )
            molecular_dipole_residual = (
                molecular_dipole_response_from_density_coefficients(
                    positions_angstrom,
                    residual,
                )
            )
            molecular_dipole_residual_l2_e_angstrom = float(
                np.linalg.norm(molecular_dipole_residual)
            )
            current_intrinsic_energy_ev = float(solvent_state.energy_ev)
            if not math.isfinite(current_intrinsic_energy_ev):
                raise RuntimeError("Field-conditioned model energy is non-finite.")
            if selected_energy_ledger == PCM_HALF_COUPLING_ONLY_V1:
                current_energy_ev = float(
                    reaction_field.scf_polarization_energy_hartree(density)
                ) * Hartree
            else:
                current_energy_ev = current_intrinsic_energy_ev
            if not math.isfinite(current_energy_ev):
                raise RuntimeError(
                    "Selected Route-2 SCF energy ledger is non-finite."
                )

            actual_residual_objective = self._normalized_actual_residual_objective(
                monopole_residual_e=monopole_residual_e,
                dipole_residual_e_angstrom=dipole_residual_e_angstrom,
                density_tolerance_e=settings.scf_density_tolerance,
                dipole_tolerance_e_angstrom=cast(
                    float,
                    settings.scf_dipole_tolerance_e_angstrom,
                ),
            )
            growth_baseline = None
            if (
                settings.scf_solver == SAFEGUARDED_ANDERSON_SOLVER
                and previous_update_method == SAFEGUARDED_ANDERSON_SOLVER
                and last_accepted_actual_residual is not None
            ):
                growth_baseline = last_accepted_actual_residual
            growth_ratio = (
                None
                if growth_baseline is None or growth_baseline == 0.0
                else actual_residual_objective / growth_baseline
            )
            should_reject_anderson = (
                growth_baseline is not None
                and actual_residual_objective
                > settings.scf_anderson_residual_growth_limit * growth_baseline
            )
            energy_residual = (
                None
                if previous_energy_ev is None
                else abs(current_energy_ev - previous_energy_ev)
            )
            reaction_potential_change_ev: float | None = None
            reaction_gradient_change_ev_per_angstrom: float | None = None
            if last_accepted_field is not None:
                field_change = field - last_accepted_field
                reaction_potential_change_ev = float(
                    np.max(np.abs(field_change[:, 0]))
                )
                reaction_gradient_change_ev_per_angstrom = float(
                    np.max(np.abs(field_change[:, 1:]))
                )

            record: Route2SCFHistoryRecord = {
                "iteration": iteration,
                "density_residual_e": density_residual,
                "monopole_residual_e": monopole_residual_e,
                "monopole_residual_rms_e": monopole_residual_rms_e,
                "dipole_residual_e_angstrom": dipole_residual_e_angstrom,
                "dipole_residual_rms_e_angstrom": (
                    dipole_residual_rms_e_angstrom
                ),
                "source_residual_rms_normalized": (
                    source_residual_rms_normalized
                ),
                "raw_response_charge_delta_e": raw_response_charge_delta_e,
                "projected_total_charge_residual_e": (
                    projected_total_charge_residual_e
                ),
                "molecular_dipole_residual_vector_e_angstrom": [
                    float(value) for value in molecular_dipole_residual
                ],
                "molecular_dipole_residual_l2_e_angstrom": (
                    molecular_dipole_residual_l2_e_angstrom
                ),
                "reaction_potential_change_ev": reaction_potential_change_ev,
                "reaction_gradient_change_ev_per_angstrom": (
                    reaction_gradient_change_ev_per_angstrom
                ),
                "energy_residual_ev": energy_residual,
                "intrinsic_energy_ev": current_intrinsic_energy_ev,
                "ledger_energy_ev": current_energy_ev,
                "energy_residual_source": energy_residual_source,
                "root_total_charge_e": root_total_charge_e,
                "raw_response_total_charge_e": raw_response_total_charge_e,
                "response_charge_projection_max_e": float(
                    np.max(np.abs(response_density - raw_response_density))
                ),
                "arrived_by": previous_update_method,
                "anderson_history_reset": bool(should_reject_anderson),
                "attempt_status": (
                    "rejected-anderson-actual-residual-growth"
                    if should_reject_anderson
                    else "accepted"
                ),
                "accepted": False,
                "rejected": bool(should_reject_anderson),
                "actual_residual_objective": actual_residual_objective,
                "actual_residual_growth_baseline_objective": growth_baseline,
                "actual_residual_growth_ratio": growth_ratio,
                "accepted_parent_attempt": last_accepted_attempt,
                "accepted_state_index": None,
                "solver_epoch": solver_epoch,
                "rollback_anchor_attempt": (
                    last_accepted_attempt if should_reject_anderson else None
                ),
                "next_density_update": None,
                "fixed_point_history_size": None,
                "anderson_predicted_residual_l2": None,
                "anderson_coefficient_l1": None,
                "anderson_step_ratio_to_picard": None,
                "anderson_fallback_reason": None,
                "density_sha256": self._array_sha256(density),
                "response_sha256": self._array_sha256(response_density),
                "field_sha256": self._array_sha256(field),
            }
            history.append(record)

            if should_reject_anderson:
                if (
                    last_accepted_sample is None
                    or last_accepted_field is None
                    or last_accepted_density is None
                    or last_accepted_residual is None
                    or last_accepted_intrinsic_energy is None
                    or last_accepted_history is None
                ):
                    raise RuntimeError(
                        "Failed to recover a prior accepted Route-2 SCF state "
                        "for Anderson rollback."
                    )
                fallback_step = next_fixed_point_density(
                    [last_accepted_sample],
                    solver=DAMPED_PICARD_SOLVER,
                    mixing=settings.scf_mixing,
                    anderson_depth=settings.scf_anderson_depth,
                    anderson_regularization=(
                        settings.scf_anderson_regularization
                    ),
                    anderson_coefficient_l1_limit=(
                        settings.scf_anderson_coefficient_l1_limit
                    ),
                    anderson_step_ratio_limit=(
                        settings.scf_anderson_step_ratio_limit
                    ),
                )
                fixed_point_samples = [last_accepted_sample]
                solver_epoch += 1
                anchor_record = cast(
                    Route2SCFHistoryRecord,
                    dict(last_accepted_history),
                )
                anchor_record["solver_epoch"] = solver_epoch
                accepted_attempts = [anchor_record]
                accepted_density_history = [
                    np.array(last_accepted_density, copy=True),
                ]
                accepted_residual_history = [
                    np.array(last_accepted_residual, copy=True),
                ]
                accepted_field_history = [
                    np.array(last_accepted_field, copy=True),
                ]
                accepted_intrinsic_energy_history = [
                    float(last_accepted_intrinsic_energy)
                ]
                record.update(
                    {
                        "next_density_update": fallback_step.method,
                        "fixed_point_history_size": fallback_step.history_size,
                        "anderson_fallback_reason": (
                            "anderson-actual-residual-growth-rejected"
                        ),
                    }
                )
                density = fallback_step.density
                previous_update_method = DAMPED_PICARD_SOLVER
                continue

            accepted_sample = FixedPointSample(
                density=density,
                residual=residual,
            )
            fixed_point_samples.append(accepted_sample)
            fixed_point_samples = fixed_point_samples[
                -(settings.scf_anderson_depth + 1) :
            ]
            accepted_attempts.append(record)
            accepted_density_history.append(np.array(density, copy=True))
            accepted_residual_history.append(np.array(residual, copy=True))
            accepted_field_history.append(np.array(field, copy=True))
            accepted_intrinsic_energy_history.append(current_energy_ev)
            record["accepted"] = True
            record["rejected"] = False
            record["accepted_parent_attempt"] = last_accepted_attempt
            record["accepted_state_index"] = accepted_state_count
            accepted_state_count += 1
            last_accepted_sample = accepted_sample
            last_accepted_field = np.array(field, copy=True)
            last_accepted_density = np.array(density, copy=True)
            last_accepted_residual = np.array(residual, copy=True)
            last_accepted_actual_residual = actual_residual_objective
            last_accepted_attempt = iteration
            last_accepted_history = record
            last_accepted_intrinsic_energy = current_energy_ev
            if (
                best_iteration_state is None
                or density_residual < best_iteration_state.density_residual_e
            ):
                best_iteration_state = Route2SCFIterationState(
                    iteration=iteration,
                    density_residual_e=density_residual,
                    intrinsic_energy_ev=current_intrinsic_energy_ev,
                    density_coefficients=density,
                    response_density_coefficients=response_density,
                    reaction_field_values_ev=field,
                    model_local_field_values_ev=drive.model_local_field_ev,
                    model_field_features=drive.model_field_features,
                )

            if (
                not scf_convergence
                and monopole_residual_e <= settings.scf_density_tolerance
                and dipole_residual_e_angstrom
                <= dipole_component_tolerance_e_angstrom
                and source_residual_rms_normalized <= 1.0
                and (
                    raw_charge_tolerance_e is None
                    or abs(raw_response_charge_delta_e)
                    <= raw_charge_tolerance_e
                )
                and (
                    total_charge_residual_tolerance_e is None
                    or projected_total_charge_residual_e
                    <= total_charge_residual_tolerance_e
                )
                and (
                    molecular_dipole_tolerance_e_angstrom is None
                    or molecular_dipole_residual_l2_e_angstrom
                    <= molecular_dipole_tolerance_e_angstrom
                )
            ):
                energy_converged = (
                    energy_residual is None
                    and not settings.scf_require_two_energy_samples
                ) or (
                    energy_residual is not None
                    and energy_residual <= settings.scf_energy_tolerance_ev
                )
                if energy_converged:
                    record["next_density_update"] = "converged"
                    scf_convergence = {
                        "reason": "nominal-density-and-energy-v1",
                        "online_candidate_iteration": iteration,
                        "final_monopole_residual_e": monopole_residual_e,
                        "final_monopole_residual_rms_e": (
                            monopole_residual_rms_e
                        ),
                        "final_dipole_residual_e_angstrom": (
                            dipole_residual_e_angstrom
                        ),
                        "final_dipole_residual_rms_e_angstrom": (
                            dipole_residual_rms_e_angstrom
                        ),
                        "final_source_residual_rms_normalized": (
                            source_residual_rms_normalized
                        ),
                        "final_raw_response_charge_delta_e": (
                            raw_response_charge_delta_e
                        ),
                        "final_projected_total_charge_residual_e": (
                            projected_total_charge_residual_e
                        ),
                        "final_molecular_dipole_residual_vector_e_angstrom": [
                            float(value) for value in molecular_dipole_residual
                        ],
                        "final_molecular_dipole_residual_l2_e_angstrom": (
                            molecular_dipole_residual_l2_e_angstrom
                        ),
                        "residual_tolerances": {
                            "monopole_max_e": settings.scf_density_tolerance,
                            "dipole_component_max_e_angstrom": (
                                dipole_component_tolerance_e_angstrom
                            ),
                            "source_rms_normalized": 1.0,
                            "raw_response_charge_delta_e": (
                                raw_charge_tolerance_e
                            ),
                            "projected_total_charge_residual_e": (
                                total_charge_residual_tolerance_e
                            ),
                            "molecular_dipole_l2_e_angstrom": (
                                molecular_dipole_tolerance_e_angstrom
                            ),
                            "energy_delta_ev": settings.scf_energy_tolerance_ev,
                        },
                        "nominal_residual_gate_passed": True,
                        "final_reaction_potential_change_ev": (
                            reaction_potential_change_ev
                        ),
                        "final_reaction_gradient_change_ev_per_angstrom": (
                            reaction_gradient_change_ev_per_angstrom
                        ),
                        "final_energy_residual_ev": energy_residual,
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
                    iteration_history=accepted_attempts,
                    density_history=accepted_density_history,
                    residual_history=accepted_residual_history,
                    field_history=accepted_field_history,
                    intrinsic_energy_history=accepted_intrinsic_energy_history,
                )
                if window is not None:
                    try:
                        replay = self._run_finite_resolution_map_replays(
                            atoms=atoms,
                            electronic_model=model,
                            gas_state=gas_state,
                            policy=finite_resolution_policy,
                            candidate_reaction_field=reaction_field,
                            candidate_density=np.array(density, copy=True),
                            candidate_field=np.array(field, copy=True),
                            candidate_drive_signature=(
                                self._reaction_field_signature(drive)
                            ),
                            candidate_response=np.array(response_density, copy=True),
                            candidate_residual=np.array(residual, copy=True),
                            candidate_intrinsic_energy_ev=(
                                current_intrinsic_energy_ev
                            ),
                            electrostatic_energy_ledger=selected_energy_ledger,
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
                        "final_monopole_residual_e": monopole_residual_e,
                        "final_monopole_residual_rms_e": (
                            monopole_residual_rms_e
                        ),
                        "final_dipole_residual_e_angstrom": (
                            dipole_residual_e_angstrom
                        ),
                        "final_dipole_residual_rms_e_angstrom": (
                            dipole_residual_rms_e_angstrom
                        ),
                        "final_source_residual_rms_normalized": (
                            source_residual_rms_normalized
                        ),
                        "final_raw_response_charge_delta_e": (
                            raw_response_charge_delta_e
                        ),
                        "final_projected_total_charge_residual_e": (
                            projected_total_charge_residual_e
                        ),
                        "final_molecular_dipole_residual_vector_e_angstrom": [
                            float(value) for value in molecular_dipole_residual
                        ],
                        "final_molecular_dipole_residual_l2_e_angstrom": (
                            molecular_dipole_residual_l2_e_angstrom
                        ),
                        "residual_tolerances": {
                            "monopole_max_e": settings.scf_density_tolerance,
                            "dipole_component_max_e_angstrom": (
                                dipole_component_tolerance_e_angstrom
                            ),
                            "source_rms_normalized": 1.0,
                            "raw_response_charge_delta_e": (
                                raw_charge_tolerance_e
                            ),
                            "projected_total_charge_residual_e": (
                                total_charge_residual_tolerance_e
                            ),
                            "molecular_dipole_l2_e_angstrom": (
                                molecular_dipole_tolerance_e_angstrom
                            ),
                            "energy_delta_ev": settings.scf_energy_tolerance_ev,
                        },
                        "nominal_residual_gate_passed": False,
                        "final_reaction_potential_change_ev": (
                            reaction_potential_change_ev
                        ),
                        "final_reaction_gradient_change_ev_per_angstrom": (
                            reaction_gradient_change_ev_per_angstrom
                        ),
                        "final_energy_residual_ev": energy_residual,
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
                                window["maximum_dipole_residual_e_angstrom"]
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

            previous_energy_ev = current_energy_ev
            if iteration < settings.scf_max_iterations:
                step = next_fixed_point_density(
                    fixed_point_samples,
                    solver=settings.scf_solver,
                    mixing=settings.scf_mixing,
                    anderson_depth=settings.scf_anderson_depth,
                    anderson_regularization=(
                        settings.scf_anderson_regularization
                    ),
                    anderson_coefficient_l1_limit=(
                        settings.scf_anderson_coefficient_l1_limit
                    ),
                    anderson_step_ratio_limit=(
                        settings.scf_anderson_step_ratio_limit
                    ),
                )
                record.update(
                    {
                        "next_density_update": step.method,
                        "fixed_point_history_size": step.history_size,
                        "anderson_predicted_residual_l2": (
                            step.predicted_residual_l2
                        ),
                        "anderson_coefficient_l1": step.coefficient_l1,
                        "anderson_step_ratio_to_picard": (
                            step.step_ratio_to_picard
                        ),
                        "anderson_fallback_reason": step.fallback_reason,
                    }
                )
                density = step.density
                previous_update_method = step.method

        else:
            last = history[-1]
            accepted_minima = [
                entry["density_residual_e"] for entry in history if entry["accepted"]
            ]
            if accepted_minima:
                minimum_density_residual = min(accepted_minima)
            else:
                minimum_density_residual = min(
                    entry["density_residual_e"] for entry in history
                )
            raise Route2SCFConvergenceError(
                f"{model.descriptor.model_family}/"
                f"{settings.continuum_label} reaction-field SCF did not "
                f"converge in {settings.scf_max_iterations} iterations "
                f"(monopole residual={last['monopole_residual_e']:.3e} e, "
                "dipole residual="
                f"{last['dipole_residual_e_angstrom']:.3e} e angstrom, "
                "normalized RMS source residual="
                f"{last['source_residual_rms_normalized']:.3e}, "
                "raw charge delta="
                f"{last['raw_response_charge_delta_e']:.3e} e, "
                "projected charge residual="
                f"{last['projected_total_charge_residual_e']:.3e} e, "
                "molecular dipole residual="
                f"{last['molecular_dipole_residual_l2_e_angstrom']:.3e} "
                "e angstrom, "
                f"energy residual={last['energy_residual_ev']!r} eV, "
                f"minimum density residual={minimum_density_residual:.3e} "
                f"(energy ledger={last['energy_residual_source']})).",
                history=history,
                best_state=best_iteration_state,
            )

        (
            polarization_energy_hartree,
            source_field_pairing_ev,
            identity_error_ev,
        ) = (
            self._checked_polarization_energy(
                reaction_field,
                density,
                field,
            )
        )

        cds_result = self.cds_evaluator(atoms)
        return Route2CoupledState(
            electronic_model_identity=model.cache_identity,
            electronic_model_family=model.descriptor.model_family,
            electronic_energy_semantics=model.descriptor.energy_semantics,
            response_mode="scf",
            atomic_numbers=np.asarray(atoms.numbers, dtype=int).copy(),
            provider_cache_signature=provider_cache_signature,
            positions_angstrom=np.asarray(
                atoms.get_positions(),
                dtype=float,
            ).copy(),
            initial_density_label=initial_label,
            initial_density_sha256=initial_density_sha256,
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
            source_field_pairing_ev=source_field_pairing_ev,
            energy_identity_error_ev=identity_error_ev,
            cds_result=cds_result,
            history=tuple(history),
            scf_convergence=scf_convergence,
        )

    def solve_frozen_source_state(
        self,
        atoms,
        electronic_model,
        gas_state,
        *,
        provider_cache_signature: Hashable,
        initial_density_coefficients: np.ndarray | None = None,
        initial_density_label: str | None = None,
        electrostatic_energy_ledger: str = PCM_HALF_COUPLING_ONLY_V1,
    ) -> Route2CoupledState:
        """Evaluate a frozen electronic source in one continuum state.

        This path intentionally does not call the field-conditioned electronic
        model.  There is no learned fixed point and no implied mutual
        polarization.  The method validates the selected ledger identity, but
        the provider/profile boundary owns whether frozen response is admitted;
        the returned state itself is reusable by either versioned ledger.
        """

        model = self._resolve_electronic_model(electronic_model)
        validate_route2_electrostatic_energy_ledger(electrostatic_energy_ledger)
        reaction_field = self.reaction_field_factory(atoms)
        density, source_label, source_sha256 = self._initial_source(
            gas_state,
            len(atoms),
            initial_density_coefficients=initial_density_coefficients,
            initial_density_label=initial_density_label,
        )
        drive = self._reaction_field_drive(
            reaction_field,
            density,
            len(atoms),
        )
        field = drive.density_dual_field_ev
        (
            polarization_energy_hartree,
            source_field_pairing_ev,
            identity_error_ev,
        ) = (
            self._checked_polarization_energy(
                reaction_field,
                density,
                field,
            )
        )
        cds_result = self.cds_evaluator(atoms)
        return Route2CoupledState(
            electronic_model_identity=model.cache_identity,
            electronic_model_family=model.descriptor.model_family,
            electronic_energy_semantics=model.descriptor.energy_semantics,
            response_mode="frozen",
            atomic_numbers=np.asarray(atoms.numbers, dtype=int).copy(),
            provider_cache_signature=provider_cache_signature,
            positions_angstrom=np.asarray(
                atoms.get_positions(),
                dtype=float,
            ).copy(),
            initial_density_label=source_label,
            initial_density_sha256=source_sha256,
            reaction_field=reaction_field,
            density_coefficients=density,
            response_density_coefficients=density,
            reaction_field_values_ev=field,
            model_local_field_values_ev=drive.model_local_field_ev,
            model_field_features=drive.model_field_features,
            reaction_field_projector=drive.projector,
            model_field_gauge=drive.model_field_gauge,
            model_field_gauge_reference_ev=(drive.model_field_gauge_reference_ev),
            solvent_state=gas_state,
            polarization_energy_hartree=polarization_energy_hartree,
            source_field_pairing_ev=source_field_pairing_ev,
            energy_identity_error_ev=identity_error_ev,
            cds_result=cds_result,
            history=(),
            scf_convergence={
                "reason": FROZEN_SOURCE_CONVERGENCE_REASON,
                "fixed_point_applicable": False,
                "source_label": source_label,
            },
        )

    def solvent_correction_force(
        self,
        atoms,
        electronic_model,
        gas_state,
        coupled: Route2CoupledState,
        *,
        force_admission_continuum: ContinuumSmoothnessContract = (
            UNSPECIFIED_CONTINUUM_SMOOTHNESS_CONTRACT
        ),
        force_admission_policy: ForceAdmissionPolicy | None = None,
        multi_start_root_agreement: bool | None = None,
        force_admission_pes_validation: ForcePESValidationContract = (
            UNSPECIFIED_FORCE_PES_VALIDATION_CONTRACT
        ),
        electrostatic_energy_ledger: str = (
            LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
        ),
    ) -> tuple[np.ndarray, dict[str, Any]]:
        model = self._resolve_electronic_model(electronic_model)
        if not coupled.fixed_point_applicable:
            raise TypeError(
                "Frozen-source Route-2 states are energy-only; the current "
                "force adjoint is defined only for learned SCF fixed points."
            )
        if coupled.electronic_model_identity != model.cache_identity:
            raise RuntimeError(
                "Route-2 derivative model does not own the coupled state."
            )
        selected_energy_ledger = validate_route2_electrostatic_energy_ledger(
            electrostatic_energy_ledger
        )
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
        drive = ReactionFieldDrive(
            density_dual_field_ev=field,
            model_local_field_ev=coupled.model_local_field_values_ev,
            model_field_features=coupled.model_field_features,
            projector=coupled.reaction_field_projector,
            model_field_gauge=coupled.model_field_gauge,
            model_field_gauge_reference_ev=(
                coupled.model_field_gauge_reference_ev
            ),
        )
        needs_fixed_field_forces = (
            selected_energy_ledger == LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
        )
        solvent_state, _ = model.evaluate_state(
            atoms,
            drive,
            compute_forces=needs_fixed_field_forces,
        )
        response_density = project_density_total_charge(
            self.validate_source(
                solvent_state.density_coefficients,
                len(atoms),
                name="Force-evaluation electronic source",
            ),
            total_charge_e=settings.scf_total_charge_e,
        )
        force_state_residual = float(np.max(np.abs(response_density - density)))
        if force_state_residual > max(
            10.0 * settings.scf_density_tolerance,
            1.0e-10,
        ):
            raise RuntimeError(
                "The electronic-model force state does not match the converged "
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
                "The electronic-model force evaluation does not reproduce the "
                "converged intrinsic energy "
                f"(absolute error={force_state_energy_error_ev:.3e} eV)."
            )

        density_response = model.linearize_source_response(atoms, drive)
        if selected_energy_ledger == PCM_HALF_COUPLING_ONLY_V1:
            physical_rhs = pcm_half_coupling_energy_density_gradient(
                coupled.reaction_field,
                reaction_field_values=field,
                pairing=model.descriptor.source_space.pairing,
            )
        else:
            intrinsic_gradient = model.field_conditioned_energy_field_gradient(
                atoms, drive
            )
            physical_rhs = fixed_cavity_energy_density_gradient(
                coupled.reaction_field,
                reaction_field_values=field,
                intrinsic_energy_field_gradient=intrinsic_gradient,
                pairing=model.descriptor.source_space.pairing,
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
        density_position_vjp = model.source_position_vjp(
            atoms,
            drive,
            source_cotangent=adjoint.solution,
        )
        if selected_energy_ledger == PCM_HALF_COUPLING_ONLY_V1:
            continuum_gradient = (
                pcm_half_coupling_continuum_coordinate_gradient(
                    coupled.reaction_field,
                    density_response,
                    density_coefficients=density,
                    adjoint_solution=adjoint.solution,
                    adjoint_density_position_vjp=density_position_vjp,
                    pairing=model.descriptor.source_space.pairing,
                )
            )
        else:
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
                    "Electronic model omitted the gas or fixed-field force partial."
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
                pairing=model.descriptor.source_space.pairing,
            )
        total = assemble_total_solvation_coordinate_gradient(
            continuum_gradient,
            coupled.cds_result.position_gradient_hartree_per_angstrom,
        )
        effective_force_admission_policy = force_admission_policy
        if effective_force_admission_policy is None:
            effective_force_admission_policy = ForceAdmissionPolicy(
                maximum_primal_monopole_residual_e=(
                    settings.scf_density_tolerance
                ),
                maximum_primal_dipole_residual_e_angstrom=cast(
                    float,
                    settings.scf_dipole_tolerance_e_angstrom,
                ),
                maximum_adjoint_relative_residual=(
                    settings.adjoint_relative_tolerance
                ),
                maximum_continuum_identity_error_ev=(
                    settings.energy_identity_tolerance_ev
                ),
            )
        final_scf_record = coupled.history[-1]
        force_admission = evaluate_force_admission(
            residual,
            nominal_root=(
                coupled.scf_convergence.get("reason")
                == "nominal-density-and-energy-v1"
            ),
            primal_monopole_residual_e=(
                final_scf_record["monopole_residual_e"]
            ),
            primal_dipole_residual_e_angstrom=(
                final_scf_record["dipole_residual_e_angstrom"]
            ),
            adjoint_relative_residual=adjoint.relative_residual,
            continuum_identity_error_ev=coupled.energy_identity_error_ev,
            continuum=force_admission_continuum,
            multi_start_root_agreement=multi_start_root_agreement,
            energy_semantics=force_energy_semantics_contract(
                selected_energy_ledger
            ),
            pes_validation=force_admission_pes_validation,
            policy=effective_force_admission_policy,
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
            "force_admission": force_admission.as_dict(),
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
        electrostatic_energy_ledger: str = (
            LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
        ),
    ) -> dict[str, float]:
        """Compose one versioned Route-2 scalar-energy ledger.

        ``pcm-half-coupling-only-v1`` retains the adapter-provided fixed-point
        source only for the continuum and reports the PCM
        half-coupling plus frozen CDS.  It deliberately excludes the
        field-conditioned electronic-model energy difference from the ledger.
        """

        selected_energy_ledger = validate_route2_electrostatic_energy_ledger(
            electrostatic_energy_ledger
        )
        field_conditioned_model_energy_change = (
            float(solvent_energy_ev) - float(gas_energy_ev)
        ) / Hartree
        delta_e_solute = (
            0.0
            if selected_energy_ledger == PCM_HALF_COUPLING_ONLY_V1
            else field_conditioned_model_energy_change
        )
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
        *,
        electrostatic_energy_ledger: str = (
            LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
        ),
    ) -> dict[str, float]:
        return cls.compose_energy_components(
            gas_energy_ev=float(gas_state.energy_ev),
            solvent_energy_ev=float(coupled.solvent_state.energy_ev),
            polarization_energy_hartree=(coupled.polarization_energy_hartree),
            cds_energy_hartree=float(coupled.cds_result.energy_hartree),
            electrostatic_energy_ledger=electrostatic_energy_ledger,
        )


__all__ = [
    "Route2ContinuumEngine",
    "Route2CoupledState",
    "Route2EngineSettings",
]
