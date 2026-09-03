"""Shared stationary coupling core for MACE-POLAR-EF and PCM functionals.

The electronic and continuum implementations remain separate identities.  This
module owns only the common scalar, fixed-point solve, passivity gate, energy
ledger, and stationary-envelope first derivative:

    L(R, c, f) = E_electronic(R, f) + U_PCM(R, c) - <c, f>.

No concrete continuum backend is selected here and no public MAPLE capability
is registered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Protocol

import numpy as np
import torch

from .electrostatic_pairing import MACE_POLAR_L1_PAIRING
from .mace_polar_ef import (
    MACE_POLAR_EF_MAXIMUM_POSITIVE_CURVATURE,
    MACE_POLAR_EF_PASSIVITY_FIELD_STEP,
    MACEPolarEFConfig,
    MACEPolarEFEnergyModel,
)
from .route2_fixed_point import (
    SAFEGUARDED_ANDERSON_SOLVER,
    FixedPointSample,
    next_fixed_point_density,
)


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class BoundPCMFunctional(Protocol):
    """One linear PCM functional bound to one exact geometry."""

    @property
    def atom_count(self) -> int: ...

    @property
    def label(self) -> str: ...

    @property
    def identity_tolerance_ev(self) -> float: ...

    @property
    def field_replay_tolerance(self) -> float: ...

    @property
    def requested_solver_tolerance(self) -> float | None: ...

    @property
    def achieved_solver_residual(self) -> float | None: ...

    def drive_cartesian(
        self,
        source_raw: np.ndarray,
        *,
        warm_start: bool,
    ) -> np.ndarray: ...

    def energy_ev(self, source_raw: np.ndarray) -> float: ...

    def coordinate_gradient_ev_per_angstrom(
        self,
        source_raw: np.ndarray,
    ) -> np.ndarray: ...

    def runtime_provenance(self) -> Mapping[str, object]: ...


class PCMFunctional(Protocol):
    """Geometry-independent construction boundary for one PCM family."""

    @property
    def atom_count(self) -> int: ...

    @property
    def label(self) -> str: ...

    def bind_geometry(self, positions_angstrom: np.ndarray) -> BoundPCMFunctional: ...

    def as_identity(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class MACEPolarEFSCFSettings:
    source_residual_tolerance: float = 5.0e-5
    maximum_iterations: int = 100
    mixing: float = 0.5
    anderson_depth: int = 6
    anderson_regularization: float = 1.0e-12
    anderson_coefficient_l1_limit: float = 100.0
    anderson_step_ratio_limit: float = 100.0
    charge_drift_tolerance: float = 2.0e-5
    passivity_field_step_ev_per_e_angstrom: float = MACE_POLAR_EF_PASSIVITY_FIELD_STEP
    maximum_positive_uniform_field_curvature: float = (
        MACE_POLAR_EF_MAXIMUM_POSITIVE_CURVATURE
    )

    def __post_init__(self) -> None:
        positive_names = (
            "source_residual_tolerance",
            "mixing",
            "anderson_regularization",
            "anderson_coefficient_l1_limit",
            "anderson_step_ratio_limit",
            "charge_drift_tolerance",
            "passivity_field_step_ev_per_e_angstrom",
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        curvature = float(self.maximum_positive_uniform_field_curvature)
        if not math.isfinite(curvature) or curvature < 0.0:
            raise ValueError(
                "maximum_positive_uniform_field_curvature must be finite "
                "and nonnegative."
            )
        object.__setattr__(
            self,
            "maximum_positive_uniform_field_curvature",
            curvature,
        )
        if self.mixing > 1.0:
            raise ValueError("mixing must not exceed one.")
        for name in ("maximum_iterations", "anderson_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    def as_dict(self) -> dict[str, object]:
        return {
            "source_residual_tolerance": self.source_residual_tolerance,
            "maximum_iterations": self.maximum_iterations,
            "solver": SAFEGUARDED_ANDERSON_SOLVER,
            "mixing": self.mixing,
            "anderson_depth": self.anderson_depth,
            "anderson_regularization": self.anderson_regularization,
            "anderson_coefficient_l1_limit": (self.anderson_coefficient_l1_limit),
            "anderson_step_ratio_limit": self.anderson_step_ratio_limit,
            "charge_drift_tolerance": self.charge_drift_tolerance,
            "charge_policy": "raw-energy-gradient-fail-on-total-charge-drift",
            "passivity_field_step_ev_per_e_angstrom": (
                self.passivity_field_step_ev_per_e_angstrom
            ),
            "maximum_positive_uniform_field_curvature": (
                self.maximum_positive_uniform_field_curvature
            ),
        }


class MACEPolarEFStationaryConfig(Protocol):
    @property
    def electronic(self) -> MACEPolarEFConfig: ...

    @property
    def continuum_functional(self) -> PCMFunctional: ...

    @property
    def scf(self) -> MACEPolarEFSCFSettings: ...

    @property
    def configuration_sha256(self) -> str: ...

    def as_dict(self) -> dict[str, object]: ...


class MACEPolarEFPassivityError(RuntimeError):
    """Raised before continuum construction when electronic concavity fails."""

    def __init__(self, message: str, *, evidence: Mapping[str, object]) -> None:
        super().__init__(message)
        self.evidence = MappingProxyType(dict(evidence))


@dataclass(frozen=True, slots=True)
class MACEPolarEFStationaryResult:
    total_energy_ev: float
    electronic_energy_ev: float
    continuum_energy_ev: float
    source_field_pairing_ev: float
    coordinate_gradient_ev_per_angstrom: np.ndarray
    source_raw: np.ndarray
    field_cartesian: np.ndarray
    density_coefficients_diagnostic: np.ndarray
    source_residual: np.ndarray
    maximum_source_residual: float
    maximum_field_replay_difference: float
    iterations: int
    configuration_sha256: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        scalar_names = (
            "total_energy_ev",
            "electronic_energy_ev",
            "continuum_energy_ev",
            "source_field_pairing_ev",
            "maximum_source_residual",
            "maximum_field_replay_difference",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        atom_count: int | None = None
        for name, columns in (
            ("coordinate_gradient_ev_per_angstrom", 3),
            ("source_raw", 4),
            ("field_cartesian", 4),
            ("density_coefficients_diagnostic", 4),
            ("source_residual", 4),
        ):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                values.ndim != 2
                or values.shape[1] != columns
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f"{name} must be finite with shape (n_atoms, {columns})."
                )
            if atom_count is None:
                atom_count = values.shape[0]
            elif values.shape[0] != atom_count:
                raise ValueError("Stationary result atom counts must match.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer.")
        if not isinstance(self.configuration_sha256, str) or not (
            len(self.configuration_sha256) == 64
        ):
            raise ValueError("configuration_sha256 must be one SHA-256 digest.")
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


class MACEPolarEFStationaryCoupling:
    """Backend-neutral stationary MACE-POLAR-EF/PCM evaluator."""

    def __init__(self, config: MACEPolarEFStationaryConfig) -> None:
        electronic = getattr(config, "electronic", None)
        continuum = getattr(config, "continuum_functional", None)
        scf = getattr(config, "scf", None)
        if not isinstance(electronic, MACEPolarEFConfig):
            raise TypeError("stationary config requires MACEPolarEFConfig.")
        if not isinstance(scf, MACEPolarEFSCFSettings):
            raise TypeError("stationary config requires MACEPolarEFSCFSettings.")
        required_continuum = ("bind_geometry", "as_identity")
        if continuum is None or any(
            not callable(getattr(continuum, name, None)) for name in required_continuum
        ):
            raise TypeError("stationary config requires one PCM functional.")
        if int(getattr(continuum, "atom_count", -1)) != electronic.atom_count:
            raise ValueError("Electronic and continuum atom counts must match.")
        self.config = config
        self.electronic = MACEPolarEFEnergyModel(electronic)

    @property
    def atom_count(self) -> int:
        return self.config.electronic.atom_count

    def _electronic_source(
        self,
        positions_cuda: torch.Tensor,
        field_cartesian: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        field = torch.tensor(
            np.asarray(field_cartesian, dtype=np.float32),
            device=positions_cuda.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        energy, source, density = self.electronic.conjugate_source_torch(
            positions_cuda,
            field,
        )
        return (
            float(energy.detach().cpu()),
            np.asarray(source.detach().cpu(), dtype=float),
            np.asarray(density.detach().cpu(), dtype=float),
        )

    def _validated_source(self, source: np.ndarray) -> np.ndarray:
        values = np.asarray(source, dtype=float)
        if values.shape != (self.atom_count, 4) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Electronic source must be finite with shape ({self.atom_count}, 4)."
            )
        raw_charge = float(np.sum(values[:, 0]))
        target_charge = float(self.config.electronic.total_charge)
        if abs(raw_charge - target_charge) > self.config.scf.charge_drift_tolerance:
            raise RuntimeError(
                "MACE-POLAR-EF conjugate source violates the total-charge gate."
            )
        return np.array(values, dtype=float, copy=True)

    def _validated_field(self, field: np.ndarray, *, name: str) -> np.ndarray:
        values = np.asarray(field, dtype=float)
        if values.shape != (self.atom_count, 4) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} must be finite with shape ({self.atom_count}, 4)."
            )
        return np.array(values, dtype=float, copy=True)

    def _require_electronic_passivity(
        self,
        positions_angstrom: np.ndarray,
    ) -> dict[str, object]:
        audit = self.electronic.audit_uniform_field_passivity(
            positions_angstrom,
            field_step_ev_per_e_angstrom=(
                self.config.scf.passivity_field_step_ev_per_e_angstrom
            ),
            maximum_positive_curvature=(
                self.config.scf.maximum_positive_uniform_field_curvature
            ),
        )
        evidence = {
            **dict(audit.provenance),
            "field_step_ev_per_e_angstrom": (audit.field_step_ev_per_e_angstrom),
            "energy_hessian": audit.energy_hessian.tolist(),
            "energy_hessian_eigenvalues": (audit.energy_hessian_eigenvalues.tolist()),
            "maximum_positive_curvature": audit.maximum_positive_curvature,
            "tolerance": audit.tolerance,
            "passivity_passed": audit.passivity_passed,
        }
        if not audit.passivity_passed:
            label = str(self.config.continuum_functional.label)
            raise MACEPolarEFPassivityError(
                "MACE-POLAR-EF-v2 failed the required external-potential "
                f"concavity/passivity gate; refusing {label} coupling.",
                evidence=evidence,
            )
        return evidence

    def _canonical_positions(self, positions_angstrom: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_angstrom, dtype=float)
        if positions.shape != (self.atom_count, 3) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError(
                f"positions_angstrom must be finite with shape ({self.atom_count}, 3)."
            )
        return np.asarray(positions.astype(np.float32), dtype=float)

    def _solve_source(
        self,
        positions_angstrom: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        BoundPCMFunctional,
    ]:
        positions = self._canonical_positions(positions_angstrom)
        positions_cuda = torch.tensor(
            positions,
            device=self.electronic.device,
            dtype=torch.float32,
        )
        continuum = self.config.continuum_functional.bind_geometry(positions)
        zero_field = np.zeros((self.atom_count, 4), dtype=float)
        _energy, source, density = self._electronic_source(
            positions_cuda,
            zero_field,
        )
        source = self._validated_source(source)
        samples: list[FixedPointSample] = []

        for iteration in range(1, self.config.scf.maximum_iterations + 1):
            field = self._validated_field(
                continuum.drive_cartesian(source, warm_start=True),
                name="PCM reaction field",
            )
            _energy, mapped_source, density = self._electronic_source(
                positions_cuda,
                field,
            )
            mapped_source = self._validated_source(mapped_source)
            residual = mapped_source - source
            if not np.all(np.isfinite(residual)):
                raise FloatingPointError("Coupled source residual is non-finite.")
            samples.append(FixedPointSample(density=source, residual=residual))
            if float(np.max(np.abs(residual))) <= (
                self.config.scf.source_residual_tolerance
            ):
                return source, field, density, residual, iteration, continuum
            step = next_fixed_point_density(
                samples,
                solver=SAFEGUARDED_ANDERSON_SOLVER,
                mixing=self.config.scf.mixing,
                anderson_depth=self.config.scf.anderson_depth,
                anderson_regularization=self.config.scf.anderson_regularization,
                anderson_coefficient_l1_limit=(
                    self.config.scf.anderson_coefficient_l1_limit
                ),
                anderson_step_ratio_limit=(self.config.scf.anderson_step_ratio_limit),
            )
            source = self._validated_source(step.density)
        raise RuntimeError("MACE-POLAR-EF/PCM source equation did not converge.")

    def evaluate(
        self,
        positions_angstrom: np.ndarray,
    ) -> MACEPolarEFStationaryResult:
        input_positions = np.asarray(positions_angstrom, dtype=float)
        positions = self._canonical_positions(input_positions)
        passivity_evidence = self._require_electronic_passivity(positions)
        source, field, _density, _residual, iterations, continuum = self._solve_source(
            positions
        )

        electronic = self.electronic.evaluate(positions, field)
        fresh_source = self._validated_source(electronic.conjugate_source_raw)
        residual = fresh_source - source
        continuum_energy = float(continuum.energy_ev(source))
        continuum_gradient = np.asarray(
            continuum.coordinate_gradient_ev_per_angstrom(source),
            dtype=float,
        )
        if continuum_gradient.shape != (self.atom_count, 3) or not np.all(
            np.isfinite(continuum_gradient)
        ):
            raise ValueError(
                "PCM coordinate gradient must be finite with shape "
                f"({self.atom_count}, 3)."
            )
        pairing_value = MACE_POLAR_L1_PAIRING.pair(source, field)
        total_energy = electronic.energy_ev + continuum_energy - pairing_value
        coordinate_gradient = (
            electronic.coordinate_gradient_ev_per_angstrom + continuum_gradient
        )
        if not math.isfinite(total_energy) or not np.all(
            np.isfinite(coordinate_gradient)
        ):
            raise FloatingPointError("Coupled MACE-POLAR-EF/PCM result is non-finite.")

        identity_error = abs(continuum_energy - 0.5 * pairing_value)
        if identity_error > continuum.identity_tolerance_ev:
            raise RuntimeError("PCM half-coupling identity failed at the root.")
        field_check = self._validated_field(
            continuum.drive_cartesian(source, warm_start=False),
            name="PCM field replay",
        )
        maximum_field_replay_difference = float(np.max(np.abs(field_check - field)))
        if maximum_field_replay_difference > continuum.field_replay_tolerance:
            raise RuntimeError("PCM field stationarity replay failed at the root.")

        maximum_residual = float(np.max(np.abs(residual)))
        if maximum_residual > self.config.scf.source_residual_tolerance:
            raise RuntimeError(
                "Fresh MACE-POLAR-EF source does not satisfy the converged "
                "stationary residual tolerance."
            )
        provenance = {
            **self.config.as_dict(),
            "configuration_sha256": self.config.configuration_sha256,
            "continuum_runtime": dict(continuum.runtime_provenance()),
            "maximum_source_residual": maximum_residual,
            "maximum_field_replay_difference": (maximum_field_replay_difference),
            "continuum_requested_tolerance": (continuum.requested_solver_tolerance),
            "continuum_achieved_residual": continuum.achieved_solver_residual,
            "p1_complete": False,
            "p1_blocker": (
                "electronic-passivity-failed-for-supplied-checkpoint"
                if passivity_evidence.get("passivity_passed") is not True
                else "full-coupled-admission-not-run"
            ),
            "iterations": iterations,
            "continuum_half_coupling_identity_error_ev": identity_error,
            "coordinate_rounding_maximum_angstrom": float(
                np.max(np.abs(input_positions - positions))
            ),
            "result_interpretation": (
                "blocked-stationary-common-scalar-first-order-diagnostic"
            ),
            "forces_available": False,
            "coordinate_gradient_available": True,
            "hessian_available": False,
            "root_well_posedness_admitted": False,
            "field_domain_admitted": False,
            "chemical_accuracy_admitted": False,
            "publicly_registered": False,
            "release_admitted": False,
            "electronic_runtime": {
                **self.config.electronic.as_provenance(),
                **dict(self.electronic.runtime_provenance),
            },
            "electronic_passivity": passivity_evidence,
        }
        return MACEPolarEFStationaryResult(
            total_energy_ev=total_energy,
            electronic_energy_ev=electronic.energy_ev,
            continuum_energy_ev=continuum_energy,
            source_field_pairing_ev=pairing_value,
            coordinate_gradient_ev_per_angstrom=coordinate_gradient,
            source_raw=source,
            field_cartesian=field,
            density_coefficients_diagnostic=(
                electronic.density_coefficients_diagnostic
            ),
            source_residual=residual,
            maximum_source_residual=maximum_residual,
            maximum_field_replay_difference=maximum_field_replay_difference,
            iterations=iterations,
            configuration_sha256=self.config.configuration_sha256,
            provenance=provenance,
        )


__all__ = [
    "BoundPCMFunctional",
    "MACEPolarEFPassivityError",
    "MACEPolarEFSCFSettings",
    "MACEPolarEFStationaryConfig",
    "MACEPolarEFStationaryCoupling",
    "MACEPolarEFStationaryResult",
    "PCMFunctional",
    "canonical_sha256",
]
