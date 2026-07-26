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
from typing import Any

import numpy as np
from ase.units import Hartree

from .gto_density import external_field_to_density_order
from .route2_derivative import (
    assemble_total_solvation_coordinate_gradient,
    continuum_coupled_solvation_coordinate_gradient,
    fixed_cavity_energy_density_gradient,
)
from .route2_response import (
    UnmixedDensityResidualLinearization,
    solve_adjoint,
)


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

    def __post_init__(self) -> None:
        if not self.continuum_label:
            raise ValueError("A continuum label is required.")
        if not 0.0 < self.scf_mixing <= 1.0:
            raise ValueError("SCF mixing must lie in (0, 1].")
        positive = {
            "scf_density_tolerance": self.scf_density_tolerance,
            "scf_energy_tolerance_ev": self.scf_energy_tolerance_ev,
            "adjoint_relative_tolerance": self.adjoint_relative_tolerance,
            "adjoint_absolute_tolerance": self.adjoint_absolute_tolerance,
            "energy_identity_tolerance_ev": (
                self.energy_identity_tolerance_ev
            ),
            "force_state_energy_tolerance_ev": (
                self.force_state_energy_tolerance_ev
            ),
            "neutral_density_tolerance": self.neutral_density_tolerance,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
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


@dataclass(frozen=True)
class Route2CoupledState:
    """One converged geometry/provider state reusable by a public wrapper."""

    calculator_identity: int
    atomic_numbers: np.ndarray
    provider_cache_signature: Hashable
    positions_angstrom: np.ndarray
    reaction_field: Any
    density_coefficients: np.ndarray
    reaction_field_values_ev: np.ndarray
    solvent_state: Any
    polarization_energy_hartree: float
    energy_identity_error_ev: float
    cds_result: Any
    history: tuple[dict[str, float | int | None], ...]

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
    ) -> Route2CoupledState:
        settings = self.settings
        reaction_field = self.reaction_field_factory(atoms)
        density = self.validate_density(
            gas_state.density_coefficients,
            len(atoms),
            name="Gas MACE-POLAR density",
        )
        previous_energy_ev: float | None = None
        history: list[dict[str, float | int | None]] = []

        for iteration in range(1, settings.scf_max_iterations + 1):
            field = self._validate_field(
                reaction_field.apply_scf(density),
                len(atoms),
            )
            solvent_state, _ = calculator.polar_state(
                atoms,
                node_potential_ev=field[:, 0],
                node_gradient_ev_per_angstrom=field[:, 1:],
            )
            response_density = self.validate_density(
                solvent_state.density_coefficients,
                len(atoms),
                name="Field-polarized MACE-POLAR density",
            )
            density_residual = float(
                np.max(np.abs(response_density - density))
            )
            current_energy_ev = float(solvent_state.energy_ev)
            if not math.isfinite(current_energy_ev):
                raise RuntimeError(
                    "Field-polarized MACE-POLAR energy is non-finite."
                )
            energy_residual = (
                None
                if previous_energy_ev is None
                else abs(current_energy_ev - previous_energy_ev)
            )
            history.append(
                {
                    "iteration": iteration,
                    "density_residual_e": density_residual,
                    "energy_residual_ev": energy_residual,
                    "intrinsic_energy_ev": current_energy_ev,
                }
            )
            energy_converged = (
                energy_residual is None
                or energy_residual <= settings.scf_energy_tolerance_ev
            )
            if (
                density_residual <= settings.scf_density_tolerance
                and energy_converged
            ):
                break
            density = (
                (1.0 - settings.scf_mixing) * density
                + settings.scf_mixing * response_density
            )
            previous_energy_ev = current_energy_ev
        else:
            last = history[-1]
            raise RuntimeError(
                "MACE-POLAR/"
                f"{settings.continuum_label} reaction-field SCF did not "
                f"converge in {settings.scf_max_iterations} iterations "
                f"(density residual={last['density_residual_e']:.3e} e, "
                f"energy residual={last['energy_residual_ev']!r} eV)."
            )

        polarization_energy_hartree = float(
            reaction_field.scf_polarization_energy_hartree(density)
        )
        if not math.isfinite(polarization_energy_hartree):
            raise RuntimeError(
                f"{settings.continuum_label} polarization energy is "
                "non-finite."
            )
        paired_energy_ev = 0.5 * float(
            np.vdot(
                density,
                external_field_to_density_order(field),
            )
        )
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
            reaction_field_values_ev=field,
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
        field = coupled.reaction_field_values_ev
        density = coupled.density_coefficients
        solvent_state, _ = calculator.polar_state(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
            compute_forces=True,
        )
        response_density = self.validate_density(
            solvent_state.density_coefficients,
            len(atoms),
            name="Force-evaluation MACE-POLAR density",
        )
        force_state_residual = float(
            np.max(np.abs(response_density - density))
        )
        if force_state_residual > max(
            10.0 * settings.scf_density_tolerance,
            1.0e-10,
        ):
            raise RuntimeError(
                "The MACE-POLAR force state does not match the converged "
                f"density root (residual={force_state_residual:.3e} e)."
            )
        force_state_energy_error_ev = abs(
            float(solvent_state.energy_ev)
            - float(coupled.solvent_state.energy_ev)
        )
        if (
            not math.isfinite(force_state_energy_error_ev)
            or force_state_energy_error_ev
            > settings.force_state_energy_tolerance_ev
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
        continuum_gradient = (
            continuum_coupled_solvation_coordinate_gradient(
                coupled.reaction_field,
                density_response,
                density_coefficients=density,
                intrinsic_energy_field_gradient=intrinsic_gradient,
                adjoint_solution=adjoint.solution,
                adjoint_density_position_vjp=density_position_vjp,
                solvent_fixed_field_forces_ev_per_angstrom=solvent_forces,
                gas_forces_ev_per_angstrom=gas_forces,
            )
        )
        total = assemble_total_solvation_coordinate_gradient(
            continuum_gradient,
            coupled.cds_result.position_gradient_hartree_per_angstrom,
        )
        derivative = {
            "continuum_position_gradient_ev_per_angstrom": (
                continuum_gradient
            ),
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
                "maximum_inner_iterations": (
                    adjoint.maximum_inner_iterations
                ),
                "operator_applications": adjoint.operator_applications,
                "residual_callback_count": (
                    adjoint.residual_callback_count
                ),
                "residual_norm": adjoint.residual_norm,
                "relative_residual": adjoint.relative_residual,
            },
        }
        return (
            total.solvent_correction_forces_hartree_per_angstrom,
            derivative,
        )

    @staticmethod
    def energy_components(
        gas_state,
        coupled: Route2CoupledState,
    ) -> dict[str, float]:
        delta_e_solute = (
            float(coupled.solvent_state.energy_ev)
            - float(gas_state.energy_ev)
        ) / Hartree
        pcm_polarization = coupled.polarization_energy_hartree
        electrostatic = delta_e_solute + pcm_polarization
        cds_energy = float(coupled.cds_result.energy_hartree)
        total_energy = electrostatic + cds_energy
        return {
            "solute_polarization": delta_e_solute,
            "pcm_polarization": pcm_polarization,
            "electrostatic": electrostatic,
            "cds": cds_energy,
            "standard_state": 0.0,
            "delta_g_solv": total_energy,
        }


__all__ = [
    "Route2ContinuumEngine",
    "Route2CoupledState",
    "Route2EngineSettings",
]
