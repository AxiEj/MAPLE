"""Diffuse reciprocal reaction-field block for the Route-2 V0-AQ-C route.

This module is the electrostatic block of the V0-AQ-C scalar, not a complete
implicit-solvent model.  It discretizes one smooth solvent-occupancy field
``m`` on a periodic Cartesian grid and uses it in the *same* symmetric
finite-volume Poisson operator for

``G_reac[rho, m] = 1/2 integral rho (phi_eps[m] - phi_vac)``.

The returned reaction potential is the density derivative of that scalar.  The
returned occupancy derivative is the envelope derivative of the same scalar.
No atomic-radius switching, source/receiver mismatch, response symmetrization,
or post-hoc half coupling is involved.

A total V0-AQ-C calculation additionally needs a stationary auxiliary
electronic functional and a source-bound liquid/cavity/short-range/dispersion
functional.  This fixed-occupancy reaction block deliberately supplies none
of those missing physical terms and cannot report a solvation free energy or
chemical accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .route2_v0_bulk_liquid_state_source import Route2V0BulkLiquidStateSource
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_DIFFUSE_DIELECTRIC_CONSTRUCTION = "route2-v0-diffuse-local-dielectric-v1"
V0_DIFFUSE_DIELECTRIC_SCOPE = "fixed-occupancy-reaction-block-only-v1"


def _finite_scalar(value: object, *, name: str) -> float:
    """Return one finite scalar without accepting booleans as numbers."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_scalar(value: object, *, name: str) -> float:
    """Return one finite strictly positive scalar."""

    result = _finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _immutable_real_grid(
    values: np.ndarray,
    *,
    shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    """Validate and freeze one finite real grid field."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real grid field.") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _mean_zero(values: np.ndarray) -> np.ndarray:
    """Return a mutable zero-average copy in the periodic potential gauge."""

    result = np.array(values, dtype=float, copy=True)
    result -= float(np.mean(result))
    return result


def _validated_iteration_cap(value: int | None, *, grid: RegularCartesianGrid) -> int:
    """Return a finite CG cap without an unbounded default iteration loop."""

    if value is None:
        return max(200, 12 * max(grid.shape) ** 2)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("Diffuse-dielectric CG iteration cap must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError("Diffuse-dielectric CG iteration cap must be positive.")
    return result


@dataclass(frozen=True)
class Route2V0DiffuseDielectricReactionState:
    """One fixed-occupancy reaction-field solve from the declared scalar."""

    charge_density_e_per_bohr3: np.ndarray
    occupancy: np.ndarray
    vacuum_potential_hartree_per_e: np.ndarray
    solvated_potential_hartree_per_e: np.ndarray
    reaction_potential_hartree_per_e: np.ndarray
    vacuum_energy_hartree: float
    solvated_energy_hartree: float
    reaction_energy_hartree: float
    vacuum_residual_inf: float
    solvated_residual_inf: float
    bulk_dielectric_constant: float
    construction: str = V0_DIFFUSE_DIELECTRIC_CONSTRUCTION
    response_scope: str = V0_DIFFUSE_DIELECTRIC_SCOPE

    def __post_init__(self) -> None:
        shape = np.asarray(self.charge_density_e_per_bohr3).shape
        if len(shape) != 3:
            raise ValueError("Diffuse-dielectric state charge density must be 3D.")
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        density = _immutable_real_grid(
            self.charge_density_e_per_bohr3,
            shape=grid_shape,
            name="Diffuse-dielectric state charge density",
        )
        occupancy = _immutable_real_grid(
            self.occupancy,
            shape=grid_shape,
            name="Diffuse-dielectric state occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError("Diffuse-dielectric state occupancy must lie in [0, 1].")
        vacuum_potential = _immutable_real_grid(
            self.vacuum_potential_hartree_per_e,
            shape=grid_shape,
            name="Diffuse-dielectric state vacuum potential",
        )
        solvated_potential = _immutable_real_grid(
            self.solvated_potential_hartree_per_e,
            shape=grid_shape,
            name="Diffuse-dielectric state solvated potential",
        )
        reaction_potential = _immutable_real_grid(
            self.reaction_potential_hartree_per_e,
            shape=grid_shape,
            name="Diffuse-dielectric state reaction potential",
        )
        vacuum_energy = _finite_scalar(
            self.vacuum_energy_hartree,
            name="State vacuum energy",
        )
        solvated_energy = _finite_scalar(
            self.solvated_energy_hartree,
            name="State solvated energy",
        )
        reaction_energy = _finite_scalar(
            self.reaction_energy_hartree,
            name="State reaction energy",
        )
        vacuum_residual = _finite_scalar(
            self.vacuum_residual_inf,
            name="State vacuum residual",
        )
        solvated_residual = _finite_scalar(
            self.solvated_residual_inf,
            name="State solvated residual",
        )
        if vacuum_residual < 0.0 or solvated_residual < 0.0:
            raise ValueError("Diffuse-dielectric state residuals must be nonnegative.")
        dielectric = _positive_scalar(
            self.bulk_dielectric_constant,
            name="Diffuse-dielectric state bulk dielectric constant",
        )
        if dielectric <= 1.0:
            raise ValueError(
                "Diffuse-dielectric state bulk dielectric must exceed one."
            )
        if self.construction != V0_DIFFUSE_DIELECTRIC_CONSTRUCTION:
            raise ValueError("Unsupported diffuse-dielectric state construction.")
        if self.response_scope != V0_DIFFUSE_DIELECTRIC_SCOPE:
            raise ValueError("Unsupported diffuse-dielectric state response scope.")
        object.__setattr__(self, "charge_density_e_per_bohr3", density)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "vacuum_potential_hartree_per_e", vacuum_potential)
        object.__setattr__(self, "solvated_potential_hartree_per_e", solvated_potential)
        object.__setattr__(self, "reaction_potential_hartree_per_e", reaction_potential)
        object.__setattr__(self, "vacuum_energy_hartree", vacuum_energy)
        object.__setattr__(self, "solvated_energy_hartree", solvated_energy)
        object.__setattr__(self, "reaction_energy_hartree", reaction_energy)
        object.__setattr__(self, "vacuum_residual_inf", vacuum_residual)
        object.__setattr__(self, "solvated_residual_inf", solvated_residual)
        object.__setattr__(self, "bulk_dielectric_constant", dielectric)


@dataclass(frozen=True)
class Route2V0DiffuseLocalDielectricOperator:
    """A self-adjoint finite-volume reaction operator for one smooth cavity.

    The face dielectric is the arithmetic mean of adjacent cell dielectrics.
    This keeps the discrete Poisson matrix symmetric and makes the occupancy
    derivative exact for the declared finite-volume energy.  It is a numerical
    discretization choice, not an error-selected cavity parameter.
    """

    grid: RegularCartesianGrid
    occupancy: np.ndarray
    bulk_dielectric_constant: float
    bulk_state_source: Route2V0BulkLiquidStateSource | None = None
    cg_relative_tolerance: float = 1.0e-12
    cg_max_iterations: int | None = None
    construction: str = V0_DIFFUSE_DIELECTRIC_CONSTRUCTION
    response_scope: str = V0_DIFFUSE_DIELECTRIC_SCOPE
    _occupancy: np.ndarray = field(init=False, repr=False, compare=False)
    _dielectric: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError("Diffuse dielectric requires a regular Cartesian grid.")
        if self.construction != V0_DIFFUSE_DIELECTRIC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 diffuse dielectric construction.")
        if self.response_scope != V0_DIFFUSE_DIELECTRIC_SCOPE:
            raise ValueError("Unsupported Route-2 diffuse dielectric response scope.")
        occupancy = _immutable_real_grid(
            self.occupancy,
            shape=self.grid.shape,
            name="Diffuse-dielectric occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError("Diffuse-dielectric occupancy must lie in [0, 1].")
        dielectric_bulk = _positive_scalar(
            self.bulk_dielectric_constant,
            name="Diffuse-dielectric bulk dielectric constant",
        )
        if dielectric_bulk <= 1.0:
            raise ValueError("Diffuse-dielectric bulk dielectric must exceed one.")
        source = self.bulk_state_source
        if source is not None:
            if not isinstance(source, Route2V0BulkLiquidStateSource):
                raise TypeError(
                    "Diffuse dielectric bulk state must be a Route-2 bulk-state source."
                )
            if not math.isclose(
                source.static_dielectric_constant,
                dielectric_bulk,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "Diffuse-dielectric bulk dielectric must equal the attached "
                    "bulk-state source."
                )
        tolerance = _finite_scalar(
            self.cg_relative_tolerance,
            name="Diffuse-dielectric CG relative tolerance",
        )
        if not 0.0 < tolerance < 1.0:
            raise ValueError(
                "Diffuse-dielectric CG relative tolerance must lie in (0, 1)."
            )
        iterations = _validated_iteration_cap(self.cg_max_iterations, grid=self.grid)
        dielectric = 1.0 + (dielectric_bulk - 1.0) * occupancy
        dielectric = _immutable_real_grid(
            dielectric,
            shape=self.grid.shape,
            name="Diffuse-dielectric cell dielectric",
        )
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "bulk_dielectric_constant", dielectric_bulk)
        object.__setattr__(self, "cg_relative_tolerance", tolerance)
        object.__setattr__(self, "cg_max_iterations", iterations)
        object.__setattr__(self, "_occupancy", occupancy)
        object.__setattr__(self, "_dielectric", dielectric)

    @property
    def dielectric_field(self) -> np.ndarray:
        """Return the immutable cell dielectric ``1 + (eps_bulk-1) m``."""

        return self._dielectric

    @property
    def is_bulk_state_bound(self) -> bool:
        """Return whether the dielectric limit has independent state provenance."""

        return self.bulk_state_source is not None

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false: no liquid/cavity/dispersion scalar exists here."""

        return False

    def require_bulk_state_source(self) -> Route2V0BulkLiquidStateSource:
        """Return the source-bound state or fail before physical promotion."""

        if self.bulk_state_source is None:
            raise ValueError(
                "Diffuse dielectric has no bulk-state provenance and is only a "
                "numerical control."
            )
        return self.bulk_state_source

    def _validate_neutral_density(
        self,
        values: np.ndarray,
        *,
        neutrality_relative_tolerance: float,
    ) -> np.ndarray:
        tolerance = _finite_scalar(
            neutrality_relative_tolerance,
            name="Diffuse-dielectric neutrality tolerance",
        )
        if not 0.0 < tolerance < 1.0:
            raise ValueError(
                "Diffuse-dielectric neutrality tolerance must lie in (0, 1)."
            )
        density = _immutable_real_grid(
            values,
            shape=self.grid.shape,
            name="Diffuse-dielectric charge density",
        )
        volume = self.grid.volume_element_bohr3
        charge = float(volume * np.sum(density))
        scale = max(1.0, float(volume * np.sum(np.abs(density))))
        if abs(charge) > tolerance * scale:
            raise ValueError(
                "Diffuse-dielectric charge density must be neutral; a periodic "
                "background is not an admitted V0 convention."
            )
        return density

    def _apply_poisson(
        self, potential: np.ndarray, dielectric: np.ndarray
    ) -> np.ndarray:
        """Apply the symmetric periodic ``-div(eps grad)`` finite-volume matrix."""

        result = np.zeros(self.grid.shape, dtype=float)
        for axis, spacing in enumerate(self.grid.spacing_bohr):
            forward_potential = np.roll(potential, -1, axis=axis)
            face_dielectric = 0.5 * (dielectric + np.roll(dielectric, -1, axis=axis))
            forward_flux = face_dielectric * (forward_potential - potential) / spacing
            result -= (forward_flux - np.roll(forward_flux, 1, axis=axis)) / spacing
        return _mean_zero(result)

    def _conjugate_gradient(
        self,
        *,
        right_hand_side: np.ndarray,
        dielectric: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Solve the mean-zero periodic Poisson system without a hidden gauge."""

        rhs = _mean_zero(right_hand_side)
        rhs_norm = float(np.linalg.norm(rhs.ravel()))
        if rhs_norm == 0.0:
            return np.zeros(self.grid.shape, dtype=float), 0.0
        target_norm = self.cg_relative_tolerance * max(1.0, rhs_norm)
        solution = np.zeros(self.grid.shape, dtype=float)
        residual = np.array(rhs, copy=True)
        direction = np.array(residual, copy=True)
        residual_norm_squared = float(np.vdot(residual, residual))
        iterations = self.cg_max_iterations
        if iterations is None:
            raise RuntimeError(
                "Diffuse-dielectric CG iteration cap was not normalized."
            )
        for _ in range(iterations):
            applied_direction = self._apply_poisson(direction, dielectric)
            denominator = float(np.vdot(direction, applied_direction))
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise RuntimeError("Diffuse-dielectric CG lost positive curvature.")
            step = residual_norm_squared / denominator
            solution = _mean_zero(solution + step * direction)
            residual = _mean_zero(residual - step * applied_direction)
            next_norm_squared = float(np.vdot(residual, residual))
            if not math.isfinite(next_norm_squared):
                raise RuntimeError("Diffuse-dielectric CG residual is non-finite.")
            if math.sqrt(next_norm_squared) <= target_norm:
                residual_inf = float(np.max(np.abs(residual)))
                return solution, residual_inf
            coefficient = next_norm_squared / residual_norm_squared
            direction = _mean_zero(residual + coefficient * direction)
            residual_norm_squared = next_norm_squared
        residual_inf = float(np.max(np.abs(residual)))
        raise RuntimeError(
            "Diffuse-dielectric CG did not converge within the declared cap; "
            f"residual_inf={residual_inf:.6e}."
        )

    def _solve_potential(
        self,
        density: np.ndarray,
        *,
        dielectric: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        potential, residual = self._conjugate_gradient(
            right_hand_side=4.0 * math.pi * density,
            dielectric=dielectric,
        )
        potential = _mean_zero(potential)
        potential.setflags(write=False)
        return potential, residual

    def _electrostatic_energy_hartree(
        self,
        density: np.ndarray,
        potential: np.ndarray,
    ) -> float:
        energy = (
            0.5 * self.grid.volume_element_bohr3 * float(np.sum(density * potential))
        )
        if not math.isfinite(energy):
            raise RuntimeError("Diffuse-dielectric electrostatic energy is non-finite.")
        return energy

    def solve(
        self,
        charge_density_e_per_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> Route2V0DiffuseDielectricReactionState:
        """Solve vacuum and diffuse-continuum fields from one reaction scalar."""

        density = self._validate_neutral_density(
            charge_density_e_per_bohr3,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        vacuum_dielectric = np.ones(self.grid.shape, dtype=float)
        vacuum_potential, vacuum_residual = self._solve_potential(
            density,
            dielectric=vacuum_dielectric,
        )
        solvated_potential, solvated_residual = self._solve_potential(
            density,
            dielectric=self._dielectric,
        )
        reaction_potential = np.array(solvated_potential - vacuum_potential, copy=True)
        reaction_potential = _mean_zero(reaction_potential)
        reaction_potential.setflags(write=False)
        vacuum_energy = self._electrostatic_energy_hartree(density, vacuum_potential)
        solvated_energy = self._electrostatic_energy_hartree(
            density, solvated_potential
        )
        reaction_energy = self._electrostatic_energy_hartree(
            density,
            reaction_potential,
        )
        numerical_scale = max(1.0, abs(vacuum_energy), abs(solvated_energy))
        if reaction_energy > 32.0 * self.cg_relative_tolerance * numerical_scale:
            raise RuntimeError(
                "Diffuse-dielectric reaction scalar violates passivity despite "
                "epsilon >= 1."
            )
        return Route2V0DiffuseDielectricReactionState(
            charge_density_e_per_bohr3=density,
            occupancy=self._occupancy,
            vacuum_potential_hartree_per_e=vacuum_potential,
            solvated_potential_hartree_per_e=solvated_potential,
            reaction_potential_hartree_per_e=reaction_potential,
            vacuum_energy_hartree=vacuum_energy,
            solvated_energy_hartree=solvated_energy,
            reaction_energy_hartree=reaction_energy,
            vacuum_residual_inf=vacuum_residual,
            solvated_residual_inf=solvated_residual,
            bulk_dielectric_constant=self.bulk_dielectric_constant,
        )

    def reaction_potential_hartree_per_e(
        self,
        charge_density_e_per_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> np.ndarray:
        """Return ``delta G_reac / delta rho`` from the same scalar."""

        return self.solve(
            charge_density_e_per_bohr3,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        ).reaction_potential_hartree_per_e

    def reaction_energy_hartree(
        self,
        charge_density_e_per_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> float:
        """Return ``1/2 int rho (phi_eps - phi_vac)`` in Hartree."""

        return self.solve(
            charge_density_e_per_bohr3,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        ).reaction_energy_hartree

    def reaction_pairing_hartree(
        self,
        left_charge_density_e_per_bohr3: np.ndarray,
        right_charge_density_e_per_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> float:
        """Return ``int left * deltaG/dright`` for a reciprocity check."""

        left = self._validate_neutral_density(
            left_charge_density_e_per_bohr3,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        state = self.solve(
            right_charge_density_e_per_bohr3,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        return float(
            self.grid.volume_element_bohr3
            * np.sum(left * state.reaction_potential_hartree_per_e)
        )

    def occupancy_energy_gradient_hartree(
        self,
        state: Route2V0DiffuseDielectricReactionState,
    ) -> np.ndarray:
        """Return the envelope derivative of reaction energy with respect to ``m``.

        The arithmetic face mean in :meth:`_apply_poisson` gives exactly

        ``dG/dm_i = -(eps_bulk-1) dV/(16*pi) sum_axes[(D+phi_i)^2 +
        (D+phi_{i-e_axis})^2]``.

        This derivative contains no ``dphi/dm`` solve because the electrostatic
        field is stationary in the same scalar.
        """

        if not isinstance(state, Route2V0DiffuseDielectricReactionState):
            raise TypeError("Diffuse-dielectric occupancy derivative requires a state.")
        if state.charge_density_e_per_bohr3.shape != self.grid.shape:
            raise ValueError(
                "Diffuse-dielectric state grid shape does not match operator."
            )
        if not np.array_equal(state.occupancy, self._occupancy):
            raise ValueError(
                "Diffuse-dielectric state occupancy does not match operator."
            )
        if not math.isclose(
            state.bulk_dielectric_constant,
            self.bulk_dielectric_constant,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Diffuse-dielectric state bulk dielectric does not match.")
        gradient = np.zeros(self.grid.shape, dtype=float)
        potential = state.solvated_potential_hartree_per_e
        for axis, spacing in enumerate(self.grid.spacing_bohr):
            forward_gradient = (np.roll(potential, -1, axis=axis) - potential) / spacing
            squared = forward_gradient**2
            gradient += squared + np.roll(squared, 1, axis=axis)
        gradient *= -(
            (self.bulk_dielectric_constant - 1.0)
            * self.grid.volume_element_bohr3
            / (16.0 * math.pi)
        )
        gradient.setflags(write=False)
        return gradient


__all__ = [
    "V0_DIFFUSE_DIELECTRIC_CONSTRUCTION",
    "V0_DIFFUSE_DIELECTRIC_SCOPE",
    "Route2V0DiffuseDielectricReactionState",
    "Route2V0DiffuseLocalDielectricOperator",
]
