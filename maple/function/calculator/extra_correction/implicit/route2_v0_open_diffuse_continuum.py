"""Open-boundary reciprocal reaction field for isolated Route-2 V0 sources.

The existing :mod:`route2_v0_diffuse_continuum` control is periodic and
therefore correctly rejects a non-neutral grid source.  That convention is
useful for bulk controls but cannot be used to hide a finite AO-grid electron
count error behind a periodic neutralising background.  This module instead
uses one cell-centred, zero-Dirichlet-face finite-volume operator for both the
vacuum and dielectric fields:

``A_eps phi_eps = 4*pi*rho`` and
``G_reac[rho, m] = 1/2 dV sum rho (phi_eps - phi_vac)``.

All grid cells are unknowns; the zero-potential boundary lies one half-cell
outside the outermost cell centres.  The corresponding matrix is real,
symmetric, and positive definite, so a source need not be artificially
neutral.  A finite box is still an approximation to infinity: its buffer and
grid refinement are explicit future convergence gates, not a source repair.

The class deliberately implements *only* a fixed-occupancy electrostatic
reaction scalar.  It has no solvent kernel, nonpolar/dispersion term,
stationary electronic functional, force/PES certificate, or solvation
accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re

import numpy as np

from .route2_v0_bulk_liquid_state_source import Route2V0BulkLiquidStateSource
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_OPEN_DIFFUSE_DIELECTRIC_CONSTRUCTION = "route2-v0-open-diffuse-dielectric-v1"
V0_OPEN_DIFFUSE_DIELECTRIC_SCOPE = "open-boundary-reaction-block-only-v1"
V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY = "zero-dirichlet-cell-face-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _finite_scalar(value: object, *, name: str) -> float:
    """Return one finite scalar without treating booleans as numbers."""

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
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real grid field.") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _validated_iteration_cap(value: int | None, *, grid: RegularCartesianGrid) -> int:
    """Return a finite CG cap without an unbounded iteration loop."""

    if value is None:
        return max(200, 12 * max(grid.shape) ** 2)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("Open diffuse-dielectric CG iteration cap must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError("Open diffuse-dielectric CG iteration cap must be positive.")
    return result


def _grid_fingerprint(grid: RegularCartesianGrid) -> str:
    """Return one immutable identity for the cell-centred open-box pairing."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(grid.origin_bohr, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(grid.spacing_bohr, dtype=np.float64).tobytes())
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY.encode("ascii"))
    return digest.hexdigest()


def _axis_slices(axis: int) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Return adjacent lower/upper cell slices for one Cartesian axis."""

    lower = [slice(None)] * 3
    upper = [slice(None)] * 3
    lower[axis] = slice(None, -1)
    upper[axis] = slice(1, None)
    return tuple(lower), tuple(upper)


def _boundary_slice(axis: int, index: int) -> tuple[slice | int, ...]:
    """Return one lower or upper boundary-cell slice."""

    result: list[slice | int] = [slice(None)] * 3
    result[axis] = index
    return tuple(result)


@dataclass(frozen=True)
class Route2V0OpenDiffuseDielectricReactionState:
    """One open-boundary fixed-occupancy reaction-field solve.

    The stored potentials are cell-centred values.  Their boundary condition
    is fixed at the six exterior *faces* of the finite box, not at artificial
    periodic images or a hidden neutralising background.
    """

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
    total_grid_charge_e: float
    bulk_dielectric_constant: float
    grid_fingerprint: str
    boundary_condition: str = V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY
    construction: str = V0_OPEN_DIFFUSE_DIELECTRIC_CONSTRUCTION
    response_scope: str = V0_OPEN_DIFFUSE_DIELECTRIC_SCOPE

    def __post_init__(self) -> None:
        shape = np.asarray(self.charge_density_e_per_bohr3).shape
        if len(shape) != 3 or any(int(length) < 2 for length in shape):
            raise ValueError(
                "Open diffuse-dielectric state charge density must have at least "
                "two cells along each axis."
            )
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        density = _immutable_real_grid(
            self.charge_density_e_per_bohr3,
            shape=grid_shape,
            name="Open diffuse-dielectric state charge density",
        )
        occupancy = _immutable_real_grid(
            self.occupancy,
            shape=grid_shape,
            name="Open diffuse-dielectric state occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError(
                "Open diffuse-dielectric state occupancy must lie in [0, 1]."
            )
        values: dict[str, np.ndarray] = {}
        for name, value in (
            (
                "vacuum_potential_hartree_per_e",
                self.vacuum_potential_hartree_per_e,
            ),
            (
                "solvated_potential_hartree_per_e",
                self.solvated_potential_hartree_per_e,
            ),
            (
                "reaction_potential_hartree_per_e",
                self.reaction_potential_hartree_per_e,
            ),
        ):
            values[name] = _immutable_real_grid(
                value,
                shape=grid_shape,
                name=f"Open diffuse-dielectric state {name}",
            )
        potential_scale = max(
            1.0,
            float(np.max(np.abs(values["vacuum_potential_hartree_per_e"]))),
            float(np.max(np.abs(values["solvated_potential_hartree_per_e"]))),
        )
        if not np.allclose(
            values["reaction_potential_hartree_per_e"],
            values["solvated_potential_hartree_per_e"]
            - values["vacuum_potential_hartree_per_e"],
            rtol=0.0,
            atol=256.0 * np.finfo(float).eps * potential_scale,
        ):
            raise ValueError(
                "Open diffuse-dielectric reaction potential is inconsistent."
            )
        for name in (
            "vacuum_energy_hartree",
            "solvated_energy_hartree",
            "reaction_energy_hartree",
            "vacuum_residual_inf",
            "solvated_residual_inf",
            "total_grid_charge_e",
        ):
            object.__setattr__(
                self, name, _finite_scalar(getattr(self, name), name=name)
            )
        if self.vacuum_residual_inf < 0.0 or self.solvated_residual_inf < 0.0:
            raise ValueError(
                "Open diffuse-dielectric state residuals must be nonnegative."
            )
        energy_scale = max(
            1.0,
            abs(self.vacuum_energy_hartree),
            abs(self.solvated_energy_hartree),
        )
        if not math.isclose(
            self.reaction_energy_hartree,
            self.solvated_energy_hartree - self.vacuum_energy_hartree,
            rel_tol=0.0,
            abs_tol=256.0 * np.finfo(float).eps * energy_scale,
        ):
            raise ValueError("Open diffuse-dielectric reaction energy is inconsistent.")
        dielectric = _positive_scalar(
            self.bulk_dielectric_constant,
            name="Open diffuse-dielectric state bulk dielectric constant",
        )
        if dielectric <= 1.0:
            raise ValueError(
                "Open diffuse-dielectric state bulk dielectric must exceed one."
            )
        if (
            not isinstance(self.grid_fingerprint, str)
            or _DIGEST.fullmatch(self.grid_fingerprint) is None
        ):
            raise ValueError(
                "Open diffuse-dielectric state grid fingerprint is invalid."
            )
        if self.boundary_condition != V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY:
            raise ValueError("Unsupported open diffuse-dielectric boundary condition.")
        if self.construction != V0_OPEN_DIFFUSE_DIELECTRIC_CONSTRUCTION:
            raise ValueError("Unsupported open diffuse-dielectric state construction.")
        if self.response_scope != V0_OPEN_DIFFUSE_DIELECTRIC_SCOPE:
            raise ValueError(
                "Unsupported open diffuse-dielectric state response scope."
            )
        object.__setattr__(self, "charge_density_e_per_bohr3", density)
        object.__setattr__(self, "occupancy", occupancy)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "bulk_dielectric_constant", dielectric)


@dataclass(frozen=True)
class Route2V0OpenDiffuseLocalDielectricOperator:
    """One self-adjoint cell-centred finite-volume reaction operator.

    The finite box has homogeneous Dirichlet potential at its exterior faces.
    Interior faces use the arithmetic dielectric mean; boundary faces use the
    adjacent cell dielectric and lie half a grid spacing away.  This yields
    the exact discrete envelope derivative returned by
    :meth:`occupancy_energy_gradient_hartree`.
    """

    grid: RegularCartesianGrid
    occupancy: np.ndarray
    bulk_dielectric_constant: float
    bulk_state_source: Route2V0BulkLiquidStateSource | None = None
    cg_relative_tolerance: float = 1.0e-12
    cg_max_iterations: int | None = None
    construction: str = V0_OPEN_DIFFUSE_DIELECTRIC_CONSTRUCTION
    response_scope: str = V0_OPEN_DIFFUSE_DIELECTRIC_SCOPE
    _occupancy: np.ndarray = field(init=False, repr=False, compare=False)
    _dielectric: np.ndarray = field(init=False, repr=False, compare=False)
    _grid_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError(
                "Open diffuse dielectric requires a regular Cartesian grid."
            )
        if any(length < 2 for length in self.grid.shape):
            raise ValueError(
                "Open diffuse dielectric requires at least two cells along each axis."
            )
        if self.construction != V0_OPEN_DIFFUSE_DIELECTRIC_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 open diffuse dielectric construction."
            )
        if self.response_scope != V0_OPEN_DIFFUSE_DIELECTRIC_SCOPE:
            raise ValueError(
                "Unsupported Route-2 open diffuse dielectric response scope."
            )
        occupancy = _immutable_real_grid(
            self.occupancy,
            shape=self.grid.shape,
            name="Open diffuse-dielectric occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError("Open diffuse-dielectric occupancy must lie in [0, 1].")
        dielectric_bulk = _positive_scalar(
            self.bulk_dielectric_constant,
            name="Open diffuse-dielectric bulk dielectric constant",
        )
        if dielectric_bulk <= 1.0:
            raise ValueError("Open diffuse-dielectric bulk dielectric must exceed one.")
        source = self.bulk_state_source
        if source is not None:
            if not isinstance(source, Route2V0BulkLiquidStateSource):
                raise TypeError(
                    "Open diffuse dielectric bulk state must be a Route-2 bulk-state source."
                )
            if not math.isclose(
                source.static_dielectric_constant,
                dielectric_bulk,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "Open diffuse-dielectric bulk dielectric must equal the attached "
                    "bulk-state source."
                )
        tolerance = _finite_scalar(
            self.cg_relative_tolerance,
            name="Open diffuse-dielectric CG relative tolerance",
        )
        if not 0.0 < tolerance < 1.0:
            raise ValueError(
                "Open diffuse-dielectric CG relative tolerance must lie in (0, 1)."
            )
        dielectric = _immutable_real_grid(
            1.0 + (dielectric_bulk - 1.0) * occupancy,
            shape=self.grid.shape,
            name="Open diffuse-dielectric cell dielectric",
        )
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "bulk_dielectric_constant", dielectric_bulk)
        object.__setattr__(self, "cg_relative_tolerance", tolerance)
        object.__setattr__(
            self,
            "cg_max_iterations",
            _validated_iteration_cap(self.cg_max_iterations, grid=self.grid),
        )
        object.__setattr__(self, "_occupancy", occupancy)
        object.__setattr__(self, "_dielectric", dielectric)
        object.__setattr__(self, "_grid_fingerprint", _grid_fingerprint(self.grid))

    @property
    def dielectric_field(self) -> np.ndarray:
        """Return ``1 + (epsilon_bulk - 1) m`` at the cell centres."""

        return self._dielectric

    @property
    def boundary_condition(self) -> str:
        """Return the one fixed open-boundary convention."""

        return V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY

    @property
    def grid_fingerprint(self) -> str:
        """Return the fixed open-grid/boundary identity for produced states."""

        return self._grid_fingerprint

    @property
    def is_bulk_state_bound(self) -> bool:
        """Return whether the dielectric has independent bulk-state provenance."""

        return self.bulk_state_source is not None

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false: liquid, nonpolar, and standard-state terms are absent."""

        return False

    def require_bulk_state_source(self) -> Route2V0BulkLiquidStateSource:
        """Return the bound bulk-state source or fail before physical promotion."""

        if self.bulk_state_source is None:
            raise ValueError(
                "Open diffuse dielectric has no bulk-state provenance and is only a "
                "numerical control."
            )
        return self.bulk_state_source

    def _apply_poisson(
        self,
        potential: np.ndarray,
        *,
        dielectric: np.ndarray,
    ) -> np.ndarray:
        """Apply ``-div(epsilon grad)`` with zero Dirichlet exterior faces."""

        result = np.zeros(self.grid.shape, dtype=float)
        for axis, spacing in enumerate(self.grid.spacing_bohr):
            lower, upper = _axis_slices(axis)
            face_dielectric = 0.5 * (dielectric[lower] + dielectric[upper])
            forward_flux = (
                face_dielectric * (potential[upper] - potential[lower]) / spacing
            )
            result[lower] -= forward_flux / spacing
            result[upper] += forward_flux / spacing
            lower_boundary = _boundary_slice(axis, 0)
            upper_boundary = _boundary_slice(axis, -1)
            result[lower_boundary] += (
                2.0
                * dielectric[lower_boundary]
                * potential[lower_boundary]
                / spacing**2
            )
            result[upper_boundary] += (
                2.0
                * dielectric[upper_boundary]
                * potential[upper_boundary]
                / spacing**2
            )
        return result

    def _conjugate_gradient(
        self,
        *,
        right_hand_side: np.ndarray,
        dielectric: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Solve the symmetric positive-definite open-boundary system."""

        rhs = np.array(right_hand_side, dtype=float, copy=True)
        rhs_norm = float(np.linalg.norm(rhs.ravel()))
        if rhs_norm == 0.0:
            return np.zeros(self.grid.shape, dtype=float), 0.0
        target_norm = self.cg_relative_tolerance * max(1.0, rhs_norm)
        solution = np.zeros(self.grid.shape, dtype=float)
        residual = np.array(rhs, copy=True)
        direction = np.array(residual, copy=True)
        residual_norm_squared = float(np.vdot(residual, residual))
        if self.cg_max_iterations is None:
            raise RuntimeError(
                "Open diffuse-dielectric CG iteration cap is unavailable."
            )
        for _ in range(self.cg_max_iterations):
            applied_direction = self._apply_poisson(direction, dielectric=dielectric)
            denominator = float(np.vdot(direction, applied_direction))
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise RuntimeError(
                    "Open diffuse-dielectric CG lost positive curvature."
                )
            step = residual_norm_squared / denominator
            solution += step * direction
            residual -= step * applied_direction
            next_norm_squared = float(np.vdot(residual, residual))
            if not math.isfinite(next_norm_squared):
                raise RuntimeError("Open diffuse-dielectric CG residual is non-finite.")
            if math.sqrt(next_norm_squared) <= target_norm:
                return solution, float(np.max(np.abs(residual)))
            coefficient = next_norm_squared / residual_norm_squared
            direction = residual + coefficient * direction
            residual_norm_squared = next_norm_squared
        residual_inf = float(np.max(np.abs(residual)))
        raise RuntimeError(
            "Open diffuse-dielectric CG did not converge within the declared cap; "
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
        potential = np.array(potential, dtype=float, copy=True)
        potential.setflags(write=False)
        return potential, residual

    def _electrostatic_energy_hartree(
        self,
        density: np.ndarray,
        potential: np.ndarray,
    ) -> float:
        """Return ``1/2 dV sum rho phi`` for the declared open-box scalar."""

        energy = (
            0.5 * self.grid.volume_element_bohr3 * float(np.sum(density * potential))
        )
        if not math.isfinite(energy):
            raise RuntimeError(
                "Open diffuse-dielectric electrostatic energy is non-finite."
            )
        return energy

    def solve(
        self,
        charge_density_e_per_bohr3: np.ndarray,
    ) -> Route2V0OpenDiffuseDielectricReactionState:
        """Solve the vacuum and dielectric fields from one isolated source.

        No periodic-neutrality condition is imposed.  Callers must instead
        certify finite-box and grid convergence for their physical source.
        """

        density = _immutable_real_grid(
            charge_density_e_per_bohr3,
            shape=self.grid.shape,
            name="Open diffuse-dielectric charge density",
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
        reaction_potential = np.array(
            solvated_potential - vacuum_potential,
            dtype=float,
            copy=True,
        )
        reaction_potential.setflags(write=False)
        vacuum_energy = self._electrostatic_energy_hartree(density, vacuum_potential)
        solvated_energy = self._electrostatic_energy_hartree(
            density,
            solvated_potential,
        )
        reaction_energy = self._electrostatic_energy_hartree(
            density,
            reaction_potential,
        )
        numerical_scale = max(1.0, abs(vacuum_energy), abs(solvated_energy))
        if reaction_energy > 32.0 * self.cg_relative_tolerance * numerical_scale:
            raise RuntimeError(
                "Open diffuse-dielectric reaction scalar violates passivity despite "
                "epsilon >= 1."
            )
        total_charge = self.grid.volume_element_bohr3 * float(np.sum(density))
        return Route2V0OpenDiffuseDielectricReactionState(
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
            total_grid_charge_e=total_charge,
            bulk_dielectric_constant=self.bulk_dielectric_constant,
            grid_fingerprint=self._grid_fingerprint,
        )

    def reaction_potential_hartree_per_e(
        self,
        charge_density_e_per_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the density derivative of the declared reaction scalar."""

        return self.solve(charge_density_e_per_bohr3).reaction_potential_hartree_per_e

    def reaction_energy_hartree(
        self,
        charge_density_e_per_bohr3: np.ndarray,
    ) -> float:
        """Return ``1/2 dV sum rho (phi_eps - phi_vac)``."""

        return self.solve(charge_density_e_per_bohr3).reaction_energy_hartree

    def reaction_pairing_hartree(
        self,
        left_charge_density_e_per_bohr3: np.ndarray,
        right_charge_density_e_per_bohr3: np.ndarray,
    ) -> float:
        """Return the declared density/reaction-potential pairing."""

        left = _immutable_real_grid(
            left_charge_density_e_per_bohr3,
            shape=self.grid.shape,
            name="Open diffuse-dielectric left charge density",
        )
        right_state = self.solve(right_charge_density_e_per_bohr3)
        return self.grid.volume_element_bohr3 * float(
            np.sum(left * right_state.reaction_potential_hartree_per_e)
        )

    def _validate_state(
        self, state: Route2V0OpenDiffuseDielectricReactionState
    ) -> None:
        """Reject a state from another occupancy, dielectric, or boundary scalar."""

        if not isinstance(state, Route2V0OpenDiffuseDielectricReactionState):
            raise TypeError(
                "Open diffuse-dielectric occupancy derivative requires a state."
            )
        if state.charge_density_e_per_bohr3.shape != self.grid.shape:
            raise ValueError("Open diffuse-dielectric state grid shape does not match.")
        if state.grid_fingerprint != self._grid_fingerprint:
            raise ValueError(
                "Open diffuse-dielectric state grid representation does not match."
            )
        expected_charge = self.grid.volume_element_bohr3 * float(
            np.sum(state.charge_density_e_per_bohr3)
        )
        charge_tolerance = (
            256.0
            * np.finfo(float).eps
            * max(
                1.0,
                abs(expected_charge),
            )
        )
        if abs(state.total_grid_charge_e - expected_charge) > charge_tolerance:
            raise ValueError(
                "Open diffuse-dielectric state total grid charge is inconsistent."
            )
        if not np.array_equal(state.occupancy, self._occupancy):
            raise ValueError("Open diffuse-dielectric state occupancy does not match.")
        if not math.isclose(
            state.bulk_dielectric_constant,
            self.bulk_dielectric_constant,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Open diffuse-dielectric state bulk dielectric does not match."
            )
        if state.boundary_condition != V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY:
            raise ValueError(
                "Open diffuse-dielectric state boundary condition does not match."
            )

    def occupancy_energy_gradient_hartree(
        self,
        state: Route2V0OpenDiffuseDielectricReactionState,
    ) -> np.ndarray:
        """Return the exact discrete envelope derivative ``dG_reac/dm``.

        An interior face contributes its squared field gradient to both
        adjacent cells.  A zero-Dirichlet exterior face lies half a cell away;
        its gradient is ``2*phi/h`` and contributes to its adjacent boundary
        cell.  The same finite-volume quadratic form appears in the Poisson
        solve and this derivative, so no response solve or force correction is
        introduced.
        """

        self._validate_state(state)
        gradient_sum = np.zeros(self.grid.shape, dtype=float)
        potential = state.solvated_potential_hartree_per_e
        for axis, spacing in enumerate(self.grid.spacing_bohr):
            lower, upper = _axis_slices(axis)
            squared = ((potential[upper] - potential[lower]) / spacing) ** 2
            gradient_sum[lower] += squared
            gradient_sum[upper] += squared
            lower_boundary = _boundary_slice(axis, 0)
            upper_boundary = _boundary_slice(axis, -1)
            gradient_sum[lower_boundary] += (
                2.0 * potential[lower_boundary] / spacing
            ) ** 2
            gradient_sum[upper_boundary] += (
                2.0 * potential[upper_boundary] / spacing
            ) ** 2
        gradient_sum *= -(
            (self.bulk_dielectric_constant - 1.0)
            * self.grid.volume_element_bohr3
            / (16.0 * math.pi)
        )
        gradient_sum.setflags(write=False)
        return gradient_sum


__all__ = [
    "V0_OPEN_DIFFUSE_DIELECTRIC_BOUNDARY",
    "V0_OPEN_DIFFUSE_DIELECTRIC_CONSTRUCTION",
    "V0_OPEN_DIFFUSE_DIELECTRIC_SCOPE",
    "Route2V0OpenDiffuseDielectricReactionState",
    "Route2V0OpenDiffuseLocalDielectricOperator",
]
