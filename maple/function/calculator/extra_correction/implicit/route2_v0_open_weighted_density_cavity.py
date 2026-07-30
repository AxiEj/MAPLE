"""Open-boundary weighted-density cavitation scalar for Route-2 V0-AQ-C.

The periodic weighted-density control cannot be joined directly to an
isolated-source/open-Poisson calculation: periodic convolution creates
periodic solute images, while zero-extending the solvent occupancy would put
vacuum rather than bulk solvent outside the finite box.  This module uses one
finite, zero-extended convolution of the *occupancy deficit* ``s - 1``:

``s_bar = 1 + W_open[s - 1]``.

The exterior condition is consequently ``s = 1`` (bulk liquid).  The local
free-energy density is evaluated on the complete finite convolution support,
so a cavity close to the computational boundary sees the explicit bulk shell
rather than a wrapped image or a silently discarded vacuum-side contribution.

The local weighted-density free-energy density is the existing pure-liquid,
no-solvation-label construction.  This module supplies its matching open
discrete scalar and derivative only.  It has no physical solvent shell
kernel, density-to-solvent-centre occupancy proof, Pauli/dispersion scalar,
stationary electronic state, force/PES certificate, runtime result, or
solvation-accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re

import numpy as np

from .route2_v0_structured_solvent import RegularCartesianGrid
from .route2_v0_weighted_density_cavity import (
    Route2V0WeightedDensityCavityThermodynamics,
)

V0_OPEN_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION = (
    "route2-v0-open-weighted-density-cavity-v1"
)
V0_OPEN_WEIGHTED_DENSITY_CAVITY_SCOPE = "open-cavitation-scalar-only-not-total-v1"
V0_OPEN_WEIGHTED_DENSITY_CAVITY_BOUNDARY = "bulk-one-zero-extended-deficit-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _immutable_grid(
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


def _immutable_relative_kernel(
    values: np.ndarray,
    *,
    grid: RegularCartesianGrid,
) -> np.ndarray:
    """Return a normalized even odd-shape relative shell kernel."""

    if np.iscomplexobj(values):
        raise ValueError("Open weighted-density shell kernel must be real-valued.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Open weighted-density shell kernel must be a finite real field."
        ) from error
    if array.ndim != 3 or not np.all(np.isfinite(array)):
        raise ValueError(
            "Open weighted-density shell kernel must be finite and three-dimensional."
        )
    if any(length <= 0 or length % 2 == 0 for length in array.shape):
        raise ValueError(
            "Open weighted-density shell kernel must have a positive odd shape."
        )
    if np.any(array < 0.0):
        raise ValueError("Open weighted-density shell kernel must be nonnegative.")
    normalization = grid.volume_element_bohr3 * float(np.sum(array))
    if not math.isclose(normalization, 1.0, rel_tol=0.0, abs_tol=2.0e-13):
        raise ValueError(
            "Open weighted-density shell kernel must integrate to one under the "
            "grid pairing."
        )
    scale = max(1.0, float(np.max(np.abs(array))))
    if not np.allclose(
        array,
        np.flip(array),
        rtol=0.0,
        atol=128.0 * np.finfo(float).eps * scale,
    ):
        raise ValueError(
            "Open weighted-density shell kernel must be centrosymmetric for an "
            "exact self-adjoint map."
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _real_fft_result(values: np.ndarray, *, name: str) -> np.ndarray:
    """Reject a non-negligible imaginary residue from a real linear FFT map."""

    real = np.real(values)
    imaginary_scale = float(np.max(np.abs(np.imag(values))))
    scale = max(1.0, float(np.max(np.abs(real))))
    if imaginary_scale > 128.0 * np.finfo(float).eps * scale:
        raise RuntimeError(f"{name} produced a non-real FFT result.")
    return np.array(real, dtype=float, copy=True)


def _operator_fingerprint(
    *,
    grid: RegularCartesianGrid,
    thermodynamics: Route2V0WeightedDensityCavityThermodynamics,
    kernel: np.ndarray,
) -> str:
    """Hash every numerical datum that defines the open cavitation scalar."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(grid.origin_bohr, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(grid.spacing_bohr, dtype=np.float64).tobytes())
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(kernel, dtype=np.float64).tobytes())
    digest.update(
        np.asarray(
            (
                thermodynamics.pressure_hartree_per_bohr3,
                thermodynamics.molecular_number_density_per_bohr3,
                thermodynamics.thermal_energy_hartree,
                thermodynamics.vapor_pressure_hartree_per_bohr3,
                thermodynamics.surface_tension_hartree_per_bohr2,
                thermodynamics.solvent_vdw_radius_bohr,
            ),
            dtype=np.float64,
        ).tobytes()
    )
    digest.update(V0_OPEN_WEIGHTED_DENSITY_CAVITY_BOUNDARY.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Route2V0OpenWeightedDensityCavityState:
    """One immutable open bulk-continued cavitation evaluation."""

    solvent_center_occupancy: np.ndarray
    weighted_occupancy: np.ndarray
    extended_weighted_occupancy: np.ndarray
    cavity_energy_hartree: float
    operator_fingerprint: str
    boundary_condition: str = V0_OPEN_WEIGHTED_DENSITY_CAVITY_BOUNDARY
    construction: str = V0_OPEN_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION
    response_scope: str = V0_OPEN_WEIGHTED_DENSITY_CAVITY_SCOPE

    def __post_init__(self) -> None:
        shape = np.asarray(self.solvent_center_occupancy).shape
        if len(shape) != 3:
            raise ValueError("Open weighted-density cavity state occupancy must be 3D.")
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        occupancy = _immutable_grid(
            self.solvent_center_occupancy,
            shape=grid_shape,
            name="Open weighted-density cavity state occupancy",
        )
        weighted = _immutable_grid(
            self.weighted_occupancy,
            shape=grid_shape,
            name="Open weighted-density cavity state weighted occupancy",
        )
        extended_shape = np.asarray(self.extended_weighted_occupancy).shape
        if len(extended_shape) != 3 or any(
            int(extended_shape[axis]) < grid_shape[axis] for axis in range(3)
        ):
            raise ValueError(
                "Open weighted-density cavity extended occupancy must contain the "
                "interior grid."
            )
        extended = _immutable_grid(
            self.extended_weighted_occupancy,
            shape=(
                int(extended_shape[0]),
                int(extended_shape[1]),
                int(extended_shape[2]),
            ),
            name="Open weighted-density cavity state extended occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError(
                "Open weighted-density cavity occupancy must lie in [0, 1]."
            )
        if np.any(weighted < 0.0) or np.any(weighted > 1.0):
            raise ValueError(
                "Open weighted-density cavity weighted occupancy must lie in [0, 1]."
            )
        if np.any(extended < 0.0) or np.any(extended > 1.0):
            raise ValueError(
                "Open weighted-density cavity extended occupancy must lie in [0, 1]."
            )
        energy = float(self.cavity_energy_hartree)
        if not math.isfinite(energy):
            raise ValueError("Open weighted-density cavity energy must be finite.")
        if (
            not isinstance(self.operator_fingerprint, str)
            or _DIGEST.fullmatch(self.operator_fingerprint) is None
        ):
            raise ValueError(
                "Open weighted-density cavity state fingerprint is invalid."
            )
        if self.boundary_condition != V0_OPEN_WEIGHTED_DENSITY_CAVITY_BOUNDARY:
            raise ValueError("Unsupported open weighted-density cavity boundary.")
        if self.construction != V0_OPEN_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION:
            raise ValueError("Unsupported open weighted-density cavity construction.")
        if self.response_scope != V0_OPEN_WEIGHTED_DENSITY_CAVITY_SCOPE:
            raise ValueError("Unsupported open weighted-density cavity scope.")
        object.__setattr__(self, "solvent_center_occupancy", occupancy)
        object.__setattr__(self, "weighted_occupancy", weighted)
        object.__setattr__(self, "extended_weighted_occupancy", extended)
        object.__setattr__(self, "cavity_energy_hartree", energy)


@dataclass(frozen=True)
class Route2V0OpenWeightedDensityCavityFunctional:
    """Open bulk-continued weighted-density cavitation scalar.

    The relative kernel is deliberately supplied rather than generated from a
    radius.  A physical caller must bind it to independent pure-liquid
    structure.  This class has no rule that its input occupancy equals the
    electrostatic iso-density occupancy; that semantic identification requires
    separate source evidence before the two scalars may be combined.
    """

    grid: RegularCartesianGrid
    thermodynamics: Route2V0WeightedDensityCavityThermodynamics
    nearest_neighbor_shell_kernel_per_bohr3: np.ndarray
    construction: str = V0_OPEN_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION
    response_scope: str = V0_OPEN_WEIGHTED_DENSITY_CAVITY_SCOPE
    _kernel: np.ndarray = field(init=False, repr=False, compare=False)
    _operator_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError("Open weighted-density cavity requires a regular grid.")
        if not isinstance(
            self.thermodynamics,
            Route2V0WeightedDensityCavityThermodynamics,
        ):
            raise TypeError(
                "Open weighted-density cavity requires pure-liquid thermodynamics."
            )
        if self.construction != V0_OPEN_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 open weighted-density construction.")
        if self.response_scope != V0_OPEN_WEIGHTED_DENSITY_CAVITY_SCOPE:
            raise ValueError("Unsupported Route-2 open weighted-density scope.")
        kernel = _immutable_relative_kernel(
            self.nearest_neighbor_shell_kernel_per_bohr3,
            grid=self.grid,
        )
        object.__setattr__(self, "nearest_neighbor_shell_kernel_per_bohr3", kernel)
        object.__setattr__(self, "_kernel", kernel)
        object.__setattr__(
            self,
            "_operator_fingerprint",
            _operator_fingerprint(
                grid=self.grid,
                thermodynamics=self.thermodynamics,
                kernel=kernel,
            ),
        )

    @property
    def operator_fingerprint(self) -> str:
        """Return the immutable open grid/thermodynamics/kernel identity."""

        return self._operator_fingerprint

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false because cavitation alone is not total solvation."""

        return False

    @property
    def relative_kernel_centre_index(self) -> tuple[int, int, int]:
        """Return the unique relative-grid index for zero displacement."""

        return tuple(length // 2 for length in self._kernel.shape)

    @property
    def extended_shape(self) -> tuple[int, int, int]:
        """Return the full support of an inner-grid source and shell kernel."""

        return tuple(
            self.grid.shape[axis] + self._kernel.shape[axis] - 1 for axis in range(3)
        )

    @property
    def interior_slices(self) -> tuple[slice, slice, slice]:
        """Return the inner-grid position in the full convolution support."""

        start = self.relative_kernel_centre_index
        return tuple(
            slice(start[axis], start[axis] + self.grid.shape[axis]) for axis in range(3)
        )

    def _full_linear_convolution(self, values: np.ndarray) -> np.ndarray:
        """Map one inner-grid field into its complete open convolution support."""

        raw_full = np.fft.ifftn(
            np.fft.fftn(values, s=self.extended_shape, axes=(0, 1, 2))
            * np.fft.fftn(self._kernel, s=self.extended_shape, axes=(0, 1, 2)),
            axes=(0, 1, 2),
        )
        full = _real_fft_result(
            raw_full,
            name="Open weighted-density shell convolution",
        )
        return np.array(
            self.grid.volume_element_bohr3 * full,
            dtype=float,
            copy=True,
        )

    def _full_adjoint_convolution(self, values: np.ndarray) -> np.ndarray:
        """Pull an extended-support functional derivative back to the inner grid."""

        full_shape = tuple(
            self.extended_shape[axis] + self._kernel.shape[axis] - 1
            for axis in range(3)
        )
        raw_full = np.fft.ifftn(
            np.fft.fftn(values, s=full_shape, axes=(0, 1, 2))
            * np.fft.fftn(np.flip(self._kernel), s=full_shape, axes=(0, 1, 2)),
            axes=(0, 1, 2),
        )
        full = _real_fft_result(
            raw_full,
            name="Open weighted-density shell adjoint",
        )
        start = tuple(length - 1 for length in self._kernel.shape)
        slices = tuple(
            slice(start[axis], start[axis] + self.grid.shape[axis]) for axis in range(3)
        )
        return np.array(
            self.grid.volume_element_bohr3 * full[slices],
            dtype=float,
            copy=True,
        )

    def _convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply the open shell map without periodic images."""

        return self._full_linear_convolution(values)[self.interior_slices]

    def _adjoint_convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply the exact grid-pairing adjoint of :meth:`_convolution`."""

        return self._convolution(values)

    def _local_energy_density_hartree_per_bohr3(self, values: np.ndarray) -> np.ndarray:
        """Return the pure-liquid weighted-density local free-energy density."""

        thermodynamics = self.thermodynamics
        gamma = thermodynamics.small_droplet_log_term
        coefficient = thermodynamics.surface_coefficient
        pair = values * (1.0 - values)
        bracket = values + (1.0 - values) * gamma + 15.0 * pair * coefficient
        return (
            thermodynamics.pressure_hartree_per_bohr3 * (1.0 - values)
            + thermodynamics.molecular_number_density_per_bohr3
            * thermodynamics.thermal_energy_hartree
            * pair
            * bracket
        )

    def _local_energy_derivative_hartree_per_bohr3(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Return the exact local derivative of the same free-energy density."""

        thermodynamics = self.thermodynamics
        gamma = thermodynamics.small_droplet_log_term
        coefficient = thermodynamics.surface_coefficient
        pair = values * (1.0 - values)
        pair_derivative = 1.0 - 2.0 * values
        bracket = values + (1.0 - values) * gamma + 15.0 * pair * coefficient
        bracket_derivative = 1.0 - gamma + 15.0 * pair_derivative * coefficient
        return (
            -thermodynamics.pressure_hartree_per_bohr3
            + thermodynamics.molecular_number_density_per_bohr3
            * thermodynamics.thermal_energy_hartree
            * (pair_derivative * bracket + pair * bracket_derivative)
        )

    def _validate_state(self, state: Route2V0OpenWeightedDensityCavityState) -> None:
        if not isinstance(state, Route2V0OpenWeightedDensityCavityState):
            raise TypeError("Open weighted-density cavity derivative requires a state.")
        if state.operator_fingerprint != self._operator_fingerprint:
            raise ValueError(
                "Open weighted-density cavity state does not match this functional."
            )
        if state.extended_weighted_occupancy.shape != self.extended_shape:
            raise ValueError(
                "Open weighted-density cavity extended support does not match this "
                "functional."
            )
        if not np.array_equal(
            state.weighted_occupancy,
            state.extended_weighted_occupancy[self.interior_slices],
        ):
            raise ValueError(
                "Open weighted-density cavity interior/extended occupancy mismatch."
            )

    def evaluate(
        self,
        solvent_center_occupancy: np.ndarray,
    ) -> Route2V0OpenWeightedDensityCavityState:
        """Evaluate the scalar with bulk liquid continued outside the box."""

        occupancy = _immutable_grid(
            solvent_center_occupancy,
            shape=self.grid.shape,
            name="Open weighted-density solvent-centre occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError(
                "Open weighted-density solvent-centre occupancy must lie in [0, 1]."
            )
        deficit = occupancy - 1.0
        extended_weighted = 1.0 + self._full_linear_convolution(deficit)
        tolerance = 512.0 * np.finfo(float).eps
        if (
            float(np.min(extended_weighted)) < -tolerance
            or float(np.max(extended_weighted)) > 1.0 + tolerance
        ):
            raise RuntimeError(
                "Open weighted-density shell average escaped [0, 1]; the supplied "
                "kernel is not a valid positive bulk-continued occupancy average."
            )
        extended_weighted = np.clip(extended_weighted, 0.0, 1.0)
        weighted = np.array(
            extended_weighted[self.interior_slices],
            dtype=float,
            copy=True,
        )
        energy = float(
            self.grid.volume_element_bohr3
            * np.sum(self._local_energy_density_hartree_per_bohr3(extended_weighted))
        )
        if not math.isfinite(energy):
            raise RuntimeError("Open weighted-density cavity scalar is non-finite.")
        weighted.setflags(write=False)
        extended_weighted.setflags(write=False)
        return Route2V0OpenWeightedDensityCavityState(
            solvent_center_occupancy=occupancy,
            weighted_occupancy=weighted,
            extended_weighted_occupancy=extended_weighted,
            cavity_energy_hartree=energy,
            operator_fingerprint=self._operator_fingerprint,
        )

    def cavity_energy_hartree(
        self,
        state: Route2V0OpenWeightedDensityCavityState,
    ) -> float:
        """Return the scalar after rejecting a crossed state."""

        self._validate_state(state)
        return state.cavity_energy_hartree

    def occupancy_functional_derivative_hartree_per_bohr3(
        self,
        state: Route2V0OpenWeightedDensityCavityState,
    ) -> np.ndarray:
        """Return ``delta G_cav / delta s`` in the declared grid pairing."""

        self._validate_state(state)
        derivative = self._full_adjoint_convolution(
            self._local_energy_derivative_hartree_per_bohr3(
                state.extended_weighted_occupancy
            )
        )
        derivative.setflags(write=False)
        return derivative

    def occupancy_coordinate_gradient_hartree(
        self,
        state: Route2V0OpenWeightedDensityCavityState,
    ) -> np.ndarray:
        """Return ``dG_cav/ds_i`` in Hartree for an occupancy-coordinate VJP."""

        result = np.array(
            self.grid.volume_element_bohr3
            * self.occupancy_functional_derivative_hartree_per_bohr3(state),
            dtype=float,
            copy=True,
        )
        result.setflags(write=False)
        return result


__all__ = [
    "V0_OPEN_WEIGHTED_DENSITY_CAVITY_BOUNDARY",
    "V0_OPEN_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION",
    "V0_OPEN_WEIGHTED_DENSITY_CAVITY_SCOPE",
    "Route2V0OpenWeightedDensityCavityFunctional",
    "Route2V0OpenWeightedDensityCavityState",
]
