"""Open-domain iso-density-product cavity for the Route-2 V0 scalar.

The periodic :mod:`route2_v0_iso_density_cavity` control is intentionally not
usable with an isolated AO density.  Its FFT convolution wraps the density
through the cell boundary, whereas the open reaction block uses a finite box
with zero Dirichlet faces.  This module supplies the matching *linear*,
zero-extended convolution and combines it only with the open reciprocal
reaction scalar.

For a frozen, centrosymmetric isolated-solvent density kernel ``k`` and a
positive threshold ``q_c``, the open cavity is

``m[n] = 1/2 erfc(log((k * n) / q_c))``.

The threshold and kernel have no defaults: a physical calculation must bind
them to independent solvent evidence before any chemistry calculation.  This
module proves only the discrete common-scalar calculus.  It has no stationary
electronic functional, physical solvent asset, nonpolar/dispersion term,
standard-state correction, force certificate, runtime claim, or accuracy
claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re

import numpy as np
from scipy.special import erfc

from .route2_v0_open_diffuse_continuum import (
    Route2V0OpenDiffuseDielectricReactionState,
    Route2V0OpenDiffuseLocalDielectricOperator,
)
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION = (
    "route2-v0-open-iso-density-product-cavity-v1"
)
V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE = (
    "open-nonlocal-cavity-electrostatic-block-only-v1"
)
V0_OPEN_ISO_DENSITY_PRODUCT_CONVOLUTION = "zero-extended-linear-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _finite_positive(value: object, *, name: str) -> float:
    """Return one finite positive scalar without accepting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite positive real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _immutable_real_grid(
    values: np.ndarray,
    *,
    shape: tuple[int, int, int],
    name: str,
    nonnegative: bool = False,
    nonzero: bool = False,
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
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if nonzero and not np.any(array > 0.0):
        raise ValueError(f"{name} must not vanish everywhere.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_centered_kernel(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return one nonnegative, centrosymmetric odd-shape relative kernel."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real kernel.") from error
    if array.ndim != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-dimensional kernel.")
    if any(length <= 0 or length % 2 == 0 for length in array.shape):
        raise ValueError(f"{name} must have a positive odd shape about zero offset.")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if not np.any(array > 0.0):
        raise ValueError(f"{name} must not vanish everywhere.")
    scale = max(1.0, float(np.max(np.abs(array))))
    if not np.allclose(
        array,
        np.flip(array),
        rtol=0.0,
        atol=128.0 * np.finfo(float).eps * scale,
    ):
        raise ValueError(
            f"{name} must be centrosymmetric for a self-adjoint open convolution."
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _operator_fingerprint(
    *,
    grid: RegularCartesianGrid,
    kernel: np.ndarray,
    overlap_threshold: float,
) -> str:
    """Return the exact open-grid/cavity identity used by every state."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(grid.origin_bohr, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(grid.spacing_bohr, dtype=np.float64).tobytes())
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(kernel, dtype=np.float64).tobytes())
    digest.update(np.asarray([overlap_threshold], dtype=np.float64).tobytes())
    digest.update(V0_OPEN_ISO_DENSITY_PRODUCT_CONVOLUTION.encode("ascii"))
    return digest.hexdigest()


def _real_fft_result(values: np.ndarray, *, name: str) -> np.ndarray:
    """Reject a non-negligible imaginary residue from a real linear FFT map."""

    real = np.real(values)
    imaginary_scale = float(np.max(np.abs(np.imag(values))))
    scale = max(1.0, float(np.max(np.abs(real))))
    if imaginary_scale > 128.0 * np.finfo(float).eps * scale:
        raise RuntimeError(f"{name} produced a non-real FFT result.")
    return np.array(real, dtype=float, copy=True)


@dataclass(frozen=True)
class Route2V0OpenIsoDensityProductCavityState:
    """One open smooth occupancy field and its exact derivative data."""

    solute_electron_density_e_per_bohr3: np.ndarray
    overlap_density_e2_per_bohr3: np.ndarray
    occupancy: np.ndarray
    occupancy_derivative_per_overlap_bohr3_per_e2: np.ndarray
    overlap_threshold_e2_per_bohr3: float
    operator_fingerprint: str
    construction: str = V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    response_scope: str = V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE

    def __post_init__(self) -> None:
        shape = np.asarray(self.solute_electron_density_e_per_bohr3).shape
        if len(shape) != 3:
            raise ValueError(
                "Open iso-density-product state electron density must be 3D."
            )
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        electron = _immutable_real_grid(
            self.solute_electron_density_e_per_bohr3,
            shape=grid_shape,
            name="Open iso-density-product state electron density",
            nonnegative=True,
            nonzero=True,
        )
        overlap = _immutable_real_grid(
            self.overlap_density_e2_per_bohr3,
            shape=grid_shape,
            name="Open iso-density-product state overlap density",
            nonnegative=True,
        )
        occupancy = _immutable_real_grid(
            self.occupancy,
            shape=grid_shape,
            name="Open iso-density-product state occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError(
                "Open iso-density-product state occupancy must lie in [0, 1]."
            )
        derivative = _immutable_real_grid(
            self.occupancy_derivative_per_overlap_bohr3_per_e2,
            shape=grid_shape,
            name="Open iso-density-product occupancy derivative",
        )
        if np.any(derivative > 0.0):
            raise ValueError(
                "Open iso-density-product occupancy derivative must be nonpositive."
            )
        threshold = _finite_positive(
            self.overlap_threshold_e2_per_bohr3,
            name="Open iso-density-product overlap threshold",
        )
        if (
            not isinstance(self.operator_fingerprint, str)
            or _DIGEST.fullmatch(self.operator_fingerprint) is None
        ):
            raise ValueError(
                "Open iso-density-product state operator fingerprint is invalid."
            )
        if self.construction != V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION:
            raise ValueError("Unsupported open iso-density-product state construction.")
        if self.response_scope != V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE:
            raise ValueError(
                "Unsupported open iso-density-product state response scope."
            )
        object.__setattr__(self, "solute_electron_density_e_per_bohr3", electron)
        object.__setattr__(self, "overlap_density_e2_per_bohr3", overlap)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(
            self,
            "occupancy_derivative_per_overlap_bohr3_per_e2",
            derivative,
        )
        object.__setattr__(self, "overlap_threshold_e2_per_bohr3", threshold)


@dataclass(frozen=True)
class Route2V0OpenIsoDensityProductCavity:
    """Zero-extended nonlocal cavity with an exact self-adjoint grid map.

    ``solvent_electron_density_kernel_e_per_bohr3`` is sampled on a relative
    offset grid whose zero vector is the central index.  It is deliberately
    not wrapped onto the simulation domain.  The output therefore changes if
    a density tail reaches the finite box boundary, which must be exposed by a
    later grid/buffer convergence study rather than hidden by periodic images.
    """

    grid: RegularCartesianGrid
    solvent_electron_density_kernel_e_per_bohr3: np.ndarray
    overlap_threshold_e2_per_bohr3: float
    construction: str = V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    response_scope: str = V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE
    _kernel: np.ndarray = field(init=False, repr=False, compare=False)
    _operator_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError("Open iso-density-product cavity requires a regular grid.")
        if self.construction != V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 open iso-density-product construction."
            )
        if self.response_scope != V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE:
            raise ValueError(
                "Unsupported Route-2 open iso-density-product response scope."
            )
        kernel = _immutable_centered_kernel(
            self.solvent_electron_density_kernel_e_per_bohr3,
            name="Open iso-density-product solvent electron-density kernel",
        )
        threshold = _finite_positive(
            self.overlap_threshold_e2_per_bohr3,
            name="Open iso-density-product overlap threshold",
        )
        object.__setattr__(self, "solvent_electron_density_kernel_e_per_bohr3", kernel)
        object.__setattr__(self, "overlap_threshold_e2_per_bohr3", threshold)
        object.__setattr__(self, "_kernel", kernel)
        object.__setattr__(
            self,
            "_operator_fingerprint",
            _operator_fingerprint(
                grid=self.grid,
                kernel=kernel,
                overlap_threshold=threshold,
            ),
        )

    @property
    def operator_fingerprint(self) -> str:
        """Return the immutable open grid/kernel/threshold identity."""

        return self._operator_fingerprint

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false: the cavity has no physical solvent or total ledger."""

        return False

    @property
    def relative_kernel_centre_index(self) -> tuple[int, int, int]:
        """Return the unique grid index representing zero relative displacement."""

        return tuple(length // 2 for length in self._kernel.shape)

    def _linear_convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply the zero-extended linear convolution in the declared pairing."""

        full_shape = tuple(
            self.grid.shape[axis] + self._kernel.shape[axis] - 1 for axis in range(3)
        )
        raw_full = np.fft.ifftn(
            np.fft.fftn(values, s=full_shape, axes=(0, 1, 2))
            * np.fft.fftn(self._kernel, s=full_shape, axes=(0, 1, 2)),
            axes=(0, 1, 2),
        )
        full = _real_fft_result(raw_full, name="Open iso-density-product convolution")
        start = self.relative_kernel_centre_index
        slices = tuple(
            slice(start[axis], start[axis] + self.grid.shape[axis]) for axis in range(3)
        )
        result = self.grid.volume_element_bohr3 * full[slices]
        return np.array(result, dtype=float, copy=True)

    def _convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply ``K[n]`` without periodic images or boundary wrapping."""

        return self._linear_convolution(values)

    def _adjoint_convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply the exact discrete adjoint of :meth:`_convolution`.

        The relative kernel is verified centrosymmetric at construction, so
        its zero-extended convolution matrix is self-adjoint in the uniform
        grid pairing.
        """

        return self._linear_convolution(values)

    def evaluate(
        self,
        solute_electron_density_e_per_bohr3: np.ndarray,
    ) -> Route2V0OpenIsoDensityProductCavityState:
        """Map one nonnegative isolated electron density to smooth occupancy."""

        electron = _immutable_real_grid(
            solute_electron_density_e_per_bohr3,
            shape=self.grid.shape,
            name="Open iso-density-product solute electron density",
            nonnegative=True,
            nonzero=True,
        )
        overlap = self._convolution(electron)
        scale = max(1.0, float(np.max(np.abs(overlap))))
        if float(np.min(overlap)) < -128.0 * np.finfo(float).eps * scale:
            raise RuntimeError("Open iso-density-product overlap lost nonnegativity.")
        overlap = np.maximum(overlap, 0.0)
        positive = overlap > 0.0
        log_ratio = np.zeros(self.grid.shape, dtype=float)
        log_ratio[positive] = np.log(
            overlap[positive] / self.overlap_threshold_e2_per_bohr3
        )
        occupancy = np.ones(self.grid.shape, dtype=float)
        occupancy[positive] = 0.5 * erfc(log_ratio[positive])
        derivative = np.zeros(self.grid.shape, dtype=float)
        derivative[positive] = -np.exp(-(log_ratio[positive] ** 2)) / (
            math.sqrt(math.pi) * overlap[positive]
        )
        overlap.setflags(write=False)
        occupancy.setflags(write=False)
        derivative.setflags(write=False)
        return Route2V0OpenIsoDensityProductCavityState(
            solute_electron_density_e_per_bohr3=electron,
            overlap_density_e2_per_bohr3=overlap,
            occupancy=occupancy,
            occupancy_derivative_per_overlap_bohr3_per_e2=derivative,
            overlap_threshold_e2_per_bohr3=self.overlap_threshold_e2_per_bohr3,
            operator_fingerprint=self._operator_fingerprint,
        )

    def occupancy_coordinate_vjp(
        self,
        state: Route2V0OpenIsoDensityProductCavityState,
        occupancy_coordinate_gradient_hartree: np.ndarray,
    ) -> np.ndarray:
        """Pull ``dE/dm`` back to its electron-density functional derivative."""

        if not isinstance(state, Route2V0OpenIsoDensityProductCavityState):
            raise TypeError("Open iso-density-product VJP requires a cavity state.")
        if state.operator_fingerprint != self._operator_fingerprint:
            raise ValueError(
                "Open iso-density-product state does not match this cavity."
            )
        gradient = _immutable_real_grid(
            occupancy_coordinate_gradient_hartree,
            shape=self.grid.shape,
            name="Open iso-density-product occupancy coordinate gradient",
        )
        local = (
            state.occupancy_derivative_per_overlap_bohr3_per_e2
            * gradient
            / self.grid.volume_element_bohr3
        )
        result = self._adjoint_convolution(local)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class Route2V0OpenIsoDensityProductReactionState:
    """One composed isolated-density open reaction-field state."""

    nuclear_charge_density_e_per_bohr3: np.ndarray
    solute_electron_density_e_per_bohr3: np.ndarray
    total_charge_density_e_per_bohr3: np.ndarray
    cavity_state: Route2V0OpenIsoDensityProductCavityState
    continuum_state: Route2V0OpenDiffuseDielectricReactionState
    bulk_dielectric_constant: float
    cavity_operator_fingerprint: str
    construction: str = V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    response_scope: str = V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE

    def __post_init__(self) -> None:
        shape = np.asarray(self.nuclear_charge_density_e_per_bohr3).shape
        if len(shape) != 3:
            raise ValueError(
                "Open iso-density-product reaction state nuclear density must be 3D."
            )
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        nuclear = _immutable_real_grid(
            self.nuclear_charge_density_e_per_bohr3,
            shape=grid_shape,
            name="Open iso-density-product reaction state nuclear density",
            nonnegative=True,
        )
        electron = _immutable_real_grid(
            self.solute_electron_density_e_per_bohr3,
            shape=grid_shape,
            name="Open iso-density-product reaction state electron density",
            nonnegative=True,
            nonzero=True,
        )
        charge = _immutable_real_grid(
            self.total_charge_density_e_per_bohr3,
            shape=grid_shape,
            name="Open iso-density-product reaction state total charge density",
        )
        expected = nuclear - electron
        tolerance = (
            128.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.max(np.abs(expected))),
            )
        )
        if not np.allclose(charge, expected, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Open iso-density-product reaction charge split is invalid."
            )
        if not isinstance(self.cavity_state, Route2V0OpenIsoDensityProductCavityState):
            raise TypeError(
                "Open iso-density-product reaction requires a cavity state."
            )
        if not isinstance(
            self.continuum_state,
            Route2V0OpenDiffuseDielectricReactionState,
        ):
            raise TypeError(
                "Open iso-density-product reaction requires an open continuum state."
            )
        if not np.array_equal(
            electron,
            self.cavity_state.solute_electron_density_e_per_bohr3,
        ):
            raise ValueError(
                "Open iso-density-product reaction electron density/cavity mismatch."
            )
        if not np.array_equal(
            self.cavity_state.occupancy, self.continuum_state.occupancy
        ):
            raise ValueError(
                "Open iso-density-product cavity/continuum occupancy mismatch."
            )
        if not np.array_equal(charge, self.continuum_state.charge_density_e_per_bohr3):
            raise ValueError(
                "Open iso-density-product reaction charge/continuum mismatch."
            )
        dielectric = _finite_positive(
            self.bulk_dielectric_constant,
            name="Open iso-density-product reaction bulk dielectric",
        )
        if dielectric <= 1.0:
            raise ValueError(
                "Open iso-density-product reaction bulk dielectric must exceed one."
            )
        if (
            not isinstance(self.cavity_operator_fingerprint, str)
            or _DIGEST.fullmatch(self.cavity_operator_fingerprint) is None
        ):
            raise ValueError(
                "Open iso-density-product reaction cavity fingerprint is invalid."
            )
        if self.cavity_operator_fingerprint != self.cavity_state.operator_fingerprint:
            raise ValueError(
                "Open iso-density-product reaction cavity fingerprint mismatch."
            )
        if not math.isclose(
            self.continuum_state.bulk_dielectric_constant,
            dielectric,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Open iso-density-product reaction continuum dielectric mismatch."
            )
        if self.construction != V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION:
            raise ValueError(
                "Unsupported open iso-density-product reaction construction."
            )
        if self.response_scope != V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE:
            raise ValueError("Unsupported open iso-density-product reaction scope.")
        object.__setattr__(self, "nuclear_charge_density_e_per_bohr3", nuclear)
        object.__setattr__(self, "solute_electron_density_e_per_bohr3", electron)
        object.__setattr__(self, "total_charge_density_e_per_bohr3", charge)
        object.__setattr__(self, "bulk_dielectric_constant", dielectric)


@dataclass(frozen=True)
class Route2V0OpenIsoDensityProductReactionOperator:
    """Compose the open iso-density cavity with one open reaction scalar."""

    cavity: Route2V0OpenIsoDensityProductCavity
    bulk_dielectric_constant: float
    cg_relative_tolerance: float = 1.0e-12
    cg_max_iterations: int | None = None
    construction: str = V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    response_scope: str = V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.cavity, Route2V0OpenIsoDensityProductCavity):
            raise TypeError(
                "Open iso-density-product reaction requires an open iso-density cavity."
            )
        dielectric = _finite_positive(
            self.bulk_dielectric_constant,
            name="Open iso-density-product reaction bulk dielectric",
        )
        if dielectric <= 1.0:
            raise ValueError(
                "Open iso-density-product reaction bulk dielectric must exceed one."
            )
        if self.construction != V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 open iso-density-product reaction construction."
            )
        if self.response_scope != V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE:
            raise ValueError(
                "Unsupported Route-2 open iso-density-product reaction scope."
            )
        object.__setattr__(self, "bulk_dielectric_constant", dielectric)

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false: no electronic or nonpolar stationary scalar is present."""

        return False

    def _continuum(
        self,
        occupancy: np.ndarray,
    ) -> Route2V0OpenDiffuseLocalDielectricOperator:
        return Route2V0OpenDiffuseLocalDielectricOperator(
            grid=self.cavity.grid,
            occupancy=occupancy,
            bulk_dielectric_constant=self.bulk_dielectric_constant,
            cg_relative_tolerance=self.cg_relative_tolerance,
            cg_max_iterations=self.cg_max_iterations,
        )

    def _validate_reaction_state(
        self,
        state: Route2V0OpenIsoDensityProductReactionState,
    ) -> None:
        if not isinstance(state, Route2V0OpenIsoDensityProductReactionState):
            raise TypeError(
                "Open iso-density-product electron potential requires a reaction state."
            )
        if state.cavity_operator_fingerprint != self.cavity.operator_fingerprint:
            raise ValueError(
                "Open iso-density-product reaction state does not match this cavity."
            )
        if not math.isclose(
            state.bulk_dielectric_constant,
            self.bulk_dielectric_constant,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Open iso-density-product reaction state bulk dielectric does not match."
            )

    def solve(
        self,
        nuclear_charge_density_e_per_bohr3: np.ndarray,
        solute_electron_density_e_per_bohr3: np.ndarray,
    ) -> Route2V0OpenIsoDensityProductReactionState:
        """Solve the common open scalar for ``rho = rho_nuc - n`` and ``m[n]``."""

        nuclear = _immutable_real_grid(
            nuclear_charge_density_e_per_bohr3,
            shape=self.cavity.grid.shape,
            name="Open iso-density-product nuclear charge density",
            nonnegative=True,
        )
        cavity_state = self.cavity.evaluate(solute_electron_density_e_per_bohr3)
        charge = np.array(
            nuclear - cavity_state.solute_electron_density_e_per_bohr3,
            dtype=float,
            copy=True,
        )
        continuum = self._continuum(cavity_state.occupancy)
        continuum_state = continuum.solve(charge)
        return Route2V0OpenIsoDensityProductReactionState(
            nuclear_charge_density_e_per_bohr3=nuclear,
            solute_electron_density_e_per_bohr3=(
                cavity_state.solute_electron_density_e_per_bohr3
            ),
            total_charge_density_e_per_bohr3=charge,
            cavity_state=cavity_state,
            continuum_state=continuum_state,
            bulk_dielectric_constant=self.bulk_dielectric_constant,
            cavity_operator_fingerprint=self.cavity.operator_fingerprint,
        )

    def reaction_energy_hartree(
        self,
        nuclear_charge_density_e_per_bohr3: np.ndarray,
        solute_electron_density_e_per_bohr3: np.ndarray,
    ) -> float:
        """Return only the composed open electrostatic reaction scalar."""

        return self.solve(
            nuclear_charge_density_e_per_bohr3,
            solute_electron_density_e_per_bohr3,
        ).continuum_state.reaction_energy_hartree

    def electron_density_reaction_potential_hartree_per_e(
        self,
        state: Route2V0OpenIsoDensityProductReactionState,
    ) -> np.ndarray:
        """Return the exact derivative of the same scalar with respect to ``n``.

        With ``rho = rho_nuc - n``, the potential is

        ``-phi_reac + K.T[h'(K n) * dG_reac/dm]``.

        The first term is the charge response and the second is the open
        density-cavity chain rule.  A future stationary electronic Euler
        equation must receive their sum, not an independently assembled
        cavity force.
        """

        self._validate_reaction_state(state)
        continuum = self._continuum(state.cavity_state.occupancy)
        occupancy_gradient = continuum.occupancy_energy_gradient_hartree(
            state.continuum_state
        )
        cavity_term = self.cavity.occupancy_coordinate_vjp(
            state.cavity_state,
            occupancy_gradient,
        )
        potential = np.array(
            -state.continuum_state.reaction_potential_hartree_per_e + cavity_term,
            dtype=float,
            copy=True,
        )
        potential.setflags(write=False)
        return potential


__all__ = [
    "V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION",
    "V0_OPEN_ISO_DENSITY_PRODUCT_CAVITY_SCOPE",
    "V0_OPEN_ISO_DENSITY_PRODUCT_CONVOLUTION",
    "Route2V0OpenIsoDensityProductCavity",
    "Route2V0OpenIsoDensityProductCavityState",
    "Route2V0OpenIsoDensityProductReactionOperator",
    "Route2V0OpenIsoDensityProductReactionState",
]
