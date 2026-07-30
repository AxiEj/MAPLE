"""Nonlocal iso-density-product cavity for the Route-2 V0-AQ-C scalar.

This module makes a smooth solvent occupancy a *deterministic differentiable
functional* of an auxiliary solute electron density instead of an atomic-radius
cavity or an independently moved grid.  For an orientation-averaged isolated
solvent electron-density kernel ``n_lq`` and a frozen overlap threshold ``q_c``,

``m[n](r) = 1/2 erfc(log((n_lq * n)(r) / q_c))``.

The kernel convolution and its exact discrete adjoint are both explicit.  When
this cavity is composed with the diffuse reaction scalar, the returned
auxiliary-electron potential contains both the ordinary charge response and
the cavity chain-rule term.  Consequently it is a derivative of the same
scalar, not a post-hoc cavity force.

The construction is inspired by the nonlocal iso-density-product cavity used
in SaLSA.  It deliberately does not supply a solvent electronic-density asset,
a source-bound threshold, a nonpolar/dispersion functional, a standard-state
term, or an auxiliary electronic functional.  It therefore remains a
structural V0-AQ-C block and cannot report a solvation free energy or chemical
accuracy.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

from .route2_v0_diffuse_continuum import (
    Route2V0DiffuseDielectricReactionState,
    Route2V0DiffuseLocalDielectricOperator,
)
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION = "route2-v0-iso-density-product-cavity-v1"
V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE = "nonlocal-cavity-electrostatic-block-only-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _finite_positive(value: object, *, name: str) -> float:
    """Return a finite, positive scalar without accepting boolean values."""

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
    """Validate and freeze one real grid field."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real grid field.") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if nonzero and not np.any(array > 0.0):
        raise ValueError(f"{name} must not vanish everywhere.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _operator_fingerprint(
    *,
    grid: RegularCartesianGrid,
    kernel: np.ndarray,
    overlap_threshold: float,
) -> str:
    """Return a content fingerprint that prevents cross-cavity derivatives."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(grid.origin_bohr, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(grid.spacing_bohr, dtype=np.float64).tobytes())
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(kernel, dtype=np.float64).tobytes())
    digest.update(np.asarray([overlap_threshold], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _real_fft_result(values: np.ndarray, *, name: str) -> np.ndarray:
    """Reject a non-negligible imaginary numerical residue from a real FFT map."""

    real = np.real(values)
    imaginary_scale = float(np.max(np.abs(np.imag(values))))
    scale = max(1.0, float(np.max(np.abs(real))))
    if imaginary_scale > 128.0 * np.finfo(float).eps * scale:
        raise RuntimeError(f"{name} produced a non-real FFT result.")
    return np.array(real, dtype=float, copy=True)


@dataclass(frozen=True)
class Route2V0IsoDensityProductCavityState:
    """One smooth occupancy field and its exact derivative data."""

    solute_electron_density_e_per_bohr3: np.ndarray
    overlap_density_e2_per_bohr3: np.ndarray
    occupancy: np.ndarray
    occupancy_derivative_per_overlap_bohr3_per_e2: np.ndarray
    overlap_threshold_e2_per_bohr3: float
    operator_fingerprint: str
    construction: str = V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    response_scope: str = V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE

    def __post_init__(self) -> None:
        shape = np.asarray(self.solute_electron_density_e_per_bohr3).shape
        if len(shape) != 3:
            raise ValueError("Iso-density-product state electron density must be 3D.")
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        electron_density = _immutable_real_grid(
            self.solute_electron_density_e_per_bohr3,
            shape=grid_shape,
            name="Iso-density-product state electron density",
            nonnegative=True,
            nonzero=True,
        )
        overlap = _immutable_real_grid(
            self.overlap_density_e2_per_bohr3,
            shape=grid_shape,
            name="Iso-density-product state overlap density",
            nonnegative=True,
        )
        occupancy = _immutable_real_grid(
            self.occupancy,
            shape=grid_shape,
            name="Iso-density-product state occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError("Iso-density-product state occupancy must lie in [0, 1].")
        derivative = _immutable_real_grid(
            self.occupancy_derivative_per_overlap_bohr3_per_e2,
            shape=grid_shape,
            name="Iso-density-product occupancy derivative",
        )
        if np.any(derivative > 0.0):
            raise ValueError(
                "Iso-density-product occupancy derivative must be nonpositive."
            )
        threshold = _finite_positive(
            self.overlap_threshold_e2_per_bohr3,
            name="Iso-density-product overlap threshold",
        )
        fingerprint = self.operator_fingerprint
        if not isinstance(fingerprint, str) or _DIGEST.fullmatch(fingerprint) is None:
            raise ValueError(
                "Iso-density-product state operator fingerprint is invalid."
            )
        if self.construction != V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION:
            raise ValueError("Unsupported iso-density-product state construction.")
        if self.response_scope != V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE:
            raise ValueError("Unsupported iso-density-product state response scope.")
        object.__setattr__(
            self, "solute_electron_density_e_per_bohr3", electron_density
        )
        object.__setattr__(self, "overlap_density_e2_per_bohr3", overlap)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(
            self, "occupancy_derivative_per_overlap_bohr3_per_e2", derivative
        )
        object.__setattr__(self, "overlap_threshold_e2_per_bohr3", threshold)


@dataclass(frozen=True)
class Route2V0IsoDensityProductCavity:
    """A periodic nonlocal cavity map with an exact discrete adjoint.

    ``solvent_electron_density_kernel_e_per_bohr3`` is the spherical isolated
    solvent electron density sampled with its physical origin at index ``(0,0,0)``
    in the periodic-grid convention.  It must be a frozen source asset in a
    physical calculation; this generic class intentionally has no default
    solvent or overlap threshold.
    """

    grid: RegularCartesianGrid
    solvent_electron_density_kernel_e_per_bohr3: np.ndarray
    overlap_threshold_e2_per_bohr3: float
    construction: str = V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    response_scope: str = V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE
    _operator_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError(
                "Iso-density-product cavity requires a regular Cartesian grid."
            )
        if self.construction != V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 iso-density-product construction.")
        if self.response_scope != V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE:
            raise ValueError("Unsupported Route-2 iso-density-product response scope.")
        kernel = _immutable_real_grid(
            self.solvent_electron_density_kernel_e_per_bohr3,
            shape=self.grid.shape,
            name="Iso-density-product solvent electron-density kernel",
            nonnegative=True,
            nonzero=True,
        )
        threshold = _finite_positive(
            self.overlap_threshold_e2_per_bohr3,
            name="Iso-density-product overlap threshold",
        )
        fingerprint = _operator_fingerprint(
            grid=self.grid,
            kernel=kernel,
            overlap_threshold=threshold,
        )
        object.__setattr__(self, "solvent_electron_density_kernel_e_per_bohr3", kernel)
        object.__setattr__(self, "overlap_threshold_e2_per_bohr3", threshold)
        object.__setattr__(self, "_operator_fingerprint", fingerprint)

    @property
    def operator_fingerprint(self) -> str:
        """Return the immutable grid/kernel/threshold identity."""

        return self._operator_fingerprint

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false: the nonpolar/standard-state scalar is intentionally absent."""

        return False

    def _convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply ``K[n] = integral n_lq(r-r') n(r') dr'`` on the periodic grid."""

        raw = self.grid.volume_element_bohr3 * np.fft.ifftn(
            np.fft.fftn(self.solvent_electron_density_kernel_e_per_bohr3)
            * np.fft.fftn(values)
        )
        return _real_fft_result(raw, name="Iso-density-product convolution")

    def _adjoint_convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply the exact discrete adjoint of :meth:`_convolution`."""

        raw = self.grid.volume_element_bohr3 * np.fft.ifftn(
            np.conj(np.fft.fftn(self.solvent_electron_density_kernel_e_per_bohr3))
            * np.fft.fftn(values)
        )
        return _real_fft_result(raw, name="Iso-density-product adjoint convolution")

    def evaluate(
        self,
        solute_electron_density_e_per_bohr3: np.ndarray,
    ) -> Route2V0IsoDensityProductCavityState:
        """Map a nonnegative electron density to a smooth solvent occupancy."""

        electron_density = _immutable_real_grid(
            solute_electron_density_e_per_bohr3,
            shape=self.grid.shape,
            name="Iso-density-product solute electron density",
            nonnegative=True,
            nonzero=True,
        )
        overlap = self._convolution(electron_density)
        scale = max(1.0, float(np.max(np.abs(overlap))))
        if float(np.min(overlap)) < -128.0 * np.finfo(float).eps * scale:
            raise RuntimeError("Iso-density-product overlap lost nonnegativity.")
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
        occupancy.setflags(write=False)
        derivative.setflags(write=False)
        overlap.setflags(write=False)
        return Route2V0IsoDensityProductCavityState(
            solute_electron_density_e_per_bohr3=electron_density,
            overlap_density_e2_per_bohr3=overlap,
            occupancy=occupancy,
            occupancy_derivative_per_overlap_bohr3_per_e2=derivative,
            overlap_threshold_e2_per_bohr3=self.overlap_threshold_e2_per_bohr3,
            operator_fingerprint=self._operator_fingerprint,
        )

    def occupancy_coordinate_vjp(
        self,
        state: Route2V0IsoDensityProductCavityState,
        occupancy_coordinate_gradient_hartree: np.ndarray,
    ) -> np.ndarray:
        """Return ``delta E / delta n`` from a coordinate gradient ``dE/dm``.

        The input is a discrete gradient with respect to each occupancy value,
        as returned by :meth:`Route2V0DiffuseLocalDielectricOperator.
        occupancy_energy_gradient_hartree`.  The result is a functional
        derivative under the grid pairing ``dV * sum(delta_n * potential)``.
        """

        if not isinstance(state, Route2V0IsoDensityProductCavityState):
            raise TypeError("Iso-density-product VJP requires a cavity state.")
        if state.operator_fingerprint != self._operator_fingerprint:
            raise ValueError("Iso-density-product state does not match this cavity.")
        gradient = _immutable_real_grid(
            occupancy_coordinate_gradient_hartree,
            shape=self.grid.shape,
            name="Iso-density-product occupancy coordinate gradient",
        )
        functional_gradient = gradient / self.grid.volume_element_bohr3
        local = (
            state.occupancy_derivative_per_overlap_bohr3_per_e2 * functional_gradient
        )
        result = self._adjoint_convolution(local)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class Route2V0IsoDensityProductReactionState:
    """One composed electron-density-dependent diffuse reaction solve."""

    nuclear_charge_density_e_per_bohr3: np.ndarray
    solute_electron_density_e_per_bohr3: np.ndarray
    total_charge_density_e_per_bohr3: np.ndarray
    cavity_state: Route2V0IsoDensityProductCavityState
    continuum_state: Route2V0DiffuseDielectricReactionState
    bulk_dielectric_constant: float
    cavity_operator_fingerprint: str
    construction: str = V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    response_scope: str = V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE

    def __post_init__(self) -> None:
        shape = np.asarray(self.nuclear_charge_density_e_per_bohr3).shape
        if len(shape) != 3:
            raise ValueError(
                "Iso-density-product reaction state nuclear density must be 3D."
            )
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        nuclear = _immutable_real_grid(
            self.nuclear_charge_density_e_per_bohr3,
            shape=grid_shape,
            name="Iso-density-product reaction state nuclear density",
            nonnegative=True,
        )
        electron = _immutable_real_grid(
            self.solute_electron_density_e_per_bohr3,
            shape=grid_shape,
            name="Iso-density-product reaction state electron density",
            nonnegative=True,
            nonzero=True,
        )
        charge = _immutable_real_grid(
            self.total_charge_density_e_per_bohr3,
            shape=grid_shape,
            name="Iso-density-product reaction state charge density",
        )
        expected_charge = nuclear - electron
        tolerance = (
            128.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.max(np.abs(expected_charge))),
            )
        )
        if not np.allclose(charge, expected_charge, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Iso-density-product reaction state charge split is invalid."
            )
        if not isinstance(self.cavity_state, Route2V0IsoDensityProductCavityState):
            raise TypeError(
                "Iso-density-product reaction state requires a cavity state."
            )
        if not isinstance(self.continuum_state, Route2V0DiffuseDielectricReactionState):
            raise TypeError(
                "Iso-density-product reaction state requires a continuum state."
            )
        if not np.array_equal(
            self.cavity_state.occupancy, self.continuum_state.occupancy
        ):
            raise ValueError(
                "Iso-density-product reaction state cavity/continuum mismatch."
            )
        dielectric = _finite_positive(
            self.bulk_dielectric_constant,
            name="Iso-density-product reaction state bulk dielectric",
        )
        if dielectric <= 1.0:
            raise ValueError(
                "Iso-density-product reaction state bulk dielectric must exceed one."
            )
        fingerprint = self.cavity_operator_fingerprint
        if not isinstance(fingerprint, str) or _DIGEST.fullmatch(fingerprint) is None:
            raise ValueError(
                "Iso-density-product reaction state cavity fingerprint is invalid."
            )
        if self.construction != V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION:
            raise ValueError(
                "Unsupported iso-density-product reaction state construction."
            )
        if self.response_scope != V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE:
            raise ValueError(
                "Unsupported iso-density-product reaction state response scope."
            )
        object.__setattr__(self, "nuclear_charge_density_e_per_bohr3", nuclear)
        object.__setattr__(self, "solute_electron_density_e_per_bohr3", electron)
        object.__setattr__(self, "total_charge_density_e_per_bohr3", charge)
        object.__setattr__(self, "bulk_dielectric_constant", dielectric)


@dataclass(frozen=True)
class Route2V0IsoDensityProductReactionOperator:
    """Compose the nonlocal cavity with the diffuse reaction-energy scalar."""

    cavity: Route2V0IsoDensityProductCavity
    bulk_dielectric_constant: float
    cg_relative_tolerance: float = 1.0e-12
    cg_max_iterations: int | None = None
    construction: str = V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION
    response_scope: str = V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.cavity, Route2V0IsoDensityProductCavity):
            raise TypeError(
                "Iso-density-product reaction requires an iso-density cavity."
            )
        dielectric = _finite_positive(
            self.bulk_dielectric_constant,
            name="Iso-density-product reaction bulk dielectric",
        )
        if dielectric <= 1.0:
            raise ValueError(
                "Iso-density-product reaction bulk dielectric must exceed one."
            )
        if self.construction != V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 iso-density-product reaction construction."
            )
        if self.response_scope != V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE:
            raise ValueError(
                "Unsupported Route-2 iso-density-product reaction response scope."
            )
        object.__setattr__(self, "bulk_dielectric_constant", dielectric)

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false: this contains no auxiliary or nonpolar scalar."""

        return False

    def _continuum(
        self,
        occupancy: np.ndarray,
    ) -> Route2V0DiffuseLocalDielectricOperator:
        return Route2V0DiffuseLocalDielectricOperator(
            grid=self.cavity.grid,
            occupancy=occupancy,
            bulk_dielectric_constant=self.bulk_dielectric_constant,
            cg_relative_tolerance=self.cg_relative_tolerance,
            cg_max_iterations=self.cg_max_iterations,
        )

    def _validate_reaction_state(
        self,
        state: Route2V0IsoDensityProductReactionState,
    ) -> None:
        if not isinstance(state, Route2V0IsoDensityProductReactionState):
            raise TypeError(
                "Iso-density-product electron potential requires a reaction state."
            )
        if state.cavity_operator_fingerprint != self.cavity.operator_fingerprint:
            raise ValueError(
                "Iso-density-product reaction state does not match this cavity."
            )
        if not math.isclose(
            state.bulk_dielectric_constant,
            self.bulk_dielectric_constant,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Iso-density-product reaction state bulk dielectric does not match."
            )

    def solve(
        self,
        nuclear_charge_density_e_per_bohr3: np.ndarray,
        solute_electron_density_e_per_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> Route2V0IsoDensityProductReactionState:
        """Solve the reaction field for ``rho = rho_nuc - n`` and ``m[n]``."""

        nuclear = _immutable_real_grid(
            nuclear_charge_density_e_per_bohr3,
            shape=self.cavity.grid.shape,
            name="Iso-density-product nuclear charge density",
            nonnegative=True,
        )
        cavity_state = self.cavity.evaluate(solute_electron_density_e_per_bohr3)
        charge = np.array(
            nuclear - cavity_state.solute_electron_density_e_per_bohr3,
            dtype=float,
            copy=True,
        )
        continuum = self._continuum(cavity_state.occupancy)
        continuum_state = continuum.solve(
            charge,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        return Route2V0IsoDensityProductReactionState(
            nuclear_charge_density_e_per_bohr3=nuclear,
            solute_electron_density_e_per_bohr3=cavity_state.solute_electron_density_e_per_bohr3,
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
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> float:
        """Return the composed electrostatic reaction scalar only."""

        return self.solve(
            nuclear_charge_density_e_per_bohr3,
            solute_electron_density_e_per_bohr3,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        ).continuum_state.reaction_energy_hartree

    def electron_density_reaction_potential_hartree_per_e(
        self,
        state: Route2V0IsoDensityProductReactionState,
    ) -> np.ndarray:
        """Return the exact derivative of the composed reaction scalar wrt ``n``.

        With ``rho = rho_nuc - n`` this is

        ``-phi_reac + K_dagger[h'(K n) * delta G_reac / delta m]``.

        It is only the solvent reaction contribution to a future auxiliary
        electronic Euler equation; gas electronic and nonpolar terms belong to
        that same future scalar and are not inserted here.
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
        potential = (
            -state.continuum_state.reaction_potential_hartree_per_e + cavity_term
        )
        potential = np.array(potential, dtype=float, copy=True)
        potential.setflags(write=False)
        return potential


__all__ = [
    "V0_ISO_DENSITY_PRODUCT_CAVITY_CONSTRUCTION",
    "V0_ISO_DENSITY_PRODUCT_CAVITY_SCOPE",
    "Route2V0IsoDensityProductCavity",
    "Route2V0IsoDensityProductCavityState",
    "Route2V0IsoDensityProductReactionOperator",
    "Route2V0IsoDensityProductReactionState",
]
