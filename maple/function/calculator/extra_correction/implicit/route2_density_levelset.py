"""Source-dependent reconstructed-density level set for Route-2 rho-DROP.

This module is the only rho-DROP boundary that knows all three of:

* Route-2's raw MACE-POLAR ``l=1`` component order;
* the public angstrom geometry/dipole convention; and
* MOIST-facing bohr coordinates and spatial derivatives.

The reconstructed quantity is deliberately named and documented as

``n_rec(r; c, R) = n_reference(r; R) - rho_residual(r; c, R)``.

It is not represented as a trained full electron density.  The corresponding
level set is ``S = 1 - n_rec/n_iso`` and follows the negative-inside convention
required by the planned external-isodensity callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from ase.units import Bohr

from .electrostatic_pairing import MACE_POLAR_L1_PAIRING
from .gto_density import MACE_POLAR_DENSITY_SIGMA_ANGSTROM
from .route2_atomic_reference_density import (
    AtomicReferenceDensity,
    AtomicReferenceDensityAsset,
    DensitySpatialJet,
)


def _readonly_finite_array(
    values: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _points(values: object, *, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(f"{name} must be finite with shape (n_points, 3).")
    return points


def _source_block(
    values: object,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if source.shape != (atom_count, 4) or not np.all(np.isfinite(source)):
        raise ValueError(
            f"{name} must be finite with shape ({atom_count}, 4); "
            f"received {source.shape}."
        )
    return source


def _atomic_numbers(values: object, *, atom_count: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (atom_count,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"Atomic numbers must be finite with shape ({atom_count},).")
    numeric = np.asarray(raw, dtype=float)
    rounded = np.rint(numeric)
    if np.any(rounded != numeric) or np.any(rounded < 1.0):
        raise ValueError("Atomic numbers must be positive integers.")
    result = rounded.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SpatialLSFJet:
    """Dimensionless level-set value and bohr-based spatial derivatives."""

    value: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    third: np.ndarray

    def __post_init__(self) -> None:
        value = _readonly_finite_array(self.value, name="Level-set value", ndim=1)
        gradient = _readonly_finite_array(
            self.gradient,
            name="Level-set gradient",
            ndim=2,
        )
        hessian = _readonly_finite_array(
            self.hessian,
            name="Level-set Hessian",
            ndim=3,
        )
        third = _readonly_finite_array(
            self.third,
            name="Level-set third derivative",
            ndim=4,
        )
        point_count = value.shape[0]
        if gradient.shape != (point_count, 3):
            raise ValueError("Level-set gradient must have shape (n_points, 3).")
        if hessian.shape != (point_count, 3, 3):
            raise ValueError("Level-set Hessian must have shape (n_points, 3, 3).")
        if third.shape != (point_count, 3, 3, 3):
            raise ValueError(
                "Level-set third derivative must have shape (n_points, 3, 3, 3)."
            )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "third", third)

    @classmethod
    def from_density_jet(cls, jet: DensitySpatialJet) -> "SpatialLSFJet":
        if not isinstance(jet, DensitySpatialJet):
            raise TypeError("A spatial level-set jet requires a density jet.")
        return cls(jet.value, jet.gradient, jet.hessian, jet.third)

    def scaled(self, factor: float) -> "SpatialLSFJet":
        scale = float(factor)
        if not np.isfinite(scale):
            raise ValueError("Level-set jet scale must be finite.")
        return SpatialLSFJet(
            value=scale * self.value,
            gradient=scale * self.gradient,
            hessian=scale * self.hessian,
            third=scale * self.third,
        )


@dataclass(frozen=True)
class LSFAdjointWeights:
    """Weights returned by the planned MOIST surface-LSF contraction.

    The Euclidean contraction is

    ``w0*S + w1.grad(S) + W2:Hess(S)``.

    Only the symmetric part of ``W2`` contributes because the analytic Hessian
    is symmetric; it is frozen here to remove an otherwise invisible null mode.
    """

    value: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray

    def __post_init__(self) -> None:
        value = _readonly_finite_array(self.value, name="LSF value weights", ndim=1)
        gradient = _readonly_finite_array(
            self.gradient,
            name="LSF gradient weights",
            ndim=2,
        )
        hessian = _readonly_finite_array(
            self.hessian,
            name="LSF Hessian weights",
            ndim=3,
        )
        point_count = value.shape[0]
        if gradient.shape != (point_count, 3):
            raise ValueError("LSF gradient weights must have shape (n_points, 3).")
        if hessian.shape != (point_count, 3, 3):
            raise ValueError("LSF Hessian weights must have shape (n_points, 3, 3).")
        symmetric_hessian = 0.5 * (hessian + np.swapaxes(hessian, -1, -2))
        symmetric_hessian.setflags(write=False)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian", symmetric_hessian)


@dataclass(frozen=True)
class DensityBoundaryDiagnostics:
    """Fail-closed local and electron-count checks for one point set."""

    minimum_reconstructed_density_e_per_bohr3: float
    minimum_level_set_gradient_norm_bohr: float
    maximum_level_set_residual: float
    integrated_electron_count: float
    expected_electron_count: float
    electron_count_absolute_error: float
    require_surface: bool
    passed: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "minimum_reconstructed_density_e_per_bohr3",
            "minimum_level_set_gradient_norm_bohr",
            "maximum_level_set_residual",
            "integrated_electron_count",
            "expected_electron_count",
            "electron_count_absolute_error",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"Boundary diagnostic {name} must be finite.")
        if not isinstance(self.require_surface, bool) or not isinstance(
            self.passed, bool
        ):
            raise TypeError("Boundary diagnostic flags must be bool.")
        if not isinstance(self.reasons, tuple) or not all(
            isinstance(reason, str) and reason for reason in self.reasons
        ):
            raise ValueError("Boundary diagnostic reasons must be nonempty strings.")


@runtime_checkable
class DensityLevelSetProvider(Protocol):
    """Model-neutral first-order source/coordinate rho-DROP contract."""

    atom_count: int

    def evaluate_spatial(self, points_bohr: object) -> SpatialLSFJet: ...

    def source_jvp(
        self,
        points_bohr: object,
        source_direction: object,
    ) -> SpatialLSFJet: ...

    def source_vjp(
        self,
        points_bohr: object,
        weights: LSFAdjointWeights,
    ) -> np.ndarray: ...

    def nuclear_vjp(
        self,
        points_bohr: object,
        weights: LSFAdjointWeights,
    ) -> np.ndarray: ...


class ReconstructedMacePolarDensityLevelSet:
    """Experimental ``n_reference-rho_residual`` rho-DROP level set."""

    density_identity = "reconstructed-MACE-polar-density"
    level_set_identity = "route2-rhodrop-scaled-isodensity-level-set-v1"

    def __init__(
        self,
        asset: AtomicReferenceDensityAsset,
        atomic_numbers: object,
        atom_positions_angstrom: object,
        density_coefficients: object,
        n_iso_e_per_bohr3: float,
        *,
        expected_total_charge_e: float,
        sigma_angstrom: float = MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
        minimum_density_e_per_bohr3: float = 0.0,
        minimum_gradient_norm_bohr: float = 1.0e-10,
        surface_tolerance: float = 1.0e-9,
        electron_count_tolerance: float = 5.0e-12,
    ) -> None:
        if not isinstance(asset, AtomicReferenceDensityAsset):
            raise TypeError(
                "Reconstructed density level set requires a verified asset."
            )
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("Atom positions must be finite with shape (n_atoms, 3).")
        atom_count = int(positions.shape[0])
        numbers = _atomic_numbers(atomic_numbers, atom_count=atom_count)
        source = _source_block(
            density_coefficients,
            atom_count=atom_count,
            name="MACE-POLAR density coefficients",
        )
        n_iso = float(n_iso_e_per_bohr3)
        sigma = float(sigma_angstrom)
        minimum_density = float(minimum_density_e_per_bohr3)
        minimum_gradient = float(minimum_gradient_norm_bohr)
        surface_tol = float(surface_tolerance)
        count_tol = float(electron_count_tolerance)
        if not np.isfinite(n_iso) or n_iso <= 0.0:
            raise ValueError("rho-DROP isodensity value must be positive and finite.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("MACE-POLAR residual Gaussian width must be positive.")
        if not np.isfinite(minimum_density) or minimum_density < 0.0:
            raise ValueError("Minimum reconstructed density must be nonnegative.")
        if (
            not np.isfinite(minimum_gradient)
            or minimum_gradient <= 0.0
            or not np.isfinite(surface_tol)
            or surface_tol <= 0.0
            or not np.isfinite(count_tol)
            or count_tol <= 0.0
        ):
            raise ValueError("rho-DROP diagnostic tolerances must be positive.")

        source_total_charge = float(np.sum(source[:, 0]))
        expected_charge = float(expected_total_charge_e)
        if not np.isfinite(expected_charge):
            raise ValueError("Expected total charge must be finite.")
        if abs(source_total_charge - expected_charge) > count_tol:
            raise ValueError(
                "MACE-POLAR source total charge does not match the independently "
                f"declared molecular charge ({source_total_charge:.16g} != "
                f"{expected_charge:.16g})."
            )

        self.asset = asset
        self.atomic_numbers = numbers
        self.atom_positions_angstrom = np.array(positions, dtype=float, copy=True)
        self.atom_positions_angstrom.setflags(write=False)
        self.positions_bohr = np.array(positions / Bohr, dtype=float, copy=True)
        self.positions_bohr.setflags(write=False)
        self.density_coefficients = np.array(source, dtype=float, copy=True)
        self.density_coefficients.setflags(write=False)
        self.n_iso_e_per_bohr3 = n_iso
        self.sigma_angstrom = sigma
        self.sigma_bohr = sigma / Bohr
        self.minimum_density_e_per_bohr3 = minimum_density
        self.minimum_gradient_norm_bohr = minimum_gradient
        self.surface_tolerance = surface_tol
        self.electron_count_tolerance = count_tol
        self.expected_total_charge_e = expected_charge
        self._reference = AtomicReferenceDensity(
            asset,
            numbers,
            self.positions_bohr,
        )

    @property
    def atom_count(self) -> int:
        return int(self.positions_bohr.shape[0])

    @property
    def expected_electron_count(self) -> float:
        return float(np.sum(self.atomic_numbers, dtype=float)) - float(
            self.expected_total_charge_e
        )

    @property
    def integrated_electron_count(self) -> float:
        # Every reference component integrates to N_k; the odd l=1 residual
        # term integrates to zero exactly.
        return self._reference.integrated_electron_count - float(
            np.sum(self.density_coefficients[:, 0])
        )

    @staticmethod
    def _to_cartesian_bohr_source(raw_source: np.ndarray) -> np.ndarray:
        """Map raw ``[q,l1_0,l1_1,l1_2]`` to ``[q,p_x,p_y,p_z]``.

        MACE dipoles are in elementary-charge angstrom.  The returned dipoles
        are elementary-charge bohr, which makes the residual density and all
        point coordinates internally atomic-unit consistent.
        """

        cartesian = MACE_POLAR_L1_PAIRING.density_to_field_order(raw_source)
        cartesian[:, 1:] /= Bohr
        return cartesian

    @staticmethod
    def _cartesian_bohr_cotangent_to_raw(
        cartesian_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Transpose the raw-order and angstrom-to-bohr source conversion."""

        cotangent = np.array(cartesian_cotangent, dtype=float, copy=True)
        cotangent[:, 1:] /= Bohr
        return MACE_POLAR_L1_PAIRING.field_to_density_order(cotangent)

    def _residual_atom_jet(
        self,
        points_bohr: np.ndarray,
        atom_index: int,
        *,
        cartesian_bohr_source: np.ndarray,
    ) -> DensitySpatialJet:
        x = points_bohr - self.positions_bohr[atom_index]
        q = float(cartesian_bohr_source[atom_index, 0])
        dipole = cartesian_bohr_source[atom_index, 1:]
        sigma = self.sigma_bohr
        a = sigma**-2
        b = a * dipole
        radius_squared = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2
        gaussian = (2.0 * np.pi * sigma**2) ** -1.5 * np.exp(-0.5 * a * radius_squared)
        h = q + x[:, 0] * b[0] + x[:, 1] * b[1] + x[:, 2] * b[2]
        value = gaussian * h
        gradient = gaussian[:, None] * (b[None, :] - a * x * h[:, None])
        identity = np.eye(3)
        hessian = np.empty((x.shape[0], 3, 3), dtype=float)
        third = np.empty((x.shape[0], 3, 3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                hessian[:, i, j] = gaussian * (
                    a**2 * x[:, i] * x[:, j] * h
                    - a * identity[i, j] * h
                    - a * (x[:, j] * b[i] + x[:, i] * b[j])
                )
                for k in range(3):
                    third[:, i, j, k] = gaussian * (
                        -(a**3) * x[:, i] * x[:, j] * x[:, k] * h
                        + a**2
                        * h
                        * (
                            x[:, k] * identity[i, j]
                            + x[:, j] * identity[i, k]
                            + x[:, i] * identity[j, k]
                        )
                        + a**2
                        * (
                            x[:, k] * x[:, j] * b[i]
                            + x[:, k] * x[:, i] * b[j]
                            + x[:, i] * x[:, j] * b[k]
                        )
                        - a
                        * (
                            identity[i, j] * b[k]
                            + identity[i, k] * b[j]
                            + identity[j, k] * b[i]
                        )
                    )
        return DensitySpatialJet(value, gradient, hessian, third)

    def _residual_jet(
        self,
        points_bohr: np.ndarray,
        *,
        raw_source: np.ndarray,
    ) -> DensitySpatialJet:
        cartesian_bohr_source = self._to_cartesian_bohr_source(raw_source)
        total = DensitySpatialJet.zeros(points_bohr.shape[0])
        for atom_index in range(self.atom_count):
            total = total.plus(
                self._residual_atom_jet(
                    points_bohr,
                    atom_index,
                    cartesian_bohr_source=cartesian_bohr_source,
                )
            )
        return total

    def evaluate_reconstructed_density(
        self,
        points_bohr: object,
    ) -> DensitySpatialJet:
        """Return ``n_reference-rho_residual`` and its spatial derivatives."""

        points = _points(points_bohr, name="Reconstructed-density evaluation points")
        reference = self._reference.evaluate_spatial(points)
        residual = self._residual_jet(
            points,
            raw_source=self.density_coefficients,
        )
        return reference.plus(residual.scaled(-1.0))

    def evaluate_spatial(self, points_bohr: object) -> SpatialLSFJet:
        density = self.evaluate_reconstructed_density(points_bohr)
        inverse_isovalue = 1.0 / self.n_iso_e_per_bohr3
        return SpatialLSFJet(
            value=1.0 - inverse_isovalue * density.value,
            gradient=-inverse_isovalue * density.gradient,
            hessian=-inverse_isovalue * density.hessian,
            third=-inverse_isovalue * density.third,
        )

    def source_jvp(
        self,
        points_bohr: object,
        source_direction: object,
    ) -> SpatialLSFJet:
        points = _points(points_bohr, name="Level-set source-JVP points")
        direction = _source_block(
            source_direction,
            atom_count=self.atom_count,
            name="Level-set source direction",
        )
        # S = 1 - (n_ref-rho_res)/n_iso, so dS = d(rho_res)/n_iso.
        residual_direction = self._residual_jet(points, raw_source=direction)
        return SpatialLSFJet.from_density_jet(
            residual_direction.scaled(1.0 / self.n_iso_e_per_bohr3)
        )

    @staticmethod
    def _validated_weights(
        weights: LSFAdjointWeights,
        *,
        point_count: int,
    ) -> LSFAdjointWeights:
        if not isinstance(weights, LSFAdjointWeights):
            raise TypeError("rho-DROP adjoints require LSFAdjointWeights.")
        if weights.value.shape != (point_count,):
            raise ValueError(
                "LSF adjoint point count does not match evaluation points."
            )
        return weights

    def source_vjp(
        self,
        points_bohr: object,
        weights: LSFAdjointWeights,
    ) -> np.ndarray:
        """Contract the level-set response directly into raw source space."""

        points = _points(points_bohr, name="Level-set source-VJP points")
        weights = self._validated_weights(weights, point_count=points.shape[0])
        identity = np.eye(3)
        a = self.sigma_bohr**-2
        cartesian_cotangent = np.zeros((self.atom_count, 4), dtype=float)
        for atom_index in range(self.atom_count):
            x = points - self.positions_bohr[atom_index]
            radius_squared = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2
            gaussian = (2.0 * np.pi * self.sigma_bohr**2) ** -1.5 * np.exp(
                -0.5 * a * radius_squared
            )

            q_gradient = -a * gaussian[:, None] * x
            q_hessian = gaussian[:, None, None] * (
                a**2 * np.einsum("mi,mj->mij", x, x) - a * identity[None, :, :]
            )
            cartesian_cotangent[atom_index, 0] = (
                np.vdot(weights.value, gaussian)
                + np.vdot(weights.gradient, q_gradient)
                + np.vdot(weights.hessian, q_hessian)
            )

            p_value = a * gaussian[:, None] * x
            p_gradient = (
                a
                * gaussian[:, None, None]
                * (identity[None, :, :] - a * np.einsum("mi,mj->mij", x, x))
            )
            p_hessian = np.empty((points.shape[0], 3, 3, 3), dtype=float)
            for i in range(3):
                for j in range(3):
                    for ell in range(3):
                        p_hessian[:, i, j, ell] = gaussian * (
                            a**3 * x[:, i] * x[:, j] * x[:, ell]
                            - a**2
                            * (
                                identity[i, j] * x[:, ell]
                                + identity[i, ell] * x[:, j]
                                + identity[j, ell] * x[:, i]
                            )
                        )
            cartesian_cotangent[atom_index, 1:] = (
                np.einsum("m,ml->l", weights.value, p_value)
                + np.einsum("mi,mil->l", weights.gradient, p_gradient)
                + np.einsum("mij,mijl->l", weights.hessian, p_hessian)
            )

        cartesian_cotangent /= self.n_iso_e_per_bohr3
        result = self._cartesian_bohr_cotangent_to_raw(cartesian_cotangent)
        result.setflags(write=False)
        return result

    def nuclear_vjp(
        self,
        points_bohr: object,
        weights: LSFAdjointWeights,
    ) -> np.ndarray:
        """Return the explicit level-set field VJP per atom and angstrom.

        Surface points are held fixed.  This is the rho-DROP ``field`` partial;
        it intentionally excludes the reference-anchor/geometric contribution
        that a future MOIST wrapper must contract separately.
        """

        points = _points(points_bohr, name="Level-set nuclear-VJP points")
        weights = self._validated_weights(weights, point_count=points.shape[0])
        result = np.zeros((self.atom_count, 3), dtype=float)
        cartesian_bohr_source = self._to_cartesian_bohr_source(
            self.density_coefficients
        )
        for atom_index in range(self.atom_count):
            reference = self._reference.atom_spatial_jet(points, atom_index)
            residual = self._residual_atom_jet(
                points,
                atom_index,
                cartesian_bohr_source=cartesian_bohr_source,
            )
            reconstructed_atom = reference.plus(residual.scaled(-1.0))
            # S_A = -n_rec,A/n_iso and f(r-R_A) gives
            # d_R S_A = +grad(n_rec,A)/n_iso.  Convert d/dR_bohr to d/dR_Angstrom.
            result[atom_index] = (
                np.einsum("m,mk->k", weights.value, reconstructed_atom.gradient)
                + np.einsum(
                    "mi,mik->k",
                    weights.gradient,
                    reconstructed_atom.hessian,
                )
                + np.einsum(
                    "mij,mijk->k",
                    weights.hessian,
                    reconstructed_atom.third,
                )
            ) / (self.n_iso_e_per_bohr3 * Bohr)
        result.setflags(write=False)
        return result

    def diagnose(
        self,
        points_bohr: object,
        *,
        require_surface: bool = True,
    ) -> DensityBoundaryDiagnostics:
        if not isinstance(require_surface, bool):
            raise TypeError("require_surface must be bool.")
        points = _points(points_bohr, name="Boundary diagnostic points")
        density = self.evaluate_reconstructed_density(points)
        level_set = self.evaluate_spatial(points)
        minimum_density = float(np.min(density.value))
        minimum_gradient = float(np.min(np.linalg.norm(level_set.gradient, axis=1)))
        maximum_residual = float(np.max(np.abs(level_set.value)))
        count_error = abs(self.integrated_electron_count - self.expected_electron_count)
        reasons: list[str] = []
        if minimum_density < self.minimum_density_e_per_bohr3:
            reasons.append(
                "negative/too-small reconstructed density: "
                f"{minimum_density:.6e} e/bohr^3"
            )
        if minimum_gradient < self.minimum_gradient_norm_bohr:
            reasons.append(
                "near-critical level-set gradient: " f"{minimum_gradient:.6e} bohr^-1"
            )
        if require_surface and maximum_residual > self.surface_tolerance:
            reasons.append(
                f"surface level-set residual is too large: {maximum_residual:.6e}"
            )
        if count_error > self.electron_count_tolerance:
            reasons.append(
                f"reconstructed electron-count error is too large: {count_error:.6e}"
            )
        return DensityBoundaryDiagnostics(
            minimum_reconstructed_density_e_per_bohr3=minimum_density,
            minimum_level_set_gradient_norm_bohr=minimum_gradient,
            maximum_level_set_residual=maximum_residual,
            integrated_electron_count=self.integrated_electron_count,
            expected_electron_count=self.expected_electron_count,
            electron_count_absolute_error=count_error,
            require_surface=require_surface,
            passed=not reasons,
            reasons=tuple(reasons),
        )

    def require_valid_boundary(
        self,
        points_bohr: object,
        validation_shell_points_bohr: object | None = None,
    ) -> DensityBoundaryDiagnostics:
        boundary = self.diagnose(points_bohr, require_surface=True)
        reasons = list(boundary.reasons)
        if validation_shell_points_bohr is not None:
            shell = self.diagnose(
                validation_shell_points_bohr,
                require_surface=False,
            )
            reasons.extend(f"validation shell: {reason}" for reason in shell.reasons)
        if reasons:
            raise ValueError(
                "rho-DROP reconstructed-density boundary admission failed: "
                + "; ".join(reasons)
            )
        return boundary


__all__ = [
    "DensityBoundaryDiagnostics",
    "DensityLevelSetProvider",
    "LSFAdjointWeights",
    "ReconstructedMacePolarDensityLevelSet",
    "SpatialLSFJet",
]
