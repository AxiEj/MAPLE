"""Nested pure-liquid Gaussian-kernel planar evidence for Route-2 V0.

At each fixed Gaussian weighted-density length ``sigma``, the quartic barrier
``B`` is fixed *only* by the same-scalar finite-density coexistence condition
``C(B, sigma)=0``.  Only after that inner solve may the exact full-SO(3)
planar restriction supply the bulk-subtracted surface tension ``Gamma(sigma)``.

The point evaluator below receives no surface-tension target: it rebuilds the
Gaussian kernel, solves the inner continuation, freezes its root ``B``, then
solves the planar state.  The companion bracket validator only checks already
computed nested points against the target frozen in the bridge asset.  It makes
no global monotonicity or uniqueness assertion about ``Gamma(sigma)``.

All current V0 assets remain controls.  This code is structural evidence for a
future source-bound physical certificate, never a physical-liquid admission,
solvation calculation, or accuracy result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import SupportsFloat, SupportsIndex, cast

import numpy as np

from .route2_v0_molecular_coexistence_continuation import (
    Route2V0MolecularQuarticCoexistenceContinuation,
    solve_route2_v0_molecular_quartic_coexistence_continuation,
)
from .route2_v0_molecular_planar_interface import (
    Route2V0MolecularConstrainedPlanarInterfaceState,
    Route2V0MolecularPlanarSymmetry,
    solve_route2_v0_molecular_constrained_planar_interface,
)
from .route2_v0_molecular_so3_quadrature import (
    Route2V0CartesianEulerProductQuadrature,
)
from .route2_v0_molecular_weighted_density_bridge import (
    V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_FAMILY_CUBIC_PLUS_QUARTIC,
    Route2V0MolecularWeightedDensityBridgeFunctional,
    Route2V0PeriodicWeightedDensityKernel,
    periodic_gaussian_weighted_density_kernel,
)

V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_POINT_CONSTRUCTION = (
    "route2-v0-molecular-gaussian-surface-tension-point-v1"
)
V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_BRACKET_CONSTRUCTION = (
    "route2-v0-molecular-gaussian-surface-tension-bracket-v1"
)
V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_BRANCH_SELECTION = (
    "declared-continuous-planar-branch-no-global-monotonicity-v1"
)


def _positive(value: object, *, name: str) -> float:
    """Return one finite positive real scalar without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be positive.")
    try:
        result = float(
            cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _same_float(left: float, right: float) -> bool:
    """Compare frozen float64 anchors at scale-aware roundoff tolerance."""

    return math.isclose(
        left,
        right,
        rel_tol=1.0e-12,
        abs_tol=1.0e-14 * max(1.0, abs(left), abs(right)),
    )


def _assert_cubic_plus_quartic(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
) -> None:
    if not isinstance(functional, Route2V0MolecularWeightedDensityBridgeFunctional):
        raise TypeError(
            "Gaussian surface-tension evidence requires a molecular bridge scalar."
        )
    if (
        functional.bridge_asset.bridge_family
        != V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_FAMILY_CUBIC_PLUS_QUARTIC
    ):
        raise ValueError(
            "Gaussian surface-tension evidence requires the cubic-plus-quartic "
            "bridge family."
        )


def _kernel_matches_gaussian(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    gaussian_width_bohr: float,
) -> bool:
    """Return whether the live discrete kernel is the declared Gaussian."""

    observed = functional.bridge_asset.kernel.kernel_bohr_minus3
    expected = periodic_gaussian_weighted_density_kernel(
        functional.bridge_asset.kernel.grid,
        gaussian_width_bohr=gaussian_width_bohr,
    )
    tolerance = 1.0e-12 * max(
        1.0,
        float(np.max(np.abs(observed))),
        float(np.max(np.abs(expected))),
    )
    return bool(np.allclose(observed, expected, rtol=1.0e-12, atol=tolerance))


def _gaussian_trial_scalar(
    template: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    gaussian_width_bohr: float,
) -> Route2V0MolecularWeightedDensityBridgeFunctional:
    """Replace only ``K`` for one no-certificate inner continuation trial."""

    _assert_cubic_plus_quartic(template)
    width = _positive(gaussian_width_bohr, name="Gaussian surface-tension width")
    kernel = Route2V0PeriodicWeightedDensityKernel(
        grid=template.bridge_asset.kernel.grid,
        kernel_bohr_minus3=periodic_gaussian_weighted_density_kernel(
            template.bridge_asset.kernel.grid,
            gaussian_width_bohr=width,
        ),
    )
    bridge = replace(template.bridge_asset, kernel=kernel)
    if bridge.pure_solvent_certificate is not None:
        raise RuntimeError(
            "A Gaussian trial retained a certificate for a different frozen kernel."
        )
    return Route2V0MolecularWeightedDensityBridgeFunctional(
        hnc_functional=template.hnc_functional,
        bridge_asset=bridge,
    )


@dataclass(frozen=True)
class Route2V0MolecularGaussianSurfaceTensionPoint:
    """One same-scalar Gaussian width with inner and planar evidence."""

    gaussian_width_bohr: float
    functional: Route2V0MolecularWeightedDensityBridgeFunctional
    coexistence_continuation: Route2V0MolecularQuarticCoexistenceContinuation
    planar_symmetry: Route2V0MolecularPlanarSymmetry
    planar_state: Route2V0MolecularConstrainedPlanarInterfaceState
    construction: str = V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_POINT_CONSTRUCTION

    def __post_init__(self) -> None:
        if (
            self.construction
            != V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_POINT_CONSTRUCTION
        ):
            raise ValueError("Unsupported Route-2 Gaussian surface-tension point.")
        width = _positive(
            self.gaussian_width_bohr,
            name="Gaussian surface-tension width",
        )
        _assert_cubic_plus_quartic(self.functional)
        if (
            not isinstance(
                self.coexistence_continuation,
                Route2V0MolecularQuarticCoexistenceContinuation,
            )
            or not self.coexistence_continuation.passes
        ):
            raise ValueError(
                "Gaussian surface-tension point requires a passing inner coexistence continuation."
            )
        if not isinstance(self.planar_symmetry, Route2V0MolecularPlanarSymmetry):
            raise TypeError("Gaussian surface-tension point requires planar symmetry.")
        if (
            not isinstance(
                self.planar_state,
                Route2V0MolecularConstrainedPlanarInterfaceState,
            )
            or not self.planar_state.passes
        ):
            raise ValueError(
                "Gaussian surface-tension point requires a zero-multiplier "
                "unconstrained planar state."
            )
        if (
            self.planar_symmetry.functional is not self.functional
            or self.planar_state.functional is not self.functional
            or self.planar_state.planar_symmetry is not self.planar_symmetry
            or self.planar_state.coexistence_continuation
            is not self.coexistence_continuation
        ):
            raise ValueError(
                "Gaussian surface-tension point must retain one exact scalar, "
                "symmetry, and inner continuation."
            )
        if not _same_float(
            self.functional.bridge_asset.quartic_coefficient_hartree_bohr15,
            self.coexistence_continuation.root.quartic_coefficient_hartree_bohr15,
        ):
            raise ValueError(
                "Gaussian surface-tension point must freeze the inner coexistence "
                "root coefficient before solving the planar state."
            )
        if not _kernel_matches_gaussian(
            self.functional,
            gaussian_width_bohr=width,
        ):
            raise ValueError(
                "Gaussian surface-tension point kernel does not match its declared width."
            )
        object.__setattr__(self, "gaussian_width_bohr", width)

    @property
    def surface_tension_hartree_per_bohr2(self) -> float:
        """Return the same-scalar bulk-subtracted planar surface excess."""

        return self.planar_state.surface_tension_hartree_per_bohr2

    @property
    def physical_liquid_admitted(self) -> bool:
        """A nested point is evidence, never a physical-liquid admission."""

        return False


def _assert_shared_outer_context(
    reference: Route2V0MolecularGaussianSurfaceTensionPoint,
    candidate: Route2V0MolecularGaussianSurfaceTensionPoint,
    *,
    label: str,
) -> None:
    """Reject an outer comparison that changes any frozen scalar convention."""

    ref_functional = reference.functional
    ref_asset = ref_functional.bridge_asset
    functional = candidate.functional
    asset = functional.bridge_asset
    if (
        functional.hnc_functional is not ref_functional.hnc_functional
        or asset.center_projection is not ref_asset.center_projection
        or candidate.planar_symmetry.cartesian_euler_quadrature
        is not reference.planar_symmetry.cartesian_euler_quadrature
        or candidate.planar_symmetry.normal_axis
        != reference.planar_symmetry.normal_axis
        or candidate.planar_state.interface_count
        != reference.planar_state.interface_count
        or not _same_float(
            candidate.planar_symmetry.transverse_area_bohr2,
            reference.planar_symmetry.transverse_area_bohr2,
        )
    ):
        raise ValueError(
            "Gaussian surface-tension bracket must retain one HNC scalar, exact "
            f"planar convention, and interface count; {label} differs."
        )
    for field_name in (
        "hnc_bulk_pressure_hartree_per_bohr3",
        "target_bulk_pressure_hartree_per_bohr3",
        "cubic_coefficient_hartree_bohr6",
        "target_surface_tension_hartree_per_bohr2",
    ):
        if not _same_float(
            getattr(asset, field_name),
            getattr(ref_asset, field_name),
        ):
            raise ValueError(
                "Gaussian surface-tension bracket must retain frozen pure-liquid "
                f"anchors; {label}.{field_name} differs."
            )
    if (
        asset.pure_solvent_certificate_sha256
        != ref_asset.pure_solvent_certificate_sha256
    ):
        raise ValueError(
            "Gaussian surface-tension bracket must retain one pure-solvent "
            f"certificate digest; {label} differs."
        )


@dataclass(frozen=True)
class Route2V0MolecularGaussianSurfaceTensionBracket:
    """A precomputed local outer bracket without a monotonicity claim."""

    lower: Route2V0MolecularGaussianSurfaceTensionPoint
    root: Route2V0MolecularGaussianSurfaceTensionPoint
    upper: Route2V0MolecularGaussianSurfaceTensionPoint
    surface_tension_tolerance_hartree_per_bohr2: float
    branch_selection: str = V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_BRANCH_SELECTION
    construction: str = V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_BRACKET_CONSTRUCTION

    def __post_init__(self) -> None:
        if (
            self.construction
            != V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_BRACKET_CONSTRUCTION
        ):
            raise ValueError("Unsupported Route-2 Gaussian surface-tension bracket.")
        if (
            self.branch_selection
            != V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_BRANCH_SELECTION
        ):
            raise ValueError("Unsupported Gaussian surface-tension branch selection.")
        points = (self.lower, self.root, self.upper)
        if not all(
            isinstance(point, Route2V0MolecularGaussianSurfaceTensionPoint)
            for point in points
        ):
            raise TypeError(
                "Gaussian surface-tension bracket requires three nested planar points."
            )
        lower_width, root_width, upper_width = tuple(
            point.gaussian_width_bohr for point in points
        )
        if not lower_width <= root_width <= upper_width or lower_width == upper_width:
            raise ValueError(
                "Gaussian surface-tension root must remain inside a nonzero width bracket."
            )
        tolerance = _positive(
            self.surface_tension_tolerance_hartree_per_bohr2,
            name="Gaussian surface-tension tolerance",
        )
        for label, point in zip(("lower", "root", "upper"), points, strict=True):
            _assert_shared_outer_context(self.root, point, label=label)

        lower_kernel = self.lower.functional.bridge_asset.kernel.kernel_bohr_minus3
        root_kernel = self.root.functional.bridge_asset.kernel.kernel_bohr_minus3
        upper_kernel = self.upper.functional.bridge_asset.kernel.kernel_bohr_minus3
        if np.array_equal(lower_kernel, upper_kernel):
            raise ValueError(
                "Gaussian surface-tension width endpoints do not resolve distinct "
                "discrete kernels on this grid."
            )
        if lower_width < root_width < upper_width and (
            np.array_equal(root_kernel, lower_kernel)
            or np.array_equal(root_kernel, upper_kernel)
        ):
            raise ValueError(
                "Gaussian surface-tension root width is not resolved from both "
                "endpoint kernels on this grid."
            )

        target = (
            self.root.functional.bridge_asset.target_surface_tension_hartree_per_bohr2
        )
        lower_gamma = self.lower.surface_tension_hartree_per_bohr2
        root_gamma = self.root.surface_tension_hartree_per_bohr2
        upper_gamma = self.upper.surface_tension_hartree_per_bohr2
        if (
            not min(lower_gamma, upper_gamma)
            <= target
            <= max(
                lower_gamma,
                upper_gamma,
            )
        ):
            raise ValueError(
                "Gaussian surface-tension endpoints do not bracket the frozen "
                "pure-liquid target."
            )
        if abs(root_gamma - target) > tolerance:
            raise ValueError(
                "Gaussian surface-tension root misses the frozen pure-liquid target."
            )
        object.__setattr__(
            self,
            "surface_tension_tolerance_hartree_per_bohr2",
            tolerance,
        )

    @property
    def target_surface_tension_hartree_per_bohr2(self) -> float:
        """Return the target carried by the frozen bridge asset."""

        return (
            self.root.functional.bridge_asset.target_surface_tension_hartree_per_bohr2
        )

    @property
    def passes(self) -> bool:
        """Return structural bracket validity, never physical admission."""

        return True

    @property
    def physical_liquid_admitted(self) -> bool:
        """The v1 bracket is evidence, not a physical-liquid certificate."""

        return False


def evaluate_route2_v0_molecular_gaussian_surface_tension_point(
    template: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    cartesian_euler_quadrature: Route2V0CartesianEulerProductQuadrature,
    gaussian_width_bohr: float,
    lower_quartic_coefficient_hartree_bohr15: float,
    upper_quartic_coefficient_hartree_bohr15: float,
    planar_liquid_fraction: float = 0.5,
    coexistence_tolerance_hartree_per_bohr3: float = 1.0e-12,
    coefficient_relative_tolerance: float = 1.0e-12,
    maximum_coexistence_iterations: int = 128,
    minimum_density_scale: float = 1.0e-10,
    gas_search_upper_density_scale: float = 0.999,
    root_sample_count: int = 513,
    directional_derivative_tolerance_hartree_per_bohr3: float = 1.0e-13,
    zero_external_tolerance_hartree: float = 1.0e-12,
    homogeneous_stationarity_tolerance: float = 1.0e-10,
    homogeneous_gradient_uniformity_tolerance: float = 1.0e-10,
    curvature_tolerance_hartree_per_bohr3: float = 1.0e-12,
    maximum_phase_bisection_iterations: int = 256,
    planar_stationarity_tolerance: float = 1.0e-10,
    planar_transverse_uniformity_tolerance: float = 1.0e-10,
    planar_constraint_multiplier_tolerance: float = 1.0e-10,
    planar_picard_mixing: float = 0.2,
    maximum_planar_iterations: int = 1000,
    normal_axis: int = 2,
) -> Route2V0MolecularGaussianSurfaceTensionPoint:
    """Compute one nested planar point without observing a gamma target.

    The quartic endpoints are the predeclared **inner** coexistence bracket.
    This function does not accept a surface-tension target, solute, experimental
    solvation value, or cavity error.  A separate outer evidence record may
    compare completed points to the frozen pure-liquid anchor.
    """

    if not isinstance(
        cartesian_euler_quadrature,
        Route2V0CartesianEulerProductQuadrature,
    ):
        raise TypeError(
            "Gaussian surface-tension point requires a Cartesian-Euler product quadrature."
        )
    width = _positive(gaussian_width_bohr, name="Gaussian surface-tension width")
    liquid_fraction = _positive(
        planar_liquid_fraction,
        name="Planar liquid fraction",
    )
    if liquid_fraction >= 1.0:
        raise ValueError(
            "Planar liquid fraction must lie strictly between zero and one."
        )
    trial = _gaussian_trial_scalar(template, gaussian_width_bohr=width)
    continuation = solve_route2_v0_molecular_quartic_coexistence_continuation(
        trial,
        lower_quartic_coefficient_hartree_bohr15=(
            lower_quartic_coefficient_hartree_bohr15
        ),
        upper_quartic_coefficient_hartree_bohr15=(
            upper_quartic_coefficient_hartree_bohr15
        ),
        coexistence_tolerance_hartree_per_bohr3=(
            coexistence_tolerance_hartree_per_bohr3
        ),
        coefficient_relative_tolerance=coefficient_relative_tolerance,
        maximum_iterations=maximum_coexistence_iterations,
        minimum_density_scale=minimum_density_scale,
        gas_search_upper_density_scale=gas_search_upper_density_scale,
        root_sample_count=root_sample_count,
        directional_derivative_tolerance_hartree_per_bohr3=(
            directional_derivative_tolerance_hartree_per_bohr3
        ),
        zero_external_tolerance_hartree=zero_external_tolerance_hartree,
        stationarity_tolerance=homogeneous_stationarity_tolerance,
        gradient_uniformity_tolerance=homogeneous_gradient_uniformity_tolerance,
        curvature_tolerance_hartree_per_bohr3=(curvature_tolerance_hartree_per_bohr3),
        maximum_phase_bisection_iterations=maximum_phase_bisection_iterations,
    )
    root_bridge = replace(
        trial.bridge_asset,
        quartic_coefficient_hartree_bohr15=(
            continuation.root.quartic_coefficient_hartree_bohr15
        ),
    )
    if root_bridge.pure_solvent_certificate is not None:
        raise RuntimeError(
            "An inner-coexistence root retained a certificate for a trial coefficient."
        )
    root_functional = Route2V0MolecularWeightedDensityBridgeFunctional(
        hnc_functional=trial.hnc_functional,
        bridge_asset=root_bridge,
    )
    symmetry = Route2V0MolecularPlanarSymmetry(
        root_functional,
        cartesian_euler_quadrature,
        normal_axis=normal_axis,
    )
    target_density_scale = continuation.root.gas_phase.density_scale + (
        liquid_fraction
        * (
            continuation.root.liquid_phase.density_scale
            - continuation.root.gas_phase.density_scale
        )
    )
    planar_state = solve_route2_v0_molecular_constrained_planar_interface(
        root_functional,
        planar_symmetry=symmetry,
        coexistence_continuation=continuation,
        target_mean_density_scale=target_density_scale,
        stationarity_tolerance=planar_stationarity_tolerance,
        transverse_uniformity_tolerance=planar_transverse_uniformity_tolerance,
        constraint_multiplier_tolerance=planar_constraint_multiplier_tolerance,
        picard_mixing=planar_picard_mixing,
        maximum_iterations=maximum_planar_iterations,
    )
    return Route2V0MolecularGaussianSurfaceTensionPoint(
        gaussian_width_bohr=width,
        functional=root_functional,
        coexistence_continuation=continuation,
        planar_symmetry=symmetry,
        planar_state=planar_state,
    )


__all__ = [
    "V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_BRACKET_CONSTRUCTION",
    "V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_BRANCH_SELECTION",
    "V0_MOLECULAR_GAUSSIAN_SURFACE_TENSION_POINT_CONSTRUCTION",
    "Route2V0MolecularGaussianSurfaceTensionBracket",
    "Route2V0MolecularGaussianSurfaceTensionPoint",
    "evaluate_route2_v0_molecular_gaussian_surface_tension_point",
]
