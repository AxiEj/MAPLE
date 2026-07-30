"""Same-scalar quartic continuation for the Route-2 V0 liquid bridge.

The cubic-plus-quartic weighted-density bridge has one pure-liquid quartic
coefficient ``B``.  It must not be selected from a solute free energy.  Nor is
it valid to vary ``B`` at a fixed chemical-potential convention, match a
planar surface tension, and *then* merely check whether a gas phase happened
to exist.  The finite-density gas branch changes with ``B``.

For the no-solute scalar ``Omega_B[nu]``, let ``nu_g(B)`` be the declared
low-density homogeneous stationary branch and let ``nu_l = nu_bulk``.  The
same-scalar coexistence gap is

``C(B) = [Omega_B[nu_g(B)] - Omega_B[nu_l]] / V``.

On a stationary branch, the envelope theorem gives

``dC/dB = [S[nu_g(B)] - S[nu_l]] / V``,

where ``S[nu] = dv sum_g rho_bar_g**2 (rho_bar_g-rho_bulk)**4`` is the
coefficient of ``B`` in the bridge scalar.  The liquid term is exactly zero
and a finite homogeneous gas with ``0 < rho_g < rho_bulk`` has a strictly
positive gas term.  Thus this derivative is positive *for this coexistence
gap* on a continuous, stable gas branch.  This is deliberately narrower than
the invalid generic claim that a planar surface tension has a positive
``d gamma / dB``: the latter also contains moving bulk subtractions and has no
such universal sign.

This module uses that signed same-scalar gap only to locate a coexistence
coefficient on a declared homogeneous branch.  It does not calculate a planar
interface, select a Gaussian width, consume a solvation label, or admit a
physical solvent asset.  A future physical construction must then solve the
planar problem at coexistence and, if an independently sourced surface tension
sets a remaining liquid length scale, retain the complete nested branch and
grid evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import SupportsFloat, SupportsIndex, cast

import numpy as np

from .route2_v0_molecular_phase_coexistence import (
    Route2V0MolecularHomogeneousPhase,
    Route2V0MolecularPhaseCoexistenceGate,
    analyze_route2_v0_molecular_homogeneous_phase_coexistence,
)
from .route2_v0_molecular_weighted_density_bridge import (
    V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_FAMILY_CUBIC_PLUS_QUARTIC,
    Route2V0MolecularWeightedDensityBridgeFunctional,
)

V0_MOLECULAR_CUBIC_PLUS_QUARTIC_COEXISTENCE_CONTINUATION = (
    "route2-v0-molecular-cubic-plus-quartic-coexistence-continuation-v1"
)
V0_MOLECULAR_LOW_DENSITY_STABLE_GAS_BRANCH_SELECTION = (
    "lowest-density-stable-homogeneous-minimum-v1"
)


def _finite(value: object, *, name: str) -> float:
    """Return one finite real scalar without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite.")
    try:
        result = float(
            cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: object, *, name: str) -> float:
    """Return one finite strictly positive real scalar."""

    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one finite strictly positive integer."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    numeric = _finite(value, name=name)
    result = int(numeric)
    if result != numeric or result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _cell_volume_bohr3(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
) -> float:
    """Return the exact periodic Cartesian volume of the molecular scalar."""

    grid = functional.projection.site_hnc_asset.grid
    volume = float(grid.point_count * grid.volume_element_bohr3)
    if not math.isfinite(volume) or volume <= 0.0:
        raise RuntimeError("Molecular coexistence-continuation cell volume is invalid.")
    return volume


def _homogeneous_configuration_density(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    density_scale: float,
) -> np.ndarray:
    """Return ``nu_x = x nu_bulk`` under the functional's exact quadrature."""

    scale = _positive(density_scale, name="Homogeneous density scale")
    density = np.full(
        functional.projection.quadrature.configuration_count,
        scale * functional.projection.uniform_configuration_density_bohr3,
        dtype=float,
    )
    density.setflags(write=False)
    return density


def _is_stable(
    phase: Route2V0MolecularHomogeneousPhase,
    *,
    gate: Route2V0MolecularPhaseCoexistenceGate,
) -> bool:
    """Apply the gate's full stationary-minimum predicate without relabelling it."""

    return bool(
        phase.classification == "stable"
        and phase.full_stationarity_residual <= gate.stationarity_tolerance
        and phase.gradient_uniformity_residual <= gate.gradient_uniformity_tolerance
        and phase.curvature_density_hartree_per_bohr3
        > gate.curvature_tolerance_hartree_per_bohr3
    )


def _select_low_density_stable_gas(
    gate: Route2V0MolecularPhaseCoexistenceGate,
) -> tuple[Route2V0MolecularHomogeneousPhase, int]:
    """Select the declared gas branch without using a lowest-energy shortcut."""

    candidates = tuple(
        phase
        for phase in gate.gas_candidates
        if _is_stable(phase, gate=gate) and 0.0 < phase.density_scale < 1.0
    )
    if not candidates:
        raise RuntimeError(
            "Quartic coexistence continuation requires a stable finite-density "
            + "low-density homogeneous branch at every bracket point."
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "Quartic coexistence continuation requires exactly one stable "
            + "low-density homogeneous gas branch at every bracket point; a "
            + "multi-branch continuation must be declared separately."
        )
    return candidates[0], 1


def _continuation_functional(
    template: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    quartic_coefficient_hartree_bohr15: float,
) -> Route2V0MolecularWeightedDensityBridgeFunctional:
    """Copy one scalar while varying exactly and only its quartic coefficient.

    The temporary functional intentionally carries no source-bound certificate:
    a certificate binds one already-frozen coefficient and cannot certify trial
    points.  The returned object is therefore continuation evidence, never a
    physical-liquid-admitted bridge asset.
    """

    if (
        template.bridge_asset.bridge_family
        != V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_FAMILY_CUBIC_PLUS_QUARTIC
    ):
        raise ValueError(
            "Quartic coexistence continuation requires the cubic-plus-quartic "
            + "bridge family."
        )
    coefficient = _positive(
        quartic_coefficient_hartree_bohr15,
        name="Quartic coexistence-continuation coefficient",
    )
    asset = replace(
        template.bridge_asset,
        quartic_coefficient_hartree_bohr15=coefficient,
    )
    if asset.pure_solvent_certificate is not None:
        raise RuntimeError(
            "A temporary quartic continuation point unexpectedly retained a "
            + "source-bound certificate."
        )
    return Route2V0MolecularWeightedDensityBridgeFunctional(
        hnc_functional=template.hnc_functional,
        bridge_asset=asset,
    )


def _quartic_coefficient_functional_hartree_per_hartree_bohr15(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    configuration_density_bohr3: np.ndarray,
) -> float:
    """Return ``S[nu]`` multiplying the quartic coefficient in ``F_bridge``."""

    weighted = functional.weighted_molecular_density(configuration_density_bohr3)
    bulk = functional.bridge_asset.molecular_bulk_number_density_bohr3
    value = float(
        functional.center_projection.grid.volume_element_bohr3
        * np.sum(weighted**2 * (weighted - bulk) ** 4)
    )
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("Quartic bridge coefficient functional is invalid.")
    return value


@dataclass(frozen=True)
class Route2V0MolecularQuarticCoexistencePoint:
    """One certified homogeneous branch point along the quartic continuation.

    ``coexistence_gap`` is gas minus liquid.  The envelope derivative uses the
    exact discrete quartic functional at those two stationary states; it is
    not a finite-difference surrogate and it is not a planar-tension
    derivative.
    """

    quartic_coefficient_hartree_bohr15: float
    phase_gate: Route2V0MolecularPhaseCoexistenceGate
    liquid_phase: Route2V0MolecularHomogeneousPhase
    gas_phase: Route2V0MolecularHomogeneousPhase
    stable_gas_candidate_count: int
    coexistence_gap_hartree_per_bohr3: float
    liquid_quartic_functional_hartree_per_hartree_bohr15: float
    gas_quartic_functional_hartree_per_hartree_bohr15: float
    coexistence_gap_envelope_derivative_bohr_minus18: float
    construction: str = V0_MOLECULAR_CUBIC_PLUS_QUARTIC_COEXISTENCE_CONTINUATION

    def __post_init__(self) -> None:
        if (
            self.construction
            != V0_MOLECULAR_CUBIC_PLUS_QUARTIC_COEXISTENCE_CONTINUATION
        ):
            raise ValueError("Unsupported Route-2 quartic coexistence continuation.")
        coefficient = _positive(
            self.quartic_coefficient_hartree_bohr15,
            name="Quartic coexistence-continuation coefficient",
        )
        count = _positive_integer(
            self.stable_gas_candidate_count,
            name="Stable homogeneous gas-candidate count",
        )
        if count != 1:
            raise ValueError(
                "Quartic continuation requires exactly one stable low-density "
                + "homogeneous gas branch."
            )
        if not _is_stable(self.liquid_phase, gate=self.phase_gate):
            raise ValueError(
                "Quartic continuation liquid phase is not a stable minimum."
            )
        if not _is_stable(self.gas_phase, gate=self.phase_gate):
            raise ValueError("Quartic continuation gas phase is not a stable minimum.")
        if not 0.0 < self.gas_phase.density_scale < 1.0:
            raise ValueError("Quartic continuation gas phase must have 0 < x_gas < 1.")
        expected_gap = (
            self.gas_phase.grand_potential_density_hartree_per_bohr3
            - self.liquid_phase.grand_potential_density_hartree_per_bohr3
        )
        gap = _finite(self.coexistence_gap_hartree_per_bohr3, name="Coexistence gap")
        gap_tolerance = 1.0e-12 * max(1.0, abs(expected_gap), abs(gap))
        if abs(gap - expected_gap) > gap_tolerance:
            raise ValueError(
                "Quartic continuation coexistence gap must equal the gas-minus-"
                + "liquid same-scalar grand-potential density."
            )
        liquid = _finite(
            self.liquid_quartic_functional_hartree_per_hartree_bohr15,
            name="Liquid quartic coefficient functional",
        )
        gas = _finite(
            self.gas_quartic_functional_hartree_per_hartree_bohr15,
            name="Gas quartic coefficient functional",
        )
        if liquid < 0.0 or gas <= 0.0:
            raise ValueError(
                "Quartic continuation requires a zero-or-positive liquid and "
                + "strictly positive finite-gas coefficient functional."
            )
        derivative = _positive(
            self.coexistence_gap_envelope_derivative_bohr_minus18,
            name="Coexistence-gap envelope derivative",
        )
        object.__setattr__(self, "quartic_coefficient_hartree_bohr15", coefficient)
        object.__setattr__(self, "stable_gas_candidate_count", count)
        object.__setattr__(self, "coexistence_gap_hartree_per_bohr3", gap)
        object.__setattr__(
            self,
            "liquid_quartic_functional_hartree_per_hartree_bohr15",
            liquid,
        )
        object.__setattr__(
            self,
            "gas_quartic_functional_hartree_per_hartree_bohr15",
            gas,
        )
        object.__setattr__(
            self,
            "coexistence_gap_envelope_derivative_bohr_minus18",
            derivative,
        )


@dataclass(frozen=True)
class Route2V0MolecularQuarticCoexistenceContinuation:
    """A bisection certificate for one finite-density homogeneous gas branch."""

    lower: Route2V0MolecularQuarticCoexistencePoint
    root: Route2V0MolecularQuarticCoexistencePoint
    upper: Route2V0MolecularQuarticCoexistencePoint
    coexistence_tolerance_hartree_per_bohr3: float
    coefficient_relative_tolerance: float
    iterations: int
    branch_selection: str = V0_MOLECULAR_LOW_DENSITY_STABLE_GAS_BRANCH_SELECTION
    construction: str = V0_MOLECULAR_CUBIC_PLUS_QUARTIC_COEXISTENCE_CONTINUATION

    def __post_init__(self) -> None:
        if (
            self.construction
            != V0_MOLECULAR_CUBIC_PLUS_QUARTIC_COEXISTENCE_CONTINUATION
        ):
            raise ValueError("Unsupported Route-2 quartic coexistence continuation.")
        if (
            self.branch_selection
            != V0_MOLECULAR_LOW_DENSITY_STABLE_GAS_BRANCH_SELECTION
        ):
            raise ValueError("Unsupported homogeneous gas-branch selection rule.")
        points = (self.lower, self.root, self.upper)
        lower_b, root_b, upper_b = tuple(
            point.quartic_coefficient_hartree_bohr15 for point in points
        )
        if not lower_b <= root_b <= upper_b or lower_b == upper_b:
            raise ValueError("Quartic coexistence root must remain inside its bracket.")
        tolerance = _positive(
            self.coexistence_tolerance_hartree_per_bohr3,
            name="Quartic coexistence tolerance",
        )
        relative = _positive(
            self.coefficient_relative_tolerance,
            name="Quartic coefficient relative tolerance",
        )
        if relative >= 1.0:
            raise ValueError(
                "Quartic coefficient relative tolerance must be below one."
            )
        iterations = _positive_integer(
            self.iterations,
            name="Quartic coexistence continuation iterations",
        )
        if self.lower.coexistence_gap_hartree_per_bohr3 >= -tolerance:
            raise ValueError(
                "Quartic coexistence lower endpoint must have a negative gas-minus-"
                + "liquid gap."
            )
        if self.upper.coexistence_gap_hartree_per_bohr3 <= tolerance:
            raise ValueError(
                "Quartic coexistence upper endpoint must have a positive gas-minus-"
                + "liquid gap."
            )
        if abs(self.root.coexistence_gap_hartree_per_bohr3) > tolerance:
            raise ValueError(
                "Quartic coexistence root misses its same-scalar gap tolerance."
            )
        object.__setattr__(self, "coexistence_tolerance_hartree_per_bohr3", tolerance)
        object.__setattr__(self, "coefficient_relative_tolerance", relative)
        object.__setattr__(self, "iterations", iterations)

    @property
    def passes(self) -> bool:
        """Return whether this record establishes the declared homogeneous root."""

        return True


def evaluate_route2_v0_molecular_quartic_coexistence_point(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    quartic_coefficient_hartree_bohr15: float,
    minimum_density_scale: float = 1.0e-10,
    gas_search_upper_density_scale: float = 0.999,
    root_sample_count: int = 513,
    directional_derivative_tolerance_hartree_per_bohr3: float = 1.0e-13,
    zero_external_tolerance_hartree: float = 1.0e-12,
    stationarity_tolerance: float = 1.0e-10,
    gradient_uniformity_tolerance: float = 1.0e-10,
    curvature_tolerance_hartree_per_bohr3: float = 1.0e-12,
    coexistence_tolerance_hartree_per_bohr3: float = 1.0e-12,
    maximum_phase_bisection_iterations: int = 256,
) -> Route2V0MolecularQuarticCoexistencePoint:
    """Evaluate one same-scalar homogeneous point on the declared gas branch."""

    trial = _continuation_functional(
        functional,
        quartic_coefficient_hartree_bohr15=quartic_coefficient_hartree_bohr15,
    )
    gate = analyze_route2_v0_molecular_homogeneous_phase_coexistence(
        trial,
        minimum_density_scale=minimum_density_scale,
        gas_search_upper_density_scale=gas_search_upper_density_scale,
        root_sample_count=root_sample_count,
        directional_derivative_tolerance_hartree_per_bohr3=(
            directional_derivative_tolerance_hartree_per_bohr3
        ),
        zero_external_tolerance_hartree=zero_external_tolerance_hartree,
        stationarity_tolerance=stationarity_tolerance,
        gradient_uniformity_tolerance=gradient_uniformity_tolerance,
        curvature_tolerance_hartree_per_bohr3=(curvature_tolerance_hartree_per_bohr3),
        coexistence_tolerance_hartree_per_bohr3=(
            coexistence_tolerance_hartree_per_bohr3
        ),
        maximum_bisection_iterations=maximum_phase_bisection_iterations,
    )
    liquid = gate.liquid_phase
    gas, stable_gas_count = _select_low_density_stable_gas(gate)
    liquid_density = _homogeneous_configuration_density(
        trial,
        liquid.density_scale,
    )
    gas_density = _homogeneous_configuration_density(trial, gas.density_scale)
    liquid_functional = _quartic_coefficient_functional_hartree_per_hartree_bohr15(
        trial,
        liquid_density,
    )
    gas_functional = _quartic_coefficient_functional_hartree_per_hartree_bohr15(
        trial,
        gas_density,
    )
    volume = _cell_volume_bohr3(trial)
    return Route2V0MolecularQuarticCoexistencePoint(
        quartic_coefficient_hartree_bohr15=(quartic_coefficient_hartree_bohr15),
        phase_gate=gate,
        liquid_phase=liquid,
        gas_phase=gas,
        stable_gas_candidate_count=stable_gas_count,
        coexistence_gap_hartree_per_bohr3=(
            gas.grand_potential_density_hartree_per_bohr3
            - liquid.grand_potential_density_hartree_per_bohr3
        ),
        liquid_quartic_functional_hartree_per_hartree_bohr15=(liquid_functional),
        gas_quartic_functional_hartree_per_hartree_bohr15=gas_functional,
        coexistence_gap_envelope_derivative_bohr_minus18=(
            (gas_functional - liquid_functional) / volume
        ),
    )


def solve_route2_v0_molecular_quartic_coexistence_continuation(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    lower_quartic_coefficient_hartree_bohr15: float,
    upper_quartic_coefficient_hartree_bohr15: float,
    coexistence_tolerance_hartree_per_bohr3: float = 1.0e-12,
    coefficient_relative_tolerance: float = 1.0e-12,
    maximum_iterations: int = 128,
    minimum_density_scale: float = 1.0e-10,
    gas_search_upper_density_scale: float = 0.999,
    root_sample_count: int = 513,
    directional_derivative_tolerance_hartree_per_bohr3: float = 1.0e-13,
    zero_external_tolerance_hartree: float = 1.0e-12,
    stationarity_tolerance: float = 1.0e-10,
    gradient_uniformity_tolerance: float = 1.0e-10,
    curvature_tolerance_hartree_per_bohr3: float = 1.0e-12,
    maximum_phase_bisection_iterations: int = 256,
) -> Route2V0MolecularQuarticCoexistenceContinuation:
    """Locate ``B`` from a finite-density same-scalar coexistence condition.

    The search is a pure-liquid mathematical construction.  It never reads a
    surface tension or a solvation result.  The two input endpoints must retain
    the same low-density stable branch and bracket a negative-to-positive
    gas-minus-liquid grand-potential-density gap.  Every trial recomputes the
    exact full-gradient phase gate; a projected density root alone cannot be
    used to continue the branch.
    """

    lower_coefficient = _positive(
        lower_quartic_coefficient_hartree_bohr15,
        name="Lower quartic coexistence coefficient",
    )
    upper_coefficient = _positive(
        upper_quartic_coefficient_hartree_bohr15,
        name="Upper quartic coexistence coefficient",
    )
    if lower_coefficient >= upper_coefficient:
        raise ValueError(
            "Quartic coexistence bracket requires lower coefficient < upper."
        )
    tolerance = _positive(
        coexistence_tolerance_hartree_per_bohr3,
        name="Quartic coexistence tolerance",
    )
    relative_tolerance = _positive(
        coefficient_relative_tolerance,
        name="Quartic coefficient relative tolerance",
    )
    if relative_tolerance >= 1.0:
        raise ValueError("Quartic coefficient relative tolerance must be below one.")
    iterations = _positive_integer(
        maximum_iterations,
        name="Quartic coexistence continuation iterations",
    )
    sample_count = _positive_integer(
        root_sample_count,
        name="Homogeneous root samples",
    )
    if sample_count < 3:
        raise ValueError("Homogeneous root samples must be at least three.")
    phase_iterations = _positive_integer(
        maximum_phase_bisection_iterations,
        name="Homogeneous root bisection iterations",
    )

    def evaluate_point(
        coefficient: float,
    ) -> Route2V0MolecularQuarticCoexistencePoint:
        return evaluate_route2_v0_molecular_quartic_coexistence_point(
            functional,
            quartic_coefficient_hartree_bohr15=coefficient,
            minimum_density_scale=minimum_density_scale,
            gas_search_upper_density_scale=gas_search_upper_density_scale,
            root_sample_count=sample_count,
            directional_derivative_tolerance_hartree_per_bohr3=(
                directional_derivative_tolerance_hartree_per_bohr3
            ),
            zero_external_tolerance_hartree=zero_external_tolerance_hartree,
            stationarity_tolerance=stationarity_tolerance,
            gradient_uniformity_tolerance=gradient_uniformity_tolerance,
            curvature_tolerance_hartree_per_bohr3=(
                curvature_tolerance_hartree_per_bohr3
            ),
            coexistence_tolerance_hartree_per_bohr3=tolerance,
            maximum_phase_bisection_iterations=phase_iterations,
        )

    lower = evaluate_point(lower_coefficient)
    upper = evaluate_point(upper_coefficient)
    if lower.coexistence_gap_hartree_per_bohr3 >= -tolerance:
        raise ValueError(
            "Quartic coexistence lower endpoint does not have a negative "
            + "gas-minus-liquid gap."
        )
    if upper.coexistence_gap_hartree_per_bohr3 <= tolerance:
        raise ValueError(
            "Quartic coexistence upper endpoint does not have a positive "
            + "gas-minus-liquid gap."
        )

    root: Route2V0MolecularQuarticCoexistencePoint | None = None
    iteration = 0
    for iteration in range(1, iterations + 1):
        midpoint = math.sqrt(
            lower.quartic_coefficient_hartree_bohr15
            * upper.quartic_coefficient_hartree_bohr15
        )
        point = evaluate_point(midpoint)
        if abs(point.coexistence_gap_hartree_per_bohr3) <= tolerance:
            root = point
            break
        if point.coexistence_gap_hartree_per_bohr3 < 0.0:
            lower = point
        else:
            upper = point
        relative_width = (
            upper.quartic_coefficient_hartree_bohr15
            - lower.quartic_coefficient_hartree_bohr15
        ) / max(
            lower.quartic_coefficient_hartree_bohr15,
            upper.quartic_coefficient_hartree_bohr15,
        )
        if relative_width <= relative_tolerance:
            midpoint = math.sqrt(
                lower.quartic_coefficient_hartree_bohr15
                * upper.quartic_coefficient_hartree_bohr15
            )
            root = evaluate_point(midpoint)
            break
    if root is None:
        raise RuntimeError(
            "Quartic coexistence continuation did not reach its coefficient or "
            + "same-scalar gap tolerance."
        )
    if abs(root.coexistence_gap_hartree_per_bohr3) > tolerance:
        raise RuntimeError(
            "Quartic coexistence continuation reached its coefficient tolerance "
            + "without satisfying the same-scalar coexistence gap."
        )
    return Route2V0MolecularQuarticCoexistenceContinuation(
        lower=lower,
        root=root,
        upper=upper,
        coexistence_tolerance_hartree_per_bohr3=tolerance,
        coefficient_relative_tolerance=relative_tolerance,
        iterations=iteration,
    )


__all__ = [
    "V0_MOLECULAR_CUBIC_PLUS_QUARTIC_COEXISTENCE_CONTINUATION",
    "V0_MOLECULAR_LOW_DENSITY_STABLE_GAS_BRANCH_SELECTION",
    "Route2V0MolecularQuarticCoexistenceContinuation",
    "Route2V0MolecularQuarticCoexistencePoint",
    "evaluate_route2_v0_molecular_quartic_coexistence_point",
    "solve_route2_v0_molecular_quartic_coexistence_continuation",
]
