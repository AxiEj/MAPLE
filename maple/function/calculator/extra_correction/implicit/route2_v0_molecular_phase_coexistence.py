"""Fail-closed homogeneous-phase gate for the Route-2 V0 liquid scalar.

The molecular HNC--WDA bridge is intended to repair the HNC pressure defect
*inside* one grand-potential scalar.  A pressure identity at the empty limit
is necessary, but it does not by itself prove that the scalar has a stable
low-density phase at the same grand-potential density as the bulk liquid.
That latter condition is a prerequisite for interpreting a planar stationary
profile as a liquid--gas interface and for using it to determine a surface
tension or a WDA range.

For the exact discrete scalar ``Omega[nu]``, this module evaluates the
homogeneous ray

``nu_x(Gamma) = x * nu_bulk(Gamma)``

without replacing its configuration-space stationarity by a scalar density
proxy.  It reports

``omega'(x) = kBT / V * sum_i w_i nu_bulk g_i(x)``

and

``omega''(x) = kBT / V * sum_i w_i nu_bulk (H_x nu_bulk)_i``.

Here ``g`` and ``H`` are the gradient and Hessian action of the *same*
molecular HNC-plus-bridge functional.  A root of the first expression is
admissible as a homogeneous phase only when the full configuration gradient
is also zero: a non-uniform quadrature can otherwise hide a nonstationary
state behind a one-dimensional projected root.

This is a diagnostic and an admission gate only.  It never changes a bridge
coefficient, chooses a Gaussian width, inserts a pressure correction, fits a
solute label, or creates a physical-liquid or solvation-accuracy claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, SupportsFloat, SupportsIndex, cast

import numpy as np

from .route2_v0_molecular_external_potential_contract import (
    require_molecular_external_potential_values,
)
from .route2_v0_molecular_weighted_density_bridge import (
    Route2V0MolecularWeightedDensityBridgeFunctional,
)
from .route2_v0_pure_solvent_bridge_certificate import (
    V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION,
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
    """Return one finite positive real scalar."""

    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one finite positive integer without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    numeric = _finite(value, name=name)
    result = int(numeric)
    if result != numeric or result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _immutable_vector(
    values: np.ndarray,
    *,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    """Copy one finite one-dimensional vector into immutable storage."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
    ):
        requirement = "a nonempty finite one-dimensional vector"
        if positive:
            requirement += " with positive values"
        raise ValueError(f"{name} must be {requirement}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _cell_volume_bohr3(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
) -> float:
    """Return the Cartesian cell volume represented by the exact scalar."""

    grid = functional.projection.site_hnc_asset.grid
    volume = float(grid.point_count * grid.volume_element_bohr3)
    if not math.isfinite(volume) or volume <= 0.0:
        raise RuntimeError("Molecular homogeneous phase cell volume is invalid.")
    return volume


def _bulk_configuration_density(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
) -> np.ndarray:
    """Return the exact uniform bulk configuration density of one scalar."""

    projection = functional.projection
    density = np.full(
        projection.quadrature.configuration_count,
        projection.uniform_configuration_density_bohr3,
        dtype=float,
    )
    return _immutable_vector(
        density,
        name="Molecular homogeneous bulk configuration density",
        positive=True,
    )


def _pure_liquid_external_residual_hartree(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
) -> float:
    """Return the exact external-potential norm, refusing a solute functional."""

    external = require_molecular_external_potential_values(
        functional.projection.external_potential,
        configuration_count=functional.projection.quadrature.configuration_count,
    )
    residual = float(np.max(np.abs(external)))
    if not math.isfinite(residual):
        raise RuntimeError("Pure-liquid external-potential residual is non-finite.")
    return residual


@dataclass(frozen=True)
class Route2V0MolecularHomogeneousPhase:
    """One homogeneous-ray point evaluated by the full molecular scalar.

    ``directional_derivative`` is useful to bracket stationary points on the
    homogeneous ray.  It is deliberately not used as a substitute for
    ``full_stationarity_residual``: the latter is the condition required for a
    true stationary configuration-density state.
    """

    density_scale: float
    grand_potential_hartree: float
    grand_potential_density_hartree_per_bohr3: float
    directional_derivative_hartree: float
    directional_derivative_density_hartree_per_bohr3: float
    curvature_hartree: float
    curvature_density_hartree_per_bohr3: float
    full_stationarity_residual: float
    gradient_uniformity_residual: float
    classification: Literal["stable", "unstable", "numerically-singular"]
    construction: str = V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION

    def __post_init__(self) -> None:
        """Validate a self-contained, finite phase diagnostic."""

        if self.construction != V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular homogeneous phase.")
        scale = _positive(self.density_scale, name="Homogeneous density scale")
        values = {
            "grand_potential_hartree": self.grand_potential_hartree,
            "grand_potential_density_hartree_per_bohr3": (
                self.grand_potential_density_hartree_per_bohr3
            ),
            "directional_derivative_hartree": self.directional_derivative_hartree,
            "directional_derivative_density_hartree_per_bohr3": (
                self.directional_derivative_density_hartree_per_bohr3
            ),
            "curvature_hartree": self.curvature_hartree,
            "curvature_density_hartree_per_bohr3": (
                self.curvature_density_hartree_per_bohr3
            ),
            "full_stationarity_residual": self.full_stationarity_residual,
            "gradient_uniformity_residual": self.gradient_uniformity_residual,
        }
        for name, value in values.items():
            checked = _finite(value, name=f"Homogeneous phase {name}")
            if name.endswith("residual") and checked < 0.0:
                raise ValueError(f"Homogeneous phase {name} must be nonnegative.")
            object.__setattr__(self, name, checked)
        if self.classification not in {
            "stable",
            "unstable",
            "numerically-singular",
        }:
            raise ValueError("Unsupported homogeneous phase classification.")
        object.__setattr__(self, "density_scale", scale)


@dataclass(frozen=True)
class Route2V0MolecularPhaseCoexistenceGate:
    """Fail-closed coexistence evidence before a planar-interface calculation.

    ``passes`` means only that the declared finite molecular scalar has two
    homogeneous stationary minima with matching grand-potential densities in
    its exact quadrature.  It is still not a surface-tension result, a
    source-bound physical-liquid asset, or a solvation result.
    """

    liquid_phase: Route2V0MolecularHomogeneousPhase
    gas_phase: Route2V0MolecularHomogeneousPhase | None
    gas_candidates: tuple[Route2V0MolecularHomogeneousPhase, ...]
    external_potential_residual_hartree: float
    zero_external_tolerance_hartree: float
    stationarity_tolerance: float
    gradient_uniformity_tolerance: float
    curvature_tolerance_hartree_per_bohr3: float
    coexistence_tolerance_hartree_per_bohr3: float
    grand_potential_density_difference_hartree_per_bohr3: float | None
    passes: bool
    construction: str = V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION

    def __post_init__(self) -> None:
        """Reject inconsistent gate labels rather than repairing them."""

        if self.construction != V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular phase-coexistence gate.")
        if not isinstance(self.liquid_phase, Route2V0MolecularHomogeneousPhase):
            raise TypeError("Phase-coexistence gate requires a liquid phase.")
        if self.gas_phase is not None and not isinstance(
            self.gas_phase, Route2V0MolecularHomogeneousPhase
        ):
            raise TypeError("Phase-coexistence gate gas phase has an invalid type.")
        candidates = tuple(self.gas_candidates)
        if any(
            not isinstance(candidate, Route2V0MolecularHomogeneousPhase)
            for candidate in candidates
        ):
            raise TypeError("Phase-coexistence gate candidates have an invalid type.")
        if self.gas_phase is not None and self.gas_phase not in candidates:
            raise ValueError("Selected gas phase must occur in the candidate list.")
        external = _finite(
            self.external_potential_residual_hartree,
            name="Pure-liquid external-potential residual",
        )
        if external < 0.0:
            raise ValueError(
                "Pure-liquid external-potential residual must be nonnegative."
            )
        zero_external = _positive(
            self.zero_external_tolerance_hartree,
            name="Pure-liquid zero-external-potential tolerance",
        )
        stationarity = _positive(
            self.stationarity_tolerance,
            name="Homogeneous stationarity tolerance",
        )
        uniformity = _positive(
            self.gradient_uniformity_tolerance,
            name="Homogeneous gradient-uniformity tolerance",
        )
        curvature = _positive(
            self.curvature_tolerance_hartree_per_bohr3,
            name="Homogeneous curvature tolerance",
        )
        coexistence = _positive(
            self.coexistence_tolerance_hartree_per_bohr3,
            name="Homogeneous coexistence tolerance",
        )
        difference = self.grand_potential_density_difference_hartree_per_bohr3
        if self.gas_phase is None:
            if difference is not None:
                raise ValueError("A missing gas phase cannot carry a coexistence gap.")
        else:
            if difference is None:
                raise ValueError("A selected gas phase requires a coexistence gap.")
            expected = (
                self.gas_phase.grand_potential_density_hartree_per_bohr3
                - self.liquid_phase.grand_potential_density_hartree_per_bohr3
            )
            actual = _finite(difference, name="Homogeneous coexistence gap")
            tolerance = 1.0e-12 * max(1.0, abs(expected), abs(actual))
            if abs(actual - expected) > tolerance:
                raise ValueError(
                    "Homogeneous coexistence gap does not equal the two phase "
                    "grand-potential densities."
                )
            object.__setattr__(
                self,
                "grand_potential_density_difference_hartree_per_bohr3",
                actual,
            )
        expected_pass = (
            external <= zero_external
            and _is_stationary_stable(
                self.liquid_phase,
                stationarity_tolerance=stationarity,
                gradient_uniformity_tolerance=uniformity,
                curvature_tolerance_hartree_per_bohr3=curvature,
            )
            and self.gas_phase is not None
            and _is_stationary_stable(
                self.gas_phase,
                stationarity_tolerance=stationarity,
                gradient_uniformity_tolerance=uniformity,
                curvature_tolerance_hartree_per_bohr3=curvature,
            )
            and abs(
                self.grand_potential_density_difference_hartree_per_bohr3
                if self.grand_potential_density_difference_hartree_per_bohr3 is not None
                else math.inf
            )
            <= coexistence
        )
        if self.passes is not expected_pass:
            raise ValueError(
                "Phase-coexistence pass label disagrees with its evidence."
            )
        object.__setattr__(self, "gas_candidates", candidates)
        object.__setattr__(self, "external_potential_residual_hartree", external)
        object.__setattr__(self, "zero_external_tolerance_hartree", zero_external)
        object.__setattr__(self, "stationarity_tolerance", stationarity)
        object.__setattr__(self, "gradient_uniformity_tolerance", uniformity)
        object.__setattr__(self, "curvature_tolerance_hartree_per_bohr3", curvature)
        object.__setattr__(self, "coexistence_tolerance_hartree_per_bohr3", coexistence)


def _is_stationary_stable(
    phase: Route2V0MolecularHomogeneousPhase,
    *,
    stationarity_tolerance: float,
    gradient_uniformity_tolerance: float,
    curvature_tolerance_hartree_per_bohr3: float,
) -> bool:
    """Return the unmodified local-minimum criterion for one phase point."""

    return bool(
        phase.classification == "stable"
        and phase.full_stationarity_residual <= stationarity_tolerance
        and phase.gradient_uniformity_residual <= gradient_uniformity_tolerance
        and phase.curvature_density_hartree_per_bohr3
        > curvature_tolerance_hartree_per_bohr3
    )


def evaluate_route2_v0_molecular_homogeneous_phase(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    density_scale: float,
    curvature_tolerance_hartree_per_bohr3: float = 1.0e-12,
) -> Route2V0MolecularHomogeneousPhase:
    """Evaluate one exact homogeneous-ray point of the molecular scalar.

    The caller may use a point on the ray to explore the liquid free-energy
    landscape, but no point is called a phase merely because its *projected*
    derivative happens to vanish.
    """

    if not isinstance(functional, Route2V0MolecularWeightedDensityBridgeFunctional):
        raise TypeError(
            "Molecular homogeneous phase evaluation requires an HNC-plus-bridge "
            "functional."
        )
    scale = _positive(density_scale, name="Homogeneous density scale")
    curvature_tolerance = _positive(
        curvature_tolerance_hartree_per_bohr3,
        name="Homogeneous curvature tolerance",
    )
    bulk = _bulk_configuration_density(functional)
    density = scale * bulk
    gradient = functional.dimensionless_gradient(density)
    action = functional.dimensionless_hessian_matvec(density, bulk)
    weights = functional.projection.quadrature.phase_space_weights_bohr3
    volume = _cell_volume_bohr3(functional)
    kbt = functional.kbt_hartree
    grand_potential = functional.grand_potential_hartree(density)
    directional_derivative = kbt * float(np.sum(weights * bulk * gradient))
    curvature = kbt * float(np.sum(weights * bulk * action))
    weighted_gradient_mean = float(np.sum(weights * gradient) / np.sum(weights))
    uniformity = float(np.max(np.abs(gradient - weighted_gradient_mean)))
    full_residual = float(np.max(np.abs(gradient)))
    values = (
        grand_potential,
        directional_derivative,
        curvature,
        weighted_gradient_mean,
        uniformity,
        full_residual,
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Molecular homogeneous phase evaluation is non-finite.")
    if curvature > curvature_tolerance:
        classification: Literal["stable", "unstable", "numerically-singular"] = "stable"
    elif curvature < -curvature_tolerance:
        classification = "unstable"
    else:
        classification = "numerically-singular"
    return Route2V0MolecularHomogeneousPhase(
        density_scale=scale,
        grand_potential_hartree=grand_potential,
        grand_potential_density_hartree_per_bohr3=grand_potential / volume,
        directional_derivative_hartree=directional_derivative,
        directional_derivative_density_hartree_per_bohr3=(
            directional_derivative / volume
        ),
        curvature_hartree=curvature,
        curvature_density_hartree_per_bohr3=curvature / volume,
        full_stationarity_residual=full_residual,
        gradient_uniformity_residual=uniformity,
        classification=classification,
    )


def _bisect_directional_root(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    lower: float,
    upper: float,
    derivative_tolerance_hartree_per_bohr3: float,
    curvature_tolerance_hartree_per_bohr3: float,
    maximum_iterations: int,
) -> Route2V0MolecularHomogeneousPhase:
    """Return one sign-bracketed homogeneous-ray root without model changes."""

    lower_phase = evaluate_route2_v0_molecular_homogeneous_phase(
        functional,
        density_scale=lower,
        curvature_tolerance_hartree_per_bohr3=curvature_tolerance_hartree_per_bohr3,
    )
    upper_phase = evaluate_route2_v0_molecular_homogeneous_phase(
        functional,
        density_scale=upper,
        curvature_tolerance_hartree_per_bohr3=curvature_tolerance_hartree_per_bohr3,
    )
    lower_derivative = lower_phase.directional_derivative_density_hartree_per_bohr3
    upper_derivative = upper_phase.directional_derivative_density_hartree_per_bohr3
    if abs(lower_derivative) <= derivative_tolerance_hartree_per_bohr3:
        return lower_phase
    if abs(upper_derivative) <= derivative_tolerance_hartree_per_bohr3:
        return upper_phase
    if (lower_derivative > 0.0) == (upper_derivative > 0.0):
        raise ValueError("Homogeneous root interval does not bracket a sign change.")
    for _ in range(maximum_iterations):
        midpoint = math.sqrt(lower * upper)
        phase = evaluate_route2_v0_molecular_homogeneous_phase(
            functional,
            density_scale=midpoint,
            curvature_tolerance_hartree_per_bohr3=curvature_tolerance_hartree_per_bohr3,
        )
        derivative = phase.directional_derivative_density_hartree_per_bohr3
        if abs(derivative) <= derivative_tolerance_hartree_per_bohr3:
            return phase
        if (lower_derivative > 0.0) != (derivative > 0.0):
            upper = midpoint
            upper_derivative = derivative
        else:
            lower = midpoint
            lower_derivative = derivative
    phase = evaluate_route2_v0_molecular_homogeneous_phase(
        functional,
        density_scale=math.sqrt(lower * upper),
        curvature_tolerance_hartree_per_bohr3=curvature_tolerance_hartree_per_bohr3,
    )
    if (
        abs(phase.directional_derivative_density_hartree_per_bohr3)
        > derivative_tolerance_hartree_per_bohr3
    ):
        raise RuntimeError(
            "Homogeneous phase bisection did not reach its directional-derivative "
            "tolerance."
        )
    return phase


def _distinct_scale(
    phases: list[Route2V0MolecularHomogeneousPhase],
    candidate: Route2V0MolecularHomogeneousPhase,
) -> bool:
    """Return whether a root is distinct in a scale-relative sense."""

    return all(
        abs(candidate.density_scale - phase.density_scale)
        > 1.0e-10 * max(1.0, candidate.density_scale, phase.density_scale)
        for phase in phases
    )


def analyze_route2_v0_molecular_homogeneous_phase_coexistence(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    minimum_density_scale: float = 1.0e-10,
    gas_search_upper_density_scale: float = 0.999,
    root_sample_count: int = 513,
    directional_derivative_tolerance_hartree_per_bohr3: float = 1.0e-12,
    zero_external_tolerance_hartree: float = 1.0e-12,
    stationarity_tolerance: float = 1.0e-10,
    gradient_uniformity_tolerance: float = 1.0e-10,
    curvature_tolerance_hartree_per_bohr3: float = 1.0e-12,
    coexistence_tolerance_hartree_per_bohr3: float = 1.0e-12,
    maximum_bisection_iterations: int = 256,
) -> Route2V0MolecularPhaseCoexistenceGate:
    """Assess the no-solute homogeneous coexistence precondition.

    A finite log-density scan is only a root-bracketing device.  The returned
    result preserves all candidate roots and passes only when a low-density
    root is a full, stable stationary phase and has the same grand-potential
    density as the declared bulk-liquid phase.  The routine does not tune a
    bridge coefficient when the condition fails.
    """

    if not isinstance(functional, Route2V0MolecularWeightedDensityBridgeFunctional):
        raise TypeError(
            "Molecular phase-coexistence analysis requires an HNC-plus-bridge "
            "functional."
        )
    minimum = _positive(minimum_density_scale, name="Minimum homogeneous density scale")
    upper = _positive(
        gas_search_upper_density_scale,
        name="Gas-search upper homogeneous density scale",
    )
    if minimum >= upper or upper >= 1.0:
        raise ValueError("Homogeneous gas search must satisfy 0 < minimum < upper < 1.")
    sample_count = _positive_integer(root_sample_count, name="Homogeneous root samples")
    if sample_count < 3:
        raise ValueError("Homogeneous root samples must be at least three.")
    derivative_tolerance = _positive(
        directional_derivative_tolerance_hartree_per_bohr3,
        name="Homogeneous directional-derivative tolerance",
    )
    external_tolerance = _positive(
        zero_external_tolerance_hartree,
        name="Pure-liquid zero-external-potential tolerance",
    )
    stationary_tolerance = _positive(
        stationarity_tolerance, name="Homogeneous stationarity tolerance"
    )
    uniformity_tolerance = _positive(
        gradient_uniformity_tolerance,
        name="Homogeneous gradient-uniformity tolerance",
    )
    curvature_tolerance = _positive(
        curvature_tolerance_hartree_per_bohr3,
        name="Homogeneous curvature tolerance",
    )
    coexistence_tolerance = _positive(
        coexistence_tolerance_hartree_per_bohr3,
        name="Homogeneous coexistence tolerance",
    )
    iterations = _positive_integer(
        maximum_bisection_iterations,
        name="Homogeneous root bisection iterations",
    )
    external_residual = _pure_liquid_external_residual_hartree(functional)
    if external_residual > external_tolerance:
        raise ValueError(
            "Homogeneous phase coexistence requires an exactly declared "
            "zero-external-potential pure-liquid scalar."
        )
    liquid = evaluate_route2_v0_molecular_homogeneous_phase(
        functional,
        density_scale=1.0,
        curvature_tolerance_hartree_per_bohr3=curvature_tolerance,
    )
    scales = np.geomspace(minimum, upper, sample_count)
    evaluations = [
        evaluate_route2_v0_molecular_homogeneous_phase(
            functional,
            density_scale=float(scale),
            curvature_tolerance_hartree_per_bohr3=curvature_tolerance,
        )
        for scale in scales
    ]
    roots: list[Route2V0MolecularHomogeneousPhase] = []
    for left, right in pairwise(evaluations):
        left_value = left.directional_derivative_density_hartree_per_bohr3
        right_value = right.directional_derivative_density_hartree_per_bohr3
        if abs(left_value) <= derivative_tolerance:
            candidate = left
        elif abs(right_value) <= derivative_tolerance:
            candidate = right
        elif (left_value > 0.0) == (right_value > 0.0):
            continue
        else:
            candidate = _bisect_directional_root(
                functional,
                lower=left.density_scale,
                upper=right.density_scale,
                derivative_tolerance_hartree_per_bohr3=derivative_tolerance,
                curvature_tolerance_hartree_per_bohr3=curvature_tolerance,
                maximum_iterations=iterations,
            )
        if _distinct_scale(roots, candidate):
            roots.append(candidate)
    stable_candidates = [
        phase
        for phase in roots
        if _is_stationary_stable(
            phase,
            stationarity_tolerance=stationary_tolerance,
            gradient_uniformity_tolerance=uniformity_tolerance,
            curvature_tolerance_hartree_per_bohr3=curvature_tolerance,
        )
    ]
    gas = min(
        stable_candidates,
        key=lambda phase: phase.grand_potential_density_hartree_per_bohr3,
        default=None,
    )
    difference = (
        None
        if gas is None
        else (
            gas.grand_potential_density_hartree_per_bohr3
            - liquid.grand_potential_density_hartree_per_bohr3
        )
    )
    passes = bool(
        _is_stationary_stable(
            liquid,
            stationarity_tolerance=stationary_tolerance,
            gradient_uniformity_tolerance=uniformity_tolerance,
            curvature_tolerance_hartree_per_bohr3=curvature_tolerance,
        )
        and gas is not None
        and difference is not None
        and abs(difference) <= coexistence_tolerance
    )
    return Route2V0MolecularPhaseCoexistenceGate(
        liquid_phase=liquid,
        gas_phase=gas,
        gas_candidates=tuple(roots),
        external_potential_residual_hartree=external_residual,
        zero_external_tolerance_hartree=external_tolerance,
        stationarity_tolerance=stationary_tolerance,
        gradient_uniformity_tolerance=uniformity_tolerance,
        curvature_tolerance_hartree_per_bohr3=curvature_tolerance,
        coexistence_tolerance_hartree_per_bohr3=coexistence_tolerance,
        grand_potential_density_difference_hartree_per_bohr3=difference,
        passes=passes,
    )


__all__ = [
    "V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION",
    "Route2V0MolecularHomogeneousPhase",
    "Route2V0MolecularPhaseCoexistenceGate",
    "analyze_route2_v0_molecular_homogeneous_phase_coexistence",
    "evaluate_route2_v0_molecular_homogeneous_phase",
]
