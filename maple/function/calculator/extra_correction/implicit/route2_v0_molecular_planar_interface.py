"""Constrained planar controls for one Route-2 V0 molecular liquid scalar.

A planar liquid--gas profile is meaningful only after the same scalar has a
finite-density homogeneous coexistence branch.  It is not valid to reduce a
molecular liquid to a scalar density profile, to use a different functional
for a planar calculation, or to select a bridge coefficient from a surface
tension before that coexistence condition is met.

This module preserves the full Cartesian-times-``SO(3)`` configuration
variable.  It restricts only the *translation symmetry* of the trial state:
for a declared normal axis, the configuration density is identical at every
transverse translation while it remains a free function of the normal grid
coordinate and every orientation.  The restriction is an exact embedding of
the existing finite scalar.  Its reduced gradient is the quadrature-adjoint
transverse average of the full gradient, so finite-difference energy changes,
stationarity, and the particle-number constraint all refer to the same
functional.

A periodic slab is solved at a declared equimolar molecular-centre count.  The
constraint is numerical support for a two-interface stationary branch; the
reported state is admitted as an unconstrained planar interface only when its
Lagrange multiplier and full scalar residual vanish within frozen tolerances.
The surface excess is then evaluated by subtracting the liquid and finite-gas
bulk grand-potential densities from the *same* scalar.

The implementation is deliberately a control layer.  It neither provides a
physical solvent source nor reads a target surface tension, solute property,
or solvation label.  A zero or negative synthetic excess is diagnostic only;
it cannot become a physical pure-liquid certificate or an accuracy claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import SupportsFloat, SupportsIndex, cast

import numpy as np

from .route2_v0_molecular_coexistence_continuation import (
    Route2V0MolecularQuarticCoexistenceContinuation,
)
from .route2_v0_molecular_external_potential_contract import (
    require_molecular_external_potential_values,
)
from .route2_v0_molecular_so3_quadrature import (
    Route2V0CartesianEulerProductQuadrature,
)
from .route2_v0_molecular_weighted_density_bridge import (
    Route2V0MolecularWeightedDensityBridgeFunctional,
    Route2V0MolecularWeightedDensityBridgeState,
)
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MOLECULAR_PLANAR_SYMMETRY_CONSTRUCTION = (
    "route2-v0-molecular-full-so3-planar-symmetry-v1"
)
V0_MOLECULAR_CONSTRAINED_PLANAR_INTERFACE_CONSTRUCTION = (
    "route2-v0-molecular-constrained-planar-interface-v1"
)
V0_MOLECULAR_EQUIMOLAR_DIVIDING_SURFACE = "molecular-centre-equimolar-v1"


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
    """Return one finite strictly positive scalar."""

    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative(value: object, *, name: str) -> float:
    """Return one finite nonnegative scalar."""

    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
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


def _normal_axis(value: object) -> int:
    """Validate one Cartesian normal axis."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Planar normal axis must be an integer in [0, 2].")
    numeric = _finite(value, name="Planar normal axis")
    result = int(numeric)
    if result != numeric or result not in (0, 1, 2):
        raise ValueError("Planar normal axis must be an integer in [0, 2].")
    return result


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    positive: bool = False,
) -> np.ndarray:
    """Return one finite immutable real array with one exact shape."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if (
        array.shape != shape
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
    ):
        requirement = f"finite with shape {shape}"
        if positive:
            requirement += " and strictly positive entries"
        raise ValueError(f"{name} must be {requirement}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _same_grid(left: RegularCartesianGrid, right: RegularCartesianGrid) -> bool:
    """Return whether two Cartesian grids are exactly the same declaration."""

    return bool(
        left.shape == right.shape
        and np.array_equal(left.origin_bohr, right.origin_bohr)
        and np.array_equal(left.spacing_bohr, right.spacing_bohr)
        and left.layout == right.layout
    )


def _same_cartesian_euler_product(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    quadrature: Route2V0CartesianEulerProductQuadrature,
) -> bool:
    """Require the functional to use exactly one declared product quadrature."""

    projection_quadrature = functional.projection.quadrature
    return bool(
        _same_grid(
            functional.projection.site_hnc_asset.grid,
            quadrature.grid,
        )
        and np.array_equal(
            projection_quadrature.configurations.translations_bohr,
            quadrature.configurations.translations_bohr,
        )
        and np.array_equal(
            projection_quadrature.configurations.rotations,
            quadrature.configurations.rotations,
        )
        and np.array_equal(
            projection_quadrature.phase_space_weights_bohr3,
            quadrature.quadrature.phase_space_weights_bohr3,
        )
    )


def _coexistence_matches_functional(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    continuation: Route2V0MolecularQuarticCoexistenceContinuation,
) -> None:
    """Require a continuation root to belong to the live scalar coefficient."""

    if not isinstance(
        continuation,
        Route2V0MolecularQuarticCoexistenceContinuation,
    ):
        raise TypeError("Planar interface requires a quartic coexistence continuation.")
    if not continuation.passes:
        raise ValueError(
            "Planar interface requires a passing coexistence continuation."
        )
    expected = functional.bridge_asset.quartic_coefficient_hartree_bohr15
    observed = continuation.root.quartic_coefficient_hartree_bohr15
    tolerance = max(1.0e-12, continuation.coefficient_relative_tolerance) * max(
        1.0,
        abs(expected),
        abs(observed),
    )
    if abs(expected - observed) > tolerance:
        raise ValueError(
            "Planar interface coefficient must equal the same-scalar coexistence "
            "continuation root."
        )
    root = continuation.root
    gate = root.phase_gate
    configuration_count = functional.projection.quadrature.configuration_count
    external = require_molecular_external_potential_values(
        functional.projection.external_potential,
        configuration_count=configuration_count,
    )
    external_residual = float(np.max(np.abs(external)))
    if external_residual > gate.zero_external_tolerance_hartree:
        raise ValueError(
            "Planar interface requires the continuation's zero-external-potential "
            "pure-liquid scalar."
        )
    grid = functional.projection.site_hnc_asset.grid
    volume = float(grid.point_count * grid.volume_element_bohr3)
    bulk_configuration_density = (
        functional.projection.uniform_configuration_density_bohr3
    )
    weights = functional.projection.quadrature.phase_space_weights_bohr3
    for label, phase in (("liquid", root.liquid_phase), ("gas", root.gas_phase)):
        density = np.full(
            configuration_count,
            phase.density_scale * bulk_configuration_density,
            dtype=float,
        )
        gradient = functional.dimensionless_gradient(density)
        grand_potential_density = functional.grand_potential_hartree(density) / volume
        reference_energy = phase.grand_potential_density_hartree_per_bohr3
        energy_tolerance = max(
            1.0e-12,
            gate.coexistence_tolerance_hartree_per_bohr3,
        ) * max(1.0, abs(grand_potential_density), abs(reference_energy))
        if abs(grand_potential_density - reference_energy) > energy_tolerance:
            raise ValueError(
                "Planar interface functional does not reproduce the continuation "
                f"{label}-branch grand-potential density."
            )
        residual = float(np.max(np.abs(gradient)))
        uniformity = float(
            np.max(
                np.abs(gradient - float(np.sum(weights * gradient) / np.sum(weights)))
            )
        )
        if (
            residual > gate.stationarity_tolerance
            or uniformity > gate.gradient_uniformity_tolerance
        ):
            raise ValueError(
                "Planar interface functional does not reproduce the continuation "
                f"{label}-branch full stationarity."
            )
        direction = np.full(configuration_count, bulk_configuration_density)
        curvature = (
            functional.kbt_hartree
            * float(
                np.sum(
                    weights
                    * direction
                    * functional.dimensionless_hessian_matvec(density, direction)
                )
            )
            / volume
        )
        if curvature <= gate.curvature_tolerance_hartree_per_bohr3:
            raise ValueError(
                "Planar interface functional does not reproduce the continuation "
                f"{label}-branch stable curvature."
            )


@dataclass(frozen=True)
class Route2V0MolecularPlanarSymmetry:
    """Exact full-``SO(3)`` planar embedding for one molecular scalar.

    The reduced variable has shape ``(n_normal, n_orientation)``.  It is
    replicated across transverse Cartesian translations, never across
    orientations.  The explicitly supplied product quadrature is required so
    that a hand-built one-pose or non-product configuration list cannot be
    relabelled as an exact planar restriction.
    """

    functional: Route2V0MolecularWeightedDensityBridgeFunctional
    cartesian_euler_quadrature: Route2V0CartesianEulerProductQuadrature
    normal_axis: int = 2
    construction: str = V0_MOLECULAR_PLANAR_SYMMETRY_CONSTRUCTION
    _orientation_count: int = field(init=False, repr=False, compare=False)
    _transverse_axes: tuple[int, int] = field(init=False, repr=False, compare=False)
    _transverse_area_bohr2: float = field(init=False, repr=False, compare=False)
    _reduced_phase_space_weights_bohr3: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.functional,
            Route2V0MolecularWeightedDensityBridgeFunctional,
        ):
            raise TypeError("Planar symmetry requires a molecular bridge functional.")
        if not isinstance(
            self.cartesian_euler_quadrature,
            Route2V0CartesianEulerProductQuadrature,
        ):
            raise TypeError(
                "Planar symmetry requires the exact Cartesian-Euler product quadrature."
            )
        if self.construction != V0_MOLECULAR_PLANAR_SYMMETRY_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular planar symmetry.")
        if not _same_cartesian_euler_product(
            self.functional,
            self.cartesian_euler_quadrature,
        ):
            raise ValueError(
                "Planar symmetry requires the exact Cartesian-Euler product "
                "quadrature used by the molecular scalar."
            )
        normal = _normal_axis(self.normal_axis)
        grid = self.functional.projection.site_hnc_asset.grid
        orientation = self.cartesian_euler_quadrature.orientation_quadrature
        orientation_count = orientation.orientation_count
        expected_configuration_count = grid.point_count * orientation_count
        if (
            self.functional.projection.quadrature.configuration_count
            != expected_configuration_count
        ):
            raise RuntimeError(
                "Cartesian-Euler product configuration count is inconsistent with "
                "the molecular scalar grid."
            )
        transverse_axes = tuple(axis for axis in range(3) if axis != normal)
        transverse_area = float(
            np.prod(
                [grid.shape[axis] * grid.spacing_bohr[axis] for axis in transverse_axes]
            )
        )
        if not math.isfinite(transverse_area) or transverse_area <= 0.0:
            raise RuntimeError("Planar transverse area is invalid.")
        reduced_weights = (
            transverse_area
            * grid.spacing_bohr[normal]
            * orientation.orientation_weights[None, :]
        )
        reduced_weights = np.broadcast_to(
            reduced_weights,
            (grid.shape[normal], orientation_count),
        )
        expected_measure = float(
            np.sum(self.functional.projection.quadrature.phase_space_weights_bohr3)
        )
        actual_measure = float(np.sum(reduced_weights))
        tolerance = 1.0e-12 * max(1.0, abs(expected_measure), abs(actual_measure))
        if abs(expected_measure - actual_measure) > tolerance:
            raise RuntimeError(
                "Planar reduced quadrature does not preserve the full molecular "
                "phase-space measure."
            )
        object.__setattr__(self, "normal_axis", normal)
        object.__setattr__(self, "_orientation_count", orientation_count)
        object.__setattr__(self, "_transverse_axes", transverse_axes)
        object.__setattr__(self, "_transverse_area_bohr2", transverse_area)
        object.__setattr__(
            self,
            "_reduced_phase_space_weights_bohr3",
            _immutable_array(
                reduced_weights,
                name="Planar reduced phase-space weights",
                shape=(grid.shape[normal], orientation_count),
                positive=True,
            ),
        )

    @property
    def normal_point_count(self) -> int:
        """Return the number of grid points normal to the planar interface."""

        return self.functional.projection.site_hnc_asset.grid.shape[self.normal_axis]

    @property
    def orientation_count(self) -> int:
        """Return the retained full-``SO(3)`` orientation count."""

        return self._orientation_count

    @property
    def transverse_axes(self) -> tuple[int, int]:
        """Return the Cartesian axes that are made translation-invariant."""

        return self._transverse_axes

    @property
    def transverse_area_bohr2(self) -> float:
        """Return the periodic transverse area represented by the exact grid."""

        return self._transverse_area_bohr2

    @property
    def reduced_shape(self) -> tuple[int, int]:
        """Return the exact planar density shape ``(normal, orientation)``."""

        return (self.normal_point_count, self.orientation_count)

    @property
    def reduced_phase_space_weights_bohr3(self) -> np.ndarray:
        """Return the reduced weights preserving the full scalar pairing."""

        return self._reduced_phase_space_weights_bohr3

    @property
    def orientation_weights(self) -> np.ndarray:
        """Return the unchanged full-``SO(3)`` Haar quadrature weights."""

        return (
            self.cartesian_euler_quadrature.orientation_quadrature.orientation_weights
        )

    def validate_reduced_configuration_density(
        self, values: np.ndarray, *, name: str
    ) -> np.ndarray:
        return _immutable_array(
            values,
            name=name,
            shape=self.reduced_shape,
            positive=True,
        )

    def _full_values(self, values: np.ndarray, *, name: str) -> np.ndarray:
        return _immutable_array(
            values,
            name=name,
            shape=(self.functional.projection.quadrature.configuration_count,),
        )

    def expand_reduced_field(
        self, reduced_values: np.ndarray, *, name: str
    ) -> np.ndarray:
        """Embed one signed reduced field without density positivity constraints."""

        reduced = _immutable_array(
            reduced_values,
            name=name,
            shape=self.reduced_shape,
        )
        grid = self.functional.projection.site_hnc_asset.grid
        shape = [1, 1, 1, self.orientation_count]
        shape[self.normal_axis] = self.normal_point_count
        full = np.broadcast_to(
            reduced.reshape(tuple(shape)),
            (*grid.shape, self.orientation_count),
        )
        result = np.array(full.reshape((-1,)), dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def expand_configuration_density(
        self, reduced_density_bohr3: np.ndarray
    ) -> np.ndarray:
        """Embed one planar density exactly into the full configuration vector."""

        reduced = self.validate_reduced_configuration_density(
            reduced_density_bohr3,
            name="Planar reduced configuration density",
        )
        return self.expand_reduced_field(
            reduced,
            name="Planar reduced configuration density",
        )

    def reduce_dimensionless_gradient(self, full_gradient: np.ndarray) -> np.ndarray:
        """Return the exact transverse-average gradient of the restricted scalar."""

        values = self._full_values(full_gradient, name="Full configuration gradient")
        grid = self.functional.projection.site_hnc_asset.grid
        reshaped = values.reshape((*grid.shape, self.orientation_count))
        reduced = np.mean(reshaped, axis=self.transverse_axes)
        return _immutable_array(
            reduced,
            name="Planar reduced configuration gradient",
            shape=self.reduced_shape,
        )

    def transverse_gradient_nonuniformity(
        self,
        full_gradient: np.ndarray,
        *,
        reduced_gradient: np.ndarray | None = None,
    ) -> float:
        """Return the residual discarded by the planar symmetry restriction."""

        full = self._full_values(full_gradient, name="Full configuration gradient")
        reduced = (
            self.reduce_dimensionless_gradient(full)
            if reduced_gradient is None
            else _immutable_array(
                reduced_gradient,
                name="Planar reduced configuration gradient",
                shape=self.reduced_shape,
            )
        )
        embedded = self.expand_reduced_field(
            reduced,
            name="Planar reduced configuration gradient",
        )
        residual = float(np.max(np.abs(full - embedded)))
        if not math.isfinite(residual):
            raise RuntimeError("Planar transverse-gradient residual is non-finite.")
        return residual

    def molecule_count(self, reduced_density_bohr3: np.ndarray) -> float:
        """Return the molecular count represented by one planar density."""

        density = self.validate_reduced_configuration_density(
            reduced_density_bohr3,
            name="Planar reduced configuration density",
        )
        count = float(np.sum(self.reduced_phase_space_weights_bohr3 * density))
        if not math.isfinite(count) or count <= 0.0:
            raise RuntimeError("Planar molecular count is invalid.")
        return count

    def molecular_number_density_profile_bohr3(
        self,
        reduced_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Integrate the retained orientations into the molecular-centre profile."""

        density = self.validate_reduced_configuration_density(
            reduced_density_bohr3,
            name="Planar reduced configuration density",
        )
        profile = np.einsum("zo,o->z", density, self.orientation_weights, optimize=True)
        return _immutable_array(
            profile,
            name="Planar molecular number-density profile",
            shape=(self.normal_point_count,),
            positive=True,
        )


@dataclass(frozen=True)
class Route2V0MolecularConstrainedPlanarInterfaceState:
    """One same-scalar periodic two-phase planar-state diagnostic.

    ``passes`` requires the particle-constrained solve to have relaxed back to
    an unconstrained stationary state.  A positive surface tension is exposed
    separately because synthetic local controls legitimately have zero excess
    and must not be promoted to physical liquid evidence.
    """

    functional: Route2V0MolecularWeightedDensityBridgeFunctional
    planar_symmetry: Route2V0MolecularPlanarSymmetry
    coexistence_continuation: Route2V0MolecularQuarticCoexistenceContinuation
    state: Route2V0MolecularWeightedDensityBridgeState
    reduced_configuration_density_bohr3: np.ndarray
    target_mean_density_scale: float
    constraint_multiplier_dimensionless: float
    constrained_residual_inf: float
    transverse_gradient_nonuniformity: float
    stationarity_tolerance: float
    transverse_uniformity_tolerance: float
    constraint_multiplier_tolerance: float
    interface_count: int
    equimolar_liquid_fraction: float
    equimolar_liquid_volume_bohr3: float
    equimolar_gas_volume_bohr3: float
    surface_tension_hartree_per_bohr2: float
    iterations: int
    dividing_surface: str = V0_MOLECULAR_EQUIMOLAR_DIVIDING_SURFACE
    construction: str = V0_MOLECULAR_CONSTRAINED_PLANAR_INTERFACE_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(
            self.functional,
            Route2V0MolecularWeightedDensityBridgeFunctional,
        ):
            raise TypeError("Planar interface state requires a molecular functional.")
        if not isinstance(self.planar_symmetry, Route2V0MolecularPlanarSymmetry):
            raise TypeError("Planar interface state requires planar symmetry.")
        if self.planar_symmetry.functional is not self.functional:
            raise ValueError("Planar interface symmetry must bind the live scalar.")
        _coexistence_matches_functional(self.functional, self.coexistence_continuation)
        if not isinstance(self.state, Route2V0MolecularWeightedDensityBridgeState):
            raise TypeError("Planar interface state requires a scalar state record.")
        if self.state.functional is not self.functional:
            raise ValueError("Planar interface state must use the live scalar state.")
        if self.construction != V0_MOLECULAR_CONSTRAINED_PLANAR_INTERFACE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 constrained planar interface.")
        if self.dividing_surface != V0_MOLECULAR_EQUIMOLAR_DIVIDING_SURFACE:
            raise ValueError("Unsupported Route-2 planar dividing-surface convention.")
        density = self.planar_symmetry.validate_reduced_configuration_density(
            self.reduced_configuration_density_bohr3,
            name="Planar reduced configuration density",
        )
        full_density = self.planar_symmetry.expand_configuration_density(density)
        if not np.array_equal(full_density, self.state.configuration_density_bohr3):
            raise ValueError(
                "Planar state density must be the exact planar embedding of its "
                "reduced density."
            )
        target = _positive(
            self.target_mean_density_scale,
            name="Planar target mean-density scale",
        )
        multiplier = _finite(
            self.constraint_multiplier_dimensionless,
            name="Planar constraint multiplier",
        )
        constrained = _nonnegative(
            self.constrained_residual_inf,
            name="Planar constrained residual",
        )
        transverse = _nonnegative(
            self.transverse_gradient_nonuniformity,
            name="Planar transverse-gradient residual",
        )
        stationarity_tolerance = _positive(
            self.stationarity_tolerance,
            name="Planar stationarity tolerance",
        )
        transverse_tolerance = _positive(
            self.transverse_uniformity_tolerance,
            name="Planar transverse-uniformity tolerance",
        )
        multiplier_tolerance = _positive(
            self.constraint_multiplier_tolerance,
            name="Planar constraint-multiplier tolerance",
        )
        interface_count = _positive_integer(
            self.interface_count,
            name="Planar interface count",
        )
        if interface_count % 2:
            raise ValueError("Periodic planar-interface count must be even.")
        liquid_fraction = _finite(
            self.equimolar_liquid_fraction,
            name="Planar equimolar liquid fraction",
        )
        if not 0.0 < liquid_fraction < 1.0:
            raise ValueError("Planar equimolar liquid fraction must lie in (0, 1).")
        liquid_volume = _positive(
            self.equimolar_liquid_volume_bohr3,
            name="Planar equimolar liquid volume",
        )
        gas_volume = _positive(
            self.equimolar_gas_volume_bohr3,
            name="Planar equimolar gas volume",
        )
        grid = self.functional.projection.site_hnc_asset.grid
        volume = float(grid.point_count * grid.volume_element_bohr3)
        volume_tolerance = 1.0e-12 * max(1.0, volume, liquid_volume + gas_volume)
        if abs(liquid_volume + gas_volume - volume) > volume_tolerance:
            raise ValueError("Planar equimolar bulk volumes must fill the full cell.")
        if abs(liquid_volume / volume - liquid_fraction) > 1.0e-12:
            raise ValueError("Planar equimolar liquid volume has the wrong fraction.")
        surface_tension = _finite(
            self.surface_tension_hartree_per_bohr2,
            name="Planar surface excess",
        )
        iterations = (
            _positive_integer(self.iterations + 1, name="Planar iterations") - 1
        )
        root = self.coexistence_continuation.root
        profile = self.planar_symmetry.molecular_number_density_profile_bohr3(density)
        bulk = self.functional.projection.molecular_bulk_number_density_bohr3
        gas_density = root.gas_phase.density_scale * bulk
        liquid_density = root.liquid_phase.density_scale * bulk
        threshold = 0.5 * (gas_density + liquid_density)
        above = profile >= threshold
        observed_count = int(np.count_nonzero(above != np.roll(above, 1)))
        if observed_count != interface_count:
            raise ValueError(
                "Planar profile crossing count must equal the declared periodic "
                "interface count."
            )
        actual_scale = self.planar_symmetry.molecule_count(density) / (volume * bulk)
        scale_tolerance = 1.0e-12 * max(1.0, abs(actual_scale), abs(target))
        if abs(actual_scale - target) > scale_tolerance:
            raise ValueError(
                "Planar state does not retain its declared molecule count."
            )
        expected_fraction = (target - root.gas_phase.density_scale) / (
            root.liquid_phase.density_scale - root.gas_phase.density_scale
        )
        if abs(expected_fraction - liquid_fraction) > 1.0e-12:
            raise ValueError(
                "Planar equimolar fraction must follow the declared homogeneous "
                "liquid and gas branches."
            )
        expected_bulk = (
            liquid_volume * root.liquid_phase.grand_potential_density_hartree_per_bohr3
            + gas_volume * root.gas_phase.grand_potential_density_hartree_per_bohr3
        )
        expected_surface = (self.state.grand_potential_hartree - expected_bulk) / (
            interface_count * self.planar_symmetry.transverse_area_bohr2
        )
        surface_tolerance = 1.0e-12 * max(
            1.0,
            abs(surface_tension),
            abs(expected_surface),
        )
        if abs(surface_tension - expected_surface) > surface_tolerance:
            raise ValueError(
                "Planar surface excess must subtract the same-scalar liquid and "
                "gas bulk grand potentials."
            )
        object.__setattr__(self, "reduced_configuration_density_bohr3", density)
        object.__setattr__(self, "target_mean_density_scale", target)
        object.__setattr__(self, "constraint_multiplier_dimensionless", multiplier)
        object.__setattr__(self, "constrained_residual_inf", constrained)
        object.__setattr__(self, "transverse_gradient_nonuniformity", transverse)
        object.__setattr__(self, "stationarity_tolerance", stationarity_tolerance)
        object.__setattr__(
            self, "transverse_uniformity_tolerance", transverse_tolerance
        )
        object.__setattr__(
            self,
            "constraint_multiplier_tolerance",
            multiplier_tolerance,
        )
        object.__setattr__(self, "interface_count", interface_count)
        object.__setattr__(self, "equimolar_liquid_fraction", liquid_fraction)
        object.__setattr__(self, "equimolar_liquid_volume_bohr3", liquid_volume)
        object.__setattr__(self, "equimolar_gas_volume_bohr3", gas_volume)
        object.__setattr__(self, "surface_tension_hartree_per_bohr2", surface_tension)
        object.__setattr__(self, "iterations", iterations)

    @property
    def unconstrained_residual_inf(self) -> float:
        """Return the full-gradient residual of the original scalar."""

        return self.state.residual_inf

    @property
    def is_unconstrained_stationary(self) -> bool:
        """Return whether the constrained branch relaxed to one scalar stationary point."""

        return bool(
            self.constrained_residual_inf <= self.stationarity_tolerance
            and self.transverse_gradient_nonuniformity
            <= self.transverse_uniformity_tolerance
            and self.unconstrained_residual_inf <= self.stationarity_tolerance
            and abs(self.constraint_multiplier_dimensionless)
            <= self.constraint_multiplier_tolerance
        )

    @property
    def has_positive_surface_tension(self) -> bool:
        """Return whether the same-scalar surface excess is strictly positive."""

        return self.surface_tension_hartree_per_bohr2 > 0.0

    @property
    def passes(self) -> bool:
        """Return the structural stationary-interface status, not physical admission."""

        return self.is_unconstrained_stationary


def _target_count(
    symmetry: Route2V0MolecularPlanarSymmetry,
    *,
    target_mean_density_scale: float,
) -> float:
    """Return the exact target molecular count for one mean-density scale."""

    target = _positive(
        target_mean_density_scale,
        name="Planar target mean-density scale",
    )
    grid = symmetry.functional.projection.site_hnc_asset.grid
    volume = float(grid.point_count * grid.volume_element_bohr3)
    bulk = symmetry.functional.projection.molecular_bulk_number_density_bohr3
    count = target * volume * bulk
    if not math.isfinite(count) or count <= 0.0:
        raise RuntimeError("Planar target molecular count is invalid.")
    return count


def _constrained_multiplier_dimensionless(
    symmetry: Route2V0MolecularPlanarSymmetry,
    reduced_gradient: np.ndarray,
) -> float:
    """Return the weighted constant offset minimizing constrained residual."""

    gradient = _immutable_array(
        reduced_gradient,
        name="Planar reduced configuration gradient",
        shape=symmetry.reduced_shape,
    )
    weights = symmetry.reduced_phase_space_weights_bohr3
    multiplier = -float(np.sum(weights * gradient) / np.sum(weights))
    if not math.isfinite(multiplier):
        raise RuntimeError("Planar constraint multiplier is non-finite.")
    return multiplier


def _default_two_interface_density(
    symmetry: Route2V0MolecularPlanarSymmetry,
    continuation: Route2V0MolecularQuarticCoexistenceContinuation,
    *,
    target_mean_density_scale: float,
) -> np.ndarray:
    """Build a neutral numeric slab seed from the declared coexistence branches."""

    root = continuation.root
    gas_scale = root.gas_phase.density_scale
    liquid_scale = root.liquid_phase.density_scale
    target = _positive(
        target_mean_density_scale,
        name="Planar target mean-density scale",
    )
    if not gas_scale < target < liquid_scale:
        raise ValueError(
            "Planar target mean density must lie strictly between the same-scalar "
            "gas and liquid branch densities."
        )
    liquid_fraction = (target - gas_scale) / (liquid_scale - gas_scale)
    normal_count = symmetry.normal_point_count
    liquid_points = round(liquid_fraction * normal_count)
    if not 0 < liquid_points < normal_count:
        raise ValueError(
            "Planar normal grid cannot represent both gas and liquid plateaux at "
            "the declared mean density."
        )
    profile_scale = np.full(normal_count, gas_scale, dtype=float)
    profile_scale[:liquid_points] = liquid_scale
    density = (
        symmetry.functional.projection.uniform_configuration_density_bohr3
        * profile_scale[:, None]
        * np.ones((1, symmetry.orientation_count), dtype=float)
    )
    target_count = _target_count(
        symmetry,
        target_mean_density_scale=target,
    )
    current_count = symmetry.molecule_count(density)
    density *= target_count / current_count
    return symmetry.validate_reduced_configuration_density(
        density,
        name="Default planar configuration density",
    )


def evaluate_route2_v0_molecular_constrained_planar_interface(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    planar_symmetry: Route2V0MolecularPlanarSymmetry,
    coexistence_continuation: Route2V0MolecularQuarticCoexistenceContinuation,
    reduced_configuration_density_bohr3: np.ndarray,
    target_mean_density_scale: float,
    constraint_multiplier_dimensionless: float,
    stationarity_tolerance: float = 1.0e-10,
    transverse_uniformity_tolerance: float = 1.0e-10,
    constraint_multiplier_tolerance: float = 1.0e-10,
    interface_count: int = 2,
    iterations: int = 0,
) -> Route2V0MolecularConstrainedPlanarInterfaceState:
    """Evaluate one constrained planar state from the exact full scalar.

    This function does not modify the functional or select an interface width.
    It evaluates all energy and gradient quantities by embedding the planar
    density into the existing full configuration space.
    """

    if not isinstance(functional, Route2V0MolecularWeightedDensityBridgeFunctional):
        raise TypeError("Planar interface requires a molecular bridge functional.")
    if not isinstance(planar_symmetry, Route2V0MolecularPlanarSymmetry):
        raise TypeError("Planar interface requires a planar-symmetry embedding.")
    if planar_symmetry.functional is not functional:
        raise ValueError("Planar interface symmetry must bind the live scalar.")
    _coexistence_matches_functional(functional, coexistence_continuation)
    density = planar_symmetry.validate_reduced_configuration_density(
        reduced_configuration_density_bohr3,
        name="Planar reduced configuration density",
    )
    target = _positive(
        target_mean_density_scale,
        name="Planar target mean-density scale",
    )
    root = coexistence_continuation.root
    if not root.gas_phase.density_scale < target < root.liquid_phase.density_scale:
        raise ValueError(
            "Planar target mean density must lie strictly between the same-scalar "
            "gas and liquid branch densities."
        )
    multiplier = _finite(
        constraint_multiplier_dimensionless,
        name="Planar constraint multiplier",
    )
    stationarity = _positive(
        stationarity_tolerance,
        name="Planar stationarity tolerance",
    )
    transverse_tolerance = _positive(
        transverse_uniformity_tolerance,
        name="Planar transverse-uniformity tolerance",
    )
    multiplier_tolerance = _positive(
        constraint_multiplier_tolerance,
        name="Planar constraint-multiplier tolerance",
    )
    count = _positive_integer(interface_count, name="Planar interface count")
    if count % 2:
        raise ValueError("Periodic planar-interface count must be even.")
    iteration_count = _positive_integer(iterations + 1, name="Planar iterations") - 1
    full_density = planar_symmetry.expand_configuration_density(density)
    state = functional.stationary_state(full_density, iterations=iteration_count)
    full_gradient = functional.dimensionless_gradient(full_density)
    reduced_gradient = planar_symmetry.reduce_dimensionless_gradient(full_gradient)
    constrained_residual = float(np.max(np.abs(reduced_gradient + multiplier)))
    transverse_residual = planar_symmetry.transverse_gradient_nonuniformity(
        full_gradient,
        reduced_gradient=reduced_gradient,
    )
    grid = functional.projection.site_hnc_asset.grid
    volume = float(grid.point_count * grid.volume_element_bohr3)
    bulk = functional.projection.molecular_bulk_number_density_bohr3
    actual_mean_scale = planar_symmetry.molecule_count(density) / (volume * bulk)
    target_scale_tolerance = 1.0e-12 * max(
        1.0,
        abs(actual_mean_scale),
        abs(target),
    )
    if abs(actual_mean_scale - target) > target_scale_tolerance:
        raise ValueError("Planar state does not retain the declared molecule count.")
    liquid_fraction = (target - root.gas_phase.density_scale) / (
        root.liquid_phase.density_scale - root.gas_phase.density_scale
    )
    liquid_volume = liquid_fraction * volume
    gas_volume = (1.0 - liquid_fraction) * volume
    bulk_subtraction = (
        liquid_volume * root.liquid_phase.grand_potential_density_hartree_per_bohr3
        + gas_volume * root.gas_phase.grand_potential_density_hartree_per_bohr3
    )
    surface_tension = (state.grand_potential_hartree - bulk_subtraction) / (
        count * planar_symmetry.transverse_area_bohr2
    )
    return Route2V0MolecularConstrainedPlanarInterfaceState(
        functional=functional,
        planar_symmetry=planar_symmetry,
        coexistence_continuation=coexistence_continuation,
        state=state,
        reduced_configuration_density_bohr3=density,
        target_mean_density_scale=target,
        constraint_multiplier_dimensionless=multiplier,
        constrained_residual_inf=constrained_residual,
        transverse_gradient_nonuniformity=transverse_residual,
        stationarity_tolerance=stationarity,
        transverse_uniformity_tolerance=transverse_tolerance,
        constraint_multiplier_tolerance=multiplier_tolerance,
        interface_count=count,
        equimolar_liquid_fraction=liquid_fraction,
        equimolar_liquid_volume_bohr3=liquid_volume,
        equimolar_gas_volume_bohr3=gas_volume,
        surface_tension_hartree_per_bohr2=surface_tension,
        iterations=iteration_count,
    )


def solve_route2_v0_molecular_constrained_planar_interface(
    functional: Route2V0MolecularWeightedDensityBridgeFunctional,
    *,
    planar_symmetry: Route2V0MolecularPlanarSymmetry,
    coexistence_continuation: Route2V0MolecularQuarticCoexistenceContinuation,
    target_mean_density_scale: float,
    initial_reduced_configuration_density_bohr3: np.ndarray | None = None,
    stationarity_tolerance: float = 1.0e-10,
    transverse_uniformity_tolerance: float = 1.0e-10,
    constraint_multiplier_tolerance: float = 1.0e-10,
    picard_mixing: float = 0.2,
    maximum_iterations: int = 1000,
) -> Route2V0MolecularConstrainedPlanarInterfaceState:
    """Solve a fixed-count periodic planar branch and require zero multiplier.

    The log-density step is normalized after each iteration to retain the
    declared equimolar molecular-centre count exactly.  This count is a
    numerical branch constraint, not an additional physical model term.  The
    returned state passes only when the corresponding Lagrange multiplier has
    decayed to zero, recovering stationarity of the unmodified grand scalar.
    """

    if not isinstance(functional, Route2V0MolecularWeightedDensityBridgeFunctional):
        raise TypeError("Planar interface requires a molecular bridge functional.")
    if not isinstance(planar_symmetry, Route2V0MolecularPlanarSymmetry):
        raise TypeError("Planar interface requires a planar-symmetry embedding.")
    if planar_symmetry.functional is not functional:
        raise ValueError("Planar interface symmetry must bind the live scalar.")
    _coexistence_matches_functional(functional, coexistence_continuation)
    target = _positive(
        target_mean_density_scale,
        name="Planar target mean-density scale",
    )
    root = coexistence_continuation.root
    if not root.gas_phase.density_scale < target < root.liquid_phase.density_scale:
        raise ValueError(
            "Planar target mean density must lie strictly between the same-scalar "
            "gas and liquid branch densities."
        )
    stationarity = _positive(
        stationarity_tolerance,
        name="Planar stationarity tolerance",
    )
    transverse_tolerance = _positive(
        transverse_uniformity_tolerance,
        name="Planar transverse-uniformity tolerance",
    )
    multiplier_tolerance = _positive(
        constraint_multiplier_tolerance,
        name="Planar constraint-multiplier tolerance",
    )
    mixing = _positive(picard_mixing, name="Planar Picard mixing")
    if mixing > 1.0:
        raise ValueError("Planar Picard mixing must be at most one.")
    maximum = _positive_integer(
        maximum_iterations,
        name="Planar maximum iterations",
    )
    if initial_reduced_configuration_density_bohr3 is None:
        density = np.array(
            _default_two_interface_density(
                planar_symmetry,
                coexistence_continuation,
                target_mean_density_scale=target,
            ),
            dtype=float,
            copy=True,
        )
    else:
        density = np.array(
            planar_symmetry.validate_reduced_configuration_density(
                initial_reduced_configuration_density_bohr3,
                name="Initial planar configuration density",
            ),
            dtype=float,
            copy=True,
        )
    target_count = _target_count(
        planar_symmetry,
        target_mean_density_scale=target,
    )
    density *= target_count / planar_symmetry.molecule_count(density)

    for iteration in range(maximum + 1):
        full_density = planar_symmetry.expand_configuration_density(density)
        full_gradient = functional.dimensionless_gradient(full_density)
        reduced_gradient = planar_symmetry.reduce_dimensionless_gradient(full_gradient)
        multiplier = _constrained_multiplier_dimensionless(
            planar_symmetry,
            reduced_gradient,
        )
        state = evaluate_route2_v0_molecular_constrained_planar_interface(
            functional,
            planar_symmetry=planar_symmetry,
            coexistence_continuation=coexistence_continuation,
            reduced_configuration_density_bohr3=density,
            target_mean_density_scale=target,
            constraint_multiplier_dimensionless=multiplier,
            stationarity_tolerance=stationarity,
            transverse_uniformity_tolerance=transverse_tolerance,
            constraint_multiplier_tolerance=multiplier_tolerance,
            iterations=iteration,
        )
        if state.passes:
            return state
        if iteration == maximum:
            break
        log_density = np.log(density)
        log_density -= mixing * (reduced_gradient + multiplier)
        if not np.all(np.isfinite(log_density)):
            raise RuntimeError(
                "Planar Picard iteration produced a non-finite log density."
            )
        density = np.exp(log_density)
        if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
            raise RuntimeError("Planar Picard iteration produced an invalid density.")
        density *= target_count / planar_symmetry.molecule_count(density)

    raise RuntimeError(
        "Constrained planar Picard solver did not recover an unconstrained "
        f"stationary interface within {maximum} iterations."
    )


__all__ = [
    "V0_MOLECULAR_CONSTRAINED_PLANAR_INTERFACE_CONSTRUCTION",
    "V0_MOLECULAR_EQUIMOLAR_DIVIDING_SURFACE",
    "V0_MOLECULAR_PLANAR_SYMMETRY_CONSTRUCTION",
    "Route2V0MolecularConstrainedPlanarInterfaceState",
    "Route2V0MolecularPlanarSymmetry",
    "evaluate_route2_v0_molecular_constrained_planar_interface",
    "solve_route2_v0_molecular_constrained_planar_interface",
]
