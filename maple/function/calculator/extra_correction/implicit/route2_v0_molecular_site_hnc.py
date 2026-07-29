"""Molecular-configuration HNC scalar control for Route-2 V0.

The Route-2 V0 molecular external-potential control assigns one energy to each
rigid solvent configuration ``Gamma = (X, Omega)``.  A site-HNC excess term,
however, is expressed in Cartesian solvent-site densities.  This module makes
that relationship explicit through one fixed discrete occupancy map instead
of adding an unrelated sitewise free energy:

``n_a(g) = dv^-1 sum_i w_i A[a, g, i] nu_i``.

``A`` deposits the sites of one molecular configuration onto the periodic
Cartesian liquid grid.  Its sum over grid points is the declared multiplicity
of each distinct solvent-site type.  With the unnormalised SO(3) measure
``int dOmega = 8*pi^2``, the quadrature must cover exactly one periodic cell,
and its uniform configuration density maps exactly to the site-HNC bulk
reference density.

For a reciprocal site--site direct-correlation control ``c_ab``, the one
scalar is

``Omega[nu] = Omega_id[nu; u]
              - kBT/2 * dv * sum_ag delta_n[a,g] (c * delta_n)[a,g]``.

The projected molecular stationarity equation is the derivative of that same
scalar.  This remains a synthetic/reference construction: it does not turn a
raw 1D-RISM table into a physical liquid kernel, supply a frozen real-solvent
asset, add the separately required long-range/pressure conventions, or claim
a force, PES, solvation free energy, or accuracy result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
)
from .route2_v0_molecular_external_potential_contract import (
    Route2V0MolecularExternalPotentialContract,
    require_molecular_external_potential_values,
)
from .route2_v0_molecular_ideal_gas import (
    RIGID_MOLECULAR_ORIENTATION_MEASURE,
    Route2V0MolecularConfigurationQuadrature,
    Route2V0MolecularIdealGasFunctional,
)
from .route2_v0_site_hnc import (
    Route2V0SiteHNCAsset,
    Route2V0SiteHNCFunctional,
)
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MOLECULAR_SITE_HNC_CONSTRUCTION = "route2-v0-molecular-site-hnc-control-v1"


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    """Return one finite immutable real array with an exact shape."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if (
        array.shape != shape
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
        or (nonnegative and np.any(array < 0.0))
    ):
        requirement = "finite"
        if positive:
            requirement += " and strictly positive"
        elif nonnegative:
            requirement += " and nonnegative"
        raise ValueError(f"{name} must be {requirement} with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _same_configurations(
    left: Route2V0MolecularConfigurations,
    right: object,
) -> bool:
    """Return whether two configuration grids have exactly equal coordinates."""

    if not isinstance(right, Route2V0MolecularConfigurations):
        return False
    return bool(
        np.array_equal(left.translations_bohr, right.translations_bohr)
        and np.array_equal(left.rotations, right.rotations)
    )


def _same_grid(left: RegularCartesianGrid, right: RegularCartesianGrid) -> bool:
    """Return whether two Cartesian grids have one exactly shared convention."""

    return bool(
        left.shape == right.shape
        and left.layout == right.layout
        and np.array_equal(left.origin_bohr, right.origin_bohr)
        and np.array_equal(left.spacing_bohr, right.spacing_bohr)
    )


def _site_type_indices(
    values: np.ndarray,
    *,
    site_count: int,
    solvent_site_count: int,
) -> np.ndarray:
    """Validate one complete molecular-site to HNC-site-type mapping."""

    raw = np.asarray(values)
    if raw.shape != (solvent_site_count,) or np.iscomplexobj(raw):
        raise ValueError(
            "Molecular solvent-site type indices must have shape "
            f"({solvent_site_count},)."
        )
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Molecular solvent-site type indices must be integers."
        ) from exc
    rounded = np.rint(numeric)
    if (
        not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, rounded)
        or np.any(rounded < 0.0)
        or np.any(rounded >= site_count)
    ):
        raise ValueError(
            "Molecular solvent-site type indices must be valid nonnegative integers."
        )
    result = rounded.astype(np.int64, copy=True)
    if np.any(np.bincount(result, minlength=site_count) == 0):
        raise ValueError(
            "Every HNC site type must map to at least one molecular solvent site."
        )
    result.setflags(write=False)
    return result


def _finite_positive_integer(value: object, *, name: str) -> int:
    """Validate one finite strictly positive integer."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Validate one finite nonnegative integer."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return integer


@dataclass(frozen=True)
class Route2V0MolecularSiteProjection:
    """A fixed molecular-configuration to periodic site-density map.

    ``site_occupancy_weights[a, g, i]`` is dimensionless.  Its spatial sum is
    the number of molecular sites of type ``a`` in configuration ``i``.  The
    configuration quadrature weights carry both the translation volume and the
    unnormalised orientation measure; the resulting site density has units of
    Bohr to the power minus three.
    """

    quadrature: Route2V0MolecularConfigurationQuadrature
    external_potential: Route2V0MolecularExternalPotentialContract
    site_hnc_asset: Route2V0SiteHNCAsset
    solvent_site_type_indices: np.ndarray
    site_occupancy_weights: np.ndarray
    construction: str = V0_MOLECULAR_SITE_HNC_CONSTRUCTION
    _site_multiplicity: np.ndarray = field(init=False, repr=False, compare=False)
    _molecular_bulk_number_density_bohr3: float = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, Route2V0MolecularConfigurationQuadrature):
            raise TypeError("Molecular site projection requires a quadrature.")
        if not isinstance(
            self.external_potential,
            Route2V0MolecularExternalPotentialContract,
        ):
            raise TypeError("Molecular site projection requires an external potential.")
        if not isinstance(self.site_hnc_asset, Route2V0SiteHNCAsset):
            raise TypeError("Molecular site projection requires a site-HNC asset.")
        if self.construction != V0_MOLECULAR_SITE_HNC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular site-HNC construction.")
        if not _same_configurations(
            self.quadrature.configurations,
            getattr(self.external_potential, "configurations", None),
        ):
            raise ValueError(
                "Molecular site projection quadrature and external-potential "
                "configurations differ."
            )
        external_grid = getattr(self.external_potential, "integration_grid", None)
        if not isinstance(external_grid, RegularCartesianGrid):
            raise TypeError(
                "Molecular site projection external potential requires a regular "
                "Cartesian integration grid."
            )
        if not _same_grid(external_grid, self.site_hnc_asset.grid):
            raise ValueError(
                "Molecular site projection requires one shared external-potential "
                "and site-HNC Cartesian grid."
            )

        site_count = self.site_hnc_asset.site_count
        solvent = getattr(self.external_potential, "solvent", None)
        if not isinstance(solvent, Route2V0MolecularSolventReference):
            raise TypeError(
                "Molecular site projection external potential requires a declared "
                "molecular solvent reference."
            )
        require_molecular_external_potential_values(
            self.external_potential,
            configuration_count=self.quadrature.configuration_count,
        )
        type_indices = _site_type_indices(
            self.solvent_site_type_indices,
            site_count=site_count,
            solvent_site_count=solvent.site_count,
        )
        multiplicity = np.bincount(type_indices, minlength=site_count).astype(
            np.int64,
            copy=True,
        )
        multiplicity.setflags(write=False)
        occupancy = _immutable_array(
            self.site_occupancy_weights,
            name="Molecular site occupancy weights",
            shape=(
                site_count,
                *self.site_hnc_asset.grid.shape,
                self.quadrature.configuration_count,
            ),
            nonnegative=True,
        )
        occupancy_flat = occupancy.reshape(
            (
                site_count,
                self.site_hnc_asset.grid.point_count,
                self.quadrature.configuration_count,
            )
        )
        occupancy_sum = np.sum(occupancy_flat, axis=1)
        scale = max(1.0, float(np.max(multiplicity)))
        if not np.allclose(
            occupancy_sum,
            multiplicity[:, None],
            rtol=0.0,
            atol=1.0e-12 * scale,
        ):
            raise ValueError(
                "Molecular site occupancy weights must sum to each declared site "
                "multiplicity for every configuration."
            )

        expected_measure = (
            self.site_hnc_asset.grid.volume_element_bohr3
            * self.site_hnc_asset.grid.point_count
            * RIGID_MOLECULAR_ORIENTATION_MEASURE
        )
        measure_scale = max(1.0, abs(expected_measure))
        if not math.isclose(
            self.quadrature.total_phase_space_measure_bohr3,
            expected_measure,
            rel_tol=0.0,
            abs_tol=1.0e-12 * measure_scale,
        ):
            raise ValueError(
                "Molecular site projection quadrature must cover exactly one periodic "
                "cell times the 8*pi**2 orientation measure."
            )

        site_bulk = self.site_hnc_asset.bulk_number_density_bohr3
        molecular_bulk = site_bulk / multiplicity
        molecular_scale = max(1.0, float(np.max(np.abs(molecular_bulk))))
        if not np.allclose(
            molecular_bulk,
            molecular_bulk[0],
            rtol=1.0e-12,
            atol=1.0e-14 * molecular_scale,
        ):
            raise ValueError(
                "Site-HNC bulk densities and molecular site multiplicities must imply "
                "one molecular bulk number density."
            )
        molecular_bulk_density = float(molecular_bulk[0])
        uniform_configuration_density = (
            molecular_bulk_density / RIGID_MOLECULAR_ORIENTATION_MEASURE
        )
        projected_bulk = (
            np.einsum(
                "agi,i,i->ag",
                occupancy_flat,
                self.quadrature.phase_space_weights_bohr3,
                np.full(
                    self.quadrature.configuration_count, uniform_configuration_density
                ),
                optimize=True,
            )
            / self.site_hnc_asset.grid.volume_element_bohr3
        ).reshape((site_count, *self.site_hnc_asset.grid.shape))
        expected_bulk = site_bulk.reshape((site_count, 1, 1, 1))
        bulk_scale = max(1.0, float(np.max(np.abs(expected_bulk))))
        if not np.allclose(
            projected_bulk,
            expected_bulk,
            rtol=1.0e-12,
            atol=1.0e-14 * bulk_scale,
        ):
            raise ValueError(
                "Molecular site occupancy weights must project the uniform molecular "
                "configuration density to the declared HNC bulk density."
            )

        object.__setattr__(self, "solvent_site_type_indices", type_indices)
        object.__setattr__(self, "site_occupancy_weights", occupancy)
        object.__setattr__(self, "_site_multiplicity", multiplicity)
        object.__setattr__(
            self,
            "_molecular_bulk_number_density_bohr3",
            molecular_bulk_density,
        )

    @property
    def site_multiplicity(self) -> np.ndarray:
        """Return the immutable number of each HNC site type per molecule."""

        return self._site_multiplicity

    @property
    def molecular_bulk_number_density_bohr3(self) -> float:
        """Return the one bulk molecular number density implied by the asset."""

        return self._molecular_bulk_number_density_bohr3

    @property
    def uniform_configuration_density_bohr3(self) -> float:
        """Return the bulk configuration density in the fixed orientation convention."""

        return (
            self.molecular_bulk_number_density_bohr3
            / RIGID_MOLECULAR_ORIENTATION_MEASURE
        )

    def _configuration_density(self, values: np.ndarray) -> np.ndarray:
        return _immutable_array(
            values,
            name="Molecular configuration density",
            shape=(self.quadrature.configuration_count,),
            positive=True,
        )

    def _site_field(self, values: np.ndarray, *, name: str) -> np.ndarray:
        return _immutable_array(
            values,
            name=name,
            shape=(self.site_hnc_asset.site_count, *self.site_hnc_asset.grid.shape),
        )

    def project_configuration_density(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Project a molecular configuration density to Cartesian site densities."""

        density = self._configuration_density(configuration_density_bohr3)
        occupancy = self.site_occupancy_weights.reshape(
            (
                self.site_hnc_asset.site_count,
                self.site_hnc_asset.grid.point_count,
                self.quadrature.configuration_count,
            )
        )
        projected = (
            np.einsum(
                "agi,i,i->ag",
                occupancy,
                self.quadrature.phase_space_weights_bohr3,
                density,
                optimize=True,
            )
            / self.site_hnc_asset.grid.volume_element_bohr3
        ).reshape((self.site_hnc_asset.site_count, *self.site_hnc_asset.grid.shape))
        result = np.array(projected, dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def site_field_adjoint_dimensionless(self, site_field: np.ndarray) -> np.ndarray:
        """Return the configuration-space adjoint in the quadrature pairing.

        The exact discrete identity is ``dv <v, P delta_n> =
        sum_i w_i (P^T v)_i delta_nu_i``.
        """

        field_values = self._site_field(site_field, name="Molecular site field")
        occupancy = self.site_occupancy_weights.reshape(
            (
                self.site_hnc_asset.site_count,
                self.site_hnc_asset.grid.point_count,
                self.quadrature.configuration_count,
            )
        )
        adjoint = np.einsum(
            "agi,ag->i",
            occupancy,
            field_values.reshape(
                (self.site_hnc_asset.site_count, self.site_hnc_asset.grid.point_count)
            ),
            optimize=True,
        )
        result = np.array(adjoint, dtype=float, copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class Route2V0MolecularSiteHNCState:
    """One auditable state of the projected molecular-site HNC scalar."""

    projection: Route2V0MolecularSiteProjection
    configuration_density_bohr3: np.ndarray
    grand_potential_hartree: float
    ideal_contribution_hartree: float
    excess_contribution_hartree: float
    external_contribution_hartree: float
    residual_inf: float
    iterations: int
    construction: str = V0_MOLECULAR_SITE_HNC_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.projection, Route2V0MolecularSiteProjection):
            raise TypeError("Molecular site-HNC state requires a projection.")
        if self.construction != V0_MOLECULAR_SITE_HNC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular site-HNC construction.")
        density = self.projection._configuration_density(
            self.configuration_density_bohr3
        )
        values = {
            "grand_potential_hartree": self.grand_potential_hartree,
            "ideal_contribution_hartree": self.ideal_contribution_hartree,
            "excess_contribution_hartree": self.excess_contribution_hartree,
            "external_contribution_hartree": self.external_contribution_hartree,
            "residual_inf": self.residual_inf,
        }
        for name, raw_value in values.items():
            value = float(raw_value)
            if not math.isfinite(value) or (name == "residual_inf" and value < 0.0):
                raise ValueError(f"Molecular site-HNC state {name} must be finite.")
            object.__setattr__(self, name, value)
        expected = (
            self.ideal_contribution_hartree
            + self.excess_contribution_hartree
            + self.external_contribution_hartree
        )
        tolerance = 1.0e-12 * max(1.0, abs(expected))
        if abs(self.grand_potential_hartree - expected) > tolerance:
            raise ValueError(
                "Molecular site-HNC grand potential must equal its three energy components."
            )
        iterations = _nonnegative_integer(
            self.iterations,
            name="Molecular site-HNC iteration count",
        )
        object.__setattr__(self, "configuration_density_bohr3", density)
        object.__setattr__(self, "iterations", iterations)


@dataclass(frozen=True)
class Route2V0MolecularSiteHNCFunctional:
    """One projected molecular HNC scalar with a numerical Picard control."""

    projection: Route2V0MolecularSiteProjection
    construction: str = V0_MOLECULAR_SITE_HNC_CONSTRUCTION
    _ideal_functional: Route2V0MolecularIdealGasFunctional = field(
        init=False,
        repr=False,
        compare=False,
    )
    _site_reference: Route2V0SiteHNCFunctional = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.projection, Route2V0MolecularSiteProjection):
            raise TypeError("Molecular site-HNC functional requires a projection.")
        if self.construction != V0_MOLECULAR_SITE_HNC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular site-HNC construction.")
        ideal = Route2V0MolecularIdealGasFunctional(
            quadrature=self.projection.quadrature,
            external_potential=self.projection.external_potential,
            bulk_molecular_number_density_bohr3=(
                self.projection.molecular_bulk_number_density_bohr3
            ),
            kbt_hartree=self.projection.site_hnc_asset.kbt_hartree,
        )
        site_reference = Route2V0SiteHNCFunctional(
            self.projection.site_hnc_asset,
            np.zeros(
                (
                    self.projection.site_hnc_asset.site_count,
                    *self.projection.site_hnc_asset.grid.shape,
                )
            ),
        )
        object.__setattr__(self, "_ideal_functional", ideal)
        object.__setattr__(self, "_site_reference", site_reference)

    def _density(self, values: np.ndarray) -> np.ndarray:
        return self.projection._configuration_density(values)

    def projected_site_density(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the declared projected Cartesian site density."""

        return self.projection.project_configuration_density(
            configuration_density_bohr3
        )

    def convolve_direct_correlation(
        self,
        site_density_delta_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the reciprocal site-HNC correlation convolution."""

        return self._site_reference.convolve_direct_correlation(
            site_density_delta_bohr3
        )

    def dimensionless_gradient(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return ``beta * delta Omega / delta nu`` at every configuration."""

        density = self._density(configuration_density_bohr3)
        projected = self.projected_site_density(density)
        bulk = self.projection.site_hnc_asset.bulk_number_density_bohr3.reshape(
            (self.projection.site_hnc_asset.site_count, 1, 1, 1)
        )
        correlation = self.convolve_direct_correlation(projected - bulk)
        gradient = np.array(
            self._ideal_functional.dimensionless_gradient(density),
            dtype=float,
            copy=True,
        )
        gradient -= self.projection.site_field_adjoint_dimensionless(correlation)
        result = np.array(gradient, dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def energy_components(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> tuple[float, float, float]:
        """Return ideal, projected-HNC-excess, and external scalar contributions."""

        density = self._density(configuration_density_bohr3)
        ideal, external = self._ideal_functional.energy_components(density)
        projected = self.projected_site_density(density)
        bulk = self.projection.site_hnc_asset.bulk_number_density_bohr3.reshape(
            (self.projection.site_hnc_asset.site_count, 1, 1, 1)
        )
        delta = projected - bulk
        correlation = self.convolve_direct_correlation(delta)
        excess = float(
            -0.5
            * self.projection.site_hnc_asset.kbt_hartree
            * self.projection.site_hnc_asset.grid.volume_element_bohr3
            * np.sum(delta * correlation)
        )
        if not all(math.isfinite(value) for value in (ideal, excess, external)):
            raise RuntimeError("Molecular site-HNC energy components are non-finite.")
        return ideal, excess, external

    def grand_potential_hartree(self, configuration_density_bohr3: np.ndarray) -> float:
        """Return the one scalar whose gradient defines molecular stationarity."""

        return float(sum(self.energy_components(configuration_density_bohr3)))

    def stationary_state(
        self,
        configuration_density_bohr3: np.ndarray,
        *,
        iterations: int,
    ) -> Route2V0MolecularSiteHNCState:
        """Build one auditable state without claiming convergence by itself."""

        iteration_count = _nonnegative_integer(
            iterations,
            name="Molecular site-HNC iteration count",
        )
        density = self._density(configuration_density_bohr3)
        ideal, excess, external = self.energy_components(density)
        return Route2V0MolecularSiteHNCState(
            projection=self.projection,
            configuration_density_bohr3=density,
            grand_potential_hartree=ideal + excess + external,
            ideal_contribution_hartree=ideal,
            excess_contribution_hartree=excess,
            external_contribution_hartree=external,
            residual_inf=float(np.max(np.abs(self.dimensionless_gradient(density)))),
            iterations=iteration_count,
        )

    def solve_picard(
        self,
        *,
        initial_configuration_density_bohr3: np.ndarray | None = None,
        residual_tolerance: float = 1.0e-10,
        picard_mixing: float = 0.2,
        max_iterations: int = 1000,
    ) -> Route2V0MolecularSiteHNCState:
        """Solve un-mixed scalar stationarity with numerical log-density mixing.

        ``picard_mixing`` affects only the numerical iteration; it is not a
        solvent-model coefficient and callers must not select it from target
        solvation errors.
        """

        tolerance = float(residual_tolerance)
        mixing = float(picard_mixing)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "Molecular site-HNC residual tolerance must be finite and positive."
            )
        if not math.isfinite(mixing) or not 0.0 < mixing <= 1.0:
            raise ValueError("Molecular site-HNC Picard mixing must be in (0, 1].")
        maximum = _finite_positive_integer(
            max_iterations,
            name="Molecular site-HNC maximum iterations",
        )
        bulk = self.projection.uniform_configuration_density_bohr3
        if initial_configuration_density_bohr3 is None:
            density = np.full(self.projection.quadrature.configuration_count, bulk)
        else:
            density = np.array(
                self._density(initial_configuration_density_bohr3),
                dtype=float,
                copy=True,
            )

        for iteration in range(maximum + 1):
            state = self.stationary_state(density, iterations=iteration)
            if state.residual_inf <= tolerance:
                return state
            if iteration == maximum:
                break
            log_ratio = np.log(density / bulk)
            log_ratio -= mixing * self.dimensionless_gradient(density)
            if not np.all(np.isfinite(log_ratio)):
                raise RuntimeError(
                    "Molecular site-HNC Picard iteration produced a non-finite log density."
                )
            density = bulk * np.exp(log_ratio)
            if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
                raise RuntimeError(
                    "Molecular site-HNC Picard iteration produced an invalid density."
                )

        raise RuntimeError(
            "Molecular site-HNC Picard solver did not meet the declared stationarity "
            f"tolerance within {maximum} iterations."
        )


__all__ = [
    "Route2V0MolecularSiteHNCFunctional",
    "Route2V0MolecularSiteHNCState",
    "Route2V0MolecularSiteProjection",
    "V0_MOLECULAR_SITE_HNC_CONSTRUCTION",
]
