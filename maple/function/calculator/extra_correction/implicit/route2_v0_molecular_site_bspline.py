"""Matrix-free molecular-site B-spline map for the Route-2 V0 HNC scalar.

The molecular configuration density has one degree of freedom per Cartesian
translation and rigid orientation.  Materialising its site-occupancy tensor
would require ``O(n_site_types * n_grid * n_configurations)`` storage.  For a
Cartesian--SO(3) product rule, ``n_configurations = n_grid * n_orientations``;
that quadratic grid scaling prevents the existing common scalar from reaching
a grid-convergence calculation.

This module stores instead one periodic cardinal cubic B-spline stencil for
each solvent atom at each molecular configuration.  It implements the same
linear map and its exact field adjoint in ``O(64 * n_solvent_sites *
n_configurations)`` storage and work.  It introduces no physical length,
fit, response filter, or change to the HNC scalar: the cardinal B-spline,
full Cartesian--SO(3) product measure, and source-bound rigid geometry remain
identical to the dense reference representation.

The compact object is a numerical representation only.  It does not by
itself make a liquid source physical, prove convergence, supply a total
solvation free energy, or establish an accuracy result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import SupportsFloat, SupportsIndex, cast

import numpy as np

from .route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
)
from .route2_v0_molecular_so3_quadrature import (
    Route2V0CartesianEulerProductQuadrature,
)
from .route2_v0_periodic_bspline import (
    V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE,
    Route2V0PeriodicCubicBSplineStencil,
    build_route2_v0_periodic_cubic_bspline_stencil,
)
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MOLECULAR_SITE_CUBIC_BSPLINE_DEPOSITION_CONSTRUCTION = (
    "route2-v0-molecular-site-cubic-bspline-deposition-v1"
)


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    positive: bool = False,
) -> np.ndarray:
    """Return one finite immutable real array with the declared shape."""

    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite with shape {shape}.") from exc
    if (
        array.shape != shape
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
    ):
        requirement = "finite"
        if positive:
            requirement += " and strictly positive"
        raise ValueError(f"{name} must be {requirement} with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one positive integer without accepting boolean values."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer.")
        result = int(numeric)
    else:
        raise TypeError(f"{name} must be a positive integer.")
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def validate_route2_v0_molecular_site_type_indices(
    values: np.ndarray,
    *,
    site_count: int,
    solvent_site_count: int,
) -> np.ndarray:
    """Validate the complete rigid-solvent-site to HNC-site map."""

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


def _positive_scalar(value: object, *, name: str) -> float:
    """Return one finite positive scalar."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite and positive.")
    try:
        result = float(
            cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True)
class Route2V0MolecularSiteCubicBSplineDeposition:
    """Compact exact site-deposition map for one rigid molecular grid.

    The object owns no thermodynamic quadrature weights.  That keeps the
    geometric map independent of the particular scalar term while allowing a
    caller to apply the one declared configuration-space measure exactly once.
    ``project_configuration_values`` returns the molecular-to-site density map
    and ``site_field_adjoint_dimensionless`` returns its matched field adjoint.
    """

    grid: RegularCartesianGrid
    configurations: Route2V0MolecularConfigurations
    solvent: Route2V0MolecularSolventReference
    solvent_site_type_indices: np.ndarray
    site_count: int
    construction: str = V0_MOLECULAR_SITE_CUBIC_BSPLINE_DEPOSITION_CONSTRUCTION
    _site_stencils: tuple[Route2V0PeriodicCubicBSplineStencil, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _site_multiplicity: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError(
                "Molecular cubic B-spline deposition requires a regular Cartesian grid."
            )
        if not isinstance(self.configurations, Route2V0MolecularConfigurations):
            raise TypeError(
                "Molecular cubic B-spline deposition requires rigid molecular "
                "configurations."
            )
        if not isinstance(self.solvent, Route2V0MolecularSolventReference):
            raise TypeError(
                "Molecular cubic B-spline deposition requires a solvent reference."
            )
        if self.construction != V0_MOLECULAR_SITE_CUBIC_BSPLINE_DEPOSITION_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 V0 molecular cubic B-spline deposition."
            )
        count = _positive_integer(self.site_count, name="Molecular HNC site count")
        type_indices = validate_route2_v0_molecular_site_type_indices(
            self.solvent_site_type_indices,
            site_count=count,
            solvent_site_count=self.solvent.site_count,
        )
        positions = self.configurations.site_positions_bohr(self.solvent)
        stencils = tuple(
            build_route2_v0_periodic_cubic_bspline_stencil(
                grid=self.grid,
                positions_bohr=positions[:, molecule_site, :],
            )
            for molecule_site in range(self.solvent.site_count)
        )
        if len(stencils) != self.solvent.site_count:
            raise RuntimeError(
                "Molecular cubic B-spline site stencil count is invalid."
            )
        multiplicity = np.bincount(type_indices, minlength=count).astype(
            np.int64,
            copy=True,
        )
        multiplicity.setflags(write=False)
        object.__setattr__(self, "solvent_site_type_indices", type_indices)
        object.__setattr__(self, "site_count", count)
        object.__setattr__(self, "_site_stencils", stencils)
        object.__setattr__(self, "_site_multiplicity", multiplicity)

    @property
    def configuration_count(self) -> int:
        """Return the rigid-configuration count of the compact map."""

        return self.configurations.configuration_count

    @property
    def site_multiplicity(self) -> np.ndarray:
        """Return the immutable molecular multiplicity of every HNC site type."""

        return self._site_multiplicity

    @property
    def storage_entry_count(self) -> int:
        """Return the compact B-spline coefficient count before expansion.

        This counts the 64 floating weights per molecular site and
        configuration.  :attr:`storage_byte_count` additionally includes the
        integer node indices needed by the sparse representation.
        """

        return (
            self.solvent.site_count
            * self.configuration_count
            * V0_PERIODIC_CUBIC_BSPLINE_STENCIL_SIZE
        )

    @property
    def storage_byte_count(self) -> int:
        """Return bytes held by the compact stencil arrays.

        The value deliberately includes the stored deposited positions,
        floating weights, and integer node indices.  It excludes the immutable
        configuration grid and solvent source geometry shared with every
        representation.
        """

        return sum(
            stencil.positions_bohr.nbytes
            + stencil.node_linear_indices.nbytes
            + stencil.weights.nbytes
            for stencil in self._site_stencils
        )

    @property
    def dense_reference_byte_count(self) -> int:
        """Return bytes required by the corresponding dense float64 tensor."""

        return (
            self.site_count
            * self.grid.point_count
            * self.configuration_count
            * np.dtype(float).itemsize
        )

    @property
    def site_stencils(self) -> tuple[Route2V0PeriodicCubicBSplineStencil, ...]:
        """Return immutable per-solvent-site stencils for diagnostic inspection."""

        return self._site_stencils

    def _configuration_values(self, values: np.ndarray, *, name: str) -> np.ndarray:
        return _immutable_array(
            values,
            name=name,
            shape=(self.configuration_count,),
        )

    def _phase_space_weights(self, values: np.ndarray) -> np.ndarray:
        return _immutable_array(
            values,
            name="Molecular configuration phase-space weights",
            shape=(self.configuration_count,),
            positive=True,
        )

    def _site_field(self, values: np.ndarray) -> np.ndarray:
        return _immutable_array(
            values,
            name="Molecular cubic B-spline site field",
            shape=(self.site_count, *self.grid.shape),
        )

    def project_configuration_values(
        self,
        configuration_values: np.ndarray,
        *,
        phase_space_weights_bohr3: np.ndarray,
        volume_element_bohr3: float,
    ) -> np.ndarray:
        """Apply the compact map with the declared configuration measure.

        This returns

        ``n_a(g) = dv^-1 sum_i w_i sum_(s in a) D_(g,i,s) value_i``.

        Values may be signed, so the same exact linear action is valid for a
        density, a tangent direction, and a finite-difference diagnostic.
        """

        values = self._configuration_values(
            configuration_values,
            name="Molecular configuration values",
        )
        weights = self._phase_space_weights(phase_space_weights_bohr3)
        volume = _positive_scalar(
            volume_element_bohr3,
            name="Molecular cubic B-spline volume element",
        )
        return self._project_configuration_values_validated(
            values,
            phase_space_weights_bohr3=weights,
            volume_element_bohr3=volume,
        )

    def _project_configuration_values_validated(
        self,
        configuration_values: np.ndarray,
        *,
        phase_space_weights_bohr3: np.ndarray,
        volume_element_bohr3: float,
    ) -> np.ndarray:
        """Apply the map after the caller has validated its scalar convention.

        This private path avoids duplicate full-vector copies when the owning
        molecular HNC projection has already validated its density, quadrature
        weights, and grid volume.  It does not alter the public validation
        boundary above.
        """

        scaled_values = (
            phase_space_weights_bohr3 * configuration_values / volume_element_bohr3
        )
        result = np.zeros((self.site_count, self.grid.point_count), dtype=float)
        for molecule_site, stencil in enumerate(self._site_stencils):
            result[self.solvent_site_type_indices[molecule_site]] += (
                stencil._deposit_validated(scaled_values).reshape(self.grid.point_count)
            )
        projected = result.reshape((self.site_count, *self.grid.shape))
        projected.setflags(write=False)
        return projected

    def site_field_adjoint_dimensionless(self, site_field: np.ndarray) -> np.ndarray:
        """Return the exact field adjoint without materializing occupancy.

        In the declared pairing,

        ``dv <v, P d> = sum_i w_i (P.T v)_i d_i``.
        """

        field_values = self._site_field(site_field)
        return self._site_field_adjoint_dimensionless_validated(field_values)

    def _site_field_adjoint_dimensionless_validated(
        self,
        site_field: np.ndarray,
    ) -> np.ndarray:
        """Apply the field adjoint after the field shape has been validated."""

        result = np.zeros(self.configuration_count, dtype=float)
        for molecule_site, stencil in enumerate(self._site_stencils):
            result += stencil._field_adjoint_validated(
                site_field[self.solvent_site_type_indices[molecule_site]].reshape(
                    self.grid.point_count
                )
            )
        result.setflags(write=False)
        return result

    def dense_occupancy_weights_reference(self) -> np.ndarray:
        """Materialize the exact dense control tensor only on caller request.

        This method is retained for dense-reference equivalence tests and for
        existing tiny-grid controls.  Production paths should call the compact
        forward/adjoint actions instead.
        """

        result = np.zeros(
            (self.site_count, *self.grid.shape, self.configuration_count),
            dtype=float,
        )
        for molecule_site, stencil in enumerate(self._site_stencils):
            result[self.solvent_site_type_indices[molecule_site]] += (
                stencil.dense_occupancy_weights()
            )
        result.setflags(write=False)
        return result


def build_route2_v0_cartesian_euler_cubic_bspline_site_deposition(
    *,
    cartesian_euler_quadrature: Route2V0CartesianEulerProductQuadrature,
    solvent: Route2V0MolecularSolventReference,
    solvent_site_type_indices: np.ndarray,
    site_count: int,
) -> Route2V0MolecularSiteCubicBSplineDeposition:
    """Build the compact standard Cartesian--Euler molecular-site map."""

    if not isinstance(
        cartesian_euler_quadrature,
        Route2V0CartesianEulerProductQuadrature,
    ):
        raise TypeError(
            "Molecular cubic B-spline deposition requires a Cartesian Euler "
            "product quadrature."
        )
    return Route2V0MolecularSiteCubicBSplineDeposition(
        grid=cartesian_euler_quadrature.grid,
        configurations=cartesian_euler_quadrature.configurations,
        solvent=solvent,
        solvent_site_type_indices=solvent_site_type_indices,
        site_count=site_count,
    )


__all__ = [
    "V0_MOLECULAR_SITE_CUBIC_BSPLINE_DEPOSITION_CONSTRUCTION",
    "Route2V0MolecularSiteCubicBSplineDeposition",
    "build_route2_v0_cartesian_euler_cubic_bspline_site_deposition",
    "validate_route2_v0_molecular_site_type_indices",
]
