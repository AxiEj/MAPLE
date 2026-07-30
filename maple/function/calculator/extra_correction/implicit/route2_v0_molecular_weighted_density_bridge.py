"""Pure-solvent anchored weighted-density bridge for the Route-2 V0 HNC scalar.

The molecular HNC control in :mod:`route2_v0_molecular_site_hnc` is a genuine
scalar, but its quadratic excess functional has the familiar HNC cavity and
pressure defect.  This module adds a *pre-minimisation* bridge functional to
that same scalar; it is not a posteriori PC/PC+ correction and it does not
touch the frozen MACE checkpoint.

For one configuration density ``nu_i``, a separately declared centre map
``D`` produces a molecular number density ``rho = D nu``.  A reciprocal
periodic kernel ``K`` then forms ``rho_bar = K rho``.  The bridge is

``F_bridge = dv sum_g [ A (rho_bar-rho_b)^3
                        + B rho_bar^2 (rho_bar-rho_b)^4 ]``.

``A`` is fixed algebraically by the *same* molecular-HNC vacuum-limit pressure
and an independently frozen bulk pressure.  ``B`` and ``K`` must be frozen
from a pure-solvent planar-interface/surface-tension and bulk-correlation
certificate before target solvation data are read.  Therefore the bridge can
improve the physical cavity functional without becoming a solvation-label fit.

The module deliberately remains an admission-gated control: a synthetic
kernel or a certificate-shaped value is not a physical solvent asset, a total
solvation result, a force/PES result, or an accuracy claim.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .route2_v0_mace_cluster_rism_bridge import (
    verify_route2_v0_asset_bound_rism_kernel,
)
from .route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
)
from .route2_v0_molecular_thermodynamics import (
    molecular_hnc_bulk_functional_pressure_hartree_per_bohr3,
)
from .route2_v0_pure_solvent_bridge_certificate import (
    Route2V0PureSolventBridgeCertificate,
    weighted_density_operator_sha256,
)
from .route2_v0_rism_energy_conjugate import Route2V0RismEnergyConjugateKernel
from .route2_v0_solvent_asset import Route2V0FrozenSolventAsset
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MOLECULAR_CENTER_PROJECTION_CONSTRUCTION = "route2-v0-molecular-center-projection-v1"
V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION = (
    "route2-v0-molecular-weighted-density-bridge-v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _finite(value: object, *, name: str) -> float:
    """Return one finite real scalar without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite.")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: object, *, name: str) -> float:
    """Return one finite positive scalar."""

    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative(value: object, *, name: str) -> float:
    """Return one finite nonnegative scalar."""

    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return one nonnegative integer without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a nonnegative integer.")
    try:
        integer = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return integer


def _positive_integer(value: object, *, name: str) -> int:
    """Return one positive integer without accepting booleans."""

    integer = _nonnegative_integer(value, name=name)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _same_grid(left: RegularCartesianGrid, right: RegularCartesianGrid) -> bool:
    """Return whether two grids carry one exact discrete convention."""

    return bool(
        left.shape == right.shape
        and left.layout == right.layout
        and np.array_equal(left.origin_bohr, right.origin_bohr)
        and np.array_equal(left.spacing_bohr, right.spacing_bohr)
    )


def _configuration_density(
    projection: Route2V0MolecularSiteProjection,
    values: np.ndarray,
    *,
    name: str,
    positive: bool,
) -> np.ndarray:
    """Validate one configuration-density vector in the shared quadrature."""

    return _immutable_array(
        values,
        name=name,
        shape=(projection.quadrature.configuration_count,),
        positive=positive,
    )


def _grid_field(
    grid: RegularCartesianGrid,
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate one real periodic Cartesian field."""

    return _immutable_array(values, name=name, shape=grid.shape)


def _sha256(value: object, *, name: str) -> str:
    """Validate a lower-case content digest rather than a free-form label."""

    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be one lower-case SHA256 digest.")
    return value


def _periodic_inverse(values: np.ndarray) -> np.ndarray:
    """Return a grid field evaluated at ``-r`` in the same periodic layout."""

    indices = tuple((-np.arange(length)) % length for length in values.shape)
    return values[np.ix_(*indices)]


@dataclass(frozen=True)
class Route2V0MolecularCenterProjection:
    """The exact molecular-centre counterpart of the site-density projection.

    ``center_occupancy_weights[g, i]`` is a dimensionless periodic deposition
    for configuration ``i``.  Its spatial sum is exactly one, and the fixed
    quadrature must map a uniform configuration density to the declared bulk
    molecular density.  The explicit map prevents a site-density proxy from
    being silently used as the weighted density of molecular centres.
    """

    projection: Route2V0MolecularSiteProjection
    center_occupancy_weights: np.ndarray
    construction: str = V0_MOLECULAR_CENTER_PROJECTION_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.projection, Route2V0MolecularSiteProjection):
            raise TypeError("Molecular centre projection requires a site projection.")
        if self.construction != V0_MOLECULAR_CENTER_PROJECTION_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular centre projection.")
        grid = self.projection.site_hnc_asset.grid
        occupancy = _immutable_array(
            self.center_occupancy_weights,
            name="Molecular centre occupancy weights",
            shape=(*grid.shape, self.projection.quadrature.configuration_count),
            nonnegative=True,
        )
        flat = occupancy.reshape(
            (grid.point_count, self.projection.quadrature.configuration_count)
        )
        if not np.allclose(
            np.sum(flat, axis=0),
            1.0,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError(
                "Molecular centre occupancy weights must sum to one for every "
                "configuration."
            )
        uniform = np.full(
            self.projection.quadrature.configuration_count,
            self.projection.uniform_configuration_density_bohr3,
        )
        projected_bulk = self._project(uniform, positive=True)
        expected_bulk = self.projection.molecular_bulk_number_density_bohr3
        scale = max(1.0, abs(expected_bulk))
        if not np.allclose(
            projected_bulk,
            expected_bulk,
            rtol=1.0e-12,
            atol=1.0e-14 * scale,
        ):
            raise ValueError(
                "Molecular centre occupancy weights must project the uniform "
                "configuration density to the declared molecular bulk density."
            )
        object.__setattr__(self, "center_occupancy_weights", occupancy)

    @property
    def grid(self) -> RegularCartesianGrid:
        """Return the exact Cartesian grid shared with the liquid scalar."""

        return self.projection.site_hnc_asset.grid

    @property
    def molecular_bulk_number_density_bohr3(self) -> float:
        """Return the molecular bulk density fixed by the source asset."""

        return self.projection.molecular_bulk_number_density_bohr3

    def _project(self, values: np.ndarray, *, positive: bool) -> np.ndarray:
        density = _configuration_density(
            self.projection,
            values,
            name="Molecular configuration density",
            positive=positive,
        )
        occupancy = self.center_occupancy_weights.reshape(
            (self.grid.point_count, self.projection.quadrature.configuration_count)
        )
        projected = (
            np.einsum(
                "gi,i,i->g",
                occupancy,
                self.projection.quadrature.phase_space_weights_bohr3,
                density,
                optimize=True,
            )
            / self.grid.volume_element_bohr3
        ).reshape(self.grid.shape)
        result = np.array(projected, dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def project_configuration_density(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Project one positive configuration density to molecular centres."""

        return self._project(configuration_density_bohr3, positive=True)

    def project_configuration_direction(
        self,
        configuration_density_direction_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Project one signed configuration-density tangent direction."""

        return self._project(configuration_density_direction_bohr3, positive=False)

    def center_field_adjoint_dimensionless(
        self, center_field: np.ndarray
    ) -> np.ndarray:
        """Return the exact quadrature adjoint of the centre projection.

        The discrete identity is ``dv <q, D d> = sum_i w_i (D^T q)_i d_i``.
        """

        field = _grid_field(
            self.grid,
            center_field,
            name="Molecular centre field",
        )
        occupancy = self.center_occupancy_weights.reshape(
            (self.grid.point_count, self.projection.quadrature.configuration_count)
        )
        adjoint = np.einsum(
            "gi,g->i",
            occupancy,
            field.reshape(self.grid.point_count),
            optimize=True,
        )
        result = np.array(adjoint, dtype=float, copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class Route2V0PeriodicWeightedDensityKernel:
    """One normalized reciprocal periodic kernel for coarse-grained density."""

    grid: RegularCartesianGrid
    kernel_bohr_minus3: np.ndarray
    construction: str = V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError("Weighted-density kernel requires a Cartesian grid.")
        if self.construction != V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 weighted-density kernel.")
        kernel = _immutable_array(
            self.kernel_bohr_minus3,
            name="Weighted-density kernel",
            shape=self.grid.shape,
            nonnegative=True,
        )
        normalization = float(self.grid.volume_element_bohr3 * np.sum(kernel))
        if not math.isfinite(normalization) or abs(normalization - 1.0) > 1.0e-12:
            raise ValueError(
                "Weighted-density kernel must have the exact discrete integral one."
            )
        scale = max(1.0, float(np.max(np.abs(kernel))))
        if not np.allclose(
            kernel,
            _periodic_inverse(kernel),
            rtol=1.0e-12,
            atol=1.0e-14 * scale,
        ):
            raise ValueError(
                "Weighted-density kernel must be reciprocal under periodic inversion."
            )
        object.__setattr__(self, "kernel_bohr_minus3", kernel)

    def convolve(self, field: np.ndarray) -> np.ndarray:
        """Return ``dv * K * field`` with the declared periodic FFT pairing."""

        values = _grid_field(self.grid, field, name="Weighted-density field")
        transformed = np.fft.ifftn(
            np.fft.fftn(self.kernel_bohr_minus3) * np.fft.fftn(values)
        )
        result = self.grid.volume_element_bohr3 * np.real(transformed)
        if not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Weighted-density convolution produced a non-finite field."
            )
        immutable = np.array(result, dtype=float, copy=True)
        immutable.setflags(write=False)
        return immutable


@dataclass(frozen=True)
class Route2V0MolecularWeightedDensityBridgeAsset:
    """Frozen pure-solvent information for a no-solvation-fit bridge scalar.

    ``cubic_coefficient_hartree_bohr6`` is not a tunable parameter.  It must
    equal ``(P_HNC - P_target) / rho_bulk**3`` for the source-bound molecular
    HNC scalar.  The positive quartic coefficient and kernel are admitted only
    with a content-addressed pure-solvent surface-tension/correlation
    certificate; the bridge module does not accept a solvation-label source.
    """

    center_projection: Route2V0MolecularCenterProjection
    kernel: Route2V0PeriodicWeightedDensityKernel
    hnc_bulk_pressure_hartree_per_bohr3: float
    target_bulk_pressure_hartree_per_bohr3: float
    cubic_coefficient_hartree_bohr6: float
    quartic_coefficient_hartree_bohr15: float
    target_surface_tension_hartree_per_bohr2: float
    pure_solvent_certificate_sha256: str
    pure_solvent_certificate: Route2V0PureSolventBridgeCertificate | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    construction: str = V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.center_projection, Route2V0MolecularCenterProjection):
            raise TypeError(
                "Weighted-density bridge asset requires a centre projection."
            )
        if not isinstance(self.kernel, Route2V0PeriodicWeightedDensityKernel):
            raise TypeError("Weighted-density bridge asset requires a periodic kernel.")
        if self.construction != V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 weighted-density bridge asset.")
        if not _same_grid(self.center_projection.grid, self.kernel.grid):
            raise ValueError(
                "Weighted-density bridge requires one shared centre and kernel grid."
            )
        hnc_pressure = _positive(
            self.hnc_bulk_pressure_hartree_per_bohr3,
            name="Molecular HNC bulk pressure",
        )
        target_pressure = _nonnegative(
            self.target_bulk_pressure_hartree_per_bohr3,
            name="Target pure-solvent bulk pressure",
        )
        if target_pressure >= hnc_pressure:
            raise ValueError(
                "Target pure-solvent pressure must be below the source HNC pressure "
                "for a stabilizing coexistence bridge."
            )
        cubic = _positive(
            self.cubic_coefficient_hartree_bohr6,
            name="Weighted-density cubic coefficient",
        )
        quartic = _positive(
            self.quartic_coefficient_hartree_bohr15,
            name="Weighted-density quartic coefficient",
        )
        surface_tension = _positive(
            self.target_surface_tension_hartree_per_bohr2,
            name="Target pure-solvent surface tension",
        )
        density = self.center_projection.molecular_bulk_number_density_bohr3
        expected_cubic = (hnc_pressure - target_pressure) / density**3
        tolerance = 1.0e-12 * max(1.0, abs(expected_cubic), abs(cubic))
        if abs(cubic - expected_cubic) > tolerance:
            raise ValueError(
                "Weighted-density cubic coefficient must be derived from the exact "
                "source HNC and target pure-solvent pressures."
            )
        certificate = self.pure_solvent_certificate
        if certificate is not None:
            if not isinstance(certificate, Route2V0PureSolventBridgeCertificate):
                raise TypeError(
                    "Weighted-density bridge source binding requires a parsed "
                    "pure-solvent certificate."
                )
            certificate_values = (
                (
                    "HNC pressure",
                    hnc_pressure,
                    certificate.hnc_bulk_pressure_hartree_per_bohr3,
                ),
                (
                    "target pressure",
                    target_pressure,
                    certificate.target_bulk_pressure_hartree_per_bohr3,
                ),
                (
                    "cubic coefficient",
                    cubic,
                    certificate.cubic_coefficient_hartree_bohr6,
                ),
                (
                    "quartic coefficient",
                    quartic,
                    certificate.quartic_coefficient_hartree_bohr15,
                ),
                (
                    "surface tension",
                    surface_tension,
                    certificate.target_surface_tension_hartree_per_bohr2,
                ),
            )
            for name, actual, certified in certificate_values:
                if not math.isclose(
                    actual,
                    certified,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-14 * max(1.0, abs(actual), abs(certified)),
                ):
                    raise ValueError(
                        "Weighted-density bridge source binding does not match "
                        f"its certificate {name}."
                    )
            if self.pure_solvent_certificate_sha256 != certificate.content_sha256:
                raise ValueError(
                    "Weighted-density bridge certificate digest does not match "
                    "its parsed source binding."
                )
        object.__setattr__(self, "hnc_bulk_pressure_hartree_per_bohr3", hnc_pressure)
        object.__setattr__(
            self, "target_bulk_pressure_hartree_per_bohr3", target_pressure
        )
        object.__setattr__(self, "cubic_coefficient_hartree_bohr6", cubic)
        object.__setattr__(self, "quartic_coefficient_hartree_bohr15", quartic)
        object.__setattr__(
            self, "target_surface_tension_hartree_per_bohr2", surface_tension
        )
        object.__setattr__(
            self,
            "pure_solvent_certificate_sha256",
            _sha256(
                self.pure_solvent_certificate_sha256,
                name="Pure-solvent bridge certificate",
            ),
        )
        object.__setattr__(self, "pure_solvent_certificate", certificate)

    @classmethod
    def from_pure_solvent_anchors(
        cls,
        *,
        center_projection: Route2V0MolecularCenterProjection,
        kernel: Route2V0PeriodicWeightedDensityKernel,
        hnc_bulk_pressure_hartree_per_bohr3: float,
        target_bulk_pressure_hartree_per_bohr3: float,
        quartic_coefficient_hartree_bohr15: float,
        target_surface_tension_hartree_per_bohr2: float,
        pure_solvent_certificate_sha256: str,
    ) -> Route2V0MolecularWeightedDensityBridgeAsset:
        """Construct an asset while deriving its cubic coefficient exactly."""

        hnc_pressure = _positive(
            hnc_bulk_pressure_hartree_per_bohr3,
            name="Molecular HNC bulk pressure",
        )
        target_pressure = _nonnegative(
            target_bulk_pressure_hartree_per_bohr3,
            name="Target pure-solvent bulk pressure",
        )
        density = center_projection.molecular_bulk_number_density_bohr3
        return cls(
            center_projection=center_projection,
            kernel=kernel,
            hnc_bulk_pressure_hartree_per_bohr3=hnc_pressure,
            target_bulk_pressure_hartree_per_bohr3=target_pressure,
            cubic_coefficient_hartree_bohr6=(hnc_pressure - target_pressure)
            / density**3,
            quartic_coefficient_hartree_bohr15=quartic_coefficient_hartree_bohr15,
            target_surface_tension_hartree_per_bohr2=(
                target_surface_tension_hartree_per_bohr2
            ),
            pure_solvent_certificate_sha256=pure_solvent_certificate_sha256,
        )

    @classmethod
    def from_source_bound_pure_solvent_certificate(
        cls,
        *,
        hnc_functional: Route2V0MolecularSiteHNCFunctional,
        frozen_solvent_asset: Route2V0FrozenSolventAsset,
        rism_kernel: Route2V0RismEnergyConjugateKernel,
        center_projection: Route2V0MolecularCenterProjection,
        kernel: Route2V0PeriodicWeightedDensityKernel,
        certificate: Route2V0PureSolventBridgeCertificate,
    ) -> Route2V0MolecularWeightedDensityBridgeAsset:
        """Build the only source-bound physical-admission bridge asset.

        The legacy direct constructor remains useful for synthetic scalar and
        derivative controls.  It cannot constitute a physical liquid asset:
        this constructor instead binds a loaded certificate to the exact HNC
        projection, source-locked RISM kernel, centre map, and weighted kernel
        that will enter the common scalar.  No solute result, cavity error, or
        user-selected bridge coefficient participates in this operation.
        """

        if not isinstance(hnc_functional, Route2V0MolecularSiteHNCFunctional):
            raise TypeError(
                "Source-bound weighted-density bridge requires a molecular HNC "
                "functional."
            )
        if not isinstance(center_projection, Route2V0MolecularCenterProjection):
            raise TypeError(
                "Source-bound weighted-density bridge requires a centre projection."
            )
        if not isinstance(kernel, Route2V0PeriodicWeightedDensityKernel):
            raise TypeError(
                "Source-bound weighted-density bridge requires a periodic kernel."
            )
        if not isinstance(certificate, Route2V0PureSolventBridgeCertificate):
            raise TypeError(
                "Source-bound weighted-density bridge requires a parsed "
                "pure-solvent certificate."
            )
        certificate.verify_content_integrity()
        if center_projection.projection is not hnc_functional.projection:
            raise ValueError(
                "Source-bound weighted-density bridge must use the exact "
                "molecular HNC projection."
            )
        verify_route2_v0_asset_bound_rism_kernel(
            frozen_solvent_asset=frozen_solvent_asset,
            rism_kernel=rism_kernel,
        )
        if hnc_functional.projection.site_hnc_asset is not rism_kernel.site_hnc_asset:
            raise ValueError(
                "Source-bound weighted-density bridge HNC functional does not "
                "use the exact frozen RISM kernel asset."
            )
        hnc_pressure = molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(
            hnc_functional
        )
        center_digest = weighted_density_operator_sha256(
            operator="molecular-centre-projection",
            construction=center_projection.construction,
            grid=center_projection.grid,
            values=center_projection.center_occupancy_weights,
        )
        kernel_digest = weighted_density_operator_sha256(
            operator="weighted-density-kernel",
            construction=kernel.construction,
            grid=kernel.grid,
            values=kernel.kernel_bohr_minus3,
        )
        source_hashes = {
            role: frozen_solvent_asset.file_for(role).sha256
            for role in certificate.source_hashes_by_role
        }
        certificate.verify_bound_inputs(
            solvent_id=frozen_solvent_asset.solvent_id,
            model_identifier=frozen_solvent_asset.model_identifier,
            closure=frozen_solvent_asset.closure,
            temperature_kelvin=frozen_solvent_asset.temperature_kelvin,
            pressure_bar=frozen_solvent_asset.pressure_bar,
            source_file_sha256=source_hashes,
            hnc_bulk_pressure_hartree_per_bohr3=hnc_pressure,
            molecular_bulk_number_density_bohr3=(
                center_projection.molecular_bulk_number_density_bohr3
            ),
            center_projection_sha256=center_digest,
            weighted_density_kernel_sha256=kernel_digest,
        )
        return cls(
            center_projection=center_projection,
            kernel=kernel,
            hnc_bulk_pressure_hartree_per_bohr3=hnc_pressure,
            target_bulk_pressure_hartree_per_bohr3=(
                certificate.target_bulk_pressure_hartree_per_bohr3
            ),
            cubic_coefficient_hartree_bohr6=(
                certificate.cubic_coefficient_hartree_bohr6
            ),
            quartic_coefficient_hartree_bohr15=(
                certificate.quartic_coefficient_hartree_bohr15
            ),
            target_surface_tension_hartree_per_bohr2=(
                certificate.target_surface_tension_hartree_per_bohr2
            ),
            pure_solvent_certificate_sha256=certificate.content_sha256,
            pure_solvent_certificate=certificate,
        )

    @property
    def molecular_bulk_number_density_bohr3(self) -> float:
        """Return the exact bulk density used in the coexistence construction."""

        return self.center_projection.molecular_bulk_number_density_bohr3

    @property
    def is_source_bound_pure_solvent_asset(self) -> bool:
        """Return whether this asset carries a parsed, matching source certificate."""

        return self.pure_solvent_certificate is not None

    def require_source_bound_pure_solvent_asset(self) -> None:
        """Reject a synthetic bridge before a physical-liquid calculation."""

        if not self.is_source_bound_pure_solvent_asset:
            raise ValueError(
                "Physical weighted-density bridge use requires a source-bound "
                "pure-solvent certificate."
            )

    def free_energy_density_hartree_per_bohr3(
        self,
        weighted_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the local bridge free-energy density on the fixed grid."""

        density = _grid_field(
            self.center_projection.grid,
            weighted_density_bohr3,
            name="Weighted molecular density",
        )
        delta = density - self.molecular_bulk_number_density_bohr3
        result = (
            self.cubic_coefficient_hartree_bohr6 * delta**3
            + self.quartic_coefficient_hartree_bohr15 * density**2 * delta**4
        )
        immutable = np.array(result, dtype=float, copy=True)
        immutable.setflags(write=False)
        return immutable

    def first_derivative_hartree(
        self,
        weighted_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the local derivative of the bridge density with respect to rho."""

        density = _grid_field(
            self.center_projection.grid,
            weighted_density_bohr3,
            name="Weighted molecular density",
        )
        delta = density - self.molecular_bulk_number_density_bohr3
        result = (
            3.0 * self.cubic_coefficient_hartree_bohr6 * delta** 2
            + self.quartic_coefficient_hartree_bohr15
            * (2.0 * density * delta**4 + 4.0 * density**2 * delta**3)
        )
        immutable = np.array(result, dtype=float, copy=True)
        immutable.setflags(write=False)
        return immutable

    def second_derivative_hartree_bohr3(
        self,
        weighted_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the local curvature of the bridge density with respect to rho."""

        density = _grid_field(
            self.center_projection.grid,
            weighted_density_bohr3,
            name="Weighted molecular density",
        )
        delta = density - self.molecular_bulk_number_density_bohr3
        result = (
            6.0 * self.cubic_coefficient_hartree_bohr6 * delta
            + self.quartic_coefficient_hartree_bohr15
            * (
                2.0 * delta**4
                + 16.0 * density * delta**3
                + 12.0 * density**2 * delta**2
            )
        )
        immutable = np.array(result, dtype=float, copy=True)
        immutable.setflags(write=False)
        return immutable

    @property
    def vacuum_bridge_free_energy_density_hartree_per_bohr3(self) -> float:
        """Return the bridge contribution at the empty molecular state."""

        density = self.molecular_bulk_number_density_bohr3
        return -self.cubic_coefficient_hartree_bohr6 * density**3

    @property
    def coexistence_pressure_hartree_per_bohr3(self) -> float:
        """Return the same-functional vacuum-limit pressure after the bridge."""

        return (
            self.hnc_bulk_pressure_hartree_per_bohr3
            + self.vacuum_bridge_free_energy_density_hartree_per_bohr3
        )


@dataclass(frozen=True)
class Route2V0MolecularWeightedDensityBridgeState:
    """One stationary-state record for the HNC-plus-bridge scalar."""

    functional: Route2V0MolecularWeightedDensityBridgeFunctional
    configuration_density_bohr3: np.ndarray
    grand_potential_hartree: float
    ideal_contribution_hartree: float
    hnc_excess_contribution_hartree: float
    bridge_contribution_hartree: float
    external_contribution_hartree: float
    residual_inf: float
    iterations: int
    construction: str = V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(
            self.functional, Route2V0MolecularWeightedDensityBridgeFunctional
        ):
            raise TypeError(
                "Weighted-density bridge state requires its exact functional."
            )
        if self.construction != V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 weighted-density bridge state.")
        density = self.functional._density(self.configuration_density_bohr3)
        values = {
            "grand_potential_hartree": self.grand_potential_hartree,
            "ideal_contribution_hartree": self.ideal_contribution_hartree,
            "hnc_excess_contribution_hartree": self.hnc_excess_contribution_hartree,
            "bridge_contribution_hartree": self.bridge_contribution_hartree,
            "external_contribution_hartree": self.external_contribution_hartree,
            "residual_inf": self.residual_inf,
        }
        for name, raw_value in values.items():
            value = _finite(raw_value, name=f"Weighted-density bridge state {name}")
            if name == "residual_inf" and value < 0.0:
                raise ValueError(
                    "Weighted-density bridge state residual must be nonnegative."
                )
            object.__setattr__(self, name, value)
        expected = (
            self.ideal_contribution_hartree
            + self.hnc_excess_contribution_hartree
            + self.bridge_contribution_hartree
            + self.external_contribution_hartree
        )
        tolerance = 1.0e-12 * max(1.0, abs(expected))
        if abs(self.grand_potential_hartree - expected) > tolerance:
            raise ValueError(
                "Weighted-density bridge grand potential must equal all four "
                "same-functional components."
            )
        object.__setattr__(self, "configuration_density_bohr3", density)
        object.__setattr__(
            self,
            "iterations",
            _nonnegative_integer(
                self.iterations, name="Weighted-density bridge iteration count"
            ),
        )


@dataclass(frozen=True)
class Route2V0MolecularWeightedDensityBridgeFunctional:
    """Molecular HNC plus a reciprocal weighted-density bridge scalar."""

    hnc_functional: Route2V0MolecularSiteHNCFunctional
    bridge_asset: Route2V0MolecularWeightedDensityBridgeAsset
    construction: str = V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION
    _beta_hartree_inverse: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.hnc_functional, Route2V0MolecularSiteHNCFunctional):
            raise TypeError(
                "Weighted-density bridge requires a molecular HNC functional."
            )
        if not isinstance(
            self.bridge_asset, Route2V0MolecularWeightedDensityBridgeAsset
        ):
            raise TypeError("Weighted-density bridge requires a frozen bridge asset.")
        if self.construction != V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 weighted-density bridge functional.")
        if (
            self.bridge_asset.center_projection.projection
            is not self.hnc_functional.projection
        ):
            raise ValueError(
                "Weighted-density bridge must use the exact projection of its "
                "molecular HNC scalar."
            )
        expected_pressure = molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(
            self.hnc_functional
        )
        actual_pressure = self.bridge_asset.hnc_bulk_pressure_hartree_per_bohr3
        tolerance = 1.0e-12 * max(1.0, abs(expected_pressure), abs(actual_pressure))
        if abs(expected_pressure - actual_pressure) > tolerance:
            raise ValueError(
                "Weighted-density bridge source pressure must come from its exact "
                "molecular HNC scalar."
            )
        kbt = self.hnc_functional.projection.site_hnc_asset.kbt_hartree
        object.__setattr__(self, "_beta_hartree_inverse", 1.0 / kbt)

    @property
    def projection(self) -> Route2V0MolecularSiteProjection:
        """Return the exact shared molecular site projection."""

        return self.hnc_functional.projection

    @property
    def center_projection(self) -> Route2V0MolecularCenterProjection:
        """Return the separately declared molecular-centre projection."""

        return self.bridge_asset.center_projection

    @property
    def kbt_hartree(self) -> float:
        """Return the common thermodynamic energy unit."""

        return self.hnc_functional.projection.site_hnc_asset.kbt_hartree

    def _density(self, values: np.ndarray) -> np.ndarray:
        return _configuration_density(
            self.projection,
            values,
            name="Molecular configuration density",
            positive=True,
        )

    def _direction(self, values: np.ndarray) -> np.ndarray:
        return _configuration_density(
            self.projection,
            values,
            name="Molecular configuration-density direction",
            positive=False,
        )

    def molecular_density(self, configuration_density_bohr3: np.ndarray) -> np.ndarray:
        """Return the molecular-centre density ``D nu`` on the Cartesian grid."""

        return self.center_projection.project_configuration_density(
            self._density(configuration_density_bohr3)
        )

    def weighted_molecular_density(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return ``K D nu`` for the declared periodic kernel."""

        return self.bridge_asset.kernel.convolve(
            self.molecular_density(configuration_density_bohr3)
        )

    def bridge_contribution_hartree(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> float:
        """Return the bridge term evaluated before minimisation."""

        weighted = self.weighted_molecular_density(configuration_density_bohr3)
        energy = float(
            self.center_projection.grid.volume_element_bohr3
            * np.sum(self.bridge_asset.free_energy_density_hartree_per_bohr3(weighted))
        )
        if not math.isfinite(energy):
            raise RuntimeError("Weighted-density bridge energy is non-finite.")
        return energy

    def bridge_dimensionless_gradient(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return ``beta * delta F_bridge / delta nu`` in the exact pairing."""

        weighted = self.weighted_molecular_density(configuration_density_bohr3)
        local_field = self.bridge_asset.first_derivative_hartree(weighted)
        smoothed_field = self.bridge_asset.kernel.convolve(local_field)
        result = (
            self._beta_hartree_inverse
            * self.center_projection.center_field_adjoint_dimensionless(smoothed_field)
        )
        immutable = np.array(result, dtype=float, copy=True)
        immutable.setflags(write=False)
        return immutable

    def bridge_dimensionless_hessian_matvec(
        self,
        configuration_density_bohr3: np.ndarray,
        direction_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the exact bridge Hessian action in the molecular pairing."""

        weighted = self.weighted_molecular_density(configuration_density_bohr3)
        direction = self._direction(direction_bohr3)
        weighted_direction = self.bridge_asset.kernel.convolve(
            self.center_projection.project_configuration_direction(direction)
        )
        local_action = (
            self.bridge_asset.second_derivative_hartree_bohr3(weighted)
            * weighted_direction
        )
        smoothed_action = self.bridge_asset.kernel.convolve(local_action)
        result = (
            self._beta_hartree_inverse
            * self.center_projection.center_field_adjoint_dimensionless(smoothed_action)
        )
        immutable = np.array(result, dtype=float, copy=True)
        immutable.setflags(write=False)
        return immutable

    def dimensionless_gradient(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the gradient of the one HNC-plus-bridge scalar."""

        density = self._density(configuration_density_bohr3)
        result = self.hnc_functional.dimensionless_gradient(
            density
        ) + self.bridge_dimensionless_gradient(density)
        immutable = np.array(result, dtype=float, copy=True)
        immutable.setflags(write=False)
        return immutable

    def dimensionless_hessian_matvec(
        self,
        configuration_density_bohr3: np.ndarray,
        direction_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the reciprocal Hessian action of the one bridge scalar."""

        density = self._density(configuration_density_bohr3)
        direction = self._direction(direction_bohr3)
        result = self.hnc_functional.dimensionless_hessian_matvec(
            density, direction
        ) + self.bridge_dimensionless_hessian_matvec(density, direction)
        immutable = np.array(result, dtype=float, copy=True)
        immutable.setflags(write=False)
        return immutable

    def hessian_quadratic_hartree(
        self,
        configuration_density_bohr3: np.ndarray,
        direction_bohr3: np.ndarray,
    ) -> float:
        """Return the scalar second variation along one signed direction."""

        direction = self._direction(direction_bohr3)
        action = self.dimensionless_hessian_matvec(
            configuration_density_bohr3,
            direction,
        )
        value = float(
            self.kbt_hartree
            * np.sum(
                self.projection.quadrature.phase_space_weights_bohr3
                * direction
                * action
            )
        )
        if not math.isfinite(value):
            raise RuntimeError("Weighted-density bridge Hessian form is non-finite.")
        return value

    def energy_components(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Return ideal, HNC, bridge, and external components of one scalar."""

        density = self._density(configuration_density_bohr3)
        ideal, hnc_excess, external = self.hnc_functional.energy_components(density)
        bridge = self.bridge_contribution_hartree(density)
        if not all(
            math.isfinite(value) for value in (ideal, hnc_excess, bridge, external)
        ):
            raise RuntimeError(
                "Weighted-density bridge energy components are non-finite."
            )
        return ideal, hnc_excess, bridge, external

    def grand_potential_hartree(self, configuration_density_bohr3: np.ndarray) -> float:
        """Return the full variational grand-potential difference."""

        return float(sum(self.energy_components(configuration_density_bohr3)))

    @property
    def bulk_functional_pressure_hartree_per_bohr3(self) -> float:
        """Return the bridge vacuum-limit pressure from the same scalar."""

        pressure = self.bridge_asset.coexistence_pressure_hartree_per_bohr3
        target = self.bridge_asset.target_bulk_pressure_hartree_per_bohr3
        tolerance = 1.0e-12 * max(1.0, abs(pressure), abs(target))
        if abs(pressure - target) > tolerance:
            raise RuntimeError(
                "Weighted-density bridge failed its exact same-functional pressure "
                "identity."
            )
        return pressure

    def stationary_state(
        self,
        configuration_density_bohr3: np.ndarray,
        *,
        iterations: int,
    ) -> Route2V0MolecularWeightedDensityBridgeState:
        """Build one auditable state without claiming convergence by itself."""

        density = self._density(configuration_density_bohr3)
        ideal, hnc_excess, bridge, external = self.energy_components(density)
        return Route2V0MolecularWeightedDensityBridgeState(
            functional=self,
            configuration_density_bohr3=density,
            grand_potential_hartree=ideal + hnc_excess + bridge + external,
            ideal_contribution_hartree=ideal,
            hnc_excess_contribution_hartree=hnc_excess,
            bridge_contribution_hartree=bridge,
            external_contribution_hartree=external,
            residual_inf=float(np.max(np.abs(self.dimensionless_gradient(density)))),
            iterations=iterations,
        )

    def solve_picard(
        self,
        *,
        initial_configuration_density_bohr3: np.ndarray | None = None,
        residual_tolerance: float = 1.0e-10,
        picard_mixing: float = 0.2,
        max_iterations: int = 1000,
    ) -> Route2V0MolecularWeightedDensityBridgeState:
        """Solve the same scalar with numerical log-density Picard mixing only."""

        tolerance = _positive(
            residual_tolerance,
            name="Weighted-density bridge residual tolerance",
        )
        mixing = _positive(picard_mixing, name="Weighted-density bridge Picard mixing")
        if mixing > 1.0:
            raise ValueError(
                "Weighted-density bridge Picard mixing must be at most one."
            )
        maximum = _positive_integer(
            max_iterations,
            name="Weighted-density bridge maximum iterations",
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
                    "Weighted-density bridge Picard iteration produced a non-finite "
                    "log density."
                )
            density = bulk * np.exp(log_ratio)
            if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
                raise RuntimeError(
                    "Weighted-density bridge Picard iteration produced an invalid "
                    "density."
                )
        raise RuntimeError(
            "Weighted-density bridge Picard solver did not meet the declared "
            f"stationarity tolerance within {maximum} iterations."
        )


__all__ = [
    "V0_MOLECULAR_CENTER_PROJECTION_CONSTRUCTION",
    "V0_MOLECULAR_WEIGHTED_DENSITY_BRIDGE_CONSTRUCTION",
    "Route2V0MolecularCenterProjection",
    "Route2V0MolecularWeightedDensityBridgeAsset",
    "Route2V0MolecularWeightedDensityBridgeFunctional",
    "Route2V0MolecularWeightedDensityBridgeState",
    "Route2V0PeriodicWeightedDensityKernel",
]
