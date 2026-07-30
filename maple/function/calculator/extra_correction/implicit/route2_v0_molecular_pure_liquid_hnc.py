"""Asset-bound, zero-external molecular-HNC pure-liquid control for Route-2 V0.

This assembly deliberately excludes a solute and therefore uses the exact
pure-liquid scalar

``Omega_pure[nu] = Omega_id[nu; 0] + F_ex^HNC[P nu]``.

It binds the HNC reciprocal kernel, molecular geometry, site multiplicities,
Cartesian grid, full SO(3) configuration rule, and C2 periodic deposition to
one content-addressed solvent asset.  This is the required common-scalar input
for a future pure-liquid phase/coexistence calculation.  It remains a
finite-grid source control: a source-complete candidate is not thereby a
physical-liquid admission, a surface-tension result, a force/PES result, a
total solvation free energy, or an accuracy result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .route2_v0_mace_cluster_rism_bridge import (
    V0_MOLECULAR_HNC_REQUIRED_RISM_CLOSURE,
    build_route2_v0_asset_bound_rism_kernel,
    verify_route2_v0_asset_bound_rism_kernel,
)
from .route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
)
from .route2_v0_molecular_ideal_gas import Route2V0MolecularConfigurationQuadrature
from .route2_v0_molecular_pure_liquid_external_potential import (
    Route2V0MolecularPureLiquidExternalPotential,
    build_route2_v0_molecular_pure_liquid_external_potential,
)
from .route2_v0_molecular_site_bspline import (
    Route2V0MolecularSiteCubicBSplineDeposition,
    build_route2_v0_cartesian_euler_cubic_bspline_site_deposition,
)
from .route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
)
from .route2_v0_molecular_so3_quadrature import (
    Route2V0CartesianEulerProductQuadrature,
)
from .route2_v0_rism_energy_conjugate import Route2V0RismEnergyConjugateKernel
from .route2_v0_solvent_asset import Route2V0FrozenSolventAsset
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MOLECULAR_PURE_LIQUID_HNC_CONSTRUCTION = (
    "route2-v0-asset-bound-molecular-pure-liquid-hnc-control-v1"
)


def _same_grid(left: RegularCartesianGrid, right: RegularCartesianGrid) -> bool:
    """Return whether two inputs declare exactly one periodic Cartesian grid."""

    return bool(
        left.shape == right.shape
        and left.layout == right.layout
        and np.array_equal(left.origin_bohr, right.origin_bohr)
        and np.array_equal(left.spacing_bohr, right.spacing_bohr)
    )


def _same_configurations(
    left: Route2V0MolecularConfigurations,
    right: object,
) -> bool:
    """Return whether two configuration lists are exactly the same rule."""

    return bool(
        isinstance(right, Route2V0MolecularConfigurations)
        and np.array_equal(left.translations_bohr, right.translations_bohr)
        and np.array_equal(left.rotations, right.rotations)
    )


def _same_molecular_reference(
    left: Route2V0MolecularSolventReference,
    right: object,
) -> bool:
    """Return whether a source uses the asset's exact molecular reference."""

    return bool(
        isinstance(right, Route2V0MolecularSolventReference)
        and left.provenance_label == right.provenance_label
        and left.target_total_charge_e == right.target_total_charge_e
        and np.array_equal(left.atomic_numbers, right.atomic_numbers)
        and np.array_equal(left.site_charges_e, right.site_charges_e)
        and np.array_equal(left.reference_positions_bohr, right.reference_positions_bohr)
    )


def _require_hnc_asset(asset: Route2V0FrozenSolventAsset) -> None:
    """Reject a non-HNC source before it enters the HNC scalar."""

    if asset.closure.casefold() != V0_MOLECULAR_HNC_REQUIRED_RISM_CLOSURE.casefold():
        raise ValueError(
            "Pure-liquid molecular-HNC assembly requires an HNC frozen solvent "
            f"asset; the asset declares {asset.closure!r}."
        )


def _site_type_indices(
    *,
    frozen_solvent_asset: Route2V0FrozenSolventAsset,
    solvent: Route2V0MolecularSolventReference,
    rism_kernel: Route2V0RismEnergyConjugateKernel,
) -> np.ndarray:
    """Map every canonical molecular atom to its frozen RISM site type."""

    canonical = frozen_solvent_asset.molecular_reference
    if not _same_molecular_reference(solvent, canonical.molecular_reference):
        raise ValueError(
            "Pure-liquid molecular-HNC assembly solvent geometry and charges must "
            "equal the frozen asset's canonical molecular reference."
        )
    metadata = rism_kernel.short_range.radial.metadata
    lookup = {name: index for index, name in enumerate(metadata.site_names)}
    try:
        indices = np.asarray(
            [lookup[name] for name in canonical.rism_site_type_names],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            "Pure-liquid molecular-HNC assembly molecular site type is absent "
            "from the frozen RISM source."
        ) from exc
    observed = np.bincount(indices, minlength=metadata.site_count)
    if not np.array_equal(observed, metadata.site_multiplicity):
        raise ValueError(
            "Pure-liquid molecular-HNC assembly molecular site multiplicities "
            "must equal the frozen RISM XVV multiplicities."
        )
    indices.setflags(write=False)
    return indices


@dataclass(frozen=True)
class Route2V0MolecularPureLiquidHNCAssembly:
    """One fail-closed zero-external HNC scalar bound to one frozen asset."""

    frozen_solvent_asset: Route2V0FrozenSolventAsset
    external_potential: Route2V0MolecularPureLiquidExternalPotential
    rism_kernel: Route2V0RismEnergyConjugateKernel
    quadrature: Route2V0MolecularConfigurationQuadrature
    compact_site_deposition: Route2V0MolecularSiteCubicBSplineDeposition
    construction: str = V0_MOLECULAR_PURE_LIQUID_HNC_CONSTRUCTION
    _site_type_indices: np.ndarray = field(init=False, repr=False, compare=False)
    _projection: Route2V0MolecularSiteProjection = field(
        init=False,
        repr=False,
        compare=False,
    )
    _functional: Route2V0MolecularSiteHNCFunctional = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_solvent_asset, Route2V0FrozenSolventAsset):
            raise TypeError(
                "Pure-liquid molecular-HNC assembly requires a frozen solvent asset."
            )
        if not isinstance(
            self.external_potential,
            Route2V0MolecularPureLiquidExternalPotential,
        ):
            raise TypeError(
                "Pure-liquid molecular-HNC assembly requires the exact-zero "
                "pure-liquid external potential."
            )
        if not isinstance(self.rism_kernel, Route2V0RismEnergyConjugateKernel):
            raise TypeError(
                "Pure-liquid molecular-HNC assembly requires an energy-conjugate "
                "RISM kernel."
            )
        if not isinstance(self.quadrature, Route2V0MolecularConfigurationQuadrature):
            raise TypeError("Pure-liquid molecular-HNC assembly requires a quadrature.")
        if not isinstance(
            self.compact_site_deposition,
            Route2V0MolecularSiteCubicBSplineDeposition,
        ):
            raise TypeError(
                "Pure-liquid molecular-HNC assembly requires compact C2 site "
                "deposition."
            )
        if self.construction != V0_MOLECULAR_PURE_LIQUID_HNC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 pure-liquid HNC assembly.")

        _require_hnc_asset(self.frozen_solvent_asset)
        self.frozen_solvent_asset.verify_integrity()
        if not _same_grid(
            self.external_potential.integration_grid,
            self.rism_kernel.grid,
        ):
            raise ValueError(
                "Pure-liquid molecular-HNC assembly requires one shared "
                "external-potential and RISM Cartesian grid."
            )
        if not _same_configurations(
            self.quadrature.configurations,
            self.external_potential.configurations,
        ):
            raise ValueError(
                "Pure-liquid molecular-HNC assembly quadrature and exact-zero "
                "external potential configurations differ."
            )
        verify_route2_v0_asset_bound_rism_kernel(
            frozen_solvent_asset=self.frozen_solvent_asset,
            rism_kernel=self.rism_kernel,
        )
        site_type_indices = _site_type_indices(
            frozen_solvent_asset=self.frozen_solvent_asset,
            solvent=self.external_potential.solvent,
            rism_kernel=self.rism_kernel,
        )
        projection = Route2V0MolecularSiteProjection(
            quadrature=self.quadrature,
            external_potential=self.external_potential,
            site_hnc_asset=self.rism_kernel.site_hnc_asset,
            solvent_site_type_indices=site_type_indices,
            compact_site_deposition=self.compact_site_deposition,
        )
        metadata = self.rism_kernel.short_range.radial.metadata
        if not np.array_equal(projection.site_multiplicity, metadata.site_multiplicity):
            raise ValueError(
                "Pure-liquid molecular-HNC projection does not preserve frozen "
                "RISM site multiplicities."
            )
        object.__setattr__(self, "_site_type_indices", site_type_indices)
        object.__setattr__(self, "_projection", projection)
        object.__setattr__(
            self,
            "_functional",
            Route2V0MolecularSiteHNCFunctional(projection),
        )

    @classmethod
    def from_cartesian_euler_cubic_bspline_matrix_free(
        cls,
        *,
        frozen_solvent_asset: Route2V0FrozenSolventAsset,
        cartesian_euler_quadrature: Route2V0CartesianEulerProductQuadrature,
    ) -> Route2V0MolecularPureLiquidHNCAssembly:
        """Build the exact full-SO(3), C2 source-bound pure-liquid control."""

        if not isinstance(frozen_solvent_asset, Route2V0FrozenSolventAsset):
            raise TypeError(
                "Pure-liquid molecular-HNC assembly requires a frozen solvent asset."
            )
        if not isinstance(
            cartesian_euler_quadrature,
            Route2V0CartesianEulerProductQuadrature,
        ):
            raise TypeError(
                "Pure-liquid molecular-HNC assembly requires a Cartesian Euler "
                "product quadrature."
            )
        _require_hnc_asset(frozen_solvent_asset)
        frozen_solvent_asset.verify_integrity()
        kernel = build_route2_v0_asset_bound_rism_kernel(
            frozen_solvent_asset=frozen_solvent_asset,
            grid=cartesian_euler_quadrature.grid,
        )
        solvent = frozen_solvent_asset.molecular_reference.molecular_reference
        external = build_route2_v0_molecular_pure_liquid_external_potential(
            integration_grid=cartesian_euler_quadrature.grid,
            solvent=solvent,
            configurations=cartesian_euler_quadrature.configurations,
        )
        site_type_indices = _site_type_indices(
            frozen_solvent_asset=frozen_solvent_asset,
            solvent=solvent,
            rism_kernel=kernel,
        )
        deposition = build_route2_v0_cartesian_euler_cubic_bspline_site_deposition(
            cartesian_euler_quadrature=cartesian_euler_quadrature,
            solvent=solvent,
            solvent_site_type_indices=site_type_indices,
            site_count=kernel.site_hnc_asset.site_count,
        )
        return cls(
            frozen_solvent_asset=frozen_solvent_asset,
            external_potential=external,
            rism_kernel=kernel,
            quadrature=cartesian_euler_quadrature.quadrature,
            compact_site_deposition=deposition,
        )

    @property
    def site_type_indices(self) -> np.ndarray:
        """Return the immutable canonical molecular-atom to RISM-site map."""

        return self._site_type_indices

    @property
    def projection(self) -> Route2V0MolecularSiteProjection:
        """Return the one exact molecular-to-site density projection."""

        return self._projection

    @property
    def functional(self) -> Route2V0MolecularSiteHNCFunctional:
        """Return the one stationary exact-zero external molecular HNC scalar."""

        return self._functional

    def verify_integrity(self) -> None:
        """Recheck the frozen asset before a downstream liquid calculation."""

        self.frozen_solvent_asset.verify_integrity()


__all__ = [
    "V0_MOLECULAR_PURE_LIQUID_HNC_CONSTRUCTION",
    "Route2V0MolecularPureLiquidHNCAssembly",
]
