"""Asset-bound zero-field-MACE/RISM molecular HNC assembly for Route-2 V0.

The V0 liquid branch uses one fixed scalar external potential for a rigid
solvent configuration,

``u_MACE(Gamma) = E0(A union B_Gamma) - E0(A) - E0(B_Gamma)``,

and one frozen energy-conjugate RISM excess functional.  This module binds
those two inputs without replacing either source or adding an empirical
correction.  Its assembled scalar is exactly the existing projected molecular
HNC functional,

``Omega[nu] = Omega_id[nu; u_MACE] + F_ex^RISM[P nu]``.

The bridge is deliberately narrow.  It verifies that the RISM reciprocal
kernel was derived from the hash-locked solvent asset, that the MACE solvent
geometry equals the asset's canonical molecular reference with its
source-declared RISM multiplicities, and that the MACE vector uses one
checkpoint-locked source.  It is a declared *hybrid reference functional*, not
evidence that the gas MACE cluster model and the bulk RISM model came from one
microscopic force field.  No pressure correction, standard-state term,
production quadrature, physical liquid result, force, solvation free energy,
or accuracy claim is supplied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from ase.units import Bohr

from .route2_v0_mace_cluster_external_potential import (
    Route2V0MaceClusterMolecularExternalPotential,
    Route2V0MaceZeroFieldSourceProvenance,
)
from .route2_v0_molecular_ideal_gas import Route2V0MolecularConfigurationQuadrature
from .route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteProjection,
)
from .route2_v0_periodic_coulomb import Route2V0PeriodicCoulombOperator
from .route2_v0_rism_energy_conjugate import Route2V0RismEnergyConjugateKernel
from .route2_v0_rism_reciprocal import Route2V0RismShortRangeReciprocalControl
from .route2_v0_solvent_asset import Route2V0FrozenSolventAsset
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MACE_CLUSTER_RISM_BRIDGE_CONSTRUCTION = "route2-v0-mace-cluster-rism-bridge-v1"
V0_MACE_CLUSTER_RISM_CROSS_MODEL_REFERENCE = (
    "zero-field-mace-cluster-plus-frozen-rism-hybrid-reference-functional-v1"
)


def _same_grid(left: RegularCartesianGrid, right: RegularCartesianGrid) -> bool:
    """Return whether two inputs declare exactly the same periodic grid."""

    return bool(
        left.shape == right.shape
        and left.layout == right.layout
        and np.array_equal(left.origin_bohr, right.origin_bohr)
        and np.array_equal(left.spacing_bohr, right.spacing_bohr)
    )


def _same_rism_metadata(left: object, right: object) -> bool:
    """Require exact source metadata equality rather than a solvent-name match."""

    attributes = (
        "site_names",
        "temperature_kelvin",
        "dielectric_constant",
        "coulomb_smear_angstrom",
        "radial_spacing_angstrom",
        "radial_point_count",
        "component_count",
    )
    if any(getattr(left, name) != getattr(right, name) for name in attributes):
        return False
    arrays = (
        "site_multiplicity",
        "bulk_number_density_angstrom3",
        "site_charges_sqrt_kT_angstrom",
    )
    return all(
        np.array_equal(getattr(left, name), getattr(right, name)) for name in arrays
    )


def _kernel_matches_frozen_asset(
    *,
    asset: Route2V0FrozenSolventAsset,
    kernel: Route2V0RismEnergyConjugateKernel,
) -> bool:
    """Require that the reciprocal kernel descends from this exact bulk asset."""

    expected_radial = asset.bulk_direct_correlation.split_coulomb_long_range()
    actual_radial = kernel.short_range.radial
    return bool(
        _same_rism_metadata(expected_radial.metadata, actual_radial.metadata)
        and np.array_equal(expected_radial.radii_angstrom, actual_radial.radii_angstrom)
        and np.array_equal(
            expected_radial.values_dimensionless,
            actual_radial.values_dimensionless,
        )
        and math.isclose(
            kernel.short_range.tail_start_angstrom,
            asset.coulomb_tail_start_angstrom,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            kernel.short_range.tail_tolerance_dimensionless,
            asset.coulomb_tail_tolerance_dimensionless,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )


def _same_molecular_reference(left: object, right: object) -> bool:
    """Require exact equality with the molecular reference frozen in the asset."""

    attributes = ("provenance_label", "target_total_charge_e")
    if any(getattr(left, name) != getattr(right, name) for name in attributes):
        return False
    arrays = ("atomic_numbers", "site_charges_e", "reference_positions_bohr")
    return all(
        np.array_equal(getattr(left, name), getattr(right, name)) for name in arrays
    )


def _molecular_site_type_indices(
    *,
    molecular_site_type_names: tuple[str, ...] | list[str],
    source_site_names: tuple[str, ...],
    source_site_multiplicity: np.ndarray,
    molecular_site_count: int,
) -> np.ndarray:
    """Map every molecular atom site to one declared RISM site type by name."""

    names = tuple(molecular_site_type_names)
    if len(names) != molecular_site_count or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError(
            "MACE/RISM bridge molecular site-type names must be nonempty and have "
            "one entry per molecular site."
        )
    lookup = {name: index for index, name in enumerate(source_site_names)}
    try:
        indices = np.asarray([lookup[name] for name in names], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            "MACE/RISM bridge molecular site-type name is absent from the frozen "
            "RISM source."
        ) from exc
    observed = np.bincount(indices, minlength=len(source_site_names))
    if not np.array_equal(observed, source_site_multiplicity):
        raise ValueError(
            "MACE/RISM bridge molecular site multiplicities must equal the frozen "
            "RISM XVV multiplicities."
        )
    indices.setflags(write=False)
    return indices


def build_route2_v0_asset_bound_rism_kernel(
    *,
    frozen_solvent_asset: Route2V0FrozenSolventAsset,
    grid: RegularCartesianGrid,
) -> Route2V0RismEnergyConjugateKernel:
    """Build the one RISM kernel allowed by a hash-locked solvent asset.

    The native radial source is split with its own SMEAR, transformed with the
    asset's preregistered tail controls, and rejoined to the matching periodic
    smeared Coulomb operator.  No dielectric-only replacement, fitted scale,
    or arbitrary electrostatic prefactor is accepted.
    """

    if not isinstance(frozen_solvent_asset, Route2V0FrozenSolventAsset):
        raise TypeError("Asset-bound RISM kernel requires a frozen solvent asset.")
    if not isinstance(grid, RegularCartesianGrid):
        raise TypeError("Asset-bound RISM kernel requires a regular Cartesian grid.")
    frozen_solvent_asset.verify_integrity()
    radial = frozen_solvent_asset.bulk_direct_correlation.split_coulomb_long_range()
    short_range = Route2V0RismShortRangeReciprocalControl.from_radial(
        radial=radial,
        grid=grid,
        tail_start_angstrom=frozen_solvent_asset.coulomb_tail_start_angstrom,
        tail_tolerance_dimensionless=(
            frozen_solvent_asset.coulomb_tail_tolerance_dimensionless
        ),
    )
    return Route2V0RismEnergyConjugateKernel(
        short_range=short_range,
        periodic_coulomb=Route2V0PeriodicCoulombOperator(
            grid=grid,
            smear_bohr=radial.metadata.coulomb_smear_angstrom / Bohr,
        ),
    )


@dataclass(frozen=True)
class Route2V0MaceClusterRismMolecularHNCBridge:
    """One fail-closed assembly of the MACE scalar and frozen RISM scalar.

    The frozen asset owns the canonical geometry, atom identity, neutral
    site-charge convention, site-model digest, and RISM site-type mapping.  A
    caller may not independently declare any of those molecular-liquid inputs
    at bridge construction time.
    """

    frozen_solvent_asset: Route2V0FrozenSolventAsset
    external_potential: Route2V0MaceClusterMolecularExternalPotential
    rism_kernel: Route2V0RismEnergyConjugateKernel
    quadrature: Route2V0MolecularConfigurationQuadrature
    site_occupancy_weights: np.ndarray
    cross_model_reference: str = V0_MACE_CLUSTER_RISM_CROSS_MODEL_REFERENCE
    construction: str = V0_MACE_CLUSTER_RISM_BRIDGE_CONSTRUCTION
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
            raise TypeError("MACE/RISM bridge requires a frozen solvent asset.")
        if not isinstance(
            self.external_potential,
            Route2V0MaceClusterMolecularExternalPotential,
        ):
            raise TypeError(
                "MACE/RISM bridge requires the zero-field MACE cluster external "
                "potential, not another external-potential source."
            )
        if not isinstance(self.rism_kernel, Route2V0RismEnergyConjugateKernel):
            raise TypeError(
                "MACE/RISM bridge requires an energy-conjugate RISM kernel."
            )
        if not isinstance(
            self.quadrature,
            Route2V0MolecularConfigurationQuadrature,
        ):
            raise TypeError("MACE/RISM bridge requires a molecular quadrature.")
        if self.construction != V0_MACE_CLUSTER_RISM_BRIDGE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 MACE/RISM bridge construction.")
        if self.cross_model_reference != V0_MACE_CLUSTER_RISM_CROSS_MODEL_REFERENCE:
            raise ValueError(
                "MACE/RISM bridge requires the declared hybrid reference convention."
            )

        self.frozen_solvent_asset.verify_integrity()
        if not _same_grid(
            self.external_potential.integration_grid, self.rism_kernel.grid
        ):
            raise ValueError(
                "MACE/RISM bridge requires the MACE external potential and frozen "
                "RISM kernel to share one Cartesian grid."
            )
        if not _kernel_matches_frozen_asset(
            asset=self.frozen_solvent_asset,
            kernel=self.rism_kernel,
        ):
            raise ValueError(
                "MACE/RISM bridge kernel does not derive from the frozen solvent "
                "asset's exact bulk direct correlation and tail controls."
            )
        canonical_reference = self.frozen_solvent_asset.molecular_reference
        if not _same_molecular_reference(
            self.external_potential.solvent,
            canonical_reference.molecular_reference,
        ):
            raise ValueError(
                "MACE/RISM bridge solvent geometry and charges must equal the "
                "frozen asset's canonical molecular reference."
            )

        metadata = self.rism_kernel.short_range.radial.metadata
        type_indices = _molecular_site_type_indices(
            molecular_site_type_names=canonical_reference.rism_site_type_names,
            source_site_names=metadata.site_names,
            source_site_multiplicity=metadata.site_multiplicity,
            molecular_site_count=self.external_potential.solvent.site_count,
        )
        projection = Route2V0MolecularSiteProjection(
            quadrature=self.quadrature,
            external_potential=self.external_potential,
            site_hnc_asset=self.rism_kernel.site_hnc_asset,
            solvent_site_type_indices=type_indices,
            site_occupancy_weights=self.site_occupancy_weights,
        )
        if not np.array_equal(projection.site_multiplicity, metadata.site_multiplicity):
            raise ValueError(
                "MACE/RISM bridge projection does not preserve frozen RISM site "
                "multiplicities."
            )
        object.__setattr__(self, "_site_type_indices", type_indices)
        object.__setattr__(self, "_projection", projection)
        object.__setattr__(
            self,
            "_functional",
            Route2V0MolecularSiteHNCFunctional(projection),
        )

    @property
    def source_provenance(self) -> Route2V0MaceZeroFieldSourceProvenance:
        """Return the checkpoint-locked source identity of the MACE vector."""

        return self.external_potential.source_provenance

    @property
    def site_type_indices(self) -> np.ndarray:
        """Return the immutable molecular-atom to RISM-site map."""

        return self._site_type_indices

    @property
    def projection(self) -> Route2V0MolecularSiteProjection:
        """Return the one validated molecular-to-site density projection."""

        return self._projection

    @property
    def functional(self) -> Route2V0MolecularSiteHNCFunctional:
        """Return the one stationary molecular HNC scalar assembly."""

        return self._functional

    def verify_integrity(self) -> None:
        """Recheck the solvent files before a downstream liquid-state operation."""

        self.frozen_solvent_asset.verify_integrity()


__all__ = [
    "Route2V0MaceClusterRismMolecularHNCBridge",
    "V0_MACE_CLUSTER_RISM_BRIDGE_CONSTRUCTION",
    "V0_MACE_CLUSTER_RISM_CROSS_MODEL_REFERENCE",
    "build_route2_v0_asset_bound_rism_kernel",
]
