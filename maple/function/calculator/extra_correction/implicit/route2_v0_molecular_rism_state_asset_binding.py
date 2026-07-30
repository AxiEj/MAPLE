"""Fail-closed state--asset binding for the molecular-RISM Route-2 V0 branch.

The molecular-RISM bulk-state record and the frozen finite-wavevector asset
serve different purposes.  The former records independently sourced
``(T, p, rho_m, epsilon(0), kappa_T, gamma)``; the latter locks one site model,
one 1D-RISM serialization, and one finite-k correlation source.  Neither is a
liquid free-energy endpoint by itself.  This module makes their join explicit
before a later bridge or planar-interface certificate may consume the pure
liquid anchors.

The join deliberately remains source-only.  It checks solvent and model
identity, the site-model digest, all RISM-observable state fields, asset
pressure, and the site-density multiplicities.  It converts the independently
recorded thermodynamic anchors into atomic units, but it neither derives a
closure, chooses a bridge coefficient, nor promotes a source-complete control
to a physical liquid or an accuracy result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ase.units import Bohr, Hartree

from .route2_v0_molecular_rism_state_source import (
    Route2V0MolecularRismBulkStateSource,
)
from .route2_v0_solvent_asset import Route2V0FrozenSolventAsset

V0_MOLECULAR_RISM_STATE_ASSET_BINDING_CONSTRUCTION = (
    "route2-v0-molecular-rism-state-asset-binding-v1"
)
V0_MOLECULAR_RISM_STATE_ASSET_BINDING_STATUS = (
    "source-bound-state-and-asset-not-physical-liquid-or-accuracy-admitted"
)

# ASE stores Hartree in eV and Bohr in Angstrom.  The elementary charge is
# exact in SI, so these factors introduce no fitted conversion parameter.
_ELECTRONIC_CHARGE_JOULE = 1.602176634e-19
_HARTREE_PER_BOHR2_TO_NEWTON_PER_METER = (
    Hartree * _ELECTRONIC_CHARGE_JOULE / (Bohr * 1.0e-10) ** 2
)
_HARTREE_PER_BOHR3_TO_PASCAL = (
    Hartree * _ELECTRONIC_CHARGE_JOULE / (Bohr * 1.0e-10) ** 3
)


def _state_match(observed: float, expected: float, *, name: str) -> None:
    """Require one scalar state edge to agree at source serialization precision."""

    if not math.isclose(observed, expected, rel_tol=1.0e-8, abs_tol=1.0e-12):
        raise ValueError(
            "Molecular-RISM bulk-state source and frozen solvent asset "
            f"{name} disagree: source={observed!r}, asset={expected!r}."
        )


def _verify_site_density_multiplicities(
    *,
    frozen_solvent_asset: Route2V0FrozenSolventAsset,
    molecular_number_density_angstrom3: float,
) -> None:
    """Check that each frozen site density is its molecular multiplicity times rho.

    AmberTools XVV text rounds printed site densities more aggressively than
    the RISM input serialisation.  The tolerance is therefore limited to that
    declared text representation rather than silently requiring a new density
    or accepting a distinct site composition.
    """

    metadata = frozen_solvent_asset.bulk_direct_correlation.metadata
    expected = molecular_number_density_angstrom3 * np.asarray(
        metadata.site_multiplicity,
        dtype=float,
    )
    observed = np.asarray(metadata.bulk_number_density_angstrom3, dtype=float)
    if not np.allclose(observed, expected, rtol=1.0e-6, atol=1.0e-10):
        raise ValueError(
            "Frozen molecular-RISM site densities do not equal the declared "
            "molecular density times the frozen site multiplicities."
        )


def _verify_binding(
    *,
    frozen_solvent_asset: Route2V0FrozenSolventAsset,
    bulk_state_source: Route2V0MolecularRismBulkStateSource,
) -> None:
    """Verify every state--asset edge without constructing liquid physics."""

    frozen_solvent_asset.verify_integrity()
    if bulk_state_source.solvent_id != frozen_solvent_asset.solvent_id:
        raise ValueError(
            "Molecular-RISM bulk-state source and frozen solvent asset solvent "
            "ID disagree."
        )
    if bulk_state_source.model_identifier != frozen_solvent_asset.model_identifier:
        raise ValueError(
            "Molecular-RISM bulk-state source and frozen solvent asset model "
            "identifier disagree."
        )
    bulk_state_source.verify_model_source_digest(
        frozen_solvent_asset.file_for("site_model").sha256
    )
    _state_match(
        bulk_state_source.temperature_kelvin,
        frozen_solvent_asset.temperature_kelvin,
        name="temperature",
    )
    _state_match(
        bulk_state_source.pressure_bar,
        frozen_solvent_asset.pressure_bar,
        name="pressure",
    )
    bulk_state_source.verify_rism1d_state(
        frozen_solvent_asset.molecular_source.rism1d_input
    )
    _verify_site_density_multiplicities(
        frozen_solvent_asset=frozen_solvent_asset,
        molecular_number_density_angstrom3=(
            bulk_state_source.molecular_number_density_angstrom3
        ),
    )


@dataclass(frozen=True)
class Route2V0MolecularRismStateAssetBinding:
    """One source-bound molecular-RISM state and frozen finite-k asset.

    The object is intentionally not a solvent backend.  Its unit-converted
    quantities are independent pure-liquid anchors for a later *same-scalar*
    bridge certificate; they cannot be used to infer a bridge width, closure,
    short-range potential, or solvation correction.
    """

    frozen_solvent_asset: Route2V0FrozenSolventAsset
    bulk_state_source: Route2V0MolecularRismBulkStateSource
    construction: str = V0_MOLECULAR_RISM_STATE_ASSET_BINDING_CONSTRUCTION
    status: str = V0_MOLECULAR_RISM_STATE_ASSET_BINDING_STATUS

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_solvent_asset, Route2V0FrozenSolventAsset):
            raise TypeError(
                "Molecular-RISM state--asset binding requires a frozen solvent asset."
            )
        if not isinstance(
            self.bulk_state_source,
            Route2V0MolecularRismBulkStateSource,
        ):
            raise TypeError(
                "Molecular-RISM state--asset binding requires a source-only "
                "molecular-RISM bulk-state record."
            )
        if self.construction != V0_MOLECULAR_RISM_STATE_ASSET_BINDING_CONSTRUCTION:
            raise ValueError("Unsupported molecular-RISM state--asset construction.")
        if self.status != V0_MOLECULAR_RISM_STATE_ASSET_BINDING_STATUS:
            raise ValueError("Molecular-RISM state--asset binding must remain source-only.")
        _verify_binding(
            frozen_solvent_asset=self.frozen_solvent_asset,
            bulk_state_source=self.bulk_state_source,
        )

    @property
    def is_molecular_liquid_asset(self) -> bool:
        """Return false: this provenance join does not complete liquid physics."""

        return False

    @property
    def molecular_bulk_number_density_bohr3(self) -> float:
        """Return the independently sourced molecular density in ``a0^-3``."""

        return self.bulk_state_source.molecular_number_density_angstrom3 * Bohr**3

    @property
    def target_bulk_pressure_hartree_per_bohr3(self) -> float:
        """Return the independently sourced pure-liquid pressure in ``Eh/a0^3``."""

        return self.bulk_state_source.pressure_bar * 1.0e5 / _HARTREE_PER_BOHR3_TO_PASCAL

    @property
    def target_surface_tension_hartree_per_bohr2(self) -> float:
        """Return the independently sourced surface tension in ``Eh/a0^2``."""

        return (
            self.bulk_state_source.surface_tension_newton_per_meter
            / _HARTREE_PER_BOHR2_TO_NEWTON_PER_METER
        )

    @property
    def isothermal_compressibility_hartree_inverse_bohr3(self) -> float:
        """Return ``kappa_T`` in inverse atomic-pressure units ``a0^3/Eh``."""

        return (
            self.bulk_state_source.isothermal_compressibility_pa_inverse
            * _HARTREE_PER_BOHR3_TO_PASCAL
        )

    @property
    def number_structure_factor_zero_mode(self) -> float:
        """Return the source-only scalar compressibility zero-mode anchor."""

        return self.bulk_state_source.number_structure_factor_zero_mode

    def verify_integrity(self) -> None:
        """Recheck the file-backed asset before a later certificate consumes it."""

        _verify_binding(
            frozen_solvent_asset=self.frozen_solvent_asset,
            bulk_state_source=self.bulk_state_source,
        )


__all__ = [
    "V0_MOLECULAR_RISM_STATE_ASSET_BINDING_CONSTRUCTION",
    "V0_MOLECULAR_RISM_STATE_ASSET_BINDING_STATUS",
    "Route2V0MolecularRismStateAssetBinding",
]
