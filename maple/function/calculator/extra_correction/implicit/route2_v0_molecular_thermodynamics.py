"""Same-functional ensemble observables for the Route-2 V0 molecular HNC scalar.

The molecular HNC control varies one rigid-molecule configuration density
``nu(X, Omega)``.  Its stationary value is a grand-potential difference, not
yet an isobaric solvation free energy.  This module derives the two quantities
needed for the fixed-solute ensemble conversion from that *same* discrete
functional:

``P_F = lim_(nu -> 0) DeltaOmega_F[nu; 0] / V``

and

``Vbar = (N_bulk - N[nu*]) / rho_bulk``.

The resulting fixed-solute pressure correction is

``DeltaG_F = DeltaOmega_F[nu*] - P_F * Vbar``.

Because the variational state is molecular, ``P_F`` contains one molecular
ideal-gas term.  It must not be replaced by the different site-count ideal
term of a stock site-density 3D-RISM functional merely because the correlation
kernel originated in a RISM asset.

This remains a thermodynamic identity/control, not a total-solvation endpoint.
It adds no PC+, universal correction, fitted partial-volume coefficient,
standard-state conversion, external-pressure work, gas-phase MACE energy, or
experimental calibration.  Those categories are deliberately kept separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .route2_v0_molecular_site_hnc import (
    Route2V0MolecularSiteHNCFunctional,
    Route2V0MolecularSiteHNCState,
)

V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS = (
    "route2-v0-molecular-hnc-fixed-solute-thermodynamics-v1"
)


def _finite(value: object, *, name: str) -> float:
    """Return one finite scalar without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        result = float(value)  # type: ignore[arg-type]
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


def _cell_volume_bohr3(functional: Route2V0MolecularSiteHNCFunctional) -> float:
    grid = functional.projection.site_hnc_asset.grid
    return float(grid.point_count * grid.volume_element_bohr3)


def _molecule_count(
    functional: Route2V0MolecularSiteHNCFunctional,
    density_bohr3: np.ndarray,
) -> float:
    density = np.asarray(density_bohr3, dtype=float)
    expected_shape = (functional.projection.quadrature.configuration_count,)
    if (
        density.shape != expected_shape
        or not np.all(np.isfinite(density))
        or np.any(density <= 0.0)
    ):
        raise ValueError(
            "Molecular HNC configuration density must be finite and positive "
            f"with shape {expected_shape}."
        )
    count = float(
        np.sum(functional.projection.quadrature.phase_space_weights_bohr3 * density)
    )
    if not math.isfinite(count) or count <= 0.0:
        raise RuntimeError("Molecular HNC integrated molecule count is invalid.")
    return count


def molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(
    functional: Route2V0MolecularSiteHNCFunctional,
) -> float:
    """Return the vacuum-limit pressure of this exact molecular HNC scalar.

    For cell volume ``V``, molecular bulk density ``rho_b``, and projected
    site bulk density ``n_b``, the analytic zero-density limit is

    ``P_F V = kBT rho_b V
             - kBT/2 * dv * <(-n_b), C(-n_b)>``.

    Evaluating the already implemented correlation operator rather than a
    separately transcribed ``k=0`` formula keeps FFT normalization, site
    multiplicity, source-SMEAR splitting, and the molecular one-ideal-term
    convention identical to the stationary scalar.
    """

    if not isinstance(functional, Route2V0MolecularSiteHNCFunctional):
        raise TypeError(
            "Molecular HNC bulk pressure requires a molecular HNC functional."
        )
    projection = functional.projection
    asset = projection.site_hnc_asset
    cell_volume = _cell_volume_bohr3(functional)
    bulk = np.broadcast_to(
        asset.bulk_number_density_bohr3.reshape((asset.site_count, 1, 1, 1)),
        (asset.site_count, *asset.grid.shape),
    )
    vacuum_delta = -bulk
    correlation = functional.convolve_direct_correlation(vacuum_delta)
    ideal_vacuum = (
        asset.kbt_hartree * projection.molecular_bulk_number_density_bohr3 * cell_volume
    )
    excess_vacuum = float(
        -0.5
        * asset.kbt_hartree
        * asset.grid.volume_element_bohr3
        * np.sum(vacuum_delta * correlation)
    )
    pressure = (ideal_vacuum + excess_vacuum) / cell_volume
    if not math.isfinite(pressure):
        raise RuntimeError("Molecular HNC bulk functional pressure is non-finite.")
    return float(pressure)


@dataclass(frozen=True)
class Route2V0MolecularHNCFixedSoluteThermodynamics:
    """Auditable fixed-solute ensemble correction from one stationary scalar.

    No standard-state term is contained in this object.  In particular, it
    cannot be relabelled as PC+, UC, MILC, or a total solvation free energy.
    """

    functional: Route2V0MolecularSiteHNCFunctional
    state: Route2V0MolecularSiteHNCState
    residual_tolerance: float
    construction: str = V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS

    def __post_init__(self) -> None:
        if not isinstance(self.functional, Route2V0MolecularSiteHNCFunctional):
            raise TypeError(
                "Fixed-solute thermodynamics requires a molecular HNC functional."
            )
        if not isinstance(self.state, Route2V0MolecularSiteHNCState):
            raise TypeError(
                "Fixed-solute thermodynamics requires a molecular HNC state."
            )
        if self.state.projection is not self.functional.projection:
            raise ValueError(
                "Fixed-solute thermodynamics requires the state's exact molecular "
                "HNC projection."
            )
        if self.construction != V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS:
            raise ValueError(
                "Unsupported Route-2 molecular HNC thermodynamic construction."
            )
        tolerance = _positive(
            self.residual_tolerance,
            name="Fixed-solute thermodynamic residual tolerance",
        )
        if self.state.residual_inf > tolerance:
            raise ValueError(
                "Fixed-solute thermodynamics requires a stationary molecular HNC "
                "state within the declared residual tolerance."
            )
        density = self.state.configuration_density_bohr3
        observed_residual = float(
            np.max(np.abs(self.functional.dimensionless_gradient(density)))
        )
        if not math.isfinite(observed_residual) or observed_residual > tolerance:
            raise ValueError(
                "Fixed-solute thermodynamics independently verifies molecular HNC "
                "stationarity rather than trusting the state's residual label."
            )
        residual_scale = max(1.0, observed_residual, self.state.residual_inf)
        if abs(observed_residual - self.state.residual_inf) > 1.0e-12 * residual_scale:
            raise ValueError(
                "Molecular HNC state residual does not match the exact functional."
            )
        expected_components = self.functional.energy_components(density)
        stored_components = (
            self.state.ideal_contribution_hartree,
            self.state.excess_contribution_hartree,
            self.state.external_contribution_hartree,
        )
        for stored, expected in zip(
            stored_components, expected_components, strict=True
        ):
            scale = max(1.0, abs(stored), abs(expected))
            if abs(stored - expected) > 1.0e-12 * scale:
                raise ValueError(
                    "Molecular HNC state energy components do not match the exact "
                    "functional."
                )
        object.__setattr__(self, "residual_tolerance", tolerance)

    @property
    def bulk_reference_molecule_count(self) -> float:
        """Return ``rho_bulk * V`` for the exact functional cell."""

        return (
            self.functional.projection.molecular_bulk_number_density_bohr3
            * _cell_volume_bohr3(self.functional)
        )

    @property
    def stationary_molecule_count(self) -> float:
        """Return the configuration-space integral of the stationary density."""

        return _molecule_count(
            self.functional,
            self.state.configuration_density_bohr3,
        )

    @property
    def partial_molar_volume_bohr3(self) -> float:
        """Return the stationary molecule deficit divided by bulk density."""

        density = self.functional.projection.molecular_bulk_number_density_bohr3
        return (
            self.bulk_reference_molecule_count - self.stationary_molecule_count
        ) / density

    @property
    def bulk_functional_pressure_hartree_per_bohr3(self) -> float:
        """Return the vacuum-limit pressure of the exact molecular scalar."""

        return molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(self.functional)

    @property
    def bulk_functional_pressure_work_hartree(self) -> float:
        """Return ``P_functional * Vbar``."""

        return (
            self.bulk_functional_pressure_hartree_per_bohr3
            * self.partial_molar_volume_bohr3
        )

    @property
    def fixed_solute_pressure_corrected_free_energy_hartree(self) -> float:
        """Return ``DeltaOmega[nu*] - P_functional * Vbar``."""

        return (
            self.state.grand_potential_hartree
            - self.bulk_functional_pressure_work_hartree
        )


def evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics(
    functional: Route2V0MolecularSiteHNCFunctional,
    state: Route2V0MolecularSiteHNCState,
    *,
    residual_tolerance: float = 1.0e-10,
) -> Route2V0MolecularHNCFixedSoluteThermodynamics:
    """Evaluate the no-fit fixed-solute ensemble ledger after stationarity."""

    return Route2V0MolecularHNCFixedSoluteThermodynamics(
        functional=functional,
        state=state,
        residual_tolerance=residual_tolerance,
    )


__all__ = [
    "V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS",
    "Route2V0MolecularHNCFixedSoluteThermodynamics",
    "evaluate_route2_v0_molecular_hnc_fixed_solute_thermodynamics",
    "molecular_hnc_bulk_functional_pressure_hartree_per_bohr3",
]
