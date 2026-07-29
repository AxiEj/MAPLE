"""Exact rigid-molecule ideal-gas functional for the Route-2 V0 controls.

The molecular external-potential control supplies one energy for every rigid
solvent configuration ``Gamma = (X, Omega)``.  This module gives those values
their first variational use without inventing a liquid excess functional:

``Omega_id[n; u] = sum_i w_i { kBT [n_i log(n_i/n_b) - (n_i - n_b)] + n_i u_i }``.

Here ``w_i`` is a fixed quadrature weight for ``d^3X dOmega``, and
``n_b = rho_bulk / (8 pi^2)`` is the uniform configuration density under the
standard unnormalised rigid-body orientation measure.  The analytic stationary
state is therefore ``n_i = n_b exp(-u_i / kBT)``.  The implementation checks
the scalar/gradient identity exactly on the declared finite quadrature.

This is an ideal-gas molecular-density-functional control, not a liquid model:
there is no molecular excess functional, bulk correlation, dispersion,
standard-state conversion, force, PES, or solvation-accuracy claim.  It exists
only to ensure that a future physical molecular-liquid functional has one
configuration-space variational interface rather than an inconsistent sum of
sitewise corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
)
from .route2_v0_molecular_external_potential_contract import (
    Route2V0MolecularExternalPotentialContract,
    require_molecular_external_potential_values,
)

V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION = "route2-v0-molecular-ideal-gas-v1"
RIGID_MOLECULAR_ORIENTATION_MEASURE = 8.0 * math.pi**2


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    positive: bool = False,
) -> np.ndarray:
    """Return one finite immutable real array with the declared shape."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
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


def _positive_scalar(value: float, *, name: str) -> float:
    """Validate one finite, strictly positive scalar."""

    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _same_configurations(
    left: Route2V0MolecularConfigurations,
    right: object,
) -> bool:
    """Require exactly the same declared quadrature configurations."""

    if not isinstance(right, Route2V0MolecularConfigurations):
        return False
    return bool(
        np.array_equal(left.translations_bohr, right.translations_bohr)
        and np.array_equal(left.rotations, right.rotations)
    )


@dataclass(frozen=True)
class Route2V0MolecularConfigurationQuadrature:
    """Fixed quadrature for the rigid-molecule measure ``d^3X dOmega``.

    Each weight has units of Bohr cubed because the orientation measure is
    dimensionless.  The orientation convention is fixed to the unnormalised
    Haar measure of ``SO(3)``, whose total measure is ``8 pi^2``.  A future
    molecular excess functional must use this same convention or explicitly
    change the construction version.
    """

    configurations: Route2V0MolecularConfigurations
    phase_space_weights_bohr3: np.ndarray
    orientation_measure: float = RIGID_MOLECULAR_ORIENTATION_MEASURE
    construction: str = V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.configurations, Route2V0MolecularConfigurations):
            raise TypeError("Molecular ideal-gas quadrature requires configurations.")
        if self.construction != V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular ideal-gas construction.")
        orientation_measure = _positive_scalar(
            self.orientation_measure,
            name="Rigid-molecule orientation measure",
        )
        if not math.isclose(
            orientation_measure,
            RIGID_MOLECULAR_ORIENTATION_MEASURE,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Molecular ideal-gas quadrature requires the 8*pi**2 orientation measure."
            )
        weights = _immutable_array(
            self.phase_space_weights_bohr3,
            name="Molecular phase-space weights",
            shape=(self.configurations.configuration_count,),
            positive=True,
        )
        if not math.isfinite(float(np.sum(weights))):
            raise ValueError("Molecular phase-space measure must be finite.")
        object.__setattr__(self, "phase_space_weights_bohr3", weights)
        object.__setattr__(self, "orientation_measure", orientation_measure)

    @property
    def configuration_count(self) -> int:
        """Return the number of declared translation--orientation points."""

        return self.configurations.configuration_count

    @property
    def total_phase_space_measure_bohr3(self) -> float:
        """Return the finite quadrature measure of the represented domain."""

        return float(np.sum(self.phase_space_weights_bohr3))


@dataclass(frozen=True)
class Route2V0MolecularIdealGasState:
    """One stationary or trial state of the declared molecular ideal scalar."""

    quadrature: Route2V0MolecularConfigurationQuadrature
    external_potential: Route2V0MolecularExternalPotentialContract
    configuration_density_bohr3: np.ndarray
    grand_potential_hartree: float
    ideal_contribution_hartree: float
    external_contribution_hartree: float
    residual_inf: float
    construction: str = V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, Route2V0MolecularConfigurationQuadrature):
            raise TypeError("Molecular ideal-gas state requires a quadrature.")
        if not isinstance(
            self.external_potential,
            Route2V0MolecularExternalPotentialContract,
        ):
            raise TypeError("Molecular ideal-gas state requires an external potential.")
        if not _same_configurations(
            self.quadrature.configurations,
            getattr(self.external_potential, "configurations", None),
        ):
            raise ValueError(
                "Molecular ideal-gas quadrature and external potential configurations differ."
            )
        if self.construction != V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular ideal-gas construction.")
        density = _immutable_array(
            self.configuration_density_bohr3,
            name="Molecular configuration density",
            shape=(self.quadrature.configuration_count,),
            positive=True,
        )
        require_molecular_external_potential_values(
            self.external_potential,
            configuration_count=self.quadrature.configuration_count,
        )
        values = {
            "grand_potential_hartree": self.grand_potential_hartree,
            "ideal_contribution_hartree": self.ideal_contribution_hartree,
            "external_contribution_hartree": self.external_contribution_hartree,
            "residual_inf": self.residual_inf,
        }
        for name, raw_value in values.items():
            value = float(raw_value)
            if not math.isfinite(value) or (name == "residual_inf" and value < 0.0):
                raise ValueError(f"Molecular ideal-gas state {name} must be finite.")
            object.__setattr__(self, name, value)
        expected = self.ideal_contribution_hartree + self.external_contribution_hartree
        tolerance = 1.0e-12 * max(1.0, abs(expected))
        if abs(self.grand_potential_hartree - expected) > tolerance:
            raise ValueError(
                "Molecular ideal-gas grand potential must equal its two energy components."
            )
        object.__setattr__(self, "configuration_density_bohr3", density)


@dataclass(frozen=True)
class Route2V0MolecularIdealGasFunctional:
    """Exact discrete ideal contribution for one molecular external potential."""

    quadrature: Route2V0MolecularConfigurationQuadrature
    external_potential: Route2V0MolecularExternalPotentialContract
    bulk_molecular_number_density_bohr3: float
    kbt_hartree: float
    construction: str = V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, Route2V0MolecularConfigurationQuadrature):
            raise TypeError("Molecular ideal-gas functional requires a quadrature.")
        if not isinstance(
            self.external_potential,
            Route2V0MolecularExternalPotentialContract,
        ):
            raise TypeError(
                "Molecular ideal-gas functional requires an external potential."
            )
        if not _same_configurations(
            self.quadrature.configurations,
            getattr(self.external_potential, "configurations", None),
        ):
            raise ValueError(
                "Molecular ideal-gas quadrature and external potential configurations differ."
            )
        if self.construction != V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular ideal-gas construction.")
        require_molecular_external_potential_values(
            self.external_potential,
            configuration_count=self.quadrature.configuration_count,
        )
        object.__setattr__(
            self,
            "bulk_molecular_number_density_bohr3",
            _positive_scalar(
                self.bulk_molecular_number_density_bohr3,
                name="Bulk molecular number density",
            ),
        )
        object.__setattr__(
            self,
            "kbt_hartree",
            _positive_scalar(self.kbt_hartree, name="Molecular ideal-gas kBT"),
        )

    @property
    def uniform_configuration_density_bohr3(self) -> float:
        """Return ``rho_bulk / (8*pi^2)`` in the declared orientation convention."""

        return (
            self.bulk_molecular_number_density_bohr3
            / self.quadrature.orientation_measure
        )

    @property
    def _external_energy_hartree(self) -> np.ndarray:
        """Return the one validated scalar source vector for this quadrature."""

        return require_molecular_external_potential_values(
            self.external_potential,
            configuration_count=self.quadrature.configuration_count,
        )

    def _density(self, values: np.ndarray) -> np.ndarray:
        return _immutable_array(
            values,
            name="Molecular configuration density",
            shape=(self.quadrature.configuration_count,),
            positive=True,
        )

    def dimensionless_gradient(
        self, configuration_density_bohr3: np.ndarray
    ) -> np.ndarray:
        """Return ``beta * delta Omega_id / delta n`` at every configuration."""

        density = self._density(configuration_density_bohr3)
        gradient = np.log(density / self.uniform_configuration_density_bohr3)
        gradient += self._external_energy_hartree / self.kbt_hartree
        result = np.array(gradient, dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def energy_components(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> tuple[float, float]:
        """Return the ideal and external contributions to ``Delta Omega_id``."""

        density = self._density(configuration_density_bohr3)
        bulk = self.uniform_configuration_density_bohr3
        weights = self.quadrature.phase_space_weights_bohr3
        ideal = self.kbt_hartree * float(
            np.sum(weights * (density * np.log(density / bulk) - (density - bulk)))
        )
        external = float(np.sum(weights * density * self._external_energy_hartree))
        if not all(math.isfinite(value) for value in (ideal, external)):
            raise RuntimeError("Molecular ideal-gas energy components are non-finite.")
        return ideal, external

    def grand_potential_hartree(self, configuration_density_bohr3: np.ndarray) -> float:
        """Return the one scalar whose gradient defines the ideal stationarity."""

        return float(sum(self.energy_components(configuration_density_bohr3)))

    def stationary_state(
        self,
        configuration_density_bohr3: np.ndarray,
    ) -> Route2V0MolecularIdealGasState:
        """Build one auditable state and report its exact scalar residual."""

        density = self._density(configuration_density_bohr3)
        gradient = self.dimensionless_gradient(density)
        ideal, external = self.energy_components(density)
        return Route2V0MolecularIdealGasState(
            quadrature=self.quadrature,
            external_potential=self.external_potential,
            configuration_density_bohr3=density,
            grand_potential_hartree=ideal + external,
            ideal_contribution_hartree=ideal,
            external_contribution_hartree=external,
            residual_inf=float(np.max(np.abs(gradient))),
        )

    def solve_analytic(self) -> Route2V0MolecularIdealGasState:
        """Return the exact stationary ideal-gas configuration density.

        This is not an approximate liquid solver: it is the closed-form
        minimizer of the declared ideal scalar and is retained as a precise
        control for a future molecular excess-functional implementation.
        """

        exponent = -self._external_energy_hartree / self.kbt_hartree
        if not np.all(np.isfinite(exponent)) or np.any(exponent > 700.0):
            raise RuntimeError(
                "Molecular ideal-gas external potential produces an unsafe exponent."
            )
        density = self.uniform_configuration_density_bohr3 * np.exp(exponent)
        if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
            raise RuntimeError("Molecular ideal-gas stationary density is invalid.")
        return self.stationary_state(density)


__all__ = [
    "RIGID_MOLECULAR_ORIENTATION_MEASURE",
    "Route2V0MolecularConfigurationQuadrature",
    "Route2V0MolecularIdealGasFunctional",
    "Route2V0MolecularIdealGasState",
    "V0_MOLECULAR_IDEAL_GAS_CONSTRUCTION",
]
