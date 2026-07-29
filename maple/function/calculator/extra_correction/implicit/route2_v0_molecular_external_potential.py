"""MACE-electrostatic plus parameter-free Pauli external potential for V0.

For one molecular-solvent configuration ``(X, Omega)``, this module evaluates
one explicitly declared external potential,

``u_ext(X, Omega) = sum_alpha z_alpha phi_MACE(X + Omega r_alpha)
                     + T_TF^nad[n_solute_ref, n_solvent_ref(X, Omega)]``.

The electrostatic summand consumes the unmodified zero-field MACE Gaussian
``l<=1`` source.  The short-range summand consumes separate, positive,
source-bound free-atom reference densities.  Thus it does not relabel the
promolecule as MACE density, does not update MACE with a field, and introduces
no fitted overlap radius or solvation-label parameter.

It is a discrete molecular-density-functional external-potential *control*.
It supplies neither a molecular-liquid excess functional, dispersion,
standard-state term, nor coordinate derivative suitable for a force/PES claim.
The tabulated radial densities are linearly interpolated and the Thomas--Fermi
integrand is not C2 at zero density, so callers must not promote this object
to Newton, optimization, scan, MD, or NVE use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from ase.units import Bohr

from .gto_density import gaussian_multipole_potential
from .route2_v0_frozen_density_embedding import Route2V0FrozenDensityPauliOverlap
from .route2_v0_molecular_external_potential_contract import (
    Route2V0MolecularExternalPotentialContract,
)
from .route2_v0_promolecular_density import Route2V0PromolecularDensityTable
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION = (
    "route2-v0-molecular-external-potential-v1"
)
V0_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE = "molecular-external-potential-control-only-v1"
V0_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY = (
    "energy-only-discrete-promolecular-control-v1"
)


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _atomic_numbers(values: np.ndarray, *, atom_count: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (atom_count,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be finite with shape ({atom_count},).")
    numeric = np.asarray(raw, dtype=float)
    rounded = np.rint(numeric)
    if np.any(rounded != numeric) or np.any(rounded < 1.0):
        raise ValueError(f"{name} must contain positive integer atomic numbers.")
    result = rounded.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonempty_text(value: str, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be nonempty.")
    return result


def _proper_rotations(values: np.ndarray) -> np.ndarray:
    rotations = np.asarray(values, dtype=float)
    if (
        rotations.ndim != 3
        or rotations.shape[0] == 0
        or rotations.shape[1:] != (3, 3)
        or not np.all(np.isfinite(rotations))
    ):
        raise ValueError("Molecular rotations must be finite with shape (n, 3, 3).")
    identities = np.einsum("nij,nkj->nik", rotations, rotations)
    determinants = np.linalg.det(rotations)
    if not np.allclose(
        identities, np.eye(3), rtol=0.0, atol=1.0e-12
    ) or not np.allclose(
        determinants,
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Molecular rotations must be proper orthogonal matrices.")
    result = np.array(rotations, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_mace_density(values: np.ndarray) -> np.ndarray:
    density = np.asarray(values, dtype=float)
    if (
        density.ndim != 2
        or density.shape[0] == 0
        or density.shape[1] != 4
        or not np.all(np.isfinite(density))
    ):
        raise ValueError(
            "MACE density coefficients must be finite with shape (n_atoms, 4)."
        )
    result = np.array(density, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0MolecularSolventReference:
    """One declared solvent molecular geometry and charge convention.

    The geometry and charges must later be content-addressed by a physical
    liquid asset.  This reference carries no claim that it is itself a
    complete solvent model or that its atomwise promolecule contains bonded
    electron-density information.
    """

    atomic_numbers: np.ndarray
    site_charges_e: np.ndarray
    reference_positions_bohr: np.ndarray
    provenance_label: str
    target_total_charge_e: float = 0.0

    def __post_init__(self) -> None:
        positions = np.asarray(self.reference_positions_bohr, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "Solvent reference positions must be finite with shape (n_sites, 3)."
            )
        atom_count = positions.shape[0]
        numbers = _atomic_numbers(
            self.atomic_numbers,
            atom_count=atom_count,
            name="Solvent atomic numbers",
        )
        charges = _immutable_array(
            self.site_charges_e,
            name="Solvent site charges",
            shape=(atom_count,),
        )
        target_charge = _finite_scalar(
            self.target_total_charge_e,
            name="Solvent target total charge",
        )
        total_charge = float(np.sum(charges))
        if abs(total_charge - target_charge) > 1.0e-12:
            raise ValueError(
                "Solvent site charges violate the declared molecular total charge."
            )
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(
            self,
            "site_charges_e",
            charges,
        )
        object.__setattr__(
            self,
            "reference_positions_bohr",
            _immutable_array(
                positions,
                name="Solvent reference positions",
                shape=(atom_count, 3),
            ),
        )
        object.__setattr__(
            self,
            "provenance_label",
            _nonempty_text(self.provenance_label, name="Solvent provenance label"),
        )
        object.__setattr__(self, "target_total_charge_e", target_charge)

    @property
    def site_count(self) -> int:
        """Return the number of atom sites in this molecular reference."""

        return int(self.atomic_numbers.shape[0])

    @property
    def total_charge_e(self) -> float:
        """Return the validated total molecular charge."""

        return float(np.sum(self.site_charges_e))


@dataclass(frozen=True)
class Route2V0MolecularConfigurations:
    """Rigid molecular translations and proper rotations in Bohr."""

    translations_bohr: np.ndarray
    rotations: np.ndarray

    def __post_init__(self) -> None:
        translations = np.asarray(self.translations_bohr, dtype=float)
        if (
            translations.ndim != 2
            or translations.shape[0] == 0
            or translations.shape[1] != 3
            or not np.all(np.isfinite(translations))
        ):
            raise ValueError(
                "Molecular translations must be finite with shape (n_configurations, 3)."
            )
        rotations = _proper_rotations(self.rotations)
        if translations.shape[0] != rotations.shape[0]:
            raise ValueError(
                "Molecular translations and rotations must have equal length."
            )
        object.__setattr__(
            self,
            "translations_bohr",
            _immutable_array(
                translations,
                name="Molecular translations",
                shape=translations.shape,
            ),
        )
        object.__setattr__(self, "rotations", rotations)

    @property
    def configuration_count(self) -> int:
        """Return the number of rigid solvent configurations."""

        return int(self.translations_bohr.shape[0])

    def site_positions_bohr(
        self,
        solvent: Route2V0MolecularSolventReference,
    ) -> np.ndarray:
        """Return global atomic site positions for every configuration."""

        if not isinstance(solvent, Route2V0MolecularSolventReference):
            raise TypeError("Molecular configurations require a solvent reference.")
        positions = (
            np.einsum(
                "nij,sj->nsi",
                self.rotations,
                solvent.reference_positions_bohr,
            )
            + self.translations_bohr[:, None, :]
        )
        positions.setflags(write=False)
        return positions


@dataclass(frozen=True)
class Route2V0FrozenMaceGaussianSource:
    """Validated unmodified zero-field MACE Gaussian electrostatic source."""

    density_coefficients: np.ndarray
    atom_positions_angstrom: np.ndarray
    target_total_charge_e: float = 0.0

    def __post_init__(self) -> None:
        density = _immutable_mace_density(self.density_coefficients)
        positions = _immutable_array(
            self.atom_positions_angstrom,
            name="MACE atom positions",
            shape=(density.shape[0], 3),
        )
        target_charge = _finite_scalar(
            self.target_total_charge_e,
            name="MACE target total charge",
        )
        total_charge = float(np.sum(density[:, 0]))
        if abs(total_charge - target_charge) > 1.0e-10:
            raise ValueError(
                "MACE Gaussian source violates its declared total-charge constraint."
            )
        object.__setattr__(self, "density_coefficients", density)
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(self, "target_total_charge_e", target_charge)

    @property
    def total_charge_e(self) -> float:
        """Return the validated MACE monopole sum."""

        return float(np.sum(self.density_coefficients[:, 0]))

    def potential_hartree_per_e(self, points_bohr: np.ndarray) -> np.ndarray:
        """Evaluate the declared all-space MACE Gaussian potential."""

        points = np.asarray(points_bohr, dtype=float)
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("MACE potential points must be finite with shape (n, 3).")
        result = np.asarray(
            gaussian_multipole_potential(
                points,
                self.atom_positions_angstrom,
                self.density_coefficients,
            ),
            dtype=float,
        )
        if result.shape != (points.shape[0],) or not np.all(np.isfinite(result)):
            raise RuntimeError("MACE Gaussian potential evaluation is invalid.")
        result = np.array(result, copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class Route2V0MolecularExternalPotential(Route2V0MolecularExternalPotentialContract):
    """One molecular external-potential evaluation with an explicit source split.

    ``pauli_repulsion_hartree`` is the exact discrete Thomas--Fermi
    nonadditive scalar evaluated separately for every rigid solvent
    configuration.  ``electrostatic_energy_hartree`` is the pairing of the
    same configuration's declared site charges with the MACE Gaussian
    potential.  Their sum is the only external potential returned here.
    """

    integration_grid: RegularCartesianGrid
    promolecular_table: Route2V0PromolecularDensityTable
    solute_atomic_numbers: np.ndarray
    solute_atom_positions_angstrom: np.ndarray
    mace_source: Route2V0FrozenMaceGaussianSource
    solvent: Route2V0MolecularSolventReference
    configurations: Route2V0MolecularConfigurations
    solute_reference_density_e_per_bohr3: np.ndarray
    pauli_repulsion_hartree: np.ndarray
    electrostatic_energy_hartree: np.ndarray
    external_potential_hartree: np.ndarray
    construction: str = V0_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION
    interaction_scope: str = V0_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE
    coordinate_derivative_policy: str = (
        V0_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY
    )

    def __post_init__(self) -> None:
        if not isinstance(self.integration_grid, RegularCartesianGrid):
            raise TypeError("Molecular external potential requires a regular grid.")
        if not isinstance(self.promolecular_table, Route2V0PromolecularDensityTable):
            raise TypeError(
                "Molecular external potential requires a promolecular table."
            )
        if not isinstance(self.mace_source, Route2V0FrozenMaceGaussianSource):
            raise TypeError("Molecular external potential requires a MACE source.")
        if not isinstance(self.solvent, Route2V0MolecularSolventReference):
            raise TypeError(
                "Molecular external potential requires a solvent reference."
            )
        if not isinstance(self.configurations, Route2V0MolecularConfigurations):
            raise TypeError("Molecular external potential requires configurations.")
        if self.construction != V0_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 molecular external-potential construction."
            )
        if self.interaction_scope != V0_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE:
            raise ValueError("Unsupported Route-2 molecular external-potential scope.")
        if (
            self.coordinate_derivative_policy
            != V0_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY
        ):
            raise ValueError(
                "Unsupported molecular external-potential derivative policy."
            )
        solute_positions = _immutable_array(
            self.solute_atom_positions_angstrom,
            name="Solute atom positions",
        )
        if (
            solute_positions.ndim != 2
            or solute_positions.shape[0] == 0
            or solute_positions.shape[1] != 3
        ):
            raise ValueError("Solute atom positions must have shape (n_atoms, 3).")
        solute_numbers = _atomic_numbers(
            self.solute_atomic_numbers,
            atom_count=solute_positions.shape[0],
            name="Solute atomic numbers",
        )
        if (
            self.mace_source.atom_positions_angstrom.shape != solute_positions.shape
            or not np.array_equal(
                self.mace_source.atom_positions_angstrom,
                solute_positions,
            )
        ):
            raise ValueError(
                "MACE Gaussian source geometry must equal the promolecular solute geometry."
            )
        density = _immutable_array(
            self.solute_reference_density_e_per_bohr3,
            name="Solute reference density",
            shape=self.integration_grid.shape,
        )
        if np.any(density < 0.0):
            raise ValueError("Solute reference density must be nonnegative.")
        count = self.configurations.configuration_count
        pauli = _immutable_array(
            self.pauli_repulsion_hartree,
            name="Pauli repulsion",
            shape=(count,),
        )
        electrostatic = _immutable_array(
            self.electrostatic_energy_hartree,
            name="Electrostatic energy",
            shape=(count,),
        )
        external = _immutable_array(
            self.external_potential_hartree,
            name="External potential",
            shape=(count,),
        )
        if np.any(pauli < -1.0e-13):
            raise ValueError("Thomas--Fermi Pauli repulsion must be nonnegative.")
        if not np.allclose(external, pauli + electrostatic, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Molecular external potential must equal Pauli plus electrostatic energy."
            )
        object.__setattr__(self, "solute_atomic_numbers", solute_numbers)
        object.__setattr__(self, "solute_atom_positions_angstrom", solute_positions)
        object.__setattr__(self, "solute_reference_density_e_per_bohr3", density)
        object.__setattr__(self, "pauli_repulsion_hartree", pauli)
        object.__setattr__(self, "electrostatic_energy_hartree", electrostatic)
        object.__setattr__(self, "external_potential_hartree", external)


def evaluate_route2_v0_molecular_external_potential(
    *,
    integration_grid: RegularCartesianGrid,
    promolecular_table: Route2V0PromolecularDensityTable,
    solute_atomic_numbers: np.ndarray,
    solute_atom_positions_angstrom: np.ndarray,
    mace_source: Route2V0FrozenMaceGaussianSource,
    solvent: Route2V0MolecularSolventReference,
    configurations: Route2V0MolecularConfigurations,
) -> Route2V0MolecularExternalPotential:
    """Evaluate the V0 molecular external potential at frozen configurations.

    The result is intentionally configuration-wise rather than a production
    translation/orientation grid.  This keeps the exact discrete scalar
    auditable and prevents a small direct quadrature control from being
    silently used as a scalable molecular-liquid backend.
    """

    if not isinstance(integration_grid, RegularCartesianGrid):
        raise TypeError("Molecular external potential requires a regular grid.")
    if not isinstance(promolecular_table, Route2V0PromolecularDensityTable):
        raise TypeError("Molecular external potential requires a promolecular table.")
    if not isinstance(mace_source, Route2V0FrozenMaceGaussianSource):
        raise TypeError("Molecular external potential requires a MACE source.")
    if not isinstance(solvent, Route2V0MolecularSolventReference):
        raise TypeError("Molecular external potential requires a solvent reference.")
    if not isinstance(configurations, Route2V0MolecularConfigurations):
        raise TypeError("Molecular external potential requires configurations.")
    solute_positions = np.asarray(solute_atom_positions_angstrom, dtype=float)
    if solute_positions.ndim != 2 or solute_positions.shape[1] != 3:
        raise ValueError("Solute atom positions must have shape (n_atoms, 3).")
    solute_numbers = _atomic_numbers(
        solute_atomic_numbers,
        atom_count=solute_positions.shape[0],
        name="Solute atomic numbers",
    )
    if (
        mace_source.atom_positions_angstrom.shape != solute_positions.shape
        or not np.array_equal(
            mace_source.atom_positions_angstrom,
            solute_positions,
        )
    ):
        raise ValueError(
            "MACE Gaussian source geometry must equal the promolecular solute geometry."
        )
    points = integration_grid.points_bohr()
    solute_density = promolecular_table.evaluate(
        points,
        solute_numbers,
        solute_positions,
    ).reshape(integration_grid.shape)
    site_positions = configurations.site_positions_bohr(solvent)
    electrostatic = np.einsum(
        "ns,s->n",
        mace_source.potential_hartree_per_e(site_positions.reshape((-1, 3))).reshape(
            (configurations.configuration_count, solvent.site_count)
        ),
        solvent.site_charges_e,
    )
    pauli = np.empty(configurations.configuration_count, dtype=float)
    for configuration_index, configuration_sites_bohr in enumerate(site_positions):
        solvent_density = promolecular_table.evaluate(
            points,
            solvent.atomic_numbers,
            configuration_sites_bohr * Bohr,
        ).reshape(integration_grid.shape)
        pauli[configuration_index] = Route2V0FrozenDensityPauliOverlap(
            grid=integration_grid,
            solute_electron_density_e_per_bohr3=solute_density,
            solvent_electron_density_e_per_bohr3=solvent_density,
        ).nonadditive_kinetic_energy_hartree()
    return Route2V0MolecularExternalPotential(
        integration_grid=integration_grid,
        promolecular_table=promolecular_table,
        solute_atomic_numbers=solute_numbers,
        solute_atom_positions_angstrom=solute_positions,
        mace_source=mace_source,
        solvent=solvent,
        configurations=configurations,
        solute_reference_density_e_per_bohr3=solute_density,
        pauli_repulsion_hartree=pauli,
        electrostatic_energy_hartree=electrostatic,
        external_potential_hartree=pauli + electrostatic,
    )


__all__ = [
    "Route2V0FrozenMaceGaussianSource",
    "Route2V0MolecularConfigurations",
    "Route2V0MolecularExternalPotential",
    "Route2V0MolecularSolventReference",
    "V0_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION",
    "V0_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY",
    "V0_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE",
    "evaluate_route2_v0_molecular_external_potential",
]
