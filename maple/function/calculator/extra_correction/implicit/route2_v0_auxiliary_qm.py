"""Auxiliary-QM stationary-difference ledger for the Route-2 V0-AQ control.

The frozen MACE-POLAR checkpoint remains the gas-phase potential.  A separate
auxiliary QM calculation may contribute only its *stationary gas-to-solvent
difference*.  It is consequently not a MACE density, does not drive MACE
field features, and cannot repair the rejected learned fixed point.

This module deliberately contains no QM or PCM implementation.  It validates
and combines source-bound stationary states emitted by an isolated auxiliary
runtime.  Keeping that runtime separate avoids turning its electron density
into an undocumented MAPLE/MACE interface and keeps the no-training V0-AQ-E
control usable as a fixed-geometry scalar/gradient ledger only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from ase.units import Bohr, Hartree

V0_AUXILIARY_QM_ELECTROSTATIC_CONSTRUCTION = (
    "route2-v0-auxiliary-qm-electrostatic-difference-v1"
)


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _immutable_gradient(
    values: np.ndarray,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    gradient = np.asarray(values, dtype=float)
    if gradient.shape != (atom_count, 3) or not np.all(np.isfinite(gradient)):
        raise ValueError(
            f"{name} must be finite with shape ({atom_count}, 3); "
            f"received {gradient.shape}."
        )
    result = np.array(gradient, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _nonempty_text(value: str, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be nonempty.")
    return result


@dataclass(frozen=True)
class AuxiliaryQMProvenance:
    """Immutable identity shared by gas and solvated auxiliary QM states.

    The content hashes make the geometry and numerical setup explicit without
    storing a mutable runtime object in a result ledger.  A gas-to-solvent
    subtraction is rejected unless every field in this provenance agrees.
    """

    implementation: str
    electronic_structure: str
    basis: str
    reference: str
    charge: int
    spin: int
    geometry_sha256: str
    numerical_settings_sha256: str
    atom_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "implementation",
            "electronic_structure",
            "basis",
            "reference",
            "geometry_sha256",
            "numerical_settings_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty_text(getattr(self, field_name), name=field_name),
            )
        charge = int(self.charge)
        spin = int(self.spin)
        atom_count = int(self.atom_count)
        if atom_count <= 0:
            raise ValueError("Auxiliary QM atom count must be positive.")
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "atom_count", atom_count)


@dataclass(frozen=True)
class AuxiliaryQMStationaryState:
    """One converged gas or solvated auxiliary QM total-energy state.

    ``total_stationary_energy_hartree`` must be the full converged total energy
    of the declared phase.  An isolated PCM implementation component is not a
    valid input to this ledger.
    """

    phase: str
    provenance: AuxiliaryQMProvenance
    total_stationary_energy_hartree: float
    nuclear_gradient_hartree_per_bohr: np.ndarray
    orbital_gradient_inf: float
    scf_converged: bool

    def __post_init__(self) -> None:
        phase = _nonempty_text(self.phase, name="Auxiliary QM phase")
        if phase not in {"gas", "solvated"}:
            raise ValueError("Auxiliary QM phase must be either 'gas' or 'solvated'.")
        if not isinstance(self.provenance, AuxiliaryQMProvenance):
            raise TypeError("Auxiliary QM state requires AuxiliaryQMProvenance.")
        if not bool(self.scf_converged):
            raise ValueError("Auxiliary QM state must be SCF-converged.")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "total_stationary_energy_hartree",
            _finite_scalar(
                self.total_stationary_energy_hartree,
                name="Auxiliary QM total stationary energy",
            ),
        )
        object.__setattr__(
            self,
            "nuclear_gradient_hartree_per_bohr",
            _immutable_gradient(
                self.nuclear_gradient_hartree_per_bohr,
                atom_count=self.provenance.atom_count,
                name="Auxiliary QM nuclear gradient",
            ),
        )
        orbital_gradient = _finite_scalar(
            self.orbital_gradient_inf,
            name="Auxiliary QM orbital-gradient infinity norm",
        )
        if orbital_gradient < 0.0:
            raise ValueError("Auxiliary QM orbital-gradient infinity norm is negative.")
        object.__setattr__(self, "orbital_gradient_inf", orbital_gradient)


@dataclass(frozen=True)
class Route2V0AuxiliaryQMElectrostaticDifference:
    """The fixed-geometry V0-AQ-E composite scalar and its same-ledger gradient.

    This is exactly

    ``E_MACE,gas + (A_aux,solvated - A_aux,gas)``.

    It is intentionally not a total solvation free energy: no standard-state,
    cavity, dispersion, or empirical CDS contribution is accepted here.
    """

    mace_gas_energy_ev: float
    mace_gas_gradient_ev_per_angstrom: np.ndarray
    auxiliary_gas_state: AuxiliaryQMStationaryState
    auxiliary_solvated_state: AuxiliaryQMStationaryState
    auxiliary_energy_difference_hartree: float = field(init=False)
    auxiliary_gradient_difference_hartree_per_bohr: np.ndarray = field(init=False)
    composite_energy_ev: float = field(init=False)
    composite_gradient_ev_per_angstrom: np.ndarray = field(init=False)
    construction: str = V0_AUXILIARY_QM_ELECTROSTATIC_CONSTRUCTION

    def __post_init__(self) -> None:
        if self.construction != V0_AUXILIARY_QM_ELECTROSTATIC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0-AQ-E construction.")
        if not isinstance(self.auxiliary_gas_state, AuxiliaryQMStationaryState):
            raise TypeError("V0-AQ-E requires a gas auxiliary QM state.")
        if not isinstance(self.auxiliary_solvated_state, AuxiliaryQMStationaryState):
            raise TypeError("V0-AQ-E requires a solvated auxiliary QM state.")
        gas = self.auxiliary_gas_state
        solvated = self.auxiliary_solvated_state
        if gas.phase != "gas" or solvated.phase != "solvated":
            raise ValueError("V0-AQ-E requires one gas and one solvated state.")
        if gas.provenance != solvated.provenance:
            raise ValueError(
                "Auxiliary gas and solvated states must share identical "
                "method, geometry, and numerical provenance."
            )
        mace_energy = _finite_scalar(self.mace_gas_energy_ev, name="MACE gas energy")
        mace_gradient = _immutable_gradient(
            self.mace_gas_gradient_ev_per_angstrom,
            atom_count=gas.provenance.atom_count,
            name="MACE gas gradient",
        )
        auxiliary_energy = float(
            solvated.total_stationary_energy_hartree
            - gas.total_stationary_energy_hartree
        )
        auxiliary_gradient = np.array(
            solvated.nuclear_gradient_hartree_per_bohr
            - gas.nuclear_gradient_hartree_per_bohr,
            dtype=float,
            copy=True,
        )
        auxiliary_gradient.setflags(write=False)
        composite_gradient = np.array(
            mace_gradient + auxiliary_gradient * Hartree / Bohr,
            dtype=float,
            copy=True,
        )
        composite_gradient.setflags(write=False)
        object.__setattr__(self, "mace_gas_energy_ev", mace_energy)
        object.__setattr__(self, "mace_gas_gradient_ev_per_angstrom", mace_gradient)
        object.__setattr__(
            self,
            "auxiliary_energy_difference_hartree",
            auxiliary_energy,
        )
        object.__setattr__(
            self,
            "auxiliary_gradient_difference_hartree_per_bohr",
            auxiliary_gradient,
        )
        object.__setattr__(
            self,
            "composite_energy_ev",
            mace_energy + auxiliary_energy * Hartree,
        )
        object.__setattr__(
            self,
            "composite_gradient_ev_per_angstrom",
            composite_gradient,
        )

    @property
    def composite_forces_ev_per_angstrom(self) -> np.ndarray:
        """Return ``-dE/dR`` for callers that separately pass force gates."""

        result = np.array(-self.composite_gradient_ev_per_angstrom, copy=True)
        result.setflags(write=False)
        return result


__all__ = [
    "AuxiliaryQMProvenance",
    "AuxiliaryQMStationaryState",
    "Route2V0AuxiliaryQMElectrostaticDifference",
    "V0_AUXILIARY_QM_ELECTROSTATIC_CONSTRUCTION",
]
