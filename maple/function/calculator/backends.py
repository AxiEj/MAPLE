"""Orthogonal backend contracts for potentials, solvation, and alchemy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .calculator_base import numerical_hessian_from_atoms
from .extra_correction.implicit.result import SolvationResult
from .model_capabilities import (
    CombinationValidator,
    ModelCapabilities,
    SolvationCapabilities,
)


@dataclass(frozen=True)
class PotentialResult:
    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray | None = None
    hessian_hartree_per_angstrom2: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


class PotentialBackend(ABC):
    """A complete potential-energy surface in MAPLE's internal Hartree units."""

    capabilities: ModelCapabilities

    @abstractmethod
    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        need_hessian: bool = False,
    ) -> PotentialResult:
        raise NotImplementedError


class SolvationBackend(ABC):
    """An additive solvent PMF in MAPLE's internal Hartree units."""

    capabilities: SolvationCapabilities

    @abstractmethod
    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        raise NotImplementedError


@dataclass(frozen=True)
class AlchemicalResult:
    energy_hartree: float
    dU_dlambda_hartree: Mapping[str, float]
    provenance: Mapping[str, Any] = field(default_factory=dict)


class AlchemicalBackend(ABC):
    """A lambda-conditioned energy surface for TI/BAR/MBAR protocols."""

    capabilities: ModelCapabilities

    @abstractmethod
    def evaluate_alchemical(
        self,
        atoms,
        lambda_state: Mapping[str, float],
    ) -> AlchemicalResult:
        raise NotImplementedError


class CompositeCalculator(Calculator):
    """Generic ``U_base + W_solv`` calculator with capability-first gates.

    This is intentionally separate from native solution-phase potentials and
    alchemical protocols.  It is the reusable composition surface for future
    potential adapters; existing ``CalcABC`` calculators may continue using
    their current in-place composition path during migration.
    """

    implemented_properties = ["energy", "free_energy", "forces", "hessian"]

    def __init__(
        self,
        potential: PotentialBackend,
        solvation: SolvationBackend,
    ) -> None:
        super().__init__()
        self.potential = potential
        self.solvation = solvation
        self.frequency_type: str | None = None

    def _validate(self, *, need_forces: bool, need_hessian: bool) -> None:
        task = "frequency" if need_hessian else "opt" if need_forces else "sp"
        profile = CombinationValidator(
            model_capabilities=self.potential.capabilities,
            solvation_capabilities=self.solvation.capabilities,
            implicit="additive",
            hessian_mode="numerical" if need_hessian else None,
            task=task,
        ).validate()
        self.frequency_type = None if profile is None else profile.frequency_type

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        properties = tuple(str(item).lower() for item in properties)
        need_hessian = "hessian" in properties
        need_forces = need_hessian or "forces" in properties
        self._validate(need_forces=need_forces, need_hessian=need_hessian)

        base = self.potential.evaluate(
            atoms,
            need_forces=need_forces,
            need_hessian=False,
        )
        solvent = self.solvation.evaluate(
            atoms,
            need_forces=need_forces,
            calculator=self,
        )
        energy = float(base.energy_hartree) + float(solvent.energy_hartree)
        results: dict[str, Any] = {
            "energy": energy,
            "free_energy": energy,
            "solvation": {
                "energy_hartree": float(solvent.energy_hartree),
                "gas_energy_hartree": float(base.energy_hartree),
                "combined_energy_hartree": energy,
                "components_hartree": dict(solvent.components_hartree),
                "provenance": dict(solvent.provenance),
                "frequency_type": self.frequency_type,
                "ase_free_energy_is_thermochemical_gibbs": False,
            },
            "potential_provenance": dict(base.provenance),
        }
        if need_forces:
            if (
                base.forces_hartree_per_angstrom is None
                or solvent.forces_hartree_per_angstrom is None
            ):
                raise NotImplementedError(
                    "Composite force evaluation requires forces from both backends."
                )
            results["forces"] = np.asarray(
                base.forces_hartree_per_angstrom, dtype=float
            ) + np.asarray(solvent.forces_hartree_per_angstrom, dtype=float)
        self.results = results

        if need_hessian:
            self.results["hessian"] = numerical_hessian_from_atoms(self, atoms)
