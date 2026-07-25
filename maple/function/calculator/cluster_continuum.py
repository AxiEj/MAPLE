"""Generic supermolecule--continuum composition for MAPLE calculators.

The inner calculator evaluates solute plus explicit first-shell solvent as one
complete supermolecule.  The outer provider evaluates a continuum correction
for that same atom list, and this wrapper adds the two contributions without
teaching individual MLIP backends about solvent models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .calculator_base import EV2HARTREE, atoms_has_pbc, numerical_hessian_from_atoms


_DERIVATIVE_PROPERTIES = {
    "force",
    "forces",
    "stress",
    "stresses",
    "virial",
    "virials",
    "hessian",
}


def _requested_properties(properties) -> list[str]:
    if properties is None:
        return ["energy"]
    return [str(prop).lower() for prop in properties]


def _finite_scalar(value, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite; got {result!r}.")
    return result


def _force_array(value, atoms, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    expected = (len(atoms), 3)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values.")
    return result


class ClusterContinuumCalculator(Calculator):
    """Add an outer-continuum correction to a MAPLE supermolecule calculator.

    Both components use MAPLE's established public units: Hartree for energy
    and Hartree/Angstrom for forces.
    """

    def __init__(self, inner_calculator: Calculator, outer_provider) -> None:
        super().__init__()
        self.inner_calculator = inner_calculator
        self.outer_provider = outer_provider

        inner_properties = {
            str(prop).lower()
            for prop in getattr(inner_calculator, "implemented_properties", ())
        }
        outer_properties = {
            str(prop).lower()
            for prop in getattr(outer_provider, "implemented_properties", ())
        }
        if "energy" not in inner_properties or "energy" not in outer_properties:
            raise ValueError(
                "Cluster-continuum composition requires energy from both the "
                "inner calculator and outer provider."
            )

        self.implemented_properties = ["energy", "free_energy"]
        if "forces" in inner_properties and "forces" in outer_properties:
            self.implemented_properties.append("forces")

    @property
    def supports_forces(self) -> bool:
        return "forces" in self.implemented_properties

    def _validate_request(self, properties: list[str]) -> None:
        requested = set(properties)
        if requested.intersection(_DERIVATIVE_PROPERTIES) and not self.supports_forces:
            raise NotImplementedError(
                "The selected outer continuum provider is energy-only; forces, "
                "stress, Hessians, and HVPs are unavailable."
            )
        unsupported = requested.difference({"energy", "free_energy", "forces"})
        if unsupported:
            raise NotImplementedError(
                "Cluster-continuum composition supports energy and, when the "
                f"outer provider supplies it, forces; unsupported: {sorted(unsupported)}."
            )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        target_atoms = atoms if atoms is not None else getattr(self, "atoms", None)
        if target_atoms is None:
            raise ValueError(
                "ClusterContinuumCalculator.calculate requires an Atoms object."
            )
        if atoms_has_pbc(target_atoms):
            raise NotImplementedError(
                "Cluster-continuum solvation is non-periodic; remove PBC."
            )

        requested = _requested_properties(properties)
        self._validate_request(requested)
        super().calculate(target_atoms, requested, system_changes)

        inner_properties = ["energy"]
        if "forces" in requested:
            inner_properties.append("forces")
        self.inner_calculator.calculate(
            target_atoms,
            properties=inner_properties,
            system_changes=system_changes,
        )
        inner_results = dict(getattr(self.inner_calculator, "results", {}) or {})
        if "energy" not in inner_results:
            raise RuntimeError("The inner calculator did not return an energy.")

        outer_results = self.outer_provider.calculate(target_atoms, requested)
        if "energy" not in outer_results:
            raise RuntimeError("The outer continuum provider did not return an energy.")

        inner_energy = _finite_scalar(inner_results["energy"], name="inner energy")
        outer_energy = _finite_scalar(outer_results["energy"], name="outer correction")
        total_energy = inner_energy + outer_energy

        self.results = {
            "energy": total_energy,
            "free_energy": total_energy,
            "inner_energy": inner_energy,
            "outer_correction": outer_energy,
        }
        if "forces" in requested:
            if "forces" not in inner_results:
                raise RuntimeError("The inner calculator did not return forces.")
            if "forces" not in outer_results:
                raise RuntimeError("The outer continuum provider did not return forces.")
            inner_forces = _force_array(
                inner_results["forces"], target_atoms, name="inner forces"
            )
            outer_forces = _force_array(
                outer_results["forces"], target_atoms, name="outer forces"
            )
            self.results.update(
                {
                    "forces": inner_forces + outer_forces,
                    "inner_forces": inner_forces,
                    "outer_forces": outer_forces,
                }
            )
        return self.results

    def get_hessian(self, atoms, delta: float = 0.002) -> np.ndarray:
        if not self.supports_forces:
            raise NotImplementedError(
                "The selected outer continuum provider is energy-only; Hessians "
                "are unavailable."
            )
        return numerical_hessian_from_atoms(self, atoms, delta)

    def get_hvp(self, atoms, n: np.ndarray, delta: float = 0.002):
        """Finite-difference total forces and return MAPLE Dimer tensors."""
        if not self.supports_forces:
            raise NotImplementedError(
                "The selected outer continuum provider is energy-only; HVPs are "
                "unavailable."
            )

        direction = np.asarray(n, dtype=float).reshape(-1)
        if direction.size != 3 * len(atoms):
            raise ValueError(
                f"HVP direction must contain {3 * len(atoms)} values; "
                f"got {direction.size}."
            )

        positions = atoms.get_positions()
        plus = atoms.copy()
        plus.set_positions(positions + delta * direction.reshape(-1, 3))
        minus = atoms.copy()
        minus.set_positions(positions - delta * direction.reshape(-1, 3))

        self.calculate(plus, properties=["forces"], system_changes=all_changes)
        forces_plus = np.asarray(self.results["forces"], dtype=float).reshape(-1)
        self.calculate(minus, properties=["forces"], system_changes=all_changes)
        forces_minus = np.asarray(self.results["forces"], dtype=float).reshape(-1)
        hvp = -(forces_plus - forces_minus) / (2.0 * delta)

        self.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        forces = np.asarray(self.results["forces"], dtype=float).reshape(-1)
        energy = float(self.results["energy"])

        import torch

        return (
            torch.as_tensor(hvp, dtype=torch.float64),
            torch.as_tensor(forces, dtype=torch.float64),
            torch.as_tensor(energy, dtype=torch.float64),
        )


class GBPolarOuterProvider:
    """Existing experimental MAPLE GB-polar/QEq correction, energy only."""

    implemented_properties = ("energy",)

    def __init__(self, solvent: str, *, device="cpu") -> None:
        from .extra_correction import GBSA, QEqTorch

        self.solvent = str(solvent).lower()
        self._gb = GBSA(solvent=self.solvent, device=device)
        self._qeq = QEqTorch(device=device)

    @staticmethod
    def _total_charge(atoms) -> float:
        if "charge" in getattr(atoms, "info", {}):
            return float(atoms.info["charge"])
        charges = np.asarray(atoms.get_initial_charges(), dtype=float)
        if charges.size and np.all(np.isfinite(charges)):
            return float(charges.sum())
        return 0.0

    def calculate(self, atoms, properties):
        requested = set(_requested_properties(properties))
        if requested.intersection(_DERIVATIVE_PROPERTIES):
            raise NotImplementedError(
                "Experimental MAPLE GB-polar/QEq outer solvation is energy-only."
            )

        cluster = atoms.copy()
        cluster.atomic_charges = self._qeq(
            cluster, total_charge=self._total_charge(cluster)
        )
        energy, _ = self._gb.get_energy(cluster)
        return {"energy": _finite_scalar(energy.item(), name="GB-polar correction")}


class TBLiteALPBDeltaProvider:
    """GFN2-xTB/ALPB minus gas-phase GFN2-xTB for the full cluster."""

    implemented_properties = ("energy", "forces")

    def __init__(
        self,
        solvent: str,
        *,
        method: str = "GFN2-xTB",
        state: str = "gsolv",
        calculator_cls: Optional[type] = None,
    ) -> None:
        self.solvent = str(solvent).strip().lower()
        self.method = str(method)
        self.state = str(state).strip().lower()
        if calculator_cls is None:
            try:
                from tblite.ase import TBLite
            except ImportError as exc:
                raise ImportError(
                    "TBLite ALPB support is optional. Install a tblite Python "
                    "build with ASE support (for example: conda install "
                    "-c conda-forge tblite) before using provider=tblite."
                ) from exc
            calculator_cls = TBLite
        self._calculator_cls = calculator_cls
        self._calculator_cache: dict[tuple[float, int], tuple[Calculator, Calculator]] = {}

    @staticmethod
    def _charge_and_multiplicity(atoms) -> tuple[float, int]:
        charge = float(getattr(atoms, "info", {}).get("charge", 0.0))
        raw_mult = float(getattr(atoms, "info", {}).get("mult", 1))
        if not raw_mult.is_integer() or raw_mult < 1:
            raise ValueError(
                "TBLite ALPB requires atoms.info['mult'] to be a positive integer."
            )
        return charge, int(raw_mult)

    def _calculators_for(self, atoms) -> tuple[Calculator, Calculator]:
        charge, multiplicity = self._charge_and_multiplicity(atoms)
        key = (charge, multiplicity)
        if key not in self._calculator_cache:
            common = {
                "method": self.method,
                "charge": charge,
                "multiplicity": multiplicity,
                "cache_api": True,
                "verbosity": 0,
            }
            vacuum = self._calculator_cls(**common)
            solvated = self._calculator_cls(
                **common,
                solvation=("alpb", self.solvent, self.state),
            )
            self._calculator_cache[key] = (vacuum, solvated)
        return self._calculator_cache[key]

    @staticmethod
    def _evaluate(calculator, atoms, *, forces: bool):
        structure = atoms.copy()
        structure.calc = calculator
        energy = float(structure.get_potential_energy())
        result = {"energy": energy}
        if forces:
            result["forces"] = np.asarray(structure.get_forces(), dtype=float)
        return result

    def calculate(self, atoms, properties):
        if atoms_has_pbc(atoms):
            raise NotImplementedError(
                "TBLite ALPB cluster-continuum correction is non-periodic."
            )
        requested = set(_requested_properties(properties))
        unsupported = requested.difference({"energy", "free_energy", "forces"})
        if unsupported:
            raise NotImplementedError(
                f"TBLite ALPB outer correction does not provide {sorted(unsupported)}."
            )

        need_forces = "forces" in requested
        vacuum_calc, solvated_calc = self._calculators_for(atoms)
        vacuum = self._evaluate(vacuum_calc, atoms, forces=need_forces)
        solvated = self._evaluate(solvated_calc, atoms, forces=need_forces)

        result = {
            "energy": (solvated["energy"] - vacuum["energy"]) * EV2HARTREE
        }
        if need_forces:
            result["forces"] = (
                solvated["forces"] - vacuum["forces"]
            ) * EV2HARTREE
        return result


def build_outer_solvent_provider(
    *,
    method: str,
    solvent: str,
    device="cpu",
    provider: Optional[str] = None,
):
    method = str(method).strip().lower()
    provider_name = None if provider is None else str(provider).strip().lower()
    if method == "gbsa":
        if provider_name not in {
            None,
            "",
            "none",
            "null",
            "false",
            "0",
            "maple",
        }:
            raise ValueError(
                "method=gbsa uses MAPLE's built-in experimental GB-polar provider; "
                "omit provider or use provider=maple."
            )
        return GBPolarOuterProvider(solvent, device=device)
    if method == "alpb":
        if provider_name != "tblite":
            raise ValueError("method=alpb currently requires provider=tblite.")
        return TBLiteALPBDeltaProvider(solvent)
    raise ValueError(f"Unsupported outer solvation method: {method!r}.")
