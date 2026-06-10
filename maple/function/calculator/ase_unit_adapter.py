"""Explicit unit adapter for attaching a raw ASE calculator to MAPLE MD.

A raw ASE calculator returns energy in eV and forces in eV/Å, but the MAPLE MD
layer consumes energy as Hartree and forces as Hartree/Å (``units.forces_au``
multiplies by ``HA_PER_ANG_TO_AU`` unconditionally).  Attaching a raw calculator
directly would be silently mis-scaled, and simply *stamping*
``maple_energy_unit="Ha"`` onto an eV calculator would be a false contract.

This adapter is the only sanctioned bridge: the user declares the calculator's
*source* units (from a fixed whitelist), the adapter performs the real numerical
conversion into the MAPLE contract, and only then declares the contract.  So a raw
calculator enters MD through an explicit, audited conversion rather than an
unchecked attribute claim.

References
----------
ASE unit convention (eV / eV·Å): https://ase-lib.org/ase/units.html
CODATA Hartree (``EV2HARTREE``): see ``_ase_unit_contract``.
"""

from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ._ase_unit_contract import (
    ASE_STRESS_UNIT,
    EV2HARTREE,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
    SUPPORTED_ENERGY_UNITS,
    SUPPORTED_FORCE_UNITS,
    SUPPORTED_STRESS_UNITS,
)


class ASEUnitAdapter(Calculator):
    """Wrap a raw ASE calculator into the MAPLE MD unit contract by real conversion.

    Energy/forces declared as eV / eV·Å are multiplied by ``EV2HARTREE`` to Hartree /
    Hartree·Å; declared as Ha / Ha·Å they pass through unchanged.  Stress is passed
    through in eV/Å³ (the MAPLE stress-unit contract).  Only whitelisted source units
    are accepted — a free-form unit string is rejected rather than trusted.

    PBC-MD capability and the neighbor cutoff are *not* assumed: pass
    ``pbc_md_supported=True`` and ``neighbor_cutoff_A=...`` explicitly to admit the
    wrapped calculator to periodic MD, just as a native MAPLE calculator would declare
    them.  (Without them the calculator is usable for non-periodic NVE/NVT only.)
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        calc: Calculator,
        *,
        energy_unit: str = "eV",
        force_unit: str = "eV/A",
        stress_unit: str = "eV/A^3",
        pbc_md_supported: Optional[bool] = None,
        stress_supported: Optional[bool] = None,
        neighbor_cutoff_A: Optional[float] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__()
        if energy_unit not in SUPPORTED_ENERGY_UNITS:
            raise ValueError(
                f"Unsupported energy_unit {energy_unit!r}; choose one of {SUPPORTED_ENERGY_UNITS}."
            )
        if force_unit not in SUPPORTED_FORCE_UNITS:
            raise ValueError(
                f"Unsupported force_unit {force_unit!r}; choose one of {SUPPORTED_FORCE_UNITS}."
            )
        if stress_unit not in SUPPORTED_STRESS_UNITS:
            raise ValueError(
                f"Unsupported stress_unit {stress_unit!r}; choose one of {SUPPORTED_STRESS_UNITS}."
            )

        self._calc = calc
        # Real numerical conversion factors into the MAPLE contract (NOT a relabel).
        self._energy_factor = EV2HARTREE if energy_unit == "eV" else 1.0
        self._force_factor = EV2HARTREE if force_unit == "eV/A" else 1.0

        # Declared MAPLE MD contract (post-conversion): always Ha / Ha·Å / eV·Å³.
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT
        self.maple_stress_unit = ASE_STRESS_UNIT
        self.maple_model_name = model_name or (
            f"ase-adapter({getattr(calc, 'name', type(calc).__name__)})"
        )
        self.maple_model_options = {
            "source_energy_unit": energy_unit,
            "source_force_unit": force_unit,
            "source_stress_unit": stress_unit,
            "wrapped": type(calc).__name__,
        }

        # PBC support / cutoff are explicit assertions, not inherited silently.
        if pbc_md_supported is not None:
            self.maple_pbc_md_supported = bool(pbc_md_supported)
        elif getattr(calc, "maple_pbc_md_supported", None) is not None:
            self.maple_pbc_md_supported = bool(getattr(calc, "maple_pbc_md_supported"))
        if stress_supported is not None:
            self.maple_stress_supported = bool(stress_supported)
        if neighbor_cutoff_A is not None:
            self.maple_neighbor_cutoff = float(neighbor_cutoff_A)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = float(self._calc.get_potential_energy(atoms)) * self._energy_factor
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        if "forces" in properties:
            self.results["forces"] = (
                np.asarray(self._calc.get_forces(atoms), dtype=float) * self._force_factor
            )
        if "stress" in properties:
            # Stress is consumed in eV/Å³ by the pressure code; pass through unchanged.
            self.results["stress"] = np.asarray(self._calc.get_stress(atoms), dtype=float)


def wrap_ase_calculator(
    calc: Calculator,
    *,
    energy_unit: str = "eV",
    force_unit: str = "eV/A",
    stress_unit: str = "eV/A^3",
    pbc_md_supported: Optional[bool] = None,
    stress_supported: Optional[bool] = None,
    neighbor_cutoff_A: Optional[float] = None,
    model_name: Optional[str] = None,
) -> ASEUnitAdapter:
    """Return an :class:`ASEUnitAdapter` wrapping ``calc`` for MAPLE MD.

    Use this to attach a raw ASE calculator (e.g. EMT, LennardJones) to MAPLE MD:
    the adapter converts its declared source units into the MAPLE contract and
    declares the contract, so the MD admission gate accepts it without silent
    mis-scaling.
    """
    return ASEUnitAdapter(
        calc,
        energy_unit=energy_unit,
        force_unit=force_unit,
        stress_unit=stress_unit,
        pbc_md_supported=pbc_md_supported,
        stress_supported=stress_supported,
        neighbor_cutoff_A=neighbor_cutoff_A,
        model_name=model_name,
    )
