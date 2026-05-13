"""Shared unit contract for official ASE calculators wrapped by MAPLE.

Official ASE calculators conventionally return energies in eV, forces in eV/Å,
and stress in eV/Å³. MAPLE's calculator layer uses Hartree for energies and
Hartree/Å for forces, while the MD/NPT pressure path intentionally consumes ASE
stress in eV/Å³. Keep that exception centralized here to avoid backend-specific
unit drift.
"""

import numpy as np

EV2HARTREE = 1.0 / 27.211386245988


def ase_properties_for_maple_request(properties) -> list:
    """Translate a MAPLE ASE-property request to the official calculator request."""
    requested = set(properties or ["energy"])
    official_properties = ["energy"]
    if "forces" in requested:
        official_properties.append("forces")
    if "stress" in requested:
        official_properties.append("stress")
    return official_properties


def copy_ase_results_to_maple_units(source_results: dict, target_results: dict) -> None:
    """Copy official ASE results into MAPLE calculator units.

    Converts:
    - energy/free_energy: eV -> Hartree
    - forces: eV/Å -> Hartree/Å

    Does not convert:
    - stress: remains ASE-native eV/Å³ for MAPLE's NPT pressure path.
    """
    if "energy" in source_results:
        energy = float(source_results["energy"]) * EV2HARTREE
        target_results["energy"] = energy
        target_results["free_energy"] = energy
    if "free_energy" in source_results:
        target_results["free_energy"] = float(source_results["free_energy"]) * EV2HARTREE
    if "forces" in source_results:
        target_results["forces"] = np.asarray(source_results["forces"], dtype=float) * EV2HARTREE
    if "stress" in source_results:
        target_results["stress"] = np.asarray(source_results["stress"], dtype=float)
