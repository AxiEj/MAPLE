"""Shared chemical-domain validation for Route-2 solvation providers."""

from __future__ import annotations

import numpy as np


SUPPORTED_ELEMENTS = frozenset(
    {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
)
FORMALLY_CHARGED_TRIPOS_TYPES = frozenset({"n.4", "c.cat", "o.co2"})
MIN_MOLECULAR_MASS_DA = 16.0
MAX_MOLECULAR_MASS_DA = 500.0


def validate_route2_domain(atoms) -> None:
    """Require one neutral, closed-shell, non-periodic organic molecule."""

    if atoms is None or len(atoms) == 0:
        raise ValueError("Route 2 requires one non-empty molecule.")
    symbols = tuple(atoms.get_chemical_symbols())
    unsupported = sorted(set(symbols).difference(SUPPORTED_ELEMENTS))
    if unsupported:
        raise ValueError(
            "Route 2 supports H/C/N/O/F/P/S/Cl/Br/I only; unsupported elements: "
            + ", ".join(unsupported)
            + "."
        )
    molecular_mass = float(np.sum(atoms.get_masses()))
    if not MIN_MOLECULAR_MASS_DA <= molecular_mass <= MAX_MOLECULAR_MASS_DA:
        raise ValueError(
            "Route 2 v1 is validated for neutral organics from 16 to 500 Da; "
            f"received {molecular_mass:.6f} Da."
        )
    try:
        charge = float(atoms.info.get("charge", 0))
        multiplicity_value = float(atoms.info.get("mult", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Route 2 requires numeric charge=0 and multiplicity=1 metadata."
        ) from exc
    if not multiplicity_value.is_integer():
        raise ValueError("Route 2 multiplicity metadata must be an integer.")
    multiplicity = int(multiplicity_value)
    if charge != 0.0 or multiplicity != 1:
        raise ValueError(
            "Route 2 v1 supports neutral closed-shell molecules only "
            f"(received charge={charge:g}, multiplicity={multiplicity})."
        )
    mol2 = atoms.info.get("mol2")
    if isinstance(mol2, dict):
        atom_types = {
            str(atom_type).strip().lower()
            for atom_type in mol2.get("atom_types", ())
        }
        charged_markers = sorted(
            atom_types.intersection(FORMALLY_CHARGED_TRIPOS_TYPES)
        )
        if charged_markers:
            raise ValueError(
                "Route 2 v1 excludes salts and zwitterions; the MOL2 uses "
                "formally charged Tripos atom type(s): "
                + ", ".join(charged_markers)
                + "."
            )
    if np.any(atoms.get_pbc()):
        raise ValueError("Route 2 SMD is non-periodic.")


__all__ = [
    "FORMALLY_CHARGED_TRIPOS_TYPES",
    "MAX_MOLECULAR_MASS_DA",
    "MIN_MOLECULAR_MASS_DA",
    "SUPPORTED_ELEMENTS",
    "validate_route2_domain",
]
