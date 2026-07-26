"""Radius providers for fixed-charge implicit-solvation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .openmm_compat import customgbforces_module, openmm_version

GB_MODELS: dict[str, dict[str, Any]] = {
    "hct": {
        "class": "GBSAHCTForce",
        "igb": 1,
        "radii": "mbondi",
        "profile": "hct-mbondi",
    },
    "obc1": {
        "class": "GBSAOBC1Force",
        "igb": 2,
        "radii": "mbondi2",
        "profile": "obc1-mbondi2",
    },
    "obc2": {
        "class": "GBSAOBC2Force",
        "igb": 5,
        "radii": "mbondi2",
        "profile": "obc2-mbondi2",
    },
    "gbn": {
        "class": "GBSAGBnForce",
        "igb": 7,
        "radii": "bondi",
        "profile": "gbn-bondi",
    },
    "gbn2": {
        "class": "GBSAGBn2Force",
        "igb": 8,
        "radii": "mbondi3",
        "profile": "gbn2-mbondi3",
    },
}


@dataclass(frozen=True)
class RadiusResult:
    """Assigned radii plus any provider-native parameters needed at runtime."""

    profile: str
    radii_angstrom: np.ndarray
    provider_parameters: np.ndarray
    provenance: dict[str, Any]


class RadiusProvider(Protocol):
    name: str

    def assign(self, topology) -> RadiusResult:
        """Assign one radius and the provider-native parameters per atom."""
        ...


def _standard_parameters(topology, force_class: str) -> np.ndarray:
    customgbforces = customgbforces_module()
    force_cls = getattr(customgbforces, force_class)
    parameters = np.asarray(force_cls.getStandardParameters(topology), dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[0] != topology.getNumAtoms():
        raise ValueError(
            "OpenMM radius-provider parameter count does not match the atom count."
        )
    if parameters.shape[1] == 0 or not np.isfinite(parameters).all():
        raise ValueError("OpenMM radius provider returned invalid standard parameters.")
    radii_angstrom = parameters[:, 0] * 10.0
    if np.any(radii_angstrom <= 0.0):
        raise ValueError(
            "OpenMM radius provider returned a non-positive atomic radius."
        )
    return parameters


def _repair_small_molecule_mbondi3_oxygen_radii(
    topology,
    parameters: np.ndarray,
) -> list[int]:
    """Match Amber's generic-MOL mbondi3 oxygen assignment.

    OpenMM identifies carboxylate oxygens from connectivity alone, which also
    matches neutral ester carbonyl oxygens. Amber's mbondi3 rules only apply the
    1.4-A oxygen radius to named biomolecular Asp/Glu/C-terminal atoms; generic
    small-molecule residues retain the mbondi2 oxygen radius of 1.5 A.
    """
    adjusted: list[int] = []
    for atom in topology.atoms():
        if not atom.residue.name.startswith("MOL") or atom.element.symbol != "O":
            continue
        radius_nm = float(parameters[atom.index, 0])
        if np.isclose(radius_nm, 0.14, rtol=0.0, atol=1.0e-12):
            parameters[atom.index, 0] = 0.15
            adjusted.append(int(atom.index))
        elif not np.isclose(radius_nm, 0.15, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Unexpected OpenMM GBn2 oxygen radius for a synthetic small-molecule "
                f"residue: atom {atom.index} has {radius_nm * 10.0:.6f} A."
            )
    return adjusted


class OpenMMAmberGBRadiusProvider:
    """Amber radius/screen assignment locked to one OpenMM GB model."""

    name = "openmm-amber-gb-radii"

    def __init__(self, model: str):
        model = str(model).lower()
        if model not in GB_MODELS:
            raise ValueError(
                f"Unknown GB model {model!r}; choose hct, obc1, obc2, gbn, or gbn2."
            )
        self.model = model
        self.model_info = dict(GB_MODELS[model])

    def assign(self, topology) -> RadiusResult:
        parameters = _standard_parameters(topology, self.model_info["class"])
        adjusted_atom_indices: list[int] = []
        if self.model == "gbn2":
            adjusted_atom_indices = _repair_small_molecule_mbondi3_oxygen_radii(
                topology,
                parameters,
            )
        implementation = (
            "openmm.app.internal.customgbforces."
            f"{self.model_info['class']}.getStandardParameters"
        )
        if self.model == "gbn2":
            implementation += " + MAPLE generic-MOL mbondi3 oxygen parity repair"
        return RadiusResult(
            profile=self.model_info["profile"],
            radii_angstrom=parameters[:, 0].copy() * 10.0,
            provider_parameters=parameters.copy(),
            provenance={
                "category": "radius",
                "name": self.name,
                "provider": "openmm",
                "provider_version": openmm_version(),
                "model": self.model,
                "profile": self.model_info["profile"],
                "radii": self.model_info["radii"],
                "implementation": implementation,
                "parameter_adjustments": {
                    "generic_mol_mbondi3_oxygen_radius_angstrom": (
                        1.5 if self.model == "gbn2" else None
                    ),
                    "adjusted_atom_indices": adjusted_atom_indices,
                },
            },
        )


class OpenMMMbondi2RadiusProvider:
    """Generic mbondi2 radii sourced from OpenMM's upstream OBC-I assignment."""

    name = "openmm-mbondi2-radii"

    def assign(self, topology) -> RadiusResult:
        force_class = "GBSAOBC1Force"
        parameters = _standard_parameters(topology, force_class)
        return RadiusResult(
            profile="generic-mbondi2",
            radii_angstrom=parameters[:, 0].copy() * 10.0,
            provider_parameters=parameters.copy(),
            provenance={
                "category": "radius",
                "name": self.name,
                "provider": "openmm",
                "provider_version": openmm_version(),
                "profile": "generic-mbondi2",
                "radii": "mbondi2",
                "implementation": (
                    "openmm.app.internal.customgbforces."
                    f"{force_class}.getStandardParameters"
                ),
            },
        )
