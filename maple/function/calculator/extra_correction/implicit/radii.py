"""Radius providers for fixed-charge implicit-solvation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .obc2_profile import (
    OBC2_FORMULA_MANIFEST_ID,
    obc2_profile_payload,
    verify_obc2_profile_manifest,
)
from .obc2_topology import OBC2Topology

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
    from .openmm_compat import customgbforces_module

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
        from .openmm_compat import openmm_version

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
        from .openmm_compat import openmm_version

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


class IndependentMbondi2RadiusProvider:
    """Narrow local mbondi2 radii and normal OBC screen assignment."""

    name = "maple-independent-mbondi2"

    def __init__(self) -> None:
        self.source_compatibility = verify_obc2_profile_manifest()
        payload = obc2_profile_payload()
        self._hydrogen = dict(payload["hydrogen_radii_angstrom"])
        self._radii = {
            int(number): float(value)
            for number, value in payload["radii_angstrom_by_atomic_number"].items()
        }
        self._screens = {
            int(number): float(value)
            for number, value in payload["screen_factors_by_atomic_number"].items()
        }
        self._default_radius = float(payload["default_radius_angstrom"])
        self._default_screen = float(payload["default_screen_factor"])

    def assign(self, topology: OBC2Topology) -> RadiusResult:
        if not isinstance(topology, OBC2Topology):
            raise TypeError("Independent mbondi2 assignment requires OBC2Topology.")
        radii_angstrom: list[float] = []
        screen_factors: list[float] = []
        hydrogen_on_nitrogen: list[int] = []
        for atom in topology.atoms:
            number = atom.atomic_number
            if number == 1:
                # OBC2Topology has already enforced exactly one neighbor.
                neighbor = topology.atoms[atom.bonded_indices[0]]
                if neighbor.atomic_number == 7:
                    radius = float(self._hydrogen["bonded_to_nitrogen"])
                    hydrogen_on_nitrogen.append(atom.index)
                else:
                    radius = float(self._hydrogen["otherwise"])
            else:
                radius = self._radii.get(number, self._default_radius)
            radii_angstrom.append(radius)
            screen_factors.append(self._screens.get(number, self._default_screen))

        radius_nm = np.asarray(radii_angstrom, dtype=np.float64) / 10.0
        parameters = np.column_stack(
            (radius_nm, np.asarray(screen_factors, dtype=np.float64))
        )
        # Mirror OpenMM's observable RadiusResult exactly: its Angstrom values
        # are obtained by converting the stored nanometer column back by 10.
        observable_radii = parameters[:, 0].copy() * 10.0
        parameters.setflags(write=False)
        observable_radii.setflags(write=False)
        return RadiusResult(
            profile="obc2-mbondi2",
            radii_angstrom=observable_radii,
            provider_parameters=parameters,
            provenance={
                "category": "radius",
                "name": self.name,
                "provider": self.name,
                "preparation_provider": self.name,
                "model": "obc2",
                "profile": "obc2-mbondi2",
                "radii": "mbondi2",
                "implementation": "maple-independent-mbondi2-v1",
                "formula_manifest_id": OBC2_FORMULA_MANIFEST_ID,
                "formula_source_sha256": self.source_compatibility["source_sha256"],
                "canonical_payload_sha256": self.source_compatibility[
                    "canonical_payload_sha256"
                ],
                "topology_sha256": topology.identity_sha256,
                "hydrogen_on_nitrogen_atom_indices": hydrogen_on_nitrogen,
                "portable_domain": "each atomic-number-1 atom has exactly one bonded neighbor",
                "runtime_openmm_required": False,
            },
        )
