"""Nonpolar providers used by fixed-charge implicit-solvation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol


def _openmm_lcpo_parameters(topology, atoms):
    from openmm import unit
    from openmm.app.internal import lcpo

    parameters = list(lcpo.getLCPOParamsTopology(topology))
    metadata = atoms.info.get("mol2", {})
    atom_types = list(metadata.get("atom_types") or [])
    if len(atom_types) != topology.getNumAtoms():
        raise ValueError("LCPO parameter assignment requires one MOL2 atom type per atom.")

    typed_atom_indices: list[int] = []
    adjusted_atom_indices: list[int] = []
    amber_oxygen_types = {
        "o": "O_sp2_1",
        "o2": "O_carboxylate",
    }
    for index, atom_type in enumerate(atom_types):
        parameter_key = amber_oxygen_types.get(str(atom_type).lower())
        if parameter_key is None:
            continue
        raw = lcpo.LCPO_PARAMETERS[parameter_key]
        expected = (
            raw[0] * unit.angstrom,
            raw[1],
            raw[2],
            raw[3],
            raw[4] / unit.angstrom**2,
        )
        typed_atom_indices.append(index)
        current_values = tuple(
            float(value._value if hasattr(value, "_value") else value)
            for value in parameters[index]
        )
        expected_values = tuple(
            float(value._value if hasattr(value, "_value") else value)
            for value in expected
        )
        if current_values != expected_values:
            adjusted_atom_indices.append(index)
        parameters[index] = expected
    return parameters, {
        "parameter_source": (
            "openmm.app.internal.lcpo.getLCPOParamsTopology with exact "
            "Amber/GAFF o and o2 atom-type overrides"
        ),
        "typed_atom_indices": typed_atom_indices,
        "adjusted_atom_indices": adjusted_atom_indices,
    }


class NonpolarProvider(Protocol):
    name: str

    @property
    def component_properties(self) -> frozenset[str]:
        """Return properties supplied by this component, not the parent solver."""

    @property
    def provenance(self) -> dict[str, object]:
        """Return an auditable provider record."""


@dataclass(frozen=True)
class OpenMMNonpolarProvider:
    """Configure an OpenMM ACE, LCPO, or disabled nonpolar term."""

    selection: str

    def __post_init__(self) -> None:
        normalized = str(self.selection).lower()
        if normalized not in {"ace", "lcpo", "none"}:
            raise ValueError("OpenMM GB nonpolar must be ace, lcpo, or none.")
        object.__setattr__(self, "selection", normalized)

    @property
    def name(self) -> str:
        return f"openmm-{self.selection}"

    @property
    def enabled(self) -> bool:
        return self.selection != "none"

    @property
    def component_properties(self) -> frozenset[str]:
        if not self.enabled:
            return frozenset()
        return frozenset({"energy", "forces"})

    @property
    def custom_gb_sa(self) -> str | None:
        return "ACE" if self.selection == "ace" else None

    def install_separate_force(self, system, topology, atoms=None) -> dict | None:
        if self.selection != "lcpo":
            return None
        try:
            from openmm.app.internal import lcpo

            if atoms is None:
                raise ValueError(
                    "nonpolar=lcpo requires MOL2 atom types for audited parameter "
                    "assignment."
                )
            parameters, audit = _openmm_lcpo_parameters(topology, atoms)
            lcpo.addLCPOForce(
                system,
                parameters,
                usePeriodic=False,
            )
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "nonpolar=lcpo requires OpenMM>=8.5 with LCPOForce support."
            ) from exc
        return audit

    @property
    def provenance(self) -> dict[str, object]:
        implementation = {
            "none": "disabled",
            "ace": "openmm.app.internal.customgbforces CustomGBForce ACE term",
            "lcpo": "openmm.app.internal.lcpo.LCPOForce",
        }[self.selection]
        return {
            "category": "nonpolar",
            "name": self.name,
            "provider": "openmm",
            "profile": self.selection,
            "implementation": implementation,
            "component_properties": sorted(self.component_properties),
        }


@dataclass(frozen=True)
class APBSSASANonpolarProvider:
    """APBS APOLAR SASA provider for the locked generic PB profile."""

    probe_radius: float = 1.4
    surface_tension: float = 0.105
    pressure: float = 0.0
    component_properties: ClassVar[frozenset[str]] = frozenset({"energy"})
    name: ClassVar[str] = "apbs-sasa"

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_radius", float(self.probe_radius))
        object.__setattr__(self, "surface_tension", float(self.surface_tension))
        object.__setattr__(self, "pressure", float(self.pressure))
        if self.probe_radius < 0:
            raise ValueError("APBS probe_radius must be non-negative.")
        if self.surface_tension < 0 or self.pressure < 0:
            raise ValueError("APBS surface_tension and pressure must be non-negative.")

    def render_input_block(self) -> str:
        return f"""\
apolar name nonpolar
    mol 1
    srfm sacc
    srad {self.probe_radius:.6f}
    swin 0.3
    sdens 10.0
    gamma {self.surface_tension:.8f}
    press {self.pressure:.8f}
    bconc 0.0
    dpos 0.05
    grid 0.5 0.5 0.5
    temp 298.15
    calcenergy total
    calcforce no
end"""

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "category": "nonpolar",
            "name": self.name,
            "provider": "apbs",
            "profile": "sasa",
            "implementation": "APBS APOLAR solvent-accessible-surface term",
            "probe_radius_angstrom": self.probe_radius,
            "surface_tension_kj_mol_a2": self.surface_tension,
            "pressure_kj_mol_a3": self.pressure,
            "bulk_solvent_density_a3": 0.0,
            "component_properties": sorted(self.component_properties),
        }
