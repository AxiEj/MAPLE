from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from maple.function.dispatcher.solvfe.protocol import RouteAProtocol

from .cavity_topology import (
    SphereUnionReport,
    validate_atom_sphere_supermolecule,
)


_FRAGMENT_ARRAY = "solvfe_fragment_id"


def _infer_covalent_fragments(atoms: Atoms) -> np.ndarray:
    positions = np.asarray(atoms.positions, dtype=float)
    numbers = np.asarray(atoms.numbers, dtype=int)
    adjacency = [set() for _ in atoms]
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            cutoff = 1.25 * (
                covalent_radii[numbers[left]] + covalent_radii[numbers[right]]
            )
            if np.linalg.norm(positions[left] - positions[right]) <= cutoff:
                adjacency[left].add(right)
                adjacency[right].add(left)

    labels = np.full(len(atoms), -1, dtype=int)
    fragment = 0
    for seed in range(len(atoms)):
        if labels[seed] >= 0:
            continue
        labels[seed] = fragment
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if labels[neighbor] >= 0:
                    continue
                labels[neighbor] = fragment
                frontier.append(neighbor)
        fragment += 1
    return labels


@dataclass(frozen=True)
class SupermoleculePCMRule:
    """Frozen v2 electrostatic-cavity construction and topology preflight."""

    base_radius_profile: str
    base_radii_A: Mapping[str, float]
    carbonyl_oxygen_element: str
    carbonyl_oxygen_atom_type: str
    carbonyl_oxygen_radius_A: float
    radius_scale: float
    tessera_area_A2: float
    minimum_added_sphere_radius_A: float
    minimum_bridge_overlap_A: float
    cds_policy: str

    @classmethod
    def from_protocol(
        cls,
        protocol: RouteAProtocol,
    ) -> "SupermoleculePCMRule":
        if protocol.data.get("protocol_version") != "2.0.0":
            raise ValueError(
                "Scaled supermolecule PCM requires Route A protocol v2.0.0."
            )
        raw = protocol.data.get("supermolecule_outer_cavity")
        if not isinstance(raw, Mapping):
            raise ValueError(
                "Route A protocol is missing supermolecule_outer_cavity."
            )
        override = raw.get("carbonyl_oxygen_override")
        radii = raw.get("base_radii_A")
        if not isinstance(override, Mapping) or not isinstance(radii, Mapping):
            raise ValueError("Route A v2 cavity radius contract is incomplete.")
        return cls(
            base_radius_profile=str(raw["base_radius_profile"]),
            base_radii_A={
                str(element): float(radius)
                for element, radius in radii.items()
            },
            carbonyl_oxygen_element=str(override["element"]),
            carbonyl_oxygen_atom_type=str(override["atom_type"]).lower(),
            carbonyl_oxygen_radius_A=float(override["radius_A"]),
            radius_scale=float(raw["radius_scale"]),
            tessera_area_A2=float(raw["tessera_area_A2"]),
            minimum_added_sphere_radius_A=float(
                raw["minimum_added_sphere_radius_A"]
            ),
            minimum_bridge_overlap_A=float(
                raw["minimum_bridge_overlap_A"]
            ),
            cds_policy=str(raw["cds_policy"]),
        )

    def radii_angstrom(self, atoms: Atoms) -> np.ndarray:
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise TypeError("Supermolecule PCM requires one non-empty ASE Atoms.")
        mol2 = atoms.info.get("mol2")
        atom_types = (
            mol2.get("atom_types") if isinstance(mol2, Mapping) else None
        )
        if atom_types is None or len(atom_types) != len(atoms):
            raise ValueError(
                "Supermolecule PCM requires one MOL2 atom type per atom."
            )

        base: list[float] = []
        for index, (symbol, atom_type) in enumerate(
            zip(atoms.get_chemical_symbols(), atom_types, strict=True)
        ):
            try:
                radius = float(self.base_radii_A[symbol])
            except KeyError as exc:
                raise ValueError(
                    "Supermolecule PCM has no frozen radius for "
                    f"atom {index} ({symbol})."
                ) from exc
            if (
                symbol == self.carbonyl_oxygen_element
                and str(atom_type).strip().lower()
                == self.carbonyl_oxygen_atom_type
            ):
                radius = self.carbonyl_oxygen_radius_A
            base.append(radius)
        return np.asarray(base, dtype=float) * self.radius_scale

    @staticmethod
    def fragment_ids(atoms: Atoms) -> np.ndarray:
        explicit = atoms.arrays.get(_FRAGMENT_ARRAY)
        if explicit is None:
            return _infer_covalent_fragments(atoms)
        values = np.asarray(explicit)
        if (
            values.shape != (len(atoms),)
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
        ):
            raise ValueError(
                f"atoms.arrays['{_FRAGMENT_ARRAY}'] must contain one "
                "non-negative integer per atom."
            )
        return values.astype(int, copy=True)

    def preflight(self, atoms: Atoms) -> SphereUnionReport:
        return validate_atom_sphere_supermolecule(
            np.asarray(atoms.positions, dtype=float),
            self.radii_angstrom(atoms),
            self.fragment_ids(atoms),
            minimum_bridge_overlap_A=self.minimum_bridge_overlap_A,
        )

    def as_request(self) -> dict[str, Any]:
        return {
            "base_radius_profile": self.base_radius_profile,
            "base_radii_A": dict(self.base_radii_A),
            "carbonyl_oxygen_override": {
                "element": self.carbonyl_oxygen_element,
                "atom_type": self.carbonyl_oxygen_atom_type,
                "radius_A": self.carbonyl_oxygen_radius_A,
            },
            "radius_scale": self.radius_scale,
            "tessera_area_A2": self.tessera_area_A2,
            "minimum_added_sphere_radius_A": (
                self.minimum_added_sphere_radius_A
            ),
            "minimum_bridge_overlap_A": self.minimum_bridge_overlap_A,
            "cds_policy": self.cds_policy,
        }


def sphere_union_report_dict(report: SphereUnionReport) -> dict[str, Any]:
    return {
        "component_count": report.component_count,
        "components": [list(component) for component in report.components],
        "fragment_component_count": report.fragment_component_count,
        "cross_fragment_edge_count": report.cross_fragment_edge_count,
        "maximum_cross_fragment_overlap_A": (
            report.maximum_cross_fragment_overlap_A
        ),
        "minimum_positive_cross_fragment_overlap_A": (
            report.minimum_positive_cross_fragment_overlap_A
        ),
        "fragment_bridge_bottleneck_A": (
            report.fragment_bridge_bottleneck_A
        ),
    }


__all__ = [
    "SupermoleculePCMRule",
    "sphere_union_report_dict",
]
