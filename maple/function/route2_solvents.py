"""Immutable solvent specifications for experimental Route-2 SMD profiles.

The values are the eight SMD solvent descriptors exposed by the tested
PySCF 2.13.1 ``solvent_db`` and trace back to the Minnesota solvation
database.  This module is data-only: continuum equations, source
representations, cavity policies, and non-polar models remain separate axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


MNSOL_DATABASE_DOI = "10.13020/3eks-j059"
MNSOL_DATABASE_URL = (
    "https://conservancy.umn.edu/items/0ae1c7f7-154f-4ab6-9a5c-"
    "ba13e03546f3"
)
PYSCF_SMD_VERSION = "2.13.1"


@dataclass(frozen=True)
class SMDSolventDescriptors:
    """The eight solvent descriptors consumed by upstream SMD."""

    refractive_index: float
    refractive_index_25c: float
    hydrogen_bond_acidity: float
    hydrogen_bond_basicity: float
    surface_tension: float
    dielectric: float
    aromatic_carbon_fraction: float
    electronegative_halogen_fraction: float

    def as_pyscf_tuple(self) -> tuple[float, ...]:
        return (
            self.refractive_index,
            self.refractive_index_25c,
            self.hydrogen_bond_acidity,
            self.hydrogen_bond_basicity,
            self.surface_tension,
            self.dielectric,
            self.aromatic_carbon_fraction,
            self.electronegative_halogen_fraction,
        )


@dataclass(frozen=True)
class Route2SolventSpec:
    """One canonical Route-2 solvent identity and its upstream mapping."""

    name: str
    pyscf_smd_name: str
    mnsol_name: str
    descriptors: SMDSolventDescriptors
    aliases: frozenset[str] = frozenset()
    descriptor_source: str = "PySCF 2.13.1 SMD solvent_db"
    experimental_dataset_doi: str = MNSOL_DATABASE_DOI
    experimental_dataset_url: str = MNSOL_DATABASE_URL


def _descriptors(values: tuple[float, ...]) -> SMDSolventDescriptors:
    return SMDSolventDescriptors(*values)


_SOLVENT_SPECS = {
    "water": Route2SolventSpec(
        name="water",
        pyscf_smd_name="water",
        mnsol_name="water",
        descriptors=_descriptors(
            (1.3328, 1.3323, 0.82, 0.35, -1.0, 78.355, -1.0, -1.0)
        ),
        aliases=frozenset({"h2o"}),
    ),
    "methanol": Route2SolventSpec(
        name="methanol",
        pyscf_smd_name="methanol",
        mnsol_name="methanol",
        descriptors=_descriptors(
            (1.3288, 1.3265, 0.43, 0.47, 31.77, 32.613, 0.0, 0.0)
        ),
        aliases=frozenset({"meoh"}),
    ),
    "ethanol": Route2SolventSpec(
        name="ethanol",
        pyscf_smd_name="ethanol",
        mnsol_name="ethanol",
        descriptors=_descriptors(
            (1.3611, 1.3593, 0.37, 0.48, 31.62, 24.852, 0.0, 0.0)
        ),
        aliases=frozenset({"etoh"}),
    ),
    "acetonitrile": Route2SolventSpec(
        name="acetonitrile",
        pyscf_smd_name="acetonitrile",
        mnsol_name="acetonitrile",
        descriptors=_descriptors(
            (1.3442, 1.3416, 0.07, 0.32, 41.25, 35.688, 0.0, 0.0)
        ),
        aliases=frozenset({"mecn"}),
    ),
    "dimethylsulfoxide": Route2SolventSpec(
        name="dimethylsulfoxide",
        pyscf_smd_name="dimethylsulfoxide",
        mnsol_name="dimethylsulfoxide",
        descriptors=_descriptors(
            (1.4783, 1.4783, 0.0, 0.88, 61.78, 46.826, 0.0, 0.0)
        ),
        aliases=frozenset({"dmso", "dimethyl sulfoxide"}),
    ),
    "dimethylformamide": Route2SolventSpec(
        name="dimethylformamide",
        pyscf_smd_name="N,N-dimethylformamide",
        mnsol_name="dimethylformamide",
        descriptors=_descriptors(
            (1.4305, 1.4280, 0.0, 0.74, 49.56, 37.219, 0.0, 0.0)
        ),
        aliases=frozenset(
            {"dmf", "N,N-dimethylformamide", "dimethyl formamide"}
        ),
    ),
    "tetrahydrofuran": Route2SolventSpec(
        name="tetrahydrofuran",
        pyscf_smd_name="tetrahydrofuran",
        mnsol_name="tetrahydrofuran",
        descriptors=_descriptors(
            (1.4050, 1.4044, 0.0, 0.48, 39.44, 7.4257, 0.0, 0.0)
        ),
        aliases=frozenset({"thf"}),
    ),
    "chloroform": Route2SolventSpec(
        name="chloroform",
        pyscf_smd_name="chloroform",
        mnsol_name="chloroform",
        descriptors=_descriptors(
            (1.4459, 1.4431, 0.15, 0.02, 38.39, 4.7113, 0.0, 0.75)
        ),
    ),
    "dichloromethane": Route2SolventSpec(
        name="dichloromethane",
        pyscf_smd_name="dichloromethane",
        mnsol_name="methylenechloride",
        descriptors=_descriptors(
            (1.4242, 1.4212, 0.10, 0.05, 39.15, 8.93, 0.0, 0.667)
        ),
        aliases=frozenset({"dcm", "methylene chloride"}),
    ),
    "toluene": Route2SolventSpec(
        name="toluene",
        pyscf_smd_name="toluene",
        mnsol_name="toluene",
        descriptors=_descriptors(
            (1.4961, 1.4936, 0.0, 0.14, 40.2, 2.3741, 0.857, 0.0)
        ),
    ),
    "hexane": Route2SolventSpec(
        name="hexane",
        pyscf_smd_name="n-hexane",
        mnsol_name="hexane",
        descriptors=_descriptors(
            (1.3749, 1.3722, 0.0, 0.0, 25.75, 1.8819, 0.0, 0.0)
        ),
        aliases=frozenset({"n-hexane"}),
    ),
}

ROUTE2_SMD_SOLVENTS: Mapping[str, Route2SolventSpec] = MappingProxyType(
    _SOLVENT_SPECS
)
SUPPORTED_ROUTE2_SMD_SOLVENTS = frozenset(ROUTE2_SMD_SOLVENTS)


def _compact_solvent_name(value: object) -> str:
    normalized = str(value).strip().lower()
    return "".join(character for character in normalized if character.isalnum())


_SOLVENT_ALIASES = {
    _compact_solvent_name(alias): spec.name
    for spec in ROUTE2_SMD_SOLVENTS.values()
    for alias in {spec.name, spec.pyscf_smd_name, *spec.aliases}
}


def normalize_route2_solvent_name(solvent: object) -> str:
    compact = _compact_solvent_name(solvent)
    try:
        return _SOLVENT_ALIASES[compact]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_ROUTE2_SMD_SOLVENTS))
        raise ValueError(
            f"Unsupported Route 2 SMD solvent: {solvent}. "
            f"Supported solvents: {supported}."
        ) from exc


def route2_solvent_spec(solvent: object) -> Route2SolventSpec:
    return ROUTE2_SMD_SOLVENTS[normalize_route2_solvent_name(solvent)]


__all__ = [
    "MNSOL_DATABASE_DOI",
    "MNSOL_DATABASE_URL",
    "PYSCF_SMD_VERSION",
    "ROUTE2_SMD_SOLVENTS",
    "Route2SolventSpec",
    "SMDSolventDescriptors",
    "SUPPORTED_ROUTE2_SMD_SOLVENTS",
    "normalize_route2_solvent_name",
    "route2_solvent_spec",
]
