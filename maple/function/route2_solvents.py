"""Immutable solvent specifications for Route-2 SMD profiles.

The full catalog is a data-only snapshot of the eight SMD solvent descriptors
exposed by PySCF 2.13.1. Runtime lookup does not import PySCF. The historical
11-solvent Route-2 profile scope remains available separately so expanding the
catalog does not silently expand existing scientific profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Mapping

from .smd_solvent_data import (
    PYSCF_SMD_SOURCE_VERSION,
    SMD_SOLVENT_DESCRIPTOR_SNAPSHOT,
)

MNSOL_DATABASE_DOI = "10.13020/3eks-j059"
MNSOL_DATABASE_URL = (
    "https://conservancy.umn.edu/items/0ae1c7f7-154f-4ab6-9a5c-" "ba13e03546f3"
)
PYSCF_SMD_VERSION = PYSCF_SMD_SOURCE_VERSION


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
    mnsol_name: str | None
    descriptors: SMDSolventDescriptors
    aliases: frozenset[str] = frozenset()
    descriptor_source: str = "PySCF 2.13.1 SMD solvent_db"
    experimental_dataset_doi: str = MNSOL_DATABASE_DOI
    experimental_dataset_url: str = MNSOL_DATABASE_URL


_HISTORICAL_ROUTE2_NAMES = (
    "water",
    "methanol",
    "ethanol",
    "acetonitrile",
    "dimethylsulfoxide",
    "dimethylformamide",
    "tetrahydrofuran",
    "chloroform",
    "dichloromethane",
    "toluene",
    "hexane",
)

_HISTORICAL_MNSOL_NAMES = {
    "water": "water",
    "methanol": "methanol",
    "ethanol": "ethanol",
    "acetonitrile": "acetonitrile",
    "dimethylsulfoxide": "dimethylsulfoxide",
    "dimethylformamide": "dimethylformamide",
    "tetrahydrofuran": "tetrahydrofuran",
    "chloroform": "chloroform",
    "dichloromethane": "methylenechloride",
    "toluene": "toluene",
    "hexane": "hexane",
}

_EXPLICIT_ALIASES = {
    "water": frozenset({"h2o"}),
    "methanol": frozenset({"meoh"}),
    "ethanol": frozenset({"etoh"}),
    "acetonitrile": frozenset({"mecn"}),
    "dimethylsulfoxide": frozenset({"dmso", "dimethyl sulfoxide"}),
    "dimethylformamide": frozenset(
        {"dmf", "N,N-dimethylformamide", "dimethyl formamide"}
    ),
    "tetrahydrofuran": frozenset({"thf"}),
    "dichloromethane": frozenset({"dcm", "methylene chloride"}),
    "hexane": frozenset({"n-hexane"}),
    "ethylethanoate": frozenset({"ethyl acetate"}),
    "methylethanoate": frozenset({"methyl acetate"}),
    "1-2-ethanediol": frozenset({"ethylene glycol"}),
    "2-propanol": frozenset({"isopropanol", "isopropyl alcohol"}),
    "1-octanol": frozenset({"n-octanol"}),
}

_PARSER_SAFE_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def _build_solvent_specs() -> dict[str, Route2SolventSpec]:
    specs: dict[str, Route2SolventSpec] = {}
    for canonical_name, pyscf_name, values in SMD_SOLVENT_DESCRIPTOR_SNAPSHOT:
        if not _PARSER_SAFE_NAME.fullmatch(canonical_name):
            raise RuntimeError(f"Invalid canonical SMD solvent name: {canonical_name}")
        if canonical_name in specs:
            raise RuntimeError(f"Duplicate canonical SMD solvent: {canonical_name}")
        if len(values) != 8 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Invalid SMD descriptors for {pyscf_name}")
        if all(value == 0 for value in values):
            raise RuntimeError(
                f"All-zero SMD sentinel leaked into catalog: {pyscf_name}"
            )

        specs[canonical_name] = Route2SolventSpec(
            name=canonical_name,
            pyscf_smd_name=pyscf_name,
            mnsol_name=_HISTORICAL_MNSOL_NAMES.get(canonical_name),
            descriptors=SMDSolventDescriptors(*values),
            aliases=_EXPLICIT_ALIASES.get(canonical_name, frozenset()),
        )
    return specs


_SMD_SOLVENT_SPECS = _build_solvent_specs()

# Full versioned SMD descriptor catalog. This is the lookup surface used by
# normalize_route2_solvent_name() and route2_solvent_spec().
SMD_SOLVENTS: Mapping[str, Route2SolventSpec] = MappingProxyType(_SMD_SOLVENT_SPECS)
SUPPORTED_SMD_SOLVENTS = frozenset(SMD_SOLVENTS)

# Historical profile scope. Do not replace this with the full catalog without
# separately validating and versioning every profile that consumes it.
ROUTE2_SMD_SOLVENTS: Mapping[str, Route2SolventSpec] = MappingProxyType(
    {name: SMD_SOLVENTS[name] for name in _HISTORICAL_ROUTE2_NAMES}
)
SUPPORTED_ROUTE2_SMD_SOLVENTS = frozenset(ROUTE2_SMD_SOLVENTS)


def _compact_solvent_name(value: object) -> str:
    normalized = str(value).strip().lower()
    return "".join(character for character in normalized if character.isalnum())


def _build_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for spec in SMD_SOLVENTS.values():
        for alias in {spec.name, spec.pyscf_smd_name, *spec.aliases}:
            compact = _compact_solvent_name(alias)
            if not compact:
                raise RuntimeError(f"Empty SMD solvent alias for {spec.name}")
            previous = aliases.setdefault(compact, spec.name)
            if previous != spec.name:
                raise RuntimeError(
                    f"Ambiguous SMD solvent alias {alias!r}: "
                    f"{previous!r} versus {spec.name!r}"
                )
    return aliases


_SOLVENT_ALIASES = _build_aliases()


def normalize_route2_solvent_name(solvent: object) -> str:
    """Return the parser-safe canonical name for any catalog solvent alias."""

    compact = _compact_solvent_name(solvent)
    try:
        return _SOLVENT_ALIASES[compact]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Route 2 SMD solvent: {solvent}. "
            f"The catalog contains {len(SUPPORTED_SMD_SOLVENTS)} solvents."
        ) from exc


def route2_solvent_spec(solvent: object) -> Route2SolventSpec:
    """Resolve a solvent against the full vendored SMD catalog."""

    return SMD_SOLVENTS[normalize_route2_solvent_name(solvent)]


__all__ = [
    "MNSOL_DATABASE_DOI",
    "MNSOL_DATABASE_URL",
    "PYSCF_SMD_VERSION",
    "ROUTE2_SMD_SOLVENTS",
    "SMD_SOLVENTS",
    "Route2SolventSpec",
    "SMDSolventDescriptors",
    "SUPPORTED_ROUTE2_SMD_SOLVENTS",
    "SUPPORTED_SMD_SOLVENTS",
    "normalize_route2_solvent_name",
    "route2_solvent_spec",
]
