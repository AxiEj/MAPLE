"""Fail-closed all-atom solvent-model sources for Route-2 V0.

This module freezes a molecular *site model* before it is permitted to enter
the Route-2 V0 liquid pipeline.  It intentionally sits below a frozen
``rism1d`` asset: a source record contains only a named all-atom molecule,
its rigid geometry, point charges, and Lennard-Jones parameters.  It does not
choose a bulk density or dielectric constant, run 1D-RISM, define a
solute--solvent short-range functional, or report a solvation free energy.

The distinction matters.  A complete all-atom molecular source is useful
evidence, but it is not yet a physical liquid endpoint.  In particular, this
module must not become a back door for a united-atom solvent, a virtual site,
or a source whose parameters were selected from target-solvation labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import atomic_masses, chemical_symbols

from .route2_v0_rism_molecular_source import AMBER_ELECTROSTATIC_CHARGE_SCALE

V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_CONSTRUCTION = (
    "route2-v0-all-atom-solvent-model-source-v1"
)
V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS = (
    "all-atom-model-source-only-not-liquid-or-accuracy-admitted"
)
V0_ALL_ATOM_SOLVENT_MODEL_REQUIRED_NO_TARGET_KEYS = frozenset(
    {
        "post_training",
        "fine_tuning",
        "experimental_solvation_fit",
        "map_or_uq_calibration",
        "target_solvation_labels_used",
    }
)
_ROOT_KEYS = frozenset(
    {
        "construction",
        "status",
        "claim_boundary",
        "solvent_id",
        "model",
        "molecular_weight_g_mol",
        "source",
        "site_types",
        "no_target_policy",
        "not_claimed",
    }
)
_MODEL_KEYS = frozenset({"family", "identifier"})
_SOURCE_KEYS = frozenset(
    {
        "document_url",
        "document_sha256",
        "source_locator",
        "retrieved_utc",
    }
)
_SITE_TYPE_KEYS = frozenset(
    {
        "label",
        "atomic_number",
        "mass_amu",
        "charge_e",
        "sigma_angstrom",
        "epsilon_kcal_per_mol",
        "positions_angstrom",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOLVENT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SITE_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9]{0,3}")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _strict_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys.")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} keys differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}."
        )


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be finite.")
    try:
        result = float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _integer(value: object, *, name: str) -> int:
    result = _finite(value, name=name)
    integer = int(result)
    if result != integer:
        raise ValueError(f"{name} must be an integer.")
    return integer


def _digest(value: object, *, name: str) -> str:
    result = _nonempty(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return result


def _immutable_positions(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    try:
        positions = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite Cartesian positions.") from exc
    if (
        positions.ndim != 2
        or positions.shape[0] < 1
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(f"{name} must have finite shape (n, 3) with n >= 1.")
    result = np.array(positions, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    result = tuple(_nonempty(item, name=f"{name} item") for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _format_integers(values: Sequence[int]) -> str:
    return "".join(f"{value:8d}" for value in values)


def _format_site_names(values: Sequence[str]) -> str:
    return "".join(f"{value:<4}" for value in values).rstrip()


def _format_reals(values: Sequence[float]) -> str:
    lines: list[str] = []
    for start in range(0, len(values), 5):
        lines.append("".join(f"{value:16.8e}" for value in values[start : start + 5]))
    return "\n".join(lines)


@dataclass(frozen=True)
class Route2V0AllAtomSolventSiteType:
    """One all-atom site type with one or more explicit atom positions."""

    label: str
    atomic_number: int
    mass_amu: float
    charge_e: float
    sigma_angstrom: float
    epsilon_kcal_per_mol: float
    positions_angstrom: np.ndarray

    def __post_init__(self) -> None:
        label = _nonempty(self.label, name="All-atom solvent site label")
        if _SITE_LABEL.fullmatch(label) is None:
            raise ValueError(
                "All-atom solvent site labels must be 1--4 alphanumeric "
                "characters beginning with a letter."
            )
        number = _integer(self.atomic_number, name=f"{label} atomic number")
        if number <= 0 or number >= len(atomic_masses):
            raise ValueError(f"{label} atomic number must identify a real element.")
        element = chemical_symbols[number]
        if not label.casefold().startswith(element.casefold()):
            raise ValueError(
                f"{label} site label must begin with the element symbol {element!r}."
            )
        mass = _positive(self.mass_amu, name=f"{label} mass")
        reference_mass = float(atomic_masses[number])
        if abs(mass - reference_mass) > max(0.11, 0.06 * reference_mass):
            raise ValueError(
                f"{label} mass is incompatible with atomic number {number}; "
                "united-atom and virtual-site sources are not admitted."
            )
        charge = _finite(self.charge_e, name=f"{label} charge")
        sigma = _positive(self.sigma_angstrom, name=f"{label} LJ sigma")
        epsilon = _positive(self.epsilon_kcal_per_mol, name=f"{label} LJ epsilon")
        positions = _immutable_positions(
            self.positions_angstrom,
            name=f"{label} positions",
        )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "atomic_number", number)
        object.__setattr__(self, "mass_amu", mass)
        object.__setattr__(self, "charge_e", charge)
        object.__setattr__(self, "sigma_angstrom", sigma)
        object.__setattr__(self, "epsilon_kcal_per_mol", epsilon)
        object.__setattr__(self, "positions_angstrom", positions)

    @property
    def multiplicity(self) -> int:
        """Return the number of explicit all-atom sites of this type."""

        return int(self.positions_angstrom.shape[0])

    @property
    def rmin_half_angstrom(self) -> float:
        """Return Amber's ``Rmin/2`` representation of this LJ sigma."""

        return 0.5 * math.pow(2.0, 1.0 / 6.0) * self.sigma_angstrom


@dataclass(frozen=True)
class Route2V0AllAtomSolventModelSource:
    """A frozen all-atom molecular model, explicitly below liquid admission."""

    solvent_id: str
    model_family: str
    model_identifier: str
    molecular_weight_g_mol: float
    document_url: str
    document_sha256: str
    source_locator: str
    retrieved_utc: str
    site_types: tuple[Route2V0AllAtomSolventSiteType, ...]
    claim_boundary: str
    not_claimed: tuple[str, ...]
    construction: str = V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_CONSTRUCTION
    status: str = V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS

    def __post_init__(self) -> None:
        solvent_id = _nonempty(self.solvent_id, name="Solvent identifier")
        if _SOLVENT_ID.fullmatch(solvent_id) is None:
            raise ValueError("Solvent identifier must be a lowercase stable slug.")
        family = _nonempty(self.model_family, name="Solvent model family")
        identifier = _nonempty(self.model_identifier, name="Solvent model identifier")
        weight = _positive(self.molecular_weight_g_mol, name="Molecular weight")
        url = _nonempty(self.document_url, name="Source document URL")
        if not url.startswith("https://"):
            raise ValueError("Source document URL must use HTTPS.")
        digest = _digest(self.document_sha256, name="Source document SHA-256")
        locator = _nonempty(self.source_locator, name="Source locator")
        retrieved = _nonempty(self.retrieved_utc, name="Source retrieval timestamp")
        if _UTC_TIMESTAMP.fullmatch(retrieved) is None:
            raise ValueError("Source retrieval timestamp must be UTC RFC3339 seconds.")
        site_types = tuple(self.site_types)
        if not site_types or any(
            not isinstance(site, Route2V0AllAtomSolventSiteType) for site in site_types
        ):
            raise TypeError("All-atom solvent model must contain site-type records.")
        labels = tuple(site.label for site in site_types)
        if len(set(labels)) != len(labels):
            raise ValueError("All-atom solvent model site labels must be unique.")
        total_mass = sum(site.mass_amu * site.multiplicity for site in site_types)
        if not math.isclose(total_mass, weight, rel_tol=5.0e-4, abs_tol=0.0):
            raise ValueError(
                "All-atom solvent model molecular weight must agree with the "
                "explicit atomic masses."
            )
        total_charge = sum(site.charge_e * site.multiplicity for site in site_types)
        if abs(total_charge) > 1.0e-12:
            raise ValueError(
                "All-atom solvent-model source must be electrically neutral."
            )
        positions = np.concatenate([site.positions_angstrom for site in site_types])
        distances = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
            axis=-1,
        )
        np.fill_diagonal(distances, math.inf)
        if float(np.min(distances)) <= 1.0e-10:
            raise ValueError("All-atom solvent model cannot contain coincident sites.")
        claim = _nonempty(self.claim_boundary, name="Model-source claim boundary")
        if not isinstance(self.not_claimed, (tuple, list)):
            raise TypeError("Model-source non-claims must be a sequence.")
        not_claimed = tuple(
            _nonempty(value, name="Model-source non-claim")
            for value in self.not_claimed
        )
        if not not_claimed:
            raise ValueError("Model-source non-claims must not be empty.")
        if self.construction != V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 all-atom solvent-model construction.")
        if self.status != V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS:
            raise ValueError("All-atom solvent model must remain source-only.")
        object.__setattr__(self, "solvent_id", solvent_id)
        object.__setattr__(self, "model_family", family)
        object.__setattr__(self, "model_identifier", identifier)
        object.__setattr__(self, "molecular_weight_g_mol", weight)
        object.__setattr__(self, "document_url", url)
        object.__setattr__(self, "document_sha256", digest)
        object.__setattr__(self, "source_locator", locator)
        object.__setattr__(self, "retrieved_utc", retrieved)
        object.__setattr__(self, "site_types", site_types)
        object.__setattr__(self, "claim_boundary", claim)
        object.__setattr__(self, "not_claimed", not_claimed)

    @property
    def atom_count(self) -> int:
        """Return the number of explicit atomic sites in one molecule."""

        return sum(site.multiplicity for site in self.site_types)

    @property
    def total_charge_e(self) -> float:
        """Return the validated source charge sum."""

        return sum(site.charge_e * site.multiplicity for site in self.site_types)

    def to_amber_mdl_text(self, *, title: str | None = None) -> str:
        """Render the frozen model as deterministic all-atom AMBER MDL text.

        This is a unit-preserving serialization only.  It does not select a
        density, dielectric response, closure, or RISM numerical setting.
        """

        label = title or self.model_identifier
        label = _nonempty(label, name="AMBER MDL title")
        atom_types: list[int] = []
        coordinates: list[float] = []
        for index, site in enumerate(self.site_types, start=1):
            atom_types.extend([index] * site.multiplicity)
            coordinates.extend(float(value) for value in site.positions_angstrom.flat)
        lines = [
            "%VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/70  00:00:00",
            "%FLAG TITLE",
            "%FORMAT(20a4)",
            label,
            "%FLAG POINTERS",
            "%FORMAT(10I8)",
            _format_integers((self.atom_count, len(self.site_types))),
            "%FLAG ATMTYP",
            "%FORMAT(10I8)",
            _format_integers(atom_types),
            "%FLAG ATMNAME",
            "%FORMAT(20a4)",
            _format_site_names(tuple(site.label for site in self.site_types)),
            "%FLAG MASS",
            "%FORMAT(5e16.8)",
            _format_reals(tuple(site.mass_amu for site in self.site_types)),
            "%FLAG CHG",
            "%FORMAT(5e16.8)",
            _format_reals(
                tuple(
                    site.charge_e * AMBER_ELECTROSTATIC_CHARGE_SCALE
                    for site in self.site_types
                )
            ),
            "%FLAG LJEPSILON",
            "%FORMAT(5e16.8)",
            _format_reals(tuple(site.epsilon_kcal_per_mol for site in self.site_types)),
            "%FLAG LJSIGMA",
            "%FORMAT(5e16.8)",
            _format_reals(tuple(site.rmin_half_angstrom for site in self.site_types)),
            "%FLAG MULTI",
            "%FORMAT(10I8)",
            _format_integers(tuple(site.multiplicity for site in self.site_types)),
            "%FLAG COORD",
            "%FORMAT(5e16.8)",
            _format_reals(coordinates),
        ]
        return "\n".join(lines) + "\n"

    def amber_mdl_sha256(self, *, title: str | None = None) -> str:
        """Return the content digest of the deterministic MDL serialization."""

        return hashlib.sha256(
            self.to_amber_mdl_text(title=title).encode("utf-8")
        ).hexdigest()


def _site_type_from_json(value: object) -> Route2V0AllAtomSolventSiteType:
    entry = _strict_mapping(value, name="All-atom solvent site type")
    _strict_keys(entry, _SITE_TYPE_KEYS, name="All-atom solvent site type")
    positions = entry["positions_angstrom"]
    if not isinstance(positions, list):
        raise TypeError("All-atom solvent site positions must be a list.")
    return Route2V0AllAtomSolventSiteType(
        label=_nonempty(entry["label"], name="Site type label"),
        atomic_number=_integer(entry["atomic_number"], name="Site type atomic number"),
        mass_amu=_positive(entry["mass_amu"], name="Site type mass"),
        charge_e=_finite(entry["charge_e"], name="Site type charge"),
        sigma_angstrom=_positive(entry["sigma_angstrom"], name="Site type sigma"),
        epsilon_kcal_per_mol=_positive(
            entry["epsilon_kcal_per_mol"],
            name="Site type epsilon",
        ),
        positions_angstrom=np.asarray(positions),
    )


def parse_route2_v0_all_atom_solvent_model_source(
    text: str,
) -> Route2V0AllAtomSolventModelSource:
    """Parse one strict, source-only Route-2 V0 all-atom model record."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("All-atom solvent-model source must be valid JSON.") from exc
    root = _strict_mapping(payload, name="All-atom solvent-model source")
    _strict_keys(root, _ROOT_KEYS, name="All-atom solvent-model source")
    model = _strict_mapping(root["model"], name="All-atom solvent model")
    _strict_keys(model, _MODEL_KEYS, name="All-atom solvent model")
    source = _strict_mapping(root["source"], name="All-atom solvent source")
    _strict_keys(source, _SOURCE_KEYS, name="All-atom solvent source")
    policy = _strict_mapping(root["no_target_policy"], name="All-atom no-target policy")
    _strict_keys(
        policy,
        V0_ALL_ATOM_SOLVENT_MODEL_REQUIRED_NO_TARGET_KEYS,
        name="All-atom no-target policy",
    )
    if any(value is not False for value in policy.values()):
        raise ValueError("All-atom solvent-model no-target policy flags must be false.")
    sites = root["site_types"]
    if not isinstance(sites, list):
        raise TypeError("All-atom solvent-model site_types must be a list.")
    not_claimed = _string_sequence(root["not_claimed"], name="Model-source not_claimed")
    return Route2V0AllAtomSolventModelSource(
        solvent_id=_nonempty(root["solvent_id"], name="Solvent identifier"),
        model_family=_nonempty(model["family"], name="Model family"),
        model_identifier=_nonempty(model["identifier"], name="Model identifier"),
        molecular_weight_g_mol=_positive(
            root["molecular_weight_g_mol"],
            name="Molecular weight",
        ),
        document_url=_nonempty(source["document_url"], name="Source document URL"),
        document_sha256=_digest(
            source["document_sha256"],
            name="Source document SHA-256",
        ),
        source_locator=_nonempty(source["source_locator"], name="Source locator"),
        retrieved_utc=_nonempty(
            source["retrieved_utc"], name="Source retrieval timestamp"
        ),
        site_types=tuple(_site_type_from_json(site) for site in sites),
        claim_boundary=_nonempty(
            root["claim_boundary"], name="Model-source claim boundary"
        ),
        not_claimed=not_claimed,
    )


def load_route2_v0_all_atom_solvent_model_source(
    path: str | Path,
) -> Route2V0AllAtomSolventModelSource:
    """Load one strict source-only Route-2 V0 all-atom solvent model."""

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"All-atom solvent-model source is absent: {source_path}.")
    return parse_route2_v0_all_atom_solvent_model_source(
        source_path.read_text(encoding="utf-8")
    )


__all__ = [
    "V0_ALL_ATOM_SOLVENT_MODEL_REQUIRED_NO_TARGET_KEYS",
    "V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_CONSTRUCTION",
    "V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS",
    "Route2V0AllAtomSolventModelSource",
    "Route2V0AllAtomSolventSiteType",
    "load_route2_v0_all_atom_solvent_model_source",
    "parse_route2_v0_all_atom_solvent_model_source",
]
