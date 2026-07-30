"""Fail-closed frozen solvent assets for the Route-2 V0-FD-S path.

This module records *solvent-side* provenance only.  It verifies that a
declared molecular-liquid asset contains the exact bulk 1D-RISM files,
site-model source, canonical rigid molecular reference, short-range-
interaction source, thermodynamic output, and no-target-label provenance
statement that were frozen before target-solute scoring.  It never accepts a
solute topology, GAFF/AM1-BCC solute parameters, an empirical correction, or
a trained/fine-tuned response model.

Loading an asset does not construct a solute--solvent short-range potential,
map a production Cartesian kernel, minimize a liquid functional, or report a
solvation free energy.  Those later physical gates remain separate.  The
purpose here is narrower: missing, changed, or insufficiently documented
solvent inputs must fail before a future liquid backend can accidentally use
them as if they were a reproducible physical asset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase.units import Bohr

from .route2_v0_molecular_external_potential import (
    Route2V0MolecularSolventReference,
)
from .route2_v0_rism_bulk import (
    Route2V0RismBulkDirectCorrelation,
    load_rism1d_bulk_direct_correlation,
)
from .route2_v0_rism_generation_source import (
    Route2V0RismGenerationSource,
    load_route2_v0_rism_generation_source,
)
from .route2_v0_rism_molecular_source import (
    Route2V0MolecularRismSource,
    load_route2_v0_rism_molecular_source,
    same_labelled_rigid_geometry,
)
from .route2_v0_rism_short_range_source import (
    Route2V0RismShortRangeSource,
    load_route2_v0_rism_short_range_source,
)
from .route2_v0_rism_thermodynamic_source import (
    Route2V0RismSelfTest,
    load_route2_v0_rism_self_test,
)

V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION = "route2-v0-frozen-solvent-asset-v2"
V0_FROZEN_SOLVENT_ASSET_SCHEMA_VERSION = 2
V0_FROZEN_SOLVENT_MOLECULAR_REFERENCE_CONSTRUCTION = (
    "route2-v0-frozen-solvent-molecular-reference-v1"
)
V0_DEFAULT_SOLVENT_IDS = (
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
V0_REQUIRED_ASSET_FILE_ROLES = frozenset(
    {
        "site_model",
        "rism1d_input",
        "xvv",
        "cvv",
        "thermodynamic_output",
        "short_range_interaction",
        "provenance_statement",
    }
)
V0_REQUIRED_EXCLUDED_TARGET_LABEL_SETS = frozenset(
    {"mnsol", "freesolv", "development", "confirmation", "blind"}
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOLVENT_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _nonempty_string(value: object, *, name: str) -> str:
    """Return one nonempty string without silently coercing manifest data."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    """Return one JSON object with string keys."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object.")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    """Return one finite positive floating-point manifest value."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _strict_false(value: object, *, name: str) -> bool:
    """Require an explicit false policy flag rather than assume one."""

    if value is not False:
        raise ValueError(f"{name} must be explicitly false.")
    return False


def _same_rism_metadata(left: object, right: object) -> bool:
    """Return whether two parsed records describe the exact same XVV source."""

    scalars = (
        "site_names",
        "temperature_kelvin",
        "dielectric_constant",
        "coulomb_smear_angstrom",
        "radial_spacing_angstrom",
        "radial_point_count",
        "component_count",
    )
    arrays = (
        "site_multiplicity",
        "bulk_number_density_angstrom3",
        "site_charges_sqrt_kT_angstrom",
    )
    return bool(
        all(getattr(left, name) == getattr(right, name) for name in scalars)
        and all(
            np.array_equal(getattr(left, name), getattr(right, name)) for name in arrays
        )
    )


def _sha256(path: Path) -> str:
    """Return the content digest of one regular frozen-asset file."""

    if not path.is_file():
        raise ValueError(f"Frozen solvent asset file is absent: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_path(
    *,
    root: Path,
    relative_path: object,
    name: str,
) -> Path:
    """Resolve one manifest-local file while forbidding path escape."""

    text = _nonempty_string(relative_path, name=f"{name}.path")
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{name}.path must be a relative path inside the asset root.")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{name}.path must remain inside the asset root.") from exc
    return resolved


@dataclass(frozen=True)
class Route2V0FrozenSolventAssetFile:
    """One content-addressed source file needed by a frozen solvent asset."""

    role: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        role = _nonempty_string(self.role, name="Frozen solvent asset file role")
        path = Path(self.path).resolve()
        digest = _nonempty_string(self.sha256, name=f"{role} SHA-256").lower()
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"{role} SHA-256 must be a lowercase SHA-256 digest.")
        observed = _sha256(path)
        if observed != digest:
            raise ValueError(
                f"Frozen solvent asset hash mismatch for {role}: "
                f"expected {digest}, observed {observed}."
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", digest)

    def verify_integrity(self) -> None:
        """Recheck this file's content hash before a downstream physical use."""

        observed = _sha256(self.path)
        if observed != self.sha256:
            raise ValueError(
                f"Frozen solvent asset hash mismatch for {self.role}: "
                f"expected {self.sha256}, observed {observed}."
            )


@dataclass(frozen=True)
class Route2V0FrozenSolventMolecularReference:
    """Canonical rigid molecular reference declared by one frozen site model.

    A molecular liquid cannot be reconstructed from a solvent name or a bulk
    dielectric constant.  This record keeps the rigid geometry, atom identity,
    neutral site-charge convention, and per-atom RISM site-type mapping in the
    same manifest as the hash-locked ``site_model`` source.  The RISM ``QV``
    values retain their native units in the bulk asset and are therefore not
    compared to ``site_charges_e`` here.
    """

    atomic_numbers: np.ndarray
    site_charges_e: np.ndarray
    reference_positions_bohr: np.ndarray
    rism_site_type_names: tuple[str, ...]
    site_model_sha256: str
    target_total_charge_e: float = 0.0
    construction: str = V0_FROZEN_SOLVENT_MOLECULAR_REFERENCE_CONSTRUCTION
    _molecular_reference: Route2V0MolecularSolventReference = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        digest = _nonempty_string(
            self.site_model_sha256,
            name="Frozen solvent molecular-reference site-model SHA-256",
        ).lower()
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(
                "Frozen solvent molecular-reference site-model SHA-256 must be a "
                "lowercase SHA-256 digest."
            )
        if not isinstance(self.rism_site_type_names, (tuple, list)):
            raise TypeError(
                "Frozen solvent molecular-reference RISM site types must be a sequence."
            )
        names = tuple(
            _nonempty_string(
                name,
                name="Frozen solvent molecular-reference RISM site type",
            )
            for name in self.rism_site_type_names
        )
        molecular_reference = Route2V0MolecularSolventReference(
            atomic_numbers=self.atomic_numbers,
            site_charges_e=self.site_charges_e,
            reference_positions_bohr=self.reference_positions_bohr,
            provenance_label=f"frozen-site-model:{digest}",
            target_total_charge_e=self.target_total_charge_e,
        )
        if len(names) != molecular_reference.site_count:
            raise ValueError(
                "Frozen solvent molecular-reference RISM site types must have one "
                "entry per molecular atom site."
            )
        if self.construction != V0_FROZEN_SOLVENT_MOLECULAR_REFERENCE_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 frozen solvent molecular-reference construction."
            )
        object.__setattr__(self, "atomic_numbers", molecular_reference.atomic_numbers)
        object.__setattr__(self, "site_charges_e", molecular_reference.site_charges_e)
        object.__setattr__(
            self,
            "reference_positions_bohr",
            molecular_reference.reference_positions_bohr,
        )
        object.__setattr__(self, "rism_site_type_names", names)
        object.__setattr__(self, "site_model_sha256", digest)
        object.__setattr__(
            self,
            "target_total_charge_e",
            molecular_reference.target_total_charge_e,
        )
        object.__setattr__(self, "_molecular_reference", molecular_reference)

    @property
    def molecular_reference(self) -> Route2V0MolecularSolventReference:
        """Return the canonical solvent geometry for the MACE cluster source."""

        return self._molecular_reference


@dataclass(frozen=True)
class Route2V0FrozenSolventAsset:
    """One source-provenanced, target-label-independent solvent asset.

    It is a prerequisite for a future molecular-liquid calculation, not that
    calculation itself.  The parsed bulk direct correlation is retained so a
    later backend can reuse the exact source-verified data rather than reopen
    an untracked file.
    """

    solvent_id: str
    model_family: str
    model_identifier: str
    temperature_kelvin: float
    pressure_bar: float
    closure: str
    standard_state: str
    pressure_definition: str
    partial_molar_volume_definition: str
    sign_convention: str
    excluded_target_label_sets: tuple[str, ...]
    files: tuple[Route2V0FrozenSolventAssetFile, ...]
    bulk_direct_correlation: Route2V0RismBulkDirectCorrelation
    molecular_reference: Route2V0FrozenSolventMolecularReference
    molecular_source: Route2V0MolecularRismSource
    thermodynamic_source: Route2V0RismSelfTest
    short_range_source: Route2V0RismShortRangeSource
    generation_source: Route2V0RismGenerationSource
    coulomb_tail_start_angstrom: float
    coulomb_tail_tolerance_dimensionless: float
    construction: str = V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION

    def __post_init__(self) -> None:
        solvent_id = _nonempty_string(self.solvent_id, name="Solvent identifier")
        if not _SOLVENT_ID_PATTERN.fullmatch(solvent_id):
            raise ValueError("Solvent identifier must be a lowercase stable slug.")
        model_family = _nonempty_string(self.model_family, name="Solvent model family")
        model_identifier = _nonempty_string(
            self.model_identifier,
            name="Solvent model identifier",
        )
        closure = _nonempty_string(self.closure, name="Liquid closure")
        standard_state = _nonempty_string(self.standard_state, name="Standard state")
        pressure_definition = _nonempty_string(
            self.pressure_definition,
            name="Pressure definition",
        )
        partial_molar_volume_definition = _nonempty_string(
            self.partial_molar_volume_definition,
            name="Partial-molar-volume definition",
        )
        sign_convention = _nonempty_string(
            self.sign_convention,
            name="Pressure sign convention",
        )
        temperature = _finite_positive(self.temperature_kelvin, name="Temperature")
        pressure = _finite_positive(self.pressure_bar, name="Pressure")
        tail_start = _finite_positive(
            self.coulomb_tail_start_angstrom,
            name="Coulomb-tail start",
        )
        tail_tolerance = _finite_positive(
            self.coulomb_tail_tolerance_dimensionless,
            name="Coulomb-tail tolerance",
        )
        excluded = tuple(
            _nonempty_string(value, name="Excluded target-label set").lower()
            for value in self.excluded_target_label_sets
        )
        if len(set(excluded)) != len(excluded):
            raise ValueError("Excluded target-label sets must be unique.")
        missing_exclusions = V0_REQUIRED_EXCLUDED_TARGET_LABEL_SETS.difference(excluded)
        if missing_exclusions:
            raise ValueError(
                "Frozen solvent asset provenance must exclude target-label sets: "
                f"{', '.join(sorted(missing_exclusions))}."
            )
        files = tuple(self.files)
        roles = tuple(file.role for file in files)
        if len(set(roles)) != len(roles) or set(roles) != V0_REQUIRED_ASSET_FILE_ROLES:
            raise ValueError(
                "Frozen solvent asset files must contain exactly the required roles."
            )
        if not isinstance(
            self.molecular_reference,
            Route2V0FrozenSolventMolecularReference,
        ):
            raise TypeError(
                "Frozen solvent asset requires a canonical molecular reference."
            )
        if not isinstance(self.molecular_source, Route2V0MolecularRismSource):
            raise TypeError(
                "Frozen solvent asset requires a parsed molecular RISM source."
            )
        if not isinstance(self.thermodynamic_source, Route2V0RismSelfTest):
            raise TypeError(
                "Frozen solvent asset requires parsed RISM thermodynamic output."
            )
        if not isinstance(self.short_range_source, Route2V0RismShortRangeSource):
            raise TypeError(
                "Frozen solvent asset requires a parsed solvent-side short-range source."
            )
        if not isinstance(self.generation_source, Route2V0RismGenerationSource):
            raise TypeError(
                "Frozen solvent asset requires parsed two-run generation provenance."
            )
        files_by_role = {file.role: file for file in files}
        generation_source = self.generation_source
        if set(excluded) != set(generation_source.excluded_target_label_sets):
            raise ValueError(
                "Frozen solvent manifest and generation provenance target-label "
                "exclusions must match."
            )
        for role, digest in generation_source.frozen_source_sha256.items():
            if files_by_role[role].sha256 != digest:
                raise ValueError(
                    "Frozen solvent generation provenance must bind the exact "
                    f"{role} source hash."
                )
        if (
            generation_source.residual_tolerance
            != self.molecular_source.rism1d_input.residual_tolerance
        ):
            raise ValueError(
                "Frozen solvent generation tolerance must match the rism1d input."
            )
        maximum_steps = self.molecular_source.rism1d_input.maximum_steps
        if any(
            run.primary_iterations > maximum_steps
            or run.temperature_derivative_iterations > maximum_steps
            for run in generation_source.runs
        ):
            raise ValueError(
                "Frozen solvent generation iterations exceed rism1d MAXSTEP."
            )
        generation_source.verify_xvv_reproduction(files_by_role["xvv"].path)
        short_range_source = self.short_range_source
        for role, digest in short_range_source.source_sha256.items():
            if files_by_role[role].sha256 != digest:
                raise ValueError(
                    "Frozen solvent short-range certificate must bind the exact "
                    f"{role} source hash."
                )
        if short_range_source.closure.casefold() != closure.casefold():
            raise ValueError(
                "Frozen solvent short-range certificate closure must match the asset."
            )
        if not math.isclose(
            short_range_source.temperature_kelvin,
            temperature,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "Frozen solvent short-range certificate temperature must match the asset."
            )
        if not math.isclose(
            short_range_source.pressure_bar,
            pressure,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "Frozen solvent short-range certificate pressure must match the asset."
            )
        if not math.isclose(
            short_range_source.coulomb_tail_start_angstrom,
            tail_start,
            rel_tol=0.0,
            abs_tol=0.0,
        ) or not math.isclose(
            short_range_source.coulomb_tail_tolerance_dimensionless,
            tail_tolerance,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "Frozen solvent short-range certificate tail gate must match the asset."
            )
        site_model_sha256 = next(
            file.sha256 for file in files if file.role == "site_model"
        )
        molecular_reference = self.molecular_reference
        if molecular_reference.site_model_sha256 != site_model_sha256:
            raise ValueError(
                "Frozen solvent molecular reference must cite the frozen asset's "
                "site-model SHA-256."
            )
        if abs(molecular_reference.molecular_reference.total_charge_e) > 1.0e-12:
            raise ValueError(
                "Frozen solvent molecular reference must be electrically neutral."
            )
        source = self.molecular_source
        canonical = molecular_reference.molecular_reference
        if not np.array_equal(canonical.atomic_numbers, source.atomic_numbers):
            raise ValueError(
                "Frozen solvent molecular-reference atomic numbers must equal "
                "the parsed MDL atomic identities."
            )
        if not np.allclose(
            canonical.site_charges_e,
            source.site_charges_e,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Frozen solvent molecular-reference site charges must equal "
                "the parsed MDL charges."
            )
        if molecular_reference.rism_site_type_names != source.atom_site_names:
            raise ValueError(
                "Frozen solvent molecular-reference RISM site map must equal "
                "the parsed MDL atom-site order."
            )
        if not same_labelled_rigid_geometry(
            canonical.reference_positions_bohr * Bohr,
            molecular_reference.rism_site_type_names,
            source.reference_positions_bohr * Bohr,
            source.atom_site_names,
        ):
            raise ValueError(
                "Frozen solvent molecular-reference geometry must be rigidly "
                "equivalent by a proper rotation to the parsed MDL geometry."
            )
        if closure.casefold() != source.rism1d_input.closure.casefold():
            raise ValueError(
                "Frozen solvent asset closure must match the parsed rism1d input."
            )
        metadata = self.bulk_direct_correlation.metadata
        if not _same_rism_metadata(metadata, source.metadata):
            raise ValueError(
                "Frozen solvent molecular source and bulk correlation must use "
                "the same XVV metadata."
            )
        site_lookup = {name: index for index, name in enumerate(metadata.site_names)}
        try:
            site_type_indices = np.asarray(
                [
                    site_lookup[name]
                    for name in molecular_reference.rism_site_type_names
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "Frozen solvent molecular-reference RISM site type is absent from "
                "the frozen XVV source."
            ) from exc
        observed_multiplicity = np.bincount(
            site_type_indices,
            minlength=metadata.site_count,
        )
        if not np.array_equal(observed_multiplicity, metadata.site_multiplicity):
            raise ValueError(
                "Frozen solvent molecular-reference RISM site multiplicities must "
                "equal the frozen XVV multiplicities."
            )
        if self.construction != V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 frozen solvent asset construction.")
        if not math.isclose(
            temperature,
            self.bulk_direct_correlation.metadata.temperature_kelvin,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ValueError(
                "Frozen solvent asset temperature must match the XVV bulk state."
            )
        residual = self.bulk_direct_correlation.coulomb_tail_residual(
            minimum_radius_angstrom=tail_start
        )
        if residual > tail_tolerance:
            raise ValueError(
                "Frozen solvent asset Coulomb-tail residual exceeds its declared "
                "tolerance."
            )
        object.__setattr__(self, "solvent_id", solvent_id)
        object.__setattr__(self, "model_family", model_family)
        object.__setattr__(self, "model_identifier", model_identifier)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "pressure_bar", pressure)
        object.__setattr__(self, "closure", closure)
        object.__setattr__(self, "standard_state", standard_state)
        object.__setattr__(self, "pressure_definition", pressure_definition)
        object.__setattr__(
            self,
            "partial_molar_volume_definition",
            partial_molar_volume_definition,
        )
        object.__setattr__(self, "sign_convention", sign_convention)
        object.__setattr__(self, "excluded_target_label_sets", excluded)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "molecular_reference", molecular_reference)
        object.__setattr__(self, "molecular_source", source)
        object.__setattr__(self, "short_range_source", short_range_source)
        object.__setattr__(self, "generation_source", generation_source)
        object.__setattr__(self, "coulomb_tail_start_angstrom", tail_start)
        object.__setattr__(
            self,
            "coulomb_tail_tolerance_dimensionless",
            tail_tolerance,
        )

    def file_for(self, role: str) -> Route2V0FrozenSolventAssetFile:
        """Return a declared file by its immutable role."""

        for file in self.files:
            if file.role == role:
                return file
        raise ValueError(f"Frozen solvent asset lacks file role {role!r}.")

    def verify_integrity(self) -> None:
        """Recheck all frozen files before a future liquid-state use."""

        for file in self.files:
            file.verify_integrity()


@dataclass(frozen=True)
class Route2V0FrozenSolventRegistry:
    """A unique collection of solvent-side assets frozen before scoring."""

    assets: tuple[Route2V0FrozenSolventAsset, ...]
    construction: str = V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION

    def __post_init__(self) -> None:
        assets = tuple(self.assets)
        identifiers = tuple(asset.solvent_id for asset in assets)
        if not assets:
            raise ValueError("Frozen solvent registry must contain at least one asset.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Frozen solvent registry contains duplicate solvent IDs.")
        if self.construction != V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 frozen solvent registry construction."
            )
        object.__setattr__(self, "assets", assets)

    def asset_for(self, solvent_id: str) -> Route2V0FrozenSolventAsset:
        """Return one registered solvent asset without a fallback choice."""

        identifier = _nonempty_string(solvent_id, name="Solvent identifier")
        for asset in self.assets:
            if asset.solvent_id == identifier:
                return asset
        raise ValueError(
            f"No frozen Route-2 V0 solvent asset exists for {identifier!r}."
        )

    def require_default_solvent_panel(self) -> None:
        """Require all pre-registered solvent strata before target scoring."""

        available = {asset.solvent_id for asset in self.assets}
        missing = [
            identifier
            for identifier in V0_DEFAULT_SOLVENT_IDS
            if identifier not in available
        ]
        if missing:
            raise ValueError(
                "Frozen Route-2 V0 solvent registry is incomplete; missing default "
                f"assets: {', '.join(missing)}."
            )

    def verify_integrity(self) -> None:
        """Recheck every file in every registered solvent asset."""

        for asset in self.assets:
            asset.verify_integrity()


def _file_from_manifest(
    *,
    root: Path,
    role: str,
    value: object,
) -> Route2V0FrozenSolventAssetFile:
    """Parse and verify one content-addressed manifest file entry."""

    entry = _mapping(value, name=f"{role} file entry")
    return Route2V0FrozenSolventAssetFile(
        role=role,
        path=_asset_path(root=root, relative_path=entry.get("path"), name=role),
        sha256=_nonempty_string(entry.get("sha256"), name=f"{role}.sha256"),
    )


def _molecular_reference_from_manifest(
    *,
    value: object,
) -> Route2V0FrozenSolventMolecularReference:
    """Parse the mandatory canonical molecular record from one asset entry."""

    entry = _mapping(value, name="Frozen solvent molecular reference")
    site_type_names = entry.get("rism_site_type_names")
    if not isinstance(site_type_names, list):
        raise TypeError(
            "Frozen solvent molecular-reference RISM site types must be a list."
        )
    return Route2V0FrozenSolventMolecularReference(
        atomic_numbers=np.asarray(entry.get("atomic_numbers")),
        site_charges_e=np.asarray(entry.get("site_charges_e")),
        reference_positions_bohr=np.asarray(entry.get("reference_positions_bohr")),
        rism_site_type_names=tuple(site_type_names),
        site_model_sha256=_nonempty_string(
            entry.get("site_model_sha256"),
            name="molecular_reference.site_model_sha256",
        ),
        target_total_charge_e=entry.get("target_total_charge_e", 0.0),
    )


def _asset_from_manifest(
    *,
    root: Path,
    value: object,
) -> Route2V0FrozenSolventAsset:
    """Load one fully declared asset from a registry manifest entry."""

    entry = _mapping(value, name="Frozen solvent asset entry")
    model = _mapping(entry.get("model"), name="Frozen solvent model")
    state = _mapping(entry.get("state"), name="Frozen solvent state")
    convention = _mapping(
        entry.get("liquid_convention"),
        name="Frozen liquid convention",
    )
    bulk = _mapping(entry.get("bulk_correlation"), name="Frozen bulk correlation")
    source_files = _mapping(entry.get("source_files"), name="Frozen source files")
    provenance = _mapping(entry.get("provenance"), name="Frozen solvent provenance")
    _strict_false(
        provenance.get("target_solvation_labels_used"),
        name="target_solvation_labels_used",
    )
    excluded_values = provenance.get("excluded_target_label_sets")
    if not isinstance(excluded_values, list):
        raise TypeError("excluded_target_label_sets must be a list.")

    files_by_role = {
        "xvv": _file_from_manifest(root=root, role="xvv", value=bulk.get("xvv")),
        "cvv": _file_from_manifest(root=root, role="cvv", value=bulk.get("cvv")),
    }
    for role in (
        "site_model",
        "rism1d_input",
        "thermodynamic_output",
        "short_range_interaction",
        "provenance_statement",
    ):
        files_by_role[role] = _file_from_manifest(
            root=root,
            role=role,
            value=source_files.get(role),
        )
    direct_correlation = load_rism1d_bulk_direct_correlation(
        xvv_path=files_by_role["xvv"].path,
        cvv_path=files_by_role["cvv"].path,
    )
    molecular_source = load_route2_v0_rism_molecular_source(
        mdl_path=files_by_role["site_model"].path,
        rism1d_input_path=files_by_role["rism1d_input"].path,
        xvv_path=files_by_role["xvv"].path,
    )
    thermodynamic_source = load_route2_v0_rism_self_test(
        files_by_role["thermodynamic_output"].path
    )
    short_range_source = load_route2_v0_rism_short_range_source(
        files_by_role["short_range_interaction"].path
    )
    generation_source = load_route2_v0_rism_generation_source(
        files_by_role["provenance_statement"].path
    )
    molecular_reference = _molecular_reference_from_manifest(
        value=entry.get("molecular_reference"),
    )
    return Route2V0FrozenSolventAsset(
        solvent_id=_nonempty_string(entry.get("solvent_id"), name="solvent_id"),
        model_family=_nonempty_string(model.get("family"), name="model.family"),
        model_identifier=_nonempty_string(
            model.get("identifier"),
            name="model.identifier",
        ),
        temperature_kelvin=_finite_positive(
            state.get("temperature_kelvin"),
            name="state.temperature_kelvin",
        ),
        pressure_bar=_finite_positive(
            state.get("pressure_bar"),
            name="state.pressure_bar",
        ),
        closure=_nonempty_string(convention.get("closure"), name="closure"),
        standard_state=_nonempty_string(
            convention.get("standard_state"),
            name="standard_state",
        ),
        pressure_definition=_nonempty_string(
            convention.get("pressure_definition"),
            name="pressure_definition",
        ),
        partial_molar_volume_definition=_nonempty_string(
            convention.get("partial_molar_volume_definition"),
            name="partial_molar_volume_definition",
        ),
        sign_convention=_nonempty_string(
            convention.get("sign_convention"),
            name="sign_convention",
        ),
        excluded_target_label_sets=tuple(excluded_values),
        files=tuple(files_by_role.values()),
        bulk_direct_correlation=direct_correlation,
        molecular_reference=molecular_reference,
        molecular_source=molecular_source,
        thermodynamic_source=thermodynamic_source,
        short_range_source=short_range_source,
        generation_source=generation_source,
        coulomb_tail_start_angstrom=_finite_positive(
            bulk.get("coulomb_tail_start_angstrom"),
            name="bulk_correlation.coulomb_tail_start_angstrom",
        ),
        coulomb_tail_tolerance_dimensionless=_finite_positive(
            bulk.get("coulomb_tail_tolerance_dimensionless"),
            name="bulk_correlation.coulomb_tail_tolerance_dimensionless",
        ),
    )


def load_route2_v0_frozen_solvent_registry(
    manifest_path: str | Path,
) -> Route2V0FrozenSolventRegistry:
    """Load a content-addressed, no-target-label Route-2 V0 solvent registry.

    This function deliberately does not manufacture missing assets or choose a
    proxy liquid model.  A caller that intends to begin target-solute scoring
    must call :meth:`Route2V0FrozenSolventRegistry.require_default_solvent_panel`
    in addition to the later liquid-functional and force-certification gates.
    """

    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise ValueError(f"Frozen solvent registry manifest is absent: {manifest}.")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Frozen solvent registry manifest must be valid JSON."
        ) from exc
    registry = _mapping(payload, name="Frozen solvent registry manifest")
    if registry.get("protocol_id") != V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION:
        raise ValueError("Unsupported Route-2 frozen solvent registry protocol.")
    if registry.get("schema_version") != V0_FROZEN_SOLVENT_ASSET_SCHEMA_VERSION:
        raise ValueError("Unsupported Route-2 frozen solvent registry schema.")
    entries = registry.get("assets")
    if not isinstance(entries, list):
        raise TypeError("Frozen solvent registry assets must be a list.")
    return Route2V0FrozenSolventRegistry(
        assets=tuple(
            _asset_from_manifest(root=manifest.parent, value=entry) for entry in entries
        )
    )


__all__ = [
    "V0_DEFAULT_SOLVENT_IDS",
    "V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION",
    "V0_FROZEN_SOLVENT_ASSET_SCHEMA_VERSION",
    "V0_FROZEN_SOLVENT_MOLECULAR_REFERENCE_CONSTRUCTION",
    "V0_REQUIRED_ASSET_FILE_ROLES",
    "V0_REQUIRED_EXCLUDED_TARGET_LABEL_SETS",
    "Route2V0FrozenSolventAsset",
    "Route2V0FrozenSolventAssetFile",
    "Route2V0FrozenSolventMolecularReference",
    "Route2V0FrozenSolventRegistry",
    "load_route2_v0_frozen_solvent_registry",
]
