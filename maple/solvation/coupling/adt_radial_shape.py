"""Role-separated radial shapes for atomic-density-translation response.

The atomic-density-translation (ADT) construction needs a positive spherical
*response shape*.  That object is not necessarily a full neutral all-electron
density.  In particular, an ECP valence pseudo-density may be the appropriate
translation shape while remaining invalid for the separate neutral
penetration potential ``Z/r - V_e``.

This module makes that distinction explicit and content addressed.  Existing
neutral-atom mixtures are reused unchanged for their established elements;
iodine is supplied by an independently versioned 25-electron def2-ECP valence
shape.  No interpolation, element fallback, solvation target, or empirical
scale enters the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.coupling.neutral_atom_penetration import (
    NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256,
    NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH,
    NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
    load_neutral_atom_penetration_mixtures,
)

ADT_TRANSLATION_SHAPE_ROLE = "adt_translation_shape"
ADT_RADIAL_SHAPE_REGISTRY_ID = (
    "route2-adt-radial-shape-registry-neutral-plus-iodine-ecp-valence-v1"
)
IODINE_ECP_VALENCE_ADT_ARTIFACT = "route2-adt-radial-shape-iodine-ecp-valence-v1"
IODINE_ECP_VALENCE_ADT_TABLE_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-adt-radial-shape-iodine-ecp-valence-v1.npz"
)
IODINE_ECP_VALENCE_ADT_MANIFEST_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-adt-radial-shape-iodine-ecp-valence-v1.json"
)
IODINE_ECP_VALENCE_ADT_TABLE_SHA256 = (
    "c7378d26481a40adf61581eea4f26d59bd493050b43028f4cd7abe7ebde040db"
)
IODINE_ECP_VALENCE_ADT_MANIFEST_SHA256 = (
    "52decde58d9e146c8b0913f7e105892cdbce252e792b62f2a44552ee13ff8a70"
)
IODINE_ECP_VALENCE_ADT_CONTENT_SHA256 = (
    "bd4bba766dd4ba5b0460038a4eae1471dc2d3a988738dfb0fcc476690d9e9adc"
)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text.")
    return value.strip()


def _mixture_copy(value: object) -> GaussianMixtureAtom:
    if not isinstance(value, GaussianMixtureAtom):
        raise TypeError("ADT radial shape mixture must be GaussianMixtureAtom.")
    return GaussianMixtureAtom(
        np.asarray(value.electron_counts, dtype=np.float64),
        np.asarray(value.gaussian_exponents_bohr2, dtype=np.float64),
    )


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    return _canonical_sha256(
        {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes": value.tobytes(order="C").hex(),
            }
            for key, value in arrays.items()
        }
    )


@dataclass(frozen=True, slots=True)
class ADTRadialShape:
    """One element's positive spherical ADT response shape."""

    atomic_number: int
    effective_electron_count: float
    ecp_core_electron_count: int
    mixture: GaussianMixtureAtom
    representation: str
    source_asset_sha256: str
    configuration_sha256: str = ""

    def __post_init__(self) -> None:
        number = _positive_integer(self.atomic_number, name="atomic_number")
        effective = float(self.effective_electron_count)
        if not math.isfinite(effective) or effective <= 0.0:
            raise ValueError("effective_electron_count must be positive finite.")
        core = self.ecp_core_electron_count
        if isinstance(core, bool) or not isinstance(core, (int, np.integer)):
            raise TypeError("ecp_core_electron_count must be an integer.")
        core = int(core)
        if core < 0 or abs(effective + core - number) > 5.0e-12:
            raise ValueError(
                "effective plus ECP-core electron counts must equal atomic number."
            )
        mixture = _mixture_copy(self.mixture)
        if abs(mixture.electron_count - effective) > 5.0e-12:
            raise ValueError(
                "ADT mixture normalization differs from effective electron count."
            )
        representation = _nonempty(self.representation, name="representation")
        asset = _digest(self.source_asset_sha256, name="source_asset_sha256")
        expected = _canonical_sha256(
            {
                "contract": "route2-role-separated-adt-radial-shape-v1",
                "role": ADT_TRANSLATION_SHAPE_ROLE,
                "atomic_number": number,
                "effective_electron_count": effective,
                "ecp_core_electron_count": core,
                "representation": representation,
                "source_asset_sha256": asset,
                "electron_counts": mixture.electron_counts.tolist(),
                "gaussian_exponents_bohr2": (mixture.gaussian_exponents_bohr2.tolist()),
                "neutral_penetration_compatible": core == 0,
            }
        )
        if self.configuration_sha256 and self.configuration_sha256 != expected:
            raise ValueError("ADT radial-shape configuration SHA256 mismatch.")
        object.__setattr__(self, "atomic_number", number)
        object.__setattr__(self, "effective_electron_count", effective)
        object.__setattr__(self, "ecp_core_electron_count", core)
        object.__setattr__(self, "mixture", mixture)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "source_asset_sha256", asset)
        object.__setattr__(self, "configuration_sha256", expected)

    @property
    def role(self) -> str:
        return ADT_TRANSLATION_SHAPE_ROLE

    @property
    def neutral_penetration_compatible(self) -> bool:
        return self.ecp_core_electron_count == 0


@dataclass(frozen=True, slots=True)
class ADTRadialShapeRegistry:
    """Immutable content-addressed mapping from element to ADT shape."""

    shapes_by_atomic_number: Mapping[int, ADTRadialShape]
    parent_asset_sha256s: tuple[str, ...]
    registry_id: str = ADT_RADIAL_SHAPE_REGISTRY_ID
    configuration_sha256: str = ""

    def __post_init__(self) -> None:
        if self.registry_id != ADT_RADIAL_SHAPE_REGISTRY_ID:
            raise ValueError("Unknown ADT radial-shape registry identity.")
        if (
            not isinstance(self.shapes_by_atomic_number, Mapping)
            or not self.shapes_by_atomic_number
        ):
            raise TypeError("shapes_by_atomic_number must be a non-empty mapping.")
        copied: dict[int, ADTRadialShape] = {}
        for raw_number, shape in self.shapes_by_atomic_number.items():
            number = _positive_integer(raw_number, name="shape atomic number")
            if not isinstance(shape, ADTRadialShape):
                raise TypeError("registry entries must be ADTRadialShape.")
            if shape.atomic_number != number:
                raise ValueError("registry key and ADT shape atomic number differ.")
            copied[number] = shape
        parents = tuple(
            _digest(value, name="parent_asset_sha256")
            for value in self.parent_asset_sha256s
        )
        if not parents or len(set(parents)) != len(parents):
            raise ValueError("parent asset SHA256 values must be unique and non-empty.")
        expected = _canonical_sha256(
            {
                "contract": "route2-role-separated-adt-radial-shape-registry-v1",
                "registry_id": self.registry_id,
                "parent_asset_sha256s": list(parents),
                "shapes": {
                    str(number): shape.configuration_sha256
                    for number, shape in sorted(copied.items())
                },
                "fallback": "forbidden",
                "interpolation": "forbidden",
                "solvation_target_fit": False,
            }
        )
        if self.configuration_sha256 and self.configuration_sha256 != expected:
            raise ValueError("ADT registry configuration SHA256 mismatch.")
        object.__setattr__(
            self,
            "shapes_by_atomic_number",
            MappingProxyType(dict(sorted(copied.items()))),
        )
        object.__setattr__(self, "parent_asset_sha256s", parents)
        object.__setattr__(self, "configuration_sha256", expected)

    @property
    def supported_atomic_numbers(self) -> tuple[int, ...]:
        return tuple(self.shapes_by_atomic_number)

    def shape(self, atomic_number: int) -> ADTRadialShape:
        number = _positive_integer(atomic_number, name="atomic_number")
        try:
            return self.shapes_by_atomic_number[number]
        except KeyError as exc:
            raise ValueError(
                f"ADT radial-shape registry has no element Z={number}."
            ) from exc


def _load_iodine_ecp_valence_shape(
    *, table_path: Path, manifest_path: Path
) -> ADTRadialShape:
    if _sha256(table_path) != IODINE_ECP_VALENCE_ADT_TABLE_SHA256:
        raise ValueError("Iodine ADT table SHA256 mismatch.")
    if _sha256(manifest_path) != IODINE_ECP_VALENCE_ADT_MANIFEST_SHA256:
        raise ValueError("Iodine ADT manifest SHA256 mismatch.")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("artifact") != IODINE_ECP_VALENCE_ADT_ARTIFACT:
        raise ValueError("Iodine ADT manifest identity is invalid.")
    if payload.get("schema_version") != 1:
        raise ValueError("Iodine ADT manifest schema is unsupported.")
    stored_payload_sha = payload.get("manifest_payload_sha256")
    payload_without_hash = dict(payload)
    payload_without_hash.pop("manifest_payload_sha256", None)
    if stored_payload_sha != _canonical_sha256(payload_without_hash):
        raise ValueError("Iodine ADT manifest payload SHA256 mismatch.")
    claim = payload.get("claim_boundary")
    expected_claim = {
        "capability_admitted": False,
        "experimental_solvation_targets_read": False,
        "full_neutral_density": False,
        "mace_outputs_read": False,
        "neutral_penetration_use_forbidden": True,
        "role": ADT_TRANSLATION_SHAPE_ROLE,
        "solvation_or_pcm_quantities_used_for_fit": False,
    }
    if claim != expected_claim:
        raise ValueError("Iodine ADT claim boundary changed.")
    table = payload.get("table")
    if (
        not isinstance(table, dict)
        or table.get("sha256") != IODINE_ECP_VALENCE_ADT_TABLE_SHA256
    ):
        raise ValueError("Iodine ADT table binding is invalid.")
    if table.get("content_sha256") != IODINE_ECP_VALENCE_ADT_CONTENT_SHA256:
        raise ValueError("Iodine ADT content binding is invalid.")
    with np.load(table_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "atomic_numbers",
            "component_offsets",
            "ecp_core_electron_counts",
            "effective_electron_counts",
            "electron_counts",
            "gaussian_exponents_bohr2",
        }:
            raise ValueError("Iodine ADT table fields changed.")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if _array_content_sha256(arrays) != IODINE_ECP_VALENCE_ADT_CONTENT_SHA256:
        raise ValueError("Iodine ADT table content SHA256 mismatch.")
    if (
        not np.array_equal(arrays["atomic_numbers"], np.asarray([53], dtype=np.int64))
        or not np.array_equal(
            arrays["component_offsets"], np.asarray([0, 6], dtype=np.int64)
        )
        or not np.array_equal(
            arrays["ecp_core_electron_counts"], np.asarray([28], dtype=np.int64)
        )
        or not np.array_equal(
            arrays["effective_electron_counts"], np.asarray([25.0], dtype=np.float64)
        )
    ):
        raise ValueError("Iodine ADT element/electron-count metadata changed.")
    element = payload.get("element")
    if (
        not isinstance(element, dict)
        or element.get("representation") != "ecp_valence_pseudodensity"
    ):
        raise ValueError("Iodine ADT representation is invalid.")
    mixture = GaussianMixtureAtom(
        np.asarray(arrays["electron_counts"], dtype=np.float64),
        np.asarray(arrays["gaussian_exponents_bohr2"], dtype=np.float64),
    )
    return ADTRadialShape(
        atomic_number=53,
        effective_electron_count=25.0,
        ecp_core_electron_count=28,
        mixture=mixture,
        representation="ecp_valence_pseudodensity",
        source_asset_sha256=IODINE_ECP_VALENCE_ADT_CONTENT_SHA256,
    )


def load_repository_adt_radial_shape_registry(
    *, source_root: str | Path
) -> ADTRadialShapeRegistry:
    """Load the frozen neutral shapes plus the iodine ECP-valence extension."""

    root = Path(source_root).expanduser().resolve()
    neutral = load_neutral_atom_penetration_mixtures(
        table_path=root / NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
        manifest_path=root / NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH,
    )
    shapes: dict[int, ADTRadialShape] = {
        int(number): ADTRadialShape(
            atomic_number=int(number),
            effective_electron_count=float(number),
            ecp_core_electron_count=0,
            mixture=mixture,
            representation="all_electron_neutral_density_reused_for_adt_shape",
            source_asset_sha256=NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256,
        )
        for number, mixture in neutral.items()
    }
    iodine = _load_iodine_ecp_valence_shape(
        table_path=root / IODINE_ECP_VALENCE_ADT_TABLE_REPO_PATH,
        manifest_path=root / IODINE_ECP_VALENCE_ADT_MANIFEST_REPO_PATH,
    )
    shapes[iodine.atomic_number] = iodine
    return ADTRadialShapeRegistry(
        shapes_by_atomic_number=shapes,
        parent_asset_sha256s=(
            NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256,
            IODINE_ECP_VALENCE_ADT_CONTENT_SHA256,
        ),
    )


__all__ = [
    "ADT_RADIAL_SHAPE_REGISTRY_ID",
    "ADT_TRANSLATION_SHAPE_ROLE",
    "ADTRadialShape",
    "ADTRadialShapeRegistry",
    "IODINE_ECP_VALENCE_ADT_ARTIFACT",
    "IODINE_ECP_VALENCE_ADT_CONTENT_SHA256",
    "IODINE_ECP_VALENCE_ADT_MANIFEST_REPO_PATH",
    "IODINE_ECP_VALENCE_ADT_MANIFEST_SHA256",
    "IODINE_ECP_VALENCE_ADT_TABLE_REPO_PATH",
    "IODINE_ECP_VALENCE_ADT_TABLE_SHA256",
    "load_repository_adt_radial_shape_registry",
]
