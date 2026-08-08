"""Analytic atomic reference densities for the experimental Route-2 rho-DROP path.

The existing :mod:`route2_v0_promolecular_density` table is intentionally a
piecewise-linear, source-control quantity.  This module does not change that
contract.  It loads a separate, content-addressed Gaussian-mixture asset whose
only purpose is to provide a positive, normalized, ``C-infinity`` atomic
reference density and Cartesian spatial derivatives through third order.

All coordinates accepted here are already in bohr.  Conversion from the Route-2
public geometry convention (angstrom) is owned by ``route2_density_levelset`` so
that the rho-DROP stack has one unit/order boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

ATOMIC_REFERENCE_DENSITY_ARTIFACT = (
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1"
)
ATOMIC_REFERENCE_DENSITY_CONSTRUCTION = (
    "positive-normalized-spherical-free-atom-hf-gaussian-mixture-v1"
)
ATOMIC_REFERENCE_DENSITY_SCHEMA_VERSION = 1
ATOMIC_REFERENCE_DENSITY_STATUS = (
    "mathematical-kernel-pass-scientific-isosurface-gate-pending"
)
ATOMIC_REFERENCE_DENSITY_TABLE_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
)
ATOMIC_REFERENCE_DENSITY_GENERATOR_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "generate_route2_rhodrop_atomic_reference_density.py"
)
ATOMIC_REFERENCE_DENSITY_TABLE_SHA256 = (
    "d81363187ce6092b23031f5626e0d794bbad4f44646f1e06e41b1112768fefc6"
)
ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256 = (
    "bc85c8375fee448ceb78f7394a23a459e9c17ea6e70f672f3e3937072112736f"
)
ATOMIC_REFERENCE_DENSITY_GENERATOR_SHA256 = (
    "ff4d8a79ad69405879a740d6af8cd503eb1d998cf500ed0a06482ea062029b98"
)
ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256 = (
    "6c98c98391fa93fda8690a491af2bc1318d891373f424196d6e40c5779f32e60"
)
ATOMIC_REFERENCE_DENSITY_SOURCE = {
    "artifact": "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1",
    "table_path": (
        "docs/implicit-solvation/benchmarks/"
        "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.npz"
    ),
    "table_sha256": (
        "1a63ae8a5d939e00f0a8dba1951a8099dad8ba2a068be61431046c880d7aae28"
    ),
    "manifest_path": (
        "docs/implicit-solvation/benchmarks/"
        "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.json"
    ),
    "manifest_sha256": (
        "9f0b56bbe5bc116d55b10ac3aa7246080e1a17a06001f62b81cd3f66b6c95f81"
    ),
}
_LOG_MIN_SUBNORMAL_FLOAT64 = float(np.log(np.nextafter(0.0, 1.0)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_finite_array(
    values: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _points(values: object, *, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(f"{name} must be finite with shape (n_points, 3).")
    return points


def _atomic_numbers(values: object, *, atom_count: int | None = None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("Atomic numbers must be a finite nonempty vector.")
    numeric = np.asarray(raw, dtype=float)
    rounded = np.rint(numeric)
    if np.any(rounded != numeric) or np.any(rounded < 1.0):
        raise ValueError("Atomic numbers must be positive integers.")
    result = rounded.astype(np.int64, copy=False)
    if atom_count is not None and result.shape != (atom_count,):
        raise ValueError(
            f"Atomic numbers must have shape ({atom_count},); received "
            f"{result.shape}."
        )
    return result


def _freeze_provenance(value: object, *, path: str = "provenance") -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(f"{path} keys must be nonempty strings.")
            frozen[raw_key] = _freeze_provenance(item, path=f"{path}.{raw_key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_provenance(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}.")


@dataclass(frozen=True)
class GaussianMixtureAtom:
    """One positive normalized spherical atomic Gaussian mixture.

    Component ``k`` is

    ``N_k * (alpha_k/pi)**(3/2) * exp(-alpha_k*|r-R|**2)``.

    ``N_k`` is therefore the component electron count, not an arbitrary peak
    amplitude.  Asset validation separately enforces ``sum_k N_k == Z``.
    """

    electron_counts: np.ndarray
    gaussian_exponents_bohr2: np.ndarray

    def __post_init__(self) -> None:
        counts = _readonly_finite_array(
            self.electron_counts,
            name="Gaussian component electron counts",
            ndim=1,
        )
        exponents = _readonly_finite_array(
            self.gaussian_exponents_bohr2,
            name="Gaussian exponents",
            ndim=1,
        )
        if counts.size == 0 or counts.shape != exponents.shape:
            raise ValueError(
                "Gaussian component counts and exponents must be nonempty and aligned."
            )
        if np.any(counts < 0.0):
            raise ValueError("Gaussian component electron counts must be nonnegative.")
        if np.any(exponents <= 0.0):
            raise ValueError("Gaussian exponents must be strictly positive.")
        if not np.any(counts > 0.0):
            raise ValueError("At least one Gaussian component must carry electrons.")
        object.__setattr__(self, "electron_counts", counts)
        object.__setattr__(self, "gaussian_exponents_bohr2", exponents)

    @property
    def electron_count(self) -> float:
        return float(np.sum(self.electron_counts))

    @property
    def component_count(self) -> int:
        return int(self.electron_counts.size)


def _mixture_content_sha256(
    mixtures_by_atomic_number: Mapping[int, GaussianMixtureAtom],
) -> str:
    """Canonical digest of unpacked Gaussian-mixture scientific content."""

    digest = hashlib.sha256()
    digest.update(b"route2-rhodrop-atomic-reference-mixtures-v1\0")
    for atomic_number in sorted(mixtures_by_atomic_number):
        mixture = mixtures_by_atomic_number[atomic_number]
        counts = np.ascontiguousarray(mixture.electron_counts, dtype="<f8")
        exponents = np.ascontiguousarray(
            mixture.gaussian_exponents_bohr2,
            dtype="<f8",
        )
        digest.update(
            np.asarray(
                [int(atomic_number), int(counts.size)],
                dtype="<i8",
            ).tobytes()
        )
        digest.update(counts.tobytes())
        digest.update(exponents.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DensitySpatialJet:
    """Value and Cartesian spatial derivatives through third order."""

    value: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    third: np.ndarray

    def __post_init__(self) -> None:
        value = _readonly_finite_array(self.value, name="Density value", ndim=1)
        gradient = _readonly_finite_array(
            self.gradient,
            name="Density gradient",
            ndim=2,
        )
        hessian = _readonly_finite_array(
            self.hessian,
            name="Density Hessian",
            ndim=3,
        )
        third = _readonly_finite_array(
            self.third,
            name="Density third derivative",
            ndim=4,
        )
        point_count = value.shape[0]
        if gradient.shape != (point_count, 3):
            raise ValueError("Density gradient must have shape (n_points, 3).")
        if hessian.shape != (point_count, 3, 3):
            raise ValueError("Density Hessian must have shape (n_points, 3, 3).")
        if third.shape != (point_count, 3, 3, 3):
            raise ValueError(
                "Density third derivative must have shape (n_points, 3, 3, 3)."
            )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "third", third)

    @classmethod
    def zeros(cls, point_count: int) -> "DensitySpatialJet":
        if point_count <= 0:
            raise ValueError("Density jet point count must be positive.")
        return cls(
            value=np.zeros(point_count, dtype=float),
            gradient=np.zeros((point_count, 3), dtype=float),
            hessian=np.zeros((point_count, 3, 3), dtype=float),
            third=np.zeros((point_count, 3, 3, 3), dtype=float),
        )

    def scaled(self, factor: float) -> "DensitySpatialJet":
        scale = float(factor)
        if not np.isfinite(scale):
            raise ValueError("Density jet scale must be finite.")
        return DensitySpatialJet(
            value=scale * self.value,
            gradient=scale * self.gradient,
            hessian=scale * self.hessian,
            third=scale * self.third,
        )

    def plus(self, other: "DensitySpatialJet") -> "DensitySpatialJet":
        if not isinstance(other, DensitySpatialJet):
            raise TypeError("Density jets can only be added to density jets.")
        return DensitySpatialJet(
            value=self.value + other.value,
            gradient=self.gradient + other.gradient,
            hessian=self.hessian + other.hessian,
            third=self.third + other.third,
        )


@dataclass(frozen=True)
class AtomicReferenceDensityAsset:
    """Content-addressed Gaussian-mixture element table."""

    mixtures_by_atomic_number: Mapping[int, GaussianMixtureAtom]
    table_sha256: str
    manifest_sha256: str
    provenance: Mapping[str, object]
    artifact: str = ATOMIC_REFERENCE_DENSITY_ARTIFACT
    construction: str = ATOMIC_REFERENCE_DENSITY_CONSTRUCTION
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.artifact != ATOMIC_REFERENCE_DENSITY_ARTIFACT:
            raise ValueError("Unsupported atomic reference-density artifact.")
        if self.construction != ATOMIC_REFERENCE_DENSITY_CONSTRUCTION:
            raise ValueError("Unsupported atomic reference-density construction.")
        for name in ("table_sha256", "manifest_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"Atomic reference-density {name} is invalid.")

        frozen: dict[int, GaussianMixtureAtom] = {}
        for raw_number, mixture in self.mixtures_by_atomic_number.items():
            if isinstance(raw_number, bool):
                raise ValueError("Atomic reference-density keys must be integers.")
            atomic_number = int(raw_number)
            if atomic_number < 1 or atomic_number != raw_number:
                raise ValueError(
                    "Atomic reference-density keys must be positive integers."
                )
            if not isinstance(mixture, GaussianMixtureAtom):
                raise TypeError(
                    "Atomic reference-density values must be Gaussian mixtures."
                )
            if not np.isclose(
                mixture.electron_count,
                float(atomic_number),
                rtol=0.0,
                atol=5.0e-12,
            ):
                raise ValueError(
                    "Gaussian component electron counts must sum to the neutral "
                    f"atomic number (Z={atomic_number}, observed="
                    f"{mixture.electron_count:.16g})."
                )
            frozen[atomic_number] = mixture
        if not frozen:
            raise ValueError("Atomic reference-density asset must contain elements.")
        content_sha256 = _mixture_content_sha256(frozen)
        claims_frozen_identity = (
            self.table_sha256 == ATOMIC_REFERENCE_DENSITY_TABLE_SHA256
            or self.manifest_sha256 == ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256
        )
        if claims_frozen_identity and not (
            self.table_sha256 == ATOMIC_REFERENCE_DENSITY_TABLE_SHA256
            and self.manifest_sha256 == ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256
            and content_sha256 == ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256
        ):
            raise ValueError(
                "Atomic reference-density frozen hash claims do not match the "
                "unpacked Gaussian-mixture content digest."
            )
        object.__setattr__(
            self,
            "mixtures_by_atomic_number",
            MappingProxyType(dict(sorted(frozen.items()))),
        )
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "content_sha256", content_sha256)

    @property
    def supported_atomic_numbers(self) -> tuple[int, ...]:
        return tuple(self.mixtures_by_atomic_number)

    def mixture(self, atomic_number: int) -> GaussianMixtureAtom:
        if isinstance(atomic_number, bool) or not isinstance(
            atomic_number,
            (int, np.integer),
        ):
            raise ValueError("Atomic reference-density lookup requires an integer Z.")
        try:
            return self.mixtures_by_atomic_number[int(atomic_number)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Atomic reference density has no Gaussian mixture for atomic "
                f"number {atomic_number!r}."
            ) from exc

    def recomputed_content_sha256(self) -> str:
        """Return the digest of the currently stored unpacked mixtures."""

        return _mixture_content_sha256(self.mixtures_by_atomic_number)

    def require_content_integrity(self) -> None:
        """Reject any object whose stored digest no longer matches its content."""

        if self.recomputed_content_sha256() != self.content_sha256:
            raise ValueError(
                "Atomic reference-density content changed after construction."
            )

    def require_frozen_v1_identity(self) -> None:
        """Recompute and require the exact production-v1 content identity."""

        observed_content_sha256 = self.recomputed_content_sha256()
        if not (
            self.artifact == ATOMIC_REFERENCE_DENSITY_ARTIFACT
            and self.construction == ATOMIC_REFERENCE_DENSITY_CONSTRUCTION
            and self.table_sha256 == ATOMIC_REFERENCE_DENSITY_TABLE_SHA256
            and self.manifest_sha256 == ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256
            and self.content_sha256 == ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256
            and observed_content_sha256 == ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256
        ):
            raise ValueError(
                "Atomic reference-density asset is not the frozen v1 scientific "
                "content."
            )


class AtomicReferenceDensity:
    """Superposition of atom-centred analytic reference-density mixtures."""

    def __init__(
        self,
        asset: AtomicReferenceDensityAsset,
        atomic_numbers: object,
        positions_bohr: object,
    ) -> None:
        if not isinstance(asset, AtomicReferenceDensityAsset):
            raise TypeError("Atomic reference density requires a verified asset.")
        asset.require_content_integrity()
        positions = _points(positions_bohr, name="Atomic positions in bohr")
        numbers = _atomic_numbers(atomic_numbers, atom_count=positions.shape[0])
        unsupported = sorted(
            set(numbers.tolist()) - set(asset.supported_atomic_numbers)
        )
        if unsupported:
            raise ValueError(
                "Atomic reference density has no Gaussian mixture for atomic "
                f"numbers {unsupported}."
            )
        self.asset = asset
        self.atomic_numbers = np.array(numbers, dtype=np.int64, copy=True)
        self.atomic_numbers.setflags(write=False)
        self.positions_bohr = np.array(positions, dtype=float, copy=True)
        self.positions_bohr.setflags(write=False)

    @property
    def atom_count(self) -> int:
        return int(self.positions_bohr.shape[0])

    @property
    def integrated_electron_count(self) -> float:
        return float(np.sum(self.atomic_numbers, dtype=float))

    def atom_spatial_jet(
        self,
        points_bohr: object,
        atom_index: int,
    ) -> DensitySpatialJet:
        points = _points(points_bohr, name="Reference-density evaluation points")
        if isinstance(atom_index, bool) or not isinstance(
            atom_index,
            (int, np.integer),
        ):
            raise TypeError("Atomic reference-density atom index must be an integer.")
        if not 0 <= int(atom_index) < self.atom_count:
            raise IndexError("Atomic reference-density atom index is out of range.")
        atom_index = int(atom_index)
        mixture = self.asset.mixture(int(self.atomic_numbers[atom_index]))
        x = points - self.positions_bohr[atom_index]
        radius_squared = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2
        alpha = mixture.gaussian_exponents_bohr2
        counts = mixture.electron_counts
        # Strict floating-point screening only: skip a component when even its
        # largest value over this point batch is below the smallest positive
        # float64 subnormal.  This changes no representable result and avoids
        # introducing a physical or geometry-dependent approximation cutoff.
        positive = counts > 0.0
        log_maximum_value = np.full(counts.shape, -np.inf, dtype=float)
        log_maximum_value[positive] = (
            np.log(counts[positive])
            + 1.5 * (np.log(alpha[positive]) - np.log(np.pi))
            - alpha[positive] * float(np.min(radius_squared))
        )
        active = log_maximum_value >= _LOG_MIN_SUBNORMAL_FLOAT64
        if not np.any(active):
            return DensitySpatialJet.zeros(points.shape[0])
        alpha = alpha[active]
        counts = counts[active]
        component_values = (
            counts[None, :]
            * (alpha[None, :] / np.pi) ** 1.5
            * np.exp(-radius_squared[:, None] * alpha[None, :])
        )
        # The density jet is part of the DROP geometry callback, so its
        # bitwise value must not depend on BLAS thread scheduling.  In
        # particular, GEMV for ``component_values @ alpha**m`` gave rare
        # last-bit normal changes and intermittent cold-replay failures.
        # Accumulate every Gaussian component in the frozen asset order.
        moment0 = np.zeros(points.shape[0], dtype=float)
        moment1 = np.zeros(points.shape[0], dtype=float)
        moment2 = np.zeros(points.shape[0], dtype=float)
        moment3 = np.zeros(points.shape[0], dtype=float)
        for component_index, exponent in enumerate(alpha):
            values = component_values[:, component_index]
            exponent_value = float(exponent)
            exponent_squared = exponent_value * exponent_value
            moment0 += values
            moment1 += values * exponent_value
            moment2 += values * exponent_squared
            moment3 += values * (exponent_squared * exponent_value)
        identity = np.eye(3)
        gradient = -2.0 * x * moment1[:, None]
        hessian = (
            4.0 * np.einsum("mi,mj->mij", x, x) * moment2[:, None, None]
            - 2.0 * identity[None, :, :] * moment1[:, None, None]
        )
        third = np.empty((x.shape[0], 3, 3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    delta_x = (
                        identity[i, j] * x[:, k]
                        + identity[i, k] * x[:, j]
                        + identity[j, k] * x[:, i]
                    )
                    third[:, i, j, k] = (
                        -8.0 * x[:, i] * x[:, j] * x[:, k] * moment3
                        + 4.0 * delta_x * moment2
                    )
        return DensitySpatialJet(moment0, gradient, hessian, third)

    def evaluate_spatial(self, points_bohr: object) -> DensitySpatialJet:
        points = _points(points_bohr, name="Reference-density evaluation points")
        total = DensitySpatialJet.zeros(points.shape[0])
        for atom_index in range(self.atom_count):
            total = total.plus(self.atom_spatial_jet(points, atom_index))
        return total


def load_atomic_reference_density_asset(
    *,
    table_path: str | Path,
    manifest_path: str | Path,
) -> AtomicReferenceDensityAsset:
    """Load and fully verify one frozen Gaussian-mixture reference asset."""

    table = Path(table_path)
    manifest = Path(manifest_path)
    try:
        observed_manifest_sha256 = _sha256(manifest)
        manifest_text = manifest.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"Cannot read atomic reference-density manifest: {exc}"
        ) from exc
    if observed_manifest_sha256 != ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256:
        raise ValueError(
            "Atomic reference-density manifest hash (SHA256) does not match the "
            "frozen v1 identity."
        )
    try:
        payload = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot parse atomic reference-density manifest: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Atomic reference-density manifest must be a JSON object.")
    if payload.get("artifact") != ATOMIC_REFERENCE_DENSITY_ARTIFACT:
        raise ValueError("Atomic reference-density artifact identity is invalid.")
    if payload.get("construction") != ATOMIC_REFERENCE_DENSITY_CONSTRUCTION:
        raise ValueError("Atomic reference-density construction identity is invalid.")
    if payload.get("schema_version") != ATOMIC_REFERENCE_DENSITY_SCHEMA_VERSION:
        raise ValueError("Atomic reference-density schema version is invalid.")
    if payload.get("status") != ATOMIC_REFERENCE_DENSITY_STATUS:
        raise ValueError("Atomic reference-density admission status is invalid.")
    if payload.get("source") != ATOMIC_REFERENCE_DENSITY_SOURCE:
        raise ValueError("Atomic reference-density source provenance is invalid.")
    if payload.get("generator") != {
        "path": ATOMIC_REFERENCE_DENSITY_GENERATOR_REPO_PATH,
        "sha256": ATOMIC_REFERENCE_DENSITY_GENERATOR_SHA256,
    }:
        raise ValueError("Atomic reference-density generator provenance is invalid.")
    runtime_environment = payload.get("runtime_environment")
    if not isinstance(runtime_environment, dict) or set(runtime_environment) != {
        "python",
        "numpy",
        "scipy",
    }:
        raise ValueError("Atomic reference-density runtime provenance is invalid.")
    table_record = payload.get("table")
    if not isinstance(table_record, dict):
        raise ValueError("Atomic reference-density manifest has no table record.")
    declared_table_sha256 = table_record.get("sha256")
    if table_record.get("path") != ATOMIC_REFERENCE_DENSITY_TABLE_REPO_PATH:
        raise ValueError("Atomic reference-density table path identity is invalid.")
    if declared_table_sha256 != ATOMIC_REFERENCE_DENSITY_TABLE_SHA256:
        raise ValueError("Atomic reference-density table SHA256 is invalid.")
    try:
        observed_table_sha256 = _sha256(table)
    except OSError as exc:
        raise ValueError(f"Cannot read atomic reference-density table: {exc}") from exc
    if observed_table_sha256 != ATOMIC_REFERENCE_DENSITY_TABLE_SHA256:
        raise ValueError(
            "Atomic reference-density table hash (SHA256) does not match the "
            "frozen v1 identity."
        )

    try:
        with np.load(table, allow_pickle=False) as archive:
            numbers = _atomic_numbers(archive["atomic_numbers"])
            offsets_raw = np.asarray(archive["component_offsets"])
            counts = np.asarray(archive["electron_counts"], dtype=float)
            exponents = np.asarray(
                archive["gaussian_exponents_bohr2"],
                dtype=float,
            )
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Cannot load atomic reference-density table: {exc}") from exc

    if (
        offsets_raw.ndim != 1
        or offsets_raw.shape != (numbers.size + 1,)
        or not np.all(np.isfinite(offsets_raw))
    ):
        raise ValueError("Atomic reference-density component offsets are invalid.")
    if np.any(np.diff(numbers) <= 0):
        raise ValueError(
            "Atomic reference-density atomic numbers must be unique and sorted."
        )
    offsets_numeric = np.asarray(offsets_raw, dtype=float)
    offsets = np.rint(offsets_numeric).astype(np.int64)
    if (
        np.any(offsets != offsets_numeric)
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or offsets[-1] != counts.size
        or counts.shape != exponents.shape
    ):
        raise ValueError("Atomic reference-density packed component table is invalid.")

    declared_numbers = payload.get("supported_atomic_numbers")
    if declared_numbers != numbers.tolist():
        raise ValueError("Atomic reference-density element coverage is inconsistent.")
    mixtures: dict[int, GaussianMixtureAtom] = {}
    for index, atomic_number in enumerate(numbers.tolist()):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        mixtures[int(atomic_number)] = GaussianMixtureAtom(
            electron_counts=counts[start:stop],
            gaussian_exponents_bohr2=exponents[start:stop],
        )

    return AtomicReferenceDensityAsset(
        mixtures_by_atomic_number=mixtures,
        table_sha256=observed_table_sha256,
        manifest_sha256=observed_manifest_sha256,
        provenance=payload,
    )


__all__ = [
    "ATOMIC_REFERENCE_DENSITY_ARTIFACT",
    "ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256",
    "ATOMIC_REFERENCE_DENSITY_CONSTRUCTION",
    "ATOMIC_REFERENCE_DENSITY_GENERATOR_SHA256",
    "ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256",
    "ATOMIC_REFERENCE_DENSITY_SCHEMA_VERSION",
    "ATOMIC_REFERENCE_DENSITY_STATUS",
    "ATOMIC_REFERENCE_DENSITY_TABLE_SHA256",
    "AtomicReferenceDensity",
    "AtomicReferenceDensityAsset",
    "DensitySpatialJet",
    "GaussianMixtureAtom",
    "load_atomic_reference_density_asset",
]
