"""Frozen positive promolecular-density tables for Route-2 V0-FD-S.

This module evaluates a separately generated superposition of spherical
free-atom reference densities,

``n_ref(r; R) = sum_a n_atom,Z_a(|r - R_a|)``.

It intentionally has no MACE electrostatic role: the MACE Gaussian multipoles
remain the only electrostatic solute source.  ``n_ref`` is an independent,
positive input for a future short-range liquid coupling and cannot be used as
a surrogate MACE electron density, a fitted cavity, or a total solvent model.
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
from ase.units import Bohr


V0_PROMOLECULAR_DENSITY_ARTIFACT = "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1"
V0_PROMOLECULAR_DENSITY_CONSTRUCTION = "spherical-free-atom-hf-promolecule-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _immutable_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array of length >= 3.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _atomic_numbers(values: np.ndarray, *, atom_count: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (atom_count,) or not np.all(np.isfinite(raw)):
        raise ValueError("Atomic numbers must be finite with shape (n_atoms,).")
    numeric = np.asarray(raw, dtype=float)
    rounded = np.rint(numeric)
    if np.any(rounded != numeric) or np.any(rounded < 1.0):
        raise ValueError("Atomic numbers must be positive integers.")
    return rounded.astype(np.int64, copy=False)


@dataclass(frozen=True)
class Route2V0PromolecularDensityTable:
    """Immutable radial free-atom density data with strict source identity."""

    radial_grid_bohr: np.ndarray
    densities_by_atomic_number: Mapping[int, np.ndarray]
    table_sha256: str
    manifest_sha256: str
    construction: str = V0_PROMOLECULAR_DENSITY_CONSTRUCTION

    def __post_init__(self) -> None:
        radial_grid = _immutable_vector(self.radial_grid_bohr, name="Radial grid")
        if radial_grid[0] != 0.0 or np.any(np.diff(radial_grid) <= 0.0):
            raise ValueError("Radial grid must start at zero and be strictly increasing.")
        if self.construction != V0_PROMOLECULAR_DENSITY_CONSTRUCTION:
            raise ValueError("Unsupported promolecular-density construction.")
        if not isinstance(self.table_sha256, str) or len(self.table_sha256) != 64:
            raise ValueError("Promolecular table SHA256 is invalid.")
        if not isinstance(self.manifest_sha256, str) or len(self.manifest_sha256) != 64:
            raise ValueError("Promolecular manifest SHA256 is invalid.")

        frozen: dict[int, np.ndarray] = {}
        for atomic_number, raw_density in self.densities_by_atomic_number.items():
            if isinstance(atomic_number, bool) or int(atomic_number) < 1:
                raise ValueError("Promolecular atomic numbers must be positive integers.")
            density = _immutable_vector(
                raw_density,
                name=f"Promolecular density Z={atomic_number}",
            )
            if density.shape != radial_grid.shape or np.any(density < 0.0):
                raise ValueError(
                    "Promolecular atomic densities must be nonnegative and match "
                    "the radial grid."
                )
            frozen[int(atomic_number)] = density
        if not frozen:
            raise ValueError("Promolecular table must contain at least one element.")
        object.__setattr__(self, "radial_grid_bohr", radial_grid)
        object.__setattr__(
            self,
            "densities_by_atomic_number",
            MappingProxyType(frozen),
        )

    @property
    def supported_atomic_numbers(self) -> tuple[int, ...]:
        """Return the frozen element domain in ascending atomic-number order."""

        return tuple(sorted(self.densities_by_atomic_number))

    def evaluate(
        self,
        points_bohr: np.ndarray,
        atomic_numbers: np.ndarray,
        atom_positions_angstrom: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the positive promolecular density in electrons per Bohr cubed.

        The density is a frozen independent reference.  At distances beyond the
        last tabulated radius the atomic contribution is exactly zero, which is
        valid only because the generating contract certifies a negligible tail
        at that radius.
        """

        points = np.asarray(points_bohr, dtype=float)
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("Evaluation points must be finite with shape (n_points, 3).")
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "Atom positions must be finite with shape (n_atoms, 3)."
            )
        numbers = _atomic_numbers(atomic_numbers, atom_count=positions.shape[0])
        unsupported = sorted(set(numbers.tolist()) - set(self.supported_atomic_numbers))
        if unsupported:
            raise ValueError(
                "Promolecular density has no frozen reference for atomic numbers "
                f"{unsupported}."
            )

        reference = np.zeros(points.shape[0], dtype=float)
        positions_bohr = positions / Bohr
        for atomic_number, center in zip(numbers, positions_bohr, strict=True):
            radius = np.linalg.norm(points - center, axis=1)
            reference += np.interp(
                radius,
                self.radial_grid_bohr,
                self.densities_by_atomic_number[int(atomic_number)],
                right=0.0,
            )
        reference.setflags(write=False)
        return reference


def load_route2_v0_promolecular_density_table(
    *,
    table_path: Path,
    manifest_path: Path,
) -> Route2V0PromolecularDensityTable:
    """Load one frozen V0-FD-S promolecular table and verify every hash.

    Arbitrary replacement tables are rejected unless their manifest identifies
    the exact V0 artifact and validates both the full table and each radial
    density vector.  This is a research-data loader, not a public custom
    solvent or custom solute API.
    """

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read promolecular manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Promolecular manifest must contain one JSON object.")
    if (
        manifest.get("artifact") != V0_PROMOLECULAR_DENSITY_ARTIFACT
        or manifest.get("status") != "pass"
    ):
        raise ValueError("Promolecular manifest identity or status is invalid.")
    table = manifest.get("table")
    if not isinstance(table, dict) or table.get("sha256") != _sha256(table_path):
        raise ValueError("Promolecular table does not match its manifest hash.")

    try:
        archive = np.load(table_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read promolecular table: {exc}") from exc
    with archive:
        required = {"radial_grid_bohr", "atomic_numbers"}
        if not required.issubset(archive.files):
            raise ValueError("Promolecular table omits required arrays.")
        radial_grid = np.asarray(archive["radial_grid_bohr"], dtype=float)
        atomic_numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
        if atomic_numbers.ndim != 1 or atomic_numbers.size == 0:
            raise ValueError("Promolecular table atomic-number array is invalid.")
        densities: dict[int, np.ndarray] = {}
        for atomic_number in atomic_numbers:
            key = f"density_Z{int(atomic_number)}"
            if key not in archive.files:
                raise ValueError(f"Promolecular table omits {key}.")
            densities[int(atomic_number)] = np.asarray(archive[key], dtype=float)

    if manifest.get("radial_grid_sha256") != _sha256_array(radial_grid):
        raise ValueError("Promolecular radial grid does not match its manifest hash.")
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != len(densities):
        raise ValueError("Promolecular manifest results are incomplete.")
    result_by_atomic_number: dict[int, dict[str, object]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Promolecular manifest result entry is invalid.")
        atomic_number = result.get("atomic_number")
        if isinstance(atomic_number, bool):
            raise ValueError("Promolecular manifest atomic number is invalid.")
        try:
            key = int(atomic_number)
        except (TypeError, ValueError) as exc:
            raise ValueError("Promolecular manifest atomic number is invalid.") from exc
        result_by_atomic_number[key] = result
    if set(result_by_atomic_number) != set(densities):
        raise ValueError("Promolecular manifest/table element sets disagree.")
    for atomic_number, density in densities.items():
        result = result_by_atomic_number[atomic_number]
        if result.get("density_sha256") != _sha256_array(density):
            raise ValueError(
                f"Promolecular density Z={atomic_number} does not match its manifest."
            )

    return Route2V0PromolecularDensityTable(
        radial_grid_bohr=radial_grid,
        densities_by_atomic_number=densities,
        table_sha256=_sha256(table_path),
        manifest_sha256=_sha256(manifest_path),
    )


__all__ = [
    "Route2V0PromolecularDensityTable",
    "V0_PROMOLECULAR_DENSITY_ARTIFACT",
    "V0_PROMOLECULAR_DENSITY_CONSTRUCTION",
    "load_route2_v0_promolecular_density_table",
]
