"""Frozen atomic-displacement response source for Route-2 V0.

This module represents a deliberately narrow, no-fit spatial-source candidate.
Each atom's independently generated spherical free-atom electron density is
translated infinitesimally, producing a neutral induced density with a
prescribed atom-centred dipole. The frozen MACE-MDP atomic partition sets those
dipoles; it does not set a radial width.

For an atom with Z electrons and positive electron density n_Z(r), the
translated-electron response for an induced dipole p_a has exterior potential

    (p_a dot r) * N_Z(r) / (Z * |r|**3),

where N_Z(r) = 4*pi*integral_0^r n_Z(s) s**2 ds. This approaches the ordinary
point-dipole potential at infinity, but keeps a source-provenanced near-field
radial shape.

The construction is a gas-phase source falsifier only. It is not a MACE
electron density, a GTO/continuum dual basis, an energy/force model, a PCM
source, or a solvation calculation. A successful QM-MEP comparison would only
justify the next separately registered same-basis variational gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

V0_ATOMIC_DISPLACEMENT_RESPONSE_ARTIFACT = (
    "route2-v0-atomic-displacement-hf-def2-tzvpd-v1"
)
V0_ATOMIC_DISPLACEMENT_RESPONSE_CONSTRUCTION = (
    "spherical-free-atom-electron-translation-tangent-v1"
)
_PARTITION_TOLERANCE = 1.0e-10
# Keep the already-pinned ASE 2014 conversion while allowing the independent
# PySCF reference environment, which intentionally has no ASE dependency, to
# generate the frozen source asset.
BOHR_ANGSTROM = 0.5291772105638411


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
        raise ValueError(
            f"{name} must be a finite one-dimensional array of length >= 3."
        )
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


def _finite_positions(values: np.ndarray) -> np.ndarray:
    positions = np.asarray(values, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("Atom positions must be finite with shape (n_atoms, 3).")
    return positions


def _finite_points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("Evaluation points must be finite with shape (n_points, 3).")
    return points


def _atomic_partition(
    atomic_dipole_weights: np.ndarray,
    *,
    atom_count: int,
) -> np.ndarray:
    weights = np.asarray(atomic_dipole_weights, dtype=float)
    if weights.shape != (atom_count, 3, 3) or not np.all(np.isfinite(weights)):
        raise ValueError(
            "Atomic dipole weights must be finite with shape (n_atoms, 3, 3)."
        )
    mismatch = float(np.linalg.norm(np.sum(weights, axis=0) - np.eye(3), ord="fro"))
    if mismatch > _PARTITION_TOLERANCE:
        raise ValueError(
            "Atomic dipole weights do not preserve the molecular induced dipole."
        )
    return weights


def atomic_induced_dipoles(
    atomic_dipole_weights: np.ndarray,
    molecular_induced_dipole_ebohr: np.ndarray,
) -> np.ndarray:
    """Return frozen MACE-partitioned atom induced dipoles in e bohr."""

    dipole = np.asarray(molecular_induced_dipole_ebohr, dtype=float)
    if dipole.shape != (3,) or not np.all(np.isfinite(dipole)):
        raise ValueError(
            "Molecular induced dipole must be finite with shape (3,) in e bohr."
        )
    weights = _atomic_partition(
        atomic_dipole_weights,
        atom_count=len(atomic_dipole_weights),
    )
    result = np.einsum("aij,j->ai", weights, dipole, optimize=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0AtomicDisplacementResponseTable:
    """Immutable enclosed-electron tables for a translated free-atom source."""

    radial_grid_bohr: np.ndarray
    enclosed_electrons_by_atomic_number: Mapping[int, np.ndarray]
    table_sha256: str
    manifest_sha256: str
    construction: str = V0_ATOMIC_DISPLACEMENT_RESPONSE_CONSTRUCTION

    def __post_init__(self) -> None:
        radial_grid = _immutable_vector(self.radial_grid_bohr, name="Radial grid")
        if radial_grid[0] != 0.0 or np.any(np.diff(radial_grid) <= 0.0):
            raise ValueError(
                "Radial grid must start at zero and be strictly increasing."
            )
        if self.construction != V0_ATOMIC_DISPLACEMENT_RESPONSE_CONSTRUCTION:
            raise ValueError("Unsupported atomic-displacement-response construction.")
        if not isinstance(self.table_sha256, str) or len(self.table_sha256) != 64:
            raise ValueError("Atomic-displacement table SHA256 is invalid.")
        if not isinstance(self.manifest_sha256, str) or len(self.manifest_sha256) != 64:
            raise ValueError("Atomic-displacement manifest SHA256 is invalid.")

        frozen: dict[int, np.ndarray] = {}
        for (
            atomic_number,
            raw_enclosed,
        ) in self.enclosed_electrons_by_atomic_number.items():
            if isinstance(atomic_number, bool) or int(atomic_number) < 1:
                raise ValueError("Atomic-displacement atomic numbers must be positive.")
            number = int(atomic_number)
            enclosed = _immutable_vector(
                raw_enclosed,
                name=f"Enclosed electron count Z={number}",
            )
            if (
                enclosed.shape != radial_grid.shape
                or enclosed[0] != 0.0
                or np.any(np.diff(enclosed) < 0.0)
                or abs(float(enclosed[-1]) - number) > 1.0e-10
            ):
                raise ValueError(
                    "Enclosed electron counts must be monotone, match the radial "
                    "grid, start at zero, and end at the atomic number."
                )
            frozen[number] = enclosed
        if not frozen:
            raise ValueError(
                "Atomic-displacement table must contain at least one element."
            )
        object.__setattr__(self, "radial_grid_bohr", radial_grid)
        object.__setattr__(
            self,
            "enclosed_electrons_by_atomic_number",
            MappingProxyType(frozen),
        )

    @property
    def supported_atomic_numbers(self) -> tuple[int, ...]:
        """Return the frozen element domain in ascending atomic-number order."""

        return tuple(sorted(self.enclosed_electrons_by_atomic_number))

    def enclosed_electrons(
        self,
        atomic_number: int,
        radius_bohr: np.ndarray,
    ) -> np.ndarray:
        """Evaluate normalized enclosed electron count at the supplied radii."""

        if (
            isinstance(atomic_number, bool)
            or int(atomic_number) not in self.enclosed_electrons_by_atomic_number
        ):
            raise ValueError(
                f"Atomic-displacement response has no frozen reference for Z={atomic_number}."
            )
        radius = np.asarray(radius_bohr, dtype=float)
        if np.any(radius < 0.0) or not np.all(np.isfinite(radius)):
            raise ValueError(
                "Atomic-displacement radii must be finite and nonnegative."
            )
        number = int(atomic_number)
        return np.interp(
            radius,
            self.radial_grid_bohr,
            self.enclosed_electrons_by_atomic_number[number],
            right=float(number),
        )

    def induced_potential(
        self,
        points_bohr: np.ndarray,
        atomic_numbers: np.ndarray,
        atom_positions_angstrom: np.ndarray,
        atomic_dipole_weights: np.ndarray,
        molecular_induced_dipole_ebohr: np.ndarray,
    ) -> np.ndarray:
        """Return induced electronic potential in Hartree per elementary charge.

        Fixed nuclei do not move under this electronic response. This is a
        neutral induced electron-cloud source, not a translated neutral atom or
        an approximation to the static molecular density.
        """

        points = _finite_points(points_bohr)
        positions = _finite_positions(atom_positions_angstrom)
        numbers = _atomic_numbers(atomic_numbers, atom_count=positions.shape[0])
        unsupported = sorted(set(numbers.tolist()) - set(self.supported_atomic_numbers))
        if unsupported:
            raise ValueError(
                "Atomic-displacement response has no frozen reference for "
                f"atomic numbers {unsupported}."
            )
        atomic_dipoles = atomic_induced_dipoles(
            _atomic_partition(
                atomic_dipole_weights,
                atom_count=positions.shape[0],
            ),
            molecular_induced_dipole_ebohr,
        )

        potential = np.zeros(points.shape[0], dtype=float)
        positions_bohr = positions / BOHR_ANGSTROM
        for number, center, dipole in zip(
            numbers,
            positions_bohr,
            atomic_dipoles,
            strict=True,
        ):
            displacement = points - center
            radius = np.linalg.norm(displacement, axis=1)
            nonzero = radius > 1.0e-14
            if not np.any(nonzero):
                continue
            enclosed = self.enclosed_electrons(int(number), radius[nonzero])
            potential[nonzero] += (
                np.einsum("si,i->s", displacement[nonzero], dipole)
                * enclosed
                / (float(number) * radius[nonzero] ** 3)
            )
        potential.setflags(write=False)
        return potential


def load_route2_v0_atomic_displacement_response_table(
    *,
    table_path: Path,
    manifest_path: Path,
) -> Route2V0AtomicDisplacementResponseTable:
    """Load the tracked response asset and verify every content hash."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read atomic-displacement response manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise TypeError("Atomic-displacement manifest must contain one JSON object.")
    if (
        manifest.get("artifact") != V0_ATOMIC_DISPLACEMENT_RESPONSE_ARTIFACT
        or manifest.get("status") != "pass"
    ):
        raise ValueError("Atomic-displacement manifest identity or status is invalid.")
    table = manifest.get("table")
    if not isinstance(table, dict) or table.get("sha256") != _sha256(table_path):
        raise ValueError("Atomic-displacement table does not match its manifest hash.")

    try:
        archive = np.load(table_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read atomic-displacement table: {exc}") from exc
    with archive:
        required = {"radial_grid_bohr", "atomic_numbers"}
        if not required.issubset(archive.files):
            raise ValueError("Atomic-displacement table omits required arrays.")
        radial_grid = np.asarray(archive["radial_grid_bohr"], dtype=float)
        atomic_numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
        if atomic_numbers.ndim != 1 or atomic_numbers.size == 0:
            raise ValueError("Atomic-displacement atomic-number array is invalid.")
        enclosed: dict[int, np.ndarray] = {}
        for atomic_number in atomic_numbers:
            key = f"enclosed_electrons_Z{int(atomic_number)}"
            if key not in archive.files:
                raise ValueError(f"Atomic-displacement table omits {key}.")
            enclosed[int(atomic_number)] = np.asarray(archive[key], dtype=float)

    if manifest.get("radial_grid_sha256") != _sha256_array(radial_grid):
        raise ValueError(
            "Atomic-displacement radial grid does not match its manifest hash."
        )
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != len(enclosed):
        raise ValueError("Atomic-displacement manifest results are incomplete.")
    result_by_atomic_number: dict[int, dict[str, object]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("Atomic-displacement manifest result entry is invalid.")
        atomic_number = result.get("atomic_number")
        if isinstance(atomic_number, bool):
            raise TypeError("Atomic-displacement manifest atomic number is invalid.")
        try:
            key = int(atomic_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Atomic-displacement manifest atomic number is invalid."
            ) from exc
        result_by_atomic_number[key] = result
    if set(result_by_atomic_number) != set(enclosed):
        raise ValueError("Atomic-displacement manifest/table element sets disagree.")
    for atomic_number, values in enclosed.items():
        result = result_by_atomic_number[atomic_number]
        if result.get("enclosed_electrons_sha256") != _sha256_array(values):
            raise ValueError(
                "Atomic-displacement enclosed-electron table does not match "
                f"the manifest for Z={atomic_number}."
            )

    return Route2V0AtomicDisplacementResponseTable(
        radial_grid_bohr=radial_grid,
        enclosed_electrons_by_atomic_number=enclosed,
        table_sha256=_sha256(table_path),
        manifest_sha256=_sha256(manifest_path),
    )


__all__ = [
    "BOHR_ANGSTROM",
    "V0_ATOMIC_DISPLACEMENT_RESPONSE_ARTIFACT",
    "V0_ATOMIC_DISPLACEMENT_RESPONSE_CONSTRUCTION",
    "Route2V0AtomicDisplacementResponseTable",
    "atomic_induced_dipoles",
    "load_route2_v0_atomic_displacement_response_table",
]
