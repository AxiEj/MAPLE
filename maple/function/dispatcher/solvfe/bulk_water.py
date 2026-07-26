from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)

from .protocol import canonical_sha256, raw_sha256
from .provenance import (
    collect_implementation_provenance,
    require_coherent_git_status,
)


MACE_MD_WATERBOX_COMMIT = "e19729524fc91920169d4e193e4edd55bc4c5707"
MACE_MD_WATERBOX_URL = (
    "https://raw.githubusercontent.com/jharrymoore/mace-md/"
    f"{MACE_MD_WATERBOX_COMMIT}/examples/example_data/waterbox.xyz"
)
MACE_MD_WATERBOX_SHA256 = (
    "a052257f5f9c068884ec7527d6dd41d05a7c3705b729e7d9703543890931ec61"
)

_AMU_PER_ANGSTROM3_TO_G_PER_ML = 1.66053906660
_SHA256_CHARS = frozenset("0123456789abcdef")


class BulkWaterValidationError(ValueError):
    """Raised when a bulk-water source or run violates its evidence contract."""


def _require_sha256(value: str, *, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in _SHA256_CHARS for character in normalized
    ):
        raise BulkWaterValidationError(
            f"{label} must be 64 lowercase hexadecimal characters."
        )
    return normalized


def _require_git_commit(value: str, *, label: str) -> str:
    normalized = str(value)
    if len(normalized) not in {40, 64} or any(
        character not in _SHA256_CHARS for character in normalized
    ):
        raise BulkWaterValidationError(
            f"{label} must be a 40- or 64-character lowercase Git object ID."
        )
    return normalized


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BulkWaterNVTConfig:
    """Three-stage NVT validation protocol for a periodic water Hamiltonian."""

    temperature_k: float = 298.15
    timestep_fs: float = 0.5
    thermalization_steps: int = 1_000
    thermalization_friction_per_fs: float = 0.01
    equilibration_steps: int = 9_000
    equilibration_friction_per_fs: float = 0.001
    production_steps: int = 20_000
    production_friction_per_fs: float = 0.001
    sample_interval_steps: int = 20
    seed: int = 20_260_726
    rdf_bin_width_angstrom: float = 0.05
    rdf_max_angstrom: float = 6.0
    rdf_block_count: int = 5
    minimum_diagnostic_duration_ps: float = 10.0
    minimum_diagnostic_frames: int = 500
    temperature_relative_tolerance: float = 0.10
    maximum_temperature_k: float = 1_000.0
    maximum_force_ev_per_angstrom: float = 50.0

    def __post_init__(self) -> None:
        positive = {
            "temperature_k": self.temperature_k,
            "timestep_fs": self.timestep_fs,
            "thermalization_friction_per_fs": (
                self.thermalization_friction_per_fs
            ),
            "equilibration_friction_per_fs": (
                self.equilibration_friction_per_fs
            ),
            "production_friction_per_fs": self.production_friction_per_fs,
            "rdf_bin_width_angstrom": self.rdf_bin_width_angstrom,
            "rdf_max_angstrom": self.rdf_max_angstrom,
            "minimum_diagnostic_duration_ps": (
                self.minimum_diagnostic_duration_ps
            ),
            "temperature_relative_tolerance": (
                self.temperature_relative_tolerance
            ),
            "maximum_temperature_k": self.maximum_temperature_k,
            "maximum_force_ev_per_angstrom": (
                self.maximum_force_ev_per_angstrom
            ),
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise BulkWaterValidationError(f"{name} must be finite and positive.")
        integer_fields = (
            "thermalization_steps",
            "equilibration_steps",
            "production_steps",
            "sample_interval_steps",
            "rdf_block_count",
            "minimum_diagnostic_frames",
            "seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int:
                raise BulkWaterValidationError(f"{name} must be an integer.")
        for name in (
            "thermalization_steps",
            "equilibration_steps",
            "production_steps",
        ):
            if getattr(self, name) <= 0:
                raise BulkWaterValidationError(f"{name} must be positive.")
        if self.sample_interval_steps <= 0:
            raise BulkWaterValidationError(
                "sample_interval_steps must be positive."
            )
        if self.production_steps % self.sample_interval_steps:
            raise BulkWaterValidationError(
                "production_steps must be divisible by sample_interval_steps."
            )
        if self.production_steps // self.sample_interval_steps < 2:
            raise BulkWaterValidationError(
                "production must contain at least two sampled frames."
            )
        production_frame_count = (
            self.production_steps // self.sample_interval_steps
        )
        if self.rdf_block_count < 2:
            raise BulkWaterValidationError("rdf_block_count must be at least two.")
        if production_frame_count % self.rdf_block_count:
            raise BulkWaterValidationError(
                "production frame count must be divisible by rdf_block_count."
            )
        if self.minimum_diagnostic_frames < 2:
            raise BulkWaterValidationError(
                "minimum_diagnostic_frames must be at least two."
            )
        if self.seed < 0:
            raise BulkWaterValidationError("seed must be a non-negative integer.")

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "schema": "maple-route-a-bulk-water-nvt-config-v2",
                **asdict(self),
            }
        )


@dataclass
class _StepwiseStabilityMonitor:
    config: BulkWaterNVTConfig
    checked_steps: set[int] = field(default_factory=set)
    maximum_temperature_k: float = -math.inf
    maximum_force_ev_per_angstrom: float = -math.inf
    minimum_oh_distance_angstrom: float = math.inf
    maximum_oh_distance_angstrom: float = -math.inf
    minimum_hoh_angle_degrees: float = math.inf
    maximum_hoh_angle_degrees: float = -math.inf

    def inspect(
        self,
        atoms: Atoms,
        *,
        stage: str,
        step: int,
    ) -> dict[str, float]:
        try:
            topology = validate_water_box(atoms)
        except BulkWaterValidationError as exc:
            raise BulkWaterValidationError(
                "DYNAMICS_TOPOLOGY_FAILURE: "
                f"stage={stage}, step={step}: {exc}"
            ) from exc
        forces = np.asarray(atoms.get_forces(), dtype=float)
        velocities = np.asarray(atoms.get_velocities(), dtype=float)
        temperature = float(atoms.get_temperature())
        potential = float(atoms.get_potential_energy())
        if (
            not np.all(np.isfinite(forces))
            or not np.all(np.isfinite(velocities))
            or not math.isfinite(temperature)
            or not math.isfinite(potential)
        ):
            raise BulkWaterValidationError(
                "DYNAMICS_NONFINITE: "
                f"stage={stage}, step={step} contains non-finite state."
            )
        force_max = float(np.max(np.linalg.norm(forces, axis=1)))
        self.checked_steps.add(int(step))
        self.maximum_temperature_k = max(
            self.maximum_temperature_k,
            temperature,
        )
        self.maximum_force_ev_per_angstrom = max(
            self.maximum_force_ev_per_angstrom,
            force_max,
        )
        self.minimum_oh_distance_angstrom = min(
            self.minimum_oh_distance_angstrom,
            topology["oh_distance_range_angstrom"][0],
        )
        self.maximum_oh_distance_angstrom = max(
            self.maximum_oh_distance_angstrom,
            topology["oh_distance_range_angstrom"][1],
        )
        self.minimum_hoh_angle_degrees = min(
            self.minimum_hoh_angle_degrees,
            topology["hoh_angle_range_degrees"][0],
        )
        self.maximum_hoh_angle_degrees = max(
            self.maximum_hoh_angle_degrees,
            topology["hoh_angle_range_degrees"][1],
        )
        if temperature > self.config.maximum_temperature_k:
            raise BulkWaterValidationError(
                "DYNAMICS_TEMPERATURE_LIMIT: "
                f"stage={stage}, step={step}, temperature={temperature:.8g} K."
            )
        if force_max > self.config.maximum_force_ev_per_angstrom:
            raise BulkWaterValidationError(
                "DYNAMICS_FORCE_LIMIT: "
                f"stage={stage}, step={step}, max_force={force_max:.8g} eV/A."
            )
        return {
            "temperature_k": temperature,
            "potential_energy_ev": potential,
            "velocity_max_angstrom_per_ase_time": float(
                np.max(np.linalg.norm(velocities, axis=1))
            ),
            "force_max_ev_per_angstrom": force_max,
            "oh_min_distance_angstrom": topology[
                "oh_distance_range_angstrom"
            ][0],
            "oh_max_distance_angstrom": topology[
                "oh_distance_range_angstrom"
            ][1],
            "hoh_min_angle_degrees": topology[
                "hoh_angle_range_degrees"
            ][0],
            "hoh_max_angle_degrees": topology[
                "hoh_angle_range_degrees"
            ][1],
        }

    def as_dict(self, *, expected_last_step: int) -> dict[str, Any]:
        expected_count = int(expected_last_step) + 1
        return {
            "checked_unique_steps": len(self.checked_steps),
            "expected_unique_steps": expected_count,
            "all_steps_checked": self.checked_steps
            == set(range(expected_count)),
            "maximum_temperature_k": self.maximum_temperature_k,
            "maximum_force_ev_per_angstrom": (
                self.maximum_force_ev_per_angstrom
            ),
            "oh_distance_range_angstrom": [
                self.minimum_oh_distance_angstrom,
                self.maximum_oh_distance_angstrom,
            ],
            "hoh_angle_range_degrees": [
                self.minimum_hoh_angle_degrees,
                self.maximum_hoh_angle_degrees,
            ],
            "topology_preserved": True,
        }


@dataclass(frozen=True)
class BulkWaterValidationResult:
    """Immutable in-memory result with semantic hashes for every numeric array."""

    result_hash: str
    summary: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    artifact_schema: str = "maple-route-a-bulk-water-validation-artifact-v2"

    def write(self, output_directory: str | Path) -> Path:
        root = Path(output_directory).expanduser().resolve()
        if canonical_sha256(self.summary) != self.result_hash:
            raise BulkWaterValidationError(
                "RESULT_HASH_MISMATCH: bulk-water summary changed before write."
            )
        actual_semantic_hashes = {
            name: _array_sha256(values)
            for name, values in sorted(self.arrays.items())
        }
        expected_semantic_hashes = self.summary.get("trajectory", {}).get(
            "semantic_array_sha256"
        )
        if actual_semantic_hashes != expected_semantic_hashes:
            raise BulkWaterValidationError(
                "RESULT_HASH_MISMATCH: bulk-water arrays changed before write."
            )
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise BulkWaterValidationError(
                f"OUTPUT_EXISTS: bulk-water result path already exists: {root}"
            ) from exc

        arrays_path = root / "arrays.npz"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".arrays.",
            suffix=".npz.tmp",
            dir=root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **self.arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, arrays_path)
        finally:
            temporary.unlink(missing_ok=True)

        arrays_sha256 = raw_sha256(arrays_path)
        summary = dict(self.summary)
        summary["result_hash"] = self.result_hash
        _atomic_json(root / "summary.json", summary)
        manifest = {
            "schema": self.artifact_schema,
            "result_hash": self.result_hash,
            "files": {
                "arrays": {
                    "path": "arrays.npz",
                    "sha256": arrays_sha256,
                },
                "summary": {
                    "path": "summary.json",
                    "canonical_sha256": canonical_sha256(summary),
                },
            },
            "semantic_array_sha256": actual_semantic_hashes,
        }
        _atomic_json(root / "manifest.json", manifest)
        return root


def _atomic_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def water_density_g_per_ml(atoms: Atoms) -> float:
    volume = float(atoms.get_volume())
    if not math.isfinite(volume) or volume <= 0.0:
        raise BulkWaterValidationError(
            "Bulk-water density requires a finite positive periodic volume."
        )
    return (
        float(np.sum(atoms.get_masses()))
        / volume
        * _AMU_PER_ANGSTROM3_TO_G_PER_ML
    )


def _water_molecule_ids(atoms: Atoms) -> np.ndarray:
    numbers = np.asarray(atoms.numbers, dtype=int)
    if len(numbers) == 0 or len(numbers) % 3:
        raise BulkWaterValidationError(
            "WATER_TOPOLOGY_INVALID: expected contiguous O-H-H triplets."
        )
    triplets = numbers.reshape((-1, 3))
    if not np.all(triplets == np.array([8, 1, 1], dtype=int)):
        raise BulkWaterValidationError(
            "WATER_TOPOLOGY_INVALID: atom order must be contiguous O-H-H triplets."
        )
    return np.repeat(np.arange(len(triplets), dtype=int), 3)


def _safe_rdf_radius(atoms: Atoms) -> float:
    cell = np.asarray(atoms.cell.array, dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    heights = []
    for index in range(3):
        other = [value for value in range(3) if value != index]
        face_area = float(np.linalg.norm(np.cross(cell[other[0]], cell[other[1]])))
        if face_area <= 0.0:
            raise BulkWaterValidationError(
                "Bulk-water RDF requires a non-singular three-dimensional cell."
            )
        heights.append(volume / face_area)
    return 0.5 * min(heights)


def validate_water_box(atoms: Atoms, *, expected_waters: int | None = None) -> dict[str, Any]:
    if not bool(np.all(atoms.get_pbc())):
        raise BulkWaterValidationError(
            "WATER_BOUNDARY_INVALID: bulk-water validation requires full 3D PBC."
        )
    if atoms.constraints:
        raise BulkWaterValidationError(
            "WATER_CONSTRAINTS_UNSUPPORTED: the flexible-water validation "
            "contract forbids ASE constraints."
        )
    molecule_ids = _water_molecule_ids(atoms)
    default_masses = Atoms(numbers=atoms.numbers).get_masses()
    if not np.array_equal(
        np.asarray(atoms.get_masses(), dtype=float),
        np.asarray(default_masses, dtype=float),
    ):
        raise BulkWaterValidationError(
            "WATER_MASSES_UNSUPPORTED: atoms must use ASE default isotope masses."
        )
    water_count = int(molecule_ids[-1]) + 1
    if expected_waters is not None and water_count != int(expected_waters):
        raise BulkWaterValidationError(
            f"WATER_COUNT_MISMATCH: expected {expected_waters}, found {water_count}."
        )

    positions = np.asarray(atoms.positions, dtype=float)
    if not np.all(np.isfinite(positions)):
        raise BulkWaterValidationError(
            "WATER_GEOMETRY_INVALID: positions must be finite."
        )
    oh_vectors = np.concatenate(
        (
            positions[1::3] - positions[0::3],
            positions[2::3] - positions[0::3],
        ),
        axis=0,
    )
    mic_vectors, oh_distances = find_mic(
        oh_vectors,
        atoms.cell,
        pbc=atoms.pbc,
    )
    if (
        not np.all(np.isfinite(oh_distances))
        or np.any(oh_distances < 0.70)
        or np.any(oh_distances > 1.30)
    ):
        raise BulkWaterValidationError(
            "WATER_GEOMETRY_INVALID: O-H distances must lie in [0.70, 1.30] A."
        )

    first = mic_vectors[:water_count]
    second = mic_vectors[water_count:]
    cosine = np.sum(first * second, axis=1) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    if (
        not np.all(np.isfinite(angles))
        or np.any(angles < 70.0)
        or np.any(angles > 140.0)
    ):
        raise BulkWaterValidationError(
            "WATER_GEOMETRY_INVALID: H-O-H angles must lie in [70, 140] degrees."
        )

    return {
        "water_count": water_count,
        "atom_count": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "volume_angstrom3": float(atoms.get_volume()),
        "density_g_per_ml": water_density_g_per_ml(atoms),
        "oh_distance_range_angstrom": [
            float(np.min(oh_distances)),
            float(np.max(oh_distances)),
        ],
        "hoh_angle_range_degrees": [
            float(np.min(angles)),
            float(np.max(angles)),
        ],
        "safe_rdf_radius_angstrom": _safe_rdf_radius(atoms),
    }


def replicate_water_box(
    atoms: Atoms,
    repetitions: tuple[int, int, int],
) -> Atoms:
    """Replicate contiguous O-H-H waters without breaking wrapped molecules."""

    topology = validate_water_box(atoms)
    try:
        normalized = tuple(repetitions)
    except TypeError as exc:
        raise BulkWaterValidationError(
            "WATER_REPLICATION_INVALID: repetitions must contain three "
            "positive integers."
        ) from exc
    if (
        len(normalized) != 3
        or any(type(value) is not int or value <= 0 for value in normalized)
    ):
        raise BulkWaterValidationError(
            "WATER_REPLICATION_INVALID: repetitions must contain three "
            "positive integers."
        )

    unwrapped = atoms.copy()
    unwrapped.calc = None
    positions = np.asarray(unwrapped.positions, dtype=float).copy()
    oxygen_positions = positions[0::3].copy()
    for offset in (1, 2):
        vectors = positions[offset::3] - oxygen_positions
        mic_vectors, _ = find_mic(
            vectors,
            unwrapped.cell,
            pbc=unwrapped.pbc,
        )
        positions[offset::3] = oxygen_positions + mic_vectors
    unwrapped.set_positions(positions)

    replicated = unwrapped.repeat(normalized)
    replicated.calc = None
    expected_waters = topology["water_count"] * math.prod(normalized)
    validate_water_box(
        replicated,
        expected_waters=expected_waters,
    )
    return replicated


def load_water_box(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_waters: int | None = None,
    source_url: str | None = None,
    source_commit: str | None = None,
) -> tuple[Atoms, dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    expected_hash = _require_sha256(
        expected_sha256,
        label="expected_sha256",
    )
    if not source_path.is_file():
        raise BulkWaterValidationError(
            f"WATER_SOURCE_MISSING: no file at {source_path}."
        )
    actual_hash = raw_sha256(source_path)
    if actual_hash != expected_hash:
        raise BulkWaterValidationError(
            "WATER_SOURCE_HASH_MISMATCH: "
            f"expected {expected_hash}, found {actual_hash}."
        )
    atoms = read(source_path, index=0)
    topology = validate_water_box(atoms, expected_waters=expected_waters)
    source = {
        "schema": "maple-route-a-bulk-water-source-v1",
        "path": source_path.as_posix(),
        "sha256": actual_hash,
        "url": source_url,
        "commit": source_commit,
        "topology": topology,
    }
    return atoms, source


def _validated_run_provenance(
    initial_atoms: Atoms,
    calculator: Calculator,
    *,
    source_provenance: Mapping[str, Any],
    calculator_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
    required_implementation_paths: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = dict(source_provenance)
    if source.get("schema") != "maple-route-a-bulk-water-source-v1":
        raise BulkWaterValidationError(
            "SOURCE_PROVENANCE_INVALID: source schema is missing or unsupported."
        )
    source_hash = _require_sha256(
        source.get("sha256", ""),
        label="source sha256",
    )
    source_path = Path(str(source.get("path", ""))).expanduser().resolve()
    if not source_path.is_file() or raw_sha256(source_path) != source_hash:
        raise BulkWaterValidationError(
            "SOURCE_PROVENANCE_INVALID: source path does not match its SHA256."
        )
    source_atoms = read(source_path, index=0)
    if _structure_evidence(source_atoms) != _structure_evidence(initial_atoms):
        raise BulkWaterValidationError(
            "SOURCE_PROVENANCE_INVALID: initial atoms differ from the "
            "hash-verified source structure."
        )
    source_topology = source.get("topology")
    if not isinstance(source_topology, Mapping) or dict(
        source_topology
    ) != validate_water_box(source_atoms):
        raise BulkWaterValidationError(
            "SOURCE_PROVENANCE_INVALID: source topology evidence is incomplete."
        )

    calculator_record = dict(calculator_provenance)
    if (
        calculator_record.get("schema")
        != "maple-route-a-bulk-water-calculator-v1"
    ):
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: calculator schema is missing "
            "or unsupported."
        )
    declared_calculator = calculator_record.get("calculator")
    if not isinstance(declared_calculator, Mapping):
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: calculator identity is missing."
        )
    declared_calculator = dict(declared_calculator)
    checkpoint_hash = _require_sha256(
        declared_calculator.get("checkpoint_sha256", ""),
        label="calculator checkpoint_sha256",
    )
    checkpoint_path = Path(
        str(declared_calculator.get("checkpoint", ""))
    ).expanduser().resolve()
    if (
        not checkpoint_path.is_file()
        or raw_sha256(checkpoint_path) != checkpoint_hash
    ):
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: checkpoint path does not match "
            "its SHA256."
        )
    expected_result_units = {
        "energy": "eV",
        "forces": "eV/angstrom",
        "stress": "eV/angstrom^3",
    }
    if declared_calculator.get("result_units") != expected_result_units:
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: result units must be the Route A "
            "eV-native energy/force/stress contract."
        )
    declared_default_dtype = declared_calculator.get("default_dtype")
    if (
        not isinstance(declared_default_dtype, str)
        or declared_default_dtype not in {"float32", "float64"}
    ):
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: default dtype must be float32 "
            "or float64."
        )
    try:
        interaction_cutoff = float(
            declared_calculator["interaction_cutoff_angstrom"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: interaction cutoff is missing "
            "or invalid."
        ) from exc
    if not math.isfinite(interaction_cutoff) or interaction_cutoff <= 0.0:
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: interaction cutoff must be "
            "finite and positive."
        )
    runtime_calculator_provenance = getattr(calculator, "provenance", None)
    if not isinstance(runtime_calculator_provenance, Mapping):
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: calculator must expose a "
            "provenance mapping."
        )
    if dict(runtime_calculator_provenance) != declared_calculator:
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: runtime calculator identity does "
            "not exactly match the declared record."
        )
    if not isinstance(calculator_record.get("runtime"), Mapping):
        raise BulkWaterValidationError(
            "CALCULATOR_PROVENANCE_INVALID: runtime environment evidence is "
            "missing."
        )

    implementation = dict(implementation_provenance)
    if (
        implementation.get("schema")
        != "maple-route-a-bulk-water-implementation-v1"
    ):
        raise BulkWaterValidationError(
            "IMPLEMENTATION_PROVENANCE_INVALID: implementation schema is "
            "missing or unsupported."
        )
    project_root = Path(
        str(implementation.get("project_root", ""))
    ).expanduser().resolve()
    executing_project_root = Path(__file__).resolve().parents[4]
    if project_root != executing_project_root:
        raise BulkWaterValidationError(
            "IMPLEMENTATION_PROVENANCE_INVALID: project_root is not the "
            "currently executing MAPLE source tree."
        )
    try:
        implementation["git_head"] = _require_git_commit(
            implementation["git_head"],
            label="implementation Git HEAD",
        )
        implementation["git_status_sha256"] = _require_sha256(
            implementation["git_status_sha256"],
            label="implementation Git status SHA256",
        )
    except (KeyError, TypeError) as exc:
        raise BulkWaterValidationError(
            "IMPLEMENTATION_PROVENANCE_INVALID: required Git identity is "
            "missing."
        ) from exc
    if type(implementation.get("git_dirty")) is not bool:
        raise BulkWaterValidationError(
            "IMPLEMENTATION_PROVENANCE_INVALID: Git dirty state must be "
            "boolean."
        )
    try:
        require_coherent_git_status(
            implementation["git_dirty"],
            implementation["git_status_sha256"],
        )
    except ValueError as exc:
        raise BulkWaterValidationError(
            "IMPLEMENTATION_PROVENANCE_INVALID: "
            f"{exc}"
        ) from exc
    file_hashes = implementation.get("implementation_file_sha256")
    if not project_root.is_dir() or not isinstance(file_hashes, Mapping) or not file_hashes:
        raise BulkWaterValidationError(
            "IMPLEMENTATION_PROVENANCE_INVALID: project root or file hashes "
            "are missing."
        )
    normalized_file_hashes = {}
    for relative_name, declared_hash in file_hashes.items():
        relative_path = Path(str(relative_name))
        if relative_path.is_absolute():
            raise BulkWaterValidationError(
                "IMPLEMENTATION_PROVENANCE_INVALID: implementation paths "
                "must be project-relative."
            )
        candidate = (project_root / relative_path).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as exc:
            raise BulkWaterValidationError(
                "IMPLEMENTATION_PROVENANCE_INVALID: implementation path "
                "escapes the project root."
            ) from exc
        expected_hash = _require_sha256(
            declared_hash,
            label=f"implementation hash for {relative_path.as_posix()}",
        )
        if not candidate.is_file() or raw_sha256(candidate) != expected_hash:
            raise BulkWaterValidationError(
                "IMPLEMENTATION_PROVENANCE_INVALID: implementation file "
                f"'{relative_path.as_posix()}' does not match its SHA256."
            )
        normalized_file_hashes[relative_path.as_posix()] = expected_hash
    if required_implementation_paths is None:
        required_implementation_paths = {
            "maple/function/dispatcher/solvfe/bulk_water.py",
            "maple/function/dispatcher/solvfe/protocol.py",
            "maple/function/dispatcher/solvfe/provenance.py",
            "maple/function/calculator/mace/_mace_upstream_calculator.py",
            "examples/solvation/route_a/validate_bulk_water.py",
        }
    missing_implementation_paths = (
        required_implementation_paths - normalized_file_hashes.keys()
    )
    if missing_implementation_paths:
        raise BulkWaterValidationError(
            "IMPLEMENTATION_PROVENANCE_INVALID: required implementation "
            "hashes are missing: "
            + ", ".join(sorted(missing_implementation_paths))
            + "."
        )
    current_implementation = collect_implementation_provenance(
        project_root,
        normalized_file_hashes,
    )
    for name in ("git_head", "git_dirty", "git_status_sha256"):
        if implementation[name] != current_implementation[name]:
            raise BulkWaterValidationError(
                "IMPLEMENTATION_PROVENANCE_INVALID: declared Git identity "
                "does not match the executing source tree."
            )
    implementation["project_root"] = project_root.as_posix()
    implementation["implementation_file_sha256"] = normalized_file_hashes
    return source, calculator_record, implementation


def _structure_evidence(atoms: Atoms) -> dict[str, Any]:
    return {
        "atomic_numbers_sha256": _array_sha256(
            np.asarray(atoms.numbers, dtype=np.int64)
        ),
        "masses_sha256": _array_sha256(
            np.asarray(atoms.get_masses(), dtype=np.float64)
        ),
        "positions_sha256": _array_sha256(
            np.asarray(atoms.positions, dtype=np.float64)
        ),
        "cell_sha256": _array_sha256(
            np.asarray(atoms.cell.array, dtype=np.float64)
        ),
        "pbc": [bool(value) for value in atoms.pbc],
        "atom_count": len(atoms),
        "formula": atoms.get_chemical_formula(),
    }


def _pair_indices(
    numbers: np.ndarray,
    molecule_ids: np.ndarray,
    first_atomic_number: int,
    second_atomic_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.flatnonzero(numbers == first_atomic_number)
    second = np.flatnonzero(numbers == second_atomic_number)
    if first_atomic_number == second_atomic_number:
        left, right = np.triu_indices(len(first), k=1)
        first_indices = first[left]
        second_indices = first[right]
    else:
        first_indices = np.repeat(first, len(second))
        second_indices = np.tile(second, len(first))
    intermolecular = (
        molecule_ids[first_indices] != molecule_ids[second_indices]
    )
    return first_indices[intermolecular], second_indices[intermolecular]


def _rdf_histograms(
    positions: np.ndarray,
    cells: np.ndarray,
    *,
    numbers: np.ndarray,
    molecule_ids: np.ndarray,
    first_atomic_number: int,
    second_atomic_number: int,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first, second = _pair_indices(
        numbers,
        molecule_ids,
        first_atomic_number,
        second_atomic_number,
    )
    if len(first) == 0:
        raise BulkWaterValidationError("RDF pair selection is empty.")
    histograms = []
    normalizations = []
    shell_volumes = (4.0 * np.pi / 3.0) * (
        edges[1:] ** 3 - edges[:-1] ** 3
    )
    for frame_positions, frame_cell in zip(positions, cells):
        displacements = frame_positions[second] - frame_positions[first]
        _, distances = find_mic(displacements, frame_cell, pbc=True)
        histograms.append(np.histogram(distances, bins=edges)[0])
        volume = abs(float(np.linalg.det(frame_cell)))
        normalizations.append(len(first) * shell_volumes / volume)
    return (
        np.asarray(histograms, dtype=float),
        np.asarray(normalizations, dtype=float),
    )


def _blocked_rdf(
    histograms: np.ndarray,
    normalizations: np.ndarray,
    *,
    block_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    frame_count = len(histograms)
    actual_blocks = int(block_count)
    if actual_blocks < 2 or actual_blocks > frame_count:
        raise BulkWaterValidationError(
            "RDF uncertainty requires at least two non-empty blocks."
        )
    if frame_count % actual_blocks:
        raise BulkWaterValidationError(
            "RDF frames must divide evenly into contiguous uncertainty blocks."
        )
    total = np.sum(histograms, axis=0)
    expected = np.sum(normalizations, axis=0)
    rdf = np.divide(
        total,
        expected,
        out=np.zeros_like(total),
        where=expected > 0.0,
    )
    block_values = []
    for indices in np.split(np.arange(frame_count), actual_blocks):
        block_total = np.sum(histograms[indices], axis=0)
        block_expected = np.sum(normalizations[indices], axis=0)
        block_values.append(
            np.divide(
                block_total,
                block_expected,
                out=np.zeros_like(block_total),
                where=block_expected > 0.0,
            )
        )
    blocks = np.asarray(block_values, dtype=float)
    standard_error = np.std(blocks, axis=0, ddof=1) / math.sqrt(actual_blocks)
    return rdf, standard_error, actual_blocks


def _oxygen_rdf_features(
    radii: np.ndarray,
    rdf: np.ndarray,
    edges: np.ndarray,
    cells: np.ndarray,
    *,
    oxygen_count: int,
) -> dict[str, float | None]:
    peak_indices = np.flatnonzero((radii >= 2.4) & (radii <= 3.4))
    if not len(peak_indices):
        return {
            "first_peak_position_angstrom": None,
            "first_peak_height": None,
            "first_minimum_position_angstrom": None,
            "coordination_number_at_first_minimum": None,
        }
    peak_index = int(peak_indices[np.argmax(rdf[peak_indices])])
    minimum_indices = np.flatnonzero(
        (radii > radii[peak_index]) & (radii <= 4.2)
    )
    if not len(minimum_indices):
        minimum_index = len(radii) - 1
    else:
        minimum_index = int(
            minimum_indices[np.argmin(rdf[minimum_indices])]
        )
    shell_volumes = (4.0 * np.pi / 3.0) * (
        edges[1:] ** 3 - edges[:-1] ** 3
    )
    mean_inverse_volume = float(
        np.mean([1.0 / abs(np.linalg.det(cell)) for cell in cells])
    )
    neighbor_density = (oxygen_count - 1) * mean_inverse_volume
    coordination = neighbor_density * float(
        np.sum(rdf[: minimum_index + 1] * shell_volumes[: minimum_index + 1])
    )
    return {
        "first_peak_position_angstrom": float(radii[peak_index]),
        "first_peak_height": float(rdf[peak_index]),
        "first_minimum_position_angstrom": float(radii[minimum_index]),
        "coordination_number_at_first_minimum": coordination,
    }


def _observe(
    atoms: Atoms,
    calculator: Calculator,
    *,
    stage: str,
    step: int,
    timestep_fs: float,
) -> dict[str, float | int | str]:
    calculator.calculate(
        atoms,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    )
    try:
        forces = np.asarray(calculator.results["forces"], dtype=float)
        stress = np.asarray(
            atoms.get_stress(include_ideal_gas=True),
            dtype=float,
        )
        potential = float(calculator.results["energy"])
        kinetic = float(atoms.get_kinetic_energy())
        temperature = float(atoms.get_temperature())
    except (KeyError, TypeError, ValueError) as exc:
        raise BulkWaterValidationError(
            "OBSERVATION_INVALID: calculator results are incomplete or malformed."
        ) from exc
    if (
        forces.shape != (len(atoms), 3)
        or stress.shape != (6,)
        or not np.all(np.isfinite(forces))
        or not np.all(np.isfinite(stress))
        or not math.isfinite(potential)
        or not math.isfinite(kinetic)
        or not math.isfinite(temperature)
    ):
        raise BulkWaterValidationError(
            "OBSERVATION_INVALID: energy, forces, stress and temperature must "
            "have finite Route A shapes and units."
        )
    force_norms = np.linalg.norm(forces, axis=1)
    return {
        "stage": stage,
        "step": int(step),
        "time_fs": float(step * timestep_fs),
        "temperature_k": temperature,
        "potential_energy_ev": potential,
        "kinetic_energy_ev": kinetic,
        "total_energy_ev": potential + kinetic,
        "force_rms_ev_per_angstrom": float(np.sqrt(np.mean(forces**2))),
        "force_max_ev_per_angstrom": float(np.max(force_norms)),
        "pressure_bar": float(-np.mean(stress[:3]) / units.bar),
        "volume_angstrom3": float(atoms.get_volume()),
        "density_g_per_ml": water_density_g_per_ml(atoms),
    }


def _observation_arrays(
    observations: list[dict[str, float | int | str]],
) -> dict[str, np.ndarray]:
    return {
        "observation_stage": _immutable_array(
            [row["stage"] for row in observations],
            dtype="U16",
        ),
        "observation_step": _immutable_array(
            [row["step"] for row in observations],
            dtype=np.int64,
        ),
        **{
            f"observation_{name}": _immutable_array(
                [row[name] for row in observations],
                dtype=np.float64,
            )
            for name in (
                "time_fs",
                "temperature_k",
                "potential_energy_ev",
                "kinetic_energy_ev",
                "total_energy_ev",
                "force_rms_ev_per_angstrom",
                "force_max_ev_per_angstrom",
                "pressure_bar",
                "volume_angstrom3",
                "density_g_per_ml",
            )
        },
    }


def run_bulk_water_nvt(
    initial_atoms: Atoms,
    *,
    calculator: Calculator,
    config: BulkWaterNVTConfig,
    source_provenance: Mapping[str, Any],
    calculator_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
    progress_callback: (
        Callable[[Mapping[str, float | int | str]], None] | None
    ) = None,
) -> BulkWaterValidationResult:
    topology = validate_water_box(initial_atoms)
    source_record, calculator_record, implementation_record = (
        _validated_run_provenance(
            initial_atoms,
            calculator,
            source_provenance=source_provenance,
            calculator_provenance=calculator_provenance,
            implementation_provenance=implementation_provenance,
        )
    )
    if config.rdf_max_angstrom > topology["safe_rdf_radius_angstrom"] + 1.0e-12:
        raise BulkWaterValidationError(
            "RDF_RADIUS_INVALID: rdf_max_angstrom exceeds half the minimum "
            "periodic cell height."
        )

    atoms = initial_atoms.copy()
    atoms.calc = calculator
    initial_evidence = _structure_evidence(atoms)
    molecule_ids = _water_molecule_ids(atoms)
    rng = np.random.default_rng(config.seed)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=config.temperature_k,
        force_temp=True,
        rng=rng,
    )
    Stationary(atoms, preserve_temperature=True)

    initial_observation = _observe(
        atoms,
        calculator,
        stage="initial",
        step=0,
        timestep_fs=config.timestep_fs,
    )
    observations = [initial_observation]
    if progress_callback is not None:
        progress_callback(initial_observation)
    stability_monitor = _StepwiseStabilityMonitor(config)
    stability_monitor.inspect(atoms, stage="initial", step=0)
    production_positions = []
    production_cells = []
    production_velocities = []
    global_step = 0
    stages = (
        (
            "thermalization",
            config.thermalization_steps,
            config.thermalization_friction_per_fs,
        ),
        (
            "equilibration",
            config.equilibration_steps,
            config.equilibration_friction_per_fs,
        ),
        (
            "production",
            config.production_steps,
            config.production_friction_per_fs,
        ),
    )
    for stage, stage_steps, friction in stages:
        stage_start_step = global_step
        dynamics = Langevin(
            atoms,
            timestep=config.timestep_fs * units.fs,
            temperature_K=config.temperature_k,
            friction=friction / units.fs,
            fixcm=True,
            rng=rng,
        )
        dynamics.attach(
            lambda stage=stage, stage_start_step=stage_start_step, dynamics=dynamics: (
                stability_monitor.inspect(
                    atoms,
                    stage=stage,
                    step=stage_start_step + int(dynamics.nsteps),
                )
            ),
            interval=1,
        )
        remaining = int(stage_steps)
        while remaining:
            chunk = min(config.sample_interval_steps, remaining)
            dynamics.run(chunk)
            global_step += chunk
            remaining -= chunk
            observation = _observe(
                atoms,
                calculator,
                stage=stage,
                step=global_step,
                timestep_fs=config.timestep_fs,
            )
            observations.append(observation)
            if progress_callback is not None:
                progress_callback(observation)
            if stage == "production":
                production_positions.append(np.asarray(atoms.positions).copy())
                production_cells.append(np.asarray(atoms.cell.array).copy())
                production_velocities.append(np.asarray(atoms.get_velocities()).copy())

    positions = _immutable_array(production_positions, dtype=np.float64)
    cells = _immutable_array(production_cells, dtype=np.float64)
    velocities = _immutable_array(production_velocities, dtype=np.float64)
    numbers = _immutable_array(atoms.numbers, dtype=np.int64)
    edges = np.arange(
        0.0,
        config.rdf_max_angstrom,
        config.rdf_bin_width_angstrom,
        dtype=float,
    )
    edges = np.append(
        edges[edges < config.rdf_max_angstrom],
        config.rdf_max_angstrom,
    )
    radii = 0.5 * (edges[1:] + edges[:-1])

    rdf_arrays: dict[str, np.ndarray] = {}
    rdf_summary: dict[str, Any] = {}
    pair_specs = {
        "oo": (8, 8),
        "oh": (8, 1),
        "hh": (1, 1),
    }
    for label, (first_number, second_number) in pair_specs.items():
        histograms, normalizations = _rdf_histograms(
            positions,
            cells,
            numbers=numbers,
            molecule_ids=molecule_ids,
            first_atomic_number=first_number,
            second_atomic_number=second_number,
            edges=edges,
        )
        rdf, rdf_sem, actual_blocks = _blocked_rdf(
            histograms,
            normalizations,
            block_count=config.rdf_block_count,
        )
        rdf_arrays[f"rdf_{label}_g"] = _immutable_array(rdf, dtype=np.float64)
        rdf_arrays[f"rdf_{label}_sem"] = _immutable_array(
            rdf_sem,
            dtype=np.float64,
        )
        rdf_summary[label] = {
            "pair": [
                int(first_number),
                int(second_number),
            ],
            "intermolecular_only": True,
            "block_count": actual_blocks,
        }

    observations_arrays = _observation_arrays(observations)
    production_mask = observations_arrays["observation_stage"] == "production"
    production_temperature = observations_arrays["observation_temperature_k"][
        production_mask
    ]
    production_force_max = observations_arrays[
        "observation_force_max_ev_per_angstrom"
    ][production_mask]
    production_potential_per_water = (
        observations_arrays["observation_potential_energy_ev"][production_mask]
        / topology["water_count"]
    )
    production_time_ps = (
        observations_arrays["observation_time_fs"][production_mask] / 1_000.0
    )
    potential_slope = float(
        np.polyfit(
            production_time_ps - production_time_ps[0],
            production_potential_per_water,
            deg=1,
        )[0]
    )
    temperature_mean = float(np.mean(production_temperature))
    temperature_relative_error = abs(
        temperature_mean - config.temperature_k
    ) / config.temperature_k
    finite_observations = all(
        np.all(np.isfinite(values))
        for name, values in observations_arrays.items()
        if name != "observation_stage"
    )
    stepwise_stability = stability_monitor.as_dict(
        expected_last_step=global_step
    )
    engineering_checks = {
        "finite_observations": bool(finite_observations),
        "all_md_steps_checked": bool(
            stepwise_stability["all_steps_checked"]
        ),
        "temperature_below_emergency_limit": bool(
            stepwise_stability["maximum_temperature_k"]
            <= config.maximum_temperature_k
        ),
        "force_below_emergency_limit": bool(
            stepwise_stability["maximum_force_ev_per_angstrom"]
            <= config.maximum_force_ev_per_angstrom
        ),
        "water_topology_preserved": bool(
            stepwise_stability["topology_preserved"]
        ),
    }
    production_duration_ps = (
        config.production_steps * config.timestep_fs / 1_000.0
    )
    minimum_diagnostic_checks = {
        "engineering_stability_passed": all(engineering_checks.values()),
        "three_stage_protocol_completed": bool(
            global_step
            == (
                config.thermalization_steps
                + config.equilibration_steps
                + config.production_steps
            )
        ),
        "production_duration_sufficient": (
            production_duration_ps
            >= config.minimum_diagnostic_duration_ps
        ),
        "production_frame_count_sufficient": (
            len(positions) >= config.minimum_diagnostic_frames
        ),
        "production_temperature_centered": (
            temperature_relative_error
            <= config.temperature_relative_tolerance
        ),
    }
    minimum_diagnostic_eligible = all(minimum_diagnostic_checks.values())
    hamiltonian_freeze_checks = {
        "minimum_nvt_diagnostic_passed": minimum_diagnostic_eligible,
        "independent_replicas_passed": False,
        "npt_density_validation_passed": False,
        "finite_size_validation_passed": False,
        "external_reference_validation_passed": False,
    }

    arrays = {
        **observations_arrays,
        "atomic_numbers": numbers,
        "production_positions_angstrom": positions,
        "production_cells_angstrom": cells,
        "production_velocities_angstrom_per_ase_time": velocities,
        "rdf_r_angstrom": _immutable_array(radii, dtype=np.float64),
        **rdf_arrays,
    }
    semantic_array_hashes = {
        name: _array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    summary_preimage = {
        "schema": "maple-route-a-bulk-water-validation-summary-v2",
        "config": {
            "schema": "maple-route-a-bulk-water-nvt-config-v2",
            **asdict(config),
            "content_hash": config.content_hash,
        },
        "source": source_record,
        "calculator": calculator_record,
        "implementation": implementation_record,
        "initial_structure": initial_evidence,
        "final_structure": _structure_evidence(atoms),
        "topology": topology,
        "trajectory": {
            "production_duration_ps": production_duration_ps,
            "production_frame_count": len(positions),
            "semantic_array_sha256": semantic_array_hashes,
            "determinism_scope": (
                "Seeded stochastic path; bitwise identity across different "
                "GPU/runtime stacks is not claimed."
            ),
        },
        "diagnostics": {
            "stepwise_stability": stepwise_stability,
            "production_temperature_mean_k": temperature_mean,
            "production_temperature_relative_error": (
                temperature_relative_error
            ),
            "production_temperature_max_k": float(
                np.max(production_temperature)
            ),
            "production_force_max_ev_per_angstrom": float(
                np.max(production_force_max)
            ),
            "production_potential_slope_ev_per_water_ps": potential_slope,
            "production_pressure_mean_bar": float(
                np.mean(
                    observations_arrays["observation_pressure_bar"][
                        production_mask
                    ]
                )
            ),
            "production_density_mean_g_per_ml": float(
                np.mean(
                    observations_arrays["observation_density_g_per_ml"][
                        production_mask
                    ]
                )
            ),
            "rdf": {
                **rdf_summary,
                "oo_features": _oxygen_rdf_features(
                    radii,
                    rdf_arrays["rdf_oo_g"],
                    edges,
                    cells,
                    oxygen_count=topology["water_count"],
                ),
            },
        },
        "gates": {
            "engineering_checks": engineering_checks,
            "engineering_stability_passed": all(
                engineering_checks.values()
            ),
            "minimum_diagnostic_checks": minimum_diagnostic_checks,
            "minimum_diagnostic_eligible": minimum_diagnostic_eligible,
            "hamiltonian_freeze_checks": hamiltonian_freeze_checks,
            "hamiltonian_freeze_eligible": all(
                hamiltonian_freeze_checks.values()
            ),
            "claim_eligible": False,
            "interpretation": (
                "The in-run diagnostic is necessary but cannot freeze the "
                "Hamiltonian without independent replicas, NPT density, "
                "finite-size and external-reference evidence. No bulk-water "
                "result alone establishes Route A hydration-free-energy accuracy."
            ),
        },
    }
    result_hash = canonical_sha256(summary_preimage)
    return BulkWaterValidationResult(
        result_hash=result_hash,
        summary=summary_preimage,
        arrays=arrays,
    )
