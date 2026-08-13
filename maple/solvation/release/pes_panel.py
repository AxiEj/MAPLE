"""Preregistered multi-molecule same-scalar PES-panel contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from maple.solvation.coupling.state_equation import geometry_sha256

PES_PANEL_SCHEMA_VERSION = "route2-fixedbox590-pes-panel-geometry-asset-v1"
PES_PANEL_CONTRACT_VERSION = "route2-fixedbox590-pes-panel-contract-v1"
PES_CARTESIAN_PANEL_CONTRACT_VERSION = (
    "route2-fixedbox590-cartesian-panel-contract-v1"
)
PES_PANEL_ASSET_SHA256 = (
    "ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3"
)
PES_PANEL_MOLECULE_COUNT = 20
PES_PANEL_VARIANT_NAMES = ("reference", "bond-compressed", "bond-stretched")
PES_PANEL_TORSION_MOLECULE_ID = "trans-butane"
PES_PANEL_TORSION_PATH_NAME = "trans-butane-central-bond-torsion"
PES_PANEL_TORSION_ANGLES_DEG = (180.0, 150.0, 120.0, 90.0, 60.0)
PES_PANEL_TORSION_AXIS = (1, 2)
PES_PANEL_TORSION_ROTATED_ATOMS = (0, 4, 6, 7, 10, 11)
PES_PANEL_STRETCH_MOLECULE_ID = "hydrogen-peroxide"
PES_PANEL_STRETCH_PATH_NAME = "hydrogen-peroxide-oxygen-oxygen-stretch"
PES_PANEL_STRETCH_BOND = (0, 1)
PES_PANEL_STRETCH_CHANGES_A = (-0.16, -0.08, 0.0, 0.08, 0.16, 0.32)
PES_PANEL_ADDITIONAL_PATHS = (
    PES_PANEL_TORSION_PATH_NAME,
    PES_PANEL_STRETCH_PATH_NAME,
)
PES_PANEL_DIRECTION_NAMES = ("seeded-internal", "radial-internal", "bond-stretch")
PES_PANEL_DIRECTIONAL_STEPS_A = (4.0e-4, 2.0e-4, 1.0e-4)
PES_CARTESIAN_PANEL_STEPS_A = PES_PANEL_DIRECTIONAL_STEPS_A
PES_CARTESIAN_PANEL_VARIANT = "reference"
PES_PANEL_BOND_DISPLACEMENT_A = 0.08
PES_PANEL_RANDOM_SEED = 20260813

DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A = 5.0e-4
DIRECTIONAL_RELATIVE_TOLERANCE = 2.0e-3
DIRECTIONAL_RELATIVE_FLOOR_EV_PER_A = 1.0e-3
CARTESIAN_RMS_TOLERANCE_EV_PER_A = 5.0e-4
CARTESIAN_MAXIMUM_TOLERANCE_EV_PER_A = 2.0e-3
COLD_WARM_ENERGY_TOLERANCE_EV = 1.0e-8
COLD_WARM_SOURCE_RELATIVE_TOLERANCE = 1.0e-8
MAXIMUM_PRIMAL_RESIDUAL = 1.0e-12
MAXIMUM_ADJOINT_RESIDUAL = 1.0e-10
RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A = 5.0e-5

_PANEL_ASSET = (
    Path(__file__).parents[3]
    / "tools"
    / "route2_release"
    / "data"
    / "fixedbox590_pes_panel_v1.json"
)


@dataclass(frozen=True, slots=True)
class PESPanelMolecule:
    molecule_id: str
    source_record_id: str
    chemical_formula: str
    scope_tags: tuple[str, ...]
    atoms: Atoms


@dataclass(frozen=True, slots=True)
class PESPanelPathPoint:
    """One immutable geometry on a preregistered physical coordinate path."""

    path_name: str
    molecule_id: str
    point_label: str
    coordinate_name: str
    coordinate_value: float
    coordinate_unit: str
    scope_tags: tuple[str, ...]
    atoms: Atoms


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def load_pes_panel(path: Path = _PANEL_ASSET) -> tuple[PESPanelMolecule, ...]:
    """Load the exact, SHA-bound neutral-singlet geometry asset."""

    resolved = Path(path)
    if _sha256_file(resolved) != PES_PANEL_ASSET_SHA256:
        raise RuntimeError("PES-panel geometry asset does not match its frozen SHA256.")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PES_PANEL_SCHEMA_VERSION:
        raise ValueError("PES-panel geometry asset schema is invalid.")
    domain = payload.get("declared_checkpoint_domain")
    if domain != {
        "charge_support": [0, 0],
        "charged_cases_included": False,
        "reason": (
            "official MACE-POLAR-1-M vNext adapter declares neutral singlets only; "
            "unsupported charged systems are excluded rather than spoofed"
        ),
        "spin_multiplicities": [1],
    }:
        raise ValueError("PES-panel checkpoint-domain declaration changed.")
    records = payload.get("molecules")
    if not isinstance(records, list) or len(records) != PES_PANEL_MOLECULE_COUNT:
        raise ValueError("PES-panel asset must contain exactly 20 molecules.")
    result: list[PESPanelMolecule] = []
    for raw in records:
        if not isinstance(raw, dict):
            raise TypeError("Each PES-panel molecule must be a JSON object.")
        numbers = np.asarray(raw.get("atomic_numbers"))
        positions = np.asarray(raw.get("positions_A"), dtype=float)
        if (
            numbers.ndim != 1
            or numbers.size < 2
            or not np.issubdtype(numbers.dtype, np.integer)
            or np.any(numbers <= 0)
            or positions.shape != (numbers.size, 3)
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("PES-panel atomic numbers/positions are invalid.")
        if raw.get("charge") != 0 or raw.get("multiplicity") != 1:
            raise ValueError("This asset is restricted to neutral singlets.")
        tags = tuple(_text(item, "scope tag") for item in raw.get("scope_tags", ()))
        if not tags:
            raise ValueError("Every PES-panel molecule requires scope tags.")
        atoms = Atoms(
            numbers=numbers.astype(int),
            positions=positions,
            info={"charge": 0, "mult": 1},
        )
        result.append(
            PESPanelMolecule(
                _text(raw.get("molecule_id"), "molecule_id"),
                _text(raw.get("source_record_id"), "source_record_id"),
                _text(raw.get("chemical_formula"), "chemical_formula"),
                tags,
                atoms,
            )
        )
    identifiers = tuple(item.molecule_id for item in result)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("PES-panel molecule IDs must be unique.")
    return tuple(result)


def _normalized_internal(values: object, atom_count: int, name: str) -> np.ndarray:
    direction = np.asarray(values, dtype=float)
    if direction.shape != (atom_count, 3) or not np.all(np.isfinite(direction)):
        raise ValueError(f"{name} must be finite with shape ({atom_count}, 3).")
    direction = direction - np.mean(direction, axis=0, keepdims=True)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError(f"{name} is singular after removing translation.")
    return direction / norm


def select_bond(atoms: Atoms) -> tuple[int, int]:
    """Select one deterministic covalent bond without runtime connectivity state."""

    positions = np.asarray(atoms.positions, dtype=float)
    numbers = np.asarray(atoms.numbers, dtype=int)
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    threshold = 1.25 * (
        covalent_radii[numbers][:, None] + covalent_radii[numbers][None, :]
    )
    bonded = np.argwhere(np.triu(distances < threshold, 1))
    if not len(bonded):
        raise ValueError("PES-panel molecule has no deterministic covalent bond.")
    heavy = [
        tuple(int(v) for v in pair) for pair in bonded if np.all(numbers[pair] > 1)
    ]
    candidates = heavy or [tuple(int(v) for v in pair) for pair in bonded]
    return max(candidates, key=lambda pair: (distances[pair], -pair[0], -pair[1]))


def bond_direction(atoms: Atoms, bond: tuple[int, int] | None = None) -> np.ndarray:
    first, second = select_bond(atoms) if bond is None else bond
    delta = np.asarray(atoms.positions[second] - atoms.positions[first], dtype=float)
    unit = delta / np.linalg.norm(delta)
    result = np.zeros((len(atoms), 3), dtype=float)
    result[first] = -unit / 2.0
    result[second] = unit / 2.0
    return _normalized_internal(result, len(atoms), "bond-stretch direction")


def panel_geometries(molecule: PESPanelMolecule) -> dict[str, Atoms]:
    """Return the frozen reference plus symmetric bond compression/stretch."""

    direction = bond_direction(molecule.atoms)
    geometries: dict[str, Atoms] = {}
    for name, coefficient in (
        ("reference", 0.0),
        ("bond-compressed", -PES_PANEL_BOND_DISPLACEMENT_A),
        ("bond-stretched", PES_PANEL_BOND_DISPLACEMENT_A),
    ):
        atoms = molecule.atoms.copy()
        atoms.positions += coefficient * direction
        geometries[name] = atoms
    return geometries


def panel_directions(atoms: Atoms, molecule_id: str) -> dict[str, np.ndarray]:
    """Return three deterministic unit directions for one panel geometry."""

    digest = hashlib.sha256(
        f"{PES_PANEL_RANDOM_SEED}:{molecule_id}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    random = np.random.default_rng(seed).normal(size=(len(atoms), 3))
    centred = atoms.positions - np.mean(atoms.positions, axis=0, keepdims=True)
    bond = bond_direction(atoms)
    radial = _normalized_internal(centred, len(atoms), "radial-internal direction")
    seeded = _normalized_internal(random, len(atoms), "seeded-internal direction")
    return {
        "seeded-internal": seeded,
        "radial-internal": radial,
        "bond-stretch": bond,
    }


def _panel_molecule(molecule_id: str) -> PESPanelMolecule:
    matches = tuple(
        molecule for molecule in load_pes_panel() if molecule.molecule_id == molecule_id
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one frozen molecule {molecule_id!r}.")
    return matches[0]


def _rotate_about_axis(
    positions: np.ndarray,
    *,
    axis: tuple[int, int],
    rotated_atoms: tuple[int, ...],
    angle_degrees: float,
) -> np.ndarray:
    """Rotate one graph-side fragment with a direct Rodrigues transform."""

    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("positions must be a finite Cartesian array.")
    origin_index, direction_index = axis
    if (
        origin_index == direction_index
        or min(origin_index, direction_index, *rotated_atoms) < 0
        or max(origin_index, direction_index, *rotated_atoms) >= len(values)
        or origin_index in rotated_atoms
        or direction_index in rotated_atoms
        or len(set(rotated_atoms)) != len(rotated_atoms)
    ):
        raise ValueError("torsion axis/fragment indices are invalid.")
    axis_vector = values[direction_index] - values[origin_index]
    norm = float(np.linalg.norm(axis_vector))
    if norm <= 1.0e-12:
        raise ValueError("torsion axis is singular.")
    unit = axis_vector / norm
    ux, uy, uz = unit
    skew = np.asarray([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=float)
    angle = math.radians(float(angle_degrees))
    rotation = (
        np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    )
    result = np.array(values, copy=True)
    selected = np.asarray(rotated_atoms, dtype=int)
    origin = values[origin_index]
    result[selected] = origin + (values[selected] - origin) @ rotation.T
    return result


def torsion_path() -> tuple[PESPanelPathPoint, ...]:
    """Five-point trans-to-gauche butane central-bond torsion path.

    The frozen source geometry has C0-C1-C2-C3 dihedral 180 degrees.  The
    preregistered rotations are relative to that source, so the reported
    coordinate is the actual 180, 150, 120, 90, or 60 degree dihedral.  The
    60-degree endpoint is a close-contact gauche conformer; it is not a TS.
    """

    molecule = _panel_molecule(PES_PANEL_TORSION_MOLECULE_ID)
    points: list[PESPanelPathPoint] = []
    for angle in PES_PANEL_TORSION_ANGLES_DEG:
        atoms = molecule.atoms.copy()
        atoms.positions[:] = _rotate_about_axis(
            molecule.atoms.positions,
            axis=PES_PANEL_TORSION_AXIS,
            rotated_atoms=PES_PANEL_TORSION_ROTATED_ATOMS,
            angle_degrees=180.0 - angle,
        )
        actual = float(atoms.get_dihedral(0, 1, 2, 3))
        if actual == 360.0:
            actual = 0.0
        if abs(actual - angle) > 2.0e-12:
            raise RuntimeError("Frozen butane torsion construction changed.")
        points.append(
            PESPanelPathPoint(
                PES_PANEL_TORSION_PATH_NAME,
                molecule.molecule_id,
                f"dihedral-{int(angle):03d}-deg",
                "C0-C1-C2-C3-dihedral",
                angle,
                "degree",
                ("flexible torsion", "close contact", "non-equilibrium path"),
                atoms,
            )
        )
    return tuple(points)


def torsion_tangent(atoms: Atoms) -> np.ndarray:
    """Return ``dR/dtheta`` in Angstrom/radian for the frozen butane path."""

    if len(atoms) != 14 or atoms.get_chemical_formula() != "C4H10":
        raise ValueError("torsion_tangent is defined only for the frozen butane path.")
    positions = np.asarray(atoms.positions, dtype=float)
    origin_index, direction_index = PES_PANEL_TORSION_AXIS
    axis = positions[direction_index] - positions[origin_index]
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise ValueError("torsion axis is singular.")
    unit = axis / norm
    result = np.zeros_like(positions)
    origin = positions[origin_index]
    selected = np.asarray(PES_PANEL_TORSION_ROTATED_ATOMS, dtype=int)
    result[selected] = np.cross(unit, positions[selected] - origin)
    if not np.all(np.isfinite(result)) or float(np.linalg.norm(result)) <= 1.0e-12:
        raise RuntimeError("frozen butane torsion tangent is invalid.")
    return result


def stretch_path() -> tuple[PESPanelPathPoint, ...]:
    """Six-point H2O2 O-O stretch reaction-coordinate surrogate.

    This is a deliberately nonstationary bond-dissociation coordinate used to
    exercise compressed, stretched, and TS-like *geometries*.  It is not an
    optimized transition state and must never be reported as one.
    """

    molecule = _panel_molecule(PES_PANEL_STRETCH_MOLECULE_ID)
    reference_direction = bond_direction(molecule.atoms, PES_PANEL_STRETCH_BOND)
    reference_length = float(
        np.linalg.norm(
            molecule.atoms.positions[PES_PANEL_STRETCH_BOND[1]]
            - molecule.atoms.positions[PES_PANEL_STRETCH_BOND[0]]
        )
    )
    points: list[PESPanelPathPoint] = []
    for change in PES_PANEL_STRETCH_CHANGES_A:
        atoms = molecule.atoms.copy()
        # bond_direction is normalized in the full Cartesian metric, so its
        # two active atoms change their separation by sqrt(2) times coefficient.
        coefficient = float(change) / math.sqrt(2.0)
        atoms.positions += coefficient * reference_direction
        actual_length = float(
            np.linalg.norm(
                atoms.positions[PES_PANEL_STRETCH_BOND[1]]
                - atoms.positions[PES_PANEL_STRETCH_BOND[0]]
            )
        )
        if abs(actual_length - (reference_length + change)) > 2.0e-12:
            raise RuntimeError("Frozen peroxide stretch construction changed.")
        points.append(
            PESPanelPathPoint(
                PES_PANEL_STRETCH_PATH_NAME,
                molecule.molecule_id,
                f"oo-change-{change:+.2f}-A",
                "O0-O1-bond-length",
                actual_length,
                "angstrom",
                (
                    "reaction-coordinate surrogate",
                    "stretched bond",
                    (
                        "TS-like geometry"
                        if change == max(PES_PANEL_STRETCH_CHANGES_A)
                        else "non-equilibrium path"
                    ),
                ),
                atoms,
            )
        )
    return tuple(points)


def stretch_tangent(atoms: Atoms) -> np.ndarray:
    """Return ``dR/d(r_OO)`` for the frozen peroxide stretch in A/A."""

    if len(atoms) != 4 or atoms.get_chemical_formula() != "H2O2":
        raise ValueError("stretch_tangent is defined only for the frozen H2O2 path.")
    return bond_direction(atoms, PES_PANEL_STRETCH_BOND) / math.sqrt(2.0)


def panel_paths() -> dict[str, tuple[PESPanelPathPoint, ...]]:
    """Return all preregistered path geometries in stable contract order."""

    return {
        PES_PANEL_TORSION_PATH_NAME: torsion_path(),
        PES_PANEL_STRETCH_PATH_NAME: stretch_path(),
    }


def summarize_pes_panel(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Aggregate raw runner records without changing preregistered thresholds."""

    values = tuple(records)
    expected = PES_PANEL_MOLECULE_COUNT * len(PES_PANEL_VARIANT_NAMES)
    if len(values) != expected:
        raise ValueError(f"PES panel must contain exactly {expected} geometry records.")
    molecules = {str(item.get("molecule_id")) for item in values}
    if len(molecules) != PES_PANEL_MOLECULE_COUNT:
        raise ValueError("PES panel must contain exactly 20 molecule IDs.")
    pairs = {
        (str(item.get("molecule_id")), str(item.get("variant"))) for item in values
    }
    expected_pairs = {
        (molecule.molecule_id, variant)
        for molecule in load_pes_panel()
        for variant in PES_PANEL_VARIANT_NAMES
    }
    if pairs != expected_pairs:
        raise ValueError("PES panel molecule/variant coverage is incomplete.")

    direction_records: list[Mapping[str, Any]] = []
    cold_warm_records: list[Mapping[str, Any]] = []
    topology_hashes_by_molecule: dict[str, set[str]] = {
        molecule.molecule_id: set() for molecule in load_pes_panel()
    }
    maximum_primal = 0.0
    maximum_adjoint = 0.0
    for item in values:
        directional = item.get("directional_force_fd")
        if (
            not isinstance(directional, Mapping)
            or set(directional) != set(PES_PANEL_DIRECTION_NAMES)
            or len(directional) != len(PES_PANEL_DIRECTION_NAMES)
        ):
            raise ValueError(
                "Every geometry must contain all preregistered directions."
            )
        for name in PES_PANEL_DIRECTION_NAMES:
            record = directional[name]
            if not isinstance(record, Mapping):
                raise ValueError("Every directional-force record must be a mapping.")
            direction_records.append(record)
        cold_warm = item.get("cold_warm")
        if not isinstance(cold_warm, Mapping):
            raise ValueError("Every geometry requires a cold/warm record.")
        cold_warm_records.append(cold_warm)
        molecule_id = str(item.get("molecule_id"))
        topology_hashes_by_molecule[molecule_id].add(
            _text(item.get("topology_hash"), "topology_hash")
        )
        maximum_primal = max(maximum_primal, float(item.get("maximum_primal_residual")))
        maximum_adjoint = max(maximum_adjoint, float(item.get("adjoint_residual")))

    all_directional = all(
        bool(record.get("all_gates_passed"))
        and bool(record.get("fixed_topology"))
        and float(record.get("maximum_primal_residual")) <= MAXIMUM_PRIMAL_RESIDUAL
        for record in direction_records
    )
    all_roots = all(
        bool(record.get("numerically_equivalent"))
        and all(bool(value) for value in record.get("gates", {}).values())
        for record in cold_warm_records
    )
    gates = {
        "all_directional_force_fd": all_directional,
        "all_cold_warm_roots": all_roots,
        "all_primal_residuals_le_1e-12": maximum_primal <= MAXIMUM_PRIMAL_RESIDUAL,
        "all_adjoint_residuals_le_1e-10": maximum_adjoint <= MAXIMUM_ADJOINT_RESIDUAL,
        "all_molecule_topologies_fixed": all(
            len(hashes) == 1 for hashes in topology_hashes_by_molecule.values()
        ),
    }
    return {
        "schema_version": "route2-fixedbox590-pes-panel-summary-v1",
        "molecule_count": PES_PANEL_MOLECULE_COUNT,
        "geometry_count": len(values),
        "directional_record_count": len(direction_records),
        "directional_sample_count": len(direction_records)
        * len(PES_PANEL_DIRECTIONAL_STEPS_A),
        "maximum_primal_residual": maximum_primal,
        "maximum_adjoint_residual": maximum_adjoint,
        "topology_hashes_by_molecule": {
            molecule_id: sorted(hashes)
            for molecule_id, hashes in sorted(topology_hashes_by_molecule.items())
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def summarize_pes_paths(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Validate every frozen path point and its local same-scalar force gate."""

    values = tuple(records)
    expected_points = tuple(point for path in panel_paths().values() for point in path)
    if len(values) != len(expected_points):
        raise ValueError(
            f"PES paths must contain exactly {len(expected_points)} records."
        )
    by_key = {
        (str(item.get("path_name")), str(item.get("point_label"))): item
        for item in values
    }
    expected_by_key = {
        (point.path_name, point.point_label): point for point in expected_points
    }
    if len(by_key) != len(values) or set(by_key) != set(expected_by_key):
        raise ValueError("PES path point coverage is incomplete or duplicated.")

    maximum_primal = 0.0
    maximum_adjoint = 0.0
    topology_hashes: dict[str, set[str]] = {
        name: set() for name in PES_PANEL_ADDITIONAL_PATHS
    }
    all_local_force = True
    all_roots = True
    for key, expected in expected_by_key.items():
        item = by_key[key]
        if (
            str(item.get("molecule_id")) != expected.molecule_id
            or str(item.get("coordinate_name")) != expected.coordinate_name
            or str(item.get("coordinate_unit")) != expected.coordinate_unit
            or float(item.get("coordinate_value")) != expected.coordinate_value
            or str(item.get("geometry_sha256")) != geometry_sha256(expected.atoms)
        ):
            raise ValueError(f"PES path record {key!r} changed its frozen geometry.")
        if not math.isfinite(float(item.get("energy_eV"))):
            raise ValueError("PES path energy must be finite.")
        local_force = item.get("local_tangent_force_fd")
        if not isinstance(local_force, Mapping):
            raise ValueError("Every path point requires a local tangent-force gate.")
        all_local_force &= bool(local_force.get("all_gates_passed")) and bool(
            local_force.get("fixed_topology")
        )
        cold_warm = item.get("cold_warm")
        if not isinstance(cold_warm, Mapping):
            raise ValueError("Every path point requires a cold/warm root record.")
        all_roots &= bool(cold_warm.get("numerically_equivalent")) and all(
            bool(value) for value in cold_warm.get("gates", {}).values()
        )
        maximum_primal = max(
            maximum_primal,
            float(item.get("maximum_primal_residual")),
            float(local_force.get("maximum_primal_residual")),
        )
        maximum_adjoint = max(maximum_adjoint, float(item.get("adjoint_residual")))
        topology_hashes[key[0]].add(_text(item.get("topology_hash"), "topology_hash"))

    gates = {
        "all_local_tangent_force_fd": all_local_force,
        "all_cold_warm_roots": all_roots,
        "all_primal_residuals_le_1e-12": maximum_primal <= MAXIMUM_PRIMAL_RESIDUAL,
        "all_adjoint_residuals_le_1e-10": maximum_adjoint <= MAXIMUM_ADJOINT_RESIDUAL,
        "all_path_topologies_fixed": all(
            len(hashes) == 1 for hashes in topology_hashes.values()
        ),
    }
    return {
        "schema_version": "route2-fixedbox590-pes-path-summary-v1",
        "path_count": len(topology_hashes),
        "path_geometry_count": len(values),
        "maximum_primal_residual": maximum_primal,
        "maximum_adjoint_residual": maximum_adjoint,
        "topology_hashes_by_path": {
            name: sorted(hashes) for name, hashes in topology_hashes.items()
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def summarize_cartesian_pes_panel(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Require one full-Cartesian reference-geometry record per molecule."""

    values = tuple(records)
    panel = load_pes_panel()
    if len(values) != PES_PANEL_MOLECULE_COUNT:
        raise ValueError(
            f"Cartesian PES panel must contain exactly {PES_PANEL_MOLECULE_COUNT} "
            "molecule records."
        )
    by_id = {str(item.get("molecule_id")): item for item in values}
    expected_ids = {molecule.molecule_id for molecule in panel}
    if len(by_id) != len(values) or set(by_id) != expected_ids:
        raise ValueError("Cartesian PES-panel molecule coverage is incomplete or duplicated.")

    maximum_primal = 0.0
    maximum_adjoint = 0.0
    all_cartesian = True
    all_roots = True
    all_topologies = True
    component_count = 0
    step_record_count = 0
    maximum_rms_error = 0.0
    maximum_component_error = 0.0
    low_error_plateau_count = 0
    topology_hashes_by_molecule: dict[str, list[str]] = {}
    for molecule in panel:
        item = by_id[molecule.molecule_id]
        if str(item.get("variant")) != PES_CARTESIAN_PANEL_VARIANT:
            raise ValueError("Cartesian PES panel is frozen to reference geometries.")
        expected_components = 3 * len(molecule.atoms)
        if int(item.get("component_count")) != expected_components:
            raise ValueError("Cartesian component count does not match the molecule.")
        component_count += expected_components
        cartesian = item.get("cartesian_force_fd")
        if not isinstance(cartesian, Mapping):
            raise ValueError("Every molecule requires a Cartesian force-FD record.")
        step_records = cartesian.get("records")
        convergence = cartesian.get("convergence")
        if (
            not isinstance(step_records, Sequence)
            or isinstance(step_records, (str, bytes))
            or len(step_records) != len(PES_CARTESIAN_PANEL_STEPS_A)
            or not isinstance(convergence, Mapping)
        ):
            raise ValueError("Cartesian force-FD records are incomplete.")
        if tuple(float(record.get("step_A")) for record in step_records) != (
            PES_CARTESIAN_PANEL_STEPS_A
        ):
            raise ValueError("Cartesian force-FD steps changed from the contract.")
        step_record_count += len(step_records)
        maximum_rms_error = max(
            maximum_rms_error,
            *(float(record.get("rms_error_eV_per_A")) for record in step_records),
        )
        maximum_component_error = max(
            maximum_component_error,
            *(
                float(record.get("maximum_error_eV_per_A"))
                for record in step_records
            ),
        )
        low_error_plateau_count += bool(convergence.get("low_error_plateau"))
        all_cartesian &= bool(cartesian.get("all_gates_passed"))
        all_cartesian &= bool(cartesian.get("fixed_topology"))
        root = item.get("cold_warm")
        if not isinstance(root, Mapping):
            raise ValueError("Every Cartesian molecule requires a cold/warm record.")
        all_roots &= bool(root.get("numerically_equivalent")) and all(
            bool(value) for value in root.get("gates", {}).values()
        )
        hashes = tuple(str(value) for value in cartesian.get("topology_hashes", ()))
        if len(hashes) != 1:
            all_topologies = False
        topology_hashes_by_molecule[molecule.molecule_id] = sorted(hashes)
        maximum_primal = max(
            maximum_primal,
            float(item.get("maximum_primal_residual")),
            float(cartesian.get("maximum_displaced_primal_residual")),
        )
        maximum_adjoint = max(maximum_adjoint, float(item.get("adjoint_residual")))

    gates = {
        "all_cartesian_force_fd": all_cartesian,
        "all_cold_warm_roots": all_roots,
        "all_primal_residuals_le_1e-12": maximum_primal <= MAXIMUM_PRIMAL_RESIDUAL,
        "all_adjoint_residuals_le_1e-10": maximum_adjoint <= MAXIMUM_ADJOINT_RESIDUAL,
        "all_molecule_topologies_fixed": all_topologies,
    }
    return {
        "schema_version": "route2-fixedbox590-cartesian-panel-summary-v1",
        "molecule_count": PES_PANEL_MOLECULE_COUNT,
        "geometry_count": len(values),
        "component_count": component_count,
        "component_sample_count": component_count * len(PES_CARTESIAN_PANEL_STEPS_A),
        "step_record_count": step_record_count,
        "maximum_rms_error_eV_per_A": maximum_rms_error,
        "maximum_component_error_eV_per_A": maximum_component_error,
        "low_error_plateau_count": low_error_plateau_count,
        "maximum_primal_residual": maximum_primal,
        "maximum_adjoint_residual": maximum_adjoint,
        "topology_hashes_by_molecule": topology_hashes_by_molecule,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


__all__ = [
    "CARTESIAN_MAXIMUM_TOLERANCE_EV_PER_A",
    "CARTESIAN_RMS_TOLERANCE_EV_PER_A",
    "COLD_WARM_ENERGY_TOLERANCE_EV",
    "COLD_WARM_SOURCE_RELATIVE_TOLERANCE",
    "DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A",
    "DIRECTIONAL_RELATIVE_FLOOR_EV_PER_A",
    "DIRECTIONAL_RELATIVE_TOLERANCE",
    "MAXIMUM_ADJOINT_RESIDUAL",
    "MAXIMUM_PRIMAL_RESIDUAL",
    "PES_PANEL_ASSET_SHA256",
    "PES_CARTESIAN_PANEL_CONTRACT_VERSION",
    "PES_CARTESIAN_PANEL_STEPS_A",
    "PES_CARTESIAN_PANEL_VARIANT",
    "PES_PANEL_ADDITIONAL_PATHS",
    "PES_PANEL_BOND_DISPLACEMENT_A",
    "PES_PANEL_CONTRACT_VERSION",
    "PES_PANEL_DIRECTION_NAMES",
    "PES_PANEL_DIRECTIONAL_STEPS_A",
    "PES_PANEL_MOLECULE_COUNT",
    "PES_PANEL_RANDOM_SEED",
    "PES_PANEL_SCHEMA_VERSION",
    "PES_PANEL_STRETCH_BOND",
    "PES_PANEL_STRETCH_CHANGES_A",
    "PES_PANEL_STRETCH_MOLECULE_ID",
    "PES_PANEL_STRETCH_PATH_NAME",
    "PES_PANEL_TORSION_ANGLES_DEG",
    "PES_PANEL_TORSION_AXIS",
    "PES_PANEL_TORSION_MOLECULE_ID",
    "PES_PANEL_TORSION_PATH_NAME",
    "PES_PANEL_TORSION_ROTATED_ATOMS",
    "PES_PANEL_VARIANT_NAMES",
    "PESPanelMolecule",
    "PESPanelPathPoint",
    "RESIDUAL_FORCE_ERROR_BUDGET_EV_PER_A",
    "bond_direction",
    "load_pes_panel",
    "panel_directions",
    "panel_geometries",
    "panel_paths",
    "select_bond",
    "stretch_path",
    "stretch_tangent",
    "summarize_pes_panel",
    "summarize_pes_paths",
    "summarize_cartesian_pes_panel",
    "torsion_path",
    "torsion_tangent",
]
