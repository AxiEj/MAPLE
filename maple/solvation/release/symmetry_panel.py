"""Preregistered rigid-symmetry and closed-loop panel contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np

from .pes_panel import PES_PANEL_MOLECULE_COUNT, load_pes_panel

SYMMETRY_PANEL_CONTRACT_VERSION = "route2-fixedbox590-symmetry-panel-contract-v1"
SYMMETRY_PANEL_SCHEMA_VERSION = "route2-fixedbox590-symmetry-panel-summary-v1"
SYMMETRY_PANEL_RANDOM_SEED = 20260813
SYMMETRY_PANEL_ROTATION_COUNT = 3
SYMMETRY_PANEL_TRANSLATION_A = (1.7, -0.8, 0.5)
SYMMETRY_PANEL_LOOP_SUBDIVISIONS = 4
SYMMETRY_PANEL_LOOP_AMPLITUDES_A = (0.02, 0.02)

RIGID_ENERGY_TOLERANCE_EV = 1.0e-6
NET_FORCE_TOLERANCE_EV_PER_A = 1.0e-5
ROTATION_FORCE_RELATIVE_TOLERANCE = 1.0e-4
TORQUE_TOLERANCE_EV = 1.0e-4
SOURCE_COVARIANCE_RELATIVE_TOLERANCE = 1.0e-4
PRIMAL_RESIDUAL_TOLERANCE = 1.0e-12
ADJOINT_RESIDUAL_TOLERANCE = 1.0e-10


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    orthogonal, triangular = np.linalg.qr(matrix)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    orthogonal = orthogonal @ np.diag(signs)
    if np.linalg.det(orthogonal) < 0.0:
        orthogonal[:, 0] *= -1.0
    return orthogonal


def symmetry_panel_rotations(molecule_id: str) -> tuple[np.ndarray, ...]:
    """Return three stable proper rotations unique to one molecule ID."""

    name = str(molecule_id).strip()
    if not name:
        raise ValueError("molecule_id must be non-empty.")
    digest = hashlib.sha256(
        f"{SYMMETRY_PANEL_RANDOM_SEED}:{name}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return tuple(_rotation(seed + index) for index in range(SYMMETRY_PANEL_ROTATION_COUNT))


def symmetry_panel_permutation(atomic_numbers: object) -> np.ndarray:
    """Swap the first deterministic pair of identical atoms."""

    numbers = np.asarray(atomic_numbers)
    if (
        numbers.ndim != 1
        or numbers.size < 2
        or not np.issubdtype(numbers.dtype, np.integer)
        or np.any(numbers <= 0)
    ):
        raise ValueError("atomic_numbers must be a positive integer vector.")
    for first in range(len(numbers)):
        matches = np.flatnonzero(numbers[first + 1 :] == numbers[first])
        if matches.size:
            second = first + 1 + int(matches[0])
            permutation = np.arange(len(numbers))
            permutation[first], permutation[second] = second, first
            return permutation
    raise ValueError("Symmetry panel requires at least one identical-atom pair.")


def rotate_radial_gto_blocks(values: object, rotation: object) -> np.ndarray:
    """Rotate the two raw real-spherical l=1 blocks in the 8-channel source."""

    result = np.asarray(values, dtype=float)
    transform = np.asarray(rotation, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[1] != 8
        or not np.all(np.isfinite(result))
        or transform.shape != (3, 3)
        or not np.all(np.isfinite(transform))
        or not np.allclose(transform @ transform.T, np.eye(3), atol=2e-14, rtol=0.0)
        or not np.isclose(np.linalg.det(transform), 1.0, atol=2e-14, rtol=0.0)
    ):
        raise ValueError("values/rotation do not satisfy the radial-GTO contract.")
    result = np.array(result, copy=True)
    for raw_indices in ((2, 3, 4), (5, 6, 7)):
        cartesian = result[:, raw_indices][:, (2, 0, 1)]
        result[:, raw_indices] = (cartesian @ transform.T)[:, (1, 2, 0)]
    return result


def summarize_rigid_symmetry(
    *,
    positions_A: object,
    base_energy_eV: float,
    base_forces_eV_per_A: object,
    base_source: object,
    base_topology_hash: str,
    translation_record: Mapping[str, Any],
    permutation_record: Mapping[str, Any],
    rotation_records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Recompute all rigid translation/rotation/force/torque gates."""

    positions = np.asarray(positions_A, dtype=float)
    forces = np.asarray(base_forces_eV_per_A, dtype=float)
    source = np.asarray(base_source, dtype=float)
    energy = float(base_energy_eV)
    if (
        positions.ndim != 2
        or positions.shape[0] < 1
        or positions.shape[1] != 3
        or forces.shape != positions.shape
        or source.shape != (len(positions), 8)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(forces))
        or not np.all(np.isfinite(source))
        or not math.isfinite(energy)
        or not isinstance(base_topology_hash, str)
        or len(base_topology_hash) != 64
    ):
        raise ValueError("Base rigid-symmetry arrays/identity are invalid.")
    rotations = tuple(rotation_records)
    if len(rotations) != SYMMETRY_PANEL_ROTATION_COUNT:
        raise ValueError("Rigid symmetry requires exactly three rotations.")

    centred = positions - np.mean(positions, axis=0, keepdims=True)
    net_force = float(np.linalg.norm(np.sum(forces, axis=0)))
    torque = float(np.linalg.norm(np.sum(np.cross(centred, forces), axis=0)))
    translation = np.asarray(translation_record.get("translation_A"), dtype=float)
    translated_forces = np.asarray(
        translation_record.get("forces_eV_per_A"), dtype=float
    )
    translated_energy = float(translation_record.get("energy_eV"))
    translated_source = np.asarray(translation_record.get("source"), dtype=float)
    if (
        translation.shape != (3,)
        or not np.array_equal(translation, np.asarray(SYMMETRY_PANEL_TRANSLATION_A))
        or translated_forces.shape != forces.shape
        or translated_source.shape != source.shape
        or not np.all(np.isfinite(translated_forces))
        or not np.all(np.isfinite(translated_source))
        or not math.isfinite(translated_energy)
    ):
        raise ValueError("Translation record changed from the frozen contract.")
    translation_energy_error = abs(translated_energy - energy)
    translation_force_error = float(np.linalg.norm(translated_forces - forces))
    translation_source_error = float(np.linalg.norm(translated_source - source))
    translation_net_force = float(np.linalg.norm(np.sum(translated_forces, axis=0)))
    translation_primal = float(translation_record.get("primal_residual"))
    translation_adjoint = float(translation_record.get("adjoint_residual"))
    translation_topology = str(translation_record.get("topology_hash"))

    permutation = np.asarray(permutation_record.get("permutation"))
    permuted_forces = np.asarray(
        permutation_record.get("forces_eV_per_A"), dtype=float
    )
    permuted_source = np.asarray(permutation_record.get("source"), dtype=float)
    permuted_energy = float(permutation_record.get("energy_eV"))
    if (
        permutation.shape != (len(positions),)
        or not np.issubdtype(permutation.dtype, np.integer)
        or not np.array_equal(np.sort(permutation), np.arange(len(positions)))
        or permuted_forces.shape != forces.shape
        or permuted_source.shape != source.shape
        or not np.all(np.isfinite(permuted_forces))
        or not np.all(np.isfinite(permuted_source))
        or not math.isfinite(permuted_energy)
    ):
        raise ValueError("Permutation record is invalid.")
    permutation = permutation.astype(int)
    permutation_energy_error = abs(permuted_energy - energy)
    permutation_force_relative = float(
        np.linalg.norm(permuted_forces - forces[permutation])
        / max(float(np.linalg.norm(forces)), 1.0)
    )
    permutation_source_relative = float(
        np.linalg.norm(permuted_source - source[permutation])
        / max(float(np.linalg.norm(source)), 1.0)
    )
    permutation_primal = float(permutation_record.get("primal_residual"))
    permutation_adjoint = float(permutation_record.get("adjoint_residual"))
    permutation_topology = str(permutation_record.get("topology_hash"))

    recomputed_rotations: list[dict[str, object]] = []
    maximum_energy = 0.0
    maximum_force_relative = 0.0
    maximum_source_relative = 0.0
    maximum_primal = max(translation_primal, permutation_primal)
    maximum_adjoint = max(translation_adjoint, permutation_adjoint)
    topology_hashes = {
        base_topology_hash,
        translation_topology,
        permutation_topology,
    }
    for index, record in enumerate(rotations):
        rotation = np.asarray(record.get("rotation_matrix"), dtype=float)
        expected = np.asarray(record.get("expected_rotation_matrix"), dtype=float)
        rotated_forces = np.asarray(record.get("forces_eV_per_A"), dtype=float)
        rotated_source = np.asarray(record.get("source"), dtype=float)
        rotated_energy = float(record.get("energy_eV"))
        primal = float(record.get("primal_residual"))
        adjoint = float(record.get("adjoint_residual"))
        topology = str(record.get("topology_hash"))
        if (
            int(record.get("index")) != index
            or rotation.shape != (3, 3)
            or expected.shape != (3, 3)
            or not np.array_equal(rotation, expected)
            or rotated_forces.shape != forces.shape
            or rotated_source.shape != source.shape
            or not np.all(np.isfinite(rotated_forces))
            or not np.all(np.isfinite(rotated_source))
            or not all(math.isfinite(v) and v >= 0.0 for v in (primal, adjoint))
            or not math.isfinite(rotated_energy)
        ):
            raise ValueError("Rotation record is invalid or out of contract order.")
        energy_error = abs(rotated_energy - energy)
        force_relative = float(
            np.linalg.norm(rotated_forces - forces @ rotation.T)
            / max(float(np.linalg.norm(forces)), 1.0)
        )
        source_relative = float(
            np.linalg.norm(rotated_source - rotate_radial_gto_blocks(source, rotation))
            / max(float(np.linalg.norm(source)), 1.0)
        )
        maximum_energy = max(maximum_energy, energy_error)
        maximum_force_relative = max(maximum_force_relative, force_relative)
        maximum_source_relative = max(maximum_source_relative, source_relative)
        maximum_primal = max(maximum_primal, primal)
        maximum_adjoint = max(maximum_adjoint, adjoint)
        topology_hashes.add(topology)
        recomputed_rotations.append(
            {
                "index": index,
                "rotation_matrix": rotation.tolist(),
                "energy_abs_eV": energy_error,
                "force_covariance_relative": force_relative,
                "source_covariance_relative": source_relative,
                "primal_residual": primal,
                "adjoint_residual": adjoint,
                "topology_hash": topology,
            }
        )

    gates = {
        "translation_energy_le_1e-6_eV": (
            translation_energy_error <= RIGID_ENERGY_TOLERANCE_EV
        ),
        "translation_force_difference_le_1e-5_eV_per_A": (
            translation_force_error <= NET_FORCE_TOLERANCE_EV_PER_A
        ),
        "translation_source_difference_le_1e-8": translation_source_error <= 1.0e-8,
        "base_and_translated_net_force_le_1e-5_eV_per_A": (
            max(net_force, translation_net_force) <= NET_FORCE_TOLERANCE_EV_PER_A
        ),
        "base_torque_le_1e-4_eV": torque <= TORQUE_TOLERANCE_EV,
        "permutation_energy_le_1e-6_eV": (
            permutation_energy_error <= RIGID_ENERGY_TOLERANCE_EV
        ),
        "permutation_force_covariance_relative_le_1e-4": (
            permutation_force_relative <= ROTATION_FORCE_RELATIVE_TOLERANCE
        ),
        "permutation_source_covariance_relative_le_1e-4": (
            permutation_source_relative <= SOURCE_COVARIANCE_RELATIVE_TOLERANCE
        ),
        "all_rotation_energies_le_1e-6_eV": maximum_energy <= RIGID_ENERGY_TOLERANCE_EV,
        "all_rotation_force_covariance_relative_le_1e-4": (
            maximum_force_relative <= ROTATION_FORCE_RELATIVE_TOLERANCE
        ),
        "all_rotation_source_covariance_relative_le_1e-4": (
            maximum_source_relative <= SOURCE_COVARIANCE_RELATIVE_TOLERANCE
        ),
        "all_primal_residuals_le_1e-12": maximum_primal <= PRIMAL_RESIDUAL_TOLERANCE,
        "all_adjoint_residuals_le_1e-10": maximum_adjoint <= ADJOINT_RESIDUAL_TOLERANCE,
        "fixed_topology": len(topology_hashes) == 1,
    }
    return {
        "schema_version": "route2-fixedbox590-rigid-symmetry-summary-v1",
        "translation": {
            "translation_A": translation.tolist(),
            "energy_abs_eV": translation_energy_error,
            "force_difference_norm_eV_per_A": translation_force_error,
            "source_difference_norm": translation_source_error,
            "net_force_norm_eV_per_A": translation_net_force,
            "primal_residual": translation_primal,
            "adjoint_residual": translation_adjoint,
            "topology_hash": translation_topology,
        },
        "permutation": {
            "permutation": permutation.tolist(),
            "energy_abs_eV": permutation_energy_error,
            "force_covariance_relative": permutation_force_relative,
            "source_covariance_relative": permutation_source_relative,
            "primal_residual": permutation_primal,
            "adjoint_residual": permutation_adjoint,
            "topology_hash": permutation_topology,
        },
        "rotations": recomputed_rotations,
        "base_net_force_norm_eV_per_A": net_force,
        "base_torque_norm_eV": torque,
        "maximum_rotation_energy_abs_eV": maximum_energy,
        "maximum_rotation_force_covariance_relative": maximum_force_relative,
        "maximum_rotation_source_covariance_relative": maximum_source_relative,
        "maximum_primal_residual": maximum_primal,
        "maximum_adjoint_residual": maximum_adjoint,
        "topology_hashes": sorted(topology_hashes),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def summarize_symmetry_panel(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Require one rigid-symmetry and one closed-loop record per molecule."""

    panel = load_pes_panel()
    values = tuple(records)
    by_id = {str(item.get("molecule_id")): item for item in values}
    expected = tuple(item.molecule_id for item in panel)
    if (
        len(values) != PES_PANEL_MOLECULE_COUNT
        or len(by_id) != len(values)
        or set(by_id) != set(expected)
    ):
        raise ValueError("Symmetry panel molecule coverage is incomplete.")
    maximums = {
        "translation_energy_abs_eV": 0.0,
        "net_force_norm_eV_per_A": 0.0,
        "torque_norm_eV": 0.0,
        "rotation_energy_abs_eV": 0.0,
        "rotation_force_covariance_relative": 0.0,
        "rotation_source_covariance_relative": 0.0,
        "permutation_energy_abs_eV": 0.0,
        "permutation_force_covariance_relative": 0.0,
        "permutation_source_covariance_relative": 0.0,
        "absolute_loop_work_eV": 0.0,
        "primal_residual": 0.0,
        "adjoint_residual": 0.0,
    }
    rigid_pass = True
    loop_pass = True
    all_topology = True
    for molecule_id in expected:
        record = by_id[molecule_id]
        if record.get("contract_version") != SYMMETRY_PANEL_CONTRACT_VERSION:
            raise ValueError("Symmetry record has the wrong contract version.")
        rigid = record.get("rigid_symmetry")
        loop = record.get("closed_loop")
        if not isinstance(rigid, Mapping) or not isinstance(loop, Mapping):
            raise ValueError("Every symmetry record requires rigid and loop evidence.")
        rigid_pass &= bool(rigid.get("all_gates_passed")) and all(
            bool(value) for value in rigid.get("gates", {}).values()
        )
        loop_pass &= bool(loop.get("all_gates_passed")) and all(
            bool(value) for value in loop.get("gates", {}).values()
        )
        all_topology &= bool(loop.get("fixed_topology"))
        maxima = {
            "translation_energy_abs_eV": rigid["translation"]["energy_abs_eV"],
            "net_force_norm_eV_per_A": max(
                rigid["base_net_force_norm_eV_per_A"],
                rigid["translation"]["net_force_norm_eV_per_A"],
            ),
            "torque_norm_eV": rigid["base_torque_norm_eV"],
            "rotation_energy_abs_eV": rigid["maximum_rotation_energy_abs_eV"],
            "rotation_force_covariance_relative": rigid[
                "maximum_rotation_force_covariance_relative"
            ],
            "rotation_source_covariance_relative": rigid[
                "maximum_rotation_source_covariance_relative"
            ],
            "permutation_energy_abs_eV": rigid["permutation"]["energy_abs_eV"],
            "permutation_force_covariance_relative": rigid["permutation"][
                "force_covariance_relative"
            ],
            "permutation_source_covariance_relative": rigid["permutation"][
                "source_covariance_relative"
            ],
            "absolute_loop_work_eV": loop["maximum_absolute_loop_work_eV"],
            "primal_residual": max(
                rigid["maximum_primal_residual"], loop["maximum_primal_residual"]
            ),
            "adjoint_residual": max(
                rigid["maximum_adjoint_residual"], loop["maximum_adjoint_residual"]
            ),
        }
        for name, value in maxima.items():
            maximums[name] = max(maximums[name], float(value))
    gates = {
        "all_rigid_symmetry_gates": rigid_pass,
        "all_bidirectional_cold_warm_loop_gates": loop_pass,
        "all_molecule_topologies_fixed": all_topology,
    }
    return {
        "schema_version": SYMMETRY_PANEL_SCHEMA_VERSION,
        "contract_version": SYMMETRY_PANEL_CONTRACT_VERSION,
        "molecule_count": len(values),
        "molecule_ids": list(expected),
        "maximum": maximums,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def summarize_bidirectional_loop_record(record: Mapping[str, Any]) -> dict[str, object]:
    """Validate one precomputed cold/warm forward/reverse loop record."""

    names = ("cold_forward", "cold_reverse", "warm_forward", "warm_reverse")
    work: dict[str, dict[str, object]] = {}
    maximum_absolute_work = 0.0
    for name in names:
        raw = record.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError("Closed-loop record is missing one traversal.")
        value = float(raw.get("simpson_work_eV"))
        threshold = float(raw.get("gate_threshold_eV"))
        if (
            not math.isfinite(value)
            or not math.isfinite(threshold)
            or threshold < 0.0
        ):
            raise ValueError("Closed-loop work/threshold must be finite.")
        passed = abs(value) <= threshold
        work[name] = {
            "simpson_work_eV": value,
            "gate_threshold_eV": threshold,
            "recomputed_gate_passed": passed,
        }
        maximum_absolute_work = max(maximum_absolute_work, abs(value))
    maximum_primal = float(record.get("maximum_primal_residual"))
    maximum_adjoint = float(record.get("maximum_adjoint_residual"))
    topology_hashes = tuple(str(value) for value in record.get("topology_hashes", ()))
    if (
        not math.isfinite(maximum_primal)
        or not math.isfinite(maximum_adjoint)
        or maximum_primal < 0.0
        or maximum_adjoint < 0.0
        or not topology_hashes
    ):
        raise ValueError("Closed-loop residual/topology record is invalid.")
    cold_roots = bool(record.get("all_cold_warm_roots"))
    warm_repeat = bool(record.get("warm_forward_reverse_repeat"))
    gates = {
        "all_four_loop_work_gates": all(
            bool(value["recomputed_gate_passed"]) for value in work.values()
        ),
        "cold_forward_reverse_antisymmetry_le_1e-10_eV": abs(
            float(work["cold_forward"]["simpson_work_eV"])
            + float(work["cold_reverse"]["simpson_work_eV"])
        )
        <= 1.0e-10,
        "warm_forward_reverse_antisymmetry_le_1e-8_eV": abs(
            float(work["warm_forward"]["simpson_work_eV"])
            + float(work["warm_reverse"]["simpson_work_eV"])
        )
        <= 1.0e-8,
        "all_loop_cold_warm_roots": cold_roots,
        "warm_forward_reverse_repeat": warm_repeat,
        "all_primal_residuals_le_1e-12": (
            maximum_primal <= PRIMAL_RESIDUAL_TOLERANCE
        ),
        "all_adjoint_residuals_le_1e-10": (
            maximum_adjoint <= ADJOINT_RESIDUAL_TOLERANCE
        ),
        "fixed_topology": len(set(topology_hashes)) == 1,
    }
    return {
        "schema_version": "route2-fixedbox590-bidirectional-loop-summary-v1",
        **work,
        "maximum_absolute_loop_work_eV": maximum_absolute_work,
        "maximum_primal_residual": maximum_primal,
        "maximum_adjoint_residual": maximum_adjoint,
        "topology_hashes": sorted(set(topology_hashes)),
        "fixed_topology": len(set(topology_hashes)) == 1,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


__all__ = [
    "ADJOINT_RESIDUAL_TOLERANCE",
    "NET_FORCE_TOLERANCE_EV_PER_A",
    "PRIMAL_RESIDUAL_TOLERANCE",
    "RIGID_ENERGY_TOLERANCE_EV",
    "ROTATION_FORCE_RELATIVE_TOLERANCE",
    "SOURCE_COVARIANCE_RELATIVE_TOLERANCE",
    "SYMMETRY_PANEL_CONTRACT_VERSION",
    "SYMMETRY_PANEL_LOOP_AMPLITUDES_A",
    "SYMMETRY_PANEL_LOOP_SUBDIVISIONS",
    "SYMMETRY_PANEL_RANDOM_SEED",
    "SYMMETRY_PANEL_ROTATION_COUNT",
    "SYMMETRY_PANEL_SCHEMA_VERSION",
    "SYMMETRY_PANEL_TRANSLATION_A",
    "TORQUE_TOLERANCE_EV",
    "rotate_radial_gto_blocks",
    "summarize_rigid_symmetry",
    "summarize_bidirectional_loop_record",
    "summarize_symmetry_panel",
    "symmetry_panel_permutation",
    "symmetry_panel_rotations",
]
