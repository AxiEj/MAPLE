#!/usr/bin/env python3
"""Target-free real-checkpoint covariance audit for the MDP/POLAR transform.

Only atomic numbers and one frozen geometry are read from SPICE.  No MBIS,
energy, force, or solvation target is loaded.  The audit applies a fixed proper
rotation, translation, and atom permutation, then checks the independently
evaluated MDP molecular polarizability, POLAR zero-field source/Jacobian, and
the derived native-field susceptibility transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release import (  # noqa: E402
    run_mdp_polar_zero_training_source_gate as prior,
)

SELECTION_PREREGISTRATION = REPO_ROOT / (
    "docs/route2/preregistrations/mdp-polar-ewald-gauge-l3-tail-v1.json"
)
IMPLEMENTATION = REPO_ROOT / (
    "maple/solvation/release/uniform_susceptibility_replacement.py"
)
ARTIFACT = "route2-target-free-mdp-polar-uniform-transform-covariance-v1"
SELECTION_INDEX = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


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


def _load_geometry(identity: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(prior.DATASET, "r") as handle:
        group = handle[str(identity["molecule"])]
        numbers = np.asarray(group["atomic_numbers"], dtype=np.int64).reshape(-1)
        positions_raw = np.asarray(group["positions"][int(identity["configuration"])])
    if _array_sha256(numbers) != identity["atomic_numbers_sha256"]:
        raise RuntimeError("Atomic-number identity changed.")
    if _array_sha256(positions_raw) != identity["positions_raw_sha256"]:
        raise RuntimeError("Geometry identity changed.")
    return numbers, np.asarray(positions_raw, dtype=np.float64) * 10.0


def _rotation() -> np.ndarray:
    axis = np.array([0.31, -0.47, 0.57], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _rotate_native_field(field: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(field, copy=True)
    for channels in ((4, 2, 3), (7, 5, 6)):
        result[:, channels] = field[:, channels] @ rotation.T
    return result


def _rotate_source(source: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(source, copy=True)
    result[:, (3, 1, 2)] = source[:, (3, 1, 2)] @ rotation.T
    return result


def _rotate_source_jacobian(jacobian: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.empty_like(jacobian)
    for axis in range(3):
        old_source = np.einsum("nsc,c->ns", jacobian, rotation.T[:, axis])
        result[:, :, axis] = _rotate_source(old_source, rotation)
    return result


def _relative(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(np.linalg.norm(right_array), np.finfo(np.float64).tiny)
    )


def _evaluate(numbers, positions, mdp, polar):
    from ase import Atoms

    from maple.solvation.release.uniform_response_manifold import (
        affine_uniform_native_field,
    )
    from maple.solvation.release.uniform_susceptibility_replacement import (
        prepare_uniform_susceptibility_field_transform,
    )

    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        info={"charge": 0, "multiplicity": 1},
    )
    mdp_state = mdp.evaluate(atoms)
    zero = np.zeros((len(atoms), 8), dtype=np.float64)
    polar_zero = polar.evaluate_source(atoms, zero)
    basis = np.stack(
        [affine_uniform_native_field(positions, np.eye(3)[axis]) for axis in range(3)],
        axis=-1,
    )
    polar_jacobian = np.stack(
        [polar.field_jvp(atoms, zero, basis[:, :, axis]) for axis in range(3)],
        axis=-1,
    )
    prepared = prepare_uniform_susceptibility_field_transform(
        positions_angstrom=positions,
        uniform_native_basis=basis,
        polar_zero_uniform_source_jacobian=polar_jacobian,
        mdp_molecular_polarizability_eangstrom2_per_volt=(
            mdp_state.public_polarizability_eangstrom2_per_volt
        ),
    )
    return mdp_state, polar_zero, polar_jacobian, prepared


def run(*, device: str) -> dict[str, Any]:
    preregistration = json.loads(SELECTION_PREREGISTRATION.read_text())
    selected = preregistration["selection"]["records"][SELECTION_INDEX]
    identity = selected["selection_identity"]
    numbers, positions = _load_geometry(identity)

    import torch

    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.models import (
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_moment_adapter,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=prior.MDP_CHECKPOINT, device="cpu"
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=prior.POLAR_CHECKPOINT,
        device=device,
        long_range_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    polar = MACEPolarOriginalSourceNativeFieldAdapter(radial)

    original = _evaluate(numbers, positions, mdp, polar)
    rotation = _rotation()
    rng = np.random.default_rng(20260818)
    permutation = rng.permutation(len(numbers))
    translation = np.array([1.7, -0.6, 0.9])
    moved_positions = (positions @ rotation.T + translation)[permutation]
    moved_numbers = numbers[permutation]
    transformed = _evaluate(moved_numbers, moved_positions, mdp, polar)

    mdp0, source0, jacobian0, chart0 = original
    mdp1, source1, jacobian1, chart1 = transformed
    expected_source = _rotate_source(source0, rotation)[permutation]
    expected_jacobian = _rotate_source_jacobian(jacobian0, rotation)[permutation]
    expected_alpha = (
        rotation @ mdp0.public_polarizability_eangstrom2_per_volt @ rotation.T
    )
    expected_transform = rotation @ chart0.uniform_coordinate_transform @ rotation.T
    field = rng.normal(size=(len(numbers), 8))
    moved_field = _rotate_native_field(field, rotation)[permutation]
    expected_corrected_field = _rotate_native_field(
        chart0.transform_field(field), rotation
    )[permutation]

    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "selection_identity": identity,
        "selection_index": SELECTION_INDEX,
        "claim_boundary": {
            "accuracy_or_capability_admitted": False,
            "experimental_or_mbis_targets_read": False,
            "fitting_or_post_training_performed": False,
            "opened_geometry_used_for_structural_diagnostic_only": True,
            "pure_mace_polar_in_scope": False,
        },
        "input_sha256": {
            "dataset": _sha256_file(prior.DATASET),
            "mace_mdp_checkpoint": _sha256_file(prior.MDP_CHECKPOINT),
            "mace_polar_checkpoint": _sha256_file(prior.POLAR_CHECKPOINT),
            "selection_preregistration": _sha256_file(SELECTION_PREREGISTRATION),
            "susceptibility_implementation": _sha256_file(IMPLEMENTATION),
            "mace_mdp_adapter": _sha256_file(
                REPO_ROOT / "maple/solvation/models/mace_mdp.py"
            ),
            "mace_polar_adapter": _sha256_file(
                REPO_ROOT / "maple/solvation/models/mace_polar_separated.py"
            ),
            "runner": _sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
        },
        "transform": {
            "rotation_matrix": rotation.tolist(),
            "translation_angstrom": translation.tolist(),
            "permutation_sha256": _array_sha256(permutation),
        },
        "metrics": {
            "mdp_molecular_alpha_covariance_relative_error": _relative(
                mdp1.public_polarizability_eangstrom2_per_volt, expected_alpha
            ),
            "polar_zero_source_covariance_relative_error": _relative(
                source1, expected_source
            ),
            "polar_uniform_source_jacobian_covariance_relative_error": _relative(
                jacobian1, expected_jacobian
            ),
            "coordinate_transform_covariance_relative_error": _relative(
                chart1.uniform_coordinate_transform, expected_transform
            ),
            "corrected_native_field_covariance_relative_error": _relative(
                chart1.transform_field(moved_field), expected_corrected_field
            ),
            "original_transform_condition_number": (
                chart0.coordinate_transform_condition_number
            ),
            "transformed_transform_condition_number": (
                chart1.coordinate_transform_condition_number
            ),
        },
        "state_sha256": {
            "original": chart0.state_sha256,
            "transformed": chart1.state_sha256,
        },
    }
    payload["record_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
