#!/usr/bin/env python3
"""Target-free real-checkpoint rank audit for the arithmetic tangent candidate.

The rejected candidate constrains its additive source correction to the three
uniform MACE-POLAR source directions ``B = J_P U``.  If the molecular dipole
response ``C_P = C B`` has rank below three, every corrected molecular response
``C_P X`` remains in the same lower-dimensional range and therefore cannot
equal the full-rank MACE-MDP molecular polarizability.

This runner evaluates that obstruction on a fixed water geometry and on an
independently rotated/translated copy.  It reads no experimental, MBIS, or
solvation target and performs no fitting.  A Moore-Penrose solve is used only
to report the exact least-squares lower bound; it is not a proposed repair.
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

MDP_CHECKPOINT = Path("/home/axie/.cache/mace/MACE-MDP.model")
POLAR_CHECKPOINT = Path("/home/axie/.cache/mace/MACEPOLAR1Mmodel")
ARTIFACT = "route2-target-free-mdp-polar-planar-rank-obstruction-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _water_geometry() -> tuple[np.ndarray, np.ndarray]:
    numbers = np.array([8, 1, 1], dtype=np.int64)
    positions = np.array(
        [
            [0.000000000000, 0.000000000000, 0.000000000000],
            [0.957200000000, 0.000000000000, 0.000000000000],
            [-0.239987208409, 0.926627206485, 0.000000000000],
        ],
        dtype=np.float64,
    )
    return numbers, positions


def _evaluate(numbers, positions, mdp, polar) -> dict[str, Any]:
    from ase import Atoms

    from maple.solvation.release.uniform_response_manifold import (
        affine_uniform_native_field,
        molecular_dipole_eangstrom,
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
    basis = np.stack(
        [affine_uniform_native_field(positions, np.eye(3)[axis]) for axis in range(3)],
        axis=-1,
    )
    source_jacobian = np.stack(
        [polar.field_jvp(atoms, zero, basis[:, :, axis]) for axis in range(3)],
        axis=-1,
    )
    molecular = np.column_stack(
        [
            molecular_dipole_eangstrom(positions, source_jacobian[:, :, axis])
            for axis in range(3)
        ]
    )
    singular_values = np.linalg.svd(molecular, compute_uv=False)
    rank_threshold = float(
        np.finfo(np.float64).eps * max(molecular.shape) * singular_values[0]
    )
    numerical_rank = int(np.sum(singular_values > rank_threshold))
    alpha = np.asarray(
        mdp_state.public_polarizability_eangstrom2_per_volt, dtype=np.float64
    )
    alpha_eigenvalues = np.linalg.eigvalsh(alpha)
    alpha_rank = int(np.linalg.matrix_rank(alpha))
    target = -alpha
    least_squares = np.linalg.pinv(molecular) @ target
    residual = molecular @ least_squares - target
    relative_lower_bound = float(
        np.linalg.norm(residual, ord="fro") / np.linalg.norm(target, ord="fro")
    )
    _, _, right_vectors = np.linalg.svd(molecular)
    right_null = right_vectors[-1]
    plane_normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
    plane_normal /= np.linalg.norm(plane_normal)
    try:
        prepare_uniform_susceptibility_field_transform(
            positions_angstrom=positions,
            uniform_native_basis=basis,
            polar_zero_uniform_source_jacobian=source_jacobian,
            mdp_molecular_polarizability_eangstrom2_per_volt=alpha,
        )
    except ValueError as exc:
        constructor_status = "rejected"
        constructor_error = str(exc)
    else:
        constructor_status = "unexpectedly-accepted"
        constructor_error = ""

    return {
        "positions_angstrom": positions.tolist(),
        "polar_uniform_molecular_dipole_jacobian": molecular.tolist(),
        "polar_uniform_singular_values": singular_values.tolist(),
        "polar_uniform_rank_threshold": rank_threshold,
        "polar_uniform_numerical_rank": numerical_rank,
        "polar_uniform_charge_response": np.sum(
            source_jacobian[:, 0, :], axis=0
        ).tolist(),
        "mdp_molecular_polarizability": alpha.tolist(),
        "mdp_molecular_polarizability_eigenvalues": alpha_eigenvalues.tolist(),
        "mdp_molecular_polarizability_rank": alpha_rank,
        "minimum_relative_frobenius_residual_in_original_polar_span": (
            relative_lower_bound
        ),
        "right_null_vector": right_null.tolist(),
        "molecular_plane_normal": plane_normal.tolist(),
        "absolute_null_plane_alignment": float(abs(right_null @ plane_normal)),
        "arithmetic_transform_constructor_status": constructor_status,
        "arithmetic_transform_constructor_error": constructor_error,
    }


def run(*, device: str) -> dict[str, Any]:
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
    mdp = build_mace_mdp_moment_adapter(checkpoint_path=MDP_CHECKPOINT, device="cpu")
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=POLAR_CHECKPOINT,
        device=device,
        long_range_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    polar = MACEPolarOriginalSourceNativeFieldAdapter(radial)

    numbers, positions = _water_geometry()
    rotation = _rotation()
    translation = np.array([1.7, -0.6, 0.9], dtype=np.float64)
    cases = {
        "reference": _evaluate(numbers, positions, mdp, polar),
        "rotated_translated": _evaluate(
            numbers, positions @ rotation.T + translation, mdp, polar
        ),
    }
    ranks = [case["polar_uniform_numerical_rank"] for case in cases.values()]
    alpha_ranks = [case["mdp_molecular_polarizability_rank"] for case in cases.values()]
    decisions = [
        case["arithmetic_transform_constructor_status"] for case in cases.values()
    ]
    if ranks != [2, 2] or alpha_ranks != [3, 3] or decisions != ["rejected"] * 2:
        raise RuntimeError("Planar-rank obstruction did not reproduce exactly.")

    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "claim_boundary": {
            "accuracy_or_capability_admitted": False,
            "experimental_or_mbis_or_solvation_targets_read": False,
            "fitting_calibration_or_post_training_performed": False,
            "pseudoinverse_used_as_production_repair": False,
            "pure_mace_polar_in_scope": False,
        },
        "mathematical_decision": {
            "restricted_correction": "K = B A G with B = J_P U and G U = I",
            "uniform_corrected_response": "C (J_P + K) U = C_P (I + A)",
            "rank_bound": "rank[C_P (I + A)] <= rank(C_P)",
            "target": "-alpha_MDP, rank three",
            "decision": "reject-arithmetic-span-tangent-on-general-domain",
            "reason": (
                "a rank-two C_P cannot be transformed into the full-rank "
                "MDP molecular susceptibility without leaving the original "
                "POLAR uniform source span"
            ),
        },
        "input_sha256": {
            "mace_mdp_checkpoint": _sha256_file(MDP_CHECKPOINT),
            "mace_polar_checkpoint": _sha256_file(POLAR_CHECKPOINT),
            "mace_mdp_adapter": _sha256_file(
                REPO_ROOT / "maple/solvation/models/mace_mdp.py"
            ),
            "mace_polar_adapter": _sha256_file(
                REPO_ROOT / "maple/solvation/models/mace_polar_separated.py"
            ),
            "susceptibility_implementation": _sha256_file(
                REPO_ROOT
                / "maple/solvation/release/uniform_susceptibility_replacement.py"
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
        },
        "cases": cases,
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
