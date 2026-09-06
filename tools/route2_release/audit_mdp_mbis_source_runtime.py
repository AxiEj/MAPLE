"""Audit the frozen MDP-MBIS source head under rigid motions and coordinate AD.

The audit opens no solvation target and evaluates no PCM energy.  It selects a
molecule-held-out neutral SPICE geometry, reruns the actual frozen MACE-MDP
backbone on CUDA/CPU after rigid transformations, and verifies the source-head
provider rather than only the algebraic projection helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np


SUPPORTED_ATOMIC_NUMBERS = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _molecule_split(name: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") % 10
    if bucket < 7:
        return "train"
    if bucket == 7:
        return "validation"
    return "test"


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    cross = np.asarray(
        (
            (0.0, -vector[2], vector[1]),
            (vector[2], 0.0, -vector[0]),
            (-vector[1], vector[0], 0.0),
        )
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(vector, vector)
        + np.sin(angle) * cross
    )


def _held_out_geometry(dataset: Path) -> tuple[str, np.ndarray, np.ndarray]:
    import h5py

    supported = set(SUPPORTED_ATOMIC_NUMBERS)
    with h5py.File(dataset, "r") as handle:
        for name in sorted(handle):
            if _molecule_split(name) != "test":
                continue
            group = handle[name]
            if not {"atomic_numbers", "positions", "total_charge"}.issubset(group):
                continue
            numbers = np.asarray(group["atomic_numbers"], dtype=int).reshape(-1)
            charge = np.asarray(group["total_charge"], dtype=float).reshape(-1)
            if not set(numbers).issubset(supported):
                continue
            neutral = np.flatnonzero(np.rint(charge) == 0.0)
            if not len(neutral):
                continue
            index = int(neutral[0])
            positions = np.asarray(group["positions"][index], dtype=float) * 10.0
            if positions.shape != (len(numbers), 3) or not np.all(
                np.isfinite(positions)
            ):
                raise RuntimeError("Held-out SPICE geometry is malformed.")
            return name, numbers, positions
    raise RuntimeError("No eligible held-out neutral SPICE geometry was found.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    source_root = arguments.source_root.expanduser().resolve(strict=True)
    dataset = arguments.dataset.expanduser().resolve(strict=True)
    checkpoint = arguments.checkpoint.expanduser().resolve(strict=True)
    source_head = arguments.source_head.expanduser().resolve(strict=True)
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")

    import sys

    sys.path.insert(0, str(source_root))
    import torch
    from ase import Atoms

    from maple.function.calculator.extra_correction.implicit.gto_density import (
        cartesian_multipoles,
    )
    from maple.solvation.models.mace_mdp_mbis import (
        MACE_MDP_MBIS_SOURCE_HEAD_EXPECTED_SHA256,
        build_mace_mdp_mbis_source_adapter,
    )

    started = time.perf_counter()
    molecule, numbers, positions = _held_out_geometry(dataset)
    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        info={"charge": 0, "mult": 1},
    )
    adapter = build_mace_mdp_mbis_source_adapter(
        checkpoint_path=checkpoint,
        source_head_path=source_head,
        device=arguments.device,
        expected_source_head_sha256=MACE_MDP_MBIS_SOURCE_HEAD_EXPECTED_SHA256,
    )
    state = adapter.evaluate_state(atoms)
    base_q, base_p = cartesian_multipoles(state.source4_raw_l1)

    rotations = (
        _rotation((0.31, -0.72, 0.61), 0.731),
        _rotation((-0.51, 0.22, 0.83), -1.117),
        _rotation((0.11, 0.94, -0.32), 2.009),
    )
    rotation_q_errors: list[float] = []
    rotation_p_errors: list[float] = []
    rotation_mu_errors: list[float] = []
    for rotation in rotations:
        transformed = atoms.copy()
        transformed.positions = positions @ rotation.T
        rotated_state = adapter.evaluate_state(transformed)
        rotated_q, rotated_p = cartesian_multipoles(rotated_state.source4_raw_l1)
        rotation_q_errors.append(float(np.max(np.abs(rotated_q - base_q))))
        rotation_p_errors.append(
            float(np.max(np.abs(rotated_p - base_p @ rotation.T)))
        )
        rotation_mu_errors.append(
            float(
                np.max(
                    np.abs(
                        rotated_state.public_molecular_dipole_eangstrom
                        - state.public_molecular_dipole_eangstrom @ rotation.T
                    )
                )
            )
        )

    translated = atoms.copy()
    translated.positions = positions + np.asarray((1.2, -0.7, 0.4))
    translated_state = adapter.evaluate_state(translated)
    translated_q, translated_p = cartesian_multipoles(
        translated_state.source4_raw_l1
    )

    order = np.arange(len(atoms))[::-1]
    permuted = Atoms(
        numbers=numbers[order],
        positions=positions[order],
        info={"charge": 0, "mult": 1},
    )
    permuted_state = adapter.evaluate_state(permuted)
    permuted_q, permuted_p = cartesian_multipoles(permuted_state.source4_raw_l1)

    rng = np.random.default_rng(20260817)
    cotangent = rng.normal(size=state.source4_raw_l1.shape)
    direction = rng.normal(size=positions.shape)
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    analytic = float(
        np.sum(adapter.source_position_vjp(atoms, cotangent) * direction)
    )

    def contraction(displacement: float) -> float:
        moved = atoms.copy()
        moved.positions = positions + displacement * direction
        return float(np.sum(adapter.evaluate_source(moved) * cotangent))

    finite_difference: list[dict[str, float]] = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        value = (contraction(step) - contraction(-step)) / (2.0 * step)
        finite_difference.append(
            {
                "step_angstrom": step,
                "central_difference": value,
                "absolute_error": abs(value - analytic),
            }
        )

    measurements = {
        "maximum_rotation_charge_error_e": max(rotation_q_errors),
        "maximum_rotation_atomic_dipole_error_eangstrom": max(rotation_p_errors),
        "maximum_rotation_public_dipole_error_eangstrom": max(rotation_mu_errors),
        "maximum_translation_charge_error_e": float(
            np.max(np.abs(translated_q - base_q))
        ),
        "maximum_translation_atomic_dipole_error_eangstrom": float(
            np.max(np.abs(translated_p - base_p))
        ),
        "maximum_permutation_charge_error_e": float(
            np.max(np.abs(permuted_q[::-1] - base_q))
        ),
        "maximum_permutation_atomic_dipole_error_eangstrom": float(
            np.max(np.abs(permuted_p[::-1] - base_p))
        ),
        "total_charge_error_e": abs(float(np.sum(base_q))),
        "molecular_dipole_closure_max_error_eangstrom": float(
            np.max(
                np.abs(
                    np.sum(base_q[:, None] * positions + base_p, axis=0)
                    - state.public_molecular_dipole_eangstrom
                )
            )
        ),
        "projection_condition_number": state.projection_condition_number,
        "position_vjp_directional_value": analytic,
        "position_vjp_finite_difference": finite_difference,
    }
    gates = {
        "rotation": max(rotation_q_errors + rotation_p_errors + rotation_mu_errors)
        <= 2.0e-9,
        "translation": max(
            measurements["maximum_translation_charge_error_e"],
            measurements["maximum_translation_atomic_dipole_error_eangstrom"],
        )
        <= 2.0e-9,
        "permutation": max(
            measurements["maximum_permutation_charge_error_e"],
            measurements["maximum_permutation_atomic_dipole_error_eangstrom"],
        )
        <= 2.0e-9,
        "constraint_closure": max(
            measurements["total_charge_error_e"],
            measurements["molecular_dipole_closure_max_error_eangstrom"],
        )
        <= 2.0e-11,
        "coordinate_vjp": finite_difference[-1]["absolute_error"] <= 2.0e-7,
    }
    if not all(gates.values()):
        raise RuntimeError(f"MDP-MBIS source runtime audit failed: {gates}")

    report = {
        "artifact": "route2-mdp-mbis-source-runtime-audit-v1",
        "claim_boundary": {
            "solvation_targets_read": False,
            "pcm_energy_evaluated": False,
            "held_out_molecule_split": True,
            "source_accuracy_admitted": False,
            "hybrid_accuracy_admitted": False,
            "force_capability_admitted": False,
        },
        "identity": {
            "molecule": molecule,
            "atom_count": len(atoms),
            "source_configuration_sha256": adapter.configuration_sha256(),
            "source_state_sha256": state.state_sha256,
        },
        "inputs_sha256": {
            "dataset": _sha256_file(dataset),
            "checkpoint": _sha256_file(checkpoint),
            "source_head": _sha256_file(source_head),
            "audit_script": _sha256_file(Path(__file__)),
            "runtime_adapter": _sha256_file(
                source_root / "maple/solvation/models/mace_mdp_mbis.py"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": arguments.device,
            "gpu": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if arguments.device.startswith("cuda")
                else None
            ),
        },
        "measurements": measurements,
        "gates": gates,
        "wall_seconds": time.perf_counter() - started,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=str(output.parent)
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
