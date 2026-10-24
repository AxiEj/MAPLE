"""Reproduce the pinned AnySolv-MoE CPU mechanics audit.

This script calls only the official public inference API.  It neither changes
the checkpoint nor relaxes the model's numerical settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


SOURCE_REVISION = "1778e1f18384e5f59de1475182765733a7401e39"
CHECKPOINT_SHA256 = "77b7d883ac1a01fc13d78b8cf5530e5709af679e6fbdec53cb086ae0a90bd251"
H_VALUES = (1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    revision = subprocess.run(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"source revision mismatch: {revision}")
    checkpoint_digest = sha256(args.checkpoint)
    if checkpoint_digest != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA256 mismatch: {checkpoint_digest}")

    sys.path.insert(0, str(args.source_root.parent))
    from anysolv import predict_solvation_energy

    numbers = np.array([8, 1, 1], dtype=int)
    positions = np.array(
        [[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]],
        dtype=float,
    )
    ry = np.array(
        [[np.cos(-0.47), 0.0, np.sin(-0.47)], [0.0, 1.0, 0.0],
         [-np.sin(-0.47), 0.0, np.cos(-0.47)]],
        dtype=float,
    )
    rz = np.array(
        [[np.cos(0.61), -np.sin(0.61), 0.0],
         [np.sin(0.61), np.cos(0.61), 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    rotation = ry @ rz

    def evaluate(coords: np.ndarray, solvent="water") -> tuple[float, np.ndarray]:
        energy, forces = predict_solvation_energy(
            (numbers, coords), charge=0, spin=1, solvent=solvent,
            checkpoint=str(args.checkpoint), device="cpu", dtype=torch.float64,
            inference_settings="default",
        )
        return float(energy), np.asarray(forces, dtype=float)

    def full_fd(coords: np.ndarray, analytic: np.ndarray) -> dict[str, object]:
        rows: dict[str, object] = {}
        for h in H_VALUES:
            fd = np.empty_like(coords)
            for atom in range(len(coords)):
                for axis in range(3):
                    plus = coords.copy(); plus[atom, axis] += h
                    minus = coords.copy(); minus[atom, axis] -= h
                    ep, _ = evaluate(plus)
                    em, _ = evaluate(minus)
                    fd[atom, axis] = -(ep - em) / (2.0 * h)
            error = analytic - fd
            rows[str(h)] = {
                "finite_difference_forces_eV_per_A": fd.tolist(),
                "max_abs_error_eV_per_A": float(np.max(np.abs(error))),
                "rms_error_eV_per_A": float(np.sqrt(np.mean(error * error))),
            }
        return rows

    base_energy, base_forces = evaluate(positions)
    rotated_positions = positions @ rotation.T
    rotated_energy, rotated_forces = evaluate(rotated_positions)
    translated_energy, translated_forces = evaluate(
        positions + np.array([1.7, -0.8, 2.3])
    )
    vacuum_energy, vacuum_forces = evaluate(positions, None)
    toluene_energy, toluene_forces = evaluate(positions, "toluene")
    expected_rotated_forces = base_forces @ rotation.T

    payload = {
        "schema_version": 1,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "source_revision": revision,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_digest,
            "inference_settings": "default",
            "device": "cpu",
            "dtype": "float64",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_force_no_weights_only_load": os.environ.get(
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
            ),
        },
        "input": {"numbers": numbers.tolist(), "positions_A": positions.tolist()},
        "raw": {
            "water": {"energy_eV": base_energy, "forces_eV_per_A": base_forces.tolist()},
            "water_rotated": {"energy_eV": rotated_energy, "forces_eV_per_A": rotated_forces.tolist()},
            "vacuum": {"energy_eV": vacuum_energy, "forces_eV_per_A": vacuum_forces.tolist()},
            "toluene": {"energy_eV": toluene_energy, "forces_eV_per_A": toluene_forces.tolist()},
        },
        "invariance": {
            "rotation_matrix": rotation.tolist(),
            "rotation_energy_abs_error_eV": abs(rotated_energy - base_energy),
            "rotation_force_max_abs_error_eV_per_A": float(
                np.max(np.abs(rotated_forces - expected_rotated_forces))
            ),
            "translation_energy_abs_error_eV": abs(translated_energy - base_energy),
            "translation_force_max_abs_error_eV_per_A": float(
                np.max(np.abs(translated_forces - base_forces))
            ),
        },
        "vacuum_gate": {
            "energy_exact_zero": vacuum_energy == 0.0,
            "force_max_abs_eV_per_A": float(np.max(np.abs(vacuum_forces))),
        },
        "finite_difference": {
            "formula": "F_i,a = -(E(R+h e_i,a)-E(R-h e_i,a))/(2h)",
            "base_orientation": full_fd(positions, base_forces),
            "rotated_orientation": full_fd(rotated_positions, rotated_forces),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "rotation_force_max_abs_error_eV_per_A": payload["invariance"]["rotation_force_max_abs_error_eV_per_A"],
        "base_fd_max_by_h": {h: x["max_abs_error_eV_per_A"] for h, x in payload["finite_difference"]["base_orientation"].items()},
        "rotated_fd_max_by_h": {h: x["max_abs_error_eV_per_A"] for h, x in payload["finite_difference"]["rotated_orientation"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
