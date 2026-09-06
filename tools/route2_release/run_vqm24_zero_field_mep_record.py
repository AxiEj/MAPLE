#!/usr/bin/env python3
"""Evaluate one converged gas checkpoint on frozen zero-field MEP probes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pyscf
from pyscf import lib


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools.route2_release import run_vqm24_localized_qm_response as finite  # noqa: E402


SELF_REPO_PATH = "tools/route2_release/run_vqm24_zero_field_mep_record.py"
ARTIFACT = "route2-vqm24-zero-field-full-mep-record-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    surface_path = args.surface.expanduser().resolve(strict=True)
    output_json = args.output_json.expanduser().resolve()
    output_npz = args.output_npz.expanduser().resolve()
    if output_json.exists() or output_npz.exists():
        raise FileExistsError(output_json if output_json.exists() else output_npz)
    if args.threads != 8 or pyscf.__version__ != "2.13.1":
        raise RuntimeError("Zero-field MEP records require PySCF 2.13.1 and 8 threads.")
    molecule = lib.chkfile.load_mol(str(checkpoint))
    if molecule.charge != 0 or molecule.spin != 0:
        raise RuntimeError("Zero-field MEP records currently accept neutral singlets only.")
    density, density_provenance = finite._closed_shell_density_from_checkpoint(
        checkpoint,
        molecule,
    )
    with np.load(surface_path, allow_pickle=False) as state:
        points = np.asarray(state["surface_points_bohr"], dtype=np.float64)
        weights = np.asarray(state["quadrature_weights"], dtype=np.float64)
        parents = np.asarray(state["parent_atom_indices"], dtype=np.int64)
        partitions = np.asarray(state["partition_indices"], dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or weights.shape != (len(points),)
        or parents.shape != (len(points),)
        or partitions.shape != (len(points),)
        or set(partitions.tolist()) != {0, 1}
    ):
        raise RuntimeError("Zero-field exterior probe surface is invalid.")
    lib.num_threads(args.threads)
    started = time.perf_counter()
    integrals = finite._coulomb_integrals(molecule, points)
    electronic = finite._electronic_potential(density, integrals)
    distances = np.linalg.norm(
        points[:, None, :] - molecule.atom_coords()[None, :, :],
        axis=2,
    )
    if np.any(distances <= 0.0):
        raise RuntimeError("Zero-field MEP probe coincides with a nucleus.")
    nuclear = np.sum(
        molecule.atom_charges()[None, :] / distances,
        axis=1,
    )
    total_mep = electronic + nuclear
    dipole_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3),
        dtype=np.float64,
    )
    total_dipole = finite._total_dipole_e_bohr(
        molecule,
        density,
        dipole_integrals,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("xb") as handle:
        np.savez(
            handle,
            surface_points_bohr=points,
            quadrature_weights=weights,
            parent_atom_indices=parents,
            partition_indices=partitions,
            total_surface_mep_hartree_per_e=total_mep,
            total_dipole_e_bohr=total_dipole,
        )
    result = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "success",
        "input": {
            "checkpoint_sha256": _sha256(checkpoint),
            "surface_npz_sha256": _sha256(surface_path),
            "surface_points_sha256": _sha256_array(points),
        },
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "checkpoint_density_helper_path": finite.BENCHMARK_HELPER_REPO_PATH,
            "checkpoint_density_helper_sha256": _sha256(
                SOURCE_ROOT / finite.BENCHMARK_HELPER_REPO_PATH
            ),
        },
        "zero_density": density_provenance,
        "output": {
            "npz_sha256": _sha256(output_npz),
            "total_surface_mep_sha256": _sha256_array(total_mep),
            "total_dipole_e_bohr": total_dipole.tolist(),
            "fit_point_count": int(np.count_nonzero(partitions == 0)),
            "audit_point_count": int(np.count_nonzero(partitions == 1)),
        },
        "runtime": {
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "threads": lib.num_threads(),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "claim_boundary": {
            "independent_qm_zero_field_mep_generated": True,
            "field_response_generated": False,
            "density_or_partition_coefficient_label_emitted": False,
            "fit_or_training_performed": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "capability_admitted": False,
        },
    }
    finite._write_exclusive_json(output_json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
