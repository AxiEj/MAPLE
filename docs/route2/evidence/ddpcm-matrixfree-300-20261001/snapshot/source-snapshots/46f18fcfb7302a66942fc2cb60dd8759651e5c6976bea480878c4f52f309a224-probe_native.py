#!/usr/bin/env python3
"""Benchmark the independent native pyddx reference; never an admission tool."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from native_source_adjoint import solve_native_reference


def _synthetic_chain(atom_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(atom_count, dtype=np.float64)
    positions = np.column_stack((1.63*x, 0.31*np.sin(x*1.37), 0.23*np.cos(x*0.79)))
    radii = np.full(atom_count, 1.7, dtype=np.float64)
    density = rng.normal(scale=0.025, size=(atom_count, 4))
    density[:, 0] -= np.mean(density[:, 0])
    density[:, 1:] *= .1
    return positions, radii, density


def _source_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    snapshot_dir = HERE / "source-snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    for name in ("native_source_adjoint.py", "probe_native.py"):
        path = HERE / name
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        target = snapshot_dir / f"{digest}-{name}"
        if not target.exists():
            target.write_bytes(payload)
        result[name] = digest
    return result


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, choices=(10, 50, 100, 300), default=10)
    parser.add_argument("--seed", type=int, default=61001)
    parser.add_argument("--source-tile", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite prior evidence: {args.output}")
    snapshot = _source_snapshot()  # immutable evidence before the timed region
    positions, radii, density = _synthetic_chain(args.atoms, args.seed)
    input_sha256 = _array_digest(positions, radii, density)
    started = time.perf_counter()
    payload = {
        "status": "research-reference-failed-not-admitted",
        "atom_count": args.atoms,
        "seed": args.seed,
        "source_tile": args.source_tile,
        "source_snapshot_sha256": snapshot,
        "input_sha256": input_sha256,
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pyddx": importlib.metadata.version("pyddx"),
        },
        "settings": {
            "backend": "pyddx-0.8.0-native",
            "model": "pcm",
            "lmax": 15,
            "n_lebedev": 1202,
            "eta": 0.1,
            "tolerance": 1.0e-12,
            "enable_fmm": False,
            "incore": False,
            "dtype": "float64",
            "requested_tolerance": 1.0e-12,
            "original_residual_measured_independently": False,
            "residual_evidence_boundary": (
                "native solver completion and requested tolerance only"
            ),
        },
        "capability_flags": {
            "mace": False,
            "cds": False,
            "hessian": False,
            "hvp": False,
            "torch_support": False,
            "public_input_path": False,
            "scientifically_admitted": False,
        },
    }
    exit_status = 1
    try:
        result = solve_native_reference(
            positions,
            radii,
            density,
            lmax=15,
            n_lebedev=1202,
            eta=0.1,
            tolerance=1.0e-12,
            source_tile=args.source_tile,
        )
        payload.update(
            {
                "status": "research-reference-only-not-admitted",
                "energy_eV": result.energy_ev,
                "source_gradient_max_abs_eV": float(
                    np.max(np.abs(result.source_gradient_ev))
                ),
                "fixed_source_position_gradient_max_abs_eV_A": float(
                    np.max(np.abs(result.position_gradient_ev_per_angstrom))
                ),
                "analytic_fixed_source_force_max_abs_eV_A": float(
                    np.max(np.abs(result.fixed_source_force_ev_per_angstrom))
                ),
                "force_sign_contract": "force=-fixed_source_position_gradient",
                "forward_iterations": result.forward_iterations,
                "adjoint_iterations": result.adjoint_iterations,
                "n_cavity": result.n_cavity,
                "n_basis": result.n_basis,
            }
        )
        exit_status = 0
    except Exception as error:
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["wall_seconds"] = time.perf_counter() - started
    payload["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    raise SystemExit(exit_status)


if __name__ == "__main__":
    main()
