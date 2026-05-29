#!/usr/bin/env python3
"""Benchmark MAPLE path energy/force batching against legacy NEB evaluation.

The benchmark intentionally measures the NEB/CINEB evaluator shape:

* legacy: evaluate every image energy, then every image force;
* path-batch: one ``calculate_many(..., ("energy", "forces"))`` snapshot.

It also reports max energy/force differences, because speedups only count when
the returned E/F values remain equivalent to the legacy calculator route.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from ase import Atoms


MODEL_DIR = Path("maple/function/calculator/model")
DEFAULT_MAX_ENERGY_DIFF_EH = 1e-7
DEFAULT_MAX_FORCE_DIFF_EH_PER_A = 1e-6

SYMBOLS = ["C", "H", "O", "C", "H", "H", "H"]
BASE_POSITIONS = np.array(
    [
        [0.82606574148010, 0.55086039108041, -0.12617945891172],
        [0.45544459832603, 1.24175134345121, -0.87843176681328],
        [0.56807530830522, 1.02465144289647, 1.11874348469907],
        [1.44110985840975, -0.58736037739486, -0.43636137324384],
        [0.90990913730383, 0.39323174998265, 1.76663207206389],
        [1.59168346964402, -0.85929296968553, -1.47418858923956],
        [1.80895345653104, -1.27347474033034, 0.32372445144544],
    ],
    dtype=np.float64,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_images(calc, n_images: int) -> list[Atoms]:
    center = 0.5 * (n_images - 1)
    images = [
        Atoms(SYMBOLS, positions=BASE_POSITIONS + (i - center) * 0.001)
        for i in range(n_images)
    ]
    for atoms in images:
        atoms.calc = calc
    return images


def _legacy_energy_forces(images: list[Atoms]) -> tuple[np.ndarray, list[np.ndarray]]:
    energies = [
        float(atoms.get_potential_energy(force_consistent=True))
        for atoms in images
    ]
    forces = [np.asarray(atoms.get_forces(), dtype=np.float64) for atoms in images]
    _sync()
    return np.asarray(energies, dtype=np.float64), forces


def _path_batch_energy_forces(images: list[Atoms]) -> tuple[np.ndarray, list[np.ndarray]]:
    from maple.function.dispatcher.ts.algorithm.neb import NEB

    neb = object.__new__(NEB)
    energies, forces = neb._path_energy_forces(images)
    _sync()
    return np.asarray(energies, dtype=np.float64), forces


def _time_call(
    fn: Callable[[list[Atoms]], tuple[np.ndarray, list[np.ndarray]]],
    images: list[Atoms],
    reps: int,
    warmups: int,
) -> tuple[float, list[float]]:
    for _ in range(warmups):
        fn(images)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(images)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2], times


def _make_calculator(name: str, device: torch.device):
    if name == "ani":
        from maple.function.calculator.ani._ani_calculator import ANICalculator

        return ANICalculator(device, model="ani1xnr", implicit="none")
    if name == "aimnet2":
        from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator

        return AIMNet2Calculator(device, model="aimnet2", implicit="none")
    if name == "aimnet2nse":
        from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator

        return AIMNet2Calculator(device, model="aimnet2nse", implicit="none")
    if name == "mace":
        from maple.function.calculator.mace._mace_calculator import MACECalculator

        return MACECalculator(
            device,
            model="maceoff23m",
            model_path=str(MODEL_DIR / "maceoff23m.pt"),
            implicit="none",
        )
    if name == "maceomol":
        from maple.function.calculator.mace._mace_calculator import MACECalculator

        return MACECalculator(
            device,
            model="maceomol",
            model_path=None,
            implicit="none",
        )
    if name == "macepol":
        from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator

        return MACEPolCalculator(
            device,
            model="macepols",
            model_path=str(MODEL_DIR / "macepols.pt"),
            implicit="none",
        )
    if name == "uma":
        from maple.function.calculator.uma._uma_calculator import UMACalculator

        return UMACalculator(
            device,
            model="uma-s-1p1",
            checkpoint_path=str(MODEL_DIR / "uma-s-1p1.pt"),
            implicit="none",
            task="omol",
        )
    if name == "egret":
        from maple.function.calculator.mace._mace_calculator import MACECalculator

        return MACECalculator(
            device,
            model="egret",
            model_path=str(MODEL_DIR / "egret1s.pt"),
            implicit="none",
        )
    raise ValueError(f"unknown backend: {name}")


def _apply_parity_gate(
    results: list[dict],
    *,
    max_energy_diff: float | None,
    max_force_diff: float | None,
) -> list[str]:
    """Annotate benchmark results and return backend names that fail parity."""
    failures = []
    if max_energy_diff is None and max_force_diff is None:
        return failures

    for result in results:
        energy_ok = (
            max_energy_diff is None
            or float(result["max_energy_diff_Eh"]) <= float(max_energy_diff)
        )
        force_ok = (
            max_force_diff is None
            or float(result["max_force_diff_Eh_per_A"]) <= float(max_force_diff)
        )
        result["parity_pass"] = bool(energy_ok and force_ok)
        if not result["parity_pass"]:
            failures.append(str(result["backend"]))
    return failures


def _benchmark_one(
    name: str,
    device: torch.device,
    n_images: int,
    reps: int,
    warmups: int,
) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc = _make_calculator(name, device)
        images = _build_images(calc, n_images)

        legacy_energies, legacy_forces = _legacy_energy_forces(images)
        batch_energies, batch_forces = _path_batch_energy_forces(images)

        max_energy_diff = float(np.max(np.abs(legacy_energies - batch_energies)))
        max_force_diff = float(
            max(
                np.max(np.abs(a - b))
                for a, b in zip(legacy_forces, batch_forces)
            )
        )

        legacy_median, legacy_all = _time_call(_legacy_energy_forces, images, reps, warmups)
        batch_median, batch_all = _time_call(_path_batch_energy_forces, images, reps, warmups)

    return {
        "backend": name,
        "legacy_median_sec": legacy_median,
        "path_batch_median_sec": batch_median,
        "speedup": legacy_median / batch_median if batch_median else None,
        "max_energy_diff_Eh": max_energy_diff,
        "max_force_diff_Eh_per_A": max_force_diff,
        "legacy_all_sec": legacy_all,
        "path_batch_all_sec": batch_all,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        choices=("ani", "aimnet2", "aimnet2nse", "mace", "maceomol", "macepol", "uma", "egret"),
        help="Backend to benchmark. Repeat to select multiple; default is all.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-images", type=int, default=12)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--require-parity",
        action="store_true",
        help=(
            "Fail with a non-zero exit code when path-batch E/F differs from "
            "legacy E/F beyond the configured tolerances. If tolerances are not "
            "provided, defaults are used."
        ),
    )
    parser.add_argument(
        "--max-energy-diff",
        type=float,
        default=None,
        help="Maximum allowed |legacy-batch| energy difference in Hartree.",
    )
    parser.add_argument(
        "--max-force-diff",
        type=float,
        default=None,
        help="Maximum allowed |legacy-batch| force component difference in Hartree/Angstrom.",
    )
    args = parser.parse_args()

    if args.n_images < 2:
        raise ValueError("--n-images must be >= 2")
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")

    device = torch.device(args.device)
    backends = args.backend or ["ani", "aimnet2", "aimnet2nse", "mace", "maceomol", "macepol", "uma", "egret"]
    max_energy_diff = args.max_energy_diff
    max_force_diff = args.max_force_diff
    if args.require_parity:
        if max_energy_diff is None:
            max_energy_diff = DEFAULT_MAX_ENERGY_DIFF_EH
        if max_force_diff is None:
            max_force_diff = DEFAULT_MAX_FORCE_DIFF_EH_PER_A

    # Keep stdout machine-readable.  Some backend loaders print optional-kernel
    # notices (for example cuequivariance availability) to stdout; route those
    # notices to stderr so callers can safely redirect stdout to a JSON file.
    with contextlib.redirect_stdout(sys.stderr):
        results = [
            _benchmark_one(name, device, args.n_images, args.reps, args.warmups)
            for name in backends
        ]
    failures = _apply_parity_gate(
        results,
        max_energy_diff=max_energy_diff,
        max_force_diff=max_force_diff,
    )
    payload = {
        "device": str(device),
        "n_images": args.n_images,
        "parity_thresholds": {
            "max_energy_diff_Eh": max_energy_diff,
            "max_force_diff_Eh_per_A": max_force_diff,
        },
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if failures:
        print(
            "Path-batch parity failed for backend(s): " + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
