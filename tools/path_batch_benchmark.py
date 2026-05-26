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

from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
from maple.function.calculator.ani._ani_calculator import ANICalculator
from maple.function.calculator.mace._mace_calculator import MACECalculator
from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator
from maple.function.calculator.uma._uma_calculator import UMACalculator
from maple.function.dispatcher.ts.algorithm.neb import NEB


MODEL_DIR = Path("maple/function/calculator/model")

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
        return ANICalculator(device, model="ani1xnr", implicit="none")
    if name == "aimnet2":
        return AIMNet2Calculator(device, model="aimnet2", implicit="none")
    if name == "aimnet2nse":
        return AIMNet2Calculator(device, model="aimnet2nse", implicit="none")
    if name == "mace":
        return MACECalculator(
            device,
            model="maceoff23m",
            model_path=str(MODEL_DIR / "maceoff23m.pt"),
            implicit="none",
        )
    if name == "maceomol":
        return MACECalculator(
            device,
            model="maceomol",
            model_path=None,
            implicit="none",
        )
    if name == "macepol":
        return MACEPolCalculator(
            device,
            model="macepols",
            model_path=str(MODEL_DIR / "macepols.pt"),
            implicit="none",
        )
    if name == "uma":
        return UMACalculator(
            device,
            model="uma-s-1p1",
            checkpoint_path=str(MODEL_DIR / "uma-s-1p1.pt"),
            implicit="none",
            task="omol",
        )
    if name == "egret":
        return MACECalculator(
            device,
            model="egret",
            model_path=str(MODEL_DIR / "egret1s.pt"),
            implicit="none",
        )
    raise ValueError(f"unknown backend: {name}")


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
    args = parser.parse_args()

    if args.n_images < 2:
        raise ValueError("--n-images must be >= 2")
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")

    device = torch.device(args.device)
    backends = args.backend or ["ani", "aimnet2", "aimnet2nse", "mace", "maceomol", "macepol", "uma", "egret"]
    # Keep stdout machine-readable.  Some backend loaders print optional-kernel
    # notices (for example cuequivariance availability) to stdout; route those
    # notices to stderr so callers can safely redirect stdout to a JSON file.
    with contextlib.redirect_stdout(sys.stderr):
        results = [
            _benchmark_one(name, device, args.n_images, args.reps, args.warmups)
            for name in backends
        ]
    print(json.dumps({"device": str(device), "n_images": args.n_images, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
