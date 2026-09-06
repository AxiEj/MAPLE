#!/usr/bin/env python3
"""Generate the frozen spherical iodine densities used by the ADT audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pyscf import dft, gto, lib
from pyscf.scf.atom_hf import AtomSphAverageRHF


CASES = {
    "iodine_def2_ecp": {
        "basis": "def2-tzvpd",
        "ecp": "def2-tzvpd",
        "x2c": False,
        "filename": "iodine_def2_ecp_atomic_density.json",
    },
    "iodine_ano_nr": {
        "basis": "ano-rcc",
        "ecp": None,
        "x2c": False,
        "filename": "iodine_ano_nr_atomic_density.json",
    },
    "iodine_ano_x2c": {
        "basis": "ano-rcc",
        "ecp": None,
        "x2c": True,
        "filename": "iodine_ano_x2c_atomic_density.json",
    },
}


def _radial_grid() -> np.ndarray:
    return np.concatenate(
        [
            np.linspace(0.0, 0.5, 251),
            np.linspace(0.502, 4.0, 1750),
            np.linspace(4.01, 15.0, 1100),
        ]
    )


def generate_case(label: str) -> dict[str, object]:
    case = CASES[label]
    started = time.monotonic()
    molecule = gto.M(
        atom="I 0 0 0",
        basis=case["basis"],
        ecp=case["ecp"] or {},
        spin=1,
        verbose=3,
    )
    mean_field = AtomSphAverageRHF(molecule)
    if case["x2c"]:
        mean_field = mean_field.x2c()
    mean_field.conv_tol = 1.0e-11
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError(f"Atomic SCF did not converge for {label}.")

    radius = _radial_grid()
    coordinates = np.zeros((len(radius), 3), dtype=np.float64)
    coordinates[:, 0] = radius
    density_matrix = mean_field.make_rdm1()
    density = np.asarray(
        dft.numint.eval_rho(
            molecule,
            molecule.eval_gto("GTOval_sph", coordinates),
            density_matrix,
        )
    )
    # Preserve the original multiplication/reduction order.  Reassociating
    # this expression changes the final bit on some BLAS/libm combinations and
    # therefore changes the content-addressed raw audit file.
    integral = float(4.0 * np.pi * np.trapezoid(radius**2 * density, radius))
    shell = 4.0 * np.pi * radius**2 * density
    enclosed = np.zeros_like(radius)
    enclosed[1:] = np.cumsum(
        0.5 * (shell[1:] + shell[:-1]) * np.diff(radius)
    )
    print(
        "DONE",
        label,
        "energy",
        energy,
        "electrons",
        molecule.nelectron,
        "nao",
        molecule.nao_nr(),
        "seconds",
        time.monotonic() - started,
        flush=True,
    )
    # Preserve the original insertion order and compact JSON serialization;
    # both are part of the frozen raw-file SHA256 evidence.
    return {
        "label": label,
        "basis": case["basis"],
        "ecp": case["ecp"],
        "x2c": case["x2c"],
        "energy": energy,
        "nelec": molecule.nelectron,
        "nao": molecule.nao_nr(),
        "integral": integral,
        "radii": radius.tolist(),
        "rho": density.tolist(),
        "enclosed_fraction": (enclosed / integral).tolist(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--label", choices=tuple(CASES), action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = tuple(args.label) if args.label else tuple(CASES)
    lib.num_threads(1)
    for label in labels:
        path = output_dir / str(CASES[label]["filename"])
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}.")
        path.write_text(json.dumps(generate_case(label)), encoding="utf-8")


if __name__ == "__main__":
    main()
