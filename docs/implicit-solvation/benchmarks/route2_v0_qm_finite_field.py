#!/usr/bin/env python3
"""Isolated PySCF finite-field helper for the acetone V0-Q falsifier.

This helper intentionally imports no MAPLE package.  It runs in the pinned
PySCF environment, evaluates one zero-field state and two central-difference
field strengths along each Cartesian axis, and writes only raw QM energies and
dipoles.  The source-bound parent runner owns all QEq construction, numerical
checks, and scientific interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pyscf
from pyscf import dft, gto, lib


EXPECTED_SYMBOLS = ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H")
FIELD_STEPS_AU = (3.0e-4, 1.0e-3)
DIRECTIONS = ("x", "y", "z")
QM_METHOD = {
    "electronic_structure": "omegaB97M-V",
    "pyscf_xc_token": "wb97m-v",
    "basis": "def2-tzvpd",
    "reference": "RKS",
    "density_fitting": True,
    "charge": 0,
    "spin": 0,
    "semilocal_grid_level": 3,
    "nonlocal_grid_profile": "50x194-SG1",
    "scf_energy_tolerance_hartree": 1.0e-10,
    "scf_gradient_tolerance": 1.0e-7,
    "maximum_scf_cycles": 100,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--max-memory-mb", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_geometry(path: Path) -> tuple[np.ndarray, list[tuple[str, tuple[float, ...]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = tuple(payload.get("elements", ()))
    positions = np.asarray(payload.get("positions_angstrom"), dtype=float)
    if symbols != EXPECTED_SYMBOLS or positions.shape != (10, 3):
        raise RuntimeError("The finite-field helper accepts only the frozen acetone geometry.")
    if not np.all(np.isfinite(positions)):
        raise RuntimeError("The finite-field geometry contains nonfinite values.")
    atoms = [
        (symbol, tuple(float(value) for value in position))
        for symbol, position in zip(symbols, positions)
    ]
    return positions, atoms


def _make_rks(
    molecule,
    *,
    max_memory_mb: int,
    shared_grids: tuple[object, object] | None = None,
):
    mean_field = dft.RKS(molecule, xc=QM_METHOD["pyscf_xc_token"]).density_fit()
    if shared_grids is None:
        mean_field.grids.level = QM_METHOD["semilocal_grid_level"]
        elements = sorted(
            {molecule.atom_symbol(index) for index in range(molecule.natm)}
        )
        mean_field.nlcgrids.atom_grid = {
            symbol: (50, 194) for symbol in elements
        }
        mean_field.nlcgrids.prune = dft.gen_grid.sg1_prune
    else:
        mean_field.grids, mean_field.nlcgrids = shared_grids
    mean_field.conv_tol = QM_METHOD["scf_energy_tolerance_hartree"]
    mean_field.conv_tol_grad = QM_METHOD["scf_gradient_tolerance"]
    mean_field.max_cycle = QM_METHOD["maximum_scf_cycles"]
    mean_field.max_memory = max_memory_mb
    return mean_field


def _total_dipole_e_bohr(
    molecule,
    density_matrix: np.ndarray,
    dipole_integrals: np.ndarray,
) -> np.ndarray:
    nuclear = np.einsum(
        "i,ix->x",
        molecule.atom_charges(),
        molecule.atom_coords(),
    )
    electronic = np.einsum("xij,ji->x", dipole_integrals, density_matrix).real
    return nuclear - electronic


def _run_field_scf(
    molecule,
    *,
    field_au: np.ndarray,
    zero_density: np.ndarray,
    base_hcore: np.ndarray,
    dipole_integrals: np.ndarray,
    shared_grids: tuple[object, object],
    max_memory_mb: int,
) -> dict[str, object]:
    field = np.asarray(field_au, dtype=float)
    mean_field = _make_rks(
        molecule,
        max_memory_mb=max_memory_mb,
        shared_grids=shared_grids,
    )
    field_hcore = base_hcore + np.einsum("x,xij->ij", field, dipole_integrals)
    mean_field.get_hcore = lambda *unused, matrix=field_hcore: matrix
    cycle_count = 0

    def count_cycle(_environment):
        nonlocal cycle_count
        cycle_count += 1

    mean_field.callback = count_cycle
    started = time.perf_counter()
    raw_energy = float(mean_field.kernel(dm0=zero_density))
    elapsed = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError(f"Finite-field SCF did not converge for field {field}.")
    density = np.asarray(mean_field.make_rdm1(), dtype=float)
    nuclear_dipole = np.einsum(
        "i,ix->x",
        molecule.atom_charges(),
        molecule.atom_coords(),
    )
    return {
        "field_au": field.tolist(),
        "energy_hartree": raw_energy - float(field @ nuclear_dipole),
        "dipole_e_bohr": _total_dipole_e_bohr(
            molecule,
            density,
            dipole_integrals,
        ).tolist(),
        "scf_cycles": cycle_count,
        "elapsed_seconds": elapsed,
    }


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def main() -> int:
    args = _parse_args()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if output.exists():
        raise FileExistsError(output)
    if args.threads != 8 or args.max_memory_mb != 8000:
        raise RuntimeError("The finite-field helper accepts only the frozen runtime.")
    if pyscf.__version__ != "2.13.1":
        raise RuntimeError("The finite-field helper requires PySCF 2.13.1.")
    positions, atoms = _load_geometry(manifest)
    lib.num_threads(args.threads)
    molecule = gto.M(
        atom=atoms,
        basis=QM_METHOD["basis"],
        charge=QM_METHOD["charge"],
        spin=QM_METHOD["spin"],
        unit="Angstrom",
        symmetry=False,
        verbose=0,
        max_memory=args.max_memory_mb,
    )
    mean_field = _make_rks(molecule, max_memory_mb=args.max_memory_mb)
    zero_cycles = 0

    def count_zero_cycle(_environment):
        nonlocal zero_cycles
        zero_cycles += 1

    mean_field.callback = count_zero_cycle
    total_started = time.perf_counter()
    zero_started = time.perf_counter()
    zero_energy = float(mean_field.kernel())
    zero_elapsed = time.perf_counter() - zero_started
    if not mean_field.converged:
        raise RuntimeError("Zero-field finite-field reference did not converge.")
    zero_density = np.asarray(mean_field.make_rdm1(), dtype=float)
    base_hcore = np.asarray(mean_field.get_hcore(), dtype=float)
    dipole_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3),
        dtype=float,
    )
    zero_dipole = _total_dipole_e_bohr(
        molecule,
        zero_density,
        dipole_integrals,
    )
    shared_grids = (mean_field.grids, mean_field.nlcgrids)

    field_records: list[dict[str, object]] = []
    for step in FIELD_STEPS_AU:
        for direction, direction_name in enumerate(DIRECTIONS):
            for sign in (-1, 1):
                field = np.zeros(3)
                field[direction] = sign * step
                record = _run_field_scf(
                    molecule,
                    field_au=field,
                    zero_density=zero_density,
                    base_hcore=base_hcore,
                    dipole_integrals=dipole_integrals,
                    shared_grids=shared_grids,
                    max_memory_mb=args.max_memory_mb,
                )
                record.update(
                    {
                        "step_au": step,
                        "direction": direction_name,
                        "sign": sign,
                    }
                )
                field_records.append(record)

    result = {
        "schema_version": 1,
        "status": "pass",
        "method": QM_METHOD,
        "finite_field_protocol": {
            "field_steps_au": list(FIELD_STEPS_AU),
            "directions": list(DIRECTIONS),
            "signs": [-1, 1],
        },
        "input": {
            "manifest_path": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "positions_angstrom": positions.tolist(),
        },
        "zero_field": {
            "energy_hartree": zero_energy,
            "dipole_e_bohr": zero_dipole.tolist(),
            "density_sha256": hashlib.sha256(
                np.ascontiguousarray(zero_density).view(np.uint8)
            ).hexdigest(),
            "scf_cycles": zero_cycles,
            "elapsed_seconds": zero_elapsed,
            "semilocal_grid_point_count": int(mean_field.grids.coords.shape[0]),
            "nonlocal_grid_point_count": int(mean_field.nlcgrids.coords.shape[0]),
        },
        "field_records": field_records,
        "runtime": {
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_resolved_sha256": _sha256(Path(sys.executable).resolve()),
            "pyscf": pyscf.__version__,
            "pyscf_init_path": str(Path(pyscf.__file__).resolve()),
            "pyscf_init_sha256": _sha256(Path(pyscf.__file__).resolve()),
            "numpy": np.__version__,
            "threads": lib.num_threads(),
            "total_elapsed_seconds": time.perf_counter() - total_started,
        },
    }
    _write_exclusive_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
