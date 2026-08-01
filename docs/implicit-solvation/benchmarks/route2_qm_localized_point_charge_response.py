#!/usr/bin/env python3
"""Independent QM response to frozen exterior point-charge perturbations.

This helper imports no MAPLE package.  It starts from a hash-bound gas-phase
PySCF checkpoint, applies central finite differences in the amplitudes of
geometry-selected exterior point charges, and evaluates the induced electronic
MEP on the already frozen exterior validation shell.  Fixed nuclei cancel from
both the induced MEP and dipole response.
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
from pyscf import dft, lib

from route2_qm_surface_mep import _closed_shell_density_from_checkpoint


FIELD_STEPS_E = (3.0e-4, 1.0e-3)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--modes", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=8000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _load_points(path: Path, *, key: str, label: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as state:
        if key not in state:
            raise RuntimeError(f"{label} omits {key}.")
        points = np.asarray(state[key], dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise RuntimeError(f"{label} must contain finite points with shape (n, 3).")
    return points


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
        mean_field.nlcgrids.atom_grid = {symbol: (50, 194) for symbol in elements}
        mean_field.nlcgrids.prune = dft.gen_grid.sg1_prune
    else:
        mean_field.grids, mean_field.nlcgrids = shared_grids
    mean_field.conv_tol = QM_METHOD["scf_energy_tolerance_hartree"]
    mean_field.conv_tol_grad = QM_METHOD["scf_gradient_tolerance"]
    mean_field.max_cycle = QM_METHOD["maximum_scf_cycles"]
    mean_field.max_memory = max_memory_mb
    return mean_field


def _coulomb_integrals(molecule, points_bohr: np.ndarray) -> np.ndarray:
    values = np.empty(
        (len(points_bohr), molecule.nao_nr(), molecule.nao_nr()),
        dtype=float,
    )
    for index, point in enumerate(points_bohr):
        with molecule.with_rinv_origin(point):
            values[index] = molecule.intor("int1e_rinv")
    return values


def _electronic_potential(
    density_matrix: np.ndarray,
    coulomb_integrals: np.ndarray,
) -> np.ndarray:
    return -np.einsum(
        "sij,ji->s",
        coulomb_integrals,
        density_matrix,
        optimize=True,
    ).real


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
    electronic = np.einsum(
        "xij,ji->x",
        dipole_integrals,
        density_matrix,
        optimize=True,
    ).real
    return nuclear - electronic


def _run_perturbation(
    molecule,
    *,
    base_hcore: np.ndarray,
    zero_density: np.ndarray,
    source_integral: np.ndarray,
    amplitude_e: float,
    dipole_integrals: np.ndarray,
    surface_integrals: np.ndarray,
    overlap: np.ndarray,
    expected_electron_count: float,
    max_memory_mb: int,
    shared_grids: tuple[object, object],
) -> dict[str, object]:
    mean_field = _make_rks(
        molecule,
        max_memory_mb=max_memory_mb,
        shared_grids=shared_grids,
    )
    # A positive external point charge has scalar potential +q/|r-s|.
    # The one-electron Hamiltonian of an electron therefore receives -q/r.
    hcore = base_hcore - amplitude_e * source_integral
    mean_field.get_hcore = lambda *unused, matrix=hcore: matrix
    cycle_count = 0

    def count_cycle(_environment):
        nonlocal cycle_count
        cycle_count += 1

    mean_field.callback = count_cycle
    started = time.perf_counter()
    energy = float(mean_field.kernel(dm0=zero_density))
    elapsed = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError(
            f"Localized finite-field SCF did not converge for amplitude {amplitude_e}."
        )
    density = np.asarray(mean_field.make_rdm1(), dtype=float)
    electron_count = float(np.einsum("ij,ji->", density, overlap))
    electron_count_error = abs(electron_count - expected_electron_count)
    if electron_count_error > 1.0e-7:
        raise RuntimeError("Localized finite-field SCF changed the electron count.")
    return {
        "density": density,
        "dipole_e_bohr": _total_dipole_e_bohr(
            molecule,
            density,
            dipole_integrals,
        ),
        "electronic_surface_potential_hartree_per_e": _electronic_potential(
            density,
            surface_integrals,
        ),
        "energy_hartree": energy,
        "scf_cycles": cycle_count,
        "elapsed_seconds": elapsed,
        "electron_count_error_e": electron_count_error,
    }


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    surface_path = args.surface.resolve()
    modes_path = args.modes.resolve()
    output_json = args.output_json.resolve()
    output_npz = args.output_npz.resolve()
    if output_json.exists() or output_npz.exists():
        raise FileExistsError(output_json if output_json.exists() else output_npz)
    if args.threads != 8 or args.max_memory_mb != 8000:
        raise RuntimeError("Localized QM response requires the frozen 8-thread/8-GB runtime.")
    if pyscf.__version__ != "2.13.1":
        raise RuntimeError("Localized QM response requires PySCF 2.13.1.")

    molecule = lib.chkfile.load_mol(str(checkpoint))
    if molecule.charge != 0 or molecule.spin != 0:
        raise RuntimeError("Localized QM response currently accepts neutral singlets only.")
    zero_density, density_provenance = _closed_shell_density_from_checkpoint(
        checkpoint,
        molecule,
    )
    surface_points = _load_points(
        surface_path,
        key="surface_points_bohr",
        label="Frozen validation surface",
    )
    source_points = _load_points(
        modes_path,
        key="source_points_bohr",
        label="Frozen localized source modes",
    )
    if len(source_points) != 4:
        raise RuntimeError("Localized response protocol requires exactly four modes.")

    lib.num_threads(args.threads)
    mean_field = _make_rks(molecule, max_memory_mb=args.max_memory_mb)
    # The molecular geometry never changes.  Build both integration grids once
    # and share their immutable coordinates/weights across every +/- response
    # solve, matching the established acetone finite-field helper.
    mean_field.grids.build()
    mean_field.nlcgrids.build()
    shared_grids = (mean_field.grids, mean_field.nlcgrids)
    base_hcore = np.asarray(mean_field.get_hcore(), dtype=float)
    overlap = molecule.intor_symmetric("int1e_ovlp")
    expected_electron_count = float(np.einsum("ij,ji->", zero_density, overlap))
    dipole_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3),
        dtype=float,
    )
    surface_integrals = _coulomb_integrals(molecule, surface_points)
    source_integrals = _coulomb_integrals(molecule, source_points)

    response_mep = np.empty(
        (len(FIELD_STEPS_E), len(source_points), len(surface_points)),
        dtype=float,
    )
    response_dipole = np.empty(
        (len(FIELD_STEPS_E), len(source_points), 3),
        dtype=float,
    )
    records: list[dict[str, object]] = []
    total_started = time.perf_counter()
    for step_index, step in enumerate(FIELD_STEPS_E):
        for mode_index, source_integral in enumerate(source_integrals):
            states = {}
            for sign in (-1, 1):
                state = _run_perturbation(
                    molecule,
                    base_hcore=base_hcore,
                    zero_density=zero_density,
                    source_integral=source_integral,
                    amplitude_e=sign * step,
                    dipole_integrals=dipole_integrals,
                    surface_integrals=surface_integrals,
                    overlap=overlap,
                    expected_electron_count=expected_electron_count,
                    max_memory_mb=args.max_memory_mb,
                    shared_grids=shared_grids,
                )
                states[sign] = state
                records.append(
                    {
                        "amplitude_e": sign * step,
                        "density_sha256": _sha256_array(state["density"]),
                        "dipole_e_bohr": state["dipole_e_bohr"].tolist(),
                        "electron_count_error_e": state["electron_count_error_e"],
                        "elapsed_seconds": state["elapsed_seconds"],
                        "energy_hartree": state["energy_hartree"],
                        "mode_index": mode_index,
                        "scf_cycles": state["scf_cycles"],
                        "sign": sign,
                        "step_e": step,
                    }
                )
            scale = 1.0 / (2.0 * step)
            response_mep[step_index, mode_index] = scale * (
                states[1]["electronic_surface_potential_hartree_per_e"]
                - states[-1]["electronic_surface_potential_hartree_per_e"]
            )
            response_dipole[step_index, mode_index] = scale * (
                states[1]["dipole_e_bohr"] - states[-1]["dipole_e_bohr"]
            )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("xb") as handle:
        np.savez(
            handle,
            field_steps_e=np.asarray(FIELD_STEPS_E),
            source_points_bohr=source_points,
            surface_points_bohr=surface_points,
            induced_surface_mep_hartree_per_e_per_source_e=response_mep,
            induced_dipole_e_bohr_per_source_e=response_dipole,
        )
    result = {
        "artifact": "route2-qm-localized-point-charge-response-v1",
        "schema_version": 1,
        "status": "success",
        "finite_field_protocol": {
            "field_steps_e": list(FIELD_STEPS_E),
            "mode_count": len(source_points),
            "perturbation": (
                "external scalar potential q/|r-s_k| with zero at infinity; "
                "electronic one-body perturbation -q int1e_rinv(s_k)"
            ),
            "signs": [-1, 1],
        },
        "input": {
            "checkpoint_sha256": _sha256(checkpoint),
            "modes_npz_sha256": _sha256(modes_path),
            "source_points_bohr_sha256": _sha256_array(source_points),
            "surface_npz_sha256": _sha256(surface_path),
            "surface_points_bohr_sha256": _sha256_array(surface_points),
        },
        "method": QM_METHOD,
        "output": {
            "npz_sha256": _sha256(output_npz),
            "response_dipole_sha256": _sha256_array(response_dipole),
            "response_mep_sha256": _sha256_array(response_mep),
        },
        "perturbation_records": records,
        "runtime": {
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "threads": lib.num_threads(),
            "total_elapsed_seconds": time.perf_counter() - total_started,
        },
        "zero_density": {
            **density_provenance,
            "density_sha256": _sha256_array(zero_density),
            "electron_count_e": expected_electron_count,
        },
    }
    _write_exclusive_json(output_json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
