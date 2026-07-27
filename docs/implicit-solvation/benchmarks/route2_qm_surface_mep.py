#!/usr/bin/env python3
"""Evaluate a frozen PySCF density MEP on externally supplied points.

This helper intentionally depends only on NumPy and PySCF.  It is launched by
the Route-2 GTO/PCM projection canary in the isolated QM-reference
environment; it does not add PySCF to MAPLE's runtime dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyscf
from pyscf import lib

BOHR_TO_ANGSTROM = 0.529177210903


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _closed_shell_density_from_checkpoint(
    checkpoint: Path,
    molecule,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Rebuild the AO density from the orbitals stored in one checkpoint."""

    mo_coeff = np.asarray(
        lib.chkfile.load(str(checkpoint), "scf/mo_coeff"),
        dtype=float,
    )
    mo_occ = np.asarray(
        lib.chkfile.load(str(checkpoint), "scf/mo_occ"),
        dtype=float,
    )
    nao = molecule.nao_nr()
    if (
        mo_coeff.ndim != 2
        or mo_coeff.shape[0] != nao
        or mo_occ.ndim != 1
        or mo_coeff.shape[1] != mo_occ.size
        or not np.all(np.isfinite(mo_coeff))
        or not np.all(np.isfinite(mo_occ))
    ):
        raise ValueError(
            "The checkpoint must contain one finite restricted-orbital "
            "coefficient matrix and one occupation vector."
        )
    occupation_distance = np.minimum(np.abs(mo_occ), np.abs(mo_occ - 2.0))
    if float(np.max(occupation_distance, initial=0.0)) > 1.0e-8:
        raise ValueError(
            "The checkpoint occupations must describe one integer-occupied "
            "closed-shell state."
        )

    overlap = molecule.intor_symmetric("int1e_ovlp")
    orthonormality_error = float(
        np.max(
            np.abs(
                mo_coeff.T @ overlap @ mo_coeff - np.eye(mo_coeff.shape[1], dtype=float)
            )
        )
    )
    if orthonormality_error > 1.0e-7:
        raise ValueError(
            "The checkpoint orbitals are not orthonormal in the checkpoint "
            "AO metric."
        )

    density = (mo_coeff * mo_occ[None, :]) @ mo_coeff.T
    density = 0.5 * (density + density.T)
    electron_count = float(np.einsum("ij,ji->", density, overlap))
    occupation_sum = float(np.sum(mo_occ))
    binding_error = abs(electron_count - occupation_sum)
    if binding_error > 1.0e-7:
        raise ValueError(
            "The checkpoint-derived AO density is inconsistent with its "
            "orbital occupations."
        )
    return density, {
        "density_source": "checkpoint:scf/mo_coeff+scf/mo_occ",
        "checkpoint_density_binding_residual_e": binding_error,
        "checkpoint_mo_orthonormality_inf": orthonormality_error,
        "checkpoint_occupation_sum_e": occupation_sum,
        "checkpoint_occupied_orbital_count": int(np.count_nonzero(mo_occ > 1.0)),
    }


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    molecule = lib.chkfile.load_mol(str(args.checkpoint))
    density, density_provenance = _closed_shell_density_from_checkpoint(
        args.checkpoint,
        molecule,
    )
    surface = np.load(args.surface)
    points_bohr = np.asarray(surface["surface_points_bohr"], dtype=float)
    if (
        points_bohr.ndim != 2
        or points_bohr.shape[0] == 0
        or points_bohr.shape[1] != 3
        or not np.all(np.isfinite(points_bohr))
    ):
        raise ValueError(
            "The external surface must contain finite points with shape "
            "(n_surface, 3) in bohr."
        )

    coordinates_bohr = np.asarray(molecule.atom_coords(), dtype=float)
    atomic_numbers = np.asarray(molecule.atom_charges(), dtype=float)
    potential = np.empty(len(points_bohr), dtype=float)
    for index, point in enumerate(points_bohr):
        distances = np.linalg.norm(coordinates_bohr - point, axis=1)
        if np.any(distances <= 1.0e-12):
            raise ValueError("A surface point coincides with a QM nucleus.")
        nuclear = float(np.dot(atomic_numbers, 1.0 / distances))
        with molecule.with_rinv_origin(point):
            inverse_distance = molecule.intor("int1e_rinv")
        electronic = float(np.einsum("ij,ji->", density, inverse_distance))
        potential[index] = nuclear - electronic

    overlap = molecule.intor_symmetric("int1e_ovlp")
    electron_count = float(np.einsum("ij,ji->", density, overlap))
    total_charge = float(np.sum(atomic_numbers) - electron_count)
    position_integrals = molecule.intor_symmetric("int1e_r", comp=3)
    dipole_bohr = np.einsum("i,ix->x", atomic_numbers, coordinates_bohr) - np.einsum(
        "xij,ji->x", position_integrals, density
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        surface_potential_hartree_per_e=potential,
        total_charge_e=np.asarray(total_charge),
        molecular_dipole_e_angstrom=dipole_bohr * BOHR_TO_ANGSTROM,
        atomic_numbers=atomic_numbers,
        atom_positions_angstrom=coordinates_bohr * BOHR_TO_ANGSTROM,
        pyscf_version=np.asarray(str(pyscf.__version__)),
        **{name: np.asarray(value) for name, value in density_provenance.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
