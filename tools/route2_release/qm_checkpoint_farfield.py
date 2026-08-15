#!/usr/bin/env python3
"""Evaluate exact checkpoint multipoles and deterministic far-field MEP shells.

This helper is deliberately independent of MAPLE and MACE.  It is executed in
the hash-pinned PySCF reference environment and reads an already frozen
closed-shell checkpoint; it performs no new SCF calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import lib

BOHR_TO_ANGSTROM = 0.529177210903
SCHEMA_VERSION = "route2-qm-checkpoint-far-field-reference-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--polar-order", type=int, required=True)
    parser.add_argument(
        "--shell-offsets-angstrom", type=float, nargs="+", required=True
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _closed_shell_density(checkpoint: Path, molecule) -> tuple[np.ndarray, dict]:
    coefficient = np.asarray(
        lib.chkfile.load(str(checkpoint), "scf/mo_coeff"), dtype=float
    )
    occupation = np.asarray(
        lib.chkfile.load(str(checkpoint), "scf/mo_occ"), dtype=float
    )
    if (
        coefficient.ndim != 2
        or coefficient.shape[0] != molecule.nao_nr()
        or occupation.shape != (coefficient.shape[1],)
        or not np.all(np.isfinite(coefficient))
        or not np.all(np.isfinite(occupation))
    ):
        raise RuntimeError("checkpoint orbitals are invalid.")
    occupation_distance = np.minimum(np.abs(occupation), np.abs(occupation - 2.0))
    if float(np.max(occupation_distance, initial=0.0)) > 1.0e-8:
        raise RuntimeError("checkpoint is not an integer-occupied closed shell.")
    overlap = molecule.intor_symmetric("int1e_ovlp")
    orthonormality_error = float(
        np.max(
            np.abs(coefficient.T @ overlap @ coefficient - np.eye(coefficient.shape[1]))
        )
    )
    if orthonormality_error > 1.0e-7:
        raise RuntimeError("checkpoint orbitals fail the AO-metric identity.")
    density = (coefficient * occupation[None, :]) @ coefficient.T
    density = 0.5 * (density + density.T)
    electron_count = float(np.einsum("ij,ji->", density, overlap))
    occupation_sum = float(np.sum(occupation))
    binding_residual = abs(electron_count - occupation_sum)
    if binding_residual > 1.0e-7:
        raise RuntimeError("checkpoint density electron count is inconsistent.")
    return density, {
        "electron_count_e": electron_count,
        "occupation_sum_e": occupation_sum,
        "density_binding_residual_e": binding_residual,
        "mo_orthonormality_inf": orthonormality_error,
        "density_sha256": _array_sha256(density),
    }


def _sphere_rule(polar_order: int) -> tuple[np.ndarray, np.ndarray]:
    if polar_order < 2 or polar_order > 64:
        raise ValueError("polar_order must be in [2,64].")
    cosine, polar_weights = np.polynomial.legendre.leggauss(polar_order)
    azimuthal_count = 2 * polar_order + 1
    azimuth = 2.0 * np.pi * np.arange(azimuthal_count) / azimuthal_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    directions = np.stack(
        (
            np.repeat(sine, azimuthal_count) * np.tile(np.cos(azimuth), polar_order),
            np.repeat(sine, azimuthal_count) * np.tile(np.sin(azimuth), polar_order),
            np.repeat(cosine, azimuthal_count),
        ),
        axis=1,
    )
    weights = np.repeat(
        polar_weights * (2.0 * np.pi / azimuthal_count), azimuthal_count
    )
    return directions, weights


def _multipoles(
    molecule, density: np.ndarray, origin_bohr: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    charges = np.asarray(molecule.atom_charges(), dtype=float)
    coordinates = np.asarray(molecule.atom_coords(), dtype=float)
    relative = coordinates - origin_bohr[None, :]
    overlap = molecule.intor_symmetric("int1e_ovlp")
    electron_count = float(np.einsum("ij,ji->", density, overlap))
    total_charge = float(np.sum(charges) - electron_count)

    first_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3), dtype=float
    )
    electronic_first_absolute = np.einsum("xij,ji->x", first_integrals, density)
    electronic_first_relative = electronic_first_absolute - origin_bohr * electron_count
    dipole = np.einsum("a,ax->x", charges, relative) - electronic_first_relative

    second_integrals = np.asarray(
        molecule.intor_symmetric("int1e_rr", comp=9), dtype=float
    ).reshape(3, 3, molecule.nao_nr(), molecule.nao_nr())
    electronic_second_absolute = np.einsum("xyij,ji->xy", second_integrals, density)
    electronic_second_relative = (
        electronic_second_absolute
        - np.outer(origin_bohr, electronic_first_absolute)
        - np.outer(electronic_first_absolute, origin_bohr)
        + electron_count * np.outer(origin_bohr, origin_bohr)
    )
    nuclear_second = np.einsum("a,ax,ay->xy", charges, relative, relative)
    charge_second = nuclear_second - electronic_second_relative
    quadrupole = 3.0 * charge_second - np.trace(charge_second) * np.eye(3)
    quadrupole = 0.5 * (quadrupole + quadrupole.T)
    if abs(float(np.trace(quadrupole))) > 2.0e-9:
        raise RuntimeError("traceless quadrupole construction is inconsistent.")
    return total_charge, dipole, quadrupole


def _potential(molecule, density: np.ndarray, points_bohr: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(molecule.atom_coords(), dtype=float)
    charges = np.asarray(molecule.atom_charges(), dtype=float)
    result = np.empty(len(points_bohr), dtype=float)
    for index, point in enumerate(points_bohr):
        distances = np.linalg.norm(coordinates - point, axis=1)
        if np.any(distances <= 1.0e-12):
            raise RuntimeError("far-field point coincides with a nucleus.")
        nuclear = float(np.dot(charges, 1.0 / distances))
        with molecule.with_rinv_origin(point):
            inverse_distance = molecule.intor("int1e_rinv")
        electronic = float(np.einsum("ij,ji->", density, inverse_distance))
        result[index] = nuclear - electronic
    return result


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    offsets = np.asarray(args.shell_offsets_angstrom, dtype=float)
    if (
        offsets.ndim != 1
        or offsets.size < 1
        or not np.all(np.isfinite(offsets))
        or np.any(offsets <= 0.0)
        or np.any(np.diff(offsets) <= 0.0)
    ):
        raise ValueError("shell offsets must be finite, positive, and increasing.")

    molecule = lib.chkfile.load_mol(str(checkpoint))
    density, density_record = _closed_shell_density(checkpoint, molecule)
    charges = np.asarray(molecule.atom_charges(), dtype=float)
    coordinates_bohr = np.asarray(molecule.atom_coords(), dtype=float)
    origin_bohr = np.einsum("a,ax->x", charges, coordinates_bohr) / float(
        np.sum(charges)
    )
    relative_angstrom = (coordinates_bohr - origin_bohr[None, :]) * BOHR_TO_ANGSTROM
    maximum_extent_angstrom = float(np.max(np.linalg.norm(relative_angstrom, axis=1)))
    shell_radii_angstrom = maximum_extent_angstrom + offsets
    directions, angular_weights = _sphere_rule(args.polar_order)
    shell_points_angstrom = [
        origin_bohr[None, :] * BOHR_TO_ANGSTROM + radius * directions
        for radius in shell_radii_angstrom
    ]
    points_angstrom = np.concatenate(shell_points_angstrom, axis=0)
    points_bohr = points_angstrom / BOHR_TO_ANGSTROM
    repeated_weights = np.tile(angular_weights, len(shell_radii_angstrom))
    shell_point_counts = np.full(len(shell_radii_angstrom), len(directions), dtype=int)

    total_charge, dipole_bohr, quadrupole_bohr2 = _multipoles(
        molecule, density, origin_bohr
    )
    potential = _potential(molecule, density, points_bohr)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "pyscf_version": str(pyscf.__version__),
        "polar_order": int(args.polar_order),
        "azimuthal_count": int(2 * args.polar_order + 1),
        "shell_offsets_angstrom": offsets.tolist(),
        "maximum_nuclear_extent_angstrom": maximum_extent_angstrom,
        "density_record_json": json.dumps(
            density_record, sort_keys=True, separators=(",", ":")
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **{name: np.asarray(value) for name, value in payload.items()},
        atomic_numbers=charges,
        atom_positions_angstrom=coordinates_bohr * BOHR_TO_ANGSTROM,
        multipole_origin_angstrom=origin_bohr * BOHR_TO_ANGSTROM,
        total_charge_e=np.asarray(total_charge),
        molecular_dipole_e_angstrom=dipole_bohr * BOHR_TO_ANGSTROM,
        traceless_quadrupole_e_angstrom2=(quadrupole_bohr2 * BOHR_TO_ANGSTROM**2),
        evaluation_points_bohr=points_bohr,
        angular_weights=repeated_weights,
        shell_radii_angstrom=shell_radii_angstrom,
        shell_point_counts=shell_point_counts,
        far_field_potential_hartree_per_e=potential,
        evaluation_points_bohr_sha256=np.asarray(_array_sha256(points_bohr)),
        far_field_potential_sha256=np.asarray(_array_sha256(potential)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
