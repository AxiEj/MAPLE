#!/usr/bin/env python3
"""Evaluate a zero-field GFN2 MOLDEN permanent source against a QM static MEP.

This is a *benchmark helper*, not a Route-2 runtime component.  It reconstructs
xTB's printed contracted Cartesian MOLDEN AOs in a pinned PySCF integral carrier
and proves the representation transfer through AO-overlap and dipole identities
before evaluating the total effective-core-plus-valence electrostatic potential.
It then evaluates the total all-electron potential from one frozen QM checkpoint
on exactly the same external points.

No continuum, solvation energy, fitted parameter, response, or experimental
label is evaluated here.  The parent preregistered runner owns scientific gates
and the accept/reject decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.route2_v0_gfn2_molden_permanent_source import (  # noqa: E402
    Route2V0GFN2MoldenAODensity,
    load_route2_v0_gfn2_molden_permanent_reference,
)

EXPECTED_PYSCF_VERSION = "2.13.1"
_CARTESIAN_POWERS_BY_ANGULAR_MOMENTUM = {
    0: ((0, 0, 0),),
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molden", type=Path, required=True)
    parser.add_argument("--xtbout-json", type=Path, required=True)
    parser.add_argument("--xtb-stdout", type=Path, required=True)
    parser.add_argument("--parameter-file", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--qm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def _immutable_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array.") from error
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must have {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def molden_shells_to_pyscf_input(
    *,
    atom_symbols: Sequence[str],
    atom_positions_bohr: np.ndarray,
    ao_centers_bohr: np.ndarray,
    ao_powers: np.ndarray,
    ao_exponents: Sequence[np.ndarray],
    ao_coefficients: Sequence[np.ndarray],
    primitive_radial_normalization: Callable[[int, np.ndarray], np.ndarray],
) -> tuple[list[tuple[str, tuple[float, float, float]]], dict[str, list[list[object]]]]:
    """Convert raw printed MOLDEN contractions to PySCF's basis input format.

    PySCF multiplies every user coefficient by its primitive radial
    normalization before evaluating a shell.  The MOLDEN adapter instead owns
    the printed raw Cartesian function directly.  Dividing each printed
    coefficient by that known PySCF factor is therefore an exact representation
    conversion, not a fitted scale.  The caller must subsequently verify the
    resulting AO overlap and dipole; this routine refuses ambiguous shell or
    AO ordering before that check is possible.
    """

    symbols = tuple(atom_symbols)
    positions = _immutable_array(
        atom_positions_bohr,
        name="MOLDEN atom positions",
        shape=(len(symbols), 3),
    )
    centers = _immutable_array(
        ao_centers_bohr,
        name="MOLDEN AO centres",
    )
    powers = np.asarray(ao_powers)
    if (
        centers.ndim != 2
        or centers.shape[1] != 3
        or powers.shape != centers.shape
        or len(ao_exponents) != centers.shape[0]
        or len(ao_coefficients) != centers.shape[0]
    ):
        raise ValueError("MOLDEN AO arrays have inconsistent shapes.")
    if not all(isinstance(symbol, str) and symbol for symbol in symbols):
        raise ValueError("MOLDEN atom symbols must be nonempty strings.")

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    basis: dict[str, list[list[object]]] = {}
    next_ao = 0
    for atom_index, (symbol, center) in enumerate(zip(symbols, positions, strict=True)):
        label = f"{symbol}{atom_index + 1}"
        atoms.append((label, tuple(float(value) for value in center)))
        matching = np.flatnonzero(np.all(centers == center, axis=1))
        expected_indices = np.arange(next_ao, next_ao + matching.size)
        if not np.array_equal(matching, expected_indices):
            raise ValueError(
                "MOLDEN AOs must be contiguous and atom ordered for PySCF transfer."
            )
        shells: list[list[object]] = []
        local_index = next_ao
        stop = next_ao + matching.size
        while local_index < stop:
            raw_power = powers[local_index]
            if not np.all(np.isfinite(raw_power)) or not np.all(
                raw_power == np.rint(raw_power)
            ):
                raise ValueError("MOLDEN AO powers must be finite integers.")
            power = tuple(int(value) for value in raw_power)
            angular_momentum = int(sum(power))
            expected_powers = _CARTESIAN_POWERS_BY_ANGULAR_MOMENTUM.get(
                angular_momentum
            )
            if expected_powers is None or power != expected_powers[0]:
                raise ValueError(
                    "MOLDEN AO sequence must start every shell in canonical s/p order."
                )
            shell_size = len(expected_powers)
            if local_index + shell_size > stop:
                raise ValueError("MOLDEN Cartesian shell is truncated.")
            exponents = _immutable_array(
                ao_exponents[local_index],
                name="MOLDEN shell exponents",
            )
            coefficients = _immutable_array(
                ao_coefficients[local_index],
                name="MOLDEN shell coefficients",
                shape=exponents.shape,
            )
            if exponents.ndim != 1 or exponents.size == 0 or np.any(exponents <= 0.0):
                raise ValueError("MOLDEN shell exponents must be positive vectors.")
            for component, expected_power in enumerate(expected_powers):
                index = local_index + component
                if (
                    tuple(int(value) for value in powers[index]) != expected_power
                    or not np.array_equal(centers[index], center)
                    or not np.array_equal(ao_exponents[index], exponents)
                    or not np.array_equal(ao_coefficients[index], coefficients)
                ):
                    raise ValueError(
                        "MOLDEN Cartesian shell components must share one contraction."
                    )
            normalization = _immutable_array(
                primitive_radial_normalization(angular_momentum, exponents),
                name="PySCF primitive radial normalization",
                shape=exponents.shape,
            )
            if np.any(normalization <= 0.0):
                raise ValueError(
                    "PySCF primitive radial normalization must be positive."
                )
            shells.append(
                [
                    angular_momentum,
                    *[
                        [float(exponent), float(coefficient / normalizer)]
                        for exponent, coefficient, normalizer in zip(
                            exponents,
                            coefficients,
                            normalization,
                            strict=True,
                        )
                    ],
                ]
            )
            local_index += shell_size
        if not shells:
            raise ValueError("Every MOLDEN atom must own at least one AO shell.")
        basis[label] = shells
        next_ao = stop
    if next_ao != centers.shape[0]:
        raise ValueError("MOLDEN AO list contains unassigned functions.")
    return atoms, basis


def _candidate_integral_molecule(density: Route2V0GFN2MoldenAODensity):
    try:
        from pyscf import gto
    except ImportError as error:
        raise RuntimeError(
            "PySCF is required only for this static-MEP helper."
        ) from error
    atoms, basis = molden_shells_to_pyscf_input(
        atom_symbols=density.atom_symbols,
        atom_positions_bohr=density.atom_positions_bohr,
        ao_centers_bohr=density.ao_centers_bohr,
        ao_powers=density.ao_powers,
        ao_exponents=density.ao_exponents,
        ao_coefficients=density.ao_coefficients,
        primitive_radial_normalization=gto.gto_norm,
    )
    molecule = gto.M(
        atom=atoms,
        basis=basis,
        charge=0,
        spin=0,
        unit="Bohr",
        cart=True,
        symmetry=False,
        verbose=0,
    )
    if molecule.nao_nr() != density.ao_count:
        raise RuntimeError("PySCF MOLDEN carrier AO count disagrees with the export.")
    return molecule


def _checkpoint_density(checkpoint: Path):
    try:
        from pyscf import lib
    except ImportError as error:
        raise RuntimeError(
            "PySCF is required only for this static-MEP helper."
        ) from error
    molecule = lib.chkfile.load_mol(str(checkpoint))
    try:
        energy_hartree = float(lib.chkfile.load(str(checkpoint), "scf/e_tot"))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("QM checkpoint has no finite SCF energy.") from error
    if not np.isfinite(energy_hartree):
        raise RuntimeError("QM checkpoint has no finite SCF energy.")
    coefficients = _immutable_array(
        lib.chkfile.load(str(checkpoint), "scf/mo_coeff"),
        name="QM checkpoint MO coefficients",
    )
    occupations = _immutable_array(
        lib.chkfile.load(str(checkpoint), "scf/mo_occ"),
        name="QM checkpoint MO occupations",
    )
    nao = molecule.nao_nr()
    if (
        coefficients.shape != (nao, nao)
        or occupations.shape != (nao,)
        or np.any(occupations < 0.0)
        or np.any(occupations > 2.0)
        or np.any(np.minimum(abs(occupations), abs(occupations - 2.0)) > 1.0e-8)
    ):
        raise RuntimeError("QM checkpoint is not a square closed-shell AO state.")
    overlap = molecule.intor_symmetric("int1e_ovlp")
    metric_error = float(
        np.linalg.norm(coefficients.T @ overlap @ coefficients - np.eye(nao), ord=2)
    )
    density = (coefficients * occupations) @ coefficients.T
    density = 0.5 * (density + density.T)
    electron_count = float(np.einsum("ij,ji->", density, overlap, optimize=True))
    occupation_count = float(np.sum(occupations))
    if abs(electron_count - occupation_count) > 1.0e-7:
        raise RuntimeError("QM checkpoint density does not reproduce occupations.")
    return (
        molecule,
        density,
        {
            "ao_count": nao,
            "mo_metric_error": metric_error,
            "electron_count_e": electron_count,
            "electron_count_error_e": abs(electron_count - occupation_count),
            "energy_hartree": energy_hartree,
        },
    )


def _validate_common_geometry(
    *,
    density: Route2V0GFN2MoldenAODensity,
    checkpoint_molecule,
) -> float:
    checkpoint_numbers = np.asarray(checkpoint_molecule.atom_charges(), dtype=int)
    checkpoint_positions = np.asarray(checkpoint_molecule.atom_coords(), dtype=float)
    if not np.array_equal(checkpoint_numbers, density.atomic_numbers):
        raise RuntimeError("GFN2 MOLDEN and QM checkpoint atomic numbers differ.")
    if checkpoint_positions.shape != density.atom_positions_bohr.shape:
        raise RuntimeError("GFN2 MOLDEN and QM checkpoint atom counts differ.")
    coordinate_error = float(
        np.max(abs(checkpoint_positions - density.atom_positions_bohr), initial=0.0)
    )
    if coordinate_error > 1.0e-8:
        raise RuntimeError("GFN2 MOLDEN and QM checkpoint geometries differ.")
    return coordinate_error


def _dipole_e_bohr(
    molecule, density: np.ndarray, nuclear_charges: np.ndarray
) -> np.ndarray:
    positions = np.asarray(molecule.atom_coords(), dtype=float)
    dipole_integrals = molecule.intor_symmetric("int1e_r", comp=3)
    return np.einsum("a,ax->x", nuclear_charges, positions) - np.einsum(
        "xij,ji->x", dipole_integrals, density, optimize=True
    )


def _electronic_dipole_e_bohr(molecule, density: np.ndarray) -> np.ndarray:
    dipole_integrals = molecule.intor_symmetric("int1e_r", comp=3)
    return np.einsum("xij,ji->x", dipole_integrals, density, optimize=True)


def _total_potential_hartree_per_e(
    *,
    molecule,
    density: np.ndarray,
    nuclear_positions_bohr: np.ndarray,
    nuclear_charges_e: np.ndarray,
    points_bohr: np.ndarray,
) -> np.ndarray:
    potential = np.empty(points_bohr.shape[0], dtype=float)
    for index, point in enumerate(points_bohr):
        distances = np.linalg.norm(nuclear_positions_bohr - point, axis=1)
        if np.any(distances <= 1.0e-12):
            raise ValueError("A frozen MEP point coincides with a source nucleus.")
        nuclear = float(np.dot(nuclear_charges_e, 1.0 / distances))
        with molecule.with_rinv_origin(point):
            inverse_distance = molecule.intor("int1e_rinv")
        electronic = float(
            np.einsum("ij,ji->", density, inverse_distance, optimize=True)
        )
        potential[index] = nuclear - electronic
    if not np.all(np.isfinite(potential)):
        raise RuntimeError("Static electrostatic potential contains non-finite values.")
    potential.setflags(write=False)
    return potential


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = {
        "molden": arguments.molden.resolve(),
        "xtbout_json": arguments.xtbout_json.resolve(),
        "xtb_stdout": arguments.xtb_stdout.resolve(),
        "parameter_file": arguments.parameter_file.resolve(),
        "points": arguments.points.resolve(),
        "qm_checkpoint": arguments.qm_checkpoint.resolve(),
    }
    if missing := [name for name, path in paths.items() if not path.is_file()]:
        raise FileNotFoundError(f"Missing static-MEP input(s): {missing}")
    try:
        import pyscf
    except ImportError as error:
        raise RuntimeError(
            "PySCF is required only for this static-MEP helper."
        ) from error
    if pyscf.__version__ != EXPECTED_PYSCF_VERSION:
        raise RuntimeError(
            f"Static-MEP helper requires PySCF {EXPECTED_PYSCF_VERSION}, got {pyscf.__version__}."
        )

    points = _immutable_array(
        np.load(paths["points"], allow_pickle=False), name="MEP points"
    )
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError("MEP points must have finite shape (n, 3) in bohr.")
    reference = load_route2_v0_gfn2_molden_permanent_reference(
        molden_path=paths["molden"],
        xtbout_json_path=paths["xtbout_json"],
        stdout_path=paths["xtb_stdout"],
        parameter_file_path=paths["parameter_file"],
        expected_net_charge_e=0.0,
    )
    candidate_molecule = _candidate_integral_molecule(reference.density)
    qm_molecule, qm_density, qm_density_record = _checkpoint_density(
        paths["qm_checkpoint"]
    )
    coordinate_error = _validate_common_geometry(
        density=reference.density,
        checkpoint_molecule=qm_molecule,
    )

    candidate_overlap = candidate_molecule.intor_symmetric("int1e_ovlp")
    candidate_metric_error = float(
        np.linalg.norm(
            reference.density.molecular_orbital_coefficients.T
            @ candidate_overlap
            @ reference.density.molecular_orbital_coefficients
            - np.eye(reference.density.ao_count),
            ord=2,
        )
    )
    candidate_electronic_dipole = _electronic_dipole_e_bohr(
        candidate_molecule, reference.density.density_matrix
    )
    candidate_electronic_dipole_error = float(
        np.linalg.norm(
            candidate_electronic_dipole
            - reference.density.valence_electronic_dipole_e_bohr(),
            ord=2,
        )
    )
    candidate_total_dipole = _dipole_e_bohr(
        candidate_molecule,
        reference.density.density_matrix,
        reference.effective_core_charges_e,
    )
    candidate_total_dipole_error = float(
        np.linalg.norm(
            candidate_total_dipole - reference.reconstructed_total_dipole_e_bohr,
            ord=2,
        )
    )
    candidate_potential = _total_potential_hartree_per_e(
        molecule=candidate_molecule,
        density=reference.density.density_matrix,
        nuclear_positions_bohr=reference.density.atom_positions_bohr,
        nuclear_charges_e=reference.effective_core_charges_e,
        points_bohr=points,
    )
    qm_nuclear_charges = _immutable_array(
        qm_molecule.atom_charges(),
        name="QM nuclear charges",
        shape=(qm_molecule.natm,),
    )
    qm_potential = _total_potential_hartree_per_e(
        molecule=qm_molecule,
        density=qm_density,
        nuclear_positions_bohr=np.asarray(qm_molecule.atom_coords(), dtype=float),
        nuclear_charges_e=qm_nuclear_charges,
        points_bohr=points,
    )
    qm_total_dipole = _dipole_e_bohr(qm_molecule, qm_density, qm_nuclear_charges)

    artifact = {
        "schema_version": 1,
        "artifact": "route2-v0-gfn2-molden-static-mep-helper-v1",
        "status": "pass",
        "claim_boundary": (
            "Raw fixed-geometry gas-phase static electrostatic potentials only. "
            "This helper makes no acceptance decision and evaluates no continuum, "
            "solvation energy, response, force, runtime, or experimental label."
        ),
        "input_files_sha256": {name: _sha256(path) for name, path in paths.items()},
        "points": {
            "count": int(points.shape[0]),
            "array_sha256": _sha256_array(points),
        },
        "candidate_representation": {
            "ao_count": reference.density.ao_count,
            "pyscf_ao_count": candidate_molecule.nao_nr(),
            "molden_sha256": reference.density.molden_sha256,
            "xtbout_json_sha256": reference.xtbout_json_sha256,
            "xtb_stdout_sha256": reference.stdout_sha256,
            "molden_ao_metric_error": reference.density.mo_metric_error,
            "pyscf_ao_metric_error": candidate_metric_error,
            "pyscf_vs_molden_overlap_relative_frobenius": _relative_frobenius(
                candidate_overlap, reference.density.overlap_matrix
            ),
            "pyscf_vs_molden_overlap_max_abs": float(
                np.max(abs(candidate_overlap - reference.density.overlap_matrix))
            ),
            "pyscf_vs_molden_electronic_dipole_error_e_bohr": (
                candidate_electronic_dipole_error
            ),
            "pyscf_vs_molden_total_dipole_error_e_bohr": candidate_total_dipole_error,
            "molden_effective_core_total_charge_e": reference.total_effective_charge_e,
            "molden_total_dipole_e_bohr": reference.reconstructed_total_dipole_e_bohr.tolist(),
            "pyscf_carrier_total_dipole_e_bohr": candidate_total_dipole.tolist(),
        },
        "qm_reference": {
            "checkpoint_ao_density": {
                **qm_density_record,
                "density_sha256": _sha256_array(qm_density),
            },
            "pyscf_version": pyscf.__version__,
            "geometry_max_abs_error_bohr": coordinate_error,
            "total_charge_e": float(
                np.sum(qm_nuclear_charges) - qm_density_record["electron_count_e"]
            ),
            "total_dipole_e_bohr": qm_total_dipole.tolist(),
        },
        "raw_static_potential": {
            "candidate_effective_core_plus_valence_hartree_per_e": candidate_potential.tolist(),
            "candidate_sha256": _sha256_array(candidate_potential),
            "qm_all_electron_hartree_per_e": qm_potential.tolist(),
            "qm_sha256": _sha256_array(qm_potential),
        },
    }
    _write_exclusive_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
