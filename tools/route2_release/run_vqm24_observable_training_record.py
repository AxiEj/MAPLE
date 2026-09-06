#!/usr/bin/env python3
"""Generate one target-free VQM24 observable-supervision record.

The record contains zero-field and +/-1e-3 exterior point-charge energies,
full molecular MEPs, and molecular dipoles.  It reuses the already validated
finite-field SCF implementation and deliberately emits no density coefficient,
PCM/cavity quantity, solvation label, or numerical energy-curvature label.
"""

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


SELF_REPO_PATH = "tools/route2_release/run_vqm24_observable_training_record.py"
SHARED_RUNNER_REPO_PATH = finite.SELF_REPO_PATH
ARTIFACT = "route2-vqm24-observable-training-record-level3-v1"
FIELD_STEP_E = 1.0e-3


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


def _nuclear_potential(molecule, points_bohr: np.ndarray) -> np.ndarray:
    displacements = points_bohr[:, None, :] - molecule.atom_coords()[None, :, :]
    distances = np.linalg.norm(displacements, axis=2)
    if np.any(distances <= 0.0):
        raise RuntimeError("An exterior point coincides with a nucleus.")
    return np.sum(molecule.atom_charges()[None, :] / distances, axis=1)


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    surface_path = args.surface.expanduser().resolve(strict=True)
    modes_path = args.modes.expanduser().resolve(strict=True)
    output_json = args.output_json.expanduser().resolve()
    output_npz = args.output_npz.expanduser().resolve()
    if output_json.exists() or output_npz.exists():
        raise FileExistsError(output_json if output_json.exists() else output_npz)
    if args.threads != 8 or args.max_memory_mb != 8000:
        raise RuntimeError("Observable records require the frozen 8-thread/8-GB runtime.")
    if pyscf.__version__ != "2.13.1":
        raise RuntimeError("Observable records require PySCF 2.13.1.")

    molecule = lib.chkfile.load_mol(str(checkpoint))
    if molecule.charge != 0 or molecule.spin != 0:
        raise RuntimeError("Observable records currently accept neutral singlets only.")
    zero_energy = float(lib.chkfile.load(str(checkpoint), "scf/e_tot"))
    zero_density, density_provenance = finite._closed_shell_density_from_checkpoint(
        checkpoint,
        molecule,
    )
    surface_points = finite._load_points(
        surface_path,
        key="surface_points_bohr",
        label="Frozen validation surface",
    )
    source_points = finite._load_points(
        modes_path,
        key="source_points_bohr",
        label="Frozen localized source modes",
    )
    if source_points.shape != (4, 3):
        raise RuntimeError("Observable records require exactly four modes.")
    with np.load(surface_path, allow_pickle=False) as state:
        quadrature_weights = np.asarray(state["quadrature_weights"], dtype=np.float64)
        parent_atom_indices = np.asarray(state["parent_atom_indices"], dtype=np.int64)
    if (
        quadrature_weights.shape != (len(surface_points),)
        or parent_atom_indices.shape != (len(surface_points),)
        or not np.all(np.isfinite(quadrature_weights))
    ):
        raise RuntimeError("Frozen exterior-surface metadata is invalid.")

    lib.num_threads(args.threads)
    mean_field = finite._make_rks(molecule, max_memory_mb=args.max_memory_mb)
    mean_field.grids.build()
    mean_field.nlcgrids.build()
    shared_grids = (mean_field.grids, mean_field.nlcgrids)
    base_hcore = np.asarray(mean_field.get_hcore(), dtype=np.float64)
    overlap = molecule.intor_symmetric("int1e_ovlp")
    expected_electron_count = float(np.einsum("ij,ji->", zero_density, overlap))
    dipole_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3),
        dtype=np.float64,
    )
    surface_integrals = finite._coulomb_integrals(molecule, surface_points)
    source_integrals = finite._coulomb_integrals(molecule, source_points)
    nuclear_surface_potential = _nuclear_potential(molecule, surface_points)
    nuclear_source_potential = _nuclear_potential(molecule, source_points)

    zero_electronic_surface_mep = finite._electronic_potential(
        zero_density,
        surface_integrals,
    )
    zero_total_surface_mep = zero_electronic_surface_mep + nuclear_surface_potential
    zero_total_dipole = finite._total_dipole_e_bohr(
        molecule,
        zero_density,
        dipole_integrals,
    )

    signs = (-1, 1)
    perturbed_enthalpy = np.empty((4, 2), dtype=np.float64)
    perturbed_surface_mep = np.empty(
        (4, 2, len(surface_points)),
        dtype=np.float64,
    )
    perturbed_dipole = np.empty((4, 2, 3), dtype=np.float64)
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for mode_index, source_integral in enumerate(source_integrals):
        for sign_index, sign in enumerate(signs):
            amplitude = sign * FIELD_STEP_E
            state = finite._run_perturbation(
                molecule,
                base_hcore=base_hcore,
                zero_density=zero_density,
                source_integral=source_integral,
                amplitude_e=amplitude,
                dipole_integrals=dipole_integrals,
                surface_integrals=surface_integrals,
                overlap=overlap,
                expected_electron_count=expected_electron_count,
                max_memory_mb=args.max_memory_mb,
                shared_grids=shared_grids,
            )
            external_nuclear_coupling = amplitude * nuclear_source_potential[
                mode_index
            ]
            total_enthalpy = float(state["energy_hartree"]) + float(
                external_nuclear_coupling
            )
            total_surface_mep = (
                np.asarray(
                    state["electronic_surface_potential_hartree_per_e"],
                    dtype=np.float64,
                )
                + nuclear_surface_potential
            )
            total_dipole = np.asarray(state["dipole_e_bohr"], dtype=np.float64)
            perturbed_enthalpy[mode_index, sign_index] = total_enthalpy
            perturbed_surface_mep[mode_index, sign_index] = total_surface_mep
            perturbed_dipole[mode_index, sign_index] = total_dipole
            records.append(
                {
                    "mode_index": mode_index,
                    "sign": sign,
                    "amplitude_e": amplitude,
                    "electronic_scf_energy_hartree": float(state["energy_hartree"]),
                    "external_nuclear_coupling_hartree": float(
                        external_nuclear_coupling
                    ),
                    "total_external_enthalpy_hartree": total_enthalpy,
                    "surface_mep_sha256": _sha256_array(total_surface_mep),
                    "dipole_e_bohr": total_dipole.tolist(),
                    "density_sha256": finite._sha256_array(state["density"]),
                    "electron_count_error_e": float(state["electron_count_error_e"]),
                    "scf_cycles": int(state["scf_cycles"]),
                    "elapsed_seconds": float(state["elapsed_seconds"]),
                }
            )

    scale = 1.0 / (2.0 * FIELD_STEP_E)
    induced_surface_mep = scale * (
        perturbed_surface_mep[:, 1] - perturbed_surface_mep[:, 0]
    )
    induced_dipole = scale * (
        perturbed_dipole[:, 1] - perturbed_dipole[:, 0]
    )
    enthalpy_slope = scale * (
        perturbed_enthalpy[:, 1] - perturbed_enthalpy[:, 0]
    )
    zero_total_source_mep = (
        finite._electronic_potential(zero_density, source_integrals)
        + nuclear_source_potential
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("xb") as handle:
        np.savez(
            handle,
            source_points_bohr=source_points,
            surface_points_bohr=surface_points,
            quadrature_weights=quadrature_weights,
            parent_atom_indices=parent_atom_indices,
            signs=np.asarray(signs, dtype=np.int64),
            field_step_e=np.asarray(FIELD_STEP_E),
            zero_energy_hartree=np.asarray(zero_energy),
            zero_total_surface_mep_hartree_per_e=zero_total_surface_mep,
            zero_total_source_mep_hartree_per_e=zero_total_source_mep,
            zero_total_dipole_e_bohr=zero_total_dipole,
            perturbed_total_enthalpy_hartree=perturbed_enthalpy,
            perturbed_total_surface_mep_hartree_per_e=perturbed_surface_mep,
            perturbed_total_dipole_e_bohr=perturbed_dipole,
            induced_surface_mep_hartree_per_e_per_source_e=induced_surface_mep,
            induced_dipole_e_bohr_per_source_e=induced_dipole,
            central_enthalpy_slope_hartree_per_e=enthalpy_slope,
        )

    result = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "success",
        "input": {
            "checkpoint_sha256": _sha256(checkpoint),
            "surface_npz_sha256": _sha256(surface_path),
            "modes_npz_sha256": _sha256(modes_path),
            "surface_points_bohr_sha256": _sha256_array(surface_points),
            "source_points_bohr_sha256": _sha256_array(source_points),
        },
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "shared_finite_field_runner_path": SHARED_RUNNER_REPO_PATH,
            "shared_finite_field_runner_sha256": _sha256(
                SOURCE_ROOT / SHARED_RUNNER_REPO_PATH
            ),
        },
        "method": finite.QM_METHOD,
        "finite_field_protocol": {
            "field_step_e": FIELD_STEP_E,
            "signs": list(signs),
            "mode_count": 4,
            "external_nuclear_coupling_included_in_total_enthalpy": True,
            "numerical_energy_curvature_label_emitted": False,
        },
        "zero_field": {
            **density_provenance,
            "energy_hartree": zero_energy,
            "electron_count_e": expected_electron_count,
            "total_surface_mep_sha256": _sha256_array(zero_total_surface_mep),
            "total_source_mep_sha256": _sha256_array(zero_total_source_mep),
            "total_dipole_e_bohr": zero_total_dipole.tolist(),
        },
        "perturbation_records": records,
        "numerical_checks": {
            "maximum_electron_count_error_e": max(
                float(record["electron_count_error_e"]) for record in records
            ),
            "minimum_scf_cycles": min(int(record["scf_cycles"]) for record in records),
            "maximum_scf_cycles": max(int(record["scf_cycles"]) for record in records),
            "enthalpy_slope_vs_zero_source_mep_relative": float(
                np.linalg.norm(enthalpy_slope - zero_total_source_mep)
                / max(
                    np.linalg.norm(enthalpy_slope),
                    np.linalg.norm(zero_total_source_mep),
                    np.finfo(float).tiny,
                )
            ),
        },
        "output": {
            "npz_sha256": _sha256(output_npz),
            "zero_surface_mep_sha256": _sha256_array(zero_total_surface_mep),
            "perturbed_enthalpy_sha256": _sha256_array(perturbed_enthalpy),
            "perturbed_surface_mep_sha256": _sha256_array(
                perturbed_surface_mep
            ),
            "perturbed_dipole_sha256": _sha256_array(perturbed_dipole),
            "induced_surface_mep_sha256": _sha256_array(induced_surface_mep),
            "induced_dipole_sha256": _sha256_array(induced_dipole),
        },
        "runtime": {
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "threads": lib.num_threads(),
            "total_elapsed_seconds": time.perf_counter() - started,
        },
        "claim_boundary": {
            "independent_qm_observable_training_record_generated": True,
            "density_or_partition_coefficient_label_emitted": False,
            "numerical_energy_curvature_label_emitted": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "fit_or_training_performed": False,
            "model_accuracy_measured": False,
            "capability_admitted": False,
        },
    }
    finite._write_exclusive_json(output_json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
