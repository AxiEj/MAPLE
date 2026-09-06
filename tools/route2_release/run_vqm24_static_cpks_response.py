#!/usr/bin/env python3
"""Generate observable static RKS response with PySCF CPKS.

The helper imports no MAPLE package.  It differentiates the same frozen
omegaB97M-V/def2-TZVPD density-fitted RKS problem used by the finite-field
VQM24 oracle.  Only observable contractions are emitted; the AO density
response is deliberately not released as a training label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pyscf
from pyscf import dft, lib
from pyscf.scf import cphf


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/run_vqm24_static_cpks_response.py"
ARTIFACT = "route2-vqm24-static-cpks-observable-response-level3-v1"
QM_METHOD = {
    "electronic_structure": "omegaB97M-V",
    "pyscf_xc_token": "wb97m-v",
    "basis": "def2-tzvpd",
    "reference": "RKS",
    "density_fitting": True,
    "charge": 0,
    "spin": 0,
    "semilocal_grid_level": 3,
    "nonlocal_grid_profile": "PySCF-level-3",
    "nonlocal_grid_level": 3,
    "with_nlc_response": True,
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


def _callable_name(value: object) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    return f"{module}.{name}"


def _load_array(path: Path, *, key: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as state:
        if key not in state:
            raise RuntimeError(f"{path} omits {key}.")
        values = np.asarray(state[key], dtype=np.float64)
    if (
        values.ndim != len(shape_tail) + 1
        or values.shape[0] == 0
        or values.shape[1:] != shape_tail
        or not np.all(np.isfinite(values))
    ):
        raise RuntimeError(f"{key} has an invalid shape or nonfinite values.")
    return values


def _rinv_integrals(molecule, points_bohr: np.ndarray) -> np.ndarray:
    values = np.empty(
        (len(points_bohr), molecule.nao_nr(), molecule.nao_nr()),
        dtype=np.float64,
    )
    for index, point in enumerate(points_bohr):
        with molecule.with_rinv_origin(point):
            values[index] = molecule.intor_symmetric("int1e_rinv")
    return values


def _nuclear_potential(molecule, points_bohr: np.ndarray) -> np.ndarray:
    displacements = points_bohr[:, None, :] - molecule.atom_coords()[None, :, :]
    distances = np.linalg.norm(displacements, axis=2)
    if np.any(distances <= 0.0):
        raise RuntimeError("An exterior probe coincides with a nucleus.")
    return np.sum(molecule.atom_charges()[None, :] / distances, axis=1)


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _relative_residuals(
    residual: np.ndarray,
    delta_u: np.ndarray,
    h_vo: np.ndarray,
    v_vo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frobenius = []
    infinity = []
    for index in range(len(residual)):
        denominator_frobenius = (
            np.linalg.norm(delta_u[index])
            + np.linalg.norm(h_vo[index])
            + np.linalg.norm(v_vo[index])
        )
        denominator_infinity = max(
            float(np.max(np.abs(delta_u[index]))),
            float(np.max(np.abs(h_vo[index]))),
            float(np.max(np.abs(v_vo[index]))),
            np.finfo(float).tiny,
        )
        frobenius.append(
            np.linalg.norm(residual[index])
            / max(float(denominator_frobenius), np.finfo(float).tiny)
        )
        infinity.append(
            float(np.max(np.abs(residual[index]))) / denominator_infinity
        )
    return np.asarray(frobenius), np.asarray(infinity)


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
        raise RuntimeError("Static CPKS requires the frozen 8-thread/8-GB runtime.")
    if pyscf.__version__ != "2.13.1":
        raise RuntimeError("Static CPKS requires PySCF 2.13.1.")

    source_points = _load_array(modes_path, key="source_points_bohr", shape_tail=(3,))
    surface_points = _load_array(
        surface_path,
        key="surface_points_bohr",
        shape_tail=(3,),
    )
    if source_points.shape != (4, 3):
        raise RuntimeError("Static CPKS requires exactly four localized modes.")
    with np.load(surface_path, allow_pickle=False) as state:
        quadrature_weights = np.asarray(state["quadrature_weights"], dtype=np.float64)
        parent_atom_indices = np.asarray(state["parent_atom_indices"], dtype=np.int64)
    if (
        quadrature_weights.shape != (len(surface_points),)
        or parent_atom_indices.shape != (len(surface_points),)
        or not np.all(np.isfinite(quadrature_weights))
    ):
        raise RuntimeError("Frozen exterior-surface metadata is invalid.")

    molecule = lib.chkfile.load_mol(str(checkpoint))
    if molecule.charge != 0 or molecule.spin != 0:
        raise RuntimeError("Static CPKS currently accepts neutral singlets only.")
    mo_coeff = np.asarray(lib.chkfile.load(str(checkpoint), "scf/mo_coeff"))
    mo_occ = np.asarray(lib.chkfile.load(str(checkpoint), "scf/mo_occ"))
    mo_energy = np.asarray(lib.chkfile.load(str(checkpoint), "scf/mo_energy"))
    zero_energy = float(lib.chkfile.load(str(checkpoint), "scf/e_tot"))
    if not np.all(np.isin(mo_occ, (0.0, 2.0))):
        raise RuntimeError("Static CPKS requires exact closed-shell 2/0 occupations.")
    occupied = mo_occ > 0.0
    virtual = mo_occ == 0.0
    c_occ = mo_coeff[:, occupied]
    c_vir = mo_coeff[:, virtual]
    epsilon_occ = mo_energy[occupied]
    epsilon_vir = mo_energy[virtual]
    density_zero = np.einsum(
        "pi,i,qi->pq",
        mo_coeff,
        mo_occ,
        mo_coeff.conj(),
        optimize=True,
    ).real

    lib.num_threads(args.threads)
    mean_field = dft.RKS(
        molecule,
        xc=QM_METHOD["pyscf_xc_token"],
    ).density_fit()
    mean_field.grids.level = QM_METHOD["semilocal_grid_level"]
    mean_field.nlcgrids.level = QM_METHOD["nonlocal_grid_level"]
    mean_field.max_memory = args.max_memory_mb
    mean_field.mo_coeff = mo_coeff
    mean_field.mo_occ = mo_occ
    mean_field.mo_energy = mo_energy
    mean_field.e_tot = zero_energy
    mean_field.converged = True

    total_started = time.perf_counter()
    mean_field.grids.build()
    mean_field.nlcgrids.build()
    resolved_auxiliary_basis = str(mean_field.with_df.auxbasis)
    auxiliary_function_count = int(mean_field.with_df.get_naoaux())
    grid_identity = {
        "semilocal": {
            "point_count": int(len(mean_field.grids.coords)),
            "coordinates_sha256": _sha256_array(mean_field.grids.coords),
            "weights_sha256": _sha256_array(mean_field.grids.weights),
            "prune": _callable_name(mean_field.grids.prune),
            "radii_adjust": _callable_name(mean_field.grids.radii_adjust),
        },
        "nonlocal": {
            "point_count": int(len(mean_field.nlcgrids.coords)),
            "coordinates_sha256": _sha256_array(mean_field.nlcgrids.coords),
            "weights_sha256": _sha256_array(mean_field.nlcgrids.weights),
            "prune": _callable_name(mean_field.nlcgrids.prune),
            "radii_adjust": _callable_name(mean_field.nlcgrids.radii_adjust),
        },
        "small_rho_cutoff": float(mean_field.small_rho_cutoff),
    }

    overlap = mean_field.get_ovlp(molecule)
    hcore_zero = mean_field.get_hcore(molecule)
    effective_zero = mean_field.get_veff(molecule, density_zero)
    fock_zero = hcore_zero + effective_zero
    orthonormality_residual = (
        mo_coeff.conj().T @ overlap @ mo_coeff - np.eye(mo_coeff.shape[1])
    )
    canonical_residual = (
        fock_zero @ mo_coeff
        - (overlap @ mo_coeff) * mo_energy[None, :]
    )
    canonical_residual_relative = float(
        np.linalg.norm(canonical_residual)
        / max(
            np.linalg.norm(fock_zero @ mo_coeff)
            + np.linalg.norm((overlap @ mo_coeff) * mo_energy[None, :]),
            np.finfo(float).tiny,
        )
    )
    fock_vo = c_vir.conj().T @ fock_zero @ c_occ
    rebuilt_energy = float(
        mean_field.energy_tot(
            dm=density_zero,
            h1e=hcore_zero,
            vhf=effective_zero,
        )
    )

    source_rinv = _rinv_integrals(molecule, source_points)
    surface_rinv = _rinv_integrals(molecule, surface_points)
    perturbation_ao = -source_rinv
    perturbation_vo = np.einsum(
        "pa,kpq,qi->kai",
        c_vir.conj(),
        perturbation_ao,
        c_occ,
        optimize=True,
    )
    nmode, nvir, nocc = perturbation_vo.shape

    response_kernel = mean_field.gen_response(
        mo_coeff,
        mo_occ,
        singlet=None,
        hermi=1,
        with_nlc=True,
    )

    def amplitudes_to_density(amplitudes: np.ndarray) -> np.ndarray:
        values = np.asarray(amplitudes).reshape(-1, nvir, nocc)
        half = np.einsum(
            "pa,xai,qi->xpq",
            c_vir,
            2.0 * values,
            c_occ.conj(),
            optimize=True,
        )
        return half + half.swapaxes(-1, -2).conj()

    def induced_vo(amplitudes: np.ndarray) -> np.ndarray:
        original_shape = np.asarray(amplitudes).shape
        density_response = amplitudes_to_density(amplitudes)
        potential_response = np.asarray(response_kernel(density_response))
        values = np.einsum(
            "pa,xpq,qi->xai",
            c_vir.conj(),
            potential_response,
            c_occ,
            optimize=True,
        )
        return values.reshape(original_shape)

    solve_started = time.perf_counter()
    amplitudes, _ = cphf.solve(
        induced_vo,
        mo_energy,
        mo_occ,
        perturbation_vo,
        s1=None,
        max_cycle=100,
        tol=1.0e-10,
        hermi=False,
        verbose=4,
        level_shift=0.0,
    )
    amplitudes = np.asarray(amplitudes).reshape(nmode, nvir, nocc)
    energy_denominator = epsilon_vir[:, None] - epsilon_occ[None, :]

    def residual_state(values: np.ndarray) -> dict[str, np.ndarray]:
        v_vo = np.asarray(induced_vo(values)).reshape(nmode, nvir, nocc)
        delta_u = energy_denominator[None, :, :] * values
        residual = delta_u + perturbation_vo + v_vo
        relative_frobenius, relative_infinity = _relative_residuals(
            residual,
            delta_u,
            perturbation_vo,
            v_vo,
        )
        return {
            "v_vo": v_vo,
            "delta_u": delta_u,
            "residual": residual,
            "relative_frobenius": relative_frobenius,
            "relative_infinity": relative_infinity,
        }

    initial_residual = residual_state(amplitudes)
    refinement_cycles: list[int] = []
    if max(initial_residual["relative_frobenius"]) > 1.0e-9:
        inverse_denominator = 1.0 / energy_denominator
        right_hand_side = -perturbation_vo * inverse_denominator[None, :, :]

        def preconditioned_action(flat_values: np.ndarray) -> np.ndarray:
            values = np.asarray(flat_values).reshape(-1, nvir, nocc)
            induced = np.asarray(induced_vo(values)).reshape(-1, nvir, nocc)
            return (induced * inverse_denominator[None, :, :]).reshape(
                np.asarray(flat_values).shape
            )

        def count_cycle(cycle: int, _xs: object, _ax: object) -> None:
            refinement_cycles.append(int(cycle))

        amplitudes = np.asarray(
            lib.krylov(
                preconditioned_action,
                right_hand_side.reshape(nmode, -1),
                x0=amplitudes.reshape(nmode, -1),
                tol=1.0e-12,
                max_cycle=100,
                lindep=1.0e-16,
                callback=count_cycle,
                hermi=False,
                max_memory=args.max_memory_mb,
                verbose=4,
            )
        ).reshape(nmode, nvir, nocc)
    final_residual = residual_state(amplitudes)
    solve_elapsed = time.perf_counter() - solve_started

    density_response = amplitudes_to_density(amplitudes).real
    induced_surface_mep = -np.einsum(
        "kpq,sqp->ks",
        density_response,
        surface_rinv,
        optimize=True,
    ).real
    dipole_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3),
        dtype=np.float64,
    )
    induced_dipole = -np.einsum(
        "kpq,xqp->kx",
        density_response,
        dipole_integrals,
        optimize=True,
    ).real
    susceptibility = np.einsum(
        "kpq,lqp->lk",
        density_response,
        perturbation_ao,
        optimize=True,
    ).real
    energy_curvature_ao = np.diag(susceptibility).copy()
    energy_curvature_vo = 4.0 * np.einsum(
        "kai,kai->k",
        perturbation_vo.conj(),
        amplitudes,
        optimize=True,
    ).real
    electron_number_derivative = np.einsum(
        "pq,kqp->k",
        overlap,
        density_response,
        optimize=True,
    ).real
    density_hermiticity_maximum = np.max(
        np.abs(density_response - density_response.swapaxes(-1, -2)),
        axis=(1, 2),
    )

    zero_electronic_surface_mep = -np.einsum(
        "pq,sqp->s",
        density_zero,
        surface_rinv,
        optimize=True,
    ).real
    zero_total_surface_mep = (
        zero_electronic_surface_mep
        + _nuclear_potential(molecule, surface_points)
    )
    zero_electronic_source_mep = -np.einsum(
        "pq,kqp->k",
        density_zero,
        source_rinv,
        optimize=True,
    ).real
    zero_total_source_mep = (
        zero_electronic_source_mep
        + _nuclear_potential(molecule, source_points)
    )
    zero_dipole = (
        np.einsum("i,ix->x", molecule.atom_charges(), molecule.atom_coords())
        - np.einsum("pq,xqp->x", density_zero, dipole_integrals, optimize=True).real
    )

    symmetric_susceptibility = 0.5 * (
        susceptibility + susceptibility.T
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("xb") as handle:
        np.savez(
            handle,
            source_points_bohr=source_points,
            surface_points_bohr=surface_points,
            quadrature_weights=quadrature_weights,
            parent_atom_indices=parent_atom_indices,
            zero_total_surface_mep_hartree_per_e=zero_total_surface_mep,
            zero_total_source_mep_hartree_per_e=zero_total_source_mep,
            zero_total_dipole_e_bohr=zero_dipole,
            induced_surface_mep_hartree_per_e_per_source_e=induced_surface_mep,
            induced_dipole_e_bohr_per_source_e=induced_dipole,
            energy_curvature_hartree_per_e2=energy_curvature_ao,
            susceptibility_hartree_per_e2=susceptibility,
        )

    result: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "success",
        "input": {
            "checkpoint_sha256": _sha256(checkpoint),
            "surface_npz_sha256": _sha256(surface_path),
            "modes_npz_sha256": _sha256(modes_path),
            "source_points_bohr_sha256": _sha256_array(source_points),
            "surface_points_bohr_sha256": _sha256_array(surface_points),
        },
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
        },
        "method": {
            **QM_METHOD,
            "resolved_auxiliary_basis": resolved_auxiliary_basis,
            "auxiliary_function_count": auxiliary_function_count,
            "only_dfj": bool(mean_field.only_dfj),
            "grids": grid_identity,
        },
        "checkpoint_consistency": {
            "orthonormality_residual_inf": float(
                np.max(np.abs(orthonormality_residual))
            ),
            "canonical_fock_residual_relative_frobenius": (
                canonical_residual_relative
            ),
            "fock_virtual_occupied_residual_inf_hartree": float(
                np.max(np.abs(fock_vo))
            ),
            "rebuilt_energy_error_hartree": rebuilt_energy - zero_energy,
        },
        "response": {
            "mode_count": nmode,
            "solve_elapsed_seconds": solve_elapsed,
            "refinement_cycle_count": len(refinement_cycles),
            "initial_residual_relative_frobenius": initial_residual[
                "relative_frobenius"
            ].tolist(),
            "final_residual_relative_frobenius": final_residual[
                "relative_frobenius"
            ].tolist(),
            "final_residual_relative_infinity": final_residual[
                "relative_infinity"
            ].tolist(),
            "density_response_sha256": _sha256_array(density_response),
            "density_hermiticity_maximum": density_hermiticity_maximum.tolist(),
            "electron_number_derivative_e_per_source_e": (
                electron_number_derivative.tolist()
            ),
            "energy_curvature_ao_hartree_per_e2": energy_curvature_ao.tolist(),
            "energy_curvature_vo_hartree_per_e2": energy_curvature_vo.tolist(),
            "energy_identity_error_hartree_per_e2": (
                energy_curvature_ao - energy_curvature_vo
            ).tolist(),
            "susceptibility_reciprocity_relative_frobenius": float(
                np.linalg.norm(susceptibility - susceptibility.T)
                / max(np.linalg.norm(susceptibility), np.finfo(float).tiny)
            ),
            "susceptibility_symmetric_eigenvalues_hartree_per_e2": (
                np.linalg.eigvalsh(symmetric_susceptibility).tolist()
            ),
        },
        "zero_field": {
            "energy_hartree": zero_energy,
            "total_surface_mep_sha256": _sha256_array(zero_total_surface_mep),
            "total_source_mep_sha256": _sha256_array(zero_total_source_mep),
            "total_dipole_e_bohr": zero_dipole.tolist(),
        },
        "output": {
            "npz_sha256": _sha256(output_npz),
            "induced_surface_mep_sha256": _sha256_array(induced_surface_mep),
            "induced_dipole_sha256": _sha256_array(induced_dipole),
            "susceptibility_sha256": _sha256_array(susceptibility),
        },
        "runtime": {
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "threads": lib.num_threads(),
            "total_elapsed_seconds": time.perf_counter() - total_started,
        },
        "claim_boundary": {
            "q_to_zero_static_response_computed": True,
            "ao_density_response_released_as_training_label": False,
            "fit_or_training_performed": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "capability_admitted": False,
        },
    }
    _write_exclusive_json(output_json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
