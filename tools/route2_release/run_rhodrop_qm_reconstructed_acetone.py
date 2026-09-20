#!/usr/bin/env python3
"""Compare QM- and reconstructed-density rho-DROP cavities on acetone.

This is a label-free, fixed-geometry diagnostic.  It freezes the MACE-POLAR
gas source and evaluates a two-by-two matrix:

* QM total MEP or MACE point-l<=1 MEP;
* QM AO-density or reconstructed-MACE-density rho-DROP cavity.

All four entries use the same MOIST C-PCM equation and dielectric.  The result
therefore separates a cavity shift, a source shift, and their interaction
without using an experimental solvation free energy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from ase.units import Bohr, Hartree, kcal, mol
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.gto_density import (  # noqa: E402
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (  # noqa: E402
    load_atomic_reference_density_asset,
)
from maple.function.calculator.extra_correction.implicit.route2_density_levelset import (  # noqa: E402
    ReconstructedMacePolarDensityLevelSet,
)
from maple.function.calculator.extra_correction.implicit.route2_moist_drop import (  # noqa: E402
    MoistDropAdapter,
    MoistDropSettings,
    MoistRuntimeProvenance,
)
from maple.function.calculator.extra_correction.implicit.route2_rhodrop_cpcm import (  # noqa: E402
    RhoDropCPCMSettings,
)
from maple.solvation.reference.pyscf_density_levelset import (  # noqa: E402
    PySCFAODensityLevelSet,
)
from maple.solvation.reference.pyscf_pcmsolver import (  # noqa: E402
    AOInverseDistanceIntegralCache,
    array_sha256,
)


ARTIFACT_ID = "route2-rhodrop-qm-reconstructed-acetone-diagnostic-v1"
EXPECTED_QM_CHECKPOINT_SHA256 = (
    "b1bfc58d3389e637fcc6000e928025dee8ac24ee58b090e096c4f26120ce43c0"
)
EXPECTED_MACE_STATE_SHA256 = (
    "56aa5bd0a7cbbac08da9ae2522567031994f77a2152058cbbe42e7bebb0c276e"
)
EXPECTED_PDF_SHA256 = (
    "f2293556d744c728f8cc9a114b96598bf82042771d88a27d4fe55aa4a2ca1388"
)
EXPECTED_COMPOUND_ID = "mobley_3867265"
N_ISO_E_PER_BOHR3 = 1.0e-3
KCAL_PER_HARTREE = Hartree / (kcal / mol)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_hash(path: Path, expected: str, *, role: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(f"{role} SHA256 drifted: {observed} != {expected}")
    return observed


def _runtime(extension: Path, shared_library: Path) -> MoistRuntimeProvenance:
    return MoistRuntimeProvenance(
        evidence_kind="real-pinned-build",
        python_extension_path=str(extension),
        python_extension_sha256=_sha256(extension),
        shared_library_path=str(shared_library),
        shared_library_sha256=_sha256(shared_library),
        build_toolchain=(
            "gfortran-11.4.0",
            "meson-1.11.2",
            "ninja-1.13.0",
            "openmp-enabled",
        ),
    )


def _surface_metrics(qm_surface, reconstructed_surface, qm_level, rec_level) -> dict:
    qm = qm_surface.snapshot
    rec = reconstructed_surface.snapshot
    qm_tree = cKDTree(qm.surface_points_bohr)
    rec_tree = cKDTree(rec.surface_points_bohr)
    rec_to_qm_distance, rec_to_qm_index = qm_tree.query(rec.surface_points_bohr)
    qm_to_rec_distance, qm_to_rec_index = rec_tree.query(qm.surface_points_bohr)

    rec_normal_overlap = np.abs(
        np.einsum(
            "ij,ij->i",
            rec.surface_normals,
            qm.surface_normals[np.asarray(rec_to_qm_index, dtype=int)],
        )
    )
    qm_normal_overlap = np.abs(
        np.einsum(
            "ij,ij->i",
            qm.surface_normals,
            rec.surface_normals[np.asarray(qm_to_rec_index, dtype=int)],
        )
    )
    rec_on_qm = rec_level.evaluate_spatial(qm.surface_points_bohr).value
    qm_on_rec = qm_level.evaluate_spatial(rec.surface_points_bohr).value

    def distance_summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_bohr": float(np.mean(values)),
            "rms_bohr": float(np.sqrt(np.mean(np.square(values)))),
            "maximum_bohr": float(np.max(values)),
            "mean_angstrom": float(np.mean(values) * Bohr),
            "rms_angstrom": float(np.sqrt(np.mean(np.square(values))) * Bohr),
            "maximum_angstrom": float(np.max(values) * Bohr),
        }

    def residual_summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_absolute": float(np.mean(np.abs(values))),
            "rms": float(np.sqrt(np.mean(np.square(values)))),
            "maximum_absolute": float(np.max(np.abs(values))),
        }

    return {
        "qm_surface": {
            "point_count": qm.surface_size,
            "area_bohr2": qm.area_bohr2,
            "volume_bohr3": qm.volume_bohr3,
            "snapshot_sha256": qm.snapshot_sha256,
            "owner_counts": np.bincount(
                qm.owner_atom_indices, minlength=qm.atom_count
            ).tolist(),
        },
        "reconstructed_surface": {
            "point_count": rec.surface_size,
            "area_bohr2": rec.area_bohr2,
            "volume_bohr3": rec.volume_bohr3,
            "snapshot_sha256": rec.snapshot_sha256,
            "owner_counts": np.bincount(
                rec.owner_atom_indices, minlength=rec.atom_count
            ).tolist(),
        },
        "relative_area_error_reconstructed_vs_qm": float(
            (rec.area_bohr2 - qm.area_bohr2) / qm.area_bohr2
        ),
        "relative_volume_error_reconstructed_vs_qm": float(
            (rec.volume_bohr3 - qm.volume_bohr3) / qm.volume_bohr3
        ),
        "nearest_surface_distance": {
            "reconstructed_to_qm": distance_summary(rec_to_qm_distance),
            "qm_to_reconstructed": distance_summary(qm_to_rec_distance),
        },
        "nearest_normal_absolute_dot": {
            "reconstructed_to_qm_mean": float(np.mean(rec_normal_overlap)),
            "reconstructed_to_qm_minimum": float(np.min(rec_normal_overlap)),
            "qm_to_reconstructed_mean": float(np.mean(qm_normal_overlap)),
            "qm_to_reconstructed_minimum": float(np.min(qm_normal_overlap)),
        },
        "cross_level_set_residual": {
            "reconstructed_level_set_on_qm_surface": residual_summary(rec_on_qm),
            "qm_level_set_on_reconstructed_surface": residual_summary(qm_on_rec),
        },
    }


def _solve_surface_potential(
    surface_state,
    potential: np.ndarray,
    *,
    settings: RhoDropCPCMSettings,
) -> dict[str, Any]:
    matrix = np.asarray(surface_state.snapshot.amat, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if potential.shape != (matrix.shape[0],) or not np.all(np.isfinite(potential)):
        raise ValueError("C-PCM surface potential is invalid.")
    factor = cho_factor(matrix, lower=True, overwrite_a=False, check_finite=False)
    rhs = -settings.dielectric_factor * potential
    charge = np.asarray(cho_solve(factor, rhs, check_finite=False), dtype=float)
    residual = matrix @ charge - rhs
    condition = float(np.linalg.cond(matrix, p=2))
    energy = 0.5 * float(np.dot(potential, charge))
    if energy > settings.energy_identity_absolute_tolerance_hartree:
        raise RuntimeError("Reference C-PCM polarization energy is positive.")
    if condition > settings.maximum_condition_number_2:
        raise RuntimeError("Reference C-PCM matrix is too ill-conditioned.")
    return {
        "energy_hartree": energy,
        "energy_kcal_mol": energy * KCAL_PER_HARTREE,
        "potential_sha256": array_sha256(potential),
        "surface_charge_sha256": array_sha256(charge),
        "surface_charge_sum_e": float(np.sum(charge)),
        "linear_residual_inf": float(np.linalg.norm(residual, ord=np.inf)),
        "condition_number_2": condition,
    }


def _qm_surface_potential(level_set, surface_state, cache_path: Path) -> np.ndarray:
    snapshot = surface_state.snapshot
    cache = AOInverseDistanceIntegralCache(
        level_set.molecule,
        snapshot.surface_points_bohr,
        path=cache_path,
        batch_size=16,
    )
    try:
        electronic = cache.electronic_surface_potential(level_set.density_matrix)
    finally:
        cache.close()
    distances = np.linalg.norm(
        snapshot.surface_points_bohr[:, None, :]
        - level_set.positions_bohr[None, :, :],
        axis=2,
    )
    if np.any(distances <= 1.0e-12):
        raise RuntimeError("A rho-DROP surface point coincides with a nucleus.")
    nuclear = (1.0 / distances) @ level_set.atomic_numbers
    return nuclear - electronic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qm-checkpoint", type=Path, required=True)
    parser.add_argument("--mace-state", type=Path, required=True)
    parser.add_argument("--reference-density", type=Path, required=True)
    parser.add_argument("--literature-pdf", type=Path, required=True)
    parser.add_argument("--moist-extension", type=Path, required=True)
    parser.add_argument("--moist-library", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    checkpoint = args.qm_checkpoint.expanduser().resolve(strict=True)
    mace_state = args.mace_state.expanduser().resolve(strict=True)
    density_table = args.reference_density.expanduser().resolve(strict=True)
    density_manifest = density_table.with_suffix(".json")
    literature = args.literature_pdf.expanduser().resolve(strict=True)
    extension = args.moist_extension.expanduser().resolve(strict=True)
    shared_library = args.moist_library.expanduser().resolve(strict=True)
    work_dir = args.work_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        "qm_checkpoint": _require_hash(
            checkpoint, EXPECTED_QM_CHECKPOINT_SHA256, role="QM checkpoint"
        ),
        "mace_state": _require_hash(
            mace_state, EXPECTED_MACE_STATE_SHA256, role="MACE state"
        ),
        "literature_pdf": _require_hash(
            literature, EXPECTED_PDF_SHA256, role="rho-DROP literature PDF"
        ),
        "reference_density_table": _sha256(density_table),
        "reference_density_manifest": _sha256(density_manifest),
    }

    import moist

    runtime = _runtime(extension, shared_library)
    runtime.verify_module(moist)
    qm_level = PySCFAODensityLevelSet.from_checkpoint(
        checkpoint,
        n_iso_e_per_bohr3=N_ISO_E_PER_BOHR3,
        expected_checkpoint_sha256=EXPECTED_QM_CHECKPOINT_SHA256,
    )
    positions_angstrom = np.asarray(qm_level.positions_bohr * Bohr, dtype=float)
    atomic_numbers = np.asarray(qm_level.atomic_numbers, dtype=int)
    with np.load(mace_state, allow_pickle=False) as archive:
        mace_source = np.asarray(archive["gas_density_coefficients"], dtype=float)
    if mace_source.shape != (qm_level.atom_count, 4):
        raise RuntimeError("Frozen MACE source shape differs from the QM molecule.")
    if abs(float(np.sum(mace_source[:, 0]))) > 1.0e-10:
        raise RuntimeError("Frozen acetone MACE source is not neutral.")

    asset = load_atomic_reference_density_asset(
        table_path=density_table,
        manifest_path=density_manifest,
    )
    reconstructed_level = ReconstructedMacePolarDensityLevelSet(
        asset,
        atomic_numbers,
        positions_angstrom,
        mace_source,
        N_ISO_E_PER_BOHR3,
        expected_total_charge_e=0.0,
    )
    drop_settings = MoistDropSettings()

    stage = time.perf_counter()
    reconstructed_surface = MoistDropAdapter(
        reconstructed_level,
        atomic_numbers,
        positions_angstrom,
        runtime=runtime,
        settings=drop_settings,
        moist_module=moist,
    ).build_surface()
    reconstructed_surface_seconds = time.perf_counter() - stage

    stage = time.perf_counter()
    qm_surface = MoistDropAdapter(
        qm_level,
        atomic_numbers,
        positions_angstrom,
        runtime=runtime,
        settings=drop_settings,
        level_set_state_sha256=qm_level.state_sha256,
        moist_module=moist,
    ).build_surface()
    qm_surface_seconds = time.perf_counter() - stage

    surface_metrics = _surface_metrics(
        qm_surface,
        reconstructed_surface,
        qm_level,
        reconstructed_level,
    )
    settings = RhoDropCPCMSettings(dielectric=80.0, require_exact_cold_replay=False)
    energy_matrix: dict[str, dict[str, dict[str, Any]]] = {
        "qm_total_mep": {},
        "mace_point_l1_mep": {},
    }
    surface_states = {
        "qm_ao_density_cavity": qm_surface,
        "reconstructed_mace_density_cavity": reconstructed_surface,
    }
    for cavity_name, surface_state in surface_states.items():
        qm_potential = _qm_surface_potential(
            qm_level,
            surface_state,
            work_dir / f"{cavity_name}-ao-rinv.npy",
        )
        mace_potential = point_multipole_potential(
            surface_state.snapshot.surface_points_bohr,
            positions_angstrom,
            mace_source,
        )
        energy_matrix["qm_total_mep"][cavity_name] = _solve_surface_potential(
            surface_state,
            qm_potential,
            settings=settings,
        )
        energy_matrix["mace_point_l1_mep"][cavity_name] = _solve_surface_potential(
            surface_state,
            mace_potential,
            settings=settings,
        )

    def energy(source: str, cavity: str) -> float:
        return float(energy_matrix[source][cavity]["energy_kcal_mol"])

    qm_cavity = "qm_ao_density_cavity"
    rec_cavity = "reconstructed_mace_density_cavity"
    source_error_qm_cavity = energy("mace_point_l1_mep", qm_cavity) - energy(
        "qm_total_mep", qm_cavity
    )
    source_error_rec_cavity = energy("mace_point_l1_mep", rec_cavity) - energy(
        "qm_total_mep", rec_cavity
    )
    qm_source_cavity_shift = energy("qm_total_mep", rec_cavity) - energy(
        "qm_total_mep", qm_cavity
    )
    mace_source_cavity_shift = energy("mace_point_l1_mep", rec_cavity) - energy(
        "mace_point_l1_mep", qm_cavity
    )
    decomposition = {
        "source_error_on_qm_cavity_kcal_mol": source_error_qm_cavity,
        "source_error_on_reconstructed_cavity_kcal_mol": source_error_rec_cavity,
        "reconstructed_minus_qm_cavity_shift_for_qm_source_kcal_mol": (
            qm_source_cavity_shift
        ),
        "reconstructed_minus_qm_cavity_shift_for_mace_source_kcal_mol": (
            mace_source_cavity_shift
        ),
        "source_cavity_interaction_kcal_mol": (
            source_error_rec_cavity - source_error_qm_cavity
        ),
        "identity_residual_kcal_mol": abs(
            (source_error_rec_cavity - source_error_qm_cavity)
            - (mace_source_cavity_shift - qm_source_cavity_shift)
        ),
    }

    payload = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "status": "diagnostic-complete-no-accuracy-admission",
        "claim_boundary": (
            "Fixed-geometry, label-free comparison of QM AO-density and "
            "reconstructed-MACE-density rho-DROP cavities under one MOIST "
            "C-PCM equation. It is not an experimental solvation-accuracy, "
            "Exact-GTO response, CDS, force, or production admission result."
        ),
        "compound_id": EXPECTED_COMPOUND_ID,
        "name": "acetone",
        "experimental_solvation_labels_read": False,
        "scientific_identity": {
            "geometry": "frozen QM-checkpoint acetone geometry",
            "qm_density": "omegaB97M-V/def2-TZVPD frozen AO density",
            "reconstructed_density": "frozen atomic reference minus MACE-POLAR residual l<=1 density",
            "electrostatic_sources": ["QM total MEP", "MACE-POLAR point-l<=1 MEP"],
            "continuum": "MOIST rho-DROP C-PCM",
            "dielectric": settings.dielectric,
            "n_iso_e_per_bohr3": N_ISO_E_PER_BOHR3,
            "nleb": drop_settings.nleb,
            "cds": "none",
            "standard_state": "none",
        },
        "input_sha256": input_hashes,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "moist": getattr(moist, "__version__", "unknown"),
            "moist_provenance": asdict(runtime),
            "moist_identity_sha256": runtime.identity_sha256,
        },
        "qm_density_state": {
            "state_sha256": qm_level.state_sha256,
            "density_sha256": qm_level.density_sha256,
            "electron_count_e": qm_level.electron_count_e,
            "expected_electron_count_e": qm_level.expected_electron_count_e,
        },
        "mace_source": {
            "source_sha256": array_sha256(mace_source),
            "total_charge_e": float(np.sum(mace_source[:, 0])),
        },
        "surface_metrics": surface_metrics,
        "cpcm_energy_matrix": energy_matrix,
        "decomposition": decomposition,
        "timing_seconds": {
            "reconstructed_surface": reconstructed_surface_seconds,
            "qm_surface": qm_surface_seconds,
            "total": time.perf_counter() - started,
        },
    }
    _write_json(output, payload)
    print(json.dumps({"output": str(output), **decomposition}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
