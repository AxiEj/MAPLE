#!/usr/bin/env python3
"""Run the research MACE-MDP/MACE-POLAR hybrid on four frozen PCM cases.

The source seen by PCMSolver is deliberately heterogeneous:

* the MACE-MDP zero-field permanent ``(q,p)`` source is evaluated as exterior
  point multipoles;
* only ``MACE-POLAR(R,u) - MACE-POLAR(R,0)`` is evaluated with the checkpoint's
  1.5-A Gaussian source basis;
* the PCMSolver surface charge is returned to MACE-POLAR through its audited
  two-width 1.5/3.0-A receiver.

This is a post-preregistration feasibility experiment, not an admission
artifact.  It compares the hybrid with the already defined fixed MACE-MDP and
fixed original-MACE-POLAR sources on the same frozen water cavities.  No
energy/force capability, fit, solvent transfer claim, or Tier-V claim follows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time

from ase.units import Bohr
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    PCMSolverExternalMEPCavityResponse,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
)
from maple.function.calculator.extra_correction.implicit.pcmsolver import (
    PCMSolverSession,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.coupling.exact_gto import (
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
)
from maple.solvation.models import (
    MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
    MACE_MDP_POLAR_HYBRID_PROFILE_ID,
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    sha256_file,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-mdp-polar-hybrid-pcmsolver-four-v1"
EXPECTED_MACE_POLAR_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)
PARENT_PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
PANEL_RELATIVE_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
CUTOFF_DIRECTORY = "cutoff-1e-12"
KCAL_PER_HARTREE = 627.5094740631
ROOT_TOLERANCE_EV = 1.0e-10
MAX_ROOT_ITERATIONS = 40
ROOT_REPLAY_ATOL_EV = 2.0e-9
TOTAL_CHARGE_ATOL_E = 1.0e-8
GEOMETRY_ATOL_ANGSTROM = 1.0e-8
SURFACE_REPLAY_ATOL_BOHR = 1.0e-12
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/function/calculator/extra_correction/implicit/continuum_response.py",
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/gto_galerkin.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/solvation/coupling/exact_gto.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/evidence.py",
    "tools/route2_release/run_mace_mdp_polar_hybrid_pcmsolver_panel.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
    parser.add_argument(
        "--mace-mdp-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-MDP.model",
    )
    parser.add_argument(
        "--mace-polar-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACEPOLAR1Mmodel",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load {name} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object.")
    return payload


def _rebased_asset_path(asset_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("Frozen asset path must be a non-empty string.")
    path = Path(raw_path)
    if path.is_file():
        return path.resolve()
    try:
        marker = path.parts.index(".omx")
    except ValueError as exc:
        raise RuntimeError(f"Cannot rebase frozen asset path {raw_path!r}.") from exc
    rebased = asset_root.joinpath(*path.parts[marker:]).resolve()
    if not rebased.is_file():
        raise FileNotFoundError(rebased)
    return rebased


def _validated_sha(path: Path, expected: object, *, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{name} omits a SHA256 binding.")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 mismatch: {actual} != {expected}.")
    return actual


def _configure_determinism(torch: object) -> None:
    torch.manual_seed(20260815)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260815)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _molecular_dipole(source4: np.ndarray, positions: np.ndarray) -> np.ndarray:
    # Raw real-l=1 order [m=0,m=1,m=-1] maps to Cartesian [y,z,x].
    return np.sum(source4[:, 0, None] * positions + source4[:, (3, 1, 2)], axis=0)


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(predicted - reference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def _solve_hybrid_root(
    *,
    atoms: object,
    hybrid: object,
    anchor: object,
    response: PCMSolverExternalMEPCavityResponse,
    permanent_potential_hartree: np.ndarray,
    induced_source_operator_hartree: np.ndarray,
    receiver_operator_ev: np.ndarray,
    initial_field_ev: np.ndarray,
) -> dict[str, object]:
    field = np.asarray(initial_field_ev, dtype=float).copy()
    residual_history: list[float] = []
    for iteration in range(1, MAX_ROOT_ITERATIONS + 1):
        induced = hybrid.induced_source(atoms, anchor, field)
        potential = permanent_potential_hartree + (
            induced_source_operator_hartree @ induced.reshape(-1)
        )
        charge = np.asarray(response.apply_energy_conjugate(potential), dtype=float)
        target = (receiver_operator_ev.T @ charge).reshape(field.shape)
        residual = float(np.linalg.norm(target - field))
        residual_history.append(residual)
        field = target
        if residual < ROOT_TOLERANCE_EV:
            break
    else:
        raise RuntimeError(
            f"Hybrid Picard root did not converge in {MAX_ROOT_ITERATIONS} steps."
        )

    induced = hybrid.induced_source(atoms, anchor, field)
    total_source = anchor.permanent_source4 + induced
    potential = permanent_potential_hartree + (
        induced_source_operator_hartree @ induced.reshape(-1)
    )
    charge = np.asarray(response.apply_energy_conjugate(potential), dtype=float)
    final_target = (receiver_operator_ev.T @ charge).reshape(field.shape)
    final_residual = float(np.linalg.norm(final_target - field))
    energy = 0.5 * float(np.vdot(potential, charge))
    return {
        "iterations": iteration,
        "field_ev": field,
        "induced_source4": induced,
        "total_source4": total_source,
        "surface_potential_hartree_per_e": potential,
        "surface_charge_e": charge,
        "polarization_energy_hartree": energy,
        "final_residual_ev": final_residual,
        "residual_history_ev": residual_history,
    }


def _record_case(
    *,
    record: dict[str, object],
    asset_root: Path,
    hybrid: object,
    pcmsolver_library: Path,
    radial_coupling: MACEPolarRadialGTOCoupling,
) -> dict[str, object]:
    compound_id = str(record["compound_id"])
    panel_root = asset_root / PANEL_RELATIVE_ROOT / compound_id
    result_path = panel_root / CUTOFF_DIRECTORY / "result.json"
    work = panel_root / CUTOFF_DIRECTORY / "work"
    result = _load_json(result_path, name=f"{compound_id} projection result")
    inputs = result.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("compound_id") != compound_id:
        raise RuntimeError(f"{compound_id} result input identity is invalid.")

    mol2 = asset_root / str(record["mol2_path"])
    _validated_sha(mol2, record.get("mol2_sha256"), name=f"{compound_id} MOL2")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    qm_path = work / "qm-surface-mep.npz"
    surface_path = work / "surface.npz"
    with np.load(qm_path) as qm:
        qm_potential = np.asarray(qm["surface_potential_hartree_per_e"], dtype=float)
        qm_positions = np.asarray(qm["atom_positions_angstrom"], dtype=float)
        qm_numbers = np.asarray(qm["atomic_numbers"], dtype=int)
        qm_dipole = np.asarray(qm["molecular_dipole_e_angstrom"], dtype=float)
        qm_total_charge = float(qm["total_charge_e"])
    if not np.array_equal(qm_numbers, atoms.numbers):
        raise RuntimeError(f"{compound_id} QM and MOL2 elements differ.")
    geometry_error = float(np.max(np.abs(qm_positions - atoms.positions)))
    if geometry_error > GEOMETRY_ATOL_ANGSTROM:
        raise RuntimeError(f"{compound_id} QM/MOL2 geometry mismatch.")

    pcm_record = inputs.get("parsed_pcm_input")
    if not isinstance(pcm_record, dict):
        raise RuntimeError(f"{compound_id} result omits its PCM input record.")
    pcm_input = _rebased_asset_path(asset_root, pcm_record.get("path"))
    _validated_sha(pcm_input, pcm_record.get("sha256"), name=f"{compound_id} PCM input")
    radii = np.asarray(pcm_record.get("cavity_radii_angstrom"), dtype=float)
    if radii.shape != (len(atoms),) or np.any(radii <= 0.0):
        raise RuntimeError(f"{compound_id} PCM radii are invalid.")

    anchor = hybrid.prepare(atoms)
    frozen_surface = np.asarray(
        np.load(surface_path)["surface_points_bohr"], dtype=float
    )
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"route2-hybrid-{compound_id}-") as tmp:
        try:
            os.chdir(tmp)
            with PCMSolverSession(
                np.asarray(atoms.numbers, dtype=float),
                atoms.positions / Bohr,
                pcm_input,
                library_path=pcmsolver_library,
            ) as session:
                points = np.asarray(session.cavity_centers_bohr, dtype=float)
                surface_error = float(np.max(np.abs(points - frozen_surface)))
                if surface_error > SURFACE_REPLAY_ATOL_BOHR:
                    raise RuntimeError(f"{compound_id} PCMSolver surface drifted.")
                response = PCMSolverExternalMEPCavityResponse(
                    session, cavity_radii_angstrom=radii
                )
                induced_operator = AtomCenteredL1GTOBasis((1.5,)).surface_operator(
                    points, atoms.positions
                )
                receiver_operator = radial_coupling.surface_operator(
                    FixedSurfaceGeometry(atoms.positions, points)
                )
                permanent_potential = point_multipole_potential(
                    points, atoms.positions, anchor.permanent_source4
                )
                polar_zero_potential = induced_operator @ (
                    anchor.response_zero_source4.reshape(-1)
                )

                def surface_energy(potential: np.ndarray) -> tuple[np.ndarray, float]:
                    charge = np.asarray(
                        response.apply_energy_conjugate(potential), dtype=float
                    )
                    return charge, 0.5 * float(np.vdot(potential, charge))

                qm_charge, qm_energy = surface_energy(qm_potential)
                mdp_charge, mdp_energy = surface_energy(permanent_potential)
                _polar_charge, polar_energy = surface_energy(polar_zero_potential)
                permanent_field = (receiver_operator.T @ mdp_charge).reshape(
                    len(atoms), 8
                )
                cold = _solve_hybrid_root(
                    atoms=atoms,
                    hybrid=hybrid,
                    anchor=anchor,
                    response=response,
                    permanent_potential_hartree=permanent_potential,
                    induced_source_operator_hartree=induced_operator,
                    receiver_operator_ev=receiver_operator,
                    initial_field_ev=np.zeros((len(atoms), 8)),
                )
                wide = _solve_hybrid_root(
                    atoms=atoms,
                    hybrid=hybrid,
                    anchor=anchor,
                    response=response,
                    permanent_potential_hartree=permanent_potential,
                    induced_source_operator_hartree=induced_operator,
                    receiver_operator_ev=receiver_operator,
                    initial_field_ev=2.0 * permanent_field,
                )
                areas = np.asarray(session.cavity_areas_bohr2, dtype=float)
        finally:
            os.chdir(previous)

    field_replay_error = float(np.max(np.abs(cold["field_ev"] - wide["field_ev"])))
    energy_replay_error = abs(
        float(cold["polarization_energy_hartree"])
        - float(wide["polarization_energy_hartree"])
    )
    if field_replay_error > ROOT_REPLAY_ATOL_EV:
        raise RuntimeError(f"{compound_id} hybrid multi-start roots disagree.")
    hybrid_potential = np.asarray(cold["surface_potential_hartree_per_e"], dtype=float)
    potential_error = hybrid_potential - qm_potential
    area_relative_mep = float(
        np.sqrt(
            np.sum(areas * potential_error * potential_error)
            / np.sum(areas * qm_potential * qm_potential)
        )
    )
    total_source = np.asarray(cold["total_source4"], dtype=float)
    hybrid_dipole = _molecular_dipole(total_source, atoms.positions)
    hybrid_energy = float(cold["polarization_energy_hartree"])
    total_charge = float(np.sum(total_source[:, 0]))
    target_from_prior = float(
        result["basis_results"]["one_radial"]["target_polarization_energy_hartree"]
    )
    if abs(qm_energy - target_from_prior) > 2.0e-13:
        raise RuntimeError(f"{compound_id} target polarization energy drifted.")
    return {
        "compound_id": compound_id,
        "name": str(record["name"]),
        "class": str(record["class"]),
        "atom_count": len(atoms),
        "geometry_max_abs_error_angstrom": geometry_error,
        "surface_replay_max_abs_error_bohr": surface_error,
        "qm_total_charge_e": qm_total_charge,
        "hybrid_total_charge_e": total_charge,
        "total_charge_absolute_error_e": abs(total_charge - qm_total_charge),
        "qm_dipole_e_angstrom": qm_dipole.tolist(),
        "hybrid_dipole_e_angstrom": hybrid_dipole.tolist(),
        "hybrid_dipole_relative_l2_error": _relative_l2(hybrid_dipole, qm_dipole),
        "hybrid_surface_mep_area_weighted_relative_l2_error": area_relative_mep,
        "qm_polarization_energy_kcal_per_mol": qm_energy * KCAL_PER_HARTREE,
        "mace_mdp_fixed_polarization_energy_kcal_per_mol": (
            mdp_energy * KCAL_PER_HARTREE
        ),
        "mace_polar_fixed_polarization_energy_kcal_per_mol": (
            polar_energy * KCAL_PER_HARTREE
        ),
        "hybrid_polarization_energy_kcal_per_mol": (hybrid_energy * KCAL_PER_HARTREE),
        "mace_mdp_fixed_absolute_error_kcal_per_mol": (
            abs(mdp_energy - qm_energy) * KCAL_PER_HARTREE
        ),
        "mace_polar_fixed_absolute_error_kcal_per_mol": (
            abs(polar_energy - qm_energy) * KCAL_PER_HARTREE
        ),
        "hybrid_absolute_error_kcal_per_mol": (
            abs(hybrid_energy - qm_energy) * KCAL_PER_HARTREE
        ),
        "hybrid_improvement_vs_mdp_kcal_per_mol": (
            (abs(mdp_energy - qm_energy) - abs(hybrid_energy - qm_energy))
            * KCAL_PER_HARTREE
        ),
        "induced_source_l2_norm": float(np.linalg.norm(cold["induced_source4"])),
        "cold_root_iterations": int(cold["iterations"]),
        "wide_root_iterations": int(wide["iterations"]),
        "cold_final_residual_ev": float(cold["final_residual_ev"]),
        "wide_final_residual_ev": float(wide["final_residual_ev"]),
        "multi_start_field_max_abs_difference_ev": field_replay_error,
        "multi_start_energy_abs_difference_hartree": energy_replay_error,
        "cold_residual_history_ev": cold["residual_history_ev"],
        "wide_residual_history_ev": wide["residual_history_ev"],
        "charge_gate_passed": bool(
            abs(total_charge - qm_total_charge) <= TOTAL_CHARGE_ATOL_E
        ),
        "root_gate_passed": bool(
            float(cold["final_residual_ev"]) < ROOT_TOLERANCE_EV
            and float(wide["final_residual_ev"]) < ROOT_TOLERANCE_EV
            and field_replay_error <= ROOT_REPLAY_ATOL_EV
        ),
        "asset_sha256": {
            "mol2": sha256_file(mol2),
            "projection_result": sha256_file(result_path),
            "qm_surface_mep": sha256_file(qm_path),
            "surface_points": sha256_file(surface_path),
            "pcm_input": sha256_file(pcm_input),
        },
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    started = time.perf_counter()
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    pcmsolver_library = args.pcmsolver_library.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mace_mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.mace_polar_checkpoint.expanduser().resolve(strict=True)
    if sha256_file(mdp_checkpoint) != MACE_MDP_EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-MDP checkpoint does not match the frozen model.")
    if sha256_file(polar_checkpoint) != EXPECTED_MACE_POLAR_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-POLAR checkpoint does not match the frozen model.")

    preregistration_path = repository.root / PARENT_PREREGISTRATION_RELATIVE_PATH
    preregistration = _load_json(
        preregistration_path, name="parent PCM preregistration"
    )
    execution_contract = preregistration.get("execution_contract")
    records = preregistration.get("records")
    if not isinstance(execution_contract, dict) or not isinstance(records, list):
        raise RuntimeError("Parent preregistration omits its panel contract.")
    _validated_sha(
        pcmsolver_library,
        execution_contract.get("pcmsolver_library_sha256"),
        name="PCMSolver shared library",
    )
    if len(records) != 4 or not all(isinstance(record, dict) for record in records):
        raise RuntimeError("Expected the frozen four-record PCM panel.")

    import torch

    _configure_determinism(torch)
    mdp = build_mace_mdp_moment_adapter(checkpoint_path=mdp_checkpoint, device="cpu")
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
    )
    radial_coupling = MACEPolarRadialGTOCoupling()
    case_records = [
        _record_case(
            record=record,
            asset_root=asset_root,
            hybrid=hybrid,
            pcmsolver_library=pcmsolver_library,
            radial_coupling=radial_coupling,
        )
        for record in sorted(records, key=lambda value: str(value["compound_id"]))
    ]
    mdp_errors = [
        float(record["mace_mdp_fixed_absolute_error_kcal_per_mol"])
        for record in case_records
    ]
    polar_errors = [
        float(record["mace_polar_fixed_absolute_error_kcal_per_mol"])
        for record in case_records
    ]
    hybrid_errors = [
        float(record["hybrid_absolute_error_kcal_per_mol"]) for record in case_records
    ]
    improved = [
        float(record["hybrid_improvement_vs_mdp_kcal_per_mol"]) > 0.0
        for record in case_records
    ]
    aggregate = {
        "record_count": len(case_records),
        "root_and_charge_pass_count": sum(
            bool(record["root_gate_passed"] and record["charge_gate_passed"])
            for record in case_records
        ),
        "hybrid_improved_case_count": sum(improved),
        "hybrid_worsened_case_count": len(improved) - sum(improved),
        "mace_mdp_fixed_mean_absolute_error_kcal_per_mol": float(np.mean(mdp_errors)),
        "mace_mdp_fixed_maximum_absolute_error_kcal_per_mol": max(mdp_errors),
        "mace_polar_fixed_mean_absolute_error_kcal_per_mol": float(
            np.mean(polar_errors)
        ),
        "hybrid_mean_absolute_error_kcal_per_mol": float(np.mean(hybrid_errors)),
        "hybrid_maximum_absolute_error_kcal_per_mol": max(hybrid_errors),
        "hybrid_case_count_below_1_kcal_per_mol": sum(
            error < 1.0 for error in hybrid_errors
        ),
        "maximum_multi_start_field_difference_ev": max(
            float(record["multi_start_field_max_abs_difference_ev"])
            for record in case_records
        ),
    }
    measurement = {
        "protocol": {
            "profile_id": MACE_MDP_POLAR_HYBRID_PROFILE_ID,
            "permanent_source": "frozen MACE-MDP atom-centred point q/p",
            "induced_source": ("MACE-POLAR(R,u)-MACE-POLAR(R,0), 1.5-A Gaussian q/p"),
            "receiver": "MACE-POLAR physical radial 1.5/3.0-A field8",
            "continuum": preregistration["method"]["continuum"],
            "solvent_scope": "water only; frozen epsilon/cavity/input per case",
            "root_method": "undamped deterministic Picard",
            "root_tolerance_ev": ROOT_TOLERANCE_EV,
            "maximum_root_iterations": MAX_ROOT_ITERATIONS,
            "second_start": "twice the permanent-source reaction field",
            "fit_or_calibration": False,
            "formal_preregistration": False,
        },
        "records": case_records,
        "aggregate": aggregate,
        "decision": {
            "hybrid_feasibility_demonstrated": bool(
                aggregate["root_and_charge_pass_count"] == len(case_records)
            ),
            "uniform_accuracy_improvement_demonstrated": bool(all(improved)),
            "multi_solvent_transfer_demonstrated": False,
            "force_available": False,
            "tier_v_available": False,
            "public_capability_admitted": False,
        },
    }
    measurement_sha256 = canonical_json_sha256(measurement)
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "post-hoc-research-hybrid-feasibility-panel",
        "status": "completed-no-capability-admission",
        "claim_boundary": (
            "This four-case water panel demonstrates that the frozen MACE-MDP "
            "permanent source and MACE-POLAR induced increment can be iterated "
            "self-consistently. It was designed after observing a preliminary "
            "acetone probe, is not a preregistered accuracy gate, and does not "
            "establish uniform improvement, solvent transfer, a physical total "
            "solvation free energy, conservative force, or Tier V."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": committed_source_hashes(repository, source_paths),
        "external_assets": {
            "asset_root": str(asset_root),
            "parent_preregistration": {
                "path": str(preregistration_path),
                "sha256": sha256_file(preregistration_path),
                "artifact_id": preregistration["artifact_id"],
            },
            "mace_mdp_checkpoint": checkpoint_record(mdp_checkpoint),
            "mace_polar_checkpoint": checkpoint_record(polar_checkpoint),
            "pcmsolver_library": {
                "path": str(pcmsolver_library),
                "sha256": sha256_file(pcmsolver_library),
            },
        },
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": str(radial.dtype).replace("torch.", ""),
        **measurement,
        "measurement_sha256": measurement_sha256,
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_MACE_MDP_POLAR_HYBRID="
        + json.dumps(
            {
                "artifact": artifact,
                "aggregate": aggregate,
                "decision": payload["decision"],
                "measurement_sha256": measurement_sha256,
                "capabilities": NO_CAPABILITIES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
