#!/usr/bin/env python3
"""Terminally audit the frozen MACE-MDP permanent atomic moment source.

The checkpoint's atom-resolved charges and dipoles are used unchanged as
exterior point multipoles and compared with total QM molecular electrostatic
potentials on four identical intrinsic PCMSolver cavities.  No radial width,
source fit, solvation label, energy ledger, or capability is selected here.
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
from maple.function.calculator.extra_correction.implicit.pcmsolver import (
    PCMSolverSession,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.solvation.models import (
    MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
    MACE_MDPMomentAdapter,
    build_mace_mdp_moment_adapter,
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

SCHEMA_VERSION = "route2-mace-mdp-permanent-source-pcmsolver-four-v1"
PREREGISTRATION_RELATIVE_PATH = (
    "docs/route2/preregistrations/" "mace-mdp-permanent-source-pcmsolver-four-v1.json"
)
PARENT_PCM_PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
PANEL_RELATIVE_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
CUTOFF_DIRECTORY = "cutoff-1e-12"
KCAL_PER_HARTREE = 627.5094740631
INHERITED_FIXED_SOURCE_ERROR_BUDGET_KCAL_PER_MOL = 1.0
TOTAL_CHARGE_ATOL_E = 1.0e-8
GEOMETRY_ATOL_ANGSTROM = 1.0e-8
SURFACE_REPLAY_ATOL_BOHR = 1.0e-12
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/function/calculator/extra_correction/implicit/continuum_response.py",
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/release/evidence.py",
    "maple/solvation/release/source_mep.py",
    "tools/route2_release/run_mace_mdp_permanent_source_pcmsolver_panel.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-MDP.model",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
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
    parts = path.parts
    try:
        marker = parts.index(".omx")
    except ValueError as exc:
        raise RuntimeError(f"Cannot rebase frozen asset path {raw_path!r}.") from exc
    rebased = asset_root.joinpath(*parts[marker:]).resolve()
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


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(predicted - reference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def _record_case(
    *,
    record: dict[str, object],
    asset_root: Path,
    electronic: MACE_MDPMomentAdapter,
    pcmsolver_library: Path,
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
        qm_density = {
            "source": str(qm["density_source"]),
            "checkpoint_density_binding_residual_e": float(
                qm["checkpoint_density_binding_residual_e"]
            ),
            "checkpoint_mo_orthonormality_inf": float(
                qm["checkpoint_mo_orthonormality_inf"]
            ),
            "checkpoint_occupation_sum_e": float(qm["checkpoint_occupation_sum_e"]),
        }
    if qm_numbers.shape != atoms.numbers.shape or not np.array_equal(
        qm_numbers, atoms.numbers
    ):
        raise RuntimeError(f"{compound_id} QM and MOL2 elements differ.")
    geometry_error = float(np.max(np.abs(qm_positions - atoms.positions)))
    if geometry_error > GEOMETRY_ATOL_ANGSTROM:
        raise RuntimeError(f"{compound_id} QM/MOL2 geometry mismatch.")

    qm_checkpoint_record = inputs.get("qm_checkpoint")
    pcm_input_record = inputs.get("parsed_pcm_input")
    if not isinstance(qm_checkpoint_record, dict) or not isinstance(
        pcm_input_record, dict
    ):
        raise RuntimeError(f"{compound_id} result omits frozen input records.")
    qm_checkpoint = _rebased_asset_path(asset_root, qm_checkpoint_record.get("path"))
    pcm_input = _rebased_asset_path(asset_root, pcm_input_record.get("path"))
    _validated_sha(
        qm_checkpoint,
        qm_checkpoint_record.get("sha256"),
        name=f"{compound_id} QM checkpoint",
    )
    _validated_sha(
        pcm_input,
        pcm_input_record.get("sha256"),
        name=f"{compound_id} PCM input",
    )
    radii = np.asarray(pcm_input_record.get("cavity_radii_angstrom"), dtype=float)
    if radii.shape != (len(atoms),) or np.any(radii <= 0.0):
        raise RuntimeError(f"{compound_id} PCM radii are invalid.")

    state = electronic.evaluate(atoms)
    source = np.asarray(state.source4_raw_l1, dtype=float)
    predicted_total_charge = state.total_charge_e
    predicted_dipole = np.asarray(state.public_dipole_eangstrom, dtype=float)

    frozen_surface = np.asarray(
        np.load(surface_path)["surface_points_bohr"], dtype=float
    )
    with tempfile.TemporaryDirectory(prefix=f"route2-source-{compound_id}-") as workdir:
        previous = Path.cwd()
        try:
            os.chdir(workdir)
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
                predicted_potential = point_multipole_potential(
                    points, atoms.positions, source
                )
                qm_charge = np.asarray(
                    response.apply_energy_conjugate(qm_potential), dtype=float
                )
                predicted_charge = np.asarray(
                    response.apply_energy_conjugate(predicted_potential), dtype=float
                )
                potential_error = predicted_potential - qm_potential
                error_charge = np.asarray(
                    response.apply_energy_conjugate(potential_error), dtype=float
                )
                areas = np.asarray(session.cavity_areas_bohr2, dtype=float)
        finally:
            os.chdir(previous)

    qm_energy = 0.5 * float(np.vdot(qm_potential, qm_charge))
    predicted_energy = 0.5 * float(np.vdot(predicted_potential, predicted_charge))
    reference_norm = float(np.sqrt(max(0.0, -float(np.vdot(qm_potential, qm_charge)))))
    error_norm = float(
        np.sqrt(max(0.0, -float(np.vdot(potential_error, error_charge))))
    )
    energy_error = abs(predicted_energy - qm_energy)
    energy_error_kcal = energy_error * KCAL_PER_HARTREE
    energy_bound = reference_norm * error_norm + 0.5 * error_norm * error_norm
    if energy_error > energy_bound + 2.0e-12:
        raise RuntimeError(f"{compound_id} response-norm bound is inconsistent.")
    area_relative_mep = float(
        np.sqrt(
            np.sum(areas * potential_error * potential_error)
            / np.sum(areas * qm_potential * qm_potential)
        )
    )
    target_from_prior = float(
        result["basis_results"]["one_radial"]["target_polarization_energy_hartree"]
    )
    if abs(qm_energy - target_from_prior) > 2.0e-13:
        raise RuntimeError(f"{compound_id} target polarization energy drifted.")
    charge_passed = bool(
        abs(predicted_total_charge - qm_total_charge) <= TOTAL_CHARGE_ATOL_E
    )
    energy_passed = bool(
        energy_error_kcal < INHERITED_FIXED_SOURCE_ERROR_BUDGET_KCAL_PER_MOL
    )
    return {
        "compound_id": compound_id,
        "name": str(record["name"]),
        "class": str(record["class"]),
        "atom_count": len(atoms),
        "asset_sha256": {
            "mol2": sha256_file(mol2),
            "projection_result": sha256_file(result_path),
            "qm_surface_mep": sha256_file(qm_path),
            "surface_points": sha256_file(surface_path),
            "qm_checkpoint": sha256_file(qm_checkpoint),
            "pcm_input": sha256_file(pcm_input),
        },
        "geometry_max_abs_error_angstrom": geometry_error,
        "surface_replay_max_abs_error_bohr": surface_error,
        "qm_density": qm_density,
        "qm_total_charge_e": qm_total_charge,
        "predicted_total_charge_e": predicted_total_charge,
        "total_charge_absolute_error_e": abs(predicted_total_charge - qm_total_charge),
        "qm_dipole_e_angstrom": qm_dipole.tolist(),
        "predicted_dipole_e_angstrom": predicted_dipole.tolist(),
        "mace_mdp_state_sha256": state.state_sha256,
        "mace_mdp_model_input_sha256": state.model_input_sha256,
        "mace_mdp_polarizability_bohr3": state.polarizability_bohr3.tolist(),
        "mace_mdp_polarizability_eigenvalues_bohr3": np.linalg.eigvalsh(
            state.polarizability_bohr3
        ).tolist(),
        "dipole_relative_l2_error": _relative_l2(predicted_dipole, qm_dipole),
        "surface_mep_relative_l2_error": _relative_l2(
            predicted_potential, qm_potential
        ),
        "surface_mep_area_weighted_relative_l2_error": area_relative_mep,
        "qm_polarization_energy_hartree": qm_energy,
        "predicted_polarization_energy_hartree": predicted_energy,
        "qm_polarization_energy_kcal_per_mol": qm_energy * KCAL_PER_HARTREE,
        "predicted_polarization_energy_kcal_per_mol": (
            predicted_energy * KCAL_PER_HARTREE
        ),
        "polarization_energy_absolute_error_kcal_per_mol": energy_error_kcal,
        "reference_response_norm_sqrt_hartree": reference_norm,
        "error_response_norm_sqrt_hartree": error_norm,
        "polarization_energy_error_upper_bound_kcal_per_mol": (
            energy_bound * KCAL_PER_HARTREE
        ),
        "charge_gate_passed": charge_passed,
        "fixed_source_energy_gate_passed": energy_passed,
        "case_passed": bool(charge_passed and energy_passed),
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    started = time.perf_counter()
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    pcmsolver_library = args.pcmsolver_library.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != MACE_MDP_EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-MDP checkpoint does not match the frozen model.")

    preregistration_path = repository.root / PREREGISTRATION_RELATIVE_PATH
    preregistration = _load_json(preregistration_path, name="preregistration")
    if (
        preregistration.get("artifact_id")
        != "route2-mace-mdp-permanent-source-pcmsolver-four-prereg-v1"
        or preregistration.get("status") != "frozen-before-transfer-execution"
    ):
        raise RuntimeError("Frozen four-case preregistration identity is invalid.")
    candidate = preregistration.get("candidate")
    gates = preregistration.get("admission_gates")
    if not isinstance(candidate, dict) or not isinstance(gates, dict):
        raise RuntimeError("Preregistration omits candidate identity or gates.")
    checkpoint_record_data = candidate.get("checkpoint")
    if (
        not isinstance(checkpoint_record_data, dict)
        or checkpoint_record_data.get("sha256") != MACE_MDP_EXPECTED_CHECKPOINT_SHA256
        or checkpoint_record_data.get("model_type") != "DipolePolarizabilityMACE"
    ):
        raise RuntimeError("Preregistered MACE-MDP checkpoint identity is invalid.")
    if (
        float(
            gates.get(
                "fixed_source_polarization_energy_absolute_error_max_kcal_per_mol_per_case",
                float("nan"),
            )
        )
        != INHERITED_FIXED_SOURCE_ERROR_BUDGET_KCAL_PER_MOL
    ):
        raise RuntimeError("Preregistered fixed-source energy gate drifted.")
    execution_contract = preregistration.get("execution_contract")
    if not isinstance(execution_contract, dict):
        raise RuntimeError("Preregistration omits its execution contract.")
    parent_path = repository.root / PARENT_PCM_PREREGISTRATION_RELATIVE_PATH
    parent = _load_json(parent_path, name="parent PCMSolver preregistration")
    parent_record = preregistration.get("parent_pcmsolver_preregistration")
    if not isinstance(parent_record, dict):
        raise RuntimeError("Preregistration omits its parent panel binding.")
    _validated_sha(
        parent_path,
        parent_record.get("sha256"),
        name="parent PCMSolver preregistration",
    )
    evidence = preregistration.get("prior_coefficient_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("Preregistration omits prior coefficient evidence.")
    for label, record in evidence.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Prior evidence {label!r} is invalid.")
        path = repository.root / str(record.get("path"))
        _validated_sha(path, record.get("sha256"), name=f"prior evidence {label}")
    records = parent.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Parent preregistration omits its panel.")
    expected_pcm_sha = execution_contract.get("pcmsolver_library_sha256")
    _validated_sha(
        pcmsolver_library,
        expected_pcm_sha,
        name="PCMSolver shared library",
    )
    if len(records) != 4 or not all(isinstance(record, dict) for record in records):
        raise RuntimeError("Expected the frozen four-record source panel.")

    import torch

    _configure_determinism(torch)
    electronic = build_mace_mdp_moment_adapter(
        checkpoint_path=checkpoint,
        device=args.device,
    )
    case_records = [
        _record_case(
            record=record,
            asset_root=asset_root,
            electronic=electronic,
            pcmsolver_library=pcmsolver_library,
        )
        for record in sorted(records, key=lambda value: str(value["compound_id"]))
    ]

    case_pass_count = sum(bool(record["case_passed"]) for record in case_records)
    source_admitted = bool(case_pass_count == len(case_records))
    measurement = {
        "protocol": {
            "source": (
                "unchanged frozen MACE-MDP atom-resolved permanent charges and "
                "dipoles"
            ),
            "source_representation": "exterior point multipoles through l<=1",
            "reference": "frozen total QM MEP on identical PCMSolver cavity",
            "continuum": parent["method"]["continuum"],
            "fixed_source_energy_error_budget_kcal_per_mol": (
                INHERITED_FIXED_SOURCE_ERROR_BUDGET_KCAL_PER_MOL
            ),
            "threshold_provenance": (
                "inherited unchanged from the earlier frozen representation-"
                "feasibility preregistration; not fitted to these source results"
            ),
            "total_charge_atol_e": TOTAL_CHARGE_ATOL_E,
            "geometry_atol_angstrom": GEOMETRY_ATOL_ANGSTROM,
            "surface_replay_atol_bohr": SURFACE_REPLAY_ATOL_BOHR,
        },
        "records": case_records,
        "aggregate": {
            "record_count": len(case_records),
            "case_pass_count": case_pass_count,
            "maximum_dipole_relative_l2_error": max(
                float(record["dipole_relative_l2_error"]) for record in case_records
            ),
            "maximum_surface_mep_area_weighted_relative_l2_error": max(
                float(record["surface_mep_area_weighted_relative_l2_error"])
                for record in case_records
            ),
            "mean_polarization_energy_absolute_error_kcal_per_mol": float(
                np.mean(
                    [
                        float(record["polarization_energy_absolute_error_kcal_per_mol"])
                        for record in case_records
                    ]
                )
            ),
            "maximum_polarization_energy_absolute_error_kcal_per_mol": max(
                float(record["polarization_energy_absolute_error_kcal_per_mol"])
                for record in case_records
            ),
        },
        "decision": {
            "mace_mdp_permanent_source_necessary_gate_passed": source_admitted,
            "broader_static_mep_panel_authorized": source_admitted,
            "independent_variational_polarization_kkt_authorized": source_admitted,
            "additional_source_patch_authorized": False,
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
        "artifact_kind": "disabled-independent-polarization-source-terminal-audit",
        "status": (
            "mace-mdp-permanent-source-passes-frozen-four-case-gate"
            if source_admitted
            else "mace-mdp-permanent-source-fails-frozen-four-case-gate"
        ),
        "claim_boundary": (
            "This artifact is a necessary permanent-source falsifier for a new "
            "MACE-MDP-based independent quadratic polarization model. It does "
            "not turn MACE-MDP into an energy model, admit a KKT scalar, select "
            "a solvation ledger, or admit E/F/H/V/M."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": committed_source_hashes(repository, source_paths),
        "external_assets": {
            "asset_root": str(asset_root),
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": sha256_file(preregistration_path),
                "artifact_id": preregistration["artifact_id"],
            },
            "mace_mdp_checkpoint": checkpoint_record(checkpoint),
            "pcmsolver_library": {
                "path": str(pcmsolver_library),
                "sha256": sha256_file(pcmsolver_library),
            },
        },
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": "float64",
        **measurement,
        "measurement_sha256": measurement_sha256,
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_MACE_MDP_PERMANENT_SOURCE_PCMSOLVER="
        + json.dumps(
            {
                "artifact": artifact,
                "status": payload["status"],
                "aggregate": payload["aggregate"],
                "decision": payload["decision"],
                "measurement_sha256": measurement_sha256,
                "capabilities": NO_CAPABILITIES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
