#!/usr/bin/env python3
"""Run the frozen chemistry-diverse FreeSolv-20 Torch COSMO-RS diagnostic."""

from __future__ import annotations

from dataclasses import asdict, replace

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.function.cosmors_torch.cosmospace import (
    OPEN_COSMORS_24A_PARAMETERS,
)  # noqa: E402
from maple.function.cosmors_torch.fixed_structure import _load_species  # noqa: E402
from maple.function.cosmors_torch.surface import write_sigma_profile  # noqa: E402
from maple.function.cosmors_torch.thermodynamics import (  # noqa: E402
    open24a_solvation_free_energy,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

WATER_POSITIONS_ANGSTROM = np.asarray(
    (
        (0.000000, 0.000000, 0.000000),
        (0.957200, 0.000000, 0.000000),
        (-0.239987, 0.927297, 0.000000),
    ),
    dtype=np.float64,
)

CORE_SOURCE_RELATIVE_PATHS = (
    "docs/implicit-solvation/benchmarks/run_mace_ef_cosmors_freesolv20.py",
    "maple/function/calculator/extra_correction/implicit/electrostatic_pairing.py",
    "maple/function/calculator/extra_correction/implicit/mace_polar_ef.py",
    "maple/function/calculator/extra_correction/implicit/mace_polar_ef_specs.py",
    "maple/function/calculator/extra_correction/implicit/mace_polar_ef_stationary.py",
    "maple/function/calculator/extra_correction/implicit/route2_fixed_point.py",
    "maple/function/cosmors_torch/cosmospace.py",
    "maple/function/cosmors_torch/fixed_structure.py",
    "maple/function/cosmors_torch/ionic_es.py",
    "maple/function/cosmors_torch/kse.py",
    "maple/function/cosmors_torch/mace_ef_segment_cosmo.py",
    "maple/function/cosmors_torch/segment_cosmo.py",
    "maple/function/cosmors_torch/surface.py",
    "maple/function/cosmors_torch/thermodynamics.py",
    "maple/function/mlip_cosmo_rs.py",
    "maple/function/read/filereader/mol2_reader.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_artifact_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Benchmark source {relative_path!r} is not committed at {commit}: "
            f"{diagnostic}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _bind_source_files(
    additional_paths: tuple[Path, ...] = (),
) -> tuple[str, dict[str, str]]:
    """Fail closed unless every provenance-bearing source equals HEAD."""

    head = _git_head()
    paths = [REPOSITORY_ROOT / item for item in CORE_SOURCE_RELATIVE_PATHS]
    paths.extend(additional_paths)
    hashes: dict[str, str] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        try:
            relative = resolved.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                "Provenance-bearing benchmark sources must live in the repository."
            ) from exc
        observed = _sha256_file(resolved)
        committed = _git_blob_sha256(head, relative)
        if observed != committed:
            raise RuntimeError(
                f"Benchmark source {relative!r} differs from committed HEAD; "
                "commit the protocol before executing it."
            )
        hashes[relative] = observed
    return head, dict(sorted(hashes.items()))


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "ase": _package_version("ase"),
        "rdkit": _package_version("rdkit"),
        "mace_torch": _package_version("mace-torch"),
    }


def _validate_selection(selection: dict[str, Any]) -> list[dict[str, Any]]:
    if selection.get("schema_version") != 1:
        raise ValueError("The frozen FreeSolv selection schema must be version 1.")
    selection_id = selection.get("selection_id")
    if not isinstance(selection_id, str) or not selection_id.strip():
        raise ValueError("The frozen FreeSolv selection needs a nonempty ID.")
    if not str(selection.get("status", "")).startswith("frozen-"):
        raise ValueError("The FreeSolv selection must be frozen before execution.")
    policy = selection.get("selection_policy")
    if not isinstance(policy, dict):
        raise ValueError("The FreeSolv selection policy is missing.")
    if policy.get("candidate_predictions_used") is not False:
        raise ValueError("Candidate predictions must not be used for selection.")
    if policy.get("experimental_values_used_to_rank_candidates") is not False:
        raise ValueError("Experimental values must not rank selection candidates.")
    records = [dict(item) for item in selection["records"]]
    if len(records) != 20:
        raise ValueError(
            "The frozen FreeSolv diagnostic must contain exactly 20 records."
        )
    ids = [str(item["compound_id"]) for item in records]
    classes = [str(item["chemical_class"]) for item in records]
    if len(set(ids)) != len(ids) or len(set(classes)) != len(classes):
        raise ValueError("FreeSolv IDs and chemical classes must both be unique.")
    if [int(item["ordinal"]) for item in records] != list(range(1, 21)):
        raise ValueError("Selection ordinals must be exactly 1 through 20.")
    return records


def _selected_record_manifest(
    selected: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for locked in selected:
        candidate = candidates[str(locked["compound_id"])]
        records.append(
            {
                "ordinal": int(locked["ordinal"]),
                "compound_id": str(locked["compound_id"]),
                "chemical_class": str(locked["chemical_class"]),
                "ring_atom_count": int(locked["ring_atom_count"]),
                "mol2_relative_path": str(candidate["mol2_relative_path"]),
                "mol2_sha256": str(candidate["mol2_sha256"]),
                "dataset_record_sha256": str(candidate["dataset_record_sha256"]),
                "structure_group_sha256": str(candidate["structure_group_sha256"]),
                "experimental_kcal_mol": float(candidate["experimental_kcal_mol"]),
                "experimental_uncertainty_kcal_mol": float(
                    candidate["experimental_uncertainty_kcal_mol"]
                ),
            }
        )
    return records


def _validate_prepared_manifest(
    selection: dict[str, Any],
    selected: list[dict[str, Any]],
    prepared: dict[str, Any],
    *,
    prepared_path: Path,
    dataset_dir: Path,
) -> tuple[dict[str, dict[str, Any]], str]:
    manifest = selection.get("prepared_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("A clean replay selection must bind prepared_manifest.")
    expected_prepared_sha = str(manifest.get("sha256", ""))
    if _sha256_file(prepared_path) != expected_prepared_sha:
        raise ValueError("Prepared FreeSolv manifest SHA256 drifted from the lock.")
    expected_count = int(selection["dataset"]["expected_record_count"])
    if (
        int(prepared.get("candidate_count", -1)) != expected_count
        or len(prepared.get("candidates", ())) != expected_count
    ):
        raise ValueError(
            "Prepared FreeSolv population does not match the selection lock."
        )
    candidate_items = [dict(item) for item in prepared["candidates"]]
    candidate_ids = [str(item["compound_id"]) for item in candidate_items]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Prepared FreeSolv compound IDs must be unique.")
    candidates = dict(zip(candidate_ids, candidate_items, strict=True))
    if any(str(item["compound_id"]) not in candidates for item in selected):
        raise ValueError("The selection contains an ID absent from prepared FreeSolv.")

    expected_commit = str(selection["dataset"]["repository_commit"])
    checks = {
        "dataset_commit": expected_commit,
        "protocol_id": str(manifest["protocol_id"]),
        "protocol_fingerprint": str(manifest["protocol_fingerprint"]),
    }
    for key, expected in checks.items():
        if prepared.get(key) != expected:
            raise ValueError(
                f"Prepared FreeSolv {key} drifted from the selection lock."
            )
    expected_artifacts = dict(manifest["dataset_artifact_sha256"])
    if prepared.get("dataset_artifact_sha256") != expected_artifacts:
        raise ValueError("Prepared FreeSolv dataset artifact hashes drifted.")
    for filename, expected in expected_artifacts.items():
        path = dataset_dir / filename
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"FreeSolv dataset artifact {filename!r} drifted.")

    selected_manifest = _selected_record_manifest(selected, candidates)
    selected_digest = _canonical_sha256(selected_manifest)
    if selected_digest != manifest.get("selected_record_manifest_sha256"):
        raise ValueError("The selected FreeSolv record manifest drifted from the lock.")
    return candidates, selected_digest


def _validate_ring_atom_count(smiles: str, expected: int) -> None:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - runtime environment gate
        raise ImportError(
            "This runner requires RDKit to audit ring-atom counts."
        ) from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse the pinned SMILES {smiles!r}.")
    observed = sum(int(atom.IsInRing()) for atom in molecule.GetAtoms())
    if observed != expected:
        raise ValueError(
            f"Ring-atom mismatch for {smiles}: expected {expected}, observed {observed}."
        )


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([item["signed_error_kcal_mol"] for item in records])
    experimental = np.asarray([item["experimental_kcal_mol"] for item in records])
    predicted = np.asarray([item["predicted_kcal_mol"] for item in records])
    absolute = np.abs(errors)
    correlation = float(np.corrcoef(experimental, predicted)[0, 1])
    slope, intercept = np.polyfit(experimental, predicted, deg=1)
    worst = records[int(np.argmax(absolute))]
    return {
        "count": len(records),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mae_kcal_mol": float(np.mean(absolute)),
        "rmse_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error_kcal_mol": float(np.median(absolute)),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
        "within_1_kcal_mol_count": int(np.sum(absolute <= 1.0)),
        "within_2_kcal_mol_count": int(np.sum(absolute <= 2.0)),
        "pearson_r": correlation,
        "least_squares_predicted_vs_experimental": {
            "slope": float(slope),
            "intercept_kcal_mol": float(intercept),
        },
        "worst_record": {
            "compound_id": worst["compound_id"],
            "name": worst["name"],
            "chemical_class": worst["chemical_class"],
            "signed_error_kcal_mol": worst["signed_error_kcal_mol"],
        },
    }


def run(args: argparse.Namespace) -> None:
    selection_path = args.selection.expanduser().resolve()
    prepared_path = args.prepared.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (selection_path, prepared_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = work_dir / "xyz"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = work_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    selection = _load_json(selection_path)
    selected = _validate_selection(selection)
    prepared = _load_json(prepared_path)
    candidates, selected_manifest_sha256 = _validate_prepared_manifest(
        selection,
        selected,
        prepared,
        prepared_path=prepared_path,
        dataset_dir=dataset_dir,
    )
    execution_git_head, source_files_sha256 = _bind_source_files((selection_path,))

    import torch

    cosmospace_parameters = replace(
        OPEN_COSMORS_24A_PARAMETERS,
        maximum_iterations=args.cosmospace_maximum_iterations,
    )

    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact": selection["selection_id"],
        "status": "running",
        "scientific_status": "running-diagnostic",
        "admission_eligible": False,
        "diagnostic_only": True,
        "started_at_utc": started,
        "repository_commit": execution_git_head,
        "execution_git_head": execution_git_head,
        "source_files_sha256": source_files_sha256,
        "selection": {
            "path": _repository_relative(selection_path),
            "sha256": _sha256_file(selection_path),
            "selection_id": selection["selection_id"],
            "selected_record_manifest_sha256": selected_manifest_sha256,
        },
        "dataset": {
            **selection["dataset"],
            "prepared_path": str(prepared_path),
            "prepared_sha256": _sha256_file(prepared_path),
            "protocol_id": prepared["protocol_id"],
            "protocol_fingerprint": prepared["protocol_fingerprint"],
            "dataset_artifact_sha256": prepared["dataset_artifact_sha256"],
            "dataset_dir": str(dataset_dir),
        },
        "model": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "device": args.device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(args.device),
            "angular_degree": args.angular_degree,
            "neutral_cosmospace_model": "openCOSMO-RS-24a",
            "cosmospace_successive_substitution_mixing": (
                cosmospace_parameters.successive_substitution_mixing
            ),
            "cosmospace_convergence_relative": (
                cosmospace_parameters.convergence_relative
            ),
            "cosmospace_maximum_iterations": (cosmospace_parameters.maximum_iterations),
            "cosmospace_parameter_sha256": _canonical_sha256(
                asdict(cosmospace_parameters)
            ),
        },
        "runtime": _runtime_versions(),
        "comparison": {
            "solvent": "water",
            "temperature_k": 298.15,
            "experimental_quantity": "FreeSolv hydration free energy",
            "predicted_quantity": "MACE-EF/Torch-segment-COSMO/openCOSMO-RS-24a solvation free energy",
            "prediction_standard_state": "open24a-1atm-gas-to-pure-liquid ledger as implemented",
            "error_definition": "predicted_kcal_mol - experimental_kcal_mol",
            "post_hoc_shift_or_rescaling": False,
            "geometry_relaxation_or_conformer_averaging": False,
            "claim_boundary": "Fixed FreeSolv conformers and a MACE-EF surface outside the ORCA surface convention used to fit open24a; diagnostic only.",
        },
        "claim_boundary": {
            "does_establish": [
                "Execution and aggregate metrics for the exact frozen 20-record fixed-geometry diagnostic.",
                "Per-record MACE-EF uniform-field passivity evidence for the supplied checkpoint.",
            ],
            "does_not_establish": [
                "MACE-EF surface equivalence to the ORCA surface convention used to fit open24a.",
                "General FreeSolv accuracy, conformer-averaged accuracy, force validity, or release admission.",
                "A fitted or validated frozen-source replacement model.",
            ],
        },
        "water_profile": None,
        "records": [],
        "failures": [],
    }
    _write_json_atomic(output_path, artifact)

    water_xyz = geometry_dir / "water.xyz"
    write(
        water_xyz,
        Atoms(symbols=("O", "H", "H"), positions=WATER_POSITIONS_ANGSTROM),
        format="xyz",
    )
    print("Preparing the MACE-EF water sigma profile...", flush=True)
    water_profile, _, water_record = _load_species(
        {
            "name": "water",
            "xyz": str(water_xyz),
            "charge": 0,
            "multiplicity": 1,
            "ring_atom_count": 0,
        },
        base=work_dir,
        checkpoint_path=checkpoint_path,
        device=args.device,
        angular_degree=args.angular_degree,
        scf_payload={},
    )
    water_profile_path = work_dir / "water.torch-cosmors.json"
    write_sigma_profile(water_profile, water_profile_path)
    artifact["water_profile"] = {
        "path": _repository_relative(water_profile_path),
        "sha256": _sha256_file(water_profile_path),
        "xyz_path": _repository_relative(water_xyz),
        "xyz_sha256": _sha256_file(water_xyz),
        "source": water_record,
        "liquid_molar_volume_cm3_mol": 18.06863632,
    }
    _write_json_atomic(output_path, artifact)

    total_start = time.perf_counter()
    for locked in selected:
        compound_id = str(locked["compound_id"])
        candidate = dict(candidates[compound_id])
        ring_atom_count = int(locked["ring_atom_count"])
        _validate_ring_atom_count(str(candidate["smiles"]), ring_atom_count)
        expected_relative = f"dataset/mol2files_gaff/{compound_id}.mol2"
        if candidate["mol2_relative_path"] != expected_relative:
            raise ValueError(f"Pinned MOL2 relative path mismatch for {compound_id}.")
        mol2_path = dataset_dir / "mol2files_gaff" / f"{compound_id}.mol2"
        if _sha256_file(mol2_path) != candidate["mol2_sha256"]:
            raise ValueError(f"Pinned MOL2 SHA256 mismatch for {compound_id}.")
        xyz_path = geometry_dir / f"{compound_id}.xyz"
        atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
        write(xyz_path, atoms, format="xyz")
        record_start = time.perf_counter()
        print(
            f"[{locked['ordinal']:02d}/20] {compound_id} {candidate['name']} "
            f"({locked['chemical_class']})",
            flush=True,
        )
        try:
            profile, _, source = _load_species(
                {
                    "name": compound_id,
                    "xyz": str(xyz_path),
                    "charge": 0,
                    "multiplicity": 1,
                    "ring_atom_count": ring_atom_count,
                },
                base=work_dir,
                checkpoint_path=checkpoint_path,
                device=args.device,
                angular_degree=args.angular_degree,
                scf_payload={},
            )
            result = open24a_solvation_free_energy(
                profile,
                water_profile,
                temperature_k=298.15,
                ring_atom_count=ring_atom_count,
                solvent_liquid_molar_volume_cm3_mol=18.06863632,
                cosmospace_parameters=cosmospace_parameters,
            )
            predicted = float(result.delta_g_solvation_kcal_mol.detach().cpu())
            experimental = float(candidate["experimental_kcal_mol"])
            signed_error = predicted - experimental
            profile_path = profile_dir / f"{compound_id}.torch-cosmors.json"
            write_sigma_profile(profile, profile_path)
            model_result = result.as_dict()
            model_result["workflow_diagnostic_only"] = True
            model_result["workflow_diagnostic_reasons"] = [
                "mace-ef-surface-outside-open24a-orca-fit-domain",
                (
                    "mace-ef-electronic-passivity-failed"
                    if source["electronic_passivity"]["passivity_passed"] is not True
                    else "full-coupled-admission-not-run"
                ),
            ]
            record = {
                "ordinal": int(locked["ordinal"]),
                "compound_id": compound_id,
                "name": candidate["name"],
                "smiles": candidate["smiles"],
                "chemical_class": locked["chemical_class"],
                "source_functional_groups": candidate["functional_groups"],
                "ring_atom_count": ring_atom_count,
                "atom_count": int(candidate["atom_count"]),
                "experimental_kcal_mol": experimental,
                "experimental_uncertainty_kcal_mol": float(
                    candidate["experimental_uncertainty_kcal_mol"]
                ),
                "predicted_kcal_mol": predicted,
                "signed_error_kcal_mol": signed_error,
                "absolute_error_kcal_mol": abs(signed_error),
                "runtime_seconds": time.perf_counter() - record_start,
                "mol2_path": str(mol2_path),
                "mol2_sha256": candidate["mol2_sha256"],
                "generated_xyz": {
                    "path": _repository_relative(xyz_path),
                    "sha256": _sha256_file(xyz_path),
                },
                "mace_ef_profile": {
                    "path": _repository_relative(profile_path),
                    "sha256": _sha256_file(profile_path),
                },
                "model_result": model_result,
                "mace_ef_surface": source,
            }
            artifact["records"].append(record)
            print(
                f"    predicted={predicted:+.4f}, experimental={experimental:+.4f}, "
                f"error={signed_error:+.4f} kcal/mol",
                flush=True,
            )
            del profile, result
            torch.cuda.empty_cache()
        except Exception as exc:  # preserve per-record evidence and continue
            artifact["failures"].append(
                {
                    "ordinal": int(locked["ordinal"]),
                    "compound_id": compound_id,
                    "name": candidate["name"],
                    "chemical_class": locked["chemical_class"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "runtime_seconds": time.perf_counter() - record_start,
                }
            )
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()
        _write_json_atomic(output_path, artifact)

    artifact["completed_at_utc"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    artifact["total_record_runtime_seconds"] = time.perf_counter() - total_start
    artifact["summary"] = _summary(artifact["records"]) if artifact["records"] else None
    artifact["status"] = (
        "complete"
        if len(artifact["records"]) == 20 and not artifact["failures"]
        else "complete-with-failures"
    )
    passivity_failures = [
        record["compound_id"]
        for record in artifact["records"]
        if record["mace_ef_surface"]["electronic_passivity"]["passivity_passed"]
        is not True
    ]
    artifact["scientific_gates"] = {
        "runtime_complete": artifact["status"] == "complete",
        "mace_ef_uniform_field_passivity_all_records": not passivity_failures,
        "open24a_surface_parameterization_equivalence": False,
        "accuracy_gate_preregistered": False,
    }
    artifact["failed_gates"] = [
        name
        for name, passed in artifact["scientific_gates"].items()
        if name != "accuracy_gate_preregistered" and passed is not True
    ]
    artifact["electronic_passivity_failures"] = passivity_failures
    artifact["scientific_status"] = (
        "complete-negative-result"
        if artifact["status"] == "complete"
        else "incomplete-diagnostic"
    )
    _write_json_atomic(output_path, artifact)
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True), flush=True)
    if artifact["failures"]:
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--angular-degree", type=int, default=4)
    parser.add_argument("--cosmospace-maximum-iterations", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.angular_degree < 2:
        parser.error("--angular-degree must be at least 2")
    if args.cosmospace_maximum_iterations < 1:
        parser.error("--cosmospace-maximum-iterations must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
