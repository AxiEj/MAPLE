#!/usr/bin/env python3
"""Run the preregistered original-source far-field terminal audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.models import (
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    FarFieldSourceThresholds,
    RepositorySnapshot,
    SourceElectrostaticObservables,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    compare_far_field_source_to_reference,
    runtime_record,
    sha256_file,
    source_electrostatic_observables,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-original-source-farfield-four-v1"
EXPECTED_MACE_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)
ORIGINAL_PANEL_PREREGISTRATION = (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
FAR_FIELD_PREREGISTRATION = (
    "docs/route2/preregistrations/mace-original-source-farfield-four-v1.json"
)
NEAR_FIELD_EVIDENCE = (
    "docs/route2/evidence/" "mace-original-source-pcmsolver-four-1d40c93b/run1.json"
)
PANEL_RELATIVE_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
CUTOFF_DIRECTORY = "cutoff-1e-12"
GEOMETRY_ATOL_ANGSTROM = 1.0e-8
REFERENCE_REPLAY_ATOL = 2.0e-10
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/evidence.py",
    "maple/solvation/release/farfield_source.py",
    "maple/solvation/release/source_mep.py",
    "tools/route2_release/qm_checkpoint_farfield.py",
    "tools/route2_release/run_mace_original_source_farfield_panel.py",
    FAR_FIELD_PREREGISTRATION,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--pyscf-python", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
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


def _validated_sha(path: Path, expected: object, *, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{name} omits a SHA256 binding.")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 mismatch: {actual} != {expected}.")
    return actual


def _rebased_asset_path(asset_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("Frozen asset path must be a non-empty string.")
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()
    try:
        marker = candidate.parts.index(".omx")
    except ValueError as exc:
        raise RuntimeError(f"Cannot rebase frozen asset path {raw_path!r}.") from exc
    rebased = asset_root.joinpath(*candidate.parts[marker:]).resolve()
    if not rebased.is_file():
        raise FileNotFoundError(rebased)
    return rebased


def _configure_determinism(torch: object) -> None:
    torch.manual_seed(20260815)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260815)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _reference_observables(
    reference: dict[str, np.ndarray],
) -> SourceElectrostaticObservables:
    return SourceElectrostaticObservables(
        total_charge_e=float(reference["total_charge_e"]),
        molecular_dipole_e_angstrom=tuple(
            float(value) for value in reference["molecular_dipole_e_angstrom"]
        ),
        traceless_quadrupole_e_angstrom2=tuple(
            tuple(float(value) for value in row)
            for row in reference["traceless_quadrupole_e_angstrom2"]
        ),
        sampled_mep_hartree_per_e=tuple(
            float(value) for value in reference["far_field_potential_hartree_per_e"]
        ),
        multipole_origin_angstrom=tuple(
            float(value) for value in reference["multipole_origin_angstrom"]
        ),
    )


def _run_qm_reference(
    *,
    pyscf_python: Path,
    helper: Path,
    checkpoint: Path,
    output: Path,
    polar_order: int,
    shell_offsets_angstrom: list[float],
) -> dict[str, np.ndarray]:
    command = [
        str(pyscf_python),
        str(helper),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--polar-order",
        str(polar_order),
        "--shell-offsets-angstrom",
        *(str(value) for value in shell_offsets_angstrom),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "QM far-field helper failed:\n" + completed.stdout + "\n" + completed.stderr
        )
    with np.load(output) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _record_case(
    *,
    record: dict[str, object],
    asset_root: Path,
    electronic: MACEPolarOriginalSourceNativeFieldAdapter,
    pyscf_python: Path,
    helper: Path,
    polar_order: int,
    shell_offsets_angstrom: list[float],
    thresholds: FarFieldSourceThresholds,
) -> dict[str, object]:
    compound_id = str(record["compound_id"])
    panel_root = asset_root / PANEL_RELATIVE_ROOT / compound_id
    result_path = panel_root / CUTOFF_DIRECTORY / "result.json"
    result = _load_json(result_path, name=f"{compound_id} projection result")
    inputs = result.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("compound_id") != compound_id:
        raise RuntimeError(f"{compound_id} projection identity is invalid.")

    mol2 = asset_root / str(record["mol2_path"])
    _validated_sha(mol2, record.get("mol2_sha256"), name=f"{compound_id} MOL2")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    checkpoint_record_value = inputs.get("qm_checkpoint")
    if not isinstance(checkpoint_record_value, dict):
        raise RuntimeError(f"{compound_id} omits its QM checkpoint.")
    qm_checkpoint = _rebased_asset_path(asset_root, checkpoint_record_value.get("path"))
    _validated_sha(
        qm_checkpoint,
        checkpoint_record_value.get("sha256"),
        name=f"{compound_id} QM checkpoint",
    )

    with tempfile.TemporaryDirectory(prefix=f"route2-far-{compound_id}-") as work:
        reference_path = Path(work) / "reference.npz"
        reference = _run_qm_reference(
            pyscf_python=pyscf_python,
            helper=helper,
            checkpoint=qm_checkpoint,
            output=reference_path,
            polar_order=polar_order,
            shell_offsets_angstrom=shell_offsets_angstrom,
        )

    atomic_numbers = np.asarray(reference["atomic_numbers"], dtype=int)
    positions = np.asarray(reference["atom_positions_angstrom"], dtype=float)
    if not np.array_equal(atomic_numbers, atoms.numbers):
        raise RuntimeError(f"{compound_id} QM/MOL2 elements differ.")
    geometry_error = float(np.max(np.abs(positions - atoms.positions)))
    if geometry_error > GEOMETRY_ATOL_ANGSTROM:
        raise RuntimeError(f"{compound_id} QM/MOL2 geometry mismatch.")

    frozen_qm_path = panel_root / CUTOFF_DIRECTORY / "work/qm-surface-mep.npz"
    with np.load(frozen_qm_path) as frozen_qm:
        frozen_charge = float(frozen_qm["total_charge_e"])
        frozen_dipole = np.asarray(
            frozen_qm["molecular_dipole_e_angstrom"], dtype=float
        )
    if abs(float(reference["total_charge_e"]) - frozen_charge) > REFERENCE_REPLAY_ATOL:
        raise RuntimeError(f"{compound_id} QM charge replay drifted.")
    if (
        float(np.max(np.abs(reference["molecular_dipole_e_angstrom"] - frozen_dipole)))
        > REFERENCE_REPLAY_ATOL
    ):
        raise RuntimeError(f"{compound_id} QM dipole replay drifted.")

    source = np.asarray(
        electronic.evaluate_source(atoms, np.zeros((len(atoms), 8))), dtype=float
    )
    predicted = source_electrostatic_observables(
        positions_angstrom=atoms.positions,
        source4=source,
        evaluation_points_bohr=reference["evaluation_points_bohr"],
        sigma_angstrom=1.5,
        multipole_origin_angstrom=reference["multipole_origin_angstrom"],
    )
    qm_observables = _reference_observables(reference)
    comparison = compare_far_field_source_to_reference(
        predicted,
        qm_observables,
        shell_radii_angstrom=reference["shell_radii_angstrom"],
        shell_point_counts=reference["shell_point_counts"],
        angular_weights=reference["angular_weights"],
        thresholds=thresholds,
    )
    density_record = json.loads(str(reference["density_record_json"]))
    return {
        "compound_id": compound_id,
        "name": str(record["name"]),
        "class": str(record["class"]),
        "atom_count": len(atoms),
        "geometry_max_abs_error_angstrom": geometry_error,
        "asset_sha256": {
            "mol2": sha256_file(mol2),
            "projection_result": sha256_file(result_path),
            "frozen_qm_surface_mep": sha256_file(frozen_qm_path),
            "qm_checkpoint": sha256_file(qm_checkpoint),
        },
        "qm_density": density_record,
        "multipole_origin_angstrom": list(qm_observables.multipole_origin_angstrom),
        "qm_total_charge_e": qm_observables.total_charge_e,
        "predicted_total_charge_e": predicted.total_charge_e,
        "qm_dipole_e_angstrom": list(qm_observables.molecular_dipole_e_angstrom),
        "predicted_dipole_e_angstrom": list(predicted.molecular_dipole_e_angstrom),
        "qm_traceless_quadrupole_e_angstrom2": [
            list(row) for row in qm_observables.traceless_quadrupole_e_angstrom2
        ],
        "predicted_traceless_quadrupole_e_angstrom2": [
            list(row) for row in predicted.traceless_quadrupole_e_angstrom2
        ],
        "reference_points_sha256": str(reference["evaluation_points_bohr_sha256"]),
        "reference_far_field_mep_sha256": str(reference["far_field_potential_sha256"]),
        "comparison": comparison.as_dict(),
        "case_passed": comparison.case_passed,
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    started = time.perf_counter()
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    pyscf_python = args.pyscf_python.expanduser().resolve(strict=True)
    helper = repository.root / "tools/route2_release/qm_checkpoint_farfield.py"
    if sha256_file(checkpoint) != EXPECTED_MACE_CHECKPOINT_SHA256:
        raise RuntimeError("MACE checkpoint does not match the frozen official model.")

    original_prereg_path = repository.root / ORIGINAL_PANEL_PREREGISTRATION
    far_prereg_path = repository.root / FAR_FIELD_PREREGISTRATION
    near_field_path = repository.root / NEAR_FIELD_EVIDENCE
    original_prereg = _load_json(original_prereg_path, name="original panel prereg")
    prereg = _load_json(far_prereg_path, name="far-field preregistration")
    near_field = _load_json(near_field_path, name="near-field source evidence")
    bindings = prereg.get("parent_bindings")
    panel = prereg.get("panel")
    far_rule = prereg.get("far_field_rule")
    threshold_values = prereg.get("thresholds")
    if (
        prereg.get("artifact_id")
        != "route2-mace-original-source-farfield-four-prereg-v1"
        or prereg.get("status") != "frozen-before-far-field-execution"
    ):
        raise RuntimeError("far-field preregistration identity is invalid.")
    if not all(
        isinstance(value, dict)
        for value in (bindings, panel, far_rule, threshold_values)
    ):
        raise RuntimeError("far-field preregistration is incomplete.")
    _validated_sha(
        original_prereg_path,
        bindings["frozen_four_case_preregistration_sha256"],
        name="original panel preregistration",
    )
    _validated_sha(
        near_field_path,
        bindings["near_field_source_gate_run1_sha256"],
        name="near-field source evidence",
    )
    _validated_sha(
        pyscf_python,
        bindings["pyscf_python_resolved_sha256"],
        name="PySCF Python",
    )
    if (
        near_field.get("measurement_sha256")
        != bindings["near_field_source_measurement_sha256"]
    ):
        raise RuntimeError("near-field source measurement binding is invalid.")
    if (
        near_field.get("decision", {}).get(
            "original_four_channel_quantitative_pcm_source_admitted"
        )
        is not False
    ):
        raise RuntimeError("far-field branch requires the frozen near-field failure.")

    records = original_prereg.get("records")
    selected_ids = list(panel["compound_ids"])
    if not isinstance(records, list) or sorted(
        str(record.get("compound_id")) for record in records
    ) != sorted(selected_ids):
        raise RuntimeError("far-field panel does not match the frozen four records.")
    thresholds = FarFieldSourceThresholds(**threshold_values)
    polar_order = int(far_rule["polar_order"])
    shell_offsets = [
        float(value)
        for value in far_rule["shell_offsets_beyond_maximum_nuclear_extent_angstrom"]
    ]

    import torch

    _configure_determinism(torch)
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    electronic = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    case_records = [
        _record_case(
            record=record,
            asset_root=asset_root,
            electronic=electronic,
            pyscf_python=pyscf_python,
            helper=helper,
            polar_order=polar_order,
            shell_offsets_angstrom=shell_offsets,
            thresholds=thresholds,
        )
        for record in sorted(records, key=lambda value: str(value["compound_id"]))
    ]
    pass_count = sum(bool(record["case_passed"]) for record in case_records)
    panel_passed = bool(pass_count == len(case_records))
    radial_authorized = panel_passed
    measurement = {
        "protocol": {
            "source": panel["source"],
            "reference": panel["checkpoint_reference"],
            "multipole_convention": prereg["multipole_convention"],
            "far_field_rule": far_rule,
            "thresholds": thresholds.as_dict(),
            "threshold_provenance": (
                "frozen in the bound preregistration before this execution"
            ),
        },
        "records": case_records,
        "aggregate": {
            "record_count": len(case_records),
            "case_pass_count": pass_count,
            "maximum_dipole_relative_l2_error": max(
                float(record["comparison"]["dipole_relative_l2_error"])
                for record in case_records
            ),
            "maximum_quadrupole_relative_frobenius_error": max(
                float(record["comparison"]["quadrupole_relative_frobenius_error"])
                for record in case_records
            ),
            "maximum_far_field_shell_relative_l2_error": max(
                float(shell["relative_l2_error"])
                for record in case_records
                for shell in record["comparison"]["shell_comparisons"]
            ),
        },
        "decision": {
            "original_four_channel_quantitative_pcm_source_admitted": False,
            "far_field_necessary_precondition_passed": panel_passed,
            "fixed_radial_embedding_research_authorized": radial_authorized,
            "radial_embedding_authorization_scope": (
                "one separately named fixed symmetry-preserving research profile; "
                "held-out cavity-near-field evidence still mandatory"
                if radial_authorized
                else "not authorized"
            ),
            "scalar_first_or_independent_polarization_branch_required": (
                not panel_passed
            ),
            "separated_phi0_ledger_evaluation_authorized": False,
            "separated_phi1_delta_ledger_evaluation_authorized": False,
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
        "artifact_kind": "disabled-original-source-far-field-terminal-audit",
        "status": (
            "far-field-pass-radial-embedding-research-authorized"
            if panel_passed
            else "far-field-fail-scalar-first-branch-required"
        ),
        "claim_boundary": prereg["claim_boundary"],
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": committed_source_hashes(repository, source_paths),
        "external_assets": {
            "asset_root": str(asset_root),
            "far_field_preregistration": {
                "path": str(far_prereg_path),
                "sha256": sha256_file(far_prereg_path),
                "artifact_id": prereg["artifact_id"],
            },
            "near_field_source_evidence": {
                "path": str(near_field_path),
                "sha256": sha256_file(near_field_path),
                "measurement_sha256": near_field["measurement_sha256"],
            },
            "mace_checkpoint": checkpoint_record(checkpoint),
            "pyscf_python": {
                "path": str(pyscf_python),
                "sha256": sha256_file(pyscf_python),
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
        "ROUTE2_MACE_ORIGINAL_SOURCE_FARFIELD="
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
