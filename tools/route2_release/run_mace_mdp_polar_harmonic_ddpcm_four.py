#!/usr/bin/env python3
"""Run the preregistered four-case heterogeneous harmonic-ddPCM diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
from maple.solvation.api.profiles import (  # noqa: E402
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.continuum.harmonic_ddpcm_functional import (  # noqa: E402
    SmoothPartitionHarmonicDDPCMFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_ddpcm_hybrid import (  # noqa: E402
    HARMONIC_DDPCM_HYBRID_CONTRACT_ID,
    build_harmonic_ddpcm_hybrid_snapshot,
)
from maple.solvation.coupling.fixed_point import FixedPointOptions  # noqa: E402
from maple.solvation.coupling.permanent_induced_state import (  # noqa: E402
    PermanentInducedOperationalStateEquation,
)
from maple.solvation.coupling.separated_fixed_point import (  # noqa: E402
    solve_separated_fixed_point,
)
from maple.solvation.coupling.spaces import (  # noqa: E402
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.models import (  # noqa: E402
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)


ARTIFACT_ID = "route2-mace-mdp-polar-harmonic-ddpcm-general-source-four-v1"
PREREGISTRATION = (
    Path("docs/implicit-solvation/benchmarks")
    / "route2-mace-mdp-polar-harmonic-ddpcm-general-source-four-prereg-v1.json"
)
PARENT_ASSET_MANIFEST = (
    Path("docs/implicit-solvation/benchmarks")
    / "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
REFERENCE_EVIDENCE = (
    Path("docs/route2/evidence")
    / "mace-mdp-polar-ddx-external-mep-four-4cf8db40.json"
)
DEFAULT_OUTPUT = (
    Path(".omx/route2")
    / "mace-mdp-polar-harmonic-ddpcm-general-source-four-v1.json"
)
DEFAULT_MDP_CHECKPOINT = Path.home() / ".cache/mace/MACE-MDP.model"
DEFAULT_POLAR_CHECKPOINT = Path.home() / ".cache/mace/MACEPOLAR1Mmodel"
KCAL_PER_EV = 23.06054783061903
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
SOURCE_FILES = (
    "maple/solvation/continuum/harmonic_ddpcm_functional.py",
    "maple/solvation/continuum/harmonic_ddpcm_hybrid.py",
    "maple/solvation/continuum/harmonic_ddpcm_primitives.py",
    "maple/solvation/continuum/harmonic_schwarz_primitives.py",
    "maple/solvation/coupling/permanent_induced_state.py",
    "maple/solvation/coupling/separated_fixed_point.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar_separated.py",
    "tools/route2_release/run_mace_mdp_polar_harmonic_ddpcm_four.py",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, default=DEFAULT_MDP_CHECKPOINT)
    parser.add_argument(
        "--polar-checkpoint", type=Path, default=DEFAULT_POLAR_CHECKPOINT
    )
    parser.add_argument("--polar-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return payload


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_sha(path: Path, expected: object, *, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} has no valid SHA256 binding.")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}.")


def _configure_torch(device: str):
    import torch

    torch.manual_seed(20260816)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260816)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for MACE-POLAR but is unavailable.")
    return torch


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[2]
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser()
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    preregistration_path = root / PREREGISTRATION
    parent_path = root / PARENT_ASSET_MANIFEST
    reference_path = root / REFERENCE_EVIDENCE
    preregistration = _json(preregistration_path)
    parent = _json(parent_path)
    reference = _json(reference_path)
    reference_binding = preregistration["reference"]
    candidate = preregistration["candidate"]
    decision = preregistration["decision_rule"]
    if not isinstance(reference_binding, dict) or not isinstance(candidate, dict):
        raise ValueError("preregistration reference/candidate sections are invalid.")
    _validate_sha(
        parent_path,
        reference_binding["parent_asset_manifest_sha256"],
        label="parent asset manifest",
    )
    _validate_sha(
        reference_path,
        reference_binding["reference_evidence_sha256"],
        label="reference evidence",
    )
    case_ids = tuple(str(value) for value in preregistration["case_ids"])
    parent_by_id = {str(row["compound_id"]): row for row in parent["records"]}
    reference_by_id = {
        str(row["compound_id"]): row for row in reference["records"]
    }
    if set(case_ids) != set(parent_by_id) or set(case_ids) != set(reference_by_id):
        raise ValueError("preregistered, parent, and reference case sets differ.")

    torch = _configure_torch(args.polar_device)
    started = time.perf_counter()
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.polar_device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    response = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=response,
    )
    options = FixedPointOptions(
        method=str(candidate["root_method"]),
        tolerance=float(candidate["root_tolerance"]),
        max_iterations=int(candidate["root_max_iterations"]),
        damping=float(candidate["root_damping"]),
        history=int(candidate["root_history"]),
    )
    agreement = candidate["cold_wide_agreement"]
    if not isinstance(agreement, dict):
        raise ValueError("cold_wide_agreement must be an object.")
    source_atol = float(agreement["maximum_induced_source_absolute_difference_e"])
    field_atol = float(
        agreement["maximum_native_field_absolute_difference_eV_per_e"]
    )
    energy_atol = float(
        agreement["maximum_continuum_energy_absolute_difference_eV"]
    )

    records: list[dict[str, object]] = []
    for case_index, case_id in enumerate(case_ids):
        row = parent_by_id[case_id]
        reference_row = reference_by_id[case_id]
        case_started = time.perf_counter()
        mol2 = asset_root / str(row["mol2_path"])
        _validate_sha(mol2, row["mol2_sha256"], label=f"{case_id} MOL2")
        atoms = MOL2Reader(str(mol2), charge=0, mult=1)
        result_path = (
            asset_root
            / ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
            / case_id
            / "cutoff-1e-12/result.json"
        )
        result = _json(result_path)
        pcm_input = result["inputs"]["parsed_pcm_input"]
        radii = np.asarray(pcm_input["cavity_radii_angstrom"], dtype=float)
        if radii.shape != (len(atoms),) or np.any(radii <= 0.0):
            raise ValueError(f"{case_id} cavity radii are invalid.")

        functional = SmoothPartitionHarmonicDDPCMFunctionalCandidate(
            atomic_numbers=tuple(int(value) for value in atoms.numbers),
            radii_angstrom=tuple(float(value) for value in radii),
            transition_width_angstrom2=float(
                candidate["transition_width_angstrom2"]
            ),
            surface_lmax=int(candidate["surface_lmax"]),
            partition_lmax=int(candidate["partition_lmax"]),
            partition_radial_quadrature_order=int(
                candidate["partition_radial_quadrature_order"]
            ),
            source_radial_quadrature_order=int(
                candidate["source_radial_quadrature_order"]
            ),
            double_layer_radial_quadrature_order=int(
                candidate["double_layer_radial_quadrature_order"]
            ),
            dielectric=float(candidate["dielectric"]),
            dtype=torch.float64,
            device="cpu",
        )
        anchor = hybrid.prepare(atoms)
        continuum = build_harmonic_ddpcm_hybrid_snapshot(
            functional,
            atoms,
            receiver_radial_quadrature_order=int(
                candidate["receiver_radial_quadrature_order"]
            ),
        )
        coordinates = AffineChargeCoordinates(
            atom_count=len(atoms),
            total_charge=0.0,
            source_space=ATOMIC_L1_SOURCE_SPACE,
        )
        equation = PermanentInducedOperationalStateEquation(
            coordinates,
            hybrid,
            anchor,
            continuum,
        )
        context = f"harmonic-general-source-four-{case_id}"
        cold = solve_separated_fixed_point(
            equation,
            atoms,
            root_context_id=context,
            options=options,
        )
        wide_seed = 20260816 + case_index
        wide_initial = cold.y_array() + 0.05 * np.random.default_rng(
            wide_seed
        ).normal(size=cold.y_array().shape)
        wide = solve_separated_fixed_point(
            equation,
            atoms,
            root_context_id=context,
            initial_y=wide_initial,
            options=options,
        )
        cold_energy_ev = continuum.continuum_energy_eV(
            equation.permanent_source,
            cold.source_array(),
        )
        wide_energy_ev = continuum.continuum_energy_eV(
            equation.permanent_source,
            wide.source_array(),
        )
        source_difference = float(
            np.max(np.abs(cold.source_array() - wide.source_array()))
        )
        field_difference = float(
            np.max(np.abs(cold.field_array() - wide.field_array()))
        )
        energy_difference = abs(cold_energy_ev - wide_energy_ev)
        root_agreement_passed = (
            source_difference <= source_atol
            and field_difference <= field_atol
            and energy_difference <= energy_atol
        )
        reference_kcal = float(
            reference_row["qm_pcmsolver_fixed_gas_density_kcal_mol"]
        )
        candidate_kcal = cold_energy_ev * KCAL_PER_EV
        signed_error = candidate_kcal - reference_kcal
        record = {
            "compound_id": case_id,
            "name": str(row["name"]),
            "class": str(row["class"]),
            "atom_count": len(atoms),
            "mol2_sha256": _sha256(mol2),
            "pcm_result_sha256": _sha256(result_path),
            "cavity_radii_angstrom": radii.tolist(),
            "reference_qm_pcmsolver_kcal_mol": reference_kcal,
            "candidate_harmonic_ddpcm_kcal_mol": candidate_kcal,
            "signed_error_kcal_mol": signed_error,
            "absolute_error_kcal_mol": abs(signed_error),
            "cold_root_residual": cold.actual_unmixed_residual_norm,
            "wide_root_residual": wide.actual_unmixed_residual_norm,
            "cold_iterations": len(cold.iterations) - 1,
            "wide_iterations": len(wide.iterations) - 1,
            "induced_source_max_abs_difference": source_difference,
            "native_field_max_abs_difference": field_difference,
            "continuum_energy_abs_difference_eV": energy_difference,
            "cold_wide_agreement_passed": root_agreement_passed,
            "continuum_configuration_sha256": continuum.configuration_sha256(),
            "continuum_provenance_sha256": continuum.provenance_sha256,
            "equation_sha256": equation.fingerprint_sha256(),
            "cold_root_sha256": cold.root_hash,
            "wide_root_sha256": wide.root_hash,
            "elapsed_seconds": time.perf_counter() - case_started,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    absolute_errors = np.asarray(
        [float(record["absolute_error_kcal_mol"]) for record in records]
    )
    signed_errors = np.asarray(
        [float(record["signed_error_kcal_mol"]) for record in records]
    )
    mae = float(np.mean(absolute_errors))
    target = float(decision["target_mae_kcal_mol"])
    roots_passed = all(
        bool(record["cold_wide_agreement_passed"]) for record in records
    )
    accuracy_passed = mae <= target
    source_hashes = {relative: _sha256(root / relative) for relative in SOURCE_FILES}
    payload = {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass" if roots_passed and accuracy_passed else "fail",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": preregistration["claim_boundary"],
        "git": {
            "head": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "dirty": bool(_git(root, "status", "--porcelain=v1")),
        },
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": _sha256(preregistration_path),
        },
        "reference": {
            "parent_asset_manifest_sha256": _sha256(parent_path),
            "reference_evidence_sha256": _sha256(reference_path),
        },
        "checkpoints": {
            "mace_mdp_sha256": mdp.checkpoint_sha256,
            "mace_mdp_device": "cpu",
            "mace_polar_sha256": response.checkpoint_sha256,
            "mace_polar_device": radial.provenance.device,
        },
        "candidate": {
            **candidate,
            "hybrid_configuration_sha256": hybrid.configuration_sha256(),
            "hybrid_provenance_sha256": hybrid.provenance_sha256,
            "continuum_contract_id": HARMONIC_DDPCM_HYBRID_CONTRACT_ID,
        },
        "records": records,
        "aggregate": {
            "case_count": len(records),
            "mean_signed_error_kcal_mol": float(np.mean(signed_errors)),
            "mae_kcal_mol": mae,
            "rmse_kcal_mol": float(np.sqrt(np.mean(signed_errors**2))),
            "max_abs_error_kcal_mol": float(np.max(absolute_errors)),
            "target_mae_kcal_mol": target,
            "accuracy_target_passed": accuracy_passed,
            "all_cold_wide_root_checks_passed": roots_passed,
        },
        "capabilities": NO_CAPABILITIES,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
            "total_seconds": time.perf_counter() - started,
        },
        "source_files_sha256": source_hashes,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True), flush=True)
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
