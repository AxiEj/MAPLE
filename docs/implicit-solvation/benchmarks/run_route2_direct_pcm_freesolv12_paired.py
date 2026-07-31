#!/usr/bin/env python3
"""Run the locked FreeSolv-12 direct-PCM/ddCOSMO Route-2 comparison.

This runner evaluates every member of the immutable twelve-record water
panel.  Ten records carry distinct functional-group labels; methane and
benzene are retained nonfunctional cavity/polarizability controls.  It never
offers a smaller accuracy panel.  The two methods differ only in the frozen
pyddx continuum equation:

* ddPCM + PySCF SMD-CDS;
* scaled ddCOSMO + the same PySCF SMD-CDS.

Both use the unmodified MACE-POLAR-1-M checkpoint, the same local-jet
receiver, the direct PCM half-coupling ledger, the same water profile, and the
same locked MOL2 conformers.  The output is deliberately private below
``.omx`` because it contains row-level experimental FreeSolv data.  A result
passes only if *every* record is strictly below 1.5 kcal/mol for both
equations; MAE never overrides an individual failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
from ase import Atoms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for _path in (REPO_ROOT, SCRIPT_DIR):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)

from benchmark_core import sha256_file, write_json_atomic  # noqa: E402
import run_mnsol_macepolar_multisolvent_pilot as paired_benchmark  # noqa: E402
from route2_v0_freesolv12_functional_groups import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    FUNCTIONAL_GROUP_PANEL_ID,
    evaluate_freesolv12_functional_group_predictions,
    load_freesolv12_functional_group_manifest,
)

from maple.function.calculator.extra_correction.implicit.correction import (  # noqa: E402
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (  # noqa: E402
    HARTREE_TO_KCAL_MOL,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
from maple.function.route2_energy_ledger import (  # noqa: E402
    PCM_HALF_COUPLING_ONLY_V1,
)


# v2 switches to the separately versioned paired-continuum numerical
# acceptance contract.  The old v1 work directory remains a valid record of
# its stricter nominal-only ddCOSMO outcome and is never resumed as v2.
ARTIFACT = "route2-direct-pcm-freesolv12-ddpcm-ddcosmo-v2"
SCHEMA_VERSION = 2
WATER_SOLVENT = "water"
METHOD_PROFILES = paired_benchmark.method_profiles_for_energy_ledger(
    PCM_HALF_COUPLING_ONLY_V1
)


def _execution_git_head() -> str:
    """Require one clean source revision for an auditable long run."""

    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "The locked FreeSolv-12 run requires a clean Git checkout; "
            "commit code changes before creating or resuming evidence."
        )
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("Could not resolve a full Git revision for this run.")
    return head


def _require_private_path(path: Path, *, label: str) -> Path:
    """Keep row-level experimental artifacts out of tracked documentation."""

    resolved = path.resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(f"{label} must be located below {private_root}.") from exc
    return resolved


def _resolve_mol2_path(mol2_root: Path, record: dict[str, Any]) -> Path:
    """Resolve one hash-locked archive member without accepting substitutions."""

    member = record.get("mol2_archive_member")
    if not isinstance(member, str) or not member:
        raise TypeError("Locked FreeSolv record lacks mol2_archive_member.")
    candidates = (mol2_root / member, mol2_root / Path(member).name)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Locked MOL2 member {member!r} is absent below {mol2_root}."
        )
    expected_hash = record.get("mol2_sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise ValueError(f"Locked FreeSolv MOL2 identity drifted: {record['compound_id']}.")
    return path


def _atoms(record: dict[str, Any], mol2_root: Path) -> Atoms:
    """Load one frozen geometry and recheck its atom count."""

    atoms = MOL2Reader(str(_resolve_mol2_path(mol2_root, record)), charge=0, mult=1)
    expected_count = int(record["natoms"])
    if len(atoms) != expected_count:
        raise RuntimeError(
            f"Locked FreeSolv atom count changed for {record['compound_id']}: "
            f"{len(atoms)} != {expected_count}."
        )
    atoms.info.update(charge=0, mult=1)
    return atoms


def _source_hashes() -> dict[str, str]:
    """Bind the long-running evidence to every selected implementation file."""

    paths = {
        "paired_multisolvent_runner": (
            SCRIPT_DIR / "run_mnsol_macepolar_multisolvent_pilot.py"
        ),
        "freesolv12_runner": Path(__file__),
        "freesolv12_gate": SCRIPT_DIR / "route2_v0_freesolv12_functional_groups.py",
        "macepolar_adapter": (
            REPO_ROOT / "maple/function/calculator/mace/_macepol_calculator.py"
        ),
        "pyddx_provider": (
            REPO_ROOT
            / "maple/function/calculator/extra_correction/implicit/ddpcm_smd.py"
        ),
        "route2_engine": (
            REPO_ROOT
            / "maple/function/calculator/extra_correction/implicit/route2_engine.py"
        ),
        "route2_ledger": REPO_ROOT / "maple/function/route2_energy_ledger.py",
        "route2_profiles": REPO_ROOT / "maple/function/route2_smd_profiles.py",
        "route2_solvents": REPO_ROOT / "maple/function/route2_solvents.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _run_lock(
    *,
    manifest_path: Path,
    records: list[dict[str, Any]],
    execution_git_head: str,
) -> dict[str, object]:
    """Return the immutable identity that makes resumable rows safe."""

    return {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "execution_git_head": execution_git_head,
        "panel_id": FUNCTIONAL_GROUP_PANEL_ID,
        "panel_manifest_sha256": sha256_file(manifest_path),
        "compound_ids": [str(record["compound_id"]) for record in records],
        "method_profiles": dict(METHOD_PROFILES),
        "energy_ledger": PCM_HALF_COUPLING_ONLY_V1,
        "solvent": WATER_SOLVENT,
        "energy_composition": paired_benchmark.paired_energy_composition(
            PCM_HALF_COUPLING_ONLY_V1
        ),
        "source_files_sha256": _source_hashes(),
        "experimental_labels_not_used_for_model_selection": True,
        "all_twelve_records_required": True,
    }


def _establish_or_validate_run_lock(work_dir: Path, lock: dict[str, object]) -> None:
    """Create one lock or refuse to mix rows created from different methods."""

    lock_path = work_dir / "run-lock.json"
    if lock_path.exists():
        observed = json.loads(lock_path.read_text(encoding="utf-8"))
        if observed != lock:
            raise RuntimeError(
                "Existing FreeSolv-12 work directory has a different source, "
                "profile, ledger, panel, or Git lock; refusing to resume it."
            )
        return
    write_json_atomic(lock_path, lock)


def _evaluate_method(
    *,
    calculator,
    atoms: Atoms,
    record: dict[str, Any],
    method: str,
    profile: str,
    work_dir: Path,
) -> dict[str, object]:
    """Use MAPLE's public Route-2 energy path for exactly one equation."""

    parameters = paired_benchmark._settings(WATER_SOLVENT, profile)
    compound_id = str(record["compound_id"])
    output = work_dir / "provider-audits" / compound_id / method / "maple.out"
    output.parent.mkdir(parents=True, exist_ok=True)
    calculator.solvent_correction = ImplicitSolvationCorrection(
        atoms,
        parameters.get("charge") or {},
        parameters["solv"],
        output=output,
    )
    calculator.reset()
    atoms.calc = calculator
    started = time.perf_counter()
    combined_energy_hartree = float(atoms.get_potential_energy())
    wall_seconds = time.perf_counter() - started

    result = calculator.results.get("solvation")
    if not isinstance(result, dict):
        raise RuntimeError("Public MAPLE Route-2 path did not emit a solvation ledger.")
    components = result.get("components_hartree")
    provenance = result.get("provenance")
    if not isinstance(components, dict) or not isinstance(provenance, dict):
        raise RuntimeError("Public MAPLE Route-2 result has invalid components/provenance.")
    if provenance.get("electrostatics_model") != method:
        raise RuntimeError(f"{compound_id}: public route selected wrong equation.")
    if provenance.get("profile") != profile:
        raise RuntimeError(f"{compound_id}: public route selected wrong profile.")
    if provenance.get("electrostatic_energy_ledger") != PCM_HALF_COUPLING_ONLY_V1:
        raise RuntimeError(f"{compound_id}: public route selected wrong ledger.")
    if abs(combined_energy_hartree - float(result["combined_energy_hartree"])) > 1.0e-12:
        raise RuntimeError(f"{compound_id}: combined-energy finalizer drifted.")
    if abs(float(components["solute_polarization"])) > 1.0e-15:
        raise RuntimeError(f"{compound_id}: direct PCM ledger included a MACE leaf.")

    correction = calculator.solvent_correction
    audit_dir = Path(correction.audit_dir)
    audit_path = audit_dir / f"route2-{method}-result.json"
    state_path = audit_dir / f"route2-{method}-state.npz"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with np.load(state_path) as state:
        root_density = np.asarray(state["density_coefficients"], dtype=float)
    monitors = paired_benchmark._terminal_scf_monitors(audit, root_density)
    experimental = float(record["experimental_kcal_mol"])
    predicted = float(result["delta_g_solv_hartree"]) * HARTREE_TO_KCAL_MOL
    return {
        "profile": profile,
        "gas_energy_hartree": float(result["gas_energy_hartree"]),
        "combined_energy_hartree": combined_energy_hartree,
        "predicted_kcal_mol": predicted,
        "signed_error_kcal_mol": predicted - experimental,
        "absolute_error_kcal_mol": abs(predicted - experimental),
        "components_kcal_mol": {
            key: float(value) * HARTREE_TO_KCAL_MOL
            for key, value in components.items()
        },
        "field_conditioned_mace_energy_change_kcal_mol": (
            float(audit["field_conditioned_mace_energy_change_hartree"])
            * HARTREE_TO_KCAL_MOL
        ),
        "scf_iterations": int(audit["scf"]["iterations"]),
        "scf_convergence": dict(audit["scf"]["convergence"]),
        "half_coupling_identity_error_ev": float(
            audit["polarization_energy_identity_error_ev"]
        ),
        "fixed_point_monitors": monitors,
        "runtime_provenance": provenance,
        "wall_seconds": wall_seconds,
    }


def _evaluate_record(
    *,
    calculator,
    record: dict[str, Any],
    mol2_root: Path,
    work_dir: Path,
) -> dict[str, object]:
    """Evaluate both equations from one identical locked molecular geometry."""

    atoms = _atoms(record, mol2_root)
    methods = {
        method: _evaluate_method(
            calculator=calculator,
            atoms=atoms,
            record=record,
            method=method,
            profile=profile,
            work_dir=work_dir,
        )
        for method, profile in METHOD_PROFILES
    }
    if abs(
        float(methods["ddpcm"]["gas_energy_hartree"])
        - float(methods["ddcosmo"]["gas_energy_hartree"])
    ) > 1.0e-12:
        raise RuntimeError(
            f"{record['compound_id']}: paired equations did not share gas energy."
        )
    functional_group = record.get("functional_group")
    return {
        "compound_id": record["compound_id"],
        "name": record["name"],
        "functional_group": functional_group,
        "is_nonfunctional_control": functional_group is None,
        "experimental_kcal_mol": float(record["experimental_kcal_mol"]),
        "mol2_archive_member": record["mol2_archive_member"],
        "mol2_sha256": record["mol2_sha256"],
        "atom_count": len(atoms),
        "methods": methods,
    }


def _error_metrics(records: list[dict[str, object]], method: str) -> dict[str, object]:
    """Report every conventional aggregate without letting it replace the gate."""

    errors = np.asarray(
        [float(record["methods"][method]["signed_error_kcal_mol"]) for record in records],
        dtype=float,
    )
    if errors.size == 0 or not np.all(np.isfinite(errors)):
        raise RuntimeError("FreeSolv paired errors must be finite and nonempty.")
    absolute = np.abs(errors)
    worst_index = int(np.argmax(absolute))
    return {
        "record_count": int(errors.size),
        "mae_kcal_mol": float(np.mean(absolute)),
        "rmse_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
        "worst_compound_id": records[worst_index]["compound_id"],
        "records_at_or_above_1_5_kcal_mol": int(np.sum(absolute >= 1.5)),
        "all_records_strictly_below_1_5_kcal_mol": bool(np.all(absolute < 1.5)),
    }


def _paired_error_comparison(records: list[dict[str, object]]) -> dict[str, object]:
    """Compare equations row-for-row at one frozen geometry and ledger."""

    pcm = np.asarray(
        [float(record["methods"]["ddpcm"]["predicted_kcal_mol"]) for record in records],
        dtype=float,
    )
    cosmo = np.asarray(
        [float(record["methods"]["ddcosmo"]["predicted_kcal_mol"]) for record in records],
        dtype=float,
    )
    pcm_abs = np.asarray(
        [float(record["methods"]["ddpcm"]["absolute_error_kcal_mol"]) for record in records],
        dtype=float,
    )
    cosmo_abs = np.asarray(
        [float(record["methods"]["ddcosmo"]["absolute_error_kcal_mol"]) for record in records],
        dtype=float,
    )
    tolerance = 1.0e-12
    return {
        "record_count": int(len(records)),
        "mean_ddcosmo_minus_ddpcm_kcal_mol": float(np.mean(cosmo - pcm)),
        "minimum_ddcosmo_minus_ddpcm_kcal_mol": float(np.min(cosmo - pcm)),
        "maximum_ddcosmo_minus_ddpcm_kcal_mol": float(np.max(cosmo - pcm)),
        "ddcosmo_lower_absolute_error_count": int(
            np.sum(cosmo_abs < pcm_abs - tolerance)
        ),
        "ddpcm_lower_absolute_error_count": int(
            np.sum(pcm_abs < cosmo_abs - tolerance)
        ),
        "absolute_error_tie_count": int(
            np.sum(np.abs(cosmo_abs - pcm_abs) <= tolerance)
        ),
        "tie_tolerance_kcal_mol": tolerance,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mol2-root",
        type=Path,
        required=True,
        help=(
            "Directory containing FreeSolv mol2files_gaff/ or that directory "
            "itself; every selected file is hash-verified."
        ),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Private full report path; defaults to WORK_DIR/summary.json.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help=(
            "Re-evaluate only persisted computational-failure rows under the "
            "same immutable run lock; successful rows are never recalculated."
        ),
    )
    return parser.parse_args(argv)


def _record_needs_run(record_path: Path, *, retry_failures: bool) -> bool:
    """Return whether a row is absent or explicitly eligible for retry."""

    if not record_path.exists():
        return True
    observed = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(observed, dict):
        raise RuntimeError(f"Persisted record is not a JSON object: {record_path}.")
    status = observed.get("status")
    if status == "success":
        return False
    if status == "failure":
        return retry_failures
    raise RuntimeError(f"Persisted record has unsupported status {status!r}: {record_path}.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    work_dir = _require_private_path(args.work_dir, label="FreeSolv work directory")
    output_path = _require_private_path(
        args.output if args.output is not None else work_dir / "summary.json",
        label="FreeSolv row-level summary",
    )
    manifest_path = args.manifest.resolve()
    manifest = load_freesolv12_functional_group_manifest(manifest_path)
    records = list(manifest["locked_records"])
    if len(records) != 12:
        raise RuntimeError("The required FreeSolv functional-group panel has 12 records.")
    mol2_root = args.mol2_root.resolve()
    for record in records:
        _resolve_mol2_path(mol2_root, record)

    execution_git_head = _execution_git_head()
    work_dir.mkdir(parents=True, exist_ok=True)
    lock = _run_lock(
        manifest_path=manifest_path,
        records=records,
        execution_git_head=execution_git_head,
    )
    _establish_or_validate_run_lock(work_dir, lock)
    records_dir = work_dir / "records"
    records_dir.mkdir(exist_ok=True)
    pending = [
        record
        for record in records
        if _record_needs_run(
            records_dir / f"{record['compound_id']}.json",
            retry_failures=bool(args.retry_failures),
        )
    ]
    calculator = None
    model_load_seconds = 0.0
    if pending:
        first_atoms = _atoms(pending[0], mol2_root)
        started = time.perf_counter()
        calculator = paired_benchmark._load_calculator(
            first_atoms,
            WATER_SOLVENT,
            work_dir,
            method_profiles=METHOD_PROFILES,
        )
        model_load_seconds = time.perf_counter() - started
        if str(calculator.dtype) != "torch.float64":
            raise RuntimeError("FreeSolv direct Route-2 panel requires float64 MACE-POLAR.")

    for index, record in enumerate(records, start=1):
        record_path = records_dir / f"{record['compound_id']}.json"
        if not _record_needs_run(
            record_path,
            retry_failures=bool(args.retry_failures),
        ):
            continue
        if calculator is None:
            raise AssertionError("A calculator is required for a pending record.")
        try:
            result = _evaluate_record(
                calculator=calculator,
                record=record,
                mol2_root=mol2_root,
                work_dir=work_dir,
            )
            result["status"] = "success"
            print(
                f"{index:02d}/12 {str(record['name']):<24.24} "
                f"PCM={result['methods']['ddpcm']['predicted_kcal_mol']:8.3f} "
                f"COSMO={result['methods']['ddcosmo']['predicted_kcal_mol']:8.3f}",
                flush=True,
            )
        except Exception as exc:  # Persist every failure for deterministic resume.
            result = {
                "compound_id": record["compound_id"],
                "name": record["name"],
                "functional_group": record["functional_group"],
                "status": "failure",
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            }
            print(f"{index:02d}/12 {record['name']} FAILURE: {exc}", flush=True)
        write_json_atomic(record_path, result)

    completed = [
        json.loads((records_dir / f"{record['compound_id']}.json").read_text())
        for record in records
    ]
    failures = [record for record in completed if record.get("status") != "success"]
    if failures:
        status = "incomplete-computational-failure"
        metrics: dict[str, object] | None = None
        paired_comparison: dict[str, object] | None = None
        acceptance: dict[str, object] | None = None
        exit_code = 2
    else:
        ddpcm_gate = evaluate_freesolv12_functional_group_predictions(
            [
                {
                    "compound_id": row["compound_id"],
                    "predicted_kcal_mol": row["methods"]["ddpcm"]["predicted_kcal_mol"],
                }
                for row in completed
            ],
            manifest_path=manifest_path,
        ).as_dict()
        ddcosmo_gate = evaluate_freesolv12_functional_group_predictions(
            [
                {
                    "compound_id": row["compound_id"],
                    "predicted_kcal_mol": row["methods"]["ddcosmo"]["predicted_kcal_mol"],
                }
                for row in completed
            ],
            manifest_path=manifest_path,
        ).as_dict()
        metrics = {
            "ddpcm": _error_metrics(completed, "ddpcm"),
            "ddcosmo": _error_metrics(completed, "ddcosmo"),
        }
        paired_comparison = _paired_error_comparison(completed)
        acceptance = {
            "ddpcm": ddpcm_gate,
            "ddcosmo": ddcosmo_gate,
            "both_equations_pass_strict_freesolv12_gate": (
                ddpcm_gate["status"] == "pass" and ddcosmo_gate["status"] == "pass"
            ),
        }
        if acceptance["both_equations_pass_strict_freesolv12_gate"]:
            status = "complete-strict-freesolv12-pass"
            exit_code = 0
        else:
            status = "complete-strict-freesolv12-fail"
            exit_code = 1

    runtime: dict[str, object] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "model_load_seconds_this_invocation": model_load_seconds,
    }
    if calculator is not None:
        runtime["mace_polar_checkpoint"] = dict(
            calculator.mace_polar_checkpoint_provenance
        )
        runtime["mace_dtype"] = str(calculator.dtype)
    payload = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_lock": lock,
        "scientific_identity": {
            "solute_model": "official-unmodified-MACE-POLAR-1-M",
            "solute_source": "MACE-POLAR coarse point l<=1 multipoles",
            "receiver": "local-jet",
            "solvent": WATER_SOLVENT,
            "continuum_equations": ["pyddx ddPCM", "pyddx ddCOSMO"],
            "nonpolar_term": "PySCF-2.13.1-SMD-CDS",
            "energy_ledger": PCM_HALF_COUPLING_ONLY_V1,
            "energy_composition": paired_benchmark.paired_energy_composition(
                PCM_HALF_COUPLING_ONLY_V1
            ),
            "strict_original_smd_equivalence": False,
            "forces_available": False,
        },
        "claim_boundary": (
            "This is a locked twelve-record FreeSolv energy diagnostic with "
            "ten distinct functional groups and no result-driven parameter "
            "selection. It reports direct PCM and ddCOSMO equation errors but "
            "does not establish common variational electronic-energy semantics, "
            "a smooth solution-phase PES, public forces, multi-solvent accuracy, "
            "or blind population generalization."
        ),
        "records_required": len(records),
        "success_count": len(completed) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "metrics": metrics,
        "paired_equation_comparison": paired_comparison,
        "strict_freesolv12_acceptance": acceptance,
        "runtime": runtime,
        "records": completed,
    }
    write_json_atomic(output_path, payload)
    print(
        f"FreeSolv12 paired direct Route-2 status={status}; wrote {output_path}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
