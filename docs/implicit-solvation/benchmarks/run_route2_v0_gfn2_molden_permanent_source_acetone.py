#!/usr/bin/env python3
"""Certify only a zero-field GFN2-xTB permanent source for the V0 acetone gate.

This runner deliberately does not evaluate an xTB response, a continuum, a
solvation energy, or an experimental label.  Its sole positive result is that
the version-bound MOLDEN AO density reproduces the converged xTB valence count
and permanent molecular dipole when paired with the effective cores declared
by the shipped GFN2 parameter file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ID = "route2-v0-gfn2-molden-permanent-source-acetone-v2"
PREREGISTRATION_PROTOCOL_ID = "route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_gfn2_molden_permanent_source_acetone.py"
)
SOURCE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_gfn2_molden_permanent_source.py"
)
GEOMETRY_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-gfn2-molden-permanent-source-acetone-v1/acetone.xyz"
)
DEFAULT_XTB = Path("/home/axie/xtb/xtb-dist/bin/xtb")
DEFAULT_PARAMETER_FILE = Path("/home/axie/xtb/xtb-dist/share/xtb/param_gfn2-xtb.txt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--xtb", type=Path, default=DEFAULT_XTB)
    parser.add_argument("--parameter-file", type=Path, default=DEFAULT_PARAMETER_FILE)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain one JSON object.")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _require_clean_tracked_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The permanent-source runner requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative in (
        PREREG_RELATIVE_PATH,
        RUNNER_RELATIVE_PATH,
        SOURCE_MODULE_RELATIVE_PATH,
        GEOMETRY_RELATIVE_PATH,
    ):
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _validate_preregistration(
    *,
    xtb: Path,
    parameter_file: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    preregistration = _load_json(
        REPO_ROOT / PREREG_RELATIVE_PATH,
        label="GFN2 MOLDEN permanent-source preregistration",
    )
    expected_sources = {
        RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH),
        SOURCE_MODULE_RELATIVE_PATH: _sha256(REPO_ROOT / SOURCE_MODULE_RELATIVE_PATH),
    }
    expected_geometry = _sha256(REPO_ROOT / GEOMETRY_RELATIVE_PATH)
    contract = preregistration.get("execution_contract")
    if (
        preregistration.get("protocol_id") != PREREGISTRATION_PROTOCOL_ID
        or preregistration.get("status") != "frozen-before-execution"
        or not isinstance(contract, dict)
        or contract.get("source_sha256") != expected_sources
        or contract.get("geometry_sha256") != expected_geometry
    ):
        raise RuntimeError("The GFN2 MOLDEN permanent-source protocol is not frozen.")
    identity = preregistration.get("candidate_identity")
    if not isinstance(identity, dict) or identity != {
        "electronic_method": "GFN2-xTB",
        "mode": "zero-field-closed-shell-permanent-source",
        "molden_output": True,
        "xtbout_json_output": True,
        "response_role": "forbidden",
    }:
        raise RuntimeError("The permanent-source candidate identity changed.")
    hard_constraints = preregistration.get("hard_constraints")
    if hard_constraints != {
        "gas_phase_gfn2_xtb_only": True,
        "xtb_builtin_solvation_disabled": True,
        "external_embedding_disabled": True,
        "xtb_field_response_used": False,
        "experimental_solvation_labels_read": False,
        "post_training": False,
        "fine_tuning": False,
        "target_fit_or_calibration": False,
        "response_rescaling_or_eigenvalue_clipping": False,
        "field_step_or_pair_distance_selection_after_execution": False,
        "legacy_public_route_changed": False,
        "no_runtime_qm_in_candidate": True,
    }:
        raise RuntimeError("The permanent-source hard constraints changed.")
    numerical = preregistration.get("numerical_gates")
    if numerical != {
        "mo_metric_relative_operator_max": 5.0e-8,
        "valence_electron_count_absolute_max": 5.0e-8,
        "effective_charge_absolute_max": 5.0e-8,
        "dipole_round_trip_e_bohr_max": 1.0e-6,
    }:
        raise RuntimeError("The permanent-source numerical gates changed.")
    runtime = preregistration.get("runtime_identity")
    if not isinstance(runtime, dict) or (
        runtime.get("xtb_binary_sha256") != _sha256(xtb)
        or runtime.get("gfn2_parameter_sha256") != _sha256(parameter_file)
        or runtime.get("xtb_version") != "6.7.1 (edcfbbe)"
    ):
        raise RuntimeError("The permanent-source xTB runtime identity changed.")
    return preregistration, expected_sources


def _geometry() -> tuple[tuple[str, ...], np.ndarray]:
    path = REPO_ROOT / GEOMETRY_RELATIVE_PATH
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise RuntimeError("Frozen acetone geometry is truncated.")
    try:
        count = int(lines[0].strip())
    except ValueError as error:
        raise RuntimeError("Frozen acetone geometry atom count is invalid.") from error
    records = [line.split() for line in lines[2:]]
    if (
        count != 10
        or len(records) != count
        or any(len(record) != 4 for record in records)
    ):
        raise RuntimeError("Frozen acetone geometry has an unexpected atom layout.")
    symbols = tuple(record[0] for record in records)
    positions = np.asarray(
        [[float(value) for value in record[1:]] for record in records]
    )
    if symbols != ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H") or (
        positions.shape != (10, 3) or not np.all(np.isfinite(positions))
    ):
        raise RuntimeError("Frozen acetone geometry identity changed.")
    return symbols, positions


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _load_source_module() -> Any:
    import importlib

    path = REPO_ROOT / SOURCE_MODULE_RELATIVE_PATH
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module(
        "maple.function.calculator.extra_correction.implicit."
        "route2_v0_gfn2_molden_permanent_source"
    )
    module_path = Path(module.__file__).resolve()
    if module_path != path.resolve():
        raise RuntimeError(
            "GFN2 MOLDEN source module was not imported from the frozen checkout."
        )
    return module


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    work_dir = arguments.work_dir.resolve()
    xtb = arguments.xtb.resolve()
    parameter_file = arguments.parameter_file.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not xtb.is_file() or not parameter_file.is_file():
        raise FileNotFoundError(xtb if not xtb.is_file() else parameter_file)
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError("Permanent-source work directory must be empty.")
    work_dir.mkdir(parents=True, exist_ok=True)
    git_head = _require_clean_tracked_checkout()
    preregistration, source_hashes = _validate_preregistration(
        xtb=xtb,
        parameter_file=parameter_file,
    )
    symbols, positions = _geometry()
    geometry_path = REPO_ROOT / GEOMETRY_RELATIVE_PATH
    run_geometry = work_dir / "acetone.xyz"
    shutil.copyfile(geometry_path, run_geometry)
    command = [
        str(xtb),
        "acetone.xyz",
        "--gfn",
        "2",
        "--parallel",
        "1",
        "--molden",
        "--json",
    ]
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=work_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_seconds = time.perf_counter() - started
    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"GFN2-xTB permanent-source run failed: {process.stderr}")
    molden_path = work_dir / "molden.input"
    json_path = work_dir / "xtbout.json"
    if not molden_path.is_file() or not json_path.is_file():
        raise RuntimeError("GFN2-xTB did not emit both MOLDEN and JSON source outputs.")
    source_module = _load_source_module()
    reference = source_module.load_route2_v0_gfn2_molden_permanent_reference(
        molden_path=molden_path,
        xtbout_json_path=json_path,
        stdout_path=stdout_path,
        parameter_file_path=parameter_file,
        expected_net_charge_e=0.0,
    )
    artifact = {
        "schema_version": 1,
        "artifact": ARTIFACT_ID,
        "status": "pass",
        "claim_boundary": (
            "This is a zero-field GFN2-xTB MOLDEN permanent-source round-trip only. "
            "It does not admit xTB response, a PCM calculation, a cavity, a force, "
            "a solvation energy, a runtime claim, or an experimental accuracy claim."
        ),
        "execution_git_head": git_head,
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": _sha256(REPO_ROOT / PREREG_RELATIVE_PATH),
        },
        "source_files_sha256": source_hashes,
        "input_files_sha256": {
            GEOMETRY_RELATIVE_PATH: _sha256(geometry_path),
            "gfn2_parameter_file": _sha256(parameter_file),
            "xtb_binary": _sha256(xtb),
        },
        "hard_constraints": preregistration["hard_constraints"],
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
            "geometry_sha256": _sha256(geometry_path),
        },
        "candidate": {
            "electronic_method": "GFN2-xTB",
            "xtb_version": reference.xtb_version,
            "parameter_file_sha256": reference.effective_core_model.parameter_file_sha256,
            "scope": reference.scope,
            "construction": reference.construction,
        },
        "permanent_source_round_trip": {
            "ao_count": reference.density.ao_count,
            "mo_metric_error": reference.density.mo_metric_error,
            "valence_electron_count_e": reference.density.valence_electron_count_e,
            "valence_electron_count_error_e": reference.density.valence_electron_count_error_e,
            "effective_core_charges_e": reference.effective_core_charges_e.tolist(),
            "total_effective_charge_e": reference.total_effective_charge_e,
            "reported_total_dipole_e_bohr": reference.reported_total_dipole_e_bohr.tolist(),
            "reconstructed_total_dipole_e_bohr": reference.reconstructed_total_dipole_e_bohr.tolist(),
            "dipole_round_trip_error_e_bohr": reference.dipole_round_trip_error_e_bohr,
            "molden_sha256": reference.density.molden_sha256,
            "xtbout_json_sha256": reference.xtbout_json_sha256,
            "stdout_sha256": reference.stdout_sha256,
        },
        "numerical_gates": preregistration["numerical_gates"],
        "decision": {
            "verdict": "admit-only-to-preregistered-static-qm-mep-source-gate",
            "rule": (
                "A passing round trip proves only that the zero-field GFN2 MOLDEN "
                "AO density and effective cores reconstruct the xTB permanent source. "
                "Do not reuse the rejected xTB external-field response or run a PCM/"
                "accuracy calculation. The next required gate is a frozen QM static-MEP "
                "comparison, followed by an all-electron cavity-density completion."
            ),
        },
        "runtime": {
            "command": command,
            "elapsed_seconds": elapsed_seconds,
            "returncode": process.returncode,
            "python": sys.version,
            "work_dir": str(work_dir),
        },
    }
    _write_exclusive_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
