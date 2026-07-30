#!/usr/bin/env python3
"""Falsify the zero-field GFN2 MOLDEN permanent source against frozen QM MEP.

The only candidate here is the version-bound, zero-field GFN2 MOLDEN valence
AO density plus the effective core charge derived from the same GFN2 parameter
file.  It is evaluated at every one of the existing 516 frozen exterior points
and compared with a frozen gas-phase omegaB97M-V/def2-TZVPD checkpoint.

The runner does not evaluate a response, PCM, cavity, solvation energy, force,
experimental label, or runtime advantage.  It owns no tunable source
coefficient: a failed preregistered gate rejects the candidate instead of
changing its core convention, points, or threshold.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ID = "route2-v0-gfn2-molden-static-mep-acetone-v2"
PREREGISTRATION_PROTOCOL_ID = "route2-v0-gfn2-molden-static-mep-acetone-prereg-v2"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-gfn2-molden-static-mep-acetone-prereg-v2.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_gfn2_molden_static_mep_acetone.py"
)
STATIC_HELPER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2_v0_gfn2_molden_static_mep.py"
)
PERMANENT_SOURCE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_gfn2_molden_permanent_source.py"
)
PERMANENT_SOURCE_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-gfn2-molden-permanent-source-acetone-v2.json"
)
PERMANENT_SOURCE_PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2.json"
)
GEOMETRY_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-gfn2-molden-permanent-source-acetone-v1/acetone.xyz"
)
POINTS_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-atomic-displacement-source-acetone-v1/"
    "frozen-exterior-qm-mep-points-bohr.npy"
)
POINTS_PROVENANCE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-atomic-displacement-source-acetone-v1/"
    "frozen-exterior-qm-mep-points-provenance.json"
)
RAW_INDUCED_MEP_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-mace-mdp-induced-source-acetone-v1/qm-induced-mep.json"
)
PREFLIGHT_FAILURE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-gfn2-molden-static-mep-acetone-preflight-failure-v1.json"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    STATIC_HELPER_RELATIVE_PATH,
    PERMANENT_SOURCE_MODULE_RELATIVE_PATH,
)
INPUT_RELATIVE_PATHS = (
    PERMANENT_SOURCE_ARTIFACT_RELATIVE_PATH,
    PERMANENT_SOURCE_PREREG_RELATIVE_PATH,
    GEOMETRY_RELATIVE_PATH,
    POINTS_RELATIVE_PATH,
    POINTS_PROVENANCE_RELATIVE_PATH,
    RAW_INDUCED_MEP_RELATIVE_PATH,
    PREFLIGHT_FAILURE_RELATIVE_PATH,
)
DEFAULT_XTB = Path("/home/axie/xtb/xtb-dist/bin/xtb")
DEFAULT_PARAMETER_FILE = Path("/home/axie/xtb/xtb-dist/share/xtb/param_gfn2-xtb.txt")
DEFAULT_PYSCF_PYTHON = Path("/home/axie/miniconda3/envs/maple/bin/python3.11")
DEFAULT_PYSCF_SITE_PACKAGES = Path(
    "/home/axie/.cache/maple-envs/pyscf-smd-qmref/lib/python3.11/site-packages"
)
DEFAULT_QM_CHECKPOINT = (
    REPO_ROOT / ".omx/benchmarks/route2-qm-ddx-acetone-center-20260725/"
    "pyscf_smd_acetone_wb97mv_def2tzvpd_g3_n50x194sg1.gas.chk"
)
DEFAULT_QM_LEDGER = (
    REPO_ROOT / ".omx/benchmarks/route2-qm-ddx-acetone-center-20260725/"
    "pyscf_smd_acetone_wb97mv_def2tzvpd_g3_n50x194sg1.gas.json"
)
EXPECTED_QM_REFERENCE = {
    "electronic_structure": "omegaB97M-V",
    "basis": "def2-tzvpd",
    "reference": "RKS",
    "density_fitting": True,
    "charge": 0,
    "spin": 0,
    "phase": "gas",
    "semilocal_grid_level": 3,
    "nonlocal_grid_profile": "pyscf-official-50x194-sg1",
    "scf_energy_tolerance_hartree": 1.0e-10,
    "scf_gradient_tolerance": 1.0e-7,
    "maximum_scf_cycles": 100,
}
EXPECTED_QM_INDUCED_MEP_METHOD = {
    "electronic_structure": "omegaB97M-V",
    "pyscf_xc_token": "wb97m-v",
    "basis": "def2-tzvpd",
    "reference": "RKS",
    "density_fitting": True,
    "charge": 0,
    "spin": 0,
    "semilocal_grid_level": 3,
    "nonlocal_grid_profile": "50x194-SG1",
    "scf_energy_tolerance_hartree": 1.0e-10,
    "scf_gradient_tolerance": 1.0e-7,
    "maximum_scf_cycles": 100,
}
EXPECTED_GEOMETRY_SYMBOLS = ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--xtb", type=Path, default=DEFAULT_XTB)
    parser.add_argument("--parameter-file", type=Path, default=DEFAULT_PARAMETER_FILE)
    parser.add_argument("--pyscf-python", type=Path, default=DEFAULT_PYSCF_PYTHON)
    parser.add_argument(
        "--pyscf-site-packages", type=Path, default=DEFAULT_PYSCF_SITE_PACKAGES
    )
    parser.add_argument("--qm-checkpoint", type=Path, default=DEFAULT_QM_CHECKPOINT)
    parser.add_argument("--qm-ledger", type=Path, default=DEFAULT_QM_LEDGER)
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


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {label}: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain exactly one JSON object.")
    return payload


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _relative_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(abs(left - right), initial=0.0)
        / max(float(np.max(abs(right), initial=0.0)), 1.0e-30)
    )


def _upper_check(value: float, maximum: float) -> dict[str, float | bool]:
    return {"value": value, "maximum": maximum, "passes": bool(value <= maximum)}


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _require_clean_tracked_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The GFN2 static-MEP runner requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative_path in (
        *SOURCE_RELATIVE_PATHS,
        *INPUT_RELATIVE_PATHS,
        PREREG_RELATIVE_PATH,
    ):
        _git("ls-files", "--error-unmatch", relative_path)
    return _git("rev-parse", "HEAD")


def _load_frozen_geometry() -> tuple[tuple[str, ...], np.ndarray]:
    path = REPO_ROOT / GEOMETRY_RELATIVE_PATH
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise RuntimeError("Frozen GFN2 acetone geometry is truncated.")
    try:
        count = int(lines[0].strip())
    except ValueError as error:
        raise RuntimeError("Frozen GFN2 acetone atom count is invalid.") from error
    records = [line.split() for line in lines[2:]]
    if (
        count != 10
        or len(records) != count
        or any(len(record) != 4 for record in records)
    ):
        raise RuntimeError("Frozen GFN2 acetone geometry layout is invalid.")
    symbols = tuple(record[0] for record in records)
    positions = np.asarray(
        [[float(value) for value in record[1:]] for record in records]
    )
    if (
        symbols != EXPECTED_GEOMETRY_SYMBOLS
        or positions.shape != (10, 3)
        or not np.all(np.isfinite(positions))
    ):
        raise RuntimeError("Frozen GFN2 acetone geometry identity changed.")
    return symbols, positions


def _validate_points() -> np.ndarray:
    points_path = REPO_ROOT / POINTS_RELATIVE_PATH
    provenance_path = REPO_ROOT / POINTS_PROVENANCE_RELATIVE_PATH
    raw_path = REPO_ROOT / RAW_INDUCED_MEP_RELATIVE_PATH
    provenance = _load_json(provenance_path, label="frozen exterior point provenance")
    if (
        provenance.get("artifact")
        != "route2-v0-atomic-displacement-source-acetone-v1-frozen-exterior-qm-mep-points"
        or provenance.get("status") != "pass"
        or provenance.get("origin", {}).get("raw_qm_induced_mep_sha256")
        != _sha256(raw_path)
        or provenance.get("points", {}).get("file_sha256") != _sha256(points_path)
    ):
        raise RuntimeError("Frozen exterior point provenance is invalid.")
    try:
        points = np.asarray(np.load(points_path, allow_pickle=False), dtype=float)
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot load frozen exterior MEP points.") from error
    if (
        points.shape != (516, 3)
        or not np.all(np.isfinite(points))
        or provenance.get("points", {}).get("array_sha256") != _sha256_array(points)
    ):
        raise RuntimeError("Frozen exterior MEP point set changed.")
    return points


def _validate_prior_permanent_source() -> dict[str, Any]:
    artifact = _load_json(
        REPO_ROOT / PERMANENT_SOURCE_ARTIFACT_RELATIVE_PATH,
        label="prior GFN2 MOLDEN permanent-source artifact",
    )
    preregistration = _load_json(
        REPO_ROOT / PERMANENT_SOURCE_PREREG_RELATIVE_PATH,
        label="prior GFN2 MOLDEN permanent-source preregistration",
    )
    expected_hard_constraints = {
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
    }
    if (
        artifact.get("artifact") != "route2-v0-gfn2-molden-permanent-source-acetone-v2"
        or artifact.get("status") != "pass"
        or artifact.get("decision", {}).get("verdict")
        != "admit-only-to-preregistered-static-qm-mep-source-gate"
        or artifact.get("hard_constraints") != expected_hard_constraints
        or artifact.get("preregistration", {}).get("sha256")
        != _sha256(REPO_ROOT / PERMANENT_SOURCE_PREREG_RELATIVE_PATH)
        or preregistration.get("protocol_id")
        != "route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("Prior GFN2 MOLDEN permanent-source gate is not admissible.")
    round_trip = artifact.get("permanent_source_round_trip")
    if not isinstance(round_trip, dict) or round_trip.get("ao_count") != 22:
        raise RuntimeError("Prior GFN2 MOLDEN source representation is incomplete.")
    return artifact


def _validate_qm_ledger(path: Path) -> dict[str, Any]:
    ledger = _load_json(path, label="frozen QM checkpoint ledger")
    system = ledger.get("system")
    numerics = ledger.get("numerics")
    if not isinstance(system, dict) or not isinstance(numerics, dict):
        raise RuntimeError("Frozen QM checkpoint ledger is incomplete.")
    actual = {
        "electronic_structure": system.get("method"),
        "basis": system.get("basis"),
        "reference": system.get("reference"),
        "density_fitting": system.get("density_fitting"),
        "charge": system.get("charge"),
        "spin": 0,
        "phase": ledger.get("phase"),
        "semilocal_grid_level": numerics.get("grid_level"),
        "nonlocal_grid_profile": numerics.get("nlc_grid_profile"),
        "scf_energy_tolerance_hartree": numerics.get("scf_energy_tolerance_hartree"),
        "scf_gradient_tolerance": numerics.get("scf_gradient_tolerance"),
        "maximum_scf_cycles": numerics.get("maximum_scf_cycles"),
    }
    if ledger.get("status") != "pass" or actual != EXPECTED_QM_REFERENCE:
        raise RuntimeError("Frozen QM checkpoint method or numerical contract changed.")
    return ledger


def _validate_preregistration(
    *,
    xtb: Path,
    parameter_file: Path,
    pyscf_python: Path,
    pyscf_site_packages: Path,
    qm_checkpoint: Path,
    qm_ledger: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    preregistration = _load_json(
        REPO_ROOT / PREREG_RELATIVE_PATH,
        label="GFN2 static-MEP preregistration",
    )
    source_hashes = {
        relative_path: _sha256(REPO_ROOT / relative_path)
        for relative_path in SOURCE_RELATIVE_PATHS
    }
    input_hashes = {
        relative_path: _sha256(REPO_ROOT / relative_path)
        for relative_path in INPUT_RELATIVE_PATHS
    }
    runtime_hashes = {
        "xtb_binary_sha256": _sha256(xtb),
        "gfn2_parameter_sha256": _sha256(parameter_file),
        "pyscf_python_sha256": _sha256(pyscf_python.resolve()),
        "pyscf_init_sha256": _sha256(pyscf_site_packages / "pyscf/__init__.py"),
        "qm_checkpoint_sha256": _sha256(qm_checkpoint),
        "qm_ledger_sha256": _sha256(qm_ledger),
    }
    expected_hard_constraints = {
        "gas_phase_gfn2_xtb_only": True,
        "xtb_builtin_solvation_disabled": True,
        "external_embedding_disabled": True,
        "xtb_field_response_used": False,
        "continuum_or_pcm_invoked": False,
        "experimental_solvation_labels_read": False,
        "post_training": False,
        "fine_tuning": False,
        "target_fit_or_calibration": False,
        "response_rescaling_or_eigenvalue_clipping": False,
        "mep_point_subset_selected_after_execution": False,
        "new_qm_scf_executed": False,
        "legacy_public_route_changed": False,
    }
    expected_numerical_gates = {
        "candidate_pyscf_overlap_relative_frobenius_max": 5.0e-8,
        "candidate_pyscf_overlap_max_abs_max": 5.0e-8,
        "candidate_pyscf_mo_metric_max": 5.0e-8,
        "candidate_pyscf_electronic_dipole_e_bohr_max": 1.0e-6,
        "candidate_pyscf_total_dipole_e_bohr_max": 1.0e-6,
        "qm_checkpoint_mo_metric_max": 1.0e-7,
        "qm_checkpoint_electron_count_e_max": 1.0e-7,
        "qm_checkpoint_total_charge_e_max": 1.0e-7,
        "geometry_max_abs_error_bohr_max": 1.0e-8,
        "qm_checkpoint_energy_abs_error_hartree_max": 1.0e-9,
        "qm_zero_field_dipole_e_bohr_abs_error_max": 1.0e-8,
    }
    expected_scientific_gates = {
        "static_mep_relative_frobenius_max": 0.2,
        "static_mep_relative_max_abs_max": 0.3,
        "static_dipole_relative_frobenius_max": 0.2,
    }
    protocol_revision = preregistration.get("protocol_revision")
    contract = preregistration.get("execution_contract")
    if (
        preregistration.get("protocol_id") != PREREGISTRATION_PROTOCOL_ID
        or preregistration.get("status") != "frozen-before-execution"
        or not isinstance(contract, dict)
        or contract.get("source_sha256") != source_hashes
        or contract.get("input_sha256") != input_hashes
        or preregistration.get("runtime_identity") != runtime_hashes
        or preregistration.get("qm_reference") != EXPECTED_QM_REFERENCE
        or preregistration.get("hard_constraints") != expected_hard_constraints
        or preregistration.get("numerical_gates") != expected_numerical_gates
        or preregistration.get("scientific_falsification_gates")
        != expected_scientific_gates
        or not isinstance(protocol_revision, dict)
        or protocol_revision.get("supersedes_protocol_id")
        != "route2-v0-gfn2-molden-static-mep-acetone-prereg-v1"
        or protocol_revision.get("preflight_failure")
        != {
            "path": PREFLIGHT_FAILURE_RELATIVE_PATH,
            "sha256": _sha256(REPO_ROOT / PREFLIGHT_FAILURE_RELATIVE_PATH),
        }
    ):
        raise RuntimeError("The GFN2 static-MEP protocol is not frozen.")
    return preregistration, source_hashes, input_hashes


def _run_xtb(*, xtb: Path, work_dir: Path) -> tuple[Path, Path, Path, float]:
    geometry = REPO_ROOT / GEOMETRY_RELATIVE_PATH
    run_geometry = work_dir / "acetone.xyz"
    shutil.copyfile(geometry, run_geometry)
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
    stdout_path = work_dir / "xtb.stdout.txt"
    stderr_path = work_dir / "xtb.stderr.txt"
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"GFN2-xTB static-MEP source run failed: {process.stderr}")
    molden_path = work_dir / "molden.input"
    json_path = work_dir / "xtbout.json"
    if not molden_path.is_file() or not json_path.is_file():
        raise RuntimeError("GFN2-xTB did not emit MOLDEN and JSON outputs.")
    return molden_path, json_path, stdout_path, elapsed_seconds


def _run_static_mep_helper(
    *,
    pyscf_python: Path,
    molden_path: Path,
    json_path: Path,
    stdout_path: Path,
    parameter_file: Path,
    pyscf_site_packages: Path,
    work_dir: Path,
    qm_checkpoint: Path,
) -> tuple[dict[str, Any], float]:
    output_path = work_dir / "static-mep-helper.json"
    command = [
        str(pyscf_python),
        str(REPO_ROOT / STATIC_HELPER_RELATIVE_PATH),
        "--molden",
        str(molden_path),
        "--xtbout-json",
        str(json_path),
        "--xtb-stdout",
        str(stdout_path),
        "--parameter-file",
        str(parameter_file),
        "--points",
        str(REPO_ROOT / POINTS_RELATIVE_PATH),
        "--qm-checkpoint",
        str(qm_checkpoint),
        "--output",
        str(output_path),
    ]
    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(pyscf_site_packages)
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    elapsed_seconds = time.perf_counter() - started
    (work_dir / "static-mep-helper.stdout.txt").write_text(
        process.stdout, encoding="utf-8"
    )
    (work_dir / "static-mep-helper.stderr.txt").write_text(
        process.stderr, encoding="utf-8"
    )
    if process.returncode != 0:
        raise RuntimeError(f"GFN2 static-MEP helper failed: {process.stderr}")
    return (
        _load_json(output_path, label="GFN2 static-MEP helper output"),
        elapsed_seconds,
    )


def _helper_arrays(
    helper: dict[str, Any], *, point_count: int
) -> tuple[np.ndarray, np.ndarray]:
    if (
        helper.get("artifact") != "route2-v0-gfn2-molden-static-mep-helper-v1"
        or helper.get("status") != "pass"
    ):
        raise RuntimeError("GFN2 static-MEP helper did not provide a valid raw result.")
    raw = helper.get("raw_static_potential")
    if not isinstance(raw, dict):
        raise RuntimeError("GFN2 static-MEP helper omits raw potentials.")
    candidate = np.asarray(
        raw.get("candidate_effective_core_plus_valence_hartree_per_e"), dtype=float
    )
    qm = np.asarray(raw.get("qm_all_electron_hartree_per_e"), dtype=float)
    if (
        candidate.shape != (point_count,)
        or qm.shape != (point_count,)
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(qm))
        or raw.get("candidate_sha256") != _sha256_array(candidate)
        or raw.get("qm_sha256") != _sha256_array(qm)
    ):
        raise RuntimeError("GFN2 static-MEP helper raw potentials are invalid.")
    return candidate, qm


def _representation_checks(
    helper: dict[str, Any], gates: dict[str, Any]
) -> dict[str, dict[str, float | bool]]:
    candidate = helper.get("candidate_representation")
    qm = helper.get("qm_reference")
    if not isinstance(candidate, dict) or not isinstance(qm, dict):
        raise RuntimeError(
            "GFN2 static-MEP helper representation records are incomplete."
        )
    qm_density = qm.get("checkpoint_ao_density")
    if not isinstance(qm_density, dict):
        raise RuntimeError("GFN2 static-MEP helper QM density record is incomplete.")
    values = {
        "candidate_pyscf_overlap_relative_frobenius": candidate.get(
            "pyscf_vs_molden_overlap_relative_frobenius"
        ),
        "candidate_pyscf_overlap_max_abs": candidate.get(
            "pyscf_vs_molden_overlap_max_abs"
        ),
        "candidate_pyscf_mo_metric": candidate.get("pyscf_ao_metric_error"),
        "candidate_pyscf_electronic_dipole_e_bohr": candidate.get(
            "pyscf_vs_molden_electronic_dipole_error_e_bohr"
        ),
        "candidate_pyscf_total_dipole_e_bohr": candidate.get(
            "pyscf_vs_molden_total_dipole_error_e_bohr"
        ),
        "qm_checkpoint_mo_metric": qm_density.get("mo_metric_error"),
        "qm_checkpoint_electron_count_e": qm_density.get("electron_count_error_e"),
        "qm_checkpoint_total_charge_e": abs(float(qm.get("total_charge_e"))),
        "geometry_max_abs_error_bohr": qm.get("geometry_max_abs_error_bohr"),
    }
    checks: dict[str, dict[str, float | bool]] = {}
    for name, raw_value in values.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Static-MEP helper omits finite {name}.") from error
        if not np.isfinite(value):
            raise RuntimeError(f"Static-MEP helper has non-finite {name}.")
        checks[name] = _upper_check(value, float(gates[f"{name}_max"]))
    return checks


def _validate_qm_reference_record(
    helper: dict[str, Any], gates: dict[str, Any]
) -> dict[str, dict[str, float | bool]]:
    """Bind the checkpoint to the frozen finite-field QM state observables.

    The older finite-field record stores a density *digest* from an independent
    SCF execution, while this gate intentionally loads a frozen checkpoint
    rather than rerunning SCF.  Its byte-level density digest is therefore not
    a valid cross-execution identity.  The checkpoint's preregistered file hash
    is the immutable identity; this check additionally requires the two frozen
    executions to agree in total energy and permanent dipole well below the
    scientific source-screen scale.
    """

    raw_induced = _load_json(
        REPO_ROOT / RAW_INDUCED_MEP_RELATIVE_PATH,
        label="frozen QM induced-MEP record",
    )
    zero_field = raw_induced.get("zero_field")
    qm_reference = helper.get("qm_reference")
    if (
        raw_induced.get("status") != "pass"
        or raw_induced.get("method") != EXPECTED_QM_INDUCED_MEP_METHOD
        or not isinstance(zero_field, dict)
        or not isinstance(qm_reference, dict)
    ):
        raise RuntimeError("Static-MEP QM reference record is incompatible.")
    checkpoint_density = qm_reference.get("checkpoint_ao_density")
    if not isinstance(checkpoint_density, dict):
        raise RuntimeError("Static-MEP QM checkpoint record is incomplete.")
    try:
        reference_energy = float(zero_field["energy_hartree"])
        checkpoint_energy = float(checkpoint_density["energy_hartree"])
        reference_dipole = np.asarray(zero_field["dipole_e_bohr"], dtype=float)
        checkpoint_dipole = np.asarray(qm_reference["total_dipole_e_bohr"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Static-MEP QM reference observables are incomplete."
        ) from error
    if (
        not np.isfinite(reference_energy)
        or not np.isfinite(checkpoint_energy)
        or reference_dipole.shape != (3,)
        or checkpoint_dipole.shape != (3,)
        or not np.all(np.isfinite(reference_dipole))
        or not np.all(np.isfinite(checkpoint_dipole))
    ):
        raise RuntimeError("Static-MEP QM reference observables are invalid.")
    checks = {
        "qm_checkpoint_energy_abs_error_hartree": _upper_check(
            abs(checkpoint_energy - reference_energy),
            float(gates["qm_checkpoint_energy_abs_error_hartree_max"]),
        ),
        "qm_zero_field_dipole_e_bohr_abs_error": _upper_check(
            float(np.linalg.norm(checkpoint_dipole - reference_dipole)),
            float(gates["qm_zero_field_dipole_e_bohr_abs_error_max"]),
        ),
    }
    if not all(check["passes"] for check in checks.values()):
        raise RuntimeError(
            "Static-MEP QM checkpoint is not physically bound to the frozen record."
        )
    return checks


def _validate_helper_runtime(helper: dict[str, Any]) -> None:
    qm_reference = helper.get("qm_reference")
    if (
        not isinstance(qm_reference, dict)
        or qm_reference.get("pyscf_version") != "2.13.1"
    ):
        raise RuntimeError("Static-MEP helper did not use the frozen PySCF runtime.")


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    work_dir = arguments.work_dir.resolve()
    xtb = arguments.xtb.resolve()
    parameter_file = arguments.parameter_file.resolve()
    pyscf_python = arguments.pyscf_python.resolve()
    pyscf_site_packages = arguments.pyscf_site_packages.resolve()
    qm_checkpoint = arguments.qm_checkpoint.resolve()
    qm_ledger = arguments.qm_ledger.resolve()
    if output.exists():
        raise FileExistsError(output)
    required_runtime_paths = (
        xtb,
        parameter_file,
        pyscf_python,
        pyscf_site_packages / "pyscf/__init__.py",
        qm_checkpoint,
        qm_ledger,
    )
    if missing := [str(path) for path in required_runtime_paths if not path.is_file()]:
        raise FileNotFoundError(f"Missing static-MEP runtime input(s): {missing}")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError("Static-MEP work directory must be empty.")
    git_head = _require_clean_tracked_checkout()
    symbols, positions = _load_frozen_geometry()
    points = _validate_points()
    prior_source = _validate_prior_permanent_source()
    _validate_qm_ledger(qm_ledger)
    preregistration, source_hashes, input_hashes = _validate_preregistration(
        xtb=xtb,
        parameter_file=parameter_file,
        pyscf_python=pyscf_python,
        pyscf_site_packages=pyscf_site_packages,
        qm_checkpoint=qm_checkpoint,
        qm_ledger=qm_ledger,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    molden_path, json_path, stdout_path, xtb_elapsed = _run_xtb(
        xtb=xtb, work_dir=work_dir
    )
    helper, helper_elapsed = _run_static_mep_helper(
        pyscf_python=pyscf_python,
        molden_path=molden_path,
        json_path=json_path,
        stdout_path=stdout_path,
        parameter_file=parameter_file,
        pyscf_site_packages=pyscf_site_packages,
        work_dir=work_dir,
        qm_checkpoint=qm_checkpoint,
    )
    _validate_helper_runtime(helper)
    qm_record_identity = _validate_qm_reference_record(
        helper, preregistration["numerical_gates"]
    )
    candidate, qm = _helper_arrays(helper, point_count=len(points))
    representation = _representation_checks(helper, preregistration["numerical_gates"])
    candidate_representation = helper["candidate_representation"]
    prior_round_trip = prior_source["permanent_source_round_trip"]
    if candidate_representation.get("molden_sha256") != prior_round_trip.get(
        "molden_sha256"
    ):
        raise RuntimeError(
            "Static-MEP xTB MOLDEN output differs from its frozen source gate."
        )
    if helper.get("points", {}).get("array_sha256") != _sha256_array(points):
        raise RuntimeError("Static-MEP helper did not use every frozen exterior point.")
    scientific_gates = preregistration["scientific_falsification_gates"]
    candidate_dipole = np.asarray(
        candidate_representation.get("molden_total_dipole_e_bohr"), dtype=float
    )
    qm_dipole = np.asarray(
        helper["qm_reference"].get("total_dipole_e_bohr"), dtype=float
    )
    if candidate_dipole.shape != (3,) or qm_dipole.shape != (3,):
        raise RuntimeError("Static-MEP helper dipoles are invalid.")
    scientific_checks = {
        "static_mep_relative_frobenius": _upper_check(
            _relative_frobenius(candidate, qm),
            float(scientific_gates["static_mep_relative_frobenius_max"]),
        ),
        "static_mep_relative_max_abs": _upper_check(
            _relative_max_abs(candidate, qm),
            float(scientific_gates["static_mep_relative_max_abs_max"]),
        ),
        "static_dipole_relative_frobenius": _upper_check(
            _relative_frobenius(candidate_dipole, qm_dipole),
            float(scientific_gates["static_dipole_relative_frobenius_max"]),
        ),
    }
    passes_all = (
        all(check["passes"] for check in qm_record_identity.values())
        and all(check["passes"] for check in representation.values())
        and all(check["passes"] for check in scientific_checks.values())
    )
    artifact = {
        "schema_version": 1,
        "artifact": ARTIFACT_ID,
        "status": "pass" if passes_all else "reject",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": git_head,
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": _sha256(REPO_ROOT / PREREG_RELATIVE_PATH),
            "protocol_id": PREREGISTRATION_PROTOCOL_ID,
        },
        "claim_boundary": preregistration["claim_boundary"],
        "hard_constraints": preregistration["hard_constraints"],
        "source_files_sha256": source_hashes,
        "input_files_sha256": input_hashes,
        "runtime_identity": preregistration["runtime_identity"],
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
            "point_count": len(points),
            "points_sha256": _sha256_array(points),
        },
        "prior_permanent_source_gate": {
            "artifact": prior_source["artifact"],
            "artifact_sha256": _sha256(
                REPO_ROOT / PERMANENT_SOURCE_ARTIFACT_RELATIVE_PATH
            ),
            "molden_sha256": prior_round_trip["molden_sha256"],
        },
        "qm_record_identity_checks": qm_record_identity,
        "representation_checks": representation,
        "scientific_falsification": {
            "checks": scientific_checks,
            "passes_all_registered_checks": passes_all,
            "verdict": (
                "admit-permanent-source-only-to-all-electron-cavity-and-common-scalar-kkt-gates"
                if passes_all
                else "reject-gfn2-molden-permanent-source"
            ),
            "admission_boundary": (
                "A pass admits only this permanent source to separately preregistered "
                "all-electron cavity completion and common-scalar/KKT gates. It does not "
                "establish response, a continuum result, force/PES, speed advantage, "
                "or experimental solvation accuracy."
            ),
        },
        "raw_static_mep": {
            "candidate_effective_core_plus_valence_hartree_per_e": candidate.tolist(),
            "candidate_sha256": _sha256_array(candidate),
            "qm_all_electron_hartree_per_e": qm.tolist(),
            "qm_sha256": _sha256_array(qm),
            "candidate_representation": candidate_representation,
            "qm_reference": helper["qm_reference"],
        },
        "runtime": {
            "xtb_elapsed_seconds": xtb_elapsed,
            "static_mep_helper_elapsed_seconds": helper_elapsed,
            "boundary": (
                "This is a frozen gas-phase source falsifier. Its timings are not an "
                "end-to-end Route-2 or QM speed comparison."
            ),
        },
    }
    _write_exclusive_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
