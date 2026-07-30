#!/usr/bin/env python3
"""Run the frozen static-QM-MEP falsifier for the V0-ARSP permanent source.

The candidate is fixed before this runner reads a QM potential: frozen
all-electron spherical atomic-HF ground densities, the frozen V0-RK response
completion, and the stationary interaction dual built from other atomic nuclei
and electron densities.  The runner invokes no continuum/PCM calculation,
new QM SCF, experimental solvation label, post-training, fine tuning, fit, or
calibration.  Its sole decision is whether that exact permanent source passes
its preregistered source-physics gates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ID = "route2-v0-atomic-response-stationary-source-acetone-v2"
PREREGISTRATION_PROTOCOL_ID = (
    "route2-v0-atomic-response-stationary-source-acetone-prereg-v2"
)
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-response-stationary-source-acetone-prereg-v2.json"
)
PREFLIGHT_FAILURE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-response-stationary-source-acetone-preflight-failure-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_atomic_response_stationary_source_acetone.py"
)
STATIC_HELPER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2_v0_atomic_response_stationary_static_mep.py"
)
ATOMIC_RESPONSE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_independent_particle_response.py"
)
ATOMIC_SURFACE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_independent_particle_surface.py"
)
RESPONSE_KERNEL_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/route2_v0_response_kernel.py"
)
STATIONARY_SOURCE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_response_stationary_source.py"
)
TABLE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1.npz"
)
TABLE_MANIFEST_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1.json"
)
TABLE_PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-independent-particle-hf-def2-tzvpd-prereg-v1.json"
)
ATOMIC_MAP_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-atomic-map-acetone-v1.json"
)
RESPONSE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-acetone-response-v1.json"
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
RAW_QM_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-mace-mdp-induced-source-acetone-v1/qm-induced-mep.json"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    STATIC_HELPER_RELATIVE_PATH,
    ATOMIC_RESPONSE_MODULE_RELATIVE_PATH,
    ATOMIC_SURFACE_MODULE_RELATIVE_PATH,
    RESPONSE_KERNEL_MODULE_RELATIVE_PATH,
    STATIONARY_SOURCE_MODULE_RELATIVE_PATH,
)
INPUT_RELATIVE_PATHS = (
    TABLE_RELATIVE_PATH,
    TABLE_MANIFEST_RELATIVE_PATH,
    TABLE_PREREG_RELATIVE_PATH,
    ATOMIC_MAP_RELATIVE_PATH,
    RESPONSE_RELATIVE_PATH,
    POINTS_RELATIVE_PATH,
    POINTS_PROVENANCE_RELATIVE_PATH,
    RAW_QM_RELATIVE_PATH,
    PREFLIGHT_FAILURE_RELATIVE_PATH,
)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH
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
HARD_CONSTRAINTS = {
    "all_electron_atomic_reference_only": True,
    "continuum_or_pcm_invoked": False,
    "experimental_solvation_labels_read": False,
    "fine_tuning": False,
    "legacy_public_route_changed": False,
    "mace_density_relabelled_as_electron_density": False,
    "mace_mdp_treated_as_energy_or_force_model": False,
    "map_or_uq_calibration": False,
    "new_qm_scf_executed": False,
    "post_training": False,
    "qeq_or_fitted_charge_transfer_added": False,
    "response_eigenvalue_clipping": False,
    "response_rescaling_or_tempering": False,
    "source_point_subset_selected_after_execution": False,
    "target_fit_or_calibration": False,
}
NUMERICAL_UPPER_GATE_NAMES = (
    "maximum_atomic_mo_metric",
    "reference_electron_count_error",
    "stationarity_residual_inf",
    "support_constraint_residual_inf",
    "stationary_energy_identity_error",
    "permanent_density_electron_count_error",
    "permanent_density_symmetry_error",
    "qm_checkpoint_geometry_max_abs_error_bohr",
    "qm_checkpoint_mo_metric",
    "qm_checkpoint_electron_count_error",
    "qm_checkpoint_total_charge",
    "qm_checkpoint_energy_compatibility",
    "qm_checkpoint_dipole_compatibility",
)
NUMERICAL_LOWER_GATE_NAMES = ("permanent_density_grid_minimum_electron_number_density",)
NUMERICAL_GATE_NAMES = NUMERICAL_UPPER_GATE_NAMES + NUMERICAL_LOWER_GATE_NAMES
SCIENTIFIC_GATE_NAMES = (
    "static_mep_relative_frobenius",
    "static_mep_relative_max_abs",
    "static_dipole_relative_frobenius",
)
SCIENTIFIC_GATE_HELPER_FIELDS = {
    "static_mep_relative_frobenius": "mep_relative_frobenius",
    "static_mep_relative_max_abs": "mep_relative_max_abs",
    "static_dipole_relative_frobenius": "dipole_relative_frobenius",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        raise RuntimeError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain exactly one JSON object.")
    return payload


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1.0e-30)
    )


def _upper_check(value: float, maximum: float) -> dict[str, float | bool]:
    return {"value": value, "maximum": maximum, "passes": bool(value <= maximum)}


def _lower_check(value: float, minimum: float) -> dict[str, float | bool]:
    return {"value": value, "minimum": minimum, "passes": bool(value >= minimum)}


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _require_clean_tracked_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The stationary-source static-MEP runner requires a clean tracked "
            f"checkout; git reported:\n{status}"
        )
    for relative_path in (
        *SOURCE_RELATIVE_PATHS,
        *INPUT_RELATIVE_PATHS,
        PREREG_RELATIVE_PATH,
    ):
        _git("ls-files", "--error-unmatch", relative_path)
    return _git("rev-parse", "HEAD")


def _validate_preregistration() -> tuple[dict[str, Any], dict[str, str], str]:
    preregistration = _load_json(
        DEFAULT_PREREGISTRATION,
        label="V0-ARSP acetone static-MEP preregistration",
    )
    if (
        preregistration.get("protocol_id") != PREREGISTRATION_PROTOCOL_ID
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The V0-ARSP source protocol is not frozen.")
    source_hashes = {
        relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
    }
    input_hashes = {
        relative: _sha256(REPO_ROOT / relative) for relative in INPUT_RELATIVE_PATHS
    }
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict) or (
        contract.get("source_sha256") != source_hashes
        or contract.get("input_sha256") != input_hashes
    ):
        raise RuntimeError(
            "Frozen V0-ARSP sources or inputs changed after preregistration."
        )
    if preregistration.get("hard_constraints") != HARD_CONSTRAINTS:
        raise RuntimeError(
            "The V0-ARSP hard constraints changed after preregistration."
        )
    if preregistration.get("qm_reference") != EXPECTED_QM_REFERENCE:
        raise RuntimeError("The V0-ARSP QM reference changed after preregistration.")
    numerical = preregistration.get("numerical_gates")
    scientific = preregistration.get("scientific_falsification_gates")
    if not isinstance(numerical, dict) or not isinstance(scientific, dict):
        raise RuntimeError("The V0-ARSP protocol omits numerical or scientific gates.")
    expected_numerical_keys = {
        *(f"{name}_max" for name in NUMERICAL_UPPER_GATE_NAMES),
        *(f"{name}_min" for name in NUMERICAL_LOWER_GATE_NAMES),
    }
    if set(numerical) != expected_numerical_keys:
        raise RuntimeError("The V0-ARSP numerical-gate schema changed.")
    if set(scientific) != {f"{name}_max" for name in SCIENTIFIC_GATE_NAMES}:
        raise RuntimeError("The V0-ARSP scientific-gate schema changed.")
    revision = preregistration.get("protocol_revision")
    if not isinstance(revision, dict) or revision.get("supersedes_protocol_id") != (
        "route2-v0-atomic-response-stationary-source-acetone-prereg-v1"
    ):
        raise RuntimeError("The V0-ARSP V2 preflight provenance is invalid.")
    return preregistration, input_hashes, _sha256(DEFAULT_PREREGISTRATION)


def _validate_external_qm_reference(
    *,
    preregistration: dict[str, Any],
    pyscf_python: Path,
    pyscf_site_packages: Path,
    qm_checkpoint: Path,
    qm_ledger: Path,
) -> dict[str, str]:
    if not pyscf_python.is_file() or not pyscf_site_packages.is_dir():
        raise RuntimeError("The preregistered PySCF runtime is unavailable.")
    if not qm_checkpoint.is_file() or not qm_ledger.is_file():
        raise RuntimeError("The frozen QM checkpoint or ledger is unavailable.")
    runtime_identity = preregistration.get("runtime_identity")
    if not isinstance(runtime_identity, dict):
        raise RuntimeError("The V0-ARSP runtime identity is invalid.")
    actual = {
        "pyscf_python_sha256": _sha256(pyscf_python.resolve()),
        "pyscf_init_sha256": _sha256(pyscf_site_packages / "pyscf/__init__.py"),
        "qm_checkpoint_sha256": _sha256(qm_checkpoint),
        "qm_ledger_sha256": _sha256(qm_ledger),
    }
    if actual != runtime_identity:
        raise RuntimeError("The V0-ARSP external runtime or QM reference changed.")
    ledger = _load_json(qm_ledger, label="frozen QM ledger")
    system = ledger.get("system")
    numerics = ledger.get("numerics")
    if (
        not isinstance(system, dict)
        or not isinstance(numerics, dict)
        or (
            ledger.get("status") != "pass"
            or ledger.get("phase") != "gas"
            or system.get("compound_id") != "mobley_3867265"
            or system.get("molecule") != "acetone"
            or system.get("method") != EXPECTED_QM_REFERENCE["electronic_structure"]
            or system.get("basis") != EXPECTED_QM_REFERENCE["basis"]
            or system.get("reference") != EXPECTED_QM_REFERENCE["reference"]
            or system.get("density_fitting") != EXPECTED_QM_REFERENCE["density_fitting"]
            or numerics.get("grid_level")
            != EXPECTED_QM_REFERENCE["semilocal_grid_level"]
            or numerics.get("nlc_grid_profile")
            != EXPECTED_QM_REFERENCE["nonlocal_grid_profile"]
            or numerics.get("scf_energy_tolerance_hartree")
            != EXPECTED_QM_REFERENCE["scf_energy_tolerance_hartree"]
            or numerics.get("scf_gradient_tolerance")
            != EXPECTED_QM_REFERENCE["scf_gradient_tolerance"]
            or numerics.get("maximum_scf_cycles")
            != EXPECTED_QM_REFERENCE["maximum_scf_cycles"]
        )
    ):
        raise RuntimeError(
            "The frozen QM ledger does not match its preregistered identity."
        )
    return actual


def _load_frozen_qm_zero_field() -> dict[str, Any]:
    payload = _load_json(REPO_ROOT / RAW_QM_RELATIVE_PATH, label="frozen QM zero field")
    zero_field = payload.get("zero_field")
    if not isinstance(zero_field, dict) or payload.get("status") != "pass":
        raise RuntimeError("The frozen QM zero-field record is invalid.")
    energy = float(zero_field.get("energy_hartree"))
    dipole = np.asarray(zero_field.get("dipole_e_bohr"), dtype=float)
    if (
        not np.isfinite(energy)
        or dipole.shape != (3,)
        or not np.all(np.isfinite(dipole))
    ):
        raise RuntimeError("The frozen QM zero-field record is incomplete.")
    return {"energy_hartree": energy, "dipole_e_bohr": dipole}


def _run_helper(
    *,
    work_dir: Path,
    pyscf_python: Path,
    pyscf_site_packages: Path,
    qm_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    helper_output = (
        work_dir / "route2-v0-atomic-response-stationary-static-mep-helper.json"
    )
    if helper_output.exists():
        raise FileExistsError(helper_output)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(REPO_ROOT), str(pyscf_site_packages), existing_pythonpath)
        if value
    )
    command = [
        str(pyscf_python),
        str(REPO_ROOT / STATIC_HELPER_RELATIVE_PATH),
        "--atomic-table",
        str(REPO_ROOT / TABLE_RELATIVE_PATH),
        "--atomic-manifest",
        str(REPO_ROOT / TABLE_MANIFEST_RELATIVE_PATH),
        "--mace-atomic-map",
        str(REPO_ROOT / ATOMIC_MAP_RELATIVE_PATH),
        "--mace-response",
        str(REPO_ROOT / RESPONSE_RELATIVE_PATH),
        "--points",
        str(REPO_ROOT / POINTS_RELATIVE_PATH),
        "--qm-checkpoint",
        str(qm_checkpoint),
        "--output",
        str(helper_output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    artifact = _load_json(helper_output, label="V0-ARSP static-MEP helper output")
    receipt = {
        "argv": command,
        "stdout_sha256": _sha256_text(completed.stdout),
        "stderr_sha256": _sha256_text(completed.stderr),
        "helper_output_sha256": _sha256(helper_output),
        "helper_output_path": str(helper_output),
    }
    return artifact, receipt


def _helper_numerical_checks(
    *,
    helper: dict[str, Any],
    zero_field: dict[str, Any],
    numerical_gates: dict[str, Any],
) -> dict[str, dict[str, float | bool]]:
    stationary = helper.get("stationary_reference")
    density = helper.get("permanent_density")
    checkpoint = helper.get("qm_checkpoint")
    if not all(isinstance(value, dict) for value in (stationary, density, checkpoint)):
        raise RuntimeError("The V0-ARSP helper output is incomplete.")
    density_grid = density.get("density_positivity_grid")
    if not isinstance(density_grid, dict):
        raise RuntimeError("The V0-ARSP helper output omits its density-source grid.")
    candidate_dipole = np.asarray(density.get("candidate_dipole_e_bohr"), dtype=float)
    qm_dipole = np.asarray(checkpoint.get("permanent_dipole_e_bohr"), dtype=float)
    if candidate_dipole.shape != (3,) or qm_dipole.shape != (3,):
        raise RuntimeError("The V0-ARSP helper output has invalid permanent dipoles.")
    return {
        "maximum_atomic_mo_metric": _upper_check(
            float(
                np.max(np.asarray(stationary["atomic_mo_metric_errors"], dtype=float))
            ),
            float(numerical_gates["maximum_atomic_mo_metric_max"]),
        ),
        "reference_electron_count_error": _upper_check(
            float(stationary["reference_electron_count_error_e"]),
            float(numerical_gates["reference_electron_count_error_max"]),
        ),
        "stationarity_residual_inf": _upper_check(
            float(stationary["stationarity_residual_inf_hartree"]),
            float(numerical_gates["stationarity_residual_inf_max"]),
        ),
        "support_constraint_residual_inf": _upper_check(
            float(stationary["support_constraint_residual_inf"]),
            float(numerical_gates["support_constraint_residual_inf_max"]),
        ),
        "stationary_energy_identity_error": _upper_check(
            float(stationary["stationary_energy_identity_error_hartree"]),
            float(numerical_gates["stationary_energy_identity_error_max"]),
        ),
        "permanent_density_electron_count_error": _upper_check(
            float(density["electron_count_error_e"]),
            float(numerical_gates["permanent_density_electron_count_error_max"]),
        ),
        "permanent_density_symmetry_error": _upper_check(
            float(density["density_symmetry_error"]),
            float(numerical_gates["permanent_density_symmetry_error_max"]),
        ),
        "permanent_density_grid_minimum_electron_number_density": _lower_check(
            float(density_grid["minimum_electron_number_density_e_per_bohr3"]),
            float(
                numerical_gates[
                    "permanent_density_grid_minimum_electron_number_density_min"
                ]
            ),
        ),
        "qm_checkpoint_geometry_max_abs_error_bohr": _upper_check(
            float(checkpoint["geometry_max_abs_error_bohr"]),
            float(numerical_gates["qm_checkpoint_geometry_max_abs_error_bohr_max"]),
        ),
        "qm_checkpoint_mo_metric": _upper_check(
            float(checkpoint["mo_metric_error"]),
            float(numerical_gates["qm_checkpoint_mo_metric_max"]),
        ),
        "qm_checkpoint_electron_count_error": _upper_check(
            abs(float(checkpoint["electron_count_e"]) - 48.0),
            float(numerical_gates["qm_checkpoint_electron_count_error_max"]),
        ),
        "qm_checkpoint_total_charge": _upper_check(
            abs(float(checkpoint["total_charge_e"])),
            float(numerical_gates["qm_checkpoint_total_charge_max"]),
        ),
        "qm_checkpoint_energy_compatibility": _upper_check(
            abs(
                float(checkpoint["energy_hartree"])
                - float(zero_field["energy_hartree"])
            ),
            float(numerical_gates["qm_checkpoint_energy_compatibility_max"]),
        ),
        "qm_checkpoint_dipole_compatibility": _upper_check(
            float(np.linalg.norm(qm_dipole - zero_field["dipole_e_bohr"])),
            float(numerical_gates["qm_checkpoint_dipole_compatibility_max"]),
        ),
    }


def _helper_scientific_checks(
    *,
    helper: dict[str, Any],
    scientific_gates: dict[str, Any],
) -> dict[str, dict[str, float | bool]]:
    comparison = helper.get("static_comparison")
    if not isinstance(comparison, dict):
        raise RuntimeError("The V0-ARSP helper output omits the static comparison.")
    return {
        registered_name: _upper_check(
            float(comparison[helper_name]),
            float(scientific_gates[f"{registered_name}_max"]),
        )
        for registered_name, helper_name in SCIENTIFIC_GATE_HELPER_FIELDS.items()
    }


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    work_dir = arguments.work_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    git_head = _require_clean_tracked_checkout()
    preregistration, input_hashes, preregistration_sha = _validate_preregistration()
    runtime_identity = _validate_external_qm_reference(
        preregistration=preregistration,
        pyscf_python=arguments.pyscf_python.resolve(),
        pyscf_site_packages=arguments.pyscf_site_packages.resolve(),
        qm_checkpoint=arguments.qm_checkpoint.resolve(),
        qm_ledger=arguments.qm_ledger.resolve(),
    )
    helper, helper_receipt = _run_helper(
        work_dir=work_dir,
        pyscf_python=arguments.pyscf_python.resolve(),
        pyscf_site_packages=arguments.pyscf_site_packages.resolve(),
        qm_checkpoint=arguments.qm_checkpoint.resolve(),
    )
    if (
        helper.get("artifact")
        != "route2-v0-atomic-response-stationary-source-acetone-static-mep-helper-v1"
        or helper.get("candidate", {}).get("construction")
        != "route2-v0-atomic-response-stationary-permanent-source-v1"
        or helper.get("system", {}).get("source_point_count") != 516
        or helper.get("input_sha256", {}).get("qm_checkpoint")
        != runtime_identity["qm_checkpoint_sha256"]
    ):
        raise RuntimeError(
            "The V0-ARSP helper output violated the frozen source contract."
        )
    expected_helper_inputs = {
        "atomic_table": input_hashes[TABLE_RELATIVE_PATH],
        "atomic_manifest": input_hashes[TABLE_MANIFEST_RELATIVE_PATH],
        "mace_atomic_map": input_hashes[ATOMIC_MAP_RELATIVE_PATH],
        "mace_response": input_hashes[RESPONSE_RELATIVE_PATH],
        "points": input_hashes[POINTS_RELATIVE_PATH],
        "qm_checkpoint": runtime_identity["qm_checkpoint_sha256"],
    }
    if helper.get("input_sha256") != expected_helper_inputs:
        raise RuntimeError("The V0-ARSP helper used unfrozen inputs.")
    if helper.get("runtime", {}).get("pyscf_version") != "2.13.1":
        raise RuntimeError("The V0-ARSP helper used an unregistered PySCF version.")
    zero_field = _load_frozen_qm_zero_field()
    numerical_checks = _helper_numerical_checks(
        helper=helper,
        zero_field=zero_field,
        numerical_gates=preregistration["numerical_gates"],
    )
    scientific_checks = _helper_scientific_checks(
        helper=helper,
        scientific_gates=preregistration["scientific_falsification_gates"],
    )
    numerical_pass = all(bool(check["passes"]) for check in numerical_checks.values())
    scientific_pass = all(bool(check["passes"]) for check in scientific_checks.values())
    passes_all = numerical_pass and scientific_pass
    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass" if passes_all else "reject",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": git_head,
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": preregistration_sha,
            "protocol_id": preregistration["protocol_id"],
        },
        "claim_boundary": preregistration["claim_boundary"],
        "hard_constraints": preregistration["hard_constraints"],
        "source_files_sha256": {
            relative: _sha256(REPO_ROOT / relative)
            for relative in SOURCE_RELATIVE_PATHS
        },
        "input_files_sha256": input_hashes,
        "runtime_identity": runtime_identity,
        "qm_zero_field_compatibility_reference": {
            "path": RAW_QM_RELATIVE_PATH,
            "sha256": input_hashes[RAW_QM_RELATIVE_PATH],
            "energy_hartree": zero_field["energy_hartree"],
            "dipole_e_bohr": zero_field["dipole_e_bohr"].tolist(),
        },
        "helper_receipt": helper_receipt,
        "candidate": helper["candidate"],
        "system": helper["system"],
        "stationary_reference": helper["stationary_reference"],
        "permanent_density": helper["permanent_density"],
        "qm_checkpoint": helper["qm_checkpoint"],
        "numerical_checks": numerical_checks,
        "scientific_falsification": {
            "checks": scientific_checks,
            "passes_all_registered_checks": passes_all,
            "verdict": (
                "admit-only-to-preregistered-broad-qm-physics-and-same-basis-kkt-gates"
                if passes_all
                else "reject-this-exact-atomic-response-stationary-permanent-source"
            ),
            "admission_boundary": (
                "A pass is only a frozen gas-phase permanent-source admission. "
                "It does not establish a continuum, cavity, force/PES, runtime, "
                "solvation energy, experimental accuracy, or the required broad "
                "functional-group QM physics certificate."
            ),
        },
        "runtime": {
            "python": sys.version,
            "boundary": (
                "The helper loads an existing QM checkpoint and makes no new QM SCF "
                "calculation. This source falsifier invokes no continuum/PCM, "
                "solvation-energy, force, experimental-label, or speed measurement."
            ),
        },
    }
    _write_exclusive_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
