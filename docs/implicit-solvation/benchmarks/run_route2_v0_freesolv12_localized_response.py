#!/usr/bin/env python3
"""Compare current Route-2 learned response with broad nonuniform QM response.

The panel is the same label-free twelve-geometry/ten-functional-group panel as
the static-source oracle.  Four well-separated exterior point-charge modes are
selected from each geometry-only probe shell before either MACE or QM response
is evaluated.  The candidate is MACE-POLAR's exact zero-field density JVP in
the current local-jet interface; the reference is a two-step central finite-
field omegaB97M-V/def2-TZVPD induced MEP.  No continuum or experimental
solvation value is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
from ase.units import Bohr, Hartree


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for _path in (REPO_ROOT, SCRIPT_DIR):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)

from benchmark_core import canonical_json_bytes, sha256_file, write_json_atomic  # noqa: E402
import run_route2_v0_freesolv12_zero_field_static_mep as static_runner  # noqa: E402
from maple.function.calculator.extra_correction.implicit.gto_density import (  # noqa: E402
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_nonuniform_response import (  # noqa: E402
    localized_point_charge_atom_jets_hartree,
    molecular_dipole_response_from_density_coefficients,
    select_farthest_exterior_point_charge_modes,
    weighted_response_matrix_discrepancy,
)
from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import (  # noqa: E402
    build_route2_smd_exterior_probe_surface,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (  # noqa: E402
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import (  # noqa: E402
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
)


ARTIFACT = "route2-v0-freesolv12-mace-localized-response-v1"
MANIFEST_PATH = (
    SCRIPT_DIR / "route2-v0-freesolv12-mace-localized-response-prereg-v1.json"
)
QM_HELPER = SCRIPT_DIR / "route2_qm_localized_point_charge_response.py"
STATIC_PUBLIC_EXECUTION = SCRIPT_DIR / (
    "route2-v0-freesolv12-zero-field-mace-static-surface-mep-"
    "execution-edfae78e.json"
)
MODE_COUNT = 4
FIELD_STEPS_E = (3.0e-4, 1.0e-3)
SELECTED_STEP_E = FIELD_STEPS_E[0]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2-root", type=Path, required=True)
    parser.add_argument("--static-source-work-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--qm-python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--retry-failures", action="store_true")
    return parser.parse_args(argv)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_source() -> str:
    if _git("status", "--porcelain"):
        raise RuntimeError(
            "The localized-response oracle requires a clean committed checkout."
        )
    return _git("rev-parse", "HEAD")


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(right)), 1.0e-30)
    return float(np.linalg.norm(left - right) / denominator)


def _per_mode_relative(left: np.ndarray, right: np.ndarray) -> list[float]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Per-mode response arrays must have one matched mode axis.")
    values = []
    for index in range(left.shape[0]):
        values.append(_relative_frobenius(left[index], right[index]))
    return values


def _upper(value: float, maximum: float) -> dict[str, float | bool]:
    return {"maximum": maximum, "passes": bool(value <= maximum), "value": value}


def _require_private_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(f"{label} must be located below {private_root}.") from exc
    return resolved


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "claim_boundary",
        "decision_rule",
        "execution_contract",
        "finite_field_protocol",
        "hard_constraints",
        "locked_record_identity_sha256",
        "locked_records",
        "mode_protocol",
        "numerical_gates",
        "parent_static_source",
        "protocol_id",
        "qm_method",
        "schema_version",
        "scientific_falsification_gates",
        "source_candidate",
        "status",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Localized-response manifest does not have the exact v1 schema.")
    if (
        payload["schema_version"] != 1
        or payload["protocol_id"]
        != "route2-v0-freesolv12-mace-localized-response-prereg-v1"
        or payload["status"] != "frozen-before-execution"
    ):
        raise ValueError("Localized-response manifest identity is invalid.")
    static_manifest = static_runner.load_static_mep_manifest(
        static_runner.MANIFEST_PATH
    )
    if payload["locked_records"] != static_manifest["locked_records"]:
        raise ValueError("Localized-response records differ from the no-label source panel.")
    if payload["locked_record_identity_sha256"] != _canonical_hash(
        payload["locked_records"]
    ):
        raise ValueError("Localized-response record identity digest is invalid.")
    if payload["finite_field_protocol"] != {
        "field_steps_e": list(FIELD_STEPS_E),
        "selected_reporting_step_e": SELECTED_STEP_E,
        "signs": [-1, 1],
    }:
        raise ValueError("Localized-response finite-field protocol changed.")
    if payload["mode_protocol"].get("mode_count") != MODE_COUNT:
        raise ValueError("Localized-response mode count changed.")
    contract = payload["execution_contract"]
    for relative, expected in contract["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"Localized-response source changed after freeze: {relative}")
    for relative, expected in contract["input_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"Localized-response input changed after freeze: {relative}")
    return payload


def _static_record_files(
    static_work_dir: Path,
    record: dict[str, Any],
) -> tuple[Path, Path, Path]:
    static_record_path = static_work_dir / "records" / f"{record['compound_id']}.json"
    static_record = json.loads(static_record_path.read_text(encoding="utf-8"))
    if static_record.get("status") != "success":
        raise RuntimeError(f"Static source record is not successful: {record['compound_id']}")
    expected_gas_json = static_record["qm_reference"]["gas_checkpoint_json_sha256"]
    expected_surface = static_record["surface"]["surface_points_sha256"]
    candidates = sorted(
        (static_work_dir / "records" / str(record["compound_id"])).glob(
            "qm-attempt-*/gas/gas.json"
        )
    )
    for gas_json in candidates:
        if sha256_file(gas_json) != expected_gas_json:
            continue
        checkpoint = gas_json.with_name("gas.chk")
        surface = gas_json.parent.parent / "surface.npz"
        if not checkpoint.is_file() or not surface.is_file():
            continue
        with np.load(surface, allow_pickle=False) as state:
            points = np.asarray(state["surface_points_bohr"], dtype=float)
        if _sha256_array(points) != expected_surface:
            continue
        return checkpoint, gas_json, surface
    raise RuntimeError(f"Cannot bind the frozen QM checkpoint for {record['compound_id']}.")


def _establish_or_validate_lock(work_dir: Path, lock: dict[str, Any]) -> None:
    path = work_dir / "run-lock.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != lock:
            raise RuntimeError("Existing localized-response work directory has another lock.")
        return
    write_json_atomic(path, lock)


def _record_needs_run(path: Path, *, retry_failures: bool) -> bool:
    if not path.exists():
        return True
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    if status == "success":
        return False
    if status == "failure":
        return retry_failures
    raise RuntimeError(f"Unexpected localized-response record status: {status!r}.")


def _attempt_directory(record_dir: Path) -> Path:
    for index in range(1, 1000):
        candidate = record_dir / f"attempt-{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Exhausted localized-response attempt directory names.")


def _run_qm(
    *,
    qm_python: Path,
    checkpoint: Path,
    surface_path: Path,
    modes_path: Path,
    attempt: Path,
) -> tuple[Path, Path]:
    output_json = attempt / "qm-response.json"
    output_npz = attempt / "qm-response.npz"
    command = [
        str(qm_python),
        str(QM_HELPER),
        "--checkpoint",
        str(checkpoint),
        "--surface",
        str(surface_path),
        "--modes",
        str(modes_path),
        "--threads",
        "8",
        "--max-memory-mb",
        "8000",
        "--output-json",
        str(output_json),
        "--output-npz",
        str(output_npz),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    (attempt / "qm.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (attempt / "qm.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Localized QM response failed; see {attempt}.")
    return output_json, output_npz


def _mace_response(
    calculator,
    atoms,
    *,
    surface_points_bohr: np.ndarray,
    source_points_bohr: np.ndarray,
) -> dict[str, Any]:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    jets = localized_point_charge_atom_jets_hartree(
        positions,
        source_points_bohr,
    )
    zeros = np.zeros((len(atoms), 3), dtype=float)
    response = calculator.linearize_density_response(
        atoms,
        node_potential_ev=np.zeros(len(atoms), dtype=float),
        node_gradient_ev_per_angstrom=zeros,
    )
    jvp_density = []
    for jet in jets:
        field = np.concatenate(
            (
                (jet[:, 0] * Hartree)[:, None],
                jet[:, 1:] * Hartree / Bohr,
            ),
            axis=1,
        )
        jvp_density.append(response.jvp(field))
    density = np.asarray(jvp_density, dtype=float)
    mep = np.column_stack(
        [
            point_multipole_potential(surface_points_bohr, positions, direction)
            for direction in density
        ]
    )
    dipole_e_angstrom = molecular_dipole_response_from_density_coefficients(
        positions,
        density,
    )

    fd_density_by_step = []
    fd_reported_dipole_by_step = []
    for step in FIELD_STEPS_E:
        step_density = []
        step_dipole = []
        for jet in jets:
            potential_direction = jet[:, 0] * Hartree
            gradient_direction = jet[:, 1:] * Hartree / Bohr
            plus, _ = calculator.polar_state(
                atoms,
                node_potential_ev=step * potential_direction,
                node_gradient_ev_per_angstrom=step * gradient_direction,
            )
            minus, _ = calculator.polar_state(
                atoms,
                node_potential_ev=-step * potential_direction,
                node_gradient_ev_per_angstrom=-step * gradient_direction,
            )
            step_density.append(
                (
                    np.asarray(plus.density_coefficients, dtype=float)
                    - np.asarray(minus.density_coefficients, dtype=float)
                )
                / (2.0 * step)
            )
            step_dipole.append(
                (
                    np.asarray(plus.dipole_e_angstrom, dtype=float)
                    - np.asarray(minus.dipole_e_angstrom, dtype=float)
                )
                / (2.0 * step)
            )
        fd_density_by_step.append(np.asarray(step_density, dtype=float))
        fd_reported_dipole_by_step.append(np.asarray(step_dipole, dtype=float))
    return {
        "atom_jets_hartree": jets,
        "density_jvp": density,
        "dipole_jvp_e_bohr": np.asarray(dipole_e_angstrom, dtype=float) / Bohr,
        "fd_density_by_step": np.asarray(fd_density_by_step, dtype=float),
        "fd_reported_dipole_by_step_e_bohr": (
            np.asarray(fd_reported_dipole_by_step, dtype=float) / Bohr
        ),
        "surface_mep_jvp": mep,
    }


def _evaluate_record(
    *,
    calculator,
    manifest: dict[str, Any],
    record: dict[str, Any],
    mol2_root: Path,
    static_work_dir: Path,
    qm_python: Path,
    record_dir: Path,
) -> dict[str, Any]:
    atoms = static_runner._atoms(record, mol2_root)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent=static_runner.WATER_SOLVENT,
        profile=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
    )
    surface = build_route2_smd_exterior_probe_surface(
        positions,
        radii,
        clearance_angstrom=static_runner.CLEARANCE_ANGSTROM,
    )
    modes = select_farthest_exterior_point_charge_modes(
        surface.surface_points_bohr,
        positions,
        mode_count=MODE_COUNT,
    )
    checkpoint, gas_json, static_surface = _static_record_files(
        static_work_dir,
        record,
    )
    with np.load(static_surface, allow_pickle=False) as state:
        reused_points = np.asarray(state["surface_points_bohr"], dtype=float)
    if not np.array_equal(reused_points, surface.surface_points_bohr):
        raise RuntimeError("Rebuilt response surface differs from the static oracle.")

    attempt = _attempt_directory(record_dir)
    attempt.mkdir(parents=True)
    modes_path = attempt / "localized-modes.npz"
    np.savez(
        modes_path,
        source_points_bohr=modes.source_points_bohr,
        source_surface_indices=modes.source_surface_indices,
    )
    qm_json_path, qm_npz_path = _run_qm(
        qm_python=qm_python,
        checkpoint=checkpoint,
        surface_path=static_surface,
        modes_path=modes_path,
        attempt=attempt,
    )
    qm_json = json.loads(qm_json_path.read_text(encoding="utf-8"))
    with np.load(qm_npz_path, allow_pickle=False) as state:
        steps = tuple(float(value) for value in state["field_steps_e"])
        qm_mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=float,
        )
        qm_dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"],
            dtype=float,
        )
    if steps != FIELD_STEPS_E or qm_mep.shape != (
        2,
        MODE_COUNT,
        surface.retained_point_count,
    ) or qm_dipole.shape != (2, MODE_COUNT, 3):
        raise RuntimeError("Localized QM response arrays violate the frozen shape.")

    mace = _mace_response(
        calculator,
        atoms,
        surface_points_bohr=surface.surface_points_bohr,
        source_points_bohr=modes.source_points_bohr,
    )
    selected_qm_mep = qm_mep[0].T
    selected_qm_dipole = qm_dipole[0]
    mep_metrics = weighted_response_matrix_discrepancy(
        mace["surface_mep_jvp"],
        selected_qm_mep,
        surface.quadrature_weights,
    )
    qm_step_mep = weighted_response_matrix_discrepancy(
        qm_mep[0].T,
        qm_mep[1].T,
        surface.quadrature_weights,
    )
    qm_dipole_step = _relative_frobenius(qm_dipole[0], qm_dipole[1])
    dipole_relative = _relative_frobenius(
        mace["dipole_jvp_e_bohr"],
        selected_qm_dipole,
    )
    dipole_by_mode = _per_mode_relative(
        mace["dipole_jvp_e_bohr"],
        selected_qm_dipole,
    )
    fd_density_errors = [
        _relative_frobenius(step_density, mace["density_jvp"])
        for step_density in mace["fd_density_by_step"]
    ]
    fd_dipole_closure = [
        _relative_frobenius(step_dipole, mace["dipole_jvp_e_bohr"])
        for step_dipole in mace["fd_reported_dipole_by_step_e_bohr"]
    ]
    charge_response_max = float(
        np.max(np.abs(np.sum(mace["density_jvp"][:, :, 0], axis=1)))
    )
    numerical_limits = manifest["numerical_gates"]
    numerical_checks = {
        "qm_mep_step_consistency_relative_frobenius": _upper(
            float(qm_step_mep["weighted_relative_frobenius"]),
            float(numerical_limits["qm_mep_step_consistency_relative_frobenius_max"]),
        ),
        "qm_dipole_step_consistency_relative_frobenius": _upper(
            qm_dipole_step,
            float(
                numerical_limits[
                    "qm_dipole_step_consistency_relative_frobenius_max"
                ]
            ),
        ),
        "mace_jvp_finite_difference_relative_frobenius_max": _upper(
            max(fd_density_errors),
            float(
                numerical_limits[
                    "mace_jvp_finite_difference_relative_frobenius_max"
                ]
            ),
        ),
        "mace_density_dipole_closure_relative_frobenius_max": _upper(
            max(fd_dipole_closure),
            float(
                numerical_limits[
                    "mace_density_dipole_closure_relative_frobenius_max"
                ]
            ),
        ),
        "mace_charge_response_absolute_max": _upper(
            charge_response_max,
            float(numerical_limits["mace_charge_response_absolute_max"]),
        ),
    }
    scientific_limits = manifest["scientific_falsification_gates"]
    scientific_checks = {
        "mep_response_relative_frobenius": _upper(
            float(mep_metrics["weighted_relative_frobenius"]),
            float(scientific_limits["mep_response_relative_frobenius_max"]),
        ),
        "mep_response_relative_mode_max": _upper(
            float(mep_metrics["maximum_weighted_relative_mode"]),
            float(scientific_limits["mep_response_relative_mode_max"]),
        ),
        "induced_dipole_response_relative_frobenius": _upper(
            dipole_relative,
            float(
                scientific_limits[
                    "induced_dipole_response_relative_frobenius_max"
                ]
            ),
        ),
    }
    numerical_pass = all(bool(check["passes"]) for check in numerical_checks.values())
    scientific_pass = all(bool(check["passes"]) for check in scientific_checks.values())
    return {
        "atom_count": len(atoms),
        "chemical_class": record["chemical_class"],
        "compound_id": record["compound_id"],
        "functional_group": record["functional_group"],
        "input_provenance": {
            "gas_checkpoint_json_sha256": sha256_file(gas_json),
            "gas_checkpoint_sha256": sha256_file(checkpoint),
            "localized_modes_npz_sha256": sha256_file(modes_path),
            "qm_response_json_sha256": sha256_file(qm_json_path),
            "qm_response_npz_sha256": sha256_file(qm_npz_path),
            "static_surface_npz_sha256": sha256_file(static_surface),
        },
        "mace_response": {
            "atom_jets_hartree_sha256": _sha256_array(mace["atom_jets_hartree"]),
            "density_jvp_sha256": _sha256_array(mace["density_jvp"]),
            "dipole_jvp_e_bohr": mace["dipole_jvp_e_bohr"].tolist(),
            "surface_mep_jvp_sha256": _sha256_array(mace["surface_mep_jvp"]),
        },
        "mep_response_metrics": mep_metrics,
        "mode_protocol": {
            "source_points_bohr": modes.source_points_bohr.tolist(),
            "source_points_bohr_sha256": _sha256_array(modes.source_points_bohr),
            "source_surface_indices": modes.source_surface_indices.tolist(),
        },
        "mol2_sha256": record["mol2_sha256"],
        "name": record["name"],
        "numerical_checks": numerical_checks,
        "passes_all_registered_checks": numerical_pass and scientific_pass,
        "qm_response": {
            "dipole_response_e_bohr_per_source_e": selected_qm_dipole.tolist(),
            "dipole_step_consistency_relative_frobenius": qm_dipole_step,
            "mep_response_sha256_by_step": [
                _sha256_array(qm_mep[index]) for index in range(len(FIELD_STEPS_E))
            ],
            "mep_step_consistency": qm_step_mep,
            "runtime_seconds": qm_json["runtime"]["total_elapsed_seconds"],
        },
        "scientific_checks": scientific_checks,
        "selected_dipole_response_relative_by_mode": dipole_by_mode,
        "surface": {
            "quadrature_weights_sha256": _sha256_array(surface.quadrature_weights),
            "retained_point_count": surface.retained_point_count,
            "surface_points_bohr_sha256": _sha256_array(surface.surface_points_bohr),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    execution_head = _require_clean_source()
    mol2_root = args.mol2_root.resolve()
    static_work_dir = _require_private_path(
        args.static_source_work_dir,
        label="Static-source work directory",
    )
    work_dir = _require_private_path(args.work_dir, label="Response work directory")
    qm_python = static_runner._executable_path_preserving_venv(args.qm_python)
    static_summary = static_work_dir / "summary.json"
    expected_static_sha = manifest["parent_static_source"]["private_summary_sha256"]
    if sha256_file(static_summary) != expected_static_sha:
        raise RuntimeError("Parent static-source result does not match the frozen digest.")
    static_payload = json.loads(static_summary.read_text(encoding="utf-8"))
    if static_payload.get("success_count") != 12 or static_payload.get("failure_count") != 0:
        raise RuntimeError("Parent static-source run is incomplete.")

    records = manifest["locked_records"]
    for record in records:
        static_runner._resolve_mol2_path(mol2_root, record)
        _static_record_files(static_work_dir, record)
    source_hashes = manifest["execution_contract"]["source_sha256"]
    lock = {
        "artifact": ARTIFACT,
        "execution_git_head": execution_head,
        "experimental_solvation_labels_read": False,
        "field_steps_e": list(FIELD_STEPS_E),
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "mode_count": MODE_COUNT,
        "parent_static_summary_sha256": sha256_file(static_summary),
        "qm_python": str(qm_python),
        "record_identity_sha256": manifest["locked_record_identity_sha256"],
        "source_files_sha256": source_hashes,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    _establish_or_validate_lock(work_dir, lock)
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
    if pending:
        calculator = static_runner._load_route2_float64_source_calculator(
            static_runner._atoms(pending[0], mol2_root),
            work_dir,
        )
    for ordinal, record in enumerate(records, start=1):
        record_path = records_dir / f"{record['compound_id']}.json"
        if not _record_needs_run(record_path, retry_failures=bool(args.retry_failures)):
            continue
        if calculator is None:
            raise AssertionError("A MACE calculator is required for pending records.")
        record_dir = records_dir / str(record["compound_id"])
        try:
            result = _evaluate_record(
                calculator=calculator,
                manifest=manifest,
                record=record,
                mol2_root=mol2_root,
                static_work_dir=static_work_dir,
                qm_python=qm_python,
                record_dir=record_dir,
            )
            result["status"] = "success"
            write_json_atomic(record_path, result)
            print(
                f"[{ordinal:02d}/12] {record['compound_id']} success "
                f"mep={result['mep_response_metrics']['weighted_relative_frobenius']:.6f} "
                f"dipole={result['scientific_checks']['induced_dipole_response_relative_frobenius']['value']:.6f}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - persisted per-record failure
            write_json_atomic(
                record_path,
                {
                    "compound_id": record["compound_id"],
                    "error_message": str(error),
                    "error_type": type(error).__name__,
                    "name": record["name"],
                    "status": "failure",
                },
            )
            print(f"[{ordinal:02d}/12] {record['compound_id']} failure: {error}", flush=True)

    completed = [
        json.loads((records_dir / f"{record['compound_id']}.json").read_text(encoding="utf-8"))
        for record in records
    ]
    successes = [row for row in completed if row.get("status") == "success"]
    failures = [row for row in completed if row.get("status") == "failure"]
    all_checks_pass = bool(successes) and len(successes) == len(records) and all(
        bool(row["passes_all_registered_checks"]) for row in successes
    )
    summary = {
        "artifact": ARTIFACT,
        "claim_boundary": manifest["claim_boundary"],
        "disposition": {
            "all_records_pass_registered_response_gates": all_checks_pass,
            "continuum_or_solvation_energy_invoked": False,
            "experimental_solvation_labels_read": False,
            "learned_fixed_point_admitted_as_variational": False,
            "next_gate": (
                "Use this broad spatial-response result to admit or reject a "
                "separately scalar V0 response candidate; never repair a failed "
                "record with a fitted scale or experimental solvation residual."
            ),
        },
        "failure_count": len(failures),
        "failures": failures,
        "records": successes,
        "records_required": len(records),
        "run_lock": lock,
        "schema_version": 1,
        "status": (
            "complete-response-gate-pass"
            if all_checks_pass
            else "complete-response-gate-reject"
            if not failures
            else "incomplete-computational-failure"
        ),
        "success_count": len(successes),
    }
    write_json_atomic(work_dir / "summary.json", summary)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
