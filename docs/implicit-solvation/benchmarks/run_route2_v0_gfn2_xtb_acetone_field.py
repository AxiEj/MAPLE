#!/usr/bin/env python3
"""Screen frozen gas-phase GFN2-xTB response without a solvent model.

The only perturbation is an asymptotically uniform pair of external point
charges.  It uses the official xTB embedding interface and the same frozen
acetone geometry/QM finite-field reference already used to reject V0-Q.  This
is deliberately a physical pre-screen, not an implicit-solvation benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_ID = "route2-v0-gfn2-xtb-acetone-field-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-gfn2-xtb-acetone-field-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_v0_gfn2_xtb_acetone_field.py"
)
QM_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-qeq-acetone-qm-field-v1.json"
)
ARCHIVE_MANIFEST_RELATIVE_PATH = (
    ".omx/benchmarks/"
    "route2-exact-gto-fixed-geometry-canary-v1-408ca3f-20260727/"
    "maple.out.implicit/manifest.json"
)
DEFAULT_XTB = Path("/home/axie/xtb/xtb-dist/bin/xtb")
BOHR_PER_ANGSTROM = 1.8897261254578281
DIRECTIONS = ("x", "y", "z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--xtb", type=Path, default=DEFAULT_XTB)
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
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The GFN2-xTB physical screen requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    _git("ls-files", "--error-unmatch", RUNNER_RELATIVE_PATH)
    _git("ls-files", "--error-unmatch", PREREG_RELATIVE_PATH)
    return _git("rev-parse", "HEAD")


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite numeric value.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _validate_preregistration(
    xtb_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    preregistration_path = REPO_ROOT / PREREG_RELATIVE_PATH
    qm_path = REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH
    manifest_path = REPO_ROOT / ARCHIVE_MANIFEST_RELATIVE_PATH
    preregistration = _load_json(preregistration_path, label="GFN2-xTB preregistration")
    qm_artifact = _load_json(qm_path, label="frozen acetone QM artifact")
    manifest = _load_json(manifest_path, label="frozen acetone manifest")

    if (
        preregistration.get("protocol_id")
        != "route2-v0-gfn2-xtb-acetone-field-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The GFN2-xTB protocol is not the frozen preregistration.")
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise TypeError("The GFN2-xTB preregistration omits its execution contract.")
    source_hashes = contract.get("source_sha256")
    if source_hashes != {
        RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
    }:
        raise RuntimeError("The GFN2-xTB runner changed after preregistration.")
    input_hashes = contract.get("input_sha256")
    expected_input_hashes = {
        QM_ARTIFACT_RELATIVE_PATH: _sha256(qm_path),
        ARCHIVE_MANIFEST_RELATIVE_PATH: _sha256(manifest_path),
    }
    if input_hashes != expected_input_hashes:
        raise RuntimeError("A frozen GFN2-xTB input changed after preregistration.")

    if xtb_path.resolve() != DEFAULT_XTB.resolve() or not xtb_path.is_file():
        raise RuntimeError(
            "The GFN2-xTB screen forbids substituting the xTB executable."
        )
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError("The GFN2-xTB preregistration omits runtime identity.")
    version_output = subprocess.check_output([str(xtb_path), "--version"], text=True)
    if (
        _sha256(xtb_path) != runtime.get("xtb_binary_sha256")
        or "xtb version 6.7.1 (edcfbbe)" not in version_output
        or runtime.get("xtb_version") != "6.7.1 (edcfbbe)"
        or runtime.get("omp_num_threads") != 1
        or runtime.get("parallel_processes") != 1
    ):
        raise RuntimeError("The xTB runtime does not match the preregistration.")

    if qm_artifact.get("artifact") != "route2-v0-qeq-acetone-qm-field-v1":
        raise RuntimeError("The frozen QM artifact identity is invalid.")
    if qm_artifact.get("status") != "pass":
        raise RuntimeError("The frozen QM reference did not pass its numerical gates.")
    return preregistration, qm_artifact, manifest, _sha256(preregistration_path)


def _frozen_geometry(
    qm_artifact: dict[str, Any], manifest: dict[str, Any]
) -> tuple[tuple[str, ...], np.ndarray]:
    system = qm_artifact.get("system")
    if not isinstance(system, dict):
        raise TypeError("The frozen QM artifact has no system identity.")
    symbols = tuple(system.get("atom_symbols", ()))
    positions = np.asarray(system.get("positions_angstrom"), dtype=float)
    if symbols != ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H"):
        raise RuntimeError("The frozen system is not the declared acetone geometry.")
    if positions.shape != (10, 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("The frozen acetone coordinates are invalid.")
    manifest_symbols = tuple(manifest.get("elements", ()))
    manifest_positions = np.asarray(manifest.get("positions_angstrom"), dtype=float)
    if manifest_symbols != symbols or not np.array_equal(manifest_positions, positions):
        raise RuntimeError("The frozen QM and MACE-manifest geometries disagree.")
    return symbols, positions


def _write_xyz(path: Path, symbols: tuple[str, ...], positions: np.ndarray) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write("frozen acetone geometry for GFN2-xTB field screen\n")
        for symbol, position in zip(symbols, positions, strict=True):
            handle.write(
                f"{symbol} {position[0]:.16g} {position[1]:.16g} {position[2]:.16g}\n"
            )


def _write_field_pair(
    directory: Path,
    *,
    center_bohr: np.ndarray,
    direction: int,
    sign: int,
    field_step_au: float,
    distance_bohr: float,
    gamma: float,
) -> float:
    """Create ±q charges whose real-point-charge field is sign*h*e_j at center."""
    magnitude_e = field_step_au * distance_bohr**2 / 2.0
    unit = np.zeros(3)
    unit[direction] = 1.0
    positive_charge_position = center_bohr - sign * distance_bohr * unit
    negative_charge_position = center_bohr + sign * distance_bohr * unit
    with (directory / "pcharge").open("x", encoding="utf-8") as handle:
        handle.write("2\n")
        handle.write(
            f"{magnitude_e:.16g} {positive_charge_position[0]:.16g} "
            f"{positive_charge_position[1]:.16g} "
            f"{positive_charge_position[2]:.16g} {gamma:.16g}\n"
        )
        handle.write(
            f"{-magnitude_e:.16g} {negative_charge_position[0]:.16g} "
            f"{negative_charge_position[1]:.16g} "
            f"{negative_charge_position[2]:.16g} {gamma:.16g}\n"
        )
    (directory / "xcontrol").write_text(
        "$embedding\n  input=pcharge\n  gradient=pcgrad\n$end\n",
        encoding="utf-8",
    )
    return magnitude_e


def _run_xtb_case(
    *,
    xtb_path: Path,
    case_directory: Path,
    symbols: tuple[str, ...],
    positions: np.ndarray,
    center_bohr: np.ndarray,
    direction: int | None,
    sign: int | None,
    field_step_au: float | None,
    distance_bohr: float,
    gamma: float,
) -> dict[str, Any]:
    case_directory.mkdir(parents=True, exist_ok=False)
    xyz_path = case_directory / "acetone.xyz"
    _write_xyz(xyz_path, symbols, positions)
    command = [
        str(xtb_path),
        xyz_path.name,
        "--gfn",
        "2",
        "--parallel",
        "1",
        "--json",
    ]
    pair_charge_e: float | None = None
    if direction is not None:
        if sign not in (-1, 1) or field_step_au is None:
            raise RuntimeError("A nonzero field case needs sign and field step.")
        pair_charge_e = _write_field_pair(
            case_directory,
            center_bohr=center_bohr,
            direction=direction,
            sign=sign,
            field_step_au=field_step_au,
            distance_bohr=distance_bohr,
            gamma=gamma,
        )
        command.extend(("--input", "xcontrol"))
    elif sign is not None or field_step_au is not None:
        raise RuntimeError("The zero field case must not create external charges.")

    stdout_path = case_directory / "xtb.stdout"
    stderr_path = case_directory / "xtb.stderr"
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    started = time.perf_counter()
    with (
        stdout_path.open("x", encoding="utf-8") as stdout_handle,
        stderr_path.open("x", encoding="utf-8") as stderr_handle,
    ):
        subprocess.run(
            command,
            cwd=case_directory,
            check=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=environment,
        )
    elapsed = time.perf_counter() - started
    payload = _load_json(case_directory / "xtbout.json", label="xTB case result")
    dipole = np.asarray(payload.get("dipole / a.u."), dtype=float)
    energy = _number(payload.get("total energy"), label="xTB total energy")
    if (
        payload.get("method") != "GFN2-xTB"
        or payload.get("xtb version") != "6.7.1 (edcfbbe)"
        or dipole.shape != (3,)
        or not np.all(np.isfinite(dipole))
        or not np.isfinite(energy)
    ):
        raise RuntimeError(
            "The GFN2-xTB output identity or electronic response is invalid."
        )
    return {
        "direction": None if direction is None else DIRECTIONS[direction],
        "field_sign": sign,
        "field_step_au": field_step_au,
        "pair_charge_magnitude_e": pair_charge_e,
        "total_energy_hartree": energy,
        "dipole_e_bohr": dipole.tolist(),
        "elapsed_seconds": elapsed,
        "command": command,
        "xcontrol_sha256": None
        if direction is None
        else _sha256(case_directory / "xcontrol"),
        "pcharge_sha256": None
        if direction is None
        else _sha256(case_directory / "pcharge"),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
        "json_sha256": _sha256(case_directory / "xtbout.json"),
    }


def _case(
    cases: list[dict[str, Any]], *, step: float, direction: int, sign: int
) -> dict[str, Any]:
    for case in cases:
        if (
            case["field_step_au"] == step
            and case["direction"] == DIRECTIONS[direction]
            and case["field_sign"] == sign
        ):
            return case
    raise RuntimeError("A preregistered xTB finite-field case is missing.")


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    xtb_path = args.xtb.resolve()
    if output.exists() or work_dir.exists():
        raise FileExistsError(output if output.exists() else work_dir)
    git_head = _require_clean_source()
    preregistration, qm_artifact, manifest, preregistration_sha = (
        _validate_preregistration(xtb_path)
    )
    symbols, positions = _frozen_geometry(qm_artifact, manifest)
    protocol = preregistration.get("finite_field_protocol")
    numerical_gates = preregistration.get("numerical_gates")
    scientific_gates = preregistration.get("scientific_falsification_gates")
    if (
        not isinstance(protocol, dict)
        or not isinstance(numerical_gates, dict)
        or not isinstance(scientific_gates, dict)
    ):
        raise TypeError("The GFN2-xTB preregistration is incomplete.")
    raw_steps = protocol.get("field_steps_au")
    if not isinstance(raw_steps, list):
        raise TypeError("The field steps must be a JSON list.")
    steps = tuple(_number(value, label="field step") for value in raw_steps)
    if steps != (3.0e-4, 1.0e-3):
        raise RuntimeError("The field steps differ from the frozen protocol.")
    distance_bohr = _number(protocol.get("pair_distance_bohr"), label="pair distance")
    gamma = _number(protocol.get("point_charge_gamma"), label="point-charge gamma")
    if distance_bohr != 200.0 or gamma != 1.0e8:
        raise RuntimeError("The external point-charge interface changed after freeze.")

    center_bohr = positions.mean(axis=0) * BOHR_PER_ANGSTROM
    position_bohr = positions * BOHR_PER_ANGSTROM
    radius_bohr = float(np.max(np.linalg.norm(position_bohr - center_bohr, axis=1)))
    inhomogeneity_proxy = (radius_bohr / distance_bohr) ** 2
    if inhomogeneity_proxy > float(numerical_gates["pair_inhomogeneity_proxy_max"]):
        raise RuntimeError(
            "The fixed pair field is not sufficiently uniform geometrically."
        )

    work_dir.mkdir(parents=True, exist_ok=False)
    total_started = time.perf_counter()
    zero = _run_xtb_case(
        xtb_path=xtb_path,
        case_directory=work_dir / "zero",
        symbols=symbols,
        positions=positions,
        center_bohr=center_bohr,
        direction=None,
        sign=None,
        field_step_au=None,
        distance_bohr=distance_bohr,
        gamma=gamma,
    )
    cases: list[dict[str, Any]] = []
    for step in steps:
        for direction in range(3):
            for sign in (-1, 1):
                cases.append(
                    _run_xtb_case(
                        xtb_path=xtb_path,
                        case_directory=(
                            work_dir
                            / f"field-{step:.1e}-{DIRECTIONS[direction]}-{sign:+d}"
                        ),
                        symbols=symbols,
                        positions=positions,
                        center_bohr=center_bohr,
                        direction=direction,
                        sign=sign,
                        field_step_au=step,
                        distance_bohr=distance_bohr,
                        gamma=gamma,
                    )
                )

    tensors: dict[float, np.ndarray] = {}
    energy_diagonals: dict[float, np.ndarray] = {}
    by_step: dict[str, dict[str, Any]] = {}
    zero_energy = float(zero["total_energy_hartree"])
    for step in steps:
        tensor = np.empty((3, 3), dtype=float)
        energy_diagonal = np.empty(3, dtype=float)
        for direction in range(3):
            negative = _case(cases, step=step, direction=direction, sign=-1)
            positive = _case(cases, step=step, direction=direction, sign=1)
            dipole_negative = np.asarray(negative["dipole_e_bohr"], dtype=float)
            dipole_positive = np.asarray(positive["dipole_e_bohr"], dtype=float)
            tensor[:, direction] = (dipole_positive - dipole_negative) / (2.0 * step)
            energy_diagonal[direction] = (
                -(
                    float(positive["total_energy_hartree"])
                    + float(negative["total_energy_hartree"])
                    - 2.0 * zero_energy
                )
                / step**2
            )
        symmetric = 0.5 * (tensor + tensor.T)
        antisymmetry = float(
            np.linalg.norm(0.5 * (tensor - tensor.T))
            / max(np.linalg.norm(symmetric), 1.0e-30)
        )
        energy_dipole = float(
            np.max(
                np.abs(energy_diagonal - np.diag(symmetric))
                / np.maximum(np.abs(np.diag(symmetric)), 1.0e-30)
            )
        )
        tensors[step] = symmetric
        energy_diagonals[step] = energy_diagonal
        by_step[f"{step:.1e}"] = {
            "dipole_derivative_tensor_bohr3": tensor.tolist(),
            "symmetric_tensor_bohr3": symmetric.tolist(),
            "energy_second_derivative_diagonal_bohr3": energy_diagonal.tolist(),
            "antisymmetry_relative_frobenius": antisymmetry,
            "energy_dipole_diagonal_relative_max": energy_dipole,
            "symmetric_eigenvalues_bohr3": np.linalg.eigvalsh(symmetric).tolist(),
        }

    selected_step = min(steps)
    selected = tensors[selected_step]
    selected_metrics = by_step[f"{selected_step:.1e}"]
    step_consistency = _relative_frobenius(tensors[min(steps)], tensors[max(steps)])
    numerical_checks = {
        "response_antisymmetry": {
            "value": selected_metrics["antisymmetry_relative_frobenius"],
            "maximum": numerical_gates["response_antisymmetry_relative_frobenius_max"],
        },
        "energy_dipole_diagonal": {
            "value": selected_metrics["energy_dipole_diagonal_relative_max"],
            "maximum": numerical_gates["energy_dipole_diagonal_relative_max"],
        },
        "field_step_consistency": {
            "value": step_consistency,
            "maximum": numerical_gates["field_step_consistency_relative_frobenius_max"],
        },
        "minimum_polarizability_eigenvalue": {
            "value": float(np.min(np.linalg.eigvalsh(selected))),
            "minimum": numerical_gates["minimum_polarizability_eigenvalue_bohr3_min"],
        },
    }
    numerical_checks["response_antisymmetry"]["passes"] = (
        numerical_checks["response_antisymmetry"]["value"]
        <= numerical_checks["response_antisymmetry"]["maximum"]
    )
    numerical_checks["energy_dipole_diagonal"]["passes"] = (
        numerical_checks["energy_dipole_diagonal"]["value"]
        <= numerical_checks["energy_dipole_diagonal"]["maximum"]
    )
    numerical_checks["field_step_consistency"]["passes"] = (
        numerical_checks["field_step_consistency"]["value"]
        <= numerical_checks["field_step_consistency"]["maximum"]
    )
    numerical_checks["minimum_polarizability_eigenvalue"]["passes"] = (
        numerical_checks["minimum_polarizability_eigenvalue"]["value"]
        >= numerical_checks["minimum_polarizability_eigenvalue"]["minimum"]
    )
    numerical_pass = all(check["passes"] for check in numerical_checks.values())

    qm_reference = qm_artifact.get("qm_reference")
    if not isinstance(qm_reference, dict):
        raise TypeError("The frozen QM artifact has no polarizability reference.")
    qm_polarizability = np.asarray(
        qm_reference.get("selected_symmetric_polarizability_bohr3"), dtype=float
    )
    if qm_polarizability.shape != (3, 3) or not np.all(np.isfinite(qm_polarizability)):
        raise RuntimeError("The frozen QM polarizability is invalid.")
    qm_eigenvalues = np.linalg.eigvalsh(qm_polarizability)
    xtb_eigenvalues = np.linalg.eigvalsh(selected)
    scientific_checks = {
        "relative_frobenius_mismatch": {
            "value": _relative_frobenius(selected, qm_polarizability),
            "maximum": scientific_gates["relative_frobenius_mismatch_max"],
        },
        "trace_ratio": {
            "value": float(np.trace(selected) / np.trace(qm_polarizability)),
            "minimum": scientific_gates["trace_ratio_min"],
            "maximum": scientific_gates["trace_ratio_max"],
        },
        "principal_value_relative_max": {
            "value": float(
                np.max(
                    np.abs(xtb_eigenvalues - qm_eigenvalues)
                    / np.maximum(np.abs(qm_eigenvalues), 1.0e-30)
                )
            ),
            "maximum": scientific_gates["principal_value_relative_max"],
        },
    }
    scientific_checks["relative_frobenius_mismatch"]["passes"] = (
        scientific_checks["relative_frobenius_mismatch"]["value"]
        <= scientific_checks["relative_frobenius_mismatch"]["maximum"]
    )
    scientific_checks["trace_ratio"]["passes"] = (
        scientific_checks["trace_ratio"]["minimum"]
        <= scientific_checks["trace_ratio"]["value"]
        <= scientific_checks["trace_ratio"]["maximum"]
    )
    scientific_checks["principal_value_relative_max"]["passes"] = (
        scientific_checks["principal_value_relative_max"]["value"]
        <= scientific_checks["principal_value_relative_max"]["maximum"]
    )
    physical_pass = numerical_pass and all(
        check["passes"] for check in scientific_checks.values()
    )
    total_elapsed = time.perf_counter() - total_started
    qm_runtime = qm_artifact.get("runtime")
    qm_total_seconds = (
        None
        if not isinstance(qm_runtime, dict)
        else qm_runtime.get("total_elapsed_seconds")
    )

    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass" if numerical_pass else "invalid-numerics",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": git_head,
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": preregistration_sha,
            "protocol_id": preregistration["protocol_id"],
        },
        "source_files_sha256": {
            RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
        },
        "input_files_sha256": {
            QM_ARTIFACT_RELATIVE_PATH: _sha256(REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH),
            ARCHIVE_MANIFEST_RELATIVE_PATH: _sha256(
                REPO_ROOT / ARCHIVE_MANIFEST_RELATIVE_PATH
            ),
        },
        "claim_boundary": preregistration["claim_boundary"],
        "hard_constraints": preregistration["hard_constraints"],
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
            "center_bohr": center_bohr.tolist(),
            "maximum_radius_bohr": radius_bohr,
            "pair_inhomogeneity_proxy": inhomogeneity_proxy,
        },
        "external_field_interface": {
            "construction": protocol["construction"],
            "pair_distance_bohr": distance_bohr,
            "point_charge_gamma": gamma,
            "field_steps_au": list(steps),
            "directions": list(DIRECTIONS),
            "signs": [-1, 1],
            "pair_charge_magnitude_e_by_step": {
                f"{step:.1e}": step * distance_bohr**2 / 2.0 for step in steps
            },
        },
        "zero_field": zero,
        "field_cases": cases,
        "xtb_response": {
            "selected_step_au": selected_step,
            "polarizability_by_step": by_step,
            "selected_symmetric_polarizability_bohr3": selected.tolist(),
            "selected_eigenvalues_bohr3": xtb_eigenvalues.tolist(),
            "field_step_consistency_relative_frobenius": step_consistency,
        },
        "qm_reference": {
            "method": qm_reference.get("method"),
            "selected_step_au": qm_reference.get("selected_step_au"),
            "selected_symmetric_polarizability_bohr3": qm_polarizability.tolist(),
            "selected_eigenvalues_bohr3": qm_eigenvalues.tolist(),
            "source_artifact": QM_ARTIFACT_RELATIVE_PATH,
        },
        "numerical_checks": numerical_checks,
        "scientific_falsification": {
            "checks": scientific_checks,
            "passes_all_registered_checks": physical_pass,
            "verdict": (
                "provisional-no-runtime-qm-gas-response-admit"
                if physical_pass
                else (
                    "invalid-embedded-field-numerics"
                    if not numerical_pass
                    else "reject-gfn2-xtb-gas-response"
                )
            ),
            "decision_rule": preregistration["decision_rule"],
        },
        "runtime": {
            "main_python": platform.python_version(),
            "main_python_executable": str(Path(sys.executable).resolve()),
            "numpy": np.__version__,
            "xtb_path": str(xtb_path),
            "xtb_binary_sha256": _sha256(xtb_path),
            "xtb_version": "6.7.1 (edcfbbe)",
            "omp_num_threads": 1,
            "parallel_processes": 1,
            "screen_total_elapsed_seconds": total_elapsed,
            "frozen_qm_reference_total_elapsed_seconds": qm_total_seconds,
            "wall_time_comparison_boundary": (
                "The xTB screen and the archived QM reference used different "
                "processes and timings; this is a fixed-geometry feasibility "
                "observation, not a production end-to-end throughput claim."
            ),
        },
        "official_references": preregistration["official_references"],
    }
    _write_exclusive_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
