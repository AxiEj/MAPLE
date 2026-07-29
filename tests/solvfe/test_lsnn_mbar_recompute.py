from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import pymbar  # noqa: F401
    from pymbar import MBAR

    _HAS_PYMBAR = True
except Exception:  # noqa: BLE001  # pragma: no cover - optional dependency
    MBAR = None
    _HAS_PYMBAR = False


def _artifact_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manual_reference_mbar(
    reduced: np.ndarray, selected_indices: list[np.ndarray], beta: float
) -> dict[str, Any]:
    selected_frames = []
    n_k = []
    for state_index, indices in enumerate(selected_indices):
        selected = reduced[state_index][:, indices]
        selected_frames.append(selected)
        n_k.append(int(selected.shape[1]))
    reduced_selected = np.concatenate(selected_frames, axis=1)

    estimator = MBAR(  # type: ignore[union-attr]
        reduced_selected,
        np.asarray(n_k, dtype=int),
        verbose=False,
        relative_tolerance=1e-10,
    )
    free_energy = estimator.compute_free_energy_differences()
    overlap = estimator.compute_overlap()
    overlap_matrix = np.asarray(overlap["matrix"], dtype=float)
    return {
        "delta_kcal_mol": float(free_energy["Delta_f"][0, -1] / (beta * 4.184)),
        "uncertainty_kcal_mol": float(free_energy["dDelta_f"][0, -1] / (beta * 4.184)),
        "n_k": [int(x) for x in n_k],
        "minimum_adjacent_overlap": float(
            min(overlap_matrix[0, 1], overlap_matrix[1, 0])
            if overlap_matrix.size
            else 1.0
        ),
    }


def _write_summary(
    path: Path,
    *,
    temperature_kelvin: float,
    lambda_states: np.ndarray,
    record: dict[str, Any],
) -> None:
    payload = {
        "protocol": {
            "temperature_kelvin": temperature_kelvin,
            "lambda_states": lambda_states.tolist(),
        },
        "records": [record],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_recompute(script: Path, summary: Path) -> tuple[int, dict[str, Any]]:
    result = subprocess.run(
        [sys.executable, str(script), "--summary", str(summary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip(), "expected json output"
    return result.returncode, json.loads(result.stdout)


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_matches_synthetic_artifact(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run"
    case_dir = work_dir / "cases" / "mobley_000001"
    case_dir.mkdir(parents=True)

    lambda_states = np.array([[0.0, 0.0], [0.5, 1.0]], dtype=float)
    reduced = np.array(
        [
            [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]],
            [[1.0, 1.5, 2.5, 3.5], [0.5, 1.5, 2.0, 2.5]],
        ],
        dtype=float,
    )
    selected = [
        np.array([0, 2], dtype=int),
        np.array([1, 3], dtype=int),
    ]
    temperature_kelvin = 300.0
    beta = 1.0 / (8.31446261815324e-3 * temperature_kelvin)

    np.savez(
        case_dir / "raw-u-kln-seed-20260729.npz",
        reduced_potentials=reduced,
        lambda_states=lambda_states,
        sampled_state_indices=np.arange(len(lambda_states), dtype=int),
        beta_mol_per_kj=np.array(beta, dtype=float),
        selected_frame_indices_state_0=selected[0],
        selected_frame_indices_state_1=selected[1],
    )
    artifact_path = "cases/mobley_000001/raw-u-kln-seed-20260729.npz"
    artifact_sha = _artifact_sha256(case_dir / "raw-u-kln-seed-20260729.npz")

    reference = _manual_reference_mbar(
        reduced=reduced, selected_indices=selected, beta=beta
    )
    _write_summary(
        work_dir / "summary.json",
        temperature_kelvin=temperature_kelvin,
        lambda_states=lambda_states,
        record={
            "compound_id": "mobley_000001",
            "name": "methane-like",
            "prediction_kcal_mol": float(reference["delta_kcal_mol"]),
            "seed_results": [
                {
                    "seed": 20260729,
                    "delta_g_kcal_mol": reference["delta_kcal_mol"],
                    "mbar_uncertainty_kcal_mol": reference["uncertainty_kcal_mol"],
                    "minimum_adjacent_overlap": reference["minimum_adjacent_overlap"],
                    "n_k": reference["n_k"],
                    "raw_energy_artifact": {
                        "path": artifact_path,
                        "sha256": artifact_sha,
                    },
                }
            ],
        },
    )

    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc == 0
    assert payload["verified"] is True
    assert payload["mismatch_count"] == 0


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_detects_delta_mismatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run2"
    case_dir = work_dir / "cases" / "mobley_000001"
    case_dir.mkdir(parents=True)

    reduced = np.array(
        [
            [[0.0, 1.0, 2.0], [1.0, 1.5, 2.5]],
            [[1.0, 0.5, 1.8], [0.5, 1.0, 2.0]],
        ],
        dtype=float,
    )
    selected = [
        np.array([0, 1], dtype=int),
        np.array([1, 2], dtype=int),
    ]
    temperature_kelvin = 298.0
    beta = 1.0 / (8.31446261815324e-3 * temperature_kelvin)

    np.savez(
        case_dir / "raw-u-kln-seed-20260729.npz",
        reduced_potentials=reduced,
        lambda_states=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        sampled_state_indices=np.arange(2, dtype=int),
        beta_mol_per_kj=np.array(beta, dtype=float),
        selected_frame_indices_state_0=selected[0],
        selected_frame_indices_state_1=selected[1],
    )
    artifact_path = "cases/mobley_000001/raw-u-kln-seed-20260729.npz"
    artifact_sha = _artifact_sha256(case_dir / "raw-u-kln-seed-20260729.npz")

    reference = _manual_reference_mbar(
        reduced=reduced, selected_indices=selected, beta=beta
    )
    lambda_states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    _write_summary(
        work_dir / "summary.json",
        temperature_kelvin=temperature_kelvin,
        lambda_states=lambda_states,
        record={
            "compound_id": "mobley_000001",
            "name": "benzene-like",
            "prediction_kcal_mol": float(reference["delta_kcal_mol"] + 1.0),
            "seed_results": [
                {
                    "seed": 20260729,
                    "delta_g_kcal_mol": float(reference["delta_kcal_mol"] + 1.0),
                    "mbar_uncertainty_kcal_mol": reference["uncertainty_kcal_mol"],
                    "minimum_adjacent_overlap": reference["minimum_adjacent_overlap"],
                    "n_k": reference["n_k"],
                    "raw_energy_artifact": {
                        "path": artifact_path,
                        "sha256": artifact_sha,
                    },
                }
            ],
        },
    )

    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc != 0
    assert payload["verified"] is False
    assert payload["records"][0]["seed_mismatch_count"] >= 1


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_detects_artifact_path_escape(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run3"
    case_dir = work_dir / "cases" / "mobley_000001"
    case_dir.mkdir(parents=True)

    reduced = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 0.5], [0.2, 0.8]],
        ],
        dtype=float,
    )
    selected = [np.array([0], dtype=int), np.array([1], dtype=int)]
    temperature_kelvin = 300.0
    beta = 1.0 / (8.31446261815324e-3 * temperature_kelvin)

    np.savez(
        case_dir / "raw-u-kln-seed-20260729.npz",
        reduced_potentials=reduced,
        lambda_states=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        sampled_state_indices=np.arange(2, dtype=int),
        beta_mol_per_kj=np.array(beta, dtype=float),
        selected_frame_indices_state_0=selected[0],
        selected_frame_indices_state_1=selected[1],
    )
    reference = _manual_reference_mbar(
        reduced=reduced, selected_indices=selected, beta=beta
    )
    _write_summary(
        work_dir / "summary.json",
        temperature_kelvin=temperature_kelvin,
        lambda_states=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        record={
            "compound_id": "mobley_000001",
            "name": "benzene-like",
            "prediction_kcal_mol": float(reference["delta_kcal_mol"]),
            "seed_results": [
                {
                    "seed": 20260729,
                    "delta_g_kcal_mol": reference["delta_kcal_mol"],
                    "mbar_uncertainty_kcal_mol": reference["uncertainty_kcal_mol"],
                    "minimum_adjacent_overlap": reference["minimum_adjacent_overlap"],
                    "n_k": reference["n_k"],
                    "raw_energy_artifact": {"path": "../outside/raw.npz"},
                }
            ],
        },
    )

    (tmp_path / "outside").mkdir()

    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc != 0
    assert payload["verified"] is False
    assert (
        "escapes work_dir" in payload["records"][0]["seed_recomputations"][0]["error"]
    )


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_detects_beta_mismatch_or_missing_and_checksum(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run4"
    case_dir = work_dir / "cases" / "mobley_000001"
    case_dir.mkdir(parents=True)

    reduced = np.array(
        [
            [[0.0, 1.0, 2.0], [1.0, 1.5, 2.5]],
            [[1.0, 0.5, 1.8], [0.5, 1.0, 2.0]],
        ],
        dtype=float,
    )
    selected = [np.array([0, 1], dtype=int), np.array([0, 2], dtype=int)]
    temperature_kelvin = 300.0
    artifact_path = "cases/mobley_000001/raw-u-kln-seed-20260729.npz"
    artifact = case_dir / "raw-u-kln-seed-20260729.npz"
    np.savez(
        artifact,
        reduced_potentials=reduced,
        lambda_states=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        sampled_state_indices=np.arange(2, dtype=int),
        beta_mol_per_kj=np.array(np.nan, dtype=float),
        selected_frame_indices_state_0=selected[0],
        selected_frame_indices_state_1=selected[1],
    )
    _write_summary(
        work_dir / "summary.json",
        temperature_kelvin=temperature_kelvin,
        lambda_states=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        record={
            "compound_id": "mobley_000001",
            "name": "benzene-like",
            "prediction_kcal_mol": 0.0,
            "seed_results": [
                {
                    "seed": 20260729,
                    "delta_g_kcal_mol": 0.0,
                    "mbar_uncertainty_kcal_mol": 0.0,
                    "minimum_adjacent_overlap": 0.0,
                    "n_k": [1, 1],
                    "raw_energy_artifact": {
                        "path": artifact_path,
                        "sha256": "0" * 64,
                    },
                }
            ],
        },
    )

    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc != 0
    assert payload["verified"] is False
    seed_error = payload["records"][0]["seed_recomputations"][0]["error"]
    assert ("invalid beta_mol_per_kj" in seed_error) or (
        "artifact sha256 mismatch" in seed_error
    )


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_fails_on_empty_records(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run5"
    work_dir.mkdir()
    payload = {
        "protocol": {
            "temperature_kelvin": 300.0,
            "lambda_states": [[0.0, 0.0], [1.0, 1.0]],
        },
        "records": [],
    }
    work_dir.joinpath("summary.json").write_text(json.dumps(payload), encoding="utf-8")
    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc != 0
    assert payload["verified"] is False
    assert payload["mismatch_count"] >= 1


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_fails_when_seed_results_empty(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run6"
    case_dir = work_dir / "cases" / "mobley_000001"
    case_dir.mkdir(parents=True)

    temperature_kelvin = 300.0
    reference_lambda_states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    _write_summary(
        work_dir / "summary.json",
        temperature_kelvin=temperature_kelvin,
        lambda_states=reference_lambda_states,
        record={
            "compound_id": "mobley_000001",
            "name": "benzene-like",
            "prediction_kcal_mol": 0.0,
            "seed_results": [],
        },
    )

    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc != 0
    assert payload["verified"] is False
    assert payload["records"][0]["seed_mismatch_count"] >= 1
    assert (
        payload["records"][0]["seed_recomputations"][0]["error"]
        == "seed_results is empty"
    )


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_fails_on_protocol_lambda_mismatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run7"
    case_dir = work_dir / "cases" / "mobley_000001"
    case_dir.mkdir(parents=True)

    reduced = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 0.5], [0.2, 0.8]],
        ],
        dtype=float,
    )
    selected = [np.array([0], dtype=int), np.array([1], dtype=int)]
    temperature_kelvin = 300.0
    beta = 1.0 / (8.31446261815324e-3 * temperature_kelvin)

    lambda_states = np.array([[0.0, 0.0], [0.5, 1.0]], dtype=float)
    np.savez(
        case_dir / "raw-u-kln-seed-20260729.npz",
        reduced_potentials=reduced,
        lambda_states=lambda_states,
        sampled_state_indices=np.arange(len(lambda_states), dtype=int),
        beta_mol_per_kj=np.array(beta, dtype=float),
        selected_frame_indices_state_0=selected[0],
        selected_frame_indices_state_1=selected[1],
    )
    reference = _manual_reference_mbar(
        reduced=reduced, selected_indices=selected, beta=beta
    )
    artifact_path = "cases/mobley_000001/raw-u-kln-seed-20260729.npz"
    artifact_sha = _artifact_sha256(case_dir / "raw-u-kln-seed-20260729.npz")
    protocol_lambda_states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    _write_summary(
        work_dir / "summary.json",
        temperature_kelvin=temperature_kelvin,
        lambda_states=protocol_lambda_states,
        record={
            "compound_id": "mobley_000001",
            "name": "benzene-like",
            "prediction_kcal_mol": float(reference["delta_kcal_mol"]),
            "seed_results": [
                {
                    "seed": 20260729,
                    "delta_g_kcal_mol": reference["delta_kcal_mol"],
                    "mbar_uncertainty_kcal_mol": reference["uncertainty_kcal_mol"],
                    "minimum_adjacent_overlap": reference["minimum_adjacent_overlap"],
                    "n_k": reference["n_k"],
                    "raw_energy_artifact": {
                        "path": artifact_path,
                        "sha256": artifact_sha,
                    },
                }
            ],
        },
    )

    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc != 0
    assert payload["verified"] is False
    assert (
        "lambda_states mismatch with protocol"
        in payload["records"][0]["seed_recomputations"][0]["error"]
    )


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_fails_on_seed_n_k_mismatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run8"
    case_dir = work_dir / "cases" / "mobley_000001"
    case_dir.mkdir(parents=True)

    reduced = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 0.5], [0.2, 0.8]],
        ],
        dtype=float,
    )
    selected = [np.array([0], dtype=int), np.array([1], dtype=int)]
    lambda_states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    temperature_kelvin = 300.0
    beta = 1.0 / (8.31446261815324e-3 * temperature_kelvin)

    np.savez(
        case_dir / "raw-u-kln-seed-20260729.npz",
        reduced_potentials=reduced,
        lambda_states=lambda_states,
        sampled_state_indices=np.arange(2, dtype=int),
        beta_mol_per_kj=np.array(beta, dtype=float),
        selected_frame_indices_state_0=selected[0],
        selected_frame_indices_state_1=selected[1],
    )
    reference = _manual_reference_mbar(
        reduced=reduced, selected_indices=selected, beta=beta
    )
    artifact_path = "cases/mobley_000001/raw-u-kln-seed-20260729.npz"
    artifact_sha = _artifact_sha256(case_dir / "raw-u-kln-seed-20260729.npz")
    _write_summary(
        work_dir / "summary.json",
        temperature_kelvin=temperature_kelvin,
        lambda_states=lambda_states,
        record={
            "compound_id": "mobley_000001",
            "name": "benzene-like",
            "prediction_kcal_mol": float(reference["delta_kcal_mol"]),
            "seed_results": [
                {
                    "seed": 20260729,
                    "delta_g_kcal_mol": reference["delta_kcal_mol"],
                    "mbar_uncertainty_kcal_mol": reference["uncertainty_kcal_mol"],
                    "minimum_adjacent_overlap": reference["minimum_adjacent_overlap"],
                    "n_k": [1, 2],
                    "raw_energy_artifact": {
                        "path": artifact_path,
                        "sha256": artifact_sha,
                    },
                }
            ],
        },
    )

    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc != 0
    assert payload["verified"] is False
    assert (
        payload["records"][0]["seed_recomputations"][0]["difference"]["n_k_abs"]
        == "mismatch"
    )


@pytest.mark.skipif(not _HAS_PYMBAR, reason="pymbar is required")
def test_recompute_fails_on_potential_reduced_inconsistency(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "docs/implicit-solvation/benchmarks/lsnn_pilot/recompute_mbar.py"
    )

    work_dir = tmp_path / "run9"
    case_dir = work_dir / "cases" / "mobley_000001"
    case_dir.mkdir(parents=True)

    reduced = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 0.5], [0.2, 0.8]],
        ],
        dtype=float,
    )
    selected = [np.array([0], dtype=int), np.array([1], dtype=int)]
    lambda_states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    temperature_kelvin = 300.0
    beta = 1.0 / (8.31446261815324e-3 * temperature_kelvin)

    np.savez(
        case_dir / "raw-u-kln-seed-20260729.npz",
        reduced_potentials=reduced,
        potential_energies=np.zeros_like(reduced),
        lambda_states=lambda_states,
        sampled_state_indices=np.arange(2, dtype=int),
        beta_mol_per_kj=np.array(beta, dtype=float),
        selected_frame_indices_state_0=selected[0],
        selected_frame_indices_state_1=selected[1],
    )
    reference = _manual_reference_mbar(
        reduced=reduced, selected_indices=selected, beta=beta
    )
    artifact_path = "cases/mobley_000001/raw-u-kln-seed-20260729.npz"
    artifact_sha = _artifact_sha256(case_dir / "raw-u-kln-seed-20260729.npz")
    _write_summary(
        work_dir / "summary.json",
        temperature_kelvin=temperature_kelvin,
        lambda_states=lambda_states,
        record={
            "compound_id": "mobley_000001",
            "name": "benzene-like",
            "prediction_kcal_mol": float(reference["delta_kcal_mol"]),
            "seed_results": [
                {
                    "seed": 20260729,
                    "delta_g_kcal_mol": reference["delta_kcal_mol"],
                    "mbar_uncertainty_kcal_mol": reference["uncertainty_kcal_mol"],
                    "minimum_adjacent_overlap": reference["minimum_adjacent_overlap"],
                    "n_k": reference["n_k"],
                    "raw_energy_artifact": {
                        "path": artifact_path,
                        "sha256": artifact_sha,
                    },
                }
            ],
        },
    )

    rc, payload = _run_recompute(script, work_dir / "summary.json")
    assert rc != 0
    assert payload["verified"] is False
    assert (
        "not beta*potential energies"
        in payload["records"][0]["seed_recomputations"][0]["error"]
    )
