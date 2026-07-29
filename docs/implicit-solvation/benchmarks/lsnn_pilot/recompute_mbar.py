#!/usr/bin/env python3
"""Recompute MBAR outputs from saved raw u_kln artifacts.

The script reads a LSNN pilot ``summary.json`` and all referenced
``raw-u-kln-*.npz`` files, then reconstructs MBAR observables from saved
``reduced_potentials`` and ``selected_frame_indices_state_*``.

It does not use experimental labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pymbar import MBAR

R_KJ_PER_MOL_K = 0.008_314_462_618_153_24
KJ_PER_KCAL = 4.184


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_artifact_path(work_dir: Path, artifact_path: str) -> Path:
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError("raw_energy_artifact.path is required")

    normalized = Path(artifact_path)
    resolved = (work_dir / normalized).resolve()
    work_dir_resolved = work_dir.resolve()
    if resolved != work_dir_resolved and not str(resolved).startswith(
        str(work_dir_resolved) + "/"
    ):
        raise ValueError(f"artifact path escapes work_dir: {artifact_path}")
    return resolved


@dataclass(frozen=True)
class Tolerances:
    delta_kcal_mol: float = 1e-8
    uncertainty_kcal_mol: float = 1e-8
    overlap: float = 1e-8
    prediction_kcal_mol: float = 1e-8
    beta_mismatch: float = 1e-10


def _read_summary(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("summary.json must be an object")
    if "records" not in payload:
        raise ValueError("summary.json missing records")
    return payload


def _protocol_lambda_states(protocol: dict[str, Any]) -> np.ndarray:
    if "lambda_states" not in protocol:
        raise ValueError("summary.protocol.lambda_states is required")
    lambda_states = np.asarray(protocol["lambda_states"], dtype=float)
    if lambda_states.ndim != 2:
        raise ValueError("summary.protocol.lambda_states must be 2D")
    return lambda_states


def _load_npz(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    if "reduced_potentials" not in data:
        raise ValueError(f"missing reduced_potentials in {npz_path}")
    return {key: data[key] for key in data.files}


def _selected_indices(npz: dict[str, Any], n_states: int) -> list[np.ndarray]:
    selected: list[np.ndarray] = []
    for state_index in range(n_states):
        key = f"selected_frame_indices_state_{state_index}"
        if key not in npz:
            raise ValueError(f"missing {key}")
        indices = np.asarray(npz[key], dtype=int)
        if indices.ndim != 1:
            raise ValueError(f"{key} must be 1D")
        if indices.size == 0:
            raise ValueError(f"{key} is empty")
        if np.any(indices < 0):
            raise ValueError(f"{key} has negative entries")
        selected.append(indices)
    return selected


def _validate_artifact_metadata(
    artifact: dict[str, Any], artifact_path: Path, *, require_sha: bool = False
) -> None:
    if "sha256" not in artifact:
        if require_sha:
            raise ValueError("raw_energy_artifact.sha256 is required")
        return

    expected = str(artifact.get("sha256"))
    if len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
        raise ValueError("raw_energy_artifact.sha256 must be a 64-character hex digest")
    actual = _sha256(artifact_path)
    if actual != expected:
        raise ValueError(
            f"artifact sha256 mismatch: {artifact_path} expected {expected} got {actual}"
        )


def _compute_mbar(
    *, reduced: np.ndarray, selected_indices: list[np.ndarray], beta: float
) -> dict[str, Any]:
    selected_frames: list[np.ndarray] = []
    n_k: list[int] = []
    for state_index, indices in enumerate(selected_indices):
        if np.max(indices) >= reduced.shape[2]:
            raise ValueError(
                f"selected index out of frame range for state {state_index}"
            )
        selected = reduced[state_index][:, indices]
        selected_frames.append(selected)
        n_k.append(int(selected.shape[1]))
    reduced_selected = np.concatenate(selected_frames, axis=1)
    if np.any(np.asarray(n_k) <= 0):
        raise ValueError("MBAR count per state must be positive")

    estimator = MBAR(
        reduced_selected,
        np.asarray(n_k, dtype=int),
        verbose=False,
        relative_tolerance=1e-10,
    )
    free_energy = estimator.compute_free_energy_differences()
    overlap = estimator.compute_overlap()
    overlap_matrix = np.asarray(overlap["matrix"], dtype=float)
    adjacency = [
        min(overlap_matrix[i, i + 1], overlap_matrix[i + 1, i])
        for i in range(overlap_matrix.shape[0] - 1)
    ]

    return {
        "delta_kcal_mol": float(free_energy["Delta_f"][0, -1] / (beta * KJ_PER_KCAL)),
        "uncertainty_kcal_mol": float(
            free_energy["dDelta_f"][0, -1] / (beta * KJ_PER_KCAL)
        ),
        "n_k": [int(x) for x in n_k],
        "overlap_matrix": overlap_matrix.tolist(),
        "minimum_adjacent_overlap": float(min(adjacency)) if adjacency else 1.0,
    }


def _compare_lambda_state_protocol(
    *, artifact_lambda_states: np.ndarray, protocol_lambda_states: np.ndarray
) -> None:
    if artifact_lambda_states.shape != protocol_lambda_states.shape:
        raise ValueError("lambda_states shape mismatch with protocol")
    if not np.array_equal(artifact_lambda_states, protocol_lambda_states):
        raise ValueError("lambda_states mismatch with protocol")


def _validate_reduced_and_raw(
    *, reduced: np.ndarray, beta: float, raw: np.ndarray | None
) -> None:
    if not np.isfinite(reduced).all():
        raise ValueError("reduced_potentials contains non-finite values")
    if raw is None:
        return
    if raw.shape != reduced.shape:
        raise ValueError(
            "potential energy shape mismatch (required to match reduced_potentials)"
        )
    if not np.isfinite(raw).all():
        raise ValueError("potential energy contains non-finite values")
    if not np.allclose(raw * beta, reduced, rtol=0.0, atol=1e-10):
        raise ValueError("reduced_potentials is not beta*potential energies")


def _compare_value(
    *, expected: float | None, actual: float, tolerance: float
) -> tuple[bool, float]:
    if expected is None:
        return False, float("inf")
    return abs(actual - expected) <= tolerance, abs(actual - expected)


def _recompute_seed(
    *,
    work_dir: Path,
    seed_result: dict[str, Any],
    protocol_temperature_kelvin: float,
    protocol_lambda_states: np.ndarray,
    tolerances: Tolerances,
) -> dict[str, Any]:
    artifact = seed_result.get("raw_energy_artifact")
    if not isinstance(artifact, dict):
        raise TypeError("seed_result missing raw_energy_artifact")

    artifact_path = _resolve_artifact_path(work_dir, str(artifact.get("path", "")))
    if not artifact_path.exists():
        raise FileNotFoundError(f"missing artifact: {artifact_path}")
    _validate_artifact_metadata(artifact, artifact_path, require_sha=True)

    npz = _load_npz(artifact_path)
    reduced = np.asarray(npz["reduced_potentials"], dtype=float)
    if reduced.ndim != 3:
        raise ValueError("reduced_potentials must be 3D")
    lambda_states = np.asarray(npz.get("lambda_states"), dtype=float)
    if lambda_states.ndim != 2:
        raise ValueError("lambda_states must be 2D")
    if npz.get("sampled_state_indices") is None:
        raise ValueError("missing sampled_state_indices")
    sampled_state_indices = np.asarray(npz["sampled_state_indices"], dtype=int)
    if sampled_state_indices.ndim != 1:
        raise ValueError("sampled_state_indices must be 1D")
    expected_sampled_indices = np.arange(reduced.shape[0], dtype=int)
    if not np.array_equal(sampled_state_indices, expected_sampled_indices):
        raise ValueError(
            f"sampled_state_indices mismatch: {sampled_state_indices.tolist()} != "
            f"{expected_sampled_indices.tolist()}"
        )
    _compare_lambda_state_protocol(
        artifact_lambda_states=lambda_states,
        protocol_lambda_states=protocol_lambda_states,
    )

    beta_from_npz = float(npz.get("beta_mol_per_kj", np.nan))
    expected_beta = 1.0 / (R_KJ_PER_MOL_K * protocol_temperature_kelvin)
    if not np.isfinite(beta_from_npz) or beta_from_npz <= 0:
        raise ValueError(f"invalid beta_mol_per_kj in artifact: {artifact_path}")
    beta_ok, beta_abs = _compare_value(
        expected=expected_beta,
        actual=beta_from_npz,
        tolerance=tolerances.beta_mismatch,
    )

    selected = _selected_indices(npz, n_states=reduced.shape[0])
    beta = beta_from_npz
    if "potential_energies" in npz:
        raw = np.asarray(npz["potential_energies"])
    elif "potential_energies_kj_mol" in npz:
        raw = np.asarray(npz["potential_energies_kj_mol"])
    else:
        raw = None
    _validate_reduced_and_raw(reduced=reduced, beta=beta, raw=raw)

    recomputed = _compute_mbar(reduced=reduced, selected_indices=selected, beta=beta)

    delta_ok, delta_abs = _compare_value(
        expected=seed_result.get("delta_g_kcal_mol"),
        actual=recomputed["delta_kcal_mol"],
        tolerance=tolerances.delta_kcal_mol,
    )
    uncertainty_ok, uncertainty_abs = _compare_value(
        expected=seed_result.get("mbar_uncertainty_kcal_mol"),
        actual=recomputed["uncertainty_kcal_mol"],
        tolerance=tolerances.uncertainty_kcal_mol,
    )
    seed_n_k: list[int] | None = None
    raw_n_k = seed_result.get("n_k")
    if isinstance(raw_n_k, list):
        try:
            seed_n_k = [int(x) for x in raw_n_k]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            seed_n_k = None
    n_k_ok = seed_n_k is not None and seed_n_k == recomputed["n_k"]
    n_k_abs = None if n_k_ok else "mismatch"

    overlap_ok = True
    overlap_abs = None
    if "minimum_adjacent_overlap" in seed_result:
        overlap_ok, overlap_abs = _compare_value(
            expected=float(seed_result["minimum_adjacent_overlap"]),
            actual=recomputed["minimum_adjacent_overlap"],
            tolerance=tolerances.overlap,
        )

    return {
        "seed": seed_result.get("seed"),
        "artifact_path": str(artifact_path),
        "protocol_temperature_kelvin": protocol_temperature_kelvin,
        "lambda_states": lambda_states.tolist(),
        "reduced_shape": list(reduced.shape),
        "selected_indices": [arr.tolist() for arr in selected],
        "summary_reference": {
            "seed_delta_kcal_mol": seed_result.get("delta_g_kcal_mol"),
            "seed_uncertainty_kcal_mol": seed_result.get("mbar_uncertainty_kcal_mol"),
            "seed_minimum_adjacent_overlap": seed_result.get(
                "minimum_adjacent_overlap"
            ),
            "seed_n_k": seed_result.get("n_k"),
        },
        "recomputed": {
            "seed_delta_kcal_mol": recomputed["delta_kcal_mol"],
            "seed_uncertainty_kcal_mol": recomputed["uncertainty_kcal_mol"],
            "seed_minimum_adjacent_overlap": recomputed["minimum_adjacent_overlap"],
            "overlap_matrix": recomputed["overlap_matrix"],
            "n_k": recomputed["n_k"],
            "beta_mol_per_kj": beta_from_npz,
            "beta_expected_from_summary": expected_beta,
            "beta_abs_diff": beta_abs,
            "beta_ok": beta_ok,
        },
        "difference": {
            "delta_kcal_mol_abs": delta_abs,
            "uncertainty_kcal_mol_abs": uncertainty_abs,
            "minimum_adjacent_overlap_abs": overlap_abs,
            "n_k_abs": n_k_abs,
        },
        "ok": bool(delta_ok and uncertainty_ok and overlap_ok and beta_ok and n_k_ok),
    }


def _summarize_records(
    *,
    summary: dict[str, Any],
    work_dir: Path,
    tolerances: Tolerances,
) -> dict[str, Any]:
    protocol = summary.get("protocol", {})
    protocol_lambda_states = _protocol_lambda_states(protocol)
    temperature_kelvin = float(protocol.get("temperature_kelvin", 0.0))
    if not temperature_kelvin:
        raise ValueError("summary.protocol.temperature_kelvin is required")

    report: list[dict[str, Any]] = []
    mismatch_count = 0
    raw_artifact_count = 0

    records_value = summary.get("records", [])
    records = records_value if isinstance(records_value, list) else []
    if not records:
        mismatch_count += 1

    for record in records:
        if not isinstance(record, dict):
            mismatch_count += 1
            report.append(
                {
                    "compound_id": None,
                    "seed_count": 0,
                    "seed_recomputations": [
                        {
                            "ok": False,
                            "error": "record is not an object",
                        }
                    ],
                    "recomputed_prediction_kcal_mol": None,
                    "summary_prediction_kcal_mol": None,
                    "prediction_diff_abs": None,
                    "prediction_ok": False,
                    "seed_mismatch_count": 1,
                }
            )
            continue

        seed_results = record.get("seed_results", [])
        if not isinstance(seed_results, list):
            mismatch_count += 1
            seed_results = []

        seed_reports: list[dict[str, Any]] = []
        seed_mismatch = 0
        recomputed_predictions: list[float] = []
        if not seed_results:
            seed_mismatch += 1
            mismatch_count += 1
            seed_reports.append({"ok": False, "error": "seed_results is empty"})
        else:
            for seed_result in seed_results:
                if not isinstance(seed_result, dict):
                    seed_mismatch += 1
                    mismatch_count += 1
                    seed_reports.append(
                        {"ok": False, "error": "seed_result is not an object"}
                    )
                    continue

                artifact = seed_result.get("raw_energy_artifact")
                if isinstance(artifact, dict) and artifact.get("path") is not None:
                    try:
                        artifact_path = _resolve_artifact_path(
                            work_dir, str(artifact.get("path"))
                        )
                    except ValueError:
                        artifact_path = None
                    else:
                        if artifact_path.exists():
                            raw_artifact_count += 1

                try:
                    seed_report = _recompute_seed(
                        work_dir=work_dir,
                        seed_result=seed_result,
                        protocol_temperature_kelvin=temperature_kelvin,
                        protocol_lambda_states=protocol_lambda_states,
                        tolerances=tolerances,
                    )
                except Exception as exc:  # noqa: BLE001
                    seed_mismatch += 1
                    mismatch_count += 1
                    seed_report = {
                        "seed": seed_result.get("seed"),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    if not seed_report["ok"]:
                        seed_mismatch += 1
                        mismatch_count += 1
                    if "seed_delta_kcal_mol" in seed_report["recomputed"]:
                        recomputed_predictions.append(
                            seed_report["recomputed"]["seed_delta_kcal_mol"]
                        )
                seed_reports.append(seed_report)

        if not seed_results or not recomputed_predictions:
            recomputed_prediction = None
            pred_abs = None
            pred_ok = "prediction_kcal_mol" not in record
            if pred_ok is False:
                mismatch_count += 1
                seed_mismatch += 1
        elif "prediction_kcal_mol" in record:
            recomputed_prediction = float(statistics.fmean(recomputed_predictions))
            pred_ok, pred_abs = _compare_value(
                expected=float(record["prediction_kcal_mol"]),
                actual=recomputed_prediction,
                tolerance=tolerances.prediction_kcal_mol,
            )
            if not pred_ok:
                mismatch_count += 1
                seed_mismatch += 1
        else:
            recomputed_prediction = None
            pred_abs = None
            pred_ok = True

        report.append(
            {
                "compound_id": record.get("compound_id"),
                "name": record.get("name"),
                "seed_count": len(seed_results),
                "seed_recomputations": seed_reports,
                "recomputed_prediction_kcal_mol": recomputed_prediction,
                "summary_prediction_kcal_mol": record.get("prediction_kcal_mol"),
                "prediction_diff_abs": pred_abs,
                "prediction_ok": pred_ok,
                "seed_mismatch_count": seed_mismatch,
            }
        )

    if raw_artifact_count == 0:
        mismatch_count += 1

    return {
        "records": report,
        "mismatch_count": mismatch_count,
        "n_records": len(records_value) if isinstance(records_value, list) else 0,
        "raw_artifact_count": raw_artifact_count,
        "verified": mismatch_count == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute MBAR from LSNN raw u-kln artifacts and compare summary values."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--delta-tol", type=float, default=Tolerances().delta_kcal_mol)
    parser.add_argument(
        "--uncertainty-tol", type=float, default=Tolerances().uncertainty_kcal_mol
    )
    parser.add_argument("--overlap-tol", type=float, default=Tolerances().overlap)
    parser.add_argument(
        "--prediction-tol", type=float, default=Tolerances().prediction_kcal_mol
    )
    parser.add_argument("--beta-tol", type=float, default=Tolerances().beta_mismatch)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tolerances = Tolerances(
        delta_kcal_mol=float(args.delta_tol),
        uncertainty_kcal_mol=float(args.uncertainty_tol),
        overlap=float(args.overlap_tol),
        prediction_kcal_mol=float(args.prediction_tol),
        beta_mismatch=float(args.beta_tol),
    )

    summary = _read_summary(args.summary)
    work_dir = args.summary.parent
    report = _summarize_records(
        summary=summary,
        work_dir=work_dir,
        tolerances=tolerances,
    )

    output = {
        "artifact": "lsnn_raw_mbar_recompute",
        "summary_path": str(args.summary),
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "tolerances": {
            "delta_kcal_mol": tolerances.delta_kcal_mol,
            "uncertainty_kcal_mol": tolerances.uncertainty_kcal_mol,
            "overlap": tolerances.overlap,
            "prediction_kcal_mol": tolerances.prediction_kcal_mol,
            "beta_mismatch": tolerances.beta_mismatch,
        },
        **report,
    }

    serialized = json.dumps(output, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized + "\n", encoding="utf-8")

    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
