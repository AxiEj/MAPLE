#!/usr/bin/env python3
"""Reweight the frozen Route 1 conformer set with MLIP gas relative energies."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_conformer_sensitivity import (  # noqa: E402
    _load_conformer_protocol,
    _read_xyz_ensemble,
)
from maple.function.free_energy.discrete_conformers import (  # noqa: E402
    analyze_discrete_conformer_ensemble,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

KCAL_PER_HARTREE = 627.5094740631
HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and set(text) <= HEX


def load_weighting_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Unsupported MLIP conformer-weighting protocol schema.")
    if protocol.get("source_partition") != "development":
        raise ValueError("MLIP conformer weighting must remain development-only.")
    for key in ("source_evidence", "model", "evaluation", "failure_policy"):
        if not isinstance(protocol.get(key), dict):
            raise ValueError(f"MLIP conformer-weighting protocol lacks {key}.")
    for key in (
        "conformer_protocol_sha256",
        "conformer_protocol_fingerprint",
        "conformer_summary_sha256",
    ):
        if not _is_sha256(protocol["source_evidence"].get(key)):
            raise ValueError(f"source_evidence.{key} must be a SHA256.")
    if not _is_sha256(protocol["model"].get("checkpoint_sha256")):
        raise ValueError("model.checkpoint_sha256 must be a SHA256.")
    temperature = protocol["evaluation"].get("temperature_kelvin")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("evaluation.temperature_kelvin must be positive.")
    fingerprint = sha256_bytes(canonical_json_bytes(protocol))
    return protocol, fingerprint


def _aligned_rmsd(
    left: np.ndarray, right: np.ndarray, atom_indices: np.ndarray
) -> float:
    left_selected = np.asarray(left, dtype=np.float64)[atom_indices]
    right_selected = np.asarray(right, dtype=np.float64)[atom_indices]
    left_centered = left_selected - left_selected.mean(axis=0)
    right_centered = right_selected - right_selected.mean(axis=0)
    covariance = left_centered.T @ right_centered
    left_vectors, _singular_values, right_vectors = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left_vectors @ right_vectors))
    rotation = left_vectors @ correction @ right_vectors
    difference = left_centered @ rotation - right_centered
    return float(np.sqrt(np.square(difference).sum() / len(atom_indices)))


def reference_geometry_union(
    frames: list[dict[str, Any]],
    reference_positions: np.ndarray,
    symbols: list[str],
    *,
    threshold_angstrom: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ensure the exact reference geometry is represented once in the state set."""

    if threshold_angstrom <= 0.0:
        raise ValueError("Reference-geometry RMSD threshold must be positive.")
    heavy = np.asarray(
        [index for index, symbol in enumerate(symbols) if symbol != "H"], dtype=int
    )
    if not len(heavy):
        heavy = np.arange(len(symbols), dtype=int)
    rmsd = np.asarray(
        [
            _aligned_rmsd(frame["positions_angstrom"], reference_positions, heavy)
            for frame in frames
        ],
        dtype=np.float64,
    )
    closest = int(np.argmin(rmsd))
    union = [dict(frame) for frame in frames]
    if float(rmsd[closest]) <= threshold_angstrom:
        union[closest] = {
            "positions_angstrom": np.asarray(reference_positions, dtype=np.float64),
            "source": "freesolv-reference-replaced-nearest",
        }
        action = "replaced"
        reference_index = closest
    else:
        union.append(
            {
                "positions_angstrom": np.asarray(reference_positions, dtype=np.float64),
                "source": "freesolv-reference-appended",
            }
        )
        action = "appended"
        reference_index = len(union) - 1
    return union, {
        "action": action,
        "reference_index": reference_index,
        "closest_source_conformer_index": closest,
        "aligned_heavy_atom_rmsd_angstrom": float(rmsd[closest]),
        "threshold_angstrom": float(threshold_angstrom),
    }


def discrete_partition_result(
    *,
    mlip_energy_hartree: list[float],
    solvent_kcal_mol: list[float],
    temperature_kelvin: float,
    reference_geometry_kcal_mol: float,
    experimental_kcal_mol: float,
) -> dict[str, Any]:
    """Evaluate the discrete conformer partition-function ratio.

    Only relative MLIP energies enter. Absolute offsets cancel by subtracting
    the lowest MLIP energy before either partition function is evaluated.
    """

    mlip = np.asarray(mlip_energy_hartree, dtype=np.float64)
    solvent = np.asarray(solvent_kcal_mol, dtype=np.float64)
    if (
        mlip.ndim != 1
        or solvent.ndim != 1
        or len(mlip) != len(solvent)
        or not len(mlip)
    ):
        raise ValueError(
            "MLIP and solvent energy arrays must be non-empty and equal length."
        )
    if not np.isfinite(mlip).all() or not np.isfinite(solvent).all():
        raise ValueError("MLIP and solvent energies must be finite.")
    if not math.isfinite(float(temperature_kelvin)) or temperature_kelvin <= 0.0:
        raise ValueError("Temperature must be positive and finite.")

    relative_mlip = (mlip - float(mlip.min())) * KCAL_PER_HARTREE
    analysis = analyze_discrete_conformer_ensemble(
        gas_energy_kcal_mol=relative_mlip,
        solvent_correction_kcal_mol=solvent,
        temperature_kelvin=temperature_kelvin,
        state_ids=[str(index) for index in range(len(mlip))],
    )
    ensemble = analysis["delta_g_discrete_kcal_mol"]
    minimum_index = int(np.argmin(mlip))
    reference = float(reference_geometry_kcal_mol)
    experimental = float(experimental_kcal_mol)

    return {
        "temperature_kelvin": float(temperature_kelvin),
        "rt_kcal_mol": analysis["rt_kcal_mol"],
        "ensemble_kcal_mol": float(ensemble),
        "reference_geometry_kcal_mol": reference,
        "lowest_mlip_conformer_endpoint_kcal_mol": float(solvent[minimum_index]),
        "lowest_mlip_conformer_index": minimum_index,
        "experimental_kcal_mol": experimental,
        "reference_error_kcal_mol": reference - experimental,
        "ensemble_error_kcal_mol": float(ensemble - experimental),
        "conformational_correction_from_reference_kcal_mol": float(
            ensemble - reference
        ),
        "mlip_relative_energy_kcal_mol": relative_mlip.tolist(),
        "mlip_relative_energy_span_kcal_mol": float(relative_mlip.max()),
        "gas_weights": analysis["gas_weights"],
        "solution_weights": analysis["solution_weights"],
        "gas_effective_conformer_count": analysis[
            "gas_effective_conformer_count"
        ],
        "solution_effective_conformer_count": analysis[
            "solution_effective_conformer_count"
        ],
        "gas_maximum_weight": analysis["gas_maximum_weight"],
        "solution_maximum_weight": analysis["solution_maximum_weight"],
        "gas_dominant_conformer_index": analysis[
            "gas_dominant_state_index"
        ],
        "solution_dominant_conformer_index": analysis[
            "solution_dominant_state_index"
        ],
        "distribution_overlap": analysis["distribution_overlap"],
        "weight_diagnostic_gates": analysis["gates"],
        "claim_boundary": analysis["claim_boundary"],
        "route_contract": analysis["route_contract"],
    }


def _metric_block(
    predicted: np.ndarray, experimental: np.ndarray
) -> dict[str, float | int]:
    errors = predicted - experimental
    return {
        "n": int(len(errors)),
        "mse_kcal_mol": float(errors.mean()),
        "mae_kcal_mol": float(np.abs(errors).mean()),
        "rmse_kcal_mol": float(np.sqrt(np.square(errors).mean())),
        "maximum_absolute_error_kcal_mol": float(np.abs(errors).max()),
    }


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty MLIP conformer record set.")

    def blocks(selected: list[dict[str, Any]]) -> dict[str, Any]:
        experimental = np.asarray(
            [record["experimental_kcal_mol"] for record in selected], dtype=np.float64
        )
        reference = np.asarray(
            [record["result"]["reference_geometry_kcal_mol"] for record in selected],
            dtype=np.float64,
        )
        ensemble = np.asarray(
            [record["result"]["ensemble_kcal_mol"] for record in selected],
            dtype=np.float64,
        )
        return {
            "reference_geometry": _metric_block(reference, experimental),
            "mlip_weighted_ensemble": _metric_block(ensemble, experimental),
        }

    metrics = blocks(records)
    outcomes = {"improved": 0, "unchanged": 0, "worsened": 0}
    corrections: list[float] = []
    for record in records:
        result = record["result"]
        before = abs(
            result["reference_geometry_kcal_mol"] - record["experimental_kcal_mol"]
        )
        after = abs(result["ensemble_kcal_mol"] - record["experimental_kcal_mol"])
        if after < before - 1.0e-12:
            outcomes["improved"] += 1
        elif after > before + 1.0e-12:
            outcomes["worsened"] += 1
        else:
            outcomes["unchanged"] += 1
        corrections.append(
            result["ensemble_kcal_mol"] - result["reference_geometry_kcal_mol"]
        )

    correction_array = np.asarray(corrections, dtype=np.float64)
    metrics["case_outcomes"] = outcomes
    metrics["conformational_correction_kcal_mol"] = {
        "minimum": float(correction_array.min()),
        "median": float(np.median(correction_array)),
        "p90_absolute": float(np.quantile(np.abs(correction_array), 0.9)),
        "maximum": float(correction_array.max()),
        "maximum_absolute": float(np.abs(correction_array).max()),
    }
    labels = sorted({record["flexibility_bin"] for record in records})
    metrics["flexibility_strata"] = {
        label: blocks(
            [record for record in records if record["flexibility_bin"] == label]
        )
        for label in labels
    }
    full_gain = (
        metrics["reference_geometry"]["mae_kcal_mol"]
        - metrics["mlip_weighted_ensemble"]["mae_kcal_mol"]
    )
    leave_one_out = {}
    for excluded_index, excluded in enumerate(records):
        retained = [
            record for index, record in enumerate(records) if index != excluded_index
        ]
        retained_blocks = blocks(retained)
        leave_one_out[excluded["compound_id"]] = (
            retained_blocks["reference_geometry"]["mae_kcal_mol"]
            - retained_blocks["mlip_weighted_ensemble"]["mae_kcal_mol"]
        )
    most_influential = max(
        leave_one_out,
        key=lambda compound_id: abs(full_gain - leave_one_out[compound_id]),
    )
    metrics["influence_analysis"] = {
        "full_mae_improvement_kcal_mol": full_gain,
        "most_influential_compound_id": most_influential,
        "mae_improvement_without_most_influential_kcal_mol": leave_one_out[
            most_influential
        ],
        "leave_one_out_minimum_mae_improvement_kcal_mol": min(leave_one_out.values()),
        "leave_one_out_maximum_mae_improvement_kcal_mol": max(leave_one_out.values()),
    }
    return metrics


def _validate_sources(
    protocol: dict[str, Any],
    protocol_path: Path,
    source_output_dir: Path,
    base_work_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source = protocol["source_evidence"]
    conformer_protocol_path = protocol_path.parent / source["conformer_protocol"]
    conformer_summary_path = protocol_path.parent / source["conformer_summary"]
    if sha256_file(conformer_protocol_path) != source["conformer_protocol_sha256"]:
        raise ValueError("Frozen conformer protocol hash changed.")
    conformer_protocol, conformer_fingerprint, _ = _load_conformer_protocol(
        conformer_protocol_path
    )
    if conformer_fingerprint != source["conformer_protocol_fingerprint"]:
        raise ValueError("Frozen conformer protocol fingerprint changed.")
    if sha256_file(conformer_summary_path) != source["conformer_summary_sha256"]:
        raise ValueError("Frozen conformer summary hash changed.")
    conformer_summary = load_json(conformer_summary_path)
    if conformer_summary.get("protocol_fingerprint") != conformer_fingerprint:
        raise ValueError("Frozen conformer summary targets a different protocol.")

    prepared = load_json(base_work_dir / "prepared.json")
    candidates = {
        candidate["compound_id"]: candidate for candidate in prepared["candidates"]
    }
    cases = conformer_protocol["cases"]
    for case in cases:
        compound_id = case["compound_id"]
        source_record_path = source_output_dir / "records" / f"{compound_id}.json"
        if (
            sha256_file(source_record_path)
            != conformer_summary["record_sha256"][compound_id]
        ):
            raise ValueError(
                f"Frozen source conformer record hash changed: {compound_id}"
            )
        candidate = candidates.get(compound_id)
        if candidate is None or candidate.get("partition") != "development":
            raise ValueError(
                f"MLIP conformer case is not development-only: {compound_id}"
            )
    return cases, conformer_summary, candidates


def _load_calculator(protocol: dict[str, Any], device: str, first_atoms):
    import torch

    from maple.function.calculator.set_calculator import SetCalculator

    requested_device = torch.device(device)
    calculator = SetCalculator(
        requested_device,
        protocol["model"]["name"],
        str(Path("/tmp") / "maple-mlip-conformer-weighting.log"),
        atoms=first_atoms,
    ).set_calculator()
    checkpoint_path = (
        REPOSITORY_ROOT
        / "maple/function/calculator/model"
        / protocol["model"]["checkpoint_filename"]
    )
    if sha256_file(checkpoint_path) != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("MLIP checkpoint hash does not match the frozen protocol.")
    observed_numbers = [int(value) for value in calculator.atomic_numbers]
    if observed_numbers != protocol["model"]["supported_atomic_numbers"]:
        raise ValueError("MLIP checkpoint atomic-number table changed.")
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(requested_device),
        "gpu_name": (
            torch.cuda.get_device_name(requested_device)
            if requested_device.type == "cuda"
            else None
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "model_atomic_numbers": observed_numbers,
        "model_cutoff_angstrom": float(calculator.r_max),
        "model_dtype": str(calculator.dtype),
        "ase_version": importlib.metadata.version("ase"),
    }
    return calculator, environment


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_weighting_protocol(protocol_path)
    source_output_dir = Path(args.source_output_dir).resolve()
    base_work_dir = Path(args.base_work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cases, conformer_summary, candidates = _validate_sources(
        protocol, protocol_path, source_output_dir, base_work_dir
    )
    source_records = {
        case["compound_id"]: load_json(
            source_output_dir / "records" / f"{case['compound_id']}.json"
        )
        for case in cases
    }
    first_case = cases[0]
    first_candidate = candidates[first_case["compound_id"]]
    first_atoms = MOL2Reader(
        str(base_work_dir / first_candidate["mol2_relative_path"]), charge=0, mult=1
    )
    calculator, environment = _load_calculator(protocol, args.device, first_atoms)
    method_key = (
        f"{protocol['evaluation']['charge_method']}/"
        f"{protocol['evaluation']['gb_model']}"
    )
    output_dir.joinpath("records").mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for case in cases:
        compound_id = case["compound_id"]
        destination = output_dir / "records" / f"{compound_id}.json"
        if destination.is_file():
            skipped += 1
            continue
        source_record = source_records[compound_id]
        candidate = candidates[compound_id]
        try:
            source_method = source_record["methods"][method_key]
            if (
                source_record.get("status") != "success"
                or source_method.get("status") != "success"
            ):
                raise ValueError(
                    f"Frozen source method is not successful: {method_key}"
                )
            ensemble_path = Path(source_record["audit_dir"]) / "crest_conformers.xyz"
            if sha256_file(ensemble_path) != source_record["ensemble_sha256"]:
                raise ValueError("Frozen CREST ensemble hash changed.")
            mol2_path = base_work_dir / candidate["mol2_relative_path"]
            atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
            frames = _read_xyz_ensemble(
                ensemble_path, list(atoms.get_chemical_symbols())
            )
            if len(frames) != source_record["conformer_count"]:
                raise ValueError("Frozen conformer count changed.")
            source_conformer_count = len(frames)
            frames, reference_union = reference_geometry_union(
                frames,
                atoms.get_positions(),
                list(atoms.get_chemical_symbols()),
                threshold_angstrom=protocol["evaluation"][
                    "reference_geometry_rmsd_dedup_angstrom"
                ],
            )
            energies: list[float] = []
            start = time.perf_counter()
            atoms.calc = calculator
            for frame in frames:
                atoms.set_positions(frame["positions_angstrom"])
                energy = float(atoms.get_potential_energy())
                if not math.isfinite(energy):
                    raise ValueError("MLIP returned a non-finite energy.")
                energies.append(energy)
            elapsed = time.perf_counter() - start
            solvent = [
                float(value["total_kcal_mol"])
                for value in source_method["conformer_values"]
            ]
            reference_index = reference_union["reference_index"]
            reference_value = float(source_method["reference_geometry_kcal_mol"])
            if reference_union["action"] == "replaced":
                solvent[reference_index] = reference_value
            else:
                solvent.append(reference_value)
            result = discrete_partition_result(
                mlip_energy_hartree=energies,
                solvent_kcal_mol=solvent,
                temperature_kelvin=protocol["evaluation"]["temperature_kelvin"],
                reference_geometry_kcal_mol=source_method[
                    "reference_geometry_kcal_mol"
                ],
                experimental_kcal_mol=candidate["experimental_kcal_mol"],
            )
            record = {
                "schema_version": 1,
                "artifact_type": "mlip-conformer-weighting-case",
                "protocol_id": protocol["protocol_id"],
                "protocol_fingerprint": fingerprint,
                "source_partition": "development",
                "compound_id": compound_id,
                "name": case["name"],
                "flexibility_bin": case["flexibility_bin"],
                "source_conformer_count": source_conformer_count,
                "partition_state_count": len(frames),
                "reference_geometry_union": reference_union,
                "experimental_kcal_mol": candidate["experimental_kcal_mol"],
                "method": method_key,
                "model": protocol["model"],
                "environment": environment,
                "source_record_sha256": conformer_summary["record_sha256"][compound_id],
                "ensemble_sha256": source_record["ensemble_sha256"],
                "mlip_energy_hartree": energies,
                "solvent_correction_kcal_mol": solvent,
                "timing": {
                    "total_seconds": elapsed,
                    "seconds_per_conformer": elapsed / len(frames),
                },
                "result": result,
                "status": "success",
                "interpretation": protocol["claim_scope"],
            }
        except Exception as exc:
            record = {
                "schema_version": 1,
                "artifact_type": "mlip-conformer-weighting-case",
                "protocol_id": protocol["protocol_id"],
                "protocol_fingerprint": fingerprint,
                "source_partition": "development",
                "compound_id": compound_id,
                "name": case["name"],
                "flexibility_bin": case["flexibility_bin"],
                "status": "failure",
                "failure": {
                    "exception_class": type(exc).__name__,
                    "reason": str(exc),
                },
            }
        write_json_atomic(destination, record)
        completed += 1
        print(f"{compound_id}: {record['status']}", flush=True)
    print(f"Wrote {completed} MLIP weighting records; resumed/skipped {skipped}.")


def summarize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_weighting_protocol(protocol_path)
    conformer_protocol, _source_fingerprint, _ = _load_conformer_protocol(
        protocol_path.parent / protocol["source_evidence"]["conformer_protocol"]
    )
    output_dir = Path(args.output_dir).resolve()
    expected = [case["compound_id"] for case in conformer_protocol["cases"]]
    record_dir = output_dir / "records"
    missing = [
        case_id
        for case_id in expected
        if not (record_dir / f"{case_id}.json").is_file()
    ]
    extra = sorted(
        path.stem
        for path in record_dir.glob("*.json")
        if path.stem not in set(expected)
    )
    if missing or extra:
        raise ValueError(
            f"MLIP weighting record reconciliation failed: missing={len(missing)}, "
            f"extra={len(extra)}."
        )
    records = [load_json(record_dir / f"{case_id}.json") for case_id in expected]
    invalid = [
        record["compound_id"]
        for record in records
        if record.get("protocol_fingerprint") != fingerprint
        or record.get("status") != "success"
    ]
    if invalid:
        raise ValueError(f"MLIP weighting has invalid/failed records: {invalid}")
    environment_payloads = {
        canonical_json_bytes(record["environment"]) for record in records
    }
    if len(environment_payloads) != 1:
        raise ValueError("MLIP weighting records used inconsistent environments.")
    total_seconds = sum(record["timing"]["total_seconds"] for record in records)
    source_conformers = sum(record["source_conformer_count"] for record in records)
    partition_states = sum(record["partition_state_count"] for record in records)
    summary = {
        "schema_version": 1,
        "artifact_type": "mlip-conformer-weighting-summary",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "case_count": len(records),
        "source_conformer_count": source_conformers,
        "partition_state_count": partition_states,
        "method": (
            f"{protocol['evaluation']['charge_method']}/"
            f"{protocol['evaluation']['gb_model']}/"
            f"{protocol['evaluation']['nonpolar']}"
        ),
        "model": protocol["model"],
        "environment": json.loads(next(iter(environment_payloads))),
        "timing": {
            "total_seconds": total_seconds,
            "seconds_per_state": total_seconds / partition_states,
            "states_per_second": partition_states / total_seconds,
        },
        "metrics": summarize_metrics(records),
        "cases": {
            record["compound_id"]: {
                "name": record["name"],
                "flexibility_bin": record["flexibility_bin"],
                "source_conformer_count": record["source_conformer_count"],
                "partition_state_count": record["partition_state_count"],
                "reference_geometry_union": record["reference_geometry_union"],
                "experimental_kcal_mol": record["experimental_kcal_mol"],
                "reference_geometry_kcal_mol": record["result"][
                    "reference_geometry_kcal_mol"
                ],
                "mlip_weighted_ensemble_kcal_mol": record["result"][
                    "ensemble_kcal_mol"
                ],
                "conformational_correction_kcal_mol": record["result"][
                    "conformational_correction_from_reference_kcal_mol"
                ],
                "gas_effective_conformer_count": record["result"][
                    "gas_effective_conformer_count"
                ],
                "solution_effective_conformer_count": record["result"][
                    "solution_effective_conformer_count"
                ],
            }
            for record in records
        },
        "record_sha256": {
            case_id: sha256_file(record_dir / f"{case_id}.json") for case_id in expected
        },
        "interpretation": protocol["claim_scope"],
        "limitations": protocol["limitations"],
        "next_gate": protocol["next_gate"],
    }
    write_json_atomic(args.output, summary)
    print(f"Wrote MLIP conformer-weighting summary to {Path(args.output).resolve()}.")


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _legacy_core_differences(
    legacy: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, float]:
    pairs = {
        "ensemble_kcal_mol": (
            legacy["ensemble_kcal_mol"],
            analysis["delta_g_discrete_kcal_mol"],
        ),
        "rt_kcal_mol": (legacy["rt_kcal_mol"], analysis["rt_kcal_mol"]),
        "mlip_relative_energy_kcal_mol": (
            legacy["mlip_relative_energy_kcal_mol"],
            analysis["gas_relative_energy_kcal_mol"],
        ),
        "gas_weights": (legacy["gas_weights"], analysis["gas_weights"]),
        "solution_weights": (
            legacy["solution_weights"],
            analysis["solution_weights"],
        ),
        "gas_effective_conformer_count": (
            legacy["gas_effective_conformer_count"],
            analysis["gas_effective_conformer_count"],
        ),
        "solution_effective_conformer_count": (
            legacy["solution_effective_conformer_count"],
            analysis["solution_effective_conformer_count"],
        ),
        "gas_maximum_weight": (
            legacy["gas_maximum_weight"],
            analysis["gas_maximum_weight"],
        ),
        "solution_maximum_weight": (
            legacy["solution_maximum_weight"],
            analysis["solution_maximum_weight"],
        ),
        "gas_dominant_conformer_index": (
            legacy["gas_dominant_conformer_index"],
            analysis["gas_dominant_state_index"],
        ),
        "solution_dominant_conformer_index": (
            legacy["solution_dominant_conformer_index"],
            analysis["solution_dominant_state_index"],
        ),
    }
    return {
        name: float(
            np.max(
                np.abs(
                    np.asarray(previous, dtype=np.float64)
                    - np.asarray(current, dtype=np.float64)
                )
            )
        )
        for name, (previous, current) in pairs.items()
    }


def audit_core(args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for configured in args.record_dir:
        record_dir = Path(configured).resolve()
        paths = sorted(record_dir.glob("*.json"))
        if not paths:
            raise ValueError(f"No conformer-weighting records found in {record_dir}.")
        for path in paths:
            source = load_json(path)
            if source.get("status") != "success":
                raise ValueError(
                    f"Conformer-weighting record is not successful: {path}."
                )
            key = (str(source["protocol_id"]), str(source["compound_id"]))
            if key in seen:
                raise ValueError(f"Duplicate conformer-weighting record: {key}.")
            seen.add(key)
            gas_hartree = np.asarray(
                source["mlip_energy_hartree"],
                dtype=np.float64,
            )
            gas_kcal_mol = (
                gas_hartree - float(np.min(gas_hartree))
            ) * KCAL_PER_HARTREE
            analysis = analyze_discrete_conformer_ensemble(
                gas_energy_kcal_mol=gas_kcal_mol,
                solvent_correction_kcal_mol=source[
                    "solvent_correction_kcal_mol"
                ],
                temperature_kelvin=source["result"]["temperature_kelvin"],
                state_ids=[str(index) for index in range(len(gas_hartree))],
            )
            differences = _legacy_core_differences(source["result"], analysis)
            records.append(
                {
                    "protocol_id": source["protocol_id"],
                    "compound_id": source["compound_id"],
                    "method": source["method"],
                    "model": source["model"]["name"],
                    "state_count": analysis["state_count"],
                    "source_record": _relative_to_repository(path),
                    "source_record_sha256": sha256_file(path),
                    "legacy_field_max_abs_difference": max(
                        differences.values()
                    ),
                    "legacy_field_differences": differences,
                    "distribution_overlap": analysis["distribution_overlap"],
                    "gas_effective_conformer_count": analysis[
                        "gas_effective_conformer_count"
                    ],
                    "solution_effective_conformer_count": analysis[
                        "solution_effective_conformer_count"
                    ],
                    "gas_maximum_weight": analysis["gas_maximum_weight"],
                    "solution_maximum_weight": analysis[
                        "solution_maximum_weight"
                    ],
                    "weight_diagnostic_gates": analysis["gates"],
                }
            )

    methods = sorted({record["method"] for record in records})
    method_summaries = {}
    for method in methods:
        selected = [record for record in records if record["method"] == method]
        overlaps = [record["distribution_overlap"] for record in selected]
        gas_effective = [
            record["gas_effective_conformer_count"] for record in selected
        ]
        solution_effective = [
            record["solution_effective_conformer_count"] for record in selected
        ]
        method_summaries[method] = {
            "record_count": len(selected),
            "state_count": sum(record["state_count"] for record in selected),
            "weight_diagnostic_pass_count": sum(
                record["weight_diagnostic_gates"]["ensemble_diagnostic_passed"]
                for record in selected
            ),
            "distribution_overlap": {
                "minimum": min(overlaps),
                "median": float(np.median(overlaps)),
                "maximum": max(overlaps),
            },
            "gas_effective_conformer_count": {
                "minimum": min(gas_effective),
                "median": float(np.median(gas_effective)),
                "maximum": max(gas_effective),
            },
            "solution_effective_conformer_count": {
                "minimum": min(solution_effective),
                "median": float(np.median(solution_effective)),
                "maximum": max(solution_effective),
            },
        }

    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-discrete-conformer-core-replay",
        "route": {
            "name": "Additive fixed-charge PB/GB implicit solvation",
            "formula": (
                "E_solution(R)=E_MLIP,gas(R)+"
                "G_polar(R,q_fixed)+G_nonpolar(R)"
            ),
            "gas_phase_mm_energy": False,
            "hydration_label_residual": False,
            "retraining": False,
        },
        "purpose": (
            "Replay historical MACE conformer records through the reusable, "
            "MLIP-agnostic discrete-conformer core and add weight diagnostics."
        ),
        "claim_boundary": {
            "historical_records_contain_experimental_fields": True,
            "experimental_fields_used_by_core_or_replay_metrics": False,
            "public_solvfe_eligible": False,
            "reason": (
                "Exact legacy replay and weight diagnostics do not establish "
                "conformer-set completeness, basin measures, or uncertainty."
            ),
        },
        "record_count": len(records),
        "state_count": sum(record["state_count"] for record in records),
        "all_legacy_fields_exact": all(
            record["legacy_field_max_abs_difference"] == 0.0
            for record in records
        ),
        "method_summaries": method_summaries,
        "records": records,
        "command_provenance": command_provenance(
            __file__,
            {
                "phase": "audit-core",
                "record_dir": args.record_dir,
                "output": args.output,
            },
            repository_root=REPOSITORY_ROOT,
        ),
    }
    seal_artifact(artifact)
    write_json_atomic(args.output, artifact)
    print(f"Wrote discrete-conformer core audit to {Path(args.output).resolve()}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    run_parser = subparsers.add_parser(
        "run", help="evaluate frozen conformers with MLIP"
    )
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--source-output-dir", required=True)
    run_parser.add_argument("--base-work-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.set_defaults(handler=run)

    summary_parser = subparsers.add_parser(
        "summarize", help="summarize MLIP conformer-weighting records"
    )
    summary_parser.add_argument("--protocol", required=True)
    summary_parser.add_argument("--output-dir", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.set_defaults(handler=summarize)

    audit_parser = subparsers.add_parser(
        "audit-core",
        help="replay historical records through the reusable conformer core",
    )
    audit_parser.add_argument("--record-dir", action="append", required=True)
    audit_parser.add_argument("--output", required=True)
    audit_parser.set_defaults(handler=audit_core)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
