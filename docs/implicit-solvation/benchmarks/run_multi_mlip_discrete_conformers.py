#!/usr/bin/env python3
"""Run a label-separated multi-MLIP discrete-conformer Route 1 diagnostic."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
from itertools import combinations
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

from ase import Atoms
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
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
from run_conformer_sensitivity import _read_xyz_ensemble  # noqa: E402
from run_mlip_conformer_weighting import reference_geometry_union  # noqa: E402

from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.free_energy import (  # noqa: E402
    analyze_discrete_conformer_ensemble,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

KCAL_PER_HARTREE = 627.5094740631
FORBIDDEN_LABEL_KEYS = {
    "experimental_kcal_mol",
    "experimental_reference",
    "experimental_uncertainty_kcal_mol",
}


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _array_payload_sha256(values: list[float]) -> str:
    return sha256_bytes(canonical_json_bytes([float(value) for value in values]))


def _contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_LABEL_KEYS or _contains_forbidden_label_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _command_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key != "handler"
    }


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Unsupported multi-MLIP conformer protocol schema.")
    if protocol.get("source_partition") != "development":
        raise ValueError("Multi-MLIP conformer work must remain development-only.")
    models = protocol.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("Protocol must pin at least two gas-phase MLIPs.")
    names = [str(model.get("name", "")) for model in models]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Protocol model names must be unique and non-empty.")
    domains = []
    for model in models:
        domain = sorted({int(value) for value in model["supported_atomic_numbers"]})
        if not domain:
            raise ValueError(f"Model {model['name']} has an empty element domain.")
        if len(str(model.get("checkpoint_sha256", ""))) != 64:
            raise ValueError(f"Model {model['name']} lacks a checkpoint SHA256.")
        domains.append(set(domain))
    common = sorted(set.intersection(*domains))
    if common != protocol["selection"]["common_atomic_numbers"]:
        raise ValueError("Pinned common model element domain is inconsistent.")
    if protocol["solvent_endpoint"] != {
        "charge_method": "am1bcc",
        "polar_model": "obc2",
        "nonpolar_model": "ace",
        "fixed_charge": True,
        "same_correction_for_every_gas_mlip": True,
    }:
        raise ValueError("Protocol solvent endpoint is not the fixed AM1-BCC default.")
    route = protocol["route"]
    if (
        route.get("gas_phase_mm_energy") is not False
        or route.get("hydration_label_residual") is not False
        or route.get("mlip_retraining") is not False
    ):
        raise ValueError("Protocol violates the Route 1 additive target.")
    if int(protocol["execution"]["repeat_count"]) < 2:
        raise ValueError("Protocol must request at least two independent evaluations.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _source_path(
    protocol_path: Path,
    protocol: dict[str, Any],
    name: str,
) -> Path:
    source = protocol["source_evidence"]
    path = protocol_path.parent / source[name]
    observed = sha256_file(path)
    expected = source[f"{name}_sha256"]
    if observed != expected:
        raise ValueError(f"Frozen source hash changed for {name}: {path}.")
    return path


def _checkpoint_path(model: dict[str, Any]) -> Path:
    path = (
        REPOSITORY_ROOT
        / "maple/function/calculator/model"
        / model["checkpoint_filename"]
    )
    if sha256_file(path) != model["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash changed for {model['name']}.")
    return path


def _case_atomic_numbers(case: dict[str, Any]) -> list[int]:
    try:
        return sorted({int(ASE_ATOMIC_NUMBERS[symbol]) for symbol in case["elements"]})
    except KeyError as exc:
        raise ValueError(
            f"Unknown element symbol in case {case['compound_id']}."
        ) from exc


def _select_cases(
    protocol: dict[str, Any],
    conformer_protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_cases = conformer_protocol["cases"]
    selection = protocol["selection"]
    if len(source_cases) != selection["expected_source_case_count"]:
        raise ValueError("Frozen conformer source-case count changed.")
    common = set(selection["common_atomic_numbers"])
    selected = []
    excluded = []
    for case in source_cases:
        numbers = _case_atomic_numbers(case)
        unsupported = sorted(set(numbers).difference(common))
        if unsupported:
            excluded.append(
                {
                    "compound_id": case["compound_id"],
                    "unsupported_atomic_numbers": unsupported,
                }
            )
        else:
            selected.append(case)
    expected_exclusions = [
        {
            "compound_id": row["compound_id"],
            "unsupported_atomic_numbers": row["unsupported_atomic_numbers"],
        }
        for row in selection["expected_exclusions"]
    ]
    if excluded != expected_exclusions:
        raise ValueError("Common-domain case exclusions changed.")
    if len(selected) != selection["expected_selected_case_count"]:
        raise ValueError("Common-domain selected-case count changed.")
    return selected, excluded


def prepare(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    conformer_protocol_path = _source_path(
        protocol_path, protocol, "conformer_protocol"
    )
    conformer_summary_path = _source_path(
        protocol_path, protocol, "conformer_summary"
    )
    weighting_protocol_path = _source_path(
        protocol_path, protocol, "weighting_protocol"
    )
    weighting_summary_path = _source_path(
        protocol_path, protocol, "weighting_summary"
    )
    conformer_protocol = load_json(conformer_protocol_path)
    conformer_summary = load_json(conformer_summary_path)
    weighting_protocol = load_json(weighting_protocol_path)
    weighting_summary = load_json(weighting_summary_path)
    selected, excluded = _select_cases(protocol, conformer_protocol)

    if weighting_protocol["evaluation"]["charge_method"] != "am1bcc":
        raise ValueError("Historical weighting source is not AM1-BCC.")
    if weighting_summary["method"] != "am1bcc/obc2/ace":
        raise ValueError("Historical weighting summary method changed.")

    conformer_record_dir = Path(args.conformer_record_dir).resolve()
    weighting_record_dir = Path(args.weighting_record_dir).resolve()
    base_work_dir = Path(args.base_work_dir).resolve()
    prepared_path = base_work_dir / "prepared.json"
    prepared = load_json(prepared_path)
    candidates = {
        candidate["compound_id"]: candidate for candidate in prepared["candidates"]
    }
    raw_dir = Path(args.raw_dir).resolve()
    state_dir = raw_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)

    for model in protocol["models"]:
        _checkpoint_path(model)

    cases = []
    total_states = 0
    for case in selected:
        compound_id = case["compound_id"]
        conformer_record_path = conformer_record_dir / f"{compound_id}.json"
        weighting_record_path = weighting_record_dir / f"{compound_id}.json"
        if (
            sha256_file(conformer_record_path)
            != conformer_summary["record_sha256"][compound_id]
        ):
            raise ValueError(f"Conformer record hash changed for {compound_id}.")
        if (
            sha256_file(weighting_record_path)
            != weighting_summary["record_sha256"][compound_id]
        ):
            raise ValueError(f"Weighting record hash changed for {compound_id}.")
        conformer_record = load_json(conformer_record_path)
        weighting_record = load_json(weighting_record_path)
        candidate = candidates[compound_id]
        if candidate["partition"] != "development":
            raise ValueError(f"Case is not development-only: {compound_id}.")
        if weighting_record["method"] != "am1bcc/obc2":
            raise ValueError(f"Weighting method changed for {compound_id}.")
        if weighting_record["source_partition"] != "development":
            raise ValueError(f"Weighting partition changed for {compound_id}.")

        ensemble_path = (
            Path(conformer_record["audit_dir"]).resolve() / "crest_conformers.xyz"
        )
        if sha256_file(ensemble_path) != conformer_record["ensemble_sha256"]:
            raise ValueError(f"CREST ensemble hash changed for {compound_id}.")
        if weighting_record["ensemble_sha256"] != conformer_record["ensemble_sha256"]:
            raise ValueError(f"Weighting/conformer ensemble mismatch: {compound_id}.")

        mol2_path = base_work_dir / candidate["mol2_relative_path"]
        if sha256_file(mol2_path) != case["mol2_sha256"]:
            raise ValueError(f"Frozen MOL2 hash changed for {compound_id}.")
        atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
        symbols = list(atoms.get_chemical_symbols())
        numbers = [int(value) for value in atoms.get_atomic_numbers()]
        if not set(numbers).issubset(protocol["selection"]["common_atomic_numbers"]):
            raise ValueError(f"Selected case left the common domain: {compound_id}.")
        frames = _read_xyz_ensemble(ensemble_path, symbols)
        if len(frames) != conformer_record["conformer_count"]:
            raise ValueError(f"Conformer count changed for {compound_id}.")
        frames, union = reference_geometry_union(
            frames,
            atoms.get_positions(),
            symbols,
            threshold_angstrom=protocol["estimator"][
                "reference_geometry_rmsd_dedup_angstrom"
            ],
        )
        if union != weighting_record["reference_geometry_union"]:
            raise ValueError(f"Reference-geometry union changed for {compound_id}.")
        positions = np.asarray(
            [frame["positions_angstrom"] for frame in frames],
            dtype=np.float64,
        )
        solvent = [
            float(value) for value in weighting_record["solvent_correction_kcal_mol"]
        ]
        mace_energy = [
            float(value) for value in weighting_record["mlip_energy_hartree"]
        ]
        if len(positions) != len(solvent) or len(positions) != len(mace_energy):
            raise ValueError(f"State-array length mismatch for {compound_id}.")
        if not np.isfinite(positions).all() or not np.isfinite(solvent).all():
            raise ValueError(f"Non-finite state data for {compound_id}.")

        state_path = state_dir / f"{compound_id}.npy"
        _write_npy_atomic(state_path, positions)
        reference_index = int(union["reference_index"])
        cases.append(
            {
                "compound_id": compound_id,
                "name": case["name"],
                "flexibility_bin": case["flexibility_bin"],
                "elements": case["elements"],
                "atomic_numbers": numbers,
                "state_count": int(len(positions)),
                "state_file": _relative_to_repository(state_path),
                "state_file_sha256": sha256_file(state_path),
                "state_shape": list(positions.shape),
                "solvent_correction_kcal_mol": solvent,
                "solvent_correction_sha256": _array_payload_sha256(solvent),
                "reference_geometry_union": union,
                "reference_endpoint_kcal_mol": solvent[reference_index],
                "historical_mace_energy_hartree": mace_energy,
                "historical_mace_energy_sha256": _array_payload_sha256(mace_energy),
                "source": {
                    "conformer_record": _relative_to_repository(
                        conformer_record_path
                    ),
                    "conformer_record_sha256": sha256_file(conformer_record_path),
                    "weighting_record": _relative_to_repository(
                        weighting_record_path
                    ),
                    "weighting_record_sha256": sha256_file(weighting_record_path),
                    "crest_ensemble": _relative_to_repository(ensemble_path),
                    "crest_ensemble_sha256": sha256_file(ensemble_path),
                    "mol2": _relative_to_repository(mol2_path),
                    "mol2_sha256": sha256_file(mol2_path),
                },
            }
        )
        total_states += len(positions)

    if total_states != protocol["selection"]["expected_selected_state_count"]:
        raise ValueError(
            f"Selected state count changed: observed {total_states}."
        )
    manifest = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-discrete-conformer-source-manifest",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "selection": {
            **protocol["selection"],
            "observed_selected_case_count": len(cases),
            "observed_selected_state_count": total_states,
            "observed_exclusions": excluded,
        },
        "source_evidence": {
            "conformer_protocol": _relative_to_repository(
                conformer_protocol_path
            ),
            "conformer_protocol_sha256": sha256_file(conformer_protocol_path),
            "conformer_summary": _relative_to_repository(
                conformer_summary_path
            ),
            "conformer_summary_sha256": sha256_file(conformer_summary_path),
            "weighting_protocol": _relative_to_repository(
                weighting_protocol_path
            ),
            "weighting_protocol_sha256": sha256_file(weighting_protocol_path),
            "weighting_summary": _relative_to_repository(
                weighting_summary_path
            ),
            "weighting_summary_sha256": sha256_file(weighting_summary_path),
            "prepared_input": _relative_to_repository(prepared_path),
            "prepared_input_sha256": sha256_file(prepared_path),
            "reference_union_script": _relative_to_repository(
                SCRIPT_DIR / "run_mlip_conformer_weighting.py"
            ),
            "reference_union_script_sha256": sha256_file(
                SCRIPT_DIR / "run_mlip_conformer_weighting.py"
            ),
        },
        "models": protocol["models"],
        "solvent_endpoint": protocol["solvent_endpoint"],
        "cases": cases,
        "route": protocol["route"],
        "label_boundary": protocol["label_boundary"],
        "claim_boundary": protocol["claim_boundary"],
        "command_provenance": command_provenance(
            __file__,
            _command_arguments(args),
            repository_root=REPOSITORY_ROOT,
        ),
    }
    if _contains_forbidden_label_key(manifest):
        raise ValueError("Prepared source manifest contains a forbidden label key.")
    seal_artifact(manifest)
    write_json_atomic(args.output, manifest)
    print(
        f"Wrote {len(cases)} cases/{total_states} states to "
        f"{Path(args.output).resolve()}."
    )


def _load_manifest(
    path: Path,
    *,
    protocol: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    manifest = load_json(path)
    recorded = manifest.get("content_sha256")
    payload = dict(manifest)
    payload.pop("content_sha256", None)
    seal_artifact(payload)
    if payload["content_sha256"] != recorded:
        raise ValueError("Source manifest self-hash is invalid.")
    if manifest.get("protocol_fingerprint") != fingerprint:
        raise ValueError("Source manifest protocol fingerprint changed.")
    if manifest["selection"]["observed_selected_case_count"] != protocol[
        "selection"
    ]["expected_selected_case_count"]:
        raise ValueError("Source manifest case count changed.")
    if manifest["selection"]["observed_selected_state_count"] != protocol[
        "selection"
    ]["expected_selected_state_count"]:
        raise ValueError("Source manifest state count changed.")
    if _contains_forbidden_label_key(manifest):
        raise ValueError("Source manifest contains a forbidden label key.")
    return manifest


def _load_case_positions(case: dict[str, Any]) -> np.ndarray:
    path = REPOSITORY_ROOT / case["state_file"]
    if sha256_file(path) != case["state_file_sha256"]:
        raise ValueError(f"State archive hash changed for {case['compound_id']}.")
    positions = np.load(path, allow_pickle=False)
    if list(positions.shape) != case["state_shape"]:
        raise ValueError(f"State archive shape changed for {case['compound_id']}.")
    if positions.dtype != np.float64 or not np.isfinite(positions).all():
        raise ValueError(f"Invalid state archive for {case['compound_id']}.")
    return positions


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _compact_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "delta_g_discrete_kcal_mol",
        "gas_relative_energy_kcal_mol",
        "gas_weights",
        "solution_weights",
        "gas_effective_conformer_count",
        "solution_effective_conformer_count",
        "gas_maximum_weight",
        "solution_maximum_weight",
        "gas_dominant_state_index",
        "solution_dominant_state_index",
        "distribution_overlap",
        "total_variation_distance",
        "bhattacharyya_coefficient",
        "identities",
        "gates",
        "route_contract",
        "claim_boundary",
    ]
    return {key: analysis[key] for key in keys}


def _model_environment(
    *,
    model: dict[str, Any],
    calculator,
    device,
    load_seconds: float,
) -> dict[str, Any]:
    import torch

    cutoff = getattr(calculator, "r_max", getattr(calculator, "cutoff", None))
    dtype = getattr(calculator, "dtype", None)
    calculator_source = Path(inspect.getsourcefile(type(calculator))).resolve()
    set_calculator_source = (
        REPOSITORY_ROOT / "maple/function/calculator/set_calculator.py"
    )
    calculator_base_source = (
        REPOSITORY_ROOT / "maple/function/calculator/calculator_base.py"
    )
    discrete_core_source = (
        REPOSITORY_ROOT
        / "maple/function/free_energy/discrete_conformers.py"
    )
    return {
        "name": model["name"],
        "checkpoint_filename": model["checkpoint_filename"],
        "checkpoint_sha256": sha256_file(_checkpoint_path(model)),
        "checkpoint_size_bytes": _checkpoint_path(model).stat().st_size,
        "declared_atomic_numbers": model["supported_atomic_numbers"],
        "observed_atomic_numbers": [
            int(value) for value in calculator.atomic_numbers
        ],
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "calculator_dtype": str(dtype) if dtype is not None else None,
        "calculator_cutoff_angstrom": (
            float(cutoff) if cutoff is not None else None
        ),
        "calculator_class": (
            f"{type(calculator).__module__}.{type(calculator).__qualname__}"
        ),
        "calculator_source": _relative_to_repository(calculator_source),
        "calculator_source_sha256": sha256_file(calculator_source),
        "set_calculator_source": _relative_to_repository(set_calculator_source),
        "set_calculator_source_sha256": sha256_file(set_calculator_source),
        "calculator_base_source": _relative_to_repository(
            calculator_base_source
        ),
        "calculator_base_source_sha256": sha256_file(calculator_base_source),
        "discrete_core_source": _relative_to_repository(discrete_core_source),
        "discrete_core_source_sha256": sha256_file(discrete_core_source),
        "ase_version": importlib.metadata.version("ase"),
        "numpy_version": np.__version__,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "load_seconds": load_seconds,
    }


def _pairwise_comparisons(
    records: list[dict[str, Any]],
    model_names: list[str],
) -> dict[str, Any]:
    index = {
        (record["model"], record["compound_id"]): record for record in records
    }
    compound_ids = sorted({record["compound_id"] for record in records})
    output = {}
    for left, right in combinations(model_names, 2):
        cases = []
        for compound_id in compound_ids:
            left_record = index[(left, compound_id)]
            right_record = index[(right, compound_id)]
            left_analysis = left_record["analysis"]
            right_analysis = right_record["analysis"]
            gas_overlap = float(
                np.minimum(
                    left_analysis["gas_weights"],
                    right_analysis["gas_weights"],
                ).sum()
            )
            solution_overlap = float(
                np.minimum(
                    left_analysis["solution_weights"],
                    right_analysis["solution_weights"],
                ).sum()
            )
            cases.append(
                {
                    "compound_id": compound_id,
                    "delta_g_difference_kcal_mol": (
                        left_analysis["delta_g_discrete_kcal_mol"]
                        - right_analysis["delta_g_discrete_kcal_mol"]
                    ),
                    "absolute_delta_g_difference_kcal_mol": abs(
                        left_analysis["delta_g_discrete_kcal_mol"]
                        - right_analysis["delta_g_discrete_kcal_mol"]
                    ),
                    "gas_weight_overlap": gas_overlap,
                    "solution_weight_overlap": solution_overlap,
                    "gas_dominant_state_agrees": (
                        left_analysis["gas_dominant_state_index"]
                        == right_analysis["gas_dominant_state_index"]
                    ),
                    "solution_dominant_state_agrees": (
                        left_analysis["solution_dominant_state_index"]
                        == right_analysis["solution_dominant_state_index"]
                    ),
                }
            )
        output[f"{left}__{right}"] = {
            "case_count": len(cases),
            "absolute_delta_g_difference_kcal_mol": _summary(
                [row["absolute_delta_g_difference_kcal_mol"] for row in cases]
            ),
            "gas_weight_overlap": _summary(
                [row["gas_weight_overlap"] for row in cases]
            ),
            "solution_weight_overlap": _summary(
                [row["solution_weight_overlap"] for row in cases]
            ),
            "gas_dominant_state_agreement_count": sum(
                row["gas_dominant_state_agrees"] for row in cases
            ),
            "solution_dominant_state_agreement_count": sum(
                row["solution_dominant_state_agrees"] for row in cases
            ),
            "cases": cases,
        }
    return output


def run_energy(args: argparse.Namespace) -> None:
    import torch

    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(
        manifest_path,
        protocol=protocol,
        fingerprint=fingerprint,
    )
    device = torch.device(protocol["execution"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Protocol requires CUDA but CUDA is unavailable.")
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    first_case = manifest["cases"][0]
    first_positions = _load_case_positions(first_case)
    first_atoms = Atoms(
        numbers=first_case["atomic_numbers"],
        positions=first_positions[0],
    )
    first_atoms.info.update({"charge": 0, "mult": 1})

    records = []
    environments = {}
    repeat_count = int(protocol["execution"]["repeat_count"])
    estimator = protocol["estimator"]
    for model in protocol["models"]:
        name = model["name"]
        start = time.perf_counter()
        calculator = SetCalculator(
            device,
            name,
            str(work_dir / f"{name}.log"),
            atoms=first_atoms,
        ).set_calculator()
        load_seconds = time.perf_counter() - start
        observed_domain = [int(value) for value in calculator.atomic_numbers]
        if observed_domain != model["supported_atomic_numbers"]:
            raise ValueError(f"Checkpoint element domain changed for {name}.")
        environments[name] = _model_environment(
            model=model,
            calculator=calculator,
            device=device,
            load_seconds=load_seconds,
        )

        for case in manifest["cases"]:
            positions = _load_case_positions(case)
            numbers = [int(value) for value in case["atomic_numbers"]]
            if not set(numbers).issubset(observed_domain):
                raise ValueError(
                    f"Case {case['compound_id']} is outside {name}'s domain."
                )
            solvent = [float(value) for value in case["solvent_correction_kcal_mol"]]
            if _array_payload_sha256(solvent) != case[
                "solvent_correction_sha256"
            ]:
                raise ValueError(
                    f"Solvent correction hash changed for {case['compound_id']}."
                )
            atoms = Atoms(numbers=numbers, positions=positions[0])
            atoms.info.update({"charge": 0, "mult": 1})
            atoms.calc = calculator
            energies_by_repeat = []
            seconds_by_repeat = []
            for _repeat in range(repeat_count):
                energies = []
                start = time.perf_counter()
                for state in positions:
                    atoms.set_positions(state)
                    energy = float(atoms.get_potential_energy())
                    if not math.isfinite(energy):
                        raise ValueError(
                            f"{name} returned non-finite energy for "
                            f"{case['compound_id']}."
                        )
                    energies.append(energy)
                seconds_by_repeat.append(time.perf_counter() - start)
                energies_by_repeat.append(energies)
            relative_by_repeat = [
                (
                    np.asarray(values, dtype=np.float64) - min(values)
                ) * KCAL_PER_HARTREE
                for values in energies_by_repeat
            ]
            maximum_repeat_difference = max(
                float(np.max(np.abs(values - relative_by_repeat[0])))
                for values in relative_by_repeat[1:]
            )
            historical_parity = None
            if name == "maceoff23m":
                historical = np.asarray(
                    case["historical_mace_energy_hartree"],
                    dtype=np.float64,
                )
                historical_relative = (
                    historical - float(np.min(historical))
                ) * KCAL_PER_HARTREE
                historical_parity = float(
                    np.max(np.abs(relative_by_repeat[0] - historical_relative))
                )
            analyses = [
                analyze_discrete_conformer_ensemble(
                    gas_energy_kcal_mol=relative,
                    solvent_correction_kcal_mol=solvent,
                    temperature_kelvin=estimator["temperature_kelvin"],
                    minimum_effective_conformer_count=estimator[
                        "minimum_effective_conformer_count"
                    ],
                    maximum_dominant_weight=estimator[
                        "maximum_dominant_weight"
                    ],
                    minimum_distribution_overlap=estimator[
                        "minimum_distribution_overlap"
                    ],
                )
                for relative in relative_by_repeat
            ]
            delta_g_by_repeat = [
                analysis["delta_g_discrete_kcal_mol"] for analysis in analyses
            ]
            maximum_repeat_delta_g_difference = max(
                abs(value - delta_g_by_repeat[0])
                for value in delta_g_by_repeat[1:]
            )
            records.append(
                {
                    "model": name,
                    "compound_id": case["compound_id"],
                    "name": case["name"],
                    "flexibility_bin": case["flexibility_bin"],
                    "state_count": case["state_count"],
                    "state_file_sha256": case["state_file_sha256"],
                    "solvent_correction_sha256": case[
                        "solvent_correction_sha256"
                    ],
                    "reference_endpoint_kcal_mol": case[
                        "reference_endpoint_kcal_mol"
                    ],
                    "energy_hartree_by_repeat": energies_by_repeat,
                    "seconds_by_repeat": seconds_by_repeat,
                    "maximum_repeat_relative_energy_difference_kcal_mol": (
                        maximum_repeat_difference
                    ),
                    "delta_g_discrete_kcal_mol_by_repeat": delta_g_by_repeat,
                    "maximum_repeat_delta_g_difference_kcal_mol": (
                        maximum_repeat_delta_g_difference
                    ),
                    "historical_mace_relative_energy_difference_kcal_mol": (
                        historical_parity
                    ),
                    "analysis": _compact_analysis(analyses[0]),
                }
            )

    model_names = [model["name"] for model in protocol["models"]]
    model_summaries = {}
    for name in model_names:
        selected = [record for record in records if record["model"] == name]
        total_states = sum(record["state_count"] for record in selected)
        primary_seconds = sum(record["seconds_by_repeat"][0] for record in selected)
        model_summaries[name] = {
            "case_count": len(selected),
            "state_count": total_states,
            "weight_diagnostic_pass_count": sum(
                record["analysis"]["gates"]["ensemble_diagnostic_passed"]
                for record in selected
            ),
            "delta_g_discrete_kcal_mol": _summary(
                [
                    record["analysis"]["delta_g_discrete_kcal_mol"]
                    for record in selected
                ]
            ),
            "distribution_overlap": _summary(
                [record["analysis"]["distribution_overlap"] for record in selected]
            ),
            "maximum_repeat_relative_energy_difference_kcal_mol": max(
                record[
                    "maximum_repeat_relative_energy_difference_kcal_mol"
                ]
                for record in selected
            ),
            "maximum_repeat_delta_g_difference_kcal_mol": max(
                record["maximum_repeat_delta_g_difference_kcal_mol"]
                for record in selected
            ),
            "primary_evaluation_seconds": primary_seconds,
            "primary_states_per_second": total_states / primary_seconds,
        }
    sensitivity_cases = []
    record_index = {
        (record["model"], record["compound_id"]): record for record in records
    }
    for case in manifest["cases"]:
        values = {
            name: record_index[(name, case["compound_id"])]["analysis"][
                "delta_g_discrete_kcal_mol"
            ]
            for name in model_names
        }
        sensitivity_cases.append(
            {
                "compound_id": case["compound_id"],
                "by_model_kcal_mol": values,
                "model_range_kcal_mol": max(values.values()) - min(values.values()),
            }
        )
    repeat_limit = protocol["execution"][
        "maximum_repeat_relative_energy_difference_kcal_mol"
    ]
    mace_limit = protocol["execution"][
        "maximum_historical_mace_relative_energy_difference_kcal_mol"
    ]
    mace_parity = max(
        record["historical_mace_relative_energy_difference_kcal_mol"]
        for record in records
        if record["model"] == "maceoff23m"
    )
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-discrete-conformer-energy",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "source_manifest": _relative_to_repository(manifest_path),
        "source_manifest_file_sha256": sha256_file(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "route": protocol["route"],
        "selection": manifest["selection"],
        "solvent_endpoint": protocol["solvent_endpoint"],
        "estimator": protocol["estimator"],
        "model_environments": environments,
        "model_summaries": model_summaries,
        "pairwise_model_comparisons": _pairwise_comparisons(records, model_names),
        "model_sensitivity": {
            "case_range_kcal_mol": _summary(
                [row["model_range_kcal_mol"] for row in sensitivity_cases]
            ),
            "cases": sensitivity_cases,
        },
        "records": records,
        "engineering_gates": {
            "all_model_case_records_present": (
                len(records) == len(model_names) * len(manifest["cases"])
            ),
            "all_repeat_differences_within_limit": all(
                record[
                    "maximum_repeat_relative_energy_difference_kcal_mol"
                ]
                <= repeat_limit
                for record in records
            ),
            "historical_mace_relative_energy_parity_within_limit": (
                mace_parity <= mace_limit
            ),
            "maximum_observed_repeat_difference_kcal_mol": max(
                record[
                    "maximum_repeat_relative_energy_difference_kcal_mol"
                ]
                for record in records
            ),
            "maximum_observed_repeat_delta_g_difference_kcal_mol": max(
                record["maximum_repeat_delta_g_difference_kcal_mol"]
                for record in records
            ),
            "maximum_historical_mace_relative_energy_difference_kcal_mol": (
                mace_parity
            ),
            "repeat_limit_kcal_mol": repeat_limit,
            "historical_mace_parity_limit_kcal_mol": mace_limit,
            "interpretation": (
                "The predeclared 1e-6 kcal/mol relative-energy repeat gate is "
                "retained even when float32 CUDA backends miss it; no threshold "
                "is relaxed after observing the run."
            ),
        },
        "label_boundary": protocol["label_boundary"],
        "claim_boundary": protocol["claim_boundary"],
        "promotion_allowed": False,
        "command_provenance": command_provenance(
            __file__,
            _command_arguments(args),
            repository_root=REPOSITORY_ROOT,
            environment_variables=(
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "PYTORCH_CUDA_ALLOC_CONF",
            ),
        ),
    }
    if _contains_forbidden_label_key(artifact):
        raise ValueError("Energy artifact contains a forbidden label key.")
    seal_artifact(artifact)
    write_json_atomic(args.output, artifact)
    print(
        f"Wrote {len(records)} model/case records to "
        f"{Path(args.output).resolve()}."
    )


def _error_metrics(errors: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(errors)),
        "mse_kcal_mol": float(np.mean(errors)),
        "mae_kcal_mol": float(np.mean(np.abs(errors))),
        "rmse_kcal_mol": float(np.sqrt(np.mean(np.square(errors)))),
        "maximum_absolute_error_kcal_mol": float(np.max(np.abs(errors))),
    }


def _paired_score(
    baseline_errors: np.ndarray,
    candidate_errors: np.ndarray,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    gain = np.abs(baseline_errors) - np.abs(candidate_errors)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(gain),
        size=(resamples, len(gain)),
    )
    bootstrap = np.mean(gain[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    outcomes = {
        "improved": int(np.sum(gain > 1.0e-12)),
        "unchanged": int(np.sum(np.abs(gain) <= 1.0e-12)),
        "worsened": int(np.sum(gain < -1.0e-12)),
    }
    return {
        "mean_mae_gain_kcal_mol": float(np.mean(gain)),
        "bootstrap_ci_kcal_mol": [
            float(np.quantile(bootstrap, alpha)),
            float(np.quantile(bootstrap, 1.0 - alpha)),
        ],
        "case_outcomes": outcomes,
    }


def score(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    energy_path = Path(args.energy_artifact).resolve()
    energy = load_json(energy_path)
    recorded = energy.get("content_sha256")
    payload = dict(energy)
    payload.pop("content_sha256", None)
    seal_artifact(payload)
    if payload["content_sha256"] != recorded:
        raise ValueError("Energy artifact self-hash is invalid.")
    if energy.get("protocol_fingerprint") != fingerprint:
        raise ValueError("Energy artifact protocol fingerprint changed.")
    if energy.get("promotion_allowed") is not False:
        raise ValueError("Energy artifact unexpectedly permits promotion.")

    prepared_path = Path(args.base_work_dir).resolve() / "prepared.json"
    prepared = load_json(prepared_path)
    candidates = {
        candidate["compound_id"]: candidate for candidate in prepared["candidates"]
    }
    model_names = [model["name"] for model in protocol["models"]]
    record_index = {
        (record["model"], record["compound_id"]): record
        for record in energy["records"]
    }
    compound_ids = [
        record["compound_id"]
        for record in energy["records"]
        if record["model"] == model_names[0]
    ]
    labels = np.asarray(
        [
            float(candidates[compound_id]["experimental_kcal_mol"])
            for compound_id in compound_ids
        ],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [
            record_index[(model_names[0], compound_id)][
                "reference_endpoint_kcal_mol"
            ]
            for compound_id in compound_ids
        ],
        dtype=np.float64,
    )
    baseline_errors = baseline - labels
    settings = protocol["scoring"]
    models = {}
    per_case = []
    for model_index, name in enumerate(model_names):
        predicted = np.asarray(
            [
                record_index[(name, compound_id)]["analysis"][
                    "delta_g_discrete_kcal_mol"
                ]
                for compound_id in compound_ids
            ],
            dtype=np.float64,
        )
        errors = predicted - labels
        models[name] = {
            "metrics": _error_metrics(errors),
            "paired_vs_fixed_geometry": _paired_score(
                baseline_errors,
                errors,
                resamples=settings["bootstrap_resamples"],
                confidence=settings["bootstrap_confidence"],
                seed=settings["bootstrap_seed"] + model_index,
            ),
        }
        for compound_id, label, baseline_value, predicted_value in zip(
            compound_ids,
            labels,
            baseline,
            predicted,
            strict=True,
        ):
            per_case.append(
                {
                    "model": name,
                    "compound_id": compound_id,
                    "experimental_kcal_mol": float(label),
                    "fixed_geometry_kcal_mol": float(baseline_value),
                    "discrete_conformer_kcal_mol": float(predicted_value),
                    "fixed_geometry_error_kcal_mol": float(
                        baseline_value - label
                    ),
                    "discrete_conformer_error_kcal_mol": float(
                        predicted_value - label
                    ),
                }
            )
    score_artifact = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-discrete-conformer-score",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "energy_artifact": _relative_to_repository(energy_path),
        "energy_artifact_file_sha256": sha256_file(energy_path),
        "energy_artifact_content_sha256": energy["content_sha256"],
        "prepared_input": _relative_to_repository(prepared_path),
        "prepared_input_sha256": sha256_file(prepared_path),
        "case_count": len(compound_ids),
        "fixed_geometry": _error_metrics(baseline_errors),
        "models": models,
        "cases": per_case,
        "label_boundary": {
            "energy_artifact_sealed_before_scoring": True,
            "selection_used_labels": False,
            "scoring_partition": "development",
            "independent_confirmation": False,
            "model_selected_or_promoted": False,
        },
        "claim_boundary": {
            "accuracy_certification": False,
            "public_solvfe_eligible": False,
            "reason": (
                "The label-exposed development score compares frozen models; "
                "it is not independent confirmation and cannot select a residual "
                "or chemistry-specific model."
            ),
        },
        "promotion_allowed": False,
        "command_provenance": command_provenance(
            __file__,
            _command_arguments(args),
            repository_root=REPOSITORY_ROOT,
        ),
    }
    seal_artifact(score_artifact)
    write_json_atomic(args.output, score_artifact)
    print(f"Wrote development score to {Path(args.output).resolve()}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="extract a label-free common-domain state manifest",
    )
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--conformer-record-dir", required=True)
    prepare_parser.add_argument("--weighting-record-dir", required=True)
    prepare_parser.add_argument("--base-work-dir", required=True)
    prepare_parser.add_argument("--raw-dir", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.set_defaults(handler=prepare)

    run_parser = subparsers.add_parser(
        "run",
        help="evaluate every pinned MLIP on the common state set",
    )
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--work-dir", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.set_defaults(handler=run_energy)

    score_parser = subparsers.add_parser(
        "score",
        help="score a sealed energy artifact on development labels",
    )
    score_parser.add_argument("--protocol", required=True)
    score_parser.add_argument("--energy-artifact", required=True)
    score_parser.add_argument("--base-work-dir", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.set_defaults(handler=score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
