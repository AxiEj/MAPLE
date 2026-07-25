#!/usr/bin/env python3
"""Run a short multi-MLIP, force-consistent OBC-II/ACE TI diagnostic."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_core import (  # noqa: E402
    artifact_content_sha256,
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_route1_task_matrix import read_xyz_frames  # noqa: E402

PROTOCOL_ID = "maple-route1-mlip-obc2-force-ti-v1"
KCAL_PER_HARTREE = 627.5094740631
K_B_KCAL_PER_MOL_K = 0.00198720425864083


def _simpson_weights(lambdas: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(lambdas), dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or len(values) % 2 == 0:
        raise ValueError("Composite Simpson integration needs an odd number of windows.")
    spacing = np.diff(values)
    if (
        not np.isfinite(values).all()
        or values[0] != 0.0
        or values[-1] != 1.0
        or np.any(spacing <= 0.0)
        or not np.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("TI lambda windows must uniformly span [0, 1].")
    weights = np.ones(len(values), dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * spacing[0] / 3.0


def composite_simpson(lambdas: Iterable[float], means: Iterable[float]) -> float:
    x = np.asarray(list(lambdas), dtype=np.float64)
    y = np.asarray(list(means), dtype=np.float64)
    if x.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("TI means must be finite and match the lambda schedule.")
    return float(np.dot(_simpson_weights(x), y))


def endpoint_fep(
    solvent_kcal_mol: Iterable[float], temperature_kelvin: float
) -> dict[str, float | int]:
    values = np.asarray(list(solvent_kcal_mol), dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not math.isfinite(temperature_kelvin)
        or temperature_kelvin <= 0.0
    ):
        raise ValueError("Endpoint FEP requires finite samples and temperature.")
    beta = 1.0 / (K_B_KCAL_PER_MOL_K * float(temperature_kelvin))
    logits = -beta * values
    maximum = float(np.max(logits))
    shifted = np.exp(logits - maximum)
    log_mean_weight = maximum + math.log(float(np.mean(shifted)))
    normalized = shifted / float(np.sum(shifted))
    ess = float(1.0 / np.dot(normalized, normalized))
    return {
        "sample_count": int(len(values)),
        "delta_g_kcal_mol": float(-log_mean_weight / beta),
        "effective_sample_count": ess,
        "effective_sample_fraction": ess / len(values),
    }


def block_standard_error(values: Iterable[float]) -> float:
    samples = np.asarray(list(values), dtype=np.float64)
    if len(samples) < 4 or not np.isfinite(samples).all():
        return float("nan")
    block_count = min(8, max(2, len(samples) // 10))
    usable = samples[: (len(samples) // block_count) * block_count]
    blocks = usable.reshape(block_count, -1).mean(axis=1)
    return float(blocks.std(ddof=1) / math.sqrt(block_count))


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only OBC-II TI protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected OBC-II TI protocol ID.")
    boundary = protocol.get("execution_boundary", {})
    if boundary.get("sampling_phase_reads_experimental_labels") is not False:
        raise ValueError("Sampling must be explicitly label-blind.")
    if boundary.get("no_experimental_fit_or_residual_model") is not True:
        raise ValueError("Residual fits must be prohibited.")
    if boundary.get("promotion_allowed") is not False:
        raise ValueError("The short TI diagnostic cannot be promotable.")
    payload = json.dumps(protocol, sort_keys=True).lower()
    if "experimental_kcal_mol" in payload or "experimental_value" in payload:
        raise ValueError("The sampling protocol contains an experimental label.")
    _simpson_weights(protocol["sampling"]["lambda_windows"])
    expected_samples = (
        protocol["sampling"]["production_steps_per_window"]
        // protocol["sampling"]["sample_stride_steps"]
    )
    if (
        protocol["sampling"]["equilibration_steps_per_window"] < 1
        or protocol["sampling"]["production_steps_per_window"] < 1
        or protocol["sampling"]["sample_stride_steps"] < 1
        or expected_samples
        != protocol["diagnostic_gates"]["expected_production_samples_per_window"]
    ):
        raise ValueError("TI sampling counts do not match the frozen gate.")
    ids = [case["compound_id"] for case in protocol["cases"]]
    models = [model["name"] for model in protocol["models"]]
    if len(ids) != len(set(ids)) or len(models) != len(set(models)):
        raise ValueError("TI case IDs and model names must be unique.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


class LambdaImplicitCalculator(Calculator):
    """Scale one fixed-charge solvent correction without changing the gas MLIP."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, gas_calculator, correction, coupling: float):
        super().__init__()
        if not 0.0 <= float(coupling) <= 1.0:
            raise ValueError("The implicit-solvent coupling must be in [0, 1].")
        self.gas_calculator = gas_calculator
        self.correction = correction
        self.coupling = float(coupling)

    def components(self, atoms, *, need_forces: bool) -> dict[str, Any]:
        properties = ["energy", "forces"] if need_forces else ["energy"]
        self.gas_calculator.calculate(
            atoms,
            properties=properties,
            system_changes=all_changes,
        )
        gas_energy = float(self.gas_calculator.results["energy"])
        gas_forces = (
            np.asarray(self.gas_calculator.results["forces"], dtype=np.float64)
            if need_forces
            else None
        )
        solvent = self.correction.evaluate(atoms, need_forces=need_forces)
        solvent_energy = float(solvent.energy_hartree)
        solvent_forces = (
            np.asarray(solvent.forces_hartree_per_angstrom, dtype=np.float64)
            if need_forces
            else None
        )
        if need_forces and solvent_forces is None:
            raise NotImplementedError("The selected solvent endpoint has no forces.")
        return {
            "gas_energy_hartree": gas_energy,
            "solvent_energy_hartree": solvent_energy,
            "energy_hartree": gas_energy + self.coupling * solvent_energy,
            "gas_forces_hartree_per_angstrom": gas_forces,
            "solvent_forces_hartree_per_angstrom": solvent_forces,
            "forces_hartree_per_angstrom": (
                None
                if not need_forces
                else gas_forces + self.coupling * solvent_forces
            ),
            "solvent_components_hartree": dict(solvent.components_hartree),
        }

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        need_forces = properties is None or "forces" in properties
        values = self.components(atoms, need_forces=need_forces)
        self.results = {
            "energy": values["energy_hartree"],
            "free_energy": values["energy_hartree"],
        }
        if need_forces:
            self.results["forces"] = values["forces_hartree_per_angstrom"]


def _torch_device(value: str):
    import torch

    text = str(value).strip().lower()
    if text.startswith("gpu"):
        suffix = text[3:] or "0"
        text = f"cuda:{suffix}"
    return torch.device(text)


def _load_gas_calculator(model: dict[str, Any], device, first_atoms, output: Path):
    from maple.function.calculator.set_calculator import SetCalculator

    checkpoint = (
        REPOSITORY_ROOT / "maple/function/calculator/model" / f"{model['name']}.pt"
    )
    if sha256_file(checkpoint) != model["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash changed for {model['name']}.")
    calculator = SetCalculator(
        device,
        model["name"],
        str(output),
        atoms=first_atoms,
    ).set_calculator()
    return calculator, checkpoint


def _source_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["compound_id"]: record for record in manifest["records"]}


def _load_case_atoms(case_dir: Path, source_record: dict[str, Any]):
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    mol2_path = case_dir / "m.mol2"
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1, validate_charge=True)
    expected = np.asarray(source_record["am1bcc_charges_e"], dtype=np.float64)
    observed = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
    if observed.shape != expected.shape or np.max(np.abs(observed - expected)) > 1.0e-10:
        raise ValueError("Normalized MOL2 does not match the AM1-BCC manifest.")
    return atoms, mol2_path


def _sample_window(
    *,
    atoms,
    gas_calculator,
    correction,
    coupling: float,
    output: Path,
    parameters: dict[str, Any],
    seed: int,
    lambda_values: np.ndarray,
) -> tuple[dict[str, Any], Any]:
    from maple.function.dispatcher.md.ensemble.nvt import NVT

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    wrapper = LambdaImplicitCalculator(gas_calculator, correction, coupling)
    atoms.calc = wrapper
    total_steps = (
        parameters["equilibration_steps_per_window"]
        + parameters["production_steps_per_window"]
    )
    stride = parameters["sample_stride_steps"]
    NVT(
        output=str(output),
        atoms=atoms,
        paras={
            "steps": total_steps,
            "timestep": parameters["timestep_fs"],
            "temperature": parameters["temperature_kelvin"],
            "thermostat": "langevin",
            "friction": parameters["friction_per_fs"],
            "traj_every": stride,
            "log_every": max(stride, parameters["production_steps_per_window"] // 4),
            "rst_every": total_steps + 1,
            "remove_com_every": 0,
            "remove_angular_every": 0,
            "random_seed": seed,
            "verbose": 0,
        },
    ).run()
    trajectory = output.with_name(f"{output.stem}_md_traj.xyz")
    frames = read_xyz_frames(trajectory)
    discarded = parameters["equilibration_steps_per_window"] // stride
    production = frames[discarded:]
    expected = parameters["production_steps_per_window"] // stride
    if len(production) != expected:
        raise ValueError(
            f"Expected {expected} production frames, observed {len(production)}."
        )

    sample_records = []
    for sample_index, frame in enumerate(production):
        sample_atoms = atoms.copy()
        sample_atoms.set_positions(frame["positions_angstrom"])
        values = wrapper.components(sample_atoms, need_forces=False)
        gas = float(values["gas_energy_hartree"])
        solvent = float(values["solvent_energy_hartree"])
        reduced = (
            gas + lambda_values * solvent
        ) * KCAL_PER_HARTREE / (
            K_B_KCAL_PER_MOL_K * parameters["temperature_kelvin"]
        )
        sample_records.append(
            {
                "sample_index": sample_index,
                "gas_energy_hartree": gas,
                "solvent_energy_hartree": solvent,
                "reduced_potentials": reduced.tolist(),
            }
        )
    solvent_kcal = np.asarray(
        [record["solvent_energy_hartree"] for record in sample_records]
    ) * KCAL_PER_HARTREE
    return (
        {
            "lambda": coupling,
            "seed": seed,
            "trajectory": str(trajectory),
            "trajectory_sha256": sha256_file(trajectory),
            "discarded_frame_count": discarded,
            "production_sample_count": len(sample_records),
            "mean_dudlambda_kcal_mol": float(solvent_kcal.mean()),
            "block_standard_error_kcal_mol": block_standard_error(solvent_kcal),
            "minimum_dudlambda_kcal_mol": float(solvent_kcal.min()),
            "maximum_dudlambda_kcal_mol": float(solvent_kcal.max()),
            "samples": sample_records,
        },
        atoms,
    )


def _run_chain(
    *,
    chain: dict[str, Any],
    case: dict[str, Any],
    atoms,
    gas_calculator,
    correction,
    work_dir: Path,
    parameters: dict[str, Any],
    lambda_values: np.ndarray,
) -> dict[str, Any]:
    order = (
        lambda_values
        if chain["lambda_order"] == "ascending"
        else lambda_values[::-1]
    )
    windows = []
    current = atoms.copy()
    for order_index, coupling in enumerate(order):
        seed = int(case["seed"] + chain["seed_offset"] + order_index)
        window, current = _sample_window(
            atoms=current,
            gas_calculator=gas_calculator,
            correction=correction,
            coupling=float(coupling),
            output=work_dir / f"lambda-{coupling:.2f}.out",
            parameters=parameters,
            seed=seed,
            lambda_values=lambda_values,
        )
        windows.append(window)
    ordered = sorted(windows, key=lambda window: window["lambda"])
    means = [window["mean_dudlambda_kcal_mol"] for window in ordered]
    ti = composite_simpson(lambda_values, means)
    gas_window = next(window for window in ordered if window["lambda"] == 0.0)
    return {
        "chain_id": chain["chain_id"],
        "lambda_order": chain["lambda_order"],
        "windows": windows,
        "ti_kcal_mol": ti,
        "endpoint_fep_from_gas_window": endpoint_fep(
            (
                sample["solvent_energy_hartree"] * KCAL_PER_HARTREE
                for sample in gas_window["samples"]
            ),
            parameters["temperature_kelvin"],
        ),
    }


def _label_record(label_root: Path, compound_id: str) -> dict[str, Any]:
    path = label_root / f"{compound_id}__am1bcc__obc2.json"
    record = load_json(path)
    if record.get("partition") != "development":
        raise ValueError("The TI diagnostic may read development labels only.")
    return record


def _metric_block(predicted: np.ndarray, experimental: np.ndarray) -> dict[str, Any]:
    error = predicted - experimental
    return {
        "n": int(len(error)),
        "mse_kcal_mol": float(error.mean()),
        "mae_kcal_mol": float(np.abs(error).mean()),
        "rmse_kcal_mol": float(np.sqrt(np.square(error).mean())),
        "maximum_absolute_error_kcal_mol": float(np.abs(error).max()),
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _load_reusable_record(
    path: Path,
    *,
    model: str,
    compound_id: str,
    protocol_fingerprint: str,
) -> dict[str, Any]:
    record = load_json(path)
    expected = record.get("content_sha256")
    if expected != artifact_content_sha256(record):
        raise ValueError(f"Reusable TI record has an invalid self-hash: {path}.")
    identity = (
        record.get("protocol_id"),
        record.get("protocol_fingerprint"),
        record.get("model"),
        record.get("compound_id"),
    )
    required = (PROTOCOL_ID, protocol_fingerprint, model, compound_id)
    if identity != required:
        raise ValueError(f"Reusable TI record has incompatible identity: {path}.")
    if "experimental_kcal_mol" in json.dumps(record, sort_keys=True).lower():
        raise ValueError(f"Reusable TI record contains an experimental label: {path}.")
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    manifest_path = protocol_path.parent / protocol["source_evidence"]["charge_manifest"]
    if sha256_file(manifest_path) != protocol["source_evidence"]["charge_manifest_sha256"]:
        raise ValueError("AM1-BCC source manifest hash changed.")
    manifest = load_json(manifest_path)
    source_by_id = _source_records(manifest)
    base_case_dir = Path(args.base_case_dir).resolve()
    label_root = Path(args.label_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = _torch_device(args.device)
    lambda_values = np.asarray(protocol["sampling"]["lambda_windows"], dtype=np.float64)
    selected_models = [
        model
        for model in protocol["models"]
        if not args.model or model["name"] in set(args.model)
    ]
    if not selected_models:
        raise ValueError("No frozen model was selected.")

    raw_records = {}
    checkpoint_records = {}
    reused_record_count = 0
    new_record_count = 0
    started = time.perf_counter()
    for model in selected_models:
        pending_cases = []
        for case in protocol["cases"]:
            record_path = (
                raw_dir / model["name"] / case["compound_id"] / "record.json"
            )
            if args.resume_existing and record_path.is_file():
                record = _load_reusable_record(
                    record_path,
                    model=model["name"],
                    compound_id=case["compound_id"],
                    protocol_fingerprint=fingerprint,
                )
                raw_records[(model["name"], case["compound_id"])] = {
                    "path": str(record_path),
                    "sha256": sha256_file(record_path),
                    "record": record,
                }
                reused_record_count += 1
            else:
                pending_cases.append(case)
        checkpoint = (
            REPOSITORY_ROOT
            / "maple/function/calculator/model"
            / f"{model['name']}.pt"
        )
        if sha256_file(checkpoint) != model["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash changed for {model['name']}.")
        checkpoint_records[model["name"]] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
        if not pending_cases:
            continue

        first_case = pending_cases[0]
        first_atoms, _ = _load_case_atoms(
            base_case_dir / first_case["compound_id"],
            source_by_id[first_case["compound_id"]],
        )
        gas_calculator, checkpoint = _load_gas_calculator(
            model,
            device,
            first_atoms,
            work_dir / f"{model['name']}-calculator.log",
        )
        for case in pending_cases:
            source_record = source_by_id[case["compound_id"]]
            atoms, mol2_path = _load_case_atoms(
                base_case_dir / case["compound_id"],
                source_record,
            )
            case_dir = raw_dir / model["name"] / case["compound_id"]
            correction = ImplicitSolvationCorrection(
                atoms,
                {
                    "source": "mol2",
                    "mode": "fixed",
                    "geometry": "keep",
                    "label": "am1bcc-frozen-manifest",
                },
                {
                    "implicit": "water",
                    "method": "gb",
                    "model": "obc2",
                    "nonpolar": "ace",
                    "platform": protocol["endpoint"]["openmm_platform"],
                    "experimental": True,
                },
                output=case_dir / "correction.out",
            )
            chains = [
                _run_chain(
                    chain=chain,
                    case=case,
                    atoms=atoms,
                    gas_calculator=gas_calculator,
                    correction=correction,
                    work_dir=case_dir / chain["chain_id"],
                    parameters=protocol["sampling"],
                    lambda_values=lambda_values,
                )
                for chain in protocol["sampling"]["chains"]
            ]
            ti_values = np.asarray(
                [chain["ti_kcal_mol"] for chain in chains], dtype=np.float64
            )
            ess_fractions = [
                chain["endpoint_fep_from_gas_window"]["effective_sample_fraction"]
                for chain in chains
            ]
            raw_record = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "protocol_fingerprint": fingerprint,
                "source_partition": "development",
                "model": model["name"],
                "compound_id": case["compound_id"],
                "name": case["name"],
                "flexibility_bin": case["flexibility_bin"],
                "mol2": str(mol2_path),
                "mol2_sha256": sha256_file(mol2_path),
                "charge_manifest_record_sha256": sha256_bytes(
                    canonical_json_bytes(source_record)
                ),
                "chains": chains,
                "ti_mean_kcal_mol": float(ti_values.mean()),
                "ti_chain_disagreement_kcal_mol": float(np.ptp(ti_values)),
                "minimum_endpoint_fep_ess_fraction": float(min(ess_fractions)),
                "finite_full_reduced_potential_matrix": all(
                    math.isfinite(value)
                    for chain in chains
                    for window in chain["windows"]
                    for sample in window["samples"]
                    for value in sample["reduced_potentials"]
                ),
                "promotion_allowed": False,
                "limitations": protocol["limitations"],
            }
            raw_record["diagnostic_checks"] = {
                "chain_ti_agreement": (
                    raw_record["ti_chain_disagreement_kcal_mol"]
                    <= protocol["diagnostic_gates"][
                        "maximum_chain_ti_disagreement_kcal_mol"
                    ]
                ),
                "endpoint_fep_ess": (
                    raw_record["minimum_endpoint_fep_ess_fraction"]
                    >= protocol["diagnostic_gates"][
                        "minimum_endpoint_fep_ess_fraction"
                    ]
                ),
                "finite_full_reduced_potential_matrix": raw_record[
                    "finite_full_reduced_potential_matrix"
                ],
                "equilibrated_sampling_proven": False,
                "mbar_overlap_proven": False,
            }
            raw_record["all_diagnostic_checks_pass"] = all(
                raw_record["diagnostic_checks"].values()
            )
            sealed = seal_artifact(raw_record)
            record_path = case_dir / "record.json"
            write_json_atomic(record_path, sealed)
            raw_records[(model["name"], case["compound_id"])] = {
                "path": str(record_path),
                "sha256": sha256_file(record_path),
                "record": sealed,
            }
            new_record_count += 1
        del gas_calculator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.publish_raw_dir:
        publish_dir = Path(args.publish_raw_dir).resolve()
        for key, raw in raw_records.items():
            model, compound_id = key
            published_record = copy.deepcopy(raw["record"])
            published_record["published_from_execution_content_sha256"] = raw[
                "record"
            ]["content_sha256"]

            source_mol2 = Path(published_record["mol2"])
            published_mol2 = publish_dir / "inputs" / f"{compound_id}.mol2"
            published_mol2.parent.mkdir(parents=True, exist_ok=True)
            if not published_mol2.is_file():
                shutil.copyfile(source_mol2, published_mol2)
            if sha256_file(published_mol2) != published_record["mol2_sha256"]:
                raise ValueError(
                    f"Published TI MOL2 hash mismatch: {published_mol2}."
                )
            published_record["mol2"] = _display_path(published_mol2)

            for chain in published_record["chains"]:
                for window in chain["windows"]:
                    source_trajectory = Path(window["trajectory"])
                    published_trajectory = (
                        publish_dir
                        / "trajectories"
                        / model
                        / compound_id
                        / chain["chain_id"]
                        / f"lambda-{window['lambda']:.2f}.xyz"
                    )
                    published_trajectory.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_trajectory, published_trajectory)
                    if (
                        sha256_file(published_trajectory)
                        != window["trajectory_sha256"]
                    ):
                        raise ValueError(
                            "Published TI trajectory hash mismatch: "
                            f"{published_trajectory}."
                        )
                    window["trajectory"] = _display_path(published_trajectory)

            published = publish_dir / model / f"{compound_id}.json"
            published.parent.mkdir(parents=True, exist_ok=True)
            seal_artifact(published_record)
            write_json_atomic(published, published_record)
            raw_records[key] = {
                "path": _display_path(published),
                "sha256": sha256_file(published),
                "record": published_record,
            }

    scored = {}
    for model in selected_models:
        for case in protocol["cases"]:
            key = (model["name"], case["compound_id"])
            raw = raw_records[key]["record"]
            label = _label_record(label_root, case["compound_id"])
            predicted = raw["ti_mean_kcal_mol"]
            experimental = float(label["experimental_kcal_mol"])
            scored[key] = {
                "model": model["name"],
                "compound_id": case["compound_id"],
                "name": case["name"],
                "predicted_ti_kcal_mol": predicted,
                "fixed_geometry_obc2_ace_kcal_mol": float(
                    label["predicted_kcal_mol"]
                ),
                "experimental_kcal_mol": experimental,
                "ti_signed_error_kcal_mol": predicted - experimental,
                "ti_absolute_error_kcal_mol": abs(predicted - experimental),
                "fixed_geometry_absolute_error_kcal_mol": abs(
                    float(label["predicted_kcal_mol"]) - experimental
                ),
                "raw_record_path": raw_records[key]["path"],
                "raw_record_sha256": raw_records[key]["sha256"],
                "all_diagnostic_checks_pass": raw["all_diagnostic_checks_pass"],
            }

    model_summaries = {}
    for model in selected_models:
        rows = [
            scored[(model["name"], case["compound_id"])]
            for case in protocol["cases"]
        ]
        experimental = np.asarray(
            [row["experimental_kcal_mol"] for row in rows], dtype=np.float64
        )
        ti = np.asarray(
            [row["predicted_ti_kcal_mol"] for row in rows], dtype=np.float64
        )
        fixed = np.asarray(
            [row["fixed_geometry_obc2_ace_kcal_mol"] for row in rows],
            dtype=np.float64,
        )
        model_summaries[model["name"]] = {
            "case_count": len(rows),
            "ti": _metric_block(ti, experimental),
            "fixed_geometry": _metric_block(fixed, experimental),
            "ti_minus_fixed_geometry_mae_kcal_mol": (
                _metric_block(ti, experimental)["mae_kcal_mol"]
                - _metric_block(fixed, experimental)["mae_kcal_mol"]
            ),
            "all_diagnostic_checks_pass": all(
                row["all_diagnostic_checks_pass"] for row in rows
            ),
        }

    result = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-force-consistent-obc2-ti-development",
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "command_provenance": command_provenance(
            __file__,
            vars(args),
            repository_root=REPOSITORY_ROOT,
            environment_variables=(
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            ),
        ),
        "route": protocol["route"],
        "label_use_boundary": (
            "All label-free per-model/per-case records were sealed and written "
            "before development labels were read for this summary."
        ),
        "standard_state": "gas 1 M to ideal-dilute solution 1 M",
        "estimator": {
            "reported": "composite-Simpson TI",
            "production_target": "multi-window MBAR with TI cross-check",
            "mbar_executed": False,
            "reason": (
                "The upstream pymbar dependency is not installed; MAPLE does not "
                "ship a handwritten MBAR substitute."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "openmm": importlib.metadata.version("openmm"),
            "ase": importlib.metadata.version("ase"),
            "requested_device": args.device,
            "resolved_device": str(device),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
        },
        "checkpoints": checkpoint_records,
        "case_results": list(scored.values()),
        "model_summaries": model_summaries,
        "execution": {
            "new_record_count": new_record_count,
            "reused_record_count": reused_record_count,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "cross_invocation_sampling_wall_time_reported": False,
        },
        "promotion_allowed": False,
        "all_diagnostic_checks_pass": all(
            summary["all_diagnostic_checks_pass"]
            for summary in model_summaries.values()
        ),
        "limitations": protocol["limitations"],
    }
    return seal_artifact(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--base-case-dir", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="gpu0")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--publish-raw-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
