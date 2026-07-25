"""Run one preregistered acetone + one-water Route A development replica.

This is a research driver, not the public ``#solvfe`` production workflow.
It exercises the real MACE-OMOL alchemical sampler and writes an immutable
sample set plus diagnostics.  The result is a fixed-n development diagnostic;
it omits the protocol-v2 packing and multi-occupancy terms and therefore is
not an absolute hydration free energy.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
from ase import Atoms, units

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maple.function.calculator.mace._mace_upstream_calculator import (
    MACEOMOLProvider,
)
from maple.function.dispatcher.solvfe.alchemy import (
    FragmentPartition,
    GaussianRepulsiveCore,
    ManyBodyInteractionEvaluator,
)
from maple.function.dispatcher.solvfe.analysis import diagnose_mbar
from maple.function.dispatcher.solvfe.moves import RigidBodyMoveConfig
from maple.function.dispatcher.solvfe.protocol import (
    RouteAProtocol,
    canonical_sha256,
    raw_sha256,
)
from maple.function.dispatcher.solvfe.sampling import (
    AlchemicalSampleSet,
    AlchemicalSchedule,
    AlchemicalWindowRunner,
    WindowRunConfig,
)
from maple.function.dispatcher.solvfe.shell import (
    FlatBottomSurfaceRestraint,
)


KCAL_PER_DIMENSIONLESS = 0.00198720425864083 * 298.15
ATOMIC_NUMBERS = (8, 6, 6, 6, 1, 1, 1, 1, 1, 1, 8, 1, 1)
SOLUTE_INDICES = tuple(range(10))
WATER_INDICES = (10, 11, 12)
RUNTIME_SOURCE_PATHS = (
    "examples/solvation/route_a/run_acetone_n1_replica.py",
    "maple/function/calculator/mace/_mace_upstream_calculator.py",
    "maple/function/dispatcher/solvfe/alchemy.py",
    "maple/function/dispatcher/solvfe/analysis.py",
    "maple/function/dispatcher/solvfe/moves.py",
    "maple/function/dispatcher/solvfe/sampling.py",
    "maple/function/dispatcher/solvfe/shell.py",
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_initial_positions(path: Path, frame_index: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        positions = np.asarray(archive["positions_angstrom"], dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != (13, 3):
        raise ValueError(
            "Initial sample archive must contain positions_angstrom with "
            "shape (frames, 13, 3)."
        )
    try:
        selected = positions[int(frame_index)]
    except IndexError as exc:
        raise ValueError(
            f"Initial frame index {frame_index} is outside {len(positions)} frames."
        ) from exc
    if not np.all(np.isfinite(selected)):
        raise ValueError("Initial coordinates contain non-finite values.")
    return selected.copy()


def _runtime_environment(device: str) -> dict[str, Any]:
    import torch

    cuda_requested = str(device).startswith("cuda")
    cuda_available = bool(torch.cuda.is_available())
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ase": importlib.metadata.version("ase"),
        "mace_torch": importlib.metadata.version("mace-torch"),
        "pymbar": importlib.metadata.version("pymbar"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_request": str(device),
        "cuda_available": cuda_available,
    }
    if cuda_requested and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available.")
    if cuda_requested:
        device_index = torch.cuda.current_device()
        environment.update(
            {
                "cuda_device_index": int(device_index),
                "cuda_device_name": torch.cuda.get_device_name(device_index),
                "cuda_device_capability": list(
                    torch.cuda.get_device_capability(device_index)
                ),
            }
        )
    return environment


def _snapshot_runtime_sources(
    output: Path,
    *,
    run_hash: str,
    source_hashes: dict[str, str],
) -> None:
    snapshot_root = output / "runtime-source"
    records: dict[str, Any] = {}
    for relative, expected in source_hashes.items():
        source = PROJECT_ROOT / relative
        if raw_sha256(source) != expected:
            raise RuntimeError(
                f"Runtime source changed before snapshot: {relative}."
            )
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied_hash = raw_sha256(target)
        if copied_hash != expected:
            raise RuntimeError(f"Runtime source copy failed: {relative}.")
        records[relative] = {
            "sha256": copied_hash,
            "bytes": target.stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "artifact_type": "route-a-runtime-source-snapshot",
        "run_hash": run_hash,
        "files": records,
    }
    manifest["artifact_sha256"] = canonical_sha256(manifest)
    _atomic_json(snapshot_root / "manifest.json", manifest)


def _half_sample_set(
    samples: AlchemicalSampleSet,
    *,
    second_half: bool,
) -> AlchemicalSampleSet:
    boundaries = np.concatenate(([0], np.cumsum(samples.N_k)))
    selected: list[int] = []
    counts: list[int] = []
    for state_index, count in enumerate(samples.N_k):
        half = count // 2
        if half == 0:
            raise ValueError("Half-trajectory analysis requires two samples/state.")
        start = int(boundaries[state_index])
        if second_half:
            indices = range(start + count - half, start + count)
        else:
            indices = range(start, start + half)
        selected.extend(indices)
        counts.append(half)
    index_array = np.asarray(selected, dtype=int)
    return AlchemicalSampleSet.create(
        schedule=samples.schedule,
        N_k=tuple(counts),
        frame_ids=tuple(samples.frame_ids[index] for index in selected),
        positions_angstrom=samples.positions_angstrom[index_array],
        basis_ev=samples.basis_ev[index_array],
        atom_list_hash=samples.atom_list_hash,
        restraint_hash=samples.restraint_hash,
        measure_id=samples.measure_id,
        boundary_conditions=samples.boundary_conditions,
    )


def _mixing_summary(samples: AlchemicalSampleSet) -> dict[str, Any]:
    first_state = samples.positions_angstrom[: samples.N_k[0]]
    oxygen = first_state[:, WATER_INDICES[0]]
    orientation = (
        first_state[:, WATER_INDICES[1]]
        - first_state[:, WATER_INDICES[0]]
    )
    orientation /= np.linalg.norm(orientation, axis=1)[:, None]
    return {
        "oxygen_axis_range_A": np.ptp(oxygen, axis=0).tolist(),
        "oxygen_rms_displacement_from_first_A": float(
            np.sqrt(np.mean(np.sum((oxygen - oxygen[0]) ** 2, axis=1)))
        ),
        "minimum_OH_orientation_cosine_vs_first": float(
            np.min(orientation @ orientation[0])
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-samples", type=Path, required=True)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--equilibration-steps", type=int, default=25)
    parser.add_argument("--production-steps", type=int, default=250)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--equilibration-moves", type=int, default=150)
    parser.add_argument("--moves-per-sample", type=int, default=2)
    parser.add_argument("--translation-step", type=float, default=1.2)
    parser.add_argument("--rotation-step", type=float, default=math.pi)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol = RouteAProtocol.load(args.protocol)
    if protocol.data["status"] != "research-protocol-partially-implemented":
        raise ValueError("This runner requires the partially implemented v2 protocol.")
    thresholds = protocol.data["thresholds"]

    initial_positions = _load_initial_positions(
        args.initial_samples,
        args.initial_frame,
    )
    atoms = Atoms(
        numbers=ATOMIC_NUMBERS,
        positions=initial_positions,
        pbc=False,
    )
    atoms.info.update({"charge": 0, "mult": 1})
    model = MACEOMOLProvider(
        device=args.device,
        model="maceomol",
        checkpoint=str(args.checkpoint),
        sha256=args.checkpoint_sha256,
        license_ack=True,
    )
    partition = FragmentPartition.from_atoms(
        atoms,
        solute_indices=SOLUTE_INDICES,
        water_indices=WATER_INDICES,
    )
    evaluator = ManyBodyInteractionEvaluator(
        calculator=model,
        partition=partition,
    )
    restraint = FlatBottomSurfaceRestraint.from_protocol(
        solute_indices=SOLUTE_INDICES,
        water_oxygen_indices=(WATER_INDICES[0],),
        water_atom_indices=WATER_INDICES,
        vdw_radii_angstrom={1: 1.20, 6: 1.70, 8: 1.52},
        lambda_s_angstrom=1.50,
        shell_boundary_id="route-a-acetone-lambda-1.50-v1",
        measure_id="nonperiodic-solute-com-reduced-v1",
        temperature_k=298.15,
        buffer_height_kbt=10.0,
        buffer_width_angstrom=0.5,
    )
    schedule = AlchemicalSchedule.initial(points_per_leg=6)
    run_config = WindowRunConfig(
        temperature_k=298.15,
        timestep_fs=0.1,
        friction_per_fs=0.05,
        equilibration_steps=args.equilibration_steps,
        production_steps=args.production_steps,
        sample_interval=args.sample_interval,
        seed=args.seed,
    )
    move_config = RigidBodyMoveConfig(
        water_groups=(WATER_INDICES,),
        equilibration_attempts=args.equilibration_moves,
        attempts_per_sample=args.moves_per_sample,
        translation_step_angstrom=args.translation_step,
        rotation_step_radians=args.rotation_step,
        translation_probability=0.5,
    )
    source_hashes = {
        relative: raw_sha256(PROJECT_ROOT / relative)
        for relative in RUNTIME_SOURCE_PATHS
    }
    run_preimage = {
        "artifact_type": "route-a-acetone-n1-independent-replica",
        "protocol_sha256": protocol.content_hash,
        "model": model.provenance,
        "initial_samples": args.initial_samples.resolve().as_posix(),
        "initial_samples_sha256": raw_sha256(
            args.initial_samples.resolve()
        ),
        "initial_frame": args.initial_frame,
        "initial_frame_sha256": canonical_sha256(
            {
                "atomic_numbers": list(ATOMIC_NUMBERS),
                "positions_angstrom": initial_positions.tolist(),
            }
        ),
        "schedule_sha256": schedule.content_hash,
        "run_config_sha256": run_config.content_hash,
        "move_config_sha256": move_config.content_hash,
        "repulsive_core": {"epsilon_ev": 0.48, "sigma_angstrom": 0.85},
        "shell_restraint_sha256": restraint.content_hash,
        "runtime_environment": _runtime_environment(args.device),
        "runtime_source_sha256": source_hashes,
    }
    run_hash = canonical_sha256(run_preimage)
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_hash") != run_hash:
            raise ValueError(
                "RESTART_HASH_MISMATCH: output exists for a different run."
            )
        raise ValueError(
            "RESTART_UNAVAILABLE: completed replica already exists; this "
            "research runner never overwrites it."
        )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "run-request.json",
        {"schema_version": 1, "run_hash": run_hash, **run_preimage},
    )
    _atomic_json(output / "protocol.json", protocol.data)
    _snapshot_runtime_sources(
        output,
        run_hash=run_hash,
        source_hashes=source_hashes,
    )

    runner = AlchemicalWindowRunner(
        evaluator=evaluator,
        repulsive_core=GaussianRepulsiveCore(
            epsilon_ev=0.48,
            sigma_angstrom=0.85,
        ),
        restraint=restraint,
        schedule=schedule,
        config=run_config,
        rigid_body_moves=move_config,
    )
    started = time.perf_counter()
    samples = runner.run(atoms)
    elapsed = time.perf_counter() - started
    samples.write(output / "samples")

    beta_ev_inverse = 1.0 / (units.kB * 298.15)
    table = samples.reduced_potential_table(
        beta_ev_inverse=beta_ev_inverse
    )
    diagnostics = diagnose_mbar(
        table,
        overlap_min=float(thresholds["overlap_min"]["value"]),
        effective_samples_min=float(thresholds["ess_min"]["value"]),
        bar_disagreement_kcal_max=float(
            thresholds["bar_disagreement_kcal"]["value"]
        ),
        kcal_per_dimensionless=KCAL_PER_DIMENSIONLESS,
    )
    first_mbar = diagnose_mbar(
        _half_sample_set(samples, second_half=False).reduced_potential_table(
            beta_ev_inverse=beta_ev_inverse
        ),
        overlap_min=0.0,
        effective_samples_min=1.0,
        bar_disagreement_kcal_max=1.0e9,
        kcal_per_dimensionless=KCAL_PER_DIMENSIONLESS,
    )["mbar"]
    second_mbar = diagnose_mbar(
        _half_sample_set(samples, second_half=True).reduced_potential_table(
            beta_ev_inverse=beta_ev_inverse
        ),
        overlap_min=0.0,
        effective_samples_min=1.0,
        bar_disagreement_kcal_max=1.0e9,
        kcal_per_dimensionless=KCAL_PER_DIMENSIONLESS,
    )["mbar"]
    first_delta = float(first_mbar["delta_f"][0, -1]) * KCAL_PER_DIMENSIONLESS
    second_delta = (
        float(second_mbar["delta_f"][0, -1]) * KCAL_PER_DIMENSIONLESS
    )
    half_difference = abs(first_delta - second_delta)
    stability_limit = float(
        thresholds["stability_half_traj_kcal"]["value"]
    )

    mbar = diagnostics["mbar"]
    delta_f = float(mbar["delta_f"][0, -1])
    uncertainty = float(mbar["delta_f_uncertainty"][0, -1])
    adjacent_standard_errors = np.asarray(
        [
            float(mbar["delta_f_uncertainty"][index, index + 1])
            * KCAL_PER_DIMENSIONLESS
            for index in range(len(samples.N_k) - 1)
        ],
        dtype=float,
    )
    maximum_adjacent_standard_error = float(
        np.max(adjacent_standard_errors)
    )
    standard_error_limit = float(
        thresholds["per_state_se_kcal_max"]["value"]
    )
    failure_codes = list(diagnostics["failure_codes"])
    if half_difference > stability_limit:
        failure_codes.append("HALF_TRAJECTORY_INSTABILITY")
    if maximum_adjacent_standard_error > standard_error_limit:
        failure_codes.append("PER_STATE_STANDARD_ERROR_TOO_HIGH")
    summary = {
        "schema_version": 1,
        "artifact_type": "route-a-acetone-n1-independent-replica",
        "scientific_status": "fixed-n-development-diagnostic",
        "run_hash": run_hash,
        "protocol_sha256": protocol.content_hash,
        "elapsed_s": elapsed,
        "temperature_k": 298.15,
        "schedule_labels": list(schedule.labels),
        "schedule_hash": schedule.content_hash,
        "N_k": list(samples.N_k),
        "run_config": run_config.__dict__,
        "run_config_hash": run_config.content_hash,
        "move_config": {
            **move_config.__dict__,
            "water_groups": [list(group) for group in move_config.water_groups],
        },
        "move_config_hash": move_config.content_hash,
        "move_diagnostics": list(runner.last_move_diagnostics),
        "sample_content_hash": samples.content_hash,
        "reduced_potential_table_hash": table.state_hash,
        "mbar_delta_f_D_to_P": delta_f,
        "mbar_delta_g_D_to_P_kcal_mol": (
            delta_f * KCAL_PER_DIMENSIONLESS
        ),
        "mbar_uncertainty_kcal_mol": (
            uncertainty * KCAL_PER_DIMENSIONLESS
        ),
        "scientific_gate_status": (
            "passed" if not failure_codes else "failed"
        ),
        "scientific_failure_codes": failure_codes,
        "minimum_adjacent_overlap": diagnostics[
            "minimum_adjacent_overlap"
        ],
        "minimum_effective_sample_number": diagnostics[
            "minimum_effective_sample_number"
        ],
        "maximum_adjacent_bar_mbar_disagreement_kcal_mol": diagnostics[
            "maximum_adjacent_bar_mbar_disagreement_kcal_mol"
        ],
        "maximum_adjacent_standard_error_kcal_mol": (
            maximum_adjacent_standard_error
        ),
        "per_state_standard_error_threshold_kcal_mol": (
            standard_error_limit
        ),
        "first_half_delta_g_kcal_mol": first_delta,
        "second_half_delta_g_kcal_mol": second_delta,
        "half_trajectory_difference_kcal_mol": half_difference,
        "half_trajectory_threshold_kcal_mol": stability_limit,
        "D_state_relative_mixing": _mixing_summary(samples),
        "model_provenance": model.provenance,
        "warning": (
            "This is one fixed-n development replica. It is not the protocol-v2 "
            "QCT hydration free energy and cannot establish Route A accuracy."
        ),
    }
    _atomic_json(manifest_path, summary)
    np.savez_compressed(
        output / "analysis_arrays.npz",
        delta_f=mbar["delta_f"],
        delta_f_uncertainty=mbar["delta_f_uncertainty"],
        overlap_matrix=mbar["overlap_matrix"],
        effective_sample_numbers=mbar["effective_sample_numbers"],
        adjacent_standard_errors_kcal_mol=adjacent_standard_errors,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "elapsed_s",
                    "mbar_delta_g_D_to_P_kcal_mol",
                    "mbar_uncertainty_kcal_mol",
                    "scientific_gate_status",
                    "scientific_failure_codes",
                    "minimum_adjacent_overlap",
                    "minimum_effective_sample_number",
                    "maximum_adjacent_bar_mbar_disagreement_kcal_mol",
                    "maximum_adjacent_standard_error_kcal_mol",
                    "half_trajectory_difference_kcal_mol",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
