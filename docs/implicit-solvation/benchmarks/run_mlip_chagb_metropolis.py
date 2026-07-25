#!/usr/bin/env python3
"""Sample a label-blind MLIP + CHA-GB hydration free energy by Metropolis TI."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np
from ase.units import Hartree, kB

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_core import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_chagb_nonpolar import (  # noqa: E402
    _gbnsr6_input,
    _pbsa_input,
    _run,
    load_evaluation_protocol,
    mol2_with_charges,
    parse_gbnsr6_components,
    parse_pbsa_components,
    require_no_frcmod_nonbonded_overrides,
    resolve_executables,
)
from run_mlip_conformer_weighting import _load_calculator  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

PROTOCOL_ID = "maple-route1-mlip-chagb-metropolis-ti-v1"
EV_PER_KCAL_MOL = 0.0433641153087705


class TargetEnergyError(RuntimeError):
    """A rejected geometry at the external solvent-energy boundary."""


def load_sampling_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(Path(path).resolve())
    if protocol.get("schema_version") != 1:
        raise ValueError(
            "Only MLIP/CHA-GB Metropolis protocol schema version 1 is supported."
        )
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected MLIP/CHA-GB Metropolis protocol ID.")
    boundary = protocol.get("execution_boundary", {})
    if boundary.get("energy_phase_reads_experimental_labels") is not False:
        raise ValueError("Sampling must be explicitly label-blind.")
    if boundary.get("no_experimental_fit_or_residual_model") is not True:
        raise ValueError("Residual models and experimental fits must be prohibited.")
    if boundary.get("confirmation_remains_closed") is not True:
        raise ValueError("This diagnostic must not open the confirmation partition.")
    cases = protocol.get("cases", [])
    ids = [case["compound_id"] for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Sampling case IDs must be non-empty and unique.")
    if any(
        case.get("flexibility_bin") not in {"rigid", "limited", "flexible"}
        for case in cases
    ):
        raise ValueError("Each case must have a predeclared flexibility bin.")
    lambdas = np.asarray(protocol["sampling"]["lambda_windows"], dtype=np.float64)
    if (
        lambdas.ndim != 1
        or len(lambdas) < 3
        or not np.isfinite(lambdas).all()
        or lambdas[0] != 0.0
        or lambdas[-1] != 1.0
        or np.any(np.diff(lambdas) <= 0.0)
    ):
        raise ValueError("Lambda windows must be finite, increasing, and span [0, 1].")
    if protocol["sampling"].get("integration") != "composite-simpson":
        raise ValueError(
            "The frozen v1 protocol requires composite Simpson integration."
        )
    composite_simpson(lambdas, np.zeros_like(lambdas))
    payload = json.dumps(protocol, sort_keys=True).lower()
    if "experimental_kcal_mol" in payload or "experimental_value" in payload:
        raise ValueError("Sampling protocol contains an experimental-label field.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def rotatable_bond_sides(atoms) -> list[tuple[int, int, tuple[int, ...]]]:
    """Return oriented, acyclic, heavy-atom single bonds and the end-side atoms."""

    bonds = atoms.info.get("mol2", {}).get("bonds", [])
    adjacency = [set() for _ in atoms]
    for raw_i, raw_j, _ in bonds:
        i, j = int(raw_i), int(raw_j)
        adjacency[i].add(j)
        adjacency[j].add(i)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    output: list[tuple[int, int, tuple[int, ...]]] = []
    for raw_i, raw_j, order in bonds:
        i, j = int(raw_i), int(raw_j)
        if str(order) != "1" or numbers[i] == 1 or numbers[j] == 1:
            continue
        stack = [j]
        visited: set[int] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(
                neighbor
                for neighbor in adjacency[current]
                if not ({current, neighbor} == {i, j})
            )
        if i in visited:
            continue
        output.append((i, j, tuple(sorted(visited))))
    return output


def rotate_bond_side(
    positions: np.ndarray,
    *,
    axis_start: int,
    axis_end: int,
    side: Iterable[int],
    angle_radians: float,
) -> np.ndarray:
    """Rigidly rotate one graph component around a bond axis."""

    output = np.asarray(positions, dtype=np.float64).copy()
    origin = output[int(axis_start)].copy()
    axis = output[int(axis_end)] - origin
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("Cannot rotate around a zero-length bond.")
    axis /= norm
    cosine = math.cos(float(angle_radians))
    sine = math.sin(float(angle_radians))
    for index in side:
        vector = output[int(index)] - origin
        output[int(index)] = origin + (
            vector * cosine
            + np.cross(axis, vector) * sine
            + axis * np.dot(axis, vector) * (1.0 - cosine)
        )
    return output


def metropolis_accept(
    *, delta_energy_ev: float, beta_ev_inverse: float, u: float
) -> bool:
    if not all(math.isfinite(value) for value in (delta_energy_ev, beta_ev_inverse, u)):
        raise ValueError("Metropolis inputs must be finite.")
    if beta_ev_inverse <= 0.0 or not 0.0 < u < 1.0:
        raise ValueError(
            "Metropolis beta and uniform variate are outside their domains."
        )
    if delta_energy_ev <= 0.0:
        return True
    return math.log(u) < -beta_ev_inverse * delta_energy_ev


def _simpson_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or len(x) % 2 == 0:
        raise ValueError(
            "Composite Simpson integration requires an odd number of points."
        )
    steps = np.diff(x)
    if not np.allclose(steps, steps[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("Composite Simpson integration requires uniform spacing.")
    weights = np.ones(len(x), dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * (steps[0] / 3.0)


def composite_simpson(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if x_array.shape != y_array.shape or not np.isfinite(y_array).all():
        raise ValueError(
            "Simpson coordinates and values must be finite and equal length."
        )
    return float(np.dot(_simpson_weights(x_array), y_array))


def write_amber_inpcrd(
    path: str | Path, positions_angstrom: np.ndarray, *, title: str
) -> None:
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Amber coordinates must be a finite (N, 3) array.")
    values = positions.reshape(-1)
    lines = [str(title)[:80], f"{len(positions):6d}"]
    for start in range(0, len(values), 6):
        lines.append("".join(f"{value:12.7f}" for value in values[start : start + 6]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _block_standard_error(samples: Iterable[float]) -> float:
    values = np.asarray(list(samples), dtype=np.float64)
    if len(values) < 8:
        return float("nan")
    block_count = min(8, max(2, len(values) // 20))
    trimmed = values[: (len(values) // block_count) * block_count]
    blocks = trimmed.reshape(block_count, -1).mean(axis=1)
    return float(blocks.std(ddof=1) / math.sqrt(block_count))


def _metric_block(
    predicted: np.ndarray, experimental: np.ndarray
) -> dict[str, float | int]:
    errors = np.asarray(predicted) - np.asarray(experimental)
    return {
        "n": int(len(errors)),
        "mse_kcal_mol": float(errors.mean()),
        "mae_kcal_mol": float(np.abs(errors).mean()),
        "rmse_kcal_mol": float(np.sqrt(np.square(errors).mean())),
        "maximum_absolute_error_kcal_mol": float(np.abs(errors).max()),
    }


class TargetEnergyEvaluator:
    """Evaluate the frozen CHA-GB + PBSA cavity/dispersion endpoint."""

    def __init__(
        self,
        *,
        case_dir: Path,
        topology: Path,
        protocol: dict[str, Any],
        executables: dict[str, dict[str, str]],
    ):
        self.case_dir = case_dir
        self.topology = topology
        self.executables = executables
        self.coordinate_path = case_dir / "sample.rst7"
        self.gb_input = case_dir / "gb.in"
        self.pb_input = case_dir / "pb.in"
        self.gb_output = case_dir / "gb.out"
        self.pb_output = case_dir / "pb.out"
        self.gb_input.write_text(_gbnsr6_input(protocol), encoding="utf-8")
        self.pb_input.write_text(_pbsa_input(protocol), encoding="utf-8")
        self.calls = 0
        self.seconds = 0.0
        self._cache_key: bytes | None = None
        self._cache_value: tuple[float, dict[str, float]] | None = None

    def evaluate(self, positions: np.ndarray) -> tuple[float, dict[str, float]]:
        array = np.ascontiguousarray(positions, dtype=np.float64)
        key = array.tobytes()
        if key == self._cache_key and self._cache_value is not None:
            return self._cache_value
        write_amber_inpcrd(self.coordinate_path, array, title="MAPLE target sample")
        started = time.perf_counter()
        try:
            _run(
                [
                    self.executables["gbnsr6"]["path"],
                    "-O",
                    "-i",
                    self.gb_input.name,
                    "-o",
                    self.gb_output.name,
                    "-p",
                    self.topology.name,
                    "-c",
                    self.coordinate_path.name,
                ],
                self.case_dir,
                label="gbnsr6 target energy",
            )
            _run(
                [
                    self.executables["pbsa"]["path"],
                    "-O",
                    "-i",
                    self.pb_input.name,
                    "-o",
                    self.pb_output.name,
                    "-p",
                    self.topology.name,
                    "-c",
                    self.coordinate_path.name,
                ],
                self.case_dir,
                label="pbsa target energy",
            )
            gb = parse_gbnsr6_components(self.gb_output.read_text(encoding="utf-8"))
            pb = parse_pbsa_components(self.pb_output.read_text(encoding="utf-8"))
        except (RuntimeError, ValueError) as exc:
            raise TargetEnergyError(str(exc)) from exc
        elapsed = time.perf_counter() - started
        components = {
            "chagb_polar": gb["chagb_polar"],
            "pbsa_cavity": pb["cavity"],
            "pbsa_dispersion": pb["dispersion"],
        }
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise ValueError("Target solvent endpoint returned a non-finite energy.")
        self.calls += 1
        self.seconds += elapsed
        self._cache_key = key
        self._cache_value = (total, components)
        return self._cache_value


def _load_frozen_sources(
    protocol_path: Path,
    protocol: dict[str, Any],
    source_work_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = protocol["source_evidence"]
    chagb_path = protocol_path.parent / source["chagb_protocol"]
    manifest_path = protocol_path.parent / source["label_free_manifest"]
    mlip_path = protocol_path.parent / source["mlip_protocol"]
    if sha256_file(chagb_path) != source["chagb_protocol_sha256"]:
        raise ValueError("Frozen CHA-GB protocol hash changed.")
    if sha256_file(manifest_path) != source["label_free_manifest_sha256"]:
        raise ValueError("Frozen label-free manifest hash changed.")
    if sha256_file(mlip_path) != source["mlip_protocol_sha256"]:
        raise ValueError("Frozen MLIP protocol hash changed.")
    prepared_path = source_work_dir / "prepared.json"
    if sha256_file(prepared_path) != source["prepared_development_sha256"]:
        raise ValueError("Frozen prepared development manifest hash changed.")
    chagb_protocol, _ = load_evaluation_protocol(chagb_path)
    manifest = load_json(manifest_path)
    manifest_payload = json.dumps(manifest, sort_keys=True).lower()
    if "experimental" in manifest_payload:
        raise ValueError("Energy source manifest contains an experimental-label field.")
    return chagb_protocol, manifest, load_json(mlip_path)


def _prepare_topology(
    *,
    case_dir: Path,
    source_mol2: Path,
    source_hash: str,
    charges: list[float],
    chagb_protocol: dict[str, Any],
    executables: dict[str, dict[str, str]],
) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    topology = case_dir / "p"
    metadata_path = case_dir / "topology.json"
    charge_hash = sha256_bytes(canonical_json_bytes(charges))
    expected = {
        "source_mol2_sha256": source_hash,
        "am1bcc_charge_vector_sha256": charge_hash,
    }
    if topology.is_file() and metadata_path.is_file():
        metadata = load_json(metadata_path)
        if all(metadata.get(key) == value for key, value in expected.items()):
            return topology
        raise ValueError(f"Existing topology metadata is incompatible: {case_dir}")
    if sha256_file(source_mol2) != source_hash:
        raise ValueError("Frozen source MOL2 hash changed.")
    local_mol2 = case_dir / "m.mol2"
    local_mol2.write_text(
        mol2_with_charges(source_mol2.read_text(encoding="utf-8"), charges),
        encoding="utf-8",
    )
    _run(
        [
            executables["parmchk2"]["path"],
            "-i",
            local_mol2.name,
            "-f",
            "mol2",
            "-o",
            "f",
            "-s",
            chagb_protocol["topology"]["parmchk2_mode"],
        ],
        case_dir,
        label="parmchk2",
    )
    require_no_frcmod_nonbonded_overrides((case_dir / "f").read_text(encoding="utf-8"))
    (case_dir / "tleap.in").write_text(
        "\n".join(
            [
                f"source {chagb_protocol['topology']['tleap_source']}",
                f"set default PBradii {chagb_protocol['topology']['pb_radii']}",
                f"MOL = loadmol2 {local_mol2.name}",
                "loadamberparams f",
                "saveamberparm MOL p initial.rst7",
                "quit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _run(
        [executables["tleap"]["path"], "-f", "tleap.in"],
        case_dir,
        label="tleap",
    )
    metadata = {
        **expected,
        "topology_sha256": sha256_file(topology),
        "provider_sha256": {
            name: details["sha256"] for name, details in sorted(executables.items())
        },
    }
    write_json_atomic(metadata_path, metadata)
    return topology


def _gas_energy_hartree(calculator, atoms) -> float:
    calculator.calculate(atoms, properties=["energy"], system_changes=["positions"])
    energy = float(calculator.results["energy"])
    if not math.isfinite(energy):
        raise ValueError("MLIP returned a non-finite gas energy.")
    return energy


def _propose_positions(
    *,
    positions: np.ndarray,
    rotatable: list[tuple[int, int, tuple[int, ...]]],
    rng: np.random.Generator,
    torsion_probability: float,
    torsion_maximum_radians: float,
    cartesian_sigma: float,
) -> tuple[np.ndarray, str]:
    if rotatable and rng.random() < torsion_probability:
        axis_start, axis_end, side = rotatable[int(rng.integers(len(rotatable)))]
        angle = rng.uniform(-torsion_maximum_radians, torsion_maximum_radians)
        return (
            rotate_bond_side(
                positions,
                axis_start=axis_start,
                axis_end=axis_end,
                side=side,
                angle_radians=float(angle),
            ),
            "torsion",
        )
    output = np.asarray(positions, dtype=np.float64).copy()
    atom_index = int(rng.integers(len(output)))
    output[atom_index] += rng.normal(0.0, cartesian_sigma, size=3)
    return output, "cartesian"


def _run_chain(
    *,
    atoms,
    calculator,
    target: TargetEnergyEvaluator,
    lambda_order: list[float],
    sampling: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    temperature = float(sampling["temperature_kelvin"])
    beta = 1.0 / (kB * temperature)
    burn_in = int(sampling["burn_in_moves_per_window"])
    production = int(sampling["production_moves_per_window"])
    stride = int(sampling["sample_stride"])
    torsion_probability = float(sampling["torsion_move_probability"])
    torsion_maximum = math.radians(float(sampling["torsion_maximum_angle_degrees"]))
    cartesian_sigma = float(sampling["cartesian_displacement_sigma_angstrom"])
    if burn_in < 0 or production <= 0 or stride <= 0:
        raise ValueError("Invalid frozen sampling move counts.")
    if not 0.0 <= torsion_probability <= 1.0:
        raise ValueError("Torsion probability must be in [0, 1].")
    if torsion_maximum <= 0.0 or cartesian_sigma <= 0.0:
        raise ValueError("Frozen proposal scales must be positive.")
    rotatable = rotatable_bond_sides(atoms)
    current_positions = np.asarray(atoms.get_positions(), dtype=np.float64).copy()
    current_gas = _gas_energy_hartree(calculator, atoms)
    current_target, _ = target.evaluate(current_positions)
    windows: dict[str, Any] = {}
    for lambda_value in lambda_order:
        attempted = {"torsion": 0, "cartesian": 0}
        accepted = {"torsion": 0, "cartesian": 0}
        provider_rejections = 0
        samples: list[float] = []
        total_moves = burn_in + production
        for move_index in range(total_moves):
            proposed, move_type = _propose_positions(
                positions=current_positions,
                rotatable=rotatable,
                rng=rng,
                torsion_probability=torsion_probability,
                torsion_maximum_radians=torsion_maximum,
                cartesian_sigma=cartesian_sigma,
            )
            attempted[move_type] += 1
            atoms.set_positions(proposed)
            proposed_gas = _gas_energy_hartree(calculator, atoms)
            try:
                proposed_target, _ = target.evaluate(proposed)
            except TargetEnergyError:
                provider_rejections += 1
                atoms.set_positions(current_positions)
            else:
                delta_ev = (proposed_gas - current_gas) * Hartree + float(
                    lambda_value
                ) * (proposed_target - current_target) * EV_PER_KCAL_MOL
                if metropolis_accept(
                    delta_energy_ev=float(delta_ev),
                    beta_ev_inverse=float(beta),
                    u=float(rng.random()),
                ):
                    current_positions = proposed
                    current_gas = proposed_gas
                    current_target = proposed_target
                    accepted[move_type] += 1
                else:
                    atoms.set_positions(current_positions)
            if move_index >= burn_in and (move_index - burn_in) % stride == 0:
                samples.append(float(current_target))
        sample_array = np.asarray(samples, dtype=np.float64)
        attempts = sum(attempted.values())
        accepts = sum(accepted.values())
        windows[f"{lambda_value:.8f}"] = {
            "lambda": float(lambda_value),
            "sample_count": int(len(sample_array)),
            "target_energy_samples_kcal_mol": sample_array.tolist(),
            "target_energy_mean_kcal_mol": float(sample_array.mean()),
            "target_energy_standard_deviation_kcal_mol": float(
                sample_array.std(ddof=1)
            ),
            "target_energy_block_standard_error_kcal_mol": _block_standard_error(
                sample_array
            ),
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "acceptance_fraction": float(accepts / attempts),
            "provider_rejections": provider_rejections,
        }
    return windows, current_positions


def _combine_chains(
    *,
    lambdas: np.ndarray,
    chain_records: list[dict[str, Any]],
) -> dict[str, Any]:
    combined_windows: dict[str, Any] = {}
    means: list[float] = []
    standard_errors: list[float] = []
    for lambda_value in lambdas:
        key = f"{lambda_value:.8f}"
        sequences = [
            np.asarray(chain["windows"][key]["target_energy_samples_kcal_mol"])
            for chain in chain_records
        ]
        samples = np.concatenate(sequences)
        means.append(float(samples.mean()))
        chain_block_errors = [_block_standard_error(sequence) for sequence in sequences]
        finite_errors = [value for value in chain_block_errors if math.isfinite(value)]
        standard_error = (
            float(
                math.sqrt(sum(value * value for value in finite_errors))
                / len(finite_errors)
            )
            if finite_errors
            else float("nan")
        )
        standard_errors.append(standard_error)
        combined_windows[key] = {
            "lambda": float(lambda_value),
            "sample_count": int(len(samples)),
            "target_energy_mean_kcal_mol": means[-1],
            "target_energy_standard_deviation_kcal_mol": float(samples.std(ddof=1)),
            "target_energy_block_standard_error_kcal_mol": standard_error,
            "chain_means_kcal_mol": [float(sequence.mean()) for sequence in sequences],
        }
    free_energy = composite_simpson(lambdas, means)
    weights = _simpson_weights(lambdas)
    finite = np.asarray(
        [0.0 if not math.isfinite(value) else value for value in standard_errors]
    )
    uncertainty = float(np.sqrt(np.sum(np.square(weights * finite))))
    chain_free_energies = []
    for chain in chain_records:
        chain_means = [
            chain["windows"][f"{value:.8f}"]["target_energy_mean_kcal_mol"]
            for value in lambdas
        ]
        chain_free_energies.append(composite_simpson(lambdas, chain_means))
    return {
        "windows": combined_windows,
        "free_energy_kcal_mol": free_energy,
        "block_standard_error_kcal_mol": uncertainty,
        "chain_free_energies_kcal_mol": chain_free_energies,
        "chain_free_energy_disagreement_kcal_mol": float(
            max(chain_free_energies) - min(chain_free_energies)
        ),
    }


def _record_path(work_dir: Path, compound_id: str) -> Path:
    return work_dir / "records" / f"{compound_id}.json"


def scientific_record_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only fields before independent-reproduction comparison."""

    projected = json.loads(json.dumps(record))
    sampling = projected.get("sampling", {})
    sampling.pop("target_provider_seconds", None)
    sampling.pop("wall_seconds", None)
    return projected


def run_case(
    *,
    case: dict[str, Any],
    protocol: dict[str, Any],
    fingerprint: str,
    chagb_protocol: dict[str, Any],
    manifest_row: dict[str, Any],
    source_work_dir: Path,
    work_dir: Path,
    calculator,
    executables: dict[str, dict[str, str]],
) -> dict[str, Any]:
    compound_id = case["compound_id"]
    source_mol2 = source_work_dir / manifest_row["source_mol2_relative_path"]
    atoms = MOL2Reader(str(source_mol2), charge=0, mult=1)
    if list(map(int, atoms.get_atomic_numbers())) and not set(
        map(int, atoms.get_atomic_numbers())
    ).issubset(set(protocol["model"]["supported_atomic_numbers"])):
        raise ValueError(f"MLIP element-domain failure for {compound_id}.")
    charges = [float(value) for value in manifest_row["am1bcc_charges_e"]]
    if len(charges) != len(atoms):
        raise ValueError(f"AM1-BCC charge length mismatch for {compound_id}.")
    atoms.set_initial_charges(charges)
    case_dir = work_dir / "cases" / compound_id
    topology = _prepare_topology(
        case_dir=case_dir,
        source_mol2=source_mol2,
        source_hash=manifest_row["source_mol2_sha256"],
        charges=charges,
        chagb_protocol=chagb_protocol,
        executables=executables,
    )
    target = TargetEnergyEvaluator(
        case_dir=case_dir,
        topology=topology,
        protocol=chagb_protocol,
        executables=executables,
    )
    initial_positions = np.asarray(atoms.get_positions(), dtype=np.float64).copy()
    fixed_target, fixed_components = target.evaluate(initial_positions)
    lambdas = np.asarray(protocol["sampling"]["lambda_windows"], dtype=np.float64)
    chains: list[dict[str, Any]] = []
    started = time.perf_counter()
    for specification in protocol["sampling"]["chains"]:
        atoms.set_positions(initial_positions)
        order = (
            lambdas.tolist()
            if specification["lambda_order"] == "ascending"
            else lambdas[::-1].tolist()
        )
        windows, final_positions = _run_chain(
            atoms=atoms,
            calculator=calculator,
            target=target,
            lambda_order=order,
            sampling=protocol["sampling"],
            seed=int(case["seed"]) + int(specification["seed_offset"]),
        )
        chain_means = [
            windows[f"{value:.8f}"]["target_energy_mean_kcal_mol"] for value in lambdas
        ]
        chains.append(
            {
                "chain_id": specification["chain_id"],
                "seed": int(case["seed"]) + int(specification["seed_offset"]),
                "lambda_order": specification["lambda_order"],
                "windows": windows,
                "free_energy_kcal_mol": composite_simpson(lambdas, chain_means),
                "final_positions_sha256": sha256_bytes(
                    np.ascontiguousarray(final_positions).tobytes()
                ),
            }
        )
    combined = _combine_chains(lambdas=lambdas, chain_records=chains)
    total_attempts = sum(
        sum(
            sum(window["attempted_moves"].values())
            for window in chain["windows"].values()
        )
        for chain in chains
    )
    total_accepts = sum(
        sum(
            sum(window["accepted_moves"].values())
            for window in chain["windows"].values()
        )
        for chain in chains
    )
    gates = protocol["diagnostic_gates"]
    acceptance = total_accepts / total_attempts
    record = {
        "schema_version": 1,
        "artifact_type": "mlip-chagb-metropolis-ti-case",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": protocol["source_partition"],
        "compound_id": compound_id,
        "name": case["name"],
        "flexibility_bin": case["flexibility_bin"],
        "source_mol2_sha256": manifest_row["source_mol2_sha256"],
        "source_record_sha256": manifest_row["source_record_sha256"],
        "am1bcc_charge_vector_sha256": sha256_bytes(canonical_json_bytes(charges)),
        "model": {
            "name": protocol["model"]["name"],
            "checkpoint_sha256": protocol["model"]["checkpoint_sha256"],
        },
        "fixed_geometry_target_kcal_mol": fixed_target,
        "fixed_geometry_components_kcal_mol": fixed_components,
        "sampling": {
            "temperature_kelvin": protocol["sampling"]["temperature_kelvin"],
            "lambda_windows": lambdas.tolist(),
            "rotatable_bond_count": len(rotatable_bond_sides(atoms)),
            "chains": chains,
            **combined,
            "total_acceptance_fraction": acceptance,
            "target_provider_calls": target.calls,
            "target_provider_seconds": target.seconds,
            "wall_seconds": time.perf_counter() - started,
        },
        "diagnostic_gates": {
            "acceptance_pass": acceptance
            >= float(gates["minimum_total_acceptance_fraction"]),
            "chain_agreement_pass": combined["chain_free_energy_disagreement_kcal_mol"]
            <= float(gates["maximum_chain_free_energy_disagreement_kcal_mol"]),
            "promotion_allowed": False,
        },
        "label_use_boundary": {
            "energy_phase_reads_experimental_labels": False,
            "experimental_fit_or_residual_model": False,
        },
        "provider_sha256": {
            name: details["sha256"] for name, details in sorted(executables.items())
        },
    }
    destination = _record_path(work_dir, compound_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, record)
    return record


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_sampling_protocol(protocol_path)
    source_work_dir = Path(args.source_work_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    chagb_protocol, manifest, _ = _load_frozen_sources(
        protocol_path, protocol, source_work_dir
    )
    executables = resolve_executables(chagb_protocol, args.amber_bin)
    rows = {row["compound_id"]: row for row in manifest["records"]}
    selected = set(args.case or [case["compound_id"] for case in protocol["cases"]])
    unknown = selected.difference(case["compound_id"] for case in protocol["cases"])
    if unknown:
        raise ValueError(f"Cases are not in the frozen protocol: {sorted(unknown)}")
    cases = [case for case in protocol["cases"] if case["compound_id"] in selected]
    first_row = rows[cases[0]["compound_id"]]
    first_atoms = MOL2Reader(
        str(source_work_dir / first_row["source_mol2_relative_path"]),
        charge=0,
        mult=1,
    )
    calculator, environment = _load_calculator(protocol, args.device, first_atoms)
    environment_path = work_dir / "environment.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        environment_path,
        {
            **environment,
            "protocol_fingerprint": fingerprint,
            "platform": platform.platform(),
        },
    )
    for index, case in enumerate(cases, start=1):
        destination = _record_path(work_dir, case["compound_id"])
        if destination.is_file() and not args.overwrite:
            existing = load_json(destination)
            if existing.get("protocol_fingerprint") == fingerprint:
                print(f"[{index}/{len(cases)}] skip {case['compound_id']}")
                continue
            raise ValueError(f"Existing record is incompatible: {destination}")
        print(f"[{index}/{len(cases)}] sample {case['compound_id']} ({case['name']})")
        record = run_case(
            case=case,
            protocol=protocol,
            fingerprint=fingerprint,
            chagb_protocol=chagb_protocol,
            manifest_row=rows[case["compound_id"]],
            source_work_dir=source_work_dir,
            work_dir=work_dir,
            calculator=calculator,
            executables=executables,
        )
        print(
            f"  fixed={record['fixed_geometry_target_kcal_mol']:.4f} "
            f"sampled={record['sampling']['free_energy_kcal_mol']:.4f} "
            f"accept={record['sampling']['total_acceptance_fraction']:.3f} "
            f"chain_gap={record['sampling']['chain_free_energy_disagreement_kcal_mol']:.3f}"
        )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_sampling_protocol(protocol_path)
    source_work_dir = Path(args.source_work_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    prepared = load_json(source_work_dir / "prepared.json")
    candidates = {
        candidate["compound_id"]: candidate for candidate in prepared["candidates"]
    }
    records = []
    for case in protocol["cases"]:
        path = _record_path(work_dir, case["compound_id"])
        if not path.is_file():
            raise FileNotFoundError(path)
        record = load_json(path)
        if record.get("protocol_fingerprint") != fingerprint:
            raise ValueError(f"Protocol mismatch for {case['compound_id']}.")
        candidate = candidates[case["compound_id"]]
        if candidate.get("partition") != "development":
            raise ValueError(
                "Sampling summary attempted to open a non-development case."
            )
        records.append((record, candidate))
    fixed = np.asarray(
        [record["fixed_geometry_target_kcal_mol"] for record, _ in records]
    )
    sampled = np.asarray(
        [record["sampling"]["free_energy_kcal_mol"] for record, _ in records]
    )
    experimental = np.asarray(
        [candidate["experimental_kcal_mol"] for _, candidate in records]
    )
    outcomes = {"improved": 0, "unchanged": 0, "worsened": 0}
    case_results = {}
    for record, candidate in records:
        before = abs(
            record["fixed_geometry_target_kcal_mol"]
            - candidate["experimental_kcal_mol"]
        )
        after = abs(
            record["sampling"]["free_energy_kcal_mol"]
            - candidate["experimental_kcal_mol"]
        )
        if after < before - 1.0e-12:
            outcomes["improved"] += 1
        elif after > before + 1.0e-12:
            outcomes["worsened"] += 1
        else:
            outcomes["unchanged"] += 1
        case_results[record["compound_id"]] = {
            "name": candidate["name"],
            "flexibility_bin": candidate["flexibility_bin"],
            "experimental_kcal_mol": candidate["experimental_kcal_mol"],
            "fixed_geometry_kcal_mol": record["fixed_geometry_target_kcal_mol"],
            "sampled_free_energy_kcal_mol": record["sampling"]["free_energy_kcal_mol"],
            "sampling_shift_kcal_mol": record["sampling"]["free_energy_kcal_mol"]
            - record["fixed_geometry_target_kcal_mol"],
            "block_standard_error_kcal_mol": record["sampling"][
                "block_standard_error_kcal_mol"
            ],
            "chain_disagreement_kcal_mol": record["sampling"][
                "chain_free_energy_disagreement_kcal_mol"
            ],
            "acceptance_fraction": record["sampling"]["total_acceptance_fraction"],
            "diagnostic_gates": record["diagnostic_gates"],
        }
    reproduction_required = bool(
        protocol["diagnostic_gates"]["required_independent_reproduction"]
    )
    reproduction = {
        "required": reproduction_required,
        "verified": False,
        "comparison_excludes": [
            "sampling.target_provider_seconds",
            "sampling.wall_seconds",
        ],
        "scientific_record_sha256": {
            record["compound_id"]: sha256_bytes(
                canonical_json_bytes(scientific_record_projection(record))
            )
            for record, _ in records
        },
    }
    if args.reproduction_work_dir:
        reproduction_work_dir = Path(args.reproduction_work_dir).resolve()
        for record, _ in records:
            compound_id = record["compound_id"]
            reproduction_path = _record_path(reproduction_work_dir, compound_id)
            if not reproduction_path.is_file():
                raise FileNotFoundError(reproduction_path)
            repeated = load_json(reproduction_path)
            if repeated.get("protocol_fingerprint") != fingerprint:
                raise ValueError(f"Reproduction protocol mismatch for {compound_id}.")
            if scientific_record_projection(repeated) != scientific_record_projection(
                record
            ):
                raise ValueError(
                    f"Independent scientific reproduction mismatch for {compound_id}."
                )
        reproduction["verified"] = True
    summary = {
        "schema_version": 1,
        "artifact_type": "mlip-chagb-metropolis-ti-summary",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": protocol["source_partition"],
        "case_count": len(records),
        "methods": {
            "fixed_geometry": _metric_block(fixed, experimental),
            "mlip_finite_temperature": _metric_block(sampled, experimental),
        },
        "paired_absolute_error_outcomes": outcomes,
        "case_results": case_results,
        "independent_reproduction": reproduction,
        "record_sha256": {
            case["compound_id"]: sha256_file(
                _record_path(work_dir, case["compound_id"])
            )
            for case in protocol["cases"]
        },
        "label_use_boundary": {
            "energy_phase": "No experimental labels read.",
            "summary_phase": "Development labels opened only after all sampling records existed.",
            "experimental_fit_or_residual_model": False,
        },
        "interpretation": (
            "Small development-only energy-based Metropolis/TI probe. The MLIP gas "
            "potential and frozen CHA-GB/cavity-dispersion endpoint define every "
            "acceptance probability. This bypasses unavailable target forces without "
            "substituting an MM population model. Promotion remains prohibited."
        ),
    }
    write_json_atomic(args.summary, summary)
    print(f"Wrote summary to {Path(args.summary).resolve()}")
    return summary


def _defaults() -> dict[str, Path]:
    return {
        "protocol": SCRIPT_DIR / "mlip_chagb_metropolis_protocol.json",
        "source_work_dir": (
            REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
        ),
        "work_dir": (
            REPOSITORY_ROOT / ".omx/benchmarks/mlip-chagb-metropolis-route1-20260724"
        ),
        "summary": SCRIPT_DIR / "freesolv-mlip-chagb-metropolis-2026-07-24.json",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "summarize", "all"))
    parser.add_argument("--protocol", default=str(defaults["protocol"]))
    parser.add_argument("--source-work-dir", default=str(defaults["source_work_dir"]))
    parser.add_argument("--work-dir", default=str(defaults["work_dir"]))
    parser.add_argument("--summary", default=str(defaults["summary"]))
    parser.add_argument("--reproduction-work-dir")
    parser.add_argument("--amber-bin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case", action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase in {"run", "all"}:
        run(args)
    if args.phase in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
