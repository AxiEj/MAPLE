#!/usr/bin/env python3
"""Measure the preregistered AIMNet2 multi-solvent force panel.

Every measurement is private below ``.omx`` and bound to one clean commit.
The primary and replay arms must run as separate processes.  Finalization
compares the two cold-process records before applying the pure reducer; it
never changes a public MAPLE capability.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

for _thread_environment_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks"
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from benchmark_core import (
    numerical_runtime_identity,
    sha256_file,
    single_threaded_numerics,
    write_json_atomic,
)
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from maple.function.calculator.extra_correction.implicit.aimnet2_frozen_smd import (
    build_aimnet2_frozen_charge_smooth_partition_smd_scalar,
)
from maple.function.calculator.aimnet._aimnet2_float64_source import (
    AIMNET_FLOAT64_RUNTIME_VERSION,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
from maple.solvation.continuum.aimnet2_smooth_partition_ddpcm import (
    AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
    AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2,
)
from maple.solvation.release.aimnet2_multisolvent_force import (
    FORCE_MEASUREMENT_ARTIFACT,
    FORCE_MEASUREMENT_CLAIM_BOUNDARY,
    FORCE_MEASUREMENT_SCHEMA_VERSION,
    FORCE_MEASUREMENT_STATUS,
    FORCE_PREQUALIFICATION_LEAVES,
    FORCE_PREQUALIFICATION_STEPS_A,
    FORCE_PREQUALIFICATION_THRESHOLDS,
    FORCE_SOURCE_PARITY_THRESHOLDS,
    NO_CAPABILITIES,
    canonical_sha256,
    cartesian_directions,
    finalize_force_replicates,
    geometry_internal_directions,
    topology_preflight,
    validate_force_measurement_record,
)
from maple.solvation.api.profiles import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_PROFILE_V1,
)
from maple.solvation.api.scalar_registry import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_V1,
)
from maple.solvation.models.aimnet2 import (
    AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT,
)

PROTOCOL_ARTIFACT = "route2-aimnet2-smooth-ddpcm-force-prequalification-protocol-v1"
PRIOR_ARTIFACT = "route2-mnsol-aimnet2-full-frozen-charge-matrix-v1"
SELECTION_ARTIFACT = "matched-qm-attribution-prereg-v2"
SELECTION_SCHEMA = "maple-pure-mace-polar-analytic-gaussian-qm-attribution-prereg-v2"
EXPECTED_PRIOR_RECORD_COUNT = 653
EXPECTED_SELECTION_RECORD_COUNT = 24
_PROCESS_IMPORT_TOKEN = hashlib.sha256(
    f"{os.getpid()}:{time.time_ns()}:{os.urandom(16).hex()}".encode()
).hexdigest()


def _object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object: {path}")
    return value


def _clean_commit() -> tuple[str, str]:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Force evidence requires a clean committed checkout.")
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or len(tree) != 40:
        raise RuntimeError("Could not resolve execution commit/tree.")
    return commit, tree


def _assert_same_clean_commit(commit: str, tree: str) -> None:
    observed_commit, observed_tree = _clean_commit()
    if observed_commit != commit or observed_tree != tree:
        raise RuntimeError("Execution commit/tree changed during force evidence.")


def _checkout_revalidation(commit: str, tree: str) -> dict[str, object]:
    try:
        _assert_same_clean_commit(commit, tree)
    except Exception as exc:
        return {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {"passed": True, "error_type": None, "error": None}


def _private_path(path: Path, *, directory: bool) -> Path:
    resolved = path.expanduser().resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(
            "Force measurements must remain below repository .omx."
        ) from exc
    relative = resolved.relative_to(REPO_ROOT).as_posix()
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "-v",
            "--no-index",
            "--",
            relative,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ignored.returncode != 0 or not ignored.stdout.strip():
        raise RuntimeError("Private force output is not ignored by Git.")
    origin = ignored.stdout.split("\t", 1)[0].rsplit(":", 2)[0]
    origin_path = Path(origin)
    if not origin_path.is_absolute():
        origin_path = REPO_ROOT / origin_path
    tracked_ignore = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", ".gitignore"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        origin_path.resolve() != (REPO_ROOT / ".gitignore").resolve()
        or tracked_ignore.returncode != 0
    ):
        raise RuntimeError(
            "Private force output must be protected by the tracked repository .gitignore."
        )
    if directory:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _validate_protocol(
    path: Path,
    *,
    source: Path,
    mnsol_protocol: Path,
    prior_private: Path,
    selection_manifest: Path,
    checkpoint: Path,
) -> tuple[dict[str, Any], str]:
    protocol = _object(path, name="force protocol")
    if (
        protocol.get("artifact") != PROTOCOL_ARTIFACT
        or protocol.get("schema_version") != 1
        or protocol.get("status") != "preregistered-not-executed"
    ):
        raise ValueError("Force prequalification protocol identity drifted.")
    pinned = protocol.get("pinned_inputs")
    selection = protocol.get("selection")
    finite_difference = protocol.get("finite_difference")
    scalar = protocol.get("scalar")
    if not isinstance(pinned, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("Force protocol inputs/selection are malformed.")
    if not isinstance(finite_difference, Mapping) or not isinstance(scalar, Mapping):
        raise ValueError("Force protocol finite-difference block is malformed.")
    if (
        scalar.get("aimnet2_runtime_kind") != AIMNET_FLOAT64_RUNTIME_VERSION
        or scalar.get("aimnet2_multisolvent_provider")
        != AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT.provider_id
        or scalar.get("scalar_id")
        != CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_V1
        or scalar.get("profile_id")
        != CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_PROFILE_V1
    ):
        raise ValueError("Force protocol AIMNet2 runtime/provider identity drifted.")
    exact_hashes = {
        "mnsol_dataset_zip_sha256": sha256_file(source),
        "mnsol_protocol_sha256": sha256_file(mnsol_protocol),
        "prior_private_653_sha256": sha256_file(prior_private),
        "aimnet2_checkpoint_sha256": sha256_file(checkpoint),
    }
    if any(pinned.get(key) != value for key, value in exact_hashes.items()):
        raise ValueError("A pinned force-protocol input hash drifted.")
    if pinned.get("aimnet2_checkpoint_bytes") != checkpoint.stat().st_size:
        raise ValueError("AIMNet2 checkpoint byte count drifted.")
    if selection.get("private_manifest_file_sha256") != sha256_file(selection_manifest):
        raise ValueError("Selection-manifest file hash drifted.")
    if tuple(finite_difference.get("steps_angstrom", ())) != (
        FORCE_PREQUALIFICATION_STEPS_A
    ):
        raise ValueError("Finite-difference step protocol drifted.")
    if tuple(finite_difference.get("leaves", ())) != FORCE_PREQUALIFICATION_LEAVES:
        raise ValueError("Force energy-leaf protocol drifted.")
    if protocol.get("thresholds") != FORCE_PREQUALIFICATION_THRESHOLDS:
        raise ValueError("Force prequalification thresholds drifted from code.")
    if protocol.get("source_response_thresholds") != FORCE_SOURCE_PARITY_THRESHOLDS:
        raise ValueError("Force source-response thresholds drifted from code.")
    if protocol.get("capabilities") != NO_CAPABILITIES:
        raise ValueError("Force preregistration must keep all capabilities false.")
    return protocol, sha256_file(path)


def _validate_selection_manifest(
    value: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selection = protocol["selection"]
    canonical = canonical_sha256(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    )
    if (
        selection.get("origin_artifact") != SELECTION_ARTIFACT
        or value.get("artifact_sha256")
        != selection.get("origin_artifact_canonical_sha256")
        or canonical != value.get("artifact_sha256")
        or value.get("schema") != SELECTION_SCHEMA
    ):
        raise ValueError("Target-free force selection manifest identity drifted.")
    panel = value.get("panel")
    if not isinstance(panel, Mapping):
        raise ValueError("Selection manifest panel is missing.")
    records = panel.get("records")
    if (
        not isinstance(records, list)
        or len(records) != EXPECTED_SELECTION_RECORD_COUNT
        or panel.get("record_count") != EXPECTED_SELECTION_RECORD_COUNT
        or panel.get("geometry_group_count") != 14
        or panel.get("record_selection_frozen") is not True
        or panel.get("reselection_after_any_result") is not False
    ):
        raise ValueError("Target-free force selection panel drifted.")
    stratum_counts = Counter(str(record.get("stratum")) for record in records)
    if (
        selection.get("record_count") != EXPECTED_SELECTION_RECORD_COUNT
        or selection.get("geometry_group_count") != 14
        or selection.get("stratum_record_counts")
        != dict(sorted(stratum_counts.items()))
        or selection.get("selection_reads_experimental_targets") is not False
        or selection.get("selection_reads_aimnet2_or_continuum_outputs") is not False
    ):
        raise ValueError("Force protocol no longer matches the target-free panel.")
    return [dict(record) for record in records]


def _prepare_task(
    *,
    dataset: Any,
    prior: Mapping[str, Any],
    manifest_records: list[dict[str, Any]],
    record_ordinal: int,
) -> dict[str, Any]:
    if (
        prior.get("artifact") != PRIOR_ARTIFACT
        or prior.get("complete_panel") is not True
        or prior.get("do_not_commit") is not True
    ):
        raise ValueError("Prior private MNSol artifact identity drifted.")
    prior_records = prior.get("records")
    if (
        not isinstance(prior_records, list)
        or len(prior_records) != EXPECTED_PRIOR_RECORD_COUNT
    ):
        raise ValueError("Prior private MNSol artifact must contain 653 records.")
    if isinstance(record_ordinal, bool) or not 0 <= record_ordinal < len(
        manifest_records
    ):
        raise ValueError("record_ordinal is outside the frozen force panel.")
    selected = manifest_records[record_ordinal]
    matches = [
        record
        for record in prior_records
        if record.get("opaque_record_id") == selected.get("opaque_record_id")
    ]
    if len(matches) != 1:
        raise ValueError("Frozen force record does not map uniquely to prior MNSol.")
    prior_record = matches[0]
    for key in (
        "selection_index",
        "geometry_sha256",
        "canonical_solvent",
        "atom_count",
        "opaque_record_id",
    ):
        if selected.get(key) != prior_record.get(key):
            raise ValueError(f"Frozen force selection field drifted: {key}.")
    handle = str(prior_record["geometry_handle"])
    geometry = dataset.geometries.get(handle)
    if (
        geometry is None
        or geometry.sha256 != selected["geometry_sha256"]
        or len(geometry.atomic_numbers) != selected["atom_count"]
    ):
        raise ValueError("Frozen force geometry bytes drifted.")
    atomic_numbers = tuple(int(value) for value in geometry.atomic_numbers)
    coordinates = tuple(
        tuple(float(component) for component in row)
        for row in geometry.coordinates_angstrom
    )
    symbols = tuple(
        __import__("ase").data.chemical_symbols[number] for number in atomic_numbers
    )
    solvent = str(selected["canonical_solvent"])
    radii = tuple(
        float(value)
        for value in route2_coulomb_radii(
            symbols,
            solvent=solvent,
            profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
        )
    )
    recorded_radii = tuple(
        float(value) for value in prior_record["cavity_radii_angstrom"]
    )
    if radii != recorded_radii:
        raise ValueError("Force-task SMD Coulomb radii drifted from the 653 run.")
    task = {
        "record_ordinal": record_ordinal,
        "selection_index": int(selected["selection_index"]),
        "opaque_record_id": str(selected["opaque_record_id"]),
        "partition": str(prior_record["partition"]),
        "geometry_handle": handle,
        "geometry_sha256": str(selected["geometry_sha256"]),
        "canonical_solvent": solvent,
        "stratum": str(selected["stratum"]),
        "role": str(selected["role"]),
        "atomic_numbers": atomic_numbers,
        "coordinates_angstrom": coordinates,
        "cavity_radii_angstrom": radii,
    }
    task["task_sha256"] = canonical_sha256(task)
    return task


def _configure_torch_threads():
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch did not enter the preregistered single-thread mode.")
    return torch


def _scalar_identity(
    *,
    task: Mapping[str, Any],
    checkpoint: Path,
    scalar: Any | None = None,
    nonpolar_evaluation: Any | None = None,
) -> dict[str, object]:
    from ase import Atoms

    atoms = Atoms(
        numbers=task["atomic_numbers"], positions=task["coordinates_angstrom"]
    )
    atoms.info.update({"charge": 0, "mult": 1})
    if scalar is None:
        _configure_torch_threads()
        scalar = build_aimnet2_frozen_charge_smooth_partition_smd_scalar(
            atoms,
            checkpoint,
            solvent=str(task["canonical_solvent"]),
            device="cpu",
        )
    runtime_provenance = scalar.model.calculator_runtime_provenance()
    if runtime_provenance.get("runtime_kind") != AIMNET_FLOAT64_RUNTIME_VERSION:
        raise ValueError("Current AIMNet2 runtime provenance kind drifted.")
    if nonpolar_evaluation is None:
        nonpolar_evaluation = scalar.nonpolar.evaluate(atoms)
    nonpolar_runtime_provenance = dict(nonpolar_evaluation.runtime_provenance)
    center_model_topology = scalar.model.neighbor_topology(atoms)
    center_continuum_topology = scalar.continuum.topology_state(atoms)
    identity = {
        "aimnet2_runtime_kind": AIMNET_FLOAT64_RUNTIME_VERSION,
        "aimnet2_runtime_provenance": runtime_provenance,
        "aimnet2_runtime_provenance_sha256": canonical_sha256(runtime_provenance),
        "aimnet2_provider_id": scalar.model.provider_id,
        "scalar_id": scalar.scalar_id,
        "profile_id": scalar.profile_id,
        "scalar_fingerprint_sha256": scalar.fingerprint_sha256(),
        "model_configuration_sha256": scalar.model.configuration_sha256(),
        "continuum_configuration_sha256": scalar.continuum.configuration_sha256(),
        "continuum_provenance_sha256": scalar.continuum.provenance_sha256,
        "nonpolar_provider_id": scalar.nonpolar.provider_id,
        "nonpolar_profile_id": scalar.nonpolar.nonpolar_profile_id,
        "nonpolar_configuration_sha256": scalar.nonpolar.configuration_sha256(),
        "nonpolar_runtime_provenance": nonpolar_runtime_provenance,
        "nonpolar_runtime_provenance_sha256": canonical_sha256(
            nonpolar_runtime_provenance
        ),
        "center_model_topology_sha256": center_model_topology.topology_sha256,
        "center_model_minimum_cutoff_margin_angstrom": (
            center_model_topology.minimum_cutoff_margin_angstrom
        ),
        "center_continuum_topology_sha256": canonical_sha256(center_continuum_topology),
    }
    if (
        identity["aimnet2_provider_id"]
        != AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT.provider_id
        or identity["scalar_id"]
        != CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_V1
        or identity["profile_id"]
        != CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_PROFILE_V1
    ):
        raise ValueError("Current force scalar identity drifted.")
    return identity


def _expected_bindings(
    *,
    execution_git_commit: str,
    execution_git_tree: str,
    protocol_file_sha256: str,
    selection_manifest_file_sha256: str,
    selection_manifest_canonical_sha256: str,
    checkpoint: Path,
    scalar_identity: Mapping[str, object],
    numerical_runtime_sha256: str,
    task: Mapping[str, Any],
    stage: str,
    mode: str,
) -> dict[str, object]:
    return {
        "execution_git_commit": execution_git_commit,
        "execution_git_tree": execution_git_tree,
        "protocol_file_sha256": protocol_file_sha256,
        "selection_manifest_file_sha256": selection_manifest_file_sha256,
        "selection_manifest_canonical_sha256": (selection_manifest_canonical_sha256),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        **{
            key: scalar_identity[key]
            for key in (
                "aimnet2_runtime_kind",
                "aimnet2_runtime_provenance_sha256",
                "aimnet2_provider_id",
                "scalar_id",
                "profile_id",
                "scalar_fingerprint_sha256",
                "model_configuration_sha256",
                "continuum_configuration_sha256",
                "continuum_provenance_sha256",
                "nonpolar_provider_id",
                "nonpolar_profile_id",
                "nonpolar_configuration_sha256",
                "nonpolar_runtime_provenance_sha256",
                "center_model_topology_sha256",
                "center_model_minimum_cutoff_margin_angstrom",
                "center_continuum_topology_sha256",
            )
        },
        "numerical_runtime_sha256": numerical_runtime_sha256,
        "task": dict(task),
        "stage": stage,
        "mode": mode,
    }


def _validate_stage(
    protocol: Mapping[str, Any], *, stage: str, record_ordinal: int, mode: str
) -> None:
    stages = protocol.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("Force protocol stage block is malformed.")
    if stage == "gate0":
        gate = stages.get("gate0_full_cartesian_canary")
        if not isinstance(gate, Mapping):
            raise ValueError("Gate-0 protocol is missing.")
        if record_ordinal != gate.get("panel_record_ordinal") or mode != gate.get(
            "mode"
        ):
            raise ValueError("Gate-0 must use its frozen full-Cartesian canary.")
    elif stage == "gate1":
        if mode != "directional":
            raise ValueError("Gate-1 accepts directional measurements only.")
    elif stage == "gate2":
        raise RuntimeError(
            "Gate-2 remains closed until Gate-0 and Gate-1 aggregate evidence passes."
        )
    else:
        raise ValueError("stage must be gate0, gate1, or gate2.")


def _energy_leaves(evaluation: Any) -> dict[str, float]:
    vacuum = float(evaluation.vacuum_energy_eV)
    continuum = float(evaluation.continuum_energy_eV)
    nonpolar = float(evaluation.nonpolar_energy_eV)
    result = {
        "vacuum": vacuum,
        "continuum": continuum,
        "electrostatic_total": vacuum + continuum,
        "nonpolar": nonpolar,
        "total": float(evaluation.total_energy_eV),
    }
    if not math.isclose(
        result["electrostatic_total"] + nonpolar,
        result["total"],
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise RuntimeError("Measured total-energy leaf ledger does not close.")
    return result


def _gradient_leaves(evaluation: Any) -> dict[str, list[list[float]]]:
    continuum = np.asarray(
        evaluation.continuum_fixed_source_gradient_eV_per_A
    ) + np.asarray(evaluation.source_response_gradient_eV_per_A)
    arrays = {
        "vacuum": np.asarray(evaluation.intrinsic_gradient_eV_per_A),
        "continuum": continuum,
        "electrostatic_total": np.asarray(
            evaluation.electrostatic_total_gradient_eV_per_A
        ),
        "nonpolar": np.asarray(evaluation.nonpolar_gradient_eV_per_A),
        "total": np.asarray(evaluation.total_gradient_eV_per_A),
    }
    if not np.allclose(
        arrays["vacuum"] + arrays["continuum"],
        arrays["electrostatic_total"],
        rtol=0.0,
        atol=2.0e-10,
    ) or not np.allclose(
        arrays["electrostatic_total"] + arrays["nonpolar"],
        arrays["total"],
        rtol=0.0,
        atol=2.0e-10,
    ):
        raise RuntimeError("Measured total-gradient leaf ledger does not close.")
    return {
        key: np.asarray(value, dtype=float).tolist() for key, value in arrays.items()
    }


def _continuum_preflight(scalar: Any, atoms: Any) -> dict[str, object]:
    return topology_preflight(
        atoms.get_positions(),
        scalar.continuum.radii_angstrom,
        transition_width_angstrom2=AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2,
        partition_lmax=AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
    )


def _energy_snapshot(scalar: Any, atoms: Any) -> dict[str, object]:
    topology = scalar.model.neighbor_topology(atoms)
    continuum_topology = scalar.continuum.topology_state(atoms)
    evaluation = scalar.evaluate_energy_components(atoms)
    return {
        "energies_eV": _energy_leaves(evaluation),
        "model_topology_sha256": topology.topology_sha256,
        "model_minimum_cutoff_margin_angstrom": topology.minimum_cutoff_margin_angstrom,
        "continuum_topology": continuum_topology,
        "continuum_preflight": _continuum_preflight(scalar, atoms),
    }


def _measure(
    *,
    task: Mapping[str, Any],
    checkpoint: Path,
    protocol_sha256: str,
    selection_manifest_file_sha256: str,
    selection_manifest_canonical_sha256: str,
    execution_git_commit: str,
    execution_git_tree: str,
    stage: str,
    mode: str,
    replicate: str,
) -> dict[str, object]:
    from ase import Atoms

    torch = _configure_torch_threads()
    atoms = Atoms(
        numbers=task["atomic_numbers"], positions=task["coordinates_angstrom"]
    )
    atoms.info.update({"charge": 0, "mult": 1})
    scalar = build_aimnet2_frozen_charge_smooth_partition_smd_scalar(
        atoms,
        checkpoint,
        solvent=str(task["canonical_solvent"]),
        device="cpu",
    )
    started = time.perf_counter()
    center_topology = scalar.model.neighbor_topology(atoms)
    center_continuum_topology = scalar.continuum.topology_state(atoms)
    center_result = scalar.evaluate(atoms)
    scalar_identity = _scalar_identity(
        task=task,
        checkpoint=checkpoint,
        scalar=scalar,
        nonpolar_evaluation=center_result.nonpolar,
    )
    numerical_runtime = numerical_runtime_identity(torch_module=torch)
    center = {
        "positions_angstrom": atoms.get_positions().tolist(),
        "energies_eV": _energy_leaves(center_result.energy),
        "gradients_eV_per_A": _gradient_leaves(center_result),
        "source": np.asarray(center_result.source, dtype=float).tolist(),
        "reaction_field": np.asarray(
            center_result.reaction_field, dtype=float
        ).tolist(),
        "aimnet2_source_response_parity": (scalar.model.last_source_response_parity()),
        "continuum_reciprocity_audit": center_result.reciprocity_audit.as_dict(),
        "model_topology_sha256": center_topology.topology_sha256,
        "model_minimum_cutoff_margin_angstrom": (
            center_topology.minimum_cutoff_margin_angstrom
        ),
        "continuum_topology": center_continuum_topology,
        "continuum_preflight": _continuum_preflight(scalar, atoms),
    }
    if mode == "directional":
        vectors = geometry_internal_directions(
            atoms.get_positions(), str(task["geometry_sha256"])
        )
    elif mode == "full-cartesian":
        vectors = cartesian_directions(len(atoms))
    else:
        raise ValueError("mode must be directional or full-cartesian.")
    directions: dict[str, object] = {}
    original = atoms.get_positions()
    for name, vector in vectors.items():
        samples = []
        for step in FORCE_PREQUALIFICATION_STEPS_A:
            plus_atoms = atoms.copy()
            minus_atoms = atoms.copy()
            plus_atoms.set_positions(original + step * vector)
            minus_atoms.set_positions(original - step * vector)
            plus = _energy_snapshot(scalar, plus_atoms)
            minus = _energy_snapshot(scalar, minus_atoms)
            samples.append(
                {
                    "step_angstrom": step,
                    "plus_energies_eV": plus["energies_eV"],
                    "minus_energies_eV": minus["energies_eV"],
                    "plus_model_topology_sha256": plus["model_topology_sha256"],
                    "minus_model_topology_sha256": minus["model_topology_sha256"],
                    "plus_model_minimum_cutoff_margin_angstrom": plus[
                        "model_minimum_cutoff_margin_angstrom"
                    ],
                    "minus_model_minimum_cutoff_margin_angstrom": minus[
                        "model_minimum_cutoff_margin_angstrom"
                    ],
                    "plus_continuum_topology": plus["continuum_topology"],
                    "minus_continuum_topology": minus["continuum_topology"],
                    "plus_continuum_preflight": plus["continuum_preflight"],
                    "minus_continuum_preflight": minus["continuum_preflight"],
                }
            )
        directions[name] = {"vector": vector.tolist(), "samples": samples}
    record: dict[str, object] = {
        "artifact": FORCE_MEASUREMENT_ARTIFACT,
        "schema_version": FORCE_MEASUREMENT_SCHEMA_VERSION,
        "status": FORCE_MEASUREMENT_STATUS,
        "do_not_commit": True,
        "execution_git_commit": execution_git_commit,
        "execution_git_tree": execution_git_tree,
        "protocol_file_sha256": protocol_sha256,
        "selection_manifest_file_sha256": selection_manifest_file_sha256,
        "selection_manifest_canonical_sha256": selection_manifest_canonical_sha256,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        **scalar_identity,
        "numerical_runtime": numerical_runtime,
        "numerical_runtime_sha256": canonical_sha256(numerical_runtime),
        "task": dict(task),
        "stage": stage,
        "mode": mode,
        "replicate": replicate,
        "process_identity": {
            "pid": os.getpid(),
            "process_import_token": _PROCESS_IMPORT_TOKEN,
            "python_executable": sys.executable,
        },
        "scalar_fingerprint_sha256": scalar.fingerprint_sha256(),
        "model_configuration_sha256": scalar.model.configuration_sha256(),
        "continuum_configuration_sha256": scalar.continuum.configuration_sha256(),
        "continuum_provenance_sha256": scalar.continuum.provenance_sha256,
        "center": center,
        "directions": directions,
        "runtime_seconds": time.perf_counter() - started,
        "capabilities": NO_CAPABILITIES,
        "claim_boundary": FORCE_MEASUREMENT_CLAIM_BOUNDARY,
    }
    record["record_sha256"] = canonical_sha256(record)
    validate_force_measurement_record(
        record,
        expected_bindings=_expected_bindings(
            execution_git_commit=execution_git_commit,
            execution_git_tree=execution_git_tree,
            protocol_file_sha256=protocol_sha256,
            selection_manifest_file_sha256=selection_manifest_file_sha256,
            selection_manifest_canonical_sha256=(selection_manifest_canonical_sha256),
            checkpoint=checkpoint,
            scalar_identity=scalar_identity,
            numerical_runtime_sha256=canonical_sha256(numerical_runtime),
            task=task,
            stage=stage,
            mode=mode,
        ),
        replicate=replicate,
    )
    return record


def _failed_measurement_record(
    *,
    task: Mapping[str, Any],
    checkpoint: Path,
    protocol_sha256: str,
    selection_manifest_file_sha256: str,
    selection_manifest_canonical_sha256: str,
    execution_git_commit: str,
    execution_git_tree: str,
    stage: str,
    mode: str,
    replicate: str,
    error: Exception,
    runtime_seconds: float,
    checkout_revalidation: Mapping[str, object],
) -> dict[str, object]:
    record: dict[str, object] = {
        "artifact": FORCE_MEASUREMENT_ARTIFACT,
        "schema_version": FORCE_MEASUREMENT_SCHEMA_VERSION,
        "status": "private-measurement-failed",
        "do_not_commit": True,
        "execution_git_commit": execution_git_commit,
        "execution_git_tree": execution_git_tree,
        "protocol_file_sha256": protocol_sha256,
        "selection_manifest_file_sha256": selection_manifest_file_sha256,
        "selection_manifest_canonical_sha256": selection_manifest_canonical_sha256,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "declared_aimnet2_runtime_kind": AIMNET_FLOAT64_RUNTIME_VERSION,
        "declared_aimnet2_provider_id": (
            AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT.provider_id
        ),
        "declared_scalar_id": (
            CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_V1
        ),
        "declared_profile_id": (
            CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_PROFILE_V1
        ),
        "task": dict(task),
        "stage": stage,
        "mode": mode,
        "replicate": replicate,
        "process_identity": {
            "pid": os.getpid(),
            "process_import_token": _PROCESS_IMPORT_TOKEN,
            "python_executable": sys.executable,
        },
        "error_type": type(error).__name__,
        "error": str(error),
        "checkout_revalidation": dict(checkout_revalidation),
        "runtime_seconds": float(runtime_seconds),
        "capabilities": dict(NO_CAPABILITIES),
        "claim_boundary": (
            "Retained private negative force-prequalification measurement; no "
            "public E/F/H/V/M or daily-task admission."
        ),
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _measure_command(args: argparse.Namespace) -> None:
    commit, tree = _clean_commit()
    source = args.source.expanduser().resolve(strict=True)
    mnsol_protocol = args.mnsol_protocol.expanduser().resolve(strict=True)
    prior_private = args.prior_private.expanduser().resolve(strict=True)
    selection_manifest = args.selection_manifest.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    protocol_path = args.force_protocol.expanduser().resolve(strict=True)
    output_dir = _private_path(args.output_dir, directory=True)
    output = output_dir / (
        f"record-{args.record_ordinal:02d}-{args.stage}-{args.mode}-{args.replicate}.json"
    )
    if output.exists():
        raise FileExistsError(f"Private force record already exists: {output}")
    protocol, protocol_sha256 = _validate_protocol(
        protocol_path,
        source=source,
        mnsol_protocol=mnsol_protocol,
        prior_private=prior_private,
        selection_manifest=selection_manifest,
        checkpoint=checkpoint,
    )
    manifest = _object(selection_manifest, name="selection manifest")
    manifest_records = _validate_selection_manifest(manifest, protocol)
    _validate_stage(
        protocol,
        stage=args.stage,
        record_ordinal=args.record_ordinal,
        mode=args.mode,
    )
    dataset = load_mnsol_v2012(source, load_mnsol_protocol(mnsol_protocol))
    task = _prepare_task(
        dataset=dataset,
        prior=_object(prior_private, name="prior private artifact"),
        manifest_records=manifest_records,
        record_ordinal=args.record_ordinal,
    )
    started = time.perf_counter()
    try:
        result = _measure(
            task=task,
            checkpoint=checkpoint,
            protocol_sha256=protocol_sha256,
            selection_manifest_file_sha256=sha256_file(selection_manifest),
            selection_manifest_canonical_sha256=str(manifest["artifact_sha256"]),
            execution_git_commit=commit,
            execution_git_tree=tree,
            stage=args.stage,
            mode=args.mode,
            replicate=args.replicate,
        )
        _assert_same_clean_commit(commit, tree)
    except Exception as exc:
        checkout_revalidation = _checkout_revalidation(commit, tree)
        failed = _failed_measurement_record(
            task=task,
            checkpoint=checkpoint,
            protocol_sha256=protocol_sha256,
            selection_manifest_file_sha256=sha256_file(selection_manifest),
            selection_manifest_canonical_sha256=str(manifest["artifact_sha256"]),
            execution_git_commit=commit,
            execution_git_tree=tree,
            stage=args.stage,
            mode=args.mode,
            replicate=args.replicate,
            error=exc,
            runtime_seconds=time.perf_counter() - started,
            checkout_revalidation=checkout_revalidation,
        )
        write_json_atomic(output, failed)
        print(output)
        print(failed["record_sha256"])
        raise
    write_json_atomic(output, result)
    print(output)
    print(result["record_sha256"])


def _finalize_command(args: argparse.Namespace) -> None:
    commit, tree = _clean_commit()
    source = args.source.expanduser().resolve(strict=True)
    mnsol_protocol = args.mnsol_protocol.expanduser().resolve(strict=True)
    prior_private = args.prior_private.expanduser().resolve(strict=True)
    selection_manifest = args.selection_manifest.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    protocol_path = args.force_protocol.expanduser().resolve(strict=True)
    primary_path = _private_path(args.primary, directory=False)
    replay_path = _private_path(args.replay, directory=False)
    output = _private_path(args.output, directory=False)
    if primary_path == replay_path:
        raise ValueError("Primary and replay records must be different files.")
    if output.exists():
        raise FileExistsError(f"Finalized private force record exists: {output}")
    protocol, protocol_sha256 = _validate_protocol(
        protocol_path,
        source=source,
        mnsol_protocol=mnsol_protocol,
        prior_private=prior_private,
        selection_manifest=selection_manifest,
        checkpoint=checkpoint,
    )
    manifest = _object(selection_manifest, name="selection manifest")
    manifest_records = _validate_selection_manifest(manifest, protocol)
    primary = _object(primary_path, name="primary force record")
    replay = _object(replay_path, name="replay force record")
    primary_task = primary.get("task")
    if not isinstance(primary_task, Mapping):
        raise TypeError("primary force task must be a mapping.")
    record_ordinal = primary_task.get("record_ordinal")
    if isinstance(record_ordinal, bool) or not isinstance(record_ordinal, int):
        raise ValueError("Primary force record ordinal is invalid.")
    stage = str(primary.get("stage", ""))
    mode = str(primary.get("mode", ""))
    _validate_stage(
        protocol,
        stage=stage,
        record_ordinal=record_ordinal,
        mode=mode,
    )
    dataset = load_mnsol_v2012(source, load_mnsol_protocol(mnsol_protocol))
    task = _prepare_task(
        dataset=dataset,
        prior=_object(prior_private, name="prior private artifact"),
        manifest_records=manifest_records,
        record_ordinal=record_ordinal,
    )
    torch = _configure_torch_threads()
    scalar_identity = _scalar_identity(task=task, checkpoint=checkpoint)
    numerical_runtime = numerical_runtime_identity(torch_module=torch)
    bindings = _expected_bindings(
        execution_git_commit=commit,
        execution_git_tree=tree,
        protocol_file_sha256=protocol_sha256,
        selection_manifest_file_sha256=sha256_file(selection_manifest),
        selection_manifest_canonical_sha256=str(manifest["artifact_sha256"]),
        checkpoint=checkpoint,
        scalar_identity=scalar_identity,
        numerical_runtime_sha256=canonical_sha256(numerical_runtime),
        task=task,
        stage=stage,
        mode=mode,
    )
    finalized = finalize_force_replicates(
        primary,
        replay,
        expected_bindings=bindings,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_same_clean_commit(commit, tree)
    write_json_atomic(output, finalized)
    print(output)
    print(finalized["record_sha256"])
    print(json.dumps(finalized["summary"]["gates"], sort_keys=True))
    if not finalized["summary"]["all_gates_passed"]:
        raise SystemExit(1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--source", type=Path, required=True)
    measure.add_argument("--mnsol-protocol", type=Path, required=True)
    measure.add_argument("--prior-private", type=Path, required=True)
    measure.add_argument("--selection-manifest", type=Path, required=True)
    measure.add_argument("--checkpoint", type=Path, required=True)
    measure.add_argument(
        "--force-protocol",
        type=Path,
        default=(
            BENCHMARK_DIR
            / "route2-aimnet2-smooth-ddpcm-force-prequalification-protocol-v1.json"
        ),
    )
    measure.add_argument("--record-ordinal", type=int, required=True)
    measure.add_argument("--stage", choices=("gate0", "gate1", "gate2"), required=True)
    measure.add_argument(
        "--mode", choices=("directional", "full-cartesian"), required=True
    )
    measure.add_argument("--replicate", choices=("primary", "replay"), required=True)
    measure.add_argument("--output-dir", type=Path, required=True)
    measure.set_defaults(function=_measure_command)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--source", type=Path, required=True)
    finalize.add_argument("--mnsol-protocol", type=Path, required=True)
    finalize.add_argument("--prior-private", type=Path, required=True)
    finalize.add_argument("--selection-manifest", type=Path, required=True)
    finalize.add_argument("--checkpoint", type=Path, required=True)
    finalize.add_argument(
        "--force-protocol",
        type=Path,
        default=(
            BENCHMARK_DIR
            / "route2-aimnet2-smooth-ddpcm-force-prequalification-protocol-v1.json"
        ),
    )
    finalize.add_argument("--primary", type=Path, required=True)
    finalize.add_argument("--replay", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(function=_finalize_command)
    return parser


def main() -> None:
    args = _parser().parse_args()
    with single_threaded_numerics():
        args.function(args)


if __name__ == "__main__":
    main()
