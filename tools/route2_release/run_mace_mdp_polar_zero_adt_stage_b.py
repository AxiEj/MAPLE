#!/usr/bin/env python3
"""Run target-free Stage B for the v3 MDP/POLAR hybrid hypothesis.

The runner consumes only the twelve Stage-A geometries and the two frozen
checkpoints.  It performs deterministic five-start root replay, an independent
cold replay, a fixed rigid-motion replay, directional derivative checks, and a
small projected local-root diagnostic.  The projection is explicitly a
rejection diagnostic, not a proof of full-space or global uniqueness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools.route2_release import (  # noqa: E402
    create_mace_mdp_polar_zero_adt_stage_b_preregistration as creator,
)
from tools.route2_release import (  # noqa: E402
    run_mace_mdp_polar_zero_adt_coupled_screen as stage_a,
)

ARTIFACT_ID = "route2-mace-mdp-polar-zero-adt-stage-b-v1"
EXPECTED_PROFILE_ID = stage_a.EXPECTED_PROFILE_ID
PREREGISTRATION_KEYS = {
    "artifact",
    "schema_version",
    "status",
    "created_utc",
    "source_git_head",
    "source_worktree_dirty_at_lock",
    "stage_a_preregistration",
    "stage_a_aggregate",
    "inherited_stage_a_source_files_sha256",
    "stage_b_source_files_sha256",
    "inputs_sha256",
    "selection",
    "selection_sha256",
    "method",
    "stage_b_gates",
    "claim_boundary",
    "next_if_stage_b_passes",
    "next_if_stage_b_fails",
    "preregistration_sha256",
}
PASS_RECORD_KEYS = {
    "artifact",
    "schema_version",
    "status",
    "screen_index",
    "selection",
    "preregistration_file_sha256",
    "preregistration_artifact_sha256",
    "profile_id",
    "evaluator_configuration_sha256",
    "response_configuration_sha256",
    "geometry_sha256",
    "root",
    "linearization",
    "projected_local_root",
    "rigid_replay",
    "gates",
    "stage_b_passed",
    "runtime_identity",
    "runtime_identity_sha256",
    "claim_boundary",
    "record_sha256",
}
FAILURE_RECORD_KEYS = {
    "artifact",
    "schema_version",
    "status",
    "screen_index",
    "selection",
    "preregistration_file_sha256",
    "preregistration_artifact_sha256",
    "failure_type",
    "failure_message",
    "failure_traceback",
    "stage_b_passed",
    "runtime_identity",
    "runtime_identity_sha256",
    "claim_boundary",
    "record_sha256",
}
GATE_KEYS = {
    "base_five_start_residual",
    "base_five_start_energy_span",
    "base_five_start_field_contract_enforced",
    "exact_cold_replay",
    "jvp_vjp_dot",
    "finite_field_second_order",
    "sampled_state_map_gain",
    "projected_residual_nonsingular",
    "projected_residual_strongly_monotone",
    "rigid_five_start_residual",
    "rigid_five_start_energy_span",
    "rigid_five_start_field_contract_enforced",
    "rigid_energy_covariance",
    "rigid_source_covariance",
    "rigid_field_covariance",
}


def _evaluator_root_replay_field_atol_ev() -> float:
    """Return the bound enforced before an evaluator state can be built.

    Keep this import lazy so the Stage-B ``--help`` and preregistration parser
    do not pull in the optional scientific runtime merely to describe the CLI.
    """

    from maple.solvation.experimental.mace_mdp_polar_adt_ddx import (
        ROOT_REPLAY_FIELD_ATOL_EV,
    )

    return float(ROOT_REPLAY_FIELD_ATOL_EV)


def _validate_evaluator_field_replay_contract(
    gates: Mapping[str, object],
) -> float:
    """Bind the preregistered field bound to the actual solve contract."""

    declared = float(gates["maximum_five_start_field_span_eV"])
    implemented = _evaluator_root_replay_field_atol_ev()
    _require(
        np.isfinite(declared) and declared > 0.0 and declared == implemented,
        "Stage-B field-replay gate does not match the evaluator solve contract.",
    )
    return implemented


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return creator._canonical_sha256(payload)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def _relative_error(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(
            np.linalg.norm(left_array),
            np.linalg.norm(right_array),
            np.finfo(float).tiny,
        )
    )


def _relative_dot_defect(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def _rotate_raw_l1(values: object, rotation: object) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    transform = np.asarray(rotation, dtype=np.float64)
    result = np.array(source, copy=True)
    result[:, (3, 1, 2)] = source[:, (3, 1, 2)] @ transform.T
    return result


def _rotate_native_field(values: object, rotation: object) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    transform = np.asarray(rotation, dtype=np.float64)
    result = np.array(field, copy=True)
    for channels in ((4, 2, 3), (7, 5, 6)):
        result[:, channels] = field[:, channels] @ transform.T
    return result


def _validate_rotation(values: object) -> np.ndarray:
    rotation = np.asarray(values, dtype=np.float64)
    _require(
        rotation.shape == (3, 3)
        and np.all(np.isfinite(rotation))
        and np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-15, rtol=0.0)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-15, rtol=0.0),
        "Stage-B rigid rotation is not a proper orthogonal matrix.",
    )
    return rotation


def _load_preregistration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _require(set(payload) == PREREGISTRATION_KEYS, "Stage-B prereg schema changed.")
    digest = payload.pop("preregistration_sha256", None)
    _require(digest == _canonical_sha256(payload), "Stage-B prereg hash changed.")
    payload["preregistration_sha256"] = digest
    _require(payload.get("artifact") == creator.ARTIFACT_ID, "Wrong Stage-B prereg.")
    _require(payload.get("schema_version") == 1, "Wrong Stage-B schema version.")
    _require(
        payload.get("status")
        == "locked-after-stage-a-pass-before-any-stage-b-evaluation",
        "Stage B was not prospectively locked after a Stage-A pass.",
    )
    _require(payload.get("claim_boundary") == creator.CLAIM_BOUNDARY, "Claim changed.")
    _require(payload.get("stage_b_gates") == creator.STAGE_B_GATES, "Gates changed.")
    _validate_evaluator_field_replay_contract(payload["stage_b_gates"])
    selection = payload.get("selection")
    _require(
        isinstance(selection, list)
        and len(selection) == creator.EXPECTED_RECORD_COUNT
        and payload.get("selection_sha256") == _canonical_sha256(selection),
        "Stage-B selection changed.",
    )
    method = payload.get("method")
    _require(
        isinstance(method, dict)
        and method.get("profile_id") == EXPECTED_PROFILE_ID
        and method.get("solve_stage")
        == "five-start-cold-replay-rigid-finite-field-stage-b",
        "Stage-B method identity changed.",
    )
    _validate_rotation(method.get("rotation_matrix"))
    _require(
        method.get("translation_angstrom") == list(creator.TRANSLATION_ANGSTROM)
        and method.get("finite_field_steps") == list(creator.FINITE_FIELD_STEPS),
        "Stage-B displacement or finite-field contract changed.",
    )
    return payload


def _validate_lineage_and_inputs(
    prereg: Mapping[str, Any],
    *,
    dataset: Path,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
) -> None:
    for name, path in (
        ("spice_dataset", dataset),
        ("mace_mdp_checkpoint", mdp_checkpoint),
        ("mace_polar_checkpoint", polar_checkpoint),
    ):
        _require(
            _sha256_file(path) == prereg["inputs_sha256"].get(name),
            f"Stage-B input changed: {name}.",
        )
    for manifest_name in (
        "inherited_stage_a_source_files_sha256",
        "stage_b_source_files_sha256",
    ):
        manifest = prereg.get(manifest_name)
        _require(isinstance(manifest, dict) and manifest, f"Missing {manifest_name}.")
        for name, expected in manifest.items():
            _require(
                _sha256_file((SOURCE_ROOT / name).resolve(strict=True)) == expected,
                f"Bound Stage-B source changed: {name}.",
            )

    stage_a_prereg_binding = prereg["stage_a_preregistration"]
    stage_a_aggregate_binding = prereg["stage_a_aggregate"]
    _require(
        set(stage_a_prereg_binding) == {"path", "file_sha256", "artifact_sha256"},
        "Stage-A preregistration binding schema changed.",
    )
    _require(
        set(stage_a_aggregate_binding)
        == {
            "path",
            "file_sha256",
            "aggregate_sha256",
            "records_path",
            "record_files_sha256",
            "record_sha256s",
            "runtime_identity_sha256",
        },
        "Stage-A aggregate binding schema changed.",
    )
    stage_a_prereg_path = Path(stage_a_prereg_binding["path"]).resolve(strict=True)
    stage_a_aggregate_path = Path(stage_a_aggregate_binding["path"]).resolve(
        strict=True
    )
    stage_a_records_dir = Path(stage_a_aggregate_binding["records_path"]).resolve(
        strict=True
    )
    _require(
        _sha256_file(stage_a_prereg_path) == stage_a_prereg_binding["file_sha256"],
        "Stage-A preregistration bytes changed.",
    )
    _require(
        _sha256_file(stage_a_aggregate_path)
        == stage_a_aggregate_binding["file_sha256"],
        "Stage-A aggregate bytes changed.",
    )
    stage_a_prereg = stage_a._load_preregistration(stage_a_prereg_path)
    stage_a._validate_source_and_inputs(
        stage_a_prereg,
        dataset=dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    _require(
        stage_a_prereg["preregistration_sha256"]
        == stage_a_prereg_binding["artifact_sha256"],
        "Stage-A preregistration identity changed.",
    )
    aggregate = json.loads(stage_a_aggregate_path.read_text())
    aggregate_sha = creator.validate_stage_a_aggregate(
        aggregate,
        stage_a_prereg=stage_a_prereg,
    )
    _require(
        aggregate_sha == stage_a_aggregate_binding["aggregate_sha256"]
        and aggregate["record_sha256s"] == stage_a_aggregate_binding["record_sha256s"],
        "Stage-A aggregate lineage changed.",
    )
    record_files = stage_a_aggregate_binding["record_files_sha256"]
    expected_names = {
        f"record-{index:03d}.json" for index in range(creator.EXPECTED_RECORD_COUNT)
    }
    _require(
        isinstance(record_files, dict)
        and set(record_files) == expected_names
        and {path.name for path in stage_a_records_dir.glob("record-*.json")}
        == expected_names,
        "Stage-A record-file closure changed.",
    )
    for name, expected in record_files.items():
        _require(
            _sha256_file(stage_a_records_dir / name) == expected,
            f"Stage-A record bytes changed: {name}.",
        )
    with tempfile.TemporaryDirectory(prefix="maple-stage-a-reaggregate-") as temporary:
        regenerated = stage_a.aggregate(
            SimpleNamespace(
                preregistration=stage_a_prereg_path,
                dataset=dataset,
                mdp_checkpoint=mdp_checkpoint,
                polar_checkpoint=polar_checkpoint,
                output_dir=stage_a_records_dir,
                aggregate_output=Path(temporary) / "aggregate.json",
            )
        )
    _require(
        regenerated == aggregate,
        "Stage-A aggregate no longer regenerates from its twelve records.",
    )


def _projected_root_diagnostic(
    *,
    state_map_jvp: Callable[[np.ndarray], np.ndarray],
    field_shape: tuple[int, int],
    uniform_native_basis: object,
    random_seed: int,
    random_probe_count: int,
) -> dict[str, object]:
    """Return a deterministic projected rejection diagnostic.

    No conclusion about unsampled directions or global uniqueness is made.
    """

    dimension = int(np.prod(field_shape))
    uniform = np.asarray(uniform_native_basis, dtype=np.float64).reshape(dimension, 3)
    rng = np.random.default_rng(random_seed)
    random = rng.normal(size=(dimension, random_probe_count))
    trial = np.column_stack((uniform, random))
    basis, upper = np.linalg.qr(trial, mode="reduced")
    _require(
        basis.shape == (dimension, 3 + random_probe_count)
        and np.min(np.abs(np.diag(upper))) > 1.0e-12,
        "Projected local-root probes lost rank.",
    )
    mapped = np.column_stack(
        [
            np.asarray(state_map_jvp(basis[:, index]), dtype=np.float64).reshape(-1)
            for index in range(basis.shape[1])
        ]
    )
    _require(mapped.shape == basis.shape and np.all(np.isfinite(mapped)), "Bad JVP.")
    projected_map = basis.T @ mapped
    projected_residual = np.eye(basis.shape[1]) - projected_map
    singular_values = np.linalg.svd(projected_residual, compute_uv=False)
    symmetric = 0.5 * (projected_residual + projected_residual.T)
    symmetric_eigenvalues = np.linalg.eigvalsh(symmetric)
    gains = np.linalg.norm(mapped, axis=0)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(projected_map))))
    return {
        "full_dimension": dimension,
        "projected_dimension": basis.shape[1],
        "random_seed": random_seed,
        "random_probe_count": random_probe_count,
        "maximum_sampled_state_map_gain": float(np.max(gains)),
        "minimum_projected_residual_singular_value": float(np.min(singular_values)),
        "maximum_projected_residual_singular_value": float(np.max(singular_values)),
        "minimum_projected_residual_symmetric_eigenvalue": float(
            np.min(symmetric_eigenvalues)
        ),
        "projected_state_map_spectral_radius": spectral_radius,
        "claim_boundary": (
            "Fixed uniform plus random projection; rejection diagnostic only, not "
            "a full-space local certificate or global uniqueness proof."
        ),
    }


def _root_summary(
    state: object, *, evaluator_field_replay_atol_ev: float
) -> dict[str, object]:
    starts = state.root_starts
    energies = [float(item.polarization_energy_ev) for item in starts]
    residuals = [float(item.residual_norm_ev) for item in starts]
    return {
        "root_sha256": state.root_sha256,
        "primal_residual_eV": float(state.primal_residual_ev),
        "polarization_energy_eV": float(state.polarization_energy_ev),
        "total_energy_eV": float(state.total_energy_ev),
        "maximum_start_residual_eV": max(residuals),
        "start_energy_span_eV": max(energies) - min(energies),
        "field_replay_contract": {
            "maximum_native_field_component_difference_eV": (
                evaluator_field_replay_atol_ev
            ),
            "enforced_inside_evaluator_solve": True,
            "achieved_span_serialized": False,
            "evidence_boundary": (
                "The evaluator raises before state construction when any of the "
                "five converged starts differs from the reference beyond this "
                "bound.  CanonicalADTRootStart stores only converged-field hashes, "
                "so this record does not claim an independently measured span."
            ),
        },
        "starts": [
            {
                "label": item.label,
                "iterations": item.iterations,
                "residual_eV": item.residual_norm_ev,
                "polarization_energy_eV": item.polarization_energy_ev,
                "record_sha256": item.record_sha256,
            }
            for item in starts
        ],
    }


def _run_one(
    *,
    selected: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    response: object,
    prereg: Mapping[str, Any],
    prereg_file_sha256: str,
    runtime: Mapping[str, object],
    runtime_sha256: str,
) -> dict[str, object]:
    from ase import Atoms

    atoms = Atoms(
        numbers=arrays["numbers"],
        positions=arrays["positions_angstrom"],
        info={"charge": 0, "multiplicity": 1},
    )
    evaluator = stage_a._build_evaluator(atoms, response)
    state = evaluator.solve(atoms)
    replay = evaluator.solve(atoms)
    gates_config = prereg["stage_b_gates"]
    field_replay_atol_ev = _validate_evaluator_field_replay_contract(gates_config)

    rng = np.random.default_rng(
        creator.PROJECTED_ROOT_RANDOM_SEED + int(selected["screen_index"])
    )
    direction = rng.normal(size=state.native_field8.shape)
    cotangent = rng.normal(size=state.native_field8.shape)
    jvp = evaluator.state_map_jvp(atoms, state.native_field8, direction)
    vjp = evaluator.state_map_vjp(atoms, state.native_field8, cotangent)
    dot_left = float(np.vdot(cotangent, jvp))
    dot_right = float(np.vdot(vjp, direction))
    dot_relative = _relative_dot_defect(dot_left, dot_right)
    finite_errors = []
    for step in creator.FINITE_FIELD_STEPS:
        plus = evaluator.state_map(atoms, state.native_field8 + step * direction)[0]
        minus = evaluator.state_map(atoms, state.native_field8 - step * direction)[0]
        finite_errors.append(float(np.max(np.abs((plus - minus) / (2.0 * step) - jvp))))

    chart = response.chart_for_geometry(atoms)
    projected = _projected_root_diagnostic(
        state_map_jvp=lambda flat: evaluator.state_map_jvp(
            atoms,
            state.native_field8,
            np.asarray(flat, dtype=np.float64).reshape(state.native_field8.shape),
        ).reshape(-1),
        field_shape=state.native_field8.shape,
        uniform_native_basis=chart.uniform_native_basis,
        random_seed=creator.PROJECTED_ROOT_RANDOM_SEED + int(selected["screen_index"]),
        random_probe_count=creator.PROJECTED_RANDOM_PROBE_COUNT,
    )

    rotation = _validate_rotation(prereg["method"]["rotation_matrix"])
    translation = np.asarray(prereg["method"]["translation_angstrom"], dtype=float)
    moved = atoms.copy()
    center = np.mean(atoms.positions, axis=0)
    moved.positions = (atoms.positions - center) @ rotation.T + center + translation
    moved.info.update(atoms.info)
    moved_evaluator = stage_a._build_evaluator(moved, response)
    moved_state = moved_evaluator.solve(moved)
    source_errors = {
        "permanent_source_relative": _relative_error(
            moved_state.permanent_source4,
            _rotate_raw_l1(state.permanent_source4, rotation),
        ),
        "radial_residual_source_relative": _relative_error(
            moved_state.radial_residual_source4,
            _rotate_raw_l1(state.radial_residual_source4, rotation),
        ),
        "adt_atomic_dipoles_relative": _relative_error(
            moved_state.adt_atomic_dipoles_eangstrom,
            state.adt_atomic_dipoles_eangstrom @ rotation.T,
        ),
    }
    field_error = _relative_error(
        moved_state.native_field8,
        _rotate_native_field(state.native_field8, rotation),
    )
    total_energy_error = abs(moved_state.total_energy_ev - state.total_energy_ev)
    polarization_energy_error = abs(
        moved_state.polarization_energy_ev - state.polarization_energy_ev
    )
    base_root = _root_summary(
        state, evaluator_field_replay_atol_ev=field_replay_atol_ev
    )
    moved_root = _root_summary(
        moved_state, evaluator_field_replay_atol_ev=field_replay_atol_ev
    )
    cold_field_error = float(np.max(np.abs(replay.native_field8 - state.native_field8)))
    cold_energy_error = abs(replay.total_energy_ev - state.total_energy_ev)

    finite_decreases = all(
        left > right for left, right in zip(finite_errors, finite_errors[1:])
    )
    gates = {
        "base_five_start_residual": base_root["maximum_start_residual_eV"]
        <= gates_config["maximum_five_start_residual_eV"],
        "base_five_start_energy_span": base_root["start_energy_span_eV"]
        <= gates_config["maximum_five_start_energy_span_eV"],
        "base_five_start_field_contract_enforced": base_root["field_replay_contract"][
            "maximum_native_field_component_difference_eV"
        ]
        == gates_config["maximum_five_start_field_span_eV"],
        "exact_cold_replay": replay.root_sha256 == state.root_sha256,
        "jvp_vjp_dot": dot_relative <= gates_config["maximum_jvp_vjp_relative_defect"],
        "finite_field_second_order": finite_decreases
        and finite_errors[-1] <= gates_config["maximum_finest_finite_field_error"],
        "sampled_state_map_gain": projected["maximum_sampled_state_map_gain"]
        <= gates_config["maximum_sampled_state_map_gain"],
        "projected_residual_nonsingular": projected[
            "minimum_projected_residual_singular_value"
        ]
        >= gates_config["minimum_projected_residual_singular_value"],
        "projected_residual_strongly_monotone": projected[
            "minimum_projected_residual_symmetric_eigenvalue"
        ]
        >= gates_config["minimum_projected_residual_symmetric_margin"],
        "rigid_five_start_residual": moved_root["maximum_start_residual_eV"]
        <= gates_config["maximum_five_start_residual_eV"],
        "rigid_five_start_energy_span": moved_root["start_energy_span_eV"]
        <= gates_config["maximum_five_start_energy_span_eV"],
        "rigid_five_start_field_contract_enforced": moved_root["field_replay_contract"][
            "maximum_native_field_component_difference_eV"
        ]
        == gates_config["maximum_five_start_field_span_eV"],
        "rigid_energy_covariance": total_energy_error
        <= gates_config["maximum_rigid_total_energy_error_eV"]
        and polarization_energy_error
        <= gates_config["maximum_rigid_polarization_energy_error_eV"],
        "rigid_source_covariance": max(source_errors.values())
        <= gates_config["maximum_rigid_source_relative_error"],
        "rigid_field_covariance": field_error
        <= gates_config["maximum_rigid_field_relative_error"],
    }
    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-record",
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "gate-failure",
        "screen_index": selected["screen_index"],
        "selection": dict(selected),
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_artifact_sha256": prereg["preregistration_sha256"],
        "profile_id": evaluator.profile_id,
        "evaluator_configuration_sha256": evaluator.configuration_sha256(),
        "response_configuration_sha256": response.configuration_sha256(),
        "geometry_sha256": state.geometry_sha256,
        "root": {
            "base": base_root,
            "cold_replay_root_sha256": replay.root_sha256,
            "cold_replay_maximum_field_error": cold_field_error,
            "cold_replay_total_energy_error_eV": cold_energy_error,
        },
        "linearization": {
            "jvp_vjp_dot_left": dot_left,
            "jvp_vjp_dot_right": dot_right,
            "jvp_vjp_relative_defect": dot_relative,
            "finite_field_steps": list(creator.FINITE_FIELD_STEPS),
            "finite_field_maximum_errors": finite_errors,
        },
        "projected_local_root": projected,
        "rigid_replay": {
            "rotation_matrix": rotation.tolist(),
            "translation_angstrom": translation.tolist(),
            "root": moved_root,
            "total_energy_absolute_error_eV": total_energy_error,
            "polarization_energy_absolute_error_eV": polarization_energy_error,
            **source_errors,
            "native_field_relative": field_error,
        },
        "gates": gates,
        "stage_b_passed": all(gates.values()),
        "runtime_identity": dict(runtime),
        "runtime_identity_sha256": runtime_sha256,
        "claim_boundary": creator.CLAIM_BOUNDARY,
    }
    payload["record_sha256"] = _canonical_sha256(payload)
    return payload


def _failure_record(
    *,
    selected: Mapping[str, Any],
    prereg: Mapping[str, Any],
    prereg_file_sha256: str,
    runtime: Mapping[str, object],
    runtime_sha256: str,
    error: Exception,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-record",
        "schema_version": 1,
        "status": "provider-failure",
        "screen_index": selected["screen_index"],
        "selection": dict(selected),
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_artifact_sha256": prereg["preregistration_sha256"],
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "failure_traceback": traceback.format_exc(),
        "stage_b_passed": False,
        "runtime_identity": dict(runtime),
        "runtime_identity_sha256": runtime_sha256,
        "claim_boundary": creator.CLAIM_BOUNDARY,
    }
    payload["record_sha256"] = _canonical_sha256(payload)
    return payload


def run_records(args: argparse.Namespace) -> None:
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    prereg = _load_preregistration(prereg_path)
    dataset = args.dataset.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    _validate_lineage_and_inputs(
        prereg,
        dataset=dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    if not 0 <= args.start <= args.stop <= creator.EXPECTED_RECORD_COUNT:
        raise ValueError("Stage-B record range is outside the frozen panel.")
    stage_a._configure_torch(args.device)
    _, response = stage_a._build_models(
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
        device=args.device,
    )
    runtime, runtime_sha = stage_a._runtime_identity(args.device)
    _require(
        runtime_sha == prereg["stage_a_aggregate"]["runtime_identity_sha256"],
        "Stage B changed the Stage-A numerical runtime.",
    )
    prereg_file_sha = _sha256_file(prereg_path)
    for index in range(args.start, args.stop):
        selected = prereg["selection"][index]
        try:
            arrays = stage_a._load_geometry_arrays(dataset, selected)
            payload = _run_one(
                selected=selected,
                arrays=arrays,
                response=response,
                prereg=prereg,
                prereg_file_sha256=prereg_file_sha,
                runtime=runtime,
                runtime_sha256=runtime_sha,
            )
        except Exception as error:  # failure is evidence, never a silent skip
            payload = _failure_record(
                selected=selected,
                prereg=prereg,
                prereg_file_sha256=prereg_file_sha,
                runtime=runtime,
                runtime_sha256=runtime_sha,
                error=error,
            )
        _write_json_exclusive(
            args.output_dir.expanduser().resolve() / f"record-{index:03d}.json",
            payload,
        )
        print(
            f"[{index + 1}/{creator.EXPECTED_RECORD_COUNT}] "
            f"source_index={selected['source_record_index']} "
            f"molecule={selected['selection_identity']['molecule']} "
            f"status={payload['status']}",
            flush=True,
        )


def _read_records(
    *, prereg: Mapping[str, Any], prereg_path: Path, records_dir: Path
) -> list[dict[str, Any]]:
    expected_names = {
        f"record-{index:03d}.json" for index in range(creator.EXPECTED_RECORD_COUNT)
    }
    observed_names = {path.name for path in records_dir.glob("record-*.json")}
    _require(observed_names == expected_names, "Stage-B twelve-record closure failed.")
    prereg_file_sha = _sha256_file(prereg_path)
    records = []
    for index in range(creator.EXPECTED_RECORD_COUNT):
        payload = json.loads((records_dir / f"record-{index:03d}.json").read_text())
        digest = payload.pop("record_sha256", None)
        _require(digest == _canonical_sha256(payload), f"Record {index} hash changed.")
        payload["record_sha256"] = digest
        status = payload.get("status")
        _require(status in {"pass", "gate-failure", "provider-failure"}, "Bad status.")
        expected_keys = (
            PASS_RECORD_KEYS
            if status in {"pass", "gate-failure"}
            else FAILURE_RECORD_KEYS
        )
        _require(set(payload) == expected_keys, f"Record {index} schema changed.")
        _require(
            payload.get("screen_index") == index
            and payload.get("selection") == prereg["selection"][index]
            and payload.get("preregistration_file_sha256") == prereg_file_sha
            and payload.get("preregistration_artifact_sha256")
            == prereg["preregistration_sha256"]
            and payload.get("claim_boundary") == creator.CLAIM_BOUNDARY,
            f"Record {index} binding changed.",
        )
        if status in {"pass", "gate-failure"}:
            gates = payload.get("gates")
            gate_passed = (
                isinstance(gates, dict)
                and set(gates) == GATE_KEYS
                and all(value is True for value in gates.values())
            )
            _require(
                payload.get("profile_id") == EXPECTED_PROFILE_ID
                and payload.get("stage_b_passed") is gate_passed
                and (status == "pass") is gate_passed,
                f"Record {index} gate status is inconsistent.",
            )
        else:
            _require(payload.get("stage_b_passed") is False, "Failure claimed pass.")
        _require(
            payload.get("runtime_identity_sha256")
            == _canonical_sha256(payload.get("runtime_identity")),
            f"Record {index} runtime hash changed.",
        )
        records.append(payload)
    return records


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    prereg = _load_preregistration(prereg_path)
    dataset = args.dataset.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    _validate_lineage_and_inputs(
        prereg,
        dataset=dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    records = _read_records(
        prereg=prereg,
        prereg_path=prereg_path,
        records_dir=args.output_dir.expanduser().resolve(strict=True),
    )
    runtime_digests = {item["runtime_identity_sha256"] for item in records}
    _require(len(runtime_digests) == 1, "Stage B mixed numerical runtimes.")
    passed = [item for item in records if item["status"] == "pass"]
    all_passed = len(passed) == creator.EXPECTED_RECORD_COUNT

    def maximum(path: tuple[str, ...]) -> float | None:
        if not passed:
            return None
        values = []
        for item in passed:
            value: object = item
            for key in path:
                value = value[key]  # type: ignore[index]
            values.append(float(value))
        return max(values)

    def minimum(path: tuple[str, ...]) -> float | None:
        if not passed:
            return None
        values = []
        for item in passed:
            value: object = item
            for key in path:
                value = value[key]  # type: ignore[index]
            values.append(float(value))
        return min(values)

    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-aggregate",
        "schema_version": 1,
        "status": "pass-stage-b" if all_passed else "fail",
        "preregistration_file_sha256": _sha256_file(prereg_path),
        "preregistration_artifact_sha256": prereg["preregistration_sha256"],
        "record_count": len(records),
        "pass_count": len(passed),
        "failure_count": len(records) - len(passed),
        "runtime_identity_sha256": next(iter(runtime_digests)),
        "summary": {
            "maximum_base_start_residual_eV": maximum(
                ("root", "base", "maximum_start_residual_eV")
            ),
            "maximum_base_start_energy_span_eV": maximum(
                ("root", "base", "start_energy_span_eV")
            ),
            "evaluator_root_replay_field_atol_eV": maximum(
                (
                    "root",
                    "base",
                    "field_replay_contract",
                    "maximum_native_field_component_difference_eV",
                )
            ),
            "maximum_jvp_vjp_relative_defect": maximum(
                ("linearization", "jvp_vjp_relative_defect")
            ),
            "maximum_finest_finite_field_error": max(
                (
                    float(item["linearization"]["finite_field_maximum_errors"][-1])
                    for item in passed
                ),
                default=None,
            ),
            "maximum_sampled_state_map_gain": maximum(
                ("projected_local_root", "maximum_sampled_state_map_gain")
            ),
            "minimum_projected_residual_singular_value": minimum(
                (
                    "projected_local_root",
                    "minimum_projected_residual_singular_value",
                )
            ),
            "minimum_projected_residual_symmetric_eigenvalue": minimum(
                (
                    "projected_local_root",
                    "minimum_projected_residual_symmetric_eigenvalue",
                )
            ),
            "maximum_rigid_total_energy_error_eV": maximum(
                ("rigid_replay", "total_energy_absolute_error_eV")
            ),
            "maximum_rigid_field_relative_error": maximum(
                ("rigid_replay", "native_field_relative")
            ),
        },
        "stage_b_passed": all_passed,
        "next_step": (
            prereg["next_if_stage_b_passes"]
            if all_passed
            else prereg["next_if_stage_b_fails"]
        ),
        "accuracy_claim_made": False,
        "capabilities_admitted": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "claim_boundary": creator.CLAIM_BOUNDARY,
        "record_sha256s": [item["record_sha256"] for item in records],
    }
    payload["aggregate_sha256"] = _canonical_sha256(payload)
    _write_json_exclusive(args.aggregate_output.expanduser().resolve(), payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("records", "aggregate"), required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=creator.EXPECTED_RECORD_COUNT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "records":
        run_records(args)
        return
    if args.aggregate_output is None:
        raise ValueError("--aggregate-output is required in aggregate mode.")
    print(json.dumps(aggregate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
