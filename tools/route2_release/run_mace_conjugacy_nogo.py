#!/usr/bin/env python3
"""Run the preregistered real-checkpoint MACE energy/source no-go canary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np
from ase import Atoms

# Must be fixed before Torch creates a CUDA context.  The runtime record binds
# this deterministic setting to the evidence artifact.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter
from maple.solvation.release import (
    ENERGY_DIRECTIONAL_FD_ABSOLUTE_TOLERANCE_EV,
    ENERGY_DIRECTIONAL_FD_RELATIVE_TOLERANCE,
    JVP_VJP_DOT_ABSOLUTE_TOLERANCE,
    JVP_VJP_DOT_RELATIVE_TOLERANCE,
    RepositorySnapshot,
    analyze_mace_polar_conjugacy,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    tolerance_contract,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-conjugacy-nogo-real-checkpoint-v2"
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
FIELD_SCALE = 5.0e-3
FIELD_SEED = 20260814
FD_STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
STATE_IDS = ("zero-field", "nonzero-deterministic-field")
POST_PREREGISTRATION_AD_ABSOLUTE_TOLERANCE_EV = 2.0e-10
POST_PREREGISTRATION_AD_RELATIVE_TOLERANCE = 5.0e-9
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/release/conjugacy.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/coupling/exact_gto.py",
    "maple/solvation/coupling/metrics.py",
    "maple/solvation/coupling/spaces.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/calculator/extra_correction/implicit/gto_field_projection.py",
    "tools/route2_release/run_mace_conjugacy_nogo.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _water() -> Atoms:
    return Atoms(
        "H2O",
        positions=np.asarray(
            [
                [0.0000, 0.0000, 0.0000],
                [0.9572, 0.0000, 0.0000],
                [-0.2390, 0.9266, 0.0000],
            ],
            dtype=float,
        ),
        info={"charge": 0, "mult": 1},
    )


def _normalized(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("The deterministic canary direction is singular.")
    return result / norm


def _deterministic_fields(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(FIELD_SEED)
    field = FIELD_SCALE * _normalized(random.normal(size=shape))
    direction = _normalized(random.normal(size=shape))
    return field, direction


def _error_record(analytic: float, finite_difference: float) -> dict[str, object]:
    absolute = abs(analytic - finite_difference)
    relative = absolute / max(abs(analytic), abs(finite_difference), 1.0e-12)
    passed = (
        absolute <= ENERGY_DIRECTIONAL_FD_ABSOLUTE_TOLERANCE_EV
        or relative <= ENERGY_DIRECTIONAL_FD_RELATIVE_TOLERANCE
    )
    return {
        "analytic_eV": analytic,
        "finite_difference_eV": finite_difference,
        "absolute_error_eV": absolute,
        "relative_error": relative,
        "gate_passed": passed,
    }


def _dot_record(left: float, right: float) -> dict[str, object]:
    absolute = abs(left - right)
    relative = absolute / max(abs(left), abs(right), 1.0e-30)
    passed = (
        absolute <= JVP_VJP_DOT_ABSOLUTE_TOLERANCE
        or relative <= JVP_VJP_DOT_RELATIVE_TOLERANCE
    )
    return {
        "jvp_dot": left,
        "vjp_dot": right,
        "absolute_error": absolute,
        "relative_error": relative,
        "gate_passed": passed,
    }


def _energy_ad_record(reverse: float, forward: float) -> dict[str, object]:
    """Record, rather than reinterpret, two AD modes on the same scalar.

    This implementation-consistency diagnostic was added after the original
    preregistered central-FD gate exposed total-energy cancellation.  It is not
    retroactively labelled preregistered and does not change any frozen v1
    tolerance or the missing-subspace decision rule.  Its separate v2 bound
    was fixed from the observed forward/reverse AD roundoff envelope, not from
    the conjugacy witness, and is recorded as post-preregistration evidence.
    """

    absolute = abs(reverse - forward)
    relative = absolute / max(abs(reverse), abs(forward), 1.0e-30)
    return {
        "reverse_mode_eV": reverse,
        "forward_mode_eV": forward,
        "absolute_error_eV": absolute,
        "relative_error": relative,
        "implementation_consistent": (
            absolute <= POST_PREREGISTRATION_AD_ABSOLUTE_TOLERANCE_EV
            or relative <= POST_PREREGISTRATION_AD_RELATIVE_TOLERANCE
        ),
        "status": "post-preregistration-implementation-diagnostic",
    }


def _state_record(adapter, atoms: Atoms, field: np.ndarray, direction: np.ndarray):
    count = len(atoms)
    source_state = adapter.evaluate_source(atoms, field, need_fixed_field_forces=False)
    source = np.asarray(source_state.source, dtype=float).copy()
    gradient = adapter.intrinsic_energy_field_gradient(atoms, field)
    jacobian = adapter.dense_source_jacobian(atoms, field)
    result = analyze_mace_polar_conjugacy(
        intrinsic_energy_field_gradient=gradient,
        original_embedded_source=source,
        physical_field=field,
        source_jacobian=jacobian,
        source_embedding=mace_polar_learned_source_embedding_matrix(),
        pairing_block=adapter.field_space.pairing_metric.block,
        charge_weights=np.asarray(adapter.source_space.effective_charge_weights),
    )

    analytic = float(np.vdot(gradient, direction))
    forward_mode = adapter.intrinsic_energy_field_directional_derivative(
        atoms, field, direction
    )
    energy_ad = _energy_ad_record(analytic, forward_mode)
    if not energy_ad["implementation_consistent"]:
        raise RuntimeError(
            "Intrinsic-energy reverse/forward AD modes disagree; no-go "
            "measurement is not admissible."
        )
    finite_differences = []
    for step in FD_STEPS:
        plus = adapter.intrinsic_energy_ev(atoms, field + step * direction)
        minus = adapter.intrinsic_energy_ev(atoms, field - step * direction)
        finite_difference = (plus - minus) / (2.0 * step)
        finite_differences.append(
            {"step": step, **_error_record(analytic, finite_difference)}
        )
    fd_gate_passed = any(record["gate_passed"] for record in finite_differences)

    cotangent = _normalized(
        np.random.default_rng(FIELD_SEED + 1).normal(size=source.shape)
    )
    jvp = adapter.source_jvp(atoms, field, direction)
    vjp = adapter.source_vjp(atoms, field, cotangent)
    dot = _dot_record(float(np.vdot(jvp, cotangent)), float(np.vdot(direction, vjp)))
    if not dot["gate_passed"]:
        raise RuntimeError("Source JVP/VJP failed its preregistered dot-product gate.")

    matrix_jvp = (jacobian @ direction.reshape(-1)).reshape(count, -1)
    dense_jvp_max_abs_error = float(np.max(np.abs(matrix_jvp - jvp)))
    if dense_jvp_max_abs_error > 2.0e-10:
        raise RuntimeError("Dense audit Jacobian does not reproduce source_jvp().")

    gauge = np.linalg.solve(
        np.kron(np.eye(count), adapter.field_space.pairing_metric.block),
        np.tile(
            np.asarray(adapter.source_space.effective_charge_weights, dtype=float),
            count,
        ),
    ).reshape(field.shape)
    gauge_source_delta = (
        adapter.evaluate_source(
            atoms, field + 1.0e-3 * gauge, need_fixed_field_forces=False
        ).source
        - source
    )

    return {
        "field": field.tolist(),
        "field_l2": float(np.linalg.norm(field)),
        "intrinsic_energy_eV": adapter.intrinsic_energy_ev(atoms, field),
        "intrinsic_energy_field_gradient": gradient.tolist(),
        "original_embedded_source": source.tolist(),
        "source_jacobian_sha256": canonical_json_sha256(jacobian.tolist()),
        "source_jacobian_frobenius": float(np.linalg.norm(jacobian, ord="fro")),
        "dense_jvp_max_abs_error": dense_jvp_max_abs_error,
        "source_jvp_vjp": dot,
        "intrinsic_energy_directional_ad": energy_ad,
        "intrinsic_energy_directional_fd": finite_differences,
        "intrinsic_energy_directional_fd_gate_passed": fd_gate_passed,
        "intrinsic_energy_directional_fd_gate_interpretation": (
            "The frozen v1 FD threshold is retained. Failure is reported "
            "without aborting only when independent forward/reverse AD agrees; "
            "the FD result is not relabelled as a pass."
        ),
        "constant_potential_step": 1.0e-3,
        "constant_potential_source_delta_l2": float(np.linalg.norm(gauge_source_delta)),
        "analysis": result.as_dict(),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()
    import torch

    torch.manual_seed(FIELD_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIELD_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    atoms = _water()
    adapter = build_official_mace_polar_1_m_radial_gto_adapter(
        device=args.device,
        checkpoint_path=checkpoint,
    )
    nonzero_field, direction = _deterministic_fields(
        adapter.field_space.shape(len(atoms))
    )
    fields = (np.zeros_like(nonzero_field), nonzero_field)
    states = {
        state_id: _state_record(adapter, atoms, field, direction)
        for state_id, field in zip(STATE_IDS, fields, strict=True)
    }
    no_go = any(
        bool(record["analysis"]["no_go_witness_detected"]) for record in states.values()
    )
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-real-checkpoint-tier-v-conjugacy-nogo-canary",
        "status": "no-go-witness-detected" if no_go else "no-witness-in-tested-states",
        "claim_boundary": (
            "A material missing-subspace witness proves that the original "
            "checkpoint intrinsic energy and original four-channel source cannot "
            "both be retained in one scalar at the tested state. Absence at two "
            "states would not prove global conjugacy, passivity, uniqueness, "
            "conservative force, or production readiness."
        ),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "exact_command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": source_hashes,
        "checkpoint": checkpoint_record(checkpoint),
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": adapter.dtype,
        "model_provider_id": adapter.provider_id,
        "model_profile_id": adapter.model_profile_id,
        "model_configuration_sha256": adapter.configuration_sha256(),
        "model_provenance_sha256": adapter.provenance_sha256,
        "source_space": adapter.source_space.metadata(),
        "source_space_sha256": adapter.source_space.metadata_hash(),
        "field_space": adapter.field_space.metadata(),
        "field_space_sha256": adapter.field_space.metadata_hash(),
        "pairing_sha256": adapter.field_space.pairing_metric.metadata_hash(),
        "field_transform": adapter.field_transform.provenance,
        "field_transform_sha256": adapter.field_transform.configuration_sha256(),
        "source_embedding": mace_polar_learned_source_embedding_matrix().tolist(),
        "source_embedding_sha256": canonical_json_sha256(
            mace_polar_learned_source_embedding_matrix().tolist()
        ),
        "geometry": {
            "formula": atoms.get_chemical_formula(),
            "atomic_numbers": atoms.numbers.tolist(),
            "positions_A": atoms.positions.tolist(),
            "charge": 0,
            "multiplicity": 1,
        },
        "protocol": {
            "field_seed": FIELD_SEED,
            "field_scale": FIELD_SCALE,
            "fd_steps": list(FD_STEPS),
            "state_ids": list(STATE_IDS),
            "thresholds": tolerance_contract(),
            "post_preregistration_ad_crosscheck": {
                "absolute_eV": POST_PREREGISTRATION_AD_ABSOLUTE_TOLERANCE_EV,
                "relative": POST_PREREGISTRATION_AD_RELATIVE_TOLERANCE,
                "purpose": (
                    "implementation consistency after central differences of "
                    "the roughly 2 keV total energy showed cancellation"
                ),
                "not_a_replacement_for_frozen_fd_gate": True,
            },
            "signs_audited": [-1, 1],
            "scalar_semantics": (
                "checkpoint intrinsic field-conditioned energy only; no explicit "
                "source-field coupling assembled"
            ),
        },
        "states": states,
        "decision": {
            "no_go_witness_detected": no_go,
            "frozen_energy_fd_gate_passed_for_all_states": all(
                bool(record["intrinsic_energy_directional_fd_gate_passed"])
                for record in states.values()
            ),
            "forward_reverse_ad_consistent_for_all_states": all(
                bool(
                    record["intrinsic_energy_directional_ad"][
                        "implementation_consistent"
                    ]
                )
                for record in states.values()
            ),
            "original_energy_original_source_common_scalar": (
                "formally-ruled-out-by-counterexample"
                if no_go
                else "unproven-no-counterexample-in-two-tested-states"
            ),
            "tier_v_admitted": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {
            "protocol": payload["protocol"],
            "states": payload["states"],
            "decision": payload["decision"],
        }
    )
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_MACE_CONJUGACY_NOGO="
        + json.dumps(
            {
                "artifact": file_record,
                "measurement_sha256": payload["measurement_sha256"],
                "decision": payload["decision"],
                "capabilities": payload["capabilities"],
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
