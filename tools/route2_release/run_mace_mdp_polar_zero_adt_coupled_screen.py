#!/usr/bin/env python3
"""Run the source-selected v3 zero-start coupled mechanism screen.

This stage reads only frozen SPICE geometries.  The QM source/MEP diagnostics
were used upstream to select twelve adversarial cases, but no labels are loaded
while solving the coupled MACE-POLAR-zero/MDP-alpha/ADT/ddX state.  Stage A is
rejection-only and cannot admit E/F/H/V/M or chemical accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools.route2_release import (  # noqa: E402
    create_mace_mdp_polar_zero_adt_coupled_screen_preregistration as creator,
)

ARTIFACT_ID = "route2-mace-mdp-polar-zero-adt-coupled-screen-v1"
PREREGISTRATION_RELATIVE = Path(
    "docs/route2/preregistrations/mace-mdp-polar-zero-adt-coupled-screen-v2.json"
)
EXPECTED_PROFILE_ID = (
    "route2-research-macepolar-zero-point-mdp-alpha-polar-residual-"
    "role-separated-adt-ddx-operational-v3"
)
PREREGISTRATION_KEYS = {
    "artifact",
    "schema_version",
    "status",
    "created_utc",
    "source_git_head",
    "source_worktree_dirty_at_lock",
    "source_files_sha256",
    "superseded_preregistration",
    "provider_remediation",
    "source_evidence",
    "inputs_sha256",
    "selection_policy",
    "selection",
    "selection_sha256",
    "method",
    "stage_a_gates",
    "claim_boundary",
    "next_if_stage_a_passes",
    "next_if_stage_a_fails",
    "preregistration_sha256",
}
SELECTION_KEYS = {
    "screen_index",
    "source_record_index",
    "source_record_sha256",
    "selection_identity",
    "selection_category",
    "selection_metric",
    "selection_metric_value",
    "selection_rank_before_duplicate_filter",
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
    "screen_state_sha256",
    "continuum_state_sha256",
    "root",
    "charge_closure",
    "uniform_chart",
    "linearization",
    "gates",
    "stage_a_passed",
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
    "stage_a_passed",
    "runtime_identity",
    "runtime_identity_sha256",
    "claim_boundary",
    "record_sha256",
}
ROOT_KEYS = {
    "iterations",
    "primal_residual_eV",
    "polarization_energy_eV_diagnostic_only",
    "native_field_l2",
    "radial_residual_source_l2",
    "adt_atomic_dipoles_l2_eangstrom",
}
CHARGE_KEYS = {"permanent_charge_e", "radial_response_charge_e"}
UNIFORM_CHART_KEYS = {
    "left_inverse_maximum_absolute_error",
    "zero_field_uniform_radial_cancellation_maximum_absolute",
    "zero_field_adt_molecular_closure_l2_eangstrom",
    "root_adt_molecular_closure_l2_eangstrom",
    "molecular_alpha_eigenvalues",
}
LINEARIZATION_KEYS = {
    "jvp_vjp_dot_left",
    "jvp_vjp_dot_right",
    "jvp_vjp_absolute_defect",
    "jvp_vjp_relative_defect",
}
GATE_KEYS = {
    "profile_identity",
    "root_residual",
    "permanent_charge",
    "radial_response_charge",
    "jvp_vjp_dot",
    "uniform_left_inverse",
    "zero_field_uniform_radial_cancellation",
    "zero_field_adt_molecular_closure",
    "root_adt_molecular_closure",
    "molecular_alpha_passive",
    "permanent_is_exact_polar_zero_source",
}
RUNTIME_IDENTITY_KEYS = {
    "python",
    "implementation",
    "platform",
    "machine",
    "cpu_model",
    "packages",
    "numpy",
    "torch",
    "environment",
    "selected_mace_polar_device",
}
RANDOM_SEED = 20260819
UNIFORM_AUDIT_VECTOR = np.asarray([0.37, -0.51, 0.78], dtype=np.float64)


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


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(
        json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\0"
        + array.tobytes()
    ).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def _load_preregistration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _require(
        set(payload) == PREREGISTRATION_KEYS,
        "Preregistration schema contains missing or unregistered fields.",
    )
    digest = payload.pop("preregistration_sha256", None)
    _require(digest == _canonical_sha256(payload), "Preregistration self hash changed.")
    payload["preregistration_sha256"] = digest
    _require(
        payload.get("artifact") == creator.ARTIFACT_ID,
        "Wrong preregistration artifact.",
    )
    _require(payload.get("schema_version") == 1, "Wrong preregistration schema.")
    _require(
        payload.get("status")
        == "locked-after-array-identity-remediation-before-any-selected-v3-"
        "coupled-state-evaluation",
        "Coupled screen was not prospectively locked.",
    )
    _require(
        payload.get("claim_boundary") == creator.CLAIM_BOUNDARY,
        "Claim boundary changed.",
    )
    method = payload.get("method")
    _require(isinstance(method, dict), "Missing method declaration.")
    _require(
        method.get("profile_id") == EXPECTED_PROFILE_ID, "Wrong v3 profile identity."
    )
    _require(
        method.get("lmax") == 12
        and method.get("n_lebedev") == 1202
        and method.get("solve_stage") == "one-zero-start-rejection-screen",
        "Stage-A numerical method changed.",
    )
    selection = payload.get("selection")
    _require(
        isinstance(selection, list)
        and len(selection) == creator.EXPECTED_SCREEN_RECORD_COUNT,
        "Screen selection must contain exactly twelve records.",
    )
    _require(
        payload.get("selection_sha256") == _canonical_sha256(selection),
        "Screen selection digest changed.",
    )
    _require(
        [item.get("screen_index") for item in selection]
        == list(range(creator.EXPECTED_SCREEN_RECORD_COUNT)),
        "Screen indices are not canonical.",
    )
    _require(
        all(set(item) == SELECTION_KEYS for item in selection),
        "Selection records contain missing or unregistered fields.",
    )
    return payload


def _validate_source_and_inputs(
    prereg: Mapping[str, Any],
    *,
    dataset: Path,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
) -> None:
    input_hashes = prereg.get("inputs_sha256")
    _require(isinstance(input_hashes, dict), "Missing input hashes.")
    for name, path in (
        ("spice_dataset", dataset),
        ("mace_mdp_checkpoint", mdp_checkpoint),
        ("mace_polar_checkpoint", polar_checkpoint),
    ):
        _require(
            _sha256_file(path) == input_hashes.get(name),
            f"Input hash mismatch: {name}.",
        )

    source_hashes = prereg.get("source_files_sha256")
    _require(isinstance(source_hashes, dict), "Missing source-file manifest.")
    _require(
        set(source_hashes) == set(creator.SOURCE_FILES),
        "Source-file manifest is incomplete.",
    )
    for name, expected in source_hashes.items():
        _require(
            _sha256_file((SOURCE_ROOT / name).resolve(strict=True)) == expected,
            f"Bound source file changed: {name}.",
        )

    evidence = prereg.get("source_evidence")
    _require(isinstance(evidence, dict), "Missing source-evidence binding.")
    evidence_root = (SOURCE_ROOT / evidence["relative_path"]).resolve(strict=True)
    aggregate_path = evidence_root / "aggregate.json"
    _require(
        _sha256_file(aggregate_path) == evidence.get("aggregate_file_sha256"),
        "Source aggregate bytes changed.",
    )
    record_hashes = evidence.get("record_files_sha256")
    _require(
        isinstance(record_hashes, dict)
        and len(record_hashes) == creator.EXPECTED_SOURCE_RECORD_COUNT,
        "Source-record file manifest is incomplete.",
    )
    for name, expected in record_hashes.items():
        _require(
            _sha256_file(evidence_root / "records" / name) == expected,
            f"Source evidence record changed: {name}.",
        )
    records = creator.load_source_records(evidence_root)
    for selected in prereg["selection"]:
        source = records[int(selected["source_record_index"])]
        _require(
            source["record_sha256"] == selected["source_record_sha256"]
            and source["selection_identity"] == selected["selection_identity"],
            "Selected geometry no longer matches its source evidence.",
        )

    superseded = prereg.get("superseded_preregistration")
    remediation = prereg.get("provider_remediation")
    _require(
        isinstance(superseded, dict) and isinstance(remediation, dict),
        "Missing exact provider-remediation lineage.",
    )
    superseded_path = (SOURCE_ROOT / superseded["relative_path"]).resolve(strict=True)
    remediation_path = (SOURCE_ROOT / remediation["relative_path"]).resolve(strict=True)
    _require(
        _sha256_file(superseded_path) == superseded.get("file_sha256")
        and _sha256_file(remediation_path) == remediation.get("file_sha256")
        and superseded.get("selection_and_method_unchanged") is True
        and remediation.get("failed_before_coupled_state_evaluation") is True
        and remediation.get("scientific_configuration_changed") is False
        and remediation.get("selection_changed") is False,
        "Provider-remediation lineage changed.",
    )


def _load_geometry_arrays(
    dataset: Path,
    selected: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Load only geometry identity; no MBIS or other QM label is read."""

    import h5py

    identity = selected["selection_identity"]
    molecule = identity["molecule"]
    configuration = int(identity["configuration"])
    with h5py.File(dataset, "r") as handle:
        _require(molecule in handle, f"Missing SPICE molecule {molecule}.")
        group = handle[molecule]
        numbers = np.asarray(group["atomic_numbers"], dtype=np.int64).reshape(-1)
        positions_raw = np.asarray(group["positions"][configuration])
    _require(
        _array_sha256(numbers) == identity["atomic_numbers_sha256"],
        "Atomic-number identity changed.",
    )
    _require(
        _array_sha256(positions_raw) == identity["positions_raw_sha256"],
        "Geometry identity changed.",
    )
    _require(len(numbers) == identity["atom_count"], "Atom count changed.")
    return {
        "numbers": numbers,
        "positions_angstrom": np.asarray(positions_raw, dtype=np.float64) * 10.0,
    }


def _configure_torch(device: str) -> object:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = __import__("torch")
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA coupled-screen path is unavailable.")
    return torch


def _runtime_identity(device: str) -> tuple[dict[str, object], str]:
    from maple.solvation.release import runtime_record

    runtime = runtime_record()
    runtime.pop("generated_at_utc", None)
    runtime["selected_mace_polar_device"] = device
    return runtime, _canonical_sha256(runtime)


def _build_models(
    *,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
    device: str,
) -> tuple[object, object]:
    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.models import (
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_moment_adapter,
        build_mdp_polar_role_separated_adt_response,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )

    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    base = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    response = build_mdp_polar_role_separated_adt_response(
        mdp=mdp,
        base=base,
        source_root=SOURCE_ROOT,
    )
    return mdp, response


def _build_evaluator(atoms: object, response: object) -> object:
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_water_coulomb_radii,
    )
    from maple.solvation.continuum import SeparatedSourceDDXBackend
    from maple.solvation.experimental.mace_mdp_polar_adt_ddx import (
        MACE_MDPPolarCanonicalADTDDXEnergy,
        POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
    )
    from maple.solvation.models import MACEPolarZeroFieldPointPermanentSource

    symbols = tuple(atoms.get_chemical_symbols())
    continuum = SeparatedSourceDDXBackend(
        symbols,
        smd_water_coulomb_radii(symbols),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=12,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
        eta=0.1,
        n_proc=1,
    )
    evaluator = MACE_MDPPolarCanonicalADTDDXEnergy(
        atoms,
        permanent=MACEPolarZeroFieldPointPermanentSource(response),
        response=response,
        continuum=continuum,
    )
    _require(
        evaluator.profile_id
        == POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
        == EXPECTED_PROFILE_ID,
        "Evaluator did not instantiate the exact v3 profile.",
    )
    return evaluator


def _relative_dot_defect(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


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
    from maple.solvation.experimental.mace_mdp_polar_adt_ddx import (
        ROOT_TOLERANCE_EV,
    )

    atoms = Atoms(
        numbers=arrays["numbers"],
        positions=arrays["positions_angstrom"],
        info={"charge": 0, "multiplicity": 1},
    )
    evaluator = _build_evaluator(atoms, response)
    state = evaluator.solve_zero_start_screen(atoms)
    chart = response.chart_for_geometry(atoms)

    rng = np.random.default_rng(RANDOM_SEED + int(selected["screen_index"]))
    direction = rng.normal(size=state.native_field8.shape)
    cotangent = rng.normal(size=state.native_field8.shape)
    jvp = evaluator.state_map_jvp(atoms, state.native_field8, direction)
    vjp = evaluator.state_map_vjp(atoms, state.native_field8, cotangent)
    dot_left = float(np.vdot(cotangent, jvp))
    dot_right = float(np.vdot(vjp, direction))
    dot_relative = _relative_dot_defect(dot_left, dot_right)

    basis_flat = chart.uniform_native_basis.reshape(chart.atom_count * 8, 3)
    left_inverse_error = float(
        np.max(np.abs(chart.uniform_gradient_left_inverse @ basis_flat - np.eye(3)))
    )
    uniform_direction = np.einsum(
        "nsc,c->ns", chart.uniform_native_basis, UNIFORM_AUDIT_VECTOR
    )
    zero_components = response.field_jvp_components(
        atoms,
        np.zeros_like(state.native_field8),
        uniform_direction,
    )
    zero_uniform_radial_cancellation = float(
        np.max(np.abs(zero_components.radial_residual_source4))
    )
    expected_adt_sum = -(
        chart.mdp_molecular_polarizability_eangstrom2_per_volt @ UNIFORM_AUDIT_VECTOR
    )
    zero_adt_closure = float(
        np.linalg.norm(
            np.sum(zero_components.adt_atomic_dipoles_eangstrom, axis=0)
            - expected_adt_sum
        )
    )
    root_coordinates = chart.uniform_coordinates(state.native_field8)
    root_expected_adt_sum = -(
        chart.mdp_molecular_polarizability_eangstrom2_per_volt @ root_coordinates
    )
    root_adt_closure = float(
        np.linalg.norm(
            np.sum(state.adt_atomic_dipoles_eangstrom, axis=0) - root_expected_adt_sum
        )
    )
    alpha_eigenvalues = np.linalg.eigvalsh(
        chart.mdp_molecular_polarizability_eangstrom2_per_volt
    )
    gates_config = prereg["stage_a_gates"]
    permanent_charge = float(np.sum(state.permanent_source4[:, 0]))
    radial_charge = float(np.sum(state.radial_residual_source4[:, 0]))
    gates = {
        "profile_identity": evaluator.profile_id == EXPECTED_PROFILE_ID,
        "root_residual": state.primal_residual_ev
        <= float(gates_config["maximum_primal_residual_eV"])
        <= ROOT_TOLERANCE_EV,
        "permanent_charge": abs(permanent_charge)
        <= float(gates_config["maximum_absolute_permanent_charge_e"]),
        "radial_response_charge": abs(radial_charge)
        <= float(gates_config["maximum_absolute_radial_response_charge_e"]),
        "jvp_vjp_dot": dot_relative
        <= float(gates_config["maximum_jvp_vjp_relative_defect"]),
        "uniform_left_inverse": left_inverse_error
        <= float(gates_config["maximum_uniform_left_inverse_error"]),
        "zero_field_uniform_radial_cancellation": (
            zero_uniform_radial_cancellation
            <= float(gates_config["maximum_zero_field_uniform_radial_cancellation"])
        ),
        "zero_field_adt_molecular_closure": zero_adt_closure
        <= float(gates_config["maximum_adt_molecular_dipole_closure_eangstrom"]),
        "root_adt_molecular_closure": root_adt_closure
        <= float(gates_config["maximum_adt_molecular_dipole_closure_eangstrom"]),
        "molecular_alpha_passive": float(np.min(alpha_eigenvalues))
        >= float(gates_config["minimum_mdp_alpha_eigenvalue"]),
        "permanent_is_exact_polar_zero_source": np.array_equal(
            state.permanent_source4, chart.polar_zero_source4
        ),
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
        "screen_state_sha256": state.screen_sha256,
        "continuum_state_sha256": state.continuum_state_sha256,
        "root": {
            "iterations": state.iterations,
            "primal_residual_eV": state.primal_residual_ev,
            "polarization_energy_eV_diagnostic_only": state.polarization_energy_ev,
            "native_field_l2": float(np.linalg.norm(state.native_field8)),
            "radial_residual_source_l2": float(
                np.linalg.norm(state.radial_residual_source4)
            ),
            "adt_atomic_dipoles_l2_eangstrom": float(
                np.linalg.norm(state.adt_atomic_dipoles_eangstrom)
            ),
        },
        "charge_closure": {
            "permanent_charge_e": permanent_charge,
            "radial_response_charge_e": radial_charge,
        },
        "uniform_chart": {
            "left_inverse_maximum_absolute_error": left_inverse_error,
            "zero_field_uniform_radial_cancellation_maximum_absolute": (
                zero_uniform_radial_cancellation
            ),
            "zero_field_adt_molecular_closure_l2_eangstrom": zero_adt_closure,
            "root_adt_molecular_closure_l2_eangstrom": root_adt_closure,
            "molecular_alpha_eigenvalues": alpha_eigenvalues.tolist(),
        },
        "linearization": {
            "jvp_vjp_dot_left": dot_left,
            "jvp_vjp_dot_right": dot_right,
            "jvp_vjp_absolute_defect": abs(dot_left - dot_right),
            "jvp_vjp_relative_defect": dot_relative,
        },
        "gates": gates,
        "stage_a_passed": all(gates.values()),
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
        "stage_a_passed": False,
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
    _validate_source_and_inputs(
        prereg,
        dataset=dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    if not 0 <= args.start <= args.stop <= creator.EXPECTED_SCREEN_RECORD_COUNT:
        raise ValueError("Record range is outside the frozen twelve-case screen.")
    _configure_torch(args.device)
    _, response = _build_models(
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
        device=args.device,
    )
    runtime, runtime_sha256 = _runtime_identity(args.device)
    prereg_file_sha256 = _sha256_file(prereg_path)
    for screen_index in range(args.start, args.stop):
        selected = prereg["selection"][screen_index]
        try:
            arrays = _load_geometry_arrays(dataset, selected)
            payload = _run_one(
                selected=selected,
                arrays=arrays,
                response=response,
                prereg=prereg,
                prereg_file_sha256=prereg_file_sha256,
                runtime=runtime,
                runtime_sha256=runtime_sha256,
            )
        except Exception as error:  # per-record failure is evidence, not a pass
            payload = _failure_record(
                selected=selected,
                prereg=prereg,
                prereg_file_sha256=prereg_file_sha256,
                runtime=runtime,
                runtime_sha256=runtime_sha256,
                error=error,
            )
        _write_json_exclusive(
            args.output_dir.expanduser().resolve() / f"record-{screen_index:03d}.json",
            payload,
        )
        print(
            f"[{screen_index + 1}/{creator.EXPECTED_SCREEN_RECORD_COUNT}] "
            f"source_index={selected['source_record_index']} "
            f"molecule={selected['selection_identity']['molecule']} "
            f"status={payload['status']}",
            flush=True,
        )


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    prereg = _load_preregistration(prereg_path)
    dataset = args.dataset.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    _validate_source_and_inputs(
        prereg,
        dataset=dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    records_dir = args.output_dir.expanduser().resolve(strict=True)
    expected_names = {
        f"record-{index:03d}.json"
        for index in range(creator.EXPECTED_SCREEN_RECORD_COUNT)
    }
    observed_names = {path.name for path in records_dir.glob("record-*.json")}
    _require(observed_names == expected_names, "Twelve-record screen closure failed.")
    prereg_file_sha256 = _sha256_file(prereg_path)
    records: list[dict[str, Any]] = []
    for index in range(creator.EXPECTED_SCREEN_RECORD_COUNT):
        payload = json.loads((records_dir / f"record-{index:03d}.json").read_text())
        digest = payload.pop("record_sha256", None)
        _require(digest == _canonical_sha256(payload), f"Record {index} hash mismatch.")
        payload["record_sha256"] = digest
        _require(
            payload.get("screen_index") == index, f"Record {index} index mismatch."
        )
        status = payload.get("status")
        _require(
            status in {"pass", "provider-failure", "gate-failure"},
            f"Record {index} has an unknown status.",
        )
        expected_keys = (
            PASS_RECORD_KEYS
            if status in {"pass", "gate-failure"}
            else FAILURE_RECORD_KEYS
        )
        _require(
            set(payload) == expected_keys,
            f"Record {index} contains missing or unregistered fields.",
        )
        if status in {"pass", "gate-failure"}:
            gates = payload.get("gates")
            gate_passed = (
                isinstance(gates, dict)
                and bool(gates)
                and all(value is True for value in gates.values())
            )
            _require(
                payload.get("profile_id") == EXPECTED_PROFILE_ID
                and set(payload["root"]) == ROOT_KEYS
                and set(payload["charge_closure"]) == CHARGE_KEYS
                and set(payload["uniform_chart"]) == UNIFORM_CHART_KEYS
                and set(payload["linearization"]) == LINEARIZATION_KEYS
                and isinstance(gates, dict)
                and set(gates) == GATE_KEYS
                and payload.get("stage_a_passed") is gate_passed
                and (status == "pass") is gate_passed,
                f"Record {index} gate status is internally inconsistent.",
            )
        else:
            _require(
                payload.get("stage_a_passed") is False,
                f"Record {index} failure claims a pass.",
            )
        _require(
            payload.get("selection") == prereg["selection"][index],
            "Selection binding changed.",
        )
        _require(
            payload.get("preregistration_file_sha256") == prereg_file_sha256
            and payload.get("preregistration_artifact_sha256")
            == prereg["preregistration_sha256"],
            "Record preregistration binding changed.",
        )
        _require(
            payload.get("claim_boundary") == creator.CLAIM_BOUNDARY,
            "Record claim changed.",
        )
        _require(
            isinstance(payload.get("runtime_identity"), dict)
            and set(payload["runtime_identity"]) == RUNTIME_IDENTITY_KEYS
            and payload.get("runtime_identity_sha256")
            == _canonical_sha256(payload["runtime_identity"]),
            f"Record {index} runtime identity changed.",
        )
        records.append(payload)

    runtime_digests = {record["runtime_identity_sha256"] for record in records}
    _require(len(runtime_digests) == 1, "Complete screen mixed numerical runtimes.")
    passed_records = [record for record in records if record.get("status") == "pass"]
    all_passed = len(passed_records) == creator.EXPECTED_SCREEN_RECORD_COUNT and all(
        record.get("stage_a_passed") is True for record in passed_records
    )

    def maximum(path: tuple[str, ...]) -> float | None:
        if not passed_records:
            return None
        values = []
        for record in passed_records:
            value: object = record
            for name in path:
                value = value[name]  # type: ignore[index]
            values.append(float(value))
        return max(values)

    def minimum(path: tuple[str, ...]) -> float | None:
        if not passed_records:
            return None
        values = []
        for record in passed_records:
            value: object = record
            for name in path:
                value = value[name]  # type: ignore[index]
            if isinstance(value, list):
                values.extend(float(item) for item in value)
            else:
                values.append(float(value))
        return min(values)

    def maximum_absolute(path: tuple[str, ...]) -> float | None:
        if not passed_records:
            return None
        values = []
        for record in passed_records:
            value: object = record
            for name in path:
                value = value[name]  # type: ignore[index]
            values.append(abs(float(value)))
        return max(values)

    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-aggregate",
        "schema_version": 1,
        "status": ("pass-stage-a-zero-start-coupled-screen" if all_passed else "fail"),
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_artifact_sha256": prereg["preregistration_sha256"],
        "record_count": len(records),
        "pass_count": len(passed_records),
        "provider_or_gate_failure_count": len(records) - len(passed_records),
        "runtime_identity_sha256": next(iter(runtime_digests)),
        "summary": {
            "maximum_iterations": maximum(("root", "iterations")),
            "maximum_primal_residual_eV": maximum(("root", "primal_residual_eV")),
            "maximum_absolute_permanent_charge_e": maximum_absolute(
                ("charge_closure", "permanent_charge_e")
            ),
            "maximum_absolute_radial_response_charge_e": maximum_absolute(
                ("charge_closure", "radial_response_charge_e")
            ),
            "maximum_jvp_vjp_relative_defect": maximum(
                ("linearization", "jvp_vjp_relative_defect")
            ),
            "maximum_uniform_left_inverse_error": maximum(
                ("uniform_chart", "left_inverse_maximum_absolute_error")
            ),
            "maximum_zero_uniform_radial_cancellation": maximum(
                (
                    "uniform_chart",
                    "zero_field_uniform_radial_cancellation_maximum_absolute",
                )
            ),
            "maximum_zero_adt_closure_eangstrom": maximum(
                ("uniform_chart", "zero_field_adt_molecular_closure_l2_eangstrom")
            ),
            "minimum_molecular_alpha_eigenvalue": minimum(
                ("uniform_chart", "molecular_alpha_eigenvalues")
            ),
        },
        "stage_a_passed": all_passed,
        "next_step": (
            prereg["next_if_stage_a_passes"]
            if all_passed
            else prereg["next_if_stage_a_fails"]
        ),
        "accuracy_claim_made": False,
        "capabilities_admitted": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "claim_boundary": creator.CLAIM_BOUNDARY,
        "record_sha256s": [record["record_sha256"] for record in records],
    }
    payload["aggregate_sha256"] = _canonical_sha256(payload)
    _write_json_exclusive(args.aggregate_output.expanduser().resolve(), payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("records", "aggregate"), required=True)
    parser.add_argument(
        "--preregistration", type=Path, default=SOURCE_ROOT / PREREGISTRATION_RELATIVE
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--stop", type=int, default=creator.EXPECTED_SCREEN_RECORD_COUNT
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "records":
        run_records(args)
        return
    if args.aggregate_output is None:
        raise ValueError("--aggregate-output is required in aggregate mode.")
    payload = aggregate(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
