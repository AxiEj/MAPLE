#!/usr/bin/env python3
"""Audit the exact field-energy semantics consumed by separated Route 2.

The official uniform-field branch and MAPLE's native eight-channel injection
branch are replayed on the same checkpoint and geometry.  The artifact freezes
whether the density/source path agrees, whether the energy graphs agree, the
explicit-work identity and sign, the zero-field baseline, raw-graph AD/FD, and
the radial charging identity.  No capability is admitted.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import shlex
import sys
import time

from ase import Atoms
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_MODEL_FEATURE_FIELD_INDICES,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.models import (
    MACEPolarOriginalSourceNativeFieldAdapter,
    NativeSemanticsCanary,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    RepositorySnapshot,
    audit_charging_path,
    audit_directional_derivative,
    audit_field_work_sign,
    audit_uniform_field_replay,
    audit_zero_field_baseline,
    build_unverified_mace_polar_field_semantics_manifest,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    sha256_file,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-field-semantics-water-v1"
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
RANDOM_SEED = 20260815
UNIFORM_GRADIENT_EV_PER_E_ANGSTROM = (0.006, -0.004, 0.003)
UNIFORM_DERIVATIVE_STEP = 2.0e-4
WORK_IDENTITY_ATOL_EV = 2.0e-10
FORWARD_REVERSE_ATOL_EV = 2.0e-9
FORWARD_REVERSE_RTOL = 2.0e-7
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/coupling/exact_gto.py",
    "maple/solvation/coupling/metrics.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/field_semantics.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/calculator/extra_correction/implicit/electrostatic_pairing.py",
    "maple/function/calculator/extra_correction/implicit/gto_field_projection.py",
    "tools/route2_release/run_mace_field_semantics.py",
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


def _configure_determinism(torch: object) -> None:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _tensor_numpy(value: object) -> np.ndarray:
    return np.asarray(value.detach().cpu(), dtype=float)


def _uniform_native_field(radial, atoms: Atoms, gradient: np.ndarray) -> np.ndarray:
    calculator = radial._calculator
    centered = atoms.positions - np.mean(atoms.positions, axis=0, keepdims=True)
    potential = centered @ gradient
    node_values = np.concatenate(
        (potential[:, None], np.broadcast_to(gradient, (len(atoms), 3))), axis=1
    )
    model_input = node_values[:, list(MACE_POLAR_MODEL_FEATURE_FIELD_INDICES)]
    matrix = _tensor_numpy(calculator._reaction_projector.upstream.matrix)
    features = np.einsum("pf,nf->np", matrix, model_input)
    native = np.linalg.solve(radial.field_transform.matrix, features.T).T
    reconstructed = radial.field_transform.to_model_features(native)
    if not np.allclose(reconstructed, features, rtol=2.0e-13, atol=2.0e-13):
        raise RuntimeError("uniform-field native transform inversion failed.")
    return native


def _upstream_output(calculator, atoms: Atoms, gradient: np.ndarray, *, forces: bool):
    import torch

    batch = calculator._batch_dict(atoms)
    batch["external_field"] = torch.tensor(
        gradient.reshape(1, 3),
        dtype=calculator.dtype,
        device=calculator.device,
    )
    return calculator._model_forward(
        batch,
        compute_force=forces,
        compute_stress=False,
        compute_hessian=False,
    )


def _upstream_energy(calculator, atoms: Atoms, gradient: np.ndarray) -> float:
    output = _upstream_output(calculator, atoms, gradient, forces=False)
    return float(output["energy"].sum().detach().cpu())


def _central_derivative(function, point: np.ndarray, direction: np.ndarray) -> float:
    step = UNIFORM_DERIVATIVE_STEP
    return (function(point + step * direction) - function(point - step * direction)) / (
        2.0 * step
    )


def _external_source_record(value: object, *, role: str) -> dict[str, object]:
    source_type = value if isinstance(value, type) else type(value)
    raw_path = inspect.getsourcefile(source_type)
    if raw_path is None:
        raise RuntimeError(f"Unable to bind external source for {role}.")
    path = Path(raw_path).resolve(strict=True)
    _, first_line = inspect.getsourcelines(source_type)
    return {
        "role": role,
        "module": source_type.__module__,
        "qualname": source_type.__qualname__,
        "resolved_path": str(path),
        "first_source_line": int(first_line),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()
    import torch

    _configure_determinism(torch)
    atoms = _water()
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    electronic = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    native_canary = NativeSemanticsCanary.from_adapter(radial)
    manifest = build_unverified_mace_polar_field_semantics_manifest(
        native_canary=native_canary,
        adapter=electronic,
    )
    calculator = radial._calculator

    zero_field = np.zeros((len(atoms), 8), dtype=float)
    vacuum = radial.evaluate_vacuum(atoms, need_forces=True)
    zero_state = radial.evaluate_source(atoms, zero_field, need_fixed_field_forces=True)
    zero_baseline = audit_zero_field_baseline(
        conditioned_zero_energy_eV=electronic.conditioned_raw_energy_ev(
            atoms, zero_field
        ),
        vacuum_energy_eV=vacuum.energy_eV,
        conditioned_zero_forces_eV_per_A=zero_state.fixed_field_forces_eV_per_A,
        vacuum_forces_eV_per_A=vacuum.forces_eV_per_A,
    )

    gradient = np.asarray(UNIFORM_GRADIENT_EV_PER_E_ANGSTROM, dtype=float)
    native_field = _uniform_native_field(radial, atoms, gradient)
    upstream = _upstream_output(calculator, atoms, gradient, forces=True)
    features = radial.field_transform.to_model_features(native_field)
    native = calculator.polar_output_torch(
        atoms,
        model_field_features=torch.tensor(
            features, dtype=calculator.dtype, device=calculator.device
        ),
        compute_forces=True,
    )
    source_error = float(
        np.max(
            np.abs(
                _tensor_numpy(upstream["density_coefficients"])
                - _tensor_numpy(native["density_coefficients"])
            )
        )
    )
    dipole_error = float(
        np.max(
            np.abs(_tensor_numpy(upstream["dipole"]) - _tensor_numpy(native["dipole"]))
        )
    )
    force_error = float(
        np.max(
            np.abs(_tensor_numpy(upstream["forces"]) - _tensor_numpy(native["forces"]))
        )
    )
    upstream_energy = float(upstream["energy"].sum().detach().cpu())
    native_energy = float(native["energy"].sum().detach().cpu())
    energy_error = abs(upstream_energy - native_energy)
    explicit_work = float(
        np.vdot(gradient, _tensor_numpy(upstream["dipole"]).reshape(-1))
    )
    upstream_native_energy_difference = upstream_energy - native_energy
    work_sign_canaries = {
        sign: audit_field_work_sign(
            energy_gradient=np.asarray([upstream_native_energy_difference]),
            reference_source=np.asarray([explicit_work]),
            sign=sign,
        )
        for sign in (-1, 1)
    }
    passing_work_signs = tuple(
        sign for sign, result in work_sign_canaries.items() if result.passed
    )
    work_sign = passing_work_signs[0] if len(passing_work_signs) == 1 else None
    explicit_work_identity_error = (
        abs(upstream_native_energy_difference - work_sign * explicit_work)
        if work_sign is not None
        else min(
            abs(upstream_native_energy_difference - explicit_work),
            abs(upstream_native_energy_difference + explicit_work),
        )
    )

    uniform_direction = gradient / np.linalg.norm(gradient)
    upstream_derivative = _central_derivative(
        lambda value: _upstream_energy(calculator, atoms, value),
        gradient,
        uniform_direction,
    )
    native_direction = _uniform_native_field(radial, atoms, uniform_direction)
    native_derivative = _central_derivative(
        lambda value: electronic.conditioned_raw_energy_ev(atoms, value),
        native_field,
        native_direction,
    )
    replay = audit_uniform_field_replay(
        source_max_abs_error=source_error,
        energy_absolute_error_eV=energy_error,
        force_max_abs_error_eV_per_A=force_error,
        dipole_max_abs_error_e_angstrom=dipole_error,
        energy_derivative_absolute_error_e_angstrom=abs(
            upstream_derivative - native_derivative
        ),
    )

    rng = np.random.default_rng(RANDOM_SEED)
    direction = rng.normal(size=native_field.shape)
    direction /= np.linalg.norm(direction)
    directional = audit_directional_derivative(
        energy=lambda value: electronic.conditioned_raw_energy_ev(atoms, value),
        gradient=lambda value: electronic.conditioned_raw_energy_field_gradient(
            atoms, value
        ),
        field=native_field,
        direction=direction,
    )
    forward_directional = electronic.conditioned_raw_energy_directional_derivative(
        atoms,
        native_field,
        direction,
    )
    forward_reverse_error = abs(
        forward_directional - directional.analytic_directional_derivative_eV
    )
    forward_reverse_scale = max(
        abs(forward_directional),
        abs(directional.analytic_directional_derivative_eV),
        np.finfo(float).tiny,
    )
    forward_reverse_passed = bool(
        forward_reverse_error
        <= FORWARD_REVERSE_ATOL_EV + FORWARD_REVERSE_RTOL * forward_reverse_scale
    )
    charging = audit_charging_path(
        energy=lambda value: electronic.conditioned_raw_energy_ev(atoms, value),
        gradient=lambda value: electronic.conditioned_raw_energy_field_gradient(
            atoms, value
        ),
        endpoint_field=native_field,
    )

    work_identity_passed = bool(
        work_sign is not None and explicit_work_identity_error <= WORK_IDENTITY_ATOL_EV
    )
    external_sources = {
        record["role"]: record
        for record in (
            _external_source_record(
                calculator.model,
                role="mace_polar_model_external_field_and_energy",
            ),
            _external_source_record(
                calculator._reaction_projector.upstream,
                role="graph_longrange_external_field_projector",
            ),
        )
    }
    semantics_measurements = {
        "zero_field_baseline": zero_baseline.as_dict(),
        "uniform_field_replay": replay.as_dict(),
        "upstream_native_energy_difference_eV": upstream_native_energy_difference,
        "explicit_uniform_work_eV": explicit_work,
        "work_sign_canaries": {
            str(sign): result.as_dict() for sign, result in work_sign_canaries.items()
        },
        "selected_work_sign": work_sign,
        "explicit_work_identity_absolute_error_eV": (explicit_work_identity_error),
        "native_raw_reverse_directional_derivative": directional.as_dict(),
        "native_raw_forward_directional_derivative_eV": forward_directional,
        "native_raw_forward_reverse_absolute_error_eV": forward_reverse_error,
        "native_raw_forward_reverse_passed": forward_reverse_passed,
        "native_raw_charging_path": charging.as_dict(),
        "external_python_sources": external_sources,
    }
    semantics_measurement_sha256 = canonical_json_sha256(semantics_measurements)
    finalized_manifest = replace(
        manifest,
        external_potential_sign=(work_sign if work_identity_passed else None),
        uniform_field_sign=(work_sign if work_identity_passed else None),
        spin_channel_factor=0.5,
        native_injection_explicit_work_included=(
            False if work_identity_passed else None
        ),
        upstream_uniform_explicit_work_included=(
            True if work_identity_passed else None
        ),
        origin_convention=(
            "arithmetic atom-position barycenter removed before uniform potential"
        ),
        evidence_measurement_sha256s=(semantics_measurement_sha256,),
    )

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-checkpoint-field-semantics-terminal-audit",
        "status": (
            "native-injection-omits-upstream-explicit-work"
            if work_identity_passed
            and replay.branch_semantics == "hybrid-branch-dependent"
            else "field-semantics-unresolved"
        ),
        "claim_boundary": (
            "This artifact interprets the exact checkpoint/native-injection energy "
            "branches. It does not validate the original source as a quantitative "
            "PCM source, select Phi0, admit Phi1Delta, or admit E/F/H/V/M."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": source_hashes,
        "external_python_sources": external_sources,
        "checkpoint": checkpoint_record(checkpoint),
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": str(radial.dtype).replace("torch.", ""),
        "native_semantics": native_canary.as_dict(),
        "field_semantics_manifest": finalized_manifest.as_dict(),
        "field_semantics_measurement_sha256": semantics_measurement_sha256,
        "protocol": {
            "random_seed": RANDOM_SEED,
            "uniform_gradient_eV_per_e_angstrom": list(gradient),
            "uniform_derivative_step": UNIFORM_DERIVATIVE_STEP,
            "work_identity_atol_eV": WORK_IDENTITY_ATOL_EV,
            "forward_reverse_atol_eV": FORWARD_REVERSE_ATOL_EV,
            "forward_reverse_rtol": FORWARD_REVERSE_RTOL,
            "long_range_evaluator": (
                MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
            ),
        },
        "zero_field_baseline": zero_baseline.as_dict(),
        "uniform_field_replay": {
            **replay.as_dict(),
            "upstream_energy_eV": upstream_energy,
            "native_injection_raw_energy_eV": native_energy,
            "upstream_native_energy_difference_eV": (upstream_native_energy_difference),
            "explicit_uniform_work_eV": explicit_work,
            "work_sign_canaries": {
                str(sign): result.as_dict()
                for sign, result in work_sign_canaries.items()
            },
            "selected_work_sign": work_sign,
            "explicit_work_identity_absolute_error_eV": (explicit_work_identity_error),
            "explicit_work_identity_passed": work_identity_passed,
            "upstream_directional_derivative_e_angstrom": upstream_derivative,
            "native_directional_derivative_e_angstrom": native_derivative,
        },
        "native_raw_directional_derivative": {
            **directional.as_dict(),
            "forward_mode_directional_derivative_eV": forward_directional,
            "forward_reverse_absolute_error_eV": forward_reverse_error,
            "forward_reverse_passed": forward_reverse_passed,
        },
        "native_raw_charging_path": charging.as_dict(),
        "decision": {
            "zero_field_baseline_passed": zero_baseline.passed,
            "native_raw_graph_directional_derivative_passed": (
                directional.passed and forward_reverse_passed
            ),
            "native_raw_graph_charging_identity_passed": charging.passed,
            "upstream_and_native_energy_branches_identical": replay.passed,
            "native_injection_complete_external_enthalpy": False,
            "phi1_delta_semantics_complete": (
                finalized_manifest.phi1_semantics_complete
            ),
            "phi1_delta_physical_ledger_admitted": False,
            "phi0_physical_ledger_admitted": False,
            "source_pcm_physics_admitted": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {
            "field_semantics_manifest": payload["field_semantics_manifest"],
            "protocol": payload["protocol"],
            "zero_field_baseline": payload["zero_field_baseline"],
            "uniform_field_replay": payload["uniform_field_replay"],
            "native_raw_directional_derivative": payload[
                "native_raw_directional_derivative"
            ],
            "native_raw_charging_path": payload["native_raw_charging_path"],
            "decision": payload["decision"],
        }
    )
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_MACE_FIELD_SEMANTICS="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "status": payload["status"],
                "decision": payload["decision"],
                "capabilities": payload["capabilities"],
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
