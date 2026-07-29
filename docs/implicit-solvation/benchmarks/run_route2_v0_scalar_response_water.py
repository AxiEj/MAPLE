#!/usr/bin/env python3
"""Run the preregistered frozen-checkpoint Route-2 V0 water falsifier.

The archived local-jet/ddPCM water field is reused only as a fixed external
field.  This runner does not solve a V0 continuum fixed point and does not
change the public Route-2 energy or response.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import ase
from ase.build import molecule
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.gto_density import (  # noqa: E402
    density_to_external_field_order,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (  # noqa: E402
    NeutralDensityCoordinates,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_scalar_response import (  # noqa: E402
    Route2V0ScalarResponseState,
    evaluate_anchored_scalar_response_v0,
)
from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.read.command_control import CommandControl  # noqa: E402
from maple.function.route2_smd_profiles import DDPCM_SMD_PROFILE  # noqa: E402


ARTIFACT_ID = "route2-v0-scalar-response-water-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-scalar-response-water-prereg-v1.json"
)
INPUT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-mace-local-field-thermodynamic-canary-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_scalar_response_water.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    PREREG_RELATIVE_PATH,
    "maple/function/calculator/mace/_macepol_calculator.py",
    (
        "maple/function/calculator/extra_correction/implicit/"
        "electrostatic_pairing.py"
    ),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_response.py"
    ),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_v0_scalar_response.py"
    ),
)
PUBLIC_SETTINGS = (
    "#model=macepol-m",
    "#sp(verbose=1)",
    (
        "#solv(implicit=water,method=smd,provider=pyddx,"
        f"profile={DDPCM_SMD_PROFILE},response=scf,"
        "standard_state=1m,experimental=true)"
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_sha256(atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
    digest.update(np.asarray(atoms.get_positions(), dtype=np.float64).tobytes())
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The Route-2 V0 canary requires a clean tracked source checkout; "
            f"git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _source_hashes() -> dict[str, str]:
    return {
        relative: _sha256(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _load_protocol_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg_path = REPO_ROOT / PREREG_RELATIVE_PATH
    input_path = REPO_ROOT / INPUT_RELATIVE_PATH
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    source = json.loads(input_path.read_text(encoding="utf-8"))
    frozen = prereg["frozen_input"]
    if prereg["status"] != "preregistered-before-execution":
        raise RuntimeError("The Route-2 V0 protocol is not preregistered.")
    if frozen["field_source_artifact"] != INPUT_RELATIVE_PATH:
        raise RuntimeError("The V0 field-source path changed after preregistration.")
    if frozen["field_source_artifact_sha256"] != _sha256(input_path):
        raise RuntimeError("The frozen V0 field-source artifact hash changed.")
    if source["artifact"] != "route2-mace-local-field-thermodynamic-canary-v1":
        raise RuntimeError("The frozen V0 input artifact identity is invalid.")
    if source["checkpoint"]["sha256"] != frozen["checkpoint_sha256"]:
        raise RuntimeError("The V0 input checkpoint does not match the preregistration.")
    if (
        source["scientific_identity"]["field_interface"]
        != prereg["construction"]["field_interface"]
    ):
        raise RuntimeError("The V0 input field interface changed.")
    return prereg, source


def _neutral_external_basis(atom_count: int) -> np.ndarray:
    coordinates = NeutralDensityCoordinates(atom_count)
    basis = np.empty((4 * atom_count, coordinates.dimension), dtype=float)
    for column in range(coordinates.dimension):
        reduced = np.zeros(coordinates.dimension, dtype=float)
        reduced[column] = 1.0
        raw_density_order = coordinates.expand(reduced)
        basis[:, column] = density_to_external_field_order(
            raw_density_order
        ).reshape(-1)
    return basis


def _state_payload(state: Route2V0ScalarResponseState) -> dict[str, Any]:
    payload = asdict(state)
    for name, value in tuple(payload.items()):
        if isinstance(value, np.ndarray):
            payload[name] = value.tolist()
    return payload


def main() -> int:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    git_head = _require_clean_source()
    prereg, source = _load_protocol_inputs()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    total_started = time.perf_counter()
    atoms = molecule("H2O")
    atoms.info.update(
        charge=0,
        mult=1,
        mol2={"atom_types": ["o", "h", "h"]},
    )
    if _geometry_sha256(atoms) != source["molecule"]["geometry_sha256"]:
        raise RuntimeError("The frozen V0 water geometry changed.")
    field = np.asarray(
        source["state_arrays"]["reaction_field_values_ev"],
        dtype=float,
    )
    if field.shape != (len(atoms), 4) or not np.all(np.isfinite(field)):
        raise RuntimeError("The frozen V0 field is invalid.")

    parameters = CommandControl.from_settings(list(PUBLIC_SETTINGS)).as_dict()
    _sync(args.device)
    started = time.perf_counter()
    calculator = SetCalculator(
        args.device,
        parameters["model"],
        str(work_dir / "maple.out"),
        atoms=atoms,
        d4=bool(parameters.get("d4", False)),
        implicit=parameters["solv"]["method"],
        solvent=parameters["solv"]["implicit"],
        model_options=parameters.get("model_options"),
        solvation_options=parameters["solv"],
        charge_options=parameters.get("charge") or {},
    ).set_calculator()
    _sync(args.device)
    model_load_seconds = time.perf_counter() - started

    checkpoint = dict(
        getattr(calculator, "mace_polar_checkpoint_provenance", {}) or {}
    )
    if checkpoint.get("sha256") != prereg["frozen_input"]["checkpoint_sha256"]:
        raise RuntimeError("The live MACE-POLAR checkpoint changed.")

    zero_field = np.zeros_like(field)
    _sync(args.device)
    started = time.perf_counter()
    zero_state, _ = calculator.polar_state(
        atoms,
        node_potential_ev=zero_field[:, 0],
        node_gradient_ev_per_angstrom=zero_field[:, 1:],
    )
    zero_gradient = calculator.intrinsic_energy_field_gradient(
        atoms,
        node_potential_ev=zero_field[:, 0],
        node_gradient_ev_per_angstrom=zero_field[:, 1:],
    )
    _sync(args.device)
    zero_reference_seconds = time.perf_counter() - started

    evaluation_count = 0

    def evaluate(current_field: np.ndarray) -> Route2V0ScalarResponseState:
        nonlocal evaluation_count
        values = np.asarray(current_field, dtype=float)
        state, _ = calculator.polar_state(
            atoms,
            node_potential_ev=values[:, 0],
            node_gradient_ev_per_angstrom=values[:, 1:],
        )
        gradient = calculator.intrinsic_energy_field_gradient(
            atoms,
            node_potential_ev=values[:, 0],
            node_gradient_ev_per_angstrom=values[:, 1:],
        )
        response = calculator.linearize_density_response(
            atoms,
            node_potential_ev=values[:, 0],
            node_gradient_ev_per_angstrom=values[:, 1:],
        )
        evaluation_count += 1
        return evaluate_anchored_scalar_response_v0(
            field_conditioned_model_energy_ev=float(state.energy_ev),
            zero_field_model_energy_ev=float(zero_state.energy_ev),
            reaction_field_values_ev=values,
            model_density_coefficients=np.asarray(
                state.density_coefficients,
                dtype=float,
            ),
            field_conditioned_energy_field_gradient=gradient,
            zero_field_energy_field_gradient=zero_gradient,
            density_response=response,
        )

    _sync(args.device)
    started = time.perf_counter()
    base = evaluate(field)
    zero = evaluate(zero_field)
    _sync(args.device)
    base_zero_seconds = time.perf_counter() - started

    step = float(prereg["gates"]["finite_difference_step"])
    dimension = field.size
    response_jacobian = np.empty((dimension, dimension), dtype=float)
    scalar_gradient_fd = np.empty(dimension, dtype=float)
    _sync(args.device)
    started = time.perf_counter()
    for column in range(dimension):
        displacement = np.zeros(dimension, dtype=float)
        displacement[column] = step
        plus = evaluate(field + displacement.reshape(field.shape))
        minus = evaluate(field - displacement.reshape(field.shape))
        plus_density = density_to_external_field_order(
            plus.response_density_coefficients
        ).reshape(-1)
        minus_density = density_to_external_field_order(
            minus.response_density_coefficients
        ).reshape(-1)
        response_jacobian[:, column] = (
            plus_density - minus_density
        ) / (2.0 * step)
        scalar_gradient_fd[column] = (
            plus.anchored_field_scalar_change_ev
            - minus.anchored_field_scalar_change_ev
        ) / (2.0 * step)
    _sync(args.device)
    finite_difference_seconds = time.perf_counter() - started

    analytic_scalar_gradient = density_to_external_field_order(
        base.response_density_coefficients
    ).reshape(-1)
    gradient_error = (
        analytic_scalar_gradient - scalar_gradient_fd
    ).reshape(field.shape)
    maximum_monopole_gradient_error = float(
        np.max(np.abs(gradient_error[:, 0]))
    )
    maximum_dipole_gradient_error = float(
        np.max(np.abs(gradient_error[:, 1:]))
    )
    zero_anchor_error = float(
        np.max(
            np.abs(
                zero.response_density_coefficients
                - zero.model_density_coefficients
            )
        )
    )

    neutral_basis = _neutral_external_basis(len(atoms))
    neutral_response = (
        neutral_basis.T @ response_jacobian @ neutral_basis
    )
    symmetric_response = 0.5 * (neutral_response + neutral_response.T)
    antisymmetric_response = 0.5 * (neutral_response - neutral_response.T)
    symmetric_norm = float(np.linalg.norm(symmetric_response))
    antisymmetric_norm = float(np.linalg.norm(antisymmetric_response))
    antisymmetric_ratio = antisymmetric_norm / max(symmetric_norm, 1.0e-30)
    eigenvalues = np.linalg.eigvalsh(symmetric_response)
    maximum_eigenvalue = float(np.max(eigenvalues))

    gates = prereg["gates"]
    gate_results = {
        "zero_field_anchor": (
            zero_anchor_error
            <= float(gates["zero_field_anchor_maximum_absolute_error_e"])
        ),
        "zero_field_charge": (
            abs(zero.response_total_charge_e)
            <= float(
                gates["zero_field_response_total_charge_absolute_max_e"]
            )
        ),
        "base_field_charge": (
            abs(base.response_total_charge_e)
            <= float(gates["base_response_total_charge_absolute_max_e"])
        ),
        "monopole_scalar_gradient": (
            maximum_monopole_gradient_error
            <= float(gates["maximum_monopole_scalar_gradient_error_e"])
        ),
        "dipole_scalar_gradient": (
            maximum_dipole_gradient_error
            <= float(
                gates["maximum_dipole_scalar_gradient_error_e_angstrom"]
            )
        ),
        "neutral_response_reciprocity": (
            antisymmetric_ratio
            <= float(
                gates[
                    "neutral_response_antisymmetric_frobenius_ratio"
                ]
            )
        ),
        "neutral_response_passivity": (
            maximum_eigenvalue
            <= float(gates["neutral_response_maximum_eigenvalue_ev"])
        ),
    }
    failed_gates = [
        name for name, passed in gate_results.items() if not passed
    ]
    status = "pass" if not failed_gates else "fail"
    total_seconds = time.perf_counter() - total_started

    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": status,
        "failed_gates": failed_gates,
        "generated_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "execution_git_head": git_head,
        "source_files_sha256": _source_hashes(),
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": _sha256(REPO_ROOT / PREREG_RELATIVE_PATH),
            "protocol_id": prereg["protocol_id"],
        },
        "frozen_input": {
            "path": INPUT_RELATIVE_PATH,
            "sha256": _sha256(REPO_ROOT / INPUT_RELATIVE_PATH),
            "artifact": source["artifact"],
        },
        "scientific_identity": {
            "solute_response_model": "official MACE-POLAR-1-M",
            "construction": base.construction,
            "field_interface": base.field_interface,
            "continuum_field_source": "archived pyddx ddPCM root",
            "reaction_field_projector": "local-jet",
            "geometry_policy": "fixed archived ASE G2 water",
            "continuum_fixed_point_solved": False,
            "public_route_changed": False,
            "training_or_fine_tuning": False,
        },
        "claim_boundary": prereg["claim_boundary"],
        "stop_condition": prereg["stop_condition"],
        "gates": {
            "thresholds": gates,
            "passed": gate_results,
        },
        "molecule": source["molecule"],
        "checkpoint": checkpoint,
        "field_summary": source["field_summary"],
        "scalar_response": {
            "base": _state_payload(base),
            "zero": _state_payload(zero),
            "zero_field_anchor_maximum_absolute_error_e": zero_anchor_error,
            "maximum_monopole_scalar_gradient_error_e": (
                maximum_monopole_gradient_error
            ),
            "maximum_dipole_scalar_gradient_error_e_angstrom": (
                maximum_dipole_gradient_error
            ),
        },
        "neutral_response": {
            "dimension": int(neutral_response.shape[0]),
            "finite_difference_step": step,
            "symmetric_frobenius_norm_ev": symmetric_norm,
            "antisymmetric_frobenius_norm_ev": antisymmetric_norm,
            "antisymmetric_to_symmetric_frobenius_ratio": (
                antisymmetric_ratio
            ),
            "minimum_eigenvalue_ev": float(np.min(eigenvalues)),
            "maximum_eigenvalue_ev": maximum_eigenvalue,
            "eigenvalues_ev": eigenvalues.tolist(),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "mace_torch": importlib.metadata.version("mace-torch"),
            "graph_longrange": importlib.metadata.version(
                "graph-longrange"
            ),
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name()
                if args.device == "cuda"
                else platform.processor()
            ),
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "v0_evaluation_count": evaluation_count,
            "timing_seconds": {
                "model_load": model_load_seconds,
                "zero_reference": zero_reference_seconds,
                "base_and_zero_v0": base_zero_seconds,
                "finite_difference_dense_response": (
                    finite_difference_seconds
                ),
                "total": total_seconds,
            },
        },
    }
    _write_exclusive_json(output, artifact)
    print(output)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
