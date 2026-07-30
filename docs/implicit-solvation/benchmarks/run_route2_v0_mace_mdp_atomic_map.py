#!/usr/bin/env python3
"""Audit whether frozen MACE-MDP exposes a closed atomic moment decomposition.

The public MACE-MDP calculator reports a molecular dipole and polarizability.
The checkpoint implementation also constructs those totals from atom-resolved
readout contributions.  This audit observes that construction without changing
weights, then rejects it unless the atom-level reconstruction exactly closes
back to the public properties.  It has no continuum, energy, force, or
experimental-solvation calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_ID = "route2-v0-mace-mdp-atomic-map-acetone-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-atomic-map-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_v0_mace_mdp_atomic_map.py"
)
RESPONSE_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-acetone-response-v1.json"
)
QM_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-qeq-acetone-qm-field-v1.json"
)
DEFAULT_MODEL = Path("/home/axie/.cache/mace/MACE-MDP.model")
MACE_MODELS_PATH = Path(
    "/home/axie/miniconda3/envs/maple/lib/python3.11/site-packages/"
    "mace/modules/models.py"
)
MACE_CALCULATOR_PATH = Path(
    "/home/axie/miniconda3/envs/maple/lib/python3.11/site-packages/"
    "mace/calculators/mace.py"
)
MACE_TORCH_TOOLS_PATH = Path(
    "/home/axie/miniconda3/envs/maple/lib/python3.11/site-packages/"
    "mace/tools/torch_tools.py"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite numeric value.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The MACE-MDP atomic-map audit requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    _git("ls-files", "--error-unmatch", RUNNER_RELATIVE_PATH)
    _git("ls-files", "--error-unmatch", PREREG_RELATIVE_PATH)
    return _git("rev-parse", "HEAD")


def _relative_norm(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _runtime_versions() -> dict[str, str]:
    return {
        "ase": metadata.version("ase"),
        "mace_torch": metadata.version("mace-torch"),
        "numpy": metadata.version("numpy"),
        "torch": metadata.version("torch"),
    }


def _validate_preregistration() -> tuple[dict[str, Any], dict[str, Any], str]:
    preregistration_path = REPO_ROOT / PREREG_RELATIVE_PATH
    response_path = REPO_ROOT / RESPONSE_ARTIFACT_RELATIVE_PATH
    qm_path = REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH
    preregistration = _load_json(
        preregistration_path, label="MACE-MDP atomic-map preregistration"
    )
    response_artifact = _load_json(response_path, label="MACE-MDP response artifact")

    if (
        preregistration.get("protocol_id") != "route2-v0-mace-mdp-atomic-map-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The MACE-MDP atomic-map protocol is not frozen.")
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise TypeError("The atomic-map preregistration omits its contract.")
    if contract.get("source_sha256") != {
        RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
    }:
        raise RuntimeError("The MACE-MDP atomic-map runner changed after freeze.")
    expected_inputs = {
        RESPONSE_ARTIFACT_RELATIVE_PATH: _sha256(response_path),
        QM_ARTIFACT_RELATIVE_PATH: _sha256(qm_path),
        str(MACE_MODELS_PATH): _sha256(MACE_MODELS_PATH),
        str(MACE_CALCULATOR_PATH): _sha256(MACE_CALCULATOR_PATH),
        str(MACE_TORCH_TOOLS_PATH): _sha256(MACE_TORCH_TOOLS_PATH),
    }
    if contract.get("input_sha256") != expected_inputs:
        raise RuntimeError("A MACE-MDP atomic-map input changed after freeze.")
    if response_artifact.get("artifact") != "route2-v0-mace-mdp-acetone-response-v1":
        raise RuntimeError("The prerequisite response artifact identity is invalid.")
    if (
        response_artifact.get("status") != "pass"
        or response_artifact.get("scientific_falsification", {}).get("verdict")
        != "admit-frozen-mace-mdp-response-coefficients-only"
    ):
        raise RuntimeError("The prerequisite MACE-MDP coefficient screen did not pass.")

    model = contract.get("model")
    if not isinstance(model, dict):
        raise TypeError("The atomic-map preregistration omits model identity.")
    if (
        not DEFAULT_MODEL.is_file()
        or model.get("path") != str(DEFAULT_MODEL)
        or model.get("sha256") != _sha256(DEFAULT_MODEL)
        or model.get("size_bytes") != DEFAULT_MODEL.stat().st_size
    ):
        raise RuntimeError("The MACE-MDP checkpoint does not match the freeze.")
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError("The atomic-map preregistration omits runtime identity.")
    if (
        runtime.get("versions") != _runtime_versions()
        or runtime.get("device") != "cpu"
        or runtime.get("default_dtype") != "float64"
        or runtime.get("torch_force_no_weights_only_load") != "1"
        or os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1"
    ):
        raise RuntimeError("The MACE-MDP atomic-map runtime does not match freeze.")
    return preregistration, response_artifact, _sha256(preregistration_path)


def _frozen_geometry() -> tuple[tuple[str, ...], np.ndarray]:
    artifact = _load_json(
        REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH, label="frozen acetone QM artifact"
    )
    system = artifact.get("system")
    if not isinstance(system, dict):
        raise TypeError("The frozen QM artifact has no system identity.")
    symbols = tuple(system.get("atom_symbols", ()))
    positions = np.asarray(system.get("positions_angstrom"), dtype=float)
    if symbols != ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H"):
        raise RuntimeError("The frozen system is not the declared acetone geometry.")
    if positions.shape != (10, 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("The frozen acetone geometry is invalid.")
    return symbols, positions


def _as_numpy(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if not hasattr(value, "detach"):
        raise TypeError(f"{name} must be a torch tensor.")
    tensor = cast(Any, value)
    array = np.asarray(tensor.detach().cpu().numpy(), dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} must be finite with shape {shape}.")
    return array


def _atomic_decomposition(
    symbols: tuple[str, ...], positions: np.ndarray
) -> dict[str, np.ndarray]:
    from ase import Atoms

    mace_calculator_module = import_module("mace.calculators.mace")
    torch = import_module("torch")
    calculator_type = getattr(mace_calculator_module, "MACECalculator", None)
    if calculator_type is None:
        raise RuntimeError("The pinned MACE runtime has no MACECalculator.")
    atoms = Atoms(symbols=symbols, positions=positions)
    calculator = calculator_type(
        model_paths=str(DEFAULT_MODEL),
        model_type="DipolePolarizabilityMACE",
        default_dtype="float64",
        device="cpu",
    )
    public_dipole = np.asarray(calculator.get_property("dipole", atoms), dtype=float)
    public_polarizability = np.asarray(
        calculator.get_property("polarizability", atoms), dtype=float
    )
    if (
        public_dipole.shape != (3,)
        or public_polarizability.shape != (3, 3)
        or not np.all(np.isfinite(public_dipole))
        or not np.all(np.isfinite(public_polarizability))
    ):
        raise RuntimeError("The public MACE-MDP properties are invalid.")

    model = calculator.models[0]
    readouts = tuple(model.readouts)
    if not readouts:
        raise RuntimeError("The frozen MACE-MDP model has no readout blocks.")
    captured: list[Any] = []

    def capture_readout(_module: object, _inputs: object, output: object) -> None:
        if not hasattr(output, "detach"):
            raise RuntimeError("A MACE-MDP readout did not return a tensor.")
        captured.append(cast(Any, output).detach().clone())

    handles = [readout.register_forward_hook(capture_readout) for readout in readouts]
    try:
        batch = calculator._atoms_to_batch(atoms)
        output = model(
            calculator._clone_batch(batch).to_dict(),
            compute_dielectric_derivatives=False,
            training=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(output, dict) or len(captured) != len(readouts):
        raise RuntimeError("The frozen MACE-MDP readout sequence did not close.")

    atom_count = len(symbols)
    direct_dipole = _as_numpy(
        output.get("dipole"), name="Direct MACE-MDP dipole", shape=(1, 3)
    )[0]
    direct_polarizability = _as_numpy(
        output.get("polarizability"),
        name="Direct MACE-MDP polarizability",
        shape=(1, 3, 3),
    )[0]
    charges = _as_numpy(
        output.get("charges"), name="MACE-MDP charges", shape=(atom_count,)
    )
    atomic_dipoles = _as_numpy(
        output.get("atomic_dipoles"),
        name="MACE-MDP atomic dipoles",
        shape=(atom_count, 3),
    )
    readout_vectors: list[Any] = []
    for index, raw in enumerate(captured):
        if not hasattr(raw, "squeeze"):
            raise RuntimeError("Captured MACE-MDP readout is not tensor-like.")
        node_output = cast(Any, raw).squeeze(-1)
        if tuple(node_output.shape) != (atom_count, 10):
            raise RuntimeError(
                "The frozen MACE-MDP readout layout changed; expected "
                f"({atom_count}, 10), got {tuple(node_output.shape)}."
            )
        readout_vectors.append(
            torch.cat((node_output[:, 1:2], node_output[:, 5:]), dim=-1)
        )
        if tuple(readout_vectors[-1].shape) != (atom_count, 6):
            raise RuntimeError(f"MACE-MDP polar readout {index} has invalid shape.")
    atomic_polarizability_sh = torch.stack(readout_vectors, dim=-1).sum(dim=-1)
    atomic_polarizability = torch.einsum(
        "ijk,ai->ajk", model.change_of_basis, atomic_polarizability_sh
    )
    atomic_polarizability_array = _as_numpy(
        atomic_polarizability,
        name="MACE-MDP atomic polarizabilities",
        shape=(atom_count, 3, 3),
    )

    try:
        inverse_total = np.linalg.inv(direct_polarizability)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("MACE-MDP total polarizability is singular.") from error
    if not np.all(np.isfinite(inverse_total)):
        raise RuntimeError("MACE-MDP total polarizability inverse is non-finite.")
    atomic_dipole_weights = atomic_polarizability_array @ inverse_total
    return {
        "public_dipole_eangstrom": public_dipole,
        "public_polarizability_eangstrom2_per_volt": public_polarizability,
        "direct_dipole_eangstrom": direct_dipole,
        "direct_polarizability_eangstrom2_per_volt": direct_polarizability,
        "charges_e": charges,
        "atomic_dipoles_eangstrom": atomic_dipoles,
        "atomic_polarizabilities_eangstrom2_per_volt": atomic_polarizability_array,
        "atomic_dipole_weights": atomic_dipole_weights,
    }


def _upper_check(value: float, maximum: float) -> dict[str, float | bool]:
    return {"value": value, "maximum": maximum, "passes": value <= maximum}


def main() -> int:
    args = _parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    git_head = _require_clean_source()
    preregistration, response_artifact, preregistration_sha = (
        _validate_preregistration()
    )
    symbols, positions = _frozen_geometry()
    decomposition = _atomic_decomposition(symbols, positions)
    public_dipole = decomposition["public_dipole_eangstrom"]
    public_polarizability = decomposition["public_polarizability_eangstrom2_per_volt"]
    direct_dipole = decomposition["direct_dipole_eangstrom"]
    direct_polarizability = decomposition["direct_polarizability_eangstrom2_per_volt"]
    charges = decomposition["charges_e"]
    atomic_dipoles = decomposition["atomic_dipoles_eangstrom"]
    atomic_polarizabilities = decomposition[
        "atomic_polarizabilities_eangstrom2_per_volt"
    ]
    atomic_dipole_weights = decomposition["atomic_dipole_weights"]

    reconstructed_dipole = np.sum(charges[:, None] * positions + atomic_dipoles, axis=0)
    reconstructed_polarizability = np.sum(atomic_polarizabilities, axis=0)
    moment_identity = np.sum(atomic_dipole_weights, axis=0)
    gates = preregistration.get("numerical_gates")
    if not isinstance(gates, dict):
        raise TypeError("The atomic-map preregistration omits numerical gates.")
    checks = {
        "public_direct_dipole": _upper_check(
            _relative_norm(direct_dipole, public_dipole),
            _number(
                gates.get("public_direct_relative_max"), label="public/direct gate"
            ),
        ),
        "public_direct_polarizability": _upper_check(
            _relative_norm(direct_polarizability, public_polarizability),
            _number(
                gates.get("public_direct_relative_max"), label="public/direct gate"
            ),
        ),
        "atomic_dipole_reconstruction": _upper_check(
            _relative_norm(reconstructed_dipole, direct_dipole),
            _number(
                gates.get("atomic_reconstruction_relative_max"), label="atomic gate"
            ),
        ),
        "atomic_polarizability_reconstruction": _upper_check(
            _relative_norm(reconstructed_polarizability, direct_polarizability),
            _number(
                gates.get("atomic_reconstruction_relative_max"), label="atomic gate"
            ),
        ),
        "net_charge": {
            "value": float(np.sum(charges)),
            "absolute_maximum": _number(
                gates.get("net_charge_absolute_max_e"), label="charge gate"
            ),
        },
        "atomic_dipole_weight_moment_identity": _upper_check(
            float(np.linalg.norm(moment_identity - np.eye(3), ord="fro")),
            _number(gates.get("moment_identity_frobenius_max"), label="moment gate"),
        ),
    }
    checks["net_charge"]["passes"] = bool(
        abs(float(checks["net_charge"]["value"]))
        <= float(checks["net_charge"]["absolute_maximum"])
    )
    passes_all = all(bool(check["passes"]) for check in checks.values())
    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass" if passes_all else "reject",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": git_head,
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": preregistration_sha,
            "protocol_id": preregistration["protocol_id"],
        },
        "source_files_sha256": {
            RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
        },
        "input_files_sha256": {
            RESPONSE_ARTIFACT_RELATIVE_PATH: _sha256(
                REPO_ROOT / RESPONSE_ARTIFACT_RELATIVE_PATH
            ),
            QM_ARTIFACT_RELATIVE_PATH: _sha256(REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH),
            str(MACE_MODELS_PATH): _sha256(MACE_MODELS_PATH),
            str(MACE_CALCULATOR_PATH): _sha256(MACE_CALCULATOR_PATH),
            str(MACE_TORCH_TOOLS_PATH): _sha256(MACE_TORCH_TOOLS_PATH),
        },
        "claim_boundary": preregistration["claim_boundary"],
        "hard_constraints": preregistration["hard_constraints"],
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
        },
        "mace_mdp_atomic_decomposition": {
            "public_dipole_eangstrom": public_dipole.tolist(),
            "direct_dipole_eangstrom": direct_dipole.tolist(),
            "charges_e": charges.tolist(),
            "atomic_dipoles_eangstrom": atomic_dipoles.tolist(),
            "reconstructed_dipole_eangstrom": reconstructed_dipole.tolist(),
            "public_polarizability_eangstrom2_per_volt": public_polarizability.tolist(),
            "direct_polarizability_eangstrom2_per_volt": direct_polarizability.tolist(),
            "atomic_polarizabilities_eangstrom2_per_volt": (
                atomic_polarizabilities.tolist()
            ),
            "reconstructed_polarizability_eangstrom2_per_volt": (
                reconstructed_polarizability.tolist()
            ),
            "atomic_dipole_weights": atomic_dipole_weights.tolist(),
            "atomic_dipole_weight_moment_identity": moment_identity.tolist(),
            "weight_definition": (
                "For a Cartesian total induced dipole p in e angstrom, atom a "
                "receives p_a=alpha_a @ inv(alpha_total) @ p. This is an "
                "identity-bound moment partition only; no GTO width, PCM "
                "source, continuum energy, or force is chosen here."
            ),
        },
        "numerical_checks": checks,
        "decision": {
            "passes_all_registered_checks": passes_all,
            "verdict": (
                "admit-atomic-moment-partition-for-source-map-gates-only"
                if passes_all
                else "reject-mace-mdp-atomic-moment-partition"
            ),
            "admission_boundary": (
                "A pass proves only that the frozen model's observed atomic "
                "readouts close exactly to its public molecular moments. It "
                "does not create a density/GTO representer, a continuum source, "
                "an energy, a force, or a solvation prediction."
            ),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "versions": _runtime_versions(),
            "device": "cpu",
            "default_dtype": "float64",
            "model_path": str(DEFAULT_MODEL),
            "model_sha256": _sha256(DEFAULT_MODEL),
            "model_size_bytes": DEFAULT_MODEL.stat().st_size,
            "torch_force_no_weights_only_load": os.environ.get(
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
            ),
            "mace_models_source_path": str(MACE_MODELS_PATH),
            "mace_models_source_sha256": _sha256(MACE_MODELS_PATH),
            "mace_calculator_source_path": str(MACE_CALCULATOR_PATH),
            "mace_calculator_source_sha256": _sha256(MACE_CALCULATOR_PATH),
            "mace_torch_tools_source_path": str(MACE_TORCH_TOOLS_PATH),
            "mace_torch_tools_source_sha256": _sha256(MACE_TORCH_TOOLS_PATH),
        },
        "prerequisite_response_artifact": {
            "path": RESPONSE_ARTIFACT_RELATIVE_PATH,
            "sha256": _sha256(REPO_ROOT / RESPONSE_ARTIFACT_RELATIVE_PATH),
            "verdict": response_artifact["scientific_falsification"]["verdict"],
        },
    }
    _write_exclusive_json(output_path, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
