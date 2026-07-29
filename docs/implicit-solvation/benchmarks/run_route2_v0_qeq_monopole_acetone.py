#!/usr/bin/env python3
"""Run the source-bound acetone structural canary for the no-training V0-Q.

The frozen input is the zero-reaction-field ``l<=1`` MACE-POLAR density from
the archived exact-GTO acetone state.  This runner never invokes MACE's
field-dependent update.  It instead reopens the same PCMSolver IEFPCM cavity,
uses one identical 1.5-A Gaussian source/receiver basis, and solves the
published-Rappe--Goddard-hardness, monopole-only V0-Q KKT state.

It is a fixed-geometry structural falsifier, not a hydration calculation.  It
does not use experimental solvation values, train or fine-tune a model,
produce a public profile, or certify forces, a PES, optimization, scans, MD,
or NVE.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import ase
from ase.data import atomic_numbers
from ase.units import Bohr
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.continuum_response import (  # noqa: E402
    PCMSolverExternalMEPCavityResponse,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (  # noqa: E402
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (  # noqa: E402
    AtomCenteredL1GTOBasis,
    FixedCavityGTOGalerkinOperator,
)
from maple.function.calculator.extra_correction.implicit.pcmsolver import (  # noqa: E402
    PCMSolverSession,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_qeq_monopole import (  # noqa: E402
    RAPPE_GODDARD_QEQ_PARAMETER_SHA256,
    RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION,
    embed_mace_polar_l1_density_in_single_radial_gto,
    evaluate_route2_v0_rappe_goddard_monopole,
)


ARTIFACT_ID = "route2-v0-qeq-monopole-acetone-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-qeq-monopole-acetone-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_qeq_monopole_acetone.py"
)
ARCHIVE_RELATIVE_DIRECTORY = (
    ".omx/benchmarks/"
    "route2-exact-gto-fixed-geometry-canary-v1-408ca3f-20260727/"
    "maple.out.implicit"
)
ARCHIVE_STATE_RELATIVE_PATH = f"{ARCHIVE_RELATIVE_DIRECTORY}/route2-state.npz"
ARCHIVE_RESULT_RELATIVE_PATH = f"{ARCHIVE_RELATIVE_DIRECTORY}/route2-result.json"
ARCHIVE_MANIFEST_RELATIVE_PATH = f"{ARCHIVE_RELATIVE_DIRECTORY}/manifest.json"
ARCHIVE_PCM_INPUT_RELATIVE_PATH = (
    f"{ARCHIVE_RELATIVE_DIRECTORY}/@route2-smd-stability-fallback.pcm"
)
PCMSOLVER_LIBRARY = Path(
    "/home/axie/.local/opt/pcmsolver-bbd992d54ebeace528cf236dede1f0c56641defb/"
    "lib/libpcm.so"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    "maple/function/calculator/extra_correction/implicit/continuum_response.py",
    "maple/function/calculator/extra_correction/implicit/electrostatic_pairing.py",
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/gto_galerkin.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/function/calculator/extra_correction/implicit/route2_v0_qeq_monopole.py",
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_v0_variational_quadratic.py"
    ),
    "maple/function/calculator/extra_correction/charge/data/qeq.dat",
)
INPUT_RELATIVE_PATHS = (
    ARCHIVE_STATE_RELATIVE_PATH,
    ARCHIVE_RESULT_RELATIVE_PATH,
    ARCHIVE_MANIFEST_RELATIVE_PATH,
    ARCHIVE_PCM_INPUT_RELATIVE_PATH,
)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH
DEFAULT_ARCHIVE_STATE = REPO_ROOT / ARCHIVE_STATE_RELATIVE_PATH
DEFAULT_ARCHIVE_RESULT = REPO_ROOT / ARCHIVE_RESULT_RELATIVE_PATH
DEFAULT_ARCHIVE_MANIFEST = REPO_ROOT / ARCHIVE_MANIFEST_RELATIVE_PATH
DEFAULT_PCM_INPUT = REPO_ROOT / ARCHIVE_PCM_INPUT_RELATIVE_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--archive-state", type=Path, default=DEFAULT_ARCHIVE_STATE)
    parser.add_argument("--archive-result", type=Path, default=DEFAULT_ARCHIVE_RESULT)
    parser.add_argument("--archive-manifest", type=Path, default=DEFAULT_ARCHIVE_MANIFEST)
    parser.add_argument("--parsed-pcm-input", type=Path, default=DEFAULT_PCM_INPUT)
    parser.add_argument("--pcmsolver-library", type=Path, default=PCMSOLVER_LIBRARY)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object.")
    return payload


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The V0-Q acetone canary requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS}


def _parsed_pcm_radii_angstrom(path: Path, atom_count: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 4 or fields[:2] != ["DBL_ARRAY", "RADII"]:
            continue
        count = int(fields[2])
        if count != atom_count:
            raise RuntimeError(
                "The parsed PCMSolver input radius count does not match the "
                "acetone atom count."
            )
        values = np.asarray(
            [float(value) for value in lines[index + 1 : index + 1 + count]],
            dtype=float,
        )
        if values.shape != (atom_count,) or np.any(values <= 0.0):
            raise RuntimeError("The parsed PCMSolver input contains invalid radii.")
        return values * Bohr
    raise RuntimeError("The parsed PCMSolver input does not expose atom radii.")


def _require_leq(value: float, ceiling: float, *, name: str) -> None:
    if not np.isfinite(value) or value > ceiling:
        raise RuntimeError(f"{name} failed: {value:.16e} > {ceiling:.16e}.")


def _require_gt(value: float, floor: float, *, name: str) -> None:
    if not np.isfinite(value) or value <= floor:
        raise RuntimeError(f"{name} failed: {value:.16e} <= {floor:.16e}.")


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _validate_preregistration(
    path: Path,
    *,
    supplied_inputs: dict[str, Path],
    pcmsolver_library: Path,
) -> tuple[dict[str, object], dict[str, float], str]:
    if path.resolve() != DEFAULT_PREREGISTRATION.resolve():
        raise RuntimeError("Only the tracked V0-Q acetone preregistration is allowed.")
    preregistration = _load_object(path, label="preregistration")
    if (
        preregistration.get("protocol_id")
        != "route2-v0-qeq-monopole-acetone-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The V0-Q acetone preregistration identity is invalid.")
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("The preregistration omits its execution contract.")
    expected_sources = contract.get("source_sha256")
    if not isinstance(expected_sources, dict) or set(expected_sources) != set(
        SOURCE_RELATIVE_PATHS
    ):
        raise RuntimeError("The preregistration source binding set is incomplete.")
    for relative, expected in expected_sources.items():
        if _sha256(REPO_ROOT / relative) != expected:
            raise RuntimeError(
                f"Tracked source {relative} no longer matches the preregistration."
            )

    expected_inputs = contract.get("input_sha256")
    if not isinstance(expected_inputs, dict) or set(expected_inputs) != set(
        INPUT_RELATIVE_PATHS
    ):
        raise RuntimeError("The preregistration input binding set is incomplete.")
    for relative, path_value in supplied_inputs.items():
        expected_path = (REPO_ROOT / relative).resolve()
        if path_value.resolve() != expected_path:
            raise RuntimeError(f"The V0-Q acetone canary forbids substituting {relative}.")
        if _sha256(path_value) != expected_inputs[relative]:
            raise RuntimeError(f"Frozen input {relative} no longer matches the freeze.")

    if _sha256(pcmsolver_library) != contract.get("pcmsolver_library_sha256"):
        raise RuntimeError("The PCMSolver library does not match the preregistration.")
    gates = preregistration.get("structural_gates")
    if not isinstance(gates, dict):
        raise RuntimeError("The preregistration omits structural gates.")
    required_gate_names = {
        "frozen_total_charge_abs_e_max",
        "operator_maximum_eigenvalue_hartree_max",
        "electronic_minimum_curvature_hartree_min",
        "joint_minimum_curvature_hartree_min",
        "stationarity_residual_inf_max",
        "charge_constraint_residual_abs_e_max",
        "additional_constraint_residual_inf_max",
        "response_antisymmetry_norm_hartree_max",
        "response_maximum_eigenvalue_hartree_max",
        "induced_l1_max_abs_e_max",
        "half_coupling_identity_error_hartree_max",
    }
    if set(gates) != required_gate_names or not all(
        isinstance(value, (float, int)) for value in gates.values()
    ):
        raise RuntimeError("The preregistration structural-gate set is invalid.")
    return preregistration, {key: float(value) for key, value in gates.items()}, _sha256(path)


def _load_frozen_acetone(
    *,
    archive_state: Path,
    archive_result: Path,
    archive_manifest: Path,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, object]]:
    manifest = _load_object(archive_manifest, label="archived acetone manifest")
    symbols = manifest.get("elements")
    positions = manifest.get("positions_angstrom")
    if not isinstance(symbols, list) or not all(isinstance(symbol, str) for symbol in symbols):
        raise RuntimeError("The archived acetone manifest has invalid element symbols.")
    if tuple(symbols) != ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H"):
        raise RuntimeError("The archived geometry is not the preregistered acetone record.")
    positions_array = np.asarray(positions, dtype=float)
    if positions_array.shape != (len(symbols), 3) or not np.all(np.isfinite(positions_array)):
        raise RuntimeError("The archived acetone positions are invalid.")
    result = _load_object(archive_result, label="archived MACE result")
    checkpoint = result.get("mace_polar_checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != (
        "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
    ):
        raise RuntimeError("The frozen source is not the declared MACE-POLAR-1-M state.")
    if result.get("reaction_field_projector") != "exact-gto-v1":
        raise RuntimeError("The archived source does not have the declared exact-GTO provenance.")
    with np.load(archive_state) as archive:
        if "gas_density_coefficients" not in archive.files:
            raise RuntimeError("The archived state omits the gas density coefficients.")
        density = np.asarray(archive["gas_density_coefficients"], dtype=float)
    if density.shape != (len(symbols), 4) or not np.all(np.isfinite(density)):
        raise RuntimeError("The archived MACE gas density has an invalid shape.")
    return tuple(symbols), positions_array, density, checkpoint


def main() -> int:
    args = _parse_args()
    git_head = _require_clean_source()
    for name in (
        "preregistration",
        "archive_state",
        "archive_result",
        "archive_manifest",
        "parsed_pcm_input",
        "pcmsolver_library",
    ):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        raise FileExistsError(work_dir)

    supplied_inputs = {
        ARCHIVE_STATE_RELATIVE_PATH: args.archive_state,
        ARCHIVE_RESULT_RELATIVE_PATH: args.archive_result,
        ARCHIVE_MANIFEST_RELATIVE_PATH: args.archive_manifest,
        ARCHIVE_PCM_INPUT_RELATIVE_PATH: args.parsed_pcm_input,
    }
    preregistration, gates, preregistration_sha = _validate_preregistration(
        args.preregistration,
        supplied_inputs=supplied_inputs,
        pcmsolver_library=args.pcmsolver_library,
    )
    symbols, positions, native_density, checkpoint = _load_frozen_acetone(
        archive_state=args.archive_state,
        archive_result=args.archive_result,
        archive_manifest=args.archive_manifest,
    )
    atomic_number_array = np.asarray([atomic_numbers[symbol] for symbol in symbols])
    radii_angstrom = _parsed_pcm_radii_angstrom(args.parsed_pcm_input, len(symbols))

    work_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    os.chdir(work_dir)
    with PCMSolverSession(
        atomic_number_array,
        positions / Bohr,
        args.parsed_pcm_input,
        library_path=args.pcmsolver_library,
    ) as session:
        response = PCMSolverExternalMEPCavityResponse(
            session,
            cavity_radii_angstrom=radii_angstrom,
        )
        operator = FixedCavityGTOGalerkinOperator(
            response,
            positions,
            AtomCenteredL1GTOBasis((MACE_POLAR_DENSITY_SIGMA_ANGSTROM,)),
        )
        frozen_density = embed_mace_polar_l1_density_in_single_radial_gto(
            native_density,
            operator.basis,
        )
        state = evaluate_route2_v0_rappe_goddard_monopole(
            symbols=symbols,
            frozen_density_coefficients=frozen_density,
            operator=operator,
            target_total_charge_e=0.0,
        )
        cavity_size = session.cavity_size
        pcm_runtime_source = session.library_source
    elapsed_seconds = time.perf_counter() - started

    response_matrix = state.response_state.joint_external_dual_response
    response_antisymmetry_norm = float(
        np.linalg.norm(0.5 * (response_matrix - response_matrix.T), ord=2)
    )
    response_maximum_eigenvalue = float(np.max(np.linalg.eigvalsh(response_matrix)))
    induced_l1_max_abs = float(
        np.max(np.abs(state.response_state.induced_density_coefficients[:, :, 1:]))
    )
    operator_provenance = operator.provenance
    metrics = {
        "frozen_total_charge_abs_e": abs(state.response_state.frozen_total_charge_e),
        "operator_maximum_eigenvalue_hartree": float(
            operator_provenance["maximum_eigenvalue_hartree"]
        ),
        "electronic_minimum_curvature_hartree": (
            state.response_state.electronic_minimum_neutral_curvature
        ),
        "joint_minimum_curvature_hartree": (
            state.response_state.joint_minimum_neutral_curvature
        ),
        "stationarity_residual_inf": state.response_state.stationarity_residual_inf,
        "charge_constraint_residual_abs_e": abs(
            state.response_state.charge_constraint_residual_e
        ),
        "additional_constraint_residual_inf": (
            state.response_state.additional_constraint_residual_inf
        ),
        "response_antisymmetry_norm_hartree": response_antisymmetry_norm,
        "response_maximum_eigenvalue_hartree": response_maximum_eigenvalue,
        "induced_l1_max_abs_e": induced_l1_max_abs,
        "half_coupling_identity_error_hartree": (
            state.response_state.continuum_snapshot.half_coupling_identity_error_hartree
        ),
    }
    _require_leq(
        metrics["frozen_total_charge_abs_e"],
        gates["frozen_total_charge_abs_e_max"],
        name="frozen total charge",
    )
    _require_leq(
        metrics["operator_maximum_eigenvalue_hartree"],
        gates["operator_maximum_eigenvalue_hartree_max"],
        name="continuum maximum eigenvalue",
    )
    _require_gt(
        metrics["electronic_minimum_curvature_hartree"],
        gates["electronic_minimum_curvature_hartree_min"],
        name="electronic minimum curvature",
    )
    _require_gt(
        metrics["joint_minimum_curvature_hartree"],
        gates["joint_minimum_curvature_hartree_min"],
        name="joint minimum curvature",
    )
    for metric_name, gate_name in (
        ("stationarity_residual_inf", "stationarity_residual_inf_max"),
        ("charge_constraint_residual_abs_e", "charge_constraint_residual_abs_e_max"),
        (
            "additional_constraint_residual_inf",
            "additional_constraint_residual_inf_max",
        ),
        (
            "response_antisymmetry_norm_hartree",
            "response_antisymmetry_norm_hartree_max",
        ),
        (
            "response_maximum_eigenvalue_hartree",
            "response_maximum_eigenvalue_hartree_max",
        ),
        ("induced_l1_max_abs_e", "induced_l1_max_abs_e_max"),
        (
            "half_coupling_identity_error_hartree",
            "half_coupling_identity_error_hartree_max",
        ),
    ):
        _require_leq(metrics[metric_name], gates[gate_name], name=metric_name)

    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": git_head,
        "source_files_sha256": _source_hashes(),
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": preregistration_sha,
            "protocol_id": preregistration["protocol_id"],
        },
        "claim_boundary": (
            "This source-bound acetone result validates only a fixed-geometry, "
            "no-training structural V0-Q control: frozen MACE-POLAR gas density, "
            "published QEq hardness curvature, a same-basis reciprocal PCMSolver "
            "operator, and the resulting KKT state. It is not full QEq and it does "
            "not report or validate experimental solvation accuracy, a nonpolar "
            "term, a total solvation free energy, force, PES, optimization, scan, "
            "MD, NVE, smooth-cavity behavior, or a public Route-2 profile."
        ),
        "hard_constraints": preregistration["hard_constraints"],
        "frozen_inputs": {
            "archived_exact_gto_state": {
                "relative_path": ARCHIVE_STATE_RELATIVE_PATH,
                "sha256": _sha256(args.archive_state),
                "gas_density_coefficients_sha256": _sha256_array(native_density),
            },
            "archived_exact_gto_result": {
                "relative_path": ARCHIVE_RESULT_RELATIVE_PATH,
                "sha256": _sha256(args.archive_result),
                "checkpoint": checkpoint,
            },
            "archived_geometry_manifest": {
                "relative_path": ARCHIVE_MANIFEST_RELATIVE_PATH,
                "sha256": _sha256(args.archive_manifest),
            },
            "parsed_pcmsolver_input": {
                "relative_path": ARCHIVE_PCM_INPUT_RELATIVE_PATH,
                "sha256": _sha256(args.parsed_pcm_input),
                "cavity_radii_angstrom": radii_angstrom.tolist(),
            },
            "pcmsolver_library": {
                "resolved_path": str(args.pcmsolver_library),
                "loaded_source": pcm_runtime_source,
                "sha256": _sha256(args.pcmsolver_library),
            },
        },
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "atom_count": len(symbols),
            "positions_angstrom": positions.tolist(),
            "target_total_charge_e": 0.0,
            "frozen_density_shape": list(frozen_density.shape),
            "cavity_size": cavity_size,
        },
        "construction": {
            "name": RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION,
            "parameter_table_sha256": RAPPE_GODDARD_QEQ_PARAMETER_SHA256,
            "frozen_solute": (
                "unmodified archived MACE-POLAR-1-M zero-reaction-field "
                "l<=1 coefficients"
            ),
            "electronic_response": (
                "reference-shifted published-hardness monopole tangent; QEq "
                "electronegativity linear terms are intentionally absent, so "
                "this is not full QEq"
            ),
            "response_subspace": "monopoles only; every induced l=1 coefficient is an exact KKT constraint",
            "continuum": operator_provenance,
        },
        "structural_gates": {
            key: {"value": metrics[key], "threshold": gates[gate_name]}
            for key, gate_name in (
                ("frozen_total_charge_abs_e", "frozen_total_charge_abs_e_max"),
                (
                    "operator_maximum_eigenvalue_hartree",
                    "operator_maximum_eigenvalue_hartree_max",
                ),
                (
                    "electronic_minimum_curvature_hartree",
                    "electronic_minimum_curvature_hartree_min",
                ),
                (
                    "joint_minimum_curvature_hartree",
                    "joint_minimum_curvature_hartree_min",
                ),
                ("stationarity_residual_inf", "stationarity_residual_inf_max"),
                (
                    "charge_constraint_residual_abs_e",
                    "charge_constraint_residual_abs_e_max",
                ),
                (
                    "additional_constraint_residual_inf",
                    "additional_constraint_residual_inf_max",
                ),
                (
                    "response_antisymmetry_norm_hartree",
                    "response_antisymmetry_norm_hartree_max",
                ),
                (
                    "response_maximum_eigenvalue_hartree",
                    "response_maximum_eigenvalue_hartree_max",
                ),
                ("induced_l1_max_abs_e", "induced_l1_max_abs_e_max"),
                (
                    "half_coupling_identity_error_hartree",
                    "half_coupling_identity_error_hartree_max",
                ),
            )
        },
        "energy_ledger_hartree": {
            "electronic_induction": state.response_state.electronic_induction_energy_hartree,
            "continuum_polarization": state.response_state.continuum_polarization_energy_hartree,
            "solute_continuum": state.response_state.solute_continuum_energy_hartree,
            "stationary_total": state.response_state.stationary_total_energy_hartree,
        },
        "diagnostics": {
            "frozen_total_charge_e": state.response_state.frozen_total_charge_e,
            "stationary_total_charge_e": state.response_state.stationary_total_charge_e,
            "curvature_stability_threshold": state.response_state.curvature_stability_threshold,
            "electronic_curvature_antisymmetry_norm": (
                state.response_state.electronic_curvature_antisymmetry_norm
            ),
            "operator_minimum_eigenvalue_hartree": float(
                operator_provenance["minimum_eigenvalue_hartree"]
            ),
            "operator_antisymmetric_norm_hartree": float(
                operator_provenance["antisymmetric_norm_hartree"]
            ),
            "qeq_monopole_hardness_sha256": _sha256_array(
                state.monopole_hardness_matrix_hartree_per_e2
            ),
            "qeq_monopole_hardness_eigenvalues_hartree_per_e2": np.linalg.eigvalsh(
                state.monopole_hardness_matrix_hartree_per_e2
            ).tolist(),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "elapsed_seconds": elapsed_seconds,
        },
    }
    _write_exclusive_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
