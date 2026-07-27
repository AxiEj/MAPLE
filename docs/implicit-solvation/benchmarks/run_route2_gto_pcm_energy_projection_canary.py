#!/usr/bin/env python3
"""Run one fixed-geometry GTO/PCM energy-norm projection canary.

The canary compares one- and two-radial ``l<=1`` atom-centred Gaussian bases
against the same frozen QM gas-density MEP on the same intrinsic PCMSolver
IEFPCM cavity.  It validates only the source/receiver Galerkin closure and the
PCM-energy-norm projection.  It does not train MACE, define a variational
MACE-POLAR energy, certify hydration accuracy, or enable forces/PES/OPT/MD.
"""

from __future__ import annotations

import argparse
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

import ase
from ase.units import Bohr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.continuum_response import (  # noqa: E402
    PCMSolverExternalMEPCavityResponse,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (  # noqa: E402
    AtomCenteredL1GTOBasis,
    FixedCavityGTOGalerkinOperator,
)
from maple.function.calculator.extra_correction.implicit.pcm_energy_projection import (  # noqa: E402
    project_surface_potential_in_pcm_energy_norm,
)
from maple.function.calculator.extra_correction.implicit.pcmsolver import (  # noqa: E402
    PCMSolverSession,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

KCAL_PER_HARTREE = 627.5094740631
ONE_RADIAL_WIDTHS = (1.5,)
TWO_RADIAL_WIDTHS = (1.5, 3.0)
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
HELPER_RELATIVE_PATH = "docs/implicit-solvation/benchmarks/route2_qm_surface_mep.py"
QM_RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_qm_gas_checkpoint.py"
)
PCM_INPUT_RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_intrinsic_pcm_input.py"
)
PCM_INPUT_GENERATOR_SOURCE_RELATIVE_PATHS = (
    PCM_INPUT_RUNNER_RELATIVE_PATH,
    "maple/function/calculator/extra_correction/implicit/route2_pcmsolver_cavity.py",
    "maple/function/calculator/extra_correction/implicit/smd.py",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/function/read/filereader/mol2_reader.py",
    "maple/function/route2_smd_profiles.py",
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_gto_pcm_energy_projection_canary.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    HELPER_RELATIVE_PATH,
    QM_RUNNER_RELATIVE_PATH,
    PCM_INPUT_RUNNER_RELATIVE_PATH,
    ("maple/function/calculator/extra_correction/implicit/" "continuum_response.py"),
    ("maple/function/calculator/extra_correction/implicit/" "electrostatic_pairing.py"),
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/gto_galerkin.py",
    "maple/function/calculator/extra_correction/implicit/pcm_energy_projection.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/function/calculator/extra_correction/implicit/route2_pcmsolver_cavity.py",
    "maple/function/calculator/extra_correction/implicit/smd.py",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/function/read/filereader/mol2_reader.py",
    "maple/function/route2_smd_profiles.py",
)
DEFAULT_MOL2 = (
    REPO_ROOT / ".omx/benchmarks/route2-macepolar-smd-smoke/dataset/"
    "mol2files_gaff/mobley_3867265.mol2"
)
DEFAULT_PARSED_PCM_INPUT = (
    REPO_ROOT / ".omx/benchmarks/"
    "route2-intrinsic-exact-gto-ten-panel-20260727-234625/"
    "mobley_3867265/exact_gto/maple.out.implicit/"
    "@route2-smd-intrinsic.pcm"
)
DEFAULT_QM_CHECKPOINT = (
    REPO_ROOT / ".omx/benchmarks/route2-qm-ddx-acetone-center-20260725/"
    "pyscf_smd_acetone_wb97mv_def2tzvpd_g3_n50x194sg1.gas.chk"
)
DEFAULT_QM_LEDGER = (
    REPO_ROOT / ".omx/benchmarks/route2-qm-ddx-acetone-center-20260725/"
    "pyscf_smd_acetone_wb97mv_def2tzvpd_g3_n50x194sg1.gas.json"
)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyscf-python", type=Path, required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
    parser.add_argument("--compound-id", default="mobley_3867265")
    parser.add_argument("--molecule-name", default="acetone")
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument("--mol2", type=Path, default=DEFAULT_MOL2)
    parser.add_argument(
        "--parsed-pcm-input",
        type=Path,
        default=DEFAULT_PARSED_PCM_INPUT,
    )
    parser.add_argument(
        "--qm-checkpoint",
        type=Path,
        default=DEFAULT_QM_CHECKPOINT,
    )
    parser.add_argument(
        "--qm-ledger",
        type=Path,
        default=DEFAULT_QM_LEDGER,
    )
    parser.add_argument("--pcm-input-ledger", type=Path)
    parser.add_argument(
        "--relative-spectral-cutoff",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument(
        "--require-two-radial-improvement",
        action="store_true",
    )
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


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The GTO/PCM projection canary requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} JSON must contain one object.")
    return payload


def _cutoff_token(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _validate_preregistration(
    path: Path,
    *,
    compound_id: str,
    molecule_name: str,
    relative_spectral_cutoff: float,
    pcmsolver_library: Path,
) -> tuple[dict[str, object], dict[str, object], str]:
    if path.resolve() != DEFAULT_PREREGISTRATION.resolve():
        raise RuntimeError(
            "This transfer canary accepts only the tracked four-record "
            "preregistration."
        )
    preregistration = _load_json_object(path, label="preregistration")
    if (
        preregistration.get("artifact_id")
        != ("route2-gto-pcm-energy-projection-four-prereg-v1")
        or preregistration.get("status") != "frozen-before-transfer-execution"
    ):
        raise RuntimeError("The transfer preregistration identity is invalid.")

    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("The preregistration omits its execution contract.")
    source_bindings = contract.get("source_sha256")
    if not isinstance(source_bindings, dict):
        raise RuntimeError("The preregistration omits source SHA bindings.")
    expected_sources = set(SOURCE_RELATIVE_PATHS)
    if set(source_bindings) != expected_sources:
        raise RuntimeError("The preregistered source binding set is incomplete.")
    for relative, expected_sha in source_bindings.items():
        if _sha256(REPO_ROOT / relative) != expected_sha:
            raise RuntimeError(
                f"Tracked source {relative} no longer matches the preregistration."
            )
    if _sha256(pcmsolver_library) != contract.get("pcmsolver_library_sha256"):
        raise RuntimeError("The PCMSolver library does not match the preregistration.")

    method = preregistration.get("method")
    if not isinstance(method, dict):
        raise RuntimeError("The preregistration omits its method definition.")
    if method.get("basis_arms_angstrom") != {
        "one_radial": list(ONE_RADIAL_WIDTHS),
        "two_radial": list(TWO_RADIAL_WIDTHS),
    }:
        raise RuntimeError("The canary GTO basis arms do not match the freeze.")
    cutoffs = method.get("relative_spectral_cutoffs")
    if not isinstance(cutoffs, list) or relative_spectral_cutoff not in cutoffs:
        raise RuntimeError(
            "The requested relative spectral cutoff was not preregistered."
        )

    records = preregistration.get("records")
    if not isinstance(records, list):
        raise RuntimeError("The preregistration omits its record panel.")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("compound_id") == compound_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one preregistered record for {compound_id}."
        )
    record = matches[0]
    if record.get("name") != molecule_name:
        raise RuntimeError(
            "The molecule name does not match the preregistered compound identity."
        )
    return preregistration, record, _sha256(path)


def _validate_pyscf_interpreter(
    invoked_path: Path,
    execution_contract: dict[str, object],
) -> dict[str, str]:
    invoked = invoked_path.absolute()
    if not invoked.is_file():
        raise FileNotFoundError(invoked)
    resolved = invoked.resolve()
    resolved_sha = _sha256(resolved)
    if resolved_sha != execution_contract.get("pyscf_python_resolved_sha256"):
        raise RuntimeError(
            "The PySCF interpreter used for the surface-MEP helper does not "
            "match the preregistration."
        )
    return {
        "invoked_path": str(invoked),
        "resolved_target": str(resolved),
        "resolved_sha256": resolved_sha,
    }


def _validate_qm_ledger(
    ledger_path: Path,
    checkpoint_path: Path,
    mol2_path: Path,
    *,
    record: dict[str, object],
    source_bindings: dict[str, str],
    execution_contract: dict[str, object],
) -> dict[str, object]:
    specification = record.get("qm_reference")
    if not isinstance(specification, dict):
        raise RuntimeError("The preregistered record omits its QM reference.")
    ledger = _load_json_object(ledger_path, label="QM ledger")
    mode = specification.get("provenance_mode")
    if _sha256(mol2_path) != record.get("mol2_sha256"):
        raise RuntimeError("The MOL2 input does not match its preregistered hash.")
    if _sha256(checkpoint_path) != specification.get("checkpoint_sha256"):
        if mode == "legacy-frozen-checkpoint-v1":
            raise RuntimeError(
                "The legacy QM checkpoint does not match its preregistered hash."
            )
    system = ledger.get("system")
    if not isinstance(system, dict):
        raise RuntimeError("The QM ledger omits its system block.")
    expected_system = {
        "compound_id": record["compound_id"],
        "molecule": record["name"],
        "charge": 0,
        "multiplicity": 1,
        "method": "omegaB97M-V",
        "basis": "def2-tzvpd",
        "reference": "RKS",
        "density_fitting": True,
    }
    for key, expected in expected_system.items():
        if system.get(key) != expected:
            raise RuntimeError(f"The QM ledger has an invalid system field: {key}.")

    if mode == "preregistered-gas-runner-v1":
        if ledger.get("status") != "pass" or ledger.get("artifact_id") != (
            f"route2-qm-gas-checkpoint-{record['compound_id']}-v1"
        ):
            raise RuntimeError("The generated QM ledger identity is invalid.")
        source = ledger.get("source")
        inputs = ledger.get("input")
        result = ledger.get("result")
        numerics = ledger.get("numerics")
        runtime = ledger.get("runtime")
        if not all(
            isinstance(value, dict)
            for value in (source, inputs, result, numerics, runtime)
        ):
            raise RuntimeError("The generated QM ledger is incomplete.")
        if source.get("runner_path") != QM_RUNNER_RELATIVE_PATH:
            raise RuntimeError("The generated QM ledger names the wrong runner.")
        source_hashes = source.get("source_sha256")
        if (
            not isinstance(source_hashes, dict)
            or source_hashes.get(QM_RUNNER_RELATIVE_PATH)
            != source_bindings[QM_RUNNER_RELATIVE_PATH]
        ):
            raise RuntimeError("The generated QM runner source is not bound.")
        if inputs.get("mol2_sha256") != record["mol2_sha256"]:
            raise RuntimeError("The generated QM ledger used the wrong MOL2.")
        checkpoint = result.get("checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != _sha256(
            checkpoint_path
        ):
            raise RuntimeError("The generated QM ledger/checkpoint binding failed.")
        expected_numerics = {
            "semilocal_grid_level": 3,
            "nonlocal_grid_profile": "pyscf-official-50x194-sg1",
            "nonlocal_grid_level": None,
            "nonlocal_atom_grid": [50, 194],
            "nonlocal_prune": "sg1_prune",
            "scf_energy_tolerance_hartree": 1.0e-10,
            "scf_gradient_tolerance": 1.0e-7,
            "maximum_scf_cycles": 100,
            "maximum_memory_mb": 8000,
        }
        for key, expected in expected_numerics.items():
            if numerics.get(key) != expected:
                raise RuntimeError(
                    f"The generated QM ledger has invalid numerics: {key}."
                )
        python_executable = runtime.get("python_executable")
        if (
            runtime.get("pyscf") != "2.13.1"
            or runtime.get("threads") != 8
            or not isinstance(python_executable, dict)
            or python_executable.get("resolved_sha256")
            != execution_contract.get("pyscf_python_resolved_sha256")
        ):
            raise RuntimeError("The generated QM runtime does not match the freeze.")
    elif mode == "legacy-frozen-checkpoint-v1":
        if _sha256(ledger_path) != specification.get("ledger_sha256"):
            raise RuntimeError("The legacy QM ledger hash is not preregistered.")
        environment = ledger.get("environment")
        numerics = ledger.get("numerics")
        provenance = ledger.get("provenance")
        if not all(
            isinstance(value, dict) for value in (environment, numerics, provenance)
        ):
            raise RuntimeError("The legacy QM ledger is incomplete.")
        if environment.get("pyscf") != "2.13.1" or environment.get("threads") != 8:
            raise RuntimeError("The legacy QM runtime does not match the freeze.")
        if provenance.get("mol2_sha256") != record["mol2_sha256"]:
            raise RuntimeError("The legacy QM ledger used the wrong MOL2.")
        if provenance.get("runner_sha256") != specification.get(
            "declared_runner_sha256"
        ):
            raise RuntimeError("The legacy QM runner declaration changed.")
        expected_numerics = {
            "grid_profile": "level",
            "grid_level": 3,
            "nlc_grid_profile": "pyscf-official-50x194-sg1",
            "nlc_atom_grid": [50, 194],
            "nlc_prune": "sg1_prune",
            "scf_energy_tolerance_hartree": 1.0e-10,
            "scf_gradient_tolerance": 1.0e-7,
            "maximum_scf_cycles": 100,
        }
        for key, expected in expected_numerics.items():
            if numerics.get(key) != expected:
                raise RuntimeError(f"The legacy QM ledger has invalid numerics: {key}.")
    else:
        raise RuntimeError(f"Unsupported QM provenance mode: {mode}.")

    return {
        "provenance_mode": mode,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }


def _validate_pcm_input_provenance(
    ledger_path: Path | None,
    parsed_input_path: Path,
    mol2_path: Path,
    *,
    record: dict[str, object],
    pcmsolver_library: Path,
    source_bindings: dict[str, str],
    execution_contract: dict[str, object],
) -> dict[str, object]:
    specification = record.get("pcm_input")
    if not isinstance(specification, dict):
        raise RuntimeError("The preregistered record omits its PCM input.")
    mode = specification.get("provenance_mode")
    parsed_sha = _sha256(parsed_input_path)
    if mode == "legacy-frozen-machine-input-v1":
        if ledger_path is not None:
            raise RuntimeError(
                "Legacy PCM inputs must not claim a new generator ledger."
            )
        if parsed_sha != specification.get("effective_machine_input_sha256"):
            raise RuntimeError("The legacy PCM input hash is not preregistered.")
        return {
            "provenance_mode": mode,
            "effective_machine_input_path": str(parsed_input_path),
            "effective_machine_input_sha256": parsed_sha,
        }
    if mode != "preregistered-intrinsic-input-runner-v1":
        raise RuntimeError(f"Unsupported PCM-input provenance mode: {mode}.")
    if ledger_path is None:
        raise RuntimeError("The generated PCM input requires its generator ledger.")
    ledger = _load_json_object(ledger_path, label="PCM-input ledger")
    if ledger.get("status") != "pass" or ledger.get("artifact_id") != (
        f"route2-intrinsic-pcm-input-{record['compound_id']}-v1"
    ):
        raise RuntimeError("The generated PCM-input ledger identity is invalid.")
    source = ledger.get("source")
    system = ledger.get("system")
    profile = ledger.get("profile")
    inputs = ledger.get("input")
    outputs = ledger.get("outputs")
    runtime = ledger.get("runtime")
    if not all(
        isinstance(value, dict)
        for value in (source, system, profile, inputs, outputs, runtime)
    ):
        raise RuntimeError("The generated PCM-input ledger is incomplete.")
    source_hashes = source.get("source_sha256")
    if source.get("runner_path") != PCM_INPUT_RUNNER_RELATIVE_PATH or not isinstance(
        source_hashes, dict
    ):
        raise RuntimeError("The generated PCM-input runner source is not bound.")
    if set(source_hashes) != set(PCM_INPUT_GENERATOR_SOURCE_RELATIVE_PATHS):
        raise RuntimeError("The generated PCM-input source ledger is incomplete.")
    for relative, observed_sha in source_hashes.items():
        if source_bindings.get(relative) != observed_sha:
            raise RuntimeError(f"The generated PCM-input source changed: {relative}.")
    for key, expected in {
        "compound_id": record["compound_id"],
        "molecule": record["name"],
        "charge": 0,
        "multiplicity": 1,
    }.items():
        if system.get(key) != expected:
            raise RuntimeError(f"The PCM-input ledger has invalid system field: {key}.")
    if inputs.get("mol2_sha256") != record["mol2_sha256"]:
        raise RuntimeError("The generated PCM input used the wrong MOL2.")
    expected_profile = {
        "name": ("macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-exact-gto-v1"),
        "provider": "pcmsolver",
        "electrostatics_model": "iefpcm",
        "dielectric_policy": "explicit-smd-water-78.355-v1",
        "coulomb_radii_policy": "smd-water-reference-smd18-v1",
        "pcmsolver_cavity_generation": "intrinsic-probe0-noaddsph-v1",
        "tessera_area_angstrom2": 0.28,
    }
    for key, expected in expected_profile.items():
        if profile.get(key) != expected:
            raise RuntimeError(
                f"The generated PCM input has invalid profile field: {key}."
            )
    effective = outputs.get("effective_machine_input")
    if not isinstance(effective, dict) or effective.get("sha256") != parsed_sha:
        raise RuntimeError("The PCM ledger/effective-input binding failed.")
    library = runtime.get("pcmsolver_library")
    parser = runtime.get("pcmsolver_parser")
    if (
        not isinstance(library, dict)
        or library.get("sha256") != _sha256(pcmsolver_library)
        or not isinstance(parser, dict)
        or parser.get("sha256") != execution_contract.get("pcmsolver_parser_sha256")
    ):
        raise RuntimeError(
            "The PCM-input generator used a different PCMSolver runtime."
        )
    return {
        "provenance_mode": mode,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "effective_machine_input_path": str(parsed_input_path),
        "effective_machine_input_sha256": parsed_sha,
    }


def _parsed_pcm_radii_angstrom(path: Path, atom_count: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 4 or fields[:2] != ["DBL_ARRAY", "RADII"]:
            continue
        count = int(fields[2])
        if count != atom_count:
            raise RuntimeError(
                "The parsed PCMSolver input radius count does not match "
                "the solute atom count."
            )
        values = np.asarray(
            [float(value) for value in lines[index + 1 : index + 1 + count]],
            dtype=float,
        )
        if values.shape != (atom_count,) or np.any(values <= 0.0):
            raise RuntimeError(
                "The parsed PCMSolver input contains invalid atom radii."
            )
        return values * Bohr
    raise RuntimeError("The parsed PCMSolver input does not expose atom radii.")


def _projection_record(
    response: PCMSolverExternalMEPCavityResponse,
    positions_angstrom: np.ndarray,
    potential_hartree_per_e: np.ndarray,
    *,
    total_charge_e: float,
    molecular_dipole_e_angstrom: np.ndarray,
    widths_angstrom: tuple[float, ...],
    relative_spectral_cutoff: float,
) -> dict[str, object]:
    started = time.perf_counter()
    operator = FixedCavityGTOGalerkinOperator(
        response,
        positions_angstrom,
        AtomCenteredL1GTOBasis(widths_angstrom),
    )
    projection = project_surface_potential_in_pcm_energy_norm(
        operator,
        potential_hartree_per_e,
        total_charge_e=total_charge_e,
        molecular_dipole_e_angstrom=molecular_dipole_e_angstrom,
        relative_spectral_cutoff=relative_spectral_cutoff,
    )
    elapsed = time.perf_counter() - started
    return {
        "widths_angstrom": list(widths_angstrom),
        "coefficient_count": operator.coefficient_count,
        "surface_point_count": int(len(potential_hartree_per_e)),
        "elapsed_seconds": elapsed,
        "operator": operator.provenance,
        "constraint_residual_inf": projection.constraint_residual_inf,
        "retained_subspace_optimality_inf": (
            projection.retained_subspace_optimality_inf
        ),
        "full_tangent_gradient_inf": projection.full_tangent_gradient_inf,
        "relative_spectral_cutoff": projection.relative_spectral_cutoff,
        "reduced_dimension": projection.reduced_dimension,
        "effective_rank": projection.effective_rank,
        "discarded_mode_count": projection.discarded_mode_count,
        "maximum_eigenvalue_hartree": (projection.maximum_eigenvalue_hartree),
        "retained_minimum_eigenvalue_hartree": (
            projection.retained_minimum_eigenvalue_hartree
        ),
        "retained_condition_number": projection.retained_condition_number,
        "coefficient_l2_norm": projection.coefficient_l2_norm,
        "coefficient_max_abs": projection.coefficient_max_abs,
        "shifted_pythagorean_error_hartree": (
            projection.shifted_pythagorean_error_hartree
        ),
        "residual_energy_norm_squared_hartree": (
            projection.residual_energy_norm_squared_hartree
        ),
        "target_polarization_energy_hartree": (
            projection.target_polarization_energy_hartree
        ),
        "fitted_polarization_energy_hartree": (
            projection.fitted_polarization_energy_hartree
        ),
        "polarization_energy_error_hartree": (
            projection.polarization_energy_error_hartree
        ),
        "target_polarization_energy_kcal_per_mol": (
            projection.target_polarization_energy_hartree * KCAL_PER_HARTREE
        ),
        "fitted_polarization_energy_kcal_per_mol": (
            projection.fitted_polarization_energy_hartree * KCAL_PER_HARTREE
        ),
        "polarization_energy_error_kcal_per_mol": (
            projection.polarization_energy_error_hartree * KCAL_PER_HARTREE
        ),
        "projected_coefficients": np.asarray(
            projection.coefficients,
            dtype=float,
        ).tolist(),
    }


def main() -> int:
    args = _parse_args()
    if args.require_two_radial_improvement:
        raise RuntimeError(
            "Per-record two-radial improvement was not preregistered; the "
            "four-record aggregate decision owns that gate."
        )
    git_head = _require_clean_source()
    args.pyscf_python = args.pyscf_python.absolute()
    input_names = (
        "pcmsolver_library",
        "preregistration",
        "mol2",
        "parsed_pcm_input",
        "qm_checkpoint",
        "qm_ledger",
    )
    for name in input_names:
        setattr(args, name, getattr(args, name).resolve())
    if args.pcm_input_ledger is not None:
        args.pcm_input_ledger = args.pcm_input_ledger.resolve()
    for path in (getattr(args, name) for name in input_names):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.pcm_input_ledger is not None and not args.pcm_input_ledger.is_file():
        raise FileNotFoundError(args.pcm_input_ledger)

    preregistration, preregistered_record, preregistration_sha = (
        _validate_preregistration(
            args.preregistration,
            compound_id=args.compound_id,
            molecule_name=args.molecule_name,
            relative_spectral_cutoff=args.relative_spectral_cutoff,
            pcmsolver_library=args.pcmsolver_library,
        )
    )
    execution_contract = preregistration["execution_contract"]
    source_bindings = execution_contract["source_sha256"]
    pyscf_interpreter_provenance = _validate_pyscf_interpreter(
        args.pyscf_python,
        execution_contract,
    )
    qm_provenance = _validate_qm_ledger(
        args.qm_ledger,
        args.qm_checkpoint,
        args.mol2,
        record=preregistered_record,
        source_bindings=source_bindings,
        execution_contract=execution_contract,
    )
    pcm_input_provenance = _validate_pcm_input_provenance(
        args.pcm_input_ledger,
        args.parsed_pcm_input,
        args.mol2,
        record=preregistered_record,
        pcmsolver_library=args.pcmsolver_library,
        source_bindings=source_bindings,
        execution_contract=execution_contract,
    )
    artifact_id = (
        "route2-gto-pcm-energy-projection-"
        f"{args.compound_id}-cutoff-{_cutoff_token(args.relative_spectral_cutoff)}-v1"
    )
    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    if output.exists():
        raise FileExistsError(output)
    os.chdir(work_dir)

    atoms = MOL2Reader(str(args.mol2), charge=0, mult=1)
    positions_angstrom = np.asarray(atoms.get_positions(), dtype=float)
    radii_angstrom = _parsed_pcm_radii_angstrom(
        args.parsed_pcm_input,
        len(atoms),
    )
    surface_path = work_dir / "surface.npz"
    qm_mep_path = work_dir / "qm-surface-mep.npz"

    os.environ["PCMSOLVER_LIBRARY"] = str(args.pcmsolver_library)
    with PCMSolverSession(
        np.asarray(atoms.numbers, dtype=float),
        positions_angstrom / Bohr,
        args.parsed_pcm_input,
        library_path=args.pcmsolver_library,
    ) as session:
        surface_points_bohr = session.cavity_centers_bohr
        np.savez(
            surface_path,
            surface_points_bohr=surface_points_bohr,
        )

    helper_command = [
        str(args.pyscf_python),
        str(REPO_ROOT / HELPER_RELATIVE_PATH),
        "--checkpoint",
        str(args.qm_checkpoint),
        "--surface",
        str(surface_path),
        "--output",
        str(qm_mep_path),
    ]
    subprocess.run(helper_command, check=True, cwd=work_dir)
    qm = np.load(qm_mep_path)
    qm_positions = np.asarray(qm["atom_positions_angstrom"], dtype=float)
    qm_numbers = np.asarray(qm["atomic_numbers"], dtype=float)
    if qm_positions.shape != positions_angstrom.shape or not np.array_equal(
        qm_numbers.astype(int), atoms.numbers
    ):
        raise RuntimeError(
            "The QM checkpoint atom identities do not match the frozen MOL2."
        )
    geometry_error = float(np.max(np.abs(qm_positions - positions_angstrom)))
    if geometry_error > 1.0e-8:
        raise RuntimeError("The QM checkpoint geometry does not match the frozen MOL2.")
    potential = np.asarray(
        qm["surface_potential_hartree_per_e"],
        dtype=float,
    )
    total_charge = float(qm["total_charge_e"])
    dipole = np.asarray(qm["molecular_dipole_e_angstrom"], dtype=float)
    if abs(total_charge) > 1.0e-8:
        raise RuntimeError("The QM reference did not recover a neutral density.")

    with PCMSolverSession(
        np.asarray(atoms.numbers, dtype=float),
        positions_angstrom / Bohr,
        args.parsed_pcm_input,
        library_path=args.pcmsolver_library,
    ) as session:
        if not np.allclose(
            session.cavity_centers_bohr,
            surface_points_bohr,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "The fixed PCMSolver cavity changed between response sessions."
            )
        response = PCMSolverExternalMEPCavityResponse(
            session,
            cavity_radii_angstrom=radii_angstrom,
        )
        one_radial = _projection_record(
            response,
            positions_angstrom,
            potential,
            total_charge_e=total_charge,
            molecular_dipole_e_angstrom=dipole,
            widths_angstrom=ONE_RADIAL_WIDTHS,
            relative_spectral_cutoff=args.relative_spectral_cutoff,
        )
        two_radial = _projection_record(
            response,
            positions_angstrom,
            potential,
            total_charge_e=total_charge,
            molecular_dipole_e_angstrom=dipole,
            widths_angstrom=TWO_RADIAL_WIDTHS,
            relative_spectral_cutoff=args.relative_spectral_cutoff,
        )

    one_error = abs(float(one_radial["polarization_energy_error_kcal_per_mol"]))
    two_error = abs(float(two_radial["polarization_energy_error_kcal_per_mol"]))
    two_radial_improves = two_error < one_error
    if args.require_two_radial_improvement and not two_radial_improves:
        raise RuntimeError(
            "The two-radial GTO basis did not improve this molecule's PCM "
            "polarization-energy projection error."
        )

    payload = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "fixed-geometry representation feasibility canary; "
            "not a variational MACE-POLAR model"
        ),
        "claim_boundary": {
            "certifies": [
                "same-basis finite-width l<=1 GTO source/receiver reciprocity",
                "fixed-cavity PCM half-coupling identity",
                "retained-subspace PCM-energy-norm projection algebra",
                (
                    f"{args.molecule_name} one- versus two-radial "
                    "representation comparison on one QM MEP"
                ),
            ],
            "does_not_certify": [
                "MACE density-energy conjugacy",
                "self-consistent variational free energy",
                "hydration free-energy accuracy",
                "forces or a solution-phase PES",
                "optimization, transition states, scans, frequencies, or MD",
            ],
        },
        "source": {
            "git_head": git_head,
            "preregistration": {
                "path": str(args.preregistration),
                "sha256": preregistration_sha,
                "artifact_id": preregistration["artifact_id"],
            },
            "source_sha256": {
                relative: _sha256(REPO_ROOT / relative)
                for relative in SOURCE_RELATIVE_PATHS
            },
            "runner_argv": [sys.executable, *sys.argv],
            "helper_command": helper_command,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "pyscf": str(qm["pyscf_version"]),
            "pyscf_python": pyscf_interpreter_provenance,
            "pcmsolver_library": {
                "path": str(args.pcmsolver_library.resolve()),
                "sha256": _sha256(args.pcmsolver_library),
            },
        },
        "inputs": {
            "compound_id": args.compound_id,
            "molecule_name": args.molecule_name,
            "molecule": {
                "path": str(args.mol2.resolve()),
                "sha256": _sha256(args.mol2),
            },
            "parsed_pcm_input": {
                "path": str(args.parsed_pcm_input.resolve()),
                "sha256": _sha256(args.parsed_pcm_input),
                "cavity_radii_angstrom": radii_angstrom.tolist(),
                "provenance": pcm_input_provenance,
            },
            "qm_checkpoint": {
                "path": str(args.qm_checkpoint.resolve()),
                "sha256": _sha256(args.qm_checkpoint),
                "provenance": qm_provenance,
            },
            "qm_density": {
                "source": str(qm["density_source"]),
                "checkpoint_density_binding_residual_e": float(
                    qm["checkpoint_density_binding_residual_e"]
                ),
                "checkpoint_mo_orthonormality_inf": float(
                    qm["checkpoint_mo_orthonormality_inf"]
                ),
                "checkpoint_occupation_sum_e": float(qm["checkpoint_occupation_sum_e"]),
                "checkpoint_occupied_orbital_count": int(
                    qm["checkpoint_occupied_orbital_count"]
                ),
            },
            "geometry_max_abs_error_angstrom": geometry_error,
            "qm_total_charge_e": total_charge,
            "qm_molecular_dipole_e_angstrom": dipole.tolist(),
        },
        "basis_results": {
            "one_radial": one_radial,
            "two_radial": two_radial,
        },
        "gates": {
            "retained_subspace_projection_gates_passed": True,
            "full_tangent_gradient_is_ungated_diagnostic": True,
            "two_radial_improvement_required": (args.require_two_radial_improvement),
            "two_radial_improves_energy_projection": two_radial_improves,
            "one_radial_absolute_error_kcal_per_mol": one_error,
            "two_radial_absolute_error_kcal_per_mol": two_error,
            "absolute_error_improvement_kcal_per_mol": one_error - two_error,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
