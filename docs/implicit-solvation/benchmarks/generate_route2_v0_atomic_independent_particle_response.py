#!/usr/bin/env python3
"""Generate the frozen Route-2 V0 atomic independent-particle response asset.

This is an offline source-asset generator, not a user-facing QM calculation.
It reads no solvation labels and performs no response fitting: every retained
transition is selected solely by the frozen spherical atomic HF occupations and
canonical orbital energies.  The generated table is a baseline ``C0`` for the
V0-RK covariance completion, not a molecular solvation method by itself.
"""

from __future__ import annotations

# Set the numerical runtime before NumPy or PySCF imports BLAS/OpenMP.
import os

THREAD_COUNT = 1
THREAD_ENVIRONMENT_VARIABLES = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
DETERMINISTIC_ARCHIVE_CONTRACT = {
    "compression": "ZIP_DEFLATED-level-9",
    "format": "npz",
    "member_order": "lexicographic-array-key",
    "zip_member_timestamp_utc": "1980-01-01T00:00:00+00:00",
}
for _thread_environment_variable in THREAD_ENVIRONMENT_VARIABLES:
    os.environ[_thread_environment_variable] = str(THREAD_COUNT)

import argparse
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ID = "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-independent-particle-hf-def2-tzvpd-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "generate_route2_v0_atomic_independent_particle_response.py"
)
SOURCE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_independent_particle_response.py"
)
SOURCE_RELATIVE_PATHS = (RUNNER_RELATIVE_PATH, SOURCE_MODULE_RELATIVE_PATH)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read preregistration {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError("The preregistration must be one JSON object.")
    return payload


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=False,
    ) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.ascontiguousarray(np.asarray(arrays[name])),
                allow_pickle=False,
            )
            entry = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, array_buffer.getvalue(), compresslevel=9)
    return buffer.getvalue()


def _write_exclusive_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_deterministic_npz_bytes(arrays))


def _require_clean_tracked_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "Atomic independent-particle asset generation requires a clean tracked "
            f"checkout; git reported:\n{status}"
        )
    return _git("rev-parse", "HEAD")


def _load_source_module() -> ModuleType:
    module_path = REPO_ROOT / SOURCE_MODULE_RELATIVE_PATH
    specification = importlib.util.spec_from_file_location(
        "route2_v0_atomic_independent_particle_response_generator_source",
        module_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load the atomic independent-particle source module.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _validate_preregistration(path: Path, *, threads: int) -> dict[str, object]:
    if path.resolve() != DEFAULT_PREREGISTRATION.resolve():
        raise RuntimeError("Only the tracked atomic-IP preregistration is allowed.")
    protocol = _load_json(path)
    if protocol.get("protocol_id") != ARTIFACT_ID + "-prereg":
        raise RuntimeError("Atomic-IP preregistration identity is invalid.")
    if protocol.get("status") != "frozen-before-asset-generation":
        raise RuntimeError("Atomic-IP preregistration is not frozen.")
    source_hashes = protocol.get("source_sha256")
    expected_hashes = {
        relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
    }
    if source_hashes != expected_hashes:
        raise RuntimeError("Atomic-IP generator no longer matches preregistration.")
    contract = protocol.get("generation_contract")
    if not isinstance(contract, dict):
        raise TypeError("Atomic-IP preregistration has no generation contract.")
    expected_environment = {
        name: str(THREAD_COUNT) for name in THREAD_ENVIRONMENT_VARIABLES
    }
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError("Atomic-IP generation runtime is missing.")
    if (
        threads != THREAD_COUNT
        or runtime.get("threads") != THREAD_COUNT
        or runtime.get("thread_environment") != expected_environment
        or runtime.get("python_resolved_sha256")
        != _sha256(Path(sys.executable).resolve())
        or contract.get("deterministic_archive") != DETERMINISTIC_ARCHIVE_CONTRACT
    ):
        raise RuntimeError("Atomic-IP numerical runtime changed after freeze.")
    return protocol


def _element_response(
    *,
    source_module: ModuleType,
    element: dict[str, object],
    basis: str,
    convergence_tolerance: float,
    numerical_relative_tolerance: float,
) -> tuple[object, dict[str, object]]:
    try:
        from pyscf import gto
        from pyscf.scf.atom_hf import AtomSphAverageRHF
    except ImportError as error:
        raise RuntimeError("Atomic-IP generation requires the frozen PySCF runtime.") from error
    try:
        symbol = str(element["symbol"])
        atomic_number = int(element["atomic_number"])
        spin_2s = int(element["spin_2s"])
        expected_electron_count = int(element["neutral_electron_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise TypeError("Atomic-IP element contract is invalid.") from error
    molecule = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        spin=spin_2s,
        verbose=0,
    )
    if molecule.atom_charge(0) != atomic_number or molecule.nelectron != expected_electron_count:
        raise RuntimeError(f"Atomic-IP element identity disagrees for {symbol}.")
    mean_field = AtomSphAverageRHF(molecule)
    mean_field.conv_tol = convergence_tolerance
    mean_field.max_cycle = 200
    energy_hartree = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError(f"Spherical atomic HF did not converge for {symbol}.")
    response = source_module.build_route2_v0_atomic_independent_particle_response(
        atomic_number=atomic_number,
        symbol=symbol,
        basis=basis,
        spin_2s=spin_2s,
        mo_coefficients=mean_field.mo_coeff,
        orbital_energies_hartree=mean_field.mo_energy,
        orbital_occupations=mean_field.mo_occ,
        overlap_matrix=molecule.intor("int1e_ovlp"),
        position_integrals_ebohr=molecule.intor("int1e_r", comp=3),
        numerical_relative_tolerance=numerical_relative_tolerance,
    )
    covariance = response.atomic_polarizability_bohr3
    isotropy_error = float(
        np.linalg.norm(covariance - np.trace(covariance) * np.eye(3) / 3.0, ord=2)
    )
    result = {
        "array_sha256": {
            "mo_coefficients": _sha256_array(response.mo_coefficients),
            "orbital_energies_hartree": _sha256_array(response.orbital_energies_hartree),
            "orbital_occupations": _sha256_array(response.orbital_occupations),
            "transition_lower_indices": _sha256_array(response.transition_lower_indices),
            "transition_upper_indices": _sha256_array(response.transition_upper_indices),
            "transition_weights_hartree_inverse": _sha256_array(
                response.transition_weights_hartree_inverse
            ),
            "transition_dipoles_ebohr": _sha256_array(response.transition_dipoles_ebohr),
            "transition_charges_e": _sha256_array(response.transition_charges_e),
        },
        "atomic_dipole_covariance_bohr3": covariance.tolist(),
        "atomic_dipole_covariance_isotropy_error": isotropy_error,
        "atomic_dipole_covariance_minimum_eigenvalue": float(
            np.min(np.linalg.eigvalsh(covariance))
        ),
        "atomic_number": atomic_number,
        "energy_hartree": energy_hartree,
        "maximum_transition_charge_e": float(
            np.max(np.abs(response.transition_charges_e))
        ),
        "mo_count": int(response.mo_coefficients.shape[1]),
        "nao": int(response.mo_coefficients.shape[0]),
        "scf_converged": True,
        "spin_2s": spin_2s,
        "symbol": symbol,
        "transition_count": response.transition_count,
    }
    return response, result


def main() -> int:
    arguments = _parse_arguments()
    output = arguments.output.resolve()
    manifest_path = arguments.manifest.resolve()
    if output.exists() or manifest_path.exists():
        raise RuntimeError("Atomic-IP output paths must not already exist.")
    protocol = _validate_preregistration(arguments.preregistration, threads=arguments.threads)
    execution_head = _require_clean_tracked_checkout()
    contract = protocol["generation_contract"]
    assert isinstance(contract, dict)
    try:
        basis = str(contract["basis"])
        elements = contract["elements"]
        convergence_tolerance = float(contract["scf_convergence_tolerance_hartree"])
        numerical_relative_tolerance = float(contract["numerical_relative_tolerance"])
        expected_pyscf_version = str(contract["pyscf_version"])
        validation_gates = contract["validation_gates"]
    except (KeyError, TypeError, ValueError) as error:
        raise TypeError("Atomic-IP generation contract is invalid.") from error
    if not isinstance(elements, list) or not elements:
        raise TypeError("Atomic-IP generation requires at least one element.")
    if not isinstance(validation_gates, dict):
        raise TypeError("Atomic-IP validation gates are invalid.")
    try:
        import pyscf
        from pyscf import lib
    except ImportError as error:
        raise RuntimeError("Atomic-IP generation requires PySCF.") from error
    if pyscf.__version__ != expected_pyscf_version:
        raise RuntimeError("Atomic-IP PySCF version differs from preregistration.")
    runtime_contract = contract["runtime"]
    assert isinstance(runtime_contract, dict)
    if _sha256(Path(pyscf.__file__).resolve()) != runtime_contract.get(
        "pyscf_init_sha256"
    ):
        raise RuntimeError("Atomic-IP PySCF package identity changed after freeze.")
    lib.num_threads(arguments.threads)
    if int(lib.num_threads()) != arguments.threads:
        raise RuntimeError("PySCF did not honor the frozen atomic-IP thread count.")
    source_module = _load_source_module()
    arrays: dict[str, np.ndarray] = {}
    results: list[dict[str, object]] = []
    responses: list[object] = []
    for raw_element in elements:
        if not isinstance(raw_element, dict):
            raise TypeError("Atomic-IP element entry must be an object.")
        response, result = _element_response(
            source_module=source_module,
            element=raw_element,
            basis=basis,
            convergence_tolerance=convergence_tolerance,
            numerical_relative_tolerance=numerical_relative_tolerance,
        )
        if (
            float(result["maximum_transition_charge_e"])
            > float(validation_gates["maximum_transition_charge_e_max"])
            or float(result["atomic_dipole_covariance_isotropy_error"])
            > float(validation_gates["atomic_dipole_covariance_isotropy_error_max"])
            or float(result["atomic_dipole_covariance_minimum_eigenvalue"])
            <= float(validation_gates["atomic_dipole_covariance_minimum_eigenvalue_min"])
        ):
            raise RuntimeError(
                "Atomic-IP response failed a frozen neutrality, isotropy, or "
                "positive-covariance gate."
            )
        prefix = f"Z{response.atomic_number}"
        arrays.update(
            {
                f"mo_coefficients_{prefix}": response.mo_coefficients,
                f"orbital_energies_hartree_{prefix}": response.orbital_energies_hartree,
                f"orbital_occupations_{prefix}": response.orbital_occupations,
                f"transition_lower_indices_{prefix}": response.transition_lower_indices,
                f"transition_upper_indices_{prefix}": response.transition_upper_indices,
                f"transition_weights_hartree_inverse_{prefix}": (
                    response.transition_weights_hartree_inverse
                ),
                f"transition_dipoles_ebohr_{prefix}": response.transition_dipoles_ebohr,
                f"transition_charges_e_{prefix}": response.transition_charges_e,
            }
        )
        responses.append(response)
        results.append(result)
    results.sort(key=lambda result: int(result["atomic_number"]))
    arrays["atomic_numbers"] = np.asarray(
        [response.atomic_number for response in sorted(responses, key=lambda item: item.atomic_number)],
        dtype=np.int64,
    )
    _write_exclusive_deterministic_npz(output, arrays)
    manifest = {
        "artifact": ARTIFACT_ID,
        "claim_boundary": (
            "This is a frozen isolated-atom independent-particle response "
            "baseline. It is not a molecular electronic functional, a "
            "continuum, a force/PES result, a total solvation method, or an "
            "accuracy result. It reads no solvation labels and is eligible only "
            "for the separately preregistered V0-RK gas-phase source falsifier."
        ),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": execution_head,
        "generation_contract": contract,
        "hard_constraints": protocol["hard_constraints"],
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": _sha256(arguments.preregistration),
            "protocol_id": protocol["protocol_id"],
        },
        "results": results,
        "runtime": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pyscf": pyscf.__version__,
            "pyscf_init_path": str(Path(pyscf.__file__).resolve()),
            "pyscf_init_sha256": _sha256(Path(pyscf.__file__).resolve()),
            "pyscf_lib_num_threads": int(lib.num_threads()),
            "python": sys.version,
            "python_resolved_sha256": _sha256(Path(sys.executable).resolve()),
            "thread_environment": {
                name: os.environ[name] for name in THREAD_ENVIRONMENT_VARIABLES
            },
            "threads": arguments.threads,
        },
        "schema_version": 1,
        "source_files_sha256": {
            relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
        },
        "status": "pass",
        "table": {"path": str(output), "sha256": _sha256(output)},
    }
    _write_exclusive_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
