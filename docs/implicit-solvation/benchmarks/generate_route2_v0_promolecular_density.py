#!/usr/bin/env python3
"""Generate the frozen positive free-atom density asset for Route-2 V0-FD-S.

The resulting radial tables are a *promolecular reference density* only.  They
are not a MACE-POLAR density, a fitted density, a cavity threshold, a
solute--solvent short-range potential, or a solvation calculation.  The
script intentionally uses only the declared isolated-atom Hartree--Fock setup
and never reads solvation labels or benchmark records.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ID = "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-promolecular-atomic-hf-def2-tzvpd-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "generate_route2_v0_promolecular_density.py"
)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
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
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read preregistration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("The preregistration must contain one JSON object.")
    return payload


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _require_clean_tracked_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "Promolecular asset generation requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    return _git("rev-parse", "HEAD")


def _require_float(value: object, *, name: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a finite float.") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = " finite and positive" if positive else " finite"
        raise RuntimeError(f"{name} must be{qualifier}.")
    return result


def _validate_preregistration(path: Path) -> dict[str, object]:
    if path.resolve() != DEFAULT_PREREGISTRATION.resolve():
        raise RuntimeError("Only the tracked promolecular preregistration is allowed.")
    protocol = _load_json(path)
    if protocol.get("protocol_id") != ARTIFACT_ID + "-prereg":
        raise RuntimeError("The promolecular preregistration identity is invalid.")
    if protocol.get("status") != "frozen-before-asset-generation":
        raise RuntimeError("The promolecular preregistration is not frozen.")
    source_hashes = protocol.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("The preregistration omits its source hash.")
    expected_hash = source_hashes.get(RUNNER_RELATIVE_PATH)
    if not isinstance(expected_hash, str) or _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH) != expected_hash:
        raise RuntimeError("The promolecular generator no longer matches preregistration.")
    contract = protocol.get("generation_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("The preregistration omits its generation contract.")
    return protocol


def _radial_grid(contract: dict[str, object]) -> np.ndarray:
    radial = contract.get("radial_grid")
    if not isinstance(radial, dict):
        raise RuntimeError("The preregistration omits the radial grid.")
    if radial.get("coordinate_rule") != "r_max * (i / (n - 1))**2":
        raise RuntimeError("Unsupported promolecular radial-grid coordinate rule.")
    try:
        point_count = int(radial.get("point_count"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Promolecular radial point count is invalid.") from exc
    if point_count < 3:
        raise RuntimeError("Promolecular radial point count must be at least three.")
    radius_max = _require_float(
        radial.get("radius_max_bohr"),
        name="Promolecular radial maximum",
        positive=True,
    )
    index = np.arange(point_count, dtype=float)
    return radius_max * (index / (point_count - 1)) ** 2


def _atomic_density(
    *,
    element: dict[str, object],
    basis: str,
    radial_grid_bohr: np.ndarray,
    spherical_probe_radius_bohr: float,
    convergence_tolerance: float,
    electron_count_tolerance: float,
    isotropy_tolerance: float,
    tail_tolerance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        from pyscf import dft, gto
        from pyscf.scf.atom_hf import AtomSphAverageRHF
    except ImportError as exc:
        raise RuntimeError(
            "This generator requires the preregistered PySCF environment."
        ) from exc

    symbol = element.get("symbol")
    if not isinstance(symbol, str):
        raise RuntimeError("Promolecular element symbol is invalid.")
    try:
        atomic_number = int(element.get("atomic_number"))
        spin = int(element.get("spin_2s"))
        expected_electron_count = float(element.get("neutral_electron_count"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Promolecular element contract is invalid for {symbol}.") from exc
    if atomic_number < 1 or expected_electron_count <= 0.0:
        raise RuntimeError(f"Promolecular element contract is invalid for {symbol}.")

    molecule = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        spin=spin,
        verbose=0,
    )
    if molecule.atom_charge(0) != atomic_number:
        raise RuntimeError(f"PySCF atomic number disagrees for {symbol}.")
    if molecule.nelectron != int(expected_electron_count):
        raise RuntimeError(f"PySCF neutral electron count disagrees for {symbol}.")

    mean_field = AtomSphAverageRHF(molecule)
    mean_field.conv_tol = convergence_tolerance
    mean_field.max_cycle = 200
    energy_hartree = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError(f"Spherically averaged atomic HF did not converge for {symbol}.")

    coordinates = np.zeros((radial_grid_bohr.size, 3), dtype=float)
    coordinates[:, 0] = radial_grid_bohr
    density_matrix = mean_field.make_rdm1()
    density = np.asarray(
        dft.numint.eval_rho(
            molecule,
            molecule.eval_gto("GTOval_sph", coordinates),
            density_matrix,
        ),
        dtype=float,
    )
    if density.shape != radial_grid_bohr.shape or not np.all(np.isfinite(density)):
        raise RuntimeError(f"Atomic density is invalid for {symbol}.")
    minimum = float(np.min(density))
    if minimum < 0.0:
        raise RuntimeError(f"Atomic density is not nonnegative for {symbol}.")

    electron_integral = float(
        4.0 * np.pi * np.trapezoid(radial_grid_bohr**2 * density, radial_grid_bohr)
    )
    electron_error = abs(electron_integral - expected_electron_count)
    if electron_error > electron_count_tolerance:
        raise RuntimeError(
            f"Atomic density normalization failed for {symbol}: {electron_error:.3e}."
        )

    probe_coordinates = np.eye(3, dtype=float) * spherical_probe_radius_bohr
    probe_density = np.asarray(
        dft.numint.eval_rho(
            molecule,
            molecule.eval_gto("GTOval_sph", probe_coordinates),
            density_matrix,
        ),
        dtype=float,
    )
    isotropy_error = float(np.max(probe_density) - np.min(probe_density))
    if isotropy_error > isotropy_tolerance:
        raise RuntimeError(
            f"Atomic density is not spherical within tolerance for {symbol}: "
            f"{isotropy_error:.3e}."
        )
    tail_density = float(density[-1])
    if tail_density > tail_tolerance:
        raise RuntimeError(
            f"Atomic density radial tail is too large for {symbol}: {tail_density:.3e}."
        )

    result = {
        "atomic_number": atomic_number,
        "density_sha256": _sha256_array(density),
        "electron_count_integral": electron_integral,
        "electron_count_integral_absolute_error": electron_error,
        "energy_hartree": energy_hartree,
        "minimum_density_e_per_bohr3": minimum,
        "neutral_electron_count": expected_electron_count,
        "radial_tail_density_e_per_bohr3": tail_density,
        "scf_converged": True,
        "spin_2s": spin,
        "spherical_probe_maximum_difference_e_per_bohr3": isotropy_error,
        "symbol": symbol,
    }
    return density, result


def main() -> int:
    arguments = _parse_args()
    protocol = _validate_preregistration(arguments.preregistration)
    if arguments.output.exists() or arguments.manifest.exists():
        raise RuntimeError("Promolecular output paths must not already exist.")
    execution_head = _require_clean_tracked_checkout()

    contract = protocol["generation_contract"]
    assert isinstance(contract, dict)
    expected_pyscf_version = contract.get("pyscf_version")
    basis = contract.get("basis")
    elements = contract.get("elements")
    if not isinstance(expected_pyscf_version, str) or not isinstance(basis, str):
        raise RuntimeError("The promolecular method identity is invalid.")
    if not isinstance(elements, list) or not elements:
        raise RuntimeError("The promolecular element set is invalid.")
    try:
        import pyscf
    except ImportError as exc:
        raise RuntimeError("This generator requires PySCF.") from exc
    if pyscf.__version__ != expected_pyscf_version:
        raise RuntimeError(
            "PySCF version does not match preregistration "
            f"({pyscf.__version__!r} != {expected_pyscf_version!r})."
        )

    radial_grid_bohr = _radial_grid(contract)
    validation = contract.get("validation_gates")
    if not isinstance(validation, dict):
        raise RuntimeError("The promolecular validation gates are missing.")
    spherical_probe_radius_bohr = _require_float(
        validation.get("spherical_probe_radius_bohr"),
        name="Spherical probe radius",
        positive=True,
    )
    convergence_tolerance = _require_float(
        validation.get("scf_convergence_tolerance_hartree"),
        name="SCF convergence tolerance",
        positive=True,
    )
    electron_count_tolerance = _require_float(
        validation.get("electron_count_absolute_error_max"),
        name="Electron-count tolerance",
        positive=True,
    )
    isotropy_tolerance = _require_float(
        validation.get("spherical_probe_maximum_difference_max"),
        name="Spherical isotropy tolerance",
        positive=True,
    )
    tail_tolerance = _require_float(
        validation.get("radial_tail_density_max_e_per_bohr3"),
        name="Radial-tail tolerance",
        positive=True,
    )

    arrays: dict[str, np.ndarray] = {"radial_grid_bohr": radial_grid_bohr}
    results: list[dict[str, object]] = []
    for raw_element in elements:
        if not isinstance(raw_element, dict):
            raise RuntimeError("Promolecular element entry is invalid.")
        density, result = _atomic_density(
            element=raw_element,
            basis=basis,
            radial_grid_bohr=radial_grid_bohr,
            spherical_probe_radius_bohr=spherical_probe_radius_bohr,
            convergence_tolerance=convergence_tolerance,
            electron_count_tolerance=electron_count_tolerance,
            isotropy_tolerance=isotropy_tolerance,
            tail_tolerance=tail_tolerance,
        )
        arrays[f"density_Z{result['atomic_number']}"] = density
        results.append(result)
    arrays["atomic_numbers"] = np.asarray(
        [result["atomic_number"] for result in results],
        dtype=np.int64,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **arrays)
    manifest = {
        "artifact": ARTIFACT_ID,
        "claim_boundary": (
            "This artifact is a frozen, positive, spherically averaged free-atom "
            "HF promolecular reference-density source. It is not a MACE-POLAR "
            "electron density, a fitted solvation component, a short-range "
            "solute-solvent potential, a liquid functional, a force/PES result, "
            "or a solvation-accuracy claim."
        ),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": execution_head,
        "hard_constraints": protocol["hard_constraints"],
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": _sha256(arguments.preregistration),
        },
        "radial_grid_sha256": _sha256_array(radial_grid_bohr),
        "results": results,
        "runtime": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pyscf": pyscf.__version__,
            "python": sys.version,
        },
        "schema_version": 1,
        "source_files_sha256": {RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)},
        "status": "pass",
        "table": {
            "path": str(arguments.output.resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(arguments.output),
        },
    }
    _write_exclusive_json(arguments.manifest, manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
