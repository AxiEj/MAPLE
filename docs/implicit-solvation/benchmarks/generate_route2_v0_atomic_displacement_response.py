#!/usr/bin/env python3
"""Generate a frozen free-atom translation-tangent response asset for Route-2 V0.

The asset contains enclosed electron counts from independently generated
spherical atomic Hartree-Fock densities. At runtime those counts define the
near-field potential of an infinitesimally translated electron cloud with a
prescribed induced dipole.

This is not an all-electron molecular density, a fitted electrostatic model, a
continuum calculation, a solvation prediction, or a force/PES result. It reads
no solvation labels. It is one narrowly declared physical input for a
gas-phase source falsifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.route2_v0_atomic_displacement_response import (
    V0_ATOMIC_DISPLACEMENT_RESPONSE_ARTIFACT,
)

PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-displacement-hf-def2-tzvpd-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "generate_route2_v0_atomic_displacement_response.py"
)
SOURCE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_displacement_response.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    SOURCE_MODULE_RELATIVE_PATH,
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
        raise TypeError("The preregistration must contain one JSON object.")
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
            "Atomic-displacement asset generation requires a clean tracked "
            f"checkout; git reported:\n{status}"
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
        raise RuntimeError(
            "Only the tracked atomic-displacement preregistration is allowed."
        )
    protocol = _load_json(path)
    if (
        protocol.get("protocol_id")
        != V0_ATOMIC_DISPLACEMENT_RESPONSE_ARTIFACT + "-prereg"
    ):
        raise RuntimeError(
            "The atomic-displacement preregistration identity is invalid."
        )
    if protocol.get("status") != "frozen-before-asset-generation":
        raise RuntimeError("The atomic-displacement preregistration is not frozen.")
    source_hashes = protocol.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise TypeError("The atomic-displacement preregistration omits source hashes.")
    expected = {
        relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
    }
    if source_hashes != expected:
        raise RuntimeError(
            "The atomic-displacement generator no longer matches preregistration."
        )
    contract = protocol.get("generation_contract")
    if not isinstance(contract, dict):
        raise TypeError("The atomic-displacement preregistration lacks a contract.")
    return protocol


def _radial_grid(contract: dict[str, object]) -> np.ndarray:
    radial = contract.get("radial_grid")
    if not isinstance(radial, dict):
        raise TypeError("The atomic-displacement contract omits the radial grid.")
    if radial.get("coordinate_rule") != "r_max * (i / (n - 1))**2":
        raise RuntimeError("Unsupported atomic-displacement radial-grid rule.")
    try:
        point_count = int(radial.get("point_count"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Atomic-displacement radial point count is invalid."
        ) from exc
    if point_count < 3:
        raise RuntimeError("Atomic-displacement radial grid must contain >= 3 points.")
    radius_max = _require_float(
        radial.get("radius_max_bohr"),
        name="Atomic-displacement radial maximum",
        positive=True,
    )
    index = np.arange(point_count, dtype=float)
    return radius_max * (index / (point_count - 1)) ** 2


def _cumulative_trapezoid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    integrand = np.asarray(values, dtype=float)
    if integrand.shape != grid.shape or not np.all(np.isfinite(integrand)):
        raise RuntimeError("Atomic-displacement radial integral input is invalid.")
    result = np.empty_like(integrand)
    result[0] = 0.0
    result[1:] = np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid),
        dtype=float,
    )
    return result


def _atomic_enclosed_electrons(
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
        raise TypeError("Atomic-displacement element symbol is invalid.")
    try:
        atomic_number = int(element.get("atomic_number"))
        spin = int(element.get("spin_2s"))
        expected_electron_count = float(element.get("neutral_electron_count"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Atomic-displacement element contract is invalid for {symbol}."
        ) from exc
    if atomic_number < 1 or expected_electron_count <= 0.0:
        raise RuntimeError(
            f"Atomic-displacement element contract is invalid for {symbol}."
        )

    molecule = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        spin=spin,
        verbose=0,
    )
    if molecule.atom_charge(0) != atomic_number:
        raise RuntimeError(f"PySCF atomic number disagrees for {symbol}.")
    if molecule.nelectron != int(expected_electron_count):
        raise RuntimeError(f"PySCF electron count disagrees for {symbol}.")

    mean_field = AtomSphAverageRHF(molecule)
    mean_field.conv_tol = convergence_tolerance
    mean_field.max_cycle = 200
    energy_hartree = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError(f"Spherical atomic HF did not converge for {symbol}.")

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
        raise RuntimeError(f"Atomic-displacement density is invalid for {symbol}.")
    if float(np.min(density)) < 0.0:
        raise RuntimeError(f"Atomic-displacement density is negative for {symbol}.")

    raw_enclosed = _cumulative_trapezoid(
        4.0 * np.pi * radial_grid_bohr**2 * density,
        radial_grid_bohr,
    )
    raw_total = float(raw_enclosed[-1])
    normalization_error = abs(raw_total - expected_electron_count)
    if normalization_error > electron_count_tolerance:
        raise RuntimeError(
            f"Atomic-displacement normalization failed for {symbol}: "
            f"{normalization_error:.3e}."
        )
    normalization_scale = expected_electron_count / raw_total
    enclosed = raw_enclosed * normalization_scale
    enclosed[-1] = expected_electron_count

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
            f"Atomic-displacement density is not spherical for {symbol}: "
            f"{isotropy_error:.3e}."
        )
    tail_density = float(density[-1])
    if tail_density > tail_tolerance:
        raise RuntimeError(
            f"Atomic-displacement radial tail is too large for {symbol}: "
            f"{tail_density:.3e}."
        )

    result = {
        "atomic_number": atomic_number,
        "density_sha256": _sha256_array(density),
        "electron_count_integral_before_normalization": raw_total,
        "electron_count_integral_absolute_error": normalization_error,
        "enclosed_electrons_sha256": _sha256_array(enclosed),
        "energy_hartree": energy_hartree,
        "normalization_scale": normalization_scale,
        "radial_tail_density_e_per_bohr3": tail_density,
        "scf_converged": True,
        "spin_2s": spin,
        "spherical_probe_maximum_difference_e_per_bohr3": isotropy_error,
        "symbol": symbol,
    }
    return enclosed, result


def main() -> int:
    arguments = _parse_args()
    protocol = _validate_preregistration(arguments.preregistration)
    if arguments.output.exists() or arguments.manifest.exists():
        raise RuntimeError("Atomic-displacement output paths must not already exist.")
    execution_head = _require_clean_tracked_checkout()

    contract = protocol["generation_contract"]
    assert isinstance(contract, dict)
    expected_pyscf_version = contract.get("pyscf_version")
    basis = contract.get("basis")
    elements = contract.get("elements")
    if not isinstance(expected_pyscf_version, str) or not isinstance(basis, str):
        raise TypeError("Atomic-displacement method identity is invalid.")
    if not isinstance(elements, list) or not elements:
        raise RuntimeError("Atomic-displacement element set is invalid.")
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
        raise TypeError("Atomic-displacement validation gates are missing.")
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
            raise TypeError("Atomic-displacement element entry is invalid.")
        enclosed, result = _atomic_enclosed_electrons(
            element=raw_element,
            basis=basis,
            radial_grid_bohr=radial_grid_bohr,
            spherical_probe_radius_bohr=spherical_probe_radius_bohr,
            convergence_tolerance=convergence_tolerance,
            electron_count_tolerance=electron_count_tolerance,
            isotropy_tolerance=isotropy_tolerance,
            tail_tolerance=tail_tolerance,
        )
        arrays[f"enclosed_electrons_Z{result['atomic_number']}"] = enclosed
        results.append(result)
    arrays["atomic_numbers"] = np.asarray(
        [result["atomic_number"] for result in results],
        dtype=np.int64,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **arrays)
    manifest = {
        "artifact": V0_ATOMIC_DISPLACEMENT_RESPONSE_ARTIFACT,
        "claim_boundary": (
            "This artifact is a frozen free-atom HF translation-tangent "
            "spatial-response source. It is not a MACE electron density, "
            "a fitted response, a common GTO/continuum basis, a PCM source, "
            "a total solvation model, a force/PES result, or an accuracy claim."
        ),
        "execution_git_head": execution_head,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "source_files_sha256": {
            relative: _sha256(REPO_ROOT / relative)
            for relative in SOURCE_RELATIVE_PATHS
        },
        "status": "pass",
        "table": {
            "path": str(arguments.output),
            "sha256": _sha256(arguments.output),
        },
    }
    _write_exclusive_json(arguments.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
