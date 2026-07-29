#!/usr/bin/env python3
"""Run the source-bound water V0-AQ-E auxiliary QM--PCM control.

This isolated PySCF runner deliberately emits only the auxiliary gas and
PCM-SCF stationary total-energy states and their analytic nuclear gradients.
It neither imports MAPLE/MACE nor evaluates experimental solvation data.
The parent V0-AQ ledger may add the resulting gas-to-solvent difference to an
unchanged MACE gas energy, but that composite operation is intentionally not
executed in this one-water structural control.
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
import time

import numpy as np
import pyscf
from pyscf import dft, gto, lib
from pyscf.solvent import pcm

REPO_ROOT = Path(__file__).resolve().parents[3]
PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/" "route2-v0-aq-water-pcm-prereg-v1.json"
)
GEOMETRY_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-aq-water-geometry-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/" "run_route2_v0_auxiliary_qm_pcm_water.py"
)
EXPECTED_ELEMENTS = ("O", "H", "H")
QM_METHOD = {
    "implementation": "PySCF",
    "pyscf_version": "2.13.1",
    "electronic_structure": "omegaB97M-V",
    "pyscf_xc_token": "wb97m-v",
    "basis": "def2-tzvpd",
    "reference": "RKS",
    "density_fitting": True,
    "charge": 0,
    "spin": 0,
    "semilocal_grid_level": 3,
    "nonlocal_grid_profile": "50x194-SG1",
    "scf_energy_tolerance_hartree": 1.0e-10,
    "scf_gradient_tolerance": 1.0e-7,
    "maximum_scf_cycles": 100,
}
PCM_CONTROL = {
    "method": "IEFPCM",
    "dielectric_constant": 78.3553,
    "vdw_scale": 1.0,
    "probe_radius_angstrom": 0.0,
    "lebedev_order": 17,
    "surface_discretization_method": "SWIG",
    "standard_state_energy_included": False,
    "empirical_cds_included": False,
}
NUMERICAL_GATES = {
    "orbital_gradient_inf_max": 1.0e-6,
    "translation_gradient_inf_hartree_per_bohr_max": 1.0e-8,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=REPO_ROOT / PREREGISTRATION_RELATIVE_PATH,
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=REPO_ROOT / GEOMETRY_RELATIVE_PATH,
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object.")
    return payload


def _require_clean_tracked_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The V0-AQ-E control requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative_path in (
        PREREGISTRATION_RELATIVE_PATH,
        RUNNER_RELATIVE_PATH,
        GEOMETRY_RELATIVE_PATH,
    ):
        _git("ls-files", "--error-unmatch", relative_path)
    return _git("rev-parse", "HEAD")


def _load_geometry(
    path: Path,
) -> tuple[dict[str, object], list[tuple[str, tuple[float, ...]]]]:
    payload = _load_json_object(path, label="V0-AQ water geometry")
    if payload.get("geometry_id") != "route2-v0-aq-water-ase-g2-v1":
        raise RuntimeError("The V0-AQ-E control requires its frozen water geometry.")
    symbols = tuple(payload.get("elements", ()))
    positions = np.asarray(payload.get("positions_angstrom"), dtype=float)
    if symbols != EXPECTED_ELEMENTS or positions.shape != (3, 3):
        raise RuntimeError("The frozen water geometry has the wrong atoms or shape.")
    if not np.all(np.isfinite(positions)):
        raise RuntimeError("The frozen water geometry contains nonfinite coordinates.")
    if (
        payload.get("charge") != QM_METHOD["charge"]
        or payload.get("spin") != QM_METHOD["spin"]
    ):
        raise RuntimeError("The frozen water geometry changed charge or spin.")
    atoms = [
        (symbol, tuple(float(value) for value in position))
        for symbol, position in zip(symbols, positions)
    ]
    return payload, atoms


def _validate_preregistration(
    path: Path,
    *,
    geometry: Path,
    threads: int,
    max_memory_mb: int,
) -> dict[str, object]:
    if path.resolve() != (REPO_ROOT / PREREGISTRATION_RELATIVE_PATH).resolve():
        raise RuntimeError(
            "The V0-AQ-E runner accepts only its tracked preregistration."
        )
    if geometry.resolve() != (REPO_ROOT / GEOMETRY_RELATIVE_PATH).resolve():
        raise RuntimeError("The V0-AQ-E runner accepts only its tracked geometry.")
    preregistration = _load_json_object(path, label="V0-AQ-E preregistration")
    if (
        preregistration.get("protocol_id") != "route2-v0-aq-water-pcm-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The V0-AQ-E preregistration identity is invalid.")
    if preregistration.get("qm_method") != QM_METHOD:
        raise RuntimeError("The V0-AQ-E QM method changed after preregistration.")
    if preregistration.get("pcm_control") != PCM_CONTROL:
        raise RuntimeError("The V0-AQ-E PCM control changed after preregistration.")
    if preregistration.get("numerical_gates") != NUMERICAL_GATES:
        raise RuntimeError("The V0-AQ-E numerical gates changed after preregistration.")
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("The V0-AQ-E preregistration omits its execution contract.")
    expected_sources = contract.get("source_sha256")
    if expected_sources != {
        RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
    }:
        raise RuntimeError("The V0-AQ-E runner source changed after preregistration.")
    expected_inputs = contract.get("input_sha256")
    if not isinstance(expected_inputs, dict) or expected_inputs.get(
        GEOMETRY_RELATIVE_PATH
    ) != _sha256(geometry):
        raise RuntimeError("The V0-AQ-E geometry changed after preregistration.")
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("The V0-AQ-E preregistration omits runtime provenance.")
    executable = Path(sys.executable).resolve()
    if (
        runtime.get("python_resolved_sha256") != _sha256(executable)
        or runtime.get("pyscf_init_sha256") != _sha256(Path(pyscf.__file__).resolve())
        or runtime.get("pyscf_version") != pyscf.__version__
        or runtime.get("threads") != threads
        or runtime.get("max_memory_mb") != max_memory_mb
    ):
        raise RuntimeError("The V0-AQ-E runtime changed after preregistration.")
    return preregistration


def _make_rks(molecule, *, max_memory_mb: int):
    mean_field = dft.RKS(molecule, xc=QM_METHOD["pyscf_xc_token"]).density_fit()
    mean_field.grids.level = QM_METHOD["semilocal_grid_level"]
    elements = sorted({molecule.atom_symbol(index) for index in range(molecule.natm)})
    mean_field.nlcgrids.atom_grid = {symbol: (50, 194) for symbol in elements}
    mean_field.nlcgrids.prune = dft.gen_grid.sg1_prune
    mean_field.conv_tol = QM_METHOD["scf_energy_tolerance_hartree"]
    mean_field.conv_tol_grad = QM_METHOD["scf_gradient_tolerance"]
    mean_field.max_cycle = QM_METHOD["maximum_scf_cycles"]
    mean_field.max_memory = max_memory_mb
    return mean_field


def _orbital_gradient_inf(mean_field) -> float:
    gradient = mean_field.get_grad(
        mean_field.mo_coeff,
        mean_field.mo_occ,
        mean_field.get_fock(),
    )
    return float(np.max(np.abs(np.asarray(gradient, dtype=float))))


def _run_stationary_state(mean_field, *, phase: str) -> dict[str, object]:
    started = time.perf_counter()
    total_energy = float(mean_field.kernel())
    elapsed = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError(f"The {phase} auxiliary QM SCF did not converge.")
    orbital_gradient = _orbital_gradient_inf(mean_field)
    if (
        not np.isfinite(orbital_gradient)
        or orbital_gradient > NUMERICAL_GATES["orbital_gradient_inf_max"]
    ):
        raise RuntimeError(f"The {phase} auxiliary QM orbital gradient failed.")
    nuclear_gradient = np.asarray(mean_field.nuc_grad_method().kernel(), dtype=float)
    if nuclear_gradient.shape != (mean_field.mol.natm, 3) or not np.all(
        np.isfinite(nuclear_gradient)
    ):
        raise RuntimeError(f"The {phase} auxiliary QM nuclear gradient is invalid.")
    translation_gradient = np.sum(nuclear_gradient, axis=0)
    translation_inf = float(np.max(np.abs(translation_gradient)))
    if (
        translation_inf
        > NUMERICAL_GATES["translation_gradient_inf_hartree_per_bohr_max"]
    ):
        raise RuntimeError(f"The {phase} auxiliary QM translation gradient failed.")
    return {
        "phase": phase,
        "total_stationary_energy_hartree": total_energy,
        "nuclear_gradient_hartree_per_bohr": nuclear_gradient.tolist(),
        "orbital_gradient_inf": orbital_gradient,
        "scf_converged": True,
        "scf_elapsed_seconds": elapsed,
        "semilocal_grid_point_count": int(mean_field.grids.coords.shape[0]),
        "nonlocal_grid_point_count": int(mean_field.nlcgrids.coords.shape[0]),
        "translation_gradient_hartree_per_bohr": translation_gradient.tolist(),
        "translation_gradient_inf_hartree_per_bohr": translation_inf,
    }


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def main() -> int:
    args = _parse_args()
    preregistration_path = args.preregistration.resolve()
    geometry_path = args.geometry.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.threads != 8 or args.max_memory_mb != 8000:
        raise RuntimeError("The V0-AQ-E control accepts only the frozen runtime.")
    if pyscf.__version__ != QM_METHOD["pyscf_version"]:
        raise RuntimeError("The V0-AQ-E control requires PySCF 2.13.1.")
    head = _require_clean_tracked_checkout()
    preregistration = _validate_preregistration(
        preregistration_path,
        geometry=geometry_path,
        threads=args.threads,
        max_memory_mb=args.max_memory_mb,
    )
    geometry, atoms = _load_geometry(geometry_path)
    lib.num_threads(args.threads)
    molecule = gto.M(
        atom=atoms,
        basis=QM_METHOD["basis"],
        charge=QM_METHOD["charge"],
        spin=QM_METHOD["spin"],
        unit="Angstrom",
        symmetry=False,
        verbose=0,
        max_memory=args.max_memory_mb,
    )
    started = time.perf_counter()
    gas_mean_field = _make_rks(molecule, max_memory_mb=args.max_memory_mb)
    gas_state = _run_stationary_state(gas_mean_field, phase="gas")

    pcm_object = pcm.PCM(molecule)
    pcm_object.method = PCM_CONTROL["method"]
    pcm_object.eps = PCM_CONTROL["dielectric_constant"]
    pcm_object.vdw_scale = PCM_CONTROL["vdw_scale"]
    pcm_object.r_probe = PCM_CONTROL["probe_radius_angstrom"]
    pcm_object.lebedev_order = PCM_CONTROL["lebedev_order"]
    pcm_object.surface_discretization_method = PCM_CONTROL[
        "surface_discretization_method"
    ]
    solvated_mean_field = _make_rks(molecule, max_memory_mb=args.max_memory_mb).PCM(
        pcm_object
    )
    solvated_state = _run_stationary_state(solvated_mean_field, phase="solvated")
    surface = solvated_mean_field.with_solvent.surface
    surface_points = np.asarray(surface["grid_coords"], dtype=float)
    if (
        surface_points.ndim != 2
        or surface_points.shape[0] == 0
        or surface_points.shape[1] != 3
        or not np.all(np.isfinite(surface_points))
    ):
        raise RuntimeError("The V0-AQ-E PCM surface is invalid.")

    auxiliary_energy = float(
        solvated_state["total_stationary_energy_hartree"]
        - gas_state["total_stationary_energy_hartree"]
    )
    auxiliary_gradient = np.asarray(
        solvated_state["nuclear_gradient_hartree_per_bohr"], dtype=float
    ) - np.asarray(gas_state["nuclear_gradient_hartree_per_bohr"], dtype=float)
    result = {
        "artifact_id": "route2-v0-aq-water-pcm-control-v1",
        "schema_version": 1,
        "status": "pass-fixed-geometry-electronic-continuum-control",
        "claim_boundary": "This records a source-bound self-consistent auxiliary QM--PCM gas-to-liquid difference at one fixed water geometry. It is not a MACE density, public Route-2 profile, total solvation free energy, solvent ranking, force/PES certification, or accuracy result.",
        "preregistration": {
            "path": PREREGISTRATION_RELATIVE_PATH,
            "sha256": _sha256(preregistration_path),
            "protocol_id": preregistration["protocol_id"],
        },
        "geometry": {
            "path": GEOMETRY_RELATIVE_PATH,
            "sha256": _sha256(geometry_path),
            "geometry_id": geometry["geometry_id"],
            "elements": geometry["elements"],
            "positions_angstrom": geometry["positions_angstrom"],
        },
        "qm_method": QM_METHOD,
        "pcm_control": PCM_CONTROL,
        "numerical_gates": NUMERICAL_GATES,
        "gas_state": gas_state,
        "solvated_state": solvated_state,
        "auxiliary_difference": {
            "energy_hartree": auxiliary_energy,
            "nuclear_gradient_hartree_per_bohr": auxiliary_gradient.tolist(),
            "ledger": "A_aux^(PCM-SCF)-A_aux^(gas-SCF); no isolated PCM component enters this difference.",
        },
        "mace_composite": {
            "not_evaluated": True,
            "required_future_ledger": "E_MACE,gas+[A_aux^(PCM-SCF)-A_aux^(gas-SCF)]",
            "forbidden": [
                "MACE field response or auxiliary-density feature feedback",
                "an additional PCM half-coupling",
                "SMD CDS or another empirical non-electrostatic term",
                "a standard-state or total-solvation claim from this control",
            ],
        },
        "pcm_surface_grid_point_count": int(surface_points.shape[0]),
        "provenance": {
            "git_head": head,
            "source_sha256": {
                RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
            },
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_resolved_sha256": _sha256(Path(sys.executable).resolve()),
            "pyscf_init_path": str(Path(pyscf.__file__).resolve()),
            "pyscf_init_sha256": _sha256(Path(pyscf.__file__).resolve()),
            "pyscf_version": pyscf.__version__,
            "numpy_version": np.__version__,
            "threads": lib.num_threads(),
            "total_elapsed_seconds": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    _write_exclusive_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
