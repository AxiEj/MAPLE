#!/usr/bin/env python3
"""Generate one fixed-geometry PySCF gas checkpoint for Route-2 references.

This runner is intentionally isolated from MAPLE's runtime dependencies.  It
creates only a gas-phase RKS checkpoint and an immutable JSON ledger.  The
checkpoint is later consumed directly by ``route2_qm_surface_mep.py``; no
separate AO density file is accepted or produced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import pyscf
from pyscf import dft, gto, lib

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_qm_gas_checkpoint.py"
)
OFFICIAL_NLC_EXAMPLE_URL = (
    "https://raw.githubusercontent.com/pyscf/pyscf/v2.13.1/"
    "examples/dft/15-nlc_functionals.py"
)
OFFICIAL_NLC_EXAMPLE_SHA256 = (
    "299f326e8f51a6522b8453146f9d7bf3975e88c0db5c097afe4ac5611e255427"
)
_GAFF_ELEMENT_PREFIXES = {
    "br": "Br",
    "cl": "Cl",
    "c": "C",
    "f": "F",
    "h": "H",
    "i": "I",
    "n": "N",
    "o": "O",
    "p": "P",
    "s": "S",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol2", type=Path, required=True)
    parser.add_argument("--compound-id", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basis", default="def2-tzvpd")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument(
        "--nlc-grid-profile",
        choices=("level", "pyscf-official-50x194-sg1"),
        default="pyscf-official-50x194-sg1",
    )
    parser.add_argument("--nlc-grid-level", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=8000)
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
            "The QM checkpoint runner requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    _git("ls-files", "--error-unmatch", RUNNER_RELATIVE_PATH)
    return _git("rev-parse", "HEAD")


def _element_from_gaff_type(atom_type: str) -> str:
    normalized = atom_type.split(".", 1)[0].lower()
    for prefix in ("br", "cl"):
        if normalized.startswith(prefix):
            return _GAFF_ELEMENT_PREFIXES[prefix]
    if not normalized:
        raise ValueError("MOL2 atom type cannot be empty.")
    try:
        return _GAFF_ELEMENT_PREFIXES[normalized[0]]
    except KeyError as exc:
        raise ValueError(f"Unsupported GAFF atom type: {atom_type}") from exc


def _element_from_atom_name(atom_name: str) -> str:
    match = re.match(r"[A-Za-z]+", atom_name)
    if match is None:
        raise ValueError(f"MOL2 atom name has no element prefix: {atom_name}")
    letters = match.group(0)
    if len(letters) >= 2 and letters[:2].lower() in {"br", "cl"}:
        return letters[:2].title()
    return letters[0].upper()


def _mol2_geometry(path: Path) -> tuple[str, list[dict[str, object]]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        start = lines.index("@<TRIPOS>ATOM") + 1
        stop = lines.index("@<TRIPOS>BOND")
    except ValueError as exc:
        raise ValueError("MOL2 input must contain ATOM and BOND sections.") from exc
    atoms: list[dict[str, object]] = []
    geometry: list[str] = []
    for line in lines[start:stop]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 6:
            raise ValueError(f"Malformed MOL2 atom row: {line}")
        symbol_from_name = _element_from_atom_name(fields[1])
        symbol_from_type = _element_from_gaff_type(fields[5])
        if symbol_from_name != symbol_from_type:
            raise ValueError(
                "MOL2 atom-name/type element mismatch: "
                f"{fields[1]} versus {fields[5]}."
            )
        coordinates = [float(fields[index]) for index in (2, 3, 4)]
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("MOL2 coordinates must be finite.")
        geometry.append(
            f"{symbol_from_type} "
            f"{coordinates[0]:.10f} {coordinates[1]:.10f} "
            f"{coordinates[2]:.10f}"
        )
        atoms.append(
            {
                "symbol": symbol_from_type,
                "position_angstrom": coordinates,
            }
        )
    if not atoms:
        raise ValueError("MOL2 input contains no atoms.")
    return "\n".join(geometry), atoms


def _configure_rks(molecule, arguments: argparse.Namespace):
    mean_field = dft.RKS(molecule, xc="wb97m-v").density_fit()
    mean_field.grids.level = arguments.grid_level
    if arguments.nlc_grid_profile == "pyscf-official-50x194-sg1":
        symbols = sorted(
            {molecule.atom_symbol(index) for index in range(molecule.natm)}
        )
        mean_field.nlcgrids.atom_grid = {symbol: (50, 194) for symbol in symbols}
        mean_field.nlcgrids.prune = dft.gen_grid.sg1_prune
    else:
        mean_field.nlcgrids.level = arguments.nlc_grid_level
    mean_field.conv_tol = 1.0e-10
    mean_field.conv_tol_grad = 1.0e-7
    mean_field.max_cycle = 100
    mean_field.max_memory = arguments.max_memory_mb
    return mean_field


def main() -> int:
    arguments = _parse_args()
    source_head = _require_clean_source()
    mol2_path = arguments.mol2.resolve()
    if not mol2_path.is_file():
        raise FileNotFoundError(mol2_path)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if arguments.threads < 1 or arguments.max_memory_mb < 1:
        raise ValueError("Threads and memory must be positive.")
    geometry, atoms = _mol2_geometry(mol2_path)

    lib.num_threads(arguments.threads)
    molecule = gto.M(
        atom=geometry,
        basis=arguments.basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry=False,
        verbose=4,
        max_memory=arguments.max_memory_mb,
    )
    checkpoint = output_dir / "gas.chk"
    result_path = output_dir / "gas.json"
    cycle_count = 0

    def count_cycle(_environment):
        nonlocal cycle_count
        cycle_count += 1

    mean_field = _configure_rks(molecule, arguments)
    mean_field.callback = count_cycle
    mean_field.chkfile = str(checkpoint)
    started = time.perf_counter()
    energy_hartree = float(mean_field.kernel())
    elapsed = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError("Gas-phase PySCF reference did not converge.")

    direct_density = np.asarray(mean_field.make_rdm1(), dtype=float)
    checkpoint_density = (
        np.asarray(mean_field.mo_coeff, dtype=float)
        * np.asarray(mean_field.mo_occ, dtype=float)[None, :]
    ) @ np.asarray(mean_field.mo_coeff, dtype=float).T
    checkpoint_density = 0.5 * (checkpoint_density + checkpoint_density.T)
    checkpoint_binding_residual = float(
        np.max(np.abs(direct_density - checkpoint_density))
    )
    if checkpoint_binding_residual > 1.0e-10:
        raise RuntimeError(
            "The saved checkpoint orbitals do not reconstruct the converged "
            "closed-shell AO density."
        )

    record = {
        "schema_version": 1,
        "artifact_id": (f"route2-qm-gas-checkpoint-{arguments.compound_id}-v1"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "source": {
            "git_head": source_head,
            "runner_path": RUNNER_RELATIVE_PATH,
            "runner_sha256": _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH),
            "source_sha256": {
                RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH),
            },
            "argv": [sys.executable, *sys.argv],
            "effective_arguments": {
                "basis": arguments.basis,
                "grid_level": arguments.grid_level,
                "nlc_grid_profile": arguments.nlc_grid_profile,
                "nlc_grid_level": arguments.nlc_grid_level,
                "threads": arguments.threads,
                "max_memory_mb": arguments.max_memory_mb,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": {
                "invoked": sys.executable,
                "resolved": str(Path(sys.executable).resolve()),
                "resolved_sha256": _sha256(Path(sys.executable).resolve()),
            },
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "threads": lib.num_threads(),
        },
        "system": {
            "compound_id": arguments.compound_id,
            "molecule": arguments.molecule_name,
            "charge": 0,
            "multiplicity": 1,
            "atom_count": molecule.natm,
            "method": "omegaB97M-V",
            "basis": arguments.basis,
            "reference": "RKS",
            "density_fitting": True,
            "geometry": atoms,
        },
        "numerics": {
            "semilocal_grid_level": arguments.grid_level,
            "nonlocal_grid_profile": arguments.nlc_grid_profile,
            "nonlocal_grid_level": (
                arguments.nlc_grid_level
                if arguments.nlc_grid_profile == "level"
                else None
            ),
            "nonlocal_atom_grid": (
                [50, 194]
                if arguments.nlc_grid_profile == "pyscf-official-50x194-sg1"
                else None
            ),
            "nonlocal_prune": (
                "sg1_prune"
                if arguments.nlc_grid_profile == "pyscf-official-50x194-sg1"
                else "PySCF level default"
            ),
            "scf_energy_tolerance_hartree": 1.0e-10,
            "scf_gradient_tolerance": 1.0e-7,
            "maximum_scf_cycles": 100,
            "maximum_memory_mb": arguments.max_memory_mb,
        },
        "input": {
            "mol2_path": str(mol2_path),
            "mol2_sha256": _sha256(mol2_path),
        },
        "result": {
            "energy_hartree": energy_hartree,
            "scf_cycles": cycle_count,
            "elapsed_seconds": elapsed,
            "semilocal_grid_point_count": int(mean_field.grids.coords.shape[0]),
            "nonlocal_grid_point_count": int(mean_field.nlcgrids.coords.shape[0]),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
            },
            "checkpoint_density_binding_residual_inf": (checkpoint_binding_residual),
        },
        "official_documentation": {
            "solvent": "https://pyscf.org/user/solvent.html",
            "nlc_example": {
                "url": OFFICIAL_NLC_EXAMPLE_URL,
                "sha256": OFFICIAL_NLC_EXAMPLE_SHA256,
            },
        },
        "claim_boundary": (
            "One neutral closed-shell fixed-geometry gas-phase electronic "
            "reference. It is not a solution free energy, continuum result, "
            "conformer ensemble, or production MAPLE dependency."
        ),
    }
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
