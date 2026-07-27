#!/usr/bin/env python3
"""Generate one audited intrinsic-cavity PCMSolver machine input.

The runner reuses MAPLE's production cavity/radius implementation but does not
load an ML model or evaluate an energy.  It exists so post-preregistration
continuum inputs can be derived by one source-bound, fail-closed procedure.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.route2_pcmsolver_cavity import (  # noqa: E402
    PCM_INTRINSIC_MIN_RADIUS_ANGSTROM,
    PCM_INTRINSIC_TESSERA_AREA_ANGSTROM2,
    PCM_PARSER_SURROGATE_PROBE_RADIUS_ANGSTROM,
    _intrinsic_pcm_parser_surrogate,
    _patch_intrinsic_pcm_machine_input,
    _pcm_input_text,
    _validate_intrinsic_pcm_machine_input,
)
from maple.function.calculator.extra_correction.implicit.smd import (  # noqa: E402
    _load_pcmsolver_parser,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (  # noqa: E402
    route2_water_coulomb_radii,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
from maple.function.route2_smd_profiles import (  # noqa: E402
    PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,
    route2_smd_profile_spec,
)

RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_intrinsic_pcm_input.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    "maple/function/calculator/extra_correction/implicit/route2_pcmsolver_cavity.py",
    "maple/function/calculator/extra_correction/implicit/smd.py",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/function/read/filereader/mol2_reader.py",
    "maple/function/route2_smd_profiles.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol2", type=Path, required=True)
    parser.add_argument("--compound-id", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
    parser.add_argument("--pcmsolver-python-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        default=PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,
        choices=(PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,),
    )
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
            "The intrinsic PCM-input runner requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> int:
    arguments = _parse_args()
    source_head = _require_clean_source()
    mol2_path = arguments.mol2.resolve()
    library_path = arguments.pcmsolver_library.resolve()
    parser_root = arguments.pcmsolver_python_path.resolve()
    for path in (mol2_path, library_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (parser_root / "pcmsolver").is_dir():
        raise FileNotFoundError(
            f"PCMSolver Python package directory not found under {parser_root}."
        )
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    os.environ["PCMSOLVER_LIBRARY"] = str(library_path)
    os.environ["PCMSOLVER_PYTHON_PATH"] = str(parser_root)
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    mol2_metadata = atoms.info.get("mol2")
    atom_types = (
        mol2_metadata.get("atom_types") if isinstance(mol2_metadata, dict) else None
    )
    profile = route2_smd_profile_spec(arguments.profile)
    radii_angstrom = route2_water_coulomb_radii(
        atoms.get_chemical_symbols(),
        atom_types=atom_types,
        profile=arguments.profile,
    )

    raw_path = output_dir / "route2-smd-intrinsic.pcm"
    surrogate_path = output_dir / "route2-smd-intrinsic.parser-surrogate.pcm"
    parser_output_path = output_dir / f"@{surrogate_path.name}"
    effective_path = output_dir / f"@{raw_path.name}"
    ledger_path = output_dir / "pcm-input.json"

    raw_text = _pcm_input_text(
        len(atoms),
        radii_angstrom,
        tessera_area_angstrom2=PCM_INTRINSIC_TESSERA_AREA_ANGSTROM2,
        minimum_added_sphere_radius_angstrom=PCM_INTRINSIC_MIN_RADIUS_ANGSTROM,
        dielectric_policy=profile.dielectric_policy,
    )
    raw_path.write_text(raw_text, encoding="utf-8")
    surrogate_path.write_text(
        _intrinsic_pcm_parser_surrogate(raw_text),
        encoding="utf-8",
    )

    parser = _load_pcmsolver_parser()
    parser_module = importlib.import_module(parser.__module__)
    parser_file = Path(getattr(parser_module, "__file__", "")).resolve()
    if not parser_file.is_file():
        raise RuntimeError("The PCMSolver parser module has no auditable file.")
    with _working_directory(output_dir):
        try:
            parser(str(surrogate_path), write_out=True)
        except (Exception, SystemExit) as exc:
            raise RuntimeError(
                f"PCMSolver failed to parse {surrogate_path}: {exc}"
            ) from exc
    if not parser_output_path.is_file():
        raise RuntimeError(
            "PCMSolver did not create the expected parser machine input."
        )
    patched_text, adapter_semantics = _patch_intrinsic_pcm_machine_input(
        parser_output_path.read_text(encoding="utf-8")
    )
    effective_path.write_text(patched_text, encoding="utf-8")
    effective_semantics = _validate_intrinsic_pcm_machine_input(patched_text)

    record = {
        "schema_version": 1,
        "artifact_id": f"route2-intrinsic-pcm-input-{arguments.compound_id}-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "source": {
            "git_head": source_head,
            "runner_path": RUNNER_RELATIVE_PATH,
            "source_sha256": {
                relative: _sha256(REPO_ROOT / relative)
                for relative in SOURCE_RELATIVE_PATHS
            },
            "argv": [sys.executable, *sys.argv],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pcmsolver_library": {
                "path": str(library_path),
                "sha256": _sha256(library_path),
            },
            "pcmsolver_parser": {
                "path": str(parser_file),
                "sha256": _sha256(parser_file),
            },
        },
        "system": {
            "compound_id": arguments.compound_id,
            "molecule": arguments.molecule_name,
            "charge": 0,
            "multiplicity": 1,
            "atom_count": len(atoms),
            "symbols": atoms.get_chemical_symbols(),
            "positions_angstrom": np.asarray(
                atoms.get_positions(), dtype=float
            ).tolist(),
        },
        "profile": {
            "name": profile.name,
            "provider": profile.provider,
            "electrostatics_model": profile.electrostatics_model,
            "dielectric_policy": profile.dielectric_policy,
            "coulomb_radii_policy": profile.coulomb_radii_policy,
            "pcmsolver_cavity_generation": profile.pcmsolver_cavity_generation,
            "tessera_area_angstrom2": PCM_INTRINSIC_TESSERA_AREA_ANGSTROM2,
            "minimum_added_sphere_radius_angstrom": (PCM_INTRINSIC_MIN_RADIUS_ANGSTROM),
            "parser_surrogate_probe_radius_angstrom": (
                PCM_PARSER_SURROGATE_PROBE_RADIUS_ANGSTROM
            ),
            "effective_semantics": effective_semantics,
            "adapter_semantics": adapter_semantics,
            "radii_angstrom": radii_angstrom.tolist(),
        },
        "input": {
            "mol2_path": str(mol2_path),
            "mol2_sha256": _sha256(mol2_path),
        },
        "outputs": {
            "raw_input": {
                "path": str(raw_path),
                "sha256": _sha256(raw_path),
            },
            "parser_surrogate_input": {
                "path": str(surrogate_path),
                "sha256": _sha256(surrogate_path),
            },
            "parser_machine_output": {
                "path": str(parser_output_path),
                "sha256": _sha256(parser_output_path),
            },
            "effective_machine_input": {
                "path": str(effective_path),
                "sha256": _sha256(effective_path),
            },
        },
        "claim_boundary": (
            "One neutral fixed-geometry intrinsic-cavity PCMSolver machine "
            "input. No ML model, energy, density response, force, or "
            "experimental accuracy is evaluated."
        ),
    }
    with ledger_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
