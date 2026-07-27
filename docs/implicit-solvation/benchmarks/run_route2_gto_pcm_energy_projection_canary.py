#!/usr/bin/env python3
"""Run the fixed-geometry acetone GTO/PCM energy-norm projection canary.

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

ARTIFACT_ID = "route2-gto-pcm-energy-projection-acetone-v1"
KCAL_PER_HARTREE = 627.5094740631
ONE_RADIAL_WIDTHS = (1.5,)
TWO_RADIAL_WIDTHS = (1.5, 3.0)
HELPER_RELATIVE_PATH = "docs/implicit-solvation/benchmarks/route2_qm_surface_mep.py"
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_gto_pcm_energy_projection_canary.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    HELPER_RELATIVE_PATH,
    ("maple/function/calculator/extra_correction/implicit/" "continuum_response.py"),
    ("maple/function/calculator/extra_correction/implicit/" "electrostatic_pairing.py"),
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/gto_galerkin.py",
    "maple/function/calculator/extra_correction/implicit/pcm_energy_projection.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/function/read/filereader/mol2_reader.py",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyscf-python", type=Path, required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
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
    )
    elapsed = time.perf_counter() - started
    return {
        "widths_angstrom": list(widths_angstrom),
        "coefficient_count": operator.coefficient_count,
        "surface_point_count": int(len(potential_hartree_per_e)),
        "elapsed_seconds": elapsed,
        "operator": operator.provenance,
        "constraint_residual_inf": projection.constraint_residual_inf,
        "tangent_optimality_inf": projection.tangent_optimality_inf,
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
    git_head = _require_clean_source()
    args.pyscf_python = args.pyscf_python.absolute()
    input_names = (
        "pcmsolver_library",
        "mol2",
        "parsed_pcm_input",
        "qm_checkpoint",
    )
    for name in input_names:
        setattr(args, name, getattr(args, name).resolve())
    for path in (getattr(args, name) for name in input_names):
        if not path.is_file():
            raise FileNotFoundError(path)
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
        raise RuntimeError(
            "The acetone QM reference did not recover a neutral density."
        )

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
        )
        two_radial = _projection_record(
            response,
            positions_angstrom,
            potential,
            total_charge_e=total_charge,
            molecular_dipole_e_angstrom=dipole,
            widths_angstrom=TWO_RADIAL_WIDTHS,
        )

    one_error = abs(float(one_radial["polarization_energy_error_kcal_per_mol"]))
    two_error = abs(float(two_radial["polarization_energy_error_kcal_per_mol"]))
    if two_error >= one_error:
        raise RuntimeError(
            "The two-radial GTO basis did not improve acetone's PCM "
            "polarization-energy projection error."
        )

    payload = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "fixed-geometry representation feasibility canary; "
            "not a variational MACE-POLAR model"
        ),
        "claim_boundary": {
            "certifies": [
                "same-basis finite-width l<=1 GTO source/receiver reciprocity",
                "fixed-cavity PCM half-coupling identity",
                "constrained PCM-energy-norm projection algebra",
                "acetone two-radial representation improvement on one QM MEP",
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
            "pyscf_python": {
                "invoked_path": str(args.pyscf_python),
                "resolved_target": str(args.pyscf_python.resolve()),
            },
            "pcmsolver_library": {
                "path": str(args.pcmsolver_library.resolve()),
                "sha256": _sha256(args.pcmsolver_library),
            },
        },
        "inputs": {
            "compound_id": "mobley_3867265",
            "molecule": {
                "path": str(args.mol2.resolve()),
                "sha256": _sha256(args.mol2),
            },
            "parsed_pcm_input": {
                "path": str(args.parsed_pcm_input.resolve()),
                "sha256": _sha256(args.parsed_pcm_input),
                "cavity_radii_angstrom": radii_angstrom.tolist(),
            },
            "qm_checkpoint": {
                "path": str(args.qm_checkpoint.resolve()),
                "sha256": _sha256(args.qm_checkpoint),
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
            "algebraic_projection_gates_passed": True,
            "two_radial_improves_energy_projection": True,
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
