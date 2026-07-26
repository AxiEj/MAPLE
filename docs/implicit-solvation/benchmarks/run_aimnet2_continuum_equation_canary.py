#!/usr/bin/env python3
"""Compare ddPCM and scaled ddCOSMO on identical fixed AIMNet2 sources.

This bounded canary changes only the pyddx continuum equation.  It is not a
solvation-free-energy benchmark: the AIMNet2 charges are fixed, SMD-CDS is
absent, and no experimental accuracy claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import ase
from ase.build import molecule
import numpy as np
import pyddx
import torch

from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNET2_PADDED_SENTINEL_TOLERANCE_E,
    AIMNET2_RAW_CHARGE_TOLERANCE_E,
    AIMNet2Calculator,
)
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXCOSMOReactionFieldLinearMap,
    PyDDXPCMReactionFieldLinearMap,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    route2_coulomb_radii,
)
from maple.function.calculator.extra_correction.implicit.source import (
    PointChargeL0Source,
    solve_fixed_charge_continuum,
)
from maple.function.route2_smd_profiles import DDPCM_SMD_PROFILE


ARTIFACT_NAME = "route2-aimnet2-ddpcm-ddcosmo-equation-canary-v1"
MOLECULES = ("H2O", "CH3COCH3")
WATER_DIELECTRIC = 78.39
DDX_LMAX = 7
DDX_N_LEBEDEV = 302
DDX_SOLVER_TOLERANCE = 1.0e-10
DDX_ETA = 0.1
HARTREE_TO_KCAL_MOL = 627.5094740631


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_sha256(atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
    digest.update(np.asarray(atoms.get_positions(), dtype=np.float64).tobytes())
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    relative_paths = (
        "maple/function/calculator/aimnet/_aimnet2_calculator.py",
        (
            "maple/function/calculator/extra_correction/implicit/source/"
            "point_charge_l0.py"
        ),
        (
            "maple/function/calculator/extra_correction/implicit/"
            "pyddx_pcm_response.py"
        ),
        (
            "maple/function/calculator/extra_correction/implicit/"
            "smd_cds.py"
        ),
        "maple/function/route2_smd_profiles.py",
        "maple/function/route2_solvents.py",
        (
            "docs/implicit-solvation/benchmarks/"
            "run_aimnet2_continuum_equation_canary.py"
        ),
    )
    return {
        relative: _sha256(REPO_ROOT / relative)
        for relative in relative_paths
    }


def _execution_git_head() -> str:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "The continuum-equation canary must run from a clean Git checkout."
        )
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("Could not resolve a full execution Git commit.")
    return head


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local AIMNet2 TorchScript checkpoint; it is never copied.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    execution_git_head = _execution_git_head()

    torch.set_num_threads(1)
    load_started = time.perf_counter()
    calculator = AIMNet2Calculator(
        torch.device("cpu"),
        model="aimnet2",
        model_path=str(checkpoint),
    )
    model_load_seconds = time.perf_counter() - load_started

    method_classes = (
        ("ddpcm", PyDDXPCMReactionFieldLinearMap),
        ("ddcosmo", PyDDXCOSMOReactionFieldLinearMap),
    )
    records = []
    for name in MOLECULES:
        atoms = molecule(name)
        atoms.info.update(charge=0, mult=1)
        charge_started = time.perf_counter()
        charge_state = calculator.charge_state(atoms)
        charge_seconds = time.perf_counter() - charge_started
        source = PointChargeL0Source(
            charges_e=charge_state.charges_e,
            declared_total_charge_e=0.0,
            source_model="aimnet2-nqe",
        )
        radii = route2_coulomb_radii(
            atoms.get_chemical_symbols(),
            solvent="water",
            profile=DDPCM_SMD_PROFILE,
        )

        method_records = {}
        for method, reaction_class in method_classes:
            build_started = time.perf_counter()
            reaction_field = reaction_class(
                atoms.get_positions(),
                radii,
                dielectric=WATER_DIELECTRIC,
                lmax=DDX_LMAX,
                n_lebedev=DDX_N_LEBEDEV,
                n_proc=1,
                solver_tolerance=DDX_SOLVER_TOLERANCE,
                eta=DDX_ETA,
            )
            build_seconds = time.perf_counter() - build_started
            solve_started = time.perf_counter()
            continuum_state = solve_fixed_charge_continuum(
                reaction_field,
                source,
                energy_identity_tolerance_ev=1.0e-8,
            )
            solve_seconds = time.perf_counter() - solve_started
            method_records[method] = {
                "polarization_energy_hartree": (
                    continuum_state.polarization_energy_hartree
                ),
                "polarization_energy_kcal_mol": (
                    continuum_state.polarization_energy_hartree
                    * HARTREE_TO_KCAL_MOL
                ),
                "half_coupling_identity_error_ev": (
                    continuum_state.energy_identity_error_ev
                ),
                "runtime_provenance": reaction_field.runtime_provenance,
                "timing_seconds": {
                    "build": build_seconds,
                    "solve": solve_seconds,
                },
            }

        pcm_energy = method_records["ddpcm"][
            "polarization_energy_kcal_mol"
        ]
        cosmo_energy = method_records["ddcosmo"][
            "polarization_energy_kcal_mol"
        ]
        records.append(
            {
                "molecule": name,
                "formula": atoms.get_chemical_formula(),
                "atom_count": len(atoms),
                "geometry_source": "ASE G2 molecule collection",
                "geometry_sha256": _geometry_sha256(atoms),
                "aimnet2_energy_ev": charge_state.energy_ev,
                "raw_charge_sum_e": charge_state.raw_charge_sum_e,
                "raw_charge_residual_e": (
                    charge_state.raw_charge_residual_e
                ),
                "charge_projection_per_atom_e": (
                    charge_state.charge_projection_per_atom_e
                ),
                "projected_charge_sum_e": (
                    charge_state.projected_charge_sum_e
                ),
                "charges_e": charge_state.charges_e.tolist(),
                "cavity_radii_angstrom": np.asarray(
                    radii,
                    dtype=float,
                ).tolist(),
                "methods": method_records,
                "ddcosmo_minus_ddpcm_kcal_mol": (
                    cosmo_energy - pcm_energy
                ),
                "timing_seconds": {
                    "aimnet2_charge": charge_seconds,
                },
            }
        )

    artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "execution_git_head": execution_git_head,
        "scientific_identity": {
            "solute_energy_model": "AIMNet2 gas-phase wB97M-D3 checkpoint",
            "solute_source": "AIMNet2 NQE point-charge-l0",
            "polarization_response": "fixed",
            "continuum_equations": ["pyddx ddPCM", "pyddx ddCOSMO"],
            "orthogonal_comparison": (
                "same geometry/source/radii/dielectric/grid; continuum "
                "equation only"
            ),
            "cds_included": False,
            "total_solvation_free_energy_reported": False,
            "mutual_ml_continuum_polarization": False,
            "cosmo_rs_included": False,
        },
        "claim_boundary": (
            "This two-molecule canary validates the orthogonal ddPCM/ddCOSMO "
            "equation axis, pyddx 0.8.0 host-side finite-dielectric COSMO "
            "scaling, and each method's discrete half-coupling identity. It "
            "does not validate SMD-CDS, experimental solvation accuracy, "
            "C-PCM, COSMO-RS, forces, self-consistent AIMNet2 response, or a "
            "solution-phase PES."
        ),
        "checkpoint": {
            "filename": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            "redistributed": False,
        },
        "charge_output_parameters": {
            "raw_total_charge_tolerance_e": (
                AIMNET2_RAW_CHARGE_TOLERANCE_E
            ),
            "padded_sentinel_tolerance_e": (
                AIMNET2_PADDED_SENTINEL_TOLERANCE_E
            ),
            "projection": "uniform-affine-float-residue-only",
        },
        "continuum_parameters": {
            "solvent_context": "water",
            "dielectric": WATER_DIELECTRIC,
            "cosmo_dielectric_scaling": (
                (WATER_DIELECTRIC - 1.0) / WATER_DIELECTRIC
            ),
            "cosmo_dielectric_scaling_owner": (
                "MAPLE host because pyddx 0.8.0 returns unscaled COSMO "
                "energy and derivatives"
            ),
            "radii_selector_profile": DDPCM_SMD_PROFILE,
            "full_profile_numerical_equivalence": False,
            "reduced_canary_discretization": True,
            "lmax": DDX_LMAX,
            "n_lebedev": DDX_N_LEBEDEV,
            "n_proc": 1,
            "solver_tolerance": DDX_SOLVER_TOLERANCE,
            "eta": DDX_ETA,
        },
        "references": {
            "ddx_documentation": "https://ddsolvation.github.io/ddX/",
            "ddx_source_tag": "v0.8.0",
            "ddx_cosmo_scaling_source": (
                "pyddx.State.energy documentation and src/ddx.h at v0.8.0"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "pyddx": pyddx.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "model_load_seconds": model_load_seconds,
        },
        "source_files_sha256": _source_hashes(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
