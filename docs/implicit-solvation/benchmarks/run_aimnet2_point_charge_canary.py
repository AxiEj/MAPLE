#!/usr/bin/env python3
"""Run the bounded AIMNet2 point-charge/ddPCM interface canary.

This is not a hydration-free-energy benchmark.  It verifies only that one
checkpoint's NQE charges conserve molecular charge, enter the continuum as a
fixed ``point-charge-l0`` source, and satisfy the reciprocal half-coupling
identity for four small ASE reference geometries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

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


MOLECULES = ("H2O", "NH3", "CH4", "CH3COCH3")
CHARGE_REPEATS = 3
WATER_DIELECTRIC = 78.39
DDPCM_LMAX = 7
DDPCM_N_LEBEDEV = 302
DDPCM_SOLVER_TOLERANCE = 1.0e-10
DDPCM_ETA = 0.1
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
    digest.update(
        np.asarray(atoms.get_positions(), dtype=np.float64).tobytes()
    )
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
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
        "docs/implicit-solvation/benchmarks/run_aimnet2_point_charge_canary.py",
    )
    return {
        relative: _sha256(root / relative)
        for relative in relative_paths
    }


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

    torch.set_num_threads(1)
    load_started = time.perf_counter()
    calculator = AIMNet2Calculator(
        torch.device("cpu"),
        model="aimnet2",
        model_path=str(checkpoint),
    )
    model_load_seconds = time.perf_counter() - load_started

    records = []
    for name in MOLECULES:
        atoms = molecule(name)
        atoms.info.update(charge=0, mult=1)
        charge_times = []
        charge_state = None
        for _ in range(CHARGE_REPEATS):
            started = time.perf_counter()
            charge_state = calculator.charge_state(atoms)
            charge_times.append(time.perf_counter() - started)
        assert charge_state is not None

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
        started = time.perf_counter()
        reaction_field = PyDDXPCMReactionFieldLinearMap(
            atoms.get_positions(),
            radii,
            dielectric=WATER_DIELECTRIC,
            lmax=DDPCM_LMAX,
            n_lebedev=DDPCM_N_LEBEDEV,
            n_proc=1,
            solver_tolerance=DDPCM_SOLVER_TOLERANCE,
            eta=DDPCM_ETA,
        )
        continuum_build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        continuum_state = solve_fixed_charge_continuum(
            reaction_field,
            source,
            energy_identity_tolerance_ev=1.0e-8,
        )
        continuum_solve_seconds = time.perf_counter() - started

        records.append(
            {
                "molecule": name,
                "formula": atoms.get_chemical_formula(),
                "atom_count": len(atoms),
                "geometry_source": "ASE G2 molecule collection",
                "geometry_sha256": _geometry_sha256(atoms),
                "energy_ev": charge_state.energy_ev,
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
                "timing_seconds": {
                    "aimnet2_charge_samples": charge_times,
                    "aimnet2_charge_median": float(
                        np.median(charge_times)
                    ),
                    "ddpcm_build": continuum_build_seconds,
                    "ddpcm_solve": continuum_solve_seconds,
                },
            }
        )

    artifact = {
        "artifact": "route2-aimnet2-point-charge-ddpcm-canary-v1",
        "scientific_identity": {
            "solute_energy_model": "AIMNet2 gas-phase wB97M-D3 checkpoint",
            "solute_source": "AIMNet2 NQE point-charge-l0",
            "polarization_response": "fixed",
            "continuum": "pyddx ddPCM",
            "cds_included": False,
            "total_solvation_free_energy_reported": False,
            "mutual_ml_continuum_polarization": False,
        },
        "claim_boundary": (
            "This four-molecule canary validates AIMNet2 charge export, "
            "float-residue-only charge projection, fixed point-charge-l0 "
            "continuum ingestion, and the discrete half-coupling identity. "
            "It does not validate SMD-CDS, absolute solvation free energies, "
            "MNSol/FreeSolv accuracy, forces, a self-consistent AIMNet2-PCM "
            "response, or a solution-phase PES."
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
            "radii_selector_profile": DDPCM_SMD_PROFILE,
            "full_profile_numerical_equivalence": False,
            "reduced_canary_discretization": True,
            "lmax": DDPCM_LMAX,
            "n_lebedev": DDPCM_N_LEBEDEV,
            "n_proc": 1,
            "solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "eta": DDPCM_ETA,
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
