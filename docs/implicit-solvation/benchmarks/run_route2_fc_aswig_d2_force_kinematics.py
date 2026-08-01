#!/usr/bin/env python3
"""Kinematic Route-2 force canary for the D2-canonical FC-aSWIG profile.

This runner evaluates the *declared direct-PCM scalar*

    0.5 <c_MACE-POLAR, f_reac_CPCM> + G_fixed-topology-SMD-CDS

for one frozen acetone conformer.  It performs three independent local
checks of its analytic fixed-point-adjoint correction force:

* three component-resolved central finite differences;
* a rigid SO(3) rotation covariance check;
* a rigid translation check.

The MACE-POLAR checkpoint is evaluated through the explicit, versioned
``jgp94-d2-canonical-v1`` operator.  This is required because the upstream
real-space electrostatic-feature block uses finite differences along laboratory
axes.  The D2 Reynolds average is a no-training symmetry repair, not a change
to the checkpoint weights or to any solvent parameter.

A passing artifact is deliberately *not* public force admission.  It leaves
coordinate-path, closed-loop, and NVE gates open, and it does not establish a
common stationary MACE--PCM electronic functional.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from ase.units import Hartree

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.fc_aswig_smd import (
    FixedTopologyASWIGAqueousSMDImplicitSolvation,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.route2_smd_profiles import (
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
)

SCHEMA_VERSION = 1
PROTOCOL_ID = "route2-fc-aswig-d2-mace-acetone-kinematic-force-v1"
COMPOUND_ID = "mobley_3867265"
COMPONENTS = ((0, 0), (1, 1), (2, 2))
STEPS_ANGSTROM = (1.0e-3, 5.0e-4, 2.5e-4)
ROTATION = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
TRANSLATION_ANGSTROM = np.asarray((0.37, -0.21, 0.18))
GATES = {
    "maximum_fd_error_ev_per_angstrom": 2.0e-5,
    "maximum_rotation_energy_error_ev": 1.0e-8,
    "maximum_rotation_force_error_ev_per_angstrom": 1.0e-6,
    "maximum_translation_energy_error_ev": 1.0e-8,
    "maximum_translation_force_error_ev_per_angstrom": 1.0e-6,
    "maximum_net_force_hartree_per_angstrom": 1.0e-10,
}
DEFAULT_PREPARED = (
    REPO_ROOT / ".omx" / "benchmarks" / "route2-macepolar-smd" / "prepared.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".omx"
    / "benchmarks"
    / "route2-fc-aswig-d2-mace-acetone-kinematic-force"
    / "result.json"
)


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_status() -> list[str] | None:
    """Return tracked/untracked source state so a result never masks a dirty tree."""

    try:
        output = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=REPO_ROOT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in output.splitlines() if line]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _settings() -> dict[str, Any]:
    settings = CommandControl.from_settings(
        (
            "#model=macepol-m",
            "#sp",
            (
                "#solv(implicit=water,method=smd,provider=fc-aswig,"
                f"profile={FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE},"
                "response=scf,standard_state=1m,experimental=true)"
            ),
        )
    ).as_dict()
    if settings["solv"]["profile"] != (
        FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE
    ):
        raise RuntimeError("The D2-canonical Route-2 profile parser drifted.")
    return settings


def _load_atoms(
    prepared_path: Path,
    *,
    compound_id: str = COMPOUND_ID,
):
    """Load one hash-pinned neutral FreeSolv geometry from a prepared panel.

    The acetone canaries retain their historical default, while the flexible
    torsion canary can reuse the identical loader without copying provenance
    or MOL2-integrity logic into another runner.
    """

    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    candidates = prepared.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("Prepared FreeSolv input has no candidates list.")
    candidate = next(
        (row for row in candidates if row.get("compound_id") == compound_id),
        None,
    )
    if candidate is None:
        raise KeyError(f"Prepared input lacks {compound_id}.")
    relative = Path(str(candidate["mol2_relative_path"]))
    mol2 = prepared_path.parent / relative
    if not mol2.is_file():
        raise FileNotFoundError(f"Frozen MOL2 does not exist: {mol2}")
    expected = str(candidate["mol2_sha256"])
    observed = _sha256(mol2)
    if observed != expected:
        raise RuntimeError(f"Frozen MOL2 digest drifted for {COMPOUND_ID}.")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    return atoms, candidate, mol2, observed


def _calculator(atoms, settings: dict[str, Any], device: str):
    return SetCalculator(
        device,
        settings["model"],
        str(REPO_ROOT / ".omx" / "benchmarks" / "route2-fc-aswig-d2-force.out"),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        model_options=settings.get("model_options"),
        solvation_options=settings["solv"],
        charge_options=settings.get("charge") or {},
    ).set_calculator()


def _evaluate(atoms, calculator, solvation_options: dict[str, Any], *, with_force: bool):
    calculator.reset()
    provider = FixedTopologyASWIGAqueousSMDImplicitSolvation(
        atoms,
        solvation_options,
    )
    if with_force:
        result = provider.evaluate_single_point_derivative_evidence(
            atoms,
            calculator=calculator,
        )
    else:
        result = provider.evaluate(atoms, calculator=calculator)
    return result


def _finite_difference_rows(
    base,
    calculator,
    solvation_options: dict[str, Any],
    force: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atom_index, axis in COMPONENTS:
        levels: list[dict[str, Any]] = []
        for step in STEPS_ANGSTROM:
            plus = base.copy()
            minus = base.copy()
            plus.positions[atom_index, axis] += step
            minus.positions[atom_index, axis] -= step
            plus_result = _evaluate(
                plus, calculator, solvation_options, with_force=False
            )
            minus_result = _evaluate(
                minus, calculator, solvation_options, with_force=False
            )
            plus_reason = plus_result.provenance["scf_convergence"].get("reason")
            minus_reason = minus_result.provenance["scf_convergence"].get("reason")
            if plus_reason != "nominal-density-and-energy-v1" or minus_reason != (
                "nominal-density-and-energy-v1"
            ):
                raise RuntimeError(
                    "A finite-difference endpoint did not reach a nominal SCF root."
                )
            plus_energy = plus_result.energy_hartree
            minus_energy = minus_result.energy_hartree
            numeric = -(plus_energy - minus_energy) / (2.0 * step)
            error = abs(numeric - force[atom_index, axis])
            levels.append(
                {
                    "step_angstrom": float(step),
                    "finite_difference_force_hartree_per_angstrom": float(numeric),
                    "absolute_error_hartree_per_angstrom": float(error),
                    "absolute_error_ev_per_angstrom": float(error * Hartree),
                    "plus_scf_reason": str(plus_reason),
                    "minus_scf_reason": str(minus_reason),
                }
            )
        rows.append(
            {
                "atom_index": atom_index,
                "cartesian_index": axis,
                "analytic_force_hartree_per_angstrom": float(force[atom_index, axis]),
                "levels": levels,
            }
        )
    return rows


def _nominal_root(result) -> dict[str, Any]:
    provenance = result.provenance
    convergence = dict(provenance["scf_convergence"])
    agreement = dict(provenance["multi_start_root_agreement"])
    if convergence.get("reason") != "nominal-density-and-energy-v1":
        raise RuntimeError("Force canary refuses a finite-resolution SCF root.")
    if not agreement.get("agreed", False):
        raise RuntimeError("Force canary requires three-start local root agreement.")
    return {"scf_convergence": convergence, "multi_start_root_agreement": agreement}


def _gate_result(payload: dict[str, Any]) -> dict[str, Any]:
    fd_errors = [
        level["absolute_error_ev_per_angstrom"]
        for row in payload["finite_difference"]
        for level in row["levels"]
    ]
    rotation = payload["rotation"]
    translation = payload["translation"]
    net_force = np.max(
        np.abs(np.asarray(payload["force_sum_hartree_per_angstrom"], dtype=float))
    )
    checks = {
        "finite_difference": max(fd_errors) <= GATES["maximum_fd_error_ev_per_angstrom"],
        "rotation_energy": abs(rotation["energy_difference_ev"])
        <= GATES["maximum_rotation_energy_error_ev"],
        "rotation_force": rotation["maximum_force_covariance_error_ev_per_angstrom"]
        <= GATES["maximum_rotation_force_error_ev_per_angstrom"],
        "translation_energy": abs(translation["energy_difference_ev"])
        <= GATES["maximum_translation_energy_error_ev"],
        "translation_force": translation["maximum_force_difference_ev_per_angstrom"]
        <= GATES["maximum_translation_force_error_ev_per_angstrom"],
        "net_force": net_force <= GATES["maximum_net_force_hartree_per_angstrom"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "gates": dict(GATES),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "open_force_admission_gates": (
            "coordinate_path_smoothness",
            "closed_loop_work",
            "short_nve",
        ),
    }


def run(*, prepared_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    base, candidate, mol2_path, mol2_digest = _load_atoms(prepared_path)
    settings = _settings()
    calculator = _calculator(base, settings, device)
    base_result = _evaluate(base, calculator, settings["solv"], with_force=True)
    force = np.asarray(base_result.forces_hartree_per_angstrom, dtype=float)
    if force.shape != (len(base), 3) or not np.all(np.isfinite(force)):
        raise RuntimeError("Analytic correction force is non-finite or has wrong shape.")
    root = _nominal_root(base_result)

    rotated = base.copy()
    rotated.positions = base.positions @ ROTATION.T
    rotated_result = _evaluate(
        rotated, calculator, settings["solv"], with_force=True
    )
    rotated_force = np.asarray(rotated_result.forces_hartree_per_angstrom, dtype=float)
    _nominal_root(rotated_result)

    translated = base.copy()
    translated.positions = base.positions + TRANSLATION_ANGSTROM
    translated_result = _evaluate(
        translated, calculator, settings["solv"], with_force=True
    )
    translated_force = np.asarray(translated_result.forces_hartree_per_angstrom, dtype=float)
    _nominal_root(translated_result)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc(),
        "git_head": _git_head(),
        "git_worktree_status": _git_worktree_status(),
        "runner": str(Path(__file__).relative_to(REPO_ROOT)),
        "runner_sha256": _sha256(Path(__file__)),
        "claim_boundary": (
            "A pass establishes only local same-scalar kinematic force evidence "
            "for one frozen acetone conformer. It does not establish coordinate "
            "path smoothness, closed-loop work, short-NVE behavior, public ASE "
            "forces, a common stationary MACE--PCM electronic free-energy, or "
            "experimental solvation accuracy."
        ),
        "record": {
            "compound_id": COMPOUND_ID,
            "name": candidate["name"],
            "atom_count": len(base),
            "mol2_path": str(mol2_path),
            "mol2_sha256": mol2_digest,
        },
        "profile": FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
        "declared_scalar": "0.5*<c_MACE-POLAR,f_reac_CPCM>+fixed-topology-aqueous-SMD-CDS",
        "force_scope": "solvent-correction operational-scalar force only",
        "energy_hartree": float(base_result.energy_hartree),
        "force_hartree_per_angstrom": force.tolist(),
        "force_sum_hartree_per_angstrom": np.sum(force, axis=0).tolist(),
        "finite_difference": _finite_difference_rows(
            base, calculator, settings["solv"], force
        ),
        "rotation": {
            "matrix": ROTATION.tolist(),
            "energy_difference_ev": float(
                (rotated_result.energy_hartree - base_result.energy_hartree)
                * Hartree
            ),
            "maximum_force_covariance_error_ev_per_angstrom": float(
                np.max(np.abs((rotated_force - force @ ROTATION.T) * Hartree))
            ),
            "rotated_force_sum_hartree_per_angstrom": np.sum(
                rotated_force, axis=0
            ).tolist(),
        },
        "translation": {
            "vector_angstrom": TRANSLATION_ANGSTROM.tolist(),
            "energy_difference_ev": float(
                (translated_result.energy_hartree - base_result.energy_hartree)
                * Hartree
            ),
            "maximum_force_difference_ev_per_angstrom": float(
                np.max(np.abs((translated_force - force) * Hartree))
            ),
            "translated_force_sum_hartree_per_angstrom": np.sum(
                translated_force, axis=0
            ).tolist(),
        },
        "root": root,
        "force_admission_snapshot": base_result.provenance["force_admission"],
        "mace_geometry_frame": base_result.provenance["mace_geometry_frame"],
    }
    payload["kinematic_gate"] = _gate_result(payload)
    _write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    arguments = parser.parse_args()
    if arguments.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = arguments.device
    payload = run(
        prepared_path=arguments.prepared.resolve(),
        output_path=arguments.output.resolve(),
        device=device,
    )
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
