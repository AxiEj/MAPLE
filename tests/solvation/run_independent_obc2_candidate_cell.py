#!/usr/bin/env python3
"""Generate test-only exact ANI2x/OBC-II candidate-cell evidence.

This script monkeypatches the one production admission function directly.  It
never changes a public input, environment variable, constructor, or CLI option.
Its output is candidate qualification, never installed qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
import validate_independent_obc2_default as qualification  # noqa: E402

OUTPUT = ROOT / ".omx/benchmarks/route1-aimnet2025-20261024/obc-ani/candidate-exact-cell.json"
CHECKPOINT = ROOT / "maple/function/calculator/model/ani2x.pt"
CHECKPOINT_SHA256 = "c8c41e89d5ccbe7b5c9f46ce4a262b913ad0c428a3a3742cc6e83ac10d4d43aa"
WATER_MOL2 = """@<TRIPOS>MOLECULE
WATER
 3 2 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 O1          0.000000    0.000000    0.000000 O.3       1 WAT       -0.834000
      2 H1          0.957200    0.000000    0.000000 H         1 WAT        0.417000
      3 H2         -0.239987    0.927297    0.000000 H         1 WAT        0.417000
@<TRIPOS>BOND
     1    1    2 1
     2    1    3 1
@<TRIPOS>SUBSTRUCTURE
     1 WAT         1 TEMP              0 ****  ****    0 ROOT
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(value) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach): value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu): value = cpu()
    return np.asarray(value, dtype=np.float64)


def seal(value: dict) -> dict:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return result


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"candidate evidence is immutable: {output}")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("ANI2x checkpoint identity changed")
    source_before = qualification.source_snapshot(require_clean=True)
    from maple.function.read.filereader.mol2_reader import MOL2Reader
    from maple.function.calculator.calculator_base import numerical_hessian_from_atoms
    import maple.function.calculator.set_calculator as module

    records = []
    with tempfile.TemporaryDirectory(prefix="maple-obc2-candidate-") as temporary:
        mol2 = Path(temporary) / "water.mol2"; mol2.write_text(WATER_MOL2, encoding="utf-8")
        for index, nonpolar in enumerate(("ace", "none")):
            atoms = MOL2Reader(str(mol2), charge=0, mult=1)
            observed = {}
            def exact_test_gate(calculator, correction, *, model, device, d4):
                precision = calculator.inference_precision_provenance
                provenance = correction.underlying_provider.provenance
                observed.update(model=model, device=device, d4=d4,
                                checkpoint_sha256=precision["original_checkpoint_sha256"],
                                dtype=precision["effective_dtype"], provider=provenance)
                if not (model == "ani2x" and str(device).split(":", 1)[0] == "cpu" and d4 is False
                        and precision["effective_dtype"] == "float64"
                        and precision["original_checkpoint_sha256"] == CHECKPOINT_SHA256
                        and provenance["runtime_provider"] == "torch-obc2"
                        and provenance["model"] == "obc2"
                        and provenance["profile"] == "obc2-mbondi2"
                        and provenance["nonpolar"] == nonpolar):
                    raise RuntimeError("candidate exact-cell identity mismatch")
                calculator.analytic_implicit_derivatives_admitted = True
                correction.analytic_task_derivatives_admitted = True
            original = module._admit_independent_obc2_analytic_cell
            module._admit_independent_obc2_analytic_cell = exact_test_gate
            try:
                factory = module.SetCalculator(
                    "cpu", "ani2x", str(Path(temporary) / nonpolar), atoms=atoms,
                    implicit="gb", solvent="water", d4=False,
                    model_options={"hessian": "analytic", "model_path": str(CHECKPOINT)},
                    solvation_options={"implicit": "water", "method": "gb", "provider": "torch",
                                        "model": "obc2", "profile": "obc2-mbondi2",
                                        "nonpolar": nonpolar, "experimental": True},
                    charge_options={"source": "mol2", "label": "frozen-candidate",
                                    "mode": "fixed", "geometry": "keep"},
                )
                calculator = factory.set_calculator()
            finally:
                module._admit_independent_obc2_analytic_cell = original
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            forces = array(atoms.get_forces())
            hessian = array(calculator.get_hessian(atoms))
            correction = calculator.solvent_correction
            solvent_hessian = array(correction.get_hessian(atoms))
            calculator.solvent_correction = None
            try:
                gas_hessian = array(calculator.get_hessian(atoms))
            finally:
                calculator.solvent_correction = correction
            composition_max = float(
                np.max(np.abs(hessian - (gas_hessian + solvent_hessian)))
            )
            fd_ladder = []
            for step in (1.0e-6, 2.0e-6, 5.0e-6):
                reference_fd = numerical_hessian_from_atoms(
                    calculator, atoms, delta=step
                )
                difference = hessian - reference_fd
                rms = float(np.sqrt(np.mean(difference**2)))
                maximum = float(np.max(np.abs(difference)))
                fd_ladder.append(
                    {
                        "step_angstrom": step,
                        "rms_hartree_per_angstrom2": rms,
                        "max_hartree_per_angstrom2": maximum,
                        "passed": rms <= 5.0e-6 and maximum <= 2.0e-5,
                    }
                )
            direction = np.random.default_rng(2026102301 + index).normal(size=3 * len(atoms))
            direction /= np.linalg.norm(direction)
            provider = correction.underlying_provider
            def forbidden_dense_hessian(*_args, **_kwargs):
                raise AssertionError("direct HVP materialized a dense Hessian")
            with (
                patch.object(
                    calculator,
                    "_analytic_hessian",
                    side_effect=forbidden_dense_hessian,
                ),
                patch.object(
                    provider,
                    "hessian",
                    side_effect=forbidden_dense_hessian,
                ),
                patch.object(
                    provider.kernel,
                    "hessian",
                    side_effect=forbidden_dense_hessian,
                ),
            ):
                hvp, hvp_forces, hvp_energy = calculator.get_hvp(atoms, direction)
            hvp = array(hvp).reshape(-1); hvp_forces = array(hvp_forces).reshape(forces.shape)
            dense = hessian @ direction
            record = {
                "nonpolar": nonpolar, "energy_hartree": energy,
                "same_scalar_energy_abs_hartree": abs(
                    float(array(hvp_energy).reshape(-1)[0]) - energy
                ),
                "same_scalar_force_max_hartree_per_angstrom": float(np.max(np.abs(hvp_forces - forces))),
                "hessian_asymmetry_max_hartree_per_angstrom2": float(np.max(np.abs(hessian - hessian.T))),
                "hessian_composition_max_hartree_per_angstrom2": composition_max,
                "hessian_force_fd_ladder": fd_ladder,
                "direct_hvp_dense_max_hartree_per_angstrom2": float(np.max(np.abs(hvp - dense))),
                "direct_hvp_dense_hessian_forbidden": True,
                "gate_identity": observed,
            }
            record["passed"] = bool(
                record["same_scalar_energy_abs_hartree"] <= 1e-12
                and record["same_scalar_force_max_hartree_per_angstrom"] <= 1e-12
                and record["hessian_asymmetry_max_hartree_per_angstrom2"] <= 1e-10
                and record["hessian_composition_max_hartree_per_angstrom2"] <= 1e-12
                and all(row["passed"] for row in fd_ladder)
                and np.allclose(hvp, dense, rtol=1e-9, atol=1e-9)
            )
            records.append(record)
    source_after = qualification.source_snapshot(require_clean=True)
    if source_before != source_after:
        raise RuntimeError("Candidate qualification source tree changed during execution")
    artifact = seal({
        "schema_version": 1, "artifact_type": "independent-obc2-candidate-exact-cell-v1",
        "qualification_stage": "candidate qualification, not installed qualification",
        "candidate_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "harness_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_snapshot": source_before,
        "checkpoint_sha256": CHECKPOINT_SHA256, "model": "ani2x", "device": "cpu",
        "dtype": "float64", "d4": False, "nonpolar": ["ace", "none"],
        "expected_count": 2, "passed_count": sum(row["passed"] for row in records),
        "records": records, "passed": all(row["passed"] for row in records),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT / ".omx").resolve())
    artifact = run(output)
    print(json.dumps({"path": str(output), "content_sha256": artifact["content_sha256"],
                      "passed": artifact["passed"]}, indent=2, sort_keys=True))
    if not artifact["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
