"""Fresh, label-free post-audit validation; never rewrite historical evidence.

Re-evaluates the original 12 OBC molecule corpus with native OpenMM Reference,
then runs the existing derivative/workflow and new contract regressions.
Snapshots implementation/test/input bytes before and after execution. This
does not validate a complete Torch CHA endpoint or experimental accuracy.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import benchmark_core as core

HISTORICAL = HERE / "route1-torch-obc2-analytic-derivatives-2026-09-22.json"
HISTORICAL_SHA = "961ba66a97a7b9a0c6d80640409bfb1218de496d42757c06e8dfd2e1e81947f5"
CASES = (
    "benzene",
    "methanol",
    "bromobenzene",
    "methyl-hexanoate",
    "iodobenzene",
    "glucose",
    "4-nitroaniline",
    "dimethyl-sulfide",
    "fluorobenzene",
    "nitralin",
    "aniline",
    "chlorobenzene",
)
TESTS = tuple(
    "tests/solvation/" + name + ".py"
    for name in (
        "test_direct_solvent_contract",
        "test_charge_options_contract",
        "test_command_contract",
        "test_provider_failure_gates",
        "test_qeq_shutdown",
        "test_obc2_parameter_contract",
        "test_continuum_chagb_inputs",
        "test_torch_quadrature_contract",
        "test_torch_dense_budgets",
        "test_torch_continuum_sav",
        "test_torch_continuum_dispersion",
        "test_torch_obc2_parity",
        "test_torch_obc2_derivatives",
        "test_implicit_analytic_derivatives",
        "test_torch_chagb",
        "test_chagb_polar_oracle",
        "test_pbsa_cavity_portable_oracle",
        "test_torch_pbsa_exact_cavity",
        "test_chagb_exact_sp_contract",
    )
)


def source_paths(historical):
    sources = set(historical["implementation_source_sha256"])
    sources.update(
        "maple/function/calculator/extra_correction/implicit/" + name + ".py"
        for name in (
            "torch_dense_budget",
            "torch_pbsa_exact_cavity",
            "torch_chagb",
            "continuum_chagb_inputs",
            "torch_sphere_union_geometry",
            "torch_continuum_sav",
            "torch_continuum_dispersion",
        )
    )
    sources.update(TESTS)
    sources.update(
        {
            str(Path(__file__).resolve().relative_to(ROOT)),
            "tests/solvation/conftest.py",
            "maple/function/read/charge_options.py",
            "maple/function/calculator/extra_correction/implicit/charges.py",
            # This consumer is checked after artifact creation, not inside its
            # own producing run. Freeze its checks without a circular test run.
            "tests/solvation/test_torch_obc2_artifacts.py",
            "tests/solvation/data/pbsa_cavity_oracle_v1.json",
            "docs/implicit-solvation/benchmarks/freeze_pbsa_cavity_oracle.py",
            "docs/implicit-solvation/benchmarks/source_compatibility.py",
            "docs/implicit-solvation/benchmarks/route1-production-safety-source-compatibility-2026-09-27.json",
        }
    )
    sources.update(
        f"tests/solvation/data/amber_gb_reference/audit/{case}/normalized.mol2"
        for case in CASES
    )
    return tuple(sorted(sources))


def snapshots(paths):
    return {name: core.sha256_file(ROOT / name) for name in paths}


def corpus():
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB
    from maple.function.calculator.extra_correction.implicit.obc2_parameters import (
        build_obc2_parameters,
    )
    from maple.function.calculator.extra_correction.implicit.torch_obc2 import TorchOBC2
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    records = []
    for index, case in enumerate(CASES):
        path = (
            ROOT
            / "tests/solvation/data/amber_gb_reference/audit"
            / case
            / "normalized.mol2"
        )
        atoms = MOL2Reader(str(path), charge=0, mult=1)
        shifted = atoms.copy()
        shifted.positions += np.random.default_rng(20260922 + index).normal(
            scale=2e-4, size=(len(atoms), 3)
        )
        for nonpolar in ("ace", "none"):
            reference = OpenMMGB(
                atoms,
                atoms.get_initial_charges(),
                model="obc2",
                nonpolar=nonpolar,
                platform="Reference",
            )
            candidate = TorchOBC2(
                build_obc2_parameters(
                    reference.charges, reference.radius_result, nonpolar=nonpolar
                ),
                device="cpu",
            )
            for name, geometry in (("base", atoms), ("displaced", shifted)):
                expected = reference.evaluate(geometry, need_forces=True)
                actual = candidate.evaluate(geometry, need_forces=True)
                difference = (
                    actual.forces_hartree_per_angstrom
                    - expected.forces_hartree_per_angstrom
                )
                e = abs(actual.energy_hartree - expected.energy_hartree)
                fmax, frms = float(np.abs(difference).max()), float(
                    np.sqrt(np.mean(difference**2))
                )
                components = {
                    key: abs(
                        actual.components_hartree[key]
                        - expected.components_hartree[key]
                    )
                    for key in ("polar", "nonpolar")
                }
                records.append(
                    {
                        "case": case,
                        "geometry": name,
                        "nonpolar": nonpolar,
                        "atom_count": len(atoms),
                        "mol2_sha256": core.sha256_file(path),
                        "coordinate_sha256": core.sha256_bytes(
                            core.canonical_json_bytes(geometry.positions.tolist())
                        ),
                        "reference_energy_hartree": expected.energy_hartree,
                        "candidate_energy_hartree": actual.energy_hartree,
                        "absolute_energy_delta_hartree": e,
                        "component_absolute_deltas_hartree": components,
                        "force_max_delta_hartree_per_angstrom": fmax,
                        "force_rms_delta_hartree_per_angstrom": frms,
                        "passed": bool(
                            e <= 2e-10 + 1e-9 * abs(expected.energy_hartree)
                            and max(components.values()) <= 2e-10
                            and fmax <= 2e-7
                            and frms <= 5e-8
                        ),
                    }
                )
    return records


def run(output, work):
    if output.exists() or work.exists():
        raise FileExistsError(
            "Use fresh output and work paths; do not overwrite validation runs"
        )
    if core.sha256_file(HISTORICAL) != HISTORICAL_SHA:
        raise ValueError("Historical evidence changed")
    historical = core.load_json(HISTORICAL)
    paths = source_paths(historical)
    before = snapshots(paths)
    work.mkdir(parents=True)
    # Freeze source inputs before any new numerical result.
    core.write_json_atomic(work / "source-freeze.json", before)
    records = corpus()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *TESTS,
        "--junitxml=" + str(work / "tests.xml"),
    ]
    executed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
        env=dict(
            os.environ,
            PYTHONPATH=str(ROOT),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    (work / "tests.stdout").write_text(executed.stdout)
    (work / "tests.stderr").write_text(executed.stderr)
    suites = ET.parse(work / "tests.xml").getroot()
    cells = [
        {
            "classname": case.get("classname"),
            "name": case.get("name"),
            "status": (
                "failed"
                if case.find("failure") is not None or case.find("error") is not None
                else "skipped" if case.find("skipped") is not None else "passed"
            ),
        }
        for case in suites.iter("testcase")
    ]
    after = snapshots(paths)
    passed = (
        before == after
        and executed.returncode == 0
        and all(row["passed"] for row in records)
        and bool(cells)
        and all(cell["status"] == "passed" for cell in cells)
    )
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-torch-post-audit-revalidation-v1",
            "recorded_date": "2026-09-27",
            "label_reads": False,
            "new_qm": False,
            "experimental_accuracy_claim": False,
            "full_torch_cha_complete": False,
            "supersedes_current_source_binding_of": {
                "file": HISTORICAL.name,
                "file_sha256": HISTORICAL_SHA,
                "content_sha256": historical["content_sha256"],
            },
            "historical_evidence_bytes_unchanged": core.sha256_file(HISTORICAL)
            == HISTORICAL_SHA,
            "qualified_workflow_cell": {
                "gas": "ani2x",
                "device": "cpu",
                "dtype": "float64",
                "d4": False,
                "solvent": "obc2",
                "openmm_platform": "Reference",
            },
            "implementation_and_test_sha256": before,
            "source_sha256_after": after,
            "source_unchanged": before == after,
            "environment": {
                name: importlib.metadata.version(name)
                for name in ("torch", "openmm", "numpy", "ase", "pytest")
            },
            "corpus": {
                "cases": list(CASES),
                "comparison_count": len(records),
                "records": records,
            },
            "regression": {
                "command": command,
                "exit_code": executed.returncode,
                "cases": cells,
                "junit_sha256": core.sha256_file(work / "tests.xml"),
                "stdout_sha256": core.sha256_file(work / "tests.stdout"),
            },
            "passed": passed,
        }
    )
    core.write_json_atomic(output, artifact)
    print(
        f"corpus={sum(row['passed'] for row in records)}/{len(records)} tests={len(cells)} passed={passed}"
    )
    if not passed:
        raise RuntimeError(f"Revalidation failed; retained outputs at {work}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output.resolve(), arguments.work_dir.resolve())
