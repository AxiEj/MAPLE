from __future__ import annotations

import importlib.util
import json
from pathlib import Path

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-openmm-platform-audit-2026-07-25.json"
)
SCRIPT_PATH = BENCHMARK_DIR / "run_openmm_platform_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "platform_audit_benchmark_core",
    BENCHMARK_DIR / "benchmark_core.py",
)
core = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(core)


def test_openmm_platform_audit_is_sealed_and_reproducibly_bound():
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert artifact["content_sha256"] == core.artifact_content_sha256(artifact)
    assert artifact["source_manifest"]["sha256"] == core.sha256_file(
        Path(__file__).parent / "data/amber_gb_reference/manifest.json"
    )
    assert artifact["command_provenance"]["script_sha256"] == core.sha256_file(
        SCRIPT_PATH
    )


def test_openmm_platform_audit_supports_the_product_default_without_physics_change():
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    summary = artifact["summary"]

    assert artifact["route"] == {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "role": "Baseline/Product Route",
        "formula": (
            "E_solution(R)=E_MLIP,gas(R)+"
            "G_polar(R,q_fixed)+G_nonpolar(R)"
        ),
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "retraining": False,
    }
    assert artifact["platform_contract"] == {
        "product_default": "CPU",
        "product_properties": {
            "DeterministicForces": "true",
            "Threads": "1",
        },
        "parity_control": "Reference",
        "parity_control_properties": {},
    }
    assert artifact["all_checks_pass"] is True
    assert all(artifact["checks"].values())
    assert summary["case_count"] == 12
    assert summary["model_slot_count"] == 60
    assert summary["polar_status_counts"] == {
        "expected-unavailable": 2,
        "success": 58,
    }
    assert summary["cpu_independent_context_repeat_maxima"] == {
        "component_max_abs_kcal_mol": 0.0,
        "energy_abs_kcal_mol": 0.0,
        "force_max_abs_kcal_mol_angstrom": 0.0,
    }
    assert (
        summary["cpu_vs_reference_maxima"]["energy_abs_kcal_mol"]
        <= artifact["tolerances"]["energy_abs_kcal_mol"]
    )
    assert (
        summary["cpu_vs_reference_maxima"][
            "force_max_abs_kcal_mol_angstrom"
        ]
        <= artifact["tolerances"]["force_max_abs_kcal_mol_angstrom"]
    )
    assert summary["obc2_ace_energy_force_speedup"]["minimum"] > 1.0
    assert any(
        "not a claim that MLIP plus implicit solvent is faster than bare MM"
        in boundary
        for boundary in artifact["claim_boundaries"]
    )


def test_openmm_platform_audit_numbers_are_bound_to_route1_documents():
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    summary = artifact["summary"]
    documents = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            BENCHMARK_DIR.parent / "README.md",
            BENCHMARK_DIR.parent / "ROUTE1_PRODUCT_SPEC.md",
            BENCHMARK_DIR.parent / "VALIDATION_STATUS.md",
            BENCHMARK_DIR / "README.md",
        )
    )

    maxima = summary["cpu_vs_reference_maxima"]
    speedup = summary["obc2_ace_energy_force_speedup"]
    energy_text = f"{maxima['energy_abs_kcal_mol']:.3e}".replace("e-0", "e-")
    force_text = (
        f"{maxima['force_max_abs_kcal_mol_angstrom']:.3e}".replace("e-0", "e-")
    )
    assert energy_text in documents
    assert force_text in documents
    assert f"{speedup['minimum']:.2f}x" in documents
    assert f"{speedup['median']:.2f}x" in documents
    assert f"{speedup['maximum']:.2f}x" in documents
