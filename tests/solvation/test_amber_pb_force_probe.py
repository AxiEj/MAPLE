from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
ARTIFACT = BENCHMARK_DIR / "amber-pb-inp2-force-probe-methyl-hexanoate-2026-07-24.json"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_amber_pb_force_probe_preserves_route1_boundary():
    artifact = _artifact()
    boundary = artifact["route1_boundary"]

    assert artifact["schema_version"] == 1
    assert boundary["formula"] == (
        "E_solution(R) = E_MLIP,gas(R) + G_polar(R,q_fixed) + G_nonpolar(R)"
    )
    assert boundary["gas_phase_mm_energy_in_reported_potential"] is False
    assert boundary["hydration_label_fit_or_residual_model"] is False
    assert boundary["internal_exact_difference"] == (
        "DeltaG_candidate(R) = E_MM,PB(inp=2)(R) - E_MM,vacuum(R)"
    )


def test_amber_pb_force_probe_rejects_product_derivatives():
    artifact = _artifact()
    force_check = artifact["force_consistency"]
    eligibility = artifact["product_eligibility"]

    assert force_check["force_definition"] == (
        "F_candidate = F_MM,PB(inp=2) - F_MM,vacuum"
    )
    assert force_check["finite_difference_definition"] == (
        "-d[DeltaG_candidate]/dR by centered energy differences"
    )
    assert force_check["all_samples_within_tolerance"] is False
    assert force_check["maximum_absolute_error_kcal_mol_angstrom"] > 1.0
    assert eligibility == {
        "single_point_reference_energy": True,
        "force_consistent_runtime": False,
        "optimization": False,
        "relaxed_scan": False,
        "default_provider": False,
    }


def test_amber_pb_force_probe_records_both_nonpolar_components_and_steps():
    artifact = _artifact()
    components = artifact["reference_geometry"]["solution_components_kcal_mol"]
    cancellation = artifact["reference_geometry"]["mm_component_cancellation"]
    force_check = artifact["force_consistency"]

    assert components["pb"] < 0.0
    assert components["cavity"] > 0.0
    assert components["dispersion"] < 0.0
    assert cancellation["all_components_cancel_within_tolerance"] is True
    assert cancellation["maximum_absolute_difference_kcal_mol"] < 1.0e-12
    assert set(cancellation["components"]) == {
        "bond",
        "angle",
        "dihedral",
        "vdw",
        "elec",
        "vdw_14",
        "elec_14",
    }
    assert force_check["steps_angstrom"] == [0.003, 0.01]
    assert len(force_check["samples"]) == 8
