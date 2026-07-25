from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
SCRIPT = BENCHMARK_DIR / "run_apbs_spline_force_probe.py"
ARTIFACT = (
    BENCHMARK_DIR / "route1-apbs-spline-force-probe-methyl-hexanoate-2026-07-25.json"
)
SPEC = importlib.util.spec_from_file_location("apbs_spline_force_probe", SCRIPT)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_polar_parser_reads_only_printed_solvated_minus_reference_force():
    parsed = PROBE.parse_polar_print_section(
        """
        CALCULATION #1 (solv): MULTIGRID
        mgF  tot 0  99 99 99
        PRINT STATEMENTS
        print energy 1 (solv) - 2 (ref) end
          Global net ELEC energy = -4.250000E+01 kJ/mol
        print force 1 (solv) - 2 (ref) end
          tot 0  1.0  2.0  3.0
          tot 1 -4.0E-1  5.0  6.0
          tot all 0.6 7.0 9.0
        """,
        expected_atoms=2,
        require_forces=True,
    )

    assert parsed["polar_energy_kj_mol"] == -42.5
    assert parsed["polar_force_kj_mol_angstrom"] == [
        [1.0, 2.0, 3.0],
        [-0.4, 5.0, 6.0],
    ]


def test_spline_probe_input_is_polar_only_and_requests_spl4_forces():
    text = PROBE.render_polar_input(
        "molecule.pqr",
        points=129,
        spacing_angstrom=0.25,
        require_forces=True,
    )

    assert text.count("srfm spl4") == 2
    assert text.count("calcforce comps") == 2
    assert "print elecForce solv - ref end" in text
    assert "apolar" not in text.lower()
    assert "srfm mol" not in text


def test_frozen_spline_force_probe_is_self_hashed_and_label_free():
    artifact = _artifact()

    assert PROBE.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["software"]["probe_script_sha256"] == PROBE.sha256_file(SCRIPT)
    assert artifact["software"]["apbs"] == {
        "path": ("/tmp/APBS-3.4.1.Linux-rmODq5/" "APBS-3.4.1.Linux/bin/apbs"),
        "sha256": PROBE.EXPECTED_APBS_EXECUTABLE_SHA256,
        "version": "3.4.1",
    }
    assert artifact["molecule"]["mol2_sha256"] == PROBE.EXPECTED_MOL2_SHA256
    assert artifact["route1_contract"]["gas_phase_mm_energy"] is False
    assert artifact["route1_contract"]["mlip_retraining"] is False
    assert artifact["route1_contract"]["hydration_label_fit_or_residual"] is False
    assert artifact["method"]["nonpolar_included"] is False
    assert artifact["method"]["radius_profile"] == "OpenMM mbondi2"
    assert artifact["method"]["spline_radius_reparameterization_validated"] is False
    assert artifact["method"]["components_checked"] == ("all 3N Cartesian components")


def test_spline_force_probe_covers_all_components_but_is_not_promoted():
    artifact = _artifact()

    assert len(artifact["grid_results"]) == 2
    for grid in artifact["grid_results"]:
        assert len(grid["polar_force_kj_mol_angstrom"]) == 23
        assert set(grid["finite_difference"]) == {"0.003", "0.010"}
        for result in grid["finite_difference"].values():
            assert result["component_count"] == 69
            assert len(result["components"]) == 69
    assert artifact["execution"]["job_count"] == 554
    assert artifact["execution"]["expected_rejection_job_count"] == 1
    assert artifact["molecular_surface_force_request"]["rejected_by_apbs"] is True
    assert artifact["molecular_surface_force_request"]["returncode"] == -6
    assert artifact["molecular_surface_force_request"]["rejection_message"] == (
        "Forces *must* be calculated with spline-based surfaces!"
    )
    assert artifact["cross_grid"]["fine_minus_coarse_energy_kj_mol"] == (
        pytest.approx(0.07266262806000157)
    )
    assert artifact["numerical_gate"]["passed"] is False
    assert artifact["decision"]["production_provider_changed"] is False
    assert artifact["decision"]["spline_force_candidate_promoted"] is False
    assert artifact["decision"]["classification"] == (
        "rejected-as-current-product-force-endpoint"
    )
