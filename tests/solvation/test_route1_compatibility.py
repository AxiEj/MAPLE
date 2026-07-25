from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
SPEC = importlib.util.spec_from_file_location(
    "run_route1_compatibility",
    BENCHMARK_DIR / "run_route1_compatibility.py",
)
compatibility = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compatibility)


def test_am1bcc_record_loader_requires_one_complete_record(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"records":[{"compound_id":"case","am1bcc_charges_e":[-0.1,0.1]}]}',
        encoding="utf-8",
    )

    charges, record = compatibility._load_am1bcc_record(
        manifest,
        "case",
        atom_count=2,
    )

    assert np.allclose(charges, [-0.1, 0.1])
    assert record["compound_id"] == "case"
    with pytest.raises(ValueError, match="one finite charge per atom"):
        compatibility._load_am1bcc_record(manifest, "case", atom_count=3)


def test_frozen_multi_mlip_trace_passes_only_the_declared_compatibility_gate():
    trace = json.loads(
        (
            BENCHMARK_DIR / "route1-compatibility-methyl-hexanoate-2026-07-24.json"
        ).read_text(encoding="utf-8")
    )

    assert trace["formula"] == (
        "E_solution(R) = E_MLIP,gas(R) + G_polar(R,q_fixed) + G_nonpolar(R)"
    )
    assert trace["prohibited_terms"] == {
        "gas_phase_mm_energy": False,
        "retraining": False,
        "hydration_label_residual": False,
    }
    assert {record["model"] for record in trace["models"]} == {
        "maceoff23m",
        "aimnet2",
    }
    assert trace["cross_model"]["solvent_energy_spread_hartree"] <= 1.0e-12
    assert trace["all_checks_pass"] is True
    assert all(all(record["passes"].values()) for record in trace["models"])
    assert all(
        record["solvation_provenance"]["component_decomposition"]
        == "single-context OpenMM energy-parameter derivative"
        for record in trace["models"]
    )
    assert all(
        record["solvation_provenance"]["energy_force_evaluations_per_call"] == 1
        for record in trace["models"]
    )
    assert all(
        record["sp"]["combined_potential_force_fd_all_coordinates"]["component_count"]
        == 3 * trace["atom_count"]
        for record in trace["models"]
    )
    assert all(
        "displaced_geometry_energy_smoke" in record and "scan_smoke" not in record
        for record in trace["models"]
    )
    assert "not broad chemical accuracy" in trace["claim_scope"]
