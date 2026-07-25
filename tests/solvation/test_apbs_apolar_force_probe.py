from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
SCRIPT = BENCHMARK_DIR / "run_apbs_apolar_force_probe.py"
ARTIFACT = (
    BENCHMARK_DIR / "route1-apbs-apolar-force-probe-methyl-hexanoate-2026-07-25.json"
)
SPEC = importlib.util.spec_from_file_location("apbs_apolar_force_probe", SCRIPT)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_apolar_parser_keeps_calculation_and_print_force_blocks_distinct():
    parsed = PROBE.parse_apolar_output(
        """
        CALCULATION #1 (nonpolar): APOLAR
          tot 0 -0.10 0.20 0.30
          tot 1  0.40 0.50 0.60
        Solvent Accessible Surface Area (SASA) for each atom:
        PRINT STATEMENTS
        print APOL energy 1 (nonpolar) end
          Global net APOL energy = 2.0 kJ/mol
        print APOL force 1 (nonpolar) end
          tot 0 1.0 2.0 3.0
          tot 1 4.0 5.0 6.0
          tot all 5.0 7.0 9.0
        ----------------------------------------
        """,
        expected_atoms=2,
        require_forces=True,
    )

    assert parsed["apolar_energy_kj_mol"] == 2.0
    assert parsed["calculation_total_force_kj_mol_angstrom"] == [
        [-0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    assert parsed["printed_total_component_native"] == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]


def test_frozen_apolar_probe_is_self_hashed_and_route1_clean():
    artifact = _artifact()

    assert PROBE.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["software"]["probe_script_sha256"] == PROBE.sha256_file(SCRIPT)
    assert artifact["base"]["apolar_energy_kj_mol"] == pytest.approx(38.01084434894)
    assert artifact["route1_contract"]["gas_phase_mm_energy"] is False
    assert artifact["route1_contract"]["mlip_retraining"] is False
    assert artifact["route1_contract"]["hydration_label_fit_or_residual"] is False
    assert artifact["method"]["component"] == "nonpolar only"
    assert artifact["method"]["components_checked"] == ("all 3N Cartesian components")


def test_apolar_print_block_requires_scaling_but_scaled_force_still_fails_fd():
    artifact = _artifact()
    base = artifact["base"]
    calculation = np.asarray(base["calculation_total_force_kj_mol_angstrom"])
    printed = np.asarray(base["printed_total_component_native"])

    assert calculation == pytest.approx(
        -PROBE.GAMMA_KJ_MOL_ANGSTROM2 * printed,
        abs=5.0e-4,
    )
    assert (
        base["print_to_calculation_relation"][
            "maximum_absolute_residual_kj_mol_angstrom"
        ]
        < 5.0e-4
    )
    for result in artifact["finite_difference"].values():
        assert result["calculation_total_force"]["component_count"] == 69
        assert result["printed_total_component"]["component_count"] == 69
    assert artifact["execution"]["job_count"] == 277
    assert artifact["numerical_gate"]["passed"] is False
    assert artifact["decision"]["apbs_apolar_force_promoted"] is False
    assert artifact["decision"]["classification"] == (
        "rejected-for-force-capable-product-use"
    )
