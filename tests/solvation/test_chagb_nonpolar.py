from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
WORK_DIR = REPOSITORY_ROOT / ".omx/benchmarks/am1bcc-chagb-nonpolar-route1-20260724"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_chagb_nonpolar as evaluation


def test_provider_component_parsers_use_the_final_energy_block():
    gb = evaluation.parse_gbnsr6_components("""
 EGB = -1.0000  ESURF = 0.1000
 intermediate text
 EGB = -6.5372  ESURF = 2.7647
""")
    pb = evaluation.parse_pbsa_components("""
 ECAVITY= 1.0000 EDISPER = -0.5000
 ECAVITY= 16.8046  EDISPER = -14.7034
""")

    assert gb == {"chagb_polar": -6.5372, "surface_tension": 2.7647}
    assert pb == {"cavity": 16.8046, "dispersion": -14.7034}


def test_provider_component_parsers_fail_closed():
    with pytest.raises(ValueError, match="EGB/ESURF"):
        evaluation.parse_gbnsr6_components("Etot = -4.0")
    with pytest.raises(ValueError, match="ECAVITY/EDISPER"):
        evaluation.parse_pbsa_components("ENPOLAR = 1.0")


def test_protocol_freezes_label_blind_inputs_and_physical_endpoints():
    protocol_path = BENCHMARK_DIR / "chagb_nonpolar_protocol.json"
    protocol, fingerprint = evaluation.load_evaluation_protocol(protocol_path)
    manifest = evaluation.load_label_free_manifest(protocol_path, protocol)

    assert len(fingerprint) == 64
    assert protocol["source_partition"] == "development"
    assert protocol["execution_boundary"] == {
        "energy_phase_reads_experimental_labels": False,
        "energy_phase_inputs": [
            "protocol",
            "label-free source manifest",
            "frozen FreeSolv GAFF MOL2",
        ],
        "summary_phase_may_read_experimental_labels": True,
        "confirmation_remains_closed": True,
        "no_experimental_fit_or_residual_model": True,
    }
    assert protocol["polar"]["model"] == "CHA-GB"
    assert protocol["polar"]["space_angstrom"] == pytest.approx(0.3)
    assert protocol["topology"]["force_field"] == "GAFF2"
    assert protocol["topology"]["atom_types"] == "Pinned FreeSolv GAFF atom types"
    assert protocol["nonpolar_endpoints"]["pbsa_cavity_dispersion"]["inp"] == 2
    assert protocol["nonpolar_endpoints"]["pbsa_cavity_dispersion"]["components"] == [
        "ECAVITY",
        "EDISPER",
    ]
    assert manifest["case_count"] == 526
    assert "experimental" not in json.dumps(manifest).lower()


def test_mol2_charge_replacement_preserves_geometry_and_atom_types():
    source = """@<TRIPOS>MOLECULE
example
 2 1 1 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
1 C1 0.0000 1.0000 2.0000 c3 1 MOL -0.1000
2 O1 1.2000 1.0000 2.0000 oh 1 MOL -0.5000
@<TRIPOS>BOND
1 1 2 1
"""
    result = evaluation.mol2_with_charges(source, [0.25, -0.25])

    atom_lines = (
        result.split("@<TRIPOS>ATOM\n", 1)[1].split("@<TRIPOS>BOND", 1)[0].splitlines()
    )
    first = atom_lines[0].split()
    second = atom_lines[1].split()
    assert first[2:6] == ["0.0000", "1.0000", "2.0000", "c3"]
    assert second[2:6] == ["1.2000", "1.0000", "2.0000", "oh"]
    assert [float(first[8]), float(second[8])] == pytest.approx([0.25, -0.25])


def test_nonbonded_frcmod_overrides_fail_closed():
    evaluation.require_no_frcmod_nonbonded_overrides(
        "MASS\n\nBOND\n\nANGLE\n\nDIHE\n\nIMPROPER\n\nNONBON\n\n"
    )
    with pytest.raises(ValueError, match="nonbonded overrides"):
        evaluation.require_no_frcmod_nonbonded_overrides(
            "MASS\n\nBOND\n\nANGLE\n\nDIHE\n\nIMPROPER\n\n" "NONBON\nn8 1.8240 0.1700\n"
        )


def test_paired_gain_is_positive_when_candidate_reduces_absolute_error():
    result = evaluation.paired_absolute_error_gain(
        baseline_errors=[2.0, -1.0, 3.0],
        candidate_errors=[1.0, -0.5, 2.0],
        resamples=1000,
        confidence=0.95,
        seed=7,
    )

    assert result["mean_mae_gain_kcal_mol"] == pytest.approx(5.0 / 6.0)
    assert result["bootstrap_probability_gain_gt_zero"] == pytest.approx(1.0)
    assert result["case_outcomes"] == {
        "improved": 3,
        "unchanged": 0,
        "worsened": 0,
    }


def test_frozen_summary_reconciles_both_requested_endpoints():
    protocol, fingerprint = evaluation.load_evaluation_protocol(
        BENCHMARK_DIR / "chagb_nonpolar_protocol.json"
    )
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-am1bcc-chagb-nonpolar-2026-07-24.json"
    )

    assert summary["protocol_fingerprint"] == fingerprint
    assert summary["case_count"] == 526
    assert summary["success_count"] == 526
    assert summary["failure_count"] == 0
    assert len(summary["record_sha256"]) == 526
    assert summary["methods"]["am1bcc_obc2_ace"]["mae"] == pytest.approx(
        1.7603512076636891
    )
    assert summary["methods"]["am1bcc_chagb_surface_tension"]["mae"] == pytest.approx(
        1.4494134980988593
    )
    assert summary["methods"]["am1bcc_chagb_cavity_dispersion"]["mae"] == pytest.approx(
        1.321851711026616
    )
    assert summary["paired_absolute_error_gain"]["obc2_ace_to_chagb_cavity_dispersion"][
        "bootstrap_ci"
    ] == pytest.approx([0.32120164766277354, 0.5586767468062764])
    assert summary["label_use_boundary"]["experimental_fit_or_residual_model"] is False
    assert summary["provider_sha256"] == {
        name: details["sha256"]
        for name, details in sorted(protocol["providers"]["executables"].items())
    }


def test_frozen_energy_records_are_label_free_and_reconcile_component_sums():
    records = sorted((WORK_DIR / "records").glob("*.json"))
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-am1bcc-chagb-nonpolar-2026-07-24.json"
    )

    assert len(records) == 526
    for path in records:
        record = core.load_json(path)
        assert "experimental" not in json.dumps(record).lower()
        components = record["components_kcal_mol"]
        predictions = record["predictions_kcal_mol"]
        assert predictions["am1bcc_chagb_surface_tension"] == pytest.approx(
            components["chagb_polar"] + components["gbnsr6_surface_tension"]
        )
        assert predictions["am1bcc_chagb_cavity_dispersion"] == pytest.approx(
            components["chagb_polar"]
            + components["pbsa_cavity"]
            + components["pbsa_dispersion"]
        )
        assert core.sha256_file(path) == summary["record_sha256"][record["compound_id"]]
