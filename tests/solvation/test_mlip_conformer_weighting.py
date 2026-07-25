from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_mlip_conformer_weighting as weighting


def test_discrete_partition_function_uses_relative_mlip_energies():
    result = weighting.discrete_partition_result(
        mlip_energy_hartree=[
            -10.0,
            -10.0 + 3.0 / weighting.KCAL_PER_HARTREE,
        ],
        solvent_kcal_mol=[-3.0, -6.0],
        temperature_kelvin=298.15,
        reference_geometry_kcal_mol=-3.0,
        experimental_kcal_mol=-3.5,
    )
    shifted = weighting.discrete_partition_result(
        mlip_energy_hartree=[
            990.0,
            990.0 + 3.0 / weighting.KCAL_PER_HARTREE,
        ],
        solvent_kcal_mol=[-3.0, -6.0],
        temperature_kelvin=298.15,
        reference_geometry_kcal_mol=-3.0,
        experimental_kcal_mol=-3.5,
    )

    assert result["ensemble_kcal_mol"] == pytest.approx(-3.406, abs=0.002)
    assert result["gas_weights"][1] == pytest.approx(0.0063, abs=0.0002)
    assert result["solution_weights"] == pytest.approx([0.5, 0.5], abs=1.0e-12)
    assert shifted["ensemble_kcal_mol"] == pytest.approx(result["ensemble_kcal_mol"])


def test_single_conformer_reduces_to_endpoint_correction():
    result = weighting.discrete_partition_result(
        mlip_energy_hartree=[-123.456],
        solvent_kcal_mol=[-4.25],
        temperature_kelvin=298.15,
        reference_geometry_kcal_mol=-4.0,
        experimental_kcal_mol=-4.5,
    )

    assert result["ensemble_kcal_mol"] == pytest.approx(-4.25)
    assert result["gas_weights"] == [1.0]
    assert result["solution_weights"] == [1.0]
    assert result["conformational_correction_from_reference_kcal_mol"] == pytest.approx(
        -0.25
    )


def test_protocol_pins_source_artifacts_and_mlip_checkpoint():
    protocol, fingerprint = weighting.load_weighting_protocol(
        BENCHMARK_DIR / "mlip_conformer_weighting_protocol.json"
    )

    assert len(fingerprint) == 64
    assert protocol["source_partition"] == "development"
    assert protocol["model"]["name"] == "maceoff23m"
    assert len(protocol["model"]["checkpoint_sha256"]) == 64
    assert protocol["model"]["maple_calculator_output_unit"] == "hartree"
    assert protocol["evaluation"] == {
        "charge_method": "abcg2",
        "gb_model": "obc2",
        "nonpolar": "ace",
        "reference_geometry_rmsd_dedup_angstrom": 0.125,
        "temperature_kelvin": 298.15,
        "weighting": "discrete-mlip-single-point-log-sum-exp-reference-union",
    }

    for path_key, hash_key in (
        ("conformer_protocol", "conformer_protocol_sha256"),
        ("conformer_summary", "conformer_summary_sha256"),
    ):
        path = BENCHMARK_DIR / protocol["source_evidence"][path_key]
        assert core.sha256_file(path) == protocol["source_evidence"][hash_key]


def test_forward_weighting_protocol_uses_am1bcc_default_explicitly():
    protocol, fingerprint = weighting.load_weighting_protocol(
        BENCHMARK_DIR / "mlip_conformer_weighting_am1bcc_protocol.json"
    )

    assert len(fingerprint) == 64
    assert protocol["protocol_id"] == "maple-route1-am1bcc-mlip-conformer-weighting-v1"
    assert protocol["evaluation"]["charge_method"] == "am1bcc"
    assert protocol["evaluation"]["gb_model"] == "obc2"
    assert protocol["evaluation"]["nonpolar"] == "ace"


def test_summary_metrics_separate_baseline_and_mlip_weighting():
    records = [
        {
            "compound_id": "rigid",
            "flexibility_bin": "rigid",
            "experimental_kcal_mol": -2.0,
            "result": {
                "reference_geometry_kcal_mol": -1.0,
                "ensemble_kcal_mol": -1.5,
            },
        },
        {
            "compound_id": "flexible",
            "flexibility_bin": "flexible",
            "experimental_kcal_mol": -4.0,
            "result": {
                "reference_geometry_kcal_mol": -6.0,
                "ensemble_kcal_mol": -4.5,
            },
        },
    ]

    metrics = weighting.summarize_metrics(records)

    assert metrics["reference_geometry"]["mae_kcal_mol"] == pytest.approx(1.5)
    assert metrics["mlip_weighted_ensemble"]["mae_kcal_mol"] == pytest.approx(0.5)
    assert metrics["case_outcomes"] == {"improved": 2, "unchanged": 0, "worsened": 0}
    assert metrics["flexibility_strata"]["flexible"]["mlip_weighted_ensemble"][
        "rmse_kcal_mol"
    ] == pytest.approx(0.5)


def test_reference_geometry_replaces_duplicate_or_appends_distinct_state():
    symbols = ["C", "C", "O", "H"]
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.0, 1.0, 0.0], [-0.5, 0.8, 0.0]]
    )
    rotated_translated = reference @ np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    ) + np.asarray([4.0, -2.0, 1.0])
    frames = [{"positions_angstrom": rotated_translated}]

    replaced, metadata = weighting.reference_geometry_union(
        frames, reference, symbols, threshold_angstrom=0.125
    )

    assert len(replaced) == 1
    assert metadata["action"] == "replaced"
    assert metadata["aligned_heavy_atom_rmsd_angstrom"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert replaced[0]["positions_angstrom"] == pytest.approx(reference)

    distinct = reference.copy()
    distinct[2, 1] += 1.0
    appended, metadata = weighting.reference_geometry_union(
        frames, distinct, symbols, threshold_angstrom=0.125
    )

    assert len(appended) == 2
    assert metadata["action"] == "appended"


def test_frozen_mlip_weighting_summary_reconciles_the_development_diagnostic():
    protocol, fingerprint = weighting.load_weighting_protocol(
        BENCHMARK_DIR / "mlip_conformer_weighting_protocol.json"
    )
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-mlip-conformer-weighting-2026-07-24.json"
    )

    assert summary["protocol_fingerprint"] == fingerprint
    assert summary["source_partition"] == "development"
    assert summary["case_count"] == 20
    assert summary["source_conformer_count"] == 1294
    assert summary["partition_state_count"] == 1298
    assert len(summary["record_sha256"]) == 20
    assert (
        summary["environment"]["checkpoint_sha256"]
        == protocol["model"]["checkpoint_sha256"]
    )
    assert summary["metrics"]["reference_geometry"]["mae_kcal_mol"] == pytest.approx(
        1.9830295462382375
    )
    assert summary["metrics"]["mlip_weighted_ensemble"][
        "mae_kcal_mol"
    ] == pytest.approx(1.7992064301252966)
    assert summary["metrics"]["influence_analysis"]["most_influential_compound_id"] == (
        "mobley_8124669"
    )
