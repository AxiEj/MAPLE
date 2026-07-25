from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "docs/implicit-solvation/benchmarks/run_mlip_obc2_ti.py"
)
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "docs/implicit-solvation/benchmarks/mlip_obc2_ti_protocol.json"
)
ARTIFACT_PATH = (
    REPOSITORY_ROOT
    / "docs/implicit-solvation/benchmarks/route1-multi-mlip-obc2-ti-2026-07-25.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_mlip_obc2_ti", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_protocol_is_label_blind_multi_mlip_and_nonpromotable():
    module = _load_module()
    protocol, fingerprint = module.load_protocol(PROTOCOL_PATH)
    payload = json.dumps(protocol, sort_keys=True).lower()

    assert len(fingerprint) == 64
    assert [model["name"] for model in protocol["models"]] == [
        "maceoff23m",
        "aimnet2",
        "ani2x",
    ]
    assert "experimental_kcal_mol" not in payload
    assert protocol["execution_boundary"]["promotion_allowed"] is False
    assert protocol["diagnostic_gates"]["promotion_allowed"] is False
    assert protocol["sampling"]["production_estimator_target"] == "mbar"


def test_composite_simpson_is_exact_for_quadratic():
    module = _load_module()
    lambdas = np.linspace(0.0, 1.0, 5)
    assert module.composite_simpson(lambdas, lambdas**2) == pytest.approx(1.0 / 3.0)


def test_endpoint_fep_is_stable_and_reports_weight_collapse():
    module = _load_module()
    constant = module.endpoint_fep([-4.0] * 20, 298.15)
    collapsed = module.endpoint_fep([-10.0] + [0.0] * 19, 298.15)

    assert constant["delta_g_kcal_mol"] == pytest.approx(-4.0)
    assert constant["effective_sample_fraction"] == pytest.approx(1.0)
    assert collapsed["effective_sample_fraction"] < 0.1


def test_lambda_calculator_scales_same_energy_and_force_terms():
    module = _load_module()

    class FakeGas:
        def calculate(self, atoms, properties, system_changes):
            self.results = {
                "energy": 2.0,
                "forces": np.full((2, 3), 3.0),
            }

    class FakeResult:
        energy_hartree = -0.5
        forces_hartree_per_angstrom = np.full((2, 3), -2.0)
        components_hartree = {"polar": -0.6, "nonpolar": 0.1}

    class FakeCorrection:
        def evaluate(self, atoms, need_forces):
            return FakeResult()

    calculator = module.LambdaImplicitCalculator(FakeGas(), FakeCorrection(), 0.25)
    components = calculator.components(object(), need_forces=True)

    assert components["energy_hartree"] == pytest.approx(1.875)
    assert components["forces_hartree_per_angstrom"] == pytest.approx(
        np.full((2, 3), 2.5)
    )


def test_reusable_record_requires_self_hash_and_matching_identity(tmp_path):
    module = _load_module()
    record = module.seal_artifact(
        {
            "protocol_id": module.PROTOCOL_ID,
            "protocol_fingerprint": "a" * 64,
            "model": "ani2x",
            "compound_id": "mobley_test",
        }
    )
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    loaded = module._load_reusable_record(
        path,
        model="ani2x",
        compound_id="mobley_test",
        protocol_fingerprint="a" * 64,
    )
    assert loaded["content_sha256"] == record["content_sha256"]

    loaded["model"] = "maceoff23m"
    path.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid self-hash"):
        module._load_reusable_record(
            path,
            model="ani2x",
            compound_id="mobley_test",
            protocol_fingerprint="a" * 64,
        )


def test_frozen_multi_mlip_ti_artifact_is_self_consistent():
    module = _load_module()
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    protocol, fingerprint = module.load_protocol(PROTOCOL_PATH)

    assert artifact["content_sha256"] == module.artifact_content_sha256(artifact)
    assert artifact["protocol_fingerprint"] == fingerprint
    assert artifact["command_provenance"]["script_sha256"] == module.sha256_file(
        SCRIPT_PATH
    )
    assert artifact["promotion_allowed"] is False
    assert artifact["estimator"]["mbar_executed"] is False
    assert len(artifact["case_results"]) == 9
    assert set(artifact["model_summaries"]) == {
        "maceoff23m",
        "aimnet2",
        "ani2x",
    }
    for summary in artifact["model_summaries"].values():
        assert summary["ti_minus_fixed_geometry_mae_kcal_mol"] > 0.0
        assert summary["all_diagnostic_checks_pass"] is False

    for scored in artifact["case_results"]:
        raw_path = REPOSITORY_ROOT / scored["raw_record_path"]
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert scored["raw_record_sha256"] == module.sha256_file(raw_path)
        assert raw["content_sha256"] == module.artifact_content_sha256(raw)
        assert "experimental_kcal_mol" not in json.dumps(raw, sort_keys=True).lower()
        mol2_path = REPOSITORY_ROOT / raw["mol2"]
        assert mol2_path.is_file()
        assert raw["mol2_sha256"] == module.sha256_file(mol2_path)
        assert all(
            window["production_sample_count"]
            == protocol["diagnostic_gates"][
                "expected_production_samples_per_window"
            ]
            for chain in raw["chains"]
            for window in chain["windows"]
        )
        for chain in raw["chains"]:
            for window in chain["windows"]:
                trajectory = REPOSITORY_ROOT / window["trajectory"]
                assert trajectory.is_file()
                assert window["trajectory_sha256"] == module.sha256_file(trajectory)
