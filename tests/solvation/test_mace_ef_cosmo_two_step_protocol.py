from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
RUNNER_PATH = BENCHMARKS / "run_mace_ef_cosmo_freesolv20_two_step.py"
PREREGISTRATION_PATH = BENCHMARKS / "mace-ef-cosmo-freesolv20-two-step-prereg-v1.json"


def _load_runner() -> ModuleType:
    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location(
        "run_mace_ef_cosmo_freesolv20_two_step_test",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeElectronic:
    def __init__(self) -> None:
        self.fields = []

    def evaluate(self, positions: np.ndarray, field: np.ndarray) -> SimpleNamespace:
        del positions
        values = np.asarray(field, dtype=np.float64)
        self.fields.append(values.copy())
        source = np.zeros_like(values)
        source[:, 0] = np.asarray((1.0, -1.0)) + 0.1 * values[:, 0]
        return SimpleNamespace(conjugate_source_raw=source)


class _FakeContinuum:
    def __init__(self) -> None:
        self.drive_count = 0

    def drive_cartesian(self, positions: np.ndarray, source: np.ndarray) -> np.ndarray:
        del positions
        self.drive_count += 1
        field = np.zeros_like(source)
        field[:, 0] = 0.2 * source[:, 0]
        return field

    def energy(self, positions: np.ndarray, source: np.ndarray) -> float:
        field = np.zeros_like(source)
        field[:, 0] = 0.2 * source[:, 0]
        del positions
        return 0.5 * MACE_POLAR_L1_PAIRING.pair(source, field)


def test_two_step_preregistration_excludes_every_extra_thermodynamic_term():
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))

    assert preregistration["status"] == "frozen-before-two-step-execution"
    assert preregistration["response_map"] == {
        "anderson_acceleration": False,
        "continuum_drive": "f_k = P_COSMO(c_k)",
        "first_update": "c1 = Q*dE_MACE-EF/df evaluated at f0",
        "initial_source": ("c0 = Q*dE_MACE-EF/df evaluated at zero external field"),
        "map_applications": 2,
        "mixing": "none",
        "no_third_model_evaluation": True,
        "reported_source": "c2",
        "second_update": "c2 = Q*dE_MACE-EF/df evaluated at f1",
    }
    energy = preregistration["reported_energy"]
    assert energy["primary_component"] == "U_segment_COSMO(c2)"
    assert energy["finite_dielectric_scaling"] is False
    excluded = set(energy["excluded_terms"])
    assert {
        "COSMO-RS or COSMOspace activity",
        "hydrogen-bond interaction",
        "SMD-CDS or another nonpolar term",
        "standard-state or reference-state correction",
        "field-conditioned MACE-EF intrinsic energy difference",
    }.issubset(excluded)
    assert (
        "hydration free-energy MAE"
        in preregistration["experimental_label_policy"]["must_not_be_called"]
    )


def test_two_step_response_applies_exactly_two_unmixed_maps_and_no_third():
    runner = _load_runner()
    electronic = _FakeElectronic()
    continuum = _FakeContinuum()
    positions = np.zeros((2, 3), dtype=np.float64)

    result = runner._two_step_response(electronic, continuum, positions)

    assert result["map_applications"] == 2
    assert result["electronic_evaluation_count"] == 3
    assert result["third_map_evaluated"] is False
    assert result["source_updates"]["coordinate_invariant_norm"] is False
    assert result["source_updates"]["mathematical_contraction_claimed"] is False
    assert len(electronic.fields) == 3
    assert continuum.drive_count == 3
    np.testing.assert_allclose(electronic.fields[0], np.zeros((2, 4)))
    assert result["source_updates"]["first_maximum_absolute_component"] == (
        pytest.approx(0.02)
    )
    assert result["source_updates"]["second_maximum_absolute_component"] == (
        pytest.approx(0.0004)
    )
    assert result["source_updates"]["second_to_first_ratio"] == pytest.approx(0.02)
    assert [stage["source_index"] for stage in result["stages"]] == [0, 1, 2]
    assert all(
        stage["half_coupling_identity_error_ev"] == pytest.approx(0.0)
        for stage in result["stages"]
    )


def test_two_step_source_and_empty_distribution_fail_closed():
    runner = _load_runner()
    bad = SimpleNamespace(
        conjugate_source_raw=np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    )
    with pytest.raises(RuntimeError, match="neutral-charge gate"):
        runner._source_array(bad)
    assert runner._distribution([], units="kcal/mol") == {
        "count": 0,
        "units": "kcal/mol",
    }
    assert runner.OUTPUT_PATH.name == "mace-ef-cosmo-freesolv20-two-step-v3.json"
    assert runner.ARTIFACT_ID == "mace-ef-cosmo-freesolv20-two-step-v3"


def test_two_step_converged_comparator_requires_exact_continuum_identity():
    runner = _load_runner()
    continuum = SimpleNamespace(config=SimpleNamespace(configuration_sha256="a" * 64))
    primary_record = {"mace_ef_surface": {"continuum_configuration_sha256": "a" * 64}}
    assert (
        runner._require_matching_continuum_config(continuum, primary_record) == "a" * 64
    )
    primary_record["mace_ef_surface"]["continuum_configuration_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="continuum configurations differ"):
        runner._require_matching_continuum_config(continuum, primary_record)
