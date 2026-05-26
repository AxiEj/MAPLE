"""WS3 — unit-layer checks of the acceptance-matrix harness (fast, fake/cheap)."""

import numpy as np
import pytest
from ase.calculators.lj import LennardJones

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT, EV2HARTREE
from maple.function.dispatcher.md.validation import (
    MapleLJReferenceCalculator,
    _block_mean_stderr,
    _lj_crystal,
    load_thresholds,
    run_stress_finite_difference,
    run_constraints_rejected,
    run_pbc_geometry,
    run_restart_determinism,
    write_report,
    lj_reference_factory,
    validation_system_summary,
)


def test_thresholds_are_versioned_and_complete():
    th = load_thresholds()
    assert "thresholds_version" in th
    for section in (
        "nve_energy_drift", "restart_determinism", "nvt_mean_temperature",
        "npt_pressure", "npt_volume_fluctuation", "npt_effective_energy_drift",
        "stress_finite_difference", "pbc_geometry", "constraints",
    ):
        assert section in th
    # The NVT gate uses the standard error of the mean (block averaging with an
    # i.i.d. floor), not the instantaneous spread; its block count is registered.
    assert th["nvt_mean_temperature"]["n_blocks"] >= 2
    assert 0.0 < th["nvt_mean_temperature"]["equilibration_fraction"] < 1.0


def test_lj_reference_calculator_honours_maple_unit_contract():
    crystal = _lj_crystal()
    crystal.calc = MapleLJReferenceCalculator()
    energy_ha = crystal.get_potential_energy()
    stress = crystal.get_stress()

    raw = _lj_crystal()
    raw.calc = LennardJones(epsilon=0.0103, sigma=3.40, rc=6.5, smooth=True)
    # Energy converted eV -> Ha; stress passed through unchanged in eV/Å³.
    assert energy_ha == pytest.approx(raw.get_potential_energy() * EV2HARTREE)
    np.testing.assert_allclose(stress, raw.get_stress())

    assert crystal.calc.maple_pbc_md_supported is True
    assert crystal.calc.maple_stress_supported is True
    assert crystal.calc.maple_stress_unit == ASE_STRESS_UNIT


def test_real_backend_validation_system_avoids_argon_species_gate():
    class RealBackendLikeCalc:
        maple_model_name = "aimnet2-pbc"

    summary = validation_system_summary(lambda: RealBackendLikeCalc())
    assert summary["dynamics"]["formula"] == "C8O16"
    assert summary["dynamics"]["n_atoms"] == 24
    assert summary["dynamics"]["pbc"] == [True, True, True]
    assert min(np.linalg.norm(row) for row in summary["dynamics"]["cell_A"]) > 12.0
    assert summary["stress_finite_difference"]["formula"] == "H16O8"
    assert summary["stress_finite_difference"]["n_atoms"] == 24


def test_pbc_geometry_class_passes(tmp_path):
    th = load_thresholds()
    result = run_pbc_geometry(lj_reference_factory(), th, tmp_path)
    assert result.passed and result.status == "pass"


def test_stress_finite_difference_reports_full_voigt_components(tmp_path):
    th = load_thresholds()
    result = run_stress_finite_difference(lj_reference_factory(), th, tmp_path)
    assert result.passed and result.status == "pass"
    components = result.metrics["components"]
    assert [item["component"] for item in components] == ["xx", "yy", "zz", "yz", "xz", "xy"]
    assert any(abs(item["stress_ev_per_ang3"]) > 1e-6 for item in components if item["component"] in {"xy", "xz", "yz"})
    for item in components:
        for delta in item["per_delta"]:
            assert "abs_error_ev_per_ang3" in delta
            assert "rel_error" in delta
            assert "log10_abs_ratio_error" in delta


def test_constraints_class_passes(tmp_path):
    th = load_thresholds()
    result = run_constraints_rejected(lj_reference_factory(), th, tmp_path)
    assert result.passed
    assert result.metrics["rejected"] is True


def test_restart_determinism_accepts_relative_workdir(tmp_path, monkeypatch):
    th = load_thresholds()
    monkeypatch.chdir(tmp_path)
    result = run_restart_determinism(lj_reference_factory(), th, "relative-runs")
    assert result.passed and result.status == "pass"


def test_block_mean_stderr_matches_iid_for_independent_series():
    rng = np.random.default_rng(0)
    n = 2000
    iid = rng.standard_normal(n)
    block = _block_mean_stderr(iid, 10)
    analytic = float(np.std(iid) / np.sqrt(n))
    # For independent samples the block SE and the i.i.d. SE agree to within the
    # noise of a 10-block estimate (same order of magnitude).
    assert block == pytest.approx(analytic, rel=0.6)


def test_block_mean_stderr_exceeds_iid_for_autocorrelated_series():
    rng = np.random.default_rng(1)
    n = 4000
    phi = 0.95  # AR(1): integrated autocorrelation time ~ (1+phi)/(1-phi) ~ 39
    ar = np.empty(n)
    ar[0] = rng.standard_normal()
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + rng.standard_normal()
    block = _block_mean_stderr(ar, 10)
    iid = float(np.std(ar) / np.sqrt(n))
    # The autocorrelation-aware block SE must be well above the naive i.i.d. one,
    # which is the whole reason the NVT gate cannot use sqrt(n).
    assert block > 3.0 * iid


def test_block_mean_stderr_nan_with_too_few_blocks():
    assert np.isnan(_block_mean_stderr(np.arange(3.0), 1))


def test_write_report_emits_markdown_and_json(tmp_path):
    th = load_thresholds()
    results = [run_pbc_geometry(lj_reference_factory(), th, tmp_path)]
    md_path = write_report(results, th, tmp_path / "reports")
    assert md_path.exists() and md_path.suffix == ".md"
    json_path = md_path.with_suffix(".json")
    assert json_path.exists()
    text = md_path.read_text()
    assert "MD acceptance matrix" in text
    assert th["thresholds_version"] in text
    payload = __import__("json").loads(json_path.read_text())
    assert payload["markdown_report_path"] == str(md_path)
    assert md_path.name == "report.md"
    assert md_path.parent.name == payload["artifact_id"]
    assert payload["run_workdir"] is None
    assert payload["manifest_files"] == []
    assert len(payload["markdown_report_sha256"]) == 64
