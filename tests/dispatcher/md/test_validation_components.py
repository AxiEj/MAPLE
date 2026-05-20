"""WS3 — unit-layer checks of the acceptance-matrix harness (fast, fake/cheap)."""

import numpy as np
import pytest
from ase.calculators.lj import LennardJones

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT, EV2HARTREE
from maple.function.dispatcher.md.validation import (
    MapleLJReferenceCalculator,
    _lj_crystal,
    load_thresholds,
    run_constraints_rejected,
    run_pbc_geometry,
    write_report,
    lj_reference_factory,
)


def test_thresholds_are_versioned_and_complete():
    th = load_thresholds()
    assert "thresholds_version" in th
    for section in (
        "nve_energy_drift", "restart_determinism", "nvt_mean_temperature",
        "npt_pressure", "stress_finite_difference", "pbc_geometry", "constraints",
    ):
        assert section in th


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


def test_pbc_geometry_class_passes(tmp_path):
    th = load_thresholds()
    result = run_pbc_geometry(lj_reference_factory(), th, tmp_path)
    assert result.passed and result.status == "pass"


def test_constraints_class_passes(tmp_path):
    th = load_thresholds()
    result = run_constraints_rejected(lj_reference_factory(), th, tmp_path)
    assert result.passed
    assert result.metrics["rejected"] is True


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
