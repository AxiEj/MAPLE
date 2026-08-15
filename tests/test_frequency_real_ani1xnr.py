from __future__ import annotations

import os
from pathlib import Path

import pytest
from ase.io import read

from maple.function.calculator.ani._ani_calculator import ANICalculator
from maple.function.dispatcher.frequency.frequency import Frequency
from maple.function.dispatcher.irc.preflight import (
    IRCPreflightParams,
    validate_irc_transition_state,
)
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView

CHECKPOINT_ENV = "MAPLE_REAL_ANI1XNR_CHECKPOINT"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not os.environ.get(CHECKPOINT_ENV),
    reason=f"set {CHECKPOINT_ENV} to run the real ANI-1xnr TS FREQ canary",
)
def test_real_ani1xnr_first_order_saddle_frequency_contract(tmp_path):
    checkpoint = Path(os.environ[CHECKPOINT_ENV]).expanduser().resolve()
    if not checkpoint.is_file():
        pytest.fail(f"{CHECKPOINT_ENV} does not name a file: {checkpoint}")

    atoms = read(REPOSITORY_ROOT / "examples/ts/neb/inp1_nebts_ts.xyz")
    atoms.calc = ANICalculator(
        device="cpu",
        model="ani1xnr",
        model_path=str(checkpoint),
    )
    output = tmp_path / "ani1xnr-ts-freq.out"
    driver = Frequency(
        str(output),
        atoms,
        paras={
            "stationary_point": "transition_state",
            "transition_state_imaginary_threshold_cm1": 50.0,
            "verbosity": 10,
        },
    )

    driver.run()

    report = output.read_text(encoding="utf-8")
    summary = output.with_suffix(".sum").read_text(encoding="utf-8")
    assert "Requested stationary point: first-order transition state" in report
    assert report.count("***imaginary mode***") == 1
    assert "Thermochemistry: withheld" in report
    assert "THERMOCHEMISTRY AT" not in report
    assert "G_corr (total)" not in summary


@pytest.mark.skipif(
    not os.environ.get(CHECKPOINT_ENV),
    reason=f"set {CHECKPOINT_ENV} to run the real ANI-1xnr IRC preflight canary",
)
def test_real_ani1xnr_irc_preflight_preserves_the_legacy_unit_boundary():
    checkpoint = Path(os.environ[CHECKPOINT_ENV]).expanduser().resolve()
    if not checkpoint.is_file():
        pytest.fail(f"{CHECKPOINT_ENV} does not name a file: {checkpoint}")

    atoms = read(REPOSITORY_ROOT / "examples/ts/neb/inp1_nebts_ts.xyz")
    raw_calculator = ANICalculator(
        device="cpu",
        model="ani1xnr",
        model_path=str(checkpoint),
    )
    legacy_calculator = LegacyHartreeJobView(raw_calculator)

    result = validate_irc_transition_state(
        atoms,
        legacy_calculator.get_hessian(atoms),
        legacy_calculator.get_forces(atoms),
        IRCPreflightParams(),
    )

    assert result.assessment.imaginary_frequency_cm1 == pytest.approx(
        -2129.29,
        abs=2.0,
    )
    assert result.maximum_force_eV_per_A < 1.0e-3
    assert result.rigid_residual_cm1 < 5.0
