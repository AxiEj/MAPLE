"""WS3 — fast smoke of the NPT volume-fluctuation acceptance class.

Exercises the EOS-slope vs fluctuation-identity compressibility path in the
default test layer with a tiny step override (the full statistical run lives in
the slow/opt-in production acceptance matrix).  It asserts the class runs end to
end and returns finite kappa_T metrics, not that the short run passes the
pre-registered consistency tolerance.
"""

import numpy as np

from maple.function.dispatcher.md.validation import (
    lj_reference_factory,
    load_smoke_thresholds,
    run_npt_volume_fluctuation,
)


def test_npt_volume_fluctuation_runs_and_reports_metrics(tmp_path):
    thresholds = load_smoke_thresholds()
    result = run_npt_volume_fluctuation(
        lj_reference_factory(), thresholds, tmp_path, steps=200, timestep=1.0, temperature=100.0
    )

    assert result.name == "npt_volume_fluctuation"
    assert result.status in {"pass", "fail", "skip"}

    metrics = result.metrics
    for key in (
        "kappa_fluct_per_bar", "kappa_eos_per_bar", "mean_V1_A3", "mean_V2_A3",
        "var_V1_A6", "var_V1_block_se_A6", "n_samples_post_eq", "rel_volume_change",
        "barostat_stride_NP", "volume_drift_sigma_P1", "volume_drift_sigma_P2",
        "max_volume_drift_sigma",
    ):
        assert key in metrics, f"missing reported metric: {key}"

    # Fluctuation kappa_T is always computed and is a non-negative, finite number.
    assert np.isfinite(metrics["kappa_fluct_per_bar"]) and metrics["kappa_fluct_per_bar"] >= 0.0
    assert np.isfinite(metrics["kappa_eos_per_bar"])
    assert metrics["mean_V1_A3"] > 0.0 and metrics["mean_V2_A3"] > 0.0
    assert metrics["n_samples_post_eq"] > 0


def test_npt_volume_fluctuation_thresholds_present():
    th = load_smoke_thresholds()["npt_volume_fluctuation"]
    assert th["log10_kappa_tol"] > 0.0
    assert len(th["pressures_bar"]) == 2 and th["pressures_bar"][1] > th["pressures_bar"][0]
    assert 0.0 < th["equilibration_fraction"] < 1.0
    assert th["min_volume_change"] < th["max_volume_change"]
    assert th["barostat_stride"] > 1
    assert th["max_volume_drift_sigma"] > 0.0
