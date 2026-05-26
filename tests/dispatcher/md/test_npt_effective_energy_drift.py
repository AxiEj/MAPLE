"""WS7 — fast smoke of the NPT effective-energy-drift acceptance class.

Exercises the reversible c-rescale effective-energy diagnostic path (H̃ = K + U +
P_0·V − Σ ΔW_ext) in the default test layer with a small step override; the full
statistical run lives in the opt-in production acceptance matrix.  It asserts the
class runs end to end, that the H_cons diagnostic thermo column is wired through
the logger, and that the LJ reference — a conservative potential — keeps the
effective-energy drift far below the gate even on a short run.
"""

import numpy as np

from maple.function.dispatcher.md.validation import (
    lj_reference_factory,
    load_smoke_thresholds,
    run_npt_effective_energy_drift,
)


def test_npt_effective_energy_drift_runs_and_reports_metrics(tmp_path):
    thresholds = load_smoke_thresholds()
    result = run_npt_effective_energy_drift(
        lj_reference_factory(), thresholds, tmp_path, steps=800, timestep=1.0, temperature=100.0
    )

    assert result.name == "npt_effective_energy_drift"
    assert result.status in {"pass", "fail"}
    for key in ("h_cons_drift_ha_per_atom_per_ps", "h_cons_range_ha", "n_atoms", "fit_window_ps"):
        assert key in result.metrics, f"missing reported metric: {key}"

    # The effective-energy diagnostic column must have been wired through (a
    # missing H_cons column is reported as an explicit failure, never silently
    # treated as zero).
    assert "error" not in result.metrics, result.detail
    # A conservative potential keeps H̃ essentially flat: the drift sits orders of
    # magnitude under the gate, and the absolute H̃ range is tiny.
    assert np.isfinite(result.metrics["h_cons_drift_ha_per_atom_per_ps"])
    assert result.metrics["h_cons_drift_ha_per_atom_per_ps"] < 1e-7
    assert result.metrics["h_cons_range_ha"] < 1e-4
    assert result.passed


def test_npt_effective_energy_drift_threshold_present():
    th = load_smoke_thresholds()["npt_effective_energy_drift"]
    assert th["max_abs_drift_ha_per_atom_per_ps"] > 0.0
