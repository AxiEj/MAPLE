"""WS3 production layer — non-skippable acceptance matrix (run with `-m production`).

Runs the full PBC-MD acceptance matrix against the backend-free Lennard-Jones
reference calculator and emits a dated report artifact. Excluded from the default
run; invoke with `pytest tests/production -m production` or
`python scripts/production_validation.py`.
"""

import pytest

from maple.function.dispatcher.md.validation import (
    lj_reference_factory,
    load_thresholds,
    run_acceptance_matrix,
    write_report,
)

pytestmark = pytest.mark.production


def test_full_lj_acceptance_matrix_passes(tmp_path):
    thresholds = load_thresholds()
    results = run_acceptance_matrix(lj_reference_factory(), thresholds, tmp_path, quick=False)
    report = write_report(results, thresholds, tmp_path / "reports", calculator_label="lj-reference")

    assert report.exists()
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"acceptance classes failed: {failed}"
    # Non-skippable contract: every class must genuinely PASS. A "skip"
    # (inconclusive, e.g. insufficient NPT volume signal) is not a pass and must
    # not let the release matrix green on an unvalidated class.
    skipped = [r.name for r in results if r.status == "skip"]
    assert not skipped, f"acceptance classes inconclusive (skipped): {skipped}"
    # Every declared class ran.
    assert {r.name for r in results} == {
        "nve_energy_drift", "restart_determinism", "nvt_mean_temperature",
        "npt_pressure", "npt_volume_fluctuation", "stress_finite_difference",
        "pbc_geometry", "constraints_rejected",
    }
