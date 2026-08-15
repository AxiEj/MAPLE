from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from maple.solvation.release.geometry_mediated_adaptive import (
    GEOMETRY_MEDIATED_ADAPTIVE_FD_MAXITER,
    GEOMETRY_MEDIATED_ADAPTIVE_FD_ORDER,
    capture_geometry_mediated_adaptive_directional_samples,
    summarize_geometry_mediated_adaptive_directional_audit,
)

_MODEL_HASH = "1" * 64
_CONTINUUM_HASH = "2" * 64


def _positions() -> np.ndarray:
    return np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])


def _direction() -> np.ndarray:
    direction = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    return direction / np.linalg.norm(direction)


def _model_topology(*, digest: str = _MODEL_HASH, margin: float = 1.0):
    return {
        "topology_sha256": digest,
        "minimum_cutoff_margin_angstrom": margin,
    }


def _continuum_topology(*, digest: str = _CONTINUUM_HASH, clearance: float = 1.0):
    return {
        "cavity_topology_sha256": digest,
        "cavity_active_node_count": 12,
        "minimum_cavity_active_set_clearance_angstrom": clearance,
    }


def _capture(function):
    def sample(displacement_A: float):
        return {
            "energy_eV": float(function(displacement_A)),
            "positions_A": (_positions() + displacement_A * _direction()).tolist(),
            "model_topology": _model_topology(),
            "continuum_topology": _continuum_topology(),
        }

    return capture_geometry_mediated_adaptive_directional_samples(sample)


def _summary(raw):
    direction = _direction()
    return summarize_geometry_mediated_adaptive_directional_audit(
        analytic_gradient_eV_per_A=2.0 * direction,
        direction=direction,
        center_positions_A=_positions(),
        center_model_topology=_model_topology(),
        center_continuum_topology=_continuum_topology(),
        adaptive_record=raw,
        reciprocity_audit={"gate_passed": True},
    )


def test_adaptive_directional_audit_replays_scipy_from_raw_samples():
    raw = _capture(lambda displacement: -100.0 + 2.0 * displacement + displacement**3)
    summary = _summary(raw)

    assert raw["protocol"]["order"] == GEOMETRY_MEDIATED_ADAPTIVE_FD_ORDER
    assert raw["protocol"]["maxiter"] == GEOMETRY_MEDIATED_ADAPTIVE_FD_MAXITER
    assert raw["scipy_result"]["status"] == 0
    assert raw["scipy_result"]["success"] is True
    assert raw["scipy_result"]["nfev"] == len(raw["samples"])
    assert summary["scipy_result_recomputed"] is True
    assert summary["adaptive_convergence_gate_passed"] is True
    assert summary["analytic_agreement_gate_passed"] is True
    assert summary["topology"]["all_samples_same_stratum"] is True
    assert summary["topology"]["all_segment_guards_passed"] is True
    assert summary["gate_passed"] is True


def test_adaptive_directional_audit_rejects_missing_extra_and_tampered_samples():
    raw = _capture(lambda displacement: 3.0 + 2.0 * displacement + displacement**3)

    missing = deepcopy(raw)
    missing["samples"].pop()
    with pytest.raises(ValueError, match="unrecorded displacement"):
        _summary(missing)

    extra = deepcopy(raw)
    extra["samples"].append(
        {
            "displacement_A": 0.123,
            "displacement_hex": float(0.123).hex(),
            "energy_eV": 1.0,
            "positions_A": (_positions() + 0.123 * _direction()).tolist(),
            "model_topology": _model_topology(),
            "continuum_topology": _continuum_topology(),
        }
    )
    with pytest.raises(ValueError, match="unrequested displacement"):
        _summary(extra)

    tampered = deepcopy(raw)
    tampered["samples"][1]["energy_eV"] += 1.0e-8
    with pytest.raises(ValueError, match="SciPy result disagrees"):
        _summary(tampered)


def test_adaptive_directional_audit_fails_closed_on_topology_or_event_clearance():
    raw = _capture(lambda displacement: 3.0 + 2.0 * displacement + displacement**3)

    changed = deepcopy(raw)
    changed["samples"][1]["continuum_topology"]["cavity_topology_sha256"] = "3" * 64
    changed_summary = _summary(changed)
    assert changed_summary["topology"]["all_samples_same_stratum"] is False
    assert changed_summary["gate_passed"] is False

    near_event = deepcopy(raw)
    for sample in near_event["samples"]:
        sample["continuum_topology"][
            "minimum_cavity_active_set_clearance_angstrom"
        ] = 1.0e-6
    near_event_summary = _summary(near_event)
    assert near_event_summary["topology"]["all_continuum_event_guards_passed"] is False
    assert near_event_summary["topology"]["all_segment_guards_passed"] is False
    assert near_event_summary["gate_passed"] is False


@pytest.mark.parametrize(
    "function, expected_status",
    (
        (lambda x: 2.0 * x + 1.0e-3 * (x > 0.0), -2),
        (lambda x: 2.0 * x + 1.0e-4 * np.sin(3.0e5 * x + 0.3), -1),
    ),
)
def test_adaptive_directional_audit_retains_failed_scipy_status(
    function, expected_status
):
    raw = _capture(function)
    summary = _summary(raw)

    assert raw["scipy_result"]["status"] == expected_status
    assert raw["scipy_result"]["success"] is False
    assert summary["adaptive_convergence_gate_passed"] is False
    assert summary["gate_passed"] is False


def test_adaptive_directional_audit_requires_real_continuum_event_clearance():
    raw = _capture(lambda displacement: 3.0 + 2.0 * displacement + displacement**3)
    for sample in raw["samples"]:
        sample["continuum_topology"].pop("minimum_cavity_active_set_clearance_angstrom")
    summary = _summary(raw)

    assert summary["topology"]["continuum_event_guard_applicable"] is True
    assert summary["topology"]["all_continuum_event_margins_available"] is False
    assert summary["gate_passed"] is False
