from __future__ import annotations

import pytest

from maple.solvation.release.box_convergence import summarize_box_convergence


def _record(box, energy, continuum, force):
    return {
        "box_length_A": box,
        "total_energy_eV": energy,
        "continuum_energy_eV": continuum,
        "forces_eV_per_A": [[force, 0.0, 0.0], [-force, 0.0, 0.0]],
        "source": [[0.1, 0.0], [-0.1, 0.0]],
        "primal_residual": 1.0e-13,
        "adjoint_residual": 1.0e-15,
        "profile_id": f"profile-{box}",
        "model_profile_id": f"model-{box}",
        "model_configuration_sha256": f"{box:064x}",
    }


def test_box_convergence_uses_largest_box_as_reference_and_passes_small_tail():
    result = summarize_box_convergence(
        (
            _record(32, -10.00020, -0.02020, 0.10020),
            _record(40, -10.00005, -0.02005, 0.10005),
            _record(48, -10.00001, -0.02001, 0.10001),
            _record(56, -10.00000, -0.02000, 0.10000),
        )
    )
    assert result["reference_box_length_A"] == 56
    assert result["tail_pair"] == [48, 56]
    assert result["tail"]["total_energy_abs_eV"] == pytest.approx(1.0e-5)
    assert result["tail"]["force_rms_eV_per_A"] == pytest.approx(1.0e-5 / 3**0.5)
    assert result["all_gates_passed"] is True
    assert len({item["profile_id"] for item in result["records"]}) == 4


def test_box_convergence_rejects_unregistered_duplicate_or_spoofed_identities():
    with pytest.raises(ValueError, match="exactly the preregistered"):
        summarize_box_convergence((_record(40, 0, 0, 0), _record(56, 0, 0, 0)))
    records = [_record(box, 0, 0, 0) for box in (32, 40, 48, 56)]
    records[2]["profile_id"] = records[1]["profile_id"]
    with pytest.raises(ValueError, match="distinct profile"):
        summarize_box_convergence(records)


def test_box_convergence_reports_failed_tail_without_changing_thresholds():
    result = summarize_box_convergence(
        tuple(
            _record(box, energy, energy / 10, force)
            for box, energy, force in (
                (32, 0.1, 0.1),
                (40, 0.05, 0.05),
                (48, 0.01, 0.01),
                (56, 0.0, 0.0),
            )
        )
    )
    assert result["gates"]["tail_total_energy_abs_le_1e-4_eV"] is False
    assert result["gates"]["tail_force_rms_le_1e-4_eV_per_A"] is False
    assert result["all_gates_passed"] is False


def test_box_convergence_rejects_nonfinite_shapes_and_loose_residuals():
    records = [_record(box, 0, 0, 0) for box in (32, 40, 48, 56)]
    records[0]["forces_eV_per_A"] = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="forces"):
        summarize_box_convergence(records)
    records = [_record(box, 0, 0, 0) for box in (32, 40, 48, 56)]
    records[0]["primal_residual"] = 1.0e-8
    with pytest.raises(ValueError, match="primal residual"):
        summarize_box_convergence(records)
    records = [_record(box, 0, 0, 0) for box in (32, 40, 48, 56)]
    records[0]["adjoint_residual"] = 1.0e-8
    with pytest.raises(ValueError, match="adjoint residual"):
        summarize_box_convergence(records)
