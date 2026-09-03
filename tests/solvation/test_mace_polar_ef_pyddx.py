from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyddx")

from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
    MACEPolarEFConfig,
)
from maple.function.calculator.extra_correction.implicit.mace_polar_ef_pyddx import (
    MACE_POLAR_EF_PYDDX_CONTRACT_ID,
    MACEPolarEFPassivityError,
    MACEPolarEFPyDDXConfig,
    MACEPolarEFPyDDXCoupling,
)
from maple.function.calculator.extra_correction.implicit.mace_polar_ef_stationary import (
    MACEPolarEFSCFSettings,
)
from maple.function.calculator.extra_correction.implicit.torch_pyddx import (
    TorchPyDDXPCMConfig,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "macepol-ef-v2.pt"
POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]],
    dtype=float,
)


def _coupling():
    if not CHECKPOINT.is_file() or not torch.cuda.is_available():
        pytest.skip("local MACE-POLAR-EF v2 CUDA checkpoint is unavailable")
    electronic = MACEPolarEFConfig(
        checkpoint_path=str(CHECKPOINT),
        atomic_numbers=(8, 1, 1),
    )
    continuum = TorchPyDDXPCMConfig(
        radii_angstrom=(1.52, 1.20, 1.20),
        dielectric=78.355,
        lmax=3,
        n_lebedev=26,
        solver_tolerance=1.0e-12,
    )
    return MACEPolarEFPyDDXCoupling(
        MACEPolarEFPyDDXConfig(
            electronic=electronic,
            continuum=continuum,
            scf=MACEPolarEFSCFSettings(
                source_residual_tolerance=5.0e-5,
                maximum_iterations=80,
                mixing=0.5,
            ),
        )
    )


def test_coupling_fails_closed_before_continuum_solve(monkeypatch):
    coupling = _coupling()
    monkeypatch.setattr(
        coupling,
        "_solve_source",
        lambda _positions: pytest.fail(
            "ddPCM solve must not run after a failed electronic passivity gate"
        ),
    )

    with pytest.raises(MACEPolarEFPassivityError) as caught:
        coupling.evaluate(POSITIONS)

    evidence = caught.value.evidence
    assert evidence["passivity_passed"] is False
    assert cast(float, evidence["maximum_positive_curvature"]) > cast(
        float, evidence["tolerance"]
    )
    eigenvalues = np.asarray(
        cast(list[float], evidence["energy_hessian_eigenvalues"]),
        dtype=float,
    )
    assert float(np.max(eigenvalues)) > 1.0


@pytest.fixture(scope="module")
def diagnostic_result():
    coupling = _coupling()
    evidence = {
        "passivity_passed": False,
        "test_only_bypass": True,
    }
    with patch.object(
        coupling,
        "_require_electronic_passivity",
        return_value=evidence,
    ):
        result = coupling.evaluate(POSITIONS)
    return coupling, result


def test_stationary_common_scalar_algebra_remains_diagnostic(diagnostic_result):
    _coupling_value, result = diagnostic_result
    assert result.provenance["contract_id"] == MACE_POLAR_EF_PYDDX_CONTRACT_ID
    assert result.maximum_source_residual <= 5.0e-5
    assert result.maximum_field_replay_difference <= 1.0e-9
    assert result.iterations <= 80
    assert result.provenance["electronic_passivity"] == {
        "passivity_passed": False,
        "test_only_bypass": True,
    }
    assert result.provenance["p1_complete"] is False
    assert result.provenance["continuum_achieved_residual"] is None


def test_energy_ledger_removes_external_field_double_counting(diagnostic_result):
    _coupling_value, result = diagnostic_result
    assert result.continuum_energy_ev == pytest.approx(
        0.5 * result.source_field_pairing_ev,
        abs=1.0e-9,
    )
    assert result.total_energy_ev == pytest.approx(
        result.electronic_energy_ev
        + result.continuum_energy_ev
        - result.source_field_pairing_ev,
        abs=2.0e-7,
    )


def test_root_source_is_unprojected_energy_gradient(diagnostic_result):
    coupling, result = diagnostic_result
    assert abs(np.sum(result.source_raw[:, 0])) < (
        coupling.config.scf.charge_drift_tolerance
    )
    assert np.max(np.abs(result.source_raw[:, 1:])) > 1.0e-3
    assert np.max(np.abs(result.density_coefficients_diagnostic[:, 1:])) == 0.0
    assert np.all(np.isfinite(result.field_cartesian))
    validated = coupling._validated_source(result.source_raw)
    np.testing.assert_array_equal(validated, result.source_raw)


def test_stationary_envelope_coordinate_gradient_is_finite(diagnostic_result):
    _coupling_value, result = diagnostic_result
    assert result.coordinate_gradient_ev_per_angstrom.shape == (3, 3)
    assert np.all(np.isfinite(result.coordinate_gradient_ev_per_angstrom))
    np.testing.assert_allclose(
        np.sum(result.coordinate_gradient_ev_per_angstrom, axis=0),
        0.0,
        atol=2.0e-5,
        rtol=0.0,
    )


def test_stationary_envelope_gradient_matches_resolved_energy_difference():
    coupling = _coupling()
    evidence = {
        "passivity_passed": False,
        "test_only_bypass": True,
    }
    with patch.object(
        coupling,
        "_require_electronic_passivity",
        return_value=evidence,
    ):
        center = coupling.evaluate(POSITIONS)
        direction = np.asarray(
            [[0.3, -0.2, 0.1], [-0.4, 0.1, -0.2], [0.1, 0.1, 0.1]],
            dtype=float,
        )
        direction -= np.mean(direction, axis=0, keepdims=True)
        direction /= np.linalg.norm(direction)
        analytic = float(np.vdot(center.coordinate_gradient_ev_per_angstrom, direction))
        step = 1.0e-2
        plus = coupling.evaluate(POSITIONS + step * direction)
        minus = coupling.evaluate(POSITIONS - step * direction)
    finite = (plus.total_energy_ev - minus.total_energy_ev) / (2.0 * step)
    # This branch is intentionally reachable only through the test-only
    # passivity bypass above.  The approximately -2080 eV float32 electronic
    # energy plus a finite-residual root makes subtraction-based derivatives
    # a coarse algebra check, not force-admission evidence.
    assert analytic == pytest.approx(finite, abs=4.0e-2)


def test_public_capabilities_remain_closed(diagnostic_result):
    _coupling_value, result = diagnostic_result
    assert result.provenance["publicly_registered"] is False
    assert result.provenance["release_admitted"] is False
    assert result.provenance["autograd_order"] == 1
    assert result.provenance["hessian_available"] is False


def test_charge_drift_is_a_gate_not_a_projection():
    coupling = _coupling()
    source = np.zeros((3, 4), dtype=float)
    source[:, 0] = [0.2, -0.1, -0.1 + 1.0e-7]

    checked = coupling._validated_source(source)
    np.testing.assert_array_equal(checked, source)
    assert checked is not source

    source[0, 0] += 1.0e-2
    with pytest.raises(RuntimeError, match="total-charge gate"):
        coupling._validated_source(source)
