from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
    MACEPolarEFConfig,
)
from maple.function.calculator.extra_correction.implicit.mace_polar_ef_stationary import (
    MACEPolarEFPassivityError,
    MACEPolarEFSCFSettings,
)
from maple.function.calculator.extra_correction.implicit.mace_polar_ef_smooth_pcm import (
    MACE_POLAR_EF_SMOOTH_PCM_CONTRACT_ID,
    MACEPolarEFSmoothPCMConfig,
    MACEPolarEFSmoothPCMCoupling,
)
from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import (
    TorchSmoothPCM,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "macepol-ef-v2.pt"
POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]],
    dtype=float,
)


def _coupling() -> MACEPolarEFSmoothPCMCoupling:
    if not CHECKPOINT.is_file() or not torch.cuda.is_available():
        pytest.skip("local MACE-POLAR-EF v2 CUDA checkpoint is unavailable")
    continuum = TorchSmoothPCM(
        atomic_numbers=(8, 1, 1),
        radii_angstrom=(1.52, 1.20, 1.20),
        transition_width_angstrom2=0.08,
        surface_lmax=2,
        partition_lmax=4,
        partition_radial_quadrature_order=32,
        source_radial_quadrature_order=48,
        double_layer_radial_quadrature_order=48,
        dielectric=78.355,
    )
    return MACEPolarEFSmoothPCMCoupling(
        MACEPolarEFSmoothPCMConfig(
            electronic=MACEPolarEFConfig(
                checkpoint_path=str(CHECKPOINT),
                atomic_numbers=(8, 1, 1),
                spin_multiplicity=1,
            ),
            continuum=continuum,
            scf=MACEPolarEFSCFSettings(
                source_residual_tolerance=5.0e-5,
                maximum_iterations=100,
                mixing=0.5,
            ),
        )
    )


def test_smooth_pcm_coupling_has_a_separate_private_identity():
    coupling = _coupling()
    identity = coupling.config.as_dict()

    assert identity["contract_id"] == MACE_POLAR_EF_SMOOTH_PCM_CONTRACT_ID
    continuum = cast(dict[str, object], identity["continuum"])
    assert continuum["model_id"] == "torch-smooth-pcm-v1"
    assert continuum["smooth_partition"] is True
    assert continuum["fixed_dimensions"] is True
    assert identity["publicly_registered"] is False
    assert identity["release_admitted"] is False


def test_smooth_pcm_coupling_fails_passivity_before_pcm_iteration(monkeypatch):
    coupling = _coupling()
    monkeypatch.setattr(
        coupling,
        "_solve_source",
        lambda _positions: pytest.fail(
            "smooth PCM iteration must not run after failed electronic passivity"
        ),
    )

    with pytest.raises(MACEPolarEFPassivityError) as caught:
        coupling.evaluate(POSITIONS)

    assert caught.value.evidence["passivity_passed"] is False
    assert cast(float, caught.value.evidence["maximum_positive_curvature"]) > 1.0


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


def test_test_only_bypass_closes_the_smooth_stationary_equations(
    diagnostic_result,
):
    coupling, result = diagnostic_result

    assert result.maximum_source_residual <= (
        coupling.config.scf.source_residual_tolerance
    )
    assert result.maximum_field_replay_difference <= 2.0e-9
    assert result.continuum_energy_ev == pytest.approx(
        0.5 * result.source_field_pairing_ev,
        abs=2.0e-9,
    )
    assert result.provenance["electronic_passivity"] == {
        "passivity_passed": False,
        "test_only_bypass": True,
    }
    assert result.provenance["result_interpretation"] == (
        "blocked-stationary-common-scalar-first-order-diagnostic"
    )


def test_smooth_coupled_coordinate_gradient_is_finite_and_translation_closed(
    diagnostic_result,
):
    _coupling_value, result = diagnostic_result

    assert result.coordinate_gradient_ev_per_angstrom.shape == (3, 3)
    assert np.all(np.isfinite(result.coordinate_gradient_ev_per_angstrom))
    np.testing.assert_allclose(
        np.sum(result.coordinate_gradient_ev_per_angstrom, axis=0),
        0.0,
        atol=3.0e-5,
        rtol=0.0,
    )


def test_smooth_pcm_and_electronic_atom_order_must_match():
    if not CHECKPOINT.is_file():
        pytest.skip("local MACE-POLAR-EF v2 checkpoint is unavailable")
    continuum = TorchSmoothPCM(
        atomic_numbers=(1, 8, 1),
        radii_angstrom=(1.20, 1.52, 1.20),
        surface_lmax=1,
        partition_lmax=2,
        partition_radial_quadrature_order=16,
        source_radial_quadrature_order=24,
        double_layer_radial_quadrature_order=24,
        dielectric=78.355,
    )

    with pytest.raises(ValueError, match="atomic-number order"):
        MACEPolarEFSmoothPCMConfig(
            electronic=MACEPolarEFConfig(
                checkpoint_path=str(CHECKPOINT),
                atomic_numbers=(8, 1, 1),
            ),
            continuum=continuum,
        )
