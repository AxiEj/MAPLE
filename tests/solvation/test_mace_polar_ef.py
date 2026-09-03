from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
    MACE_POLAR_EF_AUTOGRAD_ORDER,
    MACE_POLAR_EF_CHECKPOINT_SHA256,
    MACE_POLAR_EF_MODEL_ID,
    MACEPolarEFConfig,
    MACEPolarEFEnergyModel,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "macepol-ef-v2.pt"
POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]],
    dtype=np.float32,
)


@pytest.fixture(scope="module")
def evaluator():
    if not CHECKPOINT.is_file() or not torch.cuda.is_available():
        pytest.skip("local MACE-POLAR-EF v2 CUDA checkpoint is unavailable")
    config = MACEPolarEFConfig(
        checkpoint_path=str(CHECKPOINT),
        atomic_numbers=(8, 1, 1),
    )
    return MACEPolarEFEnergyModel(config)


def test_checkpoint_identity_and_private_boundary(evaluator):
    provenance = evaluator.config.as_provenance()
    assert provenance["model_id"] == MACE_POLAR_EF_MODEL_ID
    assert provenance["checkpoint_sha256"] == MACE_POLAR_EF_CHECKPOINT_SHA256
    assert provenance["autograd_order"] == MACE_POLAR_EF_AUTOGRAD_ORDER == 1
    assert provenance["density_coefficients_role"] == "diagnostic-only"
    assert provenance["spin_multiplicity"] == 1
    assert provenance["checkpoint_spin_input_semantics"] == "multiplicity (singlet=1)"
    assert provenance["publicly_registered"] is False
    assert provenance["release_admitted"] is False


def test_energy_conjugate_source_not_density_head(evaluator):
    field = np.zeros((3, 4), dtype=np.float32)
    result = evaluator.evaluate(POSITIONS, field)
    assert np.isfinite(result.energy_ev)
    np.testing.assert_allclose(
        result.conjugate_source_raw[:, 0],
        result.density_coefficients_diagnostic[:, 0],
        atol=2.0e-7,
        rtol=0.0,
    )
    assert np.max(np.abs(result.conjugate_source_raw[:, 1:])) > 1.0e-3
    assert np.max(np.abs(result.density_coefficients_diagnostic[:, 1:])) == 0.0
    assert abs(np.sum(result.conjugate_source_raw[:, 0])) < 1.0e-6


def test_field_gradient_matches_centered_energy_difference(evaluator):
    positions = torch.tensor(
        POSITIONS, device="cuda", dtype=torch.float32, requires_grad=True
    )
    field = torch.zeros((3, 4), device="cuda", dtype=torch.float32)
    field[:, 1:] = (
        torch.tensor(
            [[1.0, -0.3, 0.5], [-0.4, 0.8, -0.2], [0.7, 0.2, -0.6]],
            device="cuda",
        )
        * 1.0e-3
    )
    field.requires_grad_(True)
    energy, source_raw, _density = evaluator.conjugate_source_torch(positions, field)

    direction = torch.tensor(
        [
            [0.2, 0.1, -0.2, 0.3],
            [-0.1, -0.2, 0.1, 0.2],
            [-0.1, 0.1, 0.1, -0.5],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    direction /= torch.linalg.vector_norm(direction)
    raw_direction = evaluator._raw_source_from_cartesian_gradient(direction)
    analytic = torch.sum(source_raw * raw_direction)
    step = 2.0e-2
    plus = evaluator.energy_torch(positions, field + step * direction)[0]
    minus = evaluator.energy_torch(positions, field - step * direction)[0]
    finite = (plus - minus) / (2.0 * step)
    assert float(analytic.detach()) == pytest.approx(float(finite.detach()), abs=1.5e-2)
    assert energy.requires_grad


def test_second_order_autograd_is_rejected_by_supported_adapter(evaluator):
    positions = torch.tensor(POSITIONS, device="cuda", dtype=torch.float32)
    field = torch.zeros((3, 4), device="cuda", dtype=torch.float32, requires_grad=True)

    with pytest.raises(NotImplementedError, match="first derivatives only"):
        evaluator.conjugate_source_torch(
            positions,
            field,
            create_graph=True,
        )


def test_native_external_drive_preserves_independent_atomwise_rows(evaluator):
    field_a = np.asarray(
        [
            [0.02, 0.08, -0.03, 0.01],
            [-0.01, -0.02, 0.07, -0.04],
            [-0.01, -0.06, -0.04, 0.03],
        ],
        dtype=np.float32,
    )
    field_b = np.repeat(
        np.mean(field_a, axis=0, keepdims=True),
        repeats=3,
        axis=0,
    )

    atomwise = evaluator.evaluate(POSITIONS, field_a)
    collapsed = evaluator.evaluate(POSITIONS, field_b)

    assert np.max(np.abs(atomwise.field_cartesian - field_b)) > 1.0e-2
    assert atomwise.energy_ev != collapsed.energy_ev
    assert (
        np.max(np.abs(atomwise.conjugate_source_raw - collapsed.conjugate_source_raw))
        > 1.0e-4
    )


def test_atomwise_scalar_potential_is_explicit_energy_coupling_not_response(evaluator):
    zero = evaluator.evaluate(POSITIONS, np.zeros((3, 4), dtype=np.float32))
    field = np.zeros((3, 4), dtype=np.float32)
    field[:, 0] = np.asarray([0.08, -0.03, -0.05], dtype=np.float32)
    driven = evaluator.evaluate(POSITIONS, field)

    np.testing.assert_allclose(
        driven.density_coefficients_diagnostic,
        zero.density_coefficients_diagnostic,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        driven.conjugate_source_raw[:, 0],
        driven.density_coefficients_diagnostic[:, 0],
        atol=2.0e-7,
        rtol=0.0,
    )


def test_uniform_affine_field_passivity_audit_fails_for_supplied_v2(evaluator):
    for field_step in (5.0e-4, 1.0e-3, 2.0e-3):
        audit = evaluator.audit_uniform_field_passivity(
            POSITIONS,
            field_step_ev_per_e_angstrom=field_step,
        )

        assert audit.passivity_passed is False
        assert audit.maximum_positive_curvature > audit.tolerance
        assert audit.maximum_positive_curvature > 1.0
        assert audit.provenance["external_drive"] == (
            "per-atom V_i=g dot (R_i-centroid), grad(V)_i=g"
        )
        assert not audit.energy_hessian.flags.writeable
        assert not audit.energy_hessian_eigenvalues.flags.writeable


def test_coordinate_gradient_is_finite_and_translation_closed(evaluator):
    result = evaluator.evaluate(POSITIONS, np.zeros((3, 4), dtype=np.float32))
    assert np.all(np.isfinite(result.coordinate_gradient_ev_per_angstrom))
    np.testing.assert_allclose(
        np.sum(result.coordinate_gradient_ev_per_angstrom, axis=0),
        0.0,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_cutoff_margin_fails_closed(evaluator):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [5.98, 0.0, 0.0], [0.0, 0.9, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    field = torch.zeros((3, 4), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="cutoff topology margin"):
        evaluator.energy_torch(positions, field)


def test_spin_input_is_multiplicity_and_parity_checked():
    if not CHECKPOINT.is_file():
        pytest.skip("local MACE-POLAR-EF v2 checkpoint is unavailable")

    with pytest.raises(ValueError, match="spin_multiplicity must be at least one"):
        MACEPolarEFConfig(
            checkpoint_path=str(CHECKPOINT),
            atomic_numbers=(8, 1, 1),
            spin_multiplicity=0,
        )
    with pytest.raises(ValueError, match="electron-count parity"):
        MACEPolarEFConfig(
            checkpoint_path=str(CHECKPOINT),
            atomic_numbers=(8, 1, 1),
            spin_multiplicity=2,
        )
    with pytest.raises(ValueError, match="embeds CUDA device 0"):
        MACEPolarEFConfig(
            checkpoint_path=str(CHECKPOINT),
            atomic_numbers=(8, 1, 1),
            device="cuda:1",
        )
