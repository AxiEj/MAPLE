from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import (
    TorchSmoothPCM,
    block_permutation,
)

from _torch_smooth_pcm_reference import P, W, central_difference


def _model(fixture):
    return TorchSmoothPCM(
        atomic_numbers=fixture.atomic_numbers,
        radii_angstrom=fixture.radii,
        transition_width_angstrom2=fixture.transition_width,
        surface_lmax=min(fixture.surface_lmax, 2),
        partition_lmax=min(fixture.partition_lmax, 4),
        partition_radial_quadrature_order=32,
        source_radial_quadrature_order=48,
        double_layer_radial_quadrature_order=48,
        dielectric=fixture.dielectric,
        source_shell_clearance_angstrom=0.05,
    )


def test_source_directional_derivative_matches_the_sealed_scalar():
    model = _model(P)
    direction = np.random.default_rng(2026090201).normal(size=P.source.shape)
    raw_gradient = (block_permutation(2) @ model.drive_cartesian(P.positions, P.source).ravel()).reshape(P.source.shape)
    analytic = float(np.vdot(raw_gradient, direction))
    errors = []
    for step in (2e-4, 2e-5, 2e-6, 2e-7):
        actual = central_difference(lambda source: model.energy(P.positions, source), P.source, direction, step)
        errors.append(abs(actual - analytic))
    # The fixed-geometry scalar is exactly quadratic in the source, so central
    # differences have no truncation term and are rounding-error limited.
    assert max(errors) < 3e-9
    assert errors[-1] < 3e-9


def test_source_jvp_is_the_cartesian_drive_jacobian():
    model = _model(P)
    direction = np.random.default_rng(2026090202).normal(size=P.source.shape)
    expected = model.source_jvp(P.positions, P.source, direction)
    step = 2e-6
    actual = (
        model.drive_cartesian(P.positions, P.source + step * direction)
        - model.drive_cartesian(P.positions, P.source - step * direction)
    ) / (2 * step)
    assert np.allclose(actual, expected, atol=3e-9, rtol=2e-7)


def test_source_jvp_and_vjp_obey_the_raw_cartesian_adjoint_identity():
    model = _model(W)
    rng = np.random.default_rng(2026090203)
    raw_direction = rng.normal(size=W.source.shape)
    cartesian_cotangent = rng.normal(size=W.source.shape)
    jvp = model.source_jvp(W.positions, W.source, raw_direction)
    vjp = model.source_vjp(W.positions, W.source, cartesian_cotangent)
    assert float(np.vdot(cartesian_cotangent, jvp)) == pytest.approx(
        float(np.vdot(vjp, raw_direction)), abs=2e-12, rel=0.0
    )


def test_raw_source_hvp_is_only_exposed_under_an_explicit_debug_name():
    model = _model(P)
    direction = np.ones_like(P.source)
    result = model.debug_source_hvp_raw(P.positions, P.source, direction)
    assert result.shape == P.source.shape
    assert not hasattr(model, "source_hvp")


def test_coordinate_gradient_matches_energy_finite_difference():
    model = _model(P)
    direction = np.random.default_rng(2026090204).normal(size=P.positions.shape)
    direction -= np.mean(direction, axis=0)
    analytic = float(np.vdot(model.coordinate_gradient(P.positions, P.source), direction))
    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        actual = central_difference(lambda positions: model.energy(positions, P.source), P.positions, direction, step)
        errors.append(abs(actual - analytic))
    assert errors[-1] < errors[0] / 20
    assert errors[-1] < 3e-9


def test_mixed_coordinate_vjp_uses_a_cartesian_drive_cotangent():
    model = _model(P)
    rng = np.random.default_rng(2026090205)
    cotangent = rng.normal(size=P.source.shape)
    direction = rng.normal(size=P.positions.shape)
    expected = float(np.vdot(model.mixed_coordinate_vjp(P.positions, P.source, cotangent), direction))
    step = 1e-5
    actual = central_difference(
        lambda positions: float(np.vdot(cotangent, model.drive_cartesian(positions, P.source))),
        P.positions,
        direction,
        step,
    )
    assert actual == pytest.approx(expected, abs=4e-7, rel=2e-6)


def test_joint_hvp_matches_finite_difference_of_both_first_derivatives():
    model = _model(P)
    rng = np.random.default_rng(2026090206)
    d_positions = rng.normal(size=P.positions.shape)
    d_source = rng.normal(size=P.source.shape)
    coordinate_block, source_block = model.joint_hvp(P.positions, P.source, d_positions, d_source)
    step = 2e-5
    plus_gradient = model.coordinate_gradient(P.positions + step * d_positions, P.source + step * d_source)
    minus_gradient = model.coordinate_gradient(P.positions - step * d_positions, P.source - step * d_source)
    actual_coordinate = (plus_gradient - minus_gradient) / (2 * step)
    plus_raw = block_permutation(2) @ model.drive_cartesian(P.positions + step * d_positions, P.source + step * d_source).ravel()
    minus_raw = block_permutation(2) @ model.drive_cartesian(P.positions - step * d_positions, P.source - step * d_source).ravel()
    actual_source = ((plus_raw - minus_raw) / (2 * step)).reshape(P.source.shape)
    assert np.allclose(coordinate_block, actual_coordinate, atol=4e-7, rtol=2e-6)
    assert np.allclose(source_block, actual_source, atol=4e-8, rtol=2e-6)


def test_joint_hessian_has_bilinear_symmetry():
    model = _model(P)
    rng = np.random.default_rng(2026090207)
    a_r, a_c = rng.normal(size=P.positions.shape), rng.normal(size=P.source.shape)
    b_r, b_c = rng.normal(size=P.positions.shape), rng.normal(size=P.source.shape)
    ha_r, ha_c = model.joint_hvp(P.positions, P.source, a_r, a_c)
    hb_r, hb_c = model.joint_hvp(P.positions, P.source, b_r, b_c)
    lhs = float(np.vdot(b_r, ha_r) + np.vdot(b_c, ha_c))
    rhs = float(np.vdot(a_r, hb_r) + np.vdot(a_c, hb_c))
    assert lhs == pytest.approx(rhs, abs=2e-9, rel=0.0)


def test_quadratic_scalar_equals_half_source_gradient_pairing():
    model = _model(W)
    drive = model.drive_cartesian(W.positions, W.source)
    raw_gradient = (block_permutation(3) @ drive.ravel()).reshape(W.source.shape)
    assert model.energy(W.positions, W.source) == pytest.approx(
        0.5 * float(np.vdot(W.source, raw_gradient)), abs=2e-12, rel=0.0
    )


def test_fused_energy_drive_gradient_matches_separate_calls():
    model = _model(P)
    energy, drive, gradient = model.energy_drive_gradient(P.positions, P.source)
    assert energy == pytest.approx(model.energy(P.positions, P.source), abs=2e-12, rel=0.0)
    assert np.allclose(drive, model.drive_cartesian(P.positions, P.source), atol=2e-12, rtol=0.0)
    assert np.allclose(gradient, model.coordinate_gradient(P.positions, P.source), atol=2e-11, rtol=0.0)


@pytest.mark.parametrize("method,args", [
    ("source_jvp", (np.full_like(P.source, np.nan),)),
    ("source_vjp", (np.full_like(P.source, np.nan),)),
    ("mixed_coordinate_vjp", (np.full_like(P.source, np.nan),)),
    ("joint_hvp", (np.zeros_like(P.positions), np.full_like(P.source, np.nan))),
])
def test_derivative_directions_and_cotangents_require_finite_float64(method, args):
    with pytest.raises((TypeError, ValueError), match="finite|float64|direction|cotangent"):
        getattr(_model(P), method)(P.positions, P.source, *args)


def test_corrupted_dense_solve_is_rejected_by_residual_audit(monkeypatch):
    import torch

    monkeypatch.setattr(torch.linalg, "solve", lambda matrix, rhs: torch.zeros_like(rhs))
    with pytest.raises((ValueError, RuntimeError), match="residual|backward|amplification"):
        _model(P).audit(P.positions, P.source)


def test_detached_multisphere_geometry_graph_fails_instead_of_returning_zero(monkeypatch):
    original = TorchSmoothPCM._assemble_torch

    def detached(self, positions):
        assembly = original(self, positions)
        return type(assembly)(*(value.detach() if hasattr(value, "detach") else value for value in assembly))

    monkeypatch.setattr(TorchSmoothPCM, "_assemble_torch", detached)
    with pytest.raises((RuntimeError, ValueError), match="detach|geometry|gradient|graph|require grad"):
        _model(P).coordinate_gradient(P.positions, P.source)


def test_one_sphere_coordinate_gradient_is_explicitly_zero():
    from _torch_smooth_pcm_reference import S

    gradient = _model(S).coordinate_gradient(S.positions, S.source)
    assert np.array_equal(gradient, np.zeros_like(S.positions))
