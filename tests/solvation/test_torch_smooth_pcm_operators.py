from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from ase.units import Hartree
import torch

from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import (
    TorchSmoothPCM,
    block_permutation,
)
from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm.exposure import (
    _distinct_quadrature_point_source_block,
    _self_quadrature_point_source_block,
)

from _torch_smooth_pcm_reference import (
    COULOMB_EV_ANGSTROM_PER_E2,
    P,
    S,
    W,
    direct_sphere_projection,
    isolated_sphere_energy,
    implementation_point_potential_control,
    kkt_source_gradient,
    lagrangian,
    residual_metrics,
    solve_primal,
    solve_transpose_kkt,
    tangency_fixture,
)


def _model(fixture, **changes):
    options = dict(
        atomic_numbers=fixture.atomic_numbers,
        radii_angstrom=fixture.radii,
        transition_width_angstrom2=fixture.transition_width,
        surface_lmax=fixture.surface_lmax,
        partition_lmax=fixture.partition_lmax,
        partition_radial_quadrature_order=fixture.partition_order,
        source_radial_quadrature_order=fixture.source_order,
        double_layer_radial_quadrature_order=fixture.double_layer_order,
        dielectric=fixture.dielectric,
        source_shell_clearance_angstrom=0.05,
    )
    options.update(changes)
    return TorchSmoothPCM(**options)


def _fast_model(fixture, **changes):
    return _model(
        fixture,
        partition_radial_quadrature_order=32,
        source_radial_quadrature_order=48,
        double_layer_radial_quadrature_order=48,
        **changes,
    )


@pytest.fixture(scope="module")
def water_radial_refinement_results():
    results = []
    for partition_order, source_order, double_order in ((64, 96, 96), (96, 128, 128), (128, 160, 160)):
        model = _model(
            W,
            partition_radial_quadrature_order=partition_order,
            source_radial_quadrature_order=source_order,
            double_layer_radial_quadrature_order=double_order,
        )
        results.append(model.energy_drive_gradient(W.positions, W.source))
    return results


@pytest.mark.parametrize(
    "branch,source_center",
    [
        ("self", np.asarray([0.0, 0.0, 0.0])),
        ("inside", np.asarray([0.60, -0.20, 0.10])),
        ("outside", np.asarray([2.10, 0.30, -0.20])),
    ],
)
@pytest.mark.parametrize("channel", range(4), ids=("q", "py", "pz", "px"))
def test_each_point_l1_channel_matches_independent_direct_surface_projection(branch, source_center, channel):
    target_center = np.zeros(3)
    target_radius = 1.25
    source = np.zeros(4)
    source[channel] = 0.37
    reference = torch.zeros(3, dtype=torch.float64)
    if branch == "self":
        block = Hartree * _self_quadrature_point_source_block(reference, target_radius=target_radius, lmax=3)
    else:
        block = Hartree * _distinct_quadrature_point_source_block(
            torch.tensor(source_center - target_center, dtype=torch.float64),
            target_radius=target_radius,
            lmax=3,
            radial_order=128,
        )
    actual = (block.detach().numpy() @ source)
    expected = direct_sphere_projection(
        target_center=target_center,
        target_radius=target_radius,
        source_center=source_center,
        source_raw=source,
        lmax=3,
    )
    assert np.allclose(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("axis,raw_channel", [(0, 3), (1, 1), (2, 2)])
def test_positive_cartesian_dipole_has_positive_plus_axis_and_negative_minus_axis(axis, raw_channel):
    source = np.zeros((1, 4))
    source[0, raw_channel] = 0.4
    points = np.zeros((2, 3))
    points[:, axis] = (1.0, -1.0)
    from _torch_smooth_pcm_reference import point_multipole_potential_ev

    values = point_multipole_potential_ev(points, np.zeros((1, 3)), source)
    assert values[0] > 0.0
    assert values[1] < 0.0
    assert values[0] == pytest.approx(-values[1], abs=2e-14, rel=0.0)


def test_independent_point_multipole_oracle_agrees_with_existing_gto_density_control():
    from ase.units import Bohr
    from _torch_smooth_pcm_reference import point_multipole_potential_ev

    points = np.asarray([[1.7, -0.3, 0.4], [-0.8, 1.4, 0.6]])
    expected = point_multipole_potential_ev(points, P.positions, P.source)
    actual = implementation_point_potential_control(points / Bohr, P.positions, P.source)
    assert np.allclose(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("channel", range(4), ids=("q", "py", "pz", "px"))
def test_isolated_sphere_each_source_channel_matches_finite_dielectric_analytic_limit(channel):
    source = np.zeros_like(S.source)
    source[0, channel] = S.source[0, channel]
    fixture = replace(S, source=source)
    expected = isolated_sphere_energy(fixture)
    assert _model(fixture).energy(fixture.positions, source) == pytest.approx(expected, abs=3e-14, rel=0.0)


def test_isolated_sphere_combined_monopole_and_dipole_matches_analytic_limit():
    assert _model(S).energy(S.positions, S.source) == pytest.approx(isolated_sphere_energy(), abs=3e-14, rel=0.0)


def test_isolated_sphere_receiver_partition_and_schwarz_blocks_have_analytic_form():
    matrices = _model(S).debug_geometry_matrices(S.positions)
    expected_field = np.zeros((4, 4))
    expected_field[0, 0] = 1 / np.sqrt(4 * np.pi)
    gradient_scale = np.sqrt(3 / (4 * np.pi)) / S.radii[0]
    expected_field[1, 3] = gradient_scale
    expected_field[2, 1] = gradient_scale
    expected_field[3, 2] = gradient_scale
    assert np.allclose(matrices["C_field"], expected_field, atol=2e-14, rtol=0.0)
    assert np.allclose(matrices["L"], np.eye(4), atol=2e-14, rtol=0.0)
    expected_partition = np.zeros((1, (S.partition_lmax + 1) ** 2))
    expected_partition[0, 0] = np.sqrt(4 * np.pi)
    assert np.allclose(matrices["exposed_coefficients"], expected_partition, atol=2e-12, rtol=0.0)
    assert np.allclose(matrices["centered_exposure_coefficients"], expected_partition, atol=2e-12, rtol=0.0)


@pytest.mark.parametrize("fixture", [S, P, W], ids=lambda fixture: fixture.name)
def test_finite_and_conductor_operators_differ_only_by_analytic_dielectric_jump(fixture):
    matrices = _fast_model(fixture).debug_geometry_matrices(fixture.positions)
    dimension = matrices["A_eps"].shape[0]
    expected = 4 * np.pi / (fixture.dielectric - 1) * np.eye(dimension)
    assert np.allclose(matrices["A_eps"] - matrices["A_inf"], expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("fixture", [S, P, W, tangency_fixture(0.2), tangency_fixture(1.8)], ids=lambda item: item.name)
def test_debug_matrices_obey_frozen_dimensions_and_raw_cartesian_receiver_mapping(fixture):
    matrices = _fast_model(fixture).debug_geometry_matrices(fixture.positions)
    n = len(fixture.atomic_numbers)
    harmonic_dimension = n * (fixture.surface_lmax + 1) ** 2
    assert matrices["B"].shape == (harmonic_dimension, 4 * n)
    assert matrices["C_field"].shape == (4 * n, harmonic_dimension)
    assert matrices["C_raw"].shape == (4 * n, harmonic_dimension)
    for name in ("A_eps", "A_inf", "L"):
        assert matrices[name].shape == (harmonic_dimension, harmonic_dimension)
        assert np.all(np.isfinite(matrices[name]))
    assert np.allclose(matrices["C_raw"], block_permutation(n) @ matrices["C_field"], atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("fixture", [P, W, tangency_fixture(0.2), tangency_fixture(1.8)], ids=lambda item: item.name)
def test_numpy_linear_solves_reproduce_primal_states_and_scalar(fixture):
    model = _fast_model(fixture)
    matrices = model.debug_geometry_matrices(fixture.positions)
    expected = solve_primal(matrices, fixture.source)
    state = model.debug_primal_state(fixture.positions, fixture.source)
    for name in ("F", "G", "X"):
        assert np.allclose(state[name], expected[name], atol=2e-11, rtol=2e-11)
    assert state["U"] == pytest.approx(expected["energy"], abs=2e-12, rel=2e-11)
    assert model.energy(fixture.positions, fixture.source) == pytest.approx(expected["energy"], abs=2e-12, rel=2e-11)


@pytest.mark.parametrize("fixture", [P, W, tangency_fixture(0.2)], ids=lambda item: item.name)
def test_primal_and_transpose_kkt_residuals_close(fixture):
    model = _fast_model(fixture)
    matrices = model.debug_geometry_matrices(fixture.positions)
    primal = solve_primal(matrices, fixture.source)
    adjoint = solve_transpose_kkt(matrices, fixture.source)
    c = fixture.source.ravel()
    residuals = (
        primal["F"] + matrices["B"] @ c,
        matrices["A_eps"] @ primal["G"] - matrices["A_inf"] @ primal["F"],
        matrices["L"] @ primal["X"] - primal["G"],
        matrices["L"].T @ adjoint["lambda_X"] + 0.5 * matrices["C_raw"].T @ c,
        matrices["A_eps"].T @ adjoint["lambda_G"] - adjoint["lambda_X"],
        adjoint["lambda_F"] - matrices["A_inf"].T @ adjoint["lambda_G"],
    )
    assert max(np.linalg.norm(value) for value in residuals) < 2e-11
    state = model.debug_primal_state(fixture.positions, fixture.source)
    for name in ("lambda_F", "lambda_G", "lambda_X"):
        assert np.allclose(state[name], adjoint[name], atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("variable", ["F", "G", "X", "lambda_F", "lambda_G", "lambda_X"])
def test_full_lagrangian_is_stationary_in_each_primal_and_multiplier_variable(variable):
    model = _fast_model(P)
    matrices = model.debug_geometry_matrices(P.positions)
    primal = solve_primal(matrices, P.source)
    adjoint = solve_transpose_kkt(matrices, P.source)
    assert lagrangian(matrices, P.source, primal, adjoint) == pytest.approx(primal["energy"], abs=2e-12, rel=0.0)
    selected = primal if variable in primal else adjoint
    direction = np.random.default_rng(2026090208).normal(size=np.asarray(selected[variable]).shape)
    step = 1e-5
    plus_primal = dict(primal)
    minus_primal = dict(primal)
    plus_adjoint = dict(adjoint)
    minus_adjoint = dict(adjoint)
    plus = plus_primal if variable in primal else plus_adjoint
    minus = minus_primal if variable in primal else minus_adjoint
    plus[variable] = np.asarray(selected[variable]) + step * direction
    minus[variable] = np.asarray(selected[variable]) - step * direction
    derivative = (
        lagrangian(matrices, P.source, plus_primal, plus_adjoint)
        - lagrangian(matrices, P.source, minus_primal, minus_adjoint)
    ) / (2 * step)
    assert derivative == pytest.approx(0.0, abs=2e-10, rel=0.0)


def test_energy_source_gradient_matches_the_transpose_kkt_and_records_receiver_difference():
    model = _fast_model(P)
    matrices = model.debug_geometry_matrices(P.positions)
    expected_raw = kkt_source_gradient(matrices, P.source)
    drive = model.drive_cartesian(P.positions, P.source).ravel()
    assert np.allclose(drive, block_permutation(2).T @ expected_raw, atol=2e-11, rtol=2e-11)
    conventional = matrices["C_field"] @ solve_primal(matrices, P.source)["X"]
    # Equality is allowed for the reciprocal c578 lineage.  The test records
    # the quantity rather than manufacturing nonsymmetry; KKT/autograd remains
    # the derivative authority in either case.
    receiver_difference = np.linalg.norm(drive - conventional)
    assert np.isfinite(receiver_difference)
    assert receiver_difference < 2e-11


@pytest.mark.parametrize(
    "fixture,tau,relative_floor,kappa_max",
    [
        (S, 1e-10, 1e-8, 4.503599627370496e4),
        (P, 1e-9, 1e-9, 4.503599627370496e5),
        (W, 1e-8, 1e-10, 4.503599627370496e6),
        (tangency_fixture(0.2), 1e-8, 1e-10, 4.503599627370496e6),
    ],
    ids=lambda value: value.name if hasattr(value, "name") else None,
)
def test_fixture_solves_meet_conditioning_backward_error_and_amplification_gates(fixture, tau, relative_floor, kappa_max):
    model = _fast_model(fixture)
    matrices = model.debug_geometry_matrices(fixture.positions)
    primal = solve_primal(matrices, fixture.source)
    systems = (
        (matrices["A_eps"], primal["G"], matrices["A_inf"] @ primal["F"]),
        (matrices["L"], primal["X"], primal["G"]),
    )
    for operator, solution, rhs in systems:
        metrics = residual_metrics(operator, solution, rhs)
        assert metrics["sigma_min"] >= 1e-12
        assert metrics["relative_sigma_min"] >= relative_floor
        assert metrics["kappa"] <= kappa_max
        assert metrics["kappa"] <= 1e12
        assert metrics["backward_error"] <= 1e-12
        assert metrics["amplification_indicator"] <= tau


def test_near_vacuum_response_vanishes_and_conductor_limit_is_finite():
    near = _model(S, dielectric=1.0 + 1e-8).energy(S.positions, S.source)
    conductor = _model(S, dielectric=1e8).energy(S.positions, S.source)
    conductor_expected = isolated_sphere_energy(replace(S, dielectric=1e8))
    assert abs(near) < 1e-7
    assert conductor == pytest.approx(conductor_expected, abs=2e-9, rel=2e-8)


def test_paired_radial_refinement_converges_water_energy(water_radial_refinement_results):
    energies = [result[0] for result in water_radial_refinement_results]
    assert abs(energies[-1] - energies[-2]) < 2e-10


def test_paired_radial_refinement_converges_water_drive_and_gradient(water_radial_refinement_results):
    _energies, drives, gradients = zip(*water_radial_refinement_results, strict=True)
    assert np.max(np.abs(drives[-1] - drives[-2])) < 2e-8
    assert np.max(np.abs(gradients[-1] - gradients[-2])) < 2e-8


def test_source_on_target_shell_clearance_is_rejected_instead_of_switching_branch():
    fixture = replace(P, positions=np.asarray([[0.0, 0.0, 0.0], [1.23, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="shell|clearance|singular"):
        _fast_model(fixture).energy(fixture.positions, fixture.source)


def test_source_within_declared_shell_clearance_is_rejected():
    fixture = replace(P, positions=np.asarray([[0.0, 0.0, 0.0], [1.23 + 0.025, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="shell|clearance|singular"):
        _fast_model(fixture).energy(fixture.positions, fixture.source)


def test_high_condition_operator_is_rejected_without_regularization():
    model = _fast_model(S)
    matrix = torch.diag(torch.tensor([1.0, 1e-8], dtype=torch.float64))
    with pytest.raises(ValueError, match="conditioning|singular"):
        model._matrix_gate(matrix, "synthetic")
