"""Cross-implementation matrix oracle for the private smooth ddPCM scalar."""

from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import (
    TorchSmoothPCM,
)

from _torch_smooth_pcm_crossblock_reference import (
    P,
    W,
    assemble_reference,
    solve_reference,
    tangency_fixture,
)


FIXTURES = (P, W, tangency_fixture(0.2), tangency_fixture(1.8))
MATRIX_ATOL = 2.0e-11
MATRIX_RTOL = 2.0e-11
STATE_ATOL = 2.0e-11
STATE_RTOL = 2.0e-11


def _model(fixture):
    return TorchSmoothPCM(
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


@pytest.fixture(scope="module", params=FIXTURES, ids=lambda fixture: fixture.name)
def crossblock_case(request):
    fixture = request.param
    model = _model(fixture)
    independent = assemble_reference(fixture)
    production = model.debug_geometry_matrices(fixture.positions)
    return fixture, model, independent, production


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_independent_source_semantics_match_the_current_gto_density_control(fixture):
    point = np.asarray([[4.7, -2.1, 1.3]])
    cartesian_source = fixture.source[:, (0, 3, 1, 2)]
    displacement = point[:, None, :] - fixture.positions[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    expected_ev = np.sum(
        cartesian_source[None, :, 0] / distance
        + np.einsum("pai,ai->pa", displacement, cartesian_source[:, 1:]) / distance**3,
        axis=1,
    ) * Hartree * Bohr
    actual_ev = Hartree * point_multipole_potential(
        point / Bohr, fixture.positions, fixture.source
    )
    assert np.allclose(actual_ev, expected_ev, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize(
    "name",
    (
        "exposed_coefficients",
        "centered_exposure_coefficients",
        "B",
        "L",
        "localized_double_layer",
        "A_inf",
        "A_eps",
        "C_raw",
        "C_field",
    ),
)
def test_independent_numpy_reassembly_matches_each_production_matrix(crossblock_case, name):
    _fixture, _model_instance, independent, production = crossblock_case
    assert independent[name].shape == production[name].shape
    assert np.allclose(
        independent[name], production[name], atol=MATRIX_ATOL, rtol=MATRIX_RTOL
    ), f"{name} max error={np.max(np.abs(independent[name] - production[name])):.3e}"


def test_independent_primal_states_and_scalar_match_production(crossblock_case):
    fixture, model, independent, _production = crossblock_case
    expected = solve_reference(fixture, independent)
    actual = model.debug_primal_state(fixture.positions, fixture.source)
    for name in ("F", "G", "X"):
        assert np.allclose(
            expected[name], actual[name], atol=STATE_ATOL, rtol=STATE_RTOL
        ), f"{name} max error={np.max(np.abs(expected[name] - actual[name])):.3e}"
    assert actual["U"] == pytest.approx(expected["U"], abs=STATE_ATOL, rel=STATE_RTOL)
    assert model.energy(fixture.positions, fixture.source) == pytest.approx(
        expected["U"], abs=STATE_ATOL, rel=STATE_RTOL
    )


def test_independent_transpose_kkt_state_and_gradient_match_production(crossblock_case):
    fixture, model, independent, _production = crossblock_case
    expected = solve_reference(fixture, independent)
    actual = model.debug_primal_state(fixture.positions, fixture.source)
    for name in ("lambda_F", "lambda_G", "lambda_X", "raw_gradient_kkt"):
        assert np.allclose(
            expected[name], actual[name], atol=STATE_ATOL, rtol=STATE_RTOL
        ), f"{name} max error={np.max(np.abs(expected[name] - actual[name])):.3e}"
    assert np.allclose(
        expected["raw_gradient_kkt"], actual["raw_gradient_autograd"],
        atol=STATE_ATOL, rtol=STATE_RTOL,
    )
    assert actual["kkt_gradient_max_error"] < STATE_ATOL


def test_independent_equations_close_primal_and_transpose_residuals(crossblock_case):
    fixture, _model_instance, matrices, _production = crossblock_case
    state = solve_reference(fixture, matrices)
    c = fixture.source.ravel()
    residuals = (
        state["F"] + matrices["B"] @ c,
        matrices["A_eps"] @ state["G"] - matrices["A_inf"] @ state["F"],
        matrices["L"] @ state["X"] - state["G"],
        matrices["L"].T @ state["lambda_X"] + 0.5 * matrices["C_raw"].T @ c,
        matrices["A_eps"].T @ state["lambda_G"] - state["lambda_X"],
        state["lambda_F"] - matrices["A_inf"].T @ state["lambda_G"],
    )
    assert max(np.linalg.norm(residual) for residual in residuals) < 2.0e-11


def test_independent_raw_and_cartesian_receiver_obey_the_frozen_transpose_relation(crossblock_case):
    fixture, _model_instance, matrices, _production = crossblock_case
    q = np.kron(
        np.eye(len(fixture.radii)),
        np.asarray(((1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                    (0.0, 1.0, 0.0, 0.0))),
    )
    assert np.array_equal(matrices["C_field"], q.T @ matrices["C_raw"])


def test_independence_boundary_is_explicit_and_production_private_helpers_are_not_imported():
    import ast
    from pathlib import Path

    import _torch_smooth_pcm_crossblock_reference as oracle

    tree = ast.parse(Path(oracle.__file__).read_text())
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imported_from = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any("torch_smooth_pcm" in name for name in imported_modules | imported_from)
    assert "torch" not in imported_modules
    assert oracle.__doc__ is not None
    assert "independence boundary" in oracle.__doc__.lower()
