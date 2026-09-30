"""Structured operator derivatives against the untouched dense scalar algebra."""

import numpy as np
import pytest
import torch

import maple.solvation.continuum.ddpcm_response_operators as operator_module
from maple.solvation.continuum.ddpcm_response_operators import DDPCMResponseOperators
from maple.solvation.continuum.torch_ddpcm import TorchDDPCM


@pytest.fixture(params=(2, 3))
def problem(request):
    n = request.param
    symbols = ("C", "O", "H")[:n]
    radii = np.array([1.7, 1.5, 1.2])[:n]
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.8, 0.23, 0.12], [-0.27, 1.45, 0.32]][:n],
        dtype=torch.float64,
    )
    source = torch.tensor(
        [[0.4, 0.1, -0.1, 0.2], [-0.5, -0.05, 0.12, -0.18], [0.1, 0.02, -0.03, 0.04]][
            :n
        ],
        dtype=torch.float64,
    )
    kwargs = dict(dielectric=78.39, lmax=2, n_lebedev=50)
    old = TorchDDPCM(symbols, radii, **kwargs)
    new = DDPCMResponseOperators(symbols, radii, **kwargs)
    return old, new, positions, source


def dense_parts(old, positions, active):
    source = active.new_zeros((len(active), 8))
    source[:, (0, 2, 3, 4)] = active
    geometry = old._geometry(positions)
    L, D, design, ui = old._operators(positions, geometry)
    phi, psi = old._point_source(geometry[4], geometry[3], geometry[2], source)
    rhs = torch.cat(
        [-(design.T @ (geometry[1] * ui[i] * phi[i])) for i in range(len(active))]
    )
    return L, D, rhs, psi


def test_primal_operators_and_topology_match_dense_reference(problem):
    old, new, r, s = problem
    actual = new.build(r, s)
    expected = dense_parts(old, r, s)
    for value, reference in zip(
        (actual.L, actual.D, actual.rhs, actual.psi), expected, strict=True
    ):
        torch.testing.assert_close(value, reference, atol=2e-12, rtol=2e-13)
        assert not value.requires_grad
    source = s.new_zeros((len(s), 8))
    source[:, (0, 2, 3, 4)] = s
    ref = old._state_torch(r, source)
    for name in (
        "active_node_count",
        "buried_plateau_count",
        "transition_node_count",
        "minimum_switch_margin",
        "minimum_active_f_margin",
    ):
        assert getattr(actual.geometry.topology, name) == getattr(ref.topology, name)
    assert actual.geometry.topology.contract != ref.topology.contract
    assert actual.geometry.topology.one_sided_nodes == ()


def test_all_action_jacobians_and_contracted_hessians_match_autograd(problem):
    old, new, r, s = problem
    primal = new.build(r, s)
    gen = torch.Generator().manual_seed(260927)
    mu, z, lam, v0, alpha = [
        torch.randn(primal.rhs.numel(), generator=gen, dtype=torch.float64)
        for _ in range(5)
    ]
    d = new.derivatives(primal, mu=mu, z=z, lam=lam, v0=v0, alpha=alpha, order=2)
    m = r.numel()
    for name, index, vector, transpose in (
        ("l_action_jacobian", 0, z, False),
        ("d_action_jacobian", 1, v0, False),
        ("lt_action_jacobian", 0, mu, True),
        ("dt_action_jacobian", 1, lam, True),
    ):

        def action(q):
            matrix = dense_parts(old, q, s)[index]
            return (matrix.T if transpose else matrix) @ vector

        reference = torch.autograd.functional.jacobian(action, r).reshape(-1, m)
        torch.testing.assert_close(getattr(d, name), reference, atol=2e-10, rtol=2e-12)
    for name, index, left, right in (
        ("l_contraction_hessian", 0, mu, z),
        ("d_contraction_hessian", 1, lam, v0),
    ):
        reference = torch.autograd.functional.hessian(
            lambda q: left @ dense_parts(old, q, s)[index] @ right, r
        ).reshape(m, m)
        torch.testing.assert_close(getattr(d, name), reference, atol=5e-9, rtol=2e-11)
    theta = torch.cat((r.flatten(), s.flatten()))

    def rhs(q):
        return dense_parts(old, q[:m].reshape_as(r), q[m:].reshape_as(s))[2]

    def psi(q):
        return dense_parts(old, q[:m].reshape_as(r), q[m:].reshape_as(s))[3]

    torch.testing.assert_close(
        d.rhs_jacobian,
        torch.autograd.functional.jacobian(rhs, theta),
        atol=2e-11,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        d.psi_jacobian,
        torch.autograd.functional.jacobian(psi, theta),
        atol=2e-13,
        rtol=2e-13,
    )
    torch.testing.assert_close(
        d.rhs_contraction_hessian,
        torch.autograd.functional.hessian(lambda q: alpha @ rhs(q), theta),
        atol=2e-9,
        rtol=2e-11,
    )
    assert float(d.rhs_contraction_hessian[:m, m:].abs().max()) > 1e-5


def test_order_two_uses_fixed_central_coefficients_and_never_full_basis_hessians(
    problem, monkeypatch
):
    _, new, r, s = problem
    primal = new.build(r, s)
    generator = torch.Generator().manual_seed(20260927)
    mu, z, lam, v0, alpha = [
        torch.randn(primal.rhs.numel(), generator=generator, dtype=torch.float64)
        for _ in range(5)
    ]
    original = operator_module.solid_harmonic_contracted_hessian
    calls = []

    def capture(vectors, lmax, kind, coefficients):
        calls.append((kind, coefficients.detach().clone()))
        return original(vectors, lmax, kind, coefficients)

    def forbid_full_jets(*args, **kwargs):
        raise AssertionError(
            "order-two production path materialized full basis Hessians"
        )

    monkeypatch.setattr(operator_module, "solid_harmonic_contracted_hessian", capture)
    monkeypatch.setattr(operator_module, "solid_harmonic_jets", forbid_full_jets)
    new.derivatives(primal, mu=mu, z=z, lam=lam, v0=v0, alpha=alpha, order=2)

    expected = []
    q = new.n_basis
    for i in range(new.n_atoms):
        for j in range(new.n_atoms):
            if i == j:
                continue
            columns = slice(j * q, (j + 1) * q)
            expected.extend(
                (
                    ("regular", new._single_scale * z[columns]),
                    ("irregular", new._double_scale * v0[columns]),
                )
            )
    assert len(calls) == len(expected)
    for (actual_kind, actual), (expected_kind, reference) in zip(
        calls, expected, strict=True
    ):
        assert actual_kind == expected_kind
        torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)


def test_position_translation_annihilates_actions(problem):
    _, new, r, s = problem
    primal = new.build(r, s)
    v = torch.arange(primal.rhs.numel(), dtype=torch.float64) / 17
    d = new.derivatives(primal, mu=v, z=v, lam=v, v0=v, alpha=v, order=2)
    for axis in range(3):
        translation = torch.zeros_like(r)
        translation[:, axis] = 1
        for jac in (
            d.l_action_jacobian,
            d.d_action_jacobian,
            d.lt_action_jacobian,
            d.dt_action_jacobian,
        ):
            torch.testing.assert_close(
                jac @ translation.flatten(),
                torch.zeros(jac.shape[0], dtype=r.dtype),
                atol=2e-10,
                rtol=0,
            )


def test_resource_light_helpers_keep_collision_guard(problem):
    from maple.solvation.surfaces.lebedev import ordered_lebedev_grid

    old, new, r, s = problem
    changed = r.clone()
    direction = torch.tensor(ordered_lebedev_grid(50).directions[0].copy())
    changed[1] = changed[0] + new.radii_angstrom[0] * direction
    with pytest.raises(RuntimeError, match="coincides"):
        old._geometry(changed)
    with pytest.raises(RuntimeError, match="coincides"):
        new.build(changed, s)


@pytest.mark.parametrize(
    "name",
    (
        "_directions",
        "_weights",
        "_radii",
        "_ell",
        "_single_scale",
        "_double_scale",
        "_design",
        "_identity3",
    ),
)
def test_cached_constants_cannot_silently_change_scalar(problem, name):
    _, new, r, s = problem
    constant = getattr(new, name)
    with pytest.raises(AttributeError, match="immutable"):
        setattr(new, name, constant.clone())
    constant.add_(0.1)
    with pytest.raises(RuntimeError, match="mutated"):
        new.configuration_payload()
    with pytest.raises(RuntimeError, match="mutated"):
        new.build(r, s)
