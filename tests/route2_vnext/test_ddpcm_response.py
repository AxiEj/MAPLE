from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import maple.solvation.continuum.ddpcm_response as response_module
from maple.solvation.continuum.solid_harmonic_response import (
    contracted_hessian_table_nbytes,
    derivative_table_nbytes,
)
from maple.solvation.continuum.torch_ddpcm import DDPCMTopologyCertificate


@pytest.fixture(autouse=True)
def _double_precision_test_scope():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        yield
    finally:
        torch.set_default_dtype(original)


def _model(theta: torch.Tensor):
    r = theta[:3]
    s = theta[3:]
    x, y, z = r
    q, sy, sz, sx = s
    L = torch.stack(
        (
            torch.stack((2.2 + 0.13 * x + 0.03 * y * y, 0.17 + 0.04 * z)),
            torch.stack((-0.11 + 0.02 * y, 1.8 - 0.07 * x + 0.02 * z * z)),
        )
    )
    D = torch.stack(
        (
            torch.stack((0.21 + 0.03 * x * y, -0.08 + 0.02 * z)),
            torch.stack((0.05 - 0.04 * x, 0.16 + 0.02 * y * z)),
        )
    )
    # Nonlinear geometry and mixed coordinate/source dependence are deliberate:
    # they lock source curvature and the off-shell D_i r_j cross term.
    rhs = torch.stack(
        (
            0.4 + x * x + 0.3 * y * q - 0.2 * z * sy + 0.05 * sx * x,
            -0.1 + torch.sin(y) + 0.2 * z * sz + 0.07 * q * x * z,
        )
    )
    psi = torch.stack((0.7 * q - 0.2 * sy + 0.1 * sx, -0.3 * q + 0.4 * sz + 0.2 * sx))
    return L, D, rhs, psi


def _jacobian(function, theta):
    return torch.autograd.functional.jacobian(function, theta, create_graph=False)


def _hessian(function, theta):
    return torch.autograd.functional.hessian(function, theta, create_graph=False)


@dataclass
class _Derivatives:
    rhs_jacobian: torch.Tensor
    psi_jacobian: torch.Tensor
    l_action_jacobian: torch.Tensor
    d_action_jacobian: torch.Tensor
    lt_action_jacobian: torch.Tensor
    dt_action_jacobian: torch.Tensor
    rhs_contraction_hessian: torch.Tensor | None = None
    l_contraction_hessian: torch.Tensor | None = None
    d_contraction_hessian: torch.Tensor | None = None


class _FakeOperators:
    build_count = 0
    derivative_count = 0

    def __init__(self, symbols, radii, **kwargs):
        self.symbols = tuple(symbols)
        self.device = kwargs["device"]

    def build(self, positions, source_active):
        type(self).build_count += 1
        theta = torch.cat((positions.reshape(-1), source_active.reshape(-1)))
        L, D, rhs, psi = _model(theta)
        topology = DDPCMTopologyCertificate(
            "synthetic-fixed-topology", "0" * 64, 1, 0, 0, 1.0, None
        )
        return SimpleNamespace(
            L=L,
            D=D,
            rhs=rhs,
            psi=psi,
            geometry=SimpleNamespace(topology=topology, positions=positions.clone()),
            source_active=source_active.clone(),
        )

    def derivatives(self, primal, *, mu, z, lam, v0, alpha, order):
        type(self).derivative_count += 1
        theta = (
            torch.cat(
                (
                    primal.geometry.positions.reshape(-1),
                    primal.source_active.reshape(-1),
                )
            )
            .detach()
            .requires_grad_(True)
        )
        result = _Derivatives(
            rhs_jacobian=_jacobian(lambda t: _model(t)[2], theta),
            psi_jacobian=_jacobian(lambda t: _model(t)[3], theta),
            l_action_jacobian=_jacobian(lambda t: _model(t)[0] @ z, theta)[:, :3],
            d_action_jacobian=_jacobian(lambda t: _model(t)[1] @ v0, theta)[:, :3],
            lt_action_jacobian=_jacobian(lambda t: _model(t)[0].mT @ mu, theta)[:, :3],
            dt_action_jacobian=_jacobian(lambda t: _model(t)[1].mT @ lam, theta)[:, :3],
        )
        if order == 2:
            result.rhs_contraction_hessian = _hessian(
                lambda t: torch.dot(alpha, _model(t)[2]), theta
            )
            result.l_contraction_hessian = _hessian(
                lambda t: torch.dot(mu, _model(t)[0] @ z), theta
            )[:3, :3]
            result.d_contraction_hessian = _hessian(
                lambda t: torch.dot(lam, _model(t)[1] @ v0), theta
            )[:3, :3]
        return result


@pytest.fixture
def backend(monkeypatch):
    module_name = "maple.solvation.continuum.ddpcm_response_operators"
    try:
        operators = __import__(module_name, fromlist=["DDPCMResponseOperators"])
        monkeypatch.setattr(operators, "DDPCMResponseOperators", _FakeOperators)
    except ModuleNotFoundError:
        operators = types.ModuleType(module_name)
        operators.DDPCMResponseOperators = _FakeOperators
        monkeypatch.setitem(sys.modules, module_name, operators)
    _FakeOperators.build_count = 0
    _FakeOperators.derivative_count = 0
    return response_module.TorchDDPCMResponse(
        ["H"], [1.2], dielectric=78.4, lmax=1, n_lebedev=6
    )


@pytest.fixture
def inputs():
    positions = torch.tensor([[0.21, -0.17, 0.31]], requires_grad=True)
    source = torch.zeros((1, 8), requires_grad=True)
    with torch.no_grad():
        source[0, (0, 2, 3, 4)] = torch.tensor([0.4, -0.2, 0.3, 0.1])
    return positions, source


def _direct_energy(positions, source, dielectric=78.4):
    from maple.solvation.api.units import HARTREE_TO_EV

    theta = torch.cat((positions.reshape(-1), source[:, (0, 2, 3, 4)].reshape(-1)))
    L, D, rhs, psi = _model(theta)
    identity = torch.eye(2)
    A = 2 * torch.pi * (dielectric + 1) / (dielectric - 1) * identity - D
    C = 2 * torch.pi * identity - D
    y = torch.linalg.solve(A, C @ rhs)
    z = torch.linalg.solve(L, y)
    return HARTREE_TO_EV / 2 * torch.dot(psi, z)


def test_exact_gradient_hessian_hvp_and_custom_double_backward(backend, inputs):
    positions, source = inputs
    energy = backend.energy_torch(positions, source)
    reference = _direct_energy(positions, source)
    assert torch.allclose(energy, reference, atol=2e-13, rtol=2e-13)

    gradient = torch.autograd.grad(energy, (positions, source), create_graph=True)
    reference_gradient = torch.autograd.grad(
        reference, (positions, source), create_graph=True
    )
    assert torch.allclose(gradient[0], reference_gradient[0], atol=2e-11, rtol=2e-11)
    assert torch.allclose(gradient[1], reference_gradient[1], atol=2e-11, rtol=2e-11)
    assert torch.count_nonzero(gradient[1][0, (1, 5, 6, 7)]) == 0

    hessian = backend.hessian_partial(positions.detach(), source.detach())
    theta = torch.cat(
        (positions.detach().reshape(-1), source.detach()[:, (0, 2, 3, 4)].reshape(-1))
    )
    reference_hessian = torch.autograd.functional.hessian(
        lambda t: _direct_energy(
            t[:3].reshape(1, 3),
            torch.cat((t[3:4], t.new_zeros(1), t[4:7], t.new_zeros(3))).reshape(1, 8),
        ),
        theta,
    )
    assert torch.allclose(hessian, reference_hessian, atol=5e-11, rtol=0.0)
    direction = torch.linspace(-0.3, 0.5, 7)
    assert torch.allclose(
        backend.hvp_partial(positions.detach(), source.detach(), direction),
        hessian @ direction,
        atol=5e-11,
        rtol=0.0,
    )

    cotangent = (torch.arange(3.0).reshape(1, 3), torch.arange(8.0).reshape(1, 8))
    second = torch.autograd.grad(gradient, (positions, source), grad_outputs=cotangent)
    active_direction = torch.cat(
        (cotangent[0].reshape(-1), cotangent[1][:, (0, 2, 3, 4)].reshape(-1))
    )
    expected = hessian @ active_direction
    assert torch.allclose(second[0].reshape(-1), expected[:3], atol=5e-11, rtol=0.0)
    assert torch.allclose(
        second[1][:, (0, 2, 3, 4)].reshape(-1), expected[3:], atol=5e-11, rtol=0.0
    )


def test_custom_scalar_passes_gradcheck_and_gradgradcheck(backend, inputs):
    positions, source = (value.detach() for value in inputs)
    theta = torch.cat(
        (positions.reshape(-1), source[:, (0, 2, 3, 4)].reshape(-1))
    ).requires_grad_(True)

    def scalar(candidate):
        active = candidate[3:]
        full_source = torch.stack(
            (
                active[0],
                active.new_zeros(()),
                active[1],
                active[2],
                active[3],
                active.new_zeros(()),
                active.new_zeros(()),
                active.new_zeros(()),
            )
        ).reshape(1, 8)
        return backend.energy_torch(candidate[:3].reshape(1, 3), full_source)

    assert torch.autograd.gradcheck(scalar, (theta,), eps=1e-6, atol=2e-6, rtol=2e-5)
    assert torch.autograd.gradgradcheck(
        scalar, (theta,), eps=1e-6, atol=3e-5, rtol=3e-4
    )


def test_custom_forward_state_does_not_retain_its_own_output_graph(backend, inputs):
    energy = backend.energy_torch(*inputs)
    # Holding the returned Tensor inside ctx.state would form an output ->
    # context -> output cycle retaining every factor until cyclic collection.
    pending = [energy.grad_fn]
    while pending:
        node = pending.pop()
        if hasattr(node, "state"):
            assert node.state.energy.grad_fn is None
            assert node.state.energy is not energy
            break
        pending.extend(child for child, _ in node.next_functions if child is not None)
    else:
        pytest.fail("Custom scalar state was not reached from the returned energy.")


def test_custom_scalar_upgrades_derivatives_lazily_without_refactor(backend, inputs):
    positions, source = inputs
    energy = backend.energy_torch(positions, source)
    assert _FakeOperators.derivative_count == 0
    gradient = torch.autograd.grad(energy, (positions, source), create_graph=True)
    assert _FakeOperators.derivative_count == 1
    torch.autograd.grad(
        gradient,
        (positions, source),
        grad_outputs=(torch.ones_like(positions), torch.ones_like(source)),
    )
    assert _FakeOperators.derivative_count == 1


def test_regular_first_backward_stops_at_order_one(backend, inputs):
    positions, source = inputs
    energy = backend.energy_torch(positions, source)
    torch.autograd.grad(energy, (positions, source), create_graph=False)
    assert _FakeOperators.derivative_count == 1


def test_cross_term_and_nonlinear_rhs_are_required(backend, inputs):
    positions, source = (value.detach() for value in inputs)
    state = backend._linearize(positions, source, derivative_order=2)
    exact = backend.hessian_partial(positions, source)
    mutant_without_cross = state.h0 + state.K.mT @ _solve_a(state, state.T)
    assert torch.linalg.vector_norm(exact - mutant_without_cross) > 1e-4
    assert torch.linalg.vector_norm(state.h0) > 1e-4


def _solve_a(state, rhs):
    return torch.linalg.lu_solve(state.lu_a, state.piv_a, rhs)


def test_contract_failures_resource_and_immutable_diagnostics(backend, inputs):
    positions, source = (value.detach() for value in inputs)
    bad = source.clone()
    bad[0, 1] = 1e-300
    with pytest.raises(ValueError, match="exact zeros"):
        backend.energy_torch(positions, bad)
    with pytest.raises(ValueError, match="shapes"):
        backend.energy_torch(positions.repeat(2, 1), source.repeat(2, 1))
    with pytest.raises(MemoryError, match="preflight"):
        backend.preflight_resources(derivative_order=2, limit_bytes=1)
    resource = backend.estimate_resources(derivative_order=2)
    assert resource.conservative_peak_bytes == (
        resource.retained_matrix_bytes
        + resource.geometry_partial_bytes
        + resource.derivative_array_bytes
        + resource.tangent_solution_bytes
        + resource.harmonic_tile_bytes
        + resource.coefficient_table_bytes
        + resource.h0_bytes
        + resource.workspace_bytes
    )
    assert resource.retained_matrix_bytes == (
        resource.operator_matrix_bytes + resource.factorization_bytes
    )
    assert resource.geometry_partial_bytes == (
        resource.geometry_pair_grid_bytes
        + resource.geometry_node_field_bytes
        + resource.geometry_transient_bytes
    )
    assert resource.host_peak_bytes + resource.device_peak_bytes == (
        resource.conservative_peak_bytes
    )
    production_tile = response_module.TorchDDPCMResponse._resource_estimate(
        1, 15, 1202, 2, 4_000_000_000
    ).harmonic_tile_bytes
    q = (15 + 1) ** 2
    internal_q = (15 + 3) ** 2
    retained_contracted_results = 8 * 1202 * (7 * q + internal_q + 18)
    assert production_tile == 23_005_568
    assert production_tile > retained_contracted_results
    assert resource.coefficient_table_bytes == sum(
        derivative_table_nbytes(backend.lmax, kind)
        + contracted_hessian_table_nbytes(backend.lmax, kind)
        for kind in ("regular", "irregular")
    )
    diagnostics = backend.diagnostics(positions, source)
    with pytest.raises(TypeError):
        diagnostics["provider_id"] = "changed"
    assert diagnostics["provider_id"] == response_module.DDPCM_RESPONSE_PROVIDER_ID
    assert set(diagnostics["solid_harmonic_contracted_hessian_tables"]) == {
        "regular",
        "irregular",
    }
    with pytest.raises(TypeError):
        diagnostics["solid_harmonic_contracted_hessian_tables"]["regular"] = "changed"
    assert len(diagnostics["solve_records"]) == 4
    assert all(
        max(record["relative_residuals"]) <= diagnostics["solve_residual_limit"]
        for record in diagnostics["solve_records"]
    )


def test_resource_model_is_named_ordered_and_device_attributed():
    estimates = [
        response_module.TorchDDPCMResponse._resource_estimate(
            20, 15, 1202, order, 4_000_000_000, "cpu"
        )
        for order in range(3)
    ]
    assert len({item.geometry_pair_grid_bytes for item in estimates}) == 1
    assert len({item.geometry_node_field_bytes for item in estimates}) == 1
    assert len({item.geometry_transient_bytes for item in estimates}) == 1
    assert estimates[0].derivative_array_bytes == 0
    assert estimates[1].derivative_array_bytes > 0
    assert estimates[1].tangent_solution_bytes == 0
    assert estimates[2].tangent_solution_bytes > 0
    assert all(item.within_limit for item in estimates)
    assert not response_module.TorchDDPCMResponse._resource_estimate(
        50, 15, 1202, 0, 4_000_000_000
    ).within_limit
    cuda = response_module.TorchDDPCMResponse._resource_estimate(
        20, 15, 1202, 2, 4_000_000_000, "cuda"
    )
    assert cuda.host_peak_bytes == cuda.coefficient_table_bytes
    assert cuda.host_peak_bytes + cuda.device_peak_bytes == cuda.conservative_peak_bytes


def test_inactive_signed_zero_is_accepted_and_stays_inactive(backend, inputs):
    positions, source = (value.detach() for value in inputs)
    signed_zero = source.clone()
    signed_zero[:, (1, 5, 6, 7)] = -0.0
    expected = backend.energy_torch(positions, source)
    actual = backend.energy_torch(positions, signed_zero)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    _, source_gradient = backend.gradient_partial(positions, signed_zero)
    assert torch.count_nonzero(source_gradient[:, (1, 5, 6, 7)]) == 0


def test_distinct_identity_and_no_provider_factor_cache(backend, inputs):
    from maple.solvation.continuum.torch_ddpcm import TORCH_DDPCM_PROVIDER_ID

    assert backend.provider_id != TORCH_DDPCM_PROVIDER_ID
    assert backend.configuration_sha256() != backend.provenance_sha256
    assert not any("lu" in name or "factor" in name for name in backend.__slots__)
    positions, source = (value.detach() for value in inputs)
    backend.gradient_partial(positions, source)
    assert _FakeOperators.build_count == 1


def test_exactly_two_svd_and_two_lu_factorizations(monkeypatch, backend, inputs):
    counts = {"svd": 0, "lu": 0}
    original_svd = torch.linalg.svdvals
    original_lu = torch.linalg.lu_factor

    def counted_svd(*args, **kwargs):
        counts["svd"] += 1
        return original_svd(*args, **kwargs)

    def counted_lu(*args, **kwargs):
        counts["lu"] += 1
        return original_lu(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svdvals", counted_svd)
    monkeypatch.setattr(torch.linalg, "lu_factor", counted_lu)
    positions, source = (value.detach() for value in inputs)
    state = backend._linearize(positions, source, derivative_order=2)
    state.hessian_partial()
    state.hvp_partial(torch.ones(7))
    assert counts == {"svd": 2, "lu": 2}
    records = state.diagnostics()["solve_records"]
    assert len(records) == 10
    assert len(records[4]["relative_residuals"]) == 7
    assert len(records[5]["relative_residuals"]) == 7


def test_stability_residual_and_configuration_drift_fail_closed(
    monkeypatch, backend, inputs
):
    positions, source = (value.detach() for value in inputs)

    def unstable(*args, **kwargs):
        raise RuntimeError("rank deficient or ill-conditioned")

    monkeypatch.setattr(
        response_module.TorchDDPCM, "_validate_matrix_stability", unstable
    )
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        backend._linearize(positions, source)
    monkeypatch.undo()

    original_residual = response_module._relative_residual_columns
    monkeypatch.setattr(
        response_module, "_relative_residual_columns", lambda *args: (1.0,)
    )
    with pytest.raises(RuntimeError, match="residual exceeded"):
        backend._linearize(positions, source)
    monkeypatch.setattr(
        response_module, "_relative_residual_columns", original_residual
    )

    energy = backend.energy_torch(
        positions.clone().requires_grad_(True), source.clone().requires_grad_(True)
    )
    original_payload = response_module.TorchDDPCMResponse._configuration_payload

    def drifted(self):
        payload = original_payload(self)
        payload["eta"] = payload["eta"] + 1e-6
        return payload

    monkeypatch.setattr(
        response_module.TorchDDPCMResponse, "_configuration_payload", drifted
    )
    with pytest.raises(RuntimeError, match="configuration changed"):
        energy.backward()


def _response_state(energy):
    pending = [energy.grad_fn]
    while pending:
        node = pending.pop()
        if hasattr(node, "state"):
            return node.state
        pending.extend(child for child, _ in node.next_functions if child is not None)
    pytest.fail("Custom scalar state was not reached from the returned energy.")


@pytest.mark.parametrize(
    "path",
    (
        "energy",
        "z",
        "A",
        "C",
        "L",
        "lu_a",
        "lu_l",
        "piv_a",
        "piv_l",
        "v0",
        "primal.L",
        "primal.D",
        "primal.rhs",
        "primal.psi",
        "primal.geometry.positions",
        "primal.source_active",
    ),
)
@pytest.mark.parametrize("mutation", ("in_place", "replacement"))
def test_bound_forward_state_tensor_mutation_fails_closed(
    backend, inputs, path, mutation
):
    positions, source = (
        value.detach().clone().requires_grad_(True) for value in inputs
    )
    energy = backend.energy_torch(positions, source)
    state = _response_state(energy)
    owner = state
    pieces = path.split(".")
    for piece in pieces[:-1]:
        owner = getattr(owner, piece)
    name = pieces[-1]
    tensor = getattr(owner, name)
    if mutation == "in_place":
        with torch.no_grad():
            tensor.add_(1)
    else:
        setattr(owner, name, tensor.clone())
    with pytest.raises(RuntimeError, match="modified by an inplace|response tensor"):
        energy.backward()


@pytest.mark.parametrize("name", ("h0", "T", "Q", "V", "K", "gradient"))
@pytest.mark.parametrize("mutation", ("in_place", "replacement"))
def test_bound_upgraded_response_array_mutation_fails_closed(
    backend, inputs, name, mutation
):
    positions, source = (
        value.detach().clone().requires_grad_(True) for value in inputs
    )
    energy = backend.energy_torch(positions, source)
    gradient = torch.autograd.grad(energy, (positions, source), create_graph=True)
    state = _response_state(gradient[0])
    tensor = getattr(state, name)
    if mutation == "in_place":
        with torch.no_grad():
            tensor.add_(1.0)
    else:
        setattr(state, name, tensor.clone())
    with pytest.raises(RuntimeError, match="modified by an inplace|response tensor"):
        torch.autograd.grad(
            gradient,
            (positions, source),
            grad_outputs=(torch.ones_like(positions), torch.ones_like(source)),
        )


@pytest.mark.parametrize(
    "name,replacement",
    (
        ("positions_sha256", "1" * 64),
        ("source_sha256", "2" * 64),
        ("source_active_sha256", "3" * 64),
        ("configuration_sha256", "4" * 64),
        ("atom_count", 2),
        ("derivative_order", 2),
        ("solve_residual_limit", 1.0),
        ("factor_policy", "changed"),
        ("topology_sha256", "5" * 64),
    ),
)
def test_bound_non_tensor_control_reassignment_fails_closed(
    backend, inputs, name, replacement
):
    positions, source = (value.detach() for value in inputs)
    state = backend._linearize(positions, source, derivative_order=0)
    setattr(state, name, replacement)
    with pytest.raises(RuntimeError, match="controls changed"):
        state.gradient_partial()


def test_diagnostics_and_solve_record_contents_are_integrity_bound(backend, inputs):
    positions, source = (value.detach() for value in inputs)
    state = backend._linearize(positions, source, derivative_order=0)
    with pytest.raises(TypeError):
        state.diagnostics_base["solve_residual_limit"] = 1.0
    state.solve_records[0]["maximum_relative_residual"] = 1.0
    with pytest.raises(RuntimeError, match="controls changed"):
        state.gradient_partial()


@pytest.mark.parametrize("target", ("provider", "diagnostics", "records", "topology"))
def test_bound_control_objects_and_topology_metadata_fail_closed(
    backend, inputs, target
):
    positions, source = (value.detach() for value in inputs)
    state = backend._linearize(positions, source, derivative_order=0)
    if target == "provider":
        state.provider = object()
    elif target == "diagnostics":
        state.diagnostics_base = dict(state.diagnostics_base)
    elif target == "records":
        state.solve_records = list(state.solve_records)
    else:
        state.primal.geometry.topology = DDPCMTopologyCertificate(
            "changed", "0" * 64, 1, 0, 0, 1.0, None
        )
    with pytest.raises(RuntimeError, match="controls changed"):
        state.gradient_partial()


def test_real_low_degree_scalar_gradient_and_hessian_match_dense_oracle():
    from maple.solvation.continuum.torch_ddpcm import TorchDDPCM

    symbols = ("C", "O")
    radii = (1.7, 1.5)
    kwargs = dict(dielectric=78.39, lmax=2, n_lebedev=50)
    response = response_module.TorchDDPCMResponse(symbols, radii, **kwargs)
    dense = TorchDDPCM(symbols, radii, **kwargs)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.8, 0.23, 0.12]], requires_grad=True)
    active = torch.tensor([[0.4, 0.1, -0.1, 0.2], [-0.4, -0.05, 0.12, -0.18]])
    source = torch.zeros((2, 8))
    source[:, (0, 2, 3, 4)] = active
    source.requires_grad_(True)
    actual = response.energy_torch(positions, source)
    expected = dense.energy_torch(positions, source)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0.0)
    actual_gradient = torch.autograd.grad(actual, (positions, source))
    expected_gradient = torch.autograd.grad(expected, (positions, source))
    torch.testing.assert_close(
        actual_gradient[0], expected_gradient[0], atol=2e-10, rtol=0.0
    )
    torch.testing.assert_close(
        actual_gradient[1], expected_gradient[1], atol=2e-10, rtol=0.0
    )

    theta = torch.cat((positions.detach().reshape(-1), active.reshape(-1)))

    def dense_theta(candidate):
        full = candidate.new_zeros((2, 8))
        full[:, (0, 2, 3, 4)] = candidate[6:].reshape(2, 4)
        return dense.energy_torch(candidate[:6].reshape(2, 3), full)

    expected_hessian = torch.autograd.functional.hessian(dense_theta, theta)
    actual_hessian = response.hessian_partial(positions.detach(), source.detach())
    torch.testing.assert_close(actual_hessian, expected_hessian, atol=2e-8, rtol=0.0)
