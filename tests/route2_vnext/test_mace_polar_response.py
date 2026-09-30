"""Total-PES stationary-response composition and result-contract tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")

from maple.solvation.derivatives.response import ResponseHessianEvaluation
from maple.solvation.experimental.mace_polar_response import MACEPolarResponsePES

_ACTIVE = (0, 2, 3, 4)


class _NonlinearNeutralModel:
    device = "cpu"
    dtype = torch.float64

    def __init__(self):
        self.calls = 0
        self.digest = "a" * 64

    def configuration_sha256(self):
        return self.digest

    def energy_source_torch(self, atoms, positions):
        del atoms
        self.calls += 1
        x, y, z = positions.unbind(-1)
        charge_seed = x.square() + 0.3 * y * z
        charge = charge_seed - charge_seed.mean()
        learned = torch.stack(
            (charge, y.square() + x * z, z.square() - 0.2 * x * y, x.square() + y),
            dim=-1,
        )
        vacuum = 0.07 * positions.square().sum() + 0.01 * positions.pow(3).sum()
        return vacuum, learned


class _State:
    def __init__(self, positions, source):
        self._positions = positions.detach().clone().requires_grad_(True)
        self._source = source.detach().clone().requires_grad_(True)
        self.energy = _ResponseContinuum.scalar(self._positions, self._source)

    def _theta_energy(self, theta):
        n = len(self._positions)
        positions = theta[: 3 * n].reshape(n, 3)
        active = theta[3 * n :].reshape(n, 4)
        source = torch.zeros((n, 8), dtype=theta.dtype, device=theta.device)
        source[:, _ACTIVE] = active
        return _ResponseContinuum.scalar(positions, source)

    def gradient_partial(self):
        g_r, g_s = torch.autograd.grad(self.energy, (self._positions, self._source))
        return g_r.detach(), g_s.detach()

    def hessian_partial(self):
        theta = (
            torch.cat(
                (self._positions.reshape(-1), self._source[:, _ACTIVE].reshape(-1))
            )
            .detach()
            .requires_grad_(True)
        )
        return torch.autograd.functional.hessian(self._theta_energy, theta).detach()

    def hvp_partial(self, direction):
        return self.hessian_partial() @ direction

    def diagnostics(self):
        return {"backend": "synthetic-stationary-response", "residual": 0.0}


class _ResponseContinuum:
    def __init__(self):
        self.digest = "b" * 64
        self.linearize_calls = 0

    def configuration_sha256(self):
        return self.digest

    @staticmethod
    def scalar(positions, source):
        active = source[:, _ACTIVE]
        r = positions.reshape(-1)
        s = active.reshape(-1)
        return (
            0.13 * r.pow(3).sum()
            + 0.11 * s.square().sum()
            + 0.017 * s.pow(3).sum()
            + 0.09 * (positions * active[:, 1:]).sum()
            + 0.025 * s[:-1].dot(s[1:])
        )

    def energy_torch(self, positions, source):
        return self.scalar(positions, source)

    def _linearize(self, positions, source, derivative_order=2):
        assert derivative_order == 2
        self.linearize_calls += 1
        return _State(positions, source)


class _CubicCDS:
    device = "cpu"

    def __init__(self):
        self.calls = 0
        self.digest = "c" * 64

    def configuration_sha256(self):
        return self.digest

    def energy_torch(self, positions):
        self.calls += 1
        return 0.03 * positions.pow(3).sum()


@pytest.fixture
def atoms():
    return Atoms(
        "OH2",
        positions=[
            [0.18, -0.22, 0.31],
            [1.08, -0.22, 0.31],
            [-0.12, 0.63, 0.31],
        ],
        info={"charge": 0, "mult": 1},
    )


@pytest.fixture
def pes():
    return MACEPolarResponsePES._for_testing(
        symbols=("O", "H", "H"),
        model=_NonlinearNeutralModel(),
        continuum=_ResponseContinuum(),
        solvent_term=_CubicCDS(),
        device="cpu",
    )


def _direct_total(pes, atoms, positions):
    vacuum, learned = pes.model.energy_source_torch(atoms, positions)
    source = torch.zeros((len(positions), 8), dtype=positions.dtype)
    source[:, _ACTIVE] = learned
    return (
        vacuum
        + pes.continuum.energy_torch(positions, source)
        + pes.solvent_term.energy_torch(positions)
    )


@pytest.mark.parametrize(
    "method",
    [
        "get_potential_energy",
        "get_solvation_energy",
        "get_forces",
        "hessian_vector_product",
        "evaluate_hessian",
    ],
)
def test_default_dtype_drift_fails_before_model_execution(pes, atoms, method):
    original = torch.get_default_dtype()
    changed = torch.float64 if original == torch.float32 else torch.float32
    try:
        torch.set_default_dtype(changed)
        arguments = (atoms,)
        if method == "hessian_vector_product":
            arguments += (np.ones((len(atoms), 3)),)
        with pytest.raises(RuntimeError, match="configuration"):
            getattr(pes, method)(*arguments)
        assert pes.model.calls == 0
    finally:
        torch.set_default_dtype(original)


def test_result_distinguishes_tensor_dtype_from_ambient_default(pes, atoms):
    original = torch.get_default_dtype()
    result = pes.evaluate_hessian(atoms)
    assert result.diagnostics["dtype"] == "float64"
    assert result.diagnostics["torch_default_dtype"] == str(original)
    assert torch.get_default_dtype() == original


def test_nonlinear_source_outer_chain_matches_direct_scalar_hessian(pes, atoms):
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)
    expected = torch.autograd.functional.hessian(
        lambda value: _direct_total(pes, atoms, value), positions
    ).reshape(9, 9)
    pes.model.calls = pes.solvent_term.calls = 0

    result = pes.evaluate_hessian(atoms)

    assert isinstance(result, ResponseHessianEvaluation)
    np.testing.assert_allclose(result.hessian_eV_per_A2, expected.detach(), atol=5e-11)
    assert pes.model.calls == 1
    assert pes.solvent_term.calls == 1
    assert pes.continuum.linearize_calls == 1
    assert result.derivative_method == "structured-stationary-response"
    assert result.numerical_uncertainty_eV_per_A2 is None
    assert result.diagnostics["source_jacobian_frobenius_norm"] > 0.0
    assert result.diagnostics["source_coordinate_path"] is True
    assert result.diagnostics["continuum_direct_coordinate_path"] is True
    witnesses = result.diagnostics["source_witnesses"]
    assert witnesses["inactive_exact_zero"]
    assert witnesses["neutral_charge_pass"]
    assert witnesses["active_columns"] == _ACTIVE
    for key in (
        "source_jacobian_frobenius_norm",
        "source_jvp_norm",
        "source_vjp_norm",
        "mixed_R_source_max_abs",
        "weighted_source_curvature_hvp_norm",
    ):
        assert np.isfinite(witnesses[key]) and witnesses[key] > 1e-12


def test_source_curvature_witness_is_the_frozen_weight_not_total_curvature(pes, atoms):
    result = pes.evaluate_hessian(atoms)
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)
    _, learned = pes.model.energy_source_torch(atoms, positions)
    source = learned.new_zeros((len(atoms), 8))
    source[:, _ACTIVE] = learned
    independent_source = source.detach().requires_grad_(True)
    scalar = pes.continuum.scalar(positions.detach(), independent_source)
    weight = torch.autograd.grad(scalar, independent_source)[0][:, _ACTIVE].detach()
    expected = torch.autograd.functional.hessian(
        lambda r: (weight * pes.model.energy_source_torch(atoms, r)[1]).sum(),
        positions,
    ).reshape(9, 9)
    probe = torch.linspace(-1, 1, 9, dtype=torch.float64)
    probe /= torch.linalg.vector_norm(probe)
    assert result.diagnostics["source_witnesses"][
        "weighted_source_curvature_hvp_norm"
    ] == pytest.approx(float(torch.linalg.vector_norm(expected @ probe)), abs=1e-12)


def test_energy_force_and_hvp_use_the_same_live_custom_scalar(pes, atoms):
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)
    total = _direct_total(pes, atoms, positions)
    gradient = torch.autograd.grad(total, positions, create_graph=True)[0]
    direction = torch.linspace(-0.4, 0.6, 9, dtype=torch.float64).reshape(3, 3)
    expected_hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]

    assert pes.get_potential_energy(atoms) == pytest.approx(float(total.detach()))
    np.testing.assert_allclose(pes.get_forces(atoms), -gradient.detach(), atol=1e-12)
    np.testing.assert_allclose(
        pes.hessian_vector_product(atoms, direction.numpy()),
        expected_hvp.detach(),
        atol=1e-12,
    )


def test_response_result_is_hash_bound_and_recursively_immutable(pes, atoms):
    result = pes.evaluate_hessian(atoms)
    assert len(result.evaluation_sha256) == 64
    assert result.dtype == "float64"
    with pytest.raises(ValueError):
        result.hessian_eV_per_A2[0, 0] = 7.0
    with pytest.raises(TypeError):
        result.diagnostics["new"] = 1
    with pytest.raises(ValueError, match="evaluation_sha256"):
        replace(result, energy_eV=result.energy_eV + 1.0)
    with pytest.raises(ValueError, match="cannot be relabeled"):
        replace(
            result,
            scalar_contract_id=(
                "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-"
                "torch-cpu-v3"
            ),
            evaluation_sha256="",
        )


def test_component_and_implementation_drift_fail_closed(pes, atoms, monkeypatch):
    pes.model.digest = "d" * 64
    with pytest.raises(RuntimeError, match="configuration"):
        pes.get_potential_energy(atoms)
    pes.model.digest = "a" * 64
    monkeypatch.setattr(
        "maple.solvation.experimental.mace_polar_response_core.source_files_sha256",
        lambda files: (("mutated", "f" * 64),),
    )
    with pytest.raises(RuntimeError, match="configuration"):
        pes.get_potential_energy(atoms)


def test_neutral_source_contract_fails_closed(pes, atoms, monkeypatch):
    original = _NonlinearNeutralModel.energy_source_torch

    def charged(self, atoms, positions):
        energy, source = original(self, atoms, positions)
        return energy, source + torch.tensor([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(_NonlinearNeutralModel, "energy_source_torch", charged)
    with pytest.raises(RuntimeError, match="neutral charge"):
        pes.get_forces(atoms)


@pytest.mark.parametrize("detached", ["source", "geometry"])
def test_both_continuum_graph_paths_are_required(pes, atoms, monkeypatch, detached):
    original = _ResponseContinuum.energy_torch

    def broken(self, positions, source):
        if detached == "source":
            source = source.detach()
        else:
            positions = positions.detach()
        return original(self, positions, source)

    monkeypatch.setattr(_ResponseContinuum, "energy_torch", broken)
    with pytest.raises(RuntimeError, match="source|coordinate path"):
        pes.get_forces(atoms)


def test_private_injection_never_receives_official_identity(pes):
    assert pes.provider_id.startswith("unregistered-engineering-test")
    assert pes.profile_id.startswith("unregistered-engineering-test")
    assert pes.scalar_contract_id.startswith("unregistered-engineering-test")
    with pytest.raises(ValueError, match="builder|arbitrary components"):
        MACEPolarResponsePES(
            symbols=pes.symbols,
            model=pes.model,
            continuum=pes.continuum,
            solvent_term=pes.solvent_term,
        )


@pytest.mark.parametrize("override", ["lmax", "radii", "eta", "dtype", "probe"])
def test_programmatic_builder_rejects_scientific_overrides(override):
    from maple.solvation.experimental.mace_polar_response import (
        build_smd_mace_polar_response_pes,
    )

    with pytest.raises(TypeError, match="unexpected keyword"):
        build_smd_mace_polar_response_pes(
            ("O", "H", "H"), solvent="water", **{override: 0.2}
        )


def test_resource_preflight_preserves_requested_derivative_order(monkeypatch):
    import maple.solvation.experimental.mace_polar_response_core as module

    @dataclass
    class Estimate:
        conservative_peak_bytes: int = 100_000_000
        derivative_order: int = 1

    class Continuum:
        def preflight_resources(self, *, derivative_order, atom_count, limit_bytes):
            assert derivative_order == 1
            assert atom_count == 10
            assert limit_bytes == 4_000_000_000
            return Estimate()

    monkeypatch.setattr(module, "_available_host_memory_bytes", lambda: 8 * 1024**3)
    record = module._response_resource_preflight(Continuum(), "cpu", 10, 1)
    assert record["continuum"]["derivative_order"] == 1
    assert record["model_runtime_reserve_bytes"] == 2 * 1024**3 + 10 * 64 * 1024**2
    assert record["host_runtime_reserve_bytes"] == record["model_runtime_reserve_bytes"]
