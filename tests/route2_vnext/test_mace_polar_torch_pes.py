"""Full scalar graph tests, independent of the continuum/surface numerics."""

import numpy as np
import pytest
from ase import Atoms
from dataclasses import dataclass

torch = pytest.importorskip("torch")

from maple.solvation.experimental.mace_polar_torch import MACEPolarTorchPES


class PolynomialModel:
    device = "cpu"
    dtype = torch.float64

    def configuration_sha256(self):
        return "a" * 64

    def energy_source_torch(self, atoms, positions):
        x, y, z = positions.unbind(-1)
        source = torch.stack((x * 0, y**2, z**2, x**2), dim=-1)
        return 0.07 * positions.square().sum(), source


class PolynomialContinuum:
    def configuration_sha256(self):
        return "b" * 64

    def energy_torch(self, positions, source):
        assert source.shape == (len(positions), 8)
        assert torch.count_nonzero(source[:, [1, 5, 6, 7]]) == 0
        cartesian_dipoles = source[:, [4, 2, 3]]
        return 0.2 * source.square().sum() + 0.1 * (positions * cartesian_dipoles).sum()


class PolynomialCDS:
    def configuration_sha256(self):
        return "c" * 64

    def energy_torch(self, positions):
        return 0.05 * positions.pow(3).sum()


@pytest.fixture
def atoms():
    return Atoms(
        "OH2",
        positions=[[0.1, 0.2, 0.1], [1.05, 0.2, 0.1], [-0.14, 1.12, 0.1]],
        info={"charge": 0, "mult": 1},
    )


@pytest.fixture
def pes():
    return MACEPolarTorchPES._for_testing(
        symbols=("O", "H", "H"),
        model=PolynomialModel(),
        continuum=PolynomialContinuum(),
        solvent_term=PolynomialCDS(),
        device="cpu",
    )


def test_full_chain_has_exact_polynomial_energy_force_hessian(pes, atoms):
    r = atoms.positions
    # Source geometry and mixed R/source terms give the cubic/quartic terms.
    energy = np.sum(0.07 * r**2 + 0.15 * r**3 + 0.2 * r**4)
    forces = -(0.14 * r + 0.45 * r**2 + 0.8 * r**3)
    hessian = np.diag((0.14 + 0.9 * r + 2.4 * r**2).ravel())
    assert pes.get_potential_energy(atoms) == pytest.approx(energy, abs=1e-13)
    np.testing.assert_allclose(pes.get_forces(atoms), forces, atol=1e-12)
    evaluated = pes.evaluate_hessian(atoms)
    assert evaluated.energy_eV == pytest.approx(energy, abs=1e-13)
    np.testing.assert_allclose(evaluated.forces_eV_per_A, forces, atol=1e-12)
    np.testing.assert_allclose(evaluated.hessian_eV_per_A2, hessian, atol=1e-12)
    assert evaluated.numerical_uncertainty_eV_per_A2 is None
    assert sum(evaluated.component_energies_eV.values()) == pytest.approx(energy)


def test_hvp_is_direct_autograd_not_a_dense_hessian_fallback(pes, atoms, monkeypatch):
    direction = np.arange(9, dtype=float).reshape(3, 3) / 7
    monkeypatch.setattr(
        MACEPolarTorchPES,
        "evaluate_hessian",
        lambda *_: pytest.fail("HVP must not construct the full Hessian"),
    )
    expected = (0.14 + 0.9 * atoms.positions + 2.4 * atoms.positions**2) * direction
    np.testing.assert_allclose(
        pes.hessian_vector_product(atoms, direction), expected, atol=1e-12
    )


def test_component_failure_is_not_replaced_by_legacy_or_zero(pes, atoms, monkeypatch):
    def broken(*_):
        raise RuntimeError("component topology is not differentiable")

    monkeypatch.setattr(PolynomialCDS, "energy_torch", broken)
    with pytest.raises(RuntimeError, match="topology"):
        pes.evaluate_hessian(atoms)


def test_source_configuration_drift_is_rejected(pes, atoms, monkeypatch):
    monkeypatch.setattr(PolynomialModel, "configuration_sha256", lambda _: "d" * 64)
    with pytest.raises(RuntimeError, match="configuration"):
        pes.get_potential_energy(atoms)


def test_geometry_identity_and_hvp_shape_fail_closed(pes, atoms):
    changed = atoms.copy()
    changed[0].symbol = "N"
    with pytest.raises(ValueError, match="symbols|elements"):
        pes.get_potential_energy(changed)
    with pytest.raises(ValueError, match="direction"):
        pes.hessian_vector_product(atoms, np.ones(2))


def test_no_tensor_component_can_return_a_detached_energy(pes, atoms, monkeypatch):
    monkeypatch.setattr(
        PolynomialContinuum,
        "energy_torch",
        lambda self, r, source: (0.2 * source.square().sum()).detach(),
    )
    with pytest.raises(RuntimeError, match="graph|differentiab"):
        pes.evaluate_hessian(atoms)


@pytest.mark.parametrize("detached", ["source", "geometry", "new_leaf"])
def test_requires_grad_is_not_a_substitute_for_graph_connectivity(
    pes, atoms, monkeypatch, detached
):
    def broken(self, positions, source):
        if detached == "source":
            source = source.detach()
        elif detached == "geometry":
            positions = positions.detach()
        else:
            return source.square().sum().detach().requires_grad_(True)
        return (
            0.2 * source.square().sum() + 0.1 * (positions * source[:, [4, 2, 3]]).sum()
        )

    monkeypatch.setattr(PolynomialContinuum, "energy_torch", broken)
    with pytest.raises(RuntimeError, match="graph"):
        pes.evaluate_hessian(atoms)


def test_registered_identity_is_builder_only_and_tests_are_unregistered(pes):
    from maple.solvation.api.scalar_registry import SCALAR_REGISTRY

    assert pes.scalar_contract_id not in SCALAR_REGISTRY
    with pytest.raises(ValueError, match="arbitrary components"):
        MACEPolarTorchPES(
            symbols=pes.symbols,
            model=pes.model,
            continuum=pes.continuum,
            solvent_term=pes.solvent_term,
        )


@pytest.mark.parametrize(
    "override", ["lmax", "radii", "eta", "solver_tolerance", "probe"]
)
def test_registered_builder_does_not_accept_scientific_overrides(override):
    from maple.solvation.experimental.mace_polar_torch import (
        build_smd_mace_polar_torch_pes,
    )

    with pytest.raises(TypeError, match="unexpected keyword"):
        build_smd_mace_polar_torch_pes(
            ("O", "H", "H"), solvent="water", **{override: 0.2}
        )


def test_second_order_memory_preflight_includes_model_and_runs_before_graph(
    monkeypatch,
):
    import maple.solvation.experimental.mace_polar_torch as module

    @dataclass
    class Estimate:
        conservative_peak_bytes: int = 100_000_000
        derivative_order: int = 2

    class Continuum:
        def preflight_resources(self, *, derivative_order, limit_bytes):
            assert derivative_order == 2
            assert limit_bytes == 4_000_000_000
            return Estimate()

    monkeypatch.setattr(module, "_available_host_memory_bytes", lambda: 2 * 1024**3)
    with pytest.raises(MemoryError, match="including runtime reserve"):
        module._second_order_resource_preflight(Continuum(), "cpu")
    monkeypatch.setattr(module, "_available_host_memory_bytes", lambda: 8 * 1024**3)
    record = module._second_order_resource_preflight(Continuum(), "cpu")
    assert record["estimated_total_bytes"] == 100_000_000 + 2 * 1024**3
    assert record["continuum"]["derivative_order"] == 2


def test_missing_graph_inspection_api_fails_explicitly_before_construction(monkeypatch):
    from maple.solvation.experimental.mace_polar_torch import (
        build_smd_mace_polar_torch_pes,
    )

    monkeypatch.delattr(torch.autograd.graph, "get_gradient_edge")
    with pytest.raises(RuntimeError, match=r"PyTorch >=2.2"):
        build_smd_mace_polar_torch_pes(("O", "H", "H"), solvent="water")
