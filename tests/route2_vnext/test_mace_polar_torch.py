from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.solvation.models.mace_polar import (
    build_official_mace_polar_1_m_adapter,
)
from maple.solvation.models.mace_polar_torch import (
    MACEPolarTorchGraphAdapter,
    _TorchModelOnlyMACEPolCalculator,
)


class _Batch:
    def __init__(self, values):
        self._values = values

    def to_dict(self):
        return dict(self._values)


class _FakeMACE:
    def _atoms_to_batch(self, atoms):
        return _Batch(
            {
                "positions": torch.as_tensor(atoms.positions, dtype=torch.float64),
                "batch": torch.zeros(len(atoms), dtype=torch.long),
            }
        )


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float64), requires_grad=False
        )

    def forward(self, batch, **kwargs):
        del kwargs
        positions = batch["positions"]
        x, y, z = positions.unbind(dim=1)
        energy = torch.sum(0.3 * x**3 + 0.4 * y**2 + x * z + 0.2 * y * z**2)
        density = torch.stack(
            (
                x**2 + 0.2 * y * z,
                y + x * z,
                z**2 + x * y,
                x + y**2,
            ),
            dim=1,
        )
        return {"energy": energy.reshape(1), "density_coefficients": density}


class _CenteringEvaluator:
    is_default = False
    profile = "test-centering-evaluator-v1"

    def __init__(self):
        self.received_positions = None

    def prepare_batch(self, batch, *, r_max):
        del r_max
        self.received_positions = batch["positions"]
        result = dict(batch)
        result["positions"] = batch["positions"] - batch["positions"].mean(
            dim=0, keepdim=True
        )
        return result

    def forward_model(self, model, batch, **kwargs):
        return model(batch, **kwargs)


def _atoms():
    return Atoms(
        "H2O",
        positions=np.asarray(
            [[0.10, -0.20, 0.30], [0.95, 0.15, -0.10], [-0.25, 0.88, 0.05]]
        ),
        info={"charge": 0, "mult": 1},
    )


def _calculator():
    calculator = object.__new__(_TorchModelOnlyMACEPolCalculator)
    calculator._mace = _FakeMACE()
    calculator.model = _FakeModel()
    calculator.device = torch.device("cpu")
    calculator.dtype = torch.float64
    calculator.r_max = 5.0
    calculator._long_range_evaluator = _CenteringEvaluator()
    return calculator


class _Domain:
    def validate_atoms(self, atoms):
        if len(atoms) < 1:
            raise ValueError("empty geometry")


class _BaseAdapter:
    def __init__(self, calculator):
        self._calculator = calculator
        self.device = "cpu"
        self.dtype = "float64"
        self.provider_id = "test-mace-polar-provider-v1"
        self.model_profile_id = "test-mace-polar-profile-v1"
        self.provenance = SimpleNamespace(checkpoint_sha256="b" * 64)
        self.provenance_sha256 = "c" * 64
        self.domain = _Domain()
        self.release_contract = SimpleNamespace(
            checkpoint_sha256="b" * 64,
            long_range_evaluator_profile="test-centering-evaluator-v1",
        )

    def configuration_sha256(self):
        return "a" * 64


def test_external_coordinates_reach_preparation_and_keep_energy_source_gradgrad():
    atoms = _atoms()
    calculator = _calculator()
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)

    energy, density = calculator.zero_field_energy_source_torch(atoms, positions)

    assert calculator._long_range_evaluator.received_positions is positions
    assert energy.shape == ()
    assert density.shape == (3, 4)
    energy_gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    direction = torch.linspace(-0.4, 0.5, positions.numel()).reshape_as(positions)
    energy_hvp = torch.autograd.grad(torch.sum(energy_gradient * direction), positions)[
        0
    ]
    cotangent = torch.linspace(-0.3, 0.7, density.numel()).reshape_as(density)
    source_gradient = torch.autograd.grad(
        torch.sum(density * cotangent), positions, create_graph=True
    )[0]
    source_hvp = torch.autograd.grad(torch.sum(source_gradient * direction), positions)[
        0
    ]
    assert bool(torch.isfinite(energy_hvp).all())
    assert bool(torch.isfinite(source_hvp).all())
    assert float(torch.linalg.vector_norm(energy_hvp)) > 0.0
    assert float(torch.linalg.vector_norm(source_hvp)) > 0.0


def test_zero_field_source_matches_existing_upstream_forward_at_central_geometry():
    atoms = _atoms()
    calculator = _calculator()
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)

    energy, density = calculator.zero_field_energy_source_torch(atoms, positions)
    legacy_output = calculator._model_forward(
        calculator._batch_dict(atoms),
        compute_force=False,
        compute_stress=False,
        compute_hessian=False,
    )

    torch.testing.assert_close(energy.detach(), legacy_output["energy"].sum())
    torch.testing.assert_close(density.detach(), legacy_output["density_coefficients"])


def test_changed_central_geometry_fails_closed_instead_of_using_stale_topology():
    atoms = _atoms()
    calculator = _calculator()
    changed = torch.tensor(atoms.positions, dtype=torch.float64)
    changed[1, 0] += 0.25

    with pytest.raises(ValueError, match="neighbor topology"):
        calculator.zero_field_energy_source_torch(atoms, changed)


def test_graph_adapter_preserves_model_identity_and_live_tensors():
    atoms = _atoms()
    calculator = _calculator()
    adapter = MACEPolarTorchGraphAdapter(_BaseAdapter(calculator))
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)

    energy, density = adapter.energy_source_torch(atoms, positions)

    assert len(adapter.configuration_sha256()) == 64
    assert adapter.device == "cpu"
    assert adapter.dtype == "float64"
    assert adapter.model_metadata["field_state"] == "frozen-zero-external-field"
    assert adapter.metadata()["checkpoint_sha256"] == "b" * 64
    assert energy.requires_grad
    assert density.requires_grad


def test_graph_adapter_reports_stable_neighbor_topology_certificate():
    atoms = _atoms()
    adapter = MACEPolarTorchGraphAdapter(_BaseAdapter(_calculator()))

    first = adapter.topology_diagnostics(atoms)
    second = adapter.topology_diagnostics(atoms.copy())

    assert first == second
    assert first["neighbor_pair_count"] == 3
    assert first["nonself_pair_count"] == 3
    assert first["minimum_cutoff_distance_angstrom"] > 1.0e-8
    assert len(first["pair_mask_order_sha256"]) == 64
    assert first["locally_fixed_neighbor_mask"] is True
    assert first["global_regularity_proved"] is False


def test_graph_adapter_rejects_geometry_near_neighbor_cutoff():
    atoms = Atoms(
        "H2",
        positions=np.asarray([[0.0, 0.0, 0.0], [5.0 - 5.0e-9, 0.0, 0.0]]),
        info={"charge": 0, "mult": 1},
    )
    adapter = MACEPolarTorchGraphAdapter(_BaseAdapter(_calculator()))
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)

    with pytest.raises(ValueError, match="neighbor-cutoff guard"):
        adapter.energy_source_torch(atoms, positions)


@pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_REAL_MACEPOL") != "1",
    reason="set MAPLE_ROUTE2_REAL_MACEPOL=1 in the pinned real-checkpoint job",
)
def test_real_checkpoint_zero_field_parity_and_second_derivative_canary():
    device = os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu")
    base = build_official_mace_polar_1_m_adapter(
        device=device,
        checkpoint_path=os.environ.get("ROUTE2_MACE_CHECKPOINT"),
        torch_graph=True,
    )
    assert type(base._calculator) is _TorchModelOnlyMACEPolCalculator
    adapter = MACEPolarTorchGraphAdapter(base)
    atoms = _atoms()
    positions = torch.tensor(
        atoms.positions,
        dtype=torch.float64,
        device=torch.device(device),
        requires_grad=True,
    )

    energy, density = adapter.energy_source_torch(atoms, positions)
    legacy_vacuum = base.evaluate_vacuum(atoms, need_forces=False)
    legacy_source = base.evaluate_source(
        atoms, np.zeros((len(atoms), 4)), need_fixed_field_forces=False
    )
    assert float(energy.detach().cpu()) == pytest.approx(
        legacy_vacuum.energy_eV, rel=2.0e-11, abs=2.0e-11
    )
    np.testing.assert_allclose(
        density.detach().cpu().numpy(),
        legacy_source.source,
        rtol=2.0e-11,
        atol=2.0e-11,
    )

    direction = torch.linspace(
        -0.2, 0.3, positions.numel(), dtype=positions.dtype, device=positions.device
    ).reshape_as(positions)
    source_cotangent = torch.linspace(
        -0.4, 0.5, density.numel(), dtype=density.dtype, device=density.device
    ).reshape_as(density)
    combined = energy + torch.sum(source_cotangent * density)
    gradient = torch.autograd.grad(combined, positions, create_graph=True)[0]
    hvp = torch.autograd.grad(torch.sum(gradient * direction), positions)[0]
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.isfinite(hvp).all())
