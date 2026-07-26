from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNET2_RAW_CHARGE_TOLERANCE_E,
    AIMNet2Calculator,
)


class _FakeAIMNet2(torch.nn.Module):
    cutoff = 5.0
    cutoff_lr = float("inf")

    def __init__(self, charges):
        super().__init__()
        self._charges = tuple(float(value) for value in charges)

    def forward(self, data):
        return {
            "energy": torch.tensor(
                [-12.5],
                dtype=torch.float64,
                device=data["coord"].device,
            ),
            "charges": torch.tensor(
                self._charges,
                dtype=torch.float32,
                device=data["coord"].device,
            ),
        }


def _calculator(monkeypatch, charges, *, model_name="aimnet2"):
    model = _FakeAIMNet2(charges)
    monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: model)
    return AIMNet2Calculator(
        torch.device("cpu"),
        model=model_name,
        model_path="/unused/aimnet2.pt",
    )


def _water(*, charge=0):
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.239987, 0.927297, 0.0],
        ],
        info={"charge": charge, "mult": 1},
    )


def test_charge_state_projects_only_float32_total_charge_residue(monkeypatch):
    calculator = _calculator(
        monkeypatch,
        [-0.68984449, 0.34492224, 0.34492221, 0.0],
    )

    state = calculator.charge_state(_water())

    assert state.energy_ev == pytest.approx(-12.5)
    assert state.requested_total_charge_e == pytest.approx(0.0)
    assert abs(state.raw_charge_residual_e) < 1.0e-6
    assert state.charge_projection_per_atom_e == pytest.approx(
        -state.raw_charge_residual_e / 3.0
    )
    assert state.raw_charge_sum_e != 0.0
    assert state.projected_charge_sum_e == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(
        state.charges_e,
        state.raw_charges_e + state.charge_projection_per_atom_e,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert state.raw_charges_e.flags.writeable is False
    assert state.charges_e.flags.writeable is False


def test_calculator_exposes_projected_aimnet_charges_as_an_ase_property(
    monkeypatch,
):
    calculator = _calculator(
        monkeypatch,
        [-0.68984449, 0.34492224, 0.34492221, 0.0],
    )
    atoms = _water()

    calculator.calculate(atoms, properties=["energy", "charges"])

    assert set(calculator.results) == {"energy", "free_energy", "charges"}
    assert calculator.results["charges"].shape == (3,)
    assert float(np.sum(calculator.results["charges"])) == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert calculator.last_charge_state is not None


def test_charge_state_respects_a_declared_molecular_charge(monkeypatch):
    calculator = _calculator(
        monkeypatch,
        [0.19632369, 0.40183812, 0.40183812, 0.0],
    )

    state = calculator.charge_state(_water(charge=1))

    assert state.requested_total_charge_e == pytest.approx(1.0)
    assert state.projected_charge_sum_e == pytest.approx(1.0, abs=1.0e-15)


def test_charge_state_rejects_a_material_total_charge_mismatch(monkeypatch):
    calculator = _calculator(
        monkeypatch,
        [-0.5, 0.2, 0.2, 0.0],
    )

    with pytest.raises(RuntimeError, match="total-charge constraint"):
        calculator.charge_state(_water())


def test_charge_state_fails_closed_above_the_audited_float_residue_limit(
    monkeypatch,
):
    residual = 2.0 * AIMNET2_RAW_CHARGE_TOLERANCE_E
    calculator = _calculator(
        monkeypatch,
        [-0.7 + residual, 0.35, 0.35, 0.0],
    )

    with pytest.raises(RuntimeError, match="total-charge constraint"):
        calculator.charge_state(_water())


def test_charge_state_rejects_a_nonzero_padded_sentinel(monkeypatch):
    calculator = _calculator(
        monkeypatch,
        [-0.7, 0.35, 0.35, 1.0e-3],
    )

    with pytest.raises(RuntimeError, match="padded sentinel"):
        calculator.charge_state(_water())


def test_charge_state_rejects_open_shell_use_of_the_closed_shell_model(
    monkeypatch,
):
    calculator = _calculator(
        monkeypatch,
        [-0.7, 0.35, 0.35, 0.0],
    )
    atoms = _water()
    atoms.info["mult"] = 2

    with pytest.raises(NotImplementedError, match="multiplicity 1"):
        calculator.charge_state(atoms)


def test_nse_charge_channels_fail_closed_until_their_source_is_validated(
    monkeypatch,
):
    calculator = _calculator(
        monkeypatch,
        [-0.7, 0.35, 0.35, 0.0],
        model_name="aimnet2nse",
    )

    assert "charges" not in calculator.implemented_properties
    with pytest.raises(NotImplementedError, match="NSE two-channel"):
        calculator.charge_state(_water())
