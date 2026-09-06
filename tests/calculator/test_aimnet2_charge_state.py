from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNET2_RAW_CHARGE_TOLERANCE_E,
    AIMNet2ChargePositionResponse,
    AIMNet2ChargeState,
    AIMNet2Calculator,
)
from maple.function.calculator.aimnet._aimnet2_float64_source import (
    AIMNet2ReconstructedFloat64SourceCalculator,
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


class _CoordinateResponsiveFakeAIMNet2(torch.nn.Module):
    cutoff = 5.0
    cutoff_lr = float("inf")

    def forward(self, data):
        coord = data["coord"]
        atomic_coord = coord[:-1]
        base = torch.tensor(
            [-0.6, 0.3, 0.3],
            dtype=coord.dtype,
            device=coord.device,
        )
        x_centered = atomic_coord[:, 0] - atomic_coord[:, 0].mean()
        atomic_charges = base + 0.1 * x_centered
        charges = torch.cat(
            [
                atomic_charges,
                torch.zeros(1, dtype=coord.dtype, device=coord.device),
            ]
        )
        return {
            "energy": 0.5 * torch.sum(atomic_coord**2).reshape(1),
            "charges": charges,
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


def test_charge_position_response_differentiates_energy_and_projected_charges(
    monkeypatch,
):
    model = _CoordinateResponsiveFakeAIMNet2()
    monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: model)
    calculator = AIMNet2Calculator(
        torch.device("cpu"),
        model="aimnet2",
        model_path="/unused/aimnet2.pt",
    )
    atoms = _water()
    cotangent = np.asarray([-2.0, 0.5, 1.5])

    response = calculator.charge_position_response(atoms, cotangent)

    np.testing.assert_allclose(
        response.intrinsic_energy_gradient_ev_per_angstrom,
        atoms.get_positions(),
        rtol=0.0,
        atol=2.0e-7,
    )
    expected_charge_vjp = np.zeros((3, 3))
    expected_charge_vjp[:, 0] = 0.1 * (cotangent - float(np.mean(cotangent)))
    np.testing.assert_allclose(
        response.charge_position_vjp_ev_per_angstrom,
        expected_charge_vjp,
        rtol=0.0,
        atol=2.0e-7,
    )
    assert response.charge_state.projected_charge_sum_e == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert response.charge_cotangent_ev_per_e.flags.writeable is False
    assert response.intrinsic_energy_gradient_ev_per_angstrom.flags.writeable is False
    assert response.charge_position_vjp_ev_per_angstrom.flags.writeable is False


def test_charge_position_response_rejects_invalid_cotangent(monkeypatch):
    calculator = _calculator(
        monkeypatch,
        [-0.7, 0.35, 0.35, 0.0],
    )

    with pytest.raises(ValueError, match="cotangent.*shape"):
        calculator.charge_position_response(
            _water(),
            np.zeros((3, 1)),
        )


def _float64_response_state(*, energy_ev: float = -12.5) -> AIMNet2ChargeState:
    return AIMNet2ChargeState(
        energy_ev=energy_ev,
        raw_charges_e=np.asarray([-0.7, 0.35, 0.35]),
        charges_e=np.asarray([-0.7, 0.35, 0.35]),
        requested_total_charge_e=0.0,
        raw_charge_residual_e=0.0,
        charge_projection_per_atom_e=0.0,
        model_name="aimnet2",
    )


def _float64_response(
    state: AIMNet2ChargeState,
    *,
    intrinsic_energy_gradient_ev_per_angstrom: np.ndarray,
) -> AIMNet2ChargePositionResponse:
    return AIMNet2ChargePositionResponse(
        charge_state=state,
        charge_cotangent_ev_per_e=np.zeros(3),
        intrinsic_energy_gradient_ev_per_angstrom=np.asarray(
            intrinsic_energy_gradient_ev_per_angstrom,
            dtype=float,
        ),
        charge_position_vjp_ev_per_angstrom=np.zeros((3, 3)),
    )


def test_float64_response_reports_embedded_d3_intrinsic_mismatch_when_repeat_passes():
    calculator = object.__new__(AIMNet2ReconstructedFloat64SourceCalculator)
    calculator._last_charge_state = None
    calculator._last_ordinary_decomposed_response_parity = None

    state = _float64_response_state()
    ordinary = _float64_response(
        state,
        intrinsic_energy_gradient_ev_per_angstrom=np.zeros((3, 3)),
    )
    decomposed = _float64_response(
        state,
        intrinsic_energy_gradient_ev_per_angstrom=np.asarray(
            [[2.0e-7, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
    )
    repeat = _float64_response(
        state,
        intrinsic_energy_gradient_ev_per_angstrom=np.asarray(
            [[2.0e-7, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
    )

    calculator._validated_forward = lambda atoms, requires_grad: (
        "ordinary",
        "ordinary",
    )
    repeated = iter((("decomp", "decomp"), ("repeat", "repeat")))
    calculator._validated_decomposed_forward = lambda atoms: next(repeated)

    def _response_from_output(atoms, cotangent, data, output):
        if output == "ordinary":
            return ordinary
        if output == "decomp":
            return decomposed
        if output == "repeat":
            return repeat
        raise AssertionError(output)

    calculator._response_from_output = _response_from_output

    response = calculator.charge_position_response(_water(), np.zeros(3))

    assert response is decomposed
    parity = calculator.last_ordinary_decomposed_parity()
    assert parity["energy_absolute_error_eV"] == pytest.approx(0.0)
    assert parity["charge_max_absolute_error_e"] == pytest.approx(0.0)
    assert parity["charge_vjp_max_absolute_error_eV_per_A"] == pytest.approx(0.0)
    assert parity["intrinsic_gradient_max_absolute_error_eV_per_A"] == pytest.approx(
        0.0
    )
    assert parity[
        "repeat_intrinsic_gradient_max_absolute_error_eV_per_A"
    ] == pytest.approx(0.0)
    assert parity[
        "ordinary_forward_intrinsic_gradient_report_only_max_absolute_error_eV_per_A"
    ] == pytest.approx(2.0e-7)


def test_float64_response_hard_gates_smooth_repeat_intrinsic_mismatch():
    calculator = object.__new__(AIMNet2ReconstructedFloat64SourceCalculator)
    calculator._last_charge_state = None
    calculator._last_ordinary_decomposed_response_parity = None

    state = _float64_response_state()
    ordinary = _float64_response(
        state,
        intrinsic_energy_gradient_ev_per_angstrom=np.zeros((3, 3)),
    )
    decomposed = _float64_response(
        state,
        intrinsic_energy_gradient_ev_per_angstrom=np.zeros((3, 3)),
    )
    repeat = _float64_response(
        state,
        intrinsic_energy_gradient_ev_per_angstrom=np.asarray(
            [[2.0e-7, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
    )

    calculator._validated_forward = lambda atoms, requires_grad: (
        "ordinary",
        "ordinary",
    )
    repeated = iter((("decomp", "decomp"), ("repeat", "repeat")))
    calculator._validated_decomposed_forward = lambda atoms: next(repeated)

    def _response_from_output(atoms, cotangent, data, output):
        if output == "ordinary":
            return ordinary
        if output == "decomp":
            return decomposed
        if output == "repeat":
            return repeat
        raise AssertionError(output)

    calculator._response_from_output = _response_from_output

    with pytest.raises(RuntimeError, match="public-first-order repeat"):
        calculator.charge_position_response(_water(), np.zeros(3))
