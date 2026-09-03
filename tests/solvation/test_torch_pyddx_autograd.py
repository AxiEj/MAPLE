from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyddx")

from maple.function.calculator.extra_correction.implicit.gto_density import (
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXPCMReactionFieldLinearMap,
)
from maple.function.calculator.extra_correction.implicit.torch_pyddx import (
    TORCH_PYDDX_AUTOGRAD_CONTRACT_ID,
    TorchPyDDXPCMConfig,
    pyddx_pcm_energy,
)

POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]],
    dtype=np.float64,
)
RADII = (1.52, 1.20, 1.20)
SOURCE = np.asarray(
    [
        [-0.66, 0.015, -0.010, 0.020],
        [0.33, -0.005, 0.012, -0.008],
        [0.33, -0.010, -0.002, -0.012],
    ],
    dtype=np.float64,
)


@pytest.fixture(scope="module")
def config() -> TorchPyDDXPCMConfig:
    return TorchPyDDXPCMConfig(
        radii_angstrom=RADII,
        dielectric=78.355,
        lmax=3,
        n_lebedev=26,
        n_proc=1,
        solver_tolerance=1.0e-12,
        eta=0.1,
    )


def _reference(config: TorchPyDDXPCMConfig) -> PyDDXPCMReactionFieldLinearMap:
    return config.build(POSITIONS)


def test_config_is_immutable_and_content_addressed(config):
    assert config.as_dict()["contract_id"] == TORCH_PYDDX_AUTOGRAD_CONTRACT_ID
    assert config.as_dict()["pyddx_version"] == "0.8.0"
    assert len(config.configuration_sha256) == 64
    with pytest.raises(Exception):
        config.dielectric = 2.0


def test_forward_and_backward_match_standard_pyddx(config):
    positions = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    source = torch.tensor(SOURCE, dtype=torch.float64, requires_grad=True)
    energy = pyddx_pcm_energy(positions, source, config)
    position_gradient, source_gradient = torch.autograd.grad(
        energy,
        (positions, source),
    )

    reference = _reference(config)
    from ase.units import Hartree

    expected_energy = reference.polarization_energy_hartree(SOURCE) * Hartree
    expected_source = external_field_to_density_order(reference.apply(SOURCE))
    expected_position = reference.polarization_position_gradient_ev_per_angstrom(SOURCE)

    assert float(energy.detach()) == pytest.approx(expected_energy, abs=2.0e-12)
    np.testing.assert_allclose(
        source_gradient.detach().numpy(), expected_source, atol=2.0e-11, rtol=2.0e-11
    )
    np.testing.assert_allclose(
        position_gradient.detach().numpy(),
        expected_position,
        atol=2.0e-10,
        rtol=2.0e-10,
    )


def test_source_and_position_directional_derivatives(config):
    positions = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    source = torch.tensor(SOURCE, dtype=torch.float64, requires_grad=True)
    energy = pyddx_pcm_energy(positions, source, config)
    gradient_positions, gradient_source = torch.autograd.grad(
        energy, (positions, source)
    )

    rng = np.random.default_rng(20260902)
    source_direction = rng.normal(size=SOURCE.shape)
    source_direction[:, 0] -= np.mean(source_direction[:, 0])
    source_direction /= np.linalg.norm(source_direction)
    position_direction = rng.normal(size=POSITIONS.shape)
    position_direction -= np.mean(position_direction, axis=0, keepdims=True)
    position_direction /= np.linalg.norm(position_direction)

    analytic_source = float(np.vdot(gradient_source.detach().numpy(), source_direction))
    analytic_position = float(
        np.vdot(gradient_positions.detach().numpy(), position_direction)
    )
    reference = _reference(config)
    from ase.units import Hartree

    source_step = 1.0e-5
    source_finite = (
        (
            reference.polarization_energy_hartree(
                SOURCE + source_step * source_direction
            )
            - reference.polarization_energy_hartree(
                SOURCE - source_step * source_direction
            )
        )
        * Hartree
        / (2.0 * source_step)
    )

    position_step = 1.0e-4
    plus = config.build(POSITIONS + position_step * position_direction)
    minus = config.build(POSITIONS - position_step * position_direction)
    position_finite = (
        (
            plus.polarization_energy_hartree(SOURCE)
            - minus.polarization_energy_hartree(SOURCE)
        )
        * Hartree
        / (2.0 * position_step)
    )

    assert analytic_source == pytest.approx(source_finite, abs=2.0e-8)
    assert analytic_position == pytest.approx(position_finite, abs=2.0e-7)


def test_cuda_round_trip_preserves_first_derivatives(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    positions = torch.tensor(
        POSITIONS, dtype=torch.float64, device="cuda", requires_grad=True
    )
    source = torch.tensor(
        SOURCE, dtype=torch.float64, device="cuda", requires_grad=True
    )
    energy = pyddx_pcm_energy(positions, source, config)
    gradients = torch.autograd.grad(energy, (positions, source))
    assert energy.device.type == "cuda"
    assert all(value.device.type == "cuda" for value in gradients)
    assert all(bool(torch.isfinite(value).all()) for value in gradients)


@pytest.mark.parametrize(
    "bad_positions,bad_source,message",
    (
        (POSITIONS.astype(np.float32), SOURCE, "float64"),
        (POSITIONS, SOURCE.astype(np.float32), "float64"),
        (np.full_like(POSITIONS, np.nan), SOURCE, "finite"),
        (POSITIONS, np.full_like(SOURCE, np.inf), "finite"),
    ),
)
def test_invalid_tensor_inputs_fail_closed(config, bad_positions, bad_source, message):
    with pytest.raises((TypeError, ValueError), match=message):
        pyddx_pcm_energy(
            torch.as_tensor(bad_positions),
            torch.as_tensor(bad_source),
            config,
        )


def test_double_backward_is_explicitly_unavailable(config):
    positions = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    source = torch.tensor(SOURCE, dtype=torch.float64, requires_grad=True)
    energy = pyddx_pcm_energy(positions, source, config)
    (source_gradient,) = torch.autograd.grad(energy, source, create_graph=True)
    assert source_gradient.requires_grad is False
    with pytest.raises(RuntimeError):
        torch.autograd.grad(source_gradient.sum(), source)
