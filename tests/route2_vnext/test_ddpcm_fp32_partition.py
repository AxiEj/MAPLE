import warnings

import numpy as np
import pytest
import torch

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_coulomb_radii,
)
from maple.function.route2_solvents import route2_solvent_spec
from maple.solvation.continuum.fp32_partition import (
    FP32_PARTITION_POLICY,
    assess_fp32_partition,
    stable_fp32_partition,
)
from maple.solvation.continuum.torch_ddpcm import TorchDDPCM

_POSITIONS = {
    "center": [
        [0.0, 0.0, 0.0],
        [0.9571999907493591, 0.0, 0.0],
        [-0.23899999260902405, 0.9265999794006348, 0.0],
    ],
    "delta_1e-5": [
        [6.038383730810892e-07, 7.045237907732371e-06, 0.0],
        [0.9571999907493591, 0.0, 0.0],
        [-0.239000603556633, 0.9265929460525513, 0.0],
    ],
    "delta_1e-4": [
        [6.038383617124055e-06, 7.045237725833431e-05, 0.0],
        [0.9571999907493591, 0.0, 0.0],
        [-0.23900602757930756, 0.9265295267105103, 0.0],
    ],
}
_SOURCE = [
    [
        -0.03473886102437973,
        0.0,
        0.07982062548398972,
        0.0,
        0.06187484413385391,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.017370693385601044,
        0.0,
        -0.0168819110840559,
        0.0,
        0.25008583068847656,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.017368167638778687,
        0.0,
        0.23792995512485504,
        0.0,
        -0.07881047576665878,
        0.0,
        0.0,
        0.0,
    ],
]


def _continuum(dtype):
    solvent = route2_solvent_spec("acetonitrile")
    return TorchDDPCM(
        ("O", "H", "H"),
        smd_coulomb_radii(("O", "H", "H"), solvent="acetonitrile"),
        dielectric=solvent.descriptors.dielectric,
        lmax=5,
        n_lebedev=194,
        dtype=dtype,
        device="cpu",
    )


def _hessian(energy, positions):
    gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    return torch.stack(
        [
            torch.autograd.grad(value, positions, retain_graph=True)[0].reshape(-1)
            for value in gradient.reshape(-1)
        ]
    )


def test_single_interior_partition_uses_stable_complement_and_unit_normalizer():
    ratios = torch.tensor(
        [[[1.0, 0.95001], [1.0, 1.4]], [[1.4, 1.0], [1.4, 1.0]]], dtype=torch.float32
    )
    owner = torch.eye(2, dtype=torch.bool)[:, None, :]
    chi, fi, ui = stable_fp32_partition(ratios, owner, eta=0.1)
    assert 0.0 < float(ui[0, 0]) < 1.0
    assert float(fi[0, 0] + ui[0, 0]) == 1.0
    normalizer = torch.where(ui > 0, torch.ones_like(fi), fi)
    assert float(normalizer[0, 0]) == 1.0


def test_complementary_quintic_retains_tails_at_adjacent_fp32_endpoint_values():
    lower = torch.tensor(0.95, dtype=torch.float32)
    upper = torch.tensor(1.05, dtype=torch.float32)
    lower_inside = torch.nextafter(lower, upper)
    upper_inside = torch.nextafter(upper, lower)
    ratios = torch.tensor(
        [[[1.0, lower_inside], [1.0, upper_inside]], [[1.4, 1.0], [1.4, 1.0]]],
        dtype=torch.float32,
    )
    owner = torch.eye(2, dtype=torch.bool)[:, None, :]
    chi, _, ui = stable_fp32_partition(ratios, owner, eta=0.1)
    assert 0.0 < float(ui[0, 0]) < 1.0
    assert 0.0 < float(chi[0, 1, 1]) < 1.0
    assert bool(((chi >= 0.0) & (chi <= 1.0)).all())
    assert bool(((ui >= 0.0) & (ui <= 1.0)).all())


def test_centered_midpoint_adjacent_values_follow_analytic_second_order_model():
    owner = torch.eye(2, dtype=torch.bool)[:, None, :]

    def switch_value(ratio):
        one = ratio.new_tensor(1.0)
        far = ratio.new_tensor(1.4)
        ratios = torch.stack(
            (torch.stack((one, ratio)), torch.stack((far, one)))
        ).reshape(2, 1, 2)
        return stable_fp32_partition(ratios, owner, eta=0.1)[0][0, 0, 1]

    center = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    value = switch_value(center)
    gradient = torch.autograd.grad(value, center, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient, center)[0]
    assert float(value.detach()) == 0.5
    assert float(hessian.detach()) == 0.0
    for direction in (-torch.inf, torch.inf):
        adjacent = torch.nextafter(center.detach(), center.new_tensor(direction))
        displacement = adjacent - center.detach()
        analytic_taylor = (
            value.detach()
            + gradient.detach() * displacement
            + 0.5 * hessian.detach() * displacement**2
        )
        residual = torch.abs(switch_value(adjacent) - analytic_taylor)
        assert float(residual) <= 0.5 * torch.finfo(torch.float32).eps


def test_multi_interior_near_one_remains_a_hard_failure():
    ratios = torch.full((3, 1, 3), 1.4, dtype=torch.float32)
    owner = torch.eye(3, dtype=torch.bool)[:, None, :]
    ratios[owner.expand_as(ratios)] = 1.0
    # Two identical interior contributions of 0.5 make f=1 without a
    # one-sided structural proof.
    ratios[0, 0, 1:] = 1.0
    chi, fi, ui = stable_fp32_partition(ratios, owner, eta=0.1)
    details = assess_fp32_partition(
        ratios, owner, chi, fi, ui, eta=0.1, margin=100 * torch.finfo(torch.float32).eps
    )
    assert details.hard_failures == ("topology_branch_margin",)
    assert details.multiple_interior_node_count >= 1
    with pytest.raises(RuntimeError, match="topology"):
        _continuum(torch.float32)._certify_topology(ratios, chi, fi, ui)


@pytest.mark.parametrize("name", tuple(_POSITIONS))
def test_mecn_fixed_source_fp32_energy_force_hessian_is_accepted_and_tracks_fp64(name):
    results = {}
    for dtype in (torch.float32, torch.float64):
        continuum = _continuum(dtype)
        positions = torch.tensor(_POSITIONS[name], dtype=dtype, requires_grad=True)
        source = torch.tensor(_SOURCE, dtype=dtype)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state = continuum._state_torch(positions, source, enforce_checks=True)
        results[dtype] = (
            float(state.energy.detach()),
            -torch.autograd.grad(state.energy, positions, create_graph=True)[0]
            .detach()
            .cpu()
            .numpy(),
            _hessian(state.energy, positions).detach().cpu().numpy(),
        )
        if dtype is torch.float32:
            assert state.fp32_partition is not None
            assert state.fp32_partition.contract == FP32_PARTITION_POLICY
            assert not state.fp32_partition.hard_failures
            assert any(item.category is RuntimeWarning for item in caught)
        else:
            assert state.fp32_partition is None

    energy32, force32, hessian32 = results[torch.float32]
    energy64, force64, hessian64 = results[torch.float64]
    assert abs(energy32 - energy64) < 1.0e-5
    assert np.max(np.abs(force32 - force64)) < 1.0e-4
    # The former rounded-clamp branch produced a 5.45e-3 eV/A^2 spike.
    assert np.max(np.abs(hessian32 - hessian64)) < 2.0e-4


def test_explicit_fp64_geometry_retains_legacy_switch_and_clamp_exactly():
    continuum = _continuum(torch.float64)
    positions = torch.tensor(_POSITIONS["center"], dtype=torch.float64)
    geometry = continuum._geometry(positions)
    radii, distances, chi, fi, ui = (
        geometry[2],
        geometry[6],
        geometry[7],
        geometry[8],
        geometry[9],
    )
    owner = torch.eye(3, dtype=torch.bool)[:, None, :]
    ratios = torch.where(
        owner, torch.ones_like(distances), distances / radii[None, None, :]
    )
    expected_chi = torch.where(
        owner, torch.zeros_like(ratios), continuum._switch(ratios)
    )
    expected_fi = expected_chi.sum(dim=-1)
    assert torch.equal(chi, expected_chi)
    assert torch.equal(fi, expected_fi)
    assert torch.equal(ui, torch.clamp(1.0 - expected_fi, min=0.0))
