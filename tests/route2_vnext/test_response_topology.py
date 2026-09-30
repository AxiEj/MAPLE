"""Exact support proofs without changing the legacy scalar or thresholds."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maple.solvation.continuum.ddpcm_response_operators import DDPCMResponseOperators
from maple.solvation.continuum.response_topology import certify_response_topology
from maple.solvation.continuum.torch_ddpcm import TorchDDPCM


def fields(first, second):
    r = torch.full((3, 1, 3), 1.4, dtype=torch.float64)
    r[0, 0, 1], r[0, 0, 2] = first, second
    owner = torch.eye(3, dtype=torch.bool)[:, None, :]
    a = 1 - (r - 0.05)
    z = a / 0.1
    chi = torch.where(
        a <= 0, 0, torch.where(a >= 0.1, 1, z**3 * (z * (6 * z - 15) + 10))
    )
    chi = torch.where(owner, 0, chi)
    f = chi.sum(-1)
    return (
        torch.zeros((3, 3), dtype=torch.float64),
        r,
        chi,
        f,
        torch.clamp(1 - f, min=0),
    )


def certify(values):
    return certify_response_topology(*values, eta=0.1, margin=1e-8)


@pytest.mark.parametrize(
    "ratios,side",
    [((0.76, 1.0499780252153306), "upper"), ((0.9500876516943593, 1.4), "lower")],
)
def test_only_exact_one_sided_proofs_admit_near_one(ratios, side):
    values = fields(*ratios)
    result = certify(values)
    assert len(result.one_sided_nodes) == 1
    assert result.one_sided_nodes[0].side == side
    old_policy = SimpleNamespace(_symbols=("H",) * 3, _eta=0.1, _topology_margin=1e-8)
    with pytest.raises(RuntimeError, match="branch-change margin"):
        TorchDDPCM._certify_topology(old_policy, *values[1:])


def test_multi_interior_near_one_still_rejected():
    with pytest.raises(RuntimeError, match="unproved|computed equality"):
        certify(fields(1.0, 1.0 + 1e-12))


def test_interior_rounded_equality_is_not_misclassified_as_plateau():
    # Still safely beyond the endpoint margin, but quintic rounds to one.
    values = fields(0.9500001, 1.4)
    values[2][0, 0, 1] = 1.0
    f = values[2].sum(-1)
    with pytest.raises(RuntimeError, match="computed equality"):
        certify((*values[:3], f, torch.clamp(1 - f, min=0)))


def test_stable_plateau_equality_preserves_zero_derivative_branch():
    result = certify(fields(0.8, 1.4))
    assert result.buried_plateau_count == 1
    assert result.one_sided_nodes == ()


@pytest.mark.parametrize(
    "bad", [0.95, 0.95 + 5e-9, 1.05 - 5e-9, float("nan"), float("inf"), -1.0]
)
def test_endpoint_and_invalid_ratio_guards_remain(bad):
    with pytest.raises(RuntimeError, match="certified margin|Invalid radial"):
        certify(fields(bad, 1.4))


@pytest.mark.parametrize("bad", [-1e-15, 1 + 1e-15, float("nan")])
def test_switch_overshoot_fails_without_clipping(bad):
    values = fields(1.0, 1.4)
    values[2][0, 0, 1] = bad
    f = values[2].sum(-1)
    with pytest.raises(RuntimeError, match="Invalid radial"):
        certify((*values[:3], f, torch.clamp(1 - f, min=0)))


def test_sealed_acetone_proves_only_two_nodes_and_keeps_old_rejection():
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )

    panel = (
        Path(__file__).resolve().parents[2]
        / "docs/route2/evidence/route2-ddpcm-response-performance-geometries-v1.json"
    )
    case = next(
        c for c in json.loads(panel.read_text())["cases"] if c["name"] == "acetone-10"
    )
    op = DDPCMResponseOperators(
        case["symbols"],
        smd_coulomb_radii(case["symbols"], solvent="water"),
        dielectric=78.39,
    )
    positions = torch.tensor(case["positions_angstrom"], dtype=torch.float64)
    geometry = op._geometry(positions)
    proofs = geometry.topology.one_sided_nodes
    assert [(p.owner, p.lebedev_index, p.side) for p in proofs] == [
        (5, 627, "upper"),
        (9, 902, "lower"),
    ]
    ratio = geometry.distance / op._radii[None, None, :]
    with pytest.raises(RuntimeError, match="branch-change margin"):
        TorchDDPCM._certify_topology(op, ratio, geometry.chi, geometry.fi, geometry.ui)
    # Independent AD of the unchanged pointwise scalar, including tiny terms.
    for proof in proofs:
        i, node = proof.owner, proof.lebedev_index
        r = positions.clone().requires_grad_(True)

        def scalar(r):
            from ase.units import Bohr

            delta = r[i] / Bohr + op._radii[i] * op._directions[node] - r / Bohr
            t = torch.linalg.vector_norm(delta, dim=-1) / op._radii
            a = 1 - (t - 0.05)
            z = a / 0.1
            chi = torch.where(
                a <= 0, 0, torch.where(a >= 0.1, 1, z**3 * (z * (6 * z - 15) + 10))
            )
            chi = torch.where(torch.arange(len(r)) == i, 0, chi)
            f = chi.sum()
            return torch.clamp(1 - f, min=0) + 0.37 * torch.clamp(f, min=1)

        grad = torch.autograd.functional.jacobian(scalar, r)
        expected = geometry.u_jacobian[i, node] + 0.37 * geometry.g_jacobian[i, node]
        np.testing.assert_allclose(grad.flatten(), expected, rtol=0, atol=2e-12)
