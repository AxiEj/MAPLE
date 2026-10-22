"""Exact-geometry scalar SASA at multiple-cap vertices, without radius jitter."""

import numpy as np
import pytest
from ase.build import molecule

torch = pytest.importorskip("torch")

from maple.solvation.surfaces.legacy_dareal import (
    LegacyDAREALConfig,
    LegacyDAREALTopologyError,
    _require_margin,
    legacy_dareal_areas_torch,
)


def acetone(dtype=torch.float64):
    atoms = molecule("CH3COCH3")
    bondi = {"C": 1.70, "H": 1.20, "O": 1.52}
    p = torch.tensor(atoms.positions, dtype=dtype)
    r = torch.tensor(
        [bondi[s] + 0.4 for s in atoms.get_chemical_symbols()], dtype=dtype
    )
    return p, r


def scalar_areas(p, r):
    return legacy_dareal_areas_torch(
        p, r, config=LegacyDAREALConfig(scalar_energy_only=True)
    )


def test_true_shared_vertex_is_resolved_only_for_scalar_evaluation():
    p, r = acetone()
    p0, r0 = p.clone(), r.clone()
    with pytest.raises(LegacyDAREALTopologyError, match="four-sphere"):
        legacy_dareal_areas_torch(p, r)
    result = scalar_areas(p, r)
    assert torch.isfinite(result.areas_angstrom2).all()
    assert result.diagnostics.scalar_energy_only
    assert result.diagnostics.scalar_shared_pair_decisions > 0
    assert result.diagnostics.minimum_scalar_cone_margin > 1e-5
    assert torch.equal(p, p0)
    assert torch.equal(r, r0)


def test_scalar_policy_cannot_silently_supply_coordinate_derivatives():
    p, r = acetone()
    p.requires_grad_()
    with pytest.raises(ValueError, match="scalar|derivative"):
        scalar_areas(p, r)
    p.requires_grad_(False)
    r.requires_grad_()
    with pytest.raises(ValueError, match="scalar|derivative"):
        scalar_areas(p, r)


def test_scalar_area_is_atom_order_and_rotation_invariant():
    p, r = acetone()
    reference = scalar_areas(p, r).areas_angstrom2.numpy()
    rng = np.random.default_rng(20261018)
    for _ in range(8):
        order = rng.permutation(len(p))
        rotated, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotated = torch.tensor(rotated, dtype=p.dtype)
        result = scalar_areas((p @ rotated)[order], r[order]).areas_angstrom2.numpy()
        restored = np.empty_like(result)
        restored[order] = result
        np.testing.assert_allclose(restored, reference, atol=1e-10, rtol=1e-12)


def test_float32_continuous_geometry_is_not_silently_upcast():
    p, r = acetone(torch.float32)
    reference = scalar_areas(p.double(), r.double()).areas_angstrom2.numpy()
    result = scalar_areas(p, r)
    assert result.areas_angstrom2.dtype is torch.float32
    eps = torch.finfo(torch.float32).eps
    np.testing.assert_allclose(
        result.areas_angstrom2.numpy(), reference, atol=100 * eps, rtol=100 * eps
    )


def test_scalar_topology_hash_does_not_include_continuous_cone_margins():
    p, r = acetone()
    reference = scalar_areas(p, r).diagnostics.topology_sha256
    rng = np.random.default_rng(2026101801)
    for _ in range(3):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
        rotated = p @ torch.tensor(rotation, dtype=p.dtype)
        assert scalar_areas(rotated, r).diagnostics.topology_sha256 == reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_topology_hash_matches_cpu_and_cuda(dtype):
    p, r = acetone(dtype)
    cpu = scalar_areas(p, r)
    cuda = scalar_areas(p.cuda(), r.cuda())
    assert cpu.diagnostics.topology_sha256 == cuda.diagnostics.topology_sha256


def test_scalar_mode_does_not_disable_other_singular_geometry_checks():
    p = torch.zeros((2, 3), dtype=torch.float64)
    r = torch.ones(2, dtype=torch.float64)
    with pytest.raises(LegacyDAREALTopologyError, match="coincident"):
        scalar_areas(p, r)


def test_scalar_policy_switch_is_explicit_boolean():
    with pytest.raises(TypeError, match="scalar_energy_only"):
        LegacyDAREALConfig(scalar_energy_only=1)


def test_scalar_margin_is_not_a_derivative_regularity_certificate():
    strict = LegacyDAREALConfig(relative_topology_tolerance=1e-5)
    scalar = LegacyDAREALConfig(
        relative_topology_tolerance=1e-5, scalar_energy_only=True
    )
    with pytest.raises(LegacyDAREALTopologyError, match="test margin"):
        _require_margin(1e-7, 1.0, strict, "test margin")
    assert _require_margin(1e-7, 1.0, scalar, "test margin") == 1e-7


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_scalar_margin_still_rejects_zero_or_nonfinite_geometry(value):
    config = LegacyDAREALConfig(
        relative_topology_tolerance=1e-5, scalar_energy_only=True
    )
    with pytest.raises(LegacyDAREALTopologyError, match="invalid margin"):
        _require_margin(value, 1.0, config, "invalid margin")
