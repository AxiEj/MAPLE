"""Regression coverage for the CUDA-FP32 legacy DAREAL batch path."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import molecule

torch = pytest.importorskip("torch")

import maple.solvation.surfaces.legacy_dareal as dareal
from maple.solvation.surfaces.legacy_dareal import (
    LegacyDAREALConfig,
    LegacyDAREALTopologyError,
    legacy_dareal_areas_torch,
)

POLYGON_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.119262],
        [0.0, 0.763239, -0.477047],
        [0.0, -0.763239, -0.477047],
    ],
    dtype=np.float32,
)
POLYGON_RADII = np.array([1.92, 1.60, 1.60], dtype=np.float32)


@contextmanager
def _forced_implementation(name: str):
    implementation = getattr(dareal, name)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(dareal, "_solid_angle_one", implementation)
        yield


def _evaluate_with(
    name: str,
    positions: torch.Tensor,
    radii: torch.Tensor,
    *,
    config: LegacyDAREALConfig | None = None,
):
    with _forced_implementation(name):
        return legacy_dareal_areas_torch(positions, radii, config=config)


def _polygon_tensors(*, device="cpu", positions_grad=False, radii_grad=False):
    positions = torch.tensor(
        POLYGON_POSITIONS,
        dtype=torch.float32,
        device=device,
        requires_grad=positions_grad,
    )
    radii = torch.tensor(
        POLYGON_RADII,
        dtype=torch.float32,
        device=device,
        requires_grad=radii_grad,
    )
    return positions, radii


def _assert_diagnostics_close(observed, expected):
    observed_payload = asdict(observed)
    expected_payload = asdict(expected)
    for field in ("minimum_guard_margin", "minimum_scalar_cone_margin"):
        observed_margin = observed_payload.pop(field)
        expected_margin = expected_payload.pop(field)
        if expected_margin is None:
            assert observed_margin is None
        else:
            assert observed_margin == pytest.approx(
                expected_margin,
                rel=100.0 * torch.finfo(torch.float32).eps,
                abs=100.0 * torch.finfo(torch.float32).eps,
            )
    assert observed_payload == expected_payload


@pytest.mark.parametrize(
    ("positions", "radii", "expected_branches"),
    [
        (
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [1.0, 1.0],
            ("free", "free"),
        ),
        (
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [2.0, 0.5],
            ("free", "contained"),
        ),
    ],
)
def test_batched_regular_terminal_branches_match_reference(
    positions, radii, expected_branches
):
    positions = torch.tensor(positions, dtype=torch.float32)
    radii = torch.tensor(radii, dtype=torch.float32)
    reference = _evaluate_with("_solid_angle_one_reference", positions, radii)
    batched = _evaluate_with("_solid_angle_one_batched", positions, radii)

    torch.testing.assert_close(batched.areas_angstrom2, reference.areas_angstrom2)
    assert batched.diagnostics.atom_branches == expected_branches
    _assert_diagnostics_close(batched.diagnostics, reference.diagnostics)


def test_batched_polygon_preserves_full_diagnostics_and_topology_hash():
    positions, radii = _polygon_tensors()
    reference = _evaluate_with("_solid_angle_one_reference", positions, radii)
    batched = _evaluate_with("_solid_angle_one_batched", positions, radii)

    assert "polygon" in reference.diagnostics.atom_branches
    _assert_diagnostics_close(batched.diagnostics, reference.diagnostics)
    assert batched.provenance == reference.provenance


def _position_derivatives(method: str):
    positions, radii = _polygon_tensors(positions_grad=True)
    direction = torch.linspace(-0.3, 0.3, positions.numel()).reshape_as(positions)

    def scalar(candidate):
        return _evaluate_with(method, candidate, radii).areas_angstrom2.sum()

    energy = scalar(positions)
    gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]
    return energy, gradient, hvp


def test_batched_polygon_preserves_position_gradient_and_hvp():
    reference = _position_derivatives("_solid_angle_one_reference")
    batched = _position_derivatives("_solid_angle_one_batched")

    for observed, expected in zip(batched, reference, strict=True):
        torch.testing.assert_close(observed, expected, rtol=2.0e-5, atol=2.0e-5)


def _radii_and_mixed_derivatives(method: str):
    positions, radii = _polygon_tensors(positions_grad=True, radii_grad=True)
    radius_direction = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float32)
    energy = _evaluate_with(method, positions, radii).areas_angstrom2.sum()
    radius_gradient = torch.autograd.grad(energy, radii, create_graph=True)[0]
    mixed = torch.autograd.grad((radius_gradient * radius_direction).sum(), positions)[
        0
    ]
    return radius_gradient, mixed


def test_batched_polygon_preserves_radius_gradient_and_mixed_derivative():
    reference = _radii_and_mixed_derivatives("_solid_angle_one_reference")
    batched = _radii_and_mixed_derivatives("_solid_angle_one_batched")

    for observed, expected in zip(batched, reference, strict=True):
        torch.testing.assert_close(observed, expected, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize(
    ("positions", "radii", "message"),
    [
        ([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [1.0, 1.0], "tangent"),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [1.0, 1.0], "coincident"),
    ],
)
def test_batched_path_keeps_singular_geometry_fail_closed(positions, radii, message):
    positions = torch.tensor(positions, dtype=torch.float32)
    radii = torch.tensor(radii, dtype=torch.float32)

    with pytest.raises(LegacyDAREALTopologyError, match=message):
        _evaluate_with("_solid_angle_one_batched", positions, radii)


def _acetone(dtype=torch.float32, device="cpu"):
    atoms = molecule("CH3COCH3")
    bondi = {"C": 1.70, "H": 1.20, "O": 1.52}
    positions = torch.tensor(atoms.positions, dtype=dtype, device=device)
    radii = torch.tensor(
        [bondi[symbol] + 0.4 for symbol in atoms.get_chemical_symbols()],
        dtype=dtype,
        device=device,
    )
    return positions, radii


def test_batched_path_keeps_exact_shared_point_strict_rejection():
    positions, radii = _acetone()

    with pytest.raises(LegacyDAREALTopologyError, match="four-sphere"):
        _evaluate_with("_solid_angle_one_batched", positions, radii)


def test_batched_path_preserves_scalar_shared_vertex_resolution():
    positions, radii = _acetone()
    config = LegacyDAREALConfig(scalar_energy_only=True)
    reference = _evaluate_with(
        "_solid_angle_one_reference", positions, radii, config=config
    )
    batched = _evaluate_with(
        "_solid_angle_one_batched", positions, radii, config=config
    )

    torch.testing.assert_close(batched.areas_angstrom2, reference.areas_angstrom2)
    _assert_diagnostics_close(batched.diagnostics, reference.diagnostics)
    assert batched.diagnostics.scalar_shared_pair_decisions > 0


@pytest.mark.parametrize(
    ("device_type", "dtype", "expected"),
    [
        ("cpu", torch.float32, "reference"),
        ("cpu", torch.float64, "reference"),
        ("cuda", torch.float64, "reference"),
        ("cuda", torch.float32, "batched"),
    ],
)
def test_dispatch_changes_only_cuda_float32(monkeypatch, device_type, dtype, expected):
    calls = []
    sentinel = (object(), "branch", 0, 0, 1.0, "trace")

    def reference(*args, **kwargs):
        calls.append("reference")
        return sentinel

    def batched(*args, **kwargs):
        calls.append("batched")
        return sentinel

    monkeypatch.setattr(dareal, "_solid_angle_one_reference", reference)
    monkeypatch.setattr(dareal, "_solid_angle_one_batched", batched)
    positions = SimpleNamespace(
        device=SimpleNamespace(type=device_type),
        dtype=dtype,
    )

    result = dareal._solid_angle_one(
        positions,
        object(),
        0,
        LegacyDAREALConfig(),
    )

    assert result is sentinel
    assert calls == [expected]


def test_public_surface_signature_does_not_expose_batching_control():
    assert tuple(inspect.signature(legacy_dareal_areas_torch).parameters) == (
        "positions_angstrom",
        "radii_angstrom",
        "config",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float32_dispatch_preserves_dtype_device_and_diagnostics():
    positions, radii = _polygon_tensors(device="cuda:0")
    reference = _evaluate_with("_solid_angle_one_reference", positions, radii)
    observed = legacy_dareal_areas_torch(positions, radii)

    assert observed.areas_angstrom2.dtype is torch.float32
    assert observed.areas_angstrom2.device == positions.device
    assert observed.solid_angles.device == positions.device
    torch.testing.assert_close(
        observed.areas_angstrom2, reference.areas_angstrom2, rtol=2.0e-6, atol=2.0e-6
    )
    _assert_diagnostics_close(observed.diagnostics, reference.diagnostics)


def _host_calls_for_cuda_disjoint(monkeypatch, method: str) -> int:
    count = 35
    positions = torch.zeros((count, 3), dtype=torch.float32, device="cuda:0")
    positions[:, 0] = torch.arange(count, dtype=torch.float32, device="cuda:0") * 5.0
    radii = torch.ones(count, dtype=torch.float32, device="cuda:0")
    calls = 0
    original = dareal._host

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(dareal, "_host", counted)
    _evaluate_with(method, positions, radii)
    return calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_batching_removes_most_scalar_host_round_trips(monkeypatch):
    reference = _host_calls_for_cuda_disjoint(monkeypatch, "_solid_angle_one_reference")
    monkeypatch.undo()
    batched = _host_calls_for_cuda_disjoint(monkeypatch, "_solid_angle_one_batched")

    assert reference > 1000
    assert batched <= reference // 10


def _intersection_fixture(*, requires_grad=False):
    generator = torch.Generator().manual_seed(20261025)
    raw = torch.randn((20, 3), dtype=torch.float64, generator=generator)
    raw = raw.requires_grad_(requires_grad)
    directions = list(torch.nn.functional.normalize(raw, dim=-1).unbind())
    cap_cos = list(torch.zeros(20, dtype=torch.float64).unbind())
    pairs = [(i, j) for i in range(1, 20) for j in range(i)]
    assert len(pairs) == 190
    return raw, directions, cap_cos, pairs


def _reference_intersection(left, right, directions, cap_cos):
    dot = torch.dot(directions[left], directions[right])
    denominator = 1.0 - dot * dot
    inverse = 1.0 / denominator
    ai = (cap_cos[left] - cap_cos[right] * dot) * inverse
    bi = (cap_cos[right] - cap_cos[left] * dot) * inverse
    ci2 = (1.0 - ai * cap_cos[left] - bi * cap_cos[right]) * inverse
    cross = torch.linalg.cross(directions[left], directions[right])
    centre = ai * directions[left] + bi * directions[right]
    displacement = torch.sqrt(ci2) * cross
    positive, negative = centre + displacement, centre - displacement
    visibility = torch.stack(
        [
            torch.stack([torch.dot(point, normal) for normal in directions])
            for point in (positive, negative)
        ]
    )
    return denominator, ci2, positive, negative, visibility


def test_intersection_batches_preserve_pair_order_and_chunk_boundaries(monkeypatch):
    _, directions, cap_cos, pairs = _intersection_fixture()
    transfers = []
    original = dareal._host_values

    def recording_transfer(value):
        transfers.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(dareal, "_host_values", recording_transfer)
    observed = list(dareal._intersection_batches(pairs, directions, cap_cos))

    assert [(left, right) for left, right, _ in observed] == pairs
    assert transfers == [(128, 2), (128, 2, 20), (62, 2), (62, 2, 20)]
    for left, right, prepared in observed:
        (denominator_h, ci2_h), positive, negative, visibility_h = prepared
        expected = _reference_intersection(left, right, directions, cap_cos)
        assert denominator_h == expected[0].item()
        assert ci2_h == expected[1].item()
        torch.testing.assert_close(positive, expected[2], rtol=0.0, atol=0.0)
        torch.testing.assert_close(negative, expected[3], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(visibility_h, expected[4].numpy())


def _intersection_scalar(raw, *, batched):
    directions = list(torch.nn.functional.normalize(raw, dim=-1).unbind())
    cap_cos = list(torch.zeros(20, dtype=torch.float64).unbind())
    pairs = [(i, j) for i in range(1, 20) for j in range(i)]
    weights = raw.new_tensor([0.3, -0.2, 0.5])
    scalar = raw.new_zeros(())
    if batched:
        records = dareal._intersection_batches(pairs, directions, cap_cos)
        points = (
            (left, right, prepared[1], prepared[2]) for left, right, prepared in records
        )
    else:
        points = (
            (left, right, reference[2], reference[3])
            for left, right in pairs
            for reference in [_reference_intersection(left, right, directions, cap_cos)]
        )
    for index, (_, _, positive, negative) in enumerate(points, start=1):
        scalar = scalar + index * (
            torch.dot(positive, weights) + 0.25 * torch.dot(negative, weights) ** 2
        )
    return scalar


def test_intersection_batches_preserve_coordinate_gradient_and_hvp_across_chunks():
    raw, _, _, _ = _intersection_fixture(requires_grad=True)
    direction = torch.linspace(-0.4, 0.4, raw.numel(), dtype=raw.dtype).reshape_as(raw)

    def derivatives(*, batched):
        value = _intersection_scalar(raw, batched=batched)
        gradient = torch.autograd.grad(
            value, raw, create_graph=True, retain_graph=True
        )[0]
        hvp = torch.autograd.grad((gradient * direction).sum(), raw, retain_graph=True)[
            0
        ]
        return value, gradient, hvp

    reference = derivatives(batched=False)
    observed = derivatives(batched=True)

    for actual, expected in zip(observed, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)
