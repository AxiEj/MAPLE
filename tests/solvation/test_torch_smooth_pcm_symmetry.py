from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import TorchSmoothPCM
from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm.exposure import (
    _rotation_from_positive_z,
)

from _torch_smooth_pcm_reference import P, ROTATIONS, W, rotate_cartesian_field, rotate_raw_source, tangency_fixture


def _model(fixture, *, lmax=2):
    return TorchSmoothPCM(
        atomic_numbers=fixture.atomic_numbers,
        radii_angstrom=fixture.radii,
        transition_width_angstrom2=fixture.transition_width,
        surface_lmax=min(fixture.surface_lmax, lmax),
        partition_lmax=min(fixture.partition_lmax, 2 * lmax),
        partition_radial_quadrature_order=32,
        source_radial_quadrature_order=48,
        double_layer_radial_quadrature_order=48,
        dielectric=fixture.dielectric,
    )


@pytest.mark.parametrize("fixture", [P, W], ids=lambda fixture: fixture.name)
def test_common_translation_preserves_energy_drive_and_covariant_gradient(fixture):
    model = _model(fixture)
    shift = np.asarray([1.3, -0.4, 0.8])
    translated = fixture.positions + shift
    assert model.energy(translated, fixture.source) == pytest.approx(model.energy(fixture.positions, fixture.source), abs=2e-12, rel=0.0)
    assert np.allclose(model.drive_cartesian(translated, fixture.source), model.drive_cartesian(fixture.positions, fixture.source), atol=2e-11, rtol=0.0)
    assert np.allclose(model.coordinate_gradient(translated, fixture.source), model.coordinate_gradient(fixture.positions, fixture.source), atol=2e-11, rtol=0.0)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_proper_rotation_preserves_energy_and_rotates_drive_and_gradient(rotation):
    model = _model(P)
    positions = P.positions @ rotation.T
    source = rotate_raw_source(P.source, rotation)
    assert model.energy(positions, source) == pytest.approx(model.energy(P.positions, P.source), abs=2e-12, rel=0.0)
    expected_drive = rotate_cartesian_field(model.drive_cartesian(P.positions, P.source), rotation)
    assert np.allclose(model.drive_cartesian(positions, source), expected_drive, atol=2e-11, rtol=0.0)
    expected_gradient = model.coordinate_gradient(P.positions, P.source) @ rotation.T
    assert np.allclose(model.coordinate_gradient(positions, source), expected_gradient, atol=2e-11, rtol=0.0)


def test_atom_permutation_preserves_energy_and_permutes_derivatives():
    model = _model(P)
    permutation = np.asarray([1, 0])
    permuted_model = TorchSmoothPCM(
        atomic_numbers=tuple(P.atomic_numbers[index] for index in permutation),
        radii_angstrom=tuple(P.radii[index] for index in permutation),
        transition_width_angstrom2=P.transition_width,
        surface_lmax=2,
        partition_lmax=4,
        partition_radial_quadrature_order=32,
        source_radial_quadrature_order=48,
        double_layer_radial_quadrature_order=48,
        dielectric=P.dielectric,
    )
    positions, source = P.positions[permutation], P.source[permutation]
    assert permuted_model.energy(positions, source) == pytest.approx(model.energy(P.positions, P.source), abs=2e-12, rel=0.0)
    assert np.allclose(permuted_model.drive_cartesian(positions, source), model.drive_cartesian(P.positions, P.source)[permutation], atol=2e-11, rtol=0.0)
    assert np.allclose(permuted_model.coordinate_gradient(positions, source), model.coordinate_gradient(P.positions, P.source)[permutation], atol=2e-11, rtol=0.0)


def test_translation_and_rotation_invariance_imply_zero_net_gradient_and_torque():
    model = _model(W)
    gradient = model.coordinate_gradient(W.positions, W.source)
    assert np.max(np.abs(np.sum(gradient, axis=0))) < 3e-12
    source_cartesian = W.source[:, (0, 3, 1, 2)]
    raw_gradient = model.drive_cartesian(W.positions, W.source)[:, (0, 2, 3, 1)]
    gradient_cartesian = raw_gradient[:, (0, 3, 1, 2)]
    torque = np.sum(np.cross(W.positions, gradient), axis=0)
    torque += np.sum(np.cross(source_cartesian[:, 1:], gradient_cartesian[:, 1:]), axis=0)
    assert np.max(np.abs(torque)) < 2e-11


@pytest.mark.parametrize("separation", [0.2, 1.8])
def test_sphere_tangency_has_convergent_gradient_and_finite_hvp(separation):
    fixture = tangency_fixture(separation)
    model = TorchSmoothPCM(
        atomic_numbers=fixture.atomic_numbers,
        radii_angstrom=fixture.radii,
        transition_width_angstrom2=fixture.transition_width,
        surface_lmax=fixture.surface_lmax,
        partition_lmax=fixture.partition_lmax,
        partition_radial_quadrature_order=fixture.partition_order,
        source_radial_quadrature_order=fixture.source_order,
        double_layer_radial_quadrature_order=fixture.double_layer_order,
        dielectric=fixture.dielectric,
    )
    direction = np.zeros_like(fixture.positions)
    direction[1, 0] = 1.0
    analytic = float(np.vdot(model.coordinate_gradient(fixture.positions, fixture.source), direction))
    errors, jumps = [], []
    for step in (3e-4, 1e-4, 3e-5):
        plus, minus = fixture.positions + step * direction, fixture.positions - step * direction
        errors.append(abs((model.energy(plus, fixture.source) - model.energy(minus, fixture.source)) / (2 * step) - analytic))
        jumps.append(abs(float(np.vdot(model.coordinate_gradient(plus, fixture.source) - model.coordinate_gradient(minus, fixture.source), direction))))
    assert errors[1] < errors[0] / 5 and errors[2] < errors[1] / 5
    assert errors[-1] < 8e-8
    assert jumps[1] < jumps[0] / 2 and jumps[2] < jumps[1] / 2
    hvp_r, _ = model.joint_hvp(fixture.positions, fixture.source, direction, np.zeros_like(fixture.source))
    assert np.all(np.isfinite(hvp_r))
    step = 3e-6
    actual = (model.coordinate_gradient(fixture.positions + step * direction, fixture.source) - model.coordinate_gradient(fixture.positions - step * direction, fixture.source)) / (2 * step)
    assert np.allclose(hvp_r, actual, atol=2e-6, rtol=5e-5)


@pytest.mark.parametrize("offset", [1e-4, 1e-6, 1e-8])
def test_positive_z_rotation_chart_is_continuous_across_frozen_seam(offset):
    torch = pytest.importorskip("torch")
    x = np.sqrt(1.0 - (-0.5 + offset) ** 2)
    left = torch.tensor([x, 0.0, -0.5 + offset], dtype=torch.float64)
    x = np.sqrt(1.0 - (-0.5 - offset) ** 2)
    right = torch.tensor([x, 0.0, -0.5 - offset], dtype=torch.float64)
    r_left = _rotation_from_positive_z(left).detach().numpy()
    r_right = _rotation_from_positive_z(right).detach().numpy()
    assert np.all(np.isfinite(r_left)) and np.all(np.isfinite(r_right))
    assert np.linalg.det(r_left) == pytest.approx(1.0, abs=2e-12)
    assert np.linalg.det(r_right) == pytest.approx(1.0, abs=2e-12)
    assert np.allclose(r_left[:, 2], left.numpy(), atol=2e-12, rtol=0.0)
    assert np.allclose(r_right[:, 2], right.numpy(), atol=2e-12, rtol=0.0)
    # Frames may use different in-plane gauges; their physical axes must converge.
    assert np.linalg.norm(r_left[:, 2] - r_right[:, 2]) < 3 * offset


def test_scalar_gradient_and_hvp_are_continuous_across_rotation_chart_seam():
    model = _model(P)
    direction = np.zeros_like(P.positions)
    direction[1] = np.asarray([1.0, 0.0, 0.0])
    records = []
    for offset in (1e-4, 1e-6, 1e-8):
        left_z, right_z = -0.5 + offset, -0.5 - offset
        left_u = np.asarray([np.sqrt(1 - left_z**2), 0.0, left_z])
        right_u = np.asarray([np.sqrt(1 - right_z**2), 0.0, right_z])
        left = P.positions.copy(); left[1] = np.linalg.norm(P.positions[1]) * left_u
        right = P.positions.copy(); right[1] = np.linalg.norm(P.positions[1]) * right_u
        left_hvp, _ = model.joint_hvp(left, P.source, direction, np.zeros_like(P.source))
        right_hvp, _ = model.joint_hvp(right, P.source, direction, np.zeros_like(P.source))
        values = (
            abs(model.energy(left, P.source) - model.energy(right, P.source)),
            np.max(np.abs(model.coordinate_gradient(left, P.source) - model.coordinate_gradient(right, P.source))),
            np.max(np.abs(left_hvp - right_hvp)),
        )
        assert np.all(np.isfinite(values))
        records.append(values)
    records = np.asarray(records)
    assert np.all(records[1] < records[0] / 20)
    assert np.all(records[2] < records[1] / 20)


def test_fresh_process_replays_configuration_and_numerical_result(tmp_path):
    script = """
import json, numpy as np
from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import TorchSmoothPCM
R=np.array([[0.,0.,0.]])
c=np.array([[0.4,0.2,-0.1,0.3]])
m=TorchSmoothPCM(atomic_numbers=(1,),radii_angstrom=(2.,),surface_lmax=1,partition_lmax=2,partition_radial_quadrature_order=32,source_radial_quadrature_order=48,double_layer_radial_quadrature_order=48,dielectric=7.)
print(json.dumps({'hash':m.configuration_sha256(),'execution':m.execution_provenance()['execution_provenance_sha256'],'energy':m.energy(R,c),'drive':m.drive_cartesian(R,c).tolist()},sort_keys=True))
"""
    outputs = [subprocess.check_output([sys.executable, "-c", script], text=True) for _ in range(2)]
    assert outputs[0] == outputs[1]
