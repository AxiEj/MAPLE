from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ase import Atoms

pytest.importorskip("pyscf")

from maple.solvation.continuum import (
    build_pair_frame_water_cpcm_110_candidate,
)
from maple.solvation.continuum.base import ContinuumBackend
from maple.solvation.release import (
    load_pes_panel,
    rotate_radial_gto_blocks,
    symmetry_panel_permutation,
    symmetry_panel_rotations,
)


def _methanol():
    return load_pes_panel()[1].atoms


def _random_source(count, seed):
    return np.random.default_rng(seed).normal(scale=0.03, size=(count, 8))


@pytest.fixture(scope="module")
def backend():
    atoms = _methanol()
    return build_pair_frame_water_cpcm_110_candidate(atoms.get_chemical_symbols())


def test_source_adjoint_reciprocity_and_energy(backend):
    atoms = _methanol()
    source = _random_source(len(atoms), 1)
    direction = _random_source(len(atoms), 2)
    cotangent = _random_source(len(atoms), 3)
    field = backend.evaluate_field(atoms, source)
    state = backend.build_state(atoms, source)
    assert isinstance(backend, ContinuumBackend)
    np.testing.assert_array_equal(state.source, source)
    np.testing.assert_array_equal(state.reaction_field, field)
    assert not state.source.flags.writeable
    assert not state.reaction_field.flags.writeable
    assert state.frame_count == len(atoms) * (len(atoms) - 1)
    assert state.active_frame_count <= state.frame_count
    assert len(state.geometry_sha256) == 64
    assert len(state.frame_topology_sha256) == 64
    assert len(state.frame_activity_sha256) == 64
    with pytest.raises(ValueError, match="half-coupling scalar"):
        replace(state, polarization_energy_eV=state.polarization_energy_eV + 1e-6)
    with pytest.raises(ValueError, match="configuration_sha256"):
        replace(state, configuration_sha256="forged")
    with pytest.raises(ValueError, match="topology identity"):
        replace(state, frame_topology_sha256="0" * 64, state_hash="0" * 64)
    assert backend.energy(atoms, source) == pytest.approx(
        0.5 * np.vdot(source, field), abs=1e-14
    )
    jvp = backend.source_jvp(atoms, source, direction)
    vjp = backend.source_vjp(atoms, source, cotangent)
    assert np.vdot(cotangent, jvp) == pytest.approx(np.vdot(vjp, direction), abs=2e-13)
    step = 1e-5
    fd = (
        backend.evaluate_field(atoms, source + step * direction)
        - backend.evaluate_field(atoms, source - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(jvp, fd, atol=5e-10, rtol=2e-9)


def test_coordinate_vjp_is_same_bilinear_derivative(backend):
    atoms = _methanol()
    source = _random_source(len(atoms), 4)
    cotangent = _random_source(len(atoms), 5)
    analytic = backend.coordinate_vjp(atoms, source, cotangent).reshape(len(atoms), 3)
    rng = np.random.default_rng(6)
    direction = rng.normal(size=(len(atoms), 3))
    direction -= np.mean(direction, axis=0)
    direction /= np.linalg.norm(direction)
    projected = float(np.vdot(analytic, direction))
    errors = []
    for step in (2e-4, 1e-4, 5e-5):
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        finite_difference = (
            np.vdot(cotangent, backend.evaluate_field(plus, source))
            - np.vdot(cotangent, backend.evaluate_field(minus, source))
        ) / (2 * step)
        errors.append(abs(finite_difference - projected))
    assert errors[-1] < 1e-9
    assert errors[-1] < errors[0] / 8
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=2e-12)


def test_exact_rotation_translation_and_permutation_covariance(backend):
    atoms = _methanol()
    source = _random_source(len(atoms), 7)
    cotangent = _random_source(len(atoms), 8)
    base_field = backend.evaluate_field(atoms, source)
    base_value = float(np.vdot(cotangent, base_field))
    base_gradient = backend.coordinate_vjp(atoms, source, cotangent).reshape(
        len(atoms), 3
    )

    shifted = atoms.copy()
    shifted.positions += np.asarray([1.7, -0.8, 0.5])
    assert np.vdot(cotangent, backend.evaluate_field(shifted, source)) == pytest.approx(
        base_value, abs=2e-13
    )
    np.testing.assert_allclose(
        backend.coordinate_vjp(shifted, source, cotangent).reshape(len(atoms), 3),
        base_gradient,
        atol=2e-12,
    )

    for rotation in symmetry_panel_rotations("methanol"):
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.T
        rotated_source = rotate_radial_gto_blocks(source, rotation)
        rotated_cotangent = rotate_radial_gto_blocks(cotangent, rotation)
        rotated_field = backend.evaluate_field(rotated, rotated_source)
        np.testing.assert_allclose(
            rotated_field,
            rotate_radial_gto_blocks(base_field, rotation),
            atol=3e-12,
            rtol=3e-12,
        )
        rotated_gradient = backend.coordinate_vjp(
            rotated, rotated_source, rotated_cotangent
        ).reshape(len(atoms), 3)
        np.testing.assert_allclose(
            rotated_gradient, base_gradient @ rotation.T, atol=3e-12, rtol=3e-12
        )

    permutation = symmetry_panel_permutation(atoms.numbers)
    inverse = np.argsort(permutation)
    permuted = Atoms(
        numbers=atoms.numbers[permutation],
        positions=atoms.positions[permutation],
        info=dict(atoms.info),
    )
    permuted_backend = build_pair_frame_water_cpcm_110_candidate(
        permuted.get_chemical_symbols()
    )
    np.testing.assert_allclose(
        permuted_backend.evaluate_field(permuted, source[permutation])[inverse],
        base_field,
        atol=3e-12,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        permuted_backend.coordinate_vjp(
            permuted, source[permutation], cotangent[permutation]
        ).reshape(len(atoms), 3)[inverse],
        base_gradient,
        atol=3e-12,
        rtol=3e-12,
    )


def test_fail_closed_for_linear_molecule_and_capabilities_closed():
    backend = build_pair_frame_water_cpcm_110_candidate(("H", "C", "H"))
    linear = Atoms("HCH", positions=[[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="non-collinear"):
        backend.evaluate_field(linear, np.zeros((3, 8)))
    assert backend.capabilities.enabled_tiers == ()
    assert dict(backend.runtime_provenance)["scientific_status"] == (
        "new-ensemble-discretization-not-conventional-single-cpcm"
    )


def test_every_ordered_pair_and_member_surface_remains_in_fixed_topology(backend):
    atoms = _methanol()
    diagnostics = backend.frame_diagnostics(atoms)
    expected = len(atoms) * (len(atoms) - 1)
    assert diagnostics["ordered_pair_frame_count"] == expected
    assert (
        diagnostics["ordered_noncollinear_frame_count"]
        + diagnostics["zero_weight_singular_frame_count"]
        == expected
    )

    # In this H4 geometry the nuclear-charge centroid is the midpoint of two
    # opposite pairs, while the other pairs keep the normalized ensemble live.
    singular = Atoms(
        "H4",
        positions=[
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )
    singular_backend = build_pair_frame_water_cpcm_110_candidate(("H",) * 4)
    changed = singular_backend.frame_diagnostics(singular)
    singular_expected = len(singular) * (len(singular) - 1)
    assert changed["ordered_pair_frame_count"] == singular_expected
    assert changed["zero_weight_singular_frame_count"] >= 4

    state = singular_backend.build_state(singular, _random_source(len(singular), 91))
    assert state.frame_count == singular_expected
    assert state.frame_topology_sha256 == changed["frame_topology_sha256"]


def test_frame_definition_is_scale_covariant_without_absolute_length_cutoff():
    water = load_pes_panel()[0].atoms
    backend = build_pair_frame_water_cpcm_110_candidate(water.get_chemical_symbols())
    base = backend.frame_diagnostics(water)
    for scale in (1.0e-8, 1.0e8):
        scaled = water.copy()
        scaled.positions *= scale
        diagnostics = backend.frame_diagnostics(scaled)
        assert (
            diagnostics["ordered_noncollinear_frame_count"]
            == base["ordered_noncollinear_frame_count"]
        )


def test_geometry_local_linear_operator_cache_is_exact(monkeypatch, backend):
    atoms = _methanol()
    source = _random_source(len(atoms), 9)
    calls = 0
    original = type(backend.body_backend).reaction_field_matrix

    def counted(self, geometry):
        nonlocal calls
        calls += 1
        return original(self, geometry)

    monkeypatch.setattr(type(backend.body_backend), "reaction_field_matrix", counted)
    first = backend.evaluate_field(atoms, source)
    frame_count = backend.frame_diagnostics(atoms)["ordered_noncollinear_frame_count"]
    assert calls == frame_count
    np.testing.assert_array_equal(
        backend.evaluate_field(atoms, 0.5 * source), 0.5 * first
    )
    assert calls == frame_count

    displaced = atoms.copy()
    displaced.positions[0, 0] = np.nextafter(displaced.positions[0, 0], np.inf)
    backend.evaluate_field(displaced, source)
    assert calls == 2 * frame_count
