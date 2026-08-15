from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.continuum.atomic_l1_pyddx import AtomicL1PyDDXPCMBackend
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING

_BETA = 0.03


class _FakeReactionMap:
    cavity_topology_sha256 = "1" * 64
    cavity_active_node_pairs = ((0, 0), (1, 0))
    minimum_cavity_active_set_clearance_angstrom = 0.25

    def __init__(self, positions, radii, **kwargs):
        self.positions = np.asarray(positions, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        self.scale = 1.0 + _BETA * float(np.vdot(self.positions, self.positions))
        self.runtime_provenance = {
            "backend": "fake-pyddx",
            "lmax": kwargs["lmax"],
            "n_lebedev": kwargs["n_lebedev"],
        }

    def apply(self, source):
        return self.scale * ATOMIC_L1_PAIRING.source_to_field_dual(source)

    def adjoint(self, field_cotangent):
        return self.scale * ATOMIC_L1_PAIRING.field_to_source_dual(field_cotangent)

    def full_position_vjp(self, source, field_cotangent):
        contraction = float(
            np.vdot(ATOMIC_L1_PAIRING.field_to_source_dual(field_cotangent), source)
        )
        return 2.0 * _BETA * contraction * self.positions


class _MissingClearanceReactionMap(_FakeReactionMap):
    minimum_cavity_active_set_clearance_angstrom = None


def _atoms():
    return Atoms(
        "CO",
        positions=[[0.1, -0.2, 0.3], [1.2, 0.4, -0.1]],
        info={"charge": 0, "mult": 1},
    )


def _backend():
    return AtomicL1PyDDXPCMBackend(
        _atoms(),
        [1.7, 1.5],
        dielectric=20.0,
        lmax=7,
        n_lebedev=302,
        solver_tolerance=1.0e-12,
        _map_factory=_FakeReactionMap,
    )


def test_atomic_l1_pyddx_wraps_one_reciprocal_map_without_changing_spaces():
    backend = _backend()
    atoms = _atoms()
    source = np.linspace(-0.4, 0.5, 8).reshape(2, 4)
    direction = np.linspace(0.2, -0.1, 8).reshape(2, 4)
    field_cotangent = np.linspace(-0.3, 0.6, 8).reshape(2, 4)
    scale = 1.0 + _BETA * float(np.vdot(atoms.positions, atoms.positions))

    expected_field = scale * ATOMIC_L1_PAIRING.source_to_field_dual(source)
    np.testing.assert_allclose(backend.evaluate_field(atoms, source), expected_field)
    np.testing.assert_allclose(
        backend.source_jvp(atoms, source, direction),
        scale * ATOMIC_L1_PAIRING.source_to_field_dual(direction),
    )
    np.testing.assert_allclose(
        backend.source_vjp(atoms, source, field_cotangent),
        scale * ATOMIC_L1_PAIRING.field_to_source_dual(field_cotangent),
    )
    contraction = float(
        np.vdot(ATOMIC_L1_PAIRING.field_to_source_dual(field_cotangent), source)
    )
    np.testing.assert_allclose(
        backend.coordinate_vjp(atoms, source, field_cotangent).reshape(2, 3),
        2.0 * _BETA * contraction * atoms.positions,
    )
    assert backend.energy(atoms, source) == pytest.approx(
        0.5 * ATOMIC_L1_PAIRING.pair(source, expected_field)
    )
    assert backend.fixed_topology is False
    assert backend.linear_response is True
    assert backend.reciprocal is True
    assert backend.capabilities.enabled_tiers == ()


def test_atomic_l1_pyddx_state_is_immutable_and_configuration_bound():
    backend = _backend()
    atoms = _atoms()
    source = np.zeros((2, 4))
    source[:, 0] = (-0.3, 0.3)
    state = backend.build_state(atoms, source)
    assert state.source.flags.writeable is False
    assert state.field.flags.writeable is False
    assert state.runtime_provenance["backend"] == "fake-pyddx"
    assert len(state.configuration_sha256) == 64
    assert len(state.geometry_sha256) == 64
    assert state.cavity_topology_sha256 == "1" * 64
    assert state.cavity_active_node_count == 2
    assert state.minimum_cavity_active_set_clearance_angstrom == 0.25
    topology = backend.topology_state(atoms)
    assert topology["cavity_topology_sha256"] == "1" * 64
    assert topology["minimum_cavity_active_set_clearance_angstrom"] == 0.25
    assert len(backend.configuration_sha256()) == 64
    assert len(backend.provenance_sha256) == 64
    with pytest.raises(AttributeError, match="immutable"):
        backend.configuration_contract_id = "changed"


def test_atomic_l1_pyddx_rejects_atom_reordering_periodicity_and_bad_grid():
    backend = _backend()
    source = np.zeros((2, 4))
    reordered = Atoms(
        "OC",
        positions=_atoms().positions,
        info={"charge": 0, "mult": 1},
    )
    with pytest.raises(ValueError, match="identity/order"):
        backend.evaluate_field(reordered, source)
    periodic = _atoms()
    periodic.pbc = True
    with pytest.raises(ValueError, match="nonperiodic"):
        backend.evaluate_field(periodic, source)
    with pytest.raises(ValueError, match="n_lebedev"):
        AtomicL1PyDDXPCMBackend(
            _atoms(),
            [1.7, 1.5],
            dielectric=20.0,
            lmax=7,
            n_lebedev=0,
            _map_factory=_FakeReactionMap,
        )

    missing_clearance = AtomicL1PyDDXPCMBackend(
        _atoms(),
        [1.7, 1.5],
        dielectric=20.0,
        lmax=7,
        n_lebedev=302,
        _map_factory=_MissingClearanceReactionMap,
    )
    with pytest.raises(RuntimeError, match="numeric cavity active-set clearance"):
        missing_clearance.topology_state(_atoms())


def test_real_pyddx_cavity_topology_is_translation_invariant_but_lab_grid_bound():
    pytest.importorskip("pyddx")
    water = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.927297, 0.0],
        ],
    )
    backend = AtomicL1PyDDXPCMBackend(
        water,
        [1.52, 1.2, 1.2],
        dielectric=78.39,
        lmax=7,
        n_lebedev=302,
        solver_tolerance=1.0e-12,
    )
    base = backend.topology_state(water)
    assert base["minimum_cavity_active_set_clearance_angstrom"] == pytest.approx(
        5.826261813812085e-9, rel=1.0e-8, abs=1.0e-15
    )
    translated = water.copy()
    translated.positions += np.asarray([1.2, -0.7, 0.4])
    axis = np.asarray([0.3, 0.4, 0.5])
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    )
    rotated = water.copy()
    rotated.positions = water.positions @ rotation.T

    translated_topology = backend.topology_state(translated)
    assert (
        translated_topology["cavity_topology_sha256"] == base["cavity_topology_sha256"]
    )
    assert translated_topology[
        "minimum_cavity_active_set_clearance_angstrom"
    ] == pytest.approx(base["minimum_cavity_active_set_clearance_angstrom"])
    assert backend.topology_state(rotated)["cavity_topology_sha256"] != (
        base["cavity_topology_sha256"]
    )
