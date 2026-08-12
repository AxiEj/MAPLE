from __future__ import annotations

from dataclasses import dataclass
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from maple.solvation.coupling import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_PAIRING,
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
    FieldDualSpace,
    PairingMetric,
    SourceSpace,
    validate_adjoint_dot_product,
)


def test_authoritative_pairing_order_units_and_gauge():
    assert ATOMIC_L1_PAIRING.field_to_source_indices == (0, 2, 3, 1)
    assert ATOMIC_L1_SOURCE_SPACE.components[0] == "net_monopole"
    assert ATOMIC_L1_FIELD_DUAL_SPACE.components == (
        "potential", "potential_gradient_x", "potential_gradient_y", "potential_gradient_z"
    )
    assert ATOMIC_L1_FIELD_DUAL_SPACE.units[0] == "eV/e"
    assert ATOMIC_L1_FIELD_DUAL_SPACE.gauge == "continuum-zero-at-infinity"
    source = np.array([[1.0, 2.0, 3.0, 4.0]])
    field = np.array([[5.0, 7.0, 11.0, 13.0]])
    assert ATOMIC_L1_PAIRING.pair(source, field) == 1*5 + 2*11 + 3*13 + 4*7


def test_contract_metadata_is_deeply_immutable_and_constructor_inputs_are_detached():
    components = ["q", "x"]
    units = ["e", "e*angstrom"]
    permutation = [0, 1]
    metric = PairingMetric(
        "test.metric", components, ["v", "g"], units,
        ["eV/e", "eV/(e*angstrom)"], permutation, "zero", "positive",
    )
    source_space = SourceSpace("test.source", "test", components, units)
    components[0] = "mutated"
    units[0] = "mutated"
    permutation[0] = 1
    assert metric.source_components == ("q", "x")
    assert metric.field_to_source_indices == (0, 1)
    assert source_space.components == ("q", "x")
    with pytest.raises(FrozenInstanceError):
        source_space.scalar_id = "mutated"


def test_pairing_dual_transforms_accept_batched_blocks():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(2, 3, 4))
    field = rng.normal(size=(2, 3, 4))
    q_field = ATOMIC_L1_PAIRING.field_to_source_dual(field)
    qt_source = ATOMIC_L1_PAIRING.source_to_field_dual(source)
    assert q_field.shape == field.shape
    assert qt_source.shape == source.shape
    assert np.vdot(source, q_field) == pytest.approx(np.vdot(qt_source, field))
    with pytest.raises(ValueError):
        ATOMIC_L1_PAIRING.field_to_source_dual(np.full((2, 4), np.inf))


def test_space_and_metric_metadata_validation_fails_closed():
    with pytest.raises(ValueError, match="equal lengths"):
        SourceSpace("test", "test", ("q", "x"), ("e",))
    with pytest.raises(ValueError, match="outside"):
        SourceSpace("test", "test", ("q",), ("e",), charge_component=1)
    with pytest.raises(ValueError, match="permutation"):
        PairingMetric(
            "test", ("q", "x"), ("v", "g"), ("e", "ea"),
            ("ev/e", "ev/ea"), (0, 0), "zero", "positive",
        )
    with pytest.raises(TypeError, match="string"):
        SourceSpace(3, "test", ("q",), ("e",))


@pytest.mark.parametrize("atom_count,total_charge", [(1, 0.0), (2, -1.0), (5, 2.0)])
def test_affine_coordinates_charge_roundtrip_and_cotangent_pairing(atom_count, total_charge):
    coordinates = AffineChargeCoordinates(atom_count, total_charge, 0.7, 1.3)
    rng = np.random.default_rng(atom_count)
    y = rng.normal(size=coordinates.reduced_dimension)
    source = coordinates.expand(y)
    assert abs(ATOMIC_L1_SOURCE_SPACE.total_charge(source, atom_count=atom_count) - total_charge) <= 1e-12
    np.testing.assert_allclose(coordinates.reduce(source), y, atol=1e-13)
    np.testing.assert_allclose(coordinates.project_affine(source), source, atol=1e-13)
    dy = rng.normal(size=coordinates.reduced_dimension)
    cbar = rng.normal(size=(atom_count, 4))
    assert np.vdot(coordinates.expand_direction(dy), cbar) == pytest.approx(
        np.vdot(dy, coordinates.reduce_source_cotangent(cbar)), abs=1e-12
    )
    ybar = rng.normal(size=coordinates.reduced_dimension)
    dc = coordinates.project_tangent(rng.normal(size=(atom_count, 4)))
    assert np.vdot(coordinates.reduce_tangent(dc), ybar) == pytest.approx(
        np.vdot(dc, coordinates.lift_reduced_cotangent(ybar)), abs=1e-12
    )


def test_future_six_component_source_with_nonzero_charge_index():
    source_space = SourceSpace(
        "test.atomic-l2", "six-component test source",
        ("a", "b", "charge", "d", "e", "f"),
        ("u", "u", "e", "u", "u", "u"), charge_component=2,
    )
    coordinates = AffineChargeCoordinates(
        7, -2.0, source_space=source_space,
        component_scales=(0.5, 0.7, 1.1, 1.3, 1.7, 2.0),
    )
    rng = np.random.default_rng(81)
    y = rng.normal(size=coordinates.reduced_dimension)
    source = coordinates.expand(y)
    assert source.shape == (7, 6)
    assert abs(source_space.total_charge(source, atom_count=7) + 2.0) <= 1e-12
    np.testing.assert_allclose(coordinates.reduce(source), y, atol=1e-13)
    dy = rng.normal(size=coordinates.reduced_dimension)
    cbar = rng.normal(size=(7, 6))
    assert np.vdot(coordinates.expand_direction(dy), cbar) == pytest.approx(
        np.vdot(dy, coordinates.reduce_source_cotangent(cbar)), abs=1e-12
    )


def test_debug_dense_views_match_matrix_free_maps_for_small_system():
    coordinates = AffineChargeCoordinates(3, 0.0, 0.4, 1.7)
    rng = np.random.default_rng(19)
    y = rng.normal(size=coordinates.reduced_dimension)
    dc = rng.normal(size=(3, 4))
    np.testing.assert_allclose(
        (coordinates.T @ y).reshape(3, 4), coordinates.expand_direction(y), atol=1e-14
    )
    np.testing.assert_allclose(
        coordinates.T_plus @ dc.reshape(-1), coordinates.reduce_tangent(dc), atol=1e-14
    )
    np.testing.assert_allclose(
        coordinates.T_plus @ coordinates.T,
        np.eye(coordinates.reduced_dimension), atol=1e-14,
    )


def test_large_coordinates_are_matrix_free_and_dense_debug_is_guarded():
    coordinates = AffineChargeCoordinates(100_000, 1.0)
    direction = np.zeros(coordinates.reduced_dimension)
    expanded = coordinates.expand_direction(direction)
    assert expanded.shape == (100_000, 4)
    assert np.sum(expanded[:, 0]) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="debug-only"):
        _ = coordinates.T
    with pytest.raises(ValueError, match="debug-only"):
        _ = coordinates.T_plus


def test_field_space_rejects_incompatible_source_binding():
    incompatible = SourceSpace(
        "test.incompatible", "wrong units", ATOMIC_L1_SOURCE_SPACE.components,
        ("wrong",) * 4,
    )
    with pytest.raises(ValueError, match="incompatible"):
        FieldDualSpace(
            "test.field", "test", ATOMIC_L1_FIELD_DUAL_SPACE.components,
            ATOMIC_L1_FIELD_DUAL_SPACE.units, incompatible, ATOMIC_L1_PAIRING,
        )


@dataclass
class _LinearCoupling:
    matrix: np.ndarray
    scalar_id: str = "test.linear-coupling.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE

    def apply_source(self, geometry, source):
        return self.matrix @ source.reshape(-1)

    def apply_adjoint(self, geometry, surface_cotangent):
        q = np.kron(np.eye(geometry), ATOMIC_L1_PAIRING.block)
        return np.linalg.solve(q, self.matrix.T @ surface_cotangent).reshape(geometry, 4)


def test_coupling_operator_adjoint_dot_product():
    rng = np.random.default_rng(9)
    atoms = 3
    operator = _LinearCoupling(rng.normal(size=(7, atoms * 4)))
    evidence = validate_adjoint_dot_product(
        operator, atoms, rng.normal(size=(atoms, 4)), rng.normal(size=7), atom_count=atoms
    )
    assert evidence.passed
    assert evidence.absolute_error <= 1e-10


@pytest.mark.parametrize(
    "action",
    [
        lambda: ATOMIC_L1_SOURCE_SPACE.validate(np.zeros((2, 3)), atom_count=2),
        lambda: ATOMIC_L1_FIELD_DUAL_SPACE.validate([[0, 0, 0, np.nan]], atom_count=1),
        lambda: AffineChargeCoordinates(2, 0).expand(np.zeros(3)),
        lambda: ATOMIC_L1_PAIRING.pair(np.zeros((1, 4)), np.zeros((2, 4))),
    ],
)
def test_invalid_shape_and_nonfinite_fail_closed(action):
    with pytest.raises(ValueError):
        action()
