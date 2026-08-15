from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.coupling.operator import CoordinateDerivativeUnavailable
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.models.mace_mdp_polar_hybrid import (
    MACE_MDP_POLAR_HYBRID_PROFILE_ID,
    MACE_MDP_POLAR_HYBRID_PROVIDER_ID,
    PermanentAnchoredInducedSourceModel,
)


class _Permanent:
    provider_id = "test.permanent.v1"
    model_profile_id = "test.permanent-profile.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE

    def __init__(self) -> None:
        self.digest = "1" * 64
        self.source = np.array([[-0.2, 0.03, -0.02, 0.01], [0.2, -0.01, 0.04, -0.02]])

    def configuration_sha256(self) -> str:
        return self.digest

    def evaluate_source(self, _geometry: object) -> np.ndarray:
        return self.source.copy()


class _Response:
    provider_id = "test.response.v1"
    model_profile_id = "test.response-profile.v1"
    provenance_sha256 = "2" * 64
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    long_range_evaluator_profile = (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )

    def __init__(self) -> None:
        self.digest = "3" * 64
        rng = np.random.default_rng(20260815)
        jacobian = rng.normal(scale=0.04, size=(8, 16))
        jacobian[4] = -jacobian[0]
        self.jacobian = jacobian
        self.zero_source = np.array(
            [[-0.1, 0.02, 0.01, -0.03], [0.1, -0.02, 0.02, 0.01]]
        )

    def configuration_sha256(self) -> str:
        return self.digest

    def vacuum_energy_ev(self, _geometry: object) -> float:
        return -12.5

    def evaluate_source(self, _geometry: object, field: object) -> np.ndarray:
        values = np.asarray(field, dtype=float).reshape(-1)
        return self.zero_source + (self.jacobian @ values).reshape(2, 4)

    def field_jvp(
        self, _geometry: object, _field: object, direction: object
    ) -> np.ndarray:
        return (self.jacobian @ np.asarray(direction).reshape(-1)).reshape(2, 4)

    def field_vjp(
        self, _geometry: object, _field: object, cotangent: object
    ) -> np.ndarray:
        return (self.jacobian.T @ np.asarray(cotangent).reshape(-1)).reshape(2, 8)

    def conditioned_raw_energy_ev(self, _geometry: object, field: object) -> float:
        return 0.5 * float(np.vdot(field, field))


def _geometry() -> Atoms:
    atoms = Atoms("CO", positions=[[0.0, 0.1, -0.2], [1.2, -0.1, 0.3]])
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    return atoms


def _model():
    permanent = _Permanent()
    response = _Response()
    return PermanentAnchoredInducedSourceModel(permanent, response), permanent, response


def test_hybrid_zero_field_is_exactly_the_permanent_anchor():
    model, permanent, _response = _model()
    atoms = _geometry()
    anchor = model.prepare(atoms)
    zero = np.zeros((len(atoms), 8))

    np.testing.assert_array_equal(model.induced_source(atoms, anchor, zero), 0.0)
    np.testing.assert_array_equal(
        model.total_source(atoms, anchor, zero), permanent.source
    )
    assert anchor.total_charge_e == pytest.approx(0.0, abs=1.0e-15)
    assert model.provider_id == MACE_MDP_POLAR_HYBRID_PROVIDER_ID
    assert model.model_profile_id == MACE_MDP_POLAR_HYBRID_PROFILE_ID
    assert model.capabilities == ()
    assert model.variational_functional_admitted is False
    assert model.vacuum_energy_ev(atoms) == pytest.approx(-12.5)


def test_hybrid_uses_only_the_polar_field_increment_and_preserves_charge():
    model, permanent, response = _model()
    atoms = _geometry()
    anchor = model.prepare(atoms)
    field = np.linspace(-0.03, 0.05, len(atoms) * 8).reshape(len(atoms), 8)
    expected_increment = (response.jacobian @ field.reshape(-1)).reshape(2, 4)

    np.testing.assert_allclose(
        model.induced_source(atoms, anchor, field), expected_increment
    )
    total = model.total_source(atoms, anchor, field)
    np.testing.assert_allclose(total, permanent.source + expected_increment)
    assert np.sum(total[:, 0]) == pytest.approx(
        np.sum(permanent.source[:, 0]), abs=2.0e-15
    )


def test_hybrid_rectangular_jvp_vjp_is_the_polar_increment_derivative():
    model, _permanent, _response = _model()
    atoms = _geometry()
    anchor = model.prepare(atoms)
    rng = np.random.default_rng(7)
    field = rng.normal(size=(2, 8))
    direction = rng.normal(size=(2, 8))
    cotangent = rng.normal(size=(2, 4))

    jvp = model.field_jvp(atoms, anchor, field, direction)
    vjp = model.field_vjp(atoms, anchor, field, cotangent)
    assert float(np.vdot(jvp, cotangent)) == pytest.approx(
        float(np.vdot(direction, vjp)), rel=2.0e-13, abs=2.0e-14
    )


def test_hybrid_anchor_and_configuration_fail_closed():
    model, _permanent, response = _model()
    atoms = _geometry()
    anchor = model.prepare(atoms)
    moved = atoms.copy()
    moved.positions[0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="anchor geometry"):
        model.total_source(moved, anchor, np.zeros((2, 8)))
    with pytest.raises(ValueError):
        anchor.permanent_source4.setflags(write=True)
    with pytest.raises(AttributeError, match="immutable"):
        model.provider_id = "tampered"

    response.digest = "4" * 64
    with pytest.raises(RuntimeError, match="configuration drifted"):
        model.configuration_sha256()


def test_hybrid_force_path_stays_closed_but_raw_polar_energy_is_delegated():
    model, _permanent, _response = _model()
    atoms = _geometry()
    field = np.linspace(-0.02, 0.01, 16).reshape(2, 8)
    assert model.conditioned_raw_energy_ev(atoms, field) == pytest.approx(
        0.5 * float(np.vdot(field, field))
    )
    with pytest.raises(CoordinateDerivativeUnavailable, match="MACE-MDP atomic q/p"):
        model.coordinate_vjp(atoms, field, np.ones((2, 4)))
