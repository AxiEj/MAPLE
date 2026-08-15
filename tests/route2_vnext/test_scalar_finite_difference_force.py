from __future__ import annotations

import hashlib

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.derivatives import (
    RichardsonScalarForce,
    ScalarEnergySample,
)


class _PolynomialScalar:
    provider_id = "test.scalar-polynomial.v1"

    def __init__(self, *, topology_switch: bool = False) -> None:
        self.topology_switch = topology_switch

    def configuration_sha256(self) -> str:
        return ("1" if not self.topology_switch else "2") * 64

    @staticmethod
    def energy(positions: np.ndarray) -> float:
        weights = np.arange(1, positions.size + 1, dtype=float).reshape(positions.shape)
        return float(np.sum(weights * (positions**2 + 0.2 * positions**6)))

    def sample(self, geometry: object) -> ScalarEnergySample:
        positions = np.asarray(geometry.positions, dtype=float)
        state = hashlib.sha256(positions.tobytes()).hexdigest()
        topology = (
            "positive-x" if self.topology_switch and positions[0, 0] > 0.0 else "fixed"
        )
        return ScalarEnergySample(
            energy_eV=self.energy(positions),
            state_sha256=state,
            topology_id=topology,
        )


def _atoms() -> Atoms:
    return Atoms(
        "HeNe",
        positions=np.asarray([[-0.4, 0.2, 0.1], [0.8, -0.3, 0.5]], dtype=float),
    )


def test_richardson_force_is_fourth_order_scalar_gradient():
    atoms = _atoms()
    provider = _PolynomialScalar()
    backend = RichardsonScalarForce(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A=1.0e-3,
    )
    result = backend.evaluate(provider, atoms)

    positions = np.asarray(atoms.positions)
    weights = np.arange(1, positions.size + 1, dtype=float).reshape(positions.shape)
    expected = -weights * (2.0 * positions + 1.2 * positions**5)
    assert result.forces_eV_per_A == pytest.approx(expected, abs=2.0e-8)
    assert result.maximum_error_estimate_eV_per_A < 1.0e-3
    assert len(result.displaced_state_sha256) == 4 * positions.size
    assert result.evaluation_sha256
    with pytest.raises(ValueError):
        result.forces_eV_per_A.setflags(write=True)


def test_richardson_force_fails_closed_on_topology_change():
    atoms = _atoms()
    atoms.positions[0, 0] = 0.0
    with pytest.raises(RuntimeError, match="changed the continuum topology"):
        RichardsonScalarForce().evaluate(_PolynomialScalar(topology_switch=True), atoms)


def test_focused_component_uses_the_same_stencil_contract():
    atoms = _atoms()
    backend = RichardsonScalarForce(
        coarse_step_angstrom=1.0e-2,
        maximum_error_eV_per_A=1.0e-3,
    )
    component = backend.evaluate_component(
        _PolynomialScalar(), atoms, atom_index=1, axis_index=2
    )
    position = atoms.positions[1, 2]
    expected = -6.0 * (2.0 * position + 1.2 * position**5)
    assert component.force_eV_per_A == pytest.approx(expected, abs=2.0e-8)
    assert component.error_estimate_eV_per_A < 1.0e-3
    assert len(component.displaced_state_sha256) == 4


def test_richardson_force_fails_closed_when_error_budget_is_too_small():
    with pytest.raises(RuntimeError, match="error estimate exceeds"):
        RichardsonScalarForce(
            coarse_step_angstrom=0.1,
            maximum_error_eV_per_A=1.0e-12,
        ).evaluate(_PolynomialScalar(), _atoms())
