from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.models.field_energy import (
    FieldEnergyFunctional,
    TorchGeometry,
)
from maple.solvation.models.mace_polar_variational import (
    MACE_POLAR_VARIATIONAL_DUALITY_MAP,
)

ROOT = Path(__file__).resolve().parents[2]


class QuadraticFieldEnergy(FieldEnergyFunctional):
    __slots__ = ()

    def _energy_torch(self, geometry: TorchGeometry, reduced_field):
        torch = importlib.import_module("torch")
        position_factor = 1.0 + 0.05 * torch.sum(geometry.positions**2)
        return 0.5 * position_factor * torch.dot(reduced_field, reduced_field)


class LinearFieldEnergy(FieldEnergyFunctional):
    __slots__ = ()

    def _energy_torch(self, geometry: TorchGeometry, reduced_field):
        del geometry
        return 0.3 * reduced_field.sum()


class ConstantFieldEnergy(FieldEnergyFunctional):
    __slots__ = ()

    def _energy_torch(self, geometry: TorchGeometry, reduced_field):
        torch = importlib.import_module("torch")
        del geometry, reduced_field
        return torch.as_tensor(2.5, dtype=self._torch_dtype)


def _atoms() -> Atoms:
    return Atoms(
        "HC",
        positions=np.asarray([[0.1, -0.2, 0.3], [1.2, 0.4, -0.1]]),
        info={"charge": 0, "mult": 1},
    )


def _functional() -> QuadraticFieldEnergy:
    torch = pytest.importorskip("torch")
    return QuadraticFieldEnergy(
        duality_map=MACE_POLAR_VARIATIONAL_DUALITY_MAP,
        dtype=torch.float64,
        device="cpu",
    )


def test_field_energy_contract_imports_without_torch_or_model_runtimes():
    script = r"""
import sys
for name in ("torch", "mace", "aimnet2calc", "pyscf"):
    sys.modules[name] = None
from maple.solvation.models.field_energy import GaugeReducedDualityMap
from maple.solvation.models.mace_polar_variational import (
    MACE_POLAR_VARIATIONAL_DUALITY_MAP,
)
assert isinstance(MACE_POLAR_VARIATIONAL_DUALITY_MAP, GaugeReducedDualityMap)
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_duality_map_exactly_closes_reduced_pairing_charge_and_gauge():
    duality = MACE_POLAR_VARIATIONAL_DUALITY_MAP
    coordinates = duality.coordinates(atom_count=3, total_charge=-1.0)
    rng = np.random.default_rng(20260814)
    reduced = rng.normal(size=coordinates.reduced_dimension)
    source = coordinates.expand(rng.normal(size=coordinates.reduced_dimension))
    field = duality.lift_field(
        reduced,
        atom_count=3,
        total_charge=-1.0,
        gauge_potential=0.37,
    )
    recovered, gauge = duality.decompose_field(field, atom_count=3, total_charge=-1.0)
    np.testing.assert_allclose(recovered, reduced, atol=3e-14, rtol=0.0)
    assert gauge == pytest.approx(0.37, abs=3e-14)
    assert duality.reduce_field(
        field, atom_count=3, total_charge=-1.0
    ) == pytest.approx(reduced, abs=3e-14)
    pairing = duality.field_space.pair(source, field, atom_count=3)
    expected = float(np.vdot(coordinates.reduce_tangent(source), reduced) - 0.37)
    assert pairing == pytest.approx(expected, abs=4e-13)
    assert len(duality.configuration_sha256()) == 64
    assert duality.metadata()["gauge_direction"] == "g=Q^-1 a"


def test_scalar_first_source_is_full_charge_constrained_and_energy_conjugate():
    functional = _functional()
    atoms = _atoms()
    duality = functional.duality_map
    coordinates = duality.coordinates(atom_count=len(atoms), total_charge=0.0)
    rng = np.random.default_rng(72)
    reduced = rng.normal(scale=0.03, size=coordinates.reduced_dimension)
    direction = rng.normal(size=coordinates.reduced_dimension)
    source = functional.source_from_energy(atoms, reduced, total_charge=0.0)
    assert source.shape == (len(atoms), 8)
    assert duality.source_space.total_charge(
        source, atom_count=len(atoms)
    ) == pytest.approx(0.0, abs=2e-13)
    assert np.linalg.norm(source[:, (1, 5, 6, 7)]) > 1.0e-6

    step = 2.0e-6
    finite_difference = (
        functional.energy_eV(atoms, reduced + step * direction, total_charge=0.0)
        - functional.energy_eV(atoms, reduced - step * direction, total_charge=0.0)
    ) / (2.0 * step)
    field_direction = duality.lift_field(
        direction, atom_count=len(atoms), total_charge=0.0
    )
    analytic = duality.field_space.pair(source, field_direction, atom_count=len(atoms))
    forward = functional.energy_directional_derivative(
        atoms, reduced, direction, total_charge=0.0
    )
    assert analytic == pytest.approx(finite_difference, rel=2e-10, abs=2e-10)
    assert analytic == pytest.approx(forward, rel=0.0, abs=2e-13)


def test_scalar_generated_jvp_vjp_hessian_and_mixed_coordinate_derivatives():
    functional = _functional()
    atoms = _atoms()
    coordinates = functional.duality_map.coordinates(
        atom_count=len(atoms), total_charge=0.0
    )
    rng = np.random.default_rng(45)
    reduced = rng.normal(scale=0.02, size=coordinates.reduced_dimension)
    direction = rng.normal(size=coordinates.reduced_dimension)
    source_cotangent = rng.normal(size=(len(atoms), 8))

    jvp = functional.source_jvp(atoms, reduced, direction, total_charge=0.0)
    vjp = functional.source_vjp(atoms, reduced, source_cotangent, total_charge=0.0)
    assert np.vdot(jvp, source_cotangent) == pytest.approx(
        np.vdot(direction, vjp), abs=2e-12
    )
    hvp = functional.field_hvp(atoms, reduced, direction, total_charge=0.0)
    np.testing.assert_allclose(
        jvp,
        coordinates.expand_direction(hvp),
        atol=2e-13,
        rtol=0.0,
    )

    fixed_gradient = functional.fixed_field_coordinate_gradient(
        atoms, reduced, total_charge=0.0
    )
    mixed = functional.mixed_coordinate_field_vjp(
        atoms, reduced, source_cotangent, total_charge=0.0
    )
    assert fixed_gradient.shape == mixed.shape == (len(atoms), 3)
    coordinate_direction = rng.normal(size=(len(atoms), 3))
    coordinate_direction /= np.linalg.norm(coordinate_direction)
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * coordinate_direction
    minus.positions -= step * coordinate_direction
    fixed_fd = (
        functional.energy_eV(plus, reduced, total_charge=0.0)
        - functional.energy_eV(minus, reduced, total_charge=0.0)
    ) / (2.0 * step)
    assert np.vdot(fixed_gradient, coordinate_direction) == pytest.approx(
        fixed_fd, rel=3e-10, abs=3e-10
    )
    plus_source = functional.source_from_energy(plus, reduced, total_charge=0.0)
    minus_source = functional.source_from_energy(minus, reduced, total_charge=0.0)
    mixed_fd = np.vdot(source_cotangent, (plus_source - minus_source) / (2.0 * step))
    assert np.vdot(mixed, coordinate_direction) == pytest.approx(
        mixed_fd, rel=3e-10, abs=3e-10
    )


def test_linear_scalar_has_an_exact_zero_hvp_and_source_response():
    torch = pytest.importorskip("torch")
    functional = LinearFieldEnergy(
        duality_map=MACE_POLAR_VARIATIONAL_DUALITY_MAP,
        dtype=torch.float64,
        device="cpu",
    )
    atoms = _atoms()
    coordinates = functional.duality_map.coordinates(
        atom_count=len(atoms), total_charge=0.0
    )
    reduced = np.zeros(coordinates.reduced_dimension)
    direction = np.linspace(-0.1, 0.2, coordinates.reduced_dimension)
    np.testing.assert_array_equal(
        functional.field_hvp(atoms, reduced, direction, total_charge=0.0),
        0.0,
    )
    np.testing.assert_array_equal(
        functional.source_jvp(atoms, reduced, direction, total_charge=0.0),
        0.0,
    )
    np.testing.assert_array_equal(
        functional.source_vjp(
            atoms,
            reduced,
            np.ones((len(atoms), 8)),
            total_charge=0.0,
        ),
        0.0,
    )


def test_constant_scalar_returns_reference_source_and_zero_derivatives():
    torch = pytest.importorskip("torch")
    functional = ConstantFieldEnergy(
        duality_map=MACE_POLAR_VARIATIONAL_DUALITY_MAP,
        dtype=torch.float64,
        device="cpu",
    )
    atoms = _atoms()
    coordinates = functional.duality_map.coordinates(
        atom_count=len(atoms), total_charge=0.0
    )
    reduced = np.zeros(coordinates.reduced_dimension)
    direction = np.ones(coordinates.reduced_dimension)
    np.testing.assert_allclose(
        functional.source_from_energy(atoms, reduced, total_charge=0.0),
        coordinates.c_ref,
        atol=0.0,
        rtol=0.0,
    )
    assert (
        functional.energy_directional_derivative(
            atoms, reduced, direction, total_charge=0.0
        )
        == 0.0
    )
    np.testing.assert_array_equal(
        functional.fixed_field_coordinate_gradient(atoms, reduced, total_charge=0.0),
        0.0,
    )
    np.testing.assert_array_equal(
        functional.mixed_coordinate_field_vjp(
            atoms,
            reduced,
            np.ones((len(atoms), 8)),
            total_charge=0.0,
        ),
        0.0,
    )
    with pytest.raises(ValueError, match="canonical model input charge"):
        functional.energy_eV(atoms, reduced, total_charge=1.0)


def test_derivative_surface_is_final_and_cannot_be_hand_coded_or_mutated():
    with pytest.raises(TypeError, match="final"):

        class BadSource(FieldEnergyFunctional):
            def _energy_torch(self, geometry, reduced_field):
                return reduced_field.sum()

            def source_from_energy(self, *args, **kwargs):
                return None

    functional = _functional()
    with pytest.raises(AttributeError, match="immutable"):
        functional._torch_device = "cuda"
