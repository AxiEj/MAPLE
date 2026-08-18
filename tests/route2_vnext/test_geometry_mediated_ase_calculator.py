from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.coupling.geometry_mediated_ase import (
    GeometryMediatedOperationalDomainError,
    GeometryMediatedScalarASECalculator,
)


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


class _Model:
    def __init__(self, *, margin: float = 1.0, topology: str = "model") -> None:
        self.margin = margin
        self.topology = topology

    def neighbor_topology(self, atoms):
        return SimpleNamespace(
            as_dict=lambda: {
                "configuration_sha256": _sha("model-config"),
                "topology_sha256": _sha(self.topology),
                "minimum_cutoff_margin_angstrom": self.margin,
            }
        )


class _Continuum:
    def __init__(
        self,
        *,
        point_margin: float = 1.0,
        sphere_margin: float = 1.0,
        topology: str = "continuum",
    ) -> None:
        self.point_margin = point_margin
        self.sphere_margin = sphere_margin
        self.topology = topology

    def topology_state(self, atoms):
        return {
            "configuration_sha256": _sha("continuum-config"),
            "cavity_topology_sha256": _sha(self.topology),
            "coefficient_count": 4 * len(atoms),
            "minimum_point_source_shell_margin_angstrom": self.point_margin,
            "minimum_sphere_tangency_margin_angstrom": self.sphere_margin,
            "point_source_topology_sha256": _sha(f"{self.topology}:point"),
            "sphere_pair_topology_sha256": _sha(f"{self.topology}:sphere"),
        }


@dataclass
class _Scalar:
    model: _Model
    continuum: _Continuum

    def evaluate(self, atoms):
        positions = np.asarray(atoms.positions, dtype=float)
        energy = float(np.vdot(positions, positions))
        return SimpleNamespace(
            energy=SimpleNamespace(total_energy_eV=energy),
            forces_eV_per_A=-2.0 * positions,
        )


def _atoms() -> Atoms:
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.927297, 0.0],
        ],
        info={"charge": 0, "mult": 1},
    )


def _calculator(*, model=None, continuum=None):
    model = _Model() if model is None else model
    continuum = _Continuum() if continuum is None else continuum
    return GeometryMediatedScalarASECalculator(_Scalar(model, continuum))


def test_geometry_mediated_ase_calculator_exposes_only_same_scalar_energy_and_forces():
    atoms = _atoms()
    calculator = _calculator()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert calculator.implemented_properties == ("energy", "free_energy", "forces")
    assert energy == pytest.approx(float(np.vdot(atoms.positions, atoms.positions)))
    np.testing.assert_allclose(forces, -2.0 * atoms.positions, atol=0.0, rtol=0.0)
    assert calculator.results["free_energy"] == energy
    assert calculator.last_domain_guard["gate_passed"] is True
    assert calculator.last_evaluation is not None
    assert "stress" not in calculator.implemented_properties
    assert "hessian" not in calculator.implemented_properties


@pytest.mark.parametrize(
    ("model", "continuum", "gate"),
    [
        (_Model(margin=0.019), _Continuum(), "neighbor_cutoff"),
        (_Model(), _Continuum(point_margin=0.019), "point_source_shell"),
        (_Model(), _Continuum(sphere_margin=0.019), "sphere_tangency"),
    ],
)
def test_geometry_mediated_ase_calculator_rejects_event_unsafe_current_geometry(
    model, continuum, gate
):
    calculator = _calculator(model=model, continuum=continuum)
    atoms = _atoms()
    atoms.calc = calculator
    with pytest.raises(GeometryMediatedOperationalDomainError, match=gate):
        atoms.get_potential_energy()
    assert calculator.results == {}
    assert calculator.last_evaluation is None


def test_geometry_mediated_ase_calculator_rejects_topology_crossing_between_calls():
    model = _Model()
    continuum = _Continuum()
    calculator = _calculator(model=model, continuum=continuum)
    atoms = _atoms()
    atoms.calc = calculator
    atoms.get_forces()

    model.topology = "changed-model-topology"
    atoms.positions[0, 0] += 1.0e-4
    with pytest.raises(GeometryMediatedOperationalDomainError, match="same_model"):
        atoms.get_forces()
    assert calculator.last_evaluation is not None


def test_geometry_mediated_ase_calculator_rejects_periodic_systems_and_unsupported_properties():
    atoms = _atoms()
    atoms.pbc = True
    calculator = _calculator()
    with pytest.raises(GeometryMediatedOperationalDomainError, match="non-periodic"):
        calculator.calculate(atoms, ("energy",))
    with pytest.raises(NotImplementedError, match="stress"):
        calculator.calculate(_atoms(), ("stress",))
