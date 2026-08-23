from __future__ import annotations

import hashlib
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives import RichardsonScalarForce, RichardsonScalarHessian
from maple.solvation.experimental.mace_polar_frozen_ddx import (
    MACEPolarFrozenSourceDDXPES,
    SolventEnergyState,
)
from maple.solvation.models.base import VacuumState, array_sha256, model_input_sha256


class _Space:
    component_count = 8

    def __init__(self, identity: str) -> None:
        self.identity = identity

    def metadata_hash(self) -> str:
        return hashlib.sha256(self.identity.encode()).hexdigest()

    def validate(self, values, *, atom_count: int, name: str = "values"):
        array = np.asarray(values, dtype=float)
        if array.shape != (atom_count, 8) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} is invalid")
        return np.array(array, copy=True)

    def total_charge(self, source, *, atom_count: int) -> float:
        return float(np.sum(self.validate(source, atom_count=atom_count)[:, 0]))


SOURCE_SPACE = _Space("test-radial-source")
FIELD_SPACE = _Space("test-radial-field")


class _LinearPureMACEPolar:
    provider_id = "test.pure-mace-polar.radial.v1"
    provenance_sha256 = "a" * 64
    coupling_id = "test-radial-coupling-v1"
    source_space = SOURCE_SPACE
    field_space = FIELD_SPACE
    provenance = SimpleNamespace(model_family="MACE-POLAR-1-radial-GTO-response")

    def __init__(self) -> None:
        rng = np.random.default_rng(20260816)
        self._source_offset = rng.normal(scale=0.04, size=16)
        self._source_matrix = rng.normal(scale=0.025, size=(16, 6))
        # Exact neutral source for every geometry.
        self._source_offset[8] = -self._source_offset[0]
        self._source_matrix[8] = -self._source_matrix[0]
        self._vacuum_hessian = np.diag(np.linspace(0.7, 1.4, 6))
        self._vacuum_hessian[0, 3] = self._vacuum_hessian[3, 0] = -0.08

    def configuration_sha256(self) -> str:
        return "b" * 64

    def _source(self, geometry: object) -> np.ndarray:
        positions = np.asarray(geometry.positions, dtype=float).reshape(-1)
        return (self._source_offset + self._source_matrix @ positions).reshape(2, 8)

    def evaluate_vacuum(self, geometry: object, *, need_forces: bool) -> VacuumState:
        positions = np.asarray(geometry.positions, dtype=float).reshape(-1)
        energy = float(0.5 * positions @ self._vacuum_hessian @ positions)
        forces = None
        if need_forces:
            forces = -(self._vacuum_hessian @ positions).reshape(2, 3)
        return VacuumState(
            provider_id=self.provider_id,
            provenance_sha256=self.provenance_sha256,
            model_input_sha256=model_input_sha256(geometry),
            atom_count=2,
            energy_eV=energy,
            need_forces=need_forces,
            forces_eV_per_A=forces,
        )

    def evaluate_source(
        self, geometry: object, field: object, *, need_fixed_field_forces: bool
    ):
        values = FIELD_SPACE.validate(field, atom_count=2)
        assert np.array_equal(values, np.zeros((2, 8)))
        source = self._source(geometry)
        return SimpleNamespace(
            source=source,
            model_input_sha256=model_input_sha256(geometry),
            input_field_sha256=array_sha256(values, name="field"),
            need_fixed_field_forces=need_fixed_field_forces,
        )

    def source_position_vjp(
        self, geometry: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        del geometry
        FIELD_SPACE.validate(field, atom_count=2)
        cotangent = SOURCE_SPACE.validate(source_cotangent, atom_count=2)
        return (self._source_matrix.T @ cotangent.reshape(-1)).reshape(2, 3)


class _QuadraticMovingDDX:
    provider_id = "test.ddx.radial.v1"
    provenance_sha256 = "c" * 64
    coupling_id = _LinearPureMACEPolar.coupling_id
    source_space = SOURCE_SPACE
    field_space = FIELD_SPACE
    symbols = ("H", "H")

    def __init__(self, *, topology_switch: bool = False) -> None:
        self._diagonal = np.linspace(0.08, 0.31, 16)
        self._coordinate_scale = np.asarray([0.04, -0.03, 0.02, 0.01, -0.02, 0.03])
        self._topology_switch = topology_switch

    def configuration_sha256(self) -> str:
        return ("d" if not self._topology_switch else "e") * 64

    def build_state_with_fixed_source_coordinate_gradient(
        self, geometry: object, source: object
    ):
        positions = np.asarray(geometry.positions, dtype=float).reshape(-1)
        values = SOURCE_SPACE.validate(source, atom_count=2)
        vector = values.reshape(-1)
        scale = 1.0 + float(self._coordinate_scale @ positions)
        field = (scale * self._diagonal * vector).reshape(2, 8)
        base_energy = 0.5 * float(np.dot(vector, self._diagonal * vector))
        energy = scale * base_energy
        topology = (
            "positive-x" if self._topology_switch and positions[0] > 0.0 else "fixed"
        )
        state_hash = canonical_metadata_sha256(
            {
                "geometry": geometry_sha256(geometry),
                "source": values.tolist(),
                "energy": energy,
                "topology": topology,
            }
        )
        state = SimpleNamespace(
            source=values,
            reaction_field=field,
            polarization_energy_ev=energy,
            state_hash=state_hash,
            cavity_topology_sha256=hashlib.sha256(topology.encode()).hexdigest(),
        )
        gradient = (base_energy * self._coordinate_scale).reshape(2, 3)
        return state, gradient

    def build_state(self, geometry: object, source: object):
        return self.build_state_with_fixed_source_coordinate_gradient(geometry, source)[
            0
        ]


class _QuadraticSolventTerm:
    provider_id = "test.smooth-solvent-term.v1"

    def __init__(self) -> None:
        self._diagonal = np.linspace(0.03, 0.09, 6)

    def configuration_sha256(self) -> str:
        return "f" * 64

    def evaluate(self, geometry: object, *, need_gradient: bool) -> SolventEnergyState:
        positions = np.asarray(geometry.positions, dtype=float).reshape(-1)
        energy = 0.5 * float(np.dot(self._diagonal, positions**2))
        gradient = (self._diagonal * positions).reshape(2, 3) if need_gradient else None
        return SolventEnergyState(
            provider_id=self.provider_id,
            configuration_sha256=self.configuration_sha256(),
            geometry_sha256=geometry_sha256(geometry),
            topology_id="fixed-smooth-solvent-grid",
            atom_count=2,
            energy_eV=energy,
            gradient_eV_per_A=gradient,
        )


def _atoms() -> Atoms:
    return Atoms(
        "H2",
        positions=np.asarray([[-0.45, 0.12, -0.08], [0.51, -0.16, 0.11]]),
        info={"charge": 0, "mult": 1},
    )


def _pes(*, topology_switch: bool = False) -> MACEPolarFrozenSourceDDXPES:
    return MACEPolarFrozenSourceDDXPES(
        model=_LinearPureMACEPolar(),
        continuum=_QuadraticMovingDDX(topology_switch=topology_switch),
        solvent_term=_QuadraticSolventTerm(),
        numerical_force_backend=RichardsonScalarForce(
            coarse_step_angstrom=1.0e-3,
            maximum_error_eV_per_A=1.0e-7,
        ),
        hessian_backend=RichardsonScalarHessian(
            coarse_step_angstrom=1.0e-3,
            maximum_error_eV_per_A2=1.0e-7,
            maximum_antisymmetry_eV_per_A2=1.0e-7,
        ),
    )


def test_pure_frozen_source_chain_rule_force_matches_the_registered_scalar():
    atoms = _atoms()
    pes = _pes()
    state = pes.solve(atoms)
    evaluated = pes.evaluate_forces(atoms, central_state=state)
    assert state.total_energy_eV == pytest.approx(
        state.vacuum_energy_eV + state.polarization_energy_eV + state.cds_energy_eV,
        abs=0.0,
    )
    assert state.solvation_energy_eV == pytest.approx(
        state.polarization_energy_eV + state.cds_energy_eV, abs=0.0
    )
    assert state.source_total_charge_e == pytest.approx(0.0, abs=2.0e-16)

    finite_difference = np.empty((2, 3))
    step = 2.0e-5
    for atom in range(2):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            finite_difference[atom, axis] = -(
                pes.get_potential_energy(plus) - pes.get_potential_energy(minus)
            ) / (2.0 * step)
    np.testing.assert_allclose(
        evaluated.total_forces_eV_per_A,
        finite_difference,
        atol=2.0e-10,
        rtol=0.0,
    )
    numerical = pes.numerical_force_audit(atoms, central_state=state)
    np.testing.assert_allclose(
        numerical.forces_eV_per_A,
        evaluated.total_forces_eV_per_A,
        atol=2.0e-10,
        rtol=0.0,
    )
    assert numerical.maximum_error_estimate_eV_per_A < 1.0e-7


def test_molecular_virial_is_the_same_scalar_homogeneous_strain_derivative():
    atoms = _atoms()
    pes = _pes()
    evaluated = pes.evaluate_forces(atoms)
    virial = pes.molecular_virial(atoms, force_evaluation=evaluated)
    origin = virial.origin_angstrom
    strain = np.asarray([[0.21, -0.04, 0.03], [-0.04, -0.12, 0.02], [0.03, 0.02, 0.08]])
    step = 1.0e-5
    energies = []
    for sign in (1.0, -1.0):
        displaced = atoms.copy()
        transform = np.eye(3) + sign * step * strain
        displaced.positions = origin + (atoms.positions - origin) @ transform.T
        energies.append(pes.get_potential_energy(displaced))
    finite_difference = (energies[0] - energies[1]) / (2.0 * step)
    analytic = float(np.vdot(virial.strain_gradient_eV, strain))
    assert analytic == pytest.approx(finite_difference, abs=3.0e-10)
    np.testing.assert_allclose(
        virial.symmetric_virial_eV,
        0.5 * (virial.raw_virial_eV + virial.raw_virial_eV.T),
        atol=0.0,
        rtol=0.0,
    )


def test_hvp_and_full_hessian_follow_the_same_conservative_force():
    atoms = _atoms()
    pes = _pes()
    direction = np.linspace(-0.3, 0.4, 6).reshape(2, 3)
    evaluated = pes.hessian_vector_product(atoms, direction)
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    finite_difference = -(pes.get_forces(plus) - pes.get_forces(minus)) / (2.0 * step)
    np.testing.assert_allclose(
        evaluated.hvp_eV_per_A2, finite_difference, atol=2.0e-9, rtol=0.0
    )

    hessian = pes.evaluate_hessian(atoms)
    np.testing.assert_allclose(
        hessian.hessian_eV_per_A2,
        hessian.hessian_eV_per_A2.T,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        hessian.hessian_eV_per_A2 @ direction.reshape(-1),
        evaluated.hvp_eV_per_A2.reshape(-1),
        atol=3.0e-9,
        rtol=0.0,
    )
    assert hessian.maximum_error_estimate_eV_per_A2 < 1.0e-7
    assert hessian.maximum_antisymmetry_eV_per_A2 < 1.0e-7


def test_hessian_fails_closed_if_ddx_exposed_node_topology_changes():
    atoms = _atoms()
    atoms.positions[0, 0] = 0.0
    pes = _pes(topology_switch=True)
    direction = np.zeros((2, 3))
    direction[0, 0] = 1.0
    with pytest.raises(RuntimeError, match="changed the continuum topology"):
        pes.hessian_vector_product(atoms, direction)


def test_pure_route_rejects_a_mace_mdp_model_family():
    model = _LinearPureMACEPolar()
    model.provenance = SimpleNamespace(model_family="MACE-MDP-hybrid")
    with pytest.raises(TypeError, match="pure MACE-POLAR"):
        MACEPolarFrozenSourceDDXPES(
            model=model,
            continuum=_QuadraticMovingDDX(),
            solvent_term=_QuadraticSolventTerm(),
        )
