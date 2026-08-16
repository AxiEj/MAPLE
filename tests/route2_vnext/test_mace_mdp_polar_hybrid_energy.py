from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from ase import Atoms
from ase.units import Bohr
import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    SurfaceChargeState,
)
from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1,
    PROFILE_REGISTRY,
)
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.experimental.mace_mdp_polar_pcmsolver import (
    HybridPCMSolverEnergyState,
    MACE_MDPPolarHybridPCMSolverEnergy,
    MACE_MDPPolarHybridPCMSolverPES,
    ROOT_REPLAY_FIELD_ATOL_EV,
    ROOT_TOLERANCE_EV,
)
from maple.solvation.derivatives import RichardsonScalarForce
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.models.mace_mdp_polar_hybrid import (
    PermanentAnchoredInducedSourceModel,
)


class _Permanent:
    provider_id = "test.hybrid.permanent.v1"
    model_profile_id = "test.hybrid.permanent-profile.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE

    def configuration_sha256(self) -> str:
        return "1" * 64

    def evaluate_source(self, _geometry: object) -> np.ndarray:
        return np.asarray([[-0.15, 0.01, -0.02, 0.03], [0.15, -0.02, 0.01, -0.01]])

    def source_position_vjp(
        self, geometry: object, _source_cotangent: object
    ) -> np.ndarray:
        return np.zeros((len(geometry), 3), dtype=float)


class _Responsive:
    provider_id = "test.hybrid.response.v1"
    model_profile_id = "test.hybrid.response-profile.v1"
    provenance_sha256 = "2" * 64
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    long_range_evaluator_profile = (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )

    def __init__(self) -> None:
        rng = np.random.default_rng(41)
        self.jacobian = rng.normal(scale=2.0e-5, size=(8, 16))
        self.jacobian[4] = -self.jacobian[0]
        self.zero = np.asarray([[-0.08, 0.01, 0.02, -0.01], [0.08, -0.01, 0.01, 0.02]])

    def configuration_sha256(self) -> str:
        return "3" * 64

    def vacuum_energy_ev(self, _geometry: object) -> float:
        return -123.456

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        return np.zeros((len(geometry), 3), dtype=float)

    def evaluate_source(self, _geometry: object, field: object) -> np.ndarray:
        return self.zero + (self.jacobian @ np.asarray(field).reshape(-1)).reshape(2, 4)

    def field_jvp(
        self, _geometry: object, _field: object, direction: object
    ) -> np.ndarray:
        return (self.jacobian @ np.asarray(direction).reshape(-1)).reshape(2, 4)

    def field_vjp(
        self, _geometry: object, _field: object, cotangent: object
    ) -> np.ndarray:
        return (self.jacobian.T @ np.asarray(cotangent).reshape(-1)).reshape(2, 8)

    def coordinate_vjp(
        self, geometry: object, _field: object, _source_cotangent: object
    ) -> np.ndarray:
        return np.zeros((len(geometry), 3), dtype=float)


class _Response:
    provider_id = "test.pcmsolver.external-mep.v1"
    continuum_profile_id = "pcmsolver-symmetric-external-mep-electrostatic-v1"
    cavity_profile_id = "pcmsolver-input-defined-gepol-cavity-v1"
    energy_response_is_reciprocal = True

    def __init__(self, atoms: Atoms) -> None:
        self.atom_count = len(atoms)
        self.atomic_numbers = atoms.numbers.astype(float)
        self.reference_positions_bohr = atoms.positions / Bohr
        self.cavity_radii_angstrom = np.asarray([1.8, 1.7])
        self.surface_points_bohr = np.asarray(
            [
                [-4.0, 0.2, 0.1],
                [4.2, -0.1, 0.3],
                [0.3, 3.8, -0.2],
                [-0.2, -3.9, 0.4],
                [0.1, 0.3, 4.1],
                [0.4, -0.2, -4.0],
            ]
        )
        self.surface_areas_bohr2 = np.ones(6)
        self.matrix = -0.015 * np.eye(6)
        self.digest = "4" * 64

    def configuration_sha256(self) -> str:
        return self.digest

    def cavity_configuration_sha256(self) -> str:
        return "5" * 64

    def apply_energy_conjugate(self, potential: object) -> np.ndarray:
        return self.matrix @ np.asarray(potential, dtype=float)

    def solve(self, potential: object) -> SurfaceChargeState:
        values = np.asarray(potential, dtype=float)
        charge = self.apply_energy_conjugate(values)
        return SurfaceChargeState(
            surface_potential_hartree_per_e=values,
            direct_surface_charge_e=charge,
            adjoint_surface_charge_e=charge,
            energy_conjugate_surface_charge_e=charge,
            polarization_energy_hartree=0.5 * float(np.vdot(values, charge)),
        )


def _atoms() -> Atoms:
    atoms = Atoms("CO", positions=[[-0.5, 0.1, 0.0], [0.7, -0.1, 0.2]])
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    return atoms


def _evaluator():
    atoms = _atoms()
    hybrid = PermanentAnchoredInducedSourceModel(_Permanent(), _Responsive())
    return atoms, MACE_MDPPolarHybridPCMSolverEnergy(
        atoms, hybrid=hybrid, response=_Response(atoms)
    )


def test_energy_state_closes_two_start_root_charge_and_scalar():
    atoms, evaluator = _evaluator()
    state = evaluator.solve(atoms)

    assert state.primal_residual_ev < ROOT_TOLERANCE_EV
    assert state.replay_field_max_abs_difference_ev < ROOT_REPLAY_FIELD_ATOL_EV
    assert np.sum(state.induced_source4[:, 0]) == pytest.approx(0.0, abs=1.0e-12)
    assert np.sum(state.total_source4[:, 0]) == pytest.approx(0.0, abs=1.0e-12)
    assert state.total_energy_ev == pytest.approx(
        state.vacuum_energy_ev + state.polarization_energy_ev,
        abs=1.0e-14,
    )
    assert state.root_sha256
    with pytest.raises(ValueError):
        state.native_field_ev.setflags(write=True)


def test_geometry_bound_surface_fails_closed_before_admission_and_for_forces():
    atoms, evaluator = _evaluator()
    profile = PROFILE_REGISTRY[
        EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1
    ]
    assert profile.enabled is False
    with pytest.raises(RuntimeError, match="has not passed admission"):
        evaluator.evaluate(atoms)
    with pytest.raises(NotImplementedError, match="cannot move its PCMSolver cavity"):
        evaluator.evaluate(atoms, need_forces=True)
    with pytest.raises(NotImplementedError, match="cannot move its cavity"):
        evaluator.get_forces(atoms)


def test_energy_evaluator_rejects_geometry_and_configuration_drift():
    atoms, evaluator = _evaluator()
    moved = atoms.copy()
    moved.positions[0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="different geometry"):
        evaluator.solve(moved)
    evaluator._response.digest = "6" * 64
    with pytest.raises(RuntimeError, match="configuration drifted"):
        evaluator.solve(atoms)


def test_energy_evaluator_requires_analytic_long_range_identity():
    atoms = _atoms()
    response_model = _Responsive()
    response_model.long_range_evaluator_profile = "laboratory-grid"
    hybrid = PermanentAnchoredInducedSourceModel(_Permanent(), response_model)
    with pytest.raises(ValueError, match="analytic Gaussian"):
        MACE_MDPPolarHybridPCMSolverEnergy(
            atoms, hybrid=hybrid, response=_Response(atoms)
        )


def test_energy_force_admission_is_preregistered_without_accuracy_relabeling():
    root = Path(__file__).parents[2]
    path = (
        root / "docs/route2/preregistrations/"
        "mace-mdp-polar-hybrid-numerical-force-admission-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "preregistered-before-execution"
    assert payload["target_profile_id"] == (
        EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1
    )
    assert payload["target_capabilities"] == {
        "E": True,
        "F": True,
        "H": False,
        "V": False,
        "M": False,
    }
    assert payload["admission_gates"]["accuracy_threshold"] is None
    assert payload["frozen_runtime_contract"]["force_coarse_step_angstrom"] == (5.0e-4)
    assert payload["frozen_runtime_contract"]["force_fine_step_angstrom"] == (2.5e-4)
    assert payload["fit_calibration_or_case_selection"] is False


def test_geometry_resolved_pes_differentiates_one_scalar_and_checks_files(
    tmp_path, monkeypatch
):
    atoms = _atoms()
    hybrid = PermanentAnchoredInducedSourceModel(_Permanent(), _Responsive())
    parsed = tmp_path / "parsed.inp"
    library = tmp_path / "libpcm.so"
    parsed.write_text("frozen input\n", encoding="utf-8")
    library.write_bytes(b"frozen library")
    pes = MACE_MDPPolarHybridPCMSolverPES(
        hybrid=hybrid,
        atomic_numbers=atoms.numbers,
        cavity_radii_angstrom=np.asarray([1.8, 1.7]),
        parsed_input_path=parsed,
        pcmsolver_library_path=library,
        force_backend=RichardsonScalarForce(
            coarse_step_angstrom=1.0e-2,
            maximum_error_eV_per_A=1.0e-3,
        ),
    )

    @contextmanager
    def fake_open(_self, geometry):
        evaluator_configuration_sha256 = canonical_metadata_sha256(
            {
                "contract": "test-hybrid-pcmsolver-energy-evaluator-v1",
                "geometry_sha256": geometry_sha256(geometry),
                "hybrid_configuration_sha256": (_self._hybrid.configuration_sha256()),
                "cavity_radii_angstrom": (_self._cavity_radii_angstrom.tolist()),
                "parsed_input_sha256": _self._parsed_input_sha256,
                "library_sha256": _self._library_sha256,
            }
        )

        class Evaluator:
            @staticmethod
            def solve(current):
                positions = np.asarray(current.positions, dtype=float)
                weights = np.arange(1, positions.size + 1, dtype=float).reshape(
                    positions.shape
                )
                energy = float(np.sum(weights * (positions**2 + 0.2 * positions**6)))
                return HybridPCMSolverEnergyState(
                    geometry_sha256=geometry_sha256(current),
                    evaluator_configuration_sha256=(evaluator_configuration_sha256),
                    anchor_state_sha256="8" * 64,
                    native_field_ev=np.zeros((len(current), 8)),
                    induced_source4=np.zeros((len(current), 4)),
                    total_source4=np.zeros((len(current), 4)),
                    surface_potential_hartree_per_e=np.ones(6),
                    surface_charge_e=np.ones(6),
                    vacuum_energy_ev=energy,
                    polarization_energy_ev=0.0,
                    primal_residual_ev=0.0,
                    cold_iterations=1,
                    wide_iterations=1,
                    replay_field_max_abs_difference_ev=0.0,
                    replay_energy_abs_difference_ev=0.0,
                    cavity_topology_id="test-fixed-topology-v1",
                )

        yield Evaluator()

    monkeypatch.setattr(MACE_MDPPolarHybridPCMSolverPES, "_open_evaluator", fake_open)
    force = pes.numerical_force(atoms)
    positions = np.asarray(atoms.positions)
    weights = np.arange(1, positions.size + 1, dtype=float).reshape(positions.shape)
    expected = -weights * (2.0 * positions + 1.2 * positions**5)
    assert force.forces_eV_per_A == pytest.approx(expected, abs=2.0e-8)
    assert force.maximum_error_estimate_eV_per_A < 1.0e-3
    assert pes.configuration_sha256()

    other_pes = MACE_MDPPolarHybridPCMSolverPES(
        hybrid=hybrid,
        atomic_numbers=atoms.numbers,
        cavity_radii_angstrom=np.asarray([1.9, 1.7]),
        parsed_input_path=parsed,
        pcmsolver_library_path=library,
        force_backend=pes.force_backend,
    )
    with pytest.raises(ValueError, match="central_sample did not replay"):
        other_pes.numerical_force_component(
            atoms,
            atom_index=0,
            axis_index=0,
            central_state=pes.solve(atoms),
        )

    stale = replace(
        pes.solve(atoms),
        evaluator_configuration_sha256="6" * 64,
        root_sha256="",
    )
    with pytest.raises(ValueError, match="central_sample did not replay"):
        pes.numerical_force_component(
            atoms, atom_index=0, axis_index=0, central_state=stale
        )

    moved = atoms.copy()
    moved.positions[0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="different geometry"):
        pes.numerical_force(moved, central_state=pes.solve(atoms))

    parsed.write_text("drifted input\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="configuration drifted"):
        pes.configuration_sha256()
