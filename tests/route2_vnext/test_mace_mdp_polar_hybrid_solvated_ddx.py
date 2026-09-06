from __future__ import annotations

from dataclasses import replace
import hashlib

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives import (
    FiniteDifferenceTopologyObservationError,
    OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
    RichardsonScalarForce,
    RichardsonScalarHessian,
)
from maple.solvation.experimental.mace_mdp_polar_ddx import (
    HybridDDXEnergyState,
    HybridDDXForceEvaluation,
)
from maple.solvation.experimental.mace_mdp_polar_solvated_ddx import (
    HYBRID_SOLVATED_DDX_PES_PROVIDER_ID,
    HYBRID_SOLVATED_DDX_SCALAR_CONTRACT_ID,
    HybridSolvatedDDXEnergyState,
    HybridSolvatedDDXForceEvaluation,
    MACE_MDPPolarHybridSolvatedDDXPES,
)
from maple.solvation.solvent_terms import SolventEnergyState


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _atoms() -> Atoms:
    return Atoms(
        "HC",
        positions=np.asarray([[0.27, -0.18, 0.13], [-0.41, 0.32, -0.22]], dtype=float),
        info={"charge": 0, "mult": 1},
    )


class _QuadraticHybridDDXPES:
    provider_id = "test.hybrid-ddx-electrostatic-pes.v1"
    coordinate_derivative_available = True
    _weights = np.asarray([[0.7, 0.9, 1.1], [1.3, 1.5, 1.7]], dtype=float)

    def __init__(self) -> None:
        self._configuration = _digest("quadratic-hybrid-ddx")

    def configuration_sha256(self) -> str:
        return self._configuration

    def topology_id(self, geometry: object) -> str:
        del geometry
        return _digest("fixed-ddx-topology")

    def solve(self, geometry: object) -> HybridDDXEnergyState:
        positions = np.asarray(geometry.get_positions(), dtype=float)
        geometry_digest = geometry_sha256(geometry)
        return HybridDDXEnergyState(
            geometry_sha256=geometry_digest,
            evaluator_configuration_sha256=self._configuration,
            anchor_state_sha256=canonical_metadata_sha256({"anchor": geometry_digest}),
            continuum_state_sha256=canonical_metadata_sha256(
                {"continuum": geometry_digest}
            ),
            native_field_ev=np.zeros((len(positions), 8)),
            induced_source4=np.zeros((len(positions), 4)),
            total_source4=np.zeros((len(positions), 4)),
            vacuum_energy_ev=float(0.5 * np.sum(self._weights * positions**2)),
            polarization_energy_ev=float(0.25 * np.sum(positions**2)),
            primal_residual_ev=0.0,
            cold_iterations=1,
            wide_iterations=1,
            replay_field_max_abs_difference_ev=0.0,
            replay_energy_abs_difference_ev=0.0,
        )

    def evaluate_forces(
        self,
        geometry: object,
        *,
        central_state: HybridDDXEnergyState | None = None,
    ) -> HybridDDXForceEvaluation:
        replay = self.solve(geometry)
        if (
            central_state is not None
            and central_state.root_sha256 != replay.root_sha256
        ):
            raise ValueError("central electrostatic state did not replay")
        positions = np.asarray(geometry.get_positions(), dtype=float)
        zero = np.zeros_like(positions)
        return HybridDDXForceEvaluation(
            central_state=replay,
            vacuum_forces_ev_per_angstrom=-self._weights * positions,
            continuum_fixed_source_forces_ev_per_angstrom=-0.5 * positions,
            permanent_source_forces_ev_per_angstrom=zero,
            induced_source_forces_ev_per_angstrom=zero,
            model_field_geometry_forces_ev_per_angstrom=zero,
            adjoint_field_ev=np.zeros((len(positions), 8)),
            adjoint_residual_ev=0.0,
            adjoint_iterations=0,
        )


class _EnergyOnlyQuadraticHybridDDXPES(_QuadraticHybridDDXPES):
    provider_id = "test.hybrid-ddx-energy-only-pes.v1"
    coordinate_derivative_available = False


class _QuadraticSolventTerm:
    provider_id = "test.quadratic-solvent-term.v1"

    def __init__(self, *, topology_coverage: str = "complete") -> None:
        self._topology_coverage = topology_coverage
        self._configuration = canonical_metadata_sha256(
            {
                "provider_id": self.provider_id,
                "topology_coverage": topology_coverage,
                "spring_eV_per_A2": 0.4,
            }
        )

    def configuration_sha256(self) -> str:
        return self._configuration

    def evaluate(self, geometry: object, *, need_gradient: bool) -> SolventEnergyState:
        positions = np.asarray(geometry.get_positions(), dtype=float)
        components = (
            ()
            if self._topology_coverage == "complete"
            else ("test-unobservable-solvent-surface",)
        )
        return SolventEnergyState(
            provider_id=self.provider_id,
            configuration_sha256=self._configuration,
            geometry_sha256=geometry_sha256(geometry),
            topology_id=_digest("fixed-solvent-topology"),
            atom_count=len(positions),
            energy_eV=float(0.2 * np.sum(positions**2)),
            gradient_eV_per_A=(0.4 * positions if need_gradient else None),
            topology_observation_coverage=self._topology_coverage,
            unobservable_topology_components=components,
        )


def _pes(
    *,
    topology_coverage: str = "complete",
    hessian_backend: RichardsonScalarHessian | None = None,
) -> MACE_MDPPolarHybridSolvatedDDXPES:
    return MACE_MDPPolarHybridSolvatedDDXPES(
        electrostatic_pes=_QuadraticHybridDDXPES(),
        solvent_term=_QuadraticSolventTerm(topology_coverage=topology_coverage),
        numerical_force_backend=RichardsonScalarForce(
            coarse_step_angstrom=2.0e-3,
            maximum_error_eV_per_A=1.0e-9,
        ),
        hessian_backend=(
            hessian_backend
            or RichardsonScalarHessian(
                coarse_step_angstrom=2.0e-3,
                maximum_error_eV_per_A2=1.0e-8,
            )
        ),
    )


def test_hybrid_solvated_derivative_capability_propagates_fail_closed() -> None:
    atoms = _atoms()
    pes = MACE_MDPPolarHybridSolvatedDDXPES(
        electrostatic_pes=_EnergyOnlyQuadraticHybridDDXPES(),
        solvent_term=_QuadraticSolventTerm(),
    )

    assert pes.coordinate_derivative_available is False
    assert pes.force_available is False
    assert np.isfinite(pes.get_potential_energy(atoms))
    with pytest.raises(NotImplementedError, match="complete coordinate derivative"):
        pes.evaluate_forces(atoms)


def test_hybrid_solvated_state_closes_the_full_energy_ledger() -> None:
    atoms = _atoms()
    pes = _pes()
    state = pes.solve(atoms)

    assert isinstance(state, HybridSolvatedDDXEnergyState)
    assert state.provider_id == HYBRID_SOLVATED_DDX_PES_PROVIDER_ID
    assert state.scalar_contract_id == HYBRID_SOLVATED_DDX_SCALAR_CONTRACT_ID
    assert state.vacuum_energy_eV == state.electrostatic_state.vacuum_energy_ev
    assert state.polarization_energy_eV == (
        state.electrostatic_state.polarization_energy_ev
    )
    assert state.cds_energy_eV == state.solvent_state.energy_eV
    assert state.solvation_energy_eV == pytest.approx(
        state.polarization_energy_eV + state.cds_energy_eV, abs=0.0
    )
    assert state.total_energy_eV == pytest.approx(
        state.vacuum_energy_eV + state.solvation_energy_eV, abs=0.0
    )
    assert pes.get_potential_energy(atoms) == pytest.approx(
        state.total_energy_eV, abs=0.0
    )
    assert pes.get_solvation_energy(atoms) == pytest.approx(
        state.solvation_energy_eV, abs=0.0
    )
    assert state.topology_observation_coverage == "complete"
    assert state.unobservable_topology_components == ()
    with pytest.raises(AttributeError):
        pes._configuration_sha256 = _digest("tampered")


def test_hybrid_solvated_analytic_force_matches_the_same_scalar() -> None:
    atoms = _atoms()
    pes = _pes()
    state = pes.solve(atoms)
    evaluated = pes.evaluate_forces(atoms, central_state=state)

    assert isinstance(evaluated, HybridSolvatedDDXForceEvaluation)
    positions = atoms.get_positions()
    expected = -(_QuadraticHybridDDXPES._weights + 0.5 + 0.4) * positions
    np.testing.assert_allclose(
        evaluated.total_forces_eV_per_A, expected, atol=1.0e-15, rtol=0.0
    )
    np.testing.assert_allclose(
        evaluated.solvent_forces_eV_per_A, -0.4 * positions, atol=0.0, rtol=0.0
    )
    audit = pes.numerical_force_audit(atoms, central_state=state)
    np.testing.assert_allclose(
        audit.forces_eV_per_A, evaluated.total_forces_eV_per_A, atol=2.0e-12, rtol=0.0
    )
    assert audit.maximum_error_estimate_eV_per_A < 1.0e-10


def test_hybrid_solvated_virial_and_force_cache_are_same_scalar_bound() -> None:
    atoms = _atoms()
    pes = _pes()
    force = pes.evaluate_forces(atoms)
    virial = pes.molecular_virial(
        atoms,
        origin_angstrom=np.zeros(3),
        force_evaluation=force,
    )
    deformation = np.asarray(
        [[0.12, -0.07, 0.03], [0.05, -0.08, 0.02], [-0.04, 0.06, 0.09]]
    )
    step = 1.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions = atoms.positions @ (np.eye(3) + step * deformation).T
    minus.positions = atoms.positions @ (np.eye(3) - step * deformation).T
    finite = (pes.get_potential_energy(plus) - pes.get_potential_energy(minus)) / (
        2.0 * step
    )
    contraction = -float(np.vdot(virial.raw_virial_eV, deformation))
    assert contraction == pytest.approx(finite, abs=3.0e-10)

    tampered_electrostatic = replace(
        force.electrostatic_evaluation,
        vacuum_forces_ev_per_angstrom=(
            force.electrostatic_evaluation.vacuum_forces_ev_per_angstrom + 0.1
        ),
        evaluation_sha256="",
    )
    tampered = replace(
        force,
        electrostatic_evaluation=tampered_electrostatic,
        evaluation_sha256="",
    )
    with pytest.raises(ValueError, match="did not replay"):
        pes.molecular_virial(atoms, force_evaluation=tampered)


def test_hybrid_solvated_hvp_and_hessian_follow_total_analytic_force() -> None:
    atoms = _atoms()
    pes = _pes()
    direction = np.asarray([[0.31, -0.27, 0.19], [-0.23, 0.29, -0.17]], dtype=float)
    direction /= np.linalg.norm(direction)
    force = pes.evaluate_forces(atoms)
    hvp = pes.hessian_vector_product(atoms, direction, central_force=force)
    hessian = pes.evaluate_hessian(atoms, central_force=force)
    expected = np.diag((_QuadraticHybridDDXPES._weights + 0.5 + 0.4).reshape(-1))
    np.testing.assert_allclose(
        hvp.hvp_eV_per_A2.reshape(-1),
        expected @ direction.reshape(-1),
        atol=2.0e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        hessian.hessian_eV_per_A2, expected, atol=3.0e-10, rtol=0.0
    )
    assert hvp.topology_guard_status == "complete"
    assert hessian.topology_guard_status == "complete"


def test_unobservable_cds_topology_requires_explicit_experimental_hessian() -> None:
    atoms = _atoms()
    strict = _pes(topology_coverage="unobservable")
    state = strict.solve(atoms)
    assert state.topology_observation_coverage == "partial"
    assert state.unobservable_topology_components == (
        "test-unobservable-solvent-surface",
    )
    with pytest.raises(FiniteDifferenceTopologyObservationError):
        strict.hessian_vector_product(atoms, np.ones((2, 3)))

    experimental = _pes(
        topology_coverage="unobservable",
        hessian_backend=RichardsonScalarHessian(
            coarse_step_angstrom=2.0e-3,
            maximum_error_eV_per_A2=1.0e-8,
            topology_guard_policy=OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
        ),
    )
    result = experimental.hessian_vector_product(atoms, np.ones((2, 3)))
    assert result.topology_guard_status == "partial-experimental"
    assert result.unobservable_topology_components == (
        "test-unobservable-solvent-surface",
    )


def test_stale_composite_state_is_rejected_before_force_use() -> None:
    atoms = _atoms()
    pes = _pes()
    stale_geometry = atoms.copy()
    stale_geometry.positions[0, 0] += 0.01
    stale = pes.solve(stale_geometry)
    with pytest.raises(ValueError, match="did not replay"):
        pes.evaluate_forces(atoms, central_state=stale)
