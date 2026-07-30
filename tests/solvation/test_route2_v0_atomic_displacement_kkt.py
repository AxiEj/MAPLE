from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
)
from maple.function.calculator.extra_correction.implicit import (
    route2_v0_atomic_displacement_kkt as atomic_displacement_kkt,
    route2_v0_atomic_displacement_response as atomic_displacement_response,
)


ROOT = Path(__file__).resolve().parents[2]


class _LinearReciprocalContinuum:
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        surface_points_bohr: np.ndarray,
        response_matrix: np.ndarray,
    ) -> None:
        self.atom_count = len(positions_angstrom)
        self._atomic_numbers = np.ones(self.atom_count, dtype=int)
        self._positions_bohr = (
            np.asarray(positions_angstrom, dtype=float)
            / atomic_displacement_response.BOHR_ANGSTROM
        )
        self._surface_points_bohr = np.asarray(surface_points_bohr, dtype=float)
        self._response_matrix = np.asarray(response_matrix, dtype=float)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers.copy()

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        return self._positions_bohr.copy()

    @property
    def surface_points_bohr(self) -> np.ndarray:
        return self._surface_points_bohr.copy()

    def apply_energy_conjugate(self, potential: np.ndarray) -> np.ndarray:
        return self._response_matrix @ np.asarray(potential, dtype=float)


def _table() -> atomic_displacement_response.Route2V0AtomicDisplacementResponseTable:
    return atomic_displacement_response.Route2V0AtomicDisplacementResponseTable(
        radial_grid_bohr=np.asarray([0.0, 0.5, 2.0, 8.0]),
        enclosed_electrons_by_atomic_number={
            1: np.asarray([0.0, 0.3, 0.9, 1.0]),
        },
        table_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.9, -0.2, 0.4]])
    numbers = np.asarray([1, 1])
    weights = np.stack((0.4 * np.eye(3), 0.6 * np.eye(3)))
    points = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [-2.5, 1.5, 0.7],
            [1.1, -2.7, 1.8],
            [-1.2, -0.8, -2.5],
        ]
    )
    return positions, numbers, weights, points


def _coupling() -> atomic_displacement_kkt.AtomicDisplacementSurfaceCoupling:
    positions, numbers, weights, points = _geometry()
    return atomic_displacement_kkt.AtomicDisplacementSurfaceCoupling(
        response_table=_table(),
        atomic_numbers=numbers,
        atom_positions_angstrom=positions,
        atomic_dipole_weights=weights,
        surface_points_bohr=points,
    )


def test_atomic_displacement_coupling_uses_an_exact_matrix_transpose():
    coupling = _coupling()
    dipole = np.asarray([0.8, -0.4, 0.3])
    surface_charge = np.asarray([0.1, -0.2, 0.4, -0.1, 0.3, -0.2])

    left = float(coupling.surface_potential(dipole) @ surface_charge)
    right = float(dipole @ coupling.surface_to_dipole_dual(surface_charge))

    assert coupling.atomic_partition_identity_error < 1.0e-12
    assert coupling.atomic_numbers.flags.writeable is False
    assert abs(left - right) < 1.0e-14


def test_reduced_kkt_preregistration_locks_its_induced_only_boundary():
    protocol = json.loads(
        (
            ROOT
            / "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-displacement-reduced-kkt-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )
    theory = (
        ROOT
        / "docs/implicit-solvation/ROUTE2_V0_ATOMIC_DISPLACEMENT_REDUCED_KKT_THEORY.md"
    ).read_text(encoding="utf-8")

    assert protocol["status"] == (
        "structural-kernel-implemented-before-physical-continuum-binding"
    )
    assert protocol["construction"]["name"] == (
        "route2-v0-atomic-displacement-reduced-kkt-v1"
    )
    assert protocol["hard_constraints"]["post_training"] is False
    assert protocol["hard_constraints"]["fine_tuning"] is False
    assert protocol["hard_constraints"]["experimental_solvation_fit"] is False
    forbidden = " ".join(protocol["forbidden_shortcuts"])
    assert "FreeSolv" in forbidden
    assert "MNSol" in forbidden
    assert "permanent-density functional" in forbidden
    assert "induced-only reduced KKT" in theory
    assert "synthetic reciprocal continuum" in theory
    assert "not yet the Route-2V electronic functional" in theory


def test_continuum_admission_audit_keeps_v0_adt_below_physical_execution():
    audit = json.loads(
        (
            ROOT
            / "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-displacement-continuum-admission-audit-v1.json"
        ).read_text(encoding="utf-8")
    )

    assert audit["status"] == "rejected-before-v0-adt-physical-continuum-execution"
    assert audit["admitted_physical_continuum_bindings"] == []
    assert set(audit["audited_source_files_sha256"]) == {
        "docs/implicit-solvation/ROUTE2_CONTINUUM_ENGINE.md",
        "docs/implicit-solvation/ROUTE2_V0_AQ_THEORY.md",
        "maple/function/calculator/extra_correction/implicit/continuum_response.py",
        "maple/function/calculator/extra_correction/implicit/pyscf_swig_response.py",
        "maple/function/calculator/extra_correction/implicit/route2_v0_diffuse_continuum.py",
        "maple/function/calculator/extra_correction/implicit/smd.py",
    }
    assert audit["decision"]["physical_continuum_execution_performed"] is False
    assert audit["decision"]["qm_physics_panel_execution_performed"] is False
    assert audit["decision"]["experimental_solvation_labels_read"] is False
    assert audit["hard_constraints"]["experimental_solvation_fit"] is False
    assert (
        audit["hard_constraints"]["radius_or_cavity_selection_from_solvation_error"]
        is False
    )
    assert {
        candidate["name"] for candidate in audit["audited_candidates"]
    } == {
        "PCMSolverExternalMEPCavityResponse",
        "PySCFSWIGPCMResponse",
        "Route2V0DiffuseLocalDielectricOperator",
    }
    assert not any(
        candidate["admitted_for_v0_adt_physical_binding"]
        for candidate in audit["audited_candidates"]
    )


def test_reduced_kkt_has_one_energy_ledger_and_passive_reciprocal_response():
    coupling = _coupling()
    positions, _, _, points = _geometry()
    continuum = _LinearReciprocalContinuum(
        positions,
        points,
        response_matrix=-0.03 * np.eye(len(points)),
    )
    external = np.asarray([0.02, -0.01, 0.03])

    state = atomic_displacement_kkt.solve_route2_v0_atomic_displacement_reduced_kkt(
        coupling=coupling,
        continuum=continuum,
        polarizability_bohr3=2.0 * np.eye(3),
        external_dipole_dual_hartree_per_ebohr=external,
    )

    assert state.stationarity_residual_inf < 1.0e-12
    assert state.source_duality_error_hartree < 1.0e-14
    assert state.joint_minimum_curvature > state.joint_stability_threshold
    np.testing.assert_allclose(
        state.external_dipole_response_bohr3,
        state.external_dipole_response_bohr3.T,
        atol=1.0e-12,
    )
    assert np.max(np.linalg.eigvalsh(state.external_dipole_response_bohr3)) <= 1.0e-12

    step = 1.0e-6
    plus = atomic_displacement_kkt.solve_route2_v0_atomic_displacement_reduced_kkt(
        coupling=coupling,
        continuum=continuum,
        polarizability_bohr3=2.0 * np.eye(3),
        external_dipole_dual_hartree_per_ebohr=external + step * np.eye(3)[0],
    )
    minus = atomic_displacement_kkt.solve_route2_v0_atomic_displacement_reduced_kkt(
        coupling=coupling,
        continuum=continuum,
        polarizability_bohr3=2.0 * np.eye(3),
        external_dipole_dual_hartree_per_ebohr=external - step * np.eye(3)[0],
    )
    np.testing.assert_allclose(
        (plus.induced_dipole_ebohr - minus.induced_dipole_ebohr) / (2.0 * step),
        state.external_dipole_response_bohr3[:, 0],
        atol=1.0e-9,
    )
    assert (
        plus.stationary_total_energy_hartree
        - minus.stationary_total_energy_hartree
    ) / (2.0 * step) == pytest.approx(state.induced_dipole_ebohr[0], abs=1.0e-9)


def test_reduced_kkt_rejects_a_nonreciprocal_continuum_in_the_source_subspace():
    coupling = _coupling()
    positions, _, _, points = _geometry()
    response = -0.03 * np.eye(len(points))
    response[0, 1] = 0.2
    continuum = _LinearReciprocalContinuum(positions, points, response)

    with pytest.raises(RuntimeError, match="reciprocal"):
        atomic_displacement_kkt.solve_route2_v0_atomic_displacement_reduced_kkt(
            coupling=coupling,
            continuum=continuum,
            polarizability_bohr3=2.0 * np.eye(3),
        )


def test_reduced_kkt_rejects_a_cavity_built_for_other_elements():
    coupling = _coupling()
    positions, _, _, points = _geometry()
    continuum = _LinearReciprocalContinuum(
        positions,
        points,
        response_matrix=-0.03 * np.eye(len(points)),
    )
    continuum._atomic_numbers[:] = 6

    with pytest.raises(ValueError, match="atomic numbers"):
        atomic_displacement_kkt.solve_route2_v0_atomic_displacement_reduced_kkt(
            coupling=coupling,
            continuum=continuum,
            polarizability_bohr3=2.0 * np.eye(3),
        )


def test_reduced_kkt_state_rejects_an_inconsistent_source_or_stationarity_ledger():
    coupling = _coupling()
    positions, _, _, points = _geometry()
    state = atomic_displacement_kkt.solve_route2_v0_atomic_displacement_reduced_kkt(
        coupling=coupling,
        continuum=_LinearReciprocalContinuum(
            positions,
            points,
            response_matrix=-0.03 * np.eye(len(points)),
        ),
        polarizability_bohr3=2.0 * np.eye(3),
        external_dipole_dual_hartree_per_ebohr=np.asarray([0.02, -0.01, 0.03]),
    )

    with pytest.raises(ValueError, match="Source duality error"):
        replace(state, source_duality_error_hartree=1.0)
    with pytest.raises(ValueError, match="Stationarity residual"):
        replace(state, stationarity_residual_inf=1.0)


def test_reduced_kkt_rejects_an_unstable_joint_curvature_without_repairing_it():
    coupling = _coupling()
    positions, _, _, points = _geometry()
    continuum = _LinearReciprocalContinuum(
        positions,
        points,
        response_matrix=-10.0 * np.eye(len(points)),
    )

    with pytest.raises(RuntimeError, match="curvature is not positive definite"):
        atomic_displacement_kkt.solve_route2_v0_atomic_displacement_reduced_kkt(
            coupling=coupling,
            continuum=continuum,
            polarizability_bohr3=2.0 * np.eye(3),
        )
