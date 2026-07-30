from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
    FixedCavityGTOGalerkinOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_qeq_monopole import (
    RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION,
    embed_mace_polar_l1_density_in_single_radial_gto,
    evaluate_route2_v0_rappe_goddard_monopole,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_variational_admission import (
    FrozenNoTrainingFunctionalProvenance,
    GasPhasePolarizabilityScreen,
    admit_route2_v0_electronic_component,
    require_route2_v0_implicit_electronic_admission,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_variational_quadratic import (
    Route2V0VariationalQuadraticState,
    solve_route2_v0_variational_quadratic,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_variational_quadratic import (
    embed_mace_polar_l1_density_in_single_radial_gto as embed_quadratic_density,
)

ROOT = Path(__file__).resolve().parents[2]
QEQ_ACETONE_ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/route2-v0-qeq-acetone-qm-field-v1.json"
)


class _ReciprocalSurfaceResponse:
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        surface_points_bohr: np.ndarray,
        response_matrix: np.ndarray,
    ) -> None:
        self.atom_count = len(positions_angstrom)
        self._positions = np.asarray(positions_angstrom, dtype=float)
        self._points = np.asarray(surface_points_bohr, dtype=float)
        self._response = np.asarray(response_matrix, dtype=float)

    @property
    def atomic_numbers(self):
        return np.ones(self.atom_count)

    @property
    def reference_positions_bohr(self):
        return self._positions / Bohr

    @property
    def cavity_radii_angstrom(self):
        return np.full(self.atom_count, 1.5)

    @property
    def surface_points_bohr(self):
        return self._points.copy()

    @property
    def surface_areas_bohr2(self):
        return np.ones(len(self._points))

    def apply_energy_conjugate(self, potential):
        return self._response @ np.asarray(potential, dtype=float)


def _operator() -> FixedCavityGTOGalerkinOperator:
    positions = np.asarray([[-0.7, 0.1, 0.0], [0.8, -0.2, 0.3]], dtype=float)
    rng = np.random.default_rng(20260730)
    points = rng.normal(size=(18, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points *= rng.uniform(4.0, 6.0, size=(18, 1))
    factor = rng.normal(size=(18, 18))
    response = -(factor.T @ factor) / 160.0
    return FixedCavityGTOGalerkinOperator(
        _ReciprocalSurfaceResponse(positions, points, response),
        positions,
        AtomCenteredL1GTOBasis((MACE_POLAR_DENSITY_SIGMA_ANGSTROM,)),
    )


def _stable_state() -> Route2V0VariationalQuadraticState:
    operator = _operator()
    native_density = np.asarray(
        [[0.25, 0.12, -0.03, 0.07], [-0.25, -0.06, 0.08, -0.04]],
        dtype=float,
    )
    frozen = embed_quadratic_density(native_density, operator.basis)
    return solve_route2_v0_variational_quadratic(
        frozen_density_coefficients=frozen,
        operator=operator,
        electronic_curvature_coefficient_dual=np.eye(operator.coefficient_count) * 10.0,
    )


def _physical_provenance(candidate_id: str) -> FrozenNoTrainingFunctionalProvenance:
    return FrozenNoTrainingFunctionalProvenance(
        candidate_id=candidate_id,
        source_record="independent frozen physical-response record",
        coefficient_dual_pairing="same-gto-coefficient-dual-v1",
        source_kind="independent-physical-theory",
        scalar_energy_functional_declared=True,
    )


def _passing_screen(candidate_id: str) -> GasPhasePolarizabilityScreen:
    return GasPhasePolarizabilityScreen(
        candidate_id=candidate_id,
        candidate_polarizability_bohr3=np.diag([10.0, 10.5, 9.5]),
        qm_reference_polarizability_bohr3=np.diag([10.0, 10.0, 10.0]),
        candidate_geometry_id="frozen-test-geometry",
        qm_reference_geometry_id="frozen-test-geometry",
        candidate_charge_e=0,
        qm_reference_charge_e=0,
        candidate_spin_multiplicity=1,
        qm_reference_spin_multiplicity=1,
        qm_protocol_id="frozen-qm-field-protocol",
        preregistered_before_execution=True,
        qm_numerical_gates_passed=True,
    )


def test_synthetic_structural_control_cannot_be_promoted_by_a_toy_response():
    state = _stable_state()
    provenance = FrozenNoTrainingFunctionalProvenance(
        candidate_id="synthetic-v0-kkt-control",
        source_record="test-only exact quadratic",
        coefficient_dual_pairing="same-gto-coefficient-dual-v1",
        source_kind="synthetic-structural-control",
        scalar_energy_functional_declared=True,
    )

    admission = admit_route2_v0_electronic_component(
        state=state,
        provenance=provenance,
        reference_gradient_coefficient_dual_hartree=np.zeros(
            state.electronic_curvature_coefficient_dual.shape[0]
        ),
        gas_response_screen=_passing_screen(provenance.candidate_id),
    )

    assert admission.structural_checks_passed is True
    assert admission.status == "structural-only"
    assert admission.gas_response_decision is None
    assert admission.eligible_for_implicit_electrostatic_composition is False
    assert admission.eligible_for_total_solvation_scoring is False
    with pytest.raises(RuntimeError, match="not admitted"):
        require_route2_v0_implicit_electronic_admission(admission)


def test_independently_screened_electronic_component_is_not_a_chemistry_score():
    state = _stable_state()
    provenance = _physical_provenance("frozen-physical-component")

    admission = admit_route2_v0_electronic_component(
        state=state,
        provenance=provenance,
        reference_gradient_coefficient_dual_hartree=np.zeros(
            state.electronic_curvature_coefficient_dual.shape[0]
        ),
        gas_response_screen=_passing_screen(provenance.candidate_id),
    )

    assert admission.status == "electronic-component-admitted"
    assert admission.gas_response_decision is not None
    assert admission.gas_response_decision.passes is True
    assert admission.eligible_for_implicit_electrostatic_composition is True
    assert admission.eligible_for_total_solvation_scoring is False
    require_route2_v0_implicit_electronic_admission(admission)


def test_nonstationary_gas_reference_is_rejected_before_a_qm_screen_can_help():
    state = _stable_state()
    provenance = _physical_provenance("nonstationary-physical-component")
    gradient = np.zeros(state.electronic_curvature_coefficient_dual.shape[0])
    allowed_index = np.flatnonzero(np.abs(state.charge_constraint_vector) < 0.5)[0]
    gradient[allowed_index] = 1.0

    admission = admit_route2_v0_electronic_component(
        state=state,
        provenance=provenance,
        reference_gradient_coefficient_dual_hartree=gradient,
        gas_response_screen=_passing_screen(provenance.candidate_id),
    )

    assert admission.status == "rejected-structural"
    assert admission.gas_response_decision is None
    assert any("not stationary" in reason for reason in admission.reasons)


def test_frozen_qeq_artifact_rejects_qeq_from_implicit_pcm_composition():
    artifact = json.loads(QEQ_ACETONE_ARTIFACT.read_text(encoding="utf-8"))
    operator = _operator()
    native_density = np.asarray(
        [[0.25, 0.12, -0.03, 0.07], [-0.25, -0.06, 0.08, -0.04]],
        dtype=float,
    )
    qeq_state = evaluate_route2_v0_rappe_goddard_monopole(
        symbols=("H", "C"),
        frozen_density_coefficients=embed_mace_polar_l1_density_in_single_radial_gto(
            native_density,
            operator.basis,
        ),
        operator=operator,
    ).response_state
    provenance = _physical_provenance(RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION)
    qeq = artifact["qeq_control"]
    qm = artifact["qm_reference"]
    screen = GasPhasePolarizabilityScreen(
        candidate_id=provenance.candidate_id,
        candidate_polarizability_bohr3=np.asarray(qeq["polarizability_bohr3"]),
        qm_reference_polarizability_bohr3=np.asarray(
            qm["selected_symmetric_polarizability_bohr3"]
        ),
        candidate_geometry_id="acetone-frozen-geometry",
        qm_reference_geometry_id="acetone-frozen-geometry",
        candidate_charge_e=0,
        qm_reference_charge_e=0,
        candidate_spin_multiplicity=1,
        qm_reference_spin_multiplicity=1,
        qm_protocol_id="route2-v0-qeq-acetone-qm-field-prereg-v1",
        preregistered_before_execution=True,
        qm_numerical_gates_passed=True,
    )

    admission = admit_route2_v0_electronic_component(
        state=qeq_state,
        provenance=provenance,
        reference_gradient_coefficient_dual_hartree=np.zeros(
            qeq_state.electronic_curvature_coefficient_dual.shape[0]
        ),
        gas_response_screen=screen,
    )

    assert admission.structural_checks_passed is True
    assert admission.status == "rejected-gas-response"
    assert admission.gas_response_decision is not None
    assert admission.gas_response_decision.passes is False
    assert admission.gas_response_decision.relative_frobenius_mismatch == pytest.approx(
        artifact["scientific_falsification"]["checks"]["relative_frobenius_mismatch"][
            "value"
        ],
        abs=1.0e-12,
    )
    assert admission.eligible_for_implicit_electrostatic_composition is False
    with pytest.raises(RuntimeError, match="Frobenius mismatch"):
        require_route2_v0_implicit_electronic_admission(admission)


def test_no_training_provenance_rejects_a_fit_or_fine_tuning_at_construction():
    with pytest.raises(ValueError, match="forbids training"):
        FrozenNoTrainingFunctionalProvenance(
            candidate_id="not-v0",
            source_record="would-be fit",
            coefficient_dual_pairing="same-gto-coefficient-dual-v1",
            source_kind="frozen-external-model",
            scalar_energy_functional_declared=True,
            fine_tuning=True,
        )


def test_quadratic_state_rejects_a_negative_additional_constraint_residual():
    with pytest.raises(ValueError, match="additional_constraint_residual_inf"):
        replace(_stable_state(), additional_constraint_residual_inf=-1.0)


def test_mainline_theory_keeps_route2_v0_an_implicit_continuum_model():
    theory = (
        ROOT / "docs/implicit-solvation/ROUTE2_V0_VARIATIONAL_QUADRATIC_THEORY.md"
    ).read_text(encoding="utf-8")
    decision_record = (
        ROOT / "docs/implicit-solvation/ROUTE2_V0_NO_FIT_ACCURACY_RESEARCH_20260730.md"
    ).read_text(encoding="utf-8")

    assert "implicit-continuum" in theory
    assert "contains no liquid trajectory" in theory
    assert "GROMACS run" in theory
    assert "Molecular RISM/MDFT code remains an archived alternative" in " ".join(
        theory.split()
    )
    assert "excluded from implicit V0 main path" in decision_record


def test_xtb_cosmo_source_screen_stays_outside_the_no_fit_core():
    artifact_path = (
        ROOT / "docs/implicit-solvation/benchmarks/"
        "route2-v0-xtb-cosmo-source-screen-v1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["status"] == "excluded-from-no-fit-core-before-accuracy"
    assert "not a solvation accuracy" in artifact["claim_boundary"]
    observed = artifact["observed_command_semantics"]
    assert observed["gbsa_enabled_line"].endswith("true          :")
    assert observed["reported_gsasa_hartree"] > 0.0
    assert "internal GFN-xTB/COSMO" in observed["parameter_file_line"]
    assert artifact["decision"]["verdict"] == (
        "exclude-xtb-cosmo-and-alpb-cpcmx-from-route2-v0-no-fit-core"
    )
    assert "1.5569E+01 dyn/cm" in observed["surface_tension_line"]
