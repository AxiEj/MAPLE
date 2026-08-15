from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from maple.solvation.release.electrostatic_decomposition import (
    LEDGER_SELECTION_RULE,
    LedgerAssessmentPreregistration,
    MatchedElectrostaticReferenceCase,
    MatchedElectrostaticReferencePanel,
    OperationalElectrostaticLedgerPrediction,
    QuantitativeSourceCaseEvidence,
    QuantitativeSourceGatePanel,
    assess_operational_electrostatic_ledgers,
)
from maple.solvation.release.source_mep import (
    ContinuumReferenceValidationBinding,
    compare_continuum_active_source_to_reference,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _Continuum:
    cavity_profile_id = "test.smooth-harmonic-cavity.v1"
    provenance_sha256 = _digest("continuum-provenance")
    surface_operator = np.asarray([[2.0, 0.2], [0.2, 1.4]])
    source_to_boundary = np.asarray([[0.4, 0.1, 0.0, -0.2], [0.2, -0.1, 0.3, 0.1]])

    def __init__(self, case_id: str) -> None:
        self.geometry_sha256 = _digest(f"geometry-{case_id}")
        self.continuum_configuration_sha256 = _digest(f"continuum-{case_id}")
        self.topology_sha256 = _digest(f"topology-{case_id}")

    def source_rhs(self, source):
        return self.source_to_boundary @ np.asarray(source).reshape(-1)


def _validation(continuum: _Continuum) -> ContinuumReferenceValidationBinding:
    return ContinuumReferenceValidationBinding(
        validation_identity="test-independent-continuum-reference-v1",
        validation_evidence_sha256=_digest("continuum-validation-evidence"),
        geometry_sha256=continuum.geometry_sha256,
        continuum_configuration_sha256=(continuum.continuum_configuration_sha256),
        continuum_provenance_sha256=continuum.provenance_sha256,
        topology_sha256=continuum.topology_sha256,
        cavity_profile_id=continuum.cavity_profile_id,
        validated_for_source_gate=True,
    )


def _case_bundle(
    case_id: str,
    *,
    vacuum: float,
    polarized_vacuum: float,
    pcm_total: float,
    dielectric: float,
):
    continuum = _Continuum(case_id)
    source = np.asarray([[0.2, -0.1, 0.05, 0.03]])
    reference_rhs = continuum.source_rhs(source)
    comparison = compare_continuum_active_source_to_reference(
        continuum=continuum,
        predicted_source=source,
        reference_boundary_rhs=reference_rhs,
        reference_identity=case_id,
        reference_artifact_sha256=_digest(f"reference-{case_id}"),
        continuum_reference_validation=_validation(continuum),
        fixed_source_energy_error_budget_eV=1.0e-4,
    )
    reference = MatchedElectrostaticReferenceCase(
        case_id=case_id,
        molecule_group_sha256=_digest(f"molecule-{case_id}"),
        geometry_sha256=continuum.geometry_sha256,
        reference_artifact_sha256=_digest(f"reference-{case_id}"),
        reference_method_id="pinned-qm-pcm-electrostatic-v1",
        reference_protocol_sha256=_digest("reference-protocol"),
        qm_runtime_manifest_sha256=_digest("qm-runtime"),
        total_charge=0,
        spin_multiplicity=1,
        vacuum_density_sha256=_digest(f"vacuum-density-{case_id}"),
        pcm_density_sha256=_digest(f"pcm-density-{case_id}"),
        reference_boundary_rhs_sha256=(comparison.reference_boundary_rhs_sha256),
        continuum_equation_id="matched-pcmsolver-cpcm-electrostatic-v1",
        continuum_protocol_sha256=_digest("continuum-protocol"),
        continuum_configuration_sha256=(continuum.continuum_configuration_sha256),
        continuum_provenance_sha256=continuum.provenance_sha256,
        topology_sha256=continuum.topology_sha256,
        cavity_profile_id=continuum.cavity_profile_id,
        dielectric=dielectric,
        vacuum_ground_state_energy_eV=vacuum,
        polarized_density_vacuum_energy_eV=polarized_vacuum,
        pcm_electrostatic_total_energy_eV=pcm_total,
    )
    return reference, comparison, continuum, source, reference_rhs


def _panel_bundle():
    first = _case_bundle(
        "case-a",
        vacuum=-100.0,
        polarized_vacuum=-99.8,
        pcm_total=-100.1,
        dielectric=20.0,
    )
    second = _case_bundle(
        "case-b",
        vacuum=-200.0,
        polarized_vacuum=-199.7,
        pcm_total=-200.2,
        dielectric=80.0,
    )
    references = (first[0], second[0])
    panel = MatchedElectrostaticReferencePanel(
        panel_id="test-matched-electrostatic-panel-v1",
        preregistration_sha256=_digest("reference-preregistration"),
        expected_case_ids=("case-a", "case-b"),
        cases=tuple(reversed(references)),
    )
    gate = QuantitativeSourceGatePanel(
        panel_id="test-source-gate-panel-v1",
        preregistration_sha256=_digest("source-preregistration"),
        source_provider_id="test.scalar-first-source.v1",
        source_profile_id="test.scalar-first-profile.v1",
        source_configuration_sha256=_digest("source-configuration"),
        source_provenance_sha256=_digest("source-provenance"),
        source_space_sha256=_digest("source-space"),
        source_gate_protocol_sha256=_digest("source-gate-protocol"),
        reference_panel=panel,
        cases=(_physical_case(second[1]), _physical_case(first[1])),
    )
    return panel, gate, (first, second)


def _physical_case(
    comparison, *, surface_mep_passed: bool = True
) -> QuantitativeSourceCaseEvidence:
    return QuantitativeSourceCaseEvidence(
        case_id=comparison.reference_identity,
        continuum_active_comparison=comparison,
        electrostatic_observables_evidence_sha256=_digest(
            f"observables-{comparison.reference_identity}"
        ),
        far_field_evidence_sha256=_digest(f"far-field-{comparison.reference_identity}"),
        surface_mep_evidence_sha256=_digest(
            f"surface-mep-{comparison.reference_identity}"
        ),
        total_charge_gate_passed=True,
        dipole_gate_passed=True,
        quadrupole_gate_passed=True,
        far_field_gate_passed=True,
        surface_mep_gate_passed=surface_mep_passed,
        fixed_source_pcm_gate_passed=comparison.case_passed,
    )


def _preregistration(panel: MatchedElectrostaticReferencePanel):
    return LedgerAssessmentPreregistration(
        assessment_id="test-operational-ledger-assessment-v1",
        preregistration_artifact_sha256=_digest("ledger-preregistration"),
        reference_panel_sha256=panel.content_sha256,
        source_gate_protocol_sha256=_digest("source-gate-protocol"),
        expected_case_ids=panel.expected_case_ids,
        ledger_priority=("test.phi0.v1", "test.phi1-delta.v1"),
        maximum_solute_distortion_absolute_error_eV=0.05,
        maximum_continuum_stabilization_absolute_error_eV=0.05,
        maximum_total_electrostatic_absolute_error_eV=0.05,
    )


def _prediction(
    *,
    reference: MatchedElectrostaticReferenceCase,
    gate: QuantitativeSourceGatePanel,
    ledger_id: str,
    distortion: float,
    continuum: float,
) -> OperationalElectrostaticLedgerPrediction:
    return OperationalElectrostaticLedgerPrediction(
        case_id=reference.case_id,
        ledger_id=ledger_id,
        ledger_formula_sha256=_digest(f"formula-{ledger_id}"),
        ledger_configuration_sha256=_digest(f"config-{ledger_id}"),
        ledger_semantics_evidence_sha256=_digest(f"semantics-{ledger_id}"),
        ledger_semantics_gate_passed=True,
        evaluation_artifact_sha256=_digest(
            f"evaluation-{ledger_id}-{reference.case_id}"
        ),
        state_sha256=_digest(f"state-{ledger_id}-{reference.case_id}"),
        source_gate_panel_sha256=gate.content_sha256,
        reference_panel_sha256=gate.reference_panel.content_sha256,
        geometry_sha256=reference.geometry_sha256,
        continuum_equation_id=reference.continuum_equation_id,
        continuum_protocol_sha256=reference.continuum_protocol_sha256,
        continuum_configuration_sha256=(reference.continuum_configuration_sha256),
        continuum_provenance_sha256=reference.continuum_provenance_sha256,
        topology_sha256=reference.topology_sha256,
        cavity_profile_id=reference.cavity_profile_id,
        dielectric=reference.dielectric,
        solute_distortion_energy_eV=distortion,
        continuum_stabilization_energy_eV=continuum,
    )


def test_reference_case_derives_closed_components_and_content_identity() -> None:
    reference, *_ = _case_bundle(
        "case-a",
        vacuum=-100.0,
        polarized_vacuum=-99.8,
        pcm_total=-100.1,
        dielectric=20.0,
    )
    assert reference.solute_distortion_energy_eV == pytest.approx(0.2)
    assert reference.continuum_stabilization_energy_eV == pytest.approx(-0.3)
    assert reference.total_electrostatic_solvation_energy_eV == pytest.approx(-0.1)
    assert (
        reference.solute_distortion_energy_eV
        + reference.continuum_stabilization_energy_eV
        == pytest.approx(reference.total_electrostatic_solvation_energy_eV)
    )
    assert len(reference.content_sha256) == 64
    assert reference.as_dict()["capability_admitted"] is False
    assert (
        replace(reference, polarized_density_vacuum_energy_eV=-99.7).content_sha256
        != reference.content_sha256
    )
    json.dumps(reference.as_dict(), sort_keys=True)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("fixed_nuclear_geometry", False, "requires fixed geometry"),
        ("pcm_density_reoptimized_in_vacuum", True, "forbids vacuum"),
        ("electrostatics_only", False, "requires fixed geometry"),
        ("nonpolar_terms_included", True, "forbids vacuum"),
        ("standard_state_terms_included", True, "forbids vacuum"),
        ("experimental_solvation_labels_used", True, "forbids vacuum"),
    ),
)
def test_reference_case_rejects_semantic_contamination(field, value, match) -> None:
    reference, *_ = _case_bundle(
        "case-a",
        vacuum=-100.0,
        polarized_vacuum=-99.8,
        pcm_total=-100.1,
        dielectric=20.0,
    )
    with pytest.raises(ValueError, match=match):
        replace(reference, **{field: value})


def test_panels_are_canonical_and_reject_mixed_reference_identity() -> None:
    panel, gate, _ = _panel_bundle()
    assert tuple(case.case_id for case in panel.cases) == ("case-a", "case-b")
    assert tuple(case.reference_identity for case in gate.comparisons) == (
        "case-a",
        "case-b",
    )
    assert gate.case_pass_count == 2
    assert gate.passed is True
    assert gate.as_dict()["capability_admitted"] is False

    incomplete_physics = replace(
        gate,
        cases=(
            replace(gate.cases[0], surface_mep_gate_passed=False),
            gate.cases[1],
        ),
    )
    assert incomplete_physics.case_pass_count == 1
    assert incomplete_physics.passed is False
    with pytest.raises(ValueError, match="must equal"):
        replace(gate.cases[0], fixed_source_pcm_gate_passed=False)

    with pytest.raises(ValueError, match="cannot mix cavity_profile_id"):
        replace(
            panel,
            cases=(panel.cases[0], replace(panel.cases[1], cavity_profile_id="other")),
        )
    with pytest.raises(ValueError, match="reference geometry_sha256"):
        replace(
            gate,
            cases=(
                replace(
                    gate.cases[0],
                    continuum_active_comparison=replace(
                        gate.comparisons[0], geometry_sha256=_digest("wrong")
                    ),
                ),
                gate.cases[1],
            ),
        )


def test_failed_source_gate_forbids_ledger_assessment() -> None:
    panel, gate, bundles = _panel_bundle()
    reference, _, continuum, source, reference_rhs = bundles[0]
    failed_comparison = compare_continuum_active_source_to_reference(
        continuum=continuum,
        predicted_source=source + np.asarray([[0.2, 0.0, 0.0, 0.0]]),
        reference_boundary_rhs=reference_rhs,
        reference_identity=reference.case_id,
        reference_artifact_sha256=reference.reference_artifact_sha256,
        continuum_reference_validation=_validation(continuum),
        fixed_source_energy_error_budget_eV=1.0e-10,
    )
    failed_gate = replace(
        gate,
        cases=(
            _physical_case(failed_comparison),
            gate.cases[1],
        ),
    )
    assert failed_gate.passed is False
    with pytest.raises(ValueError, match="source gate failed"):
        assess_operational_electrostatic_ledgers(
            preregistration=_preregistration(panel),
            reference_panel=panel,
            source_gate_panel=failed_gate,
            predictions=(),
        )


def test_assessment_selects_only_preregistered_all_case_component_pass() -> None:
    panel, gate, _ = _panel_bundle()
    preregistration = _preregistration(panel)
    predictions = []
    for reference in panel.cases:
        predictions.append(
            _prediction(
                reference=reference,
                gate=gate,
                ledger_id="test.phi0.v1",
                distortion=0.0,
                continuum=reference.continuum_stabilization_energy_eV,
            )
        )
        predictions.append(
            _prediction(
                reference=reference,
                gate=gate,
                ledger_id="test.phi1-delta.v1",
                distortion=reference.solute_distortion_energy_eV,
                continuum=reference.continuum_stabilization_energy_eV,
            )
        )
    report = assess_operational_electrostatic_ledgers(
        preregistration=preregistration,
        reference_panel=panel,
        source_gate_panel=gate,
        predictions=tuple(reversed(predictions)),
    )
    assert preregistration.selection_rule == LEDGER_SELECTION_RULE
    assert report.passing_ledger_ids == ("test.phi1-delta.v1",)
    assert report.selected_ledger_id == "test.phi1-delta.v1"
    assert report.as_dict()["capability_admitted"] is False
    assert len(report.case_assessments) == 4
    json.dumps(report.as_dict(), sort_keys=True)
    with pytest.raises(ValueError, match="first passing"):
        replace(report, selected_ledger_id="test.phi0.v1")
    with pytest.raises(ValueError, match="canonical"):
        replace(report, case_assessments=tuple(reversed(report.case_assessments)))


def test_assessment_rejects_incomplete_matrix_and_identity_mismatch() -> None:
    panel, gate, _ = _panel_bundle()
    preregistration = _preregistration(panel)
    reference = panel.cases[0]
    one = _prediction(
        reference=reference,
        gate=gate,
        ledger_id="test.phi0.v1",
        distortion=0.0,
        continuum=reference.continuum_stabilization_energy_eV,
    )
    with pytest.raises(ValueError, match="exact preregistered"):
        assess_operational_electrostatic_ledgers(
            preregistration=preregistration,
            reference_panel=panel,
            source_gate_panel=gate,
            predictions=(one,),
        )
    with pytest.raises(ValueError, match="reference panel"):
        assess_operational_electrostatic_ledgers(
            preregistration=replace(
                preregistration, reference_panel_sha256=_digest("wrong-panel")
            ),
            reference_panel=panel,
            source_gate_panel=gate,
            predictions=(one,),
        )

    complete = []
    for ledger_id in preregistration.ledger_priority:
        for case in panel.cases:
            complete.append(
                _prediction(
                    reference=case,
                    gate=gate,
                    ledger_id=ledger_id,
                    distortion=case.solute_distortion_energy_eV,
                    continuum=case.continuum_stabilization_energy_eV,
                )
            )
    complete[0] = replace(complete[0], topology_sha256=_digest("wrong-topology"))
    with pytest.raises(ValueError, match="topology_sha256"):
        assess_operational_electrostatic_ledgers(
            preregistration=preregistration,
            reference_panel=panel,
            source_gate_panel=gate,
            predictions=tuple(complete),
        )


def test_prediction_and_preregistration_reject_posthoc_or_mixed_components() -> None:
    panel, gate, _ = _panel_bundle()
    reference = panel.cases[0]
    prediction = _prediction(
        reference=reference,
        gate=gate,
        ledger_id="test.phi0.v1",
        distortion=0.0,
        continuum=reference.continuum_stabilization_energy_eV,
    )
    assert prediction.total_electrostatic_solvation_energy_eV == pytest.approx(
        prediction.continuum_stabilization_energy_eV
    )
    with pytest.raises(ValueError, match="electrostatic-only"):
        replace(prediction, nonpolar_terms_included=True)
    with pytest.raises(ValueError, match="failed energy semantics"):
        replace(prediction, ledger_semantics_gate_passed=False)
    with pytest.raises(ValueError, match="experimental or total"):
        replace(_preregistration(panel), experimental_solvation_labels_used=True)
    with pytest.raises(ValueError, match="before blind access"):
        replace(
            _preregistration(panel),
            blind_results_accessed_before_preregistration=True,
        )


def test_constructor_collections_are_detached() -> None:
    panel, gate, _ = _panel_bundle()
    expected = list(panel.expected_case_ids)
    cases = list(panel.cases)
    detached = MatchedElectrostaticReferencePanel(
        panel_id=panel.panel_id,
        preregistration_sha256=panel.preregistration_sha256,
        expected_case_ids=expected,  # type: ignore[arg-type]
        cases=cases,  # type: ignore[arg-type]
    )
    expected[0] = "mutated"
    cases.clear()
    assert detached.expected_case_ids == panel.expected_case_ids
    assert detached.cases == panel.cases
    assert gate.reference_panel.content_sha256 == panel.content_sha256
