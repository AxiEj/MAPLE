"""Matched electrostatic references and source-gated ledger assessment.

The Route-2 fixed point selects a state, not a unique physical energy ledger.
This module therefore keeps three decisions separate and content addressed:

1. a matched QM/PCM decomposition at one fixed nuclear geometry;
2. an all-case quantitative source gate on the same continuum identities;
3. a preregistered comparison of operational ledgers against those components.

No object in this module admits E/F/H/V/M.  In particular, ledger assessment
is refused unless every source case passed first.  Experimental or total
solvation free energies, nonpolar terms, and standard-state corrections are
outside this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from maple.solvation.release.evidence import canonical_json_sha256
from maple.solvation.release.source_mep import ContinuumActiveSourceComparison

MATCHED_ELECTROSTATIC_REFERENCE_CONTRACT_VERSION = (
    "route2-matched-qm-pcm-electrostatic-reference-v1"
)
MATCHED_ELECTROSTATIC_PANEL_CONTRACT_VERSION = (
    "route2-matched-qm-pcm-electrostatic-panel-v1"
)
SOURCE_GATE_PANEL_CONTRACT_VERSION = "route2-quantitative-source-gate-panel-v1"
SOURCE_PHYSICAL_CASE_CONTRACT_VERSION = "route2-quantitative-source-physical-case-v1"
LEDGER_ASSESSMENT_PREREGISTRATION_CONTRACT_VERSION = (
    "route2-electrostatic-ledger-assessment-preregistration-v1"
)
LEDGER_PREDICTION_CONTRACT_VERSION = (
    "route2-operational-electrostatic-ledger-prediction-v1"
)
LEDGER_ASSESSMENT_CONTRACT_VERSION = (
    "route2-operational-electrostatic-ledger-assessment-v1"
)
LEDGER_SELECTION_RULE = (
    "first-passing-preregistered-priority-all-cases-all-components-v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return result


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be exactly bool.")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _ordered_unique_text(values: object, *, name: str) -> tuple[str, ...]:
    try:
        result = tuple(_text(value, name=name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings.") from exc
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values.")
    return result


def _sorted_unique_text(values: object, *, name: str) -> tuple[str, ...]:
    result = _ordered_unique_text(values, name=name)
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must be sorted for deterministic identity.")
    return result


@dataclass(frozen=True, slots=True)
class MatchedElectrostaticReferenceCase:
    """One fixed-geometry electrostatic-only QM/PCM decomposition.

    ``polarized_density_vacuum_energy_eV`` is a non-self-consistent evaluation
    of the converged PCM density in the *vacuum* electronic functional.  It is
    not a second vacuum optimization.  Consequently the three derived terms
    are

    ``distortion = E_vac[gamma_pcm] - E_vac[gamma_vac]``;
    ``continuum = E_pcm[gamma_pcm] - E_vac[gamma_pcm]``;
    ``total = E_pcm[gamma_pcm] - E_vac[gamma_vac]``.
    """

    case_id: str
    molecule_group_sha256: str
    geometry_sha256: str
    reference_artifact_sha256: str
    reference_method_id: str
    reference_protocol_sha256: str
    qm_runtime_manifest_sha256: str
    total_charge: int
    spin_multiplicity: int
    vacuum_density_sha256: str
    pcm_density_sha256: str
    reference_boundary_rhs_sha256: str
    continuum_equation_id: str
    continuum_protocol_sha256: str
    continuum_configuration_sha256: str
    continuum_provenance_sha256: str
    topology_sha256: str
    cavity_profile_id: str
    dielectric: float
    vacuum_ground_state_energy_eV: float
    polarized_density_vacuum_energy_eV: float
    pcm_electrostatic_total_energy_eV: float
    fixed_nuclear_geometry: bool = True
    vacuum_density_self_consistent: bool = True
    pcm_density_self_consistent: bool = True
    pcm_density_reoptimized_in_vacuum: bool = False
    electrostatics_only: bool = True
    nonpolar_terms_included: bool = False
    standard_state_terms_included: bool = False
    experimental_solvation_labels_used: bool = False
    energy_unit: str = "eV"
    contract_version: str = MATCHED_ELECTROSTATIC_REFERENCE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "reference_method_id",
            "continuum_equation_id",
            "cavity_profile_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "molecule_group_sha256",
            "geometry_sha256",
            "reference_artifact_sha256",
            "reference_protocol_sha256",
            "qm_runtime_manifest_sha256",
            "vacuum_density_sha256",
            "pcm_density_sha256",
            "reference_boundary_rhs_sha256",
            "continuum_protocol_sha256",
            "continuum_configuration_sha256",
            "continuum_provenance_sha256",
            "topology_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.contract_version != MATCHED_ELECTROSTATIC_REFERENCE_CONTRACT_VERSION:
            raise ValueError("matched electrostatic reference version is unsupported.")
        if self.energy_unit != "eV":
            raise ValueError("matched electrostatic reference energies must use eV.")
        if isinstance(self.total_charge, bool) or not isinstance(
            self.total_charge, int
        ):
            raise TypeError("total_charge must be an integer.")
        if (
            isinstance(self.spin_multiplicity, bool)
            or not isinstance(self.spin_multiplicity, int)
            or self.spin_multiplicity < 1
        ):
            raise ValueError("spin_multiplicity must be a positive integer.")
        dielectric = _finite(self.dielectric, name="dielectric")
        if dielectric <= 1.0:
            raise ValueError("a PCM reference dielectric must be greater than one.")
        object.__setattr__(self, "dielectric", dielectric)
        for name in (
            "vacuum_ground_state_energy_eV",
            "polarized_density_vacuum_energy_eV",
            "pcm_electrostatic_total_energy_eV",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        required_true = (
            "fixed_nuclear_geometry",
            "vacuum_density_self_consistent",
            "pcm_density_self_consistent",
            "electrostatics_only",
        )
        required_false = (
            "pcm_density_reoptimized_in_vacuum",
            "nonpolar_terms_included",
            "standard_state_terms_included",
            "experimental_solvation_labels_used",
        )
        for name in (*required_true, *required_false):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name=name))
        if not all(getattr(self, name) for name in required_true):
            raise ValueError(
                "matched decomposition requires fixed geometry, converged vacuum/PCM "
                "densities, and electrostatics-only PCM."
            )
        if any(getattr(self, name) for name in required_false):
            raise ValueError(
                "matched decomposition forbids vacuum reoptimization of the PCM "
                "density, nonpolar/standard-state terms, and experimental labels."
            )

    @property
    def solute_distortion_energy_eV(self) -> float:
        return (
            self.polarized_density_vacuum_energy_eV - self.vacuum_ground_state_energy_eV
        )

    @property
    def continuum_stabilization_energy_eV(self) -> float:
        return (
            self.pcm_electrostatic_total_energy_eV
            - self.polarized_density_vacuum_energy_eV
        )

    @property
    def total_electrostatic_solvation_energy_eV(self) -> float:
        return (
            self.pcm_electrostatic_total_energy_eV - self.vacuum_ground_state_energy_eV
        )

    @property
    def content_sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        payload = {field: getattr(self, field) for field in self.__dataclass_fields__}
        payload.update(
            {
                "solute_distortion_energy_eV": self.solute_distortion_energy_eV,
                "continuum_stabilization_energy_eV": (
                    self.continuum_stabilization_energy_eV
                ),
                "total_electrostatic_solvation_energy_eV": (
                    self.total_electrostatic_solvation_energy_eV
                ),
            }
        )
        return payload

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
            "claim_boundary": (
                "Matched electrostatic component evidence only; no CDS, standard "
                "state, total experimental solvation, or E/F/H/V/M admission."
            ),
        }


@dataclass(frozen=True, slots=True)
class MatchedElectrostaticReferencePanel:
    """A preregistered, exact set of matched reference cases."""

    panel_id: str
    preregistration_sha256: str
    expected_case_ids: tuple[str, ...]
    cases: tuple[MatchedElectrostaticReferenceCase, ...]
    contract_version: str = MATCHED_ELECTROSTATIC_PANEL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", _text(self.panel_id, name="panel_id"))
        object.__setattr__(
            self,
            "preregistration_sha256",
            _digest(self.preregistration_sha256, name="preregistration_sha256"),
        )
        if self.contract_version != MATCHED_ELECTROSTATIC_PANEL_CONTRACT_VERSION:
            raise ValueError("matched electrostatic panel version is unsupported.")
        expected = _sorted_unique_text(self.expected_case_ids, name="expected_case_ids")
        object.__setattr__(self, "expected_case_ids", expected)
        cases = tuple(self.cases)
        if not cases or not all(
            isinstance(case, MatchedElectrostaticReferenceCase) for case in cases
        ):
            raise TypeError("cases must contain matched electrostatic references.")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("matched reference case IDs must be unique.")
        cases = tuple(sorted(cases, key=lambda case: case.case_id))
        if tuple(case.case_id for case in cases) != expected:
            raise ValueError("matched reference cases differ from the frozen case set.")
        object.__setattr__(self, "cases", cases)
        shared_fields = (
            "reference_method_id",
            "reference_protocol_sha256",
            "qm_runtime_manifest_sha256",
            "continuum_equation_id",
            "continuum_protocol_sha256",
            "continuum_provenance_sha256",
            "cavity_profile_id",
        )
        for field in shared_fields:
            if len({getattr(case, field) for case in cases}) != 1:
                raise ValueError(
                    f"matched reference panel cannot mix {field} identities."
                )
        if len({(case.geometry_sha256, case.dielectric) for case in cases}) != len(
            cases
        ):
            raise ValueError(
                "matched reference geometry/dielectric cases must be unique."
            )

    @property
    def content_sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "panel_id": self.panel_id,
            "preregistration_sha256": self.preregistration_sha256,
            "expected_case_ids": list(self.expected_case_ids),
            "cases": [case.as_dict() for case in self.cases],
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
        }


@dataclass(frozen=True, slots=True)
class QuantitativeSourceCaseEvidence:
    """One complete source-physics decision on a matched reference case.

    A favorable continuum-active energy metric is not sufficient when total
    charge, angular multipoles, the far field, or the cavity-surface MEP are
    wrong.  ``passed`` is therefore derived from every evidence-bound layer.
    """

    case_id: str
    continuum_active_comparison: ContinuumActiveSourceComparison
    electrostatic_observables_evidence_sha256: str
    far_field_evidence_sha256: str
    surface_mep_evidence_sha256: str
    total_charge_gate_passed: bool
    dipole_gate_passed: bool
    quadrupole_gate_passed: bool
    far_field_gate_passed: bool
    surface_mep_gate_passed: bool
    fixed_source_pcm_gate_passed: bool
    contract_version: str = SOURCE_PHYSICAL_CASE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, name="case_id"))
        if not isinstance(
            self.continuum_active_comparison, ContinuumActiveSourceComparison
        ):
            raise TypeError(
                "continuum_active_comparison must be a matched source comparison."
            )
        if self.contract_version != SOURCE_PHYSICAL_CASE_CONTRACT_VERSION:
            raise ValueError("source physical case contract version is unsupported.")
        if self.case_id != self.continuum_active_comparison.reference_identity:
            raise ValueError("source physical case ID differs from its comparison.")
        for name in (
            "electrostatic_observables_evidence_sha256",
            "far_field_evidence_sha256",
            "surface_mep_evidence_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        for name in (
            "total_charge_gate_passed",
            "dipole_gate_passed",
            "quadrupole_gate_passed",
            "far_field_gate_passed",
            "surface_mep_gate_passed",
            "fixed_source_pcm_gate_passed",
        ):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name=name))
        if (
            self.fixed_source_pcm_gate_passed
            is not self.continuum_active_comparison.case_passed
        ):
            raise ValueError(
                "fixed_source_pcm_gate_passed must equal the continuum-active "
                "comparison decision."
            )

    @property
    def passed(self) -> bool:
        return all(
            (
                self.total_charge_gate_passed,
                self.dipole_gate_passed,
                self.quadrupole_gate_passed,
                self.far_field_gate_passed,
                self.surface_mep_gate_passed,
                self.fixed_source_pcm_gate_passed,
            )
        )

    @property
    def content_sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "case_id": self.case_id,
            "continuum_active_comparison": (self.continuum_active_comparison.as_dict()),
            "electrostatic_observables_evidence_sha256": (
                self.electrostatic_observables_evidence_sha256
            ),
            "far_field_evidence_sha256": self.far_field_evidence_sha256,
            "surface_mep_evidence_sha256": self.surface_mep_evidence_sha256,
            "total_charge_gate_passed": self.total_charge_gate_passed,
            "dipole_gate_passed": self.dipole_gate_passed,
            "quadrupole_gate_passed": self.quadrupole_gate_passed,
            "far_field_gate_passed": self.far_field_gate_passed,
            "surface_mep_gate_passed": self.surface_mep_gate_passed,
            "fixed_source_pcm_gate_passed": self.fixed_source_pcm_gate_passed,
            "passed": self.passed,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
        }


@dataclass(frozen=True, slots=True)
class QuantitativeSourceGatePanel:
    """Bind all matched source comparisons to one exact source identity."""

    panel_id: str
    preregistration_sha256: str
    source_provider_id: str
    source_profile_id: str
    source_configuration_sha256: str
    source_provenance_sha256: str
    source_space_sha256: str
    source_gate_protocol_sha256: str
    reference_panel: MatchedElectrostaticReferencePanel
    cases: tuple[QuantitativeSourceCaseEvidence, ...]
    contract_version: str = SOURCE_GATE_PANEL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("panel_id", "source_provider_id", "source_profile_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "preregistration_sha256",
            "source_configuration_sha256",
            "source_provenance_sha256",
            "source_space_sha256",
            "source_gate_protocol_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.contract_version != SOURCE_GATE_PANEL_CONTRACT_VERSION:
            raise ValueError("quantitative source gate version is unsupported.")
        if not isinstance(self.reference_panel, MatchedElectrostaticReferencePanel):
            raise TypeError("reference_panel must be a matched reference panel.")
        cases = tuple(self.cases)
        if not cases or not all(
            isinstance(case, QuantitativeSourceCaseEvidence) for case in cases
        ):
            raise TypeError("cases must contain complete source-physics evidence.")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("source physical case identities must be unique.")
        cases = tuple(sorted(cases, key=lambda case: case.case_id))
        if tuple(case.case_id for case in cases) != (
            self.reference_panel.expected_case_ids
        ):
            raise ValueError("source evidence differs from the matched case set.")
        references = {case.case_id: case for case in self.reference_panel.cases}
        for case in cases:
            comparison = case.continuum_active_comparison
            reference = references[case.case_id]
            expected = {
                "reference_artifact_sha256": reference.reference_artifact_sha256,
                "reference_boundary_rhs_sha256": (
                    reference.reference_boundary_rhs_sha256
                ),
                "geometry_sha256": reference.geometry_sha256,
                "continuum_configuration_sha256": (
                    reference.continuum_configuration_sha256
                ),
                "continuum_provenance_sha256": (reference.continuum_provenance_sha256),
                "topology_sha256": reference.topology_sha256,
                "cavity_profile_id": reference.cavity_profile_id,
            }
            for field, wanted in expected.items():
                if getattr(comparison, field) != wanted:
                    raise ValueError(
                        "source comparison does not match reference " f"{field}."
                    )
        object.__setattr__(self, "cases", cases)

    @property
    def comparisons(self) -> tuple[ContinuumActiveSourceComparison, ...]:
        return tuple(case.continuum_active_comparison for case in self.cases)

    @property
    def case_pass_count(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def passed(self) -> bool:
        return self.case_pass_count == len(self.cases)

    @property
    def content_sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "panel_id": self.panel_id,
            "preregistration_sha256": self.preregistration_sha256,
            "source_provider_id": self.source_provider_id,
            "source_profile_id": self.source_profile_id,
            "source_configuration_sha256": self.source_configuration_sha256,
            "source_provenance_sha256": self.source_provenance_sha256,
            "source_space_sha256": self.source_space_sha256,
            "source_gate_protocol_sha256": self.source_gate_protocol_sha256,
            "reference_panel_sha256": self.reference_panel.content_sha256,
            "case_records": [case.as_dict() for case in self.cases],
            "case_pass_count": self.case_pass_count,
            "case_count": len(self.cases),
            "passed": self.passed,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
            "claim_boundary": (
                "Passing authorizes component-ledger assessment only; it does not "
                "admit a ledger, force, profile, or E/F/H/V/M capability."
            ),
        }


@dataclass(frozen=True, slots=True)
class LedgerAssessmentPreregistration:
    """Freeze case set, candidate priority, and component budgets in advance."""

    assessment_id: str
    preregistration_artifact_sha256: str
    reference_panel_sha256: str
    source_gate_protocol_sha256: str
    expected_case_ids: tuple[str, ...]
    ledger_priority: tuple[str, ...]
    maximum_solute_distortion_absolute_error_eV: float
    maximum_continuum_stabilization_absolute_error_eV: float
    maximum_total_electrostatic_absolute_error_eV: float
    experimental_solvation_labels_used: bool = False
    blind_results_accessed_before_preregistration: bool = False
    selection_rule: str = LEDGER_SELECTION_RULE
    contract_version: str = LEDGER_ASSESSMENT_PREREGISTRATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "assessment_id", _text(self.assessment_id, name="assessment_id")
        )
        for name in (
            "preregistration_artifact_sha256",
            "reference_panel_sha256",
            "source_gate_protocol_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.contract_version != (
            LEDGER_ASSESSMENT_PREREGISTRATION_CONTRACT_VERSION
        ):
            raise ValueError(
                "ledger assessment preregistration version is unsupported."
            )
        if self.selection_rule != LEDGER_SELECTION_RULE:
            raise ValueError("ledger selection rule differs from the frozen contract.")
        object.__setattr__(
            self,
            "expected_case_ids",
            _sorted_unique_text(self.expected_case_ids, name="expected_case_ids"),
        )
        object.__setattr__(
            self,
            "ledger_priority",
            _ordered_unique_text(self.ledger_priority, name="ledger_priority"),
        )
        for name in (
            "maximum_solute_distortion_absolute_error_eV",
            "maximum_continuum_stabilization_absolute_error_eV",
            "maximum_total_electrostatic_absolute_error_eV",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        for name in (
            "experimental_solvation_labels_used",
            "blind_results_accessed_before_preregistration",
        ):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name=name))
        if self.experimental_solvation_labels_used:
            raise ValueError("experimental or total solvation labels are forbidden.")
        if self.blind_results_accessed_before_preregistration:
            raise ValueError("ledger assessment must be frozen before blind access.")

    @property
    def content_sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__} | {
            "expected_case_ids": list(self.expected_case_ids),
            "ledger_priority": list(self.ledger_priority),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
        }


@dataclass(frozen=True, slots=True)
class OperationalElectrostaticLedgerPrediction:
    """One candidate ledger's component prediction for one matched case."""

    case_id: str
    ledger_id: str
    ledger_formula_sha256: str
    ledger_configuration_sha256: str
    ledger_semantics_evidence_sha256: str
    ledger_semantics_gate_passed: bool
    evaluation_artifact_sha256: str
    state_sha256: str
    source_gate_panel_sha256: str
    reference_panel_sha256: str
    geometry_sha256: str
    continuum_equation_id: str
    continuum_protocol_sha256: str
    continuum_configuration_sha256: str
    continuum_provenance_sha256: str
    topology_sha256: str
    cavity_profile_id: str
    dielectric: float
    solute_distortion_energy_eV: float
    continuum_stabilization_energy_eV: float
    electrostatics_only: bool = True
    nonpolar_terms_included: bool = False
    standard_state_terms_included: bool = False
    experimental_solvation_labels_used: bool = False
    energy_unit: str = "eV"
    contract_version: str = LEDGER_PREDICTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "ledger_id",
            "continuum_equation_id",
            "cavity_profile_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "ledger_formula_sha256",
            "ledger_configuration_sha256",
            "ledger_semantics_evidence_sha256",
            "evaluation_artifact_sha256",
            "state_sha256",
            "source_gate_panel_sha256",
            "reference_panel_sha256",
            "geometry_sha256",
            "continuum_protocol_sha256",
            "continuum_configuration_sha256",
            "continuum_provenance_sha256",
            "topology_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.contract_version != LEDGER_PREDICTION_CONTRACT_VERSION:
            raise ValueError("ledger prediction contract version is unsupported.")
        object.__setattr__(
            self,
            "ledger_semantics_gate_passed",
            _strict_bool(
                self.ledger_semantics_gate_passed,
                name="ledger_semantics_gate_passed",
            ),
        )
        if not self.ledger_semantics_gate_passed:
            raise ValueError(
                "a ledger with failed energy semantics cannot enter component "
                "assessment."
            )
        if self.energy_unit != "eV":
            raise ValueError("ledger predictions must use eV.")
        dielectric = _finite(self.dielectric, name="dielectric")
        if dielectric <= 1.0:
            raise ValueError("ledger prediction dielectric must exceed one.")
        object.__setattr__(self, "dielectric", dielectric)
        for name in (
            "solute_distortion_energy_eV",
            "continuum_stabilization_energy_eV",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        for name in (
            "electrostatics_only",
            "nonpolar_terms_included",
            "standard_state_terms_included",
            "experimental_solvation_labels_used",
        ):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name=name))
        if not self.electrostatics_only or any(
            (
                self.nonpolar_terms_included,
                self.standard_state_terms_included,
                self.experimental_solvation_labels_used,
            )
        ):
            raise ValueError(
                "ledger predictions must be electrostatic-only and label-free."
            )

    @property
    def total_electrostatic_solvation_energy_eV(self) -> float:
        return self.solute_distortion_energy_eV + self.continuum_stabilization_energy_eV

    @property
    def content_sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__} | {
            "total_electrostatic_solvation_energy_eV": (
                self.total_electrostatic_solvation_energy_eV
            )
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
        }


@dataclass(frozen=True, slots=True)
class LedgerCaseAssessment:
    case_id: str
    ledger_id: str
    solute_distortion_absolute_error_eV: float
    continuum_stabilization_absolute_error_eV: float
    total_electrostatic_absolute_error_eV: float
    maximum_solute_distortion_absolute_error_eV: float
    maximum_continuum_stabilization_absolute_error_eV: float
    maximum_total_electrostatic_absolute_error_eV: float
    case_passed: bool

    def __post_init__(self) -> None:
        for name in ("case_id", "ledger_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "solute_distortion_absolute_error_eV",
            "continuum_stabilization_absolute_error_eV",
            "total_electrostatic_absolute_error_eV",
        ):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        for name in (
            "maximum_solute_distortion_absolute_error_eV",
            "maximum_continuum_stabilization_absolute_error_eV",
            "maximum_total_electrostatic_absolute_error_eV",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "case_passed",
            _strict_bool(self.case_passed, name="case_passed"),
        )
        expected = bool(
            self.solute_distortion_absolute_error_eV
            <= self.maximum_solute_distortion_absolute_error_eV
            and self.continuum_stabilization_absolute_error_eV
            <= self.maximum_continuum_stabilization_absolute_error_eV
            and self.total_electrostatic_absolute_error_eV
            <= self.maximum_total_electrostatic_absolute_error_eV
        )
        if self.case_passed is not expected:
            raise ValueError("case_passed must be derived from all component budgets.")

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class OperationalLedgerAssessmentReport:
    assessment_id: str
    preregistration_sha256: str
    reference_panel_sha256: str
    source_gate_panel_sha256: str
    expected_case_ids: tuple[str, ...]
    ledger_priority: tuple[str, ...]
    prediction_sha256s: tuple[str, ...]
    case_assessments: tuple[LedgerCaseAssessment, ...]
    passing_ledger_ids: tuple[str, ...]
    selected_ledger_id: str | None
    contract_version: str = LEDGER_ASSESSMENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "assessment_id", _text(self.assessment_id, name="assessment_id")
        )
        for name in (
            "preregistration_sha256",
            "reference_panel_sha256",
            "source_gate_panel_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.contract_version != LEDGER_ASSESSMENT_CONTRACT_VERSION:
            raise ValueError("ledger assessment report version is unsupported.")
        expected_case_ids = _sorted_unique_text(
            self.expected_case_ids, name="expected_case_ids"
        )
        ledger_priority = _ordered_unique_text(
            self.ledger_priority, name="ledger_priority"
        )
        object.__setattr__(self, "expected_case_ids", expected_case_ids)
        object.__setattr__(self, "ledger_priority", ledger_priority)
        try:
            prediction_sha256s = tuple(
                _digest(value, name="prediction_sha256s")
                for value in self.prediction_sha256s
            )
        except TypeError as exc:
            raise TypeError(
                "prediction_sha256s must be an iterable of SHA256 digests."
            ) from exc
        if not prediction_sha256s or prediction_sha256s != tuple(
            sorted(set(prediction_sha256s))
        ):
            raise ValueError(
                "prediction_sha256s must be non-empty, unique, and sorted."
            )
        object.__setattr__(self, "prediction_sha256s", prediction_sha256s)
        cases = tuple(self.case_assessments)
        if not cases or not all(
            isinstance(case, LedgerCaseAssessment) for case in cases
        ):
            raise TypeError("case_assessments must contain typed case results.")
        if len({(case.ledger_id, case.case_id) for case in cases}) != len(cases):
            raise ValueError("ledger case assessment identities must be unique.")
        canonical_cases = tuple(
            sorted(cases, key=lambda case: (case.ledger_id, case.case_id))
        )
        if cases != canonical_cases:
            raise ValueError("case_assessments must use canonical ledger/case order.")
        object.__setattr__(self, "case_assessments", cases)
        passing = (
            _ordered_unique_text(self.passing_ledger_ids, name="passing_ledger_ids")
            if self.passing_ledger_ids
            else ()
        )
        case_ids_by_ledger: dict[str, set[str]] = {}
        for case in cases:
            case_ids_by_ledger.setdefault(case.ledger_id, set()).add(case.case_id)
        frozen_case_sets = {frozenset(value) for value in case_ids_by_ledger.values()}
        if frozen_case_sets != {frozenset(expected_case_ids)}:
            raise ValueError("every reported ledger must contain the frozen case set.")
        if set(case_ids_by_ledger) != set(ledger_priority):
            raise ValueError("reported ledgers differ from the frozen priority set.")
        derived_passing = {
            ledger_id
            for ledger_id in case_ids_by_ledger
            if all(case.case_passed for case in cases if case.ledger_id == ledger_id)
        }
        if set(passing) != derived_passing:
            raise ValueError(
                "passing_ledger_ids must be derived from all case results."
            )
        if passing != tuple(
            ledger_id for ledger_id in ledger_priority if ledger_id in derived_passing
        ):
            raise ValueError(
                "passing ledgers must retain preregistered priority order."
            )
        object.__setattr__(self, "passing_ledger_ids", passing)
        if self.selected_ledger_id is None:
            if passing:
                raise ValueError("a passing ledger requires a selected ledger ID.")
        else:
            selected = _text(self.selected_ledger_id, name="selected_ledger_id")
            if not passing or selected != passing[0]:
                raise ValueError("selected_ledger_id must be the first passing ledger.")
            object.__setattr__(self, "selected_ledger_id", selected)

    @property
    def content_sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "assessment_id": self.assessment_id,
            "preregistration_sha256": self.preregistration_sha256,
            "reference_panel_sha256": self.reference_panel_sha256,
            "source_gate_panel_sha256": self.source_gate_panel_sha256,
            "expected_case_ids": list(self.expected_case_ids),
            "ledger_priority": list(self.ledger_priority),
            "prediction_sha256s": list(self.prediction_sha256s),
            "case_assessments": [case.as_dict() for case in self.case_assessments],
            "passing_ledger_ids": list(self.passing_ledger_ids),
            "selected_ledger_id": self.selected_ledger_id,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
            "capability_admitted": False,
            "claim_boundary": (
                "Selection identifies at most the preregistered ledger allowed to "
                "advance to force/PES gates; it does not admit E/F/H/V/M."
            ),
        }


def assess_operational_electrostatic_ledgers(
    *,
    preregistration: LedgerAssessmentPreregistration,
    reference_panel: MatchedElectrostaticReferencePanel,
    source_gate_panel: QuantitativeSourceGatePanel,
    predictions: tuple[OperationalElectrostaticLedgerPrediction, ...],
) -> OperationalLedgerAssessmentReport:
    """Assess a frozen candidate matrix after, and only after, source passage."""

    if not isinstance(preregistration, LedgerAssessmentPreregistration):
        raise TypeError("preregistration must be a ledger assessment plan.")
    if not isinstance(reference_panel, MatchedElectrostaticReferencePanel):
        raise TypeError("reference_panel must be a matched reference panel.")
    if not isinstance(source_gate_panel, QuantitativeSourceGatePanel):
        raise TypeError("source_gate_panel must be a quantitative source panel.")
    if preregistration.reference_panel_sha256 != reference_panel.content_sha256:
        raise ValueError("ledger plan does not match the reference panel.")
    if preregistration.expected_case_ids != reference_panel.expected_case_ids:
        raise ValueError("ledger plan does not match the frozen case set.")
    if (
        source_gate_panel.reference_panel.content_sha256
        != reference_panel.content_sha256
    ):
        raise ValueError("source gate and ledger use different reference panels.")
    if (
        source_gate_panel.source_gate_protocol_sha256
        != preregistration.source_gate_protocol_sha256
    ):
        raise ValueError("source gate protocol does not match the ledger plan.")
    if not source_gate_panel.passed:
        raise ValueError("source gate failed; ledger assessment is forbidden.")

    records = tuple(predictions)
    if not records or not all(
        isinstance(record, OperationalElectrostaticLedgerPrediction)
        for record in records
    ):
        raise TypeError("predictions must contain typed ledger predictions.")
    key_set = {(record.ledger_id, record.case_id) for record in records}
    expected_keys = {
        (ledger_id, case_id)
        for ledger_id in preregistration.ledger_priority
        for case_id in preregistration.expected_case_ids
    }
    if len(key_set) != len(records) or key_set != expected_keys:
        raise ValueError(
            "ledger predictions must be the exact preregistered ledger/case matrix."
        )
    references = {case.case_id: case for case in reference_panel.cases}
    assessments: list[LedgerCaseAssessment] = []
    for record in sorted(records, key=lambda item: (item.ledger_id, item.case_id)):
        reference = references[record.case_id]
        if record.source_gate_panel_sha256 != source_gate_panel.content_sha256:
            raise ValueError("ledger prediction is not bound to this source gate.")
        expected_identity = {
            "reference_panel_sha256": reference_panel.content_sha256,
            "geometry_sha256": reference.geometry_sha256,
            "continuum_equation_id": reference.continuum_equation_id,
            "continuum_protocol_sha256": reference.continuum_protocol_sha256,
            "continuum_configuration_sha256": (
                reference.continuum_configuration_sha256
            ),
            "continuum_provenance_sha256": (reference.continuum_provenance_sha256),
            "topology_sha256": reference.topology_sha256,
            "cavity_profile_id": reference.cavity_profile_id,
            "dielectric": reference.dielectric,
        }
        for field, wanted in expected_identity.items():
            if getattr(record, field) != wanted:
                raise ValueError(
                    "ledger prediction differs from matched reference " f"{field}."
                )
        distortion_error = abs(
            record.solute_distortion_energy_eV - reference.solute_distortion_energy_eV
        )
        continuum_error = abs(
            record.continuum_stabilization_energy_eV
            - reference.continuum_stabilization_energy_eV
        )
        total_error = abs(
            record.total_electrostatic_solvation_energy_eV
            - reference.total_electrostatic_solvation_energy_eV
        )
        passed = bool(
            distortion_error
            <= preregistration.maximum_solute_distortion_absolute_error_eV
            and continuum_error
            <= preregistration.maximum_continuum_stabilization_absolute_error_eV
            and total_error
            <= preregistration.maximum_total_electrostatic_absolute_error_eV
        )
        assessments.append(
            LedgerCaseAssessment(
                case_id=record.case_id,
                ledger_id=record.ledger_id,
                solute_distortion_absolute_error_eV=distortion_error,
                continuum_stabilization_absolute_error_eV=continuum_error,
                total_electrostatic_absolute_error_eV=total_error,
                maximum_solute_distortion_absolute_error_eV=(
                    preregistration.maximum_solute_distortion_absolute_error_eV
                ),
                maximum_continuum_stabilization_absolute_error_eV=(
                    preregistration.maximum_continuum_stabilization_absolute_error_eV
                ),
                maximum_total_electrostatic_absolute_error_eV=(
                    preregistration.maximum_total_electrostatic_absolute_error_eV
                ),
                case_passed=passed,
            )
        )
    passing = tuple(
        ledger_id
        for ledger_id in preregistration.ledger_priority
        if all(
            assessment.case_passed
            for assessment in assessments
            if assessment.ledger_id == ledger_id
        )
    )
    selected = passing[0] if passing else None
    return OperationalLedgerAssessmentReport(
        assessment_id=preregistration.assessment_id,
        preregistration_sha256=preregistration.content_sha256,
        reference_panel_sha256=reference_panel.content_sha256,
        source_gate_panel_sha256=source_gate_panel.content_sha256,
        expected_case_ids=preregistration.expected_case_ids,
        ledger_priority=preregistration.ledger_priority,
        prediction_sha256s=tuple(sorted(record.content_sha256 for record in records)),
        case_assessments=tuple(assessments),
        passing_ledger_ids=passing,
        selected_ledger_id=selected,
    )


__all__ = [
    "LEDGER_ASSESSMENT_CONTRACT_VERSION",
    "LEDGER_ASSESSMENT_PREREGISTRATION_CONTRACT_VERSION",
    "LEDGER_PREDICTION_CONTRACT_VERSION",
    "LEDGER_SELECTION_RULE",
    "MATCHED_ELECTROSTATIC_PANEL_CONTRACT_VERSION",
    "MATCHED_ELECTROSTATIC_REFERENCE_CONTRACT_VERSION",
    "SOURCE_GATE_PANEL_CONTRACT_VERSION",
    "SOURCE_PHYSICAL_CASE_CONTRACT_VERSION",
    "LedgerAssessmentPreregistration",
    "LedgerCaseAssessment",
    "MatchedElectrostaticReferenceCase",
    "MatchedElectrostaticReferencePanel",
    "OperationalElectrostaticLedgerPrediction",
    "OperationalLedgerAssessmentReport",
    "QuantitativeSourceCaseEvidence",
    "QuantitativeSourceGatePanel",
    "assess_operational_electrostatic_ledgers",
]
