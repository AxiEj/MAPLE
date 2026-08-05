"""Fail-closed P1 evidence for operational Route-2 MLIP/continuum runs.

P1 is an accounting and residual audit.  It does not turn a response fixed
point into a stationary electronic functional and it does not admit forces.
The module deliberately distinguishes a backend-requested solver tolerance
from an observed algebraic residual; only the latter can satisfy the inner
continuum-residual gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from ase.units import Hartree

from ....route2_energy_ledger import (
    PCM_HALF_COUPLING_ONLY_V1,
    validate_route2_electrostatic_energy_ledger,
)
from .result import Route2EnergyLedger
from .route2_field_state import LocalReactionField
from .route2_plugin_contracts import validate_plugin_source_provider

P1_AUDIT_VERSION = "route2-operational-ledger-and-residual-audit-v1"
RESPONSE_CONDITIONED_OPERATIONAL_PREDICTION = (
    "response-conditioned-operational-prediction"
)
FIELD_CONJUGATE_OPERATIONAL_PREDICTION = "field-conjugate-operational-prediction"

ContinuumResidualStatus = Literal[
    "residual-certified",
    "tolerance-only",
    "unattested",
]
ConjugacyStatus = Literal["passed", "failed", "unattested"]


def _nonempty(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_or_none(value: object, *, name: str) -> float | None:
    return None if value is None else _finite(value, name=name)


def _freeze_json_value(value: object, *, name: str) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _nonempty(key, name=f"{name} key"): _freeze_json_value(
                    item,
                    name=f"{name}.{key}",
                )
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, np.generic):
        return _freeze_json_value(value.item(), name=name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value, name=name)
    raise TypeError(f"{name} contains unsupported value {type(value).__name__}.")


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ContinuumSolveEvidence:
    """Observed evidence for one inner continuum linear solve."""

    backend: str
    status: ContinuumResidualStatus
    requested_tolerance: float | None
    actual_residual: float | None
    residual_definition: str
    solve_completed_without_exception: bool
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _nonempty(self.backend, name="backend"))
        object.__setattr__(
            self,
            "residual_definition",
            _nonempty(self.residual_definition, name="residual_definition"),
        )
        if self.status not in {
            "residual-certified",
            "tolerance-only",
            "unattested",
        }:
            raise ValueError("Unsupported continuum residual evidence status.")
        if not isinstance(self.solve_completed_without_exception, bool):
            raise TypeError("continuum solve completion flag must be bool.")
        tolerance = _finite_or_none(
            self.requested_tolerance,
            name="requested continuum tolerance",
        )
        residual = _finite_or_none(
            self.actual_residual,
            name="actual continuum residual",
        )
        if tolerance is not None and tolerance <= 0.0:
            raise ValueError("requested continuum tolerance must be positive.")
        if residual is not None and residual < 0.0:
            raise ValueError("actual continuum residual must be non-negative.")
        if self.status == "residual-certified":
            if tolerance is None or residual is None:
                raise ValueError(
                    "residual-certified continuum evidence requires a tolerance "
                    "and an actual residual."
                )
        elif self.status == "tolerance-only":
            if tolerance is None or residual is not None:
                raise ValueError(
                    "tolerance-only continuum evidence requires only the requested "
                    "tolerance."
                )
        elif tolerance is not None or residual is not None:
            raise ValueError(
                "unattested continuum evidence cannot claim residual data."
            )
        object.__setattr__(self, "requested_tolerance", tolerance)
        object.__setattr__(self, "actual_residual", residual)
        object.__setattr__(
            self,
            "provenance",
            _freeze_json_value(self.provenance, name="continuum provenance"),
        )

    @property
    def residual_gate_passed(self) -> bool:
        return bool(
            self.status == "residual-certified"
            and self.solve_completed_without_exception
            and self.actual_residual is not None
            and self.requested_tolerance is not None
            and self.actual_residual <= self.requested_tolerance
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "status": self.status,
            "requested_tolerance": self.requested_tolerance,
            "actual_residual": self.actual_residual,
            "residual_definition": self.residual_definition,
            "solve_completed_without_exception": (
                self.solve_completed_without_exception
            ),
            "residual_gate_passed": self.residual_gate_passed,
            "provenance": _thaw_json_value(self.provenance),
        }


def classify_continuum_solve_evidence(reaction_field: object) -> ContinuumSolveEvidence:
    """Read an explicit residual certificate or classify the provider honestly.

    A provider may expose ``route2_continuum_solve_evidence`` as an already
    validated :class:`ContinuumSolveEvidence`.  Otherwise public runtime
    provenance is inspected only for the requested ``solver_tolerance``.  No
    hidden linear operator is reconstructed to manufacture a residual.
    """

    explicit = getattr(reaction_field, "route2_continuum_solve_evidence", None)
    if explicit is not None:
        evidence = explicit() if callable(explicit) else explicit
        if not isinstance(evidence, ContinuumSolveEvidence):
            raise TypeError(
                "route2_continuum_solve_evidence must be a "
                "ContinuumSolveEvidence instance."
            )
        return evidence

    raw_provenance = getattr(reaction_field, "runtime_provenance", {})
    provenance = raw_provenance if isinstance(raw_provenance, Mapping) else {}
    backend = str(
        provenance.get("backend")
        or provenance.get("provider")
        or provenance.get("method")
        or f"{type(reaction_field).__module__}.{type(reaction_field).__qualname__}"
    )
    requested = provenance.get("solver_tolerance")
    if requested is None:
        return ContinuumSolveEvidence(
            backend=backend,
            status="unattested",
            requested_tolerance=None,
            actual_residual=None,
            residual_definition=(
                "provider exposes neither an algebraic residual nor a requested "
                "residual tolerance"
            ),
            solve_completed_without_exception=True,
            provenance=provenance,
        )
    return ContinuumSolveEvidence(
        backend=backend,
        status="tolerance-only",
        requested_tolerance=_finite(requested, name="provider solver_tolerance"),
        actual_residual=None,
        residual_definition=(
            "backend algebraic residual is not exposed by the public provider "
            "contract"
        ),
        solve_completed_without_exception=True,
        provenance=provenance,
    )


@dataclass(frozen=True)
class EnergySourceConjugacyEvidence:
    """Finite-field check that model energy and reported source are conjugate."""

    status: ConjugacyStatus
    method: str
    finite_difference_step: float | None = None
    finite_difference_derivative_ev: float | None = None
    source_pairing_derivative_ev: float | None = None
    absolute_error_ev: float | None = None
    relative_error: float | None = None
    absolute_tolerance_ev: float | None = None
    relative_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "unattested"}:
            raise ValueError("Unsupported energy/source conjugacy status.")
        object.__setattr__(self, "method", _nonempty(self.method, name="method"))
        numeric_names = (
            "finite_difference_step",
            "finite_difference_derivative_ev",
            "source_pairing_derivative_ev",
            "absolute_error_ev",
            "relative_error",
            "absolute_tolerance_ev",
            "relative_tolerance",
        )
        values = {
            name: _finite_or_none(getattr(self, name), name=name)
            for name in numeric_names
        }
        if self.status == "unattested":
            if any(value is not None for value in values.values()):
                raise ValueError(
                    "unattested conjugacy evidence cannot contain results."
                )
        elif any(value is None for value in values.values()):
            raise ValueError("tested conjugacy evidence requires all numerical fields.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if self.finite_difference_step is not None and self.finite_difference_step <= 0:
            raise ValueError("finite-difference step must be positive.")
        for name in ("absolute_error_ev", "relative_error"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        for name in ("absolute_tolerance_ev", "relative_tolerance"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

    @classmethod
    def unattested(cls, *, method: str) -> "EnergySourceConjugacyEvidence":
        return cls(status="unattested", method=method)

    def as_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "status",
                "method",
                "finite_difference_step",
                "finite_difference_derivative_ev",
                "source_pairing_derivative_ev",
                "absolute_error_ev",
                "relative_error",
                "absolute_tolerance_ev",
                "relative_tolerance",
            )
        }


def audit_plugin_energy_source_conjugacy(
    plugin: object,
    atoms: Any,
    field: LocalReactionField,
    direction: np.ndarray,
    *,
    finite_difference_step: float = 1.0e-4,
    absolute_tolerance_ev: float = 1.0e-7,
    relative_tolerance: float = 1.0e-5,
) -> EnergySourceConjugacyEvidence:
    """Compare central ``dE/df`` with the declared source/field pairing."""

    provider = validate_plugin_source_provider(plugin)
    if not isinstance(field, LocalReactionField):
        raise TypeError("energy/source audit requires LocalReactionField.")
    atom_count = len(atoms)
    base = provider.field_dual_space.validate_field(
        field.as_nodewise_jet(),
        atom_count=atom_count,
        name="conjugacy base field",
    )
    probe = provider.field_dual_space.validate_field(
        direction,
        atom_count=atom_count,
        name="conjugacy field direction",
    )
    step = _finite(finite_difference_step, name="finite_difference_step")
    absolute_tolerance = _finite(
        absolute_tolerance_ev,
        name="absolute_tolerance_ev",
    )
    relative_tolerance_value = _finite(
        relative_tolerance,
        name="relative_tolerance",
    )
    if step <= 0.0 or absolute_tolerance < 0.0 or relative_tolerance_value < 0.0:
        raise ValueError("conjugacy step/tolerances are outside their domains.")

    def evaluate(values: np.ndarray):
        state, _ = provider.evaluate_source_state(
            atoms,
            LocalReactionField.from_nodewise_jet(values),
            compute_forces=False,
        )
        return state

    centre = evaluate(base)
    plus = evaluate(base + step * probe)
    minus = evaluate(base - step * probe)
    finite_difference = (plus.energy_ev - minus.energy_ev) / (2.0 * step)
    source_pairing = provider.field_dual_space.pair(
        centre.source,
        probe,
        source_space=provider.source_space,
        atom_count=atom_count,
    )
    absolute_error = abs(finite_difference - source_pairing)
    scale = max(abs(finite_difference), abs(source_pairing), np.finfo(float).tiny)
    relative_error_value = absolute_error / scale
    passed = (
        absolute_error <= absolute_tolerance
        or relative_error_value <= relative_tolerance_value
    )
    return EnergySourceConjugacyEvidence(
        status="passed" if passed else "failed",
        method="central-finite-field-energy-vs-source-pairing-v1",
        finite_difference_step=step,
        finite_difference_derivative_ev=finite_difference,
        source_pairing_derivative_ev=source_pairing,
        absolute_error_ev=absolute_error,
        relative_error=relative_error_value,
        absolute_tolerance_ev=absolute_tolerance,
        relative_tolerance=relative_tolerance_value,
    )


@dataclass(frozen=True)
class OuterSCFEvidence:
    """Dimensioned outer fixed-point residuals and their admission result."""

    response_mode: str
    convergence_reason: str
    metrics: Mapping[str, object]
    tolerances: Mapping[str, object]
    passed: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "response_mode",
            _nonempty(self.response_mode, name="response_mode"),
        )
        object.__setattr__(
            self,
            "convergence_reason",
            _nonempty(self.convergence_reason, name="convergence_reason"),
        )
        object.__setattr__(
            self,
            "metrics",
            _freeze_json_value(self.metrics, name="outer SCF metrics"),
        )
        object.__setattr__(
            self,
            "tolerances",
            _freeze_json_value(self.tolerances, name="outer SCF tolerances"),
        )
        blockers = tuple(
            _nonempty(item, name="outer SCF blocker") for item in self.blockers
        )
        if self.passed != (len(blockers) == 0):
            raise ValueError("outer SCF pass flag and blockers disagree.")
        object.__setattr__(self, "blockers", blockers)

    def as_dict(self) -> dict[str, object]:
        return {
            "response_mode": self.response_mode,
            "convergence_reason": self.convergence_reason,
            "metrics": _thaw_json_value(self.metrics),
            "tolerances": _thaw_json_value(self.tolerances),
            "passed": self.passed,
            "blockers": list(self.blockers),
        }


def audit_outer_scf(coupled_state: object) -> OuterSCFEvidence:
    """Re-evaluate the strict nominal outer residual gate from stored evidence."""

    response_mode = str(getattr(coupled_state, "response_mode", "unattested"))
    convergence = getattr(coupled_state, "scf_convergence", {})
    convergence = convergence if isinstance(convergence, Mapping) else {}
    reason = str(convergence.get("reason", "unattested"))
    history = tuple(getattr(coupled_state, "history", ()))
    blockers: list[str] = []
    if response_mode != "scf":
        blockers.append("outer-fixed-point-not-applicable")
    if not history:
        blockers.append("outer-fixed-point-history-missing")
        return OuterSCFEvidence(
            response_mode=response_mode,
            convergence_reason=reason,
            metrics={},
            tolerances={},
            passed=False,
            blockers=tuple(blockers),
        )

    final = history[-1]
    tolerance_mapping = convergence.get("residual_tolerances", {})
    tolerances = tolerance_mapping if isinstance(tolerance_mapping, Mapping) else {}
    metrics = {
        "monopole_max_e": final.get("monopole_residual_e"),
        "monopole_rms_e": final.get("monopole_residual_rms_e"),
        "dipole_component_max_e_angstrom": final.get("dipole_residual_e_angstrom"),
        "dipole_component_rms_e_angstrom": final.get("dipole_residual_rms_e_angstrom"),
        "source_rms_normalized": final.get("source_residual_rms_normalized"),
        "raw_response_charge_delta_e": final.get("raw_response_charge_delta_e"),
        "projected_total_charge_residual_e": final.get(
            "projected_total_charge_residual_e"
        ),
        "molecular_dipole_vector_e_angstrom": final.get(
            "molecular_dipole_residual_vector_e_angstrom"
        ),
        "molecular_dipole_l2_e_angstrom": final.get(
            "molecular_dipole_residual_l2_e_angstrom"
        ),
        "energy_delta_ev": final.get("energy_residual_ev"),
    }
    if reason != "nominal-density-and-energy-v1":
        blockers.append("outer-convergence-is-not-strict-nominal")
    if convergence.get("nominal_residual_gate_passed") is not True:
        blockers.append("outer-nominal-residual-gate-not-passed")

    comparisons = (
        ("monopole_max_e", "monopole_max_e", False),
        (
            "dipole_component_max_e_angstrom",
            "dipole_component_max_e_angstrom",
            False,
        ),
        ("source_rms_normalized", "source_rms_normalized", False),
        (
            "raw_response_charge_delta_e",
            "raw_response_charge_delta_e",
            True,
        ),
        (
            "projected_total_charge_residual_e",
            "projected_total_charge_residual_e",
            False,
        ),
        (
            "molecular_dipole_l2_e_angstrom",
            "molecular_dipole_l2_e_angstrom",
            False,
        ),
        ("energy_delta_ev", "energy_delta_ev", False),
    )
    for metric_name, tolerance_name, use_absolute in comparisons:
        raw_metric = metrics.get(metric_name)
        raw_tolerance = tolerances.get(tolerance_name)
        if raw_metric is None:
            blockers.append(f"outer-{metric_name}-unobserved")
            continue
        if raw_tolerance is None:
            blockers.append(f"outer-{tolerance_name}-tolerance-unattested")
            continue
        metric = _finite(raw_metric, name=metric_name)
        tolerance = _finite(raw_tolerance, name=tolerance_name)
        compared = abs(metric) if use_absolute else metric
        if tolerance < 0.0 or compared > tolerance:
            blockers.append(f"outer-{metric_name}-exceeds-tolerance")
    if (
        final.get("accepted") is not True
        or final.get("next_density_update") != "converged"
    ):
        blockers.append("outer-final-state-not-accepted-converged-root")
    blockers = list(dict.fromkeys(blockers))
    return OuterSCFEvidence(
        response_mode=response_mode,
        convergence_reason=reason,
        metrics=metrics,
        tolerances=tolerances,
        passed=not blockers,
        blockers=tuple(blockers),
    )


@dataclass(frozen=True)
class Route2OperationalEnergyAudit:
    """Complete scalar accounting for one response-conditioned P1 state."""

    audit_version: str
    electronic_model_family: str
    electronic_energy_semantics: str
    energy_interpretation: str
    selected_energy_ledger: str
    model_zero_field_energy_ev: float
    model_fixed_field_energy_ev: float
    model_energy_change_ev: float
    source_field_pairing_ev: float
    pcm_half_coupling_ev: float
    pcm_provider_energy_ev: float
    pairing_identity_error_ev: float
    pairing_identity_tolerance_ev: float
    cds_energy_ev: float
    standard_state_energy_ev: float
    final_scalar_ev: float
    energy_ledger_hartree: Route2EnergyLedger
    energy_source_conjugacy: EnergySourceConjugacyEvidence
    outer_scf: OuterSCFEvidence
    continuum_solve: ContinuumSolveEvidence
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "audit_version",
            "electronic_model_family",
            "electronic_energy_semantics",
            "energy_interpretation",
            "selected_energy_ledger",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        for name in (
            "model_zero_field_energy_ev",
            "model_fixed_field_energy_ev",
            "model_energy_change_ev",
            "source_field_pairing_ev",
            "pcm_half_coupling_ev",
            "pcm_provider_energy_ev",
            "pairing_identity_error_ev",
            "pairing_identity_tolerance_ev",
            "cds_energy_ev",
            "standard_state_energy_ev",
            "final_scalar_ev",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.pairing_identity_error_ev < 0.0:
            raise ValueError("pairing identity error must be non-negative.")
        if self.pairing_identity_tolerance_ev <= 0.0:
            raise ValueError("pairing identity tolerance must be positive.")
        if not isinstance(self.energy_ledger_hartree, Route2EnergyLedger):
            raise TypeError("P1 audit requires a checked Route2EnergyLedger.")
        if not isinstance(self.energy_source_conjugacy, EnergySourceConjugacyEvidence):
            raise TypeError("P1 audit requires energy/source conjugacy evidence.")
        if not isinstance(self.outer_scf, OuterSCFEvidence):
            raise TypeError("P1 audit requires outer SCF evidence.")
        if not isinstance(self.continuum_solve, ContinuumSolveEvidence):
            raise TypeError("P1 audit requires continuum solve evidence.")
        selected = validate_route2_electrostatic_energy_ledger(
            self.selected_energy_ledger
        )
        if not np.isclose(
            self.model_energy_change_ev,
            self.model_fixed_field_energy_ev - self.model_zero_field_energy_ev,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("P1 model-energy delta does not close.")
        if not np.isclose(
            self.pcm_half_coupling_ev,
            0.5 * self.source_field_pairing_ev,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("P1 full pairing and PCM half coupling do not close.")
        if not np.isclose(
            self.pairing_identity_error_ev,
            abs(self.pcm_half_coupling_ev - self.pcm_provider_energy_ev),
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("P1 PCM pairing identity error is inconsistent.")
        leaves = self.energy_ledger_hartree.leaf_components_hartree
        expected_solute = (
            0.0
            if selected == PCM_HALF_COUPLING_ONLY_V1
            else self.model_energy_change_ev / Hartree
        )
        ledger_checks = {
            "solute_polarization": expected_solute,
            "pcm_polarization": self.pcm_provider_energy_ev / Hartree,
            "cds": self.cds_energy_ev / Hartree,
            "standard_state": self.standard_state_energy_ev / Hartree,
        }
        if any(
            not np.isclose(
                leaves[name],
                expected,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            for name, expected in ledger_checks.items()
        ):
            raise ValueError("P1 checked energy ledger disagrees with scalar terms.")
        if not np.isclose(
            self.final_scalar_ev,
            self.energy_ledger_hartree.derived_totals_hartree["delta_g_solv"] * Hartree,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("P1 final scalar does not close against the ledger.")
        blockers = tuple(_nonempty(item, name="P1 blocker") for item in self.blockers)
        object.__setattr__(self, "blockers", blockers)

    @property
    def p1_complete(self) -> bool:
        return bool(
            len(self.blockers) == 0
            and self.outer_scf.passed
            and self.continuum_solve.residual_gate_passed
            and self.energy_source_conjugacy.status != "unattested"
            and self.pairing_identity_error_ev <= self.pairing_identity_tolerance_ev
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "audit_version": self.audit_version,
            "p1_complete": self.p1_complete,
            "blockers": list(self.blockers),
            "electronic_model_family": self.electronic_model_family,
            "electronic_energy_semantics": self.electronic_energy_semantics,
            "energy_interpretation": self.energy_interpretation,
            "selected_energy_ledger": self.selected_energy_ledger,
            "model_zero_field_energy_ev": self.model_zero_field_energy_ev,
            "model_fixed_field_energy_ev": self.model_fixed_field_energy_ev,
            "model_energy_change_ev": self.model_energy_change_ev,
            "source_field_pairing_ev": self.source_field_pairing_ev,
            "pcm_half_coupling_ev": self.pcm_half_coupling_ev,
            "pcm_provider_energy_ev": self.pcm_provider_energy_ev,
            "pairing_identity_error_ev": self.pairing_identity_error_ev,
            "pairing_identity_tolerance_ev": self.pairing_identity_tolerance_ev,
            "cds_energy_ev": self.cds_energy_ev,
            "standard_state_energy_ev": self.standard_state_energy_ev,
            "final_scalar_ev": self.final_scalar_ev,
            "energy_ledger_hartree": dict(
                self.energy_ledger_hartree.components_hartree
            ),
            "energy_source_conjugacy": self.energy_source_conjugacy.as_dict(),
            "outer_scf": self.outer_scf.as_dict(),
            "continuum_solve": self.continuum_solve.as_dict(),
        }


def build_operational_energy_audit(
    gas_state: object,
    coupled_state: object,
    *,
    electrostatic_energy_ledger: str = PCM_HALF_COUPLING_ONLY_V1,
    standard_state_energy_hartree: float = 0.0,
    energy_source_conjugacy: EnergySourceConjugacyEvidence | None = None,
    continuum_solve: ContinuumSolveEvidence | None = None,
    pairing_identity_tolerance_ev: float = 1.0e-10,
) -> Route2OperationalEnergyAudit:
    """Build and independently close the P1 operational scalar ledger."""

    selected = validate_route2_electrostatic_energy_ledger(electrostatic_energy_ledger)
    zero_field_energy_ev = _finite(
        getattr(gas_state, "energy_ev"),
        name="zero-field model energy",
    )
    solvent_state = getattr(coupled_state, "solvent_state")
    fixed_field_energy_ev = _finite(
        getattr(solvent_state, "energy_ev"),
        name="fixed-field model energy",
    )
    model_energy_change_ev = fixed_field_energy_ev - zero_field_energy_ev
    source_field_pairing_ev = _finite(
        getattr(coupled_state, "source_field_pairing_ev"),
        name="full source/field pairing",
    )
    pcm_half_coupling_ev = 0.5 * source_field_pairing_ev
    pcm_provider_energy_ev = _finite(
        getattr(coupled_state, "polarization_energy_hartree") * Hartree,
        name="provider PCM energy",
    )
    identity_error_ev = abs(pcm_half_coupling_ev - pcm_provider_energy_ev)
    stored_identity_error_ev = _finite(
        getattr(coupled_state, "energy_identity_error_ev"),
        name="stored pairing identity error",
    )
    if not np.isclose(
        identity_error_ev,
        stored_identity_error_ev,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("coupled-state pairing identity evidence is inconsistent.")
    identity_tolerance_ev = _finite(
        pairing_identity_tolerance_ev,
        name="pairing identity tolerance",
    )
    if identity_tolerance_ev <= 0.0:
        raise ValueError("pairing identity tolerance must be positive.")

    cds_energy_hartree = _finite(
        getattr(getattr(coupled_state, "cds_result"), "energy_hartree"),
        name="CDS energy",
    )
    standard_state_hartree = _finite(
        standard_state_energy_hartree,
        name="standard-state energy",
    )
    solute_polarization_hartree = (
        0.0
        if selected == PCM_HALF_COUPLING_ONLY_V1
        else model_energy_change_ev / Hartree
    )
    ledger = Route2EnergyLedger.from_components(
        {
            "solute_polarization": solute_polarization_hartree,
            "pcm_polarization": pcm_provider_energy_ev / Hartree,
            "cds": cds_energy_hartree,
            "standard_state": standard_state_hartree,
        }
    )
    final_scalar_ev = ledger.derived_totals_hartree["delta_g_solv"] * Hartree
    conjugacy = energy_source_conjugacy or EnergySourceConjugacyEvidence.unattested(
        method="no finite-field energy/source audit supplied"
    )
    if not isinstance(conjugacy, EnergySourceConjugacyEvidence):
        raise TypeError("energy_source_conjugacy evidence is malformed.")
    interpretation = (
        FIELD_CONJUGATE_OPERATIONAL_PREDICTION
        if conjugacy.status == "passed"
        else RESPONSE_CONDITIONED_OPERATIONAL_PREDICTION
    )
    outer = audit_outer_scf(coupled_state)
    continuum = continuum_solve or classify_continuum_solve_evidence(
        getattr(coupled_state, "reaction_field")
    )
    if not isinstance(continuum, ContinuumSolveEvidence):
        raise TypeError("continuum_solve evidence is malformed.")

    blockers = list(outer.blockers)
    if not continuum.residual_gate_passed:
        blockers.append("inner-continuum-algebraic-residual-not-certified")
    if conjugacy.status == "unattested":
        blockers.append("energy-source-conjugacy-unattested")
    if identity_error_ev > identity_tolerance_ev:
        blockers.append("source-field-pcm-half-coupling-identity-failed")
    blockers = list(dict.fromkeys(blockers))
    return Route2OperationalEnergyAudit(
        audit_version=P1_AUDIT_VERSION,
        electronic_model_family=str(getattr(coupled_state, "electronic_model_family")),
        electronic_energy_semantics=str(
            getattr(coupled_state, "electronic_energy_semantics")
        ),
        energy_interpretation=interpretation,
        selected_energy_ledger=selected,
        model_zero_field_energy_ev=zero_field_energy_ev,
        model_fixed_field_energy_ev=fixed_field_energy_ev,
        model_energy_change_ev=model_energy_change_ev,
        source_field_pairing_ev=source_field_pairing_ev,
        pcm_half_coupling_ev=pcm_half_coupling_ev,
        pcm_provider_energy_ev=pcm_provider_energy_ev,
        pairing_identity_error_ev=identity_error_ev,
        pairing_identity_tolerance_ev=identity_tolerance_ev,
        cds_energy_ev=cds_energy_hartree * Hartree,
        standard_state_energy_ev=standard_state_hartree * Hartree,
        final_scalar_ev=final_scalar_ev,
        energy_ledger_hartree=ledger,
        energy_source_conjugacy=conjugacy,
        outer_scf=outer,
        continuum_solve=continuum,
        blockers=tuple(blockers),
    )


__all__ = [
    "FIELD_CONJUGATE_OPERATIONAL_PREDICTION",
    "P1_AUDIT_VERSION",
    "RESPONSE_CONDITIONED_OPERATIONAL_PREDICTION",
    "ContinuumSolveEvidence",
    "EnergySourceConjugacyEvidence",
    "OuterSCFEvidence",
    "Route2OperationalEnergyAudit",
    "audit_outer_scf",
    "audit_plugin_energy_source_conjugacy",
    "build_operational_energy_audit",
    "classify_continuum_solve_evidence",
]
