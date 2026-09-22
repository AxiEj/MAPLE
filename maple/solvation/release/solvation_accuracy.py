"""Fail-closed blind total-solvation accuracy gate for Route 2.

The release package records evidence but never enables a capability.  This
module freezes the user's 1.5 kcal/mol criterion as a total-solvation-free-
energy MAE gate on a content-addressed blind panel.  Electrostatic-only values,
partial panels, fake backends, and changed standard-state conventions are
schema errors rather than silently comparable measurements.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

SOLVATION_ACCURACY_CONTRACT_VERSION = "route2-blind-solvation-accuracy-v1"
SOLVATION_ACCURACY_MAE_THRESHOLD_KCAL_PER_MOL = 1.5
SOLVATION_ACCURACY_MINIMUM_INDEPENDENT_MOLECULES = 150
SOLVATION_ACCURACY_BOOTSTRAP_REPLICATES = 10_000
SOLVATION_ACCURACY_CONFIDENCE_LEVEL = 0.95
SOLVATION_ACCURACY_TARGET_KIND = "total-solvation-free-energy"
SOLVATION_ACCURACY_STANDARD_STATE = "1M-gas-to-1M-solution"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


@dataclass(frozen=True, slots=True)
class BlindSolvationAccuracyProtocol:
    """Content-addressed identities and immutable numerical gate settings."""

    protocol_id: str
    dataset_id: str
    dataset_sha256: str
    profile_configuration_sha256: str
    expected_case_ids: tuple[str, ...]
    solvent_id: str = "water"
    target_kind: str = SOLVATION_ACCURACY_TARGET_KIND
    standard_state: str = SOLVATION_ACCURACY_STANDARD_STATE
    mae_threshold_kcal_per_mol: float = SOLVATION_ACCURACY_MAE_THRESHOLD_KCAL_PER_MOL
    minimum_independent_molecules: int = (
        SOLVATION_ACCURACY_MINIMUM_INDEPENDENT_MOLECULES
    )
    bootstrap_replicates: int = SOLVATION_ACCURACY_BOOTSTRAP_REPLICATES
    confidence_level: float = SOLVATION_ACCURACY_CONFIDENCE_LEVEL
    bootstrap_seed: int = 20260815
    contract_version: str = SOLVATION_ACCURACY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("protocol_id", "dataset_id", "solvent_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in ("dataset_sha256", "profile_configuration_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        cases = tuple(
            _text(case_id, name="expected_case_id")
            for case_id in self.expected_case_ids
        )
        if len(cases) != len(set(cases)):
            raise ValueError("expected_case_ids must be unique.")
        if len(cases) < SOLVATION_ACCURACY_MINIMUM_INDEPENDENT_MOLECULES:
            raise ValueError(
                "blind accuracy protocol requires at least 150 independent cases."
            )
        object.__setattr__(self, "expected_case_ids", cases)
        if self.contract_version != SOLVATION_ACCURACY_CONTRACT_VERSION:
            raise ValueError("unsupported solvation accuracy contract version.")
        if self.target_kind != SOLVATION_ACCURACY_TARGET_KIND:
            raise ValueError("accuracy admission requires total solvation free energy.")
        if self.standard_state != SOLVATION_ACCURACY_STANDARD_STATE:
            raise ValueError("accuracy admission requires the frozen 1M-to-1M state.")
        if self.mae_threshold_kcal_per_mol != (
            SOLVATION_ACCURACY_MAE_THRESHOLD_KCAL_PER_MOL
        ):
            raise ValueError(
                "the preregistered MAE threshold must remain 1.5 kcal/mol."
            )
        if self.minimum_independent_molecules != (
            SOLVATION_ACCURACY_MINIMUM_INDEPENDENT_MOLECULES
        ):
            raise ValueError("the minimum independent-molecule count must remain 150.")
        if self.bootstrap_replicates != SOLVATION_ACCURACY_BOOTSTRAP_REPLICATES:
            raise ValueError("the bootstrap replicate count must remain 10000.")
        if self.confidence_level != SOLVATION_ACCURACY_CONFIDENCE_LEVEL:
            raise ValueError("the one-sided confidence level must remain 0.95.")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(
            self.bootstrap_seed, int
        ):
            raise TypeError("bootstrap_seed must be an integer.")

    def metadata(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "protocol_id": self.protocol_id,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "profile_configuration_sha256": self.profile_configuration_sha256,
            "expected_case_ids": list(self.expected_case_ids),
            "solvent_id": self.solvent_id,
            "target_kind": self.target_kind,
            "standard_state": self.standard_state,
            "mae_threshold_kcal_per_mol": self.mae_threshold_kcal_per_mol,
            "minimum_independent_molecules": self.minimum_independent_molecules,
            "bootstrap_replicates": self.bootstrap_replicates,
            "confidence_level": self.confidence_level,
            "bootstrap_seed": self.bootstrap_seed,
        }

    def configuration_sha256(self) -> str:
        encoded = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BlindSolvationAccuracyRecord:
    """One completed real-backend prediction on the frozen blind panel."""

    case_id: str
    molecule_group_id: str
    solvent_id: str
    predicted_kcal_per_mol: float
    reference_kcal_per_mol: float
    target_kind: str = SOLVATION_ACCURACY_TARGET_KIND
    standard_state: str = SOLVATION_ACCURACY_STANDARD_STATE
    backend_kind: str = "real"
    completed: bool = True

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "molecule_group_id",
            "solvent_id",
            "target_kind",
            "standard_state",
            "backend_kind",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        values = np.asarray(
            (self.predicted_kcal_per_mol, self.reference_kcal_per_mol), dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("predicted and reference energies must be finite.")
        if type(self.completed) is not bool:
            raise TypeError("completed must be exactly bool.")


@dataclass(frozen=True, slots=True)
class BlindSolvationAccuracyResult:
    protocol_configuration_sha256: str
    record_count: int
    molecule_group_count: int
    mean_error_kcal_per_mol: float
    mae_kcal_per_mol: float
    rmse_kcal_per_mol: float
    p90_absolute_error_kcal_per_mol: float
    maximum_absolute_error_kcal_per_mol: float
    one_sided_mae_ucb_kcal_per_mol: float
    mae_threshold_kcal_per_mol: float
    point_mae_gate_passed: bool
    confidence_bound_gate_passed: bool
    gate_passed: bool

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def assess_blind_solvation_accuracy(
    protocol: BlindSolvationAccuracyProtocol,
    records: tuple[BlindSolvationAccuracyRecord, ...],
) -> BlindSolvationAccuracyResult:
    """Recompute MAE and its one-sided molecule-bootstrap confidence bound."""

    if not isinstance(protocol, BlindSolvationAccuracyProtocol):
        raise TypeError("protocol must be BlindSolvationAccuracyProtocol.")
    if not isinstance(records, tuple) or not all(
        isinstance(record, BlindSolvationAccuracyRecord) for record in records
    ):
        raise TypeError("records must be a tuple of BlindSolvationAccuracyRecord.")
    by_case: dict[str, BlindSolvationAccuracyRecord] = {}
    for record in records:
        if record.case_id in by_case:
            raise ValueError("duplicate blind case record.")
        by_case[record.case_id] = record
    if set(by_case) != set(protocol.expected_case_ids):
        raise ValueError("records must exactly cover the preregistered blind panel.")
    ordered = tuple(by_case[case_id] for case_id in protocol.expected_case_ids)
    molecule_groups = tuple(record.molecule_group_id for record in ordered)
    if len(set(molecule_groups)) != len(molecule_groups):
        raise ValueError("blind water gate requires one record per molecule group.")
    for record in ordered:
        if not record.completed:
            raise ValueError("every in-domain blind record must be completed.")
        if record.backend_kind != "real":
            raise ValueError("blind admission requires a real backend.")
        if record.solvent_id != protocol.solvent_id:
            raise ValueError("record solvent differs from the frozen protocol.")
        if record.target_kind != protocol.target_kind:
            raise ValueError("electrostatic-only values cannot enter the total gate.")
        if record.standard_state != protocol.standard_state:
            raise ValueError("record standard state differs from the frozen protocol.")

    errors = np.asarray(
        [
            record.predicted_kcal_per_mol - record.reference_kcal_per_mol
            for record in ordered
        ],
        dtype=float,
    )
    absolute = np.abs(errors)
    rng = np.random.default_rng(protocol.bootstrap_seed)
    indices = rng.integers(
        0,
        len(absolute),
        size=(protocol.bootstrap_replicates, len(absolute)),
        endpoint=False,
    )
    bootstrap_mae = np.mean(absolute[indices], axis=1)
    one_sided_ucb = float(
        np.quantile(
            bootstrap_mae,
            protocol.confidence_level,
            method="higher",
        )
    )
    mae = float(np.mean(absolute))
    threshold = protocol.mae_threshold_kcal_per_mol
    point_passed = mae <= threshold
    ucb_passed = one_sided_ucb <= threshold
    return BlindSolvationAccuracyResult(
        protocol_configuration_sha256=protocol.configuration_sha256(),
        record_count=len(ordered),
        molecule_group_count=len(set(molecule_groups)),
        mean_error_kcal_per_mol=float(np.mean(errors)),
        mae_kcal_per_mol=mae,
        rmse_kcal_per_mol=float(np.sqrt(np.mean(errors**2))),
        p90_absolute_error_kcal_per_mol=float(
            np.quantile(absolute, 0.90, method="higher")
        ),
        maximum_absolute_error_kcal_per_mol=float(np.max(absolute)),
        one_sided_mae_ucb_kcal_per_mol=one_sided_ucb,
        mae_threshold_kcal_per_mol=threshold,
        point_mae_gate_passed=point_passed,
        confidence_bound_gate_passed=ucb_passed,
        gate_passed=point_passed and ucb_passed,
    )


__all__ = [
    "SOLVATION_ACCURACY_BOOTSTRAP_REPLICATES",
    "SOLVATION_ACCURACY_CONFIDENCE_LEVEL",
    "SOLVATION_ACCURACY_CONTRACT_VERSION",
    "SOLVATION_ACCURACY_MAE_THRESHOLD_KCAL_PER_MOL",
    "SOLVATION_ACCURACY_MINIMUM_INDEPENDENT_MOLECULES",
    "SOLVATION_ACCURACY_STANDARD_STATE",
    "SOLVATION_ACCURACY_TARGET_KIND",
    "BlindSolvationAccuracyProtocol",
    "BlindSolvationAccuracyRecord",
    "BlindSolvationAccuracyResult",
    "assess_blind_solvation_accuracy",
]
