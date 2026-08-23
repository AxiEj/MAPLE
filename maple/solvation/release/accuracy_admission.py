"""I/O-free contracts for leakage-safe Route-2 accuracy admission.

Prediction processes receive geometry/solvent inputs but no experimental
targets.  Mechanical scorers receive frozen prediction artifacts and labels
but import no model or continuum implementation.  This module validates the
data boundary and computes deterministic metrics; it performs no filesystem
I/O and never enables a capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from statistics import fmean
from typing import Mapping, Sequence

LABEL_FREE_INPUT_SCHEMA_ID = "maple-route2-label-free-accuracy-input-v1"
PREDICTION_TERMINAL_SCHEMA_ID = "maple-route2-accuracy-prediction-terminal-v1"
SCORED_BUNDLE_SCHEMA_ID = "maple-route2-private-accuracy-score-bundle-v1"
PUBLIC_ACCURACY_SCHEMA_ID = "maple-route2-public-accuracy-aggregate-v1"
PUBLIC_FAILURE_SCHEMA_ID = "maple-route2-public-accuracy-attempt-failure-v1"
CAPABILITIES_CLOSED = {tier: False for tier in ("E", "F", "H", "V", "M")}
EV_TO_KCAL_MOL = 23.06054783062068

_DIGEST_CHARACTERS = frozenset("0123456789abcdef")
_FORBIDDEN_LABEL_FREE_KEYS = frozenset(
    {
        "delta_g_kcal_mol",
        "experimental_delta_g_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
        "aggregate_metrics",
        "mean_absolute_error_kcal_mol",
        "root_mean_square_error_kcal_mol",
        "maximum_absolute_error_kcal_mol",
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "raw_row_sha256",
        "label",
        "labels",
        "target",
        "targets",
        "ground_truth",
        "reference_delta_g",
    }
)
_INPUT_RECORD_KEYS = frozenset(
    {
        "selection_index",
        "opaque_record_id",
        "canonical_solvent",
        "partition",
        "prior_pilot_geometry_overlap",
        "geometry_sha256",
        "normalized_geometry_sha256",
        "atom_count",
        "atomic_numbers",
        "positions_angstrom",
        "charge",
        "multiplicity",
    }
)
_PREDICTION_RECORD_KEYS = frozenset(
    {
        "selection_index",
        "opaque_record_id",
        "canonical_solvent",
        "partition",
        "prior_pilot_geometry_overlap",
        "geometry_sha256",
        "normalized_geometry_sha256",
        "atom_count",
        "solution_total_energy_eV",
        "vacuum_energy_eV",
        "predicted_delta_g_eV",
        "predicted_delta_g_kcal_mol",
        "polarization_kcal_mol",
        "cds_kcal_mol",
        "permanent_charge_e",
        "induced_charge_e",
        "combined_charge_e",
        "cold_actual_residual_norm",
        "wide_actual_residual_norm",
        "cold_iterations",
        "wide_iterations",
        "induced_source_max_abs_difference_e",
        "native_field_max_abs_difference_eV_per_e",
        "continuum_energy_abs_difference_eV",
        "gates",
        "configuration_sha256",
        "continuum_configuration_sha256",
        "continuum_provenance_sha256",
        "equation_sha256",
        "cold_root_sha256",
        "wide_root_sha256",
        "base_ledger_sha256",
        "total_ledger_sha256",
        "solvent_term_configuration_sha256",
        "state_sha256",
    }
)
_ROOT_GATE_KEYS = frozenset(
    {
        "cold_wide_source",
        "cold_wide_field",
        "cold_wide_energy",
        "actual_residuals",
        "permanent_charge",
        "induced_charge",
        "combined_charge",
    }
)
_SCORED_RECORD_KEYS = _PREDICTION_RECORD_KEYS | frozenset(
    {
        "experimental_delta_g_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
    }
)
_ACCURACY_GATE_KEYS = frozenset(
    {
        "complete_exact_frozen_panel",
        "ten_distinct_solvents",
        "all_cold_wide_roots_pass",
        "all_actual_root_residuals_pass",
        "all_permanent_and_induced_charge_checks_pass",
        "mae_at_most_1_5_kcal_mol",
    }
)
_MAXIMUM_ROOT_METRIC_KEYS = frozenset(
    {
        "actual_root_residual_norm",
        "induced_source_difference_e",
        "native_field_difference_eV_per_e",
        "continuum_energy_difference_eV",
        "permanent_charge_error_e",
        "induced_charge_error_e",
        "combined_charge_error_e",
    }
)


class AccuracyContractError(ValueError):
    """Raised when an accuracy artifact crosses or weakens a contract."""


@dataclass(frozen=True, slots=True)
class PredictionValidationContract:
    maximum_actual_residual_norm: float
    maximum_induced_source_absolute_difference_e: float
    maximum_native_field_absolute_difference_eV_per_e: float
    maximum_continuum_energy_absolute_difference_eV: float
    maximum_permanent_charge_error_e: float
    maximum_induced_charge_error_e: float
    maximum_combined_charge_error_e: float
    energy_closure_tolerance_eV: float = 1.0e-12
    kcal_closure_tolerance: float = 1.0e-10
    charge_closure_tolerance_e: float = 1.0e-15


HYBRID_MNSOL_PREDICTION_VALIDATION = PredictionValidationContract(
    maximum_actual_residual_norm=1.0e-10,
    maximum_induced_source_absolute_difference_e=2.0e-9,
    maximum_native_field_absolute_difference_eV_per_e=2.0e-9,
    maximum_continuum_energy_absolute_difference_eV=1.0e-9,
    maximum_permanent_charge_error_e=1.0e-8,
    maximum_induced_charge_error_e=1.0e-8,
    maximum_combined_charge_error_e=1.0e-8,
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalized_geometry_sha256(
    *,
    atomic_numbers: Sequence[object],
    positions_angstrom: Sequence[Sequence[object]],
    charge: object,
    multiplicity: object,
) -> str:
    """Hash the exact normalized geometry fields exposed to prediction."""

    numbers = [
        _integer(value, name="atomic number", minimum=1) for value in atomic_numbers
    ]
    if len(positions_angstrom) != len(numbers):
        raise AccuracyContractError("geometry position count differs from atoms")
    positions: list[list[float]] = []
    for row in positions_angstrom:
        if len(row) != 3:
            raise AccuracyContractError("geometry positions must be atom-by-3")
        positions.append([_finite(value, name="position") for value in row])
    payload = {
        "contract": "maple-route2-normalized-fixed-geometry-v1",
        "atomic_numbers": numbers,
        "positions_angstrom": positions,
        "charge": _integer(charge, name="charge", minimum=0),
        "multiplicity": _integer(multiplicity, name="multiplicity", minimum=1),
    }
    return canonical_sha256(payload)


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value) <= _DIGEST_CHARACTERS
    ):
        raise AccuracyContractError(f"{name} must be a lowercase SHA256 digest")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccuracyContractError(f"{name} must be nonempty text")
    return value.strip()


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AccuracyContractError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise AccuracyContractError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AccuracyContractError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise AccuracyContractError(f"{name} must be finite")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AccuracyContractError(f"{name} must be an object")
    return value


def _record_gates(record: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(record.get("gates"), name="record gates")


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise AccuracyContractError(
            f"{name} keys changed; missing={missing}, unknown={unknown}"
        )


def _reject_label_keys(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if key in _FORBIDDEN_LABEL_FREE_KEYS or any(
                marker in lowered
                for marker in (
                    "experimental_delta",
                    "ground_truth",
                    "target_value",
                    "reference_delta_g",
                )
            ):
                raise AccuracyContractError(
                    f"label-free artifact contains forbidden key at {path}.{key}"
                )
            _reject_label_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_label_keys(item, path=f"{path}[{index}]")


def _self_digest(payload: Mapping[str, object], *, name: str) -> str:
    expected = _digest(payload.get("content_sha256"), name=f"{name}.content_sha256")
    content = dict(payload)
    del content["content_sha256"]
    actual = canonical_sha256(content)
    if actual != expected:
        raise AccuracyContractError(
            f"{name} content digest mismatch: {actual} != {expected}"
        )
    return actual


def with_content_sha256(payload: Mapping[str, object]) -> dict[str, object]:
    if "content_sha256" in payload:
        raise AccuracyContractError("content_sha256 must not be supplied")
    result = dict(payload)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _validate_input_record(
    value: Mapping[str, object], *, expected_index: int
) -> dict[str, object]:
    optional = "prior_pilot_geometry_overlap"
    keys = set(value)
    expected = set(_INPUT_RECORD_KEYS)
    if optional not in keys:
        expected.remove(optional)
    _exact_keys(value, frozenset(expected), name="label-free input record")
    if _integer(value["selection_index"], name="selection_index") != expected_index:
        raise AccuracyContractError("label-free input record ordering changed")
    _digest(value["opaque_record_id"], name="opaque_record_id")
    _text(value["canonical_solvent"], name="canonical_solvent")
    if value["partition"] not in {"confirmation", "development"}:
        raise AccuracyContractError("partition must be confirmation or development")
    if optional in value and type(value[optional]) is not bool:
        raise AccuracyContractError(f"{optional} must be a bool")
    _digest(value["geometry_sha256"], name="geometry_sha256")
    normalized = _digest(
        value["normalized_geometry_sha256"],
        name="normalized_geometry_sha256",
    )
    atom_count = _integer(value["atom_count"], name="atom_count", minimum=1)
    numbers = value["atomic_numbers"]
    positions = value["positions_angstrom"]
    if not isinstance(numbers, list) or len(numbers) != atom_count:
        raise AccuracyContractError("atomic_numbers length differs from atom_count")
    if any(type(number) is not int or number < 1 for number in numbers):
        raise AccuracyContractError("atomic_numbers must be positive integers")
    if not isinstance(positions, list) or len(positions) != atom_count:
        raise AccuracyContractError("positions length differs from atom_count")
    for row in positions:
        if not isinstance(row, list) or len(row) != 3:
            raise AccuracyContractError("positions must be atom-by-3")
        for component in row:
            _finite(component, name="position")
    if value["charge"] != 0 or value["multiplicity"] != 1:
        raise AccuracyContractError("accuracy input must be neutral singlet")
    recomputed = normalized_geometry_sha256(
        atomic_numbers=numbers,
        positions_angstrom=positions,
        charge=value["charge"],
        multiplicity=value["multiplicity"],
    )
    if recomputed != normalized:
        raise AccuracyContractError("normalized geometry digest mismatch")
    return dict(value)


def validate_label_free_input_bundle(
    payload: Mapping[str, object], *, expected_record_count: int
) -> dict[str, object]:
    """Validate a private coordinate bundle with no experimental targets."""

    _reject_label_keys(payload)
    required = frozenset(
        {
            "schema_id",
            "artifact_id",
            "created_at_utc",
            "git",
            "preregistration_sha256",
            "protocol_sha256",
            "selection_sha256",
            "selection_fingerprint",
            "dataset",
            "record_count",
            "records",
            "redistribution_guard",
            "claim_boundary",
            "content_sha256",
        }
    )
    _exact_keys(payload, required, name="label-free input bundle")
    if payload["schema_id"] != LABEL_FREE_INPUT_SCHEMA_ID:
        raise AccuracyContractError("unsupported label-free input schema")
    _text(payload["artifact_id"], name="artifact_id")
    _text(payload["created_at_utc"], name="created_at_utc")
    _digest(payload["preregistration_sha256"], name="preregistration_sha256")
    _digest(payload["protocol_sha256"], name="protocol_sha256")
    _digest(payload["selection_sha256"], name="selection_sha256")
    _digest(payload["selection_fingerprint"], name="selection_fingerprint")
    git = payload["git"]
    if not isinstance(git, Mapping):
        raise AccuracyContractError("label-free git identity must be an object")
    _exact_keys(git, frozenset({"head", "tree", "clean"}), name="git identity")
    _text(git["head"], name="git.head")
    _text(git["tree"], name="git.tree")
    if git["clean"] is not True:
        raise AccuracyContractError("label-free input must bind a clean tree")
    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping):
        raise AccuracyContractError("label-free dataset identity must be an object")
    _exact_keys(
        dataset,
        frozenset(
            {
                "protocol_id",
                "protocol_fingerprint",
                "table_sha256",
                "normalized_bundle_sha256",
                "temperature_k",
                "standard_state",
            }
        ),
        name="label-free dataset identity",
    )
    _text(dataset["protocol_id"], name="dataset.protocol_id")
    _digest(dataset["protocol_fingerprint"], name="dataset.protocol_fingerprint")
    _digest(dataset["table_sha256"], name="dataset.table_sha256")
    _digest(
        dataset["normalized_bundle_sha256"],
        name="dataset.normalized_bundle_sha256",
    )
    if _finite(dataset["temperature_k"], name="dataset.temperature_k") != 298.0:
        raise AccuracyContractError("dataset temperature drifted")
    if dataset["standard_state"] != "1M-ideal-gas-to-1M-ideal-solution":
        raise AccuracyContractError("dataset standard state drifted")
    count = _integer(payload["record_count"], name="record_count", minimum=1)
    if count != expected_record_count:
        raise AccuracyContractError("label-free input record count changed")
    records = payload["records"]
    if not isinstance(records, list) or len(records) != count:
        raise AccuracyContractError("label-free input records are incomplete")
    validated = [
        _validate_input_record(record, expected_index=index)
        for index, record in enumerate(records)
        if isinstance(record, Mapping)
    ]
    if len(validated) != count:
        raise AccuracyContractError("every label-free record must be an object")
    opaque = [record["opaque_record_id"] for record in validated]
    if len(set(opaque)) != count:
        raise AccuracyContractError("label-free opaque record IDs must be unique")
    guard = payload["redistribution_guard"]
    if not isinstance(guard, Mapping) or not guard:
        raise AccuracyContractError("redistribution_guard must be a nonempty object")
    _exact_keys(
        guard,
        frozenset(
            {
                "raw_rows_emitted",
                "entry_numbers_emitted",
                "geometry_handles_emitted",
                "solute_names_emitted",
                "formulas_emitted",
                "experimental_values_emitted",
            }
        ),
        name="redistribution_guard",
    )
    if set(guard.values()) != {False}:
        raise AccuracyContractError("label-free redistribution guards must be false")
    _self_digest(payload, name="label-free input bundle")
    return dict(payload)


def _validate_root_gates(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise AccuracyContractError("prediction gates must be an object")
    _exact_keys(value, _ROOT_GATE_KEYS, name="prediction gates")
    if any(type(item) is not bool for item in value.values()):
        raise AccuracyContractError("prediction gates must be booleans")
    return {name: bool(value[name]) for name in sorted(value)}


def validate_prediction_record(
    value: Mapping[str, object],
    *,
    expected_index: int,
    validation_contract: PredictionValidationContract = (
        HYBRID_MNSOL_PREDICTION_VALIDATION
    ),
) -> dict[str, object]:
    _reject_label_keys(value)
    optional = "prior_pilot_geometry_overlap"
    keys = set(value)
    expected = set(_PREDICTION_RECORD_KEYS)
    if optional not in keys:
        expected.remove(optional)
    _exact_keys(value, frozenset(expected), name="prediction record")
    if _integer(value["selection_index"], name="selection_index") != expected_index:
        raise AccuracyContractError("prediction record ordering changed")
    _digest(value["opaque_record_id"], name="opaque_record_id")
    _text(value["canonical_solvent"], name="canonical_solvent")
    if value["partition"] not in {"confirmation", "development"}:
        raise AccuracyContractError("prediction partition is invalid")
    if optional in value and type(value[optional]) is not bool:
        raise AccuracyContractError(f"{optional} must be a bool")
    _digest(value["geometry_sha256"], name="geometry_sha256")
    _digest(
        value["normalized_geometry_sha256"],
        name="normalized_geometry_sha256",
    )
    _integer(value["atom_count"], name="atom_count", minimum=1)
    finite_fields = (
        "solution_total_energy_eV",
        "vacuum_energy_eV",
        "predicted_delta_g_eV",
        "predicted_delta_g_kcal_mol",
        "polarization_kcal_mol",
        "cds_kcal_mol",
        "permanent_charge_e",
        "induced_charge_e",
        "combined_charge_e",
        "cold_actual_residual_norm",
        "wide_actual_residual_norm",
        "induced_source_max_abs_difference_e",
        "native_field_max_abs_difference_eV_per_e",
        "continuum_energy_abs_difference_eV",
    )
    finite_values = {name: _finite(value[name], name=name) for name in finite_fields}
    _integer(value["cold_iterations"], name="cold_iterations")
    _integer(value["wide_iterations"], name="wide_iterations")
    actual_gates = _validate_root_gates(value["gates"])
    for name in (
        "configuration_sha256",
        "continuum_configuration_sha256",
        "continuum_provenance_sha256",
        "equation_sha256",
        "cold_root_sha256",
        "wide_root_sha256",
        "base_ledger_sha256",
        "total_ledger_sha256",
        "solvent_term_configuration_sha256",
        "state_sha256",
    ):
        _digest(value[name], name=name)
    if not math.isclose(
        finite_values["solution_total_energy_eV"] - finite_values["vacuum_energy_eV"],
        finite_values["predicted_delta_g_eV"],
        rel_tol=0.0,
        abs_tol=validation_contract.energy_closure_tolerance_eV,
    ):
        raise AccuracyContractError("solution total does not close DeltaG")
    if not math.isclose(
        finite_values["predicted_delta_g_kcal_mol"],
        finite_values["predicted_delta_g_eV"] * EV_TO_KCAL_MOL,
        rel_tol=0.0,
        abs_tol=validation_contract.kcal_closure_tolerance,
    ):
        raise AccuracyContractError("eV-to-kcal prediction conversion drifted")
    if not math.isclose(
        finite_values["predicted_delta_g_kcal_mol"],
        finite_values["polarization_kcal_mol"] + finite_values["cds_kcal_mol"],
        rel_tol=0.0,
        abs_tol=validation_contract.kcal_closure_tolerance,
    ):
        raise AccuracyContractError("polarization plus CDS does not close prediction")
    if not math.isclose(
        finite_values["combined_charge_e"],
        finite_values["permanent_charge_e"] + finite_values["induced_charge_e"],
        rel_tol=0.0,
        abs_tol=validation_contract.charge_closure_tolerance_e,
    ):
        raise AccuracyContractError("permanent plus induced charge does not close")
    for name in (
        "cold_actual_residual_norm",
        "wide_actual_residual_norm",
        "induced_source_max_abs_difference_e",
        "native_field_max_abs_difference_eV_per_e",
        "continuum_energy_abs_difference_eV",
    ):
        if finite_values[name] < 0.0:
            raise AccuracyContractError(f"{name} must be nonnegative")
    expected_gates = {
        "actual_residuals": max(
            finite_values["cold_actual_residual_norm"],
            finite_values["wide_actual_residual_norm"],
        )
        <= validation_contract.maximum_actual_residual_norm,
        "cold_wide_source": finite_values["induced_source_max_abs_difference_e"]
        <= validation_contract.maximum_induced_source_absolute_difference_e,
        "cold_wide_field": finite_values["native_field_max_abs_difference_eV_per_e"]
        <= validation_contract.maximum_native_field_absolute_difference_eV_per_e,
        "cold_wide_energy": finite_values["continuum_energy_abs_difference_eV"]
        <= validation_contract.maximum_continuum_energy_absolute_difference_eV,
        "permanent_charge": abs(finite_values["permanent_charge_e"])
        <= validation_contract.maximum_permanent_charge_error_e,
        "induced_charge": abs(finite_values["induced_charge_e"])
        <= validation_contract.maximum_induced_charge_error_e,
        "combined_charge": abs(finite_values["combined_charge_e"])
        <= validation_contract.maximum_combined_charge_error_e,
    }
    if actual_gates != {name: expected_gates[name] for name in sorted(expected_gates)}:
        raise AccuracyContractError("prediction gates were not mechanically derived")
    return dict(value)


def prediction_measurement_sha256(
    records: Sequence[Mapping[str, object]],
    *,
    validation_contract: PredictionValidationContract = (
        HYBRID_MNSOL_PREDICTION_VALIDATION
    ),
) -> str:
    validated = [
        validate_prediction_record(
            record,
            expected_index=index,
            validation_contract=validation_contract,
        )
        for index, record in enumerate(records)
    ]
    return canonical_sha256(
        {
            "contract": "maple-route2-label-free-prediction-measurement-v1",
            "records": validated,
        }
    )


def validate_prediction_terminal(
    payload: Mapping[str, object],
    *,
    expected_record_count: int,
    validation_contract: PredictionValidationContract = (
        HYBRID_MNSOL_PREDICTION_VALIDATION
    ),
) -> dict[str, object]:
    """Validate one immutable complete-or-failure prediction terminal."""

    _reject_label_keys(payload)
    common = {
        "schema_id",
        "artifact_id",
        "status",
        "attempt_slot_id",
        "execution_id",
        "started_at_utc",
        "finished_at_utc",
        "claim_boundary",
        "profile_id",
        "scalar_id",
        "git",
        "seal",
        "input_bundle",
        "checkpoints",
        "source_files_sha256",
        "runtime",
        "record_count_expected",
        "records",
        "timing_seconds",
        "capabilities",
        "content_sha256",
    }
    status = payload.get("status")
    expected = set(common)
    if status == "complete":
        expected.add("measurement_sha256")
    elif status == "failure":
        expected.add("failure")
    else:
        raise AccuracyContractError("prediction terminal status is invalid")
    _exact_keys(payload, frozenset(expected), name="prediction terminal")
    if payload["schema_id"] != PREDICTION_TERMINAL_SCHEMA_ID:
        raise AccuracyContractError("unsupported prediction terminal schema")
    for name in ("attempt_slot_id", "execution_id"):
        _digest(payload[name], name=name)
    for name in (
        "artifact_id",
        "started_at_utc",
        "finished_at_utc",
        "claim_boundary",
        "profile_id",
        "scalar_id",
    ):
        _text(payload[name], name=name)
    if payload["capabilities"] != CAPABILITIES_CLOSED:
        raise AccuracyContractError("prediction terminal cannot admit capabilities")
    count = _integer(
        payload["record_count_expected"],
        name="record_count_expected",
        minimum=1,
    )
    if count != expected_record_count:
        raise AccuracyContractError("prediction terminal record count changed")
    records = payload["records"]
    if not isinstance(records, list):
        raise AccuracyContractError("prediction records must be a list")
    if status == "complete" and len(records) != count:
        raise AccuracyContractError("complete prediction terminal is missing records")
    if status == "failure" and len(records) > count:
        raise AccuracyContractError("failure terminal has too many partial records")
    validated = [
        validate_prediction_record(
            record,
            expected_index=index,
            validation_contract=validation_contract,
        )
        for index, record in enumerate(records)
        if isinstance(record, Mapping)
    ]
    if len(validated) != len(records):
        raise AccuracyContractError("every prediction record must be an object")
    if status == "complete":
        expected_measurement = prediction_measurement_sha256(
            validated, validation_contract=validation_contract
        )
        if (
            _digest(payload["measurement_sha256"], name="measurement_sha256")
            != expected_measurement
        ):
            raise AccuracyContractError("prediction measurement digest mismatch")
        if not all(all(_record_gates(record).values()) for record in validated):
            raise AccuracyContractError("complete prediction contains failed gates")
    else:
        failure = payload["failure"]
        if not isinstance(failure, Mapping):
            raise AccuracyContractError("typed prediction failure must be an object")
        required_failure = frozenset(
            {
                "stage",
                "error_type",
                "message",
                "failed_selection_index",
                "failed_opaque_record_id",
            }
        )
        _exact_keys(failure, required_failure, name="typed prediction failure")
        _text(failure["stage"], name="failure.stage")
        _text(failure["error_type"], name="failure.error_type")
        _text(failure["message"], name="failure.message")
        index = failure["failed_selection_index"]
        if index is not None:
            _integer(index, name="failed_selection_index")
        record_id = failure["failed_opaque_record_id"]
        if record_id is not None:
            _digest(record_id, name="failed_opaque_record_id")
    _self_digest(payload, name="prediction terminal")
    return dict(payload)


def accuracy_statistics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise AccuracyContractError("accuracy statistics require records")
    errors = [
        _finite(record["signed_error_kcal_mol"], name="signed error")
        for record in records
    ]
    predictions = [
        _finite(record["predicted_delta_g_kcal_mol"], name="prediction")
        for record in records
    ]
    experiments = [
        _finite(record["experimental_delta_g_kcal_mol"], name="experiment")
        for record in records
    ]
    absolute = [abs(value) for value in errors]
    return {
        "record_count": len(records),
        "mean_signed_error_kcal_mol": fmean(errors),
        "mean_absolute_error_kcal_mol": fmean(absolute),
        "root_mean_square_error_kcal_mol": math.sqrt(
            fmean(value * value for value in errors)
        ),
        "maximum_absolute_error_kcal_mol": max(absolute),
        "mean_predicted_delta_g_kcal_mol": fmean(predictions),
        "mean_experimental_delta_g_kcal_mol": fmean(experiments),
        "mean_polarization_kcal_mol": fmean(
            _finite(record["polarization_kcal_mol"], name="polarization")
            for record in records
        ),
        "mean_cds_kcal_mol": fmean(
            _finite(record["cds_kcal_mol"], name="CDS") for record in records
        ),
    }


def partitioned_accuracy_metrics(
    records: Sequence[Mapping[str, object]],
    *,
    expected_partition_counts: Mapping[str, int],
) -> dict[str, object]:
    expected_total = sum(expected_partition_counts.values())
    if len(records) != expected_total:
        raise AccuracyContractError("scored record count changed")
    result: dict[str, object] = {"all": accuracy_statistics(records)}
    for partition, expected in expected_partition_counts.items():
        selected = [record for record in records if record["partition"] == partition]
        if len(selected) != expected:
            raise AccuracyContractError(f"partition count changed for {partition}")
        result[partition] = accuracy_statistics(selected)
    return result


def _prediction_projection(record: Mapping[str, object]) -> dict[str, object]:
    keys = set(_PREDICTION_RECORD_KEYS)
    if "prior_pilot_geometry_overlap" not in record:
        keys.remove("prior_pilot_geometry_overlap")
    return {name: record[name] for name in keys}


def _maximum_root_metrics(
    records: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    return {
        "actual_root_residual_norm": max(
            max(
                _finite(record["cold_actual_residual_norm"], name="cold residual"),
                _finite(record["wide_actual_residual_norm"], name="wide residual"),
            )
            for record in records
        ),
        "induced_source_difference_e": max(
            _finite(
                record["induced_source_max_abs_difference_e"],
                name="source difference",
            )
            for record in records
        ),
        "native_field_difference_eV_per_e": max(
            _finite(
                record["native_field_max_abs_difference_eV_per_e"],
                name="field difference",
            )
            for record in records
        ),
        "continuum_energy_difference_eV": max(
            _finite(
                record["continuum_energy_abs_difference_eV"],
                name="energy difference",
            )
            for record in records
        ),
        "permanent_charge_error_e": max(
            abs(_finite(record["permanent_charge_e"], name="permanent charge"))
            for record in records
        ),
        "induced_charge_error_e": max(
            abs(_finite(record["induced_charge_e"], name="induced charge"))
            for record in records
        ),
        "combined_charge_error_e": max(
            abs(_finite(record["combined_charge_e"], name="combined charge"))
            for record in records
        ),
    }


def validate_scored_bundle(
    payload: Mapping[str, object],
    *,
    expected_record_count: int,
    expected_partition_counts: Mapping[str, int],
) -> dict[str, object]:
    """Validate the complete private label-only mechanical score bundle."""

    _exact_keys(
        payload,
        frozenset(
            {
                "schema_id",
                "artifact_id",
                "status",
                "attempt_slot_id",
                "evidence_class",
                "terminal_type",
                "scored_at_utc",
                "claim_boundary",
                "profile_id",
                "scalar_id",
                "prediction",
                "dataset",
                "selection",
                "records",
                "aggregate_metrics",
                "gates",
                "maximum_root_metrics",
                "scorer_source_sha256",
                "capabilities",
                "public_projection_sha256",
                "content_sha256",
            }
        ),
        name="private scored bundle",
    )
    if payload["schema_id"] != SCORED_BUNDLE_SCHEMA_ID:
        raise AccuracyContractError("unsupported private scored-bundle schema")
    if payload["status"] not in {"pass", "fail"}:
        raise AccuracyContractError("private score status must be pass or fail")
    _digest(payload["attempt_slot_id"], name="attempt_slot_id")
    if (
        payload["evidence_class"] != "exposure-aware-known-panel-regression"
        or payload["terminal_type"] != "complete"
    ):
        raise AccuracyContractError("private score evidence class drifted")
    for name in (
        "artifact_id",
        "scored_at_utc",
        "claim_boundary",
        "profile_id",
        "scalar_id",
    ):
        _text(payload[name], name=name)
    if payload["capabilities"] != CAPABILITIES_CLOSED:
        raise AccuracyContractError("score bundle cannot admit capabilities")
    _digest(payload["scorer_source_sha256"], name="scorer_source_sha256")
    prediction = payload["prediction"]
    if not isinstance(prediction, Mapping):
        raise AccuracyContractError("score prediction binding must be an object")
    _exact_keys(
        prediction,
        frozenset(
            {
                "execution_id",
                "attempt_slot_id",
                "file_sha256",
                "content_sha256",
                "measurement_sha256",
            }
        ),
        name="score prediction binding",
    )
    for name, value in prediction.items():
        _digest(value, name=f"prediction.{name}")
    if payload["attempt_slot_id"] != prediction["attempt_slot_id"]:
        raise AccuracyContractError("score attempt slot differs from prediction")
    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping):
        raise AccuracyContractError("score dataset binding must be an object")
    _exact_keys(
        dataset,
        frozenset(
            {
                "protocol_id",
                "protocol_fingerprint",
                "protocol_sha256",
                "selection_sha256",
                "selection_fingerprint",
                "table_sha256",
                "normalized_bundle_sha256",
                "source_artifact_sha256",
                "temperature_k",
                "standard_state",
            }
        ),
        name="score dataset binding",
    )
    for name in (
        "protocol_fingerprint",
        "protocol_sha256",
        "selection_sha256",
        "selection_fingerprint",
        "table_sha256",
        "normalized_bundle_sha256",
        "source_artifact_sha256",
    ):
        _digest(dataset[name], name=f"dataset.{name}")
    _text(dataset["protocol_id"], name="dataset.protocol_id")
    if _finite(dataset["temperature_k"], name="dataset.temperature_k") != 298.0:
        raise AccuracyContractError("score dataset temperature drifted")
    if dataset["standard_state"] != "1M-ideal-gas-to-1M-ideal-solution":
        raise AccuracyContractError("score dataset standard state drifted")
    selection = payload["selection"]
    if not isinstance(selection, Mapping):
        raise AccuracyContractError("score selection summary must be an object")
    _exact_keys(
        selection,
        frozenset({"record_count", "solvent_count", "partition_counts"}),
        name="score selection summary",
    )
    if selection["record_count"] != expected_record_count:
        raise AccuracyContractError("score selection record count drifted")
    if selection["solvent_count"] != 10:
        raise AccuracyContractError("score selection solvent count drifted")
    if selection["partition_counts"] != dict(expected_partition_counts):
        raise AccuracyContractError("score partition counts drifted")
    records = payload["records"]
    if not isinstance(records, list) or len(records) != expected_record_count:
        raise AccuracyContractError("score records are incomplete")
    validated_records: list[dict[str, object]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise AccuracyContractError("every scored record must be an object")
        optional = "prior_pilot_geometry_overlap"
        expected_keys = set(_SCORED_RECORD_KEYS)
        if optional not in raw:
            expected_keys.remove(optional)
        _exact_keys(raw, frozenset(expected_keys), name="scored record")
        prediction_record = validate_prediction_record(
            _prediction_projection(raw), expected_index=index
        )
        experimental = _finite(
            raw["experimental_delta_g_kcal_mol"], name="experimental DeltaG"
        )
        predicted = _finite(
            prediction_record["predicted_delta_g_kcal_mol"], name="prediction"
        )
        signed = _finite(raw["signed_error_kcal_mol"], name="signed error")
        absolute = _finite(raw["absolute_error_kcal_mol"], name="absolute error")
        if not math.isclose(
            signed, predicted - experimental, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise AccuracyContractError("scored signed error was not mechanical")
        if not math.isclose(absolute, abs(signed), rel_tol=0.0, abs_tol=1.0e-12):
            raise AccuracyContractError("scored absolute error was not mechanical")
        validated_records.append(dict(raw))
    metrics = partitioned_accuracy_metrics(
        validated_records,
        expected_partition_counts=expected_partition_counts,
    )
    if payload["aggregate_metrics"] != metrics:
        raise AccuracyContractError("aggregate accuracy metrics were not mechanical")
    maxima = _maximum_root_metrics(validated_records)
    if payload["maximum_root_metrics"] != maxima:
        raise AccuracyContractError("maximum root metrics were not mechanical")
    gates = payload["gates"]
    if not isinstance(gates, Mapping):
        raise AccuracyContractError("accuracy gates must be an object")
    _exact_keys(gates, _ACCURACY_GATE_KEYS, name="accuracy gates")
    expected_gates = {
        "complete_exact_frozen_panel": len(validated_records) == expected_record_count,
        "ten_distinct_solvents": len(
            {str(record["canonical_solvent"]) for record in validated_records}
        )
        == 10,
        "all_cold_wide_roots_pass": all(
            bool(_record_gates(record)[name])
            for record in validated_records
            for name in ("cold_wide_source", "cold_wide_field", "cold_wide_energy")
        ),
        "all_actual_root_residuals_pass": all(
            bool(_record_gates(record)["actual_residuals"])
            for record in validated_records
        ),
        "all_permanent_and_induced_charge_checks_pass": all(
            bool(_record_gates(record)[name])
            for record in validated_records
            for name in ("permanent_charge", "induced_charge", "combined_charge")
        ),
        "mae_at_most_1_5_kcal_mol": _finite(
            _mapping(metrics["all"], name="all metrics")[
                "mean_absolute_error_kcal_mol"
            ],
            name="all MAE",
        )
        <= 1.5,
    }
    if dict(gates) != expected_gates:
        raise AccuracyContractError("accuracy gates were not mechanically derived")
    expected_status = "pass" if all(expected_gates.values()) else "fail"
    if payload["status"] != expected_status:
        raise AccuracyContractError("score status differs from mechanical gates")
    projection = _public_projection_unchecked(payload)
    if _digest(
        payload["public_projection_sha256"], name="public_projection_sha256"
    ) != canonical_sha256(projection):
        raise AccuracyContractError("public projection digest mismatch")
    _self_digest(payload, name="private scored bundle")
    return dict(payload)


def _public_projection_unchecked(private: Mapping[str, object]) -> dict[str, object]:
    records = private["records"]
    assert isinstance(records, list)
    prediction = _mapping(private["prediction"], name="private prediction binding")
    configuration_by_solvent = {
        str(record["canonical_solvent"]): str(record["configuration_sha256"])
        for record in records
        if isinstance(record, Mapping)
    }
    return with_content_sha256(
        {
            "schema_id": PUBLIC_ACCURACY_SCHEMA_ID,
            "artifact_id": private["artifact_id"],
            "status": private["status"],
            "attempt_slot_id": private["attempt_slot_id"],
            "execution_id": prediction["execution_id"],
            "evidence_class": private["evidence_class"],
            "terminal_type": private["terminal_type"],
            "scored_at_utc": private["scored_at_utc"],
            "claim_boundary": private["claim_boundary"],
            "profile_id": private["profile_id"],
            "scalar_id": private["scalar_id"],
            "prediction": private["prediction"],
            "dataset": private["dataset"],
            "aggregate_metrics": private["aggregate_metrics"],
            "gates": private["gates"],
            "maximum_root_metrics": private["maximum_root_metrics"],
            "configuration_sha256_by_solvent": configuration_by_solvent,
            "record_count": len(records),
            "row_level_data_emitted": False,
            "capabilities": CAPABILITIES_CLOSED,
            "visibility": "public-aggregate-only",
        }
    )


def validate_public_accuracy_projection(
    payload: Mapping[str, object], *, expected_record_count: int
) -> dict[str, object]:
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_id",
                "artifact_id",
                "status",
                "attempt_slot_id",
                "execution_id",
                "evidence_class",
                "terminal_type",
                "scored_at_utc",
                "claim_boundary",
                "profile_id",
                "scalar_id",
                "prediction",
                "dataset",
                "aggregate_metrics",
                "gates",
                "maximum_root_metrics",
                "configuration_sha256_by_solvent",
                "record_count",
                "row_level_data_emitted",
                "capabilities",
                "visibility",
                "content_sha256",
            }
        ),
        name="public accuracy projection",
    )
    if payload["schema_id"] != PUBLIC_ACCURACY_SCHEMA_ID:
        raise AccuracyContractError("unsupported public accuracy schema")
    _digest(payload["attempt_slot_id"], name="attempt_slot_id")
    _digest(payload["execution_id"], name="execution_id")
    if (
        payload["evidence_class"] != "exposure-aware-known-panel-regression"
        or payload["terminal_type"] != "complete"
    ):
        raise AccuracyContractError("public accuracy evidence class drifted")
    if payload["record_count"] != expected_record_count:
        raise AccuracyContractError("public record count drifted")
    if payload["row_level_data_emitted"] is not False:
        raise AccuracyContractError("public artifact cannot emit row-level data")
    if payload["capabilities"] != CAPABILITIES_CLOSED:
        raise AccuracyContractError("public aggregate cannot admit capabilities")
    if payload["visibility"] != "public-aggregate-only":
        raise AccuracyContractError("public visibility contract drifted")
    forbidden_public_keys = {
        "records",
        "opaque_record_id",
        "geometry_sha256",
        "normalized_geometry_sha256",
        "experimental_delta_g_kcal_mol",
        "predicted_delta_g_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
    }

    def inspect(value: object) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden_public_keys.intersection(value)
            if overlap:
                raise AccuracyContractError(
                    f"public artifact contains private keys: {sorted(overlap)}"
                )
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(payload)
    _self_digest(payload, name="public accuracy projection")
    return dict(payload)


def prediction_failure_public_projection(
    terminal: Mapping[str, object], *, terminal_file_sha256: str
) -> dict[str, object]:
    """Project a typed/aborted prediction failure without row-level evidence."""

    validate_prediction_terminal(terminal, expected_record_count=10)
    if terminal.get("status") != "failure":
        raise AccuracyContractError(
            "only a failed prediction can use failure projection"
        )
    failure = _mapping(terminal.get("failure"), name="prediction failure")
    records = terminal.get("records")
    if not isinstance(records, list):
        raise AccuracyContractError("prediction failure records must be a list")
    projection = with_content_sha256(
        {
            "schema_id": PUBLIC_FAILURE_SCHEMA_ID,
            "artifact_id": "route2-mace-mdp-polar-hybrid-mnsol10-failure-v1",
            "status": "fail",
            "attempt_slot_id": terminal["attempt_slot_id"],
            "execution_id": terminal["execution_id"],
            "evidence_class": "exposure-aware-known-panel-regression",
            "terminal_type": failure["stage"],
            "profile_id": terminal["profile_id"],
            "scalar_id": terminal["scalar_id"],
            "claim_boundary": terminal["claim_boundary"],
            "git": terminal["git"],
            "seal": terminal["seal"],
            "input_bundle": terminal["input_bundle"],
            "checkpoints": terminal["checkpoints"],
            "source_files_sha256": terminal["source_files_sha256"],
            "prediction_terminal": {
                "file_sha256": _digest(
                    terminal_file_sha256, name="terminal_file_sha256"
                ),
                "content_sha256": terminal["content_sha256"],
            },
            "failure": {
                "stage": failure["stage"],
                "error_type": failure["error_type"],
                "panel_completion": "incomplete",
                "expected_record_count": 10,
            },
            "row_level_data_emitted": False,
            "score_performed": False,
            "attempt_slot_closed": True,
            "capabilities": CAPABILITIES_CLOSED,
            "visibility": "public-aggregate-failure-only",
        }
    )
    validate_public_failure_projection(projection)
    return projection


def validate_public_failure_projection(
    payload: Mapping[str, object],
) -> dict[str, object]:
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_id",
                "artifact_id",
                "status",
                "attempt_slot_id",
                "execution_id",
                "evidence_class",
                "terminal_type",
                "profile_id",
                "scalar_id",
                "claim_boundary",
                "git",
                "seal",
                "input_bundle",
                "checkpoints",
                "source_files_sha256",
                "prediction_terminal",
                "failure",
                "row_level_data_emitted",
                "score_performed",
                "attempt_slot_closed",
                "capabilities",
                "visibility",
                "content_sha256",
            }
        ),
        name="public failure projection",
    )
    if (
        payload["schema_id"] != PUBLIC_FAILURE_SCHEMA_ID
        or payload["status"] != "fail"
        or payload["evidence_class"] != "exposure-aware-known-panel-regression"
        or payload["row_level_data_emitted"] is not False
        or payload["score_performed"] is not False
        or payload["attempt_slot_closed"] is not True
        or payload["capabilities"] != CAPABILITIES_CLOSED
        or payload["visibility"] != "public-aggregate-failure-only"
    ):
        raise AccuracyContractError("public failure projection contract drifted")
    _digest(payload["attempt_slot_id"], name="attempt_slot_id")
    _digest(payload["execution_id"], name="execution_id")
    failure = _mapping(payload["failure"], name="public failure summary")
    _exact_keys(
        failure,
        frozenset({"stage", "error_type", "panel_completion", "expected_record_count"}),
        name="public failure summary",
    )
    if (
        failure["panel_completion"] != "incomplete"
        or failure["expected_record_count"] != 10
    ):
        raise AccuracyContractError("public failure completion summary drifted")
    forbidden = {
        "records",
        "opaque_record_id",
        "geometry_sha256",
        "normalized_geometry_sha256",
        "experimental_delta_g_kcal_mol",
        "predicted_delta_g_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
        "failed_opaque_record_id",
    }

    def inspect(value: object) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden.intersection(value)
            if overlap:
                raise AccuracyContractError(
                    f"public failure contains private keys: {sorted(overlap)}"
                )
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(payload)
    _self_digest(payload, name="public failure projection")
    return dict(payload)


def public_projection_sha256_for_scored_payload(
    private: Mapping[str, object],
) -> str:
    """Digest the deterministic public projection before private self-sealing."""

    return canonical_sha256(_public_projection_unchecked(private))


def build_scored_bundle(
    *,
    artifact_id: str,
    scored_at_utc: str,
    claim_boundary: str,
    profile_id: str,
    scalar_id: str,
    prediction: Mapping[str, object],
    dataset: Mapping[str, object],
    scored_records: Sequence[Mapping[str, object]],
    scorer_source_sha256: str,
    expected_partition_counts: Mapping[str, int],
) -> dict[str, object]:
    """Mechanically assemble and self-validate one private score bundle."""

    records = [dict(record) for record in scored_records]
    metrics = partitioned_accuracy_metrics(
        records, expected_partition_counts=expected_partition_counts
    )
    maxima = _maximum_root_metrics(records)
    gates = {
        "complete_exact_frozen_panel": len(records)
        == sum(expected_partition_counts.values()),
        "ten_distinct_solvents": len(
            {str(record["canonical_solvent"]) for record in records}
        )
        == 10,
        "all_cold_wide_roots_pass": all(
            bool(_record_gates(record)[name])
            for record in records
            for name in ("cold_wide_source", "cold_wide_field", "cold_wide_energy")
        ),
        "all_actual_root_residuals_pass": all(
            bool(_record_gates(record)["actual_residuals"]) for record in records
        ),
        "all_permanent_and_induced_charge_checks_pass": all(
            bool(_record_gates(record)[name])
            for record in records
            for name in ("permanent_charge", "induced_charge", "combined_charge")
        ),
        "mae_at_most_1_5_kcal_mol": _finite(
            _mapping(metrics["all"], name="all metrics")[
                "mean_absolute_error_kcal_mol"
            ],
            name="all MAE",
        )
        <= 1.5,
    }
    bundle: dict[str, object] = {
        "schema_id": SCORED_BUNDLE_SCHEMA_ID,
        "artifact_id": artifact_id,
        "status": "pass" if all(gates.values()) else "fail",
        "attempt_slot_id": prediction["attempt_slot_id"],
        "evidence_class": "exposure-aware-known-panel-regression",
        "terminal_type": "complete",
        "scored_at_utc": scored_at_utc,
        "claim_boundary": claim_boundary,
        "profile_id": profile_id,
        "scalar_id": scalar_id,
        "prediction": dict(prediction),
        "dataset": dict(dataset),
        "selection": {
            "record_count": len(records),
            "solvent_count": len(
                {str(record["canonical_solvent"]) for record in records}
            ),
            "partition_counts": {
                partition: sum(record["partition"] == partition for record in records)
                for partition in expected_partition_counts
            },
        },
        "records": records,
        "aggregate_metrics": metrics,
        "gates": gates,
        "maximum_root_metrics": maxima,
        "scorer_source_sha256": scorer_source_sha256,
        "capabilities": CAPABILITIES_CLOSED,
        "public_projection_sha256": "0" * 64,
    }
    bundle["public_projection_sha256"] = public_projection_sha256_for_scored_payload(
        bundle
    )
    bundle = with_content_sha256(bundle)
    validate_scored_bundle(
        bundle,
        expected_record_count=len(records),
        expected_partition_counts=expected_partition_counts,
    )
    return bundle


def public_accuracy_projection(private: Mapping[str, object]) -> dict[str, object]:
    """Build a redistribution-safe aggregate projection from a private score."""

    validate_scored_bundle(
        private,
        expected_record_count=10,
        expected_partition_counts={"confirmation": 8, "development": 2},
    )
    projection = _public_projection_unchecked(private)
    validate_public_accuracy_projection(projection, expected_record_count=10)
    return projection


__all__ = [
    "AccuracyContractError",
    "CAPABILITIES_CLOSED",
    "EV_TO_KCAL_MOL",
    "HYBRID_MNSOL_PREDICTION_VALIDATION",
    "LABEL_FREE_INPUT_SCHEMA_ID",
    "PREDICTION_TERMINAL_SCHEMA_ID",
    "PUBLIC_ACCURACY_SCHEMA_ID",
    "PUBLIC_FAILURE_SCHEMA_ID",
    "SCORED_BUNDLE_SCHEMA_ID",
    "PredictionValidationContract",
    "accuracy_statistics",
    "canonical_json_bytes",
    "canonical_sha256",
    "build_scored_bundle",
    "normalized_geometry_sha256",
    "partitioned_accuracy_metrics",
    "prediction_measurement_sha256",
    "prediction_failure_public_projection",
    "public_accuracy_projection",
    "public_projection_sha256_for_scored_payload",
    "validate_public_accuracy_projection",
    "validate_public_failure_projection",
    "validate_scored_bundle",
    "validate_label_free_input_bundle",
    "validate_prediction_record",
    "validate_prediction_terminal",
    "with_content_sha256",
]
