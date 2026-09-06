"""Fail-closed split-conformal selective ranking for fixed Route 1 energies.

This module never calibrates, offsets, averages, or otherwise changes a
predicted energy.  It consumes a separately sealed, complete-series
calibration artifact and decides only whether the sign of one already-computed
free-energy difference has enough support to report.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from benchmark_core import artifact_content_sha256, load_json


CALIBRATION_ARTIFACT_TYPE = "route1-selective-ranking-calibration"
PREDICTION_EVIDENCE_ARTIFACT_TYPE = (
    "route1-selective-ranking-prediction-evidence"
)
STATUS_CERTIFIED = "certified"
STATUS_UNCERTAIN = "uncertain"
STATUS_OUT_OF_DOMAIN = "out_of_domain"
_STATUSES = frozenset({STATUS_CERTIFIED, STATUS_UNCERTAIN, STATUS_OUT_OF_DOMAIN})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    return numeric


def _domain_key(domain: Mapping[str, object], keys: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        value = domain.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Ranking domain lacks non-empty {key!r}.")
        values.append(value)
    return tuple(values)


def split_conformal_radius(errors: Sequence[float], *, miscoverage: float) -> float:
    """Return the finite-sample split-conformal order-statistic radius.

    `errors` must contain one complete-series maximum pair error per independent
    calibration series.  This intentionally does not treat overlapping pairs as
    independent samples.
    """
    alpha = _finite(miscoverage, name="miscoverage")
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one.")
    values = sorted(_finite(value, name="series maximum error") for value in errors)
    if not values or any(value < 0.0 for value in values):
        raise ValueError("Calibration needs non-negative errors from one or more series.")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    if rank > len(values):
        raise ValueError(
            "Calibration has too few independent series for the requested "
            "finite-sample target coverage."
        )
    return float(values[rank - 1])


def validate_calibration_artifact(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Validate a separate, qualified series-level calibration artifact."""
    value = dict(artifact)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != CALIBRATION_ARTIFACT_TYPE
        or value.get("content_sha256") != artifact_content_sha256(value)
        or not isinstance(value.get("protocol_id"), str)
        or not value.get("protocol_id")
        or not _is_sha256(value.get("source_energy_artifact_sha256"))
    ):
        raise ValueError("Selective-ranking calibration artifact is invalid.")
    boundary = value.get("boundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("energy_correction_applied") is not False
        or boundary.get("energy_residual_fit") is not False
        or boundary.get("endpoint_selection") is not False
        or boundary.get("calibrated_before_test_labels") is not True
        or boundary.get("series_disjoint_from_test") is not True
        or boundary.get("external_independent_series") is not True
    ):
        raise ValueError("Selective-ranking calibration lacks the scientific boundary.")
    parameters = value.get("calibration_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Selective-ranking calibration parameters are missing.")
    miscoverage = _finite(parameters.get("miscoverage"), name="miscoverage")
    minimum_series = parameters.get("minimum_series_per_domain")
    if (
        not 0.0 < miscoverage < 1.0
        or isinstance(minimum_series, bool)
        or not isinstance(minimum_series, int)
        or minimum_series < 1
        or parameters.get("series_error_statistic")
        != "maximum_absolute_pair_error_per_complete_series"
    ):
        raise ValueError("Selective-ranking calibration parameters are invalid.")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Selective-ranking calibration provenance is missing.")
    provenance_digest_fields = (
        "calibration_series_manifest_sha256",
        "held_out_test_series_manifest_sha256",
        "held_out_test_energy_artifact_sha256",
        "series_disjointness_audit_sha256",
        "domain_schema_sha256",
        "endpoint_fingerprint_sha256",
        "pair_disagreement_audit_protocol_sha256",
        "prediction_evidence_manifest_sha256",
    )
    if any(not _is_sha256(provenance.get(field)) for field in provenance_digest_fields):
        raise ValueError("Selective-ranking calibration provenance is invalid.")
    held_out_ids = provenance.get("held_out_test_series_ids")
    if (
        not isinstance(held_out_ids, list)
        or not held_out_ids
        or any(not isinstance(identifier, str) or not identifier for identifier in held_out_ids)
        or len(held_out_ids) != len(set(held_out_ids))
    ):
        raise ValueError("Held-out selective-ranking test series IDs are invalid.")
    allowed_prediction_evidence = provenance.get(
        "allowed_prediction_evidence_sha256"
    )
    if (
        not isinstance(allowed_prediction_evidence, list)
        or not allowed_prediction_evidence
        or any(not _is_sha256(digest) for digest in allowed_prediction_evidence)
        or len(allowed_prediction_evidence)
        != len(set(allowed_prediction_evidence))
    ):
        raise ValueError(
            "Allowed selective-ranking prediction evidence is invalid."
        )
    keys = value.get("domain_keys")
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or len(keys) != len(set(keys))
    ):
        raise ValueError("Selective-ranking domain keys are invalid.")
    records = value.get("series_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Selective-ranking calibration needs complete series records.")
    identifiers: set[str] = set()
    normalized_records: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Selective-ranking calibration record is invalid.")
        series_id = record.get("series_id")
        domain = record.get("domain")
        error = _finite(record.get("maximum_pair_error_kcal_mol"), name="maximum pair error")
        if (
            not isinstance(series_id, str)
            or not series_id
            or series_id in identifiers
            or not isinstance(domain, Mapping)
            or error < 0.0
        ):
            raise ValueError("Selective-ranking calibration record is invalid.")
        domain_key = _domain_key(domain, keys)
        normalized_records.append(
            {
                "series_id": series_id,
                "domain": {
                    key: domain_key[index] for index, key in enumerate(keys)
                },
                "maximum_pair_error_kcal_mol": error,
            }
        )
        identifiers.add(series_id)
    if identifiers & set(held_out_ids):
        raise ValueError("Calibration and held-out test series IDs overlap.")
    return {
        "protocol_id": value.get("protocol_id"),
        "domain_keys": list(keys),
        "series_records": normalized_records,
        "source_energy_artifact_sha256": value.get("source_energy_artifact_sha256"),
        "miscoverage": miscoverage,
        "minimum_series_per_domain": minimum_series,
        "provenance": {
            field: provenance.get(field) for field in provenance_digest_fields
        }
        | {
            "held_out_test_series_ids": list(held_out_ids),
            "allowed_prediction_evidence_sha256": list(
                allowed_prediction_evidence
            ),
        },
    }


def load_calibration_artifact(path: str | Path) -> dict[str, object]:
    return validate_calibration_artifact(load_json(path))


class SelectiveRankCalibrator:
    """Immutable split-conformal sign-decision policy over whole series."""

    def __init__(
        self,
        calibration: Mapping[str, object],
    ):
        normalized = validate_calibration_artifact(calibration)
        self.minimum_series_per_domain = int(
            cast(int, normalized["minimum_series_per_domain"])
        )
        self.miscoverage = float(cast(float, normalized["miscoverage"]))
        domain_keys = normalized["domain_keys"]
        records = normalized["series_records"]
        assert isinstance(domain_keys, list)
        assert isinstance(records, list)
        self.domain_keys = tuple(str(key) for key in domain_keys)
        provenance = normalized["provenance"]
        assert isinstance(provenance, Mapping)
        self.held_out_test_series_ids = frozenset(
            cast(list[str], provenance["held_out_test_series_ids"])
        )
        self.held_out_test_series_manifest_sha256 = str(
            provenance["held_out_test_series_manifest_sha256"]
        )
        self.held_out_test_energy_artifact_sha256 = str(
            provenance["held_out_test_energy_artifact_sha256"]
        )
        self.endpoint_fingerprint_sha256 = str(
            provenance["endpoint_fingerprint_sha256"]
        )
        self.domain_schema_sha256 = str(provenance["domain_schema_sha256"])
        self.pair_disagreement_audit_protocol_sha256 = str(
            provenance["pair_disagreement_audit_protocol_sha256"]
        )
        self.allowed_prediction_evidence_sha256 = frozenset(
            cast(
                list[str],
                provenance["allowed_prediction_evidence_sha256"],
            )
        )
        self._errors_by_domain: dict[tuple[str, ...], list[float]] = {}
        for record in records:
            assert isinstance(record, Mapping)
            domain = record["domain"]
            assert isinstance(domain, Mapping)
            key = _domain_key(domain, self.domain_keys)
            self._errors_by_domain.setdefault(key, []).append(
                _finite(
                    record["maximum_pair_error_kcal_mol"],
                    name="maximum pair error",
                )
            )
        self.calibration_series_count = len(records)

    def domain_summary(self, domain: Mapping[str, object]) -> dict[str, object] | None:
        key = _domain_key(domain, self.domain_keys)
        errors = self._errors_by_domain.get(key)
        if errors is None or len(errors) < self.minimum_series_per_domain:
            return None
        try:
            radius = split_conformal_radius(
                errors,
                miscoverage=self.miscoverage,
            )
        except ValueError:
            # Exact-domain support may meet the user-set minimum yet still be
            # too small for the requested finite-sample conformal quantile.
            # Refuse the domain instead of clamping to the largest observed
            # error and silently overstating coverage.
            return None
        return {
            "domain": {name: key[index] for index, name in enumerate(self.domain_keys)},
            "series_count": len(errors),
            "radius_kcal_mol": radius,
            "target_coverage": 1.0 - self.miscoverage,
            "coverage_conditions": (
                "complete calibration series and the new series are exchangeable "
                "within this exact declared domain"
            ),
        }


def _disagreement_block(
    *,
    charge_order_disagreement: bool,
    endpoint_order_disagreement: bool,
) -> tuple[list[str], dict[str, object], dict[str, object]]:
    reasons: list[str] = []
    charge = {
        "order_disagreement": bool(charge_order_disagreement),
        "policy": "force_uncertain" if charge_order_disagreement else "stable",
    }
    endpoint = {
        "order_disagreement": bool(endpoint_order_disagreement),
        "policy": "force_uncertain" if endpoint_order_disagreement else "stable",
    }
    if charge_order_disagreement:
        reasons.append("charge-order-disagreement")
    if endpoint_order_disagreement:
        reasons.append("endpoint-order-disagreement")
    return reasons, charge, endpoint


def _prediction_evidence_context(
    evidence: Mapping[str, object],
    calibrator: SelectiveRankCalibrator,
) -> dict[str, object]:
    """Validate one sealed, label-blind held-out prediction input."""

    value = dict(evidence)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != PREDICTION_EVIDENCE_ARTIFACT_TYPE
        or value.get("content_sha256") != artifact_content_sha256(value)
        or value.get("experimental_test_labels_loaded") is not False
        or value.get("energy_correction_applied") is not False
    ):
        raise ValueError("Selective-ranking prediction evidence is invalid.")
    if value["content_sha256"] not in calibrator.allowed_prediction_evidence_sha256:
        raise ValueError(
            "Selective-ranking prediction evidence was not pre-enumerated."
        )
    series_id = value.get("series_id")
    first_id = value.get("first_id")
    second_id = value.get("second_id")
    if (
        not isinstance(series_id, str)
        or series_id not in calibrator.held_out_test_series_ids
        or not isinstance(first_id, str)
        or not first_id
        or not isinstance(second_id, str)
        or not second_id
        or first_id == second_id
    ):
        raise ValueError("Selective-ranking prediction series/pair is invalid.")
    required_fingerprints = {
        "held_out_test_series_manifest_sha256": (
            calibrator.held_out_test_series_manifest_sha256
        ),
        "test_energy_artifact_sha256": (
            calibrator.held_out_test_energy_artifact_sha256
        ),
        "endpoint_fingerprint_sha256": calibrator.endpoint_fingerprint_sha256,
        "domain_schema_sha256": calibrator.domain_schema_sha256,
        "pair_disagreement_audit_protocol_sha256": (
            calibrator.pair_disagreement_audit_protocol_sha256
        ),
    }
    if any(
        value.get(field) != expected or not _is_sha256(value.get(field))
        for field, expected in required_fingerprints.items()
    ) or not _is_sha256(value.get("pair_disagreement_audit_artifact_sha256")):
        raise ValueError("Selective-ranking prediction provenance disagrees.")
    domain = value.get("domain")
    if not isinstance(domain, Mapping):
        raise ValueError("Selective-ranking prediction domain is invalid.")
    _domain_key(domain, calibrator.domain_keys)
    charge_disagreement = value.get("charge_order_disagreement")
    endpoint_disagreement = value.get("endpoint_order_disagreement")
    if type(charge_disagreement) is not bool or type(endpoint_disagreement) is not bool:
        raise ValueError("Selective-ranking disagreement evidence is invalid.")
    return {
        "series_id": series_id,
        "first_id": first_id,
        "second_id": second_id,
        "domain": dict(domain),
        "predicted_delta_kcal_mol": _finite(
            value.get("predicted_delta_kcal_mol"),
            name="predicted_delta_kcal_mol",
        ),
        "charge_order_disagreement": charge_disagreement,
        "endpoint_order_disagreement": endpoint_disagreement,
        "content_sha256": value["content_sha256"],
    }


def selective_rank_prediction(
    *,
    calibrator: SelectiveRankCalibrator | None,
    evidence: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return a sign decision, interval, or explicit refusal without energy edits.

    Lower free energy is better.  The sealed evidence carries
    `predicted_delta_kcal_mol = G_first - G_second`; therefore a negative
    certified difference favors the first item.  The returned interval is a
    bound on that fixed prediction's unknown true difference, not a calibrated
    or corrected energy.
    """
    output: dict[str, object] = {
        "rank_prediction": None,
        "rank_interval": None,
        "rank_status": STATUS_OUT_OF_DOMAIN,
        "rank_coverage": None,
        "ood_reasons": [],
        "charge_sensitivity": None,
        "endpoint_disagreement": None,
        "predicted_delta_kcal_mol": None,
        "prediction_evidence_sha256": None,
        "energy_correction_applied": False,
    }
    if calibrator is None:
        output["ood_reasons"] = ["no-qualified-series-disjoint-calibration"]
        return output
    if evidence is None:
        output["ood_reasons"] = ["missing-sealed-prediction-evidence"]
        return output
    try:
        context = _prediction_evidence_context(evidence, calibrator)
    except ValueError:
        output["ood_reasons"] = ["prediction-provenance-mismatch"]
        return output
    delta = cast(float, context["predicted_delta_kcal_mol"])
    domain = cast(Mapping[str, object], context["domain"])
    disagreement_reasons, charge, endpoint = _disagreement_block(
        charge_order_disagreement=cast(
            bool,
            context["charge_order_disagreement"],
        ),
        endpoint_order_disagreement=cast(
            bool,
            context["endpoint_order_disagreement"],
        ),
    )
    output["charge_sensitivity"] = charge
    output["endpoint_disagreement"] = endpoint
    output["predicted_delta_kcal_mol"] = delta
    output["prediction_evidence_sha256"] = context["content_sha256"]
    try:
        summary = calibrator.domain_summary(domain)
    except ValueError:
        output["ood_reasons"] = ["invalid-domain-metadata"]
        return output
    if summary is None:
        output["ood_reasons"] = ["insufficient-exact-domain-calibration"]
        return output
    radius = _finite(summary["radius_kcal_mol"], name="conformal radius")
    output["rank_interval"] = [delta - radius, delta + radius]
    output["rank_coverage"] = {
        "target": summary["target_coverage"],
        "series_count": summary["series_count"],
        "radius_kcal_mol": radius,
        "conditions": summary["coverage_conditions"],
    }
    if disagreement_reasons:
        output["rank_status"] = STATUS_UNCERTAIN
        output["ood_reasons"] = disagreement_reasons
        return output
    if delta < -radius:
        output["rank_status"] = STATUS_CERTIFIED
        output["rank_prediction"] = "first_better"
    elif delta > radius:
        output["rank_status"] = STATUS_CERTIFIED
        output["rank_prediction"] = "second_better"
    else:
        output["rank_status"] = STATUS_UNCERTAIN
        output["ood_reasons"] = ["interval-crosses-zero"]
    if output["rank_status"] not in _STATUSES:
        raise RuntimeError("Selective ranking produced an invalid status.")
    return output
