"""Frozen energy-only CHA/PBSA endpoint contract; no experimental labels read."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PANEL = _ROOT / ".omx/benchmarks/route1-torch-full-cha/panel30/manifest-v2.json"
_NATIVE_RECORDS = (
    _ROOT
    / ".omx/benchmarks/route1-cha-m1-accuracy-replay-20260923/development526/records"
)
_PANEL_FILE_SHA256 = "9de15c989a8314baa294f930099bccebf573e390f5cf8931f07ee4665af7d202"
_PANEL_CONTENT_SHA256 = (
    "1e7a585ed625703cb57ac646fb9b887f4b44f91520a7926fbd80757295e54514"
)
_FULL_PRECISION_ORACLE = _ROOT / ".omx/benchmarks/route1-torch-amber-exact-sp"
_FULL_PRECISION_PAYLOAD_SHA256 = (
    "fec19837a795ec869de21838e28a375251e821ac7a2d386785a3693fdd494ef0"
)
_PRINTED_RADIUS_PAYLOAD_SHA256 = (
    "66452ef0619937249cb4e62dd5672e86ac07598a25f219b39d0ecef680295486"
)


def test_exact_sp_target_is_polar_plus_pbsa_cavity_and_dispersion_only():
    if not _PANEL.is_file():
        pytest.skip("optional sealed Route 1 panel is not installed")
    panel_bytes = _PANEL.read_bytes()
    assert hashlib.sha256(panel_bytes).hexdigest() == _PANEL_FILE_SHA256
    panel = json.loads(panel_bytes)
    assert panel["content_sha256"] == _PANEL_CONTENT_SHA256
    assert panel["case_count"] == len(panel["records"]) == 30
    assert panel["blinded"] is False
    assert panel["label_values_not_emitted"] is True
    assert len({record["compound_id"] for record in panel["records"]}) == 30

    for item in panel["records"]:
        native_path = _NATIVE_RECORDS / f"{item['compound_id']}.json"
        if not native_path.is_file():
            pytest.fail(
                f"missing frozen native component record: {item['compound_id']}"
            )
        native_bytes = native_path.read_bytes()
        assert (
            hashlib.sha256(native_bytes).hexdigest() == item["historical_record_sha256"]
        )
        native = json.loads(native_bytes)
        assert native["status"] == "success"
        assert native["compound_id"] == item["compound_id"]
        assert (
            native["am1bcc_charge_vector_sha256"] == item["am1bcc_charge_vector_sha256"]
        )
        assert native["source_mol2_sha256"] == item["source_mol2_sha256"]
        components = native["components_kcal_mol"]
        assert all(math.isfinite(value) for value in components.values())
        selected = (
            components["chagb_polar"]
            + components["pbsa_cavity"]
            + components["pbsa_dispersion"]
        )
        assert math.isclose(
            selected,
            native["predictions_kcal_mol"]["am1bcc_chagb_cavity_dispersion"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        # GBNSR6's separate surface-tension diagnostic is not also added to
        # the selected PBSA cavity+dispersion endpoint.
        assert math.isclose(
            components["chagb_polar"] + components["gbnsr6_surface_tension"],
            native["predictions_kcal_mol"]["am1bcc_chagb_surface_tension"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_full_precision_native_oracle_replays_before_torch_parity():
    summaries = [
        _FULL_PRECISION_ORACLE / name / "summary.json"
        for name in ("native-oracle30-v1", "native-oracle30-v2")
    ]
    if not any(path.is_file() for path in summaries):
        pytest.skip("optional full-precision native oracle is not installed")
    assert all(path.is_file() for path in summaries)
    first, second = (json.loads(path.read_bytes()) for path in summaries)
    assert first["count"] == first["success_count"] == 30
    assert second["count"] == second["success_count"] == 30
    assert first["failures"] == second["failures"] == []
    assert first["ordered_ids"] == second["ordered_ids"]
    assert first["label_values_read"] is second["label_values_read"] is False
    assert (
        first["new_qm_or_charge_generation"]
        is second["new_qm_or_charge_generation"]
        is False
    )

    numeric_payload = []
    for compound_id in first["ordered_ids"]:
        records = []
        for run, summary in (
            ("native-oracle30-v1", first),
            ("native-oracle30-v2", second),
        ):
            directory = _FULL_PRECISION_ORACLE / run / "records" / compound_id
            record_path = directory / "record.json"
            assert (
                hashlib.sha256(record_path.read_bytes()).hexdigest()
                == summary["record_file_sha256"][compound_id]
            )
            record = json.loads(record_path.read_bytes())
            assert record["status"] == "success"
            for name, marker in (
                ("gbnsr6", "MAPLE_GBNSR6_ENERGY_V1"),
                ("pbsa", "MAPLE_PBSA_ENERGY_V1"),
            ):
                output = directory / f"{name}.out"
                assert marker in output.read_text()
                assert (
                    hashlib.sha256(output.read_bytes()).hexdigest()
                    == record[f"{name}_output_sha256"]
                )
            records.append(record)
        left, right = records
        assert left["components_kcal_mol"] == right["components_kcal_mol"]
        assert left["selected_total_kcal_mol"] == right["selected_total_kcal_mol"]
        numeric_payload.append(
            (
                compound_id,
                left["components_kcal_mol"],
                left["selected_total_kcal_mol"],
            )
        )
        historical = json.loads((_NATIVE_RECORDS / f"{compound_id}.json").read_bytes())
        for new_key, old_key in (
            ("polar", "chagb_polar"),
            ("cavity", "pbsa_cavity"),
            ("dispersion", "pbsa_dispersion"),
        ):
            assert (
                abs(
                    left["components_kcal_mol"][new_key]
                    - historical["components_kcal_mol"][old_key]
                )
                <= 5e-5
            )
        assert (
            abs(
                left["selected_total_kcal_mol"]
                - historical["predictions_kcal_mol"]["am1bcc_chagb_cavity_dispersion"]
            )
            <= 1.5e-4
        )
    canonical = json.dumps(
        numeric_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == _FULL_PRECISION_PAYLOAD_SHA256


def test_native_printed_born_diagnostic_is_not_unshifted_full_precision():
    summaries = [
        _FULL_PRECISION_ORACLE / name / "summary.json"
        for name in ("native-rbornstat30-v1", "native-rbornstat30-v2")
    ]
    if not any(path.is_file() for path in summaries):
        pytest.skip("optional native printed-radius diagnostic is not installed")
    assert all(path.is_file() for path in summaries)
    first, second = (json.loads(path.read_bytes()) for path in summaries)
    assert first["count"] == second["count"] == 30
    assert first["ordered_ids"] == second["ordered_ids"]
    assert first["not_unshifted_full_precision"] is True
    assert second["not_unshifted_full_precision"] is True
    rows = []
    for compound_id in first["ordered_ids"]:
        records = []
        for run, summary in (
            ("native-rbornstat30-v1", first),
            ("native-rbornstat30-v2", second),
        ):
            record_path = (
                _FULL_PRECISION_ORACLE / run / "records" / compound_id / "record.json"
            )
            assert (
                hashlib.sha256(record_path.read_bytes()).hexdigest()
                == summary["record_file_sha256"][compound_id]
            )
            record = json.loads(record_path.read_bytes())
            assert record["label_values_read"] is record["new_qm"] is False
            records.append(record)
        left, right = records
        assert (
            left["printed_rinv_per_angstrom_eight_decimals"]
            == right["printed_rinv_per_angstrom_eight_decimals"]
        )
        assert left["polar_kcal_mol"] == right["polar_kcal_mol"]
        rows.append(
            (
                compound_id,
                left["printed_rinv_per_angstrom_eight_decimals"],
                left["polar_kcal_mol"],
            )
        )
    assert sum(len(row[1]) for row in rows) == 417
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == _PRINTED_RADIUS_PAYLOAD_SHA256
