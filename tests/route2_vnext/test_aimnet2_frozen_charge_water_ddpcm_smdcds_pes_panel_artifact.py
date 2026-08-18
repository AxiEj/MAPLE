from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-panel-20f9c65c"
)
EXECUTION_HEAD = "20f9c65c3fc238089bb4ea47512e8c136664bcf3"
MEASUREMENT_SHA256 = (
    "61512993a63cc7bf04d4a3d15a288c806eabc96be104e33373b802f88f62322e"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_total_smd_pes_panel_checksums_and_source_binding():
    checksums = {
        name: digest
        for digest, name in (
            line.split(maxsplit=1)
            for line in (EVIDENCE / "SHA256SUMS")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    assert set(checksums) == {"README.md", "primary-panel.json", "replay-panel.json"}
    for name, digest in checksums.items():
        assert _sha256(EVIDENCE / name) == digest

    primary = _load("primary-panel.json")
    replay = _load("replay-panel.json")
    for artifact in (primary, replay):
        assert artifact["execution_git_head"] == EXECUTION_HEAD
        assert artifact["aggregate_measurement_sha256"] == MEASUREMENT_SHA256
        assert artifact["continuum_kind"] == "harmonic-ddpcm-water"
        assert artifact["nonpolar_kind"] == "pyscf-smd-cds-water"
        assert artifact["status"] == "diagnostic-panel-failed-not-admitted"
        assert artifact["capabilities"] == {
            "E": False,
            "F": False,
            "H": False,
            "M": False,
            "V": False,
        }
    assert_source_files_match_execution_commit(ROOT, primary)


def test_total_smd_pes_panel_replays_exactly_and_remains_closed():
    primary = _load("primary-panel.json")
    replay = _load("replay-panel.json")
    assert replay["panel_summary"] == primary["panel_summary"]

    summary = primary["panel_summary"]
    assert summary["molecule_count"] == 17
    assert summary["geometry_count"] == 51
    assert summary["directional_record_count"] == 153
    assert summary["directional_sample_count"] == 459
    assert summary["diagnostic_gates_passed"] is False
    assert summary["failed_molecule_ids"] == [
        "methanol",
        "methane",
        "dimethyl-ether",
        "acetic-acid",
        "ethylamine",
    ]
    assert summary["minimum_continuum_event_margin_A"] == pytest.approx(
        0.002235739521015301,
        abs=1.0e-15,
    )
    assert summary["minimum_sphere_tangency_margin_A"] == pytest.approx(
        0.011178247440158717,
        abs=1.0e-15,
    )
    assert summary["maximum_directional_absolute_error_eV_per_A"] == pytest.approx(
        0.23707138702659591,
        abs=1.0e-14,
    )

    passing = [
        shard
        for shard in summary["shard_summaries"]
        if shard["diagnostic_gates_passed"]
    ]
    assert len(passing) == 12
    assert max(
        shard["maximum_directional_absolute_error_eV_per_A"]
        for shard in passing
    ) == pytest.approx(5.552903772709783e-05, abs=1.0e-15)
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False
