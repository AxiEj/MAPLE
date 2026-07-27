from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_mnsol_macepolar_solvent_diversity_canary as runner  # noqa: E402  # pyright: ignore[reportMissingImports]


PREREG = (
    BENCHMARK_DIR
    / "route2-mnsol-macepolar-solvent-diversity-canary-prereg-v1.json"
)
PARENT = BENCHMARK_DIR / "route2-mnsol-pilot-selection-v1.json"


def _fake_full_selection(parent: dict) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            canonical_solvent=record["canonical_solvent"],
            opaque_record_id=record["opaque_record_id"],
        )
        for record in parent["selected_records"]
    ]


def test_canary_prereg_freezes_three_parent_records_and_fingerprint():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))

    selected = runner._validated_canary_selection(
        prereg,
        _fake_full_selection(parent),
        protocol_id=prereg["protocol_id"],
        protocol_fingerprint=parent["protocol_fingerprint"],
        parent_selection_fingerprint=parent["selection_fingerprint"],
    )

    assert [index for index, _record in selected] == [0, 3, 9]
    assert [
        record.canonical_solvent for _index, record in selected
    ] == ["water", "dimethylsulfoxide", "hexane"]
    assert runner.CLAIM_BOUNDARY.endswith("TS, scan, or MD.")


def test_canary_prereg_fails_closed_on_parent_or_record_drift():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    selection = _fake_full_selection(parent)

    with pytest.raises(ValueError, match="parent_selection_fingerprint"):
        runner._validated_canary_selection(
            prereg,
            selection,
            protocol_id=prereg["protocol_id"],
            protocol_fingerprint=parent["protocol_fingerprint"],
            parent_selection_fingerprint="0" * 64,
        )

    tampered = copy.deepcopy(prereg)
    tampered["selected_records"][1]["opaque_record_id"] = "0" * 64
    tampered["selection_fingerprint"] = runner._selection_fingerprint(
        tampered["selected_records"]
    )
    with pytest.raises(ValueError, match="record no longer matches"):
        runner._validated_canary_selection(
            tampered,
            selection,
            protocol_id=prereg["protocol_id"],
            protocol_fingerprint=parent["protocol_fingerprint"],
            parent_selection_fingerprint=parent["selection_fingerprint"],
        )


def test_canary_keeps_row_and_summary_outputs_private():
    private = ROOT / ".omx/test-mace-canary/private.json"
    summary = ROOT / ".omx/test-mace-canary/summary.json"
    work = ROOT / ".omx/test-mace-canary/work"

    assert runner._validated_output_paths(
        private_output=private,
        summary_output=summary,
        work_dir=work,
    ) == (private.resolve(), summary.resolve(), work.resolve())

    with pytest.raises(ValueError, match="Derived MNSol canary summary"):
        runner._validated_output_paths(
            private_output=private,
            summary_output=ROOT / "docs/canary-summary.json",
            work_dir=work,
        )


def test_canary_prereg_source_hash_path_is_repository_relative(tmp_path):
    assert runner._repo_relative_path(
        PREREG.relative_to(ROOT),
        kind="MNSol canary preregistration",
    ) == PREREG.relative_to(ROOT).as_posix()

    with pytest.raises(ValueError, match="must remain inside"):
        runner._repo_relative_path(
            tmp_path / "canary.json",
            kind="MNSol canary preregistration",
        )
