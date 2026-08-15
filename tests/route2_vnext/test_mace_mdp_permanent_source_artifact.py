from __future__ import annotations

import hashlib
import json
from pathlib import Path

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT / "docs/route2/evidence/mace-mdp-permanent-source-terminal-64837473"
)
RUN1 = EVIDENCE / "run1.json"
RUN2 = EVIDENCE / "run2.json"
EXECUTION_HEAD = "64837473053d39bfae5b61788d7a4edce5d4502b"
MEASUREMENT_SHA256 = "9c3801fb87c5bc7e3fe1396bf31afad41cc397288fe6b9e6f2824ebf6910664c"
CHECKPOINT_SHA256 = "126f8d1602549e6fa0df775c701a5119ddeb0e3738202af8e7aa736de6c2b692"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "run1.json": "82c99b78b59bbc568376a0f4b57fb35b8664e1ba8e8e94ce73d68ed81e483610",
    "run1.stdout": "2db32d8915f0c1e42b3e80f9ed458f8bd0719b1cee93479c615f1ae64fd44efb",
    "run1.stderr": "e60fe6ca96e6fa7c7ec196843bb1c80f768d416bdfefd3faad2c371dd4aac719",
    "run2.json": "1437b9801d06dcfebb69cffe91a33341e439508c80a5346227c7cdd0312e4a98",
    "run2.stdout": "a38538900fcf53b9f732e54d8c3c708c39866bc65ba50dcbcd94eacbcec52848",
    "run2.stderr": "e60fe6ca96e6fa7c7ec196843bb1c80f768d416bdfefd3faad2c371dd4aac719",
}


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurement(artifact: dict[str, object]) -> dict[str, object]:
    return {
        key: artifact[key]
        for key in ("protocol", "records", "aggregate", "decision")
    }


def test_unchanged_mace_mdp_permanent_source_fails_frozen_gate() -> None:
    artifact = _load(RUN1)
    assert artifact["schema_version"] == (
        "route2-mace-mdp-permanent-source-pcmsolver-four-v1"
    )
    assert artifact["status"] == (
        "mace-mdp-permanent-source-fails-frozen-four-case-gate"
    )
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["external_assets"]["mace_mdp_checkpoint"]["sha256"] == (
        CHECKPOINT_SHA256
    )
    assert artifact["aggregate"] == {
        "case_pass_count": 3,
        "maximum_dipole_relative_l2_error": 0.036103867966070964,
        "maximum_polarization_energy_absolute_error_kcal_per_mol": (
            1.8238626327341645
        ),
        "maximum_surface_mep_area_weighted_relative_l2_error": (
            0.5947660374548218
        ),
        "mean_polarization_energy_absolute_error_kcal_per_mol": (
            0.9419085578663388
        ),
        "record_count": 4,
    }
    records = {
        record["compound_id"]: {
            "passed": record["case_passed"],
            "energy_error": record[
                "polarization_energy_absolute_error_kcal_per_mol"
            ],
        }
        for record in artifact["records"]
    }
    assert records == {
        "mobley_3034976": {"passed": True, "energy_error": 0.7640562416000839},
        "mobley_3053621": {"passed": True, "energy_error": 0.8946333000649167},
        "mobley_352111": {"passed": False, "energy_error": 1.8238626327341645},
        "mobley_3867265": {"passed": True, "energy_error": 0.28508205706618966},
    }
    assert artifact["decision"] == {
        "additional_source_patch_authorized": False,
        "broader_static_mep_panel_authorized": False,
        "independent_variational_polarization_kkt_authorized": False,
        "mace_mdp_permanent_source_necessary_gate_passed": False,
        "public_capability_admitted": False,
    }
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert canonical_json_sha256(_measurement(artifact)) == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_mace_mdp_terminal_measurement_replays_exactly() -> None:
    first = _load(RUN1)
    second = _load(RUN2)
    assert _measurement(first) == _measurement(second)
    for artifact in (first, second):
        assert artifact["execution_git_head"] == EXECUTION_HEAD
        assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
        assert artifact["capabilities"] == NO_CAPABILITIES
        assert_source_files_match_execution_commit(ROOT, artifact)
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
