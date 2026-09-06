from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_misin_coarse_grain_asset_protocol.json"
ARTIFACT_PATH = BENCHMARK_DIR / "route1-misin-coarse-grain-asset-audit-2026-07-29.json"
SPEC = importlib.util.spec_from_file_location(
    "route1_misin_coarse_grain_asset_audit",
    BENCHMARK_DIR / "route1_misin_coarse_grain_asset_audit.py",
)
assert SPEC is not None
asset_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(asset_audit)


def _source_model(*, charge: float = 0.0) -> str:
    return f"""%FLAG POINTERS
%FORMAT(10I8)
       1       1
%FLAG CHG
%FORMAT(1PE24.16)
  {charge:.16E}
"""


def _source_input() -> str:
    return """&PARAMETERS
  CLOSUR='hnc',
  TEMPER=298.15, DIEps=2.0,
/
"""


def _source_xvv(*, charge: float = 0.0) -> str:
    return f"""%FLAG POINTERS
%FORMAT(10I8)
   16384       1       1
%FLAG THERMO
%FORMAT(1P5E24.16)
  2.9815000000000000E+02  1.0000000000000000E+00
%FLAG QV
%FORMAT(1P5E24.16)
  {charge:.16E}
"""


def _write_fake_archive(tmp_path: Path, *, charge: float = 0.0) -> tuple[Path, str]:
    archive_path = tmp_path / "jp6b05352_si_002.zip"
    secret_label = "SECRET_EXPERIMENTAL_RESULT_MUST_NOT_ESCAPE"
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        for asset in protocol["candidate_assets"]:
            stem = asset["source_stem"]
            prefix = "si_files/susceptibility_files/"
            archive.writestr(prefix + stem + ".mdl", _source_model(charge=charge))
            archive.writestr(prefix + stem + ".sh", _source_input())
            archive.writestr(prefix + stem + ".xvv", _source_xvv(charge=charge))
        archive.writestr(
            "si_files/solvation_free_energies/results.csv",
            secret_label.encode("ascii") + b"\t\xff\n",
        )
    return archive_path, secret_label


def _write_pinned_protocol(tmp_path: Path, archive_path: Path) -> Path:
    protocol = copy.deepcopy(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))
    data = archive_path.read_bytes()
    protocol["source"]["archive"] = {
        "name": archive_path.name,
        "md5": hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    output = tmp_path / "protocol.json"
    output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output


def test_protocol_freezes_containment_and_eight_candidate_assets() -> None:
    protocol, fingerprint = asset_audit._load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert len(protocol["required_panel_canonical_names"]) == 15
    assert len(protocol["candidate_assets"]) == 8
    assert protocol["containment"] == {
        "experimental_values_loaded": False,
        "label_file_content_read": False,
        "assets_redistributed": False,
        "candidate_is_general_multisolvent_provider": False,
        "candidate_can_count_toward_15_solvent_gate": False,
        "candidate_can_select_or_score_against_mnsol": False,
    }


def test_asset_audit_never_reads_or_emits_label_contents(tmp_path: Path) -> None:
    archive_path, secret_label = _write_fake_archive(tmp_path)
    protocol, fingerprint = asset_audit._load_protocol(
        _write_pinned_protocol(tmp_path, archive_path)
    )

    artifact = asset_audit.audit_archive(protocol, fingerprint, archive_path)
    serialized = json.dumps(artifact, sort_keys=True)

    assert artifact["content_sha256"] == asset_audit.core.artifact_content_sha256(artifact)
    assert artifact["source_archive"]["label_file_content_read"] is False
    assert artifact["source_archive"]["result_csv_member_count_detected"] == 1
    assert artifact["coverage"] == {
        "required_panel_solvents": 15,
        "candidate_assets": 8,
        "missing_required_panel_solvents": [
            "water",
            "ethanol",
            "dimethylformamide",
            "tetrahydrofuran",
            "dichloromethane",
            "hexane",
            "nitrobenzene",
        ],
    }
    assert all(asset["site_count"] == 1 for asset in artifact["candidate_assets"])
    assert all(asset["site_charge"] == 0.0 for asset in artifact["candidate_assets"])
    assert secret_label not in serialized
    assert "results.csv" not in serialized


def test_asset_audit_rejects_non_neutral_source_model(tmp_path: Path) -> None:
    archive_path, _secret_label = _write_fake_archive(tmp_path, charge=0.1)
    protocol, fingerprint = asset_audit._load_protocol(
        _write_pinned_protocol(tmp_path, archive_path)
    )

    with pytest.raises(ValueError, match="site_charge"):
        asset_audit.audit_archive(protocol, fingerprint, archive_path)


def test_committed_asset_audit_is_self_hashed_and_blocked() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert artifact["content_sha256"] == asset_audit.core.artifact_content_sha256(artifact)
    assert artifact["coverage"]["required_panel_solvents"] == 15
    assert artifact["coverage"]["candidate_assets"] == 8
    assert artifact["source_archive"]["result_csv_member_count_detected"] == 21
    assert artifact["source_archive"]["label_file_content_read"] is False
    assert artifact["conclusion"]["status"] == "blocked_for_general_multisolvent_provider"
    assert artifact["conclusion"]["accuracy_claim"] == "none"
    assert artifact["conclusion"]["product_or_runtime_change"] is False
