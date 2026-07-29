from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_solvent_asset import (
    V0_DEFAULT_SOLVENT_IDS,
    V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION,
    load_route2_v0_frozen_solvent_registry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xvv() -> str:
    return """%VERSION  VERSION_STAMP = V0001.001
%FLAG POINTERS
%FORMAT(10I8)
       4       2       1
%FLAG THERMO
%FORMAT(1P5E24.16)
  2.9800000000000000E+02  7.8497000000000000E+01  0.0  1.0 2.0000000000000000E+00 1.0000000000000000E+00
%FLAG ATOM_NAME
%FORMAT(20A4)
O
H1
%FLAG MTV
%FORMAT(10I8)
       1       2
%FLAG RHOV
%FORMAT(1P5E24.16)
  3.3000000000000000E-02  6.6000000000000000E-02
%FLAG QV
%FORMAT(1P5E24.16)
 -2.0000000000000000E+00  1.0000000000000000E+00
%FLAG XVV
%FORMAT(1P5E24.16)
  0.0
"""


def _cvv() -> str:
    charges = (-2.0, 1.0)
    rows = [
        "#RISM1D ATOM-ATOM INTERACTIONS: DIRECT CORRELATION VS. SEPARATION [A]",
        "#    SEPARATION          H1:O             O:O             H1:H1",
    ]
    for radius in (0.0, 2.0, 4.0, 6.0):
        kernel = (
            2.0 / math.sqrt(math.pi) if radius == 0.0 else math.erf(radius) / radius
        )
        rows.append(
            " ".join(
                f"{value:.16E}"
                for value in (
                    radius,
                    -charges[1] * charges[0] * kernel,
                    -charges[0] * charges[0] * kernel,
                    -charges[1] * charges[1] * kernel,
                )
            )
        )
    return "\n".join(rows) + "\n"


def _file_entry(path: Path, *, root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
    }


def _write_manifest(tmp_path: Path) -> tuple[Path, dict]:
    for name, contents in {
        "model/cSPCE.mdl": "frozen cSPCE site model\n",
        "bulk/cSPCE.inp": "frozen 1D-RISM input\n",
        "bulk/cSPCE.xvv": _xvv(),
        "bulk/cSPCE.cvv": _cvv(),
        "bulk/cSPCE.thermo": "frozen thermodynamic output\n",
        "short-range/cSPCE-source.txt": "independent short-range source\n",
        "provenance/cSPCE.txt": "no target solvation labels used\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    def entry(name: str) -> dict[str, str]:
        return _file_entry(tmp_path / name, root=tmp_path)

    payload = {
        "protocol_id": V0_FROZEN_SOLVENT_ASSET_CONSTRUCTION,
        "schema_version": 1,
        "assets": [
            {
                "solvent_id": "water",
                "model": {
                    "family": "ambertools-rism1d",
                    "identifier": "cSPCE-test-control",
                },
                "state": {"temperature_kelvin": 298.0, "pressure_bar": 1.0},
                "liquid_convention": {
                    "closure": "PSE3",
                    "standard_state": "1M solute / pure-liquid solvent",
                    "pressure_definition": "source bulk thermodynamic pressure",
                    "partial_molar_volume_definition": "source functional derivative",
                    "sign_convention": "subtract P times partial molar volume",
                },
                "bulk_correlation": {
                    "xvv": entry("bulk/cSPCE.xvv"),
                    "cvv": entry("bulk/cSPCE.cvv"),
                    "coulomb_tail_start_angstrom": 6.0,
                    "coulomb_tail_tolerance_dimensionless": 1.0e-12,
                },
                "source_files": {
                    "site_model": entry("model/cSPCE.mdl"),
                    "rism1d_input": entry("bulk/cSPCE.inp"),
                    "thermodynamic_output": entry("bulk/cSPCE.thermo"),
                    "short_range_interaction": entry("short-range/cSPCE-source.txt"),
                    "provenance_statement": entry("provenance/cSPCE.txt"),
                },
                "provenance": {
                    "target_solvation_labels_used": False,
                    "excluded_target_label_sets": [
                        "mnsol",
                        "freesolv",
                        "development",
                        "confirmation",
                        "blind",
                    ],
                },
            }
        ],
    }
    manifest = tmp_path / "route2-v0-solvent-asset-v1.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest, payload


def test_frozen_solvent_asset_binds_hashed_bulk_and_liquid_provenance(tmp_path):
    manifest, _ = _write_manifest(tmp_path)

    registry = load_route2_v0_frozen_solvent_registry(manifest)
    asset = registry.asset_for("water")

    assert asset.model_identifier == "cSPCE-test-control"
    assert asset.bulk_direct_correlation.metadata.site_names == ("O", "H1")
    assert asset.bulk_direct_correlation.metadata.temperature_kelvin == pytest.approx(
        298.0
    )
    assert asset.file_for("short_range_interaction").path.name == "cSPCE-source.txt"
    assert set(asset.excluded_target_label_sets) == {
        "mnsol",
        "freesolv",
        "development",
        "confirmation",
        "blind",
    }
    registry.verify_integrity()

    with pytest.raises(ValueError, match="missing default assets"):
        registry.require_default_solvent_panel()
    assert len(V0_DEFAULT_SOLVENT_IDS) == 11


def test_frozen_solvent_asset_rejects_mutation_and_target_label_use(tmp_path):
    manifest, payload = _write_manifest(tmp_path)
    registry = load_route2_v0_frozen_solvent_registry(manifest)

    (tmp_path / "model/cSPCE.mdl").write_text("changed model\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.verify_integrity()
    with pytest.raises(ValueError, match="hash mismatch"):
        load_route2_v0_frozen_solvent_registry(manifest)

    payload["assets"][0]["provenance"]["target_solvation_labels_used"] = True
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly false"):
        load_route2_v0_frozen_solvent_registry(manifest)


def test_frozen_solvent_asset_rejects_missing_source_or_path_escape(tmp_path):
    manifest, payload = _write_manifest(tmp_path)
    del payload["assets"][0]["source_files"]["short_range_interaction"]
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="short_range_interaction"):
        load_route2_v0_frozen_solvent_registry(manifest)

    manifest, payload = _write_manifest(tmp_path / "separate")
    payload["assets"][0]["source_files"]["site_model"]["path"] = "../escape.mdl"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="relative path inside the asset root"):
        load_route2_v0_frozen_solvent_registry(manifest)
