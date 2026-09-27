"""Portable native numeric cavity parity; no .omx assets or Amber installation."""

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_pbsa_exact_cavity import (
    cavity_from_native_grid,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "data/pbsa_cavity_oracle_v1.json"
CONTENT_SHA = "83bd0d3241b59fef9e77f697d167e77bc0e52f04ac7cf69a600fddf025b06f05"


def fixture():
    artifact = json.loads(FIXTURE.read_bytes())
    payload = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == CONTENT_SHA
    )
    assert artifact["content_sha256"] == CONTENT_SHA
    assert (
        artifact["label_reads"]
        is artifact["new_qm"]
        is artifact["upstream_source_included"]
        is False
    )
    assert len(artifact["cases"]) == 6
    return artifact


def test_portable_native_fixture_is_pinned_to_its_projection():
    artifact = fixture()
    script = ROOT / "docs/implicit-solvation/benchmarks/freeze_pbsa_cavity_oracle.py"
    assert (
        hashlib.sha256(script.read_bytes()).hexdigest() == artifact["generator_sha256"]
    )
    for case in artifact["cases"]:
        assert (
            case["native_pbsa_binary_sha256"]
            == "6b35846e397288bcf8ce8aa93a5c7721d58804cda1612fe2caaf40e8ca6d0363"
        )
        assert len(case["native_prmtop_sha256"]) == 64


@pytest.mark.parametrize("case_index", range(6))
def test_cavity_matches_portable_full_precision_native_oracle(case_index):
    case = fixture()["cases"][case_index]
    result = cavity_from_native_grid(
        torch.tensor(case["positions_angstrom"], dtype=torch.float64),
        torch.tensor(case["lj_rmin_angstrom"], dtype=torch.float64),
    )
    assert result.energy_kcal_mol.item() == pytest.approx(
        case["native_cavity_kcal_mol"], rel=0.0, abs=1e-10
    )
    assert result.supports_forces is False
