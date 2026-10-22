"""Charged-input domain contracts for the official FP32 MACE-POLAR adapter."""

from __future__ import annotations

import json

import pytest
import torch
from ase import Atoms

from maple.solvation.models.base import ModelDomain
import maple.solvation.models.mace_polar_fp32 as fp32_module


SUPPORTED_NUMBERS = (1, 6, 7, 8)


def _atoms(*, charge=0, multiplicity=1, info=None) -> Atoms:
    metadata = (
        {"charge": charge, "mult": multiplicity} if info is None else dict(info)
    )
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]], info=metadata)


@pytest.mark.parametrize("charge", [-1, 1, 2])
def test_unbounded_charge_domain_accepts_integer_ions(charge):
    domain = ModelDomain(SUPPORTED_NUMBERS, None, (1,))

    assert domain.validate_atoms(_atoms(charge=charge)) == (charge, 1)


def test_unbounded_charge_domain_keeps_singlet_restriction():
    domain = ModelDomain(SUPPORTED_NUMBERS, None, (1,))

    with pytest.raises(ValueError, match="spin multiplicity 2"):
        domain.validate_atoms(_atoms(charge=1, multiplicity=2))


def test_unbounded_charge_metadata_is_explicit_json_null():
    domain = ModelDomain(SUPPORTED_NUMBERS, None, (1,))

    assert domain.metadata()["total_charge_range"] is None
    assert json.loads(json.dumps(domain.metadata()))["total_charge_range"] is None


@pytest.mark.parametrize(
    "info, error_type, match",
    [
        ({"charge": 1.0, "mult": 1}, TypeError, "must be an integer"),
        ({"charge": True, "mult": 1}, TypeError, "must be an integer"),
        (
            {"charge": 1, "total_charge": -1, "mult": 1},
            ValueError,
            "conflicting atoms.info aliases",
        ),
    ],
)
def test_unbounded_charge_domain_still_rejects_invalid_charge_metadata(
    info, error_type, match
):
    domain = ModelDomain(SUPPORTED_NUMBERS, None, (1,))

    with pytest.raises(error_type, match=match):
        domain.validate_atoms(_atoms(info=info))


@pytest.mark.parametrize("charge", [-1, 1, 2])
def test_historical_neutral_domain_still_rejects_ions(charge):
    domain = ModelDomain(SUPPORTED_NUMBERS, (0, 0), (1,))

    with pytest.raises(ValueError, match=r"outside model domain \[0, 0\]"):
        domain.validate_atoms(_atoms(charge=charge))


class _Float32Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))


class _Calculator:
    model = _Float32Model()
    dtype = torch.float32
    atomic_numbers = SUPPORTED_NUMBERS
    device = torch.device("cpu")


def test_official_fp32_graph_base_removes_only_charge_bound(monkeypatch, tmp_path):
    checkpoint = tmp_path / "official-test-double.model"
    checkpoint.write_bytes(b"test-double")
    monkeypatch.setattr(fp32_module, "_calculator_source_files", lambda value: ())

    graph = fp32_module._FP32GraphBase(_Calculator(), checkpoint)

    assert graph.domain == ModelDomain(SUPPORTED_NUMBERS, None, (1,))
    assert graph.provenance["domain"]["total_charge_range"] is None
    assert json.loads(json.dumps(dict(graph.provenance)))["domain"][
        "total_charge_range"
    ] is None
    assert graph.domain.validate_atoms(_atoms(charge=-1)) == (-1, 1)
    assert graph.domain.validate_atoms(_atoms(charge=2)) == (2, 1)
    with pytest.raises(ValueError, match="spin multiplicity 2"):
        graph.domain.validate_atoms(_atoms(charge=1, multiplicity=2))
