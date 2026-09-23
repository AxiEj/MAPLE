"""Fixed-charge Torch CHA input identity; no Amber executable is needed here."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaCoordinates,
    ContinuumChaTopology,
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _charge_hash(charges):
    payload = json.dumps(charges, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _artifact():
    source = [0.4, -0.4]
    effective = [0.4000000002, -0.4000000002]
    payload = {
        "schema_version": 1,
        "profile": "chagb-r6-pbsa-continuum-v1",
        "atom_ids": [1, 2],
        "atom_names": ["C1", "O1"],
        "elements": ["C", "O"],
        "atomic_numbers": [6, 8],
        "gaff2_types": ["c3", "o"],
        "bonds": [[0, 1, "1"]],
        "declared_charge_e": 0,
        "source_charges_e": source,
        "effective_charges_e": effective,
        "source_charges_sha256": _charge_hash(source),
        "effective_charges_sha256": _charge_hash(effective),
        "source_mol2_sha256": "0" * 64,
        "prepared_prmtop_sha256": "1" * 64,
        "parameter_source_sha256": "2" * 64,
        "serialization_profile": "ambertools26-prmtop-charge-5e16.8-v1",
        "bondi_radii_angstrom": [1.7, 1.5],
        "cha_radii_angstrom": [2.12, 1.88],
        "lj_rmin_angstrom": [1.9069, 1.7683],
        "lj_epsilon_kcal_mol": [0.1078, 0.152],
    }
    payload["content_sha256"] = _hash(payload)
    return payload


def _load(payload, **pins):
    return ContinuumChaTopology.from_mapping(
        payload,
        expected_content_sha256=pins.pop(
            "expected_content_sha256", payload["content_sha256"]
        ),
        **pins,
    )


def test_topology_is_immutable_and_coordinates_are_dynamic():
    payload = _artifact()
    topology = _load(payload)
    assert topology.content_sha256 == payload["content_sha256"]
    assert topology.atom_ids == (1, 2)
    assert topology.atomic_numbers == (6, 8)
    assert topology.source_charges_e != topology.effective_charges_e
    fixed = topology.tensors(
        expected_content_sha256=payload["content_sha256"], device="cpu"
    )
    assert fixed.charges_e.dtype == torch.float64
    assert fixed.charges_e.tolist() == payload["effective_charges_e"]
    assert fixed.charges_e.requires_grad is False
    fixed.assert_current()

    first = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    second = first + torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    x1 = ContinuumChaCoordinates.from_tensor(first, topology)
    x2 = ContinuumChaCoordinates.from_tensor(second, topology)
    assert x1.positions_angstrom is not first
    assert x2.positions_angstrom is not second
    assert x1.content_sha256 != x2.content_sha256
    assert topology.content_sha256 == payload["content_sha256"]
    assert torch.autograd.grad(x2.positions_angstrom.sum(), first)[0].shape == (2, 3)


@pytest.mark.parametrize(
    "change",
    [
        "self_hash",
        "source_hash",
        "effective_hash",
        "charge_drift",
        "nonneutral",
        "rmin",
        "epsilon",
        "atom_ids",
        "bonds",
        "duplicate_bond_pair",
        "atomic_number",
        "schema_bool",
        "direct_drift",
        "profile",
        "extra_field",
    ],
)
def test_bad_topology_fails_closed(change):
    payload = _artifact()
    if change == "self_hash":
        payload["cha_radii_angstrom"][0] += 0.01
    elif change == "source_hash":
        payload["source_charges_sha256"] = "f" * 64
    elif change == "effective_hash":
        payload["effective_charges_sha256"] = "f" * 64
    elif change == "charge_drift":
        payload["effective_charges_e"][0] += 0.01
        payload["effective_charges_e"][1] -= 0.01
        payload["effective_charges_sha256"] = _charge_hash(
            payload["effective_charges_e"]
        )
    elif change == "nonneutral":
        payload["source_charges_e"][0] += 0.02
        payload["source_charges_sha256"] = _charge_hash(payload["source_charges_e"])
    elif change == "rmin":
        payload["lj_rmin_angstrom"][0] = 0.0
    elif change == "epsilon":
        payload["lj_epsilon_kcal_mol"][0] = -1.0
    elif change == "atom_ids":
        payload["atom_ids"] = [2, 1]
    elif change == "bonds":
        payload["bonds"] = [[0, 2, "1"]]
    elif change == "duplicate_bond_pair":
        payload["bonds"] = [[0, 1, "1"], [0, 1, "ar"]]
    elif change == "atomic_number":
        payload["atomic_numbers"] = [8, 6]
    elif change == "schema_bool":
        payload["schema_version"] = True
    elif change == "direct_drift":
        payload["serialization_profile"] = "direct-fixed-charge-v1"
    elif change == "profile":
        payload["profile"] = "wrong-profile"
    else:
        payload["unexpected"] = True
    if change != "self_hash":
        payload["content_sha256"] = _hash(
            {k: v for k, v in payload.items() if k != "content_sha256"}
        )
    with pytest.raises((ValueError, TypeError)):
        _load(payload)


def test_coordinates_reject_wrong_order_device_dtype_and_nonfinite():
    topology = _load(_artifact())
    valid = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
    for positions, atom_ids in (
        (valid, (2, 1)),
        (valid.float(), (1, 2)),
        (torch.full((2, 3), float("nan"), dtype=torch.float64), (1, 2)),
        (valid[:1], (1,)),
    ):
        with pytest.raises((ValueError, TypeError)):
            ContinuumChaCoordinates.from_tensor(positions, topology, atom_ids=atom_ids)


def test_charge_source_binding_and_no_amber_process(monkeypatch):
    import subprocess

    def forbidden(*_args, **_kwargs):
        raise AssertionError("No external process is allowed in the Torch input path.")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setenv("PATH", "")
    payload = _artifact()
    topology = _load(
        payload,
        expected_source_charge_sha256=payload["source_charges_sha256"],
    )
    assert (
        topology.tensors(
            expected_content_sha256=payload["content_sha256"], device="cpu"
        ).charges_e.tolist()
        == payload["effective_charges_e"]
    )
    with pytest.raises(ValueError):
        _load(payload, expected_source_charge_sha256="3" * 64)


@pytest.mark.parametrize(
    "field", ["gaff2_types", "cha_radii_angstrom", "effective_charges_e"]
)
def test_external_content_pin_rejects_resealed_parameter_or_charge_change(field):
    payload = _artifact()
    original_pin = payload["content_sha256"]
    if field == "gaff2_types":
        payload[field][0] = "unsupported-type"
    elif field == "cha_radii_angstrom":
        payload[field][0] += 0.5
    else:
        payload[field] = [0.40000005, -0.40000005]
        payload["effective_charges_sha256"] = _charge_hash(payload[field])
    payload["content_sha256"] = _hash(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="external|expected|pin"):
        _load(payload, expected_content_sha256=original_pin)


def test_coordinate_snapshot_and_live_fingerprint():
    topology = _load(_artifact())
    original = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    coordinates = ContinuumChaCoordinates.from_tensor(original, topology)
    original_value = coordinates.positions_angstrom.detach().clone()
    with torch.no_grad():
        original[0, 0] += 0.2
    assert torch.equal(coordinates.positions_angstrom, original_value)
    coordinates.assert_current()
    with torch.no_grad():
        coordinates.positions_angstrom[0, 0] += 0.1
    with pytest.raises(ValueError, match="coordinate.*hash"):
        coordinates.assert_current()


def test_replaced_topology_and_mutated_parameter_tensor_fail_closed():
    payload = _artifact()
    topology = _load(payload)
    with pytest.raises((TypeError, ValueError)):
        topology.tensors()
    with pytest.raises(ValueError, match="hash|identity|pin"):
        replace(topology, effective_charges_e=(0.45, -0.45))
    fixed = topology.tensors(expected_content_sha256=payload["content_sha256"])
    with torch.no_grad():
        fixed.charges_e[0] += 0.01
    with pytest.raises(ValueError, match="parameter.*hash|tensor.*hash"):
        fixed.assert_current()
