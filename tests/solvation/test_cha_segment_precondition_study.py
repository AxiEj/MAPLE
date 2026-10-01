"""Self-contained protocol tests for the coordinate-only segment study."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/characterize_cha_segment_preconditions.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("cha_segment_study", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_source():
    displacements = (-0.002, -0.001, -0.0005, 0.0005, 0.001, 0.002)
    rows = []
    for row_index, scale in enumerate((0.96, 0.98, 1.0, 1.02, 1.04)):
        denominators = []
        for denominator_index in range(9):
            event = row_index == 2 and denominator_index == 4
            denominators.append(
                {
                    "atom_index": denominator_index // 3,
                    "axis": denominator_index % 3,
                    "classification": "EVENT" if event else "REGULAR",
                    "event_kinds": ["SIGN_EVENT"] if event else [],
                    "samples": [
                        {
                            "displacement_angstrom": displacement,
                            "weighted_signs_relative_to_center": (
                                "CHANGED" if event and displacement > 0 else "SAME"
                            ),
                        }
                        for displacement in displacements
                    ],
                }
            )
        rows.append(
            {
                "positions_angstrom": [
                    [0.0, 0.4, 0.0],
                    [0.75 * scale, -0.2 * scale, 0.0],
                    [-0.75 * scale, -0.2 * scale, 0.0],
                ],
                "scale": scale,
                "row_status": "EVENT" if row_index == 2 else "REGULAR",
                "stencil": {"denominators": denominators},
            }
        )
    return {"rows": rows}


def _write_inputs(tmp_path):
    source_path = tmp_path / "source.json"
    topology_path = tmp_path / "topology.json"
    source_path.write_text(json.dumps(_synthetic_source()))
    topology_path.write_text(
        json.dumps(
            {
                "content_sha256": "2" * 64,
                "cha_radii_angstrom": [1.88, 1.04, 1.04],
            }
        )
    )
    return source_path, topology_path


def _isolate_external_pins(module, monkeypatch):
    monkeypatch.setattr(module, "_verify_approved_plan_and_handoff", lambda: None)
    monkeypatch.setattr(module, "_source_snapshot", lambda: {"fixture": "frozen"})


class _ChangingReadSurrogate:
    def __init__(self, first: bytes):
        self.first = first
        self.read_count = 0

    def read_bytes(self):
        self.read_count += 1
        return self.first if self.read_count == 1 else b'{"mutated":true}'


def test_case_inventory_retains_all_283_paths_in_frozen_order():
    module = _module()
    cases = module.build_case_inventory(_synthetic_source())
    assert len(cases) == 283
    assert [case["kind"] for case in cases[:5]] == ["singleton"] * 5
    assert [case["kind"] for case in cases[5:275]] == ["stencil"] * 270
    assert [case["kind"] for case in cases[275:]] == ["adjacent-center"] * 8
    assert len({case["case_id"] for case in cases}) == 283
    assert any(case.get("source_row_status") == "EVENT" for case in cases)
    assert all("source_denominator_classification" in case for case in cases[5:275])


def test_preregister_is_atomic_refuses_overwrite_and_rejects_drift(
    tmp_path, monkeypatch
):
    module = _module()
    _isolate_external_pins(module, monkeypatch)
    source_path, topology_path = _write_inputs(tmp_path)
    protocol = tmp_path / "protocol.json"
    module.preregister(source_path, topology_path, protocol)
    payload = json.loads(protocol.read_text())
    assert payload["case_count"] == 283
    assert payload["budget"] == {
        "max_depth": 12,
        "max_integer_bits": 8192,
        "max_nodes": 8191,
    }
    assert payload["approved_plan_sha256"] == (
        "bb0e3081e4ebe1c56b57f7a2db1987fbccd379a2e05349c06c9175d297b54bb6"
    )
    assert payload["accepted_handoff_sha256"] == (
        "2f0c76f8911a52ecde4b76fa77df29459e32b3ddda9a0cdc9a08348afc34f7aa"
    )
    assert any(item["name"] == "size_prefactor" for item in payload["constant_ledger"])
    assert all("mathematical_use" in item for item in payload["constant_ledger"])
    with pytest.raises(FileExistsError):
        module.preregister(source_path, topology_path, protocol)

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("validated JSON must be consumed from its hashed bytes")
        ),
    )
    validated = module.verify_protocol_inputs(payload, source_path, topology_path)
    assert validated.source == _synthetic_source()

    source_surrogate = _ChangingReadSurrogate(source_path.read_bytes())
    topology_surrogate = _ChangingReadSurrogate(topology_path.read_bytes())
    surrogate_validated = module.verify_protocol_inputs(
        payload, source_surrogate, topology_surrogate
    )
    assert surrogate_validated.source == _synthetic_source()
    assert source_surrogate.read_count == topology_surrogate.read_count == 1

    changed = tmp_path / "changed.json"
    changed.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source SHA256"):
        module.verify_protocol_inputs(payload, changed, topology_path)


def test_external_protocol_file_pin_precedes_internal_rehash(tmp_path, monkeypatch):
    module = _module()
    _isolate_external_pins(module, monkeypatch)
    source_path, topology_path = _write_inputs(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    module.preregister(source_path, topology_path, protocol_path)
    frozen_file_sha = module.sha256_file(protocol_path)
    payload = json.loads(protocol_path.read_text())
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("protocol must be parsed from its externally hashed bytes")
        ),
    )
    assert (
        module.load_externally_pinned_protocol(protocol_path, frozen_file_sha)
        == payload
    )
    payload["phase"] = "tampered-but-internally-rehashed"
    payload["status_rules"]["optimizer_eligible"] = True
    payload["dependency_ledger"]["omitted_weighted_sign"] = "tampered"
    payload["constant_ledger"][0]["units"] = "tampered"
    payload["content_sha256"] = module.content_sha256(payload)
    protocol_path.write_bytes(module.canonical_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="external protocol file SHA256"):
        module.load_externally_pinned_protocol(protocol_path, frozen_file_sha)


def test_post_run_hash_recheck_detects_mutation(tmp_path):
    module = _module()
    protocol, source, topology = (tmp_path / name for name in ("p", "s", "t"))
    for path, payload in zip(
        (protocol, source, topology), (b"protocol", b"source", b"topology"), strict=True
    ):
        path.write_bytes(payload)
    before = {
        "protocol": module.sha256_file(protocol),
        "source_result": module.sha256_file(source),
        "topology": module.sha256_file(topology),
    }
    assert module.recheck_run_input_hashes(protocol, source, topology, before) == before
    source.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="source_result"):
        module.recheck_run_input_hashes(protocol, source, topology, before)


def test_run_rejects_code_drift_after_protocol_validation(tmp_path, monkeypatch):
    module = _module()
    _isolate_external_pins(module, monkeypatch)
    source, topology = _write_inputs(tmp_path)
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "result.json"
    module.preregister(source, topology, protocol)
    snapshots = iter(
        ({"fixture": "frozen"}, {"fixture": "changed"}, {"fixture": "changed"})
    )
    monkeypatch.setattr(module, "_source_snapshot", lambda: next(snapshots))

    def topology_must_not_start(*args, **kwargs):
        pytest.fail("Code drift must be rejected before topology or characterization")

    monkeypatch.setattr(
        module.ContinuumChaTopology, "from_mapping", topology_must_not_start
    )
    with pytest.raises(ValueError, match="preregistered source identity"):
        module.run(
            protocol,
            source,
            topology,
            output,
            expected_protocol_sha256=module.sha256_file(protocol),
        )
    assert not output.exists()
