from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from maple.function.solvfe import (
    SOLPROPMIX_CHECKPOINT_SHA256,
    SOLPROPMIX_PRECISION_SEMANTICS,
    SOLPROPMIX_SOURCE_REVISION,
    SOLPROPMIX_SOURCE_TREE,
    SOLPROPMIX_SOURCE_URL,
    SOLPROPMIX_STATIC_DATA_HASHES,
    SolPropMixConfigError,
    SolPropMixPropertyAdapter,
    SolPropMixRuntimeError,
)
from maple.function.solvfe import _solpropmix_worker as worker
from maple.function.solvfe.solpropmix_property import (
    _SnapshotEvidence,
    _normalize_origin,
    _normalize_solvents,
    _temperature_adjusted,
    _verify_file,
)


def _adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SolPropMixPropertyAdapter:
    roots = [tmp_path / name for name in ("source", "static", "weights")]
    for root in roots:
        root.mkdir()
    monkeypatch.setattr(
        SolPropMixPropertyAdapter, "assert_artifact_identity", lambda self: None
    )
    return SolPropMixPropertyAdapter(*roots)


def _payload(request: dict[str, object]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index in range(10):
        g298 = -10.0 + index * 0.2
        h298 = -15.0 + index * 0.3
        adjusted = _temperature_adjusted(g298, h298, float(request["temperature"]))
        rows.append(
            {
                "model_index": index,
                "g298": g298,
                "h298": h298,
                "g_temperature": adjusted,
            }
        )
    values = [float(row["g_temperature"]) for row in rows]
    mean = math.fsum(values) / 10
    worker_hash = "a" * 64
    snapshot = {
        "worker_sha256": worker_hash,
        "static_sha256": dict(SOLPROPMIX_STATIC_DATA_HASHES),
        "checkpoint_sha256": list(SOLPROPMIX_CHECKPOINT_SHA256),
    }
    return {
        **request,
        "dtype": "float64" if request["precision"] == "float64_promoted" else "float32",
        "source_revision": SOLPROPMIX_SOURCE_REVISION,
        "source_tree": SOLPROPMIX_SOURCE_TREE,
        "worker_sha256": worker_hash,
        "checkpoint_sha256": list(SOLPROPMIX_CHECKPOINT_SHA256),
        "static_sha256": dict(SOLPROPMIX_STATIC_DATA_HASHES),
        "predictions": rows,
        "mean": mean,
        "std": math.sqrt(math.fsum((value - mean) ** 2 for value in values) / 10),
        "runtime": {
            "python": "3.11.0",
            "torch": "2.test",
            "cuda": None,
            "rdkit": "2026.test",
            "numpy": "2.test",
            "deterministic": True,
            "matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "amp": False,
            "fp16_reduced": False,
            "bf16_reduced": False,
            "gpu": None,
        },
        "_snapshot": snapshot,
        "_output": {
            "stdout_sha256": "b" * 64,
            "stderr_sha256": "c" * 64,
            "stderr_truncated": False,
        },
    }


def test_official_source_tree_and_artifact_set_are_pinned():
    assert SOLPROPMIX_SOURCE_REVISION == "80043ce09eb8802517c35b59254f8e9c181f2dac"
    assert SOLPROPMIX_SOURCE_TREE == "bc7b933d6cf4c55c8ad10236383cfd09afbbfdc2"
    assert len(SOLPROPMIX_CHECKPOINT_SHA256) == 10
    assert len(SOLPROPMIX_STATIC_DATA_HASHES) == 4
    assert (
        _normalize_origin("git@gitlab.kuleuven.be:creas/vermeiregroup/solprop.git")
        == SOLPROPMIX_SOURCE_URL
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "non-empty"),
        ("C[NH3+]", "net-neutral"),
        ("[CH3]", "radical"),
        ("CC.O", "disconnected"),
        ("C[Si](C)(C)C", "does not support"),
        ("O", "contain carbon"),
        ("[13CH3]CO", "isotope"),
    ],
)
def test_solute_domain_gate(value, message):
    with pytest.raises(SolPropMixConfigError, match=message):
        from maple.function.solvfe.solpropmix_property import (
            _canonical_neutral_structure,
        )

        _canonical_neutral_structure(value, "solute")


def test_net_neutral_internal_resonance_charges_are_allowed():
    from maple.function.solvfe.solpropmix_property import _canonical_neutral_structure

    canonical = _canonical_neutral_structure("C[N+](=O)[O-]", "solute")
    assert "[N+]" in canonical and "[O-]" in canonical


def test_mixture_merges_equivalents_removes_zero_and_sorts():
    first = _normalize_solvents(
        [("O", 0.25), ("[OH2]", 0.25), ("CO", 0.5), ("invalid", 0.0)]
    )
    second = _normalize_solvents([("CO", 0.5), ("O", 0.5)])
    assert first == second
    assert [(item.canonical_smiles, item.mole_fraction) for item in first] == [
        ("CO", 0.5),
        ("O", 0.5),
    ]


@pytest.mark.parametrize("count", (1, 2, 3))
def test_positive_request_shapes_preserve_component_count(tmp_path, monkeypatch, count):
    adapter = _adapter(tmp_path, monkeypatch)
    structures = ["O", "CO", "CCO"][:count]
    fractions = [(item, 1.0 / count) for item in structures]
    seen: dict[str, object] = {}

    def fake_run(request_json: str):
        request = json.loads(request_json)
        seen.update(request)
        return _payload(request)

    monkeypatch.setattr(adapter, "_run_script", fake_run)
    result = adapter.predict("CC", fractions)
    assert len(seen["solvents"]) == count
    assert len(result.solvent_components) == count


def test_result_receipt_states_precision_and_nonionic_boundary(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_run_script",
        lambda request_json: _payload(json.loads(request_json)),
    )
    result = adapter.predict("CCO", [("O", 1.0)])
    receipt = result.runtime_receipt
    assert (
        receipt.precision_semantics
        == SOLPROPMIX_PRECISION_SEMANTICS["float64_promoted"]
    )
    assert "not additional learned precision" in receipt.precision_semantics
    assert receipt.solute_carbon_requirement_verified is True
    assert receipt.isotope_free_structures_verified is True
    assert receipt.solvent_net_neutrality_verified is True
    assert receipt.solvent_nonionic_liquid_phase_structurally_verified is False
    assert receipt.solvent_nonionic_liquid_phase_caller_requirement is True
    assert "caller responsibility" in result.solvent_nonionic_liquid_phase_requirement
    assert receipt.source_tree == SOLPROPMIX_SOURCE_TREE
    assert receipt.worker_sha256 == "a" * 64


def test_parent_recomputes_and_rejects_tampering(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)

    def tampered(request_json: str):
        payload = _payload(json.loads(request_json))
        payload["mean"] = 123.0
        return payload

    monkeypatch.setattr(adapter, "_run_script", tampered)
    with pytest.raises(SolPropMixRuntimeError, match="ensemble mean"):
        adapter.predict("CCO", [("O", 1.0)])


def test_parent_rejects_worker_and_snapshot_identity_tampering(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)

    def tampered(request_json: str):
        payload = _payload(json.loads(request_json))
        payload["worker_sha256"] = "0" * 64
        return payload

    monkeypatch.setattr(adapter, "_run_script", tampered)
    with pytest.raises(SolPropMixRuntimeError, match="artifact identity"):
        adapter.predict("CCO", [("O", 1.0)])


def test_source_shadow_is_rejected_before_archive(tmp_path):
    roots = [tmp_path / name for name in ("source", "static", "weights")]
    for root in roots:
        root.mkdir()
    (roots[0] / "solvation_predictor" / "data").mkdir(parents=True)
    with pytest.raises(SolPropMixConfigError, match="shadow path"):
        SolPropMixPropertyAdapter(*roots)


@pytest.mark.parametrize("name", ("Scaler.py", "Splitter.py"))
def test_tampered_supplemental_modules_are_detected(tmp_path, name):
    path = tmp_path / name
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(SolPropMixConfigError, match="SHA256 mismatch"):
        _verify_file(
            path,
            SOLPROPMIX_STATIC_DATA_HASHES[f"solvation_predictor/data/{name}"],
            name,
        )


def test_numpy_reconstruct_prefers_2x_path(monkeypatch):
    marker = object()
    module = type("Module", (), {"_reconstruct": marker})()
    seen: list[str] = []

    def fake_import(name: str):
        seen.append(name)
        return module

    monkeypatch.setattr(worker.importlib, "import_module", fake_import)
    assert worker._numpy_reconstruct() is marker
    assert seen == ["numpy._core.multiarray"]


def test_numpy_reconstruct_falls_back_only_for_missing_2x_module(monkeypatch):
    marker = object()
    module = type("Module", (), {"_reconstruct": marker})()
    seen: list[str] = []

    def fake_import(name: str):
        seen.append(name)
        if name == "numpy._core.multiarray":
            raise ModuleNotFoundError(name)
        return module

    monkeypatch.setattr(worker.importlib, "import_module", fake_import)
    assert worker._numpy_reconstruct() is marker
    assert seen == ["numpy._core.multiarray", "numpy.core.multiarray"]


def test_worker_source_uses_exact_checkpoint_bytes():
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert "checkpoint_bytes =" in source
    assert "io.BytesIO(checkpoint_bytes)" in source
    assert "weights_only=True" in source
    assert "torch.load(checkpoint_path" not in source


def test_snapshot_worker_is_copied_by_exact_bytes(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)
    (tmp_path / "runtime").mkdir()
    monkeypatch.setattr(
        adapter, "_archive_source", lambda root: (root / "source").mkdir()
    )
    monkeypatch.setattr(
        "maple.function.solvfe.solpropmix_property._verify_file_bytes",
        lambda path, expected, label: bytes.fromhex(expected),
    )
    evidence = adapter._materialize_snapshot(tmp_path / "runtime")
    copied = (tmp_path / "runtime" / "worker.py").read_bytes()
    original = Path(worker.__file__).read_bytes()
    assert copied == original
    assert evidence.worker_sha256 == hashlib.sha256(original).hexdigest()


def test_source_identity_checks_pinned_tree(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("source", "static", "weights")]
    for root in roots:
        root.mkdir()
    for relative in SOLPROPMIX_STATIC_DATA_HASHES:
        path = roots[1] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for index in range(10):
        (roots[2] / f"model{index}.pt").touch()

    def git_output(_root, *args):
        return {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "HEAD"): SOLPROPMIX_SOURCE_REVISION,
            ("rev-parse", "HEAD^{tree}"): SOLPROPMIX_SOURCE_TREE,
            ("config", "--get", "remote.origin.url"): SOLPROPMIX_SOURCE_URL,
            ("status", "--porcelain", "--untracked-files=all"): "",
        }[args]

    monkeypatch.setattr(
        "maple.function.solvfe.solpropmix_property._git_output", git_output
    )
    monkeypatch.setattr(
        "maple.function.solvfe.solpropmix_property._git_result",
        lambda root, *args: subprocess.CompletedProcess(args, 0, "", ""),
    )
    SolPropMixPropertyAdapter(*roots)


@pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in (
            "SOLPROPMIX_SOURCE_ROOT",
            "SOLPROPMIX_STATIC_SOURCE_ROOT",
            "SOLPROPMIX_QMEXP_WEIGHTS_ROOT",
        )
    ),
    reason="real pinned SolProp-mix assets were not supplied",
)
def test_real_pinned_cpu_snapshot_smoke():
    adapter = SolPropMixPropertyAdapter(
        os.environ["SOLPROPMIX_SOURCE_ROOT"],
        os.environ["SOLPROPMIX_STATIC_SOURCE_ROOT"],
        os.environ["SOLPROPMIX_QMEXP_WEIGHTS_ROOT"],
        python_executable=os.environ.get("SOLPROPMIX_PYTHON"),
    )
    result = adapter.predict("CCO", [("O", 1.0)])
    assert len(result.model_predictions) == 10
    assert math.isfinite(result.predicted_solvation_free_energy_kcal_mol)
    assert result.runtime_receipt.source_tree == SOLPROPMIX_SOURCE_TREE
    assert (
        result.runtime_receipt.worker_sha256
        == hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()
    )
