from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("fairchem")

from maple.function.calculator.uma import _uma_calculator as uma_module


def _capture_loader(monkeypatch):
    calls = []
    predictor = object()

    def load(path, **kwargs):
        calls.append((str(path), kwargs))
        return predictor

    monkeypatch.setattr(uma_module, "load_predict_unit", load)
    return predictor, calls


def test_explicit_checkpoint_identity_uses_actual_compat_file(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    compat = tmp_path / "compat.pt"
    source.write_bytes(b"source")
    compat.write_bytes(b"actual-loaded-checkpoint")
    prepare_calls = []

    def prepare(name, path):
        prepare_calls.append((name, str(path)))
        return str(compat)

    monkeypatch.setattr(uma_module.UMACalculator, "_prepare_compat_checkpoint", prepare)
    predictor, load_calls = _capture_loader(monkeypatch)

    with pytest.warns(RuntimeWarning, match="requires CUDA"):
        loaded = uma_module.UMACalculator._build_predictor(
            "uma-s-1p1", {}, "cpu", checkpoint_path=str(source), inference_settings="turbo"
        )

    assert prepare_calls == [("uma-s-1p1", str(source))]
    assert len(load_calls) == 1
    assert load_calls[0][0] == loaded.checkpoint_path == str(compat)
    assert load_calls[0][1]["inference_settings"] == loaded.inference == "default"
    assert loaded.predictor is predictor
    assert loaded.model_fingerprint["digest"] == hashlib.sha256(compat.read_bytes()).hexdigest()
    assert loaded.reference_energies == {"atom_refs": None, "form_elem_refs": None}


def test_pretrained_resolution_and_references_are_loaded_once(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"pretrained")
    counts = {"checkpoint": 0, "atom_refs": 0, "form_elem_refs": 0}

    monkeypatch.setattr(uma_module.pretrained_mlip, "available_models", ("uma-s-1p1",))

    def resolve(name):
        assert name == "uma-s-1p1"
        counts["checkpoint"] += 1
        return str(checkpoint)

    def references(name, kind):
        assert name == "uma-s-1p1"
        counts[kind] += 1
        if kind == "atom_refs":
            return OmegaConf.create({"H": -0.5})
        return OmegaConf.create({"refs": {"H": -1.0}})

    monkeypatch.setattr(
        uma_module.pretrained_mlip, "pretrained_checkpoint_path_from_name", resolve
    )
    monkeypatch.setattr(uma_module.pretrained_mlip, "get_reference_energies", references)
    predictor, load_calls = _capture_loader(monkeypatch)

    loaded = uma_module.UMACalculator._build_predictor(
        "uma-s-1p1", {"batch_size": 2}, "cpu"
    )

    assert counts == {"checkpoint": 1, "atom_refs": 1, "form_elem_refs": 1}
    assert len(load_calls) == 1
    path, kwargs = load_calls[0]
    assert path == loaded.checkpoint_path == str(checkpoint)
    assert kwargs["atom_refs"] == loaded.reference_energies["atom_refs"] == {"H": -0.5}
    assert kwargs["form_elem_refs"] == loaded.reference_energies["form_elem_refs"] == {"H": -1.0}
    assert kwargs["overrides"] == {"batch_size": 2}
    assert loaded.predictor is predictor


def test_checkpoint_name_can_be_a_local_file(tmp_path, monkeypatch):
    source = tmp_path / "custom.pt"
    compat = tmp_path / "compat.pt"
    source.write_bytes(b"source")
    compat.write_bytes(b"compat")
    prepared = []

    def prepare(name, path):
        prepared.append((name, str(path)))
        return str(compat)

    monkeypatch.setattr(uma_module.pretrained_mlip, "available_models", ())
    monkeypatch.setattr(uma_module.UMACalculator, "_prepare_compat_checkpoint", prepare)
    _, load_calls = _capture_loader(monkeypatch)

    loaded = uma_module.UMACalculator._build_predictor(str(source), None, "cpu")

    assert prepared == [("custom", str(source))]
    assert load_calls[0][0] == loaded.checkpoint_path == str(compat)
    assert loaded.reference_energies == {"atom_refs": None, "form_elem_refs": None}


def test_pretrained_without_optional_form_references(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"pretrained")
    calls = []
    monkeypatch.setattr(uma_module.pretrained_mlip, "available_models", ("model-no-form",))
    monkeypatch.setattr(
        uma_module.pretrained_mlip,
        "pretrained_checkpoint_path_from_name",
        lambda name: str(checkpoint),
    )

    def references(name, kind):
        calls.append(kind)
        return OmegaConf.create({"H": -0.5})

    monkeypatch.setattr(uma_module.pretrained_mlip, "get_reference_energies", references)
    _, load_calls = _capture_loader(monkeypatch)

    loaded = uma_module.UMACalculator._build_predictor("model-no-form", None, "cpu")

    assert calls == ["atom_refs"]
    assert load_calls[0][1]["form_elem_refs"] is None
    assert loaded.reference_energies["form_elem_refs"] is None


def test_fallback_artifacts_each_resolve_once(tmp_path, monkeypatch):
    checkpoint = tmp_path / "download.pt"
    compat = tmp_path / "compat.pt"
    atom_refs = tmp_path / "atoms.yaml"
    form_refs = tmp_path / "form.yaml"
    checkpoint.write_bytes(b"download")
    compat.write_bytes(b"compat")
    atom_refs.write_text("H: -0.5\n")
    form_refs.write_text("refs:\n  H: -1.0\n")
    files = {
        "uma-s-1p1.pt": checkpoint,
        "iso_atom_elem_refs.yaml": atom_refs,
        "form_elem_refs.yaml": form_refs,
    }
    calls = []

    def download(**kwargs):
        calls.append(kwargs["filename"])
        return str(files[kwargs["filename"]])

    monkeypatch.setattr(uma_module.pretrained_mlip, "available_models", ())
    monkeypatch.setattr(uma_module, "hf_hub_download", download)
    monkeypatch.setattr(
        uma_module.UMACalculator,
        "_prepare_compat_checkpoint",
        lambda name, path: str(compat),
    )
    _, load_calls = _capture_loader(monkeypatch)

    loaded = uma_module.UMACalculator._build_predictor("uma-s-1p1", None, "cpu")

    assert calls == ["uma-s-1p1.pt", "iso_atom_elem_refs.yaml", "form_elem_refs.yaml"]
    assert load_calls[0][0] == loaded.checkpoint_path == str(compat)
    assert load_calls[0][1]["atom_refs"] == {"H": -0.5}
    assert load_calls[0][1]["form_elem_refs"] == {"H": -1.0}


def test_constructor_hashes_the_checkpoint_returned_by_loader(tmp_path, monkeypatch):
    loaded_checkpoint = tmp_path / "loaded.pt"
    loaded_checkpoint.write_bytes(b"bytes-actually-passed-to-loader")
    predictor = object()
    provenance = uma_module._LoadedPredictor(
        predictor=predictor,
        checkpoint_path=str(loaded_checkpoint),
        model_fingerprint={
            "algorithm": "sha256",
            "digest": hashlib.sha256(loaded_checkpoint.read_bytes()).hexdigest(),
            "source": "loaded_checkpoint",
        },
        inference="default",
        reference_energies={"atom_refs": {"H": -0.5}, "form_elem_refs": None},
    )
    monkeypatch.setattr(
        uma_module.UMACalculator,
        "_build_predictor",
        staticmethod(lambda *args, **kwargs: provenance),
    )
    monkeypatch.setattr(
        uma_module.FAIRChemCalculator,
        "__init__",
        lambda self, predict_unit, task_name: setattr(self, "_task_name", task_name),
    )

    loaded_checkpoint.write_bytes(b"changed-after-loader-returned")
    calculator = uma_module.UMACalculator("cpu", size="uma-s-1p1")

    identity = calculator.__dict__["maple_pes_identity"]
    assert calculator._predictor_unit is predictor
    assert identity["model_fingerprint"] == provenance.model_fingerprint
    assert identity["relevant_settings"]["reference_energies"] == provenance.reference_energies


def _checkpoint(path, marker):
    uma_module.torch.save(SimpleNamespace(model_config={"marker": marker}), path)


def test_compat_paths_separate_same_name_different_sources(tmp_path, monkeypatch):
    source_a = tmp_path / "a.pt"
    source_b = tmp_path / "b.pt"
    _checkpoint(source_a, "A")
    _checkpoint(source_b, "B")
    monkeypatch.setattr(uma_module, "CACHE_DIR", str(tmp_path / "cache"))

    path_a = uma_module.UMACalculator._prepare_compat_checkpoint("same", str(source_a))
    path_b = uma_module.UMACalculator._prepare_compat_checkpoint("same", str(source_b))

    assert path_a != path_b
    assert f"-{uma_module.UMA_COMPAT_SCHEMA}-" in path_a
    assert uma_module.torch.load(path_a, weights_only=False).model_config["marker"] == "A"
    assert uma_module.torch.load(path_b, weights_only=False).model_config["marker"] == "B"


def test_same_source_has_same_compat_fingerprint_across_fresh_caches(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.pt"
    _checkpoint(source, "same")

    paths = []
    digests = []
    for cache_name in ("cache-a", "cache-b"):
        monkeypatch.setattr(uma_module, "CACHE_DIR", str(tmp_path / cache_name))
        path = Path(
            uma_module.UMACalculator._prepare_compat_checkpoint("same", str(source))
        )
        paths.append(path)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())

    assert paths[0].name == paths[1].name
    assert digests[0] == digests[1]
    assert uma_module.torch.load(paths[0], weights_only=False).model_config == (
        uma_module.torch.load(paths[1], weights_only=False).model_config
    )


def test_legacy_v1_cache_does_not_change_current_compat_identity(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    _checkpoint(source, "same")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    populated_cache = tmp_path / "populated-cache"
    legacy_dir = populated_cache / "maple_compat"
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / f"same-v1-{source_digest}.pt"
    uma_module.torch.save(
        SimpleNamespace(model_config={"marker": "same"}), legacy_path
    )
    legacy_bytes = legacy_path.read_bytes()

    paths = []
    digests = []
    for cache in (populated_cache, tmp_path / "fresh-cache"):
        monkeypatch.setattr(uma_module, "CACHE_DIR", str(cache))
        path = Path(
            uma_module.UMACalculator._prepare_compat_checkpoint("same", str(source))
        )
        paths.append(path)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())

    assert all(path != legacy_path for path in paths)
    assert all(f"-{uma_module.UMA_COMPAT_SCHEMA}-" in path.name for path in paths)
    assert paths[0].name == paths[1].name
    assert digests[0] == digests[1]
    assert legacy_path.read_bytes() == legacy_bytes


def test_concurrent_compat_publication_has_one_immutable_target(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    _checkpoint(source, "same")
    monkeypatch.setattr(uma_module, "CACHE_DIR", str(tmp_path / "cache"))
    original_save = uma_module.torch.save
    save_count = 0
    count_lock = threading.Lock()
    writers_ready = threading.Barrier(2)

    def slow_save(value, file_object):
        nonlocal save_count
        with count_lock:
            save_count += 1
        writers_ready.wait()
        original_save(value, file_object)

    monkeypatch.setattr(uma_module.torch, "save", slow_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(
            pool.map(
                lambda _: uma_module.UMACalculator._prepare_compat_checkpoint(
                    "same", str(source)
                ),
                range(2),
            )
        )

    assert paths[0] == paths[1]
    assert save_count == 2
    assert Path(paths[0]).is_file()


def test_failed_compat_write_leaves_no_partial_archive(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    _checkpoint(source, "broken")
    cache = tmp_path / "cache"
    monkeypatch.setattr(uma_module, "CACHE_DIR", str(cache))

    def partial_then_fail(value, file_object):
        del value
        file_object.write(b"partial")
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(uma_module.torch, "save", partial_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        uma_module.UMACalculator._prepare_compat_checkpoint("broken", str(source))

    compat_dir = cache / "maple_compat"
    assert not list(compat_dir.glob("*.pt"))
    assert not list(compat_dir.glob("*.tmp"))


def test_path_replacement_during_hash_still_loads_open_source(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    replacement = tmp_path / "replacement.pt"
    _checkpoint(source, "original")
    _checkpoint(replacement, "replacement")
    monkeypatch.setattr(uma_module, "CACHE_DIR", str(tmp_path / "cache"))
    original_sha256 = hashlib.sha256

    class ReplacingHash:
        def __init__(self):
            self._hash = original_sha256()
            self._replaced = False

        def update(self, chunk):
            self._hash.update(chunk)
            if not self._replaced:
                os.replace(replacement, source)
                self._replaced = True

        def hexdigest(self):
            return self._hash.hexdigest()

    monkeypatch.setattr(uma_module.hashlib, "sha256", ReplacingHash)
    compat = uma_module.UMACalculator._prepare_compat_checkpoint("race", str(source))

    loaded = uma_module.torch.load(compat, weights_only=False)
    assert loaded.model_config["marker"] == "original"
