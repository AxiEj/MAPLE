"""Resident identity/cache ownership tests; synthetic unless explicitly enabled."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.extra_correction.implicit import anysolv


@pytest.fixture
def official_api(tmp_path, monkeypatch):
    @dataclass
    class Settings:
        execution_mode: str = "general"
        tf32: bool = False
        compile: bool = False
        merge_mole: bool = False

    class Model:
        norm = {"energy": {"rmsd": 2.0, "mean": 0.5}}
        eps_transform = "log"

        def __call__(self, data):
            return {
                "energy": {"energy": torch.tensor([data["spin"]], dtype=torch.float64)},
                "forces": {
                    "forces": torch.ones((len(data["numbers"]), 3), dtype=torch.float64)
                },
            }

    source = tmp_path / "anysolv"
    source.mkdir()
    package = ModuleType("anysolv")
    package.__file__ = str(source / "__init__.py")
    package.InferenceSettings = Settings
    loads, data_calls = [], []

    def load(checkpoint, **controls):
        loads.append((checkpoint, controls))
        return Model()

    def forbidden_predict(*args, **kwargs):
        pytest.fail("unowned upstream pathname cache/predictor reached")

    package.load_model = load
    package.predict_solvation_energy = forbidden_predict
    data = ModuleType("anysolv.data")

    def build(inputs, **kwargs):
        data_calls.append(kwargs)
        return {"numbers": inputs[0], **kwargs}

    data.build_atomic_data = build
    solvent = ModuleType("anysolv.solvent")
    solvent.get_solvent_vector = lambda name, **kwargs: (name, kwargs)
    predict = ModuleType("anysolv.predict")
    predict._MODEL_CACHE = {}
    for name, module in (
        ("anysolv", package),
        ("anysolv.data", data),
        ("anysolv.solvent", solvent),
        ("anysolv.predict", predict),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        source=source,
        checkpoint=tmp_path / "model.pt",
        loads=loads,
        data_calls=data_calls,
        settings=Settings,
        predict=predict,
    )


def _runtime(api, digest="verified-digest", checkpoint=None):
    return anysolv._OfficialAnySolvRuntime(
        source_root=api.source,
        checkpoint=checkpoint or api.checkpoint,
        checkpoint_sha256=digest,
        device="cpu",
    )


def test_hostile_upstream_cache_is_neither_trusted_nor_modified(
    official_api, monkeypatch
):
    api = official_api
    key = (str(api.checkpoint), "cpu", str(torch.float64), repr(api.settings()))
    hostile = object()
    api.predict._MODEL_CACHE[key] = hostile
    runtime = _runtime(api)
    assert len(api.loads) == 1
    assert runtime._resident.model is not hostile
    assert api.predict._MODEL_CACHE[key] is hostile
    monkeypatch.setattr(
        anysolv, "_sha256", lambda _: pytest.fail("per-evaluation rehash")
    )
    energy, forces = runtime.evaluate(
        np.array([1]), np.zeros((1, 3)), charge=0, multiplicity=2, solvent="water"
    )
    assert energy == 4.5
    assert np.array_equal(forces, np.full((1, 3), 2.0))
    assert api.predict._MODEL_CACHE[key] is hostile


def test_owned_model_shared_by_digest_across_paths_and_solvents(official_api):
    api = official_api
    water = _runtime(api)
    toluene = _runtime(api, checkpoint=api.checkpoint.with_name("same-bytes.pt"))
    assert water._resident is toluene._resident
    for runtime, solvent in ((water, "water"), (toluene, "toluene")):
        runtime.evaluate(
            np.array([1]), np.zeros((1, 3)), charge=0, multiplicity=1, solvent=solvent
        )
    assert len(api.loads) == 1
    controls = api.loads[0][1]
    assert controls["device"] == "cpu"
    assert controls["dtype"] is torch.float64
    assert controls["inference_settings"] == api.settings()
    assert [call["solvent"][0] for call in api.data_calls] == ["water", "toluene"]
    different = _runtime(api, digest="different-checkpoint-digest")
    assert different._resident is not water._resident
    assert len(api.loads) == 2


def test_private_cache_releases_resident_when_last_runtime_is_gone(official_api):
    runtime = _runtime(official_api)
    identity = runtime._resident.identity
    resident_ref = weakref.ref(runtime._resident)
    del runtime
    gc.collect()
    assert resident_ref() is None
    assert identity not in anysolv._RESIDENTS


@pytest.mark.skipif(
    os.environ.get("MAPLE_TEST_ANYSOLV_MOE") != "1",
    reason="explicit real AnySolv-MoE parity opt-in required",
)
def test_real_owned_model_matches_original_predictor_on_two_examples(monkeypatch):
    source = Path(
        os.environ.get(
            "MAPLE_ANYSOLV_TEST_SOURCE_ROOT", "/home/axie/.cache/maple/anysolv/anysolv"
        )
    )
    anysolv._validate_source(source)
    examples = [
        {
            "numbers": [8, 1, 1],
            "positions": [[0, 0, 0], [0.95, 0, 0], [-0.24, 0.92, 0]],
            "solvent": "water",
        },
        {
            "numbers": [6, 1, 1, 1, 1],
            "positions": [
                [0, 0, 0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ],
            "solvent": "toluene",
        },
    ]
    # The original API runs in a separate process, so parity requires only one
    # 2.3 GB resident per process and never adopts its pathname-cache entries.
    code = """
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('anysolv',root/'__init__.py',submodule_search_locations=[str(root)])
module=importlib.util.module_from_spec(spec)
sys.modules['anysolv']=module
spec.loader.exec_module(module)
records=[]
for case in json.loads(sys.argv[3]):
    energy,forces=module.predict_solvation_energy(
        (np.array(case['numbers']),np.array(case['positions'])),charge=0,spin=1,
        solvent=case['solvent'],checkpoint=sys.argv[2],device='cpu',dtype=torch.float64,
        inference_settings='default')
    records.append({'energy_ev':energy,'forces_ev_per_angstrom':forces.tolist()})
print(json.dumps(records))
"""
    environment = {
        **os.environ,
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(source),
            str(anysolv.ANYSOLV_MOE_DEFAULT_PATH),
            json.dumps(examples),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    references = json.loads(completed.stdout.splitlines()[-1])
    # Occupy the real official pathname cache with an unusable, unowned object.
    # MAPLE must neither consume it nor replace another caller's cache entry.
    from importlib import import_module, util

    if "anysolv" not in sys.modules:
        spec = util.spec_from_file_location(
            "anysolv", source / "__init__.py", submodule_search_locations=[str(source)]
        )
        assert spec is not None and spec.loader is not None
        module = util.module_from_spec(spec)
        sys.modules["anysolv"] = module
        spec.loader.exec_module(module)

    official_predict = import_module("anysolv.predict")
    official_settings = import_module("anysolv._backbone._compat.inference")
    cache_key = (
        str(anysolv.ANYSOLV_MOE_DEFAULT_PATH.resolve()),
        "cpu",
        str(torch.float64),
        repr(official_settings.guess_inference_settings("default")),
    )
    hostile = object()
    monkeypatch.setitem(official_predict._MODEL_CACHE, cache_key, hostile)
    backends = [
        anysolv.AnySolvBackend(solvent=case["solvent"], source_root=str(source))
        for case in examples
    ]
    assert backends[0]._runtime._resident is backends[1]._runtime._resident
    assert official_predict._MODEL_CACHE[cache_key] is hostile
    monkeypatch.setattr(anysolv, "_sha256", lambda _: pytest.fail("per-record rehash"))
    records = []
    for case, backend, reference in zip(examples, backends, references):
        energy, forces = backend._runtime.evaluate(
            np.array(case["numbers"]),
            np.array(case["positions"]),
            charge=0,
            multiplicity=1,
            solvent=case["solvent"],
        )
        energy_error = abs(energy - reference["energy_ev"])
        force_error = float(
            np.max(np.abs(forces - reference["forces_ev_per_angstrom"]))
        )
        assert energy_error <= 1e-12
        assert force_error <= 1e-12
        result = backend.evaluate(
            Atoms(
                numbers=case["numbers"],
                positions=case["positions"],
                info={"charge": 0, "mult": 1},
            )
        )
        assert result.energy_hartree == pytest.approx(energy / anysolv.EV_PER_HARTREE)
        assert result.forces_hartree_per_angstrom is None
        records.append(
            {
                "input": case,
                "reference": reference,
                "owned_energy_ev": energy,
                "energy_abs_error_ev": energy_error,
                "forces_max_abs_error_ev_per_angstrom": force_error,
            }
        )
    artifact = os.environ.get("MAPLE_ANYSOLV_REPAIR_PARITY_ARTIFACT")
    if artifact:
        payload = {
            "scope": "official predictor versus privately owned reference inference; no force admission",
            "source_revision": anysolv.ANYSOLV_SOURCE_REVISION,
            "checkpoint_sha256": anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256,
            "resident_identity": backends[0]._runtime._resident.identity,
            "solvents_share_one_resident": True,
            "hostile_official_path_cache_ignored_and_preserved": True,
            "absolute_tolerance": 1e-12,
            "records": records,
        }
        Path(artifact).write_text(json.dumps(payload, indent=2) + "\n")
