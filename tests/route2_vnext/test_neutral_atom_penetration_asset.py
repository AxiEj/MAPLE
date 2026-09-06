from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/route2_release/generate_neutral_atom_penetration_asset.py"
PREREGISTRATION = ROOT / (
    "docs/route2/preregistrations/neutral-atom-penetration-asset-v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("neutral_penetration_asset", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preregistration_is_target_independent_and_source_bound():
    module = _module()
    preregistration = module._load_preregistration()
    assert preregistration["claim_boundary"] == {
        "capability_admitted": False,
        "experimental_solvation_targets_read": False,
        "mace_outputs_read": False,
        "post_training_or_model_fitting_performed": False,
        "qm_atomic_density_basis_projection_only": True,
        "solvation_or_pcm_quantities_read": False,
    }
    numbers = [
        item["atomic_number"]
        for item in preregistration["generation_contract"]["elements"]
    ]
    assert numbers == [1, 6, 7, 8, 9, 15, 16, 17, 35]
    for relative, expected in preregistration["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_existing_asset_coverage_is_a_strict_subset_of_new_coverage():
    module = _module()
    legacy = module._legacy_mixtures()
    assert tuple(sorted(legacy)) == (1, 6, 7, 8, 16, 17)
    for counts, exponents in legacy.values():
        assert len(counts) == len(exponents) > 0


def test_help_does_not_import_pyscf_or_read_outputs(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import builtins\n"
        "old=builtins.__import__\n"
        "def guard(name,*a,**k):\n"
        "    if name == 'pyscf' or name.startswith('pyscf.'):\n"
        "        raise RuntimeError('pyscf import blocked')\n"
        "    return old(name,*a,**k)\n"
        "builtins.__import__=guard\n"
    )
    environment = {"PYTHONPATH": str(tmp_path)}
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output-table" in completed.stdout
    assert not list(tmp_path.glob("*.npz"))
