from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "tools/route2_release/run_mace_mdp_polar_role_separated_adt_iodine_canary.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("iodine_adt_canary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_geometry_is_target_free_and_content_addressed() -> None:
    module = _load_script()
    payload = module._load_geometry(ROOT)
    assert payload["payload_sha256"] == module.EXPECTED_GEOMETRY_PAYLOAD_SHA256
    assert payload["source"]["upstream_geometry_sha256"] == (
        module.EXPECTED_UPSTREAM_GEOMETRY_SHA256
    )
    assert payload["claim_boundary"] == {
        "experimental_solvation_target_embedded": False,
        "geometry_only": True,
        "purpose": "target-free-iodine-adt-operator-comparison",
    }
    assert "target" not in payload


def test_output_writer_never_overwrites(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "evidence.json"
    module._write_json_exclusive(output, {"first": True})
    with pytest.raises(FileExistsError):
        module._write_json_exclusive(output, {"second": True})
    assert output.read_text() == '{\n  "first": true\n}\n'


def test_help_does_not_import_optional_scientific_runtime() -> None:
    code = f"""
import builtins, runpy, sys
blocked = {{'ase', 'numpy', 'pyddx', 'scipy', 'torch', 'mace'}}
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise RuntimeError('optional import during --help: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [{str(SCRIPT)!r}, '--help']
runpy.run_path({str(SCRIPT)!r}, run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--cold-replays" in completed.stdout
