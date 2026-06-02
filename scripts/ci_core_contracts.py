#!/usr/bin/env python
"""Fast CI contracts that do not require torch-backed model execution.

These checks are deliberately suitable for standard GitHub-hosted runners:
they validate packaging metadata, command parsing fail-loud boundaries, MD
template generation, MDP parsing, and the no-backend CLI path without requiring
GPU hardware or heavyweight model files.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import maple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(message: str) -> None:
    raise AssertionError(message)


def _load_module(name: str, relative_path: str):
    """Load a light module by file path without importing package __init__ files.

    MAPLE's top-level reader package currently imports InputReader, which imports
    torch.  Core CI intentionally verifies the dependency-light path, so these
    contracts load command_control/md helpers directly from their files.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        _fail(f"could not load module {name!r} from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_command_control = _load_module("maple_ci_command_control", "maple/function/read/command_control.py")
_mdp_reader = _load_module("maple_ci_mdp_reader", "maple/function/dispatcher/md/mdp_reader.py")
_md_templates = _load_module("maple_ci_md_templates", "maple/function/dispatcher/md/md_templates.py")
CommandControl = _command_control.CommandControl
parse_mdp = _mdp_reader.parse_mdp
generate_mdp_template = _md_templates.generate_mdp_template


def _assert(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _assert_raises(expected_type: type[BaseException], expected_text: str, func: Callable[[], object]) -> None:
    try:
        func()
    except expected_type as exc:
        if expected_text not in str(exc):
            _fail(f"expected {expected_text!r} in {type(exc).__name__}: {exc}")
        return
    except Exception as exc:  # pragma: no cover - clearer CI failure message
        _fail(f"expected {expected_type.__name__}, got {type(exc).__name__}: {exc}")
    _fail(f"expected {expected_type.__name__} containing {expected_text!r}, got no exception")


def _parse(settings: list[str], tmp_path: Path) -> CommandControl:
    return CommandControl.from_settings(settings, output_path=str(tmp_path / "command-control.out"))


def _check_package_metadata() -> None:
    dist_version = importlib.metadata.version("maple")
    _assert(dist_version == maple.__version__, f"metadata version {dist_version!r} != maple.__version__ {maple.__version__!r}")
    print(f"CORE_OK package_version={dist_version} maple_file={Path(maple.__file__).resolve()}")


def _check_command_contracts(tmp_path: Path) -> None:
    sp = _parse(
        [
            "#model=uma(size=uma-s-1p1,task=omol,inference=default)",
            "#sp(verbose=1)",
        ],
        tmp_path,
    )
    _assert(sp.task == "sp", f"expected sp task, got {sp.task!r}")
    _assert(sp.params["model"] == "uma", f"expected UMA model, got {sp.params['model']!r}")
    _assert(
        sp.params["model_options"] == {
            "size": "uma-s-1p1",
            "task": "omol",
            "inference": "default",
        },
        f"unexpected UMA options: {sp.params['model_options']!r}",
    )
    _assert(sp.params["verbose"] == 1, f"SP verbose did not parse as int: {sp.params!r}")

    md = _parse(
        [
            "#model=ani2x",
            "#md(ensemble=nvt,steps=5,timestep=0.1,remove_com=false,traj_every=2,log_every=1)",
        ],
        tmp_path,
    )
    _assert(md.task == "md", f"expected md task, got {md.task!r}")
    _assert(md.params["ensemble"] == "nvt", f"expected nvt ensemble, got {md.params['ensemble']!r}")
    _assert(md.params["steps"] == 5, f"steps should coerce to int, got {md.params['steps']!r}")
    _assert(md.params["timestep"] == 0.1, f"timestep should coerce to float, got {md.params['timestep']!r}")
    _assert(md.params["remove_com"] is False, f"remove_com should coerce to bool, got {md.params['remove_com']!r}")

    _assert_raises(ValueError, "SP uses 'verbose'", lambda: _parse(["#sp(verbosity=1)"], tmp_path))
    _assert_raises(ValueError, "SP verbose must be 0 or 1", lambda: _parse(["#sp(verbose=2)"], tmp_path))
    _assert_raises(ValueError, "Unknown MD parameter: 'method'", lambda: _parse(["#md(method=nve)"], tmp_path))
    _assert_raises(ValueError, "Multiple tasks defined", lambda: _parse(["#sp", "#opt"], tmp_path))
    _assert_raises(ValueError, "Duplicate parameter", lambda: _parse(["#device=cpu", "#device=gpu0"], tmp_path))
    _assert_raises(ValueError, "Unsupported UMA size", lambda: _parse(["#model=uma(size=not-a-size)", "#sp"], tmp_path))

    print("CORE_OK command_contracts")


def _check_mdp_templates(tmp_path: Path) -> None:
    for ensemble in ("nve", "nvt", "npt"):
        output = tmp_path / f"{ensemble}.mdp"
        generate_mdp_template(ensemble, str(output), force=True)
        parsed = parse_mdp(str(output))
        _assert(parsed["ensemble"] == ensemble, f"{ensemble} template parsed as {parsed.get('ensemble')!r}")
        _assert(parsed["integrator"] == "md", f"{ensemble} template missing integrator=md")
        _assert(isinstance(parsed["steps"], int) and parsed["steps"] > 0, f"{ensemble} steps invalid: {parsed.get('steps')!r}")
        _assert(isinstance(parsed["timestep"], float) and parsed["timestep"] > 0, f"{ensemble} timestep invalid: {parsed.get('timestep')!r}")

    _assert_raises(ValueError, "expected 'key = value'", lambda: parse_mdp(str(tmp_path / "bad.mdp")))
    print("CORE_OK mdp_templates")


def _check_cli_no_backend(tmp_path: Path) -> None:
    maple_cmd = shutil.which("maple")
    _assert(maple_cmd is not None, "console script 'maple' is not on PATH after installation")

    for ensemble in ("nve", "nvt", "npt"):
        output = tmp_path / f"cli-{ensemble}.mdp"
        completed = subprocess.run(
            [maple_cmd, "md", ensemble, "-o", str(output), "-f"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            _fail(f"maple md {ensemble} failed with exit code {completed.returncode}")
        parsed = parse_mdp(str(output))
        _assert(parsed["ensemble"] == ensemble, f"CLI {ensemble} template parsed as {parsed.get('ensemble')!r}")

    print("CORE_OK cli_no_backend_templates")


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"cwd={Path.cwd()}")
    with tempfile.TemporaryDirectory(prefix="maple-core-ci-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "bad.mdp").write_text("this is not an mdp entry\n")
        _check_package_metadata()
        _check_command_contracts(tmp_path)
        _check_mdp_templates(tmp_path)
        _check_cli_no_backend(tmp_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
