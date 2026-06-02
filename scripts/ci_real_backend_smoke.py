#!/usr/bin/env python
"""CI smoke gate for real MAPLE model backends.

This intentionally exercises public TorchScript checkpoints through MAPLE's
normal SetCalculator path.  It is stronger than an import/CLI smoke test because
each model must load real weights and produce finite energy + force values for a
small molecule.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from maple.function.calculator import SetCalculator


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "maple" / "function" / "calculator" / "model"

# Keep the CI set cheap but representative: these three model names cover the
# shipped ANI, AIMNet2, and MACE-style TorchScript wrapper families using public
# auto-downloadable checkpoints.
REAL_BACKEND_CASES = (
    ("ani1ccx", "ani1ccx.pt"),
    ("aimnet2", "aimnet2.pt"),
    ("egret", "egret1s.pt"),
)


def _water() -> Atoms:
    atoms = Atoms(
        "OH2",
        positions=[
            [0.0000, 0.0000, 0.0000],
            [0.7586, 0.0000, 0.5043],
            [-0.7586, 0.0000, 0.5043],
        ],
    )
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    return atoms


def _fail(message: str) -> None:
    raise AssertionError(message)


def _assert_checkpoint(filename: str) -> Path:
    checkpoint = MODEL_DIR / filename
    if not checkpoint.is_file():
        _fail(f"expected real checkpoint file was not created: {checkpoint}")
    if checkpoint.stat().st_size < 1_000_000:
        _fail(f"checkpoint is unexpectedly small: {checkpoint} ({checkpoint.stat().st_size} bytes)")
    return checkpoint


def _run_direct_backend_smoke(tmp_path: Path) -> None:
    device = torch.device("cpu")
    for model_name, checkpoint_name in REAL_BACKEND_CASES:
        output = tmp_path / f"{model_name}.out"
        output.write_text("")

        atoms = _water()
        calc = SetCalculator(
            device,
            model_name,
            str(output),
            atoms=atoms,
            model_options={},
        ).set_calculator()
        atoms.calc = calc

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        max_force = float(np.max(np.abs(forces)))

        if not np.isfinite(energy):
            _fail(f"{model_name} returned non-finite energy: {energy!r}")
        if forces.shape != (len(atoms), 3):
            _fail(f"{model_name} returned wrong force shape: {forces.shape!r}")
        if not np.all(np.isfinite(forces)):
            _fail(f"{model_name} returned non-finite forces: {forces!r}")
        if max_force <= 0.0:
            _fail(f"{model_name} returned all-zero forces; backend likely did not run")

        checkpoint = _assert_checkpoint(checkpoint_name)
        print(
            "REAL_BACKEND_OK "
            f"model={model_name} "
            f"calculator={type(calc).__name__} "
            f"checkpoint={checkpoint} "
            f"bytes={checkpoint.stat().st_size} "
            f"energy_hartree={energy:.10f} "
            f"max_force_hartree_per_angstrom={max_force:.8f}"
        )


def _run_cli_backend_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "egret_sp.inp"
    output_path = tmp_path / "egret_sp.out"
    input_path.write_text(
        "\n".join(
            [
                "#model=egret",
                "#sp(verbose=1)",
                "#device=cpu",
                "",
                "0 1",
                "O  0.0000  0.0000  0.0000",
                "H  0.7586  0.0000  0.5043",
                "H -0.7586  0.0000  0.5043",
                "",
            ]
        )
    )

    maple_cmd = shutil.which("maple")
    cmd = [maple_cmd, str(input_path), str(output_path)] if maple_cmd else [
        sys.executable,
        "-m",
        "maple.main",
        str(input_path),
        str(output_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        _fail(f"MAPLE CLI real-backend SP failed with exit code {completed.returncode}")

    text = output_path.read_text(errors="replace")
    if "ERROR:" in text:
        _fail(f"MAPLE CLI output contains an error:\n{text}")
    match = re.search(r"Energy:\s+([-+0-9.eE]+)\s+Hartree", text)
    if not match:
        _fail(f"MAPLE CLI output did not contain an energy line:\n{text}")
    energy = float(match.group(1))
    if not np.isfinite(energy):
        _fail(f"MAPLE CLI returned non-finite energy: {energy!r}")
    if "Gradients (Hartree/Angstrom):" not in text:
        _fail("MAPLE CLI verbose SP output did not include gradients")

    print(f"REAL_BACKEND_CLI_OK model=egret energy_hartree={energy:.10f} output={output_path}")


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"model_dir={MODEL_DIR}")
    os.environ.setdefault("MAPLE_DOWNLOAD_TIMEOUT", "120")

    with tempfile.TemporaryDirectory(prefix="maple-real-backend-") as tmp:
        tmp_path = Path(tmp)
        _run_direct_backend_smoke(tmp_path)
        _run_cli_backend_smoke(tmp_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
