from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms

from maple.function.engine import engine
from maple.function.read.command_control import CommandControl
from maple.function.dispatcher.solvfe import SolvationFreeEnergyWorkflow

from .test_runtime_entry import _valid_lines


ROOT = Path(__file__).resolve().parents[3]


def _methane() -> Atoms:
    atoms = Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


def test_dry_run_writes_restartable_manifest_and_withholds_free_energy(
    tmp_path,
):
    control = CommandControl.from_settings(_valid_lines())
    prepared = SolvationFreeEnergyWorkflow.prepare(
        params=control.params,
        atoms=_methane(),
        project_root=ROOT,
    )
    output = tmp_path / "methane.out"
    output.touch()
    workflow = SolvationFreeEnergyWorkflow(
        prepared=prepared,
        atoms=_methane(),
        primary_calculator=None,
        output=output,
    )

    first = workflow.run()
    second = workflow.run()

    assert first == second
    assert first["status"] == "engineering-ready"
    assert first["delta_g_hyd_kcal_mol"] is None
    run_dir = output.with_suffix(".solvfe")
    assert json.loads((run_dir / "manifest.json").read_text())["run_hash"] == (
        prepared.run_hash
    )
    assert json.loads((run_dir / "result.json").read_text()) == first
    assert set(
        json.loads((run_dir / "stage-ledger.json").read_text())["stages"]
    ) == {"PRECHECK", "RESULT"}


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"charge": 1, "mult": 1}, "charge=0"),
        ({"charge": 0, "mult": 2}, "mult=1"),
        ({}, "charge=0"),
    ],
)
def test_precheck_rejects_unsupported_electronic_state_before_model_load(
    metadata,
    message,
):
    control = CommandControl.from_settings(_valid_lines())
    atoms = _methane()
    atoms.info.clear()
    atoms.info.update(metadata)

    with pytest.raises(ValueError, match=message):
        SolvationFreeEnergyWorkflow.prepare(
            params=control.params,
            atoms=atoms,
            project_root=ROOT,
        )


def test_engine_dry_run_does_not_load_sampler_weights(tmp_path):
    input_path = tmp_path / "methane.inp"
    output_path = tmp_path / "methane.out"
    input_path.write_text(
        "\n".join(
            [
                *_valid_lines(),
                "#device=cpu",
                "",
                "0 1",
                "C  0.000  0.000  0.000",
                "H  0.629  0.629  0.629",
                "H -0.629 -0.629  0.629",
                "H -0.629  0.629 -0.629",
                "H  0.629 -0.629 -0.629",
                "",
            ]
        )
    )

    engine()(str(input_path), str(output_path))

    text = output_path.read_text()
    assert "Route A #solvfe dry-run PRECHECK completed" in text
    assert "Unsupported model" not in text
    result = json.loads(
        output_path.with_suffix(".solvfe").joinpath("result.json").read_text()
    )
    assert result["status"] == "engineering-ready"
    assert result["diagnostics"]["precheck_only"] is True
