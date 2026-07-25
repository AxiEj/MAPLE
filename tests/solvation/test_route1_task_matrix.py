from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
SPEC = importlib.util.spec_from_file_location(
    "run_route1_task_matrix",
    BENCHMARK_DIR / "run_route1_task_matrix.py",
)
task_matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(task_matrix)
import benchmark_core


def test_rendered_tasks_use_the_frozen_route1_product_contract(tmp_path):
    common = {
        "model": "aimnet2",
        "device": "cpu",
        "mol2_path": tmp_path / "molecule.mol2",
        "openmm_platform": "Reference",
        "opt_max_iter": 2,
        "scan_atoms": (8, 9),
        "scan_step_angstrom": 0.02,
        "scan_steps": 2,
    }

    sp = task_matrix.render_input(task="sp", **common)
    opt = task_matrix.render_input(task="opt", **common)
    scan = task_matrix.render_input(task="scan", **common)
    md = task_matrix.render_input(task="md", **common)

    for rendered in (sp, opt, scan, md):
        assert "#model=aimnet2" in rendered
        assert "#charge(source=mol2,label=am1bcc-frozen-manifest)" in rendered
        assert "method=gb,model=obc2,nonpolar=ace" in rendered
        assert "0 1" in rendered
    assert "#sp(verbose=1)" in sp
    assert "#opt(method=lbfgs,max_iter=2" in opt
    assert "#scan(method=lbfgs,mode=rigid)" in scan
    assert scan.endswith("S 8 9 0.02 2\n")
    assert (
        "#md(ensemble=nvt,steps=4,timestep=0.05,temperature=298.15"
        in md
    )
    assert "random_seed=20260725" in md


def test_scan_xyz_parser_requires_maple_dispatcher_comments(tmp_path):
    scan = tmp_path / "scan.xyz"
    scan.write_text(
        "\n".join(
            [
                "2",
                "Scanning combination 1/2: [1.0000]  Energy = -1.2500000000",
                "H 0 0 0",
                "H 1 0 0",
                "2",
                "Scanning combination 2/2: [1.0200]  Energy = -1.2400000000",
                "H 0 0 0",
                "H 1.02 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    frames = task_matrix.parse_scan_frames(scan)

    assert [frame["coordinate_values"] for frame in frames] == [[1.0], [1.02]]
    assert np.allclose(
        [frame["energy_hartree"] for frame in frames],
        [-1.25, -1.24],
    )
    scan.write_text("2\nnot a MAPLE scan\nH 0 0 0\nH 1 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized MAPLE SCAN"):
        task_matrix.parse_scan_frames(scan)


def test_frozen_task_matrix_uses_actual_maple_dispatchers_for_three_mlips():
    trace = json.loads(
        (
            BENCHMARK_DIR / "route1-task-matrix-methyl-hexanoate-2026-07-24.json"
        ).read_text(encoding="utf-8")
    )

    assert trace["content_sha256"] == benchmark_core.artifact_content_sha256(trace)
    assert trace["command_provenance"]["script_sha256"] == (
        benchmark_core.sha256_file(BENCHMARK_DIR / "run_route1_task_matrix.py")
    )
    assert trace["formula"] == (
        "E_solution(R) = E_MLIP,gas(R) + G_polar(R,q_fixed) + G_nonpolar(R)"
    )
    assert trace["route"] == {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "role": "Baseline/Product Route",
        "retraining": False,
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
    }
    assert {model["model"] for model in trace["models"]} == {
        "maceoff23m",
        "aimnet2",
        "ani2x",
    }
    assert trace["model_count"] == 3
    assert "3 named registered MLIP checkpoints" in trace["limitations"]
    assert trace["all_checks_pass"] is True
    assert trace["cross_model"]["passes"] is True
    for model in trace["models"]:
        assert model["all_checks_pass"] is True
        assert set(model["tasks"]) == {"sp", "opt", "scan", "md"}
        assert all(
            task["actual_maple_engine_dispatcher"] is True
            for task in model["tasks"].values()
        )
        assert {task["resolved_device"] for task in model["tasks"].values()} == {
            trace["environment"]["resolved_device"]
        }
        assert all(task["all_checks_pass"] is True for task in model["tasks"].values())
        assert all(
            task["solvation_provenance"]["component_decomposition"]
            == "single-context OpenMM energy-parameter derivative"
            for task in model["tasks"].values()
        )
        assert all(
            task["solvation_provenance"]["energy_force_evaluations_per_call"] == 1
            for task in model["tasks"].values()
        )
        assert model["tasks"]["sp"]["sp"]["force_component_count"] == (
            3 * trace["atom_count"]
        )
        assert model["tasks"]["opt"]["opt"]["energy_change_from_sp_hartree"] <= 0
        assert model["tasks"]["scan"]["scan"]["point_count"] == 3
        assert (
            model["tasks"]["scan"]["scan"]["reported_coordinate_max_abs_error_angstrom"]
            <= 5.0e-5
        )
        assert (
            model["tasks"]["scan"]["scan"]["step_sequence_max_abs_error_angstrom"]
            <= 1.0e-8
        )
        assert model["tasks"]["md"]["md"]["ensemble"] == "nvt"
        assert model["tasks"]["md"]["md"]["step_count"] == 4
        assert model["tasks"]["md"]["md"]["trajectory_frame_count"] == 4
        assert model["tasks"]["md"]["md"]["finite_thermodynamics"] is True
    assert "not broad OPT/SCAN/MD stability" in trace["claim_scope"]
