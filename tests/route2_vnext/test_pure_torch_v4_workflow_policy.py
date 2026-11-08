from __future__ import annotations

import pytest
import maple.function.route2_smd_profiles as profile_module
import json
from pathlib import Path

from maple.function.read.command_control import CommandControl
from maple.function.read.input_reader import InputReader
from maple.function.route2_smd_profiles import (
    CANONICAL_SMD_PROFILE,
    PCMSOLVER_INTRINSIC_CAVITY_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
    PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
    route2_smd_profile_spec,
    route2_workflow_capability,
)


def test_existing_pcmsolver_cavity_property_remains_on_profile_spec():
    assert not route2_smd_profile_spec(
        CANONICAL_SMD_PROFILE
    ).uses_intrinsic_pcmsolver_cavity
    assert route2_smd_profile_spec(
        PCMSOLVER_INTRINSIC_CAVITY_PROFILE
    ).uses_intrinsic_pcmsolver_cavity


def test_v4_devices_share_one_workflow_policy_without_changing_v3():
    cpu = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    )
    cuda = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE
    )
    v3 = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
    )

    assert cpu.workflow_policy_id == cuda.workflow_policy_id
    assert cpu.workflow_policy_id == "pure-torch-fp32-v4-workflows-v1"
    assert v3.workflow_policy_id is None
    assert v3.experimental_task_allowlist == ("sp", "opt", "freq")


@pytest.mark.parametrize(
    ("task", "method", "refine", "execution_class", "image_policy"),
    [
        ("sp", "sp", None, "bounded_execution", "single"),
        ("opt", "lbfgs", None, "bounded_execution", "single"),
        ("opt", "rfo", None, "bounded_execution", "single"),
        ("scan", "rigid", None, "bounded_execution", "single"),
        ("scan", "lbfgs", None, "bounded_execution", "single"),
        ("freq", "mw", None, "bounded_execution", "single"),
        ("freq", "cartesian-curvature", None, "bounded_execution", "single"),
        ("freq", "both", None, "bounded_execution", "single"),
        ("ts", "neb", None, "bounded_execution", "multi"),
        ("ts", "neb", "cineb", "bounded_execution", "multi"),
        ("ts", "string", "cistring", "bounded_execution", "multi"),
        ("ts", "autoneb", None, "bounded_execution", "multi"),
        ("ts", "prfo", None, "bounded_execution", "single"),
        ("ts", "dimer", None, "bounded_execution", "single"),
        ("ts", "neb", "nebts", "bounded_execution", "multi"),
        ("ts", "string", "stringts", "bounded_execution", "multi"),
        ("irc", "gs", None, "bounded_execution", "single"),
        ("md", "nve", None, "bounded_execution", "single"),
        ("md", "nvt", None, "bounded_execution", "single"),
    ],
)
def test_v4_policy_matrix(task, method, refine, execution_class, image_policy):
    capability = route2_workflow_capability(
        "pure-torch-fp32-v4-workflows-v1",
        task=task,
        method=method,
        refine=refine,
    )
    assert capability.execution_class == execution_class
    assert capability.image_policy == image_policy


@pytest.mark.parametrize(
    ("task", "method", "refine", "reason"),
    [
        ("scan", "rfo", None, "constrained_hessian_projection_unavailable"),
        ("md", "npt", None, "periodic_stress_contract_unavailable"),
    ],
)
def test_v4_policy_closed_rows_are_explicit(task, method, refine, reason):
    capability = route2_workflow_capability(
        "pure-torch-fp32-v4-workflows-v1",
        task=task,
        method=method,
        refine=refine,
    )
    assert capability.execution_class == "closed"
    assert capability.reason == reason


def test_v4_constraint_policy_distinguishes_force_and_hessian_optimizers():
    force_only = route2_workflow_capability(
        PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
        task="opt",
        method="lbfgs",
    )
    hessian = route2_workflow_capability(
        PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
        task="opt",
        method="rfo",
    )
    assert force_only.constraints_allowed is True
    assert hessian.constraints_allowed is False


def test_v4_capability_registry_is_immutable():
    with pytest.raises(TypeError):
        profile_module._PURE_TORCH_FP32_V4_CAPABILITIES[("sp", "sp", None)] = None


def _v4_input(task_line: str):
    return CommandControl.from_settings(
        [
            "#model=macepolm",
            "#device=cpu",
            task_line,
            "#solv(method=smd,provider=torch,implicit=water,experimental=true)",
        ]
    )


@pytest.mark.parametrize(
    "task_line",
    [
        "#scan(mode=rigid)",
        "#scan(mode=relaxed,method=lbfgs)",
        "#freq(method=nonmw)",
        "#freq(method=cartesian-curvature)",
        "#freq(method=both)",
        "#ts(method=neb)",
        "#ts(method=neb,refine=cineb)",
        "#ts(method=string,refine=stringts)",
        "#ts(method=autoneb)",
        "#ts(method=prfo)",
        "#ts(method=dimer)",
        "#irc(method=gs)",
        "#irc(method=gs,mode_index=0)",
        "#md(ensemble=nve,steps=8,timestep=0.1,random_seed=7)",
        "#md(ensemble=nvt,thermostat=langevin,steps=8,timestep=0.1,random_seed=7)",
    ],
)
def test_v4_parser_accepts_policy_rows(task_line):
    command = _v4_input(task_line)
    assert command.params["solv"]["profile"].endswith("-v4")


@pytest.mark.parametrize(
    ("task_line", "message"),
    [
        (
            "#scan(mode=relaxed,method=rfo)",
            "constrained_hessian_projection_unavailable",
        ),
        ("#md(ensemble=npt)", "periodic_stress_contract_unavailable"),
        ("#irc(method=gs,mode_index=0,target_mode=1)", "cannot be combined"),
        ("#md(ensemble=nve,restart=true)", "restart/load_state is not qualified"),
    ],
)
def test_v4_parser_rejects_closed_or_ambiguous_rows(task_line, message):
    with pytest.raises(ValueError, match=message):
        _v4_input(task_line)


def _inline_water(shift=0.0, charge=0):
    return "\n".join(
        (
            f"{charge} 1",
            f"O {shift:.6f} 0.000000 0.000000",
            f"H {shift + 0.9572:.6f} 0.000000 0.000000",
            f"H {shift - 0.239:.6f} 0.926600 0.000000",
        )
    )


def test_v4_input_reader_accepts_policy_approved_multi_image_path(tmp_path):
    path = tmp_path / "v4-neb.inp"
    path.write_text(
        "\n".join(
            (
                "#model=macepolm",
                "#device=cpu",
                "#ts(method=neb,refine=cineb)",
                "#solv(method=smd,provider=torch,implicit=water,experimental=true)",
                "",
                _inline_water(0.0),
                "&",
                _inline_water(0.05),
                "",
            )
        )
    )
    molecules = InputReader()(str(path))
    assert len(molecules.multiatoms) == 2
    assert all(
        image.info["_maple_solvation_options"]["profile"].endswith("-v4")
        for image in molecules.multiatoms
    )


def test_v4_input_reader_rejects_mixed_path_charge(tmp_path):
    path = tmp_path / "v4-neb-mixed-charge.inp"
    path.write_text(
        "\n".join(
            (
                "#model=macepolm",
                "#device=cpu",
                "#ts(method=neb)",
                "#solv(method=smd,provider=torch,implicit=water,experimental=true)",
                "",
                _inline_water(0.0, charge=0),
                "&",
                _inline_water(0.05, charge=2),
                "",
            )
        )
    )
    with pytest.raises(ValueError, match="identical charge and multiplicity"):
        InputReader()(str(path))


def test_v4_rechecks_effective_md_ensemble_after_mdp_load(tmp_path):
    mdp = tmp_path / "override.mdp"
    mdp.write_text("ensemble = npt\nsteps = 8\n")
    with pytest.raises(ValueError, match="periodic_stress_contract_unavailable"):
        CommandControl.from_settings(
            [
                "#model=macepolm",
                "#device=cpu",
                f"#md(mdp={mdp})",
                "#solv(method=smd,provider=torch,implicit=water,experimental=true)",
            ]
        )


def test_gpu_sentinel_protocol_has_exact_budgeted_selectors_and_fields():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root / "docs/route2/preregistrations/"
            "pure-mace-polar-fp32-v4-workflows.json"
        ).read_text()
    )
    gpu = record["gpu_policy"]
    assert gpu["selectors"] == [
        "sp",
        "scan.rigid",
        "ts.neb",
        "ts.prfo",
        "ts.dimer",
        "irc.gs",
        "md.nve",
        "md.nvt-langevin",
    ]
    assert gpu["aggregate_gpu_wall_budget_seconds"] == 600
    assert gpu["original_four_case_replay"]["counts_against_total_gpu_seconds"]
    assert {
        "profile",
        "actual_device",
        "schema",
        "finite_metrics",
        "process_exit_code",
        "task_success",
        "candidate_only",
        "scientific_certified",
    }.issubset(gpu["required_fields"])
