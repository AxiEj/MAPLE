from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import pytest
from ase import Atoms

from maple.function.calculator.route2 import (
    PureMACEPolarDDXCalculator,
    is_pure_mace_polar_nonmd_calculator,
    is_pure_mace_polar_workflow_calculator,
)
from maple.function.read.command_control import CommandControl
from maple.function.read.input_reader import InputReader
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
    pure_macepolar_expected_device,
    route2_smd_profile_spec,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CPU_V2,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2,
    SCALAR_REGISTRY,
)


def _solv(profile: str) -> str:
    return (
        "#solv(implicit=water,method=smd,provider=pyddx,"
        f"profile={profile},response=frozen,experimental=true)"
    )


def _parse(profile: str, task: str, device: str, *, device_first: bool = False,
           gpuid: int | None = None) -> CommandControl:
    lines = ["#model=macepolm"]
    device_lines = [f"#device={device}"]
    if gpuid is not None:
        device_lines.append(f"#gpuid={gpuid}")
    if device_first:
        lines.extend(device_lines)
        lines.append(task)
    else:
        lines.append(task)
        lines.extend(device_lines)
    lines.append(_solv(profile))
    return CommandControl.from_settings(lines)


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
        info={"charge": 0, "mult": 1},
    )


class _PES:
    def __init__(self, scalar_contract_id: str) -> None:
        self.scalar_contract_id = scalar_contract_id


def test_v2_identities_and_expected_devices_are_exact() -> None:
    cpu = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE
    )
    cuda = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE
    )

    assert cpu.scalar_contract_id == (
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CPU_V2
    )
    assert cuda.scalar_contract_id == (
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2
    )
    assert pure_macepolar_expected_device(cpu) == "cpu"
    assert pure_macepolar_expected_device(cuda) == "cuda"
    assert cpu.experimental_task_allowlist == ("sp", "opt", "freq", "ts", "scan", "irc")
    assert cuda.experimental_task_allowlist == cpu.experimental_task_allowlist
    assert cpu.scalar_contract_id in SCALAR_REGISTRY
    assert cuda.scalar_contract_id in SCALAR_REGISTRY


@pytest.mark.parametrize("device_first", (False, True))
def test_explicit_device_is_not_overwritten_by_freq_defaults(device_first: bool) -> None:
    command = _parse(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE,
        "#freq(method=mw,treat_imag_as_real=false)",
        "cpu",
        device_first=device_first,
    )
    assert command.params["device"] == "cpu"


def test_v1_explicit_cuda_freq_is_rejected_instead_of_rewritten_to_cpu() -> None:
    with pytest.raises(ValueError, match="(?i)(cpu|device)"):
        _parse(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
            "#freq(method=mw,treat_imag_as_real=false)",
            "cuda:0",
            device_first=True,
        )


def test_cuda_profile_preserves_index_and_rejects_conflicting_gpuid(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    command = _parse(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE,
        "#sp",
        "cuda:1",
        gpuid=1,
    )
    assert command.params["device"] == "cuda:1"
    with pytest.raises(ValueError, match="(?i)(conflict|gpuid|index)"):
        _parse(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE,
            "#sp",
            "cuda:1",
            gpuid=0,
        )


def test_cuda_profile_fails_when_cuda_is_unavailable(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="(?i)(cuda|available)"):
        _parse(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE,
            "#sp",
            "cuda:0",
        )


def test_device_profiles_reject_wrong_family_and_invalid_index(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="requires #device=cpu"):
        _parse(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE,
            "#sp",
            "cuda:0",
        )
    with pytest.raises(ValueError, match="requires #device=cuda"):
        _parse(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE,
            "#sp",
            "cpu",
        )
    with pytest.raises(ValueError, match="only 1 CUDA"):
        _parse(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE,
            "#sp",
            "cuda:1",
        )


def test_cuda_calculator_keeps_exact_index_and_canonical_identity(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    profile = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE
    )
    calculator = PureMACEPolarDDXCalculator._from_test_pes(
        atoms=_water(),
        solvent="water",
        device="cuda:1",
        profile_spec=profile,
        pes=_PES(profile.scalar_contract_id),
    )
    assert calculator.device == "cuda:1"
    assert is_pure_mace_polar_workflow_calculator(calculator)
    assert is_pure_mace_polar_nonmd_calculator(calculator)


@pytest.mark.parametrize(
    "task",
    (
        "#sp",
        "#opt(method=lbfgs)",
        "#opt(method=sd)",
        "#opt(method=cg)",
        "#opt(method=sdcg)",
        "#opt(method=rfo)",
        "#scan(method=lbfgs,mode=rigid)",
        "#freq(method=mw,treat_imag_as_real=false)",
        "#ts(method=prfo)",
        "#ts(method=neb,refine=cineb)",
        "#ts(method=neb,refine=nebts)",
        "#ts(method=string,refine=cistring)",
        "#ts(method=string,refine=stringts)",
        "#ts(method=dimer)",
        "#ts(method=autoneb)",
        "#irc(method=gs)",
        "#irc(method=hpc)",
        "#irc(method=eulerpc)",
        "#irc(method=lqa)",
    ),
)
def test_cpu_v2_parses_plan_valid_nonmd_methods(task: str) -> None:
    _parse(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE,
        task,
        "cpu",
    )


@pytest.mark.parametrize(
    "task",
    (
        "#md(ensemble=nve,steps=1)",
        "#freq(method=nonmw,treat_imag_as_real=false)",
        "#freq(method=both,treat_imag_as_real=false)",
        "#freq(method=mw,treat_imag_as_real=true)",
        "#scan(method=rfo,mode=relaxed)",
    ),
)
def test_cpu_v2_rejects_plan_closed_methods(task: str) -> None:
    with pytest.raises(ValueError):
        _parse(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE,
            task,
            "cpu",
        )


def test_exact_predicates_distinguish_all_pure_profiles_from_v2_only() -> None:
    v1 = PureMACEPolarDDXCalculator._from_test_pes(
        atoms=_water(), solvent="water", pes=_PES(
            route2_smd_profile_spec(
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE
            ).scalar_contract_id
        )
    )
    cpu = PureMACEPolarDDXCalculator._from_test_pes(
        atoms=_water(),
        solvent="water",
        pes=_PES(EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CPU_V2),
        profile_spec=route2_smd_profile_spec(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE
        ),
    )
    assert is_pure_mace_polar_workflow_calculator(v1)
    assert is_pure_mace_polar_workflow_calculator(cpu)
    assert not is_pure_mace_polar_nonmd_calculator(v1)
    assert is_pure_mace_polar_nonmd_calculator(cpu)
    assert not is_pure_mace_polar_nonmd_calculator(
        SimpleNamespace(workflow_kind="pure-frozen-total-pes")
    )


def test_constructor_rejects_noncanonical_profile_clone() -> None:
    canonical = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE
    )
    with pytest.raises(ValueError, match="registered pure frozen workflow"):
        PureMACEPolarDDXCalculator._from_test_pes(
            atoms=_water(),
            solvent="water",
            pes=_PES(canonical.scalar_contract_id),
            profile_spec=replace(canonical),
        )


@pytest.mark.parametrize(
    ("profile_name", "device"),
    (
        (PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE, "cpu"),
        (PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE, "cuda:1"),
    ),
)
def test_production_factory_threads_exact_device_scalar_and_fixed_settings(
    monkeypatch, profile_name: str, device: str
) -> None:
    import torch
    import maple.function.calculator.route2._mace_polar_frozen_ddx_calculator as module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    captured: dict[str, object] = {}
    model = object()
    profile = route2_smd_profile_spec(profile_name)
    pes = _PES(profile.scalar_contract_id)

    def fake_model_builder(**kwargs):
        captured["model"] = kwargs
        return model

    def fake_pes_builder(model_arg, symbols, **kwargs):
        captured["model_arg"] = model_arg
        captured["symbols"] = symbols
        captured["pes"] = kwargs
        return pes

    monkeypatch.setattr(
        module, "build_official_mace_polar_1_m_radial_gto_adapter", fake_model_builder
    )
    monkeypatch.setattr(
        module, "build_smd_mace_polar_frozen_point_ddx_pes", fake_pes_builder
    )
    calculator = PureMACEPolarDDXCalculator(
        atoms=_water(), solvent="water", device=device, profile_spec=profile
    )

    assert calculator.device == device
    assert captured["model"] == {"device": device}
    assert captured["model_arg"] is model
    assert captured["symbols"] == ("O", "H", "H")
    kwargs = captured["pes"]
    assert kwargs["scalar_contract_id"] == profile.scalar_contract_id
    assert kwargs["lmax"] == 15
    assert kwargs["n_lebedev"] == 1202
    assert kwargs["solver_tolerance"] == 1.0e-12
    assert kwargs["eta"] == 0.1
    assert kwargs["n_proc"] == 1


def test_calculator_rejects_exact_cuda_index_drift() -> None:
    profile = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE
    )
    pes = _PES(profile.scalar_contract_id)
    pes.model = SimpleNamespace(device="cpu")
    calculator = PureMACEPolarDDXCalculator._from_test_pes(
        atoms=_water(), solvent="water", pes=pes, profile_spec=profile
    )
    pes.model.device = "cuda:0"
    with pytest.raises(RuntimeError, match="model device drifted"):
        calculator._validate_atoms(_water())


def _inline_water(x_shift: float = 0.0, *, symbols=("O", "H", "H")) -> str:
    coordinates = (
        (x_shift, 0.0, 0.0),
        (x_shift + 0.9572, 0.0, 0.0),
        (x_shift - 0.239, 0.9266, 0.0),
    )
    return "\n".join(
        f"{symbol} {x:.6f} {y:.6f} {z:.6f}"
        for symbol, (x, y, z) in zip(symbols, coordinates)
    )


def test_input_reader_accepts_ordered_v2_path_images_before_factory(tmp_path) -> None:
    input_path = tmp_path / "path.inp"
    input_path.write_text(
        "\n".join(
            (
                "#model=macepolm",
                "#device=cpu",
                "#ts(method=neb,refine=cineb)",
                _solv(PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE),
                "",
                "0 1",
                _inline_water(),
                "&",
                "0 1",
                _inline_water(0.05),
                "",
            )
        ),
        encoding="utf-8",
    )
    molecules = InputReader()(str(input_path))
    assert len(molecules.multiatoms) == 2
    assert all(
        image.info["_maple_solvation_options"]["profile"]
        == PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE
        for image in molecules.multiatoms
    )


def test_input_reader_rejects_path_symbol_order_mismatch(tmp_path) -> None:
    input_path = tmp_path / "bad-path.inp"
    input_path.write_text(
        "\n".join(
            (
                "#model=macepolm",
                "#device=cpu",
                "#ts(method=neb)",
                _solv(PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE),
                "",
                "0 1",
                _inline_water(),
                "&",
                "0 1",
                _inline_water(0.05, symbols=("O", "H", "F")),
                "",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identical atom count and ordered symbols"):
        InputReader()(str(input_path))
