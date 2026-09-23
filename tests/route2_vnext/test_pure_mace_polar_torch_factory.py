from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.route2 import (
    PureMACEPolarTorchCalculator,
    is_pure_mace_polar_torch_calculator,
    is_pure_mace_polar_workflow_calculator,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE,
    pure_macepolar_expected_device,
    route2_smd_profile_spec,
    route2_smd_profiles_for_provider,
    validate_route2_smd_profile,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3,
    get_scalar_definition,
)


def _water() -> Atoms:
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


class _FakeTorchPES:
    def __init__(self, scalar_contract_id: str, *, device: str = "cpu") -> None:
        self.scalar_contract_id = scalar_contract_id
        self.device = device
        self.calls: list[str] = []

    def configuration_sha256(self) -> str:
        return "f" * 64

    def get_potential_energy(self, atoms) -> float:
        self.calls.append("energy")
        return -2.5

    def get_forces(self, atoms) -> np.ndarray:
        self.calls.append("forces")
        return np.full((len(atoms), 3), 0.125)

    def evaluate_hessian(self, atoms):
        self.calls.append("hessian")
        return SimpleNamespace(
            hessian_eV_per_A2=np.eye(3 * len(atoms)),
            forces_eV_per_A=np.full((len(atoms), 3), 0.125),
            energy_eV=-2.5,
            derivative_method="torch-autograd",
        )

    def hessian_vector_product(self, atoms, direction):
        self.calls.append("hvp")
        return np.asarray(direction, dtype=float)


def test_direct_prfo_constructor_keeps_uncertified_v3_closed(tmp_path):
    from maple.function.dispatcher.ts.algorithm.PRFO import PRFO

    calc = _fake_calculator()
    atoms = _water()
    atoms.calc = calc
    output = tmp_path / "must-not-run.out"
    with pytest.raises(ValueError, match="P-RFO remains closed"):
        PRFO(str(output), atoms)
    assert not output.exists()
    assert calc.pes.calls == []


def _fake_calculator() -> PureMACEPolarTorchCalculator:
    atoms = _water()
    spec = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
    )
    return PureMACEPolarTorchCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        device="cpu",
        profile_spec=spec,
        pes=_FakeTorchPES(spec.scalar_contract_id),
    )


def test_v3_profiles_are_explicit_torch_identities_and_do_not_mutate_legacy():
    cpu = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
    )
    cuda = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE
    )

    assert cpu.provider == cuda.provider == "torch"
    assert (
        cpu.execution_route == cuda.execution_route == "pure-torch-analytic-total-pes"
    )
    assert cpu.nonpolar_model == cuda.nonpolar_model == "torch-legacy-smd-cds"
    assert (
        cpu.supported_solvents
        == cuda.supported_solvents
        == frozenset({"water", "hexane"})
    )
    assert cpu.experimental_task_allowlist == ("sp", "opt", "freq")
    assert cuda.experimental_task_allowlist == cpu.experimental_task_allowlist
    assert cpu.allowed_response_modes == cuda.allowed_response_modes == ("frozen",)
    assert pure_macepolar_expected_device(cpu) == "cpu"
    assert pure_macepolar_expected_device(cuda) == "cuda"
    assert route2_smd_profiles_for_provider("torch") == frozenset({cpu.name, cuda.name})
    assert validate_route2_smd_profile("torch", cpu.name, solvent="hexane") is cpu
    with pytest.raises(ValueError, match="does not support solvent"):
        validate_route2_smd_profile("torch", cpu.name, solvent="ethanol")
    with pytest.raises(ValueError, match="provider=pyddx"):
        validate_route2_smd_profile("pyddx", cpu.name, solvent="water")


@pytest.mark.parametrize(
    "scalar_id",
    [
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3,
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3,
    ],
)
def test_v3_scalar_registry_is_callable_but_not_release_admitted(scalar_id):
    scalar = get_scalar_definition(scalar_id)
    assert scalar.experimental_execution.energy
    assert scalar.experimental_execution.force
    assert scalar.experimental_execution.hessian_vector_product
    assert scalar.experimental_execution.hessian
    assert not scalar.experimental_execution.periodic_stress
    assert not scalar.enabled
    assert not scalar.admitted_capabilities.enabled_tiers
    assert scalar.nonpolar_profile == "torch-legacy-smd-cds"
    assert scalar.derivative_route.startswith("torch.float64 autograd")


def test_torch_calculator_has_analytic_only_hessian_and_exact_typeguard():
    calculator = _fake_calculator()
    atoms = _water()
    atoms.calc = calculator

    assert calculator.SUPPORTED_HESSIAN_MODES == ("analytic",)
    assert is_pure_mace_polar_torch_calculator(calculator)
    assert is_pure_mace_polar_workflow_calculator(calculator)
    assert atoms.get_potential_energy() == pytest.approx(-2.5)
    np.testing.assert_allclose(atoms.get_forces(), 0.125)
    evaluation = calculator.get_hessian_evaluation(atoms)
    np.testing.assert_allclose(evaluation.hessian_eV_per_A2, np.eye(9))
    with pytest.raises(ValueError, match="does not accept a finite-difference step"):
        calculator.get_hessian(atoms, delta=0.004)
    hvp, forces, energy = calculator.get_hvp(atoms, np.ones((3, 3)))
    np.testing.assert_allclose(np.asarray(hvp), 1.0)
    np.testing.assert_allclose(np.asarray(forces), 0.125)
    assert float(energy) == pytest.approx(-2.5)
    assert not is_pure_mace_polar_torch_calculator(
        SimpleNamespace(workflow_kind=calculator.workflow_kind)
    )


def test_torch_calculator_rejects_profile_device_mismatch(monkeypatch):
    spec = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        PureMACEPolarTorchCalculator._from_test_pes(
            atoms=_water(),
            solvent="water",
            device="cuda:0",
            profile_spec=spec,
            pes=_FakeTorchPES(spec.scalar_contract_id, device="cuda:0"),
        )
    with pytest.raises(ValueError, match="requires device=cuda"):
        PureMACEPolarTorchCalculator._from_test_pes(
            atoms=_water(),
            solvent="water",
            device="cpu",
            profile_spec=spec,
            pes=_FakeTorchPES(spec.scalar_contract_id),
        )


def test_torch_calculator_propagates_component_failure_without_fallback():
    class _FailingPES(_FakeTorchPES):
        def get_forces(self, atoms):
            raise RuntimeError("injected torch component failure")

    atoms = _water()
    spec = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
    )
    atoms.calc = PureMACEPolarTorchCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        device="cpu",
        profile_spec=spec,
        pes=_FailingPES(spec.scalar_contract_id),
    )
    with pytest.raises(RuntimeError, match="injected torch component failure"):
        atoms.get_forces()


def test_set_calculator_selects_torch_wrapper_without_legacy_provider(
    monkeypatch, tmp_path
):
    atoms = _water()
    profile = PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
    scalar_id = EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3
    fake = _FakeTorchPES(scalar_id)
    monkeypatch.setattr(
        "maple.solvation.experimental.mace_polar_torch."
        "build_smd_mace_polar_torch_pes",
        lambda symbols, **kwargs: fake,
    )
    calculator = SetCalculator(
        device="cpu",
        model="macepolm",
        output=str(tmp_path / "factory.out"),
        atoms=atoms,
        d4=False,
        implicit="smd",
        solvent="water",
        solvation_options={
            "method": "smd",
            "implicit": "water",
            "experimental": True,
            "provider": "torch",
            "profile": profile,
            "response": "frozen",
            "standard_state": "1m",
        },
    ).set_calculator()

    assert type(calculator) is PureMACEPolarTorchCalculator
    assert calculator.pes is fake
    assert calculator.profile_spec.provider == "torch"
    log = (tmp_path / "factory.out").read_text()
    assert "provider=torch" in log
    assert "Hessian policy: analytic torch.float64 autograd only" in log
    assert "PySCF" not in log
    assert "Richardson" not in log


@pytest.mark.parametrize(
    "task_line",
    ["#sp", "#opt(method=lbfgs)", "#freq(method=mw,treat_imag_as_real=false)"],
)
def test_input_contract_accepts_only_initial_v3_workflows(task_line):
    command = CommandControl.from_settings(
        [
            "#model=macepolm",
            task_line,
            "#device=cpu",
            "#solv(implicit=water,method=smd,provider=torch,"
            f"profile={PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE},"
            "response=frozen,experimental=true)",
        ]
    )
    assert command.params["solv"]["provider"] == "torch"


@pytest.mark.parametrize(
    "task_line",
    ["#ts(method=prfo)", "#irc(method=gs)", "#md(ensemble=nve,steps=1)"],
)
def test_input_contract_keeps_ts_irc_md_closed(task_line):
    with pytest.raises(ValueError, match="permits only experimental task"):
        CommandControl.from_settings(
            [
                "#model=macepolm",
                task_line,
                "#device=cpu",
                "#solv(implicit=water,method=smd,provider=torch,"
                f"profile={PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE},"
                "response=frozen,experimental=true)",
            ]
        )


def test_direct_dispatch_keeps_v3_md_closed(tmp_path):
    atoms = _water()
    atoms.calc = _fake_calculator()
    with pytest.raises(
        ValueError, match="TS, IRC, MD, and path workflows remain closed"
    ):
        Dispatcher()({}, "md", atoms, str(tmp_path / "md.out"))
