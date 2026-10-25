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
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
    pure_macepolar_expected_device,
    route2_smd_profile_spec,
    route2_smd_profiles_for_provider,
    validate_route2_smd_profile,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V4,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4,
    PURE_MACEPOLAR_TORCH_FP32_V4_ENGINEERING_EVIDENCE_ID,
    get_scalar_definition,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives.analytic import (
    AnalyticForceEvaluation,
    AnalyticHVPEvaluation,
)
from maple.solvation.models.base import model_input_sha256


def _water() -> Atoms:
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


class _FakeTorchPES:
    provider_id = "test-torch-pes"

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

    def evaluate_forces(self, atoms) -> AnalyticForceEvaluation:
        version = 4 if self.scalar_contract_id.endswith("-v4") else 3
        kind = "cuda" if self.device.startswith("cuda") else "cpu"
        return AnalyticForceEvaluation(
            energy_eV=self.get_potential_energy(atoms),
            forces_eV_per_A=self.get_forces(atoms),
            geometry_sha256=geometry_sha256(atoms),
            model_input_sha256=model_input_sha256(atoms),
            configuration_sha256=self.configuration_sha256(),
            scalar_contract_id=self.scalar_contract_id,
            device=self.device,
            component_energies_eV={"test": -2.5},
            provider_id=self.provider_id,
            profile_id=f"pure-macepolar-frozen-point-l1-ddpcm-smd-torch-{kind}-v{version}",
            verification_evidence_id=(
                "route2-pure-macepolar-torch-fp32-v4-engineering-tests"
                if version == 4
                else "route2-pure-macepolar-torch-analytic-v3-contract-tests"
            ),
            computation_dtype="float32" if version == 4 else "float64",
        )

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

    def evaluate_hvp(self, atoms, direction):
        force_evaluation = self.evaluate_forces(atoms)
        dtype = (
            np.float32
            if force_evaluation.computation_dtype == "float32"
            else np.float64
        )
        probe = np.asarray(direction, dtype=dtype)
        return AnalyticHVPEvaluation(
            force_evaluation=force_evaluation,
            direction=probe,
            hvp_eV_per_A2=self.hessian_vector_product(atoms, probe),
        )


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
    assert {cpu.name, cuda.name}.issubset(route2_smd_profiles_for_provider("torch"))
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


def test_v4_profiles_register_fp32_l5_n194_without_mutating_v3():
    cpu = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    )
    cuda = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE
    )
    v3 = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
    )

    assert cpu.provider == cuda.provider == "torch"
    assert cpu.default_eligible and cuda.default_eligible
    assert pure_macepolar_expected_device(cpu) == "cpu"
    assert pure_macepolar_expected_device(cuda) == "cuda"
    assert "float32" in cpu.electronic_profile_binding
    assert "ddpcm-l5-n194" in cpu.electronic_energy_semantics
    assert "float64" in v3.electronic_profile_binding
    assert not v3.default_eligible


@pytest.mark.parametrize(
    "scalar_id",
    [
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V4,
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4,
    ],
)
def test_v4_scalar_registry_is_experimental_and_has_only_v4_evidence(scalar_id):
    scalar = get_scalar_definition(scalar_id)
    assert scalar.experimental_execution.energy
    assert scalar.experimental_execution.force
    assert scalar.derivative_route.startswith("torch.float32 autograd")
    assert "float32-l5-n194" in scalar.continuum_profile
    assert scalar.experimental_evidence_artifact_ids == (
        PURE_MACEPOLAR_TORCH_FP32_V4_ENGINEERING_EVIDENCE_ID,
    )
    assert scalar.evidence_artifact_ids == ()
    assert not scalar.enabled


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


def test_set_calculator_defaults_torch_to_v4_and_routes_profile(monkeypatch, tmp_path):
    atoms = _water()
    profile = PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    fake = _FakeTorchPES(EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V4)
    observed = {}

    def fake_builder(symbols, **kwargs):
        observed.update(kwargs)
        return fake

    monkeypatch.setattr(
        "maple.solvation.experimental.mace_polar_torch."
        "build_smd_mace_polar_torch_pes",
        fake_builder,
    )
    calculator = SetCalculator(
        device="cpu",
        model="macepolm",
        output=str(tmp_path / "factory-v4.out"),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        solvation_options={
            "method": "smd",
            "implicit": "water",
            "experimental": True,
            "provider": "torch",
            "standard_state": "1m",
        },
    ).set_calculator()

    assert calculator.pes is fake
    assert observed["profile"] == profile
    assert calculator.profile_spec.allowed_response_modes == ("frozen",)
    log = (tmp_path / "factory-v4.out").read_text()
    assert "dtype=float32 response=frozen" in log
    assert "torch-dense-l5/n194/tol=100eps(float32)/eta0.1" in log
    assert "explicit float64 intermediates" in log


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


def test_input_contract_accepts_explicit_v4_single_point():
    command = CommandControl.from_settings(
        [
            "#model=macepolm",
            "#sp",
            "#device=cpu",
            "#solv(implicit=water,method=smd,provider=torch,"
            f"profile={PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE},"
            "response=frozen,experimental=true)",
        ]
    )
    assert command.params["solv"]["profile"].endswith("torch-cpu-v4")


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE),
        ("cuda:0", PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE),
    ],
)
def test_input_contract_defaults_only_torch_to_device_specific_v4(device, expected):
    command = CommandControl.from_settings(
        [
            "#model=macepolm",
            "#sp",
            f"#device={device}",
            "#solv(implicit=water,method=smd,provider=torch," "experimental=true)",
        ]
    )
    assert command.params["solv"]["profile"] == expected
    assert command.params["solv"]["response"] == "frozen"


def test_input_contract_keeps_explicit_v3_unchanged():
    command = CommandControl.from_settings(
        [
            "#model=macepolm",
            "#sp",
            "#device=cpu",
            "#solv(implicit=water,method=smd,provider=torch,"
            f"profile={PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE},"
            "response=frozen,experimental=true)",
        ]
    )
    assert command.params["solv"]["profile"].endswith("torch-cpu-v3")


def test_non_torch_provider_still_requires_explicit_profile():
    with pytest.raises(ValueError, match="requires an explicit versioned profile"):
        CommandControl.from_settings(
            [
                "#model=macepolm",
                "#sp",
                "#device=cpu",
                "#solv(implicit=water,method=smd,provider=pyddx,"
                "response=frozen,experimental=true)",
            ]
        )


def test_torch_explicit_scf_is_not_replaced_by_default():
    with pytest.raises(ValueError, match="requires response in: frozen"):
        CommandControl.from_settings(
            [
                "#model=macepolm",
                "#sp",
                "#device=cpu",
                "#solv(implicit=water,method=smd,provider=torch,"
                "response=scf,experimental=true)",
            ]
        )


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
