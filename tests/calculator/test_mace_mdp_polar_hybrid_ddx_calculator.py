from __future__ import annotations

from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.calculator.route2 import MACE_MDPPOLARHybridDDXCalculator
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.read.command_control import CommandControl


class _Radial:
    dtype = "torch.float64"
    device = "cuda"


class _State:
    provider_id = "test-ddx-provider"
    state_sha256 = "b" * 64
    topology_observation_coverage = "partial"
    unobservable_topology_components = ("test-smd-surface",)
    vacuum_energy_eV = -4.0
    polarization_energy_eV = -0.4
    cds_energy_eV = 0.1
    solvation_energy_eV = -0.3
    total_energy_eV = -4.3
    electrostatic_state = SimpleNamespace(root_sha256="c" * 64)


class _ForceEvaluation:
    evaluation_sha256 = "d" * 64
    total_forces_eV_per_A = np.asarray(
        [[0.2, 0.0, -0.1], [-0.2, 0.0, 0.1]], dtype=float
    )
    electrostatic_evaluation = SimpleNamespace(adjoint_residual_ev=7.0e-11)


class _Virial:
    raw_virial_eV = np.asarray(
        [[0.3, 0.1, 0.0], [-0.1, 0.2, 0.0], [0.0, 0.0, 0.4]], dtype=float
    )
    symmetric_virial_eV = 0.5 * (raw_virial_eV + raw_virial_eV.T)
    origin_angstrom = np.asarray([0.6, 0.0, 0.0], dtype=float)
    evaluation_sha256 = "e" * 64


class _Hessian:
    hessian_eV_per_A2 = np.diag(np.arange(1.0, 7.0))


class _PES:
    provider_id = "test-ddx-provider"
    scalar_contract_id = "test-ddx-scalar"
    scientific_status = "experimental-test-surface"

    def __init__(self) -> None:
        self.solve_calls = 0
        self.force_calls = 0
        self.hessian_calls = 0

    def configuration_sha256(self):
        return "f" * 64

    def solve(self, atoms):
        assert len(atoms) == 2
        self.solve_calls += 1
        return _State()

    def evaluate_forces(self, atoms, *, central_state=None):
        assert len(atoms) == 2
        if central_state is not None:
            assert isinstance(central_state, _State)
        self.force_calls += 1
        return _ForceEvaluation()

    def molecular_virial(self, atoms, *, force_evaluation=None):
        assert len(atoms) == 2
        assert isinstance(force_evaluation, _ForceEvaluation)
        return _Virial()

    def evaluate_hessian(self, atoms):
        assert len(atoms) == 2
        self.hessian_calls += 1
        return _Hessian()


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        MACE_MDPPOLARHybridDDXCalculator,
        "_load_hybrid",
        staticmethod(lambda **_kwargs: (object(), _Radial())),
    )
    pes = _PES()
    monkeypatch.setattr(
        MACE_MDPPOLARHybridDDXCalculator,
        "_build_pes",
        lambda self, atoms: pes,
    )
    mdp = tmp_path / "MACE-MDP.model"
    polar = tmp_path / "MACEPOLAR1Mmodel"
    mdp.write_bytes(b"test-mdp")
    polar.write_bytes(b"test-polar")
    return pes, {
        "mdp_checkpoint_path": mdp,
        "polar_checkpoint_path": polar,
    }


def _atoms() -> Atoms:
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    return atoms


def test_daily_surface_exposes_energy_force_hessian_and_molecular_virial(
    fake_runtime,
) -> None:
    pes, checkpoints = fake_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(
        device="cuda", solvent="water", **checkpoints
    )
    atoms = _atoms()
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(-4.3)
    np.testing.assert_allclose(
        atoms.get_forces(), _ForceEvaluation.total_forces_eV_per_A
    )
    np.testing.assert_allclose(
        calculator.get_hessian(atoms), _Hessian.hessian_eV_per_A2
    )
    np.testing.assert_allclose(
        calculator.get_molecular_virial(atoms), _Virial.raw_virial_eV
    )
    np.testing.assert_allclose(
        calculator.get_molecular_virial(atoms, symmetric=True),
        _Virial.symmetric_virial_eV,
    )

    route2 = calculator.results["route2"]
    assert route2["solvent"] == "water"
    assert route2["energy_components_eV"] == {
        "vacuum": -4.0,
        "ddx_polarization": -0.4,
        "smd_cds": 0.1,
        "solvation_total": -0.3,
    }
    assert route2["daily_job_capabilities"] == (
        "sp",
        "opt:first-order",
        "hessian:direct-experimental-richardson",
        "virial:direct-nonperiodic-molecular",
    )
    assert (
        route2["daily_property_availability"]["strict_variational_functional"] is False
    )
    assert route2["adjoint_residual_eV"] == pytest.approx(7.0e-11)
    assert pes.hessian_calls == 1


def test_factory_discovers_named_solvent_daily_surface(fake_runtime, tmp_path) -> None:
    _pes, checkpoints = fake_runtime
    calculator = SetCalculator(
        device="cuda",
        model="macemdppolarhybridddx",
        output=str(tmp_path / "maple.out"),
        atoms=_atoms(),
        model_options={
            "module": (
                "maple.function.calculator.route2."
                "_mace_mdp_polar_hybrid_ddx_calculator"
            ),
            "route2_solvent": "etoh",
            "hessian": "numerical",
            **{key: str(value) for key, value in checkpoints.items()},
        },
    ).set_calculator()
    assert isinstance(calculator, MACE_MDPPOLARHybridDDXCalculator)
    assert calculator.hessian == "numerical"
    assert calculator.solvent == "ethanol"


def test_maple_input_header_carries_owned_solvent_without_solv_block() -> None:
    parsed = CommandControl.from_settings(
        [
            "#model=macemdppolarhybridddx("
            "module=maple.function.calculator.route2."
            "_mace_mdp_polar_hybrid_ddx_calculator,"
            "route2_solvent=etoh,hessian=numerical)",
            "#sp(verbose=1)",
        ]
    ).as_dict()
    assert "solv" not in parsed
    assert parsed["model_options"]["route2_solvent"] == "etoh"


def test_daily_surface_accepts_registered_named_solvent(fake_runtime) -> None:
    _pes, checkpoints = fake_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(
        device="cuda:0", solvent="etoh", **checkpoints
    )
    assert calculator.solvent == "ethanol"


def test_daily_surface_opens_sp_and_first_order_opt_only(fake_runtime) -> None:
    _pes, checkpoints = fake_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(device="cuda", **checkpoints)
    calculator.validate_maple_job(jobtype="sp")
    for method in (None, "lbfgs", "sd", "sdcg", "cg"):
        calculator.validate_maple_job(jobtype="opt", params={"method": method})
    with pytest.raises(NotImplementedError, match="first-order"):
        calculator.validate_maple_job(jobtype="opt", params={"method": "rfo"})
    for jobtype in ("freq", "ts", "irc", "md", "scan"):
        with pytest.raises(NotImplementedError, match="gas-phase thermochemistry"):
            calculator.validate_maple_job(jobtype=jobtype)


def test_dispatcher_runs_sp_and_one_step_first_order_opt(
    fake_runtime, tmp_path
) -> None:
    _pes, checkpoints = fake_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(device="cuda", **checkpoints)

    sp_atoms = _atoms()
    sp_atoms.calc = calculator
    sp_output = tmp_path / "hybrid-ddx-sp.out"
    Dispatcher()(
        SimpleNamespace(params={"level": "medium", "verbose": 1}),
        "sp",
        sp_atoms,
        str(sp_output),
    )
    report = sp_output.read_text()
    assert f"Energy: {-4.3 * EV2HARTREE:.10f} Hartree" in report
    assert "Gradients (Hartree/Angstrom):" in report
    assert sp_atoms.calc is calculator
    assert calculator.results["energy"] == pytest.approx(-4.3)

    opt_atoms = _atoms()
    opt_atoms.calc = calculator
    initial = opt_atoms.positions.copy()
    opt_output = tmp_path / "hybrid-ddx-opt.out"
    Dispatcher()(
        SimpleNamespace(
            params={"method": "lbfgs", "level": "medium", "max_iter": 1, "verbose": 0}
        ),
        "opt",
        opt_atoms,
        str(opt_output),
    )
    assert not np.array_equal(opt_atoms.positions, initial)
    assert opt_atoms.calc is calculator
    assert opt_output.is_file()


def test_daily_surface_rejects_double_counting_periodic_stress_and_cpu(
    fake_runtime,
) -> None:
    _pes, checkpoints = fake_runtime
    with pytest.raises(ValueError, match="already owns ddX"):
        MACE_MDPPOLARHybridDDXCalculator(device="cuda", implicit="smd", **checkpoints)
    with pytest.raises(ValueError, match="D4 composition is not admitted"):
        MACE_MDPPOLARHybridDDXCalculator(device="cuda", d4=True, **checkpoints)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        MACE_MDPPOLARHybridDDXCalculator(device="cpu", **checkpoints)
    calculator = MACE_MDPPOLARHybridDDXCalculator(device="cuda", **checkpoints)
    with pytest.raises(NotImplementedError, match="molecular virial"):
        calculator.calculate(_atoms(), properties=["stress"])


def test_default_loader_uses_official_checkpoint_adapters(
    monkeypatch, tmp_path
) -> None:
    import maple.solvation.models as models
    from maple.solvation.models.runtime.analytic_gaussian_multipole import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )

    mdp_path = tmp_path / "mdp.model"
    polar_path = tmp_path / "polar.model"
    mdp_path.write_bytes(b"mdp")
    polar_path.write_bytes(b"polar")
    calls = {}
    mdp = object()
    radial = object()
    response = object()
    hybrid = object()

    def build_mdp(**kwargs):
        calls["mdp"] = kwargs
        return mdp

    def build_radial(**kwargs):
        calls["radial"] = kwargs
        return radial

    def make_response(value):
        assert value is radial
        return response

    def build_hybrid(**kwargs):
        calls["hybrid"] = kwargs
        return hybrid

    monkeypatch.setattr(models, "build_mace_mdp_moment_adapter", build_mdp)
    monkeypatch.setattr(
        models, "build_official_mace_polar_1_m_radial_gto_adapter", build_radial
    )
    monkeypatch.setattr(
        models, "MACEPolarOriginalSourceNativeFieldAdapter", make_response
    )
    monkeypatch.setattr(
        models, "build_mace_mdp_anchored_mace_polar_hybrid", build_hybrid
    )

    loaded = MACE_MDPPOLARHybridDDXCalculator._load_hybrid(
        device="cuda",
        mdp_checkpoint_path=mdp_path,
        polar_checkpoint_path=polar_path,
    )
    assert loaded == (hybrid, radial)
    assert calls["mdp"] == {"checkpoint_path": mdp_path, "device": "cpu"}
    assert calls["radial"] == {
        "checkpoint_path": polar_path,
        "device": "cuda",
        "long_range_evaluator_profile": MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    }
    assert calls["hybrid"] == {"permanent": mdp, "response": response}
