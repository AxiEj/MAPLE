from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.function.calculator.route2._mace_mdp_polar_hybrid_calculator import (
    MACE_MDPPOLARHybridCalculator,
)
from maple.function.calculator.route2._mace_mdp_polar_hybrid_ddx_calculator import (
    MACE_MDPPOLARHybridDDXCalculator,
)
from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.read.command_control import CommandControl


@dataclass(frozen=True)
class _Energy:
    name: str
    value_eV: float


@dataclass(frozen=True)
class _Force:
    name: str


class _Result:
    profile_id = "test-profile"
    scalar_id = "test-scalar"
    state_equation_id = "test-root"
    root_sha256 = "a" * 64
    primal_residual = 2.0e-11
    adjoint_residual = 4.0e-11
    force_error_estimate_eV_per_A = None
    warnings = ("experimental",)
    energy_components = (_Energy("vacuum", -3.0), _Energy("polarization", -0.2))

    def __init__(self, *, forces: bool) -> None:
        self.total_energy_eV = -3.2
        self.total_forces_eV_per_A = (
            ((0.1, 0.2, 0.3), (-0.1, -0.2, -0.3)) if forces else None
        )
        self.force_components = (_Force("analytic-adjoint"),) if forces else ()


class _PES:
    force_derivative_kind = "operational-implicit-adjoint-total-derivative-v1"

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def evaluate(self, atoms, *, need_forces: bool):
        assert len(atoms) == 2
        self.calls.append(need_forces)
        return _Result(forces=need_forces)


class _Radial:
    dtype = "torch.float64"
    device = "cuda"


class _DDXState:
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


class _DDXForceEvaluation:
    evaluation_sha256 = "d" * 64
    total_forces_eV_per_A = np.asarray(
        [[0.2, 0.0, -0.1], [-0.2, 0.0, 0.1]], dtype=float
    )
    electrostatic_evaluation = SimpleNamespace(adjoint_residual_ev=7.0e-11)


class _DDXVirial:
    raw_virial_eV = np.asarray(
        [[0.3, 0.1, 0.0], [-0.1, 0.2, 0.0], [0.0, 0.0, 0.4]], dtype=float
    )
    symmetric_virial_eV = 0.5 * (raw_virial_eV + raw_virial_eV.T)
    origin_angstrom = np.asarray([0.6, 0.0, 0.0], dtype=float)
    evaluation_sha256 = "e" * 64


class _DDXHessian:
    hessian_eV_per_A2 = np.diag(np.arange(1.0, 7.0))
    evaluation_sha256 = "1" * 64
    coarse_step_angstrom = 0.001
    fine_step_angstrom = 0.0005
    maximum_error_estimate_eV_per_A2 = 0.0001
    maximum_antisymmetry_eV_per_A2 = 0.0002
    topology_step_reductions_used = 1
    topology_guard_status = "observed-components-only-experimental-v1"
    topology_observation_coverage = "partial"
    unobservable_topology_components = ("test-smd-surface",)


class _DDXPES:
    provider_id = "test-ddx-provider"
    scalar_contract_id = "test-ddx-scalar"
    daily_profile_id = "route2-experimental-mace-mdp-polar-separated-ddx-smd-daily-v1"
    scientific_status = "experimental-test-surface"
    coordinate_derivative_available = True

    def __init__(self) -> None:
        self.solve_calls = 0
        self.force_calls = 0
        self.hessian_calls = 0

    def configuration_sha256(self):
        return "f" * 64

    def solve(self, atoms):
        assert len(atoms) == 2
        self.solve_calls += 1
        return _DDXState()

    def evaluate_forces(self, atoms, *, central_state=None):
        assert len(atoms) == 2
        if central_state is not None:
            assert isinstance(central_state, _DDXState)
        self.force_calls += 1
        return _DDXForceEvaluation()

    def molecular_virial(self, atoms, *, force_evaluation=None):
        assert len(atoms) == 2
        assert isinstance(force_evaluation, _DDXForceEvaluation)
        return _DDXVirial()

    def evaluate_hessian(self, atoms):
        assert len(atoms) == 2
        self.hessian_calls += 1
        return _DDXHessian()


class _EnergyOnlyDDXPES(_DDXPES):
    daily_profile_id = "test-energy-only-ddx-daily-profile"
    coordinate_derivative_available = False


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        MACE_MDPPOLARHybridCalculator,
        "_load_hybrid",
        staticmethod(lambda **_kwargs: (object(), _Radial())),
    )
    pes = _PES()
    monkeypatch.setattr(
        MACE_MDPPOLARHybridCalculator,
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


@pytest.fixture
def fake_ddx_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        MACE_MDPPOLARHybridDDXCalculator,
        "_load_hybrid",
        staticmethod(lambda **_kwargs: (object(), _Radial())),
    )
    pes = _DDXPES()
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


def test_hybrid_calculator_exposes_ase_energy_force_and_structured_identity(
    fake_runtime,
) -> None:
    pes, checkpoints = fake_runtime
    calculator = MACE_MDPPOLARHybridCalculator(device="cuda", **checkpoints)
    atoms = _atoms()
    atoms.calc = calculator
    assert atoms.get_potential_energy() == pytest.approx(-3.2)
    np.testing.assert_allclose(
        atoms.get_forces(),
        [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
    )

    assert calculator.results["energy"] == pytest.approx(-3.2)
    assert calculator.results["free_energy"] == pytest.approx(-3.2)
    assert calculator.results["route2"]["profile_id"] == "test-profile"
    assert calculator.results["route2"]["energy_components_eV"] == {
        "vacuum": -3.0,
        "polarization": -0.2,
    }
    assert calculator.results["route2"]["adjoint_residual"] == pytest.approx(4.0e-11)
    assert calculator.results["route2"]["force_components"] == ("analytic-adjoint",)
    assert pes.calls == [False, True]


def test_energy_only_call_clears_stale_forces(fake_runtime) -> None:
    pes, checkpoints = fake_runtime
    calculator = MACE_MDPPOLARHybridCalculator(device="cuda", **checkpoints)
    atoms = _atoms()
    calculator.calculate(atoms, properties=["forces"])
    assert "forces" in calculator.results
    calculator.calculate(atoms, properties=["energy"])
    assert "forces" not in calculator.results
    assert pes.calls == [True, False]


def test_factory_discovers_hybrid_calculator(fake_runtime, tmp_path) -> None:
    _pes, checkpoints = fake_runtime
    calculator = SetCalculator(
        device="cuda",
        model="macemdppolarhybrid",
        output=str(tmp_path / "maple.out"),
        atoms=_atoms(),
        model_options={key: str(value) for key, value in checkpoints.items()},
    ).set_calculator()
    assert isinstance(calculator, MACE_MDPPOLARHybridCalculator)
    atoms = _atoms()
    atoms.calc = calculator
    assert atoms.get_potential_energy() == pytest.approx(-3.2)
    np.testing.assert_allclose(
        atoms.get_forces(),
        [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
    )


def test_hessian_and_unsupported_properties_fail_closed(fake_runtime) -> None:
    _pes, checkpoints = fake_runtime
    calculator = MACE_MDPPOLARHybridCalculator(device="cuda", **checkpoints)
    with pytest.raises(NotImplementedError, match="H is not admitted"):
        calculator.get_hessian(_atoms())
    with pytest.raises(NotImplementedError, match="only E/F"):
        calculator.calculate(_atoms(), properties=["stress"])


def test_runtime_contract_rejects_cpu_and_second_solvent(fake_runtime) -> None:
    _pes, checkpoints = fake_runtime
    with pytest.raises(ValueError, match="requires device='cuda'"):
        MACE_MDPPOLARHybridCalculator(device="cpu", **checkpoints)
    with pytest.raises(ValueError, match="restricted to water-cavity"):
        MACE_MDPPOLARHybridCalculator(device="cuda", solvent="ethanol", **checkpoints)


def test_maple_daily_job_contract_opens_sp_and_first_order_opt_only(
    fake_runtime,
) -> None:
    _pes, checkpoints = fake_runtime
    calculator = MACE_MDPPOLARHybridCalculator(device="cuda", **checkpoints)
    calculator.validate_maple_job(jobtype="sp")
    for method in (None, "lbfgs", "sd", "sdcg", "cg"):
        calculator.validate_maple_job(jobtype="opt", params={"method": method})
    with pytest.raises(NotImplementedError, match="RFO requires H"):
        calculator.validate_maple_job(jobtype="opt", params={"method": "rfo"})
    for jobtype in ("freq", "ts", "irc", "md", "scan"):
        with pytest.raises(NotImplementedError, match="SP and first-order OPT"):
            calculator.validate_maple_job(jobtype=jobtype)


def test_dispatcher_enforces_hybrid_job_contract_before_job_import(
    fake_runtime, monkeypatch, tmp_path
) -> None:
    _pes, checkpoints = fake_runtime
    atoms = _atoms()
    atoms.calc = MACE_MDPPOLARHybridCalculator(device="cuda", **checkpoints)
    dispatcher = Dispatcher()
    called = []
    monkeypatch.setattr(
        Dispatcher,
        "_dispatch_legacy",
        lambda self, commandcontrol, jobtype, atoms, output, extra: called.append(
            jobtype
        ),
    )
    command = {"method": "lbfgs", "level": "medium"}
    dispatcher(command, "sp", atoms, str(tmp_path / "maple.out"))
    dispatcher(command, "opt", atoms, str(tmp_path / "maple.out"))
    assert called == ["sp", "opt"]
    with pytest.raises(NotImplementedError, match="SP and first-order OPT"):
        dispatcher(command, "md", atoms, str(tmp_path / "maple.out"))


def test_dispatcher_runs_one_step_hybrid_first_order_opt(
    fake_runtime, tmp_path
) -> None:
    _pes, checkpoints = fake_runtime
    atoms = _atoms()
    atoms.calc = MACE_MDPPOLARHybridCalculator(device="cuda", **checkpoints)
    initial_positions = atoms.positions.copy()
    command = SimpleNamespace(
        params={"method": "lbfgs", "level": "medium", "max_iter": 1, "verbose": 0}
    )
    output = tmp_path / "hybrid-opt.out"

    Dispatcher()(command, "opt", atoms, str(output))

    assert not np.array_equal(atoms.positions, initial_positions)
    assert atoms.calc.__class__ is MACE_MDPPOLARHybridCalculator
    assert output.is_file()
    assert (tmp_path / "hybrid-opt_opt.xyz").is_file()
    assert (tmp_path / "hybrid-opt_opt_traj.xyz").is_file()


def test_dispatcher_runs_hybrid_single_point_with_legacy_report_units(
    fake_runtime, tmp_path
) -> None:
    pes, checkpoints = fake_runtime
    atoms = _atoms()
    calculator = MACE_MDPPOLARHybridCalculator(device="cuda", **checkpoints)
    atoms.calc = calculator
    command = SimpleNamespace(params={"level": "medium", "verbose": 1})
    output = tmp_path / "hybrid-sp.out"

    Dispatcher()(command, "sp", atoms, str(output))

    report = output.read_text()
    assert f"Energy: {-3.2 * EV2HARTREE:.10f} Hartree" in report
    assert "Gradients (Hartree/Angstrom):" in report
    assert pes.calls == [True]
    assert atoms.calc is calculator
    assert calculator.results["energy"] == pytest.approx(-3.2)


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

    loaded = MACE_MDPPOLARHybridCalculator._load_hybrid(
        device="cuda",
        mdp_checkpoint_path=mdp_path,
        polar_checkpoint_path=polar_path,
    )
    assert loaded == (hybrid, radial)
    assert calls["mdp"] == {"checkpoint_path": mdp_path, "device": "cpu"}
    assert calls["radial"] == {
        "checkpoint_path": polar_path,
        "device": "cuda",
        "long_range_evaluator_profile": (
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    }
    assert calls["hybrid"] == {"permanent": mdp, "response": response}


def test_hybrid_ddx_daily_surface_exposes_energy_force_and_molecular_virial(
    fake_ddx_runtime,
) -> None:
    pes, checkpoints = fake_ddx_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(
        device="cuda", solvent="water", **checkpoints
    )
    atoms = _atoms()
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(-4.3)
    np.testing.assert_allclose(
        atoms.get_forces(), _DDXForceEvaluation.total_forces_eV_per_A
    )
    np.testing.assert_allclose(
        calculator.get_molecular_virial(atoms), _DDXVirial.raw_virial_eV
    )
    np.testing.assert_allclose(
        calculator.get_molecular_virial(atoms, symmetric=True),
        _DDXVirial.symmetric_virial_eV,
    )

    route2 = calculator.results["route2"]
    assert route2["solvent"] == "water"
    assert route2["energy_components_eV"] == {
        "vacuum": -4.0,
        "ddx_polarization": -0.4,
        "smd_cds": 0.1,
        "solvation_total": -0.3,
    }
    assert route2["force_derivative_kind"] == (
        "analytic-block-adjoint-plus-solvent-gradient-v1"
    )
    assert route2["daily_job_capabilities"] == (
        "sp",
        "opt:first-order",
        "freq:vibrational-only-experimental",
        "ts:neb-cineb-experimental",
        "hessian:direct-experimental-richardson",
        "virial:direct-nonperiodic-molecular",
    )
    assert route2["daily_property_availability"] == {
        "energy": "experimental",
        "forces": "experimental-same-scalar-analytic",
        "hessian": "experimental-richardson",
        "molecular_virial": "experimental-nonperiodic",
        "periodic_stress": False,
        "strict_variational_functional": False,
        "molecular_dynamics": False,
    }
    assert route2["adjoint_residual_eV"] == pytest.approx(7.0e-11)
    assert route2["topology_observation_coverage"] == "partial"
    assert calculator.results["solvation"] == {
        "gas_energy_hartree": pytest.approx(-4.0 * EV2HARTREE),
        "delta_g_solv_hartree": pytest.approx(-0.3 * EV2HARTREE),
        "combined_energy_hartree": pytest.approx(-4.3 * EV2HARTREE),
        "provenance": {
            "profile_id": (
                "route2-experimental-mace-mdp-polar-separated-ddx-smd-daily-v1"
            ),
            "solvent": "water",
            "standard_state": (
                "profile-defined electronic-plus-continuum ledger; no "
                "thermochemical Gibbs correction"
            ),
        },
    }
    assert pes.solve_calls == 2


def test_hybrid_ddx_daily_surface_opens_only_supported_workflows(
    fake_ddx_runtime,
) -> None:
    pes, checkpoints = fake_ddx_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(
        device="cuda", solvent="water", **checkpoints
    )
    np.testing.assert_allclose(
        calculator.get_hessian(_atoms()), _DDXHessian.hessian_eV_per_A2
    )
    assert pes.hessian_calls == 1
    assert calculator.hessian_diagnostics["topology_step_reductions_used"] == 1
    assert calculator.hessian_diagnostics["topology_observation_coverage"] == "partial"
    assert calculator.hessian_diagnostics["maximum_error_estimate_eV_per_A2"] == 0.0001
    calculator.validate_maple_job(jobtype="sp")
    calculator.validate_maple_job(jobtype="opt", params={"method": "lbfgs"})
    with pytest.raises(NotImplementedError, match="first-order"):
        calculator.validate_maple_job(jobtype="opt", params={"method": "rfo"})
    calculator.validate_maple_job(jobtype="freq")
    assert calculator.FREQUENCY_THERMOCHEMISTRY == "none"
    for refine in (None, "cineb"):
        calculator.validate_maple_job(
            jobtype="ts", params={"method": "neb", "refine": refine}
        )
    for params in (
        {"method": "neb", "refine": "nebts"},
        {"method": "neb", "refine": "unknown"},
        {"method": "prfo"},
        {"method": "dimer"},
        {"method": "autoneb"},
        {},
    ):
        with pytest.raises(NotImplementedError, match="NEB/CINEB"):
            calculator.validate_maple_job(jobtype="ts", params=params)
    for jobtype in ("md", "irc", "scan"):
        with pytest.raises(NotImplementedError, match="remain closed"):
            calculator.validate_maple_job(jobtype=jobtype)


def test_hybrid_ddx_energy_remains_callable_when_derivatives_fail_closed(
    fake_ddx_runtime, monkeypatch
) -> None:
    _pes, checkpoints = fake_ddx_runtime
    energy_only = _EnergyOnlyDDXPES()
    monkeypatch.setattr(
        MACE_MDPPOLARHybridDDXCalculator,
        "_build_pes",
        lambda self, atoms: energy_only,
    )
    calculator = MACE_MDPPOLARHybridDDXCalculator(
        device="cuda", solvent="water", **checkpoints
    )
    atoms = _atoms()
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(-4.3)
    availability = calculator.results["route2"]["daily_property_availability"]
    assert availability["energy"] == "experimental"
    assert availability["forces"] is False
    assert availability["hessian"] is False
    assert availability["molecular_virial"] is False

    with pytest.raises(NotImplementedError, match="complete coordinate derivative"):
        calculator.calculate(atoms, properties=["forces"])
    with pytest.raises(NotImplementedError, match="complete coordinate derivative"):
        calculator.get_molecular_virial(atoms)
    with pytest.raises(NotImplementedError, match="complete coordinate derivative"):
        calculator.get_hessian(atoms)


def test_hybrid_ddx_neb_identity_binds_the_actual_pes(fake_ddx_runtime) -> None:
    pes, checkpoints = fake_ddx_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(device="cuda", **checkpoints)
    assert calculator.maple_neb_pes_identity(_atoms()) == (
        "macemdppolarhybridddx", pes.daily_profile_id, pes.configuration_sha256()
    )
    assert pes.solve_calls == 0


def test_hybrid_ddx_hessian_shrinks_stencils_without_weakening_guards(monkeypatch):
    import maple.solvation.experimental as experimental
    from maple.solvation.derivatives import OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1

    recorded = {}
    def build(hybrid, symbols, **kwargs):
        recorded.update(kwargs)
        return object()
    monkeypatch.setattr(experimental, "build_smd_mace_mdp_polar_hybrid_ddx_pes", build)
    calculator = object.__new__(MACE_MDPPOLARHybridDDXCalculator)
    calculator._hybrid = object()
    calculator.solvent = "water"
    calculator._build_pes(_atoms())
    policy = recorded["hessian_backend"]
    assert policy.maximum_topology_step_reductions == 6
    assert policy.coarse_step_angstrom == 0.002
    assert policy.maximum_error_eV_per_A2 == 0.005
    assert policy.maximum_antisymmetry_eV_per_A2 == 0.005
    assert policy.topology_guard_policy == OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1


@pytest.mark.parametrize("metadata", [
    {"charge": -1}, {"charge": float("nan")}, {"mult": 3},
    {"multiplicity": 2}, {"mult": 1, "multiplicity": 3},
])
def test_hybrid_ddx_rejects_unsupported_state_even_after_cache(
    fake_ddx_runtime, metadata
) -> None:
    pes, checkpoints = fake_ddx_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(device="cuda", **checkpoints)
    atoms = _atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    calls = pes.solve_calls
    atoms.info.update(metadata)
    for operation in (
        atoms.get_potential_energy, atoms.get_forces,
        lambda: calculator.get_hessian(atoms),
        lambda: calculator.get_molecular_virial(atoms),
        lambda: calculator.maple_neb_pes_identity(atoms),
    ):
        with pytest.raises(ValueError, match="neutral singlet"):
            operation()
    assert pes.solve_calls == calls


@pytest.mark.parametrize("kind", ["pbc", "initial_charge", "magmom", "odd_electrons"])
def test_hybrid_ddx_rejects_other_unsupported_state_markers(fake_ddx_runtime, kind):
    pes, checkpoints = fake_ddx_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(device="cuda", **checkpoints)
    atoms = _atoms()
    if kind == "pbc":
        atoms.pbc = True
    elif kind == "initial_charge":
        atoms.set_initial_charges([0.5, 0.5])
    elif kind == "magmom":
        atoms.set_initial_magnetic_moments([1.0, -1.0])
    else:
        atoms.numbers[0] = 7
    with pytest.raises(ValueError, match="neutral singlet|nonperiodic"):
        calculator.get_hessian(atoms)
    assert pes.solve_calls == 0


def test_factory_discovers_hybrid_ddx_daily_surface(fake_ddx_runtime, tmp_path) -> None:
    _pes, checkpoints = fake_ddx_runtime
    calculator = SetCalculator(
        device="cuda",
        model="macemdppolarhybridddx",
        solvent="water",
        output=str(tmp_path / "maple.out"),
        atoms=_atoms(),
        model_options={
            "hessian": "numerical",
            **{key: str(value) for key, value in checkpoints.items()},
        },
    ).set_calculator()
    assert isinstance(calculator, MACE_MDPPOLARHybridDDXCalculator)
    assert calculator.hessian == "numerical"
    assert calculator.solvent == "water"


def test_factory_accepts_owned_solvent_from_model_options(
    fake_ddx_runtime, tmp_path
) -> None:
    _pes, checkpoints = fake_ddx_runtime
    calculator = SetCalculator(
        device="cuda",
        model="macemdppolarhybridddx",
        output=str(tmp_path / "maple.out"),
        atoms=_atoms(),
        model_options={
            "solvent": "etoh",
            **{key: str(value) for key, value in checkpoints.items()},
        },
    ).set_calculator()
    assert isinstance(calculator, MACE_MDPPOLARHybridDDXCalculator)
    assert calculator.solvent == "ethanol"


def test_maple_header_reaches_named_solvent_experimental_surface(
    fake_ddx_runtime, tmp_path
) -> None:
    _pes, checkpoints = fake_ddx_runtime
    mdp = checkpoints["mdp_checkpoint_path"]
    polar = checkpoints["polar_checkpoint_path"]
    command = CommandControl.from_settings(
        [
            "#model=macemdppolarhybridddx("
            f"solvent=ethanol,mdp_checkpoint_path={mdp},"
            f"polar_checkpoint_path={polar},hessian=numerical)",
            "#sp",
        ]
    )
    calculator = SetCalculator(
        device="cuda",
        model=command.params["model"],
        output=str(tmp_path / "maple.out"),
        atoms=_atoms(),
        model_options=command.params["model_options"],
    ).set_calculator()
    assert isinstance(calculator, MACE_MDPPOLARHybridDDXCalculator)
    assert calculator.solvent == "ethanol"


@pytest.mark.parametrize("header", [
    "#sp", "#opt(method=lbfgs)", "#freq(method=mw,thermochemistry=none)",
    "#ts(method=neb)", "#ts(method=neb,refine=cineb)",
    "#ts(method=neb,initial_opt=true)",
])
def test_model_header_survives_all_supported_job_parsers(fake_ddx_runtime, header):
    _pes, checkpoints = fake_ddx_runtime
    command = CommandControl.from_settings([
        "#model=macemdppolarhybridddx(solvent=water,hessian=numerical)",
        "#device=cuda", header,
    ])
    assert command.params["model"] == "macemdppolarhybridddx"
    assert command.params["model_options"]["solvent"] == "water"
    assert str(command.params["device"]).startswith("cuda")
    calculator = MACE_MDPPOLARHybridDDXCalculator(device="cuda", **checkpoints)
    calculator.validate_maple_job(jobtype=command.task, params=command.params)


def test_model_header_does_not_disable_unknown_opt_parameter_check():
    with pytest.raises(ValueError, match="Unknown OPT parameter: 'max_itre'"):
        CommandControl.from_settings([
            "#model=macemdppolarhybridddx(solvent=water)",
            "#opt(method=lbfgs,max_itre=2)",
        ])


def test_explicit_frequency_device_does_not_depend_on_header_order():
    before = CommandControl.from_settings(["#device=cuda", "#freq"])
    after = CommandControl.from_settings(["#freq", "#device=cuda"])
    assert before.params["device"] == after.params["device"]
    assert str(before.params["device"]).startswith("cuda")


@pytest.mark.parametrize("header", [
    "#sp", "#opt(method=lbfgs)", "#freq(method=mw,thermochemistry=none)",
    "#ts(method=neb,refine=cineb)",
])
def test_full_input_reader_preserves_hybrid_model_and_cuda(tmp_path, header):
    from maple.function.read.input_reader import InputReader

    geometry = "0 1\nO 0 0 0\nH 0.97 0 0\nH -0.24 0.94 0\n"
    text = (
        "#model=macemdppolarhybridddx(solvent=water,hessian=numerical)\n"
        "#device=cuda\n" + header + "\n\n" + geometry
    )
    if header.startswith("#ts"):
        text += "\n&\n" + geometry
    source = tmp_path / "job.inp"
    source.write_text(text)
    reader = InputReader()
    structures = reader(str(source), str(tmp_path / "job.out"))
    assert reader.model == "macemdppolarhybridddx"
    assert str(reader.device).startswith("cuda")
    if header.startswith("#ts"):
        assert len(structures.multiatoms) == 2
    else:
        assert len(structures) == 3


def test_hybrid_ddx_daily_surface_accepts_registered_named_solvent(
    fake_ddx_runtime,
) -> None:
    _pes, checkpoints = fake_ddx_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(
        device="cuda", solvent="etoh", **checkpoints
    )
    assert calculator.solvent == "ethanol"


@pytest.mark.parametrize("device", ("cuda", "cuda:0", "gpu0"))
def test_hybrid_ddx_accepts_maple_cuda_device_aliases(
    fake_ddx_runtime, device
) -> None:
    _pes, checkpoints = fake_ddx_runtime
    calculator = MACE_MDPPOLARHybridDDXCalculator(device=device, **checkpoints)
    assert calculator.device == "cuda"


def test_hybrid_ddx_rejects_implicit_double_counting_and_periodic_stress(
    fake_ddx_runtime,
) -> None:
    _pes, checkpoints = fake_ddx_runtime
    with pytest.raises(ValueError, match="already owns ddX"):
        MACE_MDPPOLARHybridDDXCalculator(device="cuda", implicit="smd", **checkpoints)
    with pytest.raises(ValueError, match="D4 composition is not admitted"):
        MACE_MDPPOLARHybridDDXCalculator(device="cuda", d4=True, **checkpoints)
    with pytest.raises(ValueError, match="D4 composition is not admitted"):
        SetCalculator(
            device="cuda",
            model="macemdppolarhybridddx",
            d4=True,
            output="maple.out",
            atoms=_atoms(),
            model_options={key: str(value) for key, value in checkpoints.items()},
        ).set_calculator()
    calculator = MACE_MDPPOLARHybridDDXCalculator(device="cuda", **checkpoints)
    with pytest.raises(NotImplementedError, match="molecular virial"):
        calculator.calculate(_atoms(), properties=["stress"])
