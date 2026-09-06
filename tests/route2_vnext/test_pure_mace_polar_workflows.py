"""Regression tests for the opt-in pure frozen MACE-POLAR workflow surface.

These tests are engineering checks.  They do not promote the development
accuracy evidence to release evidence or establish physical workflow accuracy.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.calculator.route2 import PureMACEPolarDDXCalculator
from maple.function.calculator.route2 import is_pure_mace_polar_workflow_calculator
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
from maple.function.route2_smd_profiles import route2_smd_profile_spec
from maple.solvation.api import EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1

PURE_WORKFLOW_PROFILE = "pure-macepolar-frozen-point-l1-ddpcm-smd-workflow-v1"


def test_public_workflow_marker_is_not_calculator_identity():
    impostor = SimpleNamespace(workflow_kind="pure-frozen-total-pes")
    assert not is_pure_mace_polar_workflow_calculator(impostor)
    assert is_pure_mace_polar_workflow_calculator(_fake_calculator())
    assert is_pure_mace_polar_workflow_calculator(
        LegacyHartreeJobView(_fake_calculator())
    )


def test_legacy_pyddx_rejects_pure_total_pes_before_engine_construction(monkeypatch):
    from maple.function.calculator.extra_correction.implicit import ddpcm_smd

    monkeypatch.setattr(
        ddpcm_smd,
        "Route2ContinuumEngine",
        lambda **kwargs: pytest.fail("legacy engine must not receive pure total PES"),
    )
    with pytest.raises(ValueError, match="total-PES"):
        ddpcm_smd.PyDDXSMDImplicitSolvation(_water(), _parse("#sp").params["solv"])


def test_legacy_correction_rejects_pure_total_pes_before_provider(
    monkeypatch, tmp_path
):
    from maple.function.calculator.extra_correction.implicit import correction

    monkeypatch.setattr(
        correction,
        "PyDDXSMDImplicitSolvation",
        lambda *args, **kwargs: pytest.fail("legacy provider must not be constructed"),
    )
    with pytest.raises(ValueError, match="total-PES"):
        correction.ImplicitSolvationCorrection(
            _water(), {}, _parse("#sp").params["solv"], output=tmp_path / "bad.out"
        )
    assert not (tmp_path / "bad.out.implicit").exists()


def _pure_solvation_line(**overrides: object) -> str:
    values: dict[str, object] = {
        "implicit": "water",
        "method": "smd",
        "provider": "pyddx",
        "profile": PURE_WORKFLOW_PROFILE,
        "response": "frozen",
        "experimental": True,
    }
    values.update(overrides)
    rendered = ",".join(f"{key}={value}" for key, value in values.items())
    return f"#solv({rendered})"


def _parse(task_line: str, *, solv_line: str | None = None) -> CommandControl:
    return CommandControl.from_settings(
        [
            "#model=macepolm",
            task_line,
            "#device=cpu",
            solv_line or _pure_solvation_line(),
        ]
    )


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
        info={"charge": 0, "mult": 1},
    )


def test_pure_workflow_profile_declaratively_owns_exact_scalar_and_task_route() -> None:
    specification = route2_smd_profile_spec(PURE_WORKFLOW_PROFILE)

    assert specification.execution_route == "pure-frozen-total-pes"
    assert (
        specification.scalar_contract_id
        == EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1
    )
    assert specification.allowed_response_modes == ("frozen",)
    assert specification.experimental_task_allowlist == ("sp", "opt", "freq", "ts")
    assert specification.provider == "pyddx"
    assert specification.electrostatics_model == "ddpcm"
    assert specification.nonpolar_model == "pyscf-smd-cds"


def test_legacy_profile_does_not_inherit_pure_workflow_metadata() -> None:
    specification = route2_smd_profile_spec(DDPCM_MULTISOLVENT_SMD_PROFILE)

    assert specification.execution_route == "legacy-additive-correction"
    assert specification.scalar_contract_id is None
    assert specification.experimental_task_allowlist == ("sp",)


@pytest.mark.parametrize(
    ("task_line", "task", "method"),
    (
        ("#sp", "sp", None),
        ("#opt", "opt", None),
        ("#opt(method=lbfgs)", "opt", "lbfgs"),
        ("#opt(method=rfo)", "opt", "rfo"),
        ("#freq(method=mw,treat_imag_as_real=false)", "freq", "mw"),
        ("#ts(method=prfo)", "ts", "prfo"),
    ),
)
def test_pure_workflow_profile_accepts_only_declared_tasks(
    task_line: str,
    task: str,
    method: str | None,
) -> None:
    command = _parse(task_line)

    assert command.task == task
    assert command.params["solv"]["profile"] == PURE_WORKFLOW_PROFILE
    assert command.params["solv"]["response"] == "frozen"
    if method is not None:
        assert command.params["method"] == method


def test_pure_workflow_profile_normalizes_case_without_changing_identity() -> None:
    command = _parse(
        "#SP",
        solv_line=(
            "#SOLV(IMPLICIT=WaTeR,METHOD=SMD,PROVIDER=PyDDX,"
            f"PROFILE={PURE_WORKFLOW_PROFILE.upper()},RESPONSE=FROZEN,EXPERIMENTAL=TRUE)"
        ),
    )

    assert command.params["solv"] == {
        "implicit": "water",
        "method": "smd",
        "provider": "pyddx",
        "profile": PURE_WORKFLOW_PROFILE,
        "response": "frozen",
        "experimental": True,
        "standard_state": "1m",
    }


@pytest.mark.parametrize(
    "task_line",
    (
        "#scan(method=lbfgs)",
        "#md(ensemble=nve,steps=1)",
        "#irc(method=gs)",
        "#ts(method=neb)",
        "#ts(method=string)",
        "#ts(method=dimer)",
        "#ts(method=autoneb)",
        "#opt(method=sd)",
        "#opt(method=cg)",
        "#opt(method=sdcg)",
    ),
)
def test_pure_workflow_profile_rejects_tasks_outside_its_allowlist(
    task_line: str,
) -> None:
    with pytest.raises(ValueError, match="(?i)(task|method|workflow|allow)"):
        _parse(task_line)


@pytest.mark.parametrize(
    "task_line",
    (
        "#freq(method=nonmw,treat_imag_as_real=false)",
        "#freq(method=both,treat_imag_as_real=false)",
        "#freq(method=mw,treat_imag_as_real=true)",
    ),
)
def test_pure_workflow_frequency_rejects_non_mass_weighted_or_sign_flipped_modes(
    task_line: str,
) -> None:
    with pytest.raises(ValueError, match="(?i)(freq|mass|imag|workflow)"):
        _parse(task_line)


def test_pure_workflow_profile_rejects_scf_response() -> None:
    with pytest.raises(ValueError, match="(?i)(frozen|response)"):
        _parse("#sp", solv_line=_pure_solvation_line(response="scf"))


@pytest.mark.parametrize("device", ("cuda", "gpu0"))
def test_pure_workflow_profile_rejects_non_cpu_device(device: str) -> None:
    lines = [
        "#model=macepolm",
        "#sp",
        f"#device={device}",
        _pure_solvation_line(),
    ]
    with pytest.raises(ValueError, match="(?i)(cpu|device)"):
        CommandControl.from_settings(lines)


def test_existing_route2_profile_remains_single_point_only() -> None:
    with pytest.raises(ValueError, match="single-point only"):
        _parse(
            "#opt(method=lbfgs)",
            solv_line=_pure_solvation_line(
                profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
                response="frozen",
            ),
        )


@pytest.mark.parametrize(
    "extra_line",
    (
        "#charge(source=mol2)",
        "#d4",
        "#pbc(20,20,20)",
        "#model=macepolm(model_path=/tmp/not-the-official-checkpoint)",
        "#model=macepolm(hessian=numerical)",
        "#model=macepols",
    ),
)
def test_pure_workflow_profile_rejects_identity_or_domain_overrides(
    extra_line: str,
) -> None:
    lines = ["#model=macepolm", "#sp", "#device=cpu", _pure_solvation_line()]
    if extra_line.startswith("#model"):
        lines[0] = extra_line
    else:
        lines.append(extra_line)

    with pytest.raises(ValueError):
        CommandControl.from_settings(lines)


@pytest.mark.parametrize("override", ({"lmax": 8}, {"eta": 0.2}, {"n_proc": 2}))
def test_pure_workflow_profile_rejects_continuum_setting_overrides(
    override: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="(?i)(option|accept|lmax|eta|n_proc)"):
        _parse("#sp", solv_line=_pure_solvation_line(**override))


@pytest.mark.parametrize(
    "info",
    (
        {"charge": 1, "mult": 1},
        {"charge": 0, "mult": 3},
    ),
)
def test_pure_workflow_factory_rejects_non_neutral_or_non_singlet_before_discovery(
    tmp_path,
    info: dict[str, int],
) -> None:
    atoms = _water()
    atoms.info.update(info)
    builder = SetCalculator(
        device="cpu",
        model="macepolm",
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        solvation_options={
            "implicit": "water",
            "method": "smd",
            "provider": "pyddx",
            "profile": PURE_WORKFLOW_PROFILE,
            "response": "frozen",
            "experimental": True,
        },
    )
    builder._discover_calculator_class = lambda _name: pytest.fail(
        "domain rejection must precede model discovery"
    )

    with pytest.raises(ValueError, match="(?i)(neutral|singlet|charge|multiplicity)"):
        builder._build_calculator()


def test_pure_workflow_factory_rejects_periodicity_before_discovery(tmp_path) -> None:
    atoms = _water()
    atoms.set_cell([12.0, 12.0, 12.0])
    atoms.set_pbc(True)
    builder = SetCalculator(
        device="cpu",
        model="macepolm",
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        solvation_options={
            "implicit": "water",
            "method": "smd",
            "provider": "pyddx",
            "profile": PURE_WORKFLOW_PROFILE,
            "response": "frozen",
            "experimental": True,
        },
    )
    builder._discover_calculator_class = lambda _name: pytest.fail(
        "PBC rejection must precede model discovery"
    )

    with pytest.raises(ValueError, match="(?i)(non-periodic|pbc)"):
        builder._build_calculator()


class _FakeTotalPES:
    """Small conservative total PES with distinct leaf energies."""

    scalar_contract_id = (
        "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-v1"
    )
    provider_id = "test.pure-frozen-total-pes"

    def __init__(self) -> None:
        self.solve_calls = 0
        self.force_calls = 0
        self.hessian_calls = 0

    def solve(self, atoms: Atoms):
        self.solve_calls += 1
        positions = np.asarray(atoms.positions, dtype=float)
        total = float(1.25 + 0.5 * np.vdot(positions, positions))
        return SimpleNamespace(
            total_energy_eV=total,
            vacuum_energy_eV=1.0,
            polarization_energy_eV=0.2,
            cds_energy_eV=total - 1.2,
            state_sha256=f"{self.solve_calls:064x}",
            topology_id="a" * 64,
            topology_observation_coverage="partial",
            unobservable_topology_components=("pyscf-smd-libsolvent-internal-surface",),
        )

    def evaluate_forces(self, atoms: Atoms, *, central_state=None):
        self.force_calls += 1
        state = central_state or self.solve(atoms)
        return SimpleNamespace(
            central_state=state,
            total_forces_eV_per_A=-np.asarray(atoms.positions, dtype=float),
            evaluation_sha256=f"{self.force_calls:064x}",
        )

    def evaluate_hessian(self, atoms: Atoms, *, central_force=None):
        del central_force
        self.hessian_calls += 1
        dimension = 3 * len(atoms)
        raw = np.eye(dimension)
        return SimpleNamespace(
            raw_hessian_eV_per_A2=raw,
            hessian_eV_per_A2=raw,
            maximum_error_estimate_eV_per_A2=1.0e-4,
            maximum_antisymmetry_eV_per_A2=0.0,
            topology_guard_status="bounded-topology-noise-experimental",
            topology_observation_coverage="partial",
            displaced_topology_changed=False,
            energy_force_discrepancies_eV_per_A=(0.0,) * 4,
            evaluation_sha256=f"{self.hessian_calls:064x}",
        )

    def hessian_vector_product(self, atoms: Atoms, direction: np.ndarray):
        state = self.solve(atoms)
        forces = -np.asarray(atoms.positions, dtype=float)
        return SimpleNamespace(
            hvp_eV_per_A2=np.asarray(direction, dtype=float),
            central_sample=SimpleNamespace(
                forces_eV_per_A=forces,
                energy_sample=SimpleNamespace(energy_eV=state.total_energy_eV),
            ),
        )


def test_fake_total_pes_fixture_is_not_a_physical_validation() -> None:
    """Make the integration fixture's deliberately limited scope executable."""

    pes = _FakeTotalPES()
    state = pes.solve(_water())
    assert state.total_energy_eV == pytest.approx(
        state.vacuum_energy_eV + state.polarization_energy_eV + state.cds_energy_eV
    )
    assert pes.provider_id.startswith("test.")


def _fake_calculator(pes: _FakeTotalPES | None = None) -> PureMACEPolarDDXCalculator:
    return PureMACEPolarDDXCalculator._from_test_pes(
        atoms=_water(),
        solvent="water",
        device="cpu",
        model="macepolm",
        pes=pes or _FakeTotalPES(),
    )


def test_pure_calculator_uses_total_pes_energy_without_legacy_double_counting() -> None:
    atoms = _water()
    calculator = _fake_calculator()
    atoms.calc = calculator

    energy = atoms.get_potential_energy()
    state = calculator.last_energy_state

    assert calculator.workflow_kind == "pure-frozen-total-pes"
    assert calculator.solvent_correction is None
    assert energy == pytest.approx(state.total_energy_eV)
    assert energy != pytest.approx(state.total_energy_eV + state.vacuum_energy_eV)


def test_pure_calculator_reuses_one_total_state_for_same_geometry_energy_and_force() -> (
    None
):
    atoms = _water()
    pes = _FakeTotalPES()
    calculator = _fake_calculator(pes)
    atoms.calc = calculator

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    np.testing.assert_allclose(forces, -atoms.positions)
    assert pes.solve_calls == 1
    assert pes.force_calls == 1


def test_pure_calculator_invalidates_total_state_when_geometry_changes() -> None:
    atoms = _water()
    pes = _FakeTotalPES()
    calculator = _fake_calculator(pes)
    atoms.calc = calculator
    first = atoms.get_potential_energy()

    atoms.positions[1, 0] += 0.05
    second = atoms.get_potential_energy()

    assert second != first
    assert pes.solve_calls == 2
    assert pes.force_calls == 2


@pytest.mark.parametrize(
    ("metadata_key", "invalid_value", "message"),
    (("charge", 1, "neutral|charge"), ("mult", 3, "singlet|multiplicity")),
)
def test_pure_calculator_rechecks_quantum_metadata_when_ase_geometry_is_cached(
    metadata_key: str,
    invalid_value: int,
    message: str,
) -> None:
    atoms = _water()
    calculator = _fake_calculator()
    atoms.calc = calculator
    atoms.get_potential_energy()

    atoms.info[metadata_key] = invalid_value

    with pytest.raises(ValueError, match=f"(?i)({message})"):
        atoms.get_potential_energy()


def test_pure_calculator_hessian_is_cached_raw_eV_and_returned_by_copy() -> None:
    atoms = _water()
    pes = _FakeTotalPES()
    calculator = _fake_calculator(pes)

    first = calculator.get_hessian(atoms)
    first[0, 0] = 99.0
    second = calculator.get_hessian(atoms)

    assert pes.hessian_calls == 1
    assert calculator.last_hessian_evaluation.raw_hessian_eV_per_A2[0, 0] == 1.0
    assert second[0, 0] == 1.0
    assert calculator.last_hessian_evaluation.topology_guard_status == (
        "bounded-topology-noise-experimental"
    )


def test_pure_calculator_rejects_a_hessian_step_that_would_change_policy() -> None:
    calculator = _fake_calculator()

    with pytest.raises(ValueError, match="(?i)(0.004|step)"):
        calculator.get_hessian(_water(), delta=0.002)


def test_pure_calculator_rejects_constrained_full_hessian() -> None:
    from ase.constraints import FixAtoms

    atoms = _water()
    atoms.set_constraint(FixAtoms(indices=[0]))

    with pytest.raises(ValueError, match="(?i)constrain"):
        _fake_calculator().get_hessian(atoms)


def test_pure_calculator_hvp_tuple_uses_public_eV_units() -> None:
    atoms = _water()
    calculator = _fake_calculator()
    direction = np.linspace(-0.4, 0.4, 9).reshape(3, 3)

    hvp, forces, energy = calculator.get_hvp(atoms, direction)

    np.testing.assert_allclose(np.asarray(hvp), direction.reshape(-1))
    np.testing.assert_allclose(np.asarray(forces), -atoms.positions.reshape(-1))
    expected = 1.25 + 0.5 * float(np.vdot(atoms.positions, atoms.positions))
    assert float(energy) == pytest.approx(expected)


def test_legacy_job_view_converts_pure_hessian_and_hvp_exactly_once() -> None:
    atoms = _water()
    calculator = _fake_calculator()
    view = LegacyHartreeJobView(calculator)
    direction = np.ones((3, 3))

    np.testing.assert_allclose(view.get_hessian(atoms), np.eye(9) * EV2HARTREE)
    hvp, forces, energy = view.get_hvp(atoms, direction)
    np.testing.assert_allclose(np.asarray(hvp), np.ones(9) * EV2HARTREE)
    np.testing.assert_allclose(
        np.asarray(forces), -atoms.positions.reshape(-1) * EV2HARTREE
    )
    expected = 1.25 + 0.5 * float(np.vdot(atoms.positions, atoms.positions))
    assert float(energy) == pytest.approx(expected * EV2HARTREE)
    np.testing.assert_allclose(calculator.get_hessian(atoms), np.eye(9))


def test_dispatcher_single_point_uses_pure_total_pes_through_legacy_boundary(
    tmp_path,
) -> None:
    atoms = _water()
    calculator = _fake_calculator()
    atoms.calc = calculator
    output = tmp_path / "pure-sp.out"
    command = SimpleNamespace(params={"level": "medium", "sp": {"verbose": 1}})

    Dispatcher()(command, "sp", atoms, str(output))

    assert atoms.calc is calculator
    state = calculator.last_energy_state
    assert calculator.results["energy"] == pytest.approx(state.total_energy_eV)
    assert f"Energy: {state.total_energy_eV * EV2HARTREE:.10f} Hartree" in (
        output.read_text(encoding="utf-8")
    )


class _FailingHessianPES(_FakeTotalPES):
    def evaluate_hessian(self, atoms: Atoms, *, central_force=None):
        del central_force
        self.hessian_calls += 1
        raise RuntimeError("synthetic displaced-stencil failure")


def test_failed_hessian_does_not_replace_center_results_or_cache() -> None:
    atoms = _water()
    calculator = _fake_calculator(_FailingHessianPES())
    atoms.calc = calculator
    center_energy = atoms.get_potential_energy()
    center_forces = atoms.get_forces().copy()

    with pytest.raises(RuntimeError, match="displaced-stencil failure"):
        calculator.get_hessian(atoms)

    assert calculator.results["energy"] == center_energy
    np.testing.assert_array_equal(calculator.results["forces"], center_forces)
    assert calculator.last_hessian_evaluation is None
    assert calculator.atoms == atoms


def test_set_calculator_routes_pure_profile_before_legacy_model_discovery(
    monkeypatch,
    tmp_path,
) -> None:
    import maple.function.calculator.route2 as route2_module

    sentinel = object()
    captured: dict[str, object] = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(route2_module, "PureMACEPolarDDXCalculator", fake_factory)
    builder = SetCalculator(
        device="cpu",
        model="macepolm",
        output=str(tmp_path / "job.out"),
        atoms=_water(),
        implicit="smd",
        solvent="water",
        solvation_options={
            "implicit": "water",
            "method": "smd",
            "provider": "pyddx",
            "profile": PURE_WORKFLOW_PROFILE,
            "response": "frozen",
            "experimental": True,
        },
    )
    builder._discover_calculator_class = lambda _name: pytest.fail(
        "pure total-PES routing must precede legacy model discovery"
    )

    assert builder._build_calculator() is sentinel
    assert captured["atoms"] is builder.atoms
    assert captured["solvent"] == "water"
    assert captured["device"] == "cpu"
    assert captured["model"] == "macepolm"
    assert captured["profile_spec"].execution_route == "pure-frozen-total-pes"


@pytest.mark.parametrize(
    "override",
    ({"device": "cuda"}, {"model": "macepols"}),
)
def test_direct_pure_calculator_construction_rejects_model_identity_overrides(
    override: dict[str, str],
) -> None:
    kwargs: dict[str, object] = {
        "atoms": _water(),
        "solvent": "water",
        "device": "cpu",
        "model": "macepolm",
        "pes": _FakeTotalPES(),
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match="(?i)(cpu|macepolm)"):
        PureMACEPolarDDXCalculator._from_test_pes(**kwargs)


def test_direct_pure_calculator_rejects_a_legacy_profile_spec() -> None:
    with pytest.raises(ValueError, match="registered pure frozen workflow"):
        PureMACEPolarDDXCalculator._from_test_pes(
            atoms=_water(),
            solvent="water",
            profile_spec=route2_smd_profile_spec(DDPCM_MULTISOLVENT_SMD_PROFILE),
            pes=_FakeTotalPES(),
        )


def test_direct_pure_calculator_rejects_a_pes_with_another_scalar() -> None:
    pes = _FakeTotalPES()
    pes.scalar_contract_id = "another-scalar"

    with pytest.raises(ValueError, match="scalar contract"):
        _fake_calculator(pes)


def test_production_constructor_does_not_expose_total_pes_injection() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        PureMACEPolarDDXCalculator(
            atoms=_water(),
            solvent="water",
            pes=_FakeTotalPES(),
        )


def test_private_total_pes_injection_requires_internal_factory_token() -> None:
    with pytest.raises(ValueError, match="not a public construction path"):
        PureMACEPolarDDXCalculator(
            atoms=_water(),
            solvent="water",
            _testing_pes=_FakeTotalPES(),
        )


def test_pure_calculator_rejects_symbol_changes_before_evaluation() -> None:
    calculator = _fake_calculator()
    methane = Atoms(
        "CH4",
        positions=np.zeros((5, 3)),
        info={"charge": 0, "mult": 1},
    )

    with pytest.raises(ValueError, match="configured symbols"):
        calculator.get_potential_energy(methane)


def test_production_pure_calculator_freezes_registered_physical_settings(
    monkeypatch,
) -> None:
    import maple.function.calculator.route2._mace_polar_frozen_ddx_calculator as module

    captured: dict[str, object] = {}
    model = object()
    pes = _FakeTotalPES()

    def fake_model_builder(**kwargs):
        captured["model_builder"] = kwargs
        return model

    def fake_pes_builder(model_arg, symbols, **kwargs):
        captured["model"] = model_arg
        captured["symbols"] = symbols
        captured["pes_builder"] = kwargs
        return pes

    monkeypatch.setattr(
        module,
        "build_official_mace_polar_1_m_radial_gto_adapter",
        fake_model_builder,
    )
    monkeypatch.setattr(
        module,
        "build_smd_mace_polar_frozen_point_ddx_pes",
        fake_pes_builder,
    )

    calculator = PureMACEPolarDDXCalculator(atoms=_water(), solvent=" WATER ")

    assert calculator.pes is pes
    assert captured["model_builder"] == {"device": "cpu"}
    assert captured["model"] is model
    assert captured["symbols"] == ("O", "H", "H")
    settings = captured["pes_builder"]
    assert settings["solvent"] == "water"
    assert settings["lmax"] == 15
    assert settings["n_lebedev"] == 1202
    assert settings["solver_tolerance"] == 1.0e-12
    assert settings["eta"] == 0.1
    assert settings["n_proc"] == 1
    policy = settings["hessian_backend"]
    assert policy.coarse_step_angstrom == 4.0e-3
    assert policy.maximum_error_eV_per_A2 == 5.0e-2
    assert policy.maximum_antisymmetry_eV_per_A2 == 5.0e-2
    assert policy.maximum_energy_force_discrepancy_eV_per_A == 3.0e-3
    assert policy.topology_guard_policy == "bounded-topology-noise-experimental-v1"
