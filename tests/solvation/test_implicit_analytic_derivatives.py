from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.calculator_base import (
    CalcABC,
    numerical_hessian_from_atoms,
)
from maple.function.dispatcher.frequency.frequency import _validate_frequency_boundary
from maple.function.dispatcher.ts.ts import _validate_implicit_ts_boundary
from maple.function.engine import engine
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.timer import timer


def _parse(*lines):
    return CommandControl.from_settings(list(lines)).as_dict()


class _AnalyticGasCalculator(CalcABC):
    MODEL_ENERGY_UNIT = "hartree"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_IMPLICIT_SOLVATION = True

    def __init__(self, gas_hessian):
        super().__init__()
        self.hessian = "analytic"
        self.analytic_implicit_derivatives_admitted = True
        self._gas_hessian = np.asarray(gas_hessian, dtype=np.float64)

    def _analytic_hessian(self, atoms):
        return self._gas_hessian.copy()


class _AnalyticCorrection:
    mode = "fixed"
    inner_mode = None
    supported_properties = frozenset({"energy", "forces"})

    def __init__(self, solvent_hessian):
        self._solvent_hessian = np.asarray(solvent_hessian, dtype=np.float64)
        self.hessian_calls = 0
        self.analytic_task_derivatives_admitted = True
        self.last_derivative_provenance = {"provider": "torch-obc2"}

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        return self._solvent_hessian.copy()

    def get_directional_derivatives(self, atoms, vector):
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        return SimpleNamespace(
            hvp_hartree_per_angstrom2=self._solvent_hessian @ vector,
            forces_hartree_per_angstrom=np.zeros((len(atoms), 3)),
            energy_hartree=0.0,
            provenance={"provider": "torch-obc2"},
        )


def test_calcabc_analytic_hessian_adds_gas_and_solvent_once():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    gas = np.diag(np.arange(1.0, 7.0))
    solvent = np.full((6, 6), 0.125)
    correction = _AnalyticCorrection(solvent)
    calculator = _AnalyticGasCalculator(gas)
    calculator.solvent_correction = correction

    observed = calculator.get_hessian(atoms)

    np.testing.assert_allclose(observed, gas + solvent, atol=0.0, rtol=0.0)
    assert correction.hessian_calls == 1
    assert calculator.last_analytic_hessian_provenance["composition"] == (
        "gas-analytic+torch-obc2-analytic"
    )


@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_command_contract_accepts_analytic_obc2_derivatives(nonpolar):
    frequency = _parse(
        "#model=ani2x(hessian=analytic)",
        "#freq(method=mw)",
        "#charge(source=mol2)",
        (
            "#solv(implicit=water,method=gb,provider=openmm,model=obc2,"
            f"nonpolar={nonpolar},platform=Reference,experimental=true)"
        ),
    )
    dimer = _parse(
        "#model=ani2x(hessian=analytic)",
        "#ts(method=dimer,use_hvp=true)",
        "#charge(source=mol2)",
        (
            "#solv(implicit=water,method=gb,provider=openmm,model=obc2,"
            f"nonpolar={nonpolar},platform=Reference,experimental=true)"
        ),
    )

    assert frequency["model_options"]["hessian"] == "analytic"
    assert dimer["use_hvp"] is True


@pytest.mark.parametrize(
    "model,nonpolar",
    [("obc1", "ace"), ("obc2", "lcpo"), ("gbn", "none")],
)
def test_command_contract_rejects_unsupported_analytic_gb_profiles(model, nonpolar):
    with pytest.raises(ValueError, match="analytic.*OBC-II|OBC-II.*analytic"):
        _parse(
            "#model=ani2x(hessian=analytic)",
            "#freq(method=mw)",
            "#charge(source=mol2)",
            (
                "#solv(implicit=water,method=gb,provider=openmm,"
                f"model={model},nonpolar={nonpolar},platform=Reference,"
                "experimental=true)"
            ),
        )


def test_command_contract_rejects_unqualified_analytic_gas_backend():
    with pytest.raises(ValueError, match="qualified ANI2x"):
        _parse(
            "#model=maceoff23m(hessian=analytic)",
            "#freq(method=mw)",
            "#charge(source=mol2)",
            (
                "#solv(implicit=water,method=gb,provider=openmm,model=obc2,"
                "nonpolar=ace,platform=Reference,experimental=true)"
            ),
        )


def test_runtime_frequency_boundary_accepts_admitted_analytic_correction():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = _AnalyticGasCalculator(np.eye(3))
    calculator.solvent_correction = _AnalyticCorrection(np.eye(3))
    atoms.calc = calculator

    _validate_frequency_boundary(atoms, "mw")


def test_runtime_frequency_boundary_rejects_analytic_without_solvent_hessian():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = _AnalyticGasCalculator(np.eye(3))
    calculator.solvent_correction = SimpleNamespace(
        mode="fixed",
        supported_properties={"energy", "forces"},
    )
    atoms.calc = calculator

    with pytest.raises(ValueError, match="analytic solvent Hessian"):
        _validate_frequency_boundary(atoms, "mw")


@pytest.mark.parametrize(
    "method,params",
    [
        ("prfo", {"prfo": {}}),
        ("dimer", {"dimer": {"use_hvp": True}}),
    ],
)
def test_runtime_ts_boundary_accepts_admitted_analytic_derivatives(method, params):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = _AnalyticGasCalculator(np.eye(3))
    calculator.solvent_correction = _AnalyticCorrection(np.eye(3))
    atoms.calc = calculator

    assert _validate_implicit_ts_boundary(atoms, method, params) is True


def test_correction_keeps_torch_derivatives_lazy_and_audited(water_mol2, tmp_path):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {
            "implicit": "water",
            "method": "gb",
            "provider": "openmm",
            "model": "obc2",
            "profile": "obc2-mbondi2",
            "nonpolar": "ace",
            "platform": "CPU",
            "experimental": True,
        },
        output=tmp_path / "lazy.out",
        model_device="cpu",
        task_context={"task": "freq", "method": "mw"},
    )

    assert correction.provider._derivative_backend is None
    assert correction.analytic_task_derivatives_admitted is False
    correction.evaluate(atoms, need_forces=True)
    assert correction.provider._derivative_backend is None

    hessian = correction.get_hessian(atoms)

    assert hessian.shape == (9, 9)
    assert correction.provider._derivative_backend is not None
    record = json.loads(
        (correction.audit_dir / "derivative-use.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "success"
    assert record["derivative"] == "hessian"
    assert record["runtime_provider"] == "openmm"
    assert record["runtime_platform"] == "CPU"
    assert record["derivative_provider"] == "torch-obc2"


def _analytic_obc2_lines(task_line: str, water_mol2) -> list[str]:
    return [
        "#model=ani2x(hessian=analytic)",
        task_line,
        "#device=cpu",
        "#charge(source=mol2,label=fixed-water-analytic)",
        (
            "#solv(implicit=water,method=gb,provider=openmm,model=obc2,"
            "profile=obc2-mbondi2,nonpolar=ace,platform=Reference,experimental=true)"
        ),
        "",
        "0 1",
        f"MOL2 {water_mol2}",
        "",
    ]


def test_actual_engine_analytic_frequency_closes_and_matches_reference_force_fd(
    water_mol2, tmp_path
):
    input_path = tmp_path / "analytic-frequency.inp"
    output_path = tmp_path / "analytic-frequency.out"
    input_path.write_text(
        "\n".join(_analytic_obc2_lines("#freq(method=mw,ilowfreq=2)", water_mol2)),
        encoding="utf-8",
    )

    timer.reset()
    job = engine()
    job(str(input_path), str(output_path))
    atoms = job.atoms
    calculator = atoms.calc
    correction = calculator.solvent_correction
    assert calculator.analytic_implicit_derivatives_admitted is True
    assert correction.analytic_task_derivatives_admitted is True
    assert str(calculator.dtype) == "torch.float64"
    assert calculator.inference_precision_provenance["reason"] == (
        "analytic_implicit_curvature_requires_float64"
    )

    composed = np.asarray(calculator.get_hessian(atoms), dtype=np.float64)
    solvent = np.asarray(correction.get_hessian(atoms), dtype=np.float64)
    calculator.solvent_correction = None
    try:
        gas = np.asarray(calculator.get_hessian(atoms), dtype=np.float64)
    finally:
        calculator.solvent_correction = correction
    np.testing.assert_allclose(composed, gas + solvent, rtol=0.0, atol=1.0e-12)

    direction = np.random.default_rng(20260922).normal(size=3 * len(atoms))
    direction /= np.linalg.norm(direction)
    combined_hvp, combined_force, combined_energy = calculator.get_hvp(atoms, direction)
    solvent_directional = correction.get_directional_derivatives(atoms, direction)
    calculator.solvent_correction = None
    try:
        gas_hvp, gas_force, gas_energy = calculator.get_hvp(atoms, direction)
    finally:
        calculator.solvent_correction = correction
    np.testing.assert_allclose(
        combined_hvp.detach().cpu().numpy(),
        gas_hvp.detach().cpu().numpy() + solvent_directional.hvp_hartree_per_angstrom2,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        combined_force.detach().cpu().numpy(),
        gas_force.detach().cpu().numpy()
        + solvent_directional.forces_hartree_per_angstrom.reshape(-1),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert float(combined_energy.detach().cpu()) == pytest.approx(
        float(gas_energy.detach().cpu()) + solvent_directional.energy_hartree,
        abs=1.0e-12,
    )

    errors = []
    for step in (1.0e-6, 2.0e-6, 5.0e-6):
        reference_fd = numerical_hessian_from_atoms(calculator, atoms, delta=step)
        difference = composed - reference_fd
        errors.append(
            (
                float(np.sqrt(np.mean(difference**2))),
                float(np.max(np.abs(difference))),
            )
        )
    assert all(rms <= 5.0e-6 for rms, _maximum in errors)
    assert all(maximum <= 2.0e-5 for _rms, maximum in errors)
    assert correction.provider.platform == "Reference"

    output_text = output_path.read_text(encoding="utf-8")
    assert "gas-analytic+torch-obc2-analytic" in output_text
    assert "ERROR:" not in output_text


@pytest.mark.parametrize(
    ("task_line", "output_marker", "audit_derivative"),
    [
        (
            "#ts(method=prfo,max_iter=1,project_rigid_modes=true)",
            "Starting Transition State Search (TS) with RS-PRFO",
            "hessian",
        ),
        (
            "#ts(method=dimer,use_hvp=true,max_iter=1,rot_max_iter=1,n_init=force)",
            "Derivative mode                         ....  hvp",
            "directional-hvp",
        ),
    ],
)
def test_actual_engine_analytic_ts_derivative_paths(
    water_mol2,
    tmp_path,
    task_line,
    output_marker,
    audit_derivative,
):
    input_path = tmp_path / f"analytic-{audit_derivative}.inp"
    output_path = tmp_path / f"analytic-{audit_derivative}.out"
    input_path.write_text(
        "\n".join(_analytic_obc2_lines(task_line, water_mol2)),
        encoding="utf-8",
    )

    timer.reset()
    job = engine()
    job(str(input_path), str(output_path))

    text = output_path.read_text(encoding="utf-8")
    assert output_marker in text
    assert "ERROR:" not in text
    correction = job.atoms.calc.solvent_correction
    record = json.loads(
        (correction.audit_dir / "derivative-use.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "success"
    assert record["derivative"] == audit_derivative
