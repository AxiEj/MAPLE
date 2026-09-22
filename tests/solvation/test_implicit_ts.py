from __future__ import annotations

from collections.abc import Collection
from typing import Any, cast

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from maple.function.dispatcher.ts import algorithm
from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
from maple.function.dispatcher.ts.ts import TransitionState
from maple.function.engine import engine
from maple.function.read.command_control import CommandControl
from maple.function.timer import timer
from maple.function.utility import Molecules


def _parse(*lines: str) -> dict:
    return CommandControl.from_settings(list(lines)).as_dict()


def test_numerical_prfo_prepares_precision_before_first_energy_force(tmp_path):
    class PreparedCalculator(Calculator):
        hessian = "numerical"
        ready = False

        def __init__(self):
            super().__init__()
            self.implemented_properties = ["energy", "free_energy", "forces"]

        def prepare_numerical_derivatives(self):
            self.ready = True
            return 0.0005

        def calculate(self, atoms=None, properties=None, system_changes=None):
            assert self.ready, "PRFO must prepare precision before its first E/F"
            assert atoms is not None
            super().calculate(atoms, properties, system_changes)
            positions = np.asarray(atoms.get_positions())
            energy = 0.5 * float(np.square(positions).sum())
            self.results = {"energy": energy, "free_energy": energy, "forces": -positions}

        def get_hessian(self, atoms):
            return np.eye(3 * len(atoms))

    atoms = Atoms("H2", positions=[[0.1, 0.0, 0.0], [0.8, 0.0, 0.0]])
    calculator = PreparedCalculator()
    atoms.calc = calculator
    PRFO(str(tmp_path / "prepared-prfo.out"), atoms, paras={"max_iter": 1}).run()
    assert calculator.ready


def _implicit_prfo_lines(
    *,
    model: str = "ani2x",
    hessian: str = "numerical",
    charge: str = "#charge(source=mol2,label=fixed-ts-smoke)",
    method: str = "prfo",
) -> list[str]:
    platform = ",platform=Reference" if hessian == "analytic" else ""
    return [
        f"#model={model}(hessian={hessian})",
        f"#ts(method={method},max_iter=1)",
        charge,
        (
            "#solv(implicit=water,method=gb,provider=openmm,model=obc2,"
            f"profile=obc2-mbondi2,nonpolar=ace{platform},experimental=true)"
        ),
    ]


def test_parser_accepts_experimental_fixed_charge_openmm_gb_prfo():
    params = _parse(*_implicit_prfo_lines())

    assert params["task"] == "ts"
    assert params["method"] == "prfo"
    assert params["model_options"]["hessian"] == "numerical"
    assert params["charge"]["mode"] == "fixed"
    assert params["solv"]["provider"] == "openmm"
    assert params["solv"]["experimental"] is True


def _implicit_dimer_lines(*, use_hvp: bool | None = None) -> list[str]:
    option = "" if use_hvp is None else f",use_hvp={str(use_hvp).lower()}"
    return [
        "#model=ani2x",
        f"#ts(method=dimer,max_iter=1,rot_max_iter=1,n_init=force{option})",
        "#charge(source=mol2,label=fixed-dimer-smoke)",
        (
            "#solv(implicit=water,method=gb,provider=openmm,model=obc2,"
            "nonpolar=ace,experimental=true)"
        ),
    ]


@pytest.mark.parametrize("use_hvp", [None, False])
def test_parser_accepts_implicit_dimer_without_full_hessian(use_hvp):
    params = _parse(*_implicit_dimer_lines(use_hvp=use_hvp))
    assert params["task"] == "ts"
    assert params["method"] == "dimer"
    assert params.get("model_options", {}).get("hessian") is None


def test_parser_rejects_gas_only_hvp_for_implicit_dimer():
    with pytest.raises(ValueError, match="HVP"):
        _parse(*_implicit_dimer_lines(use_hvp=True))


def test_parser_accepts_analytic_obc2_prfo():
    params = _parse(*_implicit_prfo_lines(hessian="analytic"))

    assert params["model_options"]["hessian"] == "analytic"


@pytest.mark.parametrize(
    "lines, message",
    [
        (
            [
                "#model=ani2x(hessian=numerical)",
                "#ts(method=prfo,max_iter=1)",
                "#charge(source=maple)",
                (
                    "#solv(implicit=water,method=gb,provider=ambertools,"
                    "model=chagb,nonpolar=cavity-dispersion,experimental=true)"
                ),
            ],
            "single-point energy-only",
        ),
        (
            _implicit_prfo_lines(charge="#charge(source=maple,mode=polarizable)"),
            "QEq/CQEq charge models are disabled",
        ),
        (_implicit_prfo_lines(method="neb"), "prfo or dimer"),
        (_implicit_prfo_lines() + ["#pbc(10,10,10)"], "non-periodic"),
    ],
    ids=[
        "energy-only-cha-gb",
        "polarizable-charges",
        "other-ts-method",
        "periodic-cell",
    ],
)
def test_parser_rejects_implicit_ts_outside_initial_contract(lines, message):
    with pytest.raises(ValueError, match=message):
        _parse(*lines)


def _direct_implicit_atoms(
    *,
    mode: str = "fixed",
    supported_properties: Collection[str] = frozenset({"energy", "forces"}),
    hessian: str = "numerical",
    supports_composition: bool = True,
    pbc: bool = False,
    inner_mode: str | None = None,
) -> Atoms:
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, -0.4], [0.0, 0.0, 0.4]],
        cell=[10.0, 10.0, 10.0] if pbc else None,
        pbc=pbc,
    )
    calculator = cast(Any, Calculator())
    calculator.hessian = hessian
    calculator.SUPPORTS_IMPLICIT_SOLVATION = supports_composition
    calculator.solvent_correction = type(
        "TestCorrection",
        (),
        {
            "mode": mode,
            "inner_mode": inner_mode,
            "supported_properties": set(supported_properties),
        },
    )()
    atoms.calc = calculator
    return atoms


@pytest.mark.parametrize(
    "atoms, message",
    [
        (_direct_implicit_atoms(mode="polarizable"), "mode=fixed"),
        (_direct_implicit_atoms(supported_properties={"energy"}), "force support"),
        (_direct_implicit_atoms(hessian="analytic"), "analytic solvent Hessian"),
        (_direct_implicit_atoms(inner_mode="prebuilt"), "without inner=prebuilt"),
        (
            _direct_implicit_atoms(supports_composition=False),
            "SUPPORTS_IMPLICIT_SOLVATION=True",
        ),
        (_direct_implicit_atoms(pbc=True), "non-periodic"),
    ],
    ids=[
        "polarizable-charges",
        "energy-only-correction",
        "analytic-hessian-without-solvent-derivative",
        "prebuilt-inner-shell",
        "calculator-without-composition-capability",
        "periodic-cell",
    ],
)
def test_direct_api_rejects_incomplete_implicit_prfo_contract(
    tmp_path,
    atoms,
    message,
):
    with pytest.raises(ValueError, match=message):
        TransitionState(
            output=str(tmp_path / "implicit-prfo.out"),
            atoms=atoms,
            method="prfo",
            params={"method": "prfo", "max_iter": 1},
        ).run()


def test_direct_api_leaves_gas_phase_prfo_unchanged(tmp_path, monkeypatch):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    observed = {}

    class CapturingPRFO:
        def __init__(self, *, atoms, output, paras):
            observed.update(atoms=atoms, output=output, paras=paras)

        def run(self):
            observed["ran"] = True

    monkeypatch.setattr(algorithm, "PRFO", CapturingPRFO)
    output = str(tmp_path / "gas-prfo.out")
    params = {"method": "prfo", "max_iter": 1}

    TransitionState(output, atoms, params, method="prfo").run()

    assert observed == {
        "atoms": atoms,
        "output": output,
        "paras": params,
        "ran": True,
    }


def test_direct_api_rejects_other_implicit_ts_methods(tmp_path):
    atoms = _direct_implicit_atoms()

    with pytest.raises(ValueError, match="prfo or dimer"):
        TransitionState(
            str(tmp_path / "implicit-neb.out"),
            atoms,
            {"method": "neb", "max_iter": 1},
            method="neb",
        ).run()


@pytest.mark.parametrize("container", [list, Molecules])
@pytest.mark.parametrize("method", ["prfo", "dimer", "neb"])
def test_direct_api_rejects_implicit_multi_geometry_ts_paths(
    tmp_path,
    container,
    method,
):
    structures = [_direct_implicit_atoms(), _direct_implicit_atoms()]

    with pytest.raises(ValueError, match="exactly one Atoms geometry"):
        TransitionState(
            str(tmp_path / "multi-geometry-implicit-prfo.out"),
            container(structures),
            {"method": method, "max_iter": 1},
            method=method,
        ).run()


def test_implicit_prfo_dispatch_receives_the_composed_calculator(
    tmp_path,
    monkeypatch,
):
    atoms = _direct_implicit_atoms()
    observed = {}

    class CapturingPRFO:
        def __init__(self, *, atoms, output, paras):
            observed.update(atoms=atoms, calculator=atoms.calc, paras=paras)

        def run(self):
            observed["ran"] = True

    monkeypatch.setattr(algorithm, "PRFO", CapturingPRFO)

    TransitionState(
        str(tmp_path / "implicit-prfo.out"),
        atoms,
        {"method": "prfo", "max_iter": 1},
        method="prfo",
    ).run()

    assert observed["ran"] is True
    assert observed["atoms"] is atoms
    assert observed["calculator"] is atoms.calc
    calculator = cast(Any, atoms.calc)
    assert observed["calculator"].solvent_correction is calculator.solvent_correction


def test_implicit_dimer_dispatch_preserves_composed_force_calculator(
    tmp_path, monkeypatch
):
    atoms = _direct_implicit_atoms()
    cast(Any, atoms.calc).hessian = None
    observed = {}

    class CapturingDimer:
        def __init__(self, *, atoms_init, output, paras):
            observed["atoms"] = atoms_init
            observed["calculator"] = atoms_init.calc

        def run(self):
            observed["ran"] = True

    monkeypatch.setattr(algorithm, "Dimer", CapturingDimer)
    TransitionState(str(tmp_path / "dimer.out"), atoms, {}, method="dimer").run()
    assert observed["ran"] is True
    assert observed["calculator"] is atoms.calc
    assert observed["atoms"] is atoms


def test_actual_engine_runs_one_experimental_implicit_prfo_iteration(
    water_mol2,
    tmp_path,
):
    input_path = tmp_path / "water-implicit-prfo.inp"
    output_path = tmp_path / "water-implicit-prfo.out"
    input_path.write_text(
        "\n".join(
            [
                "#model=ani2x(hessian=numerical)",
                "#ts(method=prfo,max_iter=1)",
                "#device=cpu",
                "#charge(source=mol2,label=fixed-water-ts-smoke)",
                (
                    "#solv(implicit=water,method=gb,provider=openmm,"
                    "model=obc2,profile=obc2-mbondi2,nonpolar=ace,"
                    "platform=Reference,experimental=true)"
                ),
                "",
                "0 1",
                f"MOL2 {water_mol2}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    timer.reset()
    maple_engine = engine()
    maple_engine(str(input_path), str(output_path))

    atoms = maple_engine.atoms
    assert isinstance(atoms, Atoms)
    calculator = cast(Any, atoms.calc)
    energy = float(atoms.get_potential_energy(force_consistent=True))
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    hessian = np.asarray(calculator.get_hessian(atoms), dtype=np.float64)
    solvation = calculator.results["solvation"]
    correction = calculator.solvent_correction
    calculator.solvent_correction = None
    try:
        calculator.reset()
        gas_energy = float(atoms.get_potential_energy(force_consistent=True))
        gas_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        gas_hessian = np.asarray(calculator.get_hessian(atoms), dtype=np.float64)
    finally:
        calculator.solvent_correction = correction

    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.isfinite(forces).all()
    assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
    assert np.isfinite(hessian).all()
    assert energy == pytest.approx(solvation["combined_energy_hartree"])
    assert abs(energy - gas_energy) > 1.0e-8
    assert np.linalg.norm(forces - gas_forces) > 1.0e-8
    assert np.linalg.norm(hessian - gas_hessian) > 1.0e-6
    assert solvation["provenance"]["provider"] == "openmm"
    assert solvation["provenance"]["model"] == "obc2"

    output_text = output_path.read_text(encoding="utf-8")
    assert "EXPERIMENTAL IMPLICIT-SOLVENT PRFO BOUNDARY" in output_text
    assert "TS candidate-search capability" in output_text
    assert "Starting Transition State Search (TS) with RS-PRFO" in output_text
    assert "Maximum Iterations Reached" in output_text
    assert "ERROR:" not in output_text


def test_actual_engine_runs_implicit_dimer_using_only_composed_forces(
    water_mol2, tmp_path, monkeypatch
):
    import re

    from ase.io import read

    from maple.function.calculator.ani._ani_calculator import ANICalculator

    def forbidden_derivative(*_args, **_kwargs):
        raise AssertionError("implicit dimer must not call gas HVP or full Hessian")

    monkeypatch.setattr(ANICalculator, "get_hvp", forbidden_derivative)
    monkeypatch.setattr(ANICalculator, "get_hessian", forbidden_derivative)
    input_path = tmp_path / "implicit-dimer.inp"
    output_path = tmp_path / "implicit-dimer.out"
    input_path.write_text(
        "\n".join(
            [
                *_implicit_dimer_lines(),
                "#device=cpu",
                "",
                "0 1",
                f"MOL2 {water_mol2}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    timer.reset()
    job = engine()
    job(str(input_path), str(output_path))
    atoms = cast(Atoms, job.atoms)
    calculator = cast(Any, atoms.calc)
    assert np.isfinite(atoms.get_potential_energy(force_consistent=True))
    assert np.isfinite(atoms.get_forces()).all()
    assert calculator.results["solvation"]["provenance"]["provider"] == "openmm"
    text = output_path.read_text(encoding="utf-8")
    assert "EXPERIMENTAL IMPLICIT-SOLVENT DIMER BOUNDARY" in text
    assert "DIMER INITIAL STATE" in text
    assert "Dimer optimization converged at iteration" not in text
    assert "Maximum Iterations Reached" in text or "not converged" in text.lower()

    # Every serialized frame's energy must describe its own geometry, not the
    # previous translation step. Preserve the original MOL2 identity metadata.
    probe = atoms.copy()
    probe.calc = calculator
    for suffix in ("_dimer_traj.xyz", "_dimer_ts.xyz"):
        path = tmp_path / f"implicit-dimer{suffix}"
        frames = cast(list[Atoms], read(str(path), index=":"))
        comments = path.read_text().splitlines()[1 :: len(atoms) + 2]
        assert len(frames) == len(comments) > 0
        for frame, comment in zip(frames, comments, strict=True):
            match = re.search(r"Energy\s*=\s*([-+0-9.eE]+)", comment)
            assert match is not None
            probe.set_positions(frame.get_positions())
            assert float(match.group(1)) == pytest.approx(
                probe.get_potential_energy(force_consistent=True), abs=1e-7
            )
