from __future__ import annotations

import hashlib
from types import SimpleNamespace

from ase import Atoms
from ase.constraints import FixAtoms
import pytest

from maple.function.read.command_control import CommandControl
from maple.function.read.input_reader import InputReader

PROVIDER = "aimnet2-smooth-ddpcm"
PROFILE = (
    "route2-profile-candidate-aimnet2-frozen-charge-multisolvent-"
    "smoothpartitionharmonic-ddpcm-pyscf-smdcds-v1"
)


def _model_line(path: str = "/sealed/aimnet2.pt", **options: object) -> str:
    values = {"model_path": path, **options}
    rendered = ",".join(f"{key}={value}" for key, value in values.items())
    return f"#model=aimnet2({rendered})"


def _solv_line(**overrides: object) -> str:
    values = {
        "method": "smd",
        "provider": PROVIDER,
        "profile": PROFILE,
        "implicit": "water",
        "experimental": "true",
        **overrides,
    }
    rendered = ",".join(f"{key}={value}" for key, value in values.items())
    return f"#solv({rendered})"


def _parse(
    task_line: str, *, model_line: str | None = None, solv_line: str | None = None
):
    return CommandControl.from_settings(
        [
            model_line or _model_line(),
            "#device=cpu",
            task_line,
            solv_line or _solv_line(),
        ]
    ).as_dict()


@pytest.mark.parametrize(
    ("task_line", "task", "method"),
    [
        ("#sp", "sp", None),
        ("#opt(lbfgs,max_iter=2)", "opt", "lbfgs"),
        ("#freq(method=mw)", "freq", "mw"),
        ("#ts(prfo,max_iter=2)", "ts", "prfo"),
    ],
)
def test_exact_experimental_selector_accepts_only_open_workflows(
    task_line: str, task: str, method: str | None
):
    params = _parse(task_line)

    assert params["task"] == task
    assert params.get("method") == method
    assert params["model_options"] == {
        "model_path": "/sealed/aimnet2.pt",
        "hessian": "numerical",
    }
    assert params["solv"] == {
        "method": "smd",
        "provider": PROVIDER,
        "profile": PROFILE,
        "implicit": "water",
        "experimental": True,
    }


def test_experimental_selector_uses_the_real_engine_cpu_default():
    params = CommandControl.from_settings(
        [_model_line(), "#sp", _solv_line()]
    ).as_dict()

    assert "device" not in params
    assert params["model_options"]["hessian"] == "numerical"


@pytest.mark.parametrize(
    ("task_line", "message"),
    [
        ("#md(ensemble=nve)", "support only"),
        ("#irc(method=gs)", "support only"),
        ("#scan(method=lbfgs)", "support only"),
        ("#opt(method=rfo)", "requires method=lbfgs"),
        ("#ts(method=dimer)", "requires method=prfo"),
    ],
)
def test_experimental_selector_rejects_unsupported_task_or_method(
    task_line: str, message: str
):
    with pytest.raises(ValueError, match=message):
        _parse(task_line)


@pytest.mark.parametrize(
    ("model_line", "solv_line", "message"),
    [
        (_model_line(), _solv_line(provider="pyddx"), "selector mismatch"),
        (_model_line(), _solv_line(profile="smd-iefpcm"), "selector mismatch"),
        (_model_line(), _solv_line(experimental="false"), "experimental=true"),
        ("#model=aimnet2", _solv_line(), "model_path"),
        (_model_line(hessian="analytic"), _solv_line(), "hessian=numerical"),
        (_model_line(module="custom.module"), _solv_line(), "model option"),
        (_model_line(), _solv_line(response="scf"), "solvation option"),
        (_model_line(), _solv_line(method="gbsa"), "method=smd"),
        (
            _model_line().replace("aimnet2(", "aimnet2nse("),
            _solv_line(),
            "model=aimnet2",
        ),
    ],
)
def test_experimental_selector_rejects_contract_mismatch(
    model_line: str, solv_line: str, message: str
):
    with pytest.raises(ValueError, match=message):
        _parse("#sp", model_line=model_line, solv_line=solv_line)


def test_experimental_selector_rejects_non_cpu_d4_charge_and_pbc():
    common = [_model_line(), "#sp", _solv_line()]
    cases = [
        ([*common, "#device=cuda"], "CPU-only"),
        ([*common, "#d4"], "do not compose D4"),
        ([*common, "#charge(source=mol2)"], "remove #charge"),
        ([*common, "#pbc(10,10,10,90,90,90)"], "non-periodic"),
    ]
    for lines, message in cases:
        with pytest.raises(ValueError, match=message):
            CommandControl.from_settings(lines)


def test_input_reader_requires_explicit_neutral_singlet_and_rejects_constraints(
    tmp_path,
):
    base = "\n".join(
        [_model_line("/sealed/aimnet2.pt"), "#device=cpu", "#sp", _solv_line()]
    )
    coordinates = "\nO 0 0 0\nH 0.9 0 0\nH -0.2 0.9 0\n"

    missing = tmp_path / "missing.inp"
    missing.write_text(base + coordinates)
    with pytest.raises(ValueError, match="explicit '0 1'"):
        InputReader()(str(missing), str(tmp_path / "missing.out"))

    constrained = tmp_path / "constrained.inp"
    constrained.write_text(base + "\n0 1" + coordinates + "\nC 1\n")
    with pytest.raises(ValueError, match="do not accept atom constraints"):
        InputReader()(str(constrained), str(tmp_path / "constrained.out"))


@pytest.mark.parametrize("symbols", ["HF", "SiH4"])
def test_experimental_atom_contract_rejects_elements_outside_hcno(symbols: str):
    from maple.function.aimnet2_experimental import validate_aimnet2_experimental_atoms

    atoms = Atoms(symbols)
    atoms.set_positions([[float(index), 0.0, 0.0] for index in range(len(atoms))])
    atoms.info.update(charge=0, mult=1)
    with pytest.raises(ValueError, match="only H, C, N, and O"):
        validate_aimnet2_experimental_atoms(atoms)


def test_calculator_route_validates_before_model_discovery_and_uses_factory_once(
    monkeypatch, tmp_path
):
    from maple.function.calculator.set_calculator import SetCalculator
    from maple.solvation.coupling.aimnet2_experimental_ase_v2 import (
        AIMNet2ExperimentalCalculatorV2,
    )
    import maple.function.calculator.extra_correction.implicit.aimnet2_frozen_smd as factory_module

    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"sealed-test-placeholder")
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.9, 0, 0], [-0.2, 0.9, 0]])
    atoms.info.update(charge=0, mult=1)
    calls = []
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = SimpleNamespace(
        checkpoint_contract=SimpleNamespace(
            checkpoint_size_bytes=checkpoint.stat().st_size,
            checkpoint_sha256=checkpoint_sha256,
        ),
        _checkpoint_path=checkpoint.resolve(),
    )
    scalar = SimpleNamespace(
        evaluate=lambda _atoms: None,
        fingerprint_sha256=lambda: "scalar-fingerprint",
        scalar_id="test-total-scalar",
        profile_id=PROFILE,
        model=model,
        continuum=object(),
    )

    def fake_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(scalar=scalar)

    monkeypatch.setattr(
        factory_module,
        "build_aimnet2_frozen_charge_smooth_partition_smd_ase_calculator",
        fake_factory,
    )
    builder = SetCalculator(
        device="cpu",
        model="aimnet2",
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        model_options={"model_path": str(checkpoint), "hessian": "numerical"},
        solvation_options={
            "method": "smd",
            "provider": PROVIDER,
            "profile": PROFILE,
            "implicit": "water",
            "experimental": True,
        },
    )
    builder._discover_calculator_class = lambda _name: pytest.fail(
        "experimental routing must precede ordinary model discovery"
    )

    calculator = builder._build_calculator()

    assert isinstance(calculator, AIMNet2ExperimentalCalculatorV2)
    assert len(calls) == 1
    assert calls[0][0] == (atoms, checkpoint.resolve())
    assert calls[0][1] == {"solvent": "water", "device": "cpu"}
    assert calculator.scalar is scalar
    assert (
        calculator.experimental_workflow_id
        == "aimnet2-smooth-ddpcm-experimental-workflows-v2"
    )
    assert not hasattr(calculator, "solvent_correction")


def test_calculator_invalid_domain_fails_before_runtime_import(monkeypatch, tmp_path):
    from maple.function.calculator.set_calculator import SetCalculator

    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"irrelevant")
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.9, 0, 0], [-0.2, 0.9, 0]])
    atoms.info.update(charge=1, mult=1)
    atoms.set_constraint(FixAtoms(indices=[0]))
    builder = SetCalculator(
        device="cpu",
        model="aimnet2",
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        model_options={"model_path": str(checkpoint)},
        solvation_options={
            "method": "smd",
            "provider": PROVIDER,
            "profile": PROFILE,
            "implicit": "water",
            "experimental": True,
        },
    )
    builder._discover_calculator_class = lambda _name: pytest.fail(
        "invalid experimental input must fail before model discovery"
    )

    with pytest.raises(ValueError, match="neutral singlet"):
        builder._build_calculator()


def test_exact_checkpoint_path_is_required_before_factory(monkeypatch, tmp_path):
    from maple.function.calculator.set_calculator import SetCalculator
    import maple.function.calculator.extra_correction.implicit.aimnet2_frozen_smd as factory_module

    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.9, 0, 0], [-0.2, 0.9, 0]])
    atoms.info.update(charge=0, mult=1)
    monkeypatch.setattr(
        factory_module,
        "build_aimnet2_frozen_charge_smooth_partition_smd_ase_calculator",
        lambda *_args, **_kwargs: pytest.fail(
            "missing checkpoint must fail before factory"
        ),
    )
    builder = SetCalculator(
        device="cpu",
        model="aimnet2",
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        model_options={"model_path": str(tmp_path / "missing.pt")},
        solvation_options={
            "method": "smd",
            "provider": PROVIDER,
            "profile": PROFILE,
            "implicit": "water",
            "experimental": True,
        },
    )

    with pytest.raises(FileNotFoundError, match="model_path"):
        builder._build_calculator()
