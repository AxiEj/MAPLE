from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ase import Atoms

from maple.function.calculator.set_calculator import SetCalculator
from maple.function.calculator.model_capabilities import load_model_provenance_card
from maple.function.calculator.uma._anisolv_uma_calculator import (
    ANISOLV_UMA_BASE_SIZE,
    AniSolvUMACalculator,
    UMA_S1P2_COMPAT_CHECKPOINT_SHA256,
)
from maple.function.calculator.uma._uma_calculator import UMACalculator
from maple.function.dispatcher.sp.sp import SinglePoint
from maple.function.read.command_control import CommandControl


def _parse(*lines):
    return CommandControl.from_settings(list(lines)).as_dict()


def _atoms():
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.0, 0.7, 0.5], [0.0, -0.7, 0.5]])
    atoms.info.update(charge=0, mult=1, mol2="sealed-identity")
    return atoms


def test_command_contract_exposes_only_the_sealed_anisolv_uma_path():
    params = _parse(
        "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt)",
        "#sp",
        "#solv(implicit=THF,method=anisolv,experimental=true)",
    )

    assert params["model"] == "anisolv-uma"
    assert params["model_options"] == {
        "base_model_path": "/models/uma-s-1p2.pt",
        "solvation_model_path": "/models/model1_compact.pt",
    }
    assert params["solv"] == {
        "implicit": "thf",
        "method": "anisolv",
        "experimental": True,
    }


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            (
                "#model=uma(solvation_model_path=/models/model1_compact.pt)",
                "#sp",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "sealed #model=anisolv-uma",
        ),
        (
            (
                "#model=anisolv-uma",
                "#sp",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "requires #model=anisolv-uma",
        ),
        (
            (
                "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt,size=uma-s-1p1)",
                "#sp",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "pinned to size=uma-s-1p2",
        ),
        (
            (
                "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt,model_path=/models/base.pt)",
                "#sp",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "accepts only base_model_path",
        ),
        (
            (
                "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt)",
                "#md",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "restricted to scalar single-point energy",
        ),
        (
            (
                "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt)",
                "#freq",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "restricted to scalar single-point energy",
        ),
        (
            (
                "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt,hessian=numerical)",
                "#sp",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "size/task/inference options",
        ),
        (
            (
                "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt)",
                "#sp(verbose=1)",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "SP gradients are disabled",
        ),
        (
            (
                "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt)",
                "#sp",
                "#solv(implicit=water,method=anisolv,provider=mock,experimental=true)",
            ),
            "does not accept continuum/provider",
        ),
        (
            (
                "#model=anisolv-uma(base_model_path=/models/uma-s-1p2.pt,solvation_model_path=/models/model1_compact.pt)",
                "#charge(source=mol2)",
                "#sp",
                "#solv(implicit=water,method=anisolv,experimental=true)",
            ),
            "does not consume a #charge",
        ),
    ],
)
def test_command_contract_rejects_unsealed_or_unsafe_configurations(lines, message):
    with pytest.raises(ValueError, match=message):
        _parse(*lines)


def test_wrapper_pins_the_uma_base_and_attaches_the_compact_backend(
    monkeypatch, tmp_path
):
    parent_calls = []
    compact_calls = []

    def fake_parent_init(self, **kwargs):
        parent_calls.append(kwargs)
        self.device = "cpu"
        self.solvent_correction = None
        self.implemented_properties = ["energy", "free_energy", "forces"]

    class FakeCompactBackend:
        def __init__(self, **kwargs):
            compact_calls.append(kwargs)

    monkeypatch.setattr(UMACalculator, "__init__", fake_parent_init)
    monkeypatch.setattr(
        "maple.function.calculator.uma._anisolv_uma_calculator.AniSolvCompactBackend",
        FakeCompactBackend,
    )
    base_checkpoint = tmp_path / "uma-s-1p2.pt"
    base_checkpoint.write_bytes(b"verified base")
    monkeypatch.setattr(
        "maple.function.calculator.uma._anisolv_uma_calculator._sha256",
        lambda _path: UMA_S1P2_COMPAT_CHECKPOINT_SHA256,
    )

    calc = AniSolvUMACalculator(
        device="cpu",
        model="anisolv-uma",
        implicit="anisolv",
        solvent="methanol",
        base_model_path=str(base_checkpoint),
        solvation_model_path="/models/model1_compact.pt",
    )

    assert parent_calls == [
        {
            "device": "cpu",
            "model": "uma",
            "overrides": None,
            "implicit": "none",
            "solvent": "none",
            "task": "omol",
            "size": ANISOLV_UMA_BASE_SIZE,
            "checkpoint_path": str(base_checkpoint),
            "inference_settings": None,
        }
    ]
    assert compact_calls == [
        {
            "model_path": "/models/model1_compact.pt",
            "solvent": "methanol",
            "device": "cpu",
            "execution_task": None,
        }
    ]
    assert isinstance(calc.solvent_correction, FakeCompactBackend)
    assert calc.implemented_properties == ["energy", "free_energy"]
    assert calc.chargecalc is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"checkpoint_path": "/models/base.pt"}, "custom checkpoint_path"),
        ({"overrides": {"model": {}}}, "overrides are disabled"),
        ({"size": "uma-s-1p1"}, "pinned to base size"),
        ({"task": "omat"}, "only permits UMA task='omol'"),
        ({"solvation_model_path": None}, "requires model option"),
        ({"base_model_path": None}, "requires model option"),
    ],
)
def test_wrapper_rejects_base_model_substitution_before_loading(kwargs, message):
    base = {
        "device": "cpu",
        "model": "anisolv-uma",
        "implicit": "anisolv",
        "solvent": "water",
        "base_model_path": "/models/uma-s-1p2.pt",
        "solvation_model_path": "/models/model1_compact.pt",
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        AniSolvUMACalculator(**base)


def test_wrapper_rejects_a_base_checkpoint_with_the_wrong_identity(tmp_path):
    base_checkpoint = tmp_path / "uma-s-1p2.pt"
    base_checkpoint.write_bytes(b"not the pinned compatibility checkpoint")

    with pytest.raises(ValueError, match="base model checksum mismatch"):
        AniSolvUMACalculator(
            device="cpu",
            model="anisolv-uma",
            implicit="anisolv",
            solvent="water",
            base_model_path=str(base_checkpoint),
            solvation_model_path="/models/model1_compact.pt",
        )


def test_set_calculator_routes_the_sealed_composition_without_a_generic_correction(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_init(self, execution_task=None, **kwargs):
        captured.update(kwargs)
        captured["execution_task"] = execution_task
        self.solvent_correction = SimpleNamespace()

    monkeypatch.setattr(AniSolvUMACalculator, "__init__", fake_init)
    setter = SetCalculator(
        device="cpu",
        model="anisolv-uma",
        output=str(tmp_path / "anisolv.out"),
        atoms=_atoms(),
        implicit="anisolv",
        solvent="methanol",
        model_options={
            "base_model_path": "/models/uma-s-1p2.pt",
            "solvation_model_path": "/models/model1_compact.pt",
        },
        solvation_options={
            "method": "anisolv",
            "implicit": "methanol",
            "experimental": True,
        },
        task="sp",
    )

    calc = setter.set_calculator()

    assert isinstance(calc, AniSolvUMACalculator)
    assert captured["model"] == "anisolv-uma"
    assert captured["implicit"] == "anisolv"
    assert captured["solvent"] == "methanol"
    assert captured["base_model_path"] == "/models/uma-s-1p2.pt"
    assert captured["solvation_model_path"] == "/models/model1_compact.pt"
    assert captured["execution_task"] == "sp"
    assert "checkpoint_path" not in captured


def test_set_calculator_rejects_non_neutral_anisolv_uma_before_loading(tmp_path):
    atoms = _atoms()
    atoms.info["charge"] = 1
    setter = SetCalculator(
        device="cpu",
        model="anisolv-uma",
        output=str(tmp_path / "anisolv.out"),
        atoms=atoms,
        implicit="anisolv",
        solvent="water",
        model_options={
            "base_model_path": "/models/uma-s-1p2.pt",
            "solvation_model_path": "/models/model1_compact.pt",
        },
        solvation_options={"method": "anisolv", "experimental": True},
    )

    with pytest.raises(ValueError, match="neutral-singlet"):
        setter.set_calculator()


@pytest.mark.parametrize(
    ("task", "model_options"),
    [
        ("opt", {}),
        ("scan", {}),
        ("freq", {"hessian": "numerical"}),
    ],
)
def test_set_calculator_rejects_force_derived_anisolv_tasks_before_loading(
    tmp_path, task, model_options
):
    options = {
        "base_model_path": "/models/uma-s-1p2.pt",
        "solvation_model_path": "/models/model1_compact.pt",
        **model_options,
    }
    setter = SetCalculator(
        device="cpu",
        model="anisolv-uma",
        output=str(tmp_path / "anisolv.out"),
        atoms=_atoms(),
        implicit="anisolv",
        solvent="water",
        model_options=options,
        solvation_options={
            "method": "anisolv",
            "implicit": "water",
            "experimental": True,
        },
        task=task,
    )

    with pytest.raises(
        ValueError,
        match="scalar single-point energy|explicitly forbids|conservative forces",
    ):
        setter.set_calculator()


def test_wrapper_records_and_renders_a_solution_pmf_not_delta_g(monkeypatch):
    class FakeCorrection:
        def evaluate(self, atoms, *, need_forces):
            assert atoms.info["charge"] == 0
            assert need_forces is False
            return SimpleNamespace(
                energy_hartree=-0.25,
                components_hartree={"anisolv_compact_delta": -0.25},
                provenance={
                    "provider": "anisolv-compact",
                    "supports_absolute_solvation": False,
                },
            )

    def fake_parent_calculate(self, atoms, properties=None, system_changes=None):
        del atoms, properties, system_changes
        self.results = {
            "energy": -10.25,
            "free_energy": -10.25,
            "forces": [[1.0, 2.0, 3.0]],
        }
        return self.results

    monkeypatch.setattr(UMACalculator, "calculate", fake_parent_calculate)
    calc = object.__new__(AniSolvUMACalculator)
    calc.solvent_correction = FakeCorrection()

    results = calc.calculate(_atoms(), properties=["energy"])
    solvation = results["solvation"]

    assert solvation["quantity"] == "geometry_level_solution_pmf"
    assert solvation["solution_pmf_correction_hartree"] == pytest.approx(-0.25)
    assert solvation["gas_energy_hartree"] == pytest.approx(-10.25)
    assert solvation["combined_energy_hartree"] == pytest.approx(-10.5)
    assert solvation["ase_free_energy_is_thermochemical_gibbs"] is False
    assert solvation["forces_exposed"] is False
    assert solvation["provenance"]["supports_absolute_solvation"] is False
    assert "forces" not in results

    rendered = "".join(
        SinglePoint._solvation_lines(
            SimpleNamespace(calc=SimpleNamespace(results=results))
        )
    )
    assert "Geometry-level solvent correction W_solv(R)" in rendered
    assert "not an absolute Delta G_solv" in rendered
    assert "Combined effective solution PMF" in rendered
    assert "Solvation free-energy correction" not in rendered


@pytest.mark.parametrize(
    ("base_energy", "base_free_energy", "correction", "component"),
    [
        (float("nan"), -10.0, -0.25, -0.25),
        (-10.0, float("inf"), -0.25, -0.25),
        (-10.0, -10.0, float("nan"), -0.25),
        (-10.0, -10.0, -0.25, float("-inf")),
        (1.0e308, 1.0e308, 1.0e308, 1.0e308),
    ],
)
def test_wrapper_rejects_nonfinite_energy_composition_transactionally(
    monkeypatch,
    base_energy,
    base_free_energy,
    correction,
    component,
):
    class FakeCorrection:
        def evaluate(self, _atoms, *, need_forces):
            assert need_forces is False
            return SimpleNamespace(
                energy_hartree=correction,
                components_hartree={"anisolv_compact_delta": component},
                provenance={},
            )

    def fake_parent_calculate(self, _atoms, properties=None, system_changes=None):
        del properties, system_changes
        self.results = {
            "energy": base_energy,
            "free_energy": base_free_energy,
            "forces": [[1.0, 2.0, 3.0]],
        }
        return self.results

    monkeypatch.setattr(UMACalculator, "calculate", fake_parent_calculate)
    calc = object.__new__(AniSolvUMACalculator)
    calc.solvent_correction = FakeCorrection()

    with pytest.raises(ValueError, match="finite"):
        calc.calculate(_atoms(), properties=["energy"])
    assert calc.results == {}


def test_wrapper_rejects_forces_and_hessian_before_base_evaluation(monkeypatch):
    def unexpected_parent(*_args, **_kwargs):
        raise AssertionError(
            "base calculator must not run for a rejected force request"
        )

    monkeypatch.setattr(UMACalculator, "calculate", unexpected_parent)
    calc = object.__new__(AniSolvUMACalculator)
    calc.solvent_correction = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="forces are disabled"):
        calc.calculate(_atoms(), properties=["forces"])
    with pytest.raises(NotImplementedError, match="Hessians are disabled"):
        calc.get_hessian(_atoms())


def test_composite_model_card_remains_energy_only():
    card = load_model_provenance_card(
        "anisolv-uma",
        model_card_root=Path(__file__).parents[2]
        / "maple/function/calculator/model_cards",
    )

    assert card.capabilities.solvation_mode == "additive"
    assert card.capabilities.forces is False
    assert card.capabilities.conservative_forces is False
    assert card.capabilities.hessian == "none"
    assert card.capabilities.supports_md is False
    assert card.capabilities.supports_absolute_solvation is False
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("absolute_solvation_free_energy")
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("md")
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("opt")
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("frequency")
