"""Factory boundary regressions; synthetic cards are not production admissions."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator import calculator_base
from maple.function.calculator.calculator_base import (
    ROUTE2_SMD_CALCULATOR_PROFILE,
    CalcABC,
)
from maple.function.calculator.model_capabilities import (
    ModelCapabilities,
    ModelCardError,
    ModelProvenanceCard,
)
from maple.function.calculator.set_calculator import SetCalculator


class SyntheticBase(CalcABC):
    MODEL_NAMES = ("factory-contract-synthetic",)
    SUPPORTED_HESSIAN_MODES = ("numerical", "analytic")

    def __init__(self, device, model, implicit, solvent):
        super().__init__()
        self.incoming_selectors = (implicit, solvent)
        self.hessian = None


class SyntheticCorrection:
    supported_properties = frozenset({"energy", "forces"})
    conservative_forces = True

    def __init__(self, atoms, **kwargs):
        self.options = kwargs


def _card(**capabilities):
    defaults = {
        "energy": True,
        "forces": True,
        "conservative_forces": True,
        "hessian": "finite_difference",
    }
    defaults.update(capabilities)
    return ModelProvenanceCard(
        model_id=SyntheticBase.MODEL_NAMES[0],
        version="test-only",
        capabilities=ModelCapabilities(**defaults),
        payload={},
    )


def _forbidden(*args, **kwargs):
    pytest.fail("Forbidden model path/weight/provider work was reached")


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    setter = SetCalculator(
        "cpu",
        SyntheticBase.MODEL_NAMES[0],
        str(tmp_path / "maple.out"),
        atoms=Atoms("H2O"),
        implicit="gb",
        solvent="water",
        solvation_options={"experimental": True},
        model_options={"hessian": "numerical"},
    )
    monkeypatch.setattr(
        setter, "_discover_calculator_class", lambda name: SyntheticBase
    )
    # Generic optional lookup is intentionally absent. Numerical composition
    # must still call the required loader, binding the explicit declared name.
    monkeypatch.setattr(setter, "_registered_model_card", lambda cls: None)
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        SyntheticCorrection,
    )
    return setter


@pytest.mark.parametrize("first", ["_fennol_calculator", "_fennix_bio1_calculator"])
def test_fennix_base_and_finetune_identities_survive_either_import_order(first):
    other = (
        "_fennix_bio1_calculator"
        if first == "_fennol_calculator"
        else "_fennol_calculator"
    )
    script = f"""
import importlib
from maple.function.calculator.set_calculator import SetCalculator
for name in {first, other!r}:
    importlib.import_module("maple.function.calculator.fennol." + name)
from maple.function.calculator.fennol._fennix_bio1_calculator import FeNNixBio1Calculator, FENNIX_BIO1_CHECKPOINTS
from maple.function.calculator.fennol._fennol_calculator import FeNNolCalculator
for name in ("fennix-bio1", "fennix-bio1-small", "fennix-bio1-medium", "FeNNix_Bio1_Medium"):
    s = SetCalculator("cpu", name, "/dev/null")
    assert s._discover_calculator_class(name) is FeNNixBio1Calculator
assert FENNIX_BIO1_CHECKPOINTS["small"]["filename"] == "fennix-bio1S.fnx"
assert FENNIX_BIO1_CHECKPOINTS["medium"]["filename"] == "fennix-bio1M.fnx"
for name in ("fennix-bio1s", "fennix-bio1m"):
    s = SetCalculator("cpu", name, "/dev/null")
    assert s._discover_calculator_class(name) is FeNNolCalculator
    assert "finetuneIons" in FeNNolCalculator.model_filename(s.model)
assert FeNNixBio1Calculator.__module__ != FeNNolCalculator.__module__
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


def test_compact_builtin_cannot_be_hijacked_by_cached_plugin(tmp_path, monkeypatch):
    module = importlib.import_module(
        "maple.function.calculator.fennol._fennix_bio1_calculator"
    )
    monkeypatch.setitem(calculator_base._REGISTRY, "fennixbio1", SyntheticBase)
    setter = SetCalculator("cpu", "fennixbio1", str(tmp_path / "maple.out"))
    assert (
        setter._discover_calculator_class(setter.model) is module.FeNNixBio1Calculator
    )
    assert setter.model == "fennix-bio1"


@pytest.mark.parametrize("builtin_first", [False, True], ids=["cold", "warm"])
def test_explicit_plugin_preserves_exact_identity_in_either_import_order(
    tmp_path, builtin_first
):
    plugin_name = "factory_explicit_fennix_plugin"
    (tmp_path / f"{plugin_name}.py").write_text(
        "from maple.function.calculator.calculator_base import CalcABC, register_calculator\n"
        "@register_calculator\n"
        "class ExplicitPlugin(CalcABC):\n"
        "    MODEL_NAMES = ('fennixbio1',)\n"
    )
    script = f"""
import importlib
import sys
from maple.function.calculator.calculator_base import get_registered_calculator
from maple.function.calculator.set_calculator import SetCalculator
sys.path.insert(0, {str(tmp_path)!r})
builtin_module = "maple.function.calculator.fennol._fennix_bio1_calculator"
assert builtin_module not in sys.modules
if {builtin_first!r}:
    importlib.import_module(builtin_module)
setter = SetCalculator("cpu", "fennixbio1", "/dev/null",
                       model_options={{"module": {plugin_name!r}}})
selected = setter._discover_calculator_class("fennixbio1")
plugin = importlib.import_module({plugin_name!r}).ExplicitPlugin
assert selected is plugin
assert setter.model == "fennixbio1"
assert get_registered_calculator("fennixbio1") is plugin
if not {builtin_first!r}:
    assert builtin_module not in sys.modules
builtin = importlib.import_module(builtin_module).FeNNixBio1Calculator
assert setter._discover_calculator_class("fennixbio1") is plugin
assert setter.model == "fennixbio1"
implicit = SetCalculator("cpu", "fennixbio1", "/dev/null")
assert implicit._discover_calculator_class("fennixbio1") is builtin
assert implicit.model == "fennix-bio1"
# Explicit module selection has no builtin-alias fallback.
unregistered = SetCalculator("cpu", "fennixbio1small", "/dev/null",
                             model_options={{"module": {plugin_name!r}}})
try:
    unregistered._discover_calculator_class("fennixbio1small")
except ValueError as exc:
    assert "Unsupported model" in str(exc)
else:
    raise AssertionError("Explicit plugin name was replaced by a builtin alias")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name,task,message",
    [
        ("mace-off24-medium", "sp", "no frozen no-loss CPU/GPU parity"),
        ("aimnet2-cpcms-v2", "md", "forbids task 'md'"),
        ("aceff-2.0", None, "explicit execution task"),
        ("anonymous", "sp", "stable MODEL_NAMES identity"),
    ],
)
def test_identity_task_and_device_gates_precede_all_model_path_work(
    name,
    task,
    message,
    tmp_path,
    monkeypatch,
):
    setter = SetCalculator(
        "cuda:0",
        name,
        str(tmp_path / "maple.out"),
        model_options={"model_path": "/forbidden/weights"},
        task=task,
    )
    if name == "anonymous":

        class Anonymous(CalcABC):
            MODEL_NAMES = ()

        monkeypatch.setattr(
            setter, "_discover_calculator_class", lambda name: Anonymous
        )
    monkeypatch.setattr(setter, "_resolve_explicit_model_path", _forbidden)
    monkeypatch.setattr(setter, "_resolve_model_path", _forbidden)
    monkeypatch.setattr("torch.jit.load", _forbidden)
    monkeypatch.setattr("torch.load", _forbidden)
    with pytest.raises(ValueError, match=message):
        setter.set_calculator()


def test_required_numerical_card_uses_declared_identity(synthetic, monkeypatch):
    calls = []

    def load(name, root):
        calls.append((name, root))
        return _card()

    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card", load
    )
    calc = synthetic.set_calculator()
    assert calls == [(SyntheticBase.MODEL_NAMES[0], synthetic._model_card_root())]
    assert calc.frequency_type == "effective_solution_pmf"
    assert calc.incoming_selectors == ("gb", "water")
    assert isinstance(calc.solvent_correction, SyntheticCorrection)


@pytest.mark.parametrize("kind", ["missing", "malformed", "forbidden", "no-hessian"])
def test_required_numerical_card_fails_closed_before_weights(
    synthetic, monkeypatch, kind
):
    def load(name, root):
        if kind in {"missing", "malformed"}:
            raise ModelCardError(kind)
        if kind == "forbidden":
            return ModelProvenanceCard(
                model_id=name,
                version="test-only",
                capabilities=_card().capabilities,
                payload={},
                forbidden_tasks=("frequency",),
            )
        return ModelProvenanceCard(
            model_id=name,
            version="test-only",
            capabilities=ModelCapabilities(
                energy=True, forces=True, conservative_forces=True
            ),
            payload={},
        )

    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card", load
    )
    monkeypatch.setattr(synthetic, "_resolve_model_path", _forbidden)
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        _forbidden,
    )
    with pytest.raises(
        ValueError, match="missing|malformed|forbids task|Frequency analysis requires"
    ):
        synthetic.set_calculator()


def test_numerical_card_loader_checks_real_missing_and_malformed_files(
    synthetic, monkeypatch, tmp_path
):
    monkeypatch.setattr(synthetic, "_model_card_root", lambda: tmp_path)
    monkeypatch.setattr(synthetic, "_resolve_model_path", _forbidden)
    with pytest.raises(ModelCardError, match="was not found"):
        synthetic.set_calculator()
    (tmp_path / f"{SyntheticBase.MODEL_NAMES[0]}.yaml").write_text("not-json")
    with pytest.raises(ModelCardError, match="JSON-compatible"):
        synthetic.set_calculator()


def test_numerical_gb_rejects_actual_nonconservative_provider(synthetic, monkeypatch):
    class Nonconservative(SyntheticCorrection):
        conservative_forces = False

    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card",
        lambda *a: _card(),
    )
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        Nonconservative,
    )
    monkeypatch.setattr(synthetic, "_resolve_model_path", _forbidden)
    with pytest.raises(ValueError, match="energy-consistent solvent force backend"):
        synthetic.set_calculator()


def test_numerical_request_does_not_override_analytic_only_card_policy(
    synthetic, monkeypatch
):
    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card",
        lambda *a: _card(hessian="analytic"),
    )
    monkeypatch.setattr(synthetic, "_resolve_model_path", _forbidden)
    with pytest.raises(ValueError, match="finite-difference Hessian support"):
        synthetic.set_calculator()


def test_numerical_card_identity_must_match_declared_model(synthetic, monkeypatch):
    wrong_card = ModelProvenanceCard(
        model_id="different-test-only-base",
        version="test-only",
        capabilities=_card().capabilities,
        payload={},
    )
    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card",
        lambda *a: wrong_card,
    )
    monkeypatch.setattr(synthetic, "_resolve_model_path", _forbidden)
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        _forbidden,
    )
    with pytest.raises(ValueError, match="model-card identity mismatch"):
        synthetic.set_calculator()


def test_native_and_additive_rejected_before_provider_and_weights(
    synthetic, monkeypatch
):
    card = _card(solvation_mode="native")
    monkeypatch.setattr(
        "maple.function.calculator.set_calculator.load_model_provenance_card",
        lambda *a: card,
    )
    monkeypatch.setattr(synthetic, "_resolve_model_path", _forbidden)
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        _forbidden,
    )
    with pytest.raises(ValueError, match="Native solution-phase"):
        synthetic.set_calculator()


def test_analytic_solvent_hessian_rejected_before_discovery(synthetic, monkeypatch):
    synthetic.model_options["hessian"] = "analytic"
    monkeypatch.setattr(synthetic, "_discover_calculator_class", _forbidden)
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        _forbidden,
    )
    with pytest.raises(ValueError, match="numerical Hessian only"):
        synthetic.set_calculator()


@pytest.mark.parametrize("method", ["gb", "pb", "smd"])
def test_modern_selectors_and_metadata_are_copied_with_explicit_overrides(
    tmp_path,
    monkeypatch,
    method,
):
    atoms = Atoms("H2O")
    atoms.info.update(
        _maple_charge_options={}
        if method == "smd"
        else {"source": "mol2", "label": "old"},
        _maple_solvation_options={
            "method": method,
            "implicit": "water",
            "experimental": False,
        },
    )
    before = json.dumps(atoms.info, sort_keys=True)

    class Compatible(SyntheticBase):
        route2_smd_profile = ROUTE2_SMD_CALCULATOR_PROFILE

        def polar_state(self, atoms):
            raise AssertionError("Synthetic declaration only; no fabricated density")

    setter = SetCalculator(
        "cpu",
        "synthetic",
        str(tmp_path / "maple.out"),
        atoms=atoms,
        implicit=method,
        solvent="water",
        solvation_options={"experimental": True},
        charge_options={} if method == "smd" else {"label": "explicit"},
    )
    monkeypatch.setattr(setter, "_discover_calculator_class", lambda name: Compatible)
    monkeypatch.setattr(setter, "_registered_model_card", lambda cls: None)
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        SyntheticCorrection,
    )
    calc = setter.set_calculator()
    options = calc.solvent_correction.options
    assert options["solvation_options"]["experimental"] is True
    assert options["solvation_options"]["method"] == method
    if method != "smd":
        assert options["charge_options"] == {"source": "mol2", "label": "explicit"}
    assert calc.incoming_selectors == (method, "water")
    assert json.dumps(atoms.info, sort_keys=True) == before


@pytest.mark.parametrize(
    "options,message",
    [
        ({"method": "pb", "experimental": True}, "Conflicting.*method"),
        ({"implicit": "toluene", "experimental": True}, "Conflicting.*implicit"),
        ({"solvent": "toluene", "experimental": True}, "Conflicting.*solvent"),
        ({"experimental": False}, "experimental=true"),
    ],
)
def test_modern_selector_conflicts_and_experimental_gate_fail_before_loading(
    synthetic,
    monkeypatch,
    options,
    message,
):
    synthetic.solvation_options = options
    monkeypatch.setattr(synthetic, "_discover_calculator_class", _forbidden)
    with pytest.raises(ValueError, match=message):
        synthetic.set_calculator()


def test_upstream_macepolar_smd_fails_before_path_or_provider(tmp_path, monkeypatch):
    setter = SetCalculator(
        "cpu",
        "macepolm",
        str(tmp_path / "maple.out"),
        atoms=Atoms("H2O"),
        implicit="smd",
        solvent="water",
        solvation_options={"experimental": True},
    )
    monkeypatch.setattr(setter, "_resolve_model_path", _forbidden)
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.ImplicitSolvationCorrection",
        _forbidden,
    )
    with pytest.raises(TypeError, match=r"polar_state\(\).*density/response API"):
        setter.set_calculator()


def test_adapter_incoming_selection_gate_is_not_bypassed(synthetic, monkeypatch):
    class NativeSelectionGate(SyntheticBase):
        def __init__(self, device, model, implicit, solvent):
            assert (implicit, solvent) == ("gb", "water")
            raise ValueError("adapter forbids additive selection")

    synthetic.model_options = {}
    monkeypatch.setattr(
        synthetic, "_discover_calculator_class", lambda name: NativeSelectionGate
    )
    with pytest.raises(ValueError, match="adapter forbids additive selection"):
        synthetic.set_calculator()


def test_public_numerical_gb_hessian_with_explicit_test_only_card(
    tmp_path,
    water_mol2,
    monkeypatch,
):
    """Real OpenMM derivatives; synthetic base proves mechanics, not chemistry."""
    from maple.function.engine import engine

    class Harmonic(SyntheticBase):
        MODEL_ENERGY_UNIT = "hartree"
        implemented_properties = ("energy", "forces", "free_energy")

        def calculate(self, atoms=None, properties=None, system_changes=None):
            atoms = super().calculate(atoms, properties)
            self._finalize_results(
                atoms,
                energy=0.5 * np.sum(atoms.positions**2),
                forces=-atoms.positions,
            )

    monkeypatch.setitem(calculator_base._REGISTRY, Harmonic.MODEL_NAMES[0], Harmonic)
    card_root = tmp_path / "test-only-cards"
    card_root.mkdir()
    (card_root / f"{Harmonic.MODEL_NAMES[0]}.json").write_text(
        json.dumps(
            {
                "model_id": Harmonic.MODEL_NAMES[0],
                "version": "synthetic-test-only",
                "capabilities": {
                    "energy": True,
                    "forces": True,
                    "conservative_forces": True,
                    "hessian": "finite_difference",
                },
            }
        )
    )
    monkeypatch.setattr(SetCalculator, "_model_card_root", lambda self: card_root)
    inp = tmp_path / "synthetic-gb.inp"
    inp.write_text(
        f"#model={Harmonic.MODEL_NAMES[0]}(hessian=numerical)\n#sp(verbose=1)\n"
        "#charge(source=mol2,label=test-only)\n"
        "#solv(implicit=water,method=gb,model=obc2,nonpolar=ace,experimental=true)\n"
        f"\n0 1\nMOL2 {water_mol2}\n"
    )
    runner = engine()
    runner(str(inp), str(tmp_path / "synthetic-gb.out"))
    calc, atoms = runner.calulator, runner.atoms
    assert calc.frequency_type == "effective_solution_pmf"
    hessian = calc.get_hessian(atoms)
    assert hessian.shape == (9, 9)
    assert np.isfinite(hessian).all()
    # CalcABC uses delta=0.002 A. Independently reconstruct solvent curvature
    # at the same displacement; the harmonic base contributes the identity.
    correction = calc.solvent_correction
    expected = np.eye(9)
    for coordinate in range(9):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions.flat[coordinate] += 0.002
        minus.positions.flat[coordinate] -= 0.002
        fp = correction.evaluate(plus, need_forces=True).forces_hartree_per_angstrom
        fm = correction.evaluate(minus, need_forces=True).forces_hartree_per_angstrom
        expected[:, coordinate] -= ((fp - fm) / 0.004).ravel()
    # The established public Hessian contract symmetrizes central differences.
    expected = 0.5 * (expected + expected.T)
    np.testing.assert_allclose(hessian, expected, atol=1e-10, rtol=0)
    assert "Gradient" in (tmp_path / "synthetic-gb.out").read_text()
