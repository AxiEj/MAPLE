import importlib
import math
import sys
import types

from maple.function.calculator._ase_unit_contract import EV2HARTREE


HARTREE_TO_EV = 27.211386245988
LEGACY_EV2HARTREE_MODULES = (
    "maple.function.calculator.uma._uma_calculator",
    "maple.function.calculator.aimnet._aimnet2_calculator",
    "maple.function.calculator.mace._mace_calculator",
    "maple.function.calculator.mace._mace_general_calculator",
    "maple.function.calculator.mace._macepol_calculator",
)


def test_ev2hartree_uses_codata_value():
    assert math.isclose(EV2HARTREE, 1.0 / HARTREE_TO_EV, rel_tol=0.0, abs_tol=1e-18)


def test_ev_hartree_round_trip_is_stable():
    for value_ev in (1.0, 13.37, HARTREE_TO_EV):
        value_hartree = value_ev * EV2HARTREE
        assert math.isclose(value_hartree * HARTREE_TO_EV, value_ev, rel_tol=0.0, abs_tol=1e-15)


def test_legacy_calculators_share_ev2hartree_object(monkeypatch):
    _install_uma_optional_import_stubs(monkeypatch)

    modules = [importlib.import_module(module_name) for module_name in LEGACY_EV2HARTREE_MODULES]
    assert all(module.EV2HARTREE is EV2HARTREE for module in modules)


def _install_uma_optional_import_stubs(monkeypatch):
    """Keep the constant identity test independent of optional UMA packages."""

    class DummyFAIRChemCalculator:
        pass

    class DummyAtomicData:
        @staticmethod
        def from_ase(*args, **kwargs):
            raise RuntimeError("UMA import stub is not a runtime calculator")

    def unavailable(*args, **kwargs):
        raise RuntimeError("UMA import stub is not a runtime calculator")

    packages = {
        "fairchem": types.ModuleType("fairchem"),
        "fairchem.core": types.ModuleType("fairchem.core"),
        "fairchem.core.calculate": types.ModuleType("fairchem.core.calculate"),
        "fairchem.core.units": types.ModuleType("fairchem.core.units"),
    }
    for module in packages.values():
        module.__path__ = []

    pretrained_mlip = types.SimpleNamespace(
        available_models={},
        get_predict_unit=unavailable,
    )

    config_module = types.ModuleType("fairchem.core._config")
    config_module.CACHE_DIR = "/tmp"

    ase_calculator_module = types.ModuleType("fairchem.core.calculate.ase_calculator")
    ase_calculator_module.AtomicData = DummyAtomicData
    ase_calculator_module.FAIRChemCalculator = DummyFAIRChemCalculator
    ase_calculator_module.UMATask = str

    mlip_unit_module = types.ModuleType("fairchem.core.units.mlip_unit")
    mlip_unit_module.load_predict_unit = unavailable

    huggingface_module = types.ModuleType("huggingface_hub")
    huggingface_module.hf_hub_download = unavailable

    omegaconf_module = types.ModuleType("omegaconf")
    omegaconf_module.OmegaConf = types.SimpleNamespace(load=unavailable)

    packages["fairchem.core"].pretrained_mlip = pretrained_mlip
    modules = {
        **packages,
        "fairchem.core._config": config_module,
        "fairchem.core.calculate.ase_calculator": ase_calculator_module,
        "fairchem.core.units.mlip_unit": mlip_unit_module,
        "huggingface_hub": huggingface_module,
        "omegaconf": omegaconf_module,
    }

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
