import numpy as np
from ase.calculators.calculator import Calculator, all_changes

import maple.function.calculator.set_calculator as set_calculator_module
from maple.function.calculator.cluster_continuum import ClusterContinuumCalculator
from maple.function.calculator.set_calculator import SetCalculator


class DummyInner(Calculator):
    MODEL_NAMES = ("dummy",)
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = None
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    implemented_properties = ["energy", "forces", "free_energy"]
    received_solvation = None

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        return {}

    def __init__(self, device, model, implicit, solvent, d4=False):
        super().__init__()
        type(self).received_solvation = (implicit, solvent)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 1.0,
            "free_energy": 1.0,
            "forces": np.zeros((len(atoms), 3)),
        }


class DummyOuter:
    implemented_properties = ("energy",)

    def calculate(self, atoms, properties):
        return {"energy": 0.1}


def test_set_calculator_wraps_custom_inner_without_double_counting(tmp_path, monkeypatch):
    output = tmp_path / "maple.out"
    setup = SetCalculator(
        "cpu",
        "dummy",
        str(output),
        implicit="gbsa",
        solvent="water",
        solvation_options={"experimental": True},
    )
    monkeypatch.setattr(setup, "_discover_calculator_class", lambda name: DummyInner)
    monkeypatch.setattr(
        set_calculator_module,
        "build_outer_solvent_provider",
        lambda **kwargs: DummyOuter(),
    )

    calculator = setup.set_calculator()

    assert isinstance(calculator, ClusterContinuumCalculator)
    assert DummyInner.received_solvation == ("none", "none")
