from maple.function.calculator.calculator_base import CalcABC, register_calculator
from maple.function.calculator.set_calculator import SetClaculator


@register_calculator
class PermissiveOptionsPlugin(CalcABC):
    MODEL_NAMES = ("permissive-options-plugin",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ()
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = None
    seen_options = None

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        cls.seen_options = dict(options)
        return {}

    def __init__(self, device, model, implicit="none", solvent="none"):
        super().__init__()
        self.device = device
        self.model_name = model


def test_option_keys_none_plugin_stays_permissive_and_d4_safe(tmp_path):
    calculator = SetClaculator(
        device="cpu",
        model="permissive-options-plugin",
        output=str(tmp_path / "maple.out"),
        d4=True,
        model_options={"custom_option": "kept"},
    ).set_calculator()

    assert isinstance(calculator, PermissiveOptionsPlugin)
    assert PermissiveOptionsPlugin.seen_options == {"custom_option": "kept"}
