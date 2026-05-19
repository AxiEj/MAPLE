import pytest

from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
from maple.function.calculator.aimnet._aimnet2_official_pbc_calculator import (
    AIMNet2OfficialPBCCalculator,
)
from maple.function.calculator.ani._ani_calculator import ANICalculator
from maple.function.calculator.mace._mace_calculator import MACECalculator
from maple.function.calculator.mace._mace_general_calculator import MACEModelCalculator
from maple.function.calculator.mace._mace_official_pbc_calculator import (
    MACEOfficialPBCCalculator,
)
from maple.function.calculator.mace._macepol_official_pbc_calculator import (
    MACEPolOfficialPBCCalculator,
)
from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator


@pytest.mark.parametrize(
    "calculator_cls",
    [AIMNet2OfficialPBCCalculator, MACEOfficialPBCCalculator, MACEPolOfficialPBCCalculator],
)
def test_pbc_backends_declare_periodic_stress_capability(calculator_cls):
    assert calculator_cls.maple_pbc_md_supported is True
    assert calculator_cls.maple_stress_supported is True


@pytest.mark.parametrize(
    "calculator_cls",
    [
        ANICalculator,
        AIMNet2Calculator,
        MACECalculator,
        MACEModelCalculator,
        MACEPolCalculator,
    ],
)
def test_legacy_non_pbc_backends_declare_no_periodic_stress_capability(calculator_cls):
    assert calculator_cls.maple_pbc_md_supported is False
    assert calculator_cls.maple_stress_supported is False
