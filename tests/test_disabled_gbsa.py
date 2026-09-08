"""Disabled implicit solvation must never advertise an experimental bypass."""
import pytest

from maple.function.calculator.extra_correction.solvent.gbsa.gbsa import (
    GBSA_UNAVAILABLE,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl


@pytest.mark.parametrize('option', ['', ',experimental=true', ',experimental=false'])
def test_command_line_gbsa_is_unconditionally_disabled(option):
    with pytest.raises(NotImplementedError, match='GBSA is disabled') as exc:
        CommandControl.from_settings([
            '#model=ani2x', '#sp', f'#solv(method=gbsa,implicit=water{option})',
        ])
    assert str(exc.value) == GBSA_UNAVAILABLE


def test_factory_uses_the_same_disabled_reason(tmp_path):
    factory = SetCalculator('cpu', 'ani2x', str(tmp_path/'out'), implicit='gbsa', solvent='water',
                            solvation_options={'experimental': True})
    with pytest.raises(NotImplementedError) as exc:
        factory._validate_solvent_config()
    assert str(exc.value) == GBSA_UNAVAILABLE


def test_explicit_solvent_options_are_unchanged():
    control = CommandControl.from_settings([
        '#model=ani2x', '#sp', '#solv(explicit=water,radius=5)',
    ])
    assert control.params['solv']['explicit'] == 'water'
    assert control.params['solv']['radius'] == 5
