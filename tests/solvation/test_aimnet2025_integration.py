"""Narrow 2025 syntax/device admission; legacy identities remain distinct."""
from pathlib import Path

import pytest
from ase import Atoms

from maple.function.read.command_control import CommandControl
from maple.function.calculator import set_calculator as factory

ALIASES=('aimnet2025','aimnet2-2025','aimnet2_2025')


def parse(model, *, provider='torch', task='#freq(method=mw)', device='cpu'):
    return CommandControl.from_settings([
        f'#model={model}(hessian=analytic)',task,f'#device={device}',
        '#charge(source=mol2)',
        f'#solv(implicit=water,method=gb,provider={provider},model=obc2,nonpolar=ace,'
        +('platform=Reference,' if provider=='openmm' else '')+'experimental=true)',
    ]).as_dict()


@pytest.mark.parametrize('model',ALIASES)
@pytest.mark.parametrize('device',['cpu','gpu0'])
def test_2025_analytic_parser_accepts_only_dedicated_torch_profile(model,device):
    params=parse(model,device=device)
    assert params['model']==model
    assert params['model_options']['hessian']=='analytic'
    assert params['solv']['provider']=='torch'


@pytest.mark.parametrize('model',ALIASES)
def test_2025_direct_hvp_parser_has_explicit_analytic_torch_path(model):
    params=parse(model,task='#ts(method=dimer,use_hvp=true)')
    assert params['use_hvp'] is True


@pytest.mark.parametrize('model',ALIASES)
def test_2025_not_inherited_into_legacy_openmm_analytic_cells(model):
    with pytest.raises(ValueError,match='analytic|qualified|Torch'):
        parse(model,provider='openmm')


@pytest.mark.parametrize('model',['aimnet2','aimnet2nse','maceoff23m','uma'])
def test_other_backends_do_not_inherit_2025_analytic_admission(model):
    with pytest.raises(ValueError,match='analytic|qualified'):
        parse(model)


def test_2025_aliases_resolve_to_new_module_not_legacy_weights():
    for alias in ALIASES:
        assert factory._BUILTIN_NAME_TO_MODULE[alias]=='maple.function.calculator.aimnet._aimnet2025_calculator'
    assert factory._BUILTIN_NAME_TO_MODULE['aimnet2']=='maple.function.calculator.aimnet._aimnet2_calculator'
    assert factory._BUILTIN_NAME_TO_MODULE['aimnet2nse']=='maple.function.calculator.aimnet._aimnet2_calculator'


def test_2025_extra_is_pinned_optional_and_keeps_openmm_out():
    text=(Path(__file__).resolve().parents[2]/'pyproject.toml').read_text()
    assert 'aimnet2025 = [' in text
    block=text.split('aimnet2025 = [',1)[1].split(']',1)[0]
    for name,version in [('aimnet','0.2.0'),('torch','2.12.0'),('nvalchemi-toolkit-ops','0.3.1'),('warp-lang','1.13.0')]:
        assert f'{name}=={version}' in block
    assert 'openmm' not in block.lower()
    assert "python_version >= '3.11'" in block
    assert 'requires-python = ">=3.10"' in text


@pytest.mark.parametrize('alias',ALIASES)
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_factory_metadata_allows_2025_devices_without_loading_model(alias,device,tmp_path):
    configured=factory.SetCalculator(
        device,alias,str(tmp_path/'job.out'),atoms=Atoms('H2',positions=[[0,0,0],[.74,0,0]]),
        implicit='gb',solvent='water',model_options={'hessian':'analytic'},
        charge_options={'source':'mol2'},
        solvation_options={'method':'gb','implicit':'water','provider':'torch','experimental':True},
    )
    cls=configured._discover_calculator_class(alias)
    configured._validate_against_class(cls)
    assert cls.CHECKPOINT_FILENAME is None
    assert cls.__name__=='AIMNet2025Calculator'
    assert not list(tmp_path.iterdir())


def test_ani_cuda_does_not_inherit_new_2025_device_metadata(tmp_path):
    configured=factory.SetCalculator(
        'cuda:0','ani2x',str(tmp_path/'job.out'),
        atoms=Atoms('H2',positions=[[0,0,0],[.74,0,0]]),
        implicit='gb',solvent='water',model_options={'hessian':'analytic'},
        charge_options={'source':'mol2'},
        solvation_options={'method':'gb','implicit':'water','provider':'torch','experimental':True},
    )
    cls=configured._discover_calculator_class('ani2x')
    with pytest.raises(ValueError,match='CPU and D4 disabled'):
        configured._validate_against_class(cls)


def test_solv_precision_is_explicitly_parseable():
    params = CommandControl.from_settings([
        '#model=aimnet2025(hessian=analytic)', '#sp', '#device=cpu',
        '#charge(source=mol2)',
        '#solv(implicit=water,method=gb,dtype=float32,experimental=true)',
    ]).as_dict()
    assert params['solv']['dtype'] == 'float32'
