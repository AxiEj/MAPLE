"""Explicit FP32 OBC is a new arithmetic profile; FP64 legacy stays intact."""
import numpy as np
import pytest
import torch
import hashlib
import json
from pathlib import Path
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.calculator.extra_correction.implicit.independent_obc2 import IndependentOBC2
from maple.function.calculator.extra_correction.implicit.correction import ImplicitSolvationCorrection


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_explicit_fp32_kernel_uses_real_fp32_for_all_derivatives(water_mol2,device,monkeypatch):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    atoms=MOL2Reader(str(water_mol2),charge=0,mult=1)
    p=IndependentOBC2(atoms,atoms.get_initial_charges(),device=device,dtype='float32')
    assert p.dtype==torch.float32
    assert p.kernel._charges.dtype==p.kernel._or.dtype==p.kernel._sr.dtype==torch.float32
    assert p.provenance['dtype']=='torch.float32'
    assert p.provenance['arithmetic_profile']=='obc2-fp32-v1'
    assert p.kernel._positions(atoms,requires_grad=False).dtype==torch.float32
    ef=p.evaluate(atoms,need_forces=True);h=p.hessian(atoms)
    assert np.isfinite(ef.energy_hartree) and np.isfinite(ef.forces_hartree_per_angstrom).all()
    assert np.isfinite(h).all()
    v=np.cos(np.arange(3*len(atoms)));v/=np.linalg.norm(v)
    monkeypatch.setattr(p.kernel,'hessian',lambda *a:pytest.fail('direct HVP formed dense H'))
    d=p.directional_derivatives(atoms,v)
    np.testing.assert_allclose(d.hvp_hartree_per_angstrom2,h@v,atol=1e-6,rtol=1e-5)
    assert d.provenance['dtype']=='torch.float32'


def test_direct_correction_can_explicitly_prepare_fp32_without_model_import(water_mol2,tmp_path):
    atoms=MOL2Reader(str(water_mol2),charge=0,mult=1)
    c=ImplicitSolvationCorrection(atoms,{'source':'mol2'},
        {'method':'gb','provider':'torch','dtype':'float32','experimental':True},output=tmp_path/'fp32.out')
    assert c.underlying_provider.dtype==torch.float32
    assert c.solvation_options['dtype']=='float32'


def test_default_provider_still_represents_legacy_fp64(water_mol2):
    atoms=MOL2Reader(str(water_mol2),charge=0,mult=1)
    p=IndependentOBC2(atoms,atoms.get_initial_charges())
    assert p.dtype==torch.float64
    assert p.provenance['dtype']=='torch.float64'


@pytest.mark.parametrize('dtype',['float16','bfloat16',True,'invalid'])
def test_fp32_option_does_not_open_half_or_unknown_precision(water_mol2,dtype):
    atoms=MOL2Reader(str(water_mol2),charge=0,mult=1)
    with pytest.raises(ValueError):
        IndependentOBC2(atoms,atoms.get_initial_charges(),dtype=dtype)


ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / 'tests/solvation/data/obc2_openmm_8_5_2_reference_v1.json'


def _frozen_oracle():
    raw = ORACLE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '2bb109734adf1b8a21a371cbfcc15dbc4b718f6d6cc00b1072f390733a087275'
    records = json.loads(raw)['records']
    assert len(records) == 48
    return records


@pytest.mark.parametrize('record', _frozen_oracle(), ids=lambda r: '-'.join((r['case'], r['geometry'], r['nonpolar'])))
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_fp32_matches_all_static_portable_oracle_rows(record, device):
    """Read archived values only; no live FP64 or OpenMM reference execution."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    path = ROOT / record['source_mol2_relative_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['source_mol2_sha256']
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    atoms.set_positions(record['coordinates_angstrom'])
    np.testing.assert_array_equal(atoms.get_initial_charges(), record['ordered_charges_e'])
    provider = IndependentOBC2(atoms, atoms.get_initial_charges(), nonpolar=record['nonpolar'],
                               device=device, dtype='float32')
    actual = provider.evaluate(atoms, need_forces=True)
    # Separate explicitly FP32 engineering bounds, not the historical FP64 gate.
    assert abs(actual.energy_hartree-record['total_hartree']) <= 2e-7
    delta = actual.forces_hartree_per_angstrom - record['forces_hartree_per_angstrom']
    assert np.max(np.abs(delta)) <= 2e-6
    assert np.sqrt(np.mean(delta**2)) <= 5e-7
