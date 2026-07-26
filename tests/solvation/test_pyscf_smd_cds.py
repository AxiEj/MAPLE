from __future__ import annotations

import types

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.pyscf_runtime import (
    TESTED_PYSCF_VERSION,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    PySCFSMDCDSResult,
    _PySCFSMDCDSRuntime,
    pyscf_smd_cds,
    pyscf_smd_water_cds,
)


class _FakeMolecule:
    natm = 2


class _FakeGTO:
    last_kwargs = None

    @classmethod
    def M(cls, **kwargs):
        cls.last_kwargs = kwargs
        return _FakeMolecule()


class _FakeSMDObject:
    def __init__(self, mol, *, solvent):
        self.mol = mol
        self.solvent = solvent


class _FakeSMD:
    SMD = _FakeSMDObject
    libsolvent = object()
    last_object = None

    @classmethod
    def get_cds_legacy(cls, smd_object):
        cls.last_object = smd_object
        return (
            0.0125,
            np.asarray(
                [
                    [0.2, -0.1, 0.05],
                    [-0.2, 0.1, -0.05],
                ]
            ),
        )


@pytest.fixture
def fake_runtime():
    _FakeGTO.last_kwargs = None
    _FakeSMD.last_object = None
    return _PySCFSMDCDSRuntime(
        version=TESTED_PYSCF_VERSION,
        gto=_FakeGTO,
        smd=_FakeSMD,
    )


def test_pyscf_smd_cds_uses_official_energy_and_gradient_in_maple_units(
    fake_runtime,
):
    positions = np.asarray(
        [
            [-0.7, 0.0, 0.1],
            [0.8, 0.2, -0.1],
        ]
    )

    result = pyscf_smd_water_cds(
        ("H", "O"),
        positions,
        _runtime=fake_runtime,
    )

    assert isinstance(result, PySCFSMDCDSResult)
    assert result.energy_hartree == pytest.approx(0.0125)
    np.testing.assert_allclose(
        result.position_gradient_hartree_per_angstrom,
        np.asarray(
            [
                [0.2, -0.1, 0.05],
                [-0.2, 0.1, -0.05],
            ]
        )
        / Bohr,
    )
    assert result.runtime_provenance == {
        "provider": "pyscf-smd-libsolvent-cds",
        "pyscf_version": TESTED_PYSCF_VERSION,
        "solvent": "water",
        "pyscf_smd_solvent": "water",
        "upstream_entrypoint": "pyscf.solvent.smd.get_cds_legacy",
    }
    assert _FakeGTO.last_kwargs == {
        "atom": [
            ("H", [-0.7, 0.0, 0.1]),
            ("O", [0.8, 0.2, -0.1]),
        ],
        "unit": "Angstrom",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
        "verbose": 0,
    }
    assert _FakeSMD.last_object.solvent == "water"
    with pytest.raises(ValueError, match="read-only"):
        result.position_gradient_hartree_per_angstrom[0, 0] = 0.0
    with pytest.raises(TypeError):
        result.runtime_provenance["provider"] = "mutated"


def test_pyscf_smd_cds_routes_canonical_name_to_upstream_name(fake_runtime):
    result = pyscf_smd_cds(
        ("H", "O"),
        np.zeros((2, 3)),
        solvent="dmf",
        _runtime=fake_runtime,
    )

    assert _FakeSMD.last_object.solvent == "N,N-dimethylformamide"
    assert result.runtime_provenance["solvent"] == "dimethylformamide"
    assert (
        result.runtime_provenance["pyscf_smd_solvent"]
        == "N,N-dimethylformamide"
    )


def test_pyscf_smd_cds_rejects_untested_runtime(fake_runtime):
    runtime = _PySCFSMDCDSRuntime(
        version="2.14.0",
        gto=fake_runtime.gto,
        smd=fake_runtime.smd,
    )

    with pytest.raises(RuntimeError, match="tested only with PySCF"):
        pyscf_smd_water_cds(
            ("H", "O"),
            np.zeros((2, 3)),
            _runtime=runtime,
        )


def test_pyscf_smd_cds_requires_compiled_libsolvent(fake_runtime):
    runtime = _PySCFSMDCDSRuntime(
        version=fake_runtime.version,
        gto=fake_runtime.gto,
        smd=types.SimpleNamespace(
            SMD=_FakeSMDObject,
            get_cds_legacy=_FakeSMD.get_cds_legacy,
            libsolvent=None,
        ),
    )

    with pytest.raises(RuntimeError, match="compiled libsolvent"):
        pyscf_smd_water_cds(
            ("H", "O"),
            np.zeros((2, 3)),
            _runtime=runtime,
        )


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (np.zeros((2, 2)), "finite with shape"),
        (np.full((2, 3), np.nan), "finite with shape"),
        (np.zeros((1, 3)), "finite with shape"),
    ],
)
def test_pyscf_smd_cds_rejects_invalid_coordinates(
    fake_runtime,
    positions,
    message,
):
    with pytest.raises(ValueError, match=message):
        pyscf_smd_water_cds(
            ("H", "O"),
            positions,
            _runtime=fake_runtime,
        )


@pytest.mark.parametrize(
    ("energy", "gradient", "message"),
    [
        (
            np.nan,
            np.zeros((2, 3)),
            "energy must be finite",
        ),
        (
            0.1,
            np.zeros((2, 2)),
            "gradient must be finite",
        ),
        (
            0.1,
            np.full((2, 3), np.nan),
            "gradient must be finite",
        ),
    ],
)
def test_pyscf_smd_cds_rejects_invalid_upstream_results(
    fake_runtime,
    monkeypatch,
    energy,
    gradient,
    message,
):
    monkeypatch.setattr(
        fake_runtime.smd,
        "get_cds_legacy",
        lambda _object: (energy, gradient),
    )

    with pytest.raises(RuntimeError, match=message):
        pyscf_smd_water_cds(
            ("H", "O"),
            np.zeros((2, 3)),
            _runtime=fake_runtime,
        )
