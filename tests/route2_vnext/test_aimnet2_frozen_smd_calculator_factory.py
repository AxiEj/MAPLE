from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
import pytest

from maple.solvation.api import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
)
from maple.solvation.coupling.geometry_mediated_ase import (
    GeometryMediatedScalarASECalculator,
)


def _atoms() -> Atoms:
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.927297, 0.0],
        ],
        info={"charge": 0, "mult": 1},
    )


def test_frozen_smd_factory_rejects_non_cpu_and_noncontract_checkpoint(tmp_path):
    from maple.function.calculator.extra_correction.implicit.aimnet2_frozen_smd import (
        build_aimnet2_frozen_charge_water_smd_scalar,
    )

    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"not the route2 checkpoint")
    with pytest.raises(ValueError, match="CPU-only"):
        build_aimnet2_frozen_charge_water_smd_scalar(
            _atoms(), checkpoint, device="cuda"
        )
    with pytest.raises(ValueError, match="checkpoint bytes"):
        build_aimnet2_frozen_charge_water_smd_scalar(_atoms(), checkpoint, device="cpu")


def test_frozen_smd_factory_binds_exact_one_shot_total_scalar(monkeypatch, tmp_path):
    from maple.function.calculator.extra_correction.implicit import aimnet2_frozen_smd

    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"synthetic")
    calls = {}

    monkeypatch.setattr(aimnet2_frozen_smd, "_verify_checkpoint", lambda path: None)

    class FakeSource:
        def __init__(self, *, model_path, device):
            calls["source"] = (Path(model_path), device)

    class FakeDomain:
        def validate_atoms(self, atoms):
            calls["validated"] = tuple(atoms.get_chemical_symbols())
            return 0, 1

    class FakeModel:
        def __init__(self, calculator, *, contract):
            calls["contract"] = contract
            self.domain = FakeDomain()

    continuum = object()

    def fake_continuum(symbols, *, dtype, device):
        calls["continuum"] = (tuple(symbols), dtype, device)
        return continuum

    class FakeElectrostatic:
        def __init__(self, model, built_continuum, *, scalar_id, profile_id):
            calls["electrostatic"] = (scalar_id, profile_id)
            assert built_continuum is continuum
            self.model = model
            self.continuum = built_continuum

    class FakeNonpolar:
        pass

    class FakeTotal:
        def __init__(self, electrostatic, nonpolar):
            calls["nonpolar"] = nonpolar
            self.model = electrostatic.model
            self.continuum = electrostatic.continuum

        def evaluate(self, atoms):  # pragma: no cover - factory-only test
            raise AssertionError("factory construction must not evaluate the scalar")

    monkeypatch.setattr(
        aimnet2_frozen_smd, "AIMNet2ReconstructedFloat64SourceCalculator", FakeSource
    )
    monkeypatch.setattr(
        aimnet2_frozen_smd, "AIMNet2GeometryMediatedModelAdapter", FakeModel
    )
    monkeypatch.setattr(
        aimnet2_frozen_smd,
        "build_water_aimnet2_frozen_charge_harmonic_ddpcm_candidate",
        fake_continuum,
    )
    monkeypatch.setattr(
        aimnet2_frozen_smd, "GeometryMediatedElectrostaticScalar", FakeElectrostatic
    )
    monkeypatch.setattr(
        aimnet2_frozen_smd, "PySCFSMDCDSNonpolarFunctional", FakeNonpolar
    )
    monkeypatch.setattr(aimnet2_frozen_smd, "GeometryMediatedSMDTotalScalar", FakeTotal)
    monkeypatch.setattr(
        aimnet2_frozen_smd,
        "torch",
        SimpleNamespace(float64="float64"),
    )

    scalar = aimnet2_frozen_smd.build_aimnet2_frozen_charge_water_smd_scalar(
        _atoms(), checkpoint
    )
    calculator = (
        aimnet2_frozen_smd.build_aimnet2_frozen_charge_water_smd_ase_calculator(
            _atoms(), checkpoint
        )
    )

    assert isinstance(scalar, FakeTotal)
    assert isinstance(calculator, GeometryMediatedScalarASECalculator)
    assert isinstance(calculator.scalar, FakeTotal)
    assert calls["source"] == (checkpoint.resolve(), "cpu")
    assert calls["validated"] == ("O", "H", "H")
    assert calls["continuum"] == (("O", "H", "H"), "float64", "cpu")
    assert calls["electrostatic"] == (
        CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
        CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
    )
    assert isinstance(calls["nonpolar"], FakeNonpolar)
