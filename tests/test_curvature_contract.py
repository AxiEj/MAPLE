from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.curvature import (
    _AIMNET2025_N32_PROFILE,
    HessianPrecisionProfile,
    build_mass_weighted_mode_basis,
    resolve_hessian_precision_profile,
    validate_cartesian_hessian,
)


def _bound_n32_calculator(monkeypatch, *, composed: bool):
    from maple.function.calculator import set_calculator
    from maple.function.calculator.aimnet._aimnet2025_calculator import (
        AIMNet2025Calculator,
    )

    receipt_path = (
        Path(set_calculator.__file__).with_name("data")
        / "independent_obc2_analytic_qualification_v6.json"
    )
    receipt = json.loads(receipt_path.read_text())
    cell = next(
        row
        for row in receipt["qualified_cells"]
        if row["gas"]["model"] == "aimnet2025"
        and row["solvent"]["nonpolar"] == "ace"
    )
    identity = cell["gas"]["runtime_profile"]
    calls = {"identity": 0}

    def live_identity(self):
        calls["identity"] += 1
        return identity

    monkeypatch.setattr(
        AIMNet2025Calculator,
        "analytic_cell_identity",
        property(live_identity),
    )
    monkeypatch.setattr(
        AIMNet2025Calculator,
        "runtime_profile_identity",
        property(live_identity),
    )
    monkeypatch.setattr(
        set_calculator,
        "_installed_runtime_source_inventory",
        lambda: receipt["installed_runtime_sources"],
    )
    calculator = AIMNet2025Calculator.__new__(AIMNet2025Calculator)
    calculator.dtype = "torch.float32"
    calculator.d4 = False
    calculator.hessian = "analytic"
    if composed:
        provider = SimpleNamespace(
            provenance=dict(
                cell["solvent"],
                equation_profile=cell["solvent"]["parameter_profile"],
            )
        )
        calculator.solvent_correction = SimpleNamespace(
            underlying_provider=provider,
            analytic_task_derivatives_admitted=True,
        )
        calculator.analytic_implicit_derivatives_admitted = True
    else:
        calculator.solvent_correction = None
    set_calculator._bind_hessian_precision_cell(calculator)
    return calculator, identity, calls


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.927, 0.0]],
    )


def _profile(relative_skew_tolerance: float) -> HessianPrecisionProfile:
    if relative_skew_tolerance == 2.0e-6:
        return _AIMNET2025_N32_PROFILE
    return HessianPrecisionProfile(
        identity="test-replay",
        relative_skew_tolerance=relative_skew_tolerance,
        source_bound=False,
    )


def test_neutral_gas_n32(monkeypatch):
    atoms = _water()
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)

    profile = resolve_hessian_precision_profile(atoms, purpose="prfo-search")

    assert profile.relative_skew_tolerance == 2.0e-6
    assert profile.source_bound is True


def test_composed_obc_n32(monkeypatch):
    atoms = _water()
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=True)

    profile = resolve_hessian_precision_profile(atoms, purpose="prfo-search")

    assert profile.relative_skew_tolerance == 2.0e-6
    assert profile.source_bound is True


def test_forged_or_drifted_n32_refuses(monkeypatch):
    atoms = _water()
    atoms.calc = SimpleNamespace(
        analytic_cell_identity={
            "model": "aimnet2025",
            "effective_ad_class": "N32",
            "effective_nn_dtype": "float32",
        }
    )
    raw = np.eye(9)
    raw[0, 1] = 5.0e-7

    profile = resolve_hessian_precision_profile(atoms, purpose="prfo-search")

    with pytest.raises(ValueError, match="asymmetry"):
        validate_cartesian_hessian(atoms, raw, profile)

    calculator, identity, _ = _bound_n32_calculator(monkeypatch, composed=False)
    drifted = dict(identity, state_sha256="0" * 64)
    calculator_type = type(calculator)
    monkeypatch.setattr(
        calculator_type,
        "analytic_cell_identity",
        property(lambda self: drifted),
    )
    monkeypatch.setattr(
        calculator_type,
        "runtime_profile_identity",
        property(lambda self: drifted),
    )
    atoms.calc = calculator
    drifted_profile = resolve_hessian_precision_profile(
        atoms,
        purpose="prfo-search",
    )
    assert drifted_profile.identity == "strict-generic"
    with pytest.raises(ValueError, match="asymmetry"):
        validate_cartesian_hessian(atoms, raw, drifted_profile)
    forged_profile = HessianPrecisionProfile("forged-n32", 2.0e-6, True)
    with pytest.raises(ValueError, match="factory-bound"):
        validate_cartesian_hessian(atoms, raw, forged_profile)


def test_bound_guard_is_reused(monkeypatch):
    atoms = _water()
    atoms.calc, _, calls = _bound_n32_calculator(monkeypatch, composed=False)
    calls_before = calls["identity"]
    profile = resolve_hessian_precision_profile(atoms, purpose="frequency")
    assert profile.identity == "qualified-aimnet2025-n32"
    assert calls["identity"] == calls_before + 2


@pytest.mark.parametrize("skew", [2.7851243245002877e-7, 4.159060036368345e-7])
def test_saved_n32_policy_accepts_raw_diagnostics_before_symmetrizing(skew):
    atoms = _water()
    raw = np.eye(9)
    raw[0, 1] += skew
    before = raw.tobytes()

    validated = validate_cartesian_hessian(atoms, raw, _profile(2.0e-6))

    assert raw.tobytes() == before
    assert validated.raw_maximum_skew == pytest.approx(skew)
    assert validated.raw_sha256
    assert validated.physical_hessian.flags.writeable is False
    np.testing.assert_array_equal(
        validated.physical_hessian,
        0.5 * (raw + raw.T),
    )


def test_saved_ace_and_none_n32_replays_are_index_one():
    checkpoint = next(
        (
            parent
            / ".omx/checkpoints/route1-aimnet-paths-20261031/"
            "stationary-fixture/RESULT.json"
            for parent in Path(__file__).resolve().parents
            if (parent / ".omx/checkpoints").is_dir()
        ),
        None,
    )
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("saved qualification replay is not present in this checkout")
    saved = json.loads(checkpoint.read_text())
    observed = {}
    for cell in saved["cells"]:
        atoms = Atoms(
            cell["source_identity"]["symbols"],
            positions=cell["frozen_written_geometry"]["positions_angstrom"],
        )
        raw = np.asarray(
            cell["frozen_written_geometry"]["hessian_hartree_per_angstrom2"],
            dtype=np.float64,
        )
        validated = validate_cartesian_hessian(atoms, raw, _profile(2.0e-6))
        modes = build_mass_weighted_mode_basis(
            atoms,
            validated,
            project_rigid_modes=True,
        )
        observed[cell["nonpolar"]] = np.count_nonzero(
            modes.frequencies_cm1 < -10.0
        )
    assert observed == {"ace": 1, "none": 1}


@pytest.mark.parametrize(
    "raw,message",
    [
        (np.eye(8), "shape"),
        (np.full((9, 9), np.nan), "finite"),
    ],
)
def test_invalid_shape_and_nonfinite_hessians_refuse(raw, message):
    with pytest.raises(ValueError, match=message):
        validate_cartesian_hessian(_water(), raw, _profile(2.0e-6))


def test_material_skew_refuses():
    raw = np.eye(9)
    raw[0, 1] = 1.0e-3
    with pytest.raises(ValueError, match="asymmetry"):
        validate_cartesian_hessian(_water(), raw, _profile(2.0e-6))


def test_bad_masses_refuse_before_mode_analysis():
    atoms = _water()
    masses = atoms.get_masses()
    masses[0] = 0.0
    atoms.set_masses(masses)
    validated = validate_cartesian_hessian(atoms, np.eye(9), _profile(1.0e-8))
    with pytest.raises(ValueError, match="positive atomic masses"):
        build_mass_weighted_mode_basis(
            atoms,
            validated,
            project_rigid_modes=False,
        )


@pytest.mark.parametrize("negative_count", [0, 1, 2])
def test_shared_mode_analysis_counts_material_negative_modes(negative_count):
    atoms = _water()
    diagonal = np.ones(9)
    diagonal[:negative_count] = -1.0
    validated = validate_cartesian_hessian(
        atoms,
        np.diag(diagonal),
        _profile(1.0e-8),
    )

    modes = build_mass_weighted_mode_basis(
        atoms,
        validated,
        project_rigid_modes=False,
    )

    assert np.count_nonzero(modes.frequencies_cm1 < -10.0) == negative_count
