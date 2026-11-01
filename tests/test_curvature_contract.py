from __future__ import annotations

from dataclasses import replace
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.curvature import (
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


def test_composed_binding_refuses_nonpolar_drift(monkeypatch):
    atoms = _water()
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=True)
    atoms.calc.solvent_correction.underlying_provider.provenance["nonpolar"] = "none"

    with pytest.raises(ValueError, match="composition changed"):
        resolve_hessian_precision_profile(atoms, purpose="prfo-search")


def test_resolved_profile_refuses_later_composition_drift(monkeypatch):
    atoms = _water()
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=True)
    profile = resolve_hessian_precision_profile(atoms, purpose="prfo-search")
    atoms.calc.solvent_correction.underlying_provider.provenance["nonpolar"] = "none"

    with pytest.raises(ValueError, match="composition changed"):
        validate_cartesian_hessian(atoms, np.eye(9), profile)


def test_exact_aimnet_numerical_mode_keeps_strict_profile(monkeypatch):
    atoms = _water()
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)
    atoms.calc.hessian = "numerical"

    profile = resolve_hessian_precision_profile(atoms, purpose="frequency")

    assert profile.identity == "strict-generic"
    assert profile.source_bound is False


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
    with pytest.raises(ValueError, match="binding"):
        resolve_hessian_precision_profile(atoms, purpose="prfo-search")
    forged_profile = HessianPrecisionProfile("forged-n32", 2.0e-6, True)
    with pytest.raises(ValueError, match="factory-bound"):
        validate_cartesian_hessian(atoms, raw, forged_profile)


def test_bound_guard_is_reused(monkeypatch):
    atoms = _water()
    atoms.calc, _, calls = _bound_n32_calculator(monkeypatch, composed=False)
    calls_before = calls["identity"]
    profile = resolve_hessian_precision_profile(atoms, purpose="frequency")
    assert profile.identity == "qualified-aimnet2025-n32-gas"
    assert calls["identity"] == calls_before + 2


def test_validation_reruns_live_identity_guard_after_profile_resolution(monkeypatch):
    atoms = _water()
    calculator, identity, calls = _bound_n32_calculator(monkeypatch, composed=False)
    atoms.calc = calculator
    profile = resolve_hessian_precision_profile(atoms, purpose="frequency")
    calls_after_resolution = calls["identity"]
    drifted = dict(identity, state_sha256="0" * 64)
    monkeypatch.setattr(
        type(calculator),
        "analytic_cell_identity",
        property(lambda self: drifted),
    )
    monkeypatch.setattr(
        type(calculator),
        "runtime_profile_identity",
        property(lambda self: drifted),
    )
    raw = np.eye(9)
    raw[0, 1] = 5.0e-7

    with pytest.raises(ValueError, match="runtime binding|identity"):
        validate_cartesian_hessian(atoms, raw, profile)

    assert calls["identity"] == calls_after_resolution


def test_gas_binding_reuses_full_receipt_gate_validator(monkeypatch):
    from maple.function.calculator import set_calculator
    from maple.function.calculator import _independent_obc2_qualified_cell as pointer

    receipt_path = (
        Path(set_calculator.__file__).with_name("data")
        / "independent_obc2_analytic_qualification_v6.json"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["qualification_gates"] = {}
    receipt["content_sha256"] = hashlib.sha256(
        set_calculator._canonical_receipt_payload(receipt)
    ).hexdigest()
    payload = json.dumps(receipt).encode()
    monkeypatch.setattr(pointer, "QUALIFICATION_RECEIPT_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(Path, "read_bytes", lambda self: payload)
    monkeypatch.setattr(
        set_calculator,
        "_installed_runtime_source_inventory",
        lambda: receipt["installed_runtime_sources"],
    )

    with pytest.raises(ValueError, match="gate set"):
        set_calculator._load_validated_qualification_receipt()


def test_bound_profile_cannot_transfer_to_other_atoms(monkeypatch):
    atoms = _water()
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)
    profile = resolve_hessian_precision_profile(atoms, purpose="frequency")
    other = _water()
    other.calc = atoms.calc
    raw = np.eye(9)
    raw[0, 1] = 5.0e-7

    with pytest.raises(ValueError, match="different Atoms"):
        validate_cartesian_hessian(other, raw, profile)


def test_profile_display_fields_cannot_widen_bound_policy(monkeypatch):
    atoms = _water()
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)
    profile = resolve_hessian_precision_profile(atoms, purpose="frequency")
    raw = np.eye(9)
    raw[0, 1] = 0.25

    with pytest.raises(ValueError, match="profile|policy|binding"):
        validate_cartesian_hessian(
            atoms,
            raw,
            replace(
                profile,
                identity="forged-wide",
                relative_skew_tolerance=1.0,
            ),
        )


def test_unbound_nan_tolerance_cannot_disable_strict_policy():
    raw = np.eye(9)
    raw[0, 1] = 1.0
    profile = HessianPrecisionProfile("strict-generic", float("nan"), False)

    with pytest.raises(ValueError, match="asymmetry|profile|policy"):
        validate_cartesian_hessian(_water(), raw, profile)


def test_initial_charge_fallback_cannot_receive_neutral_n32_profile(monkeypatch):
    atoms = _water()
    atoms.set_initial_charges([1.0, 0.0, 0.0])
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)

    with pytest.raises(ValueError, match="neutral closed-shell"):
        resolve_hessian_precision_profile(atoms, purpose="frequency")


def test_bound_profile_allows_new_geometry_and_per_image_source_provenance(monkeypatch):
    atoms = _water()
    atoms.info["mol2"] = {
        "source_path": "image-0.mol2",
        "source_sha256": "a" * 64,
    }
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)
    profile = resolve_hessian_precision_profile(atoms, purpose="path-image")
    atoms.positions[0, 0] += 0.01
    atoms.info["mol2"] = {
        "source_path": "image-1.mol2",
        "source_sha256": "b" * 64,
    }

    validated = validate_cartesian_hessian(atoms, np.eye(9), profile)

    assert validated.precision_profile.identity == "qualified-aimnet2025-n32-gas"


@pytest.mark.parametrize("info", [{"charge": 1}, {"charge": -1}, {"mult": 3}])
def test_bound_profile_refuses_unqualified_charge_and_spin(monkeypatch, info):
    atoms = _water()
    atoms.info.update(info)
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)

    with pytest.raises(ValueError, match="neutral closed-shell"):
        resolve_hessian_precision_profile(atoms, purpose="frequency")


def test_bound_profile_refuses_pbc(monkeypatch):
    atoms = _water()
    atoms.pbc = True
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)

    with pytest.raises(ValueError, match="nonperiodic"):
        resolve_hessian_precision_profile(atoms, purpose="frequency")


def test_drifted_exact_aimnet_claim_refuses_instead_of_downgrading(monkeypatch):
    atoms = _water()
    calculator, identity, _ = _bound_n32_calculator(monkeypatch, composed=False)
    drifted = dict(identity, state_sha256="0" * 64)
    calculator_type = type(calculator)
    monkeypatch.setattr(
        calculator_type, "analytic_cell_identity", property(lambda self: drifted)
    )
    monkeypatch.setattr(
        calculator_type, "runtime_profile_identity", property(lambda self: drifted)
    )
    atoms.calc = calculator

    with pytest.raises(ValueError, match="binding"):
        resolve_hessian_precision_profile(atoms, purpose="frequency")


@pytest.mark.parametrize("skew", [2.7851243245002877e-7, 4.159060036368345e-7])
def test_saved_n32_policy_accepts_raw_diagnostics_before_symmetrizing(monkeypatch, skew):
    atoms = _water()
    atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)
    profile = resolve_hessian_precision_profile(atoms, purpose="saved-replay")
    raw = np.eye(9)
    raw[0, 1] += skew
    before = raw.tobytes()

    validated = validate_cartesian_hessian(atoms, raw, profile)

    assert raw.tobytes() == before
    assert validated.raw_maximum_skew == pytest.approx(skew)
    assert validated.raw_sha256
    assert validated.physical_hessian.flags.writeable is False
    np.testing.assert_array_equal(
        validated.physical_hessian,
        0.5 * (raw + raw.T),
    )


def test_saved_ace_and_none_n32_replays_are_index_one(monkeypatch):
    fixture = (
        Path(__file__).with_name("data")
        / "aimnet2025_n32_nh3_saved_hessians.json"
    )
    saved = json.loads(fixture.read_text())
    assert saved["source_artifact_sha256"] == (
        "f44e204dc054fe36523a553d42689364f977e41998c768ce0ef12b361eb513cf"
    )
    observed = {}
    for cell in saved["cells"]:
        atoms = Atoms(
            cell["source_identity"]["symbols"],
            positions=cell["frozen_written_geometry"]["positions_angstrom"],
        )
        atoms.calc, _, _ = _bound_n32_calculator(monkeypatch, composed=False)
        profile = resolve_hessian_precision_profile(atoms, purpose="saved-replay")
        raw = np.asarray(
            cell["frozen_written_geometry"]["hessian_hartree_per_angstrom2"],
            dtype=np.float64,
        )
        validated = validate_cartesian_hessian(atoms, raw, profile)
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
        validate_cartesian_hessian(_water(), raw, _profile(1.0e-8))


def test_complex_hessian_refuses_before_float_cast():
    raw = np.eye(9, dtype=np.complex128)
    raw[0, 0] += 1000.0j
    with pytest.raises(ValueError, match="complex"):
        validate_cartesian_hessian(_water(), raw, _profile(1.0e-8))


def test_material_skew_refuses():
    raw = np.eye(9)
    raw[0, 1] = 1.0e-3
    with pytest.raises(ValueError, match="asymmetry"):
        validate_cartesian_hessian(_water(), raw, _profile(1.0e-8))


@pytest.mark.parametrize("bad_mass", [0.0, -1.0, np.nan])
def test_bad_masses_refuse_before_mode_analysis(bad_mass):
    atoms = _water()
    masses = atoms.get_masses()
    masses[0] = bad_mass
    atoms.set_masses(masses)
    validated = validate_cartesian_hessian(atoms, np.eye(9), _profile(1.0e-8))
    with pytest.raises(ValueError, match="positive atomic masses") as exc_info:
        build_mass_weighted_mode_basis(
            atoms,
            validated,
            project_rigid_modes=False,
        )
    assert type(exc_info.value).__name__ == "CurvatureValidationError"


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
