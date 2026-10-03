"""Strict pinned MACE plus analytic Gaussian-CHA composition contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_analytic_correction import (
    GaussianChaAnalyticCorrection,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_correction import (
    GaussianChaCorrection,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_profiles import (
    GAUSSIAN_CHA_R6_V2_PROFILE_ID,
)
from maple.function.calculator.mace._mace_calculator import MACECalculator
from maple.function.calculator.mace._mace_cha_analytic_calculator import (
    EXPECTED_MACEOFF23M_SHA256,
    MACEChaAnalyticCalculator,
)
from maple.function.calculator.calculator_base import get_registered_calculator

FIXTURE = Path(__file__).parent / "data/cha_r6_v2/frozen_v1_terminal_fixture.json"
MODEL = Path(__file__).parents[2] / "maple/function/calculator/model/maceoff23m.pt"


def _case():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    topology_payload = payload["topology"]
    topology = ContinuumChaTopology.from_mapping(
        topology_payload,
        expected_content_sha256=topology_payload["content_sha256"],
    )
    row = next(
        item for item in payload["rows"] if item["row_id"] == "center-00-sigma-02"
    )
    atoms = Atoms(topology.atomic_numbers, positions=row["positions_angstrom"])
    atoms.new_array(
        "_maple_implicit_atom_identity", np.asarray(topology.atom_ids, dtype=np.int64)
    )
    correction = GaussianChaAnalyticCorrection(
        atoms,
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=0.01,
        order=64,
        numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    )
    return atoms, topology, correction


@pytest.fixture(scope="module")
def prepared_pair():
    atoms, _topology, correction = _case()
    calculator = MACEChaAnalyticCalculator(
        correction=correction,
        model_path=MODEL,
        device="cpu",
        model="maceoff23m",
    )
    calculator.prepare_analytic_derivatives()
    return atoms, correction, calculator


def test_real_checkpoint_constructor_captures_exact_native_identity(prepared_pair):
    _atoms, correction, calculator = prepared_pair
    assert MACEChaAnalyticCalculator.calculate is MACECalculator.calculate
    assert calculator.checkpoint_sha256 == EXPECTED_MACEOFF23M_SHA256
    assert calculator.checkpoint_path == str(MODEL.resolve())
    assert calculator.native_model_identity == {
        "model": "maceoff23m",
        "device": "cpu",
        "dtype": "torch.float64",
        "parameter_tensor_count": 29,
        "float64_buffer_count": 48,
        "int64_buffer_count": 2,
        "other_buffer_dtypes": (),
        "r_max_angstrom": 5.0,
        "atomic_numbers": (1, 6, 7, 8, 9, 15, 16, 17, 35, 53),
        "training": False,
        "all_parameter_requires_grad_false": True,
    }
    assert calculator.solvent_correction is correction
    assert calculator.analytic_implicit_derivatives_admitted is True
    assert calculator.analytic_binding_provenance["correction_fingerprint"] == (
        correction.analytic_identity_fingerprint
    )
    assert get_registered_calculator("maceoff23m") is MACECalculator


def test_constructor_rejects_wrong_device_model_path_hash_correction_and_unknown_kwargs(
    monkeypatch, tmp_path
):
    atoms, topology, correction = _case()
    for kwargs, message in (
        ({"device": "cuda"}, "CPU"),
        ({"model": "maceoff23s"}, "maceoff23m"),
        ({"model_path": tmp_path / "missing.pt"}, "exact local"),
    ):
        values = {
            "correction": correction,
            "model_path": MODEL,
            "device": "cpu",
            "model": "maceoff23m",
            **kwargs,
        }
        with pytest.raises((TypeError, ValueError), match=message):
            MACEChaAnalyticCalculator(**values)

    base = GaussianChaCorrection(
        atoms.copy(),
        topology,
        expected_topology_sha256=correction.expected_topology_sha256,
        sigma_e=0.01,
        order=64,
        numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    )
    with pytest.raises(TypeError, match="GaussianChaAnalyticCorrection"):
        MACEChaAnalyticCalculator(
            correction=cast(Any, base),
            model_path=MODEL,
            device="cpu",
            model="maceoff23m",
        )

    class UnreviewedAnalyticCorrection(GaussianChaAnalyticCorrection):
        pass

    unreviewed = UnreviewedAnalyticCorrection(
        atoms.copy(),
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=0.01,
        order=64,
        numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    )
    with pytest.raises(TypeError, match="exact reviewed"):
        MACEChaAnalyticCalculator(
            correction=cast(Any, unreviewed),
            model_path=MODEL,
            device="cpu",
            model="maceoff23m",
        )
    with pytest.raises(TypeError):
        cast(Any, MACEChaAnalyticCalculator)(
            correction=correction,
            model_path=MODEL,
            device="cpu",
            model="maceoff23m",
            unknown_option=True,
        )

    import maple.function.calculator.mace._mace_cha_analytic_calculator as module

    monkeypatch.setattr(module, "EXPECTED_MACEOFF23M_SHA256", "0" * 64)

    def loader_must_not_run(*_args, **_kwargs):
        raise AssertionError("checkpoint hash was not checked before model load")

    monkeypatch.setattr(MACECalculator, "__init__", loader_must_not_run)
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        MACEChaAnalyticCalculator(
            correction=correction,
            model_path=MODEL,
            device="cpu",
            model="maceoff23m",
        )


def test_preparation_and_every_derivative_call_fail_closed_on_replacement_or_hash_drift(
    prepared_pair, monkeypatch
):
    atoms, correction, calculator = prepared_pair
    assert calculator.analytic_implicit_derivatives_admitted is True
    with pytest.raises(AttributeError, match="prepare_analytic_derivatives"):
        calculator.analytic_implicit_derivatives_admitted = True
    assert calculator.analytic_implicit_derivatives_admitted is True
    calculator.solvent_correction = None
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(RuntimeError, match="analytic binding"):
        calculator.get_hessian(atoms)
    calculator.solvent_correction = correction
    calculator.prepare_analytic_derivatives()

    import maple.function.calculator.mace._mace_cha_analytic_calculator as module

    observed = calculator.checkpoint_sha256
    monkeypatch.setattr(module, "_sha256_file", lambda _path: "f" * 64)
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(RuntimeError, match="checkpoint"):
        calculator.get_hvp(atoms, np.ones(9, dtype=np.float64))
    monkeypatch.setattr(module, "_sha256_file", lambda _path: observed)
    calculator.prepare_analytic_derivatives()

    _atoms, _topology, replacement = _case()
    calculator.solvent_correction = replacement
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(RuntimeError, match="construction identity"):
        calculator.get_hessian(atoms)
    calculator.solvent_correction = correction
    calculator.prepare_analytic_derivatives()

    original_dtype = calculator.dtype
    calculator.dtype = torch.float32
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(ValueError, match="effective dtype"):
        calculator.get_hessian(atoms)
    calculator.dtype = original_dtype
    calculator.prepare_analytic_derivatives()

    original_path = calculator.checkpoint_path
    original_hash = calculator.checkpoint_sha256
    calculator.checkpoint_sha256 = "e" * 64
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(RuntimeError, match="checkpoint binding"):
        calculator.get_hessian(atoms)
    calculator.checkpoint_path = original_path
    calculator.checkpoint_sha256 = original_hash
    calculator.prepare_analytic_derivatives()

    calculator.hessian = "numerical"
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(RuntimeError, match="analytic Hessian mode"):
        calculator.get_hessian(atoms)
    calculator.hessian = "analytic"
    calculator.prepare_analytic_derivatives()

    calculator.analytic_implicit_derivatives_admitted = False
    assert calculator.analytic_implicit_derivatives_admitted is False
    assert calculator.analytic_binding_provenance["prepared"] is False
    with pytest.raises(AttributeError, match="prepare_analytic_derivatives"):
        calculator.analytic_implicit_derivatives_admitted = True
    assert calculator.analytic_implicit_derivatives_admitted is False
    calculator.prepare_analytic_derivatives()


def test_actual_loaded_model_object_training_and_tensor_version_drift_are_rejected():
    atoms, _topology, correction = _case()
    calculator = MACEChaAnalyticCalculator(
        correction=correction,
        model_path=MODEL,
        device="cpu",
        model="maceoff23m",
    )
    calculator.prepare_analytic_derivatives()
    original_model = calculator.model

    replacement = torch.jit.load(str(MODEL), map_location="cpu")
    replacement.eval()
    for parameter in replacement.parameters():
        parameter.requires_grad_(False)
    calculator.model = replacement
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(RuntimeError, match="actual loaded model state"):
        calculator.get_hessian(atoms)

    calculator.model = original_model
    calculator.prepare_analytic_derivatives()
    calculator.model.train()
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(RuntimeError, match="actual loaded model state"):
        calculator.get_hessian(atoms)

    calculator.model.eval()
    calculator.prepare_analytic_derivatives()
    parameter = next(calculator.model.parameters())
    with torch.no_grad():
        parameter.add_(1e-12)
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises(RuntimeError, match="actual loaded model state"):
        calculator.get_hessian(atoms)


def test_inherited_composed_ef_dense_hessian_and_direct_hvp_are_coherent(prepared_pair):
    atoms, correction, calculator = prepared_pair
    calculator.calculate(
        atoms, properties=["energy", "forces"], system_changes=all_changes
    )
    energy = float(calculator.results["energy"])
    forces = np.asarray(calculator.results["forces"], dtype=np.float64)
    hessian = calculator.get_hessian(atoms)
    assert hessian.shape == (9, 9)
    assert np.isfinite(hessian).all()

    direction = np.asarray([0.3, -0.2, 0.1, -0.4, 0.5, -0.1, 0.1, -0.3, 0.2])
    hvp, hvp_forces, hvp_energy = calculator.get_hvp(atoms, direction)
    np.testing.assert_allclose(hvp, hessian @ direction, atol=2e-10, rtol=0.0)
    np.testing.assert_allclose(hvp_forces.reshape(3, 3), forces, atol=2e-12, rtol=0.0)
    assert hvp_energy == pytest.approx(energy, abs=2e-13)
    solvent = correction.get_directional_derivatives(atoms, direction)
    assert calculator.last_composed_hvp_provenance["solvent_derivative"] == (
        solvent.provenance["derivative"]
    )


def test_direct_composed_hvp_poison_dense_hessian_and_finite_difference_paths(
    prepared_pair, monkeypatch
):
    atoms, _correction, calculator = prepared_pair
    direction = np.asarray([0.3, -0.2, 0.1, -0.4, 0.5, -0.1, 0.1, -0.3, 0.2])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense Hessian or finite-difference path was called")

    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    import maple.function.calculator.calculator_base as calculator_base
    import maple.function.calculator.mace._mace_calculator as mace_module

    monkeypatch.setattr(calculator_base, "numerical_hessian_from_atoms", forbidden)
    monkeypatch.setattr(calculator_base, "hessian_via_double_autograd", forbidden)
    monkeypatch.setattr(mace_module, "hessian_via_double_autograd", forbidden)
    hvp, forces, energy = calculator.get_hvp(atoms, direction)
    assert hvp.shape == (9,)
    assert forces.shape == (9,)
    assert np.isfinite(hvp).all() and np.isfinite(forces).all() and np.isfinite(energy)


def test_object_level_correction_and_mace_interface_shadows_are_rejected_not_invoked(
    prepared_pair,
):
    atoms, correction, calculator = prepared_pair
    direction = np.ones(9, dtype=np.float64)
    called = False

    def fake(*_args, **_kwargs):
        nonlocal called
        called = True
        return None

    object.__setattr__(correction, "get_directional_derivatives", fake)
    assert calculator.analytic_implicit_derivatives_admitted is False
    with pytest.raises((RuntimeError, ValueError), match="reviewed interface"):
        calculator.get_hvp(atoms, direction)
    assert called is False
    object.__delattr__(correction, "get_directional_derivatives")
    calculator.prepare_analytic_derivatives()

    for name, derivative in (
        ("_gas_energy_leaf_hartree", "hvp"),
        ("_strict_direction", "hvp"),
        ("_analytic_hessian", "hessian"),
        ("_collect_native_model_identity", "hessian"),
        ("_native_model_state_signature", "hessian"),
    ):
        object.__setattr__(calculator, name, fake)
        assert calculator.analytic_implicit_derivatives_admitted is False
        with pytest.raises(RuntimeError, match="reviewed analytic interface"):
            if derivative == "hvp":
                calculator.get_hvp(atoms, direction)
            else:
                calculator.get_hessian(atoms)
        assert called is False
        object.__delattr__(calculator, name)
        calculator.prepare_analytic_derivatives()


def test_mace_trusted_class_interface_drift_is_rejected(prepared_pair, monkeypatch):
    atoms, _correction, calculator = prepared_pair

    def fake(*_args, **_kwargs):
        raise AssertionError("drifted class helper was invoked")

    with monkeypatch.context() as context:
        context.setattr(MACEChaAnalyticCalculator, "_strict_direction", fake)
        assert calculator.analytic_implicit_derivatives_admitted is False
        with pytest.raises(RuntimeError, match="reviewed analytic interface"):
            calculator.get_hvp(atoms, np.ones(9, dtype=np.float64))
    calculator.prepare_analytic_derivatives()


def test_atom_preflight_rejects_before_any_gas_hessian_or_hvp_graph(
    prepared_pair,
):
    atoms, _correction, calculator = prepared_pair
    before = calculator.analytic_graph_counters
    invalid_atoms = []
    reordered = atoms[[1, 0, 2]]
    invalid_atoms.append(reordered)
    nonfinite = atoms.copy()
    nonfinite.positions[0, 0] = np.nan
    invalid_atoms.append(nonfinite)
    periodic = atoms.copy()
    periodic.set_cell([20.0, 20.0, 20.0])
    periodic.set_pbc(True)
    invalid_atoms.append(periodic)
    outside_domain = atoms.copy()
    outside_domain.positions[:] = [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]]
    invalid_atoms.append(outside_domain)
    direction = np.ones(9, dtype=np.float64)
    for invalid in invalid_atoms:
        with pytest.raises((TypeError, ValueError)):
            calculator.get_hvp(invalid, direction)
        with pytest.raises((TypeError, ValueError)):
            calculator.get_hessian(invalid)
    assert calculator.analytic_graph_counters == before


@pytest.mark.parametrize(
    "direction",
    (
        np.zeros((3, 3), dtype=np.float64),
        np.zeros(8, dtype=np.float64),
        np.full(9, np.nan, dtype=np.float64),
        np.zeros(9, dtype=np.float32),
        [0.0] * 9,
    ),
)
def test_composed_hvp_direction_is_strict_finite_float64_flat(prepared_pair, direction):
    atoms, _correction, calculator = prepared_pair
    with pytest.raises((TypeError, ValueError), match="direction"):
        calculator.get_hvp(atoms, direction)


def test_zero_direction_composed_hvp_is_exact_zero_and_preserves_center_ef(
    prepared_pair,
):
    atoms, _correction, calculator = prepared_pair
    calculator.calculate(
        atoms, properties=["energy", "forces"], system_changes=all_changes
    )
    expected_energy = float(calculator.results["energy"])
    expected_forces = np.asarray(calculator.results["forces"]).reshape(-1)
    hvp, forces, energy = calculator.get_hvp(atoms, np.zeros(9, dtype=np.float64))
    np.testing.assert_array_equal(hvp, np.zeros(9, dtype=np.float64))
    np.testing.assert_allclose(forces, expected_forces, atol=2e-12, rtol=0.0)
    assert energy == pytest.approx(expected_energy, abs=2e-13)


def test_ase_hessian_property_remains_explicitly_unsupported_with_solvent(
    prepared_pair,
):
    atoms, _correction, calculator = prepared_pair
    with pytest.raises(NotImplementedError):
        calculator.calculate(atoms, properties=["hessian"], system_changes=all_changes)
