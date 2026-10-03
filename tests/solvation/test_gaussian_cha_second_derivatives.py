"""Focused contracts for the opt-in Gaussian-CHA analytic correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_analytic_correction import (
    FROZEN_GAUSSIAN_SIGMA_PANEL_E,
    PINNED_WATER_TOPOLOGY_SHA256,
    GaussianChaAnalyticCorrection,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_correction import (
    GaussianChaCorrection,
    KCAL_PER_MOL_PER_HARTREE,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_profiles import (
    GAUSSIAN_CHA_R6_V1_PROFILE_ID,
    GAUSSIAN_CHA_R6_V2_PROFILE_ID,
)

FIXTURE = Path(__file__).parent / "data/cha_r6_v2/frozen_v1_terminal_fixture.json"


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
    return atoms, topology


def _analytic(atoms=None, topology=None, sigma=0.01):
    if atoms is None or topology is None:
        atoms, topology = _case()
    return GaussianChaAnalyticCorrection(
        atoms,
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=sigma,
        order=64,
        numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    )


def test_constructor_is_exact_v2_order64_pinned_topology_and_frozen_width_panel():
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    assert correction.mode == "fixed"
    assert correction.inner_mode is None
    assert correction.analytic_task_derivatives_admitted is True
    assert correction.supported_properties == ("energy", "forces", "hessian", "hvp")
    assert correction.analytic_identity_provenance == {
        "model_identity": "chagb-r6-pbsa-gaussian-sign-v1",
        "numerical_profile_id": GAUSSIAN_CHA_R6_V2_PROFILE_ID,
        "topology_sha256": PINNED_WATER_TOPOLOGY_SHA256,
        "sigma_e": 0.01,
        "order": 64,
        "device": "cpu",
        "dtype": "torch.float64",
    }
    assert FROZEN_GAUSSIAN_SIGMA_PANEL_E == (0.001, 0.003, 0.01)

    for sigma in (0.002, 0.1):
        with pytest.raises(ValueError, match="sigma"):
            _analytic(atoms.copy(), topology, sigma=sigma)
    with pytest.raises(ValueError, match="order 64"):
        GaussianChaAnalyticCorrection(
            atoms.copy(),
            topology,
            expected_topology_sha256=topology.content_sha256,
            sigma_e=0.01,
            order=96,
            numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="v2 numerical profile"):
        GaussianChaAnalyticCorrection(
            atoms.copy(),
            topology,
            expected_topology_sha256=topology.content_sha256,
            sigma_e=0.01,
            order=64,
            numerical_profile_id=GAUSSIAN_CHA_R6_V1_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="pinned water topology"):
        GaussianChaAnalyticCorrection(
            atoms.copy(),
            topology,
            expected_topology_sha256="f" * 64,
            sigma_e=0.01,
            order=64,
            numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
        )
    with pytest.raises(TypeError):
        cast(Any, GaussianChaAnalyticCorrection)(
            atoms.copy(),
            topology,
            expected_topology_sha256=topology.content_sha256,
            sigma_e=0.01,
            unknown_option=True,
        )


def test_evaluate_is_exact_inherited_ef_path_and_fieldwise_equal_to_frozen_base():
    atoms, topology = _case()
    base = GaussianChaCorrection(
        atoms.copy(),
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=0.01,
        order=64,
        numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    )
    analytic_atoms = atoms.copy()
    analytic = _analytic(analytic_atoms, topology)
    assert GaussianChaAnalyticCorrection.evaluate is GaussianChaCorrection.evaluate
    expected = base.evaluate(atoms, need_forces=True)
    actual = analytic.evaluate(analytic_atoms, need_forces=True)
    assert actual.energy_hartree == expected.energy_hartree
    assert actual.components_hartree == expected.components_hartree
    np.testing.assert_array_equal(
        actual.forces_hartree_per_angstrom, expected.forces_hartree_per_angstrom
    )


def test_dense_hessian_and_direct_hvp_share_scalar_without_normalizing_direction(
    monkeypatch,
):
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    hessian = correction.get_hessian(atoms)
    assert hessian.shape == (9, 9)
    assert hessian.dtype == np.float64
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12, rtol=0.0)

    direction = np.asarray([0.3, -0.2, 0.1, -0.4, 0.5, -0.1, 0.1, -0.3, 0.2])
    directional = correction.get_directional_derivatives(atoms, direction)
    np.testing.assert_allclose(
        directional.hvp_hartree_per_angstrom2,
        hessian @ direction,
        atol=2e-12,
        rtol=0.0,
    )
    doubled = correction.get_directional_derivatives(atoms, 2.0 * direction)
    np.testing.assert_allclose(
        doubled.hvp_hartree_per_angstrom2,
        2.0 * directional.hvp_hartree_per_angstrom2,
        atol=2e-12,
        rtol=0.0,
    )
    evaluated = correction.evaluate(atoms, need_forces=True)
    assert evaluated.forces_hartree_per_angstrom is not None
    assert directional.energy_hartree == pytest.approx(
        evaluated.energy_hartree, abs=1e-15
    )
    np.testing.assert_allclose(
        directional.forces_hartree_per_angstrom,
        evaluated.forces_hartree_per_angstrom,
        atol=2e-15,
        rtol=0.0,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense Hessian helper was called by direct HVP")

    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    repeated = correction.get_directional_derivatives(atoms, direction)
    np.testing.assert_allclose(
        repeated.hvp_hartree_per_angstrom2,
        directional.hvp_hartree_per_angstrom2,
        atol=0.0,
        rtol=0.0,
    )


def test_analytic_correction_normally_forbids_instance_derivative_shadowing():
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    with pytest.raises((AttributeError, TypeError)):
        correction.get_hessian = lambda _atoms: np.zeros((9, 9))  # type: ignore[method-assign]
    with pytest.raises((AttributeError, TypeError)):
        correction.get_directional_derivatives = (  # type: ignore[method-assign]
            lambda _atoms, _direction: None
        )


def test_object_level_interface_shadow_is_detected_and_never_invoked():
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    called = False

    def fake(*_args, **_kwargs):
        nonlocal called
        called = True
        return None

    object.__setattr__(correction, "get_directional_derivatives", fake)
    assert correction.analytic_task_derivatives_admitted is False
    with pytest.raises(ValueError, match="reviewed interface"):
        correction.get_directional_derivatives(atoms, np.zeros(9, dtype=np.float64))
    assert called is False


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
def test_direction_must_be_finite_float64_flat_and_is_never_coerced(direction):
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    with pytest.raises((TypeError, ValueError), match="direction"):
        correction.get_directional_derivatives(atoms, direction)


def test_identity_and_atom_drift_fail_before_second_derivative_graph():
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    replaced = atoms.copy()
    replaced.arrays["_maple_implicit_atom_identity"][:] = [3, 2, 1]
    with pytest.raises(ValueError, match="atom identity"):
        correction.get_hessian(replaced)
    object.__setattr__(correction, "order", 96)
    with pytest.raises(ValueError, match="analytic identity"):
        correction.get_hessian(atoms)


def test_equal_content_topology_object_replacement_invalidates_analytic_identity():
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))["topology"]
    replacement = ContinuumChaTopology.from_mapping(
        payload, expected_content_sha256=payload["content_sha256"]
    )
    assert replacement == topology and replacement is not topology
    object.__setattr__(correction, "topology", replacement)
    assert correction.analytic_task_derivatives_admitted is False
    with pytest.raises(ValueError, match="topology object"):
        correction.get_hessian(atoms)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("mode", "numerical"),
        ("inner_mode", "prebuilt"),
        ("supported_properties", ("energy", "forces")),
        ("model_identity", "drifted-model"),
    ),
)
def test_declared_analytic_metadata_instance_drift_invalidates_identity(name, value):
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    object.__setattr__(correction, name, value)
    assert correction.analytic_task_derivatives_admitted is False
    with pytest.raises(ValueError, match="frozen metadata"):
        correction.get_hessian(atoms)


def test_rotational_ward_identity_uses_nonstationary_gradient_not_zero_mode():
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    hessian = correction.get_hessian(atoms)
    direction_probe = correction.get_directional_derivatives(
        atoms, np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    gradient = -direction_probe.forces_hartree_per_angstrom
    centered = atoms.positions - atoms.positions.mean(axis=0)
    omega = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rotation = centered @ omega.T
    norm = np.linalg.norm(rotation)
    assert norm > 0.0
    lhs = hessian @ (rotation.reshape(-1) / norm)
    rhs = (gradient @ omega.T).reshape(-1) / norm
    scale = max(1.0, np.max(np.abs(hessian)), np.max(np.abs(gradient)) / norm)
    np.testing.assert_allclose(lhs, rhs, atol=1e-8 * scale, rtol=0.0)
    assert np.max(np.abs(lhs)) > 1e-8


def test_hvp_force_difference_is_validation_only_with_explicit_negative_force_sign():
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    direction = np.asarray([0.3, -0.2, 0.1, -0.4, 0.5, -0.1, 0.1, -0.3, 0.2])
    direction /= np.linalg.norm(direction)
    direct = correction.get_directional_derivatives(atoms, direction)
    h = 2e-5
    displaced_forces = []
    for sign in (-1.0, 1.0):
        displaced = atoms.copy()
        displaced.positions[:] = atoms.positions + sign * h * direction.reshape(3, 3)
        result = correction.evaluate(displaced, need_forces=True)
        displaced_forces.append(result.forces_hartree_per_angstrom)
    validation_hvp = -(displaced_forces[1] - displaced_forces[0]) / (2.0 * h)
    np.testing.assert_allclose(
        direct.hvp_hartree_per_angstrom2.reshape(3, 3),
        validation_hvp,
        atol=5e-3 / KCAL_PER_MOL_PER_HARTREE,
        rtol=0.0,
    )
    assert direct.provenance["runtime_finite_difference"] is False
    assert direct.provenance["scientific_qualification_receipt"] is False


def test_zero_direction_is_valid_and_returns_exact_zero_hvp_without_changing_ef():
    atoms, topology = _case()
    correction = _analytic(atoms, topology)
    zero = correction.get_directional_derivatives(atoms, np.zeros(9, dtype=np.float64))
    np.testing.assert_array_equal(
        zero.hvp_hartree_per_angstrom2, np.zeros(9, dtype=np.float64)
    )
    evaluated = correction.evaluate(atoms, need_forces=True)
    assert evaluated.forces_hartree_per_angstrom is not None
    assert zero.energy_hartree == evaluated.energy_hartree
    np.testing.assert_allclose(
        zero.forces_hartree_per_angstrom,
        evaluated.forces_hartree_per_angstrom,
        atol=2e-15,
        rtol=0.0,
    )
