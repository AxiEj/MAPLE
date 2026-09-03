from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.extra_correction.implicit.route2_plugin_spaces import (
    ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE,
    ATOMIC_L1_PLUGIN_SOURCE_SPACE,
)
from maple.function.read.command_control import CommandControl
from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import (
    CONFIGURATION_CONTRACT_ID,
    MODEL_ID,
    PROVIDER_ID,
    Q_RAW_FROM_CARTESIAN,
    SCALAR_ID,
    TorchSmoothPCM,
    block_permutation,
)

from _torch_smooth_pcm_reference import Q, W


def _model(**changes) -> TorchSmoothPCM:
    values = {
        "atomic_numbers": W.atomic_numbers,
        "radii_angstrom": W.radii,
        "transition_width_angstrom2": W.transition_width,
        "surface_lmax": W.surface_lmax,
        "partition_lmax": W.partition_lmax,
        "partition_radial_quadrature_order": W.partition_order,
        "source_radial_quadrature_order": W.source_order,
        "double_layer_radial_quadrature_order": W.double_layer_order,
        "dielectric": W.dielectric,
        "source_shell_clearance_angstrom": 0.05,
    }
    values.update(changes)
    return TorchSmoothPCM(**values)


def test_model_reuses_the_existing_source_field_and_pairing_contracts():
    model = _model()
    assert model.source_space is ATOMIC_L1_PLUGIN_SOURCE_SPACE
    assert model.field_dual_space is ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE
    assert model.pairing is MACE_POLAR_L1_PAIRING


def test_raw_from_cartesian_permutation_is_the_frozen_yzx_mapping():
    assert np.array_equal(np.asarray(Q_RAW_FROM_CARTESIAN), Q)
    assert np.array_equal(block_permutation(3), np.kron(np.eye(3), Q))
    cartesian = np.arange(12.0).reshape(3, 4)
    expected = cartesian[:, (0, 2, 3, 1)]
    assert np.array_equal((block_permutation(3) @ cartesian.ravel()).reshape(3, 4), expected)
    assert np.array_equal(block_permutation(3).T @ block_permutation(3), np.eye(12))
    with pytest.raises((TypeError, ValueError)):
        Q_RAW_FROM_CARTESIAN[0, 0] = 2.0
    assert np.array_equal(np.asarray(Q_RAW_FROM_CARTESIAN), Q)


def test_private_model_uses_four_distinct_frozen_identities_and_no_profile():
    assert MODEL_ID == "torch-smooth-pcm-v1"
    assert PROVIDER_ID == "maple.route2.continuum.torch-smooth-pcm-schwarz.provider.v1"
    assert SCALAR_ID == "maple.route2.scalar.torch-smooth-pcm-schwarz-finite-dielectric.v1"
    assert CONFIGURATION_CONTRACT_ID == "maple.route2.continuum.torch-smooth-pcm-schwarz-config.v1"
    assert len({MODEL_ID, PROVIDER_ID, SCALAR_ID, CONFIGURATION_CONTRACT_ID}) == 4
    model = _model()
    assert not hasattr(model, "profile_id")
    assert not hasattr(model, "continuum_profile_id")


def test_private_model_admits_no_public_capability():
    model = _model()
    capabilities = model.capabilities
    if hasattr(capabilities, "enabled_tiers"):
        assert capabilities.enabled_tiers == ()
    elif isinstance(capabilities, dict):
        assert capabilities == {}
    else:
        assert tuple(capabilities) == ()
    audit = model.audit(W.positions, W.source)
    assert audit["capabilities"] == []
    assert audit["historical_evidence_transferable"] is False


def test_derivative_methods_are_final_at_class_definition():
    with pytest.raises(TypeError, match="final|same-scalar"):
        class IllegalDerivativeOverride(TorchSmoothPCM):
            def drive_cartesian(self, geometry, source):
                return np.zeros_like(source)


@pytest.mark.parametrize("atomic_numbers", [(99,), (1,), (118,), (1000,)])
def test_any_positive_integer_atomic_number_is_accepted_with_explicit_radius(atomic_numbers):
    TorchSmoothPCM(atomic_numbers=atomic_numbers, radii_angstrom=(1.5,), surface_lmax=1, partition_lmax=2)


@pytest.mark.parametrize("atomic_numbers", [(0,), (-1,), (1.5,), (True,)])
def test_nonpositive_or_noninteger_atomic_number_is_rejected(atomic_numbers):
    with pytest.raises((TypeError, ValueError), match="atomic|integer|positive"):
        TorchSmoothPCM(atomic_numbers=atomic_numbers, radii_angstrom=(1.5,), surface_lmax=1, partition_lmax=2)


@pytest.mark.parametrize("radii", [(), (0.0,), (-1.0,), (np.nan,), (np.inf,)])
def test_missing_nonpositive_or_nonfinite_radius_is_rejected(radii):
    with pytest.raises((TypeError, ValueError), match="radi|positive|finite|count"):
        TorchSmoothPCM(atomic_numbers=(1,), radii_angstrom=radii, surface_lmax=1, partition_lmax=2)


@pytest.mark.parametrize("dielectric", [1.0, 0.0, -3.0, np.nan, np.inf, True])
def test_invalid_finite_dielectric_is_rejected(dielectric):
    with pytest.raises((TypeError, ValueError), match="dielectric|epsilon"):
        _model(dielectric=dielectric)


@pytest.mark.parametrize("clearance", [0.0, -0.1, 0.049999, np.nan, np.inf])
def test_source_shell_clearance_has_a_hard_minimum_of_point_zero_five_angstrom(clearance):
    with pytest.raises((TypeError, ValueError), match="clearance|0.05|finite"):
        _model(source_shell_clearance_angstrom=clearance)


def test_configuration_is_immutable_and_content_addressed():
    model = _model()
    digest = model.configuration_sha256()
    assert len(digest) == 64
    with pytest.raises((AttributeError, TypeError)):
        model.dielectric = 2.0
    assert model.configuration_sha256() == digest


def test_implementation_provenance_drift_fails_closed(monkeypatch):
    from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import scalar

    model = _model()
    monkeypatch.setattr(scalar, "implementation_hashes", lambda: (("changed.py", "0" * 64),))
    with pytest.raises(RuntimeError, match="drift|provenance"):
        model.configuration_sha256()


def test_configuration_hash_changes_for_mathematical_settings_not_runtime_identity():
    baseline = _model()
    changes = (
        {"atomic_numbers": (7, 1, 1)},
        {"radii_angstrom": (2.295, 1.2, 1.2)},
        {"transition_width_angstrom2": 0.081},
        {"surface_lmax": 2},
        {"partition_lmax": 5},
        {"partition_radial_quadrature_order": 64},
        {"source_radial_quadrature_order": 96},
        {"double_layer_radial_quadrature_order": 96},
        {"dielectric": 79.0},
        {"source_shell_clearance_angstrom": 0.051},
    )
    digests = {baseline.configuration_sha256()}
    digests.update(_model(**change).configuration_sha256() for change in changes)
    assert len(digests) == len(changes) + 1
    for changed in (_model(**change) for change in changes):
        assert (changed.model_id, changed.provider_id, changed.scalar_id, changed.configuration_contract_id) == (
            MODEL_ID, PROVIDER_ID, SCALAR_ID, CONFIGURATION_CONTRACT_ID
        )

    record = baseline.execution_provenance()
    assert record["configuration_sha256"] == baseline.configuration_sha256()
    assert len(record["execution_provenance_sha256"]) == 64
    assert {"torch_version", "numpy_version", "scipy_version", "ase_version"} <= set(record)
    assert any("cpu" in key for key in record)
    assert any("blas" in key or "backend" in key for key in record)


def test_debug_arrays_are_detached_copies_and_cannot_mutate_the_scalar():
    model = _model(surface_lmax=1, partition_lmax=2, partition_radial_quadrature_order=32, source_radial_quadrature_order=48, double_layer_radial_quadrature_order=48)
    before = model.energy(W.positions, W.source)
    matrices = model.debug_geometry_matrices(W.positions)
    matrices["B"].flat[0] += 1000.0
    assert model.energy(W.positions, W.source) == pytest.approx(before, abs=2e-12, rel=0.0)


@pytest.mark.parametrize(
    "positions,source,match",
    [
        (np.zeros((3, 3)), W.source, "distinct|coincident"),
        (W.positions, np.zeros((3, 5)), "shape|l1|source"),
        (np.full((3, 3), np.nan), W.source, "finite"),
        (W.positions, np.full((3, 4), np.nan), "finite"),
    ],
)
def test_evaluation_fails_closed_on_invalid_geometry_or_source(positions, source, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _model().energy(positions, source)


def test_evaluation_rejects_non_float64_numpy_inputs():
    with pytest.raises(TypeError, match="float64"):
        _model().energy(W.positions.astype(np.float32), W.source)


@pytest.mark.parametrize("change", [{"surface_lmax": 17}, {"partition_lmax": 33}, {"partition_radial_quadrature_order": 4097}])
def test_orders_are_bounded_before_dense_allocation(change):
    with pytest.raises(ValueError, match="bound|order|lmax|dense"):
        _model(**change)


def test_huge_finite_source_that_produces_nonfinite_output_fails_closed():
    with pytest.raises((ValueError, RuntimeError, FloatingPointError), match="finite|overflow|output"):
        _model(surface_lmax=1, partition_lmax=2).energy(W.positions, np.full_like(W.source, 1e308))


def test_public_command_parser_rejects_private_provider_and_every_private_id():
    for provider in ("torch-smooth-pcm", MODEL_ID, PROVIDER_ID, SCALAR_ID, CONFIGURATION_CONTRACT_ID):
        with pytest.raises(ValueError, match="provider"):
            CommandControl.from_settings(
                ["#model=macepol-m", "#sp(verbose=1)", f"#solv(implicit=water,method=smd,provider={provider},profile=x,response=frozen,standard_state=1m,experimental=true)"]
            )


def test_checkpoint_is_not_loaded_by_private_continuum(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("continuum must not load MACE checkpoint"))
    model = _model(surface_lmax=1, partition_lmax=2, partition_radial_quadrature_order=32, source_radial_quadrature_order=48, double_layer_radial_quadrature_order=48)
    assert np.isfinite(model.energy(W.positions, W.source))
