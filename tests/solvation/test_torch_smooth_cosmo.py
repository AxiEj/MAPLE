from __future__ import annotations

import numpy as np
import pytest
from typing import Any

pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_smooth_cosmo import (
    TORCH_SMOOTH_COSMO_MODEL_ID,
    TorchSmoothCOSMO,
)
from maple.function.calculator.extra_correction.implicit.torch_smooth_pcm import (
    TorchSmoothPCM,
)

POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [1.35, 0.1, -0.2]],
    dtype=np.float64,
)
SOURCE = np.asarray(
    [[0.2, 0.04, -0.02, 0.03], [-0.2, -0.04, 0.02, -0.03]],
    dtype=np.float64,
)


def _kwargs() -> dict[str, Any]:
    return {
        "atomic_numbers": (8, 1),
        "radii_angstrom": (1.52, 1.20),
        "transition_width_angstrom2": 0.08,
        "surface_lmax": 1,
        "partition_lmax": 2,
        "partition_radial_quadrature_order": 16,
        "source_radial_quadrature_order": 24,
        "double_layer_radial_quadrature_order": 24,
    }


def test_conductor_operator_is_the_exact_binary64_limit():
    model = TorchSmoothCOSMO(**_kwargs())
    matrices = model.debug_geometry_matrices(POSITIONS)

    np.testing.assert_allclose(
        matrices["A_eps"],
        matrices["A_inf"],
        atol=0.0,
        rtol=0.0,
    )
    assert model.model_id == TORCH_SMOOTH_COSMO_MODEL_ID
    assert model.finite_dielectric is False
    assert model.conductor_limit is True


def test_conductor_energy_and_derivatives_match_the_finite_float_limit():
    conductor = TorchSmoothCOSMO(**_kwargs())
    finite_limit = TorchSmoothPCM(
        **_kwargs(),
        dielectric=float(np.finfo(np.float64).max),
    )

    assert conductor.energy(POSITIONS, SOURCE) == pytest.approx(
        finite_limit.energy(POSITIONS, SOURCE),
        abs=0.0,
        rel=0.0,
    )
    np.testing.assert_allclose(
        conductor.drive_cartesian(POSITIONS, SOURCE),
        finite_limit.drive_cartesian(POSITIONS, SOURCE),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        conductor.coordinate_gradient(POSITIONS, SOURCE),
        finite_limit.coordinate_gradient(POSITIONS, SOURCE),
        atol=0.0,
        rtol=0.0,
    )


def test_conductor_configuration_is_separate_and_immutable():
    model = TorchSmoothCOSMO(**_kwargs())
    payload = model._configuration_payload()

    assert payload["continuum_model"] == "COSMO"
    assert payload["conductor_limit"] is True
    assert payload["finite_dielectric"] is False
    assert model.configuration_sha256() == model.configuration_sha256()
    with pytest.raises(AttributeError):
        model.dielectric = 78.355
    with pytest.raises(ValueError, match="owns the conductor"):
        TorchSmoothCOSMO(**_kwargs(), dielectric=78.355)
