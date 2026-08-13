from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.api.scalar_registry import (
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
)
from maple.solvation.continuum import (
    ContinuumEnergyFunctional,
    ConjugateRadialGTOFixedTopologyCPCMBackend,
    FixedReciprocalCPCMFunctional,
)
from maple.solvation.continuum.functional import _torch

ROOT = Path(__file__).resolve().parents[2]
SIX_POINT_SPHERE = np.asarray(
    [
        [1.0, 0.0, 0.0, 1.0 / 6.0],
        [-1.0, 0.0, 0.0, 1.0 / 6.0],
        [0.0, 1.0, 0.0, 1.0 / 6.0],
        [0.0, -1.0, 0.0, 1.0 / 6.0],
        [0.0, 0.0, 1.0, 1.0 / 6.0],
        [0.0, 0.0, -1.0, 1.0 / 6.0],
    ],
    dtype=float,
)
POSITIONS = np.asarray([[0.0, 0.0, 0.0], [2.5, 0.2, -0.1]])
SOURCE = np.asarray(
    [
        [0.30, 0.02, 0.02, 0.01, -0.01, 0.005, -0.008, 0.012],
        [-0.32, 0.00, 0.01, -0.02, 0.03, -0.006, 0.009, -0.004],
    ]
)


def _backend():
    return ConjugateRadialGTOFixedTopologyCPCMBackend(
        ("H", "H"),
        np.asarray([1.1, 1.1]),
        dielectric=78.39,
        unit_sphere=SIX_POINT_SPHERE,
        switching_constant=4.84566077868,
        runtime_version="continuum-functional-unit-grid-v1",
    )


def _functional():
    torch = pytest.importorskip("torch")
    return FixedReciprocalCPCMFunctional(_backend(), dtype=torch.float64, device="cpu")


def test_continuum_functional_contract_imports_without_torch():
    script = r"""
import sys
sys.modules['torch'] = None
from maple.solvation.continuum import ContinuumEnergyFunctional
assert ContinuumEnergyFunctional.__name__ == 'ContinuumEnergyFunctional'
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fixed_reciprocal_functional_is_disabled_and_content_bound():
    functional = _functional()
    backend = functional.backend
    assert functional.scalar_id == (
        VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1
    )
    assert functional.capabilities.enabled_tiers == ()
    assert functional.scalar_first is True
    assert functional.electrostatics_only is True
    assert functional.include_nonpolar is False
    assert functional.fixed_topology is True
    assert functional.source_dependent_geometry is False
    assert functional.linear_response is True
    assert functional.reciprocal is True
    assert functional.source_space is backend.source_space
    assert functional.field_space is backend.field_space
    assert functional.pairing is backend.pairing
    assert len(functional.configuration_sha256()) == 64
    assert len(functional.provenance_sha256) == 64
    provenance = dict(functional.runtime_provenance())
    assert provenance["derivative_route"] == "sealed-same-scalar-autograd"
    assert provenance["capabilities"] == "none-pending-strict-variational-gates"
    with pytest.raises(AttributeError, match="immutable"):
        functional._backend = _backend()


def test_one_scalar_generates_backend_drive_energy_and_source_hvp():
    functional = _functional()
    backend = functional.backend
    rng = np.random.default_rng(20260814)
    direction = rng.normal(scale=0.08, size=SOURCE.shape)
    cotangent = rng.normal(scale=0.08, size=SOURCE.shape)

    energy = functional.energy_eV(POSITIONS, SOURCE)
    drive = functional.drive(POSITIONS, SOURCE)
    hvp = functional.source_hvp(POSITIONS, SOURCE, direction)
    jvp = functional.source_jvp(POSITIONS, SOURCE, direction)
    vjp = functional.source_vjp(POSITIONS, SOURCE, cotangent)
    np.testing.assert_allclose(
        drive, backend.evaluate_field(POSITIONS, SOURCE), rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        hvp, backend.source_vjp(POSITIONS, SOURCE, direction), rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        jvp, backend.source_jvp(POSITIONS, SOURCE, direction), rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        vjp, backend.source_vjp(POSITIONS, SOURCE, cotangent), rtol=3e-13, atol=3e-13
    )
    assert energy == pytest.approx(backend.energy(POSITIONS, SOURCE), abs=3e-13)
    assert energy == pytest.approx(
        0.5 * functional.pairing.pair(SOURCE, drive), abs=3e-13
    )
    assert np.vdot(cotangent, jvp) == pytest.approx(np.vdot(vjp, direction), abs=3e-12)

    step = 1.0e-6
    energy_fd = (
        functional.energy_eV(POSITIONS, SOURCE + step * direction)
        - functional.energy_eV(POSITIONS, SOURCE - step * direction)
    ) / (2.0 * step)
    assert np.vdot(drive, direction) == pytest.approx(energy_fd, abs=2e-8)


def test_scalar_generated_coordinate_partial_matches_backend_and_finite_difference():
    functional = _functional()
    backend = functional.backend
    analytic = functional.coordinate_partial(POSITIONS, SOURCE)
    backend_analytic = backend.coordinate_vjp(POSITIONS, SOURCE, 0.5 * SOURCE).reshape(
        2, 3
    )
    np.testing.assert_allclose(analytic, backend_analytic, rtol=0.0, atol=2e-7)

    finite_difference = np.empty((2, 3))
    step = 1.0e-5
    for atom in range(2):
        for axis in range(3):
            plus = POSITIONS.copy()
            minus = POSITIONS.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            finite_difference[atom, axis] = (
                functional.energy_eV(plus, SOURCE) - functional.energy_eV(minus, SOURCE)
            ) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(analytic.sum(axis=0), np.zeros(3), atol=8e-11)


@pytest.mark.parametrize(
    "positions",
    [
        np.asarray([[0.0, 0.0, 0.0], [2.20, -0.30, 0.25]]),
        np.asarray([[-0.4, 0.2, 0.1], [2.75, 0.45, -0.35]]),
    ],
)
def test_scalar_first_fixed_cpcm_matches_backend_across_geometries(positions):
    functional = _functional()
    backend = functional.backend
    np.testing.assert_allclose(
        functional.drive(positions, SOURCE),
        backend.evaluate_field(positions, SOURCE),
        rtol=8e-13,
        atol=8e-13,
    )
    assert functional.energy_eV(positions, SOURCE) == pytest.approx(
        backend.energy(positions, SOURCE), abs=8e-13
    )
    np.testing.assert_allclose(
        functional.coordinate_partial(positions, SOURCE),
        backend.coordinate_vjp(positions, SOURCE, 0.5 * SOURCE).reshape(2, 3),
        rtol=0.0,
        atol=3e-7,
    )


def test_derivative_surface_is_final_and_cannot_be_hand_coded():
    with pytest.raises(TypeError, match="final"):

        class BadFunctional(ContinuumEnergyFunctional):
            def _energy_torch(self, positions, source):
                del positions
                return source.sum()

            def drive(self, *args, **kwargs):
                return None

    assert callable(_torch)


def test_functional_rejects_invalid_source_shape_and_nonfinite_values():
    functional = _functional()
    with pytest.raises(ValueError):
        functional.energy_eV(POSITIONS, SOURCE[:, :4])
    bad_source = SOURCE.copy()
    bad_source[0, 0] = np.nan
    with pytest.raises(ValueError):
        functional.drive(POSITIONS, bad_source)
    with pytest.raises(ValueError):
        functional.coordinate_partial(POSITIONS[:1], SOURCE)
    wrong_elements = Atoms("HeHe", positions=POSITIONS)
    with pytest.raises(ValueError, match="atomic numbers"):
        functional.energy_eV(wrong_elements, SOURCE)
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        functional.energy_torch(
            torch.tensor(POSITIONS, dtype=torch.float32),
            torch.tensor(SOURCE, dtype=torch.float32),
        )


class PermutedQuadraticContinuum(ContinuumEnergyFunctional):
    __slots__ = ()

    def _energy_torch(self, positions, source):
        del positions
        matrix = source.new_tensor([[2.0, 0.4], [0.4, 1.3]])
        vector = source.reshape(-1)
        return 0.5 * vector @ matrix @ vector


def test_generic_scalar_first_vjp_respects_a_nonidentity_q_pairing():
    from maple.solvation.coupling.metrics import PairingMetric
    from maple.solvation.coupling.spaces import FieldDualSpace, SourceSpace

    torch = pytest.importorskip("torch")
    pairing = PairingMetric(
        scalar_id="test.permuted-pairing.v1",
        source_components=("s0", "s1"),
        field_components=("f0", "f1"),
        source_units=("u0", "u1"),
        field_units=("d0", "d1"),
        field_to_source_indices=(1, 0),
        gauge="test-gauge",
        field_convention="test-energy-dual",
    )
    source_space = SourceSpace(
        scalar_id="test.permuted-source.v1",
        representation="test source",
        components=pairing.source_components,
        units=pairing.source_units,
    )
    field_space = FieldDualSpace(
        scalar_id="test.permuted-field.v1",
        representation="test field",
        components=pairing.field_components,
        units=pairing.field_units,
        source_space=source_space,
        pairing_metric=pairing,
    )
    functional = PermutedQuadraticContinuum(
        source_space=source_space,
        field_space=field_space,
        pairing=pairing,
        dtype=torch.float64,
        device="cpu",
    )
    geometry = np.zeros((1, 3))
    source = np.asarray([[0.3, -0.2]])
    direction = np.asarray([[0.7, -0.1]])
    cotangent = np.asarray([[-0.5, 0.9]])
    jvp = functional.source_jvp(geometry, source, direction)
    vjp = functional.source_vjp(geometry, source, cotangent)
    assert np.vdot(cotangent, jvp) == pytest.approx(np.vdot(vjp, direction), abs=3e-14)
    step = 1.0e-6
    drive_fd = (
        functional.drive(geometry, source + step * direction)
        - functional.drive(geometry, source - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(jvp, drive_fd, atol=3e-11, rtol=0.0)
