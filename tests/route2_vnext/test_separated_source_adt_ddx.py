from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.gto_density import (
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.continuum import (
    CANONICAL_ADT_SEPARATED_DDX_PROVIDER_ID,
    CanonicalADTDDXProblemData,
    CanonicalADTModelFieldCotangents,
    CanonicalADTSeparatedDDXState,
    PreparedCanonicalADTSeparatedDDX,
    SeparatedSourceDDXBackend,
    embed_atomic_l1_in_first_radial_channel,
    extract_atomic_l1_first_radial_cotangent,
    prepare_canonical_adt_separated_ddx,
)
from maple.solvation.coupling.atomic_displacement_lift import (
    CanonicalAtomicDisplacementLift,
)

POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.43, 0.10, -0.20], [0.20, 1.10, 0.30]])
PERMANENT = np.asarray(
    [
        [0.12, 0.02, 0.01, -0.01],
        [-0.08, -0.01, 0.03, 0.02],
        [-0.04, 0.004, -0.005, 0.006],
    ]
)
RADIAL4 = np.asarray(
    [
        [0.010, -0.003, 0.002, 0.001],
        [-0.008, 0.001, -0.001, 0.003],
        [-0.002, 0.000, 0.0004, -0.0005],
    ]
)
ADT = np.asarray(
    [
        [0.020, -0.010, 0.006],
        [-0.012, 0.004, 0.009],
        [0.003, 0.006, -0.004],
    ]
)


def _backend() -> SeparatedSourceDDXBackend:
    return SeparatedSourceDDXBackend(
        ("H", "H", "H"),
        np.asarray([1.5, 1.6, 1.2]),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
    )


def _lift(*, atomic_number: int = 1) -> CanonicalAtomicDisplacementLift:
    mixture = GaussianMixtureAtom(np.asarray([float(atomic_number)]), np.asarray([1.2]))
    return CanonicalAtomicDisplacementLift.from_mixtures(
        atomic_numbers=np.full(3, atomic_number, dtype=np.int64),
        mixtures_by_atomic_number={atomic_number: mixture},
        source_asset_sha256="a" * 64,
    )


def _prepared() -> tuple[object, PreparedCanonicalADTSeparatedDDX]:
    base = _backend().prepare(POSITIONS, PERMANENT)
    return base, prepare_canonical_adt_separated_ddx(base, _lift())


def test_import_is_dependency_light_without_pyddx() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pyddx' or name.startswith('pyddx.'):
        raise AssertionError('pyddx imported eagerly')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from maple.solvation.continuum.separated_source_adt_ddx import PreparedCanonicalADTSeparatedDDX
print(PreparedCanonicalADTSeparatedDDX.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PreparedCanonicalADTSeparatedDDX"


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_adt_problem_data_uses_point_psi_and_finite_density_phi() -> None:
    prepared, continuum = _prepared()
    assert continuum.provider_id == CANONICAL_ADT_SEPARATED_DDX_PROVIDER_ID
    assert continuum.capabilities.enabled_tiers == ()
    data = continuum.adt_problem_data
    assert isinstance(data, CanonicalADTDDXProblemData)
    assert not data.psi_matrix.flags.writeable
    assert not data.phi_matrix.flags.writeable

    psi, phi = data.problem_data(ADT)
    external = np.zeros((len(ADT), 4))
    external[:, 1:] = ADT
    raw = external_field_to_density_order(external)
    point_psi, point_phi = prepared.point_problem_data(raw)
    np.testing.assert_allclose(psi, point_psi, rtol=0.0, atol=3.0e-13)
    expected_phi = continuum.adt_lift.tangent_potential(
        points_bohr=prepared.cavity_points_bohr,
        centers_bohr=POSITIONS / Bohr,
        atomic_dipoles_ebohr=ADT / Bohr,
    )
    np.testing.assert_allclose(phi, expected_phi, rtol=0.0, atol=3.0e-13)
    assert np.linalg.norm(phi - point_phi) > 1.0e-9


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_three_branch_solution_replays_general_problem_and_half_work() -> None:
    prepared, continuum = _prepared()
    state = continuum.solve(RADIAL4, ADT)
    assert isinstance(state, CanonicalADTSeparatedDDXState)
    assert state.half_work_identity_residual_ev == pytest.approx(0.0, abs=4.0e-12)
    assert state.energy_dual_work_ev == pytest.approx(
        2.0 * state.polarization_energy_ev, abs=4.0e-12
    )
    for values in (
        state.permanent_source4,
        state.radial_residual_source4,
        state.adt_atomic_dipoles_eangstrom,
        state.model_field8,
        state.permanent_energy_gradient4,
        state.radial_residual_energy_gradient4,
        state.adt_energy_gradient_eV_per_eangstrom,
    ):
        assert not values.flags.writeable
    with pytest.raises(ValueError, match="half-work|state_sha256"):
        replace(state, polarization_energy_ev=state.polarization_energy_ev + 1.0)

    permanent_psi, permanent_phi = prepared.bound_permanent_problem_data()
    radial_psi, radial_phi = prepared.radial_problem_data(
        embed_atomic_l1_in_first_radial_channel(RADIAL4)
    )
    adt_psi, adt_phi = continuum.adt_problem_data.problem_data(ADT)
    reference = prepared.solve_problem_data(
        permanent_psi + radial_psi + adt_psi,
        permanent_phi + radial_phi + adt_phi,
    )
    assert state.polarization_energy_ev == pytest.approx(
        reference.polarization_energy_ev, abs=2.0e-12
    )
    np.testing.assert_allclose(
        state.model_field8, reference.model_field, rtol=0.0, atol=2.0e-12
    )

    radial_only = continuum.solve(RADIAL4, np.zeros_like(ADT))
    separated = prepared.solve(embed_atomic_l1_in_first_radial_channel(RADIAL4))
    assert radial_only.polarization_energy_ev == pytest.approx(
        separated.polarization_energy_ev, abs=2.0e-12
    )
    np.testing.assert_allclose(
        radial_only.model_field8, separated.model_field, rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        radial_only.radial_residual_energy_gradient4,
        extract_atomic_l1_first_radial_cotangent(separated.energy_source_gradient),
        rtol=0.0,
        atol=2.0e-12,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_three_branch_energy_gradients_match_directional_finite_difference() -> None:
    _prepared_base, continuum = _prepared()
    state = continuum.solve(RADIAL4, ADT)
    rng = np.random.default_rng(20260818)
    radial_direction = rng.normal(scale=0.02, size=RADIAL4.shape)
    adt_direction = rng.normal(scale=0.02, size=ADT.shape)
    step = 1.0e-4
    finite_difference = (
        continuum.solve(
            RADIAL4 + step * radial_direction,
            ADT + step * adt_direction,
        ).polarization_energy_ev
        - continuum.solve(
            RADIAL4 - step * radial_direction,
            ADT - step * adt_direction,
        ).polarization_energy_ev
    ) / (2.0 * step)
    expected = float(
        np.vdot(radial_direction, state.radial_residual_energy_gradient4)
    ) + float(np.vdot(adt_direction, state.adt_energy_gradient_eV_per_eangstrom))
    assert finite_difference == pytest.approx(expected, abs=4.0e-9)


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_three_branch_model_field_jvp_vjp_and_finite_difference() -> None:
    _prepared_base, continuum = _prepared()
    rng = np.random.default_rng(71)
    radial_direction = rng.normal(scale=0.01, size=RADIAL4.shape)
    adt_direction = rng.normal(scale=0.01, size=ADT.shape)
    field_cotangent = rng.normal(scale=0.01, size=(len(ADT), 8))
    jvp = continuum.model_field_jvp(radial_direction, adt_direction)
    vjp = continuum.model_field_vjp(field_cotangent)
    assert isinstance(vjp, CanonicalADTModelFieldCotangents)
    left = float(np.vdot(field_cotangent, jvp))
    right = float(np.vdot(vjp.radial_residual_source4, radial_direction))
    right += float(np.vdot(vjp.adt_atomic_dipoles_eangstrom, adt_direction))
    assert left == pytest.approx(right, abs=1.0e-9)

    step = 1.0e-4
    plus = continuum.solve(
        RADIAL4 + step * radial_direction,
        ADT + step * adt_direction,
    ).model_field8
    minus = continuum.solve(
        RADIAL4 - step * radial_direction,
        ADT - step * adt_direction,
    ).model_field8
    np.testing.assert_allclose(
        (plus - minus) / (2.0 * step), jvp, rtol=2.0e-8, atol=8.0e-11
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_mismatched_elements_and_coordinate_derivative_fail_closed() -> None:
    prepared = _backend().prepare(POSITIONS, PERMANENT)
    with pytest.raises(ValueError, match="elements differ"):
        PreparedCanonicalADTSeparatedDDX(prepared, _lift(atomic_number=6))
    continuum = PreparedCanonicalADTSeparatedDDX(prepared, _lift())
    with pytest.raises(NotImplementedError, match="coordinate VJP"):
        continuum.coordinate_vjp(np.zeros_like(POSITIONS))
    with pytest.raises(AttributeError, match="immutable"):
        continuum.provider_id = "tampered"
