from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    MACE_POLAR_RADIAL_GTO_COUPLING_ID,
    MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
)
from maple.solvation.api.scalar_registry import (
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
)
from maple.solvation.continuum import (
    FixedHarmonicGalerkinCPCMCandidate,
    FixedHarmonicGalerkinSnapshot,
    FixedReciprocalCPCMFunctional,
    PerAtomHarmonicSpace,
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
    build_water_radial_gto_cpcm_backend,
)
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
)
from maple.solvation.coupling.variational_state import (
    VariationalCommonFunctional,
    VariationalEnvelopeGradient,
    VariationalStationaryEnergy,
    build_variational_common_functional,
)
from maple.solvation.models.field_energy import FieldEnergyFunctional, TorchGeometry
from maple.solvation.models.mace_polar_variational import (
    MACE_POLAR_VARIATIONAL_DUALITY_MAP,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class SyntheticAnchoredFieldEnergy(FieldEnergyFunctional):
    """Small same-scalar electronic oracle for the disabled integration test."""

    __slots__ = ()

    provider_id = "test.route2.scalar-first-anchored-field-energy.impl.v1"
    model_profile_id = MACE_POLAR_VARIATIONAL_EFFECTIVE_SOURCE_MODEL_PROFILE_ID
    coupling_id = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    capabilities = CapabilityStatus()
    provenance_sha256 = _sha(
        {
            "provider_id": provider_id,
            "oracle": "synthetic-same-scalar-not-chemical-evidence",
        }
    )

    def configuration_sha256(self) -> str:
        return _sha(
            {
                "provider_id": self.provider_id,
                "duality_map_sha256": self.duality_map.configuration_sha256(),
                "linear_scale": 0.003,
                "quadratic_scale": 0.012,
                "coordinate_scale": 0.004,
                "dtype": str(self._torch_dtype),
                "device": str(self._torch_device),
                "capabilities": "none",
            }
        )

    def _energy_torch(self, geometry: TorchGeometry, reduced_field):
        torch = __import__("torch")
        linear = torch.linspace(
            -0.003,
            0.003,
            reduced_field.shape[0],
            dtype=reduced_field.dtype,
            device=reduced_field.device,
        )
        coordinate_norm = torch.sum(geometry.positions**2)
        modulation = 1.0 + 0.01 * coordinate_norm
        return (
            0.004 * coordinate_norm
            + modulation * torch.dot(linear, reduced_field)
            + 0.006 * torch.dot(reduced_field, reduced_field)
        )


def _atoms() -> Atoms:
    return Atoms(
        "H2",
        positions=np.asarray([[0.05, -0.02, 0.01], [1.45, 0.18, -0.12]]),
        info={"charge": 0, "mult": 1},
    )


def _model():
    torch = pytest.importorskip("torch")
    return SyntheticAnchoredFieldEnergy(
        duality_map=MACE_POLAR_VARIATIONAL_DUALITY_MAP,
        dtype=torch.float64,
        device="cpu",
    )


def _continuum():
    torch = pytest.importorskip("torch")
    backend = build_water_radial_gto_cpcm_backend(("H", "H"))
    return FixedReciprocalCPCMFunctional(
        backend,
        dtype=torch.float64,
        device="cpu",
    )


def _harmonic_continuum():
    torch = pytest.importorskip("torch")
    space = PerAtomHarmonicSpace(atom_count=2, lmax=1)
    rng = np.random.default_rng(73)
    raw = rng.normal(scale=0.03, size=(space.dimension, space.dimension))
    surface = raw.T @ raw + 2.0 * np.eye(space.dimension)
    source = rng.normal(scale=0.02, size=(space.dimension, 16))
    snapshot = FixedHarmonicGalerkinSnapshot(
        atomic_numbers=(1, 1),
        coefficient_space=space,
        surface_operator=surface,
        source_operator=source,
        cavity_descriptor_sha256="7" * 64,
        assembly_contract_id="test-fixed-harmonic-common-state-v1",
    )
    return FixedHarmonicGalerkinCPCMCandidate(
        snapshot,
        dtype=torch.float64,
        device="cpu",
    )


def _smooth_harmonic_continuum():
    torch = pytest.importorskip("torch")
    return SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=(1, 1),
        radii_angstrom=(1.2, 1.2),
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=16,
        source_radial_quadrature_order=16,
        green_radial_quadrature_order=16,
        dtype=torch.float64,
        device="cpu",
    )


def _common(atoms: Atoms) -> VariationalCommonFunctional:
    return build_variational_common_functional(
        _model(),
        _continuum(),
        atoms,
        scalar_id=VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
        profile_id=(VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1),
    )


def _options() -> FixedPointOptions:
    return FixedPointOptions(
        method="anderson",
        tolerance=2.0e-11,
        max_iterations=80,
        damping=0.8,
        history=6,
    )


def test_variational_state_contract_imports_without_torch_or_model_runtimes():
    script = r"""
import sys
for name in ("torch", "mace", "aimnet2calc", "pyscf"):
    sys.modules[name] = None
from maple.solvation.coupling.variational_state import VariationalCommonFunctional
from maple.solvation.coupling.variational_adapters import (
    ScalarFirstElectronicResponseAdapter,
)
assert VariationalCommonFunctional.__name__ == "VariationalCommonFunctional"
assert ScalarFirstElectronicResponseAdapter.__name__.startswith("ScalarFirst")
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registered_common_scalar_builds_only_as_disabled_same_scalar_state():
    atoms = _atoms()
    common = _common(atoms)
    assert common.scalar_id == (
        VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1
    )
    assert common.capabilities.enabled_tiers == ()
    assert common.conjugacy_sign == 1
    assert common.equation.coordinates.total_charge == 0.0
    assert common.equation.source_space.component_count == 8
    assert common.equation.electronic.functional is common.model
    assert common.equation.continuum.functional is common.continuum
    assert len(common.fingerprint_sha256()) == 64

    with pytest.raises(ValueError, match="continuum profile"):
        build_variational_common_functional(
            common.model,
            common.continuum,
            atoms,
            scalar_id=(
                VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1
            ),
            profile_id=(
                VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1
            ),
        )


def test_common_state_jvp_vjp_and_coordinate_pullback_close():
    atoms = _atoms()
    common = _common(atoms)
    equation = common.equation
    rng = np.random.default_rng(20260814)
    y = rng.normal(scale=2.0e-4, size=equation.reduced_dimension)
    direction = rng.normal(size=equation.reduced_dimension)
    cotangent = rng.normal(size=equation.reduced_dimension)

    jvp = equation.jvp(atoms, y, direction)
    vjp = equation.vjp(atoms, y, cotangent)
    assert np.vdot(cotangent, jvp) == pytest.approx(np.vdot(vjp, direction), abs=2e-10)
    step = 2.0e-6
    residual_fd = (
        equation.residual(atoms, y + step * direction)
        - equation.residual(atoms, y - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(jvp, residual_fd, atol=2e-8, rtol=0.0)

    coordinate_direction = rng.normal(size=(len(atoms), 3))
    coordinate_direction /= np.linalg.norm(coordinate_direction)
    coordinate_bar = equation.coordinate_vjp(atoms, y, cotangent)
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * coordinate_direction
    minus.positions -= step * coordinate_direction
    coordinate_fd = np.vdot(
        cotangent,
        (equation.residual(plus, y) - equation.residual(minus, y)) / (2.0 * step),
    )
    assert np.vdot(coordinate_bar, coordinate_direction) == pytest.approx(
        coordinate_fd, abs=3e-7
    )


def test_harmonic_candidate_uses_the_same_disabled_common_stationarity_kernel():
    atoms = _atoms()
    common = build_variational_common_functional(
        _model(),
        _harmonic_continuum(),
        atoms,
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1
        ),
        profile_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_PROFILE_V1
        ),
    )
    state = common.solve_state(
        atoms,
        root_context_id="synthetic-fixed-harmonic-common-state-v1",
        options=_options(),
    )
    scalar = common.evaluate_state(atoms, state)
    assert state.converged
    assert scalar.residual_norm <= state.primal_tolerance
    assert common.capabilities.enabled_tiers == ()
    assert common.envelope_coordinate_gradient(atoms, state).admitted is False


def test_smooth_harmonic_candidate_closes_the_common_stationary_envelope():
    atoms = _atoms()
    common = build_variational_common_functional(
        _model(),
        _smooth_harmonic_continuum(),
        atoms,
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        profile_id=(
            VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
        ),
    )
    state = common.solve_state(
        atoms,
        root_context_id="synthetic-smooth-harmonic-common-state-v1",
        options=_options(),
    )
    assert state.converged
    envelope = common.envelope_coordinate_gradient(atoms, state)
    assert envelope.admitted is False
    direction = np.asarray([[0.3, -0.2, 0.1], [-0.3, 0.2, -0.1]])
    direction /= np.linalg.norm(direction)
    analytic = np.vdot(envelope.total_coordinate_gradient_eV_per_A, direction)
    step = 3.0e-5
    energies = []
    for sign in (1.0, -1.0):
        displaced = atoms.copy()
        displaced.positions += sign * step * direction
        displaced_state = common.solve_state(
            displaced,
            root_context_id=f"synthetic-smooth-harmonic-common-state-v1/{sign:+.0f}",
            initial_y=state.y,
            options=_options(),
        )
        assert displaced_state.converged
        energies.append(
            common.evaluate_state(displaced, displaced_state).total_energy_eV
        )
    finite_difference = (energies[0] - energies[1]) / (2.0 * step)
    assert analytic == pytest.approx(finite_difference, abs=2.0e-8)


def test_cold_warm_roots_scalar_ledger_and_stationary_envelope_close():
    atoms = _atoms()
    common = _common(atoms)
    context = "synthetic-same-scalar-h2-water-cpcm194-v1"
    cold = common.solve_state(
        atoms,
        root_context_id=context,
        options=_options(),
    )
    warm = common.solve_state(
        atoms,
        root_context_id=context,
        initial_y=np.asarray(cold.y) + 1.0e-6,
        options=_options(),
    )
    assert cold.converged and warm.converged
    assert cold.actual_unmixed_residual_norm <= cold.primal_tolerance
    assert roots_numerically_equivalent(cold, warm)

    scalar = common.evaluate_state(atoms, cold)
    assert scalar.residual_norm <= cold.primal_tolerance
    assert scalar.continuum_energy_eV == pytest.approx(
        0.5 * scalar.coupling_energy_eV, abs=2e-12
    )
    assert scalar.total_energy_eV == pytest.approx(
        scalar.model_energy_eV - scalar.continuum_energy_eV,
        abs=2e-12,
    )

    envelope = common.envelope_coordinate_gradient(atoms, cold)
    assert envelope.admitted is False
    assert np.asarray(envelope.forces_eV_per_A).shape == (len(atoms), 3)
    rng = np.random.default_rng(31)
    direction = rng.normal(size=(len(atoms), 3))
    direction /= np.linalg.norm(direction)
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    plus_state = common.solve_state(
        plus,
        root_context_id=context,
        initial_y=cold.y,
        options=_options(),
    )
    minus_state = common.solve_state(
        minus,
        root_context_id=context,
        initial_y=cold.y,
        options=_options(),
    )
    finite_difference = (
        common.evaluate_state(plus, plus_state).total_energy_eV
        - common.evaluate_state(minus, minus_state).total_energy_eV
    ) / (2.0 * step)
    analytic = np.vdot(
        np.asarray(envelope.total_coordinate_gradient_eV_per_A), direction
    )
    assert analytic == pytest.approx(finite_difference, abs=4e-7)


def test_state_replay_is_geometry_and_profile_bound_and_never_publicly_admitted():
    atoms = _atoms()
    common = _common(atoms)
    state = common.solve_state(
        atoms,
        root_context_id="variational-state-binding-v1",
        options=_options(),
    )
    moved = atoms.copy()
    moved.positions[0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="different geometry"):
        common.evaluate_state(moved, state)
    assert common.capabilities == CapabilityStatus()
    assert common.envelope_coordinate_gradient(atoms, state).admitted is False


def test_stationary_evidence_is_deeply_immutable_and_binding_drift_fails_closed():
    residual = [1.0e-12]
    scalar = VariationalStationaryEnergy(
        scalar_id=" test-scalar-v1 ",
        profile_id=" test-profile-v1 ",
        conjugacy_sign=1,
        model_energy_eV=np.float64(1.2),
        continuum_energy_eV=0.3,
        coupling_energy_eV=0.6,
        gauge_potential_eV_per_e=0.0,
        reduced_residual=residual,
    )
    model_gradient = [[1.0, 2.0, 3.0]]
    continuum_gradient = [[0.5, 0.25, -0.5]]
    total_gradient = [[1.5, 2.25, 2.5]]
    gradient = VariationalEnvelopeGradient(
        scalar=scalar,
        model_coordinate_gradient_eV_per_A=model_gradient,
        continuum_coordinate_gradient_eV_per_A=continuum_gradient,
        total_coordinate_gradient_eV_per_A=total_gradient,
    )
    residual[0] = 9.0
    model_gradient[0][0] = 99.0
    assert scalar.scalar_id == "test-scalar-v1"
    assert scalar.reduced_residual == (1.0e-12,)
    assert gradient.model_coordinate_gradient_eV_per_A == ((1.0, 2.0, 3.0),)

    common = _common(_atoms())
    object.__setattr__(common, "model", _model())
    with pytest.raises(RuntimeError, match="model/equation binding drifted"):
        common.fingerprint_sha256()
