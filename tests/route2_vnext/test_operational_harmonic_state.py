from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID,
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    get_scalar_definition,
)
from maple.solvation.continuum import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
    solve_fixed_point,
)
from maple.solvation.coupling.operational_state import (
    build_disabled_operational_electrostatic_scalar,
)
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.models import (
    ElectronicSourceState,
    ModelDomain,
    ModelProvenance,
    VacuumState,
    array_sha256,
    model_input_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
SCALAR_ID = (
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
)
PROFILE_ID = OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class _SyntheticOriginalDensityResponse:
    """Algebraic original-head oracle; never chemical or release evidence."""

    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE

    def __init__(self, atom_count: int = 2) -> None:
        dimension = atom_count * self.source_space.component_count
        response = np.zeros((dimension, dimension))
        for atom in range(atom_count):
            q1_row = atom * 8
            for other in range(atom_count):
                centering = float(atom == other) - 1.0 / atom_count
                response[q1_row, other * 8] = 0.004 * centering
                response[q1_row, other * 8 + 1] = -0.002 * centering
            for component in range(2, 5):
                response[atom * 8 + component, atom * 8 + component] = 0.003
                response[atom * 8 + component, atom * 8 + component + 3] = -0.001
        bias = np.zeros(dimension)
        bias[0] = 0.025
        bias[8] = -0.025
        self._response = response
        self._bias = bias
        self._vacuum_scale = 0.07
        self.domain = ModelDomain((1,), (0, 0), (1,))
        self.provenance = ModelProvenance(
            provider_id="test.route2.original-density-analytic-evaluator-oracle.v1",
            model_profile_id=(MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID),
            model_family="synthetic-original-density-response",
            checkpoint_sha256="1" * 64,
            upstream_version="test-only",
            upstream_commit="test-only",
            inference_code_sha256="2" * 64,
            dtype="float64",
            device="cpu",
            domain=self.domain,
            field_convention=self.field_space.field_convention,
            coordinate_frame_policy="laboratory Cartesian Angstrom",
            optimizer_parameter_groups_audited=False,
            optimizer_audit_evidence_sha256=None,
        )
        self.provider_id = self.provenance.provider_id
        self.model_profile_id = self.provenance.model_profile_id
        self.provenance_sha256 = self.provenance.sha256
        self.dtype = self.provenance.dtype
        self.device = self.provenance.device
        self.field_convention = self.provenance.field_convention
        self.coordinate_frame_policy = self.provenance.coordinate_frame_policy
        self._configuration = _sha(
            {
                "provider_id": self.provider_id,
                "response": response.tolist(),
                "bias": bias.tolist(),
                "vacuum_scale": self._vacuum_scale,
                "oracle": "synthetic-not-chemical-evidence",
            }
        )

    def configuration_sha256(self) -> str:
        return self._configuration

    def evaluate_vacuum(self, atoms, *, need_forces: bool) -> VacuumState:
        positions = np.asarray(atoms.get_positions(), dtype=float)
        displacement = positions[1] - positions[0]
        energy = float(0.5 * self._vacuum_scale * np.vdot(displacement, displacement))
        forces = None
        if need_forces:
            force = self._vacuum_scale * displacement
            forces = np.asarray((force, -force), dtype=float)
        return VacuumState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            len(atoms),
            energy,
            need_forces,
            forces,
        )

    def evaluate_source(
        self, atoms, field, *, need_fixed_field_forces: bool
    ) -> ElectronicSourceState:
        values = self.field_space.validate(field, atom_count=len(atoms))
        source = (self._bias + self._response @ values.reshape(-1)).reshape(
            len(atoms), -1
        )
        fixed_field_forces = (
            np.zeros((len(atoms), 3)) if need_fixed_field_forces else None
        )
        return ElectronicSourceState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            array_sha256(values, name="field"),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            len(atoms),
            source,
            need_fixed_field_forces,
            fixed_field_forces,
        )

    def source_jvp(self, atoms, field, field_direction):
        del field
        direction = self.field_space.validate(field_direction, atom_count=len(atoms))
        return (self._response @ direction.reshape(-1)).reshape(len(atoms), -1)

    def source_vjp(self, atoms, field, source_cotangent):
        del field
        cotangent = self.source_space.validate(source_cotangent, atom_count=len(atoms))
        return (self._response.T @ cotangent.reshape(-1)).reshape(len(atoms), -1)

    def source_position_vjp(self, atoms, field, source_cotangent):
        self.field_space.validate(field, atom_count=len(atoms))
        self.source_space.validate(source_cotangent, atom_count=len(atoms))
        return np.zeros((len(atoms), 3))


def _atoms() -> Atoms:
    return Atoms(
        "H2",
        positions=np.asarray([[0.05, -0.02, 0.01], [1.45, 0.18, -0.12]]),
        info={"charge": 0, "mult": 1},
    )


def _continuum(*, scalar_id: str = SCALAR_ID):
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
        scalar_id=scalar_id,
    )


def _scalar(atoms: Atoms):
    return build_disabled_operational_electrostatic_scalar(
        _SyntheticOriginalDensityResponse(),
        _continuum(),
        atoms,
        scalar_id=SCALAR_ID,
        profile_id=PROFILE_ID,
    )


def _solve(scalar, atoms: Atoms, *, initial_y=None, suffix: str):
    return solve_fixed_point(
        scalar.equation,
        atoms,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id=f"synthetic-operational-harmonic/{suffix}",
        initial_y=initial_y,
        options=FixedPointOptions(
            method="anderson",
            tolerance=2.0e-12,
            max_iterations=80,
            damping=0.8,
            history=6,
        ),
    )


def test_operational_builder_imports_without_torch_or_model_runtimes():
    script = r"""
import sys
for name in ("torch", "mace", "aimnet2calc", "pyscf"):
    sys.modules[name] = None
from maple.solvation.coupling.operational_state import (
    build_disabled_operational_electrostatic_scalar,
)
assert callable(build_disabled_operational_electrostatic_scalar)
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registered_operational_harmonic_scalar_is_unique_and_fully_closed():
    definition = get_scalar_definition(SCALAR_ID)
    profile = get_solvation_profile(PROFILE_ID)
    assert profile.scalar_id == definition.scalar_id == SCALAR_ID
    assert profile.enabled is definition.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    assert definition.admitted_capabilities.enabled_tiers == ()
    assert "field_conditioned_model_energy_difference" in (
        definition.excluded_components
    )
    assert "original four-channel MACE-POLAR density head" in (
        definition.source_representation
    )


def test_original_source_harmonic_root_and_implicit_gradient_close_one_scalar():
    atoms = _atoms()
    scalar = _scalar(atoms)
    cold = _solve(scalar, atoms, suffix="base")
    warm = _solve(
        scalar,
        atoms,
        initial_y=np.asarray(cold.y) + 1.0e-7,
        suffix="base",
    )
    assert roots_numerically_equivalent(cold, warm)
    assert cold.actual_unmixed_residual_norm <= cold.primal_tolerance
    assert MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.total_charge(
        cold.source, atom_count=len(atoms)
    ) == pytest.approx(0.0, abs=2.0e-13)
    source = cold.source_array()
    np.testing.assert_allclose(source[:, (1, 5, 6, 7)], 0.0, atol=2.0e-13)

    energy = scalar.evaluate_energy_components(atoms, cold.y)
    functional = scalar.equation.continuum.functional
    assert energy.continuum_energy == pytest.approx(
        functional.energy_eV(atoms, source), abs=2.0e-12
    )
    assert energy.total_energy == pytest.approx(
        energy.vacuum_energy + energy.continuum_energy, abs=1.0e-14
    )

    gradient = scalar.implicit_gradient(atoms, cold)
    assert gradient.adjoint.converged
    direction = np.asarray([[0.21, -0.13, 0.08], [-0.21, 0.13, -0.08]])
    direction /= np.linalg.norm(direction)
    analytic = float(
        np.vdot(
            np.asarray(gradient.total_coordinate_gradient).reshape(len(atoms), 3),
            direction,
        )
    )
    step = 2.0e-5
    energies = []
    for sign in (1.0, -1.0):
        displaced = atoms.copy()
        displaced.positions += sign * step * direction
        state = _solve(
            scalar,
            displaced,
            initial_y=np.asarray(cold.y),
            suffix=f"displaced-{sign:+.0f}",
        )
        assert state.actual_unmixed_residual_norm <= state.primal_tolerance
        energies.append(scalar.evaluate_energy(displaced, state.y))
    finite_difference = (energies[0] - energies[1]) / (2.0 * step)
    assert analytic == pytest.approx(finite_difference, abs=3.0e-8)


def test_operational_builder_rejects_a_different_continuum_scalar_identity():
    atoms = _atoms()
    with pytest.raises(ValueError, match="Continuum scalar ID"):
        build_disabled_operational_electrostatic_scalar(
            _SyntheticOriginalDensityResponse(),
            _continuum(
                scalar_id=(
                    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
                )
            ),
            atoms,
            scalar_id=SCALAR_ID,
            profile_id=PROFILE_ID,
        )
