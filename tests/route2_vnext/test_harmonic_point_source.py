from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.api import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1,
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_PROFILE_V1,
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
    PROFILE_REGISTRY,
    SCALAR_REGISTRY,
)
from maple.solvation.api.profiles import AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID
from maple.solvation.continuum import (
    COULOMB_EV_ANGSTROM_PER_E2,
    SmoothPointChargeHarmonicDDPCMFunctionalCandidate,
    SmoothPointChargeHarmonicGalerkinFunctionalCandidate,
    point_l0_harmonic_source_operator,
    point_monopole_harmonic_coefficients,
    point_source_topology,
)
from maple.solvation.continuum.harmonic_coefficients import (
    PerAtomHarmonicSpace,
    _real_harmonic_design,
)
from maple.solvation.coupling.geometry_mediated import (
    GeometryMediatedElectrostaticScalar,
)
from maple.solvation.models import (
    AIMNet2CheckpointContract,
    AIMNet2GeometryMediatedModelAdapter,
)

POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.72, -0.31, 0.21], [-0.39, 1.65, -0.17]])
NUMBERS = (8, 1, 6)
RADII = (1.43, 1.18, 1.57)
SOURCE = np.asarray(
    [[0.20, 0.0, 0.0, 0.0], [-0.14, 0.0, 0.0, 0.0], [-0.06, 0.0, 0.0, 0.0]]
)


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


@pytest.fixture(scope="module")
def functional():
    torch = pytest.importorskip("torch")
    return SmoothPointChargeHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=32,
        green_radial_quadrature_order=32,
        dtype=torch.float64,
        device="cpu",
    )


def _direct_projection(
    displacement: np.ndarray, *, target_radius: float, lmax: int
) -> np.ndarray:
    cosine, polar_weights = np.polynomial.legendre.leggauss(240)
    azimuthal_count = 320
    azimuth = 2.0 * np.pi * np.arange(azimuthal_count) / azimuthal_count
    sine = np.sqrt(1.0 - cosine**2)
    directions = np.stack(
        (
            np.repeat(sine, azimuthal_count) * np.tile(np.cos(azimuth), len(cosine)),
            np.repeat(sine, azimuthal_count) * np.tile(np.sin(azimuth), len(cosine)),
            np.repeat(cosine, azimuthal_count),
        ),
        axis=1,
    )
    weights = np.repeat(
        polar_weights * (2.0 * np.pi / azimuthal_count), azimuthal_count
    )
    design = _real_harmonic_design(directions, lmax=lmax)
    potential = COULOMB_EV_ANGSTROM_PER_E2 / np.linalg.norm(
        target_radius * directions - displacement, axis=1
    )
    return design.T @ (weights * potential)


@pytest.mark.parametrize(
    "displacement",
    (np.asarray([1.7, -0.4, 0.3]), np.asarray([0.5, 0.2, -0.1])),
)
def test_point_addition_theorem_matches_independent_surface_projection(displacement):
    actual = point_monopole_harmonic_coefficients(
        displacement, target_radius_angstrom=1.2, lmax=4
    )
    expected = _direct_projection(displacement, target_radius=1.2, lmax=4)
    np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=0.0)


def test_point_source_operator_is_an_exact_so3_intertwiner_for_monopoles():
    rotation = _rotation(20260815)
    base = point_l0_harmonic_source_operator(
        positions_angstrom=POSITIONS, radii_angstrom=RADII, lmax=3
    )
    rotated = point_l0_harmonic_source_operator(
        positions_angstrom=POSITIONS @ rotation.T,
        radii_angstrom=RADII,
        lmax=3,
    )
    harmonic_rotation = PerAtomHarmonicSpace(
        atom_count=len(POSITIONS), lmax=3
    ).representation_matrix(rotation)
    charge_columns = np.arange(len(POSITIONS)) * 4
    np.testing.assert_allclose(
        rotated[:, charge_columns],
        harmonic_rotation @ base[:, charge_columns],
        atol=2.0e-12,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        base[:, np.setdiff1d(np.arange(base.shape[1]), charge_columns)], 0.0
    )


def test_point_source_topology_is_rigid_invariant_and_fails_on_shell_event():
    rotation = _rotation(73)
    base = point_source_topology(POSITIONS, RADII)
    rotated = point_source_topology(POSITIONS @ rotation.T, RADII)
    translated = point_source_topology(POSITIONS + np.asarray([0.4, -0.2, 0.8]), RADII)
    assert rotated.topology_sha256 == base.topology_sha256
    assert translated.topology_sha256 == base.topology_sha256
    assert base.minimum_shell_margin_angstrom is not None
    assert base.minimum_shell_margin_angstrom > 0.1

    with pytest.raises(ValueError, match="event surface"):
        point_source_topology(
            np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
            (1.2, 1.0),
        )


def test_torch_point_source_and_stationary_matrices_match_numpy_reference(functional):
    matrices = functional.debug_geometry_matrices(POSITIONS)
    raw_source = point_l0_harmonic_source_operator(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        lmax=functional.physical_lmax,
    )
    np.testing.assert_allclose(
        matrices["raw_source"], raw_source, atol=2.0e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        matrices["source_operator"],
        matrices["weighted_basis"].T @ raw_source,
        atol=3.0e-13,
        rtol=0.0,
    )
    right_hand_side = matrices["source_operator"] @ SOURCE.reshape(-1)
    expected = (
        -0.5
        * right_hand_side
        @ np.linalg.solve(matrices["surface_operator"], right_hand_side)
    )
    assert functional.energy_eV(POSITIONS, SOURCE) == pytest.approx(
        expected, abs=2.0e-14
    )
    stationarity = functional.stationarity_audit(POSITIONS, SOURCE)
    assert stationarity["gate_passed"] is True
    assert stationarity["capability_admitted"] is False
    assert stationarity["state_dimension"] == 12
    assert stationarity["relative_residual"] <= 1.0e-10


def test_same_scalar_generates_reciprocal_source_and_coordinate_derivatives(functional):
    rng = np.random.default_rng(91)
    source_direction = np.zeros_like(SOURCE)
    source_direction[:, 0] = rng.normal(size=len(SOURCE))
    source_direction[:, 0] -= np.mean(source_direction[:, 0])
    field_cotangent = np.zeros_like(SOURCE)
    field_cotangent[:, 0] = rng.normal(size=len(SOURCE))
    coordinate_direction = rng.normal(size=POSITIONS.shape)
    coordinate_direction -= np.mean(coordinate_direction, axis=0)
    coordinate_direction /= np.linalg.norm(coordinate_direction)

    source_step = 2.0e-6
    source_fd = (
        functional.energy_eV(POSITIONS, SOURCE + source_step * source_direction)
        - functional.energy_eV(POSITIONS, SOURCE - source_step * source_direction)
    ) / (2.0 * source_step)
    assert np.vdot(functional.drive(POSITIONS, SOURCE), source_direction) == (
        pytest.approx(source_fd, abs=2.0e-11)
    )
    source_jvp = functional.source_jvp(POSITIONS, SOURCE, source_direction)
    source_vjp = functional.source_vjp(POSITIONS, SOURCE, field_cotangent)
    assert np.vdot(field_cotangent, source_jvp) == pytest.approx(
        np.vdot(source_vjp, source_direction), abs=3.0e-13
    )

    gradient = functional.coordinate_partial(POSITIONS, SOURCE)
    analytic = float(np.vdot(gradient, coordinate_direction))
    errors = []
    for step in (1.0e-3, 1.0e-4, 1.0e-5):
        finite = (
            functional.energy_eV(POSITIONS + step * coordinate_direction, SOURCE)
            - functional.energy_eV(POSITIONS - step * coordinate_direction, SOURCE)
        ) / (2.0 * step)
        errors.append(abs(analytic - finite))
    assert errors[-1] < 2.0e-10
    assert errors[-1] < errors[0] / 20.0

    mixed = functional.coordinate_vjp(POSITIONS, SOURCE, field_cotangent).reshape(
        POSITIONS.shape
    )
    mixed_step = 1.0e-5
    finite_mixed = (
        np.vdot(
            field_cotangent,
            functional.evaluate_field(
                POSITIONS + mixed_step * coordinate_direction, SOURCE
            ),
        )
        - np.vdot(
            field_cotangent,
            functional.evaluate_field(
                POSITIONS - mixed_step * coordinate_direction, SOURCE
            ),
        )
    ) / (2.0 * mixed_step)
    assert np.vdot(mixed, coordinate_direction) == pytest.approx(
        finite_mixed, abs=1.0e-9
    )


def test_point_scalar_is_rotation_translation_and_permutation_covariant(functional):
    torch = pytest.importorskip("torch")
    rotation = _rotation(907)
    energy = functional.energy_eV(POSITIONS, SOURCE)
    gradient = functional.coordinate_partial(POSITIONS, SOURCE)
    rotated_positions = POSITIONS @ rotation.T
    assert functional.energy_eV(rotated_positions, SOURCE) == pytest.approx(
        energy, abs=8.0e-14
    )
    np.testing.assert_allclose(
        functional.coordinate_partial(rotated_positions, SOURCE),
        gradient @ rotation.T,
        atol=3.0e-13,
        rtol=0.0,
    )
    shift = np.asarray([0.7, -0.2, 0.4])
    assert functional.energy_eV(POSITIONS + shift, SOURCE) == pytest.approx(
        energy, abs=8.0e-14
    )
    np.testing.assert_allclose(
        functional.coordinate_partial(POSITIONS + shift, SOURCE),
        gradient,
        atol=3.0e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.sum(gradient, axis=0), 0.0, atol=8.0e-14)
    centered = POSITIONS - np.mean(POSITIONS, axis=0)
    np.testing.assert_allclose(
        np.sum(np.cross(centered, -gradient), axis=0), 0.0, atol=2.0e-13
    )

    permutation = np.asarray([2, 0, 1])
    permuted = SmoothPointChargeHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=tuple(NUMBERS[index] for index in permutation),
        radii_angstrom=tuple(RADII[index] for index in permutation),
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=32,
        green_radial_quadrature_order=32,
        dtype=torch.float64,
        device="cpu",
    )
    assert permuted.energy_eV(
        POSITIONS[permutation], SOURCE[permutation]
    ) == pytest.approx(energy, abs=8.0e-14)
    np.testing.assert_allclose(
        permuted.coordinate_partial(POSITIONS[permutation], SOURCE[permutation]),
        gradient[permutation],
        atol=3.0e-13,
        rtol=0.0,
    )


class _FakeAIMNet2:
    model_name = "aimnet2"
    _supports_atom_charges = True
    _coulomb_method = "simple"
    solvent_correction = None
    device = "cpu"
    cutoff = 5.0
    cutoff_lr = float("inf")

    def __init__(self, model_path: Path):
        self.model_path = str(model_path)

    @staticmethod
    def _charges(atoms: Atoms) -> np.ndarray:
        x = np.asarray(atoms.positions)[:, 0]
        base = np.linspace(-0.3, 0.3, len(atoms))
        base -= np.mean(base)
        return base + 0.07 * (x - np.mean(x))

    def charge_state(self, atoms: Atoms):
        return SimpleNamespace(
            energy_ev=0.5 * float(np.vdot(atoms.positions, atoms.positions)),
            charges_e=self._charges(atoms),
            requested_total_charge_e=0.0,
            model_name="aimnet2",
        )

    def charge_position_response(self, atoms: Atoms, charge_cotangent_ev_per_e):
        cotangent = np.asarray(charge_cotangent_ev_per_e, dtype=float)
        vjp = np.zeros_like(atoms.positions)
        vjp[:, 0] = 0.07 * (cotangent - np.mean(cotangent))
        return SimpleNamespace(
            charge_state=self.charge_state(atoms),
            charge_cotangent_ev_per_e=cotangent.copy(),
            intrinsic_energy_gradient_ev_per_angstrom=np.array(
                atoms.positions, copy=True
            ),
            charge_position_vjp_ev_per_angstrom=vjp,
        )

    def charge_position_second_order(
        self,
        atoms: Atoms,
        charge_cotangent_ev_per_e,
        coordinate_direction,
    ):
        response = self.charge_position_response(atoms, charge_cotangent_ev_per_e)
        direction = np.asarray(coordinate_direction, dtype=float)
        charge_jvp = 0.07 * (direction[:, 0] - np.mean(direction[:, 0]))
        return SimpleNamespace(
            **vars(response),
            coordinate_direction=np.array(direction, copy=True),
            charge_position_jvp_e_per_angstrom=charge_jvp,
            intrinsic_energy_hvp_ev_per_angstrom2=np.array(direction, copy=True),
            contracted_charge_hessian_ev_per_angstrom2=np.zeros_like(direction),
            standard_decomposed_energy_absolute_error_ev=0.0,
            standard_decomposed_charge_max_absolute_error_e=0.0,
            standard_decomposed_intrinsic_gradient_max_absolute_error_ev_per_angstrom=0.0,
            standard_decomposed_charge_vjp_max_absolute_error_ev_per_angstrom=0.0,
            charge_tangent_residual_e_per_angstrom=abs(float(np.sum(charge_jvp))),
        )


def _model(tmp_path: Path) -> AIMNet2GeometryMediatedModelAdapter:
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"point harmonic synthetic checkpoint\n")
    contract = AIMNet2CheckpointContract(
        provider_id="maple.route2.model.test-point-harmonic-aimnet2.impl.v1",
        model_profile_id=AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID,
        checkpoint_identifier="test",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        model_name="aimnet2",
        coulomb_method="simple",
        inference_dtype="float32",
        supported_atomic_numbers=(1, 6, 7, 8),
        upstream_repository="test",
        publication_doi="10.1039/D4SC08572H",
        upstream_version="test",
        upstream_commit="test",
        checkpoint_origin_status="test",
    )
    return AIMNet2GeometryMediatedModelAdapter(_FakeAIMNet2(checkpoint), contract)


def test_registered_geometry_mediated_point_harmonic_scalar_closes_full_gradient(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    atoms = Atoms(
        "OHC",
        positions=POSITIONS,
        info={"charge": 0, "mult": 1},
    )
    continuum = SmoothPointChargeHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=24,
        green_radial_quadrature_order=24,
        dtype=torch.float64,
        device="cpu",
    )
    scalar = GeometryMediatedElectrostaticScalar(
        _model(tmp_path),
        continuum,
        scalar_id=(
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1
        ),
        profile_id=(
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_PROFILE_V1
        ),
    )
    result = scalar.evaluate(atoms)
    step = 1.0e-5
    numerical = np.zeros_like(atoms.positions)
    for atom in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            numerical[atom, axis] = (
                scalar.evaluate_energy(plus) - scalar.evaluate_energy(minus)
            ) / (2.0 * step)
    np.testing.assert_allclose(
        result.total_gradient_eV_per_A, numerical, atol=4.0e-8, rtol=2.0e-7
    )
    assert result.reciprocity_audit.gate_passed is True
    assert np.linalg.norm(result.source_response_gradient_eV_per_A) > 0.0
    assert np.linalg.norm(result.continuum_fixed_source_gradient_eV_per_A) > 0.0
    assert continuum.topology_state(atoms)["coefficient_count"] == 12

    profile = PROFILE_REGISTRY[
        DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_PROFILE_V1
    ]
    definition = SCALAR_REGISTRY[profile.scalar_id]
    assert profile.enabled is definition.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    assert definition.admitted_capabilities.enabled_tiers == ()
    assert "finite_dielectric_solvent_parameterization" in (
        definition.excluded_components
    )
    assert continuum.fixed_geometry_electronic_mutual_polarization is False


def test_registered_geometry_mediated_harmonic_ddpcm_closes_full_gradient(tmp_path):
    torch = pytest.importorskip("torch")
    atoms = Atoms(
        "OHC",
        positions=POSITIONS,
        info={"charge": 0, "mult": 1},
    )
    continuum = SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=RADII,
        dielectric=78.39,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=24,
        green_radial_quadrature_order=24,
        dtype=torch.float64,
        device="cpu",
    )
    scalar = GeometryMediatedElectrostaticScalar(
        _model(tmp_path),
        continuum,
        scalar_id=(
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
        ),
        profile_id=(
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
        ),
    )
    result = scalar.evaluate(atoms)
    direction = np.asarray(
        [[0.21, -0.07, 0.11], [-0.13, 0.05, -0.17], [-0.08, 0.02, 0.06]]
    )
    direction /= np.linalg.norm(direction)
    analytic = float(np.vdot(result.total_gradient_eV_per_A, direction))
    errors = []
    for step in (1.0e-3, 1.0e-4, 1.0e-5):
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        finite = (scalar.evaluate_energy(plus) - scalar.evaluate_energy(minus)) / (
            2.0 * step
        )
        errors.append(abs(analytic - finite))
    assert errors[-1] < 3.0e-8
    assert errors[-1] < errors[0] / 20.0
    assert result.reciprocity_audit.gate_passed is True
    assert continuum.stationarity_audit(atoms, result.source)["gate_passed"] is True

    hvp = scalar.hessian_vector_product(atoms, direction)
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    finite_hvp = (
        scalar.evaluate(plus).total_gradient_eV_per_A
        - scalar.evaluate(minus).total_gradient_eV_per_A
    ) / (2.0 * step)
    np.testing.assert_allclose(
        hvp.total_hvp_eV_per_A2, finite_hvp, atol=5.0e-7, rtol=3.0e-6
    )
    second_direction = np.asarray(
        [[-0.11, 0.08, 0.04], [0.16, -0.03, 0.07], [-0.05, -0.05, -0.11]]
    )
    second_direction /= np.linalg.norm(second_direction)
    second_hvp = scalar.hessian_vector_product(
        atoms, second_direction
    ).total_hvp_eV_per_A2
    assert np.vdot(direction, second_hvp) == pytest.approx(
        np.vdot(second_direction, hvp.total_hvp_eV_per_A2), abs=2.0e-8
    )
    assert hvp.tier_h_admitted is False
    assert continuum.fixed_geometry_electronic_mutual_polarization is False
    profile = PROFILE_REGISTRY[
        DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
    ]
    definition = SCALAR_REGISTRY[profile.scalar_id]
    assert profile.enabled is definition.enabled is False
    assert profile.capabilities.enabled_tiers == ()
