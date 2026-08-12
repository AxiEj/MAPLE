from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest
from ase.units import Bohr, Hartree as ASE_HARTREE_TO_EV

from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
)
from maple.solvation.api.scalar_registry import (
    OPERATIONAL_CPCM_ELECTROSTATIC_V1,
)
from maple.solvation.api.profiles import (
    EXACT_GTO_COUPLING_CANDIDATE_ID,
    LOCAL_JET_DIAGNOSTIC_COUPLING_ID,
    OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
    PROFILE_REGISTRY,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.exact_gto import (
    EXACT_GTO_COUPLING_PROFILE_ID,
    ExactGTOCouplingAdapter,
    FixedSurfaceGeometry,
    SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID,
    SingleWidthSameBasisGTOCouplingCandidate,
)
from maple.solvation.coupling.local_jet import (
    LOCAL_JET_COUPLING_ID,
    LOCAL_JET_DIAGNOSTIC_PROFILE_ID,
    LocalJetDiagnosticCouplingAdapter,
)
from maple.solvation.coupling.operator import (
    ConjugateSurfaceMap,
    CoordinateDerivativeUnavailable,
    validate_adjoint_dot_product,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
)


def _geometry() -> FixedSurfaceGeometry:
    return FixedSurfaceGeometry(
        np.asarray([[-0.7, 0.2, 0.3], [0.8, -0.4, 0.1]], dtype=float),
        np.asarray(
            [
                [4.2, 1.1, -0.8],
                [-3.7, 2.4, 1.5],
                [1.6, -4.3, 2.2],
                [-2.1, -2.8, -3.2],
                [3.5, 3.1, 2.7],
            ],
            dtype=float,
        ),
    )


def _reference_geometry() -> FixedSurfaceGeometry:
    return FixedSurfaceGeometry(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, -2.0, 2.5]]),
    )


def _source() -> np.ndarray:
    return np.asarray(
        [[0.35, -0.13, 0.21, 0.08], [-0.18, 0.11, -0.07, 0.16]],
        dtype=float,
    )


def _rotation() -> np.ndarray:
    axis = np.asarray([1.0, -2.0, 3.0], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _rotate_source(source: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    external = ATOMIC_L1_FIELD_DUAL_SPACE.pairing_metric.source_to_field_dual(source)
    external[:, 1:] = external[:, 1:] @ rotation.T
    return ATOMIC_L1_FIELD_DUAL_SPACE.pairing_metric.field_to_source_dual(external)


@pytest.mark.parametrize(
    "adapter", [ExactGTOCouplingAdapter(), LocalJetDiagnosticCouplingAdapter()]
)
def test_adapters_use_only_canonical_spaces_q_and_operational_scalar(adapter):
    assert adapter.source_space is ATOMIC_L1_SOURCE_SPACE
    assert adapter.field_space is ATOMIC_L1_FIELD_DUAL_SPACE
    assert adapter.pairing_metric is ATOMIC_L1_FIELD_DUAL_SPACE.pairing_metric
    assert adapter.scalar_id == OPERATIONAL_CPCM_ELECTROSTATIC_V1
    assert adapter.coupling_id != adapter.scalar_id
    assert adapter.capabilities.enabled_tiers == ()


def test_local_diagnostic_profile_is_explicitly_nonpes_unregistered_disabled():
    adapter = LocalJetDiagnosticCouplingAdapter()
    assert adapter.coupling_id == LOCAL_JET_COUPLING_ID
    assert adapter.diagnostic_profile_id == LOCAL_JET_DIAGNOSTIC_PROFILE_ID
    assert adapter.diagnostic_profile_id not in PROFILE_REGISTRY
    assert adapter.diagnostic_profile_registered_as_pes is False
    assert adapter.diagnostic_profile_enabled is False
    profile = PROFILE_REGISTRY[OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1]
    assert profile.coupling_id == EXACT_GTO_COUPLING_CANDIDATE_ID
    assert adapter.coupling_id == LOCAL_JET_DIAGNOSTIC_COUPLING_ID
    assert adapter.coupling_id != profile.coupling_id


@pytest.mark.parametrize(
    "adapter", [ExactGTOCouplingAdapter(), LocalJetDiagnosticCouplingAdapter()]
)
def test_b_and_bstar_are_one_conjugate_operator(adapter):
    validation = validate_adjoint_dot_product(
        adapter,
        _geometry(),
        _source(),
        np.asarray([0.2, -0.4, 0.1, 0.3, -0.2]),
        atom_count=2,
        relative_tolerance=2.0e-13,
        absolute_tolerance=2.0e-12,
    )
    assert validation.passed
    assert validation.absolute_error <= validation.tolerance


@pytest.mark.parametrize(
    "adapter", [ExactGTOCouplingAdapter(), LocalJetDiagnosticCouplingAdapter()]
)
def test_source_jvp_and_vjp_match_independent_dense_differences(adapter):
    geometry = _geometry()
    source = _source()
    direction = np.random.default_rng(11).normal(size=source.shape)
    cotangent = np.random.default_rng(12).normal(size=5)
    step = 1.0e-6
    finite_difference = (
        adapter.apply_source(geometry, source + step * direction)
        - adapter.apply_source(geometry, source - step * direction)
    ) / (2.0 * step)
    jvp = adapter.source_jvp(geometry, source, direction)
    raw_vjp = adapter.pairing_metric.field_to_source_dual(
        adapter.source_vjp(geometry, source, cotangent)
    )
    np.testing.assert_allclose(jvp, finite_difference, rtol=2.0e-10, atol=2.0e-9)
    assert np.vdot(jvp, cotangent) == pytest.approx(
        np.vdot(direction, raw_vjp), rel=2.0e-13, abs=2.0e-12
    )


def test_maple_versioned_unit_factor_and_fixed_numerical_references():
    assert HARTREE_TO_EV == 27.211386245988
    assert HARTREE_TO_EV != ASE_HARTREE_TO_EV
    geometry = _reference_geometry()
    source = np.asarray([[0.4, 0.1, -0.2, 0.3]])
    exact = SingleWidthSameBasisGTOCouplingCandidate(sigma_angstrom=1.5)
    local = LocalJetDiagnosticCouplingAdapter()
    np.testing.assert_allclose(
        exact.apply_source(geometry, source),
        [3.1384723788443316, 2.7065392876391865, 2.3164888959044063],
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        local.apply_source(geometry, source),
        [9.298932312393838, 4.199541138457102, 2.700041445195944],
        rtol=0.0,
        atol=2.0e-13,
    )


def test_adapters_are_thin_maple_unit_conversions_of_legacy_kernels():
    geometry = _geometry()
    source = _source()
    exact_expected = (
        AtomCenteredL1GTOBasis((1.5,)).surface_operator(
            geometry.surface_points_bohr, geometry.atom_positions_angstrom
        )
        @ source.reshape(-1)
    ) * HARTREE_TO_EV
    local_expected = (
        point_multipole_potential(
            geometry.surface_points_bohr, geometry.atom_positions_angstrom, source
        )
        * HARTREE_TO_EV
    )
    np.testing.assert_allclose(
        ExactGTOCouplingAdapter().apply_source(geometry, source),
        exact_expected,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        LocalJetDiagnosticCouplingAdapter().apply_source(geometry, source),
        local_expected,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_configuration_and_provenance_hashes_cover_basis_q_units_and_width():
    narrow = SingleWidthSameBasisGTOCouplingCandidate(sigma_angstrom=1.25)
    wide = SingleWidthSameBasisGTOCouplingCandidate(sigma_angstrom=1.50)
    for adapter in (narrow, wide, LocalJetDiagnosticCouplingAdapter()):
        metadata = adapter.provenance.metadata()
        configuration = metadata["configuration"]
        assert re.fullmatch(r"[0-9a-f]{64}", adapter.configuration_sha256)
        assert re.fullmatch(r"[0-9a-f]{64}", adapter.provenance_sha256)
        assert (
            configuration["pairing_q_sha256"] == adapter.pairing_metric.metadata_hash()
        )
        assert configuration["hartree_to_ev"] == format(HARTREE_TO_EV, ".17g")
        assert (
            configuration["algorithm_contract_sha256"]
            == metadata["algorithm_contract_sha256"]
        )
        assert (
            configuration["source_bundle_sha256"]
            == hashlib.sha256(
                json.dumps(
                    metadata["source_files_sha256"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        assert (
            configuration["implementation_version"]
            == metadata["implementation_version"]
        )
        assert metadata["tested_git_commit"] is None
        assert "unbound" in metadata["release_binding_status"]
        assert all(not name.startswith("/") for name in metadata["source_files_sha256"])
    assert narrow.configuration_sha256 != wide.configuration_sha256
    assert narrow.provenance_sha256 != wide.provenance_sha256


def test_provenance_binds_current_executable_source_dependencies():
    root = Path(__file__).parents[2]
    exact = SingleWidthSameBasisGTOCouplingCandidate().provenance.metadata()
    local = LocalJetDiagnosticCouplingAdapter().provenance.metadata()
    expected_paths = {
        "maple.solvation.coupling.operator": (
            root / "maple/solvation/coupling/operator.py"
        ),
        "maple.solvation.coupling.spaces": (
            root / "maple/solvation/coupling/spaces.py"
        ),
        "maple.solvation.api.units": root / "maple/solvation/api/units.py",
        "maple.legacy.gto_density": (
            root / "maple/function/calculator/extra_correction/implicit/gto_density.py"
        ),
    }
    for metadata in (exact, local):
        source_hashes = metadata["source_files_sha256"]
        for logical_name, path in expected_paths.items():
            assert (
                source_hashes[logical_name]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )


def test_candidate_name_and_metadata_do_not_claim_checkpoint_native_precision():
    adapter = ExactGTOCouplingAdapter()
    assert adapter.coupling_id == SINGLE_WIDTH_SAME_BASIS_GTO_COUPLING_ID
    assert EXACT_GTO_COUPLING_PROFILE_ID == adapter.coupling_id
    assert adapter.checkpoint_native is False
    assert adapter.precision_capability_blocked is True
    assert (
        "not verified against checkpoint-native" in adapter.provenance.basis_definition
    )
    assert "precision/E/F/H/V/M all blocked" in adapter.provenance.production_status


@pytest.mark.parametrize(
    "adapter", [ExactGTOCouplingAdapter(), LocalJetDiagnosticCouplingAdapter()]
)
def test_public_total_coordinate_vjp_fails_closed(adapter):
    assert adapter.coordinate_derivative_available is False
    with pytest.raises(CoordinateDerivativeUnavailable, match="no admitted total"):
        adapter.coordinate_vjp(_geometry(), _source(), np.ones(5))


def test_explicit_local_partial_kernel_vjp_matches_fixed_node_difference():
    geometry = _geometry()
    adapter = LocalJetDiagnosticCouplingAdapter()
    source = _source()
    cotangent = np.asarray([0.2, -0.4, 0.1, 0.3, -0.2])
    analytic = adapter.partial_fixed_surface_kernel_coordinate_vjp(
        geometry, source, cotangent
    )
    step = 2.0e-6
    finite_difference = np.empty_like(geometry.atom_positions_angstrom)
    for atom in range(2):
        for axis in range(3):
            plus = geometry.atom_positions_angstrom.copy()
            minus = geometry.atom_positions_angstrom.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            plus_value = np.vdot(
                adapter.apply_source(
                    FixedSurfaceGeometry(plus, geometry.surface_points_bohr), source
                ),
                cotangent,
            )
            minus_value = np.vdot(
                adapter.apply_source(
                    FixedSurfaceGeometry(minus, geometry.surface_points_bohr), source
                ),
                cotangent,
            )
            finite_difference[atom, axis] = (plus_value - minus_value) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-8, atol=2.0e-8)


def test_local_partial_kernel_vjp_is_rotation_covariant():
    geometry = _geometry()
    adapter = LocalJetDiagnosticCouplingAdapter()
    source = _source()
    cotangent = np.asarray([0.2, -0.4, 0.1, 0.3, -0.2])
    rotation = _rotation()
    rotated_geometry = FixedSurfaceGeometry(
        geometry.atom_positions_angstrom @ rotation.T,
        geometry.surface_points_bohr @ rotation.T,
    )
    base = adapter.partial_fixed_surface_kernel_coordinate_vjp(
        geometry, source, cotangent
    )
    rotated = adapter.partial_fixed_surface_kernel_coordinate_vjp(
        rotated_geometry, _rotate_source(source, rotation), cotangent
    )
    np.testing.assert_allclose(rotated, base @ rotation.T, rtol=3.0e-13, atol=3.0e-12)


@pytest.mark.parametrize(
    "adapter", [ExactGTOCouplingAdapter(), LocalJetDiagnosticCouplingAdapter()]
)
def test_rigid_translation_and_rotation_covariance(adapter):
    geometry = _geometry()
    source = _source()
    base = adapter.apply_source(geometry, source)
    shift_angstrom = np.asarray([1.7, -0.8, 0.5])
    shifted = FixedSurfaceGeometry(
        geometry.atom_positions_angstrom + shift_angstrom,
        geometry.surface_points_bohr + shift_angstrom / Bohr,
    )
    np.testing.assert_allclose(
        adapter.apply_source(shifted, source), base, rtol=2.0e-14, atol=2.0e-13
    )
    rotation = _rotation()
    rotated = FixedSurfaceGeometry(
        geometry.atom_positions_angstrom @ rotation.T,
        geometry.surface_points_bohr @ rotation.T,
    )
    np.testing.assert_allclose(
        adapter.apply_source(rotated, _rotate_source(source, rotation)),
        base,
        rtol=3.0e-13,
        atol=3.0e-12,
    )


@pytest.mark.parametrize(
    "adapter", [ExactGTOCouplingAdapter(), LocalJetDiagnosticCouplingAdapter()]
)
def test_charged_source_potential_shift_changes_pairing_and_gauge_is_not_closed(
    adapter,
):
    source = _source()  # net charge = 0.17 e
    field = np.zeros_like(source)
    shifted = field.copy()
    shift_ev_per_e = 0.73
    shifted[:, 0] += shift_ev_per_e
    charge = adapter.source_space.total_charge(source, atom_count=2)
    delta = adapter.field_space.pair(source, shifted, atom_count=2) - (
        adapter.field_space.pair(source, field, atom_count=2)
    )
    assert delta == pytest.approx(charge * shift_ev_per_e, abs=1.0e-15)
    assert delta != 0.0
    assert "mean-potential gauge not closed" in (
        adapter.provenance.charged_source_gauge_status
    )
    assert adapter.field_space.gauge == "continuum-zero-at-infinity"


def test_exact_and_local_kernels_are_distinct_noninterchangeable_couplings():
    exact = ExactGTOCouplingAdapter(sigma_angstrom=1.5)
    local = LocalJetDiagnosticCouplingAdapter()
    assert exact.coupling_id != local.coupling_id
    assert exact.provider_id != local.provider_id
    assert exact.source_space is local.source_space
    assert exact.field_space is local.field_space
    assert exact.provenance.source_kernel != local.provenance.source_kernel
    assert not np.allclose(
        exact.apply_source(_geometry(), _source()),
        local.apply_source(_geometry(), _source()),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_shared_conjugate_map_locks_matrix_algebra_but_not_total_coordinates():
    geometry = _geometry()
    rng = np.random.default_rng(91)
    base = rng.normal(size=(5, 8))
    kernels = rng.normal(size=(2, 3, 5, 8))

    def matrix_builder(current):
        positions = np.asarray(current.atom_positions_angstrom)
        return base + np.einsum("aijk,ai->jk", kernels, positions)

    def partial_coordinate_vjp(current, source, surface_cotangent):
        del current
        return np.einsum("aijk,k,j->ai", kernels, source.reshape(-1), surface_cotangent)

    coupling = ConjugateSurfaceMap(
        matrix_builder=matrix_builder,
        partial_coordinate_vjp=partial_coordinate_vjp,
    )
    source = _source()
    direction = rng.normal(size=source.shape)
    cotangent = rng.normal(size=5)
    matrix = matrix_builder(geometry)
    np.testing.assert_allclose(
        coupling.apply_source(geometry, source), matrix @ source.ravel()
    )
    np.testing.assert_allclose(
        coupling.source_jvp(geometry, source, direction), matrix @ direction.ravel()
    )
    raw_adjoint = ATOMIC_L1_FIELD_DUAL_SPACE.pairing_metric.field_to_source_dual(
        coupling.source_vjp(geometry, source, cotangent)
    )
    np.testing.assert_allclose(raw_adjoint.ravel(), matrix.T @ cotangent)
    expected_partial = np.einsum("aijk,k,j->ai", kernels, source.reshape(-1), cotangent)
    np.testing.assert_allclose(
        coupling.partial_fixed_surface_kernel_coordinate_vjp(
            geometry, source, cotangent
        ),
        expected_partial,
    )
    with pytest.raises(CoordinateDerivativeUnavailable, match="no admitted total"):
        coupling.coordinate_vjp(geometry, source, cotangent)
