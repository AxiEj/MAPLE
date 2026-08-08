from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_reaction_coupling,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    AtomicReferenceDensityAsset,
    GaussianMixtureAtom,
    load_atomic_reference_density_asset,
)
from maple.function.calculator.extra_correction.implicit.route2_moist_drop import (
    MOIST_PINNED_C_API_VERSION,
    MOIST_PINNED_COMMIT,
    MOIST_PINNED_IMPORT_PATCH_SHA256,
    MOIST_PINNED_PYTHON_VERSION,
    MOIST_PINNED_SOURCE_VERSION,
    MoistDropSurfaceSnapshot,
    MoistDropSurfaceState,
    MoistRuntimeProvenance,
    route2_source_state_sha256,
)
from maple.function.calculator.extra_correction.implicit.route2_rhodrop_cpcm import (
    RHODROP_CPCM_FORCE_STATUS,
    RhoDropCPCMReactionField,
    RhoDropCPCMSettings,
    solve_frozen_source_rhodrop_cpcm,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
)
MANIFEST = TABLE.with_suffix(".json")


class _LinearSurfaceCavity:
    """Test double for a constant A and a supplied point-motion VJP."""

    def contract_amat_surface_weights(self, left, right):
        del left, right
        return np.zeros(6), np.zeros(6), np.zeros((3, 6))

    def contract_surface_lsf_weights(self, w_xi, w_f, w_xyz):
        del w_xi, w_f
        return np.zeros(6), np.asarray(w_xyz), np.zeros((3, 3, 6))


class _LinearSurfaceLevelSet:
    def __init__(self, point_source_derivative_bohr: np.ndarray) -> None:
        self._point_source_derivative_bohr = np.asarray(
            point_source_derivative_bohr,
            dtype=float,
        )

    def source_vjp(self, points_bohr, weights):
        del points_bohr
        result = np.zeros((1, 4))
        result[0, 1] = np.vdot(
            weights.gradient,
            self._point_source_derivative_bohr,
        )
        return result


def _surface_state(
    *,
    matrix: np.ndarray | None = None,
    positions_angstrom: np.ndarray | None = None,
    source_parameter: float = 0.0,
    point_source_derivative_bohr: np.ndarray | None = None,
    bound_source: np.ndarray | None = None,
) -> MoistDropSurfaceState:
    radius = 3.0
    points = radius * np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    if point_source_derivative_bohr is None:
        point_source_derivative_bohr = np.zeros_like(points)
    point_source_derivative_bohr = np.asarray(
        point_source_derivative_bohr,
        dtype=float,
    )
    points = points + float(source_parameter) * point_source_derivative_bohr
    if positions_angstrom is None:
        positions_angstrom = np.asarray([[0.17, -0.08, 0.12]])
    if matrix is None:
        matrix = 1.7 * np.eye(6) + 0.08 * np.ones((6, 6))
    if bound_source is None:
        bound_source = np.asarray([[0.23, 0.04, -0.03, 0.05]])
    normals = points / np.linalg.norm(points, axis=1)[:, None]
    snapshot = MoistDropSurfaceSnapshot(
        atomic_numbers=np.asarray([1]),
        atom_positions_bohr=np.asarray(positions_angstrom) / Bohr,
        surface_points_bohr=points,
        surface_normals=normals,
        reference_normals=normals,
        surface_areas_bohr2=np.ones(6),
        owner_atom_indices=np.zeros(6, dtype=int),
        converged=np.ones(6, dtype=bool),
        xi=np.ones(6),
        switching_f=np.ones(6),
        wleb=np.ones(6),
        reference_displacements_bohr=np.ones(6),
        branch_rho=np.ones(6),
        amat=matrix,
        area_bohr2=6.0,
        volume_bohr3=12.0,
        geometry_sha256="1" * 64,
        level_set_state_sha256="2" * 64,
        parameter_sha256="3" * 64,
        runtime_sha256="4" * 64,
        bound_source_sha256=route2_source_state_sha256(bound_source),
    )
    return MoistDropSurfaceState(
        snapshot=snapshot,
        _cavity=_LinearSurfaceCavity(),
        _level_set=_LinearSurfaceLevelSet(point_source_derivative_bohr),
    )


def _source() -> np.ndarray:
    # Raw MACE-POLAR order: [q, p_y, p_z, p_x].
    return np.asarray([[0.23, 0.04, -0.03, 0.05]])


def _asset():
    return load_atomic_reference_density_asset(
        table_path=TABLE,
        manifest_path=MANIFEST,
    )


def _synthetic_runtime() -> MoistRuntimeProvenance:
    return MoistRuntimeProvenance(
        evidence_kind="synthetic-test-double",
        upstream_commit=MOIST_PINNED_COMMIT,
        source_version=MOIST_PINNED_SOURCE_VERSION,
        python_package_version=MOIST_PINNED_PYTHON_VERSION,
        c_api_version=MOIST_PINNED_C_API_VERSION,
        import_patch_sha256=MOIST_PINNED_IMPORT_PATCH_SHA256,
    )


def _provider_with_cached_state(
    state,
    positions_angstrom: np.ndarray,
) -> RhoDropCPCMReactionField:
    provider = RhoDropCPCMReactionField(
        _asset(),
        np.asarray([1]),
        positions_angstrom,
        expected_total_charge_e=0.23,
        runtime=_synthetic_runtime(),
        _allow_synthetic_runtime=True,
    )
    provider._scf_snapshot = state
    return provider


def test_frozen_source_cpcm_matches_the_stationary_moist_equation_and_pairing() -> None:
    state = _surface_state()
    positions = state.snapshot.atom_positions_bohr * Bohr
    source = _source()
    settings = RhoDropCPCMSettings(dielectric=80.0)
    solved = solve_frozen_source_rhodrop_cpcm(
        state,
        positions,
        source,
        settings=settings,
    )

    potential = point_multipole_potential(
        state.snapshot.surface_points_bohr,
        positions,
        source,
    )
    expected_charge = np.linalg.solve(
        state.snapshot.amat,
        -settings.dielectric_factor * potential,
    )
    expected_reaction_potential, expected_reaction_gradient = (
        point_asc_reaction_potential_gradient(
            positions,
            state.snapshot.surface_points_bohr,
            expected_charge,
        )
    )

    np.testing.assert_allclose(solved.surface_potential_hartree_per_e, potential)
    np.testing.assert_allclose(solved.surface_charge_e, expected_charge)
    np.testing.assert_allclose(
        solved.reaction_potential_hartree_per_e,
        expected_reaction_potential,
    )
    np.testing.assert_allclose(
        solved.reaction_gradient_hartree_per_e_bohr,
        expected_reaction_gradient,
    )
    expected_coupling = float(np.dot(potential, expected_charge))
    assert solved.polarization_energy_hartree == pytest.approx(
        0.5 * expected_coupling,
        abs=1.0e-15,
    )
    assert solved.density_reaction_coupling_hartree == pytest.approx(
        density_reaction_coupling(
            source,
            expected_reaction_potential,
            expected_reaction_gradient,
        ),
        abs=1.0e-15,
    )
    assert solved.polarization_energy_hartree < 0.0
    assert solved.linear_residual_absolute < 1.0e-14
    assert solved.linear_residual_relative < 1.0e-13
    np.testing.assert_allclose(
        solved.reaction_field_ev,
        np.column_stack(
            (
                expected_reaction_potential * Hartree,
                expected_reaction_gradient * Hartree / Bohr,
            )
        ),
        rtol=2.0e-14,
        atol=1.0e-15,
    )


def test_frozen_state_is_immutable_and_content_addressed() -> None:
    surface = _surface_state()
    positions = surface.snapshot.atom_positions_bohr * Bohr
    first = solve_frozen_source_rhodrop_cpcm(surface, positions, _source())
    second = solve_frozen_source_rhodrop_cpcm(surface, positions, _source())

    assert first.source_sha256 == second.source_sha256
    assert first.operator_sha256 == second.operator_sha256
    assert first.state_sha256 == second.state_sha256
    assert first.surface_snapshot.snapshot_sha256 == surface.snapshot.snapshot_sha256
    assert not first.surface_charge_e.flags.writeable
    with pytest.raises(ValueError):
        first.surface_charge_e[0] = 1.0


def test_source_dependent_energy_lookup_never_silently_rebuilds_a_stale_state() -> None:
    surface = _surface_state()
    positions = surface.snapshot.atom_positions_bohr * Bohr
    source = _source()
    state = solve_frozen_source_rhodrop_cpcm(surface, positions, source)
    provider = _provider_with_cached_state(state, positions)

    assert provider.assert_forward_state(source) == state.state_sha256
    assert provider.scf_polarization_energy_hartree(source) == (
        state.polarization_energy_hartree
    )
    changed = source.copy()
    changed[0, 1] += 1.0e-8
    with pytest.raises(RuntimeError, match="does not match the cached"):
        provider.scf_polarization_energy_hartree(changed)


def test_cached_state_rejects_provider_parameter_or_geometry_rebinding() -> None:
    surface = _surface_state()
    positions = surface.snapshot.atom_positions_bohr * Bohr
    source = _source()
    state = solve_frozen_source_rhodrop_cpcm(surface, positions, source)

    parameter_mutation = _provider_with_cached_state(state, positions)
    parameter_mutation.n_iso_e_per_bohr3 *= 2.0
    with pytest.raises(RuntimeError, match="configuration changed"):
        parameter_mutation.assert_forward_state(source)

    geometry_mutation = _provider_with_cached_state(state, positions)
    geometry_mutation.atom_positions_angstrom = (
        geometry_mutation.atom_positions_angstrom + np.asarray([[1.0e-8, 0.0, 0.0]])
    )
    with pytest.raises(RuntimeError, match="configuration changed"):
        geometry_mutation.adjoint(np.zeros_like(source))


def test_provider_configuration_binds_unpacked_asset_content() -> None:
    def provider(exponent: float) -> RhoDropCPCMReactionField:
        asset = AtomicReferenceDensityAsset(
            mixtures_by_atomic_number={
                1: GaussianMixtureAtom(
                    electron_counts=np.asarray([1.0]),
                    gaussian_exponents_bohr2=np.asarray([exponent]),
                )
            },
            table_sha256="0" * 64,
            manifest_sha256="1" * 64,
            provenance={"purpose": "configuration-fingerprint-regression"},
        )
        return RhoDropCPCMReactionField(
            asset,
            np.asarray([1]),
            np.zeros((1, 3)),
            expected_total_charge_e=0.0,
            runtime=_synthetic_runtime(),
            _allow_synthetic_runtime=True,
        )

    first = provider(0.4)
    second = provider(0.8)
    assert first.asset.table_sha256 == second.asset.table_sha256
    assert first.asset.manifest_sha256 == second.asset.manifest_sha256
    assert first.asset.content_sha256 != second.asset.content_sha256
    assert first._configuration_sha256 != second._configuration_sha256


def test_nonlinear_reaction_map_jvp_vjp_matches_independent_finite_difference() -> None:
    point_derivative = np.asarray(
        [
            [0.04, -0.01, 0.02],
            [-0.03, 0.02, 0.01],
            [0.01, 0.03, -0.02],
            [-0.02, -0.01, 0.04],
            [0.03, 0.01, 0.02],
            [-0.01, 0.04, -0.03],
        ]
    )
    source = _source()
    positions = np.asarray([[0.17, -0.08, 0.12]])

    def solve_at(candidate_source: np.ndarray):
        surface = _surface_state(
            positions_angstrom=positions,
            source_parameter=candidate_source[0, 1],
            point_source_derivative_bohr=point_derivative,
            bound_source=candidate_source,
        )
        return solve_frozen_source_rhodrop_cpcm(
            surface,
            positions,
            candidate_source,
        )

    base = solve_at(source)
    provider = _provider_with_cached_state(base, positions)
    direction = np.asarray([[0.0, 0.31, -0.27, 0.19]])
    cotangent = np.asarray([[0.7, -0.4, 0.2, 0.6]])
    jvp = provider.apply(direction)
    vjp = provider.adjoint(cotangent)

    step = 2.0e-6
    finite_difference = (
        solve_at(source + step * direction).reaction_field_ev
        - solve_at(source - step * direction).reaction_field_ev
    ) / (2.0 * step)
    np.testing.assert_allclose(
        jvp,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-9,
    )
    assert np.vdot(cotangent, jvp) == pytest.approx(
        np.vdot(vjp, direction),
        rel=2.0e-13,
        abs=2.0e-13,
    )


def test_frozen_source_cpcm_rejects_stale_geometry_and_invalid_operator() -> None:
    surface = _surface_state()
    positions = surface.snapshot.atom_positions_bohr * Bohr
    displaced = positions.copy()
    displaced[0, 0] += 1.0e-12
    with pytest.raises(RuntimeError, match="geometries differ"):
        solve_frozen_source_rhodrop_cpcm(surface, displaced, _source())

    foreign_source = _source().copy()
    foreign_source[0, 1] += 1.0e-8
    with pytest.raises(RuntimeError, match="does not own"):
        solve_frozen_source_rhodrop_cpcm(surface, positions, foreign_source)

    indefinite = np.eye(6)
    indefinite[0, 1] = indefinite[1, 0] = 2.0
    with pytest.raises(RuntimeError, match="not positive definite"):
        solve_frozen_source_rhodrop_cpcm(
            _surface_state(matrix=indefinite),
            positions,
            _source(),
        )

    ill_conditioned = np.diag([1.0e-14, 1.0, 1.1, 1.2, 1.3, 1.4])
    with pytest.raises(RuntimeError, match="condition number"):
        solve_frozen_source_rhodrop_cpcm(
            _surface_state(matrix=ill_conditioned),
            positions,
            _source(),
        )


def test_cpcm_settings_and_pr2_derivative_surface_fail_closed() -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        RhoDropCPCMSettings(dielectric=1.0)
    with pytest.raises(ValueError, match="positive and finite"):
        RhoDropCPCMSettings(linear_solve_relative_tolerance=0.0)

    provider = RhoDropCPCMReactionField(
        _asset(),
        np.asarray([1]),
        np.zeros((1, 3)),
        expected_total_charge_e=0.0,
        runtime=_synthetic_runtime(),
        _allow_synthetic_runtime=True,
    )
    assert provider.source_dependent_geometry
    assert provider.reciprocal_energy_pairing
    assert not provider.reaction_jacobian_self_adjoint
    assert provider.reaction_map_derivative_available
    assert not provider.operational_jvp_efficiency_admitted
    assert not provider.complete_position_derivative_available
    assert provider.analytic_force_status == RHODROP_CPCM_FORCE_STATUS
    with pytest.raises(RuntimeError, match="preceding apply_scf"):
        provider.apply(np.zeros((1, 4)))
    with pytest.raises(RuntimeError, match="preceding apply_scf"):
        provider.adjoint(np.zeros((1, 4)))

    with pytest.raises(ValueError, match="Gaussian width"):
        RhoDropCPCMReactionField(
            _asset(),
            np.asarray([1]),
            np.zeros((1, 3)),
            expected_total_charge_e=0.0,
            runtime=_synthetic_runtime(),
            sigma_angstrom=0.0,
            _allow_synthetic_runtime=True,
        )
    with pytest.raises(ValueError, match="minimum reconstructed density"):
        RhoDropCPCMReactionField(
            _asset(),
            np.asarray([1]),
            np.zeros((1, 3)),
            expected_total_charge_e=0.0,
            runtime=_synthetic_runtime(),
            minimum_density_e_per_bohr3=-1.0,
            _allow_synthetic_runtime=True,
        )
    with pytest.raises(ValueError, match="electron-count tolerance"):
        RhoDropCPCMReactionField(
            _asset(),
            np.asarray([1]),
            np.zeros((1, 3)),
            expected_total_charge_e=0.0,
            runtime=_synthetic_runtime(),
            electron_count_tolerance=0.0,
            _allow_synthetic_runtime=True,
        )


def _real_runtime() -> MoistRuntimeProvenance:
    import moist._libmoist as extension

    shared_library = Path(os.environ["MAPLE_ROUTE2_MOIST_LIBRARY"])
    extension_path = Path(extension.__file__)

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return MoistRuntimeProvenance(
        evidence_kind="real-pinned-build",
        python_extension_path=str(extension_path),
        python_extension_sha256=sha256(extension_path),
        shared_library_path=str(shared_library),
        shared_library_sha256=sha256(shared_library),
        build_toolchain=(
            "gfortran-11.4.0",
            "meson-1.11.2",
            "ninja-1.13.0",
            "openmp-enabled",
        ),
    )


def _real_provider(*, require_exact_cold_replay: bool):
    import moist

    return RhoDropCPCMReactionField(
        _asset(),
        np.asarray([1]),
        np.zeros((1, 3)),
        expected_total_charge_e=0.0,
        runtime=_real_runtime(),
        moist_module=moist,
        cpcm_settings=RhoDropCPCMSettings(
            require_exact_cold_replay=require_exact_cold_replay,
        ),
    )


@pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_MOIST_REAL") != "1",
    reason="requires the separately built pinned MOIST runtime",
)
def test_real_pinned_moist_nonzero_source_cpcm_cold_replay() -> None:
    provider = _real_provider(require_exact_cold_replay=True)
    source = np.asarray([[0.0, 0.002, -0.001, 0.0015]])
    field = provider.apply_scf(source)
    state = provider.scf_snapshot(source)

    assert state.surface_snapshot.surface_size == 194
    # The state identity deliberately includes the exact LAPACK solution and
    # diagnostics, whose last bits depend on the configured BLAS thread count.
    # Exact cold replay above is the within-runtime determinism gate.  Portable
    # cross-runtime checks below use physical tolerances rather than byte hashes.
    assert len(state.state_sha256) == 64
    assert len(state.operator_sha256) == 64
    assert state.polarization_energy_hartree == pytest.approx(
        -4.06526157419331e-07,
        abs=2.0e-19,
    )
    assert state.surface_coupling_hartree == pytest.approx(
        state.density_reaction_coupling_hartree,
        abs=1.0e-18,
    )
    assert state.linear_residual_absolute < 1.0e-17
    assert state.condition_number_2 == pytest.approx(
        20.792984988145154,
        rel=2.0e-13,
    )
    assert np.all(np.isfinite(field))


@pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_MOIST_REAL") != "1",
    reason="requires the separately built pinned MOIST runtime",
)
def test_real_pinned_moist_nonlinear_map_jvp_vjp_and_finite_difference() -> None:
    provider = _real_provider(require_exact_cold_replay=False)
    source = np.asarray([[0.0, 0.002, -0.001, 0.0015]])
    direction = np.asarray([[0.0, 0.31, -0.27, 0.19]])
    cotangent = np.asarray([[0.7, -0.4, 0.2, 0.6]])
    provider.apply_scf(source)
    jvp = provider.apply(direction)
    vjp = provider.adjoint(cotangent)

    step = 3.0e-6
    finite_difference = (
        provider.apply_scf(source + step * direction)
        - provider.apply_scf(source - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        jvp,
        finite_difference,
        rtol=8.0e-11,
        atol=8.0e-12,
    )
    assert np.vdot(cotangent, jvp) == pytest.approx(
        np.vdot(vjp, direction),
        rel=2.0e-14,
        abs=2.0e-14,
    )


@pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_MOIST_REAL") != "1",
    reason="requires the separately built pinned MOIST runtime",
)
def test_real_pinned_moist_water_multicentre_forward_canary() -> None:
    import moist

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.927297, 0.0],
        ]
    )
    provider = RhoDropCPCMReactionField(
        _asset(),
        np.asarray([8, 1, 1]),
        positions,
        expected_total_charge_e=0.0,
        runtime=_real_runtime(),
        moist_module=moist,
        cpcm_settings=RhoDropCPCMSettings(require_exact_cold_replay=False),
    )
    source = np.asarray(
        [
            [-0.02, 0.001, -0.0005, 0.0002],
            [0.01, -0.0002, 0.0001, 0.0003],
            [0.01, 0.0001, 0.0004, -0.0002],
        ]
    )
    direction = np.asarray(
        [
            [0.017, 0.23, -0.17, 0.11],
            [-0.009, -0.13, 0.07, 0.19],
            [-0.008, 0.04, 0.10, -0.21],
        ]
    )
    cotangent = np.asarray(
        [
            [0.7, -0.4, 0.2, 0.6],
            [-0.2, 0.3, -0.5, 0.1],
            [0.4, 0.15, -0.25, -0.35],
        ]
    )

    field = provider.apply_scf(source)
    state = provider.scf_snapshot(source)
    snapshot = state.surface_snapshot
    jvp = provider.apply(direction)
    vjp = provider.adjoint(cotangent)

    assert snapshot.surface_size == 413
    owners, owner_counts = np.unique(
        snapshot.owner_atom_indices,
        return_counts=True,
    )
    np.testing.assert_array_equal(
        owners,
        np.asarray([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        owner_counts,
        np.asarray([175, 119, 119]),
    )
    assert snapshot.area_bohr2 == pytest.approx(209.2709647504298, abs=2.0e-10)
    assert snapshot.volume_bohr3 == pytest.approx(280.8536581156884, abs=2.0e-10)
    assert len(state.state_sha256) == 64
    assert len(state.operator_sha256) == 64
    assert state.polarization_energy_hartree == pytest.approx(
        -5.011196885870286e-06,
        abs=2.0e-18,
    )
    assert state.surface_coupling_hartree == state.density_reaction_coupling_hartree
    assert np.all(np.isfinite(field))

    replay_provider = RhoDropCPCMReactionField(
        _asset(),
        np.asarray([8, 1, 1]),
        positions,
        expected_total_charge_e=0.0,
        runtime=_real_runtime(),
        moist_module=moist,
        cpcm_settings=RhoDropCPCMSettings(require_exact_cold_replay=False),
    )
    replay_provider.apply_scf(source)
    assert replay_provider.scf_snapshot(source).surface_snapshot.snapshot_sha256 == (
        snapshot.snapshot_sha256
    )

    step = 1.0e-5
    finite_difference = (
        provider.apply_scf(source + step * direction)
        - provider.apply_scf(source - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        jvp,
        finite_difference,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert np.vdot(cotangent, jvp) == pytest.approx(
        np.vdot(vjp, direction),
        rel=2.0e-13,
        abs=2.0e-13,
    )
