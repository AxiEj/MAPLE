from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import (
    route2_moist_drop as moist_drop_module,
)

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    AtomicReferenceDensityAsset,
    GaussianMixtureAtom,
    load_atomic_reference_density_asset,
)
from maple.function.calculator.extra_correction.implicit.route2_density_levelset import (
    DensityBoundaryDiagnostics,
    LSFAdjointWeights,
    ReconstructedMacePolarDensityLevelSet,
    SpatialLSFJet,
)
from maple.function.calculator.extra_correction.implicit.route2_moist_drop import (
    MOIST_CALLBACK_SCALE,
    MOIST_PINNED_C_API_VERSION,
    MOIST_PINNED_COMMIT,
    MOIST_PINNED_COMPILED_DROP_TOLERANCE,
    MOIST_PINNED_IMPORT_PATCH_REPO_PATH,
    MOIST_PINNED_IMPORT_PATCH_SHA256,
    MOIST_PINNED_PYTHON_VERSION,
    MOIST_PINNED_PYTHON_PACKAGE_SHA256,
    MOIST_PINNED_SOURCE_VERSION,
    MoistDropAdapter,
    MoistDropSettings,
    MoistRuntimeProvenance,
    loaded_moist_shared_library_path,
    reconstructed_level_set_state_sha256,
)


class _SphereLevelSet:
    atom_count = 1
    level_set_identity = "synthetic-sphere"
    density_identity = "synthetic-sphere"

    def __init__(self, radius_bohr: float = 2.0) -> None:
        self.radius_bohr = float(radius_bohr)
        self.boundary_calls = 0

    def evaluate_spatial(self, points_bohr: object) -> SpatialLSFJet:
        points = np.asarray(points_bohr, dtype=float)
        count = points.shape[0]
        return SpatialLSFJet(
            value=np.einsum("mi,mi->m", points, points) - self.radius_bohr**2,
            gradient=2.0 * points,
            hessian=np.broadcast_to(2.0 * np.eye(3), (count, 3, 3)),
            third=np.zeros((count, 3, 3, 3)),
        )

    def source_jvp(
        self,
        points_bohr: object,
        source_direction: object,
    ) -> SpatialLSFJet:
        points = np.asarray(points_bohr, dtype=float)
        return SpatialLSFJet(
            value=np.zeros(points.shape[0]),
            gradient=np.zeros_like(points),
            hessian=np.zeros((points.shape[0], 3, 3)),
            third=np.zeros((points.shape[0], 3, 3, 3)),
        )

    def source_vjp(
        self,
        points_bohr: object,
        weights: LSFAdjointWeights,
    ) -> np.ndarray:
        total = float(
            np.sum(weights.value) + np.sum(weights.gradient) + np.sum(weights.hessian)
        )
        return np.full((1, 4), total)

    def nuclear_vjp(
        self,
        points_bohr: object,
        weights: LSFAdjointWeights,
    ) -> np.ndarray:
        return np.full((1, 3), float(np.sum(weights.value)))

    def require_valid_boundary(
        self,
        points_bohr: object,
        validation_shell_points_bohr: object | None = None,
    ) -> DensityBoundaryDiagnostics:
        self.boundary_calls += 1
        points = np.asarray(points_bohr, dtype=float)
        maximum_residual = float(np.max(np.abs(self.evaluate_spatial(points).value)))
        if maximum_residual > 1.0e-12:
            raise ValueError("synthetic surface is off level set")
        if validation_shell_points_bohr is None:
            raise ValueError("validation shell is required")
        return DensityBoundaryDiagnostics(
            minimum_reconstructed_density_e_per_bohr3=1.0,
            minimum_level_set_gradient_norm_bohr=2.0 * self.radius_bohr,
            maximum_level_set_residual=maximum_residual,
            integrated_electron_count=1.0,
            expected_electron_count=1.0,
            electron_count_absolute_error=0.0,
            require_surface=True,
            passed=True,
            reasons=(),
        )


class _FakeStructure:
    def __init__(self, numbers: np.ndarray, positions: np.ndarray) -> None:
        self.numbers = np.array(numbers, copy=True)
        self.positions = np.array(positions, copy=True)


class _FakeIsodensityDROPCavity:
    last_kwargs: dict[str, object] = {}
    converged_value = True

    def __init__(self, callback, **kwargs) -> None:
        type(self).last_kwargs = dict(kwargs)
        self.callback = callback
        radius = 2.0
        points = radius * np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        self._points = points
        self._snapshot = SimpleNamespace(
            area=4.0,
            volume=8.0 / 3.0,
            ngrid=4,
            nsph=1,
            xyz=points.T,
            a=np.ones(4),
            owner=np.zeros(4, dtype=np.int32),
            converged=np.full(4, self.converged_value, dtype=bool),
            radii=np.array([radius]),
            asph=np.array([16.0]),
            nmax=4,
            normal0=(points / radius).T,
            wleb=np.ones(4),
            r_iI0=np.ones(4),
            f=np.ones(4),
            rho=np.ones(4),
        )

    def update(self, structure: _FakeStructure) -> None:
        for point in self._points:
            self.callback(point)

    @property
    def cavity(self):
        return self._snapshot

    def assemble_amat(self):
        return np.eye(4), np.arange(4, dtype=float) + 1.0

    def contract_amat_surface_weights(self, q1, q2):
        return q1 + q2, q1 - q2, np.vstack((q1, q2, q1 * q2))

    def contract_surface_lsf_weights(self, w_xi, w_f, w_xyz):
        w1 = np.asarray(w_xyz, dtype=float)
        w2 = np.zeros((3, 3, 4))
        for index in range(3):
            w2[index, index] = w_f
        return np.asarray(w_xi), w1, w2


class _FakeLibrary:
    @staticmethod
    def get_api_version() -> str:
        return MOIST_PINNED_C_API_VERSION


class _FakeMoist:
    __version__ = MOIST_PINNED_PYTHON_VERSION
    __file__ = __file__
    library = _FakeLibrary()
    Structure = _FakeStructure
    IsodensityDROPCavity = _FakeIsodensityDROPCavity


class _PermutingIsodensityDROPCavity(_FakeIsodensityDROPCavity):
    build_count = 0

    def __init__(self, callback, **kwargs) -> None:
        super().__init__(callback, **kwargs)
        type(self).build_count += 1
        if type(self).build_count % 2 == 0:
            permutation = np.asarray([2, 0, 3, 1])
            self._points = self._points[permutation]
            for name in ("a", "owner", "converged", "wleb", "r_iI0", "f", "rho"):
                setattr(
                    self._snapshot,
                    name,
                    np.asarray(getattr(self._snapshot, name))[permutation],
                )
            self._snapshot.xyz = self._points.T
            self._snapshot.normal0 = (self._points / 2.0).T
            # Simulate harmless OpenMP reduction-order noise in MOIST's scalar
            # totals.  The canonical point arrays remain exactly identical.
            self._snapshot.area += 2.0e-14
            self._snapshot.volume -= 3.0e-14

    def assemble_amat(self):
        matrix = 2.0 * np.eye(4) + 0.03 * (self._points @ self._points.T)
        xi = self._points @ np.asarray([0.3, -0.2, 0.1])
        return matrix, xi


class _PermutingMoist(_FakeMoist):
    IsodensityDROPCavity = _PermutingIsodensityDROPCavity


def _synthetic_runtime() -> MoistRuntimeProvenance:
    return MoistRuntimeProvenance(
        evidence_kind="synthetic-test-double",
        upstream_commit=MOIST_PINNED_COMMIT,
        source_version=MOIST_PINNED_SOURCE_VERSION,
        python_package_version=MOIST_PINNED_PYTHON_VERSION,
        c_api_version=MOIST_PINNED_C_API_VERSION,
        import_patch_sha256=MOIST_PINNED_IMPORT_PATCH_SHA256,
    )


def _synthetic_adapter(
    level_set: _SphereLevelSet | None = None,
    *,
    module=_FakeMoist,
) -> MoistDropAdapter:
    return MoistDropAdapter(
        level_set or _SphereLevelSet(),
        np.array([1]),
        np.zeros((1, 3)),
        runtime=_synthetic_runtime(),
        settings=MoistDropSettings(nleb=4),
        level_set_state_sha256="1" * 64,
        moist_module=module,
        _allow_synthetic_runtime=True,
    )


def test_synthetic_runtime_is_never_admitted_implicitly() -> None:
    with pytest.raises(RuntimeError, match="test doubles"):
        MoistDropAdapter(
            _SphereLevelSet(),
            np.array([1]),
            np.zeros((1, 3)),
            runtime=_synthetic_runtime(),
            level_set_state_sha256="1" * 64,
            moist_module=_FakeMoist,
        )


def test_pinned_settings_do_not_claim_unavailable_tolerance_or_double_scale() -> None:
    with pytest.raises(ValueError, match="scale must be 1.0"):
        MoistDropSettings(callback_scale=1000.0)
    with pytest.raises(ValueError, match="cannot override"):
        MoistDropSettings(compiled_drop_tolerance=1.0e-12)
    settings = MoistDropSettings()
    assert settings.callback_scale == MOIST_CALLBACK_SCALE
    assert settings.compiled_drop_tolerance == MOIST_PINNED_COMPILED_DROP_TOLERANCE


def test_archived_moist_import_patch_matches_pinned_provenance() -> None:
    root = Path(__file__).resolve().parents[2]
    patch = root / MOIST_PINNED_IMPORT_PATCH_REPO_PATH

    assert patch.is_file()
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == (
        MOIST_PINNED_IMPORT_PATCH_SHA256
    )


def test_fingerprintable_level_set_rejects_a_forged_explicit_state_hash() -> None:
    root = Path(__file__).resolve().parents[2]
    table = (
        root / "docs/implicit-solvation/benchmarks/"
        "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
    )
    asset = load_atomic_reference_density_asset(
        table_path=table,
        manifest_path=table.with_suffix(".json"),
    )
    level_set = ReconstructedMacePolarDensityLevelSet(
        asset,
        np.asarray([1]),
        np.zeros((1, 3)),
        np.zeros((1, 4)),
        1.0e-3,
        expected_total_charge_e=0.0,
    )

    with pytest.raises(RuntimeError, match="does not match"):
        MoistDropAdapter(
            level_set,
            np.asarray([1]),
            np.zeros((1, 3)),
            runtime=_synthetic_runtime(),
            level_set_state_sha256="f" * 64,
            moist_module=_FakeMoist,
            _allow_synthetic_runtime=True,
        )


def test_level_set_fingerprint_binds_unpacked_asset_content() -> None:
    def level_set(exponent: float) -> ReconstructedMacePolarDensityLevelSet:
        asset = AtomicReferenceDensityAsset(
            mixtures_by_atomic_number={
                1: GaussianMixtureAtom(
                    electron_counts=np.asarray([1.0]),
                    gaussian_exponents_bohr2=np.asarray([exponent]),
                )
            },
            table_sha256="0" * 64,
            manifest_sha256="1" * 64,
            provenance={"purpose": "content-fingerprint-regression"},
        )
        return ReconstructedMacePolarDensityLevelSet(
            asset,
            np.asarray([1]),
            np.zeros((1, 3)),
            np.zeros((1, 4)),
            1.0e-3,
            expected_total_charge_e=0.0,
        )

    first = level_set(0.4)
    second = level_set(0.8)
    assert first.asset.table_sha256 == second.asset.table_sha256
    assert first.asset.manifest_sha256 == second.asset.manifest_sha256
    assert first.asset.content_sha256 != second.asset.content_sha256
    assert reconstructed_level_set_state_sha256(first) != (
        reconstructed_level_set_state_sha256(second)
    )


def test_adapter_builds_immutable_source_bound_snapshot() -> None:
    level_set = _SphereLevelSet()
    first = _synthetic_adapter(level_set).build_surface()
    second = _synthetic_adapter(_SphereLevelSet()).build_surface()
    snapshot = first.snapshot

    assert _FakeIsodensityDROPCavity.last_kwargs["scale"] == 1.0
    assert _FakeIsodensityDROPCavity.last_kwargs["wleb_prune_level"] == 0
    assert snapshot.surface_points_bohr.shape == (4, 3)
    assert snapshot.surface_normals.shape == (4, 3)
    assert snapshot.amat.shape == (4, 4)
    assert snapshot.snapshot_sha256 == second.snapshot.snapshot_sha256
    assert level_set.boundary_calls == 1
    assert not snapshot.surface_points_bohr.flags.writeable
    with pytest.raises(ValueError):
        snapshot.surface_points_bohr[0, 0] = 0.0


def test_adapter_canonicalizes_upstream_surface_permutations_and_adjoint_order() -> (
    None
):
    _PermutingIsodensityDROPCavity.build_count = 0
    first = _synthetic_adapter(module=_PermutingMoist).build_surface()
    second = _synthetic_adapter(module=_PermutingMoist).build_surface()
    assert first.snapshot.snapshot_sha256 == second.snapshot.snapshot_sha256
    np.testing.assert_array_equal(
        first.snapshot.surface_points_bohr,
        second.snapshot.surface_points_bohr,
    )
    np.testing.assert_array_equal(first.snapshot.amat, second.snapshot.amat)

    left = np.asarray([0.4, -0.3, 0.2, 0.1])
    right = np.asarray([-0.2, 0.5, 0.7, -0.1])
    first_weights = first.contract_amat_surface_weights(left, right)
    second_weights = second.contract_amat_surface_weights(left, right)
    np.testing.assert_array_equal(first_weights.xi, second_weights.xi)
    np.testing.assert_array_equal(first_weights.switching, second_weights.switching)
    np.testing.assert_array_equal(first_weights.points_bohr, second_weights.points_bohr)
    first_lsf = first.contract_surface_lsf_weights(first_weights)
    second_lsf = second.contract_surface_lsf_weights(second_weights)
    np.testing.assert_array_equal(first_lsf.value, second_lsf.value)
    np.testing.assert_array_equal(first_lsf.gradient, second_lsf.gradient)
    np.testing.assert_array_equal(first_lsf.hessian, second_lsf.hessian)


def test_adapter_rejects_inconsistent_upstream_geometry_totals() -> None:
    class InconsistentTotalsCavity(_FakeIsodensityDROPCavity):
        def __init__(self, callback, **kwargs) -> None:
            super().__init__(callback, **kwargs)
            self._snapshot.volume += 0.1

    class InconsistentTotalsMoist(_FakeMoist):
        IsodensityDROPCavity = InconsistentTotalsCavity

    with pytest.raises(RuntimeError, match="canonical surface quadrature"):
        _synthetic_adapter(module=InconsistentTotalsMoist).build_surface()


def test_surface_adjoint_shapes_are_transposed_once_at_boundary() -> None:
    state = _synthetic_adapter().build_surface()
    left = np.arange(4, dtype=float)
    right = np.arange(4, dtype=float) + 2.0
    surface = state.contract_amat_surface_weights(left, right)
    assert surface.points_bohr.shape == (4, 3)
    lsf = state.contract_surface_lsf_weights(surface)
    assert lsf.value.shape == (4,)
    assert lsf.gradient.shape == (4, 3)
    assert lsf.hessian.shape == (4, 3, 3)
    assert state.source_vjp(surface).shape == (1, 4)
    assert state.nuclear_field_vjp(surface).shape == (1, 3)


def test_runtime_and_projection_fail_closed() -> None:
    class WrongVersion(_FakeMoist):
        __version__ = "999"

    with pytest.raises(RuntimeError, match="version is not pinned"):
        _synthetic_adapter(module=WrongVersion)

    class NonconvergedCavity(_FakeIsodensityDROPCavity):
        converged_value = False

    class NonconvergedMoist(_FakeMoist):
        IsodensityDROPCavity = NonconvergedCavity

    with pytest.raises(ValueError, match="must converge"):
        _synthetic_adapter(module=NonconvergedMoist).build_surface()


def test_callback_exceptions_are_recovered_from_cffi_boundary() -> None:
    class BrokenLevelSet(_SphereLevelSet):
        def evaluate_spatial(self, points_bohr: object) -> SpatialLSFJet:
            raise ArithmeticError("broken analytic jet")

    with pytest.raises(RuntimeError, match="callback failed") as caught:
        _synthetic_adapter(BrokenLevelSet()).build_surface()
    assert isinstance(caught.value.__cause__, ArithmeticError)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real_provenance_for_files(
    extension_path: Path,
    shared_library_path: Path,
) -> MoistRuntimeProvenance:
    return MoistRuntimeProvenance(
        evidence_kind="real-pinned-build",
        python_extension_path=str(extension_path),
        python_extension_sha256=_sha256(extension_path),
        shared_library_path=str(shared_library_path),
        shared_library_sha256=_sha256(shared_library_path),
        build_toolchain=("synthetic-toolchain-record",),
    )


def test_runtime_content_identity_is_relocation_stable(tmp_path: Path) -> None:
    identities = []
    for prefix in (tmp_path / "first", tmp_path / "second"):
        prefix.mkdir()
        extension = prefix / "_libmoist.so"
        shared = prefix / "libmoist.so.0"
        extension.write_bytes(b"same-extension-content")
        shared.write_bytes(b"same-shared-library-content")
        identities.append(_real_provenance_for_files(extension, shared).identity_sha256)
    assert identities[0] == identities[1]


def test_real_provenance_rejects_injected_module_even_with_canonical_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "moist"
    package.mkdir()
    extension = package / "_libmoist.so"
    shared = tmp_path / "libmoist.so.0"
    extension.write_bytes(b"extension")
    shared.write_bytes(b"library")
    runtime = _real_provenance_for_files(extension, shared)
    canonical = SimpleNamespace(__name__="moist")
    monkeypatch.setattr(
        moist_drop_module.importlib,
        "import_module",
        lambda _name: canonical,
    )
    injected = SimpleNamespace(
        __name__="moist",
        __file__=str(package / "__init__.py"),
        __version__=MOIST_PINNED_PYTHON_VERSION,
        library=_FakeLibrary(),
        Structure=_FakeStructure,
        IsodensityDROPCavity=_FakeIsodensityDROPCavity,
    )
    with pytest.raises(RuntimeError, match="forbids injected"):
        runtime.verify_module(injected)


def test_real_provenance_rejects_wrong_loaded_shared_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "moist"
    package.mkdir()
    extension = package / "_libmoist.so"
    declared_shared = tmp_path / "declared" / "libmoist.so.0"
    loaded_shared = tmp_path / "loaded" / "libmoist.so.0"
    declared_shared.parent.mkdir()
    loaded_shared.parent.mkdir()
    extension.write_bytes(b"extension")
    declared_shared.write_bytes(b"declared-library")
    loaded_shared.write_bytes(b"loaded-library")
    runtime = _real_provenance_for_files(extension, declared_shared)

    structure = type("Structure", (), {})
    cavity = type("IsodensityDROPCavity", (), {})
    library = SimpleNamespace(get_api_version=lambda: MOIST_PINNED_C_API_VERSION)
    interface = SimpleNamespace(
        Structure=structure,
        IsodensityDROPCavity=cavity,
    )
    canonical = SimpleNamespace(
        __name__="moist",
        __file__=str(package / "__init__.py"),
        __version__=MOIST_PINNED_PYTHON_VERSION,
        library=library,
        Structure=structure,
        IsodensityDROPCavity=cavity,
    )
    modules = {
        "moist": canonical,
        "moist.interface": interface,
        "moist.library": library,
    }
    monkeypatch.setattr(
        moist_drop_module.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        moist_drop_module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(extension)),
    )
    monkeypatch.setattr(
        moist_drop_module,
        "_sha256_python_package",
        lambda _root: MOIST_PINNED_PYTHON_PACKAGE_SHA256,
    )
    monkeypatch.setattr(
        moist_drop_module,
        "loaded_moist_shared_library_path",
        lambda: loaded_shared.resolve(),
    )

    with pytest.raises(RuntimeError, match="shared library"):
        runtime.verify_module(canonical)


@pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_MOIST_REAL") != "1",
    reason="requires the separately built pinned MOIST runtime",
)
def test_real_pinned_moist_hydrogen_isodensity_canary() -> None:
    import moist
    import moist._libmoist as extension

    shared_library = Path(os.environ["MAPLE_ROUTE2_MOIST_LIBRARY"])
    extension_path = Path(extension.__file__)
    runtime = MoistRuntimeProvenance(
        evidence_kind="real-pinned-build",
        python_extension_path=str(extension_path),
        python_extension_sha256=_sha256(extension_path),
        shared_library_path=str(shared_library),
        shared_library_sha256=_sha256(shared_library),
        build_toolchain=(
            "gfortran-11.4.0",
            "meson-1.11.2",
            "ninja-1.13.0",
            "openmp-enabled",
        ),
    )
    assert loaded_moist_shared_library_path() == shared_library.resolve()
    root = Path(__file__).resolve().parents[2]
    asset = load_atomic_reference_density_asset(
        table_path=(
            root / "docs/implicit-solvation/benchmarks/"
            "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
        ),
        manifest_path=(
            root / "docs/implicit-solvation/benchmarks/"
            "route2-rhodrop-atomic-reference-gaussian-mixture-v1.json"
        ),
    )
    level_set = ReconstructedMacePolarDensityLevelSet(
        asset,
        np.array([1]),
        np.zeros((1, 3)),
        np.zeros((1, 4)),
        1.0e-3,
        expected_total_charge_e=0.0,
    )

    def build():
        return MoistDropAdapter(
            level_set,
            np.array([1]),
            np.zeros((1, 3)),
            runtime=runtime,
            moist_module=moist,
        ).build_surface()

    first = build()
    second = build()
    snapshot = first.snapshot
    assert snapshot.surface_size == 194
    assert snapshot.snapshot_sha256 == second.snapshot.snapshot_sha256
    assert np.max(np.abs(snapshot.amat - snapshot.amat.T)) == 0.0
    assert snapshot.area_bohr2 == pytest.approx(125.15643937196013, abs=1.0e-10)
    assert snapshot.volume_bohr3 == pytest.approx(131.65992765171598, abs=1.0e-10)

    rng = np.random.default_rng(20260808)
    weights = first.contract_amat_surface_weights(
        rng.normal(size=snapshot.surface_size),
        rng.normal(size=snapshot.surface_size),
    )
    lsf = first.contract_surface_lsf_weights(weights)
    assert np.all(np.isfinite(lsf.value))
    assert np.all(np.isfinite(lsf.gradient))
    assert np.all(np.isfinite(lsf.hessian))
