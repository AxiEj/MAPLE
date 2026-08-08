from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256,
    ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256,
    ATOMIC_REFERENCE_DENSITY_TABLE_SHA256,
    AtomicReferenceDensityAsset,
    GaussianMixtureAtom,
    load_atomic_reference_density_asset,
)
from maple.function.calculator.extra_correction.implicit.route2_moist_drop import (
    MOIST_PINNED_C_API_VERSION,
    MOIST_PINNED_COMMIT,
    MOIST_PINNED_IMPORT_PATCH_REPO_PATH,
    MOIST_PINNED_IMPORT_PATCH_SHA256,
    MOIST_PINNED_PYTHON_VERSION,
    MOIST_PINNED_SOURCE_VERSION,
    MoistRuntimeProvenance,
)
from maple.function.calculator.extra_correction.implicit.route2_electronic_model import (
    ATOMIC_L1_SOURCE_SPACE,
    FIELD_CONDITIONED_OPERATIONAL_ENERGY,
    Route2ElectronicModelCapabilities,
    Route2ElectronicModelDescriptor,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
)
from maple.function.calculator.extra_correction.implicit.route2_rhodrop_profile import (
    RHODROP_CPCM_FULL_FUNCTIONAL_PROFILE_ID,
    RHODROP_CPCM_OPERATIONAL_PROFILE_ID,
    ROUTE2_RHODROP_CPCM_FULL_FUNCTIONAL_DRIVE_V0,
    ROUTE2_RHODROP_CPCM_OPERATIONAL_V1,
    RhoDropCPCMProfile,
    RhoDropZeroCDSResult,
)
from maple.function.route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    PCM_HALF_COUPLING_ONLY_V1,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
)
MANIFEST = TABLE.with_suffix(".json")


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


class _ConstantSourceElectronicModel:
    descriptor = Route2ElectronicModelDescriptor(
        adapter_name="rho-drop-integration-test-adapter-v1",
        model_family="rho-drop-integration-test-model",
        field_evaluator="local-reaction-potential-jet-v1",
        source_space=ATOMIC_L1_SOURCE_SPACE,
        capabilities=Route2ElectronicModelCapabilities(
            state_projectors=frozenset({"local-jet"}),
        ),
        energy_semantics=FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        profile_binding="rho-drop-integration-test-v1",
    )

    def __init__(self, source: np.ndarray) -> None:
        self.source = np.asarray(source, dtype=float)
        self.cache_identity = id(self)
        self.drives = []

    def cached_state(self, _atoms, *, require_forces=False):
        assert not require_forces
        return SimpleNamespace(
            energy_ev=-1.0,
            density_coefficients=self.source.copy(),
            dipole_e_angstrom=np.zeros(3),
            fixed_field_forces_ev_per_angstrom=None,
        )

    def evaluate_state(self, atoms, drive, *, compute_forces=False):
        assert not compute_forces
        self.drives.append(drive)
        return self.cached_state(atoms), {"test_double": True}

    def preprojected_field_projector(self):
        raise NotImplementedError

    def linearize_source_response(self, _atoms, _drive):
        raise NotImplementedError

    def field_conditioned_energy_field_gradient(self, _atoms, _drive):
        raise NotImplementedError

    def source_position_vjp(self, _atoms, _drive, *, source_cotangent):
        del source_cotangent
        raise NotImplementedError


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


def test_operational_profile_freezes_provenance_and_energy_only_admission() -> None:
    profile = ROUTE2_RHODROP_CPCM_OPERATIONAL_V1
    provenance = profile.as_provenance()

    assert profile.identity == RHODROP_CPCM_OPERATIONAL_PROFILE_ID
    assert profile.nleb == 194
    assert profile.n_iso_e_per_bohr3 == 1.0e-3
    assert profile.dielectric == 80.0
    assert profile.electrostatics_only
    assert not profile.include_cds
    assert profile.energy_available
    assert not profile.force_available
    assert not profile.opt_freq_md_available
    assert not profile.literature_tolerance_parity
    assert provenance["electrostatic_energy_ledger"] == PCM_HALF_COUPLING_ONLY_V1
    assert provenance["moist_commit"] == MOIST_PINNED_COMMIT
    assert provenance["moist_import_patch_path"] == (
        MOIST_PINNED_IMPORT_PATCH_REPO_PATH
    )
    assert provenance["reference_density_table_sha256"] == (
        ATOMIC_REFERENCE_DENSITY_TABLE_SHA256
    )
    assert provenance["reference_density_manifest_sha256"] == (
        ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256
    )
    assert provenance["reference_density_content_sha256"] == (
        ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256
    )
    assert (
        "Gate-B-complete-anchor-plus-field-coordinate-VJP"
        in provenance["blocked_gates"]
    )
    assert len(profile.profile_sha256) == 64
    assert profile.profile_sha256 == ROUTE2_RHODROP_CPCM_OPERATIONAL_V1.profile_sha256


def test_profile_rejects_force_or_cds_scope_expansion() -> None:
    with pytest.raises(ValueError, match="CDS disabled"):
        replace(ROUTE2_RHODROP_CPCM_OPERATIONAL_V1, include_cds=True)
    with pytest.raises(ValueError, match="remain closed"):
        replace(ROUTE2_RHODROP_CPCM_OPERATIONAL_V1, force_available=True)
    with pytest.raises(ValueError, match="runtime tolerance"):
        replace(
            ROUTE2_RHODROP_CPCM_OPERATIONAL_V1,
            runtime_drop_tolerance=1.0e-12,
        )
    with pytest.raises(ValueError, match="exact cold replay"):
        replace(
            ROUTE2_RHODROP_CPCM_OPERATIONAL_V1,
            require_exact_cold_replay=False,
        )


def test_full_functional_profile_is_metadata_only_and_cannot_build() -> None:
    profile = ROUTE2_RHODROP_CPCM_FULL_FUNCTIONAL_DRIVE_V0
    assert profile.identity == RHODROP_CPCM_FULL_FUNCTIONAL_PROFILE_ID
    assert profile.full_functional_drive
    assert not profile.energy_available
    assert profile.as_provenance()["admission_level"] == "blocked-metadata-only"
    assert not profile.force_available
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(NotImplementedError, match="metadata-only"):
        profile.build_reaction_field(
            atoms,
            asset=_asset(),
            runtime=_synthetic_runtime(),
            expected_total_charge_e=0.0,
        )
    with pytest.raises(NotImplementedError, match="metadata-only"):
        profile.build_engine(
            asset=_asset(),
            runtime=_synthetic_runtime(),
            total_charge_e=0.0,
        )


def test_operational_profile_recomputes_asset_content_identity() -> None:
    official = _asset()
    mixtures = dict(official.mixtures_by_atomic_number)
    hydrogen = mixtures[1]
    exponents = np.array(hydrogen.gaussian_exponents_bohr2, copy=True)
    exponents[0] *= 2.0
    mixtures[1] = GaussianMixtureAtom(
        electron_counts=hydrogen.electron_counts,
        gaussian_exponents_bohr2=exponents,
    )
    forged = AtomicReferenceDensityAsset(
        mixtures_by_atomic_number=mixtures,
        table_sha256="0" * 64,
        manifest_sha256="1" * 64,
        provenance={"purpose": "forgery-regression"},
    )
    # Even a caller bypassing frozen-dataclass fields cannot make the profile
    # trust self-reported identities; _verify_asset recomputes the mixtures.
    object.__setattr__(
        forged,
        "table_sha256",
        ATOMIC_REFERENCE_DENSITY_TABLE_SHA256,
    )
    object.__setattr__(
        forged,
        "manifest_sha256",
        ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256,
    )
    object.__setattr__(
        forged,
        "content_sha256",
        ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256,
    )

    with pytest.raises(ValueError, match="frozen v1 scientific content"):
        ROUTE2_RHODROP_CPCM_OPERATIONAL_V1.build_engine(
            asset=forged,
            runtime=_synthetic_runtime(),
            total_charge_e=0.0,
        )


def test_operational_profile_rejects_synthetic_runtime_and_builds_zero_cds_leaf() -> (
    None
):
    atoms = Atoms("OH2", positions=np.zeros((3, 3)))
    profile = ROUTE2_RHODROP_CPCM_OPERATIONAL_V1
    with pytest.raises(RuntimeError, match="not profile evidence"):
        profile.build_engine(
            asset=_asset(),
            runtime=_synthetic_runtime(),
            total_charge_e=0.0,
        )
    with pytest.raises(RuntimeError, match="not profile evidence"):
        profile.build_reaction_field(
            atoms,
            asset=_asset(),
            runtime=_synthetic_runtime(),
            expected_total_charge_e=0.0,
        )

    zero = RhoDropZeroCDSResult.for_atoms(atoms)
    assert zero.energy_hartree == 0.0
    np.testing.assert_array_equal(
        zero.position_gradient_hartree_per_angstrom,
        np.zeros((3, 3)),
    )
    assert not zero.position_gradient_hartree_per_angstrom.flags.writeable


def test_profile_engine_settings_bind_charge_without_opening_force_semantics() -> None:
    profile = ROUTE2_RHODROP_CPCM_OPERATIONAL_V1
    settings = profile.engine_settings(total_charge_e=-1.0)
    assert settings.scf_total_charge_e == -1.0
    assert settings.continuum_label == "rho-DROP/MOIST CPCM electrostatics-only"
    assert settings.scf_require_two_energy_samples

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    runtime = _synthetic_runtime()
    neutral_signature = profile.provider_cache_signature(
        atoms,
        runtime=runtime,
        total_charge_e=0.0,
    )
    charged_signature = profile.provider_cache_signature(
        atoms,
        runtime=runtime,
        total_charge_e=-1.0,
    )
    assert neutral_signature != charged_signature


def test_profile_identity_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        RhoDropCPCMProfile(identity="unversioned", model_drive="anything")


def test_profile_hash_binds_nested_drop_cpcm_and_level_set_numerics() -> None:
    baseline = ROUTE2_RHODROP_CPCM_OPERATIONAL_V1
    variants = (
        RhoDropCPCMProfile(
            identity=RHODROP_CPCM_OPERATIONAL_PROFILE_ID,
            model_drive=baseline.model_drive,
            maximum_condition_number_2=5.0e11,
        ),
        RhoDropCPCMProfile(
            identity=RHODROP_CPCM_OPERATIONAL_PROFILE_ID,
            model_drive=baseline.model_drive,
            validation_shell_distance_bohr=6.0e-2,
        ),
        RhoDropCPCMProfile(
            identity=RHODROP_CPCM_OPERATIONAL_PROFILE_ID,
            model_drive=baseline.model_drive,
            sigma_angstrom=1.4,
        ),
    )
    assert all(
        variant.profile_sha256 != baseline.profile_sha256 for variant in variants
    )


def test_profile_hash_covers_outer_engine_settings() -> None:
    class ModifiedEngineProfile(RhoDropCPCMProfile):
        def engine_settings(self, *, total_charge_e: float):
            return replace(
                super().engine_settings(total_charge_e=total_charge_e),
                scf_max_iterations=81,
            )

    modified = ModifiedEngineProfile(
        identity=RHODROP_CPCM_OPERATIONAL_PROFILE_ID,
        model_drive=ROUTE2_RHODROP_CPCM_OPERATIONAL_V1.model_drive,
    )
    provenance = ROUTE2_RHODROP_CPCM_OPERATIONAL_V1.as_provenance()

    assert (
        "scf_total_charge_e" not in provenance["engine_settings_without_total_charge"]
    )
    assert modified.profile_sha256 != (
        ROUTE2_RHODROP_CPCM_OPERATIONAL_V1.profile_sha256
    )


@pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_MOIST_REAL") != "1",
    reason="requires the separately built pinned MOIST runtime",
)
def test_real_profile_integrates_existing_route2_scf_energy_ledger() -> None:
    import moist

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    source = np.asarray([[0.0, 0.002, -0.001, 0.0015]])
    model = _ConstantSourceElectronicModel(source)
    profile = ROUTE2_RHODROP_CPCM_OPERATIONAL_V1
    runtime = _real_runtime()
    engine = profile.build_engine(
        asset=_asset(),
        runtime=runtime,
        total_charge_e=0.0,
        moist_module=moist,
    )
    gas_state = engine.gas_state(model, atoms, need_forces=False)
    coupled = engine.solve_coupled_state(
        atoms,
        model,
        gas_state,
        provider_cache_signature=profile.provider_cache_signature(
            atoms,
            runtime=runtime,
            total_charge_e=0.0,
        ),
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )
    components = engine.energy_components(
        gas_state,
        coupled,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )

    assert coupled.fixed_point_applicable
    assert coupled.scf_convergence["reason"] == "nominal-density-and-energy-v1"
    np.testing.assert_array_equal(coupled.density_coefficients, source)
    assert coupled.energy_identity_error_ev < 1.0e-12
    assert components["solute_polarization"] == 0.0
    assert components["cds"] == 0.0
    assert components["delta_g_solv"] == pytest.approx(
        coupled.polarization_energy_hartree,
        abs=1.0e-18,
    )
    capabilities = coupled.reaction_field_capabilities
    assert capabilities["source_dependent_geometry"]
    assert capabilities["reciprocal_energy_pairing"]
    assert not capabilities["reaction_jacobian_self_adjoint"]
    assert not capabilities["complete_position_derivative_available"]
    assert not capabilities["operational_jvp_efficiency_admitted"]
    assert capabilities["provider_state"]["forward_state_sha256"] == (
        coupled.reaction_field.scf_snapshot(source).state_sha256
    )
    assert engine.plugin_continuum_binding == profile.as_provenance()
    assert engine.required_electrostatic_energy_ledger == (PCM_HALF_COUPLING_ONLY_V1)


def test_profile_bound_engine_rejects_the_legacy_energy_ledger() -> None:
    profile = ROUTE2_RHODROP_CPCM_OPERATIONAL_V1
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: None,
        cds_evaluator=RhoDropZeroCDSResult.for_atoms,
        settings=profile.engine_settings(total_charge_e=0.0),
        required_electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )
    gas_state = SimpleNamespace(energy_ev=-1.0)
    coupled = SimpleNamespace(
        solvent_state=SimpleNamespace(energy_ev=-0.9),
        polarization_energy_hartree=-0.02,
        cds_result=SimpleNamespace(energy_hartree=0.0),
    )

    with pytest.raises(RuntimeError, match="bound to electrostatic energy ledger"):
        engine.energy_components(gas_state, coupled)
    with pytest.raises(RuntimeError, match="bound to electrostatic energy ledger"):
        engine.solve_coupled_state(
            None,
            object(),
            None,
            provider_cache_signature="ledger-rejection-must-precede-work",
        )
    with pytest.raises(RuntimeError, match="bound to electrostatic energy ledger"):
        engine.energy_components(
            gas_state,
            coupled,
            electrostatic_energy_ledger=(LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1),
        )
    components = engine.energy_components(
        gas_state,
        coupled,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )
    assert components["solute_polarization"] == 0.0
