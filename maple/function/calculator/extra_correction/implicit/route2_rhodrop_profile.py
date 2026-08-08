"""Versioned research profiles for source-dependent rho-DROP C-PCM.

Profiles bind numerical choices and admission status; they do not duplicate
the density, DROP, C-PCM, or Route-2 fixed-point implementations.  The first
profile is electrostatics-only and drives MACE-POLAR with the conventional
reaction-potential jet.  The full-functional drive is represented only as a
blocked metadata profile until the full PCM source-gradient drive is wired and
validated.  Analytic forces for that future drive additionally require
second-order continuum response.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import ModuleType
from typing import Any

import numpy as np

from ....route2_energy_ledger import PCM_HALF_COUPLING_ONLY_V1
from .route2_atomic_reference_density import (
    ATOMIC_REFERENCE_DENSITY_ARTIFACT,
    ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256,
    ATOMIC_REFERENCE_DENSITY_GENERATOR_SHA256,
    ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256,
    ATOMIC_REFERENCE_DENSITY_TABLE_SHA256,
    AtomicReferenceDensityAsset,
)
from .route2_engine import Route2ContinuumEngine, Route2EngineSettings
from .route2_fixed_point import SAFEGUARDED_ANDERSON_SOLVER
from .route2_moist_drop import (
    MOIST_DROP_ADAPTER_CONTRACT_VERSION,
    MOIST_PINNED_COMMIT,
    MOIST_PINNED_COMPILED_DROP_TOLERANCE,
    MOIST_PINNED_IMPORT_PATCH_REPO_PATH,
    MOIST_PINNED_IMPORT_PATCH_SHA256,
    MOIST_PINNED_SOURCE_VERSION,
    MOIST_UPSTREAM_REPOSITORY,
    MoistDropSettings,
    MoistRuntimeProvenance,
)
from .route2_rhodrop_cpcm import (
    RHODROP_CPCM_FORCE_STATUS,
    RHODROP_CPCM_FORWARD_CONTRACT_VERSION,
    RHODROP_CPCM_JVP_IMPLEMENTATION,
    RHODROP_CPCM_MODEL_DRIVE,
    RHODROP_CPCM_OPERATIONAL_PROFILE_KIND,
    RhoDropCPCMReactionField,
    RhoDropCPCMSettings,
)

RHODROP_CPCM_OPERATIONAL_PROFILE_ID = RHODROP_CPCM_OPERATIONAL_PROFILE_KIND
RHODROP_CPCM_FULL_FUNCTIONAL_PROFILE_ID = "route2-rhodrop-cpcm-full-functional-drive-v0"
RHODROP_LITERATURE_DROP_TOLERANCE = 1.0e-12


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class RhoDropZeroCDSResult:
    """Explicit zero nonpolar leaf for an electrostatics-only profile."""

    position_gradient_hartree_per_angstrom: np.ndarray
    energy_hartree: float = 0.0

    def __post_init__(self) -> None:
        gradient = np.asarray(
            self.position_gradient_hartree_per_angstrom,
            dtype=float,
        )
        if (
            gradient.ndim != 2
            or gradient.shape[0] == 0
            or gradient.shape[1] != 3
            or not np.all(np.isfinite(gradient))
        ):
            raise ValueError("Zero-CDS gradient must have shape (n_atoms, 3).")
        if float(self.energy_hartree) != 0.0:
            raise ValueError("The electrostatics-only CDS leaf must be exactly zero.")
        immutable = np.array(gradient, copy=True)
        immutable.setflags(write=False)
        object.__setattr__(
            self,
            "position_gradient_hartree_per_angstrom",
            immutable,
        )
        object.__setattr__(self, "energy_hartree", 0.0)

    @classmethod
    def for_atoms(cls, atoms: Any) -> "RhoDropZeroCDSResult":
        return cls(np.zeros((len(atoms), 3), dtype=float))


@dataclass(frozen=True)
class RhoDropCPCMProfile:
    """Immutable numerical/provenance/admission profile."""

    identity: str
    model_drive: str
    n_iso_e_per_bohr3: float = 1.0e-3
    nleb: int = 194
    wleb_prune_level: int = 0
    dielectric: float = 80.0
    sigma_angstrom: float = 1.5
    minimum_density_e_per_bohr3: float = 0.0
    electron_count_tolerance: float = 5.0e-12
    linear_solve_relative_tolerance: float = 5.0e-12
    linear_solve_absolute_tolerance: float = 1.0e-13
    energy_identity_relative_tolerance: float = 1.0e-10
    energy_identity_absolute_tolerance_hartree: float = 1.0e-12
    maximum_condition_number_2: float = 1.0e12
    require_exact_cold_replay: bool = True
    drop_debug: bool = False
    drop_verbosity: int = 0
    drop_do_fine: bool = False
    callback_scale: float = 1.0
    surface_tolerance: float = 1.0e-9
    minimum_gradient_norm_bohr: float = 1.0e-10
    validation_shell_distance_bohr: float = 5.0e-2
    literature_drop_tolerance: float = RHODROP_LITERATURE_DROP_TOLERANCE
    runtime_drop_tolerance: float = MOIST_PINNED_COMPILED_DROP_TOLERANCE
    electrostatics_only: bool = True
    include_cds: bool = False
    energy_available: bool = True
    force_available: bool = False
    opt_freq_md_available: bool = False
    full_functional_drive: bool = False

    def __post_init__(self) -> None:
        if self.identity not in {
            RHODROP_CPCM_OPERATIONAL_PROFILE_ID,
            RHODROP_CPCM_FULL_FUNCTIONAL_PROFILE_ID,
        }:
            raise ValueError("Unsupported rho-DROP profile identity.")
        if not math.isfinite(self.n_iso_e_per_bohr3) or self.n_iso_e_per_bohr3 <= 0.0:
            raise ValueError("rho-DROP profile n_iso must be positive.")
        if not isinstance(self.nleb, int) or self.nleb <= 0:
            raise ValueError("rho-DROP profile nleb must be a positive integer.")
        if not isinstance(self.wleb_prune_level, int) or self.wleb_prune_level != 0:
            raise ValueError("rho-DROP v1 freezes wleb_prune_level at zero.")
        if not math.isfinite(self.dielectric) or self.dielectric <= 1.0:
            raise ValueError("rho-DROP profile dielectric must exceed one.")
        if not math.isfinite(self.sigma_angstrom) or self.sigma_angstrom <= 0.0:
            raise ValueError("rho-DROP profile residual width must be positive.")
        if (
            not math.isfinite(self.minimum_density_e_per_bohr3)
            or self.minimum_density_e_per_bohr3 < 0.0
        ):
            raise ValueError("rho-DROP profile minimum density must be nonnegative.")
        if (
            not math.isfinite(self.electron_count_tolerance)
            or self.electron_count_tolerance <= 0.0
        ):
            raise ValueError(
                "rho-DROP profile electron-count tolerance must be positive."
            )
        if self.literature_drop_tolerance != RHODROP_LITERATURE_DROP_TOLERANCE:
            raise ValueError("rho-DROP profile literature tolerance changed.")
        if self.runtime_drop_tolerance != MOIST_PINNED_COMPILED_DROP_TOLERANCE:
            raise ValueError("rho-DROP profile runtime tolerance is not pinned.")
        if not self.require_exact_cold_replay:
            raise ValueError("Versioned rho-DROP profiles require exact cold replay.")
        if not self.electrostatics_only or self.include_cds:
            raise ValueError("rho-DROP v1 is electrostatics-only with CDS disabled.")
        if self.force_available or self.opt_freq_md_available:
            raise ValueError("rho-DROP forces, OPT, FREQ, and MD remain closed.")
        if self.full_functional_drive:
            if self.identity != RHODROP_CPCM_FULL_FUNCTIONAL_PROFILE_ID:
                raise ValueError("Full-functional drive requires its v0 profile.")
            if self.model_drive != "full-pcm-source-gradient":
                raise ValueError("Full-functional profile drive identity is invalid.")
            if self.energy_available:
                raise ValueError(
                    "The metadata-only full-functional profile cannot advertise "
                    "an available energy implementation."
                )
        elif (
            self.identity != RHODROP_CPCM_OPERATIONAL_PROFILE_ID
            or self.model_drive != RHODROP_CPCM_MODEL_DRIVE
        ):
            raise ValueError("Operational rho-DROP profile drive identity is invalid.")
        elif not self.energy_available:
            raise ValueError(
                "The operational rho-DROP profile must retain its admitted "
                "energy-only capability."
            )
        # Construct both nested settings objects during validation so every
        # scientific/numerical leaf is checked by its owning implementation.
        self.make_cpcm_settings()
        self.make_drop_settings()

    def make_cpcm_settings(self) -> RhoDropCPCMSettings:
        return RhoDropCPCMSettings(
            dielectric=self.dielectric,
            linear_solve_relative_tolerance=self.linear_solve_relative_tolerance,
            linear_solve_absolute_tolerance=self.linear_solve_absolute_tolerance,
            energy_identity_relative_tolerance=self.energy_identity_relative_tolerance,
            energy_identity_absolute_tolerance_hartree=(
                self.energy_identity_absolute_tolerance_hartree
            ),
            maximum_condition_number_2=self.maximum_condition_number_2,
            require_exact_cold_replay=self.require_exact_cold_replay,
        )

    def make_drop_settings(self) -> MoistDropSettings:
        return MoistDropSettings(
            nleb=self.nleb,
            wleb_prune_level=self.wleb_prune_level,
            callback_scale=self.callback_scale,
            compiled_drop_tolerance=self.runtime_drop_tolerance,
            debug=self.drop_debug,
            verbosity=self.drop_verbosity,
            do_fine=self.drop_do_fine,
            surface_tolerance=self.surface_tolerance,
            minimum_gradient_norm_bohr=self.minimum_gradient_norm_bohr,
            validation_shell_distance_bohr=self.validation_shell_distance_bohr,
        )

    @property
    def profile_sha256(self) -> str:
        return _sha256_json(self.as_provenance())

    @property
    def literature_tolerance_parity(self) -> bool:
        return self.runtime_drop_tolerance == self.literature_drop_tolerance

    def as_provenance(self) -> dict[str, object]:
        engine_settings = asdict(self.engine_settings(total_charge_e=0.0))
        zero_charge = engine_settings.pop("scf_total_charge_e")
        charged_engine_settings = asdict(self.engine_settings(total_charge_e=1.0))
        unit_charge = charged_engine_settings.pop("scf_total_charge_e")
        if (
            zero_charge != 0.0
            or unit_charge != 1.0
            or charged_engine_settings != engine_settings
        ):
            raise RuntimeError(
                "rho-DROP engine settings may depend on total charge only through "
                "the explicitly bound scf_total_charge_e field."
            )
        return {
            **asdict(self),
            "continuum_backend": "MOIST",
            "continuum_equation": "CPCM",
            "moist_drop_adapter_contract_version": (
                MOIST_DROP_ADAPTER_CONTRACT_VERSION
            ),
            "rhodrop_cpcm_forward_contract_version": (
                RHODROP_CPCM_FORWARD_CONTRACT_VERSION
            ),
            "drop_settings_sha256": self.make_drop_settings().identity_sha256,
            "cpcm_settings_sha256": self.make_cpcm_settings().identity_sha256,
            "engine_settings_without_total_charge": engine_settings,
            "engine_settings_without_total_charge_sha256": _sha256_json(
                engine_settings
            ),
            "engine_total_charge_binding": (
                "build-argument-plus-provider-cache-signature-v1"
            ),
            "cavity_policy": "reconstructed-MACE-polar-density-DROP-v1",
            "electrostatic_energy_ledger": PCM_HALF_COUPLING_ONLY_V1,
            "moist_upstream_repository": MOIST_UPSTREAM_REPOSITORY,
            "moist_commit": MOIST_PINNED_COMMIT,
            "moist_source_version": MOIST_PINNED_SOURCE_VERSION,
            "moist_import_patch_path": MOIST_PINNED_IMPORT_PATCH_REPO_PATH,
            "moist_import_patch_sha256": MOIST_PINNED_IMPORT_PATCH_SHA256,
            "reference_density_artifact": ATOMIC_REFERENCE_DENSITY_ARTIFACT,
            "reference_density_table_sha256": (ATOMIC_REFERENCE_DENSITY_TABLE_SHA256),
            "reference_density_manifest_sha256": (
                ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256
            ),
            "reference_density_generator_sha256": (
                ATOMIC_REFERENCE_DENSITY_GENERATOR_SHA256
            ),
            "reference_density_content_sha256": (
                ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256
            ),
            "literature_tolerance_parity": self.literature_tolerance_parity,
            "jvp_implementation": RHODROP_CPCM_JVP_IMPLEMENTATION,
            "operational_jvp_efficiency_admitted": False,
            "analytic_force_status": RHODROP_CPCM_FORCE_STATUS,
            "admission_level": (
                "research-energy-only"
                if self.energy_available
                else "blocked-metadata-only"
            ),
            "blocked_gates": [
                "QM-isodensity-surface-benchmark",
                "MOIST-public-1e-12-DROP-tolerance",
                "efficient-forward-mode-DROP-JVP",
                "Gate-B-complete-anchor-plus-field-coordinate-VJP",
                "Gate-C-end-to-end-force-PES-validation",
                "full-functional-source-gradient-drive-wiring-and-validation",
                "full-functional-force-second-order-response",
            ],
        }

    def _verify_asset(self, asset: AtomicReferenceDensityAsset) -> None:
        if not isinstance(asset, AtomicReferenceDensityAsset):
            raise TypeError("rho-DROP profile requires a verified density asset.")
        asset.require_frozen_v1_identity()
        if (
            asset.artifact != ATOMIC_REFERENCE_DENSITY_ARTIFACT
            or asset.table_sha256 != ATOMIC_REFERENCE_DENSITY_TABLE_SHA256
            or asset.manifest_sha256 != ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256
            or asset.content_sha256 != ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256
        ):
            raise ValueError("rho-DROP profile reference-density asset is not pinned.")

    @staticmethod
    def _verify_runtime(runtime: MoistRuntimeProvenance) -> None:
        if not isinstance(runtime, MoistRuntimeProvenance):
            raise TypeError("rho-DROP profile requires MOIST runtime provenance.")
        if runtime.evidence_kind != "real-pinned-build":
            raise RuntimeError("Synthetic MOIST runtimes are not profile evidence.")

    def build_reaction_field(
        self,
        atoms: Any,
        *,
        asset: AtomicReferenceDensityAsset,
        runtime: MoistRuntimeProvenance,
        expected_total_charge_e: float,
        moist_module: ModuleType | Any | None = None,
    ) -> RhoDropCPCMReactionField:
        """Build only the implemented conventional-drive energy profile."""

        if self.full_functional_drive:
            raise NotImplementedError(
                "The full PCM source-derivative model drive is metadata-only "
                "until the source-gradient drive is wired and validated."
            )
        self._verify_asset(asset)
        self._verify_runtime(runtime)
        return RhoDropCPCMReactionField(
            asset,
            np.asarray(atoms.numbers, dtype=int),
            np.asarray(atoms.get_positions(), dtype=float),
            expected_total_charge_e=expected_total_charge_e,
            runtime=runtime,
            n_iso_e_per_bohr3=self.n_iso_e_per_bohr3,
            cpcm_settings=self.make_cpcm_settings(),
            drop_settings=self.make_drop_settings(),
            sigma_angstrom=self.sigma_angstrom,
            minimum_density_e_per_bohr3=self.minimum_density_e_per_bohr3,
            electron_count_tolerance=self.electron_count_tolerance,
            moist_module=moist_module,
        )

    def provider_cache_signature(
        self,
        atoms: Any,
        *,
        runtime: MoistRuntimeProvenance,
        total_charge_e: float,
    ) -> str:
        numbers = np.ascontiguousarray(np.asarray(atoms.numbers, dtype="<i8"))
        positions = np.ascontiguousarray(np.asarray(atoms.get_positions(), dtype="<f8"))
        if positions.shape != (numbers.size, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("rho-DROP cache geometry is invalid.")
        charge = float(total_charge_e)
        if not math.isfinite(charge):
            raise ValueError("rho-DROP cache total charge must be finite.")
        digest = hashlib.sha256()
        digest.update(numbers.tobytes())
        digest.update(positions.tobytes())
        digest.update(np.asarray([charge], dtype="<f8").tobytes())
        digest.update(self.profile_sha256.encode("ascii"))
        digest.update(runtime.identity_sha256.encode("ascii"))
        return digest.hexdigest()

    def engine_settings(self, *, total_charge_e: float) -> Route2EngineSettings:
        charge = float(total_charge_e)
        if not math.isfinite(charge):
            raise ValueError("rho-DROP Route-2 total charge must be finite.")
        return Route2EngineSettings(
            continuum_label="rho-DROP/MOIST CPCM electrostatics-only",
            scf_mixing=1.0,
            scf_density_tolerance=2.0e-12,
            scf_dipole_tolerance_e_angstrom=2.0e-12,
            scf_energy_tolerance_ev=2.0e-12,
            scf_max_iterations=80,
            adjoint_relative_tolerance=1.0e-10,
            adjoint_absolute_tolerance=1.0e-13,
            adjoint_max_iterations=100,
            energy_identity_tolerance_ev=1.0e-10,
            force_state_energy_tolerance_ev=1.0e-10,
            neutral_density_tolerance=1.0e-8,
            scf_solver=SAFEGUARDED_ANDERSON_SOLVER,
            scf_anderson_depth=6,
            scf_anderson_regularization=1.0e-12,
            scf_anderson_coefficient_l1_limit=100.0,
            scf_anderson_step_ratio_limit=100.0,
            scf_anderson_residual_growth_limit=2.0,
            scf_total_charge_e=charge,
            scf_raw_response_charge_tolerance_e=2.0e-12,
            scf_total_charge_residual_tolerance_e=2.0e-12,
            scf_molecular_dipole_tolerance_e_angstrom=2.0e-12,
            scf_require_two_energy_samples=True,
        )

    def build_engine(
        self,
        *,
        asset: AtomicReferenceDensityAsset,
        runtime: MoistRuntimeProvenance,
        total_charge_e: float,
        moist_module: ModuleType | Any | None = None,
    ) -> Route2ContinuumEngine:
        """Compose the existing provider-neutral Route-2 engine."""

        if self.full_functional_drive or not self.energy_available:
            raise NotImplementedError(
                "The full PCM source-derivative model drive is metadata-only "
                "until its experimental source-gradient drive is wired and "
                "validated."
            )

        self._verify_asset(asset)
        self._verify_runtime(runtime)

        def reaction_field_factory(atoms: Any) -> RhoDropCPCMReactionField:
            return self.build_reaction_field(
                atoms,
                asset=asset,
                runtime=runtime,
                expected_total_charge_e=total_charge_e,
                moist_module=moist_module,
            )

        return Route2ContinuumEngine(
            reaction_field_factory=reaction_field_factory,
            cds_evaluator=RhoDropZeroCDSResult.for_atoms,
            settings=self.engine_settings(total_charge_e=total_charge_e),
            plugin_continuum_binding=self.as_provenance(),
            required_electrostatic_energy_ledger=(PCM_HALF_COUPLING_ONLY_V1),
        )


ROUTE2_RHODROP_CPCM_OPERATIONAL_V1 = RhoDropCPCMProfile(
    identity=RHODROP_CPCM_OPERATIONAL_PROFILE_ID,
    model_drive=RHODROP_CPCM_MODEL_DRIVE,
)

ROUTE2_RHODROP_CPCM_FULL_FUNCTIONAL_DRIVE_V0 = RhoDropCPCMProfile(
    identity=RHODROP_CPCM_FULL_FUNCTIONAL_PROFILE_ID,
    model_drive="full-pcm-source-gradient",
    energy_available=False,
    full_functional_drive=True,
)


__all__ = [
    "RHODROP_CPCM_FULL_FUNCTIONAL_PROFILE_ID",
    "RHODROP_CPCM_OPERATIONAL_PROFILE_ID",
    "RHODROP_LITERATURE_DROP_TOLERANCE",
    "ROUTE2_RHODROP_CPCM_FULL_FUNCTIONAL_DRIVE_V0",
    "ROUTE2_RHODROP_CPCM_OPERATIONAL_V1",
    "RhoDropCPCMProfile",
    "RhoDropZeroCDSResult",
]
