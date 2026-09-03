"""Versioned Route-2 SMD profile registry.

This lightweight module is the shared contract between input validation,
electronic-model dispatch, and continuum providers.  A public profile selects
one complete, reproducible combination of electronic-model family, source
space, field evaluator, continuum backend, cavity, energy ledger, and
nonpolar functional; users cannot compose unvalidated mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from .route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1,
    PCM_HALF_COUPLING_ONLY_V1,
    Route2ElectrostaticEnergyLedger,
    validate_route2_electrostatic_energy_ledger,
)
from .route2_solvents import SUPPORTED_ROUTE2_SMD_SOLVENTS
from .route2_model_contracts import (
    ROUTE2_ATOMIC_L1_SOURCE_SPACE,
    ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY,
    ROUTE2_KNOWN_NONPASSIVE_ENERGY_CONJUGATE_DIAGNOSTIC,
    ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
    ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING,
    ROUTE2_MACE_POLAR_MODEL_FAMILY,
    ROUTE2_MACE_POLAR_PROFILE_BINDING,
)

CANONICAL_SMD_PROFILE = "smd-iefpcm"
GAFF2_CARBONYL_O_PROFILE = "smd-iefpcm-gaff2-o"
PCMSOLVER_EXACT_GTO_FIELD_PROFILE = (
    "smd-iefpcm-point-l1-exact-gto-v1"
)
PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE = (
    "smd-iefpcm-point-l1-local-jet-atomic-mean-v1"
)
PCMSOLVER_INTRINSIC_CAVITY_PROFILE = (
    "macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-v1"
)
PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE = (
    "macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-exact-gto-v1"
)
PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE = (
    "macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-exact-gto-"
    "pcm-half-coupling-v1"
)
DDPCM_SMD_PROFILE = "smd-ddpcm-l15-n1202-v1"
DDPCM_SMD_DIRECT_PCM_PROFILE = "smd-ddpcm-l15-n1202-pcm-half-coupling-v1"
DDPCM_MULTISOLVENT_SMD_PROFILE = "smd-ddpcm-l15-n1202-multisolv-v1"
DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE = (
    "smd-ddpcm-l15-n1202-multisolv-pcm-half-coupling-v1"
)
DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE = (
    "smd-ddpcm-l15-n1202-multisolv-pcm-half-coupling-v2"
)
DDCOSMO_MULTISOLVENT_SMD_PROFILE = (
    "smd-ddcosmo-l15-n1202-multisolv-v1"
)
DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE = (
    "smd-ddcosmo-l15-n1202-multisolv-pcm-half-coupling-v1"
)
DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE = (
    "smd-ddcosmo-l15-n1202-multisolv-pcm-half-coupling-v2"
)
FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE = (
    "smd-cpcm-fc-aswig-jgp94-aqueous-pcm-half-coupling-v1"
)
FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE = (
    "smd-cpcm-fc-aswig-jgp94-d2-mace-aqueous-pcm-half-coupling-v2"
)
FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE = (
    "smd-cpcm-fc-aswig-jgp94-d2-mace-aqueous-pcm-half-coupling-force-v3"
)
DDPCM_GAFF2_CARBONYL_O_PROFILE = "smd-ddpcm-l15-n1202-gaff2-o-v1"
DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE = (
    "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1"
)
DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE = (
    "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-omp4-v1"
)
MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE = (
    "mace-polar-ef-v2-smooth-ddpcm-l3-p6-r96-128-128-"
    "water-known-nonpassive-v1"
)
MACE_POLAR_EF_SMOOTH_PCM_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE = (
    "mace-polar-ef-v2-smooth-ddpcm-l3-p6-r96-128-128-"
    "multisolv-derivatives-known-nonpassive-v2"
)
MACE_POLAR_EF_SMOOTH_COSMO_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE = (
    "mace-polar-ef-v2-smooth-cosmo-l3-p6-r96-128-128-"
    "multisolv-derivatives-known-nonpassive-v1"
)

MACEPOL_MOLECULAR_REALSPACE_PROFILE = (
    "graph-longrange-molecular-realspace-v1"
)
MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE = (
    "graph-longrange-forced-periodic-fixed-box40-v1"
)


@dataclass(frozen=True)
class Route2SMDProfileSpec:
    """One frozen audited combination, not an open run-time plug-in schema."""

    name: str
    provider: Literal[
        "pcmsolver",
        "pyddx",
        "fc-aswig",
        "torch-smooth-pcm",
        "torch-smooth-cosmo",
    ]
    cavity: Literal[
        "canonical-smd",
        "gaff2-carbonyl-o",
        "fixed-topology-smd",
        "smooth-fixed-dimensional-spheres-v1",
    ]
    mace_long_range_evaluator: str
    electrostatics_model: Literal[
        "iefpcm",
        "ddpcm",
        "ddcosmo",
        "cpcm",
        "smooth-ddpcm",
        "smooth-cosmo",
    ]
    solute_source: Literal["point-multipole-l1"]
    reaction_field_projector: Literal[
        "local-jet",
        "exact-gto-v1",
        "native-atomwise-potential-gradient-v1",
    ]
    model_field_gauge: Literal[
        "continuum-zero-at-infinity",
        "atomic-center-mean-zero-v1",
    ]
    nonpolar_model: Literal[
        "native-water-smd-cds",
        "pyscf-smd-cds",
        "fixed-topology-aqueous-smd-cds",
        "none",
    ]
    dielectric_policy: Literal[
        "pcmsolver-water-keyword",
        "explicit-smd-water-78.355-v1",
        "legacy-water-78.39",
        "pyscf-smd-2.13.1",
        "conductor-infinity-binary64-v1",
    ]
    coulomb_radii_policy: Literal[
        "smd-water-reference-smd18-v1",
        "pyscf-smd-2.13.1",
    ]
    supported_solvents: frozenset[str]
    electronic_model_family: str = ROUTE2_MACE_POLAR_MODEL_FAMILY
    electronic_source_space: str = ROUTE2_ATOMIC_L1_SOURCE_SPACE
    electronic_profile_binding: str = ROUTE2_MACE_POLAR_PROFILE_BINDING
    electronic_energy_semantics: str = (
        ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY
    )
    mace_geometry_frame_policy: Literal[
        "laboratory-v1",
        "jgp94-d2-canonical-v1",
    ] = "laboratory-v1"
    electrostatic_energy_ledger: Route2ElectrostaticEnergyLedger = (
        LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
    )
    strict_original_smd_equivalence: bool = False
    ddpcm_n_proc: int = 1
    default_eligible: bool = False
    force_release_eligible: bool = False
    known_nonpassive_diagnostic: bool = False
    diagnostic_derivative_eligible: bool = False
    pcmsolver_cavity_generation: Literal[
        "legacy-builtin-solvent-probe-v1",
        "intrinsic-probe0-noaddsph-v1",
    ] | None = None

    @property
    def uses_gaff2_carbonyl_oxygen(self) -> bool:
        return self.cavity == "gaff2-carbonyl-o"

    @property
    def model_field_evaluator(self) -> str:
        """Model-neutral alias consumed by the electronic adapter boundary."""

        return self.mace_long_range_evaluator

    def __post_init__(self) -> None:
        validate_route2_electrostatic_energy_ledger(
            self.electrostatic_energy_ledger
        )
        if not str(self.electronic_model_family).strip():
            raise ValueError("Route-2 electronic-model family must be non-empty.")
        if not str(self.electronic_source_space).strip():
            raise ValueError("Route-2 electronic source space must be non-empty.")
        if not str(self.electronic_profile_binding).strip():
            raise ValueError("Route-2 electronic profile binding must be non-empty.")
        if not str(self.model_field_evaluator).strip():
            raise ValueError("Route-2 model-field evaluator must be non-empty.")
        if not str(self.electronic_energy_semantics).strip():
            raise ValueError("Route-2 electronic energy semantics must be non-empty.")
        if self.mace_geometry_frame_policy not in {
            "laboratory-v1",
            "jgp94-d2-canonical-v1",
        }:
            raise ValueError("Unsupported Route-2 MACE geometry-frame policy.")
        if self.force_release_eligible and (
            self.provider != "fc-aswig"
            or self.cavity != "fixed-topology-smd"
            or self.electrostatics_model != "cpcm"
            or self.electronic_model_family != ROUTE2_MACE_POLAR_MODEL_FAMILY
            or self.electronic_profile_binding
            != ROUTE2_MACE_POLAR_PROFILE_BINDING
            or self.reaction_field_projector != "local-jet"
            or self.mace_geometry_frame_policy != "jgp94-d2-canonical-v1"
            or self.electrostatic_energy_ledger != PCM_HALF_COUPLING_ONLY_V1
            or self.supported_solvents != _WATER_ONLY
        ):
            raise ValueError(
                "A Route-2 public force profile must be the bounded "
                "water-only JGP94-D2 fixed-topology direct-CPCM profile."
            )
        smooth_diagnostic_pair = (
            self.provider,
            self.electrostatics_model,
        ) in {
            ("torch-smooth-pcm", "smooth-ddpcm"),
            ("torch-smooth-cosmo", "smooth-cosmo"),
        }
        if self.known_nonpassive_diagnostic and (
            not smooth_diagnostic_pair
            or self.cavity != "smooth-fixed-dimensional-spheres-v1"
            or self.electronic_energy_semantics
            != ROUTE2_KNOWN_NONPASSIVE_ENERGY_CONJUGATE_DIAGNOSTIC
            or self.reaction_field_projector
            != "native-atomwise-potential-gradient-v1"
            or self.electrostatic_energy_ledger
            != MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1
            or self.default_eligible
            or self.force_release_eligible
        ):
            raise ValueError(
                "A known-nonpassive diagnostic profile must be the exact "
                "water-only MACE-POLAR-EF-v2/smooth-ddPCM energy profile."
            )
        if self.diagnostic_derivative_eligible and (
            not self.known_nonpassive_diagnostic
            or self.supported_solvents != SUPPORTED_ROUTE2_SMD_SOLVENTS
        ):
            raise ValueError(
                "A diagnostic derivative profile must be the registered "
                "multi-solvent MACE-POLAR-EF-v2/smooth-ddPCM profile."
            )
        if (
            self.known_nonpassive_diagnostic
            and not self.diagnostic_derivative_eligible
            and self.supported_solvents != _WATER_ONLY
        ):
            raise ValueError(
                "The energy-only MACE-POLAR-EF-v2 diagnostic is water-only."
            )
        if (
            self.electronic_model_family
            == ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY
            and not self.known_nonpassive_diagnostic
        ):
            raise ValueError(
                "The current MACE-POLAR-EF-v2 checkpoint may appear only in "
                "its known-nonpassive diagnostic profile."
            )

    def supports_solvent(self, solvent: str) -> bool:
        return str(solvent).strip().lower() in self.supported_solvents

    @property
    def uses_intrinsic_pcmsolver_cavity(self) -> bool:
        return (
            self.pcmsolver_cavity_generation
            == "intrinsic-probe0-noaddsph-v1"
        )


_WATER_ONLY = frozenset({"water"})


_PROFILE_SPECS = {
    MACE_POLAR_EF_SMOOTH_COSMO_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE: (
        Route2SMDProfileSpec(
            name=(
                MACE_POLAR_EF_SMOOTH_COSMO_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE
            ),
            provider="torch-smooth-cosmo",
            cavity="smooth-fixed-dimensional-spheres-v1",
            mace_long_range_evaluator=(
                "mace-polar-ef-v2-native-atomwise-local-jet-v1"
            ),
            electrostatics_model="smooth-cosmo",
            solute_source="point-multipole-l1",
            reaction_field_projector=(
                "native-atomwise-potential-gradient-v1"
            ),
            model_field_gauge="continuum-zero-at-infinity",
            nonpolar_model="none",
            dielectric_policy="conductor-infinity-binary64-v1",
            coulomb_radii_policy="pyscf-smd-2.13.1",
            supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
            electronic_model_family=ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
            electronic_profile_binding=ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING,
            electronic_energy_semantics=(
                ROUTE2_KNOWN_NONPASSIVE_ENERGY_CONJUGATE_DIAGNOSTIC
            ),
            electrostatic_energy_ledger=(
                MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1
            ),
            known_nonpassive_diagnostic=True,
            diagnostic_derivative_eligible=True,
        )
    ),
    MACE_POLAR_EF_SMOOTH_PCM_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE: (
        Route2SMDProfileSpec(
            name=(
                MACE_POLAR_EF_SMOOTH_PCM_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE
            ),
            provider="torch-smooth-pcm",
            cavity="smooth-fixed-dimensional-spheres-v1",
            mace_long_range_evaluator=(
                "mace-polar-ef-v2-native-atomwise-local-jet-v1"
            ),
            electrostatics_model="smooth-ddpcm",
            solute_source="point-multipole-l1",
            reaction_field_projector=(
                "native-atomwise-potential-gradient-v1"
            ),
            model_field_gauge="continuum-zero-at-infinity",
            nonpolar_model="pyscf-smd-cds",
            dielectric_policy="pyscf-smd-2.13.1",
            coulomb_radii_policy="pyscf-smd-2.13.1",
            supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
            electronic_model_family=ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
            electronic_profile_binding=ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING,
            electronic_energy_semantics=(
                ROUTE2_KNOWN_NONPASSIVE_ENERGY_CONJUGATE_DIAGNOSTIC
            ),
            electrostatic_energy_ledger=(
                MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1
            ),
            known_nonpassive_diagnostic=True,
            diagnostic_derivative_eligible=True,
        )
    ),
    MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE: Route2SMDProfileSpec(
        name=MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE,
        provider="torch-smooth-pcm",
        cavity="smooth-fixed-dimensional-spheres-v1",
        mace_long_range_evaluator=(
            "mace-polar-ef-v2-native-atomwise-local-jet-v1"
        ),
        electrostatics_model="smooth-ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector=(
            "native-atomwise-potential-gradient-v1"
        ),
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=_WATER_ONLY,
        electronic_model_family=ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
        electronic_profile_binding=ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING,
        electronic_energy_semantics=(
            ROUTE2_KNOWN_NONPASSIVE_ENERGY_CONJUGATE_DIAGNOSTIC
        ),
        electrostatic_energy_ledger=(
            MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1
        ),
        known_nonpassive_diagnostic=True,
    ),
    CANONICAL_SMD_PROFILE: Route2SMDProfileSpec(
        name=CANONICAL_SMD_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="pcmsolver-water-keyword",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="legacy-builtin-solvent-probe-v1",
    ),
    GAFF2_CARBONYL_O_PROFILE: Route2SMDProfileSpec(
        name=GAFF2_CARBONYL_O_PROFILE,
        provider="pcmsolver",
        cavity="gaff2-carbonyl-o",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="pcmsolver-water-keyword",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="legacy-builtin-solvent-probe-v1",
    ),
    PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="atomic-center-mean-zero-v1",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="pcmsolver-water-keyword",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="legacy-builtin-solvent-probe-v1",
    ),
    PCMSOLVER_EXACT_GTO_FIELD_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_EXACT_GTO_FIELD_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="exact-gto-v1",
        model_field_gauge="atomic-center-mean-zero-v1",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="pcmsolver-water-keyword",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="legacy-builtin-solvent-probe-v1",
    ),
    PCMSOLVER_INTRINSIC_CAVITY_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_INTRINSIC_CAVITY_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="explicit-smd-water-78.355-v1",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="intrinsic-probe0-noaddsph-v1",
    ),
    PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="exact-gto-v1",
        model_field_gauge="atomic-center-mean-zero-v1",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="explicit-smd-water-78.355-v1",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="intrinsic-probe0-noaddsph-v1",
    ),
    PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="exact-gto-v1",
        model_field_gauge="atomic-center-mean-zero-v1",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="explicit-smd-water-78.355-v1",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        pcmsolver_cavity_generation="intrinsic-probe0-noaddsph-v1",
    ),
    DDPCM_SMD_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_SMD_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
    ),
    DDPCM_SMD_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_SMD_DIRECT_PCM_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    DDPCM_MULTISOLVENT_SMD_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_MULTISOLVENT_SMD_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
    ),
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    # v2 is a numerical-contract revision of the paired direct-PCM study.
    # The physical equation, SMD descriptors, source, and ledger are exactly
    # the v1 values.  It differs only by allowing both paired continuum
    # equations to use the pre-registered, replay-certified finite-resolution
    # energy gate.  Keeping a new profile name prevents v1 evidence from being
    # reinterpreted under a later acceptance policy.
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    DDCOSMO_MULTISOLVENT_SMD_PROFILE: Route2SMDProfileSpec(
        name=DDCOSMO_MULTISOLVENT_SMD_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddcosmo",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
    ),
    DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddcosmo",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE: Route2SMDProfileSpec(
        name=DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddcosmo",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
        provider="fc-aswig",
        cavity="fixed-topology-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="cpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="fixed-topology-aqueous-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    # v2 is deliberately a new evaluation operator rather than a silent
    # mutation of v1.  It applies a four-branch proper-D2 Reynolds average in
    # a nondegenerate JGP94 molecular frame *at the MACE boundary*, removing
    # the fixed laboratory-axis finite-difference artefact of the official
    # real-space MACE-POLAR evaluator.  Weights, continuum parameters, SMD
    # descriptors, cavity, and the direct PCM ledger are unchanged.
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE: Route2SMDProfileSpec(
        name=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
        provider="fc-aswig",
        cavity="fixed-topology-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="cpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="fixed-topology-aqueous-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        mace_geometry_frame_policy="jgp94-d2-canonical-v1",
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    # v3 changes only the public capability boundary.  It reuses the frozen
    # v2 scalar/operator and admits forces only through the per-geometry
    # force certificate; v1/v2 remain energy-only identities.
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE: Route2SMDProfileSpec(
        name=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE,
        provider="fc-aswig",
        cavity="fixed-topology-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="cpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="fixed-topology-aqueous-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        mace_geometry_frame_policy="jgp94-d2-canonical-v1",
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        force_release_eligible=True,
    ),
    DDPCM_GAFF2_CARBONYL_O_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_GAFF2_CARBONYL_O_PROFILE,
        provider="pyddx",
        cavity="gaff2-carbonyl-o",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
    ),
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
        provider="pyddx",
        cavity="gaff2-carbonyl-o",
        mace_long_range_evaluator=(
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
        ),
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
    ),
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE,
        provider="pyddx",
        cavity="gaff2-carbonyl-o",
        mace_long_range_evaluator=(
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
        ),
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        ddpcm_n_proc=4,
    ),
}

SUPPORTED_PCMSOLVER_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "pcmsolver"
)
SUPPORTED_PYDDX_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "pyddx"
)
SUPPORTED_FC_ASWIG_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "fc-aswig"
)
SUPPORTED_TORCH_SMOOTH_PCM_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "torch-smooth-pcm"
)
SUPPORTED_TORCH_SMOOTH_COSMO_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "torch-smooth-cosmo"
)
SUPPORTED_DDPCM_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "pyddx" and spec.electrostatics_model == "ddpcm"
)
SUPPORTED_ROUTE2_SMD_PROFILES = frozenset(_PROFILE_SPECS)
Route2SMDResponseMode = Literal["frozen", "scf"]
SUPPORTED_ROUTE2_SMD_RESPONSE_MODES = frozenset({"frozen", "scf"})


def register_route2_smd_profile(
    spec: Route2SMDProfileSpec,
) -> Route2SMDProfileSpec:
    """Register one complete plug-in profile without replacing an identity."""

    if not isinstance(spec, Route2SMDProfileSpec):
        raise TypeError("spec must be Route2SMDProfileSpec.")
    name = spec.name.strip().lower()
    if name != spec.name:
        raise ValueError("Route-2 profile names must already be lowercase.")
    existing = _PROFILE_SPECS.get(name)
    if existing is not None and existing != spec:
        raise ValueError(f"Route-2 profile {name!r} is already registered.")
    _PROFILE_SPECS[name] = spec
    return spec


def route2_smd_profiles_for_provider(provider: str) -> frozenset[str]:
    normalized = str(provider).strip().lower()
    if normalized in {
        "pcmsolver",
        "pyddx",
        "fc-aswig",
        "torch-smooth-pcm",
        "torch-smooth-cosmo",
    }:
        return frozenset(
            name
            for name, spec in _PROFILE_SPECS.items()
            if spec.provider == normalized
        )
    raise ValueError(f"Unsupported Route 2 SMD provider: {provider}.")


def route2_smd_profile_spec(profile: str) -> Route2SMDProfileSpec:
    normalized = str(profile).strip().lower()
    try:
        return _PROFILE_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Route 2 SMD profile: {profile}.") from exc


def validate_route2_smd_profile(
    provider: str,
    profile: str,
    *,
    solvent: str | None = None,
) -> Route2SMDProfileSpec:
    normalized_provider = str(provider).strip().lower()
    spec = route2_smd_profile_spec(profile)
    if spec.provider != normalized_provider:
        raise ValueError(
            f"Route 2 provider={normalized_provider} does not support "
            f"profile={spec.name}."
        )
    if solvent is not None and not spec.supports_solvent(solvent):
        raise ValueError(
            f"Route 2 profile={spec.name} does not support solvent={solvent}."
        )
    return spec


def validate_route2_smd_response_mode(
    profile_spec: Route2SMDProfileSpec,
    response: str,
) -> Route2SMDResponseMode:
    """Validate the response strategy bound to one scientific profile.

    This is the single response-admission boundary shared by input parsing,
    calculator construction, and continuum providers.  Keeping the rule next
    to the profile registry prevents those layers from drifting apart.
    """

    if not isinstance(profile_spec, Route2SMDProfileSpec):
        raise TypeError("Route-2 response validation requires a profile spec.")
    normalized = str(response).strip().lower()
    if normalized not in SUPPORTED_ROUTE2_SMD_RESPONSE_MODES:
        raise ValueError("SMD response must be frozen or scf.")
    if profile_spec.provider in {
        "torch-smooth-pcm",
        "torch-smooth-cosmo",
    } and normalized != "scf":
        raise ValueError(
            "The MACE-POLAR-EF/smooth-PCM diagnostic requires response='scf'."
        )
    if (
        profile_spec.provider == "pyddx"
        and normalized == "frozen"
        and profile_spec.electrostatic_energy_ledger
        != PCM_HALF_COUPLING_ONLY_V1
    ):
        raise ValueError(
            "Route 2 provider=pyddx response=frozen requires a direct PCM "
            "half-coupling profile."
        )
    if profile_spec.provider == "fc-aswig" and normalized != "scf":
        raise ValueError("Fixed-topology Route 2 requires response='scf'.")
    if (
        profile_spec.provider == "pcmsolver"
        and normalized != "scf"
        and (
            profile_spec.reaction_field_projector != "local-jet"
            or profile_spec.model_field_gauge
            != "continuum-zero-at-infinity"
        )
    ):
        raise ValueError(
            "A non-default reaction-field projector or model-field gauge "
            "requires response=scf; a frozen response would configure but "
            "never apply that model drive."
        )
    return cast(Route2SMDResponseMode, normalized)


__all__ = [
    "CANONICAL_SMD_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_PROFILE",
    "DDPCM_MULTISOLVENT_SMD_PROFILE",
    "DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE",
    "DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE",
    "DDPCM_SMD_DIRECT_PCM_PROFILE",
    "DDPCM_SMD_PROFILE",
    "FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE",
    "FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE",
    "FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE",
    "GAFF2_CARBONYL_O_PROFILE",
    "MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE",
    "MACEPOL_MOLECULAR_REALSPACE_PROFILE",
    "MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE",
    "MACE_POLAR_EF_SMOOTH_PCM_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE",
    "MACE_POLAR_EF_SMOOTH_COSMO_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE",
    "PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE",
    "PCMSOLVER_EXACT_GTO_FIELD_PROFILE",
    "PCMSOLVER_INTRINSIC_CAVITY_PROFILE",
    "PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE",
    "PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE",
    "SUPPORTED_DDPCM_SMD_PROFILES",
    "SUPPORTED_FC_ASWIG_SMD_PROFILES",
    "SUPPORTED_PCMSOLVER_SMD_PROFILES",
    "SUPPORTED_PYDDX_SMD_PROFILES",
    "SUPPORTED_ROUTE2_SMD_PROFILES",
    "SUPPORTED_ROUTE2_SMD_RESPONSE_MODES",
    "SUPPORTED_TORCH_SMOOTH_PCM_SMD_PROFILES",
    "SUPPORTED_TORCH_SMOOTH_COSMO_SMD_PROFILES",
    "Route2SMDProfileSpec",
    "Route2SMDResponseMode",
    "route2_smd_profile_spec",
    "route2_smd_profiles_for_provider",
    "register_route2_smd_profile",
    "validate_route2_smd_profile",
    "validate_route2_smd_response_mode",
]
