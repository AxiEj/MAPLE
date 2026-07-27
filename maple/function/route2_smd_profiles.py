"""Versioned Route-2 SMD profile registry.

This lightweight module is the shared contract between input validation,
calculator dispatch, and continuum providers.  A public profile selects one
complete, reproducible combination of continuum backend, cavity variant, and
MACE-POLAR long-range evaluator; users cannot compose unvalidated mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .route2_solvents import SUPPORTED_ROUTE2_SMD_SOLVENTS


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
DDPCM_SMD_PROFILE = "smd-ddpcm-l15-n1202-v1"
DDPCM_MULTISOLVENT_SMD_PROFILE = "smd-ddpcm-l15-n1202-multisolv-v1"
DDCOSMO_MULTISOLVENT_SMD_PROFILE = (
    "smd-ddcosmo-l15-n1202-multisolv-v1"
)
DDPCM_GAFF2_CARBONYL_O_PROFILE = "smd-ddpcm-l15-n1202-gaff2-o-v1"
DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE = (
    "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1"
)
DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE = (
    "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-omp4-v1"
)

MACEPOL_MOLECULAR_REALSPACE_PROFILE = (
    "graph-longrange-molecular-realspace-v1"
)
MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE = (
    "graph-longrange-forced-periodic-fixed-box40-v1"
)


@dataclass(frozen=True)
class Route2SMDProfileSpec:
    """One frozen v1 audited combination, not an open plug-in schema."""

    name: str
    provider: Literal["pcmsolver", "pyddx"]
    cavity: Literal["canonical-smd", "gaff2-carbonyl-o"]
    mace_long_range_evaluator: str
    electrostatics_model: Literal["iefpcm", "ddpcm", "ddcosmo"]
    solute_source: Literal["point-multipole-l1"]
    reaction_field_projector: Literal[
        "local-jet",
        "exact-gto-v1",
    ]
    model_field_gauge: Literal[
        "continuum-zero-at-infinity",
        "atomic-center-mean-zero-v1",
    ]
    nonpolar_model: Literal[
        "native-water-smd-cds",
        "pyscf-smd-cds",
    ]
    dielectric_policy: Literal[
        "pcmsolver-water-keyword",
        "explicit-smd-water-78.355-v1",
        "legacy-water-78.39",
        "pyscf-smd-2.13.1",
    ]
    coulomb_radii_policy: Literal[
        "smd-water-reference-smd18-v1",
        "pyscf-smd-2.13.1",
    ]
    supported_solvents: frozenset[str]
    strict_original_smd_equivalence: bool = False
    ddpcm_n_proc: int = 1
    default_eligible: bool = False
    pcmsolver_cavity_generation: Literal[
        "legacy-builtin-solvent-probe-v1",
        "intrinsic-probe0-noaddsph-v1",
    ] | None = None

    @property
    def uses_gaff2_carbonyl_oxygen(self) -> bool:
        return self.cavity == "gaff2-carbonyl-o"

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
SUPPORTED_DDPCM_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "pyddx" and spec.electrostatics_model == "ddpcm"
)
SUPPORTED_ROUTE2_SMD_PROFILES = frozenset(_PROFILE_SPECS)


def route2_smd_profiles_for_provider(provider: str) -> frozenset[str]:
    normalized = str(provider).strip().lower()
    if normalized == "pcmsolver":
        return SUPPORTED_PCMSOLVER_SMD_PROFILES
    if normalized == "pyddx":
        return SUPPORTED_PYDDX_SMD_PROFILES
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


__all__ = [
    "CANONICAL_SMD_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_PROFILE",
    "DDPCM_MULTISOLVENT_SMD_PROFILE",
    "DDPCM_SMD_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_PROFILE",
    "GAFF2_CARBONYL_O_PROFILE",
    "MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE",
    "MACEPOL_MOLECULAR_REALSPACE_PROFILE",
    "PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE",
    "PCMSOLVER_EXACT_GTO_FIELD_PROFILE",
    "PCMSOLVER_INTRINSIC_CAVITY_PROFILE",
    "PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE",
    "Route2SMDProfileSpec",
    "SUPPORTED_DDPCM_SMD_PROFILES",
    "SUPPORTED_PCMSOLVER_SMD_PROFILES",
    "SUPPORTED_PYDDX_SMD_PROFILES",
    "SUPPORTED_ROUTE2_SMD_PROFILES",
    "route2_smd_profile_spec",
    "route2_smd_profiles_for_provider",
    "validate_route2_smd_profile",
]
