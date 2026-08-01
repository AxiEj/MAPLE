from __future__ import annotations

import numpy as np
import pytest

from maple.function.route2_solvents import (
    ROUTE2_SMD_SOLVENTS,
    SUPPORTED_ROUTE2_SMD_SOLVENTS,
    normalize_route2_solvent_name,
    route2_solvent_spec,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    route2_coulomb_radii,
    smd_coulomb_radii,
)
from maple.function.route2_smd_profiles import (
    DDPCM_MULTISOLVENT_SMD_PROFILE,
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
    DDPCM_SMD_PROFILE,
    DDCOSMO_MULTISOLVENT_SMD_PROFILE,
    DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
    DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
    SUPPORTED_DDPCM_SMD_PROFILES,
    SUPPORTED_FC_ASWIG_SMD_PROFILES,
    SUPPORTED_PYDDX_SMD_PROFILES,
    route2_smd_profile_spec,
)


_EXPECTED_SOLVENTS = {
    "water": ("water", "water", 0.82, 78.355),
    "methanol": ("methanol", "methanol", 0.43, 32.613),
    "ethanol": ("ethanol", "ethanol", 0.37, 24.852),
    "acetonitrile": ("acetonitrile", "acetonitrile", 0.07, 35.688),
    "dimethylsulfoxide": (
        "dimethylsulfoxide",
        "dimethylsulfoxide",
        0.0,
        46.826,
    ),
    "dimethylformamide": (
        "N,N-dimethylformamide",
        "dimethylformamide",
        0.0,
        37.219,
    ),
    "tetrahydrofuran": (
        "tetrahydrofuran",
        "tetrahydrofuran",
        0.0,
        7.4257,
    ),
    "chloroform": ("chloroform", "chloroform", 0.15, 4.7113),
    "dichloromethane": (
        "dichloromethane",
        "methylenechloride",
        0.10,
        8.93,
    ),
    "toluene": ("toluene", "toluene", 0.0, 2.3741),
    "hexane": ("n-hexane", "hexane", 0.0, 1.8819),
}


def test_route2_solvent_registry_is_immutable_and_has_broad_mnsol_coverage():
    assert SUPPORTED_ROUTE2_SMD_SOLVENTS == frozenset(_EXPECTED_SOLVENTS)
    assert set(ROUTE2_SMD_SOLVENTS) == set(_EXPECTED_SOLVENTS)
    with pytest.raises(TypeError):
        ROUTE2_SMD_SOLVENTS["water"] = route2_solvent_spec("water")


@pytest.mark.parametrize(
    ("name", "expected"),
    sorted(_EXPECTED_SOLVENTS.items()),
)
def test_route2_solvent_specs_lock_upstream_smd_identity(
    name,
    expected,
):
    pyscf_name, mnsol_name, acidity, dielectric = expected
    spec = route2_solvent_spec(name)

    assert spec.name == name
    assert spec.pyscf_smd_name == pyscf_name
    assert spec.mnsol_name == mnsol_name
    assert spec.descriptors.hydrogen_bond_acidity == pytest.approx(acidity)
    assert spec.descriptors.dielectric == pytest.approx(dielectric)
    assert len(spec.descriptors.as_pyscf_tuple()) == 8
    assert spec.experimental_dataset_doi == "10.13020/3eks-j059"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("H2O", "water"),
        ("MeOH", "methanol"),
        ("EtOH", "ethanol"),
        ("MeCN", "acetonitrile"),
        ("DMSO", "dimethylsulfoxide"),
        ("DMF", "dimethylformamide"),
        ("N,N-dimethylformamide", "dimethylformamide"),
        ("THF", "tetrahydrofuran"),
        ("DCM", "dichloromethane"),
        ("n-hexane", "hexane"),
    ],
)
def test_route2_solvent_aliases_normalize_once(alias, canonical):
    assert normalize_route2_solvent_name(alias) == canonical
    assert route2_solvent_spec(alias).name == canonical


def test_route2_solvent_registry_rejects_unknown_solvents():
    with pytest.raises(ValueError, match="Unsupported Route 2 SMD solvent"):
        route2_solvent_spec("made-up-solvent")


@pytest.mark.parametrize(
    ("solvent", "expected_oxygen_radius"),
    [
        ("water", 1.52),
        ("methanol", 1.52),
        ("ethanol", 1.628),
        ("acetonitrile", 2.168),
        ("dimethylsulfoxide", 2.294),
    ],
)
def test_smd_coulomb_oxygen_radius_follows_solvent_acidity(
    solvent,
    expected_oxygen_radius,
):
    radii = smd_coulomb_radii(["H", "C", "O", "Br"], solvent=solvent)

    np.testing.assert_allclose(
        radii,
        [1.20, 1.85, expected_oxygen_radius, 2.60],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_multisolvent_profile_uses_tested_pyscf_element_radii():
    radii = route2_coulomb_radii(
        ["P", "S", "Cl"],
        solvent="acetonitrile",
        profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
    )

    np.testing.assert_allclose(
        radii,
        [2.12, 2.49, 2.38],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_ddpcm_and_ddcosmo_profiles_change_only_the_continuum_equation():
    ddpcm = route2_smd_profile_spec(DDPCM_MULTISOLVENT_SMD_PROFILE)
    ddcosmo = route2_smd_profile_spec(DDCOSMO_MULTISOLVENT_SMD_PROFILE)

    assert ddpcm.electrostatics_model == "ddpcm"
    assert ddcosmo.electrostatics_model == "ddcosmo"
    assert {
        key: value
        for key, value in ddpcm.__dict__.items()
        if key not in {"name", "electrostatics_model"}
    } == {
        key: value
        for key, value in ddcosmo.__dict__.items()
        if key not in {"name", "electrostatics_model"}
    }
    assert DDCOSMO_MULTISOLVENT_SMD_PROFILE in SUPPORTED_PYDDX_SMD_PROFILES
    assert DDCOSMO_MULTISOLVENT_SMD_PROFILE not in (
        SUPPORTED_DDPCM_SMD_PROFILES
    )


def test_direct_pcm_multisolvent_profiles_change_only_the_continuum_equation():
    ddpcm = route2_smd_profile_spec(DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE)
    ddcosmo = route2_smd_profile_spec(
        DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE
    )

    assert ddpcm.electrostatics_model == "ddpcm"
    assert ddcosmo.electrostatics_model == "ddcosmo"
    assert ddpcm.electrostatic_energy_ledger == "pcm-half-coupling-only-v1"
    assert ddcosmo.electrostatic_energy_ledger == "pcm-half-coupling-only-v1"
    assert {
        key: value
        for key, value in ddpcm.__dict__.items()
        if key not in {"name", "electrostatics_model"}
    } == {
        key: value
        for key, value in ddcosmo.__dict__.items()
        if key not in {"name", "electrostatics_model"}
    }
    assert DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE in (
        SUPPORTED_DDPCM_SMD_PROFILES
    )
    assert DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE in (
        SUPPORTED_PYDDX_SMD_PROFILES
    )
    assert DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE not in (
        SUPPORTED_DDPCM_SMD_PROFILES
    )


def test_direct_pcm_v1_profiles_remain_available_as_historical_controls():
    assert (
        route2_smd_profile_spec(DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE).name
        == DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE
    )


def test_fixed_topology_force_candidate_has_its_own_water_only_profile_identity():
    profile = route2_smd_profile_spec(FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE)
    assert profile.provider == "fc-aswig"
    assert profile.electrostatics_model == "cpcm"
    assert profile.cavity == "fixed-topology-smd"
    assert profile.nonpolar_model == "fixed-topology-aqueous-smd-cds"
    assert profile.electrostatic_energy_ledger == "pcm-half-coupling-only-v1"
    assert profile.supported_solvents == frozenset({"water"})
    assert FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE in (
        SUPPORTED_FC_ASWIG_SMD_PROFILES
    )
    np.testing.assert_allclose(
        route2_coulomb_radii(
            ["H", "C", "O"],
            solvent="water",
            profile=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
        ),
        [1.20, 1.85, 1.52],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert (
        route2_smd_profile_spec(DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE).name
        == DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE
    )


def test_fixed_topology_canonical_mace_profile_versions_the_new_operator():
    legacy = route2_smd_profile_spec(FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE)
    canonical = route2_smd_profile_spec(
        FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE
    )
    assert canonical.provider == legacy.provider == "fc-aswig"
    assert canonical.electrostatics_model == legacy.electrostatics_model == "cpcm"
    assert (
        canonical.electrostatic_energy_ledger
        == legacy.electrostatic_energy_ledger
        == "pcm-half-coupling-only-v1"
    )
    assert legacy.mace_geometry_frame_policy == "laboratory-v1"
    assert canonical.mace_geometry_frame_policy == "jgp94-d2-canonical-v1"
    assert canonical.name in SUPPORTED_FC_ASWIG_SMD_PROFILES


def test_water_profile_uses_atomic_number_indexed_smd_reference_radii():
    radii = route2_coulomb_radii(
        ["P", "S", "Cl"],
        solvent="water",
        profile=DDPCM_SMD_PROFILE,
    )

    np.testing.assert_allclose(
        radii,
        [2.12, 2.49, 2.38],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_registry_matches_tested_pyscf_solvent_database_when_available():
    pyscf = pytest.importorskip("pyscf")
    if str(pyscf.__version__) != "2.13.1":
        pytest.skip("Registry comparison is locked to PySCF 2.13.1.")
    from pyscf.data import elements, radii
    from pyscf.solvent import smd

    for spec in ROUTE2_SMD_SOLVENTS.values():
        assert tuple(smd.solvent_db[spec.pyscf_smd_name]) == pytest.approx(
            spec.descriptors.as_pyscf_tuple()
        )
        upstream_radii = smd.smd_radii(
            spec.descriptors.hydrogen_bond_acidity
        )
        symbols = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        expected = [
            upstream_radii[elements.charge(symbol)] * radii.BOHR
            for symbol in symbols
        ]
        np.testing.assert_allclose(
            smd_coulomb_radii(symbols, solvent=spec.name),
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )
