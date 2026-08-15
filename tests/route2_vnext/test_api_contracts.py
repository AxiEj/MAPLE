from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from maple.solvation.api import (
    ASE_PUBLIC_UNITS,
    PROFILE_REGISTRY,
    SCALAR_REGISTRY,
    STATE_REGISTRY,
    CapabilityStatus,
    DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS,
    DIAGNOSTIC_DDX_DDPCM_194_RADIAL_GTO_PROFILE_V1,
    DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1,
    DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1,
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1,
    DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1,
    DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1,
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
    DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1,
    DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1,
    EnergyComponent,
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1,
    ForceComponent,
    OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
    OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1,
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
    OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    ProvenanceBundle,
    ProvenanceRecord,
    Route2Result,
    RuntimeProvenance,
    VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
    VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    profile_registry_manifest,
    scalar_registry_manifest,
)

INITIAL_SCALAR_IDS = {
    "route2-diagnostic-ddx-ddcosmo-radialgto-electrostatic-v1",
    "route2-diagnostic-ddx-ddpcm-radialgto-electrostatic-v1",
    "route2-diagnostic-localjet-cpcm-fixedtopology-electrostatic-v1",
    "route2-operational-cpcm-fixedtopology-electrostatic-v1",
    "route2-operational-cpcm-fixedtopology-smdcds-v1",
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1,
    "route2-operational-macepolar-analytic-gaussian-multipole-"
    "smoothharmonicgalerkin-cpcm-v1",
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    "route2-variational-common-functional-v1",
    "route2-variational-macepolar-energygradient-fixedcavity-cpcm-v1",
    "route2-variational-macepolar-energygradient-fixedcavity-"
    "harmonicgalerkin-cpcm-v1",
    "route2-variational-macepolar-energygradient-smoothharmonicgalerkin-cpcm-v1",
    "route2-variational-macepolar-analytic-gaussian-multipole-"
    "energygradient-smoothharmonicgalerkin-cpcm-v1",
}


def _provenance() -> ProvenanceBundle:
    return ProvenanceBundle(
        model=ProvenanceRecord("model-v1", "model", "1", "1" * 64),
        continuum=ProvenanceRecord("cpcm-v1", "continuum", "1", "2" * 64),
        cavity=ProvenanceRecord("fixed-v1", "cavity", "1", "3" * 64),
        runtime=RuntimeProvenance("3.11", "test", (("maple", "0.1.4"),)),
    )


def _result(**changes) -> Route2Result:
    values = dict(
        atom_count=2,
        profile_id=OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
        energy_components=(
            EnergyComponent("vacuum_energy", 1.0),
            EnergyComponent("cpcm_half_coupling_electrostatic", 0.5),
        ),
        provenance=_provenance(),
        primal_residual=1.0e-12,
        adjoint_residual=2.0e-12,
        root_identity="cold-root-1",
        root_sha256="a" * 64,
        fail_closed=True,
    )
    values.update(changes)
    return Route2Result(**values)


def test_capabilities_default_false_and_variational_disabled():
    status = CapabilityStatus()
    assert status.enabled_tiers == ()
    assert status.variational_functional is False
    with pytest.raises(ValueError, match="requires energy"):
        CapabilityStatus(variational_functional=True)


def test_authoritative_profile_registry_is_immutable_and_fully_disabled():
    assert len(PROFILE_REGISTRY) == 19
    with pytest.raises(TypeError):
        PROFILE_REGISTRY["new"] = next(iter(PROFILE_REGISTRY.values()))
    for profile_id, profile in PROFILE_REGISTRY.items():
        assert profile_id == profile.profile_id
        assert profile.scalar_id in SCALAR_REGISTRY
        scalar = SCALAR_REGISTRY[profile.scalar_id]
        assert profile.state_equation_id == scalar.state_equation_id
        assert profile.state_equation_id in STATE_REGISTRY
        assert profile.enabled is False
        assert profile.capabilities.enabled_tiers == ()
        assert profile.evidence_artifact_ids == ()
    manifest = profile_registry_manifest()
    manifest[OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1]["enabled"] = True
    assert PROFILE_REGISTRY[OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1].enabled is False
    diagnostic = PROFILE_REGISTRY[DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1]
    assert diagnostic.scalar_id == DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1
    assert diagnostic.coupling_id.endswith("local-l1-jet-diagnostic.v1")
    assert diagnostic.enabled is False
    assert diagnostic.capabilities.enabled_tiers == ()
    radial = PROFILE_REGISTRY[OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1]
    assert (
        radial.scalar_id
        == PROFILE_REGISTRY[OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1].scalar_id
    )
    assert radial.coupling_id.endswith("mace-polar-native-radial-gto-v1")
    assert radial.source_space_id.endswith("mace-polar-radial-gto-source-space.v1")
    assert radial.field_space_id.endswith("mace-polar-radial-gto-field-dual-space.v1")
    assert radial.pairing_id.endswith("mace-polar-radial-gto-pairing.v1")
    assert radial.coordinate_contract_id.endswith(
        "mace-polar-radial-gto-linear-charge-coordinates.v1"
    )
    assert radial.continuum_configuration_contract_id.endswith(
        "water-eps78p39-smd-radii-lebedev194.v1"
    )
    assert radial.enabled is False
    operational_harmonic = PROFILE_REGISTRY[
        OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
    ]
    assert operational_harmonic.scalar_id == (
        OPERATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
    )
    assert "analytic-gaussian-multipole" in operational_harmonic.model_profile
    assert operational_harmonic.continuum_profile.startswith("smooth-weighted-harmonic")
    assert operational_harmonic.cavity_profile.startswith(
        "smooth-weighted-overlap-harmonic"
    )
    assert operational_harmonic.enabled is False
    assert operational_harmonic.capabilities.enabled_tiers == ()
    analytic_variational = PROFILE_REGISTRY[
        VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
    ]
    assert analytic_variational.scalar_id == (
        VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
    )
    assert "analytic-gaussian-multipole" in analytic_variational.model_profile
    assert analytic_variational.enabled is False
    assert analytic_variational.capabilities.enabled_tiers == ()
    harmonic = PROFILE_REGISTRY[
        VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
    ]
    assert (
        harmonic.scalar_id
        == VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
    )
    assert harmonic.continuum_profile.startswith("smooth-weighted-harmonic")
    assert harmonic.cavity_profile.startswith("smooth-weighted-overlap-harmonic")
    assert harmonic.enabled is False
    assert harmonic.capabilities.enabled_tiers == ()
    diagnostic_radial = PROFILE_REGISTRY[
        DIAGNOSTIC_RADIAL_GTO_CPCM_ELECTROSTATIC_PROFILE_V1
    ]
    assert diagnostic_radial.scalar_id == radial.scalar_id
    assert diagnostic_radial.coupling_id == radial.coupling_id
    assert diagnostic_radial.continuum_configuration_contract_id.endswith(
        "unbound-diagnostic.v1"
    )
    assert diagnostic_radial.enabled is False
    high_order = PROFILE_REGISTRY[DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1]
    assert high_order.scalar_id == radial.scalar_id
    assert high_order.model_profile.endswith("fixed-box40-contract-v1")
    assert high_order.continuum_configuration_contract_id.endswith(
        "water-eps78p39-smd-radii-lebedev590.v1"
    )
    assert high_order.capabilities.enabled_tiers == ()
    assert high_order.enabled is False
    assert tuple(DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS) == (
        32,
        40,
        48,
        56,
    )
    for (
        box_length,
        profile_id,
    ) in DIAGNOSTIC_FIXED_BOX_CPCM_590_RADIAL_GTO_PROFILE_IDS.items():
        box_profile = PROFILE_REGISTRY[profile_id]
        assert box_profile.model_profile.endswith(f"fixed-box{box_length}-contract-v1")
        assert box_profile.continuum_configuration_contract_id.endswith(
            "water-eps78p39-smd-radii-lebedev590.v1"
        )
        assert box_profile.enabled is False
        assert box_profile.capabilities.enabled_tiers == ()
    ultra_high_order = PROFILE_REGISTRY[
        DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1
    ]
    assert ultra_high_order.scalar_id == radial.scalar_id
    assert ultra_high_order.model_profile.endswith("fixed-box48-contract-v1")
    assert ultra_high_order.continuum_configuration_contract_id.endswith(
        "water-eps78p39-smd-radii-lebedev1202.v1"
    )
    assert ultra_high_order.enabled is False
    assert ultra_high_order.capabilities.enabled_tiers == ()
    pair_frame = PROFILE_REGISTRY[DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1]
    assert pair_frame.scalar_id == radial.scalar_id
    assert pair_frame.model_profile.endswith("fixed-box40-contract-v1")
    assert pair_frame.continuum_profile == radial.continuum_profile
    assert pair_frame.cavity_profile == radial.cavity_profile
    assert pair_frame.continuum_configuration_contract_id.endswith(
        "pairframe-lebedev110.v1"
    )
    assert pair_frame.capabilities.enabled_tiers == ()
    assert pair_frame.enabled is False
    ddx = PROFILE_REGISTRY[DIAGNOSTIC_DDX_DDPCM_194_RADIAL_GTO_PROFILE_V1]
    assert ddx.scalar_id == DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1
    assert ddx.model_profile == radial.model_profile
    assert ddx.continuum_profile == "ddx-ddpcm-radial-gto-v1"
    assert ddx.cavity_profile == "ddx-union-of-spheres-exposed-lebedev-v0p8p0"
    assert ddx.coupling_id == radial.coupling_id
    assert ddx.source_space_id == radial.source_space_id
    assert ddx.field_space_id == radial.field_space_id
    assert ddx.pairing_id == radial.pairing_id
    assert ddx.continuum_configuration_contract_id.endswith(
        "ddx-ddpcm-l8-lebedev194.v1"
    )
    assert ddx.capabilities.enabled_tiers == ()
    assert ddx.enabled is False
    ddcosmo_scalar = SCALAR_REGISTRY[DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1]
    assert ddcosmo_scalar.continuum_profile == "ddx-ddcosmo-radial-gto-v1"
    assert ddcosmo_scalar.enabled is False
    assert ddcosmo_scalar.admitted_capabilities.enabled_tiers == ()
    assert all(
        profile.scalar_id != DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1
        for profile in PROFILE_REGISTRY.values()
    )
    variational = PROFILE_REGISTRY[
        VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1
    ]
    assert (
        variational.scalar_id
        == VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1
    )
    assert variational.model_profile.endswith("variational-effective-source-v1")
    assert variational.source_space_id == radial.source_space_id
    assert variational.field_space_id == radial.field_space_id
    assert variational.pairing_id == radial.pairing_id
    assert variational.nonpolar_profile == "none"
    assert variational.enabled is False
    assert variational.capabilities.enabled_tiers == ()
    variational_scalar = SCALAR_REGISTRY[variational.scalar_id]
    assert "original_four_channel_density_as_variational_source" in (
        variational_scalar.excluded_components
    )
    assert variational_scalar.implementation_entry_point == (
        "maple.solvation.coupling.variational_state:VariationalCommonFunctional"
    )


def test_admission_records_cannot_bypass_enablement_or_evidence():
    base = PROFILE_REGISTRY[OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1]
    with pytest.raises(ValueError, match="must be enabled"):
        replace(base, capabilities=CapabilityStatus(energy=True))
    with pytest.raises(ValueError, match="requires evidence"):
        replace(base, enabled=True, capabilities=CapabilityStatus(energy=True))
    with pytest.raises(ValueError, match="must admit scalar energy"):
        replace(base, enabled=True)
    with pytest.raises(ValueError, match="without an admitted capability"):
        replace(base, evidence_artifact_ids=("fake-evidence",))
    scalar = SCALAR_REGISTRY[base.scalar_id]
    with pytest.raises(ValueError, match="requires evidence"):
        replace(
            scalar, enabled=True, admitted_capabilities=CapabilityStatus(energy=True)
        )


def test_contracts_and_nested_values_are_immutable():
    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.energy_components = ()
    mutable_metadata = {"dtype": "float64"}
    record = ProvenanceRecord("m", "model", "1", "a" * 64, mutable_metadata)
    mutable_metadata["dtype"] = "float32"
    assert record.metadata == (("dtype", "float64"),)


def test_result_derives_identity_capabilities_and_totals_from_registry_leaves():
    result = _result()
    profile = PROFILE_REGISTRY[OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1]
    assert result.scalar_id == profile.scalar_id
    assert result.state_equation_id == profile.state_equation_id
    assert result.capabilities is profile.capabilities
    assert result.total_energy_eV == 1.5
    assert result.total_forces_eV_per_A is None


@pytest.mark.parametrize(
    "forbidden",
    (
        {"scalar_id": "forged"},
        {"state_equation_id": "forged"},
        {"capabilities": CapabilityStatus(energy=True)},
        {"total_energy_eV": 999.0},
        {"total_forces_eV_per_A": ((0.0, 0.0, 0.0),) * 2},
        {"closure_tolerance": 1.0},
    ),
)
def test_result_rejects_caller_supplied_identity_capability_total_or_tolerance(
    forbidden,
):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _result(**forbidden)


def test_disabled_profile_allows_only_fail_closed_energy_evidence_and_no_force():
    with pytest.raises(ValueError, match="fail-closed internal evidence"):
        _result(fail_closed=False)
    with pytest.raises(ValueError, match="conservative-force admission"):
        _result(
            force_components=(
                ForceComponent("vacuum", ((1.0, 0.0, -1.0), (0.0, 2.0, 0.0))),
            )
        )
    with pytest.raises(ValueError, match="admitted public domain"):
        _result(admitted_domain=(("dtype", "float64"),))


def test_result_rejects_unknown_profile_and_evidence_mismatch():
    with pytest.raises(KeyError, match="Unregistered Route-2 profile"):
        _result(profile_id="forged-profile")
    with pytest.raises(ValueError, match="exactly match"):
        _result(evidence_artifact_ids=("forged-admission",))


def test_result_validates_shape_finiteness_and_provenance():
    with pytest.raises(ValueError, match="finite"):
        _result(energy_components=(EnergyComponent("bad", float("nan")),))
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        ProvenanceRecord("m", "model", "1", "not-a-digest")
    with pytest.raises(ValueError, match="kind='model'"):
        ProvenanceBundle(
            model=ProvenanceRecord("m", "other", "1", "1" * 64),
            continuum=ProvenanceRecord("c", "continuum", "1", "2" * 64),
            cavity=ProvenanceRecord("s", "cavity", "1", "3" * 64),
            runtime=RuntimeProvenance("3.11", "test"),
        )


def test_scalar_registry_has_unique_complete_state_bound_entries():
    assert set(SCALAR_REGISTRY) == INITIAL_SCALAR_IDS
    assert len(SCALAR_REGISTRY) == len(
        {entry.scalar_id for entry in SCALAR_REGISTRY.values()}
    )
    for scalar_id, entry in SCALAR_REGISTRY.items():
        assert scalar_id == entry.scalar_id
        assert entry.exact_formula
        assert entry.state_equation_id in STATE_REGISTRY
        assert entry.enabled is False
        assert entry.admitted_capabilities.enabled_tiers == ()
        assert entry.evidence_artifact_ids == ()
        assert not (set(entry.included_components) & set(entry.excluded_components))
    variational = SCALAR_REGISTRY["route2-variational-common-functional-v1"]
    assert variational.admitted_capabilities.variational_functional is False
    manifest = scalar_registry_manifest()
    assert set(manifest) == INITIAL_SCALAR_IDS
    assert manifest[variational.scalar_id]["admitted_capabilities"]["V"] is False
    assert (
        manifest["route2-operational-cpcm-fixedtopology-electrostatic-v1"][
            "implementation_entry_point"
        ]
        == "maple.solvation.coupling.energy:OperationalElectrostaticScalar"
    )
    assert manifest["route2-operational-cpcm-fixedtopology-smdcds-v1"][
        "implementation_entry_point"
    ].startswith("disabled:")
    assert manifest["route2-variational-common-functional-v1"][
        "implementation_entry_point"
    ].startswith("disabled:")


def test_public_ase_units_are_declared_without_touching_legacy_calculators():
    assert ASE_PUBLIC_UNITS.energy == "eV"
    assert ASE_PUBLIC_UNITS.forces == "eV/A"
    assert ASE_PUBLIC_UNITS.hessian == "eV/A^2"
