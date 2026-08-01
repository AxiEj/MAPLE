from __future__ import annotations

from maple.function.calculator.extra_correction.implicit.source_receiver_contract import (
    ROUTE2_SOURCE_RECEIVER_CONTRACT_VERSION,
    route2_source_receiver_contract,
)
from maple.function.route2_smd_profiles import (
    CANONICAL_SMD_PROFILE,
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE,
    PCMSOLVER_EXACT_GTO_FIELD_PROFILE,
    PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,
)


def test_local_jet_profile_separates_continuum_dual_from_electronic_functional():
    contract = route2_source_receiver_contract(CANONICAL_SMD_PROFILE)

    assert contract.profile == CANONICAL_SMD_PROFILE
    assert contract.solute_source == "point-multipole-l1"
    assert contract.reaction_field_receiver == "local-jet"
    assert contract.continuum_pairing_status == "point-multipole-local-jet-dual"
    assert contract.continuum_pairing_established is True
    assert contract.common_stationary_electronic_functional_established is False
    assert contract.public_capability == "experimental-energy-only"
    assert "variational SCRF" in contract.prohibited_claims


def test_exact_gto_profiles_are_explicitly_nonconjugate_and_energy_only():
    for profile in (
        PCMSOLVER_EXACT_GTO_FIELD_PROFILE,
        PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,
    ):
        contract = route2_source_receiver_contract(profile)

        assert contract.profile == profile
        assert contract.solute_source == "point-multipole-l1"
        assert contract.reaction_field_receiver == "exact-gto-v1"
        assert contract.continuum_pairing_status == (
            "known-nonconjugate-point-source-gto-receiver"
        )
        assert contract.continuum_pairing_established is False
        assert contract.common_stationary_electronic_functional_established is False
        assert contract.next_required_physical_gate == (
            "same-basis-gto-galerkin-source-receiver-and-common-scalar"
        )


def test_force_profile_reports_its_bounded_public_capability_without_overclaiming():
    contract = route2_source_receiver_contract(
        FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE
    )

    assert contract.continuum_pairing_established is True
    assert contract.common_stationary_electronic_functional_established is False
    assert contract.public_capability == (
        "bounded-experimental-energy-and-conservative-forces"
    )
    assert "variational SCRF" in contract.prohibited_claims
    assert "common-energy stationary electronic state" in (
        contract.prohibited_claims
    )
    assert "unbounded solution-phase PES" in contract.prohibited_claims
    assert "universal analytic solution-phase forces" in (
        contract.prohibited_claims
    )
    assert "solution-phase PES" not in contract.prohibited_claims
    assert "analytic solution-phase forces" not in contract.prohibited_claims


def test_contract_provenance_is_json_native_and_versioned():
    provenance = route2_source_receiver_contract(
        PCMSOLVER_EXACT_GTO_FIELD_PROFILE
    ).as_provenance()

    assert provenance["contract_version"] == ROUTE2_SOURCE_RECEIVER_CONTRACT_VERSION
    assert provenance["profile"] == PCMSOLVER_EXACT_GTO_FIELD_PROFILE
    assert isinstance(provenance["prohibited_claims"], list)
