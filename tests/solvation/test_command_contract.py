from __future__ import annotations

import pytest

from maple.function.read.command_control import CommandControl


def parse(*lines):
    return CommandControl.from_settings(list(lines)).as_dict()


def test_new_fixed_charge_gb_contract_normalizes_defaults():
    params = parse(
        "#model=ani2x",
        "#sp(verbose=1)",
        "#charge(source=mol2,label=resp2)",
        "#solv(implicit=water,method=gb,model=obc2,nonpolar=ace,experimental=true)",
    )

    assert params["charge"] == {
        "source": "mol2",
        "label": "resp2",
        "mode": "fixed",
        "geometry": "keep",
    }
    assert params["solv"]["provider"] == "openmm"
    assert params["solv"]["profile"] == "obc2-mbondi2"


def test_maple_charge_generation_requires_a_method():
    with pytest.raises(ValueError, match="requires method"):
        parse(
            "#model=ani2x",
            "#charge(source=maple)",
            "#solv(implicit=water,method=gb,experimental=true)",
        )


def test_explicit_qeq_still_requires_experimental_solvation_acknowledgement():
    with pytest.raises(ValueError, match="public benchmark gate"):
        parse(
            "#model=ani2x",
            "#charge(source=maple,method=qeq-gto,mode=fixed)",
            "#solv(implicit=water,method=gb,model=hct,nonpolar=ace)",
        )


def test_polarizable_qeq_cannot_switch_to_pb():
    with pytest.raises(ValueError, match="Polarizable QEq-PB is deferred"):
        parse(
            "#model=ani2x",
            "#sp",
            "#charge(source=maple,method=qeq-gto,mode=polarizable)",
            "#solv(implicit=water,method=pb,experimental=true)",
        )


def test_implicit_provider_requires_explicit_experimental_acknowledgement():
    with pytest.raises(ValueError, match="public benchmark gate"):
        parse(
            "#model=ani2x",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=gb)",
        )


def test_legacy_gbsa_syntax_fails_with_migration_message():
    with pytest.raises(ValueError, match=r"method=gb.*model=obc2"):
        parse(
            "#model=ani2x",
            "#charge(source=maple,method=qeq-gto)",
            "#solv(implicit=water,method=gbsa,experimental=true)",
        )


def test_pb_is_energy_only_in_first_release():
    with pytest.raises(ValueError, match="PB.*single-point"):
        parse(
            "#model=ani2x",
            "#opt",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=pb,provider=apbs,model=lpb,nonpolar=apbs,experimental=true)",
        )


def test_profile_model_pairing_is_locked():
    with pytest.raises(ValueError, match="profile"):
        parse(
            "#model=ani2x",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=gb,model=hct,profile=obc2-mbondi2,experimental=true)",
        )


def test_provider_specific_options_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="does not use PB provider options"):
        parse(
            "#model=ani2x",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=gb,grid_spacing=0.33,experimental=true)",
        )
