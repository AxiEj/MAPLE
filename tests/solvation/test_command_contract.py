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


def test_maple_charge_generation_defaults_to_am1bcc():
    params = parse(
        "#model=ani2x",
        "#charge(source=maple)",
        "#solv(implicit=water,method=gb,experimental=true)",
    )

    assert params["charge"] == {
        "source": "maple",
        "method": "am1bcc",
        "mode": "fixed",
        "geometry": "keep",
    }


def test_abcg2_requires_explicit_method_selection():
    params = parse(
        "#model=ani2x",
        "#charge(source=maple,method=abcg2)",
        "#solv(implicit=water,method=gb,experimental=true)",
    )

    assert params["charge"]["method"] == "abcg2"


@pytest.mark.parametrize(
    "task_line, task",
    [
        ("#sp", "sp"),
        ("#opt(method=lbfgs)", "opt"),
        ("#scan(method=lbfgs,mode=rigid)", "scan"),
    ],
)
def test_force_capable_gb_route_accepts_product_tasks(task_line, task):
    params = parse(
        "#model=ani2x",
        task_line,
        "#charge(source=maple)",
        "#solv(implicit=water,method=gb,experimental=true)",
    )

    assert params["task"] == task
    assert params["charge"]["method"] == "am1bcc"
    assert params["solv"]["provider"] == "openmm"


def test_force_capable_gb_route_accepts_explicit_numerical_frequency():
    params = parse(
        "#model=ani2x(hessian=numerical)",
        "#freq(method=mw,ilowfreq=2)",
        "#charge(source=maple)",
        "#solv(implicit=water,method=gb,experimental=true)",
    )

    assert params["task"] == "freq"
    assert params["model_options"]["hessian"] == "numerical"
    assert params["charge"]["method"] == "am1bcc"
    assert params["solv"]["provider"] == "openmm"


@pytest.mark.parametrize("method", ["nonmw", "both"])
def test_implicit_gb_frequency_rejects_nonphysical_or_unimplemented_methods(
    method,
):
    with pytest.raises(ValueError, match="method=mw"):
        parse(
            "#model=ani2x(hessian=numerical)",
            f"#freq(method={method})",
            "#charge(source=maple)",
            "#solv(implicit=water,method=gb,experimental=true)",
        )


def test_frequency_rejects_unimplemented_both_mode_before_dispatch():
    with pytest.raises(ValueError, match="not implemented"):
        parse(
            "#model=ani2x(hessian=numerical)",
            "#freq(method=both)",
        )


@pytest.mark.parametrize("ilowfreq", [-1, 4, 99])
def test_frequency_rejects_unknown_low_frequency_treatment(ilowfreq):
    with pytest.raises(ValueError, match="ilowfreq"):
        parse(
            "#model=ani2x(hessian=numerical)",
            f"#freq(method=mw,ilowfreq={ilowfreq})",
        )


def test_frequency_verbosity_parameter_reaches_the_dispatcher_contract():
    params = parse(
        "#model=ani2x(hessian=numerical)",
        "#freq(method=mw,verbosity=10)",
    )

    assert params["verbosity"] == 10


@pytest.mark.parametrize(
    "model_line",
    [
        "#model=ani2x",
        "#model=ani2x(hessian=analytic)",
    ],
)
def test_implicit_gb_frequency_requires_explicit_numerical_hessian(model_line):
    with pytest.raises(ValueError, match="hessian=numerical"):
        parse(
            model_line,
            "#freq(method=mw)",
            "#charge(source=maple)",
            "#solv(implicit=water,method=gb,experimental=true)",
        )


@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
def test_force_capable_fixed_charge_gb_route_accepts_nonperiodic_md(ensemble):
    params = parse(
        "#model=ani2x",
        f"#md(ensemble={ensemble},steps=2,timestep=0.1)",
        "#charge(source=mol2,label=am1bcc-frozen-manifest)",
        "#solv(implicit=water,method=gb,experimental=true)",
    )

    assert params["task"] == "md"
    assert params["ensemble"] == ensemble
    assert params["charge"]["mode"] == "fixed"
    assert params["solv"]["provider"] == "openmm"


def test_implicit_gb_md_rejects_npt_and_prebuilt_inner_shell():
    with pytest.raises(ValueError, match="non-periodic NVE or NVT"):
        parse(
            "#model=ani2x",
            "#md(ensemble=npt,steps=2)",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=gb,experimental=true)",
        )

    with pytest.raises(ValueError, match="prebuilt.*MD"):
        parse(
            "#model=ani2x",
            "#md(ensemble=nvt,steps=2)",
            "#charge(source=mol2)",
            ("#solv(implicit=water,inner=prebuilt,method=gb," "experimental=true)"),
        )


def test_implicit_pb_remains_energy_only_for_md():
    with pytest.raises(ValueError, match="PB.*single-point"):
        parse(
            "#model=ani2x",
            "#md(ensemble=nvt,steps=2)",
            "#charge(source=mol2)",
            (
                "#solv(implicit=water,method=pb,provider=apbs,model=lpb,"
                "nonpolar=apbs,experimental=true)"
            ),
        )


def test_implicit_gb_md_rejects_polarizable_research_charge_mode():
    with pytest.raises(ValueError, match="fixed charges"):
        parse(
            "#model=ani2x",
            "#md(ensemble=nvt,steps=2)",
            "#charge(source=maple,method=qeq-gto,mode=polarizable)",
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


@pytest.mark.parametrize("task_line", ["#opt", "#freq(method=mw)"])
def test_pb_is_energy_only_in_first_release(task_line):
    with pytest.raises(ValueError, match="PB.*single-point"):
        parse(
            "#model=ani2x",
            task_line,
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


def test_chagb_cavity_dispersion_is_explicit_am1bcc_sp_only():
    params = parse(
        "#model=ani2x",
        "#sp",
        "#charge(source=maple)",
        (
            "#solv(implicit=water,method=gb,provider=ambertools,"
            "model=chagb,nonpolar=cavity-dispersion,experimental=true)"
        ),
    )

    assert params["charge"]["method"] == "am1bcc"
    assert params["solv"]["provider"] == "ambertools"
    assert params["solv"]["model"] == "chagb"
    assert params["solv"]["profile"] == "chagb-bondi-pbsa-inp2"
    assert params["solv"]["nonpolar"] == "cavity-dispersion"


@pytest.mark.parametrize(
    "task_line, charge_line, message",
    [
        ("#opt", "#charge(source=maple)", "single-point energy-only"),
        (
            "#scan(method=lbfgs,mode=rigid)",
            "#charge(source=maple)",
            "single-point energy-only",
        ),
        (
            "#freq(method=mw)",
            "#charge(source=maple)",
            "single-point energy-only",
        ),
        (
            "#md(ensemble=nvt,steps=2)",
            "#charge(source=maple)",
            "single-point energy-only",
        ),
        ("#sp(verbose=1)", "#charge(source=maple)", "single-point energy-only"),
        (
            "#sp",
            "#charge(source=maple,method=abcg2)",
            "requires fixed.*am1bcc",
        ),
        ("#sp", "#charge(source=mol2)", "requires fixed.*am1bcc"),
    ],
)
def test_chagb_cavity_dispersion_fails_closed_outside_validated_contract(
    task_line, charge_line, message
):
    with pytest.raises(ValueError, match=message):
        parse(
            "#model=ani2x",
            task_line,
            charge_line,
            (
                "#solv(implicit=water,method=gb,provider=ambertools,"
                "model=chagb,nonpolar=cavity-dispersion,experimental=true)"
            ),
        )
