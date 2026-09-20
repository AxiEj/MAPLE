from __future__ import annotations

import pytest

from maple.function.read.command_control import CommandControl


def parse(*lines):
    return CommandControl.from_settings(list(lines)).as_dict()


AMBER_NUMERICAL = (
    "#solv(implicit=water,method=gb,provider=ambertools,model=chagb,"
    "nonpolar=cavity-dispersion,experimental=true,mode=numerical,"
    "force_step_angstrom=0.003,force_check_step_angstrom=0.001,"
    "max_scalar_evaluations=1000,max_raw_records=1100,max_audit_bytes=1000000{extra})"
)


APBS_NUMERICAL = (
    "#solv(implicit=water,method=pb,provider=apbs,model=lpb,nonpolar=apbs,"
    "experimental=true,mode=numerical,force_step_angstrom=0.003,"
    "force_check_step_angstrom=0.001,max_scalar_evaluations=1000,"
    "max_raw_records=1100,max_audit_bytes=1000000{extra})"
)


@pytest.mark.parametrize(
    ("task_line", "extra"),
    [
        ("#sp(verbose=1)", ""),
        ("#opt", ""),
        ("#scan(method=lbfgs,mode=rigid)", ""),
        ("#freq(method=mw)", ",curvature_step_angstrom=0.01"),
        ("#ts(method=prfo)", ",curvature_step_angstrom=0.01"),
        ("#ts(method=dimer,delta=0.01)", ""),
    ],
)
@pytest.mark.parametrize("template", [AMBER_NUMERICAL, APBS_NUMERICAL])
def test_numerical_scalar_provider_accepts_force_backed_non_md_tasks(
    task_line, extra, template
):
    charge = (
        "#charge(source=maple,method=am1bcc)"
        if "provider=ambertools" in template
        else "#charge(source=mol2)"
    )
    params = parse(
        "#model=ani2x(hessian=numerical)",
        task_line,
        charge,
        template.format(extra=extra),
    )

    assert params["solv"]["mode"] == "numerical"
    assert params["solv"]["force_step_angstrom"] == pytest.approx(0.003)
    assert params["solv"]["force_check_step_angstrom"] == pytest.approx(0.001)
    assert params["solv"]["max_scalar_evaluations"] == 1000


@pytest.mark.parametrize("template", [AMBER_NUMERICAL, APBS_NUMERICAL])
@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
def test_numerical_scalar_provider_keeps_md_closed(template, ensemble):
    charge = (
        "#charge(source=maple,method=am1bcc)"
        if "provider=ambertools" in template
        else "#charge(source=mol2)"
    )
    with pytest.raises(ValueError, match="numerical.*MD|MD.*numerical"):
        parse(
            "#model=ani2x",
            f"#md(ensemble={ensemble},steps=2)",
            charge,
            template.format(extra=""),
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "mode=native",
        "mode=numerical,force_step_angstrom=0.001,force_check_step_angstrom=0.003",
        "mode=numerical,force_step_angstrom=true,force_check_step_angstrom=0.001",
        "mode=numerical,force_step_angstrom=0.003,force_check_step_angstrom=0.001,max_scalar_evaluations=true",
        "mode=numerical,force_step_angstrom=0.003,force_check_step_angstrom=0.001,max_scalar_evaluations=0",
    ],
)
def test_scalar_provider_rejects_invalid_mode_steps_or_budget_before_dispatch(replacement):
    solv = (
        "#solv(implicit=water,method=pb,provider=apbs,model=lpb,nonpolar=apbs,"
        "experimental=true," + replacement
    )
    if "max_scalar_evaluations" not in replacement:
        solv += ",max_scalar_evaluations=100,max_raw_records=110,max_audit_bytes=100000"
    elif "max_scalar_evaluations=true" not in replacement and "max_scalar_evaluations=0" not in replacement:
        solv += ",max_raw_records=110,max_audit_bytes=100000"
    else:
        solv += ",max_raw_records=110,max_audit_bytes=100000"
    solv += ")"

    with pytest.raises(ValueError):
        parse("#model=ani2x", "#opt", "#charge(source=mol2)", solv)


@pytest.mark.parametrize(
    "omitted",
    ["force_step_angstrom", "force_check_step_angstrom", "max_scalar_evaluations", "max_raw_records", "max_audit_bytes"],
)
def test_numerical_scalar_provider_requires_both_steps_and_all_budgets(omitted):
    options = {
        "force_step_angstrom": "0.003",
        "force_check_step_angstrom": "0.001",
        "max_scalar_evaluations": "100",
        "max_raw_records": "110",
        "max_audit_bytes": "100000",
    }
    options.pop(omitted)
    suffix = ",".join(f"{key}={value}" for key, value in options.items())
    with pytest.raises(ValueError, match=omitted):
        parse(
            "#model=ani2x",
            "#opt",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=pb,provider=apbs,model=lpb,"
            f"nonpolar=apbs,experimental=true,mode=numerical,{suffix})",
        )


@pytest.mark.parametrize("provider", ["openmm", "ddx"])
def test_native_force_provider_rejects_numerical_wrapper_options(provider):
    if provider == "openmm":
        solv = (
            "#solv(implicit=water,method=gb,provider=openmm,experimental=true,"
            "mode=numerical,force_step_angstrom=0.003,force_check_step_angstrom=0.001,"
            "max_scalar_evaluations=100,max_raw_records=110,max_audit_bytes=100000)"
        )
        charge = "#charge(source=maple)"
    else:
        solv = (
            "#solv(implicit=water,method=pb,provider=ddx,model=lpb,"
            "profile=ddlpb-union-mbondi2-v1,nonpolar=none,"
            "solvent_kappa_inverse_angstrom=0.1,experimental=true,mode=numerical,"
            "force_step_angstrom=0.003,force_check_step_angstrom=0.001,"
            "max_scalar_evaluations=100,max_raw_records=110,max_audit_bytes=100000)"
        )
        charge = "#charge(source=mol2)"
    with pytest.raises(ValueError, match="native|numerical"):
        parse("#model=ani2x", "#opt", charge, solv)


@pytest.mark.parametrize("provider", ["openmm", "ddx"])
def test_native_force_provider_accepts_explicit_native_alias_without_fd_options(provider):
    if provider == "openmm":
        solv = "#solv(implicit=water,method=gb,provider=openmm,experimental=true,mode=native)"
        charge = "#charge(source=maple)"
    else:
        solv = (
            "#solv(implicit=water,method=pb,provider=ddx,model=lpb,"
            "profile=ddlpb-union-mbondi2-v1,nonpolar=none,"
            "solvent_kappa_inverse_angstrom=0.1,experimental=true,mode=native)"
        )
        charge = "#charge(source=mol2)"
    params = parse("#model=ani2x", "#opt", charge, solv)
    assert params["solv"]["mode"] == "native"


@pytest.mark.parametrize("task_line", ["#freq(method=mw)", "#ts(method=prfo)"])
def test_scalar_provider_curvature_task_requires_explicit_outer_step(task_line):
    with pytest.raises(ValueError, match="curvature_step_angstrom"):
        parse(
            "#model=ani2x(hessian=numerical)",
            task_line,
            "#charge(source=mol2)",
            APBS_NUMERICAL.format(extra=""),
        )


def test_dimer_task_delta_is_allowed_to_supply_outer_step():
    params = parse(
        "#model=ani2x(hessian=numerical)",
        "#ts(method=dimer,delta=0.007)",
        "#charge(source=mol2)",
        APBS_NUMERICAL.format(extra=",curvature_step_angstrom=0.02"),
    )
    assert params["delta"] == pytest.approx(0.007)
    assert params["solv"]["curvature_step_angstrom"] == pytest.approx(0.02)
