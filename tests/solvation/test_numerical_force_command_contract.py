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
def test_numerical_scalar_provider_is_not_exposed_to_user_tasks(
    task_line, extra, template
):
    charge = (
        "#charge(source=maple,method=am1bcc)"
        if "provider=ambertools" in template
        else "#charge(source=mol2)"
    )
    with pytest.raises(ValueError, match="internal diagnostic|not a runtime mode"):
        parse(
            "#model=ani2x(hessian=numerical)",
            task_line,
            charge,
            template.format(extra=extra),
        )


@pytest.mark.parametrize("template", [AMBER_NUMERICAL, APBS_NUMERICAL])
def test_numerical_scalar_provider_keeps_md_closed_by_the_same_runtime_gate(template):
    charge = (
        "#charge(source=maple,method=am1bcc)"
        if "provider=ambertools" in template
        else "#charge(source=mol2)"
    )
    with pytest.raises(ValueError, match="internal diagnostic|not a runtime mode"):
        parse(
            "#model=ani2x",
            "#md(ensemble=nvt,steps=2)",
            charge,
            template.format(extra=""),
        )


@pytest.mark.parametrize("provider", ["ambertools", "apbs"])
def test_scalar_provider_rejects_native_mode_because_no_native_force_is_admitted(provider):
    if provider == "ambertools":
        charge = "#charge(source=maple,method=am1bcc)"
        solv = (
            "#solv(implicit=water,method=gb,provider=ambertools,model=chagb,"
            "nonpolar=cavity-dispersion,experimental=true,mode=native)"
        )
    else:
        charge = "#charge(source=mol2)"
        solv = (
            "#solv(implicit=water,method=pb,provider=apbs,model=lpb,"
            "nonpolar=apbs,experimental=true,mode=native)"
        )
    with pytest.raises(ValueError, match="no admitted native force"):
        parse("#model=ani2x", "#opt", charge, solv)


@pytest.mark.parametrize("provider", ["ambertools", "apbs"])
def test_scalar_provider_keeps_energy_only_sp_available(provider):
    if provider == "ambertools":
        charge = "#charge(source=maple,method=am1bcc)"
        solv = (
            "#solv(implicit=water,method=gb,provider=ambertools,model=chagb,"
            "nonpolar=cavity-dispersion,experimental=true)"
        )
    else:
        charge = "#charge(source=mol2)"
        solv = (
            "#solv(implicit=water,method=pb,provider=apbs,model=lpb,"
            "nonpolar=apbs,experimental=true)"
        )
    params = parse("#model=ani2x", "#sp", charge, solv)
    assert "mode" not in params["solv"]


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
