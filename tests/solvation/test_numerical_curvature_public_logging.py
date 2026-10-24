"""Replay real .inp -> engine -> job .out without replacing MLIP derivatives."""

import pytest

from maple.function.engine import engine
from maple.function.timer import timer


@pytest.mark.parametrize(
    "model,mode,task,warning",
    [
        ("ani2x", "numerical", "#freq(method=mw)", "Numerical Hessian"),
        ("maceoff23m", "numerical", "#freq(method=mw)", "Numerical Hessian"),
        ("aimnet2", "numerical", "#freq(method=mw)", "Numerical Hessian"),
        ("ani2x", "analytic", "#freq(method=mw)", None),
        ("ani2x", "numerical", "#ts(method=prfo,max_iter=1)", "Numerical Hessian"),
        ("ani2x", "analytic", "#ts(method=prfo,max_iter=1)", None),
        ("ani2x", "analytic", "#ts(method=dimer,use_hvp=false,max_iter=1,rot_max_iter=1,n_init=force)", "Numerical directional curvature"),
        ("ani2x", "analytic", "#ts(method=dimer,use_hvp=true,max_iter=1,rot_max_iter=1,n_init=force)", None),
        ("ani2x", "numerical", "#sp", None),
    ],
)
def test_real_composed_jobs_log_only_curvature_actually_used(
    tmp_path, water_mol2, model, mode, task, warning
):
    source = tmp_path / "curvature.inp"
    output = tmp_path / "curvature.out"
    source.write_text("\n".join([
        f"#model={model}(hessian={mode})",
        task,
        "#device=cpu",
        "#charge(source=mol2,label=fixed-curvature-logging-smoke)",
        "#solv(implicit=water,method=gb,provider=openmm,model=obc2,"
        "profile=obc2-mbondi2,nonpolar=ace,platform=Reference,experimental=true)",
        "", "0 1", f"MOL2 {water_mol2}", "",
    ]), encoding="utf-8")

    timer.reset()
    job = engine()
    job(str(source), str(output))
    text = output.read_text(encoding="utf-8")
    assert "ERROR:" not in text
    assert job.atoms.calc.hessian == mode
    numerical_warnings = [
        line for line in text.splitlines() if line.startswith("WARNING: Numerical")
    ]
    if warning is None:
        assert numerical_warnings == []
    else:
        assert len(numerical_warnings) == 1
        assert numerical_warnings[0].startswith(f"WARNING: {warning}")
        if warning == "Numerical Hessian":
            step = job.atoms.calc.last_numerical_hessian_diagnostics[
                "cartesian_displacement_angstrom"
            ]
            assert f"force-FD step={step:.8g} Angstrom" in numerical_warnings[0]
        else:
            assert "not a full Hessian" in numerical_warnings[0]
            assert "mode=finite_difference" in numerical_warnings[0]
            assert "WARNING: Numerical Hessian" not in text
