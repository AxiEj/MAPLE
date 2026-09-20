from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.engine import engine
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.timer import timer


def _lines(*, task="#sp(verbose=1)", kappa="0.1", extra=""):
    options = "" if kappa is None else f",solvent_kappa_inverse_angstrom={kappa}"
    return [
        "#model=ani2x",
        task,
        "#charge(source=mol2,label=fixed-reference-test)",
        (
            "#solv(implicit=water,method=pb,provider=ddx,model=lpb,"
            f"experimental=true{options}{extra})"
        ),
    ]


def _parse(lines):
    return CommandControl.from_settings(lines).as_dict()


@pytest.mark.parametrize("verbose", [0, 1])
def test_reference_parser_opens_only_explicit_polar_sp(verbose):
    parsed = _parse(_lines(task=f"#sp(verbose={verbose})"))
    assert parsed["solv"]["provider"] == "ddx"
    assert parsed["solv"]["nonpolar"] == "none"
    assert parsed["solv"]["profile"] == "ddlpb-union-mbondi2-v1"
    assert parsed["solv"]["solvent_kappa_inverse_angstrom"] == pytest.approx(0.1)


@pytest.mark.parametrize("kappa", [None, "0", "-0.1", "nan", "inf", "true"])
def test_reference_parser_requires_explicit_positive_finite_kappa(kappa):
    with pytest.raises(ValueError, match="solvent_kappa_inverse_angstrom"):
        _parse(_lines(kappa=kappa))


@pytest.mark.parametrize("task", ["#md"])
def test_reference_parser_keeps_unrequested_workflows_closed(task):
    with pytest.raises(ValueError, match="reference.*SP"):
        _parse(_lines(task=task))


@pytest.mark.parametrize("mode", ["rigid", "relaxed"])
def test_reference_parser_opens_rigid_and_relaxed_scan(mode):
    parsed = _parse(_lines(task=f"#scan(method=lbfgs,mode={mode})"))

    assert parsed["task"] == "scan"
    assert parsed["mode"] == mode
    assert parsed["solv"]["provider"] == "ddx"
    assert parsed["solv"]["nonpolar"] == "none"


@pytest.mark.parametrize(
    "task,requires_hessian,expected_task",
    [
        ("#opt(method=lbfgs)", False, "opt"),
        ("#freq(method=mw)", True, "freq"),
        ("#ts(method=prfo,max_iter=1)", True, "ts"),
        ("#ts(method=dimer,max_iter=1)", False, "ts"),
    ],
)
def test_reference_parser_opens_composed_force_workflows(task, requires_hessian, expected_task):
    lines = _lines(task=task)
    if requires_hessian:
        lines[0] = "#model=ani2x(hessian=numerical)"
    parsed = _parse(lines)
    assert parsed["task"] == expected_task
    assert parsed["solv"]["provider"] == "ddx"
    assert parsed["solv"]["nonpolar"] == "none"
    assert parsed["charge"]["mode"] == "fixed"


@pytest.mark.parametrize(
    "task,model,message",
    [
        ("#freq(method=mw)", "#model=ani2x", "hessian=numerical"),
        ("#freq(method=mw)", "#model=ani2x(hessian=analytic)", "hessian=numerical"),
        ("#freq(method=nonmw)", "#model=ani2x(hessian=numerical)", "method=mw"),
        ("#ts(method=prfo)", "#model=ani2x", "hessian=numerical"),
        ("#ts(method=dimer,use_hvp=true)", "#model=ani2x", "HVP"),
        ("#ts(method=neb)", "#model=ani2x", "prfo or dimer"),
    ],
)
def test_reference_curvature_workflows_keep_derivative_boundaries(task, model, message):
    lines = _lines(task=task)
    lines[0] = model
    with pytest.raises(ValueError, match=message):
        _parse(lines)


@pytest.mark.parametrize("extra", [",grid_spacing=0.1", ",nonpolar=ace", ",profile=generic-mbondi2"])
def test_reference_parser_rejects_unmatched_profiles_and_provider_controls(extra):
    with pytest.raises(ValueError, match="ddX|ddLPB"):
        _parse(_lines(extra=extra))


@pytest.mark.parametrize("method,provider", [("pb", "apbs"), ("gb", "openmm")])
def test_reference_kappa_cannot_leak_into_other_providers(method, provider):
    lines = _lines()
    lines[-1] = (
        f"#solv(implicit=water,method={method},provider={provider},"
        "experimental=true,solvent_kappa_inverse_angstrom=0.1)"
    )
    with pytest.raises(ValueError, match="solvent_kappa_inverse_angstrom.*ddx"):
        _parse(lines)


def test_direct_pb_unknown_provider_does_not_fall_back_to_apbs(water_mol2, tmp_path, monkeypatch):
    from maple.function.calculator.extra_correction.implicit import apbs_pb

    def unexpected_apbs(*_args, **_kwargs):
        raise AssertionError("unknown provider silently constructed APBS")

    monkeypatch.setattr(apbs_pb, "APBSLPB", unexpected_apbs)
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(ValueError, match="Unsupported implicit-PB provider"):
        ImplicitSolvationCorrection(
            atoms,
            {"source": "mol2"},
            {"method": "pb", "provider": "unknown", "experimental": True},
            output=tmp_path / "unknown.out",
        )


def test_actual_reference_sp_composes_complete_polar_energy_and_force(water_mol2, tmp_path):
    pytest.importorskip("pyddx")
    inp = tmp_path / "reference.inp"
    out = tmp_path / "reference.out"
    inp.write_text("\n".join([*_lines(), "#device=cpu", "", "0 1", f"MOL2 {water_mol2}", ""]))
    timer.reset()
    job = engine()
    job(str(inp), str(out))
    atoms = cast(Atoms, job.atoms)
    calculator = cast(Any, atoms.calc)
    assert np.isfinite(atoms.get_potential_energy(force_consistent=True))
    assert np.isfinite(atoms.get_forces()).all()
    solvation = calculator.results["solvation"]
    assert solvation["provenance"]["provider"] == "ddx"
    assert set(solvation["components_hartree"]) == {"polar"}
    assert solvation["reference_correction_hartree"] == solvation["energy_hartree"]
    assert "delta_g_solv_hartree" not in solvation
    assert "cluster_continuum_correction_hartree" not in solvation
    assert "Numerical reference correction" in out.read_text()
    manifest = json.loads((tmp_path / "reference.out.implicit/manifest.json").read_text())
    assert manifest["route"]["role"] == "Reference"
    assert manifest["route"]["product_contract"] is False
    assert manifest["route"]["absolute_solvation_free_energy_claim"] is False
    assert "G_ddLPB,polar" in manifest["energy_composition"]
