from __future__ import annotations

from pathlib import Path

import pytest

from maple.function.cosmo_rs import (
    OPEN_COSMORS_24A_PARAMETERIZATION,
    OpenCOSMORS24aInputBundle,
    parse_orca_opencosmors_solvation_output,
    render_orca_opencosmors24a_input,
    validate_orca_opencosmors_completion,
)
from maple.function.read.command_control import CommandControl

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _asset(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _bundle(tmp_path: Path, **overrides) -> OpenCOSMORS24aInputBundle:
    values = {
        "solute_gas_output": _asset(
            tmp_path / "solute.gas.out",
            "ORCA gas-phase output",
        ),
        "solute_conductor_surface": _asset(
            tmp_path / "solute.orcacosmo",
            "ORCA conductor surface",
        ),
        "solvent_conductor_surface": _asset(
            tmp_path / "solvent.orcacosmo",
            "ORCA conductor surface",
        ),
        "orca_major_version": 6,
        "solute_charge": 0,
        "solute_multiplicity": 1,
        "solvent_charge": 0,
        "solvent_multiplicity": 1,
        "temperature_k": 298.15,
        "functional": "BP86",
        "basis": "def2-TZVPD",
        "parameterization": OPEN_COSMORS_24A_PARAMETERIZATION,
        "uses_parameterized_cavity_radii": True,
        "minimum_surface_segment_area_angstrom2": 0.01,
    }
    values.update(overrides)
    return OpenCOSMORS24aInputBundle.from_paths(**values)


def test_opencosmors_24a_bundle_hashes_all_three_required_assets(tmp_path):
    bundle = _bundle(tmp_path)
    manifest = bundle.as_manifest()

    assert manifest["scientific_family"] == "COSMO-RS"
    assert manifest["continuum_equation_switch"] is False
    assert manifest["parameterization"] == OPEN_COSMORS_24A_PARAMETERIZATION
    assert set(manifest["assets"]) == {
        "solute_gas_output",
        "solute_conductor_surface",
        "solvent_conductor_surface",
    }
    for asset in manifest["assets"].values():
        assert len(asset["sha256"]) == 64


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"functional": "B3LYP"}, "BP86/def2-TZVPD"),
        ({"basis": "def2-SVP"}, "BP86/def2-TZVPD"),
        ({"temperature_k": 300.0}, "298.15 K"),
        ({"solute_charge": 1}, "neutral closed-shell"),
        ({"solute_multiplicity": 2}, "neutral closed-shell"),
        ({"uses_parameterized_cavity_radii": False}, "parameterized cavity radii"),
        (
            {"minimum_surface_segment_area_angstrom2": 0.02},
            "0.01 angstrom squared",
        ),
    ],
)
def test_opencosmors_24a_bundle_rejects_incompatible_assets(
    tmp_path,
    override,
    message,
):
    with pytest.raises(ValueError, match=message):
        _bundle(tmp_path, **override)


def test_orca_opencosmors_result_parser_locks_units_and_temperature(tmp_path):
    bundle = _bundle(tmp_path)
    output = """
    ------------------------------------------------------------------------------
                              OPENCOSMO-RS CALCULATION
    ------------------------------------------------------------------------------
    Reference temperature             :          298.15 K
    Free energy of solvation (dGsolv) : -0.006626234385 Eh   -4.158026 kcal/mol
    ****ORCA TERMINATED NORMALLY****
    """

    result = parse_orca_opencosmors_solvation_output(output, inputs=bundle)

    assert result.delta_g_solvation_hartree == pytest.approx(-0.006626234385)
    assert result.delta_g_solvation_kcal_mol == pytest.approx(-4.158026)
    assert result.temperature_k == pytest.approx(298.15)
    assert result.input_manifest["parameterization"] == (
        OPEN_COSMORS_24A_PARAMETERIZATION
    )


def test_orca_opencosmors_result_parser_rejects_inconsistent_units(tmp_path):
    bundle = _bundle(tmp_path)
    output = """
    OPENCOSMO-RS CALCULATION
    Reference temperature : 298.15 K
    Free energy of solvation (dGsolv) : -0.006626234385 Eh -5.000000 kcal/mol
    ****ORCA TERMINATED NORMALLY****
    """

    with pytest.raises(ValueError, match="Hartree/kcal"):
        parse_orca_opencosmors_solvation_output(output, inputs=bundle)


def test_orca_opencosmors_input_renderer_is_fixed_geometry_and_injection_safe():
    rendered = render_orca_opencosmors24a_input(
        ("C", "O"),
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
        solvent_alias="Water",
    )

    assert rendered.startswith("! COSMORS(Water)\n%maxcore 2000\n* xyz 0 1\n")
    assert "%pal" not in rendered
    assert rendered.endswith("*\n")

    parallel = render_orca_opencosmors24a_input(
        ("C",),
        ((0.0, 0.0, 0.0),),
        solvent_alias="DMSO",
        nprocs=2,
    )
    assert "%pal nprocs 2 end" in parallel

    with pytest.raises(ValueError, match="unsafe"):
        render_orca_opencosmors24a_input(
            ("C",),
            ((0.0, 0.0, 0.0),),
            solvent_alias="Water) ! Opt",
        )
    with pytest.raises(ValueError, match="shape N by 3"):
        render_orca_opencosmors24a_input(
            ("C", "O"),
            ((0.0, 0.0, 0.0),),
            solvent_alias="Water",
        )


def test_orca_completion_gate_rejects_silent_child_failure():
    success = """
    OPENCOSMO-RS CALCULATION
    Reference temperature : 298.15 K
    Free energy of solvation (dGsolv) : -0.006626234385 Eh -4.158026 kcal/mol
    ****ORCA TERMINATED NORMALLY****
    """
    validate_orca_opencosmors_completion(success)

    with pytest.raises(ValueError, match="failure marker"):
        validate_orca_opencosmors_completion(
            success.replace(
                "****ORCA TERMINATED NORMALLY****",
                "error termination\n****ORCA TERMINATED NORMALLY****",
            )
        )
    with pytest.raises(ValueError, match="terminate normally"):
        validate_orca_opencosmors_completion(
            success.replace("****ORCA TERMINATED NORMALLY****", "")
        )


def test_orca_run_bundle_uses_the_three_expected_assets(tmp_path):
    stem = "record-01"
    _asset(tmp_path / f"{stem}.solute_vac.lastout", "gas")
    _asset(tmp_path / f"{stem}.solute.orcacosmo", "solute")
    _asset(tmp_path / f"{stem}.solvent.orcacosmo", "solvent")

    bundle = OpenCOSMORS24aInputBundle.from_orca_run(tmp_path, stem)

    assert bundle.solute_gas_output.path.name == f"{stem}.solute_vac.lastout"
    assert bundle.solute_conductor_surface.path.name == f"{stem}.solute.orcacosmo"
    assert bundle.solvent_conductor_surface.path.name == f"{stem}.solvent.orcacosmo"
    with pytest.raises(ValueError, match="safe basename"):
        OpenCOSMORS24aInputBundle.from_orca_run(tmp_path, "../escape")


def test_route2_smd_cli_refuses_to_relabel_cosmors_as_a_continuum_switch():
    with pytest.raises(ValueError, match="sigma-profile/statistical"):
        CommandControl.from_settings(
            [
                "#model=macepol-m",
                "#sp",
                "#solv(implicit=water,method=cosmo-rs,experimental=true)",
            ]
        )


def test_cosmors_documentation_preserves_the_external_evidence_boundary():
    formulas = (
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    normalized_validation = " ".join(validation.split())

    assert "COSMO-RS is a separate liquid-thermodynamics workflow" in formulas
    assert "COSMO-RS is intentionally absent from the Route-2 profile registry" in (
        benchmark
    )
    assert "not an MNSol chemical-accuracy or generalization result" in (
        normalized_validation
    )
