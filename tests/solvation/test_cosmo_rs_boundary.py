from __future__ import annotations

from pathlib import Path

import pytest

from maple.function.cosmo_rs import (
    OPEN_COSMORS_24A_PARAMETERIZATION,
    OpenCOSMORS24aInputBundle,
    parse_orca_opencosmors_solvation_output,
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
    """

    with pytest.raises(ValueError, match="Hartree/kcal"):
        parse_orca_opencosmors_solvation_output(output, inputs=bundle)


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
    assert "No ORCA/openCOSMO-RS executable or chemical accuracy benchmark" in (
        normalized_validation
    )
