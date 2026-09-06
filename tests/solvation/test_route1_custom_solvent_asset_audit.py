from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
CONTRACT_PATH = BENCHMARK_DIR / "route1_custom_solvent_asset_contract_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "route1_custom_solvent_asset_audit",
    BENCHMARK_DIR / "route1_custom_solvent_asset_audit.py",
)
assert SPEC is not None
asset_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(asset_audit)


OXYGEN_CHARGE_E = -0.8
HYDROGEN_CHARGE_E = 0.4
OXYGEN_CHARGE_AMBER = (
    OXYGEN_CHARGE_E * asset_audit.AMBER_ELECTROSTATIC_CHARGE_SCALE
)
HYDROGEN_CHARGE_AMBER = (
    HYDROGEN_CHARGE_E * asset_audit.AMBER_ELECTROSTATIC_CHARGE_SCALE
)


def _mdl(*, hydrogen_charge: float = HYDROGEN_CHARGE_AMBER) -> str:
    return f"""%FLAG POINTERS
%FORMAT(10I8)
       3       2
%FLAG ATMNAME
%FORMAT(20a4)
{"O   H1  "}
%FLAG MULTI
%FORMAT(10I8)
       1       2
%FLAG MASS
%FORMAT(5e16.8)
  1.60000000e+01  1.00000000e+00
%FLAG CHG
%FORMAT(5e16.8)
 {OXYGEN_CHARGE_AMBER:.8E}  {hydrogen_charge:.8E}
%FLAG LJEPSILON
%FORMAT(5e16.8)
  1.50000000e-01  2.00000000e-02
%FLAG LJSIGMA
%FORMAT(5e16.8)
  1.80000000e+00  6.50000000e-01
%FLAG COORD
%FORMAT(5e16.8)
  0.00000000e+00  0.00000000e+00  0.00000000e+00
  1.00000000e+00  0.00000000e+00  0.00000000e+00
 -3.00000000e-01  9.00000000e-01  0.00000000e+00
"""


def _rism_input(
    *,
    temperature: float = 298.15,
    density: float = 0.0333,
    density_units: str = "1/A^3",
) -> str:
    return f"""&PARAMETERS
  THEORY='DRISM',
  CLOSUR='KH',
  MODEL='model.mdl',
  TEMPER={temperature:.2f}, DIEps=78.50,
  DR=0.025, NR=2,
  DENSITY={density:.16E}, UNITS='{density_units}',
/
"""


def _xvv(*, hydrogen_charge: float | None = None, density: float = 0.0333) -> str:
    reduced_charge_scale = (asset_audit.BOLTZMANN_KCAL_MOL_K * 298.15) ** -0.5
    reduced_epsilon_scale = (asset_audit.BOLTZMANN_KCAL_MOL_K * 298.15) ** -1.0
    if hydrogen_charge is None:
        hydrogen_charge = HYDROGEN_CHARGE_AMBER * reduced_charge_scale
    return f"""%FLAG POINTERS
%FORMAT(10I8)
       2       2       1
%FLAG THERMO
%FORMAT(1P5E24.16)
  2.9815000000000000E+02  7.8500000000000000E+01  0.0000000000000000E+00  1.0E+00  2.5E-02
  1.0E+00
%FLAG ATOM_NAME
%FORMAT(20A4)
{"O   H1  "}
%FLAG MTV
%FORMAT(10I8)
       1       2
%FLAG NVSP
%FORMAT(10I8)
       2
%FLAG MASS
%FORMAT(1P5E24.16)
  1.6000000000000000E+01  1.0000000000000000E+00
%FLAG RHOV
%FORMAT(1P5E24.16)
  {density:.16E}  {2.0 * density:.16E}
%FLAG RHOSP
%FORMAT(1P5E24.16)
  {density:.16E}
%FLAG QV
%FORMAT(1P5E24.16)
 {OXYGEN_CHARGE_AMBER * reduced_charge_scale:.16E}  {hydrogen_charge:.16E}
%FLAG EPSV
%FORMAT(1P5E24.16)
 {0.15 * reduced_epsilon_scale:.16E}  {0.02 * reduced_epsilon_scale:.16E}
%FLAG RMIN2V
%FORMAT(1P5E24.16)
  1.8000000000000000E+00  6.5000000000000000E-01
%FLAG COORD
%FORMAT(1P3E24.16)
  0.0000000000000000E+00  0.0000000000000000E+00  0.0000000000000000E+00
  1.0000000000000000E+00  0.0000000000000000E+00  0.0000000000000000E+00
 -3.0000000000000000E-01  9.0000000000000000E-01  0.0000000000000000E+00
%FLAG XVV
%FORMAT(1P5E24.16)
  1.0E+00  2.0E+00  3.0E+00  4.0E+00  5.0E+00  6.0E+00  7.0E+00  8.0E+00
"""


def _write_asset_set(
    tmp_path: Path,
    *,
    mdl_hydrogen_charge: float = HYDROGEN_CHARGE_AMBER,
    xvv_hydrogen_charge: float | None = None,
    temperature: float = 298.15,
    density: float = 0.0333,
) -> dict[str, Path]:
    paths = {
        "mdl": tmp_path / "model.mdl",
        "rism1d_input": tmp_path / "solvent.inp",
        "xvv": tmp_path / "solvent.xvv",
        "rism1d_stdout": tmp_path / "solvent.rism1d.out",
        "generation_evidence": tmp_path / "generation_evidence.json",
    }
    paths["mdl"].write_text(_mdl(hydrogen_charge=mdl_hydrogen_charge), encoding="ascii")
    paths["rism1d_input"].write_text(
        _rism_input(temperature=temperature, density=density), encoding="ascii"
    )
    paths["xvv"].write_text(
        _xvv(hydrogen_charge=xvv_hydrogen_charge, density=density), encoding="ascii"
    )
    paths["rism1d_stdout"].write_text(
        """synthetic rism1d
step=   3     Res=  5.0000000000000000E-09     MDIIS= 20
relaxing RISM DT:
step=   2     Res=  4.0000000000000000E-09     MDIIS= 20
""",
        encoding="ascii",
    )
    evidence = asset_audit.core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-rism1d-generation-evidence-v1",
            "assets": {
                "mdl": {
                    "name": paths["mdl"].name,
                    "sha256": hashlib.sha256(paths["mdl"].read_bytes()).hexdigest(),
                },
                "input": {
                    "name": paths["rism1d_input"].name,
                    "sha256": hashlib.sha256(
                        paths["rism1d_input"].read_bytes()
                    ).hexdigest(),
                },
                "xvv": {
                    "name": paths["xvv"].name,
                    "sha256": hashlib.sha256(paths["xvv"].read_bytes()).hexdigest(),
                },
                "stdout": {
                    "name": paths["rism1d_stdout"].name,
                    "sha256": hashlib.sha256(
                        paths["rism1d_stdout"].read_bytes()
                    ).hexdigest(),
                },
            },
            "claim_scope": "Synthetic label-free generation evidence.",
            "containment": copy.deepcopy(
                asset_audit.GENERATION_EVIDENCE_CONTAINMENT
            ),
            "convergence": {
                "exit_status": 0,
                "primary_rism": {
                    "iterations": 3,
                    "final_residual": 5.0e-9,
                    "requested_tolerance": 1.0e-8,
                },
                "temperature_derivative_rism": {
                    "iterations": 2,
                    "final_residual": 4.0e-9,
                    "requested_tolerance": 1.0e-8,
                },
            },
            "generator": {
                "executable_sha256": hashlib.sha256(
                    b"synthetic-rism1d"
                ).hexdigest(),
                "name": "synthetic rism1d",
                "original_invocation": "rism1d synthetic",
                "reproduction_invocation": "rism1d synthetic",
                "reproduction_note": "Synthetic test record only.",
            },
            "recorded_date": "2026-07-29",
        }
    )
    paths["generation_evidence"].write_text(
        json.dumps(evidence, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _manifest(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_kind": "route1-custom-solvent-3drism-asset",
        "solvent": {
            "canonical_name": "synthetic-water-like-solvent",
            "temperature_kelvin": 298.15,
            "pressure_bar": 1.0,
            "dielectric_constant": 78.5,
            "species_density_angstrom_minus3": 0.0333,
            "standard_state": {
                "provider_output": "excess_chemical_potential",
                "benchmark_target": "1M_ideal_gas_to_1M_ideal_solution",
                "conversion_status": "not_yet_applied",
            },
        },
        "site_types": [
            {
                "name": "O",
                "multiplicity": 1,
                "mass_amu": 16.0,
                "charge_e": OXYGEN_CHARGE_E,
                "lj_epsilon_kcal_mol": 0.15,
                "lj_size_angstrom": 1.8,
            },
            {
                "name": "H1",
                "multiplicity": 2,
                "mass_amu": 1.0,
                "charge_e": HYDROGEN_CHARGE_E,
                "lj_epsilon_kcal_mol": 0.02,
                "lj_size_angstrom": 0.65,
            },
        ],
        "site_coordinates_angstrom": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.3, 0.9, 0.0]],
        "rism": {
            "theory": "drism",
            "closure": "kh",
            "grid_spacing_angstrom": 0.025,
            "grid_points": 2,
        },
        "assets": {
            name: {
                "relative_path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
        "provenance": {
            "molecular_model_reference": "synthetic test model; no experimental solvation label",
            "susceptibility_generation": {
                "generator": "rism1d",
                "executable_version": "synthetic-test",
                "source_inputs_immutable": True,
            },
        },
        "containment": {
            "experimental_values_loaded": False,
            "experimental_residual_fit": False,
            "endpoint_selection": False,
            "energy_calibration": False,
            "runtime_provider_enabled": False,
            "accuracy_claim": "none",
        },
    }


def _write_manifest(tmp_path: Path, manifest: dict[str, Any]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _refresh_asset_hash(manifest: dict[str, Any], name: str, path: Path) -> None:
    manifest["assets"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_freezes_no_fit_and_no_runtime_boundary() -> None:
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    assert len(fingerprint) == 64
    assert contract["containment"] == {
        "experimental_values_loaded": False,
        "experimental_residual_fit": False,
        "endpoint_selection": False,
        "energy_calibration": False,
        "runtime_provider_enabled": False,
        "accuracy_claim": "none",
    }
    assert "site charges" in contract["required_physical_inputs"]["molecular_site_model"]


def test_rism_density_converter_handles_documented_molar_units() -> None:
    assert asset_audit._density_to_angstrom_minus3(
        55.345,
        "M",
        molecular_mass_amu=18.016,
    ) == pytest.approx(0.03332953803622)


def test_rism_density_converter_handles_mass_density_units_without_factor_drift() -> None:
    expected = (
        1000.0
        * asset_audit.AVOGADRO_PER_ANGSTROM_CUBED_PER_MOLAR
        / 18.01528
    )
    assert asset_audit._density_to_angstrom_minus3(
        1.0,
        "g/cm3",
        molecular_mass_amu=18.01528,
    ) == pytest.approx(expected)
    assert asset_audit._density_to_angstrom_minus3(
        1000.0,
        "kg/m3",
        molecular_mass_amu=18.01528,
    ) == pytest.approx(expected)


def test_asset_audit_accepts_consistent_molecular_susceptibility_without_labels(
    tmp_path: Path,
) -> None:
    paths = _write_asset_set(tmp_path)
    secret_label = "SECRET_EXPERIMENTAL_DELTA_G_MUST_NOT_ESCAPE"
    (tmp_path / "labels.csv").write_text(secret_label + "\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, _manifest(paths))
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    artifact = asset_audit.audit_manifest(contract, fingerprint, manifest_path)
    serialized = json.dumps(artifact, sort_keys=True)

    assert artifact["content_sha256"] == asset_audit.core.artifact_content_sha256(artifact)
    assert artifact["solvent"]["canonical_name"] == "synthetic-water-like-solvent"
    assert artifact["molecular_site_model"] == {
        "site_type_count": 2,
        "actual_site_count": 3,
        "site_names": ["O", "H1"],
        "multiplicities": [1, 2],
        "net_charge_e": 0.0,
        "molecular_model_reference": "synthetic test model; no experimental solvation label",
    }
    assert artifact["conclusion"]["status"] == "physical_asset_consistent_not_accuracy_validated"
    assert artifact["conclusion"]["generation_evidence_bound"] is True
    assert artifact["conclusion"]["regeneration_parity_validated"] is False
    assert artifact["conclusion"]["runtime_provider_enabled"] is False
    assert artifact["conclusion"]["experimental_values_loaded"] is False
    assert artifact["rism"]["density_input"] == {"value": 0.0333, "units": "1/a^3"}
    assert secret_label not in serialized
    assert "labels.csv" not in serialized


def test_asset_audit_rejects_a_dieletric_only_or_empty_site_model(tmp_path: Path) -> None:
    paths = _write_asset_set(tmp_path)
    manifest = _manifest(paths)
    manifest["site_types"] = []
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="site_types"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )


def test_asset_audit_rejects_charge_pattern_drift_between_model_and_xvv(
    tmp_path: Path,
) -> None:
    paths = _write_asset_set(tmp_path, xvv_hydrogen_charge=5.0)
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="MDL and XVV charges disagrees"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, _manifest(paths)),
        )


def test_asset_audit_rejects_a_proportional_but_non_amber_mdl_charge_scale(
    tmp_path: Path,
) -> None:
    paths = _write_asset_set(tmp_path, mdl_hydrogen_charge=4.0)
    paths["mdl"].write_text(
        paths["mdl"].read_text(encoding="ascii").replace(
            f"{OXYGEN_CHARGE_AMBER:.8E}",
            "-8.00000000E+00",
        ),
        encoding="ascii",
    )
    manifest = _manifest(paths)
    _refresh_asset_hash(manifest, "mdl", paths["mdl"])
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match=r"fixed Amber e\*18\.2223 convention"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )


def test_asset_audit_rejects_thermodynamic_state_drift(tmp_path: Path) -> None:
    paths = _write_asset_set(tmp_path, temperature=300.0)
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="temperature disagrees"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, _manifest(paths)),
        )


def test_asset_audit_rejects_site_type_coordinate_swap_even_when_distance_multiset_matches(
    tmp_path: Path,
) -> None:
    paths = _write_asset_set(tmp_path)
    manifest = _manifest(paths)
    coordinates = manifest["site_coordinates_angstrom"]
    coordinates[0], coordinates[1] = coordinates[1], coordinates[0]
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="site geometry disagrees"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )


def test_geometry_match_allows_proper_rotation_but_rejects_nonplanar_reflection() -> None:
    reference = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    rotated_and_translated = [
        (2.0, -1.0, 3.0),
        (2.0, 0.0, 3.0),
        (1.0, -1.0, 3.0),
        (2.0, -1.0, 4.0),
    ]
    reflected = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
    ]

    asset_audit._match_distances(
        reference,
        rotated_and_translated,
        "proper rotation",
    )
    with pytest.raises(ValueError, match="site geometry disagrees"):
        asset_audit._match_distances(reference, reflected, "reflection")


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("NOT_A_NUMBER", "non-numeric"),
        ("NaN", "non-finite"),
    ],
)
def test_asset_audit_rejects_invalid_xvv_correlation_values(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    paths = _write_asset_set(tmp_path)
    paths["xvv"].write_text(
        paths["xvv"].read_text(encoding="ascii").replace(
            "1.0E+00",
            replacement,
            1,
        ),
        encoding="ascii",
    )
    manifest = _manifest(paths)
    _refresh_asset_hash(manifest, "xvv", paths["xvv"])
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match=message):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )


def test_asset_audit_rejects_extra_xvv_correlation_tokens(tmp_path: Path) -> None:
    paths = _write_asset_set(tmp_path)
    paths["xvv"].write_text(
        paths["xvv"].read_text(encoding="ascii").replace(
            "  1.0E+00  2.0E+00",
            "  0.0E+00  1.0E+00  2.0E+00",
        ),
        encoding="ascii",
    )
    manifest = _manifest(paths)
    _refresh_asset_hash(manifest, "xvv", paths["xvv"])
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="length disagrees with POINTERS"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )


def test_asset_audit_rejects_generation_stdout_not_cross_hashed_by_evidence(
    tmp_path: Path,
) -> None:
    paths = _write_asset_set(tmp_path)
    paths["rism1d_stdout"].write_text(
        paths["rism1d_stdout"].read_text(encoding="ascii").replace(
            "5.0000000000000000E-09",
            "6.0000000000000000E-09",
        ),
        encoding="ascii",
    )
    manifest = _manifest(paths)
    _refresh_asset_hash(manifest, "rism1d_stdout", paths["rism1d_stdout"])
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="stdout asset binding disagrees"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )


def test_asset_audit_rejects_an_input_unbound_from_the_mdl_model(tmp_path: Path) -> None:
    paths = _write_asset_set(tmp_path)
    paths["rism1d_input"].write_text(
        _rism_input().replace("  MODEL='model.mdl',\n", ""), encoding="ascii"
    )
    manifest = _manifest(paths)
    _refresh_asset_hash(manifest, "rism1d_input", paths["rism1d_input"])
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="molecular model"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )


def test_asset_audit_rejects_density_drift_between_input_and_xvv(tmp_path: Path) -> None:
    paths = _write_asset_set(tmp_path)
    paths["rism1d_input"].write_text(_rism_input(density=0.034), encoding="ascii")
    manifest = _manifest(paths)
    _refresh_asset_hash(manifest, "rism1d_input", paths["rism1d_input"])
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="species density"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )


def test_asset_audit_rejects_a_manifest_that_claims_energy_calibration(
    tmp_path: Path,
) -> None:
    paths = _write_asset_set(tmp_path)
    manifest = copy.deepcopy(_manifest(paths))
    manifest["containment"]["energy_calibration"] = True
    contract, fingerprint = asset_audit.load_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="label-free and fail closed"):
        asset_audit.audit_manifest(
            contract,
            fingerprint,
            _write_manifest(tmp_path, manifest),
        )
