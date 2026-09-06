from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_3drism_ethanol_pilot_protocol_v1.json"
ASSET_DIR = BENCHMARK_DIR / "route1_3drism_assets/ethanol-adf4"
CONTRACT_PATH = BENCHMARK_DIR / "route1_custom_solvent_asset_contract_v1.json"
ARTIFACT_PATH = BENCHMARK_DIR / "route1-3drism-ethanol-pilot-2026-07-29.json"
GENERATION_EVIDENCE_PATH = ASSET_DIR / "generation_evidence.json"

PILOT_SPEC = importlib.util.spec_from_file_location(
    "run_route1_3drism_single_solvent_pilot",
    BENCHMARK_DIR / "run_route1_3drism_single_solvent_pilot.py",
)
assert PILOT_SPEC is not None
pilot = importlib.util.module_from_spec(PILOT_SPEC)
assert PILOT_SPEC.loader is not None
PILOT_SPEC.loader.exec_module(pilot)

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "route1_custom_solvent_asset_audit",
    BENCHMARK_DIR / "route1_custom_solvent_asset_audit.py",
)
assert AUDIT_SPEC is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC.loader is not None
AUDIT_SPEC.loader.exec_module(audit)


def _fake_amber_root(tmp_path: Path) -> Path:
    root = tmp_path / "amber"
    binary = root / "bin"
    binary.mkdir(parents=True)
    for name in pilot.REQUIRED_EXECUTABLES:
        (binary / name).write_bytes(b"not executed")
    return root


def test_ethanol_protocol_is_predeclared_label_free_and_nonwater_only() -> None:
    protocol, fingerprint = pilot.load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert "Label-free numerical qualification" in protocol["claim_scope"]
    assert protocol["route1_boundary"]["experimental_solvation_labels_loaded"] is False
    assert protocol["route1_boundary"]["experimental_residual_fit"] is False
    assert protocol["solvent_asset"]["manifest"]["solvent"] == {
        "canonical_name": "ethanol-ADF4",
        "temperature_kelvin": 298.15,
        "pressure_bar": 1.0,
        "dielectric_constant": 24.35,
        "species_density_angstrom_minus3": 0.010267682852133002,
        "standard_state": {
            "provider_output": "excess_chemical_potential",
            "benchmark_target": "1M_ideal_gas_to_1M_ideal_solution",
            "conversion_status": "not_yet_applied",
        },
    }
    assert protocol["solvent_asset"]["manifest"]["rism"] == {
        "theory": "drism",
        "closure": "kh",
        "grid_spacing_angstrom": 0.025,
        "grid_points": 4096,
    }
    assert protocol["solvent_asset"]["amber_data_subdirectory"] == "rism1d/ethanol-adf4-site"
    assert (
        protocol["admission"]["next_gate"]
        == "independent_review_then_standard_state_and_prospective_accuracy_gate"
    )
    molecular_reference = protocol["solvent_asset"]["manifest"]["provenance"][
        "molecular_model_reference"
    ]
    assert "four-site" in molecular_reference
    assert "united-atom" in molecular_reference
    assert "Weight=47.07" in molecular_reference
    assert "46.06844 g/mol" in molecular_reference
    assert "site masses sum to 46.068" in molecular_reference
    assert "not the mass used" in molecular_reference
    assert "Table 7" in molecular_reference
    assert "scm.com/doc/ADF/Input/3D-RISM.html" in molecular_reference
    assert "doi.org/10.1021/je900998f" in molecular_reference
    assert "trc.nist.gov/ThermoML/10.1021/je060248p.html" in molecular_reference
    mass_reference = molecular_reference
    assert "C=12.01, H=1.008, O=16.00" in mass_reference
    assert "CH2=14.026 and CH3=15.034" in mass_reference
    assert "dc0c02d28fb7ee4f04d0a5edee3cd62968ba5e9bdb69e0f728bf01d242d7d062" in (
        mass_reference
    )


def test_committed_ethanol_asset_passes_only_the_physical_input_audit() -> None:
    protocol, _ = pilot.load_protocol(PROTOCOL_PATH)
    contract, contract_fingerprint = audit.load_contract(CONTRACT_PATH)
    artifact = audit.audit_manifest(
        contract,
        contract_fingerprint,
        ASSET_DIR / "manifest.json",
    )

    assert artifact["content_sha256"] == audit.core.artifact_content_sha256(artifact)
    assert artifact["conclusion"] == {
        "status": "physical_asset_consistent_not_accuracy_validated",
        "asset_consistency_scope": (
            "metadata_and_exact_finite_xvv_payload_structure"
        ),
        "generation_evidence_bound": True,
        "regeneration_parity_validated": False,
        "runtime_provider_enabled": False,
        "experimental_values_loaded": False,
        "accuracy_claim": "none",
        "next_gate": (
            "numerical_3drism_provider_parity_then_prospective_per_solvent_"
            "and_external_accuracy_validation"
        ),
    }
    assert artifact["assets"]["mdl"]["sha256"] == (
        "114fa832fcda52227f96d4bcec9480b5484cab4ab75e4f384ec7a6b1fd2f077e"
    )
    assert artifact["assets"]["rism1d_input"]["sha256"] == (
        "71ebc45f07c783ba7718ccc7a2390685e1220026a543324c6521d4b848d9b170"
    )
    assert artifact["assets"]["xvv"]["sha256"] == (
        "aee345c3b7edb59d46509b6e0afaa65e533e4383abc6703c02ea8e0ac69b7c7c"
    )
    assert artifact["assets"]["rism1d_stdout"]["sha256"] == (
        "10de93f1f9cc38c746e8e4054f89a277a89c66953e32c72008fdb84845b01824"
    )
    assert artifact["assets"]["generation_evidence"]["sha256"] == (
        "aab6f052c3990f9f4f76bffae821b3bc5c894716be088b9b5ac54a40532cf1ca"
    )
    assert artifact["rism"]["generation_evidence"]["bound"] is True
    assert artifact["rism"]["generation_evidence"]["regeneration_parity_validated"] is False
    assert artifact["containment"]["runtime_provider_enabled"] is False
    assert protocol["solvent_asset"]["manifest"] == json.loads(
        (ASSET_DIR / "manifest.json").read_text(encoding="utf-8")
    )


def test_ethanol_rism1d_generation_evidence_is_sealed_and_cross_hashed() -> None:
    evidence = json.loads(GENERATION_EVIDENCE_PATH.read_text(encoding="utf-8"))
    output = (ASSET_DIR / "ethanol_adf4_kh.rism1d.out").read_text(
        encoding="ascii"
    )

    assert evidence["content_sha256"] == pilot.core.artifact_content_sha256(evidence)
    for entry in evidence["assets"].values():
        assert entry["sha256"] == pilot.core.sha256_file(ASSET_DIR / entry["name"])
    assert evidence["convergence"] == {
        "exit_status": 0,
        "primary_rism": {
            "iterations": 137,
            "final_residual": 9.573592906123781e-09,
            "requested_tolerance": 1e-08,
        },
        "temperature_derivative_rism": {
            "iterations": 79,
            "final_residual": 9.569571046018302e-09,
            "requested_tolerance": 1e-08,
        },
    }
    assert (
        "step= 137     Res=  9.5735929061237814E-09" in output
        and "relaxing RISM DT:" in output
        and "step=  79     Res=  9.5695710460183020E-09" in output
    )
    assert evidence["containment"]["experimental_solvation_labels_loaded"] is False
    assert evidence["containment"]["accuracy_claim"] == "none"


def test_committed_ethanol_pilot_is_sealed_and_numerical_only() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    _, protocol_fingerprint = pilot.load_protocol(PROTOCOL_PATH)

    assert artifact["content_sha256"] == pilot.core.artifact_content_sha256(artifact)
    assert artifact["protocol_fingerprint"] == protocol_fingerprint
    assert artifact["command_provenance"]["script_sha256"] == pilot.core.sha256_file(
        BENCHMARK_DIR / "run_route1_3drism_single_solvent_pilot.py"
    )
    assert artifact["command_provenance"]["arguments"]["solvent_data_root"].endswith(
        "docs/implicit-solvation/benchmarks/route1_3drism_assets/ethanol-adf4"
    )
    assert artifact["solvent_asset_audit"]["conclusion"]["status"] == (
        "physical_asset_consistent_not_accuracy_validated"
    )
    topology_hashes = artifact["solute"]["generated_topology_hashes"]
    assert topology_hashes["molecule.prmtop"]["normalization"] == (
        "amber_prmtop_version_date_header_v1"
    )
    assert all(
        len(record["reproducibility_sha256"]) == 64
        for record in topology_hashes.values()
    )
    assert artifact["admission"] == {
        "all_cases_converged": True,
        "all_case_thresholds_passed": True,
        "passed": True,
        "status": "one_nonwater_solver_pilot_numerically_qualified_not_accuracy_validated",
        "accuracy_claim": "none",
        "runtime_provider_enabled": False,
        "next_gate": "independent_review_then_standard_state_and_prospective_accuracy_gate",
    }

    records = {record["id"]: record for record in artifact["cases"]}
    assert set(records) == {
        "grid_sensitivity",
        "buffer_sensitivity",
        "tolerance_sensitivity",
        "reference",
        "reference_repeat",
    }
    assert all(record["solver"]["converged"] for record in records.values())
    assert all(record["primary_difference_gate_passed"] for record in records.values())
    assert (
        records["reference_repeat"]["primary_difference_from_reference_kcal_mol"]
        <= 1e-08
    )
    assert all(
        "solutePotentialEnergy" not in record["solver"] for record in records.values()
    )


def test_explicit_solvent_root_fails_closed_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Explicit solvent asset directory"):
        pilot.run_pilot(
            protocol_path=PROTOCOL_PATH,
            amber_root_path=_fake_amber_root(tmp_path),
            solvent_data_root_path=tmp_path / "missing-solvent",
            output_path=tmp_path / "result.json",
            timeout_seconds=1.0,
            recorded_date="2026-07-29",
        )


def test_explicit_solvent_root_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    solvent_root = tmp_path / "wrong-solvent"
    solvent_root.mkdir()
    protocol, _ = pilot.load_protocol(PROTOCOL_PATH)
    for entry in protocol["solvent_asset"]["files"].values():
        (solvent_root / entry["name"]).write_bytes(b"wrong")

    with pytest.raises(ValueError, match="Pinned solvent asset changed"):
        pilot.run_pilot(
            protocol_path=PROTOCOL_PATH,
            amber_root_path=_fake_amber_root(tmp_path),
            solvent_data_root_path=solvent_root,
            output_path=tmp_path / "result.json",
            timeout_seconds=1.0,
            recorded_date="2026-07-29",
        )
