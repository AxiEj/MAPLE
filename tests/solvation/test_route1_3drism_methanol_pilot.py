from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_3drism_methanol_pilot_protocol_v1.json"
ASSET_DIR = BENCHMARK_DIR / "route1_3drism_assets/methanol-adf3"
CONTRACT_PATH = BENCHMARK_DIR / "route1_custom_solvent_asset_contract_v1.json"
ARTIFACT_PATH = BENCHMARK_DIR / "route1-3drism-methanol-pilot-2026-07-29.json"
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


def test_methanol_protocol_is_predeclared_label_free_and_nonwater_only() -> None:
    protocol, fingerprint = pilot.load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert "first non-water pilot" in protocol["claim_scope"]
    assert protocol["route1_boundary"]["experimental_solvation_labels_loaded"] is False
    assert protocol["route1_boundary"]["experimental_residual_fit"] is False
    assert protocol["solvent_asset"]["manifest"]["solvent"] == {
        "canonical_name": "methanol-ADF3",
        "temperature_kelvin": 298.15,
        "pressure_bar": 1.0,
        "dielectric_constant": 32.63,
        "species_density_angstrom_minus3": 0.014784355565800001,
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
    assert protocol["admission"]["next_gate"] == (
        "independent_review_then_standard_state_and_prospective_accuracy_gate"
    )

    cases = {case["id"]: case for case in protocol["cases"]}
    assert cases["grid_sensitivity"]["maximum_primary_difference_kcal_mol"] == 0.02
    assert cases["buffer_sensitivity"]["maximum_primary_difference_kcal_mol"] == 0.01
    assert cases["tolerance_sensitivity"]["maximum_primary_difference_kcal_mol"] == 0.005
    assert cases["reference_repeat"]["maximum_primary_difference_kcal_mol"] == 1e-08


def test_committed_methanol_asset_passes_only_the_physical_input_audit() -> None:
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
        "571a1d230bf6efcc5f76c4a2292e369a45f444eaefb980539cc77143a7bf296f"
    )
    assert artifact["assets"]["rism1d_input"]["sha256"] == (
        "0122293b5c84b3eda8f1243d9f51e59c19741fca8ed533b8fa46b0d4ae8e5a8f"
    )
    assert artifact["assets"]["xvv"]["sha256"] == (
        "2035e47acdea9d378466e37821d6445414e4fda834da4424f196fe9a3cafd2ad"
    )
    assert artifact["assets"]["rism1d_stdout"]["sha256"] == (
        "ae1883f9427e59c18ca2ee9e4adda94748e21d476f0f54f58c0976fd62f0f7bc"
    )
    assert artifact["assets"]["generation_evidence"]["sha256"] == (
        "9fc8eeed88135da192c93a5d070b41cd764c7192d296ce876d478c7e32a7c3dd"
    )
    assert artifact["rism"]["generation_evidence"]["bound"] is True
    assert artifact["rism"]["generation_evidence"]["regeneration_parity_validated"] is False
    assert protocol["solvent_asset"]["manifest"] == json.loads(
        (ASSET_DIR / "manifest.json").read_text(encoding="utf-8")
    )


def test_methanol_rism1d_generation_evidence_is_sealed_and_cross_hashed() -> None:
    evidence = json.loads(GENERATION_EVIDENCE_PATH.read_text(encoding="utf-8"))
    output = (ASSET_DIR / "methanol_adf3_kh.rism1d.out").read_text(
        encoding="ascii"
    )

    assert evidence["content_sha256"] == pilot.core.artifact_content_sha256(evidence)
    for entry in evidence["assets"].values():
        assert entry["sha256"] == pilot.core.sha256_file(ASSET_DIR / entry["name"])
    assert evidence["convergence"] == {
        "exit_status": 0,
        "primary_rism": {
            "iterations": 90,
            "final_residual": 4.137214622317103e-09,
            "requested_tolerance": 1e-08,
        },
        "temperature_derivative_rism": {
            "iterations": 66,
            "final_residual": 4.8501235289513124e-09,
            "requested_tolerance": 1e-08,
        },
    }
    assert (
        "step=  90     Res=  4.1372146223171028E-09" in output
        and "relaxing RISM DT:" in output
        and "step=  66     Res=  4.8501235289513124E-09" in output
    )
    assert evidence["containment"]["experimental_solvation_labels_loaded"] is False
    assert evidence["containment"]["accuracy_claim"] == "none"


def test_committed_methanol_pilot_is_sealed_and_numerical_only() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    _, protocol_fingerprint = pilot.load_protocol(PROTOCOL_PATH)

    assert artifact["content_sha256"] == pilot.core.artifact_content_sha256(artifact)
    assert artifact["protocol_fingerprint"] == protocol_fingerprint
    assert artifact["command_provenance"]["script_sha256"] == pilot.core.sha256_file(
        BENCHMARK_DIR / "run_route1_3drism_single_solvent_pilot.py"
    )
    assert artifact["command_provenance"]["arguments"]["solvent_data_root"].endswith(
        "docs/implicit-solvation/benchmarks/route1_3drism_assets/methanol-adf3"
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
        "status": (
            "one_nonwater_solver_pilot_numerically_qualified_not_accuracy_validated"
        ),
        "accuracy_claim": "none",
        "runtime_provider_enabled": False,
        "next_gate": (
            "independent_review_then_standard_state_and_prospective_accuracy_gate"
        ),
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
