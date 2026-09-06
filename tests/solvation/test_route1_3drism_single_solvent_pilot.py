from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = (
    BENCHMARK_DIR / "route1_3drism_single_solvent_pilot_protocol_v1.json"
)
ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-3drism-single-solvent-pilot-2026-07-29.json"
)
SPEC = importlib.util.spec_from_file_location(
    "run_route1_3drism_single_solvent_pilot",
    BENCHMARK_DIR / "run_route1_3drism_single_solvent_pilot.py",
)
assert SPEC is not None
pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pilot)


def _write_protocol(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def test_protocol_is_label_free_raw_primary_and_research_only() -> None:
    protocol, fingerprint = pilot.load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert protocol["status"] == "research_only_not_accuracy_validated"
    assert protocol["route1_boundary"]["experimental_solvation_labels_loaded"] is False
    assert protocol["route1_boundary"]["experimental_residual_fit"] is False
    assert protocol["solver"]["primary_convention"] == "raw_kh_excess_chemical_potential"
    assert protocol["solver"]["pc_plus_is_primary"] is False
    assert protocol["solver"]["standard_state_conversion_applied"] is False
    assert protocol["solvent_asset"]["files"]["xvv"]["name"] == "cSPCE_pse3.xvv"
    assert protocol["solvent_asset"]["manifest"]["rism"] == {
        "theory": "drism",
        "closure": "pse3",
        "grid_spacing_angstrom": 0.025,
        "grid_points": 16384,
    }


def test_protocol_rejects_label_access_and_pc_plus_promotion(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    label_open = copy.deepcopy(protocol)
    label_open["route1_boundary"]["experimental_solvation_labels_loaded"] = True
    with pytest.raises(ValueError, match="Route 1 boundary"):
        pilot.load_protocol(_write_protocol(tmp_path, label_open))

    pc_plus = copy.deepcopy(protocol)
    pc_plus["solver"]["pc_plus_is_primary"] = True
    with pytest.raises(ValueError, match="solver profile"):
        pilot.load_protocol(_write_protocol(tmp_path, pc_plus))


def test_protocol_freezes_reference_and_physical_sensitivity_thresholds() -> None:
    protocol, _ = pilot.load_protocol(PROTOCOL_PATH)
    cases = {case["id"]: case for case in protocol["cases"]}

    assert cases["grid_sensitivity"]["maximum_primary_difference_kcal_mol"] == 0.02
    assert cases["buffer_sensitivity"]["maximum_primary_difference_kcal_mol"] == 0.01
    assert cases["tolerance_sensitivity"]["maximum_primary_difference_kcal_mol"] == 0.005
    assert cases["reference"]["maximum_primary_difference_kcal_mol"] == 0.0
    assert cases["reference_repeat"]["comparison_to_reference"] == (
        "independent_process_determinism"
    )
    assert cases["reference_repeat"]["maximum_primary_difference_kcal_mol"] == 1e-08
    assert protocol["admission"]["all_cases_must_converge"] is True
    assert protocol["admission"]["all_case_thresholds_must_pass"] is True
    assert protocol["admission"]["next_gate"] == (
        "independent_review_then_one_predeclared_nonwater_asset_pilot"
    )


def test_parse_rism_output_extracts_only_admitted_quantities() -> None:
    output = """\
|grid size:        112 X        108 X        128
|grid spacing [A]:      0.300 X      0.300 X      0.300
|effective buffer [A]:    14.443,      14.661,      14.045
|RXRISM converged in    97 steps
solutePotentialEnergy                    -1.2340000000000000E+002
rism_excessChemicalPotential               2.3633548400394368E+001  1.0  -2.0
rism_excessChemicalPotentialGF             1.3448362770540093E+001  1.0  -2.0
rism_excessChemicalPotentialPCPLUS        -6.3717252788989889E+000
rism_partialMolarVolume                    2.4346011045276029E+002
"""
    parsed = pilot.parse_rism_output(output)

    assert parsed["converged"] is True
    assert parsed["iterations"] == 97
    assert parsed["grid_points_xyz"] == [112, 108, 128]
    assert parsed["raw_excess_chemical_potential_kcal_mol"] == pytest.approx(
        23.633548400394368
    )
    assert parsed["pc_plus_excess_chemical_potential_kcal_mol"] == pytest.approx(
        -6.371725278898989
    )
    assert "solutePotentialEnergy" not in parsed


def test_parse_rism_output_fails_closed_without_convergence() -> None:
    with pytest.raises(ValueError, match="convergence"):
        pilot.parse_rism_output(
            """\
|grid size: 1 X 1 X 1
|grid spacing [A]: 0.3 X 0.3 X 0.3
|effective buffer [A]: 1.0, 1.0, 1.0
rism_excessChemicalPotential 1.0
rism_excessChemicalPotentialGF 1.0
rism_excessChemicalPotentialPCPLUS 1.0
rism_partialMolarVolume 1.0
"""
        )


def test_generated_prmtop_reproducibility_hash_normalizes_only_date(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.prmtop"
    second = tmp_path / "second.prmtop"
    first.write_text(
        "%VERSION  VERSION_STAMP = V0001.000  DATE = 07/29/26  12:00:00\n"
        "%FLAG TITLE\n%FORMAT(20a4)\nMOL\n",
        encoding="ascii",
    )
    second.write_text(
        "%VERSION  VERSION_STAMP = V0001.000  DATE = 07/29/26  12:00:01\n"
        "%FLAG TITLE\n%FORMAT(20a4)\nMOL\n",
        encoding="ascii",
    )

    first_hashes = pilot._generated_file_hashes(first)
    altered = tmp_path / "altered.prmtop"
    altered.write_text(
        "%VERSION  VERSION_STAMP = V0001.000  DATE = 07/29/26  12:00:02\n"
        "%FLAG TITLE\n%FORMAT(20a4)\nDIFF\n",
        encoding="ascii",
    )
    second_hashes = pilot._generated_file_hashes(second)
    altered_hashes = pilot._generated_file_hashes(altered)

    assert first_hashes["byte_sha256"] != second_hashes["byte_sha256"]
    assert (
        first_hashes["reproducibility_sha256"]
        == second_hashes["reproducibility_sha256"]
    )
    assert first_hashes["normalization"] == (
        "amber_prmtop_version_date_header_v1"
    )
    assert second_hashes["normalization"] == (
        "amber_prmtop_version_date_header_v1"
    )
    assert (
        first_hashes["reproducibility_sha256"]
        != altered_hashes["reproducibility_sha256"]
    )


def test_generated_prmtop_reproducibility_hash_fails_closed_without_date(
    tmp_path: Path,
) -> None:
    prmtop = tmp_path / "missing-date.prmtop"
    prmtop.write_text(
        "%VERSION  VERSION_STAMP = V0001.000\n"
        "%FLAG TITLE\n%FORMAT(20a4)\nMOL\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="exactly one dated"):
        pilot._generated_file_hashes(prmtop)


def test_generated_prmtop_reproducibility_hash_rejects_date_line_payload(
    tmp_path: Path,
) -> None:
    prmtop = tmp_path / "unexpected-date-payload.prmtop"
    prmtop.write_text(
        "%VERSION  VERSION_STAMP = V0001.000  "
        "DATE = 07/29/26  12:00:00 UNBOUND\n"
        "%FLAG TITLE\n%FORMAT(20a4)\nMOL\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="exactly one dated"):
        pilot._generated_file_hashes(prmtop)


def test_committed_pilot_artifact_is_sealed_reproducible_and_not_accuracy_claim() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    _, protocol_fingerprint = pilot.load_protocol(PROTOCOL_PATH)

    assert artifact["content_sha256"] == pilot.core.artifact_content_sha256(artifact)
    assert artifact["protocol_fingerprint"] == protocol_fingerprint
    assert artifact["command_provenance"]["script_sha256"] == pilot.core.sha256_file(
        BENCHMARK_DIR / "run_route1_3drism_single_solvent_pilot.py"
    )
    assert artifact["solvent_asset_audit"]["conclusion"]["status"] == (
        "physical_asset_consistent_not_accuracy_validated"
    )
    assert artifact["solvent_asset_audit"]["assets"]["xvv"]["sha256"] == (
        "9e6cc013756b0fc9af24550112e11323b68f152fa8a3ca111db6bf2924dcdbc6"
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
        "status": "one_water_solver_pilot_numerically_qualified_not_accuracy_validated",
        "accuracy_claim": "none",
        "runtime_provider_enabled": False,
        "next_gate": "independent_review_then_one_predeclared_nonwater_asset_pilot",
    }
    records = {record["id"]: record for record in artifact["cases"]}
    assert set(records) == {
        "grid_sensitivity",
        "buffer_sensitivity",
        "tolerance_sensitivity",
        "reference",
        "reference_repeat",
    }
    assert all(record["primary_difference_gate_passed"] for record in records.values())
    assert (
        records["reference_repeat"]["primary_difference_from_reference_kcal_mol"]
        <= 1e-08
    )
    assert all(
        "solutePotentialEnergy" not in record["solver"] for record in records.values()
    )
