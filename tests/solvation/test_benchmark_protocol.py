from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_conformer_sensitivity as conformers
import run_freesolv as runner
import run_provider_parity as parity


MOL2 = """@<TRIPOS>MOLECULE
METHANE
 5 4 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3       1 MOL     -0.100000
      2 H1          0.6000    0.6000    0.6000 H         1 MOL      0.025000
      3 H2         -0.6000   -0.6000    0.6000 H         1 MOL      0.025000
      4 H3         -0.6000    0.6000   -0.6000 H         1 MOL      0.025000
      5 H4          0.6000   -0.6000   -0.6000 H         1 MOL      0.025000
@<TRIPOS>BOND
     1 1 2 1
     2 1 3 1
     3 1 4 1
     4 1 5 1
@<TRIPOS>SUBSTRUCTURE
     1 MOL 1 TEMP 0 **** **** 0 ROOT
"""


def _write_fixture_protocol(
    tmp_path: Path, protocol_name: str = "protocol.json"
) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    database_text = (
        "# fixture\n"
        "mobley_test; C; methane; -1.00; 0.10; -0.80; 0.02; fixture-ref; "
        "fixture-calc; fixture notes\n"
    )
    database_json = {
        "mobley_test": {
            "smiles": "C",
            "iupac": "methane",
            "expt": -1.0,
            "d_expt": 0.1,
            "expt_reference": "fixture-ref",
            "groups": ["alkane"],
        }
    }
    (source / "database.txt").write_text(database_text, encoding="utf-8")
    (source / "database.json").write_text(
        json.dumps(database_json), encoding="utf-8"
    )
    mol2_root = tmp_path / "archive-root" / "mol2files_gaff"
    mol2_root.mkdir(parents=True)
    (mol2_root / "mobley_test.mol2").write_text(MOL2, encoding="utf-8")
    with tarfile.open(source / "mol2files_gaff.tar.gz", "w:gz") as archive:
        archive.add(mol2_root.parent / "mol2files_gaff", arcname="mol2files_gaff")

    protocol = json.loads((BENCHMARK_DIR / protocol_name).read_text(encoding="utf-8"))
    protocol["dataset"]["commit"] = "0" * 40
    protocol["dataset"]["expected_record_count"] = 1
    for artifact in protocol["dataset"]["artifacts"]:
        artifact["url"] = f"https://example.invalid/{artifact['name']}"
        artifact["sha256"] = core.sha256_file(source / artifact["name"])
    protocol["partition"]["pilot_development_only"] = ["mobley_test"]
    protocol["methods"]["charge_methods"] = ["am1bcc"]
    protocol["methods"]["gb_models"] = ["hct"]
    protocol["providers"]["openmm"]["required_version"] = importlib.metadata.version(
        "openmm"
    )
    protocol["statistics"]["bootstrap_resamples"] = 100
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return protocol_path, source


def _prepare_fixture(tmp_path: Path) -> tuple[Path, Path]:
    protocol_path, source = _write_fixture_protocol(tmp_path)
    work = tmp_path / "work"
    runner.prepare(
        argparse.Namespace(
            protocol=str(protocol_path), work_dir=str(work), source_dir=str(source)
        )
    )
    return protocol_path, work




def test_repository_protocol_is_pinned_and_schema_complete():
    protocol, fingerprint = core.load_protocol(BENCHMARK_DIR / "protocol.json")

    assert len(fingerprint) == 64
    assert protocol["dataset"]["expected_record_count"] == 642
    assert protocol["methods"]["charge_methods"] == ["am1bcc", "abcg2"]
    assert protocol["methods"]["gb_models"] == ["hct", "obc1", "obc2", "gbn", "gbn2"]
    assert len(protocol["partition"]["pilot_development_only"]) == 10
    assert "finite-temperature" in protocol["claim_scope"]
    assert protocol["providers"]["apbs"]["required_version"] == "1.4.1"
    assert protocol["providers"]["openmm"]["method_restrictions"]["gbn2"][
        "unsupported_elements"
    ] == ["P"]


def test_frozen_development_summary_reconciles_all_attempts():
    protocol, fingerprint = core.load_protocol(BENCHMARK_DIR / "protocol.json")
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-development-2026-07-23.json"
    )

    assert summary["protocol_fingerprint"] == fingerprint
    assert summary["partition"] == "development"
    assert summary["candidate_count"] == 526
    assert summary["attempt_count"] == 5260
    assert summary["success_count"] == 5238
    assert summary["failure_count"] == 22
    assert len(summary["record_sha256"]) == summary["attempt_count"]
    assert set(summary["methods"]) == {
        f"{charge}/{model}"
        for charge in protocol["methods"]["charge_methods"]
        for model in protocol["methods"]["gb_models"]
    }
    assert summary["observed_provider_versions"] == {
        "ambertools": ["26.0"],
        "openmm": ["8.5.2"],
    }
    assert summary["confirmation_lock_sha256"] is None
    assert summary["methods"]["abcg2/obc2"]["mae"] == pytest.approx(
        1.6522994515889533
    )
    assert {
        (failure["charge_method"], failure["gb_model"], failure["phase"])
        for failure in summary["failures"]
    } == {
        ("am1bcc", "gbn2", "gb"),
        ("abcg2", "gbn2", "gb"),
    }
    assert all("phosphorus" in failure["reason"] for failure in summary["failures"])


def test_conformer_protocol_is_development_only_and_base_hash_pinned():
    protocol, fingerprint, protocol_path = conformers._load_conformer_protocol(
        BENCHMARK_DIR / "conformer_protocol.json"
    )
    base_summary = BENCHMARK_DIR / protocol["base_evidence"]["development_summary"]

    assert len(fingerprint) == 64
    assert protocol["source_partition"] == "development"
    assert len(protocol["cases"]) == 20
    assert protocol["selection"]["per_stratum"] == {
        "rigid": 4,
        "limited": 8,
        "flexible": 8,
    }
    assert core.sha256_file(base_summary) == protocol["base_evidence"][
        "development_summary_sha256"
    ]
    assert conformers._relative_protocol_path(
        protocol_path, protocol["base_evidence"]["protocol"]
    ) == BENCHMARK_DIR / "protocol.json"
    assert protocol["evaluation"]["weighting"].startswith("None.")


def test_conformer_xyz_parser_preserves_atom_order_and_method_stats(tmp_path):
    ensemble = tmp_path / "ensemble.xyz"
    ensemble.write_text(
        "2\n-1.0\nH 0 0 0\nF 0 0 1\n"
        "2\n-0.5\nH 0 0 0\nF 0 1 0\n",
        encoding="utf-8",
    )

    frames = conformers._read_xyz_ensemble(ensemble, ["H", "F"])
    stats = conformers._method_stats([-2.0, 1.0], reference=-0.5)

    assert [frame["crest_energy_hartree"] for frame in frames] == [-1.0, -0.5]
    assert frames[1]["positions_angstrom"][1].tolist() == [0.0, 1.0, 0.0]
    assert stats["range_kcal_mol"] == pytest.approx(3.0)
    assert stats["max_abs_delta_from_reference_kcal_mol"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="atom order"):
        conformers._read_xyz_ensemble(ensemble, ["F", "H"])


def test_frozen_conformer_summary_reconciles_cases_and_declared_failures():
    protocol, fingerprint, _protocol_path = conformers._load_conformer_protocol(
        BENCHMARK_DIR / "conformer_protocol.json"
    )
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-conformer-sensitivity-2026-07-23.json"
    )

    assert summary["protocol_fingerprint"] == fingerprint
    assert summary["source_partition"] == "development"
    assert summary["case_count"] == 20
    assert summary["successful_generator_case_count"] == 20
    assert summary["generator_failure_count"] == 0
    assert summary["total_conformer_count"] == 1294
    assert sum(summary["conformer_count_by_case"].values()) == 1294
    assert len(summary["record_sha256"]) == 20
    assert summary["environment"]["crest"]["version"] == "3.0.2"
    assert summary["environment"]["xtb"]["version"] == "6.7.1"
    assert summary["environment"]["openmm_version"] == "8.5.2"
    assert summary["conformer_count_by_flexibility"]["rigid"]["maximum"] == 1
    assert summary["conformer_count_by_flexibility"]["flexible"]["median"] == pytest.approx(
        134.5
    )
    assert summary["methods"]["abcg2/obc2"]["conformer_range_kcal_mol"][
        "p90"
    ] == pytest.approx(1.7906107608161428)
    for key, method in summary["methods"].items():
        expected_success = 19 if key.endswith("/gbn2") else 20
        assert method["successful_case_count"] == expected_success
        if key.endswith("/gbn2"):
            assert [failure["compound_id"] for failure in method["failures"]] == [
                "mobley_1770205"
            ]
            assert "phosphorus" in method["failures"][0]["reason"]
        else:
            assert method["failures"] == []










def test_provider_reference_manifests_are_hash_pinned_and_cover_declared_controls():
    protocol, _fingerprint = core.load_protocol(BENCHMARK_DIR / "protocol.json")
    amber_path = REPOSITORY_ROOT / protocol["provider_parity"][
        "amber_gb_reference_manifest"
    ]
    apbs_path = REPOSITORY_ROOT / protocol["provider_parity"]["apbs_reference_manifest"]
    amber = core.load_json(amber_path)
    apbs = core.load_json(apbs_path)

    required_models = set(protocol["methods"]["gb_models"])
    assert amber["provider"] == "amber"
    assert len(amber["cases"]) == 5
    for case in amber["cases"]:
        mol2 = (amber_path.parent / case["mol2"]).resolve()
        assert core.sha256_file(mol2) == case["mol2_sha256"]
        assert set(case["models"]) == required_models

    assert apbs["provider"] == "apbs"
    assert {case["control_kind"] for case in apbs["cases"]} == set(
        protocol["provider_parity"]["apbs_required_controls"]
    )
    for case in apbs["cases"]:
        if case["control_kind"] == "official-born-ion":
            source = (apbs_path.parent / case["pqr"]).resolve()
            assert core.sha256_file(source) == case["pqr_sha256"]
            assert case["expected_kj_mol"]["polar"] == pytest.approx(-229.59)
        else:
            source = (apbs_path.parent / case["mol2"]).resolve()
            assert core.sha256_file(source) == case["mol2_sha256"]
            assert len(case["grids"]) >= 5


def test_official_born_control_matches_documented_apbs_settings():
    rendered = parity._render_official_born_input("born.pqr", 97, 0.33)

    assert "dime 97 97 97" in rendered
    assert rendered.count("grid 0.33 0.33 0.33") == 2
    assert "sdie 78.54" in rendered
    assert "sdie 1" in rendered
    assert "lpbe" in rendered
    assert "print energy solv - ref end" in rendered


def test_measured_provider_evidence_is_pinned_but_proposed_tolerances_are_not_frozen(
    tmp_path,
):
    observations_path = REPOSITORY_ROOT / "tests/solvation/data/provider_parity_observations.json"
    proposal_path = (
        REPOSITORY_ROOT
        / "tests/solvation/data/provider_parity_tolerances.proposed.json"
    )
    observations = core.load_json(observations_path)
    proposal = core.load_json(proposal_path)

    assert proposal["evidence_artifact_sha256"] == core.sha256_file(observations_path)
    assert observations["protocol_fingerprint"] == proposal["protocol_fingerprint"]
    for path_key, hash_key in (
        ("amber_openmm_results", "amber_openmm_results_sha256"),
        ("apbs_grid_results", "apbs_grid_results_sha256"),
    ):
        raw_path = REPOSITORY_ROOT / observations["source_artifacts"][path_key]
        assert core.sha256_file(raw_path) == observations["source_artifacts"][hash_key]
    assert proposal["review_status"] == "proposed-awaiting-human-review"
    with pytest.raises(ValueError, match="not marked human-reviewed-frozen"):
        parity.verify(
            argparse.Namespace(
                protocol=str(BENCHMARK_DIR / "protocol.json"),
                artifact_dir=str(tmp_path),
                tolerances=str(proposal_path),
            )
        )


def test_prepare_verifies_hashes_and_forces_pilot_into_development(tmp_path):
    protocol_path, work = _prepare_fixture(tmp_path)
    manifest = core.load_json(work / "prepared.json")

    assert manifest["candidate_count"] == 1
    assert manifest["exclusion_count"] == 0
    assert manifest["candidates"][0]["partition"] == "development"
    assert manifest["candidates"][0]["pilot_development_only"] is True
    assert manifest["candidates"][0]["mol2_sha256"] == core.sha256_file(
        work / "dataset/mol2files_gaff/mobley_test.mol2"
    )

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["dataset"]["artifacts"][0]["sha256"] = "f" * 64
    broken = tmp_path / "broken-protocol.json"
    broken.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.prepare(
            argparse.Namespace(
                protocol=str(broken),
                work_dir=str(tmp_path / "broken-work"),
                source_dir=str(tmp_path / "source"),
            )
        )


def test_structure_group_split_is_deterministic_and_seeded():
    one = core.partition_for_smiles(
        "CCO", seed="fixed", development_fraction=0.8
    )
    two = core.partition_for_smiles(
        "CCO", seed="fixed", development_fraction=0.8
    )
    forced = core.partition_for_smiles(
        "CCO", seed="fixed", development_fraction=0.0, forced_development=True
    )

    assert one == two
    assert forced == "development"


def test_run_resumes_without_overwriting_and_summary_is_byte_stable(
    tmp_path, monkeypatch
):
    protocol_path, work = _prepare_fixture(tmp_path)

    class FakeChargeResult:
        charges = np.array([-0.1, 0.025, 0.025, 0.025, 0.025])
        provenance = {"provider": "fixture", "sum_e": 0.0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def evaluate(self, _atoms, need_forces=False):
            assert need_forces is False
            return SimpleNamespace(
                energy_hartree=1.0 / runner.KCAL_PER_HARTREE,
                components_hartree={
                    "polar": 0.75 / runner.KCAL_PER_HARTREE,
                    "nonpolar": 0.25 / runner.KCAL_PER_HARTREE,
                },
                provenance={"provider": "fixture-openmm"},
            )

    monkeypatch.setattr(runner, "prepare_charges", lambda *_args, **_kwargs: FakeChargeResult())
    monkeypatch.setattr(runner, "OpenMMGB", FakeProvider)
    namespace = argparse.Namespace(
        protocol=str(protocol_path),
        work_dir=str(work),
        partition="development",
        antechamber=None,
        max_compounds=None,
        jobs=2,
    )
    runner.run(namespace)
    record_path = work / "records/development/mobley_test__am1bcc__hct.json"
    first_hash = core.sha256_file(record_path)
    runner.run(namespace)
    assert core.sha256_file(record_path) == first_hash

    first_summary = tmp_path / "summary-1.json"
    second_summary = tmp_path / "summary-2.json"
    runner.summarize(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            output=str(first_summary),
        )
    )
    runner.summarize(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            output=str(second_summary),
        )
    )
    assert first_summary.read_bytes() == second_summary.read_bytes()
    summary = core.load_json(first_summary)
    metrics = summary["methods"]["am1bcc/hct"]
    assert metrics["n"] == 1
    assert metrics["mse"] == pytest.approx(2.0)
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)
    assert metrics["failure_rate"] == 0.0
    assert summary["environment"]["openmm_version"] == importlib.metadata.version(
        "openmm"
    )
    assert summary["observed_provider_versions"] == {
        "ambertools": [],
        "openmm": [],
    }


def test_run_jobs_dispatches_each_candidate_once(tmp_path, monkeypatch):
    protocol_path, work = _prepare_fixture(tmp_path)
    prepared_path = work / "prepared.json"
    prepared = core.load_json(prepared_path)
    second = dict(prepared["candidates"][0])
    second["compound_id"] = "mobley_test_2"
    prepared["candidates"].append(second)
    prepared["candidate_count"] = 2
    prepared["partition_counts"]["development"] = 2
    core.write_json_atomic(prepared_path, prepared)

    seen = []

    def fake_run_candidate(candidate, **_kwargs):
        seen.append(candidate["compound_id"])
        return 1, 0

    monkeypatch.setattr(runner, "_run_candidate", fake_run_candidate)
    runner.run(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            antechamber=None,
            max_compounds=None,
            jobs=2,
        )
    )

    assert sorted(seen) == ["mobley_test", "mobley_test_2"]


def test_provider_failure_is_retained_and_denominator_is_unchanged(tmp_path, monkeypatch):
    protocol_path, work = _prepare_fixture(tmp_path)

    def fail_provider(*_args, **_kwargs):
        raise RuntimeError("fixture charge provider failed")

    monkeypatch.setattr(runner, "prepare_charges", fail_provider)
    runner.run(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            antechamber=None,
            max_compounds=None,
        )
    )
    output = tmp_path / "failure-summary.json"
    runner.summarize(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            output=str(output),
        )
    )
    summary = core.load_json(output)
    metrics = summary["methods"]["am1bcc/hct"]
    assert metrics["expected_count"] == 1
    assert metrics["n"] == 0
    assert metrics["failure_rate"] == 1.0
    assert summary["failures"][0]["phase"] == "charge"
    assert summary["failures"][0]["provider"] == "ambertools-antechamber"
    assert summary["failures"][0]["command"] is None
    assert summary["failures"][0]["returncode"] is None


def test_missing_attempts_cannot_be_summarized(tmp_path):
    protocol_path, work = _prepare_fixture(tmp_path)
    with pytest.raises(ValueError, match="Attempt reconciliation failed"):
        runner.summarize(
            argparse.Namespace(
                protocol=str(protocol_path),
                work_dir=str(work),
                partition="development",
                output=str(tmp_path / "summary.json"),
            )
        )


def test_confirmation_lock_is_predeclared_and_immutable(tmp_path):
    protocol_path, work = _prepare_fixture(tmp_path)
    namespace = argparse.Namespace(
        protocol=str(protocol_path),
        work_dir=str(work),
        proposed_default="am1bcc/hct/ace",
        pass_rule="confirmation MAE <= 2.0 kcal/mol and failure_rate == 0",
    )
    runner.freeze_confirmation(namespace)
    lock = core.load_json(work / "confirmation-lock.json")
    assert lock["proposed_default"] == "am1bcc/hct/ace"
    assert lock["failed_confirmation_must_not_trigger_tuning"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        runner.freeze_confirmation(namespace)


def test_metric_fixture_matches_hand_computation():
    metrics = core.summarize_errors(
        [-1.0, 2.0], expected_count=4, resamples=100, confidence=0.95, seed=7
    )
    assert metrics["n"] == 2
    assert metrics["mse"] == pytest.approx(0.5)
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx((2.5) ** 0.5)
    assert metrics["max_absolute_error"] == pytest.approx(2.0)
    assert metrics["failure_rate"] == pytest.approx(0.5)


def _write_passing_parity_artifacts(protocol_path: Path, artifact_dir: Path) -> Path:
    protocol, fingerprint = core.load_protocol(protocol_path)
    amber_dir = artifact_dir / "amber-gb-parity"
    apbs_dir = artifact_dir / "apbs-grid"
    amber_dir.mkdir(parents=True)
    apbs_dir.mkdir(parents=True)
    core.write_json_atomic(
        amber_dir / "results.json",
        {
            "schema_version": 1,
            "artifact_type": "amber-gb-parity",
            "protocol_fingerprint": fingerprint,
            "records": [
                {
                    "case_id": "fixture-neutral",
                    "model": "hct",
                    "status": "success",
                    "signed_difference_kcal_mol": {
                        "polar": 0.001,
                        "nonpolar_lcpo": -0.002,
                        "total_lcpo": -0.001,
                    },
                    "force_difference_metrics": {
                        "polar": {"max_abs": 0.001, "rms": 0.0005},
                        "total_lcpo": {"max_abs": 0.002, "rms": 0.001},
                    },
                }
            ],
        },
    )
    core.write_json_atomic(
        apbs_dir / "results.json",
        {
            "schema_version": 1,
            "artifact_type": "apbs-grid-parity",
            "protocol_fingerprint": fingerprint,
            "records": [
                {
                    "case_id": "official-ion",
                    "control_kind": "official-born-ion",
                    "status": "success",
                    "grid_points": 97,
                    "grid_spacing_angstrom": 0.33,
                    "maple_kcal_mol": {"polar": -10.0, "nonpolar": 0.0, "total": -10.0},
                    "signed_difference_kcal_mol": {"polar": 0.001},
                },
                {
                    "case_id": "fixture-neutral",
                    "control_kind": "neutral-grid-convergence",
                    "status": "success",
                    "grid_points": 65,
                    "grid_spacing_angstrom": 0.5,
                    "maple_kcal_mol": {"polar": -1.0, "nonpolar": 0.2, "total": -0.8},
                    "signed_difference_kcal_mol": {},
                },
                {
                    "case_id": "fixture-neutral",
                    "control_kind": "neutral-grid-convergence",
                    "status": "success",
                    "grid_points": 97,
                    "grid_spacing_angstrom": 0.33,
                    "maple_kcal_mol": {"polar": -1.05, "nonpolar": 0.2, "total": -0.85},
                    "signed_difference_kcal_mol": {},
                },
            ],
        },
    )
    tolerance_path = artifact_dir / "tolerances.json"
    core.write_json_atomic(
        tolerance_path,
        {
            "schema_version": 1,
            "protocol_fingerprint": fingerprint,
            "review_status": "human-reviewed-frozen",
            "amber_gb": {
                "max_abs_polar_kcal_mol": 0.01,
                "max_abs_nonpolar_lcpo_kcal_mol": 0.01,
                "max_abs_total_lcpo_kcal_mol": 0.01,
                "max_abs_polar_force_kcal_mol_angstrom": 0.01,
                "max_abs_total_lcpo_force_kcal_mol_angstrom": 0.01,
            },
            "apbs": {
                "max_abs_official_component_kcal_mol": 0.01,
                "max_successive_grid_total_difference_kcal_mol": 0.1,
            },
        },
    )
    return tolerance_path


def test_provider_parity_verification_fails_closed_without_reviewed_tolerances(tmp_path):
    protocol_path, _source = _write_fixture_protocol(tmp_path)
    artifact_dir = tmp_path / "parity"
    _write_passing_parity_artifacts(protocol_path, artifact_dir)

    with pytest.raises(FileNotFoundError, match="Do not guess"):
        parity.verify(
            argparse.Namespace(
                protocol=str(protocol_path),
                artifact_dir=str(artifact_dir),
                tolerances=str(tmp_path / "absent.json"),
            )
        )


def test_provider_parity_verification_accepts_complete_in_tolerance_fixture(tmp_path):
    protocol_path, _source = _write_fixture_protocol(tmp_path)
    artifact_dir = tmp_path / "parity"
    tolerances = _write_passing_parity_artifacts(protocol_path, artifact_dir)

    parity.verify(
        argparse.Namespace(
            protocol=str(protocol_path),
            artifact_dir=str(artifact_dir),
            tolerances=str(tolerances),
        )
    )
    verification = core.load_json(artifact_dir / "provider-parity-verification.json")
    assert verification["passed"] is True
    assert verification["checks"]
    assert all(check["passed"] for check in verification["checks"])
