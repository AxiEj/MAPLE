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
import run_route2_freesolv as route2_runner


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
    protocol["methods"]["density_models"] = ["official-mace-polar-1-m"]
    protocol["methods"]["solvation_models"] = ["smd-iefpcm-water"]
    protocol["statistics"]["bootstrap_resamples"] = 100
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return protocol_path, source




def _prepare_route2_fixture(tmp_path: Path) -> tuple[Path, Path]:
    protocol_path, source = _write_fixture_protocol(
        tmp_path, "route2-protocol.json"
    )
    work = tmp_path / "work"
    route2_runner.prepare(
        argparse.Namespace(
            protocol=str(protocol_path), work_dir=str(work), source_dir=str(source)
        )
    )
    return protocol_path, work




def test_route2_protocol_locks_model_continuum_standard_state_and_gates():
    protocol, fingerprint = core.load_protocol(
        BENCHMARK_DIR / "route2-protocol.json"
    )

    assert len(fingerprint) == 64
    assert protocol["benchmark_kind"] == "route2-macepolar-smd"
    assert protocol["methods"]["density_models"] == ["official-mace-polar-1-m"]
    assert protocol["methods"]["solvation_models"] == ["smd-iefpcm-water"]
    assert protocol["methods"]["mace_default_dtype"] == "float64"
    assert protocol["methods"]["mace_internal_density_smearing_angstrom"] == 1.5
    assert protocol["methods"]["pcmsolver_mep_projection"] == (
        "cavity-exterior-point-monopoles-and-dipoles-no-gto-smearing-"
        "across-dielectric-boundary"
    )
    assert protocol["methods"]["response"] == "scf"
    assert protocol["methods"]["standard_state"] == "1M-gas-to-1M-solution"
    assert protocol["providers"]["pcmsolver"]["bundled"] is False
    assert "MAE <= 1.5 kcal/mol" in protocol["confirmation"][
        "predeclared_accuracy_gate"
    ]
    assert "median(total_route2_wall_time/gas_mace_wall_time) <= 2.0" == protocol[
        "confirmation"
    ]["predeclared_runtime_gate"]


def test_route2_benchmark_parses_the_exact_public_input_contract():
    params = route2_runner._public_route2_contract()

    assert route2_runner.PUBLIC_ROUTE2_SETTINGS == (
        "#model=macepol-m",
        "#sp",
        "#solv(implicit=water,method=smd,response=scf,"
        "standard_state=1m,experimental=true)",
    )
    assert params == {
        "model": "macepol-m",
        "solv": {
            "implicit": "water",
            "method": "smd",
            "response": "scf",
            "standard_state": "1m",
            "experimental": True,
            "provider": "pcmsolver",
            "profile": "smd-iefpcm",
        },
        "task": "sp",
    }


def test_route2_run_uses_setcalculator_and_integrated_finalizer(
    monkeypatch, tmp_path
):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    calls = {"set_calculator": [], "corrections": [], "calculations": 0}

    class FakeCorrection:
        def __init__(
            self,
            atoms,
            charge_options,
            solvation_options,
            *,
            output,
        ):
            self.audit_dir = Path(str(Path(output).resolve()) + ".implicit")
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            calls["corrections"].append(
                {
                    "symbols": atoms.get_chemical_symbols(),
                    "charge_options": dict(charge_options),
                    "solvation_options": dict(solvation_options),
                    "output": str(Path(output).resolve()),
                }
            )

    class FakeIntegratedCalculator(Calculator):
        implemented_properties = ["energy", "free_energy"]
        dtype = "torch.float64"

        def __init__(self, correction):
            super().__init__()
            self.solvent_correction = correction

        def calculate(
            self,
            atoms=None,
            properties=("energy",),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            calls["calculations"] += 1
            gas = -1.234
            if self.solvent_correction is None:
                self.results = {"energy": gas, "free_energy": gas}
                return
            delta = -0.002
            combined = gas + delta
            self.results = {
                "energy": combined,
                "free_energy": combined,
                "solvation": {
                    "energy_hartree": delta,
                    "delta_g_solv_hartree": delta,
                    "gas_energy_hartree": gas,
                    "combined_energy_hartree": combined,
                    "components_hartree": {
                        "electrostatic": -0.003,
                        "cds": 0.001,
                        "delta_g_solv": delta,
                    },
                    "provenance": {"integration_test": True},
                    "ase_free_energy_is_thermochemical_gibbs": False,
                },
            }

    class FakeSetCalculator:
        def __init__(self, device, model, output, **kwargs):
            calls["set_calculator"].append(
                {
                    "device": device,
                    "model": model,
                    "output": str(Path(output).resolve()),
                    **kwargs,
                }
            )
            self.output = output
            self.kwargs = kwargs

        def set_calculator(self):
            correction = FakeCorrection(
                self.kwargs["atoms"],
                self.kwargs["charge_options"],
                self.kwargs["solvation_options"],
                output=self.output,
            )
            return FakeIntegratedCalculator(correction)

    monkeypatch.setattr(route2_runner, "SetCalculator", FakeSetCalculator)
    monkeypatch.setattr(
        route2_runner, "ImplicitSolvationCorrection", FakeCorrection
    )
    monkeypatch.setattr(
        route2_runner,
        "_environment_record",
        lambda protocol, device, calculator: {"integration_test": True},
    )

    route2_runner.run(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            device="cpu",
            max_compounds=None,
        )
    )

    record = core.load_json(
        work / "records" / "development" / "mobley_test.json"
    )
    initialization = calls["set_calculator"][0]
    assert initialization["model"] == "macepol-m"
    assert initialization["implicit"] == "smd"
    assert initialization["solvent"] == "water"
    assert initialization["charge_options"] == {}
    assert initialization["solvation_options"] == (
        route2_runner._public_route2_contract()["solv"]
    )
    assert calls["calculations"] == 2
    assert record["status"] == "success"
    assert record["gas_energy_hartree"] == pytest.approx(-1.234)
    assert record["combined_energy_hartree"] == pytest.approx(-1.236)
    assert record["predicted_kcal_mol"] == pytest.approx(
        -0.002 * route2_runner.KCAL_PER_HARTREE
    )
    assert record["provider_provenance"] == {"integration_test": True}
    assert record["public_input"] == list(route2_runner.PUBLIC_ROUTE2_SETTINGS)
    assert record["ase_free_energy_is_thermochemical_gibbs"] is False
    assert "route2_integrated" in record["timing_seconds"]
    assert "route2_correction" not in record["timing_seconds"]
    assert record["audit_directory"].endswith("maple.out.implicit")


def test_route2_summary_is_complete_reproducible_and_not_certified_on_development(
    tmp_path,
):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    protocol, fingerprint = core.load_protocol(protocol_path)
    prepared = core.load_json(work / "prepared.json")
    candidate = prepared["candidates"][0]
    record_dir = work / "records" / "development"
    record_dir.mkdir(parents=True)
    core.write_json_atomic(
        record_dir / "mobley_test.json",
        {
            "schema_version": 1,
            "attempt_id": "mobley_test",
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": fingerprint,
            "partition": "development",
            "compound_id": "mobley_test",
            "status": "success",
            "signed_error_kcal_mol": 0.5,
            "timing_seconds": {
                "gas_mace": 1.0,
                "route2_integrated": 1.5,
                "route2_total": 1.5,
                "total_over_gas": 1.5,
            },
        },
    )
    output = tmp_path / "route2-summary.json"

    route2_runner.summarize(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            output=str(output),
        )
    )
    summary = core.load_json(output)

    assert candidate["compound_id"] == "mobley_test"
    assert summary["overall"]["mae"] == pytest.approx(0.5)
    assert summary["timing"]["median_total_over_gas"] == pytest.approx(1.5)
    assert summary["predeclared_gates"]["accuracy"]["passed_on_this_partition"] is True
    assert summary["predeclared_gates"]["runtime"]["passed_on_this_partition"] is True
    assert summary["predeclared_gates"]["scientifically_certified"] is False


def test_route2_candidate_shards_are_disjoint_and_cover_the_partition():
    candidates = [{"compound_id": f"compound-{index}"} for index in range(11)]

    shards = [
        route2_runner._select_shard(
            candidates,
            shard_count=4,
            shard_index=index,
        )
        for index in range(4)
    ]
    flattened = [candidate["compound_id"] for shard in shards for candidate in shard]

    assert len(flattened) == len(set(flattened)) == len(candidates)
    assert sorted(flattened) == sorted(candidate["compound_id"] for candidate in candidates)


@pytest.mark.parametrize(
    ("shard_count", "shard_index", "message"),
    [
        (0, 0, "--shard-count must be positive"),
        (2, -1, "--shard-index"),
        (2, 2, "--shard-index"),
    ],
)
def test_route2_candidate_shards_reject_invalid_coordinates(
    shard_count, shard_index, message
):
    with pytest.raises(ValueError, match=message):
        route2_runner._select_shard(
            [],
            shard_count=shard_count,
            shard_index=shard_index,
        )


def test_route2_supervisor_records_audited_pcmsolver_process_exit(
    tmp_path, monkeypatch
):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    audit_dir = (
        work
        / "provider-audit"
        / "mobley_test"
        / "maple.out.implicit"
    )

    def fake_run(command, *, stdout, stderr, check):
        assert command[2] == "run"
        assert stderr is route2_runner.subprocess.STDOUT
        assert check is False
        audit_dir.mkdir(parents=True)
        stdout.write(
            b"PCMSolver fatal error.\n"
            b"Gauss theorem check failed for the frozen cavity.\n"
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(route2_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        route2_runner,
        "_runtime_environment_record",
        lambda protocol, device, mace_dtype: {
            "device": device,
            "mace_dtype": mace_dtype,
        },
    )

    route2_runner.run_supervised(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            device="cuda",
            max_compounds=None,
            shard_count=1,
            shard_index=0,
        )
    )

    record = core.load_json(
        work / "records" / "development" / "mobley_test.json"
    )
    assert record["status"] == "failure"
    assert record["failure"]["phase"] == "pcmsolver-process"
    assert (
        record["failure"]["exception_class"]
        == "PCMSolverFatalProcessExit"
    )
    assert "Gauss theorem check failed" in record["failure"]["reason"]
    assert record["environment"] == {
        "device": "cuda",
        "mace_dtype": "torch.float64",
    }


def test_route2_supervisor_does_not_misclassify_an_unidentified_worker_exit(
    tmp_path, monkeypatch
):
    protocol_path, work = _prepare_route2_fixture(tmp_path)

    def fake_run(command, *, stdout, stderr, check):
        stdout.write(b"unexpected model initialization error\n")
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(route2_runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="without a PCMSolver fatal marker"):
        route2_runner.run_supervised(
            argparse.Namespace(
                protocol=str(protocol_path),
                work_dir=str(work),
                partition="development",
                device="cuda",
                max_compounds=None,
                shard_count=1,
                shard_index=0,
            )
        )

    assert not (
        work / "records" / "development" / "mobley_test.json"
    ).exists()


def test_route2_run_rejects_protocol_and_parser_mismatch(tmp_path, monkeypatch):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    mismatch_params = {
        "task": "sp",
        "model": "macepol-m",
        "solv": {
            "implicit": "water",
            "method": "smd",
            "response": "frozen",
            "standard_state": "1m",
            "provider": "pcmsolver",
            "profile": "smd-iefpcm",
            "experimental": True,
        },
    }

    monkeypatch.setattr(
        route2_runner, "_public_route2_contract", lambda: mismatch_params
    )

    with pytest.raises(
        ValueError,
        match="public Route-2 parser contract and benchmark protocol disagree",
    ):
        route2_runner.run(
            argparse.Namespace(
                protocol=str(protocol_path),
                work_dir=str(work),
                partition="development",
                device="cpu",
                max_compounds=None,
            )
        )


def test_route2_run_rejects_zero_max_compounds(tmp_path):
    protocol_path, work = _prepare_route2_fixture(tmp_path)

    with pytest.raises(ValueError, match="--max-compounds must be positive"):
        route2_runner.run(
            argparse.Namespace(
                protocol=str(protocol_path),
                work_dir=str(work),
                partition="development",
                device="cpu",
                max_compounds=0,
            )
        )


def test_route2_confirmation_run_requires_frozen_lock(tmp_path):
    protocol_path, work = _prepare_route2_fixture(tmp_path)

    with pytest.raises(ValueError, match="Confirmation is locked"):
        route2_runner.run(
            argparse.Namespace(
                protocol=str(protocol_path),
                work_dir=str(work),
                partition="confirmation",
                device="cpu",
                max_compounds=None,
            )
        )


def test_route2_run_skips_existing_records_without_recalculating(tmp_path, monkeypatch):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    prepared = core.load_json(work / "prepared.json")
    candidate_id = prepared["candidates"][0]["compound_id"]
    record_dir = work / "records" / "development"
    record_dir.mkdir(parents=True)
    existing = {
        "schema_version": 1,
        "attempt_id": candidate_id,
        "protocol_fingerprint": core.load_protocol(protocol_path)[1],
        "partition": "development",
        "status": "success",
    }
    core.write_json_atomic(record_dir / f"{candidate_id}.json", existing)

    class ProhibitedCalculator:
        def __init__(self, *args, **kwargs):
            raise AssertionError("SetCalculator must not be called when record exists")

    monkeypatch.setattr(route2_runner, "SetCalculator", ProhibitedCalculator)

    route2_runner.run(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="development",
            device="cpu",
            max_compounds=None,
        )
    )


def test_route2_run_rejects_mol2_checksum_mismatch_before_solver(tmp_path, monkeypatch):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    prepared = core.load_json(work / "prepared.json")
    candidate = prepared["candidates"][0]
    mol2_path = work / candidate["mol2_relative_path"]
    mol2_path.write_text("MUTATED", encoding="utf-8")

    class ProhibitedCalculator:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Computation should fail before calculator setup")

    monkeypatch.setattr(route2_runner, "SetCalculator", ProhibitedCalculator)

    with pytest.raises(ValueError, match="MOL2 changed after preparation"):
        route2_runner.run(
            argparse.Namespace(
                protocol=str(protocol_path),
                work_dir=str(work),
                partition="development",
                device="cpu",
                max_compounds=None,
            )
        )


def test_route2_summary_rejects_missing_records(tmp_path):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    output = tmp_path / "route2-summary.json"

    with pytest.raises(ValueError, match="Attempt reconciliation failed"):
        route2_runner.summarize(
            argparse.Namespace(
                protocol=str(protocol_path),
                work_dir=str(work),
                partition="development",
                output=str(output),
            )
        )


def test_route2_summary_requires_confirmation_lock_when_partition_is_confirmation(
    tmp_path,
):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    protocol, fingerprint = core.load_protocol(protocol_path)
    prepared = core.load_json(work / "prepared.json")

    candidates = prepared["candidates"]
    for candidate in candidates:
        candidate["partition"] = "confirmation"
    manifest = prepared | {
        "partition_counts": {
            "development": 0,
            "confirmation": len(candidates),
        },
        "candidate_count": len(candidates),
    }
    core.write_json_atomic(work / "prepared.json", manifest)

    output = tmp_path / "route2-summary.json"
    with pytest.raises(ValueError, match="Confirmation is locked"):
        route2_runner.summarize(
            argparse.Namespace(
                protocol=str(protocol_path),
                work_dir=str(work),
                partition="confirmation",
                output=str(output),
            )
        )


def test_route2_summary_marks_scientifically_certified_when_gates_pass_in_confirmation(
    tmp_path,
):
    protocol_path, work = _prepare_route2_fixture(tmp_path)
    protocol, fingerprint = core.load_protocol(protocol_path)
    prepared = core.load_json(work / "prepared.json")
    candidate_id = prepared["candidates"][0]["compound_id"]

    for candidate in prepared["candidates"]:
        candidate["partition"] = "confirmation"
    prepared["partition_counts"] = {
        "development": 0,
        "confirmation": len(prepared["candidates"]),
    }
    core.write_json_atomic(work / "prepared.json", prepared)

    protocol_lock = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "proposed_default": "standard",
        "pass_rule": "mae <= 1.5",
        "frozen_at_utc": "2026-07-23T00:00:00+00:00",
        "one_shot": True,
        "failed_confirmation_must_not_trigger_tuning": True,
    }
    core.write_json_atomic(
        work / protocol["confirmation"]["lock_filename"], protocol_lock
    )

    record_dir = work / "records" / "confirmation"
    record_dir.mkdir(parents=True)
    core.write_json_atomic(
        record_dir / f"{candidate_id}.json",
        {
            "schema_version": 1,
            "attempt_id": candidate_id,
            "protocol_fingerprint": fingerprint,
            "partition": "confirmation",
            "compound_id": candidate_id,
            "status": "success",
            "signed_error_kcal_mol": 1.0,
            "timing_seconds": {
                "total_over_gas": 1.2,
                "gas_mace": 0.5,
                "route2_total": 0.6,
            },
        },
    )

    output = tmp_path / "route2-summary.json"
    route2_runner.summarize(
        argparse.Namespace(
            protocol=str(protocol_path),
            work_dir=str(work),
            partition="confirmation",
            output=str(output),
        )
    )

    summary = core.load_json(output)
    assert summary["predeclared_gates"]["scientifically_certified"] is True
