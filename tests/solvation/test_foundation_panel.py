"""Acceptance guards do not require ANI checkpoints or a native PB solver."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

BENCHMARKS = Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import check_route1_foundation_evidence as evidence  # pyright: ignore[reportMissingImports]
import run_route1_foundation_panel as panel  # pyright: ignore[reportMissingImports]
from check_route1_foundation_evidence import (  # pyright: ignore[reportMissingImports]
    cross_case_identity,
)
from run_route1_foundation_panel import (  # pyright: ignore[reportMissingImports]
    ENDPOINTS,
    PANEL,
    differentiate,
    hessian_quality,
    json_safe,
    run_case,
    summarize,
)


class Quadratic(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]  # noqa: RUF012 -- ASE calculator contract
    solvent_correction: SimpleNamespace
    inference_precision_provenance: dict[str, str]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        x = self.atoms.positions.reshape(-1)
        h = np.diag(np.arange(1, len(x) + 1))
        self.results = {
            "energy": x @ h @ x / 2,
            "free_energy": x @ h @ x / 2,
            "forces": (-h @ x).reshape((-1, 3)),
        }


def test_complete_energy_force_hessian_differences_restore_geometry():
    atoms = Atoms("OH2", positions=np.arange(9).reshape((3, 3)) / 10)
    atoms.calc = Quadratic()
    before = atoms.positions.copy()
    forces = differentiate(atoms, 1e-4, hessian=False)
    hessian = differentiate(atoms, 5e-4, hessian=True)
    assert forces == pytest.approx(atoms.get_forces(), abs=1e-10)
    assert hessian == pytest.approx(np.diag(np.arange(1, 10)), abs=1e-10)
    assert np.array_equal(atoms.positions, before)


def test_failure_restores_original_geometry(monkeypatch):
    atoms = Atoms("H", positions=[[1, 2, 3]])
    before = atoms.positions.copy()

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    monkeypatch.setattr(atoms, "get_forces", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        differentiate(atoms, 1e-4, hessian=True)
    assert np.array_equal(atoms.positions, before)


def test_raw_hessian_asymmetry_cannot_be_hidden_by_symmetrization():
    h = np.eye(3)
    noisy = h.copy()
    noisy[0, 1] += 0.1
    noisy[1, 0] -= 0.1
    result = hessian_quality([h, noisy])
    assert result["passed"] is False
    assert result["symmetrized_maximum_step_change_diagnostic_only"] == 0
    assert result["raw_maximum_step_change"] == pytest.approx(0.1)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_hessian_quality_rejects_nonfinite(value):
    h = np.eye(3)
    h[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        hessian_quality([h, np.eye(3)])


def test_full_denominator_cannot_be_replaced_by_successful_subset():
    subset = [{"endpoint": "obc2", "molecule": m, "accepted": True} for m in PANEL]
    result = summarize(subset)
    assert result["selected_cases_all_accepted"] is True
    assert result["all_accepted"] is False
    assert result["expected_count"] == 12
    assert len(result["missing"]) == 6
    full = [
        {"endpoint": e, "molecule": m, "accepted": True}
        for e in ENDPOINTS
        for m in PANEL
    ]
    assert summarize(full)["all_accepted"] is True
    full[-1]["accepted"] = False
    assert summarize(full)["all_accepted"] is False
    full[-1] = full[0].copy()
    assert summarize(full)["all_accepted"] is False


def test_missing_input_is_preserved_as_failed_case(tmp_path):
    source = tmp_path / "ammonia" / "fixed.mol2"
    folder = tmp_path / "attempt"
    record = run_case(source, "obc2", folder)
    assert record["status"] == "failed"
    assert record["accepted"] is False
    assert "FileNotFoundError" in record["error"]
    assert (folder / "result.json").is_file()
    with pytest.raises(FileExistsError):
        run_case(source, "obc2", folder)


def test_json_nonfinite_values_remain_null_not_nonstandard_nan():
    assert json_safe({"raw": np.array([np.nan, np.inf, 1.0])}) == {
        "raw": [None, None, 1.0]
    }


def test_runner_attaches_constructed_calculator_before_native_checks(
    tmp_path, monkeypatch
):
    source = tmp_path / "benzene" / "fixed.mol2"
    source.parent.mkdir()
    source.write_text("mock input")
    atoms = Atoms("H", positions=[[0.1, 0.2, 0.3]])
    calc = Quadratic()
    calc.solvent_correction = SimpleNamespace(
        provider=SimpleNamespace(
            charges=np.zeros(1),
            radius_result=SimpleNamespace(radii_angstrom=np.ones(1)),
            provenance={"test": True},
        )
    )
    calc.inference_precision_provenance = {"effective_dtype": "float64"}
    monkeypatch.setattr(panel, "MOL2Reader", lambda *args, **kwargs: atoms)
    monkeypatch.setattr(
        panel,
        "SetCalculator",
        lambda *args, **kwargs: SimpleNamespace(set_calculator=lambda: calc),
    )
    record = run_case(source, "obc2", tmp_path / "attempt")
    assert record["accepted"] is True, record
    assert atoms.calc is calc


def _paired_identity_records():
    records = []
    for endpoint in ENDPOINTS:
        for name in PANEL:
            fixed = {
                "charges_e": [0.0],
                "radii_angstrom": [1.0],
                "charges_sha256": panel.array_hash([0.0]),
                "radii_sha256": panel.array_hash([1.0]),
            }
            provider = dict(ENDPOINTS[endpoint])
            if endpoint == "ddlpb":
                provider.update(
                    polar_only=True,
                    reference_only=True,
                    fixed_charge=True,
                    fixed_radius=True,
                    absolute_solvation_free_energy_claim=False,
                    radii="mbondi2",
                    required_provider_version="0.8.0",
                    settings={
                        "lmax": 9,
                        "n_lebedev": 302,
                        "eta": 0.1,
                        "shift": 0.0,
                        "solver_tolerance": 1e-10,
                        "n_proc": 1,
                    },
                )
            records.append(
                {
                    "endpoint": endpoint,
                    "molecule": name,
                    "accepted": True,
                    "input_sha256": "b" * 64,
                    "inference_precision": {
                        "effective_dtype": "float64",
                        "requested_dtype": "float64",
                        "numerical_curvature_prepared": True,
                        "original_checkpoint_sha256": "a" * 64,
                    },
                    "identity_before": deepcopy(fixed),
                    "identity_after": deepcopy(fixed),
                    "provider_provenance": provider,
                }
            )
    return records


def test_cross_case_identity_requires_one_checkpoint_and_paired_fixed_inputs():
    records = _paired_identity_records()
    assert cross_case_identity(records)["passed"] is True
    for field in ("checkpoint", "input", "charge", "radius", "provider"):
        changed = deepcopy(records)
        record = changed[-1]
        if field == "checkpoint":
            record["inference_precision"]["original_checkpoint_sha256"] = "c" * 64
        elif field == "input":
            record["input_sha256"] = "c" * 64
        elif field in ("charge", "radius"):
            vector, digest = (
                ("charges_e", "charges_sha256")
                if field == "charge"
                else ("radii_angstrom", "radii_sha256")
            )
            for identity in (record["identity_before"], record["identity_after"]):
                identity[vector] = [2.0]
                identity[digest] = panel.array_hash([2.0])
        else:
            record["provider_provenance"]["solvent_kappa_inverse_angstrom"] = 0.0
        assert cross_case_identity(changed)["passed"] is False, field


@pytest.fixture
def complete_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    run, inputs = tmp_path / "run", tmp_path / "prepared"
    checkpoint = tmp_path / "maple/function/calculator/model/ani2x.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("test-only checkpoint")
    records = _paired_identity_records()
    prepared, reconstructed = [], []
    for name in PANEL:
        folder = inputs / name
        folder.mkdir(parents=True)
        original = tmp_path / "inputs" / name
        original.mkdir(parents=True)
        for path in (
            folder / "fixed.mol2",
            folder / "reference.mol2",
            folder / "preparation.json",
            original / "reference.mol2",
            original / "preparation.json",
        ):
            path.write_text(name)
        digest = panel.sha256_file(folder / "fixed.mol2")
        prepared.append({"name": name, "status": "prepared", "fixed_sha256": digest})
        reconstructed.append(
            {
                "name": name,
                "seed": 20260913,
                "rdkit_version": "2024.09.2",
                "embedding": "ETKDGv3",
                "embedding_return_code_reconstructed": 0,
                "uff_max_iterations": 200,
                "uff_return_code_reconstructed": 0,
                "quantization_angstrom": 0.001,
                "original_reference_byte_exact_reconstruction": True,
                "rounded_reference_byte_exact_reconstruction": True,
                "original_reference_sha256": digest,
                "failed_original_preparation_sha256": digest,
                "rounded_reference_sha256": digest,
                "preparation_record_sha256": digest,
                "fixed_mol2_sha256": digest,
            }
        )
        for record in records:
            if record["molecule"] != name:
                continue
            record["input_sha256"] = digest
            record["inference_precision"]["original_checkpoint_sha256"] = (
                panel.sha256_file(checkpoint)
            )
            case = run / f"{record['endpoint']}-{name}"
            case.mkdir(parents=True)
            raw = [np.eye(3), np.eye(3)]
            record["hessian_quality"] = hessian_quality(raw)
            for label in ("panel", "minimum") if name in panel.MINIMA else ("panel",):
                for step, matrix in zip(panel.PROTOCOL["hessian_steps_angstrom"], raw):
                    np.save(case / f"{label}-raw-hessian-{step:g}.npy", matrix)
            if name in panel.MINIMA:
                record["minimum"] = {"hessian_quality": hessian_quality(raw)}
            panel.save(case / "result.json", record)
    panel.save(inputs / "manifest.json", {"records": prepared})
    generators = {}
    for name in (
        "prepare_inputs.py",
        "prepare_rounded_inputs.py",
        "reconstruct_input_lineage.py",
    ):
        path = tmp_path / name
        path.write_text("# test generator\n")
        generators[name] = panel.sha256_file(path)
    lineage, environment = tmp_path / "lineage.json", tmp_path / "environment.json"
    panel.save(
        lineage,
        {
            "records": reconstructed,
            "generator_sha256": generators,
            "prepared_manifest_sha256": panel.sha256_file(inputs / "manifest.json"),
        },
    )
    panel.save(
        run / "protocol.json",
        {
            "input_manifest_sha256": panel.sha256_file(inputs / "manifest.json"),
            "source_sha256": {},
        },
    )
    summary = summarize(records)
    summary["source_unchanged_during_run"] = True
    panel.save(run / "summary.json", summary)
    panel.save(
        environment,
        [
            {
                "command": [str(run)],
                "cwd": str(tmp_path),
                "environment": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OPENMM_CPU_THREADS": "1",
                    "PYTHONPATH": str(tmp_path),
                },
            }
        ],
    )
    return run, inputs, lineage, environment


def test_complete_postflight_binds_lineage_environment_sources_and_raw_artifacts(
    complete_evidence,
):
    record = evidence.check_evidence(*complete_evidence)
    assert record["evidence_integrity_passed"] is True, record["errors"]
    assert record["all_numerical_cases_accepted"] is True
    assert len(record["native_case_artifacts_sha256"]) == 12


@pytest.mark.parametrize(
    "corruption",
    ["cwd", "pythonpath", "native_source", "checkpoint", "lineage", "case"],
)
def test_postflight_cannot_accept_corrupt_provenance(complete_evidence, corruption):
    run, _, lineage, environment = complete_evidence
    if corruption in {"cwd", "pythonpath"}:
        payload = json.loads(environment.read_text())
        if corruption == "cwd":
            payload[0]["cwd"] = "/another-checkout"
        else:
            payload[0]["environment"]["PYTHONPATH"] = "/another-checkout"
        panel.save(environment, payload)
    elif corruption == "native_source":
        payload = json.loads((run / "summary.json").read_text())
        payload["source_unchanged_during_run"] = False
        panel.save(run / "summary.json", payload)
    elif corruption == "checkpoint":
        (run.parent / "maple/function/calculator/model/ani2x.pt").write_text("replaced")
    elif corruption == "lineage":
        payload = json.loads(lineage.read_text())
        payload["records"][0]["seed"] = 42
        panel.save(lineage, payload)
    else:
        panel.save(run / "obc2-water/result.json", {})
    record = evidence.check_evidence(*complete_evidence)
    assert record["evidence_integrity_passed"] is False
    assert record["all_numerical_cases_accepted"] is False
