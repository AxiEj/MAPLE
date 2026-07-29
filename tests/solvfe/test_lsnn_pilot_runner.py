from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("openmmtorch")
pytest.importorskip("pymbar")
pytest.importorskip("torch")


def _load_runner_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "docs/implicit-solvation/benchmarks/lsnn_pilot/run_lsnn_pilot.py"
    runner_dir = str(path.parent)
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    spec = importlib.util.spec_from_file_location("lsnn_pilot_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_compound_entry() -> dict[str, str]:
    return {
        "compound_id": "mobley_000001",
        "name": "methane-like",
        "smiles": "C",
    }


def _write_panel(
    path: Path,
    *,
    compounds: list[dict[str, str]] | None = None,
    panel_payload: dict | None = None,
) -> None:
    if panel_payload is not None:
        panel = panel_payload
    else:
        panel = {"molecules": compounds or [_default_compound_entry()]}
    path.write_text(json.dumps(panel, indent=2), encoding="utf-8")


def _stub_labels(_database: Path, compounds: list[dict]):
    return {
        compound["compound_id"]: {
            "experimental_kcal_mol": -2.0,
            "experimental_uncertainty_kcal_mol": 0.5,
            "experimental_reference": "FreeSolv",
        }
        for compound in compounds
    }


def _stub_record(
    *,
    compound,
    seeds: tuple[int, ...],
    status: str | None = None,
    prediction: float,
    replicate_sd: float,
    replicate_sem: float,
    combined_uncertainty: float,
    quality_failures: list[str] | tuple[str, ...],
    quality_notes: list[str] | tuple[str, ...],
):
    if status is None:
        status = "unconverged" if quality_failures else "ok"
    return {
        "status": status,
        "quality_failures": list(quality_failures),
        "quality_notes": list(quality_notes),
        "compound_id": compound["compound_id"],
        "name": compound["name"],
        "smiles": compound["smiles"],
        "prediction_kcal_mol": prediction,
        "replicate_sd_kcal_mol": replicate_sd,
        "replicate_count": len(seeds),
        "replicate_sem_kcal_mol": replicate_sem,
        "combined_uncertainty_kcal_mol": combined_uncertainty,
        "seed_results": [
            {
                "seed": seed,
                "delta_g_kcal_mol": prediction,
                "mbar_uncertainty_kcal_mol": 0.0,
                "quality_failures": list(quality_failures),
                "quality_notes": list(quality_notes),
            }
            for seed in seeds
        ],
        "build": [],
    }


def _run_main(
    runner,
    work_dir: Path,
    tmp_path: Path,
    monkeypatch,
    *,
    seeds: str,
    max_compounds: int | None = None,
    compound_id: str | None = None,
    sensitivity_only: bool = False,
    run_compound_stub=None,
    panel_compounds: list[dict[str, str]] | None = None,
    lambda_values: str = "0,0.5,1",
    temperature: float | str = 300.0,
    timestep_fs: float = 1.0,
    equilibration_steps: int = 200,
    production_steps: int = 1000,
    sample_interval: int = 5,
    panel_path: Path | None = None,
    panel_payload: dict | None = None,
    validate_sources_stub=None,
):
    stub_root = tmp_path

    if validate_sources_stub is None:

        def _stub_validate_sources(_upstream_repo: Path, _freesolv_repo: Path):
            # keep this lightweight: _run_compound is monkeypatched in tests that call main.
            state_dict = stub_root / "state_dict.pt"
            amber_tar = stub_root / "amber.tar.gz"
            database = stub_root / "database.txt"
            state_dict.write_text("state")
            amber_tar.write_text("tar")
            database.write_text("db")
            return state_dict, amber_tar, database

    else:
        _stub_validate_sources = validate_sources_stub

    panel_path = panel_path or (tmp_path / "panel.json")
    _write_panel(
        panel_path,
        compounds=panel_compounds,
        panel_payload=panel_payload,
    )

    monkeypatch.setattr(runner, "_validate_sources", _stub_validate_sources)
    monkeypatch.setattr(runner, "_load_freesolv_labels", _stub_labels)

    if run_compound_stub is None:

        def _run_compound(
            *,
            compound,
            state_dict,
            amber_tar,
            work_dir,
            lambda_states,
            seeds,
            temperature_kelvin,
            timestep_femtoseconds,
            equilibration_steps,
            production_steps,
            sample_interval,
            platform_name,
            sensitivity_only,
        ):
            quality_failures = []
            quality_notes: list[str] = []
            if sensitivity_only and len(seeds) < 3:
                quality_failures.append("replicate_count_below_3")
            if sensitivity_only:
                quality_notes.append("sensitivity_only_excluded_from_accuracy_metrics")
            else:
                quality_notes.append("formal_accuracy_mode")
            return _stub_record(
                compound=compound,
                seeds=seeds,
                prediction=-2.0,
                replicate_sd=0.0,
                replicate_sem=0.0,
                combined_uncertainty=0.0,
                quality_failures=quality_failures,
                quality_notes=quality_notes,
            )

        run_compound_stub = _run_compound

    monkeypatch.setattr(runner, "_run_compound", run_compound_stub)

    argv = [
        "run_lsnn_pilot.py",
        "--upstream-repo",
        str(tmp_path / "upstream"),
        "--freesolv-repo",
        str(tmp_path / "freesolv"),
        "--work-dir",
        str(work_dir),
        "--panel",
        str(panel_path),
        "--seeds",
        seeds,
        "--lambda-values",
        lambda_values,
        "--temperature-kelvin",
        str(temperature),
        "--timestep-fs",
        str(timestep_fs),
        "--equilibration-steps",
        str(equilibration_steps),
        "--production-steps",
        str(production_steps),
        "--sample-interval",
        str(sample_interval),
    ]
    if sensitivity_only:
        argv.append("--sensitivity-only")
    if max_compounds is not None:
        argv.extend(["--max-compounds", str(max_compounds)])
    if compound_id is not None:
        argv.extend(["--compound-id", compound_id])

    monkeypatch.setattr(sys, "argv", argv)
    return runner.main(), work_dir


def test_cli_rejects_empty_seed_list(monkeypatch, tmp_path):
    """Empty seed lists are rejected before launching any calculations."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-empty"
    with pytest.raises((argparse.ArgumentTypeError, SystemExit)):
        _run_main(runner, work_dir, tmp_path, monkeypatch, seeds="")


def test_cli_rejects_non_unique_seed_list(monkeypatch, tmp_path):
    """Duplicate seeds are rejected by parser validation."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-dup"
    with pytest.raises(SystemExit):
        _run_main(runner, work_dir, tmp_path, monkeypatch, seeds="20260729,20260729")


def test_formal_protocol_requires_at_least_three_unique_seeds(monkeypatch, tmp_path):
    """Formal accuracy mode requires at least three unique seeds."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-two"
    with pytest.raises(SystemExit):
        _run_main(runner, work_dir, tmp_path, monkeypatch, seeds="20260729,20260730")


def test_single_seed_requires_sensitivity_only_and_is_excluded_from_metrics(
    monkeypatch, tmp_path
):
    """Single-seed runs must use sensitivity-only mode and are excluded from MAE metrics."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-sens"

    with pytest.raises(SystemExit):
        _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729",
        )

    rc, out_dir = _run_main(
        runner,
        work_dir,
        tmp_path,
        monkeypatch,
        seeds="20260729",
        sensitivity_only=True,
    )
    assert rc == 1
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    record = summary["records"][0]

    assert summary["protocol"]["run_mode"] == "sensitivity_only"
    assert record["status"] == "unconverged"
    assert record["included_in_metrics"] is False
    assert "replicate_count_below_3" in record["metric_exclusion_reasons"]
    assert (
        "sensitivity_only_excluded_from_accuracy_metrics"
        in record["metric_exclusion_reasons"]
    )
    assert summary["metrics"]["n_success"] == 0


def test_three_seed_sensitivity_mode_is_excluded_from_accuracy_metrics(
    monkeypatch, tmp_path
):
    """Sensitivity mode with three seeds should complete successfully but not score metrics."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-sens-formal"

    rc, out_dir = _run_main(
        runner,
        work_dir,
        tmp_path,
        monkeypatch,
        seeds="20260729,20260730,20260731",
        sensitivity_only=True,
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    record = summary["records"][0]

    assert summary["protocol"]["run_mode"] == "sensitivity_only"
    assert record["status"] == "ok"
    assert record["included_in_metrics"] is False
    assert record["metric_exclusion_reasons"] == [
        "sensitivity_only_excluded_from_accuracy_metrics"
    ]
    assert summary["metrics"]["n_success"] == 0


def test_summary_provenance_and_raw_u_kn_artifacts_are_recorded(monkeypatch, tmp_path):
    """Summary records expose raw-u and source provenance for post-run reproducibility checks."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-metadata"

    def _run_compound_with_artifacts(
        *,
        compound,
        state_dict,
        amber_tar,
        work_dir,
        lambda_states,
        seeds,
        temperature_kelvin,
        timestep_femtoseconds,
        equilibration_steps,
        production_steps,
        sample_interval,
        platform_name,
        sensitivity_only,
    ):
        return {
            "status": "ok",
            "quality_failures": [],
            "quality_notes": [],
            "compound_id": compound["compound_id"],
            "name": compound["name"],
            "smiles": compound["smiles"],
            "prediction_kcal_mol": -2.0,
            "replicate_sd_kcal_mol": 0.0,
            "replicate_count": len(seeds),
            "replicate_sem_kcal_mol": 0.0,
            "combined_uncertainty_kcal_mol": 0.0,
            "seed_results": [
                {
                    "seed": seed,
                    "delta_g_kcal_mol": -2.0,
                    "mbar_uncertainty_kcal_mol": 0.0,
                    "quality_failures": [],
                    "quality_notes": [],
                    "raw_energy_artifact": {
                        "path": f"cases/{compound['compound_id']}/raw-u-kln-seed-{seed}.npz",
                        "sha256": "00" * 32,
                        "shape": [2, 3],
                        "energy_unit": "kJ/mol",
                    },
                }
                for seed in seeds
            ],
            "build": [
                {
                    "scripted_model_sha256": "11" * 32,
                    "lambda_zero_energy_kj_mol": 0.0,
                }
            ],
        }

    rc, out_dir = _run_main(
        runner,
        work_dir,
        tmp_path,
        monkeypatch,
        seeds="20260729,20260730,20260731",
        run_compound_stub=_run_compound_with_artifacts,
    )
    assert rc == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    execution = summary["execution"]
    assert protocol["run_mode"] == "formal"
    assert "runner_sha256" in execution
    assert "model_sha256" in execution
    assert "panel_sha256" in execution
    assert "command" in execution and isinstance(execution["command"], list)

    record = summary["records"][0]
    artifacts = [
        entry["raw_energy_artifact"]
        for entry in record["seed_results"]
        if "raw_energy_artifact" in entry
    ]
    assert artifacts
    for art in artifacts:
        assert "path" in art and art["path"].endswith(".npz")
        assert "sha256" in art and len(art["sha256"]) == 64
        assert art["energy_unit"] == "kJ/mol"
        assert isinstance(art["shape"], list)
    assert record["build"]
    assert all(item["scripted_model_sha256"] == "11" * 32 for item in record["build"])


def test_external_state_artifacts_revalidated_before_completion(monkeypatch, tmp_path):
    """Tampering external artifacts after launch should fail the run contract."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-tamper"
    state = {"initialized": False}
    database_path = tmp_path / "database.txt"

    state_dict = tmp_path / "state_dict.pt"
    amber_tar = tmp_path / "amber.tar.gz"

    def _validate_sources_stub(_upstream_repo: Path, _freesolv_repo: Path):
        if not state.get("initialized"):
            state_dict.write_text("state")
            amber_tar.write_text("tar")
            database_path.write_text("db")
            state["initialized"] = True
        else:
            # Enforce replay-stable external assets by failing if any file was altered
            # between launch and completion.
            if state_dict.read_text() != "state":
                raise RuntimeError("external state_dict mutated during execution")
            if amber_tar.read_text() != "tar":
                raise RuntimeError("external amber_tar mutated during execution")
            if database_path.read_text() != "db":
                raise RuntimeError("external database mutated during execution")
        return state_dict, amber_tar, database_path

    def _run_compound_tamper(
        *,
        compound,
        state_dict,
        amber_tar,
        work_dir,
        lambda_states,
        seeds,
        temperature_kelvin,
        timestep_femtoseconds,
        equilibration_steps,
        production_steps,
        sample_interval,
        platform_name,
        sensitivity_only,
    ):
        state_dict.write_text("mutated-state-dict")
        amber_tar.write_text("mutated-amber-tar")
        database_path.write_text("mutated-database")
        return _stub_record(
            compound=compound,
            seeds=seeds,
            status="ok",
            prediction=-2.0,
            replicate_sd=0.0,
            replicate_sem=0.0,
            combined_uncertainty=0.0,
            quality_failures=[],
            quality_notes=["formal_accuracy_mode"],
        )

    try:
        rc, out_dir = _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            validate_sources_stub=_validate_sources_stub,
            run_compound_stub=_run_compound_tamper,
        )
    except RuntimeError as exc:
        assert "external" in str(exc)
    else:
        assert rc != 0
        assert not (out_dir / "summary.json").exists()


def test_json_output_forbids_nonfinite_numbers(monkeypatch, tmp_path):
    """Finite summary JSON is required; NaN/Inf must be rejected."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-json"

    def _run_compound_nan(
        *,
        compound,
        state_dict,
        amber_tar,
        work_dir,
        lambda_states,
        seeds,
        temperature_kelvin,
        timestep_femtoseconds,
        equilibration_steps,
        production_steps,
        sample_interval,
        platform_name,
        sensitivity_only,
    ):
        return _stub_record(
            compound=compound,
            status="ok",
            seeds=seeds,
            prediction=math.nan,
            replicate_sd=0.0,
            replicate_sem=0.0,
            combined_uncertainty=0.0,
            quality_failures=["delta_g_nan"],
            quality_notes=[],
        )

    try:
        rc, out_dir = _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            run_compound_stub=_run_compound_nan,
        )
    except (SystemExit, RuntimeError, ValueError):
        return
    assert rc != 0
    if (out_dir / "summary.json").exists():
        summary_text = (out_dir / "summary.json").read_text(encoding="utf-8")
        assert "NaN" not in summary_text
        assert "Infinity" not in summary_text


def test_nonfinite_mbar_or_force_values_are_not_accepted(monkeypatch, tmp_path):
    """NaN/inf from MBAR or force diagnostics must be treated as failed runs."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-mbar"

    def _run_compound_bad_mbar(
        *,
        compound,
        state_dict,
        amber_tar,
        work_dir,
        lambda_states,
        seeds,
        temperature_kelvin,
        timestep_femtoseconds,
        equilibration_steps,
        production_steps,
        sample_interval,
        platform_name,
        sensitivity_only,
    ):
        return _stub_record(
            compound=compound,
            status="ok",
            seeds=seeds,
            prediction=math.inf,
            replicate_sd=math.inf,
            replicate_sem=math.inf,
            combined_uncertainty=math.inf,
            quality_failures=[],
            quality_notes=["mbar_force_numerical_pathology"],
        )

    try:
        rc, out_dir = _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            run_compound_stub=_run_compound_bad_mbar,
        )
    except (SystemExit, RuntimeError, ValueError):
        return
    assert rc != 0
    if (out_dir / "summary.json").exists():
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        assert "Infinity" not in json.dumps(summary)


def test_cli_rejects_invalid_lambda_schedule(monkeypatch, tmp_path):
    """Lambda schedule must be finite and include exact 0/1 endpoints."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-lambda"
    with pytest.raises(SystemExit):
        _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            lambda_values="0,NaN,1",
        )


@pytest.mark.parametrize("temperature", ["nan", "inf", 0, -1.0])
def test_cli_rejects_invalid_temperature(monkeypatch, tmp_path, temperature):
    """CLI must reject non-positive or non-finite temperatures."""
    runner = _load_runner_module()
    work_dir = tmp_path / f"run-temp-{str(temperature).replace('.', '_')}"
    try:
        rc, out_dir = _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            temperature=temperature,
        )
    except (SystemExit, RuntimeError, ValueError):
        return
    assert rc != 0
    assert not (out_dir / "summary.json").exists()


@pytest.mark.parametrize("interval", ["0", "-1", "-5"])
def test_cli_rejects_invalid_sample_interval(monkeypatch, tmp_path, interval):
    """Sample interval must be a positive integer and compatible with production steps."""
    runner = _load_runner_module()
    work_dir = tmp_path / f"run-interval-{interval.replace('-', 'neg')}"
    try:
        rc, out_dir = _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            sample_interval=int(interval),
            production_steps=1000,
        )
    except (SystemExit, ZeroDivisionError):
        return
    assert rc != 0
    assert not (out_dir / "summary.json").exists()


@pytest.mark.parametrize("steps", ["0", "-10", "-100", "NaN", "inf"])
def test_cli_rejects_invalid_production_steps(monkeypatch, tmp_path, steps):
    """Production steps must be positive finite and compatible with sampling interval."""
    runner = _load_runner_module()
    work_dir = tmp_path / f"run-production-{str(steps).replace('-', 'neg')}"
    try:
        rc, out_dir = _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            sample_interval=5,
            production_steps=steps,
        )
    except (SystemExit, ZeroDivisionError):
        return
    assert rc != 0
    assert not (out_dir / "summary.json").exists()


@pytest.mark.parametrize("max_compounds", [0, -1])
def test_cli_rejects_non_positive_max_compounds(monkeypatch, tmp_path, max_compounds):
    """max-compounds must be a positive integer."""
    runner = _load_runner_module()
    work_dir = tmp_path / f"run-max-{max_compounds}"
    with pytest.raises(SystemExit):
        _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            max_compounds=max_compounds,
        )


def test_cli_rejects_empty_panel(monkeypatch, tmp_path):
    """Empty panels should not be accepted."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-empty-panel"
    with pytest.raises(SystemExit):
        _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            panel_payload={"molecules": []},
        )


def test_cli_rejects_negative_seed_input(monkeypatch, tmp_path):
    """Seeds must be positive finite integers."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-seed"
    with pytest.raises(SystemExit):
        _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="-1,20260730",
        )


def test_cli_rejects_path_traversal_compound_id(monkeypatch, tmp_path):
    """Compound-id filter should reject path traversal-like identifiers."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-compound-id"
    panel_compounds = [
        {
            "compound_id": "../unsafe-id",
            "name": "path-traversal",
            "smiles": "C",
        }
    ]
    with pytest.raises(SystemExit):
        _run_main(
            runner,
            work_dir,
            tmp_path,
            monkeypatch,
            seeds="20260729,20260730,20260731",
            panel_compounds=panel_compounds,
            compound_id="../unsafe-id",
        )


def test_sensitivity_mode_is_run_mode_not_physical_status(monkeypatch, tmp_path):
    """Sensitivity mode is protocol metadata and should not override physical status."""
    runner = _load_runner_module()
    work_dir = tmp_path / "run-sens-fail"

    def _run_compound_unconverged(
        *,
        compound,
        state_dict,
        amber_tar,
        work_dir,
        lambda_states,
        seeds,
        temperature_kelvin,
        timestep_femtoseconds,
        equilibration_steps,
        production_steps,
        sample_interval,
        platform_name,
        sensitivity_only,
    ):
        return _stub_record(
            compound=compound,
            status="unconverged",
            seeds=seeds,
            prediction=-2.0,
            replicate_sd=0.0,
            replicate_sem=0.0,
            combined_uncertainty=0.0,
            quality_failures=["test_failure"],
            quality_notes=["sensitivity_only_excluded_from_accuracy_metrics"],
        )

    rc, out_dir = _run_main(
        runner,
        work_dir,
        tmp_path,
        monkeypatch,
        seeds="20260729",
        sensitivity_only=True,
        run_compound_stub=_run_compound_unconverged,
    )
    assert rc != 0
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    record = summary["records"][0]
    assert summary["protocol"]["run_mode"] == "sensitivity_only"
    assert record["status"] == "unconverged"
    assert record["included_in_metrics"] is False
    assert record["status"] != "sensitivity_only"
    assert summary["metrics"]["n_success"] == 0
