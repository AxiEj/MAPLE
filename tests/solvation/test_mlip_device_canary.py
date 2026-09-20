"""Pure guards for the provenance-bound MLIP/device canary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

BENCHMARKS = Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import run_mlip_device_canary as canary  # pyright: ignore[reportMissingImports]


class PreparedQuadratic(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]  # noqa: RUF012
    hessian = "numerical"
    device = "cuda:0"
    dtype = "native"

    def __init__(self, events):
        super().__init__()
        self.events = events
        self.model = SimpleNamespace(parameters=lambda: iter(()))
        self.solvent_correction = SimpleNamespace(
            provider=SimpleNamespace(
                charges=np.array([-0.8, 0.4, 0.4]),
                radius_result=SimpleNamespace(radii_angstrom=np.array([1.5, 1.2, 1.2])),
                provenance={
                    "provider": "openmm",
                    "platform": "CPU",
                    "execution": {"requested_platform": "CPU"},
                },
            )
        )

    def prepare_numerical_derivatives(self):
        self.events.append("prepare")
        return 5e-4

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.events.append("evaluate")
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        x = self.atoms.positions.reshape(-1)
        self.results = {
            "energy": float(x @ x / 2),
            "free_energy": float(x @ x / 2),
            "forces": -x.reshape((-1, 3)),
        }

    def get_hessian(self, atoms, delta=None):
        self.events.append("hessian")
        self.last_numerical_hessian_diagnostics = {
            "cartesian_displacement_angstrom": delta
        }
        return np.eye(3 * len(atoms))


def _fake_input(tmp_path):
    source = tmp_path / "water/fixed.mol2"
    source.parent.mkdir(parents=True)
    source.write_text("fixed test input")
    return source


def test_model_options_are_per_backend_and_do_not_promote_non_ani_models():
    assert canary.MODEL_CONFIGS["ani2x"]["model_options"] == {
        "hessian": "numerical",
        "dtype": "float64",
    }
    assert canary.MODEL_CONFIGS["aimnet2"]["model_options"] == {
        "hessian": "numerical"
    }
    assert canary.MODEL_CONFIGS["maceoff23m"]["model_options"] == {
        "hessian": "numerical"
    }


def test_case_uses_registry_factory_and_prepares_before_first_evaluation(tmp_path, monkeypatch):
    source = _fake_input(tmp_path)
    atoms = Atoms("OH2", positions=np.arange(9).reshape((3, 3)) / 10)
    events = []
    calculator = PreparedQuadratic(events)
    calls = []

    monkeypatch.setattr(canary, "MOL2Reader", lambda *args, **kwargs: atoms)

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(set_calculator=lambda: calculator)

    monkeypatch.setattr(canary, "SetCalculator", build)
    monkeypatch.setattr(
        canary,
        "_checkpoint_record",
        lambda model: {"path": model, "exists": True, "sha256": "a" * 64},
    )
    result = canary.run_efh_case(
        source,
        model="aimnet2",
        device="cuda:0",
        openmm_platform="CPU",
        attempt_dir=tmp_path / "attempt",
    )

    assert result["status"] == "completed", result
    assert events[0] == "prepare"
    assert events.index("prepare") < events.index("evaluate")
    assert calls[0][1]["model_options"] == {"hessian": "numerical"}
    assert calls[0][1]["solvation_options"]["platform"] == "CPU"
    assert result["execution_after_evaluation"]["model"]["backend"] == "PreparedQuadratic"
    assert result["execution_after_evaluation"]["solvent"]["platform"] == "CPU"
    assert result["fixed_identity_preserved"] is True
    assert Path(result["forces"]["path"]).is_file()
    assert Path(result["hessian"]["path"]).is_file()


def test_native_failure_remains_a_failed_requested_case(tmp_path, monkeypatch):
    source = _fake_input(tmp_path)
    monkeypatch.setattr(
        canary,
        "MOL2Reader",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("native failure")),
    )
    monkeypatch.setattr(
        canary,
        "_checkpoint_record",
        lambda model: {"path": model, "exists": True, "sha256": "a" * 64},
    )
    result = canary.run_efh_case(
        source,
        model="ani2x",
        device="cuda:0",
        openmm_platform="CPU",
        attempt_dir=tmp_path / "failed",
    )
    assert result["status"] == "failed"
    assert "native failure" in result["error"]
    saved = canary.json.loads((tmp_path / "failed/result.json").read_text())
    assert saved["status"] == "failed"


def test_backend_that_reports_wrong_device_cannot_complete(tmp_path, monkeypatch):
    source = _fake_input(tmp_path)
    atoms = Atoms("OH2", positions=np.arange(9).reshape((3, 3)) / 10)
    calculator = PreparedQuadratic([])
    calculator.device = "cpu"
    monkeypatch.setattr(canary, "MOL2Reader", lambda *args, **kwargs: atoms)
    monkeypatch.setattr(
        canary,
        "_build_calculator",
        lambda *args, **kwargs: calculator,
    )
    monkeypatch.setattr(
        canary,
        "_checkpoint_record",
        lambda model: {"path": model, "exists": True, "sha256": "a" * 64},
    )
    result = canary.run_efh_case(
        source,
        model="ani2x",
        device="cuda:0",
        openmm_platform="CPU",
        attempt_dir=tmp_path / "lying",
    )
    assert result["status"] == "failed"
    assert result["execution_request_check"]["passed"] is False
    assert "contradicts" in result["error"]


def test_missing_checkpoint_fails_before_factory_or_network(tmp_path, monkeypatch):
    source = _fake_input(tmp_path)
    monkeypatch.setattr(
        canary,
        "_checkpoint_record",
        lambda model: {"path": model, "exists": False, "sha256": None},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("factory must not be called")

    monkeypatch.setattr(canary, "SetCalculator", forbidden)
    result = canary.run_efh_case(
        source,
        model="aimnet2",
        device="cuda:0",
        openmm_platform="CPU",
        attempt_dir=tmp_path / "missing-checkpoint",
    )
    assert result["status"] == "failed"
    assert "Local checkpoint is required" in result["error"]


def _completed_record(tmp_path, model, device, energy, force, hessian):
    folder = tmp_path / f"{model}-{device.replace(':', '-')}"
    folder.mkdir()
    np.save(folder / "forces.npy", np.asarray(force))
    np.save(folder / "hessian.npy", np.asarray(hessian))
    return {
        "model": model,
        "device": device,
        "status": "completed",
        "energy_hartree": energy,
        "forces": {"path": str(folder / "forces.npy")},
        "hessian": {"path": str(folder / "hessian.npy")},
    }


def test_device_comparison_reports_differences_without_universal_pass_gate(tmp_path):
    cpu = _completed_record(tmp_path, "ani2x", "cpu", 1.0, [[1, 2, 3]], np.eye(3))
    gpu = _completed_record(
        tmp_path,
        "ani2x",
        "cuda:0",
        1.0001,
        [[1, 2.001, 3]],
        np.eye(3) * 1.01,
    )
    result = canary.compare_device_pairs([cpu, gpu])
    assert result[0]["status"] == "compared"
    assert result[0]["absolute_energy_difference_hartree"] == pytest.approx(1e-4)
    assert result[0]["maximum_force_difference_hartree_per_angstrom"] == pytest.approx(1e-3)
    assert result[0]["maximum_hessian_difference_hartree_per_angstrom2"] == pytest.approx(1e-2)
    assert result[0]["acceptance_gate"] is None


def test_missing_cpu_pair_is_reported_unavailable_not_silently_omitted():
    result = canary.compare_device_pairs(
        [{"model": "maceoff23m", "device": "cuda:0", "status": "failed"}]
    )
    assert result == [
        {
            "model": "maceoff23m",
            "reference_device": "cpu",
            "accelerator_device": "cuda:0",
            "status": "unavailable",
            "acceptance_gate": None,
        }
    ]


def test_protocol_claim_scope_does_not_generalize_to_registry_or_all_gpu():
    scope = canary.PROTOCOL["claim_scope"].lower()
    assert "not validation of every registered" in scope
    assert "all-gpu" in scope


def test_effective_protocol_reflects_cli_subset_instead_of_template_matrix():
    result = canary.effective_protocol(
        models=["aimnet2"],
        devices=[{"requested": "cuda", "canonical": "cuda:0"}],
        openmm_platform="Reference",
        workflow_models=[],
        workflow_devices=[],
    )
    assert result["models"] == ["aimnet2"]
    assert result["devices"] == [{"requested": "cuda", "canonical": "cuda:0"}]
    assert result["endpoint"]["platform"] == "Reference"
    assert result["workflow_models"] == []


def test_failed_workflow_stays_in_command_success_denominator():
    result = canary.summarize_execution(
        [{"status": "completed"}],
        [{"status": "failed", "actual_execution_completed": False}],
        requested_case_count=1,
        requested_workflow_case_count=1,
        source_unchanged=True,
    )
    assert result["completed_case_count"] == 1
    assert result["failed_workflow_case_count"] == 1
    assert result["actual_execution_all_completed"] is False


def test_source_change_blocks_command_success_even_when_all_tasks_executed():
    result = canary.summarize_execution(
        [{"status": "completed"}],
        [{"status": "completed", "actual_execution_completed": True}],
        requested_case_count=1,
        requested_workflow_case_count=1,
        source_unchanged=False,
    )
    assert result["actual_execution_all_completed"] is False


def test_device_alias_is_resolved_once_and_both_spellings_are_retained(monkeypatch):
    class Canonical:
        def __str__(self):
            return "cuda:0"

    monkeypatch.setattr(canary, "resolve_torch_device", lambda requested: Canonical())
    assert canary.resolve_device_specs(["gpu0"]) == [
        {"requested": "gpu0", "canonical": "cuda:0"}
    ]


def test_workflow_cli_is_opt_in_without_nonempty_append_defaults():
    args = canary.build_parser().parse_args(
        ["--inputs-dir", "inputs", "--output-dir", "output"]
    )
    assert args.workflow_model == []
    assert args.workflow_device == []


def test_iteration_count_reads_prfo_colon_format(tmp_path):
    output = tmp_path / "prfo.out"
    output.write_text("Iteration: 1\nIteration: 4\n")
    assert canary._iteration_count(output) == 4
