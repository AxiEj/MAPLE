"""Self-contained tests for the pinned Gaussian-CHA OPT transaction runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "docs/implicit-solvation/benchmarks/run_cha_gaussian_opt.py"


def _module():
    spec = importlib.util.spec_from_file_location("cha_gaussian_opt_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HarmonicComposed(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, *, fail_x_above=None, energy_offset=0.25):
        super().__init__()
        self.fail_x_above = fail_x_above
        self.energy_offset = energy_offset

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if atoms is None:
            raise ValueError("fixture requires atoms")
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if self.fail_x_above is not None and positions[0, 0] > self.fail_x_above:
            raise RuntimeError("synthetic rejected trial")
        # Represents already composed gas + solvent values.
        energy = float(np.sum(positions**2) + self.energy_offset)
        forces = -2.0 * positions
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "solvation": {
                "gas_energy_hartree": float(np.sum(positions**2)),
                "energy_hartree": self.energy_offset,
                "combined_energy_hartree": energy,
                "components_hartree": {
                    "polar": 0.4 * self.energy_offset,
                    "cavity": 0.2 * self.energy_offset,
                    "dispersion": 0.4 * self.energy_offset,
                },
            },
        }


def _protocol(module, *, calls=512, hashes=258):
    protocol, _ = module.load_protocol_template()
    protocol = json.loads(json.dumps(protocol))
    protocol["optimizer"]["per_row_caps"] = {
        "observer_calculator_calls": calls,
        "unique_successful_force_coordinate_hashes": hashes,
    }
    return protocol


def _topology():
    return {"atomic_numbers": [8, 1, 1], "content_sha256": "a" * 64}


def _row():
    return {
        "row_id": "center-00-sigma-00",
        "center_index": 0,
        "center_scale": 0.96,
        "sigma_e": 0.001,
        "positions_angstrom": [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }


def test_observer_records_only_after_complete_composed_force_and_enforces_caps():
    module = _module()
    from ase import Atoms

    atoms = Atoms("OHH", positions=_row()["positions_angstrom"])
    observer = module.ComposedRecordingCalculator(
        HarmonicComposed(),
        caps=module.ObserverCaps(calls=2, unique_force_coordinates=1),
    )
    atoms.calc = observer
    np.testing.assert_allclose(
        atoms.get_forces(), [[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert observer.frames[0]["combined_energy_hartree"] == pytest.approx(0.26)
    assert observer.frames[0]["solvation"]["combined_energy_hartree"] == pytest.approx(
        0.26
    )
    assert observer.unique_force_coordinate_count == 1

    atoms.set_positions([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(module.ObservationBudgetExceeded, match="unique"):
        atoms.get_forces()
    assert len(observer.frames) == 1
    with pytest.raises(module.ObservationBudgetExceeded, match="call"):
        atoms.get_forces()


def test_failed_trial_restores_last_force_frame_terminates_and_marks_runtime(tmp_path):
    module = _module()
    built = 0

    def factory(atoms, topology, sigma, log):
        nonlocal built
        built += 1
        return HarmonicComposed(fail_x_above=0.15 if built == 1 else None)

    class RejectingOptimization:
        def __init__(self, params, output, atoms):
            self.atoms = atoms

        def run(self):
            self.atoms.get_forces()  # last valid composed force frame
            self.atoms.set_positions(
                [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
            self.atoms.get_forces()  # rejected trial
            raise AssertionError("unreachable")

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=RejectingOptimization,
    )
    assert receipt["status"] == "RUNTIME_ERROR"
    assert receipt["rejected_evaluation"]["positions_angstrom"] == [
        [0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert receipt["retained_positions_angstrom"] == _row()["positions_angstrom"]
    assert receipt["retained_metrics"] is None
    assert receipt["fresh_final"]["tag"] == "FRESH_FINAL_UNCACHED"
    assert built == 2  # failed calculator is never reused for fresh final


def test_returned_atoms_and_low_force_do_not_turn_maxiter_into_success(tmp_path):
    module = _module()

    def factory(atoms, topology, sigma, log):
        return HarmonicComposed()

    class ReturnedAtomsOnly:
        def __init__(self, params, output, atoms):
            self.atoms = atoms

        def run(self):
            self.atoms.set_positions(np.zeros((3, 3)))
            self.atoms.get_forces()
            # Missing last-step metrics must remain infinity, never zero-filled.
            return self.atoms

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=ReturnedAtomsOnly,
    )
    assert receipt["status"] == "MAX_ITER"
    assert receipt["fresh_final_pass"] is True
    assert receipt["retained_metrics"]["dp_max_angstrom"] is None
    assert receipt["retained_metrics"]["dp_rms_angstrom"] is None
    assert receipt["returned_atoms_is_not_convergence_evidence"] is True


def test_all_four_retained_metrics_and_fresh_force_are_required(tmp_path):
    module = _module()

    def factory(atoms, topology, sigma, log):
        return HarmonicComposed()

    class ConvergedOptimization:
        def __init__(self, params, output, atoms):
            assert params == {
                "method": "lbfgs",
                "memory": 5,
                "curvature": 70.0,
                "max_step": 0.01,
                "max_iter": 256,
                "verbose": 1,
            }
            self.atoms = atoms

        def run(self):
            self.atoms.set_positions(np.zeros((3, 3)))
            self.atoms.get_forces()
            self.atoms.max_dp = 2e-4
            self.atoms.rms_dp = 1e-4
            return self.atoms

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=ConvergedOptimization,
    )
    assert receipt["status"] == "CONVERGED"
    assert receipt["fresh_final_pass"] is True
    assert receipt["observer_calculator_calls"] <= 512
    assert receipt["unique_successful_force_coordinate_hashes"] <= 258


def test_typed_size_domain_failure_is_retained_as_domain_rejected(tmp_path):
    module = _module()
    from maple.function.calculator.extra_correction.implicit.torch_chagb_gaussian import (
        GaussianChaSizeDomainError,
    )

    def factory(atoms, topology, sigma, log):
        return HarmonicComposed()

    class SizeRejectedOptimization:
        def __init__(self, params, output, atoms):
            self.atoms = atoms

        def run(self):
            self.atoms.get_forces()
            raise GaussianChaSizeDomainError(9.5)

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=SizeRejectedOptimization,
    )
    assert receipt["status"] == "DOMAIN_REJECTED"
    assert receipt["rejected_evaluation"]["type"] == "GaussianChaSizeDomainError"
    assert receipt["retained_coordinate_sha256"] == module.coordinate_sha256(
        _row()["positions_angstrom"]
    )


def test_no_successful_force_frame_falls_back_to_original_and_does_not_fake_replay(
    tmp_path,
):
    module = _module()
    built = 0

    def factory(atoms, topology, sigma, log):
        nonlocal built
        built += 1
        return HarmonicComposed(fail_x_above=0.05 if built == 1 else None)

    class FirstCallFails:
        def __init__(self, params, output, atoms):
            self.atoms = atoms

        def run(self):
            self.atoms.get_forces()

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=FirstCallFails,
    )
    assert receipt["status"] == "RUNTIME_ERROR"
    assert receipt["successful_force_frames"] == []
    assert receipt["retained_positions_angstrom"] == _row()["positions_angstrom"]
    assert receipt["fresh_replay"]["reason"] == "NO_RETAINED_COMPOSED_FORCE_FRAME"


def test_stale_fresh_energy_fails_replay_even_when_both_force_gates_pass(tmp_path):
    module = _module()
    built = 0

    def factory(atoms, topology, sigma, log):
        nonlocal built
        built += 1
        return HarmonicComposed(energy_offset=0.25 if built == 1 else 0.250001)

    class ConvergedOptimization:
        def __init__(self, params, output, atoms):
            self.atoms = atoms

        def run(self):
            self.atoms.set_positions(np.zeros((3, 3)))
            self.atoms.get_forces()
            self.atoms.max_dp = 1e-4
            self.atoms.rms_dp = 1e-4
            return self.atoms

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=ConvergedOptimization,
    )
    assert receipt["fresh_final_pass"] is True
    assert receipt["fresh_replay"]["passed"] is False
    assert receipt["status"] == "RUNTIME_ERROR"


def test_nonfinite_rejected_coordinates_are_json_safe_and_restore_last_frame(tmp_path):
    module = _module()

    def factory(atoms, topology, sigma, log):
        return HarmonicComposed()

    class NonfiniteTrial:
        def __init__(self, params, output, atoms):
            self.atoms = atoms

        def run(self):
            self.atoms.get_forces()
            positions = self.atoms.get_positions()
            positions[0, 0] = np.nan
            self.atoms.set_positions(positions)
            self.atoms.get_forces()

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=NonfiniteTrial,
    )
    assert receipt["status"] == "RUNTIME_ERROR"
    assert receipt["rejected_evaluation"]["positions_angstrom"][0][0] is None
    assert "nan" in receipt["rejected_evaluation"]["raw_positions_repr"]
    assert receipt["retained_positions_angstrom"] == _row()["positions_angstrom"]
    module.canonical_bytes(receipt)


def test_classifier_failure_never_masks_original_row_failure_and_fresh_final(
    tmp_path, monkeypatch
):
    module = _module()
    built = 0

    def factory(atoms, topology, sigma, log):
        nonlocal built
        built += 1
        if built == 1:
            raise RuntimeError("original model construction failure")
        return HarmonicComposed()

    def broken_classifier(exc):
        raise OSError("classifier import failure")

    monkeypatch.setattr(module, "_is_domain_exception", broken_classifier)
    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=lambda *args: None,
    )
    assert receipt["status"] == "RUNTIME_ERROR"
    assert receipt["rejected_evaluation"]["type"] == "RuntimeError"
    assert (
        receipt["rejected_evaluation"]["message"]
        == "original model construction failure"
    )
    assert receipt["classification_error"] == {
        "type": "OSError",
        "message": "classifier import failure",
    }
    assert receipt["fresh_final"]["tag"] == "FRESH_FINAL_UNCACHED"


def test_optimizer_import_failure_is_retained_before_factory_and_fresh_final_runs(
    tmp_path, monkeypatch
):
    module = _module()
    factory_calls = 0

    def factory(atoms, topology, sigma, log):
        nonlocal factory_calls
        factory_calls += 1
        return HarmonicComposed()

    def broken_loader():
        raise ImportError("optimizer import blocked")

    monkeypatch.setattr(
        module, "_load_optimizer_implementation", broken_loader, raising=False
    )
    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
    )
    assert receipt["status"] == "RUNTIME_ERROR"
    assert receipt["rejected_evaluation"]["type"] == "ImportError"
    assert receipt["rejected_evaluation"]["message"] == "optimizer import blocked"
    assert factory_calls == 1  # fresh-final only; optimization factory was never built
    assert receipt["fresh_final"]["tag"] == "FRESH_FINAL_UNCACHED"


def test_runtime_reporting_failure_is_retained_not_raised(tmp_path, monkeypatch):
    module = _module()

    def factory(atoms, topology, sigma, log):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        module,
        "_runtime_record",
        lambda: (_ for _ in ()).throw(RuntimeError("torch runtime probe failed")),
    )
    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=lambda *args: None,
    )
    assert receipt["status"] == "RUNTIME_ERROR"
    assert receipt["runtime"] == {
        "type": "RuntimeError",
        "message": "torch runtime probe failed",
    }


def test_contact_v2_domain_error_is_structured_and_nonfinite_values_are_tagged():
    module = _module()
    from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_contact_v2 import (
        ContactV2DomainError,
        ContactV2FailureReason,
    )

    error = ContactV2DomainError(
        ContactV2FailureReason.NONFINITE_ALGEBRA,
        "bad node",
        (
            ("A", float("nan")),
            ("Delta", float("inf")),
            ("owner_index", 1.0),
            ("receiver_index", 0.0),
            ("node_index", 7.0),
        ),
    )
    serialized = module.serialize_exception(error)
    assert serialized["type"] == "ContactV2DomainError"
    assert serialized["reason"] == "nonfinite-algebra"
    assert serialized["raw_margins"]["A"] == {
        "tag": "NONFINITE_FLOAT",
        "value": "NaN",
    }
    assert serialized["raw_margins"]["Delta"] == {
        "tag": "NONFINITE_FLOAT",
        "value": "+Infinity",
    }
    assert module._is_domain_exception(error) is True
    module.canonical_bytes(serialized)


def test_fresh_final_domain_failure_retains_reason_and_is_truthfully_classified(
    tmp_path,
):
    module = _module()
    from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_contact_v2 import (
        ContactV2DomainError,
        ContactV2FailureReason,
    )

    built = 0

    def factory(atoms, topology, sigma, log):
        nonlocal built
        built += 1
        if built == 1:
            raise RuntimeError("initial setup failed")
        raise ContactV2DomainError(
            ContactV2FailureReason.ILL_CONDITIONED_DELTA,
            "fresh node failed",
            (("q", 1e-9), ("q_min", 2**-20), ("node_index", 2.0)),
        )

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=lambda *args: None,
    )
    assert receipt["status"] == "RUNTIME_ERROR"
    assert receipt["rejected_evaluation"]["message"] == "initial setup failed"
    assert receipt["fresh_final"]["status"] == "DOMAIN_REJECTED"
    assert receipt["fresh_final"]["reason"] == "ill-conditioned-delta"
    assert receipt["fresh_final"]["raw_margins"]["node_index"] == 2.0


def test_fresh_final_only_domain_failure_sets_overall_domain_rejected(tmp_path):
    module = _module()
    from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_contact_v2 import (
        ContactV2DomainError,
        ContactV2FailureReason,
    )

    built = 0

    def factory(atoms, topology, sigma, log):
        nonlocal built
        built += 1
        if built == 1:
            return HarmonicComposed()
        raise ContactV2DomainError(
            ContactV2FailureReason.TANGENCY,
            "fresh-only tangency",
            (("owner_index", 0.0), ("receiver_index", 1.0)),
        )

    class ConvergedOptimization:
        def __init__(self, params, output, atoms):
            self.atoms = atoms

        def run(self):
            self.atoms.set_positions(np.zeros((3, 3)))
            self.atoms.get_forces()
            self.atoms.max_dp = 1e-4
            self.atoms.rms_dp = 1e-4
            return self.atoms

    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=ConvergedOptimization,
    )
    assert receipt["fresh_final"]["status"] == "DOMAIN_REJECTED"
    assert receipt["status"] == "DOMAIN_REJECTED"


def test_serializer_failure_never_masks_original_exception(tmp_path, monkeypatch):
    module = _module()

    def factory(atoms, topology, sigma, log):
        raise RuntimeError("original failure")

    monkeypatch.setattr(
        module,
        "_serialize_exception_details",
        lambda exc: (_ for _ in ()).throw(OSError("serializer broke")),
        raising=False,
    )
    receipt = module.execute_opt_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path,
        calculator_factory=factory,
        optimization_factory=lambda *args: None,
    )
    assert receipt["rejected_evaluation"]["type"] == "RuntimeError"
    assert receipt["rejected_evaluation"]["message"] == "original failure"
    assert receipt["rejected_evaluation"]["serializer_error"] == {
        "type": "OSError",
        "message": "serializer broke",
    }


def test_five_point_units_and_matrix_completeness_are_fail_closed():
    module = _module()
    h = 1e-4
    samples = [(-2 * h) ** 3, (-h) ** 3, h**3, (2 * h) ** 3]
    assert module.five_point_derivative(samples, h) == pytest.approx(0.0, abs=1e-20)
    rows = [{"status": "CONVERGED", "fresh_final_pass": True} for _ in range(15)]
    assert module.summarize_matrix(rows)["status"] == "VALIDATED"
    assert module.summarize_matrix(rows[:14])["status"] == "GAUSSIAN_OPT_INCOMPLETE"
    rows[7]["status"] = "MAX_ITER"
    assert module.summarize_matrix(rows)["status"] == "GAUSSIAN_OPT_INCOMPLETE"


class _ChangingRead:
    def __init__(self, first: bytes):
        self.first = first
        self.count = 0

    def read_bytes(self):
        self.count += 1
        return self.first if self.count == 1 else b'{"mutated":true}'


def test_hashed_json_and_external_protocol_are_single_read_race_safe(monkeypatch):
    module = _module()
    raw = b'{"key":"frozen"}'
    changing = _ChangingRead(raw)
    value, digest = module.read_hashed_json(changing, "fixture")
    assert value == {"key": "frozen"}
    assert digest == module.sha256_bytes(raw)
    assert changing.count == 1

    template = json.loads(module.PROTOCOL_TEMPLATE.read_text())
    serialized = json.dumps(template).encode()
    protocol_changing = _ChangingRead(serialized)
    context = module.DEFAULT_V1_CONTEXT._replace(
        protocol_template=protocol_changing,
        protocol_template_sha256=module.sha256_bytes(serialized),
    )
    loaded, _ = module.load_protocol_template(context)
    assert loaded == template
    assert protocol_changing.count == 1


def test_atomic_writer_refuses_overwrite_and_never_replaces_evidence(tmp_path):
    module = _module()
    path = tmp_path / "receipt.json"
    module.write_json_new_atomic(path, {"phase": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_json_new_atomic(path, {"phase": 2})
    assert path.read_bytes() == before


def test_preregistration_external_pin_is_checked_before_parse(tmp_path):
    module = _module()
    path = tmp_path / "preregistration.json"
    path.write_bytes(b"not-json")
    with pytest.raises(ValueError, match="external SHA256 pin"):
        module.load_preregistration(tmp_path, "0" * 64)


def test_preregistered_roster_is_reconstructed_not_only_counted(tmp_path, monkeypatch):
    module = _module()
    protocol, protocol_sha = module.load_protocol_template()
    source = _synthetic_center_source()
    rows = module.build_row_inventory(source, protocol)
    evidence = {
        "center": source,
        "topology": {},
        "input_sha256": {"fixture": "1" * 64},
        "source": {"fixture.py": "2" * 64},
        "protected": {"old.py": "3" * 64},
        "source_identity_sha256": "4" * 64,
        "numerical_profile_id": module.DEFAULT_V1_CONTEXT.numerical_profile_id,
        "context_id": module.DEFAULT_V1_CONTEXT.context_id,
        "external_evidence_sha256": {},
    }
    monkeypatch.setattr(module, "_phase_evidence", lambda *unused: evidence.copy())
    prereg = module.seal(
        {
            "schema_version": 1,
            "phase": "PREREGISTERED",
            "protocol_template_sha256": protocol_sha,
            "approved_plan_sha256": module.PLAN_SHA256,
            "approved_handoff_sha256": module.DEFAULT_V1_CONTEXT.approved_handoff_sha256,
            "numerical_profile_id": module.DEFAULT_V1_CONTEXT.numerical_profile_id,
            "context_id": module.DEFAULT_V1_CONTEXT.context_id,
            "external_evidence_sha256": {},
            "protocol_id": protocol["protocol_id"],
            "runtime_origins": {},
            "input_sha256": evidence["input_sha256"],
            "source_before": evidence["source"],
            "protected_before": evidence["protected"],
            "source_identity_sha256": evidence["source_identity_sha256"],
            "rows": rows,
            "prohibitions": {},
        }
    )
    path = tmp_path / "preregistration.json"
    path.write_bytes(module.canonical_bytes(prereg) + b"\n")
    digest = module.sha256_file(path)
    loaded = module.load_preregistration(tmp_path, digest)[1]
    assert loaded["rows"] == rows

    prereg["rows"][0], prereg["rows"][1] = prereg["rows"][1], prereg["rows"][0]
    prereg = module.seal(prereg)
    changed = tmp_path / "changed"
    changed.mkdir()
    changed_path = changed / "preregistration.json"
    changed_path.write_bytes(module.canonical_bytes(prereg) + b"\n")
    with pytest.raises(ValueError, match="row roster"):
        module.load_preregistration(changed, module.sha256_file(changed_path))


def test_protocol_freezes_exact_matrix_optimizer_reference_and_prohibitions():
    module = _module()
    protocol, digest = module.load_protocol_template()
    assert digest == module.PROTOCOL_TEMPLATE_SHA256
    assert protocol["inputs"]["center_scales"] == [0.96, 0.98, 1.0, 1.02, 1.04]
    assert protocol["inputs"]["sigma_e"] == [0.001, 0.003, 0.01]
    assert protocol["inputs"]["row_count"] == 15
    assert protocol["scalar"]["reference_quadrature_orders"] == [64, 96, 128]
    assert protocol["crossing_validation"]["offsets_angstrom"] == {
        "start": -0.002,
        "stop": 0.002,
        "count": 41,
    }
    assert protocol["crossing_validation"]["fd_steps_angstrom"] == [2e-5, 1e-5, 5e-6]
    assert protocol["optimizer"]["memory"] == 5
    assert protocol["optimizer"]["curvature"] == 70.0
    assert protocol["optimizer"]["max_step_angstrom"] == 0.01
    assert protocol["optimizer"]["max_iterations"] == 256
    assert protocol["optimizer"]["thresholds"] == {
        "f_max_hartree_per_angstrom": 3e-4,
        "f_rms_hartree_per_angstrom": 2e-4,
        "dp_max_angstrom": 3e-4,
        "dp_rms_angstrom": 2e-4,
    }
    assert protocol["optimizer"]["fresh_replay"] == {
        "energy_max_abs_hartree": 1e-10,
        "force_max_abs_hartree_per_angstrom": 1e-8,
        "coordinate_hash_exact": True,
        "energy_fields": [
            "combined",
            "gas",
            "solvent",
            "polar",
            "cavity",
            "dispersion",
        ],
    }
    assert (
        protocol["execution"]["run_validate_require_external_preregistration_sha256"]
        is True
    )
    for key in (
        "network",
        "subprocess",
        "model_download",
        "qm_or_recharge",
        "experimental_labels",
        "post_hoc_width_selection",
    ):
        assert protocol["execution"][key] is False


def test_run_reports_phase_passed_or_incomplete_and_rejects_unknown_phase(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "assert_runtime_origins", lambda *args: {})
    monkeypatch.setattr(
        module,
        "run_crossing_reference",
        lambda output, pin, context: {"passed": True},
    )
    passed = module.run(Path("unused"), "a" * 64, "crossing")
    assert passed["status"] == "PHASE_PASSED"

    monkeypatch.setattr(
        module,
        "run_crossing_reference",
        lambda output, pin, context: {"passed": False},
    )
    failed = module.run(Path("unused"), "a" * 64, "crossing")
    assert failed["status"] == "GAUSSIAN_OPT_INCOMPLETE"
    with pytest.raises(ValueError, match="unknown run phase"):
        module.run(Path("unused"), "a" * 64, "not-a-phase")


def test_main_returns_nonzero_when_selected_phase_fails(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_output_allowed", lambda output, context: True)
    monkeypatch.setattr(
        module,
        "run",
        lambda output, pin, phase, context: {"status": "GAUSSIAN_OPT_INCOMPLETE"},
    )
    assert (
        module.main(
            [
                "run",
                "--output",
                "/tmp/fake-campaign",
                "--expected-preregistration-sha256",
                "a" * 64,
                "--phase",
                "combined-fd",
            ]
        )
        == 1
    )
    monkeypatch.setattr(module, "run", lambda output, pin, phase, context: {})
    assert (
        module.main(
            [
                "run",
                "--output",
                "/tmp/fake-campaign",
                "--expected-preregistration-sha256",
                "a" * 64,
            ]
        )
        == 1
    )


def test_combined_fd_creates_log_parent_before_calculator_factory(
    tmp_path, monkeypatch
):
    module = _module()

    def factory(atoms, topology, sigma, log):
        assert log.parent.is_dir()
        return HarmonicComposed()

    record = module._combined_fd_row(
        _row(),
        _topology(),
        _protocol(module),
        tmp_path / "new" / "nested" / "calculator.log",
        factory,
    )
    assert record["row_id"] == _row()["row_id"]


def test_reference_geometry_metadata_retains_and_revalidates_every_order():
    module = _module()
    levels = tuple(
        SimpleNamespace(
            azimuth_order=order,
            inverse_cube_quad_error_estimate_per_angstrom3=order * 1e-12,
            meridian_evaluations=order * 2,
            inverse_cube_per_angstrom3=np.full(3, order, dtype=np.float64),
            inverse_born_per_angstrom=np.full(3, order + 1, dtype=np.float64),
            gauss_closure_vector_angstrom2=np.full(3, order + 2, dtype=np.float64),
        )
        for order in (64, 96, 128)
    )
    geometry = SimpleNamespace(
        r6_levels=levels,
        r6_diagnostics={"independent": True},
        diagnostics={
            "azimuth_orders": (64, 96, 128),
            "epsabs": 1e-12,
            "epsrel": 1e-12,
            "cavity_mixed_vector_quad_error": [1e-14, 1e-14, 1e-14],
            "dispersion_mixed_vector_quad_error": [1e-14, 1e-14, 1e-14],
            "mixed_vector_quad_errors_are_scalar_uncertainties": False,
            "error_estimates_are_rigorous_bounds": False,
        },
        cavity_volume_angstrom3=1.25,
        cavity_kcal_mol=0.5,
        dispersion_kcal_mol=-0.1,
        coordinates_sha256="1" * 64,
        parameters_sha256="2" * 64,
        identity_sha256="3" * 64,
        payload_sha256="4" * 64,
    )
    metadata = module._reference_geometry_metadata(geometry)
    assert [level["azimuth_order"] for level in metadata["r6_levels"]] == [64, 96, 128]
    assert metadata["epsabs"] == metadata["epsrel"] == 1e-12
    assert module._validate_reference_geometry_metadata(
        metadata, (64, 96, 128), 1e-12, 1e-12
    )
    metadata["r6_levels"][1]["meridian_evaluations"] += 1
    assert not module._validate_reference_geometry_metadata(
        metadata, (64, 96, 128), 1e-12, 1e-12
    )


def _synthetic_center_source():
    rows = []
    for scale in (0.96, 0.98, 1.0, 1.02, 1.04):
        rows.append(
            {
                "scale": scale,
                "row_status": "EVENT" if scale == 1.0 else "REGULAR",
                "positions_angstrom": [
                    [0.0, 0.0, 0.0],
                    [scale, 0.0, 0.0],
                    [0.0, scale, 0.0],
                ],
            }
        )
    return {"rows": rows}
