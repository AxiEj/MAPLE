"""Self-contained tests for the pinned Gaussian-CHA OPT transaction runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
    monkeypatch.setattr(module, "PROTOCOL_TEMPLATE", protocol_changing)
    monkeypatch.setattr(
        module, "PROTOCOL_TEMPLATE_SHA256", module.sha256_bytes(serialized)
    )
    loaded, _ = module.load_protocol_template()
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
    }
    monkeypatch.setattr(module, "_phase_evidence", lambda *unused: evidence.copy())
    prereg = module.seal(
        {
            "schema_version": 1,
            "phase": "PREREGISTERED",
            "protocol_template_sha256": protocol_sha,
            "approved_plan_sha256": module.PLAN_SHA256,
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
