from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import audit_cha_gaussian_terminal as audit  # pyright: ignore[reportMissingImports]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_json_checks_raw_sha_before_parsing(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("not-json")

    with pytest.raises(ValueError, match="external SHA256 pin differs"):
        audit.read_externally_pinned_json(path, "0" * 64, "fixture")


def test_atomic_writer_never_replaces_existing_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    audit.write_json_new_atomic(path, {"attempt": 1})

    with pytest.raises(FileExistsError):
        audit.write_json_new_atomic(path, {"attempt": 2})

    assert json.loads(path.read_text()) == {"attempt": 1}


def test_terminal_binding_rejects_retained_fresh_coordinate_mismatch():
    positions = np.zeros((3, 3))
    row = {
        "row_id": "center-00-sigma-0.001",
        "center_index": 0,
        "center_scale": 0.96,
        "sigma_e": 0.001,
        "status": "CONVERGED",
        "retained_positions_angstrom": positions.tolist(),
        "retained_coordinate_sha256": audit.coordinate_sha256(positions),
        "successful_force_frames": [],
        "fresh_final": {
            "tag": "FRESH_FINAL_UNCACHED",
            "positions_angstrom": (positions + 0.01).tolist(),
            "coordinate_sha256": audit.coordinate_sha256(positions + 0.01),
        },
        "content_sha256": "unused-by-this-unit",
    }

    with pytest.raises(ValueError, match="retained and fresh-final coordinates differ"):
        audit.terminal_inventory_entry(row, "a" * 64)


def test_terminal_binding_retains_failed_opt_in_fixed_inventory():
    positions = np.zeros((3, 3))
    digest = audit.coordinate_sha256(positions)
    row = {
        "row_id": "center-00-sigma-0.001",
        "center_index": 0,
        "center_scale": 0.96,
        "sigma_e": 0.001,
        "status": "MAX_ITER",
        "retained_positions_angstrom": positions.tolist(),
        "retained_coordinate_sha256": digest,
        "successful_force_frames": [],
        "fresh_final": {
            "tag": "FRESH_FINAL_UNCACHED",
            "positions_angstrom": positions.tolist(),
            "coordinate_sha256": digest,
        },
        "content_sha256": "unused-by-this-unit",
    }

    entry = audit.terminal_inventory_entry(row, "a" * 64)

    assert entry["opt_status"] == "MAX_ITER"
    assert entry["terminal_coordinate_sha256"] == digest


def test_reference_cache_reuses_only_identical_coordinate_parameter_order():
    calls = []

    class FakeOracle:
        @staticmethod
        def prepare_gaussian_reference_geometry(*args, **kwargs):
            calls.append((np.asarray(args[0]).copy(), kwargs["azimuth_orders"]))
            return object()

    topology = {
        "effective_charges_e": [-0.8, 0.4, 0.4],
        "cha_radii_angstrom": [1.8, 1.0, 1.0],
        "lj_rmin_angstrom": [1.8, 0.3, 0.3],
        "lj_epsilon_kcal_mol": [0.09, 0.004, 0.004],
    }
    cache = audit.ReferenceGeometryCache(FakeOracle(), topology, 1e-12, 1e-12)
    positions = np.zeros((3, 3))

    first = cache.get(positions, 128)
    second = cache.get(positions.copy(), 128)
    moved = positions.copy()
    moved[0, 0] = 1e-9
    third = cache.get(moved, 128)

    assert first is second
    assert third is not first
    assert len(calls) == 2


def test_five_point_stencils_rederive_force_and_disagreement():
    steps = [2e-5, 1e-5, 5e-6]
    stencils = []
    for step in steps:
        samples = []
        for multiplier in (-2.0, -1.0, 1.0, 2.0):
            x = multiplier * step
            samples.append(
                {
                    "multiplier": multiplier,
                    "energy": 3.0 * x + 0.25 * x**4,
                }
            )
        stencils.append({"step_angstrom": step, "samples": samples})

    derived = audit.derive_five_point_force(stencils, steps, energy_key="energy")

    assert derived["forces"] == pytest.approx([-3.0, -3.0, -3.0], abs=1e-12)
    assert derived["finest_step_disagreement"] < 1e-12


def test_summary_requires_campaign_validation_and_exact_15_rows():
    expected = [f"row-{index:02d}" for index in range(15)]
    rows = [
        {
            "row_id": row_id,
            "opt_status": "CONVERGED",
            "solvent_passed": True,
            "combined_passed": True,
        }
        for row_id in expected
    ]

    passed = audit.summarize_terminal_audit(rows, expected, campaign_validated=True)
    failed = audit.summarize_terminal_audit(rows, expected, campaign_validated=False)

    assert passed["status"] == "VALIDATED"
    assert failed["status"] == "GAUSSIAN_OPT_INCOMPLETE"


def test_summary_does_not_drop_failed_opt_row_from_denominator():
    expected = [f"row-{index:02d}" for index in range(15)]
    rows = [
        {
            "row_id": row_id,
            "opt_status": "CONVERGED",
            "solvent_passed": True,
            "combined_passed": True,
        }
        for row_id in expected
    ]
    rows[7]["opt_status"] = "MAX_ITER"

    result = audit.summarize_terminal_audit(rows, expected, campaign_validated=True)

    assert result["row_count"] == 15
    assert result["opt_converged_count"] == 14
    assert result["status"] == "GAUSSIAN_OPT_INCOMPLETE"


def test_resume_rejects_receipt_bound_to_different_audit_preregistration():
    runner = audit._runner()
    entry = {"row_id": "row-00"}
    record = runner.seal(
        {
            "case_identity": entry,
            "audit_preregistration_sha256": "wrong",
            "campaign_preregistration_sha256": "c" * 64,
        }
    )

    with pytest.raises(ValueError, match="audit preregistration binding differs"):
        audit.verify_audit_record_binding(record, entry, "a" * 64, "c" * 64, runner)


def test_cli_run_requires_both_external_preregistration_pins(tmp_path):
    with pytest.raises(SystemExit) as caught:
        audit.main(
            [
                "run",
                "--campaign",
                str(tmp_path / "campaign"),
                "--output",
                str(tmp_path / "terminal-audit"),
            ]
        )

    assert caught.value.code == 2


def test_combined_audit_creates_log_parent_before_calculator_factory(tmp_path):
    class FactoryReached(RuntimeError):
        pass

    class FakeRunner:
        @staticmethod
        def build_composed_calculator(atoms, topology, sigma_e, log):
            assert log.parent.is_dir()
            raise FactoryReached

    entry = {
        "terminal_positions_angstrom": np.zeros((3, 3)).tolist(),
        "sigma_e": 0.001,
    }
    topology = {"atomic_numbers": [8, 1, 1]}

    with pytest.raises(FactoryReached):
        audit._combined_audit(
            entry,
            topology,
            {"terminal_fd": {"steps_angstrom": [], "stencil_multipliers": []}},
            FakeRunner(),
            tmp_path / "fresh" / "calculator.log",
        )


def test_terminal_receipt_binds_explicit_numerical_profile_context():
    runner = audit._runner()
    context = runner.V2_RUN_CONTEXT
    entry = {"row_id": "row-v2"}
    record = audit._row_receipt(
        entry,
        "a" * 64,
        "c" * 64,
        {"status": "fixture"},
        runner,
        context,
    )
    audit.verify_audit_record_binding(
        record, entry, "a" * 64, "c" * 64, runner, context
    )
    assert record["numerical_profile_id"] == context.numerical_profile_id
    assert record["context_id"] == context.context_id


def test_campaign_recompute_uses_explicit_no_write_seam(tmp_path):
    calls = []

    class FakeRunner:
        @staticmethod
        def read_hashed_json(path, label):
            return {"status": "VALIDATED", "content_sha256": "x"}, "digest"

        @staticmethod
        def verify_seal(value, label):
            return None

        @staticmethod
        def validate(campaign, sha, context, *, write_output):
            calls.append((campaign, sha, context, write_output))
            return {"status": "VALIDATED", "content_sha256": "x"}

        @staticmethod
        def canonical_bytes(value):
            return json.dumps(value, sort_keys=True).encode()

    context = object()
    assert audit._campaign_recomputed(FakeRunner(), tmp_path, "pinned", context)
    assert calls == [(tmp_path, "pinned", context, False)]


def test_terminal_cli_reuses_v2_unique_campaign_boundary(tmp_path):
    runner = audit._runner()
    reserved = runner.V2_RUN_CONTEXT.output_root / "planning"
    with pytest.raises(SystemExit) as exc:
        audit.main(
            [
                "preregister",
                "--campaign",
                str(reserved),
                "--output",
                str(reserved / "terminal-audit"),
                "--profile",
                runner.V2_PROFILE_ID,
                "--expected-campaign-preregistration-sha256",
                "a" * 64,
            ]
        )
    assert exc.value.code == 2


def test_reference_metadata_coordinate_hash_rejects_valid_metadata_from_other_point():
    positions = np.zeros((3, 3))
    other = positions.copy()
    other[0, 0] = 0.1
    metadata = {"coordinates_sha256": audit.oracle_coordinate_sha256(other)}

    with pytest.raises(ValueError, match="metadata coordinates differ"):
        audit.validate_reference_metadata_coordinates(metadata, positions)


def test_fd_sample_rejects_swapped_valid_reference_metadata():
    positions = np.zeros((1, 3))
    steps = [2e-5, 1e-5, 5e-6]
    comparisons = []
    for flat in range(3):
        stencils = []
        for step in steps:
            samples = []
            for multiplier in (-2.0, -1.0, 1.0, 2.0):
                displaced = positions.copy()
                displaced[0, flat] += multiplier * step
                metadata_positions = displaced.copy()
                if flat == 0 and step == steps[0] and multiplier == -2.0:
                    metadata_positions[0, 1] += 0.2
                samples.append(
                    {
                        "multiplier": multiplier,
                        "positions_angstrom": displaced.tolist(),
                        "coordinate_sha256": audit.coordinate_sha256(displaced),
                        "energy": 0.0,
                        "reference_geometry_metadata": {
                            "coordinates_sha256": audit.oracle_coordinate_sha256(
                                metadata_positions
                            )
                        },
                    }
                )
            stencils.append({"step_angstrom": step, "samples": samples})
        comparisons.append({"atom_index": 0, "axis": flat, "stencils": stencils})

    class FakeRunner:
        @staticmethod
        def _validate_reference_geometry_metadata(*args):
            return True

    with pytest.raises(ValueError, match="metadata coordinates differ"):
        audit._validate_samples(
            comparisons,
            positions,
            steps,
            energy_key="energy",
            runner=FakeRunner(),
            reference_order=128,
            epsabs=1e-12,
            epsrel=1e-12,
        )


@pytest.mark.parametrize("forces", [np.zeros(9), np.full((3, 3), np.nan)])
def test_analytic_force_validation_requires_finite_three_by_three(forces):
    with pytest.raises(ValueError, match=r"finite shape \(3, 3\)"):
        audit.validated_force_array("analytic", forces)


def test_failed_row_derivation_retains_exact_15_row_denominator():
    entries = [{"row_id": f"row-{index:02d}"} for index in range(15)]
    records = [{"case_identity": entry} for entry in entries]

    class FakeRunner:
        @staticmethod
        def verify_seal(record, label):
            return None

        @staticmethod
        def serialize_exception(exc):
            return {"type": type(exc).__name__, "message": str(exc)}

    derived = audit.derive_terminal_records(records, entries, {}, FakeRunner())

    assert len(derived) == 15
    assert [row["row_id"] for row in derived] == [entry["row_id"] for entry in entries]
    assert all(row["solvent_passed"] is False for row in derived)


def test_failed_row_derivation_does_not_swallow_identity_mismatch():
    entries = [{"row_id": "expected"}]
    records = [{"case_identity": {"row_id": "different"}}]

    with pytest.raises(ValueError, match="identity differs"):
        audit.derive_terminal_records(records, entries, {}, object())


def test_post_phase_snapshot_rejects_source_or_input_drift():
    preregistration = {
        "source_before": {"helper.py": "a" * 64},
        "campaign_input_sha256": {"topology": "b" * 64},
    }

    with pytest.raises(ValueError, match="source changed"):
        audit.assert_post_phase_snapshot_stable(
            preregistration,
            {"helper.py": "c" * 64},
            {"topology": "b" * 64},
        )


def test_resealed_reference_metadata_rejects_negative_nonpolar_error():
    runner = audit._runner()
    oracle = audit._oracle()
    positions = np.array(
        [[0.011, 0.404, 0.0], [0.777, -0.223, 0.0], [-0.788, -0.181, 0.0]]
    )
    geometry = oracle.prepare_gaussian_reference_geometry(
        positions,
        [-0.784666666667, 0.392333333333, 0.392333333333],
        [1.88, 1.04, 1.04],
        [1.82, 0.3019, 0.3019],
        [0.093, 0.0047, 0.0047],
        azimuth_orders=(64,),
    )
    metadata = runner._reference_geometry_metadata(geometry)
    payload = {key: value for key, value in metadata.items() if key != "content_sha256"}
    payload["nonpolar_diagnostics"] = dict(payload["nonpolar_diagnostics"])
    payload["nonpolar_diagnostics"]["cavity_mixed_vector_quad_error"] = -1e-12
    resealed = runner.seal(payload)

    assert runner._validate_reference_geometry_metadata(resealed, (64,), 1e-12, 1e-12)
    with pytest.raises(ValueError, match="nonnegative"):
        audit.validate_reference_error_semantics(resealed)


def test_combined_base_ledger_rejects_energy_identity_mismatch():
    runner = audit._runner()
    positions = np.zeros((3, 3))
    combined = {
        "base_positions_angstrom": positions.tolist(),
        "base_coordinate_sha256": audit.coordinate_sha256(positions),
        "analytic_combined_energy_hartree": 0.5,
        "analytic_solvation_ledger": {
            "combined_energy_hartree": 0.4,
            "gas_energy_hartree": 0.3,
            "energy_hartree": 0.1,
            "components_hartree": {
                "polar": 0.05,
                "cavity": 0.03,
                "dispersion": 0.02,
            },
        },
    }

    with pytest.raises(ValueError, match="analytic combined energy"):
        audit.validate_combined_base_ledger(combined, positions, runner)
