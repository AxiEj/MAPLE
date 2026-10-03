from __future__ import annotations

from copy import deepcopy
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

import cha_gaussian_workflow_validation as validation  # pyright: ignore[reportMissingImports]

POSITIONS = np.array(
    [[0.01, 0.39, 0.0], [0.77, -0.19, 0.0], [-0.78, -0.15, 0.0]],
    dtype=np.float64,
)
MASSES = np.array([15.999, 1.008, 1.008], dtype=np.float64)


def _protocol():
    return {
        "atomic_numbers": [8, 1, 1],
        "masses_amu": MASSES.tolist(),
        "frequency_options": {
            "method": "mw",
            "temperature": 298.15,
            "pressure_kpa": 101.325,
            "ilowfreq": 2,
            "verbosity": 1,
            "treat_imag_as_real": False,
            "diagonalization_device": "cpu",
        },
        "prfo_options": {
            "method": "prfo",
            "project_rigid_modes": True,
            "max_iter": 1,
            "recalc": 1,
            "hessian_update": "bofill",
        },
        "dimer_options": {
            "method": "dimer",
            "use_hvp": True,
            "n_init": "given",
            "remove_rigid": True,
            "use_mass_weight": False,
            "max_iter": 1,
            "rot_max_iter": 1,
            "save_traj": True,
        },
        "thresholds": {
            "fresh_force_max_hartree_per_angstrom": 3e-4,
            "freq_rigid_abs_cm1": 1.0,
            "freq_internal_min_cm1": 10.0,
            "material_negative_cm1": -10.0,
        },
        "workflow_budgets": {
            "freq_seconds": 300,
            "prfo_seconds": 300,
            "dimer_seconds": 300,
            "dimer_composed_hvp_calls": 8,
        },
        "profile_id": "gaussian-cha-r6-derivative-v2-numerical-profile-20261001.2-direct-complement",
        "checkpoint_sha256": "c" * 64,
        "topology_sha256": "t" * 64,
        "gas_source_sha256": "g" * 64,
        "solvent_source_sha256": "s" * 64,
        "workflow_sources": {
            "gas_source_sha256": "g" * 64,
            "solvent_source_sha256": "s" * 64,
            "dimer_source_sha256": "d" * 64,
        },
        "expected_outcomes": {"prfo": "not_converged", "dimer": "not_converged"},
        "dimer_call_sites": {
            "initial": {"function": "run", "line": 566, "source_sha256": "d" * 64},
            "rotation": {
                "function": "_rotate_minimize_kappa",
                "line": 521,
                "source_sha256": "d" * 64,
            },
            "iteration": {"function": "run", "line": 607, "source_sha256": "d" * 64},
            "final": {"function": "run", "line": 721, "source_sha256": "d" * 64},
        },
        "dimer_replay_call_site": {
            "function": "run_dimer_workflow",
            "line": 1331,
            "source_sha256": "r" * 64,
        },
    }


def _case():
    n_given = np.arange(1.0, 10.0)
    return {
        "row_id": "center-00-sigma-00",
        "terminal_positions_angstrom": POSITIONS.tolist(),
        "dimer_n_given": n_given.tolist(),
    }


def _positive_hessian():
    projector = validation.rigid_projector(POSITIONS, MASSES)
    internal = np.linalg.eigh(projector)[1][:, -3:]
    eigenvalues = (
        np.array([100.0, 200.0, 300.0]) / validation.FREQUENCY_CONVERSION
    ) ** 2
    hmw = internal @ np.diag(eigenvalues) @ internal.T
    sqrt_mass = np.sqrt(np.repeat(MASSES, 3))
    return hmw * sqrt_mass[:, None] * sqrt_mass[None, :]


def _center(hessian=None):
    if hessian is None:
        hessian = _positive_hessian()
    return {
        "energy_hartree": -76.0,
        "forces_hartree_per_angstrom": np.zeros((3, 3)).tolist(),
        "hessian_hartree_per_angstrom2": np.asarray(hessian).tolist(),
    }


def _identity(record):
    record.update(
        {
            "atomic_numbers": [8, 1, 1],
            "masses_amu": MASSES.tolist(),
            "initial_positions_angstrom": POSITIONS.tolist(),
        }
    )
    return record


def _frequency_record(center):
    analysis = validation.independent_frequency_analysis(
        center["hessian_hartree_per_angstrom2"], POSITIONS, MASSES
    )
    return _identity(
        {
            "status": "EXECUTED",
            "requested_options": deepcopy(_protocol()["frequency_options"]),
            "effective_options": deepcopy(_protocol()["frequency_options"]),
            "elapsed_seconds": 1.0,
            "fresh_forces_hartree_per_angstrom": np.zeros((3, 3)).tolist(),
            "hessian_hartree_per_angstrom2": deepcopy(
                center["hessian_hartree_per_angstrom2"]
            ),
            "frequencies_cm1": analysis["frequencies_cm1"].tolist(),
            "modes_cart": analysis["modes_cart"].tolist(),
            "raw_output": (
                "Frequency analysis summary:\n"
                "  Zero frequencies (|nu| < 5.0 cm-1): 6\n"
                "  Imaginary frequencies (nu < -5.0 cm-1): 0\n"
                "  Real frequencies (nu > 5.0 cm-1): 3\n"
                "Frequency analysis completed\n"
            ),
        }
    )


def _prfo_record(center):
    analysis = validation.independent_frequency_analysis(
        center["hessian_hartree_per_angstrom2"], POSITIONS, MASSES
    )
    frequencies = analysis["frequencies_cm1"].tolist()
    return _identity(
        {
            "status": "INTERFACE_EXECUTED_NOT_TS",
            "requested_options": deepcopy(_protocol()["prfo_options"]),
            "effective_options": {
                key: value
                for key, value in _protocol()["prfo_options"].items()
                if key != "method"
            },
            "elapsed_seconds": 1.0,
            "raw_return_positions_angstrom": POSITIONS.tolist(),
            "raw_output": "Maximum iterations reached\n",
            "raw_algorithm_converged": False,
            "true_ts_claim": False,
            "initial_composed_hessian_hartree_per_angstrom2": deepcopy(
                center["hessian_hartree_per_angstrom2"]
            ),
            "final_composed_hessian_hartree_per_angstrom2": deepcopy(
                center["hessian_hartree_per_angstrom2"]
            ),
            "initial_frequencies_cm1": frequencies,
            "final_frequencies_cm1": frequencies,
            "initial_material_negative_count": 0,
            "final_material_negative_count": 0,
        }
    )


def _dimer_record(center):
    projected = validation.project_and_normalize_direction(
        _case()["dimer_n_given"], POSITIONS
    )
    hessian = np.asarray(center["hessian_hartree_per_angstrom2"])
    hvp = hessian @ projected
    kappa = float(projected @ hvp)
    requested = {**_protocol()["dimer_options"], "n_given": _case()["dimer_n_given"]}
    source_identity = {
        "profile_id": _protocol()["profile_id"],
        "checkpoint_sha256": _protocol()["checkpoint_sha256"],
        "topology_sha256": _protocol()["topology_sha256"],
        "gas_source_sha256": _protocol()["gas_source_sha256"],
        "solvent_source_sha256": _protocol()["solvent_source_sha256"],
    }
    evaluations = []
    for ordinal, kind in enumerate(("initial", "rotation", "iteration", "final")):
        evaluations.append(
            {
                "evaluation_id": f"hvp-{ordinal}",
                "ordinal": ordinal,
                "call_kind": kind,
                "iteration": 0 if kind == "initial" else 1,
                "positions_angstrom": POSITIONS.tolist(),
                "coordinate_sha256": validation.coordinate_sha256(POSITIONS),
                "direction": projected.tolist(),
                "direction_sha256": validation.direction_sha256(projected),
                "hessian_vector_hartree_per_angstrom2": hvp.tolist(),
                "forces_hartree_per_angstrom": deepcopy(
                    center["forces_hartree_per_angstrom"]
                ),
                "energy_hartree": center["energy_hartree"],
                "status": "RETURNED",
                "source_identity": source_identity,
                "call_site": deepcopy(_protocol()["dimer_call_sites"][kind]),
            }
        )
    final = evaluations[-1]
    replay = {
        "evaluation_id": final["evaluation_id"],
        "comparison_kind": "same_implementation_replay",
        "positions_angstrom": final["positions_angstrom"],
        "coordinate_sha256": final["coordinate_sha256"],
        "direction": final["direction"],
        "direction_sha256": final["direction_sha256"],
        "hessian_vector_hartree_per_angstrom2": final[
            "hessian_vector_hartree_per_angstrom2"
        ],
        "forces_hartree_per_angstrom": final["forces_hartree_per_angstrom"],
        "energy_hartree": final["energy_hartree"],
        "fresh_forces_hartree_per_angstrom": final["forces_hartree_per_angstrom"],
        "fresh_energy_hartree": final["energy_hartree"],
        "source_identity": source_identity,
        "call_site": deepcopy(_protocol()["dimer_replay_call_site"]),
        "counters": {
            "direct_hvp_calls": 1,
            "gas_dense_hessian_calls": 0,
            "solvent_dense_hessian_calls": 0,
        },
    }
    return _identity(
        {
            "status": "INTERFACE_EXECUTED_NOT_TS",
            "requested_options": requested,
            "effective_options": {
                key: value for key, value in requested.items() if key != "method"
            },
            "actual_initial_direction": projected.tolist(),
            "elapsed_seconds": 1.0,
            "raw_return_positions_angstrom": POSITIONS.tolist(),
            "raw_output": f"Curvature (kappa): {kappa:.6f}\nMaximum iterations reached\n",
            "raw_algorithm_converged": False,
            "true_ts_claim": False,
            "counter_before": {
                "direct_hvp_calls": 0,
                "gas_dense_hessian_calls": 0,
                "solvent_dense_hessian_calls": 0,
            },
            "counter_after": {
                "direct_hvp_calls": 4,
                "gas_dense_hessian_calls": 0,
                "solvent_dense_hessian_calls": 0,
            },
            "derivative_mode": "hvp",
            "directional_units": {
                "energy": "hartree",
                "forces": "hartree/angstrom",
                "hvp": "hartree/angstrom^2",
                "direction": "dimensionless",
            },
            "prohibited_derivative_attempts": [],
            "printed_curvature_entries": [
                {
                    "iteration": 1,
                    "evaluation_id": "hvp-2",
                    "directional_evaluation_index": 2,
                    "printed_kappa": float(f"{kappa:.6f}"),
                }
            ],
            "directional_evaluations": evaluations,
            "directional_replays": [replay],
        }
    )


@pytest.mark.parametrize("phase", ["freq", "prfo", "dimer"])
def test_synthetic_raw_workflow_receipt_passes_independent_recomputation(phase):
    center = _center()
    record = {
        "freq": _frequency_record,
        "prfo": _prfo_record,
        "dimer": _dimer_record,
    }[
        phase
    ](center)

    result = validation.validate_workflow_receipt(
        record, phase, _case(), _protocol(), center
    )

    assert result["passed"] is True
    assert result["phase"] == phase


def test_frequency_rejects_fabricated_internal_frequency():
    center = _center()
    record = _frequency_record(center)
    record["frequencies_cm1"][-1] += 20.0

    result = validation.validate_workflow_receipt(
        record, "freq", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert "frequencies" in result["reason"].lower()


@pytest.mark.parametrize(
    "field,value",
    [
        ("atomic_numbers", [1, 8, 1]),
        ("masses_amu", [1.008, 15.999, 1.008]),
        ("initial_positions_angstrom", (POSITIONS + 0.1).tolist()),
    ],
)
def test_shared_identity_tampering_fails_closed(field, value):
    center = _center()
    record = _frequency_record(center)
    record[field] = value

    result = validation.validate_workflow_receipt(
        record, "freq", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert field.split("_")[0] in result["reason"]


def test_frequency_modes_must_be_mass_orthonormal_and_eigenvectors():
    center = _center()
    record = _frequency_record(center)
    record["modes_cart"][-1][0] += 0.1

    result = validation.validate_workflow_receipt(
        record, "freq", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert "modes" in result["reason"]


def test_prfo_recomputes_indices_and_rejects_false_normal_termination():
    center = _center()
    record = _prfo_record(center)
    record["raw_output"] = "Normal Termination\n"
    record["raw_algorithm_converged"] = True

    result = validation.validate_workflow_receipt(
        record, "prfo", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert result["recomputed"]["final_material_negative_count"] == 0
    assert "INTERFACE_UNEXPECTED_CONVERGENCE" in result["reason"]


def test_prfo_rejects_reported_index_or_effective_option_tampering():
    center = _center()
    record = _prfo_record(center)
    record["initial_material_negative_count"] = 1
    record["effective_options"]["project_rigid_modes"] = False

    result = validation.validate_workflow_receipt(
        record, "prfo", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert "options" in result["reason"] or "reported" in result["reason"]


def test_dimer_rejects_fabricated_hvp_energy_force_or_position():
    center = _center()
    record = _dimer_record(center)
    evaluation = record["directional_evaluations"][0]
    evaluation["hessian_vector_hartree_per_angstrom2"][0] += 1e-3
    evaluation["energy_hartree"] += 1e-3
    evaluation["positions_angstrom"][0][0] += 1e-3

    result = validation.validate_workflow_receipt(
        record, "dimer", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert "directional evaluation" in result["reason"]


def test_dimer_rejects_counter_prohibited_call_and_direction_tampering():
    center = _center()
    record = _dimer_record(center)
    record["actual_initial_direction"][0] += 0.1
    record["counter_after"]["direct_hvp_calls"] = 3
    record["prohibited_derivative_attempts"] = ["dense-hessian"]

    result = validation.validate_workflow_receipt(
        record, "dimer", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert "direction" in result["reason"] or "prohibited" in result["reason"]


def test_dimer_printed_curvature_is_only_a_six_decimal_representation_gate():
    center = _center()
    record = _dimer_record(center)
    record["printed_curvature_entries"][-1]["printed_kappa"] += 6e-7

    result = validation.validate_workflow_receipt(
        record, "dimer", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert "printed curvature" in result["reason"]


def test_dimer_rejects_unexpected_raw_convergence_in_all_minima_protocol():
    center = _center()
    record = _dimer_record(center)
    record["raw_output"] += "converged TS candidate\n"
    record["raw_algorithm_converged"] = True

    result = validation.validate_workflow_receipt(
        record, "dimer", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert "INTERFACE_UNEXPECTED_CONVERGENCE" in result["reason"]


def test_dimer_rejects_wrong_role_call_site_binding():
    center = _center()
    record = _dimer_record(center)
    record["directional_evaluations"][1]["call_site"]["function"] = "run"

    result = validation.validate_workflow_receipt(
        record, "dimer", _case(), _protocol(), center
    )

    assert result["passed"] is False
    assert "call-site" in result["reason"]


def test_dimer_final_shifted_call_uses_replay_not_initial_dense_hessian():
    center = _center()
    record = _dimer_record(center)
    final = record["directional_evaluations"][-1]
    shifted = POSITIONS.copy()
    shifted[0, 0] += 0.01
    final["positions_angstrom"] = shifted.tolist()
    final["coordinate_sha256"] = validation.coordinate_sha256(shifted)
    final["hessian_vector_hartree_per_angstrom2"][0] += 0.02
    replay = record["directional_replays"][0]
    replay["positions_angstrom"] = shifted.tolist()
    replay["coordinate_sha256"] = validation.coordinate_sha256(shifted)
    replay["hessian_vector_hartree_per_angstrom2"] = deepcopy(
        final["hessian_vector_hartree_per_angstrom2"]
    )
    record["raw_return_positions_angstrom"] = shifted.tolist()

    result = validation.validate_workflow_receipt(
        record, "dimer", _case(), _protocol(), center
    )

    assert result["passed"] is True
    assert result["recomputed"]["hvp_replay_count"] == 1
    assert result["recomputed"]["last_printed_kappa_hartree_per_angstrom2"] != (
        float(
            np.asarray(final["direction"])
            @ np.asarray(final["hessian_vector_hartree_per_angstrom2"])
        )
    )


def test_missing_required_raw_fields_and_unknown_phase_fail_closed():
    center = _center()
    record = _frequency_record(center)
    del record["masses_amu"]

    missing = validation.validate_workflow_receipt(
        record, "freq", _case(), _protocol(), center
    )
    unknown = validation.validate_workflow_receipt(
        record, "unknown", _case(), _protocol(), center
    )

    assert missing["passed"] is False
    assert unknown["passed"] is False


def test_actual_development_frequency_raw_passes_after_in_memory_identity_enrichment():
    path = Path(
        "/home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/benchmarks/"
        "route1-cha-freq-ts-v1-20261002/root-all15-development-smoke-v1/"
        "test_real_water_analytic_frequ0/development-smoke.json"
    )
    development = json.loads(path.read_text())
    case = development["case_identity"]
    record = deepcopy(development["frequency"])
    record.update(
        {
            "atomic_numbers": [8, 1, 1],
            "masses_amu": MASSES.tolist(),
            "initial_positions_angstrom": case["terminal_positions_angstrom"],
        }
    )
    center = {
        "energy_hartree": 0.0,
        "forces_hartree_per_angstrom": record["fresh_forces_hartree_per_angstrom"],
        "hessian_hartree_per_angstrom2": record["hessian_hartree_per_angstrom2"],
    }

    result = validation.validate_workflow_receipt(
        record, "freq", case, _protocol(), center
    )

    assert result["passed"] is True


def test_actual_development_prfo_rigid_zero_representation_is_accepted():
    path = Path(
        "/home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/benchmarks/"
        "route1-cha-freq-ts-v1-20261002/root-all15-development-smoke-v1/"
        "test_real_water_analytic_frequ0/development-smoke.json"
    )
    development = json.loads(path.read_text())
    case = development["case_identity"]
    record = deepcopy(development["prfo"])
    record.update(
        {
            "atomic_numbers": [8, 1, 1],
            "masses_amu": MASSES.tolist(),
            "initial_positions_angstrom": case["terminal_positions_angstrom"],
            "true_ts_claim": False,
        }
    )
    center = {
        "energy_hartree": 0.0,
        "forces_hartree_per_angstrom": np.zeros((3, 3)).tolist(),
        "hessian_hartree_per_angstrom2": development["frequency"][
            "hessian_hartree_per_angstrom2"
        ],
    }

    result = validation.validate_workflow_receipt(
        record, "prfo", case, _protocol(), center
    )

    assert result["passed"] is True
    assert result["recomputed"]["initial_material_negative_count"] == 0
    assert result["recomputed"]["final_material_negative_count"] == 0


def test_live_protocol_freezes_identity_negative_controls_and_workflow_budgets():
    protocol = json.loads(
        (BENCHMARK_DIR / "cha_gaussian_freq_ts_protocol.json").read_text()
    )

    assert protocol["approved_plan_sha256"] == (
        "2c872a6c3035b307098b7631434dfc8c88b6ace6bb91ca86d8361d1f15ade07c"
    )
    assert protocol["atomic_numbers"] == [8, 1, 1]
    assert protocol["masses_amu"] == [15.999, 1.008, 1.008]
    assert protocol["expected_outcomes"] == {
        "prfo": "not_converged",
        "dimer": "not_converged",
    }
    assert protocol["workflow_budgets"]["dimer_composed_hvp_calls"] == 8
    assert protocol["workflow_budgets"]["freq_seconds"] == 300
    assert protocol["workflow_budgets"]["prfo_seconds"] == 300
    assert protocol["workflow_budgets"]["dimer_seconds"] == 300
