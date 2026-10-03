"""Independent NumPy validators for Gaussian-CHA FREQ/TS workflow receipts.

No production dispatcher, optimizer, or frequency implementation is imported.
All reported spectra, indices, HVPs, counters, and convergence labels are
recomputed from raw receipt arrays and logs.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

HARTREE_J = 4.3597447222071e-18
AMU_KG = 1.66053906660e-27
ANGSTROM2_M2 = 1e-20
SPEED_OF_LIGHT_M_S = 2.99792458e8
FREQUENCY_CONVERSION = math.sqrt(HARTREE_J / (AMU_KG * ANGSTROM2_M2)) / (
    2.0 * math.pi * SPEED_OF_LIGHT_M_S * 100.0
)
IDENTITY_RELATIVE_TOLERANCE = 1e-12


def _array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _relative_max_error(first: Any, second: Any) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("identity comparison arrays differ in shape or finiteness")
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return float(np.max(np.abs(left - right))) / scale


def _rigid_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    count = len(masses)
    basis = np.zeros((3 * count, 6), dtype=np.float64)
    center = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    for atom in range(count):
        root_mass = math.sqrt(float(masses[atom]))
        basis[3 * atom : 3 * atom + 3, :3] = root_mass * np.eye(3)
        x, y, z = positions[atom] - center
        basis[3 * atom : 3 * atom + 3, 3] = root_mass * np.array([0.0, z, -y])
        basis[3 * atom : 3 * atom + 3, 4] = root_mass * np.array([-z, 0.0, x])
        basis[3 * atom : 3 * atom + 3, 5] = root_mass * np.array([y, -x, 0.0])
    left, singular, _ = np.linalg.svd(basis, full_matrices=False)
    tolerance = (
        0.0
        if singular.size == 0
        else np.finfo(np.float64).eps * max(basis.shape) * singular[0]
    )
    return left[:, singular > tolerance]


def rigid_projector(positions: Any, masses_amu: Any) -> np.ndarray:
    positions_array = _array("positions", positions, (3, 3))
    masses = _array("masses", masses_amu, (3,))
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")
    basis = _rigid_basis(positions_array, masses)
    return np.eye(9) - basis @ basis.T


def independent_frequency_analysis(
    hessian_hartree_per_angstrom2: Any,
    positions_angstrom: Any,
    masses_amu: Any,
) -> dict[str, np.ndarray]:
    hessian = _array("frequency Hessian", hessian_hartree_per_angstrom2, (9, 9))
    positions = _array("frequency positions", positions_angstrom, (3, 3))
    masses = _array("frequency masses", masses_amu, (3,))
    if np.any(masses <= 0.0):
        raise ValueError("frequency masses must be positive")
    inverse_root_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    mass_weighted = hessian * inverse_root_mass[:, None] * inverse_root_mass[None, :]
    projector = rigid_projector(positions, masses)
    projected = projector @ mass_weighted @ projector
    projected = 0.5 * (projected + projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    frequencies = (
        np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * FREQUENCY_CONVERSION
    )
    modes = eigenvectors.T * inverse_root_mass[None, :]
    mass_diagonal = np.repeat(masses, 3)
    for index in range(9):
        norm = math.sqrt(float(np.sum(modes[index] ** 2 * mass_diagonal)))
        if norm > 1e-10:
            modes[index] /= norm
    zero = np.where(np.abs(frequencies) < 5.0)[0]
    imaginary = np.where(frequencies < -5.0)[0]
    real = np.where(frequencies > 5.0)[0]
    order = np.concatenate(
        (
            zero[np.argsort(np.abs(frequencies[zero]))],
            imaginary[np.argsort(frequencies[imaginary])],
            real[np.argsort(frequencies[real])],
        )
    )
    return {
        "frequencies_cm1": frequencies[order],
        "modes_cart": modes[order],
        "eigenvalues": eigenvalues[order],
        "projected_mass_weighted_hessian": projected,
    }


def _validate_common(record, case, protocol):
    atomic_numbers = record["atomic_numbers"]
    if atomic_numbers != protocol["atomic_numbers"] or atomic_numbers != [8, 1, 1]:
        raise ValueError("atomic numbers differ from protocol")
    masses = _array("masses_amu", record["masses_amu"], (3,))
    if not np.array_equal(masses, np.asarray(protocol["masses_amu"], dtype=np.float64)):
        raise ValueError("masses differ from protocol")
    positions = _array(
        "initial positions", record["initial_positions_angstrom"], (3, 3)
    )
    if not np.array_equal(
        positions, np.asarray(case["terminal_positions_angstrom"], dtype=np.float64)
    ):
        raise ValueError("initial positions differ from frozen case")
    return positions, masses


def _validate_options(record, phase, case, protocol):
    if phase == "freq":
        expected_requested = protocol["frequency_options"]
        expected_effective = expected_requested
    elif phase == "prfo":
        expected_requested = protocol["prfo_options"]
        expected_effective = {
            key: value for key, value in expected_requested.items() if key != "method"
        }
    else:
        expected_requested = {
            **protocol["dimer_options"],
            "n_given": case["dimer_n_given"],
        }
        expected_effective = {
            key: value for key, value in expected_requested.items() if key != "method"
        }
    if (
        record["requested_options"] != expected_requested
        or record["effective_options"] != expected_effective
    ):
        raise ValueError(f"{phase} requested/effective options differ")
    elapsed = float(record["elapsed_seconds"])
    limit = float(protocol["workflow_budgets"][f"{phase}_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0.0 or elapsed > limit:
        raise ValueError(f"{phase} elapsed time exceeds the frozen budget")


def _validate_modes(stored_modes, analysis, masses):
    modes = _array("frequency modes", stored_modes, (9, 9))
    mass_diagonal = np.repeat(masses, 3)
    orthonormality = modes @ np.diag(mass_diagonal) @ modes.T
    if np.max(np.abs(orthonormality - np.eye(9))) > 1e-10:
        raise ValueError("frequency modes are not mass-orthonormal")
    projected = analysis["projected_mass_weighted_hessian"]
    eigenvalues = analysis["eigenvalues"]
    for index in range(9):
        mass_weighted_vector = np.sqrt(mass_diagonal) * modes[index]
        residual = (
            projected @ mass_weighted_vector - eigenvalues[index] * mass_weighted_vector
        )
        scale = max(1.0, float(np.max(np.abs(projected))))
        if np.max(np.abs(residual)) > 1e-10 * scale:
            raise ValueError("frequency modes fail the projected eigen residual")


def _validate_frequency(record, case, protocol, center, positions, masses):
    if record["status"] != "EXECUTED":
        raise ValueError("frequency workflow status differs")
    center_hessian = _array(
        "center composed Hessian", center["hessian_hartree_per_angstrom2"], (9, 9)
    )
    raw_hessian = _array(
        "frequency raw Hessian", record["hessian_hartree_per_angstrom2"], (9, 9)
    )
    if _relative_max_error(raw_hessian, center_hessian) > IDENTITY_RELATIVE_TOLERANCE:
        raise ValueError("frequency Hessian differs from analytic center")
    fresh_force = _array(
        "frequency fresh force", record["fresh_forces_hartree_per_angstrom"], (3, 3)
    )
    center_force = _array(
        "center composed force", center["forces_hartree_per_angstrom"], (3, 3)
    )
    if _relative_max_error(fresh_force, center_force) > IDENTITY_RELATIVE_TOLERANCE:
        raise ValueError("frequency fresh force differs from analytic center")
    if (
        np.max(np.abs(fresh_force))
        > protocol["thresholds"]["fresh_force_max_hartree_per_angstrom"]
    ):
        raise ValueError("frequency fresh force exceeds the frozen gate")
    analysis = independent_frequency_analysis(raw_hessian, positions, masses)
    stored_frequencies = _array("stored frequencies", record["frequencies_cm1"], (9,))
    stored_eigenvalues = (
        np.sign(stored_frequencies)
        * (np.abs(stored_frequencies) / FREQUENCY_CONVERSION) ** 2
    )
    if _relative_max_error(stored_eigenvalues, analysis["eigenvalues"]) > (
        IDENTITY_RELATIVE_TOLERANCE
    ):
        raise ValueError("stored frequencies differ from raw Hessian eigenvalues")
    _validate_modes(record["modes_cart"], analysis, masses)
    rigid = analysis["frequencies_cm1"][:6]
    internal = analysis["frequencies_cm1"][6:]
    rigid_pass = bool(
        np.max(np.abs(rigid)) <= protocol["thresholds"]["freq_rigid_abs_cm1"]
    )
    internal_pass = bool(
        len(internal) == 3
        and np.min(internal) > protocol["thresholds"]["freq_internal_min_cm1"]
    )
    raw_output = record["raw_output"]
    counts = {
        name.lower(): int(value)
        for name, value in re.findall(
            r"(Zero|Imaginary|Real) frequencies[^:]*:\s*(\d+)", raw_output
        )
    }
    if "Frequency analysis completed" not in raw_output or counts != {
        "zero": 6,
        "imaginary": 0,
        "real": 3,
    }:
        raise ValueError("frequency raw output completion/count summary differs")
    if not rigid_pass or not internal_pass:
        raise ValueError("frequency projected rigid/internal spectrum fails gates")
    return {
        "frequencies_cm1": analysis["frequencies_cm1"].tolist(),
        "rigid_max_abs_cm1": float(np.max(np.abs(rigid))),
        "internal_min_cm1": float(np.min(internal)),
        "rigid_passed": rigid_pass,
        "internal_passed": internal_pass,
    }


def _material_negative_count(frequencies, threshold):
    return int(np.sum(np.asarray(frequencies, dtype=np.float64) < float(threshold)))


def _validate_prfo(record, case, protocol, center, positions, masses):
    if (
        record["status"] != "INTERFACE_EXECUTED_NOT_TS"
        or record.get("true_ts_claim") is not False
    ):
        raise ValueError("PRFO status/true-TS claim differs")
    initial_hessian = _array(
        "PRFO initial Hessian",
        record["initial_composed_hessian_hartree_per_angstrom2"],
        (9, 9),
    )
    center_hessian = _array(
        "center composed Hessian", center["hessian_hartree_per_angstrom2"], (9, 9)
    )
    if (
        _relative_max_error(initial_hessian, center_hessian)
        > IDENTITY_RELATIVE_TOLERANCE
    ):
        raise ValueError("PRFO initial Hessian differs from analytic center")
    final_positions = _array(
        "PRFO final positions", record["raw_return_positions_angstrom"], (3, 3)
    )
    final_hessian = _array(
        "PRFO final Hessian",
        record["final_composed_hessian_hartree_per_angstrom2"],
        (9, 9),
    )
    initial = independent_frequency_analysis(initial_hessian, positions, masses)
    final = independent_frequency_analysis(final_hessian, final_positions, masses)
    for field, derived in (
        ("initial_frequencies_cm1", initial["eigenvalues"]),
        ("final_frequencies_cm1", final["eigenvalues"]),
    ):
        stored_frequencies = _array(f"PRFO {field}", record[field], (9,))
        stored_eigenvalues = (
            np.sign(stored_frequencies)
            * (np.abs(stored_frequencies) / FREQUENCY_CONVERSION) ** 2
        )
        if (
            _relative_max_error(stored_eigenvalues, derived)
            > IDENTITY_RELATIVE_TOLERANCE
        ):
            raise ValueError(f"PRFO reported {field} differs from raw Hessian")
    threshold = protocol["thresholds"]["material_negative_cm1"]
    initial_count = _material_negative_count(initial["frequencies_cm1"], threshold)
    final_count = _material_negative_count(final["frequencies_cm1"], threshold)
    if (
        int(record["initial_material_negative_count"]) != initial_count
        or int(record["final_material_negative_count"]) != final_count
    ):
        raise ValueError("PRFO reported projected index differs from raw Hessian")
    normal = "Normal Termination" in record["raw_output"]
    if bool(record["raw_algorithm_converged"]) != normal:
        raise ValueError("PRFO raw convergence flag differs from raw log")
    if protocol["expected_outcomes"]["prfo"] != "not_converged":
        raise ValueError("PRFO expected outcome is not the frozen negative control")
    initial_passed = initial_count == 0
    convergence_passed = not normal
    return {
        "initial_material_negative_count": initial_count,
        "final_material_negative_count": final_count,
        "raw_algorithm_converged": normal,
        "scientific_ts_converged": bool(normal and final_count == 1),
        "final_positions_angstrom": final_positions.tolist(),
        "passed": bool(initial_passed and convergence_passed),
        "failure_reason": (
            None
            if initial_passed and convergence_passed
            else (
                "PRFO initial projected index is not zero"
                if not initial_passed
                else "INTERFACE_UNEXPECTED_CONVERGENCE: PRFO raw log claimed convergence"
            )
        ),
    }


def _remove_rigid_unweighted(direction, positions):
    vector = np.asarray(direction, dtype=np.float64).reshape(3, 3).copy()
    vector -= vector.mean(axis=0, keepdims=True)
    centered = positions - positions.mean(axis=0)
    for axis in np.eye(3):
        mode = np.cross(centered, axis)
        coefficient = float(np.sum(vector * mode)) / (
            float(np.sum(mode * mode)) + 1e-20
        )
        vector -= coefficient * mode
    return vector.reshape(-1)


def project_and_normalize_direction(
    direction: Any, positions_angstrom: Any
) -> np.ndarray:
    positions = _array("Dimer positions", positions_angstrom, (3, 3))
    vector = _array("Dimer n_given", direction, (9,))
    vector = _remove_rigid_unweighted(vector, positions)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-20:
        raise ValueError("Dimer initial direction is zero after rigid projection")
    return vector / norm


def coordinate_sha256(positions):
    import hashlib

    array = _array("Dimer coordinate hash positions", positions, (3, 3))
    return hashlib.sha256(
        array.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()


def direction_sha256(direction):
    import hashlib

    array = _array("Dimer direction hash vector", direction, (9,))
    return hashlib.sha256(
        array.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()


def _validate_dimer(record, case, protocol, center, positions, masses):
    del masses
    if (
        record["status"] != "INTERFACE_EXECUTED_NOT_TS"
        or record.get("true_ts_claim") is not False
    ):
        raise ValueError("Dimer status/true-TS claim differs")
    if record["derivative_mode"] != "hvp" or record["prohibited_derivative_attempts"]:
        raise ValueError("Dimer used a prohibited derivative path")
    if record["directional_units"] != {
        "energy": "hartree",
        "forces": "hartree/angstrom",
        "hvp": "hartree/angstrom^2",
        "direction": "dimensionless",
    }:
        raise ValueError("Dimer directional units differ")
    lowered_output = record["raw_output"].lower()
    if any(
        marker in lowered_output
        for marker in (
            "finite-difference curvature",
            "numerical curvature",
            "dense hessian",
        )
    ):
        raise ValueError("Dimer raw log contains a prohibited derivative marker")
    _array("Dimer returned positions", record["raw_return_positions_angstrom"], (3, 3))
    expected_initial = project_and_normalize_direction(case["dimer_n_given"], positions)
    actual_initial = _array(
        "Dimer actual initial direction", record["actual_initial_direction"], (9,)
    )
    if np.max(np.abs(actual_initial - expected_initial)) > 2e-14:
        raise ValueError("Dimer actual initial direction differs from frozen n_given")
    center_hessian = _array(
        "center composed Hessian", center["hessian_hartree_per_angstrom2"], (9, 9)
    )
    center_force = _array(
        "center composed force", center["forces_hartree_per_angstrom"], (3, 3)
    )
    center_energy = float(center["energy_hartree"])
    evaluations = record["directional_evaluations"]
    maximum = int(protocol["workflow_budgets"]["dimer_composed_hvp_calls"])
    if not isinstance(evaluations, list) or not 1 <= len(evaluations) <= maximum:
        raise ValueError("Dimer directional evaluation count differs")
    expected_roles = ["initial", "rotation", "iteration", "final"]
    if [evaluation.get("call_kind") for evaluation in evaluations] != expected_roles:
        raise ValueError("Dimer directional evaluation role inventory differs")
    expected_topology = (
        protocol["topology_sha256"]
        if "topology_sha256" in protocol
        else protocol["topology_content_sha256"]
    )
    expected_source = {
        "profile_id": protocol["profile_id"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "topology_sha256": expected_topology,
        "gas_source_sha256": protocol["workflow_sources"]["gas_source_sha256"],
        "solvent_source_sha256": protocol["workflow_sources"]["solvent_source_sha256"],
    }
    kappas = []
    for index, evaluation in enumerate(evaluations):
        if (
            evaluation.get("evaluation_id") != f"hvp-{index}"
            or int(evaluation.get("ordinal")) != index
            or int(evaluation.get("iteration")) != (0 if index == 0 else 1)
            or evaluation.get("status") != "RETURNED"
        ):
            raise ValueError("Dimer directional evaluation identity differs")
        observed_positions = _array(
            "Dimer HVP positions", evaluation["positions_angstrom"], (3, 3)
        )
        if coordinate_sha256(observed_positions) != evaluation["coordinate_sha256"]:
            raise ValueError("Dimer directional evaluation coordinate hash differs")
        direction = _array("Dimer HVP direction", evaluation["direction"], (9,))
        if (
            abs(float(np.linalg.norm(direction)) - 1.0) > 2e-14
            or direction_sha256(direction) != evaluation["direction_sha256"]
        ):
            raise ValueError("Dimer HVP direction normalization/hash differs")
        observed_hvp = _array(
            "Dimer returned HVP",
            evaluation["hessian_vector_hartree_per_angstrom2"],
            (9,),
        )
        observed_force = _array(
            "Dimer returned force", evaluation["forces_hartree_per_angstrom"], (3, 3)
        )
        observed_energy = float(evaluation["energy_hartree"])
        if not math.isfinite(observed_energy):
            raise ValueError("Dimer returned energy is nonfinite")
        if evaluation["source_identity"] != expected_source:
            raise ValueError("Dimer directional source identity differs")
        call_site = evaluation["call_site"]
        if call_site != protocol["dimer_call_sites"][evaluation["call_kind"]]:
            raise ValueError("Dimer directional call-site identity differs")
        if index < 3:
            if not np.array_equal(observed_positions, positions):
                raise ValueError(
                    "Dimer initial/rotation/iteration call left initial center"
                )
            if (
                _relative_max_error(observed_hvp, center_hessian @ direction)
                > IDENTITY_RELATIVE_TOLERANCE
                or _relative_max_error(observed_force, center_force)
                > IDENTITY_RELATIVE_TOLERANCE
                or not math.isclose(
                    observed_energy,
                    center_energy,
                    rel_tol=IDENTITY_RELATIVE_TOLERANCE,
                    abs_tol=IDENTITY_RELATIVE_TOLERANCE,
                )
            ):
                raise ValueError("Dimer initial-center directional evaluation differs")
        kappas.append(float(direction @ observed_hvp))
    replays = record["directional_replays"]
    if len(replays) != 1:
        raise ValueError("Dimer final replay inventory differs")
    replay = replays[0]
    final = evaluations[-1]
    if (
        replay.get("evaluation_id") != final["evaluation_id"]
        or replay.get("comparison_kind") != "same_implementation_replay"
        or replay.get("positions_angstrom") != final["positions_angstrom"]
        or replay.get("coordinate_sha256") != final["coordinate_sha256"]
        or replay.get("direction") != final["direction"]
        or replay.get("direction_sha256") != final["direction_sha256"]
        or replay.get("source_identity") != expected_source
        or replay.get("call_site") != protocol["dimer_replay_call_site"]
    ):
        raise ValueError("Dimer final replay identity differs")
    returned_positions = _array(
        "Dimer returned positions", record["raw_return_positions_angstrom"], (3, 3)
    )
    if (
        not np.array_equal(returned_positions, np.asarray(final["positions_angstrom"]))
        or coordinate_sha256(returned_positions) != final["coordinate_sha256"]
    ):
        raise ValueError("Dimer returned positions differ from final evaluation/replay")
    for replay_field, final_field in (
        (
            "hessian_vector_hartree_per_angstrom2",
            "hessian_vector_hartree_per_angstrom2",
        ),
        ("forces_hartree_per_angstrom", "forces_hartree_per_angstrom"),
    ):
        if _relative_max_error(replay[replay_field], final[final_field]) > (
            IDENTITY_RELATIVE_TOLERANCE
        ):
            raise ValueError("Dimer final same-implementation replay differs")
    if (
        not math.isclose(
            float(replay["energy_hartree"]),
            float(final["energy_hartree"]),
            rel_tol=IDENTITY_RELATIVE_TOLERANCE,
            abs_tol=IDENTITY_RELATIVE_TOLERANCE,
        )
        or _relative_max_error(
            replay["fresh_forces_hartree_per_angstrom"],
            replay["forces_hartree_per_angstrom"],
        )
        > IDENTITY_RELATIVE_TOLERANCE
        or not math.isclose(
            float(replay["fresh_energy_hartree"]),
            float(replay["energy_hartree"]),
            rel_tol=IDENTITY_RELATIVE_TOLERANCE,
            abs_tol=IDENTITY_RELATIVE_TOLERANCE,
        )
    ):
        raise ValueError("Dimer final replay fresh E/F differs")
    replay_counters = replay["counters"]
    expected_counter_keys = {
        "direct_hvp_calls",
        "gas_dense_hessian_calls",
        "solvent_dense_hessian_calls",
    }
    if set(replay_counters) != expected_counter_keys or replay_counters != {
        "direct_hvp_calls": 1,
        "gas_dense_hessian_calls": 0,
        "solvent_dense_hessian_calls": 0,
    }:
        raise ValueError("Dimer final replay counters differ")
    before = record["counter_before"]
    after = record["counter_after"]
    if set(before) != expected_counter_keys or set(after) != expected_counter_keys:
        raise ValueError("Dimer counter schema differs")
    if (
        int(after["direct_hvp_calls"]) - int(before["direct_hvp_calls"])
        != len(evaluations)
        or int(after["gas_dense_hessian_calls"])
        - int(before["gas_dense_hessian_calls"])
        != 0
        or int(after["solvent_dense_hessian_calls"])
        - int(before["solvent_dense_hessian_calls"])
        != 0
        or len(evaluations) + len(replays) > maximum
    ):
        raise ValueError("Dimer HVP/dense counter delta differs from evaluations")
    printed = [
        float(value)
        for value in re.findall(
            r"Curvature \(kappa\):\s+([-+0-9.eE]+)", record["raw_output"]
        )
    ]
    entries = record["printed_curvature_entries"]
    if not printed or len(entries) != len(printed):
        raise ValueError("Dimer printed curvature inventory differs")
    mapped_kappas = []
    for printed_value, entry in zip(printed, entries, strict=True):
        evaluation_index = int(entry["directional_evaluation_index"])
        if (
            int(entry["iteration"]) != 1
            or not 0 <= evaluation_index < len(evaluations)
            or entry["evaluation_id"] != evaluations[evaluation_index]["evaluation_id"]
            or evaluations[evaluation_index]["call_kind"] != "iteration"
        ):
            raise ValueError("Dimer printed curvature evaluation mapping differs")
        mapped = kappas[evaluation_index]
        mapped_kappas.append(mapped)
        if (
            abs(float(entry["printed_kappa"]) - printed_value) > 5e-7
            or abs(printed_value - mapped) > 5e-7
        ):
            raise ValueError(
                "Dimer printed curvature differs beyond six-decimal rounding"
            )
    converged_log = "converged TS candidate" in record["raw_output"]
    if bool(record["raw_algorithm_converged"]) != converged_log:
        raise ValueError("Dimer convergence flag differs from raw log")
    if protocol["expected_outcomes"]["dimer"] != "not_converged":
        raise ValueError("Dimer expected outcome is not the frozen negative control")
    convergence_passed = not converged_log
    return {
        "hvp_algorithm_call_count": len(evaluations),
        "hvp_replay_count": len(replays),
        "last_printed_kappa_hartree_per_angstrom2": mapped_kappas[-1],
        "raw_algorithm_converged": converged_log,
        "scientific_ts_converged": bool(converged_log and mapped_kappas[-1] < 0.0),
        "passed": convergence_passed,
        "failure_reason": (
            None
            if convergence_passed
            else "INTERFACE_UNEXPECTED_CONVERGENCE: Dimer raw log claimed convergence"
        ),
    }


def validate_workflow_receipt(record, phase, case, protocol, center):
    try:
        if phase not in {"freq", "prfo", "dimer"}:
            raise ValueError("unknown workflow phase")
        positions, masses = _validate_common(record, case, protocol)
        _validate_options(record, phase, case, protocol)
        if phase == "freq":
            recomputed = _validate_frequency(
                record, case, protocol, center, positions, masses
            )
        elif phase == "prfo":
            recomputed = _validate_prfo(
                record, case, protocol, center, positions, masses
            )
        else:
            recomputed = _validate_dimer(
                record, case, protocol, center, positions, masses
            )
        phase_passed = bool(recomputed.get("passed", True))
        return {
            "phase": phase,
            "row_id": case.get("row_id"),
            "passed": phase_passed,
            "reason": recomputed.get("failure_reason"),
            "recomputed": recomputed,
            "true_ts_validated": False,
        }
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        return {
            "phase": phase,
            "row_id": case.get("row_id"),
            "passed": False,
            "reason": str(exc),
            "error": type(exc).__name__,
            "recomputed": {},
            "true_ts_validated": False,
        }


__all__ = [
    "FREQUENCY_CONVERSION",
    "independent_frequency_analysis",
    "project_and_normalize_direction",
    "rigid_projector",
    "validate_workflow_receipt",
]
