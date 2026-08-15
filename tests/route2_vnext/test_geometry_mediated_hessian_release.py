from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pytest

from _geometry_mediated_records import synthetic_reciprocity_record

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.geometry_mediated_hessian import (
    AIMNET2_GEOMETRY_MEDIATED_HVP_DIRECTION_NAMES,
    AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A,
    AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_CONTRACT_VERSION,
    aimnet2_geometry_mediated_hvp_directions,
    summarize_aimnet2_geometry_mediated_hvp_water,
)
from maple.solvation.release.geometry_mediated_path import (
    aimnet2_geometry_mediated_water_loop_atoms,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _model_topology(label: str = "model") -> dict[str, object]:
    return {
        "topology_sha256": _sha(label),
        "minimum_cutoff_margin_angstrom": 1.0,
    }


def _continuum_topology(label: str = "continuum") -> dict[str, object]:
    return {
        "cavity_topology_sha256": _sha(label),
        "coefficient_count": 12,
        "point_source_topology_sha256": _sha(f"{label}:point"),
        "sphere_pair_topology_sha256": _sha(f"{label}:sphere"),
        "minimum_point_source_shell_margin_angstrom": 1.0,
        "minimum_sphere_tangency_margin_angstrom": 1.0,
    }


def _stationarity() -> dict[str, object]:
    return {
        "absolute_residual_eV_per_e": 1.0e-14,
        "relative_residual": 1.0e-15,
        "surface_condition_number": 100.0,
        "state_dimension": 12,
        "thresholds": {
            "absolute_residual_eV_per_e": 1.0e-10,
            "relative_residual": 1.0e-10,
            "surface_condition_number": 1.0e12,
        },
        "gate_passed": True,
        "capability_admitted": False,
    }


def _center() -> dict[str, object]:
    atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    positions = np.asarray(atoms.positions)
    total = 0.5 * float(np.vdot(positions, positions))
    source = np.zeros((len(atoms), 4))
    return {
        "geometry_sha256": geometry_sha256(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": positions.tolist(),
        "energy": {
            "vacuum_energy_eV": total,
            "continuum_energy_eV": 0.0,
            "total_energy_eV": total,
        },
        "source": source.tolist(),
        "reaction_field": source.tolist(),
        "intrinsic_gradient_eV_per_A": positions.tolist(),
        "continuum_fixed_source_gradient_eV_per_A": np.zeros_like(positions).tolist(),
        "source_response_gradient_eV_per_A": np.zeros_like(positions).tolist(),
        "total_gradient_eV_per_A": positions.tolist(),
        "model_topology": _model_topology(),
        "continuum_topology": _continuum_topology(),
        "stationarity": _stationarity(),
        "reciprocity": synthetic_reciprocity_record(),
    }


def _source_jvp(index: int) -> np.ndarray:
    result = np.zeros((3, 4))
    if index == 0:
        result[:, 0] = (0.1, -0.04, -0.06)
    elif index == 1:
        result[:, 0] = (-0.03, 0.08, -0.05)
    return result


def _hvp_records() -> list[dict[str, object]]:
    atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    source = np.zeros((len(atoms), 4))
    records = []
    for index, (label, direction) in enumerate(
        zip(
            AIMNET2_GEOMETRY_MEDIATED_HVP_DIRECTION_NAMES,
            aimnet2_geometry_mediated_hvp_directions(),
            strict=True,
        )
    ):
        total_hvp = direction if index < 2 else np.zeros_like(direction)
        zeros = np.zeros_like(total_hvp)
        records.append(
            {
                "label": label,
                "scalar_id": "synthetic-scalar",
                "profile_id": "synthetic-profile",
                "scalar_fingerprint_sha256": _sha("scalar"),
                "model_second_order_behavior_sha256": _sha("model-behavior"),
                "continuum_second_order_behavior_sha256": _sha("continuum-behavior"),
                "coordinate_direction": direction.tolist(),
                "source": source.tolist(),
                "source_gradient_cotangent": source.tolist(),
                "source_position_jvp": _source_jvp(index).tolist(),
                "intrinsic_energy_hvp_eV_per_A2": total_hvp.tolist(),
                "continuum_joint_position_hvp_eV_per_A2": zeros.tolist(),
                "continuum_joint_source_hvp": source.tolist(),
                "continuum_source_response_pullback_eV_per_A2": zeros.tolist(),
                "contracted_source_hessian_eV_per_A2": zeros.tolist(),
                "total_hvp_eV_per_A2": total_hvp.tolist(),
                "model_standard_decomposed_energy_absolute_error_eV": 0.0,
                "model_standard_decomposed_charge_max_absolute_error_e": 0.0,
                "model_standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A": 0.0,
                "model_standard_decomposed_charge_vjp_max_absolute_error_eV_per_A": 0.0,
                "model_charge_tangent_residual_e_per_A": 0.0,
                "diagnostic_only": True,
                "tier_h_admitted": False,
            }
        )
    return records


def _unit_source_noise() -> np.ndarray:
    result = np.zeros((3, 4))
    result[:, 0] = np.asarray((1.0, -1.0, 0.0)) / np.sqrt(2.0)
    return result


def _finite_difference_records(
    center: dict[str, object], hvps: list[dict[str, object]]
) -> list[dict[str, object]]:
    atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    positions = np.asarray(atoms.positions)
    direction = np.asarray(hvps[0]["coordinate_direction"])
    source = np.asarray(center["source"])
    gradient = np.asarray(center["total_gradient_eV_per_A"])
    source_jvp = np.asarray(hvps[0]["source_position_jvp"])
    total_hvp = np.asarray(hvps[0]["total_hvp_eV_per_A2"])
    source_noise = _unit_source_noise()
    coordinate_noise = np.zeros_like(positions)
    coordinate_noise[0, 0] = 1.0
    charge_errors = (1.0e-7, 2.0e-8, 4.0e-9)
    contracted_errors = (2.0e-6, 4.0e-7, 8.0e-8)
    total_errors = (1.0e-4, 2.0e-5, 4.0e-6)
    records = []
    for step, charge_error, contracted_error, total_error in zip(
        AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A,
        charge_errors,
        contracted_errors,
        total_errors,
        strict=True,
    ):
        plus_positions = positions + step * direction
        minus_positions = positions - step * direction
        plus_atoms = atoms.copy()
        minus_atoms = atoms.copy()
        plus_atoms.positions = plus_positions
        minus_atoms.positions = minus_positions

        def endpoint(sign: float, endpoint_atoms, endpoint_positions):
            endpoint_source = source + sign * step * (
                source_jvp + charge_error * source_noise
            )
            fixed_vjp = sign * step * contracted_error * coordinate_noise
            endpoint_gradient = gradient + sign * step * (
                total_hvp + total_error * coordinate_noise
            )
            return {
                "geometry_sha256": geometry_sha256(endpoint_atoms),
                "atomic_numbers": endpoint_atoms.numbers.tolist(),
                "positions_A": endpoint_positions.tolist(),
                "source": endpoint_source.tolist(),
                "fixed_source_cotangent_vjp_eV_per_A": fixed_vjp.tolist(),
                "total_gradient_eV_per_A": endpoint_gradient.tolist(),
                "model_topology": _model_topology(),
                "continuum_topology": _continuum_topology(),
            }

        records.append(
            {
                "step_A": step,
                "plus": endpoint(1.0, plus_atoms, plus_positions),
                "minus": endpoint(-1.0, minus_atoms, minus_positions),
            }
        )
    return records


def _raw_panel():
    center = _center()
    hvps = _hvp_records()
    finite_differences = _finite_difference_records(center, hvps)
    return center, hvps, finite_differences


def test_hvp_water_contract_recomputes_complete_local_second_order_gates():
    assert AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_CONTRACT_VERSION.endswith("-v1")
    center, hvps, finite_differences = _raw_panel()
    summary = summarize_aimnet2_geometry_mediated_hvp_water(
        center_record=center,
        direction_records=hvps,
        finite_difference_records=finite_differences,
    )
    assert summary["diagnostic_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["tier_h_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False
    assert all(value is False for value in summary["capabilities"].values())
    assert all(value is False for value in summary["workflow_admission"].values())
    assert summary["translation_zero_modes"]["total_hvp_norms_eV_per_A2"] == [
        0.0,
        0.0,
        0.0,
    ]


def test_hvp_water_contract_rejects_a_broken_component_ledger():
    center, hvps, finite_differences = _raw_panel()
    hvps[0]["intrinsic_energy_hvp_eV_per_A2"][0][0] += 1.0e-3
    with pytest.raises(ValueError, match="component ledger"):
        summarize_aimnet2_geometry_mediated_hvp_water(
            center_record=center,
            direction_records=hvps,
            finite_difference_records=finite_differences,
        )


def test_hvp_water_contract_recomputes_fd_errors_and_fails_tampering():
    center, hvps, finite_differences = _raw_panel()
    finite_differences[-1]["plus"]["total_gradient_eV_per_A"][0][0] += 1.0e-3
    summary = summarize_aimnet2_geometry_mediated_hvp_water(
        center_record=center,
        direction_records=hvps,
        finite_difference_records=finite_differences,
    )
    assert summary["gates"]["complete_hvp_total_gradient_finite_difference"] is False
    assert summary["diagnostic_gates_passed"] is False


def test_hvp_water_contract_fails_closed_on_topology_or_admission_changes():
    center, hvps, finite_differences = _raw_panel()
    changed_topology = deepcopy(finite_differences)
    changed_topology[0]["plus"]["model_topology"]["topology_sha256"] = _sha(
        "other-model-topology"
    )
    summary = summarize_aimnet2_geometry_mediated_hvp_water(
        center_record=center,
        direction_records=hvps,
        finite_difference_records=changed_topology,
    )
    assert summary["gates"]["all_stencils_same_stratum"] is False
    assert summary["diagnostic_gates_passed"] is False

    hvps[0]["tier_h_admitted"] = True
    with pytest.raises(ValueError, match="admission boundary"):
        summarize_aimnet2_geometry_mediated_hvp_water(
            center_record=center,
            direction_records=hvps,
            finite_difference_records=finite_differences,
        )
