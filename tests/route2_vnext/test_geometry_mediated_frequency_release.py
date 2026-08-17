from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pytest

from _geometry_mediated_records import (
    synthetic_ddpcm_stationarity_record,
    synthetic_reciprocity_record,
)

from maple.function.dispatcher.frequency.normal_modes import rigid_body_subspaces
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.geometry_mediated_frequency import (
    AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A,
    AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_CONTRACT_VERSION,
    summarize_aimnet2_geometry_mediated_frequency_water,
)
from maple.solvation.release.geometry_mediated_path import (
    aimnet2_geometry_mediated_water_loop_atoms,
)
from maple.solvation.release.geometry_mediated_stationary import (
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_BOUND_TRANSFORM,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_COORDINATE_NAMES,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_FACTOR,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_METHOD,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_XTOL,
    aimnet2_geometry_mediated_stationary_water_geometry,
    aimnet2_geometry_mediated_stationary_water_initial_coordinates,
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


def _search_and_atoms():
    coordinates = aimnet2_geometry_mediated_stationary_water_initial_coordinates()
    geometry = aimnet2_geometry_mediated_stationary_water_geometry(coordinates)
    atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    atoms.positions = geometry.positions_A
    search = {
        "protocol": {
            "solver": "scipy.optimize.root",
            "method": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_METHOD,
            "factor": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_FACTOR,
            "xtol": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_XTOL,
            "maxfev": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV,
            "internal_coordinate_names": list(
                AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_COORDINATE_NAMES
            ),
            "internal_bounds": [
                list(bounds)
                for bounds in AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS
            ],
            "public_optimizer": False,
        },
        "result": {
            "success": True,
            "status": 1,
            "message": "synthetic root",
            "nfev": 1,
            "njev": 1,
            "solution": coordinates.tolist(),
        },
        "evaluations": [
            {
                "internal_coordinates": coordinates.tolist(),
                "positions_A": atoms.positions.tolist(),
                "geometry_sha256": geometry_sha256(atoms),
                "energy_eV": -1.0,
                "total_gradient_eV_per_A": np.zeros((3, 3)).tolist(),
                "internal_gradient": np.zeros(3).tolist(),
                "model_topology": _model_topology(),
                "continuum_topology": _continuum_topology(),
            }
        ],
    }
    return search, atoms


def _synthetic_hessian(atoms) -> np.ndarray:
    masses = atoms.get_masses()
    subspaces = rigid_body_subspaces(masses, atoms.positions)
    vibrational = subspaces.vibrational_basis_mass_weighted
    hessian_mass_weighted = vibrational @ np.diag([1.0, 4.0, 9.0]) @ vibrational.T
    square_root_mass = np.sqrt(np.repeat(masses, 3))
    return square_root_mass[:, None] * hessian_mass_weighted * square_root_mass[None, :]


def _center(atoms) -> dict[str, object]:
    zeros_source = np.zeros((3, 4))
    zeros_gradient = np.zeros((3, 3))
    return {
        "geometry_sha256": geometry_sha256(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "masses_amu": atoms.get_masses().tolist(),
        "positions_A": atoms.positions.tolist(),
        "energy": {
            "vacuum_energy_eV": -1.0,
            "continuum_energy_eV": 0.0,
            "total_energy_eV": -1.0,
        },
        "source": zeros_source.tolist(),
        "reaction_field": zeros_source.tolist(),
        "intrinsic_gradient_eV_per_A": zeros_gradient.tolist(),
        "continuum_fixed_source_gradient_eV_per_A": zeros_gradient.tolist(),
        "source_response_gradient_eV_per_A": zeros_gradient.tolist(),
        "total_gradient_eV_per_A": zeros_gradient.tolist(),
        "model_topology": _model_topology(),
        "continuum_topology": _continuum_topology(),
        "stationarity": _stationarity(),
        "reciprocity": synthetic_reciprocity_record(),
        "replay": {
            "total_energy_eV": -1.0,
            "source": zeros_source.tolist(),
            "total_gradient_eV_per_A": zeros_gradient.tolist(),
        },
    }


def _hvp_records(hessian: np.ndarray) -> list[dict[str, object]]:
    source = np.zeros((3, 4))
    records = []
    for coordinate_index in range(9):
        direction = np.zeros((3, 3))
        direction.reshape(-1)[coordinate_index] = 1.0
        total_hvp = hessian[:, coordinate_index].reshape(3, 3)
        zeros = np.zeros((3, 3))
        records.append(
            {
                "coordinate_index": coordinate_index,
                "scalar_id": "synthetic-scalar",
                "profile_id": "synthetic-profile",
                "scalar_fingerprint_sha256": _sha("scalar"),
                "model_second_order_behavior_sha256": _sha("model"),
                "continuum_second_order_behavior_sha256": _sha("continuum"),
                "coordinate_direction": direction.tolist(),
                "source": source.tolist(),
                "source_gradient_cotangent": source.tolist(),
                "source_position_jvp": source.tolist(),
                "intrinsic_energy_hvp_eV_per_A2": total_hvp.tolist(),
                "continuum_joint_position_hvp_eV_per_A2": zeros.tolist(),
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


def _finite_differences(atoms, hessian: np.ndarray) -> list[dict[str, object]]:
    noise = np.eye(9)
    records = []
    for step in AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A:
        effective_hessian = hessian + 1.0e3 * step**2 * noise
        axes = []
        for coordinate_index in range(9):
            direction = np.zeros((3, 3))
            direction.reshape(-1)[coordinate_index] = 1.0
            derivative = effective_hessian[:, coordinate_index].reshape(3, 3)

            def endpoint(sign: float):
                displaced = atoms.copy()
                displaced.positions += sign * step * direction
                return {
                    "geometry_sha256": geometry_sha256(displaced),
                    "positions_A": displaced.positions.tolist(),
                    "total_gradient_eV_per_A": (sign * step * derivative).tolist(),
                    "model_topology": _model_topology(),
                    "continuum_topology": _continuum_topology(),
                }

            axes.append(
                {
                    "coordinate_index": coordinate_index,
                    "plus": endpoint(1.0),
                    "minus": endpoint(-1.0),
                }
            )
        records.append({"step_A": step, "axes": axes})
    return records


def _panel():
    search, atoms = _search_and_atoms()
    hessian = _synthetic_hessian(atoms)
    return (
        search,
        _center(atoms),
        _hvp_records(hessian),
        _finite_differences(atoms, hessian),
    )


def test_frequency_reducer_closes_dense_hessian_and_keeps_tasks_disabled():
    search, center, hvps, finite_differences = _panel()
    summary = summarize_aimnet2_geometry_mediated_frequency_water(
        search_record=search,
        center_record=center,
        hessian_vector_records=hvps,
        finite_difference_records=finite_differences,
    )

    assert AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_CONTRACT_VERSION.endswith("-v1")
    assert summary["diagnostic_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["vibrational_analysis"]["rigid_rank"] == 6
    assert summary["vibrational_analysis"]["vibrational_rank"] == 3
    assert summary["vibrational_analysis"]["eigenvalues_eV_per_A2_amu"] == (
        pytest.approx([1.0, 4.0, 9.0], abs=2.0e-14)
    )
    assert all(value is False for value in summary["capabilities"].values())
    assert all(value is False for value in summary["workflow_admission"].values())
    assert summary["tier_h_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False


def test_frequency_reducer_accepts_finite_dielectric_stationarity():
    search, center, hvps, finite_differences = _panel()
    search["protocol"][
        "internal_coordinate_transform"
    ] = AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_BOUND_TRANSFORM
    center["stationarity"] = synthetic_ddpcm_stationarity_record()
    summary = summarize_aimnet2_geometry_mediated_frequency_water(
        search_record=search,
        center_record=center,
        hessian_vector_records=hvps,
        finite_difference_records=finite_differences,
        continuum_kind="harmonic-ddpcm-water",
    )
    assert summary["diagnostic_gates_passed"] is True
    assert summary["center"]["stationarity"]["stationarity_kind"] == (
        "harmonic-ddpcm-primal-adjoint-kkt"
    )


def test_frequency_reducer_rejects_an_asymmetric_hvp_matrix():
    search, center, hvps, finite_differences = _panel()
    broken = deepcopy(hvps)
    broken[0]["intrinsic_energy_hvp_eV_per_A2"][0][1] += 1.0e-3
    broken[0]["total_hvp_eV_per_A2"][0][1] += 1.0e-3

    with pytest.raises(ValueError, match="not symmetric"):
        summarize_aimnet2_geometry_mediated_frequency_water(
            search_record=search,
            center_record=center,
            hessian_vector_records=broken,
            finite_difference_records=finite_differences,
        )
