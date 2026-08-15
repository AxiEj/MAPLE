from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.geometry_mediated import geometry_mediated_trial_step_guard
from maple.solvation.release.geometry_mediated_path import (
    aimnet2_geometry_mediated_water_loop_atoms,
)
from maple.solvation.release.geometry_mediated_stationary import (
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_COORDINATE_NAMES,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_FACTOR,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_METHOD,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_XTOL,
    aimnet2_geometry_mediated_stationary_water_geometry,
    aimnet2_geometry_mediated_stationary_water_initial_coordinates,
    summarize_aimnet2_geometry_mediated_stationary_water_search,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _model_topology() -> dict[str, object]:
    return {
        "topology_sha256": _sha("model"),
        "minimum_cutoff_margin_angstrom": 1.0,
    }


def _continuum_topology(*, point_margin: float = 1.0) -> dict[str, object]:
    return {
        "cavity_topology_sha256": _sha("continuum"),
        "coefficient_count": 12,
        "point_source_topology_sha256": _sha("point"),
        "sphere_pair_topology_sha256": _sha("sphere"),
        "minimum_point_source_shell_margin_angstrom": point_margin,
        "minimum_sphere_tangency_margin_angstrom": 1.0,
    }


def _search_record() -> dict[str, object]:
    coordinates = aimnet2_geometry_mediated_stationary_water_initial_coordinates()
    geometry = aimnet2_geometry_mediated_stationary_water_geometry(coordinates)
    atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    atoms.positions = geometry.positions_A
    return {
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
                "positions_A": geometry.positions_A.tolist(),
                "geometry_sha256": geometry_sha256(atoms),
                "energy_eV": -1.0,
                "total_gradient_eV_per_A": np.zeros((3, 3)).tolist(),
                "internal_gradient": np.zeros(3).tolist(),
                "model_topology": _model_topology(),
                "continuum_topology": _continuum_topology(),
            }
        ],
    }


def test_internal_coordinate_map_reproduces_reference_and_derivatives():
    coordinates = aimnet2_geometry_mediated_stationary_water_initial_coordinates()
    geometry = aimnet2_geometry_mediated_stationary_water_geometry(coordinates)
    reference = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))

    assert geometry.positions_A == pytest.approx(reference.positions, abs=1.0e-15)
    masses = reference.get_masses()
    assert np.einsum("i,ick->ck", masses, geometry.jacobian) == pytest.approx(
        np.zeros((3, 3)), abs=2.0e-15
    )

    step = 1.0e-6
    for coordinate in range(3):
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        plus_geometry = aimnet2_geometry_mediated_stationary_water_geometry(plus)
        minus_geometry = aimnet2_geometry_mediated_stationary_water_geometry(minus)
        jacobian_fd = (plus_geometry.positions_A - minus_geometry.positions_A) / (
            2.0 * step
        )
        second_fd = (plus_geometry.jacobian - minus_geometry.jacobian) / (2.0 * step)
        assert jacobian_fd == pytest.approx(
            geometry.jacobian[:, :, coordinate], abs=1.0e-10
        )
        assert second_fd == pytest.approx(
            geometry.second_derivatives[:, :, :, coordinate], abs=1.0e-10
        )


def test_stationary_search_reducer_keeps_optimizer_and_admission_closed():
    summary = summarize_aimnet2_geometry_mediated_stationary_water_search(
        _search_record()
    )

    assert summary["search_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["diagnostic_only"] is True
    assert summary["public_opt_admitted"] is False


def test_stationary_search_rejects_an_internal_gradient_not_from_cartesian_gradient():
    record = _search_record()
    record["evaluations"][0]["internal_gradient"][0] = 1.0e-3

    with pytest.raises(ValueError, match="internal-gradient ledger"):
        summarize_aimnet2_geometry_mediated_stationary_water_search(record)


def test_stationary_search_certifies_each_solver_segment_not_one_global_chord():
    record = _search_record()
    initial = aimnet2_geometry_mediated_stationary_water_initial_coordinates()
    final = np.array([1.02268043, 1.02268043, 2.01181603])
    coordinates = (initial, 0.5 * (initial + final), final)
    evaluations = []
    atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    for index, values in enumerate(coordinates):
        geometry = aimnet2_geometry_mediated_stationary_water_geometry(values)
        point = atoms.copy()
        point.positions = geometry.positions_A
        evaluations.append(
            {
                "internal_coordinates": values.tolist(),
                "positions_A": point.positions.tolist(),
                "geometry_sha256": geometry_sha256(point),
                "energy_eV": -1.0 - index,
                "total_gradient_eV_per_A": np.zeros((3, 3)).tolist(),
                "internal_gradient": np.zeros(3).tolist(),
                "model_topology": _model_topology(),
                "continuum_topology": _continuum_topology(point_margin=0.19),
            }
        )
    record["evaluations"] = evaluations
    record["result"].update(nfev=3, solution=final.tolist())

    direct = geometry_mediated_trial_step_guard(
        center_positions_A=evaluations[0]["positions_A"],
        trial_positions_A=evaluations[-1]["positions_A"],
        center_model_topology=evaluations[0]["model_topology"],
        trial_model_topology=evaluations[-1]["model_topology"],
        center_continuum_topology=evaluations[0]["continuum_topology"],
        trial_continuum_topology=evaluations[-1]["continuum_topology"],
    )
    summary = summarize_aimnet2_geometry_mediated_stationary_water_search(record)

    assert direct["gate_passed"] is False
    assert summary["search_gates_passed"] is True
    assert all(
        evaluation["event_guard"]["gate_passed"]
        for evaluation in summary["evaluations"]
    )
