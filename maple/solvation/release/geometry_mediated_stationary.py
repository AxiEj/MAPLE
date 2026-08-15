"""Frozen internal-coordinate search contract for the weak water scalar.

This is a research-evidence primitive, not a public optimizer.  It constrains a
nonlinear water molecule to its three exact internal degrees of freedom while
holding the center of mass and laboratory-frame embedding fixed.  The runner
uses SciPy's MINPACK hybrid root solver; this module only defines the geometry
map and independently reduces the raw search trace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256

from .geometry_mediated import geometry_mediated_trial_step_guard
from .geometry_mediated_path import aimnet2_geometry_mediated_water_loop_atoms

AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_SEARCH_CONTRACT_VERSION = (
    "route2-aimnet2-geometry-mediated-stationary-water-search-contract-v1"
)
AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_COORDINATE_NAMES = (
    "oh1_A",
    "oh2_A",
    "hoh_angle_rad",
)
AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS = (
    (0.85, 1.10),
    (0.85, 1.10),
    (math.radians(90.0), math.radians(120.0)),
)
AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_METHOD = "hybr"
AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_FACTOR = 0.2
AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_XTOL = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV = 20
AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_GRADIENT_TOLERANCE = 1.0e-9
AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_CARTESIAN_GRADIENT_TOLERANCE = 1.0e-8


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _array(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.array(result, copy=True)


@dataclass(frozen=True, slots=True)
class WaterInternalCoordinateGeometry:
    """Cartesian water embedding and its first/second internal derivatives."""

    positions_A: np.ndarray
    jacobian: np.ndarray
    second_derivatives: np.ndarray


def aimnet2_geometry_mediated_stationary_water_initial_coordinates() -> np.ndarray:
    """Return ``(r_OH1, r_OH2, angle_HOH)`` for the frozen water canary."""

    atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    first = np.asarray(atoms.positions[1] - atoms.positions[0], dtype=float)
    second = np.asarray(atoms.positions[2] - atoms.positions[0], dtype=float)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    result = np.array(
        (first_norm, second_norm, math.acos(float(np.clip(cosine, -1.0, 1.0)))),
        dtype=float,
    )
    result.setflags(write=False)
    return result


def aimnet2_geometry_mediated_stationary_water_geometry(
    internal_coordinates: object,
) -> WaterInternalCoordinateGeometry:
    """Map water bond lengths/angle to a fixed-COM Cartesian embedding."""

    coordinates = _array(
        internal_coordinates,
        shape=(3,),
        name="water internal coordinates",
    )
    for value, bounds, name in zip(
        coordinates,
        AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS,
        AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_COORDINATE_NAMES,
        strict=True,
    ):
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{name} lies outside the frozen search bounds.")

    reference = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    masses = np.asarray(reference.get_masses(), dtype=float)
    total_mass = float(np.sum(masses))
    reference_com = np.sum(reference.positions * masses[:, None], axis=0) / total_mass
    first_length, second_length, angle = coordinates
    sine = math.sin(angle / 2.0)
    cosine = math.cos(angle / 2.0)

    raw = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.0, first_length * sine, -first_length * cosine),
            (0.0, -second_length * sine, -second_length * cosine),
        ),
        dtype=float,
    )
    jacobian = np.zeros((3, 3, 3), dtype=float)
    jacobian[1, :, 0] = (0.0, sine, -cosine)
    jacobian[2, :, 1] = (0.0, -sine, -cosine)
    jacobian[1, :, 2] = (
        0.0,
        0.5 * first_length * cosine,
        0.5 * first_length * sine,
    )
    jacobian[2, :, 2] = (
        0.0,
        -0.5 * second_length * cosine,
        0.5 * second_length * sine,
    )

    second = np.zeros((3, 3, 3, 3), dtype=float)
    second[1, :, 0, 2] = second[1, :, 2, 0] = (
        0.0,
        0.5 * cosine,
        0.5 * sine,
    )
    second[2, :, 1, 2] = second[2, :, 2, 1] = (
        0.0,
        -0.5 * cosine,
        0.5 * sine,
    )
    second[1, :, 2, 2] = (
        0.0,
        -0.25 * first_length * sine,
        0.25 * first_length * cosine,
    )
    second[2, :, 2, 2] = (
        0.0,
        0.25 * second_length * sine,
        0.25 * second_length * cosine,
    )

    raw_com = np.sum(raw * masses[:, None], axis=0) / total_mass
    positions = raw - raw_com + reference_com
    for coordinate in range(3):
        jacobian[:, :, coordinate] -= (
            np.sum(jacobian[:, :, coordinate] * masses[:, None], axis=0) / total_mass
        )
    for first_coordinate in range(3):
        for second_coordinate in range(3):
            second[:, :, first_coordinate, second_coordinate] -= (
                np.sum(
                    second[:, :, first_coordinate, second_coordinate] * masses[:, None],
                    axis=0,
                )
                / total_mass
            )

    for values in (positions, jacobian, second):
        values.setflags(write=False)
    return WaterInternalCoordinateGeometry(positions, jacobian, second)


def summarize_aimnet2_geometry_mediated_stationary_water_search(
    search_record: Mapping[str, object],
) -> dict[str, object]:
    """Validate a raw SciPy root trace and its fixed-stratum event guards."""

    raw = _mapping(search_record, name="stationary search record")
    protocol = _mapping(raw.get("protocol"), name="stationary search protocol")
    expected_protocol = {
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
    }
    if dict(protocol) != expected_protocol:
        raise ValueError("stationary water search protocol changed from the contract.")

    result = _mapping(raw.get("result"), name="stationary search result")
    solution = _array(
        result.get("solution"),
        shape=(3,),
        name="stationary search solution",
    )
    solution_geometry = aimnet2_geometry_mediated_stationary_water_geometry(solution)
    if result.get("success") is not True or int(result.get("status")) != 1:
        raise ValueError("stationary water root solver did not report convergence.")
    function_evaluations = int(result.get("nfev"))
    jacobian_evaluations = int(result.get("njev"))
    if (
        function_evaluations <= 0
        or function_evaluations > AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV
        or jacobian_evaluations <= 0
    ):
        raise ValueError("stationary water root evaluation counts are invalid.")

    evaluations = _sequence(raw.get("evaluations"), name="stationary evaluations")
    if not evaluations or len(evaluations) > function_evaluations:
        raise ValueError("stationary search trace coverage is invalid.")
    reference_atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    initial_coordinates = (
        aimnet2_geometry_mediated_stationary_water_initial_coordinates()
    )
    summaries: list[dict[str, object]] = []
    previous_positions = np.asarray(reference_atoms.positions, dtype=float)
    previous_model_topology: Mapping[str, object] | None = None
    previous_continuum_topology: Mapping[str, object] | None = None
    all_guards_passed = True
    for index, item in enumerate(evaluations):
        evaluation = _mapping(item, name=f"stationary evaluation {index}")
        coordinates = _array(
            evaluation.get("internal_coordinates"),
            shape=(3,),
            name="evaluation internal coordinates",
        )
        geometry = aimnet2_geometry_mediated_stationary_water_geometry(coordinates)
        positions = _array(
            evaluation.get("positions_A"),
            shape=(3, 3),
            name="evaluation positions",
        )
        if not np.array_equal(positions, geometry.positions_A):
            raise ValueError("stationary evaluation does not match the water map.")
        atoms = reference_atoms.copy()
        atoms.positions = positions
        if evaluation.get("geometry_sha256") != geometry_sha256(atoms):
            raise ValueError("stationary evaluation geometry SHA changed.")
        gradient = _array(
            evaluation.get("total_gradient_eV_per_A"),
            shape=(3, 3),
            name="evaluation Cartesian gradient",
        )
        internal_gradient = _array(
            evaluation.get("internal_gradient"),
            shape=(3,),
            name="evaluation internal gradient",
        )
        reconstructed_internal = np.einsum("ick,ic->k", geometry.jacobian, gradient)
        internal_ledger_error = float(
            np.max(np.abs(reconstructed_internal - internal_gradient))
        )
        if internal_ledger_error > 2.0e-10:
            raise ValueError("stationary internal-gradient ledger does not close.")
        model_topology = _mapping(
            evaluation.get("model_topology"), name="evaluation model topology"
        )
        continuum_topology = _mapping(
            evaluation.get("continuum_topology"),
            name="evaluation continuum topology",
        )
        if index == 0:
            if not np.array_equal(coordinates, initial_coordinates) or not np.allclose(
                positions,
                reference_atoms.positions,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError(
                    "stationary search did not start at the contract geometry."
                )
            previous_model_topology = model_topology
            previous_continuum_topology = continuum_topology
        assert previous_model_topology is not None
        assert previous_continuum_topology is not None
        guard = geometry_mediated_trial_step_guard(
            center_positions_A=previous_positions,
            trial_positions_A=positions,
            center_model_topology=previous_model_topology,
            trial_model_topology=model_topology,
            center_continuum_topology=previous_continuum_topology,
            trial_continuum_topology=continuum_topology,
        )
        all_guards_passed &= bool(guard["gate_passed"])
        summaries.append(
            {
                "evaluation_index": index,
                "internal_coordinates": coordinates.tolist(),
                "geometry_sha256": geometry_sha256(atoms),
                "energy_eV": _finite(
                    evaluation.get("energy_eV"), name="evaluation energy"
                ),
                "total_gradient_eV_per_A": gradient.tolist(),
                "Cartesian_gradient_norm_eV_per_A": float(np.linalg.norm(gradient)),
                "Cartesian_gradient_max_abs_eV_per_A": float(np.max(np.abs(gradient))),
                "internal_gradient_norm": float(np.linalg.norm(internal_gradient)),
                "internal_gradient_max_abs": float(np.max(np.abs(internal_gradient))),
                "internal_gradient_ledger_max_abs": internal_ledger_error,
                "event_guard": guard,
            }
        )
        previous_positions = positions
        previous_model_topology = model_topology
        previous_continuum_topology = continuum_topology

    final = summaries[-1]
    if not np.array_equal(
        np.asarray(final["internal_coordinates"], dtype=float), solution
    ):
        raise ValueError("stationary solution and final search evaluation disagree.")
    gates = {
        "solver_converged_within_budget": True,
        "search_energy_nonincreasing": (
            float(final["energy_eV"]) <= float(summaries[0]["energy_eV"])
        ),
        "all_search_segments_event_guarded": all_guards_passed,
        "final_internal_stationarity": (
            float(final["internal_gradient_max_abs"])
            <= AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_GRADIENT_TOLERANCE
        ),
        "final_cartesian_stationarity": (
            float(final["Cartesian_gradient_max_abs_eV_per_A"])
            <= AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_CARTESIAN_GRADIENT_TOLERANCE
        ),
    }
    return {
        "contract_version": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_SEARCH_CONTRACT_VERSION
        ),
        "protocol": expected_protocol,
        "result": {
            "success": True,
            "status": 1,
            "message": str(result.get("message")),
            "nfev": function_evaluations,
            "njev": jacobian_evaluations,
            "solution": solution.tolist(),
            "positions_A": solution_geometry.positions_A.tolist(),
        },
        "evaluations": summaries,
        "gates": gates,
        "search_gates_passed": all(gates.values()),
        "diagnostic_only": True,
        "public_opt_admitted": False,
    }


__all__ = [
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_CARTESIAN_GRADIENT_TOLERANCE",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_COORDINATE_NAMES",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_GRADIENT_TOLERANCE",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_FACTOR",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_METHOD",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_XTOL",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_SEARCH_CONTRACT_VERSION",
    "WaterInternalCoordinateGeometry",
    "aimnet2_geometry_mediated_stationary_water_geometry",
    "aimnet2_geometry_mediated_stationary_water_initial_coordinates",
    "summarize_aimnet2_geometry_mediated_stationary_water_search",
]
