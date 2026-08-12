"""Execution helpers for the disabled fixed-box590 water path diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np

from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    closed_loop_work,
    displace_positions,
    reverse_closed_path,
    summarize_directional_derivatives,
    water_geometry_descriptors,
)

from fixedbox590_water_common import cold_warm_record, root_context, state_record


@dataclass(frozen=True)
class GradientEvaluation:
    atoms: Atoms
    state: object
    gradient: object
    topology_hash: str

    @property
    def energy(self) -> float:
        return float(self.gradient.scalar.total_energy)

    @property
    def forces(self) -> np.ndarray:
        return np.asarray(self.gradient.forces, dtype=float).reshape(len(self.atoms), 3)


@dataclass(frozen=True)
class EnergyEvaluation:
    atoms: Atoms
    state: object
    energy: float
    topology_hash: str


@dataclass
class AuditTracker:
    topology_hashes: set[str] = field(default_factory=set)
    maximum_primal_residual: float = 0.0
    maximum_adjoint_residual: float = 0.0

    def observe(self, evaluation: GradientEvaluation | EnergyEvaluation) -> None:
        self.topology_hashes.add(evaluation.topology_hash)
        self.maximum_primal_residual = max(
            self.maximum_primal_residual,
            evaluation.state.actual_unmixed_residual_norm,
        )
        gradient = getattr(evaluation, "gradient", None)
        if gradient is not None:
            self.maximum_adjoint_residual = max(
                self.maximum_adjoint_residual,
                gradient.adjoint.true_residual_norm,
            )


class WaterPathDiagnostic:
    """Evaluate one immutable scalar/equation over a preregistered water path."""

    def __init__(
        self,
        *,
        base_atoms: Atoms,
        directions: Mapping[str, np.ndarray],
        continuum: object,
        equation: object,
        scalar: object,
        primal_options: FixedPointOptions,
        adjoint_options: AdjointOptions,
    ) -> None:
        self.base_atoms = base_atoms.copy()
        self.directions = {
            name: np.array(value, dtype=float, copy=True)
            for name, value in directions.items()
        }
        self.continuum = continuum
        self.equation = equation
        self.scalar = scalar
        self.primal_options = primal_options
        self.adjoint_options = adjoint_options
        self.tracker = AuditTracker()

    def solve(self, geometry: Atoms, *, label: str, initial_y=None):
        return solve_fixed_point(
            self.equation,
            geometry,
            scalar_id=self.scalar.scalar_id,
            profile_id=self.scalar.profile_id,
            scalar_binding=self.scalar,
            root_context_id=root_context(geometry, label),
            initial_y=initial_y,
            options=self.primal_options,
        )

    def _topology_hash(self, geometry: Atoms, state: object) -> str:
        return self.continuum.build_state(geometry, state.source).surface.topology_hash

    def gradient(
        self, geometry: Atoms, *, label: str, initial_y=None
    ) -> GradientEvaluation:
        state = self.solve(geometry, label=label, initial_y=initial_y)
        gradient = self.scalar.implicit_gradient(
            geometry, state, adjoint_options=self.adjoint_options
        )
        result = GradientEvaluation(
            geometry.copy(), state, gradient, self._topology_hash(geometry, state)
        )
        self.tracker.observe(result)
        return result

    def energy(
        self, geometry: Atoms, *, label: str, initial_y=None
    ) -> EnergyEvaluation:
        state = self.solve(geometry, label=label, initial_y=initial_y)
        scalar = self.scalar.evaluate(geometry, state.y)
        result = EnergyEvaluation(
            geometry.copy(),
            state,
            float(scalar.total_energy),
            self._topology_hash(geometry, state),
        )
        self.tracker.observe(result)
        return result

    def _directional_record(
        self,
        name: str,
        geometry: Atoms,
        cold: GradientEvaluation,
        direction_name: str,
        direction: np.ndarray,
        steps: Sequence[float],
    ) -> dict[str, object]:
        analytic_gradient = np.asarray(
            cold.gradient.total_coordinate_gradient, dtype=float
        ).reshape(len(geometry), 3)
        analytic = float(np.vdot(analytic_gradient, direction))
        samples: list[tuple[float, float, float]] = []
        displaced: list[dict[str, object]] = []
        for step in steps:
            plus = geometry.copy()
            minus = geometry.copy()
            plus.positions += step * direction
            minus.positions -= step * direction
            plus_value = self.energy(
                plus,
                label=f"panel/{name}/{direction_name}/{step}/plus",
                initial_y=cold.state.y_array(),
            )
            minus_value = self.energy(
                minus,
                label=f"panel/{name}/{direction_name}/{step}/minus",
                initial_y=cold.state.y_array(),
            )
            samples.append((step, plus_value.energy, minus_value.energy))
            displaced.append(
                {
                    "step_A": step,
                    "plus_geometry_sha256": geometry_sha256(plus),
                    "minus_geometry_sha256": geometry_sha256(minus),
                    "plus_energy_eV": plus_value.energy,
                    "minus_energy_eV": minus_value.energy,
                    "plus_primal_residual": plus_value.state.actual_unmixed_residual_norm,
                    "minus_primal_residual": minus_value.state.actual_unmixed_residual_norm,
                    "plus_topology_hash": plus_value.topology_hash,
                    "minus_topology_hash": minus_value.topology_hash,
                }
            )
        result = summarize_directional_derivatives(analytic, samples)
        result["displaced_states"] = displaced
        return result

    def panel_record(
        self,
        panel_coefficients_A: Mapping[str, Mapping[str, float]],
        steps: Sequence[float],
    ) -> tuple[list[dict[str, object]], bool, bool, GradientEvaluation]:
        records: list[dict[str, object]] = []
        directional_pass = True
        cold_warm_pass = True
        equilibrium: GradientEvaluation | None = None
        for name, coefficients in panel_coefficients_A.items():
            geometry = self.base_atoms.copy()
            geometry.positions = displace_positions(
                self.base_atoms.positions, self.directions, coefficients
            )
            cold = self.gradient(geometry, label=f"panel/{name}")
            warm = self.solve(
                geometry,
                label=f"panel/{name}",
                initial_y=np.full(self.equation.reduced_dimension, 1.0e-3),
            )
            root_record = cold_warm_record(cold.state, warm, self.scalar, geometry)
            self.tracker.maximum_primal_residual = max(
                self.tracker.maximum_primal_residual,
                warm.actual_unmixed_residual_norm,
            )
            root_pass = bool(root_record["numerically_equivalent"]) and all(
                root_record["gates"].values()
            )
            cold_warm_pass &= root_pass
            directions = {
                direction_name: self._directional_record(
                    name, geometry, cold, direction_name, direction, steps
                )
                for direction_name, direction in self.directions.items()
            }
            directional_pass &= all(
                bool(item["all_gates_passed"]) for item in directions.values()
            )
            records.append(
                {
                    "name": name,
                    "coefficients_A": dict(coefficients),
                    "atomic_numbers": geometry.numbers.tolist(),
                    "positions_A": geometry.positions.tolist(),
                    "geometry_sha256": geometry_sha256(geometry),
                    "geometry_descriptors": water_geometry_descriptors(
                        geometry.positions
                    ),
                    "cold_state": state_record(cold.state, self.scalar, geometry),
                    "cold_warm": root_record,
                    "adjoint_residual": cold.gradient.adjoint.true_residual_norm,
                    "forces_eV_per_A": cold.forces.tolist(),
                    "topology_hash": cold.topology_hash,
                    "directional_force_fd": directions,
                }
            )
            if name == "equilibrium":
                equilibrium = cold
        if equilibrium is None:
            raise ValueError("panel must include an 'equilibrium' geometry.")
        return records, directional_pass, cold_warm_pass, equilibrium

    def _loop_geometry(
        self,
        coefficient: tuple[float, float],
        amplitudes_A: tuple[float, float],
    ) -> Atoms:
        geometry = self.base_atoms.copy()
        geometry.positions = displace_positions(
            self.base_atoms.positions,
            self.directions,
            {
                "symmetric_stretch": amplitudes_A[0] * coefficient[0],
                "bend": amplitudes_A[1] * coefficient[1],
            },
        )
        return geometry

    @staticmethod
    def _loop_label(coefficient: tuple[float, float]) -> str:
        return f"loop/point/{coefficient[0]:+.8f}/{coefficient[1]:+.8f}"

    def _warm_loop(
        self,
        coefficients: Sequence[tuple[float, float]],
        amplitudes_A: tuple[float, float],
        initial_y: np.ndarray,
    ) -> list[GradientEvaluation]:
        result: list[GradientEvaluation] = []
        previous_y = np.array(initial_y, copy=True)
        for coefficient in coefficients:
            geometry = self._loop_geometry(coefficient, amplitudes_A)
            value = self.gradient(
                geometry,
                label=self._loop_label(coefficient),
                initial_y=previous_y,
            )
            result.append(value)
            previous_y = value.state.y_array()
        return result

    def _loop_root_records(
        self,
        cold: Sequence[GradientEvaluation],
        warm: Sequence[GradientEvaluation],
    ) -> tuple[list[dict[str, object]], bool]:
        records = [
            cold_warm_record(
                cold_item.state, warm_item.state, self.scalar, cold_item.atoms
            )
            for cold_item, warm_item in zip(cold, warm, strict=True)
        ]
        passed = all(
            bool(record["numerically_equivalent"]) and all(record["gates"].values())
            for record in records
        )
        return records, passed

    @staticmethod
    def _path_repeat_record(
        cold: Sequence[GradientEvaluation],
        warm: Sequence[GradientEvaluation],
    ) -> tuple[list[dict[str, object]], bool]:
        records: list[dict[str, object]] = []
        passed = True
        for cold_item, warm_item in zip(cold, warm, strict=True):
            source_difference = float(
                np.linalg.norm(
                    cold_item.state.source_array() - warm_item.state.source_array()
                )
            )
            source_relative = source_difference / max(
                float(np.linalg.norm(cold_item.state.source_array())),
                float(np.linalg.norm(warm_item.state.source_array())),
                1.0e-15,
            )
            energy_difference = abs(cold_item.energy - warm_item.energy)
            item_passed = source_relative <= 1.0e-8 and energy_difference <= 1.0e-8
            passed &= item_passed
            records.append(
                {
                    "geometry_sha256": geometry_sha256(cold_item.atoms),
                    "source_l2_difference": source_difference,
                    "source_relative_difference": source_relative,
                    "energy_abs_difference_eV": energy_difference,
                    "gate_passed": item_passed,
                }
            )
        return records, passed

    def loop_record(
        self,
        coefficients: Sequence[tuple[float, float]],
        amplitudes_A: tuple[float, float],
        subdivisions_per_edge: int,
        initial_y: np.ndarray,
    ) -> tuple[dict[str, object], bool]:
        forward = tuple(coefficients)
        reverse = reverse_closed_path(forward)
        cold = [
            self.gradient(
                self._loop_geometry(coefficient, amplitudes_A),
                label=self._loop_label(coefficient),
            )
            for coefficient in forward
        ]
        warm_forward = self._warm_loop(forward, amplitudes_A, initial_y)
        warm_reverse = self._warm_loop(reverse, amplitudes_A, initial_y)
        cold_positions = [item.atoms.positions for item in cold]
        cold_forces = [item.forces for item in cold]
        work = {
            "cold_forward": closed_loop_work(
                cold_positions, cold_forces, subdivisions_per_edge=subdivisions_per_edge
            ),
            "cold_reverse": closed_loop_work(
                tuple(reversed(cold_positions)),
                tuple(reversed(cold_forces)),
                subdivisions_per_edge=subdivisions_per_edge,
            ),
            "warm_forward": closed_loop_work(
                [item.atoms.positions for item in warm_forward],
                [item.forces for item in warm_forward],
                subdivisions_per_edge=subdivisions_per_edge,
            ),
            "warm_reverse": closed_loop_work(
                [item.atoms.positions for item in warm_reverse],
                [item.forces for item in warm_reverse],
                subdivisions_per_edge=subdivisions_per_edge,
            ),
        }
        forward_roots, forward_root_pass = self._loop_root_records(cold, warm_forward)
        reverse_roots, reverse_root_pass = self._loop_root_records(
            tuple(reversed(cold)), warm_reverse
        )
        forward_reverse_repeat, forward_reverse_repeat_pass = self._path_repeat_record(
            warm_forward, tuple(reversed(warm_reverse))
        )
        gates = {
            "all_four_loop_work_gates": all(
                record["gate_passed"] for record in work.values()
            ),
            "cold_forward_reverse_antisymmetry_le_1e-10_eV": abs(
                work["cold_forward"]["simpson_work_eV"]
                + work["cold_reverse"]["simpson_work_eV"]
            )
            <= 1.0e-10,
            "warm_forward_reverse_antisymmetry_le_1e-8_eV": abs(
                work["warm_forward"]["simpson_work_eV"]
                + work["warm_reverse"]["simpson_work_eV"]
            )
            <= 1.0e-8,
            "all_loop_cold_warm_roots": forward_root_pass and reverse_root_pass,
            "warm_forward_reverse_repeat": forward_reverse_repeat_pass,
        }
        record = {
            "coordinate_names": ["symmetric_stretch", "bend"],
            "amplitudes_A": list(amplitudes_A),
            "subdivisions_per_edge": subdivisions_per_edge,
            "forward_coefficients": forward,
            "reverse_coefficients": reverse,
            "cold_reverse_reuses_independent_cold_point_evaluations": True,
            **work,
            "cold_forward_energy_closure_eV": cold[-1].energy - cold[0].energy,
            "warm_forward_energy_closure_eV": (
                warm_forward[-1].energy - warm_forward[0].energy
            ),
            "warm_reverse_energy_closure_eV": (
                warm_reverse[-1].energy - warm_reverse[0].energy
            ),
            "cold_warm_roots": {
                "forward": forward_roots,
                "reverse": reverse_roots,
            },
            "warm_forward_reverse_repeat": forward_reverse_repeat,
            "gates": gates,
            "states": {
                "cold": [
                    state_record(item.state, self.scalar, item.atoms) for item in cold
                ],
                "warm_forward": [
                    state_record(item.state, self.scalar, item.atoms)
                    for item in warm_forward
                ],
                "warm_reverse": [
                    state_record(item.state, self.scalar, item.atoms)
                    for item in warm_reverse
                ],
            },
        }
        return record, all(gates.values())


__all__ = ["AuditTracker", "WaterPathDiagnostic"]
