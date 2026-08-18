"""Fail-closed ASE bridge for an explicit geometry-mediated scalar.

The bridge publishes only the energy and exact first derivative returned by one
scalar evaluation.  It rejects periodic systems, unsupported properties, an
event-unsafe current geometry, and any uncertified topology transition between
successive accepted geometries.  Constructing this class does not itself admit
a MAPLE input profile; public routing remains a separate capability decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
import numpy as np


class GeometryMediatedOperationalDomainError(RuntimeError):
    """The requested geometry lies outside the guarded differentiable stratum."""


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _topologies(scalar: object, atoms: object) -> tuple[dict[str, object], ...]:
    model = getattr(scalar, "model", None)
    continuum = getattr(scalar, "continuum", None)
    neighbor_topology = getattr(model, "neighbor_topology", None)
    continuum_topology = getattr(continuum, "topology_state", None)
    if not callable(neighbor_topology) or not callable(continuum_topology):
        raise TypeError(
            "geometry-mediated ASE scalar must expose model.neighbor_topology() "
            "and continuum.topology_state()."
        )
    model_state = neighbor_topology(atoms)
    as_dict = getattr(model_state, "as_dict", None)
    if not callable(as_dict):
        raise TypeError("model neighbor topology must expose as_dict().")
    return (
        _mapping(as_dict(), name="model topology"),
        _mapping(continuum_topology(atoms), name="continuum topology"),
    )


def _trial_guard(
    *,
    center_positions: np.ndarray,
    trial_positions: np.ndarray,
    center_model_topology: Mapping[str, object],
    trial_model_topology: Mapping[str, object],
    center_continuum_topology: Mapping[str, object],
    trial_continuum_topology: Mapping[str, object],
) -> dict[str, object]:
    # Local import avoids coupling the scalar package's import graph to the
    # evidence package while reusing the exact preregistered 0.02-A guards.
    from maple.solvation.release.geometry_mediated import (
        geometry_mediated_trial_step_guard,
    )

    return geometry_mediated_trial_step_guard(
        center_positions_A=center_positions,
        trial_positions_A=trial_positions,
        center_model_topology=center_model_topology,
        trial_model_topology=trial_model_topology,
        center_continuum_topology=center_continuum_topology,
        trial_continuum_topology=trial_continuum_topology,
    )


def _failed_guard_names(record: Mapping[str, object]) -> tuple[str, ...]:
    failed: list[str] = []
    for name in ("same_model_topology", "same_continuum_topology"):
        if record.get(name) is not True:
            failed.append(name)
    for name in ("neighbor_cutoff", "point_source_shell", "sphere_tangency"):
        gate = record.get(name)
        if not isinstance(gate, Mapping) or gate.get("gate_passed") is not True:
            failed.append(name)
    return tuple(failed)


class GeometryMediatedScalarASECalculator(Calculator):
    """Expose guarded energy/free-energy/forces from one explicit scalar."""

    implemented_properties = ("energy", "free_energy", "forces")

    def __init__(self, scalar: object, **kwargs: Any) -> None:
        if not callable(getattr(scalar, "evaluate", None)):
            raise TypeError("geometry-mediated scalar must expose evaluate().")
        self.scalar = scalar
        self.last_domain_guard: dict[str, object] | None = None
        self.last_evaluation: object | None = None
        self._accepted_positions: np.ndarray | None = None
        self._accepted_atomic_numbers: tuple[int, ...] | None = None
        self._accepted_model_topology: dict[str, object] | None = None
        self._accepted_continuum_topology: dict[str, object] | None = None
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self.last_domain_guard = None
        self.last_evaluation = None
        self._accepted_positions = None
        self._accepted_atomic_numbers = None
        self._accepted_model_topology = None
        self._accepted_continuum_topology = None

    def _require_supported_properties(self, properties: Sequence[str]) -> None:
        unsupported = tuple(
            name for name in properties if name not in self.implemented_properties
        )
        if unsupported:
            raise PropertyNotImplementedError(
                "geometry-mediated AIMNet2 scalar does not implement "
                + ", ".join(unsupported)
                + "; only energy, free_energy, and forces are available."
            )

    @staticmethod
    def _require_nonperiodic(atoms: object) -> None:
        getter = getattr(atoms, "get_pbc", None)
        if not callable(getter):
            raise TypeError("ASE geometry must expose get_pbc().")
        pbc = np.asarray(getter(), dtype=bool)
        if pbc.shape != (3,) or np.any(pbc):
            raise GeometryMediatedOperationalDomainError(
                "AIMNet2 frozen-charge implicit solvent is restricted to "
                "non-periodic molecular systems."
            )

    def _guard_geometry(
        self, atoms: object
    ) -> tuple[np.ndarray, tuple[int, ...], dict[str, object], dict[str, object]]:
        self._require_nonperiodic(atoms)
        positions_getter = getattr(atoms, "get_positions", None)
        numbers_getter = getattr(atoms, "get_atomic_numbers", None)
        if not callable(positions_getter) or not callable(numbers_getter):
            raise TypeError(
                "ASE geometry must expose get_positions() and get_atomic_numbers()."
            )
        positions = np.asarray(positions_getter(), dtype=float)
        numbers_array = np.asarray(numbers_getter())
        if (
            positions.ndim != 2
            or positions.shape[0] < 1
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
            or numbers_array.shape != (len(positions),)
            or not np.issubdtype(numbers_array.dtype, np.integer)
        ):
            raise ValueError("ASE geometry positions/numbers are invalid.")
        numbers = tuple(int(value) for value in numbers_array)
        if (
            self._accepted_atomic_numbers is not None
            and numbers != self._accepted_atomic_numbers
        ):
            raise GeometryMediatedOperationalDomainError(
                "one geometry-mediated ASE calculator cannot change atomic identity."
            )
        model_topology, continuum_topology = _topologies(self.scalar, atoms)
        point_guard = _trial_guard(
            center_positions=positions,
            trial_positions=positions,
            center_model_topology=model_topology,
            trial_model_topology=model_topology,
            center_continuum_topology=continuum_topology,
            trial_continuum_topology=continuum_topology,
        )
        transition_guard = None
        if self._accepted_positions is not None:
            transition_guard = _trial_guard(
                center_positions=self._accepted_positions,
                trial_positions=positions,
                center_model_topology=self._accepted_model_topology,
                trial_model_topology=model_topology,
                center_continuum_topology=self._accepted_continuum_topology,
                trial_continuum_topology=continuum_topology,
            )
        gate_passed = point_guard.get("gate_passed") is True and (
            transition_guard is None or transition_guard.get("gate_passed") is True
        )
        combined = {
            "current_geometry": point_guard,
            "accepted_to_trial_segment": transition_guard,
            "gate_passed": gate_passed,
        }
        self.last_domain_guard = combined
        if not gate_passed:
            failures = list(_failed_guard_names(point_guard))
            if transition_guard is not None:
                failures.extend(_failed_guard_names(transition_guard))
            raise GeometryMediatedOperationalDomainError(
                "geometry-mediated operational-domain guard failed: "
                + ", ".join(dict.fromkeys(failures))
                + "."
            )
        return (
            np.array(positions, copy=True),
            numbers,
            model_topology,
            continuum_topology,
        )

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        self._require_supported_properties(properties)
        super().calculate(atoms, properties, system_changes)
        positions, numbers, model_topology, continuum_topology = self._guard_geometry(
            self.atoms
        )
        evaluation = self.scalar.evaluate(self.atoms)
        energy = float(getattr(getattr(evaluation, "energy", None), "total_energy_eV"))
        forces = np.asarray(getattr(evaluation, "forces_eV_per_A", None), dtype=float)
        if (
            not np.isfinite(energy)
            or forces.shape != positions.shape
            or not np.all(np.isfinite(forces))
        ):
            raise ValueError(
                "geometry-mediated scalar returned an invalid energy/force ledger."
            )
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": np.array(forces, copy=True),
        }
        self.last_evaluation = evaluation
        self._accepted_positions = positions
        self._accepted_atomic_numbers = numbers
        self._accepted_model_topology = model_topology
        self._accepted_continuum_topology = continuum_topology


__all__ = [
    "GeometryMediatedOperationalDomainError",
    "GeometryMediatedScalarASECalculator",
]
