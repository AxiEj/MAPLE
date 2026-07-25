"""Model-agnostic linear-Hamiltonian nonequilibrium switching primitives."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class PotentialEvaluator(Protocol):
    """Minimal endpoint interface required by the switching calculator."""

    def evaluate(self, atoms, *, need_forces: bool) -> dict[str, Any]:
        """Return Hartree energy and, when requested, Hartree/Å forces."""


def _coupling(value: float) -> float:
    if (
        isinstance(value, bool)
        or not np.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("The Hamiltonian coupling must be finite and in [0, 1].")
    return float(value)


def _endpoint_result(
    evaluator: PotentialEvaluator,
    atoms,
    *,
    name: str,
) -> tuple[float, np.ndarray]:
    result = evaluator.evaluate(atoms, need_forces=True)
    if not isinstance(result, dict):
        raise TypeError(f"{name} evaluator must return a dictionary.")
    try:
        energy = float(result["energy_hartree"])
        forces = np.asarray(
            result["forces_hartree_per_angstrom"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} evaluator must return energy_hartree and "
            "forces_hartree_per_angstrom."
        ) from exc
    if not np.isfinite(energy):
        raise ValueError(f"{name} evaluator returned a non-finite energy.")
    expected_shape = (len(atoms), 3)
    if forces.shape != expected_shape or not np.all(np.isfinite(forces)):
        raise ValueError(
            f"{name} evaluator forces must be finite with shape {expected_shape}."
        )
    return energy, forces


class LinearHamiltonianCalculator(Calculator):
    """Interpolate two endpoint energies and forces at a shared geometry.

    The target energy may be shifted by a constant for numerical conditioning.
    The constant changes the individual reference-to-target free energy by the
    same amount but leaves endpoint forces unchanged.  Thermodynamic-cycle
    callers must record the shift and either restore it or cancel one common
    shift analytically between cycle legs.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        reference: PotentialEvaluator,
        target: PotentialEvaluator,
        *,
        coupling: float,
        target_energy_offset_hartree: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(target_energy_offset_hartree, bool) or not np.isfinite(
            target_energy_offset_hartree
        ):
            raise ValueError("target_energy_offset_hartree must be finite.")
        self.reference = reference
        self.target = target
        self.coupling = _coupling(coupling)
        self.target_energy_offset_hartree = float(target_energy_offset_hartree)
        self._reference_energy_hartree: float | None = None
        self._target_energy_hartree: float | None = None
        self._reference_forces: np.ndarray | None = None
        self._target_forces: np.ndarray | None = None
        self._operation_counts = {
            "reference_energy_force_evaluations": 0,
            "target_energy_force_evaluations": 0,
        }

    @property
    def endpoint_energy_gap_hartree(self) -> float:
        if (
            self._reference_energy_hartree is None
            or self._target_energy_hartree is None
        ):
            raise RuntimeError("Endpoint energies have not been evaluated.")
        return self._target_energy_hartree - self._reference_energy_hartree

    @property
    def operation_counts(self) -> dict[str, int]:
        return dict(self._operation_counts)

    def _update_mixed_results(self) -> None:
        if (
            self._reference_energy_hartree is None
            or self._target_energy_hartree is None
            or self._reference_forces is None
            or self._target_forces is None
        ):
            return
        reference_weight = 1.0 - self.coupling
        energy = (
            reference_weight * self._reference_energy_hartree
            + self.coupling * self._target_energy_hartree
        )
        forces = (
            reference_weight * self._reference_forces
            + self.coupling * self._target_forces
        )
        self.results = {
            "energy": float(energy),
            "free_energy": float(energy),
            "forces": np.asarray(forces, dtype=np.float64),
        }

    def set_coupling(self, coupling: float) -> None:
        """Change lambda using cached endpoint values at the current geometry."""

        self.coupling = _coupling(coupling)
        self._update_mixed_results()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        reference_energy, reference_forces = _endpoint_result(
            self.reference,
            atoms,
            name="reference",
        )
        target_energy, target_forces = _endpoint_result(
            self.target,
            atoms,
            name="target",
        )
        self._operation_counts["reference_energy_force_evaluations"] += 1
        self._operation_counts["target_energy_force_evaluations"] += 1
        self._reference_energy_hartree = reference_energy
        self._target_energy_hartree = target_energy - self.target_energy_offset_hartree
        self._reference_forces = reference_forces
        self._target_forces = target_forces
        self._update_mixed_results()


def _lambda_schedule(values: Sequence[float]) -> tuple[np.ndarray, str]:
    schedule = np.asarray(values, dtype=np.float64)
    if schedule.ndim != 1 or len(schedule) < 2 or not np.all(np.isfinite(schedule)):
        raise ValueError("lambda_schedule must contain at least two finite values.")
    differences = np.diff(schedule)
    if np.all(differences > 0.0):
        direction = "forward"
        endpoints = (0.0, 1.0)
    elif np.all(differences < 0.0):
        direction = "reverse"
        endpoints = (1.0, 0.0)
    else:
        raise ValueError("lambda_schedule must be strictly monotonic.")
    if not np.isclose(schedule[0], endpoints[0]) or not np.isclose(
        schedule[-1],
        endpoints[1],
    ):
        raise ValueError("lambda_schedule must span either 0 to 1 or 1 to 0.")
    if np.any(schedule < 0.0) or np.any(schedule > 1.0):
        raise ValueError("lambda_schedule values must remain in [0, 1].")
    return schedule, direction


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def run_linear_nonequilibrium_switch(
    atoms,
    calculator: LinearHamiltonianCalculator,
    *,
    lambda_schedule: Sequence[float],
    temperature_kelvin: float,
    timestep_fs: float,
    friction_per_fs: float,
    seed: int,
    initial_velocities_au: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run one LFMiddle switching trajectory and accumulate protocol work.

    Each step first propagates coordinates at the current lambda and then
    perturbs lambda at the propagated geometry.  The work increment is

    ``(lambda_next - lambda_current) * (U_target - U_reference)``.

    Endpoint forces are evaluated once per integration step.  Changing lambda
    reuses those endpoint forces to update the mixed force without an extra
    MLIP evaluation.
    """

    from maple.function.dispatcher.md.integrator.velocity_verlet import (
        VelocityVerlet,
    )
    from maple.function.dispatcher.md.thermostat.langevin import (
        LangevinThermostat,
    )
    from maple.function.dispatcher.md.utils import (
        HA_PER_ANG_TO_AU,
        initialize_velocities,
        lfmiddle_carried_to_standard,
        standard_to_lfmiddle_carried,
    )

    schedule, direction = _lambda_schedule(lambda_schedule)
    temperature_kelvin = _positive("temperature_kelvin", temperature_kelvin)
    timestep_fs = _positive("timestep_fs", timestep_fs)
    if (
        isinstance(friction_per_fs, bool)
        or not np.isfinite(friction_per_fs)
        or float(friction_per_fs) < 0.0
    ):
        raise ValueError("friction_per_fs must be finite and nonnegative.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if atoms.calc is not calculator:
        raise ValueError("atoms must have the supplied switching calculator attached.")

    rng = np.random.default_rng(seed)
    if initial_velocities_au is None:
        velocities = initialize_velocities(
            atoms,
            temperature_kelvin,
            remove_com=True,
            remove_rotation=False,
            remove_angular=False,
            rng=rng,
        )
        velocity_source = "maxwell-boltzmann"
    else:
        velocities = np.asarray(initial_velocities_au, dtype=np.float64).copy()
        if velocities.shape != (len(atoms), 3) or not np.all(np.isfinite(velocities)):
            raise ValueError(
                "initial_velocities_au must be finite with shape " f"{(len(atoms), 3)}."
            )
        velocity_source = "provided"

    calculator.set_coupling(float(schedule[0]))
    counts_before = calculator.operation_counts
    forces = atoms.get_forces() * HA_PER_ANG_TO_AU
    integrator = VelocityVerlet(atoms, timestep_fs)
    thermostat = LangevinThermostat(
        atoms,
        temperature=temperature_kelvin,
        friction=float(friction_per_fs),
        timestep=timestep_fs,
        rng=rng,
    )
    velocities = standard_to_lfmiddle_carried(
        atoms,
        velocities,
        forces,
        integrator.timestep,
    )

    work_increments: list[float] = []
    gaps: list[float] = []
    for current, following in zip(schedule[:-1], schedule[1:], strict=True):
        if not np.isclose(calculator.coupling, current):
            raise RuntimeError(
                "Switching calculator coupling drifted from the schedule."
            )
        velocities = integrator.lfmiddle_full_kick(velocities, forces)
        integrator.half_step_r(velocities)
        velocities = thermostat.apply(velocities)
        velocities, _ = integrator.lfmiddle_post_thermostat(velocities)
        gap = calculator.endpoint_energy_gap_hartree
        increment = float((following - current) * gap)
        gaps.append(gap)
        work_increments.append(increment)
        calculator.set_coupling(float(following))
        forces = (
            np.asarray(calculator.results["forces"], dtype=np.float64)
            * HA_PER_ANG_TO_AU
        )

    final_velocities = lfmiddle_carried_to_standard(
        atoms,
        velocities,
        forces,
        integrator.timestep,
    )
    counts_after = calculator.operation_counts
    operation_counts = {
        name: counts_after[name] - counts_before[name] for name in counts_after
    }
    return {
        "schema_version": 1,
        "direction": direction,
        "temperature_kelvin": temperature_kelvin,
        "timestep_fs": timestep_fs,
        "friction_per_fs": float(friction_per_fs),
        "seed": seed,
        "velocity_source": velocity_source,
        "lambda_schedule": schedule.tolist(),
        "target_energy_offset_hartree": (calculator.target_energy_offset_hartree),
        "work_hartree": float(np.sum(work_increments)),
        "work_increments_hartree": work_increments,
        "endpoint_energy_gaps_hartree": gaps,
        "final_positions_angstrom": np.asarray(
            atoms.get_positions(),
            dtype=np.float64,
        ).tolist(),
        "final_velocities_au": np.asarray(
            final_velocities,
            dtype=np.float64,
        ).tolist(),
        "operation_counts": operation_counts,
    }
