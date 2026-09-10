"""Independent four-step Richardson Hessian trial for experimental AIMNet2.

Version 2 recomputes a complete four-step force ladder.  It does not reinterpret
or mutate the three-step v1 protocol or its retained negative evidence.
"""

from __future__ import annotations

import copy

import numpy as np
from ase.calculators.calculator import all_changes

from .aimnet2_experimental_ase import (
    AIMNet2ExperimentalBudgetError,
    AIMNet2ExperimentalCalculator,
    AIMNet2ExperimentalHessianError,
    _ANTISYMMETRY_MAX_EV_PER_A2,
    _CENTER_REPLAY_ENERGY_MAX_EV,
    _CENTER_REPLAY_FORCE_MAX_EV_PER_A,
    _REFINEMENT_PLATEAU_EV_PER_A2,
    _RICHARDSON_ERROR_MAX_EV_PER_A2,
    _json_safe,
)

EXPERIMENTAL_WORKFLOW_ID_V2 = "aimnet2-smooth-ddpcm-experimental-workflows-v2"
_HESSIAN_STEPS_ANGSTROM_V2 = (4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5)
_RICHARDSON_REFINEMENT_RATIO_MAX_V2 = 0.25


class AIMNet2ExperimentalCalculatorV2(AIMNet2ExperimentalCalculator):
    """Experimental raw-force Hessian with independent Richardson evidence."""

    experimental_workflow_id = EXPERIMENTAL_WORKFLOW_ID_V2
    hessian_evaluations_per_atom = 24

    def get_hessian(self, atoms) -> np.ndarray:
        """Return gated ``sym(K2)`` in eV/Angstrom**2 from four fresh steps."""

        hessian_budget = self._validated_live_budget("hessian_call_budget")
        if hessian_budget is not None and self.hessian_call_count >= hessian_budget:
            raise AIMNet2ExperimentalBudgetError(
                "experimental AIMNet2 Hessian call budget exhausted "
                f"({self.hessian_call_count}/{hessian_budget})."
            )
        self.hessian_call_count += 1
        self._validate_live_identity(atoms)
        sidecar = self._new_sidecar()
        before = self._snapshot_calculator_state()
        caller_positions = np.array(atoms.get_positions(), copy=True)
        caller_calculator = getattr(atoms, "calc", None)
        evaluation_start = self.evaluation_count
        diagnostics: dict[str, object] = {
            "schema_version": "aimnet2-experimental-richardson-hessian-v2",
            "workflow_id": self.experimental_workflow_id,
            "status": "running",
            "serial": self._hessian_serial,
            "hessian_call_count": self.hessian_call_count,
            "hessian_call_budget": hessian_budget,
            "evaluation_budget_absolute": self._validated_live_budget(
                "evaluation_budget"
            ),
            "sidecar_path": None if sidecar is None else str(sidecar),
            "steps_angstrom": list(_HESSIAN_STEPS_ANGSTROM_V2),
            "fixed_budgets": {
                "richardson_refinement_ratio_max": (
                    _RICHARDSON_REFINEMENT_RATIO_MAX_V2
                ),
                "plateau_max_eV_per_A2": _REFINEMENT_PLATEAU_EV_PER_A2,
                "richardson_error_max_eV_per_A2": _RICHARDSON_ERROR_MAX_EV_PER_A2,
                "raw_finest_antisymmetry_max_eV_per_A2": (_ANTISYMMETRY_MAX_EV_PER_A2),
                "raw_K2_antisymmetry_max_eV_per_A2": (_ANTISYMMETRY_MAX_EV_PER_A2),
                "center_replay_energy_max_eV": _CENTER_REPLAY_ENERGY_MAX_EV,
                "center_replay_force_max_eV_per_A": (_CENTER_REPLAY_FORCE_MAX_EV_PER_A),
            },
            "provenance": copy.deepcopy(self.provenance),
            "endpoints": [],
        }
        raw_hessian_records: list[object] = []
        richardson_records: list[object] = []
        diagnostics["raw_hessians_eV_per_A2"] = raw_hessian_records
        diagnostics["raw_richardson_hessians_eV_per_A2"] = richardson_records
        result: np.ndarray | None = None
        failure: BaseException | None = None
        try:
            count = len(atoms)
            coordinates = 3 * count
            expected = self.hessian_evaluations_per_atom * count + 2
            center = atoms.copy()
            center.calc = None
            self.last_domain_guard = None
            self.last_evaluation = None
            try:
                self.calculate(center, ("energy", "forces"), all_changes)
            except BaseException as exc:
                diagnostics["center_preflight"] = {
                    "status": "failed",
                    "positions_angstrom": center.get_positions(),
                    "guard": copy.deepcopy(self.last_domain_guard),
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                    "raw_evaluation": copy.deepcopy(self.last_attempted_evaluation),
                }
                raise
            center_state = self._snapshot_calculator_state()
            center_energy = float(self.results["energy"])
            center_forces = np.array(self.results["forces"], copy=True)
            diagnostics["center_preflight"] = {
                "status": "passed",
                "energy_eV": center_energy,
                "forces_eV_per_A": center_forces,
                "guard": copy.deepcopy(self.last_domain_guard),
            }
            raw_hessians: list[np.ndarray] = []
            endpoints = diagnostics["endpoints"]
            assert isinstance(endpoints, list)
            for step in _HESSIAN_STEPS_ANGSTROM_V2:
                raw = np.empty((coordinates, coordinates), dtype=float)
                for coordinate in self._coordinate_indices(coordinates):
                    endpoint_forces: dict[int, np.ndarray] = {}
                    for sign in (1, -1):
                        trial = center.copy()
                        positions = np.array(caller_positions, copy=True).reshape(-1)
                        positions[coordinate] += sign * step
                        trial.set_positions(positions.reshape((count, 3)))
                        trial.calc = None
                        self._restore_center_acceptance(center_state)
                        self.last_domain_guard = None
                        self.last_evaluation = None
                        endpoint = {
                            "step_angstrom": step,
                            "coordinate": coordinate,
                            "sign": sign,
                            "positions_angstrom": trial.get_positions(),
                        }
                        try:
                            self.calculate(trial, ("energy", "forces"), all_changes)
                        except BaseException as exc:
                            endpoint.update(
                                {
                                    "status": "failed",
                                    "guard": copy.deepcopy(self.last_domain_guard),
                                    "failure": {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                    "raw_evaluation": copy.deepcopy(
                                        self.last_attempted_evaluation
                                    ),
                                }
                            )
                            endpoints.append(endpoint)
                            raise
                        force = np.array(self.results["forces"], copy=True)
                        endpoint_forces[sign] = force
                        endpoint.update(
                            {
                                "status": "passed",
                                "energy_eV": float(self.results["energy"]),
                                "forces_eV_per_A": force,
                                "guard": copy.deepcopy(self.last_domain_guard),
                                "scalar_fingerprint_sha256": (
                                    self._current_scalar_fingerprint()
                                ),
                            }
                        )
                        endpoints.append(endpoint)
                    raw[:, coordinate] = -(
                        endpoint_forces[1].reshape(-1) - endpoint_forces[-1].reshape(-1)
                    ) / (2.0 * step)
                if not np.all(np.isfinite(raw)):
                    raise AIMNet2ExperimentalHessianError(
                        "v2 numerical Hessian contains non-finite values."
                    )
                raw_hessians.append(raw)
                raw_hessian_records.append(raw)

            richardson = [
                (4.0 * raw_hessians[index + 1] - raw_hessians[index]) / 3.0
                for index in range(3)
            ]
            if not all(np.all(np.isfinite(matrix)) for matrix in richardson):
                raise AIMNet2ExperimentalHessianError(
                    "v2 Richardson Hessian contains non-finite values."
                )
            richardson_records.extend(richardson)

            self._restore_center_acceptance(center_state)
            self.calculate(center, ("energy", "forces"), all_changes)
            replay_energy = float(self.results["energy"])
            replay_forces = np.array(self.results["forces"], copy=True)
            replay_energy_error = abs(replay_energy - center_energy)
            replay_force_error = float(np.max(np.abs(replay_forces - center_forces)))
            diagnostics["center_replay"] = {
                "status": "passed",
                "energy_eV": replay_energy,
                "forces_eV_per_A": replay_forces,
                "energy_difference_eV": replay_energy - center_energy,
                "energy_absolute_difference_eV": replay_energy_error,
                "force_max_abs_difference_eV_per_A": replay_force_error,
                "guard": copy.deepcopy(self.last_domain_guard),
            }
            used = self.evaluation_count - evaluation_start
            diagnostics["evaluation_attempts"] = used
            diagnostics["evaluation_budget"] = expected
            if used != expected:
                raise AIMNet2ExperimentalHessianError(
                    f"v2 numerical Hessian used {used} evaluations; expected {expected}."
                )

            raw_differences = [
                float(np.max(np.abs(raw_hessians[index + 1] - raw_hessians[index])))
                for index in range(3)
            ]
            e1 = float(np.max(np.abs(richardson[1] - richardson[0])))
            e2 = float(np.max(np.abs(richardson[2] - richardson[1])))
            plateau = e1 <= _REFINEMENT_PLATEAU_EV_PER_A2 and e2 <= (
                _REFINEMENT_PLATEAU_EV_PER_A2
            )
            ratio = e2 / e1 if e1 > 0.0 else (0.0 if e2 == 0.0 else float("inf"))
            ratio_gate = plateau or ratio <= _RICHARDSON_REFINEMENT_RATIO_MAX_V2
            richardson_error = e2 / 15.0
            finest_antisymmetry = float(
                np.max(np.abs(raw_hessians[3] - raw_hessians[3].T))
            )
            k2_antisymmetry = float(np.max(np.abs(richardson[2] - richardson[2].T)))
            gates = {
                "richardson_refinement_ratio_or_plateau": ratio_gate,
                "richardson_error": (
                    richardson_error <= _RICHARDSON_ERROR_MAX_EV_PER_A2
                ),
                "raw_finest_antisymmetry": (
                    finest_antisymmetry <= _ANTISYMMETRY_MAX_EV_PER_A2
                ),
                "raw_K2_antisymmetry": (k2_antisymmetry <= _ANTISYMMETRY_MAX_EV_PER_A2),
                "center_replay": (
                    replay_energy_error <= _CENTER_REPLAY_ENERGY_MAX_EV
                    and replay_force_error <= _CENTER_REPLAY_FORCE_MAX_EV_PER_A
                ),
                "evaluation_budget_exact": True,
            }
            diagnostics["raw_second_order_diagnostics"] = {
                "successive_max_abs_differences_eV_per_A2": raw_differences,
                "successive_ratios": [
                    (
                        raw_differences[index + 1] / raw_differences[index]
                        if raw_differences[index] > 0.0
                        else (
                            0.0 if raw_differences[index + 1] == 0.0 else float("inf")
                        )
                    )
                    for index in range(2)
                ],
            }
            diagnostics["richardson_refinement"] = {
                "E1_eV_per_A2": e1,
                "E2_eV_per_A2": e2,
                "E2_over_E1": ratio,
                "plateau": plateau,
                "E2_over_15_eV_per_A2": richardson_error,
                "raw_finest_antisymmetry_eV_per_A2": finest_antisymmetry,
                "raw_K2_antisymmetry_eV_per_A2": k2_antisymmetry,
            }
            diagnostics["gates"] = gates
            if not all(gates.values()):
                failed = ", ".join(name for name, passed in gates.items() if not passed)
                raise AIMNet2ExperimentalHessianError(
                    "v2 numerical Hessian gate failed: " + failed + "."
                )
            result = 0.5 * (richardson[2] + richardson[2].T)
            diagnostics["returned_hessian_eV_per_A2"] = result
            diagnostics["status"] = "passed"
        except BaseException as exc:
            failure = exc
            diagnostics["status"] = "failed"
            diagnostics["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            diagnostics.setdefault(
                "evaluation_attempts", self.evaluation_count - evaluation_start
            )
        finally:
            if not np.array_equal(atoms.get_positions(), caller_positions):
                atoms.set_positions(caller_positions)
            if getattr(atoms, "calc", None) is not caller_calculator:
                atoms.calc = caller_calculator
            self._restore_calculator_state(before)
            diagnostics["evaluation_count_total"] = self.evaluation_count
            safe_diagnostics = _json_safe(diagnostics)
            if not isinstance(safe_diagnostics, dict):
                raise TypeError(
                    "internal v2 Hessian diagnostics are not a JSON object."
                )
            self.last_hessian_diagnostics = safe_diagnostics
            if sidecar is not None:
                try:
                    self._write_sidecar(sidecar, diagnostics)
                except BaseException as exc:
                    self.last_hessian_diagnostics["status"] = "failed"
                    self.last_hessian_diagnostics["sidecar_failure"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    if failure is None:
                        failure = exc
        if failure is not None:
            raise failure
        assert result is not None
        return result


__all__ = [
    "AIMNet2ExperimentalCalculatorV2",
    "EXPERIMENTAL_WORKFLOW_ID_V2",
]
