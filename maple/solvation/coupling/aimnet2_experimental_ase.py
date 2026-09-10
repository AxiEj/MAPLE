"""Explicitly experimental ASE workflows for the AIMNet2 SMD total scalar.

This module does not admit the underlying scalar or profile.  It adds a
source-bound numerical Hessian to the existing guarded energy/force bridge for
one exact opt-in workflow.  All finite-difference evidence is retained before
the terminal matrix is symmetrised.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .geometry_mediated_ase import GeometryMediatedScalarASECalculator

EXPERIMENTAL_WORKFLOW_ID = "aimnet2-smooth-ddpcm-experimental-workflows-v1"
_HESSIAN_STEPS_ANGSTROM = (4.0e-4, 2.0e-4, 1.0e-4)
_REFINEMENT_RATIO_MAX = 0.55
_REFINEMENT_PLATEAU_EV_PER_A2 = 1.0e-7
_RICHARDSON_ERROR_MAX_EV_PER_A2 = 5.0e-5
_ANTISYMMETRY_MAX_EV_PER_A2 = 1.0e-4
_CENTER_REPLAY_ENERGY_MAX_EV = 1.0e-10
_CENTER_REPLAY_FORCE_MAX_EV_PER_A = 1.0e-9


class AIMNet2ExperimentalHessianError(RuntimeError):
    """A fixed numerical-Hessian gate failed closed."""


class AIMNet2ExperimentalBudgetError(RuntimeError):
    """An absolute experimental workflow evaluation budget was exhausted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _json_safe(as_dict())
    return repr(value)


def _copy_state_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return copy.deepcopy(value)


def _attribute_value(owner: object, name: str) -> object:
    value = getattr(owner, name, None)
    return value() if callable(value) else value


class AIMNet2ExperimentalCalculator(GeometryMediatedScalarASECalculator):
    """Guarded E/F calculator with one fixed raw-FD Hessian protocol."""

    experimental_workflow_id = EXPERIMENTAL_WORKFLOW_ID

    def __init__(
        self,
        scalar: object,
        *,
        checkpoint_path: str | Path,
        output: str | Path | None = None,
        evaluation_budget: int | None = None,
        hessian_call_budget: int | None = None,
    ) -> None:
        fingerprint = getattr(scalar, "fingerprint_sha256", None)
        if not callable(fingerprint):
            raise TypeError(
                "experimental AIMNet2 scalar must expose fingerprint_sha256()."
            )
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
        self.output = None if output is None else Path(output)
        self.evaluation_count = 0
        self.last_attempted_evaluation: dict[str, object] | None = None
        self.hessian_call_count = 0
        self.evaluation_budget = self._optional_budget(
            evaluation_budget, name="evaluation_budget"
        )
        self.hessian_call_budget = self._optional_budget(
            hessian_call_budget, name="hessian_call_budget"
        )
        self.last_hessian_diagnostics: dict[str, object] | None = None
        self._hessian_serial = 0
        self._initial_scalar = scalar
        self._initial_model = getattr(scalar, "model", None)
        self._initial_continuum = getattr(scalar, "continuum", None)
        self._initial_scalar_id = str(getattr(scalar, "scalar_id", ""))
        self._initial_profile_id = str(getattr(scalar, "profile_id", ""))
        if not self._initial_scalar_id or not self._initial_profile_id:
            raise ValueError("experimental AIMNet2 scalar/profile identity is missing.")
        self._initial_scalar_fingerprint = str(fingerprint())
        self._checkpoint_size_bytes = self.checkpoint_path.stat().st_size
        self._checkpoint_sha256 = _sha256_file(self.checkpoint_path)
        self.provenance = self._capture_provenance(scalar)
        self._validate_checkpoint_contract(scalar)
        super().__init__(scalar)

    @staticmethod
    def _optional_budget(value: int | None, *, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer or None.")
        return value

    def _validated_live_budget(self, name: str) -> int | None:
        return self._optional_budget(getattr(self, name), name=name)

    def _capture_provenance(self, scalar: object) -> dict[str, object]:
        model = getattr(scalar, "model", None)
        provenance = getattr(model, "provenance", None)
        metadata = getattr(provenance, "metadata", None)
        if callable(metadata):
            provenance = metadata()
        elif hasattr(provenance, "__dict__"):
            provenance = vars(provenance)
        return {
            "workflow_id": self.experimental_workflow_id,
            "scalar_id": self._initial_scalar_id,
            "profile_id": self._initial_profile_id,
            "scalar_fingerprint_sha256": self._initial_scalar_fingerprint,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha256,
            "checkpoint_size_bytes": self._checkpoint_size_bytes,
            "model_provenance": _json_safe(provenance),
            "model_configuration_sha256": _json_safe(
                _attribute_value(model, "configuration_sha256")
            ),
            "model_provenance_sha256": _json_safe(
                _attribute_value(model, "provenance_sha256")
            ),
        }

    def _validate_checkpoint_contract(self, scalar: object | None = None) -> None:
        scalar = self.scalar if scalar is None else scalar
        model = getattr(scalar, "model", None)
        contract = getattr(model, "checkpoint_contract", None)
        if contract is None:
            raise TypeError(
                "experimental AIMNet2 model must expose checkpoint_contract."
            )
        if (
            int(getattr(contract, "checkpoint_size_bytes", -1))
            != self._checkpoint_size_bytes
            or str(getattr(contract, "checkpoint_sha256", "")).lower()
            != self._checkpoint_sha256
        ):
            raise ValueError("checkpoint does not match the frozen AIMNet2 contract.")
        model_path = getattr(model, "_checkpoint_path", None)
        calculator = getattr(model, "_calculator", None)
        if model_path is None and calculator is not None:
            model_path = getattr(calculator, "model_path", None)
        if (
            model_path is not None
            and Path(model_path).resolve() != self.checkpoint_path
        ):
            raise ValueError("scalar checkpoint path differs from checkpoint_path.")

    @staticmethod
    def _validate_atoms_metadata(atoms: object) -> None:
        info = getattr(atoms, "info", None)
        if not isinstance(info, Mapping):
            raise TypeError("experimental AIMNet2 atoms.info must be a mapping.")
        if info.get("charge") != 0 or info.get("mult") != 1:
            raise ValueError(
                "experimental AIMNet2 workflows require charge=0 and mult=1."
            )
        constraints = getattr(atoms, "constraints", ())
        if constraints is not None and len(constraints) != 0:
            raise ValueError(
                "experimental AIMNet2 workflows do not support constraints."
            )

    def _validate_live_identity(self, atoms: object) -> None:
        self._validate_atoms_metadata(atoms)
        self._require_nonperiodic(atoms)
        if self.scalar is not self._initial_scalar:
            raise ValueError("experimental AIMNet2 scalar object was replaced.")
        if getattr(self.scalar, "model", None) is not self._initial_model:
            raise ValueError("experimental AIMNet2 model source was replaced.")
        if getattr(self.scalar, "continuum", None) is not self._initial_continuum:
            raise ValueError("experimental AIMNet2 continuum source was replaced.")
        domain_validator = getattr(getattr(self.scalar, "model", None), "domain", None)
        validate_atoms = getattr(domain_validator, "validate_atoms", None)
        if not callable(validate_atoms):
            raise TypeError(
                "experimental AIMNet2 model must expose domain.validate_atoms()."
            )
        validate_atoms(atoms)
        if str(getattr(self.scalar, "scalar_id", "")) != self._initial_scalar_id:
            raise ValueError("experimental AIMNet2 scalar identity changed.")
        if str(getattr(self.scalar, "profile_id", "")) != self._initial_profile_id:
            raise ValueError("experimental AIMNet2 profile identity changed.")
        current = self._current_scalar_fingerprint()
        if current != self._initial_scalar_fingerprint:
            raise ValueError("experimental AIMNet2 scalar fingerprint changed.")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("experimental AIMNet2 checkpoint is unavailable.")
        if (
            self.checkpoint_path.stat().st_size != self._checkpoint_size_bytes
            or _sha256_file(self.checkpoint_path) != self._checkpoint_sha256
        ):
            raise ValueError("experimental AIMNet2 checkpoint bytes changed.")
        self._validate_checkpoint_contract()

    def _current_scalar_fingerprint(self) -> str:
        fingerprint = getattr(self.scalar, "fingerprint_sha256", None)
        if not callable(fingerprint):
            raise TypeError("experimental AIMNet2 scalar fingerprint is unavailable.")
        return str(fingerprint())

    def get_property(self, name, atoms=None, allow_calculation=True):
        target = self.atoms if atoms is None else atoms
        if target is None:
            raise ValueError("an ASE geometry is required.")
        self._validate_live_identity(target)
        return super().get_property(name, atoms, allow_calculation)

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        self.last_attempted_evaluation = None
        target = self.atoms if atoms is None else atoms
        if target is None:
            raise ValueError("an ASE geometry is required.")
        self._validate_live_identity(target)
        budget = self._validated_live_budget("evaluation_budget")
        if budget is not None and self.evaluation_count >= budget:
            raise AIMNet2ExperimentalBudgetError(
                "experimental AIMNet2 E/F evaluation budget exhausted "
                f"({self.evaluation_count}/{budget})."
            )
        self.evaluation_count += 1
        self._require_supported_properties(properties)
        Calculator.calculate(self, atoms, properties, system_changes)
        positions, numbers, model_topology, continuum_topology = self._guard_geometry(
            self.atoms
        )
        evaluator = getattr(self.scalar, "evaluate", None)
        if not callable(evaluator):
            raise TypeError("experimental AIMNet2 scalar evaluate() is unavailable.")
        evaluation = evaluator(self.atoms)
        raw_energy = getattr(
            getattr(evaluation, "energy", None), "total_energy_eV", None
        )
        raw_forces = getattr(evaluation, "forces_eV_per_A", None)
        attempted = _json_safe(
            {
                "evaluation_type": (
                    f"{type(evaluation).__module__}.{type(evaluation).__qualname__}"
                ),
                "energy_total_eV": raw_energy,
                "forces_eV_per_A": raw_forces,
            }
        )
        if not isinstance(attempted, dict):
            raise TypeError(
                "internal attempted-evaluation ledger is not a JSON object."
            )
        self.last_attempted_evaluation = attempted
        if raw_energy is None:
            raise ValueError(
                "geometry-mediated scalar returned an invalid energy/force ledger."
            )
        energy = float(raw_energy)
        forces = np.asarray(raw_forces, dtype=float)
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

    def _snapshot_calculator_state(self) -> dict[str, object]:
        names = {
            "atoms",
            "results",
            "last_domain_guard",
            "last_evaluation",
            "_accepted_positions",
            "_accepted_atomic_numbers",
            "_accepted_model_topology",
            "_accepted_continuum_topology",
        }
        names.update(name for name in vars(self) if "history" in name.lower())
        return {
            name: (
                getattr(self, name)
                if name == "last_evaluation"
                else _copy_state_value(getattr(self, name))
            )
            for name in names
            if hasattr(self, name)
        }

    def _restore_calculator_state(self, state: Mapping[str, object]) -> None:
        for name, value in state.items():
            setattr(
                self,
                name,
                value if name == "last_evaluation" else _copy_state_value(value),
            )

    def _restore_center_acceptance(self, state: Mapping[str, object]) -> None:
        for name in (
            "_accepted_positions",
            "_accepted_atomic_numbers",
            "_accepted_model_topology",
            "_accepted_continuum_topology",
        ):
            setattr(self, name, _copy_state_value(state[name]))
        self.atoms = _copy_state_value(state["atoms"])
        self.results = {}

    def _new_sidecar(self) -> Path | None:
        if self.output is None:
            self._hessian_serial += 1
            return None
        while True:
            self._hessian_serial += 1
            candidate = Path(f"{self.output}.hessian-{self._hessian_serial:04d}.json")
            if not candidate.exists():
                return candidate

    def _coordinate_indices(self, coordinate_count: int):
        """Return the fixed Cartesian column order (test-overridable only)."""

        return range(coordinate_count)

    @staticmethod
    def _write_sidecar(path: Path, diagnostics: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_json_safe(diagnostics), indent=2, sort_keys=True) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def get_hessian(self, atoms) -> np.ndarray:
        """Return a gated numerical total Hessian in eV/Angstrom**2."""

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
            "schema_version": "aimnet2-experimental-raw-force-hessian-v1",
            "workflow_id": self.experimental_workflow_id,
            "status": "running",
            "serial": self._hessian_serial,
            "hessian_call_count": self.hessian_call_count,
            "hessian_call_budget": hessian_budget,
            "evaluation_budget_absolute": self._validated_live_budget(
                "evaluation_budget"
            ),
            "sidecar_path": None if sidecar is None else str(sidecar),
            "steps_angstrom": list(_HESSIAN_STEPS_ANGSTROM),
            "fixed_budgets": {
                "refinement_ratio_max": _REFINEMENT_RATIO_MAX,
                "plateau_max_eV_per_A2": _REFINEMENT_PLATEAU_EV_PER_A2,
                "richardson_error_max_eV_per_A2": _RICHARDSON_ERROR_MAX_EV_PER_A2,
                "terminal_antisymmetry_max_eV_per_A2": _ANTISYMMETRY_MAX_EV_PER_A2,
                "center_replay_energy_max_eV": _CENTER_REPLAY_ENERGY_MAX_EV,
                "center_replay_force_max_eV_per_A": (_CENTER_REPLAY_FORCE_MAX_EV_PER_A),
            },
            "provenance": copy.deepcopy(self.provenance),
            "endpoints": [],
        }
        raw_hessian_records: list[object] = []
        diagnostics["raw_hessians_eV_per_A2"] = raw_hessian_records
        result: np.ndarray | None = None
        failure: BaseException | None = None
        try:
            count = len(atoms)
            coordinates = 3 * count
            expected = 18 * count + 2
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
                "energy_eV": center_energy,
                "forces_eV_per_A": center_forces,
                "guard": copy.deepcopy(self.last_domain_guard),
            }
            raw_hessians: list[np.ndarray] = []
            endpoints = diagnostics["endpoints"]
            assert isinstance(endpoints, list)
            for step in _HESSIAN_STEPS_ANGSTROM:
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
                        "numerical Hessian contains non-finite values."
                    )
                raw_hessians.append(raw)
                raw_hessian_records.append(raw)

            self._restore_center_acceptance(center_state)
            self.calculate(center, ("energy", "forces"), all_changes)
            replay_energy = float(self.results["energy"])
            replay_forces = np.array(self.results["forces"], copy=True)
            replay_energy_error = abs(replay_energy - center_energy)
            replay_force_error = float(np.max(np.abs(replay_forces - center_forces)))
            diagnostics["center_replay"] = {
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
                    f"numerical Hessian used {used} evaluations; expected {expected}."
                )

            d1 = float(np.max(np.abs(raw_hessians[1] - raw_hessians[0])))
            d2 = float(np.max(np.abs(raw_hessians[2] - raw_hessians[1])))
            plateau = d1 <= _REFINEMENT_PLATEAU_EV_PER_A2 and d2 <= (
                _REFINEMENT_PLATEAU_EV_PER_A2
            )
            ratio = d2 / d1 if d1 > 0.0 else (0.0 if d2 == 0.0 else float("inf"))
            ratio_gate = plateau or ratio <= _REFINEMENT_RATIO_MAX
            richardson_error = d2 / 3.0
            antisymmetry = float(np.max(np.abs(raw_hessians[2] - raw_hessians[2].T)))
            gates = {
                "refinement_ratio_or_plateau": ratio_gate,
                "richardson_error": (
                    richardson_error <= _RICHARDSON_ERROR_MAX_EV_PER_A2
                ),
                "terminal_raw_antisymmetry": (
                    antisymmetry <= _ANTISYMMETRY_MAX_EV_PER_A2
                ),
                "center_replay": (
                    replay_energy_error <= _CENTER_REPLAY_ENERGY_MAX_EV
                    and replay_force_error <= _CENTER_REPLAY_FORCE_MAX_EV_PER_A
                ),
                "evaluation_budget_exact": True,
            }
            diagnostics["refinement"] = {
                "D1_eV_per_A2": d1,
                "D2_eV_per_A2": d2,
                "D2_over_D1": ratio,
                "plateau": plateau,
                "D2_over_3_eV_per_A2": richardson_error,
                "terminal_raw_antisymmetry_eV_per_A2": antisymmetry,
            }
            diagnostics["gates"] = gates
            if not all(gates.values()):
                failed = ", ".join(name for name, passed in gates.items() if not passed)
                raise AIMNet2ExperimentalHessianError(
                    "numerical Hessian gate failed: " + failed + "."
                )
            result = 0.5 * (raw_hessians[2] + raw_hessians[2].T)
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
                raise TypeError("internal Hessian diagnostics are not a JSON object.")
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
    "AIMNet2ExperimentalBudgetError",
    "AIMNet2ExperimentalCalculator",
    "AIMNet2ExperimentalHessianError",
    "EXPERIMENTAL_WORKFLOW_ID",
]
