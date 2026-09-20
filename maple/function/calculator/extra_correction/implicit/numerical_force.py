"""Testing-only numerical forces for scalar implicit-solvent providers.

The adapter in this module deliberately differentiates the wrapped provider's
reported total scalar.  It is not a native-force implementation and must not be
used as a production-admission signal.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np

from .result import SolvationResult


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class EvaluationAuditContext:
    """Immutable identity and pre-created directory for one scalar attempt."""

    run_id: str
    force_call_id: str
    scalar_ordinal: int
    coordinate_index: int | None
    sign: int
    step_role: str
    step_angstrom: float
    path: Path


class NumericalForceBudget:
    """Thread-safe lifetime ledger for numerical-force work."""

    def __init__(
        self,
        *,
        max_scalar_evaluations: int,
        max_raw_records: int,
        max_audit_bytes: int,
    ) -> None:
        self.max_scalar_evaluations = _positive_integer(
            max_scalar_evaluations, name="max_scalar_evaluations"
        )
        self.max_raw_records = _positive_integer(
            max_raw_records, name="max_raw_records"
        )
        self.max_audit_bytes = _positive_integer(
            max_audit_bytes, name="max_audit_bytes"
        )
        self._lock = Lock()
        self._reserved_scalar_evaluations = 0
        self._reserved_raw_records = 0
        self._reserved_audit_bytes = 0
        self._consumed_scalar_evaluations = 0
        self._consumed_raw_records = 0
        self._consumed_audit_bytes = 0

    @property
    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_scalar_evaluations": self.max_scalar_evaluations,
                "max_raw_records": self.max_raw_records,
                "max_audit_bytes": self.max_audit_bytes,
                "reserved_scalar_evaluations": self._reserved_scalar_evaluations,
                "reserved_raw_records": self._reserved_raw_records,
                "reserved_audit_bytes": self._reserved_audit_bytes,
                "consumed_scalar_evaluations": self._consumed_scalar_evaluations,
                "consumed_raw_records": self._consumed_raw_records,
                "consumed_audit_bytes": self._consumed_audit_bytes,
            }

    def reserve_complete_force(
        self, *, atom_count: int, audit_bytes: int = 0
    ) -> "_BudgetReservation":
        atom_count = _positive_integer(atom_count, name="atom_count")
        scalar_evaluations = 1 + 12 * atom_count
        return self._reserve(
            scalar_evaluations=scalar_evaluations,
            raw_records=scalar_evaluations + 1,
            audit_bytes=audit_bytes,
        )

    def reserve_task(
        self, *, scalar_evaluations: int, raw_records: int, audit_bytes: int
    ) -> "_TaskReservation":
        reservation = self._reserve(
            scalar_evaluations=scalar_evaluations,
            raw_records=raw_records,
            audit_bytes=audit_bytes,
            reservation_type=_TaskReservation,
        )
        assert isinstance(reservation, _TaskReservation)
        return reservation

    def _reserve(
        self,
        *,
        scalar_evaluations: int,
        raw_records: int,
        audit_bytes: int,
        reservation_type: type["_BudgetReservation"] = None,
    ) -> "_BudgetReservation":
        scalar_evaluations = _positive_integer(
            scalar_evaluations, name="scalar_evaluations"
        )
        raw_records = _positive_integer(raw_records, name="raw_records")
        audit_bytes = _nonnegative_integer(audit_bytes, name="audit_bytes")
        reservation_type = reservation_type or _BudgetReservation
        with self._lock:
            checks = (
                (
                    "scalar evaluation",
                    scalar_evaluations,
                    self._reserved_scalar_evaluations,
                    self._consumed_scalar_evaluations,
                    self.max_scalar_evaluations,
                ),
                (
                    "raw-record",
                    raw_records,
                    self._reserved_raw_records,
                    self._consumed_raw_records,
                    self.max_raw_records,
                ),
                (
                    "audit-byte",
                    audit_bytes,
                    self._reserved_audit_bytes,
                    self._consumed_audit_bytes,
                    self.max_audit_bytes,
                ),
            )
            for label, requested, reserved, consumed, maximum in checks:
                if requested > maximum - reserved - consumed:
                    raise RuntimeError(
                        f"{label} budget cannot reserve the complete operation"
                    )
            self._reserved_scalar_evaluations += scalar_evaluations
            self._reserved_raw_records += raw_records
            self._reserved_audit_bytes += audit_bytes
        return reservation_type(
            self,
            scalar_evaluations=scalar_evaluations,
            raw_records=raw_records,
            audit_bytes=audit_bytes,
        )

    def _finish(
        self,
        reservation: "_BudgetReservation",
        *,
        consumed_scalar_evaluations: int,
        consumed_raw_records: int,
        consumed_audit_bytes: int,
    ) -> None:
        with self._lock:
            self._reserved_scalar_evaluations -= reservation.scalar_evaluations
            self._reserved_raw_records -= reservation.raw_records
            self._reserved_audit_bytes -= reservation.audit_bytes
            self._consumed_scalar_evaluations += consumed_scalar_evaluations
            self._consumed_raw_records += consumed_raw_records
            self._consumed_audit_bytes += consumed_audit_bytes


class _BudgetReservation:
    def __init__(
        self,
        budget: NumericalForceBudget,
        *,
        scalar_evaluations: int,
        raw_records: int,
        audit_bytes: int,
    ) -> None:
        self._budget = budget
        self.scalar_evaluations = scalar_evaluations
        self.raw_records = raw_records
        self.audit_bytes = audit_bytes
        self._lock = Lock()
        self._launched = False
        self._closed = False

    def mark_provider_launch(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot launch against a closed reservation")
            self._launched = True

    def complete(
        self,
        *,
        actual_audit_bytes: int,
        actual_scalar_evaluations: int | None = None,
        actual_raw_records: int | None = None,
    ) -> None:
        actual_audit_bytes = _nonnegative_integer(
            actual_audit_bytes, name="actual_audit_bytes"
        )
        actual_scalar_evaluations = (
            self.scalar_evaluations
            if actual_scalar_evaluations is None
            else _nonnegative_integer(
                actual_scalar_evaluations, name="actual_scalar_evaluations"
            )
        )
        actual_raw_records = (
            self.raw_records
            if actual_raw_records is None
            else _nonnegative_integer(actual_raw_records, name="actual_raw_records")
        )
        if (
            actual_scalar_evaluations > self.scalar_evaluations
            or actual_raw_records > self.raw_records
            or actual_audit_bytes > self.audit_bytes
        ):
            raise RuntimeError("actual operation usage exceeds its reserved budget")
        with self._lock:
            if self._closed:
                raise RuntimeError("reservation is already closed")
            self._closed = True
        self._budget._finish(
            self,
            consumed_scalar_evaluations=actual_scalar_evaluations,
            consumed_raw_records=actual_raw_records,
            consumed_audit_bytes=actual_audit_bytes,
        )

    def abort(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            launched = self._launched
        self._budget._finish(
            self,
            consumed_scalar_evaluations=self.scalar_evaluations if launched else 0,
            consumed_raw_records=self.raw_records if launched else 0,
            consumed_audit_bytes=self.audit_bytes if launched else 0,
        )


class _TaskReservation(_BudgetReservation):
    """Parent reservation whose child forces consume no additional global slots."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._child_scalar_evaluations = 0
        self._child_raw_records = 0
        self._child_audit_bytes = 0

    def reserve_complete_force(
        self, *, atom_count: int, audit_bytes: int = 0
    ) -> "_ChildReservation":
        atom_count = _positive_integer(atom_count, name="atom_count")
        audit_bytes = _nonnegative_integer(audit_bytes, name="audit_bytes")
        scalar_evaluations = 1 + 12 * atom_count
        raw_records = scalar_evaluations + 1
        with self._lock:
            if self._closed:
                raise RuntimeError("task reservation is already closed")
            if (
                self._child_scalar_evaluations + scalar_evaluations
                > self.scalar_evaluations
                or self._child_raw_records + raw_records > self.raw_records
                or self._child_audit_bytes + audit_bytes > self.audit_bytes
            ):
                raise RuntimeError("child force exceeds its task reservation")
            self._child_scalar_evaluations += scalar_evaluations
            self._child_raw_records += raw_records
            self._child_audit_bytes += audit_bytes
        return _ChildReservation(
            self,
            scalar_evaluations=scalar_evaluations,
            raw_records=raw_records,
            audit_bytes=audit_bytes,
        )

    def reconcile(
        self,
        *,
        actual_scalar_evaluations: int,
        actual_raw_records: int,
        actual_audit_bytes: int,
    ) -> None:
        self.complete(
            actual_scalar_evaluations=actual_scalar_evaluations,
            actual_raw_records=actual_raw_records,
            actual_audit_bytes=actual_audit_bytes,
        )


class _ChildReservation:
    def __init__(
        self,
        parent: _TaskReservation,
        *,
        scalar_evaluations: int,
        raw_records: int,
        audit_bytes: int,
    ) -> None:
        self._parent = parent
        self.scalar_evaluations = scalar_evaluations
        self.raw_records = raw_records
        self.audit_bytes = audit_bytes
        self._closed = False
        self._launched = False

    def mark_provider_launch(self) -> None:
        if self._closed:
            raise RuntimeError("cannot launch against a closed child reservation")
        self._launched = True
        self._parent.mark_provider_launch()

    def complete(self, *, actual_audit_bytes: int, **_: Any) -> None:
        actual_audit_bytes = _nonnegative_integer(
            actual_audit_bytes, name="actual_audit_bytes"
        )
        if actual_audit_bytes > self.audit_bytes:
            raise RuntimeError("actual child usage exceeds its reserved budget")
        if self._closed:
            raise RuntimeError("child reservation is already closed")
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True


@dataclass
class _ActiveForceOperation:
    reservation: _TaskReservation
    atom_count: int
    force_call_count: int
    operation: str
    lock: Lock
    claimed_force_calls: int = 0
    completed_force_calls: int = 0
    actual_audit_bytes: int = 0

    def claim_child(self) -> _ChildReservation:
        with self.lock:
            if self.claimed_force_calls >= self.force_call_count:
                raise RuntimeError(
                    f"{self.operation} exceeded its reserved force-call count"
                )
            self.claimed_force_calls += 1
        return self.reservation.reserve_complete_force(
            atom_count=self.atom_count, audit_bytes=0
        )

    def record_completion(self, actual_audit_bytes: int) -> None:
        with self.lock:
            self.completed_force_calls += 1
            self.actual_audit_bytes += actual_audit_bytes


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True, default=_json_default).encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


class NumericalForceProvider:
    """Expose centered finite-difference forces from an energy-only provider."""

    supported_properties = frozenset({"energy", "forces"})
    testing_only = True
    native_force = False
    derivative_source = "centered-finite-difference-of-selected-provider-total"

    def __init__(
        self,
        provider,
        *,
        force_step_angstrom: float,
        force_check_step_angstrom: float,
        max_scalar_evaluations: int,
        max_raw_records: int,
        max_audit_bytes: int,
        audit_dir: str | Path,
    ) -> None:
        primary = self._validate_step(force_step_angstrom, "force_step_angstrom")
        check = self._validate_step(
            force_check_step_angstrom, "force_check_step_angstrom"
        )
        if check >= primary:
            raise ValueError(
                "force_check_step_angstrom must be smaller than force_step_angstrom"
            )
        if "energy" not in getattr(provider, "supported_properties", ()):
            raise TypeError("numerical-force provider must expose scalar energy")
        self.provider = provider
        self.force_step_angstrom = primary
        self.force_check_step_angstrom = check
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid4().hex
        self._run_dir = self.audit_dir / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=False)
        self._identity_lock = Lock()
        self._force_sequence = 0
        self._active_operation: ContextVar[_ActiveForceOperation | None] = ContextVar(
            f"numerical_force_operation_{id(self)}", default=None
        )
        self.budget = NumericalForceBudget(
            max_scalar_evaluations=max_scalar_evaluations,
            max_raw_records=max_raw_records,
            max_audit_bytes=max_audit_bytes,
        )

    @staticmethod
    def _validate_step(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be finite and positive")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @property
    def budget_snapshot(self) -> dict[str, int]:
        return self.budget.snapshot

    @staticmethod
    def force_operation_cost(
        *, atom_count: int, force_call_count: int
    ) -> dict[str, int]:
        """Return exact scalar and logical-record costs for known force calls."""
        atom_count = _positive_integer(atom_count, name="atom_count")
        force_call_count = _positive_integer(force_call_count, name="force_call_count")
        scalar_per_force = 1 + 12 * atom_count
        raw_records_per_force = scalar_per_force + 1
        return {
            "atom_count": atom_count,
            "force_call_count": force_call_count,
            "scalar_evaluations_per_force": scalar_per_force,
            "raw_records_per_force": raw_records_per_force,
            "scalar_evaluations": force_call_count * scalar_per_force,
            "raw_records": force_call_count * raw_records_per_force,
        }

    @contextmanager
    def reserve_force_operation(
        self, *, atom_count: int, force_call_count: int, operation: str
    ):
        """Atomically reserve and reconcile a known multi-force operation.

        Every force evaluation in the context consumes a child slice of this
        reservation.  A launched failure consumes the complete parent
        reservation; an exception before any provider launch releases it.
        """
        if self._active_operation.get() is not None:
            raise RuntimeError("nested numerical force operations are not supported")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        cost = self.force_operation_cost(
            atom_count=atom_count, force_call_count=force_call_count
        )
        declared_bytes = self._declared_audit_bytes(cost["scalar_evaluations"])
        reservation = self.budget.reserve_task(
            scalar_evaluations=cost["scalar_evaluations"],
            raw_records=cost["raw_records"],
            audit_bytes=declared_bytes,
        )
        state = _ActiveForceOperation(
            reservation=reservation,
            atom_count=atom_count,
            force_call_count=force_call_count,
            operation=operation.strip(),
            lock=Lock(),
        )
        token = self._active_operation.set(state)
        try:
            yield self
            if (
                state.claimed_force_calls != force_call_count
                or state.completed_force_calls != force_call_count
            ):
                raise RuntimeError(
                    f"{state.operation} completed "
                    f"{state.completed_force_calls}/{force_call_count} reserved "
                    "force calls"
                )
            reservation.reconcile(
                actual_scalar_evaluations=cost["scalar_evaluations"],
                actual_raw_records=cost["raw_records"],
                actual_audit_bytes=state.actual_audit_bytes,
            )
        except Exception:
            reservation.abort()
            raise
        finally:
            self._active_operation.reset(token)

    @property
    def provenance(self) -> dict[str, Any]:
        underlying = getattr(self.provider, "provenance", {})
        if callable(underlying):
            underlying = underlying()
        return {
            "underlying_provider": underlying,
            "testing_only": True,
            "native_force": False,
            "numerical_force": True,
            "derivative_source": self.derivative_source,
            "force_step_angstrom": self.force_step_angstrom,
            "force_check_step_angstrom": self.force_check_step_angstrom,
            "budget": self.budget_snapshot,
        }

    def _next_force_call(self) -> tuple[str, Path]:
        with self._identity_lock:
            self._force_sequence += 1
            sequence = self._force_sequence
        call_id = f"force-{sequence:08d}-{uuid4().hex}"
        path = self._run_dir / call_id
        path.mkdir(parents=True, exist_ok=False)
        return call_id, path

    def _context(
        self,
        *,
        force_call_id: str,
        force_path: Path,
        scalar_ordinal: int,
        coordinate_index: int | None,
        sign: int,
        step_role: str,
        step_angstrom: float,
    ) -> EvaluationAuditContext:
        label = f"scalar-{scalar_ordinal:06d}-{step_role}"
        if coordinate_index is not None:
            label += f"-c{coordinate_index:06d}-s{sign:+d}"
        path = force_path / label
        path.mkdir(parents=True, exist_ok=False)
        return EvaluationAuditContext(
            run_id=self.run_id,
            force_call_id=force_call_id,
            scalar_ordinal=scalar_ordinal,
            coordinate_index=coordinate_index,
            sign=sign,
            step_role=step_role,
            step_angstrom=step_angstrom,
            path=path,
        )

    @staticmethod
    def _detached_atoms(atoms, positions: np.ndarray):
        detached = atoms.copy()
        detached.calc = None
        detached.set_constraint()
        detached.set_positions(positions)
        return detached

    def _evaluate_scalar(
        self,
        atoms,
        positions: np.ndarray,
        *,
        reservation: _BudgetReservation,
        context: EvaluationAuditContext,
    ) -> SolvationResult:
        detached = self._detached_atoms(atoms, positions)
        reservation.mark_provider_launch()
        result = self.provider.evaluate(
            detached, need_forces=False, audit_context=context
        )
        energy = float(result.energy_hartree)
        if not math.isfinite(energy):
            raise ValueError("selected provider returned a non-finite scalar energy")
        return result

    def _declared_audit_bytes(self, scalar_evaluations: int) -> int:
        # Allocate byte capacity in the same proportion as scalar capacity.  It
        # permits the maximum number of concurrent complete forces allowed by
        # the scalar ledger without letting their aggregate exceed the byte cap.
        return max(
            1,
            math.floor(
                self.budget.max_audit_bytes
                * scalar_evaluations
                / self.budget.max_scalar_evaluations
            ),
        )

    def _preflight_stencils(
        self, atoms, positions: np.ndarray
    ) -> tuple[dict[tuple[str, int], float], list[dict[str, Any]]]:
        """Validate provider-specific coordinate serialization before launches."""
        validate_pair = getattr(self.provider, "validate_serialized_pair", None)
        if not callable(validate_pair):
            return {}, []

        containment = getattr(self.provider, "grid_containment_metadata", None)
        center = self._detached_atoms(atoms, positions)
        if callable(containment):
            containment(center)
        effective_steps: dict[tuple[str, int], float] = {}
        records: list[dict[str, Any]] = []
        flat_positions = positions.reshape(-1)
        for step_role, requested_step in (
            ("primary", self.force_step_angstrom),
            ("check", self.force_check_step_angstrom),
        ):
            for coordinate_index in range(flat_positions.size):
                minus_xyz = flat_positions.copy()
                plus_xyz = flat_positions.copy()
                minus_xyz[coordinate_index] -= requested_step
                plus_xyz[coordinate_index] += requested_step
                minus = self._detached_atoms(atoms, minus_xyz.reshape(positions.shape))
                plus = self._detached_atoms(atoms, plus_xyz.reshape(positions.shape))
                if callable(containment):
                    containment(minus)
                    containment(plus)
                metadata = validate_pair(
                    center,
                    minus,
                    plus,
                    requested_step_angstrom=requested_step,
                )
                actual = np.asarray(
                    metadata["actual_symmetric_step_angstrom"], dtype=np.float64
                ).reshape(-1)
                if actual.size != 1 or not np.isfinite(actual[0]) or actual[0] <= 0.0:
                    raise ValueError(
                        "provider serialized displacement did not yield one finite "
                        "positive symmetric Cartesian step"
                    )
                effective_steps[(step_role, coordinate_index)] = float(actual[0])
                records.append(
                    {
                        "step_role": step_role,
                        "coordinate_index": coordinate_index,
                        "requested_step_angstrom": requested_step,
                        "effective_step_angstrom": float(actual[0]),
                        "provider_serialization": metadata,
                    }
                )
        return effective_steps, records

    def evaluate(
        self, atoms, *, need_forces: bool = False, calculator=None
    ) -> SolvationResult:
        del calculator  # The selected solvent scalar, not a foreign calculator, is used.
        atom_count = len(atoms)
        scalar_count = 1 + 12 * atom_count if need_forces else 1
        raw_count = scalar_count + 1 if need_forces else 1
        active_operation = self._active_operation.get()
        if active_operation is not None:
            if not need_forces:
                raise RuntimeError(
                    "energy-only evaluations cannot be mixed into a reserved "
                    "force operation"
                )
            if atom_count != active_operation.atom_count:
                raise RuntimeError(
                    f"{active_operation.operation} reserved atom_count="
                    f"{active_operation.atom_count}, received {atom_count}"
                )
            reservation = active_operation.claim_child()
        else:
            declared_bytes = self._declared_audit_bytes(scalar_count)
            if need_forces:
                reservation = self.budget.reserve_complete_force(
                    atom_count=atom_count, audit_bytes=declared_bytes
                )
            else:
                reservation = self.budget.reserve_task(
                    scalar_evaluations=scalar_count,
                    raw_records=raw_count,
                    audit_bytes=declared_bytes,
                )
        force_call_id, force_path = self._next_force_call()
        launched = False
        try:
            positions = np.asarray(atoms.get_positions(), dtype=np.float64)
            if atom_count == 0:
                raise ValueError("numerical solvent forces require at least one atom")
            if positions.shape != (atom_count, 3) or not np.isfinite(positions).all():
                raise ValueError("finite Cartesian coordinates are required")
            effective_steps, serialization_records = (
                self._preflight_stencils(atoms, positions) if need_forces else ({}, [])
            )

            ordinal = 0
            center_context = self._context(
                force_call_id=force_call_id,
                force_path=force_path,
                scalar_ordinal=ordinal,
                coordinate_index=None,
                sign=0,
                step_role="center",
                step_angstrom=0.0,
            )
            launched = True
            center = self._evaluate_scalar(
                atoms,
                positions,
                reservation=reservation,
                context=center_context,
            )
            if not need_forces:
                used_bytes = _directory_bytes(force_path)
                assert isinstance(reservation, _TaskReservation)
                reservation.reconcile(
                    actual_scalar_evaluations=1,
                    actual_raw_records=1,
                    actual_audit_bytes=used_bytes,
                )
                return center

            primary = np.empty_like(positions)
            check = np.empty_like(positions)
            flat_positions = positions.reshape(-1)
            for step_role, step, destination in (
                ("primary", self.force_step_angstrom, primary),
                ("check", self.force_check_step_angstrom, check),
            ):
                flat_force = destination.reshape(-1)
                for coordinate_index in range(flat_positions.size):
                    pair: dict[int, float] = {}
                    for sign in (1, -1):
                        ordinal += 1
                        displaced = flat_positions.copy()
                        displaced[coordinate_index] += sign * step
                        context = self._context(
                            force_call_id=force_call_id,
                            force_path=force_path,
                            scalar_ordinal=ordinal,
                            coordinate_index=coordinate_index,
                            sign=sign,
                            step_role=step_role,
                            step_angstrom=step,
                        )
                        pair[sign] = self._evaluate_scalar(
                            atoms,
                            displaced.reshape(positions.shape),
                            reservation=reservation,
                            context=context,
                        ).energy_hartree
                    denominator_step = effective_steps.get(
                        (step_role, coordinate_index), step
                    )
                    flat_force[coordinate_index] = -(pair[1] - pair[-1]) / (
                        2.0 * denominator_step
                    )

            difference = primary - check
            diagnostic = {
                "testing_only": True,
                "native_force": False,
                "derivative_source": self.derivative_source,
                "force_step_angstrom": self.force_step_angstrom,
                "force_check_step_angstrom": self.force_check_step_angstrom,
                "primary_forces_hartree_per_angstrom": primary.tolist(),
                "check_forces_hartree_per_angstrom": check.tolist(),
                "max_abs_difference_hartree_per_angstrom": float(
                    np.max(np.abs(difference))
                ),
                "rms_difference_hartree_per_angstrom": float(
                    np.sqrt(np.mean(difference**2))
                ),
                "scalar_evaluations": scalar_count,
                "force_call_id": force_call_id,
                "provider_serialization_preflight": serialization_records,
            }
            manifest_path = force_path / "force-manifest.json"
            manifest_hash = _atomic_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "force_call_id": force_call_id,
                    "termination_reason": "complete",
                    "numerical_force": diagnostic,
                },
            )
            diagnostic["manifest_path"] = str(manifest_path)
            diagnostic["manifest_sha256"] = manifest_hash
            used_bytes = _directory_bytes(force_path)
            if active_operation is not None:
                reservation.complete(actual_audit_bytes=0)
                active_operation.record_completion(used_bytes)
            else:
                reservation.complete(actual_audit_bytes=used_bytes)
            return SolvationResult(
                energy_hartree=center.energy_hartree,
                forces_hartree_per_angstrom=primary,
                components_hartree=dict(center.components_hartree),
                provenance={
                    **center.provenance,
                    "testing_only": True,
                    "native_force": False,
                    "numerical_force": diagnostic,
                },
            )
        except Exception as error:
            if launched:
                try:
                    _atomic_json(
                        force_path / "force-manifest.json",
                        {
                            "schema_version": 1,
                            "run_id": self.run_id,
                            "force_call_id": force_call_id,
                            "termination_reason": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                except OSError:
                    pass
            reservation.abort()
            raise


__all__ = [
    "EvaluationAuditContext",
    "NumericalForceBudget",
    "NumericalForceProvider",
]
