from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..calculator.analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticIdentityError,
)
from .curvature import CurvatureValidationError


class NumericalStateError(ValueError):
    """Expected refusal for non-real or non-finite numerical state."""


def to_numpy_f64(x):
    """Convert one real finite numerical value before any float cast."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        if torch.is_complex(x):
            raise NumericalStateError(
                "Complex numerical state cannot be converted to float64."
            )
        x = x.detach().cpu().numpy()
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise NumericalStateError("Complex numerical state cannot be converted to float64.")
    converted = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise NumericalStateError("Numerical state must contain only finite values.")
    if converted.ndim == 0:
        return float(converted.item())
    return converted


def vec1d(x, n_expected=None):
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v


def _values_equal(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
        except (TypeError, ValueError):
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _constraint_identity(constraints):
    return tuple(
        constraint.todict() if hasattr(constraint, "todict") else repr(constraint)
        for constraint in constraints
    )


@dataclass
class ScientificAtomsSnapshot:
    arrays: dict
    cell: np.ndarray
    pbc: np.ndarray
    constraints: list
    constraint_identity: tuple
    info: dict
    calculator: object
    metrics: dict

    @classmethod
    def capture(cls, atoms: Atoms) -> "ScientificAtomsSnapshot":
        return cls(
            arrays={name: np.array(value, copy=True) for name, value in atoms.arrays.items()},
            cell=np.asarray(atoms.cell).copy(),
            pbc=np.asarray(atoms.pbc).copy(),
            constraints=deepcopy(list(atoms.constraints)),
            constraint_identity=_constraint_identity(atoms.constraints),
            info=deepcopy(dict(atoms.info)),
            calculator=atoms.calc,
            metrics={
                name: deepcopy(getattr(atoms, name))
                for name in ("max_f", "rms_f", "max_dp", "rms_dp")
                if hasattr(atoms, name)
            },
        )

    def matches(self, atoms: Atoms) -> bool:
        return (
            atoms.arrays.keys() == self.arrays.keys()
            and all(
                np.array_equal(atoms.arrays[name], value, equal_nan=True)
                for name, value in self.arrays.items()
            )
            and np.array_equal(np.asarray(atoms.cell), self.cell)
            and np.array_equal(np.asarray(atoms.pbc), self.pbc)
            and _constraint_identity(atoms.constraints) == self.constraint_identity
            and _values_equal(atoms.info, self.info)
            and atoms.calc is self.calculator
        )

    def restore(self, atoms: Atoms) -> None:
        atoms.arrays = {
            name: np.array(value, copy=True) for name, value in self.arrays.items()
        }
        atoms.set_cell(self.cell)
        atoms.set_pbc(self.pbc)
        atoms.set_constraint(deepcopy(self.constraints))
        atoms.info.clear()
        atoms.info.update(deepcopy(self.info))
        atoms.calc = self.calculator
        for name in ("max_f", "rms_f", "max_dp", "rms_dp"):
            if name in self.metrics:
                setattr(atoms, name, deepcopy(self.metrics[name]))
            elif hasattr(atoms, name):
                delattr(atoms, name)


@contextmanager
def scientific_atoms_transaction(atoms: Atoms, *, operation: str):
    snapshot = ScientificAtomsSnapshot.capture(atoms)
    try:
        yield
    except BaseException as exc:
        try:
            snapshot.restore(atoms)
        except Exception as restore_error:
            exc.add_note(
                f"Scientific Atoms restoration also failed: {restore_error!r}"
            )
        raise
    mutated = not snapshot.matches(atoms)
    snapshot.restore(atoms)
    if mutated:
        raise CurvatureValidationError(
            f"{operation} provider mutated scientific Atoms state during evaluation."
        )


def evaluate_provider(atoms: Atoms, *, operation: str, evaluate):
    """Evaluate one provider operation and translate only analytic refusals."""
    try:
        with scientific_atoms_transaction(atoms, operation=operation):
            return evaluate()
    except (AnalyticIdentityError, AnalyticCapabilityError) as exc:
        raise CurvatureValidationError(str(exc)) from exc


def evaluate_hessian(atoms: Atoms):
    """
    Calculate Hessian matrix from calculator.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with calculator attached

    Returns
    -------
    H : array
        Hessian matrix in float64
    """
    calc = atoms.calc
    return evaluate_provider(
        atoms,
        operation="Hessian",
        evaluate=lambda: np.asarray(calc.get_hessian(atoms)),
    )


def evaluate_energy(atoms: Atoms) -> float:
    return evaluate_provider(
        atoms,
        operation="Energy",
        evaluate=lambda: to_numpy_f64(
            atoms.get_potential_energy(force_consistent=True)
        ),
    )


def evaluate_forces(atoms: Atoms) -> np.ndarray:
    return evaluate_provider(
        atoms,
        operation="Force",
        evaluate=lambda: to_numpy_f64(atoms.get_forces()),
    )
