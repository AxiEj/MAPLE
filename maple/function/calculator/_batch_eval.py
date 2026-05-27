"""Algorithm-agnostic evaluators built on `CalcABC.calculate_many`.

These helpers are the thin layer that lets OPT (LBFGS / SDCG / RFO) and TS
(PRFO / NEB / AutoNEB / Dimer / GSM) consume the same batched calculator
interface without ever branching on model identity.

Phase 1 uses:

* `energy_forces_one` — single-structure E + F in one calculator invocation.
  Replaces the `get_potential_energy()` + `get_forces()` double-call pattern
  scattered throughout the OPT/TS algorithm files.
* `FDHessianEvaluator.hessian` — central-difference numerical Hessian that
  evaluates the `2 * 3 * N_movable` displaced geometries through
  `calculate_many`. This is the unified replacement for the per-calculator
  `_get_hessian_numerical` loops in ANI / MACE / MACEPol / MACEOMol /
  AIMNet2 / UMA.

`PathEvaluator` applies the same contract to NEB/CINEB path snapshots:
all image energies and true forces are collected together, while the
optimizer keeps the original tangent, spring-force, and trust-region logic.
`HVPEvaluator` is scaffolded for Dimer HVP capability dispatch.
"""
from __future__ import annotations

import operator
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.constraints import FixAtoms

from ._batch_types import BatchResult
from ._batch_utils import (
    atoms_list_has_pbc,
    normalize_energy_forces_request,
    sequential_calculate_many,
)


def _calculate_many_nonperiodic_batch_only(calc, atoms_list, properties) -> BatchResult:
    """Route PBC structures through single-structure calculate calls.

    MAPLE's current batch acceleration contract is validated for non-periodic
    structures only.  Periodic systems must not enter model-native
    multi-structure batch paths until each backend has a PBC-aware batch
    implementation and parity tests, so this helper disables batch acceleration
    by falling back to sequential single-structure evaluation whenever any
    candidate structure has PBC enabled.
    """
    atoms_list = list(atoms_list)
    if atoms_list_has_pbc(atoms_list):
        _, want_energy, want_forces, request = normalize_energy_forces_request(
            properties
        )
        if not request:
            return BatchResult()
        return sequential_calculate_many(
            calc, atoms_list, request, want_energy, want_forces
        )
    return calc.calculate_many(atoms_list, properties=properties)


# ---------------------------------------------------------------------------
# Single-structure E + F merge
# ---------------------------------------------------------------------------
def energy_forces_one(calc, atoms: Atoms, force_consistent: bool = True
                      ) -> Tuple[float, np.ndarray]:
    """One calculator invocation returning ``(energy, forces)``.

    Equivalent to ``atoms.get_potential_energy() + atoms.get_forces()`` but
    asks the calculator for both properties in a single ``calculate(...)``
    call. This guarantees one calculator invocation; true model-level forward
    count still depends on the subclass implementation.

    Returns
    -------
    energy
        Scalar float (Hartree for MAPLE calculators).
    forces
        ``(N, 3)`` float64 numpy array (Hartree/Å).
    """
    calc.calculate(
        atoms,
        properties=["energy", "forces"],
        system_changes=all_changes,
    )
    if force_consistent and "free_energy" in calc.results:
        energy = float(calc.results["free_energy"])
    else:
        energy = float(calc.results["energy"])
    forces = np.asarray(calc.results["forces"], dtype=np.float64)
    return energy, forces


def reset_calculator_cache(calc) -> None:
    """Clear ASE calculator cache after rolling atoms back to an old geometry."""
    reset = getattr(calc, "reset", None)
    if callable(reset):
        reset()
        return
    results = getattr(calc, "results", None)
    if hasattr(results, "clear"):
        results.clear()


def _positive_int_or_none(value, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive integer or None, got {value!r}"
        )
    try:
        coerced = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a positive integer or None, got {value!r}"
        ) from exc
    if coerced <= 0:
        raise ValueError(
            f"{name} must be a positive integer or None, got {value!r}"
        )
    return coerced


def _calculator_batch_size(calc, specific: str) -> Optional[int]:
    value = getattr(calc, specific, None)
    if value is None:
        value = getattr(calc, "batch_size", None)
    return _positive_int_or_none(value, specific)


# ---------------------------------------------------------------------------
# Numerical Hessian via batched central difference / optional FD context
# ---------------------------------------------------------------------------
def _movable_indices(atoms: Atoms, respect_fixatoms: bool) -> List[int]:
    if not respect_fixatoms:
        return list(range(len(atoms)))
    fixed = {
        i
        for c in getattr(atoms, "constraints", []) or []
        if isinstance(c, FixAtoms)
        for i in c.get_indices()
    }
    return [i for i in range(len(atoms)) if i not in fixed]


def _fixed_dofs(n_atoms: int, movable: Sequence[int]) -> np.ndarray:
    movable_set = set(movable)
    frozen = [i for i in range(n_atoms) if i not in movable_set]
    return np.asarray(
        [3 * a + k for a in frozen for k in range(3)],
        dtype=np.int64,
    )


def _copy_with_positions(template: Atoms, positions: np.ndarray) -> Atoms:
    at = template.copy()
    at.set_positions(positions)
    if getattr(template, "constraints", None):
        at.set_constraint(template.constraints)
    return at


class FDHessianContext:
    """Optional fixed-topology force context for one FD Hessian evaluation.

    Model-specific calculators may override ``CalcABC.make_fd_context`` to
    return a subclass of this object. The context is scoped to one reference
    geometry and must only be used for tiny finite-difference displacements
    around that geometry. Implementations own their safety policy:

    * ``mode="safe"`` should use cutoff skins / validity checks and rebuild
      or fail fast when graph reuse is unsafe.
    * ``mode="fast"`` may skip expensive checks, but remains opt-in.

    The base class is deliberately abstract. Returning ``None`` from
    ``make_fd_context`` keeps the Phase 1 sequential ``calculate_many`` path.
    """

    valid_modes = ("safe", "fast")

    def __init__(
        self,
        calc,
        ref_atoms: Atoms,
        *,
        mode: str = "safe",
        delta: float | None = None,
    ) -> None:
        if mode not in self.valid_modes:
            raise ValueError(
                "fd_context_mode must be 'safe' or 'fast', "
                f"got {mode!r}"
            )
        self.calc = calc
        self.ref_atoms = ref_atoms.copy()
        self.mode = mode
        self.delta = delta

    def force_at(self, positions: np.ndarray) -> np.ndarray:
        """Return forces for ``positions`` without mutating ASE results."""
        raise NotImplementedError

    def close(self) -> None:
        """Release model-specific cached state, if any."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class FDHessianEvaluator:
    """Central-difference numerical Hessian via batched displacement.

    The Hessian element is

        H[3a+k, :] = -(F(R + δ·e_{a,k}) - F(R - δ·e_{a,k})) / (2·δ)

    which is mathematically identical to the per-DOF sequential loops the
    per-model calculators ship today; only the force evaluations are routed
    through ``calc.calculate_many`` so a true-batch backend (or a sequential
    fallback) can evaluate them with a single Python-level dispatch instead
    of ``2·3·N_movable`` Python-level calls.

    FixAtoms is respected by default — frozen atoms contribute zero rows.
    """

    def __init__(
        self,
        calc,
        fd_batch_size: Optional[int] = None,
        respect_fixatoms: bool = True,
        fd_context_mode: Optional[str] = None,
    ) -> None:
        self.calc = calc
        if fd_batch_size is None:
            fd_batch_size = _calculator_batch_size(calc, "fd_batch_size")
        else:
            fd_batch_size = _positive_int_or_none(fd_batch_size, "fd_batch_size")
        self.fd_batch_size = fd_batch_size
        self.respect_fixatoms = respect_fixatoms
        self.fd_context_mode = fd_context_mode

    def hessian(self, atoms: Atoms, delta: float = 0.002) -> np.ndarray:
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta!r}")
        N = len(atoms)
        pos0 = atoms.get_positions().copy()

        movable = _movable_indices(atoms, self.respect_fixatoms)
        H = np.zeros((3 * N, 3 * N), dtype=np.float64)
        if not movable:
            return H

        context = self._make_fd_context(atoms, delta)
        if context is not None:
            try:
                return self._hessian_from_context(context, atoms, pos0, movable, H, delta)
            finally:
                atoms.set_positions(pos0)
                close = getattr(context, "close", None)
                if close is not None:
                    close()

        # Build the displaced-geometry list. Entries come in (plus, minus)
        # pairs per (atom, axis) DOF so the central-difference reduction is
        # a simple zip over the resulting force list.
        displaced: List[Atoms] = []
        rows: List[int] = []
        for a in movable:
            for k in range(3):
                rows.append(3 * a + k)
                pos_p = pos0.copy()
                pos_p[a, k] += delta
                displaced.append(_copy_with_positions(atoms, pos_p))

                pos_m = pos0.copy()
                pos_m[a, k] -= delta
                displaced.append(_copy_with_positions(atoms, pos_m))

        try:
            # Evaluate forces. We do not request energies — the central
            # difference formula only needs forces, and skipping energies lets
            # batched backends avoid a redundant scalar reduction.
            forces = self._chunked_forces(displaced)
        finally:
            # Restore the input atoms so callers do not see a perturbed geometry
            # even if a backend raises while evaluating a displacement chunk.
            atoms.set_positions(pos0)

        self._fill_rows_from_forces(H, rows, forces, delta)
        self._project_fixed_dofs(H, N, movable)
        self._symmetrize(H)

        return H

    def _make_fd_context(self, atoms: Atoms, delta: float):
        factory = getattr(self.calc, "make_fd_context", None)
        if factory is None:
            return None

        try:
            return factory(
                atoms,
                delta=delta,
                fd_context_mode=self.fd_context_mode,
            )
        except NotImplementedError:
            return None

    def _hessian_from_context(
        self,
        context: FDHessianContext,
        atoms: Atoms,
        pos0: np.ndarray,
        movable: Sequence[int],
        H: np.ndarray,
        delta: float,
    ) -> np.ndarray:
        rows: List[int] = []
        forces: List[np.ndarray] = []
        for a in movable:
            for k in range(3):
                rows.append(3 * a + k)
                pos_p = pos0.copy()
                pos_p[a, k] += delta
                forces.append(np.asarray(context.force_at(pos_p), dtype=np.float64))

                pos_m = pos0.copy()
                pos_m[a, k] -= delta
                forces.append(np.asarray(context.force_at(pos_m), dtype=np.float64))

        self._fill_rows_from_forces(H, rows, forces, delta)
        self._project_fixed_dofs(H, len(atoms), movable)
        self._symmetrize(H)
        atoms.set_positions(pos0)
        return H

    def _fill_rows_from_forces(
        self,
        H: np.ndarray,
        rows: Sequence[int],
        forces: Sequence[np.ndarray],
        delta: float,
    ) -> None:
        if len(forces) != 2 * len(rows):
            raise ValueError(
                "calculate_many returned the wrong number of force arrays: "
                f"expected {2 * len(rows)}, got {len(forces)}"
            )
        n_atoms = H.shape[0] // 3
        inv_2delta = 1.0 / (2.0 * delta)
        for j, row in enumerate(rows):
            F_plus = self._validated_force(forces[2 * j], n_atoms, 2 * j)
            F_minus = self._validated_force(forces[2 * j + 1], n_atoms, 2 * j + 1)
            H[row, :] = (-(F_plus - F_minus) * inv_2delta).reshape(-1)

    @staticmethod
    def _validated_force(force: np.ndarray, n_atoms: int, index: int) -> np.ndarray:
        arr = np.asarray(force, dtype=np.float64)
        if arr.shape != (n_atoms, 3):
            raise ValueError(
                "calculate_many returned a force array with invalid shape: "
                f"forces[{index}].shape={arr.shape}, expected {(n_atoms, 3)}"
            )
        return arr

    def _project_fixed_dofs(
        self,
        H: np.ndarray,
        n_atoms: int,
        movable: Sequence[int],
    ) -> None:
        if not self.respect_fixatoms:
            return
        frozen_dofs = _fixed_dofs(n_atoms, movable)
        if frozen_dofs.size == 0:
            return
        H[frozen_dofs, :] = 0.0
        H[:, frozen_dofs] = 0.0

    @staticmethod
    def _symmetrize(H: np.ndarray) -> None:
        """Remove finite-difference/autograd noise that breaks H = H.T."""
        H[:] = 0.5 * (H + H.T)

    def _chunked_forces(self, atoms_list: Sequence[Atoms]) -> List[np.ndarray]:
        n_total = len(atoms_list)
        if n_total == 0:
            return []

        chunk = self.fd_batch_size if self.fd_batch_size is not None else n_total
        out: List[np.ndarray] = []
        for start in range(0, n_total, chunk):
            sub = list(atoms_list[start : start + chunk])
            result = _calculate_many_nonperiodic_batch_only(
                self.calc, sub, properties=("forces",)
            )
            if result.forces is None:
                raise RuntimeError(
                    "calculate_many returned no forces; required for "
                    "FDHessianEvaluator central difference."
                )
            for f in result.forces:
                out.append(np.asarray(f, dtype=np.float64))
        return out


# ---------------------------------------------------------------------------
# Path snapshots and HVP scaffolding
# ---------------------------------------------------------------------------
class PathEvaluator:
    """Batch energy + force over a list of images (NEB / AutoNEB / GSM).

    The implementation deliberately mirrors `FDHessianEvaluator._chunked_forces`
    so both routes degrade gracefully on the sequential `calculate_many`
    fallback.  The evaluator is deliberately model-agnostic: it only changes
    how E/F values are fetched, not how a path optimizer uses those values.
    """

    def __init__(self, calc, batch_size: Optional[int] = None) -> None:
        self.calc = calc
        if batch_size is None:
            batch_size = _calculator_batch_size(calc, "path_batch_size")
        else:
            batch_size = _positive_int_or_none(batch_size, "batch_size")
        self.batch_size = batch_size

    def energy_forces(self, images: Sequence[Atoms]) -> Tuple[np.ndarray, List[np.ndarray]]:
        n_total = len(images)
        if n_total == 0:
            return np.zeros(0, dtype=np.float64), []

        chunk = self.batch_size if self.batch_size else n_total
        energies: List[float] = []
        forces: List[np.ndarray] = []
        for start in range(0, n_total, chunk):
            sub = list(images[start : start + chunk])
            result = _calculate_many_nonperiodic_batch_only(
                self.calc,
                sub, properties=("energy", "forces")
            )
            if result.energies is None or len(result.energies) != len(sub):
                got = None if result.energies is None else len(result.energies)
                raise RuntimeError(
                    "calculate_many returned the wrong number of energies "
                    f"for PathEvaluator: expected {len(sub)}, got {got}"
                )
            if result.forces is None or len(result.forces) != len(sub):
                got = None if result.forces is None else len(result.forces)
                raise RuntimeError(
                    "calculate_many returned the wrong number of force arrays "
                    f"for PathEvaluator: expected {len(sub)}, got {got}"
                )
            energies.extend(float(e) for e in result.energies.tolist())
            forces.extend(np.asarray(f, dtype=np.float64) for f in result.forces)
        return np.asarray(energies, dtype=np.float64), forces


class HVPEvaluator:
    """Hessian-vector product for Dimer (Phase 3).

    Prefers the calculator's autograd HVP (`CalcABC.get_hvp`) and falls back
    to a finite-difference pair (`R + δn`, `R - δn`) batched through
    `calculate_many` for calculators that cannot expose HVP.
    """

    def __init__(self, calc, batch_size: Optional[int] = None) -> None:
        self.calc = calc
        if batch_size is None:
            batch_size = _calculator_batch_size(calc, "hvp_batch_size")
        else:
            batch_size = _positive_int_or_none(batch_size, "batch_size")
        self.batch_size = batch_size

    def hn(
        self,
        atoms: Atoms,
        n: np.ndarray,
        delta: float = 0.005,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if getattr(self.calc, "supports_hvp", False) and hasattr(self.calc, "get_hvp"):
            Hn_t, F_t, E_t = self.calc.get_hvp(atoms, n)

            def _to_np(t):
                if hasattr(t, "detach"):
                    return t.detach().cpu().numpy().astype(np.float64, copy=False)
                return np.asarray(t, dtype=np.float64)

            Hn = _to_np(Hn_t).reshape(-1)
            F = _to_np(F_t).reshape(-1)
            if hasattr(E_t, "item"):
                E = float(E_t.item())
            else:
                E = float(E_t)
            return Hn, F, E

        n_arr = np.asarray(n, dtype=np.float64).reshape(-1, 3)
        pos0 = atoms.get_positions().copy()
        at_p = _copy_with_positions(atoms, pos0 + delta * n_arr)
        at_m = _copy_with_positions(atoms, pos0 - delta * n_arr)

        try:
            energies, forces = PathEvaluator(
                self.calc, batch_size=self.batch_size
            ).energy_forces([at_p, at_m])
        finally:
            atoms.set_positions(pos0)

        if len(forces) != 2 or len(energies) != 2:
            raise RuntimeError(
                "calculate_many must return both energies and forces for "
                "HVPEvaluator finite-difference fallback."
            )
        F_p = np.asarray(forces[0], dtype=np.float64).reshape(-1)
        F_m = np.asarray(forces[1], dtype=np.float64).reshape(-1)
        E_p = float(energies[0])
        E_m = float(energies[1])

        Hn = -(F_p - F_m) / (2.0 * delta)
        F_mid = 0.5 * (F_p + F_m)
        E_mid = 0.5 * (E_p + E_m)
        return Hn, F_mid, E_mid
