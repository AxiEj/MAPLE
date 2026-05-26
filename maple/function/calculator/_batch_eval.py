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

`PathEvaluator` and `HVPEvaluator` are scaffolded with the same contract so
Phase 2 (NEB image batching) and Phase 3 (Dimer HVP capability dispatch) can
land without redesigning the interface.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.constraints import FixAtoms

# ---------------------------------------------------------------------------
# Single-structure E + F merge
# ---------------------------------------------------------------------------
def energy_forces_one(calc, atoms: Atoms, force_consistent: bool = True
                      ) -> Tuple[float, np.ndarray]:
    """One calculator invocation returning ``(energy, forces)``.

    Equivalent to ``atoms.get_potential_energy() + atoms.get_forces()`` but
    asks the calculator for both properties in a single ``calculate(...)``
    call so the underlying ML model does at most one forward pass.

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
        self.fd_batch_size = fd_batch_size
        self.respect_fixatoms = respect_fixatoms
        self.fd_context_mode = fd_context_mode

    def hessian(self, atoms: Atoms, delta: float = 0.002) -> np.ndarray:
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
        atoms.set_positions(pos0)
        return H

    @staticmethod
    def _fill_rows_from_forces(
        H: np.ndarray,
        rows: Sequence[int],
        forces: Sequence[np.ndarray],
        delta: float,
    ) -> None:
        inv_2delta = 1.0 / (2.0 * delta)
        for j, row in enumerate(rows):
            F_plus = forces[2 * j]
            F_minus = forces[2 * j + 1]
            H[row, :] = (-(F_plus - F_minus) * inv_2delta).reshape(-1)

    def _chunked_forces(self, atoms_list: Sequence[Atoms]) -> List[np.ndarray]:
        n_total = len(atoms_list)
        if n_total == 0:
            return []

        chunk = self.fd_batch_size if self.fd_batch_size else n_total
        out: List[np.ndarray] = []
        for start in range(0, n_total, chunk):
            sub = list(atoms_list[start : start + chunk])
            result = self.calc.calculate_many(sub, properties=("forces",))
            if result.forces is None:
                raise RuntimeError(
                    "calculate_many returned no forces; required for "
                    "FDHessianEvaluator central difference."
                )
            for f in result.forces:
                out.append(np.asarray(f, dtype=np.float64))
        return out


# ---------------------------------------------------------------------------
# Scaffolding for Phase 2 (paths) and Phase 3 (HVP)
# ---------------------------------------------------------------------------
class PathEvaluator:
    """Batch energy + force over a list of images (NEB / AutoNEB / GSM).

    Phase 1 ships the API only; algorithm-level integration is Phase 2.
    The implementation deliberately mirrors `FDHessianEvaluator._chunked_forces`
    so both routes degrade gracefully on the sequential `calculate_many`
    fallback.
    """

    def __init__(self, calc, batch_size: Optional[int] = None) -> None:
        self.calc = calc
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
            result = self.calc.calculate_many(
                sub, properties=("energy", "forces")
            )
            if result.energies is not None:
                energies.extend(float(e) for e in result.energies.tolist())
            if result.forces is not None:
                forces.extend(np.asarray(f, dtype=np.float64) for f in result.forces)
        return np.asarray(energies, dtype=np.float64), forces


class HVPEvaluator:
    """Hessian-vector product for Dimer (Phase 3).

    Prefers the calculator's autograd HVP (`CalcABC.get_hvp`) and falls back
    to a finite-difference pair (`R + δn`, `R - δn`) batched through
    `calculate_many` for calculators that cannot expose HVP.
    """

    def __init__(self, calc) -> None:
        self.calc = calc

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

        result = self.calc.calculate_many(
            [at_p, at_m], properties=("energy", "forces")
        )
        atoms.set_positions(pos0)

        if result.forces is None or result.energies is None:
            raise RuntimeError(
                "calculate_many must return both energies and forces for "
                "HVPEvaluator finite-difference fallback."
            )
        F_p = np.asarray(result.forces[0], dtype=np.float64).reshape(-1)
        F_m = np.asarray(result.forces[1], dtype=np.float64).reshape(-1)
        E_p = float(result.energies[0])
        E_m = float(result.energies[1])

        Hn = -(F_p - F_m) / (2.0 * delta)
        F_mid = 0.5 * (F_p + F_m)
        E_mid = 0.5 * (E_p + E_m)
        return Hn, F_mid, E_mid
