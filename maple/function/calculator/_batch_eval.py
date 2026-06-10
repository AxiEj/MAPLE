"""Algorithm-agnostic evaluators built on `CalcABC.calculate_many`.

These helpers are the thin layer that lets OPT (LBFGS / SDCG / RFO) and TS
(PRFO / NEB / AutoNEB / Dimer / GSM) consume the same batched calculator
interface without ever branching on model identity.

Phase 1 uses:

* `energy_forces_one` — single-structure E + F in one calculator invocation.
  Replaces the `get_potential_energy()` + `get_forces()` double-call pattern
  scattered throughout the OPT/TS algorithm files.
* `FDHessianEvaluator.hessian` — central-difference numerical Hessian that
  evaluates the `2 * N_movable_dof` displaced geometries through
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
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.constraints import FixAtoms, FixCartesian

from ._batch_types import BatchResult
from ._batch_utils import (
    atoms_list_has_pbc,
    normalize_energy_forces_request,
    sequential_calculate_many,
)

AUTO_BATCH_SIZE = "auto"
AUTO_BATCH_TARGET_FRACTION = 0.75
FD_HESSIAN_ANTISYMMETRY_THRESHOLD = 1e-5


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


def shared_calculator(atoms_list: Sequence[Atoms]):
    """Return the common calculator for a structure list, or ``None``.

    Batch evaluators are only scientifically valid when all structures are
    evaluated by the same calculator instance.  Mixed calculators may represent
    different models, devices, options, or solvent state, so callers should fall
    back to their original per-structure logic in that case.
    """
    atoms_list = list(atoms_list)
    if not atoms_list:
        return None
    calc = getattr(atoms_list[0], "calc", None)
    if calc is None:
        return None
    if all(getattr(at, "calc", None) is calc for at in atoms_list):
        return calc
    return None


def supports_batch_calculation(calc) -> bool:
    """Return True only for calculators with a validated native batch path."""
    return bool(getattr(calc, "supports_batch_energy_forces", False))


def structures_have_constraints(atoms_list: Sequence[Atoms]) -> bool:
    """Return True when any structure carries ASE constraints."""
    return any(bool(getattr(at, "constraints", None)) for at in atoms_list)


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


def _is_auto_batch_size(value) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO_BATCH_SIZE


def _positive_int_auto_or_none(value, name: str):
    if value is None:
        return None
    if _is_auto_batch_size(value):
        return AUTO_BATCH_SIZE
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive integer, 'auto', or None, got {value!r}"
        )
    if isinstance(value, str):
        try:
            coerced = int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a positive integer, 'auto', or None, got {value!r}"
            ) from exc
    else:
        try:
            coerced = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be a positive integer, 'auto', or None, got {value!r}"
            ) from exc
    if coerced <= 0:
        raise ValueError(
            f"{name} must be a positive integer, 'auto', or None, got {value!r}"
        )
    return coerced


def _positive_int_or_none(value, name: str) -> Optional[int]:
    value = _positive_int_auto_or_none(value, name)
    if value == AUTO_BATCH_SIZE:
        raise ValueError(
            f"{name} must be a positive integer or None, got {value!r}"
        )
    return value


def _calculator_batch_size(calc, specific: str):
    value = getattr(calc, specific, None)
    if value is None:
        value = getattr(calc, "batch_size", None)
    return _positive_int_auto_or_none(value, specific)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cublas_status_alloc_failed" in text
        or "cuda error: out of memory" in text
    )


def _clear_cuda_cache(calc) -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    device = getattr(calc, "device", None)
    try:
        if device is not None and getattr(torch.device(device), "type", None) == "cuda":
            torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception:
        return


def _cuda_device(calc):
    try:
        import torch
    except Exception:
        return None, None
    if not torch.cuda.is_available():
        return torch, None
    device = getattr(calc, "device", None)
    try:
        device = torch.device(device) if device is not None else torch.cuda.current_device()
    except Exception:
        device = torch.cuda.current_device()
    if getattr(device, "type", "cuda") != "cuda":
        return torch, None
    return torch, device


def _auto_batch_target_fraction(calc) -> float:
    value = getattr(calc, "auto_batch_target_fraction", AUTO_BATCH_TARGET_FRACTION)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return AUTO_BATCH_TARGET_FRACTION
    if not (0.1 <= value <= 0.95):
        return AUTO_BATCH_TARGET_FRACTION
    return value


def _torch_dtype_itemsize(dtype) -> int:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(dtype, torch.dtype):
        try:
            return torch.empty((), dtype=dtype).element_size()
        except Exception:
            return 4
    try:
        return np.dtype(dtype).itemsize
    except Exception:
        return 4


def _estimate_batch_item_bytes(calc, atoms: Atoms, properties) -> int:
    """Math-only memory estimate for one structure in a batch.

    This intentionally does not run a probe calculation.  It uses atom count,
    a conservative no-PBC graph edge estimate, dtype width, and whether forces
    are requested to size a chunk against free CUDA memory.  Backoff on CUDA
    OOM remains as a safety net for models whose hidden activations exceed the
    generic estimate.
    """
    custom = getattr(calc, "estimate_batch_item_bytes", None)
    if callable(custom):
        try:
            value = int(custom(atoms, properties=properties))
            if value > 0:
                return value
        except Exception:
            pass

    n_atoms = max(1, len(atoms))
    dtype_bytes = _torch_dtype_itemsize(getattr(calc, "dtype", np.float32))
    props = tuple(properties)
    force_factor = 2.5 if "forces" in props else 1.0

    # Graph backends roughly scale with nodes plus neighbor edges.  Without
    # evaluating a real neighbor list, use a bounded dense-ish edge estimate:
    # small systems get enough overhead, large systems avoid N^2 explosion.
    edge_count = min(n_atoms * max(n_atoms - 1, 1), max(n_atoms * 96, n_atoms))
    node_bytes = n_atoms * dtype_bytes * 4096
    edge_bytes = edge_count * dtype_bytes * 256
    base_bytes = 1 * 1024 * 1024

    return int((base_bytes + node_bytes + edge_bytes) * force_factor)


def _estimate_auto_batch_size_from_item_bytes(
    *,
    n_total: int,
    item_bytes: int,
    free_bytes: int,
    target_fraction: float = AUTO_BATCH_TARGET_FRACTION,
) -> int:
    """Estimate chunk size from a direct per-item memory model."""
    if n_total <= 0:
        return 0
    if item_bytes <= 0 or free_bytes <= 0:
        return n_total
    target_bytes = max(1.0, float(free_bytes) * float(target_fraction))
    chunk = max(1, min(n_total, int(target_bytes // float(item_bytes))))
    if chunk >= int(0.9 * n_total):
        return n_total
    return chunk


def _positive_cap(value) -> Optional[int]:
    """Return a positive integer cap, or ``None`` when unset/invalid."""
    if value is None:
        return None
    try:
        cap = int(value)
    except (TypeError, ValueError):
        return None
    return cap if cap > 0 else None


class _AutoBatchSizer:
    def __init__(
        self,
        calc,
        atoms_list: Sequence[Atoms],
        properties,
        *,
        kind: str,
    ) -> None:
        self.calc = calc
        self.atoms_list = list(atoms_list)
        self.properties = tuple(properties)
        self.kind = kind
        self.n_total = len(self.atoms_list)
        self.chunk = self._initial_chunk()

    def _initial_chunk(self) -> int:
        # Memory-headroom sizing is GPU-specific (relies on
        # torch.cuda.mem_get_info), but the math/backend caps in
        # ``_apply_math_cap`` are algorithmic — they bound chunk size to
        # avoid pathological concat-dense neighbor masks and observed
        # throughput cliffs.  Apply them regardless of device so the same
        # safety bounds hold on CPU and on CUDA without mem-get-info.
        chunk = self.n_total
        torch, device = _cuda_device(self.calc)
        if torch is not None and device is not None:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                item_bytes = max(
                    _estimate_batch_item_bytes(self.calc, atoms, self.properties)
                    for atoms in self.atoms_list
                )
                chunk = _estimate_auto_batch_size_from_item_bytes(
                    n_total=self.n_total,
                    item_bytes=item_bytes,
                    free_bytes=int(free_bytes),
                    target_fraction=_auto_batch_target_fraction(self.calc),
                )
            except Exception:
                chunk = self.n_total
        chunk = self._apply_math_cap(chunk)
        setattr(self.calc, "_auto_batch_size_last", chunk)
        return chunk

    def _apply_math_cap(self, chunk: int) -> int:
        memory_model = getattr(self.calc, "batch_memory_model", None)

        # Backends with concat+dense-neighbor batch construction (AIMNet2) have
        # an O((sum N_i)^2) memory term that the per-item estimate cannot see.
        # Use explicit calculator metadata rather than class/module strings so
        # renamed subclasses keep the same safety contract.
        hard_cap = _positive_cap(getattr(self.calc, "auto_batch_hard_cap", None))
        if hard_cap is not None:
            chunk = max(1, min(chunk, hard_cap, self.n_total))

        # Kind-specific caps are capability fields too.  ``auto_path_batch_cap``
        # remains a path-throughput knob for normal backends; for concat-dense
        # memory models it is also allowed to tighten FD/HVP chunks because the
        # same dense mask risk exists for every batched workload.
        kind_cap = _positive_cap(getattr(self.calc, f"auto_{self.kind}_batch_cap", None))
        if kind_cap is None and (
            self.kind == "path" or memory_model == "concat_dense_neighbor"
        ):
            kind_cap = _positive_cap(getattr(self.calc, "auto_path_batch_cap", None))
        if kind_cap is not None:
            chunk = max(1, min(chunk, kind_cap, self.n_total))

        return chunk

    def backoff_after_oom(self) -> bool:
        if self.chunk <= 1:
            return False
        self.chunk = max(1, self.chunk // 2)
        setattr(self.calc, "_auto_batch_size_last", self.chunk)
        _clear_cuda_cache(self.calc)
        return True


# ---------------------------------------------------------------------------
# Numerical Hessian via batched central difference / optional FD context
# ---------------------------------------------------------------------------
_SUPPORTED_FD_HESSIAN_CONSTRAINTS = (FixAtoms, FixCartesian)


def _movable_dofs(atoms: Atoms, respect_constraints: bool) -> List[int]:
    """Return unconstrained Cartesian DOF indices in a 3N Hessian.

    ASE's vibrational analysis officially handles ``FixAtoms`` and
    ``FixCartesian``.  ``FixAtoms`` removes all three Cartesian directions for
    an atom; ``FixCartesian`` removes only the masked directions.  Keeping this
    as a 3N DOF mask avoids displacing partially frozen coordinates and keeps
    the projected Hessian rows/columns aligned with frequency/RFO consumers.
    Other ASE constraints are deliberately rejected when constraints are
    respected, because their constrained Hessian is not representable as simple
    Cartesian rows/columns set to zero.
    """
    n_atoms = len(atoms)
    if not respect_constraints:
        return list(range(3 * n_atoms))

    mask = np.ones(3 * n_atoms, dtype=bool)
    for constraint in getattr(atoms, "constraints", []) or []:
        if not isinstance(constraint, _SUPPORTED_FD_HESSIAN_CONSTRAINTS):
            raise NotImplementedError(
                "FD Hessian only supports FixAtoms/FixCartesian constraints; "
                f"got {type(constraint).__name__}. Disable constraint handling "
                "explicitly if you want an unconstrained Cartesian FD Hessian."
            )

        if isinstance(constraint, FixAtoms):
            for atom_idx in constraint.get_indices():
                mask[3 * int(atom_idx) : 3 * int(atom_idx) + 3] = False
        elif isinstance(constraint, FixCartesian):
            fixed_axes = np.asarray(constraint.mask, dtype=bool).reshape(3)
            for atom_idx in constraint.get_indices():
                base = 3 * int(atom_idx)
                for axis, fixed in enumerate(fixed_axes):
                    if fixed:
                        mask[base + axis] = False

    return np.flatnonzero(mask).astype(np.int64).tolist()


def _fixed_dofs(n_atoms: int, movable_dofs: Sequence[int]) -> np.ndarray:
    movable_set = {int(dof) for dof in movable_dofs}
    return np.asarray(
        [dof for dof in range(3 * n_atoms) if dof not in movable_set],
        dtype=np.int64,
    )


def _copy_with_positions(
    template: Atoms,
    positions: np.ndarray,
    *,
    apply_constraints: bool = True,
) -> Atoms:
    at = template.copy()
    at.set_positions(positions, apply_constraint=apply_constraints)
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
    of ``2·N_movable_dof`` Python-level calls.

    ``FixAtoms`` and ``FixCartesian`` are respected by default — frozen DOFs
    contribute zero rows and columns.  Other ASE constraints fail fast because
    they cannot be represented by this Cartesian DOF projection.
    """

    def __init__(
        self,
        calc,
        fd_batch_size: Optional[int] = None,
        respect_fixatoms: bool = True,
        respect_constraints: Optional[bool] = None,
        fd_context_mode: Optional[str] = None,
    ) -> None:
        self.calc = calc
        if fd_batch_size is None:
            fd_batch_size = _calculator_batch_size(calc, "fd_batch_size")
        else:
            fd_batch_size = _positive_int_auto_or_none(fd_batch_size, "fd_batch_size")
        self.fd_batch_size = fd_batch_size
        if respect_constraints is None:
            respect_constraints = respect_fixatoms
        self.respect_constraints = bool(respect_constraints)
        # Backward-compatible alias for callers/tests that still use the old
        # name from the FixAtoms-only implementation.
        self.respect_fixatoms = self.respect_constraints
        self.fd_context_mode = fd_context_mode

    def hessian(self, atoms: Atoms, delta: float = 0.002) -> np.ndarray:
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta!r}")
        N = len(atoms)
        pos0 = atoms.get_positions().copy()

        movable_dofs = _movable_dofs(atoms, self.respect_constraints)
        H = np.zeros((3 * N, 3 * N), dtype=np.float64)
        if not movable_dofs:
            return H

        context = self._make_fd_context(atoms, delta)
        if context is not None:
            try:
                return self._hessian_from_context(
                    context, atoms, pos0, movable_dofs, H, delta
                )
            finally:
                atoms.set_positions(pos0)
                close = getattr(context, "close", None)
                if close is not None:
                    close()

        # Build the displaced-geometry list. Entries come in (plus, minus)
        # pairs per 3N Cartesian DOF so the central-difference reduction is
        # a simple zip over the resulting force list.
        displaced: List[Atoms] = []
        rows: List[int] = []
        for dof in movable_dofs:
            atom_idx, axis = divmod(int(dof), 3)
            rows.append(int(dof))
            pos_p = pos0.copy()
            pos_p[atom_idx, axis] += delta
            displaced.append(
                _copy_with_positions(
                    atoms,
                    pos_p,
                    apply_constraints=self.respect_constraints,
                )
            )

            pos_m = pos0.copy()
            pos_m[atom_idx, axis] -= delta
            displaced.append(
                _copy_with_positions(
                    atoms,
                    pos_m,
                    apply_constraints=self.respect_constraints,
                )
            )

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
        self._project_fixed_dofs(H, N, movable_dofs)
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
        movable_dofs: Sequence[int],
        H: np.ndarray,
        delta: float,
    ) -> np.ndarray:
        rows: List[int] = []
        forces: List[np.ndarray] = []
        for dof in movable_dofs:
            atom_idx, axis = divmod(int(dof), 3)
            rows.append(int(dof))
            pos_p = pos0.copy()
            pos_p[atom_idx, axis] += delta
            forces.append(np.asarray(context.force_at(pos_p), dtype=np.float64))

            pos_m = pos0.copy()
            pos_m[atom_idx, axis] -= delta
            forces.append(np.asarray(context.force_at(pos_m), dtype=np.float64))

        self._fill_rows_from_forces(H, rows, forces, delta)
        self._project_fixed_dofs(H, len(atoms), movable_dofs)
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
        movable_dofs: Sequence[int],
    ) -> None:
        if not self.respect_constraints:
            return
        frozen_dofs = _fixed_dofs(n_atoms, movable_dofs)
        if frozen_dofs.size == 0:
            return
        H[frozen_dofs, :] = 0.0
        H[:, frozen_dofs] = 0.0

    def _symmetrize(self, H: np.ndarray) -> None:
        """Remove finite-difference noise only after surfacing large residuals."""
        abs_resid, rel_resid = self._antisymmetry_residual(H)
        setattr(
            self.calc,
            "_fd_hessian_last_antisymmetry",
            {
                "absolute": abs_resid,
                "relative": rel_resid,
                "threshold": self._antisymmetry_threshold(),
            },
        )
        self._handle_antisymmetry_residual(abs_resid, rel_resid)
        H[:] = 0.5 * (H + H.T)

    @staticmethod
    def _antisymmetry_residual(H: np.ndarray) -> Tuple[float, float]:
        if H.size == 0:
            return 0.0, 0.0
        abs_resid = float(np.max(np.abs(H - H.T)))
        scale = max(1.0, float(np.max(np.abs(H))))
        return abs_resid, abs_resid / scale

    def _antisymmetry_threshold(self) -> Optional[float]:
        threshold = getattr(
            self.calc,
            "fd_hessian_antisymmetry_threshold",
            FD_HESSIAN_ANTISYMMETRY_THRESHOLD,
        )
        if threshold is None:
            return None
        threshold = float(threshold)
        if threshold < 0.0:
            raise ValueError(
                "fd_hessian_antisymmetry_threshold must be non-negative "
                f"or None, got {threshold!r}"
            )
        return threshold

    def _handle_antisymmetry_residual(
        self,
        abs_resid: float,
        rel_resid: float,
    ) -> None:
        threshold = self._antisymmetry_threshold()
        if threshold is None or rel_resid <= threshold:
            return

        msg = (
            "FD Hessian central-difference force derivative has a large "
            "antisymmetric residual before symmetrization: "
            f"abs={abs_resid:.6e}, rel={rel_resid:.6e}, "
            f"threshold={threshold:.6e}. This can indicate non-conservative "
            "forces, unit drift, or batch force-order/shape mismatch; the "
            "matrix will be symmetrized only after this diagnostic is surfaced."
        )
        action = str(
            getattr(self.calc, "fd_hessian_antisymmetry_action", "warn")
        ).lower()
        if action == "ignore":
            return
        if action == "warn":
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            return
        if action == "raise":
            raise RuntimeError(msg)
        raise ValueError(
            "fd_hessian_antisymmetry_action must be 'ignore', 'warn', or "
            f"'raise', got {action!r}"
        )

    def _chunked_forces(self, atoms_list: Sequence[Atoms]) -> List[np.ndarray]:
        n_total = len(atoms_list)
        if n_total == 0:
            return []

        auto = self.fd_batch_size == AUTO_BATCH_SIZE
        sizer = _AutoBatchSizer(
            self.calc, atoms_list, ("forces",), kind="fd"
        ) if auto else None
        chunk = (
            sizer.chunk if sizer is not None
            else self.fd_batch_size if self.fd_batch_size is not None
            else n_total
        )
        out: List[np.ndarray] = []
        start = 0
        while start < n_total:
            sub = list(atoms_list[start : start + chunk])
            try:
                result = _calculate_many_nonperiodic_batch_only(
                    self.calc, sub, properties=("forces",)
                )
            except RuntimeError as exc:
                if sizer is None or not _is_cuda_oom(exc) or not sizer.backoff_after_oom():
                    raise
                chunk = sizer.chunk
                continue
            if result.forces is None:
                raise RuntimeError(
                    "calculate_many returned no forces; required for "
                    "FDHessianEvaluator central difference."
                )
            for f in result.forces:
                out.append(np.asarray(f, dtype=np.float64))
            start += len(sub)
            if sizer is not None:
                chunk = max(1, min(sizer.chunk, n_total - start or sizer.chunk))
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
            batch_size = _positive_int_auto_or_none(batch_size, "batch_size")
        self.batch_size = batch_size

    def energy_forces(self, images: Sequence[Atoms]) -> Tuple[np.ndarray, List[np.ndarray]]:
        n_total = len(images)
        if n_total == 0:
            return np.zeros(0, dtype=np.float64), []

        auto = self.batch_size == AUTO_BATCH_SIZE
        sizer = _AutoBatchSizer(
            self.calc, images, ("energy", "forces"), kind="path"
        ) if auto else None
        chunk = (
            sizer.chunk if sizer is not None
            else self.batch_size if self.batch_size else n_total
        )
        energies: List[float] = []
        forces: List[np.ndarray] = []
        start = 0
        while start < n_total:
            sub = list(images[start : start + chunk])
            try:
                result = _calculate_many_nonperiodic_batch_only(
                    self.calc,
                    sub, properties=("energy", "forces")
                )
            except RuntimeError as exc:
                if sizer is None or not _is_cuda_oom(exc) or not sizer.backoff_after_oom():
                    raise
                chunk = sizer.chunk
                continue
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
            start += len(sub)
            if sizer is not None:
                chunk = max(1, min(sizer.chunk, n_total - start or sizer.chunk))
        return np.asarray(energies, dtype=np.float64), forces


class EnergyEvaluator:
    """Batch energy-only evaluation for independent structures.

    This is the right abstraction for SP trajectories, path-output summaries,
    and endpoint ranking where forces are not needed.  It shares the same chunk
    sizing and non-periodic-batch guard as :class:`PathEvaluator`, but requests
    only ``("energy",)`` so force autograd graphs are not built unnecessarily.
    """

    def __init__(self, calc, batch_size: Optional[int] = None) -> None:
        self.calc = calc
        if batch_size is None:
            batch_size = _calculator_batch_size(calc, "path_batch_size")
        else:
            batch_size = _positive_int_auto_or_none(batch_size, "batch_size")
        self.batch_size = batch_size

    def energies(self, images: Sequence[Atoms]) -> np.ndarray:
        n_total = len(images)
        if n_total == 0:
            return np.zeros(0, dtype=np.float64)

        auto = self.batch_size == AUTO_BATCH_SIZE
        sizer = _AutoBatchSizer(
            self.calc, images, ("energy",), kind="path"
        ) if auto else None
        chunk = (
            sizer.chunk if sizer is not None
            else self.batch_size if self.batch_size else n_total
        )
        energies: List[float] = []
        start = 0
        while start < n_total:
            sub = list(images[start : start + chunk])
            try:
                result = _calculate_many_nonperiodic_batch_only(
                    self.calc,
                    sub,
                    properties=("energy",),
                )
            except RuntimeError as exc:
                if sizer is None or not _is_cuda_oom(exc) or not sizer.backoff_after_oom():
                    raise
                chunk = sizer.chunk
                continue
            if result.energies is None or len(result.energies) != len(sub):
                got = None if result.energies is None else len(result.energies)
                raise RuntimeError(
                    "calculate_many returned the wrong number of energies "
                    f"for EnergyEvaluator: expected {len(sub)}, got {got}"
                )
            energies.extend(float(e) for e in result.energies.tolist())
            start += len(sub)
            if sizer is not None:
                chunk = max(1, min(sizer.chunk, n_total - start or sizer.chunk))
        return np.asarray(energies, dtype=np.float64)


class HVPEvaluator:
    """Hessian-vector product for Dimer (Phase 3).

    Prefers the calculator's autograd HVP (`CalcABC.get_hvp`) and falls back
    to a finite-difference pair (`R + δn`, `R - δn`) plus the unperturbed
    geometry `R`, batched through `calculate_many` for calculators that cannot
    expose HVP.  Hn comes from the +-delta pair; forces/energy are reported at
    `R` so the fallback matches the autograd contract exactly.
    """

    def __init__(self, calc, batch_size: Optional[int] = None) -> None:
        self.calc = calc
        if batch_size is None:
            batch_size = _calculator_batch_size(calc, "hvp_batch_size")
        else:
            batch_size = _positive_int_auto_or_none(batch_size, "batch_size")
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
        at_0 = _copy_with_positions(atoms, pos0)

        try:
            energies, forces = PathEvaluator(
                self.calc, batch_size=self.batch_size
            ).energy_forces([at_p, at_m, at_0])
        finally:
            atoms.set_positions(pos0)

        if len(forces) != 3 or len(energies) != 3:
            raise RuntimeError(
                "calculate_many must return both energies and forces for "
                "HVPEvaluator finite-difference fallback."
            )
        F_p = np.asarray(forces[0], dtype=np.float64).reshape(-1)
        F_m = np.asarray(forces[1], dtype=np.float64).reshape(-1)
        F_0 = np.asarray(forces[2], dtype=np.float64).reshape(-1)
        E_0 = float(energies[2])

        # Match CalcABC.get_hvp: forces/energy are reported AT the unperturbed
        # geometry R, while Hn comes from the +-delta force pair.
        Hn = -(F_p - F_m) / (2.0 * delta)
        return Hn, F_0, E_0
