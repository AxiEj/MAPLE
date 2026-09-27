"""Resource admission for dense Torch pair graphs and Cartesian Hessians.

These limits are execution safeguards, not atom-count restrictions or model
physics. Conservative planning estimates charge float64/autograd storage before
a dense pair tensor or Hessian is materialized; these are not proven bounds on
allocator use or total process memory.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real

# Allow 64 float64-equivalent values per pair for the forward expression and
# double the retained-state allowance at each automatic-differentiation tier.
_PAIR_BYTES_PER_PAIR = (512, 1024, 2048)
_PAIR_WORK_PER_PAIR = (1, 2, 4)
_FLOAT64_BYTES = 8


def _require_positive_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


@dataclass(frozen=True)
class DenseTorchBudget:
    """Configurable execution limits for research-scale dense Torch kernels."""

    max_pair_graph_bytes: int = 512 * 1024 * 1024
    max_pair_work_units: int = 4_000_000
    max_dense_hessian_bytes: int = 512 * 1024 * 1024
    max_dense_hessian_sweeps: int = 768

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field.name} must be a positive finite integer.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field.name} must be finite.")
            if int(value) != value:
                raise ValueError(f"{field.name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{field.name} must be positive.")
            object.__setattr__(self, field.name, int(value))

    def admit_pair_graph(
        self, atom_count: int, *, derivative_order: int, label: str
    ) -> None:
        """Admit an NxN pair graph before its first dense allocation.

        ``derivative_order`` is 0 for energy, 1 for first derivatives, and 2
        for HVP/second-derivative graphs.  Higher-order AD retention is charged
        more heavily without changing the underlying scientific expression.
        """
        atom_count = _require_positive_int("atom_count", atom_count)
        if type(derivative_order) is not int:
            raise TypeError("derivative_order must be an integer.")
        if derivative_order not in (0, 1, 2):
            raise ValueError("derivative_order must be 0, 1, or 2.")
        pairs = atom_count * atom_count
        projected_bytes = pairs * _PAIR_BYTES_PER_PAIR[derivative_order]
        if projected_bytes > self.max_pair_graph_bytes:
            raise RuntimeError(
                f"{label} pair-graph memory budget exceeded: projected "
                f"{projected_bytes} > {self.max_pair_graph_bytes} bytes."
            )
        projected_work = pairs * _PAIR_WORK_PER_PAIR[derivative_order]
        if projected_work > self.max_pair_work_units:
            raise RuntimeError(
                f"{label} pair-work budget exceeded: projected "
                f"{projected_work} > {self.max_pair_work_units} work units."
            )

    def admit_dense_hessian(self, atom_count: int, *, label: str) -> None:
        """Admit full Cartesian materialization separately from an HVP graph."""
        atom_count = _require_positive_int("atom_count", atom_count)
        coordinates = 3 * atom_count
        # Stack output, retained row tensors, and the final NumPy copy.
        projected_bytes = 3 * coordinates * coordinates * _FLOAT64_BYTES
        if projected_bytes > self.max_dense_hessian_bytes:
            raise RuntimeError(
                f"{label} dense-Hessian memory budget exceeded: projected "
                f"{projected_bytes} > {self.max_dense_hessian_bytes} bytes."
            )
        if coordinates > self.max_dense_hessian_sweeps:
            raise RuntimeError(
                f"{label} Hessian sweep budget exceeded: projected "
                f"{coordinates} > {self.max_dense_hessian_sweeps} reverse sweeps."
            )


DEFAULT_DENSE_TORCH_BUDGET = DenseTorchBudget()
