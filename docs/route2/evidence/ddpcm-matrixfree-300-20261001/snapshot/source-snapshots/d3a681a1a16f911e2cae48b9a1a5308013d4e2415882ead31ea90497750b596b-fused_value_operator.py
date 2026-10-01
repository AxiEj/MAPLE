"""Research-only fused FP64 ddPCM D actions without an ``S x G x Q`` tensor.

The physical operator, harmonic normalization, radial powers and topology come
from :class:`StreamedDDPCM`.  Only the value-level contraction order is changed:
real harmonics are generated degree by degree and immediately contracted with
the RHS block.  Derivatives and stability certification remain unsupported.
"""

import hashlib
import math
from pathlib import Path
import time

import torch

from streamed_ddpcm import (
    StreamedDDPCM,
    _KNOWN_WORKSPACE_LIMIT_BYTES,
    known_resource_estimate,
)


def _normalizations(lmax):
    return tuple(
        math.sqrt(
            (2 * ell + 1)
            * math.exp(
                math.lgamma(ell - abs(signed_m) + 1)
                - math.lgamma(ell + abs(signed_m) + 1)
            )
            / (4.0 * math.pi)
        )
        for ell in range(lmax + 1)
        for signed_m in range(-ell, ell + 1)
    )


def _degree_values(x, y, z, lmax):
    """Yield one degree in dense-column order with only O(lmax*S*G) state."""
    phases = tuple((-1) ** m * math.prod(range(1, 2 * m, 2)) for m in range(lmax + 1))
    previous = []
    previous2 = []
    real_power, imag_power = torch.ones_like(z), torch.zeros_like(z)
    for ell in range(lmax + 1):
        if ell:
            real_power, imag_power = (
                real_power * x - imag_power * y,
                real_power * y + imag_power * x,
            )
        current = []
        for m in range(ell + 1):
            if m == ell:
                cosine = phases[m] * real_power
                sine = phases[m] * imag_power
            elif ell == m + 1:
                cosine = (2 * m + 1) * z * previous[m][0]
                sine = (2 * m + 1) * z * previous[m][1]
            else:
                cosine = (
                    (2 * ell - 1) * z * previous[m][0] - (ell + m - 1) * previous2[m][0]
                ) / (ell - m)
                sine = (
                    (2 * ell - 1) * z * previous[m][1] - (ell + m - 1) * previous2[m][1]
                ) / (ell - m)
            current.append((cosine, sine))
        yield ell, current
        previous2, previous = previous, current


def make_fused_d_kernels(lmax):
    """Return pure-Torch forward/transpose D tile contraction closures."""
    normalization = _normalizations(lmax)
    root_two = math.sqrt(2.0)

    def harmonic_column(ell, signed_m, degree):
        m = abs(signed_m)
        norm = normalization[ell * ell + ell + signed_m]
        if signed_m < 0:
            return root_two * ((-1) ** m) * norm * degree[m][1]
        if signed_m == 0:
            return norm * degree[0][0]
        return root_two * ((-1) ** m) * norm * degree[m][0]

    def forward(unit, ratio, coefficients, double_scale):
        # unit/ratio: SxGx(3/1), coefficients: SxQxK; output: GxK.
        x, y, z = unit.unbind(-1)
        result = torch.zeros(
            (unit.shape[1], coefficients.shape[-1]),
            dtype=unit.dtype,
            device=unit.device,
        )
        for ell, degree in _degree_values(x, y, z, lmax):
            radial = ratio ** (-(ell + 1))
            for signed_m in range(-ell, ell + 1):
                q = ell * ell + ell + signed_m
                amplitude = (
                    harmonic_column(ell, signed_m, degree) * radial * double_scale[q]
                )
                # The largest temporary is SxGxK, never SxGxQ.
                result = result + (
                    amplitude[:, :, None] * coefficients[:, q, None, :]
                ).sum(dim=0)
        return result

    def transpose(unit, ratio, left, double_scale):
        # left: GxK; each Q column is integrated as soon as it is generated.
        x, y, z = unit.unbind(-1)
        columns = []
        for ell, degree in _degree_values(x, y, z, lmax):
            radial = ratio ** (-(ell + 1))
            for signed_m in range(-ell, ell + 1):
                q = ell * ell + ell + signed_m
                amplitude = (
                    harmonic_column(ell, signed_m, degree) * radial * double_scale[q]
                )
                columns.append((amplitude[:, :, None] * left[None, :, :]).sum(dim=1))
        return torch.stack(columns, dim=1)

    return forward, transpose


class FusedValueStreamedDDPCM(StreamedDDPCM):
    """Value-only D/D-transpose experiment with fused harmonic contraction."""

    kernel_policy = "research-cpu-fp64-fused-d-exact-zero-pruned-values-v2"

    def __init__(self, *args, compile_kernels=False, **kwargs):
        if not isinstance(compile_kernels, bool):
            raise ValueError("compile_kernels must be a boolean.")
        lmax = kwargs.get("lmax", 15)
        forward, transpose = make_fused_d_kernels(lmax)
        if compile_kernels:
            forward = torch.compile(forward, fullgraph=True, dynamic=True)
            transpose = torch.compile(transpose, fullgraph=True, dynamic=True)
        self._fused_forward = forward
        self._fused_transpose = transpose
        self._compile_kernels = compile_kernels
        source = Path(__file__)
        self._fused_source_id = hashlib.sha256(source.read_bytes()).hexdigest()
        super().__init__(*args, **kwargs)

    def _controls(self):
        return super()._controls() + (
            id(self._fused_forward),
            id(self._fused_transpose),
            self._compile_kernels,
            self.kernel_policy,
            self._fused_source_id,
        )

    def _apply_fused_d(self, vector, transpose):
        self._check()
        if not torch.is_tensor(vector) or vector.ndim not in (1, 2):
            raise ValueError("operator input must be a vector or RHS block.")
        if vector.shape[0] != self.dimension:
            raise ValueError("operator dimension mismatch.")
        if vector.ndim == 2 and vector.shape[1] == 0:
            raise ValueError("RHS block must contain at least one column.")
        if vector.ndim == 2 and vector.shape[1] > 8:
            raise MemoryError(
                "This bounded research operator accepts at most 8 RHS columns."
            )
        columns = 1 if vector.ndim == 1 else vector.shape[1]
        estimate = known_resource_estimate(
            self.n, self.lmax, self.g, self.tile, columns
        )
        if estimate["estimated_known_peak_bytes"] > _KNOWN_WORKSPACE_LIMIT_BYTES:
            raise MemoryError(
                "Combined RHS/tile workspace exceeds its research budget."
            )
        self._validate(vector, vector.shape, "operator vector")

        begun = time.perf_counter()
        was_vector = vector.ndim == 1
        values = vector.reshape(self.n, self.q, -1)
        result = torch.zeros_like(values)
        p = self.parameters
        design, weights = p._design, p._weights
        for i in range(self.n):
            delta, distance, ratio, _ = self._row_geometry(i)
            # Geometry/topology certification above remains all-node.  Only the
            # subsequent value contraction drops nodes with an exactly zero D
            # row weight; this changes neither discretization nor tolerance.
            active = self.ui[i] > 0
            if not bool(active.any()):
                # In transpose mode result[i] may already contain contributions
                # from earlier target rows, so it must never be overwritten.
                continue
            active_design = design[active]
            target_weight = weights[active] * self.ui[i, active]
            if transpose:
                result[i] += (
                    -0.5
                    * p._single_scale[:, None]
                    * (
                        active_design.T
                        @ (target_weight[:, None] * (active_design @ values[i]))
                    )
                )
                left = target_weight[:, None] * (active_design @ values[i])
            else:
                result[i] += active_design.T @ (
                    target_weight[:, None]
                    * (active_design @ (-0.5 * p._single_scale[:, None] * values[i]))
                )
                field = torch.zeros(
                    (int(active.sum()), values.shape[-1]), dtype=torch.float64
                )

            indices = [j for j in range(self.n) if j != i]
            for start in range(0, len(indices), self.tile):
                js = indices[start : start + self.tile]
                unit = (
                    delta[active][:, js, :].permute(1, 0, 2)
                    / distance[active][:, js].T[:, :, None]
                ).contiguous()
                radial_ratio = ratio[active][:, js].T.contiguous()
                if transpose:
                    result[js] += self._fused_transpose(
                        unit, radial_ratio, left, p._double_scale
                    )
                else:
                    field += self._fused_forward(
                        unit, radial_ratio, values[js], p._double_scale
                    )
            if not transpose:
                result[i] += active_design.T @ (target_weight[:, None] * field)

        self.action_calls += 1
        self.action_seconds += time.perf_counter() - begun
        output = result.reshape(self.dimension, -1)
        return output[:, 0] if was_vector else output

    @torch.no_grad()
    def apply(self, kind, vector, *, transpose=False):
        if kind == "D":
            return self._apply_fused_d(vector, transpose)
        # A/C recurse through self.apply("D"), while L retains the frozen path.
        return super().apply(kind, vector, transpose=transpose)

    def storage_report(self):
        result = super().storage_report()
        conservative_sgq_budget = result["kernel_tile_bytes"]
        sources = min(self.tile, max(0, self.n - 1))
        result.update(
            {
                "kernel_policy": self.kernel_policy,
                "kernel_source_sha256": {
                    "fused_value_operator.py": self._fused_source_id
                },
                "compiled_value_kernels": self._compile_kernels,
                "fused_d_materializes_source_grid_basis": False,
                "fused_d_exact_zero_node_pruning": "ui > 0 after all-node certification",
                "fused_d_materialized_kernel_tile_bytes": 0,
                "reference_l_kernel_tile_bytes": conservative_sgq_budget,
                "inherited_conservative_sgq_budget_bytes": conservative_sgq_budget,
                "fused_d_max_sgk_tile_bytes": sources * self.g * 8 * 8,
                "fused_d_recurrence_state_upper_bytes": (
                    6 * (self.lmax + 1) * sources * self.g * 8
                ),
                "fused_d_largest_named_tile_rank": "SxGxK",
                "analytic_derivative_kernel_qualified": False,
            }
        )
        return result
