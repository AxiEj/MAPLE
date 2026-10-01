"""Separately identified CPU value-kernel experiment; derivatives unsupported."""

import hashlib
from pathlib import Path

import torch

from harmonic_value_kernel import make_harmonic_value_kernel
from streamed_ddpcm import StreamedDDPCM


class CompiledValueStreamedDDPCM(StreamedDDPCM):
    """Compile only a real FP64 harmonic value recurrence, not solver/AD graphs."""

    kernel_policy = "research-cpu-fp64-real-sectoral-compiled-values-v1"

    def __init__(self, *args, **kwargs):
        lmax = kwargs.get("lmax", 15)
        eager = make_harmonic_value_kernel(lmax, real_sectoral=True)
        self._compiled_evaluator = torch.compile(eager, fullgraph=True, dynamic=True)
        self._kernel_sources = tuple(
            (name, hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
            for name in ("compiled_value_operator.py", "harmonic_value_kernel.py")
        )
        super().__init__(*args, **kwargs)

    def _controls(self):
        return super()._controls() + (id(self._compiled_evaluator), self.kernel_policy,
                                     self._kernel_sources)

    def _harmonic_design(self, directions):
        return self._compiled_evaluator(directions.contiguous())

    def storage_report(self):
        result = super().storage_report()
        result["kernel_policy"] = self.kernel_policy
        result["kernel_source_sha256"] = dict(self._kernel_sources)
        result["compiler_workspace_not_in_static_tensor_count"] = True
        result["analytic_derivative_kernel_qualified"] = False
        return result
