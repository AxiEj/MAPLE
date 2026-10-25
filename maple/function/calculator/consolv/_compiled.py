"""Owner-local static export/compile for the frozen ConSolv energy graph.

Export gives each signature a real FX program rather than specializing the same
Python frame indefinitely. No Dynamo limit, code object, or global cache is
modified. An owner permits exactly energy and energy-with-autograd variants.
"""

from threading import RLock
from weakref import ref

import torch

from ..solvation_execution import STRICT_INDUCTOR_OPTIONS


class _EnergyGraph(torch.nn.Module):
    def __init__(self, potential):
        super().__init__()
        self.potential = potential

    def forward(self, species, positions, solvent_index, sender, receiver):
        return self.potential._energy_graph(
            species, positions, solvent_index, sender, receiver
        )


class _FunctionalEnergyGraph(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def forward(self, state, species, positions, solvent_index, sender, receiver):
        return torch.func.functional_call(
            self.graph,
            state,
            (species, positions, solvent_index, sender, receiver),
            strict=True,
        )


class ConSolvCompiledGraph:
    """One immutable graph signature, two bounded differentiable variants."""

    def __init__(self, potential):
        self._potential = ref(potential)
        floating = next(
            (value for value in potential.buffers() if value.is_floating_point()), None
        )
        if floating is None:
            raise ValueError(
                "ConSolv compilation requires frozen floating-point model state."
            )
        self._device, self._dtype = floating.device, floating.dtype
        self._identity = (
            getattr(potential, "_checkpoint_sha256", None),
            getattr(potential, "_potential_state", None),
        )
        self._signature = None
        self._variants = {}
        self._state = None
        self.export_records = []
        self._lock = RLock()

    @staticmethod
    def _input_signature(species, positions, solvent_index, sender, receiver):
        tensors = (species, positions, sender, receiver)
        if (
            species.ndim != 1
            or positions.shape != (species.numel(), 3)
            or sender.ndim != 1
            or receiver.shape != sender.shape
        ):
            raise ValueError("ConSolv optimized inputs have inconsistent graph shapes.")
        if any(not value.is_contiguous() for value in tensors):
            raise ValueError("ConSolv optimized inputs must be contiguous.")
        if any(value.dtype != torch.long for value in (species, sender, receiver)):
            raise ValueError("ConSolv optimized indices must use torch.int64.")
        if any(value.device != positions.device for value in tensors):
            raise ValueError("ConSolv optimized inputs must share one device.")
        return (
            tuple(species.shape),
            tuple(positions.shape),
            int(solvent_index),
            tuple(sender.shape),
            tuple(receiver.shape),
            str(positions.device),
            positions.dtype,
        )

    def _prepare(self, args):
        from torch._dynamo.backends.registry import lookup_backend

        potential = self._potential()
        if potential is None:
            raise RuntimeError("ConSolv compiled potential has been released.")
        graph = _EnergyGraph(potential).eval()
        buffers_before = {
            name: (value, value._version) for name, value in graph.named_buffers()
        }
        if self._state is None:
            # Detached views share the immutable weights without making them
            # module constants retained by the global compiler caches.
            self._state = {
                name: value.detach()
                for name, value in (*graph.named_parameters(), *graph.named_buffers())
            }
        program = torch.export.export(
            _FunctionalEnergyGraph(graph).eval(),
            args=(self._state, *args),
            strict=True,
        )
        graph_owner = program.module()
        if graph_owner.state_dict():
            raise RuntimeError(
                "ConSolv functional export unexpectedly retained model state."
            )
        if program.tensor_constants or any(
            spec.kind.name != "USER_INPUT"
            for spec in program.graph_signature.input_specs
        ):
            raise RuntimeError(
                "ConSolv functional export must lift all model state as user inputs."
            )
        buffers_after = dict(graph.named_buffers())
        if any(
            buffers_after[name] is not value or value._version != version
            for name, (value, version) in buffers_before.items()
        ):
            raise RuntimeError("ConSolv export changed the source model state.")
        self.export_records.append(
            {
                "state_tensors": len(self._state),
                "user_inputs": len(program.graph_signature.input_specs),
                "tensor_constants": 0,
                "owned_state_tensors": 0,
                "source_buffer_identity_and_versions_unchanged": True,
            }
        )
        state = {"compilations": 0, "executions": 0}
        inductor = lookup_backend("inductor")

        def backend(graph, inputs):
            if state["compilations"]:
                raise RuntimeError(
                    "ConSolv static graph unexpectedly requested recompilation."
                )
            compiled = inductor(
                graph, inputs, config_patches=dict(STRICT_INDUCTOR_OPTIONS)
            )
            state["compilations"] += 1

            def execute(*values):
                result = compiled(*values)
                state["executions"] += 1
                return result

            return execute

        compiled = torch.compile(
            graph_owner, backend=backend, fullgraph=True, dynamic=False
        )
        return compiled, state

    def __call__(self, *args):
        grad = args[1].requires_grad
        with self._lock, torch.inference_mode(False), torch.set_grad_enabled(grad):
            if args[1].device != self._device or args[1].dtype != self._dtype:
                raise ValueError(
                    "ConSolv compiled inputs must match the potential device and dtype."
                )
            signature = (self._identity, self._input_signature(*args))
            if self._signature is not None and signature != self._signature:
                raise ValueError(
                    "ConSolv optimized graph signature changed; construct a new calculator."
                )
            self._signature = signature
            if grad not in self._variants:
                self._variants[grad] = self._prepare(args)
            compiled, state = self._variants[grad]
            before = state["executions"]
            result = compiled(self._state, *args)
            if state["executions"] <= before:
                raise RuntimeError("ConSolv compiled backend did not execute.")
            return result
