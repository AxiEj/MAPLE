"""Explicit execution controls for the two pretrained solvent-model routes.

Owned contexts temporarily enforce IEEE controls and restore caller state.
Numerical admission belongs to the existing hash-bound model-card policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib import metadata
import re
from threading import RLock

_PRECISION_LOCK = RLock()
# Keys verified against the pinned Torch 2.12 Inductor config. No cache-limit changes.
STRICT_INDUCTOR_OPTIONS = {
    "use_fast_math": False,
    "cuda.use_fast_math": False,
    "cutlass.use_fast_math": False,
    "force_same_precision": True,
}


def is_pretrained_solvent_route(model, solvation_options=None):
    return str(model).strip().lower() == "consolv" or (
        isinstance(solvation_options, dict)
        and str(solvation_options.get("method", "")).lower() == "anysolv"
    )


def normalize_execution_options(options):
    """Validate header controls without touching CUDA or loading a model."""
    dtype = str(options.get("dtype", "float32")).removeprefix("torch.").strip().lower()
    inference = str(options.get("inference", "reference")).strip().lower()
    if dtype not in {"float32", "float64"}:
        raise ValueError("Solvent-model dtype must be float32 or float64.")
    if inference not in {"reference", "optimized"}:
        raise ValueError("Solvent-model inference must be reference or optimized.")
    if inference == "optimized" and dtype != "float32":
        raise ValueError("Optimized solvent inference requires CUDA and float32.")
    return {"dtype": dtype, "inference": inference}


@dataclass(frozen=True)
class SolvationExecution:
    device: object = "cuda:0"
    dtype: object = "float32"
    inference: str = "reference"

    def __post_init__(self):
        import torch

        device = str(self.device).strip().lower()
        if device != "cpu":
            match = re.fullmatch(r"(?:cuda|gpu)(?::?(\d+))?", device)
            if match is None:
                raise ValueError(
                    "Solvent models require an explicit CPU or CUDA device."
                )
            index = int(match.group(1) or 0)
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested CUDA device {index} is unavailable; CPU fallback is disabled."
                )
            device = f"cuda:{index}"
        controls = normalize_execution_options(
            {"dtype": self.dtype, "inference": self.inference}
        )
        dtype, inference = controls["dtype"], controls["inference"]
        if inference == "optimized" and (device == "cpu" or dtype != "float32"):
            raise ValueError("Optimized solvent inference requires CUDA and float32.")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "inference", inference)

    @property
    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype)

    @property
    def provenance(self):
        return {
            "device": self.device,
            "compute_dtype": self.dtype,
            "inference_mode": self.inference,
            "matmul_precision": "highest",
            "tf32": False,
            "autocast": False,
        }

    @contextmanager
    def precision_context(self):
        """Serialize owned execution and restore caller precision on every exit.

        Autocast is thread local; matmul/cuDNN switches are process global. Other
        Torch users must cooperate with this lock to share these global switches.
        """
        import torch

        with _PRECISION_LOCK:
            # Legacy setters change per-operation precision, including CPU
            # matmul. Snapshot the actual hierarchy rather than just booleans;
            # restoring booleans loses inherited "none" and medium/bf16 state.
            precision_scopes = (
                torch.backends.cuda.matmul,
                torch.backends.mkldnn.matmul,
                torch.backends.cudnn.conv,
                torch.backends.cudnn.rnn,
            )
            previous_matmul = torch.get_float32_matmul_precision()
            previous_cudnn = torch.backends.cudnn.allow_tf32
            previous = tuple(scope.fp32_precision for scope in precision_scopes)
            try:
                torch.set_float32_matmul_precision("highest")
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.conv.fp32_precision = "ieee"
                torch.backends.cudnn.rnn.fp32_precision = "ieee"
                compiler = nullcontext()
                if self.inference == "optimized":
                    import torch._inductor.config as config

                    compiler = config.patch(STRICT_INDUCTOR_OPTIONS)
                with (
                    torch.autocast(
                        device_type=self.device.split(":")[0], enabled=False
                    ),
                    compiler,
                ):
                    yield self
            finally:
                # Restore the legacy precision enum before its per-op overrides.
                torch.set_float32_matmul_precision(previous_matmul)
                torch.backends.cudnn.allow_tf32 = previous_cudnn
                for scope, precision in zip(precision_scopes, previous):
                    scope.fp32_precision = precision

    def precision_state(self):
        """Cheap live flags only; no package versions or hardware fingerprint."""
        import torch

        device_type = self.device.split(":")[0]
        compiler = {}
        if self.inference == "optimized":
            import torch._inductor.config as config

            compiler = {key: getattr(config, key) for key in STRICT_INDUCTOR_OPTIONS}
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        autocast = torch.is_autocast_enabled(device_type)
        return {
            "compute_dtype": self.dtype,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": matmul_tf32,
            "cudnn_tf32": cudnn_tf32,
            "autocast": autocast,
            "compiler_options": compiler,
            "available_reduced_precision_flags": {
                name: getattr(torch.backends.cuda.matmul, name)
                for name in (
                    "allow_fp16_reduced_precision_reduction",
                    "allow_bf16_reduced_precision_reduction",
                    "allow_fp16_reduced_precision_reduction_split_k",
                    "allow_bf16_reduced_precision_reduction_split_k",
                    "allow_fp16_accumulation",
                )
                if hasattr(torch.backends.cuda.matmul, name)
            },
        }

    def check_precision(self, *, inputs=(), model=None):
        """Validate effective flags and all floating input/parameter/buffer dtypes."""
        import torch

        state = self.precision_state()
        if (
            state["matmul_precision"] != "highest"
            or state["matmul_tf32"]
            or state["cudnn_tf32"]
            or state["autocast"]
        ):
            raise ValueError(
                "Strict solvent precision requires highest matmul precision, "
                "both matmul/cuDNN TF32 off, and autocast off."
            )
        if (
            self.inference == "optimized"
            and state["compiler_options"] != STRICT_INDUCTOR_OPTIONS
        ):
            raise ValueError(
                "Strict solvent compilation requires explicit IEEE compiler options."
            )

        def check(value):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() and value.dtype != self.torch_dtype:
                    raise ValueError(
                        "Solvent floating inputs/parameters/buffers must match selected dtype."
                    )
            elif isinstance(value, Mapping):
                for item in value.values():
                    check(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)

        check(inputs)
        if isinstance(model, torch.nn.Module):
            for tensor in (*model.parameters(), *model.buffers()):
                check(tensor)
        return state

    def runtime_context(self):
        """Full reporting/card fingerprint; caller must own precision_context()."""
        import torch

        if self.device == "cpu":
            return None
        state = self.check_precision()
        versions = {"torch": torch.__version__, "cuda": torch.version.cuda or "none"}
        for distribution in ("e3nn", "fairchem-core", "triton"):
            try:
                versions[distribution.replace("-", "_")] = metadata.version(
                    distribution
                )
            except metadata.PackageNotFoundError:
                continue
        # These are effective use flags, with their derivation/applicability
        # explicitly recorded below, rather than invented global Torch settings.
        controls = {
            "compute_dtype": self.dtype,
            "matmul_precision": state["matmul_precision"],
            "autocast": state["autocast"],
            "tf32": state["matmul_tf32"] or state["cudnn_tf32"],
            "float16": False,
            "bfloat16": False,
            "reduced_matmul": False,
            "fast_math": any(
                state["compiler_options"].get(key, False)
                for key in (
                    "use_fast_math",
                    "cuda.use_fast_math",
                    "cutlass.use_fast_math",
                )
            ),
            "relaxed_convergence": False,
            "shortened_sampling": False,
        }
        return {
            "accelerator_backend": "cuda",
            "hardware": torch.cuda.get_device_name(self.device),
            "software_versions": versions,
            "precision_controls": controls,
            "precision_state": state,
            "precision_protocol": {
                "relaxed_convergence": "inapplicable_direct_feed_forward_sp",
                "shortened_sampling": "inapplicable_direct_feed_forward_sp",
                "float16_bfloat16_reduced_matmul": "unused_selected_fp32_or_fp64_inputs_parameters_buffers_and_autocast_off",
                "fast_math": (
                    "owned_inductor_options"
                    if self.inference == "optimized"
                    else "inapplicable_reference_eager"
                ),
            },
        }

    def validate_card(self, card, *, task="sp"):
        with self.precision_context():
            card.validate_device(
                self.device,
                task=task,
                inference_mode=self.inference,
                runtime_context=self.runtime_context(),
            )
