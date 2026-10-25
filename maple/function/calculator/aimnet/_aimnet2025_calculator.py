"""Exact official AIMNet2-2025 member0, complete native N32 scalar boundary.

No legacy model, external fixed solvent charge, global dtype or installed
package is modified. The public energy-only call ALWAYS receives a grad leaf,
so official AIMNet selects its differentiable Torch D3 path for E/F/H/HVP.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import sys
from threading import RLock, local

from ase.calculators.calculator import all_changes
import numpy as np

from ..analytic_protocol import (
    AnalyticCapabilityError, AnalyticIdentityError, CheckpointFileIdentity,
    NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM, geometry_sha256,
    named_array_sha256, require_no_amp_tf32, torch_model_runtime_identity,
    torch_response_settings,
)
from ..calculator_base import CalcABC, EV2HARTREE, parse_bool_option, register_calculator
from ._aimnet2025_identity import (
    CHECKPOINT_FILENAME, CHECKPOINT_SHA256, INSTALL_HINT, MODEL_METADATA,
    PROFILE_ID, REGISTRY_KEY, bind_external_runtime as _bind_external_runtime,
    require_external_sources as _require_external_sources,
    canonical_neighbor_graph_sha256, NEIGHBOR_GRAPH_SCHEME, PRECISION_EXECUTION_POLICY,
)


_PRECISION_LOCK = RLock()
_PRECISION_NESTING = local()


def _caller_precision():
    import torch

    return {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "grad_enabled": bool(torch.is_grad_enabled())}


def _require_operation_precision():
    # CPU operations use the same effective settings. Neither AMP nor inference
    # mode is disabled/bypassed, and the common strict guard remains unchanged.
    require_no_amp_tf32("cpu")
    require_no_amp_tf32("cuda")


@contextmanager
def _native_precision_context(calculator, operation):
    """Apply this model's IEEE profile and restore caller state even on errors.

    The lock serializes our own instances/contexts only. Torch backend flags are
    process-wide: unrelated models/threads must coordinate their own execution.
    Nested entry checks the active profile, never masks an internal TF32 toggle.
    """
    if sys.version_info < (3, 11):
        raise RuntimeError(INSTALL_HINT)
    try:
        import torch
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc

    with _PRECISION_LOCK:
        depth = getattr(_PRECISION_NESTING, "depth", 0)
        before = _caller_precision()
        # Audit public selector metadata too. Avoid redundant Boolean writes:
        # setting an already-False matmul flag loses inherited 'none' metadata.
        backends = {"matmul": torch.backends.cuda.matmul, "cudnn": torch.backends.cudnn,
                    "conv": torch.backends.cudnn.conv, "rnn": torch.backends.cudnn.rnn}
        selectors = {name: backend.fp32_precision for name, backend in backends.items()}
        record = {"policy": PRECISION_EXECUTION_POLICY, "operation": operation,
                  "caller_before": before, "effective": None,
                  "caller_after": None, "status": "error",
                  "caller_backend_selectors_before": selectors,
                  "caller_backend_selectors_after": None}
        _PRECISION_NESTING.depth = depth + 1
        try:
            if depth == 0:
                if before["cuda_matmul_allow_tf32"]:
                    torch.backends.cuda.matmul.allow_tf32 = False
                if before["cudnn_allow_tf32"]:
                    torch.backends.cudnn.allow_tf32 = False
            else:
                _require_operation_precision()
            with torch.enable_grad():
                _require_operation_precision()
                record["effective"] = _caller_precision()
                yield
                _require_operation_precision()
                if not torch.is_grad_enabled():
                    raise AnalyticCapabilityError("AIMNet2025 operation lost its grad-enabled precision profile.")
            record["status"] = "success"
        except Exception:
            if hasattr(calculator, "results"):
                calculator.reset()
            # A geometry-specific cutoff refusal does not invalidate a cell.
            # Source/config drift already revokes it in _require_runtime; an
            # actual precision violation must revoke it here as well.
            try:
                _require_operation_precision()
            except Exception:
                calculator.analytic_implicit_derivatives_admitted = False
            raise
        finally:
            _PRECISION_NESTING.depth = depth
            if depth == 0:
                try:
                    if torch.backends.cuda.matmul.allow_tf32 != before["cuda_matmul_allow_tf32"]:
                        torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
                finally:
                    if torch.backends.cudnn.allow_tf32 != before["cudnn_allow_tf32"]:
                        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
                record["caller_after"] = _caller_precision()
                record["caller_backend_selectors_after"] = {
                    name: backend.fp32_precision for name, backend in backends.items()}
                calculator.last_precision_execution_provenance = record
                execution = getattr(calculator, "execution_provenance", None)
                if isinstance(execution, dict):
                    execution["precision_execution"] = deepcopy(record)


def _native_operation(method):
    @wraps(method)
    def execute(self, *args, **kwargs):
        with _native_precision_context(self, method.__name__):
            return method(self, *args, **kwargs)
    return execute


def _import_native_backend():
    try:
        from aimnet.calculators import AIMNet2Calculator
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return AIMNet2Calculator


def _resolve_checkpoint(model_path):
    # Intentionally never downloads or writes the cache. An explicit path is
    # never treated as a registry alias, and a wrong cached hash never retries.
    path = Path(model_path).expanduser() if model_path is not None else (
        Path(os.environ.get("AIMNET_CACHE_DIR", str(Path.home() / ".cache/aimnet")))
        / CHECKPOINT_FILENAME
    )
    if not path.is_file():
        offline = os.environ.get("MAPLE_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
        raise FileNotFoundError(
            f"AIMNet2025 exact member0 checkpoint is missing: {path}. "
            + ("MAPLE_OFFLINE is enabled; no network was attempted. " if offline else "No automatic download is performed. ")
            + f"Provide model_path or provision the official registry entry {REGISTRY_KEY} "
            f"in the official AIMNet cache (SHA256 {CHECKPOINT_SHA256})."
        )
    bound = CheckpointFileIdentity.bind(path)
    if bound.sha256 != CHECKPOINT_SHA256:
        raise ValueError(
            f"AIMNet2025 checkpoint SHA256 mismatch BEFORE load: {bound.sha256}; "
            f"expected {CHECKPOINT_SHA256}. No member/model fallback."
        )
    return bound


def _verify_checkpoint_state(model, path):
    """Verify original loaded tensors bitwise, including NaN embedding rows."""
    import torch

    original = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    loaded = model.state_dict()
    omitted = set(original) - set(loaded)
    if omitted != {"outputs.dipole.mass", "outputs.quadrupole.mass"} or set(loaded) - set(original):
        raise AnalyticIdentityError("AIMNet2025 checkpoint loaded-state identity mismatch.")
    for name, value in loaded.items():
        source = original[name]
        if value.dtype != source.dtype or not torch.equal(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            source.contiguous().reshape(-1).view(torch.uint8),
        ):
            raise AnalyticIdentityError(f"AIMNet2025 checkpoint tensor identity changed: {name}.")


def _configuration(backend):
    """Model-specific effective physics; no captured configuration is patched."""
    c, d, m = backend.external_coulomb, backend.external_dftd3, backend.model
    if c is None or d is None:
        raise AnalyticIdentityError("AIMNet2025 profile requires native Coulomb and D3BJ.")
    sr = m.outputs["srcoulomb"]
    return {
        "metadata": dict(backend.metadata),
        "engine": "official aimnet public scalar; forces=False,hessian=False; requires_grad leaf",
        "train": backend._train, "compiled": backend._was_compiled,
        "device": str(backend.device),
        "lr": backend.lr, "cutoff": backend.cutoff,
        "cutoff_lr": str(backend.cutoff_lr),
        "coulomb_method": backend._coulomb_method,
        "coulomb_cutoff": str(backend._coulomb_cutoff),
        "d3_cutoff": backend._dftd3_cutoff, "nb_threshold": backend.nb_threshold,
        "neighbor_cutoffs": {
            name: getattr(getattr(backend, name), "cutoff", None)
            for name in ("_nblist", "_nblist_lr", "_nblist_dftd3", "_nblist_coulomb")
        },
        "input_dtypes": {key: str(value) for key, value in backend.keys_in.items()},
        "model_input_dtypes": [str(value) for value in m._required_keys_dtype],
        "model_flags": {
            name: {key: getattr(child, key) for key in (
                "d2features", "do_vector", "num_charge_channels", "dmat_fill",
                "approximate", "reduce_sum", "key_in", "key_out"
            ) if hasattr(child, key)}
            for name, child in m.named_modules()
        },
        "coulomb": {"method": c.method, "subtract_sr": c.subtract_sr,
                    "rc": float(c.rc.item()), "envelope": c.envelope,
                    "key_in": c.key_in, "key_out": c.key_out, "factor": c._factor},
        "embedded_sr": {"rc": float(sr.rc.item()), "envelope": sr.envelope,
                        "key_in": sr.key_in, "key_out": sr.key_out, "factor": sr._factor},
        "d3": {key: getattr(d, key) for key in
               ("s6", "s8", "a1", "a2", "smoothing_on", "smoothing_off", "key_out")},
    }


@register_calculator
class AIMNet2025Calculator(CalcABC):
    implemented_properties = ["energy", "free_energy", "forces", "hessian"]
    MODEL_NAMES = ("aimnet2025", "aimnet2-2025", "aimnet2_2025")
    MODEL_ENERGY_UNIT = "eV"
    IMPLICIT_SOLVENT_DTYPE = "float32"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    SUPPORTS_IMPLICIT_SOLVATION = True
    ANALYTIC_IMPLICIT_HESSIAN_MODELS = MODEL_NAMES
    ANALYTIC_IMPLICIT_HESSIAN_DEVICES = ("cpu", "cuda")
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ("dtype", "d4", "model_path")
    MODEL_PATH_OPTION = "model_path"

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {"dtype": options.get("dtype", "auto"),
                  "d4": parse_bool_option(options.get("d4", False), name="d4"),
                  "hessian": options.get("hessian", "analytic")}
        path = resolved_model_path if resolved_model_path is not None else options.get("model_path")
        if path is not None:
            kwargs["model_path"] = path
        return kwargs

    @_native_operation
    def __init__(self, device, model="aimnet2025", model_path=None,
                 implicit="none", solvent="none", *, dtype="auto", d4=False,
                 hessian="analytic"):
        super().__init__()
        if sys.version_info < (3, 11):
            raise RuntimeError(INSTALL_HINT)
        if model.lower() not in self.MODEL_NAMES:
            raise ValueError(f"Unsupported AIMNet2025 model={model!r}; legacy models are separate.")
        if dtype not in {"auto", "float32"}:
            raise ValueError(f"Unsupported AIMNet2025 dtype={dtype!r}; native NN supports auto/float32 only, not float64.")
        if parse_bool_option(d4, name="d4"):
            raise ValueError("AIMNet2025 requires its native D3BJ; additional D4 is unsupported (d4=False only).")
        if hessian not in self.SUPPORTED_HESSIAN_MODES:
            raise ValueError(f"Unsupported AIMNet2025 hessian={hessian!r}.")
        try:
            import torch
        except ImportError as exc:
            raise ImportError(INSTALL_HINT) from exc
        if torch.get_default_dtype() is not torch.float32:
            raise AnalyticCapabilityError("AIMNet2025 native N32 requires the unchanged float32 default dtype.")
        self.requested_device = str(device)
        self.device = torch.device(device)
        if self.device.type not in self.ANALYTIC_IMPLICIT_HESSIAN_DEVICES:
            raise ValueError("AIMNet2025 supports only the requested CPU/CUDA native backend.")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype, self._requested_dtype = torch.float32, dtype
        self.hessian, self.d4 = hessian, False
        self.implicit_solv_init(implicit, solvent)
        self._checkpoint = _resolve_checkpoint(model_path)
        self._external_identity, self._external_sources = _bind_external_runtime()
        native = _import_native_backend()
        self.backend = native(self._checkpoint.path, device=str(self.device), train=False, compile_model=False)
        self._loaded_backend = self.backend
        self.model = self.backend.model
        _verify_checkpoint_state(self.model, self._checkpoint.path)
        self._validate_native_profile()
        self.atomic_numbers = list(self.backend.metadata["implemented_species"])
        self._native_config = deepcopy(_configuration(self.backend))
        self._loaded_device = str(self.device)
        self._live_identity = self._live_runtime_identity()
        self._state_sha256 = self._state_digest()
        self._analytic_prepared = False
        self.last_analytic_point_identity = None
        self.last_hvp_provenance = None
        self._evaluated_correction = None
        self._require_runtime()

    def _modules(self):
        return (("model", self.model), ("coulomb", self.backend.external_coulomb),
                ("d3", self.backend.external_dftd3))

    def _validate_native_profile(self):
        import torch

        config = _configuration(self.backend)
        for key, value in MODEL_METADATA.items():
            if config["metadata"].get(key) != value:
                raise AnalyticIdentityError(f"AIMNet2025 native profile metadata mismatch: {key}.")
        c = config["coulomb"]
        if (config["train"] or config["compiled"] or self.model.training
                or config["cutoff"] != 5.0 or not config["lr"]
                or config["cutoff_lr"] != "inf" or config["coulomb_cutoff"] != "inf"
                or config["coulomb_method"] != "simple" or c["method"] != "simple"
                or c["subtract_sr"] is not False or c["rc"] != MODEL_METADATA["coulomb_sr_rc"]
                or c["envelope"] != "exp" or config["d3_cutoff"] != 15.0
                or config["d3"] != dict(MODEL_METADATA["d3_params"], smoothing_on=12.0,
                                        smoothing_off=15.0, key_out="energy")):
            raise AnalyticIdentityError("AIMNet2025 native physics/config profile mismatch.")
        if self.backend.keys_in["coord"] is not torch.float32 or self.model._required_keys_dtype[0] is not torch.float32:
            raise AnalyticIdentityError("AIMNet2025 native coordinate dtype profile mismatch.")
        for label, module in self._modules():
            for name, value in list(module.named_parameters()) + list(module.named_buffers()):
                expected = torch.float64 if label == "model" and name == "outputs.atomic_shift.shifts.weight" else torch.float32
                if value.is_floating_point() and value.dtype is not expected:
                    raise AnalyticIdentityError(f"AIMNet2025 native dtype profile mismatch: {label}.{name}.")
                if value.device != self.device:
                    raise AnalyticIdentityError("AIMNet2025 requested model/device identity mismatch.")
        if str(torch.device(self.backend.device)) != str(self.device):
            raise AnalyticIdentityError("AIMNet2025 requested backend/device identity mismatch.")

    def _live_runtime_identity(self):
        rows = []
        for label, module in self._modules():
            if module is None:
                raise AnalyticIdentityError("AIMNet2025 native module identity changed.")
            rows.append((label, id(module), torch_model_runtime_identity(module),
                         tuple((name, id(child), type(child), child.training) for name, child in module.named_modules()),
                         tuple((name, parameter.requires_grad) for name, parameter in module.named_parameters())))
        return tuple(rows)

    def _state_digest(self):
        return named_array_sha256({f"{label}.{name}": tensor.detach().cpu().numpy()
                                   for label, module in self._modules()
                                   for name, tensor in module.state_dict().items()})

    def _require_runtime(self):
        import torch

        try:
            self._checkpoint.require_unchanged()
            _require_external_sources(self._external_sources)
            require_no_amp_tf32(self.device)
            if not torch.is_grad_enabled():
                raise AnalyticCapabilityError("AIMNet2025 requires a differentiable grad-enabled scalar for every operation.")
            if (self.dtype is not torch.float32 or self.d4 is not False
                    or str(self.device) != self._loaded_device
                    or self.backend is not self._loaded_backend
                    or self.model is not self.backend.model
                    or _configuration(self.backend) != self._native_config
                    or self._live_runtime_identity() != self._live_identity):
                raise AnalyticIdentityError("AIMNet2025 native runtime identity/config drift; rebuild the calculator.")
        except Exception:
            self.reset()
            self.analytic_implicit_derivatives_admitted = False
            raise

    @_native_operation
    def prepare_analytic_derivatives(self):
        self._require_runtime()
        # Full source/checkpoint/state hashing is done at preparation, not at
        # every force evaluation. Cheap tensor mutation/config/stat guards stay live.
        try:
            for file in (self._checkpoint, *self._external_sources.files):
                if CheckpointFileIdentity.bind(file.path).sha256 != file.sha256:
                    raise AnalyticIdentityError("AIMNet2025 prepared source/checkpoint identity changed.")
            if self._state_digest() != self._state_sha256:
                raise AnalyticIdentityError("AIMNet2025 prepared model-state identity changed.")
        except Exception:
            self.reset()
            self.analytic_implicit_derivatives_admitted = False
            raise
        self._analytic_prepared = True
        # Only the independent parent admission boundary may set admission.

    @_native_operation
    def prepare_numerical_derivatives(self):
        self._require_runtime()
        return 0.001  # Explicit N32 diagnostic; never an analytic fallback.

    @property
    @_native_operation
    def analytic_cell_identity(self):
        self._require_runtime()
        return deepcopy({"model": "aimnet2025", "registry_key": REGISTRY_KEY,
                         "profile_id": PROFILE_ID, "device": str(self.device),
                         "checkpoint_sha256": self._checkpoint.sha256,
                         "effective_nn_dtype": "float32", "effective_ad_class": "N32",
                         "neighbor_graph_scheme": NEIGHBOR_GRAPH_SCHEME,
                         "precision_execution_policy": PRECISION_EXECUTION_POLICY,
                         "neighbor_capacity_policy": "complete-single-molecule-upper-bound-v1",
                         "atomic_shift_dtype": "float64", "energy_accumulation_dtype": "float64",
                         "d4": False, "source_energy_unit": "eV",
                         "external_runtime": self._external_identity,
                         "native_configuration": self._native_config,
                         "state_sha256": self._state_sha256})

    @property
    @_native_operation
    def runtime_profile_identity(self):
        return self.analytic_cell_identity

    def _solvent_compute_dtype(self):
        correction = self.solvent_correction
        if correction is None:
            return None
        provider = getattr(correction, 'underlying_provider', correction)
        dtype = getattr(provider, 'dtype', None)
        return str(dtype).removeprefix('torch.') if dtype is not None else None

    @property
    @_native_operation
    def inference_precision_provenance(self):
        return {**self.analytic_cell_identity, "requested_dtype": self._requested_dtype,
                "effective_dtype": "float32", "original_checkpoint_path": self._checkpoint.path,
                "original_checkpoint_sha256": self._checkpoint.sha256,
                "in_memory_cast_only": False, "analytic_prepared": self._analytic_prepared,
                "solvent_dtype": self._solvent_compute_dtype(),
                "last_point": deepcopy(self.last_analytic_point_identity)}

    def _validate_atoms(self, atoms):
        self._reject_unsupported_pbc(atoms)
        x = atoms.get_positions()
        if not len(atoms) or not np.isfinite(x).all():
            raise ValueError("AIMNet2025 requires a nonempty finite geometry.")
        unsupported = sorted(set(atoms.numbers) - set(self.atomic_numbers))
        if unsupported:
            raise ValueError(f"AIMNet2025 elements outside official metadata domain: {unsupported}.")
        charge = self._total_charge_from_atoms(atoms)
        mult = float(atoms.info.get("mult", atoms.info.get("multiplicity", 1)))
        if not np.isfinite(charge) or charge != round(charge):
            raise ValueError("AIMNet2025 charge must be a finite integer.")
        if mult != 1:
            raise ValueError("AIMNet2025 supports closed-shell multiplicity 1 only.")
        if self.solvent_correction is not None and charge != 0:
            raise ValueError("AIMNet2025 implicit domain is charge 0 / multiplicity 1 only.")
        return charge, int(mult)

    @_native_operation
    def check_state(self, atoms, tol=1e-15):
        self._require_runtime()
        state = super().check_state(atoms, tol)
        current = self._validate_atoms(atoms)
        if self.atoms is not None and current != self._validate_atoms(self.atoms):
            state.append("charge/multiplicity")
        if self.solvent_correction is not self._evaluated_correction:
            state.append("solvent_correction")
        return state

    @_native_operation
    def _scalar(self, atoms, *, curvature=False):
        """The single public native seam; hooks observe, never replace, math."""
        import torch

        self._require_runtime()
        charge, mult = self._validate_atoms(atoms)
        # The upstream adaptive allocator shrinks after small inputs. In the
        # pinned runtime a later overflowing CPU/CUDA list can be truncated
        # without raising its expected overflow exception. A nonperiodic
        # single molecule has at most N-1 neighbors per atom, so allocating N
        # slots is an exact safety bound, not a cutoff or potential change.
        # Native code still builds/trims the lists and evaluates the potential.
        capacity = max(16, ((len(atoms) + 15) // 16) * 16)
        for name in ("_nblist", "_nblist_lr", "_nblist_coulomb", "_nblist_dftd3"):
            neighbors = getattr(self.backend, name, None)
            if neighbors is not None:
                neighbors.max_neighbors = capacity
        q = torch.tensor(atoms.positions, device=self.device, dtype=torch.float32, requires_grad=True)
        if not torch.isfinite(q).all():
            raise ValueError("AIMNet2025 geometry is nonfinite after native float32 quantization.")
        graph, intermediate = {}, {}

        def observe_input(module, args):
            data = args[0]
            for key, value in data.items():
                if key.startswith(("nbmat", "mol_idx", "numbers", "shifts", "mask_", "nb_pad_mask")) and isinstance(value, torch.Tensor):
                    graph[key] = value.detach().cpu().numpy().copy()

        def observe_output(module, args, output):
            intermediate.update({key: str(value.dtype) for key, value in output.items()
                                 if isinstance(value, torch.Tensor) and value.is_floating_point()})

        pre = self.model.register_forward_pre_hook(observe_input)
        post = self.model.register_forward_hook(observe_output)
        try:
            result = self.backend({"coord": q, "numbers": atoms.numbers, "charge": charge, "mult": mult},
                                  forces=False, hessian=False)
            energy = result["energy"].sum()
        finally:
            pre.remove()
            post.remove()
        self._require_runtime()
        if energy.dtype is not torch.float64 or not energy.requires_grad or not torch.isfinite(energy):
            raise AnalyticIdentityError("AIMNet2025 complete scalar lost native FP64 accumulation/grad identity.")
        # Native atomic shifts/sum remain upstream FP64; coordinates/NN/AD are
        # still FP32. Capture the actual upstream model intermediates explicitly.
        native = q.detach().cpu().numpy()
        cutoffs = {"radial": 5., "embedded_sr": MODEL_METADATA["coulomb_sr_rc"],
                   "d3_smoothing_on": 12., "d3_smoothing_off": 15.}
        gaps = []
        for positions in (atoms.positions, native.astype(np.float64)):
            for i in range(len(atoms)-1):
                distances = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
                gaps.extend(float(np.min(np.abs(distances - cutoff))) for cutoff in cutoffs.values())
        point = {
            "geometry_sha256": geometry_sha256(atoms),
            "native_geometry_sha256": named_array_sha256({"atomic_numbers": atoms.numbers, "positions_angstrom": native}),
            "native_coordinate_dtype": str(q.dtype),
            "max_coordinate_quantization_angstrom": float(np.max(np.abs(atoms.positions-native.astype(np.float64)))),
            "charge": charge, "multiplicity": mult,
            "atomic_numbers": atoms.numbers.tolist(),
            "neighbor_graph_sha256": canonical_neighbor_graph_sha256(graph),
            "neighbor_packing_sha256": named_array_sha256(graph),
            "neighbor_graph_scheme": NEIGHBOR_GRAPH_SCHEME,
            "active_cutoffs_angstrom": cutoffs,
            "minimum_cutoff_gap_angstrom": min(gaps) if gaps else None,
            "cutoff_guard_margin_angstrom": NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM,
            "intermediate_dtypes": intermediate, "total_energy_dtype": str(energy.dtype),
            "response_settings": {key: json.loads(value)
                                  for key, value in torch_response_settings(self.device)},
        }
        self.last_analytic_point_identity = point
        if curvature and gaps and min(gaps) <= NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM:
            raise AnalyticCapabilityError("AIMNet2025 analytic curvature refused within native/original cutoff guard.")
        return energy, q

    @_native_operation
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        import torch

        properties = self._normalize_properties(properties)
        self._require_runtime()
        # CalcABC's first-order provider filter intentionally rejects Hessians;
        # the inherited get_hessian below owns the separate receipt/H boundary.
        atoms = super().calculate(
            atoms, [prop for prop in properties if prop != "hessian"], system_changes
        )
        # get_hessian is intentionally inherited: it composes gas + OBC once.
        hessian = self.get_hessian(atoms) if "hessian" in properties else None
        energy, q = self._scalar(atoms)
        forces = None
        if "forces" in properties:
            gas_gradient = torch.autograd.grad(energy, q)[0]
            forces = -gas_gradient.detach().cpu().numpy().astype(np.float64)
        self._finalize_results(atoms, energy=energy.item(), forces=forces)
        self._evaluated_correction = self.solvent_correction
        if hessian is not None:
            self.results["hessian"] = hessian

    @_native_operation
    def get_hessian(self, atoms, delta=None):
        # Keep shared complete gas+solvent composition inside the same profile.
        return super().get_hessian(atoms, delta=delta)

    @_native_operation
    def _analytic_hessian(self, atoms):
        import torch

        energy, q = self._scalar(atoms, curvature=True)
        gradient = torch.autograd.grad(energy, q, create_graph=True)[0].reshape(-1)
        # Differentiate native N32 first; scale/promote only after response AD.
        hessian = torch.stack([torch.autograd.grad(value, q, retain_graph=True)[0].reshape(-1)
                               for value in gradient])
        return hessian.detach().cpu().numpy().astype(np.float64) * EV2HARTREE

    @_native_operation
    def get_hvp(self, atoms, n):
        import torch

        correction = self.solvent_correction
        if correction is not None and self.analytic_implicit_derivatives_admitted is not True:
            raise AnalyticCapabilityError("AIMNet2025 composed analytic HVP requires parent receipt admission.")
        direction = np.asarray(n, dtype=np.float64)
        if direction.shape != (3 * len(atoms),) or not np.isfinite(direction).all():
            raise ValueError("AIMNet2025 HVP direction must be a finite flattened (3N,) vector.")
        energy, q = self._scalar(atoms, curvature=True)
        gradient = torch.autograd.grad(energy, q, create_graph=True)[0]
        v = torch.as_tensor(direction, device=self.device, dtype=torch.float32).reshape_as(q)
        if not torch.isfinite(v).all():
            raise ValueError("AIMNet2025 direction is nonfinite after native quantization.")
        hvp = torch.autograd.grad((gradient * v).sum(), q)[0]
        # Promote BEFORE composition; output float64 never means gas D64.
        hvp = hvp.detach().reshape(-1).to(torch.float64) * EV2HARTREE
        forces = -gradient.detach().reshape(-1).to(torch.float64) * EV2HARTREE
        energy = energy.detach() * EV2HARTREE
        provenance = {"composition": "aimnet2025-gas-analytic", "gas_ad_class": "N32",
                      "gas_coordinate_dtype": "float32", "gas_gradient_dtype": "float32",
                      "gas_hvp_dtype": "float32", "output_dtype": "float64",
                      "direction_quantization_max": float(np.max(np.abs(direction-v.detach().cpu().numpy().reshape(-1).astype(np.float64)))),
                      "gas_point": deepcopy(self.last_analytic_point_identity), "solvent_dtype": None}
        if correction is not None:
            directional = getattr(correction, "get_directional_derivatives", None)
            if not callable(directional):
                raise AnalyticCapabilityError("Attached solvent has no admitted directional derivative backend.")
            solvent = directional(atoms, direction)
            sv = np.asarray(solvent.hvp_hartree_per_angstrom2, dtype=np.float64)
            sf = np.asarray(solvent.forces_hartree_per_angstrom, dtype=np.float64)
            se = float(solvent.energy_hartree)
            if sv.shape != direction.shape or sf.shape != (len(atoms), 3) or not np.isfinite(sv).all() or not np.isfinite(sf).all() or not np.isfinite(se):
                raise ValueError("AIMNet2025 solvent directional result has invalid shape/nonfinite values.")
            hvp = hvp + torch.as_tensor(sv, device=self.device, dtype=torch.float64)
            forces = forces + torch.as_tensor(sf.reshape(-1), device=self.device, dtype=torch.float64)
            energy = energy + torch.as_tensor(se, device=self.device, dtype=torch.float64)
            provenance.update(composition="aimnet2025-gas-analytic+torch-obc2-directional",
                              solvent_dtype=self._solvent_compute_dtype(), solvent=dict(solvent.provenance))
        self.last_hvp_provenance = provenance
        return hvp, forces, energy
