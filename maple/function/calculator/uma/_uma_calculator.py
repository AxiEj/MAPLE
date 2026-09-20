from __future__ import annotations

import importlib
import os
import threading
import warnings
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core._config import CACHE_DIR
    from fairchem.core.calculate.ase_calculator import (
        AtomicData,
        FAIRChemCalculator,
        UMATask,
    )
    from fairchem.core.common.distutils import CURRENT_DEVICE_TYPE_STR
    from fairchem.core.units.mlip_unit import load_predict_unit
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError("fairchem-core is not installed. Please install it first.")

from maple.function.device import resolve_torch_device

from .._batch_types import BatchResult
from .._batch_utils import default_calculate_many
from ..calculator_base import (
    EV2HARTREE,
    IMPLICIT_SOLVENT_FORCE_ERROR,
    calculator_execution,
    init_implicit_solvent,
    numerical_hessian_from_atoms,
    register_calculator,
    reject_implicit_solvent_derivatives,
)

UMA_DEFAULT_SIZE = "uma-s-1p1"
UMA_MODELS_MAP = {
    "uma": UMA_DEFAULT_SIZE,
    "uma-s-1p1": "uma-s-1p1",
    "uma-s-1p2": "uma-s-1p2",
    "uma-m-1p1": "uma-m-1p1",
}

UMA_FALLBACK_HF_MODELS = {"uma-s-1p1", "uma-s-1p2", "uma-m-1p1"}
SUPPORTED_UMA_TASKS = {"omol", "omat", "oc20", "odac", "omc", "oc22", "oc25"}
PERIODIC_UMA_TASKS = SUPPORTED_UMA_TASKS - {"omol"}
SUPPORTED_UMA_INFERENCE = {"default", "turbo"}

# GPU default and CPU fallback for FAIR Chemistry's inference path. "default"
# is the general-purpose backend; "turbo" speeds up fixed-composition workloads
# (NEB / TS / freq) on CUDA but routes through Triton kernels unavailable on
# CPU. Users override via `model=uma(...,inference=turbo)`; CPU coerces to
# default regardless.
UMA_INFERENCE_SETTINGS = "default"
UMA_CPU_INFERENCE_SETTINGS = "default"

_FAIRCHEM_DEVICE_BINDING_LOCK = threading.RLock()


@register_calculator
class UMACalculator(FAIRChemCalculator):
    """UMA calculator with MAPLE-specific unit conversion and Hessian support.

    Does NOT inherit CalcABC — UMA already extends third-party FAIRChemCalculator.
    Satisfies the MAPLE calculator protocol via attribute presence (class
    capability attrs + calculate + get_hessian + get_hvp where required).

    Explicit `task=` is respected. When `task` is omitted, MAPLE only infers
    `omol` for non-periodic systems; periodic UMA requires an explicit FAIR-Chem
    task because `pbc -> omat` is too broad for production use.
    """

    MODEL_NAMES = ("uma",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = True
    SUPPORTS_IMPLICIT_SOLVATION = True
    supports_batch_energy_forces = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    solvent_correction: Any
    OPTION_KEYS = (
        'task',
        'size',
        'checkpoint_path',
        'inference',
        'overrides',
    )
    MODEL_PATH_OPTION = 'checkpoint_path'

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {
            'task': options.get('task'),
            'size': options.get('size'),
            'checkpoint_path': options.get('checkpoint_path') or resolved_model_path,
            'inference_settings': options.get('inference'),
            'overrides': options.get('overrides'),
        }

    @staticmethod
    def _normalize_device(device) -> Any:
        # Preserve an explicit CUDA ordinal so multi-GPU callers reach the
        # selected device. UMA's FAIR-Chem backend currently supports CPU and
        # CUDA only, so explicit unsupported devices fail rather than silently
        # changing the requested execution path.
        device_name = str(device).lower()
        if device_name != "cpu" and not device_name.startswith("cuda"):
            raise ValueError(
                f"Unsupported UMA device {device!r}; expected cpu or an available cuda[:N]."
            )
        try:
            resolved = resolve_torch_device(device_name)
        except ValueError as exc:
            raise ValueError(f"UMA device error: {exc}") from exc
        return str(resolved)

    @staticmethod
    def _build_predictor(
        checkpoint: str,
        overrides,
        device: str,
        checkpoint_path=None,
        inference_settings=None,
    ):
        device = UMACalculator._normalize_device(device)
        # Turbo selects FAIR Chemistry's fast GPU execution path; CPU uses the
        # general-purpose backend to avoid Triton GPU kernels on CPU tensors.
        if device == "cpu":
            if inference_settings == "turbo":
                warnings.warn(
                    "UMA 'turbo' inference requires CUDA; falling back to 'default' on CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            inference_settings = UMA_CPU_INFERENCE_SETTINGS
        elif inference_settings is None:
            inference_settings = UMA_INFERENCE_SETTINGS

        def construct(factory):
            requested_device = torch.device(device)
            backend_device = requested_device.type

            with _FAIRCHEM_DEVICE_BINDING_LOCK:
                had_previous_device_type = CURRENT_DEVICE_TYPE_STR in os.environ
                previous_device_type = os.environ.get(CURRENT_DEVICE_TYPE_STR)
                try:
                    os.environ[CURRENT_DEVICE_TYPE_STR] = backend_device
                    if requested_device.type == "cuda":
                        context = torch.cuda.device(requested_device.index)
                    else:
                        context = nullcontext()
                    with context:
                        predictor = factory(backend_device)
                        actual_device = torch.device(str(predictor.device))
                finally:
                    if had_previous_device_type and previous_device_type is not None:
                        os.environ[CURRENT_DEVICE_TYPE_STR] = previous_device_type
                    else:
                        os.environ.pop(CURRENT_DEVICE_TYPE_STR, None)

            if actual_device != requested_device:
                raise RuntimeError(
                    "FAIR-Chem did not honor the requested UMA device: "
                    f"requested {requested_device}, constructed {actual_device}."
                )
            return predictor

        if checkpoint_path and os.path.isfile(checkpoint_path):
            compat_path = UMACalculator._prepare_compat_checkpoint(checkpoint, checkpoint_path)
            return construct(
                lambda backend_device: load_predict_unit(
                    compat_path,
                    inference_settings=inference_settings,
                    overrides=overrides,
                    device=backend_device,
                )
            )

        if checkpoint in pretrained_mlip.available_models:
            return construct(
                lambda backend_device: pretrained_mlip.get_predict_unit(
                    checkpoint,
                    inference_settings=inference_settings,
                    overrides=overrides,
                    device=backend_device,
                )
            )

        if os.path.isfile(checkpoint):
            compat_path = UMACalculator._prepare_compat_checkpoint(Path(checkpoint).stem, checkpoint)
            return construct(
                lambda backend_device: load_predict_unit(
                    compat_path,
                    inference_settings=inference_settings,
                    overrides=overrides,
                    device=backend_device,
                )
            )

        if checkpoint in UMA_FALLBACK_HF_MODELS:
            checkpoint_file = hf_hub_download(
                repo_id="facebook/UMA",
                subfolder="checkpoints",
                filename=f"{checkpoint}.pt",
                cache_dir=CACHE_DIR,
            )
            compat_path = UMACalculator._prepare_compat_checkpoint(checkpoint, checkpoint_file)
            atom_refs = OmegaConf.load(
                hf_hub_download(
                    repo_id="facebook/UMA",
                    subfolder="references",
                    filename="iso_atom_elem_refs.yaml",
                    cache_dir=CACHE_DIR,
                )
            )
            form_elem_refs = OmegaConf.load(
                hf_hub_download(
                    repo_id="facebook/UMA",
                    subfolder="references",
                    filename="form_elem_refs.yaml",
                    cache_dir=CACHE_DIR,
                )
            )["refs"]
            return construct(
                lambda backend_device: load_predict_unit(
                    compat_path,
                    inference_settings=inference_settings,
                    overrides=overrides,
                    device=backend_device,
                    atom_refs=atom_refs,
                    form_elem_refs=form_elem_refs,
                )
            )

        raise ValueError(
            f"Unsupported UMA checkpoint '{checkpoint}'. "
            f"Built-in models: {sorted(pretrained_mlip.available_models)}"
        )

    @staticmethod
    def _prepare_compat_checkpoint(checkpoint_name: str, checkpoint_path: str) -> str:
        compat_dir = Path(CACHE_DIR) / "maple_compat"
        compat_dir.mkdir(parents=True, exist_ok=True)
        compat_path = compat_dir / f"{checkpoint_name}.pt"

        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = raw_checkpoint.model_config
        model_config.pop("model_id", None)
        model_config.pop("supports_single_atoms", None)

        backbone = model_config.get("backbone", {})
        backbone.pop("charge_balanced_channels", None)
        backbone.pop("composition_dropout", None)
        dataset_mapping = backbone.pop("dataset_mapping", None)
        if dataset_mapping is not None and "dataset_list" not in backbone:
            backbone["dataset_list"] = list(dataset_mapping)

        for head_config in model_config.get("heads", {}).values():
            head_mapping = head_config.pop("dataset_mapping", None)
            if head_mapping is not None and "dataset_names" not in head_config:
                head_config["dataset_names"] = list(head_mapping)

        torch.save(raw_checkpoint, compat_path)
        return str(compat_path)

    def __init__(
        self,
        device,
        model: str = "uma",
        overrides=None,
        implicit: Literal["gbsa", "none"] = "none",
        solvent: str = "none",
        task=None,
        size=None,
        checkpoint_path=None,
        inference_settings=None,
    ):
        if size is not None:
            size = str(size).lower()
        checkpoint = UMA_MODELS_MAP.get(size, size) if size else UMA_MODELS_MAP.get(model, UMA_DEFAULT_SIZE)

        if task is not None:
            task = str(task).lower()
            if task not in SUPPORTED_UMA_TASKS:
                raise ValueError(f"Unsupported UMA task: '{task}'. Supported: {sorted(SUPPORTED_UMA_TASKS)}")

        if inference_settings is not None:
            inference_settings = str(inference_settings).lower()
            if inference_settings not in SUPPORTED_UMA_INFERENCE:
                raise ValueError(
                    f"Unsupported UMA inference mode: '{inference_settings}'. "
                    f"Supported: {sorted(SUPPORTED_UMA_INFERENCE)}"
                )

        device = self._normalize_device(device)

        if not importlib.util.find_spec("fairchem"):
            raise ImportError("fairchem-core is not installed. Please install it first.")

        predictor = self._build_predictor(
            checkpoint,
            overrides,
            device,
            checkpoint_path=checkpoint_path,
            inference_settings=inference_settings,
        )
        super().__init__(predict_unit=predictor, task_name=task or "omol")

        self.device = torch.device(device)
        self._predictor_unit = predictor
        self._auto_task = task is None
        self.hessian = "numerical"

        # The shared helper rejects the removed legacy GBSA/QEq constructor and
        # leaves the audited Route 1 correction to SetCalculator.
        init_implicit_solvent(self, implicit, solvent, self.device)

    def _set_task_from_atoms(self, atoms: Atoms) -> None:
        if not self._auto_task:
            return

        if any(atoms.pbc):
            raise ValueError(
                "UMA periodic calculations require an explicit FAIR-Chem task "
                "(for example task=omat, oc20, oc22, oc25, omc, or odac). "
                "MAPLE no longer silently maps every periodic system to task='omat'."
            )

        task = "omol"
        if task == self.task_name:
            return

        self._task = UMATask(task)
        self._task_name = task
        self.implemented_properties = [
            task_obj.property for task_obj in self.predictor.dataset_to_tasks[self.task_name]
        ]
        if "energy" in self.implemented_properties:
            self.implemented_properties.append("free_energy")

        if self._predictor_unit.inference_settings.external_graph_gen:
            r_edges, max_neigh = True, 300
        else:
            r_edges, max_neigh = False, None

        self.a2g = partial(
            AtomicData.from_ase,
            task_name=task,
            r_edges=r_edges,
            r_data_keys=["spin", "charge"],
            max_neigh=max_neigh,
            radius=6.0,
            target_dtype=self._predictor_unit.inference_settings.base_precision_dtype,
        )

    def _validate_task_atoms_compatibility(self, atoms: Atoms) -> None:
        if not any(atoms.pbc):
            return

        if self.task_name == "omol":
            raise ValueError(
                "UMA task='omol' is molecular and must not be used with periodic atoms. "
                "Use an explicit periodic/domain task such as task=omat, oc20, oc22, "
                "oc25, omc, or odac after confirming the system domain."
            )

        if self.task_name not in PERIODIC_UMA_TASKS:
            raise ValueError(
                f"UMA task='{self.task_name}' is not registered as a MAPLE periodic task; "
                f"supported periodic tasks: {sorted(PERIODIC_UMA_TASKS)}."
            )

    def _validate_charge_spin_task_compatibility(self, atoms: Atoms) -> tuple[int, int]:
        charge = self._integer_info(atoms, "charge", 0)
        mult = self._integer_info(atoms, "mult", 1)
        has_charge = charge != 0
        has_open_shell = mult != 1

        if self.task_name == "omol":
            if has_charge or has_open_shell:
                message = (
                    "UMA omol charged/open-shell inputs are passed through to FAIR-Chem, "
                    "but MAPLE has not yet accepted golden numerical tolerances for these "
                    "states; compare against FAIR-Chem/reference calculations before "
                    "production use."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            return charge, mult

        if has_charge or has_open_shell:
            raise ValueError(
                f"UMA task='{self.task_name}' does not use charge/spin according to "
                "FAIR-Chem's current calculator contract. Remove atoms.info['charge']/"
                "atoms.info['mult'] or use task='omol' for molecular charged/open-shell "
                "calculations."
            )

        return charge, mult

    @staticmethod
    def _integer_info(atoms: Atoms, key: str, default: int) -> int:
        value = atoms.info.get(key, default)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"UMA requires integer atoms.info['{key}']; got {value!r}.") from exc
        if not numeric_value.is_integer():
            raise ValueError(f"UMA requires integer atoms.info['{key}']; got {value!r}.")
        return int(numeric_value)

    def get_hessian(self, atoms: Atoms, delta: float | None = None) -> np.ndarray:
        """Numerical Hessian of the complete gas-plus-solvent force."""
        self.last_numerical_hessian_diagnostics = None
        correction = getattr(self, "solvent_correction", None)
        if correction is not None and "forces" not in set(
            getattr(correction, "supported_properties", {"energy"})
        ):
            raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)

        prepare = getattr(self, "prepare_numerical_derivatives", None)
        backend_hint = prepare() if callable(prepare) else None
        resolve_outer_step = getattr(
            correction, "resolve_outer_curvature_step", None
        )
        if callable(resolve_outer_step):
            numerical_solvent = getattr(correction, "force_mode", None) == "numerical"
            delta = resolve_outer_step(
                task_delta=None if numerical_solvent else delta,
                backend_hint=backend_hint,
                task=getattr(correction, "task", "freq"),
            )
            preflight = getattr(
                correction, "preflight_cartesian_outer_displacements", None
            )
            if callable(preflight):
                preflight(atoms, delta)
        elif delta is None:
            delta = 0.002 if backend_hint is None else backend_hint

        reserve_operation = getattr(
            getattr(correction, "provider", None),
            "reserve_force_operation",
            None,
        )
        operation_context = (
            reserve_operation(
                atom_count=len(atoms),
                force_call_count=6 * len(atoms),
                operation=(
                    "frequency"
                    if getattr(correction, "task", "freq") == "freq"
                    else "prfo-hessian"
                ),
            )
            if callable(reserve_operation)
            else nullcontext()
        )
        with operation_context:
            hessian, diagnostics = numerical_hessian_from_atoms(
                self, atoms, delta, return_diagnostics=True
            )
        if not np.isfinite(hessian).all():
            raise ValueError("Numerical Hessian contains non-finite values.")
        self.last_numerical_hessian_diagnostics = dict(diagnostics)
        if correction is not None and hasattr(
            correction, "last_outer_curvature_resolution"
        ):
            self.last_numerical_hessian_diagnostics["outer_step_resolution"] = dict(
                correction.last_outer_curvature_resolution
            )
        return hessian

    def calculate(self, atoms, properties=None, system_changes=None):
        properties = reject_implicit_solvent_derivatives(self, properties)
        system_changes = all_changes if system_changes is None else system_changes

        if atoms is None:
            atoms = getattr(self, "atoms", None)
        if atoms is None:
            raise ValueError("UMACalculator.calculate requires an Atoms object.")

        requested = {str(prop).lower() for prop in properties}
        if requested & {"stress", "stresses", "virial", "virials"}:
            raise NotImplementedError(
                "UMA stress/virial output is not unit-converted by MAPLE yet; "
                "request energy/forces only until stress units are validated."
            )

        self._set_task_from_atoms(atoms)
        self._validate_task_atoms_compatibility(atoms)
        charge, mult = self._validate_charge_spin_task_compatibility(atoms)

        calc_atoms = atoms.copy()
        calc_atoms.info["spin"] = mult
        calc_atoms.info["charge"] = charge

        super().calculate(calc_atoms, properties, system_changes)
        self.execution_provenance = calculator_execution(self)

        # eV → Hartree: UMA's MODEL_ENERGY_UNIT is 'eV'; equivalent to the
        # _finalize_results unit step but inlined because UMA does not inherit
        # CalcABC.
        if "energy" in self.results:
            self.results["energy"] *= EV2HARTREE
        if "free_energy" in self.results:
            self.results["free_energy"] *= EV2HARTREE
        if "forces" in self.results:
            self.results["forces"] *= EV2HARTREE

        if self.solvent_correction is not None:
            if hasattr(self.solvent_correction, "evaluate"):
                solvent_result = self.solvent_correction.evaluate(
                    calc_atoms, need_forces="forces" in self.results
                )
                if "energy" in self.results:
                    self.results["energy"] += float(solvent_result.energy_hartree)
                if "free_energy" in self.results:
                    self.results["free_energy"] += float(solvent_result.energy_hartree)
                if "forces" in self.results:
                    if solvent_result.forces_hartree_per_angstrom is None:
                        raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
                    self.results["forces"] += np.asarray(
                        solvent_result.forces_hartree_per_angstrom
                    )
                self.results["solvation"] = {
                    "energy_hartree": float(solvent_result.energy_hartree),
                    "components_hartree": dict(solvent_result.components_hartree),
                    "provenance": dict(solvent_result.provenance),
                }
            else:
                raise RuntimeError(
                    "Unstructured legacy implicit-solvent corrections are no "
                    "longer supported; attach an ImplicitSolvationCorrection."
                )

        return self.results

    def calculate_many(
        self,
        atoms_list,
        properties=("energy", "forces"),
    ) -> BatchResult:
        """Use the common result-driven fallback for the third-party wrapper."""
        return default_calculate_many(self, atoms_list, properties)
