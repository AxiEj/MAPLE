from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
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
from ..analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticOperationResult,
    CheckpointFileIdentity,
    NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM,
    NEIGHBOR_CUTOFF_GUARD_POLICY,
    ScalarCellIdentity,
    ScalarPointIdentity,
    freeze_identity_options,
    geometry_sha256,
    named_array_sha256,
)
from ..calculator_base import (
    EV2HARTREE,
    IMPLICIT_SOLVENT_FORCE_ERROR,
    calculator_execution,
    init_implicit_solvent,
    numerical_hessian_from_atoms,
    register_calculator,
    reject_implicit_solvent_derivatives,
)
from ..solvent_composition import (
    compose_analytic_hessian,
    compose_direct_hvp,
    finalize_energy_forces,
)
from ._analytic_adapter import predict_direct_hvp, prepare_compat_checkpoint

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
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = True
    SUPPORTS_IMPLICIT_SOLVATION = True
    supports_batch_energy_forces = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    solvent_correction: Any
    OPTION_KEYS = (
        "task",
        "size",
        "checkpoint_path",
        "inference",
        "overrides",
    )
    MODEL_PATH_OPTION = "checkpoint_path"

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {
            "task": options.get("task"),
            "size": options.get("size"),
            "checkpoint_path": options.get("checkpoint_path") or resolved_model_path,
            "inference_settings": options.get("inference"),
            "overrides": options.get("overrides"),
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
    def _stamp_predictor_lineage(predictor, *, original_path: str, compat_path: str):
        original = str(Path(original_path).expanduser().resolve())
        compat = str(Path(compat_path).expanduser().resolve())
        receipt_path = compat + ".receipt.json"
        if not os.path.isfile(receipt_path):
            predictor._maple_original_checkpoint_path = original
            predictor._maple_compat_checkpoint_path = compat
            predictor._maple_compat_receipt_path = None
            predictor._maple_compat_receipt = None
            return predictor
        with open(receipt_path, encoding="utf-8") as handle:
            receipt = json.load(handle)
        if receipt.get("source", {}).get("path") != original:
            raise RuntimeError(
                "UMA compatibility receipt does not bind the source checkpoint."
            )
        if receipt.get("output", {}).get("path") != compat:
            raise RuntimeError(
                "UMA compatibility receipt does not bind the output checkpoint."
            )
        predictor._maple_original_checkpoint_path = original
        predictor._maple_compat_checkpoint_path = compat
        predictor._maple_compat_receipt_path = receipt_path
        predictor._maple_compat_receipt = receipt
        return predictor

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
            compat_path = UMACalculator._prepare_compat_checkpoint(
                checkpoint, checkpoint_path
            )
            predictor = construct(
                lambda backend_device: load_predict_unit(
                    compat_path,
                    inference_settings=inference_settings,
                    overrides=overrides,
                    device=backend_device,
                )
            )
            return UMACalculator._stamp_predictor_lineage(
                predictor,
                original_path=checkpoint_path,
                compat_path=compat_path,
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
            compat_path = UMACalculator._prepare_compat_checkpoint(
                Path(checkpoint).stem, checkpoint
            )
            predictor = construct(
                lambda backend_device: load_predict_unit(
                    compat_path,
                    inference_settings=inference_settings,
                    overrides=overrides,
                    device=backend_device,
                )
            )
            return UMACalculator._stamp_predictor_lineage(
                predictor,
                original_path=checkpoint,
                compat_path=compat_path,
            )

        if checkpoint in UMA_FALLBACK_HF_MODELS:
            checkpoint_file = hf_hub_download(
                repo_id="facebook/UMA",
                subfolder="checkpoints",
                filename=f"{checkpoint}.pt",
                cache_dir=CACHE_DIR,
            )
            compat_path = UMACalculator._prepare_compat_checkpoint(
                checkpoint, checkpoint_file
            )
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
            predictor = construct(
                lambda backend_device: load_predict_unit(
                    compat_path,
                    inference_settings=inference_settings,
                    overrides=overrides,
                    device=backend_device,
                    atom_refs=atom_refs,
                    form_elem_refs=form_elem_refs,
                )
            )
            return UMACalculator._stamp_predictor_lineage(
                predictor,
                original_path=checkpoint_file,
                compat_path=compat_path,
            )

        raise ValueError(
            f"Unsupported UMA checkpoint '{checkpoint}'. "
            f"Built-in models: {sorted(pretrained_mlip.available_models)}"
        )

    @staticmethod
    def _prepare_compat_checkpoint(checkpoint_name: str, checkpoint_path: str) -> str:
        return prepare_compat_checkpoint(
            checkpoint_name,
            checkpoint_path,
            cache_dir=CACHE_DIR,
        )

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
        checkpoint = (
            UMA_MODELS_MAP.get(size, size)
            if size
            else UMA_MODELS_MAP.get(model, UMA_DEFAULT_SIZE)
        )

        if task is not None:
            task = str(task).lower()
            if task not in SUPPORTED_UMA_TASKS:
                raise ValueError(
                    f"Unsupported UMA task: '{task}'. Supported: {sorted(SUPPORTED_UMA_TASKS)}"
                )

        if inference_settings is not None:
            inference_settings = str(inference_settings).lower()
            if inference_settings not in SUPPORTED_UMA_INFERENCE:
                raise ValueError(
                    f"Unsupported UMA inference mode: '{inference_settings}'. "
                    f"Supported: {sorted(SUPPORTED_UMA_INFERENCE)}"
                )

        device = self._normalize_device(device)

        if not importlib.util.find_spec("fairchem"):
            raise ImportError(
                "fairchem-core is not installed. Please install it first."
            )

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
        self._checkpoint_name = checkpoint
        self._checkpoint_path_argument = checkpoint_path
        self._predictor_overrides = copy.deepcopy(overrides)
        self._inference_settings_name = inference_settings or UMA_INFERENCE_SETTINGS
        self._hessian_predictor_unit = None
        self._auto_task = task is None
        self.hessian = "numerical"
        self.analytic_derivative_admission = None
        self.analytic_composition_admission = None
        self.analytic_implicit_derivatives_admitted = False
        self.last_analytic_hessian_provenance = None
        self.last_hvp_provenance = None

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
            task_obj.property
            for task_obj in self.predictor.dataset_to_tasks[self.task_name]
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
            raise ValueError(
                f"UMA requires integer atoms.info['{key}']; got {value!r}."
            ) from exc
        if not numeric_value.is_integer():
            raise ValueError(
                f"UMA requires integer atoms.info['{key}']; got {value!r}."
            )
        return int(numeric_value)

    def _prepared_calc_atoms(self, atoms: Atoms) -> Atoms:
        self._set_task_from_atoms(atoms)
        self._validate_task_atoms_compatibility(atoms)
        charge, mult = self._validate_charge_spin_task_compatibility(atoms)
        calc_atoms = atoms.copy()
        calc_atoms.info["spin"] = mult
        calc_atoms.info["charge"] = charge
        return calc_atoms

    @staticmethod
    def _settings_record(
        settings, *, include_derivative_requests: bool
    ) -> dict[str, Any]:
        record = asdict(settings)
        record["base_precision_dtype"] = str(settings.base_precision_dtype)
        for key in (
            "predict_untrained_forces",
            "predict_untrained_stress",
            "predict_untrained_hessian",
        ):
            record[key] = sorted(record[key])
        if not include_derivative_requests:
            for key in (
                "predict_untrained_forces",
                "predict_untrained_stress",
                "predict_untrained_hessian",
                "hessian_vmap",
            ):
                record.pop(key, None)
        return record

    def _a2g_for_predictor(self, predictor):
        settings = predictor.inference_settings
        if settings.external_graph_gen:
            r_edges, max_neigh = True, 300
        else:
            r_edges, max_neigh = False, None
        return partial(
            AtomicData.from_ase,
            task_name=self.task_name,
            r_edges=r_edges,
            r_data_keys=["spin", "charge"],
            max_neigh=max_neigh,
            radius=6.0,
            target_dtype=settings.base_precision_dtype,
        )

    def _predict_properties(self, predictor, calc_atoms: Atoms):
        predictor.validate_atoms_data(calc_atoms, self.task_name)
        return predictor.predict(self._a2g_for_predictor(predictor)(calc_atoms))

    def _lineage_receipt(self) -> dict[str, Any]:
        receipt = getattr(self._predictor_unit, "_maple_compat_receipt", None)
        receipt_path = getattr(self._predictor_unit, "_maple_compat_receipt_path", None)
        if not isinstance(receipt, dict) or not receipt_path:
            raise AnalyticCapabilityError(
                "UMA analytic derivatives require a content-addressed original-to-compat "
                "checkpoint receipt."
            )
        if receipt.get("tensor_payload", {}).get("exactly_equal") is not True:
            raise AnalyticCapabilityError(
                "UMA analytic derivatives require exact checkpoint tensor equality."
            )
        return receipt

    def _analytic_cell_identity(self) -> ScalarCellIdentity:
        receipt = self._lineage_receipt()
        compat_path = getattr(self._predictor_unit, "_maple_compat_checkpoint_path")
        checkpoint = CheckpointFileIdentity.bind(compat_path)
        if checkpoint.sha256 != receipt["output"]["sha256"]:
            raise AnalyticCapabilityError(
                "UMA compatibility checkpoint receipt is stale."
            )
        receipt_path = Path(self._predictor_unit._maple_compat_receipt_path)
        receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        predictor = self._predictor_unit
        settings = predictor.inference_settings
        if settings.tf32:
            raise AnalyticCapabilityError("UMA analytic derivatives forbid TF32.")
        if settings.base_precision_dtype not in {torch.float32, torch.float64}:
            raise AnalyticCapabilityError(
                f"Unsupported UMA analytic dtype: {settings.base_precision_dtype}."
            )
        model = predictor.model.module
        head_names = tuple(sorted(model.output_heads.keys()))
        return ScalarCellIdentity(
            component="gas",
            implementation="fairchem-uma-analytic-v1",
            model=self._checkpoint_name,
            checkpoint_path=checkpoint.path,
            checkpoint_sha256=checkpoint.sha256,
            device=str(self.device),
            effective_dtype=str(settings.base_precision_dtype),
            d4=None,
            source_energy_unit="eV",
            cutoff_guard_policy=NEIGHBOR_CUTOFF_GUARD_POLICY,
            cutoff_guard_margin_angstrom=NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM,
            options=freeze_identity_options(
                {
                    "task": self.task_name,
                    "fairchem_version": importlib.metadata.version("fairchem-core"),
                    "original_checkpoint_path": receipt["source"]["path"],
                    "original_checkpoint_sha256": receipt["source"]["sha256"],
                    "compat_receipt_path": str(receipt_path.resolve()),
                    "compat_receipt_sha256": receipt_sha,
                    "transform_version": receipt["transform_version"],
                    "head_names": head_names,
                    "inference": self._settings_record(
                        settings, include_derivative_requests=False
                    ),
                }
            ),
        )

    def _analytic_point_identity(self, atoms: Atoms) -> ScalarPointIdentity:
        if any(atoms.pbc):
            raise AnalyticCapabilityError(
                "UMA periodic analytic derivatives require a separately qualified graph identity."
            )
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        charge = self._integer_info(atoms, "charge", 0)
        mult = self._integer_info(atoms, "mult", 1)
        cutoff = float(
            getattr(self._predictor_unit.model.module.backbone, "cutoff", 6.0)
        )
        if len(atoms) < 2:
            cutoff_gap = float("inf")
        else:
            distances = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=-1
            )
            upper = distances[np.triu_indices(len(atoms), k=1)]
            cutoff_gap = float(np.min(np.abs(upper - cutoff)))
        graph_input_sha = named_array_sha256(
            {
                "atomic_numbers": numbers,
                "positions_angstrom": positions,
                "cell_angstrom": np.asarray(atoms.cell.array, dtype=np.float64),
                "pbc": np.asarray(atoms.pbc, dtype=np.uint8),
                "cutoff_angstrom": np.asarray([cutoff], dtype=np.float64),
            }
        )
        topology_sha = named_array_sha256({"atomic_numbers": numbers})
        return ScalarPointIdentity(
            cell=self._analytic_cell_identity(),
            geometry_sha256=geometry_sha256(atoms),
            atomic_numbers=tuple(int(value) for value in numbers),
            charge=float(charge),
            multiplicity=mult,
            atom_topology_sha256=topology_sha,
            neighbor_graph_sha256=graph_input_sha,
            active_cutoffs_angstrom=(("uma_neighbor", cutoff),),
            minimum_cutoff_gap_angstrom=cutoff_gap,
            response_settings=freeze_identity_options(
                {
                    "task": self.task_name,
                    "charge": charge,
                    "multiplicity": mult,
                    "inference": self._settings_record(
                        self._predictor_unit.inference_settings,
                        include_derivative_requests=False,
                    ),
                }
            ),
        )

    def prepare_analytic_derivatives(self) -> None:
        """Validate operation capability without self-admitting a scientific cell."""
        self._analytic_cell_identity()

    def _hessian_predictor(self):
        if self._hessian_predictor_unit is not None:
            return self._hessian_predictor_unit
        original_path = getattr(
            self._predictor_unit, "_maple_original_checkpoint_path", None
        )
        if not original_path:
            raise AnalyticCapabilityError(
                "UMA full Hessian requires an exact locally bound original checkpoint."
            )
        settings = copy.deepcopy(self._predictor_unit.inference_settings)
        if settings.tf32:
            raise AnalyticCapabilityError("UMA analytic Hessian forbids TF32.")
        settings.predict_untrained_hessian = set(settings.predict_untrained_hessian) | {
            self.task_name
        }
        predictor = self._build_predictor(
            self._checkpoint_name,
            copy.deepcopy(self._predictor_overrides),
            str(self.device),
            checkpoint_path=original_path,
            inference_settings=settings,
        )
        public_scalar = self._settings_record(
            self._predictor_unit.inference_settings,
            include_derivative_requests=False,
        )
        derivative_scalar = self._settings_record(
            predictor.inference_settings,
            include_derivative_requests=False,
        )
        if derivative_scalar != public_scalar:
            raise AnalyticCapabilityError(
                "UMA Hessian predictor changed scalar-defining inference settings."
            )
        self._hessian_predictor_unit = predictor
        return predictor

    def _analytic_energy_forces_operation(
        self, atoms: Atoms, *, need_forces: bool
    ) -> AnalyticOperationResult:
        calc_atoms = self._prepared_calc_atoms(atoms)
        pred = self._predict_properties(self._predictor_unit, calc_atoms)
        energy = float(pred["energy"].detach().cpu().reshape(-1)[0]) * EV2HARTREE
        forces = None
        if need_forces:
            forces = pred["forces"].detach().cpu().numpy() * EV2HARTREE
        return AnalyticOperationResult.energy_forces(
            identity=self._analytic_point_identity(atoms),
            energy_hartree=energy,
            forces_hartree_per_angstrom=forces,
        )

    def _analytic_hessian_operation(self, atoms: Atoms) -> AnalyticOperationResult:
        calc_atoms = self._prepared_calc_atoms(atoms)
        public = self._predict_properties(self._predictor_unit, calc_atoms)
        predictor = self._hessian_predictor()
        derivative = self._predict_properties(predictor, calc_atoms)
        public_energy = (
            float(public["energy"].detach().cpu().reshape(-1)[0]) * EV2HARTREE
        )
        derivative_energy = (
            float(derivative["energy"].detach().cpu().reshape(-1)[0]) * EV2HARTREE
        )
        public_forces = public["forces"].detach().cpu().numpy() * EV2HARTREE
        derivative_forces = derivative["forces"].detach().cpu().numpy() * EV2HARTREE
        dtype = self._predictor_unit.inference_settings.base_precision_dtype
        energy_tol = (
            2.0e-10 + 1.0e-9 * abs(public_energy) if dtype == torch.float64 else 2.0e-5
        )
        force_tol = 2.0e-7 if dtype == torch.float64 else 2.0e-6
        energy_error = abs(derivative_energy - public_energy)
        force_error = float(np.max(np.abs(derivative_forces - public_forces)))
        if energy_error > energy_tol or force_error > force_tol:
            raise AnalyticCapabilityError(
                "UMA Hessian profile does not close against the public E/F scalar: "
                f"energy_error={energy_error:.6g}, force_error={force_error:.6g}."
            )
        hessian = (
            derivative["hessian"]
            .detach()
            .cpu()
            .numpy()
            .reshape(3 * len(atoms), 3 * len(atoms))
            * EV2HARTREE
        )
        self._last_uma_hessian_operation_provenance = {
            "predict_untrained_hessian": [self.task_name],
            "public_center_energy_error_hartree": energy_error,
            "public_center_force_max_error_hartree_per_angstrom": force_error,
            "source_unit": "eV/angstrom^2",
            "target_unit": "hartree/angstrom^2",
        }
        return AnalyticOperationResult.hessian(
            identity=self._analytic_point_identity(atoms),
            hessian_hartree_per_angstrom2=hessian,
        )

    def _analytic_hvp_operation(
        self, atoms: Atoms, n: np.ndarray
    ) -> AnalyticOperationResult:
        calc_atoms = self._prepared_calc_atoms(atoms)
        predictor = self._predictor_unit
        if predictor.inference_settings.predict_untrained_hessian:
            raise AnalyticCapabilityError(
                "UMA direct HVP requires the public Hessian-disabled predictor."
            )
        direction = np.asarray(n, dtype=np.float64).reshape(3 * len(atoms))
        data = self._a2g_for_predictor(predictor)(calc_atoms)
        result = predict_direct_hvp(
            predictor,
            data,
            task_name=self.task_name,
            direction=direction.reshape(len(atoms), 3),
        )
        energy = float(result["energy"].detach().cpu().reshape(-1)[0]) * EV2HARTREE
        forces = result["forces"].detach().cpu().numpy() * EV2HARTREE
        hvp = result["hvp"].detach().cpu().numpy().reshape(-1) * EV2HARTREE
        self._last_uma_hvp_operation_provenance = {
            "predict_untrained_hessian": [],
            "direct_autograd": True,
            "dense_hessian": False,
            "position_dtype": result["position_dtype"],
            "source_unit": "eV/angstrom^2",
            "target_unit": "hartree/angstrom^2",
        }
        return AnalyticOperationResult.hvp(
            identity=self._analytic_point_identity(atoms),
            direction=direction,
            hvp_hartree_per_angstrom2=hvp,
            forces_hartree_per_angstrom=forces,
            energy_hartree=energy,
        )

    def get_hessian(self, atoms: Atoms, delta: float | None = None) -> np.ndarray:
        """Return the selected analytic or explicit numerical UMA Hessian."""
        mode = getattr(self, "hessian", "numerical")
        if mode == "analytic":
            gas_operation = self._analytic_hessian_operation(atoms)
            hessian, provenance = compose_analytic_hessian(
                atoms,
                gas_operation=gas_operation,
                correction=getattr(self, "solvent_correction", None),
                composition_admission=self.analytic_composition_admission,
                gas_admission=self.analytic_derivative_admission,
                operation_provenance=self._last_uma_hessian_operation_provenance,
            )
            self.last_analytic_hessian_provenance = provenance
            self.last_numerical_hessian_diagnostics = None
            return hessian
        if mode != "numerical":
            raise ValueError(f"Unknown UMA Hessian mode: {mode!r}")

        correction = getattr(self, "solvent_correction", None)
        if correction is not None and "forces" not in set(
            getattr(correction, "supported_properties", {"energy"})
        ):
            raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)

        prepare = getattr(self, "prepare_numerical_derivatives", None)
        backend_hint = prepare() if callable(prepare) else None
        resolve_outer_step = getattr(correction, "resolve_outer_curvature_step", None)
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
        committed_diagnostics = dict(diagnostics)
        if correction is not None and hasattr(
            correction, "last_outer_curvature_resolution"
        ):
            committed_diagnostics["outer_step_resolution"] = dict(
                correction.last_outer_curvature_resolution
            )
        self.last_numerical_hessian_diagnostics = committed_diagnostics
        self.last_analytic_hessian_provenance = None
        return hessian

    def get_hvp(self, atoms: Atoms, n: np.ndarray):
        """Return a same-center direct ``(Hn, forces, energy)`` tuple."""
        gas_operation = self._analytic_hvp_operation(atoms, n)
        result, provenance = compose_direct_hvp(
            atoms,
            n,
            gas_operation=gas_operation,
            correction=getattr(self, "solvent_correction", None),
            composition_admission=self.analytic_composition_admission,
            gas_admission=self.analytic_derivative_admission,
            operation_provenance=self._last_uma_hvp_operation_provenance,
        )
        self.last_hvp_provenance = provenance
        return result

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

        previous_results = dict(getattr(self, "results", {}))
        previous_execution = getattr(self, "execution_provenance", None)
        previous_solvation = getattr(self, "solvation_result", None)
        had_solvation = hasattr(self, "solvation_result")
        try:
            execution = calculator_execution(self)
            precision = {
                "backend": "fairchem-uma",
                "dtype": (
                    str(self._predictor_unit.inference_settings.base_precision_dtype)
                    if hasattr(self, "_predictor_unit")
                    else "unknown"
                ),
                "device": str(getattr(self, "device", "unknown")),
            }
            composition_admission = getattr(
                self, "analytic_composition_admission", None
            )
            if composition_admission is not None:
                gas_operation = self._analytic_energy_forces_operation(
                    atoms,
                    need_forces="forces" in requested,
                )
                payload, solvent_result = finalize_energy_forces(
                    atoms,
                    correction=getattr(self, "solvent_correction", None),
                    gas_operation=gas_operation,
                    composition_admission=composition_admission,
                    gas_admission=getattr(self, "analytic_derivative_admission", None),
                    execution_provenance=execution,
                    precision_provenance=precision,
                )
            else:
                calc_atoms = self._prepared_calc_atoms(atoms)
                super().calculate(calc_atoms, properties, system_changes)
                gas_results = dict(self.results)
                payload, solvent_result = finalize_energy_forces(
                    calc_atoms,
                    correction=getattr(self, "solvent_correction", None),
                    energy=gas_results["energy"],
                    forces=gas_results.get("forces"),
                    source_unit="eV",
                    execution_provenance=execution,
                    precision_provenance=precision,
                )
        except Exception:
            self.results = previous_results
            if previous_execution is None:
                self.__dict__.pop("execution_provenance", None)
            else:
                self.execution_provenance = previous_execution
            if had_solvation:
                self.solvation_result = previous_solvation
            else:
                self.__dict__.pop("solvation_result", None)
            raise

        self.results = payload
        self.execution_provenance = execution
        if solvent_result is None:
            self.__dict__.pop("solvation_result", None)
        else:
            self.solvation_result = solvent_result
        return self.results

    def calculate_many(
        self,
        atoms_list,
        properties=("energy", "forces"),
    ) -> BatchResult:
        """Use the common result-driven fallback for the third-party wrapper."""
        return default_calculate_many(self, atoms_list, properties)
