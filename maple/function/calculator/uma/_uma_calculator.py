import importlib
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core._config import CACHE_DIR
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator, UMATask
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    from fairchem.core.units.mlip_unit import load_predict_unit
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError("fairchem-core is not installed. Please install it first.")

from .._batch_types import BatchResult
from .._batch_utils import atoms_list_has_pbc


EV2HARTREE = 1.0 / 27.211386245988

UMA_DEFAULT_SIZE = "uma-s-1p1"
UMA_MODELS_MAP = {
    "uma": UMA_DEFAULT_SIZE,
    "uma-s-1p1": "uma-s-1p1",
    "uma-s-1p2": "uma-s-1p2",
    "uma-m-1p1": "uma-m-1p1",
}

UMA_FALLBACK_HF_MODELS = {"uma-s-1p1", "uma-s-1p2", "uma-m-1p1"}
SUPPORTED_UMA_TASKS = {"omol", "omat", "oc20", "odac", "omc", "oc22", "oc25"}
SUPPORTED_UMA_INFERENCE = {"default", "turbo"}

# GPU default and CPU fallback for FAIR Chemistry's inference path. "default"
# is the general-purpose backend; "turbo" speeds up fixed-composition workloads
# (NEB / TS / freq) on CUDA but routes through Triton kernels unavailable on
# CPU. Users override via `model=uma(...,inference=turbo)`; CPU coerces to
# default regardless.
UMA_INFERENCE_SETTINGS = "default"
UMA_CPU_INFERENCE_SETTINGS = "default"
UMA_BATCH_INFERENCE_SETTINGS = "default"


class UMACalculator(FAIRChemCalculator):
    """
    UMA calculator with MAPLE-specific unit conversion and Hessian support.

    Explicit `task=` is respected. When `task` is omitted, MAPLE keeps the
    historical convenience behavior of inferring `omol` for non-periodic
    systems and `omat` for periodic systems.
    """

    supported_hessian_modes = ("numerical",)
    supports_batch_energy_forces = True
    supports_analytic_hessian = False
    supports_hvp = False

    @staticmethod
    def _normalize_device(device: torch.device | str | None) -> str:
        # FAIR Chemistry's MLIP unit accepts only "cpu" or "cuda".
        # Keep UMA's historical behavior: CUDA-like requests use the CUDA
        # backend token, while other strings fall back to CPU.
        device_name = str(device).lower()
        if device_name.startswith("cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _build_predictor(
        checkpoint: str,
        overrides: dict | None,
        device: str,
        checkpoint_path: str | None = None,
        inference_settings: str | None = None,
    ):
        device = UMACalculator._normalize_device(device)
        # Turbo selects FAIR Chemistry's fast GPU execution path only when the
        # user explicitly asks for it; CPU uses the general-purpose backend to
        # avoid Triton GPU kernels on CPU tensors.
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

        if checkpoint_path and os.path.isfile(checkpoint_path):
            compat_path = UMACalculator._prepare_compat_checkpoint(checkpoint, checkpoint_path)
            return load_predict_unit(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )

        if checkpoint in pretrained_mlip.available_models:
            return pretrained_mlip.get_predict_unit(
                checkpoint,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )

        if os.path.isfile(checkpoint):
            compat_path = UMACalculator._prepare_compat_checkpoint(Path(checkpoint).stem, checkpoint)
            return load_predict_unit(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
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
            return load_predict_unit(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
                atom_refs=atom_refs,
                form_elem_refs=form_elem_refs,
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
        device: torch.device,
        model: str = "uma",
        overrides: dict | None = None,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
        task: str | None = None,
        size: str | None = None,
        checkpoint_path: str | None = None,
        inference_settings: str | None = None,
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
        self._batch_predictor_unit = None
        self._checkpoint = checkpoint
        self._checkpoint_path = checkpoint_path
        self._overrides = overrides
        self._auto_task = task is None
        self.hessian = "numerical"

        if implicit == "gbsa" and solvent != "none":
            from ..extra_correction import GBSA, QEqTorch

            self.solvent_correction = GBSA(solvent=solvent, device=self.device)
            self.chargecalc = QEqTorch(device=self.device)
        else:
            self.solvent_correction = None

    def _set_task_from_atoms(self, atoms: Atoms) -> None:
        if not self._auto_task:
            return

        task = self._task_for_atoms(atoms)
        if task == self.task_name:
            return

        self._set_current_task(task)

    def _task_for_atoms(self, atoms: Atoms) -> str:
        if not self._auto_task:
            return self.task_name
        return "omat" if any(atoms.pbc) else "omol"

    def _set_current_task(self, task: str) -> None:
        self._task = UMATask(task)
        self.a2g = partial(AtomicData.from_ase, **self._a2g_kwargs(task, self._predictor_unit))
        self._task_name = task
        self.implemented_properties = [
            t.property for t in self._predictor_unit.dataset_to_tasks[self.task_name]
        ]
        if "energy" in self.implemented_properties:
            self.implemented_properties.append("free_energy")

    @staticmethod
    def _a2g_kwargs(task: str, predictor) -> dict:
        settings = predictor.inference_settings
        if settings.external_graph_gen:
            r_edges, max_neigh = True, 300
        else:
            r_edges, max_neigh = False, None

        return dict(
            task_name=task,
            r_edges=r_edges,
            r_data_keys=["spin", "charge"],
            max_neigh=max_neigh,
            radius=6.0,
            target_dtype=settings.base_precision_dtype,
        )

    @staticmethod
    def _prepare_atoms_metadata(atoms: Atoms) -> None:
        atoms.info["spin"] = int(atoms.info.get("mult", 1))
        atoms.info["charge"] = int(atoms.info.get("charge", 0))

    @staticmethod
    def _predictor_supports_batch(predictor) -> bool:
        settings = predictor.inference_settings
        # FAIR-Chem documents default mode as batch-capable and turbo mode as
        # single-system-only. In current fairchem-core, turbo is identified by
        # merged MoLE weights / compile settings and raises on multi-system
        # AtomicData batches.
        return not (
            getattr(settings, "merge_mole", False)
            or getattr(settings, "compile", False)
        )

    def _batch_predictor(self):
        if self._predictor_supports_batch(self._predictor_unit):
            return self._predictor_unit

        if self.device.type != "cuda":
            return None

        if self._batch_predictor_unit is None:
            self._batch_predictor_unit = self._build_predictor(
                self._checkpoint,
                self._overrides,
                str(self.device),
                checkpoint_path=self._checkpoint_path,
                inference_settings=UMA_BATCH_INFERENCE_SETTINGS,
            )
        if not self._predictor_supports_batch(self._batch_predictor_unit):
            return None
        return self._batch_predictor_unit

    def get_energy(self, atoms: Atoms) -> torch.Tensor:
        self.calculate(atoms, properties=["energy"], system_changes=all_changes)
        energy_value = self.results["energy"]

        if self.solvent_correction:
            energy_value += self.solvent_correction.get_energy(atoms)

        return torch.tensor(energy_value, dtype=torch.float32, device=self.device)

    def get_hessian(
        self,
        atoms: Atoms,
        delta: float = 0.002,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Central-difference numerical Hessian via batched displacement.

        UMA exposes no analytic Hessian (``supported_hessian_modes = ('numerical',)``),
        so this is the only Hessian path. The 2 * 3 * N_movable force
        evaluations are delegated to ``FDHessianEvaluator``, which dispatches
        through ``calc.calculate_many`` and uses FAIR-Chem's AtomicData batch
        predictor when the active inference mode supports batching.
        FixAtoms is respected upstream. Returns a ``(3N, 3N)`` tensor on
        ``self.device`` to preserve the original return-type contract.
        """
        from .._batch_eval import FDHessianEvaluator

        H_np = FDHessianEvaluator(
            self,
            fd_batch_size=getattr(self, "fd_batch_size", None),
        ).hessian(atoms, delta=delta)
        return torch.as_tensor(H_np, dtype=dtype, device=self.device)

    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Evaluate multiple structures through FAIR-Chem's batch predictor.

        UMA inherits from FAIR-Chem's calculator rather than MAPLE's
        ``CalcABC``. FAIR-Chem's documented batch route is
        ``AtomicData.from_ase`` -> ``atomicdata_list_to_batch`` ->
        ``predictor.predict``; use that route when the active predictor
        supports batching. CUDA turbo mode is optimized for single fixed
        composition rollouts and does not accept multi-system batches, so
        ``calculate_many`` lazily creates a default-mode predictor for batch
        calls while leaving single-structure ``calculate`` on turbo.

        Solvent corrections remain single-structure and therefore use the
        sequential fallback.
        """
        props = tuple(properties)
        if "hessian" in props:
            raise NotImplementedError(
                "UMACalculator.calculate_many does not assemble Hessians; "
                "use FDHessianEvaluator for numerical Hessians."
            )

        want_energy = "energy" in props
        want_forces = "forces" in props
        request = [p for p in props if p in ("energy", "forces")]
        if not request:
            return BatchResult()
        atoms_list = list(atoms_list)
        if not atoms_list:
            return BatchResult(
                energies=np.zeros(0, dtype=np.float64) if want_energy else None,
                forces=[] if want_forces else None,
            )

        if atoms_list_has_pbc(atoms_list):
            return self._calculate_many_sequential(atoms_list, request, want_energy, want_forces)

        if self.solvent_correction:
            return self._calculate_many_sequential(atoms_list, request, want_energy, want_forces)

        predictor = self._batch_predictor()
        if predictor is None:
            return self._calculate_many_sequential(atoms_list, request, want_energy, want_forces)

        data_list = []
        for at in atoms_list:
            task = self._task_for_atoms(at)
            self._prepare_atoms_metadata(at)
            self._check_atoms_pbc(at)
            predictor.validate_atoms_data(at, task)
            data_list.append(
                AtomicData.from_ase(at, **self._a2g_kwargs(task, predictor))
            )

        batch = atomicdata_list_to_batch(data_list)
        pred = predictor.predict(batch)

        energies = None
        if want_energy:
            energies = (
                pred["energy"].detach().cpu().numpy().astype(np.float64)
                * EV2HARTREE
            )

        forces_list = None
        if want_forces:
            forces_all = (
                pred["forces"].detach().cpu().numpy().astype(np.float64)
                * EV2HARTREE
            )
            batch_index = batch.batch.detach().cpu().numpy()
            forces_list = [
                forces_all[batch_index == i]
                for i in range(len(atoms_list))
            ]

        first_task = self._task_for_atoms(atoms_list[0])
        if all(self._task_for_atoms(at) == first_task for at in atoms_list):
            self._set_current_task(first_task)

        return BatchResult(energies=energies, forces=forces_list)

    def _calculate_many_sequential(
        self,
        atoms_list,
        request,
        want_energy: bool,
        want_forces: bool,
    ) -> BatchResult:
        energies = [] if want_energy else None
        forces_list = [] if want_forces else None

        for at in atoms_list:
            self.calculate(at, properties=list(request), system_changes=all_changes)
            if want_energy:
                if "free_energy" in self.results:
                    energies.append(float(self.results["free_energy"]))
                else:
                    energies.append(float(self.results["energy"]))
            if want_forces:
                forces_list.append(np.asarray(self.results["forces"], dtype=np.float64))

        return BatchResult(
            energies=np.asarray(energies, dtype=np.float64) if energies is not None else None,
            forces=forces_list,
        )

    def calculate(self, atoms, properties=None, system_changes=None):
        self._set_task_from_atoms(atoms)

        self._prepare_atoms_metadata(atoms)

        super().calculate(atoms, properties, system_changes)

        if "energy" in self.results:
            self.results["energy"] *= EV2HARTREE
        if "free_energy" in self.results:
            self.results["free_energy"] *= EV2HARTREE
        if "forces" in self.results:
            self.results["forces"] *= EV2HARTREE

        if self.solvent_correction:
            atoms.atomic_charges = self.chargecalc(atoms)
            solvent_energy, solvent_force = self.solvent_correction.get_energy_and_force(atoms)
            if "energy" in self.results:
                self.results["energy"] += solvent_energy.item()
            if "free_energy" in self.results:
                self.results["free_energy"] += solvent_energy.item()
            if "forces" in self.results:
                self.results["forces"] += solvent_force.detach().cpu().numpy()

        return self.results
