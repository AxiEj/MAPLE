"""Hash-bound AnySolv-MoE scalar solution-energy correction.

The released MoE predicts ``E_SMD - E_gas`` in eV.  MAPLE deliberately exposes
only that scalar correction: the pinned checkpoint has an orientation-specific
force defect under an all-component finite-difference audit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib import metadata
from importlib import import_module
from importlib import util as importlib_util
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Any, ClassVar, cast
from weakref import WeakValueDictionary, ref

import numpy as np

from ...backends import CompositeCalculator, SolvationBackend
from ...model_capabilities import SolvationCapabilities, load_model_provenance_card
from ...solvation_execution import SolvationExecution, STRICT_INDUCTOR_OPTIONS
from .result import SolvationResult

ANYSOLV_SOURCE_REVISION = "1778e1f18384e5f59de1475182765733a7401e39"
ANYSOLV_MOE_CHECKPOINT_SHA256 = (
    "77b7d883ac1a01fc13d78b8cf5530e5709af679e6fbdec53cb086ae0a90bd251"
)
ANYSOLV_MODEL_CARD_ROOT = Path(__file__).resolve().parents[2] / "model_cards"
ANYSOLV_MOE_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "model" / "anysolv" / "model_moe.pt"
)
EV_PER_HARTREE = 27.211386245988

_ACTIVE_COMPILED_OWNER = ContextVar("maple_anysolv_compiled_owner", default=None)


def _owned_inductor_backend(graph, inputs):
    """Stable compiler identity; execution attribution is call-local, not cached."""
    from torch._dynamo.backends.registry import lookup_backend

    compiled = lookup_backend("inductor")(
        graph, inputs, config_patches=dict(STRICT_INDUCTOR_OPTIONS)
    )

    def execute(*values):
        reference = _ACTIVE_COMPILED_OWNER.get()
        owner = reference() if reference is not None else None
        if owner is None:
            raise RuntimeError(
                "AnySolv compiled owner has been released or is not active."
            )
        result = compiled(*values)
        owner._compiled_backend_executions += 1
        return result

    return execute


ANYSOLV_SOURCE_ARTIFACTS = {
    "__init__.py": "93ec85c99be8dad866be3358b74883e16e9da2f332d780030766c149fc48d481",
    "predict.py": "e3ece22aa323c8b1165dbe588c8340735e0bc9be4ebe2a7276349bd691c4a0fc",
    "model.py": "12e81d1bd5aaae67ce509cb56d2e7de044990345ce639f6b413249d583388027",
    "data.py": "230310d321ccaccad92933e8eb8d04471591346802ae2e7f2cd3526ad489120d",
    "solvent.py": "301111cf396a674b615fa0ec5aec87219ca381210c99c44d78964e807f00e595",
    "LICENSE": "8e329a100b25a4caaa2db75bdbe7ba494d7b61673470a666b87745eb808d287a",
    "_const/solvent_descriptors.json": "21fc32459bea59b64441da1af2da4a0dab6ae3509920eea84bb3f16a555f3bec",
    "_const/grid_mats.pt": "35411a01d3fb6ec5b9a690c2b664afcb979634f6f64130c5e5b767b18ca19622",
    "_backbone/Jd.pt": "b4059c45be246dcb6c49c545670b65c56550eb0c2e7a9c92b4b50a92d370dbe2",
    "_backbone/_compat/inference.py": "42eda0762a18ed4d3c1c820246782dfaa93f4f0a2f55c8f74cd2d00494dca792",
}
_SOURCE_DIGEST = hashlib.sha256(
    json.dumps(
        {"revision": ANYSOLV_SOURCE_REVISION, "artifacts": ANYSOLV_SOURCE_ARTIFACTS},
        sort_keys=True,
    ).encode()
).hexdigest()


@dataclass(frozen=True)
class _OwnedResident:
    identity: tuple[str, ...]
    model: Any


# Runtimes keep strong references; the cache releases models with their owners.
# The upstream pathname cache is neither consumed nor modified.
_RESIDENTS: WeakValueDictionary[tuple[str, ...], _OwnedResident] = WeakValueDictionary()
_RESIDENT_LOCK = Lock()
_PREPARATION_LOCK = Lock()

# Only the upstream trained/validated set is executable.  Descriptor availability
# for 179 solvents is not treated as scientific admission.
_VALIDATED_SOLVENTS = {
    "water",
    "acetone",
    "acetonitrile",
    "aniline",
    "benzaldehyde",
    "benzene",
    "bromobenzene",
    "carbon tetrachloride",
    "dichloromethane",
    "chloroform",
    "chlorobenzene",
    "carbon disulfide",
    "cyclohexanone",
    "1,2-dichloroethane",
    "diiodomethane",
    "1,4-dioxane",
    "dmf",
    "dmso",
    "ethanol",
    "diethyl ether",
    "ethyl acetate",
    "n-hexadecane",
    "n-hexane",
    "iodobenzene",
    "methanol",
    "nitromethane",
    "n-methylformamide",
    "1-octanol",
    "o-dichlorobenzene",
    "n-pentane",
    "1-pentanol",
    "1-propanol",
    "2,2,2-trifluoroethanol",
    "thf",
    "toluene",
    "tributyl phosphate",
}
_SOLVENT_ALIASES = {
    "n,n-dimethylformamide": "dmf",
    "dimethylformamide": "dmf",
    "dimethyl sulfoxide": "dmso",
    "tetrahydrofuran": "thf",
    "ethyl ethanoate": "ethyl acetate",
    "pentane": "n-pentane",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_solvent(value: str) -> str:
    token = " ".join(str(value).strip().casefold().split())
    token = _SOLVENT_ALIASES.get(token, token)
    if token not in _VALIDATED_SOLVENTS:
        raise ValueError(
            f"Unsupported AnySolv-MoE solvent {value!r}; only the frozen 36-solvent "
            "trained/validated set is enabled."
        )
    return token


def _validate_source(root: Path) -> dict[str, str]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"AnySolv source root is not an auditable Git checkout: {root}"
        ) from exc
    if revision != ANYSOLV_SOURCE_REVISION:
        raise ValueError(
            f"AnySolv source revision mismatch: expected {ANYSOLV_SOURCE_REVISION}, got {revision}."
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if dirty:
        raise ValueError("AnySolv source checkout has modified tracked files.")
    for relative, expected in ANYSOLV_SOURCE_ARTIFACTS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Required AnySolv source artifact not found: {path}"
            )
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"AnySolv source checksum mismatch for {relative}: expected {expected}, got {actual}."
            )
    return {"source_revision": revision, "source_root": str(root)}


class _OfficialAnySolvRuntime:
    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint: Path,
        device: str,
        checkpoint_sha256: str,
        dtype: str = "float32",
        inference: str = "reference",
    ) -> None:
        self._execution = SolvationExecution(
            device=device, dtype=dtype, inference=inference
        )
        self._signature = None
        self._compiled_run = None
        self._compiled_backend_executions = 0
        self.compilation_active = False
        self._evaluation_lock = Lock()
        module: ModuleType | None = sys.modules.get("anysolv")
        if module is None:
            spec = importlib_util.spec_from_file_location(
                "anysolv",
                source_root / "__init__.py",
                submodule_search_locations=[str(source_root)],
            )
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Cannot create an import spec for AnySolv at {source_root}."
                )
            module = importlib_util.module_from_spec(spec)
            sys.modules["anysolv"] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                for name in tuple(sys.modules):
                    if name == "anysolv" or name.startswith("anysolv."):
                        sys.modules.pop(name, None)
                raise
        assert module is not None
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ImportError("Imported AnySolv package has no filesystem location.")
        module_root = Path(module_file).resolve().parent
        if module_root != source_root.resolve():
            raise ImportError(
                f"Imported AnySolv from {module_root}, expected pinned source {source_root}."
            )
        load_model = getattr(module, "load_model", None)
        settings_type = getattr(module, "InferenceSettings", None)
        if not callable(load_model) or not callable(settings_type):
            raise ImportError(  # noqa: TRY004 - missing package API is an import failure
                "Pinned AnySolv source does not expose load_model/InferenceSettings."
            )
        data_module = import_module("anysolv.data")
        solvent_module = import_module("anysolv.solvent")
        self._build_data = cast(Callable[..., Any], data_module.build_atomic_data)
        self._solvent_vector = cast(
            Callable[..., Any], solvent_module.get_solvent_vector
        )
        self._device = self._execution.device
        settings = (
            settings_type()
            if self._execution.inference == "reference"
            else settings_type(
                execution_mode="umas_fast_gpu",
                tf32=False,
                compile=False,
                merge_mole=True,
            )
        )
        key = (
            str(source_root.resolve()),
            _SOURCE_DIGEST,
            checkpoint_sha256,
            self._device,
            self._execution.dtype,
            self._execution.inference,
            repr(settings),
        )
        with _RESIDENT_LOCK:
            # Merged MoE weights belong to exactly one runtime/signature.
            resident = (
                _RESIDENTS.get(key)
                if self._execution.inference == "reference"
                else None
            )
            if resident is None:
                model = load_model(
                    checkpoint,
                    device=self._device,
                    dtype=self._execution.torch_dtype,
                    inference_settings=settings,
                )
                resident = _OwnedResident(identity=key, model=model)
                if self._execution.inference == "reference":
                    _RESIDENTS[key] = resident
            self._resident = resident

    def _attach_owned_compile(self):
        """Compile the exact prepared forward; upstream staging remains hybrid."""
        import torch

        model = self._resident.model
        self._compiled_run = torch.compile(
            model._raw_forward, backend=_owned_inductor_backend, dynamic=True
        )
        model._run = self._compiled_run

    def _check_optimized_activation(self, data) -> None:
        model = self._resident.model
        settings = model._settings
        backend = model.backbone.backend
        if (
            not model._prepared
            or self._compiled_run is None
            or model._run is not self._compiled_run
            or settings.execution_mode != "umas_fast_gpu"
            or settings.tf32
            or settings.compile
            or not settings.merge_mole
            or type(backend).__name__ != "UMASFastGPUBackend"
            or model.backbone._merged_composition is None
        ):
            raise RuntimeError(
                "AnySolv optimized inference did not activate owned compilation and merged "
                "umas_fast_gpu execution; eager fallback is forbidden."
            )
        # Use the pinned backbone's actual merge-signature check, in addition
        # to our stronger immutable atom-count/composition/solvent signature.
        model.backbone.on_predict_check(data)

    def evaluate(
        self,
        numbers: np.ndarray,
        positions: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        solvent: str | None,
        scalar_only: bool = False,
    ) -> tuple[float, np.ndarray | None]:
        with self._execution.precision_context():
            if self._execution.inference == "optimized":
                self.compilation_active = False
            self._execution.check_precision()
            return self._evaluate_strict(
                numbers,
                positions,
                charge=charge,
                multiplicity=multiplicity,
                solvent=solvent,
                scalar_only=scalar_only,
            )

    def _evaluate_strict(
        self,
        numbers,
        positions,
        *,
        charge,
        multiplicity,
        solvent,
        scalar_only,
    ):
        import torch

        model = self._resident.model
        solvent_vector = self._solvent_vector(
            solvent, eps_transform=getattr(model, "eps_transform", None)
        )
        with self._evaluation_lock:
            if self._execution.inference == "optimized":
                # The official vector is CPU-resident, before its direct transfer
                # by build_atomic_data. Coordinates may change; merged weights may not.
                signature = (
                    len(numbers),
                    tuple(np.bincount(numbers, minlength=100).tolist()),
                    charge,
                    multiplicity,
                    tuple(solvent_vector.reshape(-1).tolist()),
                )
                if self._signature is not None and signature != self._signature:
                    raise ValueError(
                        "AnySolv optimized input signature changed; construct a new calculator."
                    )
                self._signature = signature
            data = self._build_data(
                (numbers, positions),
                charge=charge,
                spin=multiplicity,
                solvent=solvent_vector,
                dtype=self._execution.torch_dtype,
                device=self._device,
            )
            self._execution.check_precision(inputs=data, model=model)
            if self._execution.inference == "optimized":
                # compile=False leaves upstream preparation free of Dynamo
                # cache-limit mutation. Keep preparation + invocation serialized.
                with _PREPARATION_LOCK:
                    config = torch._dynamo.config
                    if config.suppress_errors:
                        raise ValueError(
                            "AnySolv optimized inference requires Dynamo suppress_errors=False; "
                            "eager fallback is forbidden."
                        )
                    self.compilation_active = False
                    if not model._prepared:
                        model._prepare(data)
                        self._attach_owned_compile()
                    self._check_optimized_activation(data)
                    self._execution.check_precision(inputs=data, model=model)
                    previous_executions = self._compiled_backend_executions
                    token = _ACTIVE_COMPILED_OWNER.set(ref(self))
                    try:
                        with config.patch(fail_on_recompile_limit_hit=True):
                            out = model(data)
                    finally:
                        _ACTIVE_COMPILED_OWNER.reset(token)
                    self._check_optimized_activation(data)
                    if self._compiled_backend_executions <= previous_executions:
                        raise RuntimeError(
                            "AnySolv compiled backend did not execute; eager fallback is forbidden."
                        )
                    self.compilation_active = True
            else:
                out = model(data)
            self._execution.check_precision(inputs=data, model=model)
        energy = out["energy"]
        raw_energy = energy["energy"] if isinstance(energy, dict) else energy
        rmsd = model.norm["energy"]["rmsd"]
        mean = model.norm["energy"]["mean"]
        # Exact unnormalization from the pinned official predictor.
        delta_energy = float((raw_energy.reshape(-1)[0] * rmsd + mean).item())
        if scalar_only:
            return delta_energy, None
        forces = out["forces"]
        raw_forces = forces["forces"] if isinstance(forces, dict) else forces
        delta_forces = (raw_forces * rmsd).detach().cpu().numpy().astype(np.float64)
        return delta_energy, delta_forces


@dataclass(frozen=True)
class AnySolvBackend(SolvationBackend):
    """Official AnySolv-MoE correction, admitted for scalar SP only."""

    solvent: str
    model_path: str | Path = ANYSOLV_MOE_DEFAULT_PATH
    source_root: str | None = None
    device: str = "cuda:0"
    dtype: str = "float32"
    inference: str = "reference"
    execution_task: str | None = None
    _execution: SolvationExecution = field(init=False, repr=False, compare=False)
    _source_identity: dict[str, str] = field(init=False, repr=False, compare=False)
    _runtime: _OfficialAnySolvRuntime = field(init=False, repr=False, compare=False)

    capabilities = SolvationCapabilities(
        energy=True,
        forces=False,
        conservative_forces=False,
        hessian="none",
        supports_pbc=False,
        multisolvent=True,
        energy_reference="relative",
        supports_absolute_solvation=False,
    )
    supported_properties: ClassVar[frozenset[str]] = frozenset({"energy"})
    conservative_forces = False
    supports_pbc = False
    multisolvent = True
    energy_reference = "relative"
    supports_absolute_solvation = False

    def __post_init__(self) -> None:
        provenance = load_model_provenance_card("anysolv-moe", ANYSOLV_MODEL_CARD_ROOT)
        if self.execution_task is not None:
            provenance.validate_task(self.execution_task)
        execution = SolvationExecution(self.device, self.dtype, self.inference)
        execution.validate_card(provenance, task=self.execution_task or "sp")
        for name in ("device", "dtype", "inference"):
            object.__setattr__(self, name, getattr(execution, name))
        object.__setattr__(self, "_execution", execution)
        solvent = _resolve_solvent(self.solvent)
        if not isinstance(self.model_path, (str, Path)):
            raise TypeError("AnySolv-MoE model_path must be a string or Path.")
        if not str(self.model_path).strip():
            raise ValueError("AnySolv-MoE model_path cannot be empty.")
        if not isinstance(self.source_root, (str, Path)):
            raise TypeError(
                "AnySolv-MoE source_root must be an explicit string or Path to "
                "the pinned checkout."
            )
        if not str(self.source_root).strip():
            raise ValueError("AnySolv-MoE source_root cannot be empty.")
        checkpoint = Path(self.model_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AnySolv-MoE checkpoint not found: {checkpoint}")
        actual = _sha256(checkpoint)
        if actual != ANYSOLV_MOE_CHECKPOINT_SHA256:
            raise ValueError(
                "AnySolv-MoE checkpoint checksum mismatch: expected "
                f"{ANYSOLV_MOE_CHECKPOINT_SHA256}, got {actual}."
            )
        source = Path(self.source_root).expanduser().resolve()
        source_identity = _validate_source(source)
        object.__setattr__(self, "solvent", solvent)
        object.__setattr__(self, "model_path", str(checkpoint))
        object.__setattr__(self, "source_root", str(source))
        object.__setattr__(self, "_source_identity", source_identity)
        object.__setattr__(
            self,
            "_runtime",
            _OfficialAnySolvRuntime(
                source_root=source,
                checkpoint=checkpoint,
                device=execution.device,
                dtype=execution.dtype,
                inference=execution.inference,
                checkpoint_sha256=actual,
            ),
        )

    @staticmethod
    def _electronic_state(atoms) -> tuple[int, int]:
        info = getattr(atoms, "info", {})
        if "charge" not in info or "mult" not in info:
            raise ValueError(
                "AnySolv-MoE requires explicit atoms.info['charge'] and atoms.info['mult']."
            )
        try:
            charge = float(info["charge"])
            multiplicity = float(info["mult"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "AnySolv-MoE charge and multiplicity must be integers."
            ) from exc
        if not np.isfinite(charge) or not charge.is_integer():
            raise ValueError("AnySolv-MoE charge must be a finite integer.")
        if (
            not np.isfinite(multiplicity)
            or not multiplicity.is_integer()
            or multiplicity < 1
        ):
            raise ValueError("AnySolv-MoE multiplicity must be a positive integer.")
        charge = int(charge)
        multiplicity = int(multiplicity)
        if not -100 <= charge <= 100:
            raise ValueError("AnySolv-MoE charge embedding capacity is -100..100.")
        if not 1 <= multiplicity <= 100:
            raise ValueError("AnySolv-MoE multiplicity embedding capacity is 1..100.")
        return charge, multiplicity

    @property
    def provenance(self) -> dict[str, Any]:
        runtime_versions = {}
        for distribution in ("torch", "numpy", "fairchem-core"):
            try:
                runtime_versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                runtime_versions[distribution] = "not-installed"
        return {
            "provider": "anysolv-moe",
            **self._execution.provenance,
            "compiled_scope": (
                "upstream_hybrid" if self.inference == "optimized" else "none"
            ),
            "compilation_active": bool(
                getattr(self._runtime, "compilation_active", False)
            ),
            "source_digest": _SOURCE_DIGEST,
            **self._source_identity,
            "checkpoint_sha256": ANYSOLV_MOE_CHECKPOINT_SHA256,
            "model_path": self.model_path,
            "solvent": self.solvent,
            "units": {"energy": "eV", "forces": "eV/angstrom"},
            "runtime_versions": runtime_versions,
            "source_identity": (
                "pinned Git revision, clean tracked tree, and hash-bound runtime constants"
            ),
            "quantity_kind": "geometry_level_energy_correction",
            "energy_reference": "relative",
            "supports_absolute_solvation": False,
            "supports_multifragment_mechanics": True,
            "validated_multifragment_domain": "unknown",
            "forces_exposed": False,
            "force_rejection_artifact": (
                "docs/pretrained-solvation-hub/benchmarks/"
                "anysolv-moe-mechanics-2026-10-23/result.json"
            ),
        }

    def evaluate(
        self, atoms, *, need_forces: bool = False, calculator=None
    ) -> SolvationResult:
        if need_forces:
            raise NotImplementedError(
                "AnySolv-MoE forces are disabled: the pinned checkpoint fails rigid-rotation "
                "covariance and an all-component finite-difference audit at a pole orientation."
            )
        if bool(np.any(getattr(atoms, "pbc", False))):
            raise NotImplementedError(
                "AnySolv-MoE is non-periodic; PBC is unsupported."
            )
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if len(atoms) == 0:
            raise ValueError("AnySolv-MoE requires at least one atom.")
        if (
            numbers.ndim != 1
            or numbers.size != len(atoms)
            or np.any(numbers < 1)
            or np.any(numbers > 99)
        ):
            raise ValueError(
                "AnySolv-MoE atomic numbers must have shape (N,) and lie in the "
                "upstream embedding capacity 1..99; this is not a validated chemistry domain."
            )
        if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
            raise ValueError("AnySolv-MoE positions must be finite with shape (N, 3).")
        charge, multiplicity = self._electronic_state(atoms)
        if calculator is not None:
            if isinstance(calculator, CompositeCalculator):
                capabilities = calculator.potential.capabilities
                supports_charge = capabilities.supports_charge
                supports_multiplicity = capabilities.supports_multiplicity
            else:
                capabilities = getattr(calculator, "capabilities", None)
                supports_charge = supports_multiplicity = bool(
                    getattr(calculator, "SUPPORTS_CHARGE_MULT", False)
                )
            if getattr(capabilities, "solvation_mode", "none") in {
                "native",
                "alchemical",
                "property_only",
            }:
                raise ValueError("AnySolv-MoE requires a vacuum potential context.")
            if (charge != 0 and not supports_charge) or (
                multiplicity != 1 and not supports_multiplicity
            ):
                raise ValueError(
                    "AnySolv-MoE base calculator does not support this charge/multiplicity."
                )
        energy_ev, _diagnostic_forces = self._runtime.evaluate(
            numbers,
            positions,
            charge=charge,
            multiplicity=multiplicity,
            solvent=self.solvent,
            scalar_only=True,
        )
        if not np.isfinite(energy_ev):
            raise ValueError("AnySolv-MoE correction energy must be finite.")
        energy_hartree = energy_ev / EV_PER_HARTREE
        return SolvationResult(
            energy_hartree=energy_hartree,
            forces_hartree_per_angstrom=None,
            components_hartree={"anysolv_moe_delta": energy_hartree},
            provenance={
                **self.provenance,
                "charge": charge,
                "multiplicity": multiplicity,
                "validated_element_domain": "unknown",
                "scientific_status": "experimental_single_point_only",
            },
            quantity_kind="geometry_level_energy_correction",
        )
