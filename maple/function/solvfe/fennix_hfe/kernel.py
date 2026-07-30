"""Pinned native-JAX FeNNix-Bio1 alchemical mechanics kernel.

This module deliberately stops below sampling and free-energy estimation.  A
finite kernel result is not an HFE, accuracy, GPU-admission, or performance
claim.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from importlib import metadata
from pathlib import Path
from pathlib import PurePath
from typing import Any, Callable, Mapping

import numpy as np

from ...calculator.fennol._fennix_bio1_calculator import (
    FENNIX_BIO1_CARD,
    FENNIX_BIO1_CHECKPOINTS,
    FENNIX_BIO1_PMC_REVISION,
    FENNIX_BIO1_SOURCE_REVISION,
    FENNOL_ASE_SHA256,
    FENNOL_DISTRIBUTION_VERSION,
    FENNOL_PREPROCESSING_SHA256,
    _load_card,
    _official_runtime_validator,
    _sha256,
    _validate_official_card,
    _validate_runtime_receipt,
)
from .system import validate_fennix_alchemical_system
from .types import (
    FENNIX_KERNEL_SCIENTIFIC_SCOPE,
    FENNIX_PACKAGE_TREE_FILE_COUNT,
    FENNIX_PACKAGE_TREE_SHA256,
    FENNIX_REPULSION_NLH_COEFFICIENTS_PROVENANCE,
    FeNNixAlchemicalParameters,
    FeNNixAlchemicalSystem,
    FeNNixKernelIdentity,
    FeNNixKernelResult,
    FeNNixLambdaState,
    FeNNixParameterTreeReceipt,
)

FENNIX_HFE_MODEL_VARIANT = "medium"
FENNIX_NATIVE_RUNTIME_SHA256 = {
    "fennol/ase.py": FENNOL_ASE_SHA256,
    "fennol/models/fennix.py": "2e092b782b2fcabf2b1d355274d34888faa3b6aca8bee60e44fcea71245991af",
    "fennol/models/preprocessing.py": FENNOL_PREPROCESSING_SHA256,
    "fennol/models/embeddings/charge_embeddings.py": (
        "b6e731a217bca04d9ed0b3a472175839283fce4b2f34edd14c25e0b3ecd96114"
    ),
    "fennol/models/physics/repulsion.py": (
        "ee952d840432090a38bf9eedda2429b7f3a94c4b2f0d1f9dac27d2297f165c22"
    ),
    "fennol/models/physics/nlh_coeffs.dat": (
        "9f6ad25db062dec6552e6a2132d17484da2e4bfd30e70611a94a8a886cd1e32a"
    ),
    "fennol/models/misc/encodings.py": (
        "cfe1a486efc1f0a35d16a50dc09c5f28f717bb25d0cc63254f9e201aee3f1aae"
    ),
    "fennol/models/modules.py": (
        "afa28295acf9935cfa633d7ad03bec7100b1c7788f2b80ec0fa89db4e2cb19f0"
    ),
    "fennol/utils/periodic_table.py": (
        "e6b2dd254305c0c065fdcb41249fa80ed92e8b0e00b16dfa0a039ea3f313006c"
    ),
}
FENNIX_RUNTIME_PACKAGE_VERSIONS = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "flax": "0.12.8",
    "numpy": "2.4.6",
}
FENNIX_GPU_RUNTIME_PACKAGE_VERSIONS = {
    "jax-cuda12-plugin": "0.10.2",
    "jax-cuda12-pjrt": "0.10.2",
}
FENNIX_BIO1M_ORIGINAL_PARAMETER_TREE_FINGERPRINT = (
    "e0554dc2af0a652de5f42be138eb69449d519c89bad972e56c88403488e06e11"
)
FENNIX_BIO1M_DERIVED_PARAMETER_TREE_FINGERPRINT = (
    "0fe05dfdbe83b110a3a7c4bc42c3d5c7400645169721ad2d9aa3bf836e735d5c"
)
FENNIX_BIO1M_FIXED_SPECIES_ENCODING_FLOAT64_SHA256 = (
    "6e9ddf0e4b7c91aa0c8b6fdd4af617ec87f5abc0f51c0c7262395569c22dbe5b"
)


def _jax_modules() -> tuple[Any, Any]:
    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
    except ImportError as exc:
        raise ImportError(
            "The FeNNix native alchemical kernel requires the pinned optional "
            "FeNNol/JAX runtime."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("FeNNix native mechanics require JAX x64 execution.")
    if str(jax.config.jax_default_matmul_precision).lower() != "highest":
        raise RuntimeError("FeNNix native mechanics require highest matmul precision.")
    if os.environ.get("NVIDIA_TF32_OVERRIDE", "").strip() == "1":
        raise RuntimeError("FeNNix native mechanics forbid NVIDIA TF32 override.")
    return jax, jnp


def _floating_dtypes(parameter_tree: Any) -> tuple[str, ...]:
    jax, jnp = _jax_modules()
    return tuple(
        sorted(
            {
                str(np.dtype(leaf.dtype))
                for leaf in jax.tree_util.tree_leaves(parameter_tree)
                if getattr(leaf, "dtype", None) is not None
                and jnp.issubdtype(leaf.dtype, jnp.floating)
            }
        )
    )


def upcast_parameter_tree_float64(
    parameter_tree: Any,
    *,
    original_checkpoint_sha256: str,
) -> tuple[Any, FeNNixParameterTreeReceipt]:
    """Deterministically derive an in-memory float64 tree without checkpoint edits."""

    jax, jnp = _jax_modules()

    def upcast(leaf: Any) -> Any:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            # Convert on the host with NumPy before device placement.  Direct
            # JAX f32->f64 conversion flushes Bio1M subnormals on CPU but
            # preserves them on GPU.  NumPy's widening conversion is exact for
            # every float32 value, including signed subnormals.
            host = np.asarray(leaf)
            host_float64 = host.astype(np.float64)
            return jnp.asarray(host_float64)
        return leaf

    original_dtypes = _floating_dtypes(parameter_tree)
    reduced = {"float16", "bfloat16"}
    if reduced.intersection(original_dtypes):
        raise ValueError(
            "FeNNix parameter source contains reduced float16/bfloat16 precision."
        )
    original_fingerprint = _parameter_tree_fingerprint(parameter_tree)
    derived = jax.tree_util.tree_map(upcast, parameter_tree)
    for leaf in jax.tree_util.tree_leaves(derived):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            if np.dtype(dtype) != np.dtype(np.float64):
                raise RuntimeError(
                    "FeNNix derived parameter tree contains non-float64 data."
                )
    derived_dtypes = _floating_dtypes(derived)
    if derived_dtypes != ("float64",):
        raise RuntimeError(
            "FeNNix derived floating parameter dtype set must be exactly float64."
        )
    return derived, FeNNixParameterTreeReceipt(
        original_checkpoint_sha256=original_checkpoint_sha256,
        original_parameter_fingerprint=original_fingerprint,
        derived_parameter_fingerprint=_parameter_tree_fingerprint(derived),
        original_floating_dtypes=original_dtypes,
        derived_floating_dtypes=derived_dtypes,
    )


def _parameter_tree_fingerprint(parameter_tree: Any) -> str:
    jax, _ = _jax_modules()
    digest = hashlib.sha256()
    path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(parameter_tree)
    digest.update(str(tree_definition).encode("utf-8"))
    for path, leaf in path_leaves:
        array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise TypeError("FeNNix parameter trees may not contain object arrays.")
        normalized = np.ascontiguousarray(
            array.astype(array.dtype.newbyteorder("<"), copy=False)
        )
        metadata = {
            "path": jax.tree_util.keystr(path),
            "dtype": normalized.dtype.str,
            "shape": normalized.shape,
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _contains_overflow(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == "overflow" and bool(np.asarray(child).any()):
                return True
            if _contains_overflow(child):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_overflow(child) for child in value)
    return False


def _validate_native_runtime_sources() -> dict[str, str]:
    try:
        distribution = metadata.distribution("FeNNol")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"FeNNix native mechanics require FeNNol {FENNOL_DISTRIBUTION_VERSION}."
        ) from exc
    observed: dict[str, str] = {}
    for relative_path, expected in FENNIX_NATIVE_RUNTIME_SHA256.items():
        path = Path(str(distribution.locate_file(PurePath(relative_path))))
        if not path.is_file():
            raise ImportError(f"Pinned FeNNol source file is missing: {relative_path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Pinned FeNNol native source mismatch for {relative_path}: "
                f"expected {expected}, got {actual}."
            )
        observed[relative_path] = actual
    return observed


def _validate_fennol_distribution_tree() -> tuple[str, int]:
    try:
        distribution = metadata.distribution("FeNNol")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"FeNNix native mechanics require FeNNol {FENNOL_DISTRIBUTION_VERSION}."
        ) from exc
    rows: list[tuple[str, str]] = []
    for entry in distribution.files or ():
        relative_path = PurePath(str(entry))
        parts = relative_path.parts
        if not parts or parts[0] != "fennol":
            continue
        if "__pycache__" in parts or relative_path.suffix in {".pyc", ".pyo"}:
            continue
        path = Path(str(distribution.locate_file(relative_path)))
        if path.is_file():
            rows.append((relative_path.as_posix(), _sha256(path)))
    rows.sort()
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    tree_sha256 = hashlib.sha256(canonical).hexdigest()
    if (
        len(rows) != FENNIX_PACKAGE_TREE_FILE_COUNT
        or tree_sha256 != FENNIX_PACKAGE_TREE_SHA256
    ):
        raise ValueError(
            "Pinned FeNNol package source/data tree mismatch: expected "
            f"{FENNIX_PACKAGE_TREE_FILE_COUNT} files and "
            f"{FENNIX_PACKAGE_TREE_SHA256}, got {len(rows)} files and "
            f"{tree_sha256}."
        )
    return tree_sha256, len(rows)


def _official_model_loader(checkpoint_path: str) -> Any:
    try:
        fennol = importlib.import_module("fennol")
    except ImportError as exc:
        raise ImportError("The FeNNix native kernel requires FeNNol.") from exc
    loader = getattr(fennol, "load", None)
    if loader is None or not callable(loader):
        raise ImportError("The pinned FeNNol runtime does not expose fennol.load.")
    return loader(checkpoint_path)


def _official_runtime_package_versions(platform: str) -> dict[str, str]:
    names = list(FENNIX_RUNTIME_PACKAGE_VERSIONS)
    if platform == "gpu":
        names.extend(FENNIX_GPU_RUNTIME_PACKAGE_VERSIONS)
    observed: dict[str, str] = {}
    for name in names:
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise ImportError(f"FeNNix runtime package is missing: {name}") from exc
    if platform == "cpu":
        observed.update({name: "N/A" for name in FENNIX_GPU_RUNTIME_PACKAGE_VERSIONS})
    return observed


def _validate_runtime_package_versions(
    versions: Mapping[str, object],
    *,
    platform: str,
) -> dict[str, str]:
    expected = dict(FENNIX_RUNTIME_PACKAGE_VERSIONS)
    expected.update(
        FENNIX_GPU_RUNTIME_PACKAGE_VERSIONS
        if platform == "gpu"
        else {name: "N/A" for name in FENNIX_GPU_RUNTIME_PACKAGE_VERSIONS}
    )
    observed = {name: str(versions.get(name, "")).strip() for name in expected}
    if observed != expected:
        raise ValueError(
            "FeNNix requires the exact pinned JAX/XLA runtime package versions; "
            f"expected {expected}, got {observed}."
        )
    return observed


def _require_float64_tree(value: Any, *, label: str) -> None:
    jax, jnp = _jax_modules()
    for leaf in jax.tree_util.tree_leaves(value):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            if np.dtype(dtype) != np.dtype(np.float64):
                raise RuntimeError(f"FeNNix {label} contains forbidden {dtype} data.")


def _audit_gradient_jaxpr_float64(
    jax: Any,
    gradient_function: Callable[..., Any],
    variables: Any,
    inputs: Any,
) -> None:
    """Reject reduced floating-point primitives in the traced Hamiltonian."""

    make_jaxpr = getattr(jax, "make_jaxpr", None)
    if make_jaxpr is None:
        raise RuntimeError("FeNNix runtime cannot audit the scalar Hamiltonian JAXPR.")
    jaxpr_text = str(make_jaxpr(gradient_function)(variables, inputs)).lower()
    forbidden_tokens = (
        "f16[",
        "f32[",
        "bf16[",
        "float16",
        "float32",
        "bfloat16",
        "tf32",
    )
    observed = [token for token in forbidden_tokens if token in jaxpr_text]
    if observed:
        raise RuntimeError(
            "FeNNix scalar Hamiltonian JAXPR contains forbidden reduced precision: "
            + ", ".join(observed)
        )


def _tree_shape_dtype_signature(jax: Any, value: Any) -> tuple[str, tuple[tuple, ...]]:
    leaves, tree_definition = jax.tree_util.tree_flatten(value)
    leaf_signatures = tuple(
        (
            tuple(getattr(leaf, "shape", ())),
            str(getattr(leaf, "dtype", type(leaf).__name__)),
        )
        for leaf in leaves
    )
    return str(tree_definition), leaf_signatures


def _promote_fixed_species_encoding_float64(model: Any) -> str:
    """Replace Bio1M's hard-coded f32 fixed encoding with the same f64 table."""

    try:
        nn = importlib.import_module("flax.linen")
        periodic = importlib.import_module("fennol.utils.periodic_table")
    except ImportError as exc:
        raise ImportError(
            "FeNNix float64 fixed-encoding promotion requires pinned Flax/FeNNol."
        ) from exc
    jax, jnp = _jax_modules()
    zmax = 86
    atomic_number = np.arange(1, zmax + 1, dtype=np.float64).reshape(-1, 1)
    electronic = np.asarray(periodic.EL_STRUCT[1 : zmax + 1], dtype=np.float64)[:, :15]
    valence = np.asarray(periodic.VALENCE_STRUCTURE[1 : zmax + 1], dtype=np.float64)
    reference = np.asarray(
        [zmax] + [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6] + [2, 6, 10, 14],
        dtype=np.float64,
    )
    table = np.concatenate((atomic_number, electronic, valence), axis=1)
    table = table / reference[None, :]
    table = np.concatenate(
        (np.zeros((1, 20)), table, np.zeros((1, 20))),
        axis=0,
    ).astype(np.float64)

    class FixedElectronicStructureFloat64(nn.Module):
        output_key: str | None = None

        @nn.compact
        def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
            conversion = jnp.asarray(table, dtype=jnp.float64)
            output = conversion[inputs["species"]]
            key = self.name if self.output_key is None else self.output_key
            if key is None:
                raise ValueError(
                    "FeNNix fixed species encoding requires an output key."
                )
            return {**inputs, key: output}

    layers = list(model.modules.layers)
    matching = [
        index
        for index, (layer_type, params) in enumerate(layers)
        if getattr(layer_type, "__module__", "") == "fennol.models.misc.encodings"
        and getattr(layer_type, "__name__", "") == "SpeciesEncoding"
        and params.get("name") == "species_encoding"
    ]
    if matching != [3]:
        raise ValueError(
            "Pinned Bio1M module topology drifted: expected one fixed species "
            "encoding at layer 3."
        )
    _, original_params = layers[3]
    if (
        original_params.get("encoding") != "electronic_structure"
        or original_params.get("trainable") is not False
        or original_params.get("zmax") != 86
    ):
        raise ValueError("Pinned Bio1M fixed species-encoding contract drifted.")
    layers[3] = (
        FixedElectronicStructureFloat64,
        {"name": "species_encoding"},
    )
    rebuilt_modules = type(model.modules)(tuple(layers))
    object.__setattr__(model, "modules", rebuilt_modules)
    object.__setattr__(model, "_FENNIX__apply", rebuilt_modules.apply)
    object.__setattr__(model, "_apply", jax.jit(rebuilt_modules.apply))
    model.set_energy_terms(model.energy_terms)
    normalized_table = np.ascontiguousarray(table.astype("<f8", copy=False))
    return hashlib.sha256(normalized_table.tobytes()).hexdigest()


class FeNNixAlchemicalKernel:
    """Bio1M scalar Hamiltonian exposing only conservative mechanics outputs."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        allow_unadmitted_gpu_mechanics: bool = False,
        alchemical_parameters: FeNNixAlchemicalParameters | None = None,
    ) -> None:
        parameters = (
            FeNNixAlchemicalParameters()
            if alchemical_parameters is None
            else alchemical_parameters
        )
        if not isinstance(parameters, FeNNixAlchemicalParameters):
            raise TypeError(
                "alchemical_parameters must be a FeNNixAlchemicalParameters."
            )
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FeNNix-Bio1 checkpoint not found: {checkpoint}")
        expected_sha = FENNIX_BIO1_CHECKPOINTS[FENNIX_HFE_MODEL_VARIANT]["sha256"]
        actual_sha = _sha256(checkpoint)
        if actual_sha != expected_sha:
            raise ValueError(
                "FeNNix native HFE mechanics require the exact pinned Bio1M "
                f"checkpoint; expected {expected_sha}, got {actual_sha}."
            )
        card = _load_card(FENNIX_BIO1_CARD)
        _validate_official_card(card)
        runtime_receipt = _validate_runtime_receipt(_official_runtime_validator())
        native_runtime_sources = _validate_native_runtime_sources()
        fennol_package_tree_sha256, fennol_package_tree_file_count = (
            _validate_fennol_distribution_tree()
        )

        jax, _ = _jax_modules()
        requested = str(device).strip().lower()
        platform = "gpu" if requested.split(":", 1)[0] in {"gpu", "cuda"} else "cpu"
        if requested.split(":", 1)[0] not in {"cpu", "gpu", "cuda"}:
            raise ValueError(
                "FeNNix native mechanics device must be cpu, gpu, or cuda."
            )
        if platform == "gpu" and not allow_unadmitted_gpu_mechanics:
            raise ValueError(
                "FeNNix GPU production admission is disabled pending the formal "
                "ten-functional-group experimental no-degradation panel; set "
                "allow_unadmitted_gpu_mechanics=True only for mechanics evidence."
            )
        package_versions = _validate_runtime_package_versions(
            _official_runtime_package_versions(platform),
            platform=platform,
        )
        raw_index = requested.partition(":")[2]
        try:
            index = int(raw_index) if raw_index else 0
        except ValueError as exc:
            raise ValueError("FeNNix JAX device index must be an integer.") from exc
        devices = jax.devices(platform)
        if index < 0 or index >= len(devices):
            raise ValueError(f"Requested FeNNix device {device!r} is unavailable.")
        self._device = devices[index]

        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError(
                "FeNNix deterministic checkpoint restoration requires one JAX CPU device."
            )
        # Restore and promote exactly once on CPU.  Loading separately under CPU
        # and GPU produced a real one-leaf subnormal divergence in Bio1M; host
        # restoration avoids both divergent runtime trees and fingerprint-only
        # masking.  The unchanged derived CPU tree is then device_put verbatim.
        with jax.default_device(cpu_devices[0]):
            model = _official_model_loader(str(checkpoint))
            if str(getattr(model, "energy_unit", "")).strip().lower() != "ev":
                raise ValueError(
                    "The pinned Bio1M Hamiltonian must report energy in eV."
                )
            fixed_encoding_sha256 = _promote_fixed_species_encoding_float64(model)
            host_variables, parameter_receipt = upcast_parameter_tree_float64(
                model.variables,
                original_checkpoint_sha256=actual_sha,
            )
            if parameter_receipt.original_floating_dtypes != ("float32",):
                raise ValueError(
                    "The pinned Bio1M checkpoint parameter floating dtype set "
                    "must be exactly float32 before deterministic in-memory upcast."
                )
            if (
                parameter_receipt.original_parameter_fingerprint
                != FENNIX_BIO1M_ORIGINAL_PARAMETER_TREE_FINGERPRINT
            ):
                raise ValueError(
                    "Pinned Bio1M original parameter-tree fingerprint mismatch."
                )
            if (
                parameter_receipt.derived_parameter_fingerprint
                != FENNIX_BIO1M_DERIVED_PARAMETER_TREE_FINGERPRINT
            ):
                raise ValueError(
                    "Pinned Bio1M derived float64 parameter-tree fingerprint mismatch."
                )
            if (
                fixed_encoding_sha256
                != FENNIX_BIO1M_FIXED_SPECIES_ENCODING_FLOAT64_SHA256
            ):
                raise ValueError(
                    "Pinned Bio1M fixed float64 species-encoding fingerprint mismatch."
                )
        variables = jax.device_put(host_variables, self._device)
        if (
            _parameter_tree_fingerprint(variables)
            != parameter_receipt.derived_parameter_fingerprint
        ):
            raise RuntimeError(
                "FeNNix device_put changed the deterministic float64 parameter tree."
            )
        self._jax = jax
        self._model = model
        self._variables = variables
        self.alchemical_parameters = parameters
        self._gradient_function = model.get_gradient_function(
            "coordinates",
            "cells",
            "alch_elambda",
            "alch_vlambda",
            jit=True,
            variables_as_input=True,
        )
        self._audited_input_signatures: set[tuple[str, tuple[tuple, ...]]] = set()
        self.identity = FeNNixKernelIdentity(
            checkpoint_sha256=actual_sha,
            source_revision=FENNIX_BIO1_SOURCE_REVISION,
            checkpoint_source_revision=FENNIX_BIO1_PMC_REVISION,
            runtime_distribution="FeNNol",
            runtime_version=FENNOL_DISTRIBUTION_VERSION,
            runtime_source_sha256=native_runtime_sources,
            runtime_package_versions=package_versions,
            fennol_package_tree_sha256=fennol_package_tree_sha256,
            fennol_package_tree_file_count=fennol_package_tree_file_count,
            original_parameter_tree_fingerprint=(
                parameter_receipt.original_parameter_fingerprint
            ),
            derived_parameter_tree_fingerprint=(
                parameter_receipt.derived_parameter_fingerprint
            ),
            jax_enable_x64=True,
            matmul_precision="highest",
            tf32_enabled=False,
            repulsion_nlh_coefficients_provenance=(
                FENNIX_REPULSION_NLH_COEFFICIENTS_PROVENANCE
            ),
            alchemical_parameters=parameters,
            fixed_species_encoding_float64_sha256=fixed_encoding_sha256,
            scientific_scope=FENNIX_KERNEL_SCIENTIFIC_SCOPE,
        )
        self.runtime_receipt = dict(runtime_receipt)

    def _raw_inputs(self, system: FeNNixAlchemicalSystem) -> dict[str, Any]:
        species = np.asarray(system.atomic_numbers, dtype=np.int32)
        solute_mask = np.zeros(len(species), dtype=bool)
        solute_mask[np.asarray(system.solute_atom_indices, dtype=np.int32)] = True
        alchemical_group = solute_mask.astype(np.int32)
        species_set, species_count = np.unique(species, return_counts=True)
        ligand_set, ligand_count = np.unique(species[solute_mask], return_counts=True)
        ligand_by_species = dict(zip(ligand_set.tolist(), ligand_count.tolist()))
        species_ligand_count = np.asarray(
            [ligand_by_species.get(int(value), 0) for value in species_set],
            dtype=np.int32,
        )
        cell = np.asarray(system.cell_angstrom, dtype=np.float64)[None, :, :]
        return {
            "species": species,
            "coordinates": np.asarray(system.coordinates_angstrom, dtype=np.float64),
            "natoms": np.asarray([len(species)], dtype=np.int32),
            "batch_index": np.zeros(len(species), dtype=np.int32),
            "cells": cell,
            "reciprocal_cells": np.linalg.inv(cell),
            "total_charge": np.asarray(0.0, dtype=np.float64),
            "alch_group": alchemical_group,
            "alch_ligand_charge": np.asarray(0.0, dtype=np.float64),
            "species_set": species_set.astype(np.int32),
            "species_count": species_count.astype(np.int32),
            "species_ligand_count": species_ligand_count,
            "recompute_species_index": True,
            "flags": {"recompute_species_index": None},
            "alch_elambda": np.asarray(0.0, dtype=np.float64),
            "alch_vlambda": np.asarray(0.0, dtype=np.float64),
            "alch_softcore_v": np.asarray(
                self.alchemical_parameters.graph_softcore_v_angstrom,
                dtype=np.float64,
            ),
            "alch_softcore_rep": np.asarray(
                self.alchemical_parameters.repulsion_softcore_angstrom,
                dtype=np.float64,
            ),
            "alch_m": np.asarray(
                self.alchemical_parameters.repulsion_power_m,
                dtype=np.int32,
            ),
        }

    def _preprocess(self, raw: dict[str, Any]) -> Any:
        jax = self._jax
        _, jnp = _jax_modules()
        preprocessing = getattr(self._model, "preprocessing", None)
        if preprocessing is None or not hasattr(preprocessing, "process"):
            raise TypeError("Pinned FeNNix model lacks native preprocessing.process.")
        # The CPU result is discarded: it estimates static neighbor-list capacity
        # only.  All Hamiltonian inputs below are rebuilt in float64 by native JAX.
        state, _discarded_capacity_output = preprocessing.init_with_output(raw)

        def to_jax(leaf: Any) -> Any:
            if isinstance(leaf, np.ndarray):
                if np.issubdtype(leaf.dtype, np.floating):
                    return jnp.asarray(leaf, dtype=jnp.float64)
                return jnp.asarray(leaf)
            return leaf

        jax_raw = jax.tree_util.tree_map(to_jax, raw)
        processed = preprocessing.process(state, jax_raw)
        if _contains_overflow(processed):
            raise RuntimeError(
                "FeNNix native neighbor-list capacity overflowed; the mechanics "
                "evaluation is invalid and must be restarted with fixed larger capacity."
            )
        _require_float64_tree(processed, label="processed Hamiltonian input")
        return processed

    def evaluate(
        self,
        system: FeNNixAlchemicalSystem,
        lambda_state_or_progress: FeNNixLambdaState | float,
    ) -> FeNNixKernelResult:
        """Evaluate E, F, cell virial, and lambda derivatives from one scalar E."""

        normalized = validate_fennix_alchemical_system(system)
        state = (
            lambda_state_or_progress
            if isinstance(lambda_state_or_progress, FeNNixLambdaState)
            else FeNNixLambdaState.from_progress(lambda_state_or_progress)
        )
        canonical = FeNNixLambdaState.from_progress(state.progress)
        if state != canonical:
            raise ValueError(
                "FeNNixLambdaState must equal the canonical paper progress mapping."
            )

        jax = self._jax
        _, jnp = _jax_modules()
        with jax.default_device(self._device):
            inputs = self._preprocess(self._raw_inputs(normalized))
            inputs = {
                **inputs,
                "alch_elambda": jnp.asarray(state.lambda_e, dtype=jnp.float64),
                "alch_vlambda": jnp.asarray(state.lambda_v, dtype=jnp.float64),
            }
            signature = _tree_shape_dtype_signature(jax, inputs)
            if signature not in self._audited_input_signatures:
                _audit_gradient_jaxpr_float64(
                    jax,
                    self._gradient_function,
                    self._variables,
                    inputs,
                )
                self._audited_input_signatures.add(signature)
            energy, derivatives, _outputs = self._gradient_function(
                self._variables, inputs
            )
            energy = jnp.asarray(energy).sum()
            coordinate_gradient = jnp.asarray(derivatives["coordinates"])
            cell_gradient = jnp.asarray(derivatives["cells"])[0]
            d_lambda_e = jnp.asarray(derivatives["alch_elambda"]).sum()
            d_lambda_v = jnp.asarray(derivatives["alch_vlambda"]).sum()
            forces = -coordinate_gradient
            # FeNNol's official right-strain convention scales both row-vector
            # coordinates and row-vector cells: R' = R @ S, h' = h @ S.
            # Therefore dE/dS = R.T @ dE/dR + h.T @ dE/dh (positive FeNNol
            # virial / ASE-stress convention), all from this same scalar E.
            cell = jnp.asarray(inputs["cells"])[0]
            coordinates = jnp.asarray(inputs["coordinates"])
            virial = coordinates.T @ coordinate_gradient + cell.T @ cell_gradient
            d_progress = jnp.asarray(
                state.chain_derivative(
                    float(d_lambda_e),
                    float(d_lambda_v),
                ),
                dtype=jnp.float64,
            )
            values = (
                energy,
                forces,
                cell_gradient,
                virial,
                d_lambda_e,
                d_lambda_v,
                d_progress,
            )
            jax.block_until_ready(values)
        _require_float64_tree(values, label="kernel output")
        arrays = [np.asarray(value, dtype=np.float64) for value in values]
        if not all(np.isfinite(value).all() for value in arrays):
            raise FloatingPointError(
                "FeNNix native kernel returned non-finite mechanics."
            )
        energy_np, forces_np, cell_gradient_np, virial_np, dle_np, dlv_np, dp_np = (
            arrays
        )
        if forces_np.shape != (len(normalized.atomic_numbers), 3):
            raise ValueError("FeNNix force output must have shape (N, 3).")
        if cell_gradient_np.shape != (3, 3) or virial_np.shape != (3, 3):
            raise ValueError("FeNNix cell-gradient and virial outputs must be 3 x 3.")

        def rows3(array: np.ndarray) -> tuple[tuple[float, float, float], ...]:
            return tuple((float(row[0]), float(row[1]), float(row[2])) for row in array)

        return FeNNixKernelResult(
            lambda_state=state,
            identity=self.identity,
            energy_ev=float(energy_np),
            forces_ev_per_angstrom=rows3(forces_np),
            cell_gradient_ev_per_angstrom=rows3(cell_gradient_np),
            virial_ev=rows3(virial_np),
            denergy_dlambda_e_ev=float(dle_np),
            denergy_dlambda_v_ev=float(dlv_np),
            denergy_dprogress_ev=float(dp_np),
            alchemical_parameters=self.alchemical_parameters,
            scope=FENNIX_KERNEL_SCIENTIFIC_SCOPE,
        )
