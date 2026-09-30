"""Audited FAIR-Chem UMA derivative and checkpoint-lineage helpers.

This module contains no solvent physics.  It isolates two UMA-specific facts:
the legacy checkpoint schema translation required by FAIR-Chem 2.19 and the
operation-scoped graph-retention flag needed for a direct HVP.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch

from fairchem.core.datasets.atomic_data import warn_if_upcasting
from fairchem.core.models.uma.escn_md import MLP_EFS_Head

COMPAT_TRANSFORM_VERSION = "maple-compat-v1-fairchem-2.19"
_AUDITED_EFS_HEAD_MODULE = "fairchem.core.models.uma.escn_md"
_AUDITED_EFS_HEAD_NAME = "MLP_EFS_Head"
_AUDITED_FAIRCHEM_VERSION = "2.19.0"
_AUDITED_EFS_FORWARD_SHA256 = (
    "beaf1fd8189005bde93dd2c696867d20fc9232c96b4045b0777e2ab4275fdec6"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_value(item) for item in value), key=repr)
    if isinstance(value, torch.dtype):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _config_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    before = _json_value(before)
    after = _json_value(after)
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else key
            if key not in before:
                changes.append({"path": child, "change": "added", "after": after[key]})
            elif key not in after:
                changes.append(
                    {"path": child, "change": "removed", "before": before[key]}
                )
            else:
                changes.extend(_config_diff(before[key], after[key], child))
        return changes
    if before != after:
        return [{"path": path, "change": "changed", "before": before, "after": after}]
    return []


def _iter_tensors(value: Any, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(value[key], child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}[{index}]")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _tensor_payload(checkpoint: Any) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256(b"maple-uma-tensor-payload-v1\0")
    inventory: list[dict[str, Any]] = []
    for root_name in ("model_state_dict", "ema_state_dict"):
        root = getattr(checkpoint, root_name, None)
        if root is None:
            continue
        for name, tensor in _iter_tensors(root, root_name):
            payload = _tensor_bytes(tensor)
            tensor_sha = hashlib.sha256(payload).hexdigest()
            record = {
                "key": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha,
            }
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            inventory.append(record)
    if not inventory:
        raise ValueError("UMA checkpoint contains no model or EMA tensor payload.")
    return digest.hexdigest(), inventory


def _transform_model_config(model_config: Any) -> Any:
    transformed = copy.deepcopy(model_config)
    transformed.pop("model_id", None)
    transformed.pop("supports_single_atoms", None)

    backbone = transformed.get("backbone", {})
    backbone.pop("charge_balanced_channels", None)
    backbone.pop("composition_dropout", None)
    dataset_mapping = backbone.pop("dataset_mapping", None)
    if dataset_mapping is not None and "dataset_list" not in backbone:
        backbone["dataset_list"] = list(dataset_mapping)

    for head_config in transformed.get("heads", {}).values():
        head_mapping = head_config.pop("dataset_mapping", None)
        if head_mapping is not None and "dataset_names" not in head_config:
            head_config["dataset_names"] = list(head_mapping)
    return transformed


def prepare_compat_checkpoint(
    checkpoint_name: str,
    checkpoint_path: str,
    *,
    cache_dir: str | os.PathLike[str],
) -> str:
    """Create a content-addressed FAIR-Chem compatibility checkpoint and receipt."""

    source = Path(checkpoint_path).expanduser().resolve(strict=True)
    source_stat = source.stat()
    source_sha = _sha256_file(source)
    compat_dir = Path(cache_dir) / "maple_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(checkpoint_name).name
    output = compat_dir / (f"{safe_name}-{COMPAT_TRANSFORM_VERSION}-{source_sha}.pt")
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")

    if output.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("transform_version") == COMPAT_TRANSFORM_VERSION
            and receipt.get("source", {}).get("path") == str(source)
            and receipt.get("source", {}).get("sha256") == source_sha
            and receipt.get("output", {}).get("path") == str(output.resolve())
            and receipt.get("output", {}).get("sha256") == _sha256_file(output)
            and receipt.get("tensor_payload", {}).get("exactly_equal") is True
        ):
            return str(output)

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not hasattr(checkpoint, "model_config"):
        raise TypeError("UMA checkpoint is missing model_config.")
    config_before = copy.deepcopy(checkpoint.model_config)
    before_sha, before_inventory = _tensor_payload(checkpoint)
    checkpoint.model_config = _transform_model_config(checkpoint.model_config)
    config_after = copy.deepcopy(checkpoint.model_config)

    tmp_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, tmp_output)
        os.replace(tmp_output, output)
    finally:
        tmp_output.unlink(missing_ok=True)

    del checkpoint
    gc.collect()
    written = torch.load(output, map_location="cpu", weights_only=False)
    after_sha, after_inventory = _tensor_payload(written)
    written_config = copy.deepcopy(written.model_config)
    del written
    gc.collect()

    if before_inventory != after_inventory or before_sha != after_sha:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "UMA compatibility transform changed checkpoint tensor payload."
        )
    if _json_value(config_after) != _json_value(written_config):
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "UMA compatibility checkpoint config changed during serialization."
        )
    if source.stat() != source_stat or _sha256_file(source) != source_sha:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "UMA source checkpoint changed during compatibility transformation."
        )

    receipt = {
        "schema_version": 1,
        "transform_version": COMPAT_TRANSFORM_VERSION,
        "source": {
            "path": str(source),
            "sha256": source_sha,
            "size_bytes": source_stat.st_size,
        },
        "output": {
            "path": str(output.resolve()),
            "sha256": _sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        "config_before": _json_value(config_before),
        "config_after": _json_value(config_after),
        "config_diff": _config_diff(config_before, config_after),
        "tensor_payload": {
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "exactly_equal": True,
            "inventory": before_inventory,
        },
    }
    tmp_receipt = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    try:
        tmp_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_receipt, receipt_path)
    finally:
        tmp_receipt.unlink(missing_ok=True)
    return str(output)


@contextmanager
def efs_head_graph_context(heads, *, backbone):
    """Enable only the audited EFS head's create-graph switch, then restore it.

    FAIR-Chem 2.19's UMA EFS head uses its own ``training`` boolean solely as
    the ``create_graph`` argument for energy-gradient forces.  Calling
    ``head.train()`` would recursively alter child modules, so this context sets
    the single audited boolean directly and leaves the backbone and all child
    modules in evaluation mode.
    """

    selected = list(heads)
    if not selected:
        raise RuntimeError("Direct UMA HVP found no FAIR-Chem UMA EFS head.")
    if bool(getattr(backbone, "training", True)):
        raise RuntimeError(
            "Direct UMA HVP requires the backbone to remain in eval mode."
        )
    for head in selected:
        head_type = type(head)
        if (
            head_type.__module__ != _AUDITED_EFS_HEAD_MODULE
            or head_type.__name__ != _AUDITED_EFS_HEAD_NAME
        ):
            raise RuntimeError(
                "Direct UMA HVP requires the audited FAIR-Chem UMA EFS head."
            )
        if bool(getattr(head.regress_config, "hessian", False)):
            raise RuntimeError(
                "Direct UMA HVP forbids Hessian-enabled FAIR-Chem heads."
            )

    previous = [bool(head.training) for head in selected]
    try:
        for head in selected:
            head.training = True
        yield
    finally:
        for head, state in zip(selected, previous, strict=True):
            head.training = state


def _require_audited_fairchem_source() -> None:
    version = importlib.metadata.version("fairchem-core")
    source_sha = hashlib.sha256(
        inspect.getsource(MLP_EFS_Head.forward).encode("utf-8")
    ).hexdigest()
    if (
        version != _AUDITED_FAIRCHEM_VERSION
        or source_sha != _AUDITED_EFS_FORWARD_SHA256
    ):
        raise RuntimeError(
            "Direct UMA HVP is source-bound to FAIR-Chem 2.19.0's audited EFS head."
        )


def predict_direct_hvp(predictor, data, *, task_name: str, direction):
    """Run one Hessian-disabled UMA forward and differentiate ``F dot n``.

    The data preparation is deliberately kept byte-for-byte equivalent in
    ordering to FAIR-Chem 2.19 ``MLIPPredictUnit.predict``.  Unlike that public
    method, this helper retains the cloned position tensor so the directional
    second derivative can be taken without constructing a dense Hessian.
    """

    _require_audited_fairchem_source()
    settings = predictor.inference_settings
    if settings.tf32:
        raise RuntimeError("Analytic UMA HVP forbids TF32 inference.")
    if settings.predict_untrained_hessian:
        raise RuntimeError("Direct UMA HVP requires a Hessian-disabled predictor.")
    if predictor.direct_forces:
        raise RuntimeError(
            "Direct-force UMA heads do not define the admitted energy Hessian."
        )
    if not predictor.lazy_model_intialized:
        predictor._lazy_init(data)

    data_device = data.to(predictor.device).clone()
    dtype = settings.base_precision_dtype
    if not predictor._warned_upcast:
        predictor._warned_upcast = warn_if_upcasting(data_device.pos.dtype, dtype)
    for key, value in data_device:
        if torch.is_tensor(value) and value.is_floating_point():
            data_device[key] = value.to(dtype)
    predictor.model.module.on_predict_check(data_device)

    model = predictor.model.module
    heads = []
    for outer_head in model.output_heads.values():
        for head in outer_head.modules():
            if isinstance(head, MLP_EFS_Head):
                heads.append(head)
    with efs_head_graph_context(heads, backbone=model.backbone), nullcontext():
        output = predictor.model(data_device)
        processed = predictor._process_outputs(data_device, output, True)

    properties: dict[str, torch.Tensor] = {}
    for task in predictor.dataset_to_tasks[task_name]:
        properties[task.property] = processed[task.name]
    if "energy" not in properties or "forces" not in properties:
        raise RuntimeError("UMA direct HVP requires energy and gradient-force tasks.")
    forces = properties["forces"]
    energy = properties["energy"]
    requested = torch.as_tensor(direction, dtype=forces.dtype, device=forces.device)
    if requested.shape != forces.shape:
        raise ValueError(
            f"UMA HVP direction must have shape {tuple(forces.shape)}, got {tuple(requested.shape)}."
        )
    (force_direction_gradient,) = torch.autograd.grad(
        (forces * requested).sum(),
        data_device.pos,
        create_graph=False,
        retain_graph=False,
    )
    return {
        "energy": energy,
        "forces": forces,
        "hvp": -force_direction_gradient,
        "position_dtype": str(data_device.pos.dtype),
    }


__all__ = [
    "COMPAT_TRANSFORM_VERSION",
    "efs_head_graph_context",
    "predict_direct_hvp",
    "prepare_compat_checkpoint",
]
