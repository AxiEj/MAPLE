"""Private isolated inference worker for :mod:`solpropmix_property`.

This module is copied byte-for-byte into a private runtime directory.  It has no
MAPLE imports so the child can execute only the materialized upstream snapshot.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, TypedDict, cast


class SolventPayload(TypedDict):
    smiles: str
    fraction: float


class RequestPayload(TypedDict):
    solute: str
    solvents: list[SolventPayload]
    temperature: float
    device: str
    precision: str


class ModelPredictionPayload(TypedDict):
    model_index: int
    g298: float
    h298: float
    g_temperature: float


STATIC_HASHES = {
    "solvation_predictor/data/__init__.py": "2639b97ff1fe63a3c9e88402aa9d1369305be046762483db834325da22948a78",
    "solvation_predictor/data/data.py": "fd7bc4454b981686b3ef7fbb86d5069f9cb5eb7b4bbc454e3876ac61775aa83e",
    "solvation_predictor/data/Scaler.py": "5e4a65084db838238f14386af66672184d93764764b694b138fa97cf13dffa73",
    "solvation_predictor/data/Splitter.py": "f5244d923b5afec61e22d6d9ded1bef081545415b9b0efefe7282d8ab5e7d1ac",
}
CHECKPOINT_HASHES = (
    "e887f9d424134250eb6ae871e2d142f5f5e04a6a9b2e41c37e2d96935f30ffd5",
    "c6f005771274988e6c8b4c28f12d6172d2514fd1bbe543031871f79ab60ab02d",
    "44277a9bbdd06122c039c58d051b19b5f8ea4d9b6fcebae5c6408d903883fce2",
    "ca7cd715b235c0c69366dda9c1d06aac837a6cfdb959dee4db8a8d0366407edb",
    "45d2a83015c4965adb982e8a6f07d48899b1d45050d375aa9e77bb9f46856fcf",
    "5881eddaef2c22b167ed0de043743a917b1a6fe852f642443b08ba41552f10c1",
    "a571f5e48f52f61a028c82fbd5c7857133c719f808892483c9b27b0ff17b4b3c",
    "47256bfc9bf66d9cca38c378aca3a307317b891580979f6d18694abf1e9ae853",
    "4d6db1fc5099708fba40df71e2517be8ebf9816caee2ad38f099c2a4d178f97b",
    "0decb8b662177edae77fa66007ba5f9ad1abfe3e6405ffbb35110dc837a1660b",
)
SOURCE_REVISION = "80043ce09eb8802517c35b59254f8e9c181f2dac"
SOURCE_TREE = "bc7b933d6cf4c55c8ad10236383cfd09afbbfdc2"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _numpy_reconstruct() -> Any:
    """Resolve NumPy's private pickle helper on both 1.x and 2.x runtimes."""

    try:
        module = importlib.import_module("numpy._core.multiarray")
    except ModuleNotFoundError:
        module = importlib.import_module("numpy.core.multiarray")
    return module._reconstruct


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object.")
    return cast(dict[str, object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _request(value: object) -> RequestPayload:
    payload = _mapping(value, "request")
    raw_solvents = payload.get("solvents")
    if not isinstance(raw_solvents, list):
        raise TypeError("request.solvents must be a list.")
    solvents: list[SolventPayload] = []
    for index, raw in enumerate(raw_solvents):
        item = _mapping(raw, f"request.solvents[{index}]")
        solvents.append(
            {
                "smiles": _string(item.get("smiles"), "solvent smiles"),
                "fraction": _number(item.get("fraction"), "solvent fraction"),
            }
        )
    return {
        "solute": _string(payload.get("solute"), "request.solute"),
        "solvents": solvents,
        "temperature": _number(payload.get("temperature"), "request.temperature"),
        "device": _string(payload.get("device"), "request.device"),
        "precision": _string(payload.get("precision"), "request.precision"),
    }


def _read_identity(path: Path) -> tuple[str, str]:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "source identity")
    revision = _string(payload.get("revision"), "source revision")
    tree = _string(payload.get("tree"), "source tree")
    if revision != SOURCE_REVISION or tree != SOURCE_TREE:
        raise RuntimeError("Private SolProp-mix source identity mismatch.")
    return revision, tree


def _configure_torch(request: RequestPayload) -> tuple[Any, Any]:
    import torch

    if request["device"] == "cuda:0":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError(
                "Requested cuda:0 is unavailable; CPU fallback is forbidden."
            )
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        if torch.cuda.current_device() != 0:
            raise RuntimeError("SolProp-mix did not select requested cuda:0.")
    elif request["device"] == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("Unsupported worker device.")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for backend, name in (
        (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"),
        (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"),
    ):
        if hasattr(backend, name):
            setattr(backend, name, False)
    dtype = (
        torch.float64 if request["precision"] == "float64_promoted" else torch.float32
    )
    torch.set_default_dtype(dtype)
    return torch, (device, dtype)


def run(runtime_root: Path, request: RequestPayload) -> dict[str, object]:
    import numpy as np
    from rdkit import rdBase

    torch, (device, dtype) = _configure_torch(request)
    source_root = runtime_root / "source"
    weights_root = runtime_root / "weights"
    revision, source_tree = _read_identity(runtime_root / "source_identity.json")
    worker_hash = _sha256(Path(__file__).read_bytes())

    observed_static: dict[str, str] = {}
    for relative, expected in STATIC_HASHES.items():
        data = (source_root / relative).read_bytes()
        observed = _sha256(data)
        if observed != expected:
            raise RuntimeError(
                "Pinned supplemental SolProp-mix module identity mismatch."
            )
        observed_static[relative] = observed

    sys.path.insert(0, str(source_root))
    data_module = importlib.import_module("solvation_predictor.data.data")
    scaler_module = importlib.import_module("solvation_predictor.data.Scaler")
    input_module = importlib.import_module("solvation_predictor.inp")
    model_module = importlib.import_module("solvation_predictor.models.Model")
    evaluate_module = cast(
        Any, importlib.import_module("solvation_predictor.train.evaluate")
    )
    DataPoint = data_module.DataPoint
    DatapointList = data_module.DatapointList
    DataTensor = data_module.DataTensor
    MolencoderDatabase = data_module.MolencoderDatabase
    Scaler = scaler_module.Scaler
    TrainArgs = input_module.TrainArgs
    Model = model_module.Model

    class PrecisionDataTensor(DataTensor):  # type: ignore[misc, valid-type]
        def make_tensor(self) -> None:
            super().make_tensor()
            self.f_atoms = self.f_atoms.to(device=self.device, dtype=dtype)
            self.f_bonds = self.f_bonds.to(device=self.device, dtype=dtype)
            self.f_mols = self.f_mols.to(device=self.device, dtype=dtype)

    evaluate_module.DataTensor = PrecisionDataTensor
    reconstruct = _numpy_reconstruct()
    safe_globals = [
        TrainArgs,
        (np.dtype, "numpy.dtype"),
        (np.ndarray, "numpy.ndarray"),
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        type(np.dtype("float64")),
    ]

    predictions: list[ModelPredictionPayload] = []
    observed_checkpoints: list[str] = []
    for model_index, expected_hash in enumerate(CHECKPOINT_HASHES):
        checkpoint_bytes = (weights_root / f"model{model_index}.pt").read_bytes()
        observed_hash = _sha256(checkpoint_bytes)
        if observed_hash != expected_hash:
            raise RuntimeError("Pinned SolProp-mix checkpoint identity mismatch.")
        observed_checkpoints.append(observed_hash)
        with torch.serialization.safe_globals(safe_globals):
            state = torch.load(
                io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
            )
        expected_schema = {
            "input",
            "state_dict",
            "data_scaler",
            "features_scaler",
            "scale_features",
            "use_same_scaler_for_features",
        }
        if set(state) != expected_schema:
            raise RuntimeError("SolProp-mix checkpoint has unexpected schema.")
        args = state["input"]
        args.max_num_mols = len(request["solvents"]) + 1
        args.num_targets = 2
        args.solute = True
        args.cuda = device.type == "cuda"
        args.device = device
        args.batch_size = 1
        args.max_molecules = 1
        model = Model(args)
        model.load_state_dict(state["state_dict"], strict=True)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        actual_device = next(model.parameters()).device
        if actual_device.type != device.type or (
            device.type == "cuda" and actual_device.index != 0
        ):
            raise RuntimeError("SolProp-mix model is not on requested device.")

        scaler = Scaler(
            mean=np.asarray(state["data_scaler"]["means"], dtype=np.float64),
            std=np.asarray(state["data_scaler"]["stds"], dtype=np.float64),
            mean_f=np.asarray(state["features_scaler"]["means"], dtype=np.float64),
            std_f=np.asarray(state["features_scaler"]["stds"], dtype=np.float64),
            scale_features=bool(state["scale_features"]),
        )
        structures = [request["solute"]] + [
            item["smiles"] for item in request["solvents"]
        ]
        fractions = [item["fraction"] for item in request["solvents"]]
        point = DataPoint(
            structures, [0.0, 0.0], [], fractions, args, MolencoderDatabase()
        )
        data = DatapointList([point])
        scaler.transform_standard(data)
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device_type, enabled=False),
        ):
            values = evaluate_module.predict(
                model=model, data=data, scaler=scaler, inp=args
            )
        if len(values) != 1 or len(values[0]) != 2:
            raise RuntimeError("SolProp-mix returned a non-(G,H) prediction.")
        g298, h298 = (float(values[0][0]), float(values[0][1]))
        temperature = request["temperature"]
        g_temperature = temperature * (
            g298 / 298.15 - h298 * (1.0 / 298.15 - 1.0 / temperature)
        )
        if not all(math.isfinite(value) for value in (g298, h298, g_temperature)):
            raise RuntimeError("SolProp-mix returned a non-finite prediction.")
        predictions.append(
            {
                "model_index": model_index,
                "g298": g298,
                "h298": h298,
                "g_temperature": g_temperature,
            }
        )

    values = [float(item["g_temperature"]) for item in predictions]
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
    gpu: dict[str, object] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": str(properties.name),
            "uuid": str(properties.uuid),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
        }
    return {
        "solute": request["solute"],
        "solvents": request["solvents"],
        "temperature": request["temperature"],
        "device": "cuda:0" if device.type == "cuda" else "cpu",
        "precision": request["precision"],
        "dtype": str(dtype).removeprefix("torch."),
        "source_revision": revision,
        "source_tree": source_tree,
        "worker_sha256": worker_hash,
        "checkpoint_sha256": observed_checkpoints,
        "static_sha256": observed_static,
        "predictions": predictions,
        "mean": mean,
        "std": std,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "amp": False,
            "fp16_reduced": bool(
                getattr(
                    torch.backends.cuda.matmul,
                    "allow_fp16_reduced_precision_reduction",
                    False,
                )
            ),
            "bf16_reduced": bool(
                getattr(
                    torch.backends.cuda.matmul,
                    "allow_bf16_reduced_precision_reduction",
                    False,
                )
            ),
            "gpu": gpu,
        },
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        raise SystemExit("usage: _solpropmix_worker.py RUNTIME_ROOT REQUEST_JSON")
    runtime_root = Path(arguments[0]).resolve()
    request = _request(json.loads(Path(arguments[1]).read_text(encoding="utf-8")))
    print(json.dumps(run(runtime_root, request), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
