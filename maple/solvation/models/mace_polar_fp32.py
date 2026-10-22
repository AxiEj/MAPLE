"""Release-bound FP32 zero-field adapter for official MACE-POLAR-1-M.

The adapter uses the upstream gas-only FP32 construction path without changing
the checkpoint or forcing every upstream operation to FP32.  In particular,
the upstream model's explicit FP64 reductions remain untouched.  A strict
process-default guard is needed because graph-longrange 0.4.0 creates several
buffers from ``torch.get_default_dtype()``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from maple.solvation.models.base import ModelDomain
from maple.solvation.models.mace_polar import (
    OFFICIAL_MACE_POLAR_1_M_CONTRACT,
    _calculator_source_files,
)

MACE_POLAR_FP32_POLICY_ID = "macepolar-official-fp32-serialized-process-v1"
MACE_POLAR_FP32_PROVIDER_ID = "maple.macepolar-official-fp32.experimental-v1"
MACE_POLAR_FP32_MODEL_PROFILE_ID = "macepolar-1m-zero-field-fp32-v1"
_GRAPH_LONGRANGE_REALSPACE_SHA256 = (
    "2cf7e098f4960490e6e9a5868ca1ea01ca23beb56ff8ce2cd0a5fe8615aec268"
)
_DEFAULT_CHECKPOINT = Path("~/.cache/mace/MACE-POLAR-1-M.model")
_FACTORY_TOKEN = object()
_TEST_TOKEN = object()


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_process_float32(boundary: str) -> None:
    import torch

    if torch.get_default_dtype() is not torch.float32:
        raise RuntimeError(
            f"FP32 MACE-POLAR requires torch.float32 {boundary}; received "
            f"{torch.get_default_dtype()}. The adapter does not mutate the "
            "process-global default dtype."
        )


def _verify_runtime() -> dict[str, str]:
    contract = OFFICIAL_MACE_POLAR_1_M_CONTRACT
    mace_version = importlib.metadata.version("mace-torch")
    graph_version = importlib.metadata.version("graph-longrange")
    if mace_version != contract.mace_torch_version:
        raise RuntimeError(
            f"FP32 MACE-POLAR requires mace-torch=={contract.mace_torch_version}; "
            f"received {mace_version}."
        )
    if graph_version != contract.graph_longrange_version:
        raise RuntimeError(
            "FP32 MACE-POLAR requires graph-longrange=="
            f"{contract.graph_longrange_version}; received {graph_version}."
        )
    import graph_longrange

    source = (
        Path(graph_longrange.__file__).resolve().parent / "realspace_electrostatics.py"
    )
    if not source.is_file():
        raise RuntimeError("graph-longrange realspace source is unavailable.")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha256 != _GRAPH_LONGRANGE_REALSPACE_SHA256:
        raise RuntimeError("graph-longrange realspace source bytes drifted.")
    return {
        "mace_torch_version": mace_version,
        "graph_longrange_version": graph_version,
        "graph_longrange_realspace_path": "graph_longrange/realspace_electrostatics.py",
        "graph_longrange_realspace_sha256": source_sha256,
    }


def _resolve_checkpoint(checkpoint_path: str | Path | None) -> Path:
    contract = OFFICIAL_MACE_POLAR_1_M_CONTRACT
    path = Path(checkpoint_path or _DEFAULT_CHECKPOINT).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Official MACE-POLAR checkpoint is unavailable: {path}.")
    if (
        path.stat().st_size != contract.checkpoint_size_bytes
        or _sha256_file(path) != contract.checkpoint_sha256
    ):
        raise RuntimeError(
            "Checkpoint does not match the official MACE-POLAR-1-M size/SHA-256."
        )
    return path


def _prepare_cuda_device(device: str) -> None:
    """Install the CUDA policy before the first CUDA runtime query or model load."""

    if device == "cpu":
        return
    from maple.solvation.experimental.cuda_execution import prepare_cuda_execution

    prepare_cuda_execution()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"FP32 MACE-POLAR CUDA device is unavailable: {device}.")
    index = int(device.removeprefix("cuda:"))
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(
            f"FP32 MACE-POLAR requested {device}, but only {count} CUDA device(s) "
            "are visible."
        )


class _FP32GraphBase:
    """Minimal release-bound base consumed by the existing graph adapter."""

    def __init__(self, calculator: object, checkpoint: Path) -> None:
        import torch

        contract = OFFICIAL_MACE_POLAR_1_M_CONTRACT
        model = getattr(calculator, "model", None)
        parameters = tuple(model.parameters()) if model is not None else ()
        if not parameters or any(
            parameter.dtype is not torch.float32 for parameter in parameters
        ):
            raise RuntimeError("Official MACE-POLAR model parameters must be float32.")
        if getattr(calculator, "dtype", None) is not torch.float32:
            raise RuntimeError("Official MACE-POLAR calculator dtype must be float32.")
        numbers = tuple(sorted(set(int(z) for z in calculator.atomic_numbers)))
        if not numbers:
            raise RuntimeError("Official MACE-POLAR atomic-number domain is missing.")
        stat = checkpoint.stat()
        self._calculator = calculator
        self._checkpoint = checkpoint
        self._checkpoint_stat = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        )
        self.device = str(calculator.device)
        self.dtype = torch.float32
        self.provider_id = MACE_POLAR_FP32_PROVIDER_ID
        self.model_profile_id = MACE_POLAR_FP32_MODEL_PROFILE_ID
        self.domain = ModelDomain(numbers, None, (1,))
        self.release_contract = contract
        self.provenance = MappingProxyType(
            {
                "checkpoint_sha256": contract.checkpoint_sha256,
                "checkpoint_size_bytes": contract.checkpoint_size_bytes,
                "dtype": "float32",
                "device": self.device,
                "domain": self.domain.metadata(),
            }
        )
        self.provenance_sha256 = _sha(dict(self.provenance))
        self._configuration = self._current_configuration()

    def _current_configuration(self) -> str:
        stat = self._checkpoint.stat()
        current_stat = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if current_stat != self._checkpoint_stat:
            raise RuntimeError("MACE-POLAR checkpoint file identity drifted.")
        return _sha(
            {
                "schema": "route2-macepolar-fp32-graph-base-v1",
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "provenance_sha256": self.provenance_sha256,
                "calculator_source_files_sha256": list(
                    _calculator_source_files(self._calculator)
                ),
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration:
            raise RuntimeError("FP32 MACE-POLAR graph-base configuration drifted.")
        return current


def _build_graph(*, device: str, checkpoint: Path):
    from maple.solvation.models.mace_polar_torch import (
        MACEPolarTorchGraphAdapter,
        _TorchModelOnlyMACEPolCalculator,
    )

    calculator = _TorchModelOnlyMACEPolCalculator(
        device=device,
        model="macepolm",
        model_path=str(checkpoint),
        implicit="none",
        solvent="none",
    )
    return MACEPolarTorchGraphAdapter(_FP32GraphBase(calculator, checkpoint))


class MACEPolarFP32Adapter:
    """Immutable official-checkpoint FP32 energy/source graph boundary."""

    __slots__ = (
        "_configuration",
        "_graph",
        "_metadata",
        "_sealed",
        "_testing",
        "device",
        "domain",
        "dtype",
        "model_profile_id",
        "provider_id",
        "release_contract",
    )

    def __init__(
        self,
        graph: object,
        *,
        runtime: Mapping[str, str],
        _factory_token: object | None = None,
        _testing_token: object | None = None,
    ) -> None:
        import torch

        testing = _testing_token is _TEST_TOKEN
        if _testing_token is not None and not testing:
            raise ValueError("Invalid FP32 engineering-test token.")
        if not testing and _factory_token is not _FACTORY_TOKEN:
            raise ValueError("FP32 production identity requires its official factory.")
        required = (
            "configuration_sha256",
            "energy_source_torch",
            "metadata",
            "topology_diagnostics",
        )
        if any(not callable(getattr(graph, name, None)) for name in required):
            raise TypeError("FP32 adapter requires the coordinate-connected graph API.")
        graph_dtype = str(getattr(graph, "dtype", None)).removeprefix("torch.")
        if graph_dtype != "float32":
            raise TypeError("FP32 adapter requires a float32 model graph.")
        provider = (
            "unregistered-engineering-test-macepolar-fp32"
            if testing
            else MACE_POLAR_FP32_PROVIDER_ID
        )
        profile = (
            "unregistered-engineering-test-macepolar-fp32-profile"
            if testing
            else MACE_POLAR_FP32_MODEL_PROFILE_ID
        )
        release = getattr(graph, "release_contract", None)
        metadata = {
            "schema": "route2-macepolar-fp32-model-metadata-v1",
            "provider_id": provider,
            "model_profile_id": profile,
            "fp32_policy_id": MACE_POLAR_FP32_POLICY_ID,
            "model_dtype": "torch.float32",
            "input_dtype": "torch.float32",
            "output_dtypes": ("torch.float32", "torch.float32"),
            "process_default_dtype": "torch.float32",
            "library_sets_default_dtype": False,
            "all_operations_forced_to_fp32": False,
            "upstream_explicit_fp64_reductions_preserved": True,
            "checkpoint_sha256": getattr(release, "checkpoint_sha256", None),
            "base_configuration_sha256": graph.configuration_sha256(),
            "base_graph_metadata": copy.deepcopy(dict(graph.metadata())),
            **dict(runtime),
            "engineering_test_graph": testing,
        }
        configuration = _sha(
            {**metadata, "implementation_sha256": _implementation_sha256()}
        )
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "_configuration", configuration)
        object.__setattr__(self, "_testing", testing)
        object.__setattr__(self, "device", str(graph.device))
        object.__setattr__(self, "dtype", torch.float32)
        object.__setattr__(self, "provider_id", provider)
        object.__setattr__(self, "model_profile_id", profile)
        object.__setattr__(self, "domain", graph.domain)
        object.__setattr__(self, "release_contract", release)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarFP32Adapter is immutable.")
        object.__setattr__(self, name, value)

    @classmethod
    def _from_graph_for_testing(cls, graph: object):
        return cls(
            graph,
            runtime={"runtime_verification": "test-double"},
            _testing_token=_TEST_TOKEN,
        )

    @property
    def is_production_adapter(self) -> bool:
        return not self._testing

    def metadata(self) -> dict[str, object]:
        return copy.deepcopy(dict(self._metadata))

    def configuration_sha256(self) -> str:
        if not self._testing:
            current_runtime = _verify_runtime()
            expected = {key: self._metadata[key] for key in current_runtime}
            if current_runtime != expected:
                raise RuntimeError("FP32 MACE-POLAR runtime dependency drifted.")
        current = _sha(
            {
                **dict(self._metadata),
                "base_configuration_sha256": self._graph.configuration_sha256(),
                "implementation_sha256": _implementation_sha256(),
            }
        )
        if current != self._configuration:
            raise RuntimeError("FP32 MACE-POLAR configuration drifted.")
        return current

    def topology_diagnostics(self, atoms: object) -> dict[str, object]:
        self.configuration_sha256()
        return self._graph.topology_diagnostics(atoms)

    def energy_source_torch(self, atoms: object, positions_angstrom: Any):
        import torch

        self.configuration_sha256()
        _require_process_float32("before model forward")
        if (
            not torch.is_tensor(positions_angstrom)
            or positions_angstrom.dtype is not torch.float32
            or str(positions_angstrom.device) != self.device
        ):
            raise TypeError(
                "FP32 MACE-POLAR positions must be float32 on the configured device."
            )
        try:
            energy, source = self._graph.energy_source_torch(atoms, positions_angstrom)
        finally:
            _require_process_float32("after model forward")
        if (
            not torch.is_tensor(energy)
            or energy.ndim != 0
            or energy.dtype is not torch.float32
            or energy.device != positions_angstrom.device
            or not torch.is_tensor(source)
            or source.shape != (len(atoms), 4)
            or source.dtype is not torch.float32
            or source.device != positions_angstrom.device
        ):
            raise RuntimeError(
                "FP32 MACE-POLAR forward must return scalar float32 energy and "
                "float32 (N,4) source on the configured device."
            )
        self.configuration_sha256()
        return energy, source


def build_official_mace_polar_fp32_adapter(
    device: str = "cpu", checkpoint_path: str | Path | None = None
) -> MACEPolarFP32Adapter:
    """Build the official gas-only FP32 graph from explicit checkpoint bytes."""

    _require_process_float32("before official construction")
    normalized_device = "cuda:0" if device == "cuda" else str(device)
    if normalized_device != "cpu" and not (
        normalized_device.startswith("cuda:") and normalized_device[5:].isdigit()
    ):
        raise ValueError("FP32 MACE-POLAR requires cpu or explicit cuda:N.")
    _prepare_cuda_device(normalized_device)
    runtime = _verify_runtime()
    checkpoint = _resolve_checkpoint(checkpoint_path)
    try:
        graph = _build_graph(device=normalized_device, checkpoint=checkpoint)
    finally:
        _require_process_float32("after official construction")
    return MACEPolarFP32Adapter(graph, runtime=runtime, _factory_token=_FACTORY_TOKEN)


__all__ = [
    "MACEPolarFP32Adapter",
    "MACE_POLAR_FP32_MODEL_PROFILE_ID",
    "MACE_POLAR_FP32_POLICY_ID",
    "MACE_POLAR_FP32_PROVIDER_ID",
    "build_official_mace_polar_fp32_adapter",
]
