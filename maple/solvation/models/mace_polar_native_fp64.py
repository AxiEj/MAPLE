"""Strict serialized-process FP64 boundary for the official MACE-POLAR model.

This module never changes PyTorch's process-global default dtype.  Callers must
select ``torch.float64`` before importing/building MACE (the dedicated v2 CLI
does that once at process entry).  The two-sided guards here turn any ambient
dtype drift into a hard failure rather than silently generating mixed-precision
evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import copy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

NATIVE_FP64_POLICY_ID = "macepolar-native-fp64-serialized-process-policy-v2"
NATIVE_FP64_PROVIDER_ID = "maple.macepolar-official-native-fp64.experimental-v2"
NATIVE_FP64_MODEL_PROFILE_ID = "macepolar-1m-zero-field-native-fp64-v2"
_GRAPH_LONGRANGE_VERSION = "0.4.0"
_REALSPACE_ELECTROSTATICS_SHA256 = (
    "2cf7e098f4960490e6e9a5868ca1ea01ca23beb56ff8ce2cd0a5fe8615aec268"
)
_FACTORY_TOKEN = object()
_TEST_TOKEN = object()
_TEST_PROVIDER_ID = "unregistered-engineering-test-macepolar-native-fp64-v2"
_TEST_PROFILE_ID = "unregistered-engineering-test-macepolar-native-fp64-profile-v2"


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_process_float64(boundary: str) -> None:
    import torch

    if torch.get_default_dtype() is not torch.float64:
        raise RuntimeError(
            f"Native-FP64 MACE-POLAR requires torch.float64 {boundary}; "
            f"received {torch.get_default_dtype()}. The library does not mutate "
            "the process-global default dtype."
        )


def _verify_graph_longrange_runtime() -> dict[str, str]:
    version = importlib.metadata.version("graph-longrange")
    if version != _GRAPH_LONGRANGE_VERSION:
        raise RuntimeError(
            "Native-FP64 MACE-POLAR requires graph-longrange=="
            f"{_GRAPH_LONGRANGE_VERSION}; received {version}."
        )
    import graph_longrange

    source = (
        Path(graph_longrange.__file__).resolve().parent / "realspace_electrostatics.py"
    )
    if not source.is_file():
        raise RuntimeError(
            "graph-longrange realspace_electrostatics.py is unavailable."
        )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha256 != _REALSPACE_ELECTROSTATICS_SHA256:
        raise RuntimeError(
            "graph-longrange real-space kernel bytes do not match the native-FP64 "
            "policy."
        )
    return {
        "graph_longrange_version": version,
        "graph_longrange_realspace_sha256": source_sha256,
        "graph_longrange_realspace_path": str(source),
    }


def _build_official_base(*, device: str, checkpoint_path=None):
    from maple.solvation.models.mace_polar import (
        build_official_mace_polar_1_m_adapter,
    )

    return build_official_mace_polar_1_m_adapter(
        device=device,
        checkpoint_path=checkpoint_path,
        torch_graph=True,
    )


class MACEPolarNativeFP64Adapter:
    """Immutable guard around the existing coordinate-connected graph adapter."""

    __slots__ = (
        "_configuration_sha256",
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
            raise ValueError("Invalid native-FP64 engineering-test token.")
        if not testing and _factory_token is not _FACTORY_TOKEN:
            raise ValueError(
                "Native-FP64 production identity requires its official factory."
            )
        required = (
            "configuration_sha256",
            "energy_source_torch",
            "metadata",
            "topology_diagnostics",
        )
        if any(not callable(getattr(graph, name, None)) for name in required):
            raise TypeError("Native-FP64 adapter requires the official graph API.")
        graph_dtype = str(getattr(graph, "dtype", None)).removeprefix("torch.")
        if graph_dtype != "float64":
            raise TypeError("Native-FP64 adapter requires a float64 model graph.")
        base_metadata = copy.deepcopy(dict(graph.metadata()))
        release = getattr(graph, "release_contract", None)
        provider_id = _TEST_PROVIDER_ID if testing else NATIVE_FP64_PROVIDER_ID
        profile_id = _TEST_PROFILE_ID if testing else NATIVE_FP64_MODEL_PROFILE_ID
        metadata = {
            "schema": "route2-macepolar-native-fp64-model-metadata-v2",
            "provider_id": provider_id,
            "model_profile_id": profile_id,
            "fp64_policy_id": NATIVE_FP64_POLICY_ID,
            "serialized_process_contract": True,
            "library_sets_default_dtype": False,
            "model_dtype": "torch.float64",
            "input_dtype": "torch.float64",
            "output_dtypes": ("torch.float64", "torch.float64"),
            "process_default_dtype": "torch.float64",
            "torch_version": torch.__version__,
            "mace_torch_version": importlib.metadata.version("mace-torch"),
            "checkpoint_sha256": getattr(release, "checkpoint_sha256", None),
            "base_configuration_sha256": graph.configuration_sha256(),
            "base_graph_metadata": base_metadata,
            **dict(runtime),
            "engineering_test_graph": testing,
        }
        configuration = _sha(
            {**metadata, "implementation_sha256": _implementation_sha256()}
        )
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_testing", testing)
        object.__setattr__(self, "device", str(graph.device))
        object.__setattr__(self, "dtype", torch.float64)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "model_profile_id", profile_id)
        object.__setattr__(self, "domain", graph.domain)
        object.__setattr__(self, "release_contract", release)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarNativeFP64Adapter is immutable.")
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
            current_runtime = _verify_graph_longrange_runtime()
            expected_runtime = {
                key: self._metadata[key]
                for key in (
                    "graph_longrange_version",
                    "graph_longrange_realspace_sha256",
                    "graph_longrange_realspace_path",
                )
            }
            if current_runtime != expected_runtime:
                raise RuntimeError("Native-FP64 MACE-POLAR runtime dependency drifted.")
        current = _sha(
            {
                **dict(self._metadata),
                "base_configuration_sha256": self._graph.configuration_sha256(),
                "implementation_sha256": _implementation_sha256(),
            }
        )
        if current != self._configuration_sha256:
            raise RuntimeError("Native-FP64 MACE-POLAR configuration drifted.")
        return current

    def topology_diagnostics(self, atoms: object) -> dict[str, object]:
        self.configuration_sha256()
        return self._graph.topology_diagnostics(atoms)

    def energy_source_torch(self, atoms: object, positions_angstrom: Any):
        import torch

        self.configuration_sha256()
        _require_process_float64("before model forward")
        if (
            not torch.is_tensor(positions_angstrom)
            or positions_angstrom.dtype is not torch.float64
            or str(positions_angstrom.device) != self.device
        ):
            raise TypeError(
                "Native-FP64 MACE-POLAR positions must be float64 on the configured "
                "device."
            )
        try:
            energy, source = self._graph.energy_source_torch(atoms, positions_angstrom)
        finally:
            _require_process_float64("after model forward")
        if (
            not torch.is_tensor(energy)
            or not torch.is_tensor(source)
            or energy.dtype is not torch.float64
            or source.dtype is not torch.float64
        ):
            raise RuntimeError(
                "Native-FP64 MACE-POLAR forward returned a non-float64 tensor."
            )
        self.configuration_sha256()
        return energy, source


def build_official_mace_polar_native_fp64_adapter(
    device: str, checkpoint_path=None
) -> MACEPolarNativeFP64Adapter:
    """Freshly construct the official graph under the strict FP64 precondition."""

    _require_process_float64("before official construction")
    normalized_device = "cuda:0" if device == "cuda" else str(device)
    if normalized_device != "cpu" and not (
        normalized_device.startswith("cuda:") and normalized_device[5:].isdigit()
    ):
        raise ValueError("Native-FP64 MACE-POLAR requires cpu or explicit cuda:N.")
    runtime = _verify_graph_longrange_runtime()
    try:
        base = _build_official_base(
            device=normalized_device, checkpoint_path=checkpoint_path
        )
    finally:
        _require_process_float64("after official construction")
    from maple.solvation.models.mace_polar_torch import MACEPolarTorchGraphAdapter

    try:
        graph = MACEPolarTorchGraphAdapter(base)
    finally:
        _require_process_float64("after graph-adapter construction")
    return MACEPolarNativeFP64Adapter(
        graph, runtime=runtime, _factory_token=_FACTORY_TOKEN
    )


__all__ = [
    "MACEPolarNativeFP64Adapter",
    "NATIVE_FP64_MODEL_PROFILE_ID",
    "NATIVE_FP64_POLICY_ID",
    "NATIVE_FP64_PROVIDER_ID",
    "build_official_mace_polar_native_fp64_adapter",
]
