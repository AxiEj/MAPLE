"""Immutable identities and provenance for the private Torch smooth PCM."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

MODEL_ID = "torch-smooth-pcm-v1"
PROVIDER_ID = "maple.route2.continuum.torch-smooth-pcm-schwarz.provider.v1"
SCALAR_ID = "maple.route2.scalar.torch-smooth-pcm-schwarz-finite-dielectric.v1"
CONFIGURATION_CONTRACT_ID = "maple.route2.continuum.torch-smooth-pcm-schwarz-config.v1"
SOURCE_COMMIT = "c578fda2b974dd44833b8f0813bd689751f47d4d"
SOURCE_AUTHOR = "Jiahao Xie <2986086188@qq.com>"
CAPABILITIES: tuple[str, ...] = ()
LINEAGE_EVIDENCE_TRANSFERABLE = False

_LINEAGE_HASHES = (
    ("maple/solvation/continuum/harmonic_coefficients.py", "4d4de964bb7a4aec87a58096d8b298de1c6cefbb8cfd51f45c8f415a827bed01"),
    ("maple/solvation/continuum/harmonic_ddpcm_functional.py", "a6d5ae6257f9ba36ba68cc2ecc0ade62333bd51b5a8dedafa0c023f8a2b081d8"),
    ("maple/solvation/continuum/harmonic_ddpcm_primitives.py", "f19f8f967ed47f4f1aaf967b7f33f4aaed045b507f0ceb8564942dfbf800c1de"),
    ("maple/solvation/continuum/harmonic_schwarz_primitives.py", "a608e4f834a107f24d13b6521fd39c6f25af6b6e01b9d2765d9f90467b606a5e"),
    ("maple/solvation/continuum/harmonic_torch_primitives.py", "642e1202d21b5dee9805735f5e4dc32e82d033d3c29c9ed2c61bb8e6dc3e7bf0"),
)
_IMPLEMENTATION_FILES = (
    "__init__.py", "identity.py", "functional.py", "harmonics.py", "exposure.py",
    "point_l1.py", "schwarz.py", "ddpcm_operators.py", "scalar.py",
)

def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def sha256_payload(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

def lineage_hashes() -> tuple[tuple[str, str], ...]:
    return _LINEAGE_HASHES

def implementation_hashes() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parent
    return tuple((name, hashlib.sha256((root / name).read_bytes()).hexdigest()) for name in _IMPLEMENTATION_FILES)

def execution_record(configuration_sha256: str) -> dict[str, Any]:
    import ase, numpy, scipy, torch
    cpuinfo = Path("/proc/cpuinfo").read_text(errors="replace") if Path("/proc/cpuinfo").exists() else "unavailable"
    cpu_fields = tuple(line for line in cpuinfo.splitlines() if line.startswith(("model name", "flags", "Features")))
    return {
        "configuration_sha256": configuration_sha256,
        "versions": {"torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__, "ase": ase.__version__},
        "platform": platform.platform(), "processor": platform.processor(), "cpu_model_features": cpu_fields,
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "torch_parallel": torch.__config__.parallel_info(),
        "torch_build": torch.__config__.show(),
        "numpy_blas": repr(numpy.__config__.CONFIG),
    }

__all__=["MODEL_ID","PROVIDER_ID","SCALAR_ID","CONFIGURATION_CONTRACT_ID","SOURCE_COMMIT","SOURCE_AUTHOR","CAPABILITIES","LINEAGE_EVIDENCE_TRANSFERABLE","canonical_json","sha256_payload","lineage_hashes","implementation_hashes","execution_record"]
