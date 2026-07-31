"""Pinned property-only adapter for the official SolProp-mix ensemble.

SolProp-mix predicts a scalar solution free energy.  It is not a MAPLE
calculator, potential-energy surface, force provider, or absolute-solvation
backend.  Native float32 follows the released checkpoint execution precision;
float64 promotion is a MAPLE runtime promotion of released float32 parameters,
not additional learned precision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Protocol, cast

from rdkit import Chem

SOLPROPMIX_SOURCE_URL = "https://gitlab.kuleuven.be/creas/vermeiregroup/solprop"
SOLPROPMIX_SOURCE_REVISION = "80043ce09eb8802517c35b59254f8e9c181f2dac"
SOLPROPMIX_SOURCE_TREE = "bc7b933d6cf4c55c8ad10236383cfd09afbbfdc2"
SOLPROPMIX_CHECKPOINT_FAMILY = "SolPropmixQMExp"
SOLPROPMIX_DATA_RELEASE_VERSION = "v1.1"
SOLPROPMIX_DATA_RELEASE_RECORD = 15587866
SOLPROPMIX_PREVIOUS_DATA_RELEASE_RECORD = 14238055
SOLPROPMIX_STATIC_CODE_ZIP_NAME = "SolProp_ML-StaticCodeGsolv.zip"
SOLPROPMIX_STATIC_CODE_ZIP_SIZE_BYTES = 65183
SOLPROPMIX_STATIC_CODE_ZIP_MD5 = "602e6b9b4da49ac6b788023d2a6cadcd"
SOLPROPMIX_STATIC_CODE_ZIP_SHA256 = (
    "8ec40ef77699f1e8589b50daedbb00fca6255df860ec93d8a589744e898bd326"
)
SOLPROPMIX_MODELWEIGHTS_ZIP_SIZE_BYTES = 175360920
SOLPROPMIX_MODELWEIGHTS_ZIP_MD5 = "047cb1d69e3b51aced9eac2880eb10f6"
SOLPROPMIX_MODELWEIGHTS_ZIP_SHA256 = (
    "dbd391061829261e2485a6d6fcd864a8e39f14eff40a9a69bb70e2326a75b006"
)
SOLPROPMIX_SUPPLEMENTAL_SOURCE_ORIGIN = (
    "Zenodo record 15587866 (v1.1), supplemental static-code release"
)
SOLPROPMIX_CHECKPOINT_ORIGIN = (
    "Zenodo record 15587866 (v1.1), ModelWeights.zip/" "ModelWeights/SolPropmixQMExp"
)
SOLPROPMIX_CHECKPOINTS_BYTE_IDENTICAL_ACROSS_V1_0_V1_1 = True
SOLPROPMIX_WORKBOOK_ORACLE_REPRODUCTION_GUARANTEED = False
SOLPROPMIX_STATIC_DATA_HASHES = {
    "solvation_predictor/data/__init__.py": "2639b97ff1fe63a3c9e88402aa9d1369305be046762483db834325da22948a78",
    "solvation_predictor/data/data.py": "fd7bc4454b981686b3ef7fbb86d5069f9cb5eb7b4bbc454e3876ac61775aa83e",
    "solvation_predictor/data/Scaler.py": "5e4a65084db838238f14386af66672184d93764764b694b138fa97cf13dffa73",
    "solvation_predictor/data/Splitter.py": "f5244d923b5afec61e22d6d9ded1bef081545415b9b0efefe7282d8ab5e7d1ac",
}
SOLPROPMIX_CHECKPOINT_SHA256 = (
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
SOLPROPMIX_SUPPORTED_ELEMENTS = frozenset(
    {"H", "B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
)
SOLPROPMIX_PRECISIONS = frozenset({"native_float32", "float64_promoted"})
SOLPROPMIX_REFERENCE_TEMPERATURE_K = 298.15
SOLPROPMIX_PRECISION_SEMANTICS = {
    "native_float32": "literature-native execution of released float32 parameters",
    "float64_promoted": (
        "MAPLE float64 runtime promotion of released float32 parameters; "
        "not additional learned precision"
    ),
}
_MAX_OUTPUT_BYTES = 1024 * 1024
_MAX_FAILURE_TAIL_BYTES = 8000


class SolPropMixConfigError(ValueError):
    """Raised when an artifact identity or prediction request is invalid."""


class SolPropMixRuntimeError(RuntimeError):
    """Raised when isolated SolProp-mix inference cannot complete safely."""


@dataclass(frozen=True)
class SolPropMixSolventComponent:
    canonical_smiles: str
    mole_fraction: float


@dataclass(frozen=True)
class SolPropMixModelPrediction:
    model_index: int
    g298_kcal_mol: float
    h298_kcal_mol: float
    g_temperature_kcal_mol: float


@dataclass(frozen=True)
class SolPropMixRuntimeReceipt:
    property_only: bool
    absolute_solvation_backend: bool
    execution_device: str
    precision: str
    precision_semantics: str
    execution_dtype: str
    python_version: str
    torch_version: str
    cuda_version: str | None
    rdkit_version: str
    numpy_version: str
    deterministic_algorithms: bool
    float32_matmul_precision: str
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    amp_autocast_enabled: bool
    fp16_reduced_precision_reduction: bool
    bf16_reduced_precision_reduction: bool
    source_revision: str
    source_tree: str
    source_revision_scope: str
    checkpoint_family: str
    data_release_version: str
    data_release_record: int
    previous_data_release_record: int
    static_code_zip_name: str
    static_code_zip_size_bytes: int
    static_code_zip_md5: str
    static_code_zip_sha256: str
    modelweights_zip_size_bytes: int
    modelweights_zip_md5: str
    modelweights_zip_sha256: str
    supplemental_source_origin: str
    checkpoint_origin: str
    checkpoints_byte_identical_across_v1_0_v1_1: bool
    workbook_oracle_reproduction_guaranteed: bool
    worker_sha256: str
    checkpoint_sha256: tuple[str, ...]
    static_sha256: tuple[tuple[str, str], ...]
    canonical_solute_smiles: str
    solvent_components: tuple[SolPropMixSolventComponent, ...]
    solute_carbon_requirement_verified: bool
    isotope_free_structures_verified: bool
    solvent_net_neutrality_verified: bool
    solvent_nonionic_liquid_phase_structurally_verified: bool
    solvent_nonionic_liquid_phase_caller_requirement: bool
    gpu_name: str | None
    gpu_uuid: str | None
    gpu_compute_capability: str | None
    gpu_total_memory_bytes: int | None
    worker_stdout_sha256: str
    worker_stderr_sha256: str
    worker_stderr_truncated: bool


@dataclass(frozen=True)
class SolPropMixPropertyResult:
    predicted_solvation_free_energy_kcal_mol: float
    ensemble_population_std_kcal_mol: float
    model_predictions: tuple[SolPropMixModelPrediction, ...]
    canonical_solute_smiles: str
    solvent_components: tuple[SolPropMixSolventComponent, ...]
    temperature_kelvin: float
    runtime_receipt: SolPropMixRuntimeReceipt
    source_revision: str = SOLPROPMIX_SOURCE_REVISION
    unit: str = "kcal/mol"
    target_quantity: str = "property_prediction"
    precision_interpretation: str = "recorded in runtime_receipt"
    solvent_nonionic_liquid_phase_requirement: str = (
        "caller responsibility; not structurally provable from SMILES/InChI"
    )
    property_only: bool = True
    absolute_solvation_backend: bool = False


@dataclass(frozen=True)
class _SnapshotEvidence:
    worker_sha256: str
    static_sha256: tuple[tuple[str, str], ...]
    checkpoint_sha256: tuple[str, ...]


class _BinaryStream(Protocol):
    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def tell(self) -> int: ...

    def read(self, size: int = -1, /) -> bytes: ...


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(handle: _BinaryStream) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _bounded_tail(
    handle: _BinaryStream, limit: int = _MAX_FAILURE_TAIL_BYTES
) -> tuple[str, bool]:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    truncated = size > limit
    handle.seek(max(0, size - limit))
    data = handle.read(limit)
    return data.decode("utf-8", errors="replace"), truncated


def _bounded_text(handle: _BinaryStream, limit: int = _MAX_OUTPUT_BYTES) -> str:
    handle.seek(0)
    data = handle.read(limit + 1)
    if len(data) > limit:
        raise SolPropMixRuntimeError(
            f"SolProp-mix worker output exceeded the {limit}-byte bound."
        )
    return data.decode("utf-8", errors="strict")


def _normalize_origin(url: str) -> str:
    value = str(url).strip().rstrip("/").removesuffix(".git")
    lowered = value.casefold()
    if lowered.startswith("git@gitlab.kuleuven.be:"):
        return "https://gitlab.kuleuven.be/" + value.split(":", 1)[1].casefold()
    if lowered.startswith("ssh://git@gitlab.kuleuven.be/"):
        return (
            "https://gitlab.kuleuven.be/"
            + value.split("gitlab.kuleuven.be/", 1)[1].casefold()
        )
    return lowered


def _run_text_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            completed = subprocess.run(
                arguments, check=False, stdout=stdout, stderr=stderr
            )
            stdout_text, _ = _bounded_tail(stdout)
            stderr_text, _ = _bounded_tail(stderr)
    except OSError as exc:
        raise SolPropMixConfigError(
            "SolProp-mix identity verification requires git."
        ) from exc
    return subprocess.CompletedProcess(
        arguments, completed.returncode, stdout_text, stderr_text
    )


def _git_result(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run_text_command(["git", "-C", str(root), *args])


def _git_output(root: Path, *args: str) -> str:
    result = _git_result(root, *args)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SolPropMixConfigError(
            f"Unable to verify SolProp-mix checkout ({' '.join(args)}): {detail}"
        )
    return result.stdout.strip()


def _verify_file_bytes(path: Path, expected: str, label: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SolPropMixConfigError(
            f"Pinned SolProp-mix {label} is unreadable: {path}"
        ) from exc
    observed = _sha256_bytes(data)
    if observed != expected:
        raise SolPropMixConfigError(
            f"Pinned SolProp-mix {label} SHA256 mismatch: expected {expected}, got {observed}."
        )
    return data


def _verify_file(path: Path, expected: str, label: str) -> None:
    _verify_file_bytes(path, expected, label)


def _canonical_neutral_structure(value: str, role: str) -> str:
    structure = str(value).strip()
    if not structure:
        raise SolPropMixConfigError(f"SolProp-mix {role} structure must be non-empty.")
    molecule = (
        Chem.MolFromInchi(structure)
        if structure.startswith("InChI=")
        else Chem.MolFromSmiles(structure)
    )
    if molecule is None:
        raise SolPropMixConfigError(
            f"SolProp-mix {role} must be valid RDKit SMILES or InChI."
        )
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise SolPropMixConfigError(
            f"SolProp-mix does not accept disconnected {role} fragments."
        )
    if Chem.GetFormalCharge(molecule) != 0:
        raise SolPropMixConfigError(
            f"SolProp-mix is restricted to net-neutral {role} molecules."
        )
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise SolPropMixConfigError(
            f"SolProp-mix does not accept radical {role} molecules."
        )
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        raise SolPropMixConfigError(
            f"SolProp-mix does not accept isotope-labelled {role} molecules."
        )
    unsupported = sorted(
        {atom.GetSymbol() for atom in molecule.GetAtoms()}
        - SOLPROPMIX_SUPPORTED_ELEMENTS
    )
    if unsupported:
        raise SolPropMixConfigError(
            f"SolProp-mix does not support {role} element(s): {', '.join(unsupported)}."
        )
    if role == "solute" and not any(
        atom.GetSymbol() == "C" for atom in molecule.GetAtoms()
    ):
        raise SolPropMixConfigError("SolProp-mix solutes must contain carbon.")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _normalize_solvents(
    components: Mapping[str, float] | Sequence[tuple[str, float]],
) -> tuple[SolPropMixSolventComponent, ...]:
    entries = (
        list(components.items())
        if isinstance(components, Mapping)
        else list(components)
    )
    if not entries:
        raise SolPropMixConfigError(
            "SolProp-mix requires one to three solvent components."
        )
    combined: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise SolPropMixConfigError(
                "Each solvent component must be a (structure, mole_fraction) pair."
            )
        try:
            fraction = float(entry[1])
        except (TypeError, ValueError) as exc:
            raise SolPropMixConfigError(
                "Solvent mole fractions must be finite numbers."
            ) from exc
        if not math.isfinite(fraction):
            raise SolPropMixConfigError(
                "Solvent mole fractions must be finite numbers."
            )
        if fraction < 0:
            raise SolPropMixConfigError("Solvent mole fractions cannot be negative.")
        if not fraction:
            continue
        canonical = _canonical_neutral_structure(str(entry[0]), "solvent")
        combined[canonical] = math.fsum((combined.get(canonical, 0.0), fraction))
    if not combined:
        raise SolPropMixConfigError("SolProp-mix requires a nonzero solvent component.")
    if len(combined) > 3:
        raise SolPropMixConfigError(
            "SolProp-mix accepts at most three solvent components."
        )
    total = math.fsum(combined.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SolPropMixConfigError("Solvent mole fractions must sum to one.")
    return tuple(
        SolPropMixSolventComponent(identity, fraction / total)
        for identity, fraction in sorted(combined.items())
    )


def _temperature_adjusted(g298: float, h298: float, temperature: float) -> float:
    return temperature * (
        g298 / SOLPROPMIX_REFERENCE_TEMPERATURE_K
        - h298 * (1.0 / SOLPROPMIX_REFERENCE_TEMPERATURE_K - 1.0 / temperature)
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SolPropMixRuntimeError(f"SolProp-mix child returned invalid {label}.")
    return cast(Mapping[str, object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise SolPropMixRuntimeError(f"SolProp-mix child returned invalid {label}.")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SolPropMixRuntimeError(f"SolProp-mix child returned invalid {label}.")
    result = float(value)
    if not math.isfinite(result):
        raise SolPropMixRuntimeError(f"SolProp-mix child returned non-finite {label}.")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise SolPropMixRuntimeError(f"SolProp-mix child returned invalid {label}.")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise SolPropMixRuntimeError(f"SolProp-mix child returned invalid {label}.")
    return cast(list[object], value)


class SolPropMixPropertyAdapter:
    """Run a pinned SolProp-mix property ensemble from a private snapshot."""

    def __init__(
        self,
        source_root: str | Path,
        static_source_root: str | Path,
        weights_root: str | Path,
        *,
        timeout_seconds: float = 300.0,
        python_executable: str | Path | None = None,
    ):
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise SolPropMixConfigError(
                "timeout_seconds must be finite and positive."
            ) from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise SolPropMixConfigError("timeout_seconds must be finite and positive.")
        self.source_root = Path(source_root).expanduser().resolve()
        self.static_source_root = Path(static_source_root).expanduser().resolve()
        self.weights_root = Path(weights_root).expanduser().resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.timeout_seconds = timeout
        self.assert_artifact_identity()

    def assert_artifact_identity(self) -> None:
        for root, label in (
            (self.source_root, "source_root"),
            (self.static_source_root, "static_source_root"),
            (self.weights_root, "weights_root"),
        ):
            if not root.is_dir():
                raise SolPropMixConfigError(
                    f"SolProp-mix {label} is not a directory: {root}"
                )
        shadow = self.source_root / "solvation_predictor" / "data"
        if shadow.exists() or shadow.is_symlink():
            raise SolPropMixConfigError(
                "SolProp-mix source checkout contains a solvation_predictor/data shadow path."
            )
        if (
            _git_output(self.source_root, "rev-parse", "--is-inside-work-tree")
            != "true"
        ):
            raise SolPropMixConfigError(
                "SolProp-mix source_root must be a git work tree."
            )
        revision = _git_output(self.source_root, "rev-parse", "HEAD")
        if revision != SOLPROPMIX_SOURCE_REVISION:
            raise SolPropMixConfigError(
                f"SolProp-mix revision mismatch: expected {SOLPROPMIX_SOURCE_REVISION}, got {revision}."
            )
        tree = _git_output(self.source_root, "rev-parse", "HEAD^{tree}")
        if tree != SOLPROPMIX_SOURCE_TREE:
            raise SolPropMixConfigError(
                f"SolProp-mix source tree mismatch: expected {SOLPROPMIX_SOURCE_TREE}, got {tree}."
            )
        origin = _normalize_origin(
            _git_output(self.source_root, "config", "--get", "remote.origin.url")
        )
        if origin != _normalize_origin(SOLPROPMIX_SOURCE_URL):
            raise SolPropMixConfigError(
                f"SolProp-mix source must have official origin {SOLPROPMIX_SOURCE_URL}, got {origin!r}."
            )
        if _git_output(
            self.source_root, "status", "--porcelain", "--untracked-files=all"
        ):
            raise SolPropMixConfigError(
                "SolProp-mix source checkout has tracked or untracked modifications."
            )
        for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            result = _git_result(self.source_root, *args)
            if result.returncode != 0:
                raise SolPropMixConfigError(
                    "SolProp-mix source checkout has tracked modifications."
                )
        expected_names = tuple(f"model{index}.pt" for index in range(10))
        observed_names = tuple(
            sorted(path.name for path in self.weights_root.glob("*.pt"))
        )
        if observed_names != expected_names:
            raise SolPropMixConfigError(
                "SolProp-mix requires exactly model0.pt through model9.pt."
            )
        for relative in SOLPROPMIX_STATIC_DATA_HASHES:
            if not (self.static_source_root / relative).is_file():
                raise SolPropMixConfigError(
                    f"Pinned supplemental SolProp-mix module is missing: {relative}"
                )

    def predict(
        self,
        solute_structure: str,
        solvent_components: Mapping[str, float] | Sequence[tuple[str, float]],
        *,
        temperature_kelvin: float = SOLPROPMIX_REFERENCE_TEMPERATURE_K,
        device: str = "cpu",
        precision: str = "float64_promoted",
    ) -> SolPropMixPropertyResult:
        canonical_solute = _canonical_neutral_structure(solute_structure, "solute")
        canonical_solvents = _normalize_solvents(solvent_components)
        try:
            temperature = float(temperature_kelvin)
        except (TypeError, ValueError) as exc:
            raise SolPropMixConfigError("temperature_kelvin must be finite.") from exc
        if not math.isfinite(temperature) or not 280.0 <= temperature <= 350.0:
            raise SolPropMixConfigError(
                "temperature_kelvin must be within the paper-validated 280..350 K range."
            )
        if device not in {"cpu", "cuda:0"}:
            raise SolPropMixConfigError("device must be exactly 'cpu' or 'cuda:0'.")
        if precision not in SOLPROPMIX_PRECISIONS:
            raise SolPropMixConfigError(
                "precision must be 'native_float32' or 'float64_promoted'."
            )
        self.assert_artifact_identity()
        request: dict[str, object] = {
            "solute": canonical_solute,
            "solvents": [
                {"smiles": item.canonical_smiles, "fraction": item.mole_fraction}
                for item in canonical_solvents
            ],
            "temperature": temperature,
            "device": device,
            "precision": precision,
        }
        payload = self._run_script(json.dumps(request, sort_keys=True))
        return self._validate_payload(payload, request, canonical_solvents)

    def _archive_source(self, runtime_root: Path) -> None:
        archive_path = runtime_root / "source.tar"
        with archive_path.open("w+b") as archive, tempfile.TemporaryFile() as stderr:
            try:
                result = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(self.source_root),
                        "archive",
                        "--format=tar",
                        SOLPROPMIX_SOURCE_REVISION,
                    ],
                    check=False,
                    stdout=archive,
                    stderr=stderr,
                )
            except OSError as exc:
                raise SolPropMixRuntimeError(
                    "Unable to start git while archiving pinned SolProp-mix source."
                ) from exc
            if result.returncode != 0:
                detail, truncated = _bounded_tail(stderr)
                suffix = " [truncated]" if truncated else ""
                raise SolPropMixRuntimeError(
                    f"Unable to archive pinned SolProp-mix source{suffix}: {detail}"
                )
        source = runtime_root / "source"
        source.mkdir()
        with tarfile.open(archive_path, mode="r:") as bundle:
            root = source.resolve()
            for member in bundle.getmembers():
                target = (source / member.name).resolve()
                if target != root and root not in target.parents:
                    raise SolPropMixRuntimeError(
                        "Unsafe path in pinned source archive."
                    )
            bundle.extractall(source)
        archive_path.unlink()
        if (source / "solvation_predictor" / "data").exists():
            raise SolPropMixRuntimeError(
                "Pinned source archive unexpectedly contains supplemental data modules."
            )

    def _materialize_snapshot(self, runtime_root: Path) -> _SnapshotEvidence:
        self._archive_source(runtime_root)
        source = runtime_root / "source"
        static_observed: list[tuple[str, str]] = []
        for relative, expected in SOLPROPMIX_STATIC_DATA_HASHES.items():
            data = _verify_file_bytes(
                self.static_source_root / relative, expected, relative
            )
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            static_observed.append((relative, _sha256_bytes(data)))

        weights = runtime_root / "weights"
        weights.mkdir()
        checkpoint_observed: list[str] = []
        for index, expected in enumerate(SOLPROPMIX_CHECKPOINT_SHA256):
            name = f"model{index}.pt"
            data = _verify_file_bytes(self.weights_root / name, expected, name)
            (weights / name).write_bytes(data)
            checkpoint_observed.append(_sha256_bytes(data))

        worker_source = Path(__file__).with_name("_solpropmix_worker.py")
        worker_bytes = worker_source.read_bytes()
        worker_hash = _sha256_bytes(worker_bytes)
        (runtime_root / "worker.py").write_bytes(worker_bytes)
        (runtime_root / "source_identity.json").write_text(
            json.dumps(
                {
                    "revision": SOLPROPMIX_SOURCE_REVISION,
                    "tree": SOLPROPMIX_SOURCE_TREE,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        placeholder = runtime_root / "trained_models" / "SolPropmixWater"
        placeholder.mkdir(parents=True)
        return _SnapshotEvidence(
            worker_sha256=worker_hash,
            static_sha256=tuple(static_observed),
            checkpoint_sha256=tuple(checkpoint_observed),
        )

    def _child_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["NVIDIA_TF32_OVERRIDE"] = "0"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment.pop("PYTHONPATH", None)
        return environment

    def _run_script(self, request_json: str) -> Mapping[str, object]:
        with tempfile.TemporaryDirectory(prefix="maple-solpropmix-") as directory:
            runtime_root = Path(directory)
            snapshot = self._materialize_snapshot(runtime_root)
            request_path = runtime_root / "request.json"
            request_path.write_text(request_json, encoding="utf-8")
            with (
                tempfile.TemporaryFile() as stdout,
                tempfile.TemporaryFile() as stderr,
            ):
                try:
                    result = subprocess.run(
                        [
                            self.python_executable,
                            str(runtime_root / "worker.py"),
                            str(runtime_root),
                            str(request_path),
                        ],
                        cwd=runtime_root,
                        env=self._child_environment(),
                        check=False,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=self.timeout_seconds,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise SolPropMixRuntimeError(
                        "SolProp-mix inference exceeded its "
                        f"{self.timeout_seconds:g}-second timeout."
                    ) from exc
                except OSError as exc:
                    raise SolPropMixRuntimeError(
                        "Unable to start SolProp-mix inference."
                    ) from exc
                stdout_sha = _hash_file(stdout)
                stderr_sha = _hash_file(stderr)
                _, stderr_truncated = _bounded_tail(stderr)
                if result.returncode != 0:
                    detail, truncated = _bounded_tail(stderr)
                    if not detail:
                        detail, truncated = _bounded_tail(stdout)
                    marker = " [bounded tail; truncated]" if truncated else ""
                    raise SolPropMixRuntimeError(
                        "SolProp-mix inference failed"
                        f" (stderr_sha256={stderr_sha}){marker}: {detail}"
                    )
                output = _bounded_text(stdout)
        payload: Mapping[str, object] | None = None
        for line in reversed(output.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping):
                payload = cast(Mapping[str, object], candidate)
                break
        if payload is None:
            raise SolPropMixRuntimeError("SolProp-mix child emitted no JSON payload.")
        merged = dict(payload)
        merged["_snapshot"] = {
            "worker_sha256": snapshot.worker_sha256,
            "static_sha256": dict(snapshot.static_sha256),
            "checkpoint_sha256": list(snapshot.checkpoint_sha256),
        }
        merged["_output"] = {
            "stdout_sha256": stdout_sha,
            "stderr_sha256": stderr_sha,
            "stderr_truncated": stderr_truncated,
        }
        return merged

    def _validate_payload(
        self,
        payload: Mapping[str, object],
        request: Mapping[str, object],
        canonical_solvents: tuple[SolPropMixSolventComponent, ...],
    ) -> SolPropMixPropertyResult:
        if payload.get("solute") != request["solute"]:
            raise SolPropMixRuntimeError(
                "SolProp-mix child returned mismatched solute."
            )
        if payload.get("solvents") != request["solvents"]:
            raise SolPropMixRuntimeError(
                "SolProp-mix child returned mismatched solvents."
            )
        temperature = _float(payload.get("temperature"), "temperature")
        if temperature != cast(float, request["temperature"]):
            raise SolPropMixRuntimeError(
                "SolProp-mix child returned mismatched temperature."
            )
        device = _string(payload.get("device"), "device")
        precision = _string(payload.get("precision"), "precision")
        if device != request["device"] or precision != request["precision"]:
            raise SolPropMixRuntimeError(
                "SolProp-mix child returned mismatched runtime request."
            )

        rows = _list(payload.get("predictions"), "model predictions")
        if len(rows) != 10:
            raise SolPropMixRuntimeError(
                "SolProp-mix child must return exactly ten models."
            )
        predictions: list[SolPropMixModelPrediction] = []
        for index, raw_row in enumerate(rows):
            row = _mapping(raw_row, f"model prediction {index}")
            model_index = _float(row.get("model_index"), "model index")
            if not model_index.is_integer() or int(model_index) != index:
                raise SolPropMixRuntimeError(
                    "SolProp-mix child returned invalid model identity."
                )
            g298 = _float(row.get("g298"), "G298")
            h298 = _float(row.get("h298"), "H298")
            observed = _float(row.get("g_temperature"), "temperature-adjusted G")
            expected = _temperature_adjusted(g298, h298, temperature)
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise SolPropMixRuntimeError(
                    "SolProp-mix child returned invalid temperature adjustment."
                )
            predictions.append(SolPropMixModelPrediction(index, g298, h298, expected))
        values = tuple(item.g_temperature_kcal_mol for item in predictions)
        mean = math.fsum(values) / 10
        std = pstdev(values)
        if not math.isclose(
            _float(payload.get("mean"), "ensemble mean"),
            mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SolPropMixRuntimeError(
                "SolProp-mix child returned invalid ensemble mean."
            )
        if not math.isclose(
            _float(payload.get("std"), "ensemble std"),
            std,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SolPropMixRuntimeError(
                "SolProp-mix child returned invalid population std."
            )

        source_revision = _string(payload.get("source_revision"), "source revision")
        source_tree = _string(payload.get("source_tree"), "source tree")
        worker_hash = _string(payload.get("worker_sha256"), "worker hash")
        child_checkpoints = tuple(
            _string(value, "checkpoint hash")
            for value in _list(payload.get("checkpoint_sha256"), "checkpoint hashes")
        )
        child_static_map = _mapping(payload.get("static_sha256"), "static hashes")
        child_static = tuple(
            (key, _string(child_static_map.get(key), f"static hash {key}"))
            for key in SOLPROPMIX_STATIC_DATA_HASHES
        )
        snapshot = _mapping(payload.get("_snapshot"), "snapshot evidence")
        snapshot_worker = _string(snapshot.get("worker_sha256"), "snapshot worker hash")
        snapshot_checkpoints = tuple(
            _string(value, "snapshot checkpoint hash")
            for value in _list(
                snapshot.get("checkpoint_sha256"), "snapshot checkpoint hashes"
            )
        )
        snapshot_static_map = _mapping(
            snapshot.get("static_sha256"), "snapshot static hashes"
        )
        snapshot_static = tuple(
            (key, _string(snapshot_static_map.get(key), f"snapshot static hash {key}"))
            for key in SOLPROPMIX_STATIC_DATA_HASHES
        )
        if (
            source_revision != SOLPROPMIX_SOURCE_REVISION
            or source_tree != SOLPROPMIX_SOURCE_TREE
            or worker_hash != snapshot_worker
            or child_checkpoints != snapshot_checkpoints
            or child_checkpoints != SOLPROPMIX_CHECKPOINT_SHA256
            or child_static != snapshot_static
            or dict(child_static) != SOLPROPMIX_STATIC_DATA_HASHES
        ):
            raise SolPropMixRuntimeError(
                "SolProp-mix child artifact identity mismatch."
            )

        expected_dtype = "float64" if precision == "float64_promoted" else "float32"
        if _string(payload.get("dtype"), "dtype") != expected_dtype:
            raise SolPropMixRuntimeError("SolProp-mix child returned invalid dtype.")
        runtime = _mapping(payload.get("runtime"), "runtime receipt")
        safety = {
            "deterministic": True,
            "matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "amp": False,
            "fp16_reduced": False,
            "bf16_reduced": False,
        }
        for key, expected in safety.items():
            if runtime.get(key) != expected:
                raise SolPropMixRuntimeError(
                    "SolProp-mix strict precision setting mismatch."
                )
        gpu_value = runtime.get("gpu")
        gpu = None if gpu_value is None else _mapping(gpu_value, "GPU receipt")
        if device == "cuda:0" and gpu is None:
            raise SolPropMixRuntimeError(
                "SolProp-mix child omitted requested GPU identity."
            )
        if device == "cpu" and gpu is not None:
            raise SolPropMixRuntimeError(
                "SolProp-mix child reported GPU identity on CPU."
            )
        output = _mapping(payload.get("_output"), "output evidence")
        receipt = SolPropMixRuntimeReceipt(
            property_only=True,
            absolute_solvation_backend=False,
            execution_device=device,
            precision=precision,
            precision_semantics=SOLPROPMIX_PRECISION_SEMANTICS[precision],
            execution_dtype=expected_dtype,
            python_version=_string(runtime.get("python"), "Python version"),
            torch_version=_string(runtime.get("torch"), "Torch version"),
            cuda_version=(
                None
                if runtime.get("cuda") is None
                else _string(runtime.get("cuda"), "CUDA version")
            ),
            rdkit_version=_string(runtime.get("rdkit"), "RDKit version"),
            numpy_version=_string(runtime.get("numpy"), "NumPy version"),
            deterministic_algorithms=_boolean(
                runtime.get("deterministic"), "determinism"
            ),
            float32_matmul_precision=_string(
                runtime.get("matmul_precision"), "matmul precision"
            ),
            cuda_matmul_allow_tf32=_boolean(
                runtime.get("cuda_matmul_allow_tf32"), "CUDA TF32"
            ),
            cudnn_allow_tf32=_boolean(runtime.get("cudnn_allow_tf32"), "cuDNN TF32"),
            amp_autocast_enabled=_boolean(runtime.get("amp"), "AMP"),
            fp16_reduced_precision_reduction=_boolean(
                runtime.get("fp16_reduced"), "FP16 reduction"
            ),
            bf16_reduced_precision_reduction=_boolean(
                runtime.get("bf16_reduced"), "BF16 reduction"
            ),
            source_revision=source_revision,
            source_tree=source_tree,
            source_revision_scope=(
                "Git revision identifies the archived core source only; the complete "
                "runtime identity also requires the supplemental-source, checkpoint, "
                "and worker hashes recorded here."
            ),
            checkpoint_family=SOLPROPMIX_CHECKPOINT_FAMILY,
            data_release_version=SOLPROPMIX_DATA_RELEASE_VERSION,
            data_release_record=SOLPROPMIX_DATA_RELEASE_RECORD,
            previous_data_release_record=SOLPROPMIX_PREVIOUS_DATA_RELEASE_RECORD,
            static_code_zip_name=SOLPROPMIX_STATIC_CODE_ZIP_NAME,
            static_code_zip_size_bytes=SOLPROPMIX_STATIC_CODE_ZIP_SIZE_BYTES,
            static_code_zip_md5=SOLPROPMIX_STATIC_CODE_ZIP_MD5,
            static_code_zip_sha256=SOLPROPMIX_STATIC_CODE_ZIP_SHA256,
            modelweights_zip_size_bytes=SOLPROPMIX_MODELWEIGHTS_ZIP_SIZE_BYTES,
            modelweights_zip_md5=SOLPROPMIX_MODELWEIGHTS_ZIP_MD5,
            modelweights_zip_sha256=SOLPROPMIX_MODELWEIGHTS_ZIP_SHA256,
            supplemental_source_origin=SOLPROPMIX_SUPPLEMENTAL_SOURCE_ORIGIN,
            checkpoint_origin=SOLPROPMIX_CHECKPOINT_ORIGIN,
            checkpoints_byte_identical_across_v1_0_v1_1=(
                SOLPROPMIX_CHECKPOINTS_BYTE_IDENTICAL_ACROSS_V1_0_V1_1
            ),
            workbook_oracle_reproduction_guaranteed=(
                SOLPROPMIX_WORKBOOK_ORACLE_REPRODUCTION_GUARANTEED
            ),
            worker_sha256=worker_hash,
            checkpoint_sha256=child_checkpoints,
            static_sha256=child_static,
            canonical_solute_smiles=cast(str, request["solute"]),
            solvent_components=canonical_solvents,
            solute_carbon_requirement_verified=True,
            isotope_free_structures_verified=True,
            solvent_net_neutrality_verified=True,
            solvent_nonionic_liquid_phase_structurally_verified=False,
            solvent_nonionic_liquid_phase_caller_requirement=True,
            gpu_name=None if gpu is None else _string(gpu.get("name"), "GPU name"),
            gpu_uuid=None if gpu is None else _string(gpu.get("uuid"), "GPU UUID"),
            gpu_compute_capability=(
                None
                if gpu is None
                else _string(gpu.get("compute_capability"), "GPU compute capability")
            ),
            gpu_total_memory_bytes=(
                None
                if gpu is None
                else int(_float(gpu.get("total_memory_bytes"), "GPU memory"))
            ),
            worker_stdout_sha256=_string(output.get("stdout_sha256"), "stdout hash"),
            worker_stderr_sha256=_string(output.get("stderr_sha256"), "stderr hash"),
            worker_stderr_truncated=_boolean(
                output.get("stderr_truncated"), "stderr truncation"
            ),
        )
        return SolPropMixPropertyResult(
            predicted_solvation_free_energy_kcal_mol=mean,
            ensemble_population_std_kcal_mol=std,
            model_predictions=tuple(predictions),
            canonical_solute_smiles=cast(str, request["solute"]),
            solvent_components=canonical_solvents,
            temperature_kelvin=temperature,
            runtime_receipt=receipt,
        )
