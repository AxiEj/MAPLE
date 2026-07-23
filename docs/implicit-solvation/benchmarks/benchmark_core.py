"""Shared deterministic machinery for MAPLE implicit-solvation benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import Any, Iterable
import urllib.request

import numpy as np


REQUIRED_PROTOCOL_KEYS = {
    "schema_version",
    "protocol_id",
    "claim_scope",
    "dataset",
    "domain",
    "partition",
    "methods",
    "providers",
    "statistics",
    "bins",
    "confirmation",
    "provider_parity",
    "result_schema_version",
    "literature",
}
HEX = frozenset("0123456789abcdef")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | os.PathLike[str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def load_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and set(value.lower()) <= HEX


def load_protocol(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if not isinstance(protocol, dict):
        raise ValueError("Benchmark protocol must be a JSON object.")
    missing = sorted(REQUIRED_PROTOCOL_KEYS - set(protocol))
    if missing:
        raise ValueError("Benchmark protocol is missing required keys: " + ", ".join(missing))
    if protocol["schema_version"] != 1 or protocol["result_schema_version"] != 1:
        raise ValueError("Only benchmark protocol/result schema version 1 is supported.")

    dataset = protocol["dataset"]
    if not _is_hex(str(dataset.get("commit", "")), 40):
        raise ValueError("Dataset commit must be a pinned 40-character Git SHA.")
    artifacts = dataset.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Dataset artifacts must be a non-empty list.")
    artifact_names: set[str] = set()
    for artifact in artifacts:
        name = str(artifact.get("name", ""))
        if not name or Path(name).name != name or name in artifact_names:
            raise ValueError(f"Invalid or duplicate dataset artifact name: {name!r}.")
        artifact_names.add(name)
        if not _is_hex(str(artifact.get("sha256", "")), 64):
            raise ValueError(f"Dataset artifact {name!r} requires a pinned SHA256.")
        if not str(artifact.get("url", "")).startswith("https://"):
            raise ValueError(f"Dataset artifact {name!r} requires an HTTPS source URL.")
    required_artifacts = {"database.txt", "database.json", "mol2files_gaff.tar.gz"}
    if artifact_names != required_artifacts:
        raise ValueError(
            "Dataset artifacts must be exactly database.txt, database.json, and "
            "mol2files_gaff.tar.gz."
        )

    partition = protocol["partition"]
    fraction = partition.get("development_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction < 1:
        raise ValueError("development_fraction must be strictly between zero and one.")
    if not str(partition.get("seed", "")):
        raise ValueError("Partition seed must be non-empty.")
    pilots = partition.get("pilot_development_only")
    if not isinstance(pilots, list) or len(pilots) != len(set(pilots)):
        raise ValueError("pilot_development_only must contain unique compound IDs.")

    benchmark_kind = str(protocol.get("benchmark_kind", ""))
    if benchmark_kind != "route2-macepolar-smd":
        raise ValueError(
            f"Route-2 branch requires benchmark_kind='route2-macepolar-smd', got {benchmark_kind!r}."
        )
    methods = protocol["methods"]
    for key in ("density_models", "solvation_models"):
        values = methods.get(key)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"methods.{key} must be a non-empty unique list.")
    statistics = protocol["statistics"]
    if int(statistics.get("bootstrap_resamples", 0)) <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    confidence = statistics.get("bootstrap_confidence")
    if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("bootstrap_confidence must be strictly between zero and one.")

    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def fetch_and_verify_artifacts(
    protocol: dict[str, Any],
    dataset_dir: Path,
    *,
    source_dir: Path | None = None,
) -> dict[str, str]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for artifact in protocol["dataset"]["artifacts"]:
        name = artifact["name"]
        destination = dataset_dir / name
        if not destination.exists():
            if source_dir is not None:
                source = source_dir / name
                if not source.is_file():
                    raise FileNotFoundError(f"Pinned dataset artifact not found in source dir: {source}")
                destination.write_bytes(source.read_bytes())
            else:
                with urllib.request.urlopen(artifact["url"], timeout=120) as response:
                    destination.write_bytes(response.read())
        actual = sha256_file(destination)
        if actual != artifact["sha256"]:
            raise ValueError(
                f"Dataset artifact hash mismatch for {name}: expected {artifact['sha256']}, "
                f"observed {actual}."
            )
        hashes[name] = actual
    return hashes


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError(f"Unsafe path in dataset archive: {member.name!r}.")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported archive entry type: {member.name!r}.")
        archive.extractall(destination)


def partition_for_smiles(
    smiles: str,
    *,
    seed: str,
    development_fraction: float,
    forced_development: bool = False,
) -> str:
    if forced_development:
        return "development"
    key = f"{seed}\0{smiles.strip()}".encode("utf-8")
    score = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return "development" if score < development_fraction else "confirmation"


def _edge_is_in_cycle(adjacency: list[set[int]], left: int, right: int) -> bool:
    visited = {left}
    stack = [left]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if (current == left and neighbor == right) or (
                current == right and neighbor == left
            ):
                continue
            if neighbor == right:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return False


def rotatable_bond_proxy(atoms) -> int:
    symbols = atoms.get_chemical_symbols()
    bonds = atoms.info["mol2"]["bonds"]
    adjacency = [set() for _ in symbols]
    for left, right, _bond_type in bonds:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    count = 0
    for left, right, bond_type in bonds:
        left = int(left)
        right = int(right)
        if str(bond_type) not in {"1", "1.0"}:
            continue
        if symbols[left] == "H" or symbols[right] == "H":
            continue
        heavy_left = sum(symbols[index] != "H" for index in adjacency[left])
        heavy_right = sum(symbols[index] != "H" for index in adjacency[right])
        if heavy_left <= 1 or heavy_right <= 1:
            continue
        if not _edge_is_in_cycle(adjacency, left, right):
            count += 1
    return count


def classify_candidate(atoms, groups: list[str]) -> dict[str, Any]:
    symbols = atoms.get_chemical_symbols()
    heavy = [symbol for symbol in symbols if symbol != "H"]
    hetero = [symbol for symbol in heavy if symbol != "C" and symbol not in {"F", "Cl", "Br", "I"}]
    halogen_count = sum(symbol in {"F", "Cl", "Br", "I"} for symbol in heavy)
    rotatable = rotatable_bond_proxy(atoms)
    if halogen_count:
        element_class = "halogen-containing"
    elif hetero:
        element_class = "heteroatom"
    else:
        element_class = "hydrocarbon"
    heavy_count = len(heavy)
    size_bin = "small" if heavy_count <= 6 else "medium" if heavy_count <= 12 else "large"
    hetero_count = len(hetero) + halogen_count
    hetero_bin = "zero" if hetero_count == 0 else "one" if hetero_count == 1 else "multiple"
    flexibility = "rigid" if rotatable == 0 else "limited" if rotatable <= 3 else "flexible"
    return {
        "elements": sorted(set(symbols)),
        "atom_count": len(symbols),
        "heavy_atom_count": heavy_count,
        "heteroatom_count": hetero_count,
        "halogen_count": halogen_count,
        "rotatable_bond_proxy": rotatable,
        "functional_groups": sorted(set(groups)) or ["unassigned"],
        "element_class": element_class,
        "size_bin": size_bin,
        "heteroatom_bin": hetero_bin,
        "flexibility_bin": flexibility,
    }


def expected_attempt_ids(
    candidate_ids: Iterable[str], charge_methods: Iterable[str], gb_models: Iterable[str]
) -> list[str]:
    return [
        f"{compound_id}__{charge_method}__{gb_model}"
        for compound_id in sorted(candidate_ids)
        for charge_method in charge_methods
        for gb_model in gb_models
    ]


def _metric_values(errors: np.ndarray) -> dict[str, float | int | None]:
    if errors.size == 0:
        return {"n": 0, "mse": None, "mae": None, "rmse": None, "max_absolute_error": None}
    absolute = np.abs(errors)
    return {
        "n": int(errors.size),
        "mse": float(np.mean(errors)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_absolute_error": float(np.max(absolute)),
    }


def summarize_errors(
    errors: Iterable[float],
    *,
    expected_count: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Summary errors must all be finite.")
    metrics = _metric_values(values)
    metrics["expected_count"] = int(expected_count)
    metrics["failure_count"] = int(expected_count - len(values))
    metrics["failure_rate"] = (
        float((expected_count - len(values)) / expected_count) if expected_count else None
    )
    if values.size == 0:
        metrics["bootstrap_ci"] = {"mse": None, "mae": None, "rmse": None}
        return metrics
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    samples = values[indices]
    alpha = (1.0 - confidence) / 2.0
    lower, upper = alpha, 1.0 - alpha
    bootstrap = {
        "mse": np.mean(samples, axis=1),
        "mae": np.mean(np.abs(samples), axis=1),
        "rmse": np.sqrt(np.mean(samples**2, axis=1)),
    }
    metrics["bootstrap_ci"] = {
        name: [float(np.quantile(sample, lower)), float(np.quantile(sample, upper))]
        for name, sample in bootstrap.items()
    }
    return metrics


def ensure_confirmation_lock(work_dir: Path, protocol_fingerprint: str) -> dict[str, Any]:
    path = work_dir / "confirmation-lock.json"
    if not path.is_file():
        raise ValueError(
            "Confirmation is locked. Run freeze-confirmation with a proposed default and "
            "pass rule before opening confirmation results."
        )
    lock = load_json(path)
    if lock.get("protocol_fingerprint") != protocol_fingerprint:
        raise ValueError("Confirmation lock protocol fingerprint does not match this protocol.")
    for key in ("proposed_default", "pass_rule", "frozen_at_utc"):
        if not str(lock.get(key, "")).strip():
            raise ValueError(f"Confirmation lock is missing {key}.")
    return lock
