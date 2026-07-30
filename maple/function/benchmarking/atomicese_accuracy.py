"""Formal, label-free AtomicESE scalar property adapter."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from types import MappingProxyType

from .pretrained_hub import (
    AccuracyAdapterArtifactReceipt,
    AccuracyAdapterRegistration,
    AccuracyModelAdapter,
    AccuracyPredictionContext,
    BenchmarkQuantity,
)

ATOMICESE_FORMAL_ACCURACY_ADAPTER_ID = "atomicese-formal-v1"
ATOMICESE_SOURCE_REPOSITORY = "https://github.com/vyboishchikov/AtomicESE"
ATOMICESE_SOURCE_REVISION = "31e643c7e8974497c78fc2fb6f3c17778d61fa10"
ATOMICESE_SOURCE_TREE = "e08ce8bfaf73ca9c35c2b85929067e639d7bf048"
ATOMICESE_LINUX_BINARY_SHA256 = (
    "4e300a5875d5f95ed0491816487d0df2ea696f22d2ab6af9bfae04c547da4d0c"
)
ATOMICESE_RELEASE_AUDIT_SHA256 = (
    "efae3cd8355d6221a3814e6f1a8abdc06adb108d95d084f0589dbe3d0bfa708a"
)
ATOMICESE_FORMAL_RUNTIME_LOCK_SHA256 = (
    "2ccf4b70c5db73fe41afb5a1d3f8e19fa2219368e6b7711914811185199eb00e"
)
ATOMICESE_FORMAL_RUNTIME_LOCK_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/pretrained-solvation-hub/atomicese-formal-runtime-lock-2026-07-31.json"
)
ATOMICESE_RELEASE_AUDIT_PATH = (
    Path(__file__).resolve().parents[3] / "docs/pretrained-solvation-hub/benchmarks/"
    "atomicese-release-audit-2026-07-31.json"
)
_GIT_EXECUTABLE = Path("/usr/bin/git")
_REQUIRED_ARTIFACT_ROLES = (
    "adapter_code",
    "binary",
    "release_audit",
    "runner_code",
    "runtime_lock",
)
_SCALAR_PATTERN = re.compile(
    r"(?m)^\s*Total solvation free energy =\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?) kcal/mol$"
)
_ATOM_COUNT_PATTERN = re.compile(r"(?m)^\s*Number of atoms in the solute:\s+\d+$")
_CPU_TIME_PATTERN = re.compile(
    r"(?m)^\s*CPU time =\s*" r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)? seconds\.$"
)
_SUCCESS_LINE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"Program AtomicESE\. \(c\) Sergei F\. Vyboishchikov, Universitat de Girona",
        r"December, 2024",
        r"Job started\s+\d{2}\.\d{2}\.\d{4} at \d{2}:\d{2}",
        r'Solvent name ".+" was read from the command line',
        r"Dielectric constant eps=\s*[+-]?\d+(?:\.\d+)? was read from the command line",
        r"Boiling point=\s*[+-]?\d+(?:\.\d+)? will be used",
        r"Number of heavy \(non-hydrogen\) atoms in solvent =\s*\d+ will be used",
        r"Molar volume=\s*[+-]?\d+(?:\.\d+)? will be used",
        r"The number of hydrogen-bond centers =\s*\d+ will be used",
        r'Solute file ".+"',
        r"Number of atoms in the solute:\s+\d+",
        r"Atomic coordinates, calculated EE charges, and vdW radii:",
        r"X\s+Y\s+Z\s+Q\s+R",
        r"\d+\s+[A-Z][a-z]?\s+(?:[+-]?\d+(?:\.\d+)?\s+){4}[+-]?\d+(?:\.\d+)?",
        r"Total charge =\s*[+-]?\d+(?:\.\d+)?",
        r"Total solvation free energy =\s*[+-]?\d+(?:\.\d+)? kcal/mol",
        r"The program implements the AtomicESE method\.",
        r"The following paper should be cited:",
        r"Vyboishchikov S\.F\., J\. Comput\. Chem\. 2025, 46, e70104\. "
        r"DOI: 10\.1002/JCC\.70104",
        r"CPU time =\s*[+-]?\d+(?:\.\d+)? seconds\.",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected_sha256: str, *, label: str) -> Path:
    source = path.expanduser().resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Registered AtomicESE {label} must be a regular file.")
    observed_sha256 = _sha256(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Registered AtomicESE {label} SHA256 mismatch: expected "
            f"{expected_sha256}, observed {observed_sha256}."
        )
    return source


def _git_identity(source_root: Path) -> tuple[str, str]:
    if not _GIT_EXECUTABLE.is_file():
        raise RuntimeError("Formal AtomicESE registration requires /usr/bin/git.")

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            [str(_GIT_EXECUTABLE), "-C", str(source_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
        if completed.returncode != 0 or completed.stderr:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"AtomicESE Git identity inspection failed: {detail}")
        return completed.stdout.strip()

    revision = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    if revision != ATOMICESE_SOURCE_REVISION:
        raise ValueError(
            f"AtomicESE revision {revision!r} does not match the formal pin."
        )
    if tree != ATOMICESE_SOURCE_TREE:
        raise ValueError(f"AtomicESE tree {tree!r} does not match the formal pin.")
    if run("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("AtomicESE formal source checkout must be clean.")
    return revision, tree


def _parse_scalar_stdout(stdout: str, *, solvent: str) -> float:
    expected_solvent_line = f'Solvent name "{solvent}" was read from the command line'
    diagnostic_markers = (
        " is not implemented in this method.",
        " not in the list.",
        "fatal error",
    )
    if expected_solvent_line not in stdout or any(
        marker.casefold() in stdout.casefold() for marker in diagnostic_markers
    ):
        raise ValueError(
            "AtomicESE stdout contains an unexpected or diagnostic message."
        )
    unexpected_lines = [
        line
        for line in (raw_line.strip() for raw_line in stdout.splitlines())
        if line
        and not any(pattern.fullmatch(line) for pattern in _SUCCESS_LINE_PATTERNS)
    ]
    if unexpected_lines:
        raise ValueError(
            "AtomicESE stdout contains an unexpected or diagnostic message."
        )
    if _ATOM_COUNT_PATTERN.search(stdout) is None:
        raise ValueError("AtomicESE stdout atom-count line is malformed.")
    matches = tuple(_SCALAR_PATTERN.finditer(stdout))
    if len(matches) != 1 or _CPU_TIME_PATTERN.search(stdout) is None:
        raise ValueError("AtomicESE stdout does not contain one strict scalar result.")
    value = float(matches[0].group("value"))
    if not math.isfinite(value):
        raise ValueError("AtomicESE returned a non-finite scalar.")
    return value


class AtomicESEFormalAccuracyAdapter(AccuracyModelAdapter):
    """Execute the exact pinned Linux binary for one neutral solute and solvent."""

    required_artifact_roles = _REQUIRED_ARTIFACT_ROLES
    accuracy_precision_policy = "provider_packaged_unspecified"
    _SOLVENT_ALIASES = MappingProxyType(
        {
            "methanol": "methanol",
            "ethanol": "ethanol",
            "octanol": "1-octanol",
            "1-octanol": "1-octanol",
            "hexadecane": "n-hexadecane",
            "n-hexadecane": "n-hexadecane",
            "hexane": "n-hexane",
            "n-hexane": "n-hexane",
            "dmf": "n,n-dimethylformamide",
            "n,n-dimethylformamide": "n,n-dimethylformamide",
        }
    )

    def __init__(self, *, timeout_seconds: float = 60.0):
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "AtomicESE formal-adapter timeout must be finite and positive."
            )
        self.timeout_seconds = timeout

    def predict(self, *, context: AccuracyPredictionContext) -> float:
        if context.target_quantity is not BenchmarkQuantity.PROPERTY_PREDICTION:
            raise ValueError(
                "AtomicESE formal accuracy is restricted to property_prediction."
            )
        if context.molecular_input.molecular_input_format != "headerless_xyz":
            raise ValueError(
                "AtomicESE formal accuracy requires a headerless_xyz payload."
            )
        if (
            len(context.solvent_components) != 1
            or len(context.solvent_mole_fractions) != 1
            or not math.isclose(
                context.solvent_mole_fractions[0],
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "AtomicESE formal accuracy requires exactly one pure solvent at "
                "mole fraction 1."
            )
        solvent_label = context.solvent_components[0].strip().casefold()
        try:
            solvent = self._SOLVENT_ALIASES[solvent_label]
        except KeyError as exc:
            raise ValueError(
                f"Formal AtomicESE solvent {solvent_label!r} has no frozen alias."
            ) from exc

        artifact_paths = context.implementation_artifact_paths
        missing_roles = sorted(set(_REQUIRED_ARTIFACT_ROLES).difference(artifact_paths))
        if missing_roles:
            raise ValueError(
                "AtomicESE formal adapter lacks registered artifact roles: "
                + ", ".join(missing_roles)
                + "."
            )
        binary = _require_sha256(
            Path(artifact_paths["binary"]),
            ATOMICESE_LINUX_BINARY_SHA256,
            label="Linux binary",
        )
        _require_sha256(
            Path(artifact_paths["release_audit"]),
            ATOMICESE_RELEASE_AUDIT_SHA256,
            label="release audit",
        )
        _require_sha256(
            Path(artifact_paths["runtime_lock"]),
            ATOMICESE_FORMAL_RUNTIME_LOCK_SHA256,
            label="formal runtime lock",
        )
        molecular_payload = context.molecular_input.read()

        with tempfile.TemporaryDirectory(prefix="maple-atomicese-formal-") as raw:
            temporary_root = Path(raw)
            executable = temporary_root / "AtomicESE.x"
            shutil.copyfile(binary, executable)
            executable.chmod(stat.S_IRUSR | stat.S_IXUSR)
            input_path = temporary_root / "solute.xyz"
            input_path.write_bytes(molecular_payload)
            completed = subprocess.run(
                [str(executable), str(input_path), "-solvent", solvent],
                check=False,
                capture_output=True,
                text=True,
                cwd=temporary_root,
                timeout=self.timeout_seconds,
                env={"LC_ALL": "C", "LANG": "C"},
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"AtomicESE exited with code {completed.returncode}; formal "
                "prediction rejected."
            )
        if completed.stderr:
            raise RuntimeError(
                "AtomicESE emitted stderr despite exit code 0; formal prediction "
                "rejected."
            )
        return _parse_scalar_stdout(completed.stdout, solvent=solvent)


def build_atomicese_formal_accuracy_registration(
    *,
    source_root: str | Path,
    timeout_seconds: float = 60.0,
) -> AccuracyAdapterRegistration:
    """Build an exact formal registration from the pinned official checkout."""

    root = Path(source_root).expanduser().resolve(strict=True)
    revision, tree = _git_identity(root)
    binary = _require_sha256(
        root / "AtomicESE.x",
        ATOMICESE_LINUX_BINARY_SHA256,
        label="Linux binary",
    )
    release_audit = _require_sha256(
        ATOMICESE_RELEASE_AUDIT_PATH,
        ATOMICESE_RELEASE_AUDIT_SHA256,
        label="release audit",
    )
    runtime_lock = _require_sha256(
        ATOMICESE_FORMAL_RUNTIME_LOCK_PATH,
        ATOMICESE_FORMAL_RUNTIME_LOCK_SHA256,
        label="formal runtime lock",
    )
    adapter = AtomicESEFormalAccuracyAdapter(timeout_seconds=timeout_seconds)
    runtime_payload = json.loads(runtime_lock.read_text(encoding="utf-8"))
    if (
        runtime_payload.get("source_revision") != revision
        or runtime_payload.get("source_tree") != tree
        or runtime_payload.get("binary_sha256") != ATOMICESE_LINUX_BINARY_SHA256
        or runtime_payload.get("release_audit_sha256") != ATOMICESE_RELEASE_AUDIT_SHA256
        or runtime_payload.get("input_contract", {}).get("accuracy_precision_policy")
        != adapter.accuracy_precision_policy
        or runtime_payload.get("wall_times_recorded") is not False
    ):
        raise ValueError("AtomicESE formal runtime lock content is inconsistent.")
    runner_source = inspect.getsourcefile(AccuracyModelAdapter)
    if runner_source is None:
        raise ValueError("Formal accuracy-runner source file cannot be resolved.")
    return AccuracyAdapterRegistration.from_components(
        adapter_id=ATOMICESE_FORMAL_ACCURACY_ADAPTER_ID,
        adapter=adapter,
        implementation_artifacts=(
            AccuracyAdapterArtifactReceipt.from_file(
                role="adapter_code",
                path=Path(__file__),
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="binary",
                path=binary,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="release_audit",
                path=release_audit,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="runner_code",
                path=Path(runner_source),
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="runtime_lock",
                path=runtime_lock,
            ),
        ),
        configuration={
            "source_repository": ATOMICESE_SOURCE_REPOSITORY,
            "source_revision": revision,
            "source_tree": tree,
            "binary_sha256": ATOMICESE_LINUX_BINARY_SHA256,
            "release_audit_sha256": ATOMICESE_RELEASE_AUDIT_SHA256,
            "runtime_lock_sha256": ATOMICESE_FORMAL_RUNTIME_LOCK_SHA256,
            "timeout_seconds": float(timeout_seconds),
            "target_quantity": BenchmarkQuantity.PROPERTY_PREDICTION.value,
            "accuracy_precision_policy": adapter.accuracy_precision_policy,
            "wall_times_recorded": False,
        },
    )
