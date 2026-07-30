#!/usr/bin/env python3
"""Audit the pinned AtomicESE release and scalar CLI boundary.

The audit verifies exact official Git blobs and exercises only a temporary
chmod copy of the Linux executable inside a no-network bubblewrap sandbox.
It does not score experimental data, validate accuracy, enable GPU admission,
or treat AtomicESE as an MLIP, PES, force model, or free-energy protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

AUDITED_ON = "2026-07-31"
SOURCE_REPOSITORY = "https://github.com/vyboishchikov/AtomicESE"
SOURCE_REVISION = "31e643c7e8974497c78fc2fb6f3c17778d61fa10"
SOURCE_TREE = "e08ce8bfaf73ca9c35c2b85929067e639d7bf048"
LINUX_BINARY_PATH = "AtomicESE.x"
WINDOWS_BINARY_PATH = "AtomicESE.exe"
README_PATH = "README.md"
SAMPLE_PATH = "input-example-0423brt.xyz"
EXPECTED_TREE_PATHS = (
    WINDOWS_BINARY_PATH,
    LINUX_BINARY_PATH,
    README_PATH,
    SAMPLE_PATH,
)
PINNED_BLOBS: dict[str, dict[str, Any]] = {
    WINDOWS_BINARY_PATH: {
        "git_blob": "1c9fa5a93943704e8394df6e7991dce4b35367c0",
        "size_bytes": 2_147_977,
        "sha256": "7951aacd8560d7e49eed8dddfeefa4015a2ec026ba8207024ebc50ba3dca3805",
    },
    LINUX_BINARY_PATH: {
        "git_blob": "fc56f822a20a55ba598e61a1088847c9b0b27d3d",
        "size_bytes": 1_525_368,
        "sha256": "4e300a5875d5f95ed0491816487d0df2ea696f22d2ab6af9bfae04c547da4d0c",
    },
    README_PATH: {
        "git_blob": "f7641a138fbbc281f3e126c719e46b06fbaa71b5",
        "size_bytes": 4_496,
        "sha256": "cf1c6d212af52dd70c177176d7246d30b43a78c0889ac9d44b04e175c1113b25",
    },
    SAMPLE_PATH: {
        "git_blob": "d6e95c81ade5b571a8b6eb5c27675428a652b89b",
        "size_bytes": 212,
        "sha256": "f374fad795e7a1da572d9d7505733870936cbf402a69dcc9c6b6da6219aae579",
    },
}
REPEAT_COUNT = 3
EXPECTED_SAMPLE_SOLVENT = "acetone"
EXPECTED_SAMPLE_RESULT_KCAL_MOL = -5.92
EXPECTED_CLI = (
    "AtomicESE.exe xyz-file -solvent solvent "
    "[-charge charge] [-PrintAtomicContributions]"
)
BWRAP_PATH = Path("/usr/bin/bwrap")
GIT_PATH = Path("/usr/bin/git")
GPU_MARKERS = (
    "cuda",
    "cudnn",
    "cublas",
    "nvidia",
    "opencl",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            [str(GIT_PATH), "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
    except FileNotFoundError as exc:
        raise RuntimeError("AtomicESE audit requires /usr/bin/git.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"AtomicESE Git inspection failed: {detail}") from exc
    return completed.stdout


def _parse_tree(data: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for raw_line in data.decode("utf-8", errors="strict").splitlines():
        metadata, path = raw_line.split("\t", 1)
        mode, object_type, object_id = metadata.split()
        if object_type != "blob" or path in entries:
            raise ValueError("AtomicESE tree contains an unsupported entry.")
        entries[path] = (mode, object_id)
    return entries


def _verify_blob(data: bytes, *, path: str, specification: Mapping[str, Any]) -> None:
    if len(data) != specification["size_bytes"]:
        raise ValueError(f"{path} size does not match the pinned release.")
    if _sha256_bytes(data) != specification["sha256"]:
        raise ValueError(f"{path} SHA256 does not match the pinned release.")


def _verify_release(source_root: Path) -> dict[str, Any]:
    root = source_root.expanduser().resolve(strict=True)
    revision = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git_bytes(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"AtomicESE revision {revision!r} does not match {SOURCE_REVISION!r}."
        )
    if tree != SOURCE_TREE:
        raise ValueError(f"AtomicESE tree {tree!r} does not match {SOURCE_TREE!r}.")
    dirty = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if dirty:
        raise ValueError("AtomicESE checkout must be clean before auditing.")

    entries = _parse_tree(_git_bytes(root, "ls-tree", "-r", SOURCE_REVISION))
    if tuple(entries) != EXPECTED_TREE_PATHS:
        raise ValueError(
            "AtomicESE release tree paths changed; absence claims require "
            "a fresh audit."
        )

    blobs: dict[str, bytes] = {}
    for path, specification in PINNED_BLOBS.items():
        mode, object_id = entries[path]
        if mode != "100644" or object_id != specification["git_blob"]:
            raise ValueError(f"{path} Git mode or blob identity changed.")
        data = _git_bytes(root, "show", f"{SOURCE_REVISION}:{path}")
        _verify_blob(data, path=path, specification=specification)
        working_path = root / path
        if not working_path.is_file() or working_path.is_symlink():
            raise ValueError(f"{path} is not a regular checkout file.")
        _verify_blob(
            working_path.read_bytes(),
            path=f"working tree {path}",
            specification=specification,
        )
        blobs[path] = data

    return {
        "blobs": blobs,
        "identity": {
            "repository": SOURCE_REPOSITORY,
            "revision": revision,
            "tree": tree,
            "checkout_clean": True,
            "files": {
                path: {
                    "git_blob": PINNED_BLOBS[path]["git_blob"],
                    "sha256": PINNED_BLOBS[path]["sha256"],
                    "size_bytes": PINNED_BLOBS[path]["size_bytes"],
                    "verified": True,
                }
                for path in sorted(PINNED_BLOBS)
            },
        },
        "tree_paths": list(entries),
    }


def _require_release_tokens(blobs: Mapping[str, bytes]) -> None:
    readme = blobs[README_PATH].decode("utf-8", errors="strict")
    required_readme = (
        "AtomicESE.exe <i>xyz-file</i> -solvent <i>solvent</i> "
        "-charge <i>charge</i>",
        "The <code>-charge</code> option is optional; the default charge is 0.",
        "-PrintAtomicContributions",
        "Cartesian coordinates (in &#8491;)",
        "Do not include any header",
    )
    missing = [token for token in required_readme if token not in readme]
    if missing:
        raise ValueError(f"AtomicESE README contract changed: {missing!r}.")

    linux_strings = _ascii_strings(blobs[LINUX_BINARY_PATH])
    required_binary = (
        "Usage: AtomicESE.exe file.xyz -solvent solvent_name [-charge charge]",
        "Total solvation free energy =",
        "kcal/mol",
        "CPU time =",
        "Water is not implemented in this method.",
    )
    missing = [token for token in required_binary if token not in linux_strings]
    if missing:
        raise ValueError(f"AtomicESE Linux CLI strings changed: {missing!r}.")


def _ascii_strings(data: bytes, minimum_length: int = 4) -> str:
    pattern = rb"[\x20-\x7e]{" + str(minimum_length).encode("ascii") + rb",}"
    return "\n".join(
        match.group().decode("ascii") for match in re.finditer(pattern, data)
    )


def _linux_binary_metadata(data: bytes) -> dict[str, Any]:
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Pinned AtomicESE Linux artifact is not ELF64 little-endian.")
    elf_type, machine = struct.unpack_from("<HH", data, 16)
    program_header_offset = struct.unpack_from("<Q", data, 32)[0]
    program_header_size = struct.unpack_from("<H", data, 54)[0]
    program_header_count = struct.unpack_from("<H", data, 56)[0]
    if elf_type != 2 or machine != 62 or program_header_size < 4:
        raise ValueError("Pinned AtomicESE Linux ELF identity changed.")
    program_types = {
        struct.unpack_from(
            "<I",
            data,
            program_header_offset + index * program_header_size,
        )[0]
        for index in range(program_header_count)
    }
    if 2 in program_types:
        raise ValueError("AtomicESE Linux binary unexpectedly has a dynamic segment.")
    return {
        "format": "ELF64_little_endian",
        "architecture": "x86_64",
        "statically_linked_no_dynamic_segment": True,
    }


def _gpu_markers(blobs: Mapping[str, bytes]) -> list[str]:
    text = "\n".join(
        _ascii_strings(blobs[path]).casefold()
        for path in (LINUX_BINARY_PATH, WINDOWS_BINARY_PATH)
    )
    return [marker for marker in GPU_MARKERS if marker in text]


def _sandbox_command(
    temporary_root: Path,
    arguments: Sequence[str],
) -> list[str]:
    return [
        str(BWRAP_PATH),
        "--unshare-all",
        "--new-session",
        "--die-with-parent",
        "--ro-bind",
        str(temporary_root),
        "/work",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--chdir",
        "/work",
        "/work/AtomicESE.x",
        *arguments,
    ]


def _run_sandboxed(
    temporary_root: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            _sandbox_command(temporary_root, arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={},
        )
    except FileNotFoundError as exc:
        raise RuntimeError("AtomicESE runtime audit requires /usr/bin/bwrap.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("AtomicESE sandbox execution timed out.") from exc


def _require_diagnostic(
    completed: subprocess.CompletedProcess[str],
    *,
    label: str,
    token: str,
) -> None:
    if completed.returncode != 0:
        raise ValueError(
            f"AtomicESE {label} exit code changed: {completed.returncode}."
        )
    if token not in completed.stdout or token not in completed.stderr:
        raise ValueError(f"AtomicESE {label} diagnostic contract changed.")
    if "Total solvation free energy =" in completed.stdout:
        raise ValueError(f"AtomicESE {label} unexpectedly produced a scalar result.")


def _parse_sample_result(completed: subprocess.CompletedProcess[str]) -> float:
    if completed.returncode != 0 or completed.stderr:
        raise ValueError("AtomicESE sample execution did not complete cleanly.")
    matches = re.findall(
        r"Total solvation free energy =\s*([-+]?\d+(?:\.\d+)?) kcal/mol",
        completed.stdout,
    )
    if len(matches) != 1:
        raise ValueError("AtomicESE sample scalar output contract changed.")
    value = float(matches[0])
    if not math.isfinite(value) or value != EXPECTED_SAMPLE_RESULT_KCAL_MOL:
        raise ValueError(
            f"AtomicESE sample result {value!r} differs from "
            f"{EXPECTED_SAMPLE_RESULT_KCAL_MOL!r}."
        )
    required = (
        'Solvent name "acetone" was read from the command line',
        "Number of atoms in the solute:    5",
        "CPU time =",
    )
    if any(token not in completed.stdout for token in required):
        raise ValueError("AtomicESE successful sample diagnostics changed.")
    return value


def _audit_runtime(linux_binary: bytes, sample: bytes) -> dict[str, Any]:
    if not BWRAP_PATH.is_file():
        raise RuntimeError("AtomicESE runtime audit requires /usr/bin/bwrap.")
    with tempfile.TemporaryDirectory(prefix="atomicese-release-audit-") as raw:
        temporary_root = Path(raw)
        executable = temporary_root / "AtomicESE.x"
        geometry = temporary_root / SAMPLE_PATH
        executable.write_bytes(linux_binary)
        geometry.write_bytes(sample)
        executable.chmod(0o700)

        sample_arguments = (SAMPLE_PATH, "-solvent", EXPECTED_SAMPLE_SOLVENT)
        values = [
            _parse_sample_result(_run_sandboxed(temporary_root, sample_arguments))
            for _ in range(REPEAT_COUNT)
        ]
        if len(set(values)) != 1:
            raise ValueError("AtomicESE scalar result is not deterministic.")

        water = _run_sandboxed(
            temporary_root,
            (SAMPLE_PATH, "-solvent", "water"),
        )
        _require_diagnostic(
            water,
            label="unsupported-water",
            token="Water is not implemented in this method.",
        )
        unknown = _run_sandboxed(
            temporary_root,
            (SAMPLE_PATH, "-solvent", "definitely_not_a_solvent"),
        )
        _require_diagnostic(
            unknown,
            label="unknown-solvent",
            token='Solvent "definitely_not_a_solvent" not in the list.',
        )

        if executable.read_bytes() != linux_binary or geometry.read_bytes() != sample:
            raise ValueError("AtomicESE changed a read-only runtime artifact.")

    return {
        "sandbox": {
            "bubblewrap_path": str(BWRAP_PATH),
            "network_namespace": "unshared",
            "temporary_binary_copy_chmod": "0700",
            "temporary_tree_mounted_read_only": True,
            "environment_cleared": True,
            "pinned_cache_executed": False,
        },
        "sample": {
            "arguments": list(sample_arguments),
            "repeat_count": REPEAT_COUNT,
            "unique_scalar_result_count": 1,
            "scalar_result": EXPECTED_SAMPLE_RESULT_KCAL_MOL,
            "unit": "kcal/mol",
            "deterministic_for_repeated_identical_cli_inputs": True,
            "wall_times_recorded_as_admission_evidence": False,
        },
        "failure_semantics": {
            "water_supported": False,
            "water_diagnostic_verified": True,
            "unknown_solvent_diagnostic_verified": True,
            "controlled_error_exit_code": 0,
            "exit_code_alone_is_success_evidence": False,
        },
    }


def _verified_file_record(path: str) -> dict[str, Any]:
    specification = PINNED_BLOBS[path]
    return {
        "git_blob": specification["git_blob"],
        "sha256": specification["sha256"],
        "size_bytes": specification["size_bytes"],
        "verified": True,
    }


def audit_release(source_root: Path) -> dict[str, Any]:
    """Return the deterministic, fail-closed AtomicESE release/runtime audit."""

    verified = _verify_release(source_root)
    blobs = verified["blobs"]
    _require_release_tokens(blobs)
    linux_metadata = _linux_binary_metadata(blobs[LINUX_BINARY_PATH])
    gpu_markers = _gpu_markers(blobs)
    if gpu_markers:
        raise ValueError(
            f"AtomicESE packaged GPU marker inventory changed: {gpu_markers!r}."
        )
    runtime = _audit_runtime(blobs[LINUX_BINARY_PATH], blobs[SAMPLE_PATH])

    return {
        "schema_version": 1,
        "audited_on": AUDITED_ON,
        "artifact": "AtomicESE",
        "audit_type": "release_identity_and_sandboxed_scalar_cli_boundary",
        "identity": {
            **verified["identity"],
            "files": {
                path: _verified_file_record(path) for path in sorted(PINNED_BLOBS)
            },
        },
        "release_boundary": {
            "scope": "exact_pinned_official_git_tree_only",
            "tree_paths": verified["tree_paths"],
            "external_publication_or_supporting_information_audited": False,
            "publication_declared_external_supporting_information": {
                "locator": (
                    "https://onlinelibrary.wiley.com/doi/suppl/"
                    "10.1002/jcc.70104/supinfo/"
                    "jcc70104-sup-0001-supinfo.zip"
                ),
                "reported_contents": (
                    "scaling factors, weights, biases, and calculated "
                    "solvation free energies"
                ),
                "bytes_audited": False,
                "sha256": None,
            },
            "license_file_present_in_pinned_git_tree": False,
            "license_terms_verified_for_pinned_git_release": False,
            "source_code_present_in_pinned_git_tree": False,
            "separate_checkpoint_artifact_present_in_pinned_git_tree": False,
            "checkpoint_identity_ledger_present_in_pinned_git_tree": False,
            "model_training_or_overlap_ledger_present_in_pinned_git_tree": False,
            "model_identity_packaged_only_within_pinned_git_tree": True,
        },
        "cli_contract": {
            "documented_syntax": EXPECTED_CLI,
            "geometry_format": "headerless atomic symbols or numbers and Cartesian coordinates",
            "coordinate_unit": "angstrom",
            "default_charge": 0,
            "optional_atomic_contributions": True,
            "solvation_output_unit": "kcal/mol",
            "quantity_label": "standard solvation free energy DeltaG°solv",
            "exact_standard_state_definition_exposed": False,
            "standard_state_status": "unknown_from_pinned_release",
        },
        "runtime": runtime,
        "packaged_compute_evidence": {
            "linux_binary": linux_metadata,
            "windows_binary_format": "PE32+_x86_64_console",
            "selected_gpu_or_cuda_ascii_markers_present": False,
            "selected_gpu_or_cuda_ascii_markers_checked": list(GPU_MARKERS),
            "runtime_reports_cpu_time": True,
            "gpu_runtime_verified": False,
            "packaged_gpu_capability_status": (
                "unverified_no_selected_gpu_or_cuda_ascii_markers"
            ),
        },
        "validation_scope": {
            "experimental_records_read": False,
            "experimental_scoring_performed": False,
            "accuracy_validated": False,
            "maximum_error_computed": False,
            "matched_qm_benchmark_performed": False,
            "wall_time_used_for_admission": False,
        },
        "capability_boundary": {
            "scalar_solvation_estimator_only": True,
            "mlip": False,
            "potential_energy_surface": False,
            "forces": False,
            "free_energy_sampling_protocol": False,
            "thermodynamic_protocol": False,
        },
        "admission_gates": {
            "gpu_admission_eligible": False,
            "matched_qm_speed_eligible": False,
            "accuracy_admission_eligible": False,
            "acceptance_eligible": False,
        },
        "acceptance_eligible": False,
        "route4_decision": {
            "status": "audit_only_not_integrated",
            "integration_performed": False,
            "allowed_use": "release provenance and CLI boundary evidence only",
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            os.environ.get(
                "MAPLE_ATOMICESE_AUDIT_ROOT",
                "/home/axie/.cache/maple-benchmarks/AtomicESE",
            )
        ),
        help="Exact pinned official AtomicESE Git checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "benchmarks"
            / "atomicese-release-audit-2026-07-31.json"
        ),
        help="Destination for the deterministic audit artifact.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    payload = audit_release(arguments.source_root)
    _write_json_atomic(arguments.output, payload)
    print(arguments.output)


if __name__ == "__main__":
    main()
