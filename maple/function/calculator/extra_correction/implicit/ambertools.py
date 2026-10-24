"""Bounded AmberTools discovery and audited, isolated child execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


@dataclass
class ResolvedAmberTools:
    executable: str
    environment: dict[str, str]
    discovery: dict[str, Any]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _installation(executable: str, selection: str, method: str) -> ResolvedAmberTools:
    path = Path(executable).resolve()
    prefix = path.parent.parent
    environment = dict(os.environ)
    environment["PATH"] = str(path.parent) + os.pathsep + environment.get("PATH", "")
    discovery: dict[str, Any] = {"selection": selection, "prefix": str(prefix)}
    # A standalone executable may itself be a wrapper for an external install.
    # Do not claim it is ready: the audited execution is authoritative. Known
    # Amber prefixes, including registry candidates, must have local resources.
    managed = (prefix / "dat" / "antechamber").is_dir() or (
        prefix / "conda-meta"
    ).is_dir()
    if managed or selection in {"AMBERHOME", "current-prefix", "conda-registry"}:
        environment["AMBERHOME"] = str(prefix)
        resources: dict[str, Any] = {}
        for name in ("atomtype", "bondtype", "am1bcc", "sqm"):
            helper = prefix / "bin" / name
            if not helper.is_file() or not os.access(helper, os.X_OK):
                raise ImportError(
                    f"Incomplete AmberTools installation {prefix}: missing executable {name}."
                )
            resources[name] = {"path": str(helper), "sha256": file_sha256(helper)}
            wrapped_helper = prefix / "bin" / "wrapped_progs" / name
            if wrapped_helper.is_file():
                resources[name]["wrapped_binary"] = {
                    "path": str(wrapped_helper),
                    "sha256": file_sha256(wrapped_helper),
                }
        data_names = ["CONNECT.TPL", "APS.DAT", "ATOMTYPE_GFF2.DEF"]
        if method == "abcg2":
            data_names += ["ATOMTYPE_ABCG2.DEF", "BCCPARM_ABCG2.DAT"]
        else:
            data_names += ["ATOMTYPE_BCC.DEF", "BCCPARM.DAT"]
        data = {}
        for name in data_names:
            resource = prefix / "dat" / "antechamber" / name
            if not resource.is_file():
                raise ImportError(
                    f"Incomplete AmberTools installation {prefix}: missing data {name}."
                )
            data[name] = {"path": str(resource), "sha256": file_sha256(resource)}
        discovery.update(
            helpers=resources, data=data, readiness="required-resources-verified"
        )
        wrapped_executable = prefix / "bin" / "wrapped_progs" / "antechamber"
        if wrapped_executable.is_file():
            discovery["wrapped_executable"] = {
                "path": str(wrapped_executable),
                "sha256": file_sha256(wrapped_executable),
            }
    else:
        discovery["readiness"] = "external-wrapper-unverified-until-execution"
    discovery["child_environment"] = {
        key: environment.get(key) for key in ("PATH", "AMBERHOME")
    }
    return ResolvedAmberTools(str(path), environment, discovery)


def resolve_antechamber(
    executable: str | None = None, *, method: str = "am1bcc"
) -> ResolvedAmberTools:
    """Explicit > AMBERHOME > PATH > current prefix > bounded Conda registry.

    An explicit or higher-priority located install never silently falls through
    to another installation if its resources or execution are unusable.
    """
    if executable is not None:
        resolved = shutil.which(os.path.expanduser(executable))
        if resolved is None:
            raise ImportError(
                f"Unusable explicit AmberTools Antechamber executable: {executable!r}."
            )
        return _installation(resolved, "explicit", method)
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        resolved = shutil.which(
            str(Path(amberhome).expanduser() / "bin" / "antechamber")
        )
        if resolved is not None:
            return _installation(resolved, "AMBERHOME", method)
    resolved = shutil.which("antechamber")
    if resolved is not None:
        return _installation(resolved, "PATH", method)
    resolved = shutil.which(str(Path(sys.prefix) / "bin" / "antechamber"))
    if resolved is not None:
        return _installation(resolved, "current-prefix", method)

    registry = Path.home() / ".conda" / "environments.txt"
    candidates: dict[str, ResolvedAmberTools] = {}
    rejected = []
    if registry.is_file():
        # This is the only off-PATH search. No directory walking or shell init.
        lines = registry.read_text(encoding="utf-8").splitlines()
        if len(lines) > 256:
            raise ImportError(
                "Conda environment registry exceeds bounded discovery (256 entries); set #charge(executable=...)."
            )
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            prefix = Path(line.strip()).expanduser()
            if not prefix.is_absolute():
                continue
            resolved = shutil.which(str(prefix / "bin" / "antechamber"))
            if resolved is None:
                continue
            try:
                candidate = _installation(resolved, "conda-registry", method)
            except ImportError as exc:
                rejected.append(str(exc))
                continue
            candidates.setdefault(str(Path(resolved).resolve()), candidate)
    if len(candidates) > 1:
        paths = ", ".join(
            sorted(candidate.executable for candidate in candidates.values())
        )
        raise ImportError(
            f"Ambiguous registered AmberTools installations; select #charge(executable=/path/antechamber): {paths}"
        )
    if candidates:
        candidate = next(iter(candidates.values()))
        candidate.discovery.update(registry=str(registry), rejected_candidates=rejected)
        return candidate
    detail = " " + "; ".join(rejected) if rejected else ""
    raise ImportError(
        "Charge generation requires AmberTools24+ Antechamber. Set #charge(executable=/path/antechamber), "
        "AMBERHOME or PATH, or register an installed Conda environment." + detail
    )


def run_ambertools(
    command: list[str],
    *,
    installation: ResolvedAmberTools,
    audit_dir: Path,
    stem: str,
    timeout: float,
    provenance: dict[str, Any],
    working_directory: Path | None = None,
) -> None:
    """Persist attempted commands and raw output, including startup/timeouts."""
    record = {
        **provenance,
        "command": command,
        "discovery": installation.discovery,
        "status": "running",
        "working_directory": str(working_directory or audit_dir),
    }
    command_path = audit_dir / f"{stem}.command.json"
    output_path = Path(command[command.index("-o") + 1])
    # Do not accept a previous attempt's MOL2 if a tool exits without output.
    output_path.unlink(missing_ok=True)
    command_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    stdout = stderr = ""
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory or audit_dir,
            env=installation.environment,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
        record.update(returncode=completed.returncode)
        if completed.returncode != 0 or not output_path.is_file():
            operation = (
                "Native SQM AM1 optimization"
                if stem == "sqm"
                else f'Antechamber {stem.removeprefix("charges-")} '
                f'{"charge generation" if stem.startswith("charges-") else "inference"}'
            )
            raise RuntimeError(
                f"{operation} failed with exit code {completed.returncode}; see {audit_dir}."
            )
        record["status"] = "complete"
    except Exception as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            stdout, stderr = exc.stdout or "", exc.stderr or ""
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for stream, value in (("stdout", stdout), ("stderr", stderr)):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            (audit_dir / f"{stem}.{stream}.log").write_text(
                value or "", encoding="utf-8"
            )
        if output_path.is_file():
            record.update(
                output_file=str(output_path),
                output_file_sha256=file_sha256(output_path),
            )
            if output_path.suffix == ".mol2":
                record.update(
                    output_mol2=str(output_path),
                    output_mol2_sha256=file_sha256(output_path),
                )
        command_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
