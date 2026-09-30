"""Resolve measured analytic compositions without inspecting model families.

Capability is not qualification: no match means no installed admission. The
runtime cannot create receipts, add allowlist entries, or select a candidate
through input headers/environment variables. Qualification harnesses inject
their own resolver directly into the private factory constructor seam instead.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path

from ._qualified_analytic_cells import INSTALLED_QUALIFICATIONS
from .analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticCompositionAdmission,
    AnalyticDerivativeAdmission,
    AnalyticIdentityError,
    ScalarCellIdentity,
)

ANALYTIC_OPERATIONS = frozenset({"energy_forces", "hessian", "hvp"})
REQUIRED_ANALYTIC_CHECKS = (
    "center_energy",
    "center_forces",
    "raw_hessian_symmetry",
    "direct_hvp_vs_raw_hessian",
    "force_fd_finest",
    "force_fd_second_finest",
    "force_fd_pair_difference",
    "translation_leakage",
    "runtime_no_fd",
    "direct_hvp_no_dense_hessian",
    "public_workflows",
)
REQUIRED_QUALIFICATION_GATES = (
    "solvent_validation",
    "isolated_install",
    "candidate_workflows",
)
REQUIRED_RUNTIME_DEPENDENCIES = frozenset({"torch", "numpy", "scipy", "ase"})
_VALIDATION_FILES = frozenset(
    {
        "docs/implicit-solvation/benchmarks/validate_shared_mlip_analytic.py",
        "tests/solvation/test_shared_mlip_analytic_validation.py",
    }
)
_TERMINAL_STATUSES = frozenset(
    {"QUALIFIED", "FAILED", "UNAVAILABLE", "FEASIBILITY_BLOCKED"}
)
_INPUT_OPTIONS = frozenset({"canonical_payload_sha256", "ordered_charge_sha256"})


def cell_profile(cell: ScalarCellIdentity) -> dict:
    """Identify equations/checkpoint, retaining per-input identity in ``cell``.

    A relocated identical checkpoint is the same model. Fixed charges and OBC
    parameter payloads are system inputs, not a different implementation; they
    remain bound in the live admission/point and are checked on every operation.
    All other options (including solvent constants) remain qualification keys.
    """
    record = asdict(cell)
    record.pop("checkpoint_path")
    record["options"] = {
        key: json.loads(value)
        for key, value in cell.options
        if not (cell.component == "solvent" and key in _INPUT_OPTIONS)
    }
    return record


def runtime_implementation_paths() -> tuple[str, ...]:
    """All packaged code and runtime data, including public task dispatch.

    Only the later installed allowlist and its dedicated receipt directory are
    excluded. A changed parser, dispatcher, gas adapter, or equation profile
    invalidates the qualification just as a changed composition function does.
    """
    maple_root = Path(__file__).resolve().parents[2]
    paths = []
    for path in maple_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".json", ".csv", ".txt"}:
            continue
        relative = path.relative_to(maple_root.parent).as_posix()
        if (
            relative == "maple/function/calculator/_qualified_analytic_cells.py"
            or relative.startswith("maple/function/calculator/data/")
        ):
            continue
        paths.append(relative)
    return tuple(sorted(paths))


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _read_evidence(filename: str, expected_sha256: str) -> dict:
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise AnalyticIdentityError("Invalid installed qualification filename.")
    if not _is_sha256(expected_sha256):
        raise AnalyticIdentityError("Invalid installed qualification digest.")
    raw = resources.files(__package__).joinpath("data", filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AnalyticIdentityError("Installed qualification receipt bytes changed.")
    return json.loads(raw)


def _load_qualification(filename: str, expected_sha256: str) -> dict:
    artifact = _read_evidence(filename, expected_sha256)
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_type") != "route1-analytic-cell-qualification"
        or artifact.get("passed") is not True
        or artifact.get("all_cells_qualified") is not True
        or not isinstance(artifact.get("cells"), list)
        or len(artifact["cells"]) != 1
    ):
        raise AnalyticIdentityError("Incomplete installed qualification receipt.")
    for gate in REQUIRED_QUALIFICATION_GATES:
        if artifact.get("gates", {}).get(gate) is not True:
            raise AnalyticIdentityError(f"Qualification gate is not passed: {gate}.")
    validation = artifact.get("validation_files_sha256", {})
    if not _VALIDATION_FILES.issubset(validation) or not all(
        _is_sha256(value) for value in validation.values()
    ):
        raise AnalyticIdentityError(
            "Qualification validation source binding is incomplete."
        )
    source_hashes = artifact.get("implementation_files_sha256", {})
    if set(source_hashes) != set(runtime_implementation_paths()):
        raise AnalyticIdentityError(
            "Qualification runtime source manifest is incomplete or changed."
        )
    package_root = Path(__file__).resolve().parents[3]
    for relative, expected in source_hashes.items():
        path = (package_root / relative).resolve()
        if (
            not str(relative).startswith("maple/")
            or not path.is_relative_to(package_root / "maple")
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise AnalyticIdentityError(
                f"Qualified implementation source changed: {relative}."
            )
    row = artifact["cells"][0]
    if (
        row.get("status") != "QUALIFIED"
        or not isinstance(row.get("profile_id"), str)
        or not row["profile_id"]
        or row["profile_id"].startswith("CANDIDATE:")
        or set(row.get("checks", {})) != set(REQUIRED_ANALYTIC_CHECKS)
        or not all(row["checks"][name] is True for name in REQUIRED_ANALYTIC_CHECKS)
        or row.get("failures") != []
        or not REQUIRED_RUNTIME_DEPENDENCIES.issubset(
            row.get("dependency_versions", {})
        )
        or frozenset(row.get("operations", ())) != ANALYTIC_OPERATIONS
        or row.get("gas_profile", {}).get("component") != "gas"
        or row.get("solvent_profile", {}).get("component") != "solvent"
    ):
        raise AnalyticIdentityError(
            "Qualified cell is missing complete passing evidence."
        )
    matrix = _read_evidence(
        artifact.get("exhaustive_matrix_file", ""),
        artifact.get("exhaustive_matrix_sha256"),
    )
    matrix_cells = matrix.get("cells", [])
    if (
        matrix.get("artifact_type") != "route1-analytic-cell-matrix"
        or matrix.get("schema_version") != 1
        or matrix.get("final") is not True
        or not matrix_cells
        or any(item.get("status") not in _TERMINAL_STATUSES for item in matrix_cells)
        or sum(item == row for item in matrix_cells) != 1
        or len({item.get("profile_id") for item in matrix_cells}) != len(matrix_cells)
    ):
        raise AnalyticIdentityError(
            "Qualification does not match its complete terminal matrix."
        )
    return artifact


def resolve_installed_composition(
    gas_cell: ScalarCellIdentity,
    solvent_cell: ScalarCellIdentity,
) -> AnalyticCompositionAdmission:
    """Bind exact live inputs only to a reviewed, source-matching receipt."""
    gas_profile, solvent_profile = cell_profile(gas_cell), cell_profile(solvent_cell)
    for filename, digest in INSTALLED_QUALIFICATIONS:
        artifact = _load_qualification(filename, digest)
        for row in artifact["cells"]:
            if (
                row.get("status") != "QUALIFIED"
                or row.get("gas_profile") != gas_profile
                or row.get("solvent_profile") != solvent_profile
                or frozenset(row.get("operations", ())) != ANALYTIC_OPERATIONS
            ):
                continue
            for package, expected in row.get("dependency_versions", {}).items():
                if metadata.version(package) != expected:
                    raise AnalyticIdentityError(
                        f"Qualified dependency version changed: {package}."
                    )
            profile_id = row["profile_id"]
            evidence_id = f"sha256:{digest}"
            return AnalyticCompositionAdmission(
                profile_id=profile_id,
                evidence_id=evidence_id,
                gas_admission=AnalyticDerivativeAdmission(
                    f"{profile_id}:gas", evidence_id, gas_cell, ANALYTIC_OPERATIONS
                ),
                solvent_admission=AnalyticDerivativeAdmission(
                    f"{profile_id}:solvent",
                    evidence_id,
                    solvent_cell,
                    ANALYTIC_OPERATIONS,
                ),
                operations=ANALYTIC_OPERATIONS,
            )
    raise AnalyticCapabilityError(
        "No installed qualification matches this exact analytic gas/solvent "
        "checkpoint, options, device, dtype and implementation profile."
    )
