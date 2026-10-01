#!/usr/bin/env python3
"""Local, label-free validation helpers for Gaussian-CHA R6 numerical profiles.

The frozen-v1 comparator is cheap and deterministic.  The optional v2 local
runner evaluates the exact terminal coordinates against the unchanged
NumPy/SciPy oracle at orders 64/96/128; it is intentionally separate from the
root-owned campaign runner and never optimizes coordinates.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
from ase import Atoms

V1_PROFILE_ID = "gaussian-cha-r6-v1"
V2_PROFILE_ID = (
    "gaussian-cha-r6-derivative-v2-numerical-profile-20261001.2-direct-complement"
)
FROZEN_ORACLE_SHA256 = (
    "e69c73bd7d98c6d1c802ebe9231c847db73c35e3b6b6be1a93ac67cf8c3c5b75"
)
FROZEN_FIXTURE_SHA256 = (
    "341b339488eaf1aee99eee0ae4b8a2ff9f60f4e62e9ad0f7586356041ffa93a7"
)
FROZEN_FIXTURE_CONTENT_SHA256 = (
    "18f363520ee45140eea17f74484554ff4818f4355e403ea4122a67bd817d8571"
)
FROZEN_FIXTURE_LINEAGE_SHA256 = (
    "6aea8092a34138d38e611d43ccd37d8a0b8e1f8c0ed487246fbd00e00d2ebefc"
)
FROZEN_ORACLE_CLOSURE_SHA256 = {
    "docs/implicit-solvation/benchmarks/cha_gaussian_reference.py": FROZEN_ORACLE_SHA256,
    "docs/implicit-solvation/benchmarks/cha_continuum_reference.py": (
        "826091b814df9a68ac8d608be83a7381ed2ac7bc7fe838cb97667c3e097fee07"
    ),
    "maple/function/calculator/extra_correction/implicit/sphere_union_volume.py": (
        "f5631cf299df7d20ae7ec8e092d12b964213f9c6e0af093553d6e2cbc13267ed"
    ),
    "maple/function/calculator/extra_correction/implicit/sphere_union_dispersion.py": (
        "d2c1acbc98b515471d39d1fb720a940d6012ecba27c261d4064a7d09d7b62a3f"
    ),
}
EPS_FLOAT64 = 2.220446049250313e-16
KCAL_PER_HARTREE = 627.5094740631
ORDERS = (64, 96, 128)
REFERENCE_FD_ORDER = 128
FD_STEPS = (2.0e-5, 1.0e-5, 5.0e-6)
FD_MULTIPLIERS = (-2.0, -1.0, 1.0, 2.0)
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests/solvation/data/cha_r6_v2/frozen_v1_terminal_fixture.json"
)
WORKTREE = Path(__file__).resolve().parents[3]
SOURCE_RELATIVE_PATHS = (
    "docs/implicit-solvation/benchmarks/validate_cha_r6_v2_local.py",
    *FROZEN_ORACLE_CLOSURE_SHA256.keys(),
    "maple/function/calculator/extra_correction/implicit/gaussian_cha_profiles.py",
    "maple/function/calculator/extra_correction/implicit/gaussian_cha_correction.py",
    "maple/function/calculator/extra_correction/implicit/torch_continuum_chagb_gaussian.py",
    "maple/function/calculator/extra_correction/implicit/torch_continuum_r6_derivative_v2.py",
    "maple/function/calculator/extra_correction/implicit/torch_continuum_r6_contact_v2.py",
    "maple/function/calculator/extra_correction/implicit/continuum_chagb_inputs.py",
    "maple/function/calculator/extra_correction/implicit/torch_continuum_chagb_domain.py",
)
ALLOWED_PROVENANCE_DRIFT = {
    "worktree",
    "source_identity_sha256",
    "timestamp",
    "log_path",
    "counter_overhead",
    "profile_wrapper_id",
}
EXPECTED_ACTIVE_SOURCE_ORIGINS = {
    "profiles": "maple/function/calculator/extra_correction/implicit/gaussian_cha_profiles.py",
    "correction": "maple/function/calculator/extra_correction/implicit/gaussian_cha_correction.py",
    "inputs": "maple/function/calculator/extra_correction/implicit/continuum_chagb_inputs.py",
    "oracle": "docs/implicit-solvation/benchmarks/cha_gaussian_reference.py",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical(
            {key: item for key, item in value.items() if key != "content_sha256"}
        )
    ).hexdigest()


def _topology_content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical(
            {key: item for key, item in value.items() if key != "content_sha256"}
        )
    ).hexdigest()


def _coordinate_sha256(positions: Any) -> str:
    array = np.asarray(positions, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("terminal coordinates must have finite shape (3, 3)")
    return hashlib.sha256(
        array.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_FIXTURE_SHA256:
        raise ValueError("frozen-v1 fixture external SHA256 differs")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("content_sha256") != _content_sha256(
        value
    ):
        raise ValueError("frozen-v1 fixture content seal differs")
    if (
        value.get("label_free") is not True
        or value.get("experimental_labels_read") is not False
    ):
        raise ValueError("frozen-v1 fixture is not explicitly label-free")
    topology = value["topology"]
    if topology.get("content_sha256") != _topology_content_sha256(topology):
        raise ValueError("frozen-v1 topology content seal differs")
    rows = value["rows"]
    expected = [
        (center, width)
        for center in (0.96, 0.98, 1.0, 1.02, 1.04)
        for width in (0.001, 0.003, 0.01)
    ]
    observed = [(row["center_scale"], row["sigma_e"]) for row in rows]
    if observed != expected or len({row["row_id"] for row in rows}) != 15:
        raise ValueError("frozen-v1 fixture roster differs")
    for row in rows:
        if _coordinate_sha256(row["positions_angstrom"]) != row["coordinate_sha256"]:
            raise ValueError("frozen-v1 terminal coordinate hash differs")
        _validate_raw_inventory(row, value["scientific_identity"])
        derived_row = derive_frozen_v1_row(row, value)
        if derived_row != row["frozen_v1_result"]:
            raise ValueError(
                f"frozen-v1 row metrics do not derive from raw evidence: {row['row_id']}"
            )
    derived = {
        "row_count": len(rows),
        "solvent_independent_pass_count": sum(
            bool(row["frozen_v1_result"]["solvent_passed"]) for row in rows
        ),
        "combined_total_fd_pass_count": sum(
            bool(row["frozen_v1_result"]["combined_passed"]) for row in rows
        ),
        "both_pass_count": sum(
            bool(row["frozen_v1_result"]["both_passed"]) for row in rows
        ),
        "status": "GAUSSIAN_OPT_INCOMPLETE",
    }
    if derived != value["frozen_v1_outcome"]:
        raise ValueError("frozen-v1 outcome does not derive from its exact rows")
    if value["lineage"]["frozen_numpy_scipy_oracle_sha256"] != FROZEN_ORACLE_SHA256:
        raise ValueError("frozen-v1 fixture oracle identity differs")
    if (
        value.get("lineage_sha256")
        != hashlib.sha256(_canonical(value["lineage"])).hexdigest()
    ):
        raise ValueError("frozen-v1 fixture lineage SHA256 differs")
    if (
        value["content_sha256"] != FROZEN_FIXTURE_CONTENT_SHA256
        or value["lineage_sha256"] != FROZEN_FIXTURE_LINEAGE_SHA256
    ):
        raise ValueError("frozen-v1 fixture immutable identity differs")
    if value["gates"] != {
        "energy_max_abs_kcal_mol": 1e-6,
        "force_max_abs_kcal_mol_per_angstrom": 2e-5,
        "force_rms_kcal_mol_per_angstrom": 5e-6,
        "finest_step_disagreement_max_kcal_mol_per_angstrom": 1e-5,
    } or value["combined_gates"] != {
        "force_max_abs_hartree_per_angstrom": 3.187202875281032e-8,
        "force_rms_hartree_per_angstrom": 7.96800718820258e-9,
        "finest_step_disagreement_max_hartree_per_angstrom": 1.593601437640516e-8,
    }:
        raise ValueError("frozen-v1 fixture gates differ")
    return value


def _validate_raw_inventory(row: dict[str, Any], identity: dict[str, Any]) -> None:
    for domain in ("solvent", "combined"):
        raw = row[domain]["raw_fd"]
        if len(raw) != 27:
            raise ValueError(f"{domain} raw FD inventory must contain 27 stencils")
        for flat, record in enumerate(raw):
            expected_atom, expected_axis = divmod(flat // 3, 3)
            expected_step = identity["fd_steps_angstrom"][flat % 3]
            if (record["atom_index"], record["axis"]) != (
                expected_atom,
                expected_axis,
            ):
                raise ValueError(f"{domain} raw FD Cartesian order differs")
            if float(record["step_angstrom"]) != float(expected_step):
                raise ValueError(f"{domain} raw FD step differs")
            if record["multipliers"] != identity["stencil_multipliers"]:
                raise ValueError(f"{domain} raw FD multipliers differ")
            energies = np.asarray(record["energies"], dtype=np.float64)
            if energies.shape != (4,) or not np.isfinite(energies).all():
                raise ValueError(f"{domain} raw FD energies must be four finite values")


def direct_field_tolerance(baseline: float) -> float:
    return 128.0 * EPS_FLOAT64 * max(1.0, abs(float(baseline)))


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", float(value))


def _iter_direct_fields(row: dict[str, Any]):
    for key, value in row["solvent"]["components_kcal_mol"].items():
        yield f"solvent.components.{key}", float(value)
    for index, value in enumerate(
        np.asarray(row["solvent"]["forces_kcal_mol_per_angstrom"]).flat
    ):
        yield f"solvent.forces.{index}", float(value)
    for order, energies in row["solvent"][
        "reference_energies_kcal_mol_by_order"
    ].items():
        for key, value in energies.items():
            yield f"solvent.reference.{order}.{key}", float(value)
    yield "combined.energy", float(row["combined"]["energy_hartree"])
    for index, value in enumerate(
        np.asarray(row["combined"]["forces_hartree_per_angstrom"]).flat
    ):
        yield f"combined.forces.{index}", float(value)


def _direct_map(row: dict[str, Any]) -> dict[str, float]:
    return dict(_iter_direct_fields(row))


def _validate_actual_row_shapes(row: dict[str, Any]) -> None:
    for domain, key in (
        ("solvent", "forces_kcal_mol_per_angstrom"),
        ("combined", "forces_hartree_per_angstrom"),
    ):
        forces = np.asarray(row[domain][key], dtype=np.float64)
        if forces.shape != (3, 3) or not np.isfinite(forces).all():
            raise ValueError(f"actual {domain} forces must have finite shape (3,3)")
    values = np.asarray(list(_direct_map(row).values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("actual direct energy/force fields must be finite")


def _raw_energy_bits(row: dict[str, Any]):
    for domain in ("solvent", "combined"):
        for stencil_index, stencil in enumerate(row[domain]["raw_fd"]):
            for energy_index, energy in enumerate(stencil["energies"]):
                yield domain, stencil_index, energy_index, _float_bits(energy)


def _raw_metadata(row: dict[str, Any]):
    result = []
    for domain in ("solvent", "combined"):
        for stencil in row[domain]["raw_fd"]:
            energies = np.asarray(stencil["energies"], dtype=np.float64)
            result.append(
                (
                    domain,
                    int(stencil["atom_index"]),
                    int(stencil["axis"]),
                    _float_bits(stencil["step_angstrom"]),
                    tuple(_float_bits(value) for value in stencil["multipliers"]),
                    energies.shape,
                    bool(np.isfinite(energies).all()),
                )
            )
    return result


def _validate_fresh_replay_envelope(actual: Any) -> list[dict[str, Any]]:
    if not isinstance(actual, dict) or actual.get("schema") != (
        "route1-cha-r6-v1-fresh-replay-v1"
    ):
        raise ValueError("actual input must be a fresh replay envelope")
    if (
        actual.get("content_sha256") != _content_sha256(actual)
        or actual.get("fresh_run") is not True
        or actual.get("producer") != "root-v1-replay-from-auditor-v1"
        or not isinstance(actual.get("producer_source_sha256"), str)
        or len(actual["producer_source_sha256"]) != 64
        or not isinstance(actual.get("source_identity_sha256"), str)
        or len(actual["source_identity_sha256"]) != 64
        or actual.get("numerical_profile_id") != V1_PROFILE_ID
        or not isinstance(actual.get("preregistration_sha256"), str)
        or len(actual["preregistration_sha256"]) != 64
        or not actual.get("source_before")
        or actual.get("source_before") != actual.get("source_after")
        or not actual.get("input_before")
        or actual.get("input_before") != actual.get("input_after")
    ):
        raise ValueError("actual fresh replay envelope binding differs")
    rows = actual.get("rows")
    if not isinstance(rows, list):
        raise ValueError("actual fresh replay envelope rows are missing")
    fresh_receipts = actual.get("fresh_receipts")
    expected_ids = [row["row_id"] for row in rows]
    if not isinstance(fresh_receipts, dict) or list(fresh_receipts) != expected_ids:
        raise ValueError("actual fresh receipt inventory differs")
    if any(
        not isinstance(value, dict)
        or set(value) != {"relative_path", "file_sha256", "content_sha256"}
        for value in fresh_receipts.values()
    ):
        raise ValueError("actual fresh receipt descriptors differ")
    return rows


def _verify_fresh_receipt_files(
    actual: dict[str, Any], rows: list[dict[str, Any]], evidence_root: Path
) -> None:
    root = evidence_root.resolve()
    for row in rows:
        descriptor = actual["fresh_receipts"][row["row_id"]]
        path = (root / descriptor["relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("actual fresh receipt path escapes evidence root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor["file_sha256"]:
            raise ValueError("actual fresh receipt file SHA256 differs")
        receipt = json.loads(raw)
        if (
            not isinstance(receipt, dict)
            or receipt.get("content_sha256") != _content_sha256(receipt)
            or receipt.get("content_sha256") != descriptor["content_sha256"]
            or receipt.get("schema") != "route1-cha-r6-v1-fresh-replay-row-v1"
            or receipt.get("row_id") != row["row_id"]
            or receipt.get("numerical_profile_id") != V1_PROFILE_ID
            or receipt.get("preregistration_sha256") != actual["preregistration_sha256"]
            or receipt.get("normalized_row_sha256")
            != hashlib.sha256(_canonical(row)).hexdigest()
            or receipt.get("normalized_row") != row
            or receipt.get("source_identity_sha256")
            != row["provenance"]["source_identity_sha256"]
            or receipt.get("source_identity_sha256") != actual["source_identity_sha256"]
        ):
            raise ValueError("actual fresh receipt content binding differs")


def compare_v1_replay(
    actual: dict[str, Any], fixture: dict[str, Any], *, evidence_root: Path
) -> dict[str, Any]:
    actual_rows = _validate_fresh_replay_envelope(actual)
    _verify_fresh_receipt_files(actual, actual_rows, evidence_root)
    frozen_receipts = set(fixture["lineage"]["terminal_row_file_sha256"].values())
    if frozen_receipts.intersection(
        descriptor["file_sha256"] for descriptor in actual["fresh_receipts"].values()
    ):
        raise ValueError("actual fresh receipt SHA256 reuses frozen evidence")
    expected_rows = fixture["rows"]
    failures = []
    raw_mismatches = []
    provenance_failures = []
    outcome_failures = []
    outcome_metric_deltas = []
    actual_derived_results = []
    raw_metadata_failures = []
    roster_ok = [row.get("row_id") for row in actual_rows] == [
        row["row_id"] for row in expected_rows
    ]
    if len(actual_rows) != len(expected_rows):
        roster_ok = False
    if roster_ok:
        for actual, expected in zip(actual_rows, expected_rows, strict=True):
            if (
                actual.get("center_scale") != expected["center_scale"]
                or actual.get("sigma_e") != expected["sigma_e"]
                or actual.get("positions_angstrom") != expected["positions_angstrom"]
                or actual.get("coordinate_sha256") != expected["coordinate_sha256"]
            ):
                roster_ok = False
            try:
                _validate_actual_row_shapes(actual)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                failures.append(
                    {
                        "row_id": expected["row_id"],
                        "field": "direct-shape-or-finiteness",
                        "message": str(exc),
                    }
                )
                continue
            actual_direct = _direct_map(actual)
            expected_direct = _direct_map(expected)
            if actual_direct.keys() != expected_direct.keys():
                failures.append({"row_id": expected["row_id"], "field": "direct-keys"})
            else:
                for field, baseline in expected_direct.items():
                    observed = actual_direct[field]
                    if not math.isfinite(observed) or abs(
                        observed - baseline
                    ) > direct_field_tolerance(baseline):
                        failures.append(
                            {
                                "row_id": expected["row_id"],
                                "field": field,
                                "observed": observed,
                                "baseline": baseline,
                            }
                        )
            expected_bits = list(_raw_energy_bits(expected))
            actual_bits = list(_raw_energy_bits(actual))
            try:
                if _raw_metadata(actual) != _raw_metadata(expected):
                    raw_metadata_failures.append(expected["row_id"])
            except (KeyError, TypeError, ValueError, OverflowError):
                raw_metadata_failures.append(expected["row_id"])
            if len(actual_bits) != len(expected_bits):
                raw_mismatches.append(
                    {"row_id": expected["row_id"], "field": "raw-inventory"}
                )
            else:
                for actual_item, expected_item in zip(
                    actual_bits, expected_bits, strict=True
                ):
                    if actual_item != expected_item:
                        raw_mismatches.append(
                            {
                                "row_id": expected["row_id"],
                                "domain": expected_item[0],
                                "stencil_index": expected_item[1],
                                "energy_index": expected_item[2],
                            }
                        )
            actual_provenance = actual.get("provenance", {})
            expected_provenance = expected["provenance"]
            fixed_keys = set(expected_provenance) - ALLOWED_PROVENANCE_DRIFT
            if any(
                actual_provenance.get(key) != expected_provenance[key]
                for key in fixed_keys
            ) or not (set(actual_provenance) - set(expected_provenance)).issubset(
                ALLOWED_PROVENANCE_DRIFT
            ):
                provenance_failures.append(expected["row_id"])
            actual_derived = derive_frozen_v1_row(actual, fixture)
            actual_derived_results.append(actual_derived)
            if actual.get("frozen_v1_result") != actual_derived:
                outcome_failures.append(expected["row_id"])
            expected_result = expected["frozen_v1_result"]
            if any(
                actual_derived[key] != expected_result[key]
                for key in ("solvent_passed", "combined_passed", "both_passed")
            ):
                outcome_failures.append(expected["row_id"])
            outcome_metric_deltas.append(
                {
                    "row_id": expected["row_id"],
                    "solvent": {
                        key: actual_derived["solvent_metrics"][key]
                        - expected_result["solvent_metrics"][key]
                        for key in expected_result["solvent_metrics"]
                    },
                    "combined": {
                        key: actual_derived["combined_metrics"][key]
                        - expected_result["combined_metrics"][key]
                        for key in expected_result["combined_metrics"]
                    },
                }
            )
    if roster_ok:
        actual_counts = {
            "solvent": sum(row["solvent_passed"] for row in actual_derived_results),
            "combined": sum(row["combined_passed"] for row in actual_derived_results),
            "both": sum(row["both_passed"] for row in actual_derived_results),
        }
        if actual_counts != {"solvent": 4, "combined": 2, "both": 2}:
            outcome_failures.append("aggregate-4-2-2")
    return {
        "scope": "FIELD_COMPARISON_ONLY",
        "freshness_verified_by_comparator": False,
        "freshness_requires_producer_postflight": True,
        "science_qualified": False,
        "row_count": len(actual_rows),
        "roster_exact": roster_ok,
        "direct_field_failure_count": len(failures),
        "raw_fd_bit_mismatch_count": len(raw_mismatches),
        "raw_fd_metadata_failure_count": len(raw_metadata_failures),
        "provenance_failure_count": len(provenance_failures),
        "outcome_failure_count": len(outcome_failures),
        "direct_field_failures": failures,
        "raw_fd_bit_mismatches": raw_mismatches,
        "raw_fd_metadata_failures": raw_metadata_failures,
        "provenance_failures": provenance_failures,
        "outcome_failures": outcome_failures,
        "outcome_metric_deltas": outcome_metric_deltas,
        "frozen_outcome": deepcopy(fixture["frozen_v1_outcome"]),
        "passed": bool(
            roster_ok
            and not failures
            and not raw_mismatches
            and not raw_metadata_failures
            and not provenance_failures
            and not outcome_failures
        ),
        "status": (
            "COMPARATOR_PASSED_ONLY"
            if roster_ok
            and not failures
            and not raw_mismatches
            and not raw_metadata_failures
            and not provenance_failures
            and not outcome_failures
            else "COMPARATOR_FAILED"
        ),
    }


def current_source_manifest() -> dict[str, str]:
    relatives = set(SOURCE_RELATIVE_PATHS)
    relatives.update(
        str(path.relative_to(WORKTREE)) for path in (WORKTREE / "maple").rglob("*.py")
    )
    manifest = {}
    for relative in sorted(relatives):
        path = WORKTREE / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest[relative] = _sha256_file(path)
    for relative, expected in FROZEN_ORACLE_CLOSURE_SHA256.items():
        if manifest.get(relative) != expected:
            raise ValueError(f"frozen oracle dependency changed: {relative}")
    return dict(sorted(manifest.items()))


def current_dependency_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "torch", "ase")
        },
    }


def _expected_cells(
    fixture: dict[str, Any], selected_row_ids: set[str] | None
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in fixture["rows"]
        if selected_row_ids is None or row["row_id"] in selected_row_ids
    ]
    if selected_row_ids is not None and (
        not selected_row_ids or {row["row_id"] for row in rows} != selected_row_ids
    ):
        raise ValueError(
            "bounded preregistration requires a nonempty frozen row subset"
        )
    return [
        {
            "row_id": row["row_id"],
            "sigma_e": row["sigma_e"],
            "coordinate_sha256": row["coordinate_sha256"],
            "production_order": order,
            "reference_fd_order": REFERENCE_FD_ORDER,
        }
        for row in rows
        for order in ORDERS
    ]


def build_local_preregistration(
    fixture: dict[str, Any],
    selected_row_ids: set[str] | None = None,
    profile_id: str = V2_PROFILE_ID,
) -> dict[str, Any]:
    if profile_id not in {V1_PROFILE_ID, V2_PROFILE_ID}:
        raise ValueError("local preregistration numerical profile differs")
    cells = _expected_cells(fixture, selected_row_ids)
    value = {
        "schema": "route1-cha-r6-v2-local-preregistration-v1",
        "fixture_file_sha256": FROZEN_FIXTURE_SHA256,
        "fixture_content_sha256": FROZEN_FIXTURE_CONTENT_SHA256,
        "fixture_lineage_sha256": FROZEN_FIXTURE_LINEAGE_SHA256,
        "oracle_closure_sha256": dict(FROZEN_ORACLE_CLOSURE_SHA256),
        "source_before": current_source_manifest(),
        "source_manifest_scope": "all-worktree-maple-python-plus-local-validator-and-frozen-oracle",
        "dependency_versions": current_dependency_versions(),
        "input_before": {
            "fixture_file_sha256": FROZEN_FIXTURE_SHA256,
            "topology_content_sha256": fixture["topology"]["content_sha256"],
            "topology_file_sha256": fixture["scientific_identity"][
                "topology_file_sha256"
            ],
            "checkpoint_sha256": fixture["scientific_identity"]["checkpoint_sha256"],
        },
        "profile_ids": [V1_PROFILE_ID, V2_PROFILE_ID],
        "numerical_profile_id": profile_id,
        "expected_active_source_origins": dict(EXPECTED_ACTIVE_SOURCE_ORIGINS),
        "runtime_contract": {
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "torch_num_threads": 1,
        },
        "unit_contract": {
            "energy": "kcal/mol",
            "force": "kcal/mol/angstrom",
            "coordinates": "angstrom",
        },
        "gates": deepcopy(fixture["gates"]),
        "orders": list(ORDERS),
        "reference_fd_order": REFERENCE_FD_ORDER,
        "bounded": selected_row_ids is not None,
        "cells": cells,
        "full_campaign_performed": False,
        "science_qualified": False,
    }
    value["content_sha256"] = _content_sha256(value)
    return value


def load_local_preregistration(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("local preregistration external SHA256 differs")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("content_sha256") != _content_sha256(
        value
    ):
        raise ValueError("local preregistration content seal differs")
    if (
        value.get("fixture_file_sha256") != FROZEN_FIXTURE_SHA256
        or value.get("fixture_content_sha256") != FROZEN_FIXTURE_CONTENT_SHA256
        or value.get("fixture_lineage_sha256") != FROZEN_FIXTURE_LINEAGE_SHA256
        or value.get("oracle_closure_sha256") != FROZEN_ORACLE_CLOSURE_SHA256
        or value.get("orders") != list(ORDERS)
        or value.get("reference_fd_order") != REFERENCE_FD_ORDER
        or value.get("numerical_profile_id") not in {V1_PROFILE_ID, V2_PROFILE_ID}
        or value.get("source_manifest_scope")
        != "all-worktree-maple-python-plus-local-validator-and-frozen-oracle"
        or value.get("expected_active_source_origins") != EXPECTED_ACTIVE_SOURCE_ORIGINS
        or value.get("runtime_contract")
        != {
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "torch_num_threads": 1,
        }
        or value.get("unit_contract")
        != {
            "energy": "kcal/mol",
            "force": "kcal/mol/angstrom",
            "coordinates": "angstrom",
        }
        or value.get("full_campaign_performed") is not False
        or value.get("science_qualified") is not False
    ):
        raise ValueError("local preregistration immutable identity differs")
    fixture = load_fixture()
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise ValueError("local preregistration cell inventory differs")
    if value.get("bounded") is True:
        row_ids = {cell.get("row_id") for cell in cells}
        if not row_ids:
            raise ValueError("local preregistration cell inventory differs")
        expected_cells = _expected_cells(fixture, row_ids)
    elif value.get("bounded") is False:
        expected_cells = _expected_cells(fixture, None)
    else:
        raise ValueError("local preregistration bounded flag differs")
    if cells != expected_cells:
        raise ValueError("local preregistration cell inventory differs")
    expected_inputs = {
        "fixture_file_sha256": FROZEN_FIXTURE_SHA256,
        "topology_content_sha256": fixture["topology"]["content_sha256"],
        "topology_file_sha256": fixture["scientific_identity"]["topology_file_sha256"],
        "checkpoint_sha256": fixture["scientific_identity"]["checkpoint_sha256"],
    }
    if (
        value.get("input_before") != expected_inputs
        or value.get("gates") != fixture["gates"]
    ):
        raise ValueError("local preregistration input/gate identity differs")
    return value


def assert_local_sources_stable(
    preregistration: dict[str, Any], current: dict[str, str]
) -> None:
    if current != preregistration.get("source_before"):
        raise ValueError("local source manifest changed after preregistration")


class OracleCallTracer:
    """Measure Python frames reached only while calling the frozen oracle."""

    _FORBIDDEN = (
        "gaussian_cha_profiles.py",
        "torch_continuum_r6_contact_v2.py",
        "torch_continuum_r6_derivative_v2.py",
        "torch_continuum_r6.py",
        "torch_continuum_r6_patches.py",
        "torch_continuum_chagb_domain.py",
    )

    def __init__(self):
        self.call_count = 0
        self.observed_files: set[str] = set()
        self.prohibited_files: set[str] = set()
        self._resolved_filename_cache: dict[str, str] = {}
        self.filename_resolution_count = 0

    @contextmanager
    def trace(self):
        previous = sys.getprofile()

        def profile(frame, event, arg):
            if event != "call":
                return
            self.call_count += 1
            raw_filename = frame.f_code.co_filename
            filename = self._resolved_filename_cache.get(raw_filename)
            if filename is None:
                filename = str(Path(raw_filename).resolve())
                self._resolved_filename_cache[raw_filename] = filename
                self.filename_resolution_count += 1
            self.observed_files.add(filename)
            if any(token in filename for token in self._FORBIDDEN):
                self.prohibited_files.add(filename)

        sys.setprofile(profile)
        try:
            yield self
        finally:
            sys.setprofile(previous)
        if self.prohibited_files:
            raise RuntimeError("frozen oracle reached a forbidden production frame")

    def call(self, function, *args, **kwargs):
        with self.trace():
            return function(*args, **kwargs)

    def report(self) -> dict[str, Any]:
        return {
            "measurement": "python-sys-setprofile-around-oracle-calls-only",
            "observed_call_count": self.call_count,
            "observed_source_files": sorted(self.observed_files),
            "prohibited_call_count": len(self.prohibited_files),
            "prohibited_source_files": sorted(self.prohibited_files),
            "filename_resolution_count": self.filename_resolution_count,
            "filename_cache_entries": len(self._resolved_filename_cache),
        }


@dataclass(frozen=True)
class LocalRuntimeContext:
    oracle: Any
    correction_module: Any
    typed_topology: Any
    source_origins: dict[str, str]
    tracer: OracleCallTracer
    runtime: dict[str, Any]


def testing_runtime_context() -> LocalRuntimeContext:
    return LocalRuntimeContext(
        oracle=object(),
        correction_module=object(),
        typed_topology=object(),
        source_origins={},
        tracer=OracleCallTracer(),
        runtime={
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "torch_num_threads": 1,
        },
    )


class ReferenceGeometryCache:
    """Cache independent geometry only for exact coordinates and pinned parameters."""

    def __init__(
        self,
        oracle: Any,
        topology: dict[str, Any],
        epsabs: float,
        epsrel: float,
        tracer: OracleCallTracer | None = None,
    ):
        self.oracle = oracle
        self.topology = topology
        self.epsabs = float(epsabs)
        self.epsrel = float(epsrel)
        self.tracer = tracer
        self.parameter_sha256 = hashlib.sha256(
            _canonical(
                {
                    key: topology[key]
                    for key in (
                        "effective_charges_e",
                        "cha_radii_angstrom",
                        "lj_rmin_angstrom",
                        "lj_epsilon_kcal_mol",
                    )
                }
            )
        ).hexdigest()
        self._values: dict[tuple[bytes, str, int, float, float], Any] = {}

    def get(self, positions: Any, order: int):
        array = np.asarray(positions, dtype=np.float64)
        if array.shape != (3, 3) or not np.isfinite(array).all():
            raise ValueError("reference cache coordinates must have finite shape (3,3)")
        key = (
            array.astype("<f8", copy=False).tobytes(order="C"),
            self.parameter_sha256,
            int(order),
            self.epsabs,
            self.epsrel,
        )
        if key not in self._values:
            arguments = (
                array,
                self.topology["effective_charges_e"],
                self.topology["cha_radii_angstrom"],
                self.topology["lj_rmin_angstrom"],
                self.topology["lj_epsilon_kcal_mol"],
            )
            keywords = {
                "azimuth_orders": (int(order),),
                "epsabs": self.epsabs,
                "epsrel": self.epsrel,
            }
            if self.tracer is None:
                self._values[key] = self.oracle.prepare_gaussian_reference_geometry(
                    *arguments, **keywords
                )
            else:
                self._values[key] = self.tracer.call(
                    self.oracle.prepare_gaussian_reference_geometry,
                    *arguments,
                    **keywords,
                )
        return self._values[key]


def _error_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != (3, 3) or second_array.shape != (3, 3):
        raise ValueError("force comparisons require shape (3,3)")
    difference = first_array - second_array
    if not np.isfinite(difference).all():
        raise ValueError("force comparisons must be finite")
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(difference**2))),
    }


def force_comparison_metrics(
    analytic: np.ndarray, production_fd: np.ndarray, oracle_fd: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        "ad_own_fd": _error_metrics(analytic, production_fd),
        "production_fd_oracle_fd": _error_metrics(production_fd, oracle_fd),
        "ad_oracle_fd": _error_metrics(analytic, oracle_fd),
    }


def _five_point_force(energies: Iterable[float], step: float) -> float:
    em2, em1, ep1, ep2 = [float(value) for value in energies]
    return -(em2 - 8.0 * em1 + 8.0 * ep1 - ep2) / (12.0 * float(step))


def _forces_and_disagreement_from_raw(
    raw: list[dict[str, Any]],
) -> tuple[np.ndarray, float]:
    forces = np.zeros((3, 3), dtype=np.float64)
    maximum_disagreement = 0.0
    for flat in range(9):
        estimates = [
            _five_point_force(raw[flat * 3 + step]["energies"], FD_STEPS[step])
            for step in range(3)
        ]
        atom, axis = divmod(flat, 3)
        forces[atom, axis] = estimates[-1]
        maximum_disagreement = max(
            maximum_disagreement, abs(estimates[-1] - estimates[-2])
        )
    return forces, maximum_disagreement


def derive_frozen_v1_row(
    row: dict[str, Any], fixture: dict[str, Any]
) -> dict[str, Any]:
    solvent_fd, solvent_disagreement = _forces_and_disagreement_from_raw(
        row["solvent"]["raw_fd"]
    )
    solvent_analytic = np.asarray(
        row["solvent"]["forces_kcal_mol_per_angstrom"], dtype=np.float64
    )
    solvent_force_error = solvent_fd - solvent_analytic
    energy_errors = []
    for reference in row["solvent"]["reference_energies_kcal_mol_by_order"].values():
        for component in ("polar", "cavity", "dispersion", "total"):
            energy_errors.append(
                float(row["solvent"]["components_kcal_mol"][component])
                - float(reference[component])
            )
    solvent_metrics = {
        "energy_max_abs_kcal_mol": float(np.max(np.abs(energy_errors))),
        "finest_step_disagreement_max_kcal_mol_per_angstrom": float(
            solvent_disagreement
        ),
        "force_max_abs_kcal_mol_per_angstrom": float(
            np.max(np.abs(solvent_force_error))
        ),
        "force_rms_kcal_mol_per_angstrom": float(
            np.sqrt(np.mean(solvent_force_error**2))
        ),
    }
    solvent_gates = fixture["gates"]
    solvent_passed = bool(
        solvent_metrics["energy_max_abs_kcal_mol"]
        <= solvent_gates["energy_max_abs_kcal_mol"]
        and solvent_metrics["finest_step_disagreement_max_kcal_mol_per_angstrom"]
        <= solvent_gates["finest_step_disagreement_max_kcal_mol_per_angstrom"]
        and solvent_metrics["force_max_abs_kcal_mol_per_angstrom"]
        <= solvent_gates["force_max_abs_kcal_mol_per_angstrom"]
        and solvent_metrics["force_rms_kcal_mol_per_angstrom"]
        <= solvent_gates["force_rms_kcal_mol_per_angstrom"]
    )
    combined_fd, combined_disagreement = _forces_and_disagreement_from_raw(
        row["combined"]["raw_fd"]
    )
    combined_analytic = np.asarray(
        row["combined"]["forces_hartree_per_angstrom"], dtype=np.float64
    )
    combined_force_error = combined_fd - combined_analytic
    combined_metrics = {
        "finest_step_disagreement_max_hartree_per_angstrom": float(
            combined_disagreement
        ),
        "force_max_abs_hartree_per_angstrom": float(
            np.max(np.abs(combined_force_error))
        ),
        "force_rms_hartree_per_angstrom": float(
            np.sqrt(np.mean(combined_force_error**2))
        ),
    }
    combined_gates = fixture["combined_gates"]
    combined_passed = bool(
        combined_metrics["finest_step_disagreement_max_hartree_per_angstrom"]
        <= combined_gates["finest_step_disagreement_max_hartree_per_angstrom"]
        and combined_metrics["force_max_abs_hartree_per_angstrom"]
        <= combined_gates["force_max_abs_hartree_per_angstrom"]
        and combined_metrics["force_rms_hartree_per_angstrom"]
        <= combined_gates["force_rms_hartree_per_angstrom"]
    )
    return {
        "solvent_metrics": solvent_metrics,
        "combined_metrics": combined_metrics,
        "solvent_passed": solvent_passed,
        "combined_passed": combined_passed,
        "both_passed": bool(solvent_passed and combined_passed),
    }


def summarize_v2_local(
    rows: list[dict[str, Any]], expected_row_ids: list[str] | None = None
) -> dict[str, Any]:
    if expected_row_ids is None:
        expected_row_ids = [
            f"center-{center:02d}-sigma-{width:02d}"
            for center in range(5)
            for width in range(3)
        ]
    exact_rows = [row.get("row_id") for row in rows] == expected_row_ids
    counts = {
        str(order): sum(
            bool(row.get("orders", {}).get(str(order), {}).get("passed"))
            for row in rows
        )
        for order in ORDERS
    }
    passed = exact_rows and all(count == 15 for count in counts.values())
    return {
        "row_count": len(rows),
        "inventory_exact": exact_rows,
        "order_pass_counts": counts,
        "all_passed": passed,
        "full_campaign_performed": False,
        "science_qualified": False,
        "status": "LOCAL_MATRIX_PASSED_ONLY" if passed else "GAUSSIAN_OPT_INCOMPLETE",
    }


def summarize_bounded_local(
    rows: list[dict[str, Any]],
    expected_row_ids: list[str],
    *,
    sources_stable: bool,
) -> dict[str, Any]:
    inventory_exact = (
        bool(expected_row_ids)
        and [row.get("row_id") for row in rows] == expected_row_ids
    )
    expected_cells = len(expected_row_ids) * len(ORDERS)
    completed = sum(
        bool(row.get("orders", {}).get(str(order), {}).get("passed"))
        for row in rows
        for order in ORDERS
    )
    all_passed = bool(
        inventory_exact and sources_stable and completed == expected_cells
    )
    return {
        "row_count": len(rows),
        "cell_count": expected_cells,
        "passed_cell_count": completed,
        "inventory_exact": inventory_exact,
        "source_snapshot_stable": sources_stable,
        "all_passed": all_passed,
        "full_campaign_performed": False,
        "science_qualified": False,
        "status": (
            "BOUNDED_SMOKE_PASSED_ONLY" if all_passed else "GAUSSIAN_OPT_INCOMPLETE"
        ),
    }


def local_run_succeeded(summary: dict[str, Any]) -> bool:
    return summary.get("status") in {
        "LOCAL_MATRIX_PASSED_ONLY",
        "BOUNDED_SMOKE_PASSED_ONLY",
    }


def _reference_metadata(geometry: Any) -> dict[str, Any]:
    return {
        "coordinates_sha256": geometry.coordinates_sha256,
        "parameters_sha256": geometry.parameters_sha256,
        "identity_sha256": geometry.identity_sha256,
        "payload_sha256": geometry.payload_sha256,
        "r6_levels": [
            {
                "azimuth_order": level.azimuth_order,
                "inverse_cube_quad_error_estimate_per_angstrom3": (
                    level.inverse_cube_quad_error_estimate_per_angstrom3
                ),
                "meridian_evaluations": level.meridian_evaluations,
            }
            for level in geometry.r6_levels
        ],
        "r6_diagnostics": _json_safe(geometry.r6_diagnostics),
        "nonpolar_diagnostics": _json_safe(geometry.diagnostics),
    }


def _oracle_energy(
    oracle: Any, cache: ReferenceGeometryCache, geometry: Any, sigma_e: float
):
    function = oracle.gaussian_cha_from_reference_geometry
    if cache.tracer is None:
        return function(geometry, sigma_e=sigma_e)
    return cache.tracer.call(function, geometry, sigma_e=sigma_e)


def _production_result(correction: Any, atoms: Atoms, need_forces: bool):
    result = correction.evaluate(atoms, need_forces=need_forces)
    components = {
        key: float(value) * KCAL_PER_HARTREE
        for key, value in result.components_hartree.items()
    }
    components["total"] = float(result.energy_hartree) * KCAL_PER_HARTREE
    if (
        abs(
            components["polar"]
            + components["cavity"]
            + components["dispersion"]
            - components["total"]
        )
        > 1.0e-9
    ):
        raise ValueError("production component closure exceeds 1e-9 kcal/mol")
    forces = None
    if need_forces:
        forces = (
            np.asarray(result.forces_hartree_per_angstrom, dtype=np.float64)
            * KCAL_PER_HARTREE
        )
        if forces.shape != (3, 3) or not np.isfinite(forces).all():
            raise ValueError("production analytic forces must have finite shape (3,3)")
    return components, forces, _json_safe(result.provenance)


def _validate_production_provenance(
    provenance: dict[str, Any],
    profile_id: str,
    order: int,
    *,
    topology_sha256: str,
    sigma_e: float,
) -> None:
    execution = provenance["execution"]
    diagnostics = provenance["r6_backend_diagnostics"]
    point_domain = provenance["point_domain"]
    polar_diagnostics = provenance["polar_diagnostics"]
    expected_backend = {
        V1_PROFILE_ID: "torch_continuum_chagb._r6_inverse_born",
        V2_PROFILE_ID: "torch_continuum_r6_derivative_v2._r6_inverse_born_v2",
    }[profile_id]
    if (
        provenance["numerical_profile_id"] != profile_id
        or diagnostics["profile_id"] != profile_id
        or int(diagnostics["order"]) != int(order)
        or diagnostics["backend_function"] != expected_backend
        or execution["device"] != "cpu"
        or execution["dtype"] != "torch.float64"
        or execution["programmatic_only"] is not True
        or execution["public_provider_registered"] is not False
        or provenance["model_identity"] != "chagb-r6-pbsa-gaussian-sign-v1"
        or provenance["provider"] != "gaussian-cha-continuum"
        or provenance["method"] != "gb"
        or provenance["model"] != "chagb-r6-pbsa-gaussian-sign-v1"
        or provenance["profile"] != "three-site-water-programmatic-experimental-v1"
        or provenance["topology_sha256"] != topology_sha256
        or float(provenance["sigma_e"]) != float(sigma_e)
        or int(provenance["quadrature_order"]) != int(order)
        or provenance["native_force"] is not True
        or provenance["numerical_force"] is not False
        or provenance["derivatives"]
        != "torch automatic differentiation of complete scalar"
        or provenance["production_admitted"] is not False
        or provenance["accuracy_certified"] is not False
        or provenance["physical_accuracy_claim"] is not False
        or not math.isfinite(float(polar_diagnostics["electrostatic_size_angstrom"]))
        or float(polar_diagnostics["electrostatic_size_angstrom"]) >= 9.5
        or abs(float(polar_diagnostics["component_sum_residual_hartree"]))
        * KCAL_PER_HARTREE
        > 1.0e-9
        or point_domain["scope"] != "three-site-r6-geometric-point-only"
        or point_domain["segment_certified"] is not False
    ):
        raise ValueError("production profile/dtype/domain provenance differs")


def _evaluate_local_order(
    row: dict[str, Any],
    topology: dict[str, Any],
    order: int,
    profile_id: str,
    oracle: Any,
    correction_module: Any,
    typed_topology: Any,
    cache: ReferenceGeometryCache,
    gates: dict[str, float],
) -> dict[str, Any]:
    positions = np.asarray(row["positions_angstrom"], dtype=np.float64)
    atoms = Atoms(numbers=topology["atomic_numbers"], positions=positions)
    correction = correction_module.GaussianChaCorrection(
        atoms,
        typed_topology,
        expected_topology_sha256=topology["content_sha256"],
        sigma_e=row["sigma_e"],
        order=int(order),
        numerical_profile_id=profile_id,
    )
    production_components, analytic, production_provenance = _production_result(
        correction, atoms, True
    )
    if analytic is None:
        raise RuntimeError("production analytic force evaluation returned no forces")
    _validate_production_provenance(
        production_provenance,
        profile_id,
        order,
        topology_sha256=topology["content_sha256"],
        sigma_e=row["sigma_e"],
    )
    geometry = cache.get(positions, order)
    reference = _oracle_energy(oracle, cache, geometry, row["sigma_e"])
    energy_errors = {
        key: production_components[key] - float(reference.energies_kcal_mol[key])
        for key in ("polar", "cavity", "dispersion", "total")
    }
    own_fd = np.zeros((3, 3), dtype=np.float64)
    oracle_fd = np.zeros((3, 3), dtype=np.float64)
    own_disagreement = np.zeros((3, 3), dtype=np.float64)
    oracle_disagreement = np.zeros((3, 3), dtype=np.float64)
    raw = []
    for flat in range(positions.size):
        atom, axis = divmod(flat, 3)
        own_estimates = []
        oracle_estimates = []
        stencils = []
        for step in FD_STEPS:
            own_energies = []
            oracle_energies = []
            samples = []
            for multiplier in FD_MULTIPLIERS:
                displaced = positions.copy()
                displaced[atom, axis] += multiplier * step
                atoms.set_positions(displaced)
                own_components, _, displaced_provenance = _production_result(
                    correction, atoms, False
                )
                _validate_production_provenance(
                    displaced_provenance,
                    profile_id,
                    order,
                    topology_sha256=topology["content_sha256"],
                    sigma_e=row["sigma_e"],
                )
                displaced_geometry = cache.get(displaced, REFERENCE_FD_ORDER)
                independent = _oracle_energy(
                    oracle, cache, displaced_geometry, row["sigma_e"]
                )
                own_energies.append(own_components["total"])
                oracle_energies.append(independent.total_kcal_mol)
                samples.append(
                    {
                        "multiplier": multiplier,
                        "positions_angstrom": displaced.tolist(),
                        "production_total_kcal_mol": own_components["total"],
                        "oracle_total_kcal_mol": independent.total_kcal_mol,
                        "reference_metadata": _reference_metadata(displaced_geometry),
                    }
                )
            own_estimates.append(_five_point_force(own_energies, step))
            oracle_estimates.append(_five_point_force(oracle_energies, step))
            stencils.append({"step_angstrom": step, "samples": samples})
        own_fd[atom, axis] = own_estimates[-1]
        oracle_fd[atom, axis] = oracle_estimates[-1]
        own_disagreement[atom, axis] = abs(own_estimates[-1] - own_estimates[-2])
        oracle_disagreement[atom, axis] = abs(
            oracle_estimates[-1] - oracle_estimates[-2]
        )
        raw.append(
            {
                "atom_index": atom,
                "axis": axis,
                "production_order": order,
                "reference_fd_order": REFERENCE_FD_ORDER,
                "stencils": stencils,
            }
        )
    comparisons = force_comparison_metrics(analytic, own_fd, oracle_fd)
    maximum_energy_error = max(abs(value) for value in energy_errors.values())
    maximum_step_disagreement = max(
        float(np.max(own_disagreement)), float(np.max(oracle_disagreement))
    )
    passed = bool(
        maximum_energy_error <= gates["energy_max_abs_kcal_mol"]
        and maximum_step_disagreement
        <= gates["finest_step_disagreement_max_kcal_mol_per_angstrom"]
        and all(
            metrics["maximum_absolute_error"]
            <= gates["force_max_abs_kcal_mol_per_angstrom"]
            and metrics["rms_error"] <= gates["force_rms_kcal_mol_per_angstrom"]
            for metrics in comparisons.values()
        )
    )
    return {
        "order": order,
        "production_order": order,
        "reference_fd_order": REFERENCE_FD_ORDER,
        "production_components_kcal_mol": production_components,
        "production_provenance": production_provenance,
        "analytic_forces_kcal_mol_per_angstrom": analytic.tolist(),
        "reference_energies_kcal_mol": dict(reference.energies_kcal_mol),
        "reference_metadata": _reference_metadata(geometry),
        "energy_errors_kcal_mol": energy_errors,
        "force_comparisons": comparisons,
        "own_finest_step_disagreement_max_kcal_mol_per_angstrom": float(
            np.max(own_disagreement)
        ),
        "oracle_finest_step_disagreement_max_kcal_mol_per_angstrom": float(
            np.max(oracle_disagreement)
        ),
        "raw_fd": raw,
        "passed": passed,
    }


def _load_runtime_context(
    fixture: dict[str, Any], profile_id: str
) -> LocalRuntimeContext:
    if profile_id not in {V1_PROFILE_ID, V2_PROFILE_ID}:
        raise ValueError(f"unknown closed Gaussian CHA numerical profile: {profile_id}")
    worktree_text = str(WORKTREE)
    if worktree_text not in sys.path:
        sys.path.insert(0, worktree_text)
    oracle_path = Path(__file__).with_name("cha_gaussian_reference.py")
    if _sha256_file(oracle_path) != FROZEN_ORACLE_SHA256:
        raise ValueError("frozen NumPy/SciPy oracle SHA256 changed")
    profiles = importlib.import_module(
        "maple.function.calculator.extra_correction.implicit.gaussian_cha_profiles"
    )
    profile_file = getattr(profiles, "__file__", None)
    if profile_file is None:
        raise RuntimeError("Gaussian CHA profiles have no filesystem origin")
    profile_origin = Path(profile_file).resolve()
    if not profile_origin.is_relative_to(WORKTREE):
        raise RuntimeError("Gaussian CHA profiles resolved outside the v2 worktree")
    resolved = profiles.resolve_gaussian_cha_profile(profile_id)
    if resolved.profile_id != profile_id:
        raise ValueError("closed numerical profile resolver identity differs")
    correction_module = importlib.import_module(
        "maple.function.calculator.extra_correction.implicit.gaussian_cha_correction"
    )
    inputs_module = importlib.import_module(
        "maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs"
    )
    oracle = importlib.import_module("cha_gaussian_reference")
    topology = fixture["topology"]
    typed_topology = inputs_module.ContinuumChaTopology.from_mapping(
        topology, expected_content_sha256=topology["content_sha256"]
    )
    modules = {
        "profiles": profiles,
        "correction": correction_module,
        "inputs": inputs_module,
        "oracle": oracle,
    }
    origins = {}
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"{name} module has no filesystem origin")
        origin = Path(module_file).resolve()
        if not origin.is_relative_to(WORKTREE):
            raise RuntimeError(f"{name} module resolved outside the v2 worktree")
        origins[name] = str(origin.relative_to(WORKTREE))
    return LocalRuntimeContext(
        oracle=oracle,
        correction_module=correction_module,
        typed_topology=typed_topology,
        source_origins=origins,
        tracer=OracleCallTracer(),
        runtime={
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "torch_num_threads": int(
                importlib.import_module("torch").get_num_threads()
            ),
        },
    )


def run_v2_local(
    fixture: dict[str, Any],
    *,
    profile_id: str,
    row_ids: set[str] | None = None,
    runtime_context: LocalRuntimeContext | None = None,
    cell_callback=None,
) -> dict[str, Any]:
    if profile_id not in {V1_PROFILE_ID, V2_PROFILE_ID}:
        raise ValueError(f"unknown closed Gaussian CHA numerical profile: {profile_id}")
    selected = [
        row for row in fixture["rows"] if row_ids is None or row["row_id"] in row_ids
    ]
    if row_ids is not None and {row["row_id"] for row in selected} != row_ids:
        raise ValueError("requested row IDs differ from frozen fixture")
    setup_error = None
    try:
        context = (
            _load_runtime_context(fixture, profile_id)
            if runtime_context is None
            else runtime_context
        )
    except Exception as exc:
        setup_error = exc
        context = testing_runtime_context()
    topology = fixture["topology"]
    cache = ReferenceGeometryCache(
        context.oracle,
        topology,
        fixture["scientific_identity"]["reference_epsabs"],
        fixture["scientific_identity"]["reference_epsrel"],
        tracer=context.tracer,
    )
    expected_runtime = {
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "torch_num_threads": 1,
    }
    if setup_error is None and context.runtime != expected_runtime:
        setup_error = ValueError("local runtime thread identity differs")
    rows = []
    for row in selected:
        order_results = {}
        for order in ORDERS:
            try:
                if setup_error is not None:
                    raise RuntimeError(
                        f"runtime setup failed: {type(setup_error).__name__}: {setup_error}"
                    )
                record = {
                    "status": "COMPLETED",
                    **_evaluate_local_order(
                        row,
                        topology,
                        order,
                        profile_id,
                        context.oracle,
                        context.correction_module,
                        context.typed_topology,
                        cache,
                        fixture["gates"],
                    ),
                }
            except Exception as exc:
                record = {
                    "status": "FAILED",
                    "order": order,
                    "production_order": order,
                    "reference_fd_order": REFERENCE_FD_ORDER,
                    "passed": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            order_results[str(order)] = record
            if cell_callback is not None:
                cell_callback(row, order, record)
        rows.append({"row_id": row["row_id"], "orders": order_results})
    expected_ids = [row["row_id"] for row in selected]
    summary = (
        summarize_v2_local(rows, expected_ids)
        if row_ids is None
        else summarize_bounded_local(rows, expected_ids, sources_stable=True)
    )
    return {
        "schema": "route1-cha-r6-v2-local-validation-v1",
        "numerical_profile_id": profile_id,
        "fixture_content_sha256": fixture["content_sha256"],
        "frozen_oracle_sha256": FROZEN_ORACLE_SHA256,
        "oracle_incidental_torch_package_import_disclosed": True,
        "oracle_no_call_measurement": context.tracer.report(),
        "active_source_origins": context.source_origins,
        "runtime_setup_error": (
            None
            if setup_error is None
            else {"type": type(setup_error).__name__, "message": str(setup_error)}
        ),
        "runtime": context.runtime,
        "units": {
            "energy": "kcal/mol",
            "force": "kcal/mol/angstrom",
            "coordinates": "angstrom",
        },
        "rows": rows,
        "summary": summary,
    }


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value) + b"\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _read_pinned_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} external SHA256 differs")
    return json.loads(raw)


def preregister_local(
    fixture: dict[str, Any],
    output: Path,
    selected_row_ids: set[str] | None,
    profile_id: str,
) -> dict[str, Any]:
    preregistration = build_local_preregistration(fixture, selected_row_ids, profile_id)
    _write_new(output / "preregistration.json", preregistration)
    return preregistration


def run_local_campaign(
    fixture: dict[str, Any],
    output: Path,
    expected_preregistration_sha256: str,
    *,
    profile_id: str,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, Any]:
    preregistration = load_local_preregistration(
        output / "preregistration.json", expected_preregistration_sha256
    )
    if preregistration.get("numerical_profile_id") != profile_id:
        raise ValueError("run-local numerical profile differs from preregistration")
    before = current_source_manifest()
    assert_local_sources_stable(preregistration, before)
    if current_dependency_versions() != preregistration["dependency_versions"]:
        raise ValueError("local dependency versions changed after preregistration")
    if _sha256_file(fixture_path) != FROZEN_FIXTURE_SHA256:
        raise ValueError("local input fixture changed before run")
    selected_ids = []
    for cell in preregistration["cells"]:
        if cell["row_id"] not in selected_ids:
            selected_ids.append(cell["row_id"])

    def write_cell(row: dict[str, Any], order: int, record: dict[str, Any]) -> None:
        identity = next(
            cell
            for cell in preregistration["cells"]
            if cell["row_id"] == row["row_id"]
            and int(cell["production_order"]) == int(order)
            and int(cell["reference_fd_order"]) == REFERENCE_FD_ORDER
        )
        receipt = {
            "schema": "route1-cha-r6-v2-local-cell-v1",
            "preregistration_sha256": expected_preregistration_sha256,
            "numerical_profile_id": profile_id,
            "cell_identity": identity,
            "result": record,
        }
        receipt["content_sha256"] = _content_sha256(receipt)
        _write_new(output / "cells" / f"{row['row_id']}-order-{order}.json", receipt)

    result = run_v2_local(
        fixture,
        profile_id=profile_id,
        row_ids=(None if not preregistration["bounded"] else set(selected_ids)),
        cell_callback=write_cell,
    )
    after = current_source_manifest()
    assert_local_sources_stable(preregistration, after)
    if current_dependency_versions() != preregistration["dependency_versions"]:
        raise ValueError("local dependency versions changed during run")
    if _sha256_file(fixture_path) != FROZEN_FIXTURE_SHA256:
        raise ValueError("local input fixture changed during run")
    origins_stable = (
        result["active_source_origins"]
        == preregistration["expected_active_source_origins"]
    )
    if not origins_stable and result["summary"].get("all_passed"):
        raise ValueError("active production/oracle module origins differ")
    input_after = {
        **preregistration["input_before"],
        "fixture_file_sha256": _sha256_file(fixture_path),
    }
    if input_after != preregistration["input_before"]:
        raise ValueError("local input manifest changed during run")
    summary = {
        **result["summary"],
        "schema": "route1-cha-r6-v2-local-summary-v1",
        "preregistration_sha256": expected_preregistration_sha256,
        "numerical_profile_id": profile_id,
        "source_before": before,
        "source_after": after,
        "dependency_versions": current_dependency_versions(),
        "input_before": preregistration["input_before"],
        "input_after": input_after,
        "active_source_origins": result["active_source_origins"],
        "active_source_origins_stable": origins_stable,
        "runtime_setup_error": result["runtime_setup_error"],
        "runtime": result["runtime"],
        "runtime_contract_passed": result["runtime"]
        == preregistration["runtime_contract"],
        "units": result["units"],
        "unit_contract_passed": result["units"] == preregistration["unit_contract"],
        "oracle_no_call_measurement": result["oracle_no_call_measurement"],
    }
    summary["content_sha256"] = _content_sha256(summary)
    _write_new(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare-v1")
    compare.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--expected-actual-sha256", required=True)
    compare.add_argument("--output", type=Path)
    preregister = sub.add_parser("preregister")
    preregister.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    preregister.add_argument(
        "--profile", choices=(V1_PROFILE_ID, V2_PROFILE_ID), required=True
    )
    preregister.add_argument("--row-id", action="append")
    preregister.add_argument("--output", type=Path, required=True)
    local = sub.add_parser("run-local")
    local.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    local.add_argument(
        "--profile", choices=(V1_PROFILE_ID, V2_PROFILE_ID), required=True
    )
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--expected-preregistration-sha256", required=True)
    args = parser.parse_args(argv)
    fixture = load_fixture(args.fixture)
    if args.command == "compare-v1":
        actual = _read_pinned_json(
            args.actual, args.expected_actual_sha256, "actual v1 replay"
        )
        result = compare_v1_replay(actual, fixture, evidence_root=args.actual.parent)
        if args.output:
            _write_new(args.output, result)
    elif args.command == "preregister":
        result = preregister_local(
            fixture,
            args.output,
            None if args.row_id is None else set(args.row_id),
            args.profile,
        )
    else:
        result = run_local_campaign(
            fixture,
            args.output,
            args.expected_preregistration_sha256,
            profile_id=args.profile,
            fixture_path=args.fixture,
        )
    print(json.dumps(_json_safe(result), sort_keys=True))
    if args.command == "compare-v1":
        passed = result["passed"]
    elif args.command == "preregister":
        passed = result.get("schema") == "route1-cha-r6-v2-local-preregistration-v1"
    else:
        passed = local_run_succeeded(result)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
