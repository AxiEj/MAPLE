"""Tolerance-bounded replay audit for matched QM/PCMSolver references.

Raw hashes remain useful artifact identities, but a converged multithreaded SCF
is not required to reproduce every floating-point bit.  This module therefore
separates exact scientific identity (method, geometry, cavity, source graph)
from preregistered numerical replay tolerances.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

REPLAY_AUDIT_VERSION = "route2-matched-qm-pcmsolver-replay-audit-v1"

_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
_ENERGY_FIELDS = (
    "solute_distortion_energy_eV",
    "continuum_stabilization_energy_eV",
    "total_electrostatic_solvation_energy_eV",
    "pcm_electrostatic_total_energy_eV",
    "polarized_density_vacuum_energy_eV",
)
_REFERENCE_IDENTITY_FIELDS = (
    "cavity_profile_id",
    "continuum_configuration_sha256",
    "continuum_equation_id",
    "continuum_protocol_sha256",
    "continuum_provenance_sha256",
    "contract_version",
    "dielectric",
    "electrostatics_only",
    "experimental_solvation_labels_used",
    "fixed_nuclear_geometry",
    "geometry_sha256",
    "molecule_group_sha256",
    "nonpolar_terms_included",
    "pcm_density_reoptimized_in_vacuum",
    "pcm_density_self_consistent",
    "qm_runtime_manifest_sha256",
    "reference_method_id",
    "reference_protocol_sha256",
    "spin_multiplicity",
    "standard_state_terms_included",
    "topology_sha256",
    "total_charge",
    "vacuum_density_self_consistent",
    "vacuum_density_sha256",
)


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {name} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain one JSON object.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size < 1 or not np.issubdtype(array.dtype, np.number):
        raise RuntimeError(f"{name} must be a non-empty numeric array.")
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} must be finite.")
    return array


def _max_abs_difference(first: object, second: object, *, name: str) -> float:
    left = _finite_array(first, name=f"first {name}")
    right = _finite_array(second, name=f"second {name}")
    if left.shape != right.shape:
        raise RuntimeError(f"{name} shape changed across replay.")
    return float(np.max(np.abs(left - right)))


def _case_map(result: Mapping[str, object], *, name: str) -> dict[str, dict]:
    records = result.get("case_records")
    if not isinstance(records, list):
        raise RuntimeError(f"{name} omits case_records.")
    mapped: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("case_id"), str):
            raise RuntimeError(f"{name} has an invalid case record.")
        case_id = record["case_id"]
        if case_id in mapped:
            raise RuntimeError(f"{name} repeats case {case_id!r}.")
        mapped[case_id] = record
    return mapped


def _validate_complete_run(
    run_dir: Path,
    *,
    preregistration: Mapping[str, object],
    preregistration_path: Path,
    name: str,
) -> tuple[dict[str, object], dict[str, dict]]:
    result_path = run_dir / "result.json"
    result = _load_json(result_path, name=f"{name} result")
    expected_cases = sorted(preregistration["case_ids"])
    if (
        result.get("schema_version")
        != "route2-matched-qm-pcmsolver-decomposition-run-v3"
        or result.get("status") != "pass-complete-matched-electrostatic-reference-panel"
        or result.get("panel_complete") is not True
        or result.get("capabilities") != _CAPABILITIES
        or sorted(result.get("requested_case_ids", [])) != expected_cases
        or sorted(result.get("full_preregistered_case_ids", [])) != expected_cases
    ):
        raise RuntimeError(f"{name} is not a complete fail-closed v3 panel run.")
    preregistration_record = result.get("preregistration")
    if not isinstance(preregistration_record, dict) or (
        preregistration_record.get("sha256") != _sha256_file(preregistration_path)
    ):
        raise RuntimeError(f"{name} does not bind the active preregistration.")
    records = _case_map(result, name=name)
    if sorted(records) != expected_cases:
        raise RuntimeError(f"{name} case set differs from the preregistration.")
    for case_id, record in records.items():
        case_path = run_dir / case_id / "case.json"
        artifact = record.get("case_artifact")
        if not isinstance(artifact, dict) or artifact.get("sha256") != _sha256_file(
            case_path
        ):
            raise RuntimeError(f"{name} case artifact hash changed for {case_id}.")
    return result, records


def _require_equal(first: object, second: object, *, name: str) -> None:
    if first != second:
        raise RuntimeError(f"Exact replay identity changed: {name}.")


def audit_complete_reference_replay(
    first_run: str | Path,
    second_run: str | Path,
    preregistration_path: str | Path,
) -> dict[str, object]:
    """Audit two complete panel runs under the frozen v3 replay contract."""

    first_dir = Path(first_run).expanduser().resolve(strict=True)
    second_dir = Path(second_run).expanduser().resolve(strict=True)
    if first_dir == second_dir:
        raise ValueError("Replay audit requires two distinct run directories.")
    prereg_path = Path(preregistration_path).expanduser().resolve(strict=True)
    preregistration = _load_json(prereg_path, name="preregistration")
    if (
        preregistration.get("protocol_id")
        != "route2-matched-qm-pcmsolver-four-prereg-v3"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("Replay audit requires the frozen v3 preregistration.")
    gates = preregistration.get("numerical_gates")
    if not isinstance(gates, dict):
        raise RuntimeError("Preregistration omits numerical gates.")

    first_result, first_records = _validate_complete_run(
        first_dir,
        preregistration=preregistration,
        preregistration_path=prereg_path,
        name="first run",
    )
    second_result, second_records = _validate_complete_run(
        second_dir,
        preregistration=preregistration,
        preregistration_path=prereg_path,
        name="second run",
    )
    for field in (
        "repository",
        "runtime",
        "pcmsolver_library",
        "source_sha256",
        "requested_case_ids",
        "full_preregistered_case_ids",
    ):
        _require_equal(
            first_result.get(field), second_result.get(field), name=f"run.{field}"
        )

    maxima = {
        "energy_abs_eV": 0.0,
        "pcm_density_inf": 0.0,
        "boundary_mep_inf_hartree_per_e": 0.0,
        "asc_inf_e": 0.0,
        "polarization_energy_abs_hartree": 0.0,
    }
    per_case: dict[str, object] = {}
    stable_case_identity: dict[str, object] = {}
    for case_id in sorted(first_records):
        first_case = _load_json(first_dir / case_id / "case.json", name=case_id)
        second_case = _load_json(
            second_dir / case_id / "case.json", name=f"second {case_id}"
        )
        if (
            first_case.get("status") != "pass-matched-electrostatic-reference-case"
            or second_case.get("status") != "pass-matched-electrostatic-reference-case"
            or first_case.get("capabilities") != _CAPABILITIES
            or second_case.get("capabilities") != _CAPABILITIES
        ):
            raise RuntimeError(f"{case_id} is not a passed fail-closed reference case.")
        first_reference = first_case.get("reference_case")
        second_reference = second_case.get("reference_case")
        if not isinstance(first_reference, dict) or not isinstance(
            second_reference, dict
        ):
            raise RuntimeError(f"{case_id} omits reference_case.")
        identity = {}
        for field in _REFERENCE_IDENTITY_FIELDS:
            _require_equal(
                first_reference.get(field),
                second_reference.get(field),
                name=f"{case_id}.{field}",
            )
            identity[field] = first_reference.get(field)
        stable_case_identity[case_id] = identity

        energy_differences = {
            field: abs(float(first_reference[field]) - float(second_reference[field]))
            for field in _ENERGY_FIELDS
        }
        maxima["energy_abs_eV"] = max(
            maxima["energy_abs_eV"], max(energy_differences.values())
        )

        first_surface = np.load(first_dir / case_id / "surface.npz")
        second_surface = np.load(second_dir / case_id / "surface.npz")
        for field in ("surface_points_bohr", "surface_areas_bohr2"):
            if not np.array_equal(first_surface[field], second_surface[field]):
                raise RuntimeError(f"Exact replay surface changed: {case_id}.{field}.")
        first_state = np.load(first_dir / case_id / "pcm-state.npz")
        second_state = np.load(second_dir / case_id / "pcm-state.npz")
        for field in ("atomic_numbers", "atom_positions_bohr"):
            if not np.array_equal(first_state[field], second_state[field]):
                raise RuntimeError(f"Exact replay state changed: {case_id}.{field}.")
        differences = {
            "pcm_density_inf": _max_abs_difference(
                first_state["ao_density_matrix"],
                second_state["ao_density_matrix"],
                name=f"{case_id} AO density",
            ),
            "boundary_mep_inf_hartree_per_e": _max_abs_difference(
                first_state["total_surface_mep_hartree_per_e"],
                second_state["total_surface_mep_hartree_per_e"],
                name=f"{case_id} boundary MEP",
            ),
            "asc_inf_e": _max_abs_difference(
                first_state["apparent_surface_charge_e"],
                second_state["apparent_surface_charge_e"],
                name=f"{case_id} ASC",
            ),
            "polarization_energy_abs_hartree": _max_abs_difference(
                first_state["polarization_energy_hartree"],
                second_state["polarization_energy_hartree"],
                name=f"{case_id} polarization energy",
            ),
        }
        for field, difference in differences.items():
            maxima[field] = max(maxima[field], difference)
        per_case[case_id] = {
            "energy_abs_eV": energy_differences,
            **differences,
        }

    gate_map = {
        "energy_abs_eV": "repeat_energy_abs_eV_max",
        "pcm_density_inf": "repeat_pcm_density_inf_max",
        "boundary_mep_inf_hartree_per_e": ("repeat_boundary_mep_inf_hartree_per_e_max"),
        "asc_inf_e": "repeat_asc_inf_e_max",
        "polarization_energy_abs_hartree": (
            "repeat_polarization_energy_abs_hartree_max"
        ),
    }
    for metric, gate_name in gate_map.items():
        limit = float(gates[gate_name])
        if not np.isfinite(limit) or limit <= 0.0 or maxima[metric] > limit:
            raise RuntimeError(
                f"Replay metric {metric}={maxima[metric]:.17g} exceeds {gate_name}."
            )

    return {
        "schema_version": REPLAY_AUDIT_VERSION,
        "status": "pass-tolerance-bounded-complete-panel-replay",
        "capabilities": dict(_CAPABILITIES),
        "claim_boundary": (
            "Source-independent matched electrostatic reference reproducibility "
            "only; no ML source, ledger, E/F/H/V/M, or production admission."
        ),
        "preregistration": {
            "protocol_id": preregistration["protocol_id"],
            "sha256": _sha256_file(prereg_path),
        },
        "runs": [
            {
                "label": "first",
                "result_json_sha256": _sha256_file(first_dir / "result.json"),
                "scientific_payload_sha256": first_result["scientific_payload_sha256"],
            },
            {
                "label": "second",
                "result_json_sha256": _sha256_file(second_dir / "result.json"),
                "scientific_payload_sha256": second_result["scientific_payload_sha256"],
            },
        ],
        "raw_scientific_payload_sha256_equal": (
            first_result["scientific_payload_sha256"]
            == second_result["scientific_payload_sha256"]
        ),
        "stable_case_identity": stable_case_identity,
        "gates": {name: float(gates[name]) for name in gate_map.values()},
        "observed_maxima": maxima,
        "per_case": per_case,
    }


__all__ = [
    "REPLAY_AUDIT_VERSION",
    "audit_complete_reference_replay",
]
