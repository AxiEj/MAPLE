#!/usr/bin/env python3
"""Build a label-free exposed-surface electrostatics artifact for Route 1.

The artifact is a numerical diagnostic only.  It reads frozen coordinates and
the exact normalized AM1-BCC vectors, but no experimental hydration values,
published explicit-solvent components, Route 1 errors, or tail classifications.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
from maple.function.calculator.extra_correction.implicit.common import (  # noqa: E402
    build_openmm_topology,
)
from maple.function.calculator.extra_correction.implicit.openmm_compat import (  # noqa: E402
    openmm_version,
)
from maple.function.calculator.extra_correction.implicit.radii import (  # noqa: E402
    OpenMMAmberGBRadiusProvider,
    OpenMMMbondi2RadiusProvider,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

DEFAULT_PROTOCOL = SCRIPT_DIR / "route1_surface_electrostatics_protocol_v1.json"
DEFAULT_SOURCE_MANIFEST = SCRIPT_DIR / "apbs_ace_source_manifest.json"
DEFAULT_SOURCE_ROOT = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "route1-freesolv-surface-electrostatics-label-free-2026-07-29.json"
)

ARTIFACT_TYPE = "route1-freesolv-label-free-surface-electrostatics"
PROFILE_ORDER = ("primary", "control")
DESCRIPTOR_KEYS = (
    "surface_area_angstrom2",
    "phi2_e_per_angstrom",
    "fn2_e_per_angstrom2",
    "gamma_n",
)
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "calc",
        "calc_charging",
        "calc_vdw",
        "experimental",
        "experimental_kcal_mol",
        "label",
        "mismatch",
        "route_error",
        "signed_error",
        "tail",
        "target",
    }
)


def _load_object(path: Path) -> dict[str, Any]:
    value = core.load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return int(value)


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _strict_relative_path(root: Path, relative: object) -> Path:
    text = str(relative)
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe source-relative path: {text!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Source path escapes its frozen root: {text!r}") from exc
    return resolved


def _reject_forbidden_label_keys(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_LABEL_KEYS:
                raise ValueError(
                    f"Label-free source contains forbidden key {key!r} at {path}."
                )
            _reject_forbidden_label_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_label_keys(item, path=f"{path}[{index}]")


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = _load_object(Path(path).resolve())
    if protocol.get("schema_version") != 1:
        raise ValueError("Only surface-electrostatics protocol schema 1 is supported.")
    if (
        protocol.get("protocol_id")
        != "maple-route1-surface-electrostatics-diagnostic-v1"
    ):
        raise ValueError("Unexpected surface-electrostatics protocol id.")

    boundary = protocol.get("phase_boundaries", {})
    expected_boundary = {
        "label_free_surface_phase_reads_experimental_or_published_component_labels": False,
        "label_free_surface_phase_reads_existing_score_or_mismatch_artifacts": False,
        "retrospective_join_phase_reads_published_component_mismatch": True,
        "retrospective_identity_source_contains_experimental_labels": True,
        "published_explicit_component_is_experimental_truth": False,
        "experimental_hydration_free_energy_is_used": False,
        "no_fit": True,
        "no_residual": True,
        "no_endpoint_selection": True,
        "no_radius_selection": True,
        "no_threshold_tuning": True,
        "no_force_claim": True,
        "no_first_shell_causal_claim": True,
        "no_ranking_certification": True,
    }
    if boundary != expected_boundary:
        raise ValueError("Surface-electrostatics phase boundary changed.")

    evidence = protocol.get("source_evidence", {})
    if evidence.get("expected_case_count") != 526:
        raise ValueError("Surface diagnostic case count changed.")
    if evidence.get("source_partition") != "development":
        raise ValueError("Surface diagnostic must remain development-only.")
    for key in (
        "label_free_source_manifest_sha256",
        "retrospective_component_artifact_sha256",
        "retrospective_component_artifact_content_sha256",
        "prepared_identity_artifact_sha256",
    ):
        if not core._is_hex(str(evidence.get(key, "")), 64):
            raise ValueError(f"source_evidence.{key} must be a lowercase SHA256.")

    surface = protocol.get("surface_definition", {})
    if surface.get("name") != "deterministic-fibonacci-shrake-rupley-sas-proxy":
        raise ValueError("Surface definition changed.")
    if _finite(surface.get("probe_radius_angstrom"), field="probe") != 1.4:
        raise ValueError("Surface probe radius changed.")
    if (
        _finite(
            surface.get("squared_distance_tie_tolerance_angstrom2"),
            field="tie tolerance",
        )
        != 1e-12
    ):
        raise ValueError("Surface accessibility tie tolerance changed.")
    if (
        _finite(
            surface.get("minimum_pair_distance_angstrom"),
            field="minimum pair distance",
        )
        != 1e-10
    ):
        raise ValueError("Minimum pair distance changed.")
    if (
        _finite(
            surface.get("maximum_absolute_total_charge_e"),
            field="charge tolerance",
        )
        != 1e-10
    ):
        raise ValueError("Neutral-charge tolerance changed.")
    if (
        _finite(
            surface.get("minimum_field_variance_e2_per_angstrom4"),
            field="field variance",
        )
        != 1e-30
    ):
        raise ValueError("Minimum field-variance gate changed.")

    quadrature = surface.get("quadrature", {})
    if quadrature.get("generator_id") != "maple-fibonacci-sphere-v1":
        raise ValueError("Quadrature generator changed.")
    if (
        _positive_int(
            quadrature.get("coarse_direction_count"), field="coarse count"
        )
        != 1024
        or _positive_int(
            quadrature.get("fine_direction_count"), field="fine count"
        )
        != 4096
    ):
        raise ValueError("Quadrature direction counts changed.")
    for key in ("coarse_direction_sha256", "fine_direction_sha256"):
        if not core._is_hex(str(quadrature.get(key, "")), 64):
            raise ValueError(f"surface_definition.quadrature.{key} is invalid.")

    profiles = protocol.get("radius_profiles", {})
    if profiles.get("selection_between_profiles_allowed") is not False:
        raise ValueError("Radius-profile selection cannot be enabled.")
    expected_profiles = {
        "primary": ("gbn-bondi", "bondi", "OpenMMAmberGBRadiusProvider('gbn')"),
        "control": (
            "generic-mbondi2",
            "mbondi2",
            "OpenMMMbondi2RadiusProvider",
        ),
    }
    for role, expected in expected_profiles.items():
        profile = profiles.get(role, {})
        if tuple(profile.get(key) for key in ("profile_id", "radii", "provider")) != expected:
            raise ValueError(f"Radius profile {role!r} changed.")

    descriptors = protocol.get("descriptors", {})
    if set(descriptors) != set(DESCRIPTOR_KEYS) | {
        "additional_descriptors_allowed",
        "zero_field_variance_action",
    }:
        raise ValueError("Surface descriptor membership changed.")
    if (
        descriptors.get("additional_descriptors_allowed") is not False
        or descriptors.get("zero_field_variance_action") != "fail_closed"
    ):
        raise ValueError("Surface descriptor boundary changed.")

    validation = protocol.get("numerical_validation", {})
    if validation.get("all_records_require_coarse_to_fine_check") is not True:
        raise ValueError("All records must retain quadrature-refinement checks.")
    if validation.get("any_gate_failure_action") != (
        "seal_label_free_artifact_as_invalid_and_do_not_compute_"
        "retrospective_associations"
    ):
        raise ValueError("Numerical failure action changed.")
    qa_ids = validation.get("qa_subset_compound_ids")
    if not isinstance(qa_ids, list) or len(qa_ids) != 16 or len(set(qa_ids)) != 16:
        raise ValueError("QA subset must contain 16 unique compound IDs.")
    rotations = validation.get("proper_rotations")
    if not isinstance(rotations, list) or len(rotations) != 3:
        raise ValueError("Exactly three frozen proper rotations are required.")
    rotation_hash = core.sha256_bytes(core.canonical_json_bytes(rotations))
    if rotation_hash != validation.get("proper_rotations_sha256"):
        raise ValueError("Proper-rotation fingerprint mismatch.")

    decision = protocol.get("pre_registered_decision_rule", {})
    for key in (
        "first_shell_causality_established",
        "energy_correction_allowed",
        "provider_implementation_allowed",
        "endpoint_selection_allowed",
        "radius_selection_allowed",
        "parameter_update_allowed",
        "ranking_certification_allowed",
        "multisolvent_claim_allowed",
        "maximum_error_claim_allowed",
    ):
        if decision.get(key) is not False:
            raise ValueError(f"Unsupported action enabled: {key}.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


@lru_cache(maxsize=4)
def fibonacci_directions(count: int) -> np.ndarray:
    """Return the frozen equal-area Fibonacci sphere directions."""
    count = _positive_int(count, field="direction count")
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / float(count)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * indices
    directions = np.column_stack(
        (radial * np.cos(theta), radial * np.sin(theta), z)
    )
    directions.setflags(write=False)
    return directions


def _array_hash(values: np.ndarray | Sequence[object]) -> str:
    array = np.asarray(values)
    return core.sha256_bytes(core.canonical_json_bytes(array.tolist()))


def _relative_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(right)), 1e-15)


def surface_descriptors(
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    radii_angstrom: np.ndarray,
    directions: np.ndarray,
    *,
    probe_radius_angstrom: float,
    squared_distance_tie_tolerance_angstrom2: float,
    minimum_pair_distance_angstrom: float,
    minimum_field_variance_e2_per_angstrom4: float,
) -> dict[str, Any]:
    """Evaluate the four pre-registered descriptors on one SAS proxy."""
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    charges = np.asarray(charges_e, dtype=np.float64)
    radii = np.asarray(radii_angstrom, dtype=np.float64)
    unit_directions = np.asarray(directions, dtype=np.float64)
    atom_count = len(positions)
    if positions.shape != (atom_count, 3):
        raise ValueError("Positions must have shape (N, 3).")
    if charges.shape != (atom_count,) or radii.shape != (atom_count,):
        raise ValueError("Charge/radius vector length does not match positions.")
    if (
        atom_count == 0
        or unit_directions.ndim != 2
        or unit_directions.shape[1] != 3
        or not np.isfinite(positions).all()
        or not np.isfinite(charges).all()
        or not np.isfinite(radii).all()
        or not np.isfinite(unit_directions).all()
    ):
        raise ValueError("Surface descriptor input is empty or non-finite.")
    if np.any(radii <= 0.0):
        raise ValueError("Intrinsic surface radii must be positive.")
    if not np.allclose(
        np.linalg.norm(unit_directions, axis=1),
        1.0,
        rtol=0.0,
        atol=5e-15,
    ):
        raise ValueError("Surface directions are not unit vectors.")

    if atom_count > 1:
        pair_distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        upper = pair_distances[np.triu_indices(atom_count, 1)]
        if np.any(upper < minimum_pair_distance_angstrom):
            raise ValueError("Coincident or near-coincident atom centers detected.")

    expanded = radii + float(probe_radius_angstrom)
    direction_count = len(unit_directions)
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    accessible_counts: list[int] = []
    all_indices = np.arange(atom_count)
    for atom_index, (center, expanded_radius) in enumerate(
        zip(positions, expanded)
    ):
        points = center + expanded_radius * unit_directions
        other = all_indices != atom_index
        if np.any(other):
            squared_distances = np.sum(
                (points[:, None, :] - positions[None, :, :]) ** 2,
                axis=2,
            )
            accessible = np.all(
                squared_distances[:, other]
                >= (
                    expanded[None, other] ** 2
                    - squared_distance_tie_tolerance_angstrom2
                ),
                axis=1,
            )
        else:
            accessible = np.ones(direction_count, dtype=bool)
        accessible_count = int(np.count_nonzero(accessible))
        accessible_counts.append(accessible_count)
        if accessible_count == 0:
            continue

        points = points[accessible]
        normals = unit_directions[accessible]
        displacements = points[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(displacements, axis=2)
        if np.any(distances < minimum_pair_distance_angstrom):
            raise ValueError("Surface evaluation encountered a singular charge distance.")
        phi = np.sum(charges[None, :] / distances, axis=1)
        field = np.sum(
            charges[None, :, None]
            * displacements
            / distances[:, :, None] ** 3,
            axis=1,
        )
        normal_field = np.sum(field * normals, axis=1)
        values.append(np.column_stack((phi, normal_field)))
        weights.append(
            np.full(
                accessible_count,
                4.0 * np.pi * expanded_radius**2 / direction_count,
                dtype=np.float64,
            )
        )

    if not values:
        raise ValueError("Surface quadrature contains no accessible points.")
    stacked_values = np.concatenate(values)
    stacked_weights = np.concatenate(weights)
    area = float(math.fsum(stacked_weights.tolist()))
    if area <= 0.0 or not math.isfinite(area):
        raise ValueError("Surface area is not positive and finite.")
    phi = stacked_values[:, 0]
    normal_field = stacked_values[:, 1]
    phi2 = math.sqrt(float(np.sum(stacked_weights * phi * phi) / area))
    fn2 = math.sqrt(
        float(np.sum(stacked_weights * normal_field * normal_field) / area)
    )
    field_mean = float(np.sum(stacked_weights * normal_field) / area)
    centered = normal_field - field_mean
    field_variance = float(np.sum(stacked_weights * centered**2) / area)
    if (
        not math.isfinite(field_variance)
        or field_variance <= minimum_field_variance_e2_per_angstrom4
    ):
        raise ValueError("Surface normal-field variance is undefined or too small.")
    gamma = float(
        (np.sum(stacked_weights * centered**3) / area)
        / field_variance**1.5
    )
    descriptor_values = (area, phi2, fn2, gamma)
    if not all(math.isfinite(value) for value in descriptor_values):
        raise ValueError("Surface descriptor is non-finite.")
    return {
        "surface_area_angstrom2": area,
        "phi2_e_per_angstrom": phi2,
        "fn2_e_per_angstrom2": fn2,
        "gamma_n": gamma,
        "accessible_point_count": int(len(stacked_weights)),
        "accessible_point_count_by_atom": accessible_counts,
        "normal_field_mean_e_per_angstrom2": field_mean,
        "normal_field_variance_e2_per_angstrom4": field_variance,
    }


def _descriptor_view(result: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(result[key]) for key in DESCRIPTOR_KEYS}


def _profile_radii(topology) -> dict[str, Any]:
    return {
        "primary": OpenMMAmberGBRadiusProvider("gbn").assign(topology),
        "control": OpenMMMbondi2RadiusProvider().assign(topology),
    }


def _load_label_free_records(
    protocol: Mapping[str, Any],
    manifest_path: Path,
    source_root: Path,
) -> list[dict[str, Any]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(manifest_path) != evidence["label_free_source_manifest_sha256"]:
        raise ValueError("Frozen label-free source-manifest hash mismatch.")
    manifest = _load_object(manifest_path)
    if set(manifest) != set(evidence["required_manifest_keys"]):
        raise ValueError("Label-free source-manifest key set changed.")
    _reject_forbidden_label_keys(manifest)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_partition") != evidence["source_partition"]
        or manifest.get("charge_method") != "am1bcc"
    ):
        raise ValueError("Label-free source-manifest identity changed.")
    records = manifest.get("records")
    expected = int(evidence["expected_case_count"])
    if (
        manifest.get("case_count") != expected
        or not isinstance(records, list)
        or len(records) != expected
    ):
        raise ValueError("Label-free source-manifest coverage changed.")

    required_record_keys = set(evidence["required_record_keys"])
    output: list[dict[str, Any]] = []
    compound_ids: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict) or set(raw) != required_record_keys:
            raise ValueError("Label-free source record key set changed.")
        compound_id = str(raw.get("compound_id", ""))
        if not compound_id or compound_id in compound_ids:
            raise ValueError("Label-free compound IDs must be unique and nonempty.")
        compound_ids.add(compound_id)
        source_path = _strict_relative_path(
            source_root, raw["source_mol2_relative_path"]
        )
        if not source_path.is_file():
            raise FileNotFoundError(f"Frozen source MOL2 is missing: {source_path}")
        if core.sha256_file(source_path) != raw["source_mol2_sha256"]:
            raise ValueError(f"Frozen source MOL2 hash mismatch for {compound_id}.")
        charges = np.asarray(raw["am1bcc_charges_e"], dtype=np.float64)
        if charges.ndim != 1 or len(charges) == 0 or not np.isfinite(charges).all():
            raise ValueError(f"Invalid exact charge vector for {compound_id}.")
        output.append(
            {
                "compound_id": compound_id,
                "source_path": source_path,
                "source_mol2_relative_path": str(raw["source_mol2_relative_path"]),
                "source_mol2_sha256": str(raw["source_mol2_sha256"]),
                "charges_e": charges,
            }
        )
    return output


def _convergence_check(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    relative = {
        key: _relative_error(float(coarse[key]), float(fine[key]))
        for key in DESCRIPTOR_KEYS[:3]
    }
    gamma_absolute = abs(float(coarse["gamma_n"]) - float(fine["gamma_n"]))
    thresholds = validation["maximum_coarse_to_fine_relative_error"]
    passed = all(relative[key] <= float(thresholds[key]) for key in relative)
    passed = passed and (
        gamma_absolute
        <= float(validation["maximum_coarse_to_fine_absolute_error_gamma_n"])
    )
    return {
        "relative_error": relative,
        "absolute_error_gamma_n": gamma_absolute,
        "passed": bool(passed),
    }


def _transform_check(
    reference: Mapping[str, Any],
    transformed: Mapping[str, Any],
    *,
    maximum_relative_error: float,
    maximum_absolute_error_gamma_n: float,
) -> dict[str, Any]:
    relative = {
        key: _relative_error(float(transformed[key]), float(reference[key]))
        for key in DESCRIPTOR_KEYS[:3]
    }
    gamma_absolute = abs(
        float(transformed["gamma_n"]) - float(reference["gamma_n"])
    )
    return {
        "relative_error": relative,
        "absolute_error_gamma_n": gamma_absolute,
        "passed": bool(
            max(relative.values()) <= maximum_relative_error
            and gamma_absolute <= maximum_absolute_error_gamma_n
        ),
    }


def _charge_inversion_check(
    reference: Mapping[str, Any],
    inverted: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    relative = {
        key: _relative_error(float(inverted[key]), float(reference[key]))
        for key in DESCRIPTOR_KEYS[:3]
    }
    gamma_parity_error = abs(
        float(inverted["gamma_n"]) + float(reference["gamma_n"])
    )
    return {
        "relative_error_even_descriptors": relative,
        "absolute_gamma_odd_parity_error": gamma_parity_error,
        "passed": bool(
            max(relative.values())
            <= float(validation["charge_inversion_relative_tolerance"])
            and gamma_parity_error
            <= float(
                validation["charge_inversion_absolute_tolerance_gamma_n"]
            )
        ),
    }


def _qa_for_profile(
    positions: np.ndarray,
    charges: np.ndarray,
    radii: np.ndarray,
    reference: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    surface = protocol["surface_definition"]
    validation = protocol["numerical_validation"]
    fine_count = int(surface["quadrature"]["fine_direction_count"])
    directions = fibonacci_directions(fine_count)
    kwargs = {
        "probe_radius_angstrom": float(surface["probe_radius_angstrom"]),
        "squared_distance_tie_tolerance_angstrom2": float(
            surface["squared_distance_tie_tolerance_angstrom2"]
        ),
        "minimum_pair_distance_angstrom": float(
            surface["minimum_pair_distance_angstrom"]
        ),
        "minimum_field_variance_e2_per_angstrom4": float(
            surface["minimum_field_variance_e2_per_angstrom4"]
        ),
    }
    translation = np.asarray(
        validation["translation_vector_angstrom"], dtype=np.float64
    )
    translated = surface_descriptors(
        positions + translation, charges, radii, directions, **kwargs
    )
    permutation = np.arange(len(positions) - 1, -1, -1)
    permuted = surface_descriptors(
        positions[permutation],
        charges[permutation],
        radii[permutation],
        directions,
        **kwargs,
    )
    inverted = surface_descriptors(
        positions, -charges, radii, directions, **kwargs
    )
    exact_relative = float(
        validation["maximum_translation_or_permutation_relative_error"]
    )
    exact_gamma = float(
        validation["maximum_translation_or_permutation_absolute_error_gamma_n"]
    )
    translation_check = _transform_check(
        reference,
        translated,
        maximum_relative_error=exact_relative,
        maximum_absolute_error_gamma_n=exact_gamma,
    )
    permutation_check = _transform_check(
        reference,
        permuted,
        maximum_relative_error=exact_relative,
        maximum_absolute_error_gamma_n=exact_gamma,
    )
    rotation_checks = []
    rotation_relative = validation["maximum_rotation_relative_error"]
    for rotation in validation["proper_rotations"]:
        matrix = np.asarray(rotation["matrix"], dtype=np.float64)
        rotated = surface_descriptors(
            positions @ matrix.T,
            charges,
            radii,
            directions,
            **kwargs,
        )
        check = _transform_check(
            reference,
            rotated,
            maximum_relative_error=max(
                float(rotation_relative[key]) for key in DESCRIPTOR_KEYS[:3]
            ),
            maximum_absolute_error_gamma_n=float(
                validation["maximum_rotation_absolute_error_gamma_n"]
            ),
        )
        # Preserve the descriptor-specific thresholds in the final decision.
        check["passed"] = bool(
            all(
                check["relative_error"][key] <= float(rotation_relative[key])
                for key in DESCRIPTOR_KEYS[:3]
            )
            and check["absolute_error_gamma_n"]
            <= float(validation["maximum_rotation_absolute_error_gamma_n"])
        )
        rotation_checks.append(
            {
                "axis": rotation["axis"],
                "angle_radian": rotation["angle_radian"],
                **check,
            }
        )
    inversion_check = _charge_inversion_check(reference, inverted, validation)
    passed = (
        translation_check["passed"]
        and permutation_check["passed"]
        and inversion_check["passed"]
        and all(check["passed"] for check in rotation_checks)
    )
    return {
        "translation": translation_check,
        "atom_permutation": permutation_check,
        "charge_inversion": inversion_check,
        "proper_rotations": rotation_checks,
        "passed": bool(passed),
    }


def build(
    *,
    protocol_path: Path,
    source_manifest_path: Path,
    source_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(protocol_path)
    surface = protocol["surface_definition"]
    validation = protocol["numerical_validation"]
    records = _load_label_free_records(
        protocol, source_manifest_path.resolve(), source_root.resolve()
    )
    expected_qa = list(validation["qa_subset_compound_ids"])
    actual_qa = [
        compound_id
        for _, compound_id in sorted(
            (
                core.sha256_bytes(record["compound_id"].encode("utf-8")),
                record["compound_id"],
            )
            for record in records
        )[: len(expected_qa)]
    ]
    if actual_qa != expected_qa:
        raise ValueError("Frozen label-free QA subset selection does not reproduce.")

    quadrature = surface["quadrature"]
    coarse_directions = fibonacci_directions(
        int(quadrature["coarse_direction_count"])
    )
    fine_directions = fibonacci_directions(int(quadrature["fine_direction_count"]))
    if _array_hash(coarse_directions) != quadrature["coarse_direction_sha256"]:
        raise ValueError("Coarse Fibonacci-direction fingerprint mismatch.")
    if _array_hash(fine_directions) != quadrature["fine_direction_sha256"]:
        raise ValueError("Fine Fibonacci-direction fingerprint mismatch.")

    descriptor_kwargs = {
        "probe_radius_angstrom": float(surface["probe_radius_angstrom"]),
        "squared_distance_tie_tolerance_angstrom2": float(
            surface["squared_distance_tie_tolerance_angstrom2"]
        ),
        "minimum_pair_distance_angstrom": float(
            surface["minimum_pair_distance_angstrom"]
        ),
        "minimum_field_variance_e2_per_angstrom4": float(
            surface["minimum_field_variance_e2_per_angstrom4"]
        ),
    }
    result_records: list[dict[str, Any]] = []
    qa_inputs: dict[str, dict[str, Any]] = {}
    convergence_failures: list[dict[str, str]] = []
    maximum_convergence = {
        role: {
            **{key: 0.0 for key in DESCRIPTOR_KEYS[:3]},
            "absolute_error_gamma_n": 0.0,
        }
        for role in PROFILE_ORDER
    }

    for source_record in records:
        compound_id = source_record["compound_id"]
        atoms = MOL2Reader(str(source_record["source_path"]))
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        charges = np.asarray(source_record["charges_e"], dtype=np.float64)
        if len(charges) != len(atoms):
            raise ValueError(
                f"Exact charge-vector length mismatch for {compound_id}."
            )
        total_charge = float(math.fsum(charges.tolist()))
        if abs(total_charge) > float(surface["maximum_absolute_total_charge_e"]):
            raise ValueError(f"Expected a neutral normalized charge vector: {compound_id}.")
        topology = build_openmm_topology(atoms)
        radius_results = _profile_radii(topology)
        profile_records: dict[str, Any] = {}
        profile_qa_inputs: dict[str, Any] = {}
        for role in PROFILE_ORDER:
            radius_result = radius_results[role]
            expected_profile = protocol["radius_profiles"][role]
            if (
                radius_result.profile != expected_profile["profile_id"]
                or radius_result.provenance.get("radii")
                != expected_profile["radii"]
            ):
                raise ValueError(
                    f"Radius provider/profile mismatch for {compound_id}/{role}."
                )
            radii = np.asarray(radius_result.radii_angstrom, dtype=np.float64)
            coarse = surface_descriptors(
                positions,
                charges,
                radii,
                coarse_directions,
                **descriptor_kwargs,
            )
            fine = surface_descriptors(
                positions,
                charges,
                radii,
                fine_directions,
                **descriptor_kwargs,
            )
            convergence = _convergence_check(coarse, fine, validation)
            for key, value in convergence["relative_error"].items():
                maximum_convergence[role][key] = max(
                    maximum_convergence[role][key], float(value)
                )
            maximum_convergence[role]["absolute_error_gamma_n"] = max(
                maximum_convergence[role]["absolute_error_gamma_n"],
                float(convergence["absolute_error_gamma_n"]),
            )
            if not convergence["passed"]:
                convergence_failures.append(
                    {"compound_id": compound_id, "radius_profile_role": role}
                )
            profile_records[role] = {
                "profile_id": radius_result.profile,
                "radius_vector_sha256": _array_hash(radii),
                "radius_provider_provenance": radius_result.provenance,
                "descriptors": _descriptor_view(fine),
                "accessible_point_count": fine["accessible_point_count"],
                "normal_field_mean_e_per_angstrom2": fine[
                    "normal_field_mean_e_per_angstrom2"
                ],
                "normal_field_variance_e2_per_angstrom4": fine[
                    "normal_field_variance_e2_per_angstrom4"
                ],
                "quadrature_convergence": convergence,
            }
            if compound_id in expected_qa:
                profile_qa_inputs[role] = {
                    "radii": radii,
                    "reference": fine,
                }

        result_records.append(
            {
                "compound_id": compound_id,
                "atom_count": len(atoms),
                "source_mol2_relative_path": source_record[
                    "source_mol2_relative_path"
                ],
                "source_mol2_sha256": source_record["source_mol2_sha256"],
                "coordinate_vector_sha256": _array_hash(positions),
                "atom_symbol_vector_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(atoms.get_chemical_symbols())
                ),
                "exact_am1bcc_charge_vector_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(charges.tolist())
                ),
                "total_charge_e": total_charge,
                "radius_profiles": profile_records,
            }
        )
        if compound_id in expected_qa:
            qa_inputs[compound_id] = {
                "positions": positions,
                "charges": charges,
                "profiles": profile_qa_inputs,
            }

    qa_records: list[dict[str, Any]] = []
    qa_failures: list[dict[str, str]] = []
    for compound_id in expected_qa:
        inputs = qa_inputs[compound_id]
        profile_checks: dict[str, Any] = {}
        for role in PROFILE_ORDER:
            profile = inputs["profiles"][role]
            checks = _qa_for_profile(
                inputs["positions"],
                inputs["charges"],
                profile["radii"],
                profile["reference"],
                protocol,
            )
            profile_checks[role] = checks
            if not checks["passed"]:
                qa_failures.append(
                    {"compound_id": compound_id, "radius_profile_role": role}
                )
        qa_records.append(
            {"compound_id": compound_id, "radius_profiles": profile_checks}
        )

    numerical_passed = not convergence_failures and not qa_failures
    command_arguments = {
        "phase": "label-free-surface",
        "protocol": _relative_to_repository(protocol_path),
        "source_manifest": _relative_to_repository(source_manifest_path),
        "source_root": _relative_to_repository(source_root),
        "output": _relative_to_repository(output_path),
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "claim_scope": protocol["claim_scope"],
        "source_partition": protocol["source_evidence"]["source_partition"],
        "case_count": len(result_records),
        "label_use_boundary": {
            "surface_phase_reads_labels": False,
            "surface_phase_reads_score_or_mismatch_artifacts": False,
            "source_manifest_contains_exact_normalized_am1bcc_vectors": True,
            "source_mol2_charge_tokens_used": False,
        },
        "source_provenance": {
            "source_manifest": _relative_to_repository(source_manifest_path),
            "source_manifest_sha256": core.sha256_file(source_manifest_path),
            "source_root": _relative_to_repository(source_root),
        },
        "surface_definition": protocol["surface_definition"],
        "radius_profiles": protocol["radius_profiles"],
        "descriptors": protocol["descriptors"],
        "mathematical_invariants": protocol["mathematical_invariants"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "openmm_version": openmm_version(),
        },
        "numerical_validation": {
            "passed": numerical_passed,
            "quadrature_convergence_failure_count": len(convergence_failures),
            "quadrature_convergence_failures": convergence_failures,
            "maximum_quadrature_convergence_error": maximum_convergence,
            "qa_failure_count": len(qa_failures),
            "qa_failures": qa_failures,
            "qa_records": qa_records,
        },
        "records": sorted(result_records, key=lambda row: row["compound_id"]),
        "decision": {
            "status": (
                "label_free_surface_artifact_valid_for_retrospective_join"
                if numerical_passed
                else "invalid_surface_artifact_no_association"
            ),
            "retrospective_association_allowed": numerical_passed,
            "first_shell_causality_established": False,
            "energy_correction_allowed": False,
            "provider_implementation_allowed": False,
            "endpoint_selection_allowed": False,
            "radius_selection_allowed": False,
            "ranking_certification_allowed": False,
        },
        "command_provenance": core.command_provenance(
            __file__,
            command_arguments,
            repository_root=REPOSITORY_ROOT,
        ),
    }
    core.seal_artifact(artifact)
    core.write_json_atomic(output_path, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build(
        protocol_path=args.protocol.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        source_root=args.source_root.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(artifact["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
