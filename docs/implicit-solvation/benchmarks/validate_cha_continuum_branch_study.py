#!/usr/bin/env python3
"""Run the preregistered fixed-water, branch-aware CHA point study.

This is validation infrastructure, not a solvent provider or an optimizer
admission.  Production forces are analytic derivatives of the same scalar.
Finite differences use only the independent reference and are validation-only.
Every geometry certificate is point-local; no row certifies a line segment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Iterable, Mapping, NamedTuple, cast

import numpy as np

SCRIPT = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT.parents[3]
BENCHMARK_DIR = SCRIPT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from cha_continuum_reference import _cha_polar, cha_continuum_reference
from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb import (
    continuum_cha_scalar,
)

ARTIFACT_CLASS = "UNREGISTERED_POINT_STUDY"
EXPECTED_PROTOCOL_SHA256 = (
    "3fe641f0e9bb8a572e7736b8f3cb3ce2ac8a8f8f29ce6e022a0f8caf805cad64"
)
COMPONENTS = ("polar", "cavity", "dispersion", "total")
SOURCE_FILES = (
    SCRIPT,
    BENCHMARK_DIR / "cha_continuum_reference.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/continuum_chagb_inputs.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_continuum_chagb.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_continuum_chagb_domain.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_continuum_r6_patches.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_continuum_ses_geometry.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_chagb.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_sphere_union_geometry.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_dense_budget.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_continuum_sav.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_continuum_dispersion.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/sphere_union_volume.py",
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/sphere_union_dispersion.py",
)


class StudyInputs(NamedTuple):
    protocol: Mapping[str, Any]
    topology: ContinuumChaTopology
    topology_sha256: str
    positions_angstrom: tuple[tuple[float, float, float], ...]
    protocol_file_sha256: str
    prepared_files: Mapping[str, str]


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + ("\n" if newline else "")).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha(value: Mapping[str, Any], field: str = "content_sha256") -> str:
    return _sha_bytes(
        _canonical(
            _jsonable({key: item for key, item in value.items() if key != field})
        )
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, type):
        raise TypeError("type objects are not JSON data values")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    if callable(detach) and callable(cpu):
        tensor = cast(Any, detach()).cpu()
        return _jsonable(tensor.item() if tensor.ndim == 0 else tensor.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(_canonical(_jsonable(value), newline=True))
    os.replace(temporary, path)


def snapshot_files(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = _sha_file(path)
    return result


def require_unchanged_snapshot(
    before: Mapping[str, str], after: Mapping[str, str]
) -> None:
    if dict(before) != dict(after):
        changed = sorted(
            set(before) ^ set(after)
            | {path for path in set(before) & set(after) if before[path] != after[path]}
        )
        raise RuntimeError(f"source identity changed during study: {changed}")


def _load_sealed(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError(f"expected JSON object: {path}")
    if artifact.get("content_sha256") != _content_sha(artifact):
        raise ValueError(f"artifact content hash mismatch: {path}")
    return artifact


def load_study_inputs(prepared_dir: Path, protocol_path: Path) -> StudyInputs:
    prepared_dir = Path(prepared_dir).resolve()
    protocol_path = Path(protocol_path).resolve()
    if _sha_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("protocol SHA256 differs from the external study pin")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "route1-cha-analytic-branch-study-v1":
        raise ValueError("unexpected branch-study protocol identity")
    topology_path = prepared_dir / "topology.json"
    coordinates_path = prepared_dir / "coordinates.json"
    record_path = prepared_dir / "preparation-record.json"
    record = _load_sealed(record_path)
    topology_artifact = json.loads(topology_path.read_text(encoding="utf-8"))
    coordinates = json.loads(coordinates_path.read_text(encoding="utf-8"))
    expected_topology = record.get("topology_content_sha256")
    if not isinstance(expected_topology, str):
        raise ValueError("prepared record is missing its external topology pin")
    if _sha_file(topology_path) != record["native_files"]["topology.json"]["sha256"]:
        raise ValueError("prepared topology file SHA256 mismatch")
    if (
        _sha_file(coordinates_path)
        != record["native_files"]["coordinates.json"]["sha256"]
    ):
        raise ValueError("prepared coordinates file SHA256 mismatch")
    topology = ContinuumChaTopology.from_mapping(
        topology_artifact,
        expected_content_sha256=expected_topology,
        expected_source_charge_sha256=topology_artifact["source_charges_sha256"],
        expected_source_mol2_sha256=protocol["source_assets"]["fixed_mol2"]["sha256"],
    )
    coordinate_payload = {
        key: item for key, item in coordinates.items() if key != "coordinate_sha256"
    }
    if coordinates.get("coordinate_sha256") != _sha_bytes(
        _canonical(coordinate_payload)
    ):
        raise ValueError("prepared coordinate content hash mismatch")
    if coordinates.get("topology_sha256") != topology.content_sha256 or coordinates.get(
        "atom_ids"
    ) != list(topology.atom_ids):
        raise ValueError("coordinate/topology identity mismatch")
    raw_positions = coordinates["positions_angstrom"]
    if len(raw_positions) != topology.atom_count or any(
        len(row) != 3 for row in raw_positions
    ):
        raise ValueError("prepared coordinates have the wrong shape")
    positions = tuple(
        (float(row[0]), float(row[1]), float(row[2])) for row in raw_positions
    )
    return StudyInputs(
        protocol=protocol,
        topology=topology,
        topology_sha256=topology.content_sha256,
        positions_angstrom=positions,
        protocol_file_sha256=EXPECTED_PROTOCOL_SHA256,
        prepared_files={
            "topology.json": _sha_file(topology_path),
            "coordinates.json": _sha_file(coordinates_path),
            "preparation-record.json": _sha_file(record_path),
        },
    )


def scaled_water_geometry(
    positions: tuple[tuple[float, float, float], ...], scale: float
) -> np.ndarray:
    source = np.asarray(positions, dtype=np.float64)
    if source.shape != (3, 3) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "water scaling requires finite [3,3] coordinates and positive scale"
        )
    result = source.copy()
    result[1:] = source[0] + scale * (source[1:] - source[0])
    return result


def five_point_force(
    energies_by_displacement: Mapping[float, float], h: float
) -> float:
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("five-point step must be finite and positive")
    try:
        derivative = (
            float(energies_by_displacement[-2.0 * h])
            - 8.0 * float(energies_by_displacement[-h])
            + 8.0 * float(energies_by_displacement[h])
            - float(energies_by_displacement[2.0 * h])
        ) / (12.0 * h)
    except KeyError as exc:
        raise ValueError("five-point stencil is incomplete") from exc
    return -derivative


def five_point_force_uncertainty(errors: Mapping[float, float], step: float) -> float:
    """Propagate four scalar estimates into force units; not a rigorous bound."""
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    values = np.asarray(
        [errors[offset] for offset in (-2 * step, -step, step, 2 * step)]
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("scalar uncertainties must be finite and nonnegative")
    return float(np.dot([1.0, 8.0, 8.0, 1.0], values) / (12.0 * step))


def _signs(values: Iterable[float]) -> tuple[int, ...]:
    result = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("CHA weighted signs must be finite")
        result.append(0 if number == 0.0 else (1 if number > 0.0 else -1))
    return tuple(result)


def classify_denominators(
    center_weighted_signs: Iterable[float],
    denominators: Iterable[Mapping[str, Any]],
    expected_directions: Iterable[tuple[int, int]],
    *,
    center_topology_signature: Mapping[str, Any],
    center_size_branch: str,
) -> dict[str, Any]:
    center = _signs(center_weighted_signs)
    supplied = {
        (int(row["atom_index"]), int(row["axis"])): dict(row) for row in denominators
    }
    retained: list[dict[str, Any]] = []
    events = 0
    failures = 0
    for atom_index, axis in expected_directions:
        key = (atom_index, axis)
        if key not in supplied:
            failures += 1
            retained.append(
                {
                    "atom_index": atom_index,
                    "axis": axis,
                    "classification": "MISSING_DENOMINATOR",
                    "error_type": "MissingDenominator",
                    "error": "preregistered Cartesian denominator was not evaluated",
                    "samples": [],
                }
            )
            continue
        row = supplied[key]
        samples = row.get("samples", [])
        sample_failure = next(
            (
                sample
                for sample in samples
                if sample.get("status") not in {"success", "branch_event"}
            ),
            None,
        )
        branch_exception = any(
            sample.get("status") == "branch_event" for sample in samples
        )
        sign_event = False
        topology_event = False
        size_event = False
        if sample_failure is None:
            for sample in samples:
                if sample.get("status") == "success":
                    if (
                        "r6_topology_signature" not in sample
                        or "cha_size_branch" not in sample
                    ):
                        sample_failure = sample
                        sample["error_type"] = "MissingBranchSignature"
                        break
                    sample_signs = _signs(sample["weighted_signs_e"])
                    changed = sample_signs != center
                    sample["weighted_signs_relative_to_center"] = (
                        "CHANGED" if changed else "SAME"
                    )
                    sign_event = sign_event or changed
                    topology_event = (
                        topology_event
                        or sample["r6_topology_signature"] != center_topology_signature
                    )
                    size_event = (
                        size_event or sample["cha_size_branch"] != center_size_branch
                    )
        if sample_failure is not None:
            failures += 1
            classification = "POINT_FAILURE"
        elif branch_exception or sign_event or topology_event or size_event:
            events += 1
            classification = "SIGN_EVENT" if sign_event else "BRANCH_EVENT"
        else:
            classification = "REGULAR"
        row["classification"] = classification
        row["event_kinds"] = [
            name
            for name, observed in (
                ("weighted_charge_sign", sign_event),
                ("r6_topology", topology_event),
                ("electrostatic_size", size_event),
                ("guard_exception", branch_exception),
            )
            if observed
        ]
        retained.append(row)
    if failures:
        status = "FAILED"
    elif events:
        status = "EVENT"
    else:
        status = "REGULAR"
    return {
        "status": status,
        "denominator_count": len(retained),
        "event_denominator_count": events,
        "failed_denominator_count": failures,
        "force_gate_applicable": status == "REGULAR",
        "denominators": retained,
    }


def reference_scalar_budget(
    *,
    inverse_cube_96: np.ndarray,
    inverse_cube_128: np.ndarray,
    inverse_cube_quadrature_error_per_angstrom3: float,
    polar_energy_from_inverse_cube: Callable[[np.ndarray], float],
    cavity_volume_error_angstrom3: float,
    dispersion_error_kcal_mol: float,
    gate_kcal_mol: float,
) -> dict[str, Any]:
    level96 = np.asarray(inverse_cube_96, dtype=float)
    level128 = np.asarray(inverse_cube_128, dtype=float)
    if (
        level96.shape != level128.shape
        or level96.ndim != 1
        or not 1 <= len(level96) <= 3
        or not np.isfinite(level96).all()
        or not np.isfinite(level128).all()
        or np.any(level96 <= 0.0)
        or np.any(level128 <= 0.0)
    ):
        raise ValueError(
            "R6 refinement vectors must be finite positive one-to-three-site arrays"
        )
    errors = (
        inverse_cube_quadrature_error_per_angstrom3,
        cavity_volume_error_angstrom3,
        dispersion_error_kcal_mol,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in errors):
        raise ValueError("Reference error estimates must be finite and nonnegative")
    if not math.isfinite(gate_kcal_mol) or gate_kcal_mol <= 0.0:
        raise ValueError("Reference budget gate must be finite and positive")
    refinement_delta = np.abs(level128 - level96)
    delta = refinement_delta + inverse_cube_quadrature_error_per_angstrom3
    finest = float(polar_energy_from_inverse_cube(level128))
    observed = abs(finest - float(polar_energy_from_inverse_cube(level96)))
    if not math.isfinite(finest) or not math.isfinite(observed):
        raise ValueError("Reference polar energies must be finite")
    envelope = observed
    for choices in itertools.product((-1.0, 1.0), repeat=len(delta)):
        perturbed = level128 + np.asarray(choices) * delta
        if np.any(perturbed <= 0.0):
            raise ValueError("Reference uncertainty reaches nonpositive inverse cube")
        displaced_energy = float(polar_energy_from_inverse_cube(perturbed))
        if not math.isfinite(displaced_energy):
            raise ValueError("Reference uncertainty energies must be finite")
        envelope = max(envelope, abs(displaced_energy - finest))
    cavity_error = float(cavity_volume_error_angstrom3) * 0.0378
    dispersion_error = float(dispersion_error_kcal_mol)
    combined = envelope + cavity_error + dispersion_error
    within = bool(math.isfinite(combined) and combined <= gate_kcal_mol)
    return {
        "gate_kcal_mol": gate_kcal_mol,
        "r6_raw_inverse_cube_difference_recorded_only": float(np.max(refinement_delta)),
        "r6_quadrature_error_estimate_per_angstrom3": float(
            inverse_cube_quadrature_error_per_angstrom3
        ),
        "inverse_cube_units": "angstrom^-3; never compared directly to kcal/mol",
        "polar_96_vs_128_kcal_mol": observed,
        "polar_energy_sensitivity_envelope_kcal_mol": envelope,
        "cavity_error_kcal_mol": cavity_error,
        "dispersion_error_kcal_mol": dispersion_error,
        "combined_estimate_kcal_mol": combined,
        "within_numeric_gate": within,
        "rigorous_bound": False,
        "method": "96-to-128 inverse-cube refinement plus meridian quadrature estimate, propagated through independent CHA polar energy over all uncertainty corners; NOT RIGOROUS",
        "admission_status": (
            "UNRESOLVED_NONRIGOROUS_ESTIMATE"
            if within
            else "UNRESOLVED_REFERENCE_BUDGET"
        ),
    }


def _exception_record(exc: BaseException) -> dict[str, Any]:
    result = {
        "status": "failure",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    failure = getattr(exc, "failure", None)
    if failure is not None:
        result["typed_failure"] = _jsonable(failure)
    for name in ("reason", "raw_margins"):
        if hasattr(exc, name):
            result[name] = _jsonable(getattr(exc, name))
    if type(exc).__name__ == "ContinuumChaBranchError":
        result["status"] = "branch_event"
    return result


def _production_point(
    positions: np.ndarray,
    topology: ContinuumChaTopology,
    topology_sha256: str,
    order: int,
    *,
    need_forces: bool = True,
) -> dict[str, Any]:
    import torch

    coordinates = torch.tensor(
        positions, dtype=torch.float64, requires_grad=need_forces
    )
    result = continuum_cha_scalar(
        coordinates,
        topology,
        expected_topology_sha256=topology_sha256,
        order=order,
    )
    tensors = {
        "polar": result.polar_kcal_mol,
        "cavity": result.cavity_kcal_mol,
        "dispersion": result.dispersion_kcal_mol,
        "total": result.total_kcal_mol,
    }
    forces = {}
    if need_forces:
        for index, (name, energy) in enumerate(tensors.items()):
            gradient = torch.autograd.grad(
                energy,
                coordinates,
                retain_graph=index < len(tensors) - 1,
                create_graph=False,
            )[0]
            forces[name] = (-gradient).detach().cpu().tolist()
    energy_values = {name: float(value.detach()) for name, value in tensors.items()}
    force_arrays = {name: np.asarray(value) for name, value in forces.items()}
    force_identity_residual = (
        float(
            np.max(
                np.abs(
                    force_arrays["total"]
                    - force_arrays["polar"]
                    - force_arrays["cavity"]
                    - force_arrays["dispersion"]
                )
            )
        )
        if need_forces
        else None
    )
    return {
        "status": "success",
        "order": order,
        "energies_kcal_mol": energy_values,
        "analytic_forces_kcal_mol_per_angstrom": forces if need_forces else None,
        "analytic_forces_evaluated": need_forces,
        "same_point_single_scalar_graph": need_forces,
        "component_identity": {
            "energy_sum_residual_kcal_mol": abs(
                energy_values["total"]
                - energy_values["polar"]
                - energy_values["cavity"]
                - energy_values["dispersion"]
            ),
            "force_sum_max_abs_residual_kcal_mol_per_angstrom": force_identity_residual,
        },
        "weighted_signs_e": result.cha.weighted_signs_e.detach().cpu().tolist(),
        "electrostatic_size_angstrom": float(
            result.cha.electrostatic_size_angstrom.detach()
        ),
        "cha_size_branch": (
            "below-10A"
            if float(result.cha.electrostatic_size_angstrom.detach()) < 10.0
            else "above-10A"
        ),
        "r6_topology_signature": {
            "active_pairs": _jsonable(result.point_domain.active_pairs),
            "fully_occluded_pairs": _jsonable(result.point_domain.fully_occluded_pairs),
        },
        "point_domain": _jsonable(result.point_domain),
        "whole_segment_certified": False,
        "quadrature_identity": _jsonable(result.quadrature_identity),
        "resources": _jsonable(result.resources),
    }


def _reference_point(
    positions: np.ndarray, topology: ContinuumChaTopology
) -> tuple[dict[str, Any], Any]:
    result = cha_continuum_reference(
        positions,
        topology.effective_charges_e,
        topology.cha_radii_angstrom,
        topology.lj_rmin_angstrom,
        topology.lj_epsilon_kcal_mol,
        azimuth_orders=(64, 96, 128),
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )
    record = {
        "status": "success",
        "energies_kcal_mol": {
            name: float(getattr(result, f"{name}_kcal_mol")) for name in COMPONENTS
        },
        "effective_charges_e": result.effective_charges_e.tolist(),
        "diagnostics": _jsonable(result.diagnostics),
        "r6": {
            "inverse_cube_per_angstrom3": result.r6.inverse_cube_per_angstrom3.tolist(),
            "inverse_born_per_angstrom": result.r6.inverse_born_per_angstrom.tolist(),
            "gauss_closure_vector_angstrom2": result.r6.gauss_closure_vector_angstrom2.tolist(),
            "diagnostics": _jsonable(result.r6.diagnostics),
            "levels": [_jsonable(level) for level in result.r6.levels],
        },
    }
    return record, result


def _center_budget(
    reference_result: Any,
    positions: np.ndarray,
    topology: ContinuumChaTopology,
    gate: float,
) -> dict[str, Any]:
    levels = {level.azimuth_order: level for level in reference_result.r6.levels}
    cavity_error = reference_result.diagnostics.get(
        "cavity_scalar_error_estimate_angstrom3"
    )
    dispersion_error = reference_result.diagnostics.get(
        "dispersion_scalar_error_estimate_kcal_mol"
    )
    if cavity_error is None or dispersion_error is None:
        return {
            "admission_status": "UNRESOLVED_SCALAR_ERROR_UNAVAILABLE",
            "within_numeric_gate": None,
            "combined_estimate_kcal_mol": None,
            "rigorous_bound": False,
            "unavailable_components": [
                name
                for name, value in (
                    ("cavity", cavity_error),
                    ("dispersion", dispersion_error),
                )
                if value is None
            ],
            "method": "Scalar-only uncertainty is unavailable; mixed scalar/gradient norms are not energy-error estimates.",
        }

    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in (cavity_error, dispersion_error)
    ):
        raise ValueError("Scalar-only error estimates must be finite and nonnegative")
    cavity_difference = float(
        reference_result.diagnostics["cavity_generic_minus_scalar_reference_angstrom3"]
    )
    dispersion_difference = float(
        reference_result.diagnostics[
            "dispersion_generic_minus_scalar_reference_kcal_mol"
        ]
    )
    if not math.isfinite(cavity_difference) or not math.isfinite(dispersion_difference):
        raise ValueError("Generic-versus-scalar reference discrepancies must be finite")
    cavity_error = float(cavity_error) + abs(cavity_difference)
    dispersion_error = float(dispersion_error) + abs(dispersion_difference)

    def polar(inverse_cube: np.ndarray) -> float:
        state = _cha_polar(
            positions,
            np.asarray(topology.effective_charges_e),
            np.asarray(topology.cha_radii_angstrom),
            np.cbrt(inverse_cube),
        )
        return float(state.polar_kcal_mol)

    return reference_scalar_budget(
        inverse_cube_96=levels[96].inverse_cube_per_angstrom3,
        inverse_cube_128=levels[128].inverse_cube_per_angstrom3,
        inverse_cube_quadrature_error_per_angstrom3=levels[
            128
        ].inverse_cube_quad_error_estimate_per_angstrom3,
        polar_energy_from_inverse_cube=polar,
        cavity_volume_error_angstrom3=cavity_error,
        dispersion_error_kcal_mol=dispersion_error,
        gate_kcal_mol=gate,
    )


def _metrics(matrix: np.ndarray) -> dict[str, float]:
    absolute = np.abs(matrix)
    return {
        "max_abs_kcal_mol_per_angstrom": float(np.max(absolute)),
        "rms_kcal_mol_per_angstrom": float(np.sqrt(np.mean(matrix**2))),
    }


def force_comparison_metrics(
    errors: np.ndarray, *, max_gate: float, rms_gate: float
) -> dict[str, Any]:
    """Require each component to pass; zero-error components cannot dilute RMS."""
    if errors.shape != (3, 3, len(COMPONENTS)) or not np.isfinite(errors).all():
        raise ValueError("Force comparison requires every finite Cartesian component")
    components = {}
    for index, name in enumerate(COMPONENTS):
        metrics = _metrics(errors[:, :, index])
        metrics["passed"] = bool(
            metrics["max_abs_kcal_mol_per_angstrom"] <= max_gate
            and metrics["rms_kcal_mol_per_angstrom"] <= rms_gate
        )
        components[name] = metrics
    return {
        **_metrics(errors),
        "per_component": components,
        "passed": all(metrics["passed"] for metrics in components.values()),
    }


def _evaluate_stencil(
    center: np.ndarray,
    center_production: Mapping[str, Any],
    topology: ContinuumChaTopology,
    topology_sha256: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    steps = tuple(float(value) for value in protocol["fd_steps_angstrom"])
    directions = [tuple(value) for value in protocol["cartesian_directions"]]
    denominators: list[dict[str, Any]] = []
    reference_forces = {
        str(step): np.full((3, 3, len(COMPONENTS)), np.nan) for step in steps
    }
    reference_uncertainties = {
        str(step): np.full((3, 3, len(COMPONENTS)), np.nan) for step in steps
    }
    budget_fields = {
        "polar": "polar_energy_sensitivity_envelope_kcal_mol",
        "cavity": "cavity_error_kcal_mol",
        "dispersion": "dispersion_error_kcal_mol",
        "total": "combined_estimate_kcal_mol",
    }
    for atom_index, axis in directions:
        offsets = sorted(
            {
                sign * factor * step
                for step in steps
                for factor in (1.0, 2.0)
                for sign in (-1.0, 1.0)
            }
        )
        samples = []
        by_offset: dict[float, dict[str, Any]] = {}
        for offset in offsets:
            displaced = center.copy()
            displaced[atom_index, axis] += offset
            sample: dict[str, Any] = {
                "displacement_angstrom": offset,
                "whole_segment_certified": False,
            }
            try:
                sample.update(
                    _production_point(
                        displaced, topology, topology_sha256, 64, need_forces=False
                    )
                )
            except Exception as exc:
                sample.update(_exception_record(exc))
            if sample["status"] == "success":
                try:
                    reference, reference_result = _reference_point(displaced, topology)
                    sample["independent_reference"] = reference
                    if _signs(reference["effective_charges_e"]) != _signs(
                        sample["weighted_signs_e"]
                    ):
                        sample["status"] = "reference_failure"
                        sample["error_type"] = "ReferenceBranchMismatch"
                        sample["error"] = (
                            "Independent reference and production disagree on this sample's CHA sign branch"
                        )
                    try:
                        sample["reference_scalar_budget"] = _center_budget(
                            reference_result,
                            displaced,
                            topology,
                            float(protocol["gates"]["reference_scalar_kcal_mol"]),
                        )
                    except Exception as budget_error:
                        sample["reference_scalar_budget"] = {
                            "admission_status": "UNRESOLVED_REFERENCE_BUDGET",
                            "within_numeric_gate": False,
                            "rigorous_bound": False,
                            "failure": _exception_record(budget_error),
                        }
                except Exception as exc:
                    sample["independent_reference"] = _exception_record(exc)
                    sample["status"] = "reference_failure"
                    sample["error_type"] = type(exc).__name__
                    sample["error"] = str(exc)
            samples.append(sample)
            by_offset[offset] = sample
        denominators.append(
            {"atom_index": atom_index, "axis": axis, "samples": samples}
        )
        if all(sample.get("status") == "success" for sample in samples):
            for step in steps:
                for component_index, component in enumerate(COMPONENTS):
                    energies = {
                        offset: by_offset[offset]["independent_reference"][
                            "energies_kcal_mol"
                        ][component]
                        for offset in (-2.0 * step, -step, step, 2.0 * step)
                    }
                    reference_forces[str(step)][atom_index, axis, component_index] = (
                        five_point_force(energies, step)
                    )
                    error_values = {
                        offset: by_offset[offset]
                        .get("reference_scalar_budget", {})
                        .get(budget_fields[component])
                        for offset in (-2.0 * step, -step, step, 2.0 * step)
                    }
                    if all(value is not None for value in error_values.values()):
                        reference_uncertainties[str(step)][
                            atom_index, axis, component_index
                        ] = five_point_force_uncertainty(error_values, step)
    classified = classify_denominators(
        center_production["weighted_signs_e"],
        denominators,
        directions,
        center_topology_signature=center_production["r6_topology_signature"],
        center_size_branch=center_production["cha_size_branch"],
    )
    classified["reference_forces_kcal_mol_per_angstrom"] = {
        step: values.tolist() for step, values in reference_forces.items()
    }
    classified["reference_force_uncertainty_estimates_kcal_mol_per_angstrom"] = {
        step: values.tolist() for step, values in reference_uncertainties.items()
    }
    scalar_budgets_passed = all(
        sample.get("reference_scalar_budget", {}).get("within_numeric_gate") is True
        for denominator in denominators
        for sample in denominator["samples"]
    )
    if classified["status"] != "REGULAR":
        classified["force_gate"] = {
            "applicable": False,
            "reason": (
                "event rows are not force failures"
                if classified["status"] == "EVENT"
                else "one or more denominators failed"
            ),
        }
        return classified

    production = np.stack(
        [
            np.asarray(
                center_production["analytic_forces_kcal_mol_per_angstrom"][component]
            )
            for component in COMPONENTS
        ],
        axis=-1,
    )
    errors = {step: values - production for step, values in reference_forces.items()}
    finest, coarse = str(min(steps)), str(max(steps))
    disagreement = reference_forces[finest] - reference_forces[coarse]
    gates = protocol["gates"]
    per_step = {}
    budgeted_per_step = {}
    for step, matrix in errors.items():
        metrics = force_comparison_metrics(
            matrix,
            max_gate=gates["force_max_kcal_mol_per_angstrom"],
            rms_gate=gates["force_rms_kcal_mol_per_angstrom"],
        )
        per_step[step] = metrics
        uncertainty = reference_uncertainties[step]
        if np.isfinite(uncertainty).all():
            budgeted_per_step[step] = force_comparison_metrics(
                np.abs(matrix) + uncertainty + np.abs(disagreement),
                max_gate=gates["force_max_kcal_mol_per_angstrom"],
                rms_gate=gates["force_rms_kcal_mol_per_angstrom"],
            )
        else:
            budgeted_per_step[step] = {
                "passed": False,
                "status": "UNRESOLVED_REFERENCE_UNCERTAINTY",
            }
    pair_metrics = _metrics(disagreement)
    pair_metrics["passed"] = (
        pair_metrics["max_abs_kcal_mol_per_angstrom"]
        <= gates["two_step_disagreement_kcal_mol_per_angstrom"]
    )
    classified["force_gate"] = {
        "applicable": True,
        "raw_error_matrices": {
            step: matrix.tolist() for step, matrix in errors.items()
        },
        "per_step": per_step,
        "budgeted_per_step": budgeted_per_step,
        "scalar_reference_budgets_passed": scalar_budgets_passed,
        "uncertainty_method": "Observed absolute AD/FD error + propagated four-scalar estimate + two-step disagreement; NOT a rigorous bound",
        "two_step_disagreement": pair_metrics,
        "passed": scalar_budgets_passed
        and all(row["passed"] for row in budgeted_per_step.values())
        and pair_metrics["passed"],
    }
    return classified


def _evaluate_center(
    scale: float,
    inputs: StudyInputs,
    *,
    centers_only: bool,
) -> dict[str, Any]:
    center = scaled_water_geometry(inputs.positions_angstrom, scale)
    row: dict[str, Any] = {
        "scale": scale,
        "artifact_class": ARTIFACT_CLASS,
        "public_admission": False,
        "whole_segment_certified": False,
        "positions_angstrom": center.tolist(),
        "production": {},
    }
    for order in inputs.protocol["refinement_orders"]:
        try:
            row["production"][str(order)] = _production_point(
                center, inputs.topology, inputs.topology_sha256, int(order)
            )
        except Exception as exc:
            row["production"][str(order)] = _exception_record(exc)
    try:
        reference_record, reference_result = _reference_point(center, inputs.topology)
        row["independent_reference"] = reference_record
    except Exception as exc:
        row["independent_reference"] = _exception_record(exc)
        row["reference_scalar_budget"] = {
            "admission_status": "UNRESOLVED_REFERENCE_FAILURE",
            "rigorous_bound": False,
        }
    else:
        try:
            row["reference_scalar_budget"] = _center_budget(
                reference_result,
                center,
                inputs.topology,
                float(inputs.protocol["gates"]["reference_scalar_kcal_mol"]),
            )
        except Exception as exc:
            row["reference_scalar_budget"] = {
                "admission_status": "UNRESOLVED_REFERENCE_BUDGET",
                "rigorous_bound": False,
                "failure": _exception_record(exc),
            }
    finest = row["production"].get("64", {})
    reference = row["independent_reference"]
    if finest.get("status") == "success" and reference.get("status") == "success":
        deltas = {
            component: finest["energies_kcal_mol"][component]
            - reference["energies_kcal_mol"][component]
            for component in COMPONENTS
        }
        maximum = max(abs(value) for value in deltas.values())
        row["source_energy_gate"] = {
            "component_deltas_kcal_mol": deltas,
            "maximum_absolute_delta_kcal_mol": maximum,
            "threshold_kcal_mol": inputs.protocol["gates"][
                "production_reference_scalar_kcal_mol"
            ],
            "passed": maximum
            <= inputs.protocol["gates"]["production_reference_scalar_kcal_mol"],
        }
    else:
        row["source_energy_gate"] = {"passed": False, "status": "UNRESOLVED"}
    if centers_only:
        row["row_status"] = "CENTER_ONLY"
        row["stencil"] = {"status": "NOT_EXECUTED_CENTERS_ONLY"}
    elif finest.get("status") == "success":
        row["stencil"] = _evaluate_stencil(
            center, finest, inputs.topology, inputs.topology_sha256, inputs.protocol
        )
        row["row_status"] = row["stencil"]["status"]
    else:
        row["row_status"] = "FAILED"
        row["stencil"] = {
            "status": "FAILED",
            "error_type": "CenterProductionFailure",
            "error": "order-64 center production scalar/AD failed",
            "denominators": [],
        }
    return row


def run_study(
    prepared_dir: Path,
    protocol_path: Path,
    output_dir: Path,
    *,
    centers_only: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"fresh output directory required: {output_dir}")
    inputs = load_study_inputs(prepared_dir, protocol_path)
    tracked_files = (
        *SOURCE_FILES,
        Path(protocol_path).resolve(),
        *(Path(prepared_dir).resolve() / name for name in inputs.prepared_files),
    )
    before = snapshot_files(tracked_files)
    output_dir.mkdir(parents=True)
    _atomic_json(output_dir / "source-snapshot-before.json", before)
    command = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "centers_only": centers_only,
        "artifact_class": ARTIFACT_CLASS,
        "public_admission": False,
        "whole_segment_certified": False,
    }
    _atomic_json(output_dir / "command.json", command)
    rows: list[dict[str, Any]] = []
    for scale in inputs.protocol["geometry_scales"]:
        try:
            row = _evaluate_center(float(scale), inputs, centers_only=centers_only)
        except Exception as exc:
            row = {
                "scale": float(scale),
                "artifact_class": ARTIFACT_CLASS,
                "public_admission": False,
                "whole_segment_certified": False,
                "row_status": "FAILED",
                "failure": _exception_record(exc),
            }
        rows.append(row)
        _atomic_json(
            output_dir / "progress.json",
            {
                "status": "running",
                "completed_rows": len(rows),
                "expected_rows": len(inputs.protocol["geometry_scales"]),
                "rows": rows,
            },
        )
    after = snapshot_files(tracked_files)
    require_unchanged_snapshot(before, after)
    counts = {
        status: sum(row.get("row_status") == status for row in rows)
        for status in ("REGULAR", "EVENT", "FAILED", "CENTER_ONLY")
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-cha-continuum-branch-study-v1",
        "artifact_class": ARTIFACT_CLASS,
        "status": "complete",
        "mode": "centers-only" if centers_only else "full-preregistered-matrix",
        "public_admission": False,
        "whole_segment_certified": False,
        "numerical_runtime_forces_used": False,
        "finite_differences_role": "independent-reference-validation-only",
        "protocol_sha256": inputs.protocol_file_sha256,
        "topology_sha256": inputs.topology_sha256,
        "prepared_files": dict(inputs.prepared_files),
        "source_snapshot_before": before,
        "source_snapshot_after": after,
        "source_snapshot_unchanged": True,
        "summary": {"row_count": len(rows), **counts},
        "rows": rows,
    }
    artifact["content_sha256"] = _content_sha(artifact)
    _atomic_json(output_dir / "result.json", artifact)
    _atomic_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "result_file": "result.json",
            "result_content_sha256": artifact["content_sha256"],
            "summary": artifact["summary"],
        },
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--centers-only", action="store_true")
    args = parser.parse_args()
    artifact = run_study(
        args.prepared_dir,
        args.protocol,
        args.output,
        centers_only=args.centers_only,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "mode": artifact["mode"],
                "summary": artifact["summary"],
                "content_sha256": artifact["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
