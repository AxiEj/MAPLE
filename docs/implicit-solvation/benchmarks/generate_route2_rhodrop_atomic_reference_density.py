#!/usr/bin/env python3
"""Build the frozen analytic atomic reference used by rho-DROP PR1.

The input is the already frozen, positive, spherically averaged free-atom HF
table.  This script only changes its numerical representation: it performs a
nonnegative expansion in normalized spherical Gaussians and enforces the exact
neutral-atom electron-count sum rule.  It does not fit solvation energies,
cavity radii, experimental data, or MACE outputs.

The resulting representation is still subject to a separate QM-isosurface
scientific admission gate.  Passing this generator's algebraic checks is not a
claim that the reconstructed MACE-POLAR density is a QM electron density.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import scipy
from scipy.optimize import nnls

ARTIFACT = "route2-rhodrop-atomic-reference-gaussian-mixture-v1"
CONSTRUCTION = "positive-normalized-spherical-free-atom-hf-gaussian-mixture-v1"
SOURCE_ARTIFACT = "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1"
SOURCE_TABLE_SHA256 = "1a63ae8a5d939e00f0a8dba1951a8099dad8ba2a068be61431046c880d7aae28"
SOURCE_MANIFEST_SHA256 = (
    "9f0b56bbe5bc116d55b10ac3aa7246080e1a17a06001f62b81cd3f66b6c95f81"
)
CANDIDATE_EXPONENT_COUNT = 192
CANDIDATE_EXPONENT_MIN_BOHR2 = 5.0e-5
CANDIDATE_EXPONENT_MAX_BOHR2 = 5.0e5
FIT_DENSITY_MINIMUM_E_PER_BOHR3 = 1.0e-12
RELATIVE_WEIGHT_FLOOR_E_PER_BOHR3 = 1.0e-10
NORMALIZATION_ROW_WEIGHT = 1.0e6
COMPONENT_COUNT_PRUNE_THRESHOLD_E = 1.0e-10
AUDIT_ISODENSITY_E_PER_BOHR3 = 1.0e-3


ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
SOURCE_TABLE = BENCHMARKS / f"{SOURCE_ARTIFACT}.npz"
SOURCE_MANIFEST = BENCHMARKS / f"{SOURCE_ARTIFACT}.json"
OUTPUT_TABLE = BENCHMARKS / f"{ARTIFACT}.npz"
OUTPUT_MANIFEST = BENCHMARKS / f"{ARTIFACT}.json"
OUTPUT_TABLE_REPO_PATH = f"docs/implicit-solvation/benchmarks/{ARTIFACT}.npz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(values),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ without wall-clock ZIP metadata."""

    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(
                info,
                _npy_bytes(arrays[name]),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _linear_algebra_runtime(config_module: object) -> dict[str, object]:
    """Return the version-bearing BLAS/LAPACK identity used by one package."""

    config = getattr(config_module, "CONFIG", {})
    if not isinstance(config, dict):
        return {"available": False}
    dependencies = config.get("Build Dependencies", {})
    if not isinstance(dependencies, dict):
        return {"available": False}

    result: dict[str, object] = {"available": True}
    for kind in ("blas", "lapack"):
        raw_record = dependencies.get(kind, {})
        if not isinstance(raw_record, dict):
            result[kind] = {"found": False}
            continue
        result[kind] = {
            key: raw_record[key]
            for key in (
                "found",
                "name",
                "version",
                "has ilp64",
                "openblas configuration",
            )
            if key in raw_record
        }
    return result


def _runtime_environment() -> dict[str, object]:
    """Capture the numerical stack that determines the NNLS representation."""

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "numpy": {
            "version": np.__version__,
            "linear_algebra": _linear_algebra_runtime(np.__config__),
        },
        "scipy": {
            "version": scipy.__version__,
            "linear_algebra": _linear_algebra_runtime(scipy.__config__),
        },
    }


def _gaussian_design(
    radius_bohr: np.ndarray, exponents_bohr2: np.ndarray
) -> np.ndarray:
    return (exponents_bohr2[None, :] / np.pi) ** 1.5 * np.exp(
        -radius_bohr[:, None] ** 2 * exponents_bohr2[None, :]
    )


def _isodensity_radius(
    radius_bohr: np.ndarray,
    density_e_per_bohr3: np.ndarray,
    isodensity: float,
) -> float:
    crossing = np.flatnonzero(density_e_per_bohr3 <= isodensity)
    if crossing.size == 0 or crossing[0] == 0:
        raise RuntimeError("Atomic density does not bracket the audit isodensity.")
    upper = int(crossing[0])
    lower = upper - 1
    log_density = np.log(density_e_per_bohr3[[lower, upper]])
    radii = radius_bohr[[lower, upper]]
    fraction = (np.log(isodensity) - log_density[0]) / (log_density[1] - log_density[0])
    return float(radii[0] + fraction * (radii[1] - radii[0]))


def _fit_atom(
    radius_bohr: np.ndarray,
    density_e_per_bohr3: np.ndarray,
    atomic_number: int,
    candidate_exponents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    design = _gaussian_design(radius_bohr, candidate_exponents)
    fit_mask = density_e_per_bohr3 >= FIT_DENSITY_MINIMUM_E_PER_BOHR3
    target = density_e_per_bohr3[fit_mask]
    weights = 1.0 / np.maximum(
        target,
        RELATIVE_WEIGHT_FLOOR_E_PER_BOHR3,
    )
    weighted_design = design[fit_mask] * weights[:, None]
    weighted_target = target * weights
    augmented_design = np.vstack(
        [
            weighted_design,
            NORMALIZATION_ROW_WEIGHT
            * np.ones((1, candidate_exponents.size), dtype=float),
        ]
    )
    augmented_target = np.concatenate(
        [weighted_target, [NORMALIZATION_ROW_WEIGHT * atomic_number]]
    )
    counts, _ = nnls(
        augmented_design,
        augmented_target,
        maxiter=20 * candidate_exponents.size,
    )
    keep = counts > COMPONENT_COUNT_PRUNE_THRESHOLD_E
    counts = counts[keep]
    exponents = candidate_exponents[keep]
    if counts.size == 0:
        raise RuntimeError(f"Gaussian fit for Z={atomic_number} has no components.")
    counts *= float(atomic_number) / float(np.sum(counts))

    prediction = _gaussian_design(radius_bohr, exponents) @ counts
    audit_mask = density_e_per_bohr3 >= AUDIT_ISODENSITY_E_PER_BOHR3
    relative_error = (
        np.abs(prediction[audit_mask] - density_e_per_bohr3[audit_mask])
        / density_e_per_bohr3[audit_mask]
    )
    reference_radius = _isodensity_radius(
        radius_bohr,
        density_e_per_bohr3,
        AUDIT_ISODENSITY_E_PER_BOHR3,
    )
    fitted_radius = _isodensity_radius(
        radius_bohr,
        prediction,
        AUDIT_ISODENSITY_E_PER_BOHR3,
    )
    metrics: dict[str, float | int] = {
        "atomic_number": atomic_number,
        "component_count": int(counts.size),
        "electron_count": float(np.sum(counts)),
        "electron_count_absolute_error": abs(float(np.sum(counts)) - atomic_number),
        "maximum_relative_density_error_at_or_above_1e-3": float(
            np.max(relative_error)
        ),
        "rms_relative_density_error_at_or_above_1e-3": float(
            np.sqrt(np.mean(relative_error**2))
        ),
        "reference_1e-3_isodensity_radius_bohr": reference_radius,
        "fitted_1e-3_isodensity_radius_bohr": fitted_radius,
        "absolute_1e-3_isodensity_radius_error_bohr": abs(
            fitted_radius - reference_radius
        ),
    }
    return counts, exponents, metrics


def main() -> None:
    if _sha256(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("Frozen source manifest SHA256 mismatch.")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source_manifest.get("artifact") != SOURCE_ARTIFACT:
        raise RuntimeError("Frozen source manifest identity is invalid.")
    if _sha256(SOURCE_TABLE) != SOURCE_TABLE_SHA256:
        raise RuntimeError("Frozen source table SHA256 mismatch.")

    candidate_exponents = np.geomspace(
        CANDIDATE_EXPONENT_MIN_BOHR2,
        CANDIDATE_EXPONENT_MAX_BOHR2,
        CANDIDATE_EXPONENT_COUNT,
    )
    with np.load(SOURCE_TABLE, allow_pickle=False) as source:
        radius_bohr = np.asarray(source["radial_grid_bohr"], dtype=float)
        atomic_numbers = np.asarray(source["atomic_numbers"], dtype=np.int64)
        packed_counts: list[np.ndarray] = []
        packed_exponents: list[np.ndarray] = []
        offsets = [0]
        results: list[dict[str, float | int]] = []
        for raw_atomic_number in atomic_numbers:
            atomic_number = int(raw_atomic_number)
            counts, exponents, metrics = _fit_atom(
                radius_bohr,
                np.asarray(source[f"density_Z{atomic_number}"], dtype=float),
                atomic_number,
                candidate_exponents,
            )
            packed_counts.append(counts)
            packed_exponents.append(exponents)
            offsets.append(offsets[-1] + counts.size)
            results.append(metrics)

    arrays = {
        "atomic_numbers": atomic_numbers,
        "component_offsets": np.asarray(offsets, dtype=np.int64),
        "electron_counts": np.concatenate(packed_counts).astype(np.float64),
        "gaussian_exponents_bohr2": np.concatenate(packed_exponents).astype(np.float64),
    }
    _write_deterministic_npz(OUTPUT_TABLE, arrays)
    table_sha256 = _sha256(OUTPUT_TABLE)
    generator_path = Path(__file__).resolve()
    manifest = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "construction": CONSTRUCTION,
        "status": "mathematical-kernel-pass-scientific-isosurface-gate-pending",
        "claim_boundary": (
            "This is a positive normalized analytic reference used only in the "
            "experimental reconstructed-MACE-polar-density level set. It is "
            "derived without solvation or experimental fitting and is not a "
            "trained MACE electron density, a QM-isosurface validation, a "
            "continuum-energy result, or force/PES admission."
        ),
        "supported_atomic_numbers": atomic_numbers.tolist(),
        "source": {
            "artifact": SOURCE_ARTIFACT,
            "table_path": str(SOURCE_TABLE.relative_to(ROOT)),
            "table_sha256": SOURCE_TABLE_SHA256,
            "manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "generator": {
            "path": str(generator_path.relative_to(ROOT)),
            "sha256": _sha256(generator_path),
        },
        "runtime_environment": _runtime_environment(),
        "fit_protocol": {
            "kind": "nonnegative-normalized-gaussian-expansion-v1",
            "candidate_exponent_count": CANDIDATE_EXPONENT_COUNT,
            "candidate_exponent_min_bohr2": CANDIDATE_EXPONENT_MIN_BOHR2,
            "candidate_exponent_max_bohr2": CANDIDATE_EXPONENT_MAX_BOHR2,
            "fit_density_minimum_e_per_bohr3": FIT_DENSITY_MINIMUM_E_PER_BOHR3,
            "relative_weight_floor_e_per_bohr3": RELATIVE_WEIGHT_FLOOR_E_PER_BOHR3,
            "normalization_row_weight": NORMALIZATION_ROW_WEIGHT,
            "component_count_prune_threshold_e": COMPONENT_COUNT_PRUNE_THRESHOLD_E,
            "audit_isodensity_e_per_bohr3": AUDIT_ISODENSITY_E_PER_BOHR3,
            "experimental_solvation_fit": False,
            "mace_output_fit": False,
        },
        "results": results,
        "table": {
            "path": OUTPUT_TABLE_REPO_PATH,
            "sha256": table_sha256,
        },
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"table_sha256": table_sha256, "results": results}, indent=2))


if __name__ == "__main__":
    main()
