#!/usr/bin/env python3
"""Generate the role-separated iodine ADT shape and its all-electron audits.

The operational iodine shape is the positive 25-electron def2-ECP valence
pseudo-density.  The nonrelativistic and spin-free-X2C 53-electron densities
are audit references only.  None of these inputs contains a solvation target,
MACE output, cavity quantity, or fitted PCM observable.

The Gaussian projection deliberately reuses the frozen rho-DROP NNLS kernel;
this tool does not define a second radial fitting algorithm.  It also exposes
``project_audit_mixture`` so the operator canary can cold-rebuild the exact
reference mixtures from the frozen radial densities rather than reading
untracked ``/tmp`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ARTIFACT = "route2-adt-radial-shape-iodine-ecp-valence-v1"
AUDIT_ARTIFACT = "route2-iodine-adt-gaussian-projection-audit-v1"
ROLE = "adt_translation_shape"
TABLE_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-adt-radial-shape-iodine-ecp-valence-v1.npz"
)
MANIFEST_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-adt-radial-shape-iodine-ecp-valence-v1.json"
)
EVIDENCE_REPO_DIR = (
    "docs/route2/evidence/mace-mdp-polar-iodine-adt-pro-20260819"
)
AUDIT_REPO_PATH = f"{EVIDENCE_REPO_DIR}/iodine_projection_audit.json"
PROJECTION_KERNEL_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "generate_route2_rhodrop_atomic_reference_density.py"
)
GENERATOR_REPO_PATH = "tools/route2_release/generate_iodine_adt_assets.py"

RAW_DENSITIES: dict[str, dict[str, Any]] = {
    "ecp": {
        "path": f"{EVIDENCE_REPO_DIR}/iodine_def2_ecp_atomic_density.json",
        "sha256": "26bce79272783df2da3c362627b4011e2062e2fd728ee79334bf42d51c3430bf",
        "label": "iodine_def2_ecp",
        "basis": "def2-tzvpd",
        "ecp": "def2-tzvpd",
        "x2c": False,
        "electron_count": 25,
        "ao_count": 59,
    },
    "nr": {
        "path": f"{EVIDENCE_REPO_DIR}/iodine_ano_nr_atomic_density.json",
        "sha256": "d2778707e426f7a0487bc8f63a82dacb9ab37d023e7cbb390f6e78e541e99f55",
        "label": "iodine_ano_nr",
        "basis": "ano-rcc",
        "ecp": None,
        "x2c": False,
        "electron_count": 53,
        "ao_count": 139,
    },
    "x2c": {
        "path": f"{EVIDENCE_REPO_DIR}/iodine_ano_x2c_atomic_density.json",
        "sha256": "3ae8dce38c8ebcd8f4c27feb4c029661f599f1446c78281061fb844ef66c7d0e",
        "label": "iodine_ano_x2c",
        "basis": "ano-rcc",
        "ecp": None,
        "x2c": True,
        "electron_count": 53,
        "ao_count": 139,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _projection_kernel(source_root: Path) -> ModuleType:
    path = source_root / PROJECTION_KERNEL_REPO_PATH
    spec = importlib.util.spec_from_file_location("route2_iodine_projection_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen Gaussian projection kernel.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_raw_density(source_root: Path, label: str) -> dict[str, Any]:
    try:
        expected = RAW_DENSITIES[label]
    except KeyError as exc:
        raise ValueError(f"Unknown iodine density label {label!r}.") from exc
    path = source_root / str(expected["path"])
    if _sha256(path) != expected["sha256"]:
        raise RuntimeError(f"Frozen iodine {label} density SHA256 mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "label",
        "basis",
        "ecp",
        "x2c",
        "energy",
        "nelec",
        "nao",
        "integral",
        "radii",
        "rho",
        "enclosed_fraction",
    }
    if set(payload) != required:
        raise RuntimeError(f"Frozen iodine {label} density schema changed.")
    for key in ("label", "basis", "ecp", "x2c"):
        if payload[key] != expected[key]:
            raise RuntimeError(f"Frozen iodine {label} metadata {key} changed.")
    if (
        payload["nelec"] != expected["electron_count"]
        or payload["nao"] != expected["ao_count"]
    ):
        raise RuntimeError(f"Frozen iodine {label} electron/AO count changed.")
    radius = np.asarray(payload["radii"], dtype=np.float64)
    density = np.asarray(payload["rho"], dtype=np.float64)
    enclosed = np.asarray(payload["enclosed_fraction"], dtype=np.float64)
    if (
        radius.ndim != 1
        or density.shape != radius.shape
        or enclosed.shape != radius.shape
        or len(radius) != 3101
        or not np.all(np.isfinite(radius))
        or not np.all(np.isfinite(density))
        or not np.all(np.isfinite(enclosed))
        or radius[0] != 0.0
        or np.any(np.diff(radius) <= 0.0)
        or np.any(density < 0.0)
        or float(payload["integral"]) <= 0.0
    ):
        raise RuntimeError(f"Frozen iodine {label} radial density is invalid.")
    return payload


def _fit(
    source_root: Path,
    label: str,
    *,
    normalization_electrons: int,
    normalize_input_density: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    module = _projection_kernel(source_root)
    raw = _load_raw_density(source_root, label)
    radius = np.asarray(raw["radii"], dtype=np.float64)
    density = np.asarray(raw["rho"], dtype=np.float64)
    if normalize_input_density:
        density = density * (
            float(normalization_electrons) / float(raw["integral"])
        )
    candidates = np.geomspace(
        module.CANDIDATE_EXPONENT_MIN_BOHR2,
        module.CANDIDATE_EXPONENT_MAX_BOHR2,
        module.CANDIDATE_EXPONENT_COUNT,
    )
    counts, exponents, metrics = module._fit_atom(
        radius,
        density,
        normalization_electrons,
        candidates,
    )
    return (
        np.asarray(counts, dtype=np.float64),
        np.asarray(exponents, dtype=np.float64),
        metrics,
    )


def project_audit_mixture(
    source_root: str | Path, label: str
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Cold-rebuild one exact-normalized 53-electron audit mixture."""

    return _fit(
        Path(source_root).expanduser().resolve(),
        label,
        normalization_electrons=53,
        normalize_input_density=True,
    )


def _array_record(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(values)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def generate(
    *,
    source_root: Path,
    output_table: Path,
    output_manifest: Path,
    output_audit: Path,
) -> None:
    source_root = source_root.expanduser().resolve()
    generator = source_root / GENERATOR_REPO_PATH
    if Path(__file__).resolve() != generator:
        raise RuntimeError("Iodine ADT generator must run from its claimed source root.")
    kernel = _projection_kernel(source_root)

    counts, exponents, metrics = _fit(
        source_root,
        "ecp",
        normalization_electrons=25,
        normalize_input_density=False,
    )
    arrays = {
        "atomic_numbers": np.asarray([53], dtype=np.int64),
        "component_offsets": np.asarray([0, len(counts)], dtype=np.int64),
        "ecp_core_electron_counts": np.asarray([28], dtype=np.int64),
        "effective_electron_counts": np.asarray([25.0], dtype=np.float64),
        "electron_counts": counts,
        "gaussian_exponents_bohr2": exponents,
    }
    output_table.parent.mkdir(parents=True, exist_ok=True)
    kernel._write_deterministic_npz(output_table, arrays)
    content_sha256 = _canonical_sha256(
        {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes": value.tobytes(order="C").hex(),
            }
            for key, value in arrays.items()
        }
    )

    raw_ecp = _load_raw_density(source_root, "ecp")
    manifest: dict[str, object] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": (
            "target-free-role-separated-adt-shape-implemented-accuracy-gate-pending"
        ),
        "claim_boundary": {
            "capability_admitted": False,
            "experimental_solvation_targets_read": False,
            "full_neutral_density": False,
            "mace_outputs_read": False,
            "neutral_penetration_use_forbidden": True,
            "role": ROLE,
            "solvation_or_pcm_quantities_used_for_fit": False,
        },
        "element": {
            "ao_count": int(raw_ecp["nao"]),
            "atomic_number": 53,
            "basis": "def2-TZVPD",
            "ecp": "def2-TZVPD",
            "ecp_core_electron_count": 28,
            "effective_core_charge": 25.0,
            "effective_electron_count": 25.0,
            "electronic_structure_method": "PySCF AtomSphAverageRHF",
            "radial_integral_electrons": float(raw_ecp["integral"]),
            "relativistic_hamiltonian": "ECP scalar-relativistic effective core",
            "representation": "ecp_valence_pseudodensity",
            "scf_energy_hartree": float(raw_ecp["energy"]),
            "spin_2s": 1,
            "symbol": "I",
        },
        "gaussian_projection": {
            "candidate_exponent_count": kernel.CANDIDATE_EXPONENT_COUNT,
            "candidate_exponent_max_bohr2": (
                kernel.CANDIDATE_EXPONENT_MAX_BOHR2
            ),
            "candidate_exponent_min_bohr2": (
                kernel.CANDIDATE_EXPONENT_MIN_BOHR2
            ),
            "generator_path": GENERATOR_REPO_PATH,
            "generator_sha256": _sha256(generator),
            "implementation_path": PROJECTION_KERNEL_REPO_PATH,
            "implementation_sha256": _sha256(
                source_root / PROJECTION_KERNEL_REPO_PATH
            ),
            "metrics": metrics,
            "normalization_electrons": 25,
            "protocol": "nonnegative-normalized-gaussian-expansion-v1",
            "runtime_environment": kernel._runtime_environment(),
        },
        "source_density": {
            "path": RAW_DENSITIES["ecp"]["path"],
            "sha256": RAW_DENSITIES["ecp"]["sha256"],
        },
        "all_electron_audit": {
            "artifact": AUDIT_ARTIFACT,
            "path": AUDIT_REPO_PATH,
            "roles": {
                "ecp": "uniformly-normalized-algebra-control-only",
                "nr": "53-electron-nonrelativistic-audit",
                "x2c": "53-electron-spin-free-x2c-reference-audit",
            },
        },
        "decision_evidence": {
            "path": f"{EVIDENCE_REPO_DIR}/manifest.json",
            "terminal_marker": "USE ECP-VALENCE ADT WITH ALL-ELECTRON AUDIT",
            "verified_model": "Pro, 5 of 5",
        },
        "table": {
            "content_sha256": content_sha256,
            "path": TABLE_REPO_PATH,
            "sha256": _sha256(output_table),
        },
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(output_manifest, manifest)

    audit_fits: dict[str, object] = {}
    for label in ("ecp", "nr", "x2c"):
        audit_counts, audit_exponents, audit_metrics = project_audit_mixture(
            source_root, label
        )
        audit_fits[label] = {
            "source": {
                "path": RAW_DENSITIES[label]["path"],
                "sha256": RAW_DENSITIES[label]["sha256"],
            },
            "normalization_electrons": 53,
            "input_density_normalized_before_projection": True,
            "electron_counts": _array_record(audit_counts),
            "gaussian_exponents_bohr2": _array_record(audit_exponents),
            "metrics": audit_metrics,
        }
    audit = {
        "artifact": AUDIT_ARTIFACT,
        "schema_version": 1,
        "claim_boundary": {
            "operational_profile_input": False,
            "experimental_solvation_targets_read": False,
            "mace_outputs_read": False,
            "pcm_or_cavity_quantities_used_for_projection": False,
            "purpose": "iodine-adt-radial-shape-reference-audit-only",
        },
        "generator": {
            "path": GENERATOR_REPO_PATH,
            "sha256": _sha256(generator),
        },
        "projection_kernel": {
            "path": PROJECTION_KERNEL_REPO_PATH,
            "sha256": _sha256(source_root / PROJECTION_KERNEL_REPO_PATH),
        },
        "runtime_environment": kernel._runtime_environment(),
        "fits": audit_fits,
    }
    audit["payload_sha256"] = _canonical_sha256(audit)
    _write_json(output_audit, audit)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=root)
    parser.add_argument("--output-table", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--output-audit", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.source_root.expanduser().resolve()
    generate(
        source_root=root,
        output_table=args.output_table or root / TABLE_REPO_PATH,
        output_manifest=args.output_manifest or root / MANIFEST_REPO_PATH,
        output_audit=args.output_audit or root / AUDIT_REPO_PATH,
    )


if __name__ == "__main__":
    main()
