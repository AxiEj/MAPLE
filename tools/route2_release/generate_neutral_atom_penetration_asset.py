#!/usr/bin/env python3
"""Generate a target-independent neutral-atom penetration density asset.

The asset is an analytic representation of spherically averaged isolated-atom
Hartree--Fock densities.  It exists only to evaluate a preregistered
nucleus-plus-electron-cloud charge-penetration kernel.  It never reads a
solute, solvent, PCM energy, experimental target, MACE output, or residual.

The numerical atomic-density and nonnegative Gaussian-expansion primitives are
reused from the already audited rho-DROP asset generators.  Their exact source
hashes and the complete runtime are frozen by the preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT = "route2-neutral-atom-penetration-gaussian-mixture-v1"
PREREGISTRATION = ROOT / (
    "docs/route2/preregistrations/neutral-atom-penetration-asset-v1.json"
)
RUNNER_REPO_PATH = "tools/route2_release/generate_neutral_atom_penetration_asset.py"
ATOMIC_GENERATOR_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/generate_route2_v0_promolecular_density.py"
)
MIXTURE_GENERATOR_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "generate_route2_rhodrop_atomic_reference_density.py"
)
LEGACY_TABLE_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
)
TABLE_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-neutral-atom-penetration-gaussian-mixture-v1.npz"
)


def _load_generator(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load asset generator: {relative_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


atomic_density_generator = _load_generator(
    "maple_atomic_density_asset_generator", ATOMIC_GENERATOR_REPO_PATH
)
gaussian_mixture_generator = _load_generator(
    "maple_gaussian_mixture_asset_generator", MIXTURE_GENERATOR_REPO_PATH
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def _load_preregistration() -> dict[str, object]:
    path = PREREGISTRATION.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "Preregistration must be one JSON object.")
    _require(
        payload.get("artifact")
        == "route2-neutral-atom-penetration-asset-preregistration-v1",
        "Unknown neutral-atom penetration preregistration.",
    )
    _require(payload.get("schema_version") == 1, "Schema version changed.")
    _require(
        payload.get("status")
        == (
            "scientific-contract-locked-before-new-element-results-"
            "manifest-path-fix-after-first-generation"
        ),
        "Neutral-atom penetration asset is not prospectively locked.",
    )
    expected_claim = {
        "capability_admitted": False,
        "experimental_solvation_targets_read": False,
        "mace_outputs_read": False,
        "post_training_or_model_fitting_performed": False,
        "qm_atomic_density_basis_projection_only": True,
        "solvation_or_pcm_quantities_read": False,
    }
    _require(payload.get("claim_boundary") == expected_claim, "Claim changed.")
    sources = payload.get("source_sha256")
    _require(isinstance(sources, dict), "Source hashes are missing.")
    for name in (
        RUNNER_REPO_PATH,
        ATOMIC_GENERATOR_REPO_PATH,
        MIXTURE_GENERATOR_REPO_PATH,
        LEGACY_TABLE_REPO_PATH,
    ):
        _require(
            sources.get(name) == _sha256(ROOT / name),
            f"Source hash mismatch: {name}.",
        )
    return payload


def _runtime_identity() -> dict[str, str]:
    import pyscf
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyscf": pyscf.__version__,
    }


def _legacy_mixtures() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with np.load(ROOT / LEGACY_TABLE_REPO_PATH, allow_pickle=False) as archive:
        numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
        offsets = np.asarray(archive["component_offsets"], dtype=np.int64)
        counts = np.asarray(archive["electron_counts"], dtype=np.float64)
        exponents = np.asarray(archive["gaussian_exponents_bohr2"], dtype=np.float64)
    result = {}
    for index, number in enumerate(numbers):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        result[int(number)] = (counts[start:stop], exponents[start:stop])
    return result


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def main() -> None:
    args = _parse_args()
    table_path = args.output_table.expanduser().resolve()
    manifest_path = args.output_manifest.expanduser().resolve()
    if table_path.exists() or manifest_path.exists():
        raise RuntimeError("Neutral-atom penetration outputs must not exist.")

    prereg = _load_preregistration()
    runtime = _runtime_identity()
    _require(runtime == prereg.get("runtime"), "Numerical runtime changed.")
    contract = prereg.get("generation_contract")
    _require(isinstance(contract, dict), "Generation contract is missing.")
    _require(
        contract.get("method") == "PySCF AtomSphAverageRHF"
        and contract.get("basis") == "def2-TZVPD",
        "Atomic electronic-structure contract changed.",
    )
    _require(
        contract.get("existing_elements_reused_bitwise_from_frozen_v1")
        == [1, 6, 7, 8, 16, 17]
        and contract.get("new_elements_generated") == [9, 15, 35],
        "Existing/new element provenance changed.",
    )
    elements = contract.get("elements")
    _require(isinstance(elements, list) and elements, "Element contract is invalid.")
    radial = contract.get("radial_grid")
    _require(isinstance(radial, dict), "Radial-grid contract is missing.")
    _require(
        radial.get("coordinate_rule") == "r_max * (i / (n - 1))**2",
        "Radial-grid coordinate rule changed.",
    )
    point_count = int(radial["point_count"])
    radius_max = float(radial["radius_max_bohr"])
    index = np.arange(point_count, dtype=float)
    radius_bohr = radius_max * (index / (point_count - 1)) ** 2

    fit = contract.get("gaussian_projection")
    _require(isinstance(fit, dict), "Gaussian-projection contract is missing.")
    expected_fit = {
        "kind": "nonnegative-normalized-gaussian-expansion-v1",
        "candidate_exponent_count": (
            gaussian_mixture_generator.CANDIDATE_EXPONENT_COUNT
        ),
        "candidate_exponent_min_bohr2": (
            gaussian_mixture_generator.CANDIDATE_EXPONENT_MIN_BOHR2
        ),
        "candidate_exponent_max_bohr2": (
            gaussian_mixture_generator.CANDIDATE_EXPONENT_MAX_BOHR2
        ),
        "fit_density_minimum_e_per_bohr3": (
            gaussian_mixture_generator.FIT_DENSITY_MINIMUM_E_PER_BOHR3
        ),
        "relative_weight_floor_e_per_bohr3": (
            gaussian_mixture_generator.RELATIVE_WEIGHT_FLOOR_E_PER_BOHR3
        ),
        "normalization_row_weight": (
            gaussian_mixture_generator.NORMALIZATION_ROW_WEIGHT
        ),
        "component_count_prune_threshold_e": (
            gaussian_mixture_generator.COMPONENT_COUNT_PRUNE_THRESHOLD_E
        ),
        "audit_isodensity_e_per_bohr3": (
            gaussian_mixture_generator.AUDIT_ISODENSITY_E_PER_BOHR3
        ),
    }
    _require(fit == expected_fit, "Gaussian-projection contract changed.")
    candidate_exponents = np.geomspace(
        float(fit["candidate_exponent_min_bohr2"]),
        float(fit["candidate_exponent_max_bohr2"]),
        int(fit["candidate_exponent_count"]),
    )
    validation = contract.get("atomic_density_validation")
    _require(isinstance(validation, dict), "Atomic-density gates are missing.")
    legacy = _legacy_mixtures()

    packed_counts: list[np.ndarray] = []
    packed_exponents: list[np.ndarray] = []
    offsets = [0]
    atomic_numbers: list[int] = []
    results: list[dict[str, object]] = []
    for raw_element in elements:
        _require(isinstance(raw_element, dict), "Element entry is invalid.")
        atomic_number = int(raw_element["atomic_number"])
        if atomic_number in legacy:
            counts = np.array(legacy[atomic_number][0], copy=True)
            exponents = np.array(legacy[atomic_number][1], copy=True)
            density_result = None
            fit_result = None
            legacy_replay = {
                "source": LEGACY_TABLE_REPO_PATH,
                "copied_bitwise": True,
            }
        else:
            density, density_result = atomic_density_generator._atomic_density(
                element=raw_element,
                basis=str(contract["basis"]),
                radial_grid_bohr=radius_bohr,
                spherical_probe_radius_bohr=float(
                    validation["spherical_probe_radius_bohr"]
                ),
                convergence_tolerance=float(
                    validation["scf_convergence_tolerance_hartree"]
                ),
                electron_count_tolerance=float(
                    validation["electron_count_absolute_error_max"]
                ),
                isotropy_tolerance=float(
                    validation["spherical_probe_maximum_difference_max"]
                ),
                tail_tolerance=float(validation["radial_tail_density_max_e_per_bohr3"]),
            )
            counts, exponents, fit_result = gaussian_mixture_generator._fit_atom(
                radius_bohr,
                density,
                atomic_number,
                candidate_exponents,
            )
            legacy_replay = None
        atomic_numbers.append(atomic_number)
        packed_counts.append(counts)
        packed_exponents.append(exponents)
        offsets.append(offsets[-1] + len(counts))
        results.append(
            {
                "atomic_density": density_result,
                "gaussian_projection": fit_result,
                "legacy_v1_bitwise_replay": legacy_replay,
            }
        )

    arrays = {
        "atomic_numbers": np.asarray(atomic_numbers, dtype=np.int64),
        "component_offsets": np.asarray(offsets, dtype=np.int64),
        "electron_counts": np.concatenate(packed_counts).astype(np.float64),
        "gaussian_exponents_bohr2": np.concatenate(packed_exponents).astype(np.float64),
    }
    table_path.parent.mkdir(parents=True, exist_ok=True)
    gaussian_mixture_generator._write_deterministic_npz(table_path, arrays)
    table_sha256 = _sha256(table_path)
    manifest = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "atomic-density-and-analytic-kernel-only-solvation-gate-pending",
        "claim_boundary": prereg["claim_boundary"],
        "generation_contract": contract,
        "preregistration": {
            "path": str(PREREGISTRATION.relative_to(ROOT)),
            "sha256": _sha256(PREREGISTRATION),
        },
        "runtime": runtime,
        "source_sha256": prereg["source_sha256"],
        "results": results,
        "table": {
            "path": TABLE_REPO_PATH,
            "sha256": table_sha256,
            "content_sha256": _canonical_sha256(
                {
                    key: {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "bytes": value.tobytes(order="C").hex(),
                    }
                    for key, value in arrays.items()
                }
            ),
        },
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json_exclusive(manifest_path, manifest)


if __name__ == "__main__":
    main()
