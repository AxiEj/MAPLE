#!/usr/bin/env python3
"""Lock target-blind MAPLE-CDS-W1 geometry-feature generation.

The preregistration is created only from a clean committed source tree and the
already-frozen MNSol development inputs.  It binds the 306 water identities,
the smooth harmonic area definition, every relevant source file, and the
parent hybrid-v3 preregistration.  It neither opens hybrid prediction records
nor performs a fit or calibration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

_SCRIPT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_SOURCE_ROOT))

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    SMD_WATER_TENSION_PARAMETER_NAMES,
)
from maple.solvation.continuum.harmonic_cds_area import (
    POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_CONTRACT_ID,
)
from maple.solvation.continuum.harmonic_positive_exposure import (
    POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID,
    POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE,
    POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS,
    POSITIVE_BERNSTEIN_PAIR_DEGREE,
)
from maple.solvation.harmonic_cds import (
    MAPLE_CDS_W1_POSITIVE_BERNSTEIN_PROFILE_ID,
)
from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    sha256_file,
)

from generate_maple_cds_w1_features import (
    AREA_DEVICE,
    AREA_DTYPE,
    AREA_SURFACE_LMAX,
    AREA_TRANSITION_WIDTH_ANGSTROM2,
    INPUT_FILE_NAMES,
    PREREGISTRATION_ARTIFACT,
    REQUIRED_SOURCE_FILE_NAMES,
    _load_selection,
    _maximum_active_pair_factors,
    _runtime_identity,
    _water_records,
)

PARENT_ARTIFACT = "route2-hybrid-smd-development-prereg-v3"


class W1FeaturePreregistrationError(RuntimeError):
    """Raised when a target-blind W1 preregistration cannot be frozen."""


def _outside_source(path: Path, source_root: Path, *, name: str) -> None:
    try:
        path.relative_to(source_root)
    except ValueError:
        return
    raise W1FeaturePreregistrationError(f"{name} must be outside source checkout.")


def _load_parent(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise W1FeaturePreregistrationError(
            "Parent hybrid preregistration must be a JSON object."
        )
    expected = {
        "artifact_id": PARENT_ARTIFACT,
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": 505,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise W1FeaturePreregistrationError(
                f"Parent hybrid preregistration field {key!r} drifted."
            )
    return payload


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SCRIPT_SOURCE_ROOT:
        raise W1FeaturePreregistrationError(
            "Preregistration creator must execute for its own source checkout."
        )
    input_root = args.input_root.expanduser().resolve(strict=True)
    parent_path = args.parent_hybrid_preregistration.expanduser().resolve(strict=True)
    output_dir = args.feature_output_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _outside_source(output, source_root, name="preregistration")
    _outside_source(output_dir, source_root, name="feature output directory")
    if output.exists():
        raise FileExistsError(output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise W1FeaturePreregistrationError(
            "Feature output directory must be empty before preregistration."
        )

    snapshot = RepositorySnapshot.capture(source_root)
    parent = _load_parent(parent_path)
    if parent.get("input_root") != str(input_root):
        raise W1FeaturePreregistrationError(
            "Parent hybrid preregistration uses a different input root."
        )
    input_hashes = {name: sha256_file(input_root / name) for name in INPUT_FILE_NAMES}
    _protocol, rows = _load_selection(source_root=source_root, input_root=input_root)
    water, water_identity_sha256 = _water_records(rows)
    water_count = len(water)
    maximum_active = _maximum_active_pair_factors(water)
    finite_product_degree = (
        POSITIVE_BERNSTEIN_PAIR_DEGREE * maximum_active + 2 * AREA_SURFACE_LMAX
    )
    if (
        maximum_active > POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS
        or finite_product_degree > POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE
    ):
        raise W1FeaturePreregistrationError(
            "Selected positive parent exceeds its frozen structural bound."
        )
    loaded_sources = collect_loaded_repository_sources(
        source_root,
        required_paths=tuple(
            name for name in REQUIRED_SOURCE_FILE_NAMES if name.endswith(".py")
        ),
    )
    source_hashes = committed_source_hashes(
        snapshot,
        tuple(sorted(set(loaded_sources).union(REQUIRED_SOURCE_FILE_NAMES))),
    )
    runtime_identity, runtime_identity_sha256 = _runtime_identity()
    generator_relative = "tools/route2_release/generate_maple_cds_w1_features.py"
    payload: dict[str, object] = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-positive-parent-feature-evaluation",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development-water-only",
        "water_record_count": water_count,
        "water_identity_sha256": water_identity_sha256,
        "dataset_loader_parses_experimental_targets": True,
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read_by_feature_generator": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_permitted": False,
        "checkpoint_or_method_selection_permitted": False,
        "source_root": str(source_root),
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "source_files_sha256": source_hashes,
        "feature_generator_sha256": source_hashes[generator_relative],
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "input_root": str(input_root),
        "input_files_sha256": input_hashes,
        "parent_hybrid_preregistration_path": str(parent_path),
        "parent_hybrid_preregistration_sha256": sha256_file(parent_path),
        "parent_hybrid_source_git_head": parent["source_git_head"],
        "feature_output_dir": str(output_dir),
        "preregistration_path": str(output),
        "area_definition": {
            "contract_id": POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_CONTRACT_ID,
            "exposure_contract_id": POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID,
            "profile_id": MAPLE_CDS_W1_POSITIVE_BERNSTEIN_PROFILE_ID,
            "radii": "published-smd-sasa-radii-including-0.4-A-probe",
            "transition_width_angstrom2": AREA_TRANSITION_WIDTH_ANGSTROM2,
            "surface_lmax": AREA_SURFACE_LMAX,
            "retained_parent_moment_lmax": 2 * AREA_SURFACE_LMAX,
            "positive_parent_pair_degree": POSITIVE_BERNSTEIN_PAIR_DEGREE,
            "maximum_transition_factors": (
                POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS
            ),
            "maximum_integrand_degree": (POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE),
            "dtype": AREA_DTYPE,
            "device": AREA_DEVICE,
            "area_measure": "a_i^2-integral-e_i-domega",
            "reconstructed_low_band_used_as_mask": False,
        },
        "linear_basis": {
            "contract": "published-aqueous-smd-linear-18-column-v1",
            "parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
            "stock_coefficients_cal_mol_angstrom2": (
                SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
            ),
            "stock_coefficients_sha256": canonical_json_sha256(
                SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
            ),
        },
        "numerical_choice_rationale": {
            "maximum_observed_active_pair_factors": maximum_active,
            "finite_product_degree_formula": (
                "pair_degree*active_pair_factors+2*surface_lmax"
            ),
            "positive_parent_pair_degree": POSITIVE_BERNSTEIN_PAIR_DEGREE,
            "maximum_integrand_degree": finite_product_degree,
            "maximum_transition_factor_cap": (
                POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS
            ),
            "implementation_degree_limit": (
                POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE
            ),
            "selected_from_geometry_only": True,
            "accuracy_results_used": False,
        },
        "output_contract": {
            "one_exclusive_json_per_water_record": True,
            "coordinates_emitted": False,
            "experimental_targets_emitted": False,
            "hybrid_outputs_emitted": False,
            "fit_or_calibration_emitted": False,
        },
        "claim_boundary": (
            "Geometry-only W1 design features. The licensed dataset loader parses "
            "the full frozen table, including its target column, to recover and "
            "validate coordinates. The downstream feature computation does not "
            "access target attributes, hybrid predictions, or a confirmation "
            "selection manifest; emits no targets; and performs no model fitting."
        ),
    }
    payload["self_sha256"] = canonical_json_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    snapshot.assert_unchanged()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-hybrid-preregistration", type=Path, required=True)
    parser.add_argument("--feature-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = create(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(sha256_file(args.output.expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
