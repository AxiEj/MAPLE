from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
BUILD_PATH = BENCHMARK_DIR / "build_route1_surface_electrostatics.py"
DIAGNOSTIC_PATH = BENCHMARK_DIR / "run_route1_surface_electrostatics_diagnostic.py"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_surface_electrostatics_protocol_v1.json"
SOURCE_MANIFEST_PATH = BENCHMARK_DIR / "apbs_ace_source_manifest.json"
SOURCE_ROOT = REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
COMPONENT_ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-freesolv-explicit-component-diagnostic-2026-07-29.json"
)
PREPARED_IDENTITY_PATH = SOURCE_ROOT / "prepared.json"
FROZEN_SURFACE_PATH = (
    BENCHMARK_DIR / "route1-freesolv-surface-electrostatics-label-free-2026-07-29.json"
)
FROZEN_DIAGNOSTIC_PATH = (
    BENCHMARK_DIR / "route1-freesolv-surface-electrostatics-diagnostic-2026-07-29.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_module("route1_surface_electrostatics_builder", BUILD_PATH)
diagnostic = _load_module("route1_surface_electrostatics_diagnostic", DIAGNOSTIC_PATH)


def _surface_kwargs() -> dict[str, float]:
    protocol, _ = builder.load_protocol(PROTOCOL_PATH)
    definition = protocol["surface_definition"]
    return {
        "probe_radius_angstrom": definition["probe_radius_angstrom"],
        "squared_distance_tie_tolerance_angstrom2": definition[
            "squared_distance_tie_tolerance_angstrom2"
        ],
        "minimum_pair_distance_angstrom": definition["minimum_pair_distance_angstrom"],
        "minimum_field_variance_e2_per_angstrom4": definition[
            "minimum_field_variance_e2_per_angstrom4"
        ],
    }


def _write_json(path: Path, value: object) -> Path:
    builder.core.write_json_atomic(path, value)
    return path


def test_surface_protocol_freezes_only_four_descriptors_and_two_profiles() -> None:
    protocol, protocol_sha256 = builder.load_protocol(PROTOCOL_PATH)

    assert protocol_sha256 == builder.core.sha256_bytes(
        builder.core.canonical_json_bytes(protocol)
    )
    assert tuple(protocol["descriptors"])[:4] == builder.DESCRIPTOR_KEYS
    assert protocol["descriptors"]["additional_descriptors_allowed"] is False
    assert protocol["radius_profiles"]["primary"]["radii"] == "bondi"
    assert protocol["radius_profiles"]["control"]["radii"] == "mbondi2"
    assert protocol["radius_profiles"]["selection_between_profiles_allowed"] is False
    assert (
        protocol["pre_registered_decision_rule"]["energy_correction_allowed"] is False
    )
    assert (
        protocol["pre_registered_decision_rule"]["provider_implementation_allowed"]
        is False
    )


def test_fibonacci_direction_fingerprints_reproduce_protocol() -> None:
    protocol, _ = builder.load_protocol(PROTOCOL_PATH)
    quadrature = protocol["surface_definition"]["quadrature"]

    for prefix in ("coarse", "fine"):
        count = quadrature[f"{prefix}_direction_count"]
        directions = builder.fibonacci_directions(count)
        assert (
            builder._array_hash(directions) == quadrature[f"{prefix}_direction_sha256"]
        )
        assert np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0)) < 3e-16


def test_surface_descriptors_obey_translation_permutation_and_charge_parity() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]
    )
    charges = np.asarray([-0.834, 0.417, 0.417])
    radii = np.asarray([1.5, 1.2, 1.2])
    directions = builder.fibonacci_directions(4096)
    kwargs = _surface_kwargs()

    reference = builder.surface_descriptors(
        positions, charges, radii, directions, **kwargs
    )
    repeated = builder.surface_descriptors(
        positions, charges, radii, directions, **kwargs
    )
    assert reference == repeated

    translated = builder.surface_descriptors(
        positions + np.asarray([3.7, -2.1, 0.8]),
        charges,
        radii,
        directions,
        **kwargs,
    )
    permutation = np.asarray([2, 0, 1])
    permuted = builder.surface_descriptors(
        positions[permutation],
        charges[permutation],
        radii[permutation],
        directions,
        **kwargs,
    )
    inverted = builder.surface_descriptors(
        positions, -charges, radii, directions, **kwargs
    )

    for key in builder.DESCRIPTOR_KEYS[:3]:
        assert translated[key] == pytest.approx(reference[key], rel=1e-12)
        assert permuted[key] == pytest.approx(reference[key], rel=1e-12)
        assert inverted[key] == pytest.approx(reference[key], rel=1e-12)
    assert translated["gamma_n"] == pytest.approx(reference["gamma_n"], abs=1e-12)
    assert permuted["gamma_n"] == pytest.approx(reference["gamma_n"], abs=1e-12)
    assert inverted["gamma_n"] == pytest.approx(-reference["gamma_n"], abs=1e-12)


def test_surface_descriptors_fail_closed_for_zero_field_variance() -> None:
    with pytest.raises(ValueError, match="variance is undefined or too small"):
        builder.surface_descriptors(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([1.0]),
            np.asarray([1.2]),
            builder.fibonacci_directions(1024),
            **_surface_kwargs(),
        )


def test_exact_normalized_charge_vector_not_rounded_mol2_tokens() -> None:
    manifest = builder.core.load_json(SOURCE_MANIFEST_PATH)
    record = manifest["records"][0]
    exact = record["am1bcc_charges_e"]
    exact_hash = builder.core.sha256_bytes(builder.core.canonical_json_bytes(exact))
    assert exact_hash == (
        "13ca2da2abdbb9225aae1f7783841f7335714698d38486909aca545ef017f1a2"
    )
    assert sum(exact) == pytest.approx(0.0, abs=1e-15)

    rounded_path = (
        SOURCE_ROOT
        / "provider-audit"
        / record["compound_id"]
        / "am1bcc"
        / "charges-am1bcc.mol2"
    )
    rounded_atoms = builder.MOL2Reader(str(rounded_path))
    rounded = rounded_atoms.get_initial_charges().tolist()
    rounded_hash = builder.core.sha256_bytes(builder.core.canonical_json_bytes(rounded))
    assert rounded_hash != exact_hash
    assert sum(rounded) == pytest.approx(0.001001, abs=1e-12)


def test_label_free_source_rejects_forbidden_label_key() -> None:
    with pytest.raises(ValueError, match="forbidden key"):
        builder._reject_forbidden_label_keys(
            {"records": [{"compound_id": "x", "experimental_kcal_mol": -1.0}]}
        )


def test_protocol_rejects_descriptor_search(tmp_path: Path) -> None:
    protocol = copy.deepcopy(builder.core.load_json(PROTOCOL_PATH))
    protocol["descriptors"]["maximum_absolute_phi"] = "posthoc feature"
    path = _write_json(tmp_path / "feature-search-protocol.json", protocol)

    with pytest.raises(ValueError, match="descriptor membership changed"):
        builder.load_protocol(path)


def test_frozen_label_free_artifact_is_sealed_label_free_and_numerically_valid() -> (
    None
):
    artifact = builder.core.load_json(FROZEN_SURFACE_PATH)

    assert artifact["content_sha256"] == builder.core.artifact_content_sha256(artifact)
    assert artifact["case_count"] == len(artifact["records"]) == 526
    assert artifact["numerical_validation"]["passed"] is True
    assert artifact["numerical_validation"]["quadrature_convergence_failure_count"] == 0
    assert artifact["numerical_validation"]["qa_failure_count"] == 0
    assert artifact["decision"]["status"] == (
        "label_free_surface_artifact_valid_for_retrospective_join"
    )
    assert artifact["command_provenance"]["script_sha256"] == (
        builder.core.sha256_file(BUILD_PATH)
    )

    forbidden = {
        "experimental_kcal_mol",
        "polar_mismatch_kcal_mol",
        "obc2_polar_mismatch_kcal_mol",
        "route_error_kcal_mol",
        "tail",
    }
    for record in artifact["records"]:
        assert forbidden.isdisjoint(record)
        assert set(record["radius_profiles"]) == {"primary", "control"}
        for profile in record["radius_profiles"].values():
            assert set(profile["descriptors"]) == set(builder.DESCRIPTOR_KEYS)


def test_frozen_surface_record_recomputes_from_exact_sources() -> None:
    protocol, _ = builder.load_protocol(PROTOCOL_PATH)
    artifact = builder.core.load_json(FROZEN_SURFACE_PATH)
    manifest = builder.core.load_json(SOURCE_MANIFEST_PATH)
    source_by_id = {record["compound_id"]: record for record in manifest["records"]}
    frozen = artifact["records"][0]
    source = source_by_id[frozen["compound_id"]]
    atoms = builder.MOL2Reader(str(SOURCE_ROOT / source["source_mol2_relative_path"]))
    positions = atoms.get_positions()
    charges = np.asarray(source["am1bcc_charges_e"])
    topology = builder.build_openmm_topology(atoms)
    radii_by_role = builder._profile_radii(topology)
    directions = builder.fibonacci_directions(
        protocol["surface_definition"]["quadrature"]["fine_direction_count"]
    )

    for role in builder.PROFILE_ORDER:
        recomputed = builder.surface_descriptors(
            positions,
            charges,
            radii_by_role[role].radii_angstrom,
            directions,
            **_surface_kwargs(),
        )
        assert builder._descriptor_view(recomputed) == pytest.approx(
            frozen["radius_profiles"][role]["descriptors"], abs=1e-15
        )


def test_frozen_retrospective_diagnostic_reports_only_descriptive_evidence() -> None:
    artifact = diagnostic.core.load_json(FROZEN_DIAGNOSTIC_PATH)

    assert artifact["content_sha256"] == diagnostic.core.artifact_content_sha256(
        artifact
    )
    assert artifact["case_count"] == 526
    assert (
        artifact["statistical_design"]["realized_bootstrap"]["all_clusters_singletons"]
        is True
    )
    primary = artifact["descriptive_associations"]["primary"][
        "absolute_cha_polar_mismatch_vs_fn2"
    ]
    assert primary["spearman_rho"] == pytest.approx(0.5118396287703332)
    assert primary["bootstrap_interval"] == pytest.approx(
        [0.41625379434417037, 0.6009863989970409]
    )
    gamma = artifact["descriptive_associations"]["primary"][
        "absolute_cha_polar_mismatch_vs_absolute_gamma_n"
    ]
    assert gamma["bootstrap_interval"][0] < 0.0 < gamma["bootstrap_interval"][1]
    assert artifact["decision"]["status"] == (
        "surface_electrostatic_scale_association_only_nonlinear_hypothesis_unresolved"
    )
    assert artifact["decision"]["protocol_rule_status"] == (
        "surface_field_hypothesis_retained_for_independent_test_only"
    )
    assert artifact["decision"]["post_result_adversarial_review_override"] is True
    assert artifact["decision"]["robust_absolute_gamma_signal"] is False
    assert artifact["decision"]["absolute_gamma_point_estimate_signs_agree"] is True
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
        assert artifact["decision"][key] is False


def test_retrospective_runner_reproduces_frozen_result(tmp_path: Path) -> None:
    reproduced = diagnostic.run(
        protocol_path=PROTOCOL_PATH,
        surface_artifact_path=FROZEN_SURFACE_PATH,
        component_artifact_path=COMPONENT_ARTIFACT_PATH,
        prepared_identity_path=PREPARED_IDENTITY_PATH,
        output_path=tmp_path / "surface-diagnostic.json",
    )
    frozen = diagnostic.core.load_json(FROZEN_DIAGNOSTIC_PATH)

    assert reproduced["descriptive_associations"] == frozen["descriptive_associations"]
    assert reproduced["statistical_design"] == frozen["statistical_design"]
    assert reproduced["decision"] == frozen["decision"]


def test_retrospective_join_refuses_invalid_surface_artifact(
    tmp_path: Path,
) -> None:
    artifact = copy.deepcopy(builder.core.load_json(FROZEN_SURFACE_PATH))
    artifact["numerical_validation"]["passed"] = False
    artifact["decision"]["status"] = "invalid_surface_artifact_no_association"
    artifact["decision"]["retrospective_association_allowed"] = False
    builder.core.seal_artifact(artifact)
    path = _write_json(tmp_path / "invalid-surface.json", artifact)

    with pytest.raises(ValueError, match="associations are forbidden"):
        diagnostic.run(
            protocol_path=PROTOCOL_PATH,
            surface_artifact_path=path,
            component_artifact_path=COMPONENT_ARTIFACT_PATH,
            prepared_identity_path=PREPARED_IDENTITY_PATH,
            output_path=tmp_path / "must-not-exist.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_retrospective_join_rejects_component_file_tamper(
    tmp_path: Path,
) -> None:
    component = copy.deepcopy(diagnostic.core.load_json(COMPONENT_ARTIFACT_PATH))
    component["records"][0]["polar_mismatch_kcal_mol"] += 0.1
    diagnostic.core.seal_artifact(component)
    path = _write_json(tmp_path / "tampered-component.json", component)

    with pytest.raises(ValueError, match="file hash mismatch"):
        diagnostic.run(
            protocol_path=PROTOCOL_PATH,
            surface_artifact_path=FROZEN_SURFACE_PATH,
            component_artifact_path=path,
            prepared_identity_path=PREPARED_IDENTITY_PATH,
            output_path=tmp_path / "must-not-exist.json",
        )
