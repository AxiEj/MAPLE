from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
RUNNER_PATH = BENCHMARK_DIR / "run_route1_surface_electrostatics_scale_sensitivity.py"
PROTOCOL_PATH = (
    BENCHMARK_DIR / "route1_surface_electrostatics_scale_sensitivity_protocol_v1.json"
)
SURFACE_PROTOCOL_PATH = BENCHMARK_DIR / "route1_surface_electrostatics_protocol_v1.json"
SURFACE_ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-freesolv-surface-electrostatics-label-free-2026-07-29.json"
)
MISMATCH_ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-freesolv-explicit-component-diagnostic-2026-07-29.json"
)
ENERGY_ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-chagb-component-attribution-2026-07-25.json"
)
PREPARED_IDENTITY_PATH = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/neutral-water-freesolv-route1-20260723/prepared.json"
)
FROZEN_ARTIFACT_PATH = (
    BENCHMARK_DIR
    / "route1-freesolv-surface-electrostatics-scale-sensitivity-2026-07-29.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_module("route1_surface_scale_sensitivity", RUNNER_PATH)


def _write_json(path: Path, value: object) -> Path:
    runner.core.write_json_atomic(path, value)
    return path


def _run(output_path: Path, **overrides):
    arguments = {
        "protocol_path": PROTOCOL_PATH,
        "surface_protocol_path": SURFACE_PROTOCOL_PATH,
        "surface_artifact_path": SURFACE_ARTIFACT_PATH,
        "mismatch_artifact_path": MISMATCH_ARTIFACT_PATH,
        "energy_artifact_path": ENERGY_ARTIFACT_PATH,
        "prepared_identity_path": PREPARED_IDENTITY_PATH,
        "output_path": output_path,
    }
    arguments.update(overrides)
    return runner.run(**arguments)


def _decision_results(
    *,
    fn2_primary: list[float],
    fn2_control: list[float],
    phi2_primary: list[float],
    phi2_control: list[float],
) -> dict[str, object]:
    def role(fn2: list[float], phi2: list[float]) -> dict[str, object]:
        return {
            "partial_y_fn2_given_z_and_area": {"bootstrap_interval": fn2},
            "partial_y_phi2_given_z_and_area": {"bootstrap_interval": phi2},
        }

    return {
        "primary": role(fn2_primary, phi2_primary),
        "control": role(fn2_control, phi2_control),
    }


def test_scale_protocol_locks_post_v1_no_action_boundary(
    tmp_path: Path,
) -> None:
    protocol, protocol_sha256 = runner.load_protocol(PROTOCOL_PATH)

    assert protocol_sha256 == runner.core.sha256_bytes(
        runner.core.canonical_json_bytes(protocol)
    )
    boundary = protocol["historical_boundary"]
    assert boundary["v1_result_was_seen_before_this_design"] is True
    assert boundary["independent_confirmation"] is False
    assert boundary["radius_profile_agreement_is_independent_replication"] is False
    for key in (
        "no_energy_fit",
        "no_residual_correction",
        "no_provider_implementation",
        "no_threshold_tuning",
        "no_first_shell_causal_claim",
        "no_ranking_certification",
    ):
        assert boundary[key] is True

    changed = copy.deepcopy(protocol)
    changed["historical_boundary"]["independent_confirmation"] = True
    path = _write_json(tmp_path / "false-confirmation.json", changed)
    with pytest.raises(ValueError, match="label/truth boundary changed"):
        runner.load_protocol(path)

    changed = copy.deepcopy(protocol)
    changed["decision_rule"]["parameter_update_allowed"] = True
    path = _write_json(tmp_path / "parameter-update.json", changed)
    with pytest.raises(ValueError, match="Unsupported scale-sensitivity action"):
        runner.load_protocol(path)

    changed = copy.deepcopy(protocol)
    changed["statistics"]["bootstrap_unit"] = "compound_id"
    path = _write_json(tmp_path / "bootstrap-unit.json", changed)
    with pytest.raises(ValueError, match="bootstrap unit changed"):
        runner.load_protocol(path)


def test_weighted_bootstrap_partial_rank_matches_expanded_samples() -> None:
    outcome = np.asarray([5.0, 0.0, 1.0, 4.0, 2.0, 6.0, 3.0, 7.0])
    proxy = np.asarray([1.0, 6.0, 7.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    controls = [
        np.asarray([1.0, 3.0, 0.0, 5.0, 4.0, 6.0, 7.0, 2.0]),
        np.asarray([3.0, 1.0, 6.0, 7.0, 4.0, 5.0, 0.0, 2.0]),
    ]
    record_counts = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [2, 0, 1, 1, 0, 2, 1, 1],
            [0, 2, 2, 0, 1, 1, 1, 1],
        ],
        dtype=np.int16,
    )

    observed = runner._bootstrap_partial_rank(
        outcome,
        proxy,
        controls,
        record_counts,
        batch_size=2,
    )
    expected = []
    for counts in record_counts:
        expected.append(
            runner._partial_rank(
                np.repeat(outcome, counts),
                np.repeat(proxy, counts),
                [np.repeat(control, counts) for control in controls],
            )
        )
    assert observed == pytest.approx(expected, abs=1e-14)


def test_scale_decision_requires_cross_profile_stability() -> None:
    protocol, _ = runner.load_protocol(PROTOCOL_PATH)
    zero = [-0.1, 0.1]
    positive = [0.01, 0.2]
    negative = [-0.2, -0.01]

    stopped = runner._decision(
        protocol,
        _decision_results(
            fn2_primary=zero,
            fn2_control=zero,
            phi2_primary=zero,
            phi2_control=zero,
        ),
    )
    assert stopped["status"] == (
        "scale_association_only_stop_surface_proxy_to_provider_inference"
    )
    assert stopped["candidate_proxy"] is None

    candidate = runner._decision(
        protocol,
        _decision_results(
            fn2_primary=positive,
            fn2_control=positive,
            phi2_primary=zero,
            phi2_control=zero,
        ),
    )
    assert candidate["status"] == (
        "candidate_for_new_pre_registered_independent_confirmation_only"
    )
    assert candidate["candidate_proxy"] == "fn2"

    unresolved = runner._decision(
        protocol,
        _decision_results(
            fn2_primary=positive,
            fn2_control=positive,
            phi2_primary=positive,
            phi2_control=negative,
        ),
    )
    assert unresolved["status"] == "surface_definition_sensitive_or_unresolved"
    assert unresolved["candidate_proxy"] is None


def test_frozen_scale_sensitivity_artifact_is_sealed_and_stops_inference() -> None:
    artifact = runner.core.load_json(FROZEN_ARTIFACT_PATH)

    assert artifact["content_sha256"] == runner.core.artifact_content_sha256(artifact)
    assert artifact["case_count"] == 526
    assert artifact["designation"] == (
        "post-v1 reviewer-informed adversarial sensitivity analysis"
    )
    assert artifact["historical_boundary"]["independent_confirmation"] is False
    assert artifact["decision"]["status"] == (
        "scale_association_only_stop_surface_proxy_to_provider_inference"
    )
    assert artifact["decision"]["candidate_proxy"] is None
    assert artifact["radius_profile_sensitivity_boundary"] == {
        "identical_radius_vector_count": 464,
        "different_radius_vector_count": 62,
        "identical_radius_vector_fraction": pytest.approx(0.8821292775665399),
        "interpretation": (
            "Bondi-family and mbondi2 agreement is a sensitivity check, "
            "not independent replication, because most radius vectors "
            "are identical."
        ),
    }
    assert artifact["command_provenance"]["script_sha256"] == (
        runner.core.sha256_file(RUNNER_PATH)
    )

    primary = artifact["scale_sensitivity_results"]["primary"]
    assert primary["partial_y_fn2_given_z_and_area"][
        "partial_spearman_rho"
    ] == pytest.approx(0.06066689565681485)
    assert primary["partial_y_fn2_given_z_and_area"][
        "bootstrap_interval"
    ] == pytest.approx([-0.05341176274214992, 0.17313004213328936])
    assert primary["partial_y_phi2_given_z_and_area"][
        "partial_spearman_rho"
    ] == pytest.approx(0.062041308688003)
    assert primary["partial_y_phi2_given_z_and_area"][
        "bootstrap_interval"
    ] == pytest.approx([-0.04434824750054741, 0.16999060279998995])

    for role in runner.PROFILE_ORDER:
        for metric in runner.PRIMARY_METRICS:
            lower, upper = artifact["scale_sensitivity_results"][role][metric][
                "bootstrap_interval"
            ]
            assert lower < 0.0 < upper
    for key in (
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


def test_scale_runner_reproduces_frozen_result(tmp_path: Path) -> None:
    reproduced = _run(tmp_path / "scale-sensitivity.json")
    frozen = runner.core.load_json(FROZEN_ARTIFACT_PATH)

    assert (
        reproduced["scale_sensitivity_results"] == frozen["scale_sensitivity_results"]
    )
    assert reproduced["statistical_design"] == frozen["statistical_design"]
    assert (
        reproduced["radius_profile_sensitivity_boundary"]
        == frozen["radius_profile_sensitivity_boundary"]
    )
    assert reproduced["decision"] == frozen["decision"]


def test_scale_runner_rejects_tampered_energy_source(
    tmp_path: Path,
) -> None:
    energy = copy.deepcopy(runner.core.load_json(ENERGY_ARTIFACT_PATH))
    energy["records"][0]["components_kcal_mol"]["chagb_polar"] += 0.01
    runner.core.seal_artifact(energy)
    tampered_path = _write_json(tmp_path / "tampered-energy.json", energy)
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="file hash mismatch"):
        _run(
            output_path,
            energy_artifact_path=tampered_path,
        )
    assert not output_path.exists()
