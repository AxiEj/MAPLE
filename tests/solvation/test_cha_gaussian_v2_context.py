"""Explicit v1/v2 runner-context and receipt-binding contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RUNNER = ROOT / "docs/implicit-solvation/benchmarks/run_cha_gaussian_opt.py"


def _runner():
    spec = importlib.util.spec_from_file_location("cha_gaussian_context_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closed_contexts_bind_distinct_profile_protocol_plan_output_and_factory():
    runner = _runner()
    v1 = runner.DEFAULT_V1_CONTEXT
    v2 = runner.V2_RUN_CONTEXT
    assert v1.numerical_profile_id == runner.V1_PROFILE_ID
    assert v2.numerical_profile_id == runner.V2_PROFILE_ID
    assert v1.protocol_template != v2.protocol_template
    assert v1.approved_plan_sha256 != v2.approved_plan_sha256
    assert v1.output_root != v2.output_root
    assert v1.calculator_factory.numerical_profile_id == runner.V1_PROFILE_ID
    assert v2.calculator_factory.numerical_profile_id == runner.V2_PROFILE_ID
    with pytest.raises(AttributeError):
        v2.numerical_profile_id = runner.V1_PROFILE_ID
    with pytest.raises(ValueError, match="Unknown Gaussian-CHA numerical profile"):
        runner._context_for_profile("unknown-profile")


def test_v2_wrapper_protocol_is_small_pinned_and_resolves_unchanged_base_contract():
    runner = _runner()
    protocol, digest = runner.load_protocol_template(runner.V2_RUN_CONTEXT)
    assert digest == runner.V2_RUN_CONTEXT.protocol_template_sha256
    assert protocol["numerical_profile_id"] == runner.V2_PROFILE_ID
    assert protocol["context_protocol_sha256"] == digest
    assert protocol["protocol_id"] == "route1-cha-r6-derivative-repair-v2-20261001"
    assert protocol["inputs"]["sigma_e"] == [0.001, 0.003, 0.01]
    assert protocol["scalar"]["production_quadrature_order"] == 64
    assert protocol["execution"]["campaign_output"] == (
        "unique direct child campaign-v2-YYYYMMDDTHHMMSSZ"
    )
    assert protocol["optimizer"]["thresholds"] == {
        "f_max_hartree_per_angstrom": 3e-4,
        "f_rms_hartree_per_angstrom": 2e-4,
        "dp_max_angstrom": 3e-4,
        "dp_rms_angstrom": 2e-4,
    }


def test_every_bound_receipt_records_context_and_numerical_profile():
    runner = _runner()
    evidence = {
        "preregistration_sha256": "1" * 64,
        "source_identity_sha256": "2" * 64,
        "numerical_profile_id": runner.V2_PROFILE_ID,
        "context_id": runner.V2_RUN_CONTEXT.context_id,
        "protocol_template_sha256": runner.V2_RUN_CONTEXT.protocol_template_sha256,
        "approved_plan_sha256": runner.V2_RUN_CONTEXT.approved_plan_sha256,
        "approved_handoff_sha256": runner.V2_RUN_CONTEXT.approved_handoff_sha256,
        "protocol_id": "route1-cha-r6-derivative-repair-v2-20261001",
    }
    identity = {"row_id": "center-00-sigma-00"}
    receipt = runner._bind_receipt(
        {"status": "fixture"},
        evidence,
        receipt_kind="FIXTURE",
        case_id="center-00-sigma-00",
        case_identity=identity,
    )
    runner._verify_receipt_binding(
        receipt,
        evidence,
        receipt_kind="FIXTURE",
        case_id="center-00-sigma-00",
        case_identity=identity,
    )
    assert receipt["numerical_profile_id"] == runner.V2_PROFILE_ID
    assert receipt["context_id"] == runner.V2_RUN_CONTEXT.context_id


def test_unknown_cli_profile_fails_before_runtime_or_model_initialization(
    monkeypatch, tmp_path
):
    runner = _runner()
    monkeypatch.setattr(
        runner,
        "assert_runtime_origins",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime initialization must not be reached")
        ),
    )
    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "preregister",
                "--output",
                str(tmp_path),
                "--profile",
                "unknown-profile",
            ]
        )
    assert exc.value.code == 2


def test_v2_entry_context_rejects_v1_profile_before_execution(tmp_path):
    runner = _runner()
    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "preregister",
                "--output",
                str(tmp_path),
                "--profile",
                runner.V1_PROFILE_ID,
            ],
            context=runner.V2_RUN_CONTEXT,
        )
    assert exc.value.code == 2


def test_v2_campaign_boundary_accepts_only_unique_direct_campaign_children():
    runner = _runner()
    context = runner.V2_RUN_CONTEXT
    root = context.output_root
    assert not runner._output_allowed(root, context)
    assert not runner._output_allowed(root / "planning", context)
    assert not runner._output_allowed(root / "core-lane", context)
    assert not runner._output_allowed(root / "engine-lane", context)
    assert not runner._output_allowed(root / "campaign-v2-not-utc", context)
    assert not runner._output_allowed(root / "other-20261001T010203Z", context)
    assert runner._output_allowed(root / "campaign-v2-20261001T010203Z", context)
    assert not runner._output_allowed(
        root / "campaign-v2-20261001T010203Z" / "nested", context
    )


def test_v2_factory_is_immutably_owned_by_its_context():
    runner = _runner()
    context = runner.V2_RUN_CONTEXT
    factory = context.calculator_factory
    assert factory.numerical_profile_id == context.numerical_profile_id
    assert factory.worktree == context.worktree
    assert factory.root == context.root
    assert factory.context_id == context.context_id
    with pytest.raises(AttributeError):
        factory.context_id = "wrong"


def test_profile_provenance_validator_rejects_wrong_profile_backend_order_and_topology():
    runner = _runner()
    topology_sha = "a" * 64
    sigma = 0.01
    provenance = {
        "numerical_profile_id": runner.V2_PROFILE_ID,
        "topology_sha256": topology_sha,
        "sigma_e": sigma,
        "quadrature_order": 64,
        "r6_backend_diagnostics": {
            "profile_id": runner.V2_PROFILE_ID,
            "backend_function": "torch_continuum_r6_derivative_v2._r6_inverse_born_v2",
            "order": 64,
        },
    }
    runner._validate_solvation_profile(
        {"provenance": provenance}, runner.V2_PROFILE_ID, topology_sha, sigma, 64
    )
    for key, wrong in (
        ("numerical_profile_id", runner.V1_PROFILE_ID),
        ("topology_sha256", "b" * 64),
        ("quadrature_order", 96),
    ):
        changed = {**provenance, key: wrong}
        with pytest.raises(ValueError, match="solvation provenance"):
            runner._validate_solvation_profile(
                {"provenance": changed}, runner.V2_PROFILE_ID, topology_sha, sigma, 64
            )
    changed_backend = {
        **provenance,
        "r6_backend_diagnostics": {
            **provenance["r6_backend_diagnostics"],
            "backend_function": "torch_continuum_chagb._r6_inverse_born",
        },
    }
    with pytest.raises(ValueError, match="solvation provenance"):
        runner._validate_solvation_profile(
            {"provenance": changed_backend},
            runner.V2_PROFILE_ID,
            topology_sha,
            sigma,
            64,
        )
