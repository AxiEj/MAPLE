from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "pretrained-solvation-hub" / "run_fennix_bio1_release_audit.py"
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "fennix-bio1-release-audit-2026-07-30.json"
)


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("fennix_bio1_release_audit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def test_required_token_gate_fails_closed():
    with pytest.raises(ValueError, match="missing required release tokens"):
        AUDIT._require_tokens(
            "model = fennol.load(model_file)",
            label="synthetic interface",
            required=("model = fennol.load(model_file)", "alch_elambda"),
        )


def test_revision_gate_rejects_wrong_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        AUDIT,
        "_git_bytes",
        lambda root, *args: b"0000000000000000000000000000000000000000\n",
    )

    with pytest.raises(ValueError, match="does not match"):
        AUDIT._verify_revision(
            tmp_path,
            label="synthetic",
            expected=AUDIT.FENNOL_REVISION,
        )


def test_checkpoint_identity_gate_rejects_wrong_size_and_hash(tmp_path):
    checkpoint = tmp_path / "model.fnx"
    checkpoint.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="size"):
        AUDIT._verify_weight(
            checkpoint,
            label="synthetic",
            expected={"size_bytes": 6, "sha256": "unused"},
        )

    with pytest.raises(ValueError, match="SHA256"):
        AUDIT._verify_weight(
            checkpoint,
            label="synthetic",
            expected={
                "size_bytes": 5,
                "sha256": "0" * 64,
            },
        )


def test_named_path_search_is_narrow_and_case_insensitive():
    paths = [
        "generic/lambda-example.key",
        "paper/FENNIX-BIO1-input.key",
        "results/FreeSolv-ledger.csv",
    ]

    assert AUDIT._named_paper_artifact_matches(paths) == [
        "paper/FENNIX-BIO1-input.key",
        "results/FreeSolv-ledger.csv",
    ]


def _synthetic_sources():
    interface = "\n".join(
        (
            *AUDIT.TINKER_INTERFACE_REQUIRED_TOKENS,
            *AUDIT.TINKER_GPU_NUMERIC_TOKENS,
        )
    ).encode()
    fennol = {
        "ase_calculator": "\n".join(AUDIT.FENNOL_ASE_PRECISION_TOKENS).encode(),
        "md_runtime": "\n".join(AUDIT.FENNOL_MD_PRECISION_TOKENS).encode(),
        "package_metadata": b"synthetic package metadata",
    }
    tinker = {
        "gpu_fennol_lambda_interface": interface,
        "gpu_fennol_lambda_header": b"synthetic header",
        "fortran_gpu_lambda_bridge": "\n".join(
            AUDIT.TINKER_FORTRAN_LAMBDA_TOKENS
        ).encode(),
        "deep_hp_documentation": b"synthetic Deep-HP documentation",
        "lambda_abf_documentation": b"synthetic Lambda-ABF documentation",
        "generic_gpu_example_key": b"ML-MODEL ../ml_models/ani2x.fnx\n",
    }
    return fennol, tinker


def test_audit_payload_is_deterministic_and_gpu_gate_remains_closed(
    monkeypatch, tmp_path
):
    fennol_sources, tinker_sources = _synthetic_sources()
    monkeypatch.setattr(AUDIT, "_verify_revision", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(AUDIT, "_verify_weight", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        AUDIT,
        "_read_verified_blob",
        lambda root, *, label, specification: (
            fennol_sources if "FeNNol" in label else tinker_sources
        )[
            next(
                name
                for name, candidate in (
                    AUDIT.FENNOL_BLOBS.items()
                    if "FeNNol" in label
                    else AUDIT.TINKER_HP_BLOBS.items()
                )
                if candidate is specification
            )
        ],
    )
    monkeypatch.setattr(
        AUDIT,
        "_tree_paths",
        lambda root: ["v1.3/GPU/examples/Deep-HP_example.key"],
    )

    first = AUDIT.audit_release(
        tmp_path / "fennol",
        tmp_path / "tinker",
        tmp_path / "small.fnx",
        tmp_path / "medium.fnx",
    )
    second = AUDIT.audit_release(
        tmp_path / "fennol",
        tmp_path / "tinker",
        tmp_path / "small.fnx",
        tmp_path / "medium.fnx",
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    AUDIT._write_json_atomic(first_path, first)
    AUDIT._write_json_atomic(second_path, second)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["acceptance_eligible"] is False
    assert (
        first["source_capabilities"]["combined_fennol_tinker_gpu_source_present"]
        is True
    )
    assert first["gpu_acceleration"]["no_precision_loss_demonstrated"] is False
    assert first["gpu_acceleration"]["parity_executed"] == {
        "coverage": False,
        "energy": False,
        "final_free_energy_and_uncertainty": False,
        "forces": False,
        "lambda_derivatives": False,
        "maximum_error": False,
        "trajectory_or_sampled_observables": False,
        "virial": False,
    }
    assert first["validation_scope"]["predicted_rows_in_artifact"] is False
    assert "records" not in first


def test_frozen_release_artifact_preserves_negative_admission_and_no_loss_gate():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["acceptance_eligible"] is False
    assert (
        payload["identity"]["weights"]["small"]["sha256"]
        == AUDIT.WEIGHTS["small"]["sha256"]
    )
    assert (
        payload["identity"]["weights"]["medium"]["sha256"]
        == AUDIT.WEIGHTS["medium"]["sha256"]
    )
    assert payload["runtime_sources"]["fennol"]["revision"] == AUDIT.FENNOL_REVISION
    assert (
        payload["runtime_sources"]["tinker_hp"]["revision"] == AUDIT.TINKER_HP_REVISION
    )
    assert payload["runtime_sources"]["tinker_hp"]["verified_blobs"][
        "fortran_gpu_lambda_bridge"
    ] == AUDIT._verified_blob_record(AUDIT.TINKER_HP_BLOBS["fortran_gpu_lambda_bridge"])
    assert (
        payload["source_capabilities"][
            "lambda_energy_and_vdw_derivative_outputs_present"
        ]
        is True
    )
    assert (
        payload["paper_protocol_packaging"][
            "exact_fennix_bio1_freesolv_bundle_verified"
        ]
        is False
    )
    assert payload["gpu_acceleration"]["admission_status"] == (
        "blocked_until_no_loss_reference_gpu_parity"
    )
    assert payload["gpu_acceleration"]["no_precision_loss_demonstrated"] is False
    assert payload["gpu_acceleration"]["parity_executed"] == {
        "coverage": False,
        "energy": False,
        "final_free_energy_and_uncertainty": False,
        "forces": False,
        "lambda_derivatives": False,
        "maximum_error": False,
        "trajectory_or_sampled_observables": False,
        "virial": False,
    }
    assert payload["validation_scope"]["multi_solvent_evidence"] is False
    assert "records" not in payload


def test_frozen_release_artifact_equals_fresh_exact_source_audit():
    inputs = {
        "fennol_root": Path(
            os.environ.get(
                "MAPLE_FENNOL_AUDIT_ROOT",
                "/tmp/route4-fennix-runtime-audit/FeNNol",
            )
        ),
        "tinker_hp_root": Path(
            os.environ.get(
                "MAPLE_TINKER_HP_AUDIT_ROOT",
                "/tmp/route4-tinker-hp-audit-v1",
            )
        ),
        "small_checkpoint": Path(
            os.environ.get(
                "MAPLE_FENNIX_BIO1_SMALL",
                "/tmp/route4-fennix-bio1-audit/fennix-bio1S.fnx",
            )
        ),
        "medium_checkpoint": Path(
            os.environ.get(
                "MAPLE_FENNIX_BIO1_MEDIUM",
                "/tmp/route4-fennix-bio1-audit/fennix-bio1M.fnx",
            )
        ),
    }
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        pytest.skip(
            "Exact FeNNix/Tinker release inputs are not available: "
            + ", ".join(missing)
        )

    fresh = AUDIT.audit_release(**inputs)
    rendered = json.dumps(fresh, indent=2, sort_keys=True) + "\n"

    assert json.loads(ARTIFACT.read_text(encoding="utf-8")) == fresh
    assert ARTIFACT.read_text(encoding="utf-8") == rendered
