from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = (
    ROOT / "docs/pretrained-solvation-hub/compare_resolv_cpu_gpu_parity.py"
)


def _load_comparator_module():
    spec = importlib.util.spec_from_file_location(
        "compare_resolv_cpu_gpu_parity",
        COMPARATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit(platform: str) -> dict:
    records = []
    for index in range(162):
        prediction = -5.0 + index / 100.0
        experiment = prediction + ((index % 5) - 2) / 10.0
        primary_group = (
            "__unclassified_non_group__"
            if index < 9
            else f"group-{(index - 9) % 26:02d}"
        )
        records.append(
            {
                "k_index": index,
                "trajectory_id": index + 1,
                "mobley_id": f"mobley_{index:07d}",
                "smiles": "C",
                "split": "test",
                "primary_functional_group": primary_group,
                "prediction_kcal_mol": prediction,
                "experimental_kcal_mol": experiment,
                "experimental_uncertainty_kcal_mol": 0.1,
            }
        )
    return {
        "scope": "full_published_test_split",
        "paper_split_reproduction_eligible": True,
        "repeat_determinism": {
            "repeat_count": 3,
            "identity_keyed_bitwise_stable": True,
            "minimum_repeats_for_parity_admission": 3,
        },
        "provenance": {
            "source_revision": "a" * 40,
            "artifact_sha256": {
                "vacuum_model": "1" * 64,
                "water_model": "2" * 64,
                "database": "3" * 64,
            },
            "manifest_sha256": "4" * 64,
            "trajectory_receipts": [
                {
                    "k_index": index,
                    "vacuum": {"sha256": f"{index:064x}"},
                    "water": {"sha256": f"{index + 1:064x}"},
                }
                for index in range(162)
            ],
            "code_sha256": {
                "audit_runner": "5" * 64,
                "worker": "6" * 64,
                "protocol": "7" * 64,
            },
            "executing_module_code_sha256": {
                "audit_runner": "8" * 64,
                "protocol": "9" * 64,
            },
        },
        "runtime": {
            "requested_platform": platform,
            "x64_enabled": True,
            "devices": [{"platform": platform}],
        },
        "worker": {
            "backend": platform,
            "devices": [f"{platform}:0"],
            "dtype": {"jax_enable_x64": True},
        },
        "primary_functional_group_count": 26,
        "classified_primary_functional_group_count": 26,
        "primary_functional_group_label_count_including_unclassified": 27,
        "unclassified_record_count": 9,
        "metrics": {
            "pooled": {"n": 162},
            "by_primary_functional_group": {
                group: {
                    "n": sum(
                        record["primary_functional_group"] == group
                        for record in records
                    )
                }
                for group in {
                    record["primary_functional_group"] for record in records
                }
            },
        },
        "records": records,
    }


def _write_audit_directory(path: Path, audit: dict) -> None:
    path.mkdir()
    manifest_path = path / "manifest.json"
    manifest_path.write_text(
        json.dumps(audit["records"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    audit["provenance"]["manifest_sha256"] = manifest_sha256
    audit_path = path / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    records_path = path / "records.csv"
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k_index",
                "trajectory_id",
                "mobley_id",
                "prediction_kcal_mol",
            ],
        )
        writer.writeheader()
        for record in audit["records"]:
            writer.writerow(
                {
                    field: record[field]
                    for field in writer.fieldnames
                }
            )
    runtime_diagnostic_path = path / "runtime-diagnostic.json"
    runtime_diagnostic_path.write_text(
        json.dumps(
            {
                "scope": (
                    "three_or_more_cold_worker_replays_not_full_task_timing"
                ),
                "repeat_count": audit["repeat_determinism"]["repeat_count"],
                "cold_end_to_end_seconds": [1.0, 1.1, 0.9],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (path / "receipt.json").write_text(
        json.dumps(
            {
                "audit_json_sha256": hashlib.sha256(
                    audit_path.read_bytes()
                ).hexdigest(),
                "records_csv_sha256": hashlib.sha256(
                    records_path.read_bytes()
                ).hexdigest(),
                "manifest_json_sha256": manifest_sha256,
                "runtime_diagnostic_sha256": hashlib.sha256(
                    runtime_diagnostic_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_evidence_files_are_read_once_and_parsed_from_retained_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparator = _load_comparator_module()
    audit_dir = tmp_path / "cpu"
    _write_audit_directory(audit_dir, _audit("cpu"))
    original_read_bytes = Path.read_bytes
    read_counts: dict[Path, int] = {}

    def read_once_then_mutate(path: Path) -> bytes:
        resolved = path.resolve()
        read_counts[resolved] = read_counts.get(resolved, 0) + 1
        content = original_read_bytes(path)
        if path.name == "audit.json":
            path.write_bytes(b'{"mutated_after_read": true}')
        return content

    monkeypatch.setattr(Path, "read_bytes", read_once_then_mutate)

    loaded = comparator.load_audit_directory(audit_dir, platform="cpu")

    assert len(loaded["records"]) == 162
    assert read_counts == {
        (audit_dir / name).resolve(): 1
        for name in (
            "audit.json",
            "records.csv",
            "manifest.json",
            "receipt.json",
            "runtime-diagnostic.json",
        )
    }


def test_exact_binary64_parity_passes_even_when_gpu_rows_are_shuffled() -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    random.Random(7).shuffle(gpu["records"])

    report = comparator.compare_audits(cpu, gpu)

    assert report["strict_zero_loss"] is True
    assert report["strict_failure_count"] == 0
    assert report["coverage"] == {
        "record_count": 162,
        "classified_primary_functional_group_count": 26,
        "unclassified_record_count": 9,
        "label_count_including_unclassified": 27,
        "full_162_record_26_group_9_unclassified_panel": True,
    }
    assert all(row["ulp_delta"] == 0 for row in report["records"])
    assert report["metrics_parity"]["pooled"]["gpu_minus_cpu"] == {
        "mae_kcal_mol": 0.0,
        "rmse_kcal_mol": 0.0,
        "maxae_kcal_mol": 0.0,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repeat_count", 2, "repeat_count"),
        (
            "identity_keyed_bitwise_stable",
            False,
            "bitwise repeat stability",
        ),
    ],
)
def test_canonical_repeat_admission_is_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    gpu["repeat_determinism"][field] = value

    with pytest.raises(comparator.ReSolvParityError, match=message):
        comparator.compare_audits(cpu, gpu)


def test_cli_writes_canonical_parity_and_separate_wall_time(
    tmp_path: Path,
) -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    cpu_dir = tmp_path / "cpu"
    gpu_dir = tmp_path / "gpu"
    output_dir = tmp_path / "parity"
    _write_audit_directory(cpu_dir, cpu)
    _write_audit_directory(gpu_dir, gpu)

    assert comparator.main(
        [
            "--cpu-audit-dir",
            str(cpu_dir),
            "--gpu-audit-dir",
            str(gpu_dir),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    canonical = json.loads(
        (output_dir / "cpu-gpu-parity.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (output_dir / "cpu-gpu-parity-runtime.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = json.loads(
        (output_dir / "cpu-gpu-parity-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert canonical["strict_zero_loss"] is True
    assert "comparison_wall_seconds" not in canonical
    assert runtime["comparison_wall_seconds"] >= 0.0
    assert receipt["cpu_gpu_parity_json_sha256"] == hashlib.sha256(
        (output_dir / "cpu-gpu-parity.json").read_bytes()
    ).hexdigest()


def test_comparator_source_mutation_during_comparison_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparator = _load_comparator_module()
    cpu_dir = tmp_path / "cpu"
    gpu_dir = tmp_path / "gpu"
    output_dir = tmp_path / "parity"
    _write_audit_directory(cpu_dir, _audit("cpu"))
    _write_audit_directory(gpu_dir, _audit("gpu"))
    original_read_bytes = Path.read_bytes
    comparator_reads = 0

    def mutate_second_comparator_read(path: Path) -> bytes:
        nonlocal comparator_reads
        content = original_read_bytes(path)
        if path.resolve() == COMPARATOR.resolve():
            comparator_reads += 1
            if comparator_reads == 2:
                return content + b"\n# concurrent mutation\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", mutate_second_comparator_read)

    with pytest.raises(comparator.ReSolvParityError, match="source changed"):
        comparator.main(
            [
                "--cpu-audit-dir",
                str(cpu_dir),
                "--gpu-audit-dir",
                str(gpu_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
    assert comparator_reads == 2


def test_retained_source_must_match_imported_module_code_object() -> None:
    comparator = _load_comparator_module()
    mutated = COMPARATOR.read_bytes().replace(
        b'"candidate": "ReSolv"',
        b'"candidate": "ReSolX"',
        1,
    )
    with pytest.raises(
        comparator.ReSolvParityError,
        match="module code does not match",
    ):
        comparator._attest_comparator_start_source(
            mutated,
            filename=str(COMPARATOR.resolve()),
        )


def test_one_ulp_fails_strict_zero_loss_even_inside_diagnostic_tolerance() -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    original = gpu["records"][10]["prediction_kcal_mol"]
    gpu["records"][10]["prediction_kcal_mol"] = math.nextafter(
        original,
        math.inf,
    )

    report = comparator.compare_audits(
        cpu,
        gpu,
        diagnostic_abs_tolerance=1.0e-12,
    )

    changed = next(row for row in report["records"] if row["k_index"] == 10)
    assert report["strict_zero_loss"] is False
    assert report["strict_failure_count"] == 1
    assert changed["ulp_delta"] == 1
    assert changed["diagnostic_within_abs_tolerance"] is True
    assert changed["cpu_prediction_ieee754_binary64_hex"] != (
        changed["gpu_prediction_ieee754_binary64_hex"]
    )
    assert "exact_experimental_error_worsening_kcal_mol" in changed


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra"])
def test_duplicate_missing_and_extra_record_identities_fail_closed(
    mutation: str,
) -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    if mutation == "duplicate":
        gpu["records"][-1] = copy.deepcopy(gpu["records"][0])
    elif mutation == "missing":
        gpu["records"].pop()
    else:
        gpu["records"][-1]["k_index"] = 999
        gpu["records"][-1]["trajectory_id"] = 1000
        gpu["records"][-1]["mobley_id"] = "mobley_extra"

    with pytest.raises(comparator.ReSolvParityError):
        comparator.compare_audits(cpu, gpu)


def test_unclassified_bucket_is_not_counted_as_a_functional_group() -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    for audit in (cpu, gpu):
        for record in audit["records"]:
            if record["primary_functional_group"] == "group-25":
                record["primary_functional_group"] = (
                    "__unclassified_non_group__"
                )
        audit["metrics"]["by_primary_functional_group"].pop("group-25")
        audit["metrics"]["by_primary_functional_group"][
            "__unclassified_non_group__"
        ]["n"] += 5

    with pytest.raises(
        comparator.ReSolvParityError,
        match="26 classified",
    ):
        comparator.compare_audits(cpu, gpu)


def test_exactly_nine_unclassified_records_are_required() -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    for audit in (cpu, gpu):
        audit["records"][0]["primary_functional_group"] = "group-00"

    with pytest.raises(comparator.ReSolvParityError, match="9 unclassified"):
        comparator.compare_audits(cpu, gpu)


@pytest.mark.parametrize(
    "field",
    [
        "source_revision",
        "artifact_sha256",
        "manifest_sha256",
        "trajectory_receipts",
        "code_sha256",
    ],
)
def test_code_artifact_and_manifest_identity_mismatches_fail_closed(
    field: str,
) -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    gpu["provenance"][field] = {"changed": True}

    with pytest.raises(comparator.ReSolvParityError, match=field):
        comparator.compare_audits(cpu, gpu)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("cpu", "requested_platform", "gpu"),
        ("gpu", "x64_enabled", False),
    ],
)
def test_platform_and_x64_receipt_mismatches_fail_closed(
    target: str,
    field: str,
    value: object,
) -> None:
    comparator = _load_comparator_module()
    cpu = _audit("cpu")
    gpu = _audit("gpu")
    {"cpu": cpu, "gpu": gpu}[target]["runtime"][field] = value

    with pytest.raises(comparator.ReSolvParityError):
        comparator.compare_audits(cpu, gpu)
