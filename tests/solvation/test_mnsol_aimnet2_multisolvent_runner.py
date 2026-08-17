from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_mnsol_aimnet2_multisolvent_pilot as runner  # noqa: E402


def _record(
    *,
    experimental: float = -4.0,
    ddpcm_total: float = -4.5,
    ddcosmo_total: float = -4.1,
) -> dict[str, object]:
    def _method(total: float) -> dict[str, float]:
        signed_error = total - experimental
        return {
            "total_solvation_kcal_mol": total,
            "signed_error_kcal_mol": signed_error,
            "absolute_error_kcal_mol": abs(signed_error),
            "polarization_energy_kcal_mol": total - 0.5,
            "half_coupling_identity_error_ev": 2.0e-14,
        }

    return {
        "experimental_delta_g_kcal_mol": experimental,
        "smd_cds_energy_kcal_mol": 0.5,
        "methods": {
            "ddpcm": _method(ddpcm_total),
            "ddcosmo": _method(ddcosmo_total),
        },
    }


def test_single_record_smoke_selection_preserves_full_panel_contract():
    full_selection = tuple(f"record-{index}" for index in range(10))

    indexed, complete = runner._indexed_selection(full_selection, None)
    assert complete is True
    assert indexed == list(enumerate(full_selection))

    indexed, complete = runner._indexed_selection(full_selection, 3)
    assert complete is False
    assert indexed == [(3, "record-3")]

    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        runner._indexed_selection(full_selection, 10)
    with pytest.raises(RuntimeError, match="exactly 10"):
        runner._indexed_selection(full_selection[:-1], None)


def test_runtime_versions_come_from_imported_provider_modules(monkeypatch):
    modules = {
        "pyddx": SimpleNamespace(__version__="0.8.0-custom"),
        "pyscf": SimpleNamespace(__version__="2.13.1-custom"),
    }
    monkeypatch.setattr(runner, "import_module", modules.__getitem__)

    assert runner._runtime_versions() == {
        "pyddx": "0.8.0-custom",
        "pyscf": "2.13.1-custom",
    }


def test_single_record_smoke_metrics_are_descriptive_not_ten_only():
    records = [_record()]

    metrics = runner._metrics(records, "ddcosmo")
    paired = runner._paired_method_comparison(records)

    assert metrics["record_count"] == 1
    assert metrics["mean_signed_error_kcal_mol"] == pytest.approx(-0.1)
    assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(0.1)
    assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(0.1)
    assert paired["record_count"] == 1
    assert paired["ddcosmo_lower_absolute_error_count"] == 1
    assert paired["ddpcm_lower_absolute_error_count"] == 0

    with pytest.raises(RuntimeError, match="finite nonempty"):
        runner._metrics([], "ddpcm")
    with pytest.raises(RuntimeError, match="finite nonempty"):
        runner._paired_method_comparison([])


def test_single_record_smoke_forces_both_outputs_below_omx():
    private = ROOT / ".omx/test-aimnet2-smoke/private.json"
    public = ROOT / ".omx/test-aimnet2-smoke/summary.json"

    resolved = runner._validated_output_paths(
        private_output=private,
        public_output=public,
        complete_panel=False,
    )
    assert resolved == (private.resolve(), public.resolve())

    with pytest.raises(ValueError, match="Row-level MNSol pilot"):
        runner._validated_output_paths(
            private_output=private,
            public_output=ROOT / "docs/smoke-summary.json",
            complete_panel=False,
        )

    full_paths = runner._validated_output_paths(
        private_output=private,
        public_output=ROOT / "docs/full-summary.json",
        complete_panel=True,
    )
    assert full_paths[1] == (ROOT / "docs/full-summary.json").resolve()


def test_partition_shard_claim_names_its_actual_partition():
    for partition in ("development", "confirmation"):
        boundary = runner._claim_boundary(
            complete_panel=False,
            partition_shard=True,
            partition=partition,
        )
        assert f"MNSol {partition}-partition shard" in boundary
        assert "remain private under .omx" in boundary

    with pytest.raises(ValueError, match="requires its frozen partition"):
        runner._claim_boundary(
            complete_panel=False,
            partition_shard=True,
            partition=None,
        )
