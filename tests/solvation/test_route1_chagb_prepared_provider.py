from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_core import sha256_file  # pyright: ignore[reportMissingImports]
from run_route1_chagb_prepared_provider import (  # pyright: ignore[reportMissingImports]
    load_protocol,
    validate_artifact,
)


PROTOCOL_PATH = BENCHMARK_DIR / "route1_chagb_prepared_provider_protocol_v1.json"
ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-chagb-prepared-provider-parity-2026-07-29.json"
)
PROVIDER_PATH = (
    REPOSITORY_ROOT
    / "maple/function/calculator/extra_correction/implicit/amber_chagb.py"
)
RUNNER_PATH = BENCHMARK_DIR / "run_route1_chagb_prepared_provider.py"


def test_prepared_provider_protocol_keeps_the_sp_only_route_boundary():
    protocol, fingerprint = load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert protocol["route_contract"]["gas_phase_mm_energy_used"] is False
    assert protocol["route_contract"]["bonded_mm_energy_used"] is False
    assert protocol["route_contract"]["hydration_label_fit_or_residual"] is False
    assert protocol["route_contract"]["supported_tasks"] == ["sp"]
    assert protocol["comparison"]["persistent_external_worker_claimed"] is False
    assert len(protocol["coordinate_cases"]) == 4


def test_prepared_provider_artifact_locks_live_component_parity_and_cache_scope():
    artifact = validate_artifact(ARTIFACT_PATH)

    assert artifact["recorded_date"] == "2026-07-29"
    assert artifact["component_parity"]["tolerance_kcal_mol"] == 1.0e-9
    assert artifact["component_parity"]["maximum_absolute_delta_kcal_mol"] <= 1.0e-9
    assert artifact["cache_contract"] == {
        "coordinate_only_evaluation_count": 4,
        "expected_cache_hits": [False, True, True, True],
        "observed_cache_hits": [False, True, True, True],
        "topology_preparation_count": 1,
    }
    assert artifact["implementation"]["prepared_provider_sha256"] == sha256_file(
        PROVIDER_PATH
    )
    assert artifact["command"]["script_sha256"] == sha256_file(RUNNER_PATH)
    assert artifact["protocol"]["sha256"] == sha256_file(PROTOCOL_PATH)
    assert set(artifact["execution"]["executables"]) == {
        "gbnsr6",
        "parmchk2",
        "pbsa",
        "tleap",
    }
    assert artifact["prepared_topology"]["cache_scope"] == "provider-instance"
    assert artifact["implementation"]["persistent_external_worker_claimed"] is False
