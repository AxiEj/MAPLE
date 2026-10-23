"""Keep active validation inventories executable after research retirement."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import run_mlip_device_canary as canary
import run_torch_rewrite_revalidation as revalidation


def test_active_validation_inventories_resolve_to_existing_files():
    historical = revalidation.core.load_json(revalidation.HISTORICAL)
    paths = set(revalidation.source_paths(historical)) | set(canary.SOURCE_PATHS)
    assert paths
    assert all((ROOT / path).is_file() for path in paths)
    assert all((ROOT / path).is_file() for path in revalidation.TESTS)


def test_retained_validation_covers_exact_cha_and_obc_workflows():
    required = {
        "test_torch_chagb.py",
        "test_torch_pbsa_exact_cavity.py",
        "test_chagb_exact_sp_contract.py",
        "test_chagb_polar_oracle.py",
        "test_pbsa_cavity_portable_oracle.py",
        "test_torch_obc2_parity.py",
        "test_torch_obc2_derivatives.py",
        "test_implicit_analytic_derivatives.py",
    }
    assert required <= {Path(path).name for path in revalidation.TESTS}


def test_validation_sources_cover_runtime_and_evidence_helpers():
    historical = revalidation.core.load_json(revalidation.HISTORICAL)
    sources = set(revalidation.source_paths(historical))
    runtime = {str(path.relative_to(ROOT)) for path in (ROOT / "maple").rglob("*.py")}
    assert runtime <= sources
    assert set(canary.SOURCE_PATHS) <= sources
    assert "docs/implicit-solvation/benchmarks/benchmark_core.py" in sources


def test_retired_continuum_prototypes_are_not_in_the_active_package():
    package = ROOT / "maple/function/calculator/extra_correction/implicit"
    retired = (
        "continuum_chagb_inputs",
        "torch_continuum_sav",
        "torch_continuum_dispersion",
        "torch_sphere_union_geometry",
        "sphere_union_volume",
        "sphere_union_dispersion",
    )
    assert not [name for name in retired if (package / f"{name}.py").exists()]
