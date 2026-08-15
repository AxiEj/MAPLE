from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    REPO_ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-matched-qm-pcmsolver-four-prereg-v1.json"
)
INHERITED = (
    REPO_ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
RUNNER = REPO_ROOT / "tools/route2_release/run_matched_qm_pcmsolver_decomposition.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload() -> dict[str, object]:
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def test_preregistration_freezes_source_independent_matched_reference() -> None:
    payload = _payload()
    assert payload["protocol_id"] == "route2-matched-qm-pcmsolver-four-prereg-v1"
    assert payload["status"] == "frozen-before-execution"
    assert payload["case_ids"] == [
        "mobley_3053621",
        "mobley_3867265",
        "mobley_3034976",
        "mobley_352111",
    ]
    assert payload["reference_protocol"]["electronic_structure"] == "omegaB97M-V"
    assert payload["reference_protocol"]["basis"] == "def2-tzvpd"
    assert payload["reference_protocol"]["pcm_state"].startswith(
        "self-consistent PySCF density"
    )
    continuum = payload["continuum_protocol"]
    assert continuum["method"] == "IEFPCM"
    assert continuum["matrix_symmetrization_required"] is True
    assert continuum["dielectric"] == 78.355
    assert continuum["nonpolar_terms_included"] is False
    assert continuum["standard_state_terms_included"] is False
    assert "complete panel twice" in payload["execution_contract"]["repeat_policy"]
    assert any(
        "ML source" in item for item in payload["claim_boundary"]["cannot_establish"]
    )


def test_preregistration_binds_current_sources_and_inherited_assets() -> None:
    payload = _payload()
    inherited = payload["inherited_asset_contract"]
    assert inherited["path"] == str(INHERITED.relative_to(REPO_ROOT))
    assert inherited["sha256"] == _sha256(INHERITED)
    source_hashes = payload["execution_contract"]["source_sha256"]
    assert source_hashes
    for relative, expected in source_hashes.items():
        assert _sha256(REPO_ROOT / relative) == expected
    cases = payload["cases"]
    assert len(cases) == 4
    assert len({case["case_id"] for case in cases}) == 4
    for case in cases:
        for key in (
            "mol2",
            "projection_result",
            "gas_checkpoint",
            "gas_ledger",
            "pcm_input",
            "frozen_surface",
            "frozen_gas_mep",
        ):
            record = case[key]
            assert record["path"].startswith(".omx/")
            assert len(record["sha256"]) == 64


def test_numerical_gates_cover_energy_pairing_scf_and_replay() -> None:
    gates = _payload()["numerical_gates"]
    assert gates["gas_checkpoint_replay_abs_hartree_max"] <= 1.0e-9
    assert gates["surface_replay_abs_bohr_max"] <= 1.0e-12
    assert gates["response_symmetry_relative_max"] <= 1.0e-12
    assert gates["half_coupling_abs_hartree_max"] <= 1.0e-12
    assert gates["solvent_energy_fd_abs_hartree_max"] <= 1.0e-8
    assert gates["decomposition_closure_abs_hartree_max"] <= 1.0e-10
    assert gates["electron_count_abs_e_max"] <= 1.0e-8
    assert gates["orbital_gradient_inf_max"] <= 1.0e-6
    assert gates["all_scf_states_converged"] is True


def test_runner_help_keeps_pyscf_out_of_the_import_graph() -> None:
    code = f"""
import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pyscf" or name.startswith("pyscf."):
        raise RuntimeError("pyscf import forbidden before execution")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [{str(RUNNER)!r}, "--help"]
try:
    runpy.run_path({str(RUNNER)!r}, run_name="__main__")
except SystemExit as exc:
    if exc.code != 0:
        raise
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--pcmsolver-library" in result.stdout
    assert "--output-dir" in result.stdout


def test_runner_has_no_model_or_experimental_accuracy_dependency() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import mace" not in source
    assert "FreeSolv" not in source
    assert "MatchedElectrostaticReferenceCase" in source
    assert "PCMSolverSCFSolvent" in source
