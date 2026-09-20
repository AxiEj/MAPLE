from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tools.route2_release import run_rhodrop_qm_reconstructed_acetone as runner


ROOT = Path(__file__).parents[2]
PREREGISTRATION = (
    ROOT
    / "docs/route2/preregistrations/"
    "rhodrop-qm-reconstructed-acetone-diagnostic-v1.json"
)
EXECUTION_GIT_HEAD = "a53318176ac19e91c2c43ae48fd6865768741e31"
COMMITTED_EXECUTION_SOURCES = {
    "maple/function/calculator/extra_correction/implicit/route2_rhodrop_cpcm.py"
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_source_sha256(revision: str, relative_path: str) -> str:
    source = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(source).hexdigest()


def test_rhodrop_qm_reconstructed_preregistration_is_source_bound_and_label_free() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen-before-execution"
    assert payload["system"]["compound_id"] == runner.EXPECTED_COMPOUND_ID
    assert payload["scientific_identity"]["n_iso_e_per_bohr3"] == (
        runner.N_ISO_E_PER_BOHR3
    )
    assert payload["no_target_policy"]["experimental_solvation_labels_read"] is False
    assert payload["frozen_inputs"]["qm_checkpoint_sha256"] == (
        runner.EXPECTED_QM_CHECKPOINT_SHA256
    )
    assert payload["frozen_inputs"]["mace_exact_gto_state_archive_sha256"] == (
        runner.EXPECTED_MACE_STATE_SHA256
    )
    assert payload["frozen_inputs"]["literature_pdf_sha256"] == (
        runner.EXPECTED_PDF_SHA256
    )
    for relative, expected in payload["source_sha256"].items():
        actual = (
            _git_source_sha256(EXECUTION_GIT_HEAD, relative)
            if relative in COMMITTED_EXECUTION_SOURCES
            else _sha256(ROOT / relative)
        )
        assert actual == expected


def test_rhodrop_qm_reconstructed_preregistration_freezes_the_two_by_two_question() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    sources = payload["scientific_identity"]["electrostatic_sources"]
    assert sources == [
        "QM nuclei-minus-AO-density total MEP",
        "official MACE-POLAR-1-M gas point-l<=1 MEP",
    ]
    assert payload["scientific_identity"]["continuum"] == "MOIST rho-DROP C-PCM"
    assert payload["scientific_identity"]["cds"] == "none"
    assert "source-cavity interaction" in payload["reported_metrics"]
    assert "new preregistration" in payload["decision_rule"]
