from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examp"
DIRECT_PROFILE = "smd-ddpcm-l15-n1202-multisolv-pcm-half-coupling-v2"


@pytest.mark.parametrize(
    ("prefix", "response_mode", "field_conditioned", "scf_applies"),
    (
        ("01_direct_pcm_frozen_ddpcm", "frozen", False, False),
        ("02_direct_pcm_scf_ddpcm", "scf", True, True),
    ),
)
def test_route2_examples_bind_response_mode_to_a_real_audit(
    prefix: str,
    response_mode: str,
    field_conditioned: bool,
    scf_applies: bool,
):
    input_text = (EXAMPLE_DIR / f"{prefix}.inp").read_text(encoding="utf-8")
    output_text = (EXAMPLE_DIR / f"{prefix}.out").read_text(encoding="utf-8")
    audit_dir = EXAMPLE_DIR / f"{prefix}.out.implicit"
    audit = json.loads(
        (audit_dir / "route2-ddpcm-result.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (audit_dir / "route2-public-result-ledger.json").read_text(encoding="utf-8")
    )

    assert f"profile={DIRECT_PROFILE}" in input_text
    assert f"response={response_mode}" in input_text
    assert audit["profile"] == DIRECT_PROFILE
    assert audit["response_mode"] == response_mode
    assert audit["field_conditioned_model_state_evaluated"] is (field_conditioned)
    assert audit["electrostatic_energy_ledger"] == ("pcm-half-coupling-only-v1")
    assert audit["scf"]["applies"] is scf_applies
    assert audit["forces_evaluated"] is False
    assert ledger["leaf_components_hartree"]["solute_polarization"] == 0.0
    assert ledger["derived_totals_hartree"]["electrostatic"] == pytest.approx(
        ledger["leaf_components_hartree"]["pcm_polarization"],
        abs=1.0e-15,
    )
    assert "Combined E_MLIP(gas)+Delta G_solv" in output_text


def test_route2_example_directory_has_two_current_inputs_and_archived_history():
    assert {path.name for path in EXAMPLE_DIR.glob("*.inp")} == {
        "01_direct_pcm_frozen_ddpcm.inp",
        "02_direct_pcm_scf_ddpcm.inp",
    }
    assert (EXAMPLE_DIR / "_historical_legacy_ledger/01_legacy_scf_ddpcm.inp").is_file()
    assert (EXAMPLE_DIR / "_historical_exact_gto/02_legacy_scf_exact_gto.inp").is_file()
