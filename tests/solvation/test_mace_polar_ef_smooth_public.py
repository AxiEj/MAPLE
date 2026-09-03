from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.set_calculator import SetCalculator
from maple.function.dispatcher.sp.sp import SinglePoint
from maple.function.read.command_control import CommandControl
from maple.function.route2_model_contracts import (
    ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
)
from maple.function.route2_smd_profiles import (
    MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE,
    route2_smd_profile_spec,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "macepol-ef-v2.pt"


def _atoms() -> Atoms:
    atoms = Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.239987, 0.927297, 0.0],
        ],
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


def _options(*, acknowledge: bool = True) -> dict[str, object]:
    return {
        "implicit": "water",
        "method": "smd",
        "provider": "torch-smooth-pcm",
        "profile": MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE,
        "response": "scf",
        "standard_state": "1m",
        "experimental": True,
        "acknowledge_known_nonpassive": acknowledge,
    }


def _settings(*, task: str = "#sp", acknowledge: bool = True) -> tuple[str, ...]:
    return (
        f"#model=mace-polar-ef-v2(model_path={CHECKPOINT})",
        task,
        (
            "#solv(implicit=water,method=smd,provider=torch-smooth-pcm,"
            f"profile={MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE},"
            "response=scf,standard_state=1m,experimental=true,"
            f"acknowledge_known_nonpassive={'true' if acknowledge else 'false'})"
        ),
    )


def _parse(*settings: str) -> dict[str, object]:
    return CommandControl.from_settings(list(settings)).as_dict()


def test_profile_is_explicit_known_nonpassive_energy_only_identity():
    spec = route2_smd_profile_spec(MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE)

    assert spec.provider == "torch-smooth-pcm"
    assert spec.electronic_model_family == ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY
    assert spec.electrostatics_model == "smooth-ddpcm"
    assert spec.reaction_field_projector == ("native-atomwise-potential-gradient-v1")
    assert spec.known_nonpassive_diagnostic is True
    assert spec.default_eligible is False
    assert spec.force_release_eligible is False


def test_parser_accepts_only_explicit_acknowledged_single_point_profile():
    parsed = _parse(*_settings())

    assert parsed["model"] == "mace-polar-ef-v2"
    assert parsed["model_options"] == {"model_path": str(CHECKPOINT)}
    assert parsed["solv"] == _options()


@pytest.mark.parametrize(
    "settings, message",
    (
        (_settings(acknowledge=False), "acknowledge_known_nonpassive=true"),
        (_settings(task="#opt"), "Unknown OPT|single-point only"),
        (_settings(task="#sp(verbose=1)"), "energy-only"),
    ),
)
def test_parser_rejects_unacknowledged_or_broader_use(settings, message):
    with pytest.raises(ValueError, match=message):
        _parse(*settings)


def test_parser_rejects_profile_model_identity_exchange():
    settings = list(_settings())
    settings[0] = "#model=macepol-m"

    with pytest.raises(ValueError, match="MACE-POLAR-EF-v2"):
        _parse(*settings)


def test_factory_validation_accepts_the_exact_narrow_profile(tmp_path):
    builder = SetCalculator(
        device="cuda:0",
        model="mace-polar-ef-v2",
        output=str(tmp_path / "job.out"),
        atoms=_atoms(),
        implicit="smd",
        solvent="water",
        model_options={"model_path": str(CHECKPOINT)},
        solvation_options=_options(),
    )

    builder._validate_solvent_config()


@pytest.fixture(scope="module")
def public_energy_result(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyscf")
    if not CHECKPOINT.is_file() or not torch.cuda.is_available():
        pytest.skip("local MACE-POLAR-EF v2 CUDA checkpoint is unavailable")

    directory = tmp_path_factory.mktemp("mace-ef-smooth-public")
    output = directory / "job.out"
    atoms = _atoms()
    calculator = SetCalculator(
        device=torch.device("cuda:0"),
        model="mace-polar-ef-v2",
        output=str(output),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        model_options={"model_path": str(CHECKPOINT)},
        solvation_options=_options(),
    ).set_calculator()
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    return atoms, calculator, energy, output


def test_public_known_nonpassive_diagnostic_energy_runs(public_energy_result):
    atoms, calculator, energy, _output = public_energy_result

    assert np.isfinite(energy)
    assert "forces" not in calculator.results
    solvation = calculator.results["solvation"]
    assert solvation["combined_energy_hartree"] == pytest.approx(energy)
    provenance = solvation["provenance"]
    assert provenance["known_nonpassive_diagnostic"] is True
    assert provenance["known_nonpassive_acknowledged"] is True
    assert provenance["scientifically_valid"] is False
    assert provenance["accuracy_certified"] is False
    assert provenance["solution_phase_pes"] is False
    assert provenance["electronic_passivity"]["passivity_passed"] is False
    output_lines = SinglePoint._solvation_lines(atoms)
    assert any("KNOWN-NONPASSIVE DIAGNOSTIC" in line for line in output_lines)
    assert any(
        "must not be used as a physical prediction" in line for line in output_lines
    )

    with pytest.raises(NotImplementedError):
        atoms.get_forces()


def test_public_result_ledger_retains_known_invalid_status(public_energy_result):
    _atoms_value, _calculator, _energy, output = public_energy_result
    audit_dir = output.with_suffix(output.suffix + ".implicit")
    ledger = json.loads(
        (audit_dir / "route2-public-result-ledger.json").read_text(encoding="utf-8")
    )
    provider = json.loads(
        (audit_dir / "mace-polar-ef-smooth-pcm-result.json").read_text(encoding="utf-8")
    )

    assert ledger["forces_evaluated"] is False
    assert ledger["profile"] == MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE
    assert provider["known_nonpassive_diagnostic"] is True
    assert provider["electronic_passivity"]["passivity_passed"] is False
