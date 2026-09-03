from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import (
    MACE_POLAR_EF_SMOOTH_COSMO_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE,
    route2_smd_profile_spec,
)
from maple.function.route2_solvents import SUPPORTED_ROUTE2_SMD_SOLVENTS

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "macepol-ef-v2.pt"
PROFILE = MACE_POLAR_EF_SMOOTH_COSMO_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE


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


def _options(solvent: str = "acetonitrile") -> dict[str, object]:
    return {
        "implicit": solvent,
        "method": "cosmo",
        "provider": "torch-smooth-cosmo",
        "profile": PROFILE,
        "response": "scf",
        "standard_state": "1m",
        "experimental": True,
        "acknowledge_known_nonpassive": True,
        "acknowledge_unvalidated_derivatives": True,
    }


def _settings(task: str) -> list[str]:
    return [
        f"#model=mace-polar-ef-v2(model_path={CHECKPOINT})",
        task,
        "#device=gpu0",
        (
            "#solv(implicit=acetonitrile,method=cosmo,"
            f"provider=torch-smooth-cosmo,profile={PROFILE},response=scf,"
            "standard_state=1m,experimental=true,"
            "acknowledge_known_nonpassive=true,"
            "acknowledge_unvalidated_derivatives=true)"
        ),
    ]


def test_cosmo_profile_is_separate_and_multisolvent():
    spec = route2_smd_profile_spec(PROFILE)

    assert spec.provider == "torch-smooth-cosmo"
    assert spec.electrostatics_model == "smooth-cosmo"
    assert spec.dielectric_policy == "conductor-infinity-binary64-v1"
    assert spec.supported_solvents == SUPPORTED_ROUTE2_SMD_SOLVENTS
    assert spec.diagnostic_derivative_eligible is True


@pytest.mark.parametrize(
    "task",
    (
        "#sp(verbose=1)",
        "#opt(method=lbfgs,max_iter=1)",
        "#freq",
        "#ts(method=prfo,max_iter=1)",
    ),
)
def test_cosmo_profile_opens_the_derivative_task_surface(task):
    parsed = CommandControl.from_settings(_settings(task))

    assert parsed.params["solv"] == _options()


@pytest.fixture(scope="module")
def real_cosmo_result(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyscf")
    if not CHECKPOINT.is_file() or not torch.cuda.is_available():
        pytest.skip("local MACE-POLAR-EF v2 CUDA checkpoint is unavailable")

    output = tmp_path_factory.mktemp("mace-ef-cosmo") / "job.out"
    atoms = _atoms()
    calculator = SetCalculator(
        device=torch.device("cuda:0"),
        model="mace-polar-ef-v2",
        output=str(output),
        atoms=atoms,
        implicit="cosmo",
        solvent="acetonitrile",
        model_options={"model_path": str(CHECKPOINT)},
        solvation_options=_options(),
    ).set_calculator()
    atoms.calc = calculator
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return atoms, calculator, forces, output


def test_real_mace_ef_cosmo_energy_and_force_run(real_cosmo_result):
    _atoms_value, calculator, forces, _output = real_cosmo_result

    assert forces.shape == (3, 3)
    assert np.all(np.isfinite(forces))
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=2.0e-7, rtol=0.0)
    solvation = calculator.results["solvation"]
    provenance = solvation["provenance"]
    assert provenance["provider"] == "torch-smooth-cosmo"
    assert provenance["electrostatics_model"] == "smooth-cosmo"
    assert provenance["continuum"]["continuum_model"] == "COSMO"
    assert provenance["continuum"]["conductor_limit"] is True


def test_real_cosmo_result_ledger_keeps_diagnostic_force_scope(real_cosmo_result):
    _atoms_value, _calculator, _forces, output = real_cosmo_result
    ledger = json.loads(
        (
            output.with_suffix(output.suffix + ".implicit")
            / "route2-public-result-ledger.json"
        ).read_text(encoding="utf-8")
    )

    assert ledger["provider"] == "torch-smooth-cosmo"
    assert ledger["forces_evaluated"] is True
    assert ledger["force_admission"] is None
    assert ledger["diagnostic_derivative_evidence"]["release_admitted"] is False
