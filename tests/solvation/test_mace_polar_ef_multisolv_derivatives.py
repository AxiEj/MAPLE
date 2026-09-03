from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from maple.function.calculator.extra_correction.implicit.mace_polar_ef_specs import (
    MACE_POLAR_EF_V2_CHECKPOINT_SPEC,
    mace_polar_ef_checkpoint_spec,
    register_mace_polar_ef_checkpoint_spec,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.route2_model_contracts import (
    ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
    register_route2_input_model_family,
)
from maple.function.route2_smd_profiles import (
    MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE,
    MACE_POLAR_EF_SMOOTH_PCM_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE,
    register_route2_smd_profile,
    route2_smd_profile_spec,
)
from maple.function.route2_solvents import SUPPORTED_ROUTE2_SMD_SOLVENTS

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "macepol-ef-v2.pt"
PROFILE = MACE_POLAR_EF_SMOOTH_PCM_MULTISOLVENT_DERIVATIVE_DIAGNOSTIC_PROFILE


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


def _options(solvent: str = "methanol") -> dict[str, object]:
    return {
        "implicit": solvent,
        "method": "smd",
        "provider": "torch-smooth-pcm",
        "profile": PROFILE,
        "response": "scf",
        "standard_state": "1m",
        "experimental": True,
        "acknowledge_known_nonpassive": True,
        "acknowledge_unvalidated_derivatives": True,
    }


def _settings(task: str, *, solvent: str = "methanol") -> list[str]:
    return [
        f"#model=mace-polar-ef-v2(model_path={CHECKPOINT})",
        task,
        "#device=gpu0",
        (
            f"#solv(implicit={solvent},method=smd,"
            f"provider=torch-smooth-pcm,profile={PROFILE},response=scf,"
            "standard_state=1m,experimental=true,"
            "acknowledge_known_nonpassive=true,"
            "acknowledge_unvalidated_derivatives=true)"
        ),
    ]


def test_multisolvent_derivative_profile_has_explicit_capabilities():
    spec = route2_smd_profile_spec(PROFILE)

    assert spec.supported_solvents == SUPPORTED_ROUTE2_SMD_SOLVENTS
    assert len(spec.supported_solvents) == 11
    assert spec.diagnostic_derivative_eligible is True
    assert spec.known_nonpassive_diagnostic is True
    assert spec.force_release_eligible is False

    energy_only = route2_smd_profile_spec(MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE)
    assert energy_only.supported_solvents == frozenset({"water"})
    assert energy_only.diagnostic_derivative_eligible is False


@pytest.mark.parametrize(
    "task, expected_task",
    (
        ("#sp(verbose=1)", "sp"),
        ("#opt(method=lbfgs,max_iter=1)", "opt"),
        ("#freq", "freq"),
        ("#ts(method=prfo,max_iter=1)", "ts"),
    ),
)
def test_parser_opens_declared_derivative_tasks(task, expected_task):
    parsed = CommandControl.from_settings(_settings(task))

    assert parsed.task == expected_task
    assert parsed.params["solv"] == _options()


def test_parser_rejects_missing_derivative_acknowledgement():
    settings = _settings("#freq")
    settings[-1] = settings[-1].replace(
        ",acknowledge_unvalidated_derivatives=true",
        "",
    )

    with pytest.raises(
        ValueError,
        match="acknowledge_unvalidated_derivatives=true",
    ):
        CommandControl.from_settings(settings)


@pytest.mark.parametrize(
    "task, message",
    (
        ("#opt(method=rfo)", "only with method=lbfgs"),
        ("#ts(method=dimer)", "only with method=prfo"),
        ("#md", "supports only SP, OPT, FREQ, and TS"),
    ),
)
def test_parser_rejects_unopened_derivative_workflows(task, message):
    with pytest.raises(ValueError, match=message):
        CommandControl.from_settings(_settings(task))


def test_checkpoint_spec_registry_is_idempotent_and_no_replace():
    assert (
        register_mace_polar_ef_checkpoint_spec(MACE_POLAR_EF_V2_CHECKPOINT_SPEC)
        is MACE_POLAR_EF_V2_CHECKPOINT_SPEC
    )
    assert (
        mace_polar_ef_checkpoint_spec("MACE-POLAR-EF-V2")
        == MACE_POLAR_EF_V2_CHECKPOINT_SPEC
    )
    with pytest.raises(ValueError, match="already registered"):
        register_mace_polar_ef_checkpoint_spec(
            replace(
                MACE_POLAR_EF_V2_CHECKPOINT_SPEC,
                checkpoint_size=(MACE_POLAR_EF_V2_CHECKPOINT_SPEC.checkpoint_size + 1),
            )
        )


def test_model_and_profile_extension_registries_are_no_replace():
    register_route2_input_model_family(
        "mace-polar-ef-v2",
        ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
        label="MACE-POLAR-EF-v2",
        explicit_model_path=True,
    )
    with pytest.raises(ValueError, match="already registered"):
        register_route2_input_model_family(
            "mace-polar-ef-v2",
            "different-family",
        )

    spec = route2_smd_profile_spec(PROFILE)
    assert register_route2_smd_profile(spec) is spec
    with pytest.raises(ValueError, match="already registered"):
        register_route2_smd_profile(
            replace(spec, mace_long_range_evaluator="different-evaluator")
        )


@pytest.fixture(scope="module")
def methanol_force_result(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyscf")
    if not CHECKPOINT.is_file() or not torch.cuda.is_available():
        pytest.skip("local MACE-POLAR-EF v2 CUDA checkpoint is unavailable")

    directory = tmp_path_factory.mktemp("mace-ef-multisolv-force")
    output = directory / "job.out"
    atoms = _atoms()
    calculator = SetCalculator(
        device=torch.device("cuda:0"),
        model="mace-polar-ef-v2",
        output=str(output),
        atoms=atoms,
        implicit="smd",
        solvent="methanol",
        model_options={"model_path": str(CHECKPOINT)},
        solvation_options=_options(),
    ).set_calculator()
    atoms.calc = calculator
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return atoms, calculator, forces, output


def test_real_multisolvent_analytic_force_is_exposed(methanol_force_result):
    atoms, calculator, forces, _output = methanol_force_result

    assert forces.shape == (3, 3)
    assert np.all(np.isfinite(forces))
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=2.0e-7, rtol=0.0)
    provenance = calculator.results["solvation"]["provenance"]
    assert provenance["forces_available"] is True
    assert provenance["numerical_hessian_available"] is True
    assert provenance["scientifically_valid"] is False

    atoms.set_constraint(FixAtoms(indices=range(len(atoms))))
    hessian = calculator.get_hessian(atoms)
    np.testing.assert_array_equal(hessian, np.zeros((9, 9)))


def test_diagnostic_force_ledger_is_separate_from_admission(
    methanol_force_result,
):
    _atoms_value, _calculator, _forces, output = methanol_force_result
    ledger = json.loads(
        (
            output.with_suffix(output.suffix + ".implicit")
            / "route2-public-result-ledger.json"
        ).read_text(encoding="utf-8")
    )

    assert ledger["forces_evaluated"] is True
    assert ledger["force_admission"] is None
    assert ledger["diagnostic_derivative_evidence"]["scope"] == (
        "known-nonpassive-diagnostic-derivative-v1"
    )
    assert ledger["diagnostic_derivative_evidence"]["release_admitted"] is False
