from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.function.cosmors_torch.cosmospace import (
    GAS_CONSTANT_J_PER_MOL_K,
    OPEN_COSMORS_24A_PARAMETERS,
    build_segment_interaction_energy,
    molecule_residual_log_activity,
    solve_cosmospace,
)
from maple.function.cosmors_torch.kse import (
    GAS_CONSTANT_KCAL_PER_MOL_K,
    ActivationSolvationFreeEnergy,
    SolvationFreeEnergy,
    compute_relative_kinetic_solvent_effect,
    main,
)

ROOT = Path(__file__).resolve().parents[2]
ORACLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "opencosmors-python-cosmospace-oracle-v1.json"
)


def test_cosmospace_matches_frozen_upstream_python_oracle():
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    fractions = torch.tensor(
        oracle["segment_mole_fractions"],
        dtype=torch.float64,
    )
    interaction = torch.tensor(
        oracle["interaction_energy_j_mol"],
        dtype=torch.float64,
    )

    result = solve_cosmospace(
        fractions,
        interaction,
        temperature_k=oracle["temperature_k"],
    )

    np.testing.assert_allclose(
        result.segment_activity_coefficients.detach().cpu().numpy(),
        oracle["gamma"],
        atol=2.0e-15,
        rtol=0.0,
    )
    assert int(result.iterations) == oracle["iterations"]
    assert bool(result.converged)


def test_cosmospace_batches_and_retains_torch_autograd():
    fractions = torch.tensor(
        [[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]],
        dtype=torch.float64,
    )
    interaction = torch.tensor(
        [
            [0.0, 600.0, -200.0],
            [600.0, 0.0, 350.0],
            [-200.0, 350.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = solve_cosmospace(
        fractions,
        interaction,
        temperature_k=298.15,
    )
    assert result.segment_activity_coefficients.shape == (2, 3)
    assert bool(result.converged.all())
    loss = torch.log(result.segment_activity_coefficients).sum()
    (gradient,) = torch.autograd.grad(loss, interaction)
    assert gradient.shape == (3, 3)
    assert bool(torch.isfinite(gradient).all())


def test_zero_interaction_is_the_ideal_cosmospace_limit():
    fractions = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    interaction = torch.zeros((3, 3), dtype=torch.float64)

    result = solve_cosmospace(
        fractions,
        interaction,
        temperature_k=298.15,
    )

    torch.testing.assert_close(
        result.segment_activity_coefficients,
        torch.ones(3, dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert int(result.iterations) == 1


def test_open24a_interaction_matrix_is_symmetric_and_hbond_attractive():
    sigma = torch.tensor([-0.03, 0.0, 0.03], dtype=torch.float64)
    sigma_orth = torch.zeros(3, dtype=torch.float64)
    donor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    acceptor = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    with_hbond = build_segment_interaction_energy(
        sigma,
        sigma_orth,
        donor,
        acceptor,
        temperature_k=298.15,
    )
    without_hbond = build_segment_interaction_energy(
        sigma,
        sigma_orth,
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        temperature_k=298.15,
    )

    torch.testing.assert_close(with_hbond, with_hbond.T)
    assert float(with_hbond[0, 2]) < float(without_hbond[0, 2])
    assert OPEN_COSMORS_24A_PARAMETERS.name.endswith("neutral-ORCA6")


def test_molecular_residual_activity_uses_segment_counts():
    gamma = torch.tensor([1.1, 0.9, 1.2], dtype=torch.float64)
    counts = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]],
        dtype=torch.float64,
    )

    observed = molecule_residual_log_activity(gamma, counts)
    expected = torch.tensor(
        [2.0 * math.log(1.1) + math.log(1.2), 3.0 * math.log(0.9)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(observed, expected)


def _state(species: str, solvent: str, energy: float) -> SolvationFreeEnergy:
    return SolvationFreeEnergy(
        species=species,
        solvent=solvent,
        delta_g_solvation_kcal_mol=energy,
        temperature_k=298.15,
        standard_state="1M",
        provider_identity="COSMO-RS-test-identity",
    )


def _activation(solvent: str, ts: float, cn: float, methyl_bromide: float):
    return ActivationSolvationFreeEnergy.from_states(
        _state("CN--CH3Br-TS", solvent, ts),
        (
            _state("CN-", solvent, cn),
            _state("CH3Br", solvent, methyl_bromide),
        ),
    )


def test_relative_kse_subtracts_every_reactant_and_uses_tst_sign():
    target = _activation("acetonitrile", -10.0, -5.0, -2.0)
    reference = _activation("water", -8.0, -3.0, -1.0)

    result = compute_relative_kinetic_solvent_effect(target, reference)

    assert target.delta_g_activation_solvation_kcal_mol == pytest.approx(-3.0)
    assert reference.delta_g_activation_solvation_kcal_mol == pytest.approx(-4.0)
    assert result.delta_delta_g_activation_solvation_kcal_mol == pytest.approx(1.0)
    expected_ln = -1.0 / (GAS_CONSTANT_KCAL_PER_MOL_K * 298.15)
    assert result.ln_target_over_reference_rate == pytest.approx(expected_ln)
    assert result.log10_target_over_reference_rate == pytest.approx(
        expected_ln / math.log(10.0)
    )
    assert result.target_over_reference_rate == pytest.approx(math.exp(expected_ln))


def test_kse_rejects_a_changed_reaction_identity():
    target = _activation("acetonitrile", -10.0, -5.0, -2.0)
    reference = ActivationSolvationFreeEnergy.from_states(
        _state("CN--CH3Br-TS", "water", -8.0),
        (_state("CN-", "water", -3.0),),
    )

    with pytest.raises(ValueError, match="same reaction"):
        compute_relative_kinetic_solvent_effect(target, reference)


def test_kse_json_cli_writes_a_hash_bound_result(tmp_path):
    payload = {
        "temperature_k": 298.15,
        "standard_state": "1M",
        "provider_identity": "COSMO-RS-test-identity",
        "target": {
            "solvent": "acetonitrile",
            "transition_state": {
                "species": "CN--CH3Br-TS",
                "delta_g_solvation_kcal_mol": -10.0,
            },
            "reactants": [
                {"species": "CN-", "delta_g_solvation_kcal_mol": -5.0},
                {"species": "CH3Br", "delta_g_solvation_kcal_mol": -2.0},
            ],
        },
        "reference": {
            "solvent": "water",
            "transition_state": {
                "species": "CN--CH3Br-TS",
                "delta_g_solvation_kcal_mol": -8.0,
            },
            "reactants": [
                {"species": "CN-", "delta_g_solvation_kcal_mol": -3.0},
                {"species": "CH3Br", "delta_g_solvation_kcal_mol": -1.0},
            ],
        },
    }
    input_path = tmp_path / "kse-input.json"
    output_path = tmp_path / "kse-output.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main([str(input_path), str(output_path)]) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["delta_delta_g_activation_solvation_kcal_mol"] == pytest.approx(1.0)
    assert len(result["input_sha256"]) == 64
    assert [item["species"] for item in result["reaction"]["reactants"]] == [
        "CN-",
        "CH3Br",
    ]


def test_boltzmann_matrix_uses_si_gas_constant():
    energy = torch.tensor([[0.0, 1000.0], [1000.0, 0.0]], dtype=torch.float64)
    expected = math.exp(-1000.0 / (GAS_CONSTANT_J_PER_MOL_K * 298.15))
    result = solve_cosmospace(
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        energy,
        temperature_k=298.15,
    )
    assert expected < 1.0
    assert bool(torch.isfinite(result.segment_activity_coefficients).all())
