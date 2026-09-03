from __future__ import annotations

from dataclasses import replace
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
from maple.function.cosmors_torch.fixed_structure import (
    _canonicalize_declared_profile_charge,
    evaluate_fixed_structure_payload,
    _validate_fixed_structure_cutoff_margins,
)
from maple.function.cosmors_torch.ionic_es import (
    PARAMETERIZATION_C_NEUTRAL_IDENTITY,
    POLYATOMIC_ANION_SHORT_RANGE_IDENTITY,
    PUBLISHED_PARAMETERIZATION_C_NEUTRAL_COSMOSPACE,
    PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE,
    parameterization_c_neutral_hbond_weights,
    polyatomic_anion_neutral_solvent_cross_energy,
    replace_polyatomic_anion_cross_contacts,
)
from maple.function.cosmors_torch.kse import (
    GAS_CONSTANT_KCAL_PER_MOL_K,
    ActivationSolvationFreeEnergy,
    SolvationFreeEnergy,
    compute_relative_kinetic_solvent_effect,
    main,
)
from maple.function.cosmors_torch.segment_cosmo import (
    TorchSegmentCOSMO,
    TorchSegmentCOSMOConfig,
)
from maple.function.cosmors_torch.surface import (
    SigmaProfile,
    build_sigma_profile,
    discretize_open24a_profile,
    parse_orca_cosmo,
    read_sigma_profile,
    sigma_average,
    write_sigma_profile,
)
from maple.function.cosmors_torch.thermodynamics import (
    PARAMETERIZATION_C_NEUTRAL_COMBINATORIAL_PARAMETERS,
    estimate_open24a_liquid_molar_volume_cm3_mol,
    infinite_dilution_activity,
    open24a_solvation_free_energy,
    staverman_guggenheim_infinite_dilution,
)

ROOT = Path(__file__).resolve().parents[2]
ORACLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "opencosmors-python-cosmospace-oracle-v1.json"
)
ACTIVITY_ORACLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "opencosmors-python-open24a-activity-oracle-v1.json"
)
IONIC_ES_ORACLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "cosmors-es-parameterization-c-polyatomic-anion-v1.json"
)
PARAMETERIZATION_C_NEUTRAL_ORACLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "cosmors-es-parameterization-c-neutral-v1.json"
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


def test_reduced_fixed_structure_cutoff_margin_requires_acknowledgement():
    species = [
        {"name": "reactant"},
        {"name": "transition-state", "cutoff_margin_angstrom": 0.001},
    ]

    with pytest.raises(ValueError, match="fixed-structure-only"):
        _validate_fixed_structure_cutoff_margins(species, acknowledged=False)

    assert _validate_fixed_structure_cutoff_margins(
        species,
        acknowledged=True,
    ) == ["transition-state"]


@pytest.mark.parametrize("value", [0.0, -0.1, math.inf, math.nan])
def test_fixed_structure_cutoff_margin_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _validate_fixed_structure_cutoff_margins(
            [{"name": "species", "cutoff_margin_angstrom": value}],
            acknowledged=True,
        )


def test_ionic_es_requires_explicit_surface_and_domain_acknowledgements(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"identity-only")
    payload = {
        "checkpoint_path": str(checkpoint),
        "ionic_short_range_model": POLYATOMIC_ANION_SHORT_RANGE_IDENTITY,
    }

    with pytest.raises(ValueError, match="surface_mismatch"):
        evaluate_fixed_structure_payload(payload, base_directory=tmp_path)

    payload["acknowledge_ionic_es_surface_mismatch"] = True
    with pytest.raises(ValueError, match="domain_extrapolation"):
        evaluate_fixed_structure_payload(payload, base_directory=tmp_path)

    payload["acknowledge_ionic_es_domain_extrapolation"] = True
    with pytest.raises(ValueError, match="acknowledge_unvalidated_ions"):
        evaluate_fixed_structure_payload(payload, base_directory=tmp_path)


def test_parameterization_c_neutral_requires_surface_acknowledgement(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"identity-only")
    payload = {
        "checkpoint_path": str(checkpoint),
        "neutral_cosmospace_model": PARAMETERIZATION_C_NEUTRAL_IDENTITY,
    }

    with pytest.raises(ValueError, match="surface_mismatch"):
        evaluate_fixed_structure_payload(payload, base_directory=tmp_path)


def test_parameterization_c_polyatomic_anion_source_values_are_frozen():
    parameters = PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE
    oracle = json.loads(IONIC_ES_ORACLE.read_text(encoding="utf-8"))
    for name, expected in oracle["parameters"].items():
        assert getattr(parameters, name) == expected
    provenance = parameters.as_dict()
    assert provenance["source_equations"] == ["6.25", "6.26"]
    assert provenance["pdh_long_range_included"] is False
    assert provenance["single_ion_reference_scale_correction_included"] is False

    neutral = json.loads(PARAMETERIZATION_C_NEUTRAL_ORACLE.read_text(encoding="utf-8"))
    observed_neutral = PUBLISHED_PARAMETERIZATION_C_NEUTRAL_COSMOSPACE
    for name, expected in neutral["cosmospace_parameters"].items():
        assert getattr(observed_neutral, name) == expected
    assert (
        PARAMETERIZATION_C_NEUTRAL_COMBINATORIAL_PARAMETERS.combinatorial_standard_area_angstrom2
        == neutral["combinatorial"]["standard_area_angstrom2"]
    )
    assert (
        PARAMETERIZATION_C_NEUTRAL_COMBINATORIAL_PARAMETERS.combinatorial_volume_exponent
        == neutral["combinatorial"]["volume_exponent"]
    )


def test_parameterization_c_neutral_hbond_switches_follow_parent_elements():
    donor, acceptor = parameterization_c_neutral_hbond_weights(
        torch.tensor([1, 6, 7, 8, 9, 15, 16, 17, 35, 53, 2], dtype=torch.int64)
    )

    torch.testing.assert_close(
        donor,
        torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        acceptor,
        torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.float64),
    )


def test_parameterization_c_modified_sg_uses_published_volume_exponent():
    observed = staverman_guggenheim_infinite_dilution(
        40.0,
        20.0,
        60.0,
        30.0,
        parameters=PARAMETERIZATION_C_NEUTRAL_COMBINATORIAL_PARAMETERS,
    )
    parameters = PARAMETERIZATION_C_NEUTRAL_COMBINATORIAL_PARAMETERS
    phi_prime = 2.0**parameters.combinatorial_volume_exponent
    theta_prime = 2.0
    ratio = phi_prime / theta_prime
    expected = (
        math.log(phi_prime)
        + 1.0
        - phi_prime
        - 0.5
        * parameters.combinatorial_coordination_number
        * (60.0 / parameters.combinatorial_standard_area_angstrom2)
        * (math.log(ratio) + 1.0 - ratio)
    )
    assert float(observed) == pytest.approx(expected, abs=2.0e-15)


@pytest.mark.parametrize("solvent_class", ["water", "organic"])
def test_parameterization_c_cross_energy_matches_published_equation(solvent_class):
    parameters = PUBLISHED_POLYATOMIC_ANION_SHORT_RANGE
    anion_sigma = torch.tensor([0.03], dtype=torch.float64, requires_grad=True)
    anion_orthogonal = torch.tensor([0.002], dtype=torch.float64)
    solvent_sigma = torch.tensor([-0.02], dtype=torch.float64)
    solvent_orthogonal = torch.tensor([-0.001], dtype=torch.float64)

    observed = polyatomic_anion_neutral_solvent_cross_energy(
        anion_sigma,
        anion_orthogonal,
        solvent_sigma,
        solvent_orthogonal,
        solvent_class=solvent_class,
    )
    if solvent_class == "water":
        misfit_coefficient = parameters.water_misfit_j_angstrom2_per_mol_e2
        attraction_coefficient = parameters.water_attraction_j_angstrom2_per_mol_e2
        solvent_strength = min(
            0.0,
            -0.02 + parameters.hydrogen_bond_threshold_e_per_angstrom2,
        )
        anion_strength = max(
            0.0,
            0.03 - parameters.hydrogen_bond_threshold_e_per_angstrom2,
        )
    else:
        misfit_coefficient = parameters.organic_misfit_j_angstrom2_per_mol_e2
        attraction_coefficient = parameters.organic_attraction_j_angstrom2_per_mol_e2
        solvent_strength = min(
            0.0,
            -0.02 + parameters.organic_threshold_e_per_angstrom2,
        )
        anion_strength = max(
            0.0,
            0.03 - parameters.anion_threshold_e_per_angstrom2,
        )
    sigma_sum = 0.01
    orthogonal_sum = 0.001
    expected = (
        0.5
        * parameters.effective_segment_area_angstrom2
        * (
            misfit_coefficient
            * sigma_sum
            * (sigma_sum + parameters.orthogonal_misfit_factor * orthogonal_sum)
            + attraction_coefficient * anion_strength * solvent_strength
        )
    )
    assert float(observed.detach()) == pytest.approx(expected, abs=1.0e-10)
    observed.sum().backward()
    assert anion_sigma.grad is not None
    assert bool(torch.isfinite(anion_sigma.grad).all())


def test_ionic_es_replaces_only_symmetric_cross_contacts():
    base = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    base = 0.5 * (base + base.T)
    observed = replace_polyatomic_anion_cross_contacts(
        base,
        solute_segment_count=2,
        solute_sigma_e_per_angstrom2=torch.tensor([0.03, 0.02], dtype=torch.float64),
        solute_sigma_orthogonal_e_per_angstrom2=torch.zeros(2, dtype=torch.float64),
        solvent_sigma_e_per_angstrom2=torch.tensor([-0.02, -0.01], dtype=torch.float64),
        solvent_sigma_orthogonal_e_per_angstrom2=torch.zeros(2, dtype=torch.float64),
        solvent_class="organic",
    )

    torch.testing.assert_close(observed, observed.T)
    torch.testing.assert_close(observed[:2, :2], base[:2, :2])
    torch.testing.assert_close(observed[2:, 2:], base[2:, 2:])
    assert not bool(torch.equal(observed[:2, 2:], base[:2, 2:]))


def test_infinite_dilution_ionic_es_is_explicit_and_identity_tagged():
    payload = {
        "sigma_e_per_angstrom2": [0.03, 0.02],
        "sigma_orthogonal_e_per_angstrom2": [0.001, -0.001],
        "areas_angstrom2": [20.0, 20.0],
        "atomic_numbers": [6, 7],
        "cavity_volume_angstrom3": 25.0,
    }
    anion = replace(
        _synthetic_profile(payload, name="CN-"),
        molecular_charge_e=torch.tensor(-1.0, dtype=torch.float64),
        molecule_atomic_numbers=(6, 7),
    )
    water = replace(
        _synthetic_profile(
            {
                **payload,
                "sigma_e_per_angstrom2": [-0.03, 0.02],
                "atomic_numbers": [8, 1],
            },
            name="water",
        ),
        molecule_atomic_numbers=(8, 1, 1),
    )

    baseline = infinite_dilution_activity(anion, water, discretize=False)
    experimental = infinite_dilution_activity(
        anion,
        water,
        discretize=False,
        ionic_es_solvent_class="water",
    )

    assert POLYATOMIC_ANION_SHORT_RANGE_IDENTITY in (
        experimental.interaction_model_identity
    )
    assert float(experimental.total_log_activity) != pytest.approx(
        float(baseline.total_log_activity)
    )
    with pytest.raises(ValueError, match="disagrees"):
        infinite_dilution_activity(
            anion,
            water,
            discretize=False,
            ionic_es_solvent_class="organic",
        )


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


def _synthetic_profile(payload, *, name):
    segment_count = len(payload["areas_angstrom2"])
    return SigmaProfile(
        name=name,
        sigma_e_per_angstrom2=torch.tensor(
            payload["sigma_e_per_angstrom2"], dtype=torch.float64
        ),
        sigma_orthogonal_e_per_angstrom2=torch.tensor(
            payload["sigma_orthogonal_e_per_angstrom2"], dtype=torch.float64
        ),
        areas_angstrom2=torch.tensor(payload["areas_angstrom2"], dtype=torch.float64),
        atomic_numbers=torch.tensor(payload["atomic_numbers"], dtype=torch.int64),
        hydrogen_bond_donor_weight=torch.ones(segment_count, dtype=torch.float64),
        hydrogen_bond_acceptor_weight=torch.ones(segment_count, dtype=torch.float64),
        cavity_volume_angstrom3=torch.tensor(
            payload["cavity_volume_angstrom3"], dtype=torch.float64
        ),
        dielectric_energy_hartree=torch.tensor(-0.01, dtype=torch.float64),
        dielectric_energy_role="total-solvated-minus-gas",
        molecular_charge_e=torch.tensor(0.0, dtype=torch.float64),
        source_identity="synthetic-open24a-equation-oracle",
        molecule_atomic_numbers=tuple(payload["atomic_numbers"]),
    )


def test_declared_charge_canonicalization_ignores_only_float_noise():
    profile = replace(
        _synthetic_profile(
            {
                "sigma_e_per_angstrom2": [0.0],
                "sigma_orthogonal_e_per_angstrom2": [0.0],
                "areas_angstrom2": [1.0],
                "atomic_numbers": [6],
                "cavity_volume_angstrom3": 1.0,
            },
            name="neutral",
        ),
        molecular_charge_e=torch.tensor(-3.0e-8, dtype=torch.float64),
    )

    canonical = _canonicalize_declared_profile_charge(profile, declared_charge=0)
    assert float(canonical.molecular_charge_e) == 0.0
    with pytest.raises(RuntimeError, match="disagrees"):
        _canonicalize_declared_profile_charge(profile, declared_charge=-1)


def test_parameterization_c_neutral_baseline_threads_through_activity():
    solute = _synthetic_profile(
        {
            "sigma_e_per_angstrom2": [-0.02, 0.02],
            "sigma_orthogonal_e_per_angstrom2": [0.001, -0.001],
            "areas_angstrom2": [20.0, 20.0],
            "atomic_numbers": [1, 6],
            "cavity_volume_angstrom3": 30.0,
        },
        name="solute",
    )
    solvent = _synthetic_profile(
        {
            "sigma_e_per_angstrom2": [-0.025, 0.02],
            "sigma_orthogonal_e_per_angstrom2": [0.001, -0.001],
            "areas_angstrom2": [22.0, 18.0],
            "atomic_numbers": [1, 8],
            "cavity_volume_angstrom3": 25.0,
        },
        name="solvent",
    )

    baseline = infinite_dilution_activity(solute, solvent, discretize=False)
    parameterization_c = infinite_dilution_activity(
        solute,
        solvent,
        discretize=False,
        cosmospace_parameters=PUBLISHED_PARAMETERIZATION_C_NEUTRAL_COSMOSPACE,
        solvation_parameters=(PARAMETERIZATION_C_NEUTRAL_COMBINATORIAL_PARAMETERS),
    )

    assert (
        parameterization_c.interaction_model_identity
        == PARAMETERIZATION_C_NEUTRAL_IDENTITY
    )
    assert "parameterization-c" in parameterization_c.combinatorial_model_identity
    assert float(parameterization_c.total_log_activity) != pytest.approx(
        float(baseline.total_log_activity)
    )


def test_open24a_activity_matches_frozen_upstream_equation_oracle():
    oracle = json.loads(ACTIVITY_ORACLE.read_text(encoding="utf-8"))
    solute = _synthetic_profile(oracle["solute"], name="solute")
    solvent = _synthetic_profile(oracle["solvent"], name="solvent")

    result = infinite_dilution_activity(
        solute,
        solvent,
        temperature_k=oracle["temperature_k"],
    )

    expected = oracle["expected"]
    assert float(result.residual_log_activity) == pytest.approx(
        expected["residual_log_activity"], abs=3.0e-13
    )
    assert float(result.combinatorial_log_activity) == pytest.approx(
        expected["combinatorial_log_activity"], abs=5.0e-14
    )
    assert float(result.total_log_activity) == pytest.approx(
        expected["total_log_activity"], abs=3.0e-13
    )
    assert int(result.cosmospace.iterations) == expected["cosmospace_iterations"]
    assert (
        result.solute_profile.areas_angstrom2.numel()
        + result.solvent_profile.areas_angstrom2.numel()
        == expected["union_segment_type_count"]
    )


def test_sigma_average_and_open24a_gridding_preserve_autograd_and_area():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    areas = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    raw_sigma = torch.tensor([-0.02, 0.005, 0.018], dtype=torch.float64)
    averaged, _ = sigma_average(
        positions,
        areas,
        raw_sigma,
        averaging_radius_angstrom=0.5,
    )
    averaged.sum().backward()
    assert positions.grad is not None
    assert bool(torch.isfinite(positions.grad).all())

    profile = _synthetic_profile(
        {
            "sigma_e_per_angstrom2": averaged.detach().tolist(),
            "sigma_orthogonal_e_per_angstrom2": [0.0, 0.0015, -0.0025],
            "areas_angstrom2": areas.tolist(),
            "atomic_numbers": [6, 8, 1],
            "cavity_volume_angstrom3": 20.0,
        },
        name="grid-test",
    )
    gridded = discretize_open24a_profile(profile)
    assert float(gridded.cavity_area_angstrom2) == pytest.approx(9.0, abs=1.0e-13)
    assert gridded.discretization == "open24a-bilinear-0.001"


def test_torch_segment_cosmo_is_fixed_dimension_charge_conserving_and_translational():
    model = TorchSegmentCOSMO(
        TorchSegmentCOSMOConfig(
            atomic_numbers=(1,),
            radii_angstrom=(1.3,),
            angular_degree=3,
        )
    )
    geometry = np.zeros((1, 3), dtype=np.float64)
    source = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    surface = model.surface(geometry, source, name="one-sphere")

    assert surface.segment_positions_angstrom.shape == (
        model.config.points_per_atom,
        3,
    )
    assert float(surface.segment_screening_charge_e.sum()) == pytest.approx(
        -1.0, abs=2.0e-14
    )
    assert float(surface.cavity_area_angstrom2) == pytest.approx(
        4.0 * math.pi * 1.3**2, abs=2.0e-13
    )
    assert float(surface.cavity_volume_angstrom3) == pytest.approx(
        4.0 * math.pi * 1.3**3 / 3.0, abs=2.0e-13
    )
    assert float(surface.dielectric_energy_hartree) < 0.0
    np.testing.assert_allclose(
        model.coordinate_gradient(geometry, source),
        np.zeros((1, 3)),
        atol=2.0e-13,
        rtol=0.0,
    )
    translated = geometry + np.array([[2.0, -1.0, 0.5]])
    assert model.energy(translated, source) == pytest.approx(
        model.energy(geometry, source), abs=2.0e-13
    )


def test_torch_segment_cosmo_coordinate_gradient_matches_finite_difference():
    model = TorchSegmentCOSMO(
        TorchSegmentCOSMOConfig(
            atomic_numbers=(6, 1),
            radii_angstrom=(2.0, 1.3),
            angular_degree=2,
        )
    )
    geometry = np.array([[0.0, 0.0, 0.0], [1.35, 0.1, 0.0]], dtype=np.float64)
    source = np.array(
        [[-0.2, 0.01, -0.02, 0.03], [0.2, -0.01, 0.02, -0.03]],
        dtype=np.float64,
    )
    analytic = model.coordinate_gradient(geometry, source)
    step = 2.0e-5
    plus = geometry.copy()
    minus = geometry.copy()
    plus[1, 0] += step
    minus[1, 0] -= step
    numeric = (model.energy(plus, source) - model.energy(minus, source)) / (2 * step)
    assert analytic[1, 0] == pytest.approx(numeric, abs=2.0e-6, rel=2.0e-5)
    np.testing.assert_allclose(analytic.sum(axis=0), np.zeros(3), atol=2.0e-10)


def test_full_open24a_energy_requires_total_electronic_difference_and_flags_ions():
    oracle = json.loads(ACTIVITY_ORACLE.read_text(encoding="utf-8"))
    solute = _synthetic_profile(oracle["solute"], name="solute")
    solvent = _synthetic_profile(oracle["solvent"], name="solvent")
    result = open24a_solvation_free_energy(
        solute,
        solvent,
        solvent_liquid_molar_volume_cm3_mol=75.0,
    )
    ledger = result.as_dict()
    assert math.isfinite(ledger["delta_g_solvation_kcal_mol"])
    assert sum(ledger["ledger_kcal_mol"].values()) == pytest.approx(
        ledger["delta_g_solvation_kcal_mol"], abs=2.0e-12
    )

    boundary_only = SigmaProfile(
        **{
            field: getattr(solute, field)
            for field in solute.__dataclass_fields__
            if field != "dielectric_energy_role"
        },
        dielectric_energy_role="boundary-polarization-only",
    )
    with pytest.raises(ValueError, match="total solvated-minus-gas"):
        open24a_solvation_free_energy(boundary_only, solvent)

    ionic = SigmaProfile(
        **{
            field: getattr(solute, field)
            for field in solute.__dataclass_fields__
            if field != "molecular_charge_e"
        },
        molecular_charge_e=torch.tensor(-1.0, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="neutral molecules"):
        open24a_solvation_free_energy(ionic, solvent)
    diagnostic = open24a_solvation_free_energy(
        ionic,
        solvent,
        acknowledge_unvalidated_ions=True,
        solvent_liquid_molar_volume_cm3_mol=75.0,
    )
    assert diagnostic.diagnostic_only


def test_orca_cosmo_reader_is_interoperability_only(tmp_path):
    path = tmp_path / "tiny.orcacosmo"
    path.write_text(
        """tiny : DFT_CPCM_TEST

##################################################
#ENERGY
FINAL SINGLE POINT ENERGY       -1.100000000000

##################################################
#XYZ_FILE
1
energy: 0.0
H 0.0 0.0 0.0

##################################################
#COSMO
1 # Number of atoms
2 # Number of surface points
8.0 # Volume
12.0 # Area
-0.010000000 # CPCM dielectric energy

#------------------------------------------------------------
# CARTESIAN COORDINATES (A.U.) + RADII (A.U.) + ATOMIC NUMBER
#------------------------------------------------------------
0.0 0.0 0.0 2.0 1

#------------------------------------------------------------
# SURFACE POINTS (A.U.)    (Hint - charge NOT scaled by FEps)
#------------------------------------------------------------
X Y Z area potential charge w_leb Switch_F G_width atom
2.0 0.0 0.0 6.0 -0.1 0.1 1.0 1.0 1.0 0
-2.0 0.0 0.0 6.0 0.1 -0.1 1.0 1.0 1.0 0

##################################################
#COSMO_corrected
Corrected dielectric energy   =     -0.020000000
Total C-PCM charge            =      0.000000000
C-PCM corrected charges:
0.2
-0.2
##################################################
""",
        encoding="utf-8",
    )
    boundary = parse_orca_cosmo(path)
    assert boundary.dielectric_energy_role == "boundary-polarization-only"
    assert float(boundary.dielectric_energy_hartree) == pytest.approx(-0.02)
    complete = parse_orca_cosmo(path, gas_energy_hartree=-1.0)
    assert complete.dielectric_energy_role == "total-solvated-minus-gas"
    # corrected total = -1.1 - (-0.01) + (-0.02) = -1.11
    assert float(complete.dielectric_energy_hartree) == pytest.approx(-0.11)
    profile = build_sigma_profile(boundary)
    assert profile.molecule_atomic_numbers == (1,)
    assert float(profile.cavity_area_angstrom2) == pytest.approx(
        12.0 * 0.52917721092**2
    )


def test_open24a_liquid_volume_uses_exact_water_special_case():
    payload = {
        "sigma_e_per_angstrom2": [-0.02, 0.01],
        "sigma_orthogonal_e_per_angstrom2": [0.0, 0.0],
        "areas_angstrom2": [20.0, 20.0],
        "atomic_numbers": [8, 1],
        "cavity_volume_angstrom3": 25.0,
    }
    water = _synthetic_profile(payload, name="water")
    water = SigmaProfile(
        **{
            field: getattr(water, field)
            for field in water.__dataclass_fields__
            if field != "molecule_atomic_numbers"
        },
        molecule_atomic_numbers=(8, 1, 1),
    )
    assert float(estimate_open24a_liquid_molar_volume_cm3_mol(water)) == pytest.approx(
        18.06863632, abs=0.0
    )


def test_native_sigma_profile_json_round_trip_is_exact(tmp_path):
    oracle = json.loads(ACTIVITY_ORACLE.read_text(encoding="utf-8"))
    profile = _synthetic_profile(oracle["solute"], name="round-trip")
    path = tmp_path / "profile.json"

    write_sigma_profile(profile, path)
    restored = read_sigma_profile(path)

    assert restored.name == profile.name
    assert restored.source_identity == profile.source_identity
    assert restored.molecule_atomic_numbers == profile.molecule_atomic_numbers
    for field in (
        "sigma_e_per_angstrom2",
        "sigma_orthogonal_e_per_angstrom2",
        "areas_angstrom2",
        "atomic_numbers",
        "hydrogen_bond_donor_weight",
        "hydrogen_bond_acceptor_weight",
    ):
        torch.testing.assert_close(
            getattr(restored, field),
            getattr(profile, field),
            atol=0.0,
            rtol=0.0,
        )
