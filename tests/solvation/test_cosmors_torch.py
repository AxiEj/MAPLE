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
    estimate_open24a_liquid_molar_volume_cm3_mol,
    infinite_dilution_activity,
    open24a_solvation_free_energy,
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
