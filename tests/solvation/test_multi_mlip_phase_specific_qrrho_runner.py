from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator._batch_types import BatchResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
RUNNER_PATH = BENCHMARK_DIR / "run_multi_mlip_phase_specific_qrrho.py"
PROTOCOL_PATH = BENCHMARK_DIR / "multi_mlip_phase_specific_qrrho_protocol.json"

spec = importlib.util.spec_from_file_location("route1_qrrho_runner", RUNNER_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_runner_validates_the_complete_label_free_freeze():
    protocol, fingerprint, manifest = runner.load_qrrho_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert protocol["protocol_id"].endswith("-v8")
    assert protocol["solvation"]["charge_source"] == "frozen-am1bcc-json"
    assert protocol["execution"]["energy_runner_never_reads_experimental_labels"]
    assert (
        manifest["label_boundary"]["prepared_state_manifest_contains_labels"] is False
    )


def test_cost_bounded_source_selection_is_label_blind_and_deterministic():
    protocol = runner.load_json(PROTOCOL_PATH)
    manifest = runner.load_json(
        BENCHMARK_DIR / "multi_mlip_discrete_conformer_source_manifest.json"
    )

    selected = runner.select_protocol_source_cases(
        manifest["cases"],
        target_state_counts=protocol["source"]["selection_target_state_counts"],
        maximum_atom_count=protocol["source"]["maximum_atom_count"],
    )

    assert [case["compound_id"] for case in selected] == [
        "mobley_1952272",
        "mobley_1717215",
        "mobley_8118832",
        "mobley_4463913",
        "mobley_5759258",
        "mobley_1017962",
    ]
    assert sum(case["state_count"] for case in selected) == 221


def test_phase_seed_selection_is_energy_ranked_diverse_and_deterministic():
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    translated_duplicate = base + np.asarray([4.0, -2.0, 1.0])
    distinct = base.copy()
    distinct[2, 1] = 2.0
    positions = np.stack([translated_duplicate, base, distinct])

    selected = runner.select_diverse_seeds(
        positions_angstrom=positions,
        potential_hartree=np.asarray([-2.0, -1.0, -3.0]),
        symbols=["C", "C", "C"],
        maximum_count=3,
        threshold_angstrom=0.125,
    )

    assert [row["source_state_index"] for row in selected] == [2, 0]
    assert selected[0]["minimum_rmsd_to_prior_seed_angstrom"] is None
    assert selected[1]["minimum_rmsd_to_prior_seed_angstrom"] > 0.125


def test_optimized_minimum_deduplication_never_uses_arrival_count_as_weight():
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    distinct = base.copy()
    distinct[2, 1] = 2.0

    def branch(index, energy, positions):
        return {
            "status": "success",
            "converged": True,
            "source_state_index": index,
            "final_energy": {"total_energy_hartree": energy},
            "final_positions_angstrom": positions.tolist(),
        }

    accepted, rejected = runner.deduplicate_optimized_minima(
        branches=[
            branch(2, -2.0, distinct),
            branch(1, -3.0, base + 5.0),
            branch(0, -4.0, base),
        ],
        symbols=["C", "C", "C"],
        threshold_angstrom=0.125,
    )

    assert [row["source_state_index"] for row in accepted] == [0, 2]
    assert rejected == [
        {
            "source_state_index": 1,
            "duplicate_of_source_state_index": 0,
            "aligned_heavy_atom_rmsd_angstrom": pytest.approx(0.0, abs=1.0e-12),
        }
    ]


def test_selected_negative_modes_and_rigid_mode_leakage_fail_closed():
    frequencies = np.asarray([-250.0, -0.5, 0.0, 0.2, 0.5, 0.8, 100.0, 200.0, 300.0])

    nonlinear = runner.select_vibrational_modes(
        frequencies,
        atom_count=3,
        linear=False,
        imaginary_frequency_cutoff_cm1=0.0,
        maximum_rigid_mode_leakage_cm1=1.0,
    )
    assert nonlinear["expected_vibrational_mode_count"] == 3
    assert nonlinear["selected_frequencies_cm1"] == [-250.0, 200.0, 300.0]
    assert nonlinear["thermochemistry_frequencies_cm1"] == [
        -250.0,
        200.0,
        300.0,
    ]
    assert nonlinear["negative_selected_frequencies_cm1"] == [-250.0]
    assert nonlinear["maximum_rigid_mode_leakage_cm1"] == pytest.approx(100.0)
    assert nonlinear["rigid_mode_leakage_gate_passed"] is False
    assert nonlinear["valid_stationary_point"] is False

    stable = np.asarray([-0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 100.0, 200.0, 300.0])
    linear = runner.select_vibrational_modes(
        stable,
        atom_count=3,
        linear=True,
        imaginary_frequency_cutoff_cm1=0.0,
        maximum_rigid_mode_leakage_cm1=0.75,
    )
    assert linear["expected_vibrational_mode_count"] == 4
    assert linear["selected_frequencies_cm1"] == [1.0, 100.0, 200.0, 300.0]
    assert linear["negative_selected_frequencies_cm1"] == []
    assert linear["maximum_rigid_mode_leakage_cm1"] == pytest.approx(0.5)
    assert linear["valid_stationary_point"] is True

    leaked = runner.select_vibrational_modes(
        stable,
        atom_count=3,
        linear=True,
        imaginary_frequency_cutoff_cm1=0.0,
        maximum_rigid_mode_leakage_cm1=0.25,
    )
    assert leaked["valid_stationary_point"] is False
    assert leaked["rigid_mode_leakage_gate_passed"] is False


def test_unit_weight_ensemble_uses_stable_log_sum_exp():
    result = runner.stable_ensemble_free_energy(
        [0.0, 0.0],
        temperature_kelvin=298.15,
    )
    expected = -runner.R_KCAL_MOL_K * 298.15 * math.log(2.0)

    assert result["free_energy_kcal_mol"] == pytest.approx(expected)
    assert result["normalized_weights"] == pytest.approx([0.5, 0.5])
    assert result["effective_minimum_count"] == pytest.approx(2.0)
    assert result["maximum_normalized_weight"] == pytest.approx(0.5)


def test_selected_branch_accounting_rejects_failure_or_nonconvergence():
    success = {"status": "success", "converged": True, "source_state_index": 0}
    assert runner.validate_selected_branch_accounting(
        [success],
        duplicate_minima=[],
        unique_minimum_count=1,
    ) == {
        "selected_seed_count": 1,
        "valid_unique_minimum_count": 1,
        "explicit_duplicate_count": 0,
        "all_selected_seeds_resolved": True,
    }

    with pytest.raises(ValueError, match="selected optimization branch"):
        runner.validate_selected_branch_accounting(
            [
                success,
                {
                    "status": "failure",
                    "converged": False,
                    "source_state_index": 1,
                },
            ],
            duplicate_minima=[],
            unique_minimum_count=1,
        )
    with pytest.raises(ValueError, match="selected optimization branch"):
        runner.validate_selected_branch_accounting(
            [
                success,
                {
                    "status": "success",
                    "converged": False,
                    "source_state_index": 1,
                },
            ],
            duplicate_minima=[],
            unique_minimum_count=1,
        )


def test_hessian_displacement_preflight_covers_rigid_and_flexible_cases():
    protocol = runner.load_json(PROTOCOL_PATH)

    for compound_id in ("mobley_1952272", "mobley_4463913"):
        assert runner._requires_hessian_displacement_sensitivity(
            protocol,
            compound_id=compound_id,
            model_name="maceoff23m",
            phase="solution",
            minimum_index=0,
        )
    assert not runner._requires_hessian_displacement_sensitivity(
        protocol,
        compound_id="mobley_1017962",
        model_name="maceoff23m",
        phase="solution",
        minimum_index=0,
    )


def test_hessian_numerical_quality_combines_absolute_and_scale_aware_gates():
    protocol = runner.load_json(PROTOCOL_PATH)
    specification = protocol["hessian_and_stationary_point"]
    diagnostics = {
        "maximum_absolute_raw_hessian_hartree_per_angstrom2": 3.0,
        "maximum_raw_asymmetry_hartree_per_angstrom2": 1.0e-3,
        "raw_hessian_frobenius_norm_hartree_per_angstrom2": 10.0,
        "raw_asymmetry_frobenius_norm_hartree_per_angstrom2": 0.01,
        "relative_raw_asymmetry_frobenius": 1.0e-3,
    }

    runner.validate_hessian_numerical_quality(
        diagnostics,
        specification=specification,
    )

    with pytest.raises(ValueError, match="absolute sanity ceiling"):
        runner.validate_hessian_numerical_quality(
            {
                **diagnostics,
                "maximum_raw_asymmetry_hartree_per_angstrom2": 0.021,
            },
            specification=specification,
        )
    with pytest.raises(ValueError, match="scale-aware Frobenius"):
        runner.validate_hessian_numerical_quality(
            {
                **diagnostics,
                "raw_asymmetry_frobenius_norm_hartree_per_angstrom2": 0.021,
                "relative_raw_asymmetry_frobenius": 0.0021,
            },
            specification=specification,
        )


def test_imaginary_mode_recovery_tries_both_directions_and_selects_lower(
    tmp_path,
    monkeypatch,
):
    protocol = runner.load_json(PROTOCOL_PATH)
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    branch = {
        "status": "success",
        "phase": "gas",
        "source_state_index": 0,
        "converged": True,
        "final_positions_angstrom": atoms.get_positions().tolist(),
        "final_energy": {"total_energy_hartree": -10.0},
    }
    frequencies = np.asarray([0.0] * 6 + [-50.0, 100.0, 200.0])
    modes = np.eye(9)
    analysis = {
        "frequencies_cm1": frequencies,
        "modes_cartesian": modes,
        "selection": {
            "selected_indices": [6, 7, 8],
            "negative_selected_frequencies_cm1": [-50.0],
        },
    }
    starts = []

    def fake_optimization(**kwargs):
        direction = -1 if kwargs["initial_positions"][2, 0] < -0.24 else 1
        starts.append((direction, np.asarray(kwargs["initial_positions"])))
        energy = -10.01 if direction < 0 else -10.005
        return {
            "status": "success",
            "phase": "gas",
            "source_state_index": 0,
            "converged": True,
            "final_energy": {"total_energy_hartree": energy},
            "final_positions_angstrom": kwargs["initial_positions"].tolist(),
        }

    monkeypatch.setattr(runner, "_run_optimization", fake_optimization)

    accepted, evidence = runner._recover_imaginary_mode_minimum(
        branch=branch,
        initial_analysis=analysis,
        template_atoms=atoms,
        source_calculator=object(),
        protocol=protocol,
        phase="gas",
        minimum_dir=tmp_path,
    )

    assert [direction for direction, _positions in starts] == [-1, 1]
    assert evidence["selected_direction"] == -1
    assert accepted["final_energy"]["total_energy_hartree"] == -10.01
    assert evidence["energy_lowering_kcal_mol"] == pytest.approx(
        0.01 * runner.KCAL_PER_HARTREE
    )
    assert max(
        np.linalg.norm(starts[0][1] - atoms.get_positions(), axis=1)
    ) == pytest.approx(0.5)


def test_hessian_sensitivity_reference_must_be_the_primary_hessian(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    protocol = runner.load_json(PROTOCOL_PATH)
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )

    def evidence(scale, filename, displacement):
        hessian = np.eye(9) * scale
        path = repository / filename
        np.save(path, hessian, allow_pickle=False)
        primary = protocol["thermochemistry"]["primary"]
        frequency = runner.MWFrequency(
            str(repository / f"{filename}.out"),
            atoms,
            temperature=protocol["thermochemistry"]["temperature_kelvin"],
            ilowfreq=primary["ilowfreq"],
            omega0_cm1=primary["omega0_cm1"],
            nu_floor_cm1=primary["nu_floor_cm1"],
            device="cpu",
        )
        frequency.verbosity = 0
        frequencies, _ = frequency.compute_frequencies(hessian)
        specification = protocol["hessian_and_stationary_point"]
        selection = runner.select_vibrational_modes(
            frequencies,
            atom_count=len(atoms),
            linear=frequency._is_linear_molecule(
                tol=specification["linearity_moment_ratio_threshold"]
            ),
            imaginary_frequency_cutoff_cm1=specification[
                "imaginary_frequency_cutoff_cm1"
            ],
            maximum_rigid_mode_leakage_cm1=specification[
                "maximum_rigid_mode_leakage_cm1"
            ],
        )
        return (
            {
                "cartesian_displacement_angstrom": displacement,
                "maximum_absolute_raw_hessian_hartree_per_angstrom2": (
                    scale
                ),
                "maximum_raw_asymmetry_hartree_per_angstrom2": 0.0,
                "raw_hessian_frobenius_norm_hartree_per_angstrom2": (
                    3.0 * scale
                ),
                "raw_asymmetry_frobenius_norm_hartree_per_angstrom2": 0.0,
                "relative_raw_asymmetry_frobenius": 0.0,
                "path": filename,
                "sha256": runner.sha256_file(path),
                "shape": [9, 9],
            },
            frequencies,
            selection,
        )

    primary_hessian, primary_frequencies, primary_selection = evidence(
        0.01,
        "primary.npy",
        0.002,
    )
    alternative_hessian, alternative_frequencies, alternative_selection = evidence(
        0.02,
        "alternative.npy",
        0.002,
    )
    records = []
    for displacement in (0.001, 0.002, 0.004):
        hessian_record = {
            **alternative_hessian,
            "cartesian_displacement_angstrom": displacement,
        }
        records.append(
            {
                "cartesian_displacement_angstrom": displacement,
                "hessian": hessian_record,
                "all_projected_frequencies_cm1": alternative_frequencies.tolist(),
                "vibrational_mode_selection": alternative_selection,
                "selected_mode_rms_difference_from_reference_cm1": 0.0,
            }
        )
    internally_consistent_but_unanchored = {
        "required": True,
        "reference_displacement_angstrom": 0.002,
        "maximum_selected_mode_rms_difference_cm1": 0.0,
        "gate_threshold_cm1": 25.0,
        "passed": True,
        "records": records,
    }

    with pytest.raises(ValueError, match="primary-reference anchor"):
        runner._validate_hessian_displacement_sensitivity(
            internally_consistent_but_unanchored,
            atoms=atoms,
            primary_hessian_record=primary_hessian,
            primary_frequencies_cm1=primary_frequencies,
            primary_selection=primary_selection,
            protocol=protocol,
            compound_id="mobley_1952272",
            model_name="maceoff23m",
            phase="gas",
            minimum_index=0,
            evidence_root=repository,
        )


def test_source_state_energy_repeats_ignore_only_a_common_absolute_offset():
    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = [1.0, 1.5, 2.0]
    template = Atoms("H2", positions=positions[0])

    class RepeatedCalculator:
        def __init__(self, relative_scale=1.0):
            self.calls = 0
            self.relative_scale = relative_scale

        def calculate_many(self, atoms_list, properties=("energy",)):
            repeat = self.calls
            self.calls += 1
            energies = [
                self.relative_scale**repeat * atoms.get_positions()[1, 0]
                + 100.0 * repeat
                for atoms in atoms_list
            ]
            return BatchResult(energies=np.asarray(energies))

    energies, evidence = runner._evaluate_state_energies(
        calculator=RepeatedCalculator(),
        template_atoms=template,
        positions=positions,
        batch_size=3,
        repeat_count=2,
        maximum_relative_difference_kcal_mol=1.0e-9,
    )

    assert energies == pytest.approx([1.0, 1.5, 2.0])
    assert evidence["repeat_count"] == 2
    assert len(evidence["repeats"]) == 2
    assert evidence["repeats"][0]["energy_hartree"] == pytest.approx([1.0, 1.5, 2.0])
    assert evidence["repeats"][1]["relative_energy_kcal_mol"] == pytest.approx(
        [0.0, 0.5 * runner.KCAL_PER_HARTREE, runner.KCAL_PER_HARTREE]
    )
    assert evidence[
        "maximum_repeat_relative_energy_difference_kcal_mol"
    ] == pytest.approx(0.0, abs=1.0e-12)

    with pytest.raises(ValueError, match="Repeated MLIP"):
        runner._evaluate_state_energies(
            calculator=RepeatedCalculator(relative_scale=1.01),
            template_atoms=template,
            positions=positions,
            batch_size=3,
            repeat_count=2,
            maximum_relative_difference_kcal_mol=1.0e-6,
        )


def test_label_keys_are_rejected_recursively_and_incomplete_sealing_fails(tmp_path):
    assert runner._contains_forbidden_label_key(
        {"nested": [{"experimental_kcal_mol": -1.0}]}
    )
    args = type(
        "Args",
        (),
        {
            "protocol": str(PROTOCOL_PATH),
            "record_dir": str(tmp_path / "records"),
            "output": str(tmp_path / "sealed.json"),
        },
    )()

    with pytest.raises(ValueError, match="Record directory is missing"):
        runner.seal_command(args)


def test_frozen_charge_correction_reports_the_exact_artifact_provenance(
    water_mol2,
):
    atoms = runner.MOL2Reader(str(water_mol2), charge=0, mult=1)
    evidence = {
        "charge_record": "frozen/am1bcc.json",
        "charge_record_sha256": "a" * 64,
        "charge_vector_sha256": "b" * 64,
    }
    correction = runner.FrozenChargeOpenMMCorrection(
        atoms,
        atoms.get_initial_charges(),
        charge_evidence=evidence,
        model="obc2",
        nonpolar="ace",
        platform="Reference",
    )

    result = correction.evaluate(atoms, need_forces=True)

    assert correction.mode == "fixed"
    assert result.forces_hartree_per_angstrom.shape == (len(atoms), 3)
    assert result.provenance["charge_provider"] == {
        "source": "frozen-am1bcc-json",
        **evidence,
    }


def test_strict_record_validator_rejects_semantically_incomplete_self_hash():
    protocol, fingerprint, manifest = runner.load_qrrho_protocol(PROTOCOL_PATH)
    model = protocol["models"][0]
    case = protocol["cases"][0]
    incomplete = runner.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": ("route1-multi-mlip-phase-specific-qrrho-model-case"),
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": fingerprint,
            "source_partition": "development",
            "status": "success",
            "model": model,
            "compound_id": case["compound_id"],
            "name": case["name"],
            "flexibility_bin": case["flexibility_bin"],
            "claim_boundary": protocol["claim_boundary"],
        }
    )

    with pytest.raises(ValueError, match="environment is missing"):
        runner.validate_model_case_record(
            incomplete,
            protocol=protocol,
            fingerprint=fingerprint,
            source_manifest=manifest,
            model=model,
            protocol_case=case,
            evidence_root=REPOSITORY_ROOT,
        )


def test_repository_evidence_path_rejects_absolute_parent_and_symlink_escape(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    inside = repository / "inside.json"
    inside.write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (repository / "escape.json").symlink_to(outside)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)

    assert (
        runner._resolve_repository_file(
            "inside.json",
            name="test evidence",
        )
        == inside
    )
    with pytest.raises(ValueError, match="absolute or escape"):
        runner._resolve_repository_file(str(outside), name="test evidence")
    with pytest.raises(ValueError, match="absolute or escape"):
        runner._resolve_repository_file("../outside.json", name="test evidence")
    with pytest.raises(ValueError, match="symlink"):
        runner._resolve_repository_file("escape.json", name="test evidence")


def test_durable_seal_copies_hessian_by_content_hash(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    source_root = repository / "work"
    durable_root = repository / "durable"
    source_root.mkdir(parents=True)
    hessian = np.eye(3)
    source_hessian = source_root / "hessian.npy"
    np.save(source_hessian, hessian, allow_pickle=False)
    hessian_sha256 = runner.sha256_file(source_hessian)
    sensitivity_hessian = source_root / "hessian-sensitivity.npy"
    np.save(sensitivity_hessian, 2.0 * hessian, allow_pickle=False)
    sensitivity_sha256 = runner.sha256_file(sensitivity_hessian)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    record = {
        "content_sha256": "a" * 64,
        "phases": {
            "gas": {
                "minimum_analyses": [
                    {
                        "status": "valid",
                        "hessian": {
                            "path": "work/hessian.npy",
                            "sha256": hessian_sha256,
                        },
                        "hessian_displacement_sensitivity": {
                            "required": True,
                            "records": [
                                {
                                    "hessian": {
                                        "path": "work/hessian-sensitivity.npy",
                                        "sha256": sensitivity_sha256,
                                    }
                                }
                            ],
                        },
                    }
                ]
            },
            "solution": {"minimum_analyses": []},
        },
    }

    durable = runner._make_durable_record(
        record,
        source_evidence_root=source_root,
        durable_root=durable_root,
    )

    copied = durable_root / "hessians" / f"{hessian_sha256}.npy"
    sensitivity_copied = durable_root / "hessians" / f"{sensitivity_sha256}.npy"
    assert copied.is_file()
    assert sensitivity_copied.is_file()
    assert runner.sha256_file(copied) == hessian_sha256
    assert runner.sha256_file(sensitivity_copied) == sensitivity_sha256
    assert durable["phases"]["gas"]["minimum_analyses"][0]["hessian"]["path"] == (
        f"durable/hessians/{hessian_sha256}.npy"
    )
    assert (
        durable["phases"]["gas"]["minimum_analyses"][0][
            "hessian_displacement_sensitivity"
        ]["records"][0]["hessian"]["path"]
        == f"durable/hessians/{sensitivity_sha256}.npy"
    )
    assert durable["durable_evidence"] == {
        "schema_version": 1,
        "source_record_content_sha256": "a" * 64,
        "hessian_storage": "repository-relative-content-addressed-npy",
        "hessian_sha256": sorted([hessian_sha256, sensitivity_sha256]),
    }
    assert durable["content_sha256"] == runner.artifact_content_sha256(durable)


def test_solution_phase_energy_requires_exact_obc2_ace_components_and_provenance():
    protocol, _fingerprint, _manifest = runner.load_qrrho_protocol(PROTOCOL_PATH)
    charge_evidence = {
        "charge_method": "am1bcc",
        "charge_record": "charges/example.json",
        "charge_record_sha256": "a" * 64,
        "charge_vector_sha256": "b" * 64,
    }
    blank_solution = {
        "total_energy_hartree": 1.0,
        "gas_energy_hartree": 1.0,
        "solvent_energy_hartree": 0.0,
        "solvent_components_hartree": {},
        "solvent_provenance": {},
    }
    with pytest.raises(ValueError, match="exact OBC-II polar and ACE nonpolar"):
        runner._validate_energy_result(
            blank_solution,
            phase="solution",
            name="solution test energy",
            protocol=protocol,
            charge_evidence=charge_evidence,
        )

    solvent_provenance = {
        "provider": "openmm",
        "provider_version": protocol["implementation_freeze"]["required_versions"][
            "openmm"
        ],
        "method": "gb",
        "platform": "Reference",
        "model": "obc2",
        "profile": "obc2-mbondi2",
        "nonpolar": "ace",
        "solvent": "water",
        "charge_provider": {
            "source": "frozen-am1bcc-json",
            **charge_evidence,
        },
    }
    runner._validate_energy_result(
        {
            "total_energy_hartree": 1.03,
            "gas_energy_hartree": 1.0,
            "solvent_energy_hartree": 0.03,
            "solvent_components_hartree": {
                "polar": 0.02,
                "nonpolar": 0.01,
            },
            "solvent_provenance": solvent_provenance,
        },
        phase="solution",
        name="solution test energy",
        protocol=protocol,
        charge_evidence=charge_evidence,
    )
    corrupted = {
        **solvent_provenance,
        "charge_provider": {
            **solvent_provenance["charge_provider"],
            "charge_vector_sha256": "c" * 64,
        },
    }
    with pytest.raises(ValueError, match="charge_provider"):
        runner._validate_energy_result(
            {
                "total_energy_hartree": 1.03,
                "gas_energy_hartree": 1.0,
                "solvent_energy_hartree": 0.03,
                "solvent_components_hartree": {
                    "polar": 0.02,
                    "nonpolar": 0.01,
                },
                "solvent_provenance": corrupted,
            },
            phase="solution",
            name="solution test energy",
            protocol=protocol,
            charge_evidence=charge_evidence,
        )
