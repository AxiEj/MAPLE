from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units

from maple.function.dispatcher.solvfe import bulk_water_campaign
from maple.function.dispatcher.solvfe import bulk_water_evidence
from maple.function.dispatcher.solvfe.bulk_water import (
    BulkWaterValidationError,
    validate_water_box,
)
from maple.function.dispatcher.solvfe.bulk_water_campaign import (
    PREREGISTERED_NPT_CONFIG_WITHOUT_SEED_SHA256,
    BulkWaterCampaignConfig,
    _evaluate_bulk_water_replica_summaries,
    evaluate_bulk_water_replica_campaign,
    load_bulk_water_npt_artifact,
)
from maple.function.dispatcher.solvfe.bulk_water_npt import (
    NPT_IMPLEMENTATION_PATHS,
    BulkWaterNPTConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_preregistered_protocol_hash_is_literal_and_matches_defaults():
    assert PREREGISTERED_NPT_CONFIG_WITHOUT_SEED_SHA256 == (
        "59c39954c8df951fae3189dae2b483c78f347ebc2acc3937bd724263360136bf"
    )
    assert BulkWaterNPTConfig().content_hash_without_seed == (
        PREREGISTERED_NPT_CONFIG_WITHOUT_SEED_SHA256
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rehash_summary(summary: dict) -> dict:
    summary.pop("result_hash", None)
    summary["result_hash"] = bulk_water_campaign.canonical_sha256(summary)
    return summary


def _replica(
    seed: int,
    density: float,
    sem: float,
    *,
    config: BulkWaterNPTConfig | None = None,
) -> dict:
    config = config or BulkWaterNPTConfig(seed=seed)
    diagnostic_passed = (
        abs(density - config.density_reference_g_per_ml)
        / config.density_reference_g_per_ml
        <= config.density_relative_error_limit
    )
    replica = {
        "schema": "maple-route-a-bulk-water-npt-summary-v1",
        "config": {
            "schema": "maple-route-a-bulk-water-npt-config-v1",
            **asdict(config),
            "content_hash": config.content_hash,
            "content_hash_without_seed": config.content_hash_without_seed,
        },
        "source": {
            "sha256": "b" * 64,
            "topology": {"water_count": 512},
        },
        "calculator": {
            "calculator": {
                "checkpoint_sha256": "c" * 64,
                "interaction_cutoff_angstrom": 6.0,
            },
        },
        "implementation": {
            "schema": "maple-route-a-bulk-water-implementation-v1",
            "project_root": "/synthetic/maple",
            "git_head": "d" * 40,
            "git_dirty": False,
            "git_status_sha256": _sha256(""),
            "implementation_file_sha256": {
                path: _sha256(path)
                for path in sorted(NPT_IMPLEMENTATION_PATHS)
            },
        },
        "trajectory": {
            "ensemble": "NPT",
            "integrator": (
                "ase.md.nose_hoover_chain.IsotropicMTKNPT"
            ),
            "production_duration_ps": config.production_duration_ps,
            "production_frame_count": config.production_frame_count,
            "semantic_array_sha256": {
                "production_positions_angstrom": _sha256(
                    f"positions-{seed}"
                ),
                "production_cells_angstrom": _sha256(f"cells-{seed}"),
                "production_velocities_angstrom_per_ase_time": _sha256(
                    f"velocities-{seed}"
                ),
            },
        },
        "diagnostics": {
            "production_temperature_mean_k": config.temperature_k,
            "production_pressure_mean_bar": config.pressure_bar,
            "production_density_mean_g_per_ml": density,
            "production_density_block_sem_g_per_ml": sem,
            "production_density_half_relative_drift": 0.001,
        },
        "gates": {
            "engineering_checks": {
                "all_md_steps_checked": True,
                "finite_observations": True,
                "temperature_below_emergency_limit": True,
                "force_below_emergency_limit": True,
                "water_topology_preserved": True,
                "cell_cutoff_safe": True,
            },
            "engineering_stability_passed": True,
            "minimum_npt_diagnostic_eligible": diagnostic_passed,
            "paper_duration_fidelity_passed": (
                config.paper_duration_fidelity
            ),
        },
    }
    return _rehash_summary(replica)


def _write_replica_artifact(
    root: Path,
    source_summary: dict,
) -> dict:
    root.mkdir()
    seed = source_summary["config"]["seed"]
    density = source_summary["diagnostics"][
        "production_density_mean_g_per_ml"
    ]
    config = BulkWaterNPTConfig(
        timestep_fs=1.0,
        precondition_steps=0,
        equilibration_steps=4,
        production_steps=8,
        sample_interval_steps=2,
        seed=seed,
        rdf_bin_width_angstrom=0.5,
        rdf_max_angstrom=2.0,
        rdf_block_count=2,
        interaction_cutoff_angstrom=1.5,
        cutoff_margin_angstrom=0.1,
        minimum_diagnostic_duration_ps=0.008,
        minimum_diagnostic_frames=4,
    )
    production_density = np.array(
        [
            density - 0.0001,
            density + 0.0001,
            density - 0.0001,
            density + 0.0001,
        ],
        dtype=np.float64,
    )
    atom_count = 12
    numbers = np.tile(
        np.array([8, 1, 1], dtype=np.int64),
        4,
    )
    total_mass = float(np.sum(Atoms(numbers=numbers).get_masses()))
    density_to_volume = (
        lambda value: total_mass * 1.66053906660 / value
    )
    production_volumes = np.array(
        [density_to_volume(value) for value in production_density],
        dtype=np.float64,
    )
    production_edges = production_volumes ** (1.0 / 3.0)
    production_cells = np.array(
        [np.diag([edge, edge, edge]) for edge in production_edges],
        dtype=np.float64,
    )
    bond_length = 0.9572
    bond_angle = np.deg2rad(104.52)
    local_water = np.array(
        [
            [0.0, 0.0, 0.0],
            [bond_length, 0.0, 0.0],
            [
                bond_length * np.cos(bond_angle),
                bond_length * np.sin(bond_angle),
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    oxygen_centers = np.array(
        [
            [1.0, 1.0, 1.0],
            [3.4, 1.0, 1.0],
            [1.0, 3.4, 3.4],
            [3.4, 3.4, 3.4],
        ],
        dtype=np.float64,
    )
    frame_positions = np.concatenate(
        [local_water + center for center in oxygen_centers],
        axis=0,
    )
    production_positions = np.repeat(
        frame_positions[np.newaxis, :, :],
        4,
        axis=0,
    )
    topology = validate_water_box(
        Atoms(
            numbers=numbers,
            positions=frame_positions,
            cell=production_cells[0],
            pbc=True,
        )
    )
    base_velocities = np.arange(
        1,
        atom_count * 3 + 1,
        dtype=np.float64,
    ).reshape((atom_count, 3))
    target_kinetic_energy = (
        0.5 * 3.0 * atom_count * units.kB * config.temperature_k
    )
    velocity_scale = math.sqrt(
        target_kinetic_energy
        / (
            0.5
            * np.einsum(
                "a,ai,ai->",
                Atoms(numbers=numbers).get_masses(),
                base_velocities,
                base_velocities,
            )
        )
    )
    frame_velocities = base_velocities * velocity_scale
    production_velocities = np.repeat(
        frame_velocities[np.newaxis, :, :],
        4,
        axis=0,
    )
    velocity_max = float(
        np.max(np.linalg.norm(frame_velocities, axis=1))
    )
    stages = np.array(
        [
            "initial",
            "equilibration",
            "equilibration",
            "production",
            "production",
            "production",
            "production",
        ],
        dtype="U16",
    )
    steps = np.array([0, 2, 4, 6, 8, 10, 12], dtype=np.int64)
    observation_density = np.full(7, density, dtype=np.float64)
    observation_density[stages == "production"] = production_density
    observation_volume = np.array(
        [density_to_volume(value) for value in observation_density],
        dtype=np.float64,
    )
    stepwise_density = np.full(13, density, dtype=np.float64)
    stepwise_density[[6, 8, 10, 12]] = production_density
    stepwise_volume = np.array(
        [density_to_volume(value) for value in stepwise_density],
        dtype=np.float64,
    )
    stepwise_edge = stepwise_volume ** (1.0 / 3.0)
    arrays = {
        "observation_stage": stages,
        "observation_step": steps,
        "observation_time_fs": steps.astype(np.float64),
        "observation_temperature_k": np.full(
            7,
            config.temperature_k,
            dtype=np.float64,
        ),
        "observation_potential_energy_ev": np.zeros(
            7,
            dtype=np.float64,
        ),
        "observation_kinetic_energy_ev": np.zeros(
            7,
            dtype=np.float64,
        )
        + target_kinetic_energy,
        "observation_total_energy_ev": np.zeros(
            7,
            dtype=np.float64,
        )
        + target_kinetic_energy,
        "observation_force_rms_ev_per_angstrom": np.zeros(
            7,
            dtype=np.float64,
        ),
        "observation_force_max_ev_per_angstrom": np.zeros(
            7,
            dtype=np.float64,
        ),
        "observation_pressure_bar": np.full(
            7,
            config.pressure_bar,
            dtype=np.float64,
        ),
        "observation_volume_angstrom3": observation_volume,
        "observation_density_g_per_ml": observation_density,
        "atomic_numbers": numbers,
        "production_positions_angstrom": production_positions,
        "production_cells_angstrom": production_cells,
        "production_velocities_angstrom_per_ase_time": (
            production_velocities
        ),
        "production_density_g_per_ml": production_density,
        "production_density_block_g_per_ml": np.array(
            [
                np.mean(production_density[:2]),
                np.mean(production_density[2:]),
            ],
            dtype=np.float64,
        ),
        "stepwise_stage_code": np.array(
            [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
            dtype=np.uint8,
        ),
        "stepwise_step": np.arange(13, dtype=np.int64),
        "stepwise_temperature_k": np.full(
            13,
            config.temperature_k,
            dtype=np.float64,
        ),
        "stepwise_potential_energy_ev": np.zeros(
            13,
            dtype=np.float64,
        ),
        "stepwise_velocity_max_angstrom_per_ase_time": np.full(
            13,
            velocity_max,
            dtype=np.float64,
        ),
        "stepwise_force_max_ev_per_angstrom": np.zeros(
            13,
            dtype=np.float64,
        ),
        "stepwise_oh_min_distance_angstrom": np.full(
            13,
            topology["oh_distance_range_angstrom"][0],
            dtype=np.float64,
        ),
        "stepwise_oh_max_distance_angstrom": np.full(
            13,
            topology["oh_distance_range_angstrom"][1],
            dtype=np.float64,
        ),
        "stepwise_hoh_min_angle_degrees": np.full(
            13,
            topology["hoh_angle_range_degrees"][0],
            dtype=np.float64,
        ),
        "stepwise_hoh_max_angle_degrees": np.full(
            13,
            topology["hoh_angle_range_degrees"][1],
            dtype=np.float64,
        ),
        "stepwise_minimum_cell_height_angstrom": stepwise_edge,
        "stepwise_maximum_cell_height_angstrom": stepwise_edge,
        "stepwise_volume_angstrom3": stepwise_volume,
        "stepwise_density_g_per_ml": stepwise_density,
    }
    rdf_arrays, _ = bulk_water_evidence.compute_bulk_water_rdf_evidence(
        production_positions,
        production_cells,
        numbers,
        config=config,
    )
    arrays.update(rdf_arrays)
    recomputed = bulk_water_evidence.recompute_bulk_water_npt_evidence(
        arrays,
        config=config,
        water_count=4,
    )
    semantic = {
        name: bulk_water_campaign._array_sha256(value)
        for name, value in arrays.items()
    }
    summary = _replica(
        seed,
        density,
        source_summary["diagnostics"][
            "production_density_block_sem_g_per_ml"
        ],
        config=config,
    )
    summary["source"]["topology"]["water_count"] = 4
    summary["calculator"]["calculator"][
        "interaction_cutoff_angstrom"
    ] = config.interaction_cutoff_angstrom
    summary["trajectory"]["semantic_array_sha256"] = semantic
    summary["trajectory"].update(recomputed["trajectory"])
    summary["diagnostics"] = recomputed["diagnostics"]
    summary["gates"] = recomputed["gates"]
    summary.pop("result_hash", None)
    summary["result_hash"] = bulk_water_campaign.canonical_sha256(summary)
    with (root / "arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    manifest = {
        "schema": "maple-route-a-bulk-water-npt-artifact-v1",
        "result_hash": summary["result_hash"],
        "files": {
            "arrays": {
                "path": "arrays.npz",
                "sha256": bulk_water_campaign.raw_sha256(
                    root / "arrays.npz"
                ),
            },
            "summary": {
                "path": "summary.json",
                "canonical_sha256": bulk_water_campaign.canonical_sha256(
                    summary
                ),
            },
        },
        "semantic_array_sha256": semantic,
    }
    (root / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return summary


def _rewrite_rehashed_summary(root: Path, summary: dict) -> None:
    _rehash_summary(summary)
    (root / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["result_hash"] = summary["result_hash"]
    manifest["files"]["summary"]["canonical_sha256"] = (
        bulk_water_campaign.canonical_sha256(summary)
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _rewrite_rehashed_arrays(
    root: Path,
    summary: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    with (root / "arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    semantic = {
        name: bulk_water_campaign._array_sha256(value)
        for name, value in arrays.items()
    }
    summary["trajectory"]["semantic_array_sha256"] = semantic
    _rehash_summary(summary)
    (root / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["result_hash"] = summary["result_hash"]
    manifest["semantic_array_sha256"] = semantic
    manifest["files"]["arrays"]["sha256"] = (
        bulk_water_campaign.raw_sha256(root / "arrays.npz")
    )
    manifest["files"]["summary"]["canonical_sha256"] = (
        bulk_water_campaign.canonical_sha256(summary)
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_campaign_requires_independent_matching_replicas_and_reports_open_gates():
    replicas = (
        _replica(11, 0.9968, 0.0005),
        _replica(12, 0.9972, 0.0006),
        _replica(13, 0.9970, 0.0005),
    )

    result = _evaluate_bulk_water_replica_summaries(
        replicas,
        config=BulkWaterCampaignConfig(),
    )

    assert result["schema"].endswith("campaign-summary-v1")
    assert result["replica_count"] == 3
    assert result["diagnostics"]["density_relative_error"] < 0.001
    assert result["gates"]["independent_replicas_passed"] is True
    assert result["gates"]["npt_density_accuracy_passed"] is True
    assert result["gates"]["finite_size_validation_passed"] is False
    assert result["gates"]["external_rdf_validation_passed"] is False
    assert result["gates"]["cross_engine_validation_passed"] is False
    assert result["gates"]["hamiltonian_freeze_eligible"] is False
    assert result["gates"]["route_a_accuracy_claim_allowed"] is False


def test_campaign_fails_closed_on_seed_or_identity_reuse():
    first = _replica(11, 0.997, 0.0005)
    duplicate_seed = _replica(11, 0.9971, 0.0005)
    with pytest.raises(BulkWaterValidationError, match="distinct seeds"):
        _evaluate_bulk_water_replica_summaries(
            (first, duplicate_seed, _replica(13, 0.9972, 0.0005)),
            config=BulkWaterCampaignConfig(),
        )

    mismatch = copy.deepcopy(_replica(12, 0.9971, 0.0005))
    mismatch["calculator"]["calculator"]["checkpoint_sha256"] = "e" * 64
    _rehash_summary(mismatch)
    with pytest.raises(BulkWaterValidationError, match="checkpoint"):
        _evaluate_bulk_water_replica_summaries(
            (first, mismatch, _replica(13, 0.9972, 0.0005)),
            config=BulkWaterCampaignConfig(),
        )

    override = copy.deepcopy(_replica(12, 0.9971, 0.0005))
    override["implementation"][
        "implementation_campaign_identity_sha256"
    ] = "f" * 64
    _rehash_summary(override)
    with pytest.raises(BulkWaterValidationError, match="implementation"):
        _evaluate_bulk_water_replica_summaries(
            (first, override, _replica(13, 0.9972, 0.0005)),
            config=BulkWaterCampaignConfig(),
        )


def test_campaign_rejects_summary_that_does_not_match_result_hash():
    tampered = _replica(11, 0.9970, 0.0005)
    tampered["diagnostics"]["production_density_mean_g_per_ml"] = 1.5

    with pytest.raises(BulkWaterValidationError, match="result hash"):
        _evaluate_bulk_water_replica_summaries(
            (
                tampered,
                _replica(12, 0.9971, 0.0005),
                _replica(13, 0.9969, 0.0005),
            ),
            config=BulkWaterCampaignConfig(),
        )


def test_public_campaign_evaluator_requires_artifact_directories():
    with pytest.raises(BulkWaterValidationError, match="artifact director"):
        evaluate_bulk_water_replica_campaign(
            (
                _replica(11, 0.9970, 0.0005),
                _replica(12, 0.9971, 0.0005),
                _replica(13, 0.9969, 0.0005),
            ),
            config=BulkWaterCampaignConfig(),
        )


def test_campaign_rejects_duplicate_core_trajectory_identity():
    first = _replica(11, 0.9970, 0.0005)
    duplicate = _replica(12, 0.9971, 0.0005)
    duplicate["trajectory"]["semantic_array_sha256"] = copy.deepcopy(
        first["trajectory"]["semantic_array_sha256"]
    )
    _rehash_summary(duplicate)

    with pytest.raises(BulkWaterValidationError, match="trajectory"):
        _evaluate_bulk_water_replica_summaries(
            (first, duplicate, _replica(13, 0.9972, 0.0005)),
            config=BulkWaterCampaignConfig(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("git_head", None),
        ("git_dirty", None),
        ("git_dirty", True),
        ("git_status_sha256", "invalid"),
    ),
)
def test_campaign_rejects_malformed_implementation_identity(
    field,
    value,
):
    malformed = _replica(11, 0.9970, 0.0005)
    malformed["implementation"][field] = value
    _rehash_summary(malformed)

    with pytest.raises(
        BulkWaterValidationError,
        match="IMPLEMENTATION|implementation",
    ):
        _evaluate_bulk_water_replica_summaries(
            (
                malformed,
                _replica(12, 0.9971, 0.0005),
                _replica(13, 0.9969, 0.0005),
            ),
            config=BulkWaterCampaignConfig(),
        )


def test_campaign_rejects_clean_git_with_nonempty_status_digest():
    malformed = _replica(11, 0.9970, 0.0005)
    malformed["implementation"]["git_dirty"] = False
    malformed["implementation"]["git_status_sha256"] = "a" * 64
    _rehash_summary(malformed)

    with pytest.raises(BulkWaterValidationError, match="Git status"):
        _evaluate_bulk_water_replica_summaries(
            (
                malformed,
                _replica(12, 0.9971, 0.0005),
                _replica(13, 0.9969, 0.0005),
            ),
            config=BulkWaterCampaignConfig(),
        )


def test_campaign_density_gate_is_numeric_not_caller_asserted():
    replicas = (
        _replica(11, 1.08, 0.001),
        _replica(12, 1.081, 0.001),
        _replica(13, 1.079, 0.001),
    )
    result = _evaluate_bulk_water_replica_summaries(
        replicas,
        config=BulkWaterCampaignConfig(
            density_relative_error_limit=0.03,
        ),
    )

    assert result["diagnostics"]["density_relative_error"] > 0.08
    assert result["gates"]["npt_density_accuracy_passed"] is False
    assert result["gates"]["hamiltonian_freeze_eligible"] is False


def test_campaign_density_interval_must_fit_acceptance_band():
    replicas = (
        _replica(11, 0.9970, 10.0),
        _replica(12, 0.9971, 10.0),
        _replica(13, 0.9969, 10.0),
    )

    result = _evaluate_bulk_water_replica_summaries(
        replicas,
        config=BulkWaterCampaignConfig(),
    )

    assert result["diagnostics"]["density_relative_error"] < 0.001
    assert result["gates"]["npt_density_accuracy_passed"] is False


def test_campaign_uses_student_t_interval_for_three_replicas():
    result = _evaluate_bulk_water_replica_summaries(
        (
            _replica(11, 0.9968, 0.0005),
            _replica(12, 0.9972, 0.0006),
            _replica(13, 0.9970, 0.0005),
        ),
        config=BulkWaterCampaignConfig(),
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["density_interval_degrees_of_freedom"] == 2
    assert diagnostics["density_student_t_95_critical_value"] == (
        pytest.approx(4.3026527297)
    )
    mean = diagnostics["density_mean_g_per_ml"]
    sem = diagnostics["density_conservative_sem_g_per_ml"]
    assert diagnostics["density_student_t_95_interval_g_per_ml"] == (
        pytest.approx(
            [
                mean - 4.3026527297 * sem,
                mean + 4.3026527297 * sem,
            ]
        )
    )


def test_campaign_rejects_weakened_replica_acceptance_protocol():
    weak = BulkWaterNPTConfig(
        temperature_relative_tolerance=900.0,
        pressure_mean_tolerance_bar=1.0e8,
        density_half_drift_limit=5.0,
    )

    with pytest.raises(BulkWaterValidationError, match="protocol"):
        _evaluate_bulk_water_replica_summaries(
            (
                _replica(11, 0.9970, 0.0005, config=weak),
                _replica(
                    12,
                    0.9971,
                    0.0005,
                    config=BulkWaterNPTConfig(
                        **{
                            **asdict(weak),
                            "seed": 12,
                        }
                    ),
                ),
                _replica(
                    13,
                    0.9969,
                    0.0005,
                    config=BulkWaterNPTConfig(
                        **{
                            **asdict(weak),
                            "seed": 13,
                        }
                    ),
                ),
            ),
            config=BulkWaterCampaignConfig(),
        )


def test_campaign_acceptance_limits_can_only_be_tightened():
    with pytest.raises(BulkWaterValidationError, match="3%"):
        BulkWaterCampaignConfig(density_relative_error_limit=0.031)
    with pytest.raises(BulkWaterValidationError, match="2%"):
        BulkWaterCampaignConfig(replica_density_spread_limit=0.021)


def test_campaign_rejects_placeholder_engineering_checks():
    placeholder = _replica(11, 0.9970, 0.0005)
    placeholder["gates"]["engineering_checks"] = {"placeholder": True}
    _rehash_summary(placeholder)

    with pytest.raises(BulkWaterValidationError, match="engineering"):
        _evaluate_bulk_water_replica_summaries(
            (
                placeholder,
                _replica(12, 0.9971, 0.0005),
                _replica(13, 0.9969, 0.0005),
            ),
            config=BulkWaterCampaignConfig(),
        )


def test_campaign_loader_rejects_tampered_replica_artifact(tmp_path):
    root = tmp_path / "replica"
    summary = _write_replica_artifact(
        root,
        _replica(11, 0.997, 0.0005),
    )

    assert load_bulk_water_npt_artifact(root)["result_hash"] == (
        summary["result_hash"]
    )
    summary["diagnostics"]["production_density_mean_g_per_ml"] = 2.0
    (root / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    with pytest.raises(BulkWaterValidationError, match="summary hash"):
        load_bulk_water_npt_artifact(root)


def test_campaign_loader_rejects_fully_rehashed_array_summary_mismatch(
    tmp_path,
):
    root = tmp_path / "replica"
    summary = _write_replica_artifact(
        root,
        _replica(11, 0.997, 0.0005),
    )
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["production_density_g_per_ml"] = np.full(
        4,
        2.0,
        dtype=np.float64,
    )
    _rewrite_rehashed_arrays(root, summary, arrays)

    with pytest.raises(BulkWaterValidationError, match="evidence|schema"):
        load_bulk_water_npt_artifact(root)


def test_campaign_loader_rejects_rehashed_rdf_arrays(tmp_path):
    root = tmp_path / "replica"
    summary = _write_replica_artifact(
        root,
        _replica(11, 0.997, 0.0005),
    )
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    arrays["rdf_oo_g"][:] = 999.0
    arrays["rdf_oh_g"][:] = -500.0
    _rewrite_rehashed_arrays(root, summary, arrays)

    with pytest.raises(
        BulkWaterValidationError,
        match="RDF|rdf|EVIDENCE|evidence",
    ):
        load_bulk_water_npt_artifact(root)


def test_campaign_loader_rejects_rehashed_rdf_features(tmp_path):
    root = tmp_path / "replica"
    summary = _write_replica_artifact(
        root,
        _replica(11, 0.997, 0.0005),
    )
    summary["diagnostics"]["rdf"] = {
        "oo": {
            "pair": [8, 8],
            "intermolecular_only": True,
            "block_count": 2,
        },
        "oh": {
            "pair": [8, 1],
            "intermolecular_only": True,
            "block_count": 2,
        },
        "hh": {
            "pair": [1, 1],
            "intermolecular_only": True,
            "block_count": 2,
        },
        "oo_features": {
            "first_peak_position_angstrom": 99.0,
            "first_peak_height": 100.0,
            "first_minimum_position_angstrom": 100.0,
            "coordination_number_at_first_minimum": 12_345.0,
        },
    }
    _rewrite_rehashed_summary(root, summary)

    with pytest.raises(
        BulkWaterValidationError,
        match="RDF|rdf|EVIDENCE|evidence",
    ):
        load_bulk_water_npt_artifact(root)


def test_campaign_loader_rejects_rehashed_placeholder_engineering_gate(
    tmp_path,
):
    root = tmp_path / "replica"
    summary = _write_replica_artifact(
        root,
        _replica(11, 0.997, 0.0005),
    )
    summary["gates"]["engineering_checks"] = {"placeholder": True}
    summary["gates"]["engineering_stability_passed"] = True
    _rewrite_rehashed_summary(root, summary)

    with pytest.raises(
        BulkWaterValidationError,
        match="EVIDENCE|evidence",
    ):
        load_bulk_water_npt_artifact(root)


def test_campaign_loader_rejects_rehashed_promotion_gate_claims(
    tmp_path,
):
    root = tmp_path / "replica"
    summary = _write_replica_artifact(
        root,
        _replica(11, 0.997, 0.0005),
    )
    for name in (
        "paper_integrator_fidelity_passed",
        "paper_protocol_reproduced",
        "independent_replicas_passed",
        "npt_density_validation_passed",
        "finite_size_validation_passed",
        "external_rdf_validation_passed",
        "cross_engine_validation_passed",
        "hamiltonian_freeze_eligible",
        "claim_eligible",
    ):
        summary["gates"][name] = True
    _rewrite_rehashed_summary(root, summary)

    with pytest.raises(
        BulkWaterValidationError,
        match="EVIDENCE|evidence",
    ):
        load_bulk_water_npt_artifact(root)


@pytest.mark.parametrize(
    "array_name",
    (
        "production_positions_angstrom",
        "production_velocities_angstrom_per_ase_time",
    ),
)
def test_campaign_loader_rejects_physical_state_scalar_mismatch(
    tmp_path,
    array_name,
):
    root = tmp_path / "replica"
    summary = _write_replica_artifact(
        root,
        _replica(11, 0.997, 0.0005),
    )
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    arrays[array_name][:] = 42.0
    _rewrite_rehashed_arrays(root, summary, arrays)

    with pytest.raises(
        BulkWaterValidationError,
        match="EVIDENCE|evidence|topology|temperature|velocity",
    ):
        load_bulk_water_npt_artifact(root)


def test_campaign_loader_rejects_dirty_flag_with_empty_status_digest(
    tmp_path,
):
    root = tmp_path / "replica"
    summary = _write_replica_artifact(
        root,
        _replica(11, 0.997, 0.0005),
    )
    summary["implementation"]["git_dirty"] = True
    summary["implementation"]["git_status_sha256"] = hashlib.sha256(
        b""
    ).hexdigest()
    _rewrite_rehashed_summary(root, summary)

    with pytest.raises(BulkWaterValidationError, match="Git status"):
        load_bulk_water_npt_artifact(root)


def test_campaign_recomputes_config_and_gate_evidence():
    bad_config = _replica(11, 0.997, 0.0005)
    bad_config["config"]["density_relative_error_limit"] = 0.04
    _rehash_summary(bad_config)
    with pytest.raises(BulkWaterValidationError, match="config hash"):
        _evaluate_bulk_water_replica_summaries(
            (
                bad_config,
                _replica(12, 0.997, 0.0005),
                _replica(13, 0.997, 0.0005),
            ),
            config=BulkWaterCampaignConfig(),
        )

    bad_gate = _replica(11, 1.08, 0.0005)
    bad_gate["gates"]["minimum_npt_diagnostic_eligible"] = True
    _rehash_summary(bad_gate)
    with pytest.raises(BulkWaterValidationError, match="numeric evidence"):
        _evaluate_bulk_water_replica_summaries(
            (
                bad_gate,
                _replica(12, 1.08, 0.0005),
                _replica(13, 1.08, 0.0005),
            ),
            config=BulkWaterCampaignConfig(),
        )


def test_campaign_cli_verifies_inputs_and_refuses_overwrite(
    tmp_path,
    capsys,
):
    replica_paths = []
    summaries = {}
    for seed, density in ((11, 0.9968), (12, 0.9972), (13, 0.9970)):
        root = tmp_path / f"replica-{seed}"
        root.mkdir()
        summaries[root.resolve()] = _replica(seed, density, 0.0005)
        replica_paths.append(root)

    script = (
        PROJECT_ROOT
        / "examples/solvation/route_a/"
        "aggregate_bulk_water_npt_replicas.py"
    )
    spec = importlib.util.spec_from_file_location(
        "route_a_aggregate_bulk_water_npt",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.evaluate_bulk_water_replica_campaign = (
        lambda paths, config: _evaluate_bulk_water_replica_summaries(
            [
                copy.deepcopy(summaries[Path(path).resolve()])
                for path in paths
            ],
            config=config,
        )
    )

    output = tmp_path / "campaign.json"
    arguments = [
        "--output",
        output.as_posix(),
        *(path.as_posix() for path in replica_paths),
    ]
    assert module.main(arguments) == 0
    capsys.readouterr()

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema"].endswith("campaign-artifact-v1")
    assert artifact["campaign"]["replica_count"] == 3
    assert artifact["campaign"]["gates"][
        "npt_density_accuracy_passed"
    ] is True
    artifact_hash = artifact.pop("artifact_hash")
    assert artifact_hash == bulk_water_campaign.canonical_sha256(artifact)
    assert set(
        artifact["implementation"]["implementation_file_sha256"]
    ) == module.IMPLEMENTATION_PATHS

    with pytest.raises(BulkWaterValidationError, match="OUTPUT_EXISTS"):
        module.main(arguments)
