from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maple.solvation.release.aimnet2_multisolvent_force import (
    FORCE_MEASUREMENT_ARTIFACT,
    FORCE_MEASUREMENT_CLAIM_BOUNDARY,
    FORCE_MEASUREMENT_SCHEMA_VERSION,
    FORCE_MEASUREMENT_STATUS,
    FORCE_PREQUALIFICATION_LEAVES,
    FORCE_PREQUALIFICATION_STEPS_A,
    FORCE_PREQUALIFICATION_THRESHOLDS,
    FORCE_SOURCE_PARITY_THRESHOLDS,
    NO_CAPABILITIES,
    canonical_sha256,
    cartesian_directions,
    finalize_force_replicates,
    geometry_internal_directions,
    summarize_force_record,
    topology_preflight,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
    / "route2-aimnet2-smooth-ddpcm-force-prequalification-protocol-v1.json"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "route2_release"
    / "run_aimnet2_multisolvent_force_prequalification.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("aimnet2_force_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_reciprocity_audit() -> dict[str, object]:
    bilinear = []
    for probe in range(4):
        bilinear.append(
            {
                "probe_index": probe,
                "left_P_right_eV": 0.0,
                "right_P_left_eV": 0.0,
                "reciprocity_absolute_error_eV": 0.0,
                "reciprocity_relative_error": 0.0,
                "apply_adjoint_eV": 0.0,
                "apply_adjoint_absolute_error_eV": 0.0,
                "apply_adjoint_relative_error": 0.0,
            }
        )
    charge_fd = []
    for probe in range(4):
        for step in (0.001, 0.0003, 0.0001):
            charge_fd.append(
                {
                    "probe_index": probe,
                    "step_e": step,
                    "analytic_eV_per_e": 0.0,
                    "finite_difference_eV_per_e": 0.0,
                    "absolute_error_eV_per_e": 0.0,
                    "relative_error": 0.0,
                }
            )
    return {
        "seed": 20260815,
        "requested_probe_count": 4,
        "effective_probe_count": 4,
        "bilinear_records": bilinear,
        "charge_directional_fd_records": charge_fd,
        "maximum_reciprocity_absolute_error_eV": 0.0,
        "maximum_reciprocity_relative_error": 0.0,
        "maximum_apply_adjoint_absolute_error_eV": 0.0,
        "maximum_apply_adjoint_relative_error": 0.0,
        "maximum_charge_fd_absolute_error_eV_per_e": 0.0,
        "maximum_charge_fd_relative_error": 0.0,
        "source_gradient_half_error_eV_per_source_unit": 0.0,
        "charge_gauge_vjp_norm_eV_per_A": 0.0,
        "thresholds": {
            "reciprocity_absolute_eV": 1.0e-8,
            "reciprocity_relative": 1.0e-8,
            "charge_fd_absolute_eV_per_e": 5.0e-6,
            "charge_fd_relative": 5.0e-5,
            "charge_gauge_vjp_norm_eV_per_A": 1.0e-7,
        },
        "gate_passed": True,
    }


def _synthetic_record(*, replicate: str = "primary") -> dict[str, object]:
    return _synthetic_force_record(replicate=replicate)


def _synthetic_force_record(
    *,
    replicate: str = "primary",
    mode: str = "directional",
    first_direction_errors: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gradient_amplitude: float = 0.2,
) -> dict[str, object]:
    positions = np.asarray([[-1.0, 0.2, 0.0], [0.7, -0.3, 0.4], [0.1, 0.6, -0.5]])
    radii = np.asarray([1.2, 1.85, 1.52])
    continuum_configuration_sha256 = "c" * 64
    continuum_topology = {
        "active_coefficient_deletion": False,
        "cavity_topology_sha256": "f" * 64,
        "coefficient_count": 75,
        "coefficient_topology_sha256": "f" * 64,
        "configuration_sha256": continuum_configuration_sha256,
        "laboratory_fixed_surface_grid": False,
    }
    geometry_sha256 = "1" * 64
    internal = geometry_internal_directions(positions, geometry_sha256)
    base_gradient = gradient_amplitude * internal["projected_breathing"]
    scales = {
        "vacuum": 1.0,
        "continuum": 0.3,
        "electrostatic_total": 1.3,
        "nonpolar": -0.1,
        "total": 1.2,
    }
    gradients = {
        leaf: (scales[leaf] * base_gradient).tolist()
        for leaf in FORCE_PREQUALIFICATION_LEAVES
    }
    energies = {
        "vacuum": -2.0,
        "continuum": -0.4,
        "electrostatic_total": -2.4,
        "nonpolar": 0.1,
        "total": -2.3,
    }
    vectors = (
        internal if mode == "directional" else cartesian_directions(len(positions))
    )
    directions = {}
    first_name = next(iter(vectors))
    for name, vector in vectors.items():
        samples = []
        for step_index, step in enumerate(FORCE_PREQUALIFICATION_STEPS_A):
            plus = {}
            minus = {}
            for leaf in FORCE_PREQUALIFICATION_LEAVES:
                derivative = float(np.sum(np.asarray(gradients[leaf]) * vector))
                error = (
                    first_direction_errors[step_index]
                    if name == first_name
                    and leaf in {"vacuum", "electrostatic_total", "total"}
                    else 0.0
                )
                plus[leaf] = energies[leaf] + step * (derivative + error)
                minus[leaf] = energies[leaf] - step * (derivative + error)
            samples.append(
                {
                    "step_angstrom": step,
                    "plus_energies_eV": plus,
                    "minus_energies_eV": minus,
                    "plus_model_topology_sha256": "a" * 64,
                    "minus_model_topology_sha256": "a" * 64,
                    "plus_model_minimum_cutoff_margin_angstrom": 1.0,
                    "minus_model_minimum_cutoff_margin_angstrom": 1.0,
                    "plus_continuum_topology": copy.deepcopy(continuum_topology),
                    "minus_continuum_topology": copy.deepcopy(continuum_topology),
                    "plus_continuum_preflight": topology_preflight(
                        positions + step * vector,
                        radii,
                        transition_width_angstrom2=0.18,
                        partition_lmax=8,
                    ),
                    "minus_continuum_preflight": topology_preflight(
                        positions - step * vector,
                        radii,
                        transition_width_angstrom2=0.18,
                        partition_lmax=8,
                    ),
                }
            )
        directions[name] = {"vector": vector.tolist(), "samples": samples}
    task = {
        "record_ordinal": 0,
        "selection_index": 7,
        "opaque_record_id": "2" * 64,
        "partition": "development",
        "geometry_handle": "synthetic",
        "geometry_sha256": geometry_sha256,
        "canonical_solvent": "water",
        "stratum": "water",
        "role": "core",
        "atomic_numbers": [1, 6, 8],
        "coordinates_angstrom": positions.tolist(),
        "cavity_radii_angstrom": radii.tolist(),
    }
    task["task_sha256"] = canonical_sha256(task)
    runtime_provenance = {
        "runtime_kind": "aimnet-reconstructed-float64-runtime-v4",
        "aimnet_package_version": "0.2.0",
        "aimnet_runtime_files_sha256": {"synthetic.py": "3" * 64},
    }
    nonpolar_runtime_provenance = {
        "provider": "pyscf-smd-libsolvent-cds",
        "pyscf_version": "2.8.0",
        "solvent": "water",
        "pyscf_smd_solvent": "water",
        "upstream_entrypoint": "pyscf.solvent.smd.get_cds_legacy",
    }
    numerical_runtime = {
        "python_implementation": "CPython",
        "python_version": "3.11.0",
        "numpy_version": "2.0.0",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "threadpools": [{"user_api": "blas", "num_threads": 1}],
        "torch_version": "synthetic",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    record = {
        "artifact": FORCE_MEASUREMENT_ARTIFACT,
        "schema_version": FORCE_MEASUREMENT_SCHEMA_VERSION,
        "status": FORCE_MEASUREMENT_STATUS,
        "do_not_commit": True,
        "execution_git_commit": "b" * 40,
        "execution_git_tree": "c" * 40,
        "protocol_file_sha256": "4" * 64,
        "selection_manifest_file_sha256": "5" * 64,
        "selection_manifest_canonical_sha256": "6" * 64,
        "checkpoint_sha256": "7" * 64,
        "checkpoint_bytes": 1234,
        "aimnet2_runtime_kind": runtime_provenance["runtime_kind"],
        "aimnet2_runtime_provenance": runtime_provenance,
        "aimnet2_runtime_provenance_sha256": canonical_sha256(runtime_provenance),
        "aimnet2_provider_id": (
            "maple.route2.model.aimnet2-frozen-charge-multisolvent-float64.impl.v1"
        ),
        "scalar_id": (
            "route2-candidate-aimnet2-frozen-charge-multisolvent-"
            "smoothpartitionharmonic-ddpcm-pyscf-smdcds-v1"
        ),
        "profile_id": (
            "route2-profile-candidate-aimnet2-frozen-charge-multisolvent-"
            "smoothpartitionharmonic-ddpcm-pyscf-smdcds-v1"
        ),
        "task": task,
        "stage": "gate1",
        "mode": mode,
        "replicate": replicate,
        "process_identity": {
            "pid": 101 if replicate == "primary" else 202,
            "process_import_token": ("8" * 64 if replicate == "primary" else "9" * 64),
            "python_executable": "/usr/bin/python",
        },
        "scalar_fingerprint_sha256": "a" * 64,
        "model_configuration_sha256": "b" * 64,
        "continuum_configuration_sha256": continuum_configuration_sha256,
        "continuum_provenance_sha256": "d" * 64,
        "nonpolar_provider_id": "synthetic-pyscf-smd-cds",
        "nonpolar_profile_id": "synthetic-nonpolar-profile",
        "nonpolar_configuration_sha256": "e" * 64,
        "nonpolar_runtime_provenance": nonpolar_runtime_provenance,
        "nonpolar_runtime_provenance_sha256": canonical_sha256(
            nonpolar_runtime_provenance
        ),
        "center_model_topology_sha256": "a" * 64,
        "center_model_minimum_cutoff_margin_angstrom": 1.0,
        "center_continuum_topology_sha256": canonical_sha256(continuum_topology),
        "numerical_runtime": numerical_runtime,
        "numerical_runtime_sha256": canonical_sha256(numerical_runtime),
        "center": {
            "positions_angstrom": positions.tolist(),
            "energies_eV": energies,
            "gradients_eV_per_A": gradients,
            "source": [
                [0.1, 0.0, 0.0, 0.0],
                [-0.05, 0.0, 0.0, 0.0],
                [-0.05, 0.0, 0.0, 0.0],
            ],
            "reaction_field": [[0.0, 0.0, 0.0, 0.0]] * 3,
            "aimnet2_source_response_parity": {
                "energy_absolute_error_eV": 0.0,
                "charge_max_absolute_error_e": 0.0,
                "intrinsic_gradient_max_absolute_error_eV_per_A": 0.0,
                "charge_vjp_max_absolute_error_eV_per_A": 0.0,
                "repeat_energy_absolute_error_eV": 0.0,
                "repeat_charge_max_absolute_error_e": 0.0,
                "repeat_intrinsic_gradient_max_absolute_error_eV_per_A": 0.0,
                "repeat_charge_vjp_max_absolute_error_eV_per_A": 0.0,
                "ordinary_forward_intrinsic_gradient_report_only_max_absolute_error_eV_per_A": 0.0,
            },
            "continuum_reciprocity_audit": _synthetic_reciprocity_audit(),
            "model_topology_sha256": "a" * 64,
            "model_minimum_cutoff_margin_angstrom": 1.0,
            "continuum_topology": copy.deepcopy(continuum_topology),
            "continuum_preflight": topology_preflight(
                positions,
                radii,
                transition_width_angstrom2=0.18,
                partition_lmax=8,
            ),
        },
        "directions": directions,
        "runtime_seconds": 1.0,
        "capabilities": dict(NO_CAPABILITIES),
        "claim_boundary": FORCE_MEASUREMENT_CLAIM_BOUNDARY,
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _bindings(record: dict[str, object]) -> dict[str, object]:
    return {
        key: copy.deepcopy(record[key])
        for key in (
            "execution_git_commit",
            "execution_git_tree",
            "protocol_file_sha256",
            "selection_manifest_file_sha256",
            "selection_manifest_canonical_sha256",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "aimnet2_runtime_kind",
            "aimnet2_runtime_provenance_sha256",
            "aimnet2_provider_id",
            "scalar_id",
            "profile_id",
            "scalar_fingerprint_sha256",
            "model_configuration_sha256",
            "continuum_configuration_sha256",
            "continuum_provenance_sha256",
            "nonpolar_provider_id",
            "nonpolar_profile_id",
            "nonpolar_configuration_sha256",
            "nonpolar_runtime_provenance_sha256",
            "center_model_topology_sha256",
            "center_model_minimum_cutoff_margin_angstrom",
            "center_continuum_topology_sha256",
            "numerical_runtime_sha256",
            "task",
            "stage",
            "mode",
        )
    }


def _rehash(record: dict[str, object]) -> None:
    record["record_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def test_force_protocol_is_exact_target_free_and_fail_closed():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol["status"] == "preregistered-not-executed"
    assert protocol["selection"]["record_count"] == 24
    assert protocol["selection"]["geometry_group_count"] == 14
    assert protocol["selection"]["selection_reads_experimental_targets"] is False
    assert (
        protocol["selection"]["selection_reads_aimnet2_or_continuum_outputs"] is False
    )
    assert tuple(protocol["finite_difference"]["steps_angstrom"]) == (
        FORCE_PREQUALIFICATION_STEPS_A
    )
    assert tuple(protocol["finite_difference"]["leaves"]) == (
        FORCE_PREQUALIFICATION_LEAVES
    )
    assert protocol["thresholds"] == FORCE_PREQUALIFICATION_THRESHOLDS
    assert protocol["source_response_thresholds"] == FORCE_SOURCE_PARITY_THRESHOLDS
    assert (
        "ordinary_forward_intrinsic_gradient_report_only_max_absolute_error_eV_per_A"
        not in protocol["source_response_thresholds"]
    )
    assert protocol["capabilities"] == NO_CAPABILITIES
    assert protocol["scalar"]["aimnet2_runtime_kind"].endswith("runtime-v4")
    assert protocol["scalar"]["aimnet2_multisolvent_provider"].endswith(
        "multisolvent-float64.impl.v1"
    )
    assert protocol["stages"]["gate0_full_cartesian_canary"] == {
        "panel_record_ordinal": 11,
        "selection_index": 362,
        "atom_count": 6,
        "solvent": "water",
        "mode": "full-cartesian",
    }


def test_geometry_directions_are_normalized_and_rigid_projected():
    positions = np.asarray([[-1.0, 0.2, 0.0], [0.7, -0.3, 0.4], [0.1, 0.6, -0.5]])
    directions = geometry_internal_directions(positions, "1" * 64)
    centered = positions - np.mean(positions, axis=0)

    assert tuple(directions) == (
        "projected_breathing",
        "sha256_internal_0",
        "sha256_internal_1",
    )
    for direction in directions.values():
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1.0e-12)
        np.testing.assert_allclose(np.sum(direction, axis=0), 0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            np.sum(np.cross(centered, direction), axis=0), 0.0, atol=1.0e-12
        )


def test_topology_preflight_is_deterministic_and_bounded():
    result = topology_preflight(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
        [1.7, 1.2],
        transition_width_angstrom2=0.18,
        partition_lmax=8,
    )

    assert len(result["pair_state_sha256"]) == 64
    assert sum(result["pair_state_counts"].values()) == 2
    assert result["required_algebraic_degree"] <= 192


def test_exact_quadratic_synthetic_force_record_passes_reducer():
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    bindings = _bindings(primary)
    finalized = finalize_force_replicates(primary, replay, expected_bindings=bindings)
    summary = summarize_force_record(
        finalized, replay_measurement=replay, expected_bindings=bindings
    )

    assert summary["all_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["capabilities"] == NO_CAPABILITIES


def test_force_reducer_rejects_one_bad_same_scalar_leaf():
    primary = _synthetic_force_record(
        replicate="primary", first_direction_errors=(0.01, 0.01, 0.01)
    )
    replay = _synthetic_force_record(
        replicate="replay", first_direction_errors=(0.01, 0.01, 0.01)
    )
    bindings = _bindings(primary)
    finalized = finalize_force_replicates(primary, replay, expected_bindings=bindings)
    summary = summarize_force_record(
        finalized, replay_measurement=replay, expected_bindings=bindings
    )

    assert summary["all_gates_passed"] is False
    assert summary["gates"]["same_scalar_derivatives_all_leaves"] is False


def test_force_reducer_rejects_an_open_energy_component_ledger():
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    primary["center"]["energies_eV"]["total"] += 1.0e-4
    _rehash(primary)

    with pytest.raises(ValueError, match="energy-component ledger"):
        finalize_force_replicates(
            primary,
            replay,
            expected_bindings=_bindings(replay),
        )


def test_cold_replicates_are_bound_before_reduction():
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    bindings = _bindings(primary)

    finalized = finalize_force_replicates(primary, replay, expected_bindings=bindings)

    assert finalized["replicate"] == "primary-with-cold-replay"
    assert (
        finalized["deterministic_replay"]["stencil_energy_max_absolute_error_eV"] == 0.0
    )
    assert finalized["summary"]["all_gates_passed"] is True
    expected = canonical_sha256(
        {key: value for key, value in finalized.items() if key != "record_sha256"}
    )
    assert finalized["record_sha256"] == expected


def test_cold_replicate_task_drift_fails_closed():
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    replay["task"]["geometry_handle"] = "tampered"
    replay["task"]["task_sha256"] = canonical_sha256(
        {key: value for key, value in replay["task"].items() if key != "task_sha256"}
    )
    _rehash(replay)

    with pytest.raises(ValueError, match="task drifted"):
        finalize_force_replicates(
            primary,
            replay,
            expected_bindings=_bindings(primary),
        )


def test_cold_replicate_same_process_fails_closed():
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    replay["process_identity"] = copy.deepcopy(primary["process_identity"])
    _rehash(replay)

    with pytest.raises(ValueError, match="distinct cold process"):
        finalize_force_replicates(
            primary,
            replay,
            expected_bindings=_bindings(primary),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact", "not-the-force-contract"),
        ("status", "fabricated"),
        ("do_not_commit", False),
        ("capabilities", {tier: True for tier in NO_CAPABILITIES}),
        ("protocol_file_sha256", "e" * 64),
    ),
)
def test_cold_replicate_fabricated_identity_fails_closed(field, value):
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    replay[field] = value
    _rehash(replay)

    with pytest.raises(ValueError):
        finalize_force_replicates(
            primary,
            replay,
            expected_bindings=_bindings(primary),
        )


def test_cold_replicate_duplicate_direction_fails_closed():
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    names = tuple(replay["directions"])
    replay["directions"][names[1]]["vector"] = copy.deepcopy(
        replay["directions"][names[0]]["vector"]
    )
    _rehash(replay)

    with pytest.raises(ValueError, match="deterministic policy"):
        finalize_force_replicates(
            primary,
            replay,
            expected_bindings=_bindings(primary),
        )


@pytest.mark.parametrize("bad_degree", (-999, False, 1.9))
def test_force_preflight_degree_is_exactly_recomputed(bad_degree):
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    for record in (primary, replay):
        record["center"]["continuum_preflight"][
            "required_algebraic_degree"
        ] = bad_degree
        _rehash(record)

    with pytest.raises(ValueError, match="exact geometry/radii recomputation"):
        finalize_force_replicates(
            primary,
            replay,
            expected_bindings=_bindings(primary),
        )


def test_force_reciprocity_is_recomputed_from_raw_records():
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    for record in (primary, replay):
        record["center"]["continuum_reciprocity_audit"][
            "maximum_reciprocity_absolute_error_eV"
        ] = 1.0
        _rehash(record)

    with pytest.raises(ValueError, match="raw reciprocity measurements"):
        finalize_force_replicates(
            primary,
            replay,
            expected_bindings=_bindings(primary),
        )


@pytest.mark.parametrize("parity_name", tuple(FORCE_SOURCE_PARITY_THRESHOLDS))
def test_force_source_parity_hard_terms_fail_closed(parity_name):
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    for record in (primary, replay):
        record["center"]["aimnet2_source_response_parity"][parity_name] = 1.0
        _rehash(record)

    with pytest.raises(ValueError, match="source-response parity hard gate"):
        finalize_force_replicates(
            primary,
            replay,
            expected_bindings=_bindings(primary),
        )


def test_standalone_finalized_verifier_rejects_negative_replay_delta():
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    bindings = _bindings(primary)
    finalized = finalize_force_replicates(primary, replay, expected_bindings=bindings)
    finalized["deterministic_replay"]["energy_absolute_error_eV"] = -1.0
    _rehash(finalized)

    with pytest.raises(ValueError, match="must be non-negative"):
        summarize_force_record(
            finalized,
            replay_measurement=replay,
            expected_bindings=bindings,
        )


def test_directional_thresholds_do_not_apply_cartesian_rms_gate():
    primary = _synthetic_force_record(
        replicate="primary",
        mode="directional",
        first_direction_errors=(0.0014, 0.0004, 0.0001),
        gradient_amplitude=0.2,
    )
    replay = _synthetic_force_record(
        replicate="replay",
        mode="directional",
        first_direction_errors=(0.0014, 0.0004, 0.0001),
        gradient_amplitude=0.2,
    )
    bindings = _bindings(primary)
    finalized = finalize_force_replicates(primary, replay, expected_bindings=bindings)
    summary = summarize_force_record(
        finalized, replay_measurement=replay, expected_bindings=bindings
    )

    assert (
        summary["aggregate_derivative_errors"]["vacuum"]["rms_absolute_error_eV_per_A"][
            0
        ]
        > FORCE_PREQUALIFICATION_THRESHOLDS["cartesian_step_rms_eV_per_A"]
    )
    assert summary["all_gates_passed"] is True


def test_cartesian_thresholds_do_not_apply_directional_relative_gate():
    primary = _synthetic_force_record(
        replicate="primary",
        mode="full-cartesian",
        first_direction_errors=(0.00121, 0.00039, 0.00014),
        gradient_amplitude=0.02,
    )
    replay = _synthetic_force_record(
        replicate="replay",
        mode="full-cartesian",
        first_direction_errors=(0.00121, 0.00039, 0.00014),
        gradient_amplitude=0.02,
    )
    bindings = _bindings(primary)
    finalized = finalize_force_replicates(primary, replay, expected_bindings=bindings)
    summary = summarize_force_record(
        finalized, replay_measurement=replay, expected_bindings=bindings
    )
    vacuum = summary["aggregate_derivative_errors"]["vacuum"]

    assert vacuum["rms_absolute_error_eV_per_A"][0] < (
        FORCE_PREQUALIFICATION_THRESHOLDS["cartesian_step_rms_eV_per_A"]
    )
    assert vacuum["maximum_terminal_relative_error"] > (
        FORCE_PREQUALIFICATION_THRESHOLDS["directional_terminal_relative"]
    )
    assert summary["all_gates_passed"] is True


@pytest.mark.parametrize("mode", ("directional", "full-cartesian"))
def test_zero_terminal_does_not_hide_a_nonconvergent_middle_step(mode):
    primary = _synthetic_force_record(
        replicate="primary",
        mode=mode,
        first_direction_errors=(0.0, 0.001, 0.0),
    )
    replay = _synthetic_force_record(
        replicate="replay",
        mode=mode,
        first_direction_errors=(0.0, 0.001, 0.0),
    )
    bindings = _bindings(primary)
    finalized = finalize_force_replicates(primary, replay, expected_bindings=bindings)
    summary = summarize_force_record(
        finalized,
        replay_measurement=replay,
        expected_bindings=bindings,
    )

    assert summary["gates"]["every_direction_order_or_plateau"] is False
    assert summary["all_gates_passed"] is False


@pytest.mark.parametrize(
    "errors, expected_gate",
    [((0.001, 0.00025, 0.0), True), ((0.0, 0.001, 0.001), False)],
)
def test_zero_endpoint_order_remains_json_safe_when_finalized(errors, expected_gate):
    primary = _synthetic_force_record(
        replicate="primary", first_direction_errors=errors, gradient_amplitude=0.0
    )
    replay = _synthetic_force_record(
        replicate="replay", first_direction_errors=errors, gradient_amplitude=0.0
    )
    finalized = finalize_force_replicates(
        primary, replay, expected_bindings=_bindings(primary)
    )
    assert (
        finalized["summary"]["gates"]["every_direction_order_or_plateau"]
        is expected_gate
    )
    assert json.loads(json.dumps(finalized, allow_nan=False)) == finalized


def test_failed_force_measurement_is_hash_bound_and_not_finalizable(tmp_path):
    runner = _runner()
    primary = _synthetic_record(replicate="primary")
    replay = _synthetic_record(replicate="replay")
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    failed = runner._failed_measurement_record(
        task=primary["task"],
        checkpoint=checkpoint,
        protocol_sha256=primary["protocol_file_sha256"],
        selection_manifest_file_sha256=primary["selection_manifest_file_sha256"],
        selection_manifest_canonical_sha256=primary[
            "selection_manifest_canonical_sha256"
        ],
        execution_git_commit=primary["execution_git_commit"],
        execution_git_tree=primary["execution_git_tree"],
        stage=primary["stage"],
        mode=primary["mode"],
        replicate="primary",
        error=RuntimeError("synthetic failure"),
        runtime_seconds=1.0,
        checkout_revalidation={"passed": True, "error_type": None, "error": None},
    )

    assert failed["status"] == "private-measurement-failed"
    assert failed["checkout_revalidation"]["passed"] is True
    assert failed["record_sha256"] == canonical_sha256(
        {key: value for key, value in failed.items() if key != "record_sha256"}
    )
    with pytest.raises(ValueError, match="identity/status"):
        finalize_force_replicates(
            failed,
            replay,
            expected_bindings=_bindings(primary),
        )


def test_runner_freezes_gate_order_and_selection_manifest_identity():
    runner = _runner()
    parser = runner._parser()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    manifest = {
        "schema": "maple-pure-mace-polar-analytic-gaussian-qm-attribution-prereg-v2",
        "panel": {
            "record_count": 24,
            "geometry_group_count": 14,
            "record_selection_frozen": True,
            "reselection_after_any_result": False,
            "records": [
                {
                    "ordinal": index,
                    "stratum": (
                        "high_epsilon_nonwater"
                        if index < 8
                        else "low_epsilon_nonwater" if index < 16 else "water"
                    ),
                }
                for index in range(24)
            ],
        },
    }
    manifest["artifact_sha256"] = canonical_sha256(manifest)
    amended = copy.deepcopy(protocol)
    amended["selection"]["origin_artifact_canonical_sha256"] = manifest[
        "artifact_sha256"
    ]

    assert len(runner._validate_selection_manifest(manifest, amended)) == 24
    runner._validate_stage(
        protocol, stage="gate0", record_ordinal=11, mode="full-cartesian"
    )
    with pytest.raises(ValueError, match="Gate-0"):
        runner._validate_stage(
            protocol, stage="gate0", record_ordinal=10, mode="full-cartesian"
        )
    with pytest.raises(RuntimeError, match="Gate-2 remains closed"):
        runner._validate_stage(
            protocol, stage="gate2", record_ordinal=0, mode="full-cartesian"
        )
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    for command in ("measure", "finalize"):
        assert all(
            action.dest != "overwrite"
            for action in subparsers.choices[command]._actions
        )


def test_force_private_path_is_protected_by_tracked_gitignore():
    runner = _runner()
    path = ROOT / ".omx" / "benchmarks" / "force-private-path-test.json"

    assert runner._private_path(path, directory=False) == path.resolve()
    with pytest.raises(ValueError, match="below repository .omx"):
        runner._private_path(ROOT / "force-private-path-test.json", directory=False)


@pytest.mark.parametrize("passed", (False, True))
def test_finalize_exit_status_preserves_positive_and_negative_artifacts(
    tmp_path, monkeypatch, capsys, passed
):
    runner = _runner()
    input_path = tmp_path / "input.json"
    input_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "finalized.json"
    args = SimpleNamespace(
        **{
            name: input_path
            for name in (
                "source",
                "mnsol_protocol",
                "prior_private",
                "selection_manifest",
                "checkpoint",
                "force_protocol",
            )
        },
        primary=tmp_path / "primary.json",
        replay=tmp_path / "replay.json",
        output=output,
    )
    finalized = {
        "record_sha256": "f" * 64,
        "summary": {"all_gates_passed": passed, "gates": {"synthetic": passed}},
        "capabilities": dict(NO_CAPABILITIES),
    }
    monkeypatch.setattr(runner, "_clean_commit", lambda: ("commit", "tree"))
    monkeypatch.setattr(runner, "_private_path", lambda path, **kwargs: path)
    monkeypatch.setattr(runner, "_validate_protocol", lambda *a, **kw: ({}, "sha"))
    monkeypatch.setattr(
        runner,
        "_object",
        lambda *a, **kw: {
            "artifact_sha256": "a" * 64,
            "task": {"record_ordinal": 11},
            "stage": "gate0",
            "mode": "full-cartesian",
        },
    )
    for name in (
        "_validate_selection_manifest",
        "_validate_stage",
        "load_mnsol_protocol",
        "load_mnsol_v2012",
        "_prepare_task",
        "_configure_torch_threads",
        "_scalar_identity",
        "numerical_runtime_identity",
        "_expected_bindings",
        "_assert_same_clean_commit",
    ):
        monkeypatch.setattr(runner, name, lambda *a, **kw: {})
    monkeypatch.setattr(runner, "finalize_force_replicates", lambda *a, **kw: finalized)
    if passed:
        runner._finalize_command(args)
    else:
        with pytest.raises(SystemExit) as exc:
            runner._finalize_command(args)
        assert exc.value.code == 1
    assert json.loads(output.read_text(encoding="utf-8")) == finalized
    assert str(output) in capsys.readouterr().out
