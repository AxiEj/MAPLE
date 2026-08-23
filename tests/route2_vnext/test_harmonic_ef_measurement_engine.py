from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
import torch

from maple.solvation.experimental.mace_mdp_polar_harmonic import (
    HybridHarmonicEnergyState,
    HybridHarmonicRootStartDiagnostics,
    HybridHarmonicSolveDiagnostics,
    MACE_MDPPolarHybridSmoothHarmonicPES,
)
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.models.mace_mdp_polar_hybrid import (
    PermanentAnchoredInducedSourceModel,
)
from maple.solvation.release import harmonic_ef_measurement
from maple.solvation.release.admission import (
    H1_V2_PROFILE_ID,
    H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
    H1_V2_SCALAR_ID,
    H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
    H1_V2_STATE_ID,
    REPLICATE_GATE_NAMES,
)

ROOT = Path(__file__).parents[2]
V1_PREREGISTRATION = (
    ROOT / "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-harmonic-force-admission-v1.json"
)
V1_REPLICATE = (
    ROOT / "docs/route2/evidence/"
    "mace-mdp-polar-hybrid-harmonic-force-admission-replicate-a-4cf8db40.json"
)
V1_PROJECTION_GOLDEN = (
    ROOT / "tests/route2_vnext/data/harmonic_ef_v1_projection_golden.json"
)


def _historical_inputs():
    preregistration = json.loads(V1_PREREGISTRATION.read_text(encoding="utf-8"))
    replicate = json.loads(V1_REPLICATE.read_text(encoding="utf-8"))
    measurement = {
        key: replicate[key]
        for key in (
            "benzene_gepol_regression",
            "water_force_symmetry_loop",
            "aggregate",
        )
    }
    return (
        measurement,
        preregistration["frozen_runtime_contract"],
        preregistration["force_panel"],
    )


def _validate(measurement) -> None:
    _, runtime_contract, force_panel = _historical_inputs()
    harmonic_ef_measurement.validate_harmonic_ef_science_measurement(
        measurement,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )


def _rich_state_and_diagnostics(
    *, geometry_sha256: str = "a" * 64, atom_count: int = 2
):
    permanent = np.zeros((atom_count, 4))
    zero = np.zeros((atom_count, 4))
    induced = np.zeros((atom_count, 4))
    permanent[0] = [-0.2, 0.1, 0.0, 0.0]
    permanent[-1] = [0.2, -0.1, 0.0, 0.0]
    zero[0] = [-0.1, 0.0, 0.1, 0.0]
    zero[-1] = [0.1, 0.0, -0.1, 0.0]
    induced[0] = [0.01, 0.02, 0.03, 0.04]
    induced[-1] = [-0.01, -0.02, -0.03, -0.04]
    final = zero + induced
    induced = final - zero
    total = permanent + induced
    field = np.zeros((atom_count, 8))
    residual = np.zeros((atom_count, 8))
    state = HybridHarmonicEnergyState(
        geometry_sha256=geometry_sha256,
        evaluator_configuration_sha256="b" * 64,
        anchor_state_sha256="c" * 64,
        continuum_state_sha256="d" * 64,
        native_field_ev=field,
        induced_source4=induced,
        total_source4=total,
        boundary_rhs=np.asarray([1.0]),
        boundary_state=np.asarray([2.0]),
        vacuum_energy_ev=-3.0,
        polarization_energy_ev=-1.0,
        primal_residual_ev=0.0,
        cold_iterations=1,
        wide_iterations=1,
        replay_field_max_abs_difference_ev=0.0,
        replay_energy_abs_difference_ev=0.0,
        coefficient_topology_id="e" * 64,
    )
    cold = HybridHarmonicRootStartDiagnostics(
        initial_state_sha256="f" * 64,
        final_native_field_ev=field,
        final_residual_ev=residual,
        final_residual_norm_ev=0.0,
        final_polarization_energy_ev=-1.0,
        iterations=1,
        converged=True,
    )
    wide = replace(cold, initial_state_sha256="0" * 64)
    diagnostics = HybridHarmonicSolveDiagnostics(
        cold_start=cold,
        wide_start=wide,
        permanent_source4=permanent,
        response_zero_source4=zero,
        response_final_source4=final,
        induced_source4=induced,
        audit_coefficient_sum4=total,
        target_charge_e=0.0,
        source_basis_id="maple.route2.atomic-l1-source-space.v1",
        source_space_contract_sha256=H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
        source_component_order=(
            "net_monopole",
            "real_l1_m0",
            "real_l1_m1",
            "real_l1_m_minus1",
        ),
        receiver_basis_id="maple.route2.mace-polar-native-radial-field-space.v1",
        receiver_space_contract_sha256=H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
        source_role_identities=(
            ("permanent_source4", "geometry-only permanent atomic-l1 source"),
            ("response_zero_source4", "zero-field responsive subtraction reference"),
            (
                "response_final_source4",
                "field-conditioned responsive checkpoint source",
            ),
            ("induced_source4", "responsive(field)-responsive(zero-field)"),
            (
                "audit_coefficient_sum4",
                "non-operational permanent+induced coefficient audit projection",
            ),
        ),
    )
    return state, diagnostics


def test_measurement_engine_returns_only_deterministic_science_preimage(
    monkeypatch,
) -> None:
    historical, runtime_contract, force_panel = _historical_inputs()
    benzene = historical["benzene_gepol_regression"]
    water = historical["water_force_symmetry_loop"]
    monkeypatch.setattr(
        harmonic_ef_measurement, "_benzene_record", lambda **kwargs: benzene
    )
    monkeypatch.setattr(
        harmonic_ef_measurement, "_water_record", lambda **kwargs: water
    )

    result = harmonic_ef_measurement.run_harmonic_ef_measurement(
        asset_root=Path("/unused"),
        parent={},
        hybrid=object(),
        radial=object(),
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )

    assert tuple(result) == (
        "benzene_gepol_regression",
        "water_force_symmetry_loop",
        "aggregate",
    )
    assert result == historical


def test_replicate_gate_derivation_rejects_non_boolean_cleanliness() -> None:
    measurement, runtime_contract, force_panel = _historical_inputs()
    with pytest.raises(TypeError, match="exactly bool"):
        harmonic_ef_measurement.derive_harmonic_ef_replicate_gates(
            measurement,
            runtime_contract=runtime_contract,
            force_panel=force_panel,
            clean_content_addressed_execution=1,
        )


def test_replicate_gate_derivation_propagates_cleanliness_without_mutation() -> None:
    measurement, runtime_contract, force_panel = _historical_inputs()
    before = copy.deepcopy(measurement)
    gates = harmonic_ef_measurement.derive_harmonic_ef_replicate_gates(
        measurement,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
        clean_content_addressed_execution=False,
    )
    assert gates["clean_content_addressed_execution"] is False
    assert all(
        value
        for name, value in gates.items()
        if name != "clean_content_addressed_execution"
    )
    assert measurement == before
    assert tuple(gates) == REPLICATE_GATE_NAMES


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (
            ("benzene_gepol_regression", "force_convergence", "h4_difference_eV_per_A"),
            1.0,
        ),
        (
            ("aggregate", "maximum_local_richardson_error_eV_per_A"),
            1.0,
        ),
        (
            (
                "water_force_symmetry_loop",
                "independent_h4_directional_check",
                "absolute_error_eV_per_A",
            ),
            1.0,
        ),
        (
            ("water_force_symmetry_loop", "translation", "energy_absolute_error_eV"),
            1.0,
        ),
        (
            ("water_force_symmetry_loop", "rotation", "energy_absolute_error_eV"),
            1.0,
        ),
        (
            ("water_force_symmetry_loop", "closed_loop", "guarded_absolute_work_eV"),
            1.0,
        ),
        (
            ("water_force_symmetry_loop", "root_summary", "maximum_wide_iterations"),
            41,
        ),
        (
            ("water_force_symmetry_loop", "root_summary", "maximum_primal_residual_eV"),
            1.0,
        ),
    ),
)
def test_replicate_gate_derivation_rejects_contradictory_metric_tampering(
    path, replacement
) -> None:
    measurement, runtime_contract, force_panel = _historical_inputs()
    cursor = measurement
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement

    with pytest.raises(ValueError, match="contradict|gate"):
        harmonic_ef_measurement.derive_harmonic_ef_replicate_gates(
            measurement,
            runtime_contract=runtime_contract,
            force_panel=force_panel,
            clean_content_addressed_execution=True,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("aggregate", "maximum_local_richardson_error_eV_per_A"), float("nan")),
        (("water_force_symmetry_loop", "center_energy_eV"), float("inf")),
        (("water_force_symmetry_loop", "center_force", "forces_eV_per_A"), [[0.0]]),
        (("water_force_symmetry_loop", "root_summary", "topology_ids"), []),
        (
            ("water_force_symmetry_loop", "root_summary", "topology_ids"),
            ["0" * 64, "1" * 64],
        ),
        (("water_force_symmetry_loop", "closed_loop", "edges"), []),
        (
            ("water_force_symmetry_loop", "closed_loop", "cartesian_dofs"),
            [[0, 0]],
        ),
    ),
)
def test_science_measurement_validator_rejects_invalid_values(
    path, replacement
) -> None:
    measurement, _, _ = _historical_inputs()
    cursor = measurement
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises((TypeError, ValueError)):
        _validate(measurement)


@pytest.mark.parametrize("operation", ("missing", "unknown", "edge_unknown"))
def test_science_measurement_validator_rejects_key_tampering(operation) -> None:
    measurement, _, _ = _historical_inputs()
    if operation == "missing":
        del measurement["aggregate"]["water_rotation_energy_error_eV"]
    elif operation == "unknown":
        measurement["water_force_symmetry_loop"]["unexpected"] = 1
    else:
        measurement["water_force_symmetry_loop"]["closed_loop"]["edges"][0][
            "unexpected"
        ] = 1
    with pytest.raises(ValueError, match="fields are invalid"):
        _validate(measurement)


def test_gate_derivation_validates_measurement_before_reading_gates() -> None:
    measurement, runtime_contract, force_panel = _historical_inputs()
    measurement["aggregate"]["unexpected"] = False
    with pytest.raises(ValueError, match="fields are invalid"):
        harmonic_ef_measurement.derive_harmonic_ef_replicate_gates(
            measurement,
            runtime_contract=runtime_contract,
            force_panel=force_panel,
            clean_content_addressed_execution=True,
        )


def test_frozen_threshold_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        harmonic_ef_measurement.HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "maximum_closed_loop_work_abs_ev"
        ] = 1.0


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (
            (
                "water_force_symmetry_loop",
                "center_force",
                "error_estimates_eV_per_A",
                0,
                0,
            ),
            1.0,
        ),
        (("water_force_symmetry_loop", "closed_loop", "edges", 0, "work_eV"), 1.0),
        (
            (
                "water_force_symmetry_loop",
                "closed_loop",
                "edges",
                0,
                "work_error_bound_eV",
            ),
            1.0,
        ),
        (("benzene_gepol_regression", "coefficient_topology_id"), "f" * 64),
        (("water_force_symmetry_loop", "closed_loop", "gate_passed"), False),
        (("aggregate", "all_execution_gates_passed"), False),
        (("water_force_symmetry_loop", "geometry_angstrom", 0, 0), 1.0),
        (
            ("water_force_symmetry_loop", "independent_h4_directional_check", "seed"),
            20260817,
        ),
        (("water_force_symmetry_loop", "translation", "translation_angstrom", 0), 1.0),
        (("water_force_symmetry_loop", "closed_loop", "cartesian_dofs", 0, 0), 2),
        (
            ("water_force_symmetry_loop", "root_summary", "maximum_primal_residual_eV"),
            -1.0,
        ),
        (
            (
                "water_force_symmetry_loop",
                "root_summary",
                "maximum_total_charge_error_e",
            ),
            -1.0,
        ),
        (("water_force_symmetry_loop", "center_energy_eV"), "1.0"),
        (
            ("benzene_gepol_regression", "force_convergence", "fine_step_angstrom"),
            3.0e-4,
        ),
        (
            ("water_force_symmetry_loop", "translation", "energy_absolute_error_eV"),
            -1.0,
        ),
        (
            ("water_force_symmetry_loop", "rotation", "energy_absolute_error_eV"),
            -1.0,
        ),
        (("benzene_gepol_regression", "cavity_radii_angstrom", 0), -1.0),
        (("water_force_symmetry_loop", "cavity_radii_angstrom", 0), 0.0),
    ),
)
def test_validator_rejects_reviewer_adversarial_probes(path, replacement) -> None:
    measurement, _, _ = _historical_inputs()
    cursor = measurement
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises((TypeError, ValueError)):
        _validate(measurement)


def test_rich_v2_prepared_inputs_are_immutable_and_parsing_free() -> None:
    prepared = _prepared_rich_inputs()

    assert prepared.benzene.positions_angstrom == ((0.0, 0.0, 0.0),) * 12
    assert prepared.water.atomic_numbers == (8, 1, 1)
    assert copy.deepcopy(prepared) is prepared
    assert copy.deepcopy(prepared.benzene) is prepared.benzene
    assert copy.deepcopy(prepared.water) is prepared.water
    with pytest.raises(AttributeError):
        prepared.water = prepared.water
    with pytest.raises(TypeError):
        prepared.force_panel_contract["rotation_seed"] = 1


def test_rich_v2_state_leaf_capture_requires_typed_complete_inputs() -> None:
    assert harmonic_ef_measurement.RICH_HARMONIC_EF_V2_EXECUTION_ENABLED is True
    with pytest.raises(
        harmonic_ef_measurement.RichHarmonicEFStateLeafUnavailable,
        match="typed legacy state plus complete solve diagnostics",
    ):
        harmonic_ef_measurement.capture_rich_harmonic_ef_state_leaf_v2(
            object(),
            object(),
            prepared_pes_configuration_sha256="9" * 64,
        )


def test_rich_v2_state_leaf_blocker_reports_exact_missing_raw_leaves() -> None:
    state, diagnostics = _rich_state_and_diagnostics()
    missing = harmonic_ef_measurement.rich_harmonic_ef_state_leaf_missing_fields_v2(
        state
    )

    assert "geometry_sha256" not in missing
    assert "legacy_root_sha256" not in missing
    assert "provider_configuration_sha256" not in missing
    assert "prepared_pes_configuration_sha256" in missing
    assert "topology_id" not in missing
    assert "source_coefficient_basis_id" in missing
    assert "source_coefficient_order" in missing
    assert "source_space_contract_sha256" in missing
    assert "receiver_space_contract_sha256" in missing
    assert "audit_projection_contract" in missing
    assert "cold_start" in missing
    assert "wide_start" in missing
    assert "permanent_point_source4" in missing
    assert "polar_zero_reference_source4" in missing
    assert "polar_final_source4" in missing
    assert "induced_gto_source4" in missing
    assert "audit_coefficient_sum4" in missing
    assert (
        harmonic_ef_measurement.rich_harmonic_ef_state_leaf_missing_fields_v2(
            state, diagnostics, "9" * 64
        )
        == ()
    )


def test_capture_rich_v2_state_leaf_is_accepted_and_uses_total_start_energy() -> None:
    from maple.solvation.release.admission import StateLeafV2

    state, diagnostics = _rich_state_and_diagnostics()
    mapping = harmonic_ef_measurement.capture_rich_harmonic_ef_state_leaf_v2(
        state,
        diagnostics,
        prepared_pes_configuration_sha256="9" * 64,
    )

    leaf = StateLeafV2.from_mapping(mapping)
    assert leaf.state_leaf_sha256 == mapping["state_leaf_sha256"]
    assert mapping["legacy_root_sha256"] == state.root_sha256
    assert mapping["prepared_pes_configuration_sha256"] == "9" * 64
    assert mapping["state_leaf_sha256"] != mapping["legacy_root_sha256"]
    assert mapping["source_coefficient_basis_id"] == diagnostics.source_basis_id
    assert tuple(mapping["source_coefficient_order"]) == (
        "net_monopole",
        "real_l1_m0",
        "real_l1_m1",
        "real_l1_m_minus1",
    )
    assert mapping["polar_final_source4"] == diagnostics.response_final_source4.tolist()
    assert mapping["audit_coefficient_sum4"] == (
        diagnostics.audit_coefficient_sum4.tolist()
    )
    assert mapping["audit_coefficient_sum_charge_e"] == float(
        np.sum(diagnostics.audit_coefficient_sum4[:, 0])
    )
    assert mapping["cold_start"]["final_total_energy_eV"] == (
        state.vacuum_energy_ev + diagnostics.cold_start.final_polarization_energy_ev
    )
    assert mapping["wide_start"]["final_total_energy_eV"] == (
        state.vacuum_energy_ev + diagnostics.wide_start.final_polarization_energy_ev
    )
    assert mapping["cold_start"]["final_total_energy_eV"] != (
        diagnostics.cold_start.final_polarization_energy_ev
    )
    assert harmonic_ef_measurement.RICH_HARMONIC_EF_V2_EXECUTION_ENABLED is True


def test_capture_rich_v2_state_leaf_rejects_wrong_diagnostics() -> None:
    state, diagnostics = _rich_state_and_diagnostics()
    with pytest.raises(ValueError, match="source basis"):
        harmonic_ef_measurement.capture_rich_harmonic_ef_state_leaf_v2(
            state,
            replace(diagnostics, source_basis_id="wrong"),
            prepared_pes_configuration_sha256="9" * 64,
        )
    mismatched = replace(
        state,
        induced_source4=np.asarray([[0.0] * 4, [0.0] * 4]),
        root_sha256="",
    )
    with pytest.raises(ValueError, match="induced sources disagree"):
        harmonic_ef_measurement.capture_rich_harmonic_ef_state_leaf_v2(
            mismatched,
            diagnostics,
            prepared_pes_configuration_sha256="9" * 64,
        )
    with pytest.raises(ValueError, match="root_sha256"):
        replace(state, root_sha256="f" * 64)


@pytest.mark.parametrize(
    "mutation",
    (
        "cold-field",
        "cold-energy",
        "cold-residual",
        "cold-iterations",
        "wide-iterations",
        "replay-field",
        "replay-energy",
    ),
)
def test_capture_rich_v2_state_leaf_cross_binds_legacy_selection(mutation) -> None:
    state, diagnostics = _rich_state_and_diagnostics()
    cold = diagnostics.cold_start
    wide = diagnostics.wide_start
    if mutation == "cold-field":
        cold = replace(cold, final_native_field_ev=np.ones((2, 8)))
    elif mutation == "cold-energy":
        cold = replace(cold, final_polarization_energy_ev=-0.9)
    elif mutation == "cold-residual":
        residual = np.full((2, 8), 1.0e-12)
        cold = replace(
            cold,
            final_residual_ev=residual,
            final_residual_norm_ev=float(np.linalg.norm(residual)),
        )
    elif mutation == "cold-iterations":
        cold = replace(cold, iterations=2)
    elif mutation == "wide-iterations":
        wide = replace(wide, iterations=2)
    elif mutation == "replay-field":
        wide = replace(wide, final_native_field_ev=np.full((2, 8), 1.0e-12))
    elif mutation == "replay-energy":
        wide = replace(wide, final_polarization_energy_ev=-1.0 + 1.0e-12)
    else:
        raise AssertionError(mutation)
    tampered = replace(diagnostics, cold_start=cold, wide_start=wide)

    with pytest.raises(ValueError, match="legacy state and diagnostics"):
        harmonic_ef_measurement.capture_rich_harmonic_ef_state_leaf_v2(
            state,
            tampered,
            prepared_pes_configuration_sha256="9" * 64,
        )


class _SyntheticRichPES(MACE_MDPPolarHybridSmoothHarmonicPES):
    provider_id = "synthetic.prepared-harmonic-pes.v2"

    def __init__(
        self,
        *,
        atomic_numbers=(8, 1, 1),
        cavity_radii_angstrom=(1.5, 1.2, 1.2),
        configuration_sha256="9" * 64,
    ) -> None:
        self.solve_calls = 0
        self._synthetic_atomic_numbers = tuple(atomic_numbers)
        self._synthetic_cavity_radii = tuple(cavity_radii_angstrom)
        self._configuration_sha256 = configuration_sha256

    @property
    def prepared_atomic_numbers(self):
        return self._synthetic_atomic_numbers

    @property
    def prepared_cavity_radii_angstrom(self):
        return self._synthetic_cavity_radii

    @property
    def prepared_charge(self):
        return 0

    @property
    def prepared_multiplicity(self):
        return 1

    @property
    def prepared_profile_id(self):
        return "synthetic.truthful-v1.profile"

    @property
    def prepared_scalar_id(self):
        return "synthetic.truthful-v1.scalar"

    @property
    def prepared_provider_id(self):
        return self.provider_id

    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def prepared_configuration_sha256(self) -> str:
        return self.configuration_sha256()

    def solve_with_diagnostics(self, geometry):
        self.solve_calls += 1
        digest = (
            harmonic_ef_measurement.geometry_sha256(geometry)
            if hasattr(geometry, "positions")
            else str(geometry)
        )
        count = len(geometry) if hasattr(geometry, "positions") else 2
        return _rich_state_and_diagnostics(geometry_sha256=digest, atom_count=count)


def test_rich_state_recorder_deduplicates_storage_not_computation() -> None:
    pes = _SyntheticRichPES()
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(pes)

    first = recorder.solve("a" * 64)
    second = recorder.solve("a" * 64)
    serialized = recorder.serialize()

    assert pes.solve_calls == 2
    assert first.root_sha256 == second.root_sha256
    assert len(serialized["state_leaves"]) == 1
    only_leaf = next(iter(serialized["state_leaves"].values()))
    assert only_leaf["legacy_root_sha256"] == first.root_sha256
    assert only_leaf["state_leaf_sha256"] != first.root_sha256
    assert [event["event_index"] for event in serialized["solve_events"]] == [0, 1]
    assert (
        serialized["solve_events"][0]["state_leaf_sha256"]
        == serialized["solve_events"][1]["state_leaf_sha256"]
    )


def test_rich_state_recorder_tracks_geometry_changes_and_sample_occurrences() -> None:
    pes = _SyntheticRichPES()
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(pes)

    recorder.solve("a" * 64)
    sample = recorder.sample("b" * 64)
    serialized = recorder.serialize()

    assert pes.solve_calls == 2
    assert sample.energy_eV == -4.0
    assert len(serialized["state_leaves"]) == 2
    assert len(serialized["solve_events"]) == 2
    assert serialized["solve_events"][0]["state_leaf_sha256"] != (
        serialized["solve_events"][1]["state_leaf_sha256"]
    )
    assert [event["geometry_sha256"] for event in serialized["solve_events"]] == [
        "a" * 64,
        "b" * 64,
    ]


def test_rich_state_recorder_serialization_is_copy_safe_and_tamper_evident() -> None:
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(
        _SyntheticRichPES()
    )
    recorder.solve("a" * 64)
    exported = recorder.serialize()
    leaf_tamper = copy.deepcopy(exported)
    next(iter(leaf_tamper["state_leaves"].values()))["legacy_root_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="state_leaf_sha256"):
        harmonic_ef_measurement.validate_serialized_rich_state_recording_v2(leaf_tamper)
    event = exported["solve_events"][0]
    event["geometry_sha256"] = "b" * 64
    event["solve_event_sha256"] = harmonic_ef_measurement.canonical_json_sha256(
        {key: value for key, value in event.items() if key != "solve_event_sha256"}
    )

    with pytest.raises(ValueError, match="does not bind its state leaf"):
        harmonic_ef_measurement.validate_serialized_rich_state_recording_v2(exported)
    fresh = recorder.serialize()
    assert fresh["solve_events"][0]["geometry_sha256"] == "a" * 64


def test_rich_state_recording_rejects_unreferenced_extra_leaf() -> None:
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(
        _SyntheticRichPES()
    )
    recorder.solve("a" * 64)
    recorder.solve("b" * 64)
    serialized = recorder.serialize()
    serialized["solve_events"].pop()

    with pytest.raises(ValueError, match="exact solve-event referenced digest set"):
        harmonic_ef_measurement.validate_serialized_rich_state_recording_v2(serialized)


def _water_atoms() -> Atoms:
    return Atoms(
        numbers=[8, 1, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.239, 0.9266, 0.0],
        ],
        info={"charge": 0, "multiplicity": 1},
    )


def _prepared_rich_inputs() -> harmonic_ef_measurement.PreparedHarmonicEFInputsV2:
    force_panel_contract = {
        "gepol_regression_compound_id": "mobley_3053621",
        "gepol_regression_cartesian_dof": [0, 0],
        "water_geometry_angstrom": [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.239, 0.9266, 0.0],
        ],
        "water_full_cartesian_force": True,
        "independent_direction_seed": 20260816,
        "maximum_independent_directional_error_ev_per_angstrom": 5.0e-4,
        "maximum_benzene_h4_difference_ev_per_angstrom": 2.0e-4,
        "translation_angstrom": [4.2, -3.1, 1.7],
        "maximum_translation_energy_error_ev": 1.0e-7,
        "maximum_translation_force_relative_error": 2.0e-5,
        "maximum_translation_force_absolute_error_ev_per_angstrom": 5.0e-5,
        "maximum_net_force_norm_ev_per_angstrom": 1.0e-3,
        "rotation_seed": 20260817,
        "maximum_rotation_energy_error_ev": 1.0e-7,
        "maximum_rotation_force_relative_error": 2.0e-5,
        "maximum_rotation_force_absolute_error_ev_per_angstrom": 5.0e-5,
        "closed_loop_cartesian_dofs": [[0, 0], [1, 1]],
        "closed_loop_half_width_angstrom": 1.0e-3,
        "maximum_closed_loop_work_abs_ev": 2.0e-7,
    }
    return harmonic_ef_measurement.PreparedHarmonicEFInputsV2(
        benzene=harmonic_ef_measurement.PreparedHarmonicEFBenzeneInputV2(
            compound_id="mobley_3053621",
            name="benzene",
            atomic_numbers=(6,) * 12,
            positions_angstrom=((0.0, 0.0, 0.0),) * 12,
            charge=0,
            multiplicity=1,
            cavity_radii_angstrom=(1.7,) * 12,
            mol2_sha256="a" * 64,
            projection_result_sha256="b" * 64,
            predecessor_dof=(0, 0),
        ),
        water=harmonic_ef_measurement.PreparedHarmonicEFWaterInputV2(
            atomic_numbers=(8, 1, 1),
            positions_angstrom=(
                (0.0, 0.0, 0.0),
                (0.9572, 0.0, 0.0),
                (-0.239, 0.9266, 0.0),
            ),
            charge=0,
            multiplicity=1,
            cavity_radii_angstrom=(1.5, 1.2, 1.2),
        ),
        v1_preregistration_sha256="c" * 64,
        parent_panel_sha256="d" * 64,
        force_panel_contract=force_panel_contract,
        force_panel_contract_sha256=(
            harmonic_ef_measurement.canonical_json_sha256(force_panel_contract)
        ),
    )


def _prepared_synthetic_providers(prepared=None):
    prepared = _prepared_rich_inputs() if prepared is None else prepared
    benzene_base = _SyntheticRichPES(
        atomic_numbers=(6,) * 12,
        cavity_radii_angstrom=(1.7,) * 12,
        configuration_sha256="7" * 64,
    )
    water_base = _SyntheticRichPES(
        atomic_numbers=(8, 1, 1),
        cavity_radii_angstrom=(1.5, 1.2, 1.2),
        configuration_sha256="8" * 64,
    )
    return (
        harmonic_ef_measurement.H1PreparedHarmonicEFPESV2(
            benzene_base, prepared.benzene, system_role="benzene"
        ),
        harmonic_ef_measurement.H1PreparedHarmonicEFPESV2(
            water_base, prepared.water, system_role="water"
        ),
    )


def _prepared_provider_contract(benzene_pes=None, water_pes=None):
    force_panel_sha256 = _prepared_rich_inputs().force_panel_contract_sha256
    return {
        "profile_id": H1_V2_PROFILE_ID,
        "scalar_id": H1_V2_SCALAR_ID,
        "state_id": H1_V2_STATE_ID,
        "checkpoint_sha256s": {
            "mace_mdp_checkpoint": "b" * 64,
            "mace_polar_checkpoint": "c" * 64,
        },
        "provider_configuration_sha256s": {
            "hybrid_configuration": "1" * 64,
            "hybrid_provenance": "2" * 64,
            "continuum_configuration": "3" * 64,
            "continuum_provenance": "4" * 64,
            "cavity_configuration": "5" * 64,
            "prepared_input_manifest": "6" * 64,
            "force_panel_contract": force_panel_sha256,
            "benzene_pes_configuration": (
                "7" * 64 if benzene_pes is None else benzene_pes.configuration_sha256()
            ),
            "water_pes_configuration": (
                "8" * 64 if water_pes is None else water_pes.configuration_sha256()
            ),
        },
    }


def test_h1_prepared_adapter_preserves_truthful_v1_base_and_h1_identity() -> None:
    prepared = _prepared_rich_inputs()
    base = _SyntheticRichPES(
        atomic_numbers=(6,) * 12,
        cavity_radii_angstrom=(1.7,) * 12,
        configuration_sha256="7" * 64,
    )
    adapter = harmonic_ef_measurement.H1PreparedHarmonicEFPESV2(
        base, prepared.benzene, system_role="benzene"
    )

    assert base.prepared_profile_id == "synthetic.truthful-v1.profile"
    assert base.prepared_scalar_id == "synthetic.truthful-v1.scalar"
    assert adapter.prepared_profile_id == H1_V2_PROFILE_ID
    assert adapter.prepared_scalar_id == H1_V2_SCALAR_ID
    assert adapter.prepared_state_id == H1_V2_STATE_ID
    assert adapter.provider_id != base.provider_id
    assert adapter.system_role == "benzene"
    assert adapter.prepared_atomic_numbers == prepared.benzene.atomic_numbers
    assert adapter.prepared_cavity_radii_angstrom == (
        prepared.benzene.cavity_radii_angstrom
    )
    assert adapter.prepared_charge == 0
    assert adapter.prepared_multiplicity == 1
    assert adapter.configuration_sha256() == adapter.prepared_configuration_sha256
    assert adapter.configuration_sha256() != base.configuration_sha256()
    with pytest.raises(AttributeError):
        adapter._system_role = "water"
    with pytest.raises(TypeError, match="system_role"):
        harmonic_ef_measurement.H1PreparedHarmonicEFPESV2(
            base, prepared.benzene, system_role="water"
        )

    base._configuration_sha256 = "f" * 64
    with pytest.raises(RuntimeError, match="underlying v1 PES configuration changed"):
        adapter.configuration_sha256()


class _AdapterPermanent:
    provider_id = "test.adapter.permanent.v1"
    model_profile_id = "test.adapter.permanent-profile.v1"
    source_space = ATOMIC_L1_SOURCE_SPACE

    def configuration_sha256(self):
        return "1" * 64

    def evaluate_source(self, geometry):
        result = np.zeros((len(geometry), 4))
        result[0, 0] = -0.1
        result[-1, 0] = 0.1
        return result

    def source_position_vjp(self, geometry, source_cotangent):
        del source_cotangent
        return np.zeros((len(geometry), 3))


class _AdapterResponsive:
    provider_id = "test.adapter.response.v1"
    model_profile_id = "test.adapter.response-profile.v1"
    provenance_sha256 = "2" * 64
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    long_range_evaluator_profile = (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )

    def configuration_sha256(self):
        return "3" * 64

    def vacuum_energy_ev(self, geometry):
        del geometry
        return -1.0

    def vacuum_forces_ev_per_angstrom(self, geometry):
        return np.zeros((len(geometry), 3))

    def evaluate_source(self, geometry, field):
        del field
        return np.zeros((len(geometry), 4))

    def field_jvp(self, geometry, field, direction):
        del field, direction
        return np.zeros((len(geometry), 4))

    def field_vjp(self, geometry, field, cotangent):
        del field, cotangent
        return np.zeros((len(geometry), 8))

    def coordinate_vjp(self, geometry, field, source_cotangent):
        del field, source_cotangent
        return np.zeros((len(geometry), 3))


def test_real_v1_pes_passes_h1_adapter_identity_preflight_without_solve() -> None:
    prepared = _prepared_rich_inputs().water
    hybrid = PermanentAnchoredInducedSourceModel(
        _AdapterPermanent(), _AdapterResponsive()
    )
    base = MACE_MDPPolarHybridSmoothHarmonicPES(
        hybrid=hybrid,
        atomic_numbers=prepared.atomic_numbers,
        cavity_radii_angstrom=prepared.cavity_radii_angstrom,
        dtype=torch.float64,
        device="cpu",
    )

    adapter = harmonic_ef_measurement.H1PreparedHarmonicEFPESV2(
        base, prepared, system_role="water"
    )

    assert base.prepared_profile_id == base.profile_id
    assert base.prepared_scalar_id != H1_V2_SCALAR_ID
    assert adapter.prepared_profile_id == H1_V2_PROFILE_ID
    assert adapter.prepared_scalar_id == H1_V2_SCALAR_ID
    assert adapter.underlying_profile_id == base.profile_id
    assert adapter.underlying_scalar_id == base.prepared_scalar_id
    assert adapter.underlying_configuration_sha256 == base.configuration_sha256()
    assert adapter.configuration_sha256() == adapter.prepared_configuration_sha256


def test_run_rich_raw_measurement_builds_exact_i_o_free_science_tree() -> None:
    prepared = _prepared_rich_inputs()
    before = (
        prepared.benzene,
        prepared.water,
        copy.deepcopy(dict(prepared.force_panel_contract)),
        prepared.v1_preregistration_sha256,
        prepared.parent_panel_sha256,
        prepared.force_panel_contract_sha256,
    )
    benzene_pes, water_pes = _prepared_synthetic_providers()

    result = harmonic_ef_measurement.run_rich_harmonic_ef_raw_measurement_v2(
        prepared,
        benzene_pes,
        water_pes,
        prepared_provider_contract=_prepared_provider_contract(benzene_pes, water_pes),
    )

    assert set(result) == {"state_leaves", "solve_events", "systems"}
    assert len(result["solve_events"]) == 141
    assert result["systems"]["benzene"]["event_range"] == [0, 8]
    assert result["systems"]["water"]["event_range"] == [8, 141]
    assert (
        prepared.benzene,
        prepared.water,
        dict(prepared.force_panel_contract),
        prepared.v1_preregistration_sha256,
        prepared.parent_panel_sha256,
        prepared.force_panel_contract_sha256,
    ) == before
    assert harmonic_ef_measurement.RICH_HARMONIC_EF_V2_EXECUTION_ENABLED is True

    second_benzene, second_water = _prepared_synthetic_providers()
    repeated = harmonic_ef_measurement.run_rich_harmonic_ef_raw_measurement_v2(
        prepared,
        second_benzene,
        second_water,
        prepared_provider_contract=_prepared_provider_contract(
            second_benzene, second_water
        ),
    )
    assert repeated == result
    result["solve_events"][0]["geometry_sha256"] = "f" * 64
    assert repeated["solve_events"][0]["geometry_sha256"] != "f" * 64


def test_run_rich_raw_measurement_rejects_bad_provider_or_input_identity() -> None:
    prepared = _prepared_rich_inputs()
    benzene_pes, water_pes = _prepared_synthetic_providers(prepared)
    wrong_benzene_base = _SyntheticRichPES(
        atomic_numbers=(1,) * 12,
        cavity_radii_angstrom=(1.7,) * 12,
        configuration_sha256="7" * 64,
    )
    with pytest.raises(ValueError, match="atomic numbers"):
        harmonic_ef_measurement.H1PreparedHarmonicEFPESV2(
            wrong_benzene_base, prepared.benzene, system_role="benzene"
        )
    with pytest.raises(TypeError, match="H1PreparedHarmonicEFPESV2"):
        harmonic_ef_measurement.run_rich_harmonic_ef_raw_measurement_v2(
            prepared,
            wrong_benzene_base,
            water_pes,
            prepared_provider_contract=_prepared_provider_contract(
                benzene_pes, water_pes
            ),
        )
    with pytest.raises(ValueError, match="wrong system role"):
        harmonic_ef_measurement.run_rich_harmonic_ef_raw_measurement_v2(
            prepared,
            water_pes,
            water_pes,
            prepared_provider_contract=_prepared_provider_contract(
                benzene_pes, water_pes
            ),
        )
    with pytest.raises(TypeError, match="PreparedHarmonicEFInputsV2"):
        harmonic_ef_measurement.run_rich_harmonic_ef_raw_measurement_v2(
            object(),
            object(),
            object(),
            prepared_provider_contract=_prepared_provider_contract(),
        )

    good_benzene, good_water = _prepared_synthetic_providers(prepared)
    missing_configuration = _prepared_provider_contract(good_benzene, good_water)
    del missing_configuration["provider_configuration_sha256s"][
        "benzene_pes_configuration"
    ]
    with pytest.raises(ValueError, match="missing"):
        harmonic_ef_measurement.run_rich_harmonic_ef_raw_measurement_v2(
            prepared,
            good_benzene,
            good_water,
            prepared_provider_contract=missing_configuration,
        )


@pytest.fixture(scope="module")
def deterministic_v1_projection():
    prepared = _prepared_rich_inputs()
    benzene_pes, water_pes = _prepared_synthetic_providers(prepared)
    contract = _prepared_provider_contract(benzene_pes, water_pes)
    raw = harmonic_ef_measurement.run_rich_harmonic_ef_raw_measurement_v2(
        prepared,
        benzene_pes,
        water_pes,
        prepared_provider_contract=contract,
    )
    science = (
        harmonic_ef_measurement.project_rich_harmonic_ef_raw_tree_to_v1_science_v2(
            raw, prepared, contract
        )
    )
    return raw, prepared, contract, science


def test_archive_v1_projection_matches_complete_golden_and_digest(
    deterministic_v1_projection,
) -> None:
    _, _, _, science = deterministic_v1_projection
    golden = json.loads(V1_PROJECTION_GOLDEN.read_text(encoding="utf-8"))
    assert science == golden["science"]
    assert harmonic_ef_measurement.canonical_json_sha256(science) == golden["sha256"]
    assert harmonic_ef_measurement.HARMONIC_EF_V1_PROJECTION_SCOPE == (
        "archive-only-v1-shape-adapter-compatibility-non-admitting"
    )
    assert golden["scope"] == harmonic_ef_measurement.HARMONIC_EF_V1_PROJECTION_SCOPE


def test_archive_v1_projection_uses_event_order_sign_and_legacy_roots(
    deterministic_v1_projection,
) -> None:
    raw, _, _, science = deterministic_v1_projection
    events = raw["solve_events"]
    leaves = raw["state_leaves"]
    benzene = science["benzene_gepol_regression"]
    convergence = benzene["force_convergence"]
    assert (
        convergence["center_state_sha256"]
        == leaves[events[0]["state_leaf_sha256"]]["legacy_root_sha256"]
    )
    assert (
        convergence["displaced_state_sha256"]["plus_h"]
        == leaves[events[1]["state_leaf_sha256"]]["legacy_root_sha256"]
    )
    assert (
        convergence["displaced_state_sha256"]["minus_h"]
        == leaves[events[2]["state_leaf_sha256"]]["legacy_root_sha256"]
    )
    assert (
        convergence["richardson_force_eV_per_A"]
        == -convergence["richardson_energy_derivative_eV_per_A"]
    )
    assert (
        benzene["center_root_sha256"]
        == leaves[events[7]["state_leaf_sha256"]]["legacy_root_sha256"]
    )


def test_archive_v1_projection_rejects_tampered_raw_tree(
    deterministic_v1_projection,
) -> None:
    raw, prepared, contract, _ = deterministic_v1_projection
    tampered = copy.deepcopy(raw)
    tampered["systems"]["water"]["directional"]["plus_event_index"] = 0
    with pytest.raises(ValueError):
        harmonic_ef_measurement.project_rich_harmonic_ef_raw_tree_to_v1_science_v2(
            tampered, prepared, contract
        )


def test_actual_historical_v1_science_object_and_digest_remain_pinned() -> None:
    replicate = json.loads(V1_REPLICATE.read_text(encoding="utf-8"))
    historical = {
        key: replicate[key]
        for key in (
            "benzene_gepol_regression",
            "water_force_symmetry_loop",
            "aggregate",
        )
    }
    assert harmonic_ef_measurement.canonical_json_sha256(historical) == (
        "204346766f9454c3e86629a5ddc3ee65f4fa146c129e606fbd1e531f3b2a0c3b"
    )
    harmonic_ef_measurement.validate_harmonic_ef_science_measurement(historical)
    assert "no rich H0 preimage can be reconstructed" in (
        harmonic_ef_measurement.HARMONIC_EF_H0_NON_IDENTIFIABILITY
    )


def _displaced_hash(atoms, atom, axis, displacement) -> str:
    moved = atoms.copy()
    moved.positions[atom, axis] += displacement
    return harmonic_ef_measurement.geometry_sha256(moved)


def test_record_component_stencil_uses_explicit_center_and_exact_solve_order() -> None:
    pes = _SyntheticRichPES()
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(pes)
    atoms = _water_atoms()
    recorder.solve(atoms)

    mapping = harmonic_ef_measurement.record_component_stencil_v2(
        recorder,
        atoms,
        center_event_index=0,
        atom_index=1,
        axis_index=2,
        coarse_step_angstrom=5.0e-4,
    )
    serialized = recorder.serialize()

    assert pes.solve_calls == 5
    assert mapping == {
        "atom_index": 1,
        "axis_index": 2,
        "coarse_step_angstrom": 5.0e-4,
        "fine_step_angstrom": 2.5e-4,
        "center_event_index": 0,
        "plus_h_event_index": 1,
        "minus_h_event_index": 2,
        "plus_h2_event_index": 3,
        "minus_h2_event_index": 4,
    }
    assert [event["geometry_sha256"] for event in serialized["solve_events"]] == [
        harmonic_ef_measurement.geometry_sha256(atoms),
        _displaced_hash(atoms, 1, 2, 5.0e-4),
        _displaced_hash(atoms, 1, 2, -5.0e-4),
        _displaced_hash(atoms, 1, 2, 2.5e-4),
        _displaced_hash(atoms, 1, 2, -2.5e-4),
    ]


def test_record_cartesian_panel_has_one_center_and_atom_major_components() -> None:
    pes = _SyntheticRichPES()
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(pes)
    atoms = _water_atoms()

    mapping = harmonic_ef_measurement.record_cartesian_panel_v2(
        recorder,
        atoms,
        label="base",
        coarse_step_angstrom=5.0e-4,
    )
    serialized = recorder.serialize()

    assert pes.solve_calls == 37
    assert len(serialized["solve_events"]) == 37
    assert len(mapping["components"]) == 9
    assert [
        (component["atom_index"], component["axis_index"])
        for component in mapping["components"]
    ] == [divmod(index, 3) for index in range(9)]
    assert {component["center_event_index"] for component in mapping["components"]} == {
        0
    }
    assert [
        tuple(
            component[key]
            for key in (
                "plus_h_event_index",
                "minus_h_event_index",
                "plus_h2_event_index",
                "minus_h2_event_index",
            )
        )
        for component in mapping["components"]
    ] == [tuple(range(1 + 4 * index, 5 + 4 * index)) for index in range(9)]
    expected_geometry_hashes = [harmonic_ef_measurement.geometry_sha256(atoms)]
    for atom_index in range(3):
        for axis_index in range(3):
            expected_geometry_hashes.extend(
                _displaced_hash(atoms, atom_index, axis_index, displacement)
                for displacement in (5.0e-4, -5.0e-4, 2.5e-4, -2.5e-4)
            )
    assert [
        event["geometry_sha256"] for event in serialized["solve_events"]
    ] == expected_geometry_hashes


def test_record_benzene_predecessor_appends_exact_audit_only_h4_order() -> None:
    pes = _SyntheticRichPES()
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(pes)
    atoms = _water_atoms()
    recorder.solve(atoms)

    mapping = harmonic_ef_measurement.record_benzene_predecessor_stencil_v2(
        recorder,
        atoms,
        center_event_index=0,
        atom_index=0,
        axis_index=0,
        coarse_step_angstrom=5.0e-4,
    )
    serialized = recorder.serialize()

    assert pes.solve_calls == 7
    assert len(serialized["solve_events"]) == 7
    assert mapping["plus_h4_event_index"] == 5
    assert mapping["minus_h4_event_index"] == 6
    assert mapping["independent_step_angstrom"] == 1.25e-4
    assert mapping["audit_only"] is True
    assert [event["geometry_sha256"] for event in serialized["solve_events"][-2:]] == [
        _displaced_hash(atoms, 0, 0, 1.25e-4),
        _displaced_hash(atoms, 0, 0, -1.25e-4),
    ]


def test_record_water_directional_uses_frozen_seed_and_plus_minus_order() -> None:
    pes = _SyntheticRichPES()
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(pes)
    atoms = _water_atoms()

    mapping = harmonic_ef_measurement.record_water_directional_v2(
        recorder, atoms, independent_step_angstrom=1.25e-4
    )
    serialized = recorder.serialize()
    expected = np.random.default_rng(20260816).normal(size=(3, 3))
    expected /= np.linalg.norm(expected)

    assert pes.solve_calls == 2
    assert mapping["seed"] == 20260816
    assert mapping["step_angstrom"] == 1.25e-4
    assert mapping["plus_event_index"] == 0
    assert mapping["minus_event_index"] == 1
    assert np.array_equal(np.asarray(mapping["normalized_direction"]), expected)
    assert [event["geometry_sha256"] for event in serialized["solve_events"]] == [
        harmonic_ef_measurement.geometry_sha256(
            Atoms(
                numbers=atoms.numbers,
                positions=atoms.positions + 1.25e-4 * expected,
                info=atoms.info,
            )
        ),
        harmonic_ef_measurement.geometry_sha256(
            Atoms(
                numbers=atoms.numbers,
                positions=atoms.positions - 1.25e-4 * expected,
                info=atoms.info,
            )
        ),
    ]
    mapping["normalized_direction"][0][0] = 99.0
    assert recorder.serialize()["solve_events"] == serialized["solve_events"]


def test_record_translated_panel_uses_exact_vector_and_event_graph() -> None:
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(
        _SyntheticRichPES()
    )
    atoms = _water_atoms()

    mapping = harmonic_ef_measurement.record_translated_cartesian_panel_v2(
        recorder, atoms, coarse_step_angstrom=5.0e-4
    )
    serialized = recorder.serialize()
    translated = atoms.copy()
    translated.positions += np.asarray([4.2, -3.1, 1.7])

    assert mapping["translation_angstrom"] == [4.2, -3.1, 1.7]
    assert mapping["panel"]["label"] == "translated"
    assert len(serialized["solve_events"]) == 37
    assert serialized["solve_events"][0]["geometry_sha256"] == (
        harmonic_ef_measurement.geometry_sha256(translated)
    )
    assert {item["center_event_index"] for item in mapping["panel"]["components"]} == {
        0
    }


def test_record_rotated_panel_uses_frozen_proper_rotation() -> None:
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(
        _SyntheticRichPES()
    )
    atoms = _water_atoms()

    mapping = harmonic_ef_measurement.record_rotated_cartesian_panel_v2(
        recorder, atoms, coarse_step_angstrom=5.0e-4
    )
    serialized = recorder.serialize()
    rotation = harmonic_ef_measurement._proper_rotation(20260817)
    centroid = np.mean(atoms.positions, axis=0)
    rotated = atoms.copy()
    rotated.positions = (atoms.positions - centroid) @ rotation.T + centroid

    assert mapping["seed"] == 20260817
    assert np.array_equal(np.asarray(mapping["rotation_matrix"]), rotation)
    assert np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=2.0e-15)
    assert np.linalg.det(rotation) > 0.0
    assert mapping["panel"]["label"] == "rotated"
    assert len(serialized["solve_events"]) == 37
    assert serialized["solve_events"][0]["geometry_sha256"] == (
        harmonic_ef_measurement.geometry_sha256(rotated)
    )


def test_record_closed_loop_has_exact_edges_components_and_geometry_order() -> None:
    recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(
        _SyntheticRichPES()
    )
    atoms = _water_atoms()

    mapping = harmonic_ef_measurement.record_closed_loop_v2(
        recorder, atoms, coarse_step_angstrom=5.0e-4
    )
    serialized = recorder.serialize()

    assert mapping["cartesian_dofs"] == [[0, 0], [1, 1]]
    assert mapping["half_width_angstrom"] == 1.0e-3
    assert mapping["orientation"] == "counterclockwise"
    assert [edge["label"] for edge in mapping["edges"]] == [
        "bottom",
        "right",
        "top",
        "left",
    ]
    assert [edge["midpoint_offsets_angstrom"] for edge in mapping["edges"]] == [
        [0.0, -1.0e-3],
        [1.0e-3, 0.0],
        [0.0, 1.0e-3],
        [-1.0e-3, 0.0],
    ]
    assert [edge["displacement_angstrom"] for edge in mapping["edges"]] == [
        2.0e-3,
        2.0e-3,
        -2.0e-3,
        -2.0e-3,
    ]
    assert len(serialized["solve_events"]) == 20
    assert [edge["component"]["center_event_index"] for edge in mapping["edges"]] == [
        0,
        5,
        10,
        15,
    ]
    expected_center_hashes = []
    expected_event_hashes = []
    for (x, y), (atom, axis) in zip(
        ((0.0, -1.0e-3), (1.0e-3, 0.0), (0.0, 1.0e-3), (-1.0e-3, 0.0)),
        ((0, 0), (1, 1), (0, 0), (1, 1)),
        strict=True,
    ):
        midpoint = atoms.copy()
        midpoint.positions[0, 0] += x
        midpoint.positions[1, 1] += y
        center_hash = harmonic_ef_measurement.geometry_sha256(midpoint)
        expected_center_hashes.append(center_hash)
        expected_event_hashes.append(center_hash)
        expected_event_hashes.extend(
            _displaced_hash(midpoint, atom, axis, displacement)
            for displacement in (5.0e-4, -5.0e-4, 2.5e-4, -2.5e-4)
        )
    assert [
        serialized["solve_events"][index]["geometry_sha256"] for index in (0, 5, 10, 15)
    ] == expected_center_hashes
    assert [
        event["geometry_sha256"] for event in serialized["solve_events"]
    ] == expected_event_hashes


@pytest.fixture(scope="module")
def complete_local_rich_recordings():
    atoms = _water_atoms()
    benzene_recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(
        _SyntheticRichPES()
    )
    benzene_system = harmonic_ef_measurement.record_benzene_system_v2(
        benzene_recorder,
        atoms,
        atom_index=0,
        axis_index=0,
        coarse_step_angstrom=5.0e-4,
    )

    water_recorder = harmonic_ef_measurement.RichHarmonicEFStateRecorderV2(
        _SyntheticRichPES()
    )
    water_system = harmonic_ef_measurement.record_water_system_v2(
        water_recorder,
        atoms,
        coarse_step_angstrom=5.0e-4,
        independent_step_angstrom=1.25e-4,
    )
    return {
        "benzene_recording": benzene_recorder.serialize(),
        "water_recording": water_recorder.serialize(),
        "benzene_system": benzene_system,
        "water_system": water_system,
        "positions": tuple(
            tuple(float(item) for item in row) for row in atoms.positions
        ),
    }


def _merge_complete_recordings(inputs):
    return harmonic_ef_measurement.merge_rich_harmonic_ef_recordings_v2(
        inputs["benzene_recording"],
        inputs["water_recording"],
        benzene_system=inputs["benzene_system"],
        water_system=inputs["water_system"],
        prepared_water_positions=inputs["positions"],
    )


def test_merge_rich_recordings_rebuilds_order_digests_and_known_indices(
    complete_local_rich_recordings,
) -> None:
    inputs = complete_local_rich_recordings
    merged = _merge_complete_recordings(inputs)

    assert len(inputs["benzene_recording"]["solve_events"]) == 8
    assert len(inputs["water_recording"]["solve_events"]) == 133
    assert inputs["benzene_system"]["event_range"] == [0, 8]
    assert inputs["benzene_system"]["reported_center_event_index"] == 7
    assert inputs["water_system"]["event_range"] == [0, 133]
    assert inputs["benzene_recording"]["solve_events"][0]["event_index"] == 0
    assert inputs["water_recording"]["solve_events"][0]["event_index"] == 0
    assert (
        inputs["benzene_recording"]["solve_events"][0]["state_leaf_sha256"]
        == inputs["benzene_recording"]["solve_events"][7]["state_leaf_sha256"]
    )
    assert (
        inputs["benzene_recording"]["solve_events"][0]["solve_event_sha256"]
        != inputs["benzene_recording"]["solve_events"][7]["solve_event_sha256"]
    )
    assert set(inputs["benzene_recording"]["state_leaves"]) & set(
        inputs["water_recording"]["state_leaves"]
    )
    assert len(merged["solve_events"]) == 141
    assert [event["event_index"] for event in merged["solve_events"]] == list(
        range(141)
    )
    assert merged["solve_events"][0]["geometry_sha256"] == (
        inputs["benzene_recording"]["solve_events"][0]["geometry_sha256"]
    )
    assert merged["solve_events"][8]["geometry_sha256"] == (
        inputs["water_recording"]["solve_events"][0]["geometry_sha256"]
    )
    assert merged["solve_events"][8]["solve_event_sha256"] != (
        inputs["water_recording"]["solve_events"][0]["solve_event_sha256"]
    )
    assert len(merged["state_leaves"]) < (
        len(inputs["benzene_recording"]["state_leaves"])
        + len(inputs["water_recording"]["state_leaves"])
    )
    assert merged["systems"]["benzene"] == inputs["benzene_system"]
    assert merged["systems"]["benzene"]["event_range"] == [0, 8]
    assert merged["systems"]["benzene"]["reported_center_event_index"] == 7
    water = merged["systems"]["water"]
    assert water["event_range"] == [8, 141]
    assert water["base_panel"]["components"][0]["center_event_index"] == 8
    assert water["directional"]["plus_event_index"] == 45
    assert water["directional"]["minus_event_index"] == 46
    assert water["translation"]["panel"]["components"][0]["center_event_index"] == 47
    assert water["rotation"]["panel"]["components"][0]["center_event_index"] == 84
    assert [
        edge["component"]["center_event_index"]
        for edge in water["closed_loop"]["edges"]
    ] == [121, 126, 131, 136]


def test_merge_rich_recordings_is_copy_safe_and_fails_on_tamper_or_unknown_shape(
    complete_local_rich_recordings,
) -> None:
    inputs = complete_local_rich_recordings
    merged = _merge_complete_recordings(inputs)
    merged["systems"]["water"]["directional"]["plus_event_index"] = 0
    fresh = _merge_complete_recordings(inputs)
    assert fresh["systems"]["water"]["directional"]["plus_event_index"] == 45

    tampered = copy.deepcopy(inputs["water_recording"])
    event = tampered["solve_events"][0]
    event["geometry_sha256"] = "f" * 64
    event["solve_event_sha256"] = harmonic_ef_measurement.canonical_json_sha256(
        {key: value for key, value in event.items() if key != "solve_event_sha256"}
    )
    with pytest.raises(ValueError, match="does not bind its state leaf"):
        harmonic_ef_measurement.merge_rich_harmonic_ef_recordings_v2(
            inputs["benzene_recording"],
            tampered,
            benzene_system=inputs["benzene_system"],
            water_system=inputs["water_system"],
            prepared_water_positions=inputs["positions"],
        )

    unknown = {**inputs["water_system"], "unknown": {}}
    with pytest.raises(ValueError, match="unknown"):
        harmonic_ef_measurement.merge_rich_harmonic_ef_recordings_v2(
            inputs["benzene_recording"],
            inputs["water_recording"],
            benzene_system=inputs["benzene_system"],
            water_system=unknown,
            prepared_water_positions=inputs["positions"],
        )

    extra_leaf_recording = copy.deepcopy(inputs["benzene_recording"])
    extra_digest = next(
        digest
        for digest in inputs["water_recording"]["state_leaves"]
        if digest not in extra_leaf_recording["state_leaves"]
    )
    extra_leaf_recording["state_leaves"][extra_digest] = inputs["water_recording"][
        "state_leaves"
    ][extra_digest]
    with pytest.raises(ValueError, match="exact solve-event referenced digest set"):
        harmonic_ef_measurement.merge_rich_harmonic_ef_recordings_v2(
            extra_leaf_recording,
            inputs["water_recording"],
            benzene_system=inputs["benzene_system"],
            water_system=inputs["water_system"],
            prepared_water_positions=inputs["positions"],
        )


@pytest.mark.parametrize(
    "mutation",
    ("water-geometry", "water-charge", "benzene-dof", "negative-radius"),
)
def test_rich_v2_prepared_inputs_fail_closed_on_identity_drift(mutation) -> None:
    water_positions = [
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.239, 0.9266, 0.0],
    ]
    if mutation == "water-geometry":
        water_positions[0][0] = 1.0
    if mutation in ("water-geometry", "water-charge", "negative-radius"):
        radii = (1.5, -1.2, 1.2) if mutation == "negative-radius" else (1.5, 1.2, 1.2)
        with pytest.raises(ValueError):
            harmonic_ef_measurement.PreparedHarmonicEFWaterInputV2(
                atomic_numbers=(8, 1, 1),
                positions_angstrom=tuple(tuple(row) for row in water_positions),
                charge=1 if mutation == "water-charge" else 0,
                multiplicity=1,
                cavity_radii_angstrom=radii,
            )
    else:
        with pytest.raises(ValueError, match="predecessor_dof"):
            harmonic_ef_measurement.PreparedHarmonicEFBenzeneInputV2(
                compound_id="mobley_3053621",
                name="benzene",
                atomic_numbers=(6,) * 12,
                positions_angstrom=((0.0, 0.0, 0.0),) * 12,
                charge=0,
                multiplicity=1,
                cavity_radii_angstrom=(1.7,) * 12,
                mol2_sha256="a" * 64,
                projection_result_sha256="b" * 64,
                predecessor_dof=(1, 0),
            )
