from __future__ import annotations

import pytest

from maple.function.dispatcher.solvfe.packing_contract import (
    PackingBiasState,
    PackingSchedule,
)


_HASHES = {
    "membership_definition_hash": "1" * 64,
    "observation_volume_hash": "2" * 64,
    "boundary_adapter_hash": "7" * 64,
    "solute_measure_hash": "3" * 64,
    "water_hamiltonian_hash": "4" * 64,
    "oxygen_atom_map_hash": "5" * 64,
    "solute_atom_map_hash": "6" * 64,
    "cell_hash": "7" * 64,
}


def _schedule(
    *,
    states=(
        PackingBiasState("unbiased", 0.0),
        PackingBiasState("half-field", 0.5),
        PackingBiasState("full-field", 1.0),
    ),
    target_state_index=0,
    full_field_state_index=2,
    **overrides,
) -> PackingSchedule:
    values = {
        "states": states,
        "target_state_index": target_state_index,
        "full_field_state_index": full_field_state_index,
        "conditioning_measure_id": "pure-water-soft-cutoff-v3",
        "temperature_k": 298.15,
        "ensemble": "NVT",
        "pressure_bar": None,
        "boundary_conditions": "periodic-3d",
        "active_occupancy_max": 1,
        **_HASHES,
    }
    values.update(overrides)
    return PackingSchedule(**values)


def test_packing_schedule_binds_explicit_target_and_full_field_states():
    schedule = _schedule()

    assert schedule.target_state.label == "unbiased"
    assert schedule.target_state.bias_scale == 0.0
    assert schedule.full_field_state.label == "full-field"
    assert schedule.full_field_state.bias_scale == 1.0
    assert schedule.row_labels == (
        "unbiased",
        "half-field",
        "full-field",
    )
    assert schedule.content_hash


def test_packing_schedule_does_not_assume_target_is_row_zero():
    schedule = _schedule(
        states=(
            PackingBiasState("full-field", 1.0),
            PackingBiasState("half-field", 0.5),
            PackingBiasState("unbiased", 0.0),
        ),
        target_state_index=2,
        full_field_state_index=0,
    )

    assert schedule.target_state.label == "unbiased"
    assert schedule.full_field_state.label == "full-field"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "states": (
                    PackingBiasState("unbiased", 0.0),
                    PackingBiasState("full-field", 0.9),
                ),
                "target_state_index": 0,
                "full_field_state_index": 1,
            },
            "full-field",
        ),
        (
            {
                "states": (
                    PackingBiasState("unbiased", 0.0),
                    PackingBiasState("middle", 0.5),
                    PackingBiasState("backward", 0.25),
                    PackingBiasState("full-field", 1.0),
                ),
                "target_state_index": 0,
                "full_field_state_index": 3,
            },
            "monotonic",
        ),
        (
            {"membership_definition_hash": "not-a-hash"},
            "membership_definition_hash",
        ),
        (
            {"boundary_conditions": "nonperiodic"},
            "periodic-3d",
        ),
        (
            {"ensemble": "NPT", "pressure_bar": None},
            "require NVT",
        ),
        (
            {"ensemble": "NPT", "pressure_bar": 1.0},
            "require NVT",
        ),
        (
            {"ensemble": "NVT", "pressure_bar": 1.0},
            "pressure_bar",
        ),
        (
            {"active_occupancy_max": 0},
            "include n=0 and at least n=1",
        ),
    ],
)
def test_packing_schedule_rejects_ambiguous_or_inconsistent_contracts(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _schedule(**overrides)


def test_packing_schedule_hash_changes_with_scientific_measure():
    first = _schedule()
    changed_conformer = _schedule(solute_measure_hash="7" * 64)
    changed_hamiltonian = _schedule(water_hamiltonian_hash="8" * 64)

    assert first.content_hash != changed_conformer.content_hash
    assert first.content_hash != changed_hamiltonian.content_hash
