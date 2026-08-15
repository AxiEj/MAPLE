from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from maple.solvation.release.solvation_accuracy import (
    SOLVATION_ACCURACY_MAE_THRESHOLD_KCAL_PER_MOL,
    BlindSolvationAccuracyProtocol,
    BlindSolvationAccuracyRecord,
    assess_blind_solvation_accuracy,
)


def _case_ids() -> tuple[str, ...]:
    return tuple(f"blind-{index:03d}" for index in range(150))


def _protocol() -> BlindSolvationAccuracyProtocol:
    return BlindSolvationAccuracyProtocol(
        protocol_id="route2-pure-mace-polar-neutral-water-blind-v1",
        dataset_id="test-neutral-water-panel-v1",
        dataset_sha256=hashlib.sha256(b"dataset").hexdigest(),
        profile_configuration_sha256=hashlib.sha256(b"profile").hexdigest(),
        expected_case_ids=_case_ids(),
    )


def _records(absolute_error: float) -> tuple[BlindSolvationAccuracyRecord, ...]:
    return tuple(
        BlindSolvationAccuracyRecord(
            case_id=case_id,
            molecule_group_id=f"molecule-{index:03d}",
            solvent_id="water",
            reference_kcal_per_mol=-5.0,
            predicted_kcal_per_mol=-5.0
            + (absolute_error if index % 2 else -absolute_error),
        )
        for index, case_id in enumerate(_case_ids())
    )


def test_blind_gate_freezes_1p5_mae_and_one_sided_confidence_bound():
    protocol = _protocol()
    passed = assess_blind_solvation_accuracy(protocol, _records(1.0))
    assert passed.record_count == passed.molecule_group_count == 150
    assert passed.mae_threshold_kcal_per_mol == (
        SOLVATION_ACCURACY_MAE_THRESHOLD_KCAL_PER_MOL
    )
    assert passed.mae_kcal_per_mol == pytest.approx(1.0)
    assert passed.one_sided_mae_ucb_kcal_per_mol == pytest.approx(1.0)
    assert passed.point_mae_gate_passed is True
    assert passed.confidence_bound_gate_passed is True
    assert passed.gate_passed is True

    failed = assess_blind_solvation_accuracy(protocol, _records(1.6))
    assert failed.mae_kcal_per_mol == pytest.approx(1.6)
    assert failed.gate_passed is False


def test_protocol_rejects_smaller_panels_or_posthoc_threshold_changes():
    with pytest.raises(ValueError, match="at least 150"):
        BlindSolvationAccuracyProtocol(
            protocol_id="too-small",
            dataset_id="dataset",
            dataset_sha256="a" * 64,
            profile_configuration_sha256="b" * 64,
            expected_case_ids=_case_ids()[:-1],
        )
    with pytest.raises(ValueError, match="must remain 1.5"):
        replace(_protocol(), mae_threshold_kcal_per_mol=1.6)


def test_assessment_rejects_missing_fake_or_electrostatic_only_rows():
    protocol = _protocol()
    with pytest.raises(ValueError, match="exactly cover"):
        assess_blind_solvation_accuracy(protocol, _records(1.0)[:-1])

    fake = list(_records(1.0))
    fake[0] = replace(fake[0], backend_kind="synthetic")
    with pytest.raises(ValueError, match="real backend"):
        assess_blind_solvation_accuracy(protocol, tuple(fake))

    electrostatic = list(_records(1.0))
    electrostatic[0] = replace(
        electrostatic[0], target_kind="matched-electrostatic-component"
    )
    with pytest.raises(ValueError, match="electrostatic-only"):
        assess_blind_solvation_accuracy(protocol, tuple(electrostatic))


def test_assessment_rejects_duplicate_molecule_groups_and_incomplete_rows():
    protocol = _protocol()
    duplicate = list(_records(1.0))
    duplicate[1] = replace(
        duplicate[1], molecule_group_id=duplicate[0].molecule_group_id
    )
    with pytest.raises(ValueError, match="one record per molecule"):
        assess_blind_solvation_accuracy(protocol, tuple(duplicate))

    incomplete = list(_records(1.0))
    incomplete[0] = replace(incomplete[0], completed=False)
    with pytest.raises(ValueError, match="must be completed"):
        assess_blind_solvation_accuracy(protocol, tuple(incomplete))
