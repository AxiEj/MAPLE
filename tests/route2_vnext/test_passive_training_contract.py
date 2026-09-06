from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from maple.solvation.models.base import ModelDomain
from maple.solvation.models.passive_training import (
    PASSIVE_QUADRATIC_CONSTRUCTION_ID,
    PASSIVE_TRAINING_PREREGISTRATION_VERSION,
    PASSIVE_TRAINING_RUN_BINDING_VERSION,
    PassiveTrainingDataSplit,
    PassiveTrainingDomain,
    PassiveTrainingPreregistration,
    PassiveTrainingRunBinding,
    PassiveTrainingTarget,
    bind_passive_training_run,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _split(role: str, index: int) -> PassiveTrainingDataSplit:
    return PassiveTrainingDataSplit(
        split_id=f"route2-passive-{role}-split-v1",
        role=role,
        index_artifact_sha256=_digest(f"{role}-index"),
        record_sha256s=tuple(
            sorted(_digest(f"{role}-record-{item}") for item in range(index, index + 3))
        ),
        molecule_group_sha256s=tuple(
            sorted(
                _digest(f"{role}-molecule-{item}") for item in range(index, index + 2)
            )
        ),
    )


def _targets() -> tuple[PassiveTrainingTarget, ...]:
    return (
        PassiveTrainingTarget(
            target_id="qm-external-field-energy-v1",
            kind="external_field_energy",
            unit="eV",
            reference_method_id="pinned-qm-finite-field-v1",
            reference_protocol_sha256=_digest("field-energy-protocol"),
            objective_weight=1.0,
            use_for_training=True,
            use_for_model_selection=False,
            use_for_admission=True,
        ),
        PassiveTrainingTarget(
            target_id="qm-cavity-surface-mep-v1",
            kind="cavity_surface_mep",
            unit="Hartree/e",
            reference_method_id="matched-qm-cavity-mep-v1",
            reference_protocol_sha256=_digest("surface-mep-protocol"),
            objective_weight=0.0,
            use_for_training=False,
            use_for_model_selection=True,
            use_for_admission=True,
        ),
        PassiveTrainingTarget(
            target_id="matched-fixed-source-pcm-v1",
            kind="fixed_source_pcm_energy",
            unit="eV",
            reference_method_id="matched-pcmsolver-electrostatic-v1",
            reference_protocol_sha256=_digest("fixed-source-pcm-protocol"),
            objective_weight=0.0,
            use_for_training=False,
            use_for_model_selection=False,
            use_for_admission=True,
        ),
    )


def _preregistration() -> PassiveTrainingPreregistration:
    return PassiveTrainingPreregistration(
        preregistration_id="route2-passive-organic-field-head-prereg-v1",
        head_provider_id="test.route2.passive-trained-head.v1",
        model_profile_id="route2-model-passive-organic-field-head-v1",
        coupling_id="route2-coupling-radial-gto-v1",
        duality_map_sha256=_digest("duality-map"),
        domain=PassiveTrainingDomain(
            model_domain=ModelDomain((1, 6, 7, 8), (-1, 1), (1, 2)),
            maximum_atom_count=64,
            minimum_interatomic_distance_angstrom=0.65,
            maximum_reduced_field_norm=0.25,
            coordinate_frame_policy="laboratory Cartesian Angstrom",
            dtype="float64",
        ),
        data_splits=(_split("train", 0), _split("validation", 10), _split("blind", 20)),
        targets=_targets(),
        training_code_sha256=_digest("training-code"),
        inference_code_sha256=_digest("inference-code"),
        optimizer_protocol_sha256=_digest("optimizer-protocol"),
        source_gate_protocol_sha256=_digest("source-gate-protocol"),
        response_gate_protocol_sha256=_digest("response-gate-protocol"),
        root_gate_protocol_sha256=_digest("root-gate-protocol"),
        random_seeds=(17, 29),
    )


def _run(preregistration: PassiveTrainingPreregistration) -> PassiveTrainingRunBinding:
    return bind_passive_training_run(
        preregistration,
        training_run_id="route2-passive-training-run-seed17-v1",
        checkpoint_sha256=_digest("checkpoint"),
        parameter_state_sha256=_digest("parameter-state"),
        optimizer_state_sha256=_digest("optimizer-state"),
        optimizer_parameter_group_audit_sha256=_digest("optimizer-audit"),
        dataset_access_audit_sha256=_digest("dataset-access-audit"),
        runtime_manifest_sha256=_digest("runtime-manifest"),
        training_log_sha256=_digest("training-log"),
        random_seed=17,
    )


def test_preregistration_is_immutable_content_addressed_and_capability_neutral() -> (
    None
):
    preregistration = _preregistration()
    payload = preregistration.as_dict()

    assert preregistration.contract_version == (
        PASSIVE_TRAINING_PREREGISTRATION_VERSION
    )
    assert preregistration.scalar_construction_id == (PASSIVE_QUADRATIC_CONSTRUCTION_ID)
    assert len(preregistration.content_sha256) == 64
    assert payload["content_sha256"] == preregistration.content_sha256
    assert payload["capability_admitted"] is False
    assert payload["experimental_solvation_labels_used"] is False
    assert payload["blind_split_used_for_model_selection"] is False
    json.dumps(payload, sort_keys=True)

    reordered = replace(
        preregistration,
        data_splits=tuple(reversed(preregistration.data_splits)),
        targets=tuple(reversed(preregistration.targets)),
    )
    assert reordered.content_sha256 == preregistration.content_sha256


def test_partitions_fail_closed_on_record_or_molecule_family_leakage() -> None:
    preregistration = _preregistration()
    train, validation, blind = preregistration.data_splits

    leaked_record = replace(
        blind,
        record_sha256s=tuple(
            sorted((train.record_sha256s[0], *blind.record_sha256s[1:]))
        ),
    )
    with pytest.raises(ValueError, match="record_sha256s overlap"):
        replace(preregistration, data_splits=(train, validation, leaked_record))

    leaked_family = replace(
        blind,
        molecule_group_sha256s=tuple(
            sorted((train.molecule_group_sha256s[0], blind.molecule_group_sha256s[1]))
        ),
    )
    with pytest.raises(ValueError, match="molecule_group_sha256s overlap"):
        replace(preregistration, data_splits=(train, validation, leaked_family))


def test_targets_reject_experimental_total_solvation_and_pcm_fitting() -> None:
    base = _targets()[0]
    with pytest.raises(ValueError, match="not an allowed"):
        replace(base, kind="experimental_total_solvation_free_energy")

    pcm = _targets()[-1]
    with pytest.raises(ValueError, match="not a fit"):
        replace(pcm, objective_weight=1.0, use_for_training=True)

    with pytest.raises(ValueError, match="experimental solvation labels"):
        replace(_preregistration(), experimental_solvation_labels_used=True)
    with pytest.raises(ValueError, match="blind split"):
        replace(_preregistration(), blind_split_used_for_model_selection=True)


@pytest.mark.parametrize(
    "kind",
    (
        "exterior_mep",
        "nonuniform_field_energy",
        "nonuniform_exterior_mep_response",
    ),
)
def test_observable_supervised_nonuniform_qm_targets_are_allowed(kind: str) -> None:
    target = replace(
        _targets()[0],
        target_id=f"qm-{kind}-v1",
        kind=kind,
    )

    assert target.kind == kind
    assert target.use_for_training is True
    assert target.objective_weight > 0.0


def test_completed_training_run_is_bound_to_plan_seed_code_and_optimizer_audit() -> (
    None
):
    preregistration = _preregistration()
    run = _run(preregistration)

    assert run.contract_version == PASSIVE_TRAINING_RUN_BINDING_VERSION
    assert run.preregistration_sha256 == preregistration.content_sha256
    assert run.training_code_sha256 == preregistration.training_code_sha256
    assert run.inference_code_sha256 == preregistration.inference_code_sha256
    assert run.optimizer_protocol_sha256 == preregistration.optimizer_protocol_sha256
    assert run.as_dict()["capability_admitted"] is False
    run.validate_preregistration(preregistration)

    with pytest.raises(ValueError, match="training_code_sha256"):
        run.validate_preregistration(
            replace(preregistration, training_code_sha256=_digest("changed-code"))
        )
    with pytest.raises(ValueError, match="not preregistered"):
        replace(run, random_seed=23).validate_preregistration(preregistration)
    with pytest.raises(ValueError, match="incomplete or failed"):
        replace(run, completed_successfully=False)


def test_constructor_collections_are_detached_and_canonical() -> None:
    records = sorted(_digest(f"record-{index}") for index in range(2))
    groups = [_digest("group")]
    split = PassiveTrainingDataSplit(
        split_id="detached-v1",
        role="train",
        index_artifact_sha256=_digest("index"),
        record_sha256s=records,  # type: ignore[arg-type]
        molecule_group_sha256s=groups,  # type: ignore[arg-type]
    )
    records[0] = _digest("mutated")
    groups[0] = _digest("mutated-group")
    assert _digest("mutated") not in split.record_sha256s
    assert _digest("mutated-group") not in split.molecule_group_sha256s
