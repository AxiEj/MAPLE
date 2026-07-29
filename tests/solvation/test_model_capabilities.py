from __future__ import annotations

import json

import pytest

from maple.function.calculator.model_capabilities import (
    CombinationValidator,
    ModelCapabilities,
    ModelCardError,
    SolvationCapabilities,
    ModelProvenanceCard,
    load_model_provenance_card,
    validate_model_card_task,
)


def _full_pes(**overrides):
    values = {
        "energy": True,
        "forces": True,
        "conservative_forces": True,
        "hessian": "finite_difference",
        "supports_md": True,
        "solvation_mode": "none",
        "energy_reference": "absolute",
    }
    values.update(overrides)
    return ModelCapabilities(**values)


def test_model_capabilities_reject_tasks_without_explicit_evidence():
    with pytest.raises(ValueError, match="Frequency"):
        ModelCapabilities(energy=True, forces=True).validate_task("frequency")
    with pytest.raises(ValueError, match="molecular dynamics"):
        ModelCapabilities(energy=True, forces=True).validate_task("md")
    with pytest.raises(ValueError, match="absolute solvation"):
        _full_pes().validate_task("absolute_solvation_free_energy")


def test_native_solution_potential_rejects_additive_solvent():
    with pytest.raises(ValueError, match="Native solution-phase"):
        CombinationValidator(
            model_capabilities=_full_pes(solvation_mode="native"),
            implicit="gb",
        ).validate()


def test_solution_frequency_requires_conservative_forces_from_both_terms():
    with pytest.raises(ValueError, match="solvent force"):
        CombinationValidator(
            model_capabilities=_full_pes(),
            implicit="gb",
            hessian_mode="numerical",
            solvation_capabilities=SolvationCapabilities(
                energy=True,
                forces=True,
                conservative_forces=False,
            ),
            task="frequency",
        ).validate()

    profile = CombinationValidator(
        model_capabilities=_full_pes(),
        implicit="gb",
        hessian_mode="numerical",
        solvation_capabilities=SolvationCapabilities(
            energy=True,
            forces=True,
            conservative_forces=True,
        ),
        task="frequency",
    ).validate()
    assert profile is not None
    assert profile.frequency_type == "effective_solution_pmf"


def test_model_card_loader_is_json_only_and_closed_by_default(tmp_path):
    card = tmp_path / "demo.yaml"
    card.write_text(
        json.dumps(
            {
                "model_id": "demo",
                "version": "1",
                "capabilities": {
                    "energy": True,
                    "forces": True,
                    "conservative_forces": True,
                    "hessian": "finite_difference",
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_model_provenance_card("demo", tmp_path)
    assert loaded.capabilities.supports_energy_derived_forces is True
    assert loaded.capabilities.supports_md is False

    card.write_text("model_id: demo\n", encoding="utf-8")
    with pytest.raises(ModelCardError, match="JSON-compatible"):
        load_model_provenance_card("demo", tmp_path)


def test_model_card_forbidden_tasks_override_capability_flags(tmp_path):
    card = tmp_path / "demo.yaml"
    card.write_text(
        json.dumps(
            {
                "model_id": "demo",
                "version": "1",
                "capabilities": {
                    "energy": True,
                    "forces": True,
                    "conservative_forces": True,
                    "hessian": "finite_difference",
                    "supports_md": True,
                },
                "forbidden_tasks": ["frequency", "md"],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_model_provenance_card("demo", tmp_path)
    with pytest.raises(ValueError, match="forbids task"):
        validate_model_card_task(loaded, "frequency")
    with pytest.raises(ValueError, match="forbids task"):
        validate_model_card_task(loaded, "md")
    validate_model_card_task(loaded, "sp")


def test_model_card_forbidden_tasks_must_be_a_sequence(tmp_path):
    card = tmp_path / "demo.yaml"
    card.write_text(
        json.dumps(
            {"model_id": "demo", "version": "1", "forbidden_tasks": "frequency"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelCardError, match="forbidden_tasks"):
        load_model_provenance_card("demo", tmp_path)
