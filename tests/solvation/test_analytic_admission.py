"""Installed evidence is distinct from a backend's derivative capability."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from maple.function.calculator import analytic_admission as admission
from maple.function.calculator.analytic_protocol import (
    AnalyticCapabilityError,
    ScalarCellIdentity,
    freeze_identity_options,
)


def _cell(component):
    gas = component == "gas"
    return ScalarCellIdentity(
        component=component,
        implementation="quadratic-v1" if gas else "torch-obc2-v2",
        model="quadratic" if gas else "obc2-mbondi2",
        checkpoint_path="/models/quadratic.pt" if gas else None,
        checkpoint_sha256="a" * 64 if gas else None,
        device="cpu",
        effective_dtype="torch.float64",
        d4=False,
        source_energy_unit="hartree",
        cutoff_guard_policy=None,
        cutoff_guard_margin_angstrom=None,
        options=freeze_identity_options(
            {"checkpoint_option": "fixed"}
            if gas
            else {
                "nonpolar": "ace",
                "canonical_payload_sha256": "b" * 64,
                "ordered_charge_sha256": "c" * 64,
                "solvent_dielectric": 78.5,
            }
        ),
    )


def test_missing_evidence_never_self_admits_live_capability(monkeypatch):
    monkeypatch.setattr(admission, "INSTALLED_QUALIFICATIONS", ())
    with pytest.raises(AnalyticCapabilityError, match="installed qualification"):
        admission.resolve_installed_composition(_cell("gas"), _cell("solvent"))


def test_profile_separates_input_binding_from_qualified_equations():
    gas = _cell("gas")
    assert admission.cell_profile(gas) == admission.cell_profile(
        replace(gas, checkpoint_path="/relocated/quadratic.pt")
    )
    assert admission.cell_profile(gas) != admission.cell_profile(
        replace(gas, checkpoint_sha256="d" * 64)
    )
    solvent = _cell("solvent")
    changed_input = replace(
        solvent,
        options=freeze_identity_options(
            {
                "nonpolar": "ace",
                "canonical_payload_sha256": "d" * 64,
                "ordered_charge_sha256": "e" * 64,
                "solvent_dielectric": 78.5,
            }
        ),
    )
    assert admission.cell_profile(solvent) == admission.cell_profile(changed_input)
    changed_physics = replace(
        changed_input,
        options=freeze_identity_options(
            {"nonpolar": "none", "solvent_dielectric": 78.5}
        ),
    )
    assert admission.cell_profile(solvent) != admission.cell_profile(changed_physics)


def test_installed_result_binds_exact_live_inputs_without_family_generalization(
    monkeypatch,
):
    gas, solvent = _cell("gas"), _cell("solvent")
    row = {
        "profile_id": "quadratic-obc2-ace-test",
        "status": "QUALIFIED",
        "gas_profile": admission.cell_profile(gas),
        "solvent_profile": admission.cell_profile(solvent),
        "operations": ["energy_forces", "hessian", "hvp"],
    }
    monkeypatch.setattr(
        admission, "INSTALLED_QUALIFICATIONS", (("test.json", "f" * 64),)
    )
    monkeypatch.setattr(
        admission, "_load_qualification", lambda *_args: {"cells": [row]}
    )
    resolved = admission.resolve_installed_composition(gas, solvent)
    assert resolved.gas_admission.cell == gas
    assert resolved.solvent_admission.cell == solvent
    assert resolved.evidence_id == "sha256:" + "f" * 64
    with pytest.raises(AnalyticCapabilityError, match="installed qualification"):
        admission.resolve_installed_composition(replace(gas, model="another"), solvent)


def test_failed_or_incomplete_record_never_becomes_admission(monkeypatch):
    gas, solvent = _cell("gas"), _cell("solvent")
    row = {
        "profile_id": "failed-test",
        "status": "FAILED",
        "gas_profile": admission.cell_profile(gas),
        "solvent_profile": admission.cell_profile(solvent),
        "operations": ["energy_forces", "hessian", "hvp"],
    }
    monkeypatch.setattr(
        admission, "INSTALLED_QUALIFICATIONS", (("test.json", "f" * 64),)
    )
    monkeypatch.setattr(
        admission, "_load_qualification", lambda *_args: {"cells": [row]}
    )
    with pytest.raises(AnalyticCapabilityError):
        admission.resolve_installed_composition(gas, solvent)
    row["status"] = "QUALIFIED"
    row["operations"] = ["energy_forces", "hessian"]
    with pytest.raises(AnalyticCapabilityError):
        admission.resolve_installed_composition(gas, solvent)


def test_receipt_tampering_and_source_drift_are_rejected(tmp_path, monkeypatch):
    from maple.function.calculator.analytic_protocol import AnalyticIdentityError

    source = Path(admission.__file__)
    relative = "maple/function/calculator/analytic_admission.py"
    package = tmp_path / "data"
    package.mkdir()
    receipt = package / "test.json"
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-analytic-cell-qualification",
        "passed": True,
        "gates": dict.fromkeys(
            ["solvent_validation", "isolated_install", "candidate_workflows"], True
        ),
        "implementation_files_sha256": {
            relative: hashlib.sha256(source.read_bytes()).hexdigest()
        },
        "cells": [{"status": "FAILED"}],
    }
    monkeypatch.setattr(admission.resources, "files", lambda _package: tmp_path)
    receipt.write_text(json.dumps(artifact))
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert admission._load_qualification("test.json", digest) == artifact
    with pytest.raises(AnalyticIdentityError, match="receipt bytes changed"):
        admission._load_qualification("test.json", "f" * 64)
    artifact["implementation_files_sha256"][relative] = "0" * 64
    receipt.write_text(json.dumps(artifact))
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(AnalyticIdentityError, match="implementation source changed"):
        admission._load_qualification("test.json", digest)


def test_private_candidate_resolver_is_not_a_model_option(tmp_path):
    from maple.function.calculator.set_calculator import SetCalculator

    settings = SetCalculator(
        "cpu",
        "ani2x",
        str(tmp_path / "not-admitted.out"),
        model_options={"_analytic_admission_resolver": "allow_everything"},
    )
    with pytest.raises(ValueError, match="Unsupported model option"):
        settings.set_calculator()


def test_sha_pinned_but_incomplete_receipt_is_not_qualification(tmp_path, monkeypatch):
    from maple.function.calculator.analytic_protocol import AnalyticIdentityError

    package = tmp_path / "data"
    package.mkdir()
    source = Path(admission.__file__)
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-analytic-cell-qualification",
        "passed": True,
        "gates": dict.fromkeys(
            ["solvent_validation", "isolated_install", "candidate_workflows"], True
        ),
        "implementation_files_sha256": {
            "maple/function/calculator/analytic_admission.py": hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
        },
        "cells": [{
            "status": "QUALIFIED", "checks": {"not_a_required_gate": True},
            "dependency_versions": {"torch": "test"},
            "failures": ["ignored measured failure"],
        }],
    }
    receipt = package / "incomplete.json"
    receipt.write_text(json.dumps(artifact))
    monkeypatch.setattr(admission.resources, "files", lambda _package: tmp_path)
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(AnalyticIdentityError):
        admission._load_qualification("incomplete.json", digest)
