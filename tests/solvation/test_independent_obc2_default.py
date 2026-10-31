from __future__ import annotations

import hashlib
import json
import sys
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pytest

from maple.function.calculator.extra_correction.implicit.selector import (
    resolve_implicit_provider,
)
from maple.function.read.command_control import CommandControl


def _parse_solv(line: str) -> dict:
    return CommandControl.from_settings(
        ["#model=ani2x", "#charge(source=mol2)", line]
    ).as_dict()["solv"]


def _gate_objects(monkeypatch, *, precision_change=None):
    import torch

    from maple.function.calculator.ani._ani_calculator import ANICalculator
    from maple.function.calculator.set_calculator import (
        _ANI2X_QUALIFIED_CHECKPOINT_SHA256,
    )

    precision = {
        "effective_dtype": "float64",
        "original_checkpoint_sha256": _ANI2X_QUALIFIED_CHECKPOINT_SHA256,
    }
    if precision_change:
        precision.update(precision_change)
    monkeypatch.setattr(
        ANICalculator,
        "inference_precision_provenance",
        property(lambda _self: dict(precision)),
    )
    calculator = ANICalculator.__new__(ANICalculator)
    calculator.dtype = torch.float64
    calculator.analytic_implicit_derivatives_admitted = True
    provider = SimpleNamespace(
        provenance={
            "runtime_provider": "torch-obc2",
            "model": "obc2",
            "profile": "obc2-mbondi2",
            "nonpolar": "ace",
            "formula_manifest_id": "openmm-8.5.2-obc2-mbondi2-v1",
            "formula_source_sha256": (
                "f96a49a9e9ef0528e937481b1c1331b43ad03e6ec3018b478c642b301f07391a"
            ),
            "canonical_payload_sha256": (
                "6387f1a16e10569d91ea8dc03a90c56035cf4a52888df54c18a1bd5a4bf8c30f"
            ),
            "equation_profile": "obc2-zero-salt-v1",
            "dtype": "torch.float64",
        }
    )
    correction = SimpleNamespace(
        analytic_task_derivatives_admitted=True,
        underlying_provider=provider,
    )
    def set_admission(admitted):
        correction.analytic_task_derivatives_admitted = bool(admitted)

    correction._set_analytic_task_derivatives_admitted = set_admission
    return calculator, correction


def _valid_qualification_receipt():
    from maple.function.calculator import set_calculator as module
    from maple.function.calculator.extra_correction.implicit.obc2_profile import (
        OBC2_CANONICAL_PAYLOAD_SHA256,
        OBC2_FORMULA_MANIFEST_ID,
        OPENMM_CUSTOMGBFORCES_SHA256,
    )

    cells = []
    for nonpolar in ("ace", "none"):
        cells.append(
            {
                "gas": {
                    "model": "ani2x",
                    "checkpoint_sha256": module._ANI2X_QUALIFIED_CHECKPOINT_SHA256,
                    "device": "cpu",
                    "dtype": "float64",
                    "d4": False,
                },
                "solvent": {
                    "provider": "torch",
                    "runtime_provider": "torch-obc2",
                    "model": "obc2",
                    "profile": "obc2-mbondi2",
                    "nonpolar": nonpolar,
                    "formula_manifest_id": OBC2_FORMULA_MANIFEST_ID,
                    "formula_source_sha256": OPENMM_CUSTOMGBFORCES_SHA256,
                    "canonical_payload_sha256": OBC2_CANONICAL_PAYLOAD_SHA256,
                    "parameter_profile": "obc2-zero-salt-v1",
                    "dtype": "torch.float64",
                },
            }
        )
    from test_aimnet2025_admission import _aim_identities

    profiles = _aim_identities()
    cells.extend(module._qualified_independent_obc2_cells(profiles)[2:])
    gates = {
        name: {"expected_count": expected, "passed_count": passed, "passed": True}
        for name, (expected, passed) in module._QUALIFICATION_GATE_COUNTS.items()
    }
    gates["analytic_hessian_16_fd"]["steps_angstrom"] = [
        0.001,
        0.0005,
        0.00025,
    ]
    gates["direct_hvp_144"]["dense_hessian_forbidden"] = True
    gates["candidate_exact_cell_public_composition"]["nonpolar"] = [
        "ace",
        "none",
    ]
    return {
        "schema_version": 2,
        "artifact_type": "maple-independent-obc2-analytic-qualification",
        "status": "qualified",
        "installed_runtime_sources": module._installed_runtime_source_inventory(),
        "qualification_sources": {"tests/private-harness.py": "a" * 64},
        "qualified_cells": cells,
        "aimnet2025_runtime_profiles": profiles,
        "qualification_gates": gates,
    }


def _install_receipt(monkeypatch, receipt, *, seal=True):
    from maple.function.calculator import _independent_obc2_qualified_cell as pointer
    from maple.function.calculator import set_calculator as module

    if seal:
        receipt["content_sha256"] = hashlib.sha256(
            module._canonical_receipt_payload(receipt)
        ).hexdigest()
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    monkeypatch.setattr(
        pointer,
        "QUALIFICATION_RECEIPT_FILENAME",
        "independent_obc2_analytic_qualification_v5.json",
    )
    monkeypatch.setattr(
        pointer,
        "QUALIFICATION_RECEIPT_SHA256",
        hashlib.sha256(receipt_bytes).hexdigest(),
    )
    receipt_path = Path(module.__file__).with_name("data") / pointer.QUALIFICATION_RECEIPT_FILENAME
    original_read_bytes = Path.read_bytes

    def read_only_receipt(path):
        if path == receipt_path:
            return receipt_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_only_receipt)


@pytest.mark.parametrize("model", [None, "obc2"])
def test_default_obc2_resolves_to_independent_torch(model):
    model_text = "" if model is None else f",model={model}"
    options = _parse_solv(
        f"#solv(implicit=water,method=gb{model_text},experimental=true)"
    )
    assert options == {
        "implicit": "water",
        "method": "gb",
        "experimental": True,
        "provider": "torch",
        "model": "obc2",
        "profile": "obc2-mbondi2",
        "nonpolar": "ace",
    }


@pytest.mark.parametrize("model", ["hct", "obc1", "gbn", "gbn2"])
def test_explicit_legacy_model_without_provider_keeps_openmm(model):
    options = _parse_solv(
        f"#solv(implicit=water,method=gb,model={model},experimental=true)"
    )
    assert options["provider"] == "openmm"
    assert options["model"] == model


@pytest.mark.parametrize(
    "fragment", ["platform=Reference", "precision=double", "nonpolar=lcpo", "inner=prebuilt"]
)
def test_openmm_only_options_require_explicit_openmm(fragment):
    with pytest.raises(ValueError, match="provider=openmm explicitly"):
        _parse_solv(
            f"#solv(implicit=water,method=gb,{fragment},experimental=true)"
        )


def test_explicit_provider_has_precedence_and_no_failure_fallback():
    assert resolve_implicit_provider(
        {"method": "gb", "provider": "openmm", "model": "obc2", "nonpolar": "lcpo"}
    ) == "openmm"
    assert resolve_implicit_provider(
        {"method": "gb", "provider": "torch", "model": "obc2"}
    ) == "torch"
    with pytest.raises(ValueError, match="requires model=obc2"):
        _parse_solv(
            "#solv(implicit=water,method=gb,provider=torch,model=hct,experimental=true)"
        )


def test_selector_does_not_import_openmm():
    before = {name for name in sys.modules if name == "openmm" or name.startswith("openmm.")}
    assert resolve_implicit_provider({"method": "gb"}) == "torch"
    after = {name for name in sys.modules if name == "openmm" or name.startswith("openmm.")}
    assert after == before


def test_private_admission_gate_has_the_frozen_narrow_signature():
    from maple.function.calculator.set_calculator import (
        _admit_independent_obc2_analytic_cell,
    )

    assert list(signature(_admit_independent_obc2_analytic_cell).parameters) == [
        "calculator", "correction", "model", "device", "d4"
    ]


def test_unsealed_pointer_fails_closed_and_leaves_both_sides_false(monkeypatch):
    from maple.function.calculator import _independent_obc2_qualified_cell as pointer
    from maple.function.calculator.set_calculator import (
        _admit_independent_obc2_analytic_cell,
    )

    monkeypatch.setattr(pointer, "QUALIFICATION_RECEIPT_SHA256", None)
    calculator, correction = _gate_objects(monkeypatch)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit_independent_obc2_analytic_cell(
            calculator,
            correction,
            model="ani2x",
            device="cpu",
            d4=False,
        )
    assert calculator.analytic_implicit_derivatives_admitted is False
    assert correction.analytic_task_derivatives_admitted is False


def test_valid_exact_cell_receipt_admits_both_sides(monkeypatch):
    from maple.function.calculator.set_calculator import (
        _admit_independent_obc2_analytic_cell,
    )

    calculator, correction = _gate_objects(monkeypatch)
    _install_receipt(monkeypatch, _valid_qualification_receipt())
    _admit_independent_obc2_analytic_cell(
        calculator, correction, model="ani2x", device="cpu", d4=False
    )
    assert calculator.analytic_implicit_derivatives_admitted is True
    assert correction.analytic_task_derivatives_admitted is True


def test_same_labels_on_fake_calculator_class_are_rejected(monkeypatch):
    from maple.function.calculator.ani._ani_calculator import ANICalculator
    from maple.function.calculator.set_calculator import (
        _admit_independent_obc2_analytic_cell,
    )

    _calculator, correction = _gate_objects(monkeypatch)

    class SpoofedANI(ANICalculator):
        pass

    fake = SpoofedANI.__new__(SpoofedANI)
    fake.analytic_implicit_derivatives_admitted = True
    _install_receipt(monkeypatch, _valid_qualification_receipt())
    with pytest.raises(RuntimeError, match="exact shipped ANICalculator"):
        _admit_independent_obc2_analytic_cell(
            fake, correction, model="ani2x", device="cpu", d4=False
        )
    assert fake.analytic_implicit_derivatives_admitted is False
    assert correction.analytic_task_derivatives_admitted is False


def test_missing_receipt_fails_closed(monkeypatch):
    from maple.function.calculator import _independent_obc2_qualified_cell as pointer
    from maple.function.calculator import set_calculator as module

    calculator, correction = _gate_objects(monkeypatch)
    monkeypatch.setattr(pointer, "QUALIFICATION_RECEIPT_SHA256", "0" * 64)
    receipt_path = Path(module.__file__).with_name("data") / pointer.QUALIFICATION_RECEIPT_FILENAME
    original_read_bytes = Path.read_bytes

    def missing_receipt(path):
        if path == receipt_path:
            raise FileNotFoundError(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", missing_receipt)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        module._admit_independent_obc2_analytic_cell(
            calculator, correction, model="ani2x", device="cpu", d4=False
        )
    assert calculator.analytic_implicit_derivatives_admitted is False
    assert correction.analytic_task_derivatives_admitted is False


@pytest.mark.parametrize(
    "mutation",
    [
        "content_tamper",
        "source_inventory_drift",
        "extra_nonpolar_cell",
        "missing_nonpolar_cell",
        "gate_denominator",
    ],
)
def test_receipt_identity_failures_leave_both_sides_false(monkeypatch, mutation):
    from maple.function.calculator.set_calculator import (
        _admit_independent_obc2_analytic_cell,
    )

    calculator, correction = _gate_objects(monkeypatch)
    receipt = _valid_qualification_receipt()
    if mutation == "content_tamper":
        receipt["content_sha256"] = "0" * 64
        _install_receipt(monkeypatch, receipt, seal=False)
    else:
        if mutation == "source_inventory_drift":
            first = next(iter(receipt["installed_runtime_sources"]))
            receipt["installed_runtime_sources"][first] = "0" * 64
        elif mutation == "extra_nonpolar_cell":
            receipt["qualified_cells"].append(dict(receipt["qualified_cells"][0]))
        elif mutation == "missing_nonpolar_cell":
            receipt["qualified_cells"].pop()
        elif mutation == "gate_denominator":
            receipt["qualification_gates"]["sealed_oracle_1052_energy"][
                "passed_count"
            ] = 1051
        _install_receipt(monkeypatch, receipt)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit_independent_obc2_analytic_cell(
            calculator, correction, model="ani2x", device="cpu", d4=False
        )
    assert calculator.analytic_implicit_derivatives_admitted is False
    assert correction.analytic_task_derivatives_admitted is False


@pytest.mark.parametrize(
    "precision_change,d4",
    [
        ({"original_checkpoint_sha256": "0" * 64}, False),
        ({"effective_dtype": "float32"}, False),
        ({}, True),
    ],
)
def test_live_checkpoint_dtype_and_d4_mismatch_fail_closed(
    monkeypatch, precision_change, d4
):
    from maple.function.calculator.set_calculator import (
        _admit_independent_obc2_analytic_cell,
    )

    calculator, correction = _gate_objects(
        monkeypatch, precision_change=precision_change
    )
    _install_receipt(monkeypatch, _valid_qualification_receipt())
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit_independent_obc2_analytic_cell(
            calculator, correction, model="ani2x", device="cpu", d4=d4
        )
    assert calculator.analytic_implicit_derivatives_admitted is False
    assert correction.analytic_task_derivatives_admitted is False


def test_runtime_inventory_uses_strict_relative_paths_and_valid_sha256():
    from maple.function.calculator.set_calculator import (
        _installed_runtime_source_inventory,
    )

    inventory = _installed_runtime_source_inventory()
    assert inventory
    assert all(path.startswith("maple/") and ".." not in path.split("/") for path in inventory)
    assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in inventory.values())
    assert "maple/function/calculator/_independent_obc2_qualified_cell.py" not in inventory


def test_direct_default_constructs_independent_provider_without_openmm(
    water_mol2, tmp_path
):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    before = {name for name in sys.modules if name == "openmm" or name.startswith("openmm.")}
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {"method": "gb", "experimental": True},
        output=tmp_path / "default.out",
        model_device="cpu",
    )
    after = {name for name in sys.modules if name == "openmm" or name.startswith("openmm.")}
    assert after == before
    assert correction.solvation_options["provider"] == "torch"
    assert correction.provider.provenance["runtime_provider"] == "torch-obc2"
    assert correction.analytic_task_derivatives_admitted is False


@pytest.fixture
def audited_correction(water_mol2, tmp_path, request):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    return ImplicitSolvationCorrection(
        MOL2Reader(str(water_mol2), charge=0, mult=1),
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {"method": "gb", "experimental": True, "dtype": getattr(request, 'param', 'float64')},
        output=tmp_path / "admission.out",
        model_device="cpu",
    )


def _audit_admitted(correction):
    manifest = json.loads((correction.audit_dir / "manifest.json").read_text())
    assert manifest["capabilities"]["production_admitted"] is False
    assert manifest["capabilities"]["accuracy_certified"] is False
    return manifest["capabilities"]["analytic_task_derivatives_admitted"]


def test_receipt_admission_refreshes_manifest_and_rejection_clears_it(
    monkeypatch, audited_correction
):
    from maple.function.calculator import _independent_obc2_qualified_cell as pointer
    from maple.function.calculator import set_calculator as module

    calculator, _correction = _gate_objects(monkeypatch)
    _install_receipt(monkeypatch, _valid_qualification_receipt())
    assert _audit_admitted(audited_correction) is False
    module._admit_independent_obc2_analytic_cell(
        calculator, audited_correction, model="ani2x", device="cpu", d4=False
    )
    assert _audit_admitted(audited_correction) is True
    monkeypatch.setattr(pointer, "QUALIFICATION_RECEIPT_SHA256", None)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        module._admit_independent_obc2_analytic_cell(
            calculator, audited_correction, model="ani2x", device="cpu", d4=False
        )
    assert calculator.analytic_implicit_derivatives_admitted is False
    assert audited_correction.analytic_task_derivatives_admitted is False
    assert _audit_admitted(audited_correction) is False


def test_manifest_refresh_is_atomic(monkeypatch, audited_correction):
    from maple.function.calculator.extra_correction.implicit import correction as module

    destination = audited_correction.audit_dir / "manifest.json"
    original = destination.read_bytes()
    replace = module.os.replace
    replacements = []

    def observe_replace(source, target):
        assert Path(target) == destination
        assert destination.read_bytes() == original
        candidate = json.loads(Path(source).read_text())
        assert candidate["capabilities"]["analytic_task_derivatives_admitted"] is True
        replacements.append(Path(source))
        return replace(source, target)

    monkeypatch.setattr(module.os, "replace", observe_replace)
    audited_correction._set_analytic_task_derivatives_admitted(True)
    assert len(replacements) == 1
    assert not replacements[0].exists()
    assert _audit_admitted(audited_correction) is True


@pytest.mark.parametrize("failed_admission_state", [False, True])
def test_manifest_write_failure_cannot_admit_live_flags(
    monkeypatch, audited_correction, failed_admission_state
):
    from maple.function.calculator import set_calculator as module
    from maple.function.calculator.extra_correction.implicit import correction as correction_module

    calculator, _correction = _gate_objects(monkeypatch)
    _install_receipt(monkeypatch, _valid_qualification_receipt())
    replace = correction_module.os.replace

    def reject_admission(source, target):
        candidate = json.loads(Path(source).read_text())
        if candidate["capabilities"]["analytic_task_derivatives_admitted"] is failed_admission_state:
            raise OSError("audit disk unavailable")
        return replace(source, target)

    monkeypatch.setattr(correction_module.os, "replace", reject_admission)
    with pytest.raises(OSError, match="audit disk unavailable"):
        module._admit_independent_obc2_analytic_cell(
            calculator, audited_correction, model="ani2x", device="cpu", d4=False
        )
    assert calculator.analytic_implicit_derivatives_admitted is False
    assert audited_correction.analytic_task_derivatives_admitted is False
    assert _audit_admitted(audited_correction) is False
    assert not list(audited_correction.audit_dir.glob(".manifest.json.*.tmp"))


def test_legacy_factory_restores_admission_in_manifest(monkeypatch, water_mol2, tmp_path):
    pytest.importorskip("openmm")
    from maple.function.calculator import set_calculator as module
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    class GasForFactoryWiring:
        SUPPORTED_HESSIAN_MODES = ("analytic",)

        def __init__(self, **_kwargs):
            self.hessian = "analytic"
            self.device = "cpu"

        @classmethod
        def build_kwargs_from_options(cls, *_args, **_kwargs):
            return {}

        def prepare_analytic_derivatives(self):
            pass

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    factory = module.SetCalculator(
        "cpu", "ani2x", str(tmp_path / "legacy.out"), atoms=atoms,
        implicit="gb", solvent="water", model_options={"hessian": "analytic"},
        solvation_options={
            "method": "gb", "provider": "openmm", "model": "obc2",
            "platform": "Reference", "experimental": True,
        },
        charge_options={"source": "mol2", "mode": "fixed", "geometry": "keep"},
    )
    # Only gas loading/validation is stubbed; the legacy correction and factory
    # admission wiring execute unchanged, without downloading a checkpoint.
    monkeypatch.setattr(factory, "_discover_calculator_class", lambda _name: GasForFactoryWiring)
    monkeypatch.setattr(factory, "_validate_model_options", lambda _cls: None)
    monkeypatch.setattr(factory, "_validate_against_class", lambda _cls: None)
    monkeypatch.setattr(factory, "_resolve_model_path", lambda *_args: None)
    states = []
    update = ImplicitSolvationCorrection._set_analytic_task_derivatives_admitted

    def observe_update(correction, admitted):
        update(correction, admitted)
        states.append(_audit_admitted(correction))

    monkeypatch.setattr(ImplicitSolvationCorrection, "_set_analytic_task_derivatives_admitted", observe_update)
    calculator = factory.set_calculator()
    assert states == [False, True]
    assert calculator.analytic_implicit_derivatives_admitted is True
    assert calculator.solvent_correction.analytic_task_derivatives_admitted is True
    assert _audit_admitted(calculator.solvent_correction) is True


@pytest.mark.parametrize('audited_correction', ['float32'], indirect=True)
def test_aim_live_cell_authority_precedes_manifest_and_rejection_clears_it(
    monkeypatch, audited_correction
):
    from test_aimnet2025_admission import _aim_identities, _objects
    from maple.function.calculator import set_calculator as module

    calc, _correction = _objects(monkeypatch, _aim_identities()["cpu"])
    _install_receipt(monkeypatch, _valid_qualification_receipt())
    module._admit_independent_obc2_analytic_cell(
        calc, audited_correction, model="aimnet2025", device="cpu", d4=False
    )
    assert calc.analytic_implicit_derivatives_admitted is True
    assert audited_correction.analytic_task_derivatives_admitted is True
    assert _audit_admitted(audited_correction) is True
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        module._admit_independent_obc2_analytic_cell(
            calc, audited_correction, model="aimnet2025", device="cuda:0", d4=False
        )
    assert calc.analytic_implicit_derivatives_admitted is False
    assert audited_correction.analytic_task_derivatives_admitted is False
    assert _audit_admitted(audited_correction) is False
