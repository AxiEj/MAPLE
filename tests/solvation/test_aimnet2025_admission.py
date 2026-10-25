"""Synthetic admission boundaries only; never model or qualification evidence."""
from __future__ import annotations

import builtins
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from maple.function.calculator import set_calculator as runtime
from maple.function.calculator.aimnet import _aimnet2025_identity as pins
from maple.function.calculator.aimnet._aimnet2025_calculator import AIMNet2025Calculator


def _aim_identities():
    """Complete synthetic metadata; no native constructor/checkpoint is loaded."""
    profiles = {}
    for device in ("cpu", "cuda:0"):
        config = {
            "metadata": deepcopy(pins.MODEL_METADATA),
            "engine": "official aimnet public scalar; forces=False,hessian=False; requires_grad leaf",
            "train": False, "compiled": False, "lr": True, "device": device,
            "cutoff": 5.0, "cutoff_lr": "inf", "coulomb_cutoff": "inf",
            "coulomb_method": "simple", "d3_cutoff": 15.0, "nb_threshold": 120,
            "neighbor_cutoffs": {"_nblist": 5.0, "_nblist_lr": 15.0,
                                 "_nblist_dftd3": None, "_nblist_coulomb": None},
            "input_dtypes": {"coord": "torch.float32", "numbers": "torch.int32", "charge": "torch.float32"},
            "model_input_dtypes": ["torch.float32", "torch.int64", "torch.float32"],
            "model_flags": {"": {"d2features": True}},
            "d3": dict(pins.MODEL_METADATA["d3_params"], smoothing_on=12.0,
                       smoothing_off=15.0, key_out="energy"),
        }
        for term in ("coulomb", "embedded_sr"):
            config[term] = {"rc": pins.MODEL_METADATA["coulomb_sr_rc"], "envelope": "exp",
                            "key_in": "charges", "key_out": "energy", "factor": 7.199822675975274}
        config["coulomb"].update(method="simple", subtract_sr=False)
        profiles[device.split(":")[0]] = {
            "model": "aimnet2025", "registry_key": pins.REGISTRY_KEY,
            "profile_id": pins.PROFILE_ID, "checkpoint_sha256": pins.CHECKPOINT_SHA256,
            "device": device, "effective_nn_dtype": "float32", "effective_ad_class": "N32",
            "atomic_shift_dtype": "float64", "energy_accumulation_dtype": "float64",
            "d4": False, "source_energy_unit": "eV", "state_sha256": "e" * 64,
            "external_runtime": {"packages": dict(pins.PACKAGES), "source_sha256": dict(pins.SOURCE_SHA256)},
            "native_configuration": config,
        }
    return profiles


def _objects(monkeypatch, identity):
    from test_independent_obc2_default import _gate_objects

    _, correction = _gate_objects(monkeypatch)
    correction.underlying_provider.provenance['dtype'] = 'torch.float32'
    monkeypatch.setattr(AIMNet2025Calculator, "analytic_cell_identity", property(lambda _: deepcopy(identity)))
    monkeypatch.setattr(AIMNet2025Calculator, "runtime_profile_identity", property(lambda _: deepcopy(identity)))
    calculator = AIMNet2025Calculator.__new__(AIMNet2025Calculator)
    calculator.device = torch.device(identity["device"])
    calculator.dtype = torch.float32
    calculator.d4 = False
    calculator.analytic_implicit_derivatives_admitted = True
    return calculator, correction


def _install(monkeypatch, receipt=None):
    from test_independent_obc2_default import _install_receipt, _valid_qualification_receipt

    _install_receipt(monkeypatch, _valid_qualification_receipt() if receipt is None else receipt)


def _admit(calculator, correction, **changes):
    args = dict(model="aimnet2025", device=str(calculator.device), d4=False)
    args.update(changes)
    runtime._admit_independent_obc2_analytic_cell(calculator, correction, **args)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("model", ["aimnet2025", "aimnet2-2025", "aimnet2_2025"])
@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_exact_live_aim_cells_and_aliases_admit(monkeypatch, device, model, nonpolar):
    calc, correction = _objects(monkeypatch, _aim_identities()[device])
    correction.underlying_provider.provenance["nonpolar"] = nonpolar
    _install(monkeypatch)
    _admit(calc, correction, model=model, device=device)
    assert calc.analytic_implicit_derivatives_admitted is True
    assert correction.analytic_task_derivatives_admitted is True


@pytest.mark.parametrize("field,value", [
    ("model", "aimnet2"), ("checkpoint_sha256", "0" * 64),
    ("profile_id", "toy-fp64"), ("effective_nn_dtype", "float64"),
    ("effective_ad_class", "D64"), ("atomic_shift_dtype", "float32"),
    ("energy_accumulation_dtype", "float32"), ("device", "cuda:1"),
    ("registry_key", "aimnet2-wb97m-d3_0"), ("state_sha256", "0" * 64),
    ("d4", True), ("source_energy_unit", "Hartree"),
])
def test_live_identity_changes_fail_closed(monkeypatch, field, value):
    identity = _aim_identities()["cuda"]
    identity[field] = value
    calc, correction = _objects(monkeypatch, identity)
    _install(monkeypatch)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit(calc, correction)
    assert calc.analytic_implicit_derivatives_admitted is False
    assert correction.analytic_task_derivatives_admitted is False


@pytest.mark.parametrize("mutation", ["source", "table", "package", "d3", "cutoff", "flag", "missing_flag", "neighbors"])
def test_live_external_runtime_configuration_drift_rejected(monkeypatch, mutation):
    identity = _aim_identities()["cpu"]
    if mutation in {"source", "table"}:
        key = "modules/lr.py" if mutation == "source" else "dftd3_data.pt"
        identity["external_runtime"]["source_sha256"][key] = "0" * 64
    elif mutation == "package":
        identity["external_runtime"]["packages"]["aimnet"] = "0.2.1"
    else:
        config = identity["native_configuration"]
        if mutation == "d3":
            config["d3"]["s8"] = 0.0
        elif mutation == "cutoff":
            config["cutoff"] = 4.0
        elif mutation == "flag":
            config["model_flags"][""]["d2features"] = False
        elif mutation == "neighbors":
            config["neighbor_cutoffs"]["_nblist"] = 4.0
        else:
            config.pop("model_flags")
    calc, correction = _objects(monkeypatch, identity)
    _install(monkeypatch)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit(calc, correction)
    assert not calc.analytic_implicit_derivatives_admitted
    assert not correction.analytic_task_derivatives_admitted


@pytest.mark.parametrize("model", ["aimnet2", "aimnet2nse", "ani2x"])
def test_legacy_or_foreign_model_cannot_claim_2025_profile(monkeypatch, model):
    calc, correction = _objects(monkeypatch, _aim_identities()["cpu"])
    _install(monkeypatch)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit(calc, correction, model=model)
    assert not calc.analytic_implicit_derivatives_admitted
    assert not correction.analytic_task_derivatives_admitted


def test_subclass_and_foreign_profile_getter_cannot_substitute(monkeypatch):
    calc, correction = _objects(monkeypatch, _aim_identities()["cpu"])
    class Plugin(AIMNet2025Calculator):
        pass
    _install(monkeypatch)
    for impostor in (Plugin.__new__(Plugin), SimpleNamespace()):
        impostor.device = calc.device
        impostor.analytic_implicit_derivatives_admitted = True
        with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
            _admit(impostor, correction)
        assert not impostor.analytic_implicit_derivatives_admitted
        assert not correction.analytic_task_derivatives_admitted


def test_both_live_identity_getters_must_match(monkeypatch):
    calc, correction = _objects(monkeypatch, _aim_identities()["cpu"])
    monkeypatch.setattr(AIMNet2025Calculator, "runtime_profile_identity", property(lambda _: {"foreign": True}))
    _install(monkeypatch)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit(calc, correction)
    assert not calc.analytic_implicit_derivatives_admitted
    assert not correction.analytic_task_derivatives_admitted


@pytest.mark.parametrize("change", [{"device": "cuda:1"}, {"d4": True}])
def test_requested_device_and_d4_must_match_live_cell(monkeypatch, change):
    calc, correction = _objects(monkeypatch, _aim_identities()["cuda"])
    _install(monkeypatch)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit(calc, correction, **change)
    assert not calc.analytic_implicit_derivatives_admitted
    assert not correction.analytic_task_derivatives_admitted


@pytest.mark.parametrize("mutation", ["old_filename", "v2_filename", "old_schema", "missing_gate", "false_gate", "bool_count", "missing_profile", "missing_receipt"])
def test_current_receipt_required_without_historical_fallback(monkeypatch, mutation):
    from test_independent_obc2_default import _valid_qualification_receipt
    from maple.function.calculator import _independent_obc2_qualified_cell as pointer
    from pathlib import Path

    calc, correction = _objects(monkeypatch, _aim_identities()["cpu"])
    receipt = _valid_qualification_receipt()
    if mutation == "old_schema":
        receipt["schema_version"] = 1
    elif mutation == "missing_gate":
        receipt["qualification_gates"].pop("aimnet2025_composed_hessian")
    elif mutation in {"false_gate", "bool_count"}:
        gate = receipt["qualification_gates"]["source_unchanged"]
        gate["passed" if mutation == "false_gate" else "passed_count"] = False if mutation == "false_gate" else True
    elif mutation == "missing_profile":
        receipt["aimnet2025_runtime_profiles"]["cpu"].pop("profile_id")
    _install(monkeypatch, receipt)
    if mutation == "old_filename":
        monkeypatch.setattr(pointer, "QUALIFICATION_RECEIPT_FILENAME", "independent_obc2_analytic_qualification.json")
    elif mutation == "v2_filename":
        monkeypatch.setattr(pointer, "QUALIFICATION_RECEIPT_FILENAME", "independent_obc2_analytic_qualification_v2.json")
    elif mutation == "missing_receipt":
        original = Path.read_bytes
        def missing(path):
            if path.name == runtime._QUALIFICATION_RECEIPT_FILENAME:
                raise FileNotFoundError(path)
            return original(path)
        monkeypatch.setattr(Path, "read_bytes", missing)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit(calc, correction)
    assert not calc.analytic_implicit_derivatives_admitted
    assert not correction.analytic_task_derivatives_admitted


def test_ani_admission_does_not_import_native_optional_runtime(monkeypatch):
    from test_independent_obc2_default import _gate_objects
    calc, correction = _gate_objects(monkeypatch)
    _install(monkeypatch)
    original = builtins.__import__
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split(".")[0] in {"aimnet", "warp", "warp_lang", "nvalchemiops"}:
            raise AssertionError(f"optional native package imported on ANI admission: {name}")
        return original(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", guarded)
    runtime._admit_independent_obc2_analytic_cell(calc, correction, model="ani2x", device="cpu", d4=False)
    assert calc.analytic_implicit_derivatives_admitted is True
    assert correction.analytic_task_derivatives_admitted is True


def test_full_native_metadata_extension_is_bound_not_dropped(monkeypatch):
    from test_independent_obc2_default import _valid_qualification_receipt
    profiles = _aim_identities()
    for identity in profiles.values():
        identity["native_configuration"]["metadata"].update(
            family=None, supports_charged_systems=None, has_embedded_d3ts=False)
    cells = runtime._qualified_independent_obc2_cells(profiles)
    assert len(cells) == 6
    assert cells[2]["gas"]["runtime_profile"] == profiles["cpu"]
    receipt = _valid_qualification_receipt()
    receipt.update(qualified_cells=cells, aimnet2025_runtime_profiles=profiles)
    calc, correction = _objects(monkeypatch, profiles["cpu"])
    _install(monkeypatch, receipt)
    _admit(calc, correction)
    assert calc.analytic_implicit_derivatives_admitted
    # Extra metadata remains sealed, not thrown away by a projection.
    profiles["cpu"]["native_configuration"]["metadata"]["family"] = "other"
    calc, correction = _objects(monkeypatch, profiles["cpu"])
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit(calc, correction)
    assert not calc.analytic_implicit_derivatives_admitted
    assert not correction.analytic_task_derivatives_admitted


def test_cutoff_guard_extends_exact_global_registry_without_changing_old_gates():
    assert runtime._QUALIFICATION_GATE_COUNTS == {
        "sealed_oracle_1052_energy": (1052, 1052),
        "portable_oracle_48_energy_force": (48, 48),
        "analytic_hessian_16_fd": (16, 16),
        "direct_hvp_144": (144, 144),
        "no_openmm_candidate_runtime": (1, 1),
        "candidate_exact_cell_public_composition": (2, 2),
        "source_unchanged": (1, 1),
        "aimnet2025_native_gas_identity": (24, 24),
        "aimnet2025_composed_hessian": (48, 48),
        "aimnet2025_direct_hvp": (144, 144),
        "aimnet2025_cutoff_guard": (8, 8),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mutation", ["old_three", "wrong_count", "false_verdict"])
def test_missing_or_failed_cutoff_gate_cannot_admit_unchanged_model_cells(monkeypatch, device, mutation):
    from test_independent_obc2_default import _valid_qualification_receipt
    calc, correction = _objects(monkeypatch, _aim_identities()[device])
    receipt = _valid_qualification_receipt()
    # A ten-gate historical fixture is not upgraded merely by resealing it.
    if mutation == "old_three":
        receipt["qualification_gates"].pop("aimnet2025_cutoff_guard", None)
    else:
        gate = receipt["qualification_gates"].setdefault("aimnet2025_cutoff_guard", {
            "expected_count": 8, "passed_count": 8, "passed": True})
        if mutation == "wrong_count":
            gate.update(expected_count=1, passed_count=1)
        else:
            gate["passed"] = False
    _install(monkeypatch, receipt)
    with pytest.raises(RuntimeError, match="ANALYTIC_CELL_UNQUALIFIED"):
        _admit(calc, correction)
    assert not calc.analytic_implicit_derivatives_admitted
    assert not correction.analytic_task_derivatives_admitted


def test_aim_receipt_cannot_admit_legacy_fp64_solvent(monkeypatch):
    calc, correction = _objects(monkeypatch, _aim_identities()['cpu'])
    correction.underlying_provider.provenance['dtype'] = 'torch.float64'
    _install(monkeypatch)
    with pytest.raises(RuntimeError, match='ANALYTIC_CELL_UNQUALIFIED'):
        _admit(calc, correction)
    assert not calc.analytic_implicit_derivatives_admitted
    assert not correction.analytic_task_derivatives_admitted
