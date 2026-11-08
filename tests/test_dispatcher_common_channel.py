from __future__ import annotations

import inspect

from maple.function.dispatcher.dispatcher import Dispatcher


def test_legacy_executor_contains_only_task_execution_not_threshold_preparation():
    source = inspect.getsource(Dispatcher._dispatch_legacy)
    assert "set_throshould" not in source
    assert "_apply_v4_irc_threshold_precedence" not in source
    assert "is_pure_" not in source

from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace

import pytest
from ase import Atoms

from maple.function.dispatcher import channel
from maple.function.dispatcher.channel import (
    CanonicalFamily,
    DispatchTask,
    FamilyPolicy,
    NormalizedDispatchRequest,
    PreflightMode,
    ReportingMode,
    ResolvedDispatchRequest,
    SPShape,
    ThresholdPolicy,
)
from maple.function.route2_smd_profiles import Route2WorkflowCapability


_EXPECTED_POLICIES = {
    CanonicalFamily.ORDINARY: (PreflightMode.LEGACY, ReportingMode.NONE),
    CanonicalFamily.DDX_V1: (PreflightMode.LEGACY, ReportingMode.NONE),
    CanonicalFamily.DDX_V2: (PreflightMode.DDX_V2, ReportingMode.NONMD_V2),
    CanonicalFamily.TORCH_V3: (PreflightMode.TORCH_V3, ReportingMode.NONE),
    CanonicalFamily.TORCH_V4: (PreflightMode.TORCH_V4, ReportingMode.WORKFLOW),
}


def _resolved(
    family: CanonicalFamily,
    *,
    task: DispatchTask = DispatchTask.SP,
    threshold: ThresholdPolicy = ThresholdPolicy.COMMON,
    capability: Route2WorkflowCapability | None = None,
) -> ResolvedDispatchRequest:
    preflight, reporting = _EXPECTED_POLICIES[family]
    return ResolvedDispatchRequest(
        FamilyPolicy(family, preflight, reporting),
        NormalizedDispatchRequest(task, task.value, None),
        capability,
        SPShape.LEGACY,
        threshold,
    )


def _patch_family(monkeypatch, family: CanonicalFamily) -> SimpleNamespace:
    image = SimpleNamespace(calc=SimpleNamespace(family=family, profile_spec=object()))
    monkeypatch.setattr(
        channel,
        "is_pure_nonmd_v2",
        lambda candidate: candidate.calc.family is CanonicalFamily.DDX_V2,
    )
    monkeypatch.setattr(
        channel,
        "is_pure_torch_calculator",
        lambda candidate: candidate.calc.family in {
            CanonicalFamily.TORCH_V3,
            CanonicalFamily.TORCH_V4,
        },
    )
    monkeypatch.setattr(
        channel,
        "is_pure_torch_v4",
        lambda candidate: candidate.calc.family is CanonicalFamily.TORCH_V4,
    )
    monkeypatch.setattr(
        channel,
        "_all_workflow_ddx",
        lambda images: family is CanonicalFamily.DDX_V1,
    )
    capability = Route2WorkflowCapability(
        "bounded_execution", "single", "task_success", True, "test"
    )
    monkeypatch.setattr(
        channel, "route2_workflow_capability_for_params", lambda *args: capability
    )
    monkeypatch.setattr(
        channel,
        "normalize_v4_sp_request",
        lambda *args: SimpleNamespace(returns_batch=False),
    )
    return image


def test_closed_request_records_have_only_the_approved_fields():
    assert [member.name for member in PreflightMode] == [
        "LEGACY", "DDX_V2", "TORCH_V3", "TORCH_V4"
    ]
    assert [member.name for member in ReportingMode] == [
        "NONE", "NONMD_V2", "WORKFLOW"
    ]
    assert [field.name for field in fields(FamilyPolicy)] == [
        "family", "preflight", "reporting"
    ]
    assert [field.name for field in fields(NormalizedDispatchRequest)] == [
        "task", "method", "frequency_request"
    ]
    assert [field.name for field in fields(ResolvedDispatchRequest)] == [
        "family_policy", "normalized_request", "v4_capability", "sp_shape",
        "threshold_policy",
    ]
    with pytest.raises(AttributeError):
        setattr(_resolved(CanonicalFamily.ORDINARY), "sp_shape", SPShape.SINGLE)


@pytest.mark.parametrize("family", tuple(CanonicalFamily))
def test_trusted_resolver_selects_the_fixed_family_policy(monkeypatch, family):
    image = _patch_family(monkeypatch, family)

    request = channel.resolve_dispatch_request(image, "sp", {})

    expected_preflight, expected_reporting = _EXPECTED_POLICIES[family]
    assert request.family_policy == FamilyPolicy(
        family, expected_preflight, expected_reporting
    )
    assert request.normalized_request.task is DispatchTask.SP


def test_caller_cannot_select_a_dispatch_channel(monkeypatch):
    image = _patch_family(monkeypatch, CanonicalFamily.ORDINARY)
    with pytest.raises(ValueError, match="selection is internal"):
        channel.resolve_dispatch_request(image, "sp", {"channel": "torch_v4"})


def _identity_image(symbols: str) -> Atoms:
    atoms = Atoms(symbols)
    atoms.info.update(charge=0, mult=1)
    atoms.calc = SimpleNamespace(
        _validate_atoms=lambda image: ("a" * 64, "b" * 64, "profile", "scalar", "cpu")
    )
    return atoms


@pytest.mark.parametrize(
    "family",
    (CanonicalFamily.DDX_V2, CanonicalFamily.TORCH_V4),
)
def test_only_homogeneous_route2_families_apply_mixed_identity_guard(
    monkeypatch, family
):
    monkeypatch.setattr(
        channel,
        "calculator_provenance",
        lambda image: {"profile": "p", "scalar": "s", "device": "cpu"},
    )
    capability = Route2WorkflowCapability(
        "bounded_execution", "multi", "path_success", False, "test"
    )
    request = _resolved(family, task=DispatchTask.TS, capability=capability)

    with pytest.raises(ValueError, match="share symbols"):
        channel.preflight_dispatch_request(
            request, [_identity_image("H"), _identity_image("He")], {}
        )


@pytest.mark.parametrize(
    "family", (CanonicalFamily.ORDINARY, CanonicalFamily.DDX_V1)
)
def test_legacy_families_do_not_gain_mixed_identity_preflight(family):
    channel.preflight_dispatch_request(
        _resolved(family), [_identity_image("H"), _identity_image("He")], {}
    )


def test_channel_does_not_import_tasks_or_execute_backend_hooks():
    source = inspect.getsource(channel)
    for forbidden in (
        ".optimization", ".frequency", ".scan", ".ts", ".irc", ".md",
        "_dispatch_legacy", "get_hessian", "get_hvp", "typing import Any",
    ):
        assert forbidden not in source


class _FakeOutcome:
    @classmethod
    def from_internal_status(cls, status, *, capability):
        return cls()

    def to_route2_record(self, outcome_policy):
        return {"task_success": False, "termination_reason": "exception"}


def test_invalid_v4_irc_threshold_writes_one_failure_sidecar_without_executor(
    monkeypatch, tmp_path
):
    import maple.function.dispatcher.pure_nonmd_status as status_module

    capability = Route2WorkflowCapability(
        "bounded_execution", "single", "numerical_irc_auto", False, "test"
    )
    request = _resolved(
        CanonicalFamily.TORCH_V4,
        task=DispatchTask.IRC,
        threshold=ThresholdPolicy.V4_EXPLICIT_OVER_LEVEL,
        capability=capability,
    )
    monkeypatch.setattr(channel, "resolve_dispatch_request", lambda *args: request)
    monkeypatch.setattr(channel, "preflight_dispatch_request", lambda *args: None)
    monkeypatch.setattr(
        status_module,
        "make_status",
        lambda **kwargs: {
            "converged": False,
            "workflow": "irc",
            "final_metrics": {},
            "termination_class": "execution_failure",
            "termination_reason": "exception",
            "iterations": 0,
            "executed": False,
            "method": "irc",
            "geometry": [],
        },
    )
    monkeypatch.setattr(status_module, "calculator_provenance", lambda atoms: {})
    monkeypatch.setattr(status_module, "WorkflowOutcome", _FakeOutcome)
    sidecars = []
    monkeypatch.setattr(
        status_module, "write_status", lambda output, record: sidecars.append(record)
    )
    dispatcher = Dispatcher()
    executor_calls = []
    monkeypatch.setattr(
        dispatcher, "_dispatch_legacy", lambda *args: executor_calls.append(args)
    )
    atoms = Atoms("H")
    params = {
        "f_max_th": -1.0,
        "_route2_irc_threshold_receipt": {"partial": True},
    }

    with pytest.raises(ValueError, match="f_max_th must be a finite positive number"):
        dispatcher(params, "irc", atoms, str(tmp_path / "irc.out"))

    assert executor_calls == []
    assert sidecars == [{"task_success": False, "termination_reason": "exception"}]
    assert "_route2_irc_threshold_receipt" not in params


def test_preflight_failure_has_no_executor_or_sidecar(monkeypatch, tmp_path):
    import maple.function.dispatcher.pure_nonmd_status as status_module

    monkeypatch.setattr(
        channel,
        "resolve_dispatch_request",
        lambda *args: _resolved(CanonicalFamily.DDX_V2),
    )
    monkeypatch.setattr(
        channel,
        "preflight_dispatch_request",
        lambda *args: (_ for _ in ()).throw(ValueError("preflight rejected")),
    )
    sidecars = []
    monkeypatch.setattr(
        status_module, "write_status", lambda *args: sidecars.append(args)
    )
    dispatcher = Dispatcher()
    executor_calls = []
    monkeypatch.setattr(
        dispatcher, "_dispatch_legacy", lambda *args: executor_calls.append(args)
    )

    with pytest.raises(ValueError, match="preflight rejected"):
        dispatcher({}, "sp", Atoms("H"), str(tmp_path / "preflight.out"))

    assert executor_calls == []
    assert sidecars == []


def test_ordinary_request_uses_one_executor_and_keeps_raw_return(monkeypatch, tmp_path):
    import maple.function.dispatcher.legacy_units as legacy_units

    monkeypatch.setattr(
        channel,
        "resolve_dispatch_request",
        lambda *args: _resolved(CanonicalFamily.ORDINARY),
    )
    monkeypatch.setattr(channel, "preflight_dispatch_request", lambda *args: None)

    @contextmanager
    def unchanged_units(atoms):
        yield

    monkeypatch.setattr(legacy_units, "legacy_hartree_job_calculators", unchanged_units)
    dispatcher = Dispatcher()
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        dispatcher,
        "_dispatch_legacy",
        lambda *args: calls.append(args) or sentinel,
    )
    atoms = Atoms("H")

    assert dispatcher({}, "sp", atoms, str(tmp_path / "ordinary.out")) is sentinel
    assert len(calls) == 1
    assert getattr(atoms, "f_max_th") == pytest.approx(0.00285)
