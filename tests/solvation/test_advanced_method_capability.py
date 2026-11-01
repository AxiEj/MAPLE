from types import SimpleNamespace

import pytest

from maple.function.calculator.calculator_base import (
    MethodCapabilityBinding,
    _mint_hessian_precision_binding,
    _mint_method_capability_binding,
)
from maple.function.dispatcher.method_registry import (
    CapabilityKey,
    MethodCapability,
    PublicMethodKey,
    required_capabilities,
    resolve_method_capability,
)
from maple.function.read.command_control import CommandControl


def _calculator(*, capability=True, precision=True):
    calculator = SimpleNamespace()
    if precision:
        calculator._hessian_precision_binding = _mint_hessian_precision_binding(
            profile_id="qualified-aimnet2025-n32-torch-obc2",
            runtime_identity_sha256="1" * 64,
            receipt_sha256="2" * 64,
            composition="torch-obc2",
            composition_identity_sha256="3" * 64,
            owner_token=id(calculator),
        )
    if capability:
        calculator._method_capability_binding = _mint_method_capability_binding(
            capability_keys=frozenset({CapabilityKey.TS_NEB_BASE.value}),
            runtime_identity_sha256="1" * 64,
            receipt_sha256="2" * 64,
            composition_identity_sha256="3" * 64,
            owner_token=id(calculator),
        )
    return calculator


def test_all_canonical_keys_are_typed_and_resolvable_from_exact_binding():
    capability = resolve_method_capability(
        CapabilityKey.TS_NEB_BASE,
        _calculator(),
    )
    assert isinstance(capability, MethodCapability)
    assert capability.key is CapabilityKey.TS_NEB_BASE


def test_registry_only_is_insufficient():
    assert required_capabilities(PublicMethodKey.TS_NEB, {}) == {
        CapabilityKey.TS_NEB_BASE,
    }
    with pytest.raises(ValueError, match="does not admit"):
        resolve_method_capability(CapabilityKey.TS_NEB_BASE, _calculator(capability=False))


def test_receipt_only_is_insufficient_without_matching_derivative_identity():
    with pytest.raises(ValueError, match="derivative identity"):
        resolve_method_capability(CapabilityKey.TS_NEB_BASE, _calculator(precision=False))


def test_forged_binding_refuses():
    calculator = _calculator(capability=False)
    calculator._method_capability_binding = MethodCapabilityBinding(
        capability_keys=frozenset({CapabilityKey.TS_NEB_BASE.value}),
        runtime_identity_sha256="1" * 64,
        receipt_sha256="2" * 64,
        composition_identity_sha256="3" * 64,
        owner_token=id(calculator),
        _authority=object(),
    )
    with pytest.raises(ValueError, match="does not admit"):
        resolve_method_capability(CapabilityKey.TS_NEB_BASE, calculator)


def test_software_capability_is_not_ts_certification():
    capability = resolve_method_capability(CapabilityKey.TS_NEB_BASE, _calculator())
    assert not hasattr(capability, "certified_ts")


def _implicit_settings(task_line):
    return [
        "#model=aimnet2025(hessian=analytic)",
        task_line,
        "#charge(source=mol2,mode=fixed,geometry=keep)",
        "#solv(implicit=water,method=gb,provider=torch,model=obc2,"
        "profile=obc2-mbondi2,nonpolar=ace,experimental=true)",
    ]


@pytest.mark.parametrize(
    "task_line,method",
    [
        ("#irc(method=hpc,max_steps=0)", "hpc"),
        ("#irc(method=lqa,max_steps=0)", "lqa"),
        ("#irc(method=eulerpc,max_steps=0)", "eulerpc"),
        ("#ts(method=neb,n_images=1,max_iter=0,refine=nebts,cineb_max_iter=0,prfo_max_iter=0)", "neb"),
        ("#ts(method=string,n_images=3,max_iter_grow=0,refine=stringts,max_iter_relax=0,max_iter_cistring=0)", "string"),
        ("#ts(method=autoneb,n_max=3,max_iter=0,climb=false)", "autoneb"),
    ],
)
def test_command_boundary_accepts_each_advanced_method_schema(task_line, method):
    control = CommandControl.from_settings(_implicit_settings(task_line))
    assert control.params["method"] == method


@pytest.mark.parametrize("method", ["hpc", "lqa", "eulerpc"])
def test_irc_unknown_parameter_rejects_before_model_work(method):
    with pytest.raises(ValueError, match="Unknown IRC parameter"):
        CommandControl.from_settings(
            _implicit_settings(f"#irc(method={method},unknown_control=1)")
        )
