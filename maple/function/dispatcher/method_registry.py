"""Canonical public method, parameter, arity, and capability contracts."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from numbers import Real
from types import MappingProxyType

import numpy as np

from .convergence import CONVERGENCE_CONTEXT_KEYS


class PublicMethodKey(StrEnum):
    IRC_GS = "irc.gs"
    IRC_HPC = "irc.hpc"
    IRC_LQA = "irc.lqa"
    IRC_EULERPC = "irc.eulerpc"
    TS_PRFO = "ts.prfo"
    TS_DIMER = "ts.dimer"
    TS_NEB = "ts.neb"
    TS_STRING = "ts.string"
    TS_AUTONEB = "ts.autoneb"


class CapabilityKey(StrEnum):
    IRC_GS = "irc.gs"
    IRC_HPC = "irc.hpc"
    IRC_LQA = "irc.lqa"
    IRC_EULERPC = "irc.eulerpc"
    TS_PRFO = "ts.prfo"
    TS_NEB_BASE = "ts.neb.base"
    TS_NEB_CI = "ts.neb.ci"
    TS_NEB_TS = "ts.neb.ts"
    TS_STRING_BASE = "ts.string.base"
    TS_STRING_CI = "ts.string.ci"
    TS_STRING_TS = "ts.string.ts"
    TS_AUTONEB_BASE = "ts.autoneb.base"
    TS_AUTONEB_CI = "ts.autoneb.ci"
    FREQUENCY_ANALYTIC = "frequency.analytic"


@dataclass(frozen=True)
class InputArity:
    minimum: int
    maximum: int | None = None

    def validate(self, public_key: PublicMethodKey, count: int) -> None:
        words = {1: "one", 2: "two"}
        minimum = words.get(self.minimum, str(self.minimum))
        if type(count) is not int or count < self.minimum:
            if self.maximum == self.minimum:
                raise ValueError(
                    f"{public_key.value} requires exactly {minimum} input geometries."
                )
            raise ValueError(
                f"{public_key.value} requires at least {minimum} input geometries."
            )
        if self.maximum is not None and count > self.maximum:
            if self.minimum == self.maximum:
                raise ValueError(
                    f"{public_key.value} requires exactly {minimum} input geometries."
                )
            raise ValueError(
                f"{public_key.value} accepts at most {self.maximum} input geometries."
            )


@dataclass(frozen=True)
class ParameterRule:
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    choices: frozenset | None = None
    active_refinements: frozenset[object] | None = None


@dataclass(frozen=True)
class MethodSpec:
    public_key: PublicMethodKey
    task: str
    method: str
    parameter_dataclass: Callable[[], type]
    parameter_rules: Mapping[str, ParameterRule]
    aliases: Mapping[str, str]
    input_arity: InputArity
    refinements: frozenset[object]
    capability_resolver: Callable[[Mapping[str, object]], frozenset[CapabilityKey]]


@dataclass(frozen=True)
class MethodCapability:
    key: CapabilityKey
    receipt_sha256: str
    runtime_identity_sha256: str
    composition_identity_sha256: str


GLOBAL_PARAMETER_KEYS = frozenset({
    "model", "model_options", "device", "gpuid", "d4", "pbc", "solv",
    "charge", "level", "task", "method",
})


def _rule(kind: str, **kwargs) -> ParameterRule:
    return ParameterRule(kind=kind, **kwargs)


POSITIVE_REAL = _rule("real", minimum=0.0, minimum_inclusive=False)
NONNEGATIVE_REAL = _rule("real", minimum=0.0)
POSITIVE_INT = _rule("int", minimum=1)
NONNEGATIVE_INT = _rule("int", minimum=0)
BOOLEAN = _rule("bool")
OPTIONAL_NONNEGATIVE_INT = _rule("optional_int", minimum=0)
HESSIAN_UPDATE = _rule("enum", choices=frozenset({"bfgs", "bofill"}))


def _gs_type():
    from .irc.algorithm.gs import GSParams
    return GSParams


def _hpc_type():
    from .irc.algorithm.hpc import HPCParams
    return HPCParams


def _lqa_type():
    from .irc.algorithm.lqa import LQAParams
    return LQAParams


def _eulerpc_type():
    from .irc.algorithm.eulerpc import EulerPCParams
    return EulerPCParams


def _prfo_type():
    from .ts.algorithm.PRFO import PRFOParams
    return PRFOParams


def _dimer_type():
    from .ts.algorithm.dimer import DimerParams
    return DimerParams


def _neb_type():
    from .ts.algorithm.neb import NEBParams
    return NEBParams


def _string_type():
    from .ts.algorithm.string import GSMParams
    return GSMParams


def _autoneb_type():
    from .ts.algorithm.autoneb import AutoNEBParams
    return AutoNEBParams


IRC_ALIASES = MappingProxyType({
    "sd_len_bohr": "step_length_bohr",
    "steplength_bohr": "step_length_bohr",
    "max_points": "max_steps",
    "tol_maxf": "f_max_th",
    "tol_rmsf": "f_rms_th",
})


IRC_COMMON = {
    "target_mode": POSITIVE_INT,
    "step_length_bohr": POSITIVE_REAL,
    "max_steps": NONNEGATIVE_INT,
    "hessian_recalc": OPTIONAL_NONNEGATIVE_INT,
    "hessian_update": HESSIAN_UPDATE,
    "f_max_th": POSITIVE_REAL,
    "f_rms_th": POSITIVE_REAL,
    "print_each": BOOLEAN,
    "write_traj": BOOLEAN,
}


GS_RULES = {
    **IRC_COMMON,
    "max_micro_cycles": NONNEGATIVE_INT,
    "micro_step_thresh": POSITIVE_REAL,
    "project_rigid_modes": BOOLEAN,
}


HPC_RULES = {
    **IRC_COMMON,
    "euler_n": POSITIVE_INT,
    "dwi_n": POSITIVE_INT,
    "mbs_max_k": _rule("int", minimum=2),
    "mbs_points": _rule("int", minimum=2),
    "mbs_tol": POSITIVE_REAL,
}


LQA_RULES = {**IRC_COMMON, "euler_n": POSITIVE_INT}


EULERPC_RULES = {
    **IRC_COMMON,
    "max_pred_steps": POSITIVE_INT,
    "loose_cycles": NONNEGATIVE_INT,
    "dwi_n": POSITIVE_INT,
    "mbs_max_k": _rule("int", minimum=2),
    "mbs_points": _rule("int", minimum=2),
    "mbs_tol": POSITIVE_REAL,
}


PRFO_RULES = {
    "max_iter": NONNEGATIVE_INT,
    "trust_radius": POSITIVE_REAL,
    "trust_min": POSITIVE_REAL,
    "trust_max": POSITIVE_REAL,
    "eta_shrink": POSITIVE_REAL,
    "eta_expand": POSITIVE_REAL,
    "evals_eps": POSITIVE_REAL,
    "mu_margin": POSITIVE_REAL,
    "max_bisect_it": POSITIVE_INT,
    "recalc": POSITIVE_INT,
    "hessian_update": HESSIAN_UPDATE,
    "project_rigid_modes": BOOLEAN,
    "f_max_th": NONNEGATIVE_REAL,
    "f_rms_th": NONNEGATIVE_REAL,
    "dp_max_th": NONNEGATIVE_REAL,
    "dp_rms_th": NONNEGATIVE_REAL,
}


DIMER_RULES = {
    "use_hvp": _rule("optional_bool"),
    "delta": POSITIVE_REAL,
    "rot_max_iter": NONNEGATIVE_INT,
    "rot_alpha": POSITIVE_REAL,
    "rot_f_max_th": POSITIVE_REAL,
    "rot_f_rms_th": POSITIVE_REAL,
    "step0": POSITIVE_REAL,
    "step_max": POSITIVE_REAL,
    "trust_radius": POSITIVE_REAL,
    "f_max_th": POSITIVE_REAL,
    "f_rms_th": POSITIVE_REAL,
    "dp_max_th": POSITIVE_REAL,
    "dp_rms_th": POSITIVE_REAL,
    "kappa_to_flip": _rule("real"),
    "max_iter": NONNEGATIVE_INT,
    "use_mass_weight": BOOLEAN,
    "remove_rigid": BOOLEAN,
    "n_init": _rule("enum", choices=frozenset({"random", "force", "given"})),
    "n_given": _rule("any"),
    "save_traj": BOOLEAN,
    "save_metrics": BOOLEAN,
}


NEB_RULES = {
    "n_images": POSITIVE_INT,
    "k_min": POSITIVE_REAL,
    "k_max": POSITIVE_REAL,
    "use_dynamic_k": BOOLEAN,
    "k_decay": POSITIVE_REAL,
    "max_iter": NONNEGATIVE_INT,
    "lbfgs_m": POSITIVE_INT,
    "step0": POSITIVE_REAL,
    "ifidpp": _rule("int", choices=frozenset({0, 1})),
    "neb_f_max_th": POSITIVE_REAL,
    "neb_f_rms_th": POSITIVE_REAL,
    "initial_opt": BOOLEAN,
    "refine": _rule("optional_enum", choices=frozenset({"cineb", "nebts"})),
    "cineb_f_max_th": _rule("real", minimum=0.0, minimum_inclusive=False,
                             active_refinements=frozenset({"cineb", "nebts"})),
    "cineb_f_rms_th": _rule("real", minimum=0.0, minimum_inclusive=False,
                             active_refinements=frozenset({"cineb", "nebts"})),
    "cilbfgs_m": _rule("int", minimum=1,
                        active_refinements=frozenset({"cineb", "nebts"})),
    "cistep0": _rule("real", minimum=0.0, minimum_inclusive=False,
                      active_refinements=frozenset({"cineb", "nebts"})),
    "cineb_max_iter": _rule("optional_int", minimum=0,
                             active_refinements=frozenset({"cineb", "nebts"})),
    "ci_displacement_tol": _rule("optional_real", minimum=0.0,
                                  active_refinements=frozenset({"cineb", "nebts"})),
    "prfo_max_iter": _rule("int", minimum=0, active_refinements=frozenset({"nebts"})),
    "prfo_project_rigid_modes": _rule("bool", active_refinements=frozenset({"nebts"})),
}


STRING_RULES = {
    "grow_step": POSITIVE_REAL,
    "grow_step_min": POSITIVE_REAL,
    "grow_backtrack": _rule("real", minimum=0.0, maximum=1.0,
                             minimum_inclusive=False, maximum_inclusive=False),
    "grow_relax_iters": NONNEGATIVE_INT,
    "grow_relax_tol": POSITIVE_REAL,
    "grow_energy_tol": POSITIVE_REAL,
    "pause_imbalance": NONNEGATIVE_INT,
    "refine_end_every": NONNEGATIVE_INT,
    "merge_threshold": POSITIVE_REAL,
    "n_images": _rule("int", minimum=2),
    "max_iter_relax": _rule("int", minimum=0,
                            active_refinements=frozenset({"cistring", "stringts"})),
    "string_f_max_th": _rule("real", minimum=0.0, minimum_inclusive=False,
                              active_refinements=frozenset({"cistring", "stringts"})),
    "string_f_rms_th": _rule("real", minimum=0.0, minimum_inclusive=False,
                              active_refinements=frozenset({"cistring", "stringts"})),
    "lbfgs_memory": _rule("int", minimum=0,
                          active_refinements=frozenset({"cistring", "stringts"})),
    "lbfgs_curvature": _rule("real", minimum=0.0, minimum_inclusive=False,
                             active_refinements=frozenset({"cistring", "stringts"})),
    "lbfgs_max_step": _rule("real", minimum=0.0, minimum_inclusive=False,
                            active_refinements=frozenset({"cistring", "stringts"})),
    "reparam_every": _rule("int", minimum=1,
                           active_refinements=frozenset({"cistring", "stringts"})),
    "refine": _rule("optional_enum", choices=frozenset({"cistring", "stringts"})),
    "max_iter_cistring": _rule("int", minimum=0,
                               active_refinements=frozenset({"cistring", "stringts"})),
    "cistring_f_max_th": _rule("real", minimum=0.0, minimum_inclusive=False,
                                active_refinements=frozenset({"cistring", "stringts"})),
    "cistring_f_rms_th": _rule("real", minimum=0.0, minimum_inclusive=False,
                                active_refinements=frozenset({"cistring", "stringts"})),
    "prfo_params": _rule("mapping", active_refinements=frozenset({"stringts"})),
    "project_rigid_modes": _rule("bool", active_refinements=frozenset({"stringts"})),
    "max_iter_grow": NONNEGATIVE_INT,
    "log_every": POSITIVE_INT,
    "guide_conn_min": _rule("real", minimum=0.0, maximum=1.0),
    "guide_d_switch": POSITIVE_REAL,
    "guide_inc_trigger": NONNEGATIVE_INT,
}


AUTONEB_RULES = {
    "n_max": _rule("int", minimum=3),
    "max_iter": NONNEGATIVE_INT,
    "fmax": POSITIVE_REAL,
    "rms_fmax": POSITIVE_REAL,
    "climb": BOOLEAN,
    "ci_max_iter": _rule("int", minimum=0, active_refinements=frozenset({True})),
    "ci_fmax": _rule("real", minimum=0.0, minimum_inclusive=False,
                      active_refinements=frozenset({True})),
    "ci_rms_fmax": _rule("real", minimum=0.0, minimum_inclusive=False,
                          active_refinements=frozenset({True})),
    "ci_displacement_tol": _rule("real", minimum=0.0, minimum_inclusive=False,
                                  active_refinements=frozenset({True})),
    "space_energy_ratio": POSITIVE_REAL,
    "interpolation": _rule("enum", choices=frozenset({"linear", "idpp"})),
    "verbose": NONNEGATIVE_INT,
}


def _fixed(*keys: CapabilityKey):
    return lambda _params: frozenset(keys)


def _neb_capabilities(params):
    result = {CapabilityKey.TS_NEB_BASE}
    if params.get("refine") in {"cineb", "nebts"}:
        result.add(CapabilityKey.TS_NEB_CI)
    if params.get("refine") == "nebts":
        result.add(CapabilityKey.TS_NEB_TS)
    return frozenset(result)


def _string_capabilities(params):
    result = {CapabilityKey.TS_STRING_BASE}
    if params.get("refine") in {"cistring", "stringts"}:
        result.add(CapabilityKey.TS_STRING_CI)
    if params.get("refine") == "stringts":
        result.add(CapabilityKey.TS_STRING_TS)
    return frozenset(result)


def _autoneb_capabilities(params):
    result = {CapabilityKey.TS_AUTONEB_BASE}
    if params.get("climb", True):
        result.add(CapabilityKey.TS_AUTONEB_CI)
    return frozenset(result)


def _spec(key, task, method, parameter_dataclass, rules, aliases, arity,
          refinements, capability_resolver):
    return MethodSpec(
        key, task, method, parameter_dataclass, MappingProxyType(rules), aliases,
        arity, frozenset(refinements), capability_resolver,
    )


METHOD_SPECS = MappingProxyType({
    PublicMethodKey.IRC_GS: _spec(PublicMethodKey.IRC_GS, "irc", "gs", _gs_type,
        GS_RULES, IRC_ALIASES, InputArity(1, 1), (), _fixed(CapabilityKey.IRC_GS)),
    PublicMethodKey.IRC_HPC: _spec(PublicMethodKey.IRC_HPC, "irc", "hpc", _hpc_type,
        HPC_RULES, IRC_ALIASES, InputArity(1, 1), (), _fixed(CapabilityKey.IRC_HPC)),
    PublicMethodKey.IRC_LQA: _spec(PublicMethodKey.IRC_LQA, "irc", "lqa", _lqa_type,
        LQA_RULES, IRC_ALIASES, InputArity(1, 1), (), _fixed(CapabilityKey.IRC_LQA)),
    PublicMethodKey.IRC_EULERPC: _spec(PublicMethodKey.IRC_EULERPC, "irc", "eulerpc", _eulerpc_type,
        EULERPC_RULES, IRC_ALIASES, InputArity(1, 1), (), _fixed(CapabilityKey.IRC_EULERPC)),
    PublicMethodKey.TS_PRFO: _spec(PublicMethodKey.TS_PRFO, "ts", "prfo", _prfo_type,
        PRFO_RULES, MappingProxyType({}), InputArity(1, 1), (), _fixed(CapabilityKey.TS_PRFO)),
    PublicMethodKey.TS_DIMER: _spec(PublicMethodKey.TS_DIMER, "ts", "dimer", _dimer_type,
        DIMER_RULES, MappingProxyType({}), InputArity(1, 1), (), _fixed()),
    PublicMethodKey.TS_NEB: _spec(PublicMethodKey.TS_NEB, "ts", "neb", _neb_type,
        NEB_RULES, MappingProxyType({}), InputArity(2), {"cineb", "nebts"}, _neb_capabilities),
    PublicMethodKey.TS_STRING: _spec(PublicMethodKey.TS_STRING, "ts", "string", _string_type,
        STRING_RULES, MappingProxyType({}), InputArity(2, 2), {"cistring", "stringts"}, _string_capabilities),
    PublicMethodKey.TS_AUTONEB: _spec(PublicMethodKey.TS_AUTONEB, "ts", "autoneb", _autoneb_type,
        AUTONEB_RULES, MappingProxyType({}), InputArity(2), (), _autoneb_capabilities),
})


def implemented_methods(task: str) -> frozenset[str]:
    return frozenset(spec.method for spec in METHOD_SPECS.values() if spec.task == task)


def normalize_public_method(task: str, method: object) -> PublicMethodKey:
    if not isinstance(method, str):
        raise TypeError(f"{task} method must be a string.")
    normalized = method.lower()
    for key, spec in METHOD_SPECS.items():
        if spec.task == task and spec.method == normalized:
            return key
    raise ValueError(f"Method {method!r} not implemented for task {task!r}.")


def method_spec(public_key: PublicMethodKey | str) -> MethodSpec:
    key = PublicMethodKey(public_key)
    return METHOD_SPECS[key]


def validate_input_arity(public_key: PublicMethodKey | str, count: int) -> None:
    key = PublicMethodKey(public_key)
    METHOD_SPECS[key].input_arity.validate(key, count)


def allowed_refinements(task: str, method: object) -> frozenset[object]:
    return method_spec(normalize_public_method(task, method)).refinements


def parameter_names(task: str, method: object) -> frozenset[str]:
    spec = method_spec(normalize_public_method(task, method))
    return frozenset(spec.parameter_rules) | frozenset(spec.aliases)


def _keyed_items(mapping, location):
    items = []
    origins = {}
    for raw_key, value in mapping.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"{location} parameter keys must be strings")
        key = raw_key.lower()
        if key in origins:
            raise ValueError(
                f"duplicate {location} parameter {raw_key!r} conflicts with {origins[key]!r}"
            )
        origins[key] = raw_key
        items.append((raw_key, key, value))
    return items


def _normalized_parameter_items(spec: MethodSpec, params) -> list[tuple[str, object]]:
    if params is None:
        return []
    if not isinstance(params, Mapping):
        raise TypeError(f"{spec.method} parameters must be a mapping or None")
    outer = _keyed_items(params, spec.method)
    public_methods = implemented_methods(spec.task)
    group_keys = public_methods | {spec.task}
    groups = [(raw, key, value) for raw, key, value in outer if key in group_keys]
    if len(groups) > 1:
        raise ValueError("multiple method parameter groups were provided: " +
                         ", ".join(raw for raw, _key, _value in groups))
    if groups and groups[0][1] not in {spec.method, spec.task}:
        raise ValueError(
            f"parameter group {groups[0][0]!r} does not match method {spec.method!r}"
        )
    present_context = {key for _raw, key, _value in outer if key in CONVERGENCE_CONTEXT_KEYS}
    convergence_context = CONVERGENCE_CONTEXT_KEYS if present_context == CONVERGENCE_CONTEXT_KEYS else frozenset()
    context_keys = GLOBAL_PARAMETER_KEYS | convergence_context
    if groups:
        raw_group, _group_key, nested = groups[0]
        if not isinstance(nested, Mapping):
            raise TypeError(f"{raw_group} parameters must be a mapping")
        flat = [raw for raw, key, _value in outer
                if key not in context_keys and key not in group_keys]
        if flat:
            raise ValueError(
                f"{spec.method} request mixes flat parameters with nested group {raw_group!r}: "
                + ", ".join(flat)
            )
        selected = _keyed_items(nested, f"{spec.method} nested")
        nested_fields = {spec.aliases.get(key, key) for _raw, key, _value in selected}
        selected = [(raw, key, value) for raw, key, value in outer
                    if key in convergence_context
                    and spec.aliases.get(key, key) in spec.parameter_rules
                    and spec.aliases.get(key, key) not in nested_fields] + selected
    else:
        selected = outer
    normalized = []
    origins = {}
    for raw_key, key, value in selected:
        canonical = spec.aliases.get(key, key)
        if key in context_keys and canonical not in spec.parameter_rules:
            continue
        if canonical in origins:
            raise ValueError(
                f"duplicate {spec.method} parameter {raw_key!r} resolves to {canonical!r}, "
                f"already provided by {origins[canonical]!r}"
            )
        origins[canonical] = raw_key
        normalized.append((canonical, value))
    return normalized


def _validate_value(name: str, value, rule: ParameterRule) -> None:
    kind = rule.kind
    if kind == "any":
        return
    if kind in {"bool", "optional_bool"}:
        if kind == "optional_bool" and value is None:
            return
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean")
    elif kind in {"int", "optional_int"}:
        if kind == "optional_int" and value is None:
            return
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    elif kind in {"real", "optional_real"}:
        if kind == "optional_real" and value is None:
            return
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    elif kind in {"enum", "optional_enum"}:
        if kind == "optional_enum" and value is None:
            return
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
    elif kind == "mapping":
        if value is None:
            return
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
    else:
        raise RuntimeError(f"Unknown parameter rule kind: {kind}")
    if rule.choices is not None and value not in rule.choices:
        raise ValueError(f"{name} must be one of: " +
                         ", ".join(str(item) for item in sorted(rule.choices, key=str)))
    if value is None or kind in {"bool", "optional_bool", "enum", "optional_enum", "mapping"}:
        return
    if rule.minimum is not None:
        invalid = value < rule.minimum if rule.minimum_inclusive else value <= rule.minimum
        if invalid:
            raise ValueError(f"{name} is below its allowed minimum")
    if rule.maximum is not None:
        invalid = value > rule.maximum if rule.maximum_inclusive else value >= rule.maximum
        if invalid:
            raise ValueError(f"{name} is above its allowed maximum")


def _validate_prfo_values(values: Mapping[str, object]) -> None:
    if values["trust_min"] > values["trust_radius"]:
        raise ValueError("prfo.trust_min must not exceed prfo.trust_radius")
    if values["trust_radius"] > values["trust_max"]:
        raise ValueError("prfo.trust_radius must not exceed prfo.trust_max")


def validate_method_parameters(public_key: PublicMethodKey | str, params) -> dict:
    spec = method_spec(public_key)
    params_class = spec.parameter_dataclass()
    declared = {field.name for field in fields(params_class)}
    if set(spec.parameter_rules) != declared:
        raise RuntimeError(
            f"{spec.public_key.value} parameter rules do not match {params_class.__name__}"
        )
    normalized = dict(_normalized_parameter_items(spec, params))
    unknown = sorted(set(normalized) - set(spec.parameter_rules))
    if unknown:
        raise ValueError(f"Unknown {spec.method} parameter: {unknown[0]}")
    defaults = params_class()
    selector = normalized.get("refine") if "refine" in spec.parameter_rules else None
    if spec.public_key is PublicMethodKey.TS_AUTONEB:
        selector = normalized.get("climb", defaults.climb)
    for key, value in normalized.items():
        rule = spec.parameter_rules[key]
        if rule.active_refinements is not None and selector not in rule.active_refinements:
            if value != getattr(defaults, key):
                raise ValueError(f"{spec.method}.{key} is inactive for the selected workflow")
        _validate_value(f"{spec.method}.{key}", value, rule)
    values = {field.name: normalized.get(field.name, getattr(defaults, field.name))
              for field in fields(params_class)}
    if spec.public_key is PublicMethodKey.TS_PRFO:
        _validate_prfo_values(values)
    if spec.public_key in {PublicMethodKey.IRC_HPC, PublicMethodKey.IRC_EULERPC}:
        if values["dwi_n"] % 2:
            raise ValueError(f"{spec.method}.dwi_n must be a positive even integer")
    if spec.public_key is PublicMethodKey.TS_NEB and values["k_min"] > values["k_max"]:
        raise ValueError("neb.k_min must not exceed neb.k_max")
    if spec.public_key is PublicMethodKey.TS_STRING:
        if values["grow_step_min"] > values["grow_step"]:
            raise ValueError("string.grow_step_min must not exceed string.grow_step")
        if values["refine"] is not None and values["n_images"] < 3:
            raise ValueError("refined String requires n_images >= 3")
        if values["refine"] is not None and values["reparam_every"] != 1:
            raise ValueError("refined String requires reparam_every=1")
        if values["prfo_params"]:
            nested_items = _keyed_items(values["prfo_params"], "String prfo_params")
            nested_values = {key: value for _raw, key, value in nested_items}
            unknown_prfo = sorted(set(nested_values) - set(PRFO_RULES))
            if unknown_prfo:
                raise ValueError(f"Unknown prfo parameter: {unknown_prfo[0]}")
            nested_prfo = validate_method_parameters(
                PublicMethodKey.TS_PRFO,
                nested_values,
            )
            if "project_rigid_modes" in nested_prfo:
                raise ValueError(
                    "String project_rigid_modes must not be duplicated in prfo_params"
                )
            normalized["prfo_params"] = nested_prfo
    if spec.public_key is PublicMethodKey.TS_AUTONEB and values["climb"] and values["n_max"] < 3:
        raise ValueError("AutoNEB climb=true requires n_max >= 3")
    return normalized


def required_capabilities(public_key: PublicMethodKey | str,
                          normalized_params: Mapping[str, object]) -> frozenset[CapabilityKey]:
    return method_spec(public_key).capability_resolver(normalized_params)


def resolve_method_capability(
    capability_key: CapabilityKey | str,
    calculator,
) -> MethodCapability:
    from maple.function.calculator.calculator_base import (
        _is_factory_method_capability_binding,
    )

    key = CapabilityKey(capability_key)
    binding = getattr(calculator, "_method_capability_binding", None)
    if (
        not _is_factory_method_capability_binding(binding)
        or binding.owner_token != id(calculator)
        or key.value not in binding.capability_keys
    ):
        raise ValueError(f"The live factory binding does not admit {key.value}.")
    precision = getattr(calculator, "_hessian_precision_binding", None)
    if (
        precision is None
        or precision.owner_token != binding.owner_token
        or precision.receipt_sha256 != binding.receipt_sha256
        or precision.runtime_identity_sha256 != binding.runtime_identity_sha256
        or precision.composition_identity_sha256 != binding.composition_identity_sha256
    ):
        raise ValueError("The method capability binding does not match live derivative identity.")
    return MethodCapability(
        key=key,
        receipt_sha256=binding.receipt_sha256,
        runtime_identity_sha256=binding.runtime_identity_sha256,
        composition_identity_sha256=binding.composition_identity_sha256,
    )
