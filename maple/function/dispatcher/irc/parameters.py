"""Typed, method-aware parameter contract for legacy IRC integrators.

The numerical algorithms remain separate, but their input schema is owned in
one place so command parsing and direct Python use cannot silently disagree.
Validation normalizes harmless scalar representations (for example ``50.0``
to integer ``50``) before an integrator reaches ``range`` or a divisor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping, Optional

from .preflight import IRCPreflightParams, validate_irc_preflight_params


PATH_ENERGY_ABSOLUTE_TOLERANCE_HARTREE = 1.0e-7
DEFAULT_IRC_MAX_STEPS = 50
DEFAULT_IRC_MAX_FORCE_HARTREE_PER_ANGSTROM = 2.0e-3
DEFAULT_IRC_RMS_FORCE_HARTREE_PER_ANGSTROM = 5.0e-4


# These are historical input spellings only. Canonical field names are taken
# directly from the dataclasses below and therefore never duplicated here.
IRC_PARAMETER_ALIASES = {
    "sd_len_bohr": "step_length_bohr",
    "steplength_bohr": "step_length_bohr",
    "max_points": "max_steps",
    "tol_maxf": "f_max_th",
    "tol_rmsf": "f_rms_th",
}


@dataclass
class IRCPathParams(IRCPreflightParams):
    """Scientific settings shared by every bidirectional IRC integrator."""

    # These are MAPLE's established Cartesian legacy-job units. They are not
    # aliases for another program's numerically identical atomic-unit values.
    step_length_bohr: float = 0.10
    max_steps: int = DEFAULT_IRC_MAX_STEPS
    f_max_th: float = DEFAULT_IRC_MAX_FORCE_HARTREE_PER_ANGSTROM
    f_rms_th: float = DEFAULT_IRC_RMS_FORCE_HARTREE_PER_ANGSTROM
    path_energy_tolerance_hartree: float = PATH_ENERGY_ABSOLUTE_TOLERANCE_HARTREE
    print_each: bool = True
    write_traj: bool = True
    require_converged_endpoints: bool = True


@dataclass
class GSParams(IRCPathParams):
    max_micro_cycles: int = 20
    micro_step_thresh: float = 1.0e-3
    hessian_recalc: Optional[int] = None
    hessian_update: str = "bofill"


@dataclass
class LQAParams(IRCPathParams):
    euler_n: int = 5000
    hessian_recalc: Optional[int] = None
    hessian_update: str = "bofill"


@dataclass
class HPCParams(IRCPathParams):
    euler_n: int = 5000
    hessian_recalc: Optional[int] = None
    hessian_update: str = "bofill"
    dwi_n: int = 4
    mbs_max_k: int = 15
    mbs_points: int = 20
    mbs_tol: float = 1.0e-5


@dataclass
class EulerPCParams(IRCPathParams):
    max_pred_steps: int = 500
    loose_cycles: int = 3
    hessian_recalc: Optional[int] = None
    hessian_update: str = "bofill"
    dwi_n: int = 4
    mbs_max_k: int = 15
    mbs_points: int = 20
    mbs_tol: float = 1.0e-5


IRC_METHOD_PARAM_TYPES = {
    "gs": GSParams,
    "lqa": LQAParams,
    "hpc": HPCParams,
    "eulerpc": EulerPCParams,
}


def coerce_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return converted


def coerce_nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return converted


def coerce_integer_at_least(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if not math.isfinite(numeric) or numeric != converted or converted < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return converted


def _optional_positive_integer(value: object, name: str) -> Optional[int]:
    if value is None:
        return None
    return coerce_integer_at_least(value, name, 1)


def coerce_strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false.")
    return value


def _hessian_update(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("hessian_update must be 'bfgs' or 'bofill'.")
    normalized = value.strip().lower()
    if normalized not in {"bfgs", "bofill"}:
        raise ValueError("hessian_update must be 'bfgs' or 'bofill'.")
    return normalized


def _positive_even_integer(value: object, name: str) -> int:
    converted = coerce_integer_at_least(value, name, 1)
    if converted % 2:
        raise ValueError(f"{name} must be a positive even integer.")
    return converted


def normalize_irc_parameter_mapping(
    values: Mapping[str, object],
) -> dict[str, object]:
    """Return canonical IRC keys and reject alias/canonical ambiguity."""

    normalized: dict[str, object] = {}
    sources: dict[str, str] = {}
    for raw_key, value in values.items():
        key = str(raw_key).strip().lower()
        canonical = IRC_PARAMETER_ALIASES.get(key, key)
        if canonical in normalized:
            previous = sources[canonical]
            raise ValueError(
                f"IRC parameter '{canonical}' was supplied more than once "
                f"via '{previous}' and '{key}'."
            )
        normalized[canonical] = value
        sources[canonical] = key
    return normalized


def irc_parameter_names(method: str) -> set[str]:
    """Return canonical parameter names for one method or all known methods."""

    normalized = str(method).strip().lower()
    param_type = IRC_METHOD_PARAM_TYPES.get(normalized)
    if param_type is not None:
        return {field.name for field in fields(param_type)}
    return {
        field.name
        for known_type in IRC_METHOD_PARAM_TYPES.values()
        for field in fields(known_type)
    }


def irc_params_from_mapping(method: str, values: Mapping[str, object]) -> IRCPathParams:
    """Construct one method-specific parameter object from canonical values."""

    normalized_method = str(method).strip().lower()
    try:
        param_type = IRC_METHOD_PARAM_TYPES[normalized_method]
    except KeyError as exc:
        raise ValueError(f"Unknown IRC method: '{method}'.") from exc

    params = param_type()
    allowed = irc_parameter_names(normalized_method)
    for key, value in normalize_irc_parameter_mapping(values).items():
        if key in allowed:
            setattr(params, key, value)
    return params


def apply_irc_parameter_overrides(
    params: IRCPathParams,
    values: Mapping[str, object],
) -> None:
    """Apply known direct-API overrides while ignoring enclosing job settings."""

    allowed = {field.name for field in fields(type(params))}
    for key, value in normalize_irc_parameter_mapping(values).items():
        if key in allowed:
            setattr(params, key, value)


def select_irc_parameter_overrides(
    values: Mapping[str, object],
    method: str,
) -> Mapping[str, object]:
    """Select legacy overrides, validating explicit nested method blocks."""

    lowered = {str(key).strip().lower(): value for key, value in values.items()}
    for key in (str(method).strip().lower(), "irc"):
        nested = lowered.get(key)
        if isinstance(nested, Mapping):
            normalized = normalize_irc_parameter_mapping(nested)
            unknown = set(normalized) - irc_parameter_names(method)
            if unknown:
                unknown_key = sorted(unknown)[0]
                raise ValueError(
                    f"Unknown {str(method).upper()} IRC parameter: "
                    f"'{unknown_key}'."
                )
            return normalized
    return lowered


def validate_irc_path_params(params: IRCPathParams) -> None:
    """Normalize and validate common path-execution parameters in place."""

    params.step_length_bohr = coerce_positive_float(
        params.step_length_bohr,
        "step_length_bohr",
    )
    params.max_steps = coerce_integer_at_least(params.max_steps, "max_steps", 1)
    params.f_max_th = coerce_positive_float(params.f_max_th, "f_max_th")
    params.f_rms_th = coerce_positive_float(params.f_rms_th, "f_rms_th")
    params.path_energy_tolerance_hartree = coerce_positive_float(
        params.path_energy_tolerance_hartree,
        "path_energy_tolerance_hartree",
    )
    params.print_each = coerce_strict_bool(params.print_each, "print_each")
    params.write_traj = coerce_strict_bool(params.write_traj, "write_traj")
    params.require_converged_endpoints = coerce_strict_bool(
        params.require_converged_endpoints,
        "require_converged_endpoints",
    )


def _validate_hessian_controls(params: object) -> None:
    params.hessian_recalc = _optional_positive_integer(
        params.hessian_recalc,
        "hessian_recalc",
    )
    params.hessian_update = _hessian_update(params.hessian_update)


def _validate_corrector_controls(params: object) -> None:
    params.dwi_n = _positive_even_integer(params.dwi_n, "dwi_n")
    params.mbs_max_k = coerce_integer_at_least(params.mbs_max_k, "mbs_max_k", 2)
    params.mbs_points = coerce_integer_at_least(params.mbs_points, "mbs_points", 2)
    params.mbs_tol = coerce_positive_float(params.mbs_tol, "mbs_tol")


def validate_gs_params(params: GSParams) -> None:
    params.max_micro_cycles = coerce_integer_at_least(
        params.max_micro_cycles,
        "max_micro_cycles",
        1,
    )
    params.micro_step_thresh = coerce_positive_float(
        params.micro_step_thresh,
        "micro_step_thresh",
    )
    _validate_hessian_controls(params)


def validate_lqa_params(params: LQAParams) -> None:
    params.euler_n = coerce_integer_at_least(params.euler_n, "euler_n", 1)
    _validate_hessian_controls(params)


def validate_hpc_params(params: HPCParams) -> None:
    params.euler_n = coerce_integer_at_least(params.euler_n, "euler_n", 1)
    _validate_hessian_controls(params)
    _validate_corrector_controls(params)


def validate_eulerpc_params(params: EulerPCParams) -> None:
    params.max_pred_steps = coerce_integer_at_least(
        params.max_pred_steps,
        "max_pred_steps",
        1,
    )
    params.loose_cycles = coerce_integer_at_least(
        params.loose_cycles,
        "loose_cycles",
        0,
    )
    _validate_hessian_controls(params)
    _validate_corrector_controls(params)


IRC_METHOD_VALIDATORS = {
    "gs": validate_gs_params,
    "lqa": validate_lqa_params,
    "hpc": validate_hpc_params,
    "eulerpc": validate_eulerpc_params,
}


def validate_irc_params(params: IRCPathParams, method: str) -> None:
    """Normalize and validate the complete contract for one IRC method."""

    normalized_method = str(method).strip().lower()
    try:
        validator = IRC_METHOD_VALIDATORS[normalized_method]
        expected_type = IRC_METHOD_PARAM_TYPES[normalized_method]
    except KeyError as exc:
        raise ValueError(f"Unknown IRC method: '{method}'.") from exc
    if not isinstance(params, expected_type):
        raise TypeError(
            f"{normalized_method} requires {expected_type.__name__}, "
            f"got {type(params).__name__}."
        )
    validate_irc_preflight_params(params)
    validate_irc_path_params(params)
    validator(params)


__all__ = [
    "DEFAULT_IRC_MAX_FORCE_HARTREE_PER_ANGSTROM",
    "DEFAULT_IRC_MAX_STEPS",
    "DEFAULT_IRC_RMS_FORCE_HARTREE_PER_ANGSTROM",
    "EulerPCParams",
    "GSParams",
    "HPCParams",
    "IRC_METHOD_PARAM_TYPES",
    "IRC_PARAMETER_ALIASES",
    "IRCPathParams",
    "LQAParams",
    "PATH_ENERGY_ABSOLUTE_TOLERANCE_HARTREE",
    "apply_irc_parameter_overrides",
    "coerce_integer_at_least",
    "coerce_nonnegative_float",
    "coerce_positive_float",
    "coerce_strict_bool",
    "irc_parameter_names",
    "irc_params_from_mapping",
    "normalize_irc_parameter_mapping",
    "select_irc_parameter_overrides",
    "validate_eulerpc_params",
    "validate_gs_params",
    "validate_hpc_params",
    "validate_irc_params",
    "validate_irc_path_params",
    "validate_lqa_params",
]
