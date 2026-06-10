"""Shared model-option validation registry for official PBC-capable backends."""

from typing import Callable, Mapping, Optional

from .aimnet.options import AIMNET_LEGACY_MODELS, AIMNET_PBC_MODELS, validate_aimnet_options
from .mace.options import (
    MACE_PBC_MODELS,
    MACEPOL_PBC_MODELS,
    validate_mace_pbc_options,
    validate_macepol_pbc_options,
)

ModelOptionValidator = Callable[[Optional[Mapping[str, object]]], dict[str, object]]
# Routing/path controls are class-protocol concerns owned by SetCalculator, not
# backend scientific options.  Keep them out of strict AIMNet/MACE validators so
# parser canonicalization does not reject documented plugin/model-path flows.
COMMON_MODEL_OPTION_KEYS = frozenset({"module", "model_path"})


def _validate_aimnet_legacy(model_options: Optional[Mapping[str, object]]) -> dict[str, object]:
    return validate_aimnet_options(model_options, pbc=False)


def _validate_aimnet_pbc(model_options: Optional[Mapping[str, object]]) -> dict[str, object]:
    return validate_aimnet_options(model_options, pbc=True)


PBC_OPTION_VALIDATORS: tuple[tuple[frozenset[str], ModelOptionValidator], ...] = (
    (AIMNET_LEGACY_MODELS, _validate_aimnet_legacy),
    (frozenset(AIMNET_PBC_MODELS), _validate_aimnet_pbc),
    (frozenset(MACE_PBC_MODELS), validate_mace_pbc_options),
    (frozenset(MACEPOL_PBC_MODELS), validate_macepol_pbc_options),
)


def validate_model_pbc_options(
    model: str,
    model_options: Optional[Mapping[str, object]],
) -> dict[str, object]:
    model_options = dict(model_options or {})
    common_options = {
        key: value
        for key, value in model_options.items()
        if key in COMMON_MODEL_OPTION_KEYS
    }
    backend_options = {
        key: value
        for key, value in model_options.items()
        if key not in COMMON_MODEL_OPTION_KEYS
    }
    for model_names, validator in PBC_OPTION_VALIDATORS:
        if model in model_names:
            options = validator(backend_options)
            options.update(common_options)
            return options
    return {}
